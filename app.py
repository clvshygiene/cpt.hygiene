import streamlit as st
import pandas as pd
import os
import time
import io
import traceback
import threading
import uuid
import re
import sqlite3
import json
import random
import math  # [新增] 愛校服務 2.0
import concurrent.futures
from contextlib import closing
from datetime import datetime, date, timedelta
from datetime import timezone
import pytz
import gspread
from google.oauth2.service_account import Credentials as SACredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from PIL import Image, ImageOps

try:
    from notion_client import Client
    NOTION_INSTALLED = True
except ImportError:
    NOTION_INSTALLED = False

# --- 1. 網頁設定 ---
# 透過 Streamlit Secrets 判斷目前是測試區還是正式區 (預設為正式區)
sys_env = st.secrets.get("ENV", "PROD")

# [V5.31 Patch 1] 確保 set_page_config 是第一個執行的指令
if sys_env == "DEV":
    st.set_page_config(page_title="🔧測試版-中壢家商，衛愛而生", layout="wide", page_icon="🧹")
    st.sidebar.info(f"🕵️‍♀️ 系統目前抓到的身分證是：[{sys_env}]")
    st.warning("🚧 **目前位於 DEV 測試環境！** 在這裡送出的資料僅供測試，不會影響正式成績。")
else:
    st.set_page_config(page_title="中壢家商，衛愛而生", layout="wide", page_icon="🧹")


# --- 2. 核心參數與全域設定 ---
try:
    TW_TZ = pytz.timezone('Asia/Taipei')
    MAX_IMAGE_BYTES = 20 * 1024 * 1024
    # [Fix #3-B] UPLOAD_SEM 與 IMG_DIR 已移除：照片改為同步上傳 Drive，不再暫存本機
    QUEUE_DB_PATH = "local_status_v5.db"  # 只保留 service_issued dedup + system_status 監控
    
    SHEET_URL = "https://docs.google.com/spreadsheets/d/11BXtN3aevJls6Q2IR_IbT80-9XvhBkjbTCgANmsxqkg/edit"
    SHEET_TABS = {
        "main": "main_data", "settings": "settings", "roster": "roster",
        "inspectors": "inspectors", "duty": "duty",
        "appeals": "appeals", "holidays": "holidays", "service_hours": "service_hours",
        "office_areas": "office_areas", "published_results": "published_results",
        "task_queue": "task_queue_v3",  # [Fix #3-B] 雲端佇列分頁，取代本機 SQLite task_queue 表
        "student_debts": "student_debts", "debt_history": "debt_history"  # [新增] 愛校服務 2.0
    }

    EXPECTED_COLUMNS = [
        "日期", "週次", "班級", "評分項目", "檢查人員",
        "內掃原始分", "外掃原始分", "垃圾原始分", "垃圾內掃原始分", "垃圾外掃原始分", "晨間打掃原始分", "手機人數",
        "備註", "違規細項", "照片路徑", "登錄時間", "修正", "晨掃未到者", "紀錄ID"
    ]
    APPEAL_COLUMNS = ["申訴日期", "班級", "違規日期", "違規項目", "原始扣分", "申訴理由", "佐證照片", "處理狀態", "登錄時間", "對應紀錄ID", "審核回覆"]

    # ==========================================
    # Notion API 輔助函式 
    # ==========================================
    @st.cache_resource
    def get_notion_client():
        if NOTION_INSTALLED:
            token = st.secrets.get("notion_token") or st.secrets.get("system_config", {}).get("notion_token")
            if token: return Client(auth=token)
        return None

def fetch_available_notion_tasks():
        client = get_notion_client()
        db_id = st.secrets.get("notion_db_id") or st.secrets.get("system_config", {}).get("notion_db_id")
        if not client or not db_id: 
            return [], "系統尚未設定 Notion Token 或 Database ID"
        
        try:
            response = client.databases.query(
                database_id=db_id,
                filter={"property": "任務狀態", "status": {"equals": "等待認領中"}} # ⭐️ 改為無表情符號
            )
            tasks = []
            for page in response.get("results", []):
                props = page.get("properties", {})
                title = props.get("任務名稱", {}).get("title", [{}])
                title_text = title[0].get("text", {}).get("content", "未命名任務") if title else "未命名任務"
                
                date_obj = props.get("任務日期", {}).get("date", {})
                raw_date = date_obj.get("start", "未定") if date_obj else "未定"
                if raw_date != "未定":
                    try:
                        parsed_date = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                        if len(raw_date) <= 10:
                            date_val = parsed_date.strftime("%Y-%m-%d")
                        else:
                            date_val = parsed_date.strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        date_val = raw_date
                else:
                    date_val = "未定"
                
                area = props.get("任務內容", {}).get("rich_text", [{}])
                area_text = area[0].get("text", {}).get("content", "未填寫") if area else "未填寫"

                req_num_obj = props.get("需求人數", {}).get("number")
                req_num = req_num_obj if req_num_obj else 1  
                
                claimed_obj = props.get("認領學號", {}).get("rich_text", [])
                claimed_str = claimed_obj[0].get("text", {}).get("content", "") if claimed_obj else ""
                current_claimants = [s.strip() for s in claimed_str.split(",") if s.strip()]
                current_count = len(current_claimants)
                
                tasks.append({
                    "id": page["id"], "title": title_text, "date": date_val, "area": area_text,
                    "req_num": req_num, "current_count": current_count
                })
            return tasks, None
        except Exception as e:
            return [], f"Notion API 讀取失敗詳細錯誤: {str(e)}"

    def claim_notion_task(page_id, student_id, purpose_tag=""):  
        client = get_notion_client()
        if not client:
            return False, "Notion 服務目前未啟用或連線失敗，請通知管理員檢查系統設定。"
            
        try:
            page = client.pages.retrieve(page_id=page_id)
            props = page.get("properties", {})
            
            req_num_obj = props.get("需求人數", {}).get("number")
            req_num = req_num_obj if req_num_obj else 1
            
            claimed_obj = props.get("認領學號", {}).get("rich_text", [])
            claimed_str = claimed_obj[0].get("text", {}).get("content", "") if claimed_obj else ""
            current_claimants = [s.strip() for s in claimed_str.split(",") if s.strip()]
            
            if any(str(student_id) in c for c in current_claimants):  
                return False, f"學號 {student_id} 已經認領過此任務囉！"
                
            claim_label = f"{student_id}({purpose_tag})" if purpose_tag else str(student_id)  
            current_claimants.append(claim_label)
            new_claimed_str = ", ".join(current_claimants)
            
            is_full = len(current_claimants) >= req_num
            update_props = {
                "認領學號": {"rich_text": [{"text": {"content": new_claimed_str}}]}
            }
            if is_full:
                update_props["任務狀態"] = {"status": {"name": "被認領走了"}} # ⭐️ 改為新狀態

            client.pages.update(
                page_id=page_id,
                properties=update_props
            )
            return True, "滿團" if is_full else "未滿"
            
        except Exception as e:
            return False, str(e)

    def fetch_claimed_notion_tasks():
        """抓取 Notion 狀態為「被認領走了」或「任務完成囉」的任務，回傳待驗收列表"""
        client = get_notion_client()
        db_id = st.secrets.get("notion_db_id") or st.secrets.get("system_config", {}).get("notion_db_id")
        if not client or not db_id:
            return []
        try:
            response = client.databases.query(
                database_id=db_id,
                # ⭐️ 這裡使用 OR 邏輯，不管是在進行中還是學生標記完成了，組長都看得到！
                filter={
                    "or": [
                        {"property": "任務狀態", "status": {"equals": "被認領走了"}},
                        {"property": "任務狀態", "status": {"equals": "任務完成囉"}}
                    ]
                }
            )
            tasks = []
            for page in response.get("results", []):
                props = page.get("properties", {})
                title = props.get("任務名稱", {}).get("title", [{}])
                title_text = title[0].get("text", {}).get("content", "未命名任務") if title else "未命名任務"
                date_obj = props.get("任務日期", {}).get("date", {})
                raw_date = date_obj.get("start", "未定") if date_obj else "未定"
                if raw_date != "未定":
                    try:
                        parsed_date = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                        date_val = parsed_date.strftime("%Y-%m-%d")
                    except Exception:
                        date_val = raw_date
                else:
                    date_val = "未定"
                claimed_obj = props.get("認領學號", {}).get("rich_text", [])
                claimed_str = claimed_obj[0].get("text", {}).get("content", "") if claimed_obj else ""
                claimants = [s.strip() for s in claimed_str.split(",") if s.strip()]
                tasks.append({
                    "id": page["id"], "title": title_text,
                    "date": date_val, "claimants": claimants
                })
            return tasks
        except Exception as e:
            print(f"[fetch_claimed_notion_tasks] {e}")
            return []

    def update_notion_task_status(page_id, new_status):
        """將 Notion 任務的「任務狀態」改為指定值"""
        client = get_notion_client()
        if not client:
            return False
        try:
            client.pages.update(
                page_id=page_id,
                properties={"任務狀態": {"status": {"name": new_status}}}
            )
            return True
        except Exception as e:
            print(f"[update_notion_task_status] {e}")
            return False

    # ==========================================
    # SRE Utils: 重試機制
    # ==========================================
    def execute_with_retry(func, max_retries=5, base_delay=1.0, timeout=30):
        for attempt in range(max_retries):
            try:
                time.sleep(0.3 + random.uniform(0, 0.2)) 
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(func)
                    return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                print(f"API Hard Timeout on attempt {attempt+1}")
                if attempt < max_retries - 1:
                    sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    time.sleep(sleep_time)
                else: 
                    raise Exception("API 連線超時，請稍後再試")
            except Exception as e:
                error_str = str(e).lower()
                is_retryable = any(x in error_str for x in ['429', '500', '503', 'quota', 'rate limit', 'timed out', 'connection'])
                if is_retryable and attempt < max_retries - 1:
                    sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    time.sleep(sleep_time)
                else: raise e

    # ==========================================
    # Google 連線與圖片壓縮
    # ==========================================
    @st.cache_resource
    def get_credentials():
        # [Fix #7] 改用 google-auth，取代已停止維護的 oauth2client
        # SACredentials 支援執行緒安全的自動 token 刷新
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        if "gcp_service_account" not in st.secrets:
            return None
        return SACredentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]), scopes=scope
        )

    def get_gspread_client():
        creds = get_credentials()
        # [Fix #7] gspread.Client(auth=creds) 是 google-auth 的正確用法
        return gspread.Client(auth=creds) if creds else None

    def get_drive_service():
        creds = get_credentials()
        return build('drive', 'v3', credentials=creds, cache_discovery=False) if creds else None

    @st.cache_resource
    def get_spreadsheet():
        # [Fix #9] 快取整個 Spreadsheet 物件，避免每次 get_worksheet 都重新 open_by_url
        client = get_gspread_client()
        if not client: return None
        try:
            return client.open_by_url(SHEET_URL)
        except Exception as e:
            print(f"[get_spreadsheet] 無法開啟試算表: {e}")
            return None

    def get_worksheet(tab_name):
        # [Fix #9] 改用快取的 Spreadsheet 物件，只在找不到分頁時才建立新分頁
        sheet = get_spreadsheet()
        if not sheet: return None
        for attempt in range(4):
            try:
                try:
                    return sheet.worksheet(tab_name)
                except gspread.WorksheetNotFound:
                    cols = 20 if tab_name != "appeals" else 15
                    init_rows = 2000 if tab_name == "task_queue" else 500
                    ws = sheet.add_worksheet(title=tab_name, rows=init_rows, cols=cols)
                    if tab_name == "appeals": ws.append_row(APPEAL_COLUMNS)
                    if tab_name == "service_hours": ws.append_row(["日期", "學號", "班級", "類別", "時數", "紀錄ID"])
                    if tab_name == "holidays": ws.append_row(["日期", "說明"])
                    if tab_name == "office_areas": ws.append_row(["區域名稱", "負責班級"])
                    if tab_name == "published_results": ws.append_row(["週次", "排名", "年級", "班級", "總扣分", "優良次數", "總成績", "評等", "排名模式", "發布時間"])
                    if tab_name == "task_queue": ws.append_row(["id", "task_type", "created_ts", "payload_json", "status", "attempts", "last_error"])
                    if tab_name == "student_debts": ws.append_row(["學號", "未完成時數"])  # [新增] 愛校服務 2.0
                    if tab_name == "debt_history": ws.append_row(["時間", "學號", "異動時數", "剩餘時數", "事由"])  # [新增] 愛校服務 2.0
                    return ws
            except Exception as e:
                if "429" in str(e):
                    time.sleep(2 * (attempt + 1) + random.uniform(0, 1))
                    continue
                else: return None
        return None

    def compress_image_bytes(file_bytes, quality=60):
        # [V5.32] 壓縮參數調整：1200px + quality=60，減少 Drive 上傳時間
        # 對手機拍攝的現場照片而言畫質仍足夠辨認違規細節
        try:
            img = Image.open(io.BytesIO(file_bytes))
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB": img = img.convert("RGB")
            if img.width > 1200:
                ratio = 1200 / float(img.width)
                img = img.resize((1200, int(img.height * ratio)), Image.Resampling.LANCZOS)
            out_buffer = io.BytesIO()
            img.save(out_buffer, format="JPEG", quality=quality, optimize=True)
            out_buffer.seek(0)
            return out_buffer
        except Exception as e:
            print(f"[compress_image] {e}")
            return io.BytesIO(file_bytes)

    def upload_image_to_drive(file_obj, filename):
        def _upload_action():
            service = get_drive_service()
            folder_id = st.secrets["system_config"]["drive_folder_id"]
            file = service.files().create(
                body={'name': filename, 'parents': [folder_id]},
                media_body=MediaIoBaseUpload(file_obj, mimetype='image/jpeg', resumable=False), 
                fields='id', supportsAllDrives=True
            ).execute()
            try: service.permissions().create(fileId=file.get('id'), body={'role': 'reader', 'type': 'anyone'}).execute()
            except Exception: pass  # 設公開權限失敗可忽略，不影響上傳
            return f"https://drive.google.com/thumbnail?id={file.get('id')}&sz=w1000"
        return execute_with_retry(_upload_action)

    def clean_id(val):
        # [V5.31 Patch 3] 改為保留字串型態，僅去除 Excel 尾端的 .0，保護學號前導 0
        s = str(val).strip()
        if re.fullmatch(r"\d+\.0", s):
            s = s[:-2]
        return s

    # ==========================================
    # 本機 SQLite（只保留 dedup + 監控狀態，不含 task_queue）
    # ==========================================
    def open_local_db():
        # [Fix #3-B] task_queue 已移至 Google Sheets，這裡只保留兩張輕量表
        conn = sqlite3.connect(QUEUE_DB_PATH, timeout=10.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("CREATE TABLE IF NOT EXISTS service_issued (date TEXT, sid TEXT, category TEXT, PRIMARY KEY(date, sid, category))")
        conn.execute("CREATE TABLE IF NOT EXISTS system_status (key TEXT PRIMARY KEY, val TEXT)")
        return conn

    def update_worker_heartbeat():
        try:
            with closing(open_local_db()) as conn:
                conn.execute("INSERT OR REPLACE INTO system_status VALUES ('worker_heartbeat', ?)", (str(time.time()),))
        except Exception as e: print(f"[heartbeat] {e}")

    def update_last_success_time():
        try:
            with closing(open_local_db()) as conn:
                conn.execute("INSERT OR REPLACE INTO system_status VALUES ('last_success_time', ?)", (str(time.time()),))
        except Exception as e: print(f"[last_success] {e}")

    def get_worker_heartbeat_sec():
        try:
            with closing(open_local_db()) as conn:
                cur = conn.cursor()
                cur.execute("SELECT val FROM system_status WHERE key='worker_heartbeat'")
                row = cur.fetchone()
                if row: return time.time() - float(row[0])
        except Exception as e: print(f"[heartbeat_sec] {e}")
        return 999999

    def get_last_success_sec():
        try:
            with closing(open_local_db()) as conn:
                cur = conn.cursor()
                cur.execute("SELECT val FROM system_status WHERE key='last_success_time'")
                row = cur.fetchone()
                if row: return time.time() - float(row[0])
        except Exception as e: print(f"[last_success_sec] {e}")
        return 999999

    # ==========================================
    # Google Sheets 雲端佇列（取代 SQLite task_queue）
    # ==========================================
    # 欄位索引（1-based）：id=1, task_type=2, created_ts=3, payload_json=4, status=5, attempts=6, last_error=7
    _QCOL_STATUS   = 5
    _QCOL_ATTEMPTS = 6
    _QCOL_ERROR    = 7

    def enqueue_task(task_type, payload):
        # [Fix #3-B] 寫入 Sheets task_queue 分頁，不再用 SQLite
        task_id = str(uuid.uuid4())
        try:
            def _action():
                ws = get_worksheet(SHEET_TABS["task_queue"])
                if not ws: raise Exception("無法取得 task_queue 工作表")
                ws.append_row([
                    task_id, task_type,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(payload, ensure_ascii=False),
                    "PENDING", 0, ""
                ], value_input_option="RAW")
            execute_with_retry(_action)
        except Exception as e:
            print(f"[enqueue] 加入佇列失敗: {e}")
        return task_id

    def get_pending_count():
        try:
            ws = get_worksheet(SHEET_TABS["task_queue"])
            if not ws: return 0
            statuses = ws.col_values(_QCOL_STATUS)[1:]  # 跳過標題列
            return sum(1 for s in statuses if s in ("PENDING", "RETRY"))
        except Exception as e:
            print(f"[pending_count] {e}")
            return 0

    @st.cache_data(ttl=15)  # 15秒快取，避免管理後台每次重繪都打 Sheets API
    def get_queue_metrics():
        metrics = {"pending": 0, "retry": 0, "failed": 0, "oldest_pending_sec": 0, "recent_errors": []}
        try:
            ws = get_worksheet(SHEET_TABS["task_queue"])
            if not ws: return metrics
            records = ws.get_all_records()
            for r in records:
                s = r.get("status", "")
                if s == "PENDING":  metrics["pending"]  += 1
                elif s == "RETRY":  metrics["retry"]    += 1
                elif s == "FAILED": metrics["failed"]   += 1
            pending_recs = [r for r in records if r.get("status") in ("PENDING", "RETRY") and r.get("created_ts")]
            if pending_recs:
                try:
                    oldest = min(r["created_ts"] for r in pending_recs)
                    metrics["oldest_pending_sec"] = (datetime.now(pytz.utc) - datetime.fromisoformat(oldest.replace("Z", "+00:00"))).total_seconds()
                except Exception: pass
            err_recs = sorted([r for r in records if r.get("status") in ("FAILED", "RETRY") and r.get("last_error")],
                              key=lambda x: x.get("created_ts", ""), reverse=True)[:5]
            metrics["recent_errors"] = [(r.get("last_error"), r.get("created_ts")) for r in err_recs]
        except Exception as e: print(f"[queue_metrics] {e}")
        return metrics

    def fetch_next_task(max_attempts=6):
        # [Fix #3-B] 從 Sheets 讀取第一筆 PENDING 任務，標記為 IN_PROGRESS
        # 注意：background_worker 已改用 _extract_next_task(ws, records) 避免重複讀
        # 此函式保留供緊急 fallback 使用
        try:
            ws = get_worksheet(SHEET_TABS["task_queue"])
            if not ws: return None
            records = ws.get_all_records()
            return _extract_next_task(ws, records, max_attempts)
        except Exception as e:
            print(f"fetch_next_task error: {e}")
        return None

    def update_task_status(task_id, status, attempts, last_error, _row_idx=None):
        # DONE → 直接刪行，讓 task_queue 不積累；失敗/重試才保留並更新
        try:
            ws = get_worksheet(SHEET_TABS["task_queue"])
            if not ws: return

            # 定位行號
            if _row_idx:
                ridx = _row_idx
            else:
                ids = ws.col_values(1)[1:]
                if task_id not in ids: return
                ridx = ids.index(task_id) + 2

            if status == "DONE":
                # 成功完成 → 直接刪除這行，task_queue 永遠不會積累歷史紀錄
                try:
                    ws.delete_rows(ridx)
                except Exception as e:
                    print(f"[task_queue] 刪行失敗（忽略）: {e}")
            else:
                # FAILED 或 RETRY → 更新狀態保留供查閱
                ws.batch_update([
                    {"range": f"E{ridx}", "values": [[status]]},
                    {"range": f"F{ridx}", "values": [[attempts]]},
                    {"range": f"G{ridx}", "values": [[str(last_error)[:200] if last_error else ""]]}
                ])
        except Exception as e:
            print(f"update_task_status error: {e}")

    # ==========================================
    # 背景處理邏輯
    # ==========================================

    def fetch_all_pending_service_tasks(max_attempts=6):
        # [Fix #3-B] 從 Sheets 批次撈出所有 service_hours_only PENDING 任務
        # 注意：此函式現在接受外部 ws + records，避免在同一 Worker 輪次重複讀 Sheets
        # 實際呼叫由 background_worker 統一管理（見下方）
        try:
            ws = get_worksheet(SHEET_TABS["task_queue"])
            if not ws: return []
            records = ws.get_all_records()
            return _extract_svc_tasks(ws, records, max_attempts)
        except Exception as e:
            print(f"[batch] 批次抓取 service_hours 任務失敗: {e}")
            return []

    def _extract_svc_tasks(ws, records, max_attempts=6):
        # 從已讀取的 records 中提取 service_hours_only 任務，避免重複讀 Sheets
        result = []
        batch_updates = []
        for i, r in enumerate(records):
            if (r.get("task_type") == "service_hours_only"
                    and r.get("status") in ("PENDING", "RETRY")
                    and int(r.get("attempts", 0)) < max_attempts):
                row_idx = i + 2
                att_new = int(r.get("attempts", 0)) + 1
                result.append({
                    "id":        r["id"],
                    "task_type": r["task_type"],
                    "payload":   json.loads(r.get("payload_json") or "{}"),
                    "attempts":  att_new,
                    "_row_idx":  row_idx
                })
                batch_updates.append({"range": f"E{row_idx}", "values": [["IN_PROGRESS"]]})
                batch_updates.append({"range": f"F{row_idx}", "values": [[att_new]]})
        if batch_updates:
            try:
                ws.batch_update(batch_updates)
            except Exception as e:
                print(f"[batch_update] 標記 IN_PROGRESS 失敗: {e}")
        return result

    def _extract_next_task(ws, records, max_attempts=6):
        # 從已讀取的 records 中取出第一筆非 service_hours_only 的 PENDING 任務
        for i, r in enumerate(records):
            if (r.get("task_type") != "service_hours_only"
                    and r.get("status") in ("PENDING", "RETRY")
                    and int(r.get("attempts", 0)) < max_attempts):
                row_idx = i + 2
                attempts_new = int(r.get("attempts", 0)) + 1
                try:
                    ws.batch_update([
                        {"range": f"E{row_idx}", "values": [["IN_PROGRESS"]]},
                        {"range": f"F{row_idx}", "values": [[attempts_new]]}
                    ])
                except Exception as e:
                    print(f"[fetch_next] 標記 IN_PROGRESS 失敗: {e}")
                    return None
                return {
                    "id":        r["id"],
                    "task_type": r["task_type"],
                    "payload":   json.loads(r.get("payload_json") or "{}"),
                    "attempts":  attempts_new,
                    "_row_idx":  row_idx
                }
        return None

    def process_service_tasks_batch(tasks):
        # [Fix #3] 把多個 service_hours_only 任務的所有學生合併，一次 append_rows 寫入
        is_dry_run = str(st.secrets.get("system_config", {}).get("dry_run", "false")).lower() in ["true", "1"]
        if is_dry_run:
            return True, "DRY_RUN_SUCCESS"
        rows_to_write = []
        for task in tasks:
            payload = task["payload"]
            t_date = payload.get("date", str(date.today()))
            t_cat  = payload.get("category", "")
            for sid in payload.get("student_list", []):
                try:
                    with closing(open_local_db()) as conn:
                        conn.execute(
                            "INSERT INTO service_issued VALUES (?, ?, ?)",
                            (t_date, str(sid), t_cat)
                        )
                    rows_to_write.append([
                        t_date, str(sid),
                        payload.get("class_name", ""), t_cat,
                        str(payload.get("hours", 0.5)),
                        uuid.uuid4().hex[:8]
                    ])
                except sqlite3.IntegrityError:
                    pass  # 已發放過，跳過
        if not rows_to_write:
            return True, None  # 全部都是重複，視為成功
        try:
            def _batch_action():
                ws = get_worksheet(SHEET_TABS["service_hours"])
                if not ws: raise Exception("無法取得 service_hours 工作表")
                ws.append_rows(rows_to_write, value_input_option="RAW")
            execute_with_retry(_batch_action)
            return True, None
        except Exception as e:
            return False, str(e)

    def _append_main_entry_row(entry):
        def _action():
            ws = get_worksheet(SHEET_TABS["main"])
            if not ws: return
            # [防重複寫入] 先比對紀錄ID，若已存在則跳過，避免連續上傳兩筆
            try:
                existing_ids = ws.col_values(EXPECTED_COLUMNS.index("紀錄ID") + 1)
                if str(entry.get("紀錄ID", "")) in existing_ids:
                    print(f"[DEDUP] 紀錄ID {entry.get('紀錄ID')} 已存在，跳過寫入")
                    return
            except Exception as e:
                print(f"[DEDUP] 防重複檢查失敗，繼續寫入: {e}")
            row = [str(entry.get(col, "")).upper() if isinstance(entry.get(col, ""), bool) else str(entry.get(col, "")) for col in EXPECTED_COLUMNS]
            ws.append_row(row)
        execute_with_retry(_action)
    
    def _append_service_row_unique(entry):
        t_date = str(entry.get("日期", ""))
        t_sid = str(entry.get("學號", ""))
        t_cat = str(entry.get("類別", ""))
        
        try:
            with closing(open_local_db()) as conn:
                conn.execute("INSERT INTO service_issued VALUES (?, ?, ?)", (t_date, t_sid, t_cat))
        except sqlite3.IntegrityError:
            return 
            
        def _action():
            ws = get_worksheet(SHEET_TABS["service_hours"])
            if not ws: return
            new_row = [t_date, t_sid, str(entry.get("班級", "")), t_cat, str(entry.get("時數", "")), str(entry.get("紀錄ID", ""))]
            ws.append_row(new_row)
        execute_with_retry(_action)

    def update_last_error_summary(err_msg):
        try:
            with closing(open_local_db()) as conn:
                short_msg = str(err_msg)[:120]
                conn.execute("INSERT OR REPLACE INTO system_status VALUES ('last_error_summary', ?)", (short_msg,))
        except Exception as e: print(f"[error_summary] {e}")

    def get_last_error_summary():
        try:
            with closing(open_local_db()) as conn:
                cur = conn.cursor()
                cur.execute("SELECT val FROM system_status WHERE key='last_error_summary'")
                row = cur.fetchone()
                return row[0] if row else "無紀錄"
        except Exception as e:
            print(f"[error_summary_read] {e}")
            return "無紀錄"

    def process_task(task):
        task_type, payload = task["task_type"], task["payload"]
        
        is_dry_run = str(st.secrets.get("system_config", {}).get("dry_run", "false")).lower() in ["true", "1"]
        if is_dry_run:
            time.sleep(random.uniform(0.3, 0.6))
            return True, "DRY_RUN_SUCCESS"

        # [Fix #3] service_hours_only 現在由 Worker 批次處理（process_service_tasks_batch）
        # 保留此分支只作為 fallback：若任務因故繞過批次路徑，仍可單筆處理
        if task_type == "service_hours_only":
            return process_service_tasks_batch([task])

        entry = payload.get("entry", {})
        # [Fix #3-B] 照片已在 save_entry/save_appeal 同步上傳完畢，
        # entry["照片路徑"] 已包含 Drive 連結，Worker 直接寫 Sheets 即可
        try:
            if task_type in ["main_entry", "volunteer_report"]:
                _append_main_entry_row(entry)
                inspector_name = entry.get("檢查人員", "")
                # [V5.28] 根據參數決定是否發放時數 (預設發放，以防其他地方使用)
                if "學號:" in inspector_name and payload.get("award_inspector_hours", True):
                    sid = inspector_name.split("學號:")[1].strip()
                    _append_service_row_unique({"日期": entry.get("日期"), "學號": sid, "班級": "", "類別": "整潔評分糾察", "時數": 0.25, "紀錄ID": uuid.uuid4().hex[:8]})

                if task_type == "volunteer_report":
                    # [Fix #3] volunteer_report 多名學生的時數改為批次寫入（1 次 append_rows）
                    svc_rows = []
                    t_date = entry.get("日期", str(date.today()))
                    t_cat  = payload.get("custom_category", "晨掃志工")
                    for sid in payload.get("student_list", []):
                        try:
                            with closing(open_local_db()) as conn:
                                conn.execute("INSERT INTO service_issued VALUES (?, ?, ?)", (t_date, str(sid), t_cat))
                            svc_rows.append([
                                t_date, str(sid), entry.get("班級", ""), t_cat,
                                str(payload.get("custom_hours", 0.5)), uuid.uuid4().hex[:8]
                            ])
                        except sqlite3.IntegrityError:
                            pass  # 已發放，跳過
                    if svc_rows:
                        def _svc_batch():
                            ws = get_worksheet(SHEET_TABS["service_hours"])
                            if not ws: raise Exception("無法取得 service_hours 工作表")
                            ws.append_rows(svc_rows, value_input_option="RAW")
                        execute_with_retry(_svc_batch)

            elif task_type == "appeal_entry":
                # [Fix #3-B] 佐證照片已在 save_appeal 同步上傳，entry["佐證照片"] 已設定
                execute_with_retry(lambda: get_worksheet(SHEET_TABS["appeals"]).append_row([str(entry.get(col, "")) for col in APPEAL_COLUMNS]))
            return True, None
        except Exception as e: return False, str(e)

    def background_worker(stop_event=None):
        try: add_script_run_ctx(threading.current_thread(), get_script_run_ctx())
        except Exception: pass  # Streamlit context 在背景執行緒可能不存在，忽略

        # [V5.32] Stuck task recovery：追蹤連續空轉次數，每 60 輪（約 5 分鐘）掃一次卡住任務
        _idle_loops = 0
        STUCK_THRESHOLD_SEC = 300  # IN_PROGRESS 超過 5 分鐘視為卡住

        def _recover_stuck_tasks(ws, records):
            """將超過 5 分鐘仍為 IN_PROGRESS 的任務重置為 RETRY"""
            now_utc = datetime.now(pytz.utc)
            batch_updates = []
            recovered = 0
            for i, r in enumerate(records):
                if r.get("status") != "IN_PROGRESS": continue
                ts_raw = r.get("created_ts", "")
                if not ts_raw: continue
                try:
                    created = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                    if (now_utc - created).total_seconds() > STUCK_THRESHOLD_SEC:
                        row_idx = i + 2
                        attempts_new = int(r.get("attempts", 0))
                        batch_updates.append({"range": f"E{row_idx}", "values": [["RETRY"]]})
                        batch_updates.append({"range": f"F{row_idx}", "values": [[attempts_new]]})
                        batch_updates.append({"range": f"G{row_idx}", "values": [["[AUTO_RECOVERY] IN_PROGRESS timeout"]]})
                        recovered += 1
                except Exception: continue
            if batch_updates:
                try:
                    ws.batch_update(batch_updates)
                    print(f"[recovery] 重置了 {recovered} 個卡住的 IN_PROGRESS 任務")
                except Exception as e:
                    print(f"[recovery] batch_update 失敗: {e}")
            return recovered

        while True:
            if stop_event and stop_event.is_set(): break
            try: update_worker_heartbeat()
            except Exception as e:
                print(f"[worker] heartbeat 寫入失敗: {e}")
            try:
                # [Fix #3-C] 每輪只讀一次 Sheets task_queue，避免 rate limit
                ws = get_worksheet(SHEET_TABS["task_queue"])
                if not ws:
                    time.sleep(5.0)
                    continue

                try:
                    records = ws.get_all_records()
                except Exception as e:
                    err_str = str(e)
                    print(f"[worker] get_all_records 失敗: {e}")
                    # [V5.33] 429 配額耗盡 → 等 60 秒讓配額恢復；其他錯誤等 10 秒
                    if "429" in err_str:
                        time.sleep(60.0)
                    else:
                        time.sleep(10.0)
                    continue

                # [V5.32] 每 60 輪空轉或每次有資料時，檢查並回收卡住的任務
                _idle_loops += 1
                if _idle_loops >= 60:
                    _idle_loops = 0
                    _recover_stuck_tasks(ws, records)
                    # [V5.33] 移除 stuck recovery 後的重複 get_all_records，節省配額
                    # recover 只修改 status 欄，不影響後續的 _extract 邏輯判斷

                # 優先批次清空所有 service_hours_only 任務
                svc_tasks = _extract_svc_tasks(ws, records)
                if svc_tasks:
                    _idle_loops = 0
                    ok, err = process_service_tasks_batch(svc_tasks)
                    if ok: update_last_success_time()
                    else:
                        if err and "DRY_RUN" not in err: update_last_error_summary(err)
                    final_status = "DONE" if ok else "FAILED"
                    for t in svc_tasks:
                        update_task_status(t["id"], final_status, t["attempts"], err, _row_idx=t.get("_row_idx"))
                    time.sleep(2.0)
                    continue

                # 處理含照片的任務（同一批 records，不重複讀）
                task = _extract_next_task(ws, records)
                if not task:
                    # [V5.33] 空閒時改為 20 秒輪詢，大幅降低每日 Sheets read 次數
                    # 原本 5 秒：一天 ~17,280 次；改 20 秒：~4,320 次，降低 75%
                    time.sleep(20.0)
                    continue

                _idle_loops = 0
                ok, err = process_task(task)
                if ok: update_last_success_time()
                else:
                    if err and "DRY_RUN" not in err: update_last_error_summary(err)

                if not ok and err and "FILE_NOT_FOUND" in str(err): task["attempts"] = 999
                update_task_status(task["id"], "DONE" if ok else ("FAILED" if task["attempts"] >= 6 else "RETRY"), task["attempts"], err, _row_idx=task.get("_row_idx"))
                time.sleep(2.0)
            except Exception as e:
                print(f"[worker] 未預期例外: {e}")
                time.sleep(5.0)

    @st.cache_resource
    def ensure_worker_started():
        stop_event = threading.Event()
        t = threading.Thread(target=background_worker, args=(stop_event,), daemon=True)
        try:
            add_script_run_ctx(t)
        except Exception as e:
            print(f"[worker_start] add_script_run_ctx 失敗（忽略）: {e}")
        t.start()
        return stop_event
    _ = ensure_worker_started()

    # ==========================================
    # 前端資料讀取 
    # ==========================================
    @st.cache_data(ttl=21600)
    def load_holidays():
        ws = get_worksheet(SHEET_TABS["holidays"])
        if not ws: return []
        try: return [pd.to_datetime(str(r.get("日期", "")).strip()).date() for r in ws.get_all_records() if str(r.get("日期", "")).strip()]
        except Exception as e:
            print(f"[load_holidays] {e}")
            return []

    def is_within_appeal_period(violation_date, appeal_days=3):
        vd = pd.to_datetime(violation_date).date() if isinstance(violation_date, str) else violation_date
        holidays, today, current_date, workdays = load_holidays(), date.today(), vd, 0
        for _ in range(14): 
            if workdays >= appeal_days: break
            current_date += timedelta(days=1)
            if current_date.weekday() < 5 and current_date not in holidays: workdays += 1
        return today <= current_date

    @st.cache_data(ttl=300)   # [V5.32] 5分鐘快取；尖峰時段觀看者共享同一份，TTL 到期才重讀
    def load_main_data():
        # 讀取整學期 main_data，統一快取一份。
        # 需要近兩週過濾的地方在 UI 層自己用 df[df["週次"] >= now_week-2] 處理。
        ws = get_worksheet(SHEET_TABS["main"])
        if not ws: return pd.DataFrame(columns=EXPECTED_COLUMNS)
        try:
            df = pd.DataFrame(ws.get_all_records())
            if df.empty: return pd.DataFrame(columns=EXPECTED_COLUMNS)
            if "班級" in df.columns: df["班級"] = df["班級"].astype(str).str.strip()
            for col in EXPECTED_COLUMNS:
                if col not in df.columns: df[col] = ""
            if "紀錄ID" not in df.columns: df["紀錄ID"] = df.index.astype(str)
            for col in ["內掃原始分", "外掃原始分", "垃圾原始分", "垃圾內掃原始分", "垃圾外掃原始分", "晨間打掃原始分", "手機人數", "週次"]:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            if "修正" in df.columns: df["修正"] = df["修正"].astype(str).apply(lambda x: True if x.upper() == "TRUE" else False)
            return df[EXPECTED_COLUMNS]
        except Exception as e:
            print(f"[load_main_data] {e}")
            return pd.DataFrame(columns=EXPECTED_COLUMNS)

    @st.cache_data(ttl=21600)
    def load_roster_dict():
        ws = get_worksheet(SHEET_TABS["roster"])
        if not ws: return {}
        try:
            df = pd.DataFrame(ws.get_all_records())
            id_c, cls_c = next((c for c in df.columns if "學號" in c), None), next((c for c in df.columns if "班級" in c), None)
            return {clean_id(row[id_c]): str(row[cls_c]).strip() for _, row in df.iterrows()} if id_c and cls_c else {}
        except Exception as e:
            print(f"[load_roster_dict] {e}")
            return {}
    
    @st.cache_data(ttl=3600)
    def load_sorted_classes():
        ws = get_worksheet(SHEET_TABS["roster"])
        if not ws: return [], []
        try:
            records = ws.get_all_records()
            if not records:
                all_vals = ws.get_all_values()
                if len(all_vals) > 1: records = [dict(zip(all_vals[0], row)) for row in all_vals[1:]]
            df = pd.DataFrame(records)
            class_col = next((c for c in df.columns if "班級" in str(c).strip()), None)
            if not class_col: return [], []
            unique = [c for c in df[class_col].astype(str).str.strip().unique().tolist() if c]
            dept_order = {"商": 1, "英": 2, "資": 3, "家": 4, "服": 5}
            cls_order  = {"甲": 1, "乙": 2, "丙": 3, "丁": 4}
            def get_sort_key(n):
                g   = 1 if "一" in n or "1" in n else (2 if "二" in n or "2" in n else (3 if "三" in n or "3" in n else 99))
                dep = next((v for k, v in dept_order.items() if k in n), 99)
                cls = next((v for k, v in cls_order.items()  if k in n), 99)
                return (g, dep, cls)
            sorted_all = sorted(unique, key=get_sort_key)
            return sorted_all, [{"grade": f"{get_sort_key(c)[0]}年級" if get_sort_key(c)[0]!=99 else "其他", "name": c} for c in sorted_all]
        except Exception as e:
            print(f"[load_sorted_classes] {e}")
            return [], []

    @st.cache_data(ttl=300)   # [效能] 5分鐘，尖峰時段不必每分鐘重打
    def get_daily_duty(target_date):
        ws = get_worksheet(SHEET_TABS["duty"])
        if not ws: return pd.DataFrame(), "error"
        try:
            df = pd.DataFrame(ws.get_all_records())
            if df.empty: return pd.DataFrame(), "no_data"
            date_col = next((c for c in df.columns if "日期" in c), None)
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.date
                return df[df[date_col] == (target_date if isinstance(target_date, date) else target_date.date())], "success"
            return pd.DataFrame(), "missing_cols"
        except Exception as e:
            print(f"[get_daily_duty] {e}")
            return pd.DataFrame(), "error"

    @st.cache_data(ttl=3600)
    def load_office_area_map():
        ws = get_worksheet(SHEET_TABS["office_areas"])
        if not ws: return {}
        try: return {str(r.get("區域名稱", "")).strip(): str(r.get("負責班級", "")).strip() for r in ws.get_all_records() if str(r.get("區域名稱", "")).strip()}
        except Exception as e:
            print(f"[load_office_area_map] {e}")
            return {}

    @st.cache_data(ttl=21600)
    def load_settings():
        ws = get_worksheet(SHEET_TABS["settings"])
        config = {"semester_start": "2025-08-25", "standard_n": 4}
        if ws:
            try:
                for row in ws.get_all_values():
                    if len(row)>=2: config[row[0]] = int(row[1]) if row[0] == "standard_n" else row[1]
            except Exception as e: print(f"[load_settings] {e}")
        return config

    def save_setting(key, val):
        ws = get_worksheet(SHEET_TABS["settings"])
        if ws:
            try:
                cell = ws.find(key)
                if cell: ws.update_cell(cell.row, cell.col+1, val)
                else: ws.append_row([key, val])
                # [V5.32] 只清設定相關快取，不 clear() 所有快取
                # 避免在尖峰時段管理員操作設定時引發 load_main_data 雪崩重讀
                load_settings.clear()
                return True
            except Exception as e:
                print(f"[save_setting] {e}")
                return False
        return False

    @st.cache_data(ttl=300)   # [效能] 5分鐘，申訴資料不需秒級更新
    def load_appeals():
        ws = get_worksheet(SHEET_TABS["appeals"])
        if not ws: return pd.DataFrame(columns=APPEAL_COLUMNS)
        try:
            df = pd.DataFrame(ws.get_all_records())
            for col in APPEAL_COLUMNS:
                if col not in df.columns: df[col] = "待處理" if col == "處理狀態" else ""
            return df[APPEAL_COLUMNS]
        except Exception as e:
            print(f"[load_appeals] {e}")
            return pd.DataFrame(columns=APPEAL_COLUMNS)

    # [新增] 愛校服務 2.0：欠時資料存取函式 =====================
    def load_student_debts():
        """讀取 student_debts 工作表，回傳 {學號: 未完成時數} 字典"""
        ws = get_worksheet(SHEET_TABS["student_debts"])
        if not ws:
            return {}
        try:
            records = ws.get_all_records()
            if not records:
                return {}
            return {clean_id(str(r.get("學號", ""))): float(r.get("未完成時數", 0))
                    for r in records if str(r.get("學號", "")).strip()}
        except Exception as e:
            print(f"[load_student_debts] {e}")
            return {}

    def load_debt_history(sid):
        """讀取 debt_history 工作表，回傳該學號的歷史異動 DataFrame"""
        ws = get_worksheet(SHEET_TABS["debt_history"])
        cols = ["時間", "學號", "異動時數", "剩餘時數", "事由"]
        if not ws:
            return pd.DataFrame(columns=cols)
        try:
            df = pd.DataFrame(ws.get_all_records())
            if df.empty:
                return pd.DataFrame(columns=cols)
            for c in cols:
                if c not in df.columns:
                    df[c] = ""
            return df[df["學號"].astype(str).str.strip() == str(sid).strip()][cols].reset_index(drop=True)
        except Exception as e:
            print(f"[load_debt_history] {e}")
            return pd.DataFrame(columns=cols)

    def update_student_debt(sid, change_hours, reason):
        """寫入 debt_history 一筆紀錄並同步更新 student_debts 中的未完成時數"""
        sid = str(sid).strip()
        try:
            debts = load_student_debts()
            current = debts.get(sid, 0.0)
            new_remaining = round(current + change_hours, 2)
            now_str = datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")
            def _write_history():
                ws_h = get_worksheet(SHEET_TABS["debt_history"])
                if not ws_h:
                    raise Exception("無法取得 debt_history 工作表")
                ws_h.append_row([now_str, sid, change_hours, new_remaining, reason],
                                value_input_option="RAW")
            execute_with_retry(_write_history)
            def _update_debts():
                ws_d = get_worksheet(SHEET_TABS["student_debts"])
                if not ws_d:
                    raise Exception("無法取得 student_debts 工作表")
                all_vals = ws_d.get_all_values()
                found = False
                for i, row in enumerate(all_vals):
                    if i == 0:
                        continue
                    if clean_id(str(row[0]).strip()) == sid:
                        ws_d.update_cell(i + 1, 2, new_remaining)
                        found = True
                        break
                if not found:
                    ws_d.append_row([sid, new_remaining], value_input_option="RAW")
            execute_with_retry(_update_debts)
            return True
        except Exception as e:
            print(f"[update_student_debt] {e}")
            return False

    PUBLISHED_COLS = ["週次", "排名", "年級", "班級", "總扣分", "優良次數", "總成績", "評等", "排名模式", "發布時間"]

    @st.cache_data(ttl=300)   # [效能] 5分鐘快取，發布後學生很快就看得到
    def load_published_results():
        ws = get_worksheet(SHEET_TABS["published_results"])
        if not ws: return pd.DataFrame(columns=PUBLISHED_COLS)
        try:
            df = pd.DataFrame(ws.get_all_records())
            if df.empty: return pd.DataFrame(columns=PUBLISHED_COLS)
            for col in PUBLISHED_COLS:
                if col not in df.columns: df[col] = ""
            for col in ["週次", "排名", "總扣分", "優良次數", "總成績"]:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            return df
        except Exception as e:
            print(f"[load_published_results] {e}")
            return pd.DataFrame(columns=PUBLISHED_COLS)

    def publish_week_results(week_num, fin_ranked_df, rank_mode="全校"):
        """將計算好的週次排名寫入 published_results sheet，同一週再發布會覆蓋舊資料"""
        ws = get_worksheet(SHEET_TABS["published_results"])
        if not ws: return False, "無法連線至 Google Sheets"
        try:
            # 讀取現有資料，刪除同一週次的舊資料
            existing = ws.get_all_values()
            if len(existing) > 1:
                rows_to_delete = [i + 2 for i, row in enumerate(existing[1:])
                                  if row and str(row[0]) == str(week_num)]
                for ridx in sorted(rows_to_delete, reverse=True):
                    ws.delete_rows(ridx)

            # 寫入新資料
            now_str = datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M")
            for _, row in fin_ranked_df.iterrows():
                ws.append_row([
                    int(week_num),
                    int(row.get("排名", 0)),
                    str(row.get("年級", "")),
                    str(row.get("班級", "")),
                    int(row.get("總扣分", 0)),
                    int(row.get("優良次數", 0)),
                    int(row.get("總成績", 0)),
                    str(row.get("評等", "")),
                    rank_mode,
                    now_str
                ])
            load_published_results.clear()
            return True, f"第 {week_num} 週成績已發布！學生現在可以查詢。"
        except Exception as e:
            return False, str(e)

    def save_appeal(entry, proof_file=None):
        # [Fix #3-B] 佐證照片改為同步上傳 Drive，不再寫本機磁碟
        if proof_file:
            try:
                data = proof_file.read()
                if len(data) > MAX_IMAGE_BYTES:
                    st.error("❌ 照片過大（上限 20MB），請壓縮後再上傳")
                    return False
                fname = f"Appeal_{entry.get('班級', '')}_{datetime.now(TW_TZ).strftime('%H%M%S')}.jpg"
                with st.spinner("📤 上傳佐證照片中..."):
                    link = execute_with_retry(
                        lambda d=data, n=fname: upload_image_to_drive(compress_image_bytes(d), n)
                    )
                entry["佐證照片"] = link or ""
            except Exception as e:
                st.warning(f"⚠️ 佐證照片上傳失敗，將不含照片送出申訴：{e}")
                entry["佐證照片"] = ""

        entry.update({
            "申訴日期": entry.get("申訴日期", datetime.now(TW_TZ).strftime("%Y-%m-%d")),
            "處理狀態": entry.get("處理狀態", "待處理"),
            "登錄時間": entry.get("登錄時間", datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")),
            "申訴ID":   entry.get("申訴ID", datetime.now(TW_TZ).strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:4]),
            "佐證照片": entry.get("佐證照片", "")
        })
        enqueue_task("appeal_entry", {"entry": entry})
        st.success("📩 申訴已送出，請耐心等候審核")
        return True
    
    def update_appeal_status(idx, status, record_id, reply_text=""):
        ws_appeals, ws_main = get_worksheet(SHEET_TABS["appeals"]), get_worksheet(SHEET_TABS["main"])
        try:
            data = ws_appeals.get_all_records()
            t_row = next((i + 2 for i, r in enumerate(data) if str(r.get("對應紀錄ID")) == str(record_id) and str(r.get("處理狀態")) == "待處理"), None)
            if t_row:
                ws_appeals.update_cell(t_row, APPEAL_COLUMNS.index("處理狀態") + 1, status)
                if "審核回覆" in APPEAL_COLUMNS:
                    ws_appeals.update_cell(t_row, APPEAL_COLUMNS.index("審核回覆") + 1, reply_text)
                    
                if status == "已核可":
                    m_data = ws_main.get_all_records()
                    m_row = next((j + 2 for j, mr in enumerate(m_data) if str(mr.get("紀錄ID")) == str(record_id)), None)
                    if m_row: ws_main.update_cell(m_row, EXPECTED_COLUMNS.index("修正") + 1, "TRUE")
                load_main_data.clear()
                load_appeals.clear()
                return True, "更新成功"
            return False, "找不到對應的申訴列"
        except Exception as e: return False, str(e)

    def delete_rows_by_ids(ids):
        ws = get_worksheet(SHEET_TABS["main"])
        if not ws: return False
        try:
            rows = sorted([i + 2 for i, r in enumerate(ws.get_all_records()) if str(r.get("紀錄ID")) in ids], reverse=True)
            for r in rows: ws.delete_rows(r)
            time.sleep(0.8); load_main_data.clear()
            return True
        except Exception as e: st.error(f"刪除失敗: {e}"); return False

    @st.cache_data(ttl=21600)
    def load_inspector_list():
        ws = get_worksheet(SHEET_TABS["inspectors"])
        default = [{"label": "測試人員", "allowed_roles": ["內掃檢查"], "assigned_classes": [], "id_prefix": "測", "raw_role": "內掃"}]
        if not ws: return default
        try:
            df = pd.DataFrame(ws.get_all_records())
            if df.empty: return default
            inspectors, id_c, r_c, s_c = [], next((c for c in df.columns if "學號" in c or "編號" in c), None), next((c for c in df.columns if "負責" in c or "項目" in c), None), next((c for c in df.columns if "班級" in c or "範圍" in c), None)
            if id_c:
                for _, row in df.iterrows():
                    sid, s_role = clean_id(row[id_c]), str(row[r_c]).strip() if r_c else ""
                    
                    allowed = []
                    if "組長" in s_role:
                        allowed = ["內掃檢查", "外掃檢查", "垃圾/回收檢查", "晨間打掃"]
                    else:
                        if "外掃" in s_role: allowed.append("外掃檢查")
                        if "垃圾" in s_role or "回收" in s_role: allowed.append("垃圾/回收檢查")
                        if "晨" in s_role: allowed.append("晨間打掃")
                        if "內掃" in s_role: allowed.append("內掃檢查")
                        
                        if "衛生糾察隊長" in s_role or "機動" in s_role:
                            allowed = [r for r in allowed if r != "垃圾/回收檢查"]
                            if not allowed: allowed = ["內掃檢查", "外掃檢查"]
                        elif "環保糾察隊長" in s_role:
                            allowed = [r for r in allowed if r not in ["內掃檢查", "外掃檢查"]]
                            if "垃圾/回收檢查" not in allowed: allowed.append("垃圾/回收檢查")
                            
                        if not allowed: allowed = ["內掃檢查"]

                    s_classes = [c.strip() for c in str(row[s_c]).replace("、", ";").replace(",", ";").split(";") if c.strip()] if s_c and str(row[s_c]) else []
                    
                    inspectors.append({
                        "label": f"學號: {sid}", "allowed_roles": allowed, 
                        "assigned_classes": s_classes, "id_prefix": sid[0] if sid else "X",
                        "raw_role": s_role
                    })
            return inspectors or default
        except Exception as e:
            print(f"[load_inspector_list] {e}")
            return default

    def check_duplicate_record(df, check_date, inspector, role, target_class=None):
        if df.empty: return False
        try:
            # 同時比對原始 role、以及加了(優良)/(普通) 的變體，避免重複評分
            mask = (df["日期"].astype(str) == str(check_date)) & \
                   (df["檢查人員"] == inspector) & \
                   (df["評分項目"].astype(str).str.startswith(role))
            if target_class: mask &= (df["班級"] == target_class)
            return not df[mask].empty
        except Exception as e:
            print(f"[check_duplicate] {e}")
            return False

    # [V5.28] 加入 award_inspector_hours 控制是否發放時數
    # [V5.32] 尖峰時段 jitter 常數（秒）
    # 15:50~16:30 為尖峰時段，42 人同時送出時加入隨機延遲分散 API 壓力
    PEAK_START_H, PEAK_START_M = 15, 50
    PEAK_END_H,   PEAK_END_M   = 16, 30
    PEAK_JITTER_MAX = 8  # 最多延遲 8 秒

    def _is_peak_hour(now_dt):
        """判斷目前是否在尖峰時段"""
        t = now_dt.time()
        from datetime import time as dtime
        return dtime(PEAK_START_H, PEAK_START_M) <= t <= dtime(PEAK_END_H, PEAK_END_M)

    def save_entry(new_entry, uploaded_files=None, student_list=None, custom_hours=0.5, custom_category="晨掃志工", award_inspector_hours=True, skip_jitter=False):
        # [Fix #3-B] 照片改為同步上傳 Drive，不再寫本機磁碟，移除 Semaphore
        new_entry["日期"] = str(new_entry.get("日期", str(date.today())))
        new_entry["紀錄ID"] = new_entry.get("紀錄ID", f"{datetime.now(TW_TZ).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}")
        if "登錄時間" not in new_entry or not new_entry["登錄時間"]:
            new_entry["登錄時間"] = datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")

        # [V5.32] 尖峰時段 jitter：分散 42 人同時提交的 API 請求
        # [V5.34] skip_jitter=True 時略過，避免批次迴圈（垃圾登記）累積成數十秒卡頓
        if not skip_jitter and _is_peak_hour(datetime.now(TW_TZ)):
            jitter = random.uniform(0, PEAK_JITTER_MAX)
            with st.spinner(f"📶 排隊上傳中（{jitter:.1f}s）…"):
                time.sleep(jitter)

        # 同步上傳照片到 Drive（使用 ThreadPoolExecutor 並發，縮短等待時間）
        if uploaded_files:
            valid_files = []
            for i, up_file in enumerate(uploaded_files):
                if not up_file: continue
                try:
                    data = up_file.getvalue()
                    if len(data) > MAX_IMAGE_BYTES:
                        st.warning(f"檔案過大略過: {up_file.name}")
                        continue
                    valid_files.append((i, data, f"{new_entry['紀錄ID']}_{i}.jpg"))
                except Exception as e:
                    print(f"讀取檔案失敗: {e}")

            if valid_files:
                # [V5.32] 每張照片獨立顯示進度
                upload_status_area = st.empty()
                completed_links = []
                failed_count = 0

                def _upload_one_with_progress(args):
                    idx, data, fname = args
                    try:
                        link = execute_with_retry(
                            lambda d=data, n=fname: upload_image_to_drive(compress_image_bytes(d), n)
                        )
                        return link
                    except Exception as e:
                        print(f"Drive 上傳失敗 ({fname}): {e}")
                        return None

                upload_status_area.info(f"📤 準備上傳 {len(valid_files)} 張照片…")
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(valid_files), 3)) as pool:
                    futures = {pool.submit(_upload_one_with_progress, args): i for i, args in enumerate(valid_files)}
                    for done_count, future in enumerate(concurrent.futures.as_completed(futures), 1):
                        result = future.result()
                        if result:
                            completed_links.append(result)
                            upload_status_area.info(f"📤 上傳中… {done_count}/{len(valid_files)} 張完成")
                        else:
                            failed_count += 1
                            upload_status_area.warning(f"⚠️ 第 {done_count} 張上傳失敗，繼續處理其餘照片…")

                if completed_links:
                    new_entry["照片路徑"] = ";".join(completed_links)
                    upload_status_area.success(f"✅ {len(completed_links)} 張照片上傳完成！正在寫入紀錄…")
                else:
                    upload_status_area.empty()

                if failed_count > 0:
                    st.warning(f"⚠️ {failed_count} 張照片上傳失敗，資料已送出但不含這些照片。請截圖後告知組長補傳。")

        payload = {
            "entry": new_entry,
            "student_list": student_list or [],
            "custom_hours": custom_hours,
            "custom_category": custom_category,
            "award_inspector_hours": award_inspector_hours
        }
        enqueue_task("volunteer_report" if student_list is not None else "main_entry", payload)
        return True

    def load_full_semester_data_for_export():
        # 直接重用 load_main_data 的快取，不重複讀 Sheets，記憶體只存一份
        return load_main_data()

    # ==========================================
    # 3. 主程式 UI 啟動前準備
    # ==========================================
    now_tw = datetime.now(TW_TZ)
    today_tw = now_tw.date()
    
    if "last_action_time" not in st.session_state:
        st.session_state.last_action_time = 0
    
    SYSTEM_CONFIG, ROSTER_DICT, INSPECTOR_LIST = load_settings(), load_roster_dict(), load_inspector_list()
    all_classes, structured_classes = load_sorted_classes()
    if not all_classes: all_classes, structured_classes = ["測試班級"], [{"grade": "其他", "name": "測試班級"}]
    grades = sorted(list(set([c["grade"] for c in structured_classes])))
    
    def get_week_num(d):
        try:
            if isinstance(d, datetime): d = d.date()
            # [週次對照表] 優先使用 settings 裡的 week_map 手動對照
            # 格式：2025-01-23:1,2025-02-23:2（只需填「錨點」，後面正常週數自動累計）
            week_map_str = SYSTEM_CONFIG.get("week_map", "")
            if week_map_str.strip():
                entries = []
                for item in week_map_str.split(","):
                    item = item.strip()
                    if ":" in item:
                        date_str, wn = item.rsplit(":", 1)
                        try:
                            entries.append((datetime.strptime(date_str.strip(), "%Y-%m-%d").date(), int(wn.strip())))
                        except Exception: pass
                if entries:
                    entries.sort(key=lambda x: x[0])
                    # 找到最後一個「錨點日期 <= d」的錨點，從那個錨點往後正常累計週數
                    anchor_date, anchor_week = None, 0
                    for start_date, wn in entries:
                        if d >= start_date:
                            anchor_date, anchor_week = start_date, wn
                        else:
                            break
                    if anchor_date is not None:
                        # 從錨點往後每7天加一週
                        return anchor_week + (d - anchor_date).days // 7
                    return 0
            # fallback：純數學計算（未設定 week_map 時使用）
            start = datetime.strptime(SYSTEM_CONFIG["semester_start"], "%Y-%m-%d").date()
            return max(0, ((d - start).days // 7) + 1)
        except Exception as e:
            print(f"[mode3_check] {e}")
            return 0

    # ── 全域樣式注入 ─────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&display=swap');

    /* ── 全域字體 ── */
    html, body, [class*="css"] {
        font-family: 'Noto Sans TC', sans-serif !important;
    }

    /* ── 修正手機上 radio / checkbox / selectbox 文字顏色（僅主內容區，不影響側邊欄）── */
    .main .stRadio label p,
    .main .stRadio label span,
    .main [data-testid="stRadio"] label,
    .main [data-testid="stRadio"] p,
    .main .stCheckbox label p,
    .main .stCheckbox label span,
    .main [data-testid="stCheckbox"] label,
    .main [data-testid="stSelectbox"] span,
    .main [data-testid="stSelectbox"] p,
    .main .stSelectbox label,
    .main .stMarkdown p,
    .main .stText p {
        color: #1a2a4a !important;
    }

    /* ── 主內容區背景 ── */
    .stApp {
        background: #f0f4f9;
    }
    [data-testid="stAppViewContainer"] > .main {
        background: #f0f4f9;
    }

    /* ── 主內容區塊加陰影卡片感 ── */
    [data-testid="stVerticalBlock"] > div > [data-testid="stVerticalBlock"] {
        background: white;
        border-radius: 16px;
        padding: 4px 0;
    }

    /* ── 頁面標題 h1 ── */
    h1 {
        color: #1a2a4a !important;
        font-weight: 700 !important;
        font-size: 26px !important;
        border-left: 5px solid #3182ce;
        padding-left: 12px !important;
        margin-bottom: 20px !important;
    }
    h2 { color: #2c3e6b !important; font-weight: 600 !important; }
    h3 { color: #2c4a8a !important; }

    /* ── 所有按鈕 ── */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 8px 18px !important;
        border: none !important;
        background: linear-gradient(135deg, #2b6cb0, #3182ce) !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(49,130,206,0.35) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1a4a8a, #2b6cb0) !important;
        box-shadow: 0 4px 14px rgba(49,130,206,0.45) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active { transform: translateY(0px) !important; }

    /* ── 下載按鈕特別用綠色 ── */
    [data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #276749, #38a169) !important;
        box-shadow: 0 2px 8px rgba(56,161,105,0.35) !important;
    }
    [data-testid="stDownloadButton"] > button:hover {
        background: linear-gradient(135deg, #1c4f37, #276749) !important;
    }

    /* ── 表單送出按鈕 ── */
    [data-testid="stFormSubmitButton"] > button {
        width: 100% !important;
        background: linear-gradient(135deg, #553c9a, #6b46c1) !important;
        box-shadow: 0 2px 8px rgba(107,70,193,0.35) !important;
        font-size: 16px !important;
        padding: 12px !important;
    }
    [data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(135deg, #44337a, #553c9a) !important;
    }

    /* ── input / selectbox / text_area ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input {
        border-radius: 8px !important;
        border: 1.5px solid #c9d7e8 !important;
        background: #fafcff !important;
        font-size: 14px !important;
        transition: border 0.2s;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #3182ce !important;
        box-shadow: 0 0 0 3px rgba(49,130,206,0.15) !important;
    }

    /* ── selectbox ── */
    [data-testid="stSelectbox"] > div > div {
        border-radius: 8px !important;
        border: 1.5px solid #c9d7e8 !important;
        background: #fafcff !important;
    }

    /* ── info / warning / error / success 提示框 ── */
    [data-testid="stAlert"] {
        border-radius: 10px !important;
        border-left-width: 5px !important;
        font-size: 14px !important;
    }

    /* ── expander ── */
    [data-testid="stExpander"] {
        border: 1.5px solid #d8e4f0 !important;
        border-radius: 12px !important;
        background: white !important;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06) !important;
    }
    [data-testid="stExpander"] summary {
        font-weight: 600 !important;
        font-size: 15px !important;
        color: #1a2a4a !important;
        padding: 12px 16px !important;
    }

    /* ── container border ── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1.5px solid #d8e4f0 !important;
        border-radius: 14px !important;
        background: white !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
        padding: 12px !important;
    }

    /* ── dataframe / table ── */
    [data-testid="stDataFrame"] {
        border-radius: 10px !important;
        overflow: hidden !important;
        box-shadow: 0 1px 8px rgba(0,0,0,0.07) !important;
    }

    /* ── tabs ── */
    [data-testid="stTabs"] [role="tab"] {
        font-weight: 600 !important;
        font-size: 14px !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 8px 16px !important;
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: #2b6cb0 !important;
        border-bottom: 3px solid #3182ce !important;
    }

    /* ── divider ── */
    hr { border-color: #d8e4f0 !important; margin: 16px 0 !important; }

    /* ══════════════════════════════
       側邊欄
    ══════════════════════════════ */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #1a2a4a 0%, #0d1b35 100%) !important;
    }
    [data-testid="stSidebar"] * { color: #e8edf5 !important; }
    [data-testid="stSidebar"] .stRadio > label { display: none; }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
        display: flex; flex-direction: column; gap: 6px; margin-top: 4px;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        display: flex !important; align-items: center;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 10px;
        padding: 10px 14px;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 15px !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.15);
        border-color: rgba(255,255,255,0.3);
        transform: translateX(3px);
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {
        background: rgba(99,179,237,0.25);
        border-color: #63b3ed;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stRadio input[type="radio"] { display: none; }
    [data-testid="stSidebar"] h1 {
        font-size: 20px !important; font-weight: 700 !important;
        color: #ffffff !important;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255,255,255,0.15);
        margin-bottom: 14px !important;
        border-left: none !important;
        padding-left: 0 !important;
    }
    .sidebar-footer {
        position: fixed; bottom: 20px; left: 0;
        width: 240px; text-align: center;
        font-size: 11px; color: rgba(255,255,255,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("🏫 衛愛而生")
    st.sidebar.markdown("<div style='font-size:12px;color:rgba(255,255,255,0.4);margin-top:-12px;margin-bottom:16px;'>中壢家商 衛生組管理系統</div>", unsafe_allow_html=True)

    menu_options = ["糾察底家👀", "班級負責人🥸", "晨掃志工隊🧹", "愛校任務認領 🤝", "組長ㄉ窩💃"]
    app_mode = st.sidebar.radio("請選擇模式", menu_options)

    st.sidebar.markdown("---")
    st.sidebar.markdown("📅 [衛生組行事曆](https://www.notion.so/312b7f229eea80c584a1e794c7b955a4)")
    st.sidebar.markdown("📸 [衛生組 Instagram](https://www.instagram.com/clvs_captain.h/)")
    st.sidebar.markdown("📂 [衛生組公開資料區](https://drive.google.com/drive/folders/14QcUILCmHKnKhDx2X1dIUl_6PNRndCub)")
    st.sidebar.markdown("<div class='sidebar-footer'>中壢家商 衛生組 © 2025</div>", unsafe_allow_html=True)

    # --- Mode: 愛校任務認領 🤝 ---
    if app_mode == "愛校任務認領 🤝":
        st.title("🤝 愛校服務認領區")
        st.info("💡 這裡的任務清單與 Notion 行事曆即時同步！成功認領後，任務會自動標記並更新。")
        
        # [新增] 愛校服務 2.0：欠時查詢
        with st.expander("🔍 查詢我的欠時與明細"):
            _query_sid = st.text_input("請輸入你的學號", placeholder="例如：112001", key="debt_query_sid")
            if st.button("🔎 查詢", key="debt_query_btn"):
                if not _query_sid:
                    st.error("請先輸入學號！")
                else:
                    _debts_map = load_student_debts()
                    _my_hours = _debts_map.get(clean_id(_query_sid), 0.0)
                    if _my_hours > 0:
                        st.warning(f"⚠️ 你目前共有 **{_my_hours}** 小時的欠時尚未歸還。")
                        _hist_df = load_debt_history(clean_id(_query_sid))
                        if not _hist_df.empty:
                            st.dataframe(_hist_df, hide_index=True)
                        else:
                            st.caption("暫無歷史異動紀錄。")
                    else:
                        st.success("🎉 你目前沒有欠時紀錄，繼續保持！")

        n_token = st.secrets.get("notion_token") or st.secrets.get("system_config", {}).get("notion_token")
        if not NOTION_INSTALLED:
            st.error("⚠️ 系統偵測到未安裝 `notion-client` 套件，請通知管理員檢查系統設定。")
        elif not n_token:
            st.warning("⚠️ Notion 金鑰尚未設定，請通知管理員至後台設定 `notion_token`。")
        else:
            with st.spinner("正在向 Notion 獲取最新任務..."):
                tasks, error_msg = fetch_available_notion_tasks()
                
            if error_msg:
                st.error(f"⚠️ 讀取 Notion 發生錯誤！請檢查以下錯誤訊息：\n\n{error_msg}")
            elif not tasks:
                st.success("🎉 娃，目前沒有任務！")
                st.balloons()
            else:
                st.write(f"目前共有 **{len(tasks)}** 項待認領的任務：")
                
                for t in tasks:
                    with st.container(border=True):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.subheader(f"📌 {t['title']} (進度: {t['current_count']} / {t['req_num']} 人)")
                            st.write(f"📅 **執行日期:** {t['date']}")
                            st.write(f"🧹 **任務內容:** {t['area']}")
                        
                        with col2:
                            with st.form(f"claim_form_{t['id']}"):
                                s_id = st.text_input("請輸入您的【學號】來認領：", placeholder="例如：112001", key=f"claim_sid_{t['id']}")
                                # [新增] 愛校服務 2.0：服務目的選擇
                                purpose_choice = st.selectbox("本次愛校服務目的", ["還時數", "消警告", "糾察懲罰"], key=f"claim_purpose_{t['id']}")
                                if st.form_submit_button("🚀 確認認領", use_container_width=True):
                                    if time.time() - st.session_state.last_action_time < 3:
                                        st.warning("⚠️ 系統處理中，請勿連續點擊！")
                                    elif not s_id:
                                        st.error("學號不能為空！")
                                    else:
                                        st.session_state.last_action_time = time.time()
                                        # [新增] 愛校服務 2.0：欠時防呆
                                        _purpose = purpose_choice
                                        _debts_check = load_student_debts()
                                        _my_debt = _debts_check.get(clean_id(s_id), 0.0)
                                        if _my_debt > 0 and _purpose != "還時數":
                                            _purpose = "還時數"
                                            st.warning(f"⚠️ 你目前仍有 {_my_debt} 小時欠時未還，本次目的已自動改為「還時數」！")
                                        with st.spinner("連線至 Notion 更新看板中..."):
                                            success, msg = claim_notion_task(t['id'], s_id, purpose_tag=_purpose)
                                        if success:
                                            if msg == "滿團":
                                                st.success(f"✅ 學號 {s_id} 認領成功！此任務已額滿，自動從看板隱藏。")
                                            else:
                                                st.success(f"✅ 學號 {s_id} 認領成功！目前還缺人，趕緊揪同學來認領！")
                                            time.sleep(2)
                                            st.rerun()
                                        else:
                                            st.error(f"認領失敗：{msg}")

    # --- Mode 1: 糾察評分 ---
    elif app_mode == "糾察底家👀":
        st.title("📝 衛生糾察評分系統")
        if "team_logged_in" not in st.session_state: st.session_state["team_logged_in"] = False

        daily_hygiene = SYSTEM_CONFIG.get("daily_hygiene_task", "")
        if daily_hygiene:
            formatted_hygiene = daily_hygiene.replace('\n', '<br>')
            mascot_url = "https://drive.google.com/thumbnail?id=128ITPXtpGNuI-wLIt6p-qd4ZNNhCGbhd" 
            
            bubble_html_h = f"""
            <style>
            .mascot-container-h {{ display: flex; align-items: flex-start; margin-bottom: 20px; gap: 15px; }}
            .mascot-img-h {{ width: 200px; flex-shrink: 0; }}
            .speech-bubble-h {{
                position: relative; background: #D0E8F2;
                border-radius: 15px; padding: 15px 20px; color: #05445E; font-size: 16px;
                box-shadow: 2px 4px 10px rgba(0,0,0,0.1); border: 2px solid #189AB4; flex-grow: 1;
            }}
            .speech-bubble-h::before {{ content: ''; position: absolute; left: -20px; top: 30px; border: 10px solid transparent; border-right-color: #189AB4; }}
            .speech-bubble-h::after {{ content: ''; position: absolute; left: -16px; top: 30px; border: 10px solid transparent; border-right-color: #D0E8F2; }}
            @media (max-width: 500px) {{
                .mascot-img-h {{ width: 120px; }}
                .speech-bubble-h {{ font-size: 14px; padding: 10px 15px; }}
            }}
            </style>
            <div class="mascot-container-h">
                <img src="{mascot_url}" class="mascot-img-h" />
                <div class="speech-bubble-h">
                    <strong>📢 組長廣播 / 糾察重點：</strong><br>
                    {formatted_hygiene}
                </div>
            </div>
            """
            st.markdown(bubble_html_h, unsafe_allow_html=True)
        
        if not st.session_state["team_logged_in"]:
            with st.expander("🔐 身份驗證", expanded=True):
                pwd_input = st.text_input("請輸入隊伍通行碼", type="password", key="m1_login_pwd")
                if pwd_input:
                    if pwd_input == st.secrets["system_config"]["team_password"]:
                        st.session_state["team_logged_in"] = True
                        st.rerun()
                    else:
                        st.error("通行碼錯誤")
        
        if st.session_state["team_logged_in"]:
            prefixes = sorted(list(set([p["id_prefix"] for p in INSPECTOR_LIST])))
            if not prefixes: st.warning("找不到糾察名單")
            else:
                sel_p = st.radio("步驟 1：選擇開頭", [f"{p}開頭" for p in prefixes], horizontal=True, key="m1_p_radio")[0]
                inspector_name = st.radio("步驟 2：點選身份", [p["label"] for p in INSPECTOR_LIST if p["id_prefix"] == sel_p], key="m1_name_radio")
                curr_inspector = next((p for p in INSPECTOR_LIST if p["label"] == inspector_name), {})
                allowed_roles = [r for r in curr_inspector.get("allowed_roles", ["內掃檢查"]) if r != "晨間打掃"] or ["內掃檢查"]
                
                st.markdown("---")
                c_d, c_r = st.columns(2)
                input_date = c_d.date_input("檢查日期", today_tw)
                role = c_r.radio("檢查項目", allowed_roles, horizontal=True, key="m1_role_radio") if len(allowed_roles)>1 else allowed_roles[0]
                week_num = get_week_num(input_date)
                main_df = load_main_data()
                # 糾察評分只需近兩週資料（查重複用），在 UI 層過濾
                now_week = get_week_num(today_tw)
                if now_week >= 3:
                    main_df = main_df[main_df["週次"] >= now_week - 2]

                if role == "垃圾/回收檢查":
                    st.info("🗑️ 資源回收與垃圾檢查 (每日每班此項目總扣分上限2分將於結算時自動卡控)")
                    
                    step_a = st.radio("步驟 A: 選擇垃圾類別", ["一般垃圾", "紙類", "網袋aka塑膠鐵鋁", "其他"], horizontal=True, key="m1_trash_a")
                    sel_filter = st.radio("步驟 B: 篩選檢查對象", ["各處室 (外掃)"] + grades, horizontal=True, key="m1_trash_b")
                    
                    today_records = main_df[(main_df["日期"].astype(str) == str(input_date)) & (main_df["評分項目"] == "垃圾/回收檢查") & (main_df["違規細項"] == step_a)] if not main_df.empty else pd.DataFrame()
                    rows = []
                    
                    if sel_filter == "各處室 (外掃)":
                        office_map = load_office_area_map()
                        target_list = list(office_map.keys()) or ["教務處", "學務處", "總務處", "輔導室", "圖書館"]
                        for off_name in target_list:
                            cls_name = office_map.get(off_name, "未設定")
                            
                            is_dump_bad = any(f"外掃({off_name})" in str(r["備註"]) and "未倒垃圾" in str(r["備註"]) for _, r in today_records.iterrows()) if not today_records.empty else False
                            is_sort_bad = any(f"外掃({off_name})" in str(r["備註"]) and "未做好分類" in str(r["備註"]) for _, r in today_records.iterrows()) if not today_records.empty else False
                            
                            row_data = {"處室/區域": off_name, "負責班級": cls_name, "未倒垃圾": is_dump_bad, "未做好分類": is_sort_bad}
                            rows.append(row_data)
                            
                        col_config = {"處室/區域": st.column_config.TextColumn(disabled=True), "負責班級": st.column_config.TextColumn(disabled=True)}
                        col_config["未倒垃圾"] = st.column_config.CheckboxColumn("🗑️ 未倒垃圾", help="扣1分")
                        col_config["未做好分類"] = st.column_config.CheckboxColumn("♻️ 未做好分類", help="扣1分")
                        
                        edited_df = st.data_editor(pd.DataFrame(rows), column_config=col_config, hide_index=True, width="stretch", key="ed_offices")
                        
                        if st.button(f"💾 登記違規 ({step_a} - 各處室)"):
                            if time.time() - st.session_state.last_action_time < 3:
                                st.warning("⚠️ 系統處理中，請勿連續點擊！")
                            else:
                                st.session_state.last_action_time = time.time()
                                cnt = 0
                                for _, row in edited_df.iterrows():
                                    off, cls = row["處室/區域"], row["負責班級"]
                                    b_sort = row.get("未做好分類", False)
                                    b_dump = row.get("未倒垃圾", False)
                                    
                                    orig = next((x for x in rows if x["處室/區域"] == off), None)
                                    v_list = []
                                    if b_dump and not orig.get("未倒垃圾", False): v_list.append("未倒垃圾")
                                    if b_sort and not orig.get("未做好分類", False): v_list.append("未做好分類")
                                    
                                    if v_list:
                                        score = len(v_list)
                                        base = {"日期": input_date, "週次": week_num, "檢查人員": inspector_name, "登錄時間": now_tw.strftime("%Y-%m-%d %H:%M:%S"), "班級": cls, "評分項目": role, "垃圾內掃原始分": 0, "垃圾外掃原始分": score}
                                        if save_entry({**base, "備註": f"外掃({off})-{step_a}({','.join(v_list)})", "違規細項": step_a}, skip_jitter=True):
                                            cnt += 1
                                if cnt: st.success(f"✅ 已登記 {cnt} 筆違規！"); time.sleep(1.5); st.rerun()

                    else:
                        for cls_name in [c["name"] for c in structured_classes if c["grade"] == sel_filter]:
                            cls_rec = today_records[today_records["班級"] == cls_name] if not today_records.empty else pd.DataFrame()
                            
                            is_dump_bad = any("內掃" in str(r["備註"]) and "未倒垃圾" in str(r["備註"]) for _, r in cls_rec.iterrows()) if not cls_rec.empty else False
                            is_sort_bad = any("內掃" in str(r["備註"]) and "未做好分類" in str(r["備註"]) for _, r in cls_rec.iterrows()) if not cls_rec.empty else False
                            
                            row_data = {"班級": cls_name, "未倒垃圾": is_dump_bad, "未做好分類": is_sort_bad}
                            rows.append(row_data)
                            
                        col_config = {"班級": st.column_config.TextColumn(disabled=True)}
                        col_config["未倒垃圾"] = st.column_config.CheckboxColumn("🗑️ 未倒垃圾", help="扣1分")
                        col_config["未做好分類"] = st.column_config.CheckboxColumn("♻️ 未做好分類", help="扣1分")
                            
                        edited_df = st.data_editor(pd.DataFrame(rows), column_config=col_config, hide_index=True, width="stretch", key=f"ed_{sel_filter}")
                        
                        if st.button(f"💾 登記違規 ({step_a} - {sel_filter})"):
                            if time.time() - st.session_state.last_action_time < 3:
                                st.warning("⚠️ 系統處理中，請勿連續點擊！")
                            else:
                                st.session_state.last_action_time = time.time()
                                cnt = 0
                                for _, row in edited_df.iterrows():
                                    cls = row["班級"]
                                    b_sort = row.get("未做好分類", False)
                                    b_dump = row.get("未倒垃圾", False)
                                    
                                    orig = next((x for x in rows if x["班級"] == cls), None)
                                    v_list = []
                                    if b_dump and not orig.get("未倒垃圾", False): v_list.append("未倒垃圾")
                                    if b_sort and not orig.get("未做好分類", False): v_list.append("未做好分類")
                                    
                                    if v_list:
                                        score = len(v_list)
                                        base = {"日期": input_date, "週次": week_num, "檢查人員": inspector_name, "登錄時間": now_tw.strftime("%Y-%m-%d %H:%M:%S"), "班級": cls, "評分項目": role, "垃圾內掃原始分": score, "垃圾外掃原始分": 0}
                                        if save_entry({**base, "備註": f"內掃-{step_a}({','.join(v_list)})", "違規細項": step_a}, skip_jitter=True):
                                            cnt += 1
                                if cnt: st.success(f"✅ 已登記 {cnt} 筆違規！"); time.sleep(1.5); st.rerun()

                else:
                    assigned_classes = curr_inspector.get("assigned_classes", [])
                    is_last_task = True
                    pending_classes = []

                    # [初始化] 確保所有糾察（含機動/組長）都能使用此 key
                    if "submitted_inspections" not in st.session_state:
                        st.session_state.submitted_inspections = set()

                    if assigned_classes:

                        # 從 Google Sheet 讀到的已完成班級（字串，直接是班級名稱）
                        sheet_done = set(
                            main_df[
                                (main_df["日期"].astype(str) == str(input_date)) &
                                (main_df["檢查人員"] == inspector_name)
                            ]["班級"].astype(str).tolist()
                        )
                        # 從 session_state 讀到的本地已送出班級（格式：日期__糾察名__班級）
                        local_done = set(
                            key.split("__")[2]
                            for key in st.session_state.submitted_inspections
                            if key.startswith(f"{input_date}__{inspector_name}__")
                        )
                        completed_class_names = sheet_done | local_done
                        pending_classes = [c for c in assigned_classes if c not in completed_class_names]

                        # 合併「任務類型 + 進度」為同一個框
                        role_label = "🏫 內掃檢查" if role == "內掃檢查" else "🏢 外掃檢查"
                        progress_text = f"今日任務：{role_label}　|　進度：{len(completed_class_names)}/{len(assigned_classes)}"
                        if pending_classes:
                            progress_text += f"　|　尚缺：{', '.join(pending_classes)}"
                        else:
                            progress_text += "　|　✅ 今日任務全數完成！"
                        st.info(progress_text)

                        sel_cls = st.radio("選擇負責班級", assigned_classes, key="m1_cls_assigned")

                        # 判斷這是不是最後一個缺少的班級
                        if sel_cls in pending_classes and len(pending_classes) == 1:
                            is_last_task = True
                        elif sel_cls in pending_classes:
                            is_last_task = False
                        else:
                            is_last_task = False
                    else:
                        st.info("📍 今日任務：機動/隊長/組長自由巡查")
                        temp_g = st.radio("步驟 A: 選擇年級", grades, horizontal=True, key="m1_grade_select")
                        f_cls_list = [c["name"] for c in structured_classes if c["grade"] == temp_g]
                        sel_cls = st.radio("步驟 B: 選擇班級", f_cls_list, horizontal=True, key="m1_cls_select") if f_cls_list else None

                    if sel_cls:
                        st.divider()
                        # [Fix #6] 雙重防呆：先查 session_state（即時），再查快取 df（補強）
                        _submit_key_check = f"{input_date}__{inspector_name}__{sel_cls}"
                        _already_in_session = _submit_key_check in st.session_state.submitted_inspections
                        _already_in_sheet   = check_duplicate_record(main_df, input_date, inspector_name, role, sel_cls)
                        if _already_in_session or _already_in_sheet:
                            st.warning(f"⚠️ 今日已評過 {sel_cls}！")

                        # [關鍵] check_result 放在 form 外面，才能即時顯示/隱藏違規欄位
                        check_result = st.radio("檢查結果", ["⭐ 優良", "✅ 普通", "❌ 違規(需扣分)"], horizontal=True, key="m1_check_result")

                        # 違規細節區塊（在 form 外，隨 radio 即時顯示）
                        in_s, out_s, ph_c, note, sel_violations = 0, 0, 0, "", []

                        if check_result == "⭐ 優良":
                            # [需求2] 優良備註欄 - 放在 form 外顯示說明
                            st.caption("⏳ 優良紀錄將送交組長審核，審核通過前學生端顯示為「普通」。")

                        elif check_result == "✅ 普通":
                            st.caption("✅ 無扣分，表現普通。如有需要可在下方填寫備註（選填）。")

                        elif check_result == "❌ 違規(需扣分)":
                            if role == "內掃檢查":
                                in_s = st.number_input("內掃扣分", min_value=0, step=1, key="m1_in_s")
                                st.markdown("**📍 違規位置（可複選）**")
                                INNER_AREA_OPTIONS = ["走廊", "黑板", "地板", "窗戶(窗溝)", "陽台"]
                                sel_areas = st.multiselect("違規位置", INNER_AREA_OPTIONS, key="inner_areas")
                                st.markdown("**⚠️ 違規狀況（可複選）**")
                                INNER_STATUS_OPTIONS = ["髒亂", "沒拖地", "沒擦拭", "酒精未補", "掃具壞掉未換", "懸掛垃圾未清除", "人工垃圾", "蜘蛛網", "頭髮圈圈", "打掃玩手機"]
                                sel_violations = st.multiselect("違規狀況", INNER_STATUS_OPTIONS, key="inner_status")
                                extra_note = st.text_input("📝 其他補充（找不到對應選項時請在此輸入）", key="m1_extra_note")
                                note_parts = []
                                if sel_areas: note_parts.append("位置：" + "、".join(sel_areas))
                                if sel_violations: note_parts.append("狀況：" + "、".join(sel_violations))
                                if extra_note: note_parts.append(extra_note)
                                note = " | ".join(note_parts)

                            elif role == "外掃檢查":
                                out_s = st.number_input("外掃扣分", min_value=0, step=1, key="m1_out_s")
                                st.markdown("**📍 違規位置**")
                                loc_col1, loc_col2 = st.columns(2)
                                BUILDING_OPTIONS = ["", "誠信樓A棟(各處室)", "誠信樓B棟", "樸實樓(合作社)", "勤學樓(烘焙縫紉)", "敬業樓(圖書館)"]
                                FLOOR_MAP = {
                                    "誠信樓A棟(各處室)": ["", "1F", "2F", "3F", "4F", "5F", "6F"],
                                    "誠信樓B棟":         ["", "1F", "2F", "3F", "4F", "5F", "6F"],
                                    "樸實樓(合作社)":    ["", "1F", "2F", "3F", "4F", "5F"],
                                    "勤學樓(烘焙縫紉)": ["", "1F", "2F", "3F", "4F", "5F"],
                                    "敬業樓(圖書館)":   ["", "1F", "2F", "3F"],
                                }
                                sel_building = loc_col1.selectbox("大樓", BUILDING_OPTIONS, key="m1_building")
                                floor_opts = FLOOR_MAP.get(sel_building, ["", "1F", "2F", "3F", "4F", "5F", "6F"])
                                sel_floor = loc_col2.selectbox("樓層", floor_opts, key="m1_floor")
                                st.markdown("**⚠️ 違規項目（可複選）**")
                                OUTER_AREA_OPTIONS = ["男廁", "女廁", "茶水間", "無障礙廁所", "樓梯間", "洗手台", "天花板", "走廊", "地板", "陽台"]
                                sel_outer_areas = st.multiselect("違規項目", OUTER_AREA_OPTIONS, key="outer_areas")
                                st.markdown("**⚠️ 違規狀況（可複選）**")
                                OUTER_STATUS_OPTIONS = ["髒亂", "沒拖地", "沒掃地", "沒擦拭", "酒精未補", "掃具壞掉未換", "人工垃圾", "蜘蛛網", "頭髮圈圈", "打掃玩手機"]
                                sel_status = st.multiselect("違規狀況", OUTER_STATUS_OPTIONS, key="outer_status")
                                extra_note = st.text_input("📝 其他補充（找不到對應選項時請在此輸入）", key="m1_extra_note")
                                note_parts = []
                                if sel_building: note_parts.append(sel_building)
                                if sel_floor: note_parts.append(sel_floor)
                                if sel_outer_areas: note_parts.append("項目：" + "、".join(sel_outer_areas))
                                if sel_status: note_parts.append("狀況：" + "、".join(sel_status))
                                if extra_note: note_parts.append(extra_note)
                                note = " | ".join(note_parts)
                                sel_violations = sel_outer_areas + sel_status

                        # form 只負責：修正單勾選 + 照片上傳 + 送出按鈕
                        excellent_note_form = ""  # 預設值，避免未定義
                        normal_note_form = ""     # 普通備註預設值
                        with st.form("score_form", clear_on_submit=True):
                            is_fix = st.checkbox("🚩 這是修正單")
                            # 優良備註在 form 內，避免 session_state key 問題
                            if check_result == "⭐ 優良":
                                st.markdown("**📝 優良原因（選填）**")
                                excellent_note_form = st.text_input("請簡單描述優良原因，例如：地板乾淨、掃具整齊", key="m1_excellent_note_form")
                            # 普通備註在 form 內
                            elif check_result == "✅ 普通":
                                st.markdown("**📝 備註（選填）**")
                                normal_note_form = st.text_input("可填寫備註，例如：掃得不夠乾淨但未達扣分標準，下次請加強", key="m1_normal_note_form")
                            # [照片強制上傳] 三種結果都需附照，防止隨意亂評
                            st.info("📸 請先用手機相機拍好照片存到相簿，再從下方選取上傳。（優良/普通/違規均需附照）")
                            files = st.file_uploader("選取照片", accept_multiple_files=True, type=['jpg','png','jpeg'])

                            if st.form_submit_button("送出"):
                                if time.time() - st.session_state.last_action_time < 5:
                                    st.warning("⚠️ 系統處理中，請稍候 5 秒再試！")
                                elif not files:
                                    st.error("❌ 請先上傳現場照片才能送出！（優良/普通也需要附照）")
                                else:
                                    st.session_state.last_action_time = time.time()
                                    _submit_key = f"{input_date}__{inspector_name}__{sel_cls}"
                                    if check_result == "⭐ 優良":
                                        # [需求3] 存為「待審優良」，組長審核後才正式升為優良
                                        # 從 form 內的變數讀取（避免 session_state key error）
                                        try:
                                            _exc_note = excellent_note_form.strip() if excellent_note_form else ""
                                        except Exception:
                                            _exc_note = ""
                                        _note_text = f"優良原因：{_exc_note}" if _exc_note else "本次檢查表現優良，無扣分項目（待組長審核）"
                                        if save_entry({"日期": input_date, "週次": week_num, "檢查人員": inspector_name, "登錄時間": now_tw.strftime("%Y-%m-%d %H:%M:%S"), "修正": is_fix, "班級": sel_cls, "評分項目": role + "(待審優良)", "內掃原始分": 0, "外掃原始分": 0, "垃圾原始分": 0, "垃圾內掃原始分": 0, "垃圾外掃原始分": 0, "手機人數": 0, "備註": _note_text}, uploaded_files=files, award_inspector_hours=is_last_task):
                                            st.session_state.submitted_inspections.add(_submit_key)
                                            st.success("⭐ 優良紀錄已送出！等待組長審核中..."); time.sleep(1.5); st.rerun()
                                    elif check_result == "✅ 普通":
                                        try:
                                            _norm_note = normal_note_form.strip() if normal_note_form else ""
                                        except Exception:
                                            _norm_note = ""
                                        _norm_note_text = _norm_note if _norm_note else "本次檢查無扣分，表現普通"
                                        if save_entry({"日期": input_date, "週次": week_num, "檢查人員": inspector_name, "登錄時間": now_tw.strftime("%Y-%m-%d %H:%M:%S"), "修正": is_fix, "班級": sel_cls, "評分項目": role + "(普通)", "內掃原始分": 0, "外掃原始分": 0, "垃圾原始分": 0, "垃圾內掃原始分": 0, "垃圾外掃原始分": 0, "手機人數": 0, "備註": _norm_note_text}, uploaded_files=files, award_inspector_hours=is_last_task):
                                            st.session_state.submitted_inspections.add(_submit_key)
                                            st.success("✅ 普通紀錄已登記！"); time.sleep(1.5); st.rerun()
                                    elif check_result == "❌ 違規(需扣分)":
                                        # [需求1] 允許扣0分：備註自動加【警告】標記，計分面板顯示紅色但不計入成績
                                        _deduct_note = (f"【警告，扣0分】{note}".strip()) if (in_s + out_s) == 0 else note
                                        if save_entry({"日期": input_date, "週次": week_num, "檢查人員": inspector_name, "登錄時間": now_tw.strftime("%Y-%m-%d %H:%M:%S"), "修正": is_fix, "班級": sel_cls, "評分項目": role, "內掃原始分": in_s, "外掃原始分": out_s, "手機人數": ph_c, "備註": _deduct_note, "違規細項": "、".join(sel_violations) if sel_violations else ""}, uploaded_files=files, award_inspector_hours=is_last_task):
                                                st.session_state.submitted_inspections.add(_submit_key)
                                                if assigned_classes:
                                                    if is_last_task:
                                                        st.success("✅ 送出成功！今日任務已全數完成，系統將自動核發 0.25 小時！")
                                                    else:
                                                        st.success(f"✅ 送出成功！尚缺 {len(pending_classes)-1} 個班級，請繼續努力！")
                                                else:
                                                    st.success("✅ 送出成功！系統將自動排程發放本日 0.25 小時。")
                                                time.sleep(1.5)
                                                st.rerun()

    # --- Mode 2: 班級負責人 ---
    elif app_mode == "班級負責人🥸":
        st.title("🔎 班級成績查詢")
        df, appeals_df = load_full_semester_data_for_export(), load_appeals()  # 班級負責人需要整學期資料才能正確顯示學期累計扣分
        appeal_map = {str(r.get("對應紀錄ID")): {"status": str(r.get("處理狀態", "")), "reply": str(r.get("審核回覆", ""))} for _, r in appeals_df.iterrows()} if not appeals_df.empty else {}

        sel_grade_m2 = st.radio("選擇年級", grades, horizontal=True, key="m2_grade_select")
        cls_opts = [c["name"] for c in structured_classes if c["grade"] == sel_grade_m2]

        if cls_opts:
            cls = st.selectbox("選擇班級", cls_opts, key="m2_cls_select")
            if cls and not df.empty:
                cls_df = df[df["班級"] == cls].copy()

                # ── [效能] 預計算申訴期限，不在每筆 loop 裡重複呼叫 ──
                holidays = load_holidays()
                def _appeal_deadline(vd):
                    try:
                        current_date, workdays = (pd.to_datetime(str(vd)).date() if isinstance(vd, str) else vd), 0
                        for _ in range(14):
                            if workdays >= 3: break
                            current_date += timedelta(days=1)
                            if current_date.weekday() < 5 and current_date not in holidays: workdays += 1
                        return current_date
                    except Exception: return date.today()

                unique_dates = cls_df["日期"].astype(str).unique()
                appeal_deadline_map = {d: _appeal_deadline(d) for d in unique_dates}

                # ── 摘要卡片 ──
                now_week = get_week_num(today_tw)
                week_df = cls_df[cls_df["週次"] == now_week] if "週次" in cls_df.columns else pd.DataFrame()

                def _calc_tot(r):
                    tr = r['垃圾內掃原始分'] + r['垃圾外掃原始分']
                    if tr == 0: tr = r['垃圾原始分']
                    return r['內掃原始分'] + r['外掃原始分'] + tr + r['晨間打掃原始分']

                def _is_real_deduct(r):
                    item = str(r['評分項目'])
                    # 排除優良、普通、學期加分（晨掃），以及申訴已核可的修正紀錄
                    # 注意：load_main_data 將修正欄位轉為布林值 True/False
                    is_corrected = r['修正'] is True or str(r['修正']).upper() == "TRUE"
                    return (not any(x in item for x in ["優良", "普通", "學期加分"])
                            and not is_corrected)

                def _is_bonus(r):
                    # 學期加分紀錄（晨間打掃(學期加分)，分數為負）
                    return "學期加分" in str(r['評分項目']) and _calc_tot(r) < 0

                # 本週：只算正數扣分
                week_deduct = sum(max(_calc_tot(r), 0) for _, r in week_df.iterrows()
                                  if _is_real_deduct(r)) if not week_df.empty else 0
                # 學期：扣分與加分分開
                total_deduct = max(sum(max(_calc_tot(r), 0) for _, r in cls_df.iterrows() if _is_real_deduct(r)), 0)
                total_bonus  = sum(abs(_calc_tot(r)) for _, r in cls_df.iterrows() if _is_bonus(r))

                pending_appeals = sum(1 for rid in [str(r['紀錄ID']) for _, r in cls_df.iterrows()]
                                      if appeal_map.get(rid, {}).get("status") == "待處理")

                mc1, mc2, mc3 = st.columns(3)
                mc1.markdown(
                    f"<div style='background:#f0f7ff;border-radius:12px;padding:14px 16px;border-left:4px solid #3182ce'>"
                    f"<div style='font-size:12px;color:#555;margin-bottom:4px'>本週扣分</div>"
                    f"<div style='font-size:26px;font-weight:700;color:{'#e53e3e' if week_deduct>0 else '#38a169'}'>{week_deduct} 分</div>"
                    f"</div>", unsafe_allow_html=True)
                mc2.markdown(
                    f"<div style='background:#f0f7ff;border-radius:12px;padding:14px 16px;border-left:4px solid #805ad5'>"
                    f"<div style='font-size:12px;color:#555;margin-bottom:4px'>待處理申訴</div>"
                    f"<div style='font-size:26px;font-weight:700;color:{'#d69e2e' if pending_appeals>0 else '#38a169'}'>{pending_appeals} 件</div>"
                    f"</div>", unsafe_allow_html=True)
                # 第三張卡片：有加分時同時顯示扣分與加分
                if total_bonus > 0:
                    mc3_content = (
                        f"<div style='font-size:12px;color:#555;margin-bottom:4px'>學期扣分 / 加分</div>"
                        f"<div style='font-size:22px;font-weight:700;color:#e53e3e'>{total_deduct} 分</div>"
                        f"<div style='font-size:14px;font-weight:600;color:#38a169;margin-top:2px'>🌟 加分 {total_bonus} 分</div>"
                    )
                else:
                    mc3_content = (
                        f"<div style='font-size:12px;color:#555;margin-bottom:4px'>學期累計扣分</div>"
                        f"<div style='font-size:26px;font-weight:700;color:{'#e53e3e' if total_deduct>0 else '#38a169'}'>{total_deduct} 分</div>"
                    )
                mc3.markdown(
                    f"<div style='background:#f0f7ff;border-radius:12px;padding:14px 16px;border-left:4px solid #dd6b20'>"
                    f"{mc3_content}</div>", unsafe_allow_html=True)

                st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

                # ── 分頁 ──
                tab_records, tab_ranking = st.tabs(["📋 扣分明細", "🏆 各週排名"])

                with tab_records:
                    records = cls_df.sort_values("登錄時間", ascending=False)
                    if records.empty:
                        st.success("🎉 目前沒有任何評分紀錄！")
                    else:
                        import html as _html

                        # [修正] 同班+同日+同評分項目合併成一張卡片，避免多張照片變多筆
                        # 先把評分項目正規化（把 待審優良 統一視為 普通 顯示用）
                        def _display_item(item):
                            return str(item).replace("(待審優良)", "(普通)")

                        seen_groups = {}  # key=(日期, 評分項目顯示名) → 代表row index
                        group_photos = {}  # key → 所有照片 list
                        group_rids = {}   # key → 所有 rid list

                        for idx, r in records.iterrows():
                            date_str = str(r['日期'])
                            item_disp = _display_item(r['評分項目'])
                            gkey = (date_str, item_disp)
                            if gkey not in seen_groups:
                                seen_groups[gkey] = idx
                                group_photos[gkey] = []
                                group_rids[gkey] = []
                            # 合併照片
                            if str(r.get('照片路徑', '')).strip() and "http" in str(r['照片路徑']):
                                group_photos[gkey] += [p for p in str(r['照片路徑']).split(";") if "http" in p]
                            group_rids[gkey].append(str(r['紀錄ID']))

                        for gkey, rep_idx in seen_groups.items():
                            r = records.loc[rep_idx]
                            date_str, item_disp = gkey
                            rid = str(r['紀錄ID'])
                            all_photos = group_photos[gkey]
                            all_rids   = group_rids[gkey]

                            # 申訴狀態：只要有任何一筆 rid 有申訴，就顯示
                            ap_info = {}
                            for _rid in all_rids:
                                if _rid in appeal_map:
                                    ap_info = appeal_map[_rid]
                                    rid = _rid  # 用有申訴紀錄的那筆 rid 做申訴對應
                                    break
                            ap_st    = ap_info.get("status")
                            ap_reply = ap_info.get("reply")

                            trash_score = r['垃圾內掃原始分'] + r['垃圾外掃原始分']
                            if trash_score == 0: trash_score = r['垃圾原始分']
                            tot = r['內掃原始分'] + r['外掃原始分'] + trash_score + r['晨間打掃原始分']

                            is_excellent = "優良" in str(r['評分項目']) and "待審" not in str(r['評分項目'])
                            is_pending_excellent = "待審優良" in str(r['評分項目'])
                            is_normal = "普通" in str(r['評分項目']) or is_pending_excellent
                            is_corrected = r['修正'] is True or str(r['修正']).upper() == "TRUE"

                            # 決定卡片顏色
                            if ap_st == "已核可" or is_corrected:
                                card_color, border_color, tag_bg, tag_color = "#f0fff4","#38a169","#c6f6d5","#276749"
                                tag_text = "✅ 申訴成功" if ap_st == "已核可" else "🛠️ 已修正"
                            elif ap_st == "已駁回":
                                card_color, border_color, tag_bg, tag_color, tag_text = "#fff5f5","#fc8181","#fed7d7","#9b2c2c","🚫 申訴駁回"
                            elif ap_st == "待處理":
                                card_color, border_color, tag_bg, tag_color, tag_text = "#fffbeb","#f6ad55","#fefcbf","#744210","⏳ 申訴中"
                            elif is_excellent:
                                card_color, border_color, tag_bg, tag_color, tag_text = "#f0fff4","#68d391","#c6f6d5","#276749","⭐ 優良"
                            elif is_normal:
                                card_color, border_color, tag_bg, tag_color, tag_text = "#f7fafc","#a0aec0","#edf2f7","#4a5568","✅ 普通"
                            elif "學期加分" in str(r['評分項目']):
                                card_color, border_color, tag_bg, tag_color = "#f0fff4","#38a169","#c6f6d5","#276749"
                                tag_text = f"🌟 學期加 {abs(tot)} 分"
                            else:
                                card_color, border_color, tag_bg, tag_color = "#fff5f5","#fc8181","#fed7d7","#9b2c2c"
                                if tot < 0:
                                    card_color, border_color, tag_bg, tag_color = "#f0fff4","#38a169","#c6f6d5","#276749"
                                    tag_text = f"🌟 學期加 {abs(tot)} 分"
                                else:
                                    tag_text = f"❌ 扣 {tot} 分"

                            week_str = f"第{r.get('週次','')}週"
                            disp_time = str(r.get('登錄時間', ''))
                            inspector_safe = _html.escape(str(r.get('檢查人員', '未知')))
                            item_safe      = _html.escape(item_disp)
                            note_raw       = str(r.get('備註', '')).strip()
                            note_part      = f"&nbsp;|&nbsp; 📝 {_html.escape(note_raw)}" if note_raw else ""
                            time_part      = f'<div style="font-size:12px;color:#718096;margin-top:4px">登錄：{_html.escape(disp_time)}</div>' if disp_time else ""

                            st.markdown(
                                f"<div style='background:{card_color};border:1.5px solid {border_color};border-radius:12px;padding:12px 16px;margin-bottom:10px'>"
                                f"<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px'>"
                                f"<div style='font-weight:600;font-size:15px'>{date_str} <span style='color:#718096;font-size:13px'>{week_str}</span></div>"
                                f"<span style='background:{tag_bg};color:{tag_color};border-radius:20px;padding:3px 12px;font-size:13px;font-weight:600'>{tag_text}</span>"
                                f"</div>"
                                f"<div style='font-size:13px;color:#4a5568;margin-top:6px'>🧑‍✈️ {inspector_safe} &nbsp;|&nbsp; 📌 {item_safe}{note_part}</div>"
                                f"{time_part}"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                            # 申訴回覆
                            if ap_st == "已核可":
                                st.success(f"✅ 申訴成功。組長回覆：{ap_reply if ap_reply else '無'}")
                            elif ap_st == "已駁回":
                                st.error(f"🚫 申訴駁回。組長回覆：{ap_reply if ap_reply else '無'}")
                            elif ap_st == "待處理":
                                st.info("⏳ 申訴審核中，請耐心等候...")

                            # 照片與申訴整合（合併所有照片）
                            has_photo = len(all_photos) > 0
                            deadline = appeal_deadline_map.get(date_str, date.today())
                            can_appeal = not ap_st and date.today() <= deadline and (tot > 0 or r['手機人數'] > 0)

                            if has_photo or can_appeal:
                                with st.expander("📋 查看詳情" + (f"　|　📣 可申訴（截止 {deadline.strftime('%m/%d')}）" if can_appeal else "")):
                                    if has_photo:
                                        st.markdown("**📷 評分照片**")
                                        st.image(all_photos, width=200)
                                    if can_appeal:
                                        st.markdown("---")
                                        st.markdown("**📣 提出申訴**")
                                        with st.form(f"ap_{rid}"):
                                            rsn = st.text_area("申訴理由（必填）", key=f"rsn_{rid}")
                                            pf = st.file_uploader("佐證照片（選填）", type=['jpg','png'], key=f"pf_{rid}")
                                            if st.form_submit_button("送出申訴"):
                                                if time.time() - st.session_state.last_action_time < 3:
                                                    st.warning("⚠️ 系統處理中，請勿連續點擊！")
                                                elif not rsn:
                                                    st.error("請填寫申訴理由")
                                                else:
                                                    st.session_state.last_action_time = time.time()
                                                    if save_appeal({"班級": cls, "違規日期": date_str, "違規項目": r['評分項目'], "原始扣分": str(tot), "申訴理由": rsn, "對應紀錄ID": rid}, pf if pf else None):
                                                        time.sleep(1.5)
                                                        st.rerun()

                with tab_ranking:
                    pub_df = load_published_results()
                    if pub_df.empty:
                        st.info("📭 組長尚未發布任何週次的成績，請耐心等候！")
                    else:
                        available_pub_weeks = sorted(pub_df["週次"].unique(), reverse=True)
                        sel_pub_week = st.selectbox("選擇查詢週次", available_pub_weeks,
                                                    format_func=lambda w: f"第 {w} 週", key="m2_pub_week")
                        week_pub = pub_df[pub_df["週次"] == sel_pub_week]
                        cls_row = week_pub[week_pub["班級"] == cls]

                        pub_time = week_pub["發布時間"].iloc[0] if not week_pub.empty else ""

                        if not cls_row.empty:
                            cr = cls_row.iloc[0]
                            rank_val = int(cr["排名"])
                            score_val = int(cr["總成績"])
                            deduct_val = int(cr["總扣分"])
                            exc_val = int(cr["優良次數"])

                            # 用扣分決定卡片顏色，不顯示評等避免與糾察優良混淆
                            color = "#f0fff4" if deduct_val == 0 else ("#fffbeb" if deduct_val <= 3 else "#fff5f5")
                            border = "#38a169" if deduct_val == 0 else ("#f6ad55" if deduct_val <= 3 else "#fc8181")
                            st.markdown(
                                f"<div style='background:{color};border:2px solid {border};border-radius:14px;padding:20px 24px;text-align:center;margin-bottom:16px'>"
                                f"<div style='font-size:14px;color:#718096;margin-bottom:4px'>第 {sel_pub_week} 週 &nbsp;·&nbsp; {cls}</div>"
                                f"<div style='font-size:48px;font-weight:700;color:#2d3748'>#{rank_val}</div>"
                                f"<div style='font-size:15px;color:#4a5568;margin-top:6px'>"
                                f"總成績 {score_val} 分 &nbsp;|&nbsp; 扣分 {deduct_val} 分 &nbsp;|&nbsp; ⭐ 優良評分 {exc_val} 次"
                                f"</div></div>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.warning(f"找不到 {cls} 在第 {sel_pub_week} 週的排名資料。")

                        # 直接讀取發布時儲存的排名模式，不再靠偵測推斷
                        pub_rank_mode = str(week_pub["排名模式"].iloc[0]) if "排名模式" in week_pub.columns and not week_pub.empty else "年級"

                        if pub_rank_mode == "全校":
                            st.markdown("##### 全校完整排名")
                            st.dataframe(
                                week_pub[["排名","年級","班級","總扣分","優良次數","總成績"]].sort_values("排名").reset_index(drop=True),
                                hide_index=True
                            )
                        else:
                            cls_grade = next((c["grade"] for c in structured_classes if c["name"] == cls), "")
                            grade_pub = week_pub[week_pub["年級"] == cls_grade] if cls_grade else week_pub
                            if not grade_pub.empty:
                                st.markdown(f"##### {cls_grade} 完整排名")
                                st.dataframe(
                                    grade_pub[["排名","班級","總扣分","優良次數","總成績"]].reset_index(drop=True),
                                    hide_index=True
                                )
                        if pub_time:
                            st.caption(f"發布時間：{pub_time}")

    # --- Mode 3: 晨掃志工隊🧹 ---
    elif app_mode == "晨掃志工隊🧹":
        st.title("🧹 晨掃志工回報專區")
        
        cutoff_hour = 24 if sys_env == "DEV" else 16
        
        if now_tw.hour >= cutoff_hour: 
            st.error("🚫 今日回報已截止 (16:00)")
        else:
            if sys_env == "DEV" and now_tw.hour >= 16:
                st.info("🔧 **[測試機特權開啟]** 目前已超過 16:00，但因為是 DEV 環境，允許繼續測試！")
                
            my_cls = st.selectbox("選擇班級", all_classes, key="m3_cls_select")
            main_df = load_main_data()
            # 晨掃填報只需近兩週資料
            _now_week = get_week_num(today_tw)
            if _now_week >= 3:
                main_df = main_df[main_df["週次"] >= _now_week - 2]
            
            # [新增防呆] 建立一個本地暫存，記住剛送出的班級
            if "just_submitted_morning" not in st.session_state:
                st.session_state.just_submitted_morning = []
                
            is_in_sheet = not main_df[(main_df["日期"].astype(str)==str(today_tw)) & (main_df["班級"]==my_cls) & (main_df["評分項目"].astype(str).str.contains("晨間打掃"))].empty
            is_just_submitted = f"{today_tw}_{my_cls}" in st.session_state.just_submitted_morning
            
            if is_in_sheet or is_just_submitted: 
                st.warning(f"⚠️ {my_cls} 今日已回報，或資料正在系統排隊處理中囉！")
            else:
                duty_df, _ = get_daily_duty(today_tw)
                
                has_duty = False 
                area_name_str = ""
                n_std = 4
                
                # 1. 先查今天有沒有排班
                if not duty_df.empty:
                    m_d = duty_df[duty_df["負責班級"]==my_cls]
                    if not m_d.empty:
                        has_duty = True
                        area_name_str = str(m_d.iloc[0].get('掃地區域', '未指定區域'))
                        try: n_std = int(m_d.iloc[0].get('標準人數', 4))
                        except Exception: n_std = 4
                
                is_makeup = False
                found_duty = has_duty

                # 2. 如果今天沒排班，且是一、二年級 -> 往前翻找本週班表
                if not has_duty and ("一" in my_cls or "二" in my_cls):
                    from datetime import timedelta
                    
                    # [V5.30 Patch 1] 防呆：先檢查本週是不是已經交過晨掃了！(不管有沒有被核可)
                    start_of_week = today_tw - timedelta(days=today_tw.weekday())
                    already_done = False
                    for _, r in main_df[main_df["班級"] == my_cls].iterrows():
                        if "晨間打掃" in str(r["評分項目"]):
                            try:
                                r_date = pd.to_datetime(str(r["日期"])).date()
                                if start_of_week <= r_date <= today_tw:
                                    already_done = True
                                    break
                            except Exception: pass  # 日期解析失敗忽略

                    # 如果本週「還沒交過」，才開啟時光機去查哪一天缺交
                    if not already_done:
                        for d in range(1, 7):
                            past_date = today_tw - timedelta(days=d)
                            if past_date < start_of_week: 
                                break # 只找本週的紀錄，超過本週就不補了
                                
                            p_duty, _ = get_daily_duty(past_date)
                            if not p_duty.empty and "負責班級" in p_duty.columns:
                                m_p = p_duty[p_duty["負責班級"].astype(str)==my_cls]
                                if not m_p.empty:
                                    area_name_str = str(m_p.iloc[0].get('掃地區域', '未指定區域'))
                                    try: n_std = int(m_p.iloc[0].get('標準人數', 4))
                                    except Exception: n_std = 4
                                    is_makeup = True # 判定為跨日補掃
                                    found_duty = True
                                    break
                                
                # 3. 如果今天有排班，但超過 15:00 -> 當日遲交補掃
                if has_duty and now_tw.hour >= 15:
                    is_makeup = True

                # 若完全找不到班表 (例如三年級沒排班，或一二年級連上週都沒排)
                if not found_duty:
                    st.success(f"🎉 恭喜！系統顯示 **{my_cls}** 近期沒有被分配到晨掃任務，好好休息吧！")
                    st.balloons()
                else:
                    areas = [a.strip() for a in area_name_str.split('、') if a.strip()]
                    if not areas: areas = ["打掃區域"]
                    
                    # 依據是否為補掃，顯示不同的上方提示
                    if is_makeup:
                        st.info(f"💡 **{my_cls}** 進行補打掃任務。本班任務總應到: {n_std} 人\n\n*(補掃通過將給予學期總分 +1，並核發志工時數)*")
                    else:
                        st.info(f"📍 本班任務總應到: {n_std} 人")
                    
                    # 顯示每日廣播大聲公
                    daily_task = SYSTEM_CONFIG.get("daily_morning_task", "")
                    if daily_task:
                        formatted_task = daily_task.replace('\n', '<br>')
                        mascot_url = "https://drive.google.com/thumbnail?id=128ITPXtpGNuI-wLIt6p-qd4ZNNhCGbhd"
                        
                        bubble_html = f"""
                        <style>
                        .mascot-container {{ display: flex; align-items: flex-start; margin-bottom: 20px; gap: 15px; }}
                        .mascot-img {{ width: 160px; flex-shrink: 0; }}
                        .speech-bubble {{
                            position: relative; background: #FFF3CD; border-radius: 15px; padding: 15px 20px;
                            color: #664d03; font-size: 16px; box-shadow: 2px 4px 10px rgba(0,0,0,0.1); border: 2px solid #ffecb5; flex-grow: 1; 
                        }}
                        .speech-bubble::before {{ content: ''; position: absolute; left: -20px; top: 30px; width: 0; height: 0; border: 10px solid transparent; border-right-color: #ffecb5; }}
                        .speech-bubble::after {{ content: ''; position: absolute; left: -16px; top: 30px; width: 0; height: 0; border: 10px solid transparent; border-right-color: #FFF3CD; }}
                        @media (max-width: 500px) {{
                            .mascot-img {{ width: 120px; }}
                            .speech-bubble {{ font-size: 14px; padding: 10px 15px; }}
                        }}
                        </style>
                        <div class="mascot-container">
                            <img src="{mascot_url}" class="mascot-img" />
                            <div class="speech-bubble">
                                <strong>📢 組長廣播 / 今日任務：</strong><br>
                                {formatted_task}
                            </div>
                        </div>
                        """
                        st.markdown(bubble_html, unsafe_allow_html=True)
                    
                    with st.form("vol_form"):
                        st.write("請依照下方分配的區域，分別填寫打掃同學並上傳照片：")
                        
                        present_dict = {}
                        files_dict = {}
                        class_roster = [s for s, c in ROSTER_DICT.items() if c == my_cls]
                        
                        for idx, area in enumerate(areas):
                            with st.container(border=True):
                                st.markdown(f"#### 🏷️ 區域 {idx+1}: **{area}**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    present_dict[area] = st.multiselect(f"✅ 負責此區同學", class_roster, key=f"ms_{idx}")
                                with col2:
                                    files_dict[area] = st.file_uploader(f"📸 {area} 成果照片", accept_multiple_files=True, type=['jpg','png'], key=f"fu_{idx}")
                                    
                        # 依據 is_makeup 動態變更按鈕文字
                        btn_text = "🚀 我們完成補打掃了喔" if is_makeup else "🚀 確認送出全部回報"
                        
                        if st.form_submit_button(btn_text):
                            if time.time() - st.session_state.last_action_time < 3:
                                st.warning("⚠️ 系統處理中，請勿連續點擊！")
                            else:
                                st.session_state.last_action_time = time.time()
                                
                                all_present = []
                                all_files = []
                                note_parts = []
                                
                                for area in areas:
                                    if present_dict[area]:
                                        all_present.extend(present_dict[area])
                                        note_parts.append(f"[{area}]負責:{','.join(present_dict[area])}")
                                    if files_dict[area]:
                                        all_files.extend(files_dict[area])
                                        
                                all_present = list(set(all_present))
                                final_note = " | ".join(note_parts)
                                
                                if not all_present or not all_files:
                                    st.error("❌ 請至少選擇一位打掃同學，並上傳至少一張照片！")
                                else:
                                    # 依據 is_makeup 變更存入的任務名稱
                                    task_name = "晨間打掃(補掃)" if is_makeup else "晨間打掃"
                                    # [給分自動計算] 把應到人數也存進備註，供組長審核時自動算分
                                    score_note = f"[應到:{n_std}人 實到:{len(all_present)}人 {'補掃' if is_makeup else '準時'}] {final_note}"

                                    ok = save_entry(
                                        {
                                            "日期": str(today_tw), 
                                            "週次": get_week_num(today_tw),
                                            "班級": my_cls, 
                                            "評分項目": task_name, 
                                            "檢查人員": f"志工(實到:{len(all_present)})", 
                                            "登錄時間": now_tw.strftime("%Y-%m-%d %H:%M:%S"), 
                                            "晨間打掃原始分": 0, 
                                            "備註": score_note
                                        }, 
                                        uploaded_files=all_files, 
                                        student_list=all_present, 
                                        custom_hours=0.5, 
                                        custom_category="晨掃志工"
                                    )
                                    if ok:
                                    # [新增防呆] 送出成功後，立刻把今天和班級記在手機瀏覽器裡
                                        st.session_state.just_submitted_morning.append(f"{today_tw}_{my_cls}")
                                        st.success("✅ 回報成功！所有區域皆已記錄，辛苦了！")
                                        time.sleep(1.5)
                                        st.rerun()
    # --- Mode 4: 組長後台 ---
    elif app_mode == "組長ㄉ窩💃":
        st.title("⚙️ 管理後台")
        metrics = get_queue_metrics()
        hb_sec = get_worker_heartbeat_sec()
        ls_sec = get_last_success_sec()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("待處理", metrics.get("pending", 0))
        col2.metric("失敗", metrics.get("failed", 0))
        col3.metric("延遲(s)", int(metrics.get("oldest_pending_sec", 0)))
        
        hb_status = "🟢 正常運作" if hb_sec < 60 else "🔴 已休眠/停止"
        is_dry_run = str(st.secrets.get("system_config", {}).get("dry_run", "false")).lower() in ["true", "1"]
        
        if is_dry_run: hb_status = "🟡 演習模式 (Dry Run)"
        
        if ls_sec == 999999: ls_text = "無紀錄"
        elif ls_sec < 120: ls_text = f"✅ {int(ls_sec)}秒前"
        else: ls_text = f"⚠️ {int(ls_sec//60)}分鐘前 (API可能卡住)"
            
        col4.metric("背景 Worker", f"{hb_status}", f"心跳: {int(hb_sec)}秒前 | 成功: {ls_text}")
        
        last_err = get_last_error_summary()
        if last_err != "無紀錄":
            st.error(f"🚨 **最後錯誤紀錄:** {last_err}")

        pwd_input = st.text_input("管理密碼", type="password", key="admin_pwd")
        if pwd_input == st.secrets["system_config"]["admin_password"]:
            
            t_mon, t_rollcall, t4, t_appeal, t_excellent, t2, t1, t_settings, t3, t_debt = st.tabs([
                "👀 衛生糾察", "👮 環保糾察", "📝 扣分明細", "📣 申訴", "⭐ 優良審核", "📊 成績總表", 
                "🧹 晨掃審核", "⚙️ 設定", "🎖️ 服務時數發放", "🤝 愛校與欠時管理"  # [新增] 愛校服務 2.0
            ])
            
            with t_mon:
                st.subheader("🕵️ 今日「衛生糾察」進度監控")
                monitor_date = st.date_input("監控日期", today_tw, key="monitor_date")
                st.caption(f"📅 此區顯示負責「內掃、外掃、機動」的評分糾察進度。")

                df = load_main_data()
                # 監控只需近兩週
                _nw = get_week_num(today_tw)
                if _nw >= 3:
                    df = df[df["週次"] >= _nw - 2]
                submitted_names = set()
                if not df.empty:
                    today_records = df[df["日期"].astype(str) == str(monitor_date)]
                    submitted_names = set(today_records["檢查人員"].unique())

                cleaning_inspectors = [p for p in INSPECTOR_LIST if any(x in p.get("raw_role", "") for x in ["內掃", "外掃", "機動", "隊長", "組長"])]
                
                regular_inspectors, mobile_inspectors = [], []
                for p in cleaning_inspectors:
                    p_name = p["label"]
                    is_mobile = len(p.get("assigned_classes", [])) == 0
                    status_obj = {"name": p_name, "role_desc": p.get("raw_role", ""), "done": p_name in submitted_names}
                    if is_mobile: mobile_inspectors.append(status_obj)
                    else: regular_inspectors.append(status_obj)

                col_reg, col_mob = st.columns(2)
                with col_reg:
                    st.write("#### 🔴 班級評分員 (未完成)")
                    missing_reg = [x for x in regular_inspectors if not x["done"]]
                    if missing_reg:
                        for p in missing_reg: st.error(f"❌ {p['name']}")
                    else: st.success("🎉 全員完成！")
                with col_mob:
                    st.write("#### 🟠 機動/隊長 (未完成)")
                    st.caption("機動人員若今日無違規，可能不會送出資料。")
                    missing_mob = [x for x in mobile_inspectors if not x["done"]]
                    if missing_mob:
                        for p in missing_mob: st.warning(f"⚠️ {p['name']} \n ({p['role_desc']})")
                    else: st.success("🎉 全員完成！")

            with t_rollcall:
                st.subheader("👮 環保糾察 (資收場) 出勤點名")
                st.info("💡 說明：此區專為資收場的環保糾察設計。勾選沒來的人，系統會自動幫有來的人發放 0.25 小時。")
                
                rc_date = st.date_input("出勤日期", today_tw, key="insp_rc_date")
                
                trash_inspectors = [p for p in INSPECTOR_LIST if "垃圾" in p.get("raw_role", "") or "回收" in p.get("raw_role", "") or "環保" in p.get("raw_role", "")]
                insp_names = [p["label"] for p in trash_inspectors]
                
                if not insp_names:
                    st.warning("⚠️ 目前名單中沒有負責「環保/垃圾/回收」的糾察。")
                else:
                    with st.form("insp_rc_form"):
                        st.write(f"資收場糾察名單共 {len(insp_names)} 人")
                        absent_insps = st.multiselect("❌ 勾選【請假 / 未到】的糾察 (扣除法)", insp_names)
                        present_insps = [n for n in insp_names if n not in absent_insps]
                        
                        st.write(f"✅ 預計發放對象：共 {len(present_insps)} 人 (每人 0.25 小時)")
                        
                        if st.form_submit_button("🚀 發放環保糾察時數"):
                            if time.time() - st.session_state.last_action_time < 3:
                                st.warning("⚠️ 系統處理中，請勿連續點擊！")
                            else:
                                st.session_state.last_action_time = time.time()
                                present_ids = [name.split("學號:")[1].strip() for name in present_insps if "學號:" in name]
                                if present_ids:
                                    payload = {
                                        "student_list": present_ids,
                                        "date": str(rc_date),
                                        "class_name": "糾察隊",
                                        "category": "資源回收糾察",
                                        "hours": 0.25
                                    }
                                    enqueue_task("service_hours_only", payload)
                                    st.success(f"✅ 已排程發放 {len(present_ids)} 人的出勤時數！(系統會自動阻擋同一天的重複發放)")
                                    time.sleep(1.5)
                                    st.rerun()
                                else:
                                    st.warning("沒有可發放時數的對象")

            with t4:
                df = load_main_data()
                if not df.empty:
                    st.dataframe(df[["登錄時間", "日期", "班級", "評分項目", "檢查人員", "備註", "違規細項", "紀錄ID"]].sort_values("登錄時間", ascending=False))

            with t_appeal:
                st.subheader("📣 申訴審核")
                ap_df = load_appeals()
                pending_aps = ap_df[ap_df["處理狀態"]=="待處理"]
                
                if pending_aps.empty: 
                    st.success("目前無待審核的申訴案件。")
                else:
                    for i, r in pending_aps.iterrows():
                        with st.container(border=True):
                            c1, c2 = st.columns([3,2])
                            c1.write(f"### {r['班級']} | {r['違規項目']} (扣 {r['原始扣分']} 分)")
                            c1.write(f"**申訴理由**: {r['申訴理由']}")
                            c1.caption(f"違規日期: {r['違規日期']} | 申訴時間: {r['登錄時間']}")
                            
                            img_urls = str(r.get('佐證照片', ''))
                            if img_urls and "http" in img_urls:
                                c2.image([p for p in img_urls.split(";") if "http" in p], width=250)
                            else:
                                c2.info("無佐證照片")
                                
                            reply_text = c1.text_input("💬 審核回覆 (填寫後學生將在查詢頁面看到此說明)", key=f"reply_{i}")
                            
                            col_btn1, col_btn2 = c1.columns(2)
                            if col_btn1.button("✅ 核可並撤銷扣分", key=f"ok_{i}"): 
                                update_appeal_status(i, "已核可", r["對應紀錄ID"], reply_text)
                                st.rerun()
                            if col_btn2.button("🚫 駁回維持原判", key=f"ng_{i}"): 
                                update_appeal_status(i, "已駁回", r["對應紀錄ID"], reply_text)
                                st.rerun()

            with t_excellent:
                st.subheader("⭐ 優良審核")
                ex_df = load_main_data()
                pending_ex = ex_df[ex_df["評分項目"].astype(str).str.contains("待審優良")] if not ex_df.empty else pd.DataFrame()

                if pending_ex.empty:
                    st.success("🎉 目前沒有待審核的優良紀錄！")
                else:
                    if "approved_excellent_ids" not in st.session_state:
                        st.session_state.approved_excellent_ids = set()

                    # [修正] 依 班級+日期+評分項目 群組，避免多張照片變多筆審核卡
                    pending_ex = pending_ex[~pending_ex["紀錄ID"].astype(str).isin(st.session_state.approved_excellent_ids)]
                    groups = list(pending_ex.groupby(["班級", "日期", "評分項目"]))
                    st.caption(f"共 {len(groups)} 筆待審核，審核後不會跳頁，可以繼續審核其他筆。")

                    for (cls_name, date_val, item_val), grp in groups:
                        r = grp.iloc[0]
                        all_rids = grp["紀錄ID"].astype(str).tolist()
                        all_photos = []
                        for _, gr in grp.iterrows():
                            if "http" in str(gr.get("照片路徑", "")):
                                all_photos += [p for p in str(gr["照片路徑"]).split(";") if "http" in p]

                        with st.container(border=True):
                            c1, c2, c3 = st.columns([2, 2, 1])
                            c1.write(f"⭐ **{cls_name}** | {r['檢查人員']}")
                            c1.caption(f"{date_val} | {item_val}")
                            c1.write(f"📝 {r['備註']}")
                            if grp.shape[0] > 1:
                                c1.caption(f"（共 {grp.shape[0]} 筆紀錄合併，{len(all_photos)} 張照片）")
                            if all_photos:
                                c2.image(all_photos[:6], width=120)

                            gid = all_rids[0]  # 用第一筆 ID 當按鈕 key

                            def _approve_group(rids, eval_suffix):
                                try:
                                    ws = get_worksheet(SHEET_TABS["main"])
                                    if not ws:
                                        st.error("❌ 無法連線至 Google Sheets，請稍後再試")
                                        return False
                                    id_list = ws.col_values(EXPECTED_COLUMNS.index("紀錄ID") + 1)
                                    # [Fix] 將 id_list 統一轉為 str 並 strip，避免型別不一致
                                    id_list = [str(v).strip() for v in id_list]
                                    success_count = 0
                                    for rid in rids:
                                        rid = str(rid).strip()
                                        if rid in id_list:
                                            ridx = id_list.index(rid) + 1
                                            matched = ex_df.loc[ex_df["紀錄ID"].astype(str).str.strip() == rid, "評分項目"]
                                            if not matched.empty:
                                                new_item = str(matched.iloc[0]).replace("(待審優良)", eval_suffix)
                                                ws.update_cell(ridx, EXPECTED_COLUMNS.index("評分項目") + 1, new_item)
                                            # [Fix] 只有成功更新 Sheet 後才標記為已處理
                                            st.session_state.approved_excellent_ids.add(rid)
                                            success_count += 1
                                        else:
                                            st.warning(f"⚠️ 紀錄 {rid[:12]}... 在 Sheet 中找不到，可能資料尚在佇列中，請稍後重新整理再試。")
                                    load_main_data.clear()
                                    return success_count > 0
                                except Exception as e:
                                    st.error(f"❌ 審核寫入失敗: {e}")
                                    return False

                            if c3.button("✅ 核可優良", key=f"ex_ok_{gid}"):
                                if _approve_group(all_rids, "(優良)"):
                                    c1.success("✅ 已核可為優良！")
                            if c3.button("🚫 駁回(改普通)", key=f"ex_ng_{gid}"):
                                if _approve_group(all_rids, "(普通)"):
                                    c1.info("已改為普通。")

                    if st.button("🔄 重新整理列表", key="refresh_excellent"):
                        st.session_state.approved_excellent_ids.clear()
                        load_main_data.clear()
                        st.rerun()

            with t2:
                st.subheader("📊 成績總表")
                full = load_full_semester_data_for_export()

                # ── 共用計算函式 ──────────────────────────────────────────
                def calc_scores(df_raw):
                    """回傳含結算欄位的 DataFrame（已排除優良/普通紀錄，以及申訴核可的修正紀錄）"""
                    # [Fix] 加括號修正運算子優先順序（& 優先於 ~，不加括號邏輯會錯）
                    df = df_raw[
                        (~df_raw["評分項目"].astype(str).str.contains("優良|普通")) &
                        (~df_raw["修正"].astype(str).str.upper().eq("TRUE"))   # 排除申訴已核可的紀錄
                    ].copy()
                    df["內掃結算"] = df["內掃原始分"].clip(upper=2)
                    df["外掃結算"] = df["外掃原始分"].clip(upper=2)
                    trash = df["垃圾內掃原始分"] + df["垃圾外掃原始分"]
                    trash = trash.where(trash > 0, df["垃圾原始分"])
                    df["垃圾結算"] = trash.clip(upper=2)
                    df["總扣分"] = df["內掃結算"] + df["外掃結算"] + df["垃圾結算"] + df["晨間打掃原始分"].clip(lower=0) + df["手機人數"]
                    return df

                def build_ranking(scored_df, df_all, classes_struct, base_score=90):
                    """依班級彙總扣分，合併完整班級清單，加上總成績與優良次數"""
                    rep = scored_df.groupby("班級")["總扣分"].sum().reset_index()
                    # [Fix] 計算優良次數：排除「待審優良」，只算正式核可的優良
                    excellent_mask = (
                        df_all["評分項目"].astype(str).str.contains("優良") &
                        ~df_all["評分項目"].astype(str).str.contains("待審")
                    )
                    excellent_counts = df_all[excellent_mask].groupby("班級").size().reset_index(name="優良次數")
                    cls_df = pd.DataFrame(classes_struct).rename(columns={"grade": "年級", "name": "班級"})
                    fin = pd.merge(cls_df, rep, on="班級", how="left").fillna(0)
                    fin = pd.merge(fin, excellent_counts, on="班級", how="left").fillna(0)
                    fin["總成績"] = base_score - fin["總扣分"]
                    fin["優良次數"] = fin["優良次數"].astype(int)
                    return fin

                def add_rank_and_label(fin_df, by_grade=False, threshold_good=3):
                    """排序並加上排名、評等欄位；同分以優良次數排名，使用標準競賽排名（並列同名次，下一名跳號）"""
                    def label(s):
                        if s == 0:   return "⭐ 優良"
                        if s <= threshold_good: return "✅ 普通"
                        return "⚠️ 需加強"

                    def apply_competition_rank(df):
                        # 先依「總成績高→優良次數多」排序
                        df = df.sort_values(["總成績", "優良次數"], ascending=[False, False]).reset_index(drop=True)
                        # 標準競賽排名：同名次並列，下一名跳號（例：1,1,1,4,5）
                        ranks = []
                        rank = 1
                        for i, row in df.iterrows():
                            if i == 0:
                                ranks.append(rank)
                            else:
                                prev = df.iloc[i-1]
                                if row["總成績"] == prev["總成績"] and row["優良次數"] == prev["優良次數"]:
                                    ranks.append(ranks[-1])  # 並列同名次
                                else:
                                    rank = i + 1  # 跳到實際位置號
                                    ranks.append(rank)
                        df.insert(0, "排名", ranks)
                        df["評等"] = df["總扣分"].apply(label)
                        return df

                    if not by_grade:
                        return apply_competition_rank(fin_df.copy())
                    else:
                        pieces = []
                        for g in sorted(fin_df["年級"].unique()):
                            if g == "其他": continue
                            pieces.append(apply_competition_rank(fin_df[fin_df["年級"]==g].copy()))
                        return pd.concat(pieces, ignore_index=True) if pieces else fin_df

                def build_detail(scored_df):
                    """各班扣分明細：每筆違規紀錄，含日期/週次/評分項目/扣分/備註"""
                    cols = ["日期", "週次", "班級", "評分項目", "檢查人員",
                            "內掃結算", "外掃結算", "垃圾結算", "手機人數", "總扣分", "備註", "違規細項"]
                    avail = [c for c in cols if c in scored_df.columns]
                    detail = scored_df[scored_df["總扣分"] > 0][avail].sort_values(["班級","日期"])
                    return detail

                def to_excel_bytes(sheets_dict):
                    """產生多分頁 Excel，sheets_dict = {分頁名稱: DataFrame}"""
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                        for sheet_name, df in sheets_dict.items():
                            safe_name = sheet_name[:31]  # Excel 分頁名稱上限 31 字
                            df.to_excel(writer, sheet_name=safe_name, index=False)
                            ws = writer.sheets[safe_name]
                            # 自動調整欄寬
                            for col_cells in ws.columns:
                                max_len = max((len(str(c.value)) if c.value else 0) for c in col_cells)
                                ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 4, 40)
                    buf.seek(0)
                    return buf.getvalue()

                # ── UI ───────────────────────────────────────────────────
                if full.empty:
                    st.info("目前無評分資料")
                else:
                    tab_week, tab_semester = st.tabs(["📅 單週成績結算", "🏆 全學期總結算"])

                    with tab_week:
                        available_weeks = sorted([w for w in full["週次"].unique() if w > 0])
                        if not available_weeks:
                            st.warning("尚無有效的週次資料")
                        else:
                            sel_week = st.selectbox("請選擇結算週次", available_weeks, index=len(available_weeks)-1)
                            is_fall = (today_tw.month >= 8 or today_tw.month == 1)
                            default_mode = "年級 (上學期制)" if is_fall else "全校 (下學期制)"
                            st.info(f"💡 系統偵測目前為 **{'上' if is_fall else '下'}學期**，預設採用 **{default_mode}** 排名。")
                            rank_mode = st.radio("排名方式 (可手動更改)", ["年級", "全校"], index=0 if is_fall else 1, horizontal=True)
                            
                            col_calc, col_refresh = st.columns([3, 1])
                            if col_refresh.button("🔄 重新讀取資料", key="refresh_export", help="若有申訴剛核可，請先點此確保資料是最新的"):
                                load_full_semester_data_for_export.clear()
                                load_main_data.clear()
                                st.success("✅ 資料已刷新！")
                                st.rerun()

                            if col_calc.button("🚀 計算並顯示當週成績"):
                                week_raw = full[full["週次"] == sel_week]  # 當週原始資料
                                scored = calc_scores(week_raw)
                                fin = build_ranking(scored, week_raw, structured_classes)  # 優良次數只算當週
                                by_grade = (rank_mode == "年級")
                                fin_ranked = add_rank_and_label(fin, by_grade=by_grade)
                                detail = build_detail(scored)
                                # 儲存供發布使用
                                st.session_state["last_computed_week"] = sel_week
                                st.session_state["last_computed_ranking"] = fin_ranked

                                # ── 畫面顯示 ──
                                if by_grade:
                                    for g in sorted(fin_ranked["年級"].unique()):
                                        st.write(f"#### {g} 排名")
                                        g_df = fin_ranked[fin_ranked["年級"]==g]
                                        st.dataframe(g_df[["排名","班級","總扣分","優良次數","總成績","評等"]], hide_index=True)
                                else:
                                    st.dataframe(fin_ranked[["排名","年級","班級","總扣分","優良次數","總成績","評等"]], hide_index=True)

                                st.markdown("---")
                                st.write(f"##### 📋 第 {sel_week} 週扣分明細（共 {len(detail)} 筆）")
                                if detail.empty:
                                    st.success("本週無任何扣分紀錄！")
                                else:
                                    st.dataframe(detail, hide_index=True)

                                # ── Excel 下載 ──
                                st.markdown("---")
                                sheets = {"排名總表": fin_ranked[["排名","年級","班級","總扣分","優良次數","總成績","評等"]]}
                                if by_grade:
                                    for g in sorted(fin_ranked["年級"].unique()):
                                        sheets[f"{g}排名"] = fin_ranked[fin_ranked["年級"]==g][["排名","班級","總扣分","優良次數","總成績","評等"]]
                                sheets["扣分明細"] = detail
                                excel_bytes = to_excel_bytes(sheets)
                                st.download_button(
                                    label=f"📥 下載第 {sel_week} 週成績報表 (Excel)",
                                    data=excel_bytes,
                                    file_name=f"衛生成績_第{sel_week}週_{today_tw.strftime('%Y%m%d')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )

                            # ── 發布按鈕（計算完後才會出現）──
                            if st.session_state.get("last_computed_week") == sel_week and \
                               st.session_state.get("last_computed_ranking") is not None:
                                st.markdown("---")
                                st.info(f"💡 計算完成後，可將第 {sel_week} 週成績發布給學生查詢。")
                                if st.button(f"📢 發布第 {sel_week} 週成績給學生", key="publish_week_btn"):
                                    ok, msg = publish_week_results(
                                        sel_week,
                                        st.session_state["last_computed_ranking"],
                                        rank_mode=rank_mode  # 把排名模式一起存進去
                                    )
                                    if ok:
                                        st.success(f"✅ {msg}")
                                    else:
                                        st.error(f"❌ 發布失敗：{msg}")

                    with tab_semester:
                        st.write("計算全學期累計總扣分與總成績")
                        sem_rank_mode = st.radio("學期排名方式", ["全校", "年級"], horizontal=True, key="sem_rank")

                        if st.button("🚀 計算並顯示全學期成績", key="sem_btn"):
                            scored = calc_scores(full)
                            fin = build_ranking(scored, full, structured_classes)
                            by_grade = (sem_rank_mode == "年級")
                            fin_ranked = add_rank_and_label(fin, by_grade=by_grade, threshold_good=10)
                            detail = build_detail(scored)

                            # ── 畫面顯示 ──
                            if by_grade:
                                for g in sorted(fin_ranked["年級"].unique()):
                                    st.write(f"#### {g} 排名")
                                    g_df = fin_ranked[fin_ranked["年級"]==g]
                                    st.dataframe(g_df[["排名","班級","總扣分","優良次數","總成績","評等"]], hide_index=True)
                            else:
                                st.dataframe(fin_ranked[["排名","年級","班級","總扣分","優良次數","總成績","評等"]], hide_index=True)

                            st.markdown("---")
                            st.write(f"##### 📋 全學期扣分明細（共 {len(detail)} 筆）")
                            if detail.empty:
                                st.success("學期內無任何扣分紀錄！")
                            else:
                                # 扣分明細太多時，提供依班級篩選
                                all_cls_options = ["全部班級"] + sorted(detail["班級"].unique().tolist())
                                sel_cls_filter = st.selectbox("篩選班級（可只看單班明細）", all_cls_options, key="sem_detail_filter")
                                detail_show = detail if sel_cls_filter == "全部班級" else detail[detail["班級"]==sel_cls_filter]
                                st.dataframe(detail_show, hide_index=True)

                            # ── Excel 下載（多分頁：總排名 + 各年級 + 各班明細） ──
                            st.markdown("---")
                            sheets = {"學期排名總表": fin_ranked[["排名","年級","班級","總扣分","優良次數","總成績","評等"]]}
                            if by_grade:
                                for g in sorted(fin_ranked["年級"].unique()):
                                    sheets[f"{g}排名"] = fin_ranked[fin_ranked["年級"]==g][["排名","班級","總扣分","優良次數","總成績","評等"]]
                            # 各班各週明細：樞紐表（班級 x 週次）
                            if not detail.empty:
                                pivot = detail.pivot_table(index="班級", columns="週次", values="總扣分", aggfunc="sum", fill_value=0)
                                pivot["學期總扣分"] = pivot.sum(axis=1)
                                pivot = pivot.reset_index()
                                sheets["各班週次扣分樞紐"] = pivot
                                sheets["全學期扣分明細"] = detail
                                # 每個年級各自一張扣分明細分頁
                                cls_df_map = pd.DataFrame(structured_classes).rename(columns={"grade":"年級","name":"班級"})
                                for g in sorted(cls_df_map["年級"].unique()):
                                    if g == "其他": continue
                                    g_classes = cls_df_map[cls_df_map["年級"]==g]["班級"].tolist()
                                    g_detail = detail[detail["班級"].isin(g_classes)]
                                    if not g_detail.empty:
                                        sheets[f"{g}扣分明細"] = g_detail
                            excel_bytes = to_excel_bytes(sheets)
                            st.download_button(
                                label="📥 下載全學期成績報表 (Excel，含多分頁)",
                                data=excel_bytes,
                                file_name=f"衛生成績_全學期_{today_tw.strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
            with t1:
                # [V5.29 Patch] 本週晨掃進度追蹤 (含過去缺交)
                st.subheader("🕵️‍♀️ 晨掃進度追蹤 (本週)")
                main_df = load_main_data()
                # 晨掃進度追蹤只需近兩週
                _nw2 = get_week_num(today_tw)
                if _nw2 >= 3:
                    main_df = main_df[main_df["週次"] >= _nw2 - 2]
                
                from datetime import timedelta
                # 計算本週一是哪一天
                start_of_week = today_tw - timedelta(days=today_tw.weekday())
                
                weekly_assigned = {}
                # 1. 抓取從本週一到今天，每天被排班的班級
                for i in range((today_tw - start_of_week).days + 1):
                    check_date = start_of_week + timedelta(days=i)
                    c_duty_df, _ = get_daily_duty(check_date)
                    if not c_duty_df.empty:
                        for c in c_duty_df["負責班級"].dropna().astype(str).tolist():
                            # [V5.30 Patch 2] 確保班級名稱沒有前後空白，避免對不上
                            if c.strip(): weekly_assigned[c.strip()] = check_date
                            
                # 2. 抓取本週「有交過任何晨掃紀錄」(包含準時交跟跨日補掃) 的班級
                submitted_classes = set()
                for _, r in main_df.iterrows():
                    if "晨間打掃" in str(r["評分項目"]):
                        try:
                            # 安全地將字串轉換為日期進行比較
                            r_date = pd.to_datetime(str(r["日期"])).date()
                            if start_of_week <= r_date <= today_tw:
                                submitted_classes.add(str(r["班級"]))
                        except Exception: pass  # 日期解析失敗忽略
                        
                # 3. 交叉比對找出缺交名單
                today_missing = []
                past_missing = []
                
                for cls, a_date in weekly_assigned.items():
                    if cls not in submitted_classes:
                        if a_date == today_tw:
                            today_missing.append(cls)
                        else:
                            # 如果是過去缺交的，在後面加上 (月/日) 標籤
                            past_missing.append(f"{cls} ({a_date.month}/{a_date.day})")
                            
                # 4. 將結果顯示在畫面上
                if not weekly_assigned:
                    st.info("本週至今無晨掃排班任務。")
                elif not today_missing and not past_missing:
                    st.success("🎉 太棒了！本週至今所有排定班級皆已完成晨掃回報！")
                else:
                    # 顯示今天的缺交狀態
                    if today_missing:
                        st.error(f"🚨 **今日尚未回報 ({len(today_missing)}班)：** {', '.join(sorted(today_missing))}")
                    else:
                        st.success("🎉 今日排定班級皆已完成回報！")
                        
                    # 顯示過去尚未補掃的狀態 (這就是妳要的功能！)
                    if past_missing:
                        st.warning(f"⚠️ **本週未補掃名單 ({len(past_missing)}班)：** {', '.join(sorted(past_missing))}")
                        
                st.markdown("---")
                st.subheader("📝 待審核回報列表")

                # [防跳掉] 用 session_state 記錄本地已審核的 ID，避免每次按鈕都刷新整頁
                if "approved_morning_ids" not in st.session_state:
                    st.session_state.approved_morning_ids = set()

                df = main_df
                
                # [Fix] 找出本週已經有「審核完成」紀錄的班級+日期組合
                # 這些班級的其他重複紀錄應視為已處理，不再顯示在待審列表
                approved_cls_dates = set()
                for _, r in df.iterrows():
                    item_str = str(r["評分項目"])
                    if item_str in ["晨間打掃(學期加分)", "晨間打掃(已駁回)"]:
                        try:
                            r_date = pd.to_datetime(str(r["日期"])).date()
                            if start_of_week <= r_date <= today_tw:
                                approved_cls_dates.add((str(r["班級"]).strip(), str(r["日期"]).strip()))
                        except Exception:
                            pass
                
                pending_df = df[
                    df["評分項目"].isin(["晨間打掃", "晨間打掃(當日補掃)", "晨間打掃(補掃)"]) &
                    (df["晨間打掃原始分"] == 0) &
                    (~df["修正"]) &
                    (~df["紀錄ID"].astype(str).isin(st.session_state.approved_morning_ids))
                ].drop_duplicates(subset=["紀錄ID"])
                
                # [Fix] 排除已有審核完成紀錄的重複項目
                if approved_cls_dates and not pending_df.empty:
                    dup_mask = pending_df.apply(
                        lambda r: (str(r["班級"]).strip(), str(r["日期"]).strip()) in approved_cls_dates,
                        axis=1
                    )
                    dup_count = dup_mask.sum()
                    pending_df = pending_df[~dup_mask]
                    if dup_count > 0:
                        st.info(f"ℹ️ 已自動排除 {dup_count} 筆重複紀錄（該班級同日已有審核結果）。")

                if pending_df.empty:
                    st.success("🎉 目前沒有待審核的晨掃回報！")
                else:
                    st.caption(f"共 {len(pending_df)} 筆待審核，審核後不會立刻跳頁，可以繼續審核其他筆。")

                # [Fix #5] _do_approve 移至迴圈外，所有依賴都透過參數明確傳入，
                # 避免 Python closure 在迴圈中捕捉到最後一次的 r 變數
                def _do_approve(record_id, s_val, note_text, reply, col_ref, cached_main_df):
                    ws = get_worksheet(SHEET_TABS["main"])
                    id_list = ws.col_values(EXPECTED_COLUMNS.index("紀錄ID") + 1)
                    # [Fix] 統一轉為 str 並 strip，避免型別或空白不一致
                    id_list = [str(v).strip() for v in id_list]
                    record_id_str = str(record_id).strip()
                    if record_id_str in id_list:
                        ridx = id_list.index(record_id_str) + 1
                        ws.update_cell(ridx, EXPECTED_COLUMNS.index("晨間打掃原始分") + 1, s_val)
                        ws.update_cell(ridx, EXPECTED_COLUMNS.index("評分項目") + 1, "晨間打掃(學期加分)")
                        matched = cached_main_df.loc[cached_main_df["紀錄ID"].astype(str) == str(record_id), "備註"]
                        old_note = str(matched.iloc[0]) if not matched.empty else ""
                        new_note = f"{old_note} \n組長回覆: {reply}" if reply else f"{old_note} \n組長核可: {note_text}"
                        ws.update_cell(ridx, EXPECTED_COLUMNS.index("備註") + 1, new_note)
                        st.session_state.approved_morning_ids.add(str(record_id))
                        load_main_data.clear()
                        col_ref.success(f"✅ 已核可，學期加 {abs(s_val):g} 分")

                for i, r in pending_df.iterrows():
                    with st.container(border=True):
                        c1, c2, c3 = st.columns([2, 2, 1.3])

                        is_makeup = "補掃" in str(r["評分項目"])
                        title_badge = "🩹 **[補掃]**" if is_makeup else "🧹"

                        # ── 自動解析應到/實到人數，計算建議給分 ──
                        note_str = str(r.get("備註", ""))
                        inspector_str = str(r.get("檢查人員", ""))

                        # 從備註解析應到人數（格式：[應到:4人 實到:3人 ...]）
                        import re as _re
                        m_req = _re.search(r"應到:(\d+)人", note_str)
                        m_act = _re.search(r"實到:(\d+)人", note_str)
                        # fallback：從檢查人員欄位解析實到
                        if not m_act:
                            m_act = _re.search(r"實到:(\d+)", inspector_str)

                        n_required = int(m_req.group(1)) if m_req else None
                        n_actual   = int(m_act.group(1)) if m_act else None

                        # 計算建議給分：基礎分依應到人數，不足打折，補掃再打折
                        if n_required is not None and n_actual is not None:
                            base_score  = 2.0 if n_required >= 4 else 1.0
                            full_attend = n_actual >= n_required
                            attend_mult = 1.0 if full_attend else 0.5
                            makeup_mult = 0.5 if is_makeup else 1.0
                            suggested   = base_score * attend_mult * makeup_mult
                            score_label = (f"應到 {n_required} 人，實到 {n_actual} 人"
                                           f"{'（人數足夠）' if full_attend else '（人數不足）'}"
                                           f"{'，補掃' if is_makeup else ''}"
                                           f" → 建議給 **{suggested:g} 分**")
                            score_val = -suggested  # 負數代表學期加分
                        else:
                            suggested   = None
                            score_label = "⚠️ 無法自動解析人數，請手動判斷"
                            score_val   = None

                        c1.write(f"{title_badge} **{r['班級']}** | {inspector_str}")
                        c1.caption(f"登錄時間：{r['登錄時間']}")
                        c1.info(score_label)

                        if "http" in str(r['照片路徑']):
                            c2.image([p for p in str(r['照片路徑']).split(";") if "http" in p], width=150)

                        reply_msg = c1.text_input("💬 給予回應 (可留白)", key=f"rm_{r['紀錄ID']}_{i}")

                        # ── 審核按鈕：給分 or 駁回 ──
                        if score_val is not None:
                            if c3.button(f"✅ 給分 ({suggested:g}分)", key=f"approve_{r['紀錄ID']}_{i}"):
                                _do_approve(r["紀錄ID"], score_val,
                                            f"給分{suggested:g}分（{score_label.split('→')[0].strip()}）",
                                            reply_msg, c1, main_df)
                        else:
                            # 無法自動計算時，提供手動選項
                            manual_score = c3.selectbox("給分", [2, 1, 0.5, 0.25], key=f"manual_{r['紀錄ID']}_{i}")
                            if c3.button("✅ 給分", key=f"approve_{r['紀錄ID']}_{i}"):
                                _do_approve(r["紀錄ID"], -manual_score, f"手動給分 {manual_score} 分",
                                            reply_msg, c1, main_df)

                        if c3.button("🗑️ 駁回", key=f"r_{r['紀錄ID']}_{i}"):
                            ws = get_worksheet(SHEET_TABS["main"])
                            id_list = ws.col_values(EXPECTED_COLUMNS.index("紀錄ID") + 1)
                            # [Fix] 統一轉為 str 並 strip
                            id_list = [str(v).strip() for v in id_list]
                            rid_str = str(r["紀錄ID"]).strip()
                            if rid_str in id_list:
                                ridx = id_list.index(rid_str) + 1
                                ws.update_cell(ridx, EXPECTED_COLUMNS.index("評分項目") + 1, "晨間打掃(已駁回)")
                                old_note = str(r['備註'])
                                rej_msg  = reply_msg if reply_msg else "未達標準，請見諒"
                                new_note = f"{old_note} \n組長駁回: {rej_msg}"
                                ws.update_cell(ridx, EXPECTED_COLUMNS.index("備註") + 1, new_note)
                                st.session_state.approved_morning_ids.add(rid_str)
                                load_main_data.clear()
                                c1.error("🗑️ 已駁回")

                if not pending_df.empty or st.session_state.approved_morning_ids:
                    if st.button("🔄 審核完畢，重新整理列表"):
                        st.session_state.approved_morning_ids.clear()
                        load_main_data.clear()
                        st.rerun()

            with t_settings:
                st.subheader("⚙️ 系統設定與維護")
                curr = SYSTEM_CONFIG.get("semester_start")
                nd = st.date_input("開學日", datetime.strptime(curr, "%Y-%m-%d").date() if curr else today_tw)
                if st.button("更新開學日"): save_setting("semester_start", str(nd))
                
                st.markdown("---")
                st.write("📅 **週次手動對照表**（解決寒假跨週問題）")
                st.caption("只需填「錨點」：每個學期重置點的週一日期與週次號碼，用逗號分隔。\n\n例如：`2025-01-23:1,2025-02-23:2`\n\n這樣填即可：第1週從1/23起，第2週從2/23起，第3週之後系統會自動從2/23往後每7天累計，不需要填完所有週次。")
                curr_week_map = SYSTEM_CONFIG.get("week_map", "")
                new_week_map = st.text_area("週次對照表", value=curr_week_map, placeholder="2025-01-23:1,2025-02-23:2,2025-03-02:3,...")
                if st.button("💾 儲存週次對照表"):
                    if save_setting("week_map", new_week_map.strip()):
                        st.success("✅ 週次對照表已更新！")
                
                st.markdown("---")
                st.write("📢 晨掃志工每日廣播/任務")
                current_task = SYSTEM_CONFIG.get("daily_morning_task", "今日無特殊任務，請確實完成各區打掃即可！")
                new_task = st.text_area("請輸入想給志工看的話（例如：拍照請比 YA、今天請加強拖地等）", value=current_task, key="ta_morning_task")
                if st.button("💾 更新每日任務"):
                    if save_setting("daily_morning_task", new_task):
                        st.success("✅ 每日任務已更新！學生現在起會看到最新廣播。")

                st.markdown("---")

                st.write("📢 衛生糾察每日廣播/提醒")
                current_hygiene_task = SYSTEM_CONFIG.get("daily_hygiene_task", "今日無特殊任務，請確實完成各區檢查即可！")
                new_hygiene_task = st.text_area("請輸入想給糾察隊看的話（例如：今天重點檢查黑板、窗台）", value=current_hygiene_task, key="ta_hygiene_task")
                if st.button("💾 更新糾察任務"):
                    if save_setting("daily_hygiene_task", new_hygiene_task):
                        st.success("✅ 糾察任務已更新！糾察隊現在起會看到最新廣播。")

                st.markdown("---")
                st.write("🔧 系統連線狀態")
                if get_gspread_client(): st.success("✅ Google Sheets 連線正常")
                else: st.error("❌ Google Sheets 連線失敗")
                
                if NOTION_INSTALLED: st.success("✅ Notion 模組載入正常")
                else: st.warning("⚠️ 尚未安裝 Notion 模組")
                
                st.info("若需修改名單請直接至 Google Sheet 修改 inspectors / roster / office_areas 分頁")
                if st.button("🔄 重讀名單 (清除快取)"): st.cache_data.clear(); st.success("已清除快取！")

            with t3:
                st.subheader("🎖️ 服務時數發放")

                # ── 共用欄位 ──
                SVC_CATEGORIES = ["返校打掃", "校外服務", "社區服務", "班級義工", "其他（手動輸入）"]
                c_s1, c_s2 = st.columns(2)
                rd = c_s1.date_input("日期", today_tw, key="svc_date")
                sel_cat = c_s2.selectbox("活動類別", SVC_CATEGORIES, key="svc_cat")

                if sel_cat == "其他（手動輸入）":
                    custom_cat_input = st.text_input("請輸入活動名稱", key="svc_custom_cat", placeholder="例如：環境清潔日")
                    base_category = custom_cat_input.strip() if custom_cat_input.strip() else "其他"
                else:
                    base_category = sel_cat

                remark_input = st.text_input("備註（選填）", key="svc_remark", placeholder="例如：返校打掃第一梯次")
                # 備註塞入類別欄，格式：活動名稱｜備註內容（不改 Sheet 欄位結構）
                final_category = f"{base_category}｜{remark_input.strip()}" if remark_input.strip() else base_category

                st.markdown("---")
                target_mode = st.radio("發放對象模式", ["🏫 班級模式", "🔢 直接輸入學號"], horizontal=True, key="svc_mode")
                st.markdown("")

                # ── 班級模式 ──
                if target_mode == "🏫 班級模式":
                    rc = st.selectbox("選擇班級", all_classes, key="svc_cls")
                    mems = [s for s, c_val in ROSTER_DICT.items() if c_val == rc]

                    if not mems:
                        st.warning("⚠️ 此班級在 Roster 中找不到成員，請確認 Google Sheet。")
                    else:
                        with st.form("svc_class_form"):
                            absent = st.multiselect(f"❌ 缺席名單（共 {len(mems)} 人，扣除法）", mems, key="svc_absent")
                            pool = [m for m in mems if m not in absent]
                            st.caption(f"✅ 一般組：{len(pool)} 人")
                            base_h = st.number_input("基礎時數", value=2.0, step=0.5, min_value=0.0, key="svc_base_h")

                            st.markdown("---")
                            spec = st.multiselect("⭐ 加強組（從一般組挑選，另給不同時數）", pool, key="svc_spec")
                            spec_h = st.number_input("加強組時數", value=3.0, step=0.5, min_value=0.0, key="svc_spec_h")
                            st.caption("（若無加強組，此欄留空即可，不影響結果）")

                            pf = st.file_uploader("📸 照片（選填）", type=['jpg', 'png', 'jpeg'], key="svc_pf_cls")

                            if st.form_submit_button("🚀 發放"):
                                if time.time() - st.session_state.last_action_time < 3:
                                    st.warning("⚠️ 系統處理中，請勿連續點擊！")
                                elif not pool:
                                    st.error("❌ 發放名單為空，請確認名單設定！")
                                else:
                                    st.session_state.last_action_time = time.time()
                                    norm = [m for m in pool if m not in spec]
                                    ok_norm, ok_spec = True, True

                                    fb = None
                                    if pf:
                                        pf.seek(0)
                                        fb = pf.read()

                                    if norm:
                                        files_norm = [io.BytesIO(fb)] if fb else None
                                        if files_norm: files_norm[0].name = "p.jpg"
                                        ok_norm = save_entry(
                                            {"日期": str(rd), "班級": rc, "評分項目": "服務時數發放",
                                             "登錄時間": now_tw.strftime("%Y-%m-%d %H:%M:%S")},
                                            files_norm, norm, base_h, final_category
                                        )
                                    if spec:
                                        files_spec = [io.BytesIO(fb)] if fb else None
                                        if files_spec: files_spec[0].name = "p.jpg"
                                        ok_spec = save_entry(
                                            {"日期": str(rd), "班級": rc, "評分項目": "服務時數發放",
                                             "登錄時間": now_tw.strftime("%Y-%m-%d %H:%M:%S")},
                                            files_spec, spec, spec_h, f"{final_category}(加強)"
                                        )

                                    if ok_norm and ok_spec:
                                        result_msg = f"✅ 已發放！一般組 {len(norm)} 人 ({base_h}h)"
                                        if spec:
                                            result_msg += f"，加強組 {len(spec)} 人 ({spec_h}h)"
                                        st.success(result_msg)
                                        time.sleep(1.5)
                                        st.rerun()

                # ── 直接輸入學號模式 ──
                else:
                    raw_sid_input = st.text_area(
                        "輸入學號（每行一個，或用逗號分隔）",
                        key="svc_sid_raw",
                        placeholder="例如：\n112001\n112002, 112003",
                        height=150
                    )

                    # 即時解析與驗證（放在 form 外，輸入時即時顯示結果）
                    valid_ids, invalid_ids = [], []
                    if raw_sid_input.strip():
                        raw_ids_list = [s.strip() for s in re.split(r'[\n,、，\s]+', raw_sid_input) if s.strip()]
                        for sid in raw_ids_list:
                            cleaned_sid = clean_id(sid)
                            if cleaned_sid in ROSTER_DICT:
                                if cleaned_sid not in valid_ids:
                                    valid_ids.append(cleaned_sid)
                            else:
                                if sid not in invalid_ids:
                                    invalid_ids.append(sid)

                    if valid_ids:
                        st.success(f"✅ 有效學號 {len(valid_ids)} 人，通過驗證，將發放服務時數。")
                    if invalid_ids:
                        st.error(f"❌ 以下學號不在 Roster 名單中，請確認後再送出：**{', '.join(invalid_ids)}**")

                    with st.form("svc_sid_form"):
                        hours_sid = st.number_input("時數（全體統一）", value=1.0, step=0.5, min_value=0.0, key="svc_sid_h")
                        pf_sid = st.file_uploader("📸 照片（選填）", type=['jpg', 'png', 'jpeg'], key="svc_pf_sid")

                        if st.form_submit_button("🚀 發放"):
                            if time.time() - st.session_state.last_action_time < 3:
                                st.warning("⚠️ 系統處理中，請勿連續點擊！")
                            elif invalid_ids:
                                st.error(f"❌ 尚有 {len(invalid_ids)} 個無效學號，請先修正後再送出！")
                            elif not valid_ids:
                                st.error("❌ 請先在上方輸入至少一個有效學號！")
                            else:
                                st.session_state.last_action_time = time.time()

                                # 按班級分組，讓 service_hours 的班級欄位正確對應
                                cls_groups = {}
                                for sid in valid_ids:
                                    cls_name = ROSTER_DICT.get(sid, "")
                                    if cls_name not in cls_groups:
                                        cls_groups[cls_name] = []
                                    cls_groups[cls_name].append(sid)

                                fb_sid = None
                                if pf_sid:
                                    pf_sid.seek(0)
                                    fb_sid = pf_sid.read()

                                all_ok = True
                                for i, (cls_name, sids) in enumerate(cls_groups.items()):
                                    # 照片只在第一個班級群組帶入，避免重複上傳到 Drive
                                    files_sid = [io.BytesIO(fb_sid)] if (fb_sid and i == 0) else None
                                    if files_sid: files_sid[0].name = "p.jpg"
                                    ok = save_entry(
                                        {"日期": str(rd), "班級": cls_name, "評分項目": "服務時數發放",
                                         "登錄時間": now_tw.strftime("%Y-%m-%d %H:%M:%S")},
                                        files_sid, sids, hours_sid, final_category
                                    )
                                    if not ok:
                                        all_ok = False

                                if all_ok:
                                    st.success(f"✅ 已為 {len(valid_ids)} 位同學發放 {hours_sid}h 服務時數！")
                                    time.sleep(1.5)
                                    st.rerun()

            # [新增] 愛校服務 2.0：愛校與欠時管理 Tab
            with t_debt:
                st.subheader("🤝 愛校與欠時管理")

                # ── 區塊 A：📥 待驗收愛校任務 ──
                with st.expander("📥 待驗收愛校任務", expanded=True):
                    _claimed_tasks = fetch_claimed_notion_tasks()
                    if not _claimed_tasks:
                        st.success("🎉 目前沒有待驗收的任務！")
                    else:
                        for _ct in _claimed_tasks:
                            with st.container(border=True):
                                st.write(f"📌 **{_ct['title']}**")
                                st.caption(f"📅 日期：{_ct['date']}　|　認領學生：{', '.join(_ct['claimants'])}")
                                if st.button("✅ 驗收通過並扣除時數", key=f"verify_{_ct['id']}"):
                                    _hr_match = re.search(r"[\(\uff08]([\d.]+)\s*(?:hr|小時|h)[\)\uff09]", _ct["title"], re.IGNORECASE)
                                    _task_hours = float(_hr_match.group(1)) if _hr_match else 1.0
                                    _ok_count = 0
                                    _issued_sids = []  # ⭐️ 收集要發放時數的學號名單
                                    
                                    for _clm in _ct["claimants"]:
                                        _sid_match = re.match(r"(\d+)", _clm)
                                        if _sid_match:
                                            _sid_val = _sid_match.group(1)
                                            # 1. 扣除欠時
                                            if update_student_debt(_sid_val, -_task_hours, f"愛校驗收：{_ct['title']}"):
                                                _ok_count += 1
                                                _issued_sids.append(_sid_val)  
                                                
                                    # 2. ⭐️ 將名單送進背景佇列，自動發放實體的「服務時數」
                                    if _issued_sids:
                                        _payload = {
                                            "student_list": _issued_sids,
                                            "date": str(today_tw),
                                            "class_name": "愛校打掃", 
                                            "category": "返校打掃", 
                                            "hours": _task_hours
                                        }
                                        enqueue_task("service_hours_only", _payload)
                                                
                                    update_notion_task_status(_ct["id"], "任務已驗收")  # ⭐️ 配合新的 Notion 狀態
                                    st.success(f"✅ 已驗收！共扣除 {_ok_count} 位學生欠時，並已自動排程發放 {_task_hours} 小時服務時數！")
                                    time.sleep(2.0)
                                    st.rerun()

                # ── 區塊 B：⚠️ 欠時懲處結算報表 ──
                with st.expander("⚠️ 欠時懲處結算報表", expanded=True):
                    if st.button("🚀 結算滿 1 小時警告名單", key="debt_settle_btn"):
                        _all_debts = load_student_debts()
                        _warn_list = []
                        for _dsid, _dhours in _all_debts.items():
                            if _dhours >= 1.0:
                                _warnings = math.floor(_dhours)
                                _remaining = round(_dhours - _warnings, 2)
                                _cls_name = ROSTER_DICT.get(_dsid, "未知班級")
                                _warn_list.append({
                                    "學號": _dsid, "班級": _cls_name,
                                    "未完成時數": _dhours, "應記警告支數": _warnings,
                                    "結算後剩餘欠時": _remaining
                                })
                        if not _warn_list:
                            st.success("🎉 目前沒有學生達到記警告的門檻！")
                        else:
                            _warn_df = pd.DataFrame(_warn_list).sort_values(["班級", "學號"]).reset_index(drop=True)
                            st.warning(f"⚠️ 共有 **{len(_warn_df)}** 位學生達到記警告門檻：")
                            st.dataframe(_warn_df, hide_index=True)
                            _csv_data = _warn_df.to_csv(index=False).encode("utf-8-sig")
                            st.download_button(
                                label="📥 下載結算報表 (CSV)",
                                data=_csv_data,
                                file_name=f"欠時結算報表_{datetime.now(TW_TZ).strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                            _excel_buf = io.BytesIO()
                            with pd.ExcelWriter(_excel_buf, engine="openpyxl") as _ew:
                                _warn_df.to_excel(_ew, sheet_name="欠時結算", index=False)
                            _excel_buf.seek(0)
                            st.download_button(
                                label="📥 下載結算報表 (Excel)",
                                data=_excel_buf.getvalue(),
                                file_name=f"欠時結算報表_{datetime.now(TW_TZ).strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="debt_excel_dl"
                            )

        elif pwd_input != "":
            st.error("密碼錯誤")

except Exception as e:
    st.error(f"❌ 系統發生錯誤: {str(e)}")
    st.code(traceback.format_exc())
