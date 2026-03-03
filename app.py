import streamlit as st
import pandas as pd
import os
import smtplib
import time
import io
import traceback
import threading
import uuid
import re
import sqlite3
import json
import random
import concurrent.futures
from contextlib import closing
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, date, timedelta
from datetime import timezone
import pytz
import gspread
from oauth2client.service_account import ServiceAccountCredentials
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

# 👇 [抓蟲專用] 讓系統在側邊欄大聲說出它拿到的身分證是什麼！
import streamlit as st # 確保有載入
st.sidebar.info(f"🕵️‍♀️ 系統目前抓到的身分證是：[{sys_env}]")

if sys_env == "DEV":
    st.set_page_config(page_title="🔧測試版-中壢家商，衛愛而生", layout="wide", page_icon="🧹")
else:
    st.set_page_config(page_title="中壢家商，衛愛而生", layout="wide", page_icon="🧹")


# --- 2. 核心參數與全域設定 ---
try:
    TW_TZ = pytz.timezone('Asia/Taipei')
    MAX_IMAGE_BYTES = 20 * 1024 * 1024  
    UPLOAD_SEM = threading.BoundedSemaphore(4) 
    QUEUE_DB_PATH = "task_queue_v4_wal.db"
    IMG_DIR = "evidence_photos"
    os.makedirs(IMG_DIR, exist_ok=True)
    
    SHEET_URL = "https://docs.google.com/spreadsheets/d/11BXtN3aevJls6Q2IR_IbT80-9XvhBkjbTCgANmsxqkg/edit"
    SHEET_TABS = {
        "main": "main_data", "settings": "settings", "roster": "roster",
        "inspectors": "inspectors", "duty": "duty",
        "appeals": "appeals", "holidays": "holidays", "service_hours": "service_hours",
        "office_areas": "office_areas"
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
                filter={"property": "任務狀態", "status": {"equals": "等待認領中😿"}}
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

    def claim_notion_task(page_id, student_id):
        client = get_notion_client()
        try:
            page = client.pages.retrieve(page_id=page_id)
            props = page.get("properties", {})
            
            req_num_obj = props.get("需求人數", {}).get("number")
            req_num = req_num_obj if req_num_obj else 1
            
            claimed_obj = props.get("認領學號", {}).get("rich_text", [])
            claimed_str = claimed_obj[0].get("text", {}).get("content", "") if claimed_obj else ""
            current_claimants = [s.strip() for s in claimed_str.split(",") if s.strip()]
            
            if str(student_id) in current_claimants:
                return False, f"學號 {student_id} 已經認領過此任務囉！"
                
            current_claimants.append(str(student_id))
            new_claimed_str = ", ".join(current_claimants)
            
            is_full = len(current_claimants) >= req_num
            update_props = {
                "認領學號": {"rich_text": [{"text": {"content": new_claimed_str}}]}
            }
            if is_full:
                update_props["任務狀態"] = {"status": {"name": "已認領"}}

            client.pages.update(
                page_id=page_id,
                properties=update_props
            )
            return True, "滿團" if is_full else "未滿"
            
        except Exception as e:
            return False, str(e)

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
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        if "gcp_service_account" not in st.secrets:
            return None
        return ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)

    def get_gspread_client():
        creds = get_credentials()
        return gspread.authorize(creds) if creds else None

    def get_drive_service():
        creds = get_credentials()
        return build('drive', 'v3', credentials=creds, cache_discovery=False) if creds else None

    def get_worksheet(tab_name):
        client = get_gspread_client()
        if not client: return None
        for attempt in range(4):
            try:
                sheet = client.open_by_url(SHEET_URL)
                try: 
                    return sheet.worksheet(tab_name)
                except gspread.WorksheetNotFound:
                    cols = 20 if tab_name != "appeals" else 15
                    ws = sheet.add_worksheet(title=tab_name, rows=500, cols=cols)
                    if tab_name == "appeals": ws.append_row(APPEAL_COLUMNS)
                    if tab_name == "service_hours": ws.append_row(["日期", "學號", "班級", "類別", "時數", "紀錄ID"])
                    if tab_name == "holidays": ws.append_row(["日期", "說明"])
                    if tab_name == "office_areas": ws.append_row(["區域名稱", "負責班級"])
                    return ws
            except Exception as e:
                if "429" in str(e): 
                    time.sleep(2 * (attempt + 1) + random.uniform(0, 1))
                    continue
                else: return None
        return None

    def compress_image_bytes(file_bytes, quality=70):
        try:
            img = Image.open(io.BytesIO(file_bytes))
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB": img = img.convert("RGB")
            if img.width > 1600:
                ratio = 1600 / float(img.width)
                img = img.resize((1600, int(img.height * ratio)), Image.Resampling.LANCZOS)
            out_buffer = io.BytesIO()
            img.save(out_buffer, format="JPEG", quality=quality, optimize=True)
            out_buffer.seek(0)
            return out_buffer
        except: return io.BytesIO(file_bytes)

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
            except: pass 
            return f"https://drive.google.com/thumbnail?id={file.get('id')}&sz=w1000"
        return execute_with_retry(_upload_action)

    def clean_id(val):
        try: return str(int(float(val))).strip()
        except: return str(val).strip()

    # ==========================================
    # SQLite 背景佇列 (V5.8: Two-stage Throttling & Last Success Time)
    # ==========================================
    def open_queue_conn():
        conn = sqlite3.connect(QUEUE_DB_PATH, timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.execute("CREATE TABLE IF NOT EXISTS task_queue (id TEXT PRIMARY KEY, task_type TEXT, created_ts TEXT, payload_json TEXT, status TEXT, attempts INTEGER, last_error TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS service_issued (date TEXT, sid TEXT, category TEXT, PRIMARY KEY(date, sid, category))")
        conn.execute("CREATE TABLE IF NOT EXISTS system_status (key TEXT PRIMARY KEY, val TEXT)") 
        return conn

    def get_pending_count():
        try:
            with closing(open_queue_conn()) as conn:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM task_queue WHERE status='PENDING'")
                return cur.fetchone()[0]
        except: return 0

    def update_worker_heartbeat():
        try:
            with closing(open_queue_conn()) as conn:
                # 改用最單純的 time.time() 記錄秒數，避免雲端解析字串失敗
                conn.execute("INSERT OR REPLACE INTO system_status VALUES ('worker_heartbeat', ?)", (str(time.time()),))
        except: pass

    # [V5.8] 新增紀錄最後成功處理時間
    def update_last_success_time():
        try:
            with closing(open_queue_conn()) as conn:
                conn.execute("INSERT OR REPLACE INTO system_status VALUES ('last_success_time', ?)", (str(time.time()),))
        except: pass

    def get_worker_heartbeat_sec():
        try:
            with closing(open_queue_conn()) as conn:
                cur = conn.cursor()
                cur.execute("SELECT val FROM system_status WHERE key='worker_heartbeat'")
                row = cur.fetchone()
                if row:
                    return time.time() - float(row[0])
        except: pass
        return 999999

    # [V5.8] 獲取最後成功處理任務至今的秒數
    def get_last_success_sec():
        try:
            with closing(open_queue_conn()) as conn:
                cur = conn.cursor()
                cur.execute("SELECT val FROM system_status WHERE key='last_success_time'")
                row = cur.fetchone()
                if row:
                    return time.time() - float(row[0])
        except: pass
        return 999999

    def enqueue_task(task_type, payload):
        task_id = str(uuid.uuid4())
        with closing(open_queue_conn()) as conn:
            conn.execute(
                "INSERT INTO task_queue VALUES (?, ?, ?, ?, 'PENDING', 0, NULL)",
                (task_id, task_type, datetime.now(timezone.utc).isoformat(), json.dumps(payload, ensure_ascii=False))
            )
        return task_id

    def get_queue_metrics():
        metrics = {"pending": 0, "retry": 0, "failed": 0, "oldest_pending_sec": 0, "recent_errors": []}
        try:
            with closing(open_queue_conn()) as conn:
                cur = conn.cursor()
                cur.execute("SELECT status, COUNT(*) FROM task_queue GROUP BY status")
                for s, c in cur.fetchall():
                    if s == 'PENDING': metrics["pending"] = c
                    elif s == 'RETRY': metrics["retry"] = c
                    elif s == 'FAILED': metrics["failed"] = c
                
                cur.execute("SELECT MIN(created_ts) FROM task_queue WHERE status IN ('PENDING', 'RETRY')")
                oldest = cur.fetchone()[0]
                if oldest:
                    try: metrics["oldest_pending_sec"] = (datetime.now(pytz.utc) - datetime.fromisoformat(oldest.replace("Z", "+00:00"))).total_seconds()
                    except: pass
                cur.execute("SELECT last_error, created_ts FROM task_queue WHERE status='FAILED' OR status='RETRY' ORDER BY created_ts DESC LIMIT 5")
                metrics["recent_errors"] = cur.fetchall()
        except: pass
        return metrics

    def fetch_next_task(max_attempts=6):
        try:
            with closing(open_queue_conn()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                cur = conn.cursor()
                cur.execute("""
                    SELECT id, task_type, payload_json, attempts 
                    FROM task_queue 
                    WHERE status IN ('PENDING', 'RETRY') AND attempts < ?
                    ORDER BY created_ts ASC LIMIT 1
                """, (max_attempts,))
                
                row = cur.fetchone()
                if not row:
                    conn.execute("COMMIT")
                    return None
                    
                task_id = row[0]
                cur.execute("UPDATE task_queue SET status='IN_PROGRESS', attempts=attempts+1 WHERE id=?", (task_id,))
                conn.execute("COMMIT")
                
                return {"id": task_id, "task_type": row[1], "payload": json.loads(row[2] or "{}"), "attempts": row[3]}
        except Exception as e:
            print(f"抓取任務時發生錯誤: {e}")
            return None

    def update_task_status(task_id, status, attempts, last_error):
        with closing(open_queue_conn()) as conn:
            conn.execute(
                "UPDATE task_queue SET status=?, attempts=?, last_error=? WHERE id=?",
                (status, attempts, last_error, task_id)
            )

    # ==========================================
    # 背景處理邏輯
    # ==========================================
    def _append_main_entry_row(entry):
        def _action():
            ws = get_worksheet(SHEET_TABS["main"])
            if not ws: return
            row = [str(entry.get(col, "")).upper() if isinstance(entry.get(col, ""), bool) else str(entry.get(col, "")) for col in EXPECTED_COLUMNS]
            ws.append_row(row)
        execute_with_retry(_action)
    
    def _append_service_row_unique(entry):
        t_date = str(entry.get("日期", ""))
        t_sid = str(entry.get("學號", ""))
        t_cat = str(entry.get("類別", ""))
        
        try:
            with closing(open_queue_conn()) as conn:
                conn.execute("INSERT INTO service_issued VALUES (?, ?, ?)", (t_date, t_sid, t_cat))
        except sqlite3.IntegrityError:
            return 
            
        def _action():
            ws = get_worksheet(SHEET_TABS["service_hours"])
            if not ws: return
            new_row = [t_date, t_sid, str(entry.get("班級", "")), t_cat, str(entry.get("時數", "")), str(entry.get("紀錄ID", ""))]
            ws.append_row(new_row)
        execute_with_retry(_action)

    # [V5.9 Patch] 實作 Dry Run 與 Error Summary
    def update_last_error_summary(err_msg):
        try:
            with closing(open_queue_conn()) as conn:
                short_msg = str(err_msg)[:120]
                conn.execute("INSERT OR REPLACE INTO system_status VALUES ('last_error_summary', ?)", (short_msg,))
        except: pass

    def get_last_error_summary():
        try:
            with closing(open_queue_conn()) as conn:
                cur = conn.cursor()
                cur.execute("SELECT val FROM system_status WHERE key='last_error_summary'")
                row = cur.fetchone()
                return row[0] if row else "無紀錄"
        except: return "無紀錄"

    def process_task(task):
        task_type, payload = task["task_type"], task["payload"]
        
        # 1. 檢查是否開啟「演習模式 (Dry Run)」
        is_dry_run = str(st.secrets.get("system_config", {}).get("dry_run", "false")).lower() in ["true", "1"]
        if is_dry_run:
            # 演習模式：假裝很忙，暫停 0.3~0.6 秒，然後直接回報成功
            time.sleep(random.uniform(0.3, 0.6))
            return True, "DRY_RUN_SUCCESS"

        # 2. 正常執行邏輯 (以下保留原有邏輯不變)
        if task_type == "service_hours_only":
            try:
                for sid in payload.get("student_list", []):
                    log_entry = {
                        "日期": payload.get("date", str(date.today())), "學號": sid,
                        "班級": payload.get("class_name", ""), "類別": payload.get("category", ""), 
                        "時數": payload.get("hours", 0.5), "紀錄ID": uuid.uuid4().hex[:8]
                    }
                    _append_service_row_unique(log_entry) 
                return True, None
            except Exception as e: return False, str(e)

        entry = payload.get("entry", {})
        try:
            image_paths, filenames, drive_links = payload.get("image_paths", []), payload.get("filenames", []), []
            for path in image_paths:
                if path and not os.path.exists(path): return False, "FILE_NOT_FOUND: 找不到實體圖片檔案，直接放棄"
            for path, fname in zip(image_paths, filenames):
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        drive_links.append(upload_image_to_drive(compress_image_bytes(f.read()), fname) or "UPLOAD_FAILED_API")
            if drive_links: entry["照片路徑"] = ";".join(drive_links)

            if task_type in ["main_entry", "volunteer_report"]:
                _append_main_entry_row(entry)
                inspector_name = entry.get("檢查人員", "")
                if "學號:" in inspector_name:
                    sid = inspector_name.split("學號:")[1].strip()
                    _append_service_row_unique({"日期": entry.get("日期"), "學號": sid, "班級": "", "類別": "整潔評分糾察", "時數": 0.25, "紀錄ID": uuid.uuid4().hex[:8]}) 
                
                if task_type == "volunteer_report":
                    for sid in payload.get("student_list", []):
                        _append_service_row_unique({"日期": entry.get("日期", str(date.today())), "學號": sid, "班級": entry.get("班級", ""), "類別": payload.get("custom_category", "晨掃志工"), "時數": payload.get("custom_hours", 0.5), "紀錄ID": uuid.uuid4().hex[:8]})

            elif task_type == "appeal_entry":
                image_info = payload.get("image_file")
                if image_info:
                    if not os.path.exists(image_info["path"]): return False, "FILE_NOT_FOUND: 找不到佐證照片檔案，直接放棄"
                    with open(image_info["path"], "rb") as f: entry["佐證照片"] = upload_image_to_drive(compress_image_bytes(f.read()), image_info["filename"])
                execute_with_retry(lambda: get_worksheet(SHEET_TABS["appeals"]).append_row([str(entry.get(c, "")) for c in APPEAL_COLUMNS]))
            return True, None
        except Exception as e: return False, str(e)

    def background_worker(stop_event=None):
        try: add_script_run_ctx(threading.current_thread(), get_script_run_ctx())
        except: pass
        while True:
            if stop_event and stop_event.is_set(): break
            update_worker_heartbeat() 
            try:
                task = fetch_next_task()
                if not task: time.sleep(2.0); continue
                ok, err = process_task(task)
                
                if ok: update_last_success_time()
                else: 
                    # 紀錄最後一次失敗原因
                    if err and "DRY_RUN" not in err: update_last_error_summary(err)

                try:
                    paths = task["payload"].get("image_paths", []) + ([task["payload"]["image_file"]["path"]] if "image_file" in task["payload"] else [])
                    for p in paths:
                        if p and os.path.exists(p): os.remove(p)
                except: pass

                if not ok and err and "FILE_NOT_FOUND" in str(err): task["attempts"] = 999
                update_task_status(task["id"], "DONE" if ok else ("FAILED" if task["attempts"] >= 6 else "RETRY"), task["attempts"], err)
                time.sleep(0.5)
            except Exception as e: time.sleep(3.0)

    # [V5.10 Patch] 修正 Streamlit 新版快取機制的潛在衝突
    @st.cache_resource
    def ensure_worker_started():
        stop_event = threading.Event()
        t = threading.Thread(target=background_worker, args=(stop_event,), daemon=True)
        add_script_run_ctx(t)
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
        except: return []

    def is_within_appeal_period(violation_date, appeal_days=3):
        vd = pd.to_datetime(violation_date).date() if isinstance(violation_date, str) else violation_date
        holidays, today, current_date, workdays = load_holidays(), date.today(), vd, 0
        for _ in range(14): 
            if workdays >= appeal_days: break
            current_date += timedelta(days=1)
            if current_date.weekday() < 5 and current_date not in holidays: workdays += 1
        return today <= current_date

    # [V5.8] TTL 延長至 360 秒，有效節省 API 呼叫次數
    @st.cache_data(ttl=360)
    def load_main_data():
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
        except: return pd.DataFrame(columns=EXPECTED_COLUMNS)

    @st.cache_data(ttl=21600)
    def load_roster_dict():
        ws = get_worksheet(SHEET_TABS["roster"])
        if not ws: return {}
        try:
            df = pd.DataFrame(ws.get_all_records())
            id_c, cls_c = next((c for c in df.columns if "學號" in c), None), next((c for c in df.columns if "班級" in c), None)
            return {clean_id(row[id_c]): str(row[cls_c]).strip() for _, row in df.iterrows()} if id_c and cls_c else {}
        except: return {}
    
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
            def get_sort_key(n):
                g = 1 if "一" in n or "1" in n else (2 if "二" in n or "2" in n else (3 if "三" in n or "3" in n else 99))
                return (g, next((v for k, v in dept_order.items() if k in n), 99), n)
            sorted_all = sorted(unique, key=get_sort_key)
            return sorted_all, [{"grade": f"{get_sort_key(c)[0]}年級" if get_sort_key(c)[0]!=99 else "其他", "name": c} for c in sorted_all]
        except: return [], []

    @st.cache_data(ttl=60)
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
        except: return pd.DataFrame(), "error"

    @st.cache_data(ttl=3600)
    def load_office_area_map():
        ws = get_worksheet(SHEET_TABS["office_areas"])
        if not ws: return {}
        try: return {str(r.get("區域名稱", "")).strip(): str(r.get("負責班級", "")).strip() for r in ws.get_all_records() if str(r.get("區域名稱", "")).strip()}
        except: return {}

    @st.cache_data(ttl=21600)
    def load_settings():
        ws = get_worksheet(SHEET_TABS["settings"])
        config = {"semester_start": "2025-08-25", "standard_n": 4}
        if ws:
            try:
                for row in ws.get_all_values():
                    if len(row)>=2: config[row[0]] = int(row[1]) if row[0] == "standard_n" else row[1]
            except: pass
        return config

    def save_setting(key, val):
        ws = get_worksheet(SHEET_TABS["settings"])
        if ws:
            try:
                cell = ws.find(key)
                if cell: ws.update_cell(cell.row, cell.col+1, val)
                else: ws.append_row([key, val])
                st.cache_data.clear(); return True
            except: return False
        return False

    @st.cache_data(ttl=60)
    def load_appeals():
        ws = get_worksheet(SHEET_TABS["appeals"])
        if not ws: return pd.DataFrame(columns=APPEAL_COLUMNS)
        try:
            df = pd.DataFrame(ws.get_all_records())
            for col in APPEAL_COLUMNS:
                if col not in df.columns: df[col] = "待處理" if col == "處理狀態" else ""
            return df[APPEAL_COLUMNS]
        except: return pd.DataFrame(columns=APPEAL_COLUMNS)

    def save_appeal(entry, proof_file=None):
        pending_count = get_pending_count()
        # [V5.8] 兩段式熔斷：超過 300 阻擋，超過 150 警告但放行
        if pending_count > 300:
            st.error("⚠️ 系統目前排隊任務過多 (大於 300 筆)，為保護系統，請稍等幾分鐘後再送出申訴！")
            return False
        elif pending_count > 150:
            st.warning("⏳ 系統目前正在大量處理照片中，您的申訴將會安全送出，但審核進度可能會稍有延遲喔！")

        image_info = None
        if proof_file:
            try:
                print(f"Waiting for UPLOAD slot... (Appeal: {proof_file.name})")
                with UPLOAD_SEM:
                    print(f"UPLOAD slot acquired (Appeal: {proof_file.name})")
                    data = proof_file.read()
                    if len(data) > MAX_IMAGE_BYTES: st.error("照片過大"); return False
                    fname = f"Appeal_{entry.get('班級', '')}_{datetime.now(TW_TZ).strftime('%H%M%S')}.jpg"
                    l_path = os.path.join(IMG_DIR, f"{datetime.now(TW_TZ).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}_{fname}")
                    with open(l_path, "wb") as f: f.write(data)
                    image_info = {"path": l_path, "filename": fname}
            except Exception as e: st.error(f"寫入失敗: {e}"); return False

        entry.update({"申訴日期": entry.get("申訴日期", datetime.now(TW_TZ).strftime("%Y-%m-%d")), "處理狀態": entry.get("處理狀態", "待處理"),
                      "登錄時間": entry.get("登錄時間", datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")), 
                      "申訴ID": entry.get("申訴ID", datetime.now(TW_TZ).strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:4]),
                      "佐證照片": entry.get("佐證照片", "")})
        enqueue_task("appeal_entry", {"entry": entry, "image_file": image_info})
        st.success("📩 申訴已排入背景處理")
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
        except: return default

    def check_duplicate_record(df, check_date, inspector, role, target_class=None):
        if df.empty: return False
        try:
            mask = (df["日期"].astype(str) == str(check_date)) & (df["檢查人員"] == inspector) & (df["評分項目"] == role)
            if target_class: mask &= (df["班級"] == target_class)
            return not df[mask].empty
        except: return False

    def save_entry(new_entry, uploaded_files=None, student_list=None, custom_hours=0.5, custom_category="晨掃志工"):
        pending_count = get_pending_count()
        # [V5.8] 兩段式熔斷保護
        if pending_count > 300:
            st.error("⚠️ 系統目前排隊上傳的任務過多 (大於 300 筆)，為保護系統，請稍等幾分鐘後再送出！")
            return False
        elif pending_count > 150:
            st.warning("⏳ 系統目前正在大量處理照片中，您的資料將會安全送出，但更新至成績表可能會稍有延遲喔！")

        new_entry["日期"] = str(new_entry.get("日期", str(date.today())))
        new_entry["紀錄ID"] = new_entry.get("紀錄ID", f"{datetime.now(TW_TZ).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}")
        if "登錄時間" not in new_entry or not new_entry["登錄時間"]:
            new_entry["登錄時間"] = datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")

        image_paths, file_names = [], []
        if uploaded_files:
            for i, up_file in enumerate(uploaded_files):
                if not up_file: continue
                try:
                    print(f"Waiting for UPLOAD slot... ({up_file.name})")
                    with UPLOAD_SEM:
                        print(f"UPLOAD slot acquired ({up_file.name})")
                        data = up_file.getvalue()
                        if len(data) > MAX_IMAGE_BYTES: st.warning(f"檔案略過 (過大): {up_file.name}"); continue
                        fname = f"{new_entry['紀錄ID']}_{i}.jpg"
                        local_path = os.path.join(IMG_DIR, fname)
                        with open(local_path, "wb") as f: f.write(data)
                        image_paths.append(local_path); file_names.append(fname)
                except Exception as e: print(f"Save Error: {e}")

        payload = {
            "entry": new_entry, "image_paths": image_paths, "filenames": file_names,
            "student_list": student_list or [], "custom_hours": custom_hours, "custom_category": custom_category
        }
        enqueue_task("volunteer_report" if student_list is not None else "main_entry", payload)
        return True

    # [V5.8] TTL 延長至 360 秒
    @st.cache_data(ttl=360)
    def load_full_semester_data_for_export():
        ws = get_worksheet(SHEET_TABS["main"])
        if not ws: return pd.DataFrame(columns=EXPECTED_COLUMNS)
        try:
            df = pd.DataFrame(ws.get_all_records())
            if df.empty: return pd.DataFrame(columns=EXPECTED_COLUMNS)
            for col in EXPECTED_COLUMNS:
                if col not in df.columns: df[col] = ""
            for col in ["備註", "違 ক্যাম", "班級", "檢查人員", "修正", "晨掃未到者", "照片路徑", "紀錄ID"]:
                if col in df.columns: df[col] = df[col].fillna("").astype(str)
            for col in ["內掃原始分", "外掃原始分", "垃圾原始分", "垃圾內掃原始分", "垃圾外掃原始分", "晨間打掃原始分", "手機人數", "週次"]:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            return df[EXPECTED_COLUMNS]
        except: return pd.DataFrame()

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
            start = datetime.strptime(SYSTEM_CONFIG["semester_start"], "%Y-%m-%d").date()
            if isinstance(d, datetime): d = d.date()
            return max(0, ((d - start).days // 7) + 1)
        except: return 0

    st.sidebar.title("🏫 功能選單")
    
    menu_options = ["糾察底家👀", "班級負責人🥸", "晨掃志工隊🧹", "愛校任務認領 🤝", "組長ㄉ窩💃"]
    app_mode = st.sidebar.radio("請選擇模式", menu_options)

    # --- Mode: 愛校任務認領 🤝 ---
    if app_mode == "愛校任務認領 🤝":
        st.title("🤝 愛校服務認領區")
        st.info("💡 這裡的任務清單與 Notion 行事曆即時同步！成功認領後，任務會自動標記並更新。")
        
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
                                s_id = st.text_input("請輸入您的【學號】來認領：", placeholder="例如：112001")
                                if st.form_submit_button("🚀 確認認領", use_container_width=True):
                                    if time.time() - st.session_state.last_action_time < 3:
                                        st.warning("⚠️ 系統處理中，請勿連續點擊！")
                                    elif not s_id:
                                        st.error("學號不能為空！")
                                    else:
                                        st.session_state.last_action_time = time.time()
                                        with st.spinner("連線至 Notion 更新看板中..."):
                                            success, msg = claim_notion_task(t['id'], s_id)
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

        # [V5.22 Patch 2] 衛生糾察專屬：藍色公仔大聲公
        daily_hygiene = SYSTEM_CONFIG.get("daily_hygiene_task", "")
        if daily_hygiene:
            formatted_hygiene = daily_hygiene.replace('\n', '<br>')
            mascot_url = "https://drive.google.com/thumbnail?id=128ITPXtpGNuI-wLIt6p-qd4ZNNhCGbhd" 
            
            # 這裡把對話框改成代表糾察隊的「水藍色系」，用 class 名稱區隔避免衝突
            bubble_html_h = f"""
            <style>
            .mascot-container-h {{ display: flex; align-items: flex-start; margin-bottom: 20px; gap: 15px; }}
            .mascot-img-h {{ width: 200px; flex-shrink: 0; }}
            .speech-bubble-h {{
                position: relative; background: #D0E8F2; /* 淡淡的水藍色 */
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
                    <strong>📢 組長廣播 / 糾察重點：</strong><br><br>
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
                            
                            row_data = {"處室/區域": off_name, "負責班級": cls_name, "未做好分類": is_sort_bad}
                            if step_a != "一般垃圾": row_data["未倒垃圾"] = is_dump_bad
                            rows.append(row_data)
                            
                        col_config = {"處室/區域": st.column_config.TextColumn(disabled=True), "負責班級": st.column_config.TextColumn(disabled=True)}
                        if step_a != "一般垃圾": col_config["未倒垃圾"] = st.column_config.CheckboxColumn("🗑️ 未倒垃圾", help="扣1分")
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
                                        if save_entry({**base, "備註": f"外掃({off})-{step_a}({','.join(v_list)})", "違規細項": step_a}):
                                            cnt += 1
                                if cnt: st.success(f"✅ 已登記 {cnt} 筆違規！"); time.sleep(1.5); st.rerun()

                    else:
                        for cls_name in [c["name"] for c in structured_classes if c["grade"] == sel_filter]:
                            cls_rec = today_records[today_records["班級"] == cls_name] if not today_records.empty else pd.DataFrame()
                            
                            is_dump_bad = any("內掃" in str(r["備註"]) and "未倒垃圾" in str(r["備註"]) for _, r in cls_rec.iterrows()) if not cls_rec.empty else False
                            is_sort_bad = any("內掃" in str(r["備註"]) and "未做好分類" in str(r["備註"]) for _, r in cls_rec.iterrows()) if not cls_rec.empty else False
                            
                            row_data = {"班級": cls_name, "未做好分類": is_sort_bad}
                            if step_a != "一般垃圾": row_data["未倒垃圾"] = is_dump_bad
                            rows.append(row_data)
                            
                        col_config = {"班級": st.column_config.TextColumn(disabled=True)}
                        if step_a != "一般垃圾": col_config["未倒垃圾"] = st.column_config.CheckboxColumn("🗑️ 未倒垃圾", help="扣1分")
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
                                        if save_entry({**base, "備註": f"內掃-{step_a}({','.join(v_list)})", "違規細項": step_a}):
                                            cnt += 1
                                if cnt: st.success(f"✅ 已登記 {cnt} 筆違規！"); time.sleep(1.5); st.rerun()

                else:
                    assigned_classes = curr_inspector.get("assigned_classes", [])
                    if assigned_classes:
                        sel_cls = st.radio("選擇負責班級", assigned_classes, key="m1_cls_assigned")
                    else:
                        temp_g = st.radio("步驟 A: 選擇年級", grades, horizontal=True, key="m1_grade_select")
                        f_cls_list = [c["name"] for c in structured_classes if c["grade"] == temp_g]
                        sel_cls = st.radio("步驟 B: 選擇班級", f_cls_list, horizontal=True, key="m1_cls_select") if f_cls_list else None

                    if sel_cls:
                        st.divider()
                        if check_duplicate_record(main_df, input_date, inspector_name, role, sel_cls): st.warning(f"⚠️ 今日已評過 {sel_cls}！")
                        with st.form("score_form", clear_on_submit=True):
                            in_s, out_s, ph_c, note = 0, 0, 0, ""
                            if st.radio("檢查結果", ["❌ 違規", "✨ 乾淨"], horizontal=True) == "❌ 違規":
                                if role == "內掃檢查":
                                    in_s = st.number_input("內掃扣分", 0)
                                    note = " ".join([x for x in [st.selectbox("區塊", ["", "走廊", "黑板", "地板"]), st.selectbox("狀況", ["", "髒亂", "沒拖地"]), st.text_input("補充")] if x])
                                else:
                                    out_s = st.number_input("外掃扣分", 0)
                                    note = " ".join([x for x in [st.selectbox("區域", ["", "走廊", "樓梯", "廁所", "操場"]), st.selectbox("狀況", ["", "很髒", "沒掃"]), st.text_input("補充")] if x])
                            is_fix = st.checkbox("🚩 這是修正單")
                            files = st.file_uploader("📸 違規照片", accept_multiple_files=True)
                            
                            if st.form_submit_button("送出"):
                                if time.time() - st.session_state.last_action_time < 3:
                                    st.warning("⚠️ 系統處理中，請勿連續點擊！")
                                else:
                                    st.session_state.last_action_time = time.time()
                                    if (in_s + out_s) > 0 and not files: 
                                        st.error("扣分需照片")
                                    else:
                                        if save_entry({"日期": input_date, "週次": week_num, "檢查人員": inspector_name, "登錄時間": now_tw.strftime("%Y-%m-%d %H:%M:%S"), "修正": is_fix, "班級": sel_cls, "評分項目": role, "內掃原始分": in_s, "外掃原始分": out_s, "手機人數": ph_c, "備註": note}, uploaded_files=files):
                                            st.success("✅ 送出成功！系統將自動排程發放本日 0.25 小時。")
                                            time.sleep(1.5)
                                            st.rerun()

    # --- Mode 2: 班級負責人 ---
    elif app_mode == "班級負責人🥸":
        st.title("🔎 班級成績查詢")
        df, appeals_df = load_main_data(), load_appeals()
        appeal_map = {str(r.get("對應紀錄ID")): {"status": str(r.get("處理狀態", "")), "reply": str(r.get("審核回覆", ""))} for _, r in appeals_df.iterrows()} if not appeals_df.empty else {}
        
        sel_grade_m2 = st.radio("選擇年級", grades, horizontal=True, key="m2_grade_select")
        cls_opts = [c["name"] for c in structured_classes if c["grade"] == sel_grade_m2]
        
        if cls_opts:
            cls = st.selectbox("選擇班級", cls_opts, key="m2_cls_select")
            if cls and not df.empty:
                for idx, r in df[df["班級"] == cls].sort_values("登錄時間", ascending=False).iterrows():
                    trash_score = r['垃圾內掃原始分'] + r['垃圾外掃原始分']
                    if trash_score == 0: trash_score = r['垃圾原始分']
                    
                    tot = r['內掃原始分'] + r['外掃原始分'] + trash_score + r['晨間打掃原始分']
                    rid = str(r['紀錄ID'])
                    ap_info = appeal_map.get(rid, {})
                    ap_st = ap_info.get("status")
                    ap_reply = ap_info.get("reply")
                    
                    icon = "✅" if ap_st=="已核可" else "🚫" if ap_st=="已駁回" else "⏳" if ap_st=="待處理" else "🛠️" if str(r['修正'])=="TRUE" else ""
                    
                    disp_time = str(r.get('登錄時間', ''))
                    time_str = disp_time.split(' ')[-1] if disp_time else ''
                    
                    # [V5.15] 判斷如果是負分(學期加分)，顯示成「加分」
                    score_disp = f"加 {abs(tot)} 分 (學期)" if tot < 0 else f"扣: {tot}"
                    
                    with st.expander(f"{icon} {r['日期']} {time_str} - {r['評分項目']} ({score_disp})"):
                        st.caption(f"登錄時間：{disp_time if disp_time else '未紀錄'}") 
                        st.write(f"🧑‍✈️ **評分人員:** {r.get('檢查人員', '未知')}")
                        st.write(f"📝 **備註:** {r['備註']}")
                        
                        if ap_st:
                            if ap_st == "待處理": st.info("⏳ 申訴審核中...")
                            elif ap_st == "已核可": st.success(f"✅ 申訴成功。組長回覆: {ap_reply if ap_reply else '無'}")
                            elif ap_st == "已駁回": st.error(f"🚫 申訴駁回。組長回覆: {ap_reply if ap_reply else '無'}")
                            
                        if str(r['照片路徑']) and "http" in str(r['照片路徑']): st.image([p for p in str(r['照片路徑']).split(";") if "http" in p], width=200)
                        if not ap_st and is_within_appeal_period(r['日期']) and (tot > 0 or r['手機人數'] > 0):
                            with st.form(f"ap_{rid}"):
                                rsn, pf = st.text_area("理由"), st.file_uploader("佐證", type=['jpg','png'])
                                if st.form_submit_button("申訴"):
                                    if time.time() - st.session_state.last_action_time < 3:
                                        st.warning("⚠️ 系統處理中，請勿連續點擊！")
                                    elif rsn and pf:
                                        st.session_state.last_action_time = time.time()
                                        if save_appeal({"班級": cls, "違規日期": str(r["日期"]), "違規項目": r['評分項目'], "原始扣分": str(tot), "申訴理由": rsn, "對應紀錄ID": rid}, pf):
                                            time.sleep(1.5)
                                            st.rerun()
                                    else:
                                        st.error("請填寫理由並上傳照片")

    # --- Mode 3: 晨掃志工隊🧹 ---
    elif app_mode == "晨掃志工隊🧹":
        st.title("🧹 晨掃志工回報專區")
        
        # [V5.25 Patch] 測試機特權：DEV 環境設定為 24 點關門，PROD 正式環境維持 16 點關門
        cutoff_hour = 24 if sys_env == "DEV" else 16
        
        if now_tw.hour >= cutoff_hour: 
            st.error("🚫 今日回報已截止 (16:00)")
        else:
            if sys_env == "DEV" and now_tw.hour >= 16:
                st.info("🔧 **[測試機特權開啟]** 目前已超過 16:00，但因為是 DEV 環境，允許繼續測試！")
                
            my_cls = st.selectbox("選擇班級", all_classes, key="m3_cls_select")
            main_df = load_main_data()
            # [V5.16 Patch] 改用 str.contains 模糊比對，只要包含「晨間打掃」一律擋下
            if not main_df[(main_df["日期"].astype(str)==str(today_tw)) & (main_df["班級"]==my_cls) & (main_df["評分項目"].astype(str).str.contains("晨間打掃"))].empty: 
                st.warning(f"⚠️ {my_cls} 今日已回報或已審核完畢囉！")
            else:
                duty_df, _ = get_daily_duty(today_tw)
                
                # [V5.13 Patch] 預設狀態為「沒有任務」
                has_duty = False 
                
                if not duty_df.empty:
                    m_d = duty_df[duty_df["負責班級"]==my_cls]
                    if not m_d.empty:
                        has_duty = True
                        area_name_str = str(m_d.iloc[0].get('掃地區域', '未指定區域'))
                        try: n_std = int(m_d.iloc[0].get('標準人數', 4))
                        except: n_std = 4
                
                # 判斷：如果今天沒有任務
                if not has_duty:
                    # [V5.24 Patch 1] 一、二年級專屬補打掃機制
                    if "一" in my_cls or "二" in my_cls:
                        st.info(f"💡 **{my_cls}** 今日無排定任務。若是本週錯過打掃，可在此進行「補打掃」回報！\n\n*(補掃通過將給予學期總分 +1，並核發志工時數)*")
                        
                        with st.form("makeup_form"):
                            st.write("請填寫補打掃資訊與上傳照片：")
                            # 因為沒排班不知道區域，讓學生自己填寫補掃的地方
                            mk_area = st.text_input("📍 實際補掃區域 (例如：外掃區A、308教室等)")
                            class_roster = [s for s, c in ROSTER_DICT.items() if c == my_cls]
                            mk_present = st.multiselect("✅ 參與補掃同學", class_roster)
                            mk_files = st.file_uploader("📸 補掃成果照片", accept_multiple_files=True, type=['jpg','png'])
                            
                            if st.form_submit_button("🚀 確認送出補掃回報"):
                                if time.time() - st.session_state.last_action_time < 3:
                                    st.warning("⚠️ 系統處理中，請勿連續點擊！")
                                elif not mk_area or not mk_present or not mk_files:
                                    st.error("❌ 區域、參與同學與照片皆為必填項目！")
                                else:
                                    st.session_state.last_action_time = time.time()
                                    final_note = f"[補掃:{mk_area}] 負責:{','.join(mk_present)}"
                                    
                                    ok = save_entry(
                                        {
                                            "日期": str(today_tw), 
                                            "班級": my_cls, 
                                            # 特別標註這是補掃紀錄，讓後台認得出來
                                            "評分項目": "晨間打掃(補掃)", 
                                            "檢查人員": f"志工補掃(實到:{len(mk_present)})", 
                                            "登錄時間": now_tw.strftime("%Y-%m-%d %H:%M:%S"), 
                                            "晨間打掃原始分": 0, 
                                            "備註": final_note
                                        }, 
                                        uploaded_files=mk_files, 
                                        student_list=mk_present, 
                                        custom_hours=0.5, 
                                        custom_category="晨掃志工"
                                    )
                                    if ok:
                                        st.success("✅ 補掃回報成功！辛苦了！")
                                        time.sleep(1.5)
                                        st.rerun()
                    else:
                        # 三年級如果沒任務，就直接讓他們去休息
                        st.success(f"🎉 恭喜！系統顯示 **{my_cls}** 今日沒有被分配到晨掃任務，好好休息吧！")
                        st.balloons()
                else:
                    # 有任務才會顯示下方的填寫與上傳區塊
                    areas = [a.strip() for a in area_name_str.split('、') if a.strip()]
                    if not areas: areas = ["打掃區域"]
                    
                    st.info(f"📍 本班任務總應到: {n_std} 人")
                    
                    # [V5.21 Patch] 放大公仔版面
                    daily_task = SYSTEM_CONFIG.get("daily_morning_task", "")
                    if daily_task:
                        formatted_task = daily_task.replace('\n', '<br>')
                        
                        # 這裡放妳的公仔網址 (ImgBB 或 GitHub Raw 網址)
                        mascot_url = "https://drive.google.com/thumbnail?id=128ITPXtpGNuI-wLIt6p-qd4ZNNhCGbhd"
                        
                        bubble_html = f"""
                        <style>
                        .mascot-container {{
                            display: flex;
                            align-items: flex-start;
                            margin-bottom: 20px;
                            gap: 15px;
                        }}
                        /* 🌟 調整 1：把電腦版的公仔從 80px 放大到 110px */
                        .mascot-img {{
                            width: 160px; 
                            flex-shrink: 0; 
                        }}
                        .speech-bubble {{
                            position: relative;
                            background: #FFF3CD;
                            border-radius: 15px;
                            padding: 15px 20px;
                            color: #664d03;
                            font-size: 16px;
                            box-shadow: 2px 4px 10px rgba(0,0,0,0.1);
                            border: 2px solid #ffecb5;
                            flex-grow: 1; 
                        }}
                        .speech-bubble::before {{
                            content: '';
                            position: absolute;
                            left: -20px;
                            top: 30px; /* 為了配合變大的公仔，把對話尾巴稍微往下移一點對齊嘴巴 */
                            width: 0;
                            height: 0;
                            border: 10px solid transparent;
                            border-right-color: #ffecb5;
                        }}
                        .speech-bubble::after {{
                            content: '';
                            position: absolute;
                            left: -16px;
                            top: 30px; /* 同上，跟著下移 */
                            width: 0;
                            height: 0;
                            border: 10px solid transparent;
                            border-right-color: #FFF3CD;
                        }}
                        
                        /* 📱 調整 2：把手機版的公仔從 60px 放大到 85px */
                        @media (max-width: 500px) {{
                            .mascot-img {{
                                width: 120px; 
                            }}
                            .speech-bubble {{
                                font-size: 14px;
                                padding: 10px 15px;
                            }}
                        }}
                        </style>
                        
                        <div class="mascot-container">
                            <img src="{mascot_url}" class="mascot-img" />
                            <div class="speech-bubble">
                                <strong>📢 組長廣播 / 今日任務：</strong><br><br>
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
                                    
                        if st.form_submit_button("🚀 確認送出全部回報"):
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
                                    ok = save_entry(
                                        {
                                            "日期": str(today_tw), 
                                            "班級": my_cls, 
                                            "評分項目": "晨間打掃", 
                                            "檢查人員": f"志工(實到:{len(all_present)})", 
                                            "登錄時間": now_tw.strftime("%Y-%m-%d %H:%M:%S"), 
                                            "晨間打掃原始分": 0, 
                                            "備註": final_note
                                        }, 
                                        uploaded_files=all_files, 
                                        student_list=all_present, 
                                        custom_hours=0.5, 
                                        custom_category="晨掃志工"
                                    )
                                    if ok:
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
        
        # [V5.9 Patch] 儀表板更新
        hb_status = "🟢 正常運作" if hb_sec < 60 else "🔴 已休眠/停止"
        is_dry_run = str(st.secrets.get("system_config", {}).get("dry_run", "false")).lower() in ["true", "1"]
        
        if is_dry_run: hb_status = "🟡 演習模式 (Dry Run)"
        
        if ls_sec == 999999: ls_text = "無紀錄"
        elif ls_sec < 120: ls_text = f"✅ {int(ls_sec)}秒前"
        else: ls_text = f"⚠️ {int(ls_sec//60)}分鐘前 (API可能卡住)"
            
        col4.metric("背景 Worker", f"{hb_status}", f"心跳: {int(hb_sec)}秒前 | 成功: {ls_text}")
        
        # 顯示最後錯誤
        last_err = get_last_error_summary()
        if last_err != "無紀錄":
            st.error(f"🚨 **最後錯誤紀錄:** {last_err}")

        pwd_input = st.text_input("管理密碼", type="password", key="admin_pwd")
        if pwd_input == st.secrets["system_config"]["admin_password"]:
            
            t_mon, t_rollcall, t4, t_appeal, t2, t1, t_settings, t3 = st.tabs([
                "👀 衛生糾察", "👮 環保糾察", "📝 扣分明細", "📣 申訴", "📊 成績總表", 
                "🧹 晨掃審核", "⚙️ 設定", "🏫 返校打掃"
            ])
            
            with t_mon:
                st.subheader("🕵️ 今日「衛生糾察」進度監控")
                monitor_date = st.date_input("監控日期", today_tw, key="monitor_date")
                st.caption(f"📅 此區顯示負責「內掃、外掃、機動」的評分糾察進度。")

                df = load_main_data()
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

            with t2:
                st.subheader("📊 成績總表")
                full = load_full_semester_data_for_export()
                
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
                            
                            if st.button("🚀 計算當週成績"):
                                week_df = full[full["週次"] == sel_week].copy()
                                week_df["內掃結算"] = week_df["內掃原始分"].clip(upper=2)
                                week_df["外掃結算"] = week_df["外掃原始分"].clip(upper=2)
                                trash_total = week_df["垃圾內掃原始分"] + week_df["垃圾外掃原始分"]
                                trash_total = trash_total.where(trash_total > 0, week_df["垃圾原始分"])
                                week_df["垃圾結算"] = trash_total.clip(upper=2)
                                
                                # [V5.15] 將負分(學期加分)歸零，確保「當週成績」不受影響
                                week_morning_penalty = week_df["晨間打掃原始分"].clip(lower=0)
                                week_df["總扣分"] = week_df["內掃結算"]+week_df["外掃結算"]+week_df["垃圾結算"]+week_morning_penalty+week_df["手機人數"]
                                
                                rep = week_df.groupby("班級")["總扣分"].sum().reset_index()
                                cls_df = pd.DataFrame(structured_classes).rename(columns={"grade":"年級","name":"班級"})
                                fin = pd.merge(cls_df, rep, on="班級", how="left").fillna(0)
                                fin["總成績"] = 90 - fin["總扣分"]
                                
                                if rank_mode == "全校": st.dataframe(fin.sort_values("總成績", ascending=False))
                                else:
                                    for g in sorted(fin["年級"].unique()):
                                        if g != "其他": 
                                            st.write(f"#### {g} 排名")
                                            st.dataframe(fin[fin["年級"]==g].sort_values("總成績", ascending=False))
                    
                    with tab_semester:
                        st.write("計算全學期累計總扣分與總成績")
                        sem_rank_mode = st.radio("學期排名方式", ["全校", "年級"], horizontal=True, key="sem_rank")
                        
                        if st.button("🚀 計算全學期成績", key="sem_btn"):
                            full["內掃結算"] = full["內掃原始分"].clip(upper=2)
                            full["外掃結算"] = full["外掃原始分"].clip(upper=2)
                            trash_total = full["垃圾內掃原始分"] + full["垃圾外掃原始分"]
                            trash_total = trash_total.where(trash_total > 0, full["垃圾原始分"])
                            full["垃圾結算"] = trash_total.clip(upper=2)
                            
                            # [V5.15] 這裡直接納入原始分(包含負數的加分)，總扣分減少 = 學期總成績增加！
                            full["總扣分"] = full["內掃結算"]+full["外掃結算"]+full["垃圾結算"]+full["晨間打掃原始分"]+full["手機人數"]
                            rep = full.groupby("班級")["總扣分"].sum().reset_index()
                            cls_df = pd.DataFrame(structured_classes).rename(columns={"grade":"年級","name":"班級"})
                            fin = pd.merge(cls_df, rep, on="班級", how="left").fillna(0)
                            
                            # [V5.15] 順便修復了原本隱藏的變數 Bug，確保算分正確
                            fin["總成績"] = 90 - fin["總扣分"] 
                            
                            if sem_rank_mode == "全校": st.dataframe(fin.sort_values("總成績", ascending=False))
                            else:
                                for g in sorted(fin["年級"].unique()):
                                    if g != "其他": 
                                        st.write(f"#### {g}")
                                        st.dataframe(fin[fin["年級"]==g].sort_values("總成績", ascending=False))
            with t1:
                # [V5.22 Patch 3] 晨掃未完成名單追蹤
                st.subheader("🕵️‍♀️ 今日晨掃進度追蹤")
                today_str = str(today_tw)
                duty_df, _ = get_daily_duty(today_tw)
                main_df = load_main_data()
                
                if not duty_df.empty:
                    # 取得今天有排班的班級
                    assigned_classes = set(duty_df["負責班級"].dropna().astype(str).tolist())
                    # 取得今天已經回報的班級 (只要項目包含晨間打掃就算)
                    reported_classes = set(main_df[(main_df["日期"].astype(str) == today_str) & (main_df["評分項目"].astype(str).str.contains("晨間打掃"))]["班級"].astype(str).tolist())
                    
                    # 互相抵銷，抓出還沒掃的班級
                    missing_classes = assigned_classes - reported_classes
                    
                    if missing_classes:
                        st.error(f"🚨 **尚未回報班級 ({len(missing_classes)}班)：** {', '.join(sorted(list(missing_classes)))}")
                    else:
                        st.success("🎉 太棒了！今日所有排定班級皆已完成晨掃回報！")
                else:
                    st.info("今日無晨掃排班任務。")
                    
                st.markdown("---")
                st.subheader("📝 待審核回報列表")
                
                # [V5.24 Patch 2] 支援審核「正常晨掃」與「補掃」紀錄
                df = main_df 
                # 把條件改成 isin，同時抓取這兩種項目
                for i, r in df[df["評分項目"].isin(["晨間打掃", "晨間打掃(補掃)"]) & (df["晨間打掃原始分"]==0) & (df["修正"]!="TRUE")].iterrows():
                    with st.container(border=True):
                        c1, c2, c3 = st.columns([2,2,1.3])
                        
                        # 判斷這筆是不是補掃，並加上顯眼的標籤
                        is_makeup = (r["評分項目"] == "晨間打掃(補掃)")
                        title_badge = "🩹 **[補掃]**" if is_makeup else "🧹"
                        
                        c1.write(f"{title_badge} **{r['班級']}** | {r['檢查人員']}")
                        c1.caption(f"登錄時間：{r['登錄時間']}") 
                        
                        if "http" in str(r['照片路徑']): 
                            c2.image([p for p in str(r['照片路徑']).split(";") if "http" in p], width=150) 
                        
                        reply_msg = c1.text_input("💬 給予回應 (可留白)", key=f"rm_{r['紀錄ID']}")

                        if is_makeup:
                            # 🩹 補掃專屬按鈕 (+1分)
                            if c3.button("✅ 補掃核可(學期+1)", key=f"pm_{r['紀錄ID']}"):
                                ws = get_worksheet(SHEET_TABS["main"])
                                id_list = ws.col_values(EXPECTED_COLUMNS.index("紀錄ID")+1)
                                if str(r["紀錄ID"]) in id_list:
                                    ridx = id_list.index(str(r["紀錄ID"])) + 1
                                    # 寫入 -1，代表總分加 1 分
                                    ws.update_cell(ridx, EXPECTED_COLUMNS.index("晨間打掃原始分")+1, -1)
                                    ws.update_cell(ridx, EXPECTED_COLUMNS.index("評分項目")+1, "晨間打掃(學期加分)")
                                    
                                    old_note = str(r['備註'])
                                    new_note = f"{old_note} \n組長回覆: {reply_msg}" if reply_msg else f"{old_note} \n組長核可: 補掃通過(學期總分+1)"
                                    ws.update_cell(ridx, EXPECTED_COLUMNS.index("備註")+1, new_note)
                                    
                                    load_main_data.clear()
                                    st.rerun()
                        else:
                            # 🧹 原本的正常晨掃按鈕
                            if c3.button("✅ 4人全到(學期+2)", key=f"p4_{r['紀錄ID']}"): 
                                ws = get_worksheet(SHEET_TABS["main"])
                                id_list = ws.col_values(EXPECTED_COLUMNS.index("紀錄ID")+1)
                                if str(r["紀錄ID"]) in id_list:
                                    ridx = id_list.index(str(r["紀錄ID"])) + 1
                                    ws.update_cell(ridx, EXPECTED_COLUMNS.index("晨間打掃原始分")+1, -2)
                                    ws.update_cell(ridx, EXPECTED_COLUMNS.index("評分項目")+1, "晨間打掃(學期加分)")
                                    old_note = str(r['備註'])
                                    new_note = f"{old_note} \n組長回覆: {reply_msg}" if reply_msg else f"{old_note} \n組長核可: 4人全到(學期總分+2)"
                                    ws.update_cell(ridx, EXPECTED_COLUMNS.index("備註")+1, new_note)
                                    load_main_data.clear()
                                    st.rerun()

                            if c3.button("✅ 2人全到(學期+1)", key=f"p2_{r['紀錄ID']}"): 
                                ws = get_worksheet(SHEET_TABS["main"])
                                id_list = ws.col_values(EXPECTED_COLUMNS.index("紀錄ID")+1)
                                if str(r["紀錄ID"]) in id_list:
                                    ridx = id_list.index(str(r["紀錄ID"])) + 1
                                    ws.update_cell(ridx, EXPECTED_COLUMNS.index("晨間打掃原始分")+1, -1)
                                    ws.update_cell(ridx, EXPECTED_COLUMNS.index("評分項目")+1, "晨間打掃(學期加分)")
                                    old_note = str(r['備註'])
                                    new_note = f"{old_note} \n組長回覆: {reply_msg}" if reply_msg else f"{old_note} \n組長核可: 2人全到(學期總分+1)"
                                    ws.update_cell(ridx, EXPECTED_COLUMNS.index("備註")+1, new_note)
                                    load_main_data.clear()
                                    st.rerun()

                        # 共用的駁回按鈕
                        if c3.button("🗑️ 駁回", key=f"r_{r['紀錄ID']}"): 
                            ws = get_worksheet(SHEET_TABS["main"])
                            id_list = ws.col_values(EXPECTED_COLUMNS.index("紀錄ID")+1)
                            if str(r["紀錄ID"]) in id_list:
                                ridx = id_list.index(str(r["紀錄ID"])) + 1
                                ws.update_cell(ridx, EXPECTED_COLUMNS.index("評分項目")+1, "晨間打掃(已駁回)")
                                old_note = str(r['備註'])
                                rej_msg = reply_msg if reply_msg else "未達標準，請見諒"
                                new_note = f"{old_note} \n組長駁回: {rej_msg}"
                                ws.update_cell(ridx, EXPECTED_COLUMNS.index("備註")+1, new_note)
                                
                                load_main_data.clear()
                                st.rerun()

            with t_settings:
                st.subheader("⚙️ 系統設定與維護")
                curr = SYSTEM_CONFIG.get("semester_start")
                nd = st.date_input("開學日", datetime.strptime(curr, "%Y-%m-%d").date() if curr else today_tw)
                if st.button("更新開學日"): save_setting("semester_start", str(nd))
                
                st.markdown("---")
                # [V5.17 Patch] 新增晨掃每日任務廣播
                st.write("📢 晨掃志工每日廣播/任務")
                current_task = SYSTEM_CONFIG.get("daily_morning_task", "今日無特殊任務，請確實完成各區打掃即可！")
                new_task = st.text_area("請輸入想給志工看的話（例如：拍照請比 YA、今天請加強拖地等）", value=current_task)
                if st.button("💾 更新每日任務"): 
                    if save_setting("daily_morning_task", new_task):
                        st.success("✅ 每日任務已更新！學生現在起會看到最新廣播。")
                
                st.markdown("---")
                
                # [V5.22 Patch 1] 新增衛生糾察廣播設定
                st.write("📢 衛生糾察每日廣播/提醒")
                current_hygiene_task = SYSTEM_CONFIG.get("daily_hygiene_task", "今日無特殊任務，請確實完成各區檢查即可！")
                new_hygiene_task = st.text_area("請輸入想給糾察隊看的話（例如：今天重點檢查黑板、窗台）", value=current_hygiene_task)
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
                c1, c2 = st.columns(2)
                rd, rc = c1.date_input("日期", today_tw, key="ret_date"), c2.selectbox("班級", all_classes, key="ret_cls")
                mems = [s for s, c in ROSTER_DICT.items() if c == rc]
                if mems:
                    with st.form("ret_clean"):
                        absent = st.multiselect("缺席名單", mems)
                        pool = [m for m in mems if m not in absent]
                        base_h = st.number_input("基礎時數", value=2.0, step=0.5)
                        spec = st.multiselect("加強組", pool)
                        spec_h = st.number_input("特別時數", value=3.0, step=0.5)
                        pf = st.file_uploader("照片", type=['jpg','png'])
                        
                        if st.form_submit_button("發放"):
                            if time.time() - st.session_state.last_action_time < 3:
                                st.warning("⚠️ 系統處理中，請勿連續點擊！")
                            elif pf:
                                st.session_state.last_action_time = time.time()
                                pf.seek(0); fb = pf.read()
                                norm = [m for m in pool if m not in spec]
                                ok_norm, ok_spec = True, True
                                if norm: 
                                    pf_n = io.BytesIO(fb); pf_n.name="p.jpg"
                                    ok_norm = save_entry({"日期": str(rd), "班級": rc, "評分項目": "返校打掃", "登錄時間": now_tw.strftime("%Y-%m-%d %H:%M:%S")}, [pf_n], norm, base_h, "返校打掃(一般)")
                                if spec: 
                                    pf_s = io.BytesIO(fb); pf_s.name="p.jpg"
                                    ok_spec = save_entry({"日期": str(rd), "班級": rc, "評分項目": "返校打掃", "登錄時間": now_tw.strftime("%Y-%m-%d %H:%M:%S")}, [pf_s], spec, spec_h, "返校打掃(加強)")
                                
                                if ok_norm and ok_spec:
                                    st.success("已登記！"); time.sleep(1.5); st.rerun()
                            else:
                                st.error("需上傳照片")

        elif pwd_input != "":
            st.error("密碼錯誤")

except Exception as e:
    st.error(f"❌ 系統發生錯誤: {str(e)}")
    st.code(traceback.format_exc())
