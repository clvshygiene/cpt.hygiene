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
import tempfile
from datetime import datetime, date, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 第三方套件
import pytz
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# --- 1. 網頁設定 ---
st.set_page_config(page_title="中壢家商，衛愛而生", layout="wide", page_icon="🧹")

# ==========================================
# 0. 基礎設定與常數
# ==========================================
TW_TZ = pytz.timezone('Asia/Taipei')
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 單檔圖片 10MB 上限

# [SRE] 使用系統暫存目錄，適應 Streamlit Cloud Ephemeral 環境
TEMP_DIR = tempfile.gettempdir()
QUEUE_DB_PATH = os.path.join(TEMP_DIR, "task_queue_v9_threadsafe.db") # Version updated
IMG_DIR = os.path.join(TEMP_DIR, "evidence_photos")
os.makedirs(IMG_DIR, exist_ok=True)

# Google Sheet 網址
SHEET_URL = "https://docs.google.com/spreadsheets/d/11BXtN3aevJls6Q2IR_IbT80-9XvhBkjbTCgANmsxqkg/edit"

SHEET_TABS = {
    "main": "main_data", 
    "settings": "settings",
    "roster": "roster",
    "inspectors": "inspectors",
    "duty": "duty",
    "teachers": "teachers",
    "appeals": "appeals"
}

EXPECTED_COLUMNS = [
    "日期", "週次", "班級", "評分項目", "檢查人員",
    "內掃原始分", "外掃原始分", "垃圾原始分", "垃圾內掃原始分", "垃圾外掃原始分", "晨間打掃原始分", "手機人數",
    "備註", "違規細項", "照片路徑", "登錄時間", "修正", "晨掃未到者", "紀錄ID"
]

APPEAL_COLUMNS = [
    "申訴日期", "班級", "違規日期", "違規項目", "原始扣分", "申訴理由", "佐證照片", "處理狀態", "登錄時間", "對應紀錄ID"
]

# ==========================================
# 1. 工具函式 (Utils)
# ==========================================

def compress_image_bytes(raw: bytes, max_side: int = 1600, quality: int = 75) -> bytes:
    """
    [SRE] Pass-through 模式，避免 PIL C-extension 在雲端環境 Crash。
    """
    return raw

def clean_id(val):
    try:
        if pd.isna(val) or val == "": return ""
        return str(int(float(val))).strip()
    except: return str(val).strip()

def execute_with_retry(func, max_retries=5, base_delay=1.0):
    """
    SRE Pattern: 指數退避重試機制
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_str = str(e).lower()
            is_retryable = any(x in error_str for x in ['429', '500', '503', 'quota', 'rate limit', 'timed out', 'connection'])
            
            if is_retryable and attempt < max_retries - 1:
                sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                print(f"⚠️ API 忙碌 ({e})，第 {attempt+1} 次重試，等待 {sleep_time:.2f}秒...")
                time.sleep(sleep_time)
            else:
                raise e

# ==========================================
# 2. Google API 連線 (Thread-Safe Fix)
# ==========================================

# [SRE Fix] 這是原本的 Cache 版本，給 UI 顯示用 (讀取 Sheet)
@st.cache_resource
def get_credentials_cached():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    if "gcp_service_account" not in st.secrets:
        return None
    creds_dict = dict(st.secrets["gcp_service_account"])
    return ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)

@st.cache_resource
def get_gspread_client():
    try:
        creds = get_credentials_cached()
        if not creds: return None
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ Google Sheet 連線失敗: {e}")
        return None

@st.cache_resource(ttl=3600)
def get_spreadsheet_object():
    client = get_gspread_client()
    if not client: return None
    try: return client.open_by_url(SHEET_URL)
    except Exception as e: 
        st.error(f"❌ 無法開啟試算表: {e}")
        return None

# [SRE Fix] 這是「無快取」版本，專門給 Thread 使用
# 避免在 Thread 中呼叫 st.cache_resource 導致 Missing ScriptRunContext
def get_drive_service_raw():
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        if "gcp_service_account" not in st.secrets:
            return None
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        return build('drive', 'v3', credentials=creds, cache_discovery=False)
    except Exception as e:
        print(f"Drive Service Error: {e}")
        return None

def get_worksheet(tab_name):
    max_retries = 3
    wait_time = 2
    sheet = get_spreadsheet_object()
    if not sheet: return None
    for attempt in range(max_retries):
        try:
            try: return sheet.worksheet(tab_name)
            except gspread.WorksheetNotFound:
                cols = 20 if tab_name != "appeals" else 15
                ws = sheet.add_worksheet(title=tab_name, rows=100, cols=cols)
                if tab_name == "appeals": ws.append_row(APPEAL_COLUMNS)
                return ws
        except Exception as e:
            if "429" in str(e): 
                time.sleep(wait_time * (attempt + 1))
                continue
            else: 
                print(f"❌ 讀取分頁 '{tab_name}' 失敗: {e}")
                return None
    return None

def upload_image_to_drive(file_obj, filename):
    # [SRE Fix] 這裡改用 raw service，不使用 Streamlit Cache
    service = get_drive_service_raw()
    if not service: 
        print("❌ Drive Service Not Available (Raw)")
        return None
    
    folder_id = None
    if "system_config" in st.secrets and "drive_folder_id" in st.secrets["system_config"]:
        folder_id = st.secrets["system_config"]["drive_folder_id"]

    def _upload_action():
        metadata = {'name': filename}
        if folder_id:
            metadata['parents'] = [folder_id]
            
        media = MediaIoBaseUpload(file_obj, mimetype='image/jpeg', resumable=True)
        file = service.files().create(body=metadata, media_body=media, fields='id,webViewLink').execute()
        return file.get('webViewLink') or f"https://drive.google.com/file/d/{file.get('id')}/view"

    try:
        return execute_with_retry(_upload_action)
    except Exception as e:
        print(f"⚠️ Drive 上傳最終失敗: {str(e)}")
        return None

def upload_images_parallel(files_list, entry_data):
    if not files_list:
        return [], True

    upload_results = [None] * len(files_list)
    tasks = []
    
    for i, up_file in enumerate(files_list):
        up_file.seek(0)
        raw = up_file.read()
        data = compress_image_bytes(raw)

        safe_class = str(entry_data.get("班級", "unknown"))
        logical_fname = f"{entry_data.get('日期', '')}_{safe_class}_{i}.jpg"
        unique_prefix = f"{datetime.now(TW_TZ).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        drive_filename = f"{unique_prefix}_{logical_fname}"
        
        tasks.append((io.BytesIO(data), drive_filename, i))

    # [SRE] 並行數限制 2
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_to_index = {
            executor.submit(upload_image_to_drive, f_obj, fname): idx 
            for f_obj, fname, idx in tasks
        }
        
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                link = future.result()
                upload_results[idx] = link
            except Exception as e:
                print(f"上傳失敗: {e}")
                upload_results[idx] = None

    if any(link is None for link in upload_results):
        return [], False
    
    return upload_results, True

# ==========================================
# 3. SQLite 背景佇列系統 (SRE Hardened v4 - Production Safe)
# ==========================================

_db_lock = threading.Lock()

def get_new_db_connection():
    try:
        conn = sqlite3.connect(
            QUEUE_DB_PATH, 
            check_same_thread=False, 
            timeout=30.0, 
            isolation_level=None
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn
    except Exception as e:
        print(f"[CRITICAL] DB Connect Error: {e}")
        return None

def init_db_if_needed():
    with _db_lock:
        conn = get_new_db_connection()
        if conn:
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS task_queue (
                        id TEXT PRIMARY KEY,
                        task_type TEXT NOT NULL,
                        created_ts TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        status TEXT NOT NULL,
                        attempts INTEGER NOT NULL DEFAULT 0,
                        last_error TEXT
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_status_created ON task_queue (status, created_ts);")
                conn.close()
            except Exception as e:
                print(f"[INIT ERROR] {e}")

init_db_if_needed()

def recover_stale_tasks():
    with _db_lock:
        conn = get_new_db_connection()
        if not conn: return
        try:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.cursor()
            cur.execute(
                "UPDATE task_queue SET status='RETRY', last_error='Worker Restarted Recovery', attempts=attempts WHERE status='RUNNING'"
            )
            count = cur.rowcount
            conn.commit()
            if count > 0:
                print(f"♻️  已復原 {count} 筆殭屍任務")
        except Exception as e:
            print(f"[RECOVERY ERROR] {e}")
        finally:
            conn.close()

def enqueue_task(task_type: str, payload: dict) -> str:
    ensure_worker_started()
    
    task_id = str(uuid.uuid4())
    created_ts = datetime.utcnow().isoformat() + "Z"
    payload_json = json.dumps(payload, ensure_ascii=False)

    with _db_lock:
        conn = get_new_db_connection()
        if not conn: return "DB_ERROR"
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO task_queue (id, task_type, created_ts, payload_json, status, attempts, last_error) "
                "VALUES (?, ?, ?, ?, 'PENDING', 0, NULL)",
                (task_id, task_type, created_ts, payload_json)
            )
            conn.commit()
            return task_id
        except Exception as e:
            try: conn.rollback()
            except: pass
            print(f"[ERROR] Enqueue Failed: {e}")
            return "DB_ERROR"
        finally:
            conn.close()

def get_queue_metrics():
    metrics = {"pending": 0, "retry": 0, "failed": 0, "running": 0, "oldest_pending_sec": 0, "recent_errors": []}
    with _db_lock:
        conn = get_new_db_connection()
        if not conn: return metrics
        try:
            cur = conn.cursor()
            cur.execute("SELECT status, COUNT(*) FROM task_queue GROUP BY status")
            rows = cur.fetchall()
            for status, count in rows:
                if status == 'PENDING': metrics["pending"] = count
                elif status == 'RETRY': metrics["retry"] = count
                elif status == 'FAILED': metrics["failed"] = count
                elif status == 'RUNNING': metrics["running"] = count
            
            cur.execute("SELECT MIN(created_ts) FROM task_queue WHERE status IN ('PENDING', 'RETRY', 'RUNNING')")
            row = cur.fetchone()
            oldest_ts_str = row[0] if row else None
            
            cur.execute("SELECT last_error, created_ts FROM task_queue WHERE status='FAILED' OR status='RETRY' ORDER BY created_ts DESC LIMIT 5")
            metrics["recent_errors"] = cur.fetchall()
            
            if oldest_ts_str:
                try:
                    created = datetime.fromisoformat(oldest_ts_str.replace("Z", "+00:00"))
                    now = datetime.now(pytz.utc)
                    metrics["oldest_pending_sec"] = (now - created).total_seconds()
                except: pass
        except Exception as e: 
            print(f"[WARN] Metrics Error: {e}")
        finally:
            conn.close()
    return metrics

def fetch_next_task(max_attempts: int = 6):
    with _db_lock:
        conn = get_new_db_connection()
        if not conn: return None
        try:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.cursor()
            
            cur.execute(
                """
                SELECT id, task_type, created_ts, payload_json, status, attempts
                FROM task_queue
                WHERE status IN ('PENDING', 'RETRY')
                AND attempts < ?
                ORDER BY created_ts ASC
                LIMIT 1
                """,
                (max_attempts,)
            )
            row = cur.fetchone()
            
            if not row:
                conn.commit()
                return None

            task_id, task_type, created_ts, payload_json, status, attempts = row
            new_attempts = attempts + 1
            
            # [SRE Fix] 防禦性更新
            cur.execute(
                "UPDATE task_queue SET status = 'RUNNING', attempts = ? WHERE id = ? AND status IN ('PENDING', 'RETRY')",
                (new_attempts, task_id)
            )
            
            if cur.rowcount == 0:
                conn.rollback()
                return None
            
            conn.commit()
            
            try: payload = json.loads(payload_json)
            except: payload = {}
                
            return {
                "id": task_id,
                "task_type": task_type,
                "created_ts": created_ts,
                "payload": payload,
                "attempts": new_attempts,
            }
        except Exception as e:
            try: conn.rollback()
            except: pass
            print(f"[ERROR] Fetch Task Error: {e}")
            return None
        finally:
            conn.close()

def update_task_status(task_id: str, status: str, last_error: str | None):
    with _db_lock:
        conn = get_new_db_connection()
        if not conn: return
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE task_queue SET status = ?, last_error = ? WHERE id = ?",
                (status, last_error, task_id),
            )
            conn.commit()
        except Exception as e: 
            try: conn.rollback()
            except: pass
            print(f"[ERROR] Update Status Failed ({task_id}): {e}")
        finally:
            conn.close()

# --- Worker Logic ---

def _append_main_entry_row(entry: dict):
    def _action():
        ws = get_worksheet(SHEET_TABS["main"])
        if not ws: raise Exception("Failed to get main worksheet")
        
        all_vals = ws.get_all_values()
        if not all_vals: ws.append_row(EXPECTED_COLUMNS)

        row = []
        for col in EXPECTED_COLUMNS:
            val = entry.get(col, "")
            if isinstance(val, bool): val = str(val).upper()
            if col == "日期": val = str(val)
            row.append(val)
        ws.append_row(row)
    execute_with_retry(_action)

def _append_appeal_row(entry: dict):
    def _action():
        ws = get_worksheet(SHEET_TABS["appeals"])
        if not ws: raise Exception("Failed to get appeals worksheet")

        all_vals = ws.get_all_values()
        if not all_vals: ws.append_row(APPEAL_COLUMNS)

        row = [str(entry.get(col, "")) for col in APPEAL_COLUMNS]
        ws.append_row(row)
    execute_with_retry(_action)

def process_task(task: dict) -> tuple[bool, str | None]:
    task_type = task["task_type"]
    payload = task["payload"]
    entry = payload.get("entry", {}) or {}

    try:
        if task_type == "main_entry":
            _append_main_entry_row(entry)
            return True, None

        elif task_type == "appeal_entry":
            image_info = payload.get("image_file")
            if image_info and image_info.get("path") and os.path.exists(image_info["path"]):
                with open(image_info["path"], "rb") as f:
                    link = upload_image_to_drive(f, image_info["filename"])
                entry["佐證照片"] = link if link else "UPLOAD_FAILED"
            else:
                entry["佐證照片"] = entry.get("佐證照片", "")

            _append_appeal_row(entry)
            return True, None
        else:
            return True, None

    except Exception as e:
        return False, str(e)

def process_task_wrapper(task, max_attempts):
    task_id = task["id"]
    attempts = task["attempts"] 
    payload = task["payload"]
    
    ok = False
    err_msg = None
    try:
        ok, err_msg = process_task(task)
    except Exception as e:
        err_msg = f"UNHANDLED: {e}\n{traceback.format_exc()}"
        ok = False

    try:
        image_paths = []
        if isinstance(payload, dict):
            if "image_file" in payload and "path" in payload["image_file"]:
                image_paths.append(payload["image_file"]["path"])
        for p in image_paths:
            if p and os.path.exists(p): os.remove(p)
    except: pass

    if ok:
        update_task_status(task_id, "DONE", None)
        print(f"✅ Task {task_id} 完成")
    else:
        status = "FAILED" if attempts >= max_attempts else "RETRY"
        print(f"⚠️ Task {task_id} 失敗: {err_msg}")
        update_task_status(task_id, status, err_msg or "unknown")

def background_worker(stop_event: threading.Event | None = None):
    recover_stale_tasks()
    
    # [SRE] 背景 Worker 仍然使用 ThreadPool 是安全的，因為它們不涉及 Streamlit Context
    max_attempts = 6
    MAX_WORKERS = 2 
    print(f"🚀 背景工作者啟動 (Workers: {MAX_WORKERS})...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        
        while True:
            if stop_event is not None and stop_event.is_set():
                break

            done_futures = [f for f in futures if f.done()]
            for f in done_futures:
                futures.remove(f)
                try: f.result() 
                except Exception as e: print(f"[ERROR] Thread Future Error: {e}")

            if len(futures) >= MAX_WORKERS:
                time.sleep(0.5)
                continue

            task = fetch_next_task(max_attempts=max_attempts)
            if not task:
                time.sleep(2.0) 
                continue

            print(f"⚡ 任務 {task['id']} (Try: {task['attempts']}) 執行中")
            future = executor.submit(process_task_wrapper, task, max_attempts)
            futures.append(future)

# --- Phoenix Keeper ---
_worker_lock = threading.Lock()
_worker_thread = None
_worker_stop_event = None

def ensure_worker_started():
    global _worker_thread, _worker_stop_event
    with _worker_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            print("❤️‍🔥 啟動/重啟背景工作者...")
            _worker_stop_event = threading.Event()
            _worker_thread = threading.Thread(target=background_worker, args=(_worker_stop_event,), daemon=True)
            _worker_thread.start()

# ==========================================
# 4. 資料讀取邏輯 (Frontend)
# ==========================================

@st.cache_data(ttl=60)
def load_main_data():
    ws = get_worksheet(SHEET_TABS["main"])
    if not ws:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)
    try:
        data = ws.get_all_records()
        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame(columns=EXPECTED_COLUMNS)

        for col in EXPECTED_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        if "紀錄ID" not in df.columns:
            df["紀錄ID"] = df.index.astype(str)
        else:
            df["紀錄ID"] = df["紀錄ID"].astype(str)

        if "照片路徑" in df.columns:
            df["照片路徑"] = df["照片路徑"].fillna("").astype(str)

        numeric_cols = ["內掃原始分", "外掃原始分", "垃圾原始分", "晨間打掃原始分", "手機人數"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        if "週次" in df.columns:
            df["週次"] = pd.to_numeric(df["週次"], errors="coerce").fillna(0).astype(int)
    
    except Exception as e:
        st.error(f"讀取資料錯誤: {e}")
        return pd.DataFrame(columns=EXPECTED_COLUMNS)
    
    return df[EXPECTED_COLUMNS]

def save_entry(new_entry, uploaded_files=None):
    if "日期" in new_entry and new_entry["日期"]:
        new_entry["日期"] = str(new_entry["日期"])

    files_list = [f for f in uploaded_files if f] if uploaded_files else []

    if len(files_list) > 4:
        st.error("❌ 一次最多只能上傳 4 張照片，請刪減後再送出。")
        return False

    drive_links = []
    
    # 嚴格模式：前景上傳
    if files_list:
        with st.spinner("☁️ 正在上傳照片並驗證證據..."):
            links, success = upload_images_parallel(files_list, new_entry)
        
        if not success:
            st.error("🛑 **上傳失敗，評分未送出！**\n\n系統偵測到照片上傳雲端失敗，為了避免「有扣分無證據」的爭議，本筆紀錄已被系統攔截。\n請檢查網路連線後重試。")
            return False
        
        drive_links = links

    if drive_links:
        new_entry["照片路徑"] = ";".join(drive_links)

    if "紀錄ID" not in new_entry or not new_entry["紀錄ID"]:
        unique_suffix = uuid.uuid4().hex[:6]
        timestamp = datetime.now(TW_TZ).strftime("%Y%m%d%H%M%S")
        new_entry["紀錄ID"] = f"{timestamp}_{unique_suffix}"

    payload = {"entry": new_entry}

    try:
        res = enqueue_task("main_entry", payload)
        if res == "DB_ERROR":
            st.error("❌ 本地資料庫錯誤，請重新整理頁面。")
            return False
        return True
    except Exception as e:
        st.error(f"❌ 寫入佇列失敗: {e}")
        return False

def save_appeal(entry, proof_file=None):
    image_info = None

    if proof_file:
        try:
            proof_file.seek(0)
            data = proof_file.read()
        except Exception as e:
            st.error(f"❌ 讀取佐證照片失敗: {e}")
            return False

        if not data:
            st.error("❌ 佐證照片為空檔案")
            return False
        
        # 存到系統暫存區
        logical_fname = f"Appeal_{entry.get('班級', '')}_{datetime.now(TW_TZ).strftime('%H%M%S')}.jpg"
        tmp_fname = f"{datetime.now(TW_TZ).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}_{logical_fname}"
        local_path = os.path.join(IMG_DIR, tmp_fname)
        try:
            with open(local_path, "wb") as f:
                f.write(data)
            image_info = {"path": local_path, "filename": logical_fname}
        except Exception as e:
            st.error(f"❌ 寫入佐證暫存檔失敗: {e}")
            return False

    if "申訴日期" not in entry or not entry["申訴日期"]:
        entry["申訴日期"] = datetime.now(TW_TZ).strftime("%Y-%m-%d")
    entry["處理狀態"] = entry.get("處理狀態", "待處理")
    if "登錄時間" not in entry or not entry["登錄時間"]:
        entry["登錄時間"] = datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")
    if "申訴ID" not in entry or not entry["申訴ID"]:
        entry["申訴ID"] = datetime.now(TW_TZ).strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:4]
    if "佐證照片" not in entry:
        entry["佐證照片"] = ""

    payload = {
        "entry": entry,
        "image_file": image_info,
    }
    enqueue_task("appeal_entry", payload)
    st.success("📩 申訴已排入背景處理")
    return True

@st.cache_data(ttl=60)
def load_appeals():
    ws = get_worksheet(SHEET_TABS["appeals"])
    if not ws:
        return pd.DataFrame(columns=APPEAL_COLUMNS)

    try:
        records = ws.get_all_records()
        df = pd.DataFrame(records)
    except Exception:
        return pd.DataFrame(columns=APPEAL_COLUMNS)

    for col in APPEAL_COLUMNS:
        if col not in df.columns:
            if col == "處理狀態":
                df[col] = "待處理"
            else:
                df[col] = ""

    df = df[APPEAL_COLUMNS]
    return df

def delete_rows_by_ids(record_ids_to_delete):
    ws = get_worksheet(SHEET_TABS["main"])
    if not ws: return False
    try:
        records = ws.get_all_records()
        rows_to_delete = []
        for i, record in enumerate(records):
            if str(record.get("紀錄ID")) in record_ids_to_delete:
                rows_to_delete.append(i + 2)
        
        rows_to_delete.sort(reverse=True)
        for row_idx in rows_to_delete:
            ws.delete_rows(row_idx)
            time.sleep(0.8)
    
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"刪除失敗: {e}")
        return False

def update_appeal_status(appeal_row_idx, status, record_id):
    ws_appeals = get_worksheet(SHEET_TABS["appeals"])
    ws_main = get_worksheet(SHEET_TABS["main"])
    try:
        appeals_data = ws_appeals.get_all_records()
        target_row = None
        for i, row in enumerate(appeals_data):
            if str(row.get("對應紀錄ID")) == str(record_id) and str(row.get("處理狀態")) == "待處理":
                target_row = i + 2 
                break
        if target_row:
            col_idx = APPEAL_COLUMNS.index("處理狀態") + 1
            ws_appeals.update_cell(target_row, col_idx, status)
            if status == "已核可" and record_id:
                main_data = ws_main.get_all_records()
                main_target_row = None
                for j, m_row in enumerate(main_data):
                    if str(m_row.get("紀錄ID")) == str(record_id):
                        main_target_row = j + 2
                        break
                if main_target_row:
                    fix_col_idx = EXPECTED_COLUMNS.index("修正") + 1
                    ws_main.update_cell(main_target_row, fix_col_idx, "TRUE")
            st.cache_data.clear()
            return True, "更新成功"
        else: return False, "找不到對應的申訴列"
    except Exception as e: return False, str(e)

@st.cache_data(ttl=21600)
def load_roster_dict():
    ws = get_worksheet(SHEET_TABS["roster"])
    roster_dict = {}
    if ws:
        try:
            df = pd.DataFrame(ws.get_all_records())
            id_col = next((c for c in df.columns if "學號" in c), None)
            class_col = next((c for c in df.columns if "班級" in c), None)
            if id_col and class_col:
                for _, row in df.iterrows():
                    sid = clean_id(row[id_col])
                    if sid: roster_dict[sid] = str(row[class_col]).strip()
        except: pass
    return roster_dict
    
@st.cache_data(ttl=3600)
def load_sorted_classes():
    ws = get_worksheet(SHEET_TABS["roster"])
    if not ws: return [], []
    try:
        df = pd.DataFrame(ws.get_all_records())
        class_col = next((c for c in df.columns if "班級" in c), None)
        if not class_col: return [], []
        
        unique_classes = df[class_col].dropna().unique().tolist()
        unique_classes = [str(c).strip() for c in unique_classes if str(c).strip()]
        
        dept_order = {"商": 1, "英": 2, "資": 3, "家": 4, "服": 5}
        
        def get_sort_key(name):
            grade = 99
            if "一" in name or "1" in name: grade = 1
            if "二" in name or "2" in name: grade = 2
            if "三" in name or "3" in name: grade = 3
            
            dept_score = 99
            for k, v in dept_order.items():
                if k in name:
                    dept_score = v
                    break
            return (grade, dept_score, name)
        
        sorted_all = sorted(unique_classes, key=get_sort_key)
        
        structured = []
        for c in sorted_all:
            grade_val = get_sort_key(c)[0]
            g_label = f"{grade_val}年級" if grade_val != 99 else "其他"
            structured.append({"grade": g_label, "name": c})
            
        return sorted_all, structured
    except Exception as e:
        print(f"Sorting Error: {e}")
        return [], []

@st.cache_data(ttl=21600)
def load_teacher_emails():
    ws = get_worksheet(SHEET_TABS["teachers"])
    email_dict = {}
    if ws:
        try:
            df = pd.DataFrame(ws.get_all_records())
            class_col = next((c for c in df.columns if "班級" in c), None)
            mail_col = next((c for c in df.columns if "Email" in c or "信箱" in c or "郵件" in c), None)
            name_col = next((c for c in df.columns if "導師" in c or "姓名" in c), None)
            if class_col and mail_col:
                for _, row in df.iterrows():
                    cls = str(row[class_col]).strip()
                    mail = str(row[mail_col]).strip()
                    name = str(row[name_col]).strip() if name_col else "老師"
                    if cls and mail and "@" in mail:
                        email_dict[cls] = {"email": mail, "name": name}
        except: pass
    return email_dict

@st.cache_data(ttl=21600)
def load_inspector_list():
    ws = get_worksheet(SHEET_TABS["inspectors"])
    default = [{"label": "測試人員", "allowed_roles": ["內掃檢查"], "assigned_classes": [], "id_prefix": "測"}]
    if not ws: return default
    try:
        df = pd.DataFrame(ws.get_all_records())
        if df.empty: return default
        inspectors = []
        id_col = next((c for c in df.columns if "學號" in c or "編號" in c), None)
        role_col = next((c for c in df.columns if "負責" in c or "項目" in c), None)
        scope_col = next((c for c in df.columns if "班級" in c or "範圍" in c), None)
        if id_col:
            for _, row in df.iterrows():
                s_id = clean_id(row[id_col])
                s_role = str(row[role_col]).strip() if role_col else ""
                allowed = []
                if "組長" in s_role: allowed = ["內掃檢查", "外掃檢查", "垃圾/回收檢查", "晨間打掃"]
                elif "機動" in s_role: allowed = ["內掃檢查", "外掃檢查", "垃圾/回收檢查"]
                else:
                    if "外掃" in s_role: allowed.append("外掃檢查")
                    if "垃圾" in s_role: allowed.append("垃圾/回收檢查")
                    if "晨" in s_role: allowed.append("晨間打掃")
                    if "內掃" in s_role: allowed.append("內掃檢查")
                if not allowed: allowed = ["內掃檢查"]
            
                s_classes = []
                if scope_col and str(row[scope_col]):
                    raw = str(row[scope_col])
                    s_classes = [c.strip() for c in raw.replace("、", ";").replace(",", ";").split(";") if c.strip()]
                prefix = s_id[0] if len(s_id) > 0 else "X"
                inspectors.append({"label": f"學號: {s_id}", "allowed_roles": allowed, "assigned_classes": s_classes, "id_prefix": prefix})
        return inspectors if inspectors else default
    except: return default

@st.cache_data(ttl=60)
def get_daily_duty(target_date):
    ws = get_worksheet(SHEET_TABS["duty"])
    if not ws: return [], "error"
    try:
        df = pd.DataFrame(ws.get_all_records())
        if df.empty: return [], "no_data"
        date_col = next((c for c in df.columns if "日期" in c), None)
        id_col = next((c for c in df.columns if "學號" in c), None)
        loc_col = next((c for c in df.columns if "地點" in c), None)
        if date_col and id_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.date
            t_date = target_date if isinstance(target_date, date) else target_date.date()
            today_df = df[df[date_col] == t_date]
            res = []
            for _, row in today_df.iterrows():
                res.append({"學號": clean_id(row[id_col]), "掃地區域": str(row[loc_col]).strip() if loc_col else "", "已完成打掃": False})
            return res, "success"
        return [], "missing_cols"
    except: return [], "error"

@st.cache_data(ttl=21600)
def load_settings():
    ws = get_worksheet(SHEET_TABS["settings"])
    config = {"semester_start": "2025-08-25"}
    if ws:
        try:
            data = ws.get_all_values()
            for row in data:
                if len(row)>=2 and row[0] == "semester_start": config["semester_start"] = row[1]
        except: pass
    return config

def save_setting(key, val):
    ws = get_worksheet(SHEET_TABS["settings"])
    if ws:
        try:
            cell = ws.find(key)
            if cell: ws.update_cell(cell.row, cell.col+1, val)
            else: ws.append_row([key, val])
            st.cache_data.clear()
            return True
        except: return False
    return False

def send_bulk_emails(email_list):
    if "system_config" not in st.secrets:
        return 0, "Secrets 未設定"
    
    sender_email = st.secrets["system_config"].get("smtp_email")
    sender_password = st.secrets["system_config"].get("smtp_password")
    if not sender_email or not sender_password: return 0, "Secrets 未設定 Email"

    sent_count = 0
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        for item in email_list:
            try:
                msg = MIMEMultipart()
                msg['From'] = sender_email
                msg['To'] = item['email']
                msg['Subject'] = item['subject']
                msg.attach(MIMEText(item['body'], 'plain'))
                server.sendmail(sender_email, item['email'], msg.as_string())
                sent_count += 1
            except Exception as inner_e:
                print(f"個別寄送失敗: {inner_e}")
        server.quit()
        return sent_count, "發送作業結束"
    except Exception as e:
        return sent_count, str(e)

def check_duplicate_record(df, check_date, inspector, role, target_class=None):
    if df.empty: return False
    try:
        df["日期Str"] = df["日期"].astype(str)
        check_date_str = str(check_date)
        mask = (df["日期Str"] == check_date_str) & (df["檢查人員"] == inspector) & (df["評分項目"] == role)
        if target_class: mask = mask & (df["班級"] == target_class)
        return not df[mask].empty
    except: return False

# ==========================================
# 5. 主程式介面
# ==========================================

# 讀取設定與資料
SYSTEM_CONFIG = load_settings()
ROSTER_DICT = load_roster_dict()
INSPECTOR_LIST = load_inspector_list()
TEACHER_MAILS = load_teacher_emails()

all_classes, structured_classes = load_sorted_classes()
if not all_classes:
    all_classes = ["測試班級"]
    structured_classes = [{"grade": "其他", "name": "測試班級"}]

grades = sorted(list(set([c["grade"] for c in structured_classes])))

def get_week_num(d):
    try:
        start = datetime.strptime(SYSTEM_CONFIG["semester_start"], "%Y-%m-%d").date()
        if isinstance(d, datetime): d = d.date()
        return max(0, ((d - start).days // 7) + 1)
    except: return 0

now_tw = datetime.now(TW_TZ)
today_tw = now_tw.date()

st.sidebar.title("🏫 功能選單")
app_mode = st.sidebar.radio("請選擇模式", ["糾察底家👀", "班級負責人🥸", "組長ㄉ窩💃"])

with st.sidebar.expander("🔧 系統連線診斷", expanded=False):
    if get_gspread_client(): 
        st.success("✅ Google Sheets 連線正常")
    else: 
        st.error("❌ Sheets 連線失敗")
        
    if "gcp_service_account" in st.secrets: 
        st.success("✅ GCP 憑證已讀取")
    else: 
        st.error("⚠️ 未設定 GCP Service Account")
        
    if "system_config" in st.secrets and "drive_folder_id" in st.secrets["system_config"]:
        st.success("✅ Drive 資料夾 ID 已設定")
    else:
        st.warning("⚠️ 未設定 Drive 資料夾 ID")

# --- 模式1: 糾察評分 ---
if app_mode == "糾察底家👀":
    st.title("📝 衛生糾察評分系統")
    if "team_logged_in" not in st.session_state: st.session_state["team_logged_in"] = False
    
    if not st.session_state["team_logged_in"]:
        with st.expander("🔐 身份驗證", expanded=True):
            input_code = st.text_input("請輸入隊伍通行碼", type="password")
            if st.button("登入"):
                if input_code == st.secrets["system_config"]["team_password"]:
                    st.session_state["team_logged_in"] = True
                    st.rerun()
                else: st.error("通行碼錯誤")
    
    if st.session_state["team_logged_in"]:
        prefixes = sorted(list(set([p["id_prefix"] for p in INSPECTOR_LIST])))
        prefix_labels = [f"{p}開頭" for p in prefixes]
        if not prefix_labels: st.warning("找不到糾察名單，請通知老師在後台建立名單 (Sheet: inspectors)。")
        else:
            selected_prefix_label = st.radio("步驟 1：選擇開頭", prefix_labels, horizontal=True)
            selected_prefix = selected_prefix_label[0]
            filtered_inspectors = [p for p in INSPECTOR_LIST if p["id_prefix"] == selected_prefix]
            inspector_name = st.radio("步驟 2：點選身份", [p["label"] for p in filtered_inspectors])
            current_inspector_data = next((p for p in INSPECTOR_LIST if p["label"] == inspector_name), None)
            allowed_roles = current_inspector_data.get("allowed_roles", ["內掃檢查"])
            
            allowed_roles = [r for r in allowed_roles if r != "晨間打掃"]
            if not allowed_roles: allowed_roles = ["內掃檢查"] 
            assigned_classes = current_inspector_data.get("assigned_classes", [])
            
            st.markdown("---")
            col_date, col_role = st.columns(2)
            input_date = col_date.date_input("檢查日期", today_tw)
            if len(allowed_roles) > 1: role = col_role.radio("請選擇檢查項目", allowed_roles, horizontal=True)
            else: role = allowed_roles[0]
            col_role.info(f"📋 您的負責項目：**{role}**")
            
            week_num = get_week_num(input_date)
            st.caption(f"📅 第 {week_num} 週")
            main_df = load_main_data()

            if role == "垃圾/回收檢查":
                st.info("🗑️ 全校垃圾檢查 (每日每班上限扣2分)")
                trash_cat = st.radio("違規項目", ["一般垃圾", "紙類", "網袋", "其他回收"], horizontal=True)
                with st.form("trash_form"):
                    t_data = [{"班級": c, "無簽名": False, "無分類": False} for c in all_classes]
                    edited_t_df = st.data_editor(pd.DataFrame(t_data), hide_index=True, height=400, use_container_width=True)
                    if st.form_submit_button("送出"):
                        base = {"日期": input_date, "週次": week_num, "檢查人員": inspector_name, "登錄時間": now_tw.strftime("%Y-%m-%d %H:%M:%S"), "修正": False}
                        cnt = 0
                        for _, row in edited_t_df.iterrows():
                            vios = []
                            if row["無簽名"]: vios.append("無簽名")
                            if row["無分類"]: vios.append("無分類")
                            if vios:
                                save_entry({**base, "班級": row["班級"], "評分項目": role, "垃圾原始分": len(vios), "備註": f"{trash_cat}-{'、'.join(vios)}", "違規細項": trash_cat})
                                cnt += 1
                        st.success(f"已排入背景處理： {cnt} 班" if cnt else "無違規"); st.rerun()
            else:
                st.markdown("### 🏫 選擇受檢班級")
                if assigned_classes:
                    radio_key = f"radio_assigned_{inspector_name}"
                    selected_class = st.radio(
                        "請點選您的負責班級", 
                        assigned_classes, 
                        key=radio_key
                    )
                else:
                    g = st.radio("步驟 A: 選擇年級", grades, horizontal=True, key="radio_grade_select")
                    filtered_classes = [c["name"] for c in structured_classes if c["grade"] == g]
                    
                    if not filtered_classes:
                        st.warning("⚠️ 此年級無班級資料")
                        selected_class = None
                    else:
                        selected_class = st.radio(
                            "步驟 B: 選擇班級", 
                            filtered_classes, 
                            horizontal=True,
                            key=f"radio_class_select_{g}"
                        )
        
                if selected_class:
                    st.markdown(f"👉 目前鎖定評分對象： **<span style='color:red;font-size:1.2em'>{selected_class}</span>**", unsafe_allow_html=True)
                    if check_duplicate_record(main_df, input_date, inspector_name, role, selected_class):
                            st.warning(f"⚠️ 注意：您今天已經評過「{selected_class}」了！")
                    st.info(f"📍 正在評分：**{selected_class}**")
                    
                    day_key = f"{str(input_date)}|{inspector_name}|{role}"
                    if "scored_map" not in st.session_state:
                        st.session_state["scored_map"] = {}
                    scored_today = st.session_state["scored_map"].setdefault(day_key, set())

                    if selected_class in scored_today:
                        st.success(f"✅ 今日「{selected_class}」已評分（本次登入期間）")

                    form_id = f"scoring_form_{str(input_date)}_{inspector_name}_{role}_{selected_class}"

                    with st.form(form_id, clear_on_submit=True):
                        in_s = 0
                        out_s = 0
                        ph_c = 0
                        note = ""

                        result_key = f"result_{form_id}"
                        note_key = f"note_{form_id}"
                        phone_key = f"phone_{form_id}"
                        in_key = f"in_{form_id}"
                        out_key = f"out_{form_id}"
                        fix_key = f"fix_{form_id}"
                        files_key = f"files_{form_id}"

                        if role == "內掃檢查":
                            result = st.radio("結果", ["❌ 違規", "✨ 乾淨"], horizontal=True, key=result_key)
                            if result == "❌ 違規":
                                in_s = st.number_input("內掃扣分 (上限2分)", min_value=0, max_value=2, value=0, step=1, key=in_key)
                                note = st.text_input("說明", placeholder="黑板未擦", key=note_key)
                                ph_c = st.number_input("手機人數 (無上限)", min_value=0, value=0, step=1, key=phone_key)
                            else:
                                note = "【優良】"

                        elif role == "外掃檢查":
                            result = st.radio("結果", ["❌ 違規", "✨ 乾淨"], horizontal=True, key=result_key)
                            if result == "❌ 違規":
                                out_s = st.number_input("外掃扣分 (上限2分)", min_value=0, max_value=2, value=0, step=1, key=out_key)
                                note = st.text_input("說明", placeholder="走廊垃圾", key=note_key)
                                ph_c = st.number_input("手機人數 (無上限)", min_value=0, value=0, step=1, key=phone_key)
                            else:
                                note = "【優良】"

                        is_fix = st.checkbox("🚩 修正單", key=fix_key)
                        files = st.file_uploader("照片(自動上傳雲端)", accept_multiple_files=True, type=["jpg", "jpeg", "png"], key=files_key)

                        if st.form_submit_button("送出"):
                            if files and len(files) > 4:
                                st.error("❌ 一次最多只能上傳 4 張照片，請刪減後再送出。")
                            else:
                                ok = save_entry(
                                    {"日期": input_date, "週次": week_num, "檢查人員": inspector_name, "登錄時間": now_tw.strftime("%Y-%m-%d %H:%M:%S"),
                                        "修正": is_fix, "班級": selected_class, "評分項目": role, "內掃原始分": in_s, "外掃原始分": out_s, "手機人數": ph_c, "備註": note},
                                    uploaded_files=files
                                )
                                if ok:
                                    scored_today.add(selected_class)
                                    st.toast(f"✅ 已成功送出：{selected_class}（照片已上傳雲端）")
                                    time.sleep(1) # 讓 toast 顯示
                                    st.rerun()

# --- 模式2: 衛生股長 ---
elif app_mode == "班級負責人🥸":
    st.title("🔎 班級成績查詢")
    df = load_main_data()
    
    appeals_df = load_appeals()
    appeal_map = {}
    if not appeals_df.empty:
        for _, a_row in appeals_df.iterrows():
            rid = str(a_row.get("對應紀錄ID", "")).strip()
            if rid:
                appeal_map[rid] = a_row.get("處理狀態", "待處理")

    if not df.empty:
        st.write("請依照步驟選擇：")
        g = st.radio("步驟 1：選擇年級", grades, horizontal=True)
        class_options = [c["name"] for c in structured_classes if c["grade"] == g]
        
        if not class_options:
            st.warning("查無此年級班級")
            cls = None
        else:
            cls = st.radio("步驟 2：選擇班級", class_options, horizontal=True)

        st.divider()
        
        if cls:
            c_df = df[df["班級"] == cls].sort_values("登錄時間", ascending=False)
            three_days_ago = date.today() - timedelta(days=3)
            
            if not c_df.empty:
                st.subheader(f"📊 {cls} 近期紀錄與申訴狀態")
        
                for idx, r in c_df.iterrows():
                    total_raw = r['內掃原始分']+r['外掃原始分']+r['垃圾原始分']+r['晨間打掃原始分']
                    phone_msg = f" | 📱手機: {r['手機人數']}" if r['手機人數'] > 0 else ""
                    
                    record_id = str(r['紀錄ID']).strip()
                    appeal_status = appeal_map.get(record_id, None)
            
                    status_icon = ""
                    if appeal_status == "已核可": status_icon = "✅ [申訴成功] "
                    elif appeal_status == "已駁回": status_icon = "🚫 [申訴駁回] "
                    elif appeal_status == "待處理": status_icon = "⏳ [審核中] "
                    elif str(r['修正']) == "TRUE": status_icon = "🛠️ [已修正] "

                    week_val = r.get('週次', 0)
                    week_label = f"[第{week_val}週] " if week_val and str(week_val) != "0" else ""

                    title_text = f"{status_icon}{week_label}{r['日期']} - {r['評分項目']} (扣分: {total_raw}){phone_msg}"
                    
                    with st.expander(title_text):
                        if appeal_status == "已核可":
                            st.success("🎉 恭喜！衛生組已核可您的申訴，本筆扣分已撤銷。")
                        elif appeal_status == "已駁回":
                            st.error("🛑 很遺憾，您的申訴已被駁回，維持原判。")
                        elif appeal_status == "待處理":
                            st.warning("⏳ 申訴案件目前正在排隊審核中，請耐心等候...")

                        st.write(f"📝 說明: {r['備註']}")
                        st.caption(f"檢查人員: {r['檢查人員']}")
                        
                        raw_photo_path = str(r.get("照片路徑", "")).strip()
                        if raw_photo_path and raw_photo_path.lower() != "nan":
                            path_list = [p.strip() for p in raw_photo_path.split(";") if p.strip()]
                            valid_photos = [p for p in path_list if p != "UPLOAD_FAILED" and (p.startswith("http") or os.path.exists(p))]
                            if valid_photos:
                                captions = [f"違規照片 ({i+1})" for i in range(len(valid_photos))]
                                st.image(valid_photos, caption=captions, width=300)
                            elif "UPLOAD_FAILED" in path_list: st.warning("⚠️ 照片上傳失敗")

                        if total_raw > 2 and r['晨間打掃原始分'] == 0:
                            st.info("💡系統提示：單項每日扣分上限為 2 分 (手機、晨掃除外)，最終成績將由後台自動計算上限。")

                        record_date_obj = pd.to_datetime(r['日期']).date() if isinstance(r['日期'], str) else r['日期']
        
                        if appeal_status:
                            pass 
                        elif record_date_obj >= three_days_ago and (total_raw > 0 or r['手機人數'] > 0):
                            st.markdown("---")
                            st.markdown("#### 🚨 我要申訴")
                            form_key = f"appeal_form_{r['紀錄ID']}_{idx}"
                            with st.form(form_key):
                                reason = st.text_area("申訴理由", height=80, placeholder="詳細說明...")
                                proof_file = st.file_uploader("上傳佐證 (必填)", type=["jpg", "png", "jpeg"], key=f"file_{idx}")
                                if st.form_submit_button("提交申訴"):
                                    if not reason or not proof_file: 
                                        st.error("❌ 請填寫理由並上傳照片")
                                    else:
                                        appeal_entry = {
                                            "申訴日期": str(date.today()), 
                                            "班級": cls, 
                                            "違規日期": str(r["日期"]),
                                            "違規項目": f"{r['評分項目']} ({r['備註']})", 
                                            "原始扣分": str(total_raw),
                                            "申訴理由": reason, 
                                            "處理狀態": "待處理",
                                            "登錄時間": datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S"),
                                            "對應紀錄ID": r['紀錄ID']
                                        }
                                        if save_appeal(appeal_entry, proof_file): 
                                            st.success("✅ 申訴已提交！請重新整理頁面查看狀態。")
                                            time.sleep(1.5)
                                            st.rerun()
                                        else: 
                                            st.error("提交失敗")
                        elif total_raw > 0 and not appeal_status:
                            st.caption("⏳ Sorry Bro，已超過 3 天申訴期限。")
            else:
                st.info("🎉 最近沒有違規紀錄，尼們班很讚！")

# --- 模式3: 後台 ---
elif app_mode == "組長ㄉ窩💃":
    st.title("⚙️ 管理後台")
    
    # 確保 Worker 啟動以更新狀態
    ensure_worker_started()
    
    # [SRE] 監控面板
    metrics = get_queue_metrics()
    q_count = metrics["pending"] + metrics["retry"]
    oldest_age = metrics["oldest_pending_sec"]
    recent_errs = metrics["recent_errors"]
    
    with st.container(border=True):
        st.write("#### 📡 SRE 監控面板")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pending Queue", q_count, delta="Safe" if q_count < 50 else "High Load", delta_color="inverse")
        m2.metric("Running Tasks", metrics["running"])
        m3.metric("Failed Tasks", metrics["failed"])
        m4.metric("Oldest Task Age", f"{int(oldest_age)}s", delta="Lagging" if oldest_age > 300 else "Normal", delta_color="inverse")

        if q_count > 100:
            st.error(f"🔥 **系統過載警告**：積壓 {q_count} 筆資料！")
        elif oldest_age > 300:
            st.warning(f"🐢 **寫入延遲警告**：滯留 {int(oldest_age)} 秒。")
        
        if recent_errs:
            with st.expander("查看最近錯誤日誌 (Top 5)"):
                for err_msg, ts in recent_errs:
                    st.error(f"[{ts}] {err_msg}")

    pwd = st.text_input("管理密碼", type="password")
    if pwd == st.secrets["system_config"]["admin_password"]:
        monitor_tab, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "👀 進度監控", "📊 成績總表", "📝 扣分明細", "📧 寄送通知", 
            "📣 申訴審核", "⚙️ 系統設定", "📄 名單更新", "🧹 晨掃點名"
        ])
        
        with monitor_tab:
            st.subheader("🕵️ 今日評分進度監控")
            
            # 1. 設定監控日期 (預設今天)
            monitor_date = st.date_input("監控日期", today_tw, key="monitor_date")
            st.caption(f"📅 檢查目標：{monitor_date} (建議於 16:30 前完成)")

            # 2. 準備資料
            df = load_main_data()
            
            # 取得今日已回報的人員名單 (去重)
            submitted_names = set()
            if not df.empty:
                # 轉成字串比對，確保格式一致
                df["日期Str"] = df["日期"].astype(str)
                target_str = str(monitor_date)
                today_records = df[df["日期Str"] == target_str]
                submitted_names = set(today_records["檢查人員"].unique())

            # 3. 分類檢查人員 (一般評分 vs 機動/組長)
            # 邏輯：有分配 "assigned_classes" 的是班級評分員，沒有的是機動
            regular_inspectors = [] # 有固定班級
            mobile_inspectors = []  # 機動/組長 (無固定班級)

            for p in INSPECTOR_LIST:
                p_name = p["label"]
                # 判斷是否為機動：看 assigned_classes 是否為空
                is_mobile = len(p.get("assigned_classes", [])) == 0
                
                # 建立狀態物件
                status_obj = {
                    "name": p_name,
                    "role_desc": "、".join(p.get("allowed_roles", [])),
                    "done": p_name in submitted_names
                }

                if is_mobile:
                    mobile_inspectors.append(status_obj)
                else:
                    regular_inspectors.append(status_obj)

            # 4. 顯示儀表板
            # --- 計算數據 ---
            total_regular = len(regular_inspectors)
            done_regular = sum(1 for x in regular_inspectors if x["done"])
            
            total_mobile = len(mobile_inspectors)
            done_mobile = sum(1 for x in mobile_inspectors if x["done"])

            # --- 顯示進度條 ---
            c1, c2 = st.columns(2)
            with c1:
                st.metric("班級評分員完成率", f"{done_regular}/{total_regular}", delta=f"尚缺 {total_regular - done_regular} 人")
                if total_regular > 0:
                    st.progress(done_regular / total_regular)
            with c2:
                st.metric("機動/組長完成率", f"{done_mobile}/{total_mobile}", delta=f"尚缺 {total_mobile - done_mobile} 人")
                if total_mobile > 0:
                    st.progress(done_mobile / total_mobile)

            st.divider()

            # 5. 顯示未完成名單 (左右並列)
            col_reg, col_mob = st.columns(2)
            
            with col_reg:
                st.write("#### 🔴 班級評分員 (未完成)")
                missing_reg = [x for x in regular_inspectors if not x["done"]]
                if missing_reg:
                    for p in missing_reg:
                        st.error(f"❌ {p['name']}")
                else:
                    st.success("🎉 全員完成！")

                with st.expander("查看已完成名單"):
                    for p in regular_inspectors:
                        if p["done"]: st.write(f"✅ {p['name']}")

            with col_mob:
                st.write("#### 🟠 機動/組長 (未完成)")
                st.caption("機動人員若今日無違規需登記，可能也不會送出資料，請斟酌參考。")
                missing_mob = [x for x in mobile_inspectors if not x["done"]]
                if missing_mob:
                    for p in missing_mob:
                        # 機動組顯示負責項目，方便組長判斷他今天是不是真的沒事
                        st.warning(f"⚠️ {p['name']} \n   (負責: {p['role_desc']})")
                else:
                    st.success("🎉 全員完成！")

                with st.expander("查看已完成名單"):
                    for p in mobile_inspectors:
                        if p["done"]: st.write(f"✅ {p['name']}")

        with tab1: # 成績總表
            st.subheader("成績總表")
            df = load_main_data()
            all_classes_df = pd.DataFrame(all_classes, columns=["班級"])
            if not df.empty:
                valid_weeks = sorted(df[df["週次"]>0]["週次"].unique())
                selected_weeks = st.multiselect("選擇週次", valid_weeks, default=valid_weeks[-1:] if valid_weeks else [], key='week_select_summary')
                if selected_weeks:
                    wdf = df[df["週次"].isin(selected_weeks)].copy()
                    daily_agg = wdf.groupby(["日期", "班級"]).agg({
                        "內掃原始分": "sum", "外掃原始分": "sum", "垃圾原始分": "sum",
                        "晨間打掃原始分": "sum", "手機人數": "sum"
                    }).reset_index()
                    daily_agg["內掃結算"] = daily_agg["內掃原始分"].apply(lambda x: min(x, 2))
                    daily_agg["外掃結算"] = daily_agg["外掃原始分"].apply(lambda x: min(x, 2))
                    daily_agg["垃圾結算"] = daily_agg["垃圾原始分"].apply(lambda x: min(x, 2))
                    daily_agg["每日總扣分"] = (daily_agg["內掃結算"] + daily_agg["外掃結算"] + 
                                            daily_agg["垃圾結算"] + daily_agg["晨間打掃原始分"] + daily_agg["手機人數"])
                    violation_report = daily_agg.groupby("班級").agg({
                        "內掃結算": "sum", "外掃結算": "sum", "垃圾結算": "sum",
                        "晨間打掃原始分": "sum", "手機人數": "sum", "每日總扣分": "sum"
                    }).reset_index()
                    violation_report.columns = ["班級", "內掃扣分", "外掃扣分", "垃圾扣分", "晨掃扣分", "手機扣分", "總扣分"]
                    final_report = pd.merge(all_classes_df, violation_report, on="班級", how="left").fillna(0)
                    final_report["總成績"] = 90 - final_report["總扣分"]
                    final_report = final_report.sort_values("總成績", ascending=False)
                    st.dataframe(final_report, column_config={
                        "總成績": st.column_config.ProgressColumn("總成績", format="%d", min_value=60, max_value=90),
                        "總扣分": st.column_config.NumberColumn("總扣分", format="%d 分")
                    }, use_container_width=True)
                    csv = final_report.to_csv(index=False).encode('utf-8-sig')
                    st.download_button("📥 下載 (CSV)", csv, f"report_weeks_{selected_weeks}.csv")
                else: st.info("請選擇週次")
            else: st.warning("無資料")

        with tab2: # 詳細明細
            st.subheader("📝 違規詳細流水帳")
            df = load_main_data()
            if not df.empty:
                valid_weeks = sorted(df[df["週次"]>0]["週次"].unique())
                s_weeks = st.multiselect("選擇週次", valid_weeks, default=valid_weeks[-1:] if valid_weeks else [], key='week_select_detail')
                if s_weeks:
                    detail_df = df[df["週次"].isin(s_weeks)].copy()
                    detail_df["該筆扣分"] = detail_df["內掃原始分"] + detail_df["外掃原始分"] + detail_df["垃圾原始分"] + detail_df["晨間打掃原始分"] + detail_df["手機人數"]
                    detail_df = detail_df[detail_df["該筆扣分"] > 0]
                    display_cols = ["日期", "班級", "評分項目", "該筆扣分", "備註", "檢查人員", "違規細項", "紀錄ID"]
                    detail_df = detail_df[display_cols].sort_values(["日期", "班級"])
                    st.dataframe(detail_df, use_container_width=True)
                    csv_detail = detail_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button("📥 下載 (CSV)", csv_detail, f"detail_log_{s_weeks}.csv")
                else: st.info("請選擇週次")
            else: st.info("無資料")

        with tab3: # 寄送通知
            st.subheader("📧 每日違規通知")
            target_date = st.date_input("選擇日期", today_tw)
            if "mail_preview" not in st.session_state: st.session_state.mail_preview = None
            if st.button("🔍 統整當日違規"):
                df = load_main_data()
                try:
                    df["日期Obj"] = pd.to_datetime(df["日期"], errors='coerce').dt.date
                    day_df = df[df["日期Obj"] == target_date]
                except: day_df = pd.DataFrame()
                if not day_df.empty:
                    stats = day_df.groupby("班級")[["內掃原始分", "外掃原始分", "垃圾原始分", "晨間打掃原始分", "手機人數"]].sum().reset_index()
                    stats["內掃"] = stats["內掃原始分"].clip(upper=2)
                    stats["外掃"] = stats["外掃原始分"].clip(upper=2)
                    stats["垃圾"] = stats["垃圾原始分"].clip(upper=2)
                    stats["當日總扣分"] = stats["內掃"] + stats["外掃"] + stats["垃圾"] + stats["晨間打掃原始分"] + stats["手機人數"]
                    violation_classes = stats[stats["當日總扣分"] > 0]
                    if not violation_classes.empty:
                        preview_data = []
                        for _, row in violation_classes.iterrows():
                            cls_name = row["班級"]
                            t_info = TEACHER_MAILS.get(cls_name, {})
                            t_name = t_info.get('name', "❌ 缺名單")
                            t_email = t_info.get('email', "❌ 無法寄送")
                            status = "準備寄送" if "@" in t_email else "異常"
                            preview_data.append({"班級": cls_name, "當日總扣分": row["當日總扣分"], "導師姓名": t_name, "收件信箱": t_email, "狀態": status})
                        st.session_state.mail_preview = pd.DataFrame(preview_data)
                        st.success(f"找到 {len(violation_classes)} 筆違規班級")
                    else: st.session_state.mail_preview = None; st.info("今日無違規")
                else: st.session_state.mail_preview = None; st.info("今日無資料")

            if st.session_state.mail_preview is not None:
                st.write("### 📨 寄送預覽清單"); st.dataframe(st.session_state.mail_preview)
                if st.button("🚀 一鍵寄出！"):
                    mail_queue_list = []
                    for _, row in st.session_state.mail_preview.iterrows():
                        if row["狀態"] == "準備寄送":
                            subject = f"🔔({target_date})衛生組評分報表 - {row['班級']}"
                            content = f"老師您好：\n\n這是來自衛生組系統自動發送之每日報表。\n依據今日({target_date}) 的評分記錄\n衛生評分總扣分為：{row['當日總扣分']} 分。\n這邊要拜託老師鼓勵及提醒負責學生，一起來為學校的環境努力一下🪄\n真心感恩辛苦的導師\n\n如有疑問可至衛生組評分系統查詢扣分細節哦!\n https://clvshygiene.streamlit.app/ \n\n學務處衛生組敬上"
                            mail_queue_list.append({'email': row["收件信箱"], 'subject': subject, 'body': content})
                    
                    if mail_queue_list:
                        with st.spinner("📧 正在建立 SMTP 連線並批次寄送..."):
                            count, msg = send_bulk_emails(mail_queue_list)
                            if count > 0: st.success(f"✅ 成功寄出 {count} 封信件！ ({msg})")
                            else: st.error(f"❌ 寄送失敗: {msg}")
                        st.session_state.mail_preview = None
                    else: st.warning("沒有可寄送的對象")

        with tab4: # 申訴審核
            st.subheader("📣 申訴案件審核")
            appeals_df = load_appeals()
            pending = appeals_df[appeals_df["處理狀態"] == "待處理"]
            if not pending.empty:
                st.info(f"待審核: {len(pending)} 件")
                for idx, row in pending.iterrows():
                    with st.container(border=True):
                        c1, c2 = st.columns([2, 1])
                        with c1:
                            st.markdown(f"**{row['班級']}** | {row['違規項目']} | 扣 {row['原始扣分']} 分")
                            st.markdown(f"理由：{row['申訴理由']}")
                        with c2:
                            url = row.get("佐證照片", "")
                            if url and url != "UPLOAD_FAILED": st.image(url, width=150)
                        b1, b2 = st.columns(2)
                        if b1.button("✅ 核可", key=f"ok_{idx}"):
                            succ, msg = update_appeal_status(idx, "已核可", row["對應紀錄ID"])
                            if succ: st.success("已核可"); time.sleep(1); st.rerun()
                        
                        if b2.button("🚫 駁回", key=f"ng_{idx}"):
                            succ, msg = update_appeal_status(idx, "已駁回", row["對應紀錄ID"])
                            if succ: st.warning("已駁回"); time.sleep(1); st.rerun()
            else: st.success("無待審核案件")
            with st.expander("歷史案件"): st.dataframe(appeals_df[appeals_df["處理狀態"] != "待處理"])

        with tab5: # 系統設定
            st.subheader("⚙️ 系統設定")
            curr = SYSTEM_CONFIG.get("semester_start", "2025-08-25")
            nd = st.date_input("開學日", datetime.strptime(curr, "%Y-%m-%d").date())
            if st.button("更新開學日"): save_setting("semester_start", str(nd)); st.success("已更新")
            st.divider()
            st.markdown("### 🗑️ 資料維護")
            df = load_main_data()
            if not df.empty:
                del_mode = st.radio("刪除模式", ["單筆刪除", "日期區間刪除"])
                if del_mode == "單筆刪除":
                    df_display = df.sort_values("登錄時間", ascending=False).head(50)
                    opts = {r['紀錄ID']: f"{r['日期']} | {r['班級']} | {r['評分項目']} (ID:{r['紀錄ID']})" for _, r in df_display.iterrows()}
                    sel_ids = st.multiselect("選擇要刪除的紀錄", list(opts.keys()), format_func=lambda x: opts[x], key='del_multiselect')
                    if st.button("🗑️ 確認刪除"):
                        if delete_rows_by_ids(sel_ids): st.success("刪除成功"); st.rerun()
                elif del_mode == "日期區間刪除":
                    c1, c2 = st.columns(2)
                    d_start = c1.date_input("開始"); d_end = c2.date_input("結束")
                    if st.button("⚠️ 確認刪除區間資料"):
                        df["d_tmp"] = pd.to_datetime(df["日期"], errors='coerce').dt.date
                        target_ids = df[(df["d_tmp"] >= d_start) & (df["d_tmp"] <= d_end)]["紀錄ID"].tolist()
                        if target_ids:
                            if delete_rows_by_ids(target_ids): st.success(f"已刪除 {len(target_ids)} 筆"); st.rerun()
                        else: st.warning("無資料")
            else: st.info("無資料")

        with tab6:
            st.info("請至 Google Sheets 修改名單")
            if st.button("🔄 重新讀取快取"): st.cache_data.clear(); st.success("OK")
            st.markdown(f"[開啟試算表]({SHEET_URL})")

        with tab7: # 晨掃管理
            st.subheader("🧹 晨掃評分")
            m_date = st.date_input("日期", today_tw, key="m_d")
            m_week = get_week_num(m_date)
            duty_list, status = get_daily_duty(m_date)
            if status == "success":
                st.write(f"應到: {len(duty_list)} 人")
                with st.form("m_form"):
                    edited = st.data_editor(pd.DataFrame(duty_list), hide_index=True, use_container_width=True)
                    score = st.number_input("扣分", min_value=1, value=1)
                    if st.form_submit_button("送出"):
                        base = {"日期": m_date, "週次": m_week, "檢查人員": "衛生組", "登錄時間": now_tw.strftime("%Y-%m-%d %H:%M:%S"), "修正": False}
                        cnt = 0
                        for _, r in edited[edited["已完成打掃"] == False].iterrows():
                            tid = clean_id(r["學號"])
                            cls = ROSTER_DICT.get(tid, f"查無({tid})")
                            save_entry({**base, "班級": cls, "評分項目": "晨間打掃", "晨間打掃原始分": score, "備註": f"未到-學號:{tid}", "晨掃未到者": tid})
                            cnt += 1
                        st.success(f"已排入背景：{cnt} 人"); st.rerun()
            else: st.warning(f"無輪值資料 ({status})")

    else: st.error("密碼錯誤")