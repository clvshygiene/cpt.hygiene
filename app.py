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

# [SRE] 移除 PIL 以避免 Segfault，犧牲壓縮換取穩定性
# from PIL import Image 

# --- 1. 網頁設定 ---
st.set_page_config(page_title="中壢家商，衛愛而生", layout="wide", page_icon="🧹")

# ==========================================
# 0. 基礎設定與常數
# ==========================================
TW_TZ = pytz.timezone('Asia/Taipei')
MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 放寬到 15MB，因為不壓縮了

# [SRE] 使用系統暫存目錄
TEMP_DIR = tempfile.gettempdir()
QUEUE_DB_PATH = os.path.join(TEMP_DIR, "task_queue_v11_hybrid.db")
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
# 1. 工具函式
# ==========================================

def clean_id(val):
    try:
        if pd.isna(val) or val == "": return ""
        return str(int(float(val))).strip()
    except: return str(val).strip()

def execute_with_retry(func, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(base_delay + random.uniform(0, 1))
            else:
                raise e

# ==========================================
# 2. Google API 連線 (分離模式)
# ==========================================

# --- 前端 UI 專用 (有 Cache) ---
@st.cache_resource
def get_credentials_cached():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    if "gcp_service_account" not in st.secrets: return None
    creds_dict = dict(st.secrets["gcp_service_account"])
    return ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)

@st.cache_resource
def get_gspread_client():
    try:
        creds = get_credentials_cached()
        return gspread.authorize(creds) if creds else None
    except: return None

@st.cache_resource(ttl=3600)
def get_spreadsheet_object():
    client = get_gspread_client()
    try: return client.open_by_url(SHEET_URL) if client else None
    except: return None

# --- 背景 Worker 專用 (無 Cache，避免 Context Error) ---
def get_raw_sheet_client():
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        return gspread.authorize(creds)
    except: return None

def get_raw_drive_service():
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        return build('drive', 'v3', credentials=creds, cache_discovery=False)
    except: return None

# ==========================================
# 3. 圖片上傳 (前景平行化)
# ==========================================

def upload_single_image(args):
    """單張圖片上傳邏輯"""
    file_bytes, filename = args
    service = get_raw_drive_service() # 每個 Thread 自己拿連線
    if not service: return None
    
    folder_id = st.secrets["system_config"].get("drive_folder_id")
    
    try:
        metadata = {'name': filename}
        if folder_id: metadata['parents'] = [folder_id]
        
        media = MediaIoBaseUpload(io.BytesIO(file_bytes), mimetype='image/jpeg', resumable=True)
        file = service.files().create(body=metadata, media_body=media, fields='id,webViewLink').execute()
        return file.get('webViewLink') or f"https://drive.google.com/file/d/{file.get('id')}/view"
    except Exception as e:
        print(f"[Upload Error] {e}")
        return None

def upload_images_hybrid(files_list, entry_data):
    """
    [Hybrid Mode] 使用 ThreadPoolExecutor(max_workers=2)
    在前景執行，確保拿到連結才繼續，但比單執行緒快。
    """
    if not files_list: return [], True

    tasks = []
    for i, up_file in enumerate(files_list):
        up_file.seek(0)
        raw = up_file.read() # 讀取為 Bytes
        
        safe_class = str(entry_data.get("班級", "unknown"))
        logical_fname = f"{entry_data.get('日期', '')}_{safe_class}_{i}.jpg"
        unique_prefix = f"{datetime.now(TW_TZ).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        drive_filename = f"{unique_prefix}_{logical_fname}"
        
        tasks.append((raw, drive_filename))

    uploaded_links = [None] * len(tasks)
    
    # 使用 2 個 Worker 平行上傳 (比 4 個安全，比 1 個快)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_to_idx = {executor.submit(upload_single_image, task): i for i, task in enumerate(tasks)}
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                link = future.result()
                uploaded_links[idx] = link
            except:
                uploaded_links[idx] = None

    if any(l is None for l in uploaded_links):
        return [], False # 嚴格模式：只要有一張失敗就全擋
    
    return uploaded_links, True

# ==========================================
# 4. SQLite 背景佇列 (Queue) - 復活 SRE 面板
# ==========================================

_db_lock = threading.Lock()

def get_db_connection():
    # 每次連線，確保 Thread Safe
    try:
        conn = sqlite3.connect(QUEUE_DB_PATH, check_same_thread=False, timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        return conn
    except: return None

def init_db():
    with _db_lock:
        conn = get_db_connection()
        if conn:
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON task_queue (status);")
            conn.close()

init_db()

def enqueue_task(task_type, payload):
    # 將資料寫入 SQLite (極快，不會卡 UI)
    task_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"
    js = json.dumps(payload, ensure_ascii=False)
    
    with _db_lock:
        conn = get_db_connection()
        if conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "INSERT INTO task_queue (id, task_type, created_ts, payload_json, status) VALUES (?, ?, ?, ?, 'PENDING')",
                    (task_id, task_type, now, js)
                )
                conn.commit()
                conn.close()
                ensure_worker_started() # 喚醒 Worker
                return True
            except: 
                conn.close()
    return False

def get_queue_metrics():
    # SRE 面板需要的數據
    metrics = {"pending": 0, "processed": 0, "failed": 0}
    with _db_lock:
        conn = get_db_connection()
        if conn:
            try:
                cur = conn.cursor()
                cur.execute("SELECT status, COUNT(*) FROM task_queue GROUP BY status")
                for s, c in cur.fetchall():
                    if s == 'PENDING': metrics["pending"] = c
                    elif s == 'DONE': metrics["processed"] = c
                    elif s == 'FAILED': metrics["failed"] = c
                conn.close()
            except: pass
    return metrics

# --- 背景 Worker (純淨版：不碰 Streamlit Context) ---

def worker_loop(stop_event):
    print("🚀 Background Worker Started")
    client = get_raw_sheet_client() # 預先建立連線
    
    while not stop_event.is_set():
        task = None
        conn = get_db_connection()
        if not conn:
            time.sleep(5)
            continue

        # 1. 領取任務 (Atomic Claim)
        try:
            with _db_lock:
                conn.execute("BEGIN IMMEDIATE")
                cur = conn.cursor()
                cur.execute("SELECT id, task_type, payload_json FROM task_queue WHERE status='PENDING' LIMIT 1")
                row = cur.fetchone()
                if row:
                    task = row
                    conn.execute("UPDATE task_queue SET status='RUNNING' WHERE id=?", (task[0],))
                    conn.commit()
                else:
                    conn.commit()
        except: pass
        finally: conn.close()

        if not task:
            time.sleep(2) # 沒事做就休息
            continue

        # 2. 執行任務 (寫入 Sheet)
        t_id, t_type, t_payload = task
        try:
            data = json.loads(t_payload)
            entry = data.get("entry")
            
            # 這裡重新取得 client 以防過期，並具備重試機制
            local_client = get_raw_sheet_client()
            sheet = local_client.open_by_url(SHEET_URL)
            
            target_tab = SHEET_TABS["main"] if t_type == "main_entry" else SHEET_TABS["appeals"]
            try:
                ws = sheet.worksheet(target_tab)
            except:
                ws = sheet.add_worksheet(target_tab, 100, 20)
                # 補表頭
                header = EXPECTED_COLUMNS if t_type == "main_entry" else APPEAL_COLUMNS
                ws.append_row(header)

            # 準備 Row
            row_vals = []
            cols = EXPECTED_COLUMNS if t_type == "main_entry" else APPEAL_COLUMNS
            for col in cols:
                val = entry.get(col, "")
                if isinstance(val, bool): val = str(val).upper()
                row_vals.append(val)
            
            ws.append_row(row_vals)
            
            # 3. 標記完成
            with _db_lock:
                c2 = get_db_connection()
                c2.execute("UPDATE task_queue SET status='DONE' WHERE id=?", (t_id,))
                c2.commit()
                c2.close()
            print(f"✅ Task {t_id} Done")

        except Exception as e:
            print(f"❌ Task {t_id} Failed: {e}")
            with _db_lock:
                c3 = get_db_connection()
                c3.execute("UPDATE task_queue SET status='FAILED', last_error=? WHERE id=?", (str(e), t_id))
                c3.commit()
                c3.close()

# --- 不死鳥機制 ---
_worker_thread = None
def ensure_worker_started():
    global _worker_thread
    if _worker_thread is None or not _worker_thread.is_alive():
        stop_ev = threading.Event()
        _worker_thread = threading.Thread(target=worker_loop, args=(stop_ev,), daemon=True)
        _worker_thread.start()

ensure_worker_started()

# ==========================================
# 5. 前端讀取與 UI
# ==========================================

@st.cache_data(ttl=60)
def load_main_data():
    ws_obj = get_worksheet(SHEET_TABS["main"]) # 使用 cached helper
    if not ws_obj: return pd.DataFrame(columns=EXPECTED_COLUMNS)
    try:
        return pd.DataFrame(ws_obj.get_all_records())
    except: return pd.DataFrame(columns=EXPECTED_COLUMNS)

def get_worksheet(tab_name):
    # 為了 UI 讀取方便的 Helper
    sheet = get_spreadsheet_object()
    if not sheet: return None
    try: return sheet.worksheet(tab_name)
    except: return None

# ==========================================
# 6. 主程式介面
# ==========================================

# 讀取必要設定
all_classes, _ = load_sorted_classes()
if not all_classes: all_classes = ["測試班級"]

st.sidebar.title("🏫 評分系統 (Hybrid Pro)")
app_mode = st.sidebar.radio("模式", ["評分輸入", "資料查詢", "後台監控"])

if app_mode == "評分輸入":
    st.title("📝 評分輸入")
    pwd = st.text_input("通行碼", type="password")
    if pwd == st.secrets["system_config"]["team_password"]:
        
        c1, c2 = st.columns(2)
        d_input = c1.date_input("日期", date.today())
        insp = c2.text_input("檢查人員", "衛生組")
        cls = st.selectbox("班級", all_classes)
        role = st.radio("項目", ["內掃檢查", "外掃檢查", "垃圾檢查"])
        
        with st.form("main_form"):
            score = st.number_input("扣分", min_value=0, step=1)
            note = st.text_input("說明")
            files = st.file_uploader("照片 (最多4張)", accept_multiple_files=True)
            
            if st.form_submit_button("送出"):
                entry = {
                    "日期": str(d_input),
                    "週次": get_week_num(d_input),
                    "班級": cls,
                    "評分項目": role,
                    "檢查人員": insp,
                    "內掃原始分": score if role=="內掃檢查" else 0,
                    "外掃原始分": score if role=="外掃檢查" else 0,
                    "垃圾原始分": score if role=="垃圾檢查" else 0,
                    "備註": note,
                    "登錄時間": datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S"),
                    "紀錄ID": f"{datetime.now(TW_TZ).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
                }
                
                # 1. 嚴格模式：照片必須先上傳成功 (前景平行處理)
                if files:
                    if len(files) > 4:
                        st.error("❌ 照片過多")
                        st.stop()
                    
                    with st.spinner("☁️ 正在極速上傳照片 (驗證中)..."):
                        links, ok = upload_images_hybrid(files, entry)
                        if not ok:
                            st.error("❌ 照片上傳失敗，為保全證據，本筆資料未送出。")
                            st.stop()
                        entry["照片路徑"] = ";".join(links)
                
                # 2. 資料寫入：丟給背景佇列 (秒回)
                if enqueue_task("main_entry", {"entry": entry}):
                    st.success("✅ 資料已排入佇列，將自動寫入試算表！")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ 系統繁忙 (DB Locked)，請重試")

elif app_mode == "後台監控":
    st.title("📡 SRE 監控面板")
    metrics = get_queue_metrics()
    c1, c2, c3 = st.columns(3)
    c1.metric("待處理 (Pending)", metrics["pending"])
    c2.metric("已完成 (Done)", metrics["processed"])
    c3.metric("失敗 (Failed)", metrics["failed"])
    
    if st.button("手動喚醒 Worker"):
        ensure_worker_started()
        st.toast("已發送喚醒訊號")

elif app_mode == "資料查詢":
    st.title("📊 資料查詢")
    df = load_main_data()
    st.dataframe(df)
