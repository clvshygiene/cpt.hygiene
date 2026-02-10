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
from PIL import Image  # 圖片處理核心套件

# --- 1. 網頁設定 ---
st.set_page_config(page_title="中壢家商，衛愛而生 V3", layout="wide", page_icon="🧹")

# --- 2. 核心參數與全域設定 ---
try:
    TW_TZ = pytz.timezone('Asia/Taipei')
    MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB
    QUEUE_DB_PATH = "task_queue_v4_wal.db"
    IMG_DIR = "evidence_photos"
    os.makedirs(IMG_DIR, exist_ok=True)
    
    # Google Sheet 設定
    SHEET_URL = "https://docs.google.com/spreadsheets/d/11BXtN3aevJls6Q2IR_IbT80-9XvhBkjbTCgANmsxqkg/edit"
    
    SHEET_TABS = {
        "main": "main_data", 
        "settings": "settings",
        "roster": "roster",
        "inspectors": "inspectors",
        "duty": "duty",
        "teachers": "teachers",
        "appeals": "appeals",
        "holidays": "holidays",
        "service_hours": "service_hours" # 服務時數記錄表
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
    # SRE Utils: 重試機制
    # ==========================================
    def execute_with_retry(func, max_retries=5, base_delay=1.0):
        for attempt in range(max_retries):
            try:
                time.sleep(0.3 + random.uniform(0, 0.2)) 
                return func()
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
            st.error("❌ 找不到 secrets 設定")
            return None
        creds_dict = dict(st.secrets["gcp_service_account"])
        return ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)

    def get_gspread_client():
        creds = get_credentials()
        return gspread.authorize(creds) if creds else None

    def get_drive_service():
        creds = get_credentials()
        return build('drive', 'v3', credentials=creds, cache_discovery=False) if creds else None

    def get_spreadsheet_object():
        client = get_gspread_client()
        if not client: return None
        try: return client.open_by_url(SHEET_URL)
        except Exception as e: 
            print(f"Spreadsheet Error: {e}")
            return None

    def get_worksheet(tab_name):
        max_retries = 3
        sheet = get_spreadsheet_object()
        if not sheet: return None
        for attempt in range(max_retries):
            try:
                try: return sheet.worksheet(tab_name)
                except gspread.WorksheetNotFound:
                    # 自動建立缺少的 sheet
                    cols = 20 if tab_name != "appeals" else 15
                    ws = sheet.add_worksheet(title=tab_name, rows=500, cols=cols)
                    if tab_name == "appeals": ws.append_row(APPEAL_COLUMNS)
                    if tab_name == "service_hours": ws.append_row(["日期", "學號", "班級", "類別", "時數", "紀錄ID"])
                    return ws
            except Exception as e:
                if "429" in str(e): 
                    time.sleep(2 * (attempt + 1))
                    continue
                else: return None
        return None

    def compress_image_bytes(file_bytes, quality=70):
        """Pillow 圖片壓縮：限制寬度 1600px 並轉 JPEG"""
        try:
            img = Image.open(io.BytesIO(file_bytes))
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
            folder_id = st.secrets["system_config"].get("drive_folder_id")
            metadata = {'name': filename, 'parents': [folder_id]}
            media = MediaIoBaseUpload(file_obj, mimetype='image/jpeg', resumable=True)
            file = service.files().create(body=metadata, media_body=media, fields='id', supportsAllDrives=True).execute()
            try: service.permissions().create(fileId=file.get('id'), body={'role': 'reader', 'type': 'anyone'}).execute()
            except: pass 
            return f"https://drive.google.com/thumbnail?id={file.get('id')}&sz=w1000"
        return execute_with_retry(_upload_action)

    def clean_id(val):
        try: return str(int(float(val))).strip()
        except: return str(val).strip()

    # ==========================================
    # SQLite 背景佇列 (核心引擎)
    # ==========================================
    _queue_lock = threading.Lock()

    @st.cache_resource
    def get_queue_connection():
        conn = sqlite3.connect(QUEUE_DB_PATH, check_same_thread=False, timeout=30.0, isolation_level="IMMEDIATE")
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=30000;")
        except: pass
        conn.execute("""
            CREATE TABLE IF NOT EXISTS task_queue (
                id TEXT PRIMARY KEY, task_type TEXT, created_ts TEXT, 
                payload_json TEXT, status TEXT, attempts INTEGER, last_error TEXT
            )
        """)
        conn.commit()
        return conn

    def enqueue_task(task_type, payload):
        conn = get_queue_connection()
        task_id = str(uuid.uuid4())
        with _queue_lock:
            conn.execute("INSERT INTO task_queue VALUES (?, ?, ?, ?, 'PENDING', 0, NULL)",
                (task_id, task_type, datetime.now(timezone.utc).isoformat(), json.dumps(payload, ensure_ascii=False)))
            conn.commit()
        return task_id

    def get_queue_metrics():
        conn = get_queue_connection()
        metrics = {"pending": 0, "retry": 0, "failed": 0, "oldest_pending_sec": 0, "recent_errors": []}
        with _queue_lock:
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
        return metrics

    def fetch_next_task(max_attempts=6):
        conn = get_queue_connection()
        with _queue_lock:
            cur = conn.cursor()
            cur.execute("SELECT id, task_type, created_ts, payload_json, status, attempts, last_error FROM task_queue WHERE status IN ('PENDING', 'RETRY') AND attempts < ? ORDER BY created_ts ASC LIMIT 1", (max_attempts,))
            row = cur.fetchone()
            if not row: return None
            
            task_id, task_type, created_ts, payload_json, status, attempts, last_error = row
            cur.execute("UPDATE task_queue SET status = 'IN_PROGRESS', attempts = attempts + 1 WHERE id = ?", (task_id,))
            conn.commit()
            
            try: payload = json.loads(payload_json)
            except: payload = {}
            return {"id": task_id, "task_type": task_type, "payload": payload, "attempts": attempts + 1}

    def update_task_status(task_id, status, attempts, last_error):
        conn = get_queue_connection()
        with _queue_lock:
            conn.execute("UPDATE task_queue SET status = ?, attempts = ?, last_error = ? WHERE id = ?", (status, attempts, last_error, task_id))
            conn.commit()

    # ==========================================
    # 背景處理邏輯 (Worker)
    # ==========================================
    def _append_main_entry_row(entry):
        def _action():
            ws = get_worksheet(SHEET_TABS["main"])
            if not ws: return
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
    
    def _append_service_row_helper(entry):
        def _action():
            ws = get_worksheet(SHEET_TABS["service_hours"])
            if not ws: return
            row = [
                str(entry.get("日期", "")), str(entry.get("學號", "")),
                str(entry.get("班級", "")), str(entry.get("類別", "")),
                str(entry.get("時數", "")), str(entry.get("紀錄ID", ""))
            ]
            ws.append_row(row)
        execute_with_retry(_action)

    def process_task(task):
        task_type = task["task_type"]
        payload = task["payload"]
        entry = payload.get("entry", {})

        try:
            # 1. 圖片處理 (共用邏輯)
            image_paths = payload.get("image_paths", [])
            filenames = payload.get("filenames", [])
            drive_links = []
            
            for path, fname in zip(image_paths, filenames):
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        compressed = compress_image_bytes(f.read())
                        link = upload_image_to_drive(compressed, fname)
                    drive_links.append(link if link else "UPLOAD_FAILED_API")
            
            if drive_links: entry["照片路徑"] = ";".join(drive_links)

            # 2. 寫入主表 (志工回報 / 糾察評分)
            if task_type in ["main_entry", "volunteer_report"]:
                _append_main_entry_row(entry)

                # [自動時數 - A] 糾察評分獎勵
                inspector_name = entry.get("檢查人員", "")
                if "學號:" in inspector_name:
                    try:
                        sid = inspector_name.split("學號:")[1].strip()
                        log_entry = {
                            "日期": entry.get("日期"), "學號": sid,
                            "班級": "", "類別": "衛生糾察", "時數": 0.5, "紀錄ID": uuid.uuid4().hex[:8]
                        }
                        _append_service_row_helper(log_entry)
                    except: pass
                
                # [自動時數 - B] 志工回報 / 返校打掃
                if task_type == "volunteer_report":
                    student_list = payload.get("student_list", [])
                    cls_name = entry.get("班級", "")
                    report_date = entry.get("日期", str(date.today()))
                    
                    # [支援自訂時數]：返校打掃給 2.0，晨掃預設 0.5
                    hours = payload.get("custom_hours", 0.5) 
                    category = payload.get("custom_category", "晨掃志工")

                    for sid in student_list:
                        log_entry = {
                            "日期": report_date, "學號": sid,
                            "班級": cls_name, "類別": category, 
                            "時數": hours, "紀錄ID": uuid.uuid4().hex[:8]
                        }
                        _append_service_row_helper(log_entry)

            # 3. 申訴處理
            elif task_type == "appeal_entry":
                # (略，與原版相同，但加入圖片壓縮)
                image_info = payload.get("image_file")
                if image_info and os.path.exists(image_info["path"]):
                    with open(image_info["path"], "rb") as f:
                        compressed = compress_image_bytes(f.read())
                        link = upload_image_to_drive(compressed, image_info["filename"])
                    entry["佐證照片"] = link
                
                def _app_action():
                    ws = get_worksheet(SHEET_TABS["appeals"])
                    ws.append_row([str(entry.get(c, "")) for c in APPEAL_COLUMNS])
                execute_with_retry(_app_action)

            return True, None
        except Exception as e:
            return False, str(e)

    def background_worker(stop_event=None):
        print("🚀 背景工作者啟動...")
        try: add_script_run_ctx(threading.current_thread(), get_script_run_ctx())
        except: pass
        while True:
            if stop_event and stop_event.is_set(): break
            try:
                task = fetch_next_task()
                if not task:
                    time.sleep(2.0)
                    continue
                
                # 執行任務
                ok, err = process_task(task)
                
                # 清理暫存檔
                try:
                    payload = task["payload"]
                    paths = payload.get("image_paths", [])
                    if "image_file" in payload: paths.append(payload["image_file"]["path"])
                    for p in paths:
                        if p and os.path.exists(p): os.remove(p)
                except: pass

                status = "DONE" if ok else ("FAILED" if task["attempts"] >= 6 else "RETRY")
                update_task_status(task["id"], status, task["attempts"], err)
                time.sleep(0.5)
            except Exception as e:
                print(f"Worker Error: {e}")
                time.sleep(3.0)

    # 確保 Worker 活著
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
    @st.cache_data(ttl=300)
    def load_main_data():
        ws = get_worksheet(SHEET_TABS["main"])
        if not ws: return pd.DataFrame(columns=EXPECTED_COLUMNS)
        try:
            data = ws.get_all_records()
            df = pd.DataFrame(data)
            if df.empty: return pd.DataFrame(columns=EXPECTED_COLUMNS)
            if "班級" in df.columns: df["班級"] = df["班級"].astype(str).str.strip()
            for col in EXPECTED_COLUMNS:
                if col not in df.columns: df[col] = ""
            if "紀錄ID" not in df.columns: df["紀錄ID"] = df.index.astype(str)
            
            # 數值轉型
            for col in ["內掃原始分", "外掃原始分", "垃圾原始分", "晨間打掃原始分", "手機人數"]:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            return df[EXPECTED_COLUMNS]
        except: return pd.DataFrame(columns=EXPECTED_COLUMNS)

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
                        roster_dict[clean_id(row[id_col])] = str(row[class_col]).strip()
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
            unique = df[class_col].dropna().unique().tolist()
            unique = [str(c).strip() for c in unique if str(c).strip()]
            
            dept_order = {"商": 1, "英": 2, "資": 3, "家": 4, "服": 5}
            def get_sort_key(name):
                grade = 99
                if "一" in name or "1" in name: grade = 1
                elif "二" in name or "2" in name: grade = 2
                elif "三" in name or "3" in name: grade = 3
                dept_score = 99
                for k, v in dept_order.items():
                    if k in name: dept_score = v; break
                return (grade, dept_score, name)
            
            sorted_all = sorted(unique, key=get_sort_key)
            structured = [{"grade": f"{get_sort_key(c)[0]}年級" if get_sort_key(c)[0]!=99 else "其他", "name": c} for c in sorted_all]
            return sorted_all, structured
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

    @st.cache_data(ttl=21600)
    def load_settings():
        ws = get_worksheet(SHEET_TABS["settings"])
        config = {"semester_start": "2025-08-25", "standard_n": 4}
        if ws:
            try:
                for row in ws.get_all_values():
                    if len(row)>=2:
                        if row[0] == "semester_start": config["semester_start"] = row[1]
                        if row[0] == "standard_n": config["standard_n"] = int(row[1])
            except: pass
        return config

    # 封裝 Save Entry (處理暫存與 Enqueue)
    def save_entry(new_entry, uploaded_files=None, student_list=None, custom_hours=0.5, custom_category="晨掃志工"):
        if "日期" in new_entry: new_entry["日期"] = str(new_entry["日期"])
        if "紀錄ID" not in new_entry: new_entry["紀錄ID"] = f"{datetime.now(TW_TZ).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"

        image_paths, file_names = [], []
        if uploaded_files:
            for i, up_file in enumerate(uploaded_files):
                if not up_file: continue
                # 這裡只存暫存，壓縮交給 Worker
                try:
                    data = up_file.getvalue()
                    if len(data) > MAX_IMAGE_BYTES: 
                        st.warning(f"檔案過大略過: {up_file.name}"); continue
                    
                    fname = f"{new_entry['紀錄ID']}_{i}.jpg"
                    local_path = os.path.join(IMG_DIR, fname)
                    with open(local_path, "wb") as f: f.write(data)
                    image_paths.append(local_path)
                    file_names.append(fname)
                except Exception as e: print(f"File Save Error: {e}")

        task_type = "volunteer_report" if student_list is not None else "main_entry"
        payload = {
            "entry": new_entry, "image_paths": image_paths, "filenames": file_names,
            "student_list": student_list if student_list else [],
            "custom_hours": custom_hours, "custom_category": custom_category
        }
        return enqueue_task(task_type, payload)

    # ==========================================
    # 3. 主程式 UI
    # ==========================================
    SYSTEM_CONFIG = load_settings()
    ROSTER_DICT = load_roster_dict()
    all_classes, structured_classes = load_sorted_classes()
    if not all_classes: all_classes = ["測試班級"]
    
    now_tw = datetime.now(TW_TZ)
    today_tw = now_tw.date()
    
    st.sidebar.title("🏫 功能選單")
    app_mode = st.sidebar.radio("請選擇模式", ["糾察底家👀", "班級負責人🥸", "晨掃志工隊🧹", "組長ㄉ窩💃"])

    # 系統診斷
    with st.sidebar.expander("🔧 系統狀態", expanded=False):
        if get_gspread_client(): st.success("Google Sheets: OK")
        else: st.error("Google Sheets: Error")

    # --- Mode 1: 糾察評分 ---
    if app_mode == "糾察底家👀":
        st.title("📝 衛生糾察評分系統")
        if "team_logged_in" not in st.session_state: st.session_state["team_logged_in"] = False
        
        if not st.session_state["team_logged_in"]:
            pwd = st.text_input("輸入通行碼", type="password")
            if st.button("登入"):
                if pwd == st.secrets["system_config"]["team_password"]:
                    st.session_state["team_logged_in"] = True
                    st.rerun()
                else: st.error("密碼錯誤")
        else:
            inspector_list = pd.DataFrame(get_worksheet(SHEET_TABS["inspectors"]).get_all_records())
            st.info("👋 歡迎回來，糾察隊！提交評分後，系統將自動記錄 0.5 小時服務時數。")
            
            # (簡化：直接進入評分表單)
            col1, col2 = st.columns(2)
            input_date = col1.date_input("日期", today_tw)
            inspector_name = col2.text_input("檢查人員 (請輸入 學號:姓名)", placeholder="例如 110123:王小明")
            
            st.divider()
            target_cls = st.selectbox("受檢班級", all_classes)
            role = st.selectbox("檢查項目", ["內掃檢查", "外掃檢查", "垃圾/回收檢查"])
            
            with st.form("inspector_form"):
                score = 0
                result = st.radio("檢查結果", ["✨ 通過", "❌ 違規扣分"], horizontal=True)
                if result == "❌ 違規扣分":
                    score = st.number_input("扣分 (1-2分)", min_value=1, max_value=2)
                
                note = st.text_input("違規說明 / 備註")
                files = st.file_uploader("違規照片", accept_multiple_files=True, type=['jpg', 'png'])
                
                if st.form_submit_button("送出評分"):
                    if not inspector_name or ":" not in inspector_name:
                        st.error("請依照格式輸入姓名 (學號:姓名)")
                    elif score > 0 and not files:
                        st.error("扣分必須上傳照片")
                    else:
                        entry = {
                            "日期": input_date, "班級": target_cls, "評分項目": role,
                            "檢查人員": f"學號:{inspector_name.split(':')[0]}", # 為了觸發自動時數
                            "備註": f"{inspector_name.split(':')[1]} - {note}",
                            f"{role[:2]}原始分": score # 簡易對應
                        }
                        save_entry(entry, uploaded_files=files)
                        st.success("✅ 評分已送出！")

    # --- Mode 2: 班級負責人 ---
    elif app_mode == "班級負責人🥸":
        st.title("💻 班級管理專區")
        sub_tab = st.radio("功能選擇", ["🔎 成績查詢", "🏫 返校打掃回報 (全班)"], horizontal=True)

        if sub_tab == "🔎 成績查詢":
            # (維持原有查詢邏輯，略為簡化顯示)
            df = load_main_data()
            my_cls = st.selectbox("選擇班級查詢", all_classes)
            if not df.empty and my_cls:
                c_df = df[df["班級"]==my_cls].sort_values("日期", ascending=False)
                st.dataframe(c_df[["日期", "評分項目", "檢查人員", "備註", "內掃原始分", "外掃原始分"]])

        elif sub_tab == "🏫 返校打掃回報 (全班)":
            st.info("💡 說明：此功能用於全班性返校打掃。出席者將自動獲得 **2小時** 服務時數。")
            with st.form("return_clean"):
                r_date = st.date_input("打掃日期", today_tw)
                r_class = st.selectbox("班級", all_classes)
                
                members = [sid for sid, c in ROSTER_DICT.items() if c == r_class]
                if not members:
                    st.error("❌ 找不到該班名單")
                    st.form_submit_button("無法送出")
                else:
                    st.write(f"全班 {len(members)} 人，請勾選 **缺席** 者 (扣除法)：")
                    absent = st.multiselect("缺席名單", members)
                    present = [m for m in members if m not in absent]
                    st.write(f"📊 預計發放時數：**{len(present)}** 人 (每人 2.0 小時)")
                    
                    proof = st.file_uploader("📸 上傳全班集合照 (必填)", type=['jpg', 'png'])
                    if st.form_submit_button("確認送出"):
                        if not proof: st.error("請上傳照片")
                        else:
                            entry = {
                                "日期": r_date, "班級": r_class, "評分項目": "返校打掃",
                                "檢查人員": f"返校回報(實到:{len(present)})",
                                "備註": f"缺席: {','.join(absent)}"
                            }
                            # 關鍵：custom_hours=2.0
                            save_entry(entry, uploaded_files=[proof], student_list=present, custom_hours=2.0, custom_category="返校打掃")
                            st.success("✅ 回報成功！時數已排程發放。")

    # --- Mode 3: 晨掃志工隊 (新功能) ---
    elif app_mode == "晨掃志工隊🧹":
        st.title("🧹 晨掃志工回報專區")
        
        # 1. 時間限制
        if now_tw.hour >= 16:
            st.error("🚫 今日回報已截止 (每日 16:00 關閉)。")
        else:
            # 2. 重複偵測
            my_class = st.selectbox("選擇班級", all_classes)
            main_df = load_main_data()
            is_dup = False
            if not main_df.empty:
                check = main_df[
                    (main_df["日期"].astype(str)==str(today_tw)) & 
                    (main_df["班級"]==my_class) & 
                    (main_df["評分項目"]=="晨間打掃")
                ]
                if not check.empty: is_dup = True
            
            if is_dup:
                st.warning(f"⚠️ {my_class} 今天已經回報過了！")
            else:
                # 3. 掃區提示
                duty_df, _ = get_daily_duty(today_tw)
                task_info = "無特定掃區"
                if not duty_df.empty:
                    my_duty = duty_df[duty_df["負責班級"]==my_class]
                    if not my_duty.empty: task_info = f"{my_duty.iloc[0]['掃地區域']} (標準:{my_duty.iloc[0]['標準人數']}人)"
                st.info(f"📍 今日任務：{task_info}")
                
                # 4. 回報表單
                with st.form("morning_form"):
                    members = [sid for sid, c in ROSTER_DICT.items() if c == my_class]
                    present = st.multiselect("✅ 勾選實際參與同學 (發放 0.5hr)", members)
                    files = st.file_uploader("📸 成果照片", accept_multiple_files=True, type=['jpg', 'png'])
                    
                    if st.form_submit_button("送出回報"):
                        if not present or not files: st.error("請勾選名單並上傳照片")
                        else:
                            entry = {
                                "日期": today_tw, "班級": my_class, "評分項目": "晨間打掃",
                                "檢查人員": f"志工回報(實到:{len(present)})",
                                "晨間打掃原始分": 0, # 等待審核
                                "備註": f"名單:{','.join(present)}"
                            }
                            # 關鍵：custom_hours=0.5
                            save_entry(entry, uploaded_files=files, student_list=present, custom_hours=0.5, custom_category="晨掃志工")
                            st.success("✅ 回報成功！待老師審核通過後生效。")

    # --- Mode 4: 組長後台 ---
    elif app_mode == "組長ㄉ窩💃":
        st.title("⚙️ 管理後台")
        pwd = st.text_input("管理密碼", type="password")
        if pwd == st.secrets["system_config"]["admin_password"]:
            # 監控面板
            metrics = get_queue_metrics()
            c1, c2, c3 = st.columns(3)
            c1.metric("待處理任務", metrics["pending"])
            c2.metric("失敗任務", metrics["failed"])
            c3.metric("最久延遲(秒)", int(metrics["oldest_pending_sec"]))

            tab1, tab2 = st.tabs(["🧹 晨掃審核", "📊 成績總表"])
            
            with tab1:
                st.subheader("待審核晨掃回報")
                df = load_main_data()
                pending = df[
                    (df["評分項目"]=="晨間打掃") & 
                    (df["晨間打掃原始分"]==0) & 
                    (df["修正"]!="TRUE")
                ]
                
                if pending.empty: st.success("🎉 目前無待審核案件")
                else:
                    for i, row in pending.iterrows():
                        with st.container(border=True):
                            c1, c2, c3 = st.columns([2,2,1])
                            c1.write(f"**{row['班級']}** | {row['日期']}")
                            c1.caption(f"{row['檢查人員']}")
                            
                            raw_p = str(row.get("照片路徑", ""))
                            if raw_p and "http" in raw_p: c2.image(raw_p.split(";")[0], width=200)
                            else: c2.warning("無照片")
                            
                            if c3.button("✅ 通過 (+2分)", key=f"pass_{row['紀錄ID']}"):
                                ws = get_worksheet(SHEET_TABS["main"])
                                try:
                                    ids = ws.col_values(EXPECTED_COLUMNS.index("紀錄ID")+1)
                                    ridx = ids.index(str(row["紀錄ID"])) + 1
                                    cidx = EXPECTED_COLUMNS.index("晨間打掃原始分") + 1
                                    ws.update_cell(ridx, cidx, 2)
                                    st.success("已核可！")
                                    st.cache_data.clear()
                                    time.sleep(1); st.rerun()
                                except: st.error("更新失敗")
                            
                            if c3.button("🗑️ 駁回", key=f"del_{row['紀錄ID']}"):
                                ws = get_worksheet(SHEET_TABS["main"])
                                try:
                                    ids = ws.col_values(EXPECTED_COLUMNS.index("紀錄ID")+1)
                                    ridx = ids.index(str(row["紀錄ID"])) + 1
                                    ws.delete_rows(ridx)
                                    st.warning("已刪除")
                                    st.cache_data.clear()
                                    st.rerun()
                                except: st.error("刪除失敗")

            with tab2:
                st.info("請點選下方按鈕產生報表")
                if st.button("計算全校排名"):
                    # (此處可貼上原有的報表計算邏輯，為節省篇幅略)
                    st.write("功能開發中...")

except Exception as e:
    st.error("❌ 系統錯誤")
    print(traceback.format_exc())
