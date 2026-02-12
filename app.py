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
st.set_page_config(page_title="中壢家商，衛愛而生 V3.2", layout="wide", page_icon="🧹")

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
        "service_hours": "service_hours"
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
                    cols = 20 if tab_name != "appeals" else 15
                    ws = sheet.add_worksheet(title=tab_name, rows=500, cols=cols)
                    if tab_name == "appeals": ws.append_row(APPEAL_COLUMNS)
                    if tab_name == "service_hours": ws.append_row(["日期", "學號", "班級", "類別", "時數", "紀錄ID"])
                    if tab_name == "holidays": ws.append_row(["日期", "說明"])
                    return ws
            except Exception as e:
                if "429" in str(e): 
                    time.sleep(2 * (attempt + 1))
                    continue
                else: return None
        return None

    def compress_image_bytes(file_bytes, quality=70):
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
            # 寫入 6 個欄位: 日期, 學號, 班級, 類別, 時數, 紀錄ID
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

            if task_type in ["main_entry", "volunteer_report"]:
                _append_main_entry_row(entry)

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
                
                if task_type == "volunteer_report":
                    student_list = payload.get("student_list", [])
                    cls_name = entry.get("班級", "")
                    report_date = entry.get("日期", str(date.today()))
                    hours = payload.get("custom_hours", 0.5) 
                    category = payload.get("custom_category", "晨掃志工")

                    for sid in student_list:
                        log_entry = {
                            "日期": report_date, "學號": sid,
                            "班級": cls_name, "類別": category, 
                            "時數": hours, "紀錄ID": uuid.uuid4().hex[:8]
                        }
                        _append_service_row_helper(log_entry)

            elif task_type == "appeal_entry":
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
                    time.sleep(2.0); continue
                
                ok, err = process_task(task)
                
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
                print(f"Worker Error: {e}"); time.sleep(3.0)

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
        holiday_list = []
        if ws:
            try:
                records = ws.get_all_records()
                for r in records:
                    d_str = str(r.get("日期", "")).strip()
                    if d_str:
                        try:
                            d_obj = pd.to_datetime(d_str).date()
                            holiday_list.append(d_obj)
                        except: pass
            except: pass
        return holiday_list

    def is_within_appeal_period(violation_date, appeal_days=3):
        if isinstance(violation_date, str):
            violation_date = pd.to_datetime(violation_date).date()
        holidays = load_holidays()
        today = date.today()
        current_date = violation_date
        workdays_counted = 0
        for _ in range(14): 
            if workdays_counted >= appeal_days: break
            current_date += timedelta(days=1)
            if current_date.weekday() >= 5 or current_date in holidays: continue
            else: workdays_counted += 1
        return today <= current_date

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
            for col in ["內掃原始分", "外掃原始分", "垃圾原始分", "垃圾內掃原始分", "垃圾外掃原始分", "晨間打掃原始分", "手機人數"]:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            if "週次" in df.columns:
                df["週次"] = pd.to_numeric(df["週次"], errors='coerce').fillna(0).astype(int)
            if "修正" in df.columns:
                df["修正"] = df["修正"].astype(str).apply(lambda x: True if x.upper() == "TRUE" else False)
            return df[EXPECTED_COLUMNS]
        except: return pd.DataFrame(columns=EXPECTED_COLUMNS)

    def load_full_semester_data_for_export():
        ws = get_worksheet(SHEET_TABS["main"])
        if not ws: return pd.DataFrame(columns=EXPECTED_COLUMNS)
        try:
            data = ws.get_all_records()
            df = pd.DataFrame(data)
            if df.empty: return pd.DataFrame(columns=EXPECTED_COLUMNS)
            for col in EXPECTED_COLUMNS:
                if col not in df.columns: df[col] = ""
            text_cols = ["備註", "違規細項", "班級", "檢查人員", "修正", "晨掃未到者", "照片路徑", "紀錄ID"]
            for col in text_cols:
                if col in df.columns: df[col] = df[col].fillna("").astype(str)
            numeric_cols = ["內掃原始分", "外掃原始分", "垃圾原始分", "垃圾內掃原始分", "垃圾外掃原始分", "晨間打掃原始分", "手機人數", "週次"]
            for col in numeric_cols:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            return df[EXPECTED_COLUMNS]
        except: return pd.DataFrame()

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
            records = ws.get_all_records()
            if not records:
                all_vals = ws.get_all_values()
                if len(all_vals) > 1:
                    headers = all_vals[0]
                    records = [dict(zip(headers, row)) for row in all_vals[1:]]
            df = pd.DataFrame(records)
            if df.empty: return [], []
            class_col = next((c for c in df.columns if "班級" in str(c).strip()), None)
            if not class_col: return [], []
            unique = df[class_col].astype(str).str.strip().unique().tolist()
            unique = [c for c in unique if c]
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

    @st.cache_data(ttl=60)
    def load_appeals():
        ws = get_worksheet(SHEET_TABS["appeals"])
        if not ws: return pd.DataFrame(columns=APPEAL_COLUMNS)
        try:
            records = ws.get_all_records()
            df = pd.DataFrame(records)
        except Exception: return pd.DataFrame(columns=APPEAL_COLUMNS)
        for col in APPEAL_COLUMNS:
            if col not in df.columns:
                if col == "處理狀態": df[col] = "待處理"
                else: df[col] = ""
        return df[APPEAL_COLUMNS]

    def save_appeal(entry, proof_file=None):
        image_info = None
        if proof_file:
            try:
                proof_file.seek(0)
                data = proof_file.read()
                if len(data) > MAX_IMAGE_BYTES:
                    st.error(f"照片過大"); return False
                logical_fname = f"Appeal_{entry.get('班級', '')}_{datetime.now(TW_TZ).strftime('%H%M%S')}.jpg"
                tmp_fname = f"{datetime.now(TW_TZ).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}_{logical_fname}"
                local_path = os.path.join(IMG_DIR, tmp_fname)
                with open(local_path, "wb") as f: f.write(data)
                image_info = {"path": local_path, "filename": logical_fname}
            except Exception as e:
                st.error(f"寫入失敗: {e}"); return False

        if "申訴日期" not in entry: entry["申訴日期"] = datetime.now(TW_TZ).strftime("%Y-%m-%d")
        entry["處理狀態"] = entry.get("處理狀態", "待處理")
        if "登錄時間" not in entry: entry["登錄時間"] = datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")
        if "申訴ID" not in entry: entry["申訴ID"] = datetime.now(TW_TZ).strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:4]
        if "佐證照片" not in entry: entry["佐證照片"] = ""

        payload = {"entry": entry, "image_file": image_info}
        enqueue_task("appeal_entry", payload)
        st.success("📩 申訴已排入背景處理")
        return True
    
    def update_appeal_status(appeal_row_idx, status, record_id):
        ws_appeals = get_worksheet(SHEET_TABS["appeals"])
        ws_main = get_worksheet(SHEET_TABS["main"])
        try:
            appeals_data = ws_appeals.get_all_records()
            target_row = None
            for i, row in enumerate(appeals_data):
                if str(row.get("對應紀錄ID")) == str(record_id) and str(row.get("處理狀態")) == "待處理":
                    target_row = i + 2; break
            if target_row:
                col_idx = APPEAL_COLUMNS.index("處理狀態") + 1
                ws_appeals.update_cell(target_row, col_idx, status)
                if status == "已核可" and record_id:
                    main_data = ws_main.get_all_records()
                    main_target_row = None
                    for j, m_row in enumerate(main_data):
                        if str(m_row.get("紀錄ID")) == str(record_id):
                            main_target_row = j + 2; break
                    if main_target_row:
                        fix_col_idx = EXPECTED_COLUMNS.index("修正") + 1
                        ws_main.update_cell(main_target_row, fix_col_idx, "TRUE")
                st.cache_data.clear()
                return True, "更新成功"
            else: return False, "找不到對應的申訴列"
        except Exception as e: return False, str(e)
    
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

    def send_bulk_emails(email_list):
        sender_email = st.secrets["system_config"]["smtp_email"]
        sender_password = st.secrets["system_config"]["smtp_password"]
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
                except Exception as inner_e: print(f"個別寄送失敗: {inner_e}")
            server.quit()
            return sent_count, "發送作業結束"
        except Exception as e: return sent_count, str(e)

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
            for row_idx in rows_to_delete: ws.delete_rows(row_idx)
            time.sleep(0.8)
            st.cache_data.clear()
            return True
        except Exception as e:
            st.error(f"刪除失敗: {e}"); return False

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

    def check_duplicate_record(df, check_date, inspector, role, target_class=None):
        if df.empty: return False
        try:
            df["日期Str"] = df["日期"].astype(str)
            check_date_str = str(check_date)
            mask = (df["日期Str"] == check_date_str) & (df["檢查人員"] == inspector) & (df["評分項目"] == role)
            if target_class: mask = mask & (df["班級"] == target_class)
            return not df[mask].empty
        except: return False

    # 封裝 Save Entry (處理暫存與 Enqueue)
    def save_entry(new_entry, uploaded_files=None, student_list=None, custom_hours=0.5, custom_category="晨掃志工"):
        if "日期" in new_entry: new_entry["日期"] = str(new_entry["日期"])
        if "紀錄ID" not in new_entry: new_entry["紀錄ID"] = f"{datetime.now(TW_TZ).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"

        image_paths, file_names = [], []
        if uploaded_files:
            for i, up_file in enumerate(uploaded_files):
                if not up_file: continue
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
    app_mode = st.sidebar.radio("請選擇模式", ["糾察底家👀", "班級負責人🥸", "晨掃志工隊🧹", "組長ㄉ窩💃"])

    with st.sidebar.expander("🔧 系統狀態 (名單異常請點此)", expanded=True):
        if get_gspread_client(): st.success("✅ Google Sheets 連線正常")
        else: st.error("❌ Google Sheets 連線失敗")
        
        # [新增] 強制重讀按鈕
        if st.button("🔄 重讀名單 (清除快取)"):
            st.cache_data.clear()
            st.rerun()

    # --- Mode 1: 糾察評分 ---
    if app_mode == "糾察底家👀":
        st.title("📝 衛生糾察評分系統")
        if "team_logged_in" not in st.session_state: st.session_state["team_logged_in"] = False
        if "last_submitted_class" not in st.session_state: st.session_state["last_submitted_class"] = None
        
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
            if not prefix_labels: st.warning("找不到糾察名單")
            else:
                sel_p = st.radio("步驟 1：選擇開頭", prefix_labels, horizontal=True)[0]
                filtered = [p for p in INSPECTOR_LIST if p["id_prefix"] == sel_p]
                inspector_name = st.radio("步驟 2：點選身份", [p["label"] for p in filtered])
                curr_inspector = next((p for p in INSPECTOR_LIST if p["label"] == inspector_name), None)
                allowed_roles = [r for r in curr_inspector.get("allowed_roles", ["內掃檢查"]) if r != "晨間打掃"]
                if not allowed_roles: allowed_roles = ["內掃檢查"] 
                assigned_classes = curr_inspector.get("assigned_classes", [])
                
                st.markdown("---")
                c_d, c_r = st.columns(2)
                input_date = c_d.date_input("檢查日期", today_tw)
                role = c_r.radio("檢查項目", allowed_roles, horizontal=True) if len(allowed_roles)>1 else allowed_roles[0]
                c_r.info(f"📋 負責項目：**{role}**")
                week_num = get_week_num(input_date)
                main_df = load_main_data()

                if role == "垃圾/回收檢查":
                    # [V3.2] 垃圾檢查介面大改版：定點檢查模式 (左右分欄)
                    st.info("🗑️ 資收場定點檢查模式：請先選擇班級，再勾選違規項目")
                    
                    target_cls = st.selectbox("👉 請選擇班級", all_classes)
                    
                    with st.form("trash_check_form"):
                        col_in, col_out = st.columns(2)
                        
                        with col_in:
                            st.subheader("🏠 內掃 (教室)")
                            v_in_1 = st.checkbox("未分類", key=f"in_1_{target_cls}")
                            v_in_2 = st.checkbox("未簽名", key=f"in_2_{target_cls}")
                            
                        with col_out:
                            st.subheader("🏢 外掃 (處室)")
                            v_out_1 = st.checkbox("外掃-未分類", key=f"out_1_{target_cls}")
                            
                            # 外掃必須指定處室
                            office_list = ["", "教務處", "學務處", "總務處", "輔導室", "圖書館", "實習處", "健康中心", "體育組", "校長室", "人事室", "會計室", "其他"]
                            target_office = st.selectbox("⚠️ 違規處室 (若勾選外掃違規請務必選擇)", office_list, key=f"off_{target_cls}")
                        
                        st.divider()
                        note_ext = st.text_input("📝 補充說明 (選填)")
                        
                        if st.form_submit_button("🚀 送出違規"):
                            # 計算分數
                            score_in = 0
                            score_out = 0
                            violations = []
                            
                            if v_in_1: score_in += 1; violations.append("內掃-未分類")
                            if v_in_2: score_in += 1; violations.append("內掃-未簽名")
                            
                            office_note = ""
                            if v_out_1:
                                score_out += 1
                                if not target_office:
                                    st.error("❌ 登記外掃違規，必須選擇「違規處室」！")
                                    st.stop()
                                violations.append(f"外掃({target_office})-未分類")
                                office_note = target_office
                            
                            if score_in == 0 and score_out == 0:
                                st.warning("未勾選任何違規項目")
                            else:
                                base = {
                                    "日期": input_date, "週次": week_num, "檢查人員": inspector_name,
                                    "登錄時間": now_tw.strftime("%Y-%m-%d %H:%M:%S"), "修正": False,
                                    "班級": target_cls, "評分項目": role,
                                    "垃圾內掃原始分": score_in, "垃圾外掃原始分": score_out, # 分開寫入
                                    "備註": f"{','.join(violations)} {note_ext}",
                                    "違規細項": "垃圾違規"
                                }
                                save_entry(base)
                                st.success(f"✅ 已登記：{target_cls} (內掃:{score_in}分, 外掃:{score_out}分)")
                                time.sleep(1.5)
                                st.rerun()

                else:
                    st.markdown("### 🏫 選擇受檢班級")
                    if assigned_classes:
                        selected_class = st.radio("請點選您的負責班級", assigned_classes, key=f"radio_assigned_{inspector_name}")
                    else:
                        g = st.radio("步驟 A: 選擇年級", grades, horizontal=True)
                        f_cls = [c["name"] for c in structured_classes if c["grade"] == g]
                        selected_class = st.radio("步驟 B: 選擇班級", f_cls, horizontal=True) if f_cls else None
            
                    if selected_class:
                        st.divider()
                        if st.session_state.get("last_submitted_class") == selected_class:
                            st.warning(f"⚠️ 注意：您剛剛才評過 **{selected_class}**")
                        st.markdown(f"#### 👉 正在評分： <span style='color:#e05858;font-size:1.3em'>{selected_class}</span>", unsafe_allow_html=True)
                        if check_duplicate_record(main_df, input_date, inspector_name, role, selected_class):
                            st.warning(f"⚠️ 系統紀錄顯示：您今天已經評過「{selected_class}」了！")
                        
                        with st.form("scoring_form", clear_on_submit=True):
                            in_s, out_s, ph_c, note = 0, 0, 0, ""
                            if role == "內掃檢查":
                                if st.radio("檢查結果", ["❌ 違規", "✨ 乾淨"], horizontal=True, key=f"rd_{selected_class}") == "❌ 違規":
                                    in_s = st.number_input("內掃扣分 (上限2分)", 0)
                                    c1, c2 = st.columns(2)
                                    sel_area = c1.selectbox("區塊", ["", "走廊", "陽台", "黑板", "地板", "窗戶"])
                                    sel_status = c2.selectbox("狀況", ["", "髒亂", "有垃圾", "頭髮圈圈", "沒拖地"])
                                    manual_note = st.text_input("📝 補充說明")
                                    note = " ".join([x for x in [sel_area, sel_status, manual_note] if x])
                                    ph_c = st.number_input("手機人數", 0)
                                else: note = "【優良】"
                            elif role == "外掃檢查":
                                if st.radio("檢查結果", ["❌ 違規", "✨ 乾淨"], horizontal=True, key=f"rd_{selected_class}") == "❌ 違規":
                                    out_s = st.number_input("外掃扣分 (上限2分)", 0)
                                    c1, c2 = st.columns(2)
                                    sel_area = c1.selectbox("區域", ["", "走廊", "樓梯", "廁所", "露臺", "操場", "資收場"])
                                    sel_bad = c2.selectbox("狀況", ["", "很髒", "沒掃", "有垃圾", "頭髮圈圈"])
                                    manual_note = st.text_input("📝 補充說明")
                                    note = " ".join([x for x in [sel_area, sel_bad, manual_note] if x])
                                    ph_c = st.number_input("手機人數", 0)
                                else: note = "【優良】"

                            is_fix = st.checkbox("🚩 這是修正單")
                            files = st.file_uploader("📸 違規照片", accept_multiple_files=True)
                            
                            if st.form_submit_button("🚀 送出"):
                                if (in_s+out_s) > 0 and not files: st.error("扣分需上傳照片")
                                else:
                                    save_entry({
                                        "日期": input_date, "週次": week_num, "檢查人員": inspector_name,
                                        "登錄時間": now_tw.strftime("%Y-%m-%d %H:%M:%S"), "修正": is_fix,
                                        "班級": selected_class, "評分項目": role, "內掃原始分": in_s,
                                        "外掃原始分": out_s, "手機人數": ph_c, "備註": note
                                    }, uploaded_files=files)
                                    st.session_state["last_submitted_class"] = selected_class
                                    st.rerun()

    # --- Mode 2: 班級負責人 (純查詢) ---
    elif app_mode == "班級負責人🥸":
        st.title("🔎 班級成績查詢")
        df = load_main_data()
        appeals_df = load_appeals()
        appeal_map = {str(r.get("對應紀錄ID")): r.get("處理狀態") for _, r in appeals_df.iterrows() if str(r.get("對應紀錄ID"))}

        st.info("👋 嗨！這裡是衛生股長專區，可以在這裡查詢班級的評分紀錄與申訴狀態。")
        if not df.empty:
            c1, c2 = st.columns(2)
            g = c1.radio("選擇年級", grades, horizontal=True)
            cls_opts = [c["name"] for c in structured_classes if c["grade"] == g]
            if not cls_opts: st.warning("無資料")
            else:
                cls = c2.selectbox("選擇班級", cls_opts)
                st.divider()
                if cls:
                    c_df = df[df["班級"] == cls].sort_values("登錄時間", ascending=False)
                    if not c_df.empty:
                        st.subheader(f"📊 {cls} 近期紀錄")
                        for idx, r in c_df.iterrows():
                            # 顯示垃圾分開計分
                            trash_score = r['垃圾內掃原始分'] + r['垃圾外掃原始分']
                            if trash_score == 0: trash_score = r['垃圾原始分'] # 相容舊資料
                            
                            tot = r['內掃原始分']+r['外掃原始分']+trash_score+r['晨間打掃原始分']
                            ph = f" | 📱:{r['手機人數']}" if r['手機人數'] > 0 else ""
                            rid = str(r['紀錄ID'])
                            ap_st = appeal_map.get(rid)
                            icon = "✅" if ap_st=="已核可" else "🚫" if ap_st=="已駁回" else "⏳" if ap_st=="待處理" else "🛠️" if str(r['修正'])=="TRUE" else ""
                            
                            with st.expander(f"{icon} {r['日期']} - {r['評分項目']} (扣:{tot}){ph}"):
                                if ap_st: st.info(f"申訴狀態: {ap_st}")
                                st.write(f"備註: {r['備註']}")
                                if str(r['照片路徑']) and "http" in str(r['照片路徑']): 
                                    st.image([p for p in str(r['照片路徑']).split(";") if "http" in p], width=200)
                                
                                allow_ap = is_within_appeal_period(r['日期'])
                                if not ap_st and allow_ap and (tot>0 or r['手機人數']>0):
                                    with st.form(f"ap_{rid}"):
                                        rsn = st.text_area("申訴理由")
                                        pf = st.file_uploader("佐證照片", type=['jpg','png'])
                                        if st.form_submit_button("申訴"):
                                            if not rsn or not pf: st.error("請填寫理由並上傳照片")
                                            else:
                                                ap_entry = {
                                                    "申訴日期": str(date.today()), "班級": cls, "違規日期": str(r["日期"]),
                                                    "違規項目": f"{r['評分項目']}", "原始扣分": str(tot), "申訴理由": rsn, "對應紀錄ID": rid
                                                }
                                                save_appeal(ap_entry, pf)
                                                st.rerun()
                    else: st.info("無違規紀錄")

    # --- Mode 3: 晨掃志工隊 ---
    elif app_mode == "晨掃志工隊🧹":
        st.title("🧹 晨掃志工回報專區")
        if now_tw.hour >= 16: st.error("🚫 今日回報已截止 (16:00)")
        else:
            my_cls = st.selectbox("選擇班級", all_classes)
            main_df = load_main_data()
            is_dup = not main_df[(main_df["日期"].astype(str)==str(today_tw)) & (main_df["班級"]==my_cls) & (main_df["評分項目"]=="晨間打掃")].empty
            
            if is_dup: st.warning(f"⚠️ {my_cls} 今天已經回報過了！")
            else:
                duty_df, _ = get_daily_duty(today_tw)
                info = "無特定掃區"
                n_std = 4
                if not duty_df.empty:
                    m_d = duty_df[duty_df["負責班級"]==my_cls]
                    if not m_d.empty: 
                        info = m_d.iloc[0]['掃地區域']
                        n_std = int(m_d.iloc[0]['標準人數'])
                
                st.info(f"📍 任務: {info} (應到:{n_std}人)")
                with st.form("vol_form"):
                    mems = [sid for sid, c in ROSTER_DICT.items() if c == my_cls]
                    present = st.multiselect("✅ 勾選實到同學 (給 0.5hr)", mems)
                    files = st.file_uploader("📸 成果照片", accept_multiple_files=True, type=['jpg','png'])
                    if st.form_submit_button("送出"):
                        if not present or not files: st.error("請勾選名單並上傳照片")
                        else:
                            ent = {
                                "日期": str(today_tw), "班級": my_cls, "評分項目": "晨間打掃",
                                "檢查人員": f"志工回報(實到:{len(present)})", "晨間打掃原始分": 0,
                                "備註": f"名單:{','.join(present)}", "n_actual": len(present), "n_standard": n_std
                            }
                            save_entry(ent, uploaded_files=files, student_list=present, custom_hours=0.5, custom_category="晨掃志工")
                            st.success("✅ 回報成功！"); st.rerun()

    # --- Mode 4: 組長後台 ---
    elif app_mode == "組長ㄉ窩💃":
        st.title("⚙️ 管理後台")
        metrics = get_queue_metrics()
        c1, c2, c3 = st.columns(3)
        c1.metric("待處理", metrics["pending"])
        c2.metric("失敗", metrics["failed"])
        c3.metric("延遲(s)", int(metrics["oldest_pending_sec"]))

        pwd = st.text_input("管理密碼", type="password")
        if pwd == st.secrets["system_config"]["admin_password"]:
            t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
                "🧹 晨掃審核", "📊 成績總表", "🏫 返校打掃", "📝 扣分明細", 
                "📧 寄信", "📣 申訴", "⚙️ 設定", "📄 名單"
            ])
            
            # T1: 晨掃審核
            with t1:
                st.subheader("待審核回報")
                df = load_main_data()
                pending = df[(df["評分項目"]=="晨間打掃") & (df["晨間打掃原始分"]==0) & (df["修正"]!="TRUE")]
                if pending.empty: st.success("無待審核案件")
                else:
                    for i, r in pending.iterrows():
                        with st.container(border=True):
                            c1, c2, c3 = st.columns([2,2,1])
                            c1.write(f"**{r['班級']}** | {r['日期']}")
                            c1.caption(r['檢查人員'])
                            if "http" in str(r['照片路徑']): c2.image(str(r['照片路徑']).split(";")[0], width=150)
                            if c3.button("✅ 通過(+2)", key=f"pass_{r['紀錄ID']}"):
                                ws = get_worksheet(SHEET_TABS["main"])
                                ids = ws.col_values(EXPECTED_COLUMNS.index("紀錄ID")+1)
                                ridx = ids.index(str(r["紀錄ID"])) + 1
                                ws.update_cell(ridx, EXPECTED_COLUMNS.index("晨間打掃原始分")+1, 2)
                                st.success("已核可"); st.cache_data.clear(); st.rerun()
                            if c3.button("🗑️ 駁回", key=f"rej_{r['紀錄ID']}"):
                                delete_rows_by_ids([str(r["紀錄ID"])])
                                st.warning("已刪除"); st.rerun()

            # T2: 成績總表
            with t2:
                st.subheader("📊 成績總表")
                mode = st.radio("排名", ["全校", "年級"], horizontal=True)
                if st.button("🚀 計算全學期成績"):
                    full = load_full_semester_data_for_export()
                    if not full.empty:
                        # [V3.2 修正] 垃圾分數分開結算
                        full["內掃結算"] = full["內掃原始分"].clip(upper=2)
                        full["外掃結算"] = full["外掃原始分"].clip(upper=2)
                        
                        # 垃圾分數邏輯：若是新資料用分開的，舊資料用合併的
                        trash_total = full["垃圾內掃原始分"] + full["垃圾外掃原始分"]
                        # 若新欄位都是0，嘗試用舊欄位
                        trash_total = trash_total.where(trash_total > 0, full["垃圾原始分"])
                        
                        full["垃圾結算"] = trash_total.clip(upper=2)
                        
                        full["總扣分"] = full["內掃結算"]+full["外掃結算"]+full["垃圾結算"]+full["晨間打掃原始分"]+full["手機人數"]
                        rep = full.groupby("班級")["總扣分"].sum().reset_index()
                        
                        cls_df = pd.DataFrame(structured_classes).rename(columns={"grade":"年級","name":"班級"})
                        fin = pd.merge(cls_df, rep, on="班級", how="left").fillna(0)
                        fin["總成績"] = 90 - fin["總扣分"]
                        
                        if mode=="全校": st.dataframe(fin.sort_values("總成績", ascending=False))
                        else:
                            for g in sorted(fin["年級"].unique()):
                                if g!="其他": st.write(g); st.dataframe(fin[fin["年級"]==g].sort_values("總成績", ascending=False))
                    else: st.error("無資料")

            # T3: 返校打掃
            with t3:
                st.subheader("🏫 全班返校打掃登記 (組長用)")
                
                # [V3.2] 修正：選單移出 form，確保名單連動
                c1, c2 = st.columns(2)
                rd = c1.date_input("日期", today_tw)
                rc = c2.selectbox("班級", all_classes)
                
                mems = [s for s, c in ROSTER_DICT.items() if c == rc]
                if not mems: st.error("無名單，請檢查 Roster")
                else:
                    with st.form("ret_clean"):
                        st.write(f"全班 {len(mems)} 人")
                        
                        # A. 扣除缺席
                        absent = st.multiselect("1. 勾選缺席 (沒來的)", mems)
                        present_pool = [m for m in mems if m not in absent]
                        
                        st.divider()
                        st.write("時數設定：")
                        base_h = st.number_input("基礎服務時數 (全班)", value=2.0, step=0.5)
                        
                        # B. 加強組
                        with st.expander("🌟 加強組/特別組 (另外給時數)", expanded=True):
                            special_list = st.multiselect("2. 勾選掃特別久的同學", present_pool)
                            special_h = st.number_input("特別時數 (例如 3.0)", value=3.0, step=0.5)
                        
                        # 計算一般組
                        normal_list = [m for m in present_pool if m not in special_list]
                        
                        st.info(f"預覽：一般組 {len(normal_list)} 人 ({base_h}hr) | 特別組 {len(special_list)} 人 ({special_h}hr)")
                        
                        pf = st.file_uploader("存證照片", type=['jpg','png'])
                        
                        if st.form_submit_button("登記並發放"):
                            if not pf: st.error("需照片")
                            else:
                                # 讀取照片 bytes 一次，供兩次呼叫使用
                                pf.seek(0); file_bytes = pf.read()
                                
                                # 1. 一般組
                                if normal_list:
                                    pf_norm = io.BytesIO(file_bytes); pf_norm.name="proof.jpg"
                                    ent_n = {
                                        "日期": str(rd), "班級": rc, "評分項目": "返校打掃",
                                        "檢查人員": f"返校(一般:{len(normal_list)}人)", "備註": f"缺席:{len(absent)}人"
                                    }
                                    save_entry(ent_n, uploaded_files=[pf_norm], student_list=normal_list, custom_hours=base_h, custom_category="返校打掃(一般)")
                                
                                # 2. 特別組
                                if special_list:
                                    pf_spec = io.BytesIO(file_bytes); pf_spec.name="proof.jpg"
                                    ent_s = {
                                        "日期": str(rd), "班級": rc, "評分項目": "返校打掃",
                                        "檢查人員": f"返校(加強:{len(special_list)}人)", "備註": f"名單:{','.join(special_list)}"
                                    }
                                    save_entry(ent_s, uploaded_files=[pf_spec], student_list=special_list, custom_hours=special_h, custom_category="返校打掃(加強)")
                                
                                st.success("已登記！"); time.sleep(1); st.rerun()

            # T4: 明細
            with t4:
                st.subheader("📝 流水帳")
                df = load_main_data()
                if not df.empty: st.dataframe(df)

            # T5: 寄信
            with t5:
                st.subheader("📧 寄送通知")
                td = st.date_input("日期", today_tw, key="mail_d")
                if st.button("預覽寄送名單"):
                    df = load_main_data()
                    day_df = df[df["日期"].astype(str) == str(td)]
                    if day_df.empty: st.info("無資料")
                    else:
                        stats = day_df.groupby("班級")[["內掃原始分","外掃原始分","垃圾原始分","垃圾內掃原始分","垃圾外掃原始分","晨間打掃原始分","手機人數"]].sum()
                        
                        # [V3.2] 合併垃圾分數
                        trash_t = stats["垃圾內掃原始分"] + stats["垃圾外掃原始分"]
                        # 若新欄位無值，加回舊的
                        stats["Total"] = stats["內掃原始分"]+stats["外掃原始分"]+stats["晨間打掃原始分"]+stats["手機人數"] + trash_t + stats["垃圾原始分"]
                        
                        vios = stats[stats["Total"]>0].reset_index()
                        mail_list = []
                        for _, r in vios.iterrows():
                            t_info = TEACHER_MAILS.get(r["班級"], {})
                            mail_list.append({"班級":r["班級"], "扣分":r["Total"], "Email":t_info.get("email","")})
                        st.dataframe(pd.DataFrame(mail_list))
                        if st.button("確認寄出"):
                            q = []
                            for m in mail_list:
                                if "@" in m["Email"]:
                                    q.append({"email":m["Email"], "subject":f"衛生組通知-{m['班級']}", "body":f"今日扣分:{m['扣分']}"})
                            cnt, msg = send_bulk_emails(q)
                            st.success(f"寄出 {cnt} 封")

            # T6: 申訴
            with t6:
                st.subheader("📣 申訴審核")
                ap_df = load_appeals()
                p_ap = ap_df[ap_df["處理狀態"]=="待處理"]
                if p_ap.empty: st.success("無待審核")
                else:
                    for i, r in p_ap.iterrows():
                        with st.container(border=True):
                            c1, c2 = st.columns([3,1])
                            c1.write(f"{r['班級']} | {r['違規項目']}")
                            c1.write(f"理由: {r['申訴理由']}")
                            if "http" in str(r['佐證照片']): c2.image(str(r['佐證照片']), width=100)
                            if c1.button("核可", key=f"ap_ok_{i}"):
                                update_appeal_status(i, "已核可", r["對應紀錄ID"]); st.rerun()
                            if c1.button("駁回", key=f"ap_ng_{i}"):
                                update_appeal_status(i, "已駁回", r["對應紀錄ID"]); st.rerun()

            # T7: 設定
            with t7:
                st.subheader("⚙️ 設定")
                curr = SYSTEM_CONFIG.get("semester_start")
                nd = st.date_input("開學日", datetime.strptime(curr, "%Y-%m-%d").date() if curr else today_tw)
                if st.button("更新開學日"): save_setting("semester_start", str(nd))

            # T8: 名單
            with t8:
                st.info("請直接至 Google Sheet 修改 inspectors / roster 分頁")
                if st.button("清除快取"): st.cache_data.clear(); st.success("Done")

        else: st.error("密碼錯誤")

except Exception as e:
    st.error("❌ 系統發生錯誤")
    print(traceback.format_exc())
