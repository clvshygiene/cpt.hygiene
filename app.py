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
from PIL import Image

# --- 1. 網頁設定 ---
st.set_page_config(page_title="中壢家商，衛愛而生 V3.9", layout="wide", page_icon="🧹")

# --- 2. 核心參數與全域設定 ---
try:
    TW_TZ = pytz.timezone('Asia/Taipei')
    MAX_IMAGE_BYTES = 10 * 1024 * 1024
    QUEUE_DB_PATH = "task_queue_v4_wal.db"
    IMG_DIR = "evidence_photos"
    os.makedirs(IMG_DIR, exist_ok=True)
    
    SHEET_URL = "https://docs.google.com/spreadsheets/d/11BXtN3aevJls6Q2IR_IbT80-9XvhBkjbTCgANmsxqkg/edit"
    SHEET_TABS = {
        "main": "main_data", "settings": "settings", "roster": "roster",
        "inspectors": "inspectors", "duty": "duty", "teachers": "teachers",
        "appeals": "appeals", "holidays": "holidays", "service_hours": "service_hours",
        "office_areas": "office_areas"
    }

    EXPECTED_COLUMNS = [
        "日期", "週次", "班級", "評分項目", "檢查人員",
        "內掃原始分", "外掃原始分", "垃圾原始分", "垃圾內掃原始分", "垃圾外掃原始分", "晨間打掃原始分", "手機人數",
        "備註", "違規細項", "照片路徑", "登錄時間", "修正", "晨掃未到者", "紀錄ID"
    ]
    APPEAL_COLUMNS = ["申訴日期", "班級", "違規日期", "違規項目", "原始扣分", "申訴理由", "佐證照片", "處理狀態", "登錄時間", "對應紀錄ID"]

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
        sheet = client.open_by_url(SHEET_URL)
        for attempt in range(3):
            try:
                try: return sheet.worksheet(tab_name)
                except gspread.WorksheetNotFound:
                    cols = 20 if tab_name != "appeals" else 15
                    ws = sheet.add_worksheet(title=tab_name, rows=500, cols=cols)
                    if tab_name == "appeals": ws.append_row(APPEAL_COLUMNS)
                    if tab_name == "service_hours": ws.append_row(["日期", "學號", "班級", "類別", "時數", "紀錄ID"])
                    if tab_name == "holidays": ws.append_row(["日期", "說明"])
                    if tab_name == "office_areas": ws.append_row(["區域名稱", "負責班級"])
                    return ws
            except Exception as e:
                if "429" in str(e): time.sleep(2 * (attempt + 1)); continue
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
            folder_id = st.secrets["system_config"]["drive_folder_id"]
            file = service.files().create(
                body={'name': filename, 'parents': [folder_id]},
                media_body=MediaIoBaseUpload(file_obj, mimetype='image/jpeg', resumable=True),
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
    # SQLite 背景佇列
    # ==========================================
    _queue_lock = threading.Lock()

    @st.cache_resource
    def get_queue_connection():
        conn = sqlite3.connect(QUEUE_DB_PATH, check_same_thread=False, timeout=30.0, isolation_level="IMMEDIATE")
        try: conn.execute("PRAGMA journal_mode=WAL;"); conn.execute("PRAGMA busy_timeout=30000;")
        except: pass
        conn.execute("CREATE TABLE IF NOT EXISTS task_queue (id TEXT PRIMARY KEY, task_type TEXT, created_ts TEXT, payload_json TEXT, status TEXT, attempts INTEGER, last_error TEXT)")
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
            cur.execute("UPDATE task_queue SET status = 'IN_PROGRESS', attempts = attempts + 1 WHERE id = ?", (row[0],))
            conn.commit()
            return {"id": row[0], "task_type": row[1], "payload": json.loads(row[3]) if row[3] else {}, "attempts": row[5] + 1}

    def update_task_status(task_id, status, attempts, last_error):
        with _queue_lock:
            get_queue_connection().execute("UPDATE task_queue SET status = ?, attempts = ?, last_error = ? WHERE id = ?", (status, attempts, last_error, task_id))
            get_queue_connection().commit()

    # ==========================================
    # 背景處理邏輯
    # ==========================================
    def process_task(task):
        task_type, payload = task["task_type"], task["payload"]
        entry = payload.get("entry", {})

        try:
            image_paths, filenames, drive_links = payload.get("image_paths", []), payload.get("filenames", []), []
            for path, fname in zip(image_paths, filenames):
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        drive_links.append(upload_image_to_drive(compress_image_bytes(f.read()), fname) or "UPLOAD_FAILED_API")
            if drive_links: entry["照片路徑"] = ";".join(drive_links)

            if task_type in ["main_entry", "volunteer_report"]:
                def _main_act():
                    ws = get_worksheet(SHEET_TABS["main"])
                    if not ws.get_all_values(): ws.append_row(EXPECTED_COLUMNS)
                    ws.append_row([str(entry.get(col, "")).upper() if isinstance(entry.get(col, ""), bool) else str(entry.get(col, "")) for col in EXPECTED_COLUMNS])
                execute_with_retry(_main_act)

                inspector_name = entry.get("檢查人員", "")
                if "學號:" in inspector_name:
                    sid = inspector_name.split("學號:")[1].strip()
                    execute_with_retry(lambda: get_worksheet(SHEET_TABS["service_hours"]).append_row([entry.get("日期"), sid, "", "衛生糾察", 0.5, uuid.uuid4().hex[:8]]))
                
                if task_type == "volunteer_report":
                    for sid in payload.get("student_list", []):
                        execute_with_retry(lambda: get_worksheet(SHEET_TABS["service_hours"]).append_row([entry.get("日期", str(date.today())), sid, entry.get("班級", ""), payload.get("custom_category", "晨掃志工"), payload.get("custom_hours", 0.5), uuid.uuid4().hex[:8]]))

            elif task_type == "appeal_entry":
                image_info = payload.get("image_file")
                if image_info and os.path.exists(image_info["path"]):
                    with open(image_info["path"], "rb") as f:
                        entry["佐證照片"] = upload_image_to_drive(compress_image_bytes(f.read()), image_info["filename"])
                execute_with_retry(lambda: get_worksheet(SHEET_TABS["appeals"]).append_row([str(entry.get(c, "")) for c in APPEAL_COLUMNS]))
            return True, None
        except Exception as e: return False, str(e)

    def background_worker(stop_event=None):
        try: add_script_run_ctx(threading.current_thread(), get_script_run_ctx())
        except: pass
        while True:
            if stop_event and stop_event.is_set(): break
            try:
                task = fetch_next_task()
                if not task: time.sleep(2.0); continue
                ok, err = process_task(task)
                
                try:
                    paths = task["payload"].get("image_paths", []) + ([task["payload"]["image_file"]["path"]] if "image_file" in task["payload"] else [])
                    for p in paths:
                        if p and os.path.exists(p): os.remove(p)
                except: pass

                update_task_status(task["id"], "DONE" if ok else ("FAILED" if task["attempts"] >= 6 else "RETRY"), task["attempts"], err)
                time.sleep(0.5)
            except Exception as e: time.sleep(3.0)

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

    @st.cache_data(ttl=300)
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
        image_info = None
        if proof_file:
            try:
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
    
    def update_appeal_status(idx, status, record_id):
        ws_appeals, ws_main = get_worksheet(SHEET_TABS["appeals"]), get_worksheet(SHEET_TABS["main"])
        try:
            data = ws_appeals.get_all_records()
            t_row = next((i + 2 for i, r in enumerate(data) if str(r.get("對應紀錄ID")) == str(record_id) and str(r.get("處理狀態")) == "待處理"), None)
            if t_row:
                ws_appeals.update_cell(t_row, APPEAL_COLUMNS.index("處理狀態") + 1, status)
                if status == "已核可":
                    m_data = ws_main.get_all_records()
                    m_row = next((j + 2 for j, mr in enumerate(m_data) if str(mr.get("紀錄ID")) == str(record_id)), None)
                    if m_row: ws_main.update_cell(m_row, EXPECTED_COLUMNS.index("修正") + 1, "TRUE")
                st.cache_data.clear(); return True, "更新成功"
            return False, "找不到對應的申訴列"
        except Exception as e: return False, str(e)
    
    @st.cache_data(ttl=21600)
    def load_teacher_emails():
        ws = get_worksheet(SHEET_TABS["teachers"])
        if not ws: return {}
        try:
            df = pd.DataFrame(ws.get_all_records())
            c_col, m_col, n_col = next((c for c in df.columns if "班級" in c), None), next((c for c in df.columns if "Email" in c or "信箱" in c), None), next((c for c in df.columns if "導師" in c or "姓名" in c), None)
            return {str(row[c_col]).strip(): {"email": str(row[m_col]).strip(), "name": str(row[n_col]).strip() if n_col else "老師"} for _, row in df.iterrows() if c_col and m_col and "@" in str(row[m_col])}
        except: return {}

    def send_bulk_emails(email_list):
        s_email, s_pwd = st.secrets["system_config"]["smtp_email"], st.secrets["system_config"]["smtp_password"]
        if not s_email or not s_pwd: return 0, "Secrets 未設定 Email"
        cnt = 0
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587); server.starttls(); server.login(s_email, s_pwd)
            for item in email_list:
                try:
                    msg = MIMEMultipart()
                    msg['From'], msg['To'], msg['Subject'] = s_email, item['email'], item['subject']
                    msg.attach(MIMEText(item['body'], 'plain'))
                    server.sendmail(s_email, item['email'], msg.as_string())
                    cnt += 1
                except: pass
            server.quit(); return cnt, "發送作業結束"
        except Exception as e: return cnt, str(e)

    def delete_rows_by_ids(ids):
        ws = get_worksheet(SHEET_TABS["main"])
        if not ws: return False
        try:
            rows = sorted([i + 2 for i, r in enumerate(ws.get_all_records()) if str(r.get("紀錄ID")) in ids], reverse=True)
            for r in rows: ws.delete_rows(r)
            time.sleep(0.8); st.cache_data.clear(); return True
        except Exception as e: st.error(f"刪除失敗: {e}"); return False

    @st.cache_data(ttl=21600)
    def load_inspector_list():
        ws = get_worksheet(SHEET_TABS["inspectors"])
        default = [{"label": "測試人員", "allowed_roles": ["內掃檢查"], "assigned_classes": [], "id_prefix": "測"}]
        if not ws: return default
        try:
            df = pd.DataFrame(ws.get_all_records())
            if df.empty: return default
            inspectors, id_c, r_c, s_c = [], next((c for c in df.columns if "學號" in c or "編號" in c), None), next((c for c in df.columns if "負責" in c or "項目" in c), None), next((c for c in df.columns if "班級" in c or "範圍" in c), None)
            if id_c:
                for _, row in df.iterrows():
                    sid, s_role = clean_id(row[id_c]), str(row[r_c]).strip() if r_c else ""
                    allowed = ["內掃檢查", "外掃檢查", "垃圾/回收檢查", "晨間打掃"] if "組長" in s_role else (["內掃檢查", "外掃檢查", "垃圾/回收檢查"] if "機動" in s_role else [r for r in ["外掃檢查", "垃圾/回收檢查", "晨間打掃", "內掃檢查"] if r[:2] in s_role])
                    s_classes = [c.strip() for c in str(row[s_c]).replace("、", ";").replace(",", ";").split(";") if c.strip()] if s_c and str(row[s_c]) else []
                    inspectors.append({"label": f"學號: {sid}", "allowed_roles": allowed or ["內掃檢查"], "assigned_classes": s_classes, "id_prefix": sid[0] if sid else "X"})
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
        new_entry["日期"] = str(new_entry.get("日期", str(date.today())))
        new_entry["紀錄ID"] = new_entry.get("紀錄ID", f"{datetime.now(TW_TZ).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}")

        image_paths, file_names = [], []
        if uploaded_files:
            for i, up_file in enumerate(uploaded_files):
                if not up_file: continue
                try:
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
        return enqueue_task("volunteer_report" if student_list is not None else "main_entry", payload)

    def load_full_semester_data_for_export():
        ws = get_worksheet(SHEET_TABS["main"])
        if not ws: return pd.DataFrame(columns=EXPECTED_COLUMNS)
        try:
            df = pd.DataFrame(ws.get_all_records())
            if df.empty: return pd.DataFrame(columns=EXPECTED_COLUMNS)
            for col in EXPECTED_COLUMNS:
                if col not in df.columns: df[col] = ""
            for col in ["備註", "違規細項", "班級", "檢查人員", "修正", "晨掃未到者", "照片路徑", "紀錄ID"]:
                if col in df.columns: df[col] = df[col].fillna("").astype(str)
            for col in ["內掃原始分", "外掃原始分", "垃圾原始分", "垃圾內掃原始分", "垃圾外掃原始分", "晨間打掃原始分", "手機人數", "週次"]:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            return df[EXPECTED_COLUMNS]
        except: return pd.DataFrame()

    # ==========================================
    # 3. 主程式 UI 啟動前準備
    # ==========================================
    
    # 🚨 [修正點 1] 補回被刪掉的 now_tw 定義
    now_tw = datetime.now(TW_TZ)
    today_tw = now_tw.date()
    
    SYSTEM_CONFIG, ROSTER_DICT, INSPECTOR_LIST, TEACHER_MAILS = load_settings(), load_roster_dict(), load_inspector_list(), load_teacher_emails()
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
    app_mode = st.sidebar.radio("請選擇模式", ["糾察底家👀", "班級負責人🥸", "晨掃志工隊🧹", "組長ㄉ窩💃"])

    with st.sidebar.expander("🔧 系統狀態 (名單異常請點此)", expanded=True):
        if get_gspread_client(): st.success("✅ Google Sheets 連線正常")
        else: st.error("❌ Google Sheets 連線失敗")
        if st.button("🔄 重讀名單 (清除快取)"): st.cache_data.clear(); st.rerun()

    # --- Mode 1: 糾察評分 ---
    if app_mode == "糾察底家👀":
        st.title("📝 衛生糾察評分系統")
        if "team_logged_in" not in st.session_state: st.session_state["team_logged_in"] = False
        
        if not st.session_state["team_logged_in"]:
            with st.expander("🔐 身份驗證", expanded=True):
                if st.button("登入") if st.text_input("請輸入隊伍通行碼", type="password") == st.secrets["system_config"]["team_password"] else False:
                    st.session_state["team_logged_in"] = True; st.rerun()
        
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
                    st.info("🗑️ 資收場專用：負面表列模式 (有違規才打勾，系統將自動記錄扣分)")
                    
                    sel_filter = st.radio("篩選檢查對象", ["各處室 (外掃)"] + grades, horizontal=True, key="m1_trash_filter")
                    today_records = main_df[(main_df["日期"].astype(str) == str(input_date)) & (main_df["評分項目"] == "垃圾/回收檢查")] if not main_df.empty else pd.DataFrame()
                    rows = []
                    
                    if sel_filter == "各處室 (外掃)":
                        office_map = load_office_area_map()
                        for off_name in list(office_map.keys()) or ["教務處", "學務處", "總務處", "輔導室", "圖書館"]:
                            cls_name = office_map.get(off_name, "未設定")
                            is_gen_bad = any(f"外掃({off_name})" in str(r["備註"]) and "未分類" in str(r["備註"]) and "一般" in str(r["備註"]) for _, r in today_records.iterrows()) if not today_records.empty else False
                            is_recyc_bad = any(f"外掃({off_name})" in str(r["備註"]) and ("未分類" in str(r["備註"]) or "未倒" in str(r["備註"])) and "回收" in str(r["備註"]) for _, r in today_records.iterrows()) if not today_records.empty else False
                            rows.append({"處室/區域": off_name, "負責班級": cls_name, "一般-未分類": is_gen_bad, "回收-未倒/未分類": is_recyc_bad})
                            
                        edited_df = st.data_editor(pd.DataFrame(rows), column_config={"處室/區域": st.column_config.TextColumn(disabled=True), "負責班級": st.column_config.TextColumn(disabled=True)}, hide_index=True, use_container_width=True, key="ed_offices")
                        if st.button("💾 登記違規 (各處室)"):
                            cnt = 0
                            for _, row in edited_df.iterrows():
                                off, cls, gen_bad, recyc_bad = row["處室/區域"], row["負責班級"], row["一般-未分類"], row["回收-未倒/未分類"]
                                orig = next((x for x in rows if x["處室/區域"] == off), None)
                                base = {"日期": input_date, "週次": week_num, "檢查人員": inspector_name, "班級": cls, "評分項目": role, "垃圾內掃原始分": 0, "垃圾外掃原始分": 1}
                                if gen_bad and not orig["一般-未分類"]: save_entry({**base, "備註": f"外掃({off})-一般未分類", "違規細項": "一般垃圾"}); cnt += 1
                                if recyc_bad and not orig["回收-未倒/未分類"]: save_entry({**base, "備註": f"外掃({off})-回收未倒/未分類", "違規細項": "資源回收"}); cnt += 1
                            if cnt: st.success(f"✅ 已登記 {cnt} 筆違規！"); time.sleep(1); st.rerun()

                    else:
                        for cls_name in [c["name"] for c in structured_classes if c["grade"] == sel_filter]:
                            cls_rec = today_records[today_records["班級"] == cls_name] if not today_records.empty else pd.DataFrame()
                            is_gen_bad = any("內掃" in str(r["備註"]) and "未分類" in str(r["備註"]) and "一般" in str(r["備註"]) for _, r in cls_rec.iterrows()) if not cls_rec.empty else False
                            is_recyc_bad = any("內掃" in str(r["備註"]) and ("未分類" in str(r["備註"]) or "未倒" in str(r["備註"])) and "回收" in str(r["備註"]) for _, r in cls_rec.iterrows()) if not cls_rec.empty else False
                            rows.append({"班級": cls_name, "一般-未分類": is_gen_bad, "回收-未倒/未分類": is_recyc_bad})
                            
                        edited_df = st.data_editor(pd.DataFrame(rows), column_config={"班級": st.column_config.TextColumn(disabled=True)}, hide_index=True, use_container_width=True, key=f"ed_{sel_filter}")
                        if st.button(f"💾 登記違規 ({sel_filter})"):
                            cnt = 0
                            for _, row in edited_df.iterrows():
                                cls, gen_bad, recyc_bad = row["班級"], row["一般-未分類"], row["回收-未倒/未分類"]
                                orig = next((x for x in rows if x["班級"] == cls), None)
                                base = {"日期": input_date, "週次": week_num, "檢查人員": inspector_name, "班級": cls, "評分項目": role, "垃圾內掃原始分": 1, "垃圾外掃原始分": 0}
                                if gen_bad and not orig["一般-未分類"]: save_entry({**base, "備註": "內掃-一般未分類", "違規細項": "一般垃圾"}); cnt += 1
                                if recyc_bad and not orig["回收-未倒/未分類"]: save_entry({**base, "備註": "內掃-回收未倒/未分類", "違規細項": "資源回收"}); cnt += 1
                            if cnt: st.success(f"✅ 已登記 {cnt} 筆違規！"); time.sleep(1); st.rerun()

                else:
                    assigned_classes = curr_inspector.get("assigned_classes", [])
                    
                    # 🚨 [修正點 2] 解開導致當機的 list comprehension radio 寫法
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
                                if (in_s + out_s) > 0 and not files: st.error("扣分需照片")
                                else:
                                    save_entry({"日期": input_date, "週次": week_num, "檢查人員": inspector_name, "修正": is_fix, "班級": sel_cls, "評分項目": role, "內掃原始分": in_s, "外掃原始分": out_s, "手機人數": ph_c, "備註": note}, uploaded_files=files)
                                    st.success("✅ 送出成功！"); st.rerun()

    # --- Mode 2: 班級負責人 ---
    elif app_mode == "班級負責人🥸":
        st.title("🔎 班級成績查詢")
        df, appeals_df = load_main_data(), load_appeals()
        appeal_map = {str(r.get("對應紀錄ID")): r.get("處理狀態") for _, r in appeals_df.iterrows()} if not appeals_df.empty else {}
        
        # 🚨 [修正點 2] 解開導致當機的 list comprehension radio 寫法
        sel_grade_m2 = st.radio("選擇年級", grades, horizontal=True, key="m2_grade_select")
        cls_opts = [c["name"] for c in structured_classes if c["grade"] == sel_grade_m2]
        
        if cls_opts:
            cls = st.selectbox("選擇班級", cls_opts, key="m2_cls_select")
            if cls and not df.empty:
                for idx, r in df[df["班級"] == cls].sort_values("登錄時間", ascending=False).iterrows():
                    trash_score = r['垃圾內掃原始分'] + r['垃圾外掃原始分']
                    if trash_score == 0: trash_score = r['垃圾原始分']
                    
                    tot = r['內掃原始分'] + r['外掃原始分'] + trash_score + r['晨間打掃原始分']
                    rid, ap_st = str(r['紀錄ID']), appeal_map.get(str(r['紀錄ID']))
                    icon = "✅" if ap_st=="已核可" else "🚫" if ap_st=="已駁回" else "⏳" if ap_st=="待處理" else "🛠️" if str(r['修正'])=="TRUE" else ""
                    with st.expander(f"{icon} {r['日期']} - {r['評分項目']} (扣:{tot})"):
                        st.write(f"備註: {r['備註']}")
                        if str(r['照片路徑']) and "http" in str(r['照片路徑']): st.image([p for p in str(r['照片路徑']).split(";") if "http" in p], width=200)
                        if not ap_st and is_within_appeal_period(r['日期']) and (tot > 0 or r['手機人數'] > 0):
                            with st.form(f"ap_{rid}"):
                                rsn, pf = st.text_area("理由"), st.file_uploader("佐證", type=['jpg','png'])
                                if st.form_submit_button("申訴") and rsn and pf:
                                    save_appeal({"班級": cls, "違規日期": str(r["日期"]), "違規項目": r['評分項目'], "原始扣分": str(tot), "申訴理由": rsn, "對應紀錄ID": rid}, pf)
                                    st.rerun()

    # --- Mode 3: 晨掃志工隊 ---
    elif app_mode == "晨掃志工隊🧹":
        st.title("🧹 晨掃志工回報專區")
        if now_tw.hour >= 16: st.error("🚫 今日回報已截止 (16:00)")
        else:
            my_cls = st.selectbox("選擇班級", all_classes, key="m3_cls_select")
            main_df = load_main_data()
            if not main_df[(main_df["日期"].astype(str)==str(today_tw)) & (main_df["班級"]==my_cls) & (main_df["評分項目"]=="晨間打掃")].empty: st.warning(f"⚠️ {my_cls} 已回報！")
            else:
                duty_df, _ = get_daily_duty(today_tw)
                area_name = "無"
                n_std = 4
                if not duty_df.empty:
                    m_d = duty_df[duty_df["負責班級"]==my_cls]
                    if not m_d.empty:
                        area_name = m_d.iloc[0].get('掃地區域', '無')
                        # 加入防錯機制，即使 N 被刪掉也能抓到對應欄位
                        try:
                            n_std = int(m_d.iloc[0].get('標準人數', 4))
                        except:
                            n_std = 4
                
                st.info(f"📍 任務: {area_name} (應到:{n_std}人)")
                with st.form("vol_form"):
                    present = st.multiselect("✅ 實到同學", [s for s, c in ROSTER_DICT.items() if c == my_cls])
                    files = st.file_uploader("📸 成果照片", accept_multiple_files=True, type=['jpg','png'])
                    if st.form_submit_button("送出") and present and files:
                        save_entry({"日期": str(today_tw), "班級": my_cls, "評分項目": "晨間打掃", "檢查人員": f"志工(實到:{len(present)})", "晨間打掃原始分": 0, "備註": f"名單:{','.join(present)}"}, uploaded_files=files, student_list=present, custom_hours=0.5, custom_category="晨掃志工")
                        st.success("✅ 回報成功！"); st.rerun()

    # --- Mode 4: 組長後台 ---
    elif app_mode == "組長ㄉ窩💃":
        st.title("⚙️ 管理後台")
        metrics = get_queue_metrics()
        c1, c2, c3 = st.columns(3)
        c1.metric("待處理", metrics["pending"])
        c2.metric("失敗", metrics["failed"])
        c3.metric("延遲(s)", int(metrics["oldest_pending_sec"]))

        if st.text_input("管理密碼", type="password", key="admin_pwd") == st.secrets["system_config"]["admin_password"]:
            t1, t2, t3, t4, t5, t6, t7 = st.tabs(["🧹 晨掃審核", "📊 成績總表", "🏫 返校打掃", "📝 扣分明細", "📧 寄信", "📣 申訴", "⚙️ 設定"])
            
            with t1:
                df = load_main_data()
                for i, r in df[(df["評分項目"]=="晨間打掃") & (df["晨間打掃原始分"]==0) & (df["修正"]!="TRUE")].iterrows():
                    with st.container(border=True):
                        c1, c2, c3 = st.columns([2,2,1])
                        c1.write(f"**{r['班級']}** | {r['檢查人員']}"); c2.image(str(r['照片路徑']).split(";")[0], width=150) if "http" in str(r['照片路徑']) else None
                        
                        if c3.button("✅ 通過", key=f"p_{r['紀錄ID']}"): 
                            ws = get_worksheet(SHEET_TABS["main"])
                            id_list = ws.col_values(EXPECTED_COLUMNS.index("紀錄ID")+1)
                            if str(r["紀錄ID"]) in id_list:
                                ridx = id_list.index(str(r["紀錄ID"])) + 1
                                ws.update_cell(ridx, EXPECTED_COLUMNS.index("晨間打掃原始分")+1, 2)
                                st.cache_data.clear()
                                st.rerun()
                        if c3.button("🗑️ 駁回", key=f"r_{r['紀錄ID']}"): delete_rows_by_ids([str(r["紀錄ID"])]); st.rerun()

            with t2:
                if st.button("🚀 計算全學期成績"):
                    full = load_full_semester_data_for_export()
                    full["總扣分"] = full["內掃原始分"].clip(upper=2) + full["外掃原始分"].clip(upper=2) + (full["垃圾內掃原始分"]+full["垃圾外掃原始分"]).where((full["垃圾內掃原始分"]+full["垃圾外掃原始分"])>0, full["垃圾原始分"]).clip(upper=2) + full["晨間打掃原始分"] + full["手機人數"]
                    fin = pd.merge(pd.DataFrame(structured_classes).rename(columns={"grade":"年級","name":"班級"}), full.groupby("班級")["總扣分"].sum().reset_index(), on="班級", how="left").fillna(0)
                    fin["總成績"] = 90 - fin["總扣分"]
                    st.dataframe(fin.sort_values("總成績", ascending=False))

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
                        if st.form_submit_button("發放") and pf:
                            pf.seek(0); fb = pf.read()
                            norm = [m for m in pool if m not in spec]
                            if norm: pf_n = io.BytesIO(fb); pf_n.name="p.jpg"; save_entry({"日期": str(rd), "班級": rc, "評分項目": "返校打掃"}, [pf_n], norm, base_h, "返校打掃(一般)")
                            if spec: pf_s = io.BytesIO(fb); pf_s.name="p.jpg"; save_entry({"日期": str(rd), "班級": rc, "評分項目": "返校打掃"}, [pf_s], spec, spec_h, "返校打掃(加強)")
                            st.success("已登記！"); st.rerun()

            with t4: st.dataframe(load_main_data())
            with t6:
                ap_df = load_appeals()
                for i, r in ap_df[ap_df["處理狀態"]=="待處理"].iterrows():
                    with st.container(border=True):
                        c1, c2 = st.columns([3,1])
                        c1.write(f"{r['班級']} | {r['申訴理由']}")
                        if c1.button("核可", key=f"ok_{i}"): update_appeal_status(i, "已核可", r["對應紀錄ID"]); st.rerun()
                        if c1.button("駁回", key=f"ng_{i}"): update_appeal_status(i, "已駁回", r["對應紀錄ID"]); st.rerun()

            with t7:
                st.info("系統維護區")
                if st.button("清除快取"): st.cache_data.clear(); st.success("Done")

except Exception as e:
    st.error(f"❌ 系統發生錯誤: {str(e)}")
    st.code(traceback.format_exc())
