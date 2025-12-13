import streamlit as st
import pandas as pd
import smtplib
import time
import io
import traceback
import uuid
import os
from datetime import datetime, date, timedelta
import pytz
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# ==========================================
# 1. 系統設定與全域變數
# ==========================================
st.set_page_config(page_title="中壢家商，衛愛而生", layout="wide", page_icon="🧹")

TW_TZ = pytz.timezone('Asia/Taipei')
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
    "申訴日期", "班級", "違規日期", "違規項目", "原始扣分", "申訴理由", "佐證照片", "處理狀態", "登錄時間", "對應紀錄ID", "申訴ID"
]

# ==========================================
# 2. Google API 連線 (核心)
# ==========================================

@st.cache_resource
def get_credentials():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    if "gcp_service_account" not in st.secrets:
        st.error("❌ 找不到 secrets 設定 (gcp_service_account)")
        return None
    creds_dict = dict(st.secrets["gcp_service_account"])
    return ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)

@st.cache_resource
def get_gspread_client():
    try:
        creds = get_credentials()
        if not creds: return None
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ Google Sheet 連線失敗: {e}")
        return None

@st.cache_resource
def get_drive_service():
    try:
        creds = get_credentials()
        if not creds: return None
        return build('drive', 'v3', credentials=creds, cache_discovery=False)
    except Exception as e:
        st.warning(f"⚠️ Google Drive 連線失敗: {e}")
        return None

@st.cache_resource(ttl=3600)
def get_spreadsheet_object():
    client = get_gspread_client()
    if not client: return None
    try: return client.open_by_url(SHEET_URL)
    except Exception as e: st.error(f"❌ 無法開啟試算表: {e}")
    return None

def get_worksheet(tab_name):
    """取得工作表，若不存在則自動建立"""
    sheet = get_spreadsheet_object()
    if not sheet: return None
    try:
        return sheet.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        # 自動建立分頁
        cols = 20 if tab_name != "appeals" else 15
        ws = sheet.add_worksheet(title=tab_name, rows=100, cols=cols)
        if tab_name == "appeals": ws.append_row(APPEAL_COLUMNS)
        elif tab_name == "main": ws.append_row(EXPECTED_COLUMNS)
        return ws
    except Exception as e:
        st.error(f"❌ 讀取分頁 '{tab_name}' 失敗: {e}")
        return None

# ==========================================
# 3. 同步處理核心 (取代原本的 Queue)
# ==========================================

def upload_image_direct(file_obj, filename):
    """
    直接從記憶體上傳到 Google Drive (不存本地暫存檔)
    """
    try:
        service = get_drive_service()
        if not service: return "SERVICE_ERROR"
        
        folder_id = st.secrets["system_config"].get("drive_folder_id")
        if not folder_id: return "NO_FOLDER_ID"

        # 確保指標在開頭
        file_obj.seek(0)
        
        file_metadata = {'name': filename, 'parents': [folder_id]}
        # 使用 resumable=True 提高大檔傳輸穩定性
        media = MediaIoBaseUpload(file_obj, mimetype='image/jpeg', resumable=True)
        
        file = service.files().create(
            body=file_metadata, media_body=media, fields='id', supportsAllDrives=True
        ).execute()
        
        # 開放權限供檢視
        try:
            service.permissions().create(fileId=file.get('id'), body={'role': 'reader', 'type': 'anyone'}).execute()
        except: pass 
        
        return f"https://drive.google.com/thumbnail?id={file.get('id')}&sz=w1000"
    except Exception as e:
        print(f"Drive Upload Error: {e}")
        return "UPLOAD_FAILED"

def save_entry_sync(new_entry, uploaded_files=None):
    """
    同步儲存流程：上傳照片 -> 寫入 Sheet -> 回傳結果
    此函式會阻斷執行直到完成，確保資料一致性。
    """
    # 使用 st.status 給予使用者明確的進度回饋
    status_container = st.status("☁️ 資料處理中...", expanded=True)
    
    try:
        # 1. 處理照片
        drive_links = []
        if uploaded_files:
            status_container.write("📸 正在上傳照片到雲端...")
            for i, up_file in enumerate(uploaded_files):
                # 產生邏輯檔名
                safe_cls = str(new_entry.get('班級', 'unknown'))
                fname = f"{new_entry['日期']}_{safe_cls}_{i+1}.jpg"
                
                link = upload_image_direct(up_file, fname)
                drive_links.append(link)
        
        new_entry["照片路徑"] = ";".join(drive_links)

        # 2. 補齊 ID 與欄位
        if "紀錄ID" not in new_entry:
            new_entry["紀錄ID"] = f"{datetime.now(TW_TZ).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:4]}"

        row_values = []
        for col in EXPECTED_COLUMNS:
            val = new_entry.get(col, "")
            if isinstance(val, bool): val = str(val).upper()
            row_values.append(str(val))

        # 3. 寫入 Google Sheet
        status_container.write("📝 正在寫入資料庫...")
        ws = get_worksheet(SHEET_TABS["main"])
        ws.append_row(row_values)
        
        # 4. 完成
        st.cache_data.clear() # 清除快取，確保下次讀取到最新資料
        status_container.update(label="✅ 資料已成功儲存！", state="complete", expanded=False)
        time.sleep(1) # 稍作停留讓使用者看到綠勾勾
        return True

    except Exception as e:
        status_container.update(label="❌ 儲存失敗", state="error", expanded=False)
        st.error(f"寫入錯誤: {e}")
        return False

def save_appeal_sync(entry, proof_file):
    """同步儲存申訴單"""
    status_container = st.status("📨 正在提交申訴...", expanded=True)
    
    try:
        # 1. 上傳佐證
        if proof_file:
            status_container.write("📸 上傳佐證照片...")
            fname = f"Appeal_{entry.get('班級','')}_{datetime.now(TW_TZ).strftime('%H%M%S')}.jpg"
            link = upload_image_direct(proof_file, fname)
            entry["佐證照片"] = link
        else:
            entry["佐證照片"] = ""

        # 2. 寫入 Sheet
        status_container.write("📝 寫入申訴紀錄...")
        if "申訴ID" not in entry:
            entry["申訴ID"] = f"{datetime.now(TW_TZ).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:4]}"

        row_values = [str(entry.get(col, "")) for col in APPEAL_COLUMNS]
        ws = get_worksheet(SHEET_TABS["appeals"])
        ws.append_row(row_values)
        
        st.cache_data.clear()
        status_container.update(label="✅ 申訴已送出！", state="complete", expanded=False)
        return True
    except Exception as e:
        status_container.update(label="❌ 提交失敗", state="error", expanded=False)
        st.error(f"錯誤: {e}")
        return False

# ==========================================
# 4. 資料讀取與輔助函式
# ==========================================

def clean_id(val):
    try:
        if pd.isna(val) or val == "": return ""
        return str(int(float(val))).strip()
    except: return str(val).strip()

@st.cache_data(ttl=60)
def load_main_data():
    ws = get_worksheet(SHEET_TABS["main"])
    if not ws: return pd.DataFrame(columns=EXPECTED_COLUMNS)
    try:
        data = ws.get_all_records()
        df = pd.DataFrame(data)
        if df.empty: return pd.DataFrame(columns=EXPECTED_COLUMNS)
        
        # 補齊欄位
        for col in EXPECTED_COLUMNS:
            if col not in df.columns: df[col] = ""
            
        # 型別轉換
        df["紀錄ID"] = df["紀錄ID"].astype(str)
        numeric_cols = ["內掃原始分", "外掃原始分", "垃圾原始分", "晨間打掃原始分", "手機人數"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        
        if "週次" in df.columns:
            df["週次"] = pd.to_numeric(df["週次"], errors="coerce").fillna(0).astype(int)
            
        return df[EXPECTED_COLUMNS]
    except Exception as e:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

@st.cache_data(ttl=60)
def load_appeals():
    ws = get_worksheet(SHEET_TABS["appeals"])
    if not ws: return pd.DataFrame(columns=APPEAL_COLUMNS)
    try:
        df = pd.DataFrame(ws.get_all_records())
        for col in APPEAL_COLUMNS:
            if col not in df.columns: df[col] = ""
        return df
    except: return pd.DataFrame(columns=APPEAL_COLUMNS)

def delete_rows_by_ids(record_ids_to_delete):
    ws = get_worksheet(SHEET_TABS["main"])
    if not ws: return False
    try:
        records = ws.get_all_records()
        rows_to_delete = []
        # 注意：get_all_records 不含標題，所以 row index 從 2 開始
        for i, record in enumerate(records):
            if str(record.get("紀錄ID")) in record_ids_to_delete:
                rows_to_delete.append(i + 2)
        
        # 從後面開始刪除，避免 index 跑掉
        rows_to_delete.sort(reverse=True)
        for row_idx in rows_to_delete:
            ws.delete_rows(row_idx)
            time.sleep(0.5) # 避免 API 限制
        
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"刪除失敗: {e}")
        return False

def update_appeal_status(appeal_id, status, related_id):
    ws_app = get_worksheet(SHEET_TABS["appeals"])
    ws_main = get_worksheet(SHEET_TABS["main"])
    
    try:
        # 更新申訴表
        app_data = ws_app.get_all_records()
        target_row = None
        # 尋找對應的 row
        for i, row in enumerate(app_data):
            # 這裡用申訴ID或對應紀錄ID來找都可以，這裡邏輯維持原本
            if str(row.get("對應紀錄ID")) == str(related_id) and str(row.get("處理狀態")) == "待處理":
                target_row = i + 2
                break
        
        if target_row:
            col_idx = APPEAL_COLUMNS.index("處理狀態") + 1
            ws_app.update_cell(target_row, col_idx, status)
            
            # 如果核可，去更新主表
            if status == "已核可" and related_id:
                main_data = ws_main.get_all_records()
                m_row_idx = None
                for j, m_row in enumerate(main_data):
                    if str(m_row.get("紀錄ID")) == str(related_id):
                        m_row_idx = j + 2
                        break
                if m_row_idx:
                    fix_idx = EXPECTED_COLUMNS.index("修正") + 1
                    ws_main.update_cell(m_row_idx, fix_idx, "TRUE")
            
            st.cache_data.clear()
            return True, "更新成功"
        return False, "找不到對應申訴"
    except Exception as e: return False, str(e)

# --- 名單讀取 ---
@st.cache_data(ttl=21600)
def load_roster_data():
    """一次讀取並整理所有靜態名單"""
    res = {"classes": [], "structured": [], "roster_dict": {}, "inspectors": [], "teachers": {}}
    
    # 1. 班級名單
    ws_roster = get_worksheet(SHEET_TABS["roster"])
    if ws_roster:
        try:
            df = pd.DataFrame(ws_roster.get_all_records())
            if not df.empty:
                class_col = next((c for c in df.columns if "班級" in c), None)
                id_col = next((c for c in df.columns if "學號" in c), None)
                
                if class_col:
                    unique = sorted(df[class_col].dropna().unique().tolist())
                    res["classes"] = [str(c).strip() for c in unique if str(c).strip()]
                    
                    # 排序邏輯
                    dept_order = {"商":1, "英":2, "資":3, "家":4, "服":5}
                    def sort_key(name):
                        g = 99
                        if "一" in name or "1" in name: g=1
                        elif "二" in name or "2" in name: g=2
                        elif "三" in name or "3" in name: g=3
                        d = 99
                        for k,v in dept_order.items():
                            if k in name: d=v; break
                        return (g, d, name)
                    
                    res["classes"].sort(key=sort_key)
                    for c in res["classes"]:
                        g_val = sort_key(c)[0]
                        g_lbl = f"{g_val}年級" if g_val!=99 else "其他"
                        res["structured"].append({"grade": g_lbl, "name": c})
                
                if id_col and class_col:
                    for _, r in df.iterrows():
                        sid = clean_id(r[id_col])
                        if sid: res["roster_dict"][sid] = str(r[class_col]).strip()
        except: pass

    # 2. 糾察名單
    ws_insp = get_worksheet(SHEET_TABS["inspectors"])
    if ws_insp:
        try:
            df = pd.DataFrame(ws_insp.get_all_records())
            if not df.empty:
                id_col = next((c for c in df.columns if "學號" in c), None)
                role_col = next((c for c in df.columns if "負責" in c), None)
                scope_col = next((c for c in df.columns if "班級" in c), None)
                
                if id_col:
                    for _, r in df.iterrows():
                        sid = clean_id(r[id_col])
                        roles = str(r[role_col]).strip() if role_col else ""
                        scopes = str(r[scope_col]).strip() if scope_col else ""
                        
                        allowed = ["內掃檢查"]
                        if "組長" in roles: allowed = ["內掃檢查", "外掃檢查", "垃圾/回收檢查", "晨間打掃"]
                        elif "機動" in roles: allowed = ["內掃檢查", "外掃檢查", "垃圾/回收檢查"]
                        else:
                            tmp = []
                            if "外掃" in roles: tmp.append("外掃檢查")
                            if "垃圾" in roles: tmp.append("垃圾/回收檢查")
                            if "晨" in roles: tmp.append("晨間打掃")
                            if "內掃" in roles: tmp.append("內掃檢查")
                            if tmp: allowed = tmp
                        
                        s_list = [x.strip() for x in scopes.replace("、",";").replace(",",";").split(";") if x.strip()]
                        
                        res["inspectors"].append({
                            "label": f"學號: {sid}",
                            "id_prefix": sid[0] if sid else "X",
                            "allowed_roles": allowed,
                            "assigned_classes": s_list
                        })
        except: pass

    # 3. 導師名單
    ws_teach = get_worksheet(SHEET_TABS["teachers"])
    if ws_teach:
        try:
            df = pd.DataFrame(ws_teach.get_all_records())
            if not df.empty:
                c_col = next((c for c in df.columns if "班級" in c), None)
                m_col = next((c for c in df.columns if "Email" in c or "信箱" in c), None)
                n_col = next((c for c in df.columns if "導師" in c or "姓名" in c), None)
                if c_col and m_col:
                    for _, r in df.iterrows():
                        if "@" in str(r[m_col]):
                            res["teachers"][str(r[c_col]).strip()] = {
                                "email": str(r[m_col]).strip(),
                                "name": str(r[n_col]).strip() if n_col else "老師"
                            }
        except: pass
    
    return res

@st.cache_data(ttl=21600)
def load_settings():
    ws = get_worksheet(SHEET_TABS["settings"])
    config = {"semester_start": "2025-08-25"}
    if ws:
        try:
            data = ws.get_all_values()
            for r in data:
                if len(r)>=2 and r[0]=="semester_start": config["semester_start"] = r[1]
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
        except: pass
    return False

def check_duplicate_record(df, date_val, inspector, role, cls):
    if df.empty: return False
    d_str = str(date_val)
    mask = (df["日期"].astype(str) == d_str) & (df["檢查人員"] == inspector) & (df["評分項目"] == role) & (df["班級"] == cls)
    return not df[mask].empty

def get_week_num(d):
    try:
        conf = load_settings()
        start = datetime.strptime(conf["semester_start"], "%Y-%m-%d").date()
        if isinstance(d, datetime): d = d.date()
        return max(0, ((d - start).days // 7) + 1)
    except: return 0

def send_bulk_emails(email_list):
    sender_email = st.secrets["system_config"]["smtp_email"]
    sender_password = st.secrets["system_config"]["smtp_password"]
    sent = 0
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        for item in email_list:
            try:
                msg = MIMEMultipart()
                msg['From'] = sender_email
                msg['To'] = item['email']
                msg['Subject'] = item['subject']
                msg.attach(MIMEText(item['body'], 'plain'))
                server.sendmail(sender_email, item['email'], msg.as_string())
                sent += 1
            except: pass
        server.quit()
        return sent, "完成"
    except Exception as e: return sent, str(e)

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

# ==========================================
# 5. 主程式介面
# ==========================================

ALL_DATA = load_roster_data()
INSPECTOR_LIST = ALL_DATA["inspectors"]
TEACHER_MAILS = ALL_DATA["teachers"]
STRUCTURED_CLASSES = ALL_DATA["structured"]
ROSTER_DICT = ALL_DATA["roster_dict"]
GRADES = sorted(list(set([c["grade"] for c in STRUCTURED_CLASSES])))
all_classes_list = ALL_DATA["classes"]

st.sidebar.title("🏫 功能選單")
app_mode = st.sidebar.radio("請選擇模式", ["糾察底家👀", "班級負責人🥸", "組長ㄉ窩💃"])

# --- 模式1: 糾察評分 ---
if app_mode == "糾察底家👀":
    st.title("📝 衛生糾察評分系統")
    
    if "team_logged_in" not in st.session_state: st.session_state["team_logged_in"] = False
    
    if not st.session_state["team_logged_in"]:
        with st.expander("🔐 身份驗證", expanded=True):
            pwd = st.text_input("請輸入隊伍通行碼", type="password")
            if st.button("登入"):
                if pwd == st.secrets["system_config"]["team_password"]:
                    st.session_state["team_logged_in"] = True
                    st.rerun()
                else: st.error("通行碼錯誤")
    else:
        # 選擇檢查人員
        prefixes = sorted(list(set([p["id_prefix"] for p in INSPECTOR_LIST])))
        if not prefixes:
            st.warning("⚠️ 名單未載入，請確認後台 inspectors 分頁")
        else:
            sel_prefix_lbl = st.radio("步驟 1：選擇開頭", [f"{p}開頭" for p in prefixes], horizontal=True)
            sel_prefix = sel_prefix_lbl[0]
            
            filtered_insp = [p for p in INSPECTOR_LIST if p["id_prefix"] == sel_prefix]
            insp_name = st.radio("步驟 2：點選身份", [p["label"] for p in filtered_insp])
            
            curr_insp = next((p for p in INSPECTOR_LIST if p["label"] == insp_name), None)
            roles = curr_insp.get("allowed_roles", ["內掃檢查"])
            my_classes = curr_insp.get("assigned_classes", [])
            
            st.divider()
            
            col_d, col_r = st.columns(2)
            today_date = datetime.now(TW_TZ).date()
            chk_date = col_d.date_input("檢查日期", today_date)
            chk_role = col_r.radio("檢查項目", roles, horizontal=True) if len(roles)>1 else roles[0]
            col_r.info(f"項目: {chk_role}")
            
            wk_num = get_week_num(chk_date)
            
            # --- 垃圾檢查 (批量處理) ---
            if chk_role == "垃圾/回收檢查":
                st.subheader("🗑️ 垃圾回收檢查")
                trash_cat = st.radio("違規項目", ["一般垃圾", "紙類", "網袋", "其他回收"], horizontal=True)
                
                df_init = pd.DataFrame({"班級": all_classes_list if all_classes_list else ["無資料"], "無簽名": False, "無分類": False})
                edited_df = st.data_editor(df_init, height=400, use_container_width=True, hide_index=True)
                
                if st.button("送出垃圾檢查結果"):
                    cnt = 0
                    success_flags = []
                    # 批次同步處理
                    with st.status("🗑️ 正在儲存資料...", expanded=True) as status:
                        for _, row in edited_df.iterrows():
                            vios = []
                            if row["無簽名"]: vios.append("無簽名")
                            if row["無分類"]: vios.append("無分類")
                            
                            if vios:
                                status.write(f"正在寫入：{row['班級']}...")
                                entry = {
                                    "日期": chk_date, "週次": wk_num, "檢查人員": insp_name, 
                                    "班級": row["班級"], "評分項目": chk_role,
                                    "垃圾原始分": len(vios), "違規細項": trash_cat,
                                    "備註": f"{trash_cat}-{'、'.join(vios)}",
                                    "登錄時間": datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")
                                }
                                # 這裡簡化處理，因為是批次，我們直接寫入Sheet
                                try:
                                    ws = get_worksheet(SHEET_TABS["main"])
                                    if "紀錄ID" not in entry: entry["紀錄ID"] = f"{datetime.now(TW_TZ).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:4]}"
                                    row_vals = []
                                    for col in EXPECTED_COLUMNS: row_vals.append(str(entry.get(col, "")))
                                    ws.append_row(row_vals)
                                    cnt += 1
                                except: pass
                        st.cache_data.clear()
                        status.update(label=f"✅ 完成！共記錄 {cnt} 個違規班級", state="complete")
                    time.sleep(1.5)
                    st.rerun()
            
            # --- 一般評分 (內掃/外掃) ---
            else:
                st.subheader(f"🏫 評分對象選擇 ({chk_role})")
                
                target_cls = None
                
                if my_classes:
                    st.info("您有指定的負責班級：")
                    target_cls = st.radio("選擇班級", my_classes)
                else:
                    g_sel = st.radio("年級", GRADES, horizontal=True)
                    cls_opts = [c["name"] for c in STRUCTURED_CLASSES if c["grade"] == g_sel]
                    if cls_opts:
                        target_cls = st.radio("班級", cls_opts, horizontal=True)
                    else:
                        st.warning("無班級")
                
                if target_cls:
                    # 顯示是否重複
                    if "last_submit" not in st.session_state: st.session_state.last_submit = None
                    df_main = load_main_data()
                    is_dup = check_duplicate_record(df_main, chk_date, insp_name, chk_role, target_cls)
                    
                    if st.session_state.last_submit == target_cls:
                        st.success(f"✨ {target_cls} 剛剛已送出成功！")
                    elif is_dup:
                        st.warning(f"⚠️ {target_cls} 今日已評分過！")
                    
                    st.markdown(f"#### 👉 正在評分：<span style='color:orange'>{target_cls}</span>", unsafe_allow_html=True)
                    
                    with st.form(key=f"form_{target_cls}", clear_on_submit=True):
                        s_in, s_out, ph = 0, 0, 0
                        note = ""
                        
                        is_ok = st.radio("結果", ["❌ 違規扣分", "✨ 完美乾淨"], horizontal=True)
                        if is_ok == "❌ 違規扣分":
                            if chk_role == "內掃檢查":
                                s_in = st.number_input("內掃扣分", 1, 2, 1)
                            elif chk_role == "外掃檢查":
                                s_out = st.number_input("外掃扣分", 1, 2, 1)
                            note = st.text_input("違規說明 (必填)", placeholder="例如：黑板未擦、走廊有紙屑")
                            ph = st.number_input("手機違規人數", 0, 10, 0)
                        else:
                            note = "【優良】"
                        
                        is_fix = st.checkbox("🚩 開立修正單")
                        files = st.file_uploader("📸 拍照存證 (可多張)", accept_multiple_files=True)
                        
                        # 重要：改成同步送出
                        submit_btn = st.form_submit_button("🚀 確認送出")
                        
                        if submit_btn:
                            if is_ok == "❌ 違規扣分" and not note:
                                st.error("❌ 扣分時請務必填寫說明！")
                            else:
                                data = {
                                    "日期": chk_date, "週次": wk_num, "檢查人員": insp_name,
                                    "班級": target_cls, "評分項目": chk_role,
                                    "內掃原始分": s_in, "外掃原始分": s_out, "手機人數": ph,
                                    "備註": note, "修正": is_fix,
                                    "登錄時間": datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")
                                }
                                # 呼叫同步寫入
                                if save_entry_sync(data, files):
                                    st.session_state.last_submit = target_cls
                                    st.rerun()

# --- 模式2: 班級查詢 ---
elif app_mode == "班級負責人🥸":
    st.title("🔎 班級成績查詢")
    
    g_sel = st.radio("年級", GRADES, horizontal=True)
    cls_opts = [c["name"] for c in STRUCTURED_CLASSES if c["grade"] == g_sel]
    
    if cls_opts:
        my_cls = st.selectbox("請選擇班級", cls_opts)
        
        df = load_main_data()
        df_app = load_appeals()
        
        app_status = {}
        for _, r in df_app.iterrows():
            rid = str(r.get("對應紀錄ID", "")).strip()
            if rid: app_status[rid] = r.get("處理狀態", "待處理")
            
        if not df.empty:
            my_recs = df[df["班級"] == my_cls].sort_values("登錄時間", ascending=False)
            
            if my_recs.empty:
                st.info("🎉 目前沒有違規紀錄，保持下去！")
            else:
                st.write("---")
                for i, r in my_recs.iterrows():
                    total = r["內掃原始分"] + r["外掃原始分"] + r["垃圾原始分"] + r["晨間打掃原始分"]
                    ph = r["手機人數"]
                    rid = str(r["紀錄ID"]).strip()
                    stt = app_status.get(rid, "")
                    
                    icon = "📝"
                    if stt == "已核可": icon = "✅ [申訴成功撤銷]"
                    elif stt == "已駁回": icon = "🚫 [申訴駁回]"
                    elif stt == "待處理": icon = "⏳ [申訴審核中]"
                    elif str(r["修正"]) == "TRUE": icon = "🛠️ [已修正]"
                    
                    with st.expander(f"{icon} {r['日期']} - {r['評分項目']} (扣 {total} 分)"):
                        st.write(f"說明: {r['備註']}")
                        if ph > 0: st.write(f"📱 手機違規: {ph} 人")
                        
                        if r["照片路徑"] and r["照片路徑"] != "nan":
                            imgs = [x for x in str(r["照片路徑"]).split(";") if x.startswith("http")]
                            if imgs: st.image(imgs, width=200)
                        
                        d_obj = pd.to_datetime(r["日期"]).date()
                        is_expired = (date.today() - d_obj).days > 3
                        
                        if not stt and total > 0 and not is_expired:
                            with st.form(f"app_{rid}"):
                                reason = st.text_area("申訴理由")
                                proof = st.file_uploader("佐證照片 (必填)", type=["jpg","png"])
                                if st.form_submit_button("提交申訴"):
                                    if not reason or not proof:
                                        st.error("請填寫理由並上傳照片")
                                    else:
                                        ap_data = {
                                            "申訴日期": str(date.today()),
                                            "班級": my_cls,
                                            "違規日期": str(r["日期"]),
                                            "違規項目": r["評分項目"],
                                            "原始扣分": total,
                                            "申訴理由": reason,
                                            "處理狀態": "待處理",
                                            "對應紀錄ID": rid,
                                            "登錄時間": datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")
                                        }
                                        if save_appeal_sync(ap_data, proof):
                                            st.rerun()

# --- 模式3: 後台 ---
elif app_mode == "組長ㄉ窩💃":
    st.title("⚙️ 管理後台")
    pwd = st.text_input("管理密碼", type="password")
    
    if pwd == st.secrets["system_config"]["admin_password"]:
        monitor_tab, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "👀 進度監控", "📊 成績總表", "📝 扣分明細", "📧 寄送通知", 
                "📣 申訴審核", "⚙️ 系統設定", "📄 名單更新", "🧹 晨掃點名"
            ])
        
        with monitor_tab:
            st.subheader("🕵️ 今日評分進度監控")
            monitor_date = st.date_input("監控日期", datetime.now(TW_TZ).date())
            df = load_main_data()
            submitted_names = set()
            if not df.empty:
                df["日期Str"] = df["日期"].astype(str)
                today_records = df[df["日期Str"] == str(monitor_date)]
                submitted_names = set(today_records["檢查人員"].unique())
            
            regular_inspectors = []
            mobile_inspectors = []
            for p in INSPECTOR_LIST:
                is_mobile = len(p.get("assigned_classes", [])) == 0
                obj = {"name": p["label"], "done": p["label"] in submitted_names}
                if is_mobile: mobile_inspectors.append(obj)
                else: regular_inspectors.append(obj)
            
            done_reg = sum(1 for x in regular_inspectors if x["done"])
            st.write(f"**班級評分員完成率**: {done_reg}/{len(regular_inspectors)}")
            if len(regular_inspectors) > 0: st.progress(done_reg/len(regular_inspectors))
            
            c1, c2 = st.columns(2)
            with c1:
                st.write("🔴 未完成(班級)")
                for p in regular_inspectors:
                    if not p["done"]: st.error(p["name"])
            with c2:
                st.write("🟠 未完成(機動)")
                for p in mobile_inspectors:
                    if not p["done"]: st.warning(p["name"])

        with tab1:
            st.subheader("成績總表")
            df = load_main_data()
            all_classes_df = pd.DataFrame(all_classes_list, columns=["班級"])
            if not df.empty:
                valid_weeks = sorted(df[df["週次"]>0]["週次"].unique())
                selected_weeks = st.multiselect("選擇週次", valid_weeks, default=valid_weeks[-1:] if valid_weeks else [])
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
                    st.dataframe(final_report)
            else: st.info("無資料")
            
        with tab2: # 詳細明細
            st.subheader("📝 違規詳細流水帳")
            df = load_main_data()
            if not df.empty:
                st.dataframe(df)

        with tab3: # 寄送通知
            st.subheader("📧 每日違規通知")
            target_date = st.date_input("選擇日期", datetime.now(TW_TZ).date(), key="mail_d")
            if st.button("🔍 預覽名單"):
                df = load_main_data()
                df["d_obj"] = pd.to_datetime(df["日期"], errors='coerce').dt.date
                day_df = df[df["d_obj"] == target_date]
                if not day_df.empty:
                    agg = day_df.groupby("班級")[["內掃原始分","外掃原始分","垃圾原始分","手機人數"]].sum().reset_index()
                    agg["總扣分"] = agg.iloc[:,1:].sum(axis=1)
                    agg = agg[agg["總扣分"] > 0]
                    
                    preview = []
                    for _, r in agg.iterrows():
                        c = r["班級"]
                        t_info = TEACHER_MAILS.get(c, {})
                        preview.append({
                            "班級": c, "總扣分": r["總扣分"], 
                            "Email": t_info.get("email",""), "導師": t_info.get("name","")
                        })
                    st.session_state.mail_list = pd.DataFrame(preview)
                    st.dataframe(st.session_state.mail_list)
                else: st.info("無違規")
            
            if "mail_list" in st.session_state and not st.session_state.mail_list.empty:
                if st.button("確認寄出"):
                    q = []
                    for _, item in st.session_state.mail_list.iterrows():
                        if "@" in item["Email"]:
                            body = f"老師好，{item['班級']} 今日扣分合計：{item['總扣分']} 分。\n請協助督導，謝謝。"
                            q.append({"email": item["Email"], "subject": f"衛生組通知-{target_date}", "body": body})
                    
                    cnt, msg = send_bulk_emails(q)
                    st.success(f"已寄送 {cnt} 封")
        
        with tab4: # 申訴
            df_app = load_appeals()
            pending = df_app[df_app["處理狀態"] == "待處理"]
            if not pending.empty:
                for i, r in pending.iterrows():
                    with st.container(border=True):
                        st.write(f"**{r['班級']}**: {r['申訴理由']}")
                        if r["佐證照片"]: st.image(r["佐證照片"], width=200)
                        c1, c2 = st.columns(2)
                        if c1.button("核可", key=f"ok_{i}"):
                            update_appeal_status(r["申訴ID"], "已核可", r["對應紀錄ID"]); st.rerun()
                        if c2.button("駁回", key=f"no_{i}"):
                            update_appeal_status(r["申訴ID"], "已駁回", r["對應紀錄ID"]); st.rerun()
            else: st.info("無待審案件")

        with tab5: # 設定
            curr = load_settings().get("semester_start", "")
            st.write(f"目前開學日: {curr}")
            nd = st.date_input("新開學日")
            if st.button("更新設定"): save_setting("semester_start", str(nd)); st.success("OK")
            
            st.divider()
            if st.button("清除快取 (更新名單後使用)"): st.cache_data.clear(); st.success("已清除")

            st.divider()
            st.write("🗑️ 資料刪除")
            del_id = st.text_input("輸入紀錄ID刪除")
            if st.button("刪除單筆"): 
                if delete_rows_by_ids([del_id]): st.success("刪除成功")

        with tab6:
            st.markdown(f"[點此開啟 Google Sheet 編輯名單]({SHEET_URL})")

        with tab7: # 晨掃
            st.subheader("🧹 晨掃點名")
            m_date = st.date_input("日期", datetime.now(TW_TZ).date(), key="md_d")
            m_week = get_week_num(m_date)
            duty_list, status = get_daily_duty(m_date)
            if status == "success":
                with st.form("m_form"):
                    edited = st.data_editor(pd.DataFrame(duty_list), hide_index=True, use_container_width=True)
                    score = st.number_input("扣分", min_value=1, value=1)
                    if st.form_submit_button("送出"):
                        # 同步處理
                        with st.status("正在儲存...", expanded=True):
                            ws = get_worksheet(SHEET_TABS["main"])
                            cnt = 0
                            for _, r in edited[edited["已完成打掃"] == False].iterrows():
                                tid = clean_id(r["學號"])
                                cls = ROSTER_DICT.get(tid, f"查無({tid})")
                                entry = {
                                    "日期": m_date, "週次": m_week, "檢查人員": "衛生組", 
                                    "班級": cls, "評分項目": "晨間打掃", "晨間打掃原始分": score, 
                                    "備註": f"未到-學號:{tid}", "晨掃未到者": tid,
                                    "登錄時間": datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S"),
                                    "紀錄ID": f"{datetime.now(TW_TZ).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:4]}"
                                }
                                row_vals = []
                                for col in EXPECTED_COLUMNS: row_vals.append(str(entry.get(col, "")))
                                ws.append_row(row_vals)
                                cnt += 1
                        st.cache_data.clear()
                        st.success(f"已登記 {cnt} 人未到")
                        time.sleep(1)
                        st.rerun()
            else: st.warning(f"無輪值資料 ({status})")

    else:
        if pwd: st.error("密碼錯誤")
