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

# --- 1. 網頁設定 ---
st.set_page_config(page_title="中壢家商，衛愛而生", layout="wide", page_icon="🧹")

# ==========================================
# 0. 基礎設定與時區
# ==========================================
TW_TZ = pytz.timezone('Asia/Taipei')

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
    "申訴日期", "班級", "違規日期", "違規項目", "原始扣分", "申訴理由", "佐證照片", "處理狀態", "登錄時間", "對應紀錄ID", "申訴ID"
]

# ==========================================
# 1. Google 連線整合 (同步模式)
# ==========================================

@st.cache_resource
def get_credentials():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    if "gcp_service_account" not in st.secrets:
        st.error("❌ 找不到 secrets 設定")
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
    sheet = get_spreadsheet_object()
    if not sheet: return None
    try:
        return sheet.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        cols = 20 if tab_name != "appeals" else 15
        ws = sheet.add_worksheet(title=tab_name, rows=100, cols=cols)
        if tab_name == "appeals": ws.append_row(APPEAL_COLUMNS)
        elif tab_name == "main": ws.append_row(EXPECTED_COLUMNS)
        return ws
    except Exception as e:
        st.error(f"❌ 讀取分頁 '{tab_name}' 失敗: {e}")
        return None

# --- 同步上傳圖片 (直接從記憶體上傳，不存暫存檔) ---
def upload_image_direct(file_obj, filename):
    try:
        service = get_drive_service()
        if not service: return "SERVICE_ERROR"
        
        folder_id = st.secrets["system_config"].get("drive_folder_id")
        if not folder_id: return "NO_FOLDER_ID"

        # 重置指針，確保從頭讀取
        file_obj.seek(0)
        
        file_metadata = {'name': filename, 'parents': [folder_id]}
        media = MediaIoBaseUpload(file_obj, mimetype='image/jpeg', resumable=True)
        
        file = service.files().create(
            body=file_metadata, media_body=media, fields='id', supportsAllDrives=True
        ).execute()
        
        # 開權限
        try:
            service.permissions().create(fileId=file.get('id'), body={'role': 'reader', 'type': 'anyone'}).execute()
        except: pass 
        
        return f"https://drive.google.com/thumbnail?id={file.get('id')}&sz=w1000"
    except Exception as e:
        print(f"Drive Upload Error: {e}")
        return "UPLOAD_FAILED"

def clean_id(val):
    try:
        if pd.isna(val) or val == "": return ""
        return str(int(float(val))).strip()
    except: return str(val).strip()

# ==========================================
# 2. 資料讀寫邏輯 (同步版)
# ==========================================

@st.cache_data(ttl=10) # 縮短快取時間，確保切換班級看到最新
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
            
        return df[EXPECTED_COLUMNS]
    except Exception as e:
        # 當欄位變動時，get_all_records 可能報錯，fallback
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

def save_entry_sync(new_entry, uploaded_files=None):
    """
    同步寫入：上傳照片 -> 寫 Sheet -> 回傳結果
    """
    status_msg = st.empty()
    status_msg.info("⏳ 正在上傳資料，請稍候...")

    # 1. 上傳照片
    drive_links = []
    if uploaded_files:
        for i, up_file in enumerate(uploaded_files):
            # 檔名: 日期_班級_序號.jpg
            safe_cls = str(new_entry.get('班級', 'unknown'))
            fname = f"{new_entry['日期']}_{safe_cls}_{i+1}.jpg"
            link = upload_image_direct(up_file, fname)
            if link not in ["SERVICE_ERROR", "NO_FOLDER_ID", "UPLOAD_FAILED"]:
                drive_links.append(link)
            else:
                new_entry["備註"] = str(new_entry.get("備註", "")) + " (部分照片上傳失敗)"

    new_entry["照片路徑"] = ";".join(drive_links)

    # 2. 準備 Row Data
    if "紀錄ID" not in new_entry:
        new_entry["紀錄ID"] = f"{datetime.now(TW_TZ).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:4]}"

    row_values = []
    for col in EXPECTED_COLUMNS:
        val = new_entry.get(col, "")
        if isinstance(val, bool): val = str(val).upper()
        row_values.append(str(val))

    # 3. 寫入 Sheet
    try:
        ws = get_worksheet(SHEET_TABS["main"])
        ws.append_row(row_values)
        st.cache_data.clear() # 清除快取，確保下次讀取是新的
        status_msg.success("✅ 資料已成功儲存！")
        time.sleep(1)
        status_msg.empty()
        return True
    except Exception as e:
        status_msg.error(f"❌ 寫入失敗: {e}")
        return False

def save_appeal_sync(entry, proof_file):
    status_msg = st.empty()
    status_msg.info("⏳ 正在提交申訴...")
    
    # 上傳佐證
    if proof_file:
        fname = f"Appeal_{entry.get('班級','')}_{datetime.now(TW_TZ).strftime('%H%M%S')}.jpg"
        link = upload_image_direct(proof_file, fname)
        entry["佐證照片"] = link
    else:
        entry["佐證照片"] = ""

    # 寫入
    row_values = [str(entry.get(col, "")) for col in APPEAL_COLUMNS]
    try:
        ws = get_worksheet(SHEET_TABS["appeals"])
        ws.append_row(row_values)
        st.cache_data.clear()
        status_msg.success("✅ 申訴已提交！")
        return True
    except Exception as e:
        status_msg.error(f"❌ 提交失敗: {e}")
        return False

# 其他輔助函式 (不涉及複雜運算的保留)
@st.cache_data(ttl=300)
def load_appeals():
    ws = get_worksheet(SHEET_TABS["appeals"])
    if not ws: return pd.DataFrame(columns=APPEAL_COLUMNS)
    try:
        df = pd.DataFrame(ws.get_all_records())
        # 確保欄位存在
        for col in APPEAL_COLUMNS:
            if col not in df.columns: df[col] = ""
        return df
    except: return pd.DataFrame(columns=APPEAL_COLUMNS)

def update_appeal_status(record_id, new_status, related_id):
    ws_app = get_worksheet(SHEET_TABS["appeals"])
    ws_main = get_worksheet(SHEET_TABS["main"])
    
    try:
        # 更新申訴表
        app_data = ws_app.get_all_records()
        target_row = None
        for i, row in enumerate(app_data):
            if str(row.get("對應紀錄ID")) == str(related_id) and str(row.get("處理狀態")) == "待處理":
                target_row = i + 2
                break
        
        if target_row:
            col_idx = APPEAL_COLUMNS.index("處理狀態") + 1
            ws_app.update_cell(target_row, col_idx, new_status)
            
            # 如果核可，去更新主表
            if new_status == "已核可" and related_id:
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

# 讀取名單相關 (Cache 時間拉長)
@st.cache_data(ttl=3600)
def load_roster_data():
    # 一次讀取所有需要的靜態名單
    res = {"classes": [], "structured": [], "roster_dict": {}, "inspectors": [], "teachers": {}}
    
    # 1. 班級名單
    ws_roster = get_worksheet(SHEET_TABS["roster"])
    if ws_roster:
        df = pd.DataFrame(ws_roster.get_all_records())
        if not df.empty:
            class_col = next((c for c in df.columns if "班級" in c), None)
            id_col = next((c for c in df.columns if "學號" in c), None)
            
            if class_col:
                unique = sorted(df[class_col].dropna().unique().tolist())
                res["classes"] = [str(c).strip() for c in unique if str(c).strip()]
                
                # 簡單排序邏輯
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

    # 2. 糾察名單
    ws_insp = get_worksheet(SHEET_TABS["inspectors"])
    if ws_insp:
        df = pd.DataFrame(ws_insp.get_all_records())
        if not df.empty:
            # 簡單處理
            id_col = next((c for c in df.columns if "學號" in c), None)
            role_col = next((c for c in df.columns if "負責" in c), None)
            scope_col = next((c for c in df.columns if "班級" in c), None)
            
            if id_col:
                for _, r in df.iterrows():
                    sid = clean_id(r[id_col])
                    roles = str(r[role_col]).strip() if role_col else ""
                    scopes = str(r[scope_col]).strip() if scope_col else ""
                    
                    allowed = ["內掃檢查"] # Default
                    if "組長" in roles: allowed = ["內掃檢查", "外掃檢查", "垃圾/回收檢查", "晨間打掃"]
                    elif "機動" in roles: allowed = ["內掃檢查", "外掃檢查", "垃圾/回收檢查"]
                    elif "外掃" in roles: allowed = ["外掃檢查"]
                    elif "垃圾" in roles: allowed = ["垃圾/回收檢查"]
                    elif "晨" in roles: allowed = ["晨間打掃"]
                    
                    s_list = [x.strip() for x in scopes.replace("、",";").split(";") if x.strip()]
                    
                    res["inspectors"].append({
                        "label": f"學號: {sid}",
                        "id_prefix": sid[0] if sid else "X",
                        "allowed_roles": allowed,
                        "assigned_classes": s_list
                    })

    # 3. 導師名單
    ws_teach = get_worksheet(SHEET_TABS["teachers"])
    if ws_teach:
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
    
    return res

@st.cache_data(ttl=3600)
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

def check_duplicate(df, date_val, inspector, role, cls):
    if df.empty: return False
    # 確保轉型
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

# ==========================================
# 3. 主程式介面
# ==========================================

# 載入資料
ALL_DATA = load_roster_data()
INSPECTOR_LIST = ALL_DATA["inspectors"]
TEACHER_MAILS = ALL_DATA["teachers"]
STRUCTURED_CLASSES = ALL_DATA["structured"]
GRADES = sorted(list(set([c["grade"] for c in STRUCTURED_CLASSES])))

st.sidebar.title("🏫 功能選單")
app_mode = st.sidebar.radio("請選擇模式", ["糾察底家👀", "班級負責人🥸", "組長ㄉ窩💃"])

# --- 模式1: 糾察評分 (重構重點) ---
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
            
            # --- 垃圾檢查 (特殊介面) ---
            if chk_role == "垃圾/回收檢查":
                st.subheader("🗑️ 垃圾回收檢查")
                trash_cat = st.radio("違規項目", ["一般垃圾", "紙類", "網袋", "其他回收"], horizontal=True)
                
                # 建立所有班級的 DataFrame 供編輯
                all_cls_names = ALL_DATA["classes"]
                if not all_cls_names: all_cls_names = ["無班級資料"]
                
                df_init = pd.DataFrame({"班級": all_cls_names, "無簽名": False, "無分類": False})
                edited_df = st.data_editor(df_init, height=400, use_container_width=True, hide_index=True)
                
                if st.button("送出垃圾檢查結果"):
                    cnt = 0
                    for _, row in edited_df.iterrows():
                        vios = []
                        if row["無簽名"]: vios.append("無簽名")
                        if row["無分類"]: vios.append("無分類")
                        
                        if vios:
                            entry = {
                                "日期": chk_date, "週次": wk_num, "檢查人員": insp_name, 
                                "班級": row["班級"], "評分項目": chk_role,
                                "垃圾原始分": len(vios), "違規細項": trash_cat,
                                "備註": f"{trash_cat}-{'、'.join(vios)}",
                                "登錄時間": datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")
                            }
                            save_entry_sync(entry)
                            cnt += 1
                    if cnt > 0: st.success(f"已記錄 {cnt} 個違規班級！")
                    else: st.info("無違規資料")
            
            # --- 一般評分 (內掃/外掃) ---
            else:
                st.subheader(f"🏫 評分對象選擇 ({chk_role})")
                
                target_cls = None
                
                # 選班級邏輯
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
                    # 狀態回饋區域
                    if "last_submit" not in st.session_state: st.session_state.last_submit = None
                    
                    # 讀取現有資料檢查重複
                    df_main = load_main_data()
                    is_dup = check_duplicate(df_main, chk_date, insp_name, chk_role, target_cls)
                    
                    if st.session_state.last_submit == target_cls:
                        st.success(f"✨ {target_cls} 剛剛已送出成功！")
                    elif is_dup:
                        st.warning(f"⚠️ {target_cls} 今日已評分過！(重複送出將新增第二筆)")
                    
                    st.markdown(f"#### 👉 正在評分：<span style='color:orange'>{target_cls}</span>", unsafe_allow_html=True)
                    
                    with st.form(key=f"form_{target_cls}", clear_on_submit=True):
                        # 依照 Role 顯示欄位
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
                        
                        if st.form_submit_button("🚀 確認送出 (請等待轉圈圈)"):
                            # 建立資料包
                            data = {
                                "日期": chk_date, "週次": wk_num, "檢查人員": insp_name,
                                "班級": target_cls, "評分項目": chk_role,
                                "內掃原始分": s_in, "外掃原始分": s_out, "手機人數": ph,
                                "備註": note, "修正": is_fix,
                                "登錄時間": datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            if is_ok == "❌ 違規扣分" and not note:
                                st.error("❌ 扣分時請務必填寫說明！")
                            else:
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
        
        # 建立申訴狀態 Map
        app_status = {}
        for _, r in df_app.iterrows():
            rid = str(r.get("對應紀錄ID", "")).strip()
            if rid: app_status[rid] = r.get("處理狀態", "待處理")
            
        if not df.empty:
            my_recs = df[df["班級"] == my_cls].sort_values("登錄時間", ascending=False)
            
            if my_recs.empty:
                st.info("🎉 目前沒有違規紀錄，保持下去！")
            else:
                for i, r in my_recs.iterrows():
                    total = r["內掃原始分"] + r["外掃原始分"] + r["垃圾原始分"] + r["晨間打掃原始分"]
                    ph = r["手機人數"]
                    rid = str(r["紀錄ID"]).strip()
                    stt = app_status.get(rid, "")
                    
                    icon = "📝"
                    if stt == "已核可": icon = "✅ [申訴成功撤銷]"
                    elif stt == "已駁回": icon = "🚫 [申訴駁回]"
                    elif stt == "待處理": icon = "⏳ [申訴審核中]"
                    
                    with st.expander(f"{icon} {r['日期']} - {r['評分項目']} (扣 {total} 分)"):
                        st.write(f"說明: {r['備註']}")
                        if ph > 0: st.write(f"📱 手機違規: {ph} 人")
                        
                        # 照片顯示
                        if r["照片路徑"] and r["照片路徑"] != "nan":
                            imgs = [x for x in str(r["照片路徑"]).split(";") if x.startswith("http")]
                            if imgs: st.image(imgs, width=200)
                        
                        # 申訴按鈕 (限制3天內)
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
    st.title("⚙️ 管理後台 (Lite)")
    pwd = st.text_input("管理密碼", type="password")
    
    if pwd == st.secrets["system_config"]["admin_password"]:
        t1, t2, t3, t4 = st.tabs(["📊 今日概況", "📧 寄信", "📣 申訴", "⚙️ 設定"])
        
        with t1:
            st.write("今日已評分班級概況")
            df = load_main_data()
            if not df.empty:
                df["d_str"] = df["日期"].astype(str)
                today_str = str(date.today())
                today_df = df[df["d_str"] == today_str]
                st.dataframe(today_df)
            else: st.info("無資料")
            
        with t2:
            st.write("一鍵寄送今日違規通知")
            target_d = st.date_input("選擇日期", date.today())
            if st.button("生成預覽"):
                df = load_main_data()
                # (簡化的邏輯)
                df["d_obj"] = pd.to_datetime(df["日期"], errors='coerce').dt.date
                day_df = df[df["d_obj"] == target_d]
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
                    st.session_state.mail_list = preview
                else: st.info("無違規")
            
            if "mail_list" in st.session_state and st.session_state.mail_list:
                st.dataframe(st.session_state.mail_list)
                if st.button("確認寄出"):
                    q = []
                    for item in st.session_state.mail_list:
                        if "@" in item["Email"]:
                            body = f"老師好，{item['班級']} 今日扣分合計：{item['總扣分']} 分。\n請協助督導，謝謝。"
                            q.append({"email": item["Email"], "subject": f"衛生組通知-{target_d}", "body": body})
                    
                    cnt, msg = send_bulk_emails(q)
                    st.success(f"已寄送 {cnt} 封")
        
        with t3:
            df_app = load_appeals()
            pending = df_app[df_app["處理狀態"] == "待處理"]
            if not pending.empty:
                for i, r in pending.iterrows():
                    with st.container(border=True):
                        c1, c2 = st.columns([3,1])
                        c1.write(f"**{r['班級']}** : {r['申訴理由']}")
                        if r["佐證照片"]: c2.image(r["佐證照片"])
                        
                        b1, b2 = st.columns(2)
                        if b1.button("核可", key=f"ok_{i}"):
                            update_appeal_status(r["申訴ID"], "已核可", r["對應紀錄ID"])
                            st.rerun()
                        if b2.button("駁回", key=f"no_{i}"):
                            update_appeal_status(r["申訴ID"], "已駁回", r["對應紀錄ID"])
                            st.rerun()
            else: st.info("無待審案件")

        with t4:
            if st.button("清除快取 (更新名單用)"):
                st.cache_data.clear()
                st.success("OK")
    else:
        if pwd: st.error("密碼錯誤")
