import streamlit as st
import pandas as pd
import os
import time
import io
import traceback
import uuid
import random
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

def clean_id(val):
    try:
        if pd.isna(val) or val == "": return ""
        return str(int(float(val))).strip()
    except: return str(val).strip()

def execute_with_retry(func, max_retries=3, base_delay=1.0):
    """
    簡單的重試邏輯，用於網路波動
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(base_delay + random.uniform(0, 1))
                print(f"⚠️ API Retry ({attempt+1}): {e}")
            else:
                raise e

# ==========================================
# 2. Google API 連線 (直連模式)
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

# 注意：這裡不快取 Drive Service，避免 Session Context 問題
def get_drive_service():
    try:
        creds = get_credentials()
        if not creds: return None
        return build('drive', 'v3', credentials=creds, cache_discovery=False)
    except Exception as e:
        print(f"⚠️ Drive 連線失敗: {e}")
        return None

@st.cache_resource(ttl=3600)
def get_spreadsheet_object():
    client = get_gspread_client()
    if not client: return None
    try: return client.open_by_url(SHEET_URL)
    except Exception as e: 
        st.error(f"❌ 無法開啟試算表: {e}")
        return None

def get_worksheet(tab_name):
    sheet = get_spreadsheet_object()
    if not sheet: return None
    try:
        return sheet.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        # 如果找不到分頁，自動建立
        cols = 20 if tab_name != "appeals" else 15
        ws = sheet.add_worksheet(title=tab_name, rows=100, cols=cols)
        if tab_name == "appeals": ws.append_row(APPEAL_COLUMNS)
        return ws
    except Exception as e:
        print(f"❌ 讀取分頁 '{tab_name}' 失敗: {e}")
        return None

# ==========================================
# 3. 同步寫入邏輯 (無 Queue, 無 Thread)
# ==========================================

def upload_image_sync(file_obj, filename):
    """
    同步上傳圖片，失敗會丟出 Exception 或回傳 None
    """
    service = get_drive_service()
    if not service: return None
    
    folder_id = None
    if "system_config" in st.secrets and "drive_folder_id" in st.secrets["system_config"]:
        folder_id = st.secrets["system_config"]["drive_folder_id"]

    def _action():
        metadata = {'name': filename}
        if folder_id:
            metadata['parents'] = [folder_id]
        
        # 這裡不壓縮，直接上傳原始 Bytes，避免 PIL 造成 Segfault
        media = MediaIoBaseUpload(file_obj, mimetype='image/jpeg', resumable=True)
        file = service.files().create(body=metadata, media_body=media, fields='id,webViewLink').execute()
        return file.get('webViewLink') or f"https://drive.google.com/file/d/{file.get('id')}/view"

    return execute_with_retry(_action)

def append_row_sync(tab_name, row_data):
    """
    同步寫入 Google Sheet
    """
    ws = get_worksheet(tab_name)
    if not ws: raise Exception("無法取得工作表")
    
    # 檢查表頭
    if len(ws.get_all_values()) == 0:
        header = EXPECTED_COLUMNS if tab_name == SHEET_TABS["main"] else APPEAL_COLUMNS
        ws.append_row(header)

    def _action():
        ws.append_row(row_data)
        
    execute_with_retry(_action)

def save_entry_sync(new_entry, uploaded_files=None):
    """
    完全同步的存檔流程：
    1. 準備資料
    2. 迴圈上傳照片 (一張傳完才傳下一張)
    3. 寫入 Sheet
    4. 回傳成功
    """
    # 1. 準備資料
    if "日期" in new_entry and new_entry["日期"]:
        new_entry["日期"] = str(new_entry["日期"])
    
    if "紀錄ID" not in new_entry or not new_entry["紀錄ID"]:
        unique_suffix = uuid.uuid4().hex[:6]
        timestamp = datetime.now(TW_TZ).strftime("%Y%m%d%H%M%S")
        new_entry["紀錄ID"] = f"{timestamp}_{unique_suffix}"

    files_list = [f for f in uploaded_files if f] if uploaded_files else []
    drive_links = []

    # 2. 上傳照片 (同步執行，會卡住 UI，這在 Streamlit 是正常的)
    if files_list:
        if len(files_list) > 4:
            st.error("❌ 最多上傳 4 張")
            return False
            
        progress_text = "☁️ 正在上傳照片... (請勿關閉視窗)"
        my_bar = st.progress(0, text=progress_text)
        
        for i, up_file in enumerate(files_list):
            try:
                # 檔名處理
                safe_class = str(new_entry.get("班級", "unknown"))
                logical_fname = f"{new_entry.get('日期', '')}_{safe_class}_{i}.jpg"
                unique_prefix = f"{datetime.now(TW_TZ).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
                drive_filename = f"{unique_prefix}_{logical_fname}"
                
                # 讀取檔案
                up_file.seek(0)
                file_bytes = up_file.read()
                
                # 上傳
                link = upload_image_sync(io.BytesIO(file_bytes), drive_filename)
                
                if link:
                    drive_links.append(link)
                else:
                    st.error(f"❌ 第 {i+1} 張照片上傳失敗，請重試。")
                    return False
                
                # 更新進度條
                my_bar.progress((i + 1) / len(files_list), text=f"已上傳 {i+1}/{len(files_list)} 張...")
                
            except Exception as e:
                st.error(f"❌ 上傳錯誤: {e}")
                return False
        
        my_bar.empty()

    if drive_links:
        new_entry["照片路徑"] = ";".join(drive_links)

    # 3. 寫入 Sheet
    try:
        with st.spinner("📝 正在寫入紀錄..."):
            row = []
            for col in EXPECTED_COLUMNS:
                val = new_entry.get(col, "")
                if isinstance(val, bool): val = str(val).upper()
                row.append(val)
            
            append_row_sync(SHEET_TABS["main"], row)
            return True
            
    except Exception as e:
        st.error(f"❌ 資料寫入失敗: {e}")
        return False

def save_appeal_sync(entry, proof_file=None):
    # 申訴的同步存檔
    if "申訴日期" not in entry: entry["申訴日期"] = str(date.today())
    if "登錄時間" not in entry: entry["登錄時間"] = datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")
    if "申訴ID" not in entry: entry["申訴ID"] = uuid.uuid4().hex[:8]
    
    link = ""
    if proof_file:
        try:
            proof_file.seek(0)
            data = proof_file.read()
            fname = f"Appeal_{entry.get('班級')}_{uuid.uuid4().hex[:6]}.jpg"
            
            with st.spinner("☁️ 上傳佐證照片..."):
                link = upload_image_sync(io.BytesIO(data), fname)
                if not link:
                    st.error("照片上傳失敗")
                    return False
        except Exception as e:
            st.error(f"上傳錯誤: {e}")
            return False
            
    entry["佐證照片"] = link
    
    try:
        with st.spinner("📝 送出申訴..."):
            row = [str(entry.get(col, "")) for col in APPEAL_COLUMNS]
            append_row_sync(SHEET_TABS["appeals"], row)
            return True
    except Exception as e:
        st.error(f"寫入失敗: {e}")
        return False

# ==========================================
# 4. 資料讀取邏輯
# ==========================================

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
            
        # 轉型
        numeric_cols = ["內掃原始分", "外掃原始分", "垃圾原始分", "晨間打掃原始分", "手機人數"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
                
        return df[EXPECTED_COLUMNS]
    except:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

@st.cache_data(ttl=60)
def load_appeals():
    ws = get_worksheet(SHEET_TABS["appeals"])
    if not ws: return pd.DataFrame(columns=APPEAL_COLUMNS)
    try:
        data = ws.get_all_records()
        return pd.DataFrame(data)
    except: return pd.DataFrame(columns=APPEAL_COLUMNS)

# 其他輔助讀取函式 (Roster, Settings 等) 維持原樣，因篇幅關係省略，
# 但因為它們原本就是 cache_data 且唯讀，不會有 thread 問題。
# 在此補上必要的讀取函式：

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
        unique = df[class_col].dropna().unique().tolist()
        return sorted([str(c).strip() for c in unique if str(c).strip()]), [] # 簡化回傳
    except: return [], []

@st.cache_data(ttl=21600)
def load_inspector_list():
    ws = get_worksheet(SHEET_TABS["inspectors"])
    default = [{"label": "測試人員", "allowed_roles": ["內掃檢查"], "assigned_classes": [], "id_prefix": "測"}]
    if not ws: return default
    try:
        df = pd.DataFrame(ws.get_all_records())
        if df.empty: return default
        # 簡化邏輯：只回傳基本列表，避免複雜解析錯誤
        return [{"label": str(row.get("姓名", "人員")), "allowed_roles": ["內掃檢查", "外掃檢查"], "id_prefix": "X"} for _, row in df.iterrows()]
    except: return default

# ==========================================
# 5. 主程式
# ==========================================

# 簡單載入資料
ROSTER_DICT = load_roster_dict()
all_classes, _ = load_sorted_classes()
if not all_classes: all_classes = ["測試班級"]

# 側邊欄
st.sidebar.title("🏫 評分系統 (穩定版)")
app_mode = st.sidebar.radio("模式", ["評分輸入", "資料查詢"])

if app_mode == "評分輸入":
    st.title("📝 評分輸入")
    
    # 簡易登入檢查
    pwd = st.text_input("通行碼", type="password")
    if pwd == st.secrets["system_config"]["team_password"]:
        
        c1, c2 = st.columns(2)
        date_input = c1.date_input("日期", date.today())
        inspector = c2.text_input("檢查人員", "衛生組")
        
        cls = st.selectbox("班級", all_classes)
        role = st.radio("項目", ["內掃檢查", "外掃檢查", "垃圾檢查"])
        
        with st.form("score_form"):
            score = st.number_input("扣分", min_value=0, step=1)
            note = st.text_input("說明")
            files = st.file_uploader("照片", accept_multiple_files=True)
            
            if st.form_submit_button("送出評分"):
                entry = {
                    "日期": date_input,
                    "週次": 0, # 可之後補算
                    "班級": cls,
                    "評分項目": role,
                    "檢查人員": inspector,
                    "內掃原始分": score if role=="內掃檢查" else 0,
                    "外掃原始分": score if role=="外掃檢查" else 0,
                    "垃圾原始分": score if role=="垃圾檢查" else 0,
                    "備註": note,
                    "登錄時間": datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # 呼叫同步存檔
                if save_entry_sync(entry, files):
                    st.success("✅ 成功送出！")
                    time.sleep(1)
                    st.rerun()

elif app_mode == "資料查詢":
    st.title("📊 資料查詢")
    df = load_main_data()
    st.dataframe(df)

