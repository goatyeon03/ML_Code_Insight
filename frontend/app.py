import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3, os, json, hashlib, requests, re, time
from datetime import datetime
from collections import defaultdict

# =========================================================
# CONFIG
# =========================================================
API_URL = "http://localhost:8000"
DB_PATH = "ml_insight.db"
UPLOAD_DIR = "uploads"
SESSION_FILE = "session_state.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="ML Code Insight", layout="wide")
st.title("ML Code Insight Dashboard")

# =========================================================
# DB INIT
# =========================================================
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    filename TEXT NOT NULL,
    filetype TEXT NOT NULL,
    summary_json TEXT,
    preview_json TEXT,
    uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
)
""")
cur.execute("""
CREATE UNIQUE INDEX IF NOT EXISTS ux_files_user_name_type
ON files(user_id, filename, filetype)
""")
conn.commit()

# =========================================================
# HELPERS
# =========================================================
def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

def hash_password(pw): return hashlib.sha256(pw.encode()).hexdigest()
def verify_password(pw, h): return hash_password(pw) == h

def save_session(user_id, username):
    with open(SESSION_FILE, "w") as f:
        json.dump({"user_id": user_id, "username": username}, f)

def load_session():
    if os.path.exists(SESSION_FILE):
        try:
            data = json.load(open(SESSION_FILE))
            return data.get("user_id"), data.get("username")
        except:
            return None, None
    return None, None

def clear_session():
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)

def coerce_summary(x):
    if isinstance(x, dict): return x
    return {"dataset": {}, "model": {"class_name": "Unknown"}, "training": {}, "misc": {}}

def get_training(s, k, d=""): s = coerce_summary(s); return s.get("training", {}).get(k, d)
def get_model_name(s): s = coerce_summary(s); return s.get("model", {}).get("class_name", "Unknown")

def detect_task_type(df):
    joined = " ".join(c.lower() for c in df.columns)
    if any(k in joined for k in ["acc","accuracy","f1","precision","recall"]): return "classification"
    if any(k in joined for k in ["mse","mae","r2","rmse","loss"]): return "regression"
    return "unknown"

def normalize_name(name):
    base = re.sub(r"\.[^.]+$", "", name.lower())
    base = re.sub(r"[_\-\.\s]+", " ", base).strip()
    drop = {"train","training","result","results","metric","metrics","log","logs"}
    tokens = [t for t in base.split() if t not in drop]
    return " ".join(tokens)

def tokens(name): return set(normalize_name(name).split())
def jaccard(a,b): return len(a & b)/len(a | b) if a and b else 0.0
def longest_common_prefix(a,b):
    n=min(len(a),len(b))
    for i in range(n):
        if a[i]!=b[i]: return i
    return n
def match_code_and_results(code_files,result_files):
    code_tok={c:tokens(c) for c in code_files}
    res_tok={r:tokens(r) for r in result_files}
    pairs=defaultdict(list)
    for r in result_files:
        best_code,best_score=None,-1
        for c in code_files:
            score_j=jaccard(code_tok[c],res_tok[r])
            lcp=longest_common_prefix(normalize_name(c),normalize_name(r))
            score=score_j+(lcp/20.0)
            if score>best_score:
                best_code,best_score=c,score
        if best_code and (best_score>=0.3 or lcp>=4):
            pairs[best_code].append(r)
    for k,v in pairs.items():
        pairs[k]=sorted(set(v))
    return pairs

def upsert_code(user_id, cf):
    files = {"file": (cf.name, cf.getvalue(), "text/x-python")}
    res = requests.post(f"{API_URL}/upload_code", files=files, timeout=30)
    summary = res.json().get("summary", {})
    cur.execute("""
        INSERT INTO files (user_id, filename, filetype, summary_json, uploaded_at)
        VALUES (?, ?, 'code', ?, ?)
        ON CONFLICT(user_id, filename, filetype)
        DO UPDATE SET summary_json=excluded.summary_json, uploaded_at=excluded.uploaded_at
    """, (user_id, cf.name, json.dumps(summary), datetime.now().isoformat()))
    conn.commit()
    return summary

def upsert_result(user_id, rf):
    files = {"file": (rf.name, rf.getvalue(), "application/json")}
    res = requests.post(f"{API_URL}/upload_result", files=files, timeout=30)
    preview = res.json().get("preview", {})
    cur.execute("""
        INSERT INTO files (user_id, filename, filetype, preview_json, uploaded_at)
        VALUES (?, ?, 'result', ?, ?)
        ON CONFLICT(user_id, filename, filetype)
        DO UPDATE SET preview_json=excluded.preview_json, uploaded_at=excluded.uploaded_at
    """, (user_id, rf.name, json.dumps(preview), datetime.now().isoformat()))
    conn.commit()
    return preview

# =========================================================
# LOGIN / SIGNUP
# =========================================================
st.sidebar.header("👤 Account")

if "user_id" not in st.session_state:
    uid, uname = load_session()
    if uid:
        st.session_state["user_id"] = uid
        st.session_state["username"] = uname

if st.session_state.get("user_id"):
    st.sidebar.markdown(f"**{st.session_state['username']}** logged in.")
    if st.sidebar.button("Logout"):
        clear_session()
        st.session_state.clear()
        safe_rerun()
else:
    tab_login, tab_register = st.sidebar.tabs(["Login", "Register"])
    with tab_login:
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            submit_login = st.form_submit_button("Login", use_container_width=True)
        if submit_login:
            cur.execute("SELECT id, password_hash FROM users WHERE username=?", (username,))
            row = cur.fetchone()
            if row and verify_password(password, row[1]):
                st.session_state["user_id"] = row[0]
                st.session_state["username"] = username
                save_session(row[0], username)
                st.sidebar.success(f"✅ Welcome, {username}!")
                safe_rerun()
            else:
                st.sidebar.error("❌ Invalid credentials")
    with tab_register:
        with st.form("register_form", clear_on_submit=True):
            reg_user = st.text_input("New Username", key="reg_user")
            reg_pass = st.text_input("New Password", type="password", key="reg_pass")
            submit_register = st.form_submit_button("Register", use_container_width=True)
        if submit_register:
            try:
                cur.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                            (reg_user, hash_password(reg_pass)))
                conn.commit()
                st.sidebar.success("🎉 Registration complete! Please log in.")
            except sqlite3.IntegrityError:
                st.sidebar.error("⚠️ Username already exists.")

# =========================================================
# MAIN APP
# =========================================================
if st.session_state.get("user_id"):
    user_id = st.session_state["user_id"]
    username = st.session_state["username"]
    st.markdown(f"### Hello, **{username}**!")

    # 업로드
    col_code, col_result = st.columns(2)
    with col_code:
        st.markdown("### 🧠 Upload Python Code (.py)")
        code_files = st.file_uploader(" ", type=["py"], accept_multiple_files=True, key="code_uploader")
        if code_files:
            for cf in code_files:
                msg_box = st.empty()
                msg_box.markdown(f"<small style='color:gray;'>⏳ Uploading <b>{cf.name}</b>...</small>", unsafe_allow_html=True)
                upsert_code(user_id, cf)
                msg_box.markdown(f"<small style='color:green;'>✅ Uploaded <b>{cf.name}</b></small>", unsafe_allow_html=True)
                time.sleep(1)
                msg_box.empty()
    with col_result:
        st.markdown("### 📊 Upload Result Files (.json)")
        result_files = st.file_uploader(" ", type=["json"], accept_multiple_files=True, key="result_uploader")
        if result_files:
            for rf in result_files:
                msg_box = st.empty()
                msg_box.markdown(f"<small style='color:gray;'>⏳ Uploading <b>{rf.name}</b>...</small>", unsafe_allow_html=True)
                upsert_result(user_id, rf)
                msg_box.markdown(f"<small style='color:green;'>✅ Uploaded <b>{rf.name}</b></small>", unsafe_allow_html=True)
                time.sleep(1)
                msg_box.empty()

    # =========================================================
    # FILE LIST
    # =========================================================
    st.markdown("---")
    st.markdown("### 📂 Uploaded Files")

    if "file_checklist" not in st.session_state:
        st.session_state.file_checklist = set()

    cur.execute("""
        SELECT id, filename, filetype, uploaded_at, summary_json, preview_json
        FROM files WHERE user_id=? ORDER BY datetime(uploaded_at) DESC
    """, (user_id,))
    file_rows = cur.fetchall()


    if file_rows:
        # =========================================================
        # SEARCH + FILTER HEADER (style tuned)
        # =========================================================
        st.markdown("""
        <style>
        div[data-testid="stTextInput"] > div:first-child {
            border-radius: 6px;
            height: 38px;
        }
        div[data-baseweb="select"] {
            min-height: 38px;
        }
        div[data-testid="stHorizontalBlock"] {
            max-width: 1100px;      /* 리스트 전체 폭 제한 */
            margin: 0 auto;         /* 중앙 정렬 느낌 */
        }
        </style>
        """, unsafe_allow_html=True)

        col_search, col_sort, col_filter = st.columns([5, 2, 3])

        with col_search:
            search_query = st.text_input("", placeholder="🔍 Search files...", label_visibility="collapsed")

        with col_sort:
            sort_order = st.selectbox("Sort order", ["Newest", "Oldest"], label_visibility="collapsed")

        with col_filter:
            view_tab = st.radio("", ["total", "code", "result"], horizontal=True, label_visibility="collapsed")

        # 필터링
        filtered = [r for r in file_rows if search_query.lower() in r[1].lower()]
        if view_tab == "code": filtered = [r for r in filtered if r[2] == "code"]
        elif view_tab == "result": filtered = [r for r in filtered if r[2] == "result"]
        if sort_order == "Oldest": filtered = filtered[::-1]

        # 선택/삭제를 하나의 form으로 묶는다
        with st.form("delete_form", clear_on_submit=False):
            # 체크박스들의 현재 상태를 모을 임시 컨테이너
            selected_pairs = []

            # 스타일(간격 최소화)
            st.markdown("""
            <style>
            .rowline { margin: 0.15rem 0 0.15rem 0; }
            hr { margin: 0.2rem 0 !important; opacity: 0.25; }
            </style>
            """, unsafe_allow_html=True)

            # 파일 행 렌더링 + 체크박스
            selected_ids = []
            for idx, (fid, name, ftype, uploaded, summary_json, preview_json) in enumerate(filtered):
                icon = "🧠" if ftype == "code" else "📊"
                # 날짜/시간 풀로 표기 (YYYY-MM-DD HH:MM:SS)
                ts = (uploaded or "").replace("T", " ")[:19]

                c1, c2, c3 = st.columns([6, 2, 1])
                with c1:
                    st.markdown(f"<div class='rowline'>{icon} <code>{name}</code></div>", unsafe_allow_html=True)
                with c2:
                    st.caption(ts if ts else "")
                with c3:
                    # 폼 안의 위젯은 제출 시점 값이 유지됨
                    checked = st.checkbox("", key=f"sel_{fid}")
                    if checked:
                        selected_ids.append(fid)

                st.markdown("<hr>", unsafe_allow_html=True)

            # 폼 제출 버튼 (이 시점에 selected_pairs 확정)
            delete_click = st.form_submit_button("Delete Selected Files", use_container_width=True)


        # 폼 밖에서 실제 삭제 실행 (DB + 파일 동기화)
        if delete_click:
            if not selected_ids:
                st.warning("선택된 파일이 없습니다.")
            else:
                cur.executemany("DELETE FROM files WHERE user_id=? AND id=?", [(user_id, fid) for fid in selected_ids])
                conn.commit()
                st.success("✅ 선택한 항목을 삭제했습니다.")
                time.sleep(0.8)
                safe_rerun()

    else:
        st.info("No files uploaded yet.")

    # =========================================================
    # 📊 Uploaded Codes Summary (표 형식)
    # =========================================================
    st.markdown("---")
    st.markdown("## 🧾 Uploaded Codes Summary")

    df_codes = pd.read_sql("""
        SELECT filename, summary_json, uploaded_at
        FROM files WHERE user_id=? AND filetype='code'
        ORDER BY datetime(uploaded_at) DESC
    """, conn, params=(user_id,))

    if not df_codes.empty:
        rows = []
        for _, row in df_codes.iterrows():
            s = json.loads(row["summary_json"]) if row["summary_json"] else {}
            rows.append({
                "Filename": row["filename"],
                "Model": get_model_name(s),
                "Optimizer": get_training(s, "optimizer", ""),
                "LR": get_training(s, "learning_rate", ""),
                "Batch": get_training(s, "batch_size", ""),
                "Epochs": get_training(s, "epochs", ""),
                "Loss": get_training(s, "loss", ""),
                "Scheduler": get_training(s, "scheduler", ""),
                "Device": get_training(s, "device", ""),
                # "Uploaded": row["uploaded_at"]
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No code files uploaded yet.")



    # =========================================================
    # VISUALIZATION (pair별 토글)
    # =========================================================
    st.markdown("---")
    st.markdown("## 📈 Matched Pair Visualization")

    df_codes = pd.read_sql("SELECT filename FROM files WHERE user_id=? AND filetype='code'", conn, params=(user_id,))
    df_results = pd.read_sql("SELECT filename, preview_json FROM files WHERE user_id=? AND filetype='result'", conn, params=(user_id,))
    code_files_db = df_codes["filename"].tolist()
    result_files_db = df_results["filename"].tolist()
    pairs = match_code_and_results(code_files_db, result_files_db)

    # ================== ⚠️ 경고문 추가 ==================
    matched_all = {r for lst in pairs.values() for r in lst}
    unmatched_codes = [c for c in code_files_db if c not in pairs.keys() or not pairs[c]]
    unmatched_results = [r for r in result_files_db if r not in matched_all]

    if unmatched_codes or unmatched_results:
        warn_msgs = []
        if unmatched_codes:
            warn_msgs.append("⚠️ Unmatched code files: " + ", ".join(unmatched_codes))
        if unmatched_results:
            warn_msgs.append("⚠️ Unmatched result files: " + ", ".join(unmatched_results))
        st.warning("\n\n".join(warn_msgs))
    else:
        st.success("✅ All code and result files matched successfully!")
    # ===================================================


    if pairs:
        for code_name, results in pairs.items():
            with st.expander(f"🧠 {code_name} ↔ {', '.join(results)}", expanded=False):
                for rname in results:
                    res_row = df_results[df_results["filename"] == rname]
                    if res_row.empty: continue
                    try:
                        preview_json = res_row.iloc[0]["preview_json"]
                        data = json.loads(preview_json)
                        df = pd.DataFrame(data if isinstance(data, list) else [data])
                        if "epoch" in df.columns:
                            metric_cols = [c for c in df.columns if c.lower() not in ["epoch","step","iteration"]]
                            train_cols = [c for c in metric_cols if c.startswith("train_")]
                            val_cols = [c for c in metric_cols if c.startswith("val_")]
                            if train_cols and val_cols:
                                for metric in sorted(set(c.split("_",1)[1] for c in train_cols)):
                                    t_col, v_col = f"train_{metric}", f"val_{metric}"
                                    if t_col in df.columns and v_col in df.columns:
                                        plot_df = df.melt(id_vars="epoch", value_vars=[t_col, v_col],
                                                          var_name="Type", value_name="Value")
                                        fig = px.line(plot_df, x="epoch", y="Value", color="Type", markers=True,
                                                      title=f"{metric.upper()} (Train vs Val)")
                                        st.plotly_chart(fig, use_container_width=True, key=f"{code_name}_{rname}_{metric}")
                            else:
                                melt_df = df.melt(id_vars="epoch", var_name="metric", value_name="value")
                                fig = px.line(melt_df, x="epoch", y="value", color="metric", markers=True)
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.dataframe(df, use_container_width=True)
                        
                    except Exception as e:
                        st.warning(f"⚠️ Could not visualize {rname}: {e}")
    # else:
    #     st.info("No matched pairs found.")
else:
    st.info("👈 Please log in or register to continue.")
    st.stop()
