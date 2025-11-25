# utils/auth.py
import streamlit as st
import hashlib, json, os
from utils.db import get_conn

SESSION_FILE = "session_state.json"


def _hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def _verify_password(pw: str, h: str) -> bool:
    return _hash_password(pw) == h


def _save_session(user_id: int, username: str):
    with open(SESSION_FILE, "w") as f:
        json.dump({"user_id": user_id, "username": username}, f)


def _load_session():
    if os.path.exists(SESSION_FILE):
        try:
            data = json.load(open(SESSION_FILE))
            return data.get("user_id"), data.get("username")
        except Exception:
            return None, None
    return None, None


def _clear_session():
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)


def ensure_session():
    # 1) 세션 파일 로드
    if "user_id" not in st.session_state:
        uid, uname = _load_session()
        if uid:
            st.session_state["user_id"] = uid
            st.session_state["username"] = uname

    # 2) 로그인 되어 있으면 그냥 return
    if st.session_state.get("user_id"):
        return

    # 3) 로그인 UI는 여기서 렌더링되지 않음
    #    대신 render_sidebar()에서 UI를 렌더링하게 됨
    st.session_state["user_id"] = None

def render_login_ui():
    conn = get_conn()
    cur = conn.cursor()

    st.sidebar.header("👤 Account")

    tab_login, tab_register = st.sidebar.tabs(["Login", "Register"])

    # ---- Login ----
    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

        if submit:
            cur.execute("SELECT id, password_hash FROM users WHERE username=?", (username,))
            row = cur.fetchone()

            if row and _verify_password(password, row[1]):
                st.session_state["user_id"] = row[0]
                st.session_state["username"] = username
                _save_session(row[0], username)
                st.sidebar.success("Logged in successfully!")
                st.rerun()
            else:
                st.sidebar.error("Invalid username or password.")

    # ---- Register ----
    with tab_register:
        with st.form("register_form"):
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password", type="password")
            submit_reg = st.form_submit_button("Register")

        if submit_reg:
            try:
                cur.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (new_user, _hash_password(new_pass))
                )
                conn.commit()
                st.sidebar.success("Registration successful!")
            except Exception:
                st.sidebar.error("Username already exists.")


