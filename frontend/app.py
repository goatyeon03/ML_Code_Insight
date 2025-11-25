# app.py  (🔥 최종 안정 버전)
import streamlit as st
from utils.auth import ensure_session
from modules.manage_files import (
    render_project_sidebar,
    render_upload_section,
    render_project_files
)

from modules.project_dashboard import render_project_dashboard

st.set_page_config(page_title="ML Code Insight", layout="wide")

# 🔥 절대로 Streamlit에서 init_db()를 호출하지 말 것!
# from utils.db import init_db
# init_db()   # ❌ 제거 — FastAPI 서버에서만 실행해야 함!

# 사이드바 X 버튼 스타일 (기존 유지)
st.sidebar.markdown("""
<style>
.delete-btn .stButton>button {
    width: 26px;
    height: 26px;
    padding: 0;
    border-radius: 6px;
    border: 1px solid #ddd;
    background: white;
    color: #d9534f;
    font-size: 19px;
    line-height: 19px;
    text-align: center;
}
.delete-btn .stButton>button:hover {
    background: #ffe7e7;
    border-color: #d9534f;
}
.file-row {
    display: flex;
    align-items: center;
    gap: 4px;
}
</style>
""", unsafe_allow_html=True)

ensure_session()


# -----------------------------------------------------------
# 🔥 사이드바
# -----------------------------------------------------------
def render_sidebar():
    if not st.session_state.get("user_id"):
        from utils.auth import render_login_ui
        render_login_ui()
        return

    st.sidebar.header("👤 Account")
    st.sidebar.markdown(f"**{st.session_state['username']}** logged in.")
    if st.sidebar.button("Logout"):
        from utils.auth import _clear_session
        _clear_session()
        st.session_state.clear()
        st.rerun()

    st.sidebar.markdown("---")

    # 1) 프로젝트 선택 / 생성
    project_id = render_project_sidebar()

    # 2) 선택된 프로젝트의 파일 목록
    if project_id:
        render_project_files(project_id, st.session_state["user_id"])

    # 3) 코드 업로드
    st.sidebar.markdown("---")
    render_upload_section(st.session_state["user_id"])

    # 4) Danger Zone (항상 맨 아래)
    from modules.manage_files import render_danger_zone
    render_danger_zone(st.session_state["user_id"])


# -----------------------------------------------------------
# 🔥 메인 화면
# -----------------------------------------------------------
def main():
    if not st.session_state.get("user_id"):
        st.markdown("""
        <div style="text-align: center; padding-top: 60px;">

        <h1 style="font-size: 42px; font-weight: 700; margin-bottom: 10px;">
            🧠 ML Code Insight
        </h1>

        <h3 style="color: #666; font-weight: 400; margin-bottom: 30px;">
            A unified workspace for exploring, comparing, and visualizing your Machine Learning experiments.
        </h3>

        <p style="font-size: 18px; line-height: 1.5; color: #444; max-width: 650px; margin: 0 auto 40px auto;">
            Upload your training scripts, analyze model structures, 
            view result visualizations, and track performance across experiments—all in one project dashboard.
        </p>

        <div style="font-size: 20px; margin-top: 40px; font-weight: 500;">
            👉 Please <span style="color:#4A90E2;">log in</span> or 
            <span style="color:#4A90E2;">register</span> using the left sidebar to get started.
        </div>

        </div>
        """, unsafe_allow_html=True)

        st.stop()

    project_id = st.session_state.get("selected_project")

    if not project_id:
        st.markdown(
        """
        <h1 style="font-size: 40px; font-weight: 750; color: #2C3E50; margin-bottom: 20px;">
        Welcome!
        </h1>

        <p style="font-size: 20px; color: #555; margin-bottom: 25px;">
        To get started with <strong>ML Code Insight</strong>, follow the steps below:
        </p>

        <ol style="font-size: 18px; color: #444; line-height: 1.7; margin-bottom: 30px;">
        <li>⬅️ Create or select a <strong>Project</strong> from the left sidebar.</li>
        <li>Upload your <strong>Python training scripts</strong> to that project.</li>
        <li>Upload your <strong>JSON result files</strong> that correspond to those scripts.</li>
        <li>View visualizations and compare final performance metrics.</li>
        </ol>
        """,
        unsafe_allow_html=True,
        )


        return

    render_project_dashboard(project_id, st.session_state["user_id"])


if __name__ == "__main__":
    render_sidebar()
    main()
