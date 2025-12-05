import streamlit as st
from utils.auth import ensure_session
from modules.manage_files import (
    render_project_sidebar,
    render_upload_section,
    render_project_files
)

from modules.project_dashboard import render_project_dashboard

st.set_page_config(page_title="Pytorch Experiment Dashboard", layout="wide")

st.markdown("""
<style>
.help-container {
    position: fixed;   /* 화면 전체 기준 고정 */
    bottom: 20px;
    right: 20px;
    z-index: 999999;   /* 최상단 */
}

.help-icon {
    font-size: 30px;
    cursor: default;
}

.help-tooltip {
    visibility: hidden;
    opacity: 0;
    transition: opacity 0.2s ease;

    position: absolute;
    bottom: 15px;
    right: 40px;

    width: 300px;
    background: #333;
    color: white;
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 15px;
}
.help-container:hover .help-tooltip{
    visibility: visible;
    opacity: 1;
}
</style>

<div class="help-container">
    <div class="help-icon">❓</div>
    <div class="help-tooltip">
        <b>사이트 이용 방법</b><br><br>
        • 프로젝트 생성/선택<br>
        • 코드 파일 업로드 → 자동 분석<br>
        • 결과 JSON 업로드 → 자동 매칭/시각화<br>
        • 리더보드에서 성능 확인<br><br>
        <div>⭐ 만일 오류가 있다면 uaua1595@seoultech.ac.kr 로 연락주세요!</div>
    </div>
    
</div>
""", unsafe_allow_html=True)




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
# 브라우저 새로고침 시 selected_project 복원
# -----------------------------------------------------------
if "project" in st.query_params:
    try:
        st.session_state["selected_project"] = int(st.query_params["project"])
    except:
        pass



# -----------------------------------------------------------
# 사이드바
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
# 메인 화면
# -----------------------------------------------------------
def main():

    if not st.session_state.get("user_id"):
        st.markdown("""
        <div style="text-align: center; padding-top: 60px;">

        <h1 style="font-size: 42px; font-weight: 700; margin-bottom: 10px;">
            🎯 Pytorch Experiment Dashboard
        </h1>

        <h3 style="color: #666; font-weight: 400; margin-bottom: 30px;">
            A unified workspace for exploring, comparing, and visualizing your <b>PyTorch</b> experiments.
        </h3>

        <p style="font-size: 18px; line-height: 1.5; color: #444; max-width: 650px; margin: 0 auto 40px auto;">
            Upload your <b>PyTorch</b> training scripts, analyze model architectures,
            explore parsed components, visualize experiment results, and track performance across runs—all in one dashboard.
        </p>

        <!-- limitation 빨간 박스 -->
        <div style="
            margin: 0 auto 35px auto;
            max-width: 600px;
            padding: 14px 18px;
            border-radius: 8px;
            background: #ffecec;
            border-left: 5px solid #ff6b6b;
            color: #7a0000;
            font-size: 15px;
            line-height: 1.5;
            text-align: left;
        ">
            <b>⚠️ Limitation</b><br>
            <ul>
                <li>Only PyTorch model definitions contained in <b>a single file</b> are currently supported.</li>
                <li>Multi-file model architectures may not be fully recognized in this version.</li>
            </ul>
            I am actively working to enhance multi-file support in future updates.
        </div>

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
        To get started with <strong>Pytorch Experiment Dashboard</strong>, follow the steps below:
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

    # --------------------------------------------------------
    # Add JS Event Listener for postMessage
    # --------------------------------------------------------
    streamlit_message_listener = """
    <script>
    window.addEventListener("message", (event) => {
        if (event.data && event.data.type === "restore_project") {
            const pid = event.data.project;
            const url = new URL(window.location.href);
            url.searchParams.set("selected_project_msg", pid);
            window.location.href = url.toString();
        }
    });
    </script>
    """

    st.markdown(streamlit_message_listener, unsafe_allow_html=True)



if __name__ == "__main__":
    render_sidebar()
    main()
