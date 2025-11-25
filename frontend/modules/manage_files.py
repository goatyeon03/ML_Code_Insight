# modules/manage_files.py
# 🔥 DB write는 모두 FastAPI로 이동한 안정 버전

import streamlit as st
from utils.db import get_conn  # READ ONLY
from utils.file_ops import (
    upload_code_api,
    create_project_api,
    delete_project_api,
    # delete_account_api  # 나중에 구현할 거면 여기서 import
)

# ----------------------------------------------------------
# 1) 프로젝트 선택 / 생성 (Danger Zone 없음)
# ----------------------------------------------------------
def render_project_sidebar():
    user_id = st.session_state.get("user_id")
    if not user_id:
        return

    st.sidebar.markdown("### 📁 Projects")

    conn = get_conn()
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT id, project_name
        FROM projects
        WHERE user_id=?
        ORDER BY created_at DESC
        """,
        (user_id,),
    ).fetchall()
    conn.close()

    project_list = {name: pid for (pid, name) in rows}
    names = list(project_list.keys())

    options = ["(Select a project)", "➕ Create New Project"] + names
    selected = st.sidebar.selectbox(
        "Select Project",
        options,
        key="project_select",
        label_visibility="collapsed",
    )

    # 새 프로젝트 생성
    if selected == "➕ Create New Project":
        new_name = st.sidebar.text_input(
            "New Project Name", key="new_project_name"
        )
        if st.sidebar.button("Create", key="btn_create_project"):
            if new_name.strip():
                res = create_project_api(user_id, new_name.strip())
                if "error" in res:
                    st.sidebar.error(f"Failed: {res['error']}")
                else:
                    st.sidebar.success("Project created!")
                    st.session_state["selected_project"] = res["project_id"]
                    st.rerun()
        return None

    # 아무 것도 선택 안 한 경우
    if selected == "(Select a project)":
        st.session_state["selected_project"] = None
        return None

    # 기존 프로젝트 선택
    pid = project_list[selected]
    st.session_state["selected_project"] = pid
    return pid


# ----------------------------------------------------------
# 2) 코드 업로드 (file_uploader key 중복 해결)
# ----------------------------------------------------------
def render_upload_section(user_id: int):
    st.sidebar.markdown("### 📤 Upload Python File")

    project_id = st.session_state.get("selected_project")
    if not project_id:
        st.sidebar.warning("Please select a project first.")
        return

    # 🔑 user_id + project_id 조합으로 항상 유일한 key 사용
    uploader_key = f"sidebar_code_upload_{user_id}_{project_id}"

    code_files = st.sidebar.file_uploader(
        "Drag and drop .py files here",
        type=["py"],
        accept_multiple_files=True,
        key=uploader_key,
    )

    if code_files:
        for f in code_files:
            msg = st.sidebar.empty()
            msg.write(f"⏳ Uploading `{f.name}`...")

            res = upload_code_api(user_id, project_id, f)

            if "error" in res:
                msg.error(f"Upload failed: {res['error']}")
            else:
                msg.success(f"Uploaded `{f.name}`")

        # 업로드 한 번 처리 후 다시 그리기
        st.rerun()


# ----------------------------------------------------------
# 3) 프로젝트 안의 파일 목록 (READ ONLY)
# ----------------------------------------------------------
def render_project_files(project_id, user_id):
    st.sidebar.markdown("#### Files")

    conn = get_conn()
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT DISTINCT f.id, f.filename
        FROM files f
        JOIN project_files pf ON pf.file_id = f.id
        WHERE pf.project_id=? AND f.user_id=? AND f.filetype='code'
        ORDER BY f.uploaded_at DESC
        """,
        (project_id, user_id),
    ).fetchall()
    conn.close()

    if not rows:
        st.sidebar.caption("No files in this project yet.")
        return

    seen = set()
    unique_files = []
    for fid, fname in rows:
        if fid in seen:
            continue
        seen.add(fid)
        unique_files.append((fid, fname))

    for fid, fname in unique_files:
        col1, col2 = st.sidebar.columns([8, 1])

        if col1.button(fname, key=f"open_{fid}"):
            st.session_state["selected_code_file"] = fid
            st.rerun()

        if col2.button("×", key=f"del_{fid}", help="Delete file"):
            # 파일 삭제는 나중에 별도 모달로 만들고 싶으면 여기서 state 세팅
            st.session_state["pending_delete_id"] = fid
            st.session_state["pending_delete_name"] = fname


# ----------------------------------------------------------
# 4) Danger Zone (사이드바 맨 아래에 표시)
#    - 프로젝트 삭제 버튼 + 계정 삭제 버튼
#    - 실제 삭제 모달도 여기서 호출
# ----------------------------------------------------------
def render_danger_zone(user_id: int):
    project_id = st.session_state.get("selected_project")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔥 Danger Zone")

    # 프로젝트 삭제 버튼
    if project_id:
        if st.sidebar.button("🗑 Delete This Project", key="btn_delete_project"):
            st.session_state["pending_project_delete"] = project_id

    # 계정 삭제 버튼 (API 없으면 UI만 둬도 됨)
    if st.sidebar.button("🚫 Delete Account", key="btn_delete_account"):
        st.sidebar.warning("Account deletion is not implemented yet.")

    # 삭제 모달 표시
    pending_pid = st.session_state.get("pending_project_delete")
    if pending_pid is not None:
        _render_project_delete_modal(user_id, pending_pid)


def _render_project_delete_modal(user_id: int, project_id: int):
    box = st.sidebar.container()
    box.markdown("### ⚠️ Delete Project")
    box.error("Are you sure you want to delete this project?")
    box.caption("All files used only in this project will also be deleted.")

    colA, colB = box.columns(2)

    # ✅ 실제 삭제
    if colA.button("Yes", key=f"confirm_project_del_{project_id}"):
        res = delete_project_api(user_id, project_id)

        if "error" in res:
            st.sidebar.error(f"Delete failed: {res['error']}")
        else:
            st.sidebar.success("Project deleted!")

        # 🔥 상태 초기화 → 모달 닫히고 프로젝트 선택 해제
        st.session_state["pending_project_delete"] = None
        st.session_state["selected_project"] = None
        st.session_state["selected_code_file"] = None
        st.rerun()

    # ❌ 취소
    if colB.button("Cancel", key=f"cancel_project_del_{project_id}"):
        st.session_state["pending_project_delete"] = None
        st.rerun()
