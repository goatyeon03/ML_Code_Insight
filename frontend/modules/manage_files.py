# modules/manage_files.py
# 🔥 DB write는 모두 FastAPI로 이동한 안정 버전

import streamlit as st
import time
import requests

from utils.db import get_conn  # READ ONLY
from utils.file_ops import (
    upload_code_api,
    create_project_api,
    delete_project_api,
    delete_file_api,
    API_URL
)
import pandas as pd
from modules.manage_account import render_account_delete_modal
from modules.diff_utils import compare_code_text
from modules.version_utils import generate_new_version_name 



# ----------------------------------------------------------
# 1) 프로젝트 선택 / 생성 (Danger Zone 없음)
# ----------------------------------------------------------
def render_project_sidebar():
    user_id = st.session_state.get("user_id")
    if not user_id:
        return

    st.sidebar.markdown("### 📁 Projects")

    # --------------------------------------------
    # 🔥 0. 프로젝트 생성 모드 처리 (최우선)
    # --------------------------------------------
    if st.session_state.get("pending_create_project"):
        # 프로젝트 생성 UI만 렌더링
        new_name = st.sidebar.text_input("New Project Name", key="new_project_name")

        colA, colB = st.sidebar.columns([1, 1])

        # Create 버튼
        if colA.button("Create", key="btn_create_project_confirm"):
            if new_name.strip():
                res = create_project_api(user_id, new_name.strip())
                if "error" in res:
                    st.sidebar.error(f"Failed: {res['error']}")
                else:
                    new_pid = res["project_id"]
                    st.sidebar.success("Project created!")

                    # 프로젝트 선택
                    st.session_state["selected_project"] = new_pid
                    # URL query param 갱신
                    st.query_params["project"] = new_pid

                st.session_state["pending_create_project"] = False
                st.rerun()

        # Cancel 버튼
        if colB.button("Cancel", key="btn_create_project_cancel"):
            st.session_state["pending_create_project"] = False
            st.rerun()

        # 프로젝트 생성 모드일 때는 아래 코드 절대 실행되지 않도록 return
        return

    # --------------------------------------------
    # 1. 프로젝트 목록 로드
    # --------------------------------------------
    conn = get_conn()
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT id, project_name
        FROM projects
        WHERE user_id=?
        ORDER BY created_at DESC
    """, (user_id,)).fetchall()
    conn.close()

    project_list = {name: pid for (pid, name) in rows}
    names = list(project_list.keys())

    # 현재 선택된 프로젝트
    current_pid = st.session_state.get("selected_project")

    # Selectbox 옵션
    options = ["(Select a project)", "➕ Create New Project"] + names

    # 기본값
    if current_pid in project_list.values():
        current_name = [n for n, p in project_list.items() if p == current_pid][0]
        default_index = options.index(current_name)
    else:
        default_index = 0

    # --------------------------------------------
    # 2. Selectbox 표시
    # --------------------------------------------
    selected = st.sidebar.selectbox(
        "Select Project",
        options,
        index=default_index,
        key="project_select",
        label_visibility="collapsed",
    )

    # --------------------------------------------
    # 3. 새로운 프로젝트 생성 선택 시
    # --------------------------------------------
    if selected == "➕ Create New Project":
        st.session_state["pending_create_project"] = True
        st.rerun()
        return

    # --------------------------------------------
    # 4. 선택 없음
    # --------------------------------------------
    if selected == "(Select a project)":
        st.session_state["selected_project"] = None
        return None

    # --------------------------------------------
    # 5. 기존 프로젝트 선택
    # --------------------------------------------
    pid = project_list[selected]
    st.session_state["selected_project"] = pid

    # URL 쿼리 반영
    st.query_params["project"] = pid

    return pid


# ----------------------------------------------------------
# 2) 코드 업로드 (file_uploader key 중복 해결)
# ----------------------------------------------------------
def normalize(name: str):
    return name.strip().lower()


def render_upload_section(user_id):
    st.sidebar.markdown("### 📤 Upload Python File")

    project_id = st.session_state.get("selected_project")
    if not project_id:
        st.sidebar.warning("Please select a project first.")
        return

    base_key = f"upload_code_{project_id}"
    uploader_key = st.session_state.get("uploader_key", base_key)

    # === 1) 기존 파일 로드 ===
    conn = get_conn()
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT f.filename
        FROM files f
        JOIN project_files pf ON pf.file_id = f.id
        WHERE pf.project_id=? AND f.user_id=? AND f.filetype='code'
    """, (project_id, user_id)).fetchall()
    conn.close()

    existing_names = {r[0] for r in rows}
    existing_norm = {normalize(n) for n in existing_names}


    # === 2) FileUploader ===
    uploaded_files = st.sidebar.file_uploader(
        "Upload .py files",
        type=["py"],
        accept_multiple_files=True,
        key=uploader_key
    )


    if uploaded_files:

        # Streamlit uploader는 동일 파일 업로드 시 이벤트를 안 보내므로,
        # 업로드 시점에 즉시 key 재설정해서 캐시 초기화
        st.session_state["uploader_key"] = base_key + "_reset_" + str(time.time())

        # 여러 파일 중 첫 번째만 처리 (필요하면 loop로 확장 가능)
        f = uploaded_files[0]
        uploaded_name = f.name
        uploaded_norm = normalize(uploaded_name)


        # === CASE A: 중복 아님 → 즉시 업로드 ===
        if uploaded_norm not in existing_norm:
            _perform_upload(user_id, project_id, f, base_key)
            st.rerun()

        # === CASE B: 중복 → 모달 표시 준비 ===
        st.session_state["pending_overwrite"] = uploaded_name
        st.session_state["pending_file_obj"] = f
        st.session_state["existing_file_list"] = list(existing_names)
        st.rerun()


    # --------------------------------------------------
    # CASE C) 중복 모달 UI
    # --------------------------------------------------
    if st.session_state.get("pending_overwrite"):
        fname = st.session_state["pending_overwrite"]
        fobj = st.session_state["pending_file_obj"]
        file_list = st.session_state["existing_file_list"]

        box = st.sidebar.container()
        box.markdown("### ⚠️ Duplicate File")
        box.warning(f"`{fname}` already exists. Choose an action:")


        # === 1) OVERWRITE ===
        if box.button("Overwrite", key="overwrite_yes"):
            _perform_upload(user_id, project_id, fobj, base_key)
            _clear_pending()
            st.rerun()

        # === 2) COMPARE & AUTO VERSIONING ===
        if box.button("Compare & Save New Version", key="overwrite_compare"):
            old_text = requests.get(
                f"{API_URL}/get_file?type=code&filename={fname}"
            ).text
            new_text = fobj.getbuffer().tobytes().decode("utf-8")   # ✔ pointer-safe

            diff = compare_code_text(old_text, new_text)
            changed = diff["added"] or diff["removed"] or diff["modified"]

            if not changed:
                box.info("Files are identical. Upload cancelled.")
                _clear_pending()
                st.session_state["uploader_key"] = base_key + "_reset"
                st.rerun()
            else:
                new_name = generate_new_version_name(fname, file_list)

                # 새 이름 강제 업로드
                upload_code_api(user_id, project_id, fobj, override_name=new_name)
                box.success(f"Uploaded as `{new_name}`")

                _clear_pending()
                st.session_state["uploader_key"] = base_key + "_reset"
                st.rerun()

        # === 3) CANCEL ===
        if box.button("Cancel", key="overwrite_cancel"):
            _clear_pending()
            st.session_state["uploader_key"] = base_key + "_reset"
            st.rerun()


def _clear_pending():
    st.session_state["pending_overwrite"] = None
    st.session_state["pending_file_obj"] = None
    st.session_state["existing_file_list"] = None


from io import BytesIO

def _perform_upload(user_id, project_id, f, base_key):
    msg = st.sidebar.empty()
    msg.write(f"⏳ Uploading `{f.name}`...")

    # --- 파일 내용을 안전하게 추출 (포인터 소모 안 함) ---
    content = f.getbuffer().tobytes()

    # --- FastAPI 요청 ---
    files = {"file": (f.name, BytesIO(content), "text/plain")}
    data = {"user_id": user_id, "project_id": project_id}

    resp = requests.post(
        f"{API_URL}/upload_code",
        data=data,
        files=files,
        timeout=30,
    )

    if resp.status_code != 200:
        msg.error(f"❌ Upload failed: {resp.text}")
    else:
        msg.success(f"✅ Uploaded `{f.name}`")

    st.session_state["uploader_key"] = base_key + "_reset_" + str(time.time())



# ----------------------------------------------------------
# 3) 프로젝트 안의 파일 목록 (READ ONLY)
# ----------------------------------------------------------
def render_project_files(project_id, user_id):
    st.sidebar.markdown("#### Files")


    conn = get_conn()
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT DISTINCT f.id, f.filename
        FROM files f
        JOIN project_files pf ON pf.file_id = f.id
        WHERE pf.project_id=? AND f.user_id=? AND f.filetype='code'
        ORDER BY f.uploaded_at DESC
    """, (project_id, user_id)).fetchall()
    conn.close()

    if not rows:
        st.sidebar.caption("No files in this project yet.")
        return

    # --------------------------------------------
    # 🔥 파일 리스트를 체크박스 형태로 렌더링
    # --------------------------------------------
    # st.sidebar.markdown("*Select Files to Delete")

    if "selected_files" not in st.session_state:
        st.session_state.selected_files = set()

    selected_files = st.session_state.selected_files
    file_map = {}   # fid → fname

    for fid, fname in rows:
        file_map[fid] = fname

        checked = st.sidebar.checkbox(fname, key=f"chk_{fid}")

        # 체크 상태를 session_state에 반영
        if checked:
            selected_files.add(fid)
        else:
            selected_files.discard(fid)

    # 업데이트 반영
    st.session_state.selected_files = selected_files
    selected_files = set(st.session_state.selected_files)


    # --------------------------------------------
    # 🔥 삭제 버튼 렌더링
    # --------------------------------------------
    if selected_files:
        if st.sidebar.button("🗑 Delete Selected Files", use_container_width=True):
            st.sidebar.caption("*Select files to delete.")
            st.session_state.pending_delete_multiple = list(selected_files)
    else:
        st.sidebar.button("🗑 Delete Selected Files", disabled=True, use_container_width=True)
        st.sidebar.caption("*Select files to delete.")

    

    # --------------------------------------------
    # 🔥 삭제 모달
    # --------------------------------------------
    if st.session_state.get("pending_delete_multiple"):
        to_delete = st.session_state["pending_delete_multiple"]

        box = st.sidebar.container()
        box.markdown("### ⚠️ Delete Selected Files")
        box.error("Are you sure you want to delete the selected files?")

        colA, colB = box.columns(2, gap="small")

        if colA.button("Yes", key="confirm_multi_delete"):
            for fid in to_delete:
                delete_file_api(user_id, fid)

            st.sidebar.success("Selected files deleted!")

            # 상태 초기화
            st.session_state.pending_delete_multiple = None
            st.session_state.selected_files = set()
            st.rerun()

        if colB.button("Cancel", key="cancel_multi_delete"):
            st.session_state.pending_delete_multiple = None
            st.rerun()




# ----------------------------------------------------------
# 4) Danger Zone (사이드바 맨 아래에 표시)
#    - 프로젝트 삭제 버튼 + 계정 삭제 버튼
#    - 실제 삭제 모달도 여기서 호출
# ----------------------------------------------------------
def render_danger_zone(user_id: int):
    project_id = st.session_state.get("selected_project")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💣 Danger Zone")

    # 프로젝트 삭제 버튼
    if project_id:
        if st.sidebar.button("Delete This Project", key="btn_delete_project"):
            st.session_state["pending_project_delete"] = project_id

    # 계정 삭제 버튼
    if st.sidebar.button("Delete Account", key="btn_delete_account"):
        st.session_state["pending_account_delete"] = True

    # --- 삭제 모달 처리 ---
    # 프로젝트 삭제 모달
    pending_pid = st.session_state.get("pending_project_delete")
    if pending_pid is not None:
        _render_project_delete_modal(user_id, pending_pid)

    # 계정 삭제 모달
    if st.session_state.get("pending_account_delete"):
        render_account_delete_modal()



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
