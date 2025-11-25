import streamlit as st
from utils.file_ops import upsert_code

def app():

    uploaded = st.session_state.get("uploaded_file")
    if not uploaded:
        st.error("업로드할 파일이 없습니다.")
        return

    # 백엔드로 저장
    fname = upsert_code(uploaded)
    
    # 세션에 추가
    files = st.session_state.get("files", [])
    if fname not in files:
        files.append(fname)
    st.session_state["files"] = files

    # 새 페이지로 이동
    st.session_state["current_page"] = f"file::{fname}"
    st.rerun()
