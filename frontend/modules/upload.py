# pages/upload.py
import streamlit as st
import time
from utils.file_ops import upsert_code, upsert_result


def app():
    st.header("📤 Upload Files")

    user_id = st.session_state.get("user_id")
    if not user_id:
        st.info("Please log in first.")
        return

    col_code, col_result = st.columns(2)

    with col_code:
        st.markdown("### 🧠 Upload Python Code (.py)")
        code_files = st.file_uploader(
            " ", type=["py"], accept_multiple_files=True, key="code_uploader"
        )
        if code_files:
            for cf in code_files:
                msg_box = st.empty()
                msg_box.markdown(
                    f"<small style='color:gray;'>⏳ Uploading <b>{cf.name}</b>...</small>",
                    unsafe_allow_html=True,
                )
                upsert_code(user_id, cf)
                msg_box.markdown(
                    f"<small style='color:green;'>✅ Uploaded <b>{cf.name}</b></small>",
                    unsafe_allow_html=True,
                )
                time.sleep(0.5)
                msg_box.empty()

    with col_result:
        st.markdown("### 📊 Upload Result Files (.json)")
        result_files = st.file_uploader(
            " ", type=["json"], accept_multiple_files=True, key="result_uploader"
        )
        if result_files:
            for rf in result_files:
                msg_box = st.empty()
                msg_box.markdown(
                    f"<small style='color:gray;'>⏳ Uploading <b>{rf.name}</b>...</small>",
                    unsafe_allow_html=True,
                )
                upsert_result(user_id, rf)
                msg_box.markdown(
                    f"<small style='color:green;'>✅ Uploaded <b>{rf.name}</b></small>",
                    unsafe_allow_html=True,
                )
                time.sleep(0.5)
                msg_box.empty()
