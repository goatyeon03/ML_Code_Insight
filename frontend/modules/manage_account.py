import streamlit as st

def render_account_delete_modal():
    if not st.session_state.get("pending_account_delete"):
        return

    box = st.sidebar.container()
    box.markdown("### ⚠️ Delete Account")
    box.error("This action will permanently delete your account.\nAll projects and files will be removed.")

    colA, colB = box.columns([1, 1])

    if colA.button("Yes", key="confirm_account_delete"):
        from utils.file_ops import delete_account_api
        user_id = st.session_state["user_id"]

        res = delete_account_api(user_id)

        if "error" in res:
            st.sidebar.error(f"Failed: {res['error']}")
        else:
            st.sidebar.success("Account deleted successfully!")

        # 세션 초기화 후 로그인 화면으로 이동
        from utils.auth import _clear_session
        _clear_session()
        st.session_state.clear()
        st.rerun()

    if colB.button("Cancel", key="cancel_account_delete"):
        st.session_state["pending_account_delete"] = None
        st.rerun()
