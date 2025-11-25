# pages/files.py
import streamlit as st
import time
from utils.db import get_conn


def app():
    st.header("📂 Uploaded Files")

    user_id = st.session_state.get("user_id")
    if not user_id:
        st.info("Please log in first.")
        return

    conn = get_conn()
    cur = conn.cursor()

    if "file_checklist" not in st.session_state:
        st.session_state.file_checklist = set()

    cur.execute("""
        SELECT id, filename, filetype, uploaded_at, summary_json, preview_json
        FROM files WHERE user_id=? ORDER BY datetime(uploaded_at) DESC
    """, (user_id,))
    file_rows = cur.fetchall()

    if not file_rows:
        st.info("No files uploaded yet.")
        return

    # 스타일
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
        max-width: 1100px;
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True)

    col_search, col_sort, col_filter = st.columns([5, 2, 3])

    with col_search:
        search_query = st.text_input("", placeholder="🔍 Search files...",
                                     label_visibility="collapsed")

    with col_sort:
        sort_order = st.selectbox("Sort order", ["Newest", "Oldest"],
                                  label_visibility="collapsed")

    with col_filter:
        view_tab = st.radio("", ["total", "code", "result"],
                            horizontal=True, label_visibility="collapsed")

    # 필터링
    filtered = [r for r in file_rows if search_query.lower() in r[1].lower()]
    if view_tab == "code":
        filtered = [r for r in filtered if r[2] == "code"]
    elif view_tab == "result":
        filtered = [r for r in filtered if r[2] == "result"]
    if sort_order == "Oldest":
        filtered = filtered[::-1]

    # 삭제 폼
    with st.form("delete_form", clear_on_submit=False):
        st.markdown("""
        <style>
        .rowline { margin: 0.15rem 0 0.15rem 0; }
        hr { margin: 0.2rem 0 !important; opacity: 0.25; }
        </style>
        """, unsafe_allow_html=True)

        selected_ids = []

        for fid, name, ftype, uploaded, *_ in filtered:
            icon = "🧠" if ftype == "code" else "📊"
            ts = (uploaded or "").replace("T", " ")[:19]

            c1, c2, c3 = st.columns([6, 2, 1])
            with c1:
                st.markdown(
                    f"<div class='rowline'>{icon} <code>{name}</code></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.caption(ts if ts else "")
            with c3:
                checked = st.checkbox("", key=f"sel_{fid}")
                if checked:
                    selected_ids.append(fid)

            st.markdown("<hr>", unsafe_allow_html=True)

        delete_click = st.form_submit_button("Delete Selected Files",
                                             use_container_width=True)

    if delete_click:
        if not selected_ids:
            st.warning("선택된 파일이 없습니다.")
        else:
            cur.executemany(
                "DELETE FROM files WHERE user_id=? AND id=?",
                [(user_id, fid) for fid in selected_ids],
            )
            conn.commit()
            st.success("✅ 선택한 항목을 삭제했습니다.")
            time.sleep(0.5)
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()
