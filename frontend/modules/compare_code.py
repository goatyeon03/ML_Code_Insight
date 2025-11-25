# modules/compare_code.py

import streamlit as st
import requests
import re
import difflib

API_URL = "http://localhost:8000"


def remove_comments(code_text: str):
    """
    코드 내의 블록 주석, 라인 주석, 빈 줄 등을 제거하고
    diff 비교에 적합한 라인 리스트로 변환한다.
    """
    lines = []
    in_block = False
    block_delimiter = None

    for line in code_text.splitlines():
        stripped = line.strip()

        # 블록 주석 ''' """ 처리
        if not in_block and (stripped.startswith("'''") or stripped.startswith('"""')):
            block_delimiter = stripped[:3]
            in_block = True

            # ''' """ 같은 줄에서 바로 닫히는 경우
            if stripped.count(block_delimiter) >= 2:
                in_block = False
            continue

        if in_block:
            if block_delimiter and block_delimiter in stripped:
                in_block = False
            continue

        # 한 줄 주석
        if stripped.startswith("#") or stripped == "":
            continue

        # inline 주석 제거
        pure = re.sub(r"#.*", "", line).rstrip()
        lines.append(pure)

    return lines


def app():
    st.header("🧩 Code Comparison (Diff Viewer)")

    # -----------------------
    # 1) 파일 목록 불러오기
    # -----------------------
    try:
        resp = requests.get(f"{API_URL}/list_files?type=code", timeout=10)
        code_files = [
            f for f in resp.json().get("files", [])
            if f.endswith(".py")
        ]
    except Exception as e:
        st.error(f"파일 목록 불러오기 실패: {e}")
        return

    if not code_files:
        st.warning("업로드된 .py 파일이 없습니다.")
        return

    colA, colB = st.columns(2)
    with colA:
        file_a = st.selectbox("🅰️ 코드 A 선택", code_files, index=0)
    with colB:
        file_b = st.selectbox("🅱️ 코드 B 선택", code_files, index=min(1, len(code_files)-1))

    # -----------------------
    # 2) 비교 버튼
    # -----------------------
    if st.button("🔍 Compare", type="primary"):
        with st.spinner("비교 중..."):
            try:
                a_text = requests.get(
                    f"{API_URL}/get_file?type=code&filename={file_a}", timeout=10
                ).text
                b_text = requests.get(
                    f"{API_URL}/get_file?type=code&filename={file_b}", timeout=10
                ).text
            except Exception as e:
                st.error(f"파일 읽기 실패: {e}")
                return

            # 주석 제거
            a_lines = remove_comments(a_text)
            b_lines = remove_comments(b_text)

            # 라인 단위 diff
            diff = difflib.ndiff(a_lines, b_lines)

            left_html = []
            right_html = []

            for line in diff:
                typ = line[:2]
                text = line[2:]

                if typ == "- ":
                    left_html.append(f"<div style='background:#ffeef0;'>{text}</div>")
                    right_html.append(f"<div></div>")  # A만 있음
                elif typ == "+ ":
                    left_html.append(f"<div></div>")
                    right_html.append(f"<div style='background:#e6ffed;'>{text}</div>")
                else:
                    # unchanged
                    left_html.append(f"<div>{text}</div>")
                    right_html.append(f"<div>{text}</div>")

            # HTML 결합
            left_block = "\n".join(left_html)
            right_block = "\n".join(right_html)

            # -----------------------
            # 3) 렌더링
            # -----------------------
            st.markdown(f"### 📄 `{file_a}` ↔ `{file_b}`")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**🅰️ {file_a}**", unsafe_allow_html=True)
                st.components.v1.html(
                    f"""
                    <div style="background:#f6f8fa;padding:8px;
                               font-family:monospace;white-space:pre;
                               overflow-x:auto;height:750px;">
                        {left_block}
                    </div>
                    """,
                    height=750,
                    scrolling=True,
                )

            with col2:
                st.markdown(f"**🅱️ {file_b}**", unsafe_allow_html=True)
                st.components.v1.html(
                    f"""
                    <div style="background:#f6f8fa;padding:8px;
                               font-family:monospace;white-space:pre;
                               overflow-x:auto;height:750px;">
                        {right_block}
                    </div>
                    """,
                    height=750,
                    scrolling=True,
                )
