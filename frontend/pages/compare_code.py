import streamlit as st
import requests, re

API_URL = "http://localhost:8000"
st.set_page_config(page_title="🧩 2열 코드 비교", layout="wide")
st.title("🧩 코드 비교 (좌우 분리 뷰어)")

# -------------------------------
# 파일 목록 불러오기
# -------------------------------
try:
    resp = requests.get(f"{API_URL}/list_files?type=code", timeout=10)
    code_files = [f for f in resp.json().get("files", []) if f.endswith(".py")]
except Exception as e:
    st.error(f"파일 목록 불러오기 실패: {e}")
    st.stop()

if not code_files:
    st.warning("업로드된 .py 파일이 없습니다.")
    st.stop()

colA, colB = st.columns(2)
with colA:
    file_a = st.selectbox("🅰️ 코드 A", code_files, index=0)
with colB:
    file_b = st.selectbox("🅱️ 코드 B", code_files, index=min(1, len(code_files)-1))

# -------------------------------
# 주석 제거 함수
# -------------------------------
def remove_comments(code_text: str):
    lines = []
    in_block_comment = False
    block_delim = None

    for line in code_text.splitlines():
        stripped = line.strip()

        # 블록 주석 처리
        if not in_block_comment and (stripped.startswith("'''") or stripped.startswith('"""')):
            in_block_comment = True
            block_delim = stripped[:3]
            if stripped.count(block_delim) >= 2:
                in_block_comment = False
            continue
        if in_block_comment:
            if block_delim and block_delim in stripped:
                in_block_comment = False
            continue

        # 한 줄 주석
        if stripped.startswith("#") or stripped == "":
            continue

        # inline 주석 제거
        line = re.sub(r"#.*", "", line)
        lines.append(line.rstrip())
    return lines


# -------------------------------
# Compare 버튼
# -------------------------------
if st.button("🔍 Compare", type="primary"):
    with st.spinner("비교 중..."):
        a_text = requests.get(f"{API_URL}/get_file?type=code&filename={file_a}", timeout=10).text
        b_text = requests.get(f"{API_URL}/get_file?type=code&filename={file_b}", timeout=10).text

        a_lines = remove_comments(a_text)
        b_lines = remove_comments(b_text)

        # 줄 단위 diff 계산
        import difflib
        diff = difflib.ndiff(a_lines, b_lines)

        # 좌우별 색 입힌 HTML 변환
        left_html, right_html = "", ""
        for line in diff:
            if line.startswith("- "):
                left_html += f'<div style="background:#ffeef0;">{line[2:]}</div>'
            elif line.startswith("+ "):
                right_html += f'<div style="background:#e6ffed;">{line[2:]}</div>'
            elif line.startswith("? "):
                # 세부 변경 표시는 무시
                continue
            else:
                # 동일한 라인은 양쪽에 그대로
                left_html += f'<div>{line[2:]}</div>'
                right_html += f'<div>{line[2:]}</div>'

        # -------------------------------
        # 좌우 2열 코드 렌더링
        # -------------------------------
        st.markdown(f"#### 📄 {file_a} ↔ {file_b}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**🅰️ {file_a}**", unsafe_allow_html=True)
            st.components.v1.html(
                f"<div style='background:#f6f8fa;padding:8px;font-family:monospace;"
                f"white-space:pre;overflow-x:auto;height:750px;'>{left_html}</div>",
                height=750, scrolling=True,
            )
        with c2:
            st.markdown(f"**🅱️ {file_b}**", unsafe_allow_html=True)
            st.components.v1.html(
                f"<div style='background:#f6f8fa;padding:8px;font-family:monospace;"
                f"white-space:pre;overflow-x:auto;height:750px;'>{right_html}</div>",
                height=750, scrolling=True,
            )
