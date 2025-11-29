# modules/model_blocks.py
import streamlit as st
import pandas as pd
import requests
from utils.db import get_conn

import graphviz as gv

API_URL = "http://localhost:8000"



# ------------------------------------------------------
# 1) Model Tree (Block Table)
# ------------------------------------------------------
def render_model_tree(model_name, models, depth=0):
    model = models.get(model_name)
    if not model:
        st.error(f"Model not found: {model_name}")
        return

    # st.markdown(f"### Class: `{model_name}`")

    blocks = model.get("blocks", [])

    # Block 테이블 출력
    for i, blk in enumerate(blocks):
        src = blk.get("source", "(unknown)")
        line_no = blk.get("line_no", "?")
        layers = blk.get("layers", [])

        with st.expander(f"Block {i+1}: `{src}` (line {line_no})", expanded=False):
            df = pd.DataFrame(layers)
            # 숨기고 싶은 열 제거
            columns_to_show = ["attribute", "layer_type", "args", "kwargs"]
            df = df[columns_to_show]
            st.dataframe(df, use_container_width=True)


    # Submodules
    children = model.get("children", [])
    if children:
        st.markdown(f"##### 🔻 Submodules of `{model_name}`")

    for child in children:
        with st.expander(f"Submodel: `{child}`", expanded=False):
            render_model_tree(child, models, depth+1)



# ------------------------------------------------------
# 3) Pipeline Flow (streamlit-flow-component)
# ------------------------------------------------------
def render_pipeline_graphviz(pipeline, models, area):
    dot = gv.Digraph("Pipeline", format="svg")
    dot.attr(rankdir="LR", splines="ortho", nodesep="1.0")
    # HTML label 쓸 거라 fontsize는 여기선 크게 의미 없음
    dot.attr("node", shape="box", style="rounded,filled", fillcolor="#FFF8E1")

    for cls in pipeline:
        model_info = models.get(cls, {})
        blocks = model_info.get("blocks", [])

        # 레이어 목록 문자열
        bullet_lines = []
        for blk in blocks:
            for layer in blk.get("layers", []):
                ltype = layer.get("layer_type", "")
                args = layer.get("args", "")
                bullet_lines.append(f"{ltype}({args})")   # ← 여기서 bullet 제거!


        if bullet_lines:
            # HTML label용 줄바꿈
            bullet_html = "<BR ALIGN='LEFT'/>".join(bullet_lines)
        else:
            bullet_html = ""

        layers_html = "".join(
            f"<TR><TD ALIGN='LEFT'><FONT POINT-SIZE='10'>• {line}</FONT></TD></TR>"
            for line in bullet_lines
        )

        label_html = f"""<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2">
        <TR>
            <TD ALIGN="CENTER">
            <B><FONT POINT-SIZE="12">{cls}</FONT></B>
            </TD>
        </TR>
        {layers_html}
        </TABLE>
        >"""


        dot.node(cls, label=label_html)

    for i in range(len(pipeline) - 1):
        dot.edge(pipeline[i], pipeline[i + 1])

    # 아래는 너가 이미 쓰고 있는 SVG→img 부분 그대로
    import base64
    svg_bytes = dot.pipe(format="svg")
    svg_base64 = base64.b64encode(svg_bytes).decode("utf-8")

    html = f"""
    <div style="display:flex; justify-content:center;">
        <img src="data:image/svg+xml;base64,{svg_base64}" style="max-width: 100%; height: auto;">
    </div>
    """

    area.markdown(html, unsafe_allow_html=True)



# ------------------------------------------------------
# 4) Streamlit App Entry
# ------------------------------------------------------
def app():

    # 분석 결과 저장용 SessionState
    if "model_parse_data" not in st.session_state:
        st.session_state["model_parse_data"] = None

    st.header("🧱 Model Blocks Explorer")

    # 로그인 체크
    user_id = st.session_state.get("user_id")
    if not user_id:
        st.info("Please log in first.")
        return

    # --------------------------------------------------
    # 파일 목록 불러오기
    # --------------------------------------------------
    conn = get_conn()
    df_codes = pd.read_sql(
        """
        SELECT filename
        FROM files
        WHERE user_id=? AND filetype='code'
        ORDER BY datetime(uploaded_at) DESC
    """,
        conn,
        params=(user_id,),
    )

    if df_codes.empty:
        st.info("No code files uploaded.")
        return

    filenames = df_codes["filename"].tolist()
    filename = st.selectbox("Choose a code file", filenames)

    # --------------------------------------------------
    # 모델 분석 실행 버튼
    # --------------------------------------------------
    if st.button("🔍 Analyze Model Structure"):
        try:
            resp = requests.get(
                f"{API_URL}/model_blocks",
                params={"filename": filename},
                timeout=30,
            )
            st.session_state["model_parse_data"] = resp.json()
        except Exception as e:
            st.error(f"API error: {e}")
            return

    # --------------------------------------------------
    # 버튼 바깥: 모델 정보 렌더링
    # --------------------------------------------------
    data = st.session_state["model_parse_data"]
    if data is None:
        st.info("Click the button to analyze model structure.")
        return

    if data.get("error"):
        st.warning(f"Parser message: {data['error']}")

    models = data.get("models", {})
    pipeline = data.get("pipeline", [])
    top = data.get("top_model")

    if not models:
        st.warning("No model structure found.")
        return


    # --------------------------------------------------
    # 아래: 전체 block tree
    # --------------------------------------------------
    model_names = [name for name in models.keys() if not name.startswith("_")]
    default_idx = model_names.index(top) if top in model_names else 0

    selected = st.selectbox(
        "Choose a model to inspect",
        options=model_names,
        index=default_idx,
    )

    render_model_tree(selected, models)
