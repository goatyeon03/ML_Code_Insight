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
        st.markdown(f"#### 🔻 Submodules of `{model_name}`")

    for child in children:
        with st.expander(f"Submodel: `{child}`", expanded=False):
            render_model_tree(child, models, depth+1)


# ------------------------------------------------------
# 2) 입력한 차원 파싱
# ------------------------------------------------------
def parse_dim_input(text: str) -> str:
    if not text:
        return ""

    cleaned = (
        text.replace("X", "x")
            .replace(",", " ")
            .replace("-", " ")
            .replace("|", " ")
            .replace("x", " ")
    )
    parts = [p for p in cleaned.split() if p.isdigit()]
    if not parts:
        return ""

    return "x".join(parts)


# ------------------------------------------------------
# 3) Pipeline Flow (streamlit-flow-component)
# ------------------------------------------------------
def render_pipeline_graphviz(pipeline, models, input_dim=None, output_dim=None):
    """
    Graphviz를 이용해 정적 다이어그램(가로 흐름)을 렌더링하는 함수.
    """
    dot = gv.Digraph("Pipeline", format="svg")
    dot.attr(rankdir="LR", splines="ortho", nodesep="1.0")

    # 공통 스타일
    dot.attr("node", shape="box", style="rounded,filled", fillcolor="#FFF8E1", fontsize="12")

    # -------------------------------
    # Input node
    # -------------------------------
    # if input_dim:
    #     dot.node("INPUT", f"Input\nshape={input_dim}", fillcolor="#E3F2FD")

    # -------------------------------
    # Each model block
    # -------------------------------
    for cls in pipeline:
        model_info = models.get(cls, {})
        blocks = model_info.get("blocks", [])

        bullet_lines = []
        for blk in blocks:
            for layer in blk.get("layers", []):
                ltype = layer.get("layer_type", "")
                args = layer.get("args", "")
                bullet_lines.append(f"• {ltype}({args})")

        label = f"{cls}\n" + "\n".join(bullet_lines)
        dot.node(cls, label)

    # -------------------------------
    # Output node
    # -------------------------------
    # if output_dim:
    #     dot.node("OUTPUT", f"Output\nshape={output_dim}", fillcolor="#E8F5E9")

    # -------------------------------
    # Arrows
    # -------------------------------
    # if input_dim:
    #     dot.edge("INPUT", pipeline[0])

    for i in range(len(pipeline)-1):
        dot.edge(pipeline[i], pipeline[i+1])

    # if output_dim:
    #     dot.edge(pipeline[-1], "OUTPUT")

    # -------------------------------
    # Render
    # -------------------------------
    # st.markdown("### Model Pipeline Diagram")
    st.graphviz_chart(dot)


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
    # 모델 선택
    # --------------------------------------------------
    # st.markdown("### 📐 Model Input/Output Shape")

    # raw_input_dim = st.text_input(
    #     "Input Dimension",
    #     placeholder="e.g. 128,1000",
    #     key="pipeline_input_dim"
    # )
    # raw_output_dim = st.text_input(
    #     "Output Dimension",
    #     placeholder="e.g. 1 or 128,32",
    #     key="pipeline_output_dim"
    # )

    # input_dim = parse_dim_input(raw_input_dim)
    # output_dim = parse_dim_input(raw_output_dim)

    # --------------------------------------------------
    # Pipeline Flow 렌더링
    # --------------------------------------------------
    if "pipeline" in data:
        render_pipeline_graphviz(
            pipeline=data["pipeline"],
            models=models,
            # input_dim=input_dim,
            # output_dim=output_dim
        )


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
