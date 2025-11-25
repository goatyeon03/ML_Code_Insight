import streamlit as st
import pandas as pd
import requests
import json

from utils.db import get_conn
from utils.summarize import coerce_summary
from utils.match_utils import match_code_and_results

API_URL = "http://localhost:8000"

def render_code_detail(filename):
    st.title(f"📄 {filename}")

    user_id = st.session_state["user_id"]
    conn = get_conn()

    # --------------------------
    # 1) Code Summary
    # --------------------------
    row = conn.execute("""
        SELECT summary_json 
        FROM files 
        WHERE user_id=? AND filename=? AND filetype='code'
    """, (user_id, filename)).fetchone()

    summary = coerce_summary(row[0])
    st.markdown("## 🧠 Code Summary")

    st.dataframe(pd.DataFrame({
        "Model Class": [summary["model"].get("class_name")],
        "Optimizer": [summary["training"].get("optimizer")],
        "Learning Rate": [summary["training"].get("learning_rate")],
        "Batch Size": [summary["training"].get("batch_size")],
        "Epochs": [summary["training"].get("epochs")],
        "Loss": [summary["training"].get("loss")],
    }), use_container_width=True)

    st.markdown("---")

    # --------------------------
    # 2) Model Structure 분석 자동 실행
    # --------------------------

    if "model_detail" not in st.session_state or \
       st.session_state.get("model_loaded_for") != filename:

        try:
            resp = requests.get(
                f"{API_URL}/model_blocks",
                params={"filename": filename},
                timeout=30,
            )
            st.session_state["model_detail"] = resp.json()
            st.session_state["model_loaded_for"] = filename

        except Exception as e:
            st.error(f"Model parsing failed: {e}")
            return

    data = st.session_state["model_detail"]

    st.markdown("## 🧱 Model Structure")

    if data.get("error"):
        st.warning(f"Parser message: {data['error']}")

    models = data.get("models", {})
    pipeline = data.get("pipeline", [])
    top = data.get("top_model")

    # --------------------------
    # 3) Pipeline Diagram
    # --------------------------

    # st.markdown("## 📊 Model Pipeline Diagram")

    if pipeline:
        from modules.model_blocks import render_pipeline_graphviz
        render_pipeline_graphviz(pipeline, models)

    # --------------------------
    # 4) Block Trees
    # --------------------------

    model_names = [name for name in models if not name.startswith("_")]
    default_idx = model_names.index(top) if top in model_names else 0

    selected = st.selectbox("Choose a model to inspect", model_names, index=default_idx)

    from modules.model_blocks import render_model_tree
    render_model_tree(selected, models)


    st.markdown("---")

    # --------------------------------------------------------
    # 3. 결과 파일 매칭 + 시각화
    # --------------------------------------------------------
    st.markdown("## 📊 Result Files")

    df_results = pd.read_sql("""
        SELECT filename, preview_json 
        FROM files 
        WHERE user_id=? AND filetype='result'
    """, conn, params=(user_id,))

    code_files = [filename]
    result_files = df_results["filename"].tolist()

    pairs = match_code_and_results(code_files, result_files)

    if filename not in pairs:
        st.info("아직 매칭된 결과 파일이 없습니다. JSON을 업로드해주세요.")
        return

    for result_name in pairs[filename]:
        st.markdown(f"### 📁 {result_name}")
        preview = json.loads(
            df_results[df_results["filename"] == result_name].iloc[0]["preview_json"]
        )
        df = pd.DataFrame(preview if isinstance(preview, list) else [preview])
        st.dataframe(df, use_container_width=True)
