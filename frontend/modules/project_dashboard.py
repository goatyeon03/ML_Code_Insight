# modules/project_dashboard.py  (🔥 최종 안정 버전)
# 전체 구조 점검 + DB read-only + API write-only
# -------------------------------------------------------------

import json
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
import time

from utils.db import get_conn  # ✔ SELECT 용도로만 사용됨
from utils.file_ops import upload_result_api  # ✔ write는 FastAPI에서만 수행
from utils.match_utils import match_code_and_results


API_URL = "http://localhost:8000"

def group_metrics(metric_names):
    groups = {"acc": [], "f1": [], "loss": [], "other": []}

    for m in metric_names:
        ml = m.lower()
        if "acc" in ml:
            groups["acc"].append(m)
        elif "f1" in ml:
            groups["f1"].append(m)
        elif "loss" in ml or "mse" in ml or "rmse" in ml:
            groups["loss"].append(m)
        else:
            groups["other"].append(m)
    return groups


# -------------------------------------------------------------
# (1) 프로젝트 내 코드 파일 목록 로드 (DB SELECT만)
# -------------------------------------------------------------
def _load_code_files(project_id: int, user_id: int):
    conn = get_conn()
    cur = conn.cursor()

    rows = cur.execute("""
        SELECT DISTINCT f.id, f.filename, f.summary_json
        FROM files f
        JOIN project_files pf ON pf.file_id = f.id
        WHERE pf.project_id=? AND f.user_id=? AND f.filetype='code'
        ORDER BY f.uploaded_at DESC
    """, (project_id, user_id)).fetchall()

    conn.close()
    return rows


# -------------------------------------------------------------
# (2) 메인 렌더링 함수
# -------------------------------------------------------------
def render_project_dashboard(project_id: int, user_id: int):

    # === 프로젝트 이름 로드 ===
    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute(
        "SELECT project_name FROM projects WHERE id=? AND user_id=?",
        (project_id, user_id),
    ).fetchone()
    conn.close()

    project_name = row[0] if row else f"Project {project_id}"
    st.title(project_name)

    

    # === 코드 파일들 ===
    code_rows = _load_code_files(project_id, user_id)


    # --- 0) 파일이 없을 때 예시 화면 출력 ---
    if not code_rows:
        st.markdown("### 👋 Welcome to your new project!")
        st.markdown("Below are example files you can use on this site.")

        col1, col2 = st.columns(2, gap="large")

        # -------- 왼쪽: example code --------
        with col1:
            st.markdown("#### 📘 Example Training Code")
            
            example_py = """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, 10)

    model = SimpleMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 5
    """
            st.code(example_py, language="python")

        # -------- 오른쪽: example JSON --------
        with col2:
            st.markdown("#### 📊 Example Result JSON")

            example_json = [
                {"epoch": 1, "train_loss": 1.9, "val_loss": 1.6, "val_acc": 0.78},
                {"epoch": 2, "train_loss": 1.3, "val_loss": 1.1, "val_acc": 0.83},
                {"epoch": 3, "train_loss": 0.9, "val_loss": 0.8, "val_acc": 0.87}
            ]

            st.code(json.dumps(example_json, indent=4), language="json")


            # st.dataframe(example_json, use_container_width=True)

        st.stop()  # 예시 화면을 보여주고 나머지는 렌더링하지 않음

    

    # =========================================================
    # 1) Select Code File
    # =========================================================
    # st.markdown("### 🧠 Select Code File")

    if not code_rows:
        st.info("No code files in this project yet.")
        return

    # id → filename
    id2name = {fid: fname for fid, fname, _ in code_rows}

    # 현재 선택된 파일
    selected_id = st.session_state.get("selected_code_file", code_rows[0][0])
    if selected_id not in id2name:
        selected_id = code_rows[0][0]

    labels = [id2name[fid] for fid in id2name]

    st.markdown("### ℹ️ Training Information per Code File")

    # 현재 선택된 파일 ID
    selected_id = st.session_state.get("selected_code_file", code_rows[0][0])

    selected_label = st.selectbox(
        "Choose a code file",
        labels,
        key=f"code_select_{project_id}"
    )

    # id 역매칭
    for _id, _name in id2name.items():
        if _name == st.session_state[f"code_select_{project_id}"]:
            selected_id = _id
            break

    if st.session_state.get("selected_code_file") != selected_id:
        st.session_state["selected_code_file"] = selected_id
        st.session_state["model_data"] = None


    # 선택된 summary_json 불러오기
    raw_summary = [sj for (fid, _, sj) in code_rows if fid == selected_id][0]
    summary = json.loads(raw_summary) if raw_summary else {}
    training = summary.get("training", {})
    model_info = summary.get("model", {})

    # =========================================================
    # 2) Upper Layout: Left = Model Structure / Right = Params
    # =========================================================
    col_left, col_right = st.columns([1.2, 0.8])

    # -------------------------
    # (왼쪽) Model Structure
    # -------------------------
    with col_left:
        st.markdown("#### 🧱 Model Structure")

        # model_data 캐싱
        model_data = st.session_state.get("model_data")

        if (st.session_state.get("selected_file_changed", False)) or (model_data is None):
            try:
                resp = requests.get(
                    f"{API_URL}/model_blocks",
                    params={"filename": id2name[selected_id]},
                    timeout=20,
                )
                model_data = resp.json()
                st.session_state["model_data"] = model_data
                st.session_state["selected_file_changed"] = False
            except Exception as e:
                st.error(f"Model parsing failed: {e}")
                model_data = None

        # ---- 모델 구조 렌더링 ----
        if model_data:
            if model_data.get("error"):
                st.warning(f"⚠️ Parser: {model_data['error']}")

            models = model_data.get("models", {})
            pipeline = model_data.get("pipeline", [])
            top = model_data.get("top_model")

            # Pipeline
            if pipeline:
                from modules.model_blocks import render_pipeline_graphviz
                diagram_area = st.empty()  # 🔥 여기가 핵심
                render_pipeline_graphviz(pipeline, models, diagram_area)

            # Block Tree
            model_names = [m for m in models if not m.startswith("_")]
            if model_names:
                idx = model_names.index(top) if top in model_names else 0
                chosen = st.selectbox("Choose a model class", model_names, index=idx)
                from modules.model_blocks import render_model_tree
                render_model_tree(chosen, models)

    # -------------------------
    # (오른쪽) Model Parameters
    # -------------------------
    with col_right:
        st.markdown("#### ⚙️ Training Parameters")

        # 선택된 summary_json 기반
        def show_param(key, value):
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; padding:6px 12px; 
                            border: 1px solid #eee; border-radius:6px; margin-bottom:6px;">
                    <strong>{key}</strong>
                    <span>{value}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        # 기본 summary 정보 출력
        # show_param("Filename", id2name[selected_id])
        show_param("Model Class", model_info.get("class_name", "-"))
        show_param("Optimizer", training.get("optimizer", "-"))
        show_param("Learning Rate", training.get("learning_rate", training.get("lr", "-")))
        show_param("Batch Size", training.get("batch_size", "-"))
        show_param("Epochs", training.get("epochs", "-"))
        show_param("Loss Function", training.get("loss", "-"))
        show_param("Scheduler", training.get("scheduler", "-"))
        show_param("Device", training.get("device", "-"))

        # -------------------------
        # 🔥 총 파라미터 수 계산 (model_data 안 쓰고, filename + class_name 만 사용)
        # -------------------------
        total_params = None
        param_error = None

        model_class_name = model_info.get("class_name")

        if model_class_name:
            try:
                resp = requests.get(
                    f"{API_URL}/param_count",
                    params={"filename": id2name[selected_id]},
                    timeout=10,
                )
                data = resp.json()

                # 전체 API 에러 처리
                if data.get("error"):
                    param_error = data["error"]

                else:
                    # 모델 클래스 이름으로 접근
                    cls_info = data.get("results", {}).get(model_class_name)

                    if cls_info:
                        total_params = cls_info.get("total_params")
                        param_error = cls_info.get("error")
                    else:
                        param_error = f"No param info for class '{model_class_name}'."

            except Exception as e:
                param_error = str(e)
        else:
            param_error = "Model class name not found in summary."


        # -------------------------
        # UI 출력
        # -------------------------
        if total_params is not None:
            show_param("Total Parameters", f"{int(total_params):,}")
        else:
            show_param("Total Parameters", "N/A")
            if param_error:
                st.caption(f"Param count error: {param_error}")



    st.markdown("---")

    # =========================================================
    # 4) Matched Result Visualization (DB read-only)
    # =========================================================
    st.markdown("### 📊 Result Visualizations per Code File")

    # =========================================================
    # 3) Result Upload (write는 FastAPI에서만)
    # =========================================================
    
    st.markdown(
        """
    <div style="display:flex; justify-content:space-between; align-items:center; width:100%; margin: 10px 0 20px 0;">
        <!-- LEFT LIST -->
        <div style="flex: 1.2; margin-right: 40px;">
            <ul style="font-size: 1rem; color: #444; margin: 0; padding-left: 18px;">
                <li style="margin-bottom: 12px;">
                    <span style="background-color:#FFF59D; padding:4px 8px; border-radius:5px;">
                        Only <b>.json</b> result files are supported.
                    </span>
                </li>
                <li style="margin-bottom: 10px;">The result file must share a prefix with the training script.</li>
                <li>Matching happens automatically after upload.</li>
            </ul>
        </div>
        <!-- RIGHT BOX -->
        <div style="flex: 1; padding: 20px; background-color: #F7F9FC; border-radius: 10px; font-size: 1.1rem; font-weight: 600; color: #333; line-height: 1.4; white-space: nowrap;">
            <span style="font-size: 1.3rem; padding-right: 10px;">⬅️</span>
            Please review the following guidelines and Upload your result files.
        </div>
    </div>
        """,
        unsafe_allow_html=True
    )

    uploads = st.file_uploader(
        " ",
        type=["json"],
        accept_multiple_files=True,
        key=f"result_upload_{project_id}_{user_id}"
    )

    if uploads:
        for rf in uploads:
            msg = st.empty()
            msg.write(f"⏳ Uploading `{rf.name}` ...")

            res = upload_result_api(user_id, project_id, rf)
            if "error" in res:
                msg.error(f"Upload failed: {res['error']}")
            else:
                msg.success(f"Uploaded `{rf.name}`")
                time.sleep(1.0)
                msg.empty()

        # 🔥 DB insert가 끝나기 전에 rerun되면 SELECT에서 안 잡힘
        time.sleep(0.5)   # <<< 추가 (0.3~0.8 추천)

        st.rerun()


    # ---- 1. 코드/결과 파일 목록 가져오기 ----
    conn = get_conn()

    df_codes = pd.read_sql("""
        SELECT f.filename
        FROM files f
        JOIN project_files pf ON pf.file_id = f.id
        WHERE pf.project_id=? AND f.user_id=? AND f.filetype='code'
    """, conn, params=(project_id, user_id))

    df_results = pd.read_sql("""
        SELECT f.filename
        FROM files f
        JOIN project_files pf ON pf.file_id = f.id
        WHERE pf.project_id=? AND f.user_id=? AND f.filetype='result'
    """, conn, params=(project_id, user_id))


    code_files = df_codes["filename"].tolist()
    result_files = df_results["filename"].tolist()

    def normalize_prefix(name: str):
        base = name.rsplit(".", 1)[0]
        base = base.replace("_result", "")  # 결과 suffix 제거
        base = base.replace("_output", "")  # 필요하면 추가
        return base

    code_prefixes = {normalize_prefix(cf) for cf in code_files}
    result_prefixes = {normalize_prefix(rf) for rf in result_files}

    unmatched_codes = [
        cf for cf in code_files
        if normalize_prefix(cf) not in result_prefixes
    ]


    # ---- 2. 업로드 박스 위에 경고 메시지 표시 ----
    if unmatched_codes:
        st.warning(
            "⚠️ Some code files do not have matching result files:\n\n" +
            "\n".join([f"- `{f}`" for f in unmatched_codes])
        )

    # 🔥 conn 새로 열기 (절대 기존 conn 재사용 금지)
    conn2 = get_conn()
    df_results = pd.read_sql("""
        SELECT f.id, f.filename, f.preview_json
        FROM files f
        JOIN project_files pf ON pf.file_id = f.id
        WHERE pf.project_id=? AND f.user_id=? AND f.filetype='result'
        ORDER BY f.uploaded_at DESC
    """, conn2, params=(project_id, user_id))
    conn2.close()     # 🔥 바로 닫기

    # if df_results.empty:
    #     st.info("No result files in this project.")
    #     st.stop()

    code_files = [fname for (_, fname, _) in code_rows]
    result_files = df_results["filename"].tolist()

    pairs = match_code_and_results(code_files, result_files)

    # if not pairs:
    #     st.warning("No matched code-result pairs found.")
    #     st.stop()

    metric_records = []

    # 🔥 페어별 Accordion UI
    for code_name, result_list in pairs.items():

        with st.expander(f"{code_name}", expanded=False):

            for rname in result_list:
                st.markdown(f"#### 📄 {rname}")

                row = df_results[df_results["filename"] == rname].iloc[0]
                preview = json.loads(row["preview_json"])

                df = pd.DataFrame(preview) if isinstance(preview, list) else pd.DataFrame([preview])

                metric_cols = [c for c in df.columns 
                            if c.lower() not in ["epoch", "step", "iteration"]]

                groups = group_metrics(metric_cols)
                tabs = st.tabs(["ACC", "F1", "Loss/MSE", "Other"])

                # ---- ACC TAB ----
                with tabs[0]:
                    if groups["acc"]:
                        dfa = df[["epoch"] + groups["acc"]]
                        melt = dfa.melt("epoch", groups["acc"], "Metric", "Value")
                        fig = px.line(melt, x="epoch", y="Value", color="Metric", markers=True)
                        st.plotly_chart(fig, use_container_width=True)

                # ---- F1 TAB ----
                with tabs[1]:
                    if groups["f1"]:
                        dff = df[["epoch"] + groups["f1"]]
                        melt = dff.melt("epoch", groups["f1"], "Metric", "Value")
                        fig = px.line(melt, x="epoch", y="Value", color="Metric", markers=True)
                        st.plotly_chart(fig, use_container_width=True)

                # ---- LOSS TAB ----
                with tabs[2]:
                    if groups["loss"]:
                        dfl = df[["epoch"] + groups["loss"]]
                        melt = dfl.melt("epoch", groups["loss"], "Metric", "Value")
                        fig = px.line(melt, x="epoch", y="Value", color="Metric", markers=True)
                        st.plotly_chart(fig, use_container_width=True)

                # ---- OTHER TAB ----
                with tabs[3]:
                    if groups["other"]:
                        dfo = df[["epoch"] + groups["other"]]
                        melt = dfo.melt("epoch", groups["other"], "Metric", "Value")
                        fig = px.line(melt, x="epoch", y="Value", color="Metric", markers=True)
                        st.plotly_chart(fig, use_container_width=True)

                # 최종 값 기록
                final_row = df.iloc[-1]
                for m in metric_cols:
                    metric_records.append({
                        "Code File": code_name,
                        "Result File": rname,
                        "Metric": m,
                        "Final Value": final_row[m]
                    })

    # =========================================================
    # 5) Leaderboard Table
    # =========================================================

    def is_test_metric(m):
        return m.lower().startswith("test_")

    def is_val_metric(m):
        return m.lower().startswith("val_")

    st.markdown("---")
    st.markdown("### 🏁 Final Performance Leaderboard")
    st.caption("- Based on the final epoch metrics from the matched result files.  " \
    "\n- Test metrics are considered first; If no test metrics are available, validation metrics are used as the primary criteria.  " \
    "\n- The results are automatically sorted by performance.")

    # 1) test metric만 우선 선택
    test_rows = [rec for rec in metric_records if is_test_metric(rec["Metric"])]

    if test_rows:
        filtered_records = test_rows
    else:
        # 2) val metric만 선택 (train 제외)
        val_rows = [rec for rec in metric_records if is_val_metric(rec["Metric"])]
        filtered_records = val_rows

    # 🔥 필터링된 metric만 사용
    if filtered_records:
        df_leader = pd.DataFrame(filtered_records)

        # 중복 제거
        df_leader = df_leader.drop_duplicates(
            subset=["Code File", "Result File", "Metric"]
        )

        # 성능 기준 판별
        def metric_direction(m):
            ml = m.lower()
            if "acc" in ml or "f1" in ml:
                return "max"
            if "loss" in ml or "mse" in ml or "rmse" in ml or "mae" in ml:
                return "min"
            return "max"  # safe default

        df_leader["SortDir"] = df_leader["Metric"].apply(metric_direction)
        df_leader["RankValue"] = df_leader.apply(
            lambda r: r["Final Value"] if r["SortDir"] == "max" else -r["Final Value"],
            axis=1
        )

        df_leader = df_leader.sort_values("RankValue", ascending=False)

        # 🔥 Metric 종류별로 분리 (여기만 변경됨)
        metric_types = sorted(df_leader["Metric"].unique())

        for m in metric_types:
            st.markdown(f"#### 📌 Performance : **{m}**")

            sub = df_leader[df_leader["Metric"] == m]

            st.dataframe(
                sub[["Code File", "Result File", "Metric", "Final Value"]].reset_index(drop=True),
                use_container_width=True
            )

    else:
        st.info("No validation/test metrics extracted.")


