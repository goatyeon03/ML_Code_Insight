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

    # =========================================================
    # 1) Model Structure
    # =========================================================
    st.markdown("### 🧱 Model Structure")

    if not code_rows:
        st.info("No code files in this project yet.")
    else:
        # id → filename
        id2name = {fid: fname for fid, fname, _ in code_rows}

        # 현재 선택된 파일
        selected_id = st.session_state.get("selected_code_file", code_rows[0][0])
        if selected_id not in id2name:
            selected_id = code_rows[0][0]
        st.session_state["selected_code_file"] = selected_id

        labels = [id2name[fid] for fid in id2name]
        ids = list(id2name.keys())
        default_index = ids.index(selected_id)

        def _update_selected():
            st.session_state["selected_file_changed"] = True
            label = st.session_state["selected_label"]
            for _id, _name in id2name.items():
                if _name == label:
                    st.session_state["selected_code_file"] = _id

        selected_label = st.selectbox(
            "Choose a code file to inspect",
            labels,
            index=default_index,
            key="selected_label",
            on_change=_update_selected,
        )

        # id 다시 찾기
        for _id, _name in id2name.items():
            if _name == selected_label:
                selected_id = _id

        # model_data 캐시 초기화
        if st.session_state.get("selected_file_changed"):
            st.session_state["model_data"] = None
            st.session_state["selected_file_changed"] = False

        # -----------------------------
        # FastAPI 모델 구조 요청
        # -----------------------------
        model_data = st.session_state.get("model_data")

        if model_data is None:
            try:
                resp = requests.get(
                    f"{API_URL}/model_blocks",
                    params={"filename": id2name[selected_id]},
                    timeout=20,
                )
                model_data = resp.json()
                st.session_state["model_data"] = model_data
            except Exception as e:
                st.error(f"Model parsing failed: {e}")
                model_data = None

        # -----------------------------
        # 모델 구조 렌더링
        # -----------------------------
        if model_data:
            if model_data.get("error"):
                st.warning(f"⚠ Parser message: {model_data['error']}")

            models = model_data.get("models", {})
            pipeline = model_data.get("pipeline", [])
            top = model_data.get("top_model")

            # Pipeline
            if pipeline:
                from modules.model_blocks import render_pipeline_graphviz
                render_pipeline_graphviz(pipeline, models)

            # Block Tree
            model_names = [m for m in models if not m.startswith("_")]
            if model_names:
                idx = model_names.index(top) if top in model_names else 0
                chosen = st.selectbox(
                    "Choose a model class",
                    model_names,
                    index=idx,
                )
                from modules.model_blocks import render_model_tree
                render_model_tree(chosen, models)
        else:
            st.info("No model structure available.")

    st.markdown("---")

    # =========================================================
    # 2) Code Summary
    # =========================================================
    st.markdown("### 🧠 Code Summary")

    summary_records = []
    for fid, fname, summary_json in code_rows:
        try:
            info = json.loads(summary_json) if summary_json else {}
        except:
            info = {}

        training = info.get("training", {})

        summary_records.append(
            {
                "Filename": fname,
                "Model Class": (
                    info.get("model", {}).get("class_name")
                    or info.get("model_class")
                    or info.get("model_name")
                    or "-"
                ),
                "Optimizer": training.get("optimizer", "-"),
                "Learning Rate": training.get("learning_rate", training.get("lr", "-")),
                "Batch Size": training.get("batch_size", "-"),
                "Epochs": training.get("epochs", "-"),
                "Loss": training.get("loss", "-"),
                "Scheduler": training.get("scheduler", "-"),
                "Device": training.get("device", "-"),
            }
        )

    if summary_records:
        df = pd.DataFrame(summary_records)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No parsed summaries available.")

    st.markdown("---")

    # =========================================================
    # 3) Result Upload (write는 FastAPI에서만)
    # =========================================================
    st.markdown("### 📤 Upload Result Files (.json)")
    st.caption("""
    - Only `.json` result files are supported.
    - The result file must share a prefix with the training script.
    - Matching happens automatically after upload.
    """)

    uploads = st.file_uploader(
        "Upload result JSON files",
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
                time.sleep(1.2)        # ⏳ 1.2초 정도 유지
                msg.empty() 

    st.markdown("---")

    # =========================================================
    # 4) Matched Result Visualization (DB read-only)
    # =========================================================
    st.markdown("### 📊 Matched Result Visualizations (per Code File)")

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

    if df_results.empty:
        st.info("No result files in this project.")
        st.stop()

    code_files = [fname for (_, fname, _) in code_rows]
    result_files = df_results["filename"].tolist()

    pairs = match_code_and_results(code_files, result_files)

    if not pairs:
        st.warning("No matched code-result pairs found.")
        st.stop()

    metric_records = []

    # 🔥 페어별 Accordion UI
    for code_name, result_list in pairs.items():

        with st.expander(f"🧠 {code_name}", expanded=False):

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

    # 1) test metric만 우선 선택
    test_rows = [rec for rec in metric_records if is_test_metric(rec["Metric"])]

    if test_rows:
        filtered_records = test_rows

    else:
        # 2) val metric만 선택 (train 제외)
        val_rows = [rec for rec in metric_records if is_val_metric(rec["Metric"])]
        filtered_records = val_rows

    # 🔥 필터링된 metric만 사용
    # filtered_records는 위에서 생성됨
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
            return "max"

        df_leader["SortDir"] = df_leader["Metric"].apply(metric_direction)
        df_leader["RankValue"] = df_leader.apply(
            lambda r: r["Final Value"] if r["SortDir"] == "max" else -r["Final Value"],
            axis=1
        )

        df_leader = df_leader.sort_values("RankValue", ascending=False)

        st.dataframe(
            df_leader[["Code File", "Result File", "Metric", "Final Value"]],
            use_container_width=True
        )
    else:
        st.info("No validation/test metrics extracted.")

