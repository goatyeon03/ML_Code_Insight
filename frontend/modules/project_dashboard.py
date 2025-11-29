# modules/project_dashboard.py  (수정 버전)

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

        st.stop()  # 예시 화면을 보여주고 나머지는 렌더링하지 않음

    # =========================================================
    # 1) Select Code File
    # =========================================================
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


    st.markdown("### 🔍 DEBUG: Training Summary Data")
    st.json(training)



    # =========================================================
    # 2) Upper Layout: Left = Model Structure / Right = Params
    # =========================================================
    col_left, col_right = st.columns([1.2, 0.8])

    # -------------------------
    # (왼쪽) Model Structure
    # -------------------------
    with col_left:
        st.markdown("#### 🧱 Model Structure")

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

        if model_data:
            if model_data.get("error"):
                st.warning(f"⚠️ Parser: {model_data['error']}")

            models = model_data.get("models", {})
            pipeline = model_data.get("pipeline", [])
            top = model_data.get("top_model")

            # Pipeline
            if pipeline:
                from modules.model_blocks import render_pipeline_graphviz
                diagram_area = st.empty()
                render_pipeline_graphviz(pipeline, models, diagram_area)

            # Block Tree
            model_names = [m for m in models if not m.startswith("_")]
            if model_names:
                idx = model_names.index(top) if top in model_names else 0
                chosen = st.selectbox("Choose a model class", model_names, index=idx)
                from modules.model_blocks import render_model_tree
                render_model_tree(chosen, models)

    # -------------------------
    # (오른쪽) Model + Training Parameters (Stage-aware)
    # -------------------------
    with col_right:
        st.markdown("#### ⚙️ Training Parameters")

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

        # -------------------------
        # 🔥 Pretrained 플래그
        # -------------------------
        pretrained_raw = training.get("pretrained")

        # 표시용 + 내부 bool 플래그 둘 다 만들기
        if isinstance(pretrained_raw, bool):
            is_pretrained = pretrained_raw
        elif isinstance(pretrained_raw, str):
            pl = pretrained_raw.strip().lower()
            if pl in ("yes", "y", "true", "1"):
                is_pretrained = True
            elif pl in ("no", "n", "false", "0"):
                is_pretrained = False
            else:
                is_pretrained = False   # 애매하면 False 쪽으로
        else:
            is_pretrained = False

        pretrained_str = "Yes" if is_pretrained else "No"
        show_param("Pretrained", pretrained_str)

        # -------------------------
        # 🔧 공통 포맷터: pretrain / finetune 값 합쳐서 보여주기
        # -------------------------
        def format_stage_value(
            pre_val,
            ft_val,
            single_val=None,
            add_labels=True,
        ):
            """
            - pre_val, ft_val: pretrain/finetune용 값 (None 또는 값)
            - single_val: pretrain 안 쓰는 경우 fallback 값
            """
            if not is_pretrained:
                return single_val if (single_val is not None and single_val != "") else "-"

            parts = []

            if pre_val is not None and pre_val != "":
                parts.append(f"{pre_val} (pretrain)" if add_labels else str(pre_val))
            if ft_val is not None and ft_val != "":
                parts.append(f"{ft_val} (finetune)" if add_labels else str(ft_val))

            if not parts:
                return "-"

            # 둘 다 있으면 "A (pretrain) → B (finetune)" 형식
            return " → ".join(parts)

        # ========== EPOCHS ==========
        pretrain_epochs = training.get("pretrain_epochs")
        finetune_epochs = training.get("finetune_epochs")
        epochs = training.get("epochs")

        show_param(
            "Epochs",
            format_stage_value(pretrain_epochs, finetune_epochs, single_val=epochs)
        )

        # ========== BATCH SIZE ==========
        pretrain_bs = training.get("pretrain_batch_size")
        finetune_bs = training.get("finetune_batch_size")
        batch_size = training.get("batch_size")

        show_param(
            "Batch Size",
            format_stage_value(pretrain_bs, finetune_bs, single_val=batch_size)
        )

        # ========== LEARNING RATE ==========
        pretrain_lr = training.get("pretrain_learning_rate")
        finetune_lr = training.get("finetune_learning_rate")
        lr = training.get("learning_rate") or training.get("lr")

        show_param(
            "Learning Rate",
            format_stage_value(pretrain_lr, finetune_lr, single_val=lr)
        )

        # ========== LOSS FUNCTION ==========
        pretrain_loss = training.get("pretrain_loss")
        finetune_loss = training.get("finetune_loss")
        loss = training.get("loss")

        show_param(
            "Loss Function",
            format_stage_value(pretrain_loss, finetune_loss, single_val=loss)
        )

        # ========== OPTIMIZER / SCHEDULER / DEVICE ==========
        optimizer = training.get("optimizer")
        show_param("Optimizer", optimizer if optimizer else "-")

        scheduler = training.get("scheduler")
        show_param("Scheduler", scheduler if scheduler else "-")

        device = training.get("device")
        show_param("Device", device if device else "-")



        # -------------------------
        # PARAM COUNT (unchanged)
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

                if data.get("error"):
                    param_error = data["error"]
                else:
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
                <li>ex) train_cnn.py ↔ train_cnn_result.json </li>
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

    # 🔑 업로더 & 업로드 중복 처리용 키
    upload_widget_key = f"result_upload_{project_id}_{user_id}"
    last_files_key = f"last_result_upload_files_{project_id}_{user_id}"

    uploads = st.file_uploader(
        " ",
        type=["json"],
        accept_multiple_files=True,
        key=upload_widget_key
    )

    # 업로드가 아예 없는 경우에는 이전 기록 리셋 (같은 파일 이름 다시 업로드 가능하게)
    if not uploads:
        st.session_state.pop(last_files_key, None)

    if uploads:
        # 현재 선택된 파일 이름 목록 (정렬해서 순서 영향 제거)
        current_files = tuple(sorted(f.name for f in uploads))
        prev_files = st.session_state.get(last_files_key)

        # 이전에 처리한 적 없는 새로운 조합일 때만 업로드 처리
        if current_files != prev_files:
            for rf in uploads:
                msg = st.empty()
                msg.write(f"⏳ Uploading `{rf.name}` ...")

                res = upload_result_api(user_id, project_id, rf)
                if "error" in res:
                    msg.error(f"Upload failed: {res['error']}")
                else:
                    msg.success(f"Uploaded `{rf.name}`")
                time.sleep(0.4)
                msg.empty()

            # 이번 조합은 처리 완료로 기록
            st.session_state[last_files_key] = current_files

            # DB 갱신 후 다시 렌더링
            time.sleep(0.3)
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
        base = base.replace("_result", "")
        base = base.replace("_output", "")
        return base

    code_prefixes = {normalize_prefix(cf) for cf in code_files}
    result_prefixes = {normalize_prefix(rf) for rf in result_files}

    unmatched_codes = [
        cf for cf in code_files
        if normalize_prefix(cf) not in result_prefixes
    ]

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
    conn2.close()

    code_files = [fname for (_, fname, _) in code_rows]
    result_files = df_results["filename"].tolist()

    pairs = match_code_and_results(code_files, result_files)

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

                # metric 그룹화
                groups = group_metrics(metric_cols)

                # 탭 구성
                tabs = st.tabs(["Loss", "Classification", "Regression", "Other"])

                # ------------------------------------
                # 🔷 1) Classification Tab
                # ------------------------------------
                with tabs[1]:
                    acc_cols = groups["acc"]
                    f1_cols = groups["f1"]
                    
                    # ACC
                    if acc_cols:
                        dfa = df[["epoch"] + acc_cols]
                        melt = dfa.melt("epoch", acc_cols, "Metric", "Value")
                        fig = px.line(melt, x="epoch", y="Value", color="Metric",
                                    markers=True, title="Accuracy")
                        st.plotly_chart(fig, use_container_width=True)

                    # F1
                    if f1_cols:
                        dff = df[["epoch"] + f1_cols]
                        melt = dff.melt("epoch", f1_cols, "Metric", "Value")
                        fig = px.line(melt, x="epoch", y="Value", color="Metric",
                                    markers=True, title="F1 Score")
                        st.plotly_chart(fig, use_container_width=True)

                    if not acc_cols and not f1_cols:
                        st.info("No classification metrics found.")

                # ------------------------------------
                # 🔶 2) Loss Tab  (loss만 전담)
                # ------------------------------------
                with tabs[0]:

                    loss_cols = [m for m in metric_cols if "loss" in m.lower()]
                    
                    # train/val prefix만
                    selected = [
                        c for c in loss_cols
                        if c.startswith("train_") or c.startswith("val_")
                    ]

                    if selected:
                        dfx = df[["epoch"] + selected]
                        melt = dfx.melt("epoch", selected, "Metric", "Value")
                        fig = px.line(
                            melt, x="epoch", y="Value", color="Metric", markers=True,
                            title="Loss"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No train/val loss metrics found.")

                # ------------------------------------
                # 🔶 3) Regression Tab (mse/mae/rmse/r2)
                # ------------------------------------
                with tabs[2]:

                    # Loss는 제거하고 regression metric만
                    groups_reg = {
                        "mse":  [m for m in metric_cols if "mse"  in m.lower()],
                        "mae":  [m for m in metric_cols if "mae"  in m.lower()],
                        "rmse": [m for m in metric_cols if "rmse" in m.lower()],
                        "r2":   [m for m in metric_cols if "r2"   in m.lower()],
                    }

                    def select_train_val(cols):
                        return [
                            c for c in cols
                            if c.startswith("train_") or c.startswith("val_")
                        ]

                    drew_any_graph = False

                    for metric_name, cols in groups_reg.items():
                        selected = select_train_val(cols)
                        if not selected:
                            continue

                        dfx = df[["epoch"] + selected]
                        melt = dfx.melt("epoch", selected, "Metric", "Value")

                        fig = px.line(
                            melt, x="epoch", y="Value", color="Metric", markers=True,
                            title=f"{metric_name.upper()} (train + val)"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        drew_any_graph = True

                    if not drew_any_graph:
                        st.info("No regression metrics (train/val) available.")

                # ------------------------------------
                # 🔘 4) Other Tab
                # ------------------------------------
                with tabs[3]:
                    if groups["other"]:
                        dfo = df[["epoch"] + groups["other"]]
                        melt = dfo.melt("epoch", groups["other"], "Metric", "Value")
                        fig = px.line(melt, x="epoch", y="Value", color="Metric",
                                    markers=True, title="Other Metrics")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No other metrics.")

                
                # ------------------------------------
                # 최종 값 기록 (metric_records 축적)
                # ------------------------------------
                final_row = df.iloc[-1]

                for m in metric_cols:
                    metric_records.append({
                        "Code File": code_name,
                        "Result File": rname,
                        "Metric": m,
                        "Final Value": final_row[m]
                    })


    # =========================================================
    # 5) Final Performance Leaderboard (TEST ONLY)
    # =========================================================
    st.markdown("---")
    st.markdown("### 🏁 Final Performance Leaderboard")
    st.caption(
        "- Based only on **Test Metrics** from the matched result files.  "
        "\n- Metrics from the final epoch are used.  "
        "\n- Sorted automatically by metric direction (higher or lower is better)."
    )

    # test metric 필터링
    test_rows = [
        rec for rec in metric_records
        if rec["Metric"].lower().startswith("test_")
    ]

    if not test_rows:
        st.info("No test metrics available. Please upload result files containing test_* metrics.")
    else:
        df_leader = pd.DataFrame(test_rows)
        df_leader = df_leader.drop_duplicates(
            subset=["Code File", "Result File", "Metric"]
        )

        # metric 방향 정의
        def metric_direction(m):
            ml = m.lower()
            if "acc" in ml or "f1" in ml or "r2" in ml:
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

        # metric 종류별로 나누어 표시
        for m in df_leader["Metric"].unique():
            st.markdown(f"#### 📌 Performance : **{m}**")

            sub = df_leader[df_leader["Metric"] == m]

            st.dataframe(
                sub[["Code File", "Result File", "Metric", "Final Value"]].reset_index(drop=True),
                use_container_width=True
            )