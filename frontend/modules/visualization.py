# pages/visualizations.py
import streamlit as st
import pandas as pd
import json
import plotly.express as px

from utils.db import get_conn
from utils.match_utils import match_code_and_results


def app():
    st.header("📈 Matched Pair Visualization")

    user_id = st.session_state.get("user_id")
    if not user_id:
        st.info("Please log in first.")
        return
    
    project_id = st.session_state.get("selected_project")
    if not project_id:
        st.info("Please select a project first.")
        return

    conn = get_conn()

    df_codes = pd.read_sql("""
        SELECT f.filename
        FROM files f
        JOIN project_files pf ON pf.file_id = f.id
        WHERE pf.project_id=? AND f.user_id=? AND f.filetype='code'
    """, conn, params=(project_id, user_id))

    df_results = pd.read_sql("""
        SELECT f.filename, f.preview_json
        FROM files f
        JOIN project_files pf ON pf.file_id = f.id
        WHERE pf.project_id=? AND f.user_id=? AND f.filetype='result'
    """, conn, params=(project_id, user_id))


    code_files = df_codes["filename"].tolist()
    result_files = df_results["filename"].tolist()

    pairs = match_code_and_results(code_files, result_files)

    # unmatched 경고
    matched_all = {r for lst in pairs.values() for r in lst}
    unmatched_codes = [c for c in code_files if c not in pairs or not pairs[c]]
    unmatched_results = [r for r in result_files if r not in matched_all]

    if unmatched_codes or unmatched_results:
        msgs = []
        if unmatched_codes:
            msgs.append("⚠️ Unmatched code files: " + ", ".join(unmatched_codes))
        if unmatched_results:
            msgs.append("⚠️ Unmatched result files: " + ", ".join(unmatched_results))
        st.warning("\n\n".join(msgs))
    elif pairs:
        st.success("✅ All code and result files matched successfully!")

    if not pairs:
        st.info("No matched pairs found.")
        return

    # pair별 expander
    for code_name, results in pairs.items():
        with st.expander(f"🧠 {code_name} ↔ {', '.join(results)}", expanded=False):
            for rname in results:
                row = df_results[df_results["filename"] == rname]
                if row.empty:
                    continue

                try:
                    preview_json = row.iloc[0]["preview_json"]
                    data = json.loads(preview_json)
                    df = pd.DataFrame(data if isinstance(data, list) else [data])

                    if "epoch" in df.columns:
                        metric_cols = [
                            c for c in df.columns
                            if c.lower() not in ["epoch", "step", "iteration"]
                        ]
                        train_cols = [c for c in metric_cols if c.startswith("train_")]
                        val_cols = [c for c in metric_cols if c.startswith("val_")]

                        if train_cols and val_cols:
                            metrics = sorted({c.split("_", 1)[1] for c in train_cols})
                            for metric in metrics:
                                t_col = f"train_{metric}"
                                v_col = f"val_{metric}"
                                if t_col in df.columns and v_col in df.columns:
                                    melt = df.melt(
                                        id_vars="epoch",
                                        value_vars=[t_col, v_col],
                                        var_name="Type",
                                        value_name="Value",
                                    )
                                    fig = px.line(
                                        melt,
                                        x="epoch",
                                        y="Value",
                                        color="Type",
                                        markers=True,
                                        title=f"{metric.upper()} (Train vs Val)",
                                    )
                                    st.plotly_chart(
                                        fig,
                                        use_container_width=True,
                                        key=f"{code_name}_{rname}_{metric}",
                                    )
                        else:
                            melt = df.melt(
                                id_vars="epoch",
                                var_name="metric",
                                value_name="value",
                            )
                            fig = px.line(
                                melt,
                                x="epoch",
                                y="value",
                                color="metric",
                                markers=True,
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.dataframe(df, use_container_width=True)

                except Exception as e:
                    st.warning(f"⚠️ Could not visualize {rname}: {e}")
