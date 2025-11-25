# pages/summaries.py
import streamlit as st
import pandas as pd
import json
from utils.db import get_conn
from utils.summarize import get_model_name, get_training


def app():
    st.header("🧾 Uploaded Codes Summary")

    user_id = st.session_state.get("user_id")
    if not user_id:
        st.info("Please log in first.")
        return

    conn = get_conn()
    df_codes = pd.read_sql("""
        SELECT filename, summary_json, uploaded_at
        FROM files WHERE user_id=? AND filetype='code'
        ORDER BY datetime(uploaded_at) DESC
    """, conn, params=(user_id,))

    if df_codes.empty:
        st.info("No code files uploaded yet.")
        return

    rows = []
    for _, row in df_codes.iterrows():
        s = json.loads(row["summary_json"]) if row["summary_json"] else {}
        rows.append({
            "Filename": row["filename"],
            "Model": get_model_name(s),
            "Optimizer": get_training(s, "optimizer", ""),
            "LR": get_training(s, "learning_rate", ""),
            "Batch": get_training(s, "batch_size", ""),
            "Epochs": get_training(s, "epochs", ""),
            "Loss": get_training(s, "loss", ""),
            "Scheduler": get_training(s, "scheduler", ""),
            "Device": get_training(s, "device", ""),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)
