# utils/summarize.py
import json


def coerce_summary(x):
    """
    summary_json이 dict or JSON string일 수 있으니 안전하게 dict로 변환.
    """
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            pass
    return {"dataset": {}, "model": {"class_name": "Unknown"},
            "training": {}, "misc": {}}


def get_training(s, k, d=""):
    s = coerce_summary(s)
    return s.get("training", {}).get(k, d)


def get_model_name(s):
    s = coerce_summary(s)
    return s.get("model", {}).get("class_name", "Unknown")


def detect_task_type(df):
    """
    result preview DataFrame의 컬럼을 보고
    classification / regression / unknown 추정.
    """
    joined = " ".join(c.lower() for c in df.columns)
    if any(k in joined for k in ["acc", "accuracy", "f1", "precision", "recall"]):
        return "classification"
    if any(k in joined for k in ["mse", "mae", "r2", "rmse", "loss"]):
        return "regression"
    return "unknown"
