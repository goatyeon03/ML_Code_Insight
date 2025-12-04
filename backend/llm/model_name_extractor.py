import re
import json
from backend.llm.gemini import gemini_free


LLM_MODEL_CLASS_PROMPT = """
You are an AI code analyzer. Your task is to identify the MAIN PyTorch model class
defined or instantiated in the given code.

RULES:
- Return ONLY the model class name (e.g., "ResNet18", "MyEncoder", "UNet").
- Do NOT return variable names such as "model" or "encoder".
- If multiple model classes appear, choose the one that is used for training 
  (optimizer = ..., loss_fn(...), model(x), model.parameters()).
- DO NOT return functions such as model.to(...), model.train(), model.cuda().
- NEVER return built-in function names like "to".
- You MUST return only a class name, not code.

STRICT OUTPUT FORMAT:
{"model_class": "<name or null>", "reason": "<short explanation>"}
"""

def llm_extract_model_class(code_text: str):
    prompt = LLM_MODEL_CLASS_PROMPT + "\n\nCODE:\n" + code_text
    try:
        raw = gemini_free(prompt)

        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
        else:
            parsed = {"model_class": None, "reason": "LLM returned non-JSON"}

        return parsed

    except Exception as e:
        return {
            "model_class": None,
            "reason": f"LLM error: {e}"
        }
