# backend/llm/param_estimator.py

import json
import re
from backend.llm.gemini import gemini_free


# ---------------------------------------------------------
# 1) 프롬프트 템플릿 정의
# ---------------------------------------------------------
LLM_PARAM_COUNT_PROMPT = """
You are an expert in analyzing PyTorch model code.

Your job is to estimate the total number of trainable parameters across ALL modules
that participate in training, not just one model.

You are given:

1) A list of modules that were detected as trainable:
   MODULES = {{modules_json}}

2) For each module, you are given structural blocks extracted from the AST model parser:
   STRUCTURE = {{structure_json}}

Rules:
- Compute parameters PER MODULE when shapes are inferable.
- If a module's parameter count cannot be determined due to missing sizes,
  set estimated_params = null for that module.
- The total parameters = sum of all module estimated_params (ignoring nulls).
- DO NOT invent layers or input sizes not present in STRUCTURE.
- DO NOT infer shapes from variable names; only use explicit numeric or tuple shapes.
- If a module has no valid layers with numeric shapes, its estimated_params = null.
- If ALL modules have null, total_params must be null.

STRICT JSON OUTPUT:
{
  "total_params": <int or null>,
  "modules": {
      "<module_class>": {
          "estimated_params": <int or null>,
          "reason": "<explanation>"
      },
      ...
  },
  "reasoning": "<global explanation>"
}
"""



# ---------------------------------------------------------
# MULTI-MODULE LLM ESTIMATOR
# ---------------------------------------------------------
def estimate_params_with_llm(modules: dict, structures: dict):
    """
    modules: {"encoder": "CBraModEncoder", "rt_head": "RtHead"}
    structures: {"CBraModEncoder": {...}, "RtHead": {...}}

    return:
    {
        "total_params": int or None,
        "modules": {...},
        "reasoning": "..."
    }
    """
    try:
        modules_json = json.dumps(modules, indent=2)
        structure_json = json.dumps(structures, indent=2)

        prompt = (
            LLM_PARAM_COUNT_PROMPT
            .replace("{{modules_json}}", modules_json)
            .replace("{{structure_json}}", structure_json)
        )

        # LLM 호출
        llm_raw = gemini_free(prompt)

        # JSON 추출
        match = re.search(r"(\{.*\})", llm_raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group(1))
        else:
            parsed = {
                "total_params": None,
                "modules": {},
                "reasoning": "LLM returned non-JSON output."
            }

        return parsed

    except Exception as e:
        return {
            "total_params": None,
            "modules": {},
            "reasoning": f"LLM error: {e}"
        }
