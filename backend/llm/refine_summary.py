import json
import re
from backend.llm.gemini import gemini_free

def refine_summary_with_gemini(ast_summary, code_text):

    """
    Refine AST summary using Gemini LLM.
    Force LLM to MIGRATE values from 'overall' to specific stages (pretrain/finetune).
    """

    training_flow = ast_summary.get("training_flow", {})
    
    # AST가 찾은 초기 값들을 프롬프트에 명확히 보여줌
    initial_overall = ast_summary.get("training", {}).get("overall", {})

    prompt = f"""
You are an AI system that refines machine-learning training summaries.
Your goal is to STRUCTURE the training metadata correctly.


----------------------------------------------------
### INPUT DATA

1. **AST SIGNALS (Logic Hints)**:
   - has_training_loop   : {training_flow.get("has_training_loop")}
   - has_pretrained_load : {training_flow.get("has_pretrained_load")} (If True, likely Finetuning)
   - has_backward        : {training_flow.get("has_backward")}

2. **AST DETECTED VALUES (Currently in 'Overall')**:
   {json.dumps(initial_overall, indent=2)}
   *(WARNING: AST puts everything in 'overall'. You must MOVE these to the correct stage if a specific stage is detected.)*

3. **FULL SOURCE CODE**:
```python
{code_text}

4. AST DETECTED MODEL CLASS:
    {json.dumps(ast_summary.get("model", {}), indent=2)}
    *You MUST preserve the detected model class name from AST unless code clearly defines a different final model.
----------------------------------------------------
### YOUR TASK (LOGIC FLOW)
1. Detect the Stage:
    - Scenario A (Finetuning): Code loads weights (pretrained=True, load_state_dict) AND trains. -> ACTION: Move ALL valid hyperparameters to training.stages.finetune. -> CLEAR training.overall (make it empty).
    - Scenario B (Pretraining): Code initializes from scratch and trains. -> ACTION: Move ALL valid hyperparameters to training.stages.pretrain. -> CLEAR training.overall.
    - Scenario C (Pretrain + Finetune): Two distinct training phases detected. -> ACTION: Split parameters into stages.pretrain and stages.finetune.
    - Scenario D (Generic/Unknown): No specific transfer learning logic. -> ACTION: Keep values in training.overall.
    - Scenario E (Standard Training):
        If has_pretrained_load == False
        AND has_training_loop == True:
            -> CLASSIFY SCRIPT AS TRAINING.
            -> Move ALL hyperparameters into training.stages.train.
            -> training.stages.pretrain = {{}}
            -> training.stages.finetune = {{}}
            -> training.overall = {{}}

2. IMPORTANT:
    Even if AST did NOT detect optimizer, loss, epochs, batch_size, or learning rate,
    you MUST extract these values directly from the Python code.
    Do NOT leave them null if they exist anywhere in the script.

----------------------------------------------------
### OUTPUT FORMAT (STRICT JSON)
Output ONLY valid JSON inside a code block.
{{
  "dataset": {{ "name": null, "path": null }},
  "model": {{"name": "<MODEL_NAME>", "backbone": null }},
  "training": {{
      "detected_stage_type": "finetune" OR "pretrain" OR "overall",
      "overall": {{}}, 
      "stages": {{
          "pretrain": {{ "epochs": null, "batch_size": null, "lr": null, "optimizer": null, "loss": null }},
          "train": {{}},
          "finetune": {{ "epochs": null, "batch_size": null, "lr": null, "optimizer": null, "loss": null }}
      }},
      "device": null
  }},
  "notes": "Explain why you chose this stage."
}}
----------------------------------------------------


### CRITICAL RULES:
- If you fill stages.finetune, training.overall MUST be empty {{}}.
- Do NOT leave parameters in overall if you detected a specific stage.
- Return JSON only. """

    # LLM 호출
    output = gemini_free(prompt)


    # ---------------------------------------------------------
    # 강력한 JSON 파싱 로직 + 모델 이름 복원 + stages 반영
    # ---------------------------------------------------------
    try:
        # 1) 먼저 ```json ... ``` 코드 블록 찾기
        match = re.search(r"```json\s*\n*(\{.*?\})\s*```", output, re.DOTALL)
        if match:
            json_str = match.group(1)
            refined = json.loads(json_str)
        else:
            # 2) fallback — JSON 전체 범위 탐색
            match2 = re.search(r"(\{.*\})", output, re.DOTALL)
            if match2:
                json_str = match2.group(1)
                refined = json.loads(json_str)
            else:
                return ast_summary   # JSON 자체가 없음 → AST 반환

        # -------------------------------
        # ⭐ 모델 이름 복원 (AST 기반)
        # -------------------------------
        ast_model_name = ast_summary.get("model", {}).get("name")

        if not refined.get("model"):
            refined["model"] = {}

        if refined["model"].get("name") in (None, "", "null", {}, []):
            refined["model"]["name"] = ast_model_name

        # -------------------------------
        # ⭐ training 구조 보정
        # -------------------------------
        if "training" not in refined:
            refined["training"] = {}

        refined["training"].setdefault("overall", {})
        refined["training"].setdefault("stages", {})
        refined["training"]["stages"].setdefault("pretrain", {})
        refined["training"]["stages"].setdefault("train", {})
        refined["training"]["stages"].setdefault("finetune", {})

        # -------------------------------
        # ⭐ 모델 클래스는 딱 1곳에만 저장
        # -------------------------------
        if ast_model_name:
            refined["training"]["overall"]["model_class"] = ast_model_name

        return refined


    except Exception as e:
        with open("/home/goatyeon/ml_code_insight/backend/llm/refine_debug2.txt", "a") as f:
            f.write("[DEBUG] JSON PARSE FAILED with EXCEPTION\n")
            f.write("ERROR: " + str(e) + "\n")
            f.write("RAW OUTPUT:\n" + output + "\n")

        return ast_summary








