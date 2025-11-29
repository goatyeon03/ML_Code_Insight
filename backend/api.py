# backend/api.py

from fastapi import FastAPI, Query, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import os
import shutil
import json

from frontend.utils.db import get_conn, init_db
from frontend.utils.match_utils import match_code_and_results

from backend.routes.account import router as account_router
from backend.parsers.param_counter import get_param_count_for_class

# 서버 시작 시 한 번만 스키마 초기화
init_db()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_CODE = os.path.join(BASE_DIR, "uploads", "code")
UPLOAD_RESULT = os.path.join(BASE_DIR, "uploads", "results")
os.makedirs(UPLOAD_CODE, exist_ok=True)
os.makedirs(UPLOAD_RESULT, exist_ok=True)

app = FastAPI(title="ML Code Insight API", version="0.2.0")

app.include_router(account_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


# ==========================================================
# 1) Code Upload → DB write + project_files 연결 (백엔드 전담)
# ==========================================================
@app.post("/upload_code")
async def upload_code(
    user_id: int = Form(...),
    project_id: int = Form(...),
    file: UploadFile = File(...),
    override_name: str = Form(None),
):
    # --------------------------
    # 1) 저장 파일명 결정
    # --------------------------
    filename = override_name if override_name else file.filename
    save_path = os.path.join(UPLOAD_CODE, filename)

    # 파일 저장
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()

    # --------------------------
    # 2) AST summary 생성
    # --------------------------
    from backend.parsers.code_parser import summarize_code
    try:
        ast_summary = summarize_code(save_path)
    except Exception as e:
        ast_summary = {"error": f"AST parse error: {e}"}

    # summary 구조 보정
    if not isinstance(ast_summary, dict):
        ast_summary = {"error": "invalid summary"}

    # --------------------------
    # 3) 코드 원문 읽기
    # --------------------------
    try:
        with open(save_path, "r", encoding="utf-8") as f:
            code_text = f.read()
    except Exception as e:
        code_text = ""
        ast_summary.setdefault("notes", "")
        ast_summary["notes"] += f"\n[WARN] Could not read code text: {e}"

    # --------------------------
    # 4) LLM refine 호출
    # --------------------------
    from backend.llm.refine_summary import refine_summary_with_gemini


    try:
        refined_summary = refine_summary_with_gemini(ast_summary, code_text)
        

    except Exception as e:
        # LLM 오류 발생하면 AST summary 사용
        refined_summary = ast_summary
        refined_summary.setdefault("notes", "")
        refined_summary["notes"] += f"\n[LLM refine failed: {e}]"

    summary = refined_summary

    # summary 기본 키 유지
    summary.setdefault("model", {})
    summary.setdefault("training", {})
    summary.setdefault("dataset", {})
    summary.setdefault("misc", {})

    # --------------------------
    # 5) DB 저장
    # --------------------------
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO files (user_id, filename, filetype, summary_json)
            VALUES (?, ?, 'code', ?)
            ON CONFLICT(user_id, filename, filetype)
            DO UPDATE SET summary_json = excluded.summary_json
            """,
            (user_id, filename, json.dumps(summary)),
        )

        cur.execute(
            """
            SELECT id FROM files
            WHERE user_id = ? AND filename = ? AND filetype = 'code'
            """,
            (user_id, filename),
        )
        row = cur.fetchone()
        if not row:
            raise RuntimeError("Failed to fetch file_id after insert.")
        file_id = row[0]

        cur.execute(
            """
            INSERT OR IGNORE INTO project_files (project_id, file_id)
            VALUES (?, ?)
            """,
            (project_id, file_id),
        )

        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"upload_code failed: {e}")
    finally:
        conn.close()

    # --------------------------
    # 6) 반환
    # --------------------------
    return {
        "filename": filename,
        "summary": summary,
        "file_id": file_id,
    }


# ==========================================================
# 2) Result Upload → DB write + project_files 연결 (백엔드 전담)
# ==========================================================
@app.post("/upload_result")
async def upload_result(
    user_id: int = Form(...),
    project_id: int = Form(...),
    file: UploadFile = File(...)
):
    save_path = os.path.join(UPLOAD_RESULT, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # preview 생성
    try:
        with open(save_path) as jf:
            data = json.load(jf)
        if isinstance(data, list):
            preview = data[:5]
        else:
            preview = data
    except Exception as e:
        preview = {"error": str(e)}

    # === DB insert (safe transaction) ===
    import sqlite3
    from frontend.utils.db import get_conn

    conn = get_conn()
    cur = conn.cursor()

    # 파일 INSERT (result)
    cur.execute("""
        INSERT INTO files (user_id, filename, filetype, preview_json)
        VALUES (?, ?, 'result', ?)
        ON CONFLICT(user_id, filename, filetype)
        DO UPDATE SET preview_json=excluded.preview_json
    """, (user_id, file.filename, json.dumps(preview)))
    conn.commit()

    # file_id 가져오기
    cur.execute("""
        SELECT id FROM files
        WHERE user_id=? AND filename=? AND filetype='result'
    """, (user_id, file.filename))
    row = cur.fetchone()
    if not row:
        conn.close()
        return {"error": "DB insert failed (file not found after insert)"}

    file_id = row[0]

    # project_files 관계 INSERT
    cur.execute("""
        INSERT INTO project_files (project_id, file_id)
        VALUES (?, ?)
        ON CONFLICT(project_id, file_id)
        DO NOTHING
    """, (project_id, file_id))
    conn.commit()

    conn.close()
    return {"filename": file.filename, "preview": preview}


# ==========================================================
# 3) 기타 API (읽기 전용)
# ==========================================================
@app.get("/get_file", response_class=PlainTextResponse)
def get_file(type: str, filename: str):
    base_dir = os.path.join(os.path.dirname(__file__), "uploads", type)
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        return PlainTextResponse("파일을 찾을 수 없습니다.", status_code=404)
    with open(path, "r", encoding="utf-8") as f:
        return PlainTextResponse(f.read())


@app.get("/model_blocks")
def get_model_blocks(filename: str = Query(..., description="코드 파일 이름")):
    file_path = os.path.join(UPLOAD_CODE, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Code file not found.")

    from backend.parsers.model_parser import parse_model_structure

    try:
        return parse_model_structure(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model parsing failed: {e}")

@app.post("/delete_file")
async def delete_file(user_id: int = Form(...), file_id: int = Form(...)):
    """
    코드 파일 삭제 시 → 매칭된 result 파일까지 자동 삭제.
    result 파일 삭제 시 → 그 파일만 삭제.
    """

    conn = get_conn()
    cur = conn.cursor()

    try:
        # 1) 삭제 대상 파일 정보 조회 (filename, filetype)
        cur.execute("""
            SELECT filename, filetype
            FROM files
            WHERE id=? AND user_id=?
        """, (file_id, user_id))
        row = cur.fetchone()

        if not row:
            conn.close()
            return {"error": "File not found"}

        filename, filetype = row

        # 2) 실제 파일 삭제 경로 결정
        base_path = UPLOAD_CODE if filetype == "code" else UPLOAD_RESULT
        full_path = os.path.join(base_path, filename)

        # 3) 프로젝트 ID 조회 (해당 파일이 속해 있는 모든 프로젝트)
        cur.execute("""
            SELECT project_id
            FROM project_files
            WHERE file_id=?
        """, (file_id,))
        project_rows = cur.fetchall()
        project_ids = [r[0] for r in project_rows]

        matched_results = []

        # 4) code 파일인 경우 → result 파일 자동 삭제 로직 실행
        if filetype == "code":
            from frontend.utils.match_utils import match_code_and_results

            for pid in project_ids:
                # pid 내 모든 code/result 파일 조회
                cur.execute("""
                    SELECT f.filename
                    FROM files f JOIN project_files pf ON pf.file_id=f.id
                    WHERE pf.project_id=? AND f.user_id=? AND f.filetype='code'
                """, (pid, user_id))
                code_files = [r[0] for r in cur.fetchall()]

                cur.execute("""
                    SELECT f.filename
                    FROM files f JOIN project_files pf ON pf.file_id=f.id
                    WHERE pf.project_id=? AND f.user_id=? AND f.filetype='result'
                """, (pid, user_id))
                result_files = [r[0] for r in cur.fetchall()]

                # 매칭 실행
                pairs = match_code_and_results(code_files, result_files)
                matched = pairs.get(filename, [])
                matched_results.extend(matched)

                # result 파일 DB + 실제 파일 삭제
                for rname in matched:
                    # DB에서 찾기
                    cur.execute("""
                        SELECT id FROM files
                        WHERE user_id=? AND filename=? AND filetype='result'
                    """, (user_id, rname))
                    rrow = cur.fetchone()
                    if rrow:
                        rid = rrow[0]
                        cur.execute("DELETE FROM project_files WHERE file_id=?", (rid,))
                        cur.execute("DELETE FROM files WHERE id=? AND user_id=?", (rid, user_id))

                        # 실제 파일 삭제
                        result_path = os.path.join(UPLOAD_RESULT, rname)
                        if os.path.exists(result_path):
                            os.remove(result_path)

        # 5) 코드/결과 파일 자체 삭제 (DB + 실제 파일)
        cur.execute("DELETE FROM project_files WHERE file_id=?", (file_id,))
        cur.execute("DELETE FROM files WHERE id=? AND user_id=?", (file_id, user_id))

        conn.commit()
    except Exception as e:
        conn.rollback()
        return {"error": str(e)}
    finally:
        conn.close()

    # 실제 파일 삭제
    if os.path.exists(full_path):
        os.remove(full_path)

    return {
        "status": "ok",
        "deleted_code_or_result": filename,
        "auto_deleted_results": matched_results
    }


@app.post("/create_project")
async def create_project(user_id: int = Form(...), project_name: str = Form(...)):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO projects (user_id, project_name)
            VALUES (?, ?)
        """, (user_id, project_name))
        pid = cur.lastrowid
        conn.commit()
    except Exception as e:
        conn.rollback()
        return {"error": str(e)}
    finally:
        conn.close()
    return {"project_id": pid}

@app.post("/delete_project")
def api_delete_project(user_id: int = Form(...), project_id: int = Form(...)):
    from frontend.utils.db import delete_project_and_unused_files
    result = delete_project_and_unused_files(project_id, user_id)
    return result

@app.get("/param_count")
def api_param_count(filename: str, class_name: str):
    file_path = os.path.join(UPLOAD_CODE, filename)
    return get_param_count_for_class(file_path, class_name)


@app.post("/param_count_enhanced")
async def param_count_enhanced(
    filename: str = Form(...),
    class_name: str = Form(...)
):
    from backend.parsers.param_counter import get_param_count_for_class
    from backend.llm.gemini import gemini_free

    file_path = os.path.join(UPLOAD_CODE, filename)

    # -----------------------------
    # 1) param_counter 기반 계산
    # -----------------------------
    pc_result = get_param_count_for_class(file_path, class_name)

    # -----------------------------
    # 2) LLM 추정 (프롬프트 생략)
    # -----------------------------
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code_text = f.read()
    except:
        code_text = ""

    llm_prompt = """
You are an AI system that analyzes arbitrary PyTorch code and computes the 
number of trainable parameters WITHOUT relying on any externally provided 
model class name. You must infer everything directly from the code.

Your tasks:

===============================================================
1) Identify the actual model used for training
===============================================================
From the entire code:

- Identify which object serves as the REAL MODEL.
- Acceptable candidates:
  * Any class inheriting from nn.Module
  * Any object created via composition (front/backbone/head)
  * Any nn.Sequential used as the top-level model

Rules:
- Ignore helper functions such as evaluate(), train_epoch(), inference(), validate(), etc.
- Ignore dataset/dataloader/optimizer/loss definitions.
- Ignore modules that are only used inside another module (SEBlock, CBraModBlock, etc.).
- If a model is created via:
      model = RegressionHead(backbone)
  then RegressionHead IS the real model.

- If the model is:
      model = nn.Sequential(front, se, backbone)
  treat the Sequential composition as the FULL model.

- If multiple nn.Module classes exist:
  choose the one that produces final predictions and is instantiated before training.

===============================================================
2) Reconstruct model architecture and compute parameters
===============================================================
For every layer, compute parameter counts explicitly.

Linear(in, out):
  params = (in * out) + out

Conv1d(in_channels, out_channels, kernel_size):
  params = (in_channels * out_channels * kernel_size) + out_channels

Conv2d(in, out, kH, kW):
  params = (in * out * kH * kW) + out

BatchNorm:
  params = num_features * 2

For custom modules:
- Inspect __init__()
- Identify submodules (Conv/Linear/etc.)
- Recursively sum parameters

If necessary, infer missing shapes logically from context.

===============================================================
3) If shapes cannot be determined reliably:
===============================================================
Set estimated_params = null  
and explain which shapes caused ambiguity.

===============================================================
4) Final output (STRICT JSON)
===============================================================
Return ONLY JSON:

{
  "model_class": "<the model class or 'Sequential' or composition>",
  "estimated_params": <int or null>,
  "reasoning": "<step-by-step explanation of: 
                  (a) how the model was identified, 
                  (b) each layer's parameter calculation,
                  (c) any inference or fallback logic used>"
}

NO markdown. NO backticks. NO text outside JSON.

===============================================================
Python Code:
```python
{code_text}


"""

    llm_raw = gemini_free(llm_prompt)

    import json, re
    llm_estimate = {"estimated_params": None, "reasoning": ""}
    try:
        match = re.search(r"(\{.*\})", llm_raw, re.DOTALL)
        if match:
            llm_estimate = json.loads(match.group(1))
    except:
        pass

    # -----------------------------
    # 3) 결과 병합
    # -----------------------------
    pc_val = None
    try:
        pc_val = pc_result["results"][class_name]["total_params"]
    except:
        pc_val = None

    llm_val = llm_estimate.get("estimated_params")

    # 병합 규칙
    if pc_val and llm_val:
        merged = pc_val
        notes = f"LLM estimated {llm_val}, param_counter calculated {pc_val}."
    elif pc_val:
        merged = pc_val
        notes = "LLM estimate unavailable; using param_counter value."
    elif llm_val:
        merged = llm_val
        notes = "param_counter failed; using LLM estimate."
    else:
        merged = None
        notes = "Both param_counter and LLM estimation failed."

    return {
        "merged": merged,
        "param_counter": pc_result,
        "llm_estimate": llm_estimate,
        "notes": notes,
    }

 