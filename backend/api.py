# FastAPI 기반 백엔드

'''
- 코드(.py) 업로드 받아서 서버에 저장
- 업로드된 코드에서 AST 요약 → Gemini로 요약 정제 → DB 저장
- result(json) 업로드 받아서 preview 만들고 DB 저장
- 업로드된 파일 읽기/삭제, 프로젝트 생성/삭제
'''

from fastapi import FastAPI, Query, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from contextlib import asynccontextmanager
import os
import shutil
import json
import subprocess

# 백앤드가 프론트 모듈을 import하고 있음 -> 배포할 때 ModuleNotFoundError 주의 (백앤드 기준으로 통일)
from backend.db import get_conn, init_db
from backend.routes.account import router as account_router
from backend.llm.param_estimator import estimate_params_with_llm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 업로드 루트
UPLOAD_ROOT = os.getenv("UPLOAD_ROOT", os.path.join(BASE_DIR, "uploads"))
UPLOAD_CODE = os.path.join(UPLOAD_ROOT, "code")
UPLOAD_RESULT = os.path.join(UPLOAD_ROOT, "results")

# CORS: 배포 시에는 Streamlit 도메인만 넣는 걸 추천
# 예) ALLOW_ORIGINS="https://xxxx.streamlit.app,https://yyyy.streamlit.app"
_raw_origins = os.getenv("ALLOW_ORIGINS", "")
ALLOW_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]
if not ALLOW_ORIGINS:
    # 개발 편의 기본값 (운영에서는 환경변수로 꼭 지정 권장)
    ALLOW_ORIGINS = ["*"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 앱 시작 시 1회
    os.makedirs(UPLOAD_CODE, exist_ok=True)
    os.makedirs(UPLOAD_RESULT, exist_ok=True)
    init_db()
    yield
    # 종료 시 정리할 것이 있으면 여기에


# app 생성
app = FastAPI(
    title="Pytorch Experiment Dashboard API", 
    version="0.2.0",
    lifespan=lifespan,
    )

# 라우터 등록
# 계정 삭제 엔드포인트
app.include_router(account_router)

# CORS
# 프론트가 백앤드 호출할 때 막히지 않도록
# 배포 시 도메인 제한 필요
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 배포 후 서버가 살아있는 지 확인용
@app.get("/health")
def health():
    return {"status": "ok"}


# ==========================================================
# 1) Code Upload → DB write + project_files 연결 (백엔드 전담)
# ==========================================================
# 배포 시 업로드 경로 일관되게 하기
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
    # AST + code text
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
    # Gemini로 요약 정제
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
        # files 테이블에 summary_json 저장
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
        
        # project_files 연결
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
        "status": "ok",
        "filename": filename,
        "summary": summary,
        #"file_id": file_id,
    }


# ==========================================================
# 2) Result Upload → DB write + project_files 연결 (백엔드 전담)
# ==========================================================
# 트랜잭션 스타일이 upload_code와 다름에 주의
@app.post("/upload_result")
async def upload_result(
    user_id: int = Form(...),
    project_id: int = Form(...),
    file: UploadFile = File(...),
    override_name: str = Form(None),
):
    filename = override_name if override_name else file.filename
    save_path = os.path.join(UPLOAD_RESULT, filename)

    # 업로드 된 json 파일 저장
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()


    # json 읽어서 preview 생성(앞의 5개)
    try:
        with open(save_path, "r", encoding="utf-8") as f:
            result_json = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {{e}}")
    
    summary = {
        "type": "result",
        "metrics": result_json.get("metrics", result_json),
    }


    conn = get_conn()
    cur = conn.cursor()

    try:
        # DB에 preview_json 저장
        cur.execute(
        """
        INSERT INTO files (user_id, filename, filetype, summary_json)
        VALUES (?, ?, 'result', ?)
        ON CONFLICT(user_id, filename, filetype)
        DO UPDATE SET summary_json=excluded.summary_json
        """, 
        (user_id, filename, json.dumps(summary))
        )

        cur.execute(
            """
            SELECT id FROM files
            WWHERE user_id = ? AND filename = ? AND filetype = 'result'
            """,
            (user_id, filename),
        )
        row = cur.fetchone()
        if not row:
            raise RuntimeError("Failed to fetch file_id after insert.")
        file_id = row[0]

        cur.execute(
            """
            INSERT INTO project_files (project_id, file_id)
            VALUES (?, ?)
            """,
            (project_id, file_id),
        )

        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()
    
    return {"status": "ok", "filename": filename, "summary": summary}


# ==========================================================
# 3) LLM-based param estimation endpoint
# ==========================================================
@app.post("/estimate_params_llm")
async def estimate_params_llm_endpoint(
    model_name: str = Form(...),
    code_text: str = Form(...),
):
    try:
        result = estimate_params_with_llm(model_name=model_name, code_text=code_text)
        return {"status": "ok", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================
# 4) Local param counter (subprocess)
# ==========================================================
@app.post("/count_params_local")
async def count_params_local(
    file_path: str = Form(...),
    timeout_sec: int = Form(10),
):
    worker_path = os.path.join(BASE_DIR, "parsers", "param_worker.py")
    try:
        proc = subprocess.run(
            ["python3", worker_path, file_path],
            capture_output=True,
            text=True,
            timeout=int(timeout_sec),
        )
        if proc.returncode != 0:
            raise HTTPException(status_code=500, detail=proc.stderr.strip() or "worker failed")
        return PlainTextResponse(proc.stdout)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="param worker timeout")
