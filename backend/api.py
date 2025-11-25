# backend/api.py

from fastapi import FastAPI, Query, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import os
import shutil
import json

from frontend.utils.db import get_conn, init_db

# 서버 시작 시 한 번만 스키마 초기화
init_db()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_CODE = os.path.join(BASE_DIR, "uploads", "code")
UPLOAD_RESULT = os.path.join(BASE_DIR, "uploads", "results")
os.makedirs(UPLOAD_CODE, exist_ok=True)
os.makedirs(UPLOAD_RESULT, exist_ok=True)

app = FastAPI(title="ML Code Insight API", version="0.2.0")

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
):
    # --- 파일 저장 ---
    save_path = os.path.join(UPLOAD_CODE, file.filename)
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        # UploadFile 내부 파일 포인터 닫기
        file.file.close()

    # --- 코드 요약 생성 ---
    from backend.parsers.code_parser import summarize_code

    try:
        summary = summarize_code(save_path)
    except Exception as e:
        # 요약에 실패해도 업로드 자체는 실패로 보는 편이 자연스러워서 500
        raise HTTPException(status_code=500, detail=f"Failed to summarize code: {e}")

    conn = get_conn()
    cur = conn.cursor()
    try:
        # files 테이블 upsert
        cur.execute(
            """
            INSERT INTO files (user_id, filename, filetype, summary_json)
            VALUES (?, ?, 'code', ?)
            ON CONFLICT(user_id, filename, filetype)
            DO UPDATE SET summary_json = excluded.summary_json
            """,
            (user_id, file.filename, json.dumps(summary)),
        )

        # file_id 가져오기
        cur.execute(
            """
            SELECT id FROM files
            WHERE user_id = ? AND filename = ? AND filetype = 'code'
            """,
            (user_id, file.filename),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("Failed to fetch file_id after insert.")
        file_id = row[0]

        # project_files 연결 (중복 방지)
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

    return {
        "filename": file.filename,
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
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM project_files WHERE file_id=?", (file_id,))
        cur.execute("DELETE FROM files WHERE id=? AND user_id=?", (file_id, user_id))
        conn.commit()
    except Exception as e:
        conn.rollback()
        return {"error": str(e)}
    finally:
        conn.close()

    return {"status": "ok", "file_id": file_id}

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
