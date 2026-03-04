# ml_code_insight/backend/db.py
import os
import sqlite3
import json
from typing import List, Dict, Optional

# Railway에서는 기본 파일시스템이 휘발성일 수 있어서,
# 최소한 경로는 환경변수로 뺄 수 있게 해둡니다.
DB_PATH = os.getenv("DB_PATH", "ml_insight.db")


# ------------------------------------------------------------
# (1) DB 연결 함수 — 연결만 생성 (WAL 모드)
# ------------------------------------------------------------
def get_conn():
    # timeout: 다른 write가 있을 때 기다리는 시간(초)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


# ------------------------------------------------------------
# (2) DB 스키마 초기화 — 앱 시작 시 1회 호출
# ------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    cur = conn.cursor()

    try:
        # users 테이블
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # files 테이블
        cur.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            filetype TEXT NOT NULL,
            summary_json TEXT,
            preview_json TEXT,
            uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
        """)

        # 파일 유니크 인덱스
        cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_files_user_name_type
        ON files(user_id, filename, filetype);
        """)

        # projects 테이블
        cur.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            project_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
        """)

        # project_files 테이블
        cur.execute("""
        CREATE TABLE IF NOT EXISTS project_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            file_id INTEGER NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects (id),
            FOREIGN KEY (file_id) REFERENCES files (id)
        );
        """)

        conn.commit()

    except sqlite3.OperationalError as e:
        # 동시에 여러 프로세스가 init_db를 치면 잠깐 락이 걸릴 수 있음
        if "database is locked" in str(e):
            pass
        else:
            raise
    finally:
        conn.close()


# ------------------------------------------------------------
# (3) 유저 코드 파일 목록 조회
# ------------------------------------------------------------
def list_user_code_files(user_id: int) -> List[str]:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT filename
        FROM files
        WHERE user_id = ? AND filetype = 'code'
        ORDER BY datetime(uploaded_at) DESC
    """, (user_id,))

    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]


# ------------------------------------------------------------
# (4) 특정 코드 파일 상세 정보 조회
# ------------------------------------------------------------
def get_code_file_info(user_id: int, filename: str) -> Optional[Dict]:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, filename, filetype, summary_json, preview_json, uploaded_at
        FROM files
        WHERE user_id = ? AND filename = ? AND filetype = 'code'
    """, (user_id, filename))

    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "id": row[0],
        "filename": row[1],
        "filetype": row[2],
        "summary": json.loads(row[3]) if row[3] else None,
        "preview": json.loads(row[4]) if row[4] else None,
        "uploaded_at": row[5],
    }


# ------------------------------------------------------------
# (5) 코드 파일 삭제
# ------------------------------------------------------------
def delete_code_file(user_id: int, filename: str):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        DELETE FROM files
        WHERE user_id = ? AND filename = ? AND filetype = 'code'
    """, (user_id, filename))

    conn.commit()
    conn.close()


# ------------------------------------------------------------
# (6) 특정 코드 파일과 매칭되는 result 파일 목록 조회
# ------------------------------------------------------------
def list_results_for_code(user_id: int, code_filename: str) -> List[Dict]:
    conn = get_conn()
    cur = conn.cursor()

    like_pattern = code_filename.replace(".py", "") + "%"

    cur.execute("""
        SELECT filename, summary_json, uploaded_at
        FROM files
        WHERE user_id = ? AND filetype = 'result' AND filename LIKE ?
        ORDER BY datetime(uploaded_at) DESC
    """, (user_id, like_pattern))

    rows = cur.fetchall()
    conn.close()

    return [
        {
            "filename": r[0],
            "summary": json.loads(r[1]) if r[1] else None,
            "uploaded_at": r[2]
        }
        for r in rows
    ]


def delete_project_and_unused_files(project_id: int, user_id: int):
    conn = get_conn()
    cur = conn.cursor()

    # 프로젝트 존재 / 권한 확인
    cur.execute("SELECT id FROM projects WHERE id=? AND user_id=?", (project_id, user_id))
    row = cur.fetchone()
    if not row:
        conn.close()
        return {"error": "Project does not exist or permission denied."}

    # 이 프로젝트가 사용하고 있는 file_id 조회
    cur.execute("SELECT file_id FROM project_files WHERE project_id=?", (project_id,))
    file_ids = [r[0] for r in cur.fetchall()]

    # project_files에서 프로젝트 row 삭제
    cur.execute("DELETE FROM project_files WHERE project_id=?", (project_id,))

    # 파일별로 다른 프로젝트에 연결 여부 확인
    for fid in file_ids:
        cur.execute("SELECT COUNT(*) FROM project_files WHERE file_id=?", (fid,))
        cnt = cur.fetchone()[0]

        if cnt > 0:
            continue

        cur.execute("DELETE FROM files WHERE id=?", (fid,))

    # 프로젝트 자체 삭제
    cur.execute("DELETE FROM projects WHERE id=?", (project_id,))

    conn.commit()
    conn.close()

    return {"ok": True}