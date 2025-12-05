# utils/db.py
import sqlite3
import json
from typing import List, Dict, Optional

DB_PATH = "ml_insight.db"


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
# (2) DB 스키마 초기화 — 앱 시작 시 1회 호출 (여러 프로세스가 동시에
#     호출해도 database is locked 가 나면 조용히 무시하고 빠져나오도록 처리)
# ------------------------------------------------------------
def init_db():
    # init 단계에서도 WAL + timeout 설정
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
        # 다른 프로세스가 이미 init_db 를 실행 중이라 잠깐 락이 걸릴 수 있음
        # 이 경우에는 "이미 누가 초기화하고 있다"고 보고 그냥 넘어가도 됨.
        if "database is locked" in str(e):
            # 필요하다면 여기서 로그만 남기고 조용히 패스
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
    """
    결과 파일 규칙 예시:
    - train_xxx.py  → train_xxx_result.json
    - train_xxx.py  → train_xxx_*.json
    등 prefix 매칭용.
    """
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
    """
    프로젝트 삭제 + 해당 프로젝트에 속한 파일들이 다른 프로젝트에 사용되지 않는다면 files에서도 삭제
    """
    conn = get_conn()
    cur = conn.cursor()

    # --- 1) 프로젝트 존재 / 권한 확인 ---
    cur.execute("""
        SELECT id FROM projects WHERE id=? AND user_id=?
    """, (project_id, user_id))
    row = cur.fetchone()
    if not row:
        conn.close()
        return {"error": "Project does not exist or permission denied."}

    # --- 2) 이 프로젝트가 사용하고 있는 모든 file_id 조회 ---
    cur.execute("""
        SELECT file_id FROM project_files 
        WHERE project_id=?
    """, (project_id,))
    file_ids = [r[0] for r in cur.fetchall()]

    # --- 3) project_files 에서 이 프로젝트 row 삭제 ---
    cur.execute("DELETE FROM project_files WHERE project_id=?", (project_id,))

    # --- 4) 파일별로 '다른 프로젝트에 연결 여부 확인' ---
    for fid in file_ids:
        cur.execute("""
            SELECT COUNT(*) FROM project_files 
            WHERE file_id=? 
        """, (fid,))
        cnt = cur.fetchone()[0]

        # 다른 프로젝트에서도 사용되면 삭제하지 않음
        if cnt > 0:
            continue

        # files 테이블에서 제거
        cur.execute("DELETE FROM files WHERE id=?", (fid,))

    # --- 5) 프로젝트 자체 삭제 ---
    cur.execute("DELETE FROM projects WHERE id=?", (project_id,))

    conn.commit()
    conn.close()

    return {"ok": True}

