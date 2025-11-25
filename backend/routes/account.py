from pydantic import BaseModel
import os
from fastapi import APIRouter, HTTPException
from frontend.utils.db import get_conn

router = APIRouter()

class UserDeleteRequest(BaseModel):
    user_id: int

@router.delete("/delete_account")
async def delete_account(req: UserDeleteRequest):
    user_id = req.user_id

    conn = get_conn()
    cur = conn.cursor()

    # 1) user가 가진 모든 project 조회
    projects = cur.execute("""
        SELECT id FROM projects WHERE user_id=?
    """, (user_id,)).fetchall()
    project_ids = [p[0] for p in projects]

    # 2) 연결된 file_id 조회
    file_ids = []
    for pid in project_ids:
        rows = cur.execute("""
            SELECT file_id FROM project_files WHERE project_id=?
        """, (pid,)).fetchall()
        file_ids.extend([r[0] for r in rows])

    file_ids = list(set(file_ids))

    # 3) 다른 프로젝트에서 사용되는지 체크 후 실제 삭제할 파일만 선택
    deletable_files = []
    for fid in file_ids:
        cnt = cur.execute("""
            SELECT COUNT(*) FROM project_files WHERE file_id=? 
        """, (fid,)).fetchone()[0]

        if cnt == 1:
            deletable_files.append(fid)

    # 4) 파일 삭제
    for fid in deletable_files:
        row = cur.execute("SELECT filepath FROM files WHERE id=?", (fid,)).fetchone()
        if row:
            path = row[0]
            if os.path.exists(path):
                os.remove(path)
        cur.execute("DELETE FROM files WHERE id=?", (fid,))

    # 5) 프로젝트 삭제
    for pid in project_ids:
        cur.execute("DELETE FROM project_files WHERE project_id=?", (pid,))
        cur.execute("DELETE FROM projects WHERE id=?", (pid,))

    # 6) user 삭제
    cur.execute("DELETE FROM users WHERE id=?", (user_id,))

    conn.commit()
    conn.close()

    return {
        "status": "ok",
        "deleted_projects": len(project_ids),
        "deleted_files": len(deletable_files)
    }
