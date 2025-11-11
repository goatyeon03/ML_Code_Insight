from backend.parsers.code_parser import summarize_code
from backend.parsers.diff_utils import generate_html_diff, compare_functions, summarize_functions
from backend.parsers.compare_code_ast import compare_code_ast

from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import os, shutil, json



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_CODE = os.path.join(BASE_DIR, "uploads", "code")
UPLOAD_RESULT = os.path.join(BASE_DIR, "uploads", "results")
os.makedirs(UPLOAD_CODE, exist_ok=True)
os.makedirs(UPLOAD_RESULT, exist_ok=True)

app = FastAPI(title="ML Code Insight API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload_code")
async def upload_code(file: UploadFile = File(...)):
    save_path = os.path.join(UPLOAD_CODE, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    summary = summarize_code(save_path)
    return {"filename": file.filename, "summary": summary}

@app.post("/upload_result")
async def upload_result(file: UploadFile = File(...)):
    save_path = os.path.join(UPLOAD_RESULT, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    preview = {}
    try:
        if file.filename.endswith(".json"):
            with open(save_path) as jf:
                data = json.load(jf)
            if isinstance(data, list):
                preview = data[:5]
            elif isinstance(data, dict):
                preview = {k: (v if not isinstance(v, list) else v[:5]) for k, v in data.items()}
        else:
            preview = {"info": "CSV 등은 다음 단계에서 지원"}
    except Exception as e:
        preview = {"error": str(e)}
    return {"filename": file.filename, "preview": preview}

@app.post("/compare_code")
def compare_code(payload: dict):
    import os, json
    type_ = "code"
    file_a = payload.get("file_a")
    file_b = payload.get("file_b")
    base_dir = os.path.join(os.path.dirname(__file__), "uploads", type_)

    path_a = os.path.join(base_dir, file_a)
    path_b = os.path.join(base_dir, file_b)
    if not (os.path.exists(path_a) and os.path.exists(path_b)):
        return {"error": "파일을 찾을 수 없습니다."}

    with open(path_a, "r", encoding="utf-8") as f:
        a_code = f.read()
    with open(path_b, "r", encoding="utf-8") as f:
        b_code = f.read()

    changes = compare_code_ast(a_code, b_code)
    return {"function_changes": {
                "added": changes["added"],
                "removed": changes["removed"],
                "modified": changes["modified"]
            },
            "ranges": changes["ranges"]}

@app.get("/list_files")
def list_files(type: str = "code"):
    """type='code'면 uploads/code 폴더를 스캔"""
    import os
    base_dir = os.path.join(os.path.dirname(__file__), "uploads", type)
    if not os.path.exists(base_dir):
        return {"files": [], "base_dir": base_dir, "exists": False}
    files = [f for f in os.listdir(base_dir) if f.endswith(".py")]
    return {"files": sorted(files), "base_dir": base_dir, "exists": True}

@app.get("/get_file", response_class=PlainTextResponse)
def get_file(type: str, filename: str):
    import os
    base_dir = os.path.join(os.path.dirname(__file__), "uploads", type)
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        return PlainTextResponse("파일을 찾을 수 없습니다.", status_code=404)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return PlainTextResponse(content)


