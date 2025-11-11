# backend/parsers/diff_utils.py
import os, ast, difflib, re

CODE_ROOT = os.path.join(os.path.dirname(__file__), "..", "uploads", "code")

def _full_path(filename: str) -> str:
    # 업로드된 코드 디렉토리 기준의 안전 경로
    path = os.path.normpath(os.path.join(CODE_ROOT, filename))
    if not path.startswith(os.path.normpath(CODE_ROOT)):
        raise ValueError("Invalid path")
    return path

def read_text(filename: str) -> list[str]:
    with open(_full_path(filename), "r", encoding="utf-8") as f:
        return f.read().splitlines(keepends=False)

def generate_html_diff(file_a: str, file_b: str) -> str:
    a_lines = read_text(file_a)
    b_lines = read_text(file_b)
    h = difflib.HtmlDiff(wrapcolumn=100)
    return h.make_table(a_lines, b_lines, fromdesc=file_a, todesc=file_b)

def _extract_functions(filepath: str):
    """함수/메서드 정보를 [(name, lineno, end_lineno, src_hash)]로 반환"""
    full = _full_path(filepath)
    with open(full, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)

    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Python 3.8+ 에서 end_lineno 있음
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            name = node.name
            # 소스 해시(간단 비교용)
            if start and end:
                snippet = "".join(lines[start-1:end])
            else:
                snippet = name  # fallback
            src_hash = hash(snippet)
            funcs.append((name, start, end, src_hash))
    return funcs

def compare_functions(file_a: str, file_b: str):
    """추가/삭제/수정 함수 목록 리턴"""
    fa = _extract_functions(file_a)
    fb = _extract_functions(file_b)

    dict_a = {n: (s, e, h) for n, s, e, h in fa}
    dict_b = {n: (s, e, h) for n, s, e, h in fb}

    names_a = set(dict_a.keys())
    names_b = set(dict_b.keys())

    added = sorted(list(names_b - names_a))
    removed = sorted(list(names_a - names_b))

    modified = []
    for n in names_a & names_b:
        if dict_a[n][2] != dict_b[n][2]:
            modified.append(n)
    modified.sort()

    return {
        "added": added,
        "removed": removed,
        "modified": modified
    }

def summarize_functions(file_x: str):
    """프론트에서 목록 보여줄 때 쓸 간단 요약"""
    out = []
    for n, s, e, _ in _extract_functions(file_x):
        out.append({"name": n, "lineno": s, "end_lineno": e})
    return out
