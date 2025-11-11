import ast
import difflib

def extract_blocks(code_text):
    """함수, 클래스 단위로 파싱해서 (이름, 시작라인, 끝라인, 소스) 반환"""
    try:
        tree = ast.parse(code_text)
    except Exception:
        return []

    blocks = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
            start = node.lineno - 1
            end = max(getattr(node, "end_lineno", start + 1), start + 1)
            src = "\n".join(code_text.splitlines()[start:end])
            blocks.append((name, start, end, src))
    return blocks


def compare_code_ast(a_code, b_code):
    
    blocks_a = extract_blocks(a_code)
    blocks_b = extract_blocks(b_code)

    names_a = {name: (start, end, src) for name, start, end, src in blocks_a}
    names_b = {name: (start, end, src) for name, start, end, src in blocks_b}

    added, removed, modified = [], [], []
    changed_ranges = []   # ✅ 추가: 수정된 함수 범위 저장

    for name in names_a:
        if name not in names_b:
            removed.append(name)
            changed_ranges.append(("removed", *names_a[name][:2]))  # (type, start, end)
    for name in names_b:
        if name not in names_a:
            added.append(name)
            changed_ranges.append(("added", *names_b[name][:2]))
        else:
            a_start, a_end, a_src = names_a[name]
            b_start, b_end, b_src = names_b[name]
            if a_src.strip() != b_src.strip():
                if not _is_only_reordered(a_src, b_src):
                    modified.append(name)
                    changed_ranges.append(("modified", b_start, b_end))

    return {
        "added": added,
        "removed": removed,
        "modified": modified,
        "ranges": changed_ranges,   # ✅ 반환 추가
    }



def _is_only_reordered(a_func, b_func):
    """
    단순 순서 변경인지 판단:
    - 같은 줄 내용 집합(set)이 동일하면 reorder로 간주 (True 반환)
    """
    a_lines = [l.strip() for l in a_func.splitlines() if l.strip()]
    b_lines = [l.strip() for l in b_func.splitlines() if l.strip()]
    return set(a_lines) == set(b_lines)
