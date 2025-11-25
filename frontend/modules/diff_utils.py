# frontend/modules/diff_utils.py
import ast

def compare_code_text(a_text: str, b_text: str):
    """
    frontend에서 사용하는 lightweight 함수.
    함수 블록 기준으로 비교하지 않고,
    전체 텍스트 기반 비교 (사용자가 원하는 '변경 여부' 판단만).
    """

    if a_text.strip() == b_text.strip():
        return {"added": [], "removed": [], "modified": []}

    # 최소 기능: 다른 부분이 있다면 modified로 처리
    return {"added": [], "removed": [], "modified": ["changed"]}
