"""
Param worker that uses model_parser's class relationship.

- model_parser.parse_model_structure(path)로 nn.Module 클래스와 children 관계를 가져온다.
- target_class 기준으로 필요한 자식 클래스들을 모두 찾는다.
- 의존성 순서(자식 → 부모)대로 클래스 정의를 sandbox 네임스페이스에 exec 한다.
- 마지막에 target_class 인스턴스를 생성해서 파라미터 수를 계산한다.

결과 형식:
{
  "results": {
    "<target_class>": {
      "total_params": <int or None>,
      "trainable_params": <int or None>,
      "error": <str or None>
    }
  }
}
"""

import ast
import json
import sys
import traceback
import torch

from model_parser import parse_model_structure


# ---------------------------------------------------------
# 1) 파일에서 모든 class 정의 소스 추출
# ---------------------------------------------------------
def extract_class_sources(src_text: str):
    """
    src_text 에 포함된 모든 클래스 정의를 찾아
    {class_name: class_source_code} 형태로 반환.
    """
    tree = ast.parse(src_text)
    class_map = {}

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            src = ast.get_source_segment(src_text, node)
            if src is None:
                continue
            class_map[node.name] = src

    return class_map


# ---------------------------------------------------------
# 2) model_parser 의 children 정보 기반 의존성 정렬
# ---------------------------------------------------------
def collect_required_classes(models_dict, target_class: str):
    """
    models_dict: model_parser.parse_model_structure()["models"]
    target_class 로부터 children 을 따라가며 필요한 클래스 이름들의 집합을 반환.
    """
    needed = set()

    def dfs(cls):
        if cls in needed:
            return
        if cls not in models_dict:
            return
        needed.add(cls)
        for child in models_dict[cls].get("children", []):
            dfs(child)

    dfs(target_class)
    return needed


def topo_order_from_children(models_dict, needed_classes):
    """
    children 관계 (부모 -> 자식)를 이용해서
    자식이 먼저, 부모가 나중에 오도록 topological order 생성.
    """
    visited = set()
    order = []

    def dfs(cls):
        if cls in visited:
            return
        visited.add(cls)
        for child in models_dict.get(cls, {}).get("children", []):
            if child in needed_classes:
                dfs(child)
        order.append(cls)

    for cls in needed_classes:
        dfs(cls)

    # 중복 제거된 순서 (이미 visited로 처리되어 있으니 order 자체가 topological)
    return order


# ---------------------------------------------------------
# 3) sandbox 네임스페이스에서 클래스 exec + 인스턴스 생성
# ---------------------------------------------------------
def safe_exec_classes(class_order, class_sources):
    """
    class_order 순서대로 class_sources[cls] 를 exec 하여
    하나의 네임스페이스(ns)를 반환.
    """
    safe_globals = {
        "__builtins__": __builtins__,
        "torch": torch,
        "nn": torch.nn,
        "F": torch.nn.functional,
    }
    ns = {}

    for cls in class_order:
        src = class_sources.get(cls)
        if not src:
            # 소스가 없으면 그냥 넘어감 (instantiate 시점에서 실패 처리)
            continue
        exec(src, safe_globals, ns)

    return ns


def count_parameters(model):
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


# ---------------------------------------------------------
# 4) main
# ---------------------------------------------------------
def main():
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Usage: param_worker <file> <class_name>"}))
        return

    file_path = sys.argv[1]
    target_class = sys.argv[2]

    try:
        # 0) 코드 전체 읽기
        with open(file_path, "r", encoding="utf-8") as f:
            src = f.read()

        # 1) model_parser 로 구조 분석
        structure = parse_model_structure(file_path)
        models = structure.get("models", {})

        if target_class not in models:
            # model_parser 기준으로 nn.Module 모델이 아닌 경우
            print(json.dumps({
                "results": {
                    target_class: {
                        "total_params": None,
                        "trainable_params": None,
                        "error": f"class '{target_class}' not found in model_parser models"
                    }
                }
            }))
            return

        # 2) 필요한 클래스 집합 & 의존성 순서
        needed = collect_required_classes(models, target_class)
        class_order = topo_order_from_children(models, needed)

        # 3) 실제 파일에서 클래스 소스 추출
        class_sources = extract_class_sources(src)

        # 4) sandbox 에서 순서대로 exec
        ns = safe_exec_classes(class_order, class_sources)

        if target_class not in ns:
            print(json.dumps({
                "results": {
                    target_class: {
                        "total_params": None,
                        "trainable_params": None,
                        "error": "class execution failed in sandbox"
                    }
                }
            }))
            return

        cls = ns[target_class]

        # 5) 인스턴스 생성 (기본적으로 __init__ 인자를 요구하지 않는다고 가정)
        try:
            model = cls()
        except Exception as e:
            print(json.dumps({
                "results": {
                    target_class: {
                        "total_params": None,
                        "trainable_params": None,
                        "error": f"instantiation failed: {e}"
                    }
                }
            }))
            return

        # 6) 파라미터 수 계산
        total, trainable = count_parameters(model)

        print(json.dumps({
            "results": {
                target_class: {
                    "total_params": total,
                    "trainable_params": trainable,
                    "error": None
                }
            }
        }))

    except Exception as e:
        print(json.dumps({
            "results": {
                target_class: {
                    "total_params": None,
                    "trainable_params": None,
                    "error": f"worker crashed: {e}\n{traceback.format_exc()}"
                }
            }
        }))


if __name__ == "__main__":
    main()
