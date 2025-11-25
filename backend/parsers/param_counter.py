# backend/parsers/param_counter.py

import ast
import os
import tempfile
import importlib.util
import torch.nn as nn


# ---------------------------------------------------------------
# 1) AST 분석: import / class 정의만 추출하기
# ---------------------------------------------------------------

def extract_model_class_ast(file_path: str):
    """
    원본 학습 코드에서:
    - import 문들
    - nn.Module을 상속하는 Class 정의
    만 AST로 추출
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)

    import_nodes = []
    class_nodes = []

    for node in tree.body:
        # import 관련 노드
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_nodes.append(node)

        # class 정의 노드
        elif isinstance(node, ast.ClassDef):
            # base classes 확인
            is_nn_module = False
            for base in node.bases:
                # 예: nn.Module
                if isinstance(base, ast.Attribute) and base.attr == "Module":
                    is_nn_module = True
                # 예: Module (from torch.nn import Module)
                elif isinstance(base, ast.Name) and base.id == "Module":
                    is_nn_module = True

            if is_nn_module:
                class_nodes.append(node)

    return import_nodes, class_nodes


# ---------------------------------------------------------------
# 2) 추출한 AST → 임시 파일 생성
# ---------------------------------------------------------------

def create_temp_model_file(import_nodes, class_nodes):
    """
    import + class_def AST를 하나의 코드로 재구성하여
    임시 파이썬 파일 생성
    """
    temp_code = ""

    # import 문 다시 코드로 변환
    for node in import_nodes:
        temp_code += ast.unparse(node) + "\n"

    temp_code += "\n"

    # class 정의 코드 추가
    for node in class_nodes:
        temp_code += ast.unparse(node) + "\n\n"

    # 임시 파일 생성
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8")
    tmp.write(temp_code)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------
# 3) 임시 파일에서 클래스 import
# ---------------------------------------------------------------

def load_classes_from_module(temp_file):
    """
    임시 파이썬 파일을 import해서 nn.Module subclass들을 로드
    """
    spec = importlib.util.spec_from_file_location("temp_model_module", temp_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_classes = []
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
            model_classes.append(obj)

    return model_classes


# ---------------------------------------------------------------
# 4) param count 계산
# ---------------------------------------------------------------

def get_param_count_for_class(cls):
    """
    기본 생성자에서 인자 없는 경우만 instantiate하여 param count 계산
    """
    import inspect
    sig = inspect.signature(cls.__init__)

    # 기본값 없는 인자가 있으면 스킵
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if p.default is inspect.Parameter.empty:
            return {
                "class_name": cls.__name__,
                "total_params": None,
                "error": f"Constructor of {cls.__name__} requires arguments."
            }

    # instantiate
    try:
        model = cls()
    except Exception as e:
        return {
            "class_name": cls.__name__,
            "total_params": None,
            "error": f"Instantiation failed: {e}"
        }

    # param count
    try:
        total = sum(p.numel() for p in model.parameters())
        return {
            "class_name": cls.__name__,
            "total_params": int(total),
            "error": None
        }
    except Exception as e:
        return {
            "class_name": cls.__name__,
            "total_params": None,
            "error": f"Param count failed: {e}"
        }


# ---------------------------------------------------------------
# 5) 파일 전체에 대한 param count 결과 반환
# ---------------------------------------------------------------

def get_param_count(file_path: str):
    """
    최종 API에서 호출할 함수.
    파일에서 class 정의만 추출 → 임시 파일 생성 → param count 계산 → 반환
    """
    try:
        imports, classes = extract_model_class_ast(file_path)

        if not classes:
            return {"error": "No nn.Module classes detected."}

        # 임시 파일 생성
        temp_file = create_temp_model_file(imports, classes)

        # 임시 파일에서 클래스 로드
        model_classes = load_classes_from_module(temp_file)

        # 임시 파일 삭제
        os.remove(temp_file)

        # 파라미터 개수 계산
        results = {}
        for cls in model_classes:
            results[cls.__name__] = get_param_count_for_class(cls)

        return {
            "error": None,
            "results": results
        }

    except Exception as e:
        return {"error": f"Param counter fatal error: {e}"}
