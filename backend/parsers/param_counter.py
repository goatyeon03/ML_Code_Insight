import ast
import inspect
import importlib.util
import os
import sys
import tempfile
import torch
import torch.nn as nn
from types import ModuleType

def force_single_model(model_name: str):
    """
    param_counter의 결과 dict 구조 형태에 맞춘 단일 모델 강제 설정.
    """
    return {
        "results": {
            model_name: {
                "total_params": None,
                "trainable_params": None,
                "error": None
            }
        },
        "error": None
    }


# ============================================================
# 🔥 Robust NN.Module inheritance detection
# ============================================================
def is_module_subclass(node, imported_bases):
    """
    node: ast.ClassDef
    imported_bases: dict mapping alias → actual module path
    ex: {'nn': 'torch.nn', 'F': 'torch.nn.functional'}
    """
    for base in node.bases:
        # direct name: class Foo(nn.Module)
        if isinstance(base, ast.Attribute):
            full = f"{imported_bases.get(base.value.id, base.value.id)}.{base.attr}"
            if full == "torch.nn.Module":
                return True

        if isinstance(base, ast.Name):
            # ex: class Foo(Module)
            if base.id == "Module":
                return True
            # alias 처리: class Foo(nn.Module)
            if base.id in imported_bases:
                if imported_bases[base.id] == "torch.nn":
                    return True

    return False


# ============================================================
# 🔥 Dummy constructor args generator
# ============================================================
def generate_dummy_args(cls):
    sig = inspect.signature(cls.__init__)
    kwargs = {}

    for name, p in sig.parameters.items():
        if name == "self":
            continue

        # default값 있으면 사용
        if p.default is not inspect.Parameter.empty:
            continue

        # annotation 기반 dummy
        if p.annotation in [int, float]:
            kwargs[name] = 1
        elif p.annotation == bool:
            kwargs[name] = False
        elif p.annotation == str:
            kwargs[name] = ""
        elif p.annotation in [tuple, list]:
            kwargs[name] = [1]
        else:
            # 모르는 타입이면 None
            kwargs[name] = None

    return kwargs


# ============================================================
# 🔥 Instantiate model with dummy arguments
# ============================================================
def instantiate_model(cls):
    try:
        # 1차: 기본 생성자
        return cls()
    except Exception:
        pass

    try:
        # 2차: dummy args 기반 시도
        dummy = generate_dummy_args(cls)
        return cls(**dummy)
    except Exception as e:
        return f"Instantiation failed: {e}"


# ============================================================
# 🔥 Extract nn.Module classes using AST (robust)
# ============================================================
def extract_model_classes(src: str):
    tree = ast.parse(src)

    imported_bases = {}
    model_classes = []

    for node in ast.walk(tree):

        # import torch.nn as nn
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.asname:
                    imported_bases[n.asname] = n.name

        # from torch import nn
        if isinstance(node, ast.ImportFrom):
            if node.module:
                for n in node.names:
                    if n.asname:
                        imported_bases[n.asname] = f"{node.module}"
                    else:
                        imported_bases[n.name] = f"{node.module}"

        # class detection
        if isinstance(node, ast.ClassDef):
            if is_module_subclass(node, imported_bases):
                model_classes.append(node.name)

    return model_classes


# ============================================================
# 🔥 Load model class from temp module
# ============================================================
def load_class_from_file(filepath, class_name):
    """
    File → temp module import → get attribute
    """
    spec = importlib.util.spec_from_file_location("temp_user_model", filepath)
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return None, f"Module load failed: {e}"

    if not hasattr(module, class_name):
        return None, f"class `{class_name}` not found in module"

    return getattr(module, class_name), None


# ============================================================
# 🔥 Count params
# ============================================================
def count_params(model):
    try:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable, None
    except Exception as e:
        return None, None, str(e)


# ============================================================
# 🔥 Public API
# ============================================================
def get_param_count(file_path: str):
    """
    Returns:
    {
      "results": {
         class_name: {
            "total_params": ...,
            "trainable_params": ...,
            "error": None or "msg"
         }
      },
      "error": None
    }
    """

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception as e:
        return {"results": {}, "error": f"file read error: {e}"}

    # -----------------------------------------
    # 1) AST 기반 모델 클래스 추출
    # -----------------------------------------
    model_classes = extract_model_classes(src)

    if not model_classes:
        # 안전한 기본 반환
        return {
            "results": {},
            "error": "No nn.Module subclasses found"
        }

    results = {}

    # -----------------------------------------
    # 2) temp file 생성 후 모델 import
    # -----------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_file = os.path.join(tmpdir, "model_temp.py")

        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(src)

        for cls_name in model_classes:
            cls, err = load_class_from_file(temp_file, cls_name)
            if err:
                results[cls_name] = {
                    "total_params": None,
                    "trainable_params": None,
                    "error": err
                }
                continue

            # -----------------------------------------
            # 3) instantiate model (dummy args allowed)
            # -----------------------------------------
            instance = instantiate_model(cls)
            if isinstance(instance, str):  # error message
                results[cls_name] = {
                    "total_params": None,
                    "trainable_params": None,
                    "error": instance
                }
                continue

            # -----------------------------------------
            # 4) count params
            # -----------------------------------------
            total, trainable, err = count_params(instance)

            results[cls_name] = {
                "total_params": total,
                "trainable_params": trainable,
                "error": err
            }

    return {
        "results": results,
        "error": None
    }

def get_param_count_for_class(file_path: str, class_name: str):
    """
    refined_summary 또는 dashboard가 전달한 class_name 기반으로만 파라미터 계산.
    extract_model_classes를 사용하지 않고, 지정된 클래스만 사용.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception as e:
        return {"results": {}, "error": f"file read error: {e}"}

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_file = os.path.join(tmpdir, "model_temp.py")

        # 사용자 모델 임시 저장
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(src)

        # 모델 로드 시도
        cls, err = load_class_from_file(temp_file, class_name)
        if err:
            return force_single_model(class_name)

        # 인스턴스 생성
        instance = instantiate_model(cls)
        if isinstance(instance, str):  # 에러 메세지
            return force_single_model(class_name)

        # 파라미터 계산
        total, trainable, err = count_params(instance)

        return {
            "results": {
                class_name: {
                    "total_params": total,
                    "trainable_params": trainable,
                    "error": err
                }
            },
            "error": None
        }
