import ast
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


@dataclass
class LayerNode:
    """
    한 레이어(모듈) 정의 정보
    """
    class_name: str        # 모델 클래스 이름 (예: "MyNet")
    attribute: str         # self.conv1 에서 conv1
    layer_type: str        # Conv2d, Linear, ReLU 등
    args: str              # 위치 인자 표현 (문자열)
    kwargs: str            # 키워드 인자 표현 (문자열)
    line_no: int           # 소스 코드 라인 번호 (1-base)
    source: str            # 해당 라인의 원본 코드


# ============================================================
# AST 유틸 함수들
# ============================================================
def _expr_to_str(node: ast.AST) -> str:
    """
    인자 표현을 문자열로 안전하게 변환.
    - 숫자/문자열/리스트/튜플 같은 literal이면 literal_eval 시도
    - 그 외에는 ast.unparse (3.9+) 혹은 repr로 fallback
    """
    import ast as _ast

    try:
        value = _ast.literal_eval(node)
        return repr(value)
    except Exception:
        try:
            return _ast.unparse(node)  # Python 3.9+
        except Exception:
            return repr(node)


def _is_nn_module_base(base: ast.expr) -> bool:
    """
    class Foo(nn.Module): 혹은 class Foo(Module): 같은 패턴 감지
    """
    # nn.Module / torch.nn.Module 등
    if isinstance(base, ast.Attribute):
        if base.attr == "Module" and isinstance(base.value, ast.Name):
            return True

    # from torch.nn import Module; class Foo(Module):
    if isinstance(base, ast.Name) and base.id == "Module":
        return True

    return False


def _find_model_classes(tree: ast.AST) -> List[ast.ClassDef]:
    """
    최상위에서 nn.Module을 상속한 클래스들만 찾기
    """
    classes: List[ast.ClassDef] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if any(_is_nn_module_base(b) for b in node.bases):
                classes.append(node)
    return classes


def _find_init_method(cls: ast.ClassDef) -> Optional[ast.FunctionDef]:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            return node
    return None


def _extract_layers_from_init(
    cls: ast.ClassDef,
    init_func: ast.FunctionDef,
    lines: List[str],
) -> List[LayerNode]:
    """
    __init__ 내부에서 self.xxx = Something(...) 패턴을 찾아 LayerNode 리스트로 변환
    Sequential 은 내부 sub layer까지 풀어서 기록
    """
    layers: List[LayerNode] = []

    for node in ast.walk(init_func):
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue

        if len(targets) != 1:
            continue
        target = targets[0]

        if not isinstance(target, ast.Attribute):
            continue
        if not (isinstance(target.value, ast.Name) and target.value.id == "self"):
            continue

        # 왼쪽: self.xxx
        attr_name = target.attr

        # 오른쪽이 함수 호출인지 확인
        if not isinstance(value, ast.Call):
            continue

        func = value.func

        # 레이어 타입 이름
        if isinstance(func, ast.Attribute):
            layer_type = func.attr
        elif isinstance(func, ast.Name):
            layer_type = func.id
        else:
            layer_type = type(func).__name__

        lineno = getattr(node, "lineno", None)
        src_line = ""
        if lineno and 1 <= lineno <= len(lines):
            src_line = lines[lineno - 1].strip()

        # -------------------------
        # Sequential 처리
        # -------------------------
        if layer_type == "Sequential":
            type_counter: Dict[str, int] = {}

            for sub in value.args:
                if not isinstance(sub, ast.Call):
                    continue

                sub_func = sub.func
                if isinstance(sub_func, ast.Attribute):
                    sub_type = sub_func.attr
                elif isinstance(sub_func, ast.Name):
                    sub_type = sub_func.id
                else:
                    sub_type = type(sub_func).__name__

                type_counter.setdefault(sub_type, 0)
                type_counter[sub_type] += 1
                idx = type_counter[sub_type]

                sub_attr = f"{attr_name}.{sub_type}{idx}"

                args_str = ", ".join(_expr_to_str(a) for a in sub.args)
                kwargs_str = ", ".join(
                    f"{kw.arg}={_expr_to_str(kw.value)}"
                    for kw in sub.keywords
                    if kw.arg
                )

                layers.append(
                    LayerNode(
                        class_name=cls.name,
                        attribute=sub_attr,
                        layer_type=sub_type,
                        args=args_str,
                        kwargs=kwargs_str,
                        line_no=lineno or -1,
                        source=src_line,
                    )
                )

            # Sequential 자체는 별도 레이어로 기록하지 않음
            continue

        # -------------------------
        # 일반 레이어 처리
        # -------------------------
        args_str = ", ".join(_expr_to_str(a) for a in value.args)
        kwargs_str = ", ".join(
            f"{kw.arg}={_expr_to_str(kw.value)}"
            for kw in value.keywords
            if kw.arg
        )

        layers.append(
            LayerNode(
                class_name=cls.name,
                attribute=attr_name,
                layer_type=layer_type,
                args=args_str,
                kwargs=kwargs_str,
                line_no=lineno or -1,
                source=src_line,
            )
        )

    layers.sort(key=lambda n: n.line_no if n.line_no else 10**9)
    return layers


# ============================================================
# 메인 함수 (외부에서 사용하는 진입점)
# ============================================================
def parse_model_structure(path: str) -> Dict[str, Any]:
    """
    전체 파일의 모델 구조를 분석:
    - 모든 nn.Module subclass 파싱
    - 클래스 기반 blocks 추출
    - top-level 변수 기반 blocks 추출
    - model graph & pipeline 구성
    """
    # 1) 파일 읽기 & AST 생성
    try:
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
    except FileNotFoundError:
        return {"top_model": None, "models": {}, "pipeline": [], "error": f"File not found: {path}"}

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"top_model": None, "models": {}, "pipeline": [], "error": f"SyntaxError: {e}"}

    lines = code.splitlines()

    # 2) 모든 모델 클래스 찾기
    model_classes = _find_model_classes(tree)
    model_names = {cls.name for cls in model_classes}

    # 저장 구조
    models: Dict[str, Dict[str, Any]] = {}

    # 3) class 기반 blocks / children 추출
    for cls in model_classes:
        init_func = _find_init_method(cls)
        if init_func:
            layer_nodes = _extract_layers_from_init(cls, init_func, lines)
            blocks = group_nodes_by_source(layer_nodes)
        else:
            blocks = []

        # children (다른 모델 클래스를 __init__에서 생성하는 경우)
        children: List[str] = []
        if init_func:
            for node in ast.walk(init_func):
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    value = node.value
                else:
                    continue

                if not isinstance(value, ast.Call):
                    continue

                func = value.func
                if isinstance(func, ast.Name):
                    fname = func.id
                elif isinstance(func, ast.Attribute):
                    fname = func.attr
                else:
                    continue

                if fname in model_names and fname != cls.name:
                    children.append(fname)

        models[cls.name] = {
            "blocks": blocks,
            "children": children,
        }

    # 4) Top-level assignment 기반 블럭 추출
    top_assigns = _find_top_level_assignments(tree)
    top_blocks = _build_top_level_blocks(top_assigns, lines)

    models["_TopLevelModule"] = {
        "blocks": top_blocks,
        "children": [],
    }

    # 5) top-model 결정: "마지막으로 생성된 모델 클래스" 기준
    usage_order: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                fname = func.id
            elif isinstance(func, ast.Attribute):
                fname = func.attr
            else:
                continue

            if fname in model_names:
                usage_order.append(fname)

    if usage_order:
        top_model = usage_order[-1]
    else:
        # fallback: children 에 등장하지 않는 클래스 중 파일에서 가장 마지막에 등장한 것
        all_children = set()
        for m in models.values():
            all_children.update(m["children"])
        candidates = list(model_names - all_children)

        if candidates:
            class_order = [cls.name for cls in model_classes]
            top_model = None
            for name in reversed(class_order):
                if name in candidates:
                    top_model = name
                    break
        else:
            top_model = "_TopLevelModule"

    pipeline = extract_pipeline_by_class(tree, model_names)

    return {
        "top_model": top_model,
        "models": models,
        "pipeline": pipeline,
        "error": None,
    }


# ============================================================
# Block 그룹핑 / Top-level 분석 / Pipeline 추출
# ============================================================
def group_nodes_by_source(layer_nodes: List[LayerNode]) -> List[Dict[str, Any]]:
    """
    같은 source(한 줄)에서 생성된 레이어들을 한 block으로 묶음
    """
    blocks: Dict[str, List[Dict[str, Any]]] = {}
    for node in layer_nodes:
        key = node.source
        blocks.setdefault(key, []).append(asdict(node))

    block_list: List[Dict[str, Any]] = []
    for source, nodes in blocks.items():
        valid_lines = [n["line_no"] for n in nodes if n.get("line_no", 0) > 0]
        line_no = min(valid_lines) if valid_lines else -1
        block_list.append({
            "source": source,
            "line_no": line_no,
            "layers": nodes,
        })

    block_list.sort(key=lambda b: b["line_no"])
    return block_list


def _find_top_level_assignments(tree: ast.AST) -> List[Dict[str, Any]]:
    """
    파일 최상단에서 나타나는 X = Something(...) 패턴을 모두 찾는다.
    """
    assigns: List[Dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            targets = []
            for t in node.targets:
                if isinstance(t, ast.Name):
                    targets.append(t.id)
            if not targets:
                continue

            func = node.value.func
            if isinstance(func, ast.Name):
                called = func.id
            elif isinstance(func, ast.Attribute):
                called = func.attr
            else:
                continue

            assigns.append({
                "vars": targets,
                "call": called,
                "value": node.value,
                "lineno": node.lineno,
            })
    return assigns


def _build_top_level_blocks(assignments: List[Dict[str, Any]],
                            lines: List[str]) -> List[Dict[str, Any]]:
    """
    top-level 변수 할당에서 block 정보를 구성
    (예: model = MyModel(...), front = nn.Sequential(...))
    """
    blocks: List[Dict[str, Any]] = []

    for a in assignments:
        lineno = a["lineno"]
        src_line = lines[lineno - 1].strip() if 1 <= lineno <= len(lines) else ""

        value = a["value"]

        # nn.Sequential(...) 패턴인지 확인
        if isinstance(value.func, ast.Attribute) and value.func.attr == "Sequential":
            layer_nodes = []
            type_counter: Dict[str, int] = {}

            for sub in value.args:
                if not isinstance(sub, ast.Call):
                    continue

                sub_func = sub.func
                if isinstance(sub_func, ast.Name):
                    ltype = sub_func.id
                elif isinstance(sub_func, ast.Attribute):
                    ltype = sub_func.attr
                else:
                    continue

                type_counter.setdefault(ltype, 0)
                type_counter[ltype] += 1
                idx = type_counter[ltype]

                layer_nodes.append({
                    "class_name": "_TopLevelModule",
                    "attribute": f"{a['vars'][0]}.{ltype}{idx}",
                    "layer_type": ltype,
                    "args": ", ".join(_expr_to_str(arg) for arg in sub.args),
                    "kwargs": ", ".join(
                        f"{kw.arg}={_expr_to_str(kw.value)}"
                        for kw in sub.keywords
                        if kw.arg
                    ),
                    "line_no": lineno,
                    "source": src_line,
                })

            blocks.append({
                "source": src_line,
                "line_no": lineno,
                "layers": layer_nodes,
            })

        else:
            # 일반 모듈 할당 (예: model = MyModel(...))
            called = a["call"]
            blocks.append({
                "source": src_line,
                "line_no": lineno,
                "layers": [{
                    "class_name": "_TopLevelModule",
                    "attribute": a["vars"][0],
                    "layer_type": called,
                    "args": "",
                    "kwargs": "",
                    "line_no": lineno,
                    "source": src_line,
                }],
            })

    blocks.sort(key=lambda b: b["line_no"])
    return blocks


def extract_pipeline_by_class(tree: ast.AST,
                              model_classes) -> List[str]:
    """
    AST에서 호출된 모델 클래스 순서를 class 단위로만 추출.
    Conv/ReLU 등 Layer는 포함되지 않음.
    """
    classes = set(model_classes)
    order: List[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                fname = func.id
            elif isinstance(func, ast.Attribute):
                fname = func.attr
            else:
                continue

            if fname in classes:
                order.append(fname)

    final: List[str] = []
    for name in order:
        if name not in final:
            final.append(name)
    return final

