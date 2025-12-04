import ast
from backend.llm.model_name_extractor import llm_extract_model_class
from backend.parsers.module_collector import collect_trainable_modules


def extract_model_classes(text):
    """
    다양한 형태의 PyTorch 모델 선언을 robust하게 감지.
    - model = ClassName(...)
    - encoder = ClassName(...)
    - net = ClassName(...)
    - model = ClassName().to(...)
    - encoder = ClassName().cuda()
    - 클래스 이름은 대문자로 시작한다고 가정 (PyTorch 일반 관례)
    """
    import re
    
    candidates = []

    # 1) 가장 일반적인 형태:   variable = ClassName(...)
    pattern_basic = r"([a-zA-Z_][\w]*)\s*=\s*([A-Z][A-Za-z0-9_]*)\s*\("
    for var, cls in re.findall(pattern_basic, text):
        candidates.append((var, cls))

    # 2) 체이닝 포함:   var = ClassName(...).to(...)
    pattern_chain = r"([a-zA-Z_][\w]*)\s*=\s*([A-Z][A-Za-z0-9_]*)\s*\([^)]*\)\s*\."
    for var, cls in re.findall(pattern_chain, text):
        candidates.append((var, cls))

    # 중복 제거
    seen = set()
    final = []
    for v, c in candidates:
        if (v, c) not in seen:
            seen.add((v, c))
            final.append((v, c))

    return final


class MLCodeParser(ast.NodeVisitor):
    """
    V3: AST는 '값' + '학습 흐름 단서(training flow signals)'를 추출한다.
    Stage(pretrain/train/finetune) 판단은 하지 않으며,
    판단을 위한 신호를 LLM에 제공한다.
    """

    def __init__(self):
        self.summary = {
            "dataset": {},
            "model": {},
            "training": {
                "overall": {},
                "stages": {
                    "pretrain": {},
                    "train": {},
                    "finetune": {}
                }
            },
            "training_flow": {
                "has_training_loop": False,
                "loop_epoch_var": None,
                "has_backward": False,
                "has_optimizer_step": False,
                "has_model_train": False,
                "has_pretrained_load": False,
            }
        }

        self.variables = {}
        self.current_loop_vars = []   # for detecting training loop

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------
    def _resolve(self, value):
        if isinstance(value, ast.Constant):
            return value.value
        if isinstance(value, ast.Name):
            return self.variables.get(value.id)
        if isinstance(value, (int, float, str)):
            return value
        return None
    
    def visit_ClassDef(self, node):

        class_name = node.name
        is_model_class = False

        # 상속 구조 디버깅
        for base in node.bases:
            # try:
            #     # print(f"[DEBUG]   base: {ast.unparse(base)}")
            # except:
            #     pass

            if isinstance(base, ast.Attribute) and base.attr == "Module":
                # print("[DEBUG]   → inherits nn.Module via Attribute")
                is_model_class = True
            if isinstance(base, ast.Name) and base.id == "Module":
                # print("[DEBUG]   → inherits nn.Module via Name")
                is_model_class = True

        # __init__ 내부 확인
        for body_item in node.body:
            if isinstance(body_item, ast.FunctionDef) and body_item.name == "__init__":
                # print(f"[DEBUG]   Checking __init__ of {class_name}")
                try:
                    text = ast.unparse(body_item)
                    if any(x in text for x in ["nn.Conv", "nn.Linear", "nn.BatchNorm"]):
                        # print(f"[DEBUG]   → {class_name} has NN layers ⇒ model class detected")
                        is_model_class = True
                except:
                    pass

        if is_model_class:
            # print(f"[DEBUG] >>> MODEL CLASS DETECTED: {class_name}")
            self.summary["model"]["name"] = class_name

        self.generic_visit(node)



    # ---------------------------------------------------------
    # Variable assignment
    # ---------------------------------------------------------
    def visit_Assign(self, node):
        value = None

        # 기존 device 변수 처리
        if isinstance(node.value, ast.Constant):
            value = node.value.value
        elif isinstance(node.value, ast.Name):
            value = self.variables.get(node.value.id)
        elif isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Attribute):
                if node.value.func.attr == "device":
                    if node.value.args:
                        value = self._resolve(node.value.args[0])

        # 모델 할당 여부 출력
        if isinstance(node.value, ast.Call):
            try:
                call_repr = ast.unparse(node.value)
                # print(f"[DEBUG] Assign Call: {call_repr}")
            except:
                pass
        
        LOSS_CLASSES = {
            "CrossEntropyLoss",
            "MSELoss",
            "L1Loss",
            "SmoothL1Loss",
        }

        OPTIMIZER_CLASSES = {
            "Adam", "AdamW", "SGD", "RMSprop"
        }


        # 모델 생성 감지 (Name 기반)
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):

            class_name = node.value.func.id

            # Loss / Optimizer 는 model class 로 취급하면 안됨
            if class_name in LOSS_CLASSES:
                return
            if class_name in OPTIMIZER_CLASSES:
                return

            # print(f"[DEBUG] Model instantiation detected (Name): {class_name}")
            self.summary["model"]["name"] = class_name


        # 모델 생성 감지 (Attribute 기반)
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):

            class_name = node.value.func.attr

            if class_name in LOSS_CLASSES:
                return
            if class_name in OPTIMIZER_CLASSES:
                return

            # print(f"[DEBUG] Model instantiation detected (Attribute): {class_name}")
            self.summary["model"]["name"] = class_name
        

        self.generic_visit(node)


    # ---------------------------------------------------------
    # Call detection: optimizer, loss, backward, step, load weights
    # ---------------------------------------------------------
    def visit_Call(self, node):
        # ---------- Optimizer detection ----------
        try:
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in ["Adam", "AdamW", "SGD", "RMSprop"]:
                    self.summary["training"]["overall"]["optimizer"] = node.func.attr
        except:
            pass

        # ---------- Loss detection ----------
        try:
            if isinstance(node.func, ast.Name):
                if node.func.id in ["CrossEntropyLoss", "MSELoss", "L1Loss", "SmoothL1Loss"]:
                    self.summary["training"]["overall"]["loss"] = node.func.id
        except:
            pass

        # ---------- backward() detection ----------
        try:
            if isinstance(node.func, ast.Attribute) and node.func.attr == "backward":
                self.summary["training_flow"]["has_backward"] = True
        except:
            pass

        # ---------- optimizer.step() detection ----------
        try:
            if isinstance(node.func, ast.Attribute) and node.func.attr == "step":
                self.summary["training_flow"]["has_optimizer_step"] = True
        except:
            pass

        # ---------- model.train() detection ----------
        try:
            if isinstance(node.func, ast.Attribute) and node.func.attr == "train":
                self.summary["training_flow"]["has_model_train"] = True
        except:
            pass

        # ---------- pretrained model load detection ----------
        try:
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in ["load_state_dict", "load_weights"]:
                    self.summary["training_flow"]["has_pretrained_load"] = True
        except:
            pass

        # torch.load(...)
        try:
            if isinstance(node.func, ast.Attribute) and node.func.attr == "load":
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "torch":
                    self.summary["training_flow"]["has_pretrained_load"] = True
        except:
            pass

        self.generic_visit(node)

    # ---------------------------------------------------------
    # For loop detection (training loop)
    # ---------------------------------------------------------
    def visit_For(self, node):
        # Detect loops over range(EPOCHS)
        is_training_loop = False

        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
            if node.iter.func.id == "range":
                is_training_loop = True
                # Try extracting epoch variable
                if isinstance(node.target, ast.Name):
                    self.summary["training_flow"]["loop_epoch_var"] = node.target.id

        if is_training_loop:
            self.summary["training_flow"]["has_training_loop"] = True

        self.generic_visit(node)
    
    


    # ---------------------------------------------------------
    # Finalize: extract patterns like lr, batch, epochs, device
    # ---------------------------------------------------------
    def finalize(self):
        # print("[DEBUG] FINALIZING VARIABLES:")
        for k, v in self.variables.items():
            print(f"  {k} = {v}")

        overall = self.summary["training"]["overall"]

        # print("[DEBUG] Before finalize overall:", overall)

        # epochs
        for k, v in self.variables.items():
            if "epoch" in k.lower():
                # print(f"[DEBUG] epoch var detected: {k} = {v}")
                overall["epochs"] = v

        # lr
        for k, v in self.variables.items():
            if "lr" in k.lower() or "learning_rate" in k.lower():
                # print(f"[DEBUG] lr var detected: {k} = {v}")
                overall["learning_rate"] = v

        # batch
        for k, v in self.variables.items():
            if "batch" in k.lower() or "bs" in k.lower():
                # print(f"[DEBUG] batch var detected: {k} = {v}")
                overall["batch_size"] = v

        # device
        for k, v in self.variables.items():
            if "device" in k.lower():
                # print(f"[DEBUG] device var detected: {k} = {v}")
                overall["device"] = v

        # print("[DEBUG] After finalize overall:", overall)


    # ---------------------------------------------------------
    def parse(self, src):
        tree = ast.parse(src)
        self.visit(tree)

        debug_logs = []

        # =======================================================
        # 1) AST 기반 모델 정보 (MLCodeParser가 찾은 것)
        # =======================================================
        ast_model_name = self.summary.get("model", {}).get("class_name")
        debug_logs.append(f"[AST] detected model class: {ast_model_name}")

        # =======================================================
        # 2) 정규식 기반 탐색
        # =======================================================
        model_vars = extract_model_classes(src)
        debug_logs.append(f"[Regex] model_vars found: {model_vars}")

        if model_vars:
            var, cls = model_vars[-1]
            self.summary.setdefault("model", {})
            self.summary["model"]["class_name"] = cls
            self.summary["model"]["variable_name"] = var
            self.summary["model"]["all_detected_models"] = [
                {"var": v, "class": c} for v, c in model_vars
            ]
            debug_logs.append(f"[Regex] final chosen class: {cls}")

        else:
            debug_logs.append("[Regex] no model vars detected → fallback to LLM")

            # =======================================================
            # 3) LLM 기반 추출
            # =======================================================
            from backend.llm.model_name_extractor import llm_extract_model_class
            llm_res = llm_extract_model_class(src)
            debug_logs.append(f"[LLM] extraction result: {llm_res}")

            if llm_res.get("model_class"):
                self.summary.setdefault("model", {})
                self.summary["model"]["class_name"] = llm_res["model_class"]
                self.summary["model"]["llm_verified"] = True
                self.summary["model"]["llm_reason"] = llm_res.get("reason")
                debug_logs.append(f"[LLM] LLM final class: {llm_res['model_class']}")
            else:
                debug_logs.append("[LLM] no model extracted")

        # =======================================================
        # Save debug logs so Streamlit sees it
        # =======================================================
        self.summary["debug_model_parser"] = debug_logs

        modules = collect_trainable_modules(src)
        self.summary["model"]["trainable_modules"] = modules

        return self.summary


def summarize_code(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        src = f.read()

    parser = MLCodeParser()
    return parser.parse(src)
