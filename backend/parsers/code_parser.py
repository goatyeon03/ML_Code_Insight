import ast


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
        print(f"[DEBUG] Found class: {node.name}")

        class_name = node.name
        is_model_class = False

        # 상속 구조 디버깅
        for base in node.bases:
            try:
                print(f"[DEBUG]   base: {ast.unparse(base)}")
            except:
                pass

            if isinstance(base, ast.Attribute) and base.attr == "Module":
                print("[DEBUG]   → inherits nn.Module via Attribute")
                is_model_class = True
            if isinstance(base, ast.Name) and base.id == "Module":
                print("[DEBUG]   → inherits nn.Module via Name")
                is_model_class = True

        # __init__ 내부 확인
        for body_item in node.body:
            if isinstance(body_item, ast.FunctionDef) and body_item.name == "__init__":
                print(f"[DEBUG]   Checking __init__ of {class_name}")
                try:
                    text = ast.unparse(body_item)
                    if any(x in text for x in ["nn.Conv", "nn.Linear", "nn.BatchNorm"]):
                        print(f"[DEBUG]   → {class_name} has NN layers ⇒ model class detected")
                        is_model_class = True
                except:
                    pass

        if is_model_class:
            print(f"[DEBUG] >>> MODEL CLASS DETECTED: {class_name}")
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
                print(f"[DEBUG] Assign Call: {call_repr}")
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

            print(f"[DEBUG] Model instantiation detected (Name): {class_name}")
            self.summary["model"]["name"] = class_name


        # 모델 생성 감지 (Attribute 기반)
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):

            class_name = node.value.func.attr

            if class_name in LOSS_CLASSES:
                return
            if class_name in OPTIMIZER_CLASSES:
                return

            print(f"[DEBUG] Model instantiation detected (Attribute): {class_name}")
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
        print("[DEBUG] FINALIZING VARIABLES:")
        for k, v in self.variables.items():
            print(f"  {k} = {v}")

        overall = self.summary["training"]["overall"]

        print("[DEBUG] Before finalize overall:", overall)

        # epochs
        for k, v in self.variables.items():
            if "epoch" in k.lower():
                print(f"[DEBUG] epoch var detected: {k} = {v}")
                overall["epochs"] = v

        # lr
        for k, v in self.variables.items():
            if "lr" in k.lower() or "learning_rate" in k.lower():
                print(f"[DEBUG] lr var detected: {k} = {v}")
                overall["learning_rate"] = v

        # batch
        for k, v in self.variables.items():
            if "batch" in k.lower() or "bs" in k.lower():
                print(f"[DEBUG] batch var detected: {k} = {v}")
                overall["batch_size"] = v

        # device
        for k, v in self.variables.items():
            if "device" in k.lower():
                print(f"[DEBUG] device var detected: {k} = {v}")
                overall["device"] = v

        print("[DEBUG] After finalize overall:", overall)


    # ---------------------------------------------------------
    def parse(self, source):
        tree = ast.parse(source)
        self.visit(tree)
        self.finalize()
        print("[DEBUG] FINAL SUMMARY:", self.summary)
        return self.summary


def summarize_code(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        src = f.read()

    parser = MLCodeParser()
    return parser.parse(src)
