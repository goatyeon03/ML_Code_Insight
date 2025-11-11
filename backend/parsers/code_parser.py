import ast, re

class MLCodeParser(ast.NodeVisitor):
    """
    ML 코드 요약기 (AST + 정규식 혼합 + 대소문자/약어 대응)
    """
    def __init__(self):
        self.summary = {
            "dataset": {},
            "model": {},
            "training": {},
            "misc": {}
        }

    # -------------------------------------------------
    # 변수 할당 탐색
    # -------------------------------------------------
    def visit_Assign(self, node):
        code = ast.unparse(node)

        # Optimizer
        opt_match = re.search(r"(?:torch\.)?optim\.(\w+)", code, re.IGNORECASE)
        if opt_match:
            self.summary["training"]["optimizer"] = opt_match.group(1)

        # Learning rate
        lr_match = re.findall(r"(?:lr|learning[_\s]?rate)\s*=\s*([\d\.eE-]+)", code, re.IGNORECASE)
        if lr_match:
            self.summary["training"]["learning_rate"] = lr_match[0]

        # Batch size
        bs_match = re.findall(r"batch[_\s]?size\s*=\s*(\d+)", code, re.IGNORECASE)
        if bs_match:
            self.summary["training"]["batch_size"] = bs_match[0]

        # Epochs
        ep_match = re.findall(r"(?:num[_\s]?epochs|epochs?)\s*=\s*(\d+)", code, re.IGNORECASE)
        if ep_match:
            self.summary["training"]["epochs"] = ep_match[0]

        # Loss
        loss_match = re.search(r"nn\.(\w+Loss)", code, re.IGNORECASE)
        if loss_match:
            self.summary["training"]["loss"] = loss_match.group(1)

        # Scheduler
        sch_match = re.search(r"(?:scheduler\.)?(\w+LR)\s*\(", code, re.IGNORECASE)
        if sch_match:
            self.summary["training"]["scheduler"] = sch_match.group(1)

        # Device
        dev_match = re.search(r"device\s*=\s*['\"]([\w:]+)['\"]", code, re.IGNORECASE)
        if dev_match:
            self.summary["training"]["device"] = dev_match.group(1)

        self.generic_visit(node)

    # -------------------------------------------------
    # 모델 클래스 정의
    # -------------------------------------------------
    def visit_ClassDef(self, node):
        base_names = []
        for b in node.bases:
            if isinstance(b, ast.Name):
                base_names.append(b.id)
            elif isinstance(b, ast.Attribute):
                base_names.append(b.attr)
        if any("Module" in name for name in base_names):
            self.summary["model"]["class_name"] = node.name
        self.generic_visit(node)

    # -------------------------------------------------
    # Dataset / DataLoader 감지
    # -------------------------------------------------
    def visit_Call(self, node):
        fn = ast.unparse(node.func)
        if "DataLoader" in fn:
            self.summary["dataset"]["loader"] = fn
        if "Dataset" in fn:
            self.summary["dataset"]["dataset_class"] = fn
        self.generic_visit(node)


def summarize_code(filepath: str):
    """코드 파일 파싱 + 요약 반환"""
    with open(filepath, "r", encoding="utf-8") as f:
        src = f.read()

    try:
        tree = ast.parse(src)
        parser = MLCodeParser()
        parser.visit(tree)

        # 텍스트 기반 보조 탐색
        text = src

        if "optimizer" not in parser.summary["training"]:
            opt_match = re.search(r"(?:torch\.)?optim\.(\w+)", text, re.IGNORECASE)
            if opt_match:
                parser.summary["training"]["optimizer"] = opt_match.group(1)

        if "learning_rate" not in parser.summary["training"]:
            lr_match = re.findall(r"(?:lr|learning[_\s]?rate)\s*=\s*([\d\.eE-]+)", text, re.IGNORECASE)
            if lr_match:
                parser.summary["training"]["learning_rate"] = lr_match[0]

        if "epochs" not in parser.summary["training"]:
            ep_match = re.findall(r"(?:num[_\s]?epochs|epochs?)\s*=\s*(\d+)", text, re.IGNORECASE)
            if ep_match:
                parser.summary["training"]["epochs"] = ep_match[0]

        # Debug log
        print(f"\n[DEBUG] Parsed summary for {filepath}:")
        for sec, vals in parser.summary.items():
            print(f"  {sec}: {vals}")
        print("--------------------------------------------------")

        return parser.summary

    except Exception as e:
        print(f"[ERROR] summarize_code failed: {e}")
        return {"error": str(e)}
