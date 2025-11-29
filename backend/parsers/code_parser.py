import ast
import re

class RobustMLParser(ast.NodeVisitor):

    def __init__(self):
        # 최종 summary
        self.summary = {
            "model": {},
            "training": {
                "pretrained": False
            },
            "dataset": {},
            "misc": {}
        }

        # AST 기반 후보 저장
        self.pretrain_epochs = []
        self.finetune_epochs = []
        self.pretrain_bs = []
        self.finetune_bs = []
        self.pretrain_lr = []
        self.finetune_lr = []
        self.pretrain_loss = []
        self.finetune_loss = []

    # =======================================
    # Model Class
    # =======================================
    def visit_ClassDef(self, node):
        try:
            for base in node.bases:
                base_name = getattr(base, "id", None) or getattr(base, "attr", None)
                if base_name == "Module":
                    self.summary["model"]["class_name"] = node.name
        except:
            pass
        self.generic_visit(node)

    


    # =======================================
    # AST 기반 기본 변수 추출
    # =======================================
    def visit_Assign(self, node):
        try:
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if not targets:
                return

            # literal만 처리
            if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float, str))):
                return

            val = node.value.value

            for name in targets:
                lname = name.lower()

                # --- helper: SSL / pretrain 변수 감지 ---
                def _is_pretrain_var(name: str):
                    name_up = name.upper()
                    name_low = name.lower()
                    return (
                        name_up.endswith("_SSL")
                        or "ssl" in name_low
                        or "pretrain" in name_low
                        or "contrastive" in name_low
                    )

                if "epoch" in lname:
                    if _is_pretrain_var(name):
                        self.pretrain_epochs.append(val)
                    else:
                        self.finetune_epochs.append(val)

                if ("batch" in lname) or lname.startswith("bs") or lname.endswith("bs"):
                    if _is_pretrain_var(name):
                        self.pretrain_bs.append(val)
                    else:
                        self.finetune_bs.append(val)

                if "lr" in lname or "learning" in lname:
                    if isinstance(val, (int, float)):
                        if _is_pretrain_var(name):
                            self.pretrain_lr.append(val)
                        else:
                            self.finetune_lr.append(val)


                # -------- Loss 분리 -------
                if "loss" in lname or "criterion" in lname:
                    if isinstance(val, str):
                        if ("pretrain" in lname) or ("ssl" in lname):
                            self.pretrain_loss.append(val)
                        else:
                            self.finetune_loss.append(val)

        except:
            pass

        self.generic_visit(node)

    # =======================================
    # Dataset detection
    # =======================================
    def visit_Call(self, node):
        try:
            fn = ast.unparse(node.func)
            if "DataLoader" in fn:
                self.summary["dataset"]["loader"] = fn
            if "Dataset" in fn:
                self.summary["dataset"]["dataset_class"] = fn
        except:
            pass

        self.generic_visit(node)


# ============================================================
# Regex 기반 하이퍼파라미터 추출 
# (대문자/약어/camelCase/prefix/suffix 모두 감지)
# ============================================================
def _regex_extract(text, summary):

    # --------------------
    # Learning rate
    # --------------------
    lr_regex = r"(?i)\b[a-zA-Z0-9_\.]*lr[a-zA-Z0-9_]*\s*=\s*([0-9\.eE-]+)"
    for m in re.finditer(lr_regex, text):
        summary["training"].setdefault("_lr_candidates", []).append(m.group(1))

    # --------------------
    # Batch size
    # --------------------
    batch_patterns = [
        r"(?i)\b[a-zA-Z0-9_\.]*batch[a-zA-Z0-9_]*\s*=\s*(\d+)",
        r"(?i)\b[a-zA-Z0-9_\.]*bs[a-zA-Z0-9_]*\s*=\s*(\d+)",
    ]
    for pat in batch_patterns:
        for m in re.finditer(pat, text):
            summary["training"].setdefault("_bs_candidates", []).append(int(m.group(1)))

    # --------------------
    # Loss
    # --------------------
    m = re.search(r"(?i)nn\.(\w+Loss)", text)
    if m:
        summary["training"]["loss"] = m.group(1)

    crit = re.search(r"(?i)\b(loss|criterion|loss_fn)\b\s*=\s*(\w+)", text)
    if crit:
        summary["training"]["loss"] = crit.group(2)

    # --------------------
    # Optimizer
    # --------------------
    m = re.search(r"(?i)(?:torch\.)?optim\.(\w+)", text)
    if m:
        summary["training"]["optimizer"] = m.group(1)

    # --------------------
    # Device
    # --------------------
    m = re.search(r"(?i)device\s*=\s*['\"]([\w:]+)['\"]", text)
    if m:
        summary["training"]["device"] = m.group(1)

    m = re.search(r"\.to\(['\"](cuda.*?|cpu)['\"]\)", text)
    if m:
        summary["training"]["device"] = m.group(1)

    # --------------------
    # Pretrained 여부
    # --------------------
    pretrained_patterns = [
        r"load_state_dict",
        r"from_pretrained",
        r"torch\.load",
        r"load_weights",
        r"load_pretrained",
        r"\bpretrain\b",
        r"ssl_train",
        r"run_ssl",
        r"ssl_"
    ]
    for p in pretrained_patterns:
        if re.search(p, text):
            summary["training"]["pretrained"] = True
            break

    return summary


# ============================================================
# Public API
# ============================================================
def summarize_code(filepath: str):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception as e:
        return {"error": f"Failed to read file: {e}"}

    parser = RobustMLParser()

    # AST 파싱
    try:
        tree = ast.parse(src)
        parser.visit(tree)
    except:
        pass

    summary = parser.summary
    summary = _regex_extract(src, summary)

    T = summary["training"]

    # ---------------------------
    # Epochs (two-phase handling)
    # ---------------------------
    if T["pretrained"]:
        if parser.pretrain_epochs:
            T["pretrain_epochs"] = parser.pretrain_epochs[-1]
        if parser.finetune_epochs:
            T["finetune_epochs"] = parser.finetune_epochs[-1]
    else:
        if parser.finetune_epochs:
            T["epochs"] = parser.finetune_epochs[-1]

    # ---------------------------
    # Batch size (two-phase)
    # ---------------------------
    if T["pretrained"]:
        if parser.pretrain_bs:
            T["pretrain_batch_size"] = parser.pretrain_bs[-1]
        if parser.finetune_bs:
            T["finetune_batch_size"] = parser.finetune_bs[-1]
    else:
        if parser.finetune_bs:
            T["batch_size"] = parser.finetune_bs[-1]

    # Regex batch도 fallback
    bs_candidates = T.get("_bs_candidates")
    if bs_candidates and "batch_size" not in T and not T["pretrained"]:
        T["batch_size"] = bs_candidates[-1]

    # ---------------------------
    # Learning rate (two-phase)
    # ---------------------------
    if T["pretrained"]:
        if parser.pretrain_lr:
            T["pretrain_learning_rate"] = parser.pretrain_lr[-1]
        if parser.finetune_lr:
            T["finetune_learning_rate"] = parser.finetune_lr[-1]
    else:
        if parser.finetune_lr:
            T["learning_rate"] = parser.finetune_lr[-1]

    lr_candidates = T.get("_lr_candidates")
    if lr_candidates and "learning_rate" not in T and not T["pretrained"]:
        T["learning_rate"] = lr_candidates[-1]

    # ---------------------------
    # Loss (two-phase)
    # ---------------------------
    if T["pretrained"]:
        if parser.pretrain_loss:
            T["pretrain_loss"] = parser.pretrain_loss[-1]
        if parser.finetune_loss:
            T["finetune_loss"] = parser.finetune_loss[-1]
    else:
        if parser.finetune_loss:
            T["loss"] = parser.finetune_loss[-1]
    
    # If SSL code detected, assign contrastive loss to pretrain
    if "contrastive_loss" in src or "nt_xent" in src:
        summary["training"]["pretrain_loss"] = "contrastive_loss"


    # cleanup temporary fields
    T.pop("_bs_candidates", None)
    T.pop("_lr_candidates", None)

    return summary
