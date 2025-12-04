import re

def collect_trainable_modules(src_text: str):
    module_map = {}

    # ------------------------------------------------------
    # 1) parameters() 기반 탐지 (멀티라인 포함)
    # ------------------------------------------------------
    param_pattern = re.compile(r"([a-zA-Z_]\w*)\s*\.parameters\s*\(", re.MULTILINE)
    for var in param_pattern.findall(src_text):
        module_map.setdefault(var, None)

    # ------------------------------------------------------
    # 2) train()/eval() 기반 탐지
    # ------------------------------------------------------
    train_pattern = re.compile(r"([a-zA-Z_]\w*)\s*\.\s*(train|eval)\s*\(", re.MULTILINE)
    for var, _ in train_pattern.findall(src_text):
        module_map.setdefault(var, None)

    # ------------------------------------------------------
    # 3) 객체 생성 패턴 강화
    #    encoder = CBraModEncoder().to(DEVICE) 같은 것도 잡기
    # ------------------------------------------------------
    init_pattern = re.compile(
        r"([a-zA-Z_]\w*)\s*=\s*([A-Z][A-Za-z0-9_]*)\s*\(",
        re.MULTILINE,
    )
    for var, cls in init_pattern.findall(src_text):
        module_map[var] = cls  # class name은 확실히 저장

    # ------------------------------------------------------
    # 4) Dataset / DataLoader 계열 제거
    #    - 클래스 이름이 *Dataset, *Loader 인 것
    #    - 변수 이름에 dataset / loader 가 들어간 것
    # ------------------------------------------------------
    cleaned = {}
    dataset_suffix = re.compile(r"Dataset$", re.IGNORECASE)
    loader_suffix = re.compile(r"Loader$", re.IGNORECASE)

    for var, cls in module_map.items():
        lower_var = var.lower()

        # var 이름으로 필터링
        if "dataset" in lower_var or "loader" in lower_var:
            continue

        if cls is not None:
            # 정확 이름
            if cls in ("Dataset", "DataLoader"):
                continue
            # 접미사 기반
            if dataset_suffix.search(cls) or loader_suffix.search(cls):
                continue

        cleaned[var] = cls

    return cleaned
