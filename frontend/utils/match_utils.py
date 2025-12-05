import re
from collections import defaultdict


def normalize_name(name):
    base = re.sub(r"\.[^.]+$", "", name.lower())
    base = re.sub(r"[_\-\.\s]+", " ", base).strip()
    drop = {"train", "training", "result", "results",
            "metric", "metrics", "log", "logs"}
    tokens = [t for t in base.split() if t not in drop]
    return " ".join(tokens)


def tokens(name):
    return set(normalize_name(name).split())


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def longest_common_prefix(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def match_code_and_results(code_files, result_files):
    code_tok = {c: tokens(c) for c in code_files}
    res_tok = {r: tokens(r) for r in result_files}
    pairs = defaultdict(list)

    for r in result_files:
        best_code, best_score = None, -1
        for c in code_files:
            score_j = jaccard(code_tok[c], res_tok[r])
            lcp = longest_common_prefix(normalize_name(c), normalize_name(r))
            score = score_j + (lcp / 20.0)
            if score > best_score:
                best_code, best_score = c, score
        if best_code and (best_score >= 0.3 or lcp >= 4):
            pairs[best_code].append(r)

    for k, v in pairs.items():
        pairs[k] = sorted(set(v))
    return pairs
