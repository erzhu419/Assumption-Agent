"""Exp 82 step 2: extract features per (wisdom, problem).

Wisdom features (per cid):
  - w_cid_X: one-hot for which candidate (12 dims) — captures "wisdom identity"
  - w_authority_*: classical / modern / proverb / scientific
  - w_pattern_*: regex flags for which methodological pattern the
                 aphorism mentions (decompose, constraint, perspective,
                 estimate, verify, prevention, priority, perspective)
  - w_aphorism_len: char count
  - w_signal_len: char count of the signal field

Problem features (per pid):
  - p_domain_*: one-hot for problem domain (business / engineering /
                software / mathematics / daily_life / science / chemistry / ...)
  - p_has_numbers: 0/1
  - p_length_chars: char count
  - p_n_constraints: count of explicit-constraint markers

Cross / interaction terms (auto-generated):
  - all w_pattern_X * p_domain_Y crosses
  - w_aphorism_len_long (>40 chars) * p_length_long (>200 chars)
  - All authority * domain crosses

Output: phase six/exp82/features.json
"""
import json
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
IN_MATRIX = Path(__file__).parent / "verdict_matrix.json"
OUT_FEATURES = Path(__file__).parent / "features.json"


# Wisdom pattern detectors — each matches keywords in (aphorism + signal + unpacked)
WISDOM_PATTERNS = {
    "pattern_decompose": [
        r"分解", r"拆", r"步骤", r"逐", r"分而治之", r"divide", r"step",
    ],
    "pattern_constraint": [
        r"约束", r"限制", r"边界", r"前提", r"条件", r"规", r"瓶颈",
        r"constraint", r"limit", r"bound",
    ],
    "pattern_perspective": [
        r"兼听", r"多角度", r"反思", r"全局", r"立场", r"视角",
        r"perspective", r"holistic", r"viewpoint",
    ],
    "pattern_estimate": [
        r"估算", r"量级", r"大致", r"approxi", r"order", r"estimate",
        r"quick", r"rough",
    ],
    "pattern_verify": [
        r"验证", r"核", r"检验", r"考察", r"试", r"verif", r"check",
        r"test", r"validate", r"audit",
    ],
    "pattern_prevention": [
        r"预防", r"未病", r"防患", r"先发", r"prevent", r"early",
    ],
    "pattern_priority": [
        r"主次", r"重点", r"关键", r"重要", r"先后", r"主要",
        r"priority", r"key", r"critical",
    ],
    "pattern_simplify": [
        r"简", r"奥卡姆", r"剃刀", r"复杂", r"simpli", r"occam",
    ],
    "pattern_iterate": [
        r"迭代", r"反复", r"循环", r"itera", r"loop",
    ],
    "pattern_pattern_match": [
        r"类比", r"先例", r"过往", r"历史", r"模式", r"重现",
        r"pattern", r"analog", r"precedent",
    ],
}


# Authority classifier — based on source string
def classify_authority(source):
    s = (source or "").lower()
    # Classical pre-modern
    classical_kws = ["素问", "黄帝", "道德经", "论语", "庄子", "孟子", "孙子",
                       "周易", "易经", "诗经", "大学", "中庸", "韩非",
                       "confucius", "lao", "sun tzu", "i ching"]
    if any(kw.lower() in s for kw in classical_kws):
        return "classical"
    # Western classical pre-1900
    western_classical = ["aristotle", "plato", "socrates", "marcus", "stoic",
                           "epicurus", "machiavelli", "亚里士多德", "柏拉图"]
    if any(kw in s for kw in western_classical):
        return "classical"
    # Modern (20th-21st century thinkers)
    modern_kws = ["drucker", "munger", "buffett", "feynman", "polya",
                    "einstein", "popper", "kuhn", "kahneman", "taleb",
                    "彼得·德鲁克", "查理·芒格", "巴菲特", "费曼", "波利亚",
                    "毛泽东", "鲁迅", "陈丹然"]
    if any(kw.lower() in s for kw in modern_kws):
        return "modern"
    # Proverb / folk
    proverb_kws = ["民间", "谚语", "俗语", "俗话", "老话", "谚",
                     "proverb", "folk"]
    if any(kw in s for kw in proverb_kws):
        return "proverb"
    # Religious / biblical
    religious_kws = ["圣经", "传道书", "论语", "佛", "道", "禅",
                       "bible", "ecclesiastes"]
    if any(kw.lower() in s for kw in religious_kws):
        return "religious"
    return "other"


def extract_wisdom_features(cand):
    """Returns dict of feature_name -> 0/1 or numeric."""
    text = " ".join([
        cand.get("aphorism", ""),
        cand.get("signal", ""),
        cand.get("unpacked", ""),
    ])
    feat = {}
    # patterns
    for pat_name, kws in WISDOM_PATTERNS.items():
        feat[f"w_{pat_name}"] = int(any(re.search(kw, text) for kw in kws))
    # authority
    auth = classify_authority(cand.get("source", ""))
    for label in ["classical", "modern", "proverb", "religious", "other"]:
        feat[f"w_authority_{label}"] = int(auth == label)
    # length features
    feat["w_aphorism_len"] = len(cand.get("aphorism", ""))
    feat["w_signal_len"] = len(cand.get("signal", ""))
    feat["w_unpacked_len"] = len(cand.get("unpacked", ""))
    # cid one-hot
    cid = cand.get("cid", "")
    for i in range(1, 12):
        feat[f"w_cid_WCAND{i:02d}"] = int(cid == f"WCAND{i:02d}")
    feat["w_cid_WCROSSL01"] = int(cid == "WCROSSL01")
    return feat


def extract_problem_features(prob):
    """Returns dict per problem."""
    desc = prob.get("description", "") or ""
    feat = {}
    # domain one-hot — pid prefix often encodes domain
    domain_str = prob.get("domain", "") or prob.get("pid", "").split("_")[0]
    for d in ["business", "engineering", "software", "software_engineering",
                "mathematics", "math", "daily_life", "science", "chemistry",
                "physics", "biology", "art"]:
        feat[f"p_domain_{d}"] = int(domain_str.startswith(d))
    # other features
    feat["p_has_numbers"] = int(bool(re.search(r"\d", desc)))
    feat["p_length_chars"] = len(desc)
    feat["p_length_long"] = int(len(desc) > 200)
    feat["p_length_short"] = int(len(desc) < 80)
    feat["p_n_constraints"] = sum(1 for kw in ["必须", "需要", "应当", "限制", "must", "should", "需"] if kw in desc)
    feat["p_has_question"] = int("?" in desc or "？" in desc)
    feat["p_n_steps"] = sum(1 for kw in ["首先", "然后", "接着", "最后", "first", "then", "next", "finally"] if kw in desc.lower())
    return feat


def make_cross_features(w_feat, p_feat):
    """Auto-generate cross / interaction features.
    Limit to: pattern_X × domain_Y (semantically meaningful) + authority × domain.
    """
    crosses = {}
    pattern_names = [k for k in w_feat if k.startswith("w_pattern_")]
    domain_names = [k for k in p_feat if k.startswith("p_domain_") and p_feat[k] == 1]
    auth_names = [k for k in w_feat if k.startswith("w_authority_") and w_feat[k] == 1]
    for pname in pattern_names:
        for dname in domain_names:
            crosses[f"x_{pname[2:]}_X_{dname[2:]}"] = w_feat[pname] * p_feat[dname]
    for aname in auth_names:
        for dname in domain_names:
            crosses[f"x_{aname[2:]}_X_{dname[2:]}"] = w_feat[aname] * p_feat[dname]
    # length crosses
    crosses["x_long_aphorism_long_problem"] = int(w_feat["w_aphorism_len"] > 40) * p_feat["p_length_long"]
    return crosses


def main():
    print("Loading verdict matrix...", flush=True)
    data = json.loads(IN_MATRIX.read_text(encoding="utf-8"))
    cands = data["candidates"]
    problems = data["problems"]
    matrix = data["matrix"]
    print(f"  {len(cands)} candidates, {len(problems)} problems", flush=True)

    # Build wisdom and problem feature dicts
    cid_feat = {c["cid"]: extract_wisdom_features(c) for c in cands}
    pid_feat = {p["pid"]: extract_problem_features(p) for p in problems}

    # Print summary of features
    sample_w = cid_feat[cands[0]["cid"]]
    sample_p = pid_feat[problems[0]["pid"]]
    print(f"\n  Wisdom features ({len(sample_w)} dims):", flush=True)
    print(f"    {list(sample_w.keys())[:8]}...", flush=True)
    print(f"  Problem features ({len(sample_p)} dims):", flush=True)
    print(f"    {list(sample_p.keys())[:8]}...", flush=True)

    # Per-pattern coverage across wisdoms
    print(f"\n  Wisdom pattern coverage:", flush=True)
    for pat in [k for k in sample_w if k.startswith("w_pattern_")]:
        n = sum(1 for cid, f in cid_feat.items() if f.get(pat, 0))
        print(f"    {pat}: {n}/{len(cands)} wisdoms", flush=True)
    print(f"\n  Wisdom authority distribution:", flush=True)
    for auth in [k for k in sample_w if k.startswith("w_authority_")]:
        n = sum(1 for cid, f in cid_feat.items() if f.get(auth, 0))
        print(f"    {auth}: {n}/{len(cands)} wisdoms", flush=True)

    # Per-problem domain
    print(f"\n  Problem domain distribution:", flush=True)
    for d in [k for k in sample_p if k.startswith("p_domain_")]:
        n = sum(1 for pid, f in pid_feat.items() if f.get(d, 0))
        if n > 0:
            print(f"    {d}: {n}/{len(problems)} problems", flush=True)

    # Build full per-(W, P) feature row, with crosses
    rows = []  # list of {"cid", "pid", "judge", "y", **features}
    for pid in matrix:
        if pid not in pid_feat:
            continue
        for cid in matrix[pid]:
            if cid not in cid_feat:
                continue
            for judge, verdict in matrix[pid][cid].items():
                if verdict not in ("ext", "base", "tie"):
                    continue
                w = cid_feat[cid]
                p = pid_feat[pid]
                cx = make_cross_features(w, p)
                row = {"cid": cid, "pid": pid, "judge": judge,
                         "verdict": verdict,
                         "y_ext": int(verdict == "ext"),
                         "y_decided": int(verdict in ("ext", "base"))}
                row.update(w)
                row.update(p)
                row.update(cx)
                rows.append(row)

    print(f"\n  Total (W, P, J) rows: {len(rows)}", flush=True)

    # Drop tie rows for binary classification (we keep them in matrix for ref)
    decided = [r for r in rows if r["verdict"] in ("ext","base")]
    print(f"  Decided (ext/base, no ties): {len(decided)}", flush=True)
    n_ext = sum(r["y_ext"] for r in decided)
    print(f"  ext={n_ext}, base={len(decided)-n_ext}, ext rate={n_ext/len(decided):.1%}", flush=True)

    OUT_FEATURES.write_text(json.dumps({
        "n_rows": len(rows),
        "n_decided": len(decided),
        "rows": rows,
    }, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_FEATURES.relative_to(PROJECT)}", flush=True)


if __name__ == "__main__":
    main()
