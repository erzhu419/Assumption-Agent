"""
Cached A/B framework for Self-Discover ablation series.

Key design: separate answer generation from judging.

  - Answers are cached per (variant_name, problem_id).
  - Judgments are cached per (variant_a, variant_b, problem_id, swap_seed).

Once baseline + vanilla-39 + ours-27 are cached, all future variant
comparisons (phase2, phase3, phase4) only need to generate the NEW variant's
answers and judge against any existing cached variant. Re-runs of the same
(variant, problem) pair are free.

CLI:
  python cached_framework.py --variant baseline --n 100
  python cached_framework.py --variant vanilla_39 --n 100
  python cached_framework.py --variant ours_27 --n 100
  python cached_framework.py --judge vanilla_39 baseline --n 100
  python cached_framework.py --judge ours_27 baseline --n 100
  python cached_framework.py --judge ours_27 vanilla_39 --n 100
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "phase zero" / "scripts"))

import _config as cfg
from llm_client import create_client, parse_json_from_llm


def _generate_with_retry(client, prompt: str, max_tokens: int = 800,
                         temperature: float = 0.3, max_retries: int = 6) -> Dict:
    """Wrap client.generate with exponential backoff on 503/429 errors."""
    for attempt in range(max_retries):
        try:
            return client.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:
            msg = str(e)
            low = msg.lower()
            is_transient = ("503" in msg or "429" in msg or "500" in msg
                           or "UNAVAILABLE" in msg or "RESOURCE_EXHAUSTED" in msg
                           or "overloaded" in low or "high demand" in low
                           or "disconnect" in low or "remoteprotocol" in low
                           or "timeout" in low or "connection" in low)
            if not is_transient or attempt == max_retries - 1:
                raise
            wait = min(2 ** attempt * 5, 120)  # 5, 10, 20, 40, 80, 120s
            print(f"    [retry {attempt+1}/{max_retries}] {msg[:80]}... waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


CACHE_ROOT = PROJECT.parent / "phase two" / "analysis" / "cache"
ANSWERS_DIR = CACHE_ROOT / "answers"
STRUCTURES_DIR = CACHE_ROOT / "structures"
JUDGMENTS_DIR = CACHE_ROOT / "judgments"
SAMPLES_PATH = CACHE_ROOT / "sample_100.json"
JUDGE_CONTENT_CACHE_PATH = CACHE_ROOT / "judge_content_cache.json"
for d in [ANSWERS_DIR, STRUCTURES_DIR, JUDGMENTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# Content-hash judge cache: key = sha256(problem | answer_a | answer_b) → verdict.
# Survives across pair-names (judgments/{a}_vs_{b}.json) so the same answer
# pair judged under a different pair name returns cached.
_CONTENT_CACHE = None


def _load_content_cache():
    global _CONTENT_CACHE
    if _CONTENT_CACHE is not None:
        return _CONTENT_CACHE
    if JUDGE_CONTENT_CACHE_PATH.exists():
        try:
            _CONTENT_CACHE = json.loads(JUDGE_CONTENT_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            _CONTENT_CACHE = {}
    else:
        _CONTENT_CACHE = {}
    return _CONTENT_CACHE


def _save_content_cache():
    if _CONTENT_CACHE is not None:
        JUDGE_CONTENT_CACHE_PATH.write_text(
            json.dumps(_CONTENT_CACHE, ensure_ascii=False), encoding="utf-8")


def _content_hash(problem: str, a: str, b: str) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(problem.encode("utf-8"))
    h.update(b"\x00")
    h.update(a.encode("utf-8"))
    h.update(b"\x00")
    h.update(b.encode("utf-8"))
    return h.hexdigest()


# ========================================================================
# Prompts
# ========================================================================

SELECT_PROMPT = """# 你是方法论选择专家。

## 候选推理模块
{modules}

## 任务类别
领域: {domain}
难度: {difficulty}

## 该类别的 3 个代表性问题
{examples}

## 你的任务
从模块中选出 **3-5 个**最可能帮助解决这类问题的模块。

输出 JSON（不要代码块）：
{{"selected_ids": ["模块ID1", ...], "reasoning": "简短理由"}}
"""

ADAPT_PROMPT = """# 你是方法论改写专家。

## 任务类别
领域: {domain}
难度: {difficulty}

## 该类别的 3 个代表性问题
{examples}

## 被选中的推理模块
{selected_text}

## 你的任务
把每个模块改写为针对该任务类别的具体说法。每条 30-50 字。

输出 JSON：
{{"adapted": [{{"id": "模块ID", "adapted_description": "..."}}]}}
"""

IMPLEMENT_PROMPT = """# 你是推理结构设计专家。

## 任务类别
领域: {domain}
难度: {difficulty}

## 该类别的 3 个代表性问题
{examples}

## 改写后的推理模块
{adapted_text}

## 你的任务
组装成 JSON 推理结构：key 是可填空的步骤名（如"核心矛盾识别"、"关键假设列表"、"final_answer"），value 留空字符串。3-6 个 key，最后必须是 "final_answer"。

输出 JSON（不要代码块，不要解释）：
"""

EXECUTE_PROMPT = """# 解决下面的问题，按给定 JSON 结构填空。

## 问题
{problem}

## 推理结构
{structure}

## 要求
1. 按 JSON key 顺序填写每个 value
2. 每个 value 有实质内容
3. "final_answer" 必须是直接可操作的答案
4. 输出完整 JSON

输出 JSON：
"""

BASELINE_PROMPT = """你是一位严谨的问题解决者。针对下面的问题，给出一个清晰、结构化的解决方案。

要求：
1. 先简要重述问题核心
2. 给出你的分析和推理步骤
3. 给出最终建议/解答
4. 语言精炼，不超过 400 字

## 问题
{problem}
"""

JUDGE_PROMPT = """你是方法论评审专家。下面是同一个问题的两个解答。

## 问题
{problem}

## 解答 A
{answer_a}

## 解答 B
{answer_b}

## 评审
四个维度：问题理解、分析深度、结构化程度、实用性。

输出 JSON（不要代码块）：
{{"winner": "A"或"B"或"tie", "score_a": 1-10整数, "score_b": 1-10整数,
  "reasoning": "80字内说明胜因"}}
"""


# Vanilla Self-Discover's 39 paper modules (English as in paper)
VANILLA_39_MODULES = [
    ("M01", "How could I devise an experiment to help solve that problem?"),
    ("M02", "Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made."),
    ("M03", "How could I measure progress on this problem?"),
    ("M04", "How can I simplify the problem so that it is easier to solve?"),
    ("M05", "What are the key assumptions underlying this problem?"),
    ("M06", "What are the potential risks and drawbacks of each solution?"),
    ("M07", "What are the alternative perspectives or viewpoints on this problem?"),
    ("M08", "What are the long-term implications of this problem and its solutions?"),
    ("M09", "How can I break down this problem into smaller, more manageable parts?"),
    ("M10", "Critical Thinking: analyze from different perspectives, question assumptions, evaluate evidence."),
    ("M11", "Try creative thinking: generate innovative, out-of-the-box ideas."),
    ("M12", "Seek input and collaboration from others to solve the problem."),
    ("M13", "Systems thinking: consider the problem as part of a larger system."),
    ("M14", "Risk Analysis: evaluate potential risks, uncertainties, tradeoffs."),
    ("M15", "Reflective Thinking: step back, introspect, examine personal biases."),
    ("M16", "What is the core issue or problem that needs to be addressed?"),
    ("M17", "What are the underlying causes or factors contributing to the problem?"),
    ("M18", "Are there potential solutions or strategies that have been tried before? outcomes and lessons?"),
    ("M19", "What are the potential obstacles or challenges that might arise in solving this problem?"),
    ("M20", "Are there any relevant data or information that can provide insights?"),
    ("M21", "Are there stakeholders or individuals directly affected? their perspectives and needs?"),
    ("M22", "What resources (financial, human, technological) are needed to tackle the problem?"),
    ("M23", "How can progress or success in solving the problem be measured or evaluated?"),
    ("M24", "What indicators or metrics can be used?"),
    ("M25", "Is the problem technical/practical requiring specific expertise, or conceptual/theoretical?"),
    ("M26", "Does the problem involve a physical constraint (limited resources, infrastructure, space)?"),
    ("M27", "Is the problem related to human behavior — social, cultural, or psychological?"),
    ("M28", "Does the problem involve decision-making or planning under uncertainty?"),
    ("M29", "Is the problem analytical — requiring data analysis, modeling, optimization?"),
    ("M30", "Is the problem a design challenge requiring creative solutions and innovation?"),
    ("M31", "Does the problem require addressing systemic or structural issues rather than individual instances?"),
    ("M32", "Is the problem time-sensitive or urgent, requiring immediate action?"),
    ("M33", "What kinds of solution typically are produced for this kind of problem specification?"),
    ("M34", "Given the problem specification and the current best solution, guess about other possible solutions."),
    ("M35", "Imagine the current best solution is totally wrong — what other ways to think about the problem?"),
    ("M36", "What is the best way to modify this current best solution?"),
    ("M37", "Ignoring the current best solution, create an entirely new solution to the problem."),
    ("M38", "Let's think step by step."),
    ("M39", "Let's make a step by step plan and implement it with good notion and explanation."),
]


# ========================================================================
# Module loaders per variant
# ========================================================================

def load_ours_27_modules() -> List[Tuple[str, str]]:
    kb = {}
    for f in sorted(cfg.KB_DIR.glob("S*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        kb[d["id"]] = d
    return [
        (sid, f"{kb[sid]['name']['zh']}: {kb[sid]['description']['one_sentence']}")
        for sid in sorted(kb.keys())
    ]


def load_vanilla_39_modules() -> List[Tuple[str, str]]:
    return VANILLA_39_MODULES


# ========================================================================
# Cache helpers
# ========================================================================

def cache_load(path: Path) -> Dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def cache_save(path: Path, obj: Dict):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def load_or_sample_problems(n: int, seed: int) -> List[Dict]:
    if SAMPLES_PATH.exists():
        data = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
        if len(data) >= n and data[0].get("_seed") == seed:
            return data[:n]
    from task_env.base_env import TaskEnvironment
    kb = {}
    for f in sorted(cfg.KB_DIR.glob("S*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        kb[d["id"]] = d
    env = TaskEnvironment(strategy_kb=kb)
    test_pool = env.get_all_problems("test")
    rng = random.Random(seed)
    rng.shuffle(test_pool)
    sample = test_pool[:n]
    # annotate seed for reproducibility
    sample[0]["_seed"] = seed
    SAMPLES_PATH.write_text(json.dumps(sample, ensure_ascii=False, indent=2))
    return sample


# ========================================================================
# Generators
# ========================================================================

def extract_ids(raw, valid: set) -> List[str]:
    items = raw if isinstance(raw, list) else re.findall(r"[SM]\d{2}", str(raw))
    out = []
    for it in items:
        s = str(it).strip().upper()
        m = re.search(r"[SM]\d{2}", s)
        if m and m.group(0) in valid:
            out.append(m.group(0))
    return list(dict.fromkeys(out))[:5]


def discover_structure(client, modules: List[Tuple[str, str]],
                       domain: str, difficulty: str,
                       examples: List[Dict]) -> Dict:
    mod_text = "\n".join(f"- {mid}: {desc}" for mid, desc in modules)
    ex_text = "\n\n".join(
        f"示例 {i+1}: {p.get('description', '')[:400]}"
        for i, p in enumerate(examples)
    )
    valid_ids = {mid for mid, _ in modules}
    mod_map = dict(modules)

    # SELECT
    try:
        r = _generate_with_retry(client, SELECT_PROMPT.format(
            modules=mod_text, domain=domain, difficulty=difficulty,
            examples=ex_text), max_tokens=400, temperature=0.2)
        parsed = parse_json_from_llm(r["text"])
        selected = extract_ids(parsed.get("selected_ids", []), valid_ids)
    except Exception:
        selected = []
    if not selected:
        selected = [mid for mid, _ in modules[:3]]

    sel_text = "\n".join(f"- {mid}: {mod_map[mid]}" for mid in selected)

    # ADAPT
    try:
        r = _generate_with_retry(client, ADAPT_PROMPT.format(
            domain=domain, difficulty=difficulty, examples=ex_text,
            selected_text=sel_text), max_tokens=500, temperature=0.2)
        parsed = parse_json_from_llm(r["text"])
        adapted = parsed.get("adapted", [])
    except Exception:
        adapted = [{"id": mid, "adapted_description": mod_map[mid]} for mid in selected]

    ad_text = "\n".join(f"- [{a.get('id', '?')}] {a.get('adapted_description', '')}" for a in adapted)

    # IMPLEMENT
    try:
        r = _generate_with_retry(client, IMPLEMENT_PROMPT.format(
            domain=domain, difficulty=difficulty, examples=ex_text,
            adapted_text=ad_text), max_tokens=400, temperature=0.2)
        structure = parse_json_from_llm(r["text"])
        if not isinstance(structure, dict) or not structure:
            raise ValueError("empty")
    except Exception:
        structure = {a.get("id", f"step_{i}"): "" for i, a in enumerate(adapted)}
    if "final_answer" not in structure:
        structure["final_answer"] = ""

    return {"selected": selected, "adapted": adapted, "structure": structure}


def format_filled_json(raw: str, structure: Dict) -> str:
    try:
        filled = parse_json_from_llm(raw)
        parts = []
        for k in structure.keys():
            v = filled.get(k, "")
            if k == "final_answer":
                parts.append(f"【最终答案】{v}")
            else:
                parts.append(f"【{k}】{v}")
        return "\n".join(parts)
    except Exception:
        return raw.strip()


def generate_self_discover_answer(client, problem: str, structure: Dict) -> str:
    r = _generate_with_retry(client, EXECUTE_PROMPT.format(
        problem=problem,
        structure=json.dumps(structure, ensure_ascii=False, indent=2),
    ), max_tokens=800, temperature=0.3)
    return format_filled_json(r["text"], structure)


def generate_baseline_answer(client, problem: str) -> str:
    r = _generate_with_retry(client, BASELINE_PROMPT.format(problem=problem),
                        max_tokens=800, temperature=0.3)
    return r["text"].strip()


def judge_pair(client, problem: str, a: str, b: str) -> Dict:
    cache = _load_content_cache()
    key = _content_hash(problem, a, b)
    if key in cache:
        return cache[key]
    r = _generate_with_retry(client, JUDGE_PROMPT.format(problem=problem, answer_a=a, answer_b=b),
                        max_tokens=256, temperature=0.1)
    try:
        verdict = parse_json_from_llm(r["text"])
    except Exception:
        verdict = {"winner": "tie", "score_a": 5, "score_b": 5, "reasoning": "parse_failure"}
    cache[key] = verdict
    return verdict


# ========================================================================
# Pipeline
# ========================================================================

def run_variant(variant: str, problems: List[Dict], seed: int):
    """Generate answers for one variant and cache them."""
    answers_path = ANSWERS_DIR / f"{variant}_answers.json"
    answers = cache_load(answers_path)
    client = create_client()
    t0 = time.time()
    new_count = 0
    hit_count = 0

    if variant == "baseline":
        for i, p in enumerate(problems):
            pid = p["problem_id"]
            if pid in answers:
                hit_count += 1
                continue
            answers[pid] = generate_baseline_answer(client, p.get("description", ""))
            new_count += 1
            if new_count % 10 == 0:
                cache_save(answers_path, answers)
                print(f"  [{variant}] {i+1}/{len(problems)} "
                      f"(new={new_count} hit={hit_count}) {time.time()-t0:.0f}s")
        cache_save(answers_path, answers)

    elif variant in ("ours_27", "vanilla_39"):
        modules = load_ours_27_modules() if variant == "ours_27" else load_vanilla_39_modules()
        struct_path = STRUCTURES_DIR / f"{variant}_structures.json"
        structures = cache_load(struct_path)

        # Stage 1: per-category structure discovery
        from collections import defaultdict
        from task_env.base_env import TaskEnvironment
        kb = {}
        for f in sorted(cfg.KB_DIR.glob("S*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            kb[d["id"]] = d
        env = TaskEnvironment(strategy_kb=kb)
        train_pool = env.get_all_problems("train")

        by_cat = defaultdict(list)
        for p in train_pool:
            by_cat[(p.get("domain", "?"), p.get("difficulty", "?"))].append(p)

        # only discover structures for categories present in test sample
        needed_cats = set()
        for p in problems:
            needed_cats.add((p.get("domain", "?"), p.get("difficulty", "?")))

        for cat in sorted(needed_cats):
            key = f"{cat[0]}__{cat[1]}"
            if key in structures:
                continue
            if cat not in by_cat or len(by_cat[cat]) < 3:
                print(f"  [skip-category] {cat}")
                continue
            rng = random.Random(hash((variant, cat)) & 0xFFFFFFFF)
            examples = rng.sample(by_cat[cat], 3)
            print(f"  [discover] {cat}  examples={[p['problem_id'] for p in examples]}")
            structures[key] = discover_structure(client, modules, cat[0], cat[1], examples)
            cache_save(struct_path, structures)

        # Stage 2: instance-level answers
        for i, p in enumerate(problems):
            pid = p["problem_id"]
            if pid in answers:
                hit_count += 1
                continue
            key = f"{p.get('domain', '?')}__{p.get('difficulty', '?')}"
            struct = structures.get(key)
            if struct is None:
                print(f"  [skip-answer] {pid}: no structure for {key}")
                continue
            answers[pid] = generate_self_discover_answer(
                client, p.get("description", ""), struct["structure"])
            new_count += 1
            if new_count % 10 == 0:
                cache_save(answers_path, answers)
                print(f"  [{variant}] {i+1}/{len(problems)} "
                      f"(new={new_count} hit={hit_count}) {time.time()-t0:.0f}s")
        cache_save(answers_path, answers)
    else:
        raise ValueError(f"unknown variant: {variant}")

    print(f"\n  [{variant}] total: new={new_count} hit={hit_count} "
          f"file={answers_path.name} ({time.time()-t0:.0f}s)")


def run_judge(variant_a: str, variant_b: str, problems: List[Dict], seed: int):
    """Judge variant_a vs variant_b on cached answers."""
    ans_a = cache_load(ANSWERS_DIR / f"{variant_a}_answers.json")
    ans_b = cache_load(ANSWERS_DIR / f"{variant_b}_answers.json")
    judgments_path = JUDGMENTS_DIR / f"{variant_a}_vs_{variant_b}.json"
    judgments = cache_load(judgments_path)

    client = create_client()
    t0 = time.time()
    rng = random.Random(seed)
    new_count = 0
    hit_count = 0

    for i, p in enumerate(problems):
        pid = p["problem_id"]
        if pid in judgments:
            hit_count += 1
            continue
        a_ans = ans_a.get(pid)
        b_ans = ans_b.get(pid)
        if not a_ans or not b_ans:
            continue
        # random side-swap
        if rng.random() < 0.5:
            left, right, a_was = a_ans, b_ans, "A"
        else:
            left, right, a_was = b_ans, a_ans, "B"
        v = judge_pair(client, p.get("description", ""), left, right)
        winner_raw = v.get("winner", "tie")
        if winner_raw == "tie":
            winner = "tie"
        elif winner_raw == a_was:
            winner = variant_a
        else:
            winner = variant_b
        sc_a = int(v.get("score_a", 5))
        sc_b = int(v.get("score_b", 5))
        score_a_variant = sc_a if a_was == "A" else sc_b
        score_b_variant = sc_b if a_was == "A" else sc_a
        judgments[pid] = {
            "winner": winner,
            "score_a": score_a_variant,
            "score_b": score_b_variant,
            "reasoning": v.get("reasoning", ""),
            "a_was": a_was,
            "domain": p.get("domain", "?"),
            "difficulty": p.get("difficulty", "?"),
        }
        new_count += 1
        if new_count % 10 == 0:
            cache_save(judgments_path, judgments)
            _save_content_cache()
            print(f"  [judge {variant_a} vs {variant_b}] {i+1}/{len(problems)} "
                  f"(new={new_count} hit={hit_count}) {time.time()-t0:.0f}s")

    cache_save(judgments_path, judgments)
    _save_content_cache()
    # Report
    report_judgments(variant_a, variant_b, judgments)
    print(f"  total: new={new_count} hit={hit_count} ({time.time()-t0:.0f}s)")


def report_judgments(variant_a: str, variant_b: str, judgments: Dict):
    results = list(judgments.values())
    n = len(results)
    a_wins = sum(1 for r in results if r["winner"] == variant_a)
    b_wins = sum(1 for r in results if r["winner"] == variant_b)
    ties = sum(1 for r in results if r["winner"] == "tie")
    decided = a_wins + b_wins
    wr = a_wins / decided if decided else 0.5
    mean_d = np.mean([r["score_a"] - r["score_b"] for r in results]) if results else 0

    print(f"\n{'='*60}\n  {variant_a} vs {variant_b} — {n} problems\n{'='*60}")
    print(f"  {variant_a}={a_wins}  {variant_b}={b_wins}  tie={ties}  "
          f"win_rate={wr:.1%}  mean_Δ={mean_d:+.2f}")

    by_dom = defaultdict(lambda: [0, 0, 0])
    for r in results:
        idx = 0 if r["winner"] == variant_a else 1 if r["winner"] == variant_b else 2
        by_dom[r["domain"]][idx] += 1
    print("\n  By domain:")
    for dom in sorted(by_dom.keys()):
        w, l, t = by_dom[dom]
        d = w + l
        rate = w / d if d else 0.5
        print(f"    {dom:<22}: {variant_a}={w:>2} {variant_b}={l:>2} tie={t} wr={rate:.1%} (n={w+l+t})")

    by_diff = defaultdict(lambda: [0, 0, 0])
    for r in results:
        idx = 0 if r["winner"] == variant_a else 1 if r["winner"] == variant_b else 2
        by_diff[r["difficulty"]][idx] += 1
    print("\n  By difficulty:")
    for d_ in ["easy", "medium", "hard"]:
        if d_ not in by_diff: continue
        w, l, t = by_diff[d_]
        dec = w + l
        rate = w / dec if dec else 0.5
        print(f"    {d_:<8}: {variant_a}={w:>2} {variant_b}={l:>2} tie={t} wr={rate:.1%} (n={w+l+t})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variant", help="Generate answers for one variant")
    ap.add_argument("--judge", nargs=2, metavar=("A", "B"),
                    help="Judge variant A vs variant B")
    ap.add_argument("--report", nargs=2, metavar=("A", "B"),
                    help="Re-report cached judgments")
    ap.add_argument("--sample", default=None,
                    help="Override sample file (e.g. sample_holdout_50.json)")
    args = ap.parse_args()

    if args.sample:
        problems = json.loads((CACHE_ROOT / args.sample).read_text(encoding="utf-8"))
        # strip meta marker if present
        problems = [p for p in problems if "description" in p]
        print(f"Problem sample: {len(problems)} from {args.sample}")
    else:
        problems = load_or_sample_problems(args.n, args.seed)
        print(f"Problem sample: {len(problems)} (seed={args.seed})")

    if args.variant:
        run_variant(args.variant, problems, args.seed)
    elif args.judge:
        run_judge(args.judge[0], args.judge[1], problems, args.seed)
    elif args.report:
        jpath = JUDGMENTS_DIR / f"{args.report[0]}_vs_{args.report[1]}.json"
        report_judgments(args.report[0], args.report[1], cache_load(jpath))
    else:
        print("Specify --variant, --judge, or --report")


if __name__ == "__main__":
    main()
