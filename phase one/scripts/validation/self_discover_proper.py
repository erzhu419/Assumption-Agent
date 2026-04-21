"""
Self-Discover — PROPER implementation per the paper.

Two-stage, task-LEVEL structure discovery + instance-level execution:

  Stage 1 (ONCE per task category = domain × difficulty):
    examples  → SELECT modules (LLM from 27 KB strategies)
              → ADAPT modules (LLM rephrases for this category)
              → IMPLEMENT (LLM produces JSON key-value reasoning structure)
    The JSON structure is cached per category.

  Stage 2 (per instance, 1 LLM call):
    problem + cached JSON structure
      → LLM fills in the JSON values and produces final answer.

This is what the paper actually does. My earlier implementation ran SELECT/
ADAPT/IMPLEMENT per-instance and cost 4× as many LLM calls per problem — hence
the 25% win rate was methodological failure, not Self-Discover failing.

Usage:
    python self_discover_proper.py --n 100           # default: 100 held-out problems
    python self_discover_proper.py --build-only      # just build structures, no A/B
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


# ========================================================================
# Prompt templates (paper fig. 3 style)
# ========================================================================

SELECT_PROMPT = """# 你是方法论选择专家。

## 27 个候选推理模块（每条给出 ID + 一句话描述）
{modules}

## 任务类别
领域: {domain}
难度: {difficulty}

## 该类别的 3 个代表性问题
{examples}

## 你的任务
这些问题在结构上有共性。请从 27 个模块中选出 **3-5 个**最可能帮助解决这类问题的模块。

输出 JSON（不要代码块）：
{{"selected_ids": ["S0X", ...], "reasoning": "简短理由（30-60字）"}}
"""


ADAPT_PROMPT = """# 你是方法论改写专家。

## 任务类别
领域: {domain}
难度: {difficulty}

## 该类别的 3 个代表性问题
{examples}

## 被选中的推理模块（原始描述）
{selected_text}

## 你的任务
把每个被选中的模块的描述**改写为针对该任务类别的具体说法**。不要复述原描述——要让它直接指导解决这类问题。每条改写 30-50 字。

输出 JSON（不要代码块）：
{{"adapted": [
  {{"id": "S0X", "adapted_description": "针对该类别的具体化描述"}}
]}}
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
把改写后的推理模块组装成一个 **JSON 推理结构**。它是一个 key-value 结构，key 是推理步骤的名称，value 留空字符串让后续 LLM 填写。

要求：
1. key 必须是具体的、可填空的（如 "核心矛盾识别"、"关键假设列表"、"最终建议"），不是抽象方法论名
2. key 按自然执行顺序排列，3-6 个 key
3. 最后一个 key 必须是 "final_answer"，让 LLM 输出最终答案

参考示例（颜色物体类任务）：
{{
  "type_and_color_of_each_item": "",
  "number_of_items_of_each_color": "",
  "number_of_items_of_each_type": "",
  "final_answer": ""
}}

针对本任务类别输出 JSON 推理结构（不要代码块）：
"""


EXECUTE_PROMPT = """# 你需要解决下面的具体问题。

## 问题
{problem}

## 推理结构（按此结构填空）
{structure}

## 要求
1. 按 JSON 中的 key 顺序依次填写每个 value
2. 每个 value 都要有实质内容，不能空泛
3. "final_answer" 字段必须是直接的、可操作的答案
4. 输出完整的 JSON（不要代码块）
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


JUDGE_PROMPT = """你是方法论评审专家。下面是同一个问题的两个解答，请客观评判。

## 问题
{problem}

## 解答 A
{answer_a}

## 解答 B
{answer_b}

## 评审任务
从四个维度综合评分：
1. 问题理解  2. 分析深度  3. 结构化程度  4. 实用性

输出 JSON（不要代码块）：
{{"winner": "A"或"B"或"tie", "score_a": 1-10 整数, "score_b": 1-10 整数,
  "reasoning": "80 字内说明胜因"}}
"""


# ========================================================================
# Helpers
# ========================================================================

@dataclass
class CategoryStructure:
    domain: str
    difficulty: str
    selected_ids: List[str]
    adapted: List[Dict]
    structure: Dict[str, str]  # JSON template
    examples_used: List[str]   # problem_ids used as seeds


def format_modules(kb: Dict[str, Dict]) -> str:
    return "\n".join(
        f"- {sid} ({kb[sid]['name']['zh']}): {kb[sid]['description']['one_sentence']}"
        for sid in sorted(kb.keys())
    )


def format_examples(examples: List[Dict]) -> str:
    return "\n\n".join(
        f"示例 {i+1}: {p.get('description', '')[:400]}"
        for i, p in enumerate(examples)
    )


def extract_strategy_ids(raw, kb_ids: set) -> List[str]:
    out = []
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        items = re.findall(r"S\d{2}", raw)
    else:
        return []
    for item in items:
        s = str(item).strip().upper()
        m = re.search(r"S\d{2}", s)
        if m and m.group(0) in kb_ids:
            out.append(m.group(0))
    return list(dict.fromkeys(out))[:5]


def discover_structure(client, kb: Dict[str, Dict],
                       domain: str, difficulty: str,
                       examples: List[Dict]) -> CategoryStructure:
    """Stage 1: SELECT → ADAPT → IMPLEMENT for one task category."""
    modules = format_modules(kb)
    ex_text = format_examples(examples)

    # SELECT
    sel_resp = client.generate(
        SELECT_PROMPT.format(modules=modules, domain=domain,
                             difficulty=difficulty, examples=ex_text),
        max_tokens=400, temperature=0.2,
    )
    try:
        sel = parse_json_from_llm(sel_resp["text"])
        selected = extract_strategy_ids(sel.get("selected_ids", []), set(kb.keys()))
    except Exception:
        selected = []
    if not selected:
        selected = sorted(kb.keys())[:3]

    selected_text = "\n".join(
        f"- {sid} ({kb[sid]['name']['zh']}): {kb[sid]['description']['one_sentence']}"
        for sid in selected
    )

    # ADAPT
    ad_resp = client.generate(
        ADAPT_PROMPT.format(domain=domain, difficulty=difficulty,
                            examples=ex_text, selected_text=selected_text),
        max_tokens=500, temperature=0.2,
    )
    try:
        ad = parse_json_from_llm(ad_resp["text"])
        adapted = ad.get("adapted", [])
    except Exception:
        adapted = [{"id": sid, "adapted_description": kb[sid]["description"]["one_sentence"]}
                   for sid in selected]

    adapted_text = "\n".join(
        f"- [{a.get('id', '?')}] {a.get('adapted_description', '')}"
        for a in adapted
    )

    # IMPLEMENT
    imp_resp = client.generate(
        IMPLEMENT_PROMPT.format(domain=domain, difficulty=difficulty,
                                examples=ex_text, adapted_text=adapted_text),
        max_tokens=400, temperature=0.2,
    )
    try:
        structure = parse_json_from_llm(imp_resp["text"])
        if not isinstance(structure, dict) or not structure:
            raise ValueError("empty structure")
    except Exception:
        # Fallback: build a minimal structure from adapted modules
        structure = {a.get("id", f"step_{i}"): "" for i, a in enumerate(adapted)}
        structure["final_answer"] = ""

    if "final_answer" not in structure:
        structure["final_answer"] = ""

    return CategoryStructure(
        domain=domain, difficulty=difficulty,
        selected_ids=selected, adapted=adapted,
        structure=structure,
        examples_used=[p["problem_id"] for p in examples],
    )


def execute_with_structure(client, problem: str, structure: Dict) -> str:
    """Stage 2: 1 LLM call to fill the JSON structure."""
    resp = client.generate(
        EXECUTE_PROMPT.format(problem=problem,
                              structure=json.dumps(structure, ensure_ascii=False, indent=2)),
        max_tokens=800, temperature=0.3,
    )
    raw = resp["text"].strip()
    # Try to parse as JSON; if parseable, render as a readable answer
    try:
        filled = parse_json_from_llm(raw)
        parts = []
        for k, v in filled.items():
            if k == "final_answer":
                parts.append(f"【最终答案】{v}")
            else:
                parts.append(f"【{k}】{v}")
        return "\n".join(parts)
    except Exception:
        return raw  # fallback to raw text


def baseline_solve(client, problem: str) -> str:
    return client.generate(
        BASELINE_PROMPT.format(problem=problem),
        max_tokens=800, temperature=0.3,
    )["text"].strip()


def judge(client, problem: str, a: str, b: str) -> Dict:
    resp = client.generate(
        JUDGE_PROMPT.format(problem=problem, answer_a=a, answer_b=b),
        max_tokens=256, temperature=0.1,
    )
    try:
        return parse_json_from_llm(resp["text"])
    except Exception:
        return {"winner": "tie", "score_a": 5, "score_b": 5, "reasoning": "parse_failure"}


# ========================================================================
# Main
# ========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--output", default="self_discover_proper_results.json")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    # Load KB + problems
    kb = {}
    for f in sorted(cfg.KB_DIR.glob("S*.json")):
        d = json.load(open(f, encoding="utf-8"))
        kb[d["id"]] = d
    print(f"Loaded {len(kb)} KB strategies")

    from task_env.base_env import TaskEnvironment
    env = TaskEnvironment(strategy_kb=kb)
    train_problems = env.get_all_problems("train")
    test_problems = env.get_all_problems("test")

    # Group TRAIN problems by (domain, difficulty) for seed examples
    groups = defaultdict(list)
    for p in train_problems:
        key = (p.get("domain", "unknown"), p.get("difficulty", "medium"))
        groups[key].append(p)
    print(f"Task categories: {len(groups)}")

    # Seed the structure for each category using 3 train examples
    client = create_client()
    structures: Dict[Tuple[str, str], CategoryStructure] = {}
    cache_path = PROJECT.parent / "phase two" / "analysis" / "self_discover_structures.json"
    cache_path.parent.mkdir(exist_ok=True)

    # Resume from cache if available
    if cache_path.exists() and not args.build_only:
        print(f"Loading cached structures from {cache_path.name}")
        try:
            cached = json.loads(cache_path.read_text())
            for rec in cached:
                key = (rec["domain"], rec["difficulty"])
                structures[key] = CategoryStructure(**rec)
        except Exception:
            structures = {}

    # Build missing structures
    print(f"\n[Stage 1] Building reasoning structures per category...")
    t0 = time.time()
    for key in sorted(groups.keys()):
        if key in structures:
            continue
        dom, diff = key
        available = groups[key]
        if len(available) < 3:
            print(f"  [skip] {dom}/{diff}: only {len(available)} train examples")
            continue
        examples = random.sample(available, 3)
        print(f"  discovering structure for {dom}/{diff} "
              f"(using {[p['problem_id'] for p in examples]})")
        try:
            struct = discover_structure(client, kb, dom, diff, examples)
            structures[key] = struct
            print(f"    SELECT={struct.selected_ids}  "
                  f"structure keys={list(struct.structure.keys())}")
        except Exception as e:
            print(f"    [error] {e}")

    # Cache
    cache_path.write_text(json.dumps(
        [asdict(s) for s in structures.values()],
        indent=2, ensure_ascii=False))
    print(f"\n  cached {len(structures)} structures to {cache_path.name} "
          f"({time.time() - t0:.0f}s)")

    if args.build_only:
        return

    # Sample held-out test problems
    random.seed(args.seed)
    random.shuffle(test_problems)
    test_sample = test_problems[: args.n]
    print(f"\n[Stage 2] A/B on {len(test_sample)} held-out test problems")

    results = []
    t0 = time.time()
    for i, p in enumerate(test_sample):
        pid = p["problem_id"]
        desc = p.get("description", "")
        dom = p.get("domain", "unknown")
        diff = p.get("difficulty", "medium")
        struct = structures.get((dom, diff))
        if struct is None:
            continue

        try:
            sd_ans = execute_with_structure(client, desc, struct.structure)
            base_ans = baseline_solve(client, desc)
        except Exception as e:
            print(f"  [skip] {pid}: {e}")
            continue

        # Random side-swap
        if random.random() < 0.5:
            a, b, sd_was = sd_ans, base_ans, "A"
        else:
            a, b, sd_was = base_ans, sd_ans, "B"

        v = judge(client, desc, a, b)
        winner_raw = v.get("winner", "tie")
        if winner_raw == "tie":
            winner = "tie"
        elif winner_raw == sd_was:
            winner = "self_discover"
        else:
            winner = "baseline"

        score_a = int(v.get("score_a", 5))
        score_b = int(v.get("score_b", 5))
        sd_score = score_a if sd_was == "A" else score_b
        bs_score = score_b if sd_was == "A" else score_a

        results.append({
            "problem_id": pid, "domain": dom, "difficulty": diff,
            "selected_ids": struct.selected_ids,
            "winner": winner,
            "sd_score": sd_score, "baseline_score": bs_score,
            "reasoning": v.get("reasoning", ""),
        })

        if (i + 1) % 10 == 0:
            w = sum(1 for r in results if r["winner"] == "self_discover")
            b_ = sum(1 for r in results if r["winner"] == "baseline")
            t_ = sum(1 for r in results if r["winner"] == "tie")
            print(f"  [{i+1}/{len(test_sample)}] SD={w} base={b_} tie={t_} "
                  f"{time.time() - t0:.0f}s")

    # Report
    n = len(results)
    w = sum(1 for r in results if r["winner"] == "self_discover")
    b_ = sum(1 for r in results if r["winner"] == "baseline")
    t_ = sum(1 for r in results if r["winner"] == "tie")
    decided = w + b_
    wr = w / decided if decided else 0.5
    mean_d = np.mean([r["sd_score"] - r["baseline_score"] for r in results]) if results else 0

    print(f"\n{'='*60}\n  SELF-DISCOVER (PROPER) — {n} problems\n{'='*60}")
    print(f"  Self-Discover wins: {w}  baseline: {b_}  ties: {t_}")
    print(f"  win_rate (excl. ties): {wr:.1%}")
    print(f"  mean judge score Δ: {mean_d:+.2f}")

    by_dom = defaultdict(lambda: [0, 0, 0])
    for r in results:
        idx = 0 if r["winner"] == "self_discover" else 1 if r["winner"] == "baseline" else 2
        by_dom[r["domain"]][idx] += 1
    print("\n  By domain:")
    for dom in sorted(by_dom.keys()):
        gw, gb, gt = by_dom[dom]
        d = gw + gb
        wr_ = gw / d if d else 0.5
        print(f"    {dom:<22}: SD={gw:>2} base={gb:>2} tie={gt} wr={wr_:.1%}  (n={gw+gb+gt})")

    by_diff = defaultdict(lambda: [0, 0, 0])
    for r in results:
        idx = 0 if r["winner"] == "self_discover" else 1 if r["winner"] == "baseline" else 2
        by_diff[r["difficulty"]][idx] += 1
    print("\n  By difficulty:")
    for diff in ["easy", "medium", "hard"]:
        if diff not in by_diff:
            continue
        gw, gb, gt = by_diff[diff]
        d = gw + gb
        wr_ = gw / d if d else 0.5
        print(f"    {diff:<8}: SD={gw:>2} base={gb:>2} tie={gt} wr={wr_:.1%}  (n={gw+gb+gt})")

    out_path = PROJECT.parent / "phase two" / "analysis" / args.output
    out_path.write_text(json.dumps({
        "n": n, "sd_wins": w, "baseline_wins": b_, "ties": t_,
        "win_rate": wr, "mean_score_delta": float(mean_d),
        "results": results,
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
