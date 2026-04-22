"""
Orientation-aware extension of cached_framework.

Key differences from technique-form pipeline:

  SELECT is shown 'trigger' + 'attention_priors' instead of 'operational_steps'.
  ADAPT becomes: take the orientations and phrase them as self-questions for
    THIS task category (not 'steps' for solving it).
  IMPLEMENT output shape changes from {"step1":"", "final_answer":""} to
    {"attention_priors": [...], "answer": ""}.
  EXECUTE prompt shows the priors as what-to-hold-in-mind, NOT steps to follow.
    LLM then generates a direct answer (free form), with the priors as attention
    direction rather than a fill-in-the-blank template.

This is a minimal change — reuses the cached baseline/ours_27 answers,
only introduces a new variant (orient_12 or orient_27) that uses this pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "phase zero" / "scripts"))

import _config as cfg  # noqa: E402
from llm_client import create_client, parse_json_from_llm  # noqa: E402


# Reuse cache root from cached_framework
CACHE_ROOT = PROJECT.parent / "phase two" / "analysis" / "cache"
ANSWERS_DIR = CACHE_ROOT / "answers"
STRUCTURES_DIR = CACHE_ROOT / "structures"
JUDGMENTS_DIR = CACHE_ROOT / "judgments"
SAMPLES_PATH = CACHE_ROOT / "sample_100.json"
for d in [ANSWERS_DIR, STRUCTURES_DIR, JUDGMENTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

KB_DIR = cfg.KB_DIR
ORIENT_DIR = cfg.PHASE0_DIR / "kb" / "strategies_orientation"
ORIENT_CANONICAL_DIR = cfg.PHASE0_DIR / "kb" / "strategies_canonical_orientation"
ORIENT_TRANSLATED_DIR = cfg.PHASE0_DIR / "kb" / "strategies_translated"


# ========================================================================
# Retry helper
# ========================================================================

def _gen(client, prompt: str, max_tokens: int = 800, temperature: float = 0.3,
         max_retries: int = 6) -> Dict:
    for attempt in range(max_retries):
        try:
            return client.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:
            msg = str(e).lower()
            transient = any(k in msg for k in [
                "503", "429", "500", "unavailable", "resource_exhausted",
                "overloaded", "high demand", "disconnect", "remoteprotocol",
                "timeout", "connection",
            ])
            if not transient or attempt == max_retries - 1:
                raise
            wait = min(2 ** attempt * 5, 120)
            print(f"    [retry {attempt+1}/{max_retries}] {str(e)[:80]}... waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


# ========================================================================
# Prompts — orientation form
# ========================================================================

SELECT_ORIENT_PROMPT = """# 你是方法论选择专家。

## 候选模块（每条是一个"意识朝向"，不是流程）
{modules}

## 任务类别
领域: {domain}
难度: {difficulty}

## 该类别的 3 个代表性问题
{examples}

## 你的任务
选出 **3-5 个**对解决这类问题最有用的意识朝向。不是"用什么技巧"，是"让 LLM 保持什么觉知"。

输出 JSON（不要代码块）：
{{"selected_ids": ["ID1", ...], "reasoning": "为什么这几个觉知对这类问题最关键"}}
"""


ADAPT_ORIENT_PROMPT = """# 你是方法论改写专家。

## 任务类别
领域: {domain}
难度: {difficulty}

## 该类别的 3 个代表性问题
{examples}

## 被选中的意识朝向
{selected_text}

## 你的任务
把这些"意识朝向"改写为**针对该任务类别的 self-questions**。每一条不是步骤，是在解题时需要持续自问的问题。30-60 字。

输出 JSON：
{{"adapted_orientations": [
  {{"id": "模块ID",
    "adapted_question": "针对该类别的具体化自问"}}
]}}
"""


IMPLEMENT_ORIENT_PROMPT = """# 你是意识框架设计专家。

## 任务类别
领域: {domain}
难度: {difficulty}

## 改写后的自问
{adapted_text}

## 你的任务
把这些自问组合成一个 **attention prior 清单**（不是流程！），让 LLM 在解这类问题时能把这些问题保持在"思维背景"中，而不是按顺序"执行"。

输出 JSON（不要代码块）：
{{
  "attention_priors": [
    "自问 1（在读完问题后应该保持觉知的第一件事）",
    "自问 2（分析过程中持续警觉的第二件事）",
    "自问 3（生成答案后应该反向检验的事）",
    "（最多 5 条；顺序只是'思维重心'，不是'必须按此顺序'）"
  ],
  "note_on_execution": "这些 priors 是 attention direction，不是 checklist。LLM 应该带着它们直接回答问题，不要显式 'step 1, step 2'。"
}}
"""


EXECUTE_ORIENT_PROMPT = """# 你要解决下面的问题。

## 在开始前，请把以下几个"觉知朝向"放在脑后，持续保持（**不是按步骤执行**）：

{priors_block}

## 问题
{problem}

## 要求
- **不要**用 "Step 1、Step 2" 格式解答
- 带着上面的觉知，**直接、自然地**写出你的分析和答案
- 如果某个觉知让你发现了问题的新维度（比如"我的答案有 confirmation bias"），在答案末尾用一两句话指出
- 语言精炼，不超过 500 字

请直接开始你的答案：
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


# ========================================================================
# Module loaders
# ========================================================================

def load_orient_modules(hybrid: bool = True, source: str = "paraphrase") -> List[Tuple[str, Dict]]:
    """
    Load strategies.

    source="paraphrase": strategies_orientation/ (my Phase 1 paraphrases)
    source="canonical":  strategies_canonical_orientation/ (Polya/Popper originals)
    source="translated": strategies_translated/ (LLM-translated from canonical)

    If hybrid=True: 12 orient-form + 15 original technique-form.
    If hybrid=False: only the 12 orient-form.
    """
    modules = []
    orient_ids = set()

    if source == "canonical":
        src_dir = ORIENT_CANONICAL_DIR
    elif source == "translated":
        src_dir = ORIENT_TRANSLATED_DIR
    else:
        src_dir = ORIENT_DIR

    for f in sorted(src_dir.glob("S*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        modules.append((d["id"], d))
        orient_ids.add(d["id"])

    if hybrid:
        for f in sorted(KB_DIR.glob("S*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            if d["id"] not in orient_ids:
                modules.append((d["id"], d))

    modules.sort(key=lambda x: x[0])
    return modules


def format_modules(modules: List[Tuple[str, Dict]]) -> str:
    lines = []
    for sid, d in modules:
        name = d["name"]["zh"]
        form = d.get("form", "technique")
        one = d["description"]["one_sentence"]
        if form == "orientation":
            priors = d.get("attention_priors", [])
            priors_str = "; ".join(priors[:2]) if priors else ""
            lines.append(f"- {sid} [意识朝向] ({name}): {one}\n  核心自问: {priors_str}")
        else:
            lines.append(f"- {sid} [技巧] ({name}): {one}")
    return "\n".join(lines)


def format_examples(examples: List[Dict]) -> str:
    return "\n\n".join(
        f"示例 {i+1}: {p.get('description', '')[:400]}"
        for i, p in enumerate(examples)
    )


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


def load_sample(n: int) -> List[Dict]:
    if not SAMPLES_PATH.exists():
        raise FileNotFoundError("sample_100.json missing — run cached_framework first")
    data = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
    return data[:n]


# ========================================================================
# Pipeline
# ========================================================================

def extract_ids(raw, valid: set) -> List[str]:
    items = raw if isinstance(raw, list) else re.findall(r"S\d{2}", str(raw))
    out = []
    for it in items:
        s = str(it).strip().upper()
        m = re.search(r"S\d{2}", s)
        if m and m.group(0) in valid:
            out.append(m.group(0))
    return list(dict.fromkeys(out))[:5]


def discover_structure_orient(client, modules: List[Tuple[str, Dict]],
                              domain: str, difficulty: str,
                              examples: List[Dict]) -> Dict:
    mod_text = format_modules(modules)
    ex_text = format_examples(examples)
    valid_ids = {sid for sid, _ in modules}
    mod_map = dict(modules)

    # SELECT
    try:
        r = _gen(client, SELECT_ORIENT_PROMPT.format(
            modules=mod_text, domain=domain, difficulty=difficulty,
            examples=ex_text), max_tokens=500, temperature=0.2)
        parsed = parse_json_from_llm(r["text"])
        selected = extract_ids(parsed.get("selected_ids", []), valid_ids)
    except Exception:
        selected = []
    if not selected:
        selected = [sid for sid, _ in modules[:3]]

    sel_text_parts = []
    for sid in selected:
        d = mod_map[sid]
        form = d.get("form", "technique")
        name = d["name"]["zh"]
        one = d["description"]["one_sentence"]
        if form == "orientation":
            trigger = d.get("trigger", "")
            priors = d.get("attention_priors", [])
            sel_text_parts.append(
                f"- {sid} [意识朝向] ({name})\n"
                f"  一句话: {one}\n"
                f"  触发条件: {trigger}\n"
                f"  核心自问: " + " | ".join(priors)
            )
        else:
            sel_text_parts.append(f"- {sid} [技巧] ({name}): {one}")
    sel_text = "\n".join(sel_text_parts)

    # ADAPT
    try:
        r = _gen(client, ADAPT_ORIENT_PROMPT.format(
            domain=domain, difficulty=difficulty, examples=ex_text,
            selected_text=sel_text), max_tokens=600, temperature=0.2)
        parsed = parse_json_from_llm(r["text"])
        adapted = parsed.get("adapted_orientations", [])
    except Exception:
        adapted = [{"id": sid, "adapted_question": mod_map[sid]["description"]["one_sentence"]}
                   for sid in selected]

    ad_text = "\n".join(
        f"- [{a.get('id', '?')}] {a.get('adapted_question', '')}"
        for a in adapted
    )

    # IMPLEMENT
    try:
        r = _gen(client, IMPLEMENT_ORIENT_PROMPT.format(
            domain=domain, difficulty=difficulty, adapted_text=ad_text),
            max_tokens=500, temperature=0.2)
        parsed = parse_json_from_llm(r["text"])
        priors = parsed.get("attention_priors", [])
        if not priors:
            raise ValueError("no priors")
    except Exception:
        priors = [a.get("adapted_question", "") for a in adapted if a.get("adapted_question")]
        if not priors:
            priors = [mod_map[sid]["description"]["one_sentence"] for sid in selected]

    return {"selected": selected, "adapted": adapted, "attention_priors": priors}


def execute_orient(client, problem: str, structure: Dict) -> str:
    priors = structure.get("attention_priors", [])
    priors_block = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(priors))
    r = _gen(client, EXECUTE_ORIENT_PROMPT.format(
        priors_block=priors_block, problem=problem),
        max_tokens=900, temperature=0.3)
    return r["text"].strip()


# ========================================================================
# Main
# ========================================================================

def run_variant(variant: str, problems: List[Dict], hybrid: bool = True,
                source: str = "paraphrase"):
    answers_path = ANSWERS_DIR / f"{variant}_answers.json"
    struct_path = STRUCTURES_DIR / f"{variant}_structures.json"
    answers = cache_load(answers_path)
    structures = cache_load(struct_path)

    modules = load_orient_modules(hybrid=hybrid, source=source)
    print(f"  [{variant}] loaded {len(modules)} modules "
          f"({sum(1 for _, d in modules if d.get('form') == 'orientation')} orientation, "
          f"{sum(1 for _, d in modules if d.get('form', 'technique') == 'technique')} technique)")

    # Group problems by (domain, difficulty) using train pool for seed examples
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

    needed_cats = set()
    for p in problems:
        needed_cats.add((p.get("domain", "?"), p.get("difficulty", "?")))

    client = create_client()
    t0 = time.time()

    # Stage 1: category structures
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
        structures[key] = discover_structure_orient(client, modules, cat[0], cat[1], examples)
        cache_save(struct_path, structures)

    # Stage 2: instance answers
    new_count = 0
    hit_count = 0
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
        answers[pid] = execute_orient(client, p.get("description", ""), struct)
        new_count += 1
        if new_count % 10 == 0:
            cache_save(answers_path, answers)
            print(f"  [{variant}] {i+1}/{len(problems)} "
                  f"(new={new_count} hit={hit_count}) {time.time()-t0:.0f}s")
    cache_save(answers_path, answers)
    print(f"  [{variant}] total: new={new_count} hit={hit_count} {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="e.g. orient_hybrid, orient_canonical_hybrid")
    ap.add_argument("--hybrid", action="store_true", default=True,
                    help="12 orientation + 15 technique (default)")
    ap.add_argument("--pure", action="store_true",
                    help="only 12 orientation strategies")
    ap.add_argument("--source", choices=["paraphrase", "canonical", "translated"],
                    default="paraphrase",
                    help="orientation source: paraphrase | canonical | translated")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    hybrid = not args.pure
    problems = load_sample(args.n)
    print(f"Running {args.variant} ({'hybrid' if hybrid else 'pure orientation'}, "
          f"source={args.source}) on {len(problems)} problems")
    run_variant(args.variant, problems, hybrid=hybrid, source=args.source)


if __name__ == "__main__":
    main()
