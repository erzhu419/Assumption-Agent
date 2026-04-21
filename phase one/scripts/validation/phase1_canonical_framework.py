"""
Phase 1 redo: Canonical orientation from Polya/Popper source texts.

Loads 12 canonical JSONs (with original heuristic questions) + 15 technique JSONs.
Per-category Stage 1 picks 3-5 modules. EXECUTE uses **persona prompting** —
LLM takes on the stance of someone internalizing canonical heuristic questions,
rather than "following steps".

Stacks with Phase 2 triggers.

Two variants produced:
  phase1_canonical         — canonical only, no Phase 2 triggers (clean ablation)
  phase1_canonical_plus_p2 — canonical + Phase 2 failure triggers (final test)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "phase zero" / "scripts"))

import _config as cfg  # noqa: E402
from llm_client import create_client, parse_json_from_llm  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from cached_framework import _generate_with_retry
from phase2_framework import select_triggers_for_category

CACHE = PROJECT.parent / "phase two" / "analysis" / "cache"
ANSWERS_DIR = CACHE / "answers"
STRUCTURES_DIR = CACHE / "structures"
TRIGGERS_PATH = CACHE / "trigger_library.json"

KB_DIR = cfg.KB_DIR
CANONICAL_DIR = cfg.PHASE0_DIR / "kb" / "strategies_canonical"


# ========================================================================
# Prompts
# ========================================================================

SELECT_PROMPT = """# 你是方法论选择专家。

## 候选模块 — 两类：
- **canonical (意识朝向)**: 带原典启发式问句的策略 (Polya/Popper 原文)
- **technique (技巧)**: 操作步骤型策略

{modules}

## 任务类别
领域: {domain}
难度: {difficulty}

## 代表问题
{examples}

## 你的任务
选 **3-5 个**对这类任务最有帮助的模块。意识朝向和技巧可以混选，选你认为最直接有用的。

输出 JSON：
{{"selected_ids": ["S0X", ...], "reasoning": "简短理由"}}
"""


ADAPT_PROMPT = """# 改写选中模块。

## 任务类别: {domain} / {difficulty}

## 代表问题: {examples}

## 选中模块:
{selected_text}

## 你的任务
- canonical 模块：在**原典问句基础上**，改写为针对该任务类别的具体自问（不是替换原问句，是"这类问题下，原问句如何具体化")。
- technique 模块：改写为针对该任务类别的具体说法。

每条 30-50 字。

输出 JSON:
{{"adapted": [{{"id": "S0X", "form": "canonical/technique", "adapted_text": "..."}}]}}
"""


# EXECUTE for canonical-dominated categories: persona-style
EXECUTE_PERSONA = """# 你作为一个**内化了以下方法论精神**的思考者回答下面的问题。

## 你已经内化的原典气质（persona）

{canonical_block}

## 从历史失败中沉淀的 category-specific 警觉 ({triggers_count} 条)

{triggers_block}

## 任务类别的 attention priors（来自 adapt 阶段）

{priors_block}

---

## 问题

{problem}

## 要求

- **不要**用 "Step 1、Step 2" 格式解答
- 让原典问句**在你读问题时自动浮现**，而不是被应用
- 带着气质直接回答，像一个已经内化了 Polya/Popper 视角的人会怎么做
- 结尾如果某个原典问句让你发现答案的盲点，点出即可
- 语言精炼，不超过 500 字

开始：
"""


# ========================================================================
# Module loaders
# ========================================================================

def load_canonical_modules() -> List[Tuple[str, Dict]]:
    """Load 12 canonical + 15 technique. Canonical first (lower ID order)."""
    modules = {}
    # Canonical
    for f in sorted(CANONICAL_DIR.glob("S*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        modules[d["id"]] = d
    canonical_ids = set(modules.keys())
    # Technique (non-canonical)
    for f in sorted(KB_DIR.glob("S*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d["id"] not in canonical_ids:
            modules[d["id"]] = d
    return sorted(modules.items(), key=lambda x: x[0])


def format_module_list(modules: List[Tuple[str, Dict]]) -> str:
    lines = []
    for sid, d in modules:
        form = d.get("form", "technique")
        name = d["name"]["zh"]
        one = d["description"]["one_sentence"]
        if form == "canonical_orientation":
            # Show first 2 heuristic questions as hint
            hq_zh = d.get("heuristic_questions_zh", [])[:2]
            hq_str = " | ".join(hq_zh)
            lines.append(f"- {sid} [canonical] ({name}): {one}\n  原典问句: {hq_str}")
        else:
            lines.append(f"- {sid} [technique] ({name}): {one}")
    return "\n".join(lines)


# ========================================================================
# Pipeline
# ========================================================================

def extract_ids(raw, valid: set) -> List[str]:
    items = raw if isinstance(raw, list) else re.findall(r"S\d{2}", str(raw))
    out = []
    for it in items:
        m = re.search(r"S\d{2}", str(it).upper())
        if m and m.group(0) in valid:
            out.append(m.group(0))
    return list(dict.fromkeys(out))[:5]


def discover_structure_canonical(client, modules: List[Tuple[str, Dict]],
                                  domain: str, difficulty: str,
                                  examples: List[Dict]) -> Dict:
    mod_map = dict(modules)
    mod_text = format_module_list(modules)
    ex_text = "\n\n".join(
        f"示例 {i+1}: {p.get('description', '')[:400]}"
        for i, p in enumerate(examples)
    )
    valid_ids = {sid for sid, _ in modules}

    # SELECT
    try:
        r = _generate_with_retry(client, SELECT_PROMPT.format(
            modules=mod_text, domain=domain, difficulty=difficulty,
            examples=ex_text), max_tokens=500, temperature=0.2)
        parsed = parse_json_from_llm(r["text"])
        selected = extract_ids(parsed.get("selected_ids", []), valid_ids)
    except Exception:
        selected = []
    if not selected:
        selected = [sid for sid, _ in modules[:3]]

    # Build selected text with canonical questions embedded
    sel_lines = []
    for sid in selected:
        d = mod_map[sid]
        form = d.get("form", "technique")
        name = d["name"]["zh"]
        one = d["description"]["one_sentence"]
        if form == "canonical_orientation":
            hq = d.get("heuristic_questions_zh", [])
            src = d.get("source", {})
            sel_lines.append(
                f"- {sid} [canonical] {name}\n"
                f"  一句话: {one}\n"
                f"  原典问句 (来自 {src.get('author','?')} {src.get('year','?')}):"
                + "\n    " + "\n    ".join(f"• {q}" for q in hq)
            )
        else:
            sel_lines.append(f"- {sid} [technique] {name}: {one}")
    sel_text = "\n".join(sel_lines)

    # ADAPT
    try:
        r = _generate_with_retry(client, ADAPT_PROMPT.format(
            domain=domain, difficulty=difficulty, examples=ex_text,
            selected_text=sel_text), max_tokens=600, temperature=0.2)
        parsed = parse_json_from_llm(r["text"])
        adapted = parsed.get("adapted", [])
    except Exception:
        adapted = [{"id": sid, "form": mod_map[sid].get("form", "technique"),
                    "adapted_text": mod_map[sid]["description"]["one_sentence"]}
                   for sid in selected]

    # Structure: list of priors per selected module
    priors = []
    for a in adapted:
        text = a.get("adapted_text", "")
        if text:
            priors.append(text)

    return {
        "selected": selected,
        "adapted": adapted,
        "attention_priors": priors,
        # Preserve raw canonical heuristic questions for EXECUTE
        "canonical_questions": [
            {"sid": sid,
             "name": mod_map[sid]["name"]["zh"],
             "source": mod_map[sid].get("source", {}),
             "questions": mod_map[sid].get("heuristic_questions_zh", [])}
            for sid in selected
            if mod_map[sid].get("form") == "canonical_orientation"
        ],
    }


def execute_persona(client, problem: str, structure: Dict,
                    triggers: List[str]) -> str:
    canonicals = structure.get("canonical_questions", [])
    if canonicals:
        can_parts = []
        for c in canonicals:
            src = c.get("source", {})
            src_str = f"来自 {src.get('author','?')} {src.get('year','?')}"
            qs = "\n    ".join(f"• {q}" for q in c.get("questions", [])[:5])
            can_parts.append(f"【{c['sid']} {c['name']} — {src_str}】\n    {qs}")
        canonical_block = "\n\n".join(can_parts)
    else:
        canonical_block = "(本类别没选中 canonical 模块)"

    priors = structure.get("attention_priors", [])
    priors_block = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(priors))

    triggers_block = "\n".join(f"  - {t}" for t in triggers) if triggers else "  (无)"

    r = _generate_with_retry(client, EXECUTE_PERSONA.format(
        canonical_block=canonical_block,
        triggers_count=len(triggers),
        triggers_block=triggers_block,
        priors_block=priors_block,
        problem=problem,
    ), max_tokens=900, temperature=0.3)
    return r["text"].strip()


# ========================================================================
# Main
# ========================================================================

def cache_load(p: Path):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def cache_save(p: Path, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["phase1_canonical", "phase1_canonical_plus_p2"])
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    use_triggers = args.variant == "phase1_canonical_plus_p2"

    answers_path = ANSWERS_DIR / f"{args.variant}_answers.json"
    struct_path = STRUCTURES_DIR / f"{args.variant}_structures.json"
    answers = cache_load(answers_path)
    structures = cache_load(struct_path)

    # Both variants share structures (only EXECUTE differs via triggers)
    # If either has structures cached, copy to the other
    alt_variant = "phase1_canonical_plus_p2" if args.variant == "phase1_canonical" else "phase1_canonical"
    alt_struct_path = STRUCTURES_DIR / f"{alt_variant}_structures.json"
    if not structures and alt_struct_path.exists():
        structures = cache_load(alt_struct_path)
        cache_save(struct_path, structures)
        print(f"  [reused] structures from {alt_variant}")

    modules = load_canonical_modules()
    print(f"  loaded {len(modules)} modules "
          f"({sum(1 for _, d in modules if d.get('form') == 'canonical_orientation')} canonical, "
          f"{sum(1 for _, d in modules if d.get('form', 'technique') != 'canonical_orientation')} technique)")

    triggers_db = cache_load(TRIGGERS_PATH) if use_triggers else {}
    sample = json.loads((CACHE / "sample_100.json").read_text(encoding="utf-8"))[: args.n]

    # Group train for seed examples
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
    for p in sample:
        needed_cats.add((p.get("domain", "?"), p.get("difficulty", "?")))

    client = create_client()
    t0 = time.time()

    # Stage 1
    for cat in sorted(needed_cats):
        key = f"{cat[0]}__{cat[1]}"
        if key in structures:
            continue
        if cat not in by_cat or len(by_cat[cat]) < 3:
            continue
        rng = random.Random(hash(("canon", cat)) & 0xFFFFFFFF)
        examples = rng.sample(by_cat[cat], 3)
        print(f"  [discover] {cat}  examples={[p['problem_id'] for p in examples]}")
        structures[key] = discover_structure_canonical(
            client, modules, cat[0], cat[1], examples)
        sel = structures[key].get("selected", [])
        cn = len(structures[key].get("canonical_questions", []))
        print(f"    selected {sel}  ({cn} canonical)")
        cache_save(struct_path, structures)

    # Stage 2
    print("\n[Stage 2] Generating answers...")
    new_count = 0
    hit_count = 0
    for i, p in enumerate(sample):
        pid = p["problem_id"]
        if pid in answers:
            hit_count += 1
            continue
        key = f"{p.get('domain', '?')}__{p.get('difficulty', '?')}"
        struct = structures.get(key)
        if struct is None:
            print(f"  [skip] {pid}: no structure for {key}")
            continue
        triggers = (select_triggers_for_category(
            triggers_db, p.get("domain", "?"), p.get("difficulty", "?"))
                   if use_triggers else [])
        answers[pid] = execute_persona(
            client, p.get("description", ""), struct, triggers)
        new_count += 1
        if new_count % 10 == 0:
            cache_save(answers_path, answers)
            print(f"  [{args.variant}] {i+1}/{len(sample)} "
                  f"(new={new_count} hit={hit_count}) {time.time()-t0:.0f}s")
    cache_save(answers_path, answers)
    print(f"\n  total: new={new_count} hit={hit_count} {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
