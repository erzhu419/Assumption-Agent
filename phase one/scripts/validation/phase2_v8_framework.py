"""
Phase 2 v8 — computation-focused hygiene for math/science.

Diagnosis of v5 math losses: Flash wastes tokens acknowledging scaffolding
(quoting 【W023】, applying "sunk cost prior", etc) before touching the actual
derivation. Judge reasons repeatedly cite "过多方法论引用，分散核心证明" and
"结尾处未完成证明".

Design:
  - mathematics: bare baseline prompt + 3-line COMPUTATION hygiene (not methodology)
  - science:     bare baseline prompt + 3-line scientific-method hygiene
  - engineering / business / daily_life / software_engineering: v5 full stack

Key difference from v7b: v7b used Stage-1 priors for math/sci (still abstract
methodology). v8 uses NO priors/triggers/wisdom on math — only computation
hygiene phrased as self-check, not orientation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "phase zero" / "scripts"))

import _config as cfg  # noqa: E402
from llm_client import create_client  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from cached_framework import _generate_with_retry
from phase2_framework import select_triggers_for_category


CACHE = PROJECT.parent / "phase two" / "analysis" / "cache"
ANSWERS_DIR = CACHE / "answers"
STRUCTURES_DIR = CACHE / "structures"
WISDOM_PATH = CACHE / "wisdom_library.json"
SELECTIONS_PATH = CACHE / "phase2_v3_selections.json"
TRIGGERS_PATH = CACHE / "trigger_library.json"


EXECUTE_MATH = """你是一位严谨的问题解决者。针对下面的数学问题，给出清晰、完整的解答。

## 计算自检（作答时在脑中执行，不必写出）
- 审题：变量范围、约束、定义都明确了吗？
- 过程：每步代数/逻辑变换是可逆或等价的吗？有没有隐含的除零或开根？
- 自检：答案代入原始条件是否自洽？极端 case (n=0, n=1, 无穷) 下是否合理？

## 要求
- 直接给证明/推导，不要引用外部方法论
- 结构清晰，每步说明理由
- 语言精炼，不超过 550 字（数学题可适当放宽）

## 问题
{problem}
"""


EXECUTE_SCIENCE = """你是一位严谨的问题解决者。针对下面的科学问题，给出清晰、结构化的解答。

## 科学自检（作答时在脑中执行）
- 量纲与单位：每个量的单位都对得上吗？
- 机制 vs 描述：答案给出的是机制解释，还是只描述了现象？
- 可证伪性：如果我的假设错了，会产生什么不同的观察？

## 要求
- 直接给分析和结论，不必引用外部方法论
- 结构清晰，每一论断都有推理支撑
- 语言精炼，不超过 500 字

## 问题
{problem}
"""


EXECUTE_FULL = """# 你要解决下面的问题。

## 思维背景 1：本类别的通用 attention priors
{priors_block}

## 思维背景 2：从历史失败中沉淀的 category-specific 警觉
{triggers_block}

## 思维背景 3：可能适用的跨文明 wisdom（已 pre-unpacked，非强制引用）
{wisdom_block}

## 问题
{problem}

## 要求
- **不要**用 Step 1/2 格式
- 带这三层觉知直接答题；引用 aphorism 仅当真 fire
- 语言精炼，不超过 500 字

开始：
"""


def cache_load(p: Path):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def cache_save(p: Path, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def format_priors(priors: List[str]) -> str:
    return "\n".join(f"  {i+1}. {p}" for i, p in enumerate(priors)) or "  (无)"


def format_triggers(trs: List[str]) -> str:
    return "\n".join(f"  - {t}" for t in trs) or "  (无)"


def format_wisdom(selected: List[Dict]) -> str:
    if not selected:
        return "  (本题没有特别适用的 wisdom)"
    parts = []
    for e in selected:
        parts.append(
            f"  • 【{e['id']} — {e['source']}】《{e['aphorism']}》\n"
            f"      {e['unpacked_for_llm']}"
        )
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="phase2_v8")
    ap.add_argument("--base", default="orient_hybrid")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    answers_path = ANSWERS_DIR / f"{args.variant}_answers.json"
    struct_path = STRUCTURES_DIR / f"{args.variant}_structures.json"
    answers = cache_load(answers_path)

    if not struct_path.exists():
        base_struct = cache_load(STRUCTURES_DIR / f"{args.base}_structures.json")
        if base_struct:
            cache_save(struct_path, base_struct)
    structures = cache_load(struct_path)

    library = json.loads(WISDOM_PATH.read_text(encoding="utf-8"))
    lib_by_id = {e["id"]: e for e in library}
    selections = cache_load(SELECTIONS_PATH)
    triggers_db = cache_load(TRIGGERS_PATH)
    print(f"  wisdom: {len(library)}, triggers: {sum(len(v) for v in triggers_db.values())} (v5 library)")
    print(f"  math/science: computation-hygiene only | others: v5 full stack")

    sample = json.loads((CACHE / "sample_100.json").read_text(encoding="utf-8"))[: args.n]

    client = create_client()
    t0 = time.time()
    new_count = 0
    hit_count = 0
    math_count = sci_count = full_count = 0

    for i, p in enumerate(sample):
        pid = p["problem_id"]
        if pid in answers:
            hit_count += 1
            continue
        dom = p.get("domain", "?")
        diff = p.get("difficulty", "?")
        problem = p.get("description", "")

        if dom == "mathematics":
            prompt = EXECUTE_MATH.format(problem=problem)
            max_tok = 1100  # raise for math (v5 loss: truncated proofs)
            math_count += 1
        elif dom == "science":
            prompt = EXECUTE_SCIENCE.format(problem=problem)
            max_tok = 900
            sci_count += 1
        else:
            key = f"{dom}__{diff}"
            struct = structures.get(key)
            if struct is None:
                print(f"  [skip] {pid}: no structure for {key}")
                continue
            priors = struct.get("attention_priors", [])
            priors_block = format_priors(priors)
            triggers = select_triggers_for_category(triggers_db, dom, diff)
            triggers_block = format_triggers(triggers)
            sel_ids = selections.get(pid, [])
            wisdom_entries = [lib_by_id[sid] for sid in sel_ids if sid in lib_by_id]
            wisdom_block = format_wisdom(wisdom_entries)
            prompt = EXECUTE_FULL.format(
                priors_block=priors_block,
                triggers_block=triggers_block,
                wisdom_block=wisdom_block,
                problem=problem,
            )
            max_tok = 900
            full_count += 1

        try:
            resp = _generate_with_retry(client, prompt, max_tokens=max_tok, temperature=0.3)
            answers[pid] = resp["text"].strip()
        except Exception as e:
            print(f"  [error] {pid}: {e}")
            continue

        new_count += 1
        if new_count % 10 == 0:
            cache_save(answers_path, answers)
            print(f"  [{args.variant}] {i+1}/{len(sample)} math={math_count} sci={sci_count} full={full_count} {time.time()-t0:.0f}s")

    cache_save(answers_path, answers)
    print(f"\n  [{args.variant}] done: math={math_count} sci={sci_count} full={full_count} {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
