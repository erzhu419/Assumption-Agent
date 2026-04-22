"""
Phase 2 v12 variants — isolate L1/L3/L4 improvements over v11.

All use EXECUTE_FULL structure (priors + triggers + wisdom) except math/science
in v12c/v12 which route to EXECUTE_HYGIENE.

Variants (selected via --variant):
  phase2_v12a  — L1: verified ∩ compressed triggers + orig wisdom
  phase2_v12b  — L3: v11 compressed triggers + compressed wisdom
  phase2_v12c  — L4: v11 compressed triggers + orig wisdom + math/sci hygiene routing
  phase2_v12   — ALL three stacked: verified+compressed triggers + compressed wisdom + math/sci hygiene
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
SELECTIONS_PATH = CACHE / "phase2_v3_selections.json"

WISDOM_ORIG = CACHE / "wisdom_library.json"
WISDOM_V12 = CACHE / "wisdom_library_v12.json"
TRIGGERS_V11 = CACHE / "trigger_library_v11.json"
TRIGGERS_V12 = CACHE / "trigger_library_v12.json"

MATH_SCI = {"mathematics", "science"}


EXECUTE_FULL = """# 你要解决下面的问题。

## 思维背景 1：本类别的通用 attention priors
{priors_block}

## 思维背景 2：本类别常见警觉
{triggers_block}

## 思维背景 3：可能适用的跨文明 wisdom
{wisdom_block}

## 问题
{problem}

## 要求
- **不要**用 Step 1/2 格式
- 带这三层觉知直接答题
- 语言精炼，不超过 500 字

开始：
"""


EXECUTE_MATH = """你是一位严谨的问题解决者。针对下面的数学问题，给出清晰、完整的解答。

## 计算自检（脑中执行）
- 审题：变量范围、约束、定义都明确了吗？
- 过程：每步代数/逻辑变换是可逆或等价的吗？
- 自检：答案代入原始条件是否自洽？极端 case (n=0, n=1) 下是否合理？

## 要求
- 直接给证明/推导，不要引用外部方法论
- 结构清晰，每步说明理由
- 不超过 550 字

## 问题
{problem}
"""


EXECUTE_SCIENCE = """你是一位严谨的问题解决者。针对下面的科学问题，给出清晰、结构化的解答。

## 科学自检（脑中执行）
- 量纲与单位：每个量的单位都对得上吗？
- 机制 vs 描述：答案给出的是机制解释，还是只描述了现象？
- 可证伪性：如果我的假设错了，会产生什么不同的观察？

## 要求
- 直接给分析和结论，不必引用外部方法论
- 结构清晰，每一论断都有推理支撑
- 不超过 500 字

## 问题
{problem}
"""


def cache_load(p):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def cache_save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def format_priors(priors):
    return "\n".join(f"  {i+1}. {p}" for i, p in enumerate(priors)) or "  (无)"


def format_triggers(trs):
    return "\n".join(f"  - {t}" for t in trs) or "  (无)"


def format_wisdom(selected):
    if not selected:
        return "  (本题没有特别适用的 wisdom)"
    return "\n".join(
        f"  • 【{e['id']} — {e['source']}】《{e['aphorism']}》\n"
        f"      {e['unpacked_for_llm']}"
        for e in selected)


def get_variant_config(variant: str):
    """Returns (triggers_path, wisdom_path, use_math_hygiene: bool)."""
    if variant == "phase2_v12a":
        return TRIGGERS_V12, WISDOM_ORIG, False  # verified+compressed triggers, orig wisdom
    if variant == "phase2_v12b":
        return TRIGGERS_V11, WISDOM_V12, False   # v11 compressed, compressed wisdom
    if variant == "phase2_v12c":
        return TRIGGERS_V11, WISDOM_ORIG, True   # v11 compressed, hygiene on math/sci
    if variant == "phase2_v12":
        return TRIGGERS_V12, WISDOM_V12, True    # all three stacked
    raise ValueError(f"unknown variant: {variant}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["phase2_v12", "phase2_v12a", "phase2_v12b", "phase2_v12c"])
    ap.add_argument("--base", default="orient_hybrid")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    triggers_path, wisdom_path, use_hygiene = get_variant_config(args.variant)

    answers_path = ANSWERS_DIR / f"{args.variant}_answers.json"
    struct_path = STRUCTURES_DIR / f"{args.variant}_structures.json"
    answers = cache_load(answers_path)

    if not struct_path.exists():
        base_struct = cache_load(STRUCTURES_DIR / f"{args.base}_structures.json")
        if base_struct:
            cache_save(struct_path, base_struct)
    structures = cache_load(struct_path)

    library = json.loads(wisdom_path.read_text(encoding="utf-8"))
    lib_by_id = {e["id"]: e for e in library}
    selections = cache_load(SELECTIONS_PATH)
    triggers_db = json.loads(triggers_path.read_text(encoding="utf-8"))
    print(f"  [{args.variant}]")
    print(f"  triggers: {triggers_path.name} ({sum(len(v) for v in triggers_db.values())} total)")
    print(f"  wisdom:   {wisdom_path.name} ({len(library)} entries)")
    print(f"  math/sci hygiene routing: {use_hygiene}")

    sample = json.loads((CACHE / "sample_100.json").read_text(encoding="utf-8"))[: args.n]

    client = create_client()
    t0 = time.time()
    new_count = hit_count = 0
    hygiene_count = full_count = 0

    for i, p in enumerate(sample):
        pid = p["problem_id"]
        if pid in answers:
            hit_count += 1
            continue
        dom = p.get("domain", "?")
        diff = p.get("difficulty", "?")
        problem = p.get("description", "")

        if use_hygiene and dom in MATH_SCI:
            if dom == "mathematics":
                prompt = EXECUTE_MATH.format(problem=problem)
                max_tok = 1100
            else:
                prompt = EXECUTE_SCIENCE.format(problem=problem)
                max_tok = 900
            hygiene_count += 1
        else:
            key = f"{dom}__{diff}"
            struct = structures.get(key)
            if struct is None:
                continue
            priors = struct.get("attention_priors", [])
            triggers = select_triggers_for_category(triggers_db, dom, diff)
            sel_ids = selections.get(pid, [])
            wisdom_entries = [lib_by_id[sid] for sid in sel_ids if sid in lib_by_id]

            prompt = EXECUTE_FULL.format(
                priors_block=format_priors(priors),
                triggers_block=format_triggers(triggers),
                wisdom_block=format_wisdom(wisdom_entries),
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
            print(f"  [{args.variant}] {i+1}/{len(sample)} "
                  f"hyg={hygiene_count} full={full_count} {time.time()-t0:.0f}s")

    cache_save(answers_path, answers)
    print(f"\n  [{args.variant}] done: hyg={hygiene_count} full={full_count} {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
