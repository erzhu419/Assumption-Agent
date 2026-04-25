"""
Phase 2 v13-reflect — 2-turn reflection architecture.

Hypothesis: meta-knowledge (wisdom/priors) needs an explicit reflection pass to
actually fire. 1-pass execute tends to treat them as decorative prologue.

Turn 1 (draft): priors + triggers + wisdom + problem → first draft (same as v11)
Turn 2 (reflect+revise):
  "Your draft: {draft}.
   Attention priors for this category: {priors}.
   Cross-civilizational wisdom relevant: {wisdom}.
   For each prior/wisdom, self-audit: was it really applied, or just cited?
   Identify 1-2 biggest blindspots your draft has.
   Produce a revised final answer addressing those blindspots."

Key design choices:
  - Softer tone: "self-audit" not "critique" (critique triggers anxious-LLM
    over-rewrites that lose substance)
  - Only 1-2 blindspots: constrains LLM from totally rewriting, keeps revision focused
  - Math/sci keep v12c hygiene path (no reflection) — meta-knowledge routing
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
from phase2_v12_framework import (EXECUTE_MATH, EXECUTE_SCIENCE, format_priors,
                                   format_triggers, format_wisdom, MATH_SCI)


CACHE = PROJECT.parent / "phase two" / "analysis" / "cache"
ANSWERS_DIR = CACHE / "answers"
STRUCTURES_DIR = CACHE / "structures"
WISDOM_PATH = CACHE / "wisdom_library.json"
SELECTIONS_PATH = CACHE / "phase2_v3_selections.json"
TRIGGERS_V11 = CACHE / "trigger_library_v11.json"


# Turn 1: standard v11-style draft
DRAFT_PROMPT = """你要解决下面的问题。

## 思维背景 1: 本类别的通用 attention priors
{priors_block}

## 思维背景 2: 本类别常见警觉
{triggers_block}

## 思维背景 3: 可能适用的跨文明 wisdom
{wisdom_block}

## 问题
{problem}

## 要求
- 直接给出草稿答案，不要步骤标签
- 不超过 500 字

开始：
"""

# Turn 2: softer self-audit + revise
REFLECT_PROMPT = """你刚给出了一个草稿答案。

## 问题
{problem}

## 你的草稿
{draft}

## 本类别的关键警觉（对照用）
{priors_block}

## 本类别的 wisdom
{wisdom_block}

## 自检任务
对每条 prior / wisdom，判断：
- A. 草稿里**真的塑形了答案**（比如某个分析视角直接来自它）
- B. 草稿里**只是表面提及或根本没用到**（token 浪费 or 空转）

找出 **1-2 个** B 类中**最该被应用**的（不是装饰性的），在最终答案里真正把它 integrate 进来。

## 最终答案
直接输出修订后的答案（不要列出 audit 过程）。不超过 600 字。
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="phase2_v13_reflect")
    ap.add_argument("--base", default="orient_hybrid")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--sample", default="sample_100.json")
    args = ap.parse_args()

    answers_path = ANSWERS_DIR / f"{args.variant}_answers.json"
    drafts_path = ANSWERS_DIR / f"{args.variant}_drafts.json"
    struct_path = STRUCTURES_DIR / f"{args.variant}_structures.json"
    answers = cache_load(answers_path)
    drafts = cache_load(drafts_path)

    if not struct_path.exists():
        base_struct = cache_load(STRUCTURES_DIR / f"{args.base}_structures.json")
        if base_struct:
            cache_save(struct_path, base_struct)
    structures = cache_load(struct_path)

    library = json.loads(WISDOM_PATH.read_text(encoding="utf-8"))
    lib_by_id = {e["id"]: e for e in library}
    selections = cache_load(SELECTIONS_PATH)
    triggers_db = json.loads(TRIGGERS_V11.read_text(encoding="utf-8"))
    print(f"  {args.variant}: wisdom={len(library)}, triggers(v11)={sum(len(v) for v in triggers_db.values())}")
    print(f"  math/sci -> v12c hygiene (no reflection); others -> 2-turn reflect")

    sample = json.loads((CACHE / args.sample).read_text(encoding="utf-8"))[: args.n]

    client = create_client()
    t0 = time.time()
    new = hit = reflect_count = hygiene_count = 0

    for i, p in enumerate(sample):
        pid = p["problem_id"]
        if pid in answers:
            hit += 1
            continue
        dom = p.get("domain", "?")
        diff = p.get("difficulty", "?")
        problem = p.get("description", "")

        # Math/sci: bypass reflection, use v12c hygiene
        if dom in MATH_SCI:
            if dom == "mathematics":
                prompt = EXECUTE_MATH.format(problem=problem)
                max_tok = 1100
            else:
                prompt = EXECUTE_SCIENCE.format(problem=problem)
                max_tok = 900
            try:
                resp = _generate_with_retry(client, prompt, max_tokens=max_tok, temperature=0.3)
                answers[pid] = resp["text"].strip()
            except Exception as e:
                print(f"  [err] {pid}: {e}")
                continue
            hygiene_count += 1
        else:
            # 2-turn reflect path
            key = f"{dom}__{diff}"
            struct = structures.get(key, {"attention_priors": []})
            priors = struct.get("attention_priors", [])
            triggers = select_triggers_for_category(triggers_db, dom, diff)
            sel_ids = selections.get(pid, [])
            wisdom_entries = [lib_by_id[sid] for sid in sel_ids if sid in lib_by_id]

            priors_fmt = format_priors(priors)
            triggers_fmt = format_triggers(triggers)
            wisdom_fmt = format_wisdom(wisdom_entries)

            # Turn 1: draft
            if pid in drafts:
                draft = drafts[pid]
            else:
                try:
                    r1 = _generate_with_retry(client, DRAFT_PROMPT.format(
                        priors_block=priors_fmt, triggers_block=triggers_fmt,
                        wisdom_block=wisdom_fmt, problem=problem),
                        max_tokens=900, temperature=0.3)
                    draft = r1["text"].strip()
                    drafts[pid] = draft
                except Exception as e:
                    print(f"  [err draft {pid}]: {e}")
                    continue

            # Turn 2: reflect + revise
            try:
                r2 = _generate_with_retry(client, REFLECT_PROMPT.format(
                    problem=problem, draft=draft,
                    priors_block=priors_fmt, wisdom_block=wisdom_fmt),
                    max_tokens=1000, temperature=0.3)
                answers[pid] = r2["text"].strip()
            except Exception as e:
                print(f"  [err reflect {pid}]: {e}")
                continue
            reflect_count += 1

        new += 1
        if new % 10 == 0:
            cache_save(answers_path, answers)
            cache_save(drafts_path, drafts)
            print(f"  [{args.variant}] {i+1}/{len(sample)} hyg={hygiene_count} refl={reflect_count} {time.time()-t0:.0f}s")

    cache_save(answers_path, answers)
    cache_save(drafts_path, drafts)
    print(f"\n  [{args.variant}] done: hyg={hygiene_count} refl={reflect_count} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
