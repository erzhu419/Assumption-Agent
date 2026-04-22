"""
v16-cases-reflect: combine v15's case-based draft with v13-reflect's audit pass.

Turn 1 (draft with cases): identical to v15-exemplar
  priors + 2 wisdoms × (3 cross-domain exemplars + 1 same-domain exemplar) + problem
  → draft

Turn 2 (audit + revise): identical spirit to v13-reflect
  draft + priors + wisdom → self-audit prior application → revise

Hypothesis: if v15's cases provide a richer draft foundation AND v13-reflect's
audit catches drift, the combination should inherit both strengths.

Math/science bypass to v12c hygiene (cases + reflection both useless for proof).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "phase zero" / "scripts"))

import _config as cfg  # noqa: E402
from llm_client import create_client  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from cached_framework import _generate_with_retry
from phase2_v12_framework import (EXECUTE_MATH, EXECUTE_SCIENCE, format_priors,
                                   MATH_SCI)
from phase2_v15_exemplar_framework import (
    EXECUTE_V15, build_same_domain_exemplar, format_wisdom_with_cases)


CACHE = PROJECT.parent / "phase two" / "analysis" / "cache"
ANSWERS_DIR = CACHE / "answers"
STRUCTURES_DIR = CACHE / "structures"
WISDOM_PATH = CACHE / "wisdom_library.json"
SELECTIONS_PATH = CACHE / "phase2_v3_selections.json"
EXEMPLARS_PATH = CACHE / "wisdom_diverse_exemplars.json"
EMB_PATH = CACHE / "signal_embeddings.npz"

V13_REFLECT = ANSWERS_DIR / "phase2_v13_reflect_answers.json"
OURS_27 = ANSWERS_DIR / "ours_27_answers.json"


# Turn 2: audit with wisdom references (don't re-include cases — too heavy)
REFLECT_PROMPT = """你刚给出了一个草稿答案。

## 问题
{problem}

## 你的草稿
{draft}

## 本类别的 attention priors（对照用）
{priors_block}

## 激活的 wisdom 原则（精简版，仅用于自检）
{wisdom_brief}

## 自检任务
对每条 prior / wisdom，判断：
- A. 草稿里**真的塑形了答案**（某个分析视角直接来自它）
- B. 草稿里**只是表面提及或根本没用到**（空转、装饰）

找出 **1-2 个** B 类中**最该被应用**的，在最终答案里真正把它 integrate 进来。

## 最终答案
直接输出修订后的答案（不要列出 audit 过程）。不超过 650 字。
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


def format_wisdom_brief(wisdom_entries):
    """Compact wisdom reminder for Turn 2 (no cases, just reference)."""
    if not wisdom_entries:
        return "  (无)"
    lines = []
    for w in wisdom_entries:
        lines.append(f"  • {w['aphorism']} ({w.get('source', '?')}): "
                     f"{w.get('unpacked_for_llm', '')[:100]}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="phase2_v16_cases_reflect")
    ap.add_argument("--base", default="orient_hybrid")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-wisdoms", type=int, default=2)
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
    diverse_exs = cache_load(EXEMPLARS_PATH)
    v13 = json.loads(V13_REFLECT.read_text(encoding="utf-8"))
    ours = json.loads(OURS_27.read_text(encoding="utf-8"))

    emb = np.load(EMB_PATH, allow_pickle=True)
    prob_emb = emb["problem_emb"]
    prob_ids_emb = emb["problem_ids"].tolist()
    pid_to_emb_idx = {pid: i for i, pid in enumerate(prob_ids_emb)}

    print(f"  {args.variant}: wisdom={len(library)}, exemplars for {len(diverse_exs)} wisdoms")
    print(f"  math/sci -> hygiene; others -> cases Turn 1 + audit Turn 2")

    sample = json.loads((CACHE / "sample_100.json").read_text(encoding="utf-8"))[: args.n]

    client = create_client()
    t0 = time.time()
    new = hit = hygiene_ct = combo_ct = 0

    for i, p in enumerate(sample):
        pid = p["problem_id"]
        if pid in answers:
            hit += 1
            continue
        dom = p.get("domain", "?")
        diff = p.get("difficulty", "?")
        problem = p.get("description", "")

        # Math/science: hygiene bypass
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
            hygiene_ct += 1
        else:
            # v16 2-turn path
            key = f"{dom}__{diff}"
            struct = structures.get(key, {"attention_priors": []})
            priors = struct.get("attention_priors", [])
            priors_fmt = format_priors(priors)

            sel_ids = selections.get(pid, [])[: args.max_wisdoms]
            wisdom_entries = [lib_by_id[sid] for sid in sel_ids if sid in lib_by_id]
            wisdom_entries = [w for w in wisdom_entries if w["id"] in diverse_exs]
            if not wisdom_entries:
                wisdom_entries = [lib_by_id[sid] for sid in selections.get(pid, [])[:2]
                                  if sid in lib_by_id]

            # Turn 1: draft with cases
            if pid in drafts:
                draft = drafts[pid]
            else:
                w_blocks = []
                for w in wisdom_entries:
                    pre_mined = diverse_exs.get(w["id"], [])
                    same_dom_ex = build_same_domain_exemplar(
                        pid, dom, sample, prob_emb, pid_to_emb_idx, v13, ours)
                    cross = [e for e in pre_mined if e.get("domain") != dom][:3]
                    if len(cross) < 3:
                        cross += [e for e in pre_mined if e not in cross][:3 - len(cross)]
                    w_blocks.append(format_wisdom_with_cases(w, cross, same_dom_ex))
                wisdom_case_block = "\n\n---\n\n".join(w_blocks) if w_blocks else "  (无 wisdom 激活)"

                try:
                    r1 = _generate_with_retry(client, EXECUTE_V15.format(
                        priors_block=priors_fmt, wisdom_case_block=wisdom_case_block,
                        problem=problem), max_tokens=1000, temperature=0.3)
                    draft = r1["text"].strip()
                    drafts[pid] = draft
                except Exception as e:
                    print(f"  [err draft {pid}]: {e}")
                    continue

            # Turn 2: audit + revise (brief wisdom, no cases repeat)
            wisdom_brief = format_wisdom_brief(wisdom_entries)
            try:
                r2 = _generate_with_retry(client, REFLECT_PROMPT.format(
                    problem=problem, draft=draft,
                    priors_block=priors_fmt, wisdom_brief=wisdom_brief),
                    max_tokens=1100, temperature=0.3)
                answers[pid] = r2["text"].strip()
            except Exception as e:
                print(f"  [err reflect {pid}]: {e}")
                continue
            combo_ct += 1

        new += 1
        if new % 10 == 0:
            cache_save(answers_path, answers)
            cache_save(drafts_path, drafts)
            print(f"  [{args.variant}] {i+1}/{len(sample)} hyg={hygiene_ct} combo={combo_ct} {time.time()-t0:.0f}s")

    cache_save(answers_path, answers)
    cache_save(drafts_path, drafts)
    print(f"\n  [{args.variant}] done: hyg={hygiene_ct} combo={combo_ct} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
