"""
v19b: Problem Rewriting Architecture.

Turn 0 REWRITES the problem statement to reflect its true frame. Turn 1 solves
the rewritten problem (with original kept for grounding). Turn 2 audit.

Hypothesis: instead of adding frame hints alongside original problem (which
keeps LLM anchored to the original framing), transforming the input itself
forces LLM into new solution mode.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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
from phase2_v15_exemplar_framework import (build_same_domain_exemplar,
                                            format_wisdom_with_cases)


CACHE = PROJECT.parent / "phase two" / "analysis" / "cache"
ANSWERS_DIR = CACHE / "answers"
STRUCTURES_DIR = CACHE / "structures"
WISDOM_PATH = CACHE / "wisdom_library.json"
SELECTIONS_PATH = CACHE / "phase2_v3_selections.json"
EXEMPLARS_PATH = CACHE / "wisdom_diverse_exemplars.json"
EMB_PATH = CACHE / "signal_embeddings.npz"
V13_REFLECT = ANSWERS_DIR / "phase2_v13_reflect_answers.json"
OURS_27 = ANSWERS_DIR / "ours_27_answers.json"


REWRITE_PROMPT = """# 问题重写任务

你是问题分析专家。读下面问题，输出一个**更精确的重写版本**，让真正需要被回答的点更凸显。

## 原始问题
{problem}

## 重写规则
1. 若原问题就是清晰的对象层（证明、计算、实现），保留原貌或做 minimal 改写
2. 若原问题 implicit 涉及 stakeholder / 监管 / 投入 / 范式选择等，把这些**显式化**到重写版本里
3. 不要改变问题的核心要求，只让"真正该答什么"更清楚

## 输出 JSON（不要代码块）
{{
  "rewritten_problem": "重写后的问题陈述（完整段落，可比原文稍长）",
  "what_changed": "和原问题相比，凸显了什么 (20-50 字)"
}}
"""


EXECUTE_V19B = """# 你要解决一个问题。

## 原问题（用于对照真实性）
{problem}

## 重写版本（Turn 0 分析后，显式化了该问题真正需要回答的维度）
{rewritten_problem}

*分析点评*: {what_changed}

## 本类别 attention priors
{priors_block}

## Wisdom + 判例集
{wisdom_case_block}

## 要求
- **针对重写版本作答**，但不要脱离原问题的具体事实
- 若重写版本添加的 stakeholder/criteria 显然存在，就按那个去答
- 若重写版本是 minimal 改写，就按原问题直接答
- 不超过 600 字

开始：
"""


REFLECT_PROMPT_V19B = """草稿已产。

## 原问题
{problem}

## 重写版本
{rewritten_problem}

## 草稿
{draft}

## 自检
1. 草稿是针对 "原问题的字面表述" 还是 "重写版本显式化的真实要求" 作答？
2. 有没有陷入原问题的表面 framing 里？
3. 草稿里有 1-2 条可以 integrate 得更好的 insight 吗？

输出最终答案（不要 audit 过程），不超过 650 字：
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
    ap.add_argument("--variant", default="phase2_v19b")
    ap.add_argument("--base", default="orient_hybrid")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-wisdoms", type=int, default=2)
    args = ap.parse_args()

    answers_path = ANSWERS_DIR / f"{args.variant}_answers.json"
    rewrites_path = ANSWERS_DIR / f"{args.variant}_rewrites.json"
    drafts_path = ANSWERS_DIR / f"{args.variant}_drafts.json"
    struct_path = STRUCTURES_DIR / f"{args.variant}_structures.json"
    answers = cache_load(answers_path)
    rewrites = cache_load(rewrites_path)
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

    sample = json.loads((CACHE / "sample_100.json").read_text(encoding="utf-8"))[: args.n]

    from llm_client import parse_json_from_llm

    client = create_client()
    t0 = time.time()
    new = hit = hyg = full = 0

    for i, p in enumerate(sample):
        pid = p["problem_id"]
        if pid in answers:
            hit += 1
            continue
        dom = p.get("domain", "?")
        diff = p.get("difficulty", "?")
        problem = p.get("description", "")

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
                print(f"  [err {pid}] {e}")
                continue
            hyg += 1
        else:
            # Turn 0: rewrite
            if pid in rewrites:
                rw = rewrites[pid]
            else:
                try:
                    r0 = _generate_with_retry(client, REWRITE_PROMPT.format(problem=problem),
                                              max_tokens=600, temperature=0.3)
                    rw = parse_json_from_llm(r0["text"])
                    if "rewritten_problem" not in rw:
                        raise ValueError("no rewritten_problem")
                    rewrites[pid] = rw
                except Exception as e:
                    print(f"  [err rewrite {pid}] {e}")
                    rw = {"rewritten_problem": problem, "what_changed": "fallback"}
                    rewrites[pid] = rw

            # Turn 1: solve rewritten
            key = f"{dom}__{diff}"
            struct = structures.get(key, {"attention_priors": []})
            priors = struct.get("attention_priors", [])

            sel_ids = selections.get(pid, [])[: args.max_wisdoms]
            wisdom_entries = [lib_by_id[sid] for sid in sel_ids if sid in lib_by_id]
            wisdom_entries = [w for w in wisdom_entries if w["id"] in diverse_exs]

            w_blocks = []
            for w in wisdom_entries:
                pre_mined = diverse_exs.get(w["id"], [])
                same_dom_ex = build_same_domain_exemplar(
                    pid, dom, sample, prob_emb, pid_to_emb_idx, v13, ours)
                cross = [e for e in pre_mined if e.get("domain") != dom][:3]
                if len(cross) < 3:
                    cross += [e for e in pre_mined if e not in cross][:3 - len(cross)]
                w_blocks.append(format_wisdom_with_cases(w, cross, same_dom_ex))
            wisdom_case_block = "\n\n---\n\n".join(w_blocks) if w_blocks else "  (无)"

            if pid in drafts:
                draft = drafts[pid]
            else:
                try:
                    r1 = _generate_with_retry(client, EXECUTE_V19B.format(
                        problem=problem,
                        rewritten_problem=rw.get("rewritten_problem", problem),
                        what_changed=rw.get("what_changed", ""),
                        priors_block=format_priors(priors),
                        wisdom_case_block=wisdom_case_block,
                    ), max_tokens=1000, temperature=0.3)
                    draft = r1["text"].strip()
                    drafts[pid] = draft
                except Exception as e:
                    print(f"  [err draft {pid}] {e}")
                    continue

            # Turn 2: audit
            try:
                r2 = _generate_with_retry(client, REFLECT_PROMPT_V19B.format(
                    problem=problem,
                    rewritten_problem=rw.get("rewritten_problem", problem),
                    draft=draft
                ), max_tokens=1100, temperature=0.3)
                answers[pid] = r2["text"].strip()
            except Exception as e:
                print(f"  [err audit {pid}] {e}")
                continue
            full += 1

        new += 1
        if new % 10 == 0:
            cache_save(answers_path, answers)
            cache_save(rewrites_path, rewrites)
            cache_save(drafts_path, drafts)
            print(f"  [{args.variant}] {i+1}/{len(sample)} hyg={hyg} full={full} {time.time()-t0:.0f}s")

    cache_save(answers_path, answers)
    cache_save(rewrites_path, rewrites)
    cache_save(drafts_path, drafts)
    print(f"\n  [{args.variant}] done: hyg={hyg} full={full} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
