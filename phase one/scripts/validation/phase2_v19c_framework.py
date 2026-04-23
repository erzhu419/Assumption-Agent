"""
v19c: Double-Path + Pick Architecture.

Turn 1a: Solve assuming object-level (no frame wisdoms, just priors+hygiene)
Turn 1b: Solve assuming paradigm-level (full v16-style with cases)
Turn 2:  Judge picks better draft for this specific problem

Hypothesis: generate 2 different "framings" explicitly, let LLM external-judge
which one better matches the problem's true frame. No single prompt needs to
carry both possibilities.

Math/sci bypass to v12c hygiene (1-pass).
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
from cached_framework import _generate_with_retry, BASELINE_PROMPT
from phase2_v12_framework import (EXECUTE_MATH, EXECUTE_SCIENCE, format_priors,
                                   MATH_SCI)
from phase2_v15_exemplar_framework import (EXECUTE_V15, build_same_domain_exemplar,
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


# Object-level path: bare + priors
EXECUTE_OBJ = """你是一位严谨的问题解决者。针对下面问题给出直接、具体的解答，优先输出对象层的精确答案（公式/算法/实现/具体方案）。

## 本类别 attention priors
{priors_block}

## 问题
{problem}

## 要求
- 直接给解答，不要反问"真正要解什么"
- 如果有明确计算/证明/实现要求，按标准做法展开
- 不超过 550 字

开始：
"""


# Paradigm-level path: full v16-style with cases
# (reuse EXECUTE_V15)


PICK_PROMPT = """你是答案评审员。下面同一问题有 2 个候选答案，框架完全不同。

## 问题
{problem}

## 答案 A (object-level 路径)
{answer_a}

## 答案 B (paradigm-level 路径，带 stakeholder/cross-domain 分析)
{answer_b}

## 评审任务
1. 问题真正需要的是 object-level 还是 paradigm-level 答案？
2. 选 A 或 B (哪个对这个具体问题更合适)
3. 对所选答案做必要修订，输出最终版本

输出 JSON（不要代码块）：
{{
  "pick": "A" | "B",
  "reason": "为什么选这个 (20-40 字)",
  "final_answer": "最终答案（可以是所选答案的小修订，不超过 650 字）"
}}
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
    ap.add_argument("--variant", default="phase2_v19c")
    ap.add_argument("--base", default="orient_hybrid")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-wisdoms", type=int, default=2)
    args = ap.parse_args()

    answers_path = ANSWERS_DIR / f"{args.variant}_answers.json"
    path_a = ANSWERS_DIR / f"{args.variant}_path_a.json"
    path_b = ANSWERS_DIR / f"{args.variant}_path_b.json"
    picks_path = ANSWERS_DIR / f"{args.variant}_picks.json"
    struct_path = STRUCTURES_DIR / f"{args.variant}_structures.json"
    answers = cache_load(answers_path)
    drafts_a = cache_load(path_a)
    drafts_b = cache_load(path_b)
    picks = cache_load(picks_path)

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
            key = f"{dom}__{diff}"
            struct = structures.get(key, {"attention_priors": []})
            priors = struct.get("attention_priors", [])

            # Turn 1a: object-level path
            if pid in drafts_a:
                a_ans = drafts_a[pid]
            else:
                try:
                    ra = _generate_with_retry(client, EXECUTE_OBJ.format(
                        priors_block=format_priors(priors), problem=problem
                    ), max_tokens=900, temperature=0.3)
                    a_ans = ra["text"].strip()
                    drafts_a[pid] = a_ans
                except Exception as e:
                    print(f"  [err A {pid}] {e}")
                    continue

            # Turn 1b: paradigm-level path (v16-style)
            if pid in drafts_b:
                b_ans = drafts_b[pid]
            else:
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

                try:
                    rb = _generate_with_retry(client, EXECUTE_V15.format(
                        priors_block=format_priors(priors),
                        wisdom_case_block=wisdom_case_block,
                        problem=problem
                    ), max_tokens=1000, temperature=0.3)
                    b_ans = rb["text"].strip()
                    drafts_b[pid] = b_ans
                except Exception as e:
                    print(f"  [err B {pid}] {e}")
                    continue

            # Turn 2: pick
            if pid in picks:
                pick_data = picks[pid]
            else:
                try:
                    rpick = _generate_with_retry(client, PICK_PROMPT.format(
                        problem=problem, answer_a=a_ans[:900], answer_b=b_ans[:900]
                    ), max_tokens=1200, temperature=0.2)
                    pick_data = parse_json_from_llm(rpick["text"])
                    if "final_answer" not in pick_data:
                        pick_data = {"pick": "B", "reason": "parse fallback", "final_answer": b_ans}
                    picks[pid] = pick_data
                except Exception as e:
                    print(f"  [err pick {pid}] {e}")
                    pick_data = {"pick": "B", "reason": "error", "final_answer": b_ans}

            answers[pid] = pick_data.get("final_answer", b_ans)
            full += 1

        new += 1
        if new % 10 == 0:
            cache_save(answers_path, answers)
            cache_save(path_a, drafts_a)
            cache_save(path_b, drafts_b)
            cache_save(picks_path, picks)
            print(f"  [{args.variant}] {i+1}/{len(sample)} hyg={hyg} full={full} {time.time()-t0:.0f}s")

    cache_save(answers_path, answers)
    cache_save(path_a, drafts_a)
    cache_save(path_b, drafts_b)
    cache_save(picks_path, picks)
    print(f"\n  [{args.variant}] done: hyg={hyg} full={full} ({time.time()-t0:.0f}s)")

    # Pick distribution
    from collections import Counter
    pc = Counter(p.get("pick", "?") for p in picks.values())
    print(f"  Pick distribution: {dict(pc)}")


if __name__ == "__main__":
    main()
