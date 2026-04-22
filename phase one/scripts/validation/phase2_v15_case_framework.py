"""
v15-case: pure case-based reasoning — no wisdom text, only retrieved exemplars.

User's insight: giving wisdom without cases is like civil law without precedents
— too much ambiguity. Test the opposite extreme: give ONLY precedents, no
statute text.

For each problem P:
  1. Find 3 most similar problems from sample_100 (by problem-embedding cosine)
  2. For each exemplar, use the 'best-so-far' answer:
     - math/science: ours_27 (Self-Discover) — strongest on these
     - others: phase2_v13_reflect — strongest on soft
  3. Inject as few-shot examples, then solve the current problem

No priors, no triggers, no wisdom text. Pure analogy-from-cases.
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


CACHE = PROJECT.parent / "phase two" / "analysis" / "cache"
ANSWERS_DIR = CACHE / "answers"
EMB_PATH = CACHE / "signal_embeddings.npz"
V13_REFLECT = ANSWERS_DIR / "phase2_v13_reflect_answers.json"
OURS_27 = ANSWERS_DIR / "ours_27_answers.json"

MATH_SCI = {"mathematics", "science"}


EXECUTE_CASE = """# 解决下面的问题。

下面是 3 个类似问题和它们的高质量解答——作为思路参考。注意：不要机械套用某个例子的具体方案，而是**借鉴它们处理同类问题的思考方式**（识别什么、假设什么、如何权衡）。

## 参考案例

### 案例 1
**问题**: {ex1_problem}

**参考解答**:
{ex1_answer}

---

### 案例 2
**问题**: {ex2_problem}

**参考解答**:
{ex2_answer}

---

### 案例 3
**问题**: {ex3_problem}

**参考解答**:
{ex3_answer}

---

## 你要解决的问题
{problem}

## 要求
- 不要照抄任何案例的结构或措辞
- 借鉴处理思路，针对本题的具体细节给出解答
- 语言精炼，不超过 600 字

开始：
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


def select_best_answer(pid: str, domain: str, v13: Dict, ours: Dict) -> str:
    """Pick the strongest answer for this exemplar based on its domain."""
    if domain in MATH_SCI:
        return ours.get(pid, v13.get(pid, ""))
    return v13.get(pid, ours.get(pid, ""))


def truncate_for_context(text: str, max_chars: int = 900) -> str:
    """Trim long exemplar answers to fit in-context."""
    if len(text) <= max_chars:
        return text
    # Find nearest paragraph boundary before max_chars
    import re
    cut = text[:max_chars]
    m = list(re.finditer(r'[。！？\n](?=\S|$)', cut))
    if m and m[-1].end() >= max_chars * 0.7:
        return cut[:m[-1].end()].rstrip() + "\n[...]"
    return cut.rstrip() + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="phase2_v15_case")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=3, help="number of exemplars")
    args = ap.parse_args()

    answers_path = ANSWERS_DIR / f"{args.variant}_answers.json"
    exemplars_path = CACHE / f"{args.variant}_exemplars.json"
    answers = cache_load(answers_path)
    exemplar_map = cache_load(exemplars_path)

    sample = json.loads((CACHE / "sample_100.json").read_text(encoding="utf-8"))[: args.n]
    pid_to_idx = {p["problem_id"]: i for i, p in enumerate(sample)}
    pid_to_info = {p["problem_id"]: p for p in sample}

    # Load problem embeddings
    emb = np.load(EMB_PATH, allow_pickle=True)
    prob_emb = emb["problem_emb"]
    prob_ids_emb = emb["problem_ids"].tolist()
    pid_to_emb_idx = {pid: i for i, pid in enumerate(prob_ids_emb)}

    # Best answers per source
    v13 = json.loads(V13_REFLECT.read_text(encoding="utf-8"))
    ours = json.loads(OURS_27.read_text(encoding="utf-8"))

    print(f"  v15-case: k={args.k} exemplars per problem")
    print(f"  exemplar sources: ours_27 for math/sci, v13_reflect for others")

    client = create_client()
    t0 = time.time()
    new = hit = 0

    for i, p in enumerate(sample):
        pid = p["problem_id"]
        if pid in answers:
            hit += 1
            continue
        if pid not in pid_to_emb_idx:
            print(f"  [skip] {pid}: no embedding")
            continue

        # Find top-K similar problems (exclude self)
        if pid in exemplar_map:
            sel_ids = exemplar_map[pid]
        else:
            qv = prob_emb[pid_to_emb_idx[pid]]
            sims = prob_emb @ qv
            order = np.argsort(-sims)
            sel_ids = []
            for j in order:
                other_pid = prob_ids_emb[j]
                if other_pid == pid:
                    continue
                if other_pid not in pid_to_info:
                    continue
                sel_ids.append(other_pid)
                if len(sel_ids) >= args.k:
                    break
            exemplar_map[pid] = sel_ids

        # Build prompt
        exemplars = []
        for eid in sel_ids:
            e_info = pid_to_info[eid]
            e_prob = e_info.get("description", "")[:400]  # truncate long problem text
            e_dom = e_info.get("domain", "?")
            e_ans = select_best_answer(eid, e_dom, v13, ours)
            e_ans = truncate_for_context(e_ans, max_chars=900)
            exemplars.append({"problem": e_prob, "answer": e_ans})

        prompt = EXECUTE_CASE.format(
            ex1_problem=exemplars[0]["problem"],
            ex1_answer=exemplars[0]["answer"],
            ex2_problem=exemplars[1]["problem"],
            ex2_answer=exemplars[1]["answer"],
            ex3_problem=exemplars[2]["problem"],
            ex3_answer=exemplars[2]["answer"],
            problem=p.get("description", ""),
        )

        try:
            resp = _generate_with_retry(client, prompt, max_tokens=1000, temperature=0.3)
            answers[pid] = resp["text"].strip()
        except Exception as e:
            print(f"  [err] {pid}: {e}")
            continue

        new += 1
        if new % 10 == 0:
            cache_save(answers_path, answers)
            cache_save(exemplars_path, exemplar_map)
            print(f"  [{args.variant}] {i+1}/{len(sample)} {time.time()-t0:.0f}s")

    cache_save(answers_path, answers)
    cache_save(exemplars_path, exemplar_map)
    print(f"\n  [{args.variant}] done: {new} new, {hit} cached ({time.time()-t0:.0f}s)")

    # Stats: most frequently referenced exemplars
    from collections import Counter
    refs = Counter()
    for exs in exemplar_map.values():
        for e in exs:
            refs[e] += 1
    print("\n  Top 10 most-referenced exemplars:")
    for eid, n in refs.most_common(10):
        if eid in pid_to_info:
            dom = pid_to_info[eid].get("domain", "?")
            print(f"    {n:>3}x  [{eid}] {dom}")


if __name__ == "__main__":
    main()
