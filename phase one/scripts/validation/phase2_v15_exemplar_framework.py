"""
v15-exemplar: wisdom with diverse exemplars + on-the-fly same-domain case.

Design (per user):
1. Pre-mined (offline via build_diverse_exemplars_v15.py):
   For each wisdom W, 3 MAXIMALLY DIVERSE cross-domain exemplars
   (law-school approach: see principle in very different surface forms)

2. On-the-fly (runtime, for problem P in domain D):
   Retrieve 1 additional exemplar from the SAME domain as P
   (domain-transfer bridge: "and here's how this principle looks in YOUR field")

So for each selected wisdom, LLM sees 4 total cases:
  - 3 cross-domain (for abstraction)
  - 1 same-domain (for concretization)

Core hypothesis: cases > philosophy alone. Principle alone = 法条无判例 = too ambiguous.
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
STRUCTURES_DIR = CACHE / "structures"
WISDOM_PATH = CACHE / "wisdom_library.json"
SELECTIONS_PATH = CACHE / "phase2_v3_selections.json"
EXEMPLARS_PATH = CACHE / "wisdom_diverse_exemplars.json"
EMB_PATH = CACHE / "signal_embeddings.npz"

V13_REFLECT = ANSWERS_DIR / "phase2_v13_reflect_answers.json"
OURS_27 = ANSWERS_DIR / "ours_27_answers.json"

MATH_SCI = {"mathematics", "science"}


EXECUTE_V15 = """你要解决下面的问题。

在回答前，我给你 2 条 wisdom 原则。为了让你真正理解每条原则的 operational 意义，每条都配备了**判例集**：3 个跨越不同领域的案例（帮你抽象出不变原则）+ 1 个和本题同领域的案例（帮你落回本题上下文）。

{priors_block}

---

## Wisdom + 判例集

{wisdom_case_block}

---

## 你要解决的问题
{problem}

## 要求
- 判例是**理解原则的参考**，不要机械套用任一个的具体方案
- 先识别本题激活了哪条原则（从判例模式看出来，不要只看 wisdom 文字）
- 再借鉴判例里**处理同类结构的思考方式**
- 答出针对本题细节的具体解答
- 不超过 600 字

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


def format_priors(priors):
    return "## 类别 attention priors\n" + (
        "\n".join(f"  {i+1}. {p}" for i, p in enumerate(priors)) or "  (无)")


def format_wisdom_with_cases(w_entry, diverse_exs, extra_ex_sd):
    """Format one wisdom + 3 cross-domain + 1 same-domain exemplars."""
    lines = [f"### {w_entry['aphorism']}  (*{w_entry.get('source', '?')}*)"]
    lines.append(f"   {w_entry.get('unpacked_for_llm', '')[:200]}")
    lines.append("")
    lines.append("**跨域判例（看差异抽象出不变原则）：**")
    for i, e in enumerate(diverse_exs, 1):
        lines.append(f"\n  *案例 {i} — [{e['domain']}] (pid={e['pid']})*")
        lines.append(f"  问题: {e['problem_sketch'][:250]}...")
        lines.append(f"  为什么适用: {e['why_applies']}")
        lines.append(f"  参考答案摘要: {e['answer_snippet'][:400]}...")
    if extra_ex_sd:
        lines.append(f"\n**同领域判例（帮你落回本题上下文）：**")
        lines.append(f"\n  *案例 4 — [{extra_ex_sd['domain']}] (pid={extra_ex_sd['pid']})*")
        lines.append(f"  问题: {extra_ex_sd['problem_sketch'][:250]}...")
        lines.append(f"  参考答案摘要: {extra_ex_sd['answer_snippet'][:400]}...")
    return "\n".join(lines)


def build_same_domain_exemplar(pid: str, dom: str, sample: List,
                                prob_emb: np.ndarray, pid_to_emb_idx: Dict,
                                v13: Dict, ours: Dict) -> Dict:
    """Find the most-similar problem in the same domain (excluding self).
    Use its best answer."""
    if pid not in pid_to_emb_idx:
        return None
    qv = prob_emb[pid_to_emb_idx[pid]]
    # Score all same-domain problems
    same_dom_pids = [p["problem_id"] for p in sample
                      if p.get("domain") == dom and p["problem_id"] != pid]
    best = (-1.0, None)
    for other_pid in same_dom_pids:
        if other_pid not in pid_to_emb_idx:
            continue
        sim = float(qv @ prob_emb[pid_to_emb_idx[other_pid]])
        if sim > best[0]:
            best = (sim, other_pid)
    if best[1] is None:
        return None
    other_pid = best[1]
    info = next(p for p in sample if p["problem_id"] == other_pid)
    ans_src = ours.get(other_pid) if dom in MATH_SCI else v13.get(other_pid)
    ans_src = ans_src or v13.get(other_pid) or ours.get(other_pid) or ""
    return {
        "pid": other_pid,
        "domain": dom,
        "problem_sketch": info.get("description", "")[:350],
        "answer_snippet": ans_src[:700],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="phase2_v15_exemplar")
    ap.add_argument("--base", default="orient_hybrid")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-wisdoms", type=int, default=2,
                    help="max wisdoms per problem (2 to keep context manageable)")
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
    diverse_exs = cache_load(EXEMPLARS_PATH)
    if not diverse_exs:
        print("ERROR: run build_diverse_exemplars_v15.py first to build "
              "wisdom_diverse_exemplars.json")
        return
    print(f"  wisdom: {len(library)}  diverse exemplars for {len(diverse_exs)} wisdoms")

    v13 = json.loads(V13_REFLECT.read_text(encoding="utf-8"))
    ours = json.loads(OURS_27.read_text(encoding="utf-8"))

    emb = np.load(EMB_PATH, allow_pickle=True)
    prob_emb = emb["problem_emb"]
    prob_ids_emb = emb["problem_ids"].tolist()
    pid_to_emb_idx = {pid: i for i, pid in enumerate(prob_ids_emb)}

    sample = json.loads((CACHE / "sample_100.json").read_text(encoding="utf-8"))[: args.n]

    client = create_client()
    t0 = time.time()
    new = hit = 0

    for i, p in enumerate(sample):
        pid = p["problem_id"]
        if pid in answers:
            hit += 1
            continue
        dom = p.get("domain", "?")
        diff = p.get("difficulty", "?")
        key = f"{dom}__{diff}"
        struct = structures.get(key, {"attention_priors": []})
        priors = struct.get("attention_priors", [])

        # Select wisdoms (reuse v3 selections, cap at max-wisdoms for context)
        sel_ids = selections.get(pid, [])[: args.max_wisdoms]
        wisdom_entries = [lib_by_id[sid] for sid in sel_ids if sid in lib_by_id]
        # Filter to those with pre-mined exemplars
        wisdom_entries = [w for w in wisdom_entries if w["id"] in diverse_exs]
        if not wisdom_entries:
            # fallback: first 2 from selections even without exemplars — or use prior only
            wisdom_entries = [lib_by_id[sid] for sid in selections.get(pid, [])[:2]
                              if sid in lib_by_id]

        # Build wisdom + case blocks
        w_blocks = []
        for w in wisdom_entries:
            pre_mined = diverse_exs.get(w["id"], [])
            same_dom_ex = build_same_domain_exemplar(
                pid, dom, sample, prob_emb, pid_to_emb_idx, v13, ours)
            # Filter pre-mined to those whose domain != current (to maximize diversity)
            cross = [e for e in pre_mined if e.get("domain") != dom][:3]
            if len(cross) < 3:
                # Pad with whatever pre-mined is left
                cross += [e for e in pre_mined if e not in cross][:3 - len(cross)]
            w_blocks.append(format_wisdom_with_cases(w, cross, same_dom_ex))
        wisdom_case_block = "\n\n---\n\n".join(w_blocks) if w_blocks else "  (无 wisdom 激活)"

        prompt = EXECUTE_V15.format(
            priors_block=format_priors(priors),
            wisdom_case_block=wisdom_case_block,
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
            print(f"  [{args.variant}] {i+1}/{len(sample)} {time.time()-t0:.0f}s")

    cache_save(answers_path, answers)
    print(f"\n  [{args.variant}] done: {new} new, {hit} cached ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
