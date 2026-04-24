"""Exp 25 — Multi-family subset labeling + Cohen's κ.

Directly addresses reviewer objection #3 from the harsh review:
  "Subset labels in Exp 17/24 are done by Claude without audit;
   no inter-annotator reliability, circular: Claude is both
   labeler and judge."

Fix: label the same 50 pids on 4 subsets with 3 cheap-tier families
(Gemini-3-flash + Claude-Haiku-4.5 + GPT-5.4-mini) in parallel.
Compute pairwise Cohen's κ + majority-vote label.

Output:
  phase four/autonomous/exp25_multifamily_labels.json
  phase four/autonomous/exp25_kappa_log.json

Then rerun Exp 24 on majority-vote labels and compare.
"""

import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
sys.path.insert(0, str(PROJECT / "phase four"))

from model_router import cheap_panel
from llm_client import parse_json_from_llm

CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
LABELS_OUT = AUTO_DIR / "exp25_multifamily_labels.json"
KAPPA_OUT = AUTO_DIR / "exp25_kappa_log.json"

PARALLEL = 8

LABEL_PROMPT = """判断下面问题在 4 个维度上是否属于对应子集 (pure problem-level property, wisdom-agnostic):

== 问题 ==
{problem}

== 4 个子集 ==
1. S_reframe: 原题表述**不足**以直接决定好解法（约束隐含、视角需重构才能推进）。对比面：题目明确到"照字面做"就够了 → 不属于。
2. S_delta:   存在**真实**内容增量空间（好答案明显比烂答案多出实质内容）。对比面：格式对就够的简单问答 → 不属于。
3. S_wisdom:  需要**策略取舍 / 优先级 / 风险下注**。对比面：有标准正解的题 → 不属于。
4. S_anti:    对常见**坏模式**高度敏感（容易被空泛高举、过度 taxonomy、避开真实 tradeoff 糊弄过去）。对比面：很难被套路糊弄的题 → 不属于。

注意：多数问题可能属于多个子集，也可能都不属于。按字面判断，不放松。

== 输出 JSON（不要代码块）==
{{"S_reframe": true/false, "S_delta": true/false,
  "S_wisdom": true/false, "S_anti": true/false}}
"""


def label_one(client, problem):
    try:
        r = client.generate(LABEL_PROMPT.format(problem=problem),
                            max_tokens=250, temperature=0.0)
        v = parse_json_from_llm(r["text"])
        return {k: bool(v.get(k, False))
                for k in ("S_reframe", "S_delta", "S_wisdom", "S_anti")}
    except Exception as e:
        return {"err": str(e)[:60]}


def label_all_by_family(clients, problems):
    """Returns {family_name: {pid: {subset: bool}}}"""
    out = {c.family: {} for c in clients}
    total = len(problems) * len(clients)
    done = 0
    t0 = time.time()

    def task(client, p):
        return client.family, p["problem_id"], label_one(client, p["description"])

    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(task, c, p) for c in clients for p in problems]
        for f in as_completed(futs):
            fam, pid, lab = f.result()
            out[fam][pid] = lab
            done += 1
            if done % 40 == 0:
                dt = time.time() - t0
                eta = dt / done * (total - done)
                print(f"  {done}/{total}  ({dt:.0f}s elapsed, {eta:.0f}s ETA)")
    return out


def cohen_kappa(labels_a, labels_b):
    """Cohen's κ for binary labels. Input: two dicts pid→bool."""
    pids = sorted(set(labels_a) & set(labels_b))
    if not pids: return None
    n = len(pids)
    agree = sum(1 for p in pids if labels_a[p] == labels_b[p])
    po = agree / n
    # expected agreement by chance
    pa_pos = sum(labels_a[p] for p in pids) / n
    pb_pos = sum(labels_b[p] for p in pids) / n
    pe = pa_pos * pb_pos + (1 - pa_pos) * (1 - pb_pos)
    if abs(1 - pe) < 1e-9: return 1.0
    return (po - pe) / (1 - pe)


def majority_vote(families_labels):
    """Given {family: {pid: {subset: bool}}}, produce {pid: {subset: majority_bool}}"""
    mv = {}
    fams = list(families_labels.keys())
    pids = set()
    for fam, lbs in families_labels.items():
        pids.update(lbs.keys())
    for pid in pids:
        res = {}
        for sub in ("S_reframe", "S_delta", "S_wisdom", "S_anti"):
            votes = []
            for fam in fams:
                lab = families_labels[fam].get(pid, {})
                if isinstance(lab, dict) and sub in lab:
                    votes.append(lab[sub])
            if not votes: continue
            n_true = sum(1 for v in votes if v)
            # Strict: require >= 2/3 for True
            res[sub] = n_true > len(votes) / 2
        mv[pid] = res
    return mv


def main():
    problems = json.loads((CACHE / "sample_holdout_50.json").read_text(encoding="utf-8"))
    problems = [p for p in problems if "description" in p]

    clients = cheap_panel()
    print(f"Cheap panel: {[c.model for c in clients]}\n")

    print(f"[1/3] Labeling {len(problems)} pids × {len(clients)} families...")
    per_family = label_all_by_family(clients, problems)

    print(f"\n[2/3] Per-family subset counts:")
    for fam, lbs in per_family.items():
        counts = {sub: sum(1 for lab in lbs.values()
                           if isinstance(lab, dict) and lab.get(sub))
                   for sub in ("S_reframe", "S_delta", "S_wisdom", "S_anti")}
        err = sum(1 for lab in lbs.values() if "err" in lab)
        print(f"  {fam:15s} {counts}  err={err}")

    # Pairwise κ per subset
    print(f"\n[3/3] Pairwise Cohen's κ per subset:")
    fams = sorted(per_family.keys())
    pairs = [(fams[i], fams[j]) for i in range(len(fams)) for j in range(i+1, len(fams))]
    kappa_table = {}
    for sub in ("S_reframe", "S_delta", "S_wisdom", "S_anti"):
        kappa_table[sub] = {}
        for a, b in pairs:
            la = {pid: lab[sub] for pid, lab in per_family[a].items()
                   if isinstance(lab, dict) and sub in lab}
            lb = {pid: lab[sub] for pid, lab in per_family[b].items()
                   if isinstance(lab, dict) and sub in lab}
            k = cohen_kappa(la, lb)
            kappa_table[sub][f"{a}_vs_{b}"] = k
        ks = kappa_table[sub]
        mean_k = sum(v for v in ks.values() if v is not None) / max(1, len(ks))
        print(f"  {sub:12s} mean κ={mean_k:+.2f}  "
              f"{', '.join(f'{k}={v:+.2f}' for k, v in ks.items())}")

    # Majority vote labels
    mv = majority_vote(per_family)
    mv_counts = {sub: sum(1 for lab in mv.values() if lab.get(sub))
                  for sub in ("S_reframe", "S_delta", "S_wisdom", "S_anti")}
    print(f"\n  Majority-vote subset sizes: {mv_counts}")

    # Save
    LABELS_OUT.write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "per_family": per_family,
        "majority_vote": mv,
    }, ensure_ascii=False, indent=2))
    KAPPA_OUT.write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kappa_table": kappa_table,
        "majority_subset_sizes": mv_counts,
    }, ensure_ascii=False, indent=2))
    print(f"\nSaved: {LABELS_OUT.name}, {KAPPA_OUT.name}")


if __name__ == "__main__":
    main()
