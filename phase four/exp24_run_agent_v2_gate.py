"""Exp 24 — Actually run agent's v2 gate (from Exp 22 Phase C).

Agent's v2 gate spec requires 4 pid-level subsets:
  S_reframe: pids where "原题表述不足以直接决定好解法"
  S_delta:   pids "存在真实内容增量空间"
  S_wisdom:  pids "需要做策略取舍、优先级排序或风险下注"
  S_anti:    pids "对常见坏模式高度敏感"

For each subset we label every held-out pid (50) with Claude — 200 labels total.
Then compute the 4 conditioned components per candidate, apply agent's
combination rule, and compare predicted 3/12 with actual.

Combination rule (from Phase C output):
  antipattern_avoidance MUST PASS
  wisdom_alignment MUST PASS
  reframe_depth OR content_delta PASS AND the other not strongly negative
"""

import json
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
sys.path.insert(0, str(PROJECT / "phase four"))

from claude_proxy_client import ClaudeProxyClient
from gpt5_client import GPT5Client
from llm_client import parse_json_from_llm
from exp21_data_api import list_candidates, candidate_info, per_pid_records

CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
SUBSET_LABELS_PATH = AUTO_DIR / "exp24_subset_labels.json"
OUT_LOG = AUTO_DIR / "exp24_agent_v2_gate_log.json"

PARALLEL = 8


SUBSET_PROMPT = """判断下面问题在 4 个维度上是否属于对应子集 (纯 problem-level 属性，和具体 wisdom 无关):

== 问题 ==
{problem}

== 4 个子集定义 ==

1. S_reframe: **原题表述不足以直接决定好解法**
   (表述模糊、约束隐含、视角需要重构才能推进；"照字面直接做" 不够好)

2. S_delta: **存在真实内容增量空间**
   (好答案明显比烂答案多出实质内容；不是"只要格式对就够了"的题)

3. S_wisdom: **需要做策略取舍、优先级排序或风险下注**
   (有多条备选路径互有利弊；不是有标准正解的题)

4. S_anti: **对常见坏模式高度敏感**
   (很容易被"空泛高举、过度 taxonomy、avoid 真实 tradeoff、一刀切"之类套路糊弄过去)

== 输出 JSON（不要代码块） ==
{{"S_reframe": true/false,
  "S_delta": true/false,
  "S_wisdom": true/false,
  "S_anti": true/false,
  "reasoning": "80-150字，对 4 个子集的 yes/no 各给一句简短理由"}}
"""


def cache_load(p, default=None):
    try: return json.loads(Path(p).read_text(encoding="utf-8"))
    except: return default


# ---------- step 1: label 50 pids on 4 subsets ----------

def label_subsets(client, problems):
    labels = cache_load(SUBSET_LABELS_PATH, default={})
    todo = [p for p in problems if p["problem_id"] not in labels]
    if not todo:
        print(f"All {len(labels)} labels cached")
        return labels

    print(f"Labeling {len(todo)} problems on 4 subsets...")
    def task(p):
        try:
            r = client.generate(SUBSET_PROMPT.format(problem=p["description"]),
                                 max_tokens=500, temperature=0.0)
            v = parse_json_from_llm(r["text"])
            return p["problem_id"], {
                "S_reframe": bool(v.get("S_reframe", False)),
                "S_delta": bool(v.get("S_delta", False)),
                "S_wisdom": bool(v.get("S_wisdom", False)),
                "S_anti": bool(v.get("S_anti", False)),
            }
        except Exception as e:
            return p["problem_id"], {"err": str(e)[:80]}

    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(task, p) for p in todo]
        for f in as_completed(futs):
            pid, lab = f.result()
            labels[pid] = lab
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(todo)} ({time.time()-t0:.0f}s)")
    SUBSET_LABELS_PATH.write_text(json.dumps(labels, ensure_ascii=False, indent=2))
    print(f"  Labels → {SUBSET_LABELS_PATH.name}")
    return labels


# ---------- step 2: text metrics (same as exp21c_complete_gate.py) ----------

def tokenize(text):
    if not text: return []
    out = []
    for chunk in re.findall(r"[A-Za-z0-9_一-鿿]+", text.lower()):
        if re.search(r"[一-鿿]", chunk):
            out.extend(list(chunk))
        else:
            out.append(chunk)
    return out


def char_ngrams(text, n=3):
    c = Counter()
    for i in range(max(0, len(text) - n + 1)):
        c[text[i:i+n]] += 1
    return c


def word_ngrams(tokens, n=1):
    c = Counter()
    for i in range(max(0, len(tokens) - n + 1)):
        c[" ".join(tokens[i:i+n])] += 1
    return c


def cosine(a, b):
    if not a or not b: return 0.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = (sum(v*v for v in a.values())) ** 0.5
    nb = (sum(v*v for v in b.values())) ** 0.5
    if na < 1e-12 or nb < 1e-12: return 0.0
    return dot / (na * nb)


def text_sim(a, b):
    if not a or not b: return 0.0
    ta, tb = tokenize(a), tokenize(b)
    sc = cosine(char_ngrams(a), char_ngrams(b))
    su = cosine(word_ngrams(ta, 1), word_ngrams(tb, 1))
    sbi = cosine(word_ngrams(ta, 2), word_ngrams(tb, 2))
    return 0.3 * sc + 0.4 * su + 0.3 * sbi


def text_dist(a, b):
    return max(0.0, min(1.0, 1.0 - text_sim(a, b)))


def median(vals):
    if not vals: return 0.0
    s = sorted(vals); n = len(s)
    return s[n//2] if n % 2 == 1 else (s[n//2-1] + s[n//2]) / 2.0


# ---------- step 3: conditioned evaluators ----------

def eval_reframe_depth_conditioned(cid, labels):
    rows = per_pid_records(cid)
    dists = []
    for r in rows:
        lab = labels.get(r["pid"], {})
        if not lab.get("S_reframe"): continue
        if r["problem"] and r["ext_what_changed"]:
            dists.append(text_dist(r["problem"], r["ext_what_changed"]))
    med = median(dists)
    return {"median_dist": med, "n": len(dists),
            "passed": med >= 0.30 and len(dists) >= 5}


def eval_content_delta_conditioned(cid, labels):
    rows = per_pid_records(cid)
    dists = []
    for r in rows:
        lab = labels.get(r["pid"], {})
        if not lab.get("S_delta"): continue
        if r["base_answer"] and r["ext_answer"]:
            dists.append(text_dist(r["base_answer"], r["ext_answer"]))
    med = median(dists)
    frac = sum(1 for d in dists if d >= 0.15) / len(dists) if dists else 0.0
    return {"median_dist": med, "fraction_above_0.15": frac, "n": len(dists),
            "passed": med >= 0.25 and frac >= 0.65 and len(dists) >= 5}


def eval_wisdom_alignment_conditioned(cid, labels):
    info = candidate_info(cid)
    unpacked = info["unpacked"]
    if not unpacked: return {"passed": False, "n": 0}
    rows = per_pid_records(cid)
    sims = []
    for r in rows:
        lab = labels.get(r["pid"], {})
        if not lab.get("S_wisdom"): continue
        if r["problem"]:
            sims.append(text_sim(unpacked, r["problem"]))
    if len(sims) < 3:
        return {"passed": False, "n": len(sims), "mean_sim": 0.0}
    mu = sum(sims) / len(sims)
    var = sum((s - mu)**2 for s in sims) / len(sims)
    sigma = var ** 0.5
    strong = sum(1 for s in sims if s > mu)
    # agent's criterion: wisdom must be meaningfully aligned on "strategy choice" pids
    return {"mean_sim": mu, "n": len(sims), "strong_count": strong,
            "sigma": sigma,
            "passed": mu >= 0.08 and strong >= len(sims) * 0.3 and len(sims) >= 5}


def eval_antipattern_avoidance_conditioned(cid, labels):
    rows = per_pid_records(cid)
    rates = []
    for r in rows:
        lab = labels.get(r["pid"], {})
        if not lab.get("S_anti"): continue
        aps = r.get("ext_anti_patterns", [])
        if not aps: continue
        ans = (r.get("ext_answer") or "").lower()
        hits = 0; total = 0
        for ap in aps:
            if not isinstance(ap, str): continue
            total += 1
            key = ap.strip()[:6].lower()
            if key and key in ans:
                hits += 1
        if total:
            rates.append(1.0 - hits / total)
    med = median(rates) if rates else 0.0
    return {"median_avoidance": med, "n": len(rates),
            "passed": med >= 0.65 and len(rates) >= 5}


# ---------- step 4: combination rule + full pipeline ----------

def combine(comps):
    """Agent's combination rule:
    - antipattern_avoidance MUST PASS
    - wisdom_alignment MUST PASS
    - reframe_depth OR content_delta pass AND other not strongly negative"""
    c = comps
    if not c["antipattern_avoidance"]["passed"]: return False
    if not c["wisdom_alignment"]["passed"]: return False
    r_pass = c["reframe_depth"]["passed"]
    d_pass = c["content_delta"]["passed"]
    if not (r_pass or d_pass): return False
    # "not strongly negative" = their value is at most 0.05 below threshold
    if r_pass and not d_pass:
        if c["content_delta"].get("median_dist", 0) < 0.20: return False
    if d_pass and not r_pass:
        if c["reframe_depth"].get("median_dist", 0) < 0.25: return False
    return True


def main():
    problems = json.loads((CACHE / "sample_holdout_50.json").read_text(encoding="utf-8"))
    problems = [p for p in problems if "description" in p]

    # Use Claude for labeling; fall back to GPT-5.4
    try:
        client = ClaudeProxyClient()
        client.generate("ping", max_tokens=5, temperature=0.0)
        print(f"Labeler: {client.model}")
    except Exception as e:
        print(f"Claude unavailable ({str(e)[:60]}), using GPT-5.4")
        client = GPT5Client()

    # Step 1: label pids
    labels = label_subsets(client, problems)

    # Subset stats
    counts = {k: sum(1 for lab in labels.values() if lab.get(k))
              for k in ("S_reframe", "S_delta", "S_wisdom", "S_anti")}
    print(f"\nSubset sizes across {len(labels)} pids: {counts}")

    # Step 2-4: evaluate each candidate
    print(f"\n{'cid':10s} {'refr':10s} {'delta':10s} {'wisdom':10s} {'anti':10s} overall")
    print("-" * 90)
    results = {}
    for cid in list_candidates():
        comps = {
            "reframe_depth": eval_reframe_depth_conditioned(cid, labels),
            "content_delta": eval_content_delta_conditioned(cid, labels),
            "wisdom_alignment": eval_wisdom_alignment_conditioned(cid, labels),
            "antipattern_avoidance": eval_antipattern_avoidance_conditioned(cid, labels),
        }
        overall = combine(comps)
        results[cid] = {"components": comps, "overall_pass": overall}
        marks = (
            f"P(n={comps['reframe_depth']['n']})" if comps['reframe_depth']['passed']
               else f"F(n={comps['reframe_depth']['n']})",
            f"P(n={comps['content_delta']['n']})" if comps['content_delta']['passed']
               else f"F(n={comps['content_delta']['n']})",
            f"P(n={comps['wisdom_alignment']['n']})" if comps['wisdom_alignment']['passed']
               else f"F(n={comps['wisdom_alignment']['n']})",
            f"P(n={comps['antipattern_avoidance']['n']})" if comps['antipattern_avoidance']['passed']
               else f"F(n={comps['antipattern_avoidance']['n']})",
        )
        print(f"  {cid:9s} {marks[0]:10s} {marks[1]:10s} {marks[2]:10s} {marks[3]:10s} "
              f"{'PASS' if overall else 'FAIL'}")

    # Compare
    pass_cids = sorted(cid for cid, r in results.items() if r["overall_pass"])
    n_pass = len(pass_cids)
    e17 = json.loads((AUTO_DIR / "exp17_trigger_conditioned_log.json").read_text())[-1]["results"]
    e17_pass = {r["cid"] for r in e17 if r.get("gate_pass")}

    print(f"\n=== SUMMARY ===")
    print(f"  Agent v2 gate PASS: {n_pass}/12  {pass_cids}")
    print(f"  Agent predicted: ~3/12")
    print(f"  Researcher Exp 17 PASS: 4/12  {sorted(e17_pass)}")
    print(f"  Overlap (both PASS): {sorted(set(pass_cids) & e17_pass)}")
    print(f"  Only agent v2 PASS: {sorted(set(pass_cids) - e17_pass)}")
    print(f"  Only Exp 17 PASS: {sorted(e17_pass - set(pass_cids))}")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "labeler": client.model,
           "subset_sizes": counts, "n_pass": n_pass,
           "agent_v2_pass": pass_cids,
           "e17_pass": sorted(e17_pass),
           "results": results}
    prev = cache_load(OUT_LOG, default=[]) or []
    prev.append(out)
    OUT_LOG.write_text(json.dumps(prev, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
