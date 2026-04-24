"""Exp 16 — Citation audit (architectural change #2).

Hypothesis: the solver prompt should cite which wisdom is applied on
which step. If we retrospectively ask a cross-family judge 'in this
ext-library answer, where is wisdom W concretely used?', the judge
should return a quotable passage for problems on which W was retrieved.
If W cannot be concretely located in the answer, adding W to the
library was a context-level perturbation, not a wisdom-level one.

For each (cid, pid) in a sample:
  Show judge: problem, ext-answer, W's aphorism + unpacked.
  Judge returns:
    CITED_CLEAR  — a specific passage clearly applies W's orientation; quotes it
    CITED_WEAK   — some passage is arguably in W's direction but not dispositive
    NOT_CITED    — no passage clearly applies W; the answer does not use W

Threshold: a wisdom is cite-able iff CITED_CLEAR rate >= 0.30 or
(CITED_CLEAR + CITED_WEAK) >= 0.60.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
sys.path.insert(0, str(PROJECT / "phase four"))

from claude_proxy_client import ClaudeProxyClient
from llm_client import parse_json_from_llm

CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp16_citation_audit_log.json"

PARALLEL = 6
N_SAMPLE_PIDS = 15   # per candidate

CANDIDATES = [
    {"cid": "WCAND01", "aphorism": "上工治未病，不治已病"},
    {"cid": "WCAND02", "aphorism": "别高效解决一个被看错的问题"},
    {"cid": "WCAND03", "aphorism": "凡事预则立，不预则废"},
    {"cid": "WCAND04", "aphorism": "急则治其标，缓则治其本"},
    {"cid": "WCAND05", "aphorism": "凡益之道，与时偕行", "committed_id": "W076"},
    {"cid": "WCAND06", "aphorism": "覆水难收，向前算账"},
    {"cid": "WCAND07", "aphorism": "亲兄弟，明算账"},
    {"cid": "WCAND08", "aphorism": "想理解行为，先看激励"},
    {"cid": "WCAND09", "aphorism": "不谋全局者，不足谋一域"},
    {"cid": "WCAND10", "aphorism": "没有调查，就没有发言权", "committed_id": "W077"},
    {"cid": "WCAND11", "aphorism": "若不是品牌，你就只是商品。"},
    {"cid": "WCROSSL01", "aphorism": "是骡子是马，拉出来遛遛", "committed_id": "W078"},
]


CITATION_PROMPT = """你是一位 **citation 审计员**。你要判断一份答卷里有没有明确用到某条 wisdom。

== 问题 ==
{problem}

== 该 wisdom ==
aphorism: {aphorism}
unpacked: {unpacked}

== 答卷 ==
{answer}

== 你的任务 ==
判断答卷里**是否存在一个具体段落/短语**，明确体现了该 wisdom 的 orientation。

三档：
- CITED_CLEAR: 答卷里有一处**具体段落**明确执行了该 wisdom 的核心思路（不是主题类似，是动作或论证步骤上明确）。**必须引用原文**（30 字以内）。
- CITED_WEAK:  有段落与 wisdom **方向一致**但不是该 wisdom 独占的（其他通用方法论也能解释）；或只是主题类似。
- NOT_CITED:   找不到具体段落能定位到该 wisdom；该 wisdom 并未真正在这份答卷里起作用。

输出 JSON（不要代码块）：
{{"verdict": "CITED_CLEAR" 或 "CITED_WEAK" 或 "NOT_CITED",
  "quoted_passage": "30字内原文引用或空串",
  "reasoning": "50-100字"}}
"""


def cache_load(p, default=None):
    if Path(p).exists():
        try: return json.loads(Path(p).read_text(encoding="utf-8"))
        except: return default
    return default


def load_candidate_record(aphorism):
    for src in ("success_distilled_candidates.json", "cross_llm_candidates.json"):
        data = cache_load(AUTO_DIR / src, default=[])
        for c in data:
            if c.get("aphorism", "").strip() == aphorism.strip():
                return c
    return None


def judge_citation(client, problem, aphorism, unpacked, answer):
    prompt = CITATION_PROMPT.format(
        problem=problem, aphorism=aphorism, unpacked=unpacked, answer=answer,
    )
    try:
        r = client.generate(prompt, max_tokens=500, temperature=0.0)
        v = parse_json_from_llm(r["text"])
        return (v.get("verdict", "ERR"),
                v.get("quoted_passage", ""),
                v.get("reasoning", ""))
    except Exception as e:
        return "ERR", "", f"{e}"[:80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_SAMPLE_PIDS)
    args = ap.parse_args()

    claude = ClaudeProxyClient()
    holdout = json.loads((CACHE / "sample_holdout_50.json").read_text(encoding="utf-8"))
    pid_to_problem = {p["problem_id"]: p["description"] for p in holdout if "description" in p}

    import random
    rng = random.Random(42)

    print(f"Judge: {claude.model}\n")
    results = []
    for cand in CANDIDATES:
        cid = cand["cid"]; wid = cand.get("committed_id") or "----"
        rec = load_candidate_record(cand["aphorism"])
        if not rec:
            print(f"  [{cid}] record missing"); continue
        ans_path = CACHE / "answers" / f"_valp_v20_ext_{cid}_answers.json"
        if not ans_path.exists():
            print(f"  [{cid}] ext answers missing"); continue
        answers = json.loads(ans_path.read_text(encoding="utf-8"))
        shared = sorted(set(answers.keys()) & set(pid_to_problem.keys()))
        if not shared:
            print(f"  [{cid}] no shared pids"); continue
        sample = rng.sample(shared, min(args.n, len(shared)))

        def task(pid):
            return (pid,) + judge_citation(
                claude, pid_to_problem[pid], rec["aphorism"],
                rec["unpacked_for_llm"], answers[pid],
            )

        print(f"\n=== [{cid}/{wid}] {cand['aphorism']} ({len(sample)} pids) ===")
        t0 = time.time()
        c = {"CITED_CLEAR": 0, "CITED_WEAK": 0, "NOT_CITED": 0, "ERR": 0}
        per_pid = {}
        with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
            futs = [ex.submit(task, pid) for pid in sample]
            for f in as_completed(futs):
                pid, v, q, r = f.result()
                c[v] = c.get(v, 0) + 1
                per_pid[pid] = {"verdict": v, "quote": q, "reasoning": r}
        dt = time.time() - t0
        total = c["CITED_CLEAR"] + c["CITED_WEAK"] + c["NOT_CITED"]
        clear = c["CITED_CLEAR"] / total if total else 0
        any_cite = (c["CITED_CLEAR"] + c["CITED_WEAK"]) / total if total else 0
        cite_pass = clear >= 0.30 or any_cite >= 0.60
        print(f"  CLEAR={c['CITED_CLEAR']} WEAK={c['CITED_WEAK']} "
              f"NOT={c['NOT_CITED']} ERR={c['ERR']}  "
              f"clear_rate={clear:.2f} any_cite={any_cite:.2f}  "
              f"{'PASS' if cite_pass else 'FAIL'} ({dt:.0f}s)")

        # sample 1-2 quotes
        samples = [(pid, d) for pid, d in per_pid.items()
                    if d["verdict"] == "CITED_CLEAR" and d["quote"]][:2]
        for pid, d in samples:
            print(f"    [{pid}] {d['quote'][:60]}")

        results.append({
            "cid": cid, "wid": cand.get("committed_id"),
            "aphorism": cand["aphorism"],
            "counts": c, "clear_rate": clear, "any_cite_rate": any_cite,
            "passes_cite_gate": cite_pass,
            "per_pid": per_pid,
        })

    # Summary
    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    print(f"{'cid':10s} {'wid':5s} {'CLEAR':6s} {'WEAK':5s} {'NOT':4s} "
          f"{'clear':6s} {'any':6s}  result")
    for r in results:
        wid = r["wid"] or "----"
        c = r["counts"]
        print(f"  {r['cid']:9s} {wid:4s} {c['CITED_CLEAR']:5d}  {c['CITED_WEAK']:4d}  "
              f"{c['NOT_CITED']:3d}  {r['clear_rate']:.2f}   {r['any_cite_rate']:.2f}   "
              f"{'PASS' if r['passes_cite_gate'] else 'FAIL'}")
    n_pass = sum(1 for r in results if r["passes_cite_gate"])
    print(f"\n  Pass cite gate: {n_pass}/{len(results)}")

    log = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "judge": claude.model, "n_sample": args.n,
           "results": results}
    prev = json.loads(OUT_LOG.read_text()) if OUT_LOG.exists() else []
    prev.append(log)
    OUT_LOG.write_text(json.dumps(prev, ensure_ascii=False, indent=2))
    print(f"Saved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
