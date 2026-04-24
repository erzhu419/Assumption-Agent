"""Exp 13 — LLM-judged faithfulness (a sharper version of Stage A).

Stage A in rigorous_gate.py measured *embedding-space* alignment between
the Turn-0 perturbation direction and the wisdom's self-description.
Values near 0 could mean either (a) the wisdom is unfaithful or (b) the
sentence-transformer can't see the alignment at the granularity we care
about. This experiment uses an LLM judge (Claude Opus 4.6, chosen to be
cross-family to the solver) to answer the same question in natural
language:

  "On this problem, does the EXT solver's what_changed reflect the
   orientation described in the wisdom's unpacked_for_llm, relative to
   the BASE solver's what_changed?"

Judge returns YES / NO / PARTIAL per pid. We report per-wisdom:
  faithfulness_rate = YES / (YES + NO + PARTIAL)
  aligned_rate      = (YES + PARTIAL) / total

Threshold: faithfulness_rate ≥ 0.30 OR aligned_rate ≥ 0.50. These are
calibrated to be non-trivial — half the pids should show at least
partial alignment if the wisdom is doing what it claims.
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
OUT_LOG = AUTO_DIR / "exp13_faithfulness_log.json"

PARALLEL = 6
# Sample N pids per candidate (cost: N × 12 ≈ N * 12 * 2s with parallelism)
N_SAMPLE_PIDS = 20

# Same 12 candidates
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


FAITHFULNESS_PROMPT = """你是一个方法论评审专家。请判断某条 wisdom 的 self-description 是否确实体现在它对一个问题的推理扰动里。

== Wisdom 自述（该 wisdom 声称自己应该怎么用）==
aphorism: {aphorism}
signal: {signal}
unpacked_for_llm: {unpacked}

== 同一个问题，两个 solver 的 Turn-0 reframing 输出 ==
**BASE** (不包含该 wisdom 的 library):
{base_wc}

**EXT** (包含该 wisdom 的 library):
{ext_wc}

== 你的判断 ==
EXT 相对于 BASE 多出的那些 reframing element / critical reframe，是否体现了该 wisdom 的 unpacked_for_llm 所描述的 orientation？

三档判断：
- YES: EXT 明确多出的 reframing 和 wisdom 的自述**概念一致且可映射**（不只是用词类似，是思路上真的被 wisdom 引导了）
- PARTIAL: 有某个元素可能对应 wisdom，但不突出，也可能是巧合
- NO: EXT 的变化和 wisdom 自述**不相关**（可能只是 prompt 增长带来的普通扰动）

== 输出 JSON（不要代码块）==
{{"verdict": "YES" 或 "PARTIAL" 或 "NO",
  "reasoning": "60-120字，指出 EXT 多出的关键点以及是否真的映射到 wisdom 的 unpacked 描述"}}
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


def load_meta(stem):
    p = CACHE / "answers" / f"{stem}_meta.json"
    return cache_load(p) or {}


def judge_one(client, wisdom, base_wc, ext_wc):
    prompt = FAITHFULNESS_PROMPT.format(
        aphorism=wisdom["aphorism"],
        signal=wisdom.get("signal", ""),
        unpacked=wisdom.get("unpacked_for_llm", ""),
        base_wc=base_wc, ext_wc=ext_wc,
    )
    try:
        r = client.generate(prompt, max_tokens=500, temperature=0.0)
        v = parse_json_from_llm(r["text"])
        return v.get("verdict", "ERR"), v.get("reasoning", "")
    except Exception as e:
        return "ERR", f"{e}"[:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sample", type=int, default=N_SAMPLE_PIDS)
    args = ap.parse_args()

    claude = ClaudeProxyClient()
    print(f"Faithfulness judge: {claude.model}\n")

    # Pick deterministic sample of pids
    import random
    rng = random.Random(42)

    base_defaults = {"WCROSSL01": "_valp_v20_base"}
    default_base = "_valp_v20p1_base"

    results = []
    for cand in CANDIDATES:
        cid = cand["cid"]; wid = cand.get("committed_id") or "----"
        wisdom = load_candidate_record(cand["aphorism"])
        if not wisdom:
            print(f"  [{cid}] wisdom record not found; skip"); continue
        base_stem = base_defaults.get(cid, default_base)
        base_meta = load_meta(base_stem)
        ext_meta = load_meta(f"_valp_v20_ext_{cid}")
        if not base_meta or not ext_meta:
            print(f"  [{cid}] meta missing; skip"); continue

        shared = sorted(set(base_meta.keys()) & set(ext_meta.keys()))
        sample = rng.sample(shared, min(args.n_sample, len(shared)))
        print(f"\n=== [{cid}/{wid}] {cand['aphorism']} ({len(sample)} pids) ===")
        t0 = time.time()

        def task(pid):
            b = base_meta[pid].get("what_changed", "").strip()
            e = ext_meta[pid].get("what_changed", "").strip()
            if not b or not e:
                return pid, "MISSING", ""
            v, r = judge_one(claude, wisdom, b, e)
            return pid, v, r

        verdicts = {}
        with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
            futs = [ex.submit(task, pid) for pid in sample]
            for f in as_completed(futs):
                pid, v, r = f.result()
                verdicts[pid] = {"verdict": v, "reasoning": r}

        c = {"YES": 0, "PARTIAL": 0, "NO": 0, "MISSING": 0, "ERR": 0}
        for v in verdicts.values():
            c[v["verdict"]] = c.get(v["verdict"], 0) + 1
        total = c["YES"] + c["PARTIAL"] + c["NO"]
        faith = c["YES"] / total if total else 0.0
        aligned = (c["YES"] + c["PARTIAL"]) / total if total else 0.0
        pass_faith = faith >= 0.30
        pass_align = aligned >= 0.50
        composite = "PASS" if (pass_faith or pass_align) else "FAIL"
        dt = time.time() - t0
        print(f"  YES={c['YES']} PARTIAL={c['PARTIAL']} NO={c['NO']} MISS={c['MISSING']} "
              f"ERR={c['ERR']}  faith={faith:.2f} aligned={aligned:.2f}  "
              f"{composite}  ({dt:.0f}s)")

        results.append({
            "cid": cid, "wid": cand.get("committed_id"),
            "aphorism": cand["aphorism"],
            "n_sampled": len(sample),
            "counts": c,
            "faithfulness_rate": faith,
            "aligned_rate": aligned,
            "pass_faith": pass_faith,
            "pass_aligned": pass_align,
            "composite": composite,
            "verdicts": verdicts,
        })

    # Summary
    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    print(f"{'cid':10s} {'wid':5s} {'Y':4s} {'P':4s} {'N':4s} "
          f"{'faith':6s} {'align':6s}  result")
    for r in results:
        wid = r["wid"] or "----"
        c = r["counts"]
        print(f"  {r['cid']:9s} {wid:4s} {c['YES']:3d} {c['PARTIAL']:3d} {c['NO']:3d}  "
              f"{r['faithfulness_rate']:.2f}   {r['aligned_rate']:.2f}   "
              f"{r['composite']}")
    n_pass = sum(1 for r in results if r["composite"] == "PASS")
    print(f"\n  Passed faithfulness gate: {n_pass}/{len(results)}")

    log = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "judge": claude.model, "n_sample_pids": args.n_sample,
           "results": results}
    prev = json.loads(OUT_LOG.read_text()) if OUT_LOG.exists() else []
    prev.append(log)
    OUT_LOG.write_text(json.dumps(prev, ensure_ascii=False, indent=2))
    print(f"Saved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
