"""Experiment 2: Self-rebuttal.

The system is invited to attack its own KEEP decisions. For each
committed wisdom, GPT-5.4 is given:
  - the wisdom record
  - the original 50-pair A/B evidence summary
  - a mandate to propose THREE strongest plausible counter-hypotheses
    ("this KEEP may be spurious because ...")
and asked to specify, for each, what evidence would refute the
counter-hypothesis.

This is not a statistical test — it is a probe of whether the system
can generate legible self-doubt. The output becomes appendix material
documenting the failure modes the agent itself predicts for its own
KEEP decisions.
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
sys.path.insert(0, str(PROJECT / "phase four"))

from gpt5_client import GPT5Client
from llm_client import parse_json_from_llm
from wisdom_registry import load_or_init_registry


AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp2_self_rebuttal_log.json"

KEPT_IDS = ["W076", "W077", "W078"]


REBUTTAL_PROMPT = """你是一个苛刻的科学审稿人。下面是 wisdom library agent 刚自主 commit 的一条新 wisdom，以及支撑它的 A/B 证据。

## 该 wisdom
id: {wid}
aphorism: {aphorism}
source: {source}
signal: {signal}
unpacked_for_llm: {unpacked}
provenance: {provenance}

## A/B 证据
在 {n_problems} 个 held-out 问题上，带该 wisdom 的 ext library
赢 {ext_wins} 次，不带的 base library 赢 {base_wins} 次（ties={ties}）。
win rate = {wr_ext:.2f}，刚好越过 +{threshold_pp}pp 阈值。
判官：{judge}

## 你的任务
给出**三条最强的反驳假设**——每条说明"这次 KEEP 之所以看起来正的，可能其实是什么别的原因导致的"。避免笼统的"样本小"——给具体的 confound。

对每条反驳，说清楚：
  alt_hypothesis: 具体的替代解释（30-50 字）
  mechanism:     为什么这个机制能产生 wr=0.60 的假象（40-80 字）
  falsifier:     什么实验/数据能明确拒掉这条反驳（30-60 字）
  severity:      high/medium/low（这条反驳如果成立，对 KEEP 决定的威胁有多大）

最后给一条 overall_verdict：你会推荐 accept / further_test / reject 这次 commit。

## 输出 JSON（不要代码块）
{{
  "rebuttals": [
    {{"alt_hypothesis": "...", "mechanism": "...", "falsifier": "...", "severity": "..."}},
    {{"alt_hypothesis": "...", "mechanism": "...", "falsifier": "...", "severity": "..."}},
    {{"alt_hypothesis": "...", "mechanism": "...", "falsifier": "...", "severity": "..."}}
  ],
  "overall_verdict": "accept / further_test / reject",
  "verdict_reason": "50-100字"
}}
"""


def load_keep_evidence(registry, validation_log):
    """Map W076/W077/W078 -> (wisdom_record, ab_evidence)."""
    evidence = {}
    # Walk validation_log_parallel.json + cross_llm_log.json for A/B stats
    for entry in validation_log:
        for r in entry.get("results", []):
            cid = r.get("committed_id")
            if cid in KEPT_IDS:
                evidence[cid] = {
                    "wr_ext": r["ab"]["wr_a"],
                    "ext_wins": r["ab"]["wins_a"],
                    "base_wins": r["ab"]["wins_b"],
                    "ties": r["ab"]["ties"],
                    "n_problems": entry.get("test_n", 50),
                    "judge": "gemini-3-flash",
                }

    # Get wisdom records
    wisdoms = {w["id"]: w for w in registry["wisdoms"]}
    packed = {}
    for wid in KEPT_IDS:
        if wid not in wisdoms or wid not in evidence:
            continue
        packed[wid] = {"wisdom": wisdoms[wid], "ab": evidence[wid]}
    return packed


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    registry = load_or_init_registry()
    val_log = json.loads((AUTO_DIR / "validation_log_parallel.json").read_text())
    packed = load_keep_evidence(registry, val_log)
    print(f"Loaded {len(packed)} KEEP cases: {list(packed.keys())}")

    client = GPT5Client()
    rebuttals = []
    for wid, data in packed.items():
        w = data["wisdom"]; ab = data["ab"]
        print(f"\n=== [{wid}] {w['aphorism']} ===")
        prompt = REBUTTAL_PROMPT.format(
            wid=wid, aphorism=w["aphorism"], source=w["source"],
            signal=w["signal"], unpacked=w["unpacked_for_llm"],
            provenance=w.get("source", "?"),  # provenance tag
            n_problems=ab["n_problems"], ext_wins=ab["ext_wins"],
            base_wins=ab["base_wins"], ties=ab["ties"], wr_ext=ab["wr_ext"],
            threshold_pp=10, judge=ab["judge"],
        )
        resp = client.generate(prompt, max_tokens=1500, temperature=0.45)
        parsed = parse_json_from_llm(resp["text"])
        r = parsed.get("rebuttals", [])
        verdict = parsed.get("overall_verdict", "?")
        reason = parsed.get("verdict_reason", "")

        print(f"  Rebuttals ({len(r)}):")
        for i, rb in enumerate(r, 1):
            sev = rb.get("severity", "?")
            ah = rb.get("alt_hypothesis", "")[:80]
            print(f"    [{i}] [{sev}] {ah}")
        print(f"  Verdict: {verdict}  ({reason[:80]}...)")

        rebuttals.append({
            "wid": wid, "aphorism": w["aphorism"],
            "rebuttals": r,
            "overall_verdict": verdict,
            "verdict_reason": reason,
        })

    log = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "generator_model": client.model,
           "cases": rebuttals}
    OUT_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\nLog → {OUT_LOG.name}")

    # Summary table
    print("\n=== SUMMARY ===")
    c = {"accept": 0, "further_test": 0, "reject": 0, "?": 0}
    for r in rebuttals:
        v = r["overall_verdict"]
        c[v] = c.get(v, 0) + 1
    for verdict, n in c.items():
        print(f"  {verdict}: {n}")


if __name__ == "__main__":
    main()
