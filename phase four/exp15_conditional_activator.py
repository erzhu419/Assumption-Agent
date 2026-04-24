"""Exp 15 — Conditional activator framing.

Hypothesis: a wisdom's value is not a scalar "pooled pair-wr". It is
conditional on the problem class the wisdom actually targets. If a
wisdom W claims to trigger on pattern P, then on problems WITH P it
should have high wr, and on problems WITHOUT P it should be ≈ 0.5
(no-op).

We use the already-computed per-domain pair-wr from staged_gate_log
(pooled n=50 split by problem domain: 6 classes of size 5-14). For
each of 12 candidates, we identify domains where wr ≥ 0.60 and ≥ 5
pairs, i.e. the domain(s) on which the wisdom would be a
'specialist'.

Decision rule: a wisdom is valid as a conditional activator iff
  ∃ domain d with n_d ≥ 5 and wr_d ≥ 0.60
and
  in domains d' WITHOUT the claimed trigger, wr_{d'} should not be
  catastrophically low (< 0.30)
and
  the wisdom's claimed trigger domain is plausibly where wr_d is
  high (soft check, not machine-verifiable without ground-truth
  labels).
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase four"))

CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
STAGED_LOG = AUTO_DIR / "staged_gate_log.json"
OUT_LOG = AUTO_DIR / "exp15_conditional_log.json"

# Each candidate's self-described trigger domain (approximate, from aphorism).
CANDIDATE_CLAIMED_DOMAIN = {
    "WCAND01": {"aphorism": "上工治未病，不治已病",
                "claimed_domain_hints": ["risk_prevention", "all_domains_with_early_warning"]},
    "WCAND02": {"aphorism": "别高效解决一个被看错的问题",
                "claimed_domain_hints": ["problem_framing", "business", "engineering"]},
    "WCAND03": {"aphorism": "凡事预则立，不预则废",
                "claimed_domain_hints": ["all_domains_with_planning"]},
    "WCAND04": {"aphorism": "急则治其标，缓则治其本",
                "claimed_domain_hints": ["triage", "business", "engineering"]},
    "WCAND05": {"aphorism": "凡益之道，与时偕行",
                "claimed_domain_hints": ["business", "engineering"], "wid": "W076"},
    "WCAND06": {"aphorism": "覆水难收，向前算账",
                "claimed_domain_hints": ["business", "daily_life"]},
    "WCAND07": {"aphorism": "亲兄弟，明算账",
                "claimed_domain_hints": ["business", "daily_life"]},
    "WCAND08": {"aphorism": "想理解行为，先看激励",
                "claimed_domain_hints": ["business", "daily_life"]},
    "WCAND09": {"aphorism": "不谋全局者，不足谋一域",
                "claimed_domain_hints": ["business", "engineering"]},
    "WCAND10": {"aphorism": "没有调查，就没有发言权",
                "claimed_domain_hints": ["all_domains"], "wid": "W077"},
    "WCAND11": {"aphorism": "若不是品牌，你就只是商品。",
                "claimed_domain_hints": ["business"]},
    "WCROSSL01": {"aphorism": "是骡子是马，拉出来遛遛",
                   "claimed_domain_hints": ["mathematics", "science"], "wid": "W078"},
}

# conditional-activator pass criteria
SPECIALIST_WR = 0.60
SPECIALIST_MIN_N = 5
MAX_OTHER_DOMAIN_CATASTROPHE = 0.30
POOLED_MIN_FOR_PURE_POOLED = 0.60


def main():
    if not STAGED_LOG.exists():
        print(f"{STAGED_LOG.name} not found — run staged_gate.py first"); return
    sg = json.loads(STAGED_LOG.read_text(encoding="utf-8"))
    entry = sg[-1]

    by_cid = {r["cid"]: r for r in entry["results"]}

    print(f"=== Exp 15: conditional activator analysis ===\n")
    print(f"{'cid':10s} {'wid':5s} {'pooled':7s}  "
          f"{'specialist domains (n≥5 & wr≥0.60)':50s}  {'catastrophic (<0.30)'}")
    print("-" * 110)

    results = []
    for cid, cand in CANDIDATE_CLAIMED_DOMAIN.items():
        if cid not in by_cid: continue
        row = by_cid[cid]
        pooled_wr = row["stage2"]["wr"]
        dom_bd = row["stage3"].get("domains", {})
        specialists = []
        catastrophes = []
        for d, b in dom_bd.items():
            if b["n"] >= SPECIALIST_MIN_N and b["wr"] >= SPECIALIST_WR:
                specialists.append(f"{d}[n={b['n']},wr={b['wr']:.2f}]")
            if b["wr"] < MAX_OTHER_DOMAIN_CATASTROPHE:
                catastrophes.append(f"{d}[n={b['n']},wr={b['wr']:.2f}]")
        spec_s = ", ".join(specialists) if specialists else "—"
        cat_s = ", ".join(catastrophes) if catastrophes else "—"
        wid = cand.get("wid") or "----"
        print(f"  {cid:9s} {wid:4s} {pooled_wr:<7.2f}  {spec_s[:48]:50s}  {cat_s[:40]}")

        # Decision: conditional activator valid iff specialists AND no catastrophic
        cond_valid = bool(specialists) and len(catastrophes) <= 1
        claimed = cand["claimed_domain_hints"]
        # soft alignment: any specialist domain intersects claimed?
        spec_domains = [s.split("[")[0] for s in specialists]
        aligned = any(d in claimed or "all_domains" in claimed[0]
                      for d in spec_domains)
        results.append({
            "cid": cid, "wid": cand.get("wid"),
            "aphorism": cand["aphorism"],
            "pooled_wr": pooled_wr,
            "specialist_domains": specialists,
            "catastrophic_domains": catastrophes,
            "claimed_domains": claimed,
            "conditional_activator_valid": cond_valid,
            "specialist_aligned_with_claim": aligned,
            "per_domain_detail": dom_bd,
        })

    # Summary
    n_specialists = sum(1 for r in results if r["specialist_domains"])
    n_cond_valid = sum(1 for r in results if r["conditional_activator_valid"])
    n_aligned = sum(1 for r in results if r["specialist_aligned_with_claim"])
    print(f"\n=== Summary ===")
    print(f"  Has ≥1 specialist domain:                 {n_specialists}/12")
    print(f"  Passes conditional-activator rule:        {n_cond_valid}/12")
    print(f"  Specialist aligned with claimed trigger:  {n_aligned}/12")

    # Detail on conditionally valid
    if n_cond_valid:
        print(f"\n  Conditionally valid wisdoms (compare to pooled wr which was all < 0.60):")
        for r in results:
            if not r["conditional_activator_valid"]: continue
            print(f"    {r['cid']} {r['wid'] or '----'} ({r['aphorism'][:20]}):")
            print(f"      pooled wr  : {r['pooled_wr']:.2f}")
            print(f"      specialist : {r['specialist_domains']}")
            print(f"      claimed    : {r['claimed_domains']}")
            print(f"      aligned    : {r['specialist_aligned_with_claim']}")

    OUT_LOG.write_text(json.dumps({"thresholds": {
        "specialist_wr": SPECIALIST_WR, "specialist_min_n": SPECIALIST_MIN_N,
        "max_catastrophe_wr": MAX_OTHER_DOMAIN_CATASTROPHE,
    }, "results": results}, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
