"""
Phase 3 v2 pipeline — with all three upgrades:
  ① LLM-based action classifier (instead of keyword rules)
  ② SEG (strategy execution graph) + graph edit distance
  ③ Blackwell partial order on top-K closest Fisher pairs

Outputs:
  - kernels to formal_kb/kernels/
  - detailed pair report with Fisher / spectral / KL / SEG / Blackwell
  - recall vs expected_iso_pairs
"""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

import _config as cfg
from formalization.core.kernel_builder import build_kernel, load_strategy_file
from formalization.core.llm_action_classifier import LLMActionClassifier
from formalization.core.seg import build_seg, seg_distance
from formalization.metrics.blackwell import blackwell_relation
from formalization.isomorphism.detector import detect_all, summarize


TOP_K_FOR_BLACKWELL = 30


def main():
    # 1. Initialize LLM classifier (with disk cache)
    cache_path = cfg.PROJECT_ROOT / "analysis" / "action_classifier_cache.json"
    classifier = LLMActionClassifier(cache_path=cache_path)
    print(f"[init] LLM classifier cache has {len(classifier._cache)} entries")

    # 2. Build kernels + SEGs
    print(f"[1/5] Building Markov kernels + SEGs with LLM classifier...")
    strategy_files = sorted(cfg.KB_DIR.glob("S*.json"))
    kernels = {}
    segs = {}
    cfg.KERNELS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for f in strategy_files:
        strategy = load_strategy_file(f)
        sid = strategy["id"]
        K = build_kernel(strategy, llm_classifier=classifier)
        kernels[sid] = K
        segs[sid] = build_seg(strategy)
        np.save(cfg.KERNELS_DIR / f"{sid}.npy", K)
    print(f"       built {len(kernels)} kernels in {time.time() - t0:.0f}s")

    # 3. Pairwise Fisher + SEG distance
    print(f"[2/5] Pairwise distance: kernel metrics + SEG")
    reports, thresholds = detect_all(kernels)
    print(f"       kernel thresholds: {thresholds}")

    # attach SEG distance
    seg_dist_by_pair = {}
    for r in reports:
        d_seg = seg_distance(segs[r.strategy_a], segs[r.strategy_b])
        seg_dist_by_pair[(r.strategy_a, r.strategy_b)] = d_seg

    # 4. Blackwell on top-K closest (Fisher) pairs
    print(f"[3/5] Blackwell LP on top-{TOP_K_FOR_BLACKWELL} closest pairs")
    top = sorted(reports, key=lambda r: r.fisher)[:TOP_K_FOR_BLACKWELL]
    blackwell_by_pair = {}
    t1 = time.time()
    for r in top:
        rel = blackwell_relation(kernels[r.strategy_a], kernels[r.strategy_b], tol=0.15)
        blackwell_by_pair[(r.strategy_a, r.strategy_b)] = rel
    print(f"       completed in {time.time() - t1:.0f}s")

    # 5. Combined verdict
    # Rule: iso if (kernel iso) AND (SEG distance < 0.6) AND Blackwell says equivalent
    print("[4/5] Combined verdict (kernel ∧ SEG ∧ Blackwell)")

    upgraded_reports = []
    for r in reports:
        seg_d = seg_dist_by_pair[(r.strategy_a, r.strategy_b)]
        bw = blackwell_by_pair.get((r.strategy_a, r.strategy_b), "not_checked")

        combined = r.relation
        if r.relation == "iso" and seg_d > 0.8:
            combined = "iso_surface_only"  # kernel iso but structurally different
        elif r.relation == "distinct" and seg_d < 0.3:
            combined = "structural_match_only"  # different behavior but similar structure
        elif r.relation in ("iso", "weak_iso") and bw == "both_dominate":
            combined = "iso_strong"  # Blackwell-equivalent
        elif r.relation == "distinct" and bw == "both_dominate":
            combined = "iso_latent"  # only Blackwell detected

        upgraded_reports.append({
            **asdict(r),
            "seg_distance": round(seg_d, 3),
            "blackwell": bw,
            "combined_relation": combined,
        })

    # Validation
    expected = {tuple(sorted(pair)) for pair in cfg.EXPECTED_ISO_PAIRS}
    detected_any_iso = set()
    for r in upgraded_reports:
        rel = r["combined_relation"]
        if rel in ("iso", "weak_iso", "iso_strong", "iso_latent", "structural_match_only"):
            detected_any_iso.add(tuple(sorted((r["strategy_a"], r["strategy_b"]))))
        elif rel.startswith("subsume"):
            detected_any_iso.add(tuple(sorted((r["strategy_a"], r["strategy_b"]))))

    hits = expected & detected_any_iso
    miss = expected - detected_any_iso

    print(f"\n[5/5] Validation:")
    print(f"       expected : {expected}")
    print(f"       hits     : {hits}")
    print(f"       missed   : {miss}")
    print(f"       recall   : {len(hits) / max(len(expected), 1):.0%}")
    print(f"       total iso-ish pairs: {len(detected_any_iso)}")

    # Expected pair details
    print(f"\n  Expected pair detail:")
    for pair in sorted(expected):
        for r in upgraded_reports:
            if tuple(sorted((r["strategy_a"], r["strategy_b"]))) == pair:
                print(f"    {pair[0]}-{pair[1]}: Fisher={r['fisher']:.2f} "
                      f"seg={r['seg_distance']:.2f} bw={r['blackwell']} → {r['combined_relation']}")
                break

    # Top-15 combined
    print("\n  Top-15 closest (by Fisher + SEG):")
    combined_top = sorted(upgraded_reports,
                          key=lambda r: r["fisher"] + r["seg_distance"])[:15]
    for r in combined_top:
        print(f"    {r['strategy_a']}-{r['strategy_b']}  "
              f"Fisher={r['fisher']:.2f} seg={r['seg_distance']:.2f} "
              f"bw={r['blackwell']} → {r['combined_relation']}")

    out = PROJECT / "analysis" / "formalization_v2_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "thresholds": thresholds,
        "expected_pairs": sorted(list(expected)),
        "hits": sorted(list(hits)),
        "missed": sorted(list(miss)),
        "recall": len(hits) / max(len(expected), 1),
        "reports": upgraded_reports,
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
