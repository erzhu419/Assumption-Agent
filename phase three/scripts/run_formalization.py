"""
End-to-end Phase 3 pipeline:
  1. Build Markov kernel for every strategy in kb/strategies/
  2. Save kernels to formal_kb/kernels/
  3. Compute pairwise distance matrix (27×27)
  4. Detect isomorphism candidates, classify by relation
  5. Validate against EXPECTED_ISO_PAIRS from _config.py
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

import _config as cfg
from formalization.core.kernel_builder import build_kernel, load_strategy_file
from formalization.isomorphism.detector import detect_all, summarize


def main():
    # 1. Build kernels
    print(f"[1/4] Building Markov kernels in {cfg.KB_DIR}")
    strategy_files = sorted(cfg.KB_DIR.glob("S*.json"))
    kernels = {}
    cfg.KERNELS_DIR.mkdir(parents=True, exist_ok=True)

    for f in strategy_files:
        strategy = load_strategy_file(f)
        sid = strategy["id"]
        K = build_kernel(strategy)
        kernels[sid] = K
        np.save(cfg.KERNELS_DIR / f"{sid}.npy", K)
    print(f"       built {len(kernels)} kernels, shape={next(iter(kernels.values())).shape}")

    # 2. Pairwise distances / relations
    print(f"[2/4] Computing {len(kernels) * (len(kernels) - 1) // 2} pairwise distances")
    reports, thresholds = detect_all(kernels)
    print(f"       calibrated thresholds: {thresholds}")

    # 3. Summarize
    summary = summarize(reports)
    print(f"[3/4] Summary: {summary}")

    # 4. Validate against expected pairs
    print(f"[4/4] Validating expected iso-pair recall")
    expected = {tuple(sorted(pair)) for pair in cfg.EXPECTED_ISO_PAIRS}
    detected_iso = {tuple(sorted((r.strategy_a, r.strategy_b)))
                    for r in reports if r.relation == "iso"}
    detected_weak_iso = {tuple(sorted((r.strategy_a, r.strategy_b)))
                         for r in reports if r.relation == "weak_iso"}
    detected_subsume = {tuple(sorted((r.strategy_a, r.strategy_b)))
                        for r in reports
                        if r.relation.startswith("subsume")}

    hits_any = expected & (detected_iso | detected_weak_iso | detected_subsume)
    miss = expected - (detected_iso | detected_weak_iso | detected_subsume)

    print(f"       expected pairs : {expected}")
    print(f"       detected iso   : {detected_iso} ({len(detected_iso)} total)")
    print(f"       detected weak  : {len(detected_weak_iso)} pairs")
    print(f"       detected subsume: {len(detected_subsume)} pairs")
    print(f"       expected hits  : {hits_any}")
    print(f"       missed         : {miss}")
    print(f"       recall (any match): {len(hits_any) / max(len(expected), 1):.0%}")

    # Top-10 closest (non-trivial) pairs
    print("\n[+] Top-10 closest (Fisher) pairs:")
    top = sorted(reports, key=lambda r: r.fisher)[:10]
    for r in top:
        print(f"       {r.strategy_a}-{r.strategy_b}  Fisher={r.fisher:.3f} "
              f"spec={r.spectral:.3f} klasym={r.kl_log_asym:+.2f} [{r.relation}]")

    out = PROJECT / "analysis" / "formalization_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "summary": summary,
        "expected_pairs": list(map(list, expected)),
        "detected_iso": list(map(list, detected_iso)),
        "detected_subsume": list(map(list, detected_subsume)),
        "missed": list(map(list, miss)),
        "reports": [asdict(r) for r in reports],
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
