"""Isomorphism detection: pairwise metric computation + threshold decision."""

from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import _config as cfg
from formalization.metrics.metric_suite import all_metrics


@dataclass
class PairReport:
    strategy_a: str
    strategy_b: str
    frobenius: float
    spectral: float
    fisher: float
    kl_ab: float
    kl_ba: float
    kl_log_asym: float
    relation: str           # "iso" | "subsume_a_over_b" | "subsume_b_over_a" | "weak_iso" | "distinct"
    confidence: float


def _classify(metrics: Dict[str, float], fisher_thr: float,
              spec_thr: float, symmetry_thr: float = 0.3) -> Tuple[str, float]:
    fisher = metrics["fisher"]
    spec = metrics["spectral"]
    asym = metrics["kl_log_asym"]

    symmetric = abs(asym) < symmetry_thr
    fisher_tight = fisher < fisher_thr * 0.7   # strong iso threshold
    fisher_loose = fisher < fisher_thr         # weak iso threshold
    spec_ok = spec < spec_thr

    conf = float(max(0.0, 1.0 - fisher / max(fisher_thr, 1e-6)))

    if fisher_tight and spec_ok and symmetric:
        return "iso", conf
    if fisher_loose and symmetric:
        return "weak_iso", conf * 0.7
    if fisher_loose and not symmetric:
        if asym < 0:
            return "subsume_b_over_a", conf * 0.6
        return "subsume_a_over_b", conf * 0.6
    return "distinct", 1.0 - conf


def _calibrate_thresholds(reports_metrics: List[Dict]) -> Tuple[float, float]:
    """Use distance distribution to pick iso thresholds.
    Fisher threshold = 1 stdev below mean."""
    fishers = np.array([m["fisher"] for m in reports_metrics])
    specs = np.array([m["spectral"] for m in reports_metrics])
    fisher_thr = float(fishers.mean() - 0.5 * fishers.std())
    spec_thr = float(specs.mean() + 0.2 * specs.std())
    return fisher_thr, spec_thr


def detect_all(kernels: Dict[str, np.ndarray]) -> Tuple[List[PairReport], Dict]:
    # pass 1: compute all metrics
    pairs = []
    all_metrics_list = []
    for a, b in combinations(sorted(kernels.keys()), 2):
        m = all_metrics(kernels[a], kernels[b])
        pairs.append((a, b, m))
        all_metrics_list.append(m)

    # pass 2: calibrate thresholds from distribution
    fisher_thr, spec_thr = _calibrate_thresholds(all_metrics_list)

    # pass 3: classify
    reports = []
    for a, b, m in pairs:
        relation, conf = _classify(m, fisher_thr, spec_thr)
        reports.append(PairReport(
            strategy_a=a, strategy_b=b,
            relation=relation, confidence=float(round(conf, 3)),
            **{k: float(round(v, 4)) for k, v in m.items()}
        ))

    thresholds = {"fisher_thr": round(fisher_thr, 3),
                  "spec_thr": round(spec_thr, 3)}
    return reports, thresholds


def summarize(reports: List[PairReport]) -> Dict:
    from collections import Counter
    rel = Counter(r.relation for r in reports)
    fishers = [r.fisher for r in reports]
    return {
        "total_pairs": len(reports),
        "by_relation": dict(rel),
        "fisher_mean": float(np.mean(fishers)),
        "fisher_min":  float(np.min(fishers)),
        "fisher_max":  float(np.max(fishers)),
    }
