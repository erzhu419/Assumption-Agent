"""Discretize continuous problem features into a finite state set."""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import _config as cfg


def _bin_value(feature_name: str, value) -> str:
    """Map a single feature's continuous value to its discrete bin label."""
    bins = cfg.FEATURE_BINS[feature_name]
    if feature_name == "has_baseline":
        # Boolean feature
        return "yes" if bool(value) else "no"
    for entry in bins:
        label, lo, hi = entry
        v = float(value) if value is not None else 0.0
        if lo <= v < hi:
            return label
    return bins[-1][0]


def discretize(features: Dict) -> Tuple[str, ...]:
    """Return a tuple of bin labels, one per feature in FEATURE_ORDER."""
    return tuple(_bin_value(f, features.get(f)) for f in cfg.FEATURE_ORDER)


def enumerate_state_space() -> List[Tuple[str, ...]]:
    """All discrete states in canonical order. |X| = product of bin counts."""
    lists = []
    for f in cfg.FEATURE_ORDER:
        labels = [b[0] for b in cfg.FEATURE_BINS[f]]
        lists.append(labels)
    return [tuple(s) for s in product(*lists)]


def state_to_index(state: Tuple[str, ...]) -> int:
    """Canonical index in enumerate_state_space() for a given state tuple."""
    all_states = enumerate_state_space()
    return all_states.index(state)


def state_count() -> int:
    n = 1
    for f in cfg.FEATURE_ORDER:
        n *= len(cfg.FEATURE_BINS[f])
    return n
