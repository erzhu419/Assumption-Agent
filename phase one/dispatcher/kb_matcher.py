"""
KB match score computation for dispatcher input features.

For each strategy, produces a scalar in [0, 1] = (favorable matches) - (unfavorable matches),
where "match" is defined over conditions that have structural `derived_from` hints.
This gives the dispatcher a direct signal tied to the evolving KB — so KB updates
can actually move policy behavior (without retraining being the only channel).

Foundational conditions without derived_from hints are ignored by the matcher;
they can still shape behavior via the raw structural features + embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


THRESHOLD_HIGH = 0.65
THRESHOLD_LOW = 0.35


def _feature_direction(value: float) -> Optional[str]:
    if value >= THRESHOLD_HIGH:
        return "high"
    if value <= THRESHOLD_LOW:
        return "low"
    return None


class KBMatcher:
    """Loads KB strategies and scores problem features against them."""

    def __init__(self, kb_dir: Path, strategy_ids: List[str]):
        self.strategy_ids = list(strategy_ids)
        self._strategies: Dict[str, dict] = {}
        for sid in strategy_ids:
            matches = list(kb_dir.glob(f"{sid}_*.json"))
            if matches:
                self._strategies[sid] = json.loads(matches[0].read_text(encoding="utf-8"))

    def _score_conditions(self, conds: List[dict], features: Dict) -> float:
        """Sum of confidence over matching conditions (those with derived_from)."""
        total = 0.0
        for c in conds:
            meta = c.get("derived_from", {}) or {}
            fkey = meta.get("feature_key")
            fdir = meta.get("feature_direction")
            if not fkey or not fdir:
                continue
            val = features.get(fkey)
            if val is None:
                continue
            if _feature_direction(float(val)) == fdir:
                total += float(c.get("confidence", 0.5))
        return total

    def compute_scores(self, features: Dict, action_space: List[str]) -> np.ndarray:
        """Return one score per action (strategies + compositions + special).
        Compositions and specials get 0 — they have no direct KB match."""
        out = np.zeros(len(action_space), dtype=np.float32)
        for i, a in enumerate(action_space):
            if a not in self._strategies:
                continue
            ac = self._strategies[a].get("applicability_conditions", {})
            fav = self._score_conditions(ac.get("favorable", []), features)
            unfav = self._score_conditions(ac.get("unfavorable", []), features)
            # Net score, squashed to [0, 1] via sigmoid-ish scale
            net = fav - unfav
            out[i] = float(1.0 / (1.0 + np.exp(-net)))
        return out
