"""
Discretize continuous problem features into finite state space.
Used by the statistical world model to build lookup tables.

State space: 5 dimensions → 3×2×2×3×3 = 108 discrete states
"""

from dataclasses import dataclass
from typing import Dict, Tuple

# Discretization bins (from Phase 3 dev doc section 1.2)
FEATURE_BINS = {
    "coupling_estimate": [("low", 0.0, 0.3), ("medium", 0.3, 0.7), ("high", 0.7, 1.0)],
    "decomposability": [("low", 0.0, 0.4), ("high", 0.4, 1.0)],
    "has_baseline": [("no", 0, 0.5), ("yes", 0.5, 1.01)],  # treat as float
    "information_completeness": [("low", 0.0, 0.4), ("medium", 0.4, 0.7), ("high", 0.7, 1.0)],
    "component_count": [("few", 0, 4), ("moderate", 4, 8), ("many", 8, 999)],
}

STATE_SPACE_SIZE = 3 * 2 * 2 * 3 * 3  # = 108


@dataclass(frozen=True)
class DiscreteState:
    coupling: str
    decomposability: str
    has_baseline: str
    info_completeness: str
    component_count: str

    def to_tuple(self) -> Tuple[str, ...]:
        return (self.coupling, self.decomposability, self.has_baseline,
                self.info_completeness, self.component_count)

    def to_index(self) -> int:
        """Convert to a flat index 0..107."""
        dims = [
            (["low", "medium", "high"], self.coupling),
            (["low", "high"], self.decomposability),
            (["no", "yes"], self.has_baseline),
            (["low", "medium", "high"], self.info_completeness),
            (["few", "moderate", "many"], self.component_count),
        ]
        idx = 0
        multiplier = 1
        for labels, val in reversed(dims):
            idx += labels.index(val) * multiplier
            multiplier *= len(labels)
        return idx


def _bin_value(value: float, bins: list) -> str:
    """Assign a continuous value to a discrete bin."""
    for label, low, high in bins:
        if low <= value < high:
            return label
    return bins[-1][0]  # fallback to last bin


def discretize(features: Dict) -> DiscreteState:
    """
    Convert raw problem features (dict with float values) to a discrete state.

    Accepts either:
    - Phase 1's ProblemFeatures dict (coupling_estimate, decomposability, ...)
    - Phase 0's complexity_features dict (similar keys)
    """
    # Normalize key names (handle both formats)
    coupling = features.get("coupling_estimate", features.get("coupling", 0.5))
    decomp = features.get("decomposability", 0.5)
    baseline = features.get("has_baseline", 0.5)
    if isinstance(baseline, bool):
        baseline = 1.0 if baseline else 0.0
    info = features.get("information_completeness", features.get("info_completeness", 0.5))
    comp_count = features.get("component_count", 5)
    if isinstance(comp_count, str):
        comp_count = {"few": 2, "moderate": 6, "many": 10}.get(comp_count, 5)

    return DiscreteState(
        coupling=_bin_value(float(coupling), FEATURE_BINS["coupling_estimate"]),
        decomposability=_bin_value(float(decomp), FEATURE_BINS["decomposability"]),
        has_baseline=_bin_value(float(baseline), FEATURE_BINS["has_baseline"]),
        info_completeness=_bin_value(float(info), FEATURE_BINS["information_completeness"]),
        component_count=_bin_value(float(comp_count), FEATURE_BINS["component_count"]),
    )


def build_full_state_space() -> list:
    """Generate all 108 possible discrete states."""
    states = []
    for c in ["low", "medium", "high"]:
        for d in ["low", "high"]:
            for b in ["no", "yes"]:
                for i in ["low", "medium", "high"]:
                    for n in ["few", "moderate", "many"]:
                        states.append(DiscreteState(c, d, b, i, n))
    assert len(states) == STATE_SPACE_SIZE
    return states
