"""
Out-of-Distribution detector for the world model.
Forces real execution when problem features are completely unseen.
"""

import numpy as np
from dataclasses import dataclass
from typing import Set, List, Optional

from .feature_discretizer import DiscreteState, discretize


@dataclass
class OODResult:
    is_ood: bool
    reason: str = ""
    recommendation: str = ""  # "force_real_execution" | "ok"


class OODDetector:
    """
    Detects if a problem's features are outside the world model's training distribution.

    Two checks:
    1. Discrete state never seen before
    2. Continuous feature vector too far from all training data (k-NN distance)
    """

    def __init__(self):
        self.seen_states: Set[tuple] = set()
        self._continuous_vectors: List[np.ndarray] = []
        self._distance_threshold: float = float("inf")  # Until calibrated

    def update(self, features: dict):
        """Record a new observed problem."""
        state = discretize(features)
        self.seen_states.add(state.to_tuple())

        vec = self._features_to_continuous(features)
        self._continuous_vectors.append(vec)

        # Recalibrate threshold periodically
        if len(self._continuous_vectors) % 50 == 0 and len(self._continuous_vectors) >= 20:
            self._calibrate_threshold()

    def check(self, features: dict) -> OODResult:
        """Check if features are OOD."""
        state = discretize(features)

        # Check 1: discrete state never seen
        if state.to_tuple() not in self.seen_states:
            return OODResult(
                is_ood=True,
                reason="discrete_state_unseen",
                recommendation="force_real_execution",
            )

        # Check 2: continuous distance (only if we have enough data)
        if len(self._continuous_vectors) >= 20:
            vec = self._features_to_continuous(features)
            vectors = np.array(self._continuous_vectors)
            dists = np.linalg.norm(vectors - vec, axis=1)
            min_5_avg = np.sort(dists)[:5].mean()

            if min_5_avg > self._distance_threshold:
                return OODResult(
                    is_ood=True,
                    reason=f"knn_distance={min_5_avg:.2f} > threshold={self._distance_threshold:.2f}",
                    recommendation="force_real_execution",
                )

        return OODResult(is_ood=False, recommendation="ok")

    def _calibrate_threshold(self):
        """Set threshold at 95th percentile of internal k-NN distances."""
        if len(self._continuous_vectors) < 20:
            return

        vectors = np.array(self._continuous_vectors)
        internal_distances = []

        # Sample to keep this O(n) not O(n²)
        n = len(vectors)
        sample_size = min(n, 100)
        indices = np.random.choice(n, sample_size, replace=False)

        for i in indices:
            dists = np.linalg.norm(vectors - vectors[i], axis=1)
            dists_sorted = np.sort(dists)[1:6]  # skip self, take k=5
            internal_distances.append(dists_sorted.mean())

        self._distance_threshold = float(np.percentile(internal_distances, 95))

    @staticmethod
    def _features_to_continuous(features: dict) -> np.ndarray:
        """Extract continuous feature vector from raw features dict."""
        return np.array([
            float(features.get("coupling_estimate", features.get("coupling", 0.5))),
            float(features.get("decomposability", 0.5)),
            float(1.0 if features.get("has_baseline", False) else 0.0)
                if isinstance(features.get("has_baseline"), bool)
                else float(features.get("has_baseline", 0.5)),
            float(features.get("information_completeness",
                              features.get("info_completeness", 0.5))),
            float(features.get("component_count", 5)) / 10.0,  # normalize
        ], dtype=np.float32)
