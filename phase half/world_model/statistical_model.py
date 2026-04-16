"""
Statistical World Model (Route A from Phase 0.5 dev doc).
Lookup table + k-NN interpolation for predicting strategy success probability.
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

from .feature_discretizer import DiscreteState, discretize


@dataclass
class RunningStats:
    """Online mean/count tracker for a (state, strategy) pair."""
    successes: int = 0
    total: int = 0

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.5  # uninformative prior
        return self.successes / self.total

    @property
    def count(self) -> int:
        return self.total

    def add(self, success: bool):
        self.total += 1
        if success:
            self.successes += 1


@dataclass
class WorldModelPrediction:
    predicted_success_probability: float
    prediction_confidence: float  # 0 = no data, 1 = very confident
    predicted_failure_modes: List[str] = field(default_factory=list)
    predicted_steps_to_complete: int = 10
    condition_match_score: float = 0.5
    is_ood: bool = False
    source: str = "statistical"  # "statistical" | "llm" | "hybrid"


class StatisticalWorldModel:
    """
    Lookup table based world model.
    - O(1) prediction for known (state, strategy) pairs
    - k-NN interpolation for unknown pairs
    - Quality filtering on updates (Phase 0.5 dev doc fix)
    """

    def __init__(self):
        self.table: Dict[Tuple, RunningStats] = defaultdict(RunningStats)
        # For k-NN: store feature vectors alongside discrete states
        self._feature_vectors: List[np.ndarray] = []
        self._feature_labels: List[Tuple] = []  # (state_tuple, strategy_id)
        self._feature_outcomes: List[float] = []

    def predict(self, state: DiscreteState, strategy_id: str) -> WorldModelPrediction:
        key = (state.to_tuple(), strategy_id)
        stats = self.table.get(key)

        if stats and stats.count >= 10:
            # Enough data: use direct lookup
            return WorldModelPrediction(
                predicted_success_probability=stats.success_rate,
                prediction_confidence=min(0.9, stats.count / 50),
                source="statistical",
            )
        elif stats and stats.count >= 3:
            # Some data: lower confidence
            return WorldModelPrediction(
                predicted_success_probability=stats.success_rate,
                prediction_confidence=stats.count / 20,
                source="statistical",
            )
        else:
            # No data or very little: try k-NN interpolation
            return self._knn_predict(state, strategy_id)

    def update(self, features: Dict, strategy_id: str, success: bool,
               selector_confidence: float = 1.0,
               strategy_consistency: float = 1.0,
               is_simulated: bool = False):
        """
        Update with a new execution record.
        Includes quality filtering (Phase 0.5 dev doc section 1.2 fix).
        """
        # Quality filter 1: low dispatcher confidence = random exploration
        if selector_confidence < 0.2:
            return

        # Quality filter 2: executor didn't follow the strategy
        if strategy_consistency < 0.3:
            return

        # Quality filter 3: simulated results don't update statistical model
        if is_simulated:
            return

        state = discretize(features)
        key = (state.to_tuple(), strategy_id)
        self.table[key].add(success)

        # Store for k-NN
        vec = self._state_to_vector(state, strategy_id)
        self._feature_vectors.append(vec)
        self._feature_labels.append(key)
        self._feature_outcomes.append(1.0 if success else 0.0)

    def _knn_predict(self, state: DiscreteState, strategy_id: str,
                     k: int = 5) -> WorldModelPrediction:
        """k-NN interpolation for unseen (state, strategy) pairs."""
        if len(self._feature_vectors) < k:
            # Not enough data for k-NN
            return WorldModelPrediction(
                predicted_success_probability=0.5,
                prediction_confidence=0.0,
                source="statistical_nodata",
            )

        query = self._state_to_vector(state, strategy_id)
        vectors = np.array(self._feature_vectors)
        outcomes = np.array(self._feature_outcomes)

        # L2 distances
        dists = np.linalg.norm(vectors - query, axis=1)
        nearest_idx = np.argsort(dists)[:k]
        nearest_outcomes = outcomes[nearest_idx]
        nearest_dists = dists[nearest_idx]

        # Distance-weighted average
        weights = 1.0 / (nearest_dists + 1e-6)
        pred = np.average(nearest_outcomes, weights=weights)

        return WorldModelPrediction(
            predicted_success_probability=float(pred),
            prediction_confidence=0.3,  # k-NN is always lower confidence
            source="statistical_knn",
        )

    @staticmethod
    def _state_to_vector(state: DiscreteState, strategy_id: str) -> np.ndarray:
        """Encode (state, strategy) as a numeric vector for k-NN."""
        # State encoding: ordinal
        coupling_map = {"low": 0, "medium": 1, "high": 2}
        binary_map = {"no": 0, "yes": 1, "low": 0, "high": 1}
        info_map = {"low": 0, "medium": 1, "high": 2}
        comp_map = {"few": 0, "moderate": 1, "many": 2}

        state_vec = [
            coupling_map.get(state.coupling, 1),
            binary_map.get(state.decomposability, 0),
            binary_map.get(state.has_baseline, 0),
            info_map.get(state.info_completeness, 1),
            comp_map.get(state.component_count, 1),
        ]

        # Strategy encoding: hash to a number
        strategy_hash = hash(strategy_id) % 100 / 100.0
        state_vec.append(strategy_hash)

        return np.array(state_vec, dtype=np.float32)

    def get_coverage_stats(self) -> Dict:
        """Return statistics about data coverage."""
        non_empty = sum(1 for s in self.table.values() if s.total > 0)
        total_records = sum(s.total for s in self.table.values())
        return {
            "non_empty_cells": non_empty,
            "total_records": total_records,
            "knn_vectors": len(self._feature_vectors),
        }
