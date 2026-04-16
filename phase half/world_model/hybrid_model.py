"""
Hybrid World Model (Route C from Phase 0.5 dev doc).
Combines statistical model + LLM simulator + OOD detection.
"""

import random
from typing import Dict, Optional, List
from dataclasses import dataclass

from .statistical_model import StatisticalWorldModel, WorldModelPrediction
from .llm_simulator import LLMSimulator
from .ood_detector import OODDetector
from .feature_discretizer import DiscreteState, discretize


@dataclass
class SequenceSimulation:
    """Result of simulating a strategy sequence (COMP_*)."""
    steps: List[WorldModelPrediction]
    overall_success_probability: float
    best_step_index: int
    total_predicted_steps: int


class HybridWorldModel:
    """
    The main world model used by Phase 1's dispatcher training.

    Decision logic:
    - OOD → return uninformative prior + force real execution
    - Statistical confidence ≥ 0.6 → use statistical model
    - Statistical confidence 0.3-0.6 → fuse statistical + LLM
    - Statistical confidence < 0.3 → use LLM simulator
    """

    def __init__(self, strategy_kb: Dict = None):
        self.stat_model = StatisticalWorldModel()
        self.llm_model = LLMSimulator(strategy_kb=strategy_kb)
        self.ood_detector = OODDetector()

    def predict(self, features: Dict, strategy_id: str,
                strategy_name: str = "", strategy_description: str = "",
                previous_attempts: str = "") -> WorldModelPrediction:
        """
        Predict strategy outcome for given problem features.

        Returns binary-compatible evaluation (matching real environment)
        with predicted_probability available for analysis.
        """
        state = discretize(features)

        # OOD check
        ood = self.ood_detector.check(features)
        if ood.is_ood:
            return WorldModelPrediction(
                predicted_success_probability=0.5,
                prediction_confidence=0.0,
                predicted_failure_modes=[f"OOD: {ood.reason}"],
                is_ood=True,
                source="ood",
            )

        # Statistical model prediction
        stat_pred = self.stat_model.predict(state, strategy_id)

        if stat_pred.prediction_confidence >= 0.6:
            return stat_pred

        elif stat_pred.prediction_confidence >= 0.3:
            # Fuse statistical + LLM
            llm_pred = self.llm_model.predict(
                state, strategy_id, strategy_name, strategy_description,
                previous_attempts
            )
            return self._fuse(stat_pred, llm_pred)

        else:
            # Pure LLM
            return self.llm_model.predict(
                state, strategy_id, strategy_name, strategy_description,
                previous_attempts
            )

    def _fuse(self, stat_pred: WorldModelPrediction,
              llm_pred: WorldModelPrediction) -> WorldModelPrediction:
        """Weighted fusion of statistical and LLM predictions."""
        w_stat = stat_pred.prediction_confidence
        w_llm = 1 - w_stat

        fused_prob = (
            w_stat * stat_pred.predicted_success_probability +
            w_llm * llm_pred.predicted_success_probability
        )

        return WorldModelPrediction(
            predicted_success_probability=fused_prob,
            prediction_confidence=max(
                stat_pred.prediction_confidence,
                llm_pred.prediction_confidence,
            ),
            predicted_failure_modes=llm_pred.predicted_failure_modes,
            predicted_steps_to_complete=llm_pred.predicted_steps_to_complete,
            source="hybrid",
        )

    def update(self, features: Dict, strategy_id: str, success: bool,
               selector_confidence: float = 1.0,
               strategy_consistency: float = 1.0,
               is_simulated: bool = False):
        """Update with new execution record."""
        self.stat_model.update(
            features, strategy_id, success,
            selector_confidence=selector_confidence,
            strategy_consistency=strategy_consistency,
            is_simulated=is_simulated,
        )
        if not is_simulated:
            self.ood_detector.update(features)

    def simulate_execution(self, features: Dict, strategy_id: str,
                           strategy_name: str = "",
                           strategy_description: str = "") -> Dict:
        """
        Simulate execution and return binary outcome (matching real environment).
        Phase 0.5 fix: evaluation_score is binary, not continuous probability.
        """
        pred = self.predict(features, strategy_id, strategy_name, strategy_description)
        prob = pred.predicted_success_probability

        # Sample binary outcome
        simulated_success = random.random() < prob
        simulated_partial = (not simulated_success and random.random() < 0.3)

        # Binary evaluation score (matching real environment)
        if simulated_success:
            eval_score = 1.0
        elif simulated_partial:
            eval_score = 0.5
        else:
            eval_score = 0.0

        return {
            "success": simulated_success,
            "partial_success": simulated_partial,
            "evaluation_score": eval_score,
            "predicted_probability": prob,  # For analysis only
            "prediction_confidence": pred.prediction_confidence,
            "predicted_failure_modes": pred.predicted_failure_modes,
            "predicted_steps": pred.predicted_steps_to_complete,
            "is_simulated": True,
            "is_ood": pred.is_ood,
            "source": pred.source,
        }

    def simulate_strategy_sequence(
        self, features: Dict, strategy_sequence: List[str],
        strategy_kb: Dict = None, max_steps: int = 3
    ) -> SequenceSimulation:
        """Simulate a strategy composition (COMP_*)."""
        kb = strategy_kb or self.llm_model.strategy_kb
        results = []
        prev_attempts = ""

        for i, sid in enumerate(strategy_sequence[:max_steps]):
            name = kb.get(sid, {}).get("name", {}).get("zh", sid) if kb else sid
            desc = kb.get(sid, {}).get("description", {}).get("one_sentence", "") if kb else ""

            pred = self.predict(
                features, sid, name, desc, prev_attempts
            )
            results.append(pred)

            if pred.predicted_success_probability > 0.7:
                break

            prev_attempts += f"策略 {sid} 预测成功率 {pred.predicted_success_probability:.0%}; "

        # Overall: 1 - P(all fail)
        overall = 1.0 - float(
            __import__("numpy").prod([
                1.0 - r.predicted_success_probability for r in results
            ])
        )
        best_idx = int(
            __import__("numpy").argmax([
                r.predicted_success_probability for r in results
            ])
        )

        return SequenceSimulation(
            steps=results,
            overall_success_probability=overall,
            best_step_index=best_idx,
            total_predicted_steps=sum(r.predicted_steps_to_complete for r in results),
        )

    def get_stats(self) -> Dict:
        """Return model statistics."""
        return {
            "statistical": self.stat_model.get_coverage_stats(),
            "ood_seen_states": len(self.ood_detector.seen_states),
            "llm_correction_factor": self.llm_model.correction_factor,
        }
