"""
World Model Calibrator.
Measures prediction accuracy including:
- Kendall's tau (ranking accuracy)
- Veto precision (can it reliably say "this won't work"?)
- Stratified calibration (high-data vs low-data accuracy)
- LLM optimism bias measurement
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from scipy.stats import kendalltau


@dataclass
class CalibrationReport:
    kendall_tau: float
    tau_p_value: float
    veto_precision: Optional[float]
    n_samples: int
    stratified: Optional["StratifiedReport"] = None
    optimism_bias: Optional["OptimismBias"] = None
    needs_rebuild: bool = False
    force_real_for_rare: bool = False


@dataclass
class StratifiedReport:
    tau_high_data: Optional[float]
    tau_low_data: Optional[float]
    n_high: int
    n_low: int
    force_real_for_rare: bool


@dataclass
class OptimismBias:
    bias: float  # positive = optimistic
    mean_predicted: float
    mean_actual: float
    is_significant: bool
    direction: str  # "optimistic" | "pessimistic" | "neutral"


class WorldModelCalibrator:
    """
    Calibrates world model predictions against real execution results.
    Run periodically (every 500 episodes or monthly).
    """

    def calibrate(
        self,
        world_model,  # HybridWorldModel
        real_records: List[Dict],
        n_calibration: int = 50,
    ) -> CalibrationReport:
        """
        Full calibration using recent real execution records.

        Each record should have:
        - "task.complexity_features": dict of problem features
        - "strategy_selection.selected_strategy": strategy ID
        - "outcome.success": bool
        """
        records = real_records[-n_calibration:]
        if len(records) < 10:
            return CalibrationReport(
                kendall_tau=0.0, tau_p_value=1.0, veto_precision=None,
                n_samples=len(records),
            )

        predictions = []
        actuals = []

        for record in records:
            features = record.get("task", {}).get("complexity_features", {})
            strategy = record.get("strategy_selection", {}).get("selected_strategy", "")
            success = record.get("outcome", {}).get("success", False)

            if not features or not strategy:
                continue

            pred = world_model.predict(features, strategy)
            predictions.append(pred.predicted_success_probability)
            actuals.append(1.0 if success else 0.0)

        if len(predictions) < 10:
            return CalibrationReport(
                kendall_tau=0.0, tau_p_value=1.0, veto_precision=None,
                n_samples=len(predictions),
            )

        # Kendall's tau
        tau, p_value = kendalltau(predictions, actuals)

        # Veto precision: when model says < 0.2, is it actually bad?
        low_pred_idx = [i for i, p in enumerate(predictions) if p < 0.2]
        if low_pred_idx:
            veto_precision = sum(
                1 for i in low_pred_idx if actuals[i] == 0.0
            ) / len(low_pred_idx)
        else:
            veto_precision = None

        # Stratified calibration
        stratified = self._stratified_calibration(
            world_model, records
        )

        # Optimism bias
        optimism = self._measure_optimism_bias(predictions, actuals)

        # Apply corrections
        needs_rebuild = tau < 0.3
        if needs_rebuild:
            world_model.stat_model.table.clear()
            # Rebuild from records
            for record in real_records:
                features = record.get("task", {}).get("complexity_features", {})
                strategy = record.get("strategy_selection", {}).get("selected_strategy", "")
                success = record.get("outcome", {}).get("success", False)
                if features and strategy:
                    world_model.stat_model.update(features, strategy, success)

        if optimism.is_significant:
            world_model.llm_model.correction_factor = -optimism.bias

        return CalibrationReport(
            kendall_tau=tau,
            tau_p_value=p_value,
            veto_precision=veto_precision,
            n_samples=len(predictions),
            stratified=stratified,
            optimism_bias=optimism,
            needs_rebuild=needs_rebuild,
            force_real_for_rare=stratified.force_real_for_rare if stratified else False,
        )

    def _stratified_calibration(
        self, world_model, records: List[Dict]
    ) -> StratifiedReport:
        """Separate accuracy for high-data vs low-data regions."""
        from .feature_discretizer import discretize

        high_preds, high_actuals = [], []
        low_preds, low_actuals = [], []

        for record in records:
            features = record.get("task", {}).get("complexity_features", {})
            strategy = record.get("strategy_selection", {}).get("selected_strategy", "")
            success = record.get("outcome", {}).get("success", False)

            if not features or not strategy:
                continue

            state = discretize(features)
            key = (state.to_tuple(), strategy)
            stats = world_model.stat_model.table.get(key)
            count = stats.count if stats else 0

            pred = world_model.predict(features, strategy)
            actual = 1.0 if success else 0.0

            if count >= 20:
                high_preds.append(pred.predicted_success_probability)
                high_actuals.append(actual)
            elif count < 5:
                low_preds.append(pred.predicted_success_probability)
                low_actuals.append(actual)

        tau_high = kendalltau(high_preds, high_actuals)[0] if len(high_preds) >= 10 else None
        tau_low = kendalltau(low_preds, low_actuals)[0] if len(low_preds) >= 10 else None

        return StratifiedReport(
            tau_high_data=tau_high,
            tau_low_data=tau_low,
            n_high=len(high_preds),
            n_low=len(low_preds),
            force_real_for_rare=(tau_low is not None and tau_low < 0.2),
        )

    @staticmethod
    def _measure_optimism_bias(
        predictions: List[float], actuals: List[float]
    ) -> OptimismBias:
        """Measure if predictions are systematically too high or low."""
        mean_pred = np.mean(predictions)
        mean_actual = np.mean(actuals)
        bias = mean_pred - mean_actual

        return OptimismBias(
            bias=bias,
            mean_predicted=mean_pred,
            mean_actual=mean_actual,
            is_significant=(abs(bias) > 0.15),
            direction="optimistic" if bias > 0.05 else (
                "pessimistic" if bias < -0.05 else "neutral"
            ),
        )
