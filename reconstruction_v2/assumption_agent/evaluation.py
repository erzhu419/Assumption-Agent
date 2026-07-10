from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Protocol, Sequence

from .events import Event, EventSink, NullEventSink
from .models import (
    CounterfactualPair,
    ExternalOutcome,
    HypothesisProgram,
    HypothesisStatus,
    RuntimeExecution,
    SplitName,
    TaskInput,
    stable_hash,
)
from .runtime import PolicyRuntime


class ExternalEvaluator(Protocol):
    id: str
    epoch: str

    def evaluate(self, task: TaskInput, execution: RuntimeExecution) -> ExternalOutcome: ...


class CounterfactualRunner:
    def __init__(
        self,
        *,
        runtime: PolicyRuntime,
        evaluator: ExternalEvaluator,
        event_sink: EventSink | None = None,
    ) -> None:
        self.runtime = runtime
        self.evaluator = evaluator
        self.event_sink = event_sink or NullEventSink()

    def run(
        self,
        tasks: Sequence[TaskInput],
        *,
        program: HypothesisProgram,
        baseline_programs: Sequence[HypothesisProgram] = (),
        split: SplitName,
        trace_id: str = "counterfactual",
    ) -> tuple[CounterfactualPair, ...]:
        if program.evaluator_epoch != self.evaluator.epoch:
            raise ValueError("program and external evaluator must use the same frozen epoch")
        pairs: list[CounterfactualPair] = []
        for task in tasks:
            pair_trace = stable_hash({"trace_id": trace_id, "task_id": task.id, "program_id": program.id})[:20]
            baseline = self.runtime.execute(
                task,
                baseline_programs,
                allowed_statuses={HypothesisStatus.PROMOTED},
                trace_id=f"{pair_trace}:off",
            )
            candidate = self.runtime.execute(
                task,
                (*baseline_programs, program),
                allowed_statuses={
                    HypothesisStatus.CANDIDATE,
                    HypothesisStatus.SHADOW,
                    HypothesisStatus.PROMOTED,
                },
                trace_id=f"{pair_trace}:on",
            )
            baseline_outcome = self.evaluator.evaluate(task, baseline)
            candidate_outcome = self.evaluator.evaluate(task, candidate)
            if baseline_outcome.evaluator_epoch != candidate_outcome.evaluator_epoch:
                raise ValueError("counterfactual outcomes crossed evaluator epochs")
            pair = CounterfactualPair(
                task_id=task.id,
                split=split,
                evaluator_epoch=self.evaluator.epoch,
                baseline=baseline,
                candidate=candidate,
                baseline_outcome=baseline_outcome,
                candidate_outcome=candidate_outcome,
            )
            pairs.append(pair)
            self.event_sink.emit(
                Event(
                    event="counterfactual_pair_completed",
                    stage="evaluation.counterfactual",
                    trace_id=pair_trace,
                    payload={
                        "task_id_hash": stable_hash({"task_id": task.id}),
                        "split": split.value,
                        "hypothesis_id": program.id,
                        "baseline_hypothesis_ids": [row.id for row in baseline_programs],
                        "evaluator_id": self.evaluator.id,
                        "evaluator_epoch": self.evaluator.epoch,
                        "baseline_success": baseline_outcome.success,
                        "candidate_success": candidate_outcome.success,
                        "baseline_score": baseline_outcome.score,
                        "candidate_score": candidate_outcome.score,
                        "baseline_cost": baseline.total_cost,
                        "candidate_cost": candidate.total_cost,
                        "action_activated": candidate.action_activated,
                        "selection_changed": (
                            stable_hash({"answer": baseline.selected_result.answer})
                            != stable_hash({"answer": candidate.selected_result.answer})
                        ),
                        "baseline_preserved": candidate.baseline_preserved,
                    },
                )
            )
        return tuple(pairs)


@dataclass(frozen=True)
class PairSummary:
    pair_count: int
    baseline_success_count: int
    candidate_success_count: int
    gain_count: int
    harm_count: int
    tie_count: int
    activation_count: int
    selection_change_count: int
    baseline_preserved_count: int
    invalid_pair_count: int
    provider_mismatch_count: int
    budget_mismatch_count: int
    baseline_mean_cost: float
    candidate_mean_cost: float
    cost_ratio: float
    mean_effect: float
    effect_standard_error: float

    @property
    def harm_rate(self) -> float:
        return self.harm_count / self.pair_count if self.pair_count else 1.0

    @property
    def activation_rate(self) -> float:
        return self.activation_count / self.pair_count if self.pair_count else 0.0

    def effect_lower_bound(self, confidence: float) -> float:
        if not 0.5 < confidence < 1.0:
            raise ValueError("confidence must be between 0.5 and 1.0")
        z_score = statistics.NormalDist().inv_cdf(confidence)
        return self.mean_effect - z_score * self.effect_standard_error

    def to_dict(self, *, confidence: float = 0.9) -> dict[str, float | int]:
        return {
            "pair_count": self.pair_count,
            "baseline_success_count": self.baseline_success_count,
            "candidate_success_count": self.candidate_success_count,
            "gain_count": self.gain_count,
            "harm_count": self.harm_count,
            "tie_count": self.tie_count,
            "activation_count": self.activation_count,
            "selection_change_count": self.selection_change_count,
            "baseline_preserved_count": self.baseline_preserved_count,
            "invalid_pair_count": self.invalid_pair_count,
            "provider_mismatch_count": self.provider_mismatch_count,
            "budget_mismatch_count": self.budget_mismatch_count,
            "baseline_mean_cost": self.baseline_mean_cost,
            "candidate_mean_cost": self.candidate_mean_cost,
            "cost_ratio": self.cost_ratio,
            "mean_effect": self.mean_effect,
            "effect_standard_error": self.effect_standard_error,
            "effect_lower_bound": self.effect_lower_bound(confidence),
            "harm_rate": self.harm_rate,
            "activation_rate": self.activation_rate,
        }


def summarize_pairs(pairs: Sequence[CounterfactualPair]) -> PairSummary:
    differences = [
        float(pair.candidate_outcome.success) - float(pair.baseline_outcome.success)
        for pair in pairs
    ]
    baseline_costs = [pair.baseline.total_cost for pair in pairs]
    candidate_costs = [pair.candidate.total_cost for pair in pairs]
    baseline_mean_cost = statistics.fmean(baseline_costs) if baseline_costs else 0.0
    candidate_mean_cost = statistics.fmean(candidate_costs) if candidate_costs else 0.0
    if baseline_mean_cost > 0:
        cost_ratio = candidate_mean_cost / baseline_mean_cost
    else:
        cost_ratio = 1.0 if candidate_mean_cost == 0 else math.inf
    return PairSummary(
        pair_count=len(pairs),
        baseline_success_count=sum(pair.baseline_outcome.success for pair in pairs),
        candidate_success_count=sum(pair.candidate_outcome.success for pair in pairs),
        gain_count=sum(not pair.baseline_outcome.success and pair.candidate_outcome.success for pair in pairs),
        harm_count=sum(pair.baseline_outcome.success and not pair.candidate_outcome.success for pair in pairs),
        tie_count=sum(pair.baseline_outcome.success == pair.candidate_outcome.success for pair in pairs),
        activation_count=sum(pair.candidate.action_activated for pair in pairs),
        selection_change_count=sum(
            stable_hash({"answer": pair.baseline.selected_result.answer})
            != stable_hash({"answer": pair.candidate.selected_result.answer})
            for pair in pairs
        ),
        baseline_preserved_count=sum(pair.candidate.baseline_preserved for pair in pairs),
        invalid_pair_count=sum(
            min(
                pair.baseline_outcome.metrics.get("evaluation_valid", 1.0),
                pair.candidate_outcome.metrics.get("evaluation_valid", 1.0),
            )
            < 1.0
            for pair in pairs
        ),
        provider_mismatch_count=sum(
            pair.baseline.selected_result.metadata.get("provider_fingerprint")
            != pair.candidate.selected_result.metadata.get("provider_fingerprint")
            for pair in pairs
            if pair.baseline.selected_result.metadata.get("provider_fingerprint") is not None
            or pair.candidate.selected_result.metadata.get("provider_fingerprint") is not None
        ),
        budget_mismatch_count=sum(
            pair.baseline.selected_result.metadata.get("fairness_fingerprint")
            != pair.candidate.selected_result.metadata.get("fairness_fingerprint")
            for pair in pairs
            if pair.baseline.selected_result.metadata.get("fairness_fingerprint") is not None
            or pair.candidate.selected_result.metadata.get("fairness_fingerprint") is not None
        ),
        baseline_mean_cost=round(baseline_mean_cost, 6),
        candidate_mean_cost=round(candidate_mean_cost, 6),
        cost_ratio=round(cost_ratio, 6) if math.isfinite(cost_ratio) else math.inf,
        mean_effect=statistics.fmean(differences) if differences else 0.0,
        effect_standard_error=(statistics.stdev(differences) / math.sqrt(len(differences))) if len(differences) > 1 else 1.0,
    )


@dataclass(frozen=True)
class PromotionGateSpec:
    minimum_pairs: int = 20
    confidence: float = 0.9
    minimum_net_gain_count: int = 1
    minimum_activation_rate: float = 0.1


@dataclass(frozen=True)
class PromotionDecision:
    allowed: bool
    blockers: tuple[str, ...]
    summary: PairSummary
    effect_lower_bound: float
    evaluator_epoch: str
    confidence: float
    policy: str = "paired_validation_lower_bound_v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "blockers": list(self.blockers),
            "summary": self.summary.to_dict(confidence=self.confidence),
            "effect_lower_bound": self.effect_lower_bound,
            "evaluator_epoch": self.evaluator_epoch,
            "policy": self.policy,
        }


class PromotionGate:
    def __init__(self, spec: PromotionGateSpec | None = None, *, event_sink: EventSink | None = None) -> None:
        self.spec = spec or PromotionGateSpec()
        self.event_sink = event_sink or NullEventSink()

    def evaluate(
        self,
        program: HypothesisProgram,
        pairs: Sequence[CounterfactualPair],
        *,
        sealed_test_accessed: bool,
        trace_id: str = "promotion",
    ) -> PromotionDecision:
        summary = summarize_pairs(pairs)
        lower_bound = summary.effect_lower_bound(self.spec.confidence)
        blockers: list[str] = []
        if not pairs or any(pair.split is not SplitName.VALIDATION for pair in pairs):
            blockers.append("promotion_requires_validation_pairs_only")
        if any(pair.evaluator_epoch != program.evaluator_epoch for pair in pairs):
            blockers.append("mixed_or_wrong_evaluator_epoch")
        if sealed_test_accessed:
            blockers.append("sealed_test_accessed_before_promotion")
        if summary.pair_count < self.spec.minimum_pairs:
            blockers.append("insufficient_paired_validation_rows")
        if summary.gain_count - summary.harm_count < self.spec.minimum_net_gain_count:
            blockers.append("insufficient_net_gain_count")
        if summary.activation_rate < self.spec.minimum_activation_rate:
            blockers.append("insufficient_runtime_activation")
        if summary.baseline_preserved_count != summary.pair_count:
            blockers.append("slow_baseline_not_preserved")
        if summary.invalid_pair_count:
            blockers.append("invalid_counterfactual_pairs")
        if summary.provider_mismatch_count:
            blockers.append("counterfactual_provider_mismatch")
        if summary.budget_mismatch_count:
            blockers.append("counterfactual_budget_mismatch")
        if summary.harm_rate > program.expected_effect.maximum_harm_rate:
            blockers.append("harm_rate_exceeded")
        if summary.cost_ratio > program.expected_effect.maximum_cost_ratio:
            blockers.append("cost_ratio_exceeded")
        if lower_bound < program.expected_effect.minimum_delta:
            blockers.append("paired_effect_lower_bound_below_target")
        decision = PromotionDecision(
            allowed=not blockers,
            blockers=tuple(blockers),
            summary=summary,
            effect_lower_bound=lower_bound,
            evaluator_epoch=program.evaluator_epoch,
            confidence=self.spec.confidence,
        )
        self.event_sink.emit(
            Event(
                event="hypothesis_promotion_decided",
                stage="promotion",
                trace_id=trace_id,
                payload={
                    "hypothesis_id": program.id,
                    "hypothesis_hash": program.payload_hash,
                    "allowed": decision.allowed,
                    "blockers": list(decision.blockers),
                    "pair_summary": summary.to_dict(confidence=self.spec.confidence),
                    "evaluator_epoch": program.evaluator_epoch,
                    "sealed_test_accessed": sealed_test_accessed,
                },
            )
        )
        return decision
