from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class HypothesisKind(str, Enum):
    TASK = "task"
    POLICY = "policy"
    EVALUATOR = "evaluator"


class HypothesisStatus(str, Enum):
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"


class SplitName(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


_MISSING = object()


@dataclass(frozen=True)
class FeaturePredicate:
    key: str
    op: str
    value: Any = None

    def matches(self, features: Mapping[str, Any]) -> bool:
        actual = features.get(self.key, _MISSING)
        if self.op == "exists":
            return (actual is not _MISSING) is bool(self.value)
        if actual is _MISSING:
            return False
        if self.op == "eq":
            return actual == self.value
        if self.op == "ne":
            return actual != self.value
        if self.op == "in":
            return isinstance(self.value, (list, tuple, set, frozenset)) and actual in self.value
        if self.op == "contains":
            try:
                return self.value in actual
            except TypeError:
                return False
        if self.op == "gte":
            try:
                return actual >= self.value
            except TypeError:
                return False
        if self.op == "lte":
            try:
                return actual <= self.value
            except TypeError:
                return False
        raise ValueError(f"unsupported predicate operator: {self.op}")

    def validate(self) -> list[str]:
        issues: list[str] = []
        if not self.key.strip():
            issues.append("predicate_key_missing")
        if self.op not in {"exists", "eq", "ne", "in", "contains", "gte", "lte"}:
            issues.append("predicate_operator_invalid")
        return issues

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeaturePredicate":
        return cls(key=str(data.get("key") or ""), op=str(data.get("op") or ""), value=data.get("value"))


@dataclass(frozen=True)
class TriggerSpec:
    all_of: tuple[FeaturePredicate, ...] = ()
    any_of: tuple[FeaturePredicate, ...] = ()
    none_of: tuple[FeaturePredicate, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (self.all_of or self.any_of or self.none_of)

    def matches(self, features: Mapping[str, Any]) -> bool:
        if self.is_empty:
            return True
        return (
            all(predicate.matches(features) for predicate in self.all_of)
            and (not self.any_of or any(predicate.matches(features) for predicate in self.any_of))
            and not any(predicate.matches(features) for predicate in self.none_of)
        )

    def validate(self, *, require_positive: bool = False) -> list[str]:
        issues = [issue for predicate in (*self.all_of, *self.any_of, *self.none_of) for issue in predicate.validate()]
        if require_positive and not (self.all_of or self.any_of):
            issues.append("positive_trigger_missing")
        return issues

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "TriggerSpec":
        data = data or {}
        return cls(
            all_of=tuple(FeaturePredicate.from_dict(row) for row in data.get("all_of", []) if isinstance(row, Mapping)),
            any_of=tuple(FeaturePredicate.from_dict(row) for row in data.get("any_of", []) if isinstance(row, Mapping)),
            none_of=tuple(FeaturePredicate.from_dict(row) for row in data.get("none_of", []) if isinstance(row, Mapping)),
        )


@dataclass(frozen=True)
class ActionNode:
    id: str
    operation: str
    target: str
    value: Any = None
    depends_on: tuple[str, ...] = ()

    def validate(self) -> list[str]:
        issues: list[str] = []
        if not self.id.strip():
            issues.append("action_id_missing")
        if self.operation not in {
            "enable_lane",
            "disable_lane",
            "prioritize_lane",
            "set_parameter",
            "require_verifier",
            "abstain",
            "execute_step",
            "check_condition",
            "produce_artifact",
            "request_evidence",
        }:
            issues.append("action_operation_invalid")
        if self.operation != "abstain" and not self.target.strip():
            issues.append("action_target_missing")
        return issues

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionNode":
        return cls(
            id=str(data.get("id") or ""),
            operation=str(data.get("operation") or ""),
            target=str(data.get("target") or ""),
            value=data.get("value"),
            depends_on=tuple(str(value) for value in data.get("depends_on", [])),
        )


@dataclass(frozen=True)
class ExpectedEffect:
    metric: str = "task_success"
    minimum_delta: float = 0.0
    maximum_harm_rate: float = 0.05
    maximum_cost_ratio: float = 1.5

    def validate(self) -> list[str]:
        issues: list[str] = []
        if not self.metric.strip():
            issues.append("effect_metric_missing")
        if not -1.0 <= self.minimum_delta <= 1.0:
            issues.append("minimum_delta_out_of_range")
        if not 0.0 <= self.maximum_harm_rate <= 1.0:
            issues.append("maximum_harm_rate_out_of_range")
        if self.maximum_cost_ratio < 1.0:
            issues.append("maximum_cost_ratio_below_one")
        return issues

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ExpectedEffect":
        data = data or {}
        return cls(
            metric=str(data.get("metric") or "task_success"),
            minimum_delta=float(data.get("minimum_delta", 0.0)),
            maximum_harm_rate=float(data.get("maximum_harm_rate", 0.05)),
            maximum_cost_ratio=float(data.get("maximum_cost_ratio", 1.5)),
        )


@dataclass(frozen=True)
class VerifierContract:
    checks: tuple[str, ...]
    required_evidence: tuple[str, ...] = ()
    anchor_id: str = ""
    repair_on_failure: bool = True
    max_repair_depth: int = 2

    def validate(self) -> list[str]:
        issues: list[str] = []
        if not self.checks:
            issues.append("verifier_checks_missing")
        if not self.anchor_id.strip():
            issues.append("verifier_anchor_missing")
        if self.max_repair_depth < 0:
            issues.append("verifier_repair_depth_invalid")
        return issues

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "VerifierContract":
        data = data or {}
        return cls(
            checks=tuple(str(value) for value in data.get("checks", [])),
            required_evidence=tuple(str(value) for value in data.get("required_evidence", [])),
            anchor_id=str(data.get("anchor_id") or ""),
            repair_on_failure=bool(data.get("repair_on_failure", True)),
            max_repair_depth=int(data.get("max_repair_depth", 2)),
        )


@dataclass(frozen=True)
class HypothesisProgram:
    id: str
    kind: HypothesisKind
    statement: str
    trigger: TriggerSpec
    anti_trigger: TriggerSpec
    action_graph: tuple[ActionNode, ...]
    expected_effect: ExpectedEffect
    verifier: VerifierContract
    evaluator_epoch: str
    fallback: str = "preserve_baseline"
    parent_id: str | None = None
    lineage: tuple[str, ...] = ()
    created_from_transition_ids: tuple[str, ...] = ()
    status: HypothesisStatus = HypothesisStatus.CANDIDATE

    @property
    def payload_hash(self) -> str:
        return stable_hash(self.to_dict())

    def validate(self) -> list[str]:
        issues: list[str] = []
        if not self.id.strip():
            issues.append("hypothesis_id_missing")
        if not self.statement.strip():
            issues.append("hypothesis_statement_missing")
        if not self.evaluator_epoch.strip():
            issues.append("evaluator_epoch_missing")
        if self.fallback != "preserve_baseline":
            issues.append("unsafe_fallback")
        issues.extend(self.trigger.validate(require_positive=True))
        issues.extend(self.anti_trigger.validate())
        issues.extend(self.expected_effect.validate())
        issues.extend(self.verifier.validate())
        if not self.action_graph:
            issues.append("action_graph_missing")
        action_ids = [action.id for action in self.action_graph]
        if len(action_ids) != len(set(action_ids)):
            issues.append("duplicate_action_id")
        known = set(action_ids)
        for action in self.action_graph:
            issues.extend(action.validate())
            if any(dependency not in known for dependency in action.depends_on):
                issues.append("unknown_action_dependency")
        if _has_action_cycle(self.action_graph):
            issues.append("action_graph_cycle")
        if self.parent_id and (not self.lineage or self.lineage[-1] != self.parent_id):
            issues.append("lineage_parent_mismatch")
        return sorted(set(issues))

    def matches(self, features: Mapping[str, Any]) -> bool:
        anti_match = not self.anti_trigger.is_empty and self.anti_trigger.matches(features)
        return self.trigger.matches(features) and not anti_match

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HypothesisProgram":
        return cls(
            id=str(data.get("id") or ""),
            kind=HypothesisKind(str(data.get("kind") or "policy")),
            statement=str(data.get("statement") or ""),
            trigger=TriggerSpec.from_dict(data.get("trigger") if isinstance(data.get("trigger"), Mapping) else {}),
            anti_trigger=TriggerSpec.from_dict(
                data.get("anti_trigger") if isinstance(data.get("anti_trigger"), Mapping) else {}
            ),
            action_graph=tuple(
                ActionNode.from_dict(row) for row in data.get("action_graph", []) if isinstance(row, Mapping)
            ),
            expected_effect=ExpectedEffect.from_dict(
                data.get("expected_effect") if isinstance(data.get("expected_effect"), Mapping) else {}
            ),
            verifier=VerifierContract.from_dict(
                data.get("verifier") if isinstance(data.get("verifier"), Mapping) else {}
            ),
            evaluator_epoch=str(data.get("evaluator_epoch") or ""),
            fallback=str(data.get("fallback") or "preserve_baseline"),
            parent_id=str(data["parent_id"]) if data.get("parent_id") else None,
            lineage=tuple(str(value) for value in data.get("lineage", [])),
            created_from_transition_ids=tuple(str(value) for value in data.get("created_from_transition_ids", [])),
            status=HypothesisStatus(str(data.get("status") or "candidate")),
        )


@dataclass(frozen=True)
class TaskInput:
    id: str
    family: str
    features: Mapping[str, Any]
    payload: Any = None


@dataclass(frozen=True)
class LaneResult:
    lane: str
    answer: Any
    confidence: float
    cost: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeExecution:
    task_id: str
    selected_result: LaneResult
    lane_results: tuple[LaneResult, ...]
    activated_hypothesis_ids: tuple[str, ...]
    plan_hash: str
    action_activated: bool
    baseline_preserved: bool

    @property
    def total_cost(self) -> float:
        return sum(result.cost for result in self.lane_results)


@dataclass(frozen=True)
class ExternalOutcome:
    task_id: str
    success: bool
    score: float
    evaluator_id: str
    evaluator_epoch: str
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ResidualExample:
    transition_id: str
    task_id: str
    family: str
    split: SplitName
    features: Mapping[str, Any]
    failure_type: str
    evaluator_feedback: tuple[str, ...]
    baseline_success: bool
    context: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        issues: list[str] = []
        if self.split is not SplitName.TRAIN:
            issues.append("proposal_residual_not_training")
        if _contains_forbidden_answer_key(self.features):
            issues.append("gold_field_in_proposal_features")
        if _contains_forbidden_answer_key(self.context):
            issues.append("gold_field_in_proposal_context")
        return issues


@dataclass(frozen=True)
class CounterfactualPair:
    task_id: str
    split: SplitName
    evaluator_epoch: str
    baseline: RuntimeExecution
    candidate: RuntimeExecution
    baseline_outcome: ExternalOutcome
    candidate_outcome: ExternalOutcome


def _has_action_cycle(actions: tuple[ActionNode, ...]) -> bool:
    dependencies = {action.id: set(action.depends_on) for action in actions}
    visited: set[str] = set()
    active: set[str] = set()

    def visit(action_id: str) -> bool:
        if action_id in active:
            return True
        if action_id in visited:
            return False
        active.add(action_id)
        for dependency in dependencies.get(action_id, set()):
            if visit(dependency):
                return True
        active.remove(action_id)
        visited.add(action_id)
        return False

    return any(visit(action_id) for action_id in dependencies)


def _contains_forbidden_answer_key(value: Any) -> bool:
    forbidden = {"gold", "gold_label", "correct_answer", "_answer"}
    if isinstance(value, Mapping):
        return bool(forbidden & set(value)) or any(
            _contains_forbidden_answer_key(child) for child in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_answer_key(child) for child in value)
    return False
