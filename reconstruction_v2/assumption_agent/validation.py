from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Protocol, Sequence

from .events import Event, EventSink, NullEventSink
from .models import (
    HypothesisKind,
    HypothesisProgram,
    ResidualExample,
    SplitName,
    stable_hash,
)
from .proposer import HypothesisProposalCallError, StructuredHypothesisProposer


ALL_ACTION_OPERATIONS = frozenset(
    {
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
    }
)


@dataclass(frozen=True)
class ValidationContext:
    evaluator_epoch: str
    residuals: tuple[ResidualExample, ...]
    available_lanes: frozenset[str]
    baseline_lane: str
    trigger_feature_catalog: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    allowed_runtime_kinds: frozenset[HypothesisKind] = field(
        default_factory=lambda: frozenset(HypothesisKind)
    )
    allowed_action_operations: frozenset[str] = field(
        default_factory=lambda: ALL_ACTION_OPERATIONS
    )
    action_semantics: str = "typed_runtime_action_v1"
    external_evidence_is_hidden: bool = False


def backend_action_contract_issues(
    program: HypothesisProgram,
    *,
    allowed_operations: frozenset[str],
    external_evidence_is_hidden: bool,
) -> tuple[str, ...]:
    """Return structural backend-lowering issues for one action graph.

    SkillLearn's external verifier and paired policy-off/on outcomes are not
    available while the task agent is running.  This check deliberately uses
    structured action fields rather than trusting a trailing prose warning in
    the compiled skill.
    """

    issues: list[str] = []
    external_anchor = _canonical_action_reference(program.verifier.anchor_id)
    external_evidence_references = tuple(
        reference
        for value in program.verifier.required_evidence
        if (reference := _canonical_action_reference(value))
    )
    for action in program.action_graph:
        if action.operation not in allowed_operations:
            issues.append(f"unsupported_action_operation:{action.operation}")
        if not external_evidence_is_hidden:
            continue
        references = (
            action.target,
            *_nested_action_reference_strings(action.value),
        )
        if any(
            _is_hidden_external_reference(
                value,
                external_anchor=external_anchor,
                external_evidence_references=external_evidence_references,
            )
            for value in references
        ):
            issues.append(f"hidden_external_evidence_reference:{action.id}")
    return tuple(sorted(set(issues)))


def _nested_action_reference_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(
            child
            for key, item in value.items()
            for child in (
                *_nested_action_reference_strings(key),
                *_nested_action_reference_strings(item),
            )
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(
            child
            for item in value
            for child in _nested_action_reference_strings(item)
        )
    return ()


def _canonical_action_reference(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _is_hidden_external_reference(
    value: str,
    *,
    external_anchor: str,
    external_evidence_references: tuple[str, ...],
) -> bool:
    normalized = _canonical_action_reference(value)
    if not normalized:
        return False
    if external_anchor and external_anchor in normalized:
        return True
    if any(reference in normalized for reference in external_evidence_references):
        return True
    compact = normalized.replace("_", "")
    return any(
        marker in normalized or marker.replace("_", "") in compact
        for marker in (
            "policy_off",
            "policy_on",
            "benchmark_verifier",
            "hidden_verifier",
            "external_task_verifier",
        )
    )


@dataclass(frozen=True)
class CheckResult:
    check: str
    passed: bool
    reason: str
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "passed": self.passed,
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


class HypothesisCheck(Protocol):
    name: str

    def evaluate(self, program: HypothesisProgram, context: ValidationContext) -> CheckResult: ...


class SchemaCheck:
    name = "schema"

    def evaluate(self, program: HypothesisProgram, context: ValidationContext) -> CheckResult:
        issues = program.validate()
        return CheckResult(
            check=self.name,
            passed=not issues,
            reason="valid_hypothesis_program" if not issues else "hypothesis_contract_failed",
            evidence={"issues": issues, "hypothesis_hash": program.payload_hash},
        )


class TriggerVocabularyCheck:
    name = "trigger_vocabulary"

    def evaluate(self, program: HypothesisProgram, context: ValidationContext) -> CheckResult:
        allowed = frozenset(context.trigger_feature_catalog)
        predicates = (
            *(('trigger', row) for row in program.trigger.all_of),
            *(('trigger', row) for row in program.trigger.any_of),
            *(('trigger', row) for row in program.trigger.none_of),
            *(('anti_trigger', row) for row in program.anti_trigger.all_of),
            *(('anti_trigger', row) for row in program.anti_trigger.any_of),
            *(('anti_trigger', row) for row in program.anti_trigger.none_of),
        )
        unknown = sorted(
            {
                (scope, predicate.key, predicate.op)
                for scope, predicate in predicates
                if predicate.key not in allowed
            }
        )
        invalid_operators = sorted(
            {
                (scope, predicate.key, predicate.op)
                for scope, predicate in predicates
                if predicate.key in context.trigger_feature_catalog
                and predicate.op
                not in context.trigger_feature_catalog[predicate.key].get(
                    "allowed_operators", ()
                )
            }
        )
        passed = not unknown and not invalid_operators
        return CheckResult(
            check=self.name,
            passed=passed,
            reason=(
                "trigger_uses_runtime_feature_vocabulary"
                if passed
                else "trigger_uses_non_runtime_features"
            ),
            evidence={
                "allowed_feature_keys": sorted(allowed),
                "feature_catalog_present": bool(allowed),
                "allowed_feature_catalog_hash": stable_hash(
                    context.trigger_feature_catalog
                ),
                "unknown_predicates": [
                    {"scope": scope, "key": key, "op": op}
                    for scope, key, op in unknown
                ],
                "invalid_operator_predicates": [
                    {"scope": scope, "key": key, "op": op}
                    for scope, key, op in invalid_operators
                ],
            },
        )


class RuntimeCandidateKindCheck:
    name = "runtime_candidate_kind"

    def evaluate(self, program: HypothesisProgram, context: ValidationContext) -> CheckResult:
        passed = program.kind in context.allowed_runtime_kinds
        return CheckResult(
            check=self.name,
            passed=passed,
            reason=(
                "hypothesis_kind_can_control_runtime"
                if passed
                else "evaluator_hypothesis_requires_epoch_challenger"
            ),
            evidence={
                "program_kind": program.kind.value,
                "allowed_runtime_kinds": sorted(
                    kind.value for kind in context.allowed_runtime_kinds
                ),
            },
        )


class TrainingSupportCheck:
    name = "training_support"

    def __init__(self, min_support: int = 2) -> None:
        self.min_support = min_support

    def evaluate(self, program: HypothesisProgram, context: ValidationContext) -> CheckResult:
        train_rows = [row for row in context.residuals if row.split is SplitName.TRAIN]
        matching = [row for row in train_rows if program.matches(row.features)]
        anti_matching = [
            row
            for row in train_rows
            if not program.anti_trigger.is_empty and program.anti_trigger.matches(row.features)
        ]
        passed = len(matching) >= self.min_support and len(matching) > len(anti_matching)
        return CheckResult(
            check=self.name,
            passed=passed,
            reason="sufficient_training_trigger_support" if passed else "insufficient_or_antiscope_support",
            evidence={
                "training_residual_count": len(train_rows),
                "trigger_support_count": len(matching),
                "anti_trigger_support_count": len(anti_matching),
                "minimum_support": self.min_support,
            },
        )


class RuntimeActionCheck:
    name = "runtime_action"

    def evaluate(self, program: HypothesisProgram, context: ValidationContext) -> CheckResult:
        contract_issues = backend_action_contract_issues(
            program,
            allowed_operations=context.allowed_action_operations,
            external_evidence_is_hidden=context.external_evidence_is_hidden,
        )
        unsupported_operations = sorted(
            issue.split(":", 1)[1]
            for issue in contract_issues
            if issue.startswith("unsupported_action_operation:")
        )
        hidden_external_reference_action_ids = sorted(
            issue.split(":", 1)[1]
            for issue in contract_issues
            if issue.startswith("hidden_external_evidence_reference:")
        )
        lane_actions = [
            action
            for action in program.action_graph
            if action.operation in {"enable_lane", "disable_lane", "prioritize_lane"}
        ]
        unknown_lanes = sorted(
            {
                action.target
                for action in lane_actions
                if action.target not in context.available_lanes
            }
        )
        disables_baseline = any(
            action.operation == "disable_lane" and action.target == context.baseline_lane
            for action in lane_actions
        )
        lowerable_actions = [
            action
            for action in program.action_graph
            if action.operation in context.allowed_action_operations
        ]
        passed = (
            bool(lowerable_actions)
            and not contract_issues
            and not unknown_lanes
            and not disables_baseline
        )
        return CheckResult(
            check=self.name,
            passed=passed,
            reason=(
                "action_graph_lowerable_for_backend"
                if passed
                else "action_graph_not_lowerable_for_backend"
            ),
            evidence={
                "lowerable_action_count": len(lowerable_actions),
                "unsupported_operations": unsupported_operations,
                "hidden_external_reference_action_ids": (
                    hidden_external_reference_action_ids
                ),
                "action_contract_issues": list(contract_issues),
                "allowed_action_operations": sorted(
                    context.allowed_action_operations
                ),
                "action_semantics": context.action_semantics,
                "external_evidence_is_hidden": (
                    context.external_evidence_is_hidden
                ),
                "unknown_lanes": unknown_lanes,
                "disables_baseline": disables_baseline,
                "baseline_lane": context.baseline_lane,
            },
        )


class EvaluatorEpochCheck:
    name = "evaluator_epoch"

    def evaluate(self, program: HypothesisProgram, context: ValidationContext) -> CheckResult:
        passed = bool(program.evaluator_epoch) and program.evaluator_epoch == context.evaluator_epoch
        return CheckResult(
            check=self.name,
            passed=passed,
            reason="frozen_epoch_matches" if passed else "evaluator_epoch_mismatch",
            evidence={
                "program_epoch": program.evaluator_epoch,
                "active_epoch": context.evaluator_epoch,
            },
        )


def build_trigger_feature_catalog(
    residuals: Sequence[ResidualExample],
    *,
    maximum_values_per_feature: int = 24,
) -> dict[str, dict[str, Any]]:
    return build_runtime_feature_catalog(
        [residual.features for residual in residuals],
        maximum_values_per_feature=maximum_values_per_feature,
    )


def build_runtime_feature_catalog(
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    maximum_values_per_feature: int = 24,
) -> dict[str, dict[str, Any]]:
    if maximum_values_per_feature <= 0:
        raise ValueError("trigger feature catalog value limit must be positive")
    by_key: dict[str, list[Any]] = {}
    for features in feature_rows:
        for key, value in features.items():
            by_key.setdefault(str(key), []).append(value)
    catalog: dict[str, dict[str, Any]] = {}
    for key, values in sorted(by_key.items()):
        scalar_values: dict[str, Any] = {}
        member_values: dict[str, Any] = {}
        observed_types: set[str] = set()
        collection = False
        for value in values:
            observed_types.add(type(value).__name__)
            if isinstance(value, (list, tuple, set, frozenset)):
                collection = True
                for member in value:
                    member_values.setdefault(stable_hash(member), member)
            else:
                scalar_values.setdefault(stable_hash(value), value)
        if collection:
            observed = [
                member_values[value_hash]
                for value_hash in sorted(member_values)[:maximum_values_per_feature]
            ]
            operators = ["contains", "exists"]
            value_field = "observed_members"
        else:
            observed = [
                scalar_values[value_hash]
                for value_hash in sorted(scalar_values)[:maximum_values_per_feature]
            ]
            numeric = all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in values
            )
            operators = ["eq", "ne", "in", "exists"]
            if numeric:
                operators.extend(["gte", "lte"])
            value_field = "observed_values"
        catalog[key] = {
            "observed_types": sorted(observed_types),
            value_field: observed,
            "allowed_operators": operators,
            "training_row_count": len(values),
        }
    return catalog


@dataclass(frozen=True)
class ValidationNode:
    program: HypothesisProgram
    depth: int
    checks: tuple[CheckResult, ...]
    child_id: str | None
    terminal_reason: str

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.checks)


@dataclass(frozen=True)
class ValidationTree:
    root_id: str
    nodes: tuple[ValidationNode, ...]
    accepted_program: HypothesisProgram | None

    @property
    def recursion_depth(self) -> int:
        return max((node.depth for node in self.nodes), default=0)

    @property
    def repair_model_failure_count(self) -> int:
        return sum(
            node.terminal_reason == "repair_model_failed" for node in self.nodes
        )


class RecursiveValidationEngine:
    def __init__(
        self,
        checks: Sequence[HypothesisCheck],
        *,
        proposer: StructuredHypothesisProposer | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        self.checks = tuple(checks)
        self.proposer = proposer
        self.event_sink = event_sink or NullEventSink()

    def validate(
        self,
        program: HypothesisProgram,
        context: ValidationContext,
        *,
        trace_id: str | None = None,
    ) -> ValidationTree:
        trace_id = trace_id or stable_hash({"hypothesis_id": program.id, "epoch": context.evaluator_epoch})[:20]
        nodes: list[ValidationNode] = []
        accepted = self._validate_recursive(program, context, depth=0, nodes=nodes, trace_id=trace_id)
        tree = ValidationTree(root_id=program.id, nodes=tuple(nodes), accepted_program=accepted)
        self.event_sink.emit(
            Event(
                event="recursive_validation_completed",
                stage="validation",
                trace_id=trace_id,
                payload={
                    "root_id": program.id,
                    "node_count": len(nodes),
                    "recursion_depth": tree.recursion_depth,
                    "accepted_hypothesis_id": accepted.id if accepted else None,
                    "accepted": accepted is not None,
                    "repair_model_failure_count": tree.repair_model_failure_count,
                    "tree_hash": stable_hash(
                        [
                            {
                                "hypothesis_id": node.program.id,
                                "depth": node.depth,
                                "passed": node.passed,
                                "child_id": node.child_id,
                                "terminal_reason": node.terminal_reason,
                            }
                            for node in nodes
                        ]
                    ),
                },
            )
        )
        return tree

    def _validate_recursive(
        self,
        program: HypothesisProgram,
        context: ValidationContext,
        *,
        depth: int,
        nodes: list[ValidationNode],
        trace_id: str,
    ) -> HypothesisProgram | None:
        results = tuple(check.evaluate(program, context) for check in self.checks)
        failed = tuple(result for result in results if not result.passed)
        self.event_sink.emit(
            Event(
                event="hypothesis_validation_node_evaluated",
                stage="validation.node",
                trace_id=trace_id,
                payload={
                    "hypothesis_id": program.id,
                    "hypothesis_hash": program.payload_hash,
                    "depth": depth,
                    "check_results": [result.to_dict() for result in results],
                    "passed": not failed,
                },
            )
        )
        if not failed:
            nodes.append(
                ValidationNode(
                    program=program,
                    depth=depth,
                    checks=results,
                    child_id=None,
                    terminal_reason="accepted",
                )
            )
            return program
        max_depth = program.verifier.max_repair_depth
        can_repair = (
            self.proposer is not None
            and program.verifier.repair_on_failure
            and depth < max_depth
        )
        if not can_repair:
            nodes.append(
                ValidationNode(
                    program=program,
                    depth=depth,
                    checks=results,
                    child_id=None,
                    terminal_reason="static_rejected",
                )
            )
            return None
        try:
            child = self.proposer.revise(
                program,
                failed_checks=[result.to_dict() for result in failed],
                residuals=context.residuals,
                depth=depth + 1,
                capabilities={
                    "available_lanes": sorted(context.available_lanes),
                    "baseline_lane": context.baseline_lane,
                    "runtime_trigger_contract": {
                        "allowed_feature_catalog": dict(
                            context.trigger_feature_catalog
                        ),
                        "forbidden_context_only_keys": [
                            "task_instruction",
                            "observed_metrics",
                            "execution_signals",
                        ],
                        "context_is_for_action_design_only": True,
                    },
                    "runtime_candidate_kinds": sorted(
                        kind.value for kind in context.allowed_runtime_kinds
                    ),
                    "action_contract": {
                        "allowed_action_operations": sorted(
                            context.allowed_action_operations
                        ),
                        "semantics": context.action_semantics,
                        "external_evidence_is_hidden": (
                            context.external_evidence_is_hidden
                        ),
                    },
                    "evaluator_hypotheses_require_separate_epoch_challenger": True,
                },
                trace_id=trace_id,
            )
        except HypothesisProposalCallError as exc:
            self.event_sink.emit(
                Event(
                    event="hypothesis_repair_abandoned_after_model_failure",
                    stage="validation.repair",
                    trace_id=trace_id,
                    payload={
                        "parent_id": program.id,
                        "parent_hash": program.payload_hash,
                        "repair_depth": depth + 1,
                        "request_kind": exc.request_kind,
                        "request_hash": exc.request_hash,
                        "error_type": exc.error_type,
                        "candidate_local_failure": True,
                        "raw_error_persisted": False,
                    },
                )
            )
            nodes.append(
                ValidationNode(
                    program=program,
                    depth=depth,
                    checks=results,
                    child_id=None,
                    terminal_reason="repair_model_failed",
                )
            )
            return None
        nodes.append(
            ValidationNode(
                program=program,
                depth=depth,
                checks=results,
                child_id=child.id,
                terminal_reason="repair_proposed",
            )
        )
        return self._validate_recursive(child, context, depth=depth + 1, nodes=nodes, trace_id=trace_id)
