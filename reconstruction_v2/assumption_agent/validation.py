from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .events import Event, EventSink, NullEventSink
from .models import HypothesisProgram, ResidualExample, SplitName, stable_hash
from .proposer import StructuredHypothesisProposer


@dataclass(frozen=True)
class ValidationContext:
    evaluator_epoch: str
    residuals: tuple[ResidualExample, ...]
    available_lanes: frozenset[str]
    baseline_lane: str


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
        runtime_mutations = [
            action
            for action in program.action_graph
            if action.operation in {
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
        ]
        passed = bool(runtime_mutations) and not unknown_lanes and not disables_baseline
        return CheckResult(
            check=self.name,
            passed=passed,
            reason="runtime_action_is_executable" if passed else "runtime_action_not_executable",
            evidence={
                "runtime_mutation_count": len(runtime_mutations),
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


@dataclass(frozen=True)
class ValidationNode:
    program: HypothesisProgram
    depth: int
    checks: tuple[CheckResult, ...]
    child_id: str | None

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
                    "tree_hash": stable_hash(
                        [
                            {
                                "hypothesis_id": node.program.id,
                                "depth": node.depth,
                                "passed": node.passed,
                                "child_id": node.child_id,
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
            nodes.append(ValidationNode(program=program, depth=depth, checks=results, child_id=None))
            return program
        max_depth = program.verifier.max_repair_depth
        can_repair = (
            self.proposer is not None
            and program.verifier.repair_on_failure
            and depth < max_depth
        )
        if not can_repair:
            nodes.append(ValidationNode(program=program, depth=depth, checks=results, child_id=None))
            return None
        child = self.proposer.revise(
            program,
            failed_checks=[result.to_dict() for result in failed],
            residuals=context.residuals,
            depth=depth + 1,
            capabilities={
                "available_lanes": sorted(context.available_lanes),
                "baseline_lane": context.baseline_lane,
            },
            trace_id=trace_id,
        )
        nodes.append(ValidationNode(program=program, depth=depth, checks=results, child_id=child.id))
        return self._validate_recursive(child, context, depth=depth + 1, nodes=nodes, trace_id=trace_id)
