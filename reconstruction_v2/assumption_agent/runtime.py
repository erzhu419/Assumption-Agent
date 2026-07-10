from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .events import Event, EventSink, NullEventSink
from .models import (
    ActionNode,
    HypothesisProgram,
    HypothesisStatus,
    LaneResult,
    RuntimeExecution,
    TaskInput,
    stable_hash,
)


class Lane(Protocol):
    name: str

    def run(self, task: TaskInput, parameters: Mapping[str, Any]) -> LaneResult: ...


@dataclass(frozen=True)
class RuntimePlan:
    baseline_lane: str
    enabled_lanes: tuple[str, ...]
    disabled_lanes: tuple[str, ...]
    lane_priorities: Mapping[str, int]
    parameters: Mapping[str, Any]
    required_verifiers: tuple[str, ...]
    operator_actions: tuple[Mapping[str, Any], ...]
    activated_hypothesis_ids: tuple[str, ...]
    abstain_to_baseline: bool
    applied_action_ids: tuple[str, ...]

    @property
    def plan_hash(self) -> str:
        return stable_hash(
            {
                "baseline_lane": self.baseline_lane,
                "enabled_lanes": self.enabled_lanes,
                "disabled_lanes": self.disabled_lanes,
                "lane_priorities": dict(self.lane_priorities),
                "parameters": dict(self.parameters),
                "required_verifiers": self.required_verifiers,
                "operator_actions": self.operator_actions,
                "activated_hypothesis_ids": self.activated_hypothesis_ids,
                "abstain_to_baseline": self.abstain_to_baseline,
                "applied_action_ids": self.applied_action_ids,
            }
        )


class LaneRegistry:
    def __init__(self, lanes: Sequence[Lane]) -> None:
        self._lanes = {lane.name: lane for lane in lanes}
        if len(self._lanes) != len(lanes):
            raise ValueError("lane names must be unique")

    def require(self, name: str) -> Lane:
        try:
            return self._lanes[name]
        except KeyError as exc:
            raise KeyError(f"unknown lane: {name}") from exc

    @property
    def names(self) -> set[str]:
        return set(self._lanes)


class PolicyRuntime:
    """Compile promoted hypotheses into a plan that materially changes execution."""

    def __init__(
        self,
        *,
        registry: LaneRegistry,
        baseline_lane: str,
        event_sink: EventSink | None = None,
        runtime_version: str = "policy_runtime_v1",
    ) -> None:
        registry.require(baseline_lane)
        self.registry = registry
        self.baseline_lane = baseline_lane
        self.event_sink = event_sink or NullEventSink()
        self.runtime_version = runtime_version

    def build_plan(
        self,
        task: TaskInput,
        programs: Sequence[HypothesisProgram] = (),
        *,
        allowed_statuses: set[HypothesisStatus] | None = None,
        trace_id: str | None = None,
    ) -> RuntimePlan:
        trace_id = trace_id or stable_hash({"task_id": task.id, "runtime": self.runtime_version})[:20]
        allowed = allowed_statuses or {HypothesisStatus.PROMOTED}
        enabled = {self.baseline_lane}
        disabled: set[str] = set()
        priorities = {self.baseline_lane: 0}
        parameters: dict[str, Any] = {"selection.minimum_confidence": 0.5}
        required_verifiers: set[str] = set()
        operator_actions: list[dict[str, Any]] = []
        activated_programs: list[str] = []
        applied_actions: list[str] = []
        abstain = False

        for program in sorted(programs, key=lambda row: row.id):
            if program.status not in allowed:
                self._emit_program_decision(trace_id, task, program, "status_blocked")
                continue
            if program.validate():
                self._emit_program_decision(trace_id, task, program, "validation_blocked")
                continue
            if not program.matches(task.features):
                self._emit_program_decision(trace_id, task, program, "trigger_miss")
                continue
            activated_programs.append(program.id)
            self._emit_program_decision(trace_id, task, program, "activated")
            for action in _topological_actions(program.action_graph):
                self._apply_action(
                    action,
                    enabled=enabled,
                    disabled=disabled,
                    priorities=priorities,
                    parameters=parameters,
                    required_verifiers=required_verifiers,
                    operator_actions=operator_actions,
                )
                if action.operation == "abstain":
                    abstain = True
                applied_actions.append(action.id)

        enabled.add(self.baseline_lane)
        disabled.discard(self.baseline_lane)
        if abstain:
            enabled = {self.baseline_lane}
        for lane in enabled:
            self.registry.require(lane)
        if operator_actions:
            parameters["hypothesis.operator_actions"] = tuple(operator_actions)
        plan = RuntimePlan(
            baseline_lane=self.baseline_lane,
            enabled_lanes=tuple(sorted(enabled, key=lambda name: (-priorities.get(name, 0), name))),
            disabled_lanes=tuple(sorted(disabled)),
            lane_priorities=dict(sorted(priorities.items())),
            parameters=dict(sorted(parameters.items())),
            required_verifiers=tuple(sorted(required_verifiers)),
            operator_actions=tuple(operator_actions),
            activated_hypothesis_ids=tuple(activated_programs),
            abstain_to_baseline=abstain,
            applied_action_ids=tuple(applied_actions),
        )
        self.event_sink.emit(
            Event(
                event="runtime_plan_built",
                stage="runtime.plan",
                trace_id=trace_id,
                payload={
                    "task_id_hash": stable_hash({"task_id": task.id}),
                    "family_hash": stable_hash({"family": task.family}),
                    "runtime_version": self.runtime_version,
                    "enabled_lanes": list(plan.enabled_lanes),
                    "disabled_lanes": list(plan.disabled_lanes),
                    "activated_hypothesis_ids": list(plan.activated_hypothesis_ids),
                    "applied_action_ids": list(plan.applied_action_ids),
                    "required_verifiers": list(plan.required_verifiers),
                    "operator_action_count": len(plan.operator_actions),
                    "plan_hash": plan.plan_hash,
                    "baseline_preserved": self.baseline_lane in plan.enabled_lanes,
                },
            )
        )
        return plan

    def execute(
        self,
        task: TaskInput,
        programs: Sequence[HypothesisProgram] = (),
        *,
        allowed_statuses: set[HypothesisStatus] | None = None,
        trace_id: str | None = None,
    ) -> RuntimeExecution:
        trace_id = trace_id or stable_hash({"task_id": task.id, "programs": [row.id for row in programs]})[:20]
        plan = self.build_plan(
            task,
            programs,
            allowed_statuses=allowed_statuses,
            trace_id=trace_id,
        )
        results = tuple(
            self.registry.require(lane_name).run(task, plan.parameters)
            for lane_name in plan.enabled_lanes
        )
        baseline = next(result for result in results if result.lane == self.baseline_lane)
        selected = self._select(results, plan, baseline)
        execution = RuntimeExecution(
            task_id=task.id,
            selected_result=selected,
            lane_results=results,
            activated_hypothesis_ids=plan.activated_hypothesis_ids,
            plan_hash=plan.plan_hash,
            action_activated=bool(plan.applied_action_ids),
            baseline_preserved=any(result.lane == self.baseline_lane for result in results),
        )
        self.event_sink.emit(
            Event(
                event="runtime_execution_completed",
                stage="runtime.execute",
                trace_id=trace_id,
                payload={
                    "task_id_hash": stable_hash({"task_id": task.id}),
                    "plan_hash": plan.plan_hash,
                    "lane_count": len(results),
                    "lane_names": [result.lane for result in results],
                    "selected_lane": selected.lane,
                    "selected_answer_hash": stable_hash({"answer": selected.answer}),
                    "selected_confidence": selected.confidence,
                    "total_cost": execution.total_cost,
                    "action_activated": execution.action_activated,
                    "baseline_preserved": execution.baseline_preserved,
                },
            )
        )
        return execution

    def _apply_action(
        self,
        action: ActionNode,
        *,
        enabled: set[str],
        disabled: set[str],
        priorities: dict[str, int],
        parameters: dict[str, Any],
        required_verifiers: set[str],
        operator_actions: list[dict[str, Any]],
    ) -> None:
        if action.operation == "enable_lane":
            self.registry.require(action.target)
            enabled.add(action.target)
            disabled.discard(action.target)
        elif action.operation == "disable_lane":
            if action.target != self.baseline_lane:
                enabled.discard(action.target)
                disabled.add(action.target)
        elif action.operation == "prioritize_lane":
            self.registry.require(action.target)
            priorities[action.target] = int(action.value)
        elif action.operation == "set_parameter":
            parameters[action.target] = action.value
        elif action.operation == "require_verifier":
            required_verifiers.add(action.target)
        elif action.operation == "abstain":
            return
        elif action.operation in {
            "execute_step",
            "check_condition",
            "produce_artifact",
            "request_evidence",
        }:
            operator_actions.append(
                {
                    "id": action.id,
                    "operation": action.operation,
                    "target": action.target,
                    "value": action.value,
                    "depends_on": list(action.depends_on),
                }
            )
            if action.operation == "check_condition":
                required_verifiers.add(action.target)

    @staticmethod
    def _select(results: tuple[LaneResult, ...], plan: RuntimePlan, baseline: LaneResult) -> LaneResult:
        minimum_confidence = float(plan.parameters.get("selection.minimum_confidence", 0.5))
        candidates = [result for result in results if result.confidence >= minimum_confidence]
        if not candidates:
            return baseline
        candidates.sort(
            key=lambda result: (
                plan.lane_priorities.get(result.lane, 0),
                result.confidence,
                result.lane != plan.baseline_lane,
                result.lane,
            ),
            reverse=True,
        )
        return candidates[0]

    def _emit_program_decision(
        self,
        trace_id: str,
        task: TaskInput,
        program: HypothesisProgram,
        decision: str,
    ) -> None:
        self.event_sink.emit(
            Event(
                event="hypothesis_runtime_decision",
                stage="runtime.policy",
                trace_id=trace_id,
                payload={
                    "task_id_hash": stable_hash({"task_id": task.id}),
                    "hypothesis_id": program.id,
                    "hypothesis_hash": program.payload_hash,
                    "hypothesis_kind": program.kind.value,
                    "decision": decision,
                    "evaluator_epoch": program.evaluator_epoch,
                },
            )
        )


def _topological_actions(actions: tuple[ActionNode, ...]) -> tuple[ActionNode, ...]:
    by_id = {action.id: action for action in actions}
    pending = {action.id: set(action.depends_on) for action in actions}
    ordered: list[ActionNode] = []
    while pending:
        ready = sorted(action_id for action_id, dependencies in pending.items() if not dependencies)
        if not ready:
            raise ValueError("action graph contains a cycle")
        for action_id in ready:
            ordered.append(by_id[action_id])
            pending.pop(action_id)
            for dependencies in pending.values():
                dependencies.discard(action_id)
    return tuple(ordered)
