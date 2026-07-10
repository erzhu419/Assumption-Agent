from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Protocol, Sequence

from .events import Event, EventSink, NullEventSink
from .models import HypothesisProgram, ResidualExample, SplitName, stable_hash


class ProposalModel(Protocol):
    """Model adapter that returns parsed JSON, never unstructured prose."""

    def complete(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


class StructuredHypothesisProposer:
    def __init__(self, model: ProposalModel, *, event_sink: EventSink | None = None) -> None:
        self.model = model
        self.event_sink = event_sink or NullEventSink()

    def propose(
        self,
        residuals: Sequence[ResidualExample],
        *,
        evaluator_epoch: str,
        max_hypotheses: int = 3,
        capabilities: Mapping[str, Any] | None = None,
        trace_id: str = "proposal",
    ) -> tuple[HypothesisProgram, ...]:
        if not residuals:
            raise ValueError("at least one training residual is required")
        issues = [issue for residual in residuals for issue in residual.validate()]
        if issues:
            raise PermissionError(f"proposal data isolation failed: {sorted(set(issues))}")
        payload = {
            "request_kind": "propose_hypothesis_programs",
            "contract_version": "hypothesis_program_v1",
            "evaluator_epoch": evaluator_epoch,
            "constraints": _proposal_constraints(),
            "output_schema": {"hypotheses": [_program_schema()]},
            "capabilities": dict(capabilities or {}),
            "residuals": [_residual_payload(residual) for residual in residuals],
            "max_hypotheses": max_hypotheses,
        }
        self._emit_model_event("hypothesis_proposal_requested", trace_id, payload)
        response = self._complete(payload, trace_id=trace_id)
        rows = response.get("hypotheses", [])
        if not isinstance(rows, list):
            raise ValueError("proposal model must return a hypotheses list")
        programs: list[HypothesisProgram] = []
        transition_ids = tuple(sorted(residual.transition_id for residual in residuals))
        for index, row in enumerate(rows[:max_hypotheses]):
            if not isinstance(row, Mapping):
                continue
            normalized = dict(row)
            normalized["evaluator_epoch"] = evaluator_epoch
            normalized["created_from_transition_ids"] = list(transition_ids)
            normalized.setdefault(
                "id",
                f"hyp_{stable_hash({'response': row, 'evaluator_epoch': evaluator_epoch, 'index': index})[:16]}",
            )
            program = HypothesisProgram.from_dict(normalized)
            programs.append(program)
            self.event_sink.emit(
                Event(
                    event="hypothesis_proposed",
                    stage="proposal",
                    trace_id=trace_id,
                    payload={
                        "hypothesis_id": program.id,
                        "hypothesis_hash": program.payload_hash,
                        "kind": program.kind.value,
                        "transition_count": len(transition_ids),
                        "validation_issues": program.validate(),
                        "evaluator_epoch": evaluator_epoch,
                    },
                )
            )
        if not programs:
            raise ValueError("proposal model returned no hypothesis programs")
        return tuple(programs)

    def revise(
        self,
        parent: HypothesisProgram,
        *,
        failed_checks: Sequence[Mapping[str, Any]],
        residuals: Sequence[ResidualExample],
        depth: int,
        capabilities: Mapping[str, Any] | None = None,
        trace_id: str,
    ) -> HypothesisProgram:
        if any(residual.split is not SplitName.TRAIN for residual in residuals):
            raise PermissionError("recursive repair may use training residuals only")
        payload = {
            "request_kind": "repair_hypothesis_program",
            "contract_version": "hypothesis_program_v1",
            "evaluator_epoch": parent.evaluator_epoch,
            "constraints": _proposal_constraints(),
            "output_schema": {"hypothesis": _program_schema()},
            "capabilities": dict(capabilities or {}),
            "parent": parent.to_dict(),
            "failed_checks": [dict(row) for row in failed_checks],
            "residuals": [_residual_payload(residual) for residual in residuals],
            "repair_depth": depth,
        }
        self._emit_model_event("hypothesis_repair_requested", trace_id, payload)
        response = self._complete(payload, trace_id=trace_id)
        row = response.get("hypothesis")
        if not isinstance(row, Mapping):
            raise ValueError("repair model must return one hypothesis object")
        normalized = dict(row)
        normalized["evaluator_epoch"] = parent.evaluator_epoch
        normalized["parent_id"] = parent.id
        normalized["lineage"] = [*parent.lineage, parent.id]
        normalized["created_from_transition_ids"] = list(parent.created_from_transition_ids)
        normalized.setdefault(
            "id",
            f"hyp_{stable_hash({'parent': parent.id, 'response': row, 'depth': depth})[:16]}",
        )
        child = HypothesisProgram.from_dict(normalized)
        child = replace(child, parent_id=parent.id, lineage=(*parent.lineage, parent.id))
        self.event_sink.emit(
            Event(
                event="hypothesis_repair_proposed",
                stage="proposal.repair",
                trace_id=trace_id,
                payload={
                    "parent_id": parent.id,
                    "child_id": child.id,
                    "child_hash": child.payload_hash,
                    "repair_depth": depth,
                    "failed_check_count": len(failed_checks),
                    "validation_issues": child.validate(),
                },
            )
        )
        return child

    def _emit_model_event(self, event: str, trace_id: str, payload: Mapping[str, Any]) -> None:
        self.event_sink.emit(
            Event(
                event=event,
                stage="proposal.model",
                trace_id=trace_id,
                payload={
                    "request_kind": payload.get("request_kind"),
                    "request_hash": stable_hash(payload),
                    "residual_count": len(payload.get("residuals", [])),
                    "evaluator_epoch": payload.get("evaluator_epoch"),
                    "raw_content_persisted": False,
                },
            )
        )

    def _complete(self, payload: Mapping[str, Any], *, trace_id: str) -> Mapping[str, Any]:
        traced = getattr(self.model, "complete_with_trace", None)
        if callable(traced):
            return traced(payload, trace_id=trace_id)
        return self.model.complete(payload)


def _residual_payload(residual: ResidualExample) -> dict[str, Any]:
    return {
        "transition_id": residual.transition_id,
        "task_id_hash": stable_hash({"task_id": residual.task_id}),
        "family": residual.family,
        "features": dict(residual.features),
        "failure_type": residual.failure_type,
        "evaluator_feedback": list(residual.evaluator_feedback),
        "baseline_success": residual.baseline_success,
        "context": dict(residual.context),
    }


def _proposal_constraints() -> dict[str, Any]:
    return {
        "allowed_kinds": ["task", "policy", "evaluator"],
        "fallback_must_equal": "preserve_baseline",
        "trigger_must_use_structured_features": True,
        "action_graph_must_change_runtime": True,
        "gold_answer_fields_forbidden": True,
        "required_verifier_anchor": True,
        "required_expected_effect": ["metric", "minimum_delta", "maximum_harm_rate", "maximum_cost_ratio"],
        "allowed_action_operations": [
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
        ],
    }


def _program_schema() -> dict[str, Any]:
    predicate = {"key": "feature name", "op": "eq|ne|in|contains|exists|gte|lte", "value": "JSON value"}
    return {
        "id": "stable descriptive ID",
        "kind": "task|policy|evaluator",
        "statement": "falsifiable hypothesis",
        "trigger": {"all_of": [predicate], "any_of": [], "none_of": []},
        "anti_trigger": {"all_of": [], "any_of": [predicate], "none_of": []},
        "action_graph": [
            {
                "id": "action ID",
                "operation": "one allowed action operation",
                "target": "declared capability, step, verifier, or artifact",
                "value": "JSON value",
                "depends_on": [],
            }
        ],
        "expected_effect": {
            "metric": "task_success",
            "minimum_delta": 0.0,
            "maximum_harm_rate": 0.05,
            "maximum_cost_ratio": 1.5,
        },
        "verifier": {
            "checks": ["named check"],
            "required_evidence": ["paired policy-off/policy-on outcome"],
            "anchor_id": "external anchor ID",
            "repair_on_failure": True,
            "max_repair_depth": 2,
        },
        "fallback": "preserve_baseline",
        "status": "candidate",
    }
