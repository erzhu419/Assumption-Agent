from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any, Mapping, Protocol, Sequence

from .events import Event, EventSink, NullEventSink
from .models import HypothesisProgram, ResidualExample, SplitName, stable_hash


ROOT_PROPOSAL_REPLAY_POLICY_VERSION = "request_identical_root_proposal_replay_v1"


class ProposalModel(Protocol):
    """Model adapter that returns parsed JSON, never unstructured prose."""

    def complete(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


class HypothesisProposalCallError(RuntimeError):
    """Sanitized failure from one proposal or repair model request."""

    def __init__(
        self,
        *,
        request_kind: str,
        request_hash: str,
        error_type: str,
    ) -> None:
        super().__init__(f"{request_kind} model call failed ({error_type})")
        self.request_kind = request_kind
        self.request_hash = request_hash
        self.error_type = error_type


class StructuredHypothesisProposer:
    def __init__(self, model: ProposalModel, *, event_sink: EventSink | None = None) -> None:
        self.model = model
        self.event_sink = event_sink or NullEventSink()
        self._root_replay_lock = threading.Lock()
        self._root_replay_records: dict[
            str,
            tuple[tuple[HypothesisProgram, ...], str, str],
        ] = {}

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
        request_hash = stable_hash(payload)
        with self._root_replay_lock:
            replay = self._root_replay_records.get(request_hash)
        if replay is not None:
            programs, source_trace_id, program_set_hash = replay
            self.event_sink.emit(
                Event(
                    event="root_proposal_evidence_replayed",
                    stage="proposal.replay",
                    trace_id=trace_id,
                    payload={
                        "policy": ROOT_PROPOSAL_REPLAY_POLICY_VERSION,
                        "request_hash": request_hash,
                        "source_trace_id": source_trace_id,
                        "target_trace_id": trace_id,
                        "program_count": len(programs),
                        "program_set_hash": program_set_hash,
                        "request_identical": True,
                        "new_proposal_model_executions": 0,
                        "evaluator_epoch": evaluator_epoch,
                        "sealed_test_accessed": False,
                        "raw_content_persisted": False,
                    },
                )
            )
            return programs
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
        result = tuple(programs)
        program_set_hash = stable_hash(
            {"program_hashes": [program.payload_hash for program in result]}
        )
        with self._root_replay_lock:
            self._root_replay_records.setdefault(
                request_hash,
                (result, trace_id, program_set_hash),
            )
        self.event_sink.emit(
            Event(
                event="root_proposal_evidence_recorded",
                stage="proposal.replay",
                trace_id=trace_id,
                payload={
                    "policy": ROOT_PROPOSAL_REPLAY_POLICY_VERSION,
                    "request_hash": request_hash,
                    "source_trace_id": trace_id,
                    "program_count": len(result),
                    "program_set_hash": program_set_hash,
                    "new_proposal_model_executions": 1,
                    "evaluator_epoch": evaluator_epoch,
                    "sealed_test_accessed": False,
                    "raw_content_persisted": False,
                },
            )
        )
        return result

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
        request_kind = str(payload.get("request_kind") or "hypothesis_proposal")
        request_hash = stable_hash(payload)
        try:
            traced = getattr(self.model, "complete_with_trace", None)
            if callable(traced):
                return traced(payload, trace_id=trace_id)
            return self.model.complete(payload)
        except HypothesisProposalCallError:
            raise
        except Exception as exc:
            self.event_sink.emit(
                Event(
                    event="hypothesis_proposal_model_call_failed",
                    stage="proposal.model",
                    trace_id=trace_id,
                    payload={
                        "request_kind": request_kind,
                        "request_hash": request_hash,
                        "error_type": type(exc).__name__,
                        "candidate_local_failure": (
                            request_kind == "repair_hypothesis_program"
                        ),
                        "raw_error_persisted": False,
                        "raw_content_persisted": False,
                    },
                )
            )
            raise HypothesisProposalCallError(
                request_kind=request_kind,
                request_hash=request_hash,
                error_type=type(exc).__name__,
            ) from exc


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
        "trigger_keys_must_come_from_capabilities_runtime_trigger_contract": True,
        "residual_context_may_shape_actions_but_must_not_be_used_in_trigger_or_anti_trigger": True,
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
