from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any, Mapping, Protocol, Sequence

from .events import Event, EventSink, NullEventSink
from .models import (
    HypothesisProgram,
    HypothesisStatus,
    ResidualExample,
    SplitName,
    stable_hash,
)


ROOT_PROPOSAL_REPLAY_POLICY_VERSION = "request_identical_root_proposal_replay_v1"
REPAIR_BRANCH_ID_POLICY_VERSION = "parent_content_scoped_repair_id_v1"
REPAIR_REQUEST_SCOPE_POLICY_VERSION = (
    "single_candidate_excludes_root_batch_contract_v1"
)
LEGACY_PROPOSAL_DIVERSITY_POLICY_VERSION = (
    "exact_count_pairwise_train_failure_activation_v1"
)
PROPOSAL_DIVERSITY_POLICY_VERSION = (
    "exact_count_train_failure_activation_audit_only_v2"
)


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
        failure_phase: str = "model_call",
        response_hash: str | None = None,
    ) -> None:
        super().__init__(
            f"{request_kind} failed during {failure_phase} ({error_type})"
        )
        self.request_kind = request_kind
        self.request_hash = request_hash
        self.error_type = error_type
        self.failure_phase = failure_phase
        self.response_hash = response_hash


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
        capability_payload = dict(capabilities or {})
        primary_metric = str(
            capability_payload.get("primary_metric") or "task_success"
        ).strip()
        if not primary_metric:
            raise ValueError("proposal primary metric is missing")
        proposal_batch_contract = capability_payload.get("proposal_batch_contract")
        diversity_contract_enabled = bool(
            isinstance(proposal_batch_contract, Mapping)
            and proposal_batch_contract.get("policy")
            == PROPOSAL_DIVERSITY_POLICY_VERSION
        )
        payload = {
            "request_kind": "propose_hypothesis_programs",
            "contract_version": "hypothesis_program_v1",
            "evaluator_epoch": evaluator_epoch,
            "constraints": _proposal_constraints(capability_payload),
            "output_schema": {
                "hypotheses": [_program_schema(capability_payload)]
            },
            "capabilities": capability_payload,
            "residuals": [
                _residual_payload(
                    residual,
                    labeled=bool(
                        capability_payload.get("training_evidence_contract")
                    ),
                )
                for residual in residuals
            ],
            "max_hypotheses": max_hypotheses,
        }
        if diversity_contract_enabled:
            configured_profile_roles = proposal_batch_contract.get("profile_roles", ())
            profile_roles = (
                [str(value) for value in configured_profile_roles]
                if isinstance(configured_profile_roles, (list, tuple))
                else []
            )
            configured_action_limit = proposal_batch_contract.get(
                "max_action_nodes_per_hypothesis",
                4,
            )
            payload["proposal_batch_contract"] = {
                "policy": PROPOSAL_DIVERSITY_POLICY_VERSION,
                "response_field": "hypotheses",
                "response_type": "array",
                "required_count": max_hypotheses,
                "diversity_unit": "train_failure_activation_or_action_treatment",
                "activation_signature_distinctness": (
                    "search_preference_audit_only"
                ),
                "action_treatment_diversity": (
                    "allowed_when_activation_signatures_coincide"
                ),
                "max_action_nodes_per_hypothesis": int(configured_action_limit),
                "profile_roles": profile_roles,
                "compact_output": True,
            }
            payload["output_schema"] = {
                "type": "object",
                "required": ["hypotheses"],
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "minItems": max_hypotheses,
                        "maxItems": max_hypotheses,
                        "items": _program_schema(capability_payload),
                    }
                },
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
        if not isinstance(response, Mapping):
            raise self._response_contract_error(
                payload=payload,
                response=response,
                expected_field="hypotheses",
                failure_phase="response_envelope",
                trace_id=trace_id,
            )
        rows = response.get("hypotheses")
        if not isinstance(rows, list) or (not rows and not diversity_contract_enabled):
            raise self._response_contract_error(
                payload=payload,
                response=response,
                expected_field="hypotheses",
                failure_phase="response_envelope",
                trace_id=trace_id,
            )
        staged_programs: list[tuple[int, HypothesisProgram]] = []
        transition_ids = tuple(sorted(residual.transition_id for residual in residuals))
        returned_rows = rows if diversity_contract_enabled else rows[:max_hypotheses]
        for index, row in enumerate(returned_rows):
            if not isinstance(row, Mapping):
                raise self._response_contract_error(
                    payload=payload,
                    response=response,
                    expected_field="hypotheses",
                    failure_phase="response_envelope",
                    trace_id=trace_id,
                    consumed_row_index=index,
                    consumed_row=row,
                )
            normalized = dict(row)
            _normalize_expected_effect_metric(normalized, primary_metric)
            normalized["evaluator_epoch"] = evaluator_epoch
            normalized["created_from_transition_ids"] = list(transition_ids)
            normalized.setdefault(
                "id",
                f"hyp_{stable_hash({'response': row, 'evaluator_epoch': evaluator_epoch, 'index': index})[:16]}",
            )
            try:
                program = HypothesisProgram.from_dict(normalized)
            except (TypeError, ValueError, OverflowError) as exc:
                raise self._response_contract_error(
                    payload=payload,
                    response=response,
                    expected_field="hypotheses",
                    failure_phase="response_program_parse",
                    trace_id=trace_id,
                    consumed_row_index=index,
                    consumed_row=row,
                    parse_error=exc,
                ) from exc
            staged_programs.append((index, program))
        if not staged_programs and not diversity_contract_enabled:
            raise self._response_contract_error(
                payload=payload,
                response=response,
                expected_field="hypotheses",
                failure_phase="response_envelope",
                trace_id=trace_id,
            )
        programs = [program for _, program in staged_programs]
        if diversity_contract_enabled and len(programs) != max_hypotheses:
            raise self._response_contract_error(
                payload=payload,
                response=response,
                expected_field="hypotheses",
                failure_phase="response_exact_count",
                trace_id=trace_id,
                expected_item_count=max_hypotheses,
                response_contract_policy=PROPOSAL_DIVERSITY_POLICY_VERSION,
            )
        if diversity_contract_enabled:
            failure_rows = tuple(
                residual
                for residual in residuals
                if residual.split is SplitName.TRAIN
                and residual.baseline_success is False
            )
            activation_signatures: list[tuple[bool | None, ...]] = []
            activation_audit_error_count = 0
            for _, program in staged_programs:
                signature: list[bool | None] = []
                for residual in failure_rows:
                    try:
                        signature.append(program.matches(residual.features))
                    except (TypeError, ValueError, OverflowError):
                        # Static validation owns malformed trigger rejection. This
                        # search audit must never turn signature diversity into a
                        # second response gate or a proposal retry.
                        signature.append(None)
                        activation_audit_error_count += 1
                activation_signatures.append(tuple(signature))
            signature_group_sizes: dict[str, int] = {}
            for signature in activation_signatures:
                signature_hash = stable_hash(
                    {"train_failure_activation_signature": signature}
                )
                signature_group_sizes[signature_hash] = (
                    signature_group_sizes.get(signature_hash, 0) + 1
                )
            signature_group_hashes = sorted(signature_group_sizes)
            self.event_sink.emit(
                Event(
                    event="proposal_activation_signature_audited",
                    stage="proposal.search_audit",
                    trace_id=trace_id,
                    payload={
                        "policy": PROPOSAL_DIVERSITY_POLICY_VERSION,
                        "candidate_count": len(programs),
                        "failure_train_row_count": len(failure_rows),
                        "distinct_activation_signature_count": len(
                            signature_group_hashes
                        ),
                        "activation_signature_group_hashes": (
                            signature_group_hashes
                        ),
                        "activation_signature_group_sizes": [
                            signature_group_sizes[signature_hash]
                            for signature_hash in signature_group_hashes
                        ],
                        "activation_audit_error_count": (
                            activation_audit_error_count
                        ),
                        "pairwise_distinct_required": False,
                        "search_preference_only": True,
                        "response_rejected": False,
                        "proposal_retry_requested": False,
                        "raw_content_persisted": False,
                    },
                )
            )
        for index, program in staged_programs:
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
        capability_payload = dict(capabilities or {})
        repair_request_scope_policy = capability_payload.get(
            "repair_request_scope_policy"
        )
        if repair_request_scope_policy not in {
            None,
            REPAIR_REQUEST_SCOPE_POLICY_VERSION,
        }:
            raise ValueError("unsupported repair request scope policy")
        scoped_single_repair = (
            repair_request_scope_policy == REPAIR_REQUEST_SCOPE_POLICY_VERSION
        )
        if scoped_single_repair:
            capability_payload.pop("proposal_batch_contract", None)
        payload = {
            "request_kind": "repair_hypothesis_program",
            "contract_version": "hypothesis_program_v1",
            "evaluator_epoch": parent.evaluator_epoch,
            "constraints": _proposal_constraints(capability_payload),
            "output_schema": {
                "hypothesis": _program_schema(capability_payload)
            },
            "capabilities": capability_payload,
            "parent": parent.to_dict(),
            "failed_checks": [dict(row) for row in failed_checks],
            "residuals": [
                _residual_payload(
                    residual,
                    labeled=bool(
                        capability_payload.get("training_evidence_contract")
                    ),
                )
                for residual in residuals
            ],
            "repair_depth": depth,
        }
        if scoped_single_repair:
            payload["repair_request_scope_policy"] = (
                REPAIR_REQUEST_SCOPE_POLICY_VERSION
            )
            payload["repair_response_contract"] = {
                "response_field": "hypothesis",
                "response_type": "object",
                "required_count": 1,
                "root_batch_contract_applies": False,
                "compact_output": True,
            }
        self._emit_model_event("hypothesis_repair_requested", trace_id, payload)
        response = self._complete(payload, trace_id=trace_id)
        if not isinstance(response, Mapping):
            raise self._response_contract_error(
                payload=payload,
                response=response,
                expected_field="hypothesis",
                failure_phase="response_envelope",
                trace_id=trace_id,
            )
        row = response.get("hypothesis")
        if not isinstance(row, Mapping):
            raise self._response_contract_error(
                payload=payload,
                response=response,
                expected_field="hypothesis",
                failure_phase="response_envelope",
                trace_id=trace_id,
            )
        normalized = dict(row)
        _normalize_expected_effect_metric(
            normalized,
            parent.expected_effect.metric,
        )
        normalized["evaluator_epoch"] = parent.evaluator_epoch
        normalized["parent_id"] = parent.id
        normalized["lineage"] = [*parent.lineage, parent.id]
        normalized["created_from_transition_ids"] = list(parent.created_from_transition_ids)
        model_supplied_id = str(normalized.get("id") or "").strip()
        normalized["status"] = HypothesisStatus.CANDIDATE.value
        normalized["id"] = "repair_identity_placeholder"
        try:
            canonical_child = HypothesisProgram.from_dict(normalized)
        except (TypeError, ValueError, OverflowError) as exc:
            raise self._response_contract_error(
                payload=payload,
                response=response,
                expected_field="hypothesis",
                failure_phase="response_program_parse",
                trace_id=trace_id,
                consumed_row=row,
                parse_error=exc,
            ) from exc
        canonical_child = replace(
            canonical_child,
            parent_id=parent.id,
            lineage=(*parent.lineage, parent.id),
        )
        canonical_child_content = canonical_child.to_dict()
        canonical_child_content.pop("id")
        parent_content = parent.to_dict()
        parent_content.pop("id")
        parent_content.pop("status")
        parent_content_hash = stable_hash(parent_content)
        branch_identity_hash = stable_hash(
            {
                "policy": REPAIR_BRANCH_ID_POLICY_VERSION,
                "parent_id": parent.id,
                "parent_content_hash": parent_content_hash,
                "repair_depth": depth,
                "canonical_program_without_id": canonical_child_content,
            }
        )
        child = replace(canonical_child, id=f"repair_{branch_identity_hash}")
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
                    "branch_id_policy": REPAIR_BRANCH_ID_POLICY_VERSION,
                    "branch_identity_hash": branch_identity_hash,
                    "parent_content_hash": parent_content_hash,
                    "model_supplied_child_id_hash": (
                        stable_hash({"id": model_supplied_id})
                        if model_supplied_id
                        else None
                    ),
                    "model_supplied_child_id_used": False,
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
                    "repair_request_scope_policy": payload.get(
                        "repair_request_scope_policy"
                    ),
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
                failure_phase="model_call",
            ) from exc

    def _response_contract_error(
        self,
        *,
        payload: Mapping[str, Any],
        response: Any,
        expected_field: str,
        failure_phase: str,
        trace_id: str,
        consumed_row_index: int | None = None,
        consumed_row: Any = None,
        parse_error: Exception | None = None,
        expected_item_count: int | None = None,
        failure_train_row_count: int | None = None,
        distinct_activation_signature_count: int | None = None,
        response_contract_policy: str | None = None,
    ) -> HypothesisProposalCallError:
        request_kind = str(payload.get("request_kind") or "hypothesis_proposal")
        request_hash = stable_hash(payload)
        response_hash = stable_hash(response)
        response_is_mapping = isinstance(response, Mapping)
        top_level_keys = (
            sorted(str(key) for key in response)
            if response_is_mapping
            else []
        )
        expected_field_present = (
            response_is_mapping and expected_field in response
        )
        expected_value = (
            response.get(expected_field) if expected_field_present else None
        )
        consumed_row_present = consumed_row_index is not None or consumed_row is not None
        self.event_sink.emit(
            Event(
                event="hypothesis_proposal_response_rejected",
                stage="proposal.response_contract",
                trace_id=trace_id,
                payload={
                    "request_kind": request_kind,
                    "request_hash": request_hash,
                    "response_hash": response_hash,
                    "failure_phase": failure_phase,
                    "error_type": "MalformedProposalResponse",
                    "candidate_local_failure": (
                        request_kind == "repair_hypothesis_program"
                    ),
                    "expected_field": expected_field,
                    "top_level_type": type(response).__name__,
                    "top_level_key_count": len(top_level_keys),
                    "top_level_key_set_hash": stable_hash(
                        {"keys": top_level_keys}
                    ),
                    "expected_field_present": expected_field_present,
                    "expected_field_type": (
                        type(expected_value).__name__
                        if expected_field_present
                        else None
                    ),
                    "expected_field_item_count": (
                        len(expected_value)
                        if isinstance(expected_value, (list, tuple))
                        else None
                    ),
                    "expected_item_count": expected_item_count,
                    "failure_train_row_count": failure_train_row_count,
                    "distinct_activation_signature_count": (
                        distinct_activation_signature_count
                    ),
                    "response_contract_policy": response_contract_policy,
                    "consumed_row_present": consumed_row_present,
                    "consumed_row_index": consumed_row_index,
                    "consumed_row_type": (
                        type(consumed_row).__name__
                        if consumed_row_present
                        else None
                    ),
                    "parse_error_type": (
                        type(parse_error).__name__ if parse_error else None
                    ),
                    "raw_error_persisted": False,
                    "raw_content_persisted": False,
                },
            )
        )
        return HypothesisProposalCallError(
            request_kind=request_kind,
            request_hash=request_hash,
            error_type="MalformedProposalResponse",
            failure_phase=failure_phase,
            response_hash=response_hash,
        )


def _normalize_expected_effect_metric(
    payload: dict[str, Any],
    primary_metric: str,
) -> None:
    """Replace the model's metric label with the evaluator-owned metric."""

    expected_effect = payload.get("expected_effect")
    normalized_effect = (
        dict(expected_effect) if isinstance(expected_effect, Mapping) else {}
    )
    normalized_effect["metric"] = primary_metric
    payload["expected_effect"] = normalized_effect


def _residual_payload(
    residual: ResidualExample,
    *,
    labeled: bool = False,
) -> dict[str, Any]:
    payload = {
        "transition_id": residual.transition_id,
        "task_id_hash": stable_hash({"task_id": residual.task_id}),
        "family": residual.family,
        "features": dict(residual.features),
        "failure_type": residual.failure_type,
        "evaluator_feedback": list(residual.evaluator_feedback),
        "baseline_success": residual.baseline_success,
        "context": dict(residual.context),
    }
    if labeled:
        payload["evidence_label"] = (
            "success_control" if residual.baseline_success else "failure"
        )
    return payload


def _proposal_constraints(
    capabilities: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    action_contract = (capabilities or {}).get("action_contract")
    backend_operations = None
    if isinstance(action_contract, Mapping):
        operations = action_contract.get("allowed_action_operations")
        if isinstance(operations, (list, tuple)):
            backend_operations = [str(value) for value in operations]
    action_semantics = (
        str(action_contract.get("semantics"))
        if isinstance(action_contract, Mapping)
        else "typed_runtime_action_v1"
    )
    prompt_directive_backend = "prompt_directive" in action_semantics
    external_evidence_is_hidden = bool(
        action_contract.get("external_evidence_is_hidden", False)
        if isinstance(action_contract, Mapping)
        else False
    )
    constraints = {
        "allowed_kinds": ["task", "policy", "evaluator"],
        "fallback_must_equal": "preserve_baseline",
        "trigger_must_use_structured_features": True,
        "trigger_keys_must_come_from_capabilities_runtime_trigger_contract": True,
        "residual_context_may_shape_actions_but_must_not_be_used_in_trigger_or_anti_trigger": True,
        "action_graph_must_change_runtime": not prompt_directive_backend,
        "action_graph_must_change_backend_treatment": True,
        "fine_grained_action_execution_receipt_expected": (
            not prompt_directive_backend
        ),
        "gold_answer_fields_forbidden": True,
        "required_verifier_anchor": True,
        "required_expected_effect": ["metric", "minimum_delta", "maximum_harm_rate", "maximum_cost_ratio"],
        "allowed_action_operations": backend_operations
        or [
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
        "action_semantics": action_semantics,
        "external_verifier_is_agent_callable": not external_evidence_is_hidden,
        "forbidden_action_references": (
            [
                "external verifier anchor",
                "hidden benchmark verifier",
                "policy_off outcome",
                "policy_on outcome",
            ]
            if external_evidence_is_hidden
            else []
        ),
    }
    if prompt_directive_backend:
        constraints.update(
            {
                "prompt_directive_action_values_must_be_complete_imperative_"
                "task_local_sentences": True,
                "prompt_directive_action_value_grounding_source": (
                    "TRAIN residual context.task_instruction"
                ),
                "prompt_directive_enum_only_action_values_forbidden": True,
                "prompt_directive_mapping_mode_check_labels_forbidden": True,
                "prompt_directive_activated_action_preserve_baseline_claim_"
                "forbidden": True,
                "prompt_directive_top_level_fallback_remains_preserve_baseline": True,
            }
        )
    training_evidence_contract = (capabilities or {}).get(
        "training_evidence_contract"
    )
    if isinstance(training_evidence_contract, Mapping):
        constraints.update(
            {
                "training_rows_are_explicitly_labeled": True,
                "training_row_label_field": "baseline_success",
                "success_rows_are_anti_trigger_negative_controls": True,
                "success_rows_must_not_increase_failure_trigger_support": True,
                "success_control_context_must_be_empty": True,
                "training_evidence_policy": str(
                    training_evidence_contract.get("policy") or ""
                ),
            }
        )
    proposal_batch_contract = (capabilities or {}).get(
        "proposal_batch_contract"
    )
    train_coverage_objective = (capabilities or {}).get(
        "train_coverage_objective"
    )
    if (
        isinstance(proposal_batch_contract, Mapping)
        and proposal_batch_contract.get("policy")
        == PROPOSAL_DIVERSITY_POLICY_VERSION
    ):
        constraints.update(
            {
                "proposal_count_must_equal_requested_count": True,
                "proposal_activation_signature_distinctness_is_search_preference": True,
                "proposal_activation_signatures_are_audited_not_rejected": True,
                "proposal_action_or_backend_treatment_diversity_allowed": True,
                "proposal_same_activation_signature_allowed_when_treatment_differs": True,
                "proposal_output_must_be_compact": True,
                "proposal_max_action_nodes_per_hypothesis": int(
                    proposal_batch_contract.get(
                        "max_action_nodes_per_hypothesis",
                        4,
                    )
                ),
            }
        )
    if isinstance(train_coverage_objective, Mapping):
        constraints.update(
            {
                "candidate_search_uses_train_only": True,
                "candidate_search_coverage_unit": str(
                    train_coverage_objective.get("coverage_unit") or ""
                ),
                "candidate_search_family_target": int(
                    train_coverage_objective.get(
                        "failure_activation_family_target",
                        0,
                    )
                ),
                "candidate_search_validation_outcomes_forbidden": True,
            }
        )
    return constraints


def _program_schema(
    capabilities: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    action_contract = (capabilities or {}).get("action_contract")
    action_semantics = (
        str(action_contract.get("semantics"))
        if isinstance(action_contract, Mapping)
        else "typed_runtime_action_v1"
    )
    prompt_directive_backend = "prompt_directive" in action_semantics
    external_evidence_is_hidden = bool(
        action_contract.get("external_evidence_is_hidden", False)
        if isinstance(action_contract, Mapping)
        else False
    )
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
                "target": (
                    "task-local subject of the imperative directive"
                    if prompt_directive_backend
                    else (
                        "task-local step, condition, evidence, or artifact; never the external verifier or policy-off/on outcome"
                        if external_evidence_is_hidden
                        else "declared capability, step, verifier, or artifact"
                    )
                ),
                "value": (
                    "complete imperative task-local sentence grounded in TRAIN "
                    "residual context.task_instruction; never an enum-only value, "
                    "mapping/mode/check label, or preserve_baseline claim"
                    if prompt_directive_backend
                    else "JSON value"
                ),
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
