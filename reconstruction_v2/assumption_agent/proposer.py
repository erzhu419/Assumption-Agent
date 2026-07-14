from __future__ import annotations

import json
import re
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
from .typed_operator_grammar import (
    TYPED_OPERATOR_GRAMMAR_VERSION,
    TYPED_REPAIR_BRANCH_ID_POLICY_VERSION,
    TYPED_RECIPE_SELECTION_VERSION,
    TypedProgramBindingRegistry,
    TypedRecipeSelectionSnapshot,
    canonical_typed_recipe_selection_request,
    canonical_typed_recipe_selection_response,
    materialize_recipe_selection,
    typed_recipe_id_for_program,
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
TRAIN_ACTION_DESIGN_POLICY_VERSION = (
    "train_only_material_action_delta_prompt_audit_v1"
)
FAMILY_SLOT_PROPOSAL_FORMATION_POLICY_VERSION = (
    "train_only_profile_grounded_family_slots_v1"
)
FAMILY_SLOT_PROPOSAL_FORMATION_POLICY_V2 = (
    "train_only_profile_grounded_family_slots_v2"
)
FAMILY_SLOT_PROPOSAL_FORMATION_POLICY_VERSIONS = frozenset(
    {
        FAMILY_SLOT_PROPOSAL_FORMATION_POLICY_VERSION,
        FAMILY_SLOT_PROPOSAL_FORMATION_POLICY_V2,
    }
)
TRAIN_ACTION_DESIGN_POLICY_VERSIONS = frozenset(
    {TRAIN_ACTION_DESIGN_POLICY_VERSION}
)
TRAIN_ACTION_DESIGN_INTERNAL_CONTEXT_KEY = "_train_action_design_profile"
TYPED_ROOT_RECIPE_SELECTION_REQUEST = "select_typed_root_recipe"
TYPED_REPAIR_RECIPE_SELECTION_REQUEST = "select_typed_repair_recipe"
def train_action_quality_contract(policy: str | None) -> dict[str, Any] | None:
    if policy is None:
        return None
    if policy not in TRAIN_ACTION_DESIGN_POLICY_VERSIONS:
        raise ValueError(f"unsupported TRAIN action design policy: {policy}")
    return {
        "policy": policy,
        "task_instruction_role": "baseline_requirement_not_treatment",
        "profile_reference_field": "action_context_profile_hash",
        "profile_map_field": "train_action_design_profiles",
        "required_material_delta_kinds": [
            "exact_constant_or_mapping",
            "concrete_local_tool_command",
            "artifact_internal_manipulation",
        ],
        "minimum_material_delta_per_hypothesis": 1,
        "vague_placeholder_policy": (
            "do_not_substitute_request_collect_or_verify_for_missing_detail"
        ),
        "static_model_knowledge_allowed": True,
        "proposal_external_tools_allowed": False,
        "runtime_network_or_install_actions_allowed": False,
        "enforcement": "prompt_and_audit_only",
        "response_rejection_allowed": False,
        "proposal_retry_allowed": False,
        "candidate_selection_affected": False,
        "evidence_scope": (
            "train_instruction_runtime_profile_and_policy_off_action_trace"
        ),
        "validation_outcomes_used": False,
        "verifier_content_used": False,
        "test_content_used": False,
    }


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
    def __init__(
        self,
        model: ProposalModel,
        *,
        event_sink: EventSink | None = None,
        typed_program_registry: TypedProgramBindingRegistry | None = None,
    ) -> None:
        self.model = model
        self.event_sink = event_sink or NullEventSink()
        self.typed_program_registry = (
            typed_program_registry or TypedProgramBindingRegistry()
        )
        self._root_replay_lock = threading.Lock()
        self._root_replay_records: dict[
            str,
            tuple[tuple[HypothesisProgram, ...], str, str],
        ] = {}
        self._family_slot_target_lock = threading.Lock()
        self._family_slot_targets_by_program_hash: dict[str, set[str]] = {}
        self._latest_family_slot_batch_targets: dict[
            str,
            tuple[tuple[str, ...], tuple[str | None, ...]],
        ] = {}
        self._typed_selection_replay_lock = threading.Lock()
        self._typed_selection_replay_records: dict[
            str,
            tuple[HypothesisProgram, str, str],
        ] = {}

    def family_slot_target_for(
        self,
        program: HypothesisProgram,
    ) -> str | None:
        with self._family_slot_target_lock:
            targets = set(
                self._family_slot_targets_by_program_hash.get(
                    program.payload_hash,
                    set(),
                )
            )
        if len(targets) != 1:
            return None
        return next(iter(targets))

    def record_family_slot_batch(
        self,
        programs: Sequence[HypothesisProgram],
        targets: Sequence[str],
    ) -> None:
        if len(programs) != len(targets):
            raise ValueError("family-slot batch programs and targets must align")
        program_hashes = tuple(program.payload_hash for program in programs)
        canonical_targets = tuple(
            str(target).strip() or None for target in targets
        )
        proposal_set_hash = stable_hash(
            {"candidate_hashes": sorted(program_hashes)}
        )
        with self._family_slot_target_lock:
            self._latest_family_slot_batch_targets[proposal_set_hash] = (
                program_hashes,
                canonical_targets,
            )

    def family_slot_targets_for(
        self,
        programs: Sequence[HypothesisProgram],
    ) -> tuple[str | None, ...]:
        program_hashes = tuple(program.payload_hash for program in programs)
        proposal_set_hash = stable_hash(
            {"candidate_hashes": sorted(program_hashes)}
        )
        with self._family_slot_target_lock:
            batch = self._latest_family_slot_batch_targets.get(
                proposal_set_hash
            )
        if batch is not None:
            recorded_hashes, recorded_targets = batch
            if recorded_hashes == program_hashes:
                return recorded_targets
            target_by_program_hash = dict(
                zip(recorded_hashes, recorded_targets)
            )
            return tuple(
                target_by_program_hash.get(program_hash)
                for program_hash in program_hashes
            )
        return tuple(
            self.family_slot_target_for(program) for program in programs
        )

    def _record_family_slot_target(
        self,
        programs: Sequence[HypothesisProgram],
        *,
        target_family: str,
    ) -> None:
        target = target_family.strip()
        if not target:
            return
        with self._family_slot_target_lock:
            for program in programs:
                self._family_slot_targets_by_program_hash.setdefault(
                    program.payload_hash,
                    set(),
                ).add(target)

    def select_typed_recipe(
        self,
        *,
        snapshot: TypedRecipeSelectionSnapshot,
        evaluator_epoch: str,
        trace_id: str,
        parent: HypothesisProgram | None = None,
        failed_checks: Sequence[Mapping[str, Any]] = (),
        depth: int = 0,
        excluded_recipe_ids: Sequence[str] = (),
        selection_round: int | None = None,
    ) -> HypothesisProgram:
        """Select one opaque recipe and materialize it with harness-owned text."""

        snapshot_issues = snapshot.validate()
        if snapshot_issues:
            raise PermissionError(
                f"typed selection snapshot is invalid: {list(snapshot_issues)}"
            )
        snapshot_ledger = (
            self.typed_program_registry.require_snapshot_ledger(snapshot)
        )
        if not evaluator_epoch.strip():
            raise ValueError("typed recipe selection evaluator epoch is missing")
        repair = parent is not None
        if repair != bool(depth):
            raise ValueError("typed repair depth and parent must be paired")
        if selection_round is not None and (
            not isinstance(selection_round, int)
            or isinstance(selection_round, bool)
            or selection_round < 1
        ):
            raise ValueError("typed selection round must be a positive integer")
        graph_recipe_ids = {
            row.recipe_id for row in snapshot.graph.recipes
        }
        if (
            any(
                not isinstance(recipe_id, str) or not recipe_id
                for recipe_id in excluded_recipe_ids
            )
        ):
            raise PermissionError("typed selection exclusions are malformed")
        canonical_excluded_recipe_ids = set(excluded_recipe_ids)
        if not canonical_excluded_recipe_ids.issubset(graph_recipe_ids):
            raise PermissionError("typed selection exclusions are malformed")
        parent_recipe_id: str | None = None
        parent_binding = None
        if parent is not None:
            if parent.evaluator_epoch != evaluator_epoch:
                raise PermissionError("typed repair crossed evaluator epochs")
            parent_binding = (
                self.typed_program_registry.require_for_snapshot(
                    parent,
                    snapshot,
                )
            )
            parent_recipe_id = typed_recipe_id_for_program(
                parent,
                snapshot=snapshot,
            )
            if parent_recipe_id is None:
                raise PermissionError(
                    "typed repair parent is outside the frozen snapshot"
                )
            if parent_binding.recipe_id != parent_recipe_id:
                raise PermissionError(
                    "typed repair parent recipe binding mismatch"
                )
            canonical_excluded_recipe_ids.update(
                parent_binding.excluded_recipe_ids
            )
            canonical_excluded_recipe_ids.update(
                self.typed_program_registry.lineage_recipe_ids(
                    parent,
                    snapshot=snapshot,
                )
            )

        canonical_excluded = tuple(sorted(canonical_excluded_recipe_ids))
        expected_selection_round = len(canonical_excluded) + 1
        if selection_round is None:
            selection_round = expected_selection_round
        elif selection_round != expected_selection_round:
            raise PermissionError(
                "typed selection round does not match the complete exclusion scope"
            )
        request_kind = (
            TYPED_REPAIR_RECIPE_SELECTION_REQUEST
            if repair
            else TYPED_ROOT_RECIPE_SELECTION_REQUEST
        )
        payload = canonical_typed_recipe_selection_request(
            snapshot=snapshot,
            snapshot_ledger=snapshot_ledger,
            evaluator_epoch=evaluator_epoch,
            selection_round=selection_round,
            excluded_recipe_ids=canonical_excluded,
            parent_program_hash=(
                parent_binding.program_identity_hash
                if parent_binding is not None
                else None
            ),
            parent_recipe_id=parent_recipe_id,
            failed_checks=failed_checks,
            repair_depth=depth,
        )
        if payload["request_kind"] != request_kind:
            raise PermissionError("typed selection request kind is inconsistent")
        output_schema = payload["output_schema"]
        allowed_recipe_ids = list(
            output_schema["properties"]["recipe_id"]["enum"]
        )
        request_hash = stable_hash(payload)
        self.event_sink.emit(
            Event(
                event="typed_recipe_selection_requested",
                stage="proposal.typed_selection",
                trace_id=trace_id,
                payload={
                    "request_kind": request_kind,
                    "request_hash": request_hash,
                    "snapshot_hash": snapshot.snapshot_hash,
                    "snapshot_ledger_hash": snapshot_ledger.ledger_hash,
                    "graph_hash": snapshot.expected_graph_hash,
                    "model_catalog_hash": (
                        snapshot.expected_model_catalog_hash
                    ),
                    "target_family_hash": (
                        snapshot.graph.target_family_hash
                    ),
                    "allowed_recipe_count": len(allowed_recipe_ids),
                    "selection_round": selection_round,
                    "excluded_recipe_count": len(canonical_excluded),
                    "excluded_recipe_set_hash": stable_hash(
                        {"recipe_ids": list(canonical_excluded)}
                    ),
                    "repair_depth": depth,
                    "model_output_fields": ["recipe_id"],
                    "model_authored_primitive_count": 0,
                    "raw_artifact_locator_disclosure_count": 0,
                    "raw_content_persisted": False,
                },
            )
        )
        with self._typed_selection_replay_lock:
            replay = self._typed_selection_replay_records.get(request_hash)
        if replay is not None:
            program, response_hash, source_trace_id = replay
            self.typed_program_registry.require(program)
            self.event_sink.emit(
                Event(
                    event="typed_recipe_selection_replayed",
                    stage="proposal.typed_selection",
                    trace_id=trace_id,
                    payload={
                        "request_kind": request_kind,
                        "request_hash": request_hash,
                        "response_hash": response_hash,
                        "snapshot_hash": snapshot.snapshot_hash,
                        "snapshot_ledger_hash": snapshot_ledger.ledger_hash,
                        "source_trace_id": source_trace_id,
                        "target_trace_id": trace_id,
                        "program_id": program.id,
                        "program_hash": program.payload_hash,
                        "new_selector_calls": 0,
                        "raw_content_persisted": False,
                    },
                )
            )
            return program

        response = self._complete(payload, trace_id=trace_id)
        if not isinstance(response, Mapping) or set(response) != {"recipe_id"}:
            raise self._response_contract_error(
                payload=payload,
                response=response,
                expected_field="recipe_id",
                failure_phase="typed_selection_response_envelope",
                trace_id=trace_id,
                response_contract_policy=TYPED_RECIPE_SELECTION_VERSION,
            )
        recipe_id = response.get("recipe_id")
        if not isinstance(recipe_id, str) or recipe_id not in allowed_recipe_ids:
            raise self._response_contract_error(
                payload=payload,
                response=response,
                expected_field="recipe_id",
                failure_phase="typed_selection_recipe_reference",
                trace_id=trace_id,
                response_contract_policy=TYPED_RECIPE_SELECTION_VERSION,
            )
        try:
            program = materialize_recipe_selection(
                {"recipe_id": recipe_id},
                graph=snapshot.graph,
                evaluator_epoch=evaluator_epoch,
                expected_graph_hash=snapshot.expected_graph_hash,
                expected_model_catalog_hash=(
                    snapshot.expected_model_catalog_hash
                ),
            )
        except PermissionError as exc:
            raise self._response_contract_error(
                payload=payload,
                response=response,
                expected_field="recipe_id",
                failure_phase="typed_selection_materialization",
                trace_id=trace_id,
                parse_error=exc,
                response_contract_policy=TYPED_RECIPE_SELECTION_VERSION,
            ) from exc

        parent_hash: str | None = None
        if parent is not None:
            assert parent_binding is not None
            parent_hash = parent_binding.program_identity_hash
            canonical_content = program.to_dict()
            canonical_content.pop("id", None)
            branch_identity_hash = stable_hash(
                {
                    "policy": TYPED_REPAIR_BRANCH_ID_POLICY_VERSION,
                    "parent_program_hash": parent_hash,
                    "snapshot_hash": snapshot.snapshot_hash,
                    "snapshot_ledger_hash": snapshot_ledger.ledger_hash,
                    "recipe_id": recipe_id,
                    "repair_depth": depth,
                    "canonical_program_without_id": canonical_content,
                }
            )
            program = replace(
                program,
                id=f"repair_{branch_identity_hash}",
                parent_id=parent.id,
                lineage=(*parent.lineage, parent.id),
                created_from_transition_ids=(
                    parent.created_from_transition_ids
                ),
            )
        canonical_response = canonical_typed_recipe_selection_response(
            recipe_id
        )
        if dict(response) != canonical_response:
            raise PermissionError("typed selection response is not canonical")
        response_hash = stable_hash(canonical_response)
        binding = self.typed_program_registry.register(
            program,
            snapshot=snapshot,
            recipe_id=recipe_id,
            request_kind=request_kind,
            request_hash=request_hash,
            response_hash=response_hash,
            selection_round=selection_round,
            excluded_recipe_ids=canonical_excluded,
            parent=parent,
            failed_checks=failed_checks,
            repair_depth=depth,
        )
        with self._typed_selection_replay_lock:
            existing = self._typed_selection_replay_records.get(request_hash)
            if existing is not None and existing[0] != program:
                raise PermissionError("typed selection replay conflict")
            self._typed_selection_replay_records[request_hash] = (
                program,
                response_hash,
                trace_id,
            )
        self.event_sink.emit(
            Event(
                event="typed_recipe_selection_materialized",
                stage="proposal.typed_selection",
                trace_id=trace_id,
                payload={
                    "request_kind": request_kind,
                    "request_hash": request_hash,
                    "response_hash": response_hash,
                    "snapshot_hash": snapshot.snapshot_hash,
                    "graph_hash": snapshot.expected_graph_hash,
                    "model_catalog_hash": (
                        snapshot.expected_model_catalog_hash
                    ),
                    "recipe_id": recipe_id,
                    "program_id": program.id,
                    "program_hash": program.payload_hash,
                    "binding_hash": binding.binding_hash,
                    "selection_round": selection_round,
                    "excluded_recipe_count": len(canonical_excluded),
                    "excluded_recipe_set_hash": stable_hash(
                        {"recipe_ids": list(canonical_excluded)}
                    ),
                    "repair_depth": depth,
                    "model_authored_primitive_count": 0,
                    "harness_owned_materialization": True,
                    "raw_content_persisted": False,
                },
            )
        )
        return program

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
        family_slot_contract = capability_payload.get("family_slot_contract")
        family_slot_policy = (
            str(family_slot_contract.get("policy") or "")
            if isinstance(family_slot_contract, Mapping)
            else ""
        )
        family_slot_contract_enabled = bool(
            isinstance(family_slot_contract, Mapping)
            and family_slot_policy
            in FAMILY_SLOT_PROPOSAL_FORMATION_POLICY_VERSIONS
        )
        if family_slot_contract_enabled and max_hypotheses != 1:
            raise ValueError("family-slot proposal requests exactly one hypothesis")
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
        elif family_slot_contract_enabled:
            payload["family_slot_response_contract"] = {
                "policy": family_slot_policy,
                "response_field": "hypothesis",
                "response_type": "object",
                "required_count": 1,
                "root_batch_contract_applies": False,
                "response_rejection_by_diversity_allowed": False,
                "proposal_retry_by_diversity_allowed": False,
                "compact_output": True,
            }
            payload["output_schema"] = {
                "type": "object",
                "required": ["hypothesis"],
                "properties": {
                    "hypothesis": _program_schema(capability_payload),
                },
            }
        self._emit_model_event("hypothesis_proposal_requested", trace_id, payload)
        request_hash = stable_hash(payload)
        with self._root_replay_lock:
            replay = self._root_replay_records.get(request_hash)
        if replay is not None:
            programs, source_trace_id, program_set_hash = replay
            if family_slot_contract_enabled:
                self._record_family_slot_target(
                    programs,
                    target_family=str(
                        family_slot_contract.get("target_failure_family") or ""
                    ),
                )
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
        expected_response_field = (
            "hypothesis" if family_slot_contract_enabled else "hypotheses"
        )
        if not isinstance(response, Mapping):
            raise self._response_contract_error(
                payload=payload,
                response=response,
                expected_field=expected_response_field,
                failure_phase="response_envelope",
                trace_id=trace_id,
            )
        if family_slot_contract_enabled:
            singular_row = response.get("hypothesis")
            if not isinstance(singular_row, Mapping):
                raise self._response_contract_error(
                    payload=payload,
                    response=response,
                    expected_field="hypothesis",
                    failure_phase="response_envelope",
                    trace_id=trace_id,
                    expected_item_count=1,
                    response_contract_policy=family_slot_policy,
                )
            rows = [singular_row]
        else:
            rows = response.get("hypotheses")
            if not isinstance(rows, list) or (
                not rows and not diversity_contract_enabled
            ):
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
                    expected_field=expected_response_field,
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
                    expected_field=expected_response_field,
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
                expected_field=expected_response_field,
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
        self._emit_action_delta_audit(
            programs,
            residuals=residuals,
            capabilities=capability_payload,
            trace_id=trace_id,
            request_kind="root_proposal",
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
        if family_slot_contract_enabled:
            self._record_family_slot_target(
                result,
                target_family=str(
                    family_slot_contract.get("target_failure_family") or ""
                ),
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
        typed_recipe_snapshot: TypedRecipeSelectionSnapshot | None = None,
        trace_id: str,
    ) -> HypothesisProgram:
        if any(residual.split is not SplitName.TRAIN for residual in residuals):
            raise PermissionError("recursive repair may use training residuals only")
        try:
            parent_typed_binding = self.typed_program_registry.require(parent)
        except PermissionError:
            parent_typed_binding = None
        if (
            parent_typed_binding is None
            and self.typed_program_registry.matches_registered_snapshot(parent)
        ):
            raise PermissionError(
                "unbound typed parent cannot enter generic free-text repair"
            )
        if parent_typed_binding is not None and typed_recipe_snapshot is None:
            raise PermissionError(
                "typed parent cannot enter generic free-text repair"
            )
        if typed_recipe_snapshot is not None:
            return self.select_typed_recipe(
                snapshot=typed_recipe_snapshot,
                evaluator_epoch=parent.evaluator_epoch,
                trace_id=trace_id,
                parent=parent,
                failed_checks=failed_checks,
                depth=depth,
            )
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
        self._emit_action_delta_audit(
            (child,),
            residuals=residuals,
            capabilities=capability_payload,
            trace_id=trace_id,
            request_kind="repair_proposal",
        )
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

    def _emit_action_delta_audit(
        self,
        programs: Sequence[HypothesisProgram],
        *,
        residuals: Sequence[ResidualExample],
        capabilities: Mapping[str, Any],
        trace_id: str,
        request_kind: str,
    ) -> None:
        contract = capabilities.get("action_quality_contract")
        if not isinstance(contract, Mapping):
            return
        if contract.get("policy") != TRAIN_ACTION_DESIGN_POLICY_VERSION:
            return
        try:
            profile_rows = capabilities.get("train_action_design_profiles")
            profiles = profile_rows if isinstance(profile_rows, Mapping) else {}
            audits = [
                _action_delta_audit_row(
                    program,
                    residuals=residuals,
                    profiles=profiles,
                )
                for program in programs
            ]
            self.event_sink.emit(
                Event(
                    event="proposal_action_delta_audited",
                    stage="proposal.action_quality_audit",
                    trace_id=trace_id,
                    payload={
                        "policy": TRAIN_ACTION_DESIGN_POLICY_VERSION,
                        "request_kind": request_kind,
                        "candidate_count": len(audits),
                        "candidate_audits": audits,
                        "candidate_with_material_delta_count": sum(
                            bool(row["observed_delta_kinds"]) for row in audits
                        ),
                        "candidate_with_restatement_risk_count": sum(
                            bool(row["restatement_risk"]) for row in audits
                        ),
                        "response_rejected": False,
                        "proposal_retry_requested": False,
                        "recursive_repair_requested_by_audit": False,
                        "candidate_selection_affected": False,
                        "promotion_gate_affected": False,
                        "validation_outcomes_used": False,
                        "verifier_content_used": False,
                        "test_content_used": False,
                        "raw_content_persisted": False,
                    },
                )
            )
        except Exception as exc:
            # This policy is deliberately diagnostic-only: malformed audit input
            # must never reject, retry, re-rank, or otherwise change a proposal.
            try:
                self.event_sink.emit(
                    Event(
                        event="proposal_action_delta_audit_failed",
                        stage="proposal.action_quality_audit",
                        trace_id=trace_id,
                        payload={
                            "policy": TRAIN_ACTION_DESIGN_POLICY_VERSION,
                            "request_kind": request_kind,
                            "error_type": type(exc).__name__,
                            "response_rejected": False,
                            "proposal_retry_requested": False,
                            "candidate_selection_affected": False,
                            "promotion_gate_affected": False,
                            "raw_error_persisted": False,
                        },
                    )
                )
            except Exception:
                pass

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
                            request_kind
                            in {
                                "repair_hypothesis_program",
                                TYPED_REPAIR_RECIPE_SELECTION_REQUEST,
                            }
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
                        request_kind
                        in {
                            "repair_hypothesis_program",
                            TYPED_REPAIR_RECIPE_SELECTION_REQUEST,
                        }
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
    public_context = dict(residual.context)
    public_context.pop(TRAIN_ACTION_DESIGN_INTERNAL_CONTEXT_KEY, None)
    payload = {
        "transition_id": residual.transition_id,
        "task_id_hash": stable_hash({"task_id": residual.task_id}),
        "family": residual.family,
        "features": dict(residual.features),
        "failure_type": residual.failure_type,
        "evaluator_feedback": list(residual.evaluator_feedback),
        "baseline_success": residual.baseline_success,
        "context": public_context,
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
    action_quality_contract = (capabilities or {}).get(
        "action_quality_contract"
    )
    action_quality_policy = (
        str(action_quality_contract.get("policy") or "")
        if isinstance(action_quality_contract, Mapping)
        else ""
    )
    material_action_delta_prompt = (
        action_quality_policy == TRAIN_ACTION_DESIGN_POLICY_VERSION
    )
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
                    (
                        "TRAIN residual task requirement plus referenced "
                        "train_action_design_profiles and model static knowledge"
                    )
                    if material_action_delta_prompt
                    else "TRAIN residual context.task_instruction"
                ),
                "prompt_directive_enum_only_action_values_forbidden": True,
                "prompt_directive_mapping_mode_check_labels_forbidden": True,
                "prompt_directive_activated_action_preserve_baseline_claim_"
                "forbidden": True,
                "prompt_directive_top_level_fallback_remains_preserve_baseline": True,
            }
        )
    if material_action_delta_prompt:
        constraints.update(
            {
                "task_instruction_is_baseline_requirement_not_treatment": True,
                "each_hypothesis_must_add_material_runtime_delta": True,
                "material_delta_kinds": list(
                    action_quality_contract.get(
                        "required_material_delta_kinds",
                        (),
                    )
                ),
                "material_delta_must_not_already_be_explicit_in_instruction": True,
                "concrete_tool_command_or_artifact_operation_preferred_when_profile_supports_it": True,
                "exact_constants_may_use_model_static_knowledge": True,
                "request_collect_or_verify_placeholder_for_missing_detail_forbidden": True,
                "proposal_must_not_call_tools_files_or_network": True,
                "runtime_action_must_use_preinstalled_local_resources_only": True,
                "runtime_network_fetch_or_package_install_forbidden": True,
                "action_quality_enforcement": "prompt_and_audit_only",
                "action_quality_audit_must_not_reject_retry_or_affect_selection": True,
                "validation_outcomes_for_action_design_forbidden": True,
                "verifier_content_for_action_design_forbidden": True,
            }
        )
    family_slot_contract = (capabilities or {}).get("family_slot_contract")
    family_slot_policy = (
        str(family_slot_contract.get("policy") or "")
        if isinstance(family_slot_contract, Mapping)
        else ""
    )
    if (
        isinstance(family_slot_contract, Mapping)
        and family_slot_policy
        in FAMILY_SLOT_PROPOSAL_FORMATION_POLICY_VERSIONS
    ):
        portable_recipe_policy = family_slot_contract.get(
            "portable_recipe_policy"
        )
        reusable_preferred_primitive_count = int(
            portable_recipe_policy.get(
                "reusable_preferred_primitive_count",
                0,
            )
            if isinstance(portable_recipe_policy, Mapping)
            else 0
        )
        target_failure_family = str(
            family_slot_contract.get("target_failure_family") or ""
        )
        family_constraints: dict[str, Any] = {
                "proposal_targets_exactly_one_train_failure_family": True,
                "proposal_target_family_field": "target_failure_family",
                "proposal_target_failure_family": target_failure_family,
                "trigger_must_include_exact_target_family_predicate": {
                    "key": "family",
                    "op": "eq",
                    "value": target_failure_family,
                },
                "anti_trigger_must_not_block_target_family": True,
                "proposal_other_failure_families_forbidden": True,
                "proposal_all_train_success_controls_are_negative_controls": True,
                "portable_recipe_literal_requires_two_same_family_train_rows": True,
                "portable_recipe_without_repeated_literal_extracts_from_current_task_or_artifact": True,
                "successful_allowlisted_profile_primitives_preferred": True,
                "reusable_preferred_primitive_count": (
                    reusable_preferred_primitive_count
                ),
                "reusable_preferred_primitive_requires_action_binding": (
                    reusable_preferred_primitive_count > 0
                ),
                "exact_constant_alone_is_insufficient": True,
                "required_portable_delta_choice": [
                    "concrete_local_tool_command",
                    "artifact_internal_manipulation",
                    "current_task_or_artifact_extraction",
                ],
                "hardcoded_train_instance_path_field_or_hex_requires_two_same_family_identical_evidence_rows": True,
                "failed_profile_primitives_must_be_avoided": True,
                "family_slot_validation_outcomes_forbidden": True,
                "family_slot_verifier_content_forbidden": True,
                "family_slot_test_content_forbidden": True,
                "family_slot_response_is_singular_transport_contract": True,
                "family_slot_response_diversity_rejection_forbidden": True,
                "family_slot_diversity_retry_forbidden": True,
        }
        if family_slot_policy == FAMILY_SLOT_PROPOSAL_FORMATION_POLICY_V2:
            recommended_artifact = (
                portable_recipe_policy.get("recommended_artifact")
                if isinstance(portable_recipe_policy, Mapping)
                else None
            )
            workflow_blueprint = (
                portable_recipe_policy.get(
                    "required_artifact_workflow_blueprint"
                )
                if isinstance(portable_recipe_policy, Mapping)
                else None
            )
            family_constraints.update(
                {
                    "trigger_schema_must_equal_exact_target_family_only": {
                        "all_of": [
                            {
                                "key": "family",
                                "op": "eq",
                                "value": target_failure_family,
                            }
                        ],
                        "any_of": [],
                        "none_of": [],
                    },
                    "anti_trigger_schema_must_equal_empty": {
                        "all_of": [],
                        "any_of": [],
                        "none_of": [],
                    },
                    "recommended_artifact": recommended_artifact,
                    "recommended_artifact_value_must_be_mentioned_exactly": (
                        str(recommended_artifact.get("value") or "")
                        if isinstance(recommended_artifact, Mapping)
                        else ""
                    ),
                    "required_artifact_workflow_order": [
                        "read",
                        "parse",
                        "update",
                        "serialize",
                        "write_back",
                    ],
                    "required_artifact_workflow_blueprint": (
                        workflow_blueprint
                    ),
                    "failed_profile_primitive_values_disclosed_to_model": False,
                    "failed_profile_primitive_summary_only": True,
                }
            )
            if prompt_directive_backend:
                constraints[
                    "prompt_directive_action_value_grounding_source"
                ] = (
                    "TRAIN residual task requirement plus the derived preferred "
                    "primitive summary and exact recommended-artifact blueprint"
                )
        constraints.update(family_constraints)
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
    action_quality_contract = (capabilities or {}).get(
        "action_quality_contract"
    )
    material_action_delta_prompt = bool(
        isinstance(action_quality_contract, Mapping)
        and action_quality_contract.get("policy")
        == TRAIN_ACTION_DESIGN_POLICY_VERSION
    )
    family_slot_contract = (capabilities or {}).get("family_slot_contract")
    family_slot_policy = (
        str(family_slot_contract.get("policy") or "")
        if isinstance(family_slot_contract, Mapping)
        else ""
    )
    family_slot_prompt = bool(
        isinstance(family_slot_contract, Mapping)
        and family_slot_policy
        in FAMILY_SLOT_PROPOSAL_FORMATION_POLICY_VERSIONS
    )
    family_slot_v2_prompt = (
        family_slot_policy == FAMILY_SLOT_PROPOSAL_FORMATION_POLICY_V2
    )
    portable_recipe_policy = (
        family_slot_contract.get("portable_recipe_policy")
        if family_slot_prompt
        else None
    )
    reusable_preferred_primitive_count = int(
        portable_recipe_policy.get(
            "reusable_preferred_primitive_count",
            0,
        )
        if isinstance(portable_recipe_policy, Mapping)
        else 0
    )
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
    target_failure_family = (
        str(family_slot_contract.get("target_failure_family") or "")
        if family_slot_prompt
        else ""
    )
    trigger_schema = (
        {
            "all_of": [
                {
                    "key": "family",
                    "op": "eq",
                    "value": target_failure_family,
                }
            ],
            "any_of": [],
            "none_of": [],
        }
        if family_slot_v2_prompt
        else {"all_of": [predicate], "any_of": [], "none_of": []}
    )
    anti_trigger_schema = (
        {"all_of": [], "any_of": [], "none_of": []}
        if family_slot_v2_prompt
        else {"all_of": [], "any_of": [predicate], "none_of": []}
    )
    recommended_artifact = (
        portable_recipe_policy.get("recommended_artifact")
        if isinstance(portable_recipe_policy, Mapping)
        else None
    )
    recommended_artifact_value = (
        str(recommended_artifact.get("value") or "")
        if isinstance(recommended_artifact, Mapping)
        else ""
    )
    required_artifact_workflow_blueprint = (
        str(
            portable_recipe_policy.get(
                "required_artifact_workflow_blueprint"
            )
            or ""
        )
        if isinstance(portable_recipe_policy, Mapping)
        else ""
    )
    return {
        "id": "stable descriptive ID",
        "kind": "task|policy|evaluator",
        "statement": "falsifiable hypothesis",
        "trigger": trigger_schema,
        "anti_trigger": anti_trigger_schema,
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
                    (
                        (
                            (
                                "complete imperative task-local sentence that "
                                "mentions the recommended artifact value exactly "
                                f"as {recommended_artifact_value!r} "
                                "and implements this exact ordered read -> parse -> "
                                "update -> serialize -> write-back blueprint: "
                                f"{required_artifact_workflow_blueprint} Never use "
                                "validation outcomes, verifier content, test content, "
                                "network fetches, package installation, or a failed "
                                "profile primitive."
                            )
                            if family_slot_v2_prompt
                            else
                            (
                            "complete imperative task-local sentence for the target "
                            "family. It must provide at least one portable operational "
                            "delta: a concrete preinstalled local tool command, an "
                            "artifact-internal manipulation, or extraction of a needed "
                            "value from the current task/artifact; an exact constant "
                            "alone is insufficient. "
                            + (
                                "At least one action must bind a canonical preferred "
                                "profile primitive because reusable preferred evidence "
                                f"count is {reusable_preferred_primitive_count}. "
                                if reusable_preferred_primitive_count > 0
                                else ""
                            )
                            + "Avoid every failed profile primitive. Do not hardcode a "
                            "TRAIN-instance path, field, or HEX literal unless the same "
                            "value appears in at least two target-family TRAIN evidence "
                            "rows; otherwise extract it from the current task/artifact. "
                            "Never use validation outcomes, verifier content, test "
                            "content, network fetches, or package installation."
                            )
                        )
                        if family_slot_prompt
                        else (
                            "complete imperative task-local sentence that directly states "
                            "at least one material delta absent from the baseline task "
                            "instruction: an exact constant/mapping, a concrete local tool "
                            "command with flags and path, or an artifact-internal API/field "
                            "operation. Bind to the referenced TRAIN runtime/action profile "
                            "when available; model static knowledge may supply an exact "
                            "constant. Never substitute request/collect/verify for the "
                            "missing detail, and never claim later access to tools, files, "
                            "network, verifier, or validation outcomes. Use only preinstalled "
                            "local runtime resources; never prescribe network fetches or "
                            "package installation."
                        )
                        if material_action_delta_prompt
                        else (
                            "complete imperative task-local sentence grounded in TRAIN "
                            "residual context.task_instruction; never an enum-only value, "
                            "mapping/mode/check label, or preserve_baseline claim"
                        )
                    )
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


def _action_delta_audit_row(
    program: HypothesisProgram,
    *,
    residuals: Sequence[ResidualExample],
    profiles: Mapping[str, Any],
) -> dict[str, Any]:
    matched_failures: list[ResidualExample] = []
    for residual in residuals:
        if residual.split is not SplitName.TRAIN or residual.baseline_success:
            continue
        try:
            matched = program.matches(residual.features)
        except (TypeError, ValueError, OverflowError):
            matched = False
        if matched:
            matched_failures.append(residual)
    instructions = [
        str(row.context.get("task_instruction") or "")
        for row in matched_failures
    ]
    instruction_tokens = _audit_tokens(" ".join(instructions))
    profile_hashes = sorted(
        {
            str(row.context.get("action_context_profile_hash") or "")
            for row in matched_failures
            if row.context.get("action_context_profile_hash")
        }
    )
    selected_profiles = [
        profiles[profile_hash]
        for profile_hash in profile_hashes
        if isinstance(profiles.get(profile_hash), Mapping)
    ]
    environment_terms, baseline_tokens = _action_profile_terms(selected_profiles)
    action_rows: list[dict[str, Any]] = []
    observed_delta_kinds: set[str] = set()
    new_primitives: set[str] = set()
    for action in program.action_graph:
        value_text = (
            action.value
            if isinstance(action.value, str)
            else json.dumps(action.value, sort_keys=True, ensure_ascii=True)
        )
        action_text = f"{action.target} {value_text}".strip()
        lowered = action_text.lower()
        tokens = _audit_tokens(action_text)
        novel_tokens = tokens - instruction_tokens
        overlap_ratio = (
            len(tokens & instruction_tokens) / len(tokens) if tokens else 1.0
        )
        mentioned_environment_terms = {
            term
            for term in environment_terms
            if term in lowered
        }
        mentioned_new_primitives = {
            term
            for term in mentioned_environment_terms
            if term not in baseline_tokens and term not in instruction_tokens
        }
        new_primitives.update(mentioned_new_primitives)
        delta_kinds: set[str] = set()
        if re.search(r"#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?\b", action_text):
            delta_kinds.add("exact_constant_or_mapping")
        if (
            re.search(r"--[a-zA-Z0-9][a-zA-Z0-9-]*", action_text)
            or any(
                term in lowered
                for term in mentioned_environment_terms
                if not term.startswith("/root/")
            )
        ):
            delta_kinds.add("concrete_local_tool_command")
        if (
            re.search(r"\b[A-Za-z_][A-Za-z0-9_.]*\([^\n)]*\)", action_text)
            or re.search(
                r"\b(?:get_fields|update_page_form_field_values|dictwriter|pdfreader|pdfwriter|acroform|checkbox|round[- ]trip)\b",
                lowered,
            )
        ):
            delta_kinds.add("artifact_internal_manipulation")
        vague_placeholder = bool(
            re.search(
                r"\b(?:request|collect|look up|lookup|obtain|translate|determine|verify)\b",
                lowered,
            )
            and not delta_kinds
        )
        restatement_only = bool(
            not delta_kinds
            and not mentioned_new_primitives
            and (overlap_ratio >= 0.70 or len(novel_tokens) <= 3)
        )
        observed_delta_kinds.update(delta_kinds)
        action_rows.append(
            {
                "action_id_hash": stable_hash({"action_id": action.id}),
                "observed_delta_kinds": sorted(delta_kinds),
                "instruction_token_overlap_milli": round(
                    overlap_ratio * 1000
                ),
                "environment_binding_count": len(
                    mentioned_environment_terms
                ),
                "new_environment_primitive_count": len(
                    mentioned_new_primitives
                ),
                "vague_placeholder": vague_placeholder,
                "instruction_restatement_only": restatement_only,
            }
        )
    return {
        "hypothesis_id_hash": stable_hash({"hypothesis_id": program.id}),
        "hypothesis_hash": program.payload_hash,
        "matched_failure_count": len(matched_failures),
        "action_profile_count": len(selected_profiles),
        "action_count": len(action_rows),
        "observed_delta_kinds": sorted(observed_delta_kinds),
        "new_environment_primitive_count": len(new_primitives),
        "vague_placeholder_action_count": sum(
            bool(row["vague_placeholder"]) for row in action_rows
        ),
        "instruction_restatement_only_action_count": sum(
            bool(row["instruction_restatement_only"]) for row in action_rows
        ),
        "restatement_risk": not observed_delta_kinds,
        "action_audits": action_rows,
    }


def _action_profile_terms(
    profiles: Sequence[Mapping[str, Any]],
) -> tuple[set[str], set[str]]:
    environment_terms: set[str] = set()
    baseline_tokens: set[str] = set()
    for profile in profiles:
        environment = profile.get("runtime_environment")
        if isinstance(environment, Mapping):
            for key in (
                "declared_os_packages",
                "declared_python_packages",
                "declared_task_local_paths",
                "copied_task_files",
                "environment_source_files",
            ):
                values = environment.get(key)
                if not isinstance(values, (list, tuple)):
                    continue
                for value in values:
                    term = str(value).strip().lower()
                    if "==" in term:
                        term = term.split("==", 1)[0]
                    if len(term) >= 3:
                        environment_terms.add(term)
        trace = profile.get("baseline_action_trace")
        if not isinstance(trace, Mapping):
            continue
        command_rows = trace.get("command_signatures")
        if not isinstance(command_rows, list):
            continue
        for command_row in command_rows:
            if not isinstance(command_row, Mapping):
                continue
            executable = str(
                command_row.get("executable_basename") or ""
            ).strip()
            if executable:
                baseline_tokens.update(_audit_tokens(executable))
            for key in ("safe_flags", "task_local_paths"):
                values = command_row.get(key)
                if not isinstance(values, (list, tuple)):
                    continue
                for value in values:
                    baseline_tokens.update(_audit_tokens(str(value)))
    return environment_terms, baseline_tokens


def _audit_tokens(value: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_./#-]+", value)
        if len(token) >= 3
    }
