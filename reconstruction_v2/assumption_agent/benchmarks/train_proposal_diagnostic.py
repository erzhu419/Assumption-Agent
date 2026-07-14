from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib

from ..archive import PolicyArchive
from ..events import Event, EventSink, JsonlEventSink, NullEventSink
from ..evolution import (
    EvolutionKernel,
    PROPOSAL_FORMATION_POLICY_V2,
)
from ..models import (
    HypothesisKind,
    HypothesisProgram,
    ResidualExample,
    SplitName,
    stable_hash,
)
from ..proposer import (
    FAMILY_SLOT_PROPOSAL_FORMATION_POLICY_V2,
    HypothesisProposalCallError,
    TRAIN_ACTION_DESIGN_INTERNAL_CONTEXT_KEY,
    TRAIN_ACTION_DESIGN_POLICY_VERSION,
    StructuredHypothesisProposer,
    _action_delta_audit_row,
)
from ..splits import AccessPhase, BenchmarkItem, SplitAccessGuard, SplitManifest
from ..validation import ValidationContext, build_trigger_feature_catalog
from .skilllearn_compiler import (
    SKILLLEARN_ALLOWED_ACTION_OPERATIONS,
    SKILL_ACTION_LOWERING_VERSION,
)
from .skilllearnbench import SkillLearnBenchAdapter


TRAIN_PROPOSAL_DIAGNOSTIC_VERSION = (
    "v315_train_evidence_v317_family_slot_proposal_only_v2"
)
TRAIN_SOURCE_RECEIPT_VERSION = "v315_train_source_provenance_receipt_v1"
SOURCE_PROTOCOL_VERSION = "3.15.0"
TARGET_PROTOCOL_VERSION = "3.17.0"
TARGET_PROPOSAL_FORMATION_POLICY_VERSION = (
    FAMILY_SLOT_PROPOSAL_FORMATION_POLICY_V2
)
EXPECTED_TRAIN_OBSERVATIONS = 38
EXPECTED_TRAIN_FAILURES = 32
EXPECTED_TRAIN_SUCCESSES = 6
EXPECTED_ACTION_PROFILE_COUNT = 31
REQUIRED_SLOT_COUNT = 3
MINIMUM_TARGET_SUPPORT = 2
_ACCEPTANCE_KEYS = frozenset(
    {
        "root_count_passed",
        "distinct_single_family_signatures_passed",
        "minimum_target_support_passed",
        "target_anti_trigger_self_block_absent",
        "profile_environment_binding_passed",
        "profile_grounded_executable_delta_passed",
        "restatement_only_absent",
        "failed_profile_primitive_avoidance_passed",
        "schema_validation_passed",
    }
)

_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9._+-]+")
_COMMON_BOUNDARY_FLAGS = {
    "backend_accessed": False,
    "evaluator_accessed": False,
    "validation_accessed": False,
    "validation_features_used": False,
    "validation_outcomes_used": False,
    "verifier_content_used": False,
    "test_content_accessed": False,
    "test_content_used": False,
    "sealed_test_accessed": False,
    "promotion_gate_evaluated": False,
    "secret_value_persisted": False,
    "raw_content_persisted": False,
}

_PERSISTED_EVENT_FIELDS: Mapping[str, frozenset[str]] = {
    "model_provider_chain_built": frozenset(
        {
            "requested_providers",
            "active_providers",
            "unavailable_providers",
            "model",
            "provider_chain_hash",
        }
    ),
    "model_provider_attempted": frozenset(
        {
            "provider",
            "provider_position",
            "provider_count",
            "provider_chain_hash",
            "request_hash",
            "model",
        }
    ),
    "model_provider_failed": frozenset(
        {
            "provider",
            "provider_chain_hash",
            "request_hash",
            "model",
            "error_type",
            "fallback_available",
            "circuit_opened",
            "raw_error_persisted",
        }
    ),
    "model_provider_selected": frozenset(
        {
            "provider",
            "provider_chain_hash",
            "request_hash",
            "response_hash",
            "model",
            "failover_used",
            "prior_failure_count",
        }
    ),
    "model_provider_chain_exhausted": frozenset(
        {
            "provider_chain_hash",
            "request_hash",
            "model",
            "failure_types",
            "raw_error_persisted",
        }
    ),
    "model_attempt_started": frozenset(
        {
            "request_hash",
            "attempt",
            "attempt_limit",
            "model",
            "timeout_seconds",
            "endpoint_hash",
        }
    ),
    "model_attempt_succeeded": frozenset(
        {
            "request_hash",
            "response_hash",
            "attempt",
            "elapsed_seconds",
            "model",
        }
    ),
    "model_attempt_failed": frozenset(
        {
            "request_hash",
            "attempt",
            "elapsed_seconds",
            "error_type",
            "http_status",
            "retryable",
            "model",
        }
    ),
    "split_access_authorized": frozenset(
        {"item_id_hash", "phase", "split", "archive_frozen"}
    ),
    "proposal_family_slot_plan_created": frozenset(
        {
            "policy",
            "slot_count",
            "distinct_target_family_count",
            "available_train_failure_family_count",
            "train_success_control_count",
            "slot_plan_hash",
            "slots",
        }
    ),
    "hypothesis_proposal_requested": frozenset(
        {
            "request_kind",
            "request_hash",
            "repair_request_scope_policy",
            "residual_count",
            "evaluator_epoch",
        }
    ),
    "hypothesis_proposal_response_rejected": frozenset(
        {
            "request_kind",
            "request_hash",
            "response_hash",
            "failure_phase",
            "error_type",
            "candidate_local_failure",
            "expected_field",
            "top_level_type",
            "top_level_key_count",
            "top_level_key_set_hash",
            "expected_field_present",
            "expected_field_type",
            "expected_field_item_count",
            "expected_item_count",
            "failure_train_row_count",
            "distinct_activation_signature_count",
            "response_contract_policy",
            "consumed_row_present",
            "consumed_row_index",
            "consumed_row_type",
            "parse_error_type",
            "raw_error_persisted",
        }
    ),
    "hypothesis_proposal_model_call_failed": frozenset(
        {
            "request_kind",
            "request_hash",
            "error_type",
            "candidate_local_failure",
            "raw_error_persisted",
        }
    ),
    "proposal_action_delta_audited": frozenset(
        {
            "policy",
            "request_kind",
            "candidate_count",
            "candidate_audits",
            "candidate_with_material_delta_count",
            "candidate_with_restatement_risk_count",
            "response_rejected",
            "proposal_retry_requested",
            "recursive_repair_requested_by_audit",
            "candidate_selection_affected",
            "promotion_gate_affected",
        }
    ),
    "proposal_action_delta_audit_failed": frozenset(
        {
            "policy",
            "request_kind",
            "error_type",
            "response_rejected",
            "proposal_retry_requested",
            "candidate_selection_affected",
            "promotion_gate_affected",
            "raw_error_persisted",
        }
    ),
    "hypothesis_proposed": frozenset(
        {
            "hypothesis_id_hash",
            "hypothesis_hash",
            "kind",
            "transition_count",
            "validation_issues",
            "evaluator_epoch",
        }
    ),
    "root_proposal_evidence_recorded": frozenset(
        {
            "policy",
            "request_hash",
            "source_trace_id",
            "program_count",
            "program_set_hash",
            "new_proposal_model_executions",
            "evaluator_epoch",
            "sealed_test_accessed",
        }
    ),
    "root_proposal_evidence_replayed": frozenset(
        {
            "policy",
            "request_hash",
            "source_trace_id",
            "target_trace_id",
            "program_count",
            "program_set_hash",
            "request_identical",
            "new_proposal_model_executions",
            "evaluator_epoch",
            "sealed_test_accessed",
        }
    ),
    "proposal_family_slot_usage_recorded": frozenset(
        {
            "policy",
            "source",
            "proposal_set_hash",
            "usage_identity_hash",
            "candidate_count",
            "requested_target_count",
            "distinct_requested_target_count",
            "requested_target_set_hash",
            "actual_matched_count",
            "distinct_actual_matched_family_count",
            "actual_matched_family_set_hash",
            "proposal_set_replayed",
            "family_use_updated",
            "new_family_use_count",
            "family_use_count_state_hash",
        }
    ),
    "proposal_family_slot_usage_replayed": frozenset(
        {
            "policy",
            "source",
            "proposal_set_hash",
            "usage_identity_hash",
            "candidate_count",
            "requested_target_count",
            "distinct_requested_target_count",
            "requested_target_set_hash",
            "actual_matched_count",
            "distinct_actual_matched_family_count",
            "actual_matched_family_set_hash",
            "proposal_set_replayed",
            "family_use_updated",
            "new_family_use_count",
            "family_use_count_state_hash",
        }
    ),
    "proposal_family_slot_completed": frozenset(
        {
            "policy",
            "slot_id",
            "slot_plan_hash",
            "target_family_hash",
            "profile_evidence_hash",
            "preferred_primitive_count",
            "preferred_primitive_set_hash",
            "failed_primitive_count",
            "failed_primitive_set_hash",
            "candidate_hash",
            "matched_family_hash",
            "candidate_matched_target_support",
            "candidate_matched_target",
            "matched_family_count",
            "profile_binding_count",
            "failed_profile_binding_count",
            "portable_delta_kinds",
            "response_rejected_by_diversity",
            "proposal_retry_by_diversity",
        }
    ),
    "proposal_family_slots_completed": frozenset(
        {
            "policy",
            "slot_plan_hash",
            "candidate_count",
            "candidate_set_hash",
            "matched_family_count",
            "distinct_matched_family_count",
        }
    ),
    "train_proposal_diagnostic_started": frozenset(
        {
            "diagnostic_policy",
            "manifest_hash",
            "target_protocol_hash",
            "proposal_only",
            "source_agent_trials_reexecuted",
        }
    ),
    "train_proposal_source_evidence_reconstructed": frozenset(
        {
            "diagnostic_policy",
            "manifest_hash",
            "source_train_receipt_hash",
            "source_protocol_hash",
            "source_protocol_lock_hash",
            "source_observation_set_hash",
            "source_row_set_hash",
            "train_observation_count",
            "train_failure_count",
            "train_success_control_count",
            "action_profile_count",
            "action_profile_set_hash",
            "success_control_context_empty",
        }
    ),
    "train_proposal_slot_plan_frozen": frozenset(
        {
            "diagnostic_policy",
            "proposal_formation_policy",
            "production_evolution_kernel_used",
            "slot_plan_hash",
            "slot_count",
            "target_family_hashes",
            "target_failure_counts",
            "preferred_primitive_set_hashes",
            "failed_primitive_set_hashes",
        }
    ),
    "train_proposal_slot_completed": frozenset(
        {
            "diagnostic_policy",
            "proposal_formation_policy",
            "slot_plan_hash",
            "slot_index",
            "target_family_hash",
            "hypothesis_id_hash",
            "hypothesis_hash",
            "activation_signature_hash",
            "matched_target_support",
            "matched_failure_family_count",
            "profile_environment_binding_count",
            "profile_grounded_delta_kinds",
            "restatement_only_action_count",
        }
    ),
    "train_proposal_diagnostic_completed": frozenset(
        {
            "diagnostic_policy",
            "target_protocol_hash",
            "slot_plan_hash",
            "proposal_count",
            "proposal_set_hash",
            "distinct_activation_signature_count",
            "diagnostic_passed",
            "acceptance",
            "failure_blocks_future_trial_spend_only",
            "promotion_gate_or_score",
        }
    ),
}

_PERSISTED_RAW_KEYS = frozenset(
    {
        "hypothesis_id",
        "candidate_hypothesis_ids",
        "target_family",
        "target_failure_family",
        "statement",
        "action_graph",
        "task_instruction",
        "residuals",
        "train_action_design_profiles",
        "response",
    }
)
@dataclass(frozen=True)
class ReconstructedTrainEvidence:
    manifest: SplitManifest
    residuals: tuple[ResidualExample, ...]
    action_profiles: Mapping[str, Mapping[str, Any]]
    source_train_receipt_hash: str
    source_protocol_hash: str
    source_protocol_lock_hash: str
    source_observation_set_hash: str
    source_row_set_hash: str
    action_profile_count: int
    action_profile_set_hash: str
    source_checkpoint_hash: str
    source_model: str

    @property
    def failures(self) -> tuple[ResidualExample, ...]:
        return tuple(row for row in self.residuals if not row.baseline_success)

    @property
    def success_controls(self) -> tuple[ResidualExample, ...]:
        return tuple(row for row in self.residuals if row.baseline_success)


@dataclass(frozen=True)
class ProfilePrimitive:
    kind: str
    value: str
    train_failure_evidence_count: int

    @property
    def reusable(self) -> bool:
        return self.train_failure_evidence_count >= MINIMUM_TARGET_SUPPORT

    @property
    def primitive_hash(self) -> str:
        return stable_hash(self.model_payload())

    def model_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "train_failure_evidence_count": (
                self.train_failure_evidence_count
            ),
            "reusable_across_same_family_failures": self.reusable,
        }

    def safe_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "train_failure_evidence_count": (
                self.train_failure_evidence_count
            ),
            "reusable_across_same_family_failures": self.reusable,
            "primitive_hash": self.primitive_hash,
        }


@dataclass(frozen=True)
class FamilyProposalSlot:
    index: int
    target_family: str
    target_failure_count: int
    target_profile_hashes: tuple[str, ...]
    profile_evidence_hash: str
    preferred_primitives: tuple[ProfilePrimitive, ...]
    failed_primitives: tuple[ProfilePrimitive, ...]
    prior_use_count: int = 0
    declared_failed_primitive_count: int | None = None
    declared_failed_primitive_set_hash: str | None = None

    @property
    def target_family_hash(self) -> str:
        return stable_hash({"family": self.target_family})

    @property
    def preferred_primitive_set_hash(self) -> str:
        return stable_hash(
            {
                "primitives": [
                    row.model_payload() for row in self.preferred_primitives
                ]
            }
        )

    @property
    def failed_primitive_set_hash(self) -> str:
        if self.declared_failed_primitive_set_hash is not None:
            return self.declared_failed_primitive_set_hash
        return stable_hash(
            {
                "primitives": [
                    row.model_payload() for row in self.failed_primitives
                ]
            }
        )

    @property
    def failed_primitive_count(self) -> int:
        if self.declared_failed_primitive_count is not None:
            return self.declared_failed_primitive_count
        return len(self.failed_primitives)

    @property
    def reusable_preferred_primitive_count(self) -> int:
        return sum(row.reusable for row in self.preferred_primitives)

    def formation_payload(self) -> dict[str, Any]:
        """Return the exact V3.17 production slot-plan payload.

        This payload is used only for the plan hash and model contract.  The
        report uses :meth:`safe_payload`, which never persists primitive values.
        """

        preferred = [row.model_payload() for row in self.preferred_primitives]
        slot_number = self.index + 1
        return {
            "slot_id": f"train-family-slot-{slot_number}",
            "target_family": self.target_family,
            "target_family_hash": self.target_family_hash,
            "target_failure_support_count": self.target_failure_count,
            "prior_family_use_count": self.prior_use_count,
            "profile_reference_count": len(self.target_profile_hashes),
            "profile_evidence_hash": self.profile_evidence_hash,
            "preferred_primitive_count": len(preferred),
            "preferred_primitive_set_hash": self.preferred_primitive_set_hash,
            "reusable_preferred_primitive_count": (
                self.reusable_preferred_primitive_count
            ),
            "failed_primitive_count": self.failed_primitive_count,
            "failed_primitive_set_hash": self.failed_primitive_set_hash,
            "raw_content_persisted": False,
        }

    def safe_payload(self) -> dict[str, Any]:
        return {
            "slot_index": self.index,
            "target_family_hash": self.target_family_hash,
            "target_failure_count": self.target_failure_count,
            "target_profile_count": len(self.target_profile_hashes),
            "profile_evidence_hash": self.profile_evidence_hash,
            "target_profile_set_hash": stable_hash(
                {"profile_hashes": list(self.target_profile_hashes)}
            ),
            "preferred_primitive_count": len(self.preferred_primitives),
            "preferred_primitive_set_hash": self.preferred_primitive_set_hash,
            "reusable_preferred_primitive_count": (
                self.reusable_preferred_primitive_count
            ),
            "failed_primitive_count": self.failed_primitive_count,
            "failed_primitive_set_hash": self.failed_primitive_set_hash,
        }


class _DiagnosticEventSink:
    """Keep raw audit events in memory and persist only allowlisted metadata."""

    def __init__(self, delegate: EventSink) -> None:
        self.delegate = delegate
        # Production request/plan binding needs the raw family name during this
        # process only.  ``persisted_events`` and the delegate never receive it.
        self.events: list[dict[str, Any]] = []
        self.persisted_events: list[dict[str, Any]] = []

    def emit(self, event: Event) -> None:
        payload = dict(event.payload)
        for key, expected in _COMMON_BOUNDARY_FLAGS.items():
            if key in payload and payload[key] is not expected:
                raise PermissionError(
                    f"proposal-only event attempted forbidden boundary: {key}"
                )
        enriched = Event(
            event=event.event,
            stage=event.stage,
            trace_id=event.trace_id,
            payload={**payload, **_COMMON_BOUNDARY_FLAGS},
        )
        persisted = Event(
            event=event.event,
            stage=event.stage,
            trace_id=event.trace_id,
            payload=_sanitize_persisted_event_payload(
                event.event,
                enriched.payload,
            ),
        )
        self.events.append(enriched.to_dict())
        self.persisted_events.append(persisted.to_dict())
        self.delegate.emit(persisted)


def _sanitize_persisted_event_payload(
    event: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    allowed = _PERSISTED_EVENT_FIELDS.get(event)
    if allowed is None:
        raise PermissionError(
            f"proposal diagnostic attempted to persist unknown event: {event}"
        )
    source = dict(payload)
    if event == "hypothesis_proposed":
        raw_id = source.pop("hypothesis_id", None)
        if raw_id is not None:
            source["hypothesis_id_hash"] = stable_hash(
                {"hypothesis_id": str(raw_id)}
            )
    elif event == "proposal_family_slot_completed":
        raw_family = source.pop("target_family", None)
        if raw_family is not None:
            expected_hash = stable_hash({"family": str(raw_family)})
            if source.get("target_family_hash") != expected_hash:
                raise PermissionError(
                    "proposal family completion target hash mismatch"
                )

    globally_allowed = set(_COMMON_BOUNDARY_FLAGS)
    unexpected = set(source) - set(allowed) - globally_allowed
    if unexpected:
        raise PermissionError(
            "proposal diagnostic event contains non-allowlisted fields: "
            f"{event}:{sorted(unexpected)}"
        )
    safe = {
        key: source[key]
        for key in allowed
        if key in source
    }
    if event == "proposal_family_slot_plan_created":
        rows = safe.get("slots")
        if not isinstance(rows, list):
            raise PermissionError("proposal family slot plan rows are malformed")
        safe["slots"] = [
            _sanitize_persisted_slot_plan_row(row) for row in rows
        ]
    safe.update(_COMMON_BOUNDARY_FLAGS)
    _assert_no_persisted_raw_keys(safe)
    _assert_persisted_hash_fields(safe)
    return safe


def _sanitize_persisted_slot_plan_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise PermissionError("proposal family slot plan row is malformed")
    source = dict(row)
    raw_family = source.pop("target_family", None)
    if raw_family is not None:
        expected_hash = stable_hash({"family": str(raw_family)})
        if source.get("target_family_hash") != expected_hash:
            raise PermissionError("proposal family slot target hash mismatch")
    allowed = {
        "slot_id",
        "target_family_hash",
        "target_failure_support_count",
        "prior_family_use_count",
        "profile_reference_count",
        "profile_evidence_hash",
        "preferred_primitive_count",
        "preferred_primitive_set_hash",
        "reusable_preferred_primitive_count",
        "failed_primitive_count",
        "failed_primitive_set_hash",
        "raw_content_persisted",
    }
    unexpected = set(source) - allowed
    if unexpected:
        raise PermissionError(
            "proposal family slot plan contains non-allowlisted fields: "
            f"{sorted(unexpected)}"
        )
    safe = {key: source[key] for key in allowed if key in source}
    safe["raw_content_persisted"] = False
    _assert_no_persisted_raw_keys(safe)
    _assert_persisted_hash_fields(safe)
    return safe


def _assert_no_persisted_raw_keys(value: Any, *, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        forbidden = set(value) & set(_PERSISTED_RAW_KEYS)
        if forbidden:
            raise PermissionError(
                f"proposal diagnostic raw fields reached persistence at {path}: "
                f"{sorted(forbidden)}"
            )
        for key, child in value.items():
            _assert_no_persisted_raw_keys(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_persisted_raw_keys(child, path=f"{path}[{index}]")


def _assert_persisted_hash_fields(value: Any, *, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if (
                key.endswith("_hash")
                and not (key == "matched_family_hash" and child is None)
                and not _is_sha256(child)
            ):
                raise PermissionError(
                    f"proposal diagnostic has malformed persisted hash: {child_path}"
                )
            if key.endswith("_hashes"):
                if not isinstance(child, (list, tuple)) or any(
                    not _is_sha256(item) for item in child
                ):
                    raise PermissionError(
                        "proposal diagnostic has malformed persisted hash list: "
                        f"{child_path}"
                    )
            _assert_persisted_hash_fields(child, path=child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_persisted_hash_fields(child, path=f"{path}[{index}]")


class _RecordingProposalModel:
    """Keep production family-slot contracts in memory for post-call audit."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.requests: list[Mapping[str, Any]] = []

    def _record(self, payload: Mapping[str, Any]) -> None:
        capabilities = payload.get("capabilities")
        family_slot = (
            capabilities.get("family_slot_contract")
            if isinstance(capabilities, Mapping)
            else None
        )
        if not isinstance(family_slot, Mapping):
            raise PermissionError(
                "proposal diagnostic received a non-family-slot model request"
            )
        self.requests.append(payload)

    def complete(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self._record(payload)
        return self.delegate.complete(payload)

    def complete_with_trace(
        self,
        payload: Mapping[str, Any],
        *,
        trace_id: str,
    ) -> Mapping[str, Any]:
        self._record(payload)
        traced = getattr(self.delegate, "complete_with_trace", None)
        if callable(traced):
            return traced(payload, trace_id=trace_id)
        return self.delegate.complete(payload)


@dataclass(frozen=True)
class _ProposalOnlyPromotionSpec:
    metric: str


@dataclass(frozen=True)
class _ProposalOnlyPromotionGate:
    spec: _ProposalOnlyPromotionSpec


class _ForbiddenRuntimeDependency:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(
            f"proposal-only diagnostic touched forbidden runtime dependency: {name}"
        )


def build_v315_train_source_receipt(
    *,
    root: str | Path,
    manifest_path: str | Path,
    source_run_root: str | Path,
) -> dict[str, Any]:
    """Build the standalone receipt from TRAIN artifacts only."""

    manifest = SplitManifest.read(manifest_path)
    _, receipt = _reconstruct_v315_train_material(
        root=root,
        manifest=manifest,
        source_run_root=source_run_root,
        event_sink=NullEventSink(),
    )
    return receipt


def reconstruct_v315_train_evidence(
    *,
    root: str | Path,
    manifest: SplitManifest,
    source_run_root: str | Path,
    source_train_receipt: str | Path,
    event_sink: EventSink | None = None,
) -> ReconstructedTrainEvidence:
    """Rebuild and bind V3.15 evidence without opening development reports/events."""

    sink = event_sink or NullEventSink()
    evidence, computed_receipt = _reconstruct_v315_train_material(
        root=root,
        manifest=manifest,
        source_run_root=source_run_root,
        event_sink=sink,
    )
    receipt_path = Path(source_train_receipt).expanduser().resolve(strict=True)
    declared_receipt = _read_json_object(
        receipt_path,
        label="source TRAIN receipt",
    )
    _validate_source_train_receipt(
        declared_receipt,
        computed=computed_receipt,
    )
    _emit_boundary_event(
        sink,
        event="train_proposal_source_evidence_reconstructed",
        trace_id=evidence.source_train_receipt_hash[:24],
        payload={
            "diagnostic_policy": TRAIN_PROPOSAL_DIAGNOSTIC_VERSION,
            "manifest_hash": manifest.manifest_hash,
            "source_train_receipt_hash": evidence.source_train_receipt_hash,
            "source_protocol_hash": evidence.source_protocol_hash,
            "source_protocol_lock_hash": evidence.source_protocol_lock_hash,
            "source_observation_set_hash": evidence.source_observation_set_hash,
            "source_row_set_hash": evidence.source_row_set_hash,
            "train_observation_count": len(evidence.residuals),
            "train_failure_count": len(evidence.failures),
            "train_success_control_count": len(evidence.success_controls),
            "action_profile_count": evidence.action_profile_count,
            "action_profile_set_hash": evidence.action_profile_set_hash,
            "success_control_context_empty": all(
                not row.context for row in evidence.success_controls
            ),
        },
    )
    return evidence


def _reconstruct_v315_train_material(
    *,
    root: str | Path,
    manifest: SplitManifest,
    source_run_root: str | Path,
    event_sink: EventSink,
) -> tuple[ReconstructedTrainEvidence, dict[str, Any]]:
    benchmark_root = _resolve_directory(Path(root), label="benchmark root")
    source_root = _resolve_directory(
        Path(source_run_root), label="source run root"
    )
    source_identity = _source_protocol_lock_identity(
        source_root,
        manifest=manifest,
    )
    items = _public_train_items(benchmark_root, manifest)
    adapter = SkillLearnBenchAdapter(benchmark_root)
    adapter._items = dict(items)  # type: ignore[attr-defined]
    adapter._required_env_by_item = {  # type: ignore[attr-defined]
        item_id: () for item_id in items
    }
    guard = SplitAccessGuard(manifest, event_sink=event_sink)
    upstream_root = _resolve_directory(
        source_root
        / "development_recursive"
        / "upstream_trials"
        / "no_skill",
        label="source TRAIN upstream root",
    )
    from .skilllearn_lifecycle import _extract_train_action_trace_profile

    residuals: list[ResidualExample] = []
    action_profiles: dict[str, Mapping[str, Any]] = {}
    source_rows: list[dict[str, Any]] = []
    observation_hashes: list[str] = []
    profile_hashes: set[str] = set()
    success_count = 0
    failure_count = 0

    for item_id in manifest.train_ids:
        item = items[item_id]
        guard.authorize(item_id, AccessPhase.PROPOSAL)
        result_path, trace_path, receipt_path = _source_train_artifact_paths(
            upstream_root,
            family=item.family,
            item_id=item.id,
        )
        result = _read_json_object(result_path, label="source TRAIN result")
        action_receipt = _read_json_object(
            receipt_path, label="source action-budget receipt"
        )
        trace_hash = _validate_source_trial(
            result=result,
            receipt=action_receipt,
            trace_path=trace_path,
            family=item.family,
            item_id=item.id,
            expected_model=str(source_identity["model"]),
        )
        passed = result.get("passed")
        if not isinstance(passed, bool):
            raise PermissionError("source TRAIN result has a non-boolean outcome")
        result_hash = _sha256_file(result_path)
        action_receipt_hash = str(action_receipt["receipt_hash"])
        instruction = adapter.load_instruction(
            item_id,
            phase=AccessPhase.PROPOSAL,
            guard=guard,
        ).strip()
        runtime_environment = adapter.load_action_design_context(
            item_id,
            phase=AccessPhase.PROPOSAL,
            guard=guard,
        )
        instruction_hash = stable_hash({"instruction": instruction})
        environment_hash = stable_hash(runtime_environment)
        observation_descriptor = {
            "item_id_hash": stable_hash({"item_id": item.id}),
            "family_hash": stable_hash({"family": item.family}),
            "passed": passed,
            "result_hash": result_hash,
            "trace_hash": trace_hash,
            "action_receipt_hash": action_receipt_hash,
            "instruction_hash": instruction_hash,
            "environment_hash": environment_hash,
        }
        source_observation_hash = stable_hash(observation_descriptor)
        observation_hashes.append(source_observation_hash)
        transition_id = "transition_diag_" + source_observation_hash[:18]

        if passed:
            success_count += 1
            context: Mapping[str, Any] = {}
            failure_type = "baseline_success_control"
            evaluator_feedback: tuple[str, ...] = ()
        else:
            failure_count += 1
            action_profile = {
                "policy": TRAIN_ACTION_DESIGN_POLICY_VERSION,
                "runtime_environment": runtime_environment,
                "baseline_action_trace": _extract_train_action_trace_profile(
                    trace_path,
                    containment_root=upstream_root,
                ),
                "evidence_scope": "train_policy_off_nonoracle_only",
                "validation_outcomes_used": False,
                "verifier_content_used": False,
                "test_content_used": False,
            }
            profile_hash = stable_hash(action_profile)
            action_profiles.setdefault(profile_hash, action_profile)
            profile_hashes.add(profile_hash)
            context = {
                "task_instruction": instruction,
                "observed_metrics": {
                    "evaluation_valid": 1.0,
                    "task_success": 0.0,
                },
                "execution_signals": {
                    "total_tokens": int(
                        (action_receipt.get("token_usage") or {}).get(
                            "total_tokens", 0
                        )
                    ),
                    "steps": int(action_receipt.get("observed_steps") or 0),
                    "duration_seconds": 0.0,
                },
                "action_context_profile_hash": profile_hash,
                TRAIN_ACTION_DESIGN_INTERNAL_CONTEXT_KEY: action_profile,
            }
            failure_type = "external_task_verifier_failed"
            evaluator_feedback = (
                "The offline TRAIN verifier rejected the baseline outcome.",
                "Infer a concrete reusable corrective operator from the TRAIN "
                "task instruction; use complete imperative task-local steps and "
                "do not default to a generic completeness check.",
            )

        residual = ResidualExample(
            transition_id=transition_id,
            task_id=item.id,
            family=item.family,
            split=SplitName.TRAIN,
            features={**dict(item.features), "family": item.family},
            failure_type=failure_type,
            evaluator_feedback=evaluator_feedback,
            baseline_success=passed,
            context=context,
        )
        issues = residual.validate()
        if issues:
            raise PermissionError(
                f"reconstructed TRAIN residual violates isolation: {issues}"
            )
        if residual.baseline_success and residual.context:
            raise PermissionError("success-control context must remain empty")
        residuals.append(residual)
        source_rows.append(
            {
                **observation_descriptor,
                "source_observation_hash": source_observation_hash,
                "transition_id": transition_id,
            }
        )

    if len(residuals) != EXPECTED_TRAIN_OBSERVATIONS:
        raise PermissionError("source TRAIN artifact count is not 38")
    if failure_count != EXPECTED_TRAIN_FAILURES:
        raise PermissionError("source TRAIN failure count is not 32")
    if success_count != EXPECTED_TRAIN_SUCCESSES:
        raise PermissionError("source TRAIN success count is not 6")
    if len(profile_hashes) != EXPECTED_ACTION_PROFILE_COUNT:
        raise PermissionError("source TRAIN action-profile count is not 31")

    action_profile_set_hash = stable_hash(
        {"profile_hashes": sorted(profile_hashes)}
    )
    source_observation_set_hash = stable_hash(
        {"hashes": sorted(observation_hashes)}
    )
    source_row_set_hash = stable_hash(
        {
            "source_train_rows": sorted(
                source_rows,
                key=lambda row: str(row["item_id_hash"]),
            )
        }
    )
    receipt = _source_train_receipt_payload(
        manifest=manifest,
        source_identity=source_identity,
        source_observation_set_hash=source_observation_set_hash,
        source_row_set_hash=source_row_set_hash,
        action_profile_set_hash=action_profile_set_hash,
    )
    evidence = ReconstructedTrainEvidence(
        manifest=manifest,
        residuals=tuple(residuals),
        action_profiles=dict(sorted(action_profiles.items())),
        source_train_receipt_hash=str(receipt["receipt_hash"]),
        source_protocol_hash=str(source_identity["protocol_hash"]),
        source_protocol_lock_hash=str(source_identity["protocol_lock_hash"]),
        source_observation_set_hash=source_observation_set_hash,
        source_row_set_hash=source_row_set_hash,
        action_profile_count=EXPECTED_ACTION_PROFILE_COUNT,
        action_profile_set_hash=action_profile_set_hash,
        source_checkpoint_hash=str(receipt["source_checkpoint_hash"]),
        source_model=str(source_identity["model"]),
    )
    return evidence, receipt


def _source_protocol_lock_identity(
    source_root: Path,
    *,
    manifest: SplitManifest,
) -> dict[str, str]:
    lock_path = _contained_file(
        source_root / "protocol_lock.json",
        anchor=source_root,
        label="source protocol lock",
    )
    lock = _read_json_object(lock_path, label="source protocol lock")
    declared_lock_hash = lock.get("lock_hash")
    unlocked = dict(lock)
    unlocked.pop("lock_hash", None)
    if not _is_sha256(declared_lock_hash) or stable_hash(unlocked) != (
        declared_lock_hash
    ):
        raise PermissionError("source protocol lock hash mismatch")
    identity = {
        "protocol_id": str(lock.get("protocol_id") or ""),
        "protocol_hash": str(lock.get("protocol_hash") or ""),
        "protocol_lock_hash": str(declared_lock_hash),
        "manifest_hash": str(lock.get("primary_manifest_hash") or ""),
        "model": str(lock.get("model") or ""),
    }
    if "-v3.15-" not in identity["protocol_id"]:
        raise PermissionError("source protocol lock is not V3.15")
    for key in ("protocol_hash", "protocol_lock_hash", "manifest_hash"):
        if not _is_sha256(identity[key]):
            raise PermissionError(
                f"source protocol lock identity hash is malformed: {key}"
            )
    if identity["manifest_hash"] != manifest.manifest_hash:
        raise PermissionError("source protocol lock manifest hash mismatch")
    if not identity["model"]:
        raise PermissionError("source protocol lock model is missing")
    return identity


def _source_train_receipt_payload(
    *,
    manifest: SplitManifest,
    source_identity: Mapping[str, str],
    source_observation_set_hash: str,
    source_row_set_hash: str,
    action_profile_set_hash: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "receipt_version": TRAIN_SOURCE_RECEIPT_VERSION,
        "source_split": SplitName.TRAIN.value,
        "source_identity": dict(source_identity),
        "train_action_design_policy": TRAIN_ACTION_DESIGN_POLICY_VERSION,
        "train_item_set_hash": stable_hash(
            {"item_ids": sorted(manifest.train_ids)}
        ),
        "train_observation_count": EXPECTED_TRAIN_OBSERVATIONS,
        "train_failure_count": EXPECTED_TRAIN_FAILURES,
        "train_success_control_count": EXPECTED_TRAIN_SUCCESSES,
        "source_observation_set_hash": source_observation_set_hash,
        "source_row_set_hash": source_row_set_hash,
        "action_profile_count": EXPECTED_ACTION_PROFILE_COUNT,
        "action_profile_set_hash": action_profile_set_hash,
        "raw_content_persisted": False,
    }
    payload["source_checkpoint_hash"] = stable_hash(payload)
    payload["receipt_hash"] = stable_hash(payload)
    return payload


def _validate_source_train_receipt(
    receipt: Mapping[str, Any],
    *,
    computed: Mapping[str, Any],
) -> None:
    if set(receipt) != set(computed):
        raise PermissionError("source TRAIN receipt fields mismatch")
    identity = receipt.get("source_identity")
    expected_identity = computed.get("source_identity")
    if not isinstance(identity, Mapping) or not isinstance(
        expected_identity, Mapping
    ) or set(identity) != set(expected_identity):
        raise PermissionError("source TRAIN receipt identity fields mismatch")
    declared_hash = receipt.get("receipt_hash")
    unsigned = dict(receipt)
    unsigned.pop("receipt_hash", None)
    if not _is_sha256(declared_hash) or stable_hash(unsigned) != declared_hash:
        raise PermissionError("source TRAIN receipt hash mismatch")
    checkpoint = unsigned.pop("source_checkpoint_hash", None)
    if not _is_sha256(checkpoint) or stable_hash(unsigned) != checkpoint:
        raise PermissionError("source TRAIN receipt checkpoint hash mismatch")
    for key in (
        "source_identity",
        "train_observation_count",
        "train_failure_count",
        "train_success_control_count",
        "source_observation_set_hash",
        "source_row_set_hash",
        "action_profile_count",
        "action_profile_set_hash",
        "source_checkpoint_hash",
        "receipt_hash",
    ):
        if receipt.get(key) != computed.get(key):
            raise PermissionError(f"source TRAIN receipt drift: {key}")
    if dict(receipt) != dict(computed):
        raise PermissionError("source TRAIN receipt drift")


def run_train_proposal_diagnostic(
    *,
    root: str | Path,
    manifest_path: str | Path,
    source_run_root: str | Path,
    source_train_receipt: str | Path,
    protocol_path: str | Path,
    proposal_model: Any,
    event_sink: EventSink | None = None,
) -> dict[str, Any]:
    sink = (
        event_sink
        if isinstance(event_sink, _DiagnosticEventSink)
        else _DiagnosticEventSink(event_sink or NullEventSink())
    )
    manifest = SplitManifest.read(manifest_path)
    protocol, protocol_hash = _read_target_protocol(protocol_path)
    execution = protocol["execution"]
    evolution = protocol["evolution"]
    trace_id = stable_hash(
        {
            "diagnostic_policy": TRAIN_PROPOSAL_DIAGNOSTIC_VERSION,
            "manifest_hash": manifest.manifest_hash,
            "protocol_hash": protocol_hash,
        }
    )[:24]
    _emit_boundary_event(
        sink,
        event="train_proposal_diagnostic_started",
        trace_id=trace_id,
        payload={
            "diagnostic_policy": TRAIN_PROPOSAL_DIAGNOSTIC_VERSION,
            "manifest_hash": manifest.manifest_hash,
            "target_protocol_hash": protocol_hash,
            "proposal_only": True,
            "source_agent_trials_reexecuted": 0,
        },
    )
    evidence = reconstruct_v315_train_evidence(
        root=root,
        manifest=manifest,
        source_run_root=source_run_root,
        source_train_receipt=source_train_receipt,
        event_sink=sink,
    )
    if evidence.source_model != str(protocol["model"]):
        raise PermissionError("source and target proposal model identities differ")
    if PROPOSAL_FORMATION_POLICY_V2 != (
        TARGET_PROPOSAL_FORMATION_POLICY_VERSION
    ):
        raise RuntimeError("V3.17 family-slot policy constants disagree")

    recording_model = _RecordingProposalModel(proposal_model)
    proposer = StructuredHypothesisProposer(recording_model, event_sink=sink)
    evaluator_epoch = f"proposal-diagnostic-{manifest.manifest_hash[:12]}"
    validation_context = ValidationContext(
        evaluator_epoch=evaluator_epoch,
        residuals=evidence.residuals,
        available_lanes=frozenset(
            {"skilllearn_challenger", "skilllearn_incumbent"}
        ),
        baseline_lane="skilllearn_incumbent",
        trigger_feature_catalog=build_trigger_feature_catalog(
            evidence.residuals
        ),
        allowed_runtime_kinds=frozenset(
            HypothesisKind(str(value))
            for value in execution["runtime_candidate_kinds"]
        ),
        allowed_action_operations=frozenset(
            SKILLLEARN_ALLOWED_ACTION_OPERATIONS
        ),
        action_semantics=SKILL_ACTION_LOWERING_VERSION,
        external_evidence_is_hidden=True,
        contrastive_training_evidence_policy=str(
            execution["contrastive_training_evidence_policy"]
        ),
        repair_request_scope_policy=str(
            execution["repair_request_scope_policy"]
        ),
        train_action_design_policy=str(
            execution["train_action_design_policy"]
        ),
        action_design_profiles=evidence.action_profiles,
    )
    kernel = EvolutionKernel(
        proposer=proposer,
        validator=_ForbiddenRuntimeDependency(),  # type: ignore[arg-type]
        counterfactual_runner=_ForbiddenRuntimeDependency(),  # type: ignore[arg-type]
        promotion_gate=_ProposalOnlyPromotionGate(  # type: ignore[arg-type]
            _ProposalOnlyPromotionSpec(str(protocol["promotion"]["metric"]))
        ),
        archive=PolicyArchive(event_sink=sink),
        split_guard=SplitAccessGuard(manifest, event_sink=sink),
        proposal_candidates_per_generation=int(
            evolution["proposal_candidates_per_generation"]
        ),
        candidate_selection_policy=str(
            execution["proposal_candidate_selection"]
        ),
        candidate_bundle_policy=str(execution["candidate_bundle_policy"]),
        contrastive_training_evidence_policy=str(
            execution["contrastive_training_evidence_policy"]
        ),
        train_action_design_policy=str(
            execution["train_action_design_policy"]
        ),
        proposal_formation_policy=str(
            execution["proposal_formation_policy"]
        ),
        repair_request_scope_policy=str(
            execution["repair_request_scope_policy"]
        ),
        event_sink=sink,
    )
    programs = list(
        kernel.propose_candidates(
            evidence.residuals,
            validation_context=validation_context,
            trace_id=f"{trace_id}:production-v317",
        )
    )
    if len(programs) != REQUIRED_SLOT_COUNT:
        raise PermissionError("production V3.17 did not return exactly three roots")
    if len(recording_model.requests) != REQUIRED_SLOT_COUNT:
        raise PermissionError("production V3.17 did not make exactly three model calls")

    plan_event = _single_production_event(
        sink.events,
        event="proposal_family_slot_plan_created",
        trace_id=f"{trace_id}:production-v317",
    )
    completed_events = _production_events(
        sink.events,
        event="proposal_family_slot_completed",
        trace_prefix=f"{trace_id}:production-v317:family-slot-",
    )
    if len(completed_events) != REQUIRED_SLOT_COUNT:
        raise PermissionError("production V3.17 slot completion ledger is incomplete")
    plan_payload = plan_event["payload"]
    production_plan_rows = plan_payload.get("slots")
    if not isinstance(production_plan_rows, list) or len(
        production_plan_rows
    ) != REQUIRED_SLOT_COUNT:
        raise PermissionError("production V3.17 slot plan is malformed")
    slot_plan_hash = str(plan_payload.get("slot_plan_hash") or "")
    if slot_plan_hash != stable_hash(
        {
            "policy": TARGET_PROPOSAL_FORMATION_POLICY_VERSION,
            "slots": production_plan_rows,
        }
    ):
        raise PermissionError("production V3.17 slot plan hash mismatch")

    slots = tuple(
        _slot_from_production_request(
            request,
            plan_row=production_plan_rows[index],
            index=index,
            action_profiles=evidence.action_profiles,
        )
        for index, request in enumerate(recording_model.requests)
    )
    safe_slots = [slot.safe_payload() for slot in slots]
    for slot, production_row in zip(slots, production_plan_rows):
        if slot.formation_payload() != production_row:
            raise PermissionError(
                "in-memory model contract differs from production slot plan"
            )
    _emit_boundary_event(
        sink,
        event="train_proposal_slot_plan_frozen",
        trace_id=trace_id,
        payload={
            "diagnostic_policy": TRAIN_PROPOSAL_DIAGNOSTIC_VERSION,
            "proposal_formation_policy": (
                TARGET_PROPOSAL_FORMATION_POLICY_VERSION
            ),
            "production_evolution_kernel_used": True,
            "slot_plan_hash": slot_plan_hash,
            "slot_count": len(slots),
            "target_family_hashes": [
                slot.target_family_hash for slot in slots
            ],
            "target_failure_counts": [
                slot.target_failure_count for slot in slots
            ],
            "preferred_primitive_set_hashes": [
                slot.formation_payload()["preferred_primitive_set_hash"]
                for slot in slots
            ],
            "failed_primitive_set_hashes": [
                slot.formation_payload()["failed_primitive_set_hash"]
                for slot in slots
            ],
        },
    )

    proposal_rows: list[dict[str, Any]] = []
    failure_rows = evidence.failures
    for slot, program, completed in zip(slots, programs, completed_events):
        audit = _audit_slot_program(
            program,
            slot=slot,
            failures=failure_rows,
            action_profiles=evidence.action_profiles,
        )
        _bind_production_completion(
            audit,
            slot=slot,
            program=program,
            completed_payload=completed["payload"],
        )
        proposal_rows.append(audit)
        _emit_boundary_event(
            sink,
            event="train_proposal_slot_completed",
            trace_id=f"{trace_id}:slot-{slot.index}",
            payload={
                "diagnostic_policy": TRAIN_PROPOSAL_DIAGNOSTIC_VERSION,
                "proposal_formation_policy": (
                    TARGET_PROPOSAL_FORMATION_POLICY_VERSION
                ),
                "slot_plan_hash": slot_plan_hash,
                "slot_index": slot.index,
                "target_family_hash": slot.target_family_hash,
                "hypothesis_id_hash": stable_hash(
                    {"hypothesis_id": program.id}
                ),
                "hypothesis_hash": program.payload_hash,
                "activation_signature_hash": audit[
                    "activation_signature_hash"
                ],
                "matched_target_support": audit["matched_target_support"],
                "matched_failure_family_count": audit[
                    "matched_failure_family_count"
                ],
                "profile_environment_binding_count": audit[
                    "profile_environment_binding_count"
                ],
                "profile_grounded_delta_kinds": audit[
                    "profile_grounded_delta_kinds"
                ],
                "restatement_only_action_count": audit[
                    "restatement_only_action_count"
                ],
            },
        )

    signature_hashes = {
        str(row["activation_signature_hash"]) for row in proposal_rows
    }
    root_count_passed = len(programs) == REQUIRED_SLOT_COUNT
    single_family_signatures_passed = bool(
        root_count_passed
        and len(signature_hashes) == REQUIRED_SLOT_COUNT
        and all(
            int(row["matched_failure_family_count"]) == 1
            and bool(row["matched_target_family_only"])
            for row in proposal_rows
        )
    )
    minimum_support_passed = bool(
        proposal_rows
        and all(
            int(row["matched_target_support"]) >= MINIMUM_TARGET_SUPPORT
            for row in proposal_rows
        )
    )
    anti_trigger_passed = bool(
        proposal_rows
        and all(
            int(row["target_anti_trigger_self_block_count"]) == 0
            for row in proposal_rows
        )
    )
    environment_binding_passed = bool(
        proposal_rows
        and all(
            int(row["profile_environment_binding_count"]) > 0
            for row in proposal_rows
        )
    )
    executable_delta_passed = bool(
        proposal_rows
        and all(
            bool(
                {
                    "concrete_local_tool_command",
                    "artifact_internal_manipulation",
                }
                & set(row["profile_grounded_delta_kinds"])
            )
            for row in proposal_rows
        )
    )
    restatement_passed = bool(
        proposal_rows
        and all(
            int(row["restatement_only_action_count"]) == 0
            and not bool(row["restatement_risk"])
            for row in proposal_rows
        )
    )
    failed_primitive_avoidance_passed = bool(
        proposal_rows
        and all(
            int(row["failed_primitive_binding_count"]) == 0
            for row in proposal_rows
        )
    )
    schema_passed = bool(
        proposal_rows
        and all(not row["validation_issues"] for row in proposal_rows)
    )
    acceptance = {
        "root_count_passed": root_count_passed,
        "distinct_single_family_signatures_passed": (
            single_family_signatures_passed
        ),
        "minimum_target_support_passed": minimum_support_passed,
        "target_anti_trigger_self_block_absent": anti_trigger_passed,
        "profile_environment_binding_passed": environment_binding_passed,
        "profile_grounded_executable_delta_passed": executable_delta_passed,
        "restatement_only_absent": restatement_passed,
        "failed_profile_primitive_avoidance_passed": (
            failed_primitive_avoidance_passed
        ),
        "schema_validation_passed": schema_passed,
    }
    diagnostic_passed = all(acceptance.values())
    proposal_set_hash = stable_hash(
        {
            "proposal_hashes": [
                program.payload_hash for program in programs
            ]
        }
    )
    report = {
        "diagnostic_policy": TRAIN_PROPOSAL_DIAGNOSTIC_VERSION,
        "diagnostic_only": True,
        "diagnostic_passed": diagnostic_passed,
        "failure_blocks_future_trial_spend_only": True,
        "promotion_gate_or_score": False,
        "production_evolution_kernel_used": True,
        "validation_task_count": 0,
        "backend_call_count": 0,
        "evaluator_call_count": 0,
        "proposal_set_hash": proposal_set_hash,
        "source": {
            "source_train_receipt_hash": evidence.source_train_receipt_hash,
            "source_protocol_hash": evidence.source_protocol_hash,
            "source_protocol_lock_hash": evidence.source_protocol_lock_hash,
            "source_checkpoint_hash": evidence.source_checkpoint_hash,
            "source_observation_set_hash": evidence.source_observation_set_hash,
            "source_row_set_hash": evidence.source_row_set_hash,
            "manifest_hash": manifest.manifest_hash,
            "train_observation_count": len(evidence.residuals),
            "train_failure_count": len(evidence.failures),
            "train_success_control_count": len(evidence.success_controls),
            "source_agent_trials_reexecuted": 0,
        },
        "target_protocol": {
            "protocol_id": str(protocol["protocol_id"]),
            "protocol_hash": protocol_hash,
            "proposal_formation_policy": (
                TARGET_PROPOSAL_FORMATION_POLICY_VERSION
            ),
            "proposal_formation_policy_hash": stable_hash(
                {
                    "policy": TARGET_PROPOSAL_FORMATION_POLICY_VERSION
                }
            ),
            "proposal_model": str(protocol["model"]),
            "proposal_model_call_count": len(recording_model.requests),
        },
        "action_profiles": {
            "profile_count": evidence.action_profile_count,
            "profile_set_hash": evidence.action_profile_set_hash,
            "source_provenance_matched": True,
        },
        "slot_plan": {
            "policy": TARGET_PROPOSAL_FORMATION_POLICY_VERSION,
            "slot_plan_hash": slot_plan_hash,
            "slot_count": len(slots),
            "target_family_hashes": [
                slot.target_family_hash for slot in slots
            ],
            "slots": [
                safe for safe in safe_slots
            ],
        },
        "proposals": proposal_rows,
        "action_audit_summary": {
            "candidate_count": len(proposal_rows),
            "profile_environment_bound_candidate_count": sum(
                int(row["profile_environment_binding_count"]) > 0
                for row in proposal_rows
            ),
            "concrete_local_tool_candidate_count": sum(
                "concrete_local_tool_command"
                in row["profile_grounded_delta_kinds"]
                for row in proposal_rows
            ),
            "artifact_internal_manipulation_candidate_count": sum(
                "artifact_internal_manipulation"
                in row["profile_grounded_delta_kinds"]
                for row in proposal_rows
            ),
            "restatement_risk_candidate_count": sum(
                bool(row["restatement_risk"]) for row in proposal_rows
            ),
            "restatement_only_action_count": sum(
                int(row["restatement_only_action_count"])
                for row in proposal_rows
            ),
        },
        "acceptance": acceptance,
        **_COMMON_BOUNDARY_FLAGS,
    }
    _assert_no_persisted_raw_keys(report, path="report")
    _assert_persisted_hash_fields(report, path="report")
    _emit_boundary_event(
        sink,
        event="train_proposal_diagnostic_completed",
        trace_id=trace_id,
        payload={
            "diagnostic_policy": TRAIN_PROPOSAL_DIAGNOSTIC_VERSION,
            "target_protocol_hash": protocol_hash,
            "slot_plan_hash": slot_plan_hash,
            "proposal_count": len(programs),
            "proposal_set_hash": proposal_set_hash,
            "distinct_activation_signature_count": len(signature_hashes),
            "diagnostic_passed": diagnostic_passed,
            "acceptance": acceptance,
            "failure_blocks_future_trial_spend_only": True,
            "promotion_gate_or_score": False,
        },
    )
    return report


def verify_existing_train_proposal_diagnostic(
    *,
    root: str | Path,
    manifest_path: str | Path,
    source_run_root: str | Path,
    source_train_receipt: str | Path,
    protocol_path: str | Path,
    report_path: str | Path,
    events_path: str | Path,
) -> dict[str, Any]:
    """Verify a completed diagnostic for safe, fully local reuse.

    Verification performs no proposal-model call.  It reconstructs the frozen
    V3.15 TRAIN evidence again so a changed result, trace, source receipt,
    manifest, protocol lock, or action profile cannot reuse an earlier pass.
    """

    report_file = Path(report_path).expanduser().resolve(strict=True)
    events_file = Path(events_path).expanduser().resolve(strict=True)
    report = _read_json_object(
        report_file,
        label="existing proposal diagnostic report",
    )
    _assert_no_persisted_raw_keys(report, path="existing_report")
    _assert_persisted_hash_fields(report, path="existing_report")
    manifest = SplitManifest.read(manifest_path)
    protocol, protocol_hash = _read_target_protocol(protocol_path)
    evidence = reconstruct_v315_train_evidence(
        root=root,
        manifest=manifest,
        source_run_root=source_run_root,
        source_train_receipt=source_train_receipt,
        event_sink=NullEventSink(),
    )

    required_top_level = {
        "diagnostic_policy": TRAIN_PROPOSAL_DIAGNOSTIC_VERSION,
        "diagnostic_only": True,
        "diagnostic_passed": True,
        "failure_blocks_future_trial_spend_only": True,
        "promotion_gate_or_score": False,
        "production_evolution_kernel_used": True,
        "validation_task_count": 0,
        "backend_call_count": 0,
        "evaluator_call_count": 0,
    }
    _require_exact_fields(
        report,
        required_top_level,
        label="existing diagnostic report",
    )
    _require_false_boundary_flags(
        report,
        label="existing diagnostic report",
    )
    acceptance = _required_mapping(report, "acceptance")
    if set(acceptance) != set(_ACCEPTANCE_KEYS) or any(
        acceptance.get(key) is not True for key in _ACCEPTANCE_KEYS
    ):
        raise PermissionError(
            "existing diagnostic acceptance is incomplete or not passing"
        )

    source = _required_mapping(report, "source")
    _require_exact_fields(
        source,
        {
            "source_train_receipt_hash": evidence.source_train_receipt_hash,
            "source_protocol_hash": evidence.source_protocol_hash,
            "source_protocol_lock_hash": evidence.source_protocol_lock_hash,
            "source_checkpoint_hash": evidence.source_checkpoint_hash,
            "source_observation_set_hash": (
                evidence.source_observation_set_hash
            ),
            "source_row_set_hash": evidence.source_row_set_hash,
            "manifest_hash": manifest.manifest_hash,
            "train_observation_count": len(evidence.residuals),
            "train_failure_count": len(evidence.failures),
            "train_success_control_count": len(evidence.success_controls),
            "source_agent_trials_reexecuted": 0,
        },
        label="existing diagnostic source binding",
    )
    target = _required_mapping(report, "target_protocol")
    _require_exact_fields(
        target,
        {
            "protocol_id": str(protocol["protocol_id"]),
            "protocol_hash": protocol_hash,
            "proposal_formation_policy": (
                TARGET_PROPOSAL_FORMATION_POLICY_VERSION
            ),
            "proposal_formation_policy_hash": stable_hash(
                {"policy": TARGET_PROPOSAL_FORMATION_POLICY_VERSION}
            ),
            "proposal_model": str(protocol["model"]),
            "proposal_model_call_count": REQUIRED_SLOT_COUNT,
        },
        label="existing diagnostic target binding",
    )
    profiles = _required_mapping(report, "action_profiles")
    _require_exact_fields(
        profiles,
        {
            "profile_count": evidence.action_profile_count,
            "profile_set_hash": evidence.action_profile_set_hash,
            "source_provenance_matched": True,
        },
        label="existing diagnostic action-profile binding",
    )

    slot_plan = _required_mapping(report, "slot_plan")
    _require_exact_fields(
        slot_plan,
        {
            "policy": TARGET_PROPOSAL_FORMATION_POLICY_VERSION,
            "slot_count": REQUIRED_SLOT_COUNT,
        },
        label="existing diagnostic slot plan",
    )
    slot_plan_hash = str(slot_plan.get("slot_plan_hash") or "")
    if not _is_sha256(slot_plan_hash):
        raise PermissionError("existing diagnostic slot-plan hash is malformed")
    slots = slot_plan.get("slots")
    target_family_hashes = slot_plan.get("target_family_hashes")
    if not isinstance(slots, list) or len(slots) != REQUIRED_SLOT_COUNT:
        raise PermissionError("existing diagnostic slot rows are incomplete")
    if not isinstance(target_family_hashes, list) or len(
        target_family_hashes
    ) != REQUIRED_SLOT_COUNT:
        raise PermissionError(
            "existing diagnostic target-family hashes are incomplete"
        )
    if len(set(str(value) for value in target_family_hashes)) != (
        REQUIRED_SLOT_COUNT
    ) or any(not _is_sha256(value) for value in target_family_hashes):
        raise PermissionError(
            "existing diagnostic target-family hashes are malformed"
        )
    if [row.get("target_family_hash") for row in slots if isinstance(row, Mapping)] != list(
        target_family_hashes
    ):
        raise PermissionError(
            "existing diagnostic slot target-family hashes disagree"
        )

    proposals = report.get("proposals")
    if not isinstance(proposals, list) or len(proposals) != REQUIRED_SLOT_COUNT:
        raise PermissionError("existing diagnostic proposals are incomplete")
    proposal_hashes = [
        str(row.get("hypothesis_hash") or "")
        for row in proposals
        if isinstance(row, Mapping)
    ]
    if len(proposal_hashes) != REQUIRED_SLOT_COUNT or any(
        not _is_sha256(value) for value in proposal_hashes
    ):
        raise PermissionError("existing diagnostic proposal hashes are malformed")
    proposal_set_hash = stable_hash(
        {"proposal_hashes": proposal_hashes}
    )
    if report.get("proposal_set_hash") != proposal_set_hash:
        raise PermissionError("existing diagnostic proposal-set hash mismatch")

    events = _read_verified_diagnostic_events(events_file)
    completed = _event_rows(
        events,
        event="train_proposal_diagnostic_completed",
    )
    if len(completed) != 1:
        raise PermissionError(
            "existing diagnostic completion event count mismatch"
        )
    completion = _required_mapping(completed[0], "payload")
    _require_exact_fields(
        completion,
        {
            "diagnostic_policy": TRAIN_PROPOSAL_DIAGNOSTIC_VERSION,
            "target_protocol_hash": protocol_hash,
            "slot_plan_hash": slot_plan_hash,
            "proposal_count": REQUIRED_SLOT_COUNT,
            "proposal_set_hash": proposal_set_hash,
            "diagnostic_passed": True,
            "acceptance": dict(acceptance),
            "failure_blocks_future_trial_spend_only": True,
            "promotion_gate_or_score": False,
        },
        label="existing diagnostic completion event",
    )
    source_events = _event_rows(
        events,
        event="train_proposal_source_evidence_reconstructed",
    )
    if len(source_events) != 1:
        raise PermissionError(
            "existing diagnostic source event count mismatch"
        )
    source_event = _required_mapping(source_events[0], "payload")
    _require_exact_fields(
        source_event,
        {
            "manifest_hash": manifest.manifest_hash,
            "source_train_receipt_hash": evidence.source_train_receipt_hash,
            "source_protocol_hash": evidence.source_protocol_hash,
            "source_protocol_lock_hash": evidence.source_protocol_lock_hash,
            "source_observation_set_hash": (
                evidence.source_observation_set_hash
            ),
            "source_row_set_hash": evidence.source_row_set_hash,
            "action_profile_count": evidence.action_profile_count,
            "action_profile_set_hash": evidence.action_profile_set_hash,
        },
        label="existing diagnostic source event",
    )
    production_plans = _event_rows(
        events,
        event="proposal_family_slot_plan_created",
    )
    if len(production_plans) != 1 or _required_mapping(
        production_plans[0], "payload"
    ).get("slot_plan_hash") != slot_plan_hash:
        raise PermissionError(
            "existing diagnostic production slot-plan binding mismatch"
        )
    production_completions = sorted(
        _event_rows(events, event="proposal_family_slot_completed"),
        key=lambda row: str(row.get("trace_id") or ""),
    )
    if len(production_completions) != REQUIRED_SLOT_COUNT or [
        _required_mapping(row, "payload").get("candidate_hash")
        for row in production_completions
    ] != proposal_hashes:
        raise PermissionError(
            "existing diagnostic production proposal binding mismatch"
        )
    diagnostic_slots = sorted(
        _event_rows(events, event="train_proposal_slot_completed"),
        key=lambda row: int(
            _required_mapping(row, "payload").get("slot_index", -1)
        ),
    )
    if len(diagnostic_slots) != REQUIRED_SLOT_COUNT or [
        _required_mapping(row, "payload").get("hypothesis_hash")
        for row in diagnostic_slots
    ] != proposal_hashes:
        raise PermissionError(
            "existing diagnostic slot audit binding mismatch"
        )
    _verify_live_three_slot_proposal_flow(
        events,
        report=report,
        protocol=protocol,
    )

    return {
        "diagnostic_reuse_verified": True,
        "diagnostic_policy": TRAIN_PROPOSAL_DIAGNOSTIC_VERSION,
        "report_hash": _sha256_file(report_file),
        "event_ledger_hash": _sha256_file(events_file),
        "event_count": len(events),
        "manifest_hash": manifest.manifest_hash,
        "source_train_receipt_hash": evidence.source_train_receipt_hash,
        "source_protocol_lock_hash": evidence.source_protocol_lock_hash,
        "source_observation_set_hash": evidence.source_observation_set_hash,
        "source_row_set_hash": evidence.source_row_set_hash,
        "action_profile_set_hash": evidence.action_profile_set_hash,
        "target_protocol_hash": protocol_hash,
        "slot_plan_hash": slot_plan_hash,
        "proposal_set_hash": proposal_set_hash,
        "live_provider_model_ledger_verified": True,
        **_COMMON_BOUNDARY_FLAGS,
    }


def _verify_live_three_slot_proposal_flow(
    events: Sequence[Mapping[str, Any]],
    *,
    report: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    """Require three non-replayed provider/model/proposer executions."""

    exact_counts = {
        "model_provider_chain_built": 1,
        "train_proposal_diagnostic_started": 1,
        "train_proposal_source_evidence_reconstructed": 1,
        "proposal_family_slot_plan_created": 1,
        "hypothesis_proposal_requested": REQUIRED_SLOT_COUNT,
        "model_provider_attempted": REQUIRED_SLOT_COUNT,
        "model_attempt_succeeded": REQUIRED_SLOT_COUNT,
        "model_provider_selected": REQUIRED_SLOT_COUNT,
        "proposal_action_delta_audited": REQUIRED_SLOT_COUNT,
        "hypothesis_proposed": REQUIRED_SLOT_COUNT,
        "root_proposal_evidence_recorded": REQUIRED_SLOT_COUNT,
        "proposal_family_slot_usage_recorded": 1,
        "proposal_family_slot_completed": REQUIRED_SLOT_COUNT,
        "proposal_family_slots_completed": 1,
        "train_proposal_slot_plan_frozen": 1,
        "train_proposal_slot_completed": REQUIRED_SLOT_COUNT,
        "train_proposal_diagnostic_completed": 1,
    }
    forbidden_counts = {
        "model_provider_failed": 0,
        "model_provider_chain_exhausted": 0,
        "hypothesis_proposal_model_call_failed": 0,
        "hypothesis_proposal_response_rejected": 0,
        "proposal_action_delta_audit_failed": 0,
        "root_proposal_evidence_replayed": 0,
        "proposal_family_slot_usage_replayed": 0,
    }
    for event, expected in {**exact_counts, **forbidden_counts}.items():
        if len(_event_rows(events, event=event)) != expected:
            raise PermissionError(
                f"proposal diagnostic live ledger count mismatch: {event}"
            )
    attempt_starts = _event_rows(events, event="model_attempt_started")
    attempt_failures = _event_rows(events, event="model_attempt_failed")
    if not REQUIRED_SLOT_COUNT <= len(attempt_starts) <= (
        REQUIRED_SLOT_COUNT * 2
    ):
        raise PermissionError(
            "proposal diagnostic live ledger transport-attempt count mismatch"
        )
    if len(attempt_failures) != len(attempt_starts) - REQUIRED_SLOT_COUNT:
        raise PermissionError(
            "proposal diagnostic live ledger transport-retry count mismatch"
        )

    provider_chain = list(protocol.get("proposal_provider_chain") or [])
    if provider_chain != ["openai_compatible"]:
        raise PermissionError(
            "proposal diagnostic live ledger requires the frozen provider chain"
        )
    model = str(protocol.get("model") or "")
    provider_chain_hash = stable_hash(
        {"providers": provider_chain, "model": model}
    )
    chain_index, chain_row = _only_indexed_event(
        events,
        event="model_provider_chain_built",
    )
    chain_payload = _required_mapping(chain_row, "payload")
    _require_exact_fields(
        chain_payload,
        {
            "requested_providers": provider_chain,
            "active_providers": provider_chain,
            "unavailable_providers": [],
            "model": model,
            "provider_chain_hash": provider_chain_hash,
        },
        label="proposal diagnostic live provider chain",
    )

    plan_index, plan_event = _only_indexed_event(
        events,
        event="proposal_family_slot_plan_created",
    )
    started_index, _ = _only_indexed_event(
        events,
        event="train_proposal_diagnostic_started",
    )
    source_index, _ = _only_indexed_event(
        events,
        event="train_proposal_source_evidence_reconstructed",
    )
    if not chain_index < started_index < source_index < plan_index:
        raise PermissionError(
            "proposal diagnostic live ledger setup order mismatch"
        )
    root_trace = str(plan_event.get("trace_id") or "")
    plan_payload = _required_mapping(plan_event, "payload")
    slot_plan_hash = str(plan_payload.get("slot_plan_hash") or "")
    report_slot_plan = _required_mapping(report, "slot_plan")
    if report_slot_plan.get("slot_plan_hash") != slot_plan_hash:
        raise PermissionError(
            "proposal diagnostic live ledger slot-plan hash mismatch"
        )
    plan_slots = plan_payload.get("slots")
    report_slots = report_slot_plan.get("slots")
    proposals = report.get("proposals")
    if not all(
        isinstance(rows, list) and len(rows) == REQUIRED_SLOT_COUNT
        for rows in (plan_slots, report_slots, proposals)
    ):
        raise PermissionError(
            "proposal diagnostic live ledger slot rows are incomplete"
        )
    assert isinstance(plan_slots, list)
    assert isinstance(report_slots, list)
    assert isinstance(proposals, list)
    proposal_hashes = [
        str(_required_mapping({"row": row}, "row").get("hypothesis_hash") or "")
        for row in proposals
    ]
    if any(not _is_sha256(value) for value in proposal_hashes):
        raise PermissionError(
            "proposal diagnostic live ledger proposal hashes are malformed"
        )

    endpoint = str(protocol.get("provider_endpoint_origin") or "").rstrip("/")
    endpoint_url = (
        f"{endpoint}/chat/completions"
        if endpoint.endswith("/v1")
        else f"{endpoint}/v1/chat/completions"
    )
    endpoint_hash = stable_hash({"url": endpoint_url})
    proposal_request_hashes: list[str] = []
    transport_request_hashes: list[str] = []
    failed_binding_counts: list[int] = []
    accounted_attempt_starts = 0
    accounted_attempt_failures = 0
    prior_slot_recorded_index = plan_index

    for slot_number in range(1, REQUIRED_SLOT_COUNT + 1):
        slot_trace = f"{root_trace}:family-slot-{slot_number}"
        request_index, request_event = _only_indexed_event(
            events,
            event="hypothesis_proposal_requested",
            trace_id=slot_trace,
        )
        provider_index, provider_event = _only_indexed_event(
            events,
            event="model_provider_attempted",
            trace_id=slot_trace,
        )
        success_index, success_event = _only_indexed_event(
            events,
            event="model_attempt_succeeded",
            trace_id=slot_trace,
        )
        selected_index, selected_event = _only_indexed_event(
            events,
            event="model_provider_selected",
            trace_id=slot_trace,
        )
        audit_index, _ = _only_indexed_event(
            events,
            event="proposal_action_delta_audited",
            trace_id=slot_trace,
        )
        proposed_index, proposed_event = _only_indexed_event(
            events,
            event="hypothesis_proposed",
            trace_id=slot_trace,
        )
        recorded_index, recorded_event = _only_indexed_event(
            events,
            event="root_proposal_evidence_recorded",
            trace_id=slot_trace,
        )
        completion_index, completion_event = _only_indexed_event(
            events,
            event="proposal_family_slot_completed",
            trace_id=slot_trace,
        )
        if not (
            prior_slot_recorded_index
            < request_index
            < provider_index
            < success_index
            < selected_index
            < audit_index
            < proposed_index
            < recorded_index
            < completion_index
        ):
            raise PermissionError(
                "proposal diagnostic live ledger event order mismatch"
            )
        prior_slot_recorded_index = recorded_index

        request_payload = _required_mapping(request_event, "payload")
        provider_payload = _required_mapping(provider_event, "payload")
        success_payload = _required_mapping(success_event, "payload")
        selected_payload = _required_mapping(selected_event, "payload")
        proposed_payload = _required_mapping(proposed_event, "payload")
        recorded_payload = _required_mapping(recorded_event, "payload")
        completion_payload = _required_mapping(completion_event, "payload")
        proposal_audit_row = _required_mapping(
            {"row": proposals[slot_number - 1]},
            "row",
        )
        proposal_request_hash = str(request_payload.get("request_hash") or "")
        proposal_request_hashes.append(proposal_request_hash)
        _require_exact_fields(
            provider_payload,
            {
                "provider": provider_chain[0],
                "provider_position": 0,
                "provider_count": 1,
                "provider_chain_hash": provider_chain_hash,
                "request_hash": proposal_request_hash,
                "model": model,
            },
            label="proposal diagnostic live provider attempt",
        )
        _require_exact_fields(
            selected_payload,
            {
                "provider": provider_chain[0],
                "provider_chain_hash": provider_chain_hash,
                "request_hash": proposal_request_hash,
                "model": model,
                "failover_used": False,
                "prior_failure_count": 0,
            },
            label="proposal diagnostic live provider selection",
        )
        if selected_payload.get("response_hash") != success_payload.get(
            "response_hash"
        ):
            raise PermissionError(
                "proposal diagnostic live ledger response hash mismatch"
            )

        starts = _indexed_events(
            events,
            event="model_attempt_started",
            trace_id=slot_trace,
        )
        failures = _indexed_events(
            events,
            event="model_attempt_failed",
            trace_id=slot_trace,
        )
        if len(starts) not in {1, 2} or len(failures) != len(starts) - 1:
            raise PermissionError(
                "proposal diagnostic live ledger per-slot retry mismatch"
            )
        accounted_attempt_starts += len(starts)
        accounted_attempt_failures += len(failures)
        start_payloads = [
            _required_mapping(row, "payload") for _, row in starts
        ]
        transport_hash = str(start_payloads[0].get("request_hash") or "")
        transport_request_hashes.append(transport_hash)
        if [payload.get("attempt") for payload in start_payloads] != list(
            range(1, len(starts) + 1)
        ) or any(
            payload.get("attempt_limit") != 2
            or payload.get("request_hash") != transport_hash
            or payload.get("model") != model
            or payload.get("endpoint_hash") != endpoint_hash
            for payload in start_payloads
        ):
            raise PermissionError(
                "proposal diagnostic live ledger transport start mismatch"
            )
        if not provider_index < starts[0][0]:
            raise PermissionError(
                "proposal diagnostic live ledger transport order mismatch"
            )
        if success_payload.get("attempt") != len(starts) or any(
            success_payload.get(key) != value
            for key, value in {
                "request_hash": transport_hash,
                "model": model,
            }.items()
        ):
            raise PermissionError(
                "proposal diagnostic live ledger transport success mismatch"
            )
        if failures:
            failure_index, failure_event = failures[0]
            failure_payload = _required_mapping(failure_event, "payload")
            if not (
                starts[0][0] < failure_index < starts[1][0] < success_index
            ) or any(
                failure_payload.get(key) != value
                for key, value in {
                    "attempt": 1,
                    "request_hash": transport_hash,
                    "model": model,
                    "retryable": True,
                }.items()
            ):
                raise PermissionError(
                    "proposal diagnostic live ledger transport retry order mismatch"
                )
        elif not starts[0][0] < success_index:
            raise PermissionError(
                "proposal diagnostic live ledger transport order mismatch"
            )

        proposal_hash = proposal_hashes[slot_number - 1]
        if proposed_payload.get("hypothesis_hash") != proposal_hash:
            raise PermissionError(
                "proposal diagnostic live ledger hypothesis hash mismatch"
            )
        _require_exact_fields(
            recorded_payload,
            {
                "request_hash": proposal_request_hash,
                "program_count": 1,
                "program_set_hash": stable_hash(
                    {"program_hashes": [proposal_hash]}
                ),
                "new_proposal_model_executions": 1,
            },
            label="proposal diagnostic live root evidence",
        )
        plan_row = _required_mapping(
            {"row": plan_slots[slot_number - 1]},
            "row",
        )
        report_row = _required_mapping(
            {"row": report_slots[slot_number - 1]},
            "row",
        )
        _require_exact_fields(
            completion_payload,
            {
                "slot_id": f"train-family-slot-{slot_number}",
                "slot_plan_hash": slot_plan_hash,
                "target_family_hash": plan_row.get("target_family_hash"),
                "candidate_hash": proposal_hash,
            },
            label="proposal diagnostic live slot completion",
        )
        raw_failed_binding_count = completion_payload.get(
            "failed_profile_binding_count"
        )
        if (
            isinstance(raw_failed_binding_count, bool)
            or not isinstance(raw_failed_binding_count, int)
            or raw_failed_binding_count < 0
        ):
            raise PermissionError(
                "proposal diagnostic live failed-binding count is malformed"
            )
        failed_binding_counts.append(raw_failed_binding_count)
        _require_exact_fields(
            proposal_audit_row,
            {
                "production_failed_profile_binding_count": (
                    raw_failed_binding_count
                ),
                "failed_primitive_binding_count": raw_failed_binding_count,
            },
            label="proposal diagnostic live failed-binding audit",
        )
        if report_row.get("target_family_hash") != plan_row.get(
            "target_family_hash"
        ):
            raise PermissionError(
                "proposal diagnostic live target-family binding mismatch"
            )

    if len(set(proposal_request_hashes)) != REQUIRED_SLOT_COUNT or any(
        not _is_sha256(value) for value in proposal_request_hashes
    ):
        raise PermissionError(
            "proposal diagnostic live proposal request identities mismatch"
        )
    if accounted_attempt_starts != len(attempt_starts) or (
        accounted_attempt_failures != len(attempt_failures)
    ):
        raise PermissionError(
            "proposal diagnostic live transport events escaped slot binding"
        )
    if len(set(transport_request_hashes)) != REQUIRED_SLOT_COUNT or any(
        not _is_sha256(value) for value in transport_request_hashes
    ):
        raise PermissionError(
            "proposal diagnostic live transport request identities mismatch"
        )
    acceptance = _required_mapping(report, "acceptance")
    expected_failed_avoidance = all(
        count == 0 for count in failed_binding_counts
    )
    if acceptance.get("failed_profile_primitive_avoidance_passed") is not (
        expected_failed_avoidance
    ):
        raise PermissionError(
            "proposal diagnostic live failed-binding acceptance mismatch"
        )

    usage_index, usage_event = _only_indexed_event(
        events,
        event="proposal_family_slot_usage_recorded",
    )
    slots_index, slots_event = _only_indexed_event(
        events,
        event="proposal_family_slots_completed",
    )
    frozen_index, frozen_event = _only_indexed_event(
        events,
        event="train_proposal_slot_plan_frozen",
    )
    final_index, _ = _only_indexed_event(
        events,
        event="train_proposal_diagnostic_completed",
    )
    completion_indices = [
        index
        for index, _ in _indexed_events(
            events,
            event="proposal_family_slot_completed",
        )
    ]
    diagnostic_indices = [
        index
        for index, _ in _indexed_events(
            events,
            event="train_proposal_slot_completed",
        )
    ]
    if not (
        prior_slot_recorded_index
        < usage_index
        < min(completion_indices)
        <= max(completion_indices)
        < slots_index
        < frozen_index
        < min(diagnostic_indices)
        <= max(diagnostic_indices)
        < final_index
    ):
        raise PermissionError(
            "proposal diagnostic live ledger generation order mismatch"
        )
    usage_payload = _required_mapping(usage_event, "payload")
    _require_exact_fields(
        usage_payload,
        {
            "source": "generated_family_slots",
            "candidate_count": REQUIRED_SLOT_COUNT,
            "requested_target_count": REQUIRED_SLOT_COUNT,
            "distinct_requested_target_count": REQUIRED_SLOT_COUNT,
            "proposal_set_hash": stable_hash(
                {"candidate_hashes": sorted(proposal_hashes)}
            ),
            "proposal_set_replayed": False,
            "family_use_updated": True,
            "new_family_use_count": REQUIRED_SLOT_COUNT,
        },
        label="proposal diagnostic live family usage",
    )
    slots_payload = _required_mapping(slots_event, "payload")
    _require_exact_fields(
        slots_payload,
        {
            "slot_plan_hash": slot_plan_hash,
            "candidate_count": REQUIRED_SLOT_COUNT,
            "candidate_set_hash": stable_hash(
                {"candidate_hashes": proposal_hashes}
            ),
        },
        label="proposal diagnostic live candidate set",
    )
    frozen_payload = _required_mapping(frozen_event, "payload")
    _require_exact_fields(
        frozen_payload,
        {
            "slot_plan_hash": slot_plan_hash,
            "slot_count": REQUIRED_SLOT_COUNT,
            "production_evolution_kernel_used": True,
        },
        label="proposal diagnostic live frozen plan",
    )


def _indexed_events(
    events: Sequence[Mapping[str, Any]],
    *,
    event: str,
    trace_id: str | None = None,
) -> list[tuple[int, Mapping[str, Any]]]:
    return [
        (index, row)
        for index, row in enumerate(events)
        if row.get("event") == event
        and (trace_id is None or row.get("trace_id") == trace_id)
    ]


def _only_indexed_event(
    events: Sequence[Mapping[str, Any]],
    *,
    event: str,
    trace_id: str | None = None,
) -> tuple[int, Mapping[str, Any]]:
    rows = _indexed_events(events, event=event, trace_id=trace_id)
    if len(rows) != 1:
        raise PermissionError(
            f"proposal diagnostic live ledger event count mismatch: {event}"
        )
    return rows[0]


def _read_verified_diagnostic_events(path: Path) -> list[Mapping[str, Any]]:
    events: list[Mapping[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise PermissionError(
                f"diagnostic event ledger is malformed at line {line_number}"
            ) from exc
        if not isinstance(row, Mapping):
            raise PermissionError(
                f"diagnostic event is not an object at line {line_number}"
            )
        expected_envelope_fields = {
            "event",
            "stage",
            "trace_id",
            "payload",
            "payload_hash",
            "event_id",
            "raw_content_persisted",
        }
        if set(row) != expected_envelope_fields:
            raise PermissionError(
                "diagnostic event envelope fields mismatch at line "
                f"{line_number}"
            )
        event = str(row.get("event") or "")
        stage = str(row.get("stage") or "")
        trace_id = str(row.get("trace_id") or "")
        payload = row.get("payload")
        if not event or not stage or not trace_id or not isinstance(
            payload, Mapping
        ):
            raise PermissionError(
                f"diagnostic event envelope is incomplete at line {line_number}"
            )
        canonical_payload = _sanitize_persisted_event_payload(event, payload)
        if dict(payload) != canonical_payload:
            raise PermissionError(
                f"diagnostic event is not canonically sanitized at line {line_number}"
            )
        if row.get("payload_hash") != stable_hash(dict(payload)):
            raise PermissionError(
                f"diagnostic event payload hash mismatch at line {line_number}"
            )
        expected_event_id = stable_hash(
            {
                "event": event,
                "stage": stage,
                "trace_id": trace_id,
                "payload": dict(payload),
            }
        )[:24]
        if row.get("event_id") != expected_event_id:
            raise PermissionError(
                f"diagnostic event ID mismatch at line {line_number}"
            )
        if row.get("raw_content_persisted") is not False:
            raise PermissionError(
                f"diagnostic event raw-content flag failed at line {line_number}"
            )
        _require_false_boundary_flags(
            payload,
            label=f"diagnostic event line {line_number}",
        )
        events.append(row)
    if not events:
        raise PermissionError("diagnostic event ledger is empty")
    return events


def _event_rows(
    events: Sequence[Mapping[str, Any]],
    *,
    event: str,
) -> list[Mapping[str, Any]]:
    return [row for row in events if row.get("event") == event]


def _required_mapping(
    payload: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise PermissionError(f"proposal diagnostic mapping is missing: {key}")
    return value


def _require_exact_fields(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise PermissionError(f"{label} mismatch: {key}")


def _require_false_boundary_flags(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for key, expected in _COMMON_BOUNDARY_FLAGS.items():
        if payload.get(key) is not expected:
            raise PermissionError(f"{label} boundary flag mismatch: {key}")


def _production_events(
    events: Sequence[Mapping[str, Any]],
    *,
    event: str,
    trace_prefix: str,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in events
        if row.get("event") == event
        and str(row.get("trace_id") or "").startswith(trace_prefix)
    ]


def _single_production_event(
    events: Sequence[Mapping[str, Any]],
    *,
    event: str,
    trace_id: str,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in events
        if row.get("event") == event and row.get("trace_id") == trace_id
    ]
    if len(matches) != 1:
        raise PermissionError(
            f"production V3.17 event count mismatch: {event}"
        )
    return matches[0]


def _slot_from_production_request(
    request: Mapping[str, Any],
    *,
    plan_row: Any,
    index: int,
    action_profiles: Mapping[str, Mapping[str, Any]],
) -> FamilyProposalSlot:
    if not isinstance(plan_row, Mapping):
        raise PermissionError("production V3.17 slot plan row is malformed")
    if request.get("max_hypotheses") != 1:
        raise PermissionError("production family slot was not singular")
    if "proposal_batch_contract" in request:
        raise PermissionError("production family slot used the root batch contract")
    response_contract = request.get("family_slot_response_contract")
    if not isinstance(response_contract, Mapping) or response_contract.get(
        "policy"
    ) != TARGET_PROPOSAL_FORMATION_POLICY_VERSION:
        raise PermissionError("production family-slot response contract is missing")
    if response_contract.get("response_field") != "hypothesis":
        raise PermissionError("production family-slot response is not singular")
    capabilities = request.get("capabilities")
    if not isinstance(capabilities, Mapping):
        raise PermissionError("production family-slot capabilities are missing")
    contract = capabilities.get("family_slot_contract")
    if not isinstance(contract, Mapping) or contract.get("policy") != (
        TARGET_PROPOSAL_FORMATION_POLICY_VERSION
    ):
        raise PermissionError("production family-slot contract is missing")
    for key in (
        "validation_features_used",
        "validation_outcomes_used",
        "verifier_content_used",
        "test_content_used",
    ):
        if contract.get(key) is not False:
            raise PermissionError(
                f"production family-slot contract crossed boundary: {key}"
            )
    portable = contract.get("portable_recipe_policy")
    if not isinstance(portable, Mapping):
        raise PermissionError("production portable-recipe policy is missing")
    for key in (
        "validation_features_used",
        "validation_outcomes_used",
        "verifier_content_used",
        "test_content_used",
    ):
        if portable.get(key) is not False:
            raise PermissionError(
                f"production portable-recipe policy crossed boundary: {key}"
            )
    preferred = _parse_profile_primitives(
        portable.get("preferred_allowlisted_profile_primitives"),
        label="preferred",
    )
    if "failed_profile_primitives_to_avoid" in portable:
        raise PermissionError(
            "production V2 family slot disclosed failed primitive values"
        )
    reusable_count = sum(row.reusable for row in preferred)
    if portable.get("reusable_preferred_primitive_count") != reusable_count:
        raise PermissionError(
            "production reusable preferred-primitive count mismatch"
        )
    if "train_action_design_profiles" in capabilities:
        raise PermissionError(
            "production V2 family slot disclosed the TRAIN profile map"
        )
    profile_summary = capabilities.get("train_action_design_profile_summary")
    if not isinstance(profile_summary, Mapping):
        raise PermissionError("production V2 profile summary is missing")
    if profile_summary.get("failed_primitive_values_disclosed") is not False:
        raise PermissionError("production V2 failed primitive disclosure flag drifted")
    target_family = str(contract.get("target_failure_family") or "")
    if not target_family:
        raise PermissionError("production family slot target is missing")
    support = contract.get("target_failure_support_count")
    if isinstance(support, bool) or not isinstance(support, int):
        raise PermissionError("production family-slot support is malformed")
    if support < MINIMUM_TARGET_SUPPORT:
        raise PermissionError("production family-slot support is below two")
    profile_evidence_hash = str(contract.get("profile_evidence_hash") or "")
    if not _is_sha256(profile_evidence_hash):
        raise PermissionError("production profile-evidence hash is malformed")
    if contract.get("success_control_count") != EXPECTED_TRAIN_SUCCESSES:
        raise PermissionError("production family slot omitted success controls")
    request_residuals = request.get("residuals")
    if not isinstance(request_residuals, list):
        raise PermissionError("production family-slot residuals are missing")
    failure_rows = [
        row
        for row in request_residuals
        if isinstance(row, Mapping) and row.get("baseline_success") is False
    ]
    success_rows = [
        row
        for row in request_residuals
        if isinstance(row, Mapping) and row.get("baseline_success") is True
    ]
    if len(failure_rows) != support or any(
        row.get("family") != target_family for row in failure_rows
    ):
        raise PermissionError("production family-slot failure scope drifted")
    if len(success_rows) != EXPECTED_TRAIN_SUCCESSES or any(
        row.get("context") != {} for row in success_rows
    ):
        raise PermissionError("production success-control scope drifted")
    profile_hashes = tuple(
        sorted(
            {
                str((row.get("context") or {}).get("action_context_profile_hash") or "")
                for row in failure_rows
                if isinstance(row.get("context"), Mapping)
            }
        )
    )
    if not profile_hashes or any(not _is_sha256(value) for value in profile_hashes):
        raise PermissionError("production family slot has malformed profile references")
    expected_profile_evidence_hash = stable_hash(
        {
            "profile_references": [
                {
                    "profile_hash": profile_hash,
                    "profile_payload_hash": stable_hash(
                        dict(action_profiles[profile_hash])
                    ),
                }
                for profile_hash in profile_hashes
                if isinstance(action_profiles.get(profile_hash), Mapping)
            ]
        }
    )
    if (
        any(profile_hash not in action_profiles for profile_hash in profile_hashes)
        or contract.get("profile_reference_count") != len(profile_hashes)
        or profile_summary.get("profile_reference_count") != len(profile_hashes)
        or profile_summary.get("profile_evidence_hash") != profile_evidence_hash
        or expected_profile_evidence_hash != profile_evidence_hash
    ):
        raise PermissionError("production V2 profile summary drifted")

    failed_count = portable.get("failed_primitive_count")
    failed_set_hash = str(portable.get("failed_primitive_set_hash") or "")
    if (
        isinstance(failed_count, bool)
        or not isinstance(failed_count, int)
        or failed_count < 0
        or not _is_sha256(failed_set_hash)
        or portable.get("failed_primitive_values_disclosed") is not False
        or profile_summary.get("failed_primitive_count") != failed_count
        or profile_summary.get("failed_primitive_set_hash") != failed_set_hash
        or plan_row.get("failed_primitive_count") != failed_count
        or plan_row.get("failed_primitive_set_hash") != failed_set_hash
    ):
        raise PermissionError("production V2 failed-primitive summary drifted")

    recommended_rows = _parse_profile_primitives(
        [portable.get("recommended_artifact")],
        label="recommended artifact",
    )
    recommended = recommended_rows[0]
    if (
        not recommended.reusable
        or not recommended.kind.startswith("artifact_")
        or recommended.model_payload()
        not in [row.model_payload() for row in preferred]
    ):
        raise PermissionError("production V2 recommended artifact is invalid")
    blueprint = str(portable.get("required_artifact_workflow_blueprint") or "")
    lowered_blueprint = blueprint.lower()
    positions = [
        lowered_blueprint.find(token)
        for token in ("read", "parse", "update", "serialize", "write")
    ]
    if (
        recommended.value not in blueprint
        or any(position < 0 for position in positions)
        or positions != sorted(positions)
        or portable.get("recommended_artifact_value_must_be_mentioned_exactly")
        is not True
    ):
        raise PermissionError("production V2 artifact blueprint drifted")

    output_schema = request.get("output_schema")
    hypothesis_schema = (
        output_schema.get("properties", {}).get("hypothesis")
        if isinstance(output_schema, Mapping)
        and isinstance(output_schema.get("properties"), Mapping)
        else None
    )
    expected_trigger = {
        "all_of": [{"key": "family", "op": "eq", "value": target_family}],
        "any_of": [],
        "none_of": [],
    }
    expected_anti_trigger = {"all_of": [], "any_of": [], "none_of": []}
    if (
        not isinstance(hypothesis_schema, Mapping)
        or hypothesis_schema.get("trigger") != expected_trigger
        or hypothesis_schema.get("anti_trigger") != expected_anti_trigger
    ):
        raise PermissionError("production V2 trigger schema drifted")
    slot = FamilyProposalSlot(
        index=index,
        target_family=target_family,
        target_failure_count=support,
        target_profile_hashes=profile_hashes,
        profile_evidence_hash=profile_evidence_hash,
        preferred_primitives=preferred,
        failed_primitives=(),
        prior_use_count=int(plan_row.get("prior_family_use_count") or 0),
        declared_failed_primitive_count=failed_count,
        declared_failed_primitive_set_hash=failed_set_hash,
    )
    if contract.get("slot_id") != slot.formation_payload()["slot_id"]:
        raise PermissionError("production family-slot identifier drifted")
    return slot


def _parse_profile_primitives(
    payload: Any,
    *,
    label: str,
) -> tuple[ProfilePrimitive, ...]:
    if not isinstance(payload, list):
        raise PermissionError(
            f"production {label} profile primitives are malformed"
        )
    rows: list[ProfilePrimitive] = []
    for raw in payload:
        if not isinstance(raw, Mapping):
            raise PermissionError(
                f"production {label} profile primitive is malformed"
            )
        kind = str(raw.get("kind") or "")
        value = str(raw.get("value") or "")
        count = raw.get("train_failure_evidence_count")
        if not kind or not value or isinstance(count, bool) or not isinstance(
            count, int
        ) or count <= 0:
            raise PermissionError(
                f"production {label} profile primitive is incomplete"
            )
        row = ProfilePrimitive(
            kind=kind,
            value=value,
            train_failure_evidence_count=count,
        )
        if raw.get("reusable_across_same_family_failures") is not row.reusable:
            raise PermissionError(
                f"production {label} profile primitive reuse flag drifted"
            )
        rows.append(row)
    return tuple(rows)


def _bind_production_completion(
    audit: dict[str, Any],
    *,
    slot: FamilyProposalSlot,
    program: HypothesisProgram,
    completed_payload: Any,
) -> None:
    if not isinstance(completed_payload, Mapping):
        raise PermissionError("production family-slot completion is malformed")
    formation = slot.formation_payload()
    expected = {
        "policy": TARGET_PROPOSAL_FORMATION_POLICY_VERSION,
        "slot_id": formation["slot_id"],
        "target_family": slot.target_family,
        "target_family_hash": slot.target_family_hash,
        "profile_evidence_hash": slot.profile_evidence_hash,
        "preferred_primitive_count": len(slot.preferred_primitives),
        "preferred_primitive_set_hash": formation[
            "preferred_primitive_set_hash"
        ],
        "failed_primitive_count": slot.failed_primitive_count,
        "failed_primitive_set_hash": formation["failed_primitive_set_hash"],
        "candidate_hash": program.payload_hash,
        "response_rejected_by_diversity": False,
        "proposal_retry_by_diversity": False,
    }
    for key, value in expected.items():
        if completed_payload.get(key) != value:
            raise PermissionError(
                f"production family-slot completion drifted: {key}"
            )
    raw_failed_binding_count = completed_payload.get(
        "failed_profile_binding_count"
    )
    if (
        isinstance(raw_failed_binding_count, bool)
        or not isinstance(raw_failed_binding_count, int)
        or raw_failed_binding_count < 0
    ):
        raise PermissionError(
            "production failed-profile binding count is malformed"
        )
    production_failed_binding_count = raw_failed_binding_count
    audit.update(
        {
            "production_slot_completion_hash": stable_hash(
                dict(completed_payload)
            ),
            "production_candidate_matched_target_support": int(
                completed_payload.get("candidate_matched_target_support") or 0
            ),
            "production_matched_failure_family_count": int(
                completed_payload.get("matched_family_count") or 0
            ),
            "production_profile_binding_count": int(
                completed_payload.get("profile_binding_count") or 0
            ),
            "production_failed_profile_binding_count": (
                production_failed_binding_count
            ),
            "production_portable_delta_kinds": list(
                completed_payload.get("portable_delta_kinds") or []
            ),
            # V2 deliberately withholds failed primitive values from the model
            # request.  The production kernel retains them locally and this
            # bound completion audit is therefore the authoritative source for
            # the unchanged failed-primitive avoidance acceptance check.
            "failed_primitive_binding_count": production_failed_binding_count,
            "failed_primitive_binding_set_hash": stable_hash(
                {
                    "candidate_hash": program.payload_hash,
                    "failed_primitive_set_hash": slot.failed_primitive_set_hash,
                    "failed_profile_binding_count": (
                        production_failed_binding_count
                    ),
                }
            ),
        }
    )


def _audit_slot_program(
    program: HypothesisProgram,
    *,
    slot: FamilyProposalSlot,
    failures: Sequence[ResidualExample],
    action_profiles: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    signature: list[bool | None] = []
    matched_rows: list[ResidualExample] = []
    anti_block_count = 0
    for row in failures:
        try:
            matched = program.matches(row.features)
        except (TypeError, ValueError, OverflowError):
            signature.append(None)
            continue
        signature.append(matched)
        if matched:
            matched_rows.append(row)
        if row.family == slot.target_family:
            try:
                trigger_matches = program.trigger.matches(row.features)
                anti_matches = (
                    not program.anti_trigger.is_empty
                    and program.anti_trigger.matches(row.features)
                )
            except (TypeError, ValueError, OverflowError):
                anti_matches = True
                trigger_matches = True
            if trigger_matches and anti_matches:
                anti_block_count += 1

    generic = _action_delta_audit_row(
        program,
        residuals=failures,
        profiles=action_profiles,
    )
    action_texts = [
        " ".join(
            (
                str(action.target),
                action.value
                if isinstance(action.value, str)
                else json.dumps(action.value, sort_keys=True, ensure_ascii=True),
            )
        ).strip()
        for action in program.action_graph
    ]
    bound_preferred = {
        primitive.primitive_hash
        for primitive in slot.preferred_primitives
        if any(_primitive_mentioned(primitive, text) for text in action_texts)
    }
    bound_failed = {
        primitive.primitive_hash
        for primitive in slot.failed_primitives
        if any(_primitive_mentioned(primitive, text) for text in action_texts)
    }
    grounded_delta_kinds = set(generic["observed_delta_kinds"])
    if _concrete_profile_command_present(
        slot.preferred_primitives, action_texts
    ):
        grounded_delta_kinds.add("concrete_local_tool_command")
    if _profile_artifact_manipulation_present(
        slot.preferred_primitives, action_texts
    ):
        grounded_delta_kinds.add("artifact_internal_manipulation")
    matched_family_hashes = sorted(
        {stable_hash({"family": row.family}) for row in matched_rows}
    )
    matched_target_support = sum(
        row.family == slot.target_family for row in matched_rows
    )
    return {
        "hypothesis_id_hash": stable_hash({"hypothesis_id": program.id}),
        "hypothesis_hash": program.payload_hash,
        "slot_index": slot.index,
        "target_family_hash": slot.target_family_hash,
        "activation_signature_hash": stable_hash(
            {"train_failure_activation_signature": signature}
        ),
        "matched_failure_count": len(matched_rows),
        "matched_target_support": matched_target_support,
        "matched_failure_family_count": len(matched_family_hashes),
        "matched_failure_family_hashes": matched_family_hashes,
        "matched_target_family_only": (
            bool(matched_rows)
            and matched_family_hashes == [slot.target_family_hash]
        ),
        "target_anti_trigger_self_block_count": anti_block_count,
        "validation_issues": program.validate(),
        "generic_observed_delta_kinds": list(
            generic["observed_delta_kinds"]
        ),
        "profile_grounded_delta_kinds": sorted(grounded_delta_kinds),
        "profile_environment_binding_count": len(bound_preferred),
        "profile_environment_binding_set_hash": stable_hash(
            {"primitive_hashes": sorted(bound_preferred)}
        ),
        "failed_primitive_binding_count": len(bound_failed),
        "failed_primitive_binding_set_hash": stable_hash(
            {"primitive_hashes": sorted(bound_failed)}
        ),
        "generic_environment_binding_count": sum(
            int(row["environment_binding_count"])
            for row in generic["action_audits"]
        ),
        "new_environment_primitive_count": int(
            generic["new_environment_primitive_count"]
        ),
        "restatement_only_action_count": int(
            generic["instruction_restatement_only_action_count"]
        ),
        "vague_placeholder_action_count": int(
            generic["vague_placeholder_action_count"]
        ),
        "restatement_risk": bool(generic["restatement_risk"]),
    }


def _primitive_mentioned(primitive: ProfilePrimitive, text: str) -> bool:
    value = primitive.value.lower().strip()
    lowered = text.lower()
    if not value:
        return False
    if value.startswith("-") or "/" in value or "." in value:
        return value in lowered
    return bool(re.search(rf"(?<![a-z0-9_]){re.escape(value)}(?![a-z0-9_])", lowered))


def _concrete_profile_command_present(
    primitives: Sequence[ProfilePrimitive],
    action_texts: Sequence[str],
) -> bool:
    executables = [
        row
        for row in primitives
        if row.kind == "executable"
    ]
    qualifiers = [
        row
        for row in primitives
        if row.kind
        in {
            "artifact_command_path",
            "artifact_task_local_path",
            "artifact_copied_file",
            "artifact_environment_source_file",
        }
    ]
    for text in action_texts:
        if not any(_primitive_mentioned(row, text) for row in executables):
            continue
        if any(_primitive_mentioned(row, text) for row in qualifiers):
            return True
        if re.search(r"(?:^|[;`])\s*[A-Za-z0-9_./+-]+\s+--?[A-Za-z0-9-]+", text):
            return True
    return False


def _profile_artifact_manipulation_present(
    primitives: Sequence[ProfilePrimitive],
    action_texts: Sequence[str],
) -> bool:
    artifacts = [
        row
        for row in primitives
        if row.kind
        in {
            "artifact_command_path",
            "artifact_task_local_path",
            "artifact_copied_file",
            "artifact_environment_source_file",
        }
    ]
    operation = re.compile(
        r"\b(?:read|write|update|set|replace|append|merge|fill|render|parse|serialize)\b|"
        r"\b[A-Za-z_][A-Za-z0-9_.]*\([^\n)]*\)",
        re.IGNORECASE,
    )
    return any(
        operation.search(text)
        and any(_primitive_mentioned(row, text) for row in artifacts)
        for text in action_texts
    )


def _read_target_protocol(
    path: str | Path,
) -> tuple[Mapping[str, Any], str]:
    protocol_path = _contained_file(
        Path(path),
        anchor=Path(path).expanduser().resolve().parent,
        label="target protocol",
    )
    protocol = _read_json_object(protocol_path, label="target protocol")
    if protocol.get("protocol_version") != TARGET_PROTOCOL_VERSION:
        raise ValueError("proposal diagnostic requires V3.17 protocol")
    if protocol.get("sealed_test_content_accessed") is not False:
        raise ValueError("target protocol already accessed sealed test content")
    if protocol.get("raw_content_persisted") is not False:
        raise ValueError("target protocol permits raw content persistence")
    execution = protocol.get("execution")
    evolution = protocol.get("evolution")
    promotion = protocol.get("promotion")
    if not all(
        isinstance(row, Mapping)
        for row in (execution, evolution, promotion)
    ):
        raise ValueError("target protocol contracts are missing")
    assert isinstance(execution, Mapping)
    assert isinstance(evolution, Mapping)
    if execution.get("proposal_formation_policy") != (
        TARGET_PROPOSAL_FORMATION_POLICY_VERSION
    ):
        raise ValueError("target proposal formation policy mismatch")
    if execution.get("train_action_design_policy") != (
        TRAIN_ACTION_DESIGN_POLICY_VERSION
    ):
        raise ValueError("target action-design policy mismatch")
    if evolution.get("proposal_candidates_per_generation") != (
        REQUIRED_SLOT_COUNT
    ):
        raise ValueError("target protocol does not declare three proposal slots")
    return protocol, stable_hash(protocol)


def _public_train_items(
    root: Path,
    manifest: SplitManifest,
) -> dict[str, BenchmarkItem]:
    items: dict[str, BenchmarkItem] = {}
    tasks_root = _resolve_directory(root / "tasks", label="benchmark tasks root")
    for item_id in manifest.train_ids:
        family = str(manifest.family_by_id[item_id])
        _require_safe_component(family, label="TRAIN family")
        _require_safe_component(item_id, label="TRAIN item")
        instance_dir = _resolve_directory(
            tasks_root / family / item_id,
            label="public TRAIN task",
            anchor=tasks_root,
        )
        task_toml = _contained_file(
            instance_dir / "task.toml",
            anchor=instance_dir,
            label="public TRAIN task metadata",
        )
        config = tomllib.loads(task_toml.read_text(encoding="utf-8"))
        metadata = (
            config.get("metadata", {})
            if isinstance(config.get("metadata"), Mapping)
            else {}
        )
        instruction = _contained_file(
            instance_dir / "instruction.md",
            anchor=instance_dir,
            label="public TRAIN instruction",
        )
        environment = instance_dir / "environment"
        environment_file_count = _public_environment_file_count(
            environment, anchor=instance_dir
        )
        items[item_id] = BenchmarkItem(
            id=item_id,
            family=family,
            features={
                "benchmark": "skilllearnbench",
                "family": family,
                "category": str(metadata.get("category") or ""),
                "difficulty": str(metadata.get("difficulty") or ""),
                "tags": tuple(
                    str(value) for value in metadata.get("tags", [])
                ),
                "environment_file_count": environment_file_count,
                "has_container_environment": (
                    environment / "Dockerfile"
                ).is_file(),
            },
            content_ref=str(instruction.relative_to(root)),
            verifier_ref_hash=stable_hash(
                {
                    "item_id_hash": stable_hash({"item_id": item_id}),
                    "verifier_content_accessed": False,
                }
            ),
        )
    return items


def _source_train_artifact_paths(
    upstream_root: Path,
    *,
    family: str,
    item_id: str,
) -> tuple[Path, Path, Path]:
    _require_safe_component(family, label="source TRAIN family")
    _require_safe_component(item_id, label="source TRAIN item")
    item_root = _resolve_directory(
        upstream_root / family / item_id,
        label="source TRAIN item artifacts",
        anchor=upstream_root,
    )
    result_candidates: list[Path] = []
    for trial_dir in item_root.iterdir():
        if trial_dir.is_symlink() or not trial_dir.is_dir():
            continue
        if not trial_dir.name.startswith("v2_policy_off_"):
            continue
        candidate = trial_dir / "result.json"
        if candidate.is_file() and not candidate.is_symlink():
            result_candidates.append(candidate)
    if len(result_candidates) != 1:
        raise PermissionError(
            "source TRAIN item does not have exactly one policy-off result"
        )
    result_path = _contained_file(
        result_candidates[0],
        anchor=upstream_root,
        label="source TRAIN result",
    )
    trial_dir = result_path.parent
    trace_path = _contained_file(
        trial_dir / "agent" / "codex.txt",
        anchor=upstream_root,
        label="source TRAIN Codex trace",
    )
    receipt_path = _contained_file(
        trial_dir / "agent" / "codex_action_budget_receipt.json",
        anchor=upstream_root,
        label="source TRAIN action-budget receipt",
    )
    return result_path, trace_path, receipt_path


def _validate_source_trial(
    *,
    result: Mapping[str, Any],
    receipt: Mapping[str, Any],
    trace_path: Path,
    family: str,
    item_id: str,
    expected_model: str,
) -> str:
    if result.get("task_id") != f"{family}/{item_id}":
        raise PermissionError("source TRAIN result task identity mismatch")
    if result.get("skill_config") != "no_skill":
        raise PermissionError("source TRAIN result is not policy-off")
    if result.get("agent") != "codex" or result.get("model") != expected_model:
        raise PermissionError("source TRAIN result model identity mismatch")
    if result.get("agent_exit") != 0 or result.get("agent_timed_out") is not False:
        raise PermissionError("source TRAIN agent did not complete cleanly")
    if result.get("verifier_exit") != 0:
        raise PermissionError("source TRAIN verifier infrastructure failed")
    if not isinstance(result.get("passed"), bool):
        raise PermissionError("source TRAIN result outcome is malformed")
    receipt_hash = receipt.get("receipt_hash")
    receipt_without_hash = dict(receipt)
    receipt_without_hash.pop("receipt_hash", None)
    if not _is_sha256(receipt_hash) or stable_hash(receipt_without_hash) != receipt_hash:
        raise PermissionError("source action-budget receipt hash mismatch")
    trace_hash = _sha256_file(trace_path)
    required_receipt = {
        "trace_sha256": trace_hash,
        "turn_completed_observed": True,
        "turn_completed_count": 1,
        "turn_failed_count": 0,
        "invalid_terminal_usage_count": 0,
        "invalid_action_event_count": 0,
        "token_usage_complete": True,
        "agent_processes_exit_confirmed": True,
        "process_task_scan_complete": True,
        "residual_process_count": 0,
    }
    for key, expected in required_receipt.items():
        if receipt.get(key) != expected:
            raise PermissionError(
                f"source action-budget receipt mismatch: {key}"
            )
    return trace_hash


def _public_environment_file_count(path: Path, *, anchor: Path) -> int:
    if not path.exists():
        return 0
    environment = _resolve_directory(
        path, label="public TRAIN environment", anchor=anchor
    )
    count = 0
    for candidate in environment.rglob("*"):
        if candidate.is_symlink():
            raise PermissionError("public TRAIN environment contains a symlink")
        if candidate.is_file():
            _contained_file(
                candidate,
                anchor=environment,
                label="public TRAIN environment file",
            )
            count += 1
    return count


def _read_json_object(path: Path, *, label: str) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def _resolve_directory(
    path: Path,
    *,
    label: str,
    anchor: Path | None = None,
) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise PermissionError(f"{label} symlinks are forbidden")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} is not a directory")
    if anchor is not None:
        resolved_anchor = anchor.expanduser().resolve(strict=True)
        _require_within(resolved, resolved_anchor, label=label)
        _reject_symlink_components(expanded, resolved_anchor, label=label)
    return resolved


def _contained_file(path: Path, *, anchor: Path, label: str) -> Path:
    resolved_anchor = anchor.expanduser().resolve(strict=True)
    _reject_symlink_components(path, resolved_anchor, label=label)
    resolved = path.expanduser().resolve(strict=True)
    _require_within(resolved, resolved_anchor, label=label)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file")
    return resolved


def _reject_symlink_components(path: Path, anchor: Path, *, label: str) -> None:
    try:
        relative = path.expanduser().absolute().relative_to(anchor.absolute())
    except ValueError as exc:
        raise PermissionError(f"{label} escapes its allowed root") from exc
    current = anchor
    for part in relative.parts:
        if part in {"", ".", ".."}:
            raise PermissionError(f"{label} has an unsafe path component")
        current = current / part
        if current.is_symlink():
            raise PermissionError(f"{label} symlinks are forbidden")


def _require_within(path: Path, anchor: Path, *, label: str) -> None:
    try:
        path.relative_to(anchor)
    except ValueError as exc:
        raise PermissionError(f"{label} escapes its allowed root") from exc


def _require_safe_component(value: str, *, label: str) -> None:
    if not _SAFE_COMPONENT.fullmatch(value) or value in {".", ".."}:
        raise PermissionError(f"{label} contains an unsafe path component")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _emit_boundary_event(
    sink: EventSink,
    *,
    event: str,
    trace_id: str,
    payload: Mapping[str, Any],
) -> None:
    sink.emit(
        Event(
            event=event,
            stage="benchmark.skilllearn.train_proposal_diagnostic",
            trace_id=trace_id,
            payload={**dict(payload), **_COMMON_BOUNDARY_FLAGS},
        )
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    temporary.replace(destination)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild frozen V3.15 TRAIN evidence and spend only three V3.17 "
            "proposal slots; no task agent, evaluator, validation, or sealed access."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--source-train-receipt", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.events.exists() or args.out.exists():
        raise FileExistsError(
            "proposal diagnostic events/out paths must be fresh"
        )
    from ..provider_chain import build_proposal_model, configured_provider_chain
    from ..secure_env import configured_model, load_dotenv, map_legacy_model_env

    load_dotenv(args.env_file)
    map_legacy_model_env()
    protocol, _ = _read_target_protocol(args.protocol)
    if configured_model() != str(protocol["model"]):
        raise RuntimeError("configured proposal model differs from V3.17")
    if list(configured_provider_chain()) != list(
        protocol["proposal_provider_chain"]
    ):
        raise RuntimeError("configured proposal provider chain differs from V3.17")
    sink = _DiagnosticEventSink(JsonlEventSink(args.events))
    proposal_model = build_proposal_model(
        event_sink=sink,
        max_tokens=int(protocol["execution"]["proposal_response_max_tokens"]),
    )
    try:
        report = run_train_proposal_diagnostic(
            root=args.root,
            manifest_path=args.manifest,
            source_run_root=args.source_run_root,
            source_train_receipt=args.source_train_receipt,
            protocol_path=args.protocol,
            proposal_model=proposal_model,
            event_sink=sink,
        )
    except HypothesisProposalCallError:
        print(
            json.dumps(
                {
                    "status": "proposal_diagnostic_failed",
                    "error_type": "proposal_model_or_response_failure",
                    "raw_error_persisted": False,
                    "secret_value_persisted": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    _verify_live_three_slot_proposal_flow(
        sink.persisted_events,
        report=report,
        protocol=protocol,
    )
    _write_json_atomic(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))
    if not report["diagnostic_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
