from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from ..archive import PolicyArchive
from ..events import Event, EventSink, JsonlEventSink, MemoryEventSink, NullEventSink
from ..evolution import (
    TYPED_RECIPE_PROPOSAL_FORMATION_POLICY_VERSION,
    EvolutionKernel,
)
from ..models import (
    HypothesisKind,
    HypothesisProgram,
    HypothesisStatus,
    stable_hash,
)
from ..proposer import (
    HypothesisProposalCallError,
    StructuredHypothesisProposer,
)
from ..splits import SplitManifest
from ..typed_operator_grammar import (
    MAX_REGISTERED_ARTIFACTS_PER_FAMILY,
    TrialTraceEvidence,
    TypedRecipeSelectionSnapshot,
    TypedSelectionFreezeAuthorization,
    TypedSelectionSnapshotLedger,
    TypedProgramBindingRegistry,
    TYPED_SELECTION_FREEZE_AUTHORIZATION_VERSION,
    build_family_capability_graph,
    freeze_typed_recipe_selection_snapshot,
    freeze_typed_selection_snapshot_ledger,
)
from ..validation import (
    CheckResult,
    RecursiveValidationEngine,
    ValidationContext,
    backend_action_contract_issues,
    build_trigger_feature_catalog,
)
from .skilllearn_compiler import (
    SKILLLEARN_ALLOWED_ACTION_OPERATIONS,
    SkillLearnProgramCompiler,
    _lower_skilllearn_program,
)
from .skilllearnbench import SkillLearnBenchAdapter
from .train_proposal_diagnostic import (
    ReconstructedTrainEvidence,
    _read_json_object,
    _sha256_file,
    reconstruct_v315_train_evidence,
)
from .typed_operator_feasibility import (
    _all_primitive_values,
    _extract_all_train_trials,
    _implementation_file_set_hash,
    _leaf_strings,
)


TYPED_SELECTION_INTEGRATION_VERSION = (
    "v315_train_typed_selection_production_integration_v1"
)
TYPED_SELECTION_RESULT_RECEIPT_VERSION = (
    "typed_selection_integration_result_receipt_v1"
)
TYPED_SELECTION_EVALUATOR_EPOCH = "typed-selection-integration-v1"
REQUIRED_SLOT_COUNT = 3
MINIMUM_FAMILY_SUPPORT = 2

_TAMPER_PROBE_IDS = (
    "root_response_extra_field",
    "root_response_unknown_recipe",
    "root_response_non_string_recipe",
    "shared_batch_missing_candidate",
    "shared_batch_reordered",
    "shared_program_executable_mutation",
    "compiler_unbound_program",
    "compiler_executable_mutation",
    "recursive_repair_unbound_parent",
    "direct_repair_unbound_parent",
    "snapshot_ledger_rebinding",
    "runtime_skill_content_mutation",
)

_EXPECTED_PREDECISION_EVENT_COUNTS: Mapping[str, int] = {
    "typed_selection_integration_started": 1,
    "typed_selection_upstream_feasibility_verified": 1,
    "typed_selection_train_evidence_reconstructed": 1,
    "typed_selection_snapshots_frozen": 1,
    "typed_selection_protocol_loader_verified": 1,
    "proposal_typed_recipe_plan_created": 3,
    "typed_recipe_selection_requested": 11,
    "typed_recipe_selection_materialized": 8,
    "proposal_typed_recipe_slot_completed": 9,
    "proposal_family_slot_usage_recorded": 2,
    "proposal_typed_recipes_completed": 3,
    "typed_recipe_selection_replayed": 3,
    "proposal_family_slot_usage_replayed": 1,
    "typed_shared_proposal_batch_validated": 1,
    "archive_typed_selection_recorded": 4,
    "archive_hypothesis_registered": 4,
    "archive_hypothesis_status_changed": 2,
    "hypothesis_validation_node_evaluated": 4,
    "recursive_validation_completed": 2,
    "typed_compiler_binding_verified": 1,
    "typed_selection_tamper_probes_completed": 1,
}

_ACCEPTANCE_PREDICATES: Mapping[str, str] = {
    "upstream_feasibility_binding_passed": (
        "the frozen typed-operator preregistration, versioned result receipt, "
        "report, event ledger and completed decision lock match their external "
        "stable hashes, file hashes, decision hash and report hash"
    ),
    "source_reconstruction_passed": (
        "the bound V3.15 source receipt reconstructs exactly 38 TRAIN "
        "observations, 32 failures, 6 success controls and 38 complete "
        "receipt-bound action traces without source-agent reexecution"
    ),
    "snapshot_external_ledger_passed": (
        "three freshly rebuilt frozen snapshots match the externally recorded "
        "ordered target-family, graph, model-catalog, graph-set and "
        "model-catalog-set commitments from the formal feasibility decision"
    ),
    "production_protocol_loader_path_passed": (
        "a structurally accepted typed paper protocol traverses the same "
        "PaperProtocol.read and CLI snapshot loader used by production; the "
        "diagnostic-only loader returns a non-authorized ledger rejected by "
        "the evolution boundary, while only exact formal result-receipt "
        "verification can attach protocol-freeze authority"
    ),
    "production_selection_path_passed": (
        "the shared production snapshot loader and current EvolutionKernel "
        "typed policy invoke "
        "StructuredHypothesisProposer.select_typed_recipe for all three slots, "
        "materializes three harness-owned programs and registers their exact "
        "snapshot/request/response bindings"
    ),
    "closed_model_surface_passed": (
        "each local selector request exposes only the closed recipe_id schema "
        "and safe catalog, with no raw artifact locator, observed primitive, "
        "residual, free-text action or model-authored primitive field"
    ),
    "shared_receipt_reuse_passed": (
        "the production shared-proposal validator returns the exact candidate, "
        "snapshot and binding receipt with zero new selector calls"
    ),
    "typed_recursive_repair_passed": (
        "one forced root-only static failure reselects a different recipe from "
        "the same frozen snapshot, records a parent-bound child, never falls "
        "back to generic free-text repair, and the real frozen-archive reader "
        "restores both status-invariant bindings plus their selection history"
    ),
    "compiler_provenance_passed": (
        "the typed compiler boundary returns the same registry binding hashes "
        "for the three roots and repaired child and persists their binding set, "
        "snapshot-ledger, compile-manifest and full compiler-event-set "
        "commitments plus complete binding coverage through prompt lowering and "
        "compile events; all 38 TRAIN item source trees receive verified runtime "
        "receipts and a post-compile content mutation fails closed, with zero "
        "action-contract issue"
    ),
    "fixed_tamper_probes_passed": (
        "all twelve preregistered malformed-response, shared-batch, executable, "
        "compiler, runtime-source and unbound-parent repair probes fail closed"
    ),
    "deterministic_replay_passed": (
        "a second production candidate request returns the exact same ordered "
        "programs from the proposer replay registry with zero new selector calls"
    ),
    "multi_generation_diversity_passed": (
        "after all first-round selections are written to an attempt ledger "
        "independent of behavior or promotion archival, the next production "
        "round selects three different recipes with selection_round two and "
        "the exact prior-recipe exclusion bound into every request and registry "
        "receipt; an actual repair of a next-generation root also excludes both "
        "prior recipes and cannot cycle back to the first-generation recipe"
    ),
    "offline_boundary_contract_passed": (
        "the complete predecision event ledger has the preregistered event "
        "cardinalities and records only stored TRAIN reconstruction, local "
        "deterministic selection, static repair and compiler-contract work; no "
        "live model, task backend, evaluator, held-out split, verifier or "
        "promotion access occurs"
    ),
}
_ACCEPTANCE_KEYS = frozenset(_ACCEPTANCE_PREDICATES)

_BOUNDARY_FLAGS: Mapping[str, bool] = {
    "source_agent_trials_reexecuted": False,
    "stored_offline_train_outcomes_used": True,
    "local_deterministic_selector_used": True,
    "local_contract_validation_used": True,
    "live_model_invoked": False,
    "live_task_backend_invoked": False,
    "live_evaluator_invoked": False,
    "validation_split_accessed": False,
    "test_split_accessed": False,
    "sealed_split_accessed": False,
    "verifier_content_accessed": False,
    "promotion_policy_evaluated": False,
    "secret_value_persisted": False,
    "raw_content_persisted": False,
}


@dataclass(frozen=True)
class FrozenTypedSelectionLedger:
    """Externally bound TRAIN evidence and snapshots reusable by production.

    The evidence object contains TRAIN-only harness material in memory.  Only
    :meth:`safe_payload` is suitable for persistence.
    """

    evidence: ReconstructedTrainEvidence
    trials: tuple[TrialTraceEvidence, ...]
    snapshots: tuple[TypedRecipeSelectionSnapshot, ...]
    production_snapshot_ledger: TypedSelectionSnapshotLedger
    upstream_binding_hash: str
    trial_evidence_hash: str
    graph_set_hash: str
    model_catalog_set_hash: str
    freeze_authorization: TypedSelectionFreezeAuthorization | None = None

    @property
    def snapshot_set_hash(self) -> str:
        return stable_hash(
            {
                "snapshot_hashes": [
                    row.snapshot_hash for row in self.snapshots
                ]
            }
        )

    @property
    def ledger_hash(self) -> str:
        return stable_hash(self.safe_payload(include_hash=False))

    def require_freeze_authorization(
        self,
    ) -> TypedSelectionFreezeAuthorization:
        authorization = self.freeze_authorization
        if authorization is None:
            raise PermissionError(
                "typed selection ledger is diagnostic-only and cannot freeze or execute"
            )
        issues = authorization.validate_for(
            self.production_snapshot_ledger
        )
        if issues:
            raise PermissionError(
                f"typed selection freeze authorization is invalid: {list(issues)}"
            )
        return authorization

    def safe_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "upstream_binding_hash": self.upstream_binding_hash,
            "source_train_receipt_hash": (
                self.evidence.source_train_receipt_hash
            ),
            "trial_evidence_hash": self.trial_evidence_hash,
            "graph_set_hash": self.graph_set_hash,
            "model_catalog_set_hash": self.model_catalog_set_hash,
            "snapshot_set_hash": self.snapshot_set_hash,
            "production_snapshot_ledger_hash": (
                self.production_snapshot_ledger.ledger_hash
            ),
            "snapshot_hashes": [
                row.snapshot_hash for row in self.snapshots
            ],
            "target_family_hashes": [
                row.graph.target_family_hash for row in self.snapshots
            ],
            "graph_hashes": [
                row.expected_graph_hash for row in self.snapshots
            ],
            "model_catalog_hashes": [
                row.expected_model_catalog_hash for row in self.snapshots
            ],
            "train_observation_count": len(self.evidence.residuals),
            "train_failure_count": len(self.evidence.failures),
            "train_success_control_count": len(
                self.evidence.success_controls
            ),
            "trace_count": len(self.trials),
            "source_agent_trials_reexecuted": 0,
            "raw_content_persisted": False,
        }
        if include_hash:
            payload["ledger_hash"] = self.ledger_hash
        return payload


class _BoundaryAuditEventSink:
    def __init__(self, delegate: EventSink) -> None:
        self.delegate = delegate
        self.events: list[dict[str, Any]] = []

    def emit(self, event: Event) -> None:
        audited = Event(
            event=event.event,
            stage=event.stage,
            trace_id=event.trace_id,
            payload={**dict(event.payload), **_BOUNDARY_FLAGS},
        )
        row = audited.to_dict()
        self.events.append(row)
        self.delegate.emit(audited)


class _OfflineRecipeSelector:
    """Deterministic local selector exercising the production model adapter."""

    def __init__(self, expected_ledger_hash: str) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responses: list[dict[str, str]] = []
        self.expected_ledger_hash = expected_ledger_hash

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def complete(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        request = dict(payload)
        _validate_closed_selector_request(
            request,
            expected_ledger_hash=self.expected_ledger_hash,
        )
        recipe_ids = request["output_schema"]["properties"]["recipe_id"][
            "enum"
        ]
        response = {"recipe_id": recipe_ids[0]}
        self.requests.append(request)
        self.responses.append(response)
        return response


class _FixedResponseSelector:
    def __init__(self, response: Any, *, expected_ledger_hash: str) -> None:
        self.response = response
        self.call_count = 0
        self.expected_ledger_hash = expected_ledger_hash

    def complete(self, payload: Mapping[str, Any]) -> Any:
        _validate_closed_selector_request(
            dict(payload),
            expected_ledger_hash=self.expected_ledger_hash,
        )
        self.call_count += 1
        return self.response


class _RootOnlyFailureCheck:
    name = "typed_integration_forced_root_only_failure"

    def __init__(self, root_id: str) -> None:
        self.root_id = root_id

    def evaluate(
        self,
        program: HypothesisProgram,
        context: ValidationContext,
    ) -> CheckResult:
        passed = program.id != self.root_id
        return CheckResult(
            check=self.name,
            passed=passed,
            reason=(
                "alternate_typed_recipe_selected"
                if passed
                else "root_recipe_rejected_for_integration_probe"
            ),
            evidence={"program_hash": program.payload_hash},
        )


def run_typed_selection_integration(
    *,
    root: str | Path,
    manifest_path: str | Path,
    source_run_root: str | Path,
    source_train_receipt: str | Path,
    preregistration_path: str | Path,
    report_path: str | Path,
    events_path: str | Path,
) -> dict[str, Any]:
    preregistration = _read_preregistration(preregistration_path)
    canonical = _canonical_decision_paths(
        preregistration,
        preregistration_path=preregistration_path,
    )
    report = Path(report_path).expanduser().resolve()
    events = Path(events_path).expanduser().resolve()
    if report != canonical["report"] or events != canonical["events"]:
        raise PermissionError(
            "formal typed-selection integration must use canonical paths"
        )
    if (
        report.exists()
        or events.exists()
        or canonical["result_receipt"].exists()
    ):
        raise FileExistsError(
            "formal typed-selection integration output already exists"
        )
    _reserve_decision_lock(
        canonical["decision_lock"],
        preregistration_hash=stable_hash(preregistration),
    )
    try:
        result = _compute_integration(
            root=root,
            manifest_path=manifest_path,
            source_run_root=source_run_root,
            source_train_receipt=source_train_receipt,
            preregistration_path=preregistration_path,
            event_sink=JsonlEventSink(events),
        )
        _write_json_atomic(report, result)
        _write_json_atomic(
            canonical["decision_lock"],
            {
                "lock_version": (
                    "typed_selection_integration_decision_lock_v1"
                ),
                "decision_ordinal": 1,
                "state": "completed",
                "preregistration_hash": stable_hash(preregistration),
                "decision_hash": result["decision_hash"],
                "report_hash": result["report_hash"],
                "integration_passed": result["integration_passed"],
                "raw_content_persisted": False,
            },
        )
        return result
    except Exception:
        # A failed or crashed formal decision still consumes the single budget.
        raise


def verify_existing_typed_selection_integration(
    *,
    root: str | Path,
    manifest_path: str | Path,
    source_run_root: str | Path,
    source_train_receipt: str | Path,
    preregistration_path: str | Path,
    report_path: str | Path,
    events_path: str | Path,
) -> dict[str, Any]:
    preregistration = _read_preregistration(preregistration_path)
    canonical = _canonical_decision_paths(
        preregistration,
        preregistration_path=preregistration_path,
    )
    report = Path(report_path).expanduser().resolve(strict=True)
    events = Path(events_path).expanduser().resolve(strict=True)
    if report != canonical["report"] or events != canonical["events"]:
        raise PermissionError(
            "typed-selection integration replay paths are not canonical"
        )
    declared = _read_json_object(
        report,
        label="typed-selection integration report",
    )
    memory = MemoryEventSink()
    expected = _compute_integration(
        root=root,
        manifest_path=manifest_path,
        source_run_root=source_run_root,
        source_train_receipt=source_train_receipt,
        preregistration_path=preregistration_path,
        event_sink=memory,
    )
    if dict(declared) != expected:
        raise PermissionError(
            "typed-selection integration report does not replay exactly"
        )
    if _read_event_rows(events) != memory.events:
        raise PermissionError(
            "typed-selection integration event ledger does not replay exactly"
        )
    lock = _read_json_object(
        canonical["decision_lock"].resolve(strict=True),
        label="typed-selection integration decision lock",
    )
    _verify_decision_lock(
        lock,
        report=expected,
        preregistration_hash=stable_hash(preregistration),
    )
    expected_receipt = _build_result_receipt(
        preregistration=preregistration,
        preregistration_path=preregistration_path,
        report=expected,
        events=_read_event_rows(events),
        decision_lock=lock,
        canonical=canonical,
    )
    receipt_path = canonical["result_receipt"]
    if receipt_path.exists():
        declared_receipt = _read_json_object(
            receipt_path.resolve(strict=True),
            label="typed-selection integration result receipt",
        )
        if dict(declared_receipt) != expected_receipt:
            raise PermissionError(
                "typed-selection integration result receipt drifted"
            )
    else:
        _write_json_atomic(receipt_path, expected_receipt)
    verify_typed_selection_integration_result_receipt(
        preregistration_path=preregistration_path,
        result_receipt_path=receipt_path,
    )
    return {
        **expected,
        "integration_reuse_verified": True,
        "result_receipt_path": str(receipt_path),
        "result_receipt_file_sha256": _sha256_file(receipt_path),
    }


def verify_typed_selection_integration_result_receipt(
    *,
    preregistration_path: str | Path,
    result_receipt_path: str | Path,
) -> Mapping[str, Any]:
    """Verify the production authority against every canonical artifact.

    A true flag in the receipt is never sufficient: this reopens the report,
    full event ledger, completed decision lock, integration preregistration,
    source receipt, and split manifest and reproduces the exact receipt.
    """

    preregistration = _read_preregistration(preregistration_path)
    canonical = _canonical_decision_paths(
        preregistration,
        preregistration_path=preregistration_path,
    )
    receipt = Path(result_receipt_path).expanduser().resolve(strict=True)
    if receipt != canonical["result_receipt"]:
        raise PermissionError(
            "typed-selection result receipt path is not canonical"
        )
    declared = _read_json_object(
        receipt,
        label="typed-selection integration result receipt",
    )
    report = _read_json_object(
        canonical["report"].resolve(strict=True),
        label="typed-selection integration report",
    )
    events = _read_event_rows(canonical["events"].resolve(strict=True))
    decision_lock = _read_json_object(
        canonical["decision_lock"].resolve(strict=True),
        label="typed-selection integration decision lock",
    )
    expected = _build_result_receipt(
        preregistration=preregistration,
        preregistration_path=preregistration_path,
        report=report,
        events=events,
        decision_lock=decision_lock,
        canonical=canonical,
    )
    if dict(declared) != expected:
        raise PermissionError(
            "typed-selection integration result receipt or artifact drifted"
        )
    return declared


def _build_result_receipt(
    *,
    preregistration: Mapping[str, Any],
    preregistration_path: str | Path,
    report: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    decision_lock: Mapping[str, Any],
    canonical: Mapping[str, Path],
) -> dict[str, Any]:
    _validate_completed_result_artifacts(
        preregistration=preregistration,
        report=report,
        events=events,
        decision_lock=decision_lock,
    )
    project_root = _project_root(preregistration_path)
    preregistration_file = Path(preregistration_path).expanduser().resolve(
        strict=True
    )
    manifest_file = _resolve_bound_file(
        project_root, preregistration["manifest"]
    )
    source_receipt_file = _resolve_bound_file(
        project_root, preregistration["source_train_receipt"]
    )
    _resolve_bound_directory(
        project_root, preregistration["source_run_root"]
    )
    frozen = report["frozen_snapshot_ledger"]
    event_counts = dict(
        sorted(Counter(str(row.get("event") or "") for row in events).items())
    )
    declared_paths = preregistration["canonical_decision_paths"]
    receipt = {
        "result_receipt_version": TYPED_SELECTION_RESULT_RECEIPT_VERSION,
        "integration_policy": TYPED_SELECTION_INTEGRATION_VERSION,
        "decision_budget": 1,
        "decision_ordinal": 1,
        "decision_lock_state": "completed",
        "integration_manifest": {
            "path": preregistration_file.relative_to(project_root).as_posix(),
            "stable_hash": stable_hash(preregistration),
            "file_sha256": _sha256_file(preregistration_file),
            "implementation_file_set_hash": report["preregistration"][
                "implementation_file_set_hash"
            ],
        },
        "source_binding": {
            "manifest": preregistration["manifest"],
            "manifest_hash": preregistration["manifest_hash"],
            "manifest_file_sha256": _sha256_file(manifest_file),
            "source_run_root": preregistration["source_run_root"],
            "source_train_receipt": preregistration[
                "source_train_receipt"
            ],
            "source_train_receipt_hash": preregistration[
                "source_train_receipt_hash"
            ],
            "source_train_receipt_file_sha256": _sha256_file(
                source_receipt_file
            ),
            "upstream_binding_hash": frozen["upstream_binding_hash"],
            "snapshot_ledger_hash": frozen["ledger_hash"],
            "production_snapshot_ledger_hash": frozen[
                "production_snapshot_ledger_hash"
            ],
        },
        "decision_hash": report["decision_hash"],
        "report_hash": report["report_hash"],
        "canonical_artifacts": {
            "report": {
                "path": declared_paths["report"],
                "sha256": _sha256_file(canonical["report"]),
            },
            "events": {
                "path": declared_paths["events"],
                "sha256": _sha256_file(canonical["events"]),
                "event_count": len(events),
                "event_counts": event_counts,
                "event_set_hash": stable_hash({"events": list(events)}),
            },
            "decision_lock": {
                "path": declared_paths["decision_lock"],
                "sha256": _sha256_file(canonical["decision_lock"]),
                "stable_hash": stable_hash(decision_lock),
            },
        },
        "compiler_provenance": {
            "compile_manifest_hash": report["compiler_provenance"][
                "compile_manifest_hash"
            ],
            "compiled_binding_set_hash": report["compiler_provenance"][
                "compiled_binding_set_hash"
            ],
            "compiler_event_set_hash": report["compiler_provenance"][
                "compiler_event_set_hash"
            ],
            "compiler_event_path_normalization": report[
                "compiler_provenance"
            ]["compiler_event_path_normalization"],
            "compiler_binding_coverage_hash": report[
                "compiler_provenance"
            ]["compiler_binding_coverage_hash"],
            "runtime_source_receipt_count": report[
                "compiler_provenance"
            ]["runtime_source_receipt_count"],
            "runtime_source_receipt_set_hash": report[
                "compiler_provenance"
            ]["runtime_source_receipt_set_hash"],
            "runtime_source_routed_count": report[
                "compiler_provenance"
            ]["runtime_source_routed_count"],
            "runtime_source_no_skill_count": report[
                "compiler_provenance"
            ]["runtime_source_no_skill_count"],
        },
        "integration_passed": True,
        "acceptance": dict(report["acceptance"]),
        "exact_replay_verified": True,
        "fresh_development_protocol_freeze_eligible": True,
        "development_task_execution_authorized": False,
        "promotion_gate_or_score": False,
        "raw_content_persisted": False,
    }
    return receipt


def _validate_completed_result_artifacts(
    *,
    preregistration: Mapping[str, Any],
    report: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    decision_lock: Mapping[str, Any],
) -> None:
    report_without_hash = dict(report)
    declared_report_hash = report_without_hash.pop("report_hash", None)
    if (
        declared_report_hash != stable_hash(report_without_hash)
        or report.get("integration_policy")
        != TYPED_SELECTION_INTEGRATION_VERSION
        or report.get("integration_passed") is not True
        or report.get("acceptance") != preregistration["acceptance"]
        or report.get("fresh_development_protocol_freeze_eligible_if_passed")
        is not True
        or report.get("development_task_execution_currently_authorized")
        is not False
    ):
        raise PermissionError(
            "typed-selection completed report is not authoritative"
        )
    _verify_decision_lock(
        decision_lock,
        report=report,
        preregistration_hash=stable_hash(preregistration),
    )
    if not events or events[-1].get("event") != (
        "typed_selection_integration_completed"
    ):
        raise PermissionError(
            "typed-selection completed event is missing"
        )
    predecision = list(events[:-1])
    offline = report.get("offline_boundary_contract")
    if not isinstance(offline, Mapping) or (
        offline.get("predecision_event_count") != len(predecision)
        or offline.get("predecision_event_set_hash")
        != stable_hash({"events": predecision})
        or offline.get("predecision_event_counts")
        != dict(
            sorted(
                Counter(
                    str(row.get("event") or "") for row in predecision
                ).items()
            )
        )
    ):
        raise PermissionError(
            "typed-selection predecision event commitment drifted"
        )
    completed_payload = events[-1].get("payload")
    if not isinstance(completed_payload, Mapping) or (
        completed_payload.get("decision_hash") != report["decision_hash"]
        or completed_payload.get("report_hash") != report["report_hash"]
        or completed_payload.get("integration_passed") is not True
    ):
        raise PermissionError(
            "typed-selection completion event drifted"
        )
    compiler = report.get("compiler_provenance")
    compiler_events = [
        row
        for row in predecision
        if row.get("event") == "typed_compiler_binding_verified"
    ]
    if not isinstance(compiler, Mapping) or len(compiler_events) != 1:
        raise PermissionError(
            "typed-selection compiler provenance is missing"
        )
    compiler_payload = compiler_events[0].get("payload")
    hash_fields = (
        "compile_manifest_hash",
        "compiled_binding_set_hash",
        "compiler_event_set_hash",
        "compiler_binding_coverage_hash",
        "runtime_source_receipt_set_hash",
    )
    count_fields = (
        "runtime_source_receipt_count",
        "runtime_source_routed_count",
        "runtime_source_no_skill_count",
    )
    for field in (*hash_fields, *count_fields):
        if (
            not isinstance(compiler_payload, Mapping)
            or compiler_payload.get(field) != compiler.get(field)
        ):
            raise PermissionError(
                "typed-selection compiler event commitment drifted"
            )
    if compiler.get("compiler_event_path_normalization") != (
        "item_hypothesis_content_route_v1"
    ) or compiler_payload.get("compiler_event_path_normalization") != (
        compiler["compiler_event_path_normalization"]
    ):
        raise PermissionError(
            "typed-selection compiler event normalization drifted"
        )
    if any(not _is_sha256_text(compiler.get(field)) for field in hash_fields):
        raise PermissionError(
            "typed-selection compiler hash commitment is malformed"
        )
    if any(
        isinstance(compiler.get(field), bool)
        or not isinstance(compiler.get(field), int)
        or int(compiler[field]) < 0
        for field in count_fields
    ) or (
        compiler["runtime_source_receipt_count"] != 38
        or compiler["runtime_source_routed_count"]
        + compiler["runtime_source_no_skill_count"]
        != compiler["runtime_source_receipt_count"]
    ):
        raise PermissionError(
            "typed-selection runtime source receipt coverage is malformed"
        )


def load_frozen_typed_selection_ledger(
    *,
    root: str | Path,
    manifest_path: str | Path,
    source_run_root: str | Path,
    source_train_receipt: str | Path,
    preregistration_path: str | Path,
) -> FrozenTypedSelectionLedger:
    """Load the production snapshot ledger from externally frozen evidence.

    This public helper is intentionally shared with the production CLI.  It
    verifies the upstream feasibility artifacts before constructing any graph;
    a graph's own self-hash is never sufficient authority.
    """

    preregistration = _read_preregistration(preregistration_path)
    project_root = _project_root(preregistration_path)
    _require_declared_input_path(
        manifest_path,
        project_root=project_root,
        declared=preregistration["manifest"],
        label="split manifest",
    )
    _require_declared_input_path(
        source_train_receipt,
        project_root=project_root,
        declared=preregistration["source_train_receipt"],
        label="source TRAIN receipt",
    )
    _require_declared_input_directory(
        source_run_root,
        project_root=project_root,
        declared=preregistration["source_run_root"],
        label="source run root",
    )
    upstream = _verify_upstream_feasibility(
        preregistration,
        preregistration_path=preregistration_path,
    )
    manifest = SplitManifest.read(manifest_path)
    if manifest.manifest_hash != preregistration["manifest_hash"]:
        raise PermissionError("typed-selection manifest binding mismatch")
    evidence = reconstruct_v315_train_evidence(
        root=root,
        manifest=manifest,
        source_run_root=source_run_root,
        source_train_receipt=source_train_receipt,
        event_sink=NullEventSink(),
    )
    if evidence.source_train_receipt_hash != preregistration[
        "source_train_receipt_hash"
    ]:
        raise PermissionError("typed-selection TRAIN receipt binding mismatch")
    trials = _extract_all_train_trials(
        evidence=evidence,
        source_run_root=source_run_root,
    )
    trial_evidence_hash = stable_hash(
        {"trial_evidence_hashes": [row.evidence_hash for row in trials]}
    )
    if trial_evidence_hash != preregistration[
        "expected_trial_evidence_hash"
    ]:
        raise PermissionError("typed-selection trial ledger drifted")

    family_by_hash = {
        stable_hash({"family": family}): family
        for family in {row.family for row in evidence.failures}
    }
    snapshots: list[TypedRecipeSelectionSnapshot] = []
    trial_by_id = {row.trial_id_hash: row for row in trials}
    for target_hash in preregistration["expected_target_family_hashes"]:
        family = family_by_hash.get(target_hash)
        if family is None:
            raise PermissionError(
                "typed-selection target family is absent from TRAIN failures"
            )
        graph = build_family_capability_graph(
            target_family=family,
            failures=evidence.failures,
            action_profiles=evidence.action_profiles,
            trial_evidence=trial_by_id,
            minimum_support=MINIMUM_FAMILY_SUPPORT,
            maximum_artifacts=MAX_REGISTERED_ARTIFACTS_PER_FAMILY,
        )
        snapshots.append(freeze_typed_recipe_selection_snapshot(graph))
    frozen = tuple(snapshots)
    graph_hashes = [row.expected_graph_hash for row in frozen]
    model_catalog_hashes = [
        row.expected_model_catalog_hash for row in frozen
    ]
    graph_set_hash = stable_hash(
        {
            "outcomes": [
                {
                    "target_family_hash": row.graph.target_family_hash,
                    "graph_hash": row.expected_graph_hash,
                    "availability_error_hash": None,
                }
                for row in frozen
            ]
        }
    )
    model_catalog_set_hash = stable_hash(
        {"catalog_hashes": model_catalog_hashes}
    )
    if (
        len(frozen) != REQUIRED_SLOT_COUNT
        or [row.graph.target_family_hash for row in frozen]
        != preregistration["expected_target_family_hashes"]
        or graph_hashes != preregistration["expected_graph_hashes"]
        or model_catalog_hashes
        != preregistration["expected_model_catalog_hashes"]
        or graph_set_hash != preregistration["expected_graph_set_hash"]
        or model_catalog_set_hash
        != preregistration["expected_model_catalog_set_hash"]
    ):
        raise PermissionError(
            "typed-selection snapshots drifted from external frozen ledger"
        )
    upstream_graphs = upstream["report"]["typed_operator_graph"]["graphs"]
    if [row["graph_hash"] for row in upstream_graphs] != graph_hashes or [
        row["model_catalog_hash"] for row in upstream_graphs
    ] != model_catalog_hashes:
        raise PermissionError(
            "typed-selection snapshots drifted from upstream report rows"
        )
    production_snapshot_ledger = freeze_typed_selection_snapshot_ledger(
        frozen,
        feasibility_preregistration_hash=upstream[
            "safe_summary"
        ]["upstream_preregistration_hash"],
        feasibility_result_receipt_sha256=upstream[
            "safe_summary"
        ]["upstream_result_receipt_file_sha256"],
        feasibility_decision_hash=upstream[
            "safe_summary"
        ]["upstream_decision_hash"],
        feasibility_report_hash=upstream[
            "safe_summary"
        ]["upstream_report_hash"],
        manifest_hash=manifest.manifest_hash,
        source_train_receipt_hash=evidence.source_train_receipt_hash,
        expected_graph_set_hash=graph_set_hash,
        expected_model_catalog_set_hash=model_catalog_set_hash,
        expected_target_family_hashes=preregistration[
            "expected_target_family_hashes"
        ],
    )
    return FrozenTypedSelectionLedger(
        evidence=evidence,
        trials=trials,
        snapshots=frozen,
        production_snapshot_ledger=production_snapshot_ledger,
        upstream_binding_hash=upstream["binding_hash"],
        trial_evidence_hash=trial_evidence_hash,
        graph_set_hash=graph_set_hash,
        model_catalog_set_hash=model_catalog_set_hash,
    )


def _exercise_production_protocol_loader(
    *,
    root: str | Path,
    manifest_path: str | Path,
    source_run_root: str | Path,
    source_train_receipt: str | Path,
    preregistration: Mapping[str, Any],
    preregistration_path: str | Path,
    expected_ledger: FrozenTypedSelectionLedger,
) -> dict[str, Any]:
    """Traverse PaperProtocol.read and the real CLI loader without authority."""

    from .paper_protocol import (
        PaperProtocol,
        TYPED_SELECTION_MODEL_INFERENCE_SLOTS,
        TYPED_SELECTION_PHASE_PARALLEL_WORKERS,
        TYPED_SELECTION_PROTOCOL_VERSION,
    )
    from .skilllearn_experiment import (
        _load_typed_selection_for_execution,
    )

    project_root = _project_root(preregistration_path)
    probe = preregistration["protocol_contract_probe"]
    template_path = _resolve_bound_file(project_root, probe["template"])
    if _sha256_file(template_path) != probe["template_file_sha256"]:
        raise PermissionError(
            "typed-selection protocol probe template drifted"
        )
    payload = json.loads(template_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PermissionError(
            "typed-selection protocol probe template is malformed"
        )
    payload["protocol_id"] = (
        "assumption-agent-typed-selection-integration-protocol-probe"
    )
    payload["protocol_version"] = TYPED_SELECTION_PROTOCOL_VERSION
    execution = payload.get("execution")
    if not isinstance(execution, dict):
        raise PermissionError(
            "typed-selection protocol probe execution policy is malformed"
        )
    execution["proposal_formation_policy"] = (
        TYPED_RECIPE_PROPOSAL_FORMATION_POLICY_VERSION
    )
    execution["model_inference_slots"] = (
        TYPED_SELECTION_MODEL_INFERENCE_SLOTS
    )
    for phase_name, workers in (
        TYPED_SELECTION_PHASE_PARALLEL_WORKERS.items()
    ):
        payload["phases"][phase_name]["parallel_workers"] = workers
    execution["typed_selection_snapshot_source"] = {
        "preregistration": Path(preregistration_path)
        .expanduser()
        .resolve(strict=True)
        .relative_to(project_root)
        .as_posix(),
        "preregistration_file_sha256": _sha256_file(
            Path(preregistration_path).expanduser().resolve(strict=True)
        ),
        "source_run_root": preregistration["source_run_root"],
        "source_train_receipt": preregistration[
            "source_train_receipt"
        ],
        "source_train_receipt_file_sha256": _sha256_file(
            _resolve_bound_file(
                project_root, preregistration["source_train_receipt"]
            )
        ),
        "integration_result_receipt": preregistration[
            "canonical_decision_paths"
        ]["result_receipt"],
        "integration_result_receipt_file_sha256": "0" * 64,
        "snapshot_ledger_hash": (
            expected_ledger.production_snapshot_ledger.ledger_hash
        ),
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=project_root / "manifests",
            prefix=".typed-selection-protocol-probe-",
            suffix=".json",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        protocol = PaperProtocol.read(temporary_path)
        loaded = _load_typed_selection_for_execution(
            root=root,
            manifest_path=manifest_path,
            protocol=protocol,
            execution_contract=protocol.payload["execution"],
            proposal_formation_policy=(
                TYPED_RECIPE_PROPOSAL_FORMATION_POLICY_VERSION
            ),
            integration_diagnostic_policy=(
                TYPED_SELECTION_INTEGRATION_VERSION
            ),
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    if loaded.production_snapshot_ledger.ledger_hash != (
        expected_ledger.production_snapshot_ledger.ledger_hash
    ):
        raise PermissionError(
            "typed-selection production loader reconstructed a different ledger"
        )
    diagnostic_execution_rejected = False
    try:
        loaded.require_freeze_authorization()
    except PermissionError:
        diagnostic_execution_rejected = True
    if not diagnostic_execution_rejected:
        raise PermissionError(
            "typed-selection diagnostic loader leaked freeze authority"
        )
    return {
        "protocol_version": TYPED_SELECTION_PROTOCOL_VERSION,
        "proposal_formation_policy": (
            TYPED_RECIPE_PROPOSAL_FORMATION_POLICY_VERSION
        ),
        "protocol_contract_hash": stable_hash(payload),
        "paper_protocol_read_passed": True,
        "production_cli_loader_passed": True,
        "production_snapshot_ledger_hash": (
            loaded.production_snapshot_ledger.ledger_hash
        ),
        "integration_result_receipt_used": False,
        "freeze_authorization_present": False,
        "diagnostic_execution_rejected": diagnostic_execution_rejected,
        "development_task_execution_authorized": False,
        "live_model_invoked": False,
        "live_task_backend_invoked": False,
        "raw_content_persisted": False,
    }


def _compute_integration(
    *,
    root: str | Path,
    manifest_path: str | Path,
    source_run_root: str | Path,
    source_train_receipt: str | Path,
    preregistration_path: str | Path,
    event_sink: EventSink,
) -> dict[str, Any]:
    benchmark_root = root
    preregistration = _read_preregistration(preregistration_path)
    preregistration_hash = stable_hash(preregistration)
    implementation_file_set_hash = _implementation_file_set_hash(
        preregistration,
        preregistration_path=preregistration_path,
    )
    trace_id = stable_hash(
        {
            "policy": TYPED_SELECTION_INTEGRATION_VERSION,
            "preregistration_hash": preregistration_hash,
            "implementation_file_set_hash": implementation_file_set_hash,
        }
    )[:24]
    audit_sink = _BoundaryAuditEventSink(event_sink)
    _emit(
        audit_sink,
        event="typed_selection_integration_started",
        trace_id=trace_id,
        payload={
            "integration_policy": TYPED_SELECTION_INTEGRATION_VERSION,
            "preregistration_hash": preregistration_hash,
            "implementation_file_set_hash": implementation_file_set_hash,
            "decision_budget": 1,
            "offline_only": True,
            "non_scoring": True,
        },
    )

    upstream = _verify_upstream_feasibility(
        preregistration,
        preregistration_path=preregistration_path,
    )
    _emit(
        audit_sink,
        event="typed_selection_upstream_feasibility_verified",
        trace_id=trace_id,
        payload={
            **upstream["safe_summary"],
            "external_frozen_ledger_required": True,
        },
    )
    ledger = load_frozen_typed_selection_ledger(
        root=root,
        manifest_path=manifest_path,
        source_run_root=source_run_root,
        source_train_receipt=source_train_receipt,
        preregistration_path=preregistration_path,
    )
    protocol_loader_receipt = _exercise_production_protocol_loader(
        root=root,
        manifest_path=manifest_path,
        source_run_root=source_run_root,
        source_train_receipt=source_train_receipt,
        preregistration=preregistration,
        preregistration_path=preregistration_path,
        expected_ledger=ledger,
    )
    evidence = ledger.evidence
    trials = ledger.trials
    snapshots = ledger.snapshots
    _emit(
        audit_sink,
        event="typed_selection_train_evidence_reconstructed",
        trace_id=trace_id,
        payload={
            "source_train_receipt_hash": evidence.source_train_receipt_hash,
            "trial_evidence_hash": ledger.trial_evidence_hash,
            "train_observation_count": len(evidence.residuals),
            "train_failure_count": len(evidence.failures),
            "train_success_control_count": len(evidence.success_controls),
            "complete_trace_count": sum(row.trace_complete for row in trials),
            "source_agent_trials_reexecuted_count": 0,
        },
    )
    _emit(
        audit_sink,
        event="typed_selection_snapshots_frozen",
        trace_id=trace_id,
        payload={
            **ledger.safe_payload(),
            "external_graph_ledger_matched": True,
            "self_certifying_snapshot_authority": False,
        },
    )

    selector = _OfflineRecipeSelector(
        ledger.production_snapshot_ledger.ledger_hash
    )
    proposer = StructuredHypothesisProposer(
        selector,
        event_sink=audit_sink,
    )
    context = _validation_context(ledger)
    kernel = _kernel(
        proposer=proposer,
        ledger=ledger,
        event_sink=audit_sink,
    )
    diagnostic_evolution_boundary_rejected = False
    try:
        kernel.evolve_once(
            residuals=evidence.residuals,
            validation_tasks=(),
            validation_context=context,
            trace_id=f"{trace_id}:diagnostic-execution-canary",
        )
    except PermissionError as exc:
        diagnostic_evolution_boundary_rejected = (
            "protocol-lock task execution authorization" in str(exc)
        )
    if not diagnostic_evolution_boundary_rejected:
        raise PermissionError(
            "typed-selection diagnostic ledger reached the evolution task "
            "boundary"
        )
    protocol_loader_receipt = {
        **protocol_loader_receipt,
        "diagnostic_evolution_boundary_rejected": True,
    }
    _emit(
        audit_sink,
        event="typed_selection_protocol_loader_verified",
        trace_id=trace_id,
        payload=protocol_loader_receipt,
    )
    programs = kernel.propose_candidates(
        evidence.residuals,
        validation_context=context,
        trace_id=f"{trace_id}:root",
    )
    root_selector_call_count = selector.call_count
    root_bindings = tuple(
        proposer.typed_program_registry.require(row) for row in programs
    )
    replay_before = selector.call_count
    replayed_programs = kernel.propose_candidates(
        evidence.residuals,
        validation_context=context,
        trace_id=f"{trace_id}:root-replay",
    )
    replay_new_selector_calls = selector.call_count - replay_before

    shared_before = selector.call_count
    shared_bindings = kernel.validate_typed_shared_proposal_candidates(
        programs,
        trace_id=f"{trace_id}:shared",
    )
    shared_new_selector_calls = selector.call_count - shared_before
    shared_receipt = {
        "policy": TYPED_RECIPE_PROPOSAL_FORMATION_POLICY_VERSION,
        "candidate_hashes": [row.payload_hash for row in programs],
        "snapshot_hashes": [row.snapshot_hash for row in snapshots],
        "binding_hashes": [row["binding_hash"] for row in shared_bindings],
        "new_selector_calls": shared_new_selector_calls,
        "validation_features_used": False,
        "validation_outcomes_used": False,
        "raw_content_persisted": False,
    }
    shared_receipt_hash = stable_hash(shared_receipt)

    kernel.record_typed_selection_attempts(
        programs,
        trace_id=f"{trace_id}:selection-attempts",
    )
    attempt_history_binding_hashes_before_next_generation = tuple(
        sorted(kernel.archive.typed_selection_history)
    )
    hypothesis_count_before_next_generation = len(
        kernel.archive.hypotheses
    )
    typed_binding_count_before_next_generation = len(
        kernel.archive.typed_bindings
    )
    next_generation_before = selector.call_count
    next_generation_programs = kernel.propose_candidates(
        evidence.residuals,
        validation_context=context,
        trace_id=f"{trace_id}:next-generation",
    )
    next_generation_new_selector_calls = (
        selector.call_count - next_generation_before
    )
    next_generation_bindings = tuple(
        proposer.typed_program_registry.require(row)
        for row in next_generation_programs
    )

    for program in programs:
        kernel.archive.register_hypothesis(
            program,
            typed_binding=proposer.typed_program_registry.safe_binding(
                program
            ),
            trace_id=f"{trace_id}:archive-root",
        )
    archive_payload = kernel.archive.to_dict()
    archive_bindings = archive_payload.get("typed_bindings")
    if not isinstance(archive_bindings, Mapping):
        raise PermissionError(
            "typed-selection archive omitted typed bindings"
        )
    restored_archive_registry = TypedProgramBindingRegistry(
        snapshot_ledger=ledger.production_snapshot_ledger
    )
    restored_archive_binding_hashes: list[str] = []
    for program in sorted(programs, key=lambda row: len(row.lineage)):
        binding_payload = archive_bindings.get(program.id)
        if not isinstance(binding_payload, Mapping):
            raise PermissionError(
                "typed-selection archive binding coverage is incomplete"
            )
        restored_archive_binding_hashes.append(
            restored_archive_registry.restore_safe_payload(
                program, binding_payload
            ).binding_hash
        )
    archive_binding_set_hash = stable_hash(
        {
            "binding_hashes": sorted(restored_archive_binding_hashes)
        }
    )

    root = programs[0]
    root_binding = proposer.typed_program_registry.require(root)
    repair_before = selector.call_count
    repair_tree = RecursiveValidationEngine(
        (_RootOnlyFailureCheck(root.id),),
        proposer=proposer,
        event_sink=audit_sink,
    ).validate(
        root,
        context,
        trace_id=f"{trace_id}:repair",
    )
    repair_new_selector_calls = selector.call_count - repair_before
    repaired = repair_tree.accepted_program
    repaired_binding = (
        proposer.typed_program_registry.require(repaired)
        if repaired is not None
        else None
    )
    next_generation_root = next_generation_programs[0]
    next_generation_root_binding = next_generation_bindings[0]
    cross_generation_repair_before = selector.call_count
    cross_generation_repair_tree = RecursiveValidationEngine(
        (_RootOnlyFailureCheck(next_generation_root.id),),
        proposer=proposer,
        event_sink=audit_sink,
    ).validate(
        next_generation_root,
        context,
        trace_id=f"{trace_id}:cross-generation-repair",
    )
    cross_generation_repair_new_selector_calls = (
        selector.call_count - cross_generation_repair_before
    )
    cross_generation_repaired = (
        cross_generation_repair_tree.accepted_program
    )
    cross_generation_repaired_binding = (
        proposer.typed_program_registry.require(cross_generation_repaired)
        if cross_generation_repaired is not None
        else None
    )

    if repaired is None or repaired_binding is None:
        raise PermissionError(
            "typed-selection recursive archive canary has no repair"
        )
    kernel.archive.register_hypothesis(
        repaired,
        typed_binding=repaired_binding.safe_payload(),
        trace_id=f"{trace_id}:archive-repair",
    )
    kernel.archive.set_hypothesis_status(
        root.id,
        HypothesisStatus.SHADOW,
        trace_id=f"{trace_id}:archive-status-root",
    )
    kernel.archive.set_hypothesis_status(
        repaired.id,
        HypothesisStatus.SHADOW,
        trace_id=f"{trace_id}:archive-status-repair",
    )
    recursive_archive_payload = kernel.archive.to_dict()
    recursive_archive_round_trip_passed = False
    recursive_archive_binding_hashes: tuple[str, ...] = ()
    from .paper_freeze import read_frozen_archive
    from .paper_protocol import (
        PaperProtocol,
        TYPED_SELECTION_PROTOCOL_VERSION,
    )

    project_root = _project_root(preregistration_path)
    protocol_template = _resolve_bound_file(
        project_root,
        preregistration["protocol_contract_probe"]["template"],
    )
    promotion_spec = PaperProtocol.read(
        protocol_template
    ).promotion_gate_spec
    recursive_report = {
        "archive_hash": recursive_archive_payload["archive_hash"],
        "generations": [
            {
                "accepted_hypothesis_id": None,
                "selected_candidate_hypothesis_ids": [],
                "promotion_decision": None,
                "promoted": False,
            }
        ],
    }
    with tempfile.TemporaryDirectory(
        prefix="typed-selection-recursive-archive-"
    ) as recursive_archive_root:
        recursive_archive_path = (
            Path(recursive_archive_root) / "archive.json"
        )
        kernel.archive.write(recursive_archive_path)
        frozen_recursive_archive = read_frozen_archive(
            recursive_archive_path,
            expected_evaluator_epoch=TYPED_SELECTION_EVALUATOR_EPOCH,
            expected_report=recursive_report,
            promotion_spec=promotion_spec,
            protocol_version=TYPED_SELECTION_PROTOCOL_VERSION,
            typed_selection_ledger=(
                ledger.production_snapshot_ledger
            ),
        )
        assert frozen_recursive_archive.typed_program_registry is not None
        restored_programs = tuple(
            frozen_recursive_archive.typed_program_registry.require(
                kernel.archive.hypotheses[program_id]
            )
            for program_id in (root.id, repaired.id)
        )
        recursive_archive_binding_hashes = tuple(
            row.binding_hash for row in restored_programs
        )
        restored_selection_history_binding_hashes = tuple(
            sorted(
                str(row["binding_hash"])
                for row in frozen_recursive_archive.typed_selection_history
            )
        )
        recursive_archive_round_trip_passed = (
            recursive_archive_binding_hashes
            == (root_binding.binding_hash, repaired_binding.binding_hash)
            and restored_selection_history_binding_hashes
            == tuple(sorted(kernel.archive.typed_selection_history))
        )

    compiler_event_sink = MemoryEventSink()
    compiler = SkillLearnProgramCompiler(
        event_sink=compiler_event_sink,
        typed_program_registry=proposer.typed_program_registry,
        require_typed_bindings=True,
    )
    compiler_programs = (
        *programs,
        *((repaired,) if repaired is not None else ()),
    )
    compiler_binding_hashes = compiler.require_program_bindings(
        compiler_programs
    )
    lowered_action_hashes: list[str] = []
    compiler_issue_count = 0
    for program in compiler_programs:
        compiler_issue_count += len(
            backend_action_contract_issues(
                program,
                allowed_operations=SKILLLEARN_ALLOWED_ACTION_OPERATIONS,
                external_evidence_is_hidden=True,
            )
        )
        lowered = _lower_skilllearn_program(program)
        lowered_action_hashes.append(
            stable_hash({"actions": [row.to_dict() for row in lowered]})
        )
    compile_manifest: Mapping[str, Any]
    runtime_source_receipt_rows: tuple[dict[str, Any], ...]
    runtime_source_mutation_rejected = False
    runtime_source_mutation_error_type_hash: str | None = None
    with tempfile.TemporaryDirectory(
        prefix="typed-selection-compiler-"
    ) as temporary_compile_root:
        compile_split_manifest = SplitManifest.read(manifest_path)
        compile_result = compiler.compile(
            programs=compiler_programs,
            items=SkillLearnBenchAdapter(benchmark_root).discover(),
            split_manifest=compile_split_manifest,
            output_root=temporary_compile_root,
            method_name="typed-selection-integration",
            allowed_statuses={HypothesisStatus.CANDIDATE},
            target_item_ids=compile_split_manifest.train_ids,
            target_split="train",
            trace_id=f"{trace_id}:compiler",
        )
        compile_manifest = _read_json_object(
            compile_result.output_root / "compile_manifest.json",
            label="typed-selection compiler manifest",
        )
        compile_manifest_hash = stable_hash(compile_manifest)
        compiled_binding_hashes = compile_result.typed_binding_hashes
        compiled_binding_set_hash = compile_result.typed_binding_set_hash
        compiled_snapshot_ledger_hash = (
            compile_result.typed_snapshot_ledger_hash
        )
        runtime_source_receipts = tuple(
            (
                item_id,
                compile_result.source_receipt_for(item_id),
            )
            for item_id in sorted(compile_split_manifest.train_ids)
        )
        runtime_source_receipt_rows = tuple(
            {
                "item_id_hash": receipt.item_id_hash,
                "receipt_hash": receipt.receipt_hash,
                "source_route_present": receipt.source_route is not None,
                "source_file_count": len(receipt.source_file_hashes),
                "source_tree_hash": receipt.source_tree_hash,
                "treatment_hash": receipt.treatment_hash,
                "compile_manifest_hash": receipt.compile_manifest_hash,
                "typed_binding_set_hash": receipt.typed_binding_set_hash,
                "typed_snapshot_ledger_hash": (
                    receipt.typed_snapshot_ledger_hash
                ),
                "raw_content_persisted": False,
            }
            for _, receipt in runtime_source_receipts
        )
        routed_receipts = tuple(
            (item_id, receipt)
            for item_id, receipt in runtime_source_receipts
            if receipt.source_route is not None
        )
        if not routed_receipts:
            raise PermissionError(
                "typed-selection compiler produced no routed source canary"
            )
        canary_item_id, canary_receipt = routed_receipts[0]
        canary_source = compile_result.source_for(canary_item_id)
        assert canary_source is not None
        canary_relative_path = canary_receipt.source_file_hashes[0][0]
        canary_path = canary_source / canary_relative_path
        canary_text = canary_path.read_text(encoding="utf-8")
        try:
            canary_path.write_text(
                canary_text + "\npost-compile-mutation-probe\n",
                encoding="utf-8",
            )
            try:
                compile_result.source_receipt_for(canary_item_id)
            except PermissionError as exc:
                runtime_source_mutation_rejected = True
                runtime_source_mutation_error_type_hash = stable_hash(
                    {"error_type": type(exc).__name__}
                )
        finally:
            canary_path.write_text(canary_text, encoding="utf-8")
        if compile_result.source_receipt_for(
            canary_item_id
        ).receipt_hash != canary_receipt.receipt_hash:
            raise PermissionError(
                "typed-selection compiler canary did not restore exactly"
            )
        compiler_event_rows = tuple(compiler_event_sink.events)
    runtime_source_receipt_set_hash = stable_hash(
        {"receipts": list(runtime_source_receipt_rows)}
    )
    runtime_source_routed_count = sum(
        row["source_route_present"] for row in runtime_source_receipt_rows
    )
    runtime_source_no_skill_count = (
        len(runtime_source_receipt_rows) - runtime_source_routed_count
    )
    compiler_event_commitment_rows = _canonical_compiler_event_rows(
        compiler_event_rows
    )
    compiler_event_set_hash = stable_hash(
        {"events": list(compiler_event_commitment_rows)}
    )
    compiler_event_binding_hashes = tuple(
        sorted(
            {
                str(row.get("payload", {}).get("typed_binding_hash") or "")
                for row in compiler_event_rows
                if row.get("payload", {}).get("typed_binding_hash")
            }
        )
    )
    compiler_binding_coverage_hash = stable_hash(
        {"binding_hashes": list(compiler_event_binding_hashes)}
    )
    _emit(
        audit_sink,
        event="typed_compiler_binding_verified",
        trace_id=trace_id,
        payload={
            "program_count": len(compiler_programs),
            "binding_set_hash": stable_hash(
                {"binding_hashes": list(compiler_binding_hashes)}
            ),
            "lowered_action_set_hash": stable_hash(
                {"lowered_action_hashes": lowered_action_hashes}
            ),
            "compiler_issue_count": compiler_issue_count,
            "binding_hashes_returned_by_compiler": True,
            "compile_manifest_hash": compile_manifest_hash,
            "compiled_binding_set_hash": compiled_binding_set_hash,
            "compiled_snapshot_ledger_hash": (
                compiled_snapshot_ledger_hash
            ),
            "compiled_event_count": len(compiler_event_rows),
            "compiler_event_set_hash": compiler_event_set_hash,
            "compiler_event_path_normalization": (
                "item_hypothesis_content_route_v1"
            ),
            "compiler_binding_coverage_hash": (
                compiler_binding_coverage_hash
            ),
            "runtime_source_receipt_count": len(
                runtime_source_receipt_rows
            ),
            "runtime_source_receipt_set_hash": (
                runtime_source_receipt_set_hash
            ),
            "runtime_source_routed_count": runtime_source_routed_count,
            "runtime_source_no_skill_count": runtime_source_no_skill_count,
            "harness_owned_materialization": True,
        },
    )

    tamper_rows = _run_tamper_probes(
        kernel=kernel,
        proposer=proposer,
        compiler=compiler,
        programs=programs,
        snapshots=snapshots,
        production_snapshot_ledger=ledger.production_snapshot_ledger,
        context=context,
        selector=selector,
    )
    tamper_rows.append(
        {
            "probe_id": "runtime_skill_content_mutation",
            "rejected": runtime_source_mutation_rejected,
            "error_type_hash": runtime_source_mutation_error_type_hash,
        }
    )
    _emit(
        audit_sink,
        event="typed_selection_tamper_probes_completed",
        trace_id=trace_id,
        payload={
            "probe_count": len(tamper_rows),
            "rejection_count": sum(row["rejected"] for row in tamper_rows),
            "probe_set_hash": stable_hash({"probes": tamper_rows}),
            "direct_unbound_parent_repair_rejected": next(
                row["rejected"]
                for row in tamper_rows
                if row["probe_id"] == "direct_repair_unbound_parent"
            ),
        },
    )

    primitive_values = _all_primitive_values(trials)
    raw_locator_disclosure_count = 0
    raw_primitive_disclosure_count = 0
    for request, snapshot in _request_snapshot_pairs(
        selector.requests,
        snapshots,
    ):
        catalog = request["selection_snapshot"]["catalog"]
        leaves = _leaf_strings(catalog)
        raw_locator_disclosure_count += sum(
            artifact.locator in leaves for artifact in snapshot.graph.artifacts
        )
        raw_primitive_disclosure_count += sum(
            value in leaves for value in primitive_values
        )
    request_surface_issues = tuple(
        issue
        for request in selector.requests
        for issue in _selector_request_surface_issues(
            request,
            expected_ledger_hash=(
                ledger.production_snapshot_ledger.ledger_hash
            ),
        )
    )

    source_reconstruction_passed = bool(
        len(evidence.residuals) == 38
        and len(evidence.failures) == 32
        and len(evidence.success_controls) == 6
        and len(trials) == 38
        and all(row.trace_complete for row in trials)
        and evidence.source_train_receipt_hash
        == preregistration["source_train_receipt_hash"]
        and ledger.trial_evidence_hash
        == preregistration["expected_trial_evidence_hash"]
    )
    snapshot_external_ledger_passed = bool(
        len(snapshots) == REQUIRED_SLOT_COUNT
        and ledger.graph_set_hash
        == preregistration["expected_graph_set_hash"]
        and ledger.model_catalog_set_hash
        == preregistration["expected_model_catalog_set_hash"]
        and [row.expected_graph_hash for row in snapshots]
        == preregistration["expected_graph_hashes"]
        and [row.expected_model_catalog_hash for row in snapshots]
        == preregistration["expected_model_catalog_hashes"]
    )
    production_protocol_loader_path_passed = bool(
        protocol_loader_receipt[
            "production_snapshot_ledger_hash"
        ]
        == ledger.production_snapshot_ledger.ledger_hash
        and protocol_loader_receipt["paper_protocol_read_passed"] is True
        and protocol_loader_receipt["production_cli_loader_passed"] is True
        and protocol_loader_receipt["integration_result_receipt_used"]
        is False
        and protocol_loader_receipt["freeze_authorization_present"] is False
        and protocol_loader_receipt["diagnostic_execution_rejected"] is True
        and protocol_loader_receipt[
            "diagnostic_evolution_boundary_rejected"
        ]
        is True
        and protocol_loader_receipt[
            "development_task_execution_authorized"
        ]
        is False
    )
    production_selection_path_passed = bool(
        len(programs) == REQUIRED_SLOT_COUNT
        and root_selector_call_count == REQUIRED_SLOT_COUNT
        and len(root_bindings) == REQUIRED_SLOT_COUNT
        and tuple(row.snapshot_hash for row in root_bindings)
        == tuple(row.snapshot_hash for row in snapshots)
        and all(
            row.snapshot_ledger_hash
            == ledger.production_snapshot_ledger.ledger_hash
            for row in root_bindings
        )
        and all(not row.validate() for row in programs)
        and all(row.action_graph for row in programs)
        and set(archive_bindings) == {row.id for row in programs}
        and set(restored_archive_binding_hashes)
        == {row.binding_hash for row in root_bindings}
        and archive_binding_set_hash
        == stable_hash(
            {
                "binding_hashes": sorted(
                    row.binding_hash for row in root_bindings
                )
            }
        )
    )
    closed_model_surface_passed = bool(
        len(selector.requests) == (REQUIRED_SLOT_COUNT * 2) + 2
        and not request_surface_issues
        and raw_locator_disclosure_count == 0
        and raw_primitive_disclosure_count == 0
    )
    shared_receipt_reuse_passed = bool(
        len(shared_bindings) == REQUIRED_SLOT_COUNT
        and shared_new_selector_calls == 0
        and [row["binding_hash"] for row in shared_bindings]
        == [row.binding_hash for row in root_bindings]
    )
    typed_recursive_repair_passed = bool(
        repaired is not None
        and repaired_binding is not None
        and repair_tree.recursion_depth == 1
        and repaired.parent_id == root.id
        and repaired_binding.parent_program_hash
        == root_binding.program_identity_hash
        and repaired_binding.snapshot_hash == root_binding.snapshot_hash
        and repaired_binding.recipe_id != root_binding.recipe_id
        and repair_new_selector_calls == 1
        and recursive_archive_round_trip_passed
        and recursive_archive_binding_hashes
        == (root_binding.binding_hash, repaired_binding.binding_hash)
    )
    compiler_provenance_passed = bool(
        len(compiler_binding_hashes) == REQUIRED_SLOT_COUNT + 1
        and tuple(compiler_binding_hashes)
        == tuple(
            proposer.typed_program_registry.require(row).binding_hash
            for row in compiler_programs
        )
        and compiler_issue_count == 0
        and set(compiled_binding_hashes) == set(compiler_binding_hashes)
        and compiled_binding_set_hash
        == compile_manifest.get("typed_binding_set_hash")
        and compiled_snapshot_ledger_hash
        == ledger.production_snapshot_ledger.ledger_hash
        and compile_manifest.get("typed_snapshot_ledger_hash")
        == ledger.production_snapshot_ledger.ledger_hash
        and compile_manifest_hash == compile_result.manifest_hash
        and len(compiler_event_rows) > 0
        and set(compiler_event_binding_hashes)
        == set(compiler_binding_hashes)
        and all(
            row.get("payload", {}).get("compile_manifest_hash")
            == compile_manifest_hash
            and row.get("payload", {}).get("typed_binding_set_hash")
            == compiled_binding_set_hash
            and set(
                row.get("payload", {}).get("typed_binding_hashes") or []
            )
            == set(compiler_binding_hashes)
            and row.get("payload", {}).get(
                "typed_snapshot_ledger_hash"
            )
            == ledger.production_snapshot_ledger.ledger_hash
            for row in compiler_event_rows
        )
        and len(runtime_source_receipt_rows)
        == len(compile_split_manifest.train_ids)
        == 38
        and runtime_source_routed_count + runtime_source_no_skill_count
        == len(runtime_source_receipt_rows)
        and runtime_source_routed_count > 0
        and all(
            row["compile_manifest_hash"] == compile_manifest_hash
            and row["typed_binding_set_hash"]
            == compiled_binding_set_hash
            and row["typed_snapshot_ledger_hash"]
            == ledger.production_snapshot_ledger.ledger_hash
            for row in runtime_source_receipt_rows
        )
        and runtime_source_mutation_rejected
        and len(lowered_action_hashes) == REQUIRED_SLOT_COUNT + 1
    )
    fixed_tamper_probes_passed = bool(
        tuple(row["probe_id"] for row in tamper_rows) == _TAMPER_PROBE_IDS
        and all(row["rejected"] for row in tamper_rows)
    )
    deterministic_replay_passed = bool(
        replayed_programs == programs and replay_new_selector_calls == 0
    )
    multi_generation_diversity_passed = bool(
        len(next_generation_programs) == REQUIRED_SLOT_COUNT
        and next_generation_new_selector_calls == REQUIRED_SLOT_COUNT
        and hypothesis_count_before_next_generation == 0
        and typed_binding_count_before_next_generation == 0
        and set(attempt_history_binding_hashes_before_next_generation)
        == {row.binding_hash for row in root_bindings}
        and all(
            next_binding.snapshot_hash == root_row.snapshot_hash
            and next_binding.recipe_id != root_row.recipe_id
            and next_binding.selection_round == 2
            and next_binding.excluded_recipe_ids == (root_row.recipe_id,)
            for root_row, next_binding in zip(
                root_bindings,
                next_generation_bindings,
            )
        )
        and cross_generation_repaired is not None
        and cross_generation_repaired_binding is not None
        and cross_generation_repair_tree.recursion_depth == 1
        and cross_generation_repaired_binding.parent_program_hash
        == next_generation_root_binding.program_identity_hash
        and cross_generation_repaired_binding.snapshot_hash
        == next_generation_root_binding.snapshot_hash
        and cross_generation_repaired_binding.recipe_id
        not in {
            root_bindings[0].recipe_id,
            next_generation_root_binding.recipe_id,
        }
        and set(cross_generation_repaired_binding.excluded_recipe_ids)
        >= {
            root_bindings[0].recipe_id,
            next_generation_root_binding.recipe_id,
        }
        and cross_generation_repair_new_selector_calls == 1
    )
    offline_boundary_issues = _offline_boundary_contract_issues(
        audit_sink.events,
        preregistration=preregistration,
    )
    acceptance = {
        "upstream_feasibility_binding_passed": True,
        "source_reconstruction_passed": source_reconstruction_passed,
        "snapshot_external_ledger_passed": (
            snapshot_external_ledger_passed
        ),
        "production_protocol_loader_path_passed": (
            production_protocol_loader_path_passed
        ),
        "production_selection_path_passed": (
            production_selection_path_passed
        ),
        "closed_model_surface_passed": closed_model_surface_passed,
        "shared_receipt_reuse_passed": shared_receipt_reuse_passed,
        "typed_recursive_repair_passed": typed_recursive_repair_passed,
        "compiler_provenance_passed": compiler_provenance_passed,
        "fixed_tamper_probes_passed": fixed_tamper_probes_passed,
        "deterministic_replay_passed": deterministic_replay_passed,
        "multi_generation_diversity_passed": (
            multi_generation_diversity_passed
        ),
        "offline_boundary_contract_passed": not offline_boundary_issues,
    }
    if set(acceptance) != _ACCEPTANCE_KEYS:
        raise RuntimeError("typed-selection acceptance contract drifted")
    integration_passed = all(acceptance.values())
    decision_evidence = {
        "preregistration_hash": preregistration_hash,
        "implementation_file_set_hash": implementation_file_set_hash,
        "upstream_binding_hash": upstream["binding_hash"],
        "ledger_hash": ledger.ledger_hash,
        "program_hashes": [row.payload_hash for row in programs],
        "binding_hashes": [row.binding_hash for row in root_bindings],
        "archive_hash": archive_payload["archive_hash"],
        "archive_binding_set_hash": archive_binding_set_hash,
        "recursive_archive_hash": recursive_archive_payload[
            "archive_hash"
        ],
        "recursive_archive_binding_hashes": list(
            recursive_archive_binding_hashes
        ),
        "next_generation_binding_hashes": [
            row.binding_hash for row in next_generation_bindings
        ],
        "attempt_history_binding_hashes_before_next_generation": list(
            attempt_history_binding_hashes_before_next_generation
        ),
        "hypothesis_count_before_next_generation": (
            hypothesis_count_before_next_generation
        ),
        "typed_binding_count_before_next_generation": (
            typed_binding_count_before_next_generation
        ),
        "cross_generation_repair_binding_hash": (
            cross_generation_repaired_binding.binding_hash
            if cross_generation_repaired_binding is not None
            else None
        ),
        "shared_receipt_hash": shared_receipt_hash,
        "repair_program_hash": (
            repaired.payload_hash if repaired is not None else None
        ),
        "repair_binding_hash": (
            repaired_binding.binding_hash
            if repaired_binding is not None
            else None
        ),
        "compiler_binding_hashes": list(compiler_binding_hashes),
        "compiler_event_set_hash": compiler_event_set_hash,
        "compiler_binding_coverage_hash": compiler_binding_coverage_hash,
        "runtime_source_receipt_set_hash": (
            runtime_source_receipt_set_hash
        ),
        "tamper_probe_set_hash": stable_hash({"probes": tamper_rows}),
        "predecision_event_set_hash": stable_hash(
            {"events": audit_sink.events}
        ),
        "acceptance": acceptance,
        "integration_passed": integration_passed,
    }
    decision_hash = stable_hash(decision_evidence)
    report: dict[str, Any] = {
        "integration_policy": TYPED_SELECTION_INTEGRATION_VERSION,
        "integration_only": True,
        "non_scoring": True,
        "integration_passed": integration_passed,
        "decision_hash": decision_hash,
        "preregistration": {
            "preregistration_hash": preregistration_hash,
            "implementation_file_set_hash": implementation_file_set_hash,
            "decision_budget": 1,
            "decision_ordinal": 1,
            "acceptance_contract_hash": stable_hash(
                preregistration["acceptance"]
            ),
            "acceptance_predicate_hash": stable_hash(
                preregistration["acceptance_predicates"]
            ),
            "no_acceptance_adaptation_after_decision": True,
        },
        "upstream_feasibility": upstream["safe_summary"],
        "frozen_snapshot_ledger": ledger.safe_payload(),
        "production_protocol_loader": protocol_loader_receipt,
        "production_selection": {
            "policy": TYPED_RECIPE_PROPOSAL_FORMATION_POLICY_VERSION,
            "candidate_count": len(programs),
            "candidate_set_hash": stable_hash(
                {"candidate_hashes": [row.payload_hash for row in programs]}
            ),
            "program_hashes": [row.payload_hash for row in programs],
            "binding_set_hash": stable_hash(
                {"binding_hashes": [row.binding_hash for row in root_bindings]}
            ),
            "binding_hashes": [row.binding_hash for row in root_bindings],
            "archive_hash": archive_payload["archive_hash"],
            "archive_binding_set_hash": archive_binding_set_hash,
            "archive_binding_count": len(archive_bindings),
            "archive_binding_round_trip_passed": True,
            "recursive_archive_hash": recursive_archive_payload[
                "archive_hash"
            ],
            "recursive_archive_binding_hashes": list(
                recursive_archive_binding_hashes
            ),
            "recursive_archive_round_trip_passed": (
                recursive_archive_round_trip_passed
            ),
            "selected_recipe_hashes": [
                stable_hash({"recipe_id": row.recipe_id})
                for row in root_bindings
            ],
            "root_selector_call_count": root_selector_call_count,
            "replay_new_selector_call_count": replay_new_selector_calls,
            "next_generation_new_selector_call_count": (
                next_generation_new_selector_calls
            ),
            "attempt_history_binding_hashes_before_next_generation": list(
                attempt_history_binding_hashes_before_next_generation
            ),
            "hypothesis_count_before_next_generation": (
                hypothesis_count_before_next_generation
            ),
            "typed_binding_count_before_next_generation": (
                typed_binding_count_before_next_generation
            ),
            "next_generation_program_hashes": [
                row.payload_hash for row in next_generation_programs
            ],
            "next_generation_binding_hashes": [
                row.binding_hash for row in next_generation_bindings
            ],
            "next_generation_selection_rounds": [
                row.selection_round for row in next_generation_bindings
            ],
            "next_generation_excluded_recipe_sets": [
                list(row.excluded_recipe_ids)
                for row in next_generation_bindings
            ],
            "cross_generation_repair_new_selector_call_count": (
                cross_generation_repair_new_selector_calls
            ),
            "cross_generation_repair_program_hash": (
                cross_generation_repaired.payload_hash
                if cross_generation_repaired is not None
                else None
            ),
            "cross_generation_repair_binding_hash": (
                cross_generation_repaired_binding.binding_hash
                if cross_generation_repaired_binding is not None
                else None
            ),
            "cross_generation_repair_recipe_exclusion_preserved": bool(
                cross_generation_repaired_binding is not None
                and root_bindings[0].recipe_id
                in cross_generation_repaired_binding.excluded_recipe_ids
            ),
            "local_deterministic_selector": True,
            "live_model_invoked": False,
            "model_authored_primitive_count": 0,
            "harness_owned_materialization": True,
            "raw_content_persisted": False,
        },
        "closed_model_surface": {
            "request_count": len(selector.requests),
            "request_set_hash": stable_hash(
                {
                    "request_hashes": [
                        stable_hash(row) for row in selector.requests
                    ]
                }
            ),
            "response_set_hash": stable_hash(
                {
                    "response_hashes": [
                        stable_hash(row) for row in selector.responses
                    ]
                }
            ),
            "request_surface_issues": list(request_surface_issues),
            "raw_artifact_locator_disclosure_count": (
                raw_locator_disclosure_count
            ),
            "raw_observed_primitive_disclosure_count": (
                raw_primitive_disclosure_count
            ),
            "model_output_fields": ["recipe_id"],
            "raw_content_persisted": False,
        },
        "shared_proposal_receipt": {
            **shared_receipt,
            "shared_receipt_hash": shared_receipt_hash,
        },
        "typed_recursive_repair": {
            "attempt_count": 1,
            "repair_new_selector_call_count": repair_new_selector_calls,
            "recursion_depth": repair_tree.recursion_depth,
            "root_program_hash": root.payload_hash,
            "root_binding_hash": root_binding.binding_hash,
            "repair_program_hash": (
                repaired.payload_hash if repaired is not None else None
            ),
            "repair_binding_hash": (
                repaired_binding.binding_hash
                if repaired_binding is not None
                else None
            ),
            "same_snapshot": bool(
                repaired_binding is not None
                and repaired_binding.snapshot_hash
                == root_binding.snapshot_hash
            ),
            "different_recipe": bool(
                repaired_binding is not None
                and repaired_binding.recipe_id != root_binding.recipe_id
            ),
            "generic_free_text_repair_used": False,
            "raw_content_persisted": False,
        },
        "compiler_provenance": {
            "program_count": len(compiler_programs),
            "compiler_binding_hashes": list(compiler_binding_hashes),
            "binding_set_hash": stable_hash(
                {"binding_hashes": list(compiler_binding_hashes)}
            ),
            "lowered_action_set_hash": stable_hash(
                {"lowered_action_hashes": lowered_action_hashes}
            ),
            "compiler_issue_count": compiler_issue_count,
            "binding_hashes_returned_by_compiler": True,
            "compile_manifest_hash": compile_manifest_hash,
            "compiled_binding_hashes": list(compiled_binding_hashes),
            "compiled_binding_set_hash": compiled_binding_set_hash,
            "compiled_snapshot_ledger_hash": (
                compiled_snapshot_ledger_hash
            ),
            "compiler_event_count": len(compiler_event_rows),
            "compiler_event_set_hash": compiler_event_set_hash,
            "compiler_event_path_normalization": (
                "item_hypothesis_content_route_v1"
            ),
            "compiler_binding_coverage_hash": (
                compiler_binding_coverage_hash
            ),
            "compiler_event_binding_hashes": list(
                compiler_event_binding_hashes
            ),
            "runtime_source_receipt_count": len(
                runtime_source_receipt_rows
            ),
            "runtime_source_receipt_set_hash": (
                runtime_source_receipt_set_hash
            ),
            "runtime_source_routed_count": runtime_source_routed_count,
            "runtime_source_no_skill_count": runtime_source_no_skill_count,
            "runtime_source_receipt_rows": list(
                runtime_source_receipt_rows
            ),
            "runtime_source_mutation_rejected": (
                runtime_source_mutation_rejected
            ),
            "compile_manifest_provenance_persisted": True,
            "capability_implementation_verified": False,
            "restricted_runtime_executor_claimed": False,
            "raw_content_persisted": False,
        },
        "tamper_probes": {
            "probe_count": len(tamper_rows),
            "rejection_count": sum(row["rejected"] for row in tamper_rows),
            "probe_set_hash": stable_hash({"probes": tamper_rows}),
            "probes": tamper_rows,
            "raw_content_persisted": False,
        },
        "offline_boundary_contract": {
            "predecision_event_count": len(audit_sink.events),
            "predecision_event_counts": dict(
                sorted(
                    Counter(
                        str(row.get("event") or "")
                        for row in audit_sink.events
                    ).items()
                )
            ),
            "predecision_event_set_hash": stable_hash(
                {"events": audit_sink.events}
            ),
            "issues": list(offline_boundary_issues),
            "stored_offline_train_outcomes_used": True,
            "local_deterministic_selector_call_count": selector.call_count,
            "tamper_local_selector_call_count": 3,
            "live_model_call_count": 0,
            "backend_call_count": 0,
            "evaluator_call_count": 0,
            "validation_task_count": 0,
            "validation_split_accessed": False,
            "test_split_accessed": False,
            "sealed_split_accessed": False,
            "verifier_content_accessed": False,
            "promotion_policy_evaluated": False,
            "raw_content_persisted": False,
        },
        "acceptance": acceptance,
        "fresh_development_protocol_freeze_eligible_if_passed": (
            integration_passed
        ),
        "development_task_execution_currently_authorized": False,
        "controls_currently_authorized": False,
        "family_out_currently_authorized": False,
        "hipporag_transfer_currently_authorized": False,
        "sealed_test_currently_authorized": False,
        "promotion_gate_or_score": False,
        "model_call_count": 0,
        "backend_call_count": 0,
        "evaluator_call_count": 0,
        **_BOUNDARY_FLAGS,
    }
    report["report_hash"] = stable_hash(report)
    _emit(
        audit_sink,
        event="typed_selection_integration_completed",
        trace_id=trace_id,
        payload={
            "integration_policy": TYPED_SELECTION_INTEGRATION_VERSION,
            "decision_hash": decision_hash,
            "report_hash": report["report_hash"],
            "integration_passed": integration_passed,
            "acceptance_hash": stable_hash(acceptance),
            "ledger_hash": ledger.ledger_hash,
            "shared_receipt_hash": shared_receipt_hash,
            "promotion_gate_or_score": False,
        },
    )
    return report


def _validation_context(
    ledger: FrozenTypedSelectionLedger,
) -> ValidationContext:
    return ValidationContext(
        evaluator_epoch=TYPED_SELECTION_EVALUATOR_EPOCH,
        residuals=ledger.evidence.residuals,
        available_lanes=frozenset({"baseline", "candidate"}),
        baseline_lane="baseline",
        trigger_feature_catalog=build_trigger_feature_catalog(
            ledger.evidence.residuals
        ),
        allowed_runtime_kinds=frozenset(HypothesisKind),
        allowed_action_operations=SKILLLEARN_ALLOWED_ACTION_OPERATIONS,
        action_semantics="skilllearn_prompt_directive_lowering_v2",
        external_evidence_is_hidden=True,
        action_design_profiles=ledger.evidence.action_profiles,
        typed_selection_snapshots=ledger.snapshots,
        typed_selection_ledger_hash=(
            ledger.production_snapshot_ledger.ledger_hash
        ),
    )


def _canonical_compiler_event_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Remove only the ephemeral temporary-root component from compile events."""

    canonical: list[dict[str, Any]] = []
    for row in rows:
        payload = row.get("payload")
        if not isinstance(payload, Mapping):
            raise PermissionError("typed compiler event payload is malformed")
        normalized_payload = dict(payload)
        if "skill_path_hash" not in normalized_payload:
            raise PermissionError(
                "typed compiler event path receipt is missing"
            )
        normalized_payload["skill_path_hash"] = stable_hash(
            {
                "route_policy": "item_hypothesis_content_route_v1",
                "item_id_hash": normalized_payload.get("item_id_hash"),
                "hypothesis_id": normalized_payload.get("hypothesis_id"),
                "skill_content_hash": normalized_payload.get(
                    "skill_content_hash"
                ),
            }
        )
        canonical.append(
            Event(
                event=str(row.get("event") or ""),
                stage=str(row.get("stage") or ""),
                trace_id=str(row.get("trace_id") or ""),
                payload=normalized_payload,
            ).to_dict()
        )
    return tuple(canonical)


def _kernel(
    *,
    proposer: StructuredHypothesisProposer,
    ledger: FrozenTypedSelectionLedger,
    event_sink: EventSink,
) -> EvolutionKernel:
    return EvolutionKernel(
        proposer=proposer,
        validator=RecursiveValidationEngine(
            (), proposer=proposer, event_sink=event_sink
        ),
        counterfactual_runner=SimpleNamespace(
            evaluator=SimpleNamespace(epoch=TYPED_SELECTION_EVALUATOR_EPOCH)
        ),
        promotion_gate=SimpleNamespace(
            spec=SimpleNamespace(metric="task_success")
        ),
        archive=PolicyArchive(event_sink=event_sink),
        split_guard=SimpleNamespace(authorize=lambda *_: None),
        proposal_candidates_per_generation=REQUIRED_SLOT_COUNT,
        proposal_formation_policy=(
            TYPED_RECIPE_PROPOSAL_FORMATION_POLICY_VERSION
        ),
        typed_selection_snapshots=ledger.snapshots,
        typed_selection_ledger=ledger.production_snapshot_ledger,
        event_sink=event_sink,
    )


def _run_tamper_probes(
    *,
    kernel: EvolutionKernel,
    proposer: StructuredHypothesisProposer,
    compiler: SkillLearnProgramCompiler,
    programs: Sequence[HypothesisProgram],
    snapshots: Sequence[TypedRecipeSelectionSnapshot],
    production_snapshot_ledger: TypedSelectionSnapshotLedger,
    context: ValidationContext,
    selector: _OfflineRecipeSelector,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def record_exception(probe_id: str, function: Any) -> None:
        rejected = False
        error_type_hash: str | None = None
        try:
            function()
        except (HypothesisProposalCallError, PermissionError, ValueError) as exc:
            rejected = True
            error_type_hash = stable_hash({"error_type": type(exc).__name__})
        rows.append(
            {
                "probe_id": probe_id,
                "rejected": rejected,
                "error_type_hash": error_type_hash,
            }
        )

    first_snapshot = snapshots[0]
    unknown_recipe = "recipe_" + ("f" * 64)
    for probe_id, response in (
        (
            "root_response_extra_field",
            {
                "recipe_id": first_snapshot.graph.recipes[0].recipe_id,
                "free_text_action": "forbidden",
            },
        ),
        ("root_response_unknown_recipe", {"recipe_id": unknown_recipe}),
        ("root_response_non_string_recipe", {"recipe_id": 7}),
    ):
        model = _FixedResponseSelector(
            response,
            expected_ledger_hash=production_snapshot_ledger.ledger_hash,
        )
        probe_proposer = StructuredHypothesisProposer(
            model,
            event_sink=NullEventSink(),
            typed_program_registry=TypedProgramBindingRegistry(
                snapshot_ledger=production_snapshot_ledger
            ),
        )
        record_exception(
            probe_id,
            lambda p=probe_proposer: p.select_typed_recipe(
                snapshot=first_snapshot,
                evaluator_epoch=TYPED_SELECTION_EVALUATOR_EPOCH,
                trace_id=f"tamper:{probe_id}",
            ),
        )

    record_exception(
        "shared_batch_missing_candidate",
        lambda: kernel.validate_typed_shared_proposal_candidates(
            programs[:-1], trace_id="tamper:shared-missing"
        ),
    )
    record_exception(
        "shared_batch_reordered",
        lambda: kernel.validate_typed_shared_proposal_candidates(
            tuple(reversed(programs)), trace_id="tamper:shared-reordered"
        ),
    )
    mutated = replace(
        programs[0],
        statement="tampered executable statement",
    )
    record_exception(
        "shared_program_executable_mutation",
        lambda: kernel.validate_typed_shared_proposal_candidates(
            (mutated, *programs[1:]), trace_id="tamper:shared-executable"
        ),
    )
    unbound = replace(programs[0], id="typed-unbound-compiler-probe")
    record_exception(
        "compiler_unbound_program",
        lambda: compiler.require_program_bindings((unbound,)),
    )
    record_exception(
        "compiler_executable_mutation",
        lambda: compiler.require_program_bindings((mutated,)),
    )

    before = selector.call_count
    indirect_tree = RecursiveValidationEngine(
        (_RootOnlyFailureCheck(unbound.id),),
        proposer=proposer,
        event_sink=NullEventSink(),
    ).validate(
        unbound,
        context,
        trace_id="tamper:recursive-unbound",
    )
    indirect_rejected = bool(
        indirect_tree.accepted_program is None
        and indirect_tree.nodes
        and indirect_tree.nodes[0].terminal_reason == "typed_snapshot_unbound"
        and selector.call_count == before
    )
    rows.append(
        {
            "probe_id": "recursive_repair_unbound_parent",
            "rejected": indirect_rejected,
            "error_type_hash": None,
        }
    )

    record_exception(
        "direct_repair_unbound_parent",
        lambda: proposer.select_typed_recipe(
            snapshot=first_snapshot,
            evaluator_epoch=TYPED_SELECTION_EVALUATOR_EPOCH,
            trace_id="tamper:direct-unbound",
            parent=unbound,
            failed_checks=(
                {
                    "check": "forced",
                    "passed": False,
                    "reason": "tamper_probe",
                    "evidence": {},
                },
            ),
            depth=1,
        ),
    )
    tampered_ledger = replace(
        production_snapshot_ledger,
        feasibility_report_hash=stable_hash(
            {"tampered": "feasibility-report"}
        ),
    )
    record_exception(
        "snapshot_ledger_rebinding",
        lambda: proposer.typed_program_registry.bind_snapshot_ledger(
            tampered_ledger
        ),
    )
    return rows


def _validate_closed_selector_request(
    request: Mapping[str, Any],
    *,
    expected_ledger_hash: str,
) -> None:
    issues = _selector_request_surface_issues(
        request,
        expected_ledger_hash=expected_ledger_hash,
    )
    if issues:
        raise PermissionError(
            f"offline typed selector request is not closed: {list(issues)}"
        )


def _selector_request_surface_issues(
    request: Mapping[str, Any],
    *,
    expected_ledger_hash: str,
) -> tuple[str, ...]:
    issues: list[str] = []
    request_kind = request.get("request_kind")
    root_keys = {
        "request_kind",
        "contract_version",
        "grammar_version",
        "evaluator_epoch",
        "selection_scope",
        "selection_snapshot",
        "selection_authority",
        "output_schema",
        "constraints",
    }
    expected_keys = (
        root_keys | {"repair_context"}
        if request_kind == "select_typed_repair_recipe"
        else root_keys
    )
    if set(request) != expected_keys:
        issues.append("selector_top_level_fields_not_closed")
    schema = request.get("output_schema")
    if not isinstance(schema, Mapping) or set(schema) != {
        "type",
        "additionalProperties",
        "required",
        "properties",
    }:
        issues.append("selector_schema_envelope_invalid")
    else:
        properties = schema.get("properties")
        recipe_schema = (
            properties.get("recipe_id")
            if isinstance(properties, Mapping)
            else None
        )
        recipe_ids = (
            recipe_schema.get("enum")
            if isinstance(recipe_schema, Mapping)
            else None
        )
        if (
            schema.get("type") != "object"
            or schema.get("additionalProperties") is not False
            or schema.get("required") != ["recipe_id"]
            or not isinstance(properties, Mapping)
            or set(properties) != {"recipe_id"}
            or not isinstance(recipe_schema, Mapping)
            or set(recipe_schema) != {"type", "enum"}
            or recipe_schema.get("type") != "string"
            or not isinstance(recipe_ids, list)
            or not recipe_ids
            or recipe_ids != sorted(set(recipe_ids))
            or not all(isinstance(value, str) for value in recipe_ids)
        ):
            issues.append("selector_recipe_id_schema_invalid")
    constraints = request.get("constraints")
    expected_constraints = {
        "model_output_fields": ["recipe_id"],
        "primitive_values_model_authored": False,
        "artifact_locators_model_authored": False,
        "free_text_actions_model_authored": False,
        "harness_owned_materialization": True,
    }
    if constraints != expected_constraints:
        issues.append("selector_constraints_drifted")
    selection_scope = request.get("selection_scope")
    excluded_recipe_ids: list[str] = []
    if not isinstance(selection_scope, Mapping) or set(selection_scope) != {
        "selection_round",
        "excluded_recipe_ids",
        "excluded_recipe_set_hash",
        "excluded_recipe_count",
    }:
        issues.append("selector_selection_scope_invalid")
    else:
        raw_exclusions = selection_scope.get("excluded_recipe_ids")
        selection_round = selection_scope.get("selection_round")
        if isinstance(raw_exclusions, list):
            excluded_recipe_ids = raw_exclusions
        if (
            isinstance(selection_round, bool)
            or not isinstance(selection_round, int)
            or selection_round < 1
            or not isinstance(raw_exclusions, list)
            or raw_exclusions != sorted(set(raw_exclusions))
            or not all(isinstance(value, str) for value in raw_exclusions)
            or selection_scope.get("excluded_recipe_count")
            != len(raw_exclusions)
            or selection_scope.get("excluded_recipe_set_hash")
            != stable_hash({"recipe_ids": raw_exclusions})
        ):
            issues.append("selector_selection_scope_invalid")
    authority = request.get("selection_authority")
    if not isinstance(authority, Mapping):
        issues.append("selector_authority_missing")
    else:
        authority_payload = dict(authority)
        ledger_hash = authority_payload.pop("ledger_hash", None)
        if (
            ledger_hash != expected_ledger_hash
            or stable_hash(authority_payload) != ledger_hash
            or authority.get("raw_content_persisted") is not False
        ):
            issues.append("selector_authority_invalid")
    snapshot = request.get("selection_snapshot")
    if not isinstance(snapshot, Mapping):
        issues.append("selector_snapshot_missing")
    else:
        catalog = snapshot.get("catalog")
        if not isinstance(catalog, Mapping):
            issues.append("selector_catalog_missing")
        elif (
            catalog.get("raw_artifact_locators_disclosed") is not False
            or catalog.get("raw_executables_disclosed") is not False
            or catalog.get("model_authored_primitive_fields") != []
            or catalog.get("model_output_schema") != schema
        ):
            issues.append("selector_catalog_surface_invalid")
        if isinstance(authority, Mapping) and snapshot.get(
            "snapshot_hash"
        ) not in authority.get("snapshot_hashes", []):
            issues.append("selector_snapshot_outside_authority")
    if isinstance(schema, Mapping):
        properties = schema.get("properties")
        recipe_schema = (
            properties.get("recipe_id")
            if isinstance(properties, Mapping)
            else None
        )
        allowed = (
            recipe_schema.get("enum")
            if isinstance(recipe_schema, Mapping)
            else []
        )
        if any(value in allowed for value in excluded_recipe_ids):
            issues.append("selector_excluded_recipe_still_allowed")
    if "residuals" in request or "action_graph" in request:
        issues.append("selector_raw_training_or_action_content_disclosed")
    if request_kind == "select_typed_repair_recipe":
        repair = request.get("repair_context")
        if not isinstance(repair, Mapping) or (
            repair.get("parent_action_graph_disclosed") is not False
            or repair.get("free_text_repair_fields_allowed") is not False
        ):
            issues.append("selector_repair_surface_invalid")
    return tuple(sorted(set(issues)))


def _request_snapshot_pairs(
    requests: Sequence[Mapping[str, Any]],
    snapshots: Sequence[TypedRecipeSelectionSnapshot],
) -> tuple[tuple[Mapping[str, Any], TypedRecipeSelectionSnapshot], ...]:
    by_hash = {row.snapshot_hash: row for row in snapshots}
    rows: list[tuple[Mapping[str, Any], TypedRecipeSelectionSnapshot]] = []
    for request in requests:
        snapshot_payload = request.get("selection_snapshot")
        snapshot_hash = (
            snapshot_payload.get("snapshot_hash")
            if isinstance(snapshot_payload, Mapping)
            else None
        )
        snapshot = by_hash.get(str(snapshot_hash or ""))
        if snapshot is None:
            raise PermissionError(
                "selector request references an unbound snapshot"
            )
        rows.append((request, snapshot))
    return tuple(rows)


def _verify_upstream_feasibility(
    preregistration: Mapping[str, Any],
    *,
    preregistration_path: str | Path,
) -> dict[str, Any]:
    project_root = _project_root(preregistration_path)
    upstream = preregistration["upstream_feasibility"]
    upstream_prereg_path = _resolve_bound_file(
        project_root, upstream["preregistration"]
    )
    result_receipt_path = _resolve_bound_file(
        project_root, upstream["result_receipt"]
    )
    upstream_prereg = _read_json_object(
        upstream_prereg_path, label="upstream typed feasibility preregistration"
    )
    result_receipt = _read_json_object(
        result_receipt_path, label="upstream typed feasibility result receipt"
    )
    if (
        stable_hash(upstream_prereg)
        != upstream["preregistration_stable_hash"]
        or _sha256_file(upstream_prereg_path)
        != upstream["preregistration_file_sha256"]
        or stable_hash(result_receipt)
        != upstream["result_receipt_stable_hash"]
        or _sha256_file(result_receipt_path)
        != upstream["result_receipt_file_sha256"]
    ):
        raise PermissionError("upstream feasibility receipt binding drifted")
    artifact_rows = upstream["artifacts"]
    paths = {
        key: _resolve_bound_file(project_root, artifact_rows[key]["path"])
        for key in ("report", "events", "decision_lock")
    }
    actual_hashes = {key: _sha256_file(path) for key, path in paths.items()}
    if any(
        actual_hashes[key] != artifact_rows[key]["sha256"]
        for key in actual_hashes
    ):
        raise PermissionError("upstream feasibility artifact hash drifted")
    report = _read_json_object(paths["report"], label="upstream report")
    lock = _read_json_object(paths["decision_lock"], label="upstream lock")
    events = _read_event_rows(paths["events"])
    report_without_hash = dict(report)
    report_hash = report_without_hash.pop("report_hash", None)
    if stable_hash(report_without_hash) != report_hash:
        raise PermissionError("upstream feasibility report self-hash drifted")
    if (
        report_hash != upstream["report_hash"]
        or report.get("decision_hash") != upstream["decision_hash"]
        or report.get("feasibility_passed") is not True
        or not all(report.get("acceptance", {}).values())
        or result_receipt.get("report_hash") != report_hash
        or result_receipt.get("decision_hash") != report.get("decision_hash")
        or result_receipt.get("feasibility_passed") is not True
        or result_receipt.get("exact_replay_verified") is not True
        or lock.get("state") != "completed"
        or lock.get("decision_hash") != report.get("decision_hash")
        or lock.get("report_hash") != report_hash
        or lock.get("preregistration_hash") != stable_hash(upstream_prereg)
    ):
        raise PermissionError("upstream feasibility decision binding drifted")
    for key in ("report", "events", "decision_lock"):
        receipt_path_key = key
        receipt_hash_key = f"{key}_sha256"
        canonical = result_receipt["canonical_artifacts"]
        if (
            canonical.get(receipt_path_key) != artifact_rows[key]["path"]
            or canonical.get(receipt_hash_key) != actual_hashes[key]
        ):
            raise PermissionError(
                "upstream feasibility versioned receipt artifact drifted"
            )
    if (
        len(events) != 9
        or events[-1].get("event") != "typed_operator_feasibility_completed"
        or events[-1].get("payload", {}).get("decision_hash")
        != report.get("decision_hash")
        or events[-1].get("payload", {}).get("report_hash") != report_hash
    ):
        raise PermissionError("upstream feasibility event ledger drifted")
    graph_section = report.get("typed_operator_graph")
    if not isinstance(graph_section, Mapping) or (
        graph_section.get("graph_set_hash")
        != preregistration["expected_graph_set_hash"]
        or graph_section.get("model_catalog_set_hash")
        != preregistration["expected_model_catalog_set_hash"]
    ):
        raise PermissionError("upstream graph ledger binding drifted")
    binding_hash = stable_hash(
        {
            "preregistration_stable_hash": stable_hash(upstream_prereg),
            "result_receipt_stable_hash": stable_hash(result_receipt),
            "artifact_hashes": actual_hashes,
            "decision_hash": report["decision_hash"],
            "report_hash": report_hash,
        }
    )
    return {
        "report": report,
        "binding_hash": binding_hash,
        "safe_summary": {
            "upstream_binding_hash": binding_hash,
            "upstream_preregistration_hash": stable_hash(upstream_prereg),
            "upstream_result_receipt_hash": stable_hash(result_receipt),
            "upstream_result_receipt_file_sha256": _sha256_file(
                result_receipt_path
            ),
            "upstream_decision_hash": report["decision_hash"],
            "upstream_report_hash": report_hash,
            "upstream_report_file_sha256": actual_hashes["report"],
            "upstream_events_file_sha256": actual_hashes["events"],
            "upstream_decision_lock_file_sha256": actual_hashes[
                "decision_lock"
            ],
            "upstream_event_count": len(events),
            "upstream_feasibility_passed": True,
            "upstream_exact_replay_verified": True,
            "raw_content_persisted": False,
        },
    }


def _offline_boundary_contract_issues(
    events: Sequence[Mapping[str, Any]],
    *,
    preregistration: Mapping[str, Any],
) -> tuple[str, ...]:
    issues: list[str] = []
    counts = Counter(str(row.get("event") or "") for row in events)
    expected_counts = preregistration["expected_predecision_event_counts"]
    if dict(sorted(counts.items())) != dict(sorted(expected_counts.items())):
        issues.append("offline_event_type_or_count_mismatch")
    expected_total = sum(expected_counts.values())
    if len(events) != expected_total:
        issues.append("offline_event_total_mismatch")
    forbidden_markers = (
        "backend_invoked",
        "counterfactual_evidence_recorded",
        "promotion_decision",
        "sealed",
    )
    for row in events:
        payload = row.get("payload")
        if not isinstance(payload, Mapping):
            issues.append("offline_event_payload_missing")
            continue
        for key, expected in _BOUNDARY_FLAGS.items():
            if payload.get(key) is not expected:
                issues.append(f"offline_boundary_flag_mismatch:{key}")
        event_name = str(row.get("event") or "")
        if any(marker in event_name for marker in forbidden_markers):
            issues.append("offline_forbidden_event_observed")
    materialized = [
        row
        for row in events
        if row.get("event") == "typed_recipe_selection_materialized"
    ]
    if len(materialized) != (REQUIRED_SLOT_COUNT * 2) + 2 or any(
        row.get("payload", {}).get("harness_owned_materialization") is not True
        or row.get("payload", {}).get("model_authored_primitive_count") != 0
        for row in materialized
    ):
        issues.append("offline_materialization_event_mismatch")
    shared = [
        row
        for row in events
        if row.get("event") == "typed_shared_proposal_batch_validated"
    ]
    if len(shared) != 1 or shared[0].get("payload", {}).get(
        "new_selector_calls"
    ) != 0:
        issues.append("offline_shared_receipt_event_mismatch")
    return tuple(sorted(set(issues)))


def _read_preregistration(
    path: str | Path,
    *,
    allow_unfrozen_implementation: bool = False,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    payload = dict(
        _read_json_object(
            resolved,
            label="typed-selection integration preregistration",
        )
    )
    fixed_values = {
        "integration_policy": TYPED_SELECTION_INTEGRATION_VERSION,
        "decision_budget": 1,
        "decision_scope": "offline_non_scoring_selection_integration_only",
        "typed_proposal_policy": (
            TYPED_RECIPE_PROPOSAL_FORMATION_POLICY_VERSION
        ),
        "evaluator_epoch": TYPED_SELECTION_EVALUATOR_EPOCH,
        "slot_count": REQUIRED_SLOT_COUNT,
        "minimum_family_support": MINIMUM_FAMILY_SUPPORT,
        "fixed_tamper_probe_ids": list(_TAMPER_PROBE_IDS),
        "expected_predecision_event_counts": dict(
            _EXPECTED_PREDECISION_EVENT_COUNTS
        ),
        "acceptance_predicates": dict(_ACCEPTANCE_PREDICATES),
        "acceptance": {key: True for key in _ACCEPTANCE_PREDICATES},
        "boundary_contract": dict(_BOUNDARY_FLAGS),
    }
    for key, expected in fixed_values.items():
        if payload.get(key) != expected:
            raise PermissionError(
                f"typed-selection integration preregistration drifted: {key}"
            )
    if set(payload.get("acceptance", {})) != _ACCEPTANCE_KEYS:
        raise PermissionError("typed-selection acceptance vector drifted")
    for key in (
        "manifest_hash",
        "source_train_receipt_hash",
        "expected_trial_evidence_hash",
        "expected_graph_set_hash",
        "expected_model_catalog_set_hash",
        "expected_implementation_file_set_hash",
    ):
        if not _is_sha256_text(payload.get(key)):
            raise PermissionError(
                f"typed-selection hash binding is malformed: {key}"
            )
    for key in ("manifest", "source_run_root", "source_train_receipt"):
        _safe_relative_path(payload.get(key))
    protocol_probe = payload.get("protocol_contract_probe")
    if (
        not isinstance(protocol_probe, Mapping)
        or set(protocol_probe) != {"template", "template_file_sha256"}
        or not _is_sha256_text(
            protocol_probe.get("template_file_sha256")
        )
    ):
        raise PermissionError(
            "typed-selection protocol contract probe is malformed"
        )
    _safe_relative_path(protocol_probe["template"])
    for key in (
        "expected_target_family_hashes",
        "expected_graph_hashes",
        "expected_model_catalog_hashes",
    ):
        values = payload.get(key)
        if (
            not isinstance(values, list)
            or len(values) != REQUIRED_SLOT_COUNT
            or len(set(values)) != REQUIRED_SLOT_COUNT
            or not all(_is_sha256_text(value) for value in values)
        ):
            raise PermissionError(
                f"typed-selection ordered ledger malformed: {key}"
            )
    upstream = payload.get("upstream_feasibility")
    if not isinstance(upstream, Mapping):
        raise PermissionError("typed-selection upstream ledger missing")
    for key in (
        "preregistration_stable_hash",
        "preregistration_file_sha256",
        "result_receipt_stable_hash",
        "result_receipt_file_sha256",
        "decision_hash",
        "report_hash",
    ):
        if not _is_sha256_text(upstream.get(key)):
            raise PermissionError(
                f"typed-selection upstream hash malformed: {key}"
            )
    artifacts = upstream.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "report",
        "events",
        "decision_lock",
    }:
        raise PermissionError("typed-selection upstream artifacts missing")
    for row in artifacts.values():
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256"}
            or not _is_sha256_text(row.get("sha256"))
        ):
            raise PermissionError(
                "typed-selection upstream artifact binding malformed"
            )
    _canonical_decision_paths(payload, preregistration_path=resolved)
    actual = _implementation_file_set_hash(
        payload,
        preregistration_path=resolved,
    )
    expected = payload["expected_implementation_file_set_hash"]
    placeholder = "0" * 64
    if actual != expected and not (
        allow_unfrozen_implementation and expected == placeholder
    ):
        raise PermissionError(
            "typed-selection integration implementation binding drifted"
        )
    return payload


def _canonical_decision_paths(
    preregistration: Mapping[str, Any],
    *,
    preregistration_path: str | Path,
) -> dict[str, Path]:
    declared = preregistration.get("canonical_decision_paths")
    if not isinstance(declared, Mapping) or set(declared) != {
        "report",
        "events",
        "decision_lock",
        "result_receipt",
    }:
        raise PermissionError("typed-selection canonical paths are missing")
    project_root = _project_root(preregistration_path)
    result: dict[str, Path] = {}
    for key in (
        "report",
        "events",
        "decision_lock",
        "result_receipt",
    ):
        relative = _safe_relative_path(declared[key])
        resolved = (project_root / relative).resolve()
        try:
            resolved.relative_to(project_root)
        except ValueError as exc:
            raise PermissionError(
                "typed-selection canonical path escaped project root"
            ) from exc
        result[key] = resolved
    if len(set(result.values())) != 4:
        raise PermissionError("typed-selection canonical paths overlap")
    return result


def _project_root(preregistration_path: str | Path) -> Path:
    return (
        Path(preregistration_path)
        .expanduser()
        .resolve(strict=True)
        .parent.parent
    )


def _safe_relative_path(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PermissionError("typed-selection bound path malformed")
    relative = Path(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise PermissionError("typed-selection bound path unsafe")
    return relative


def _resolve_bound_file(project_root: Path, value: Any) -> Path:
    relative = _safe_relative_path(value)
    current = project_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise PermissionError("typed-selection bound symlink forbidden")
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise PermissionError(
            "typed-selection bound file escaped project root"
        ) from exc
    if not resolved.is_file():
        raise PermissionError("typed-selection bound file missing")
    return resolved


def _resolve_bound_directory(project_root: Path, value: Any) -> Path:
    relative = _safe_relative_path(value)
    current = project_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise PermissionError("typed-selection bound symlink forbidden")
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise PermissionError(
            "typed-selection bound directory escaped project root"
        ) from exc
    if not resolved.is_dir():
        raise PermissionError("typed-selection bound directory missing")
    return resolved


def _require_declared_input_path(
    actual: str | Path,
    *,
    project_root: Path,
    declared: Any,
    label: str,
) -> Path:
    expected = _resolve_bound_file(project_root, declared)
    resolved = Path(actual).expanduser().resolve(strict=True)
    if resolved != expected:
        raise PermissionError(f"typed-selection {label} path is not canonical")
    return resolved


def _require_declared_input_directory(
    actual: str | Path,
    *,
    project_root: Path,
    declared: Any,
    label: str,
) -> Path:
    expected = _resolve_bound_directory(project_root, declared)
    resolved = Path(actual).expanduser().resolve(strict=True)
    if resolved != expected:
        raise PermissionError(f"typed-selection {label} path is not canonical")
    return resolved


def _reserve_decision_lock(
    path: Path,
    *,
    preregistration_hash: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise FileExistsError(
            "typed-selection integration decision budget is already consumed"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "lock_version": (
                    "typed_selection_integration_decision_lock_v1"
                ),
                "decision_ordinal": 1,
                "state": "reserved",
                "preregistration_hash": preregistration_hash,
                "raw_content_persisted": False,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def _verify_decision_lock(
    lock: Mapping[str, Any],
    *,
    report: Mapping[str, Any],
    preregistration_hash: str,
) -> None:
    expected = {
        "lock_version": "typed_selection_integration_decision_lock_v1",
        "decision_ordinal": 1,
        "state": "completed",
        "preregistration_hash": preregistration_hash,
        "decision_hash": report["decision_hash"],
        "report_hash": report["report_hash"],
        "integration_passed": report["integration_passed"],
        "raw_content_persisted": False,
    }
    if dict(lock) != expected:
        raise PermissionError(
            "typed-selection integration decision lock does not match"
        )


def _emit(
    sink: EventSink,
    *,
    event: str,
    trace_id: str,
    payload: Mapping[str, Any],
) -> None:
    sink.emit(
        Event(
            event=event,
            stage="benchmark.skilllearn.typed_selection_integration",
            trace_id=trace_id,
            payload=dict(payload),
        )
    )


def _read_event_rows(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        parsed = json.loads(raw)
        if not isinstance(parsed, Mapping):
            raise PermissionError(
                "typed-selection event is malformed at line "
                f"{line_number}"
            )
        rows.append(parsed)
    if not rows:
        raise PermissionError("typed-selection event ledger is empty")
    return rows


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


def _is_sha256_text(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run or exactly replay the single preregistered offline, "
            "non-scoring production typed-selection integration diagnostic."
        )
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-run-root", type=Path)
    parser.add_argument("--source-train-receipt", type=Path)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--print-implementation-hash", action="store_true")
    args = parser.parse_args(argv)

    if args.print_implementation_hash:
        preregistration = _read_preregistration(
            args.preregistration,
            allow_unfrozen_implementation=True,
        )
        print(
            _implementation_file_set_hash(
                preregistration,
                preregistration_path=args.preregistration,
            )
        )
        return
    required = {
        "--root": args.root,
        "--manifest": args.manifest,
        "--source-run-root": args.source_run_root,
        "--source-train-receipt": args.source_train_receipt,
        "--events": args.events,
        "--out": args.out,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))
    kwargs = {
        "root": args.root,
        "manifest_path": args.manifest,
        "source_run_root": args.source_run_root,
        "source_train_receipt": args.source_train_receipt,
        "preregistration_path": args.preregistration,
        "report_path": args.out,
        "events_path": args.events,
    }
    if args.verify_existing:
        report = verify_existing_typed_selection_integration(**kwargs)
    else:
        report = run_typed_selection_integration(**kwargs)
    print(
        json.dumps(
            {
                "decision_hash": report["decision_hash"],
                "integration_passed": report["integration_passed"],
                "integration_reuse_verified": report.get(
                    "integration_reuse_verified", False
                ),
                "model_call_count": report["model_call_count"],
                "backend_call_count": report["backend_call_count"],
                "evaluator_call_count": report["evaluator_call_count"],
            },
            sort_keys=True,
        )
    )
    if not report["integration_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
