from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..evolution import TYPED_RECIPE_PROPOSAL_FORMATION_POLICY_VERSION
from ..events import Event, EventSink, JsonlEventSink, MemoryEventSink
from ..models import HypothesisProgram, HypothesisStatus, SplitName, stable_hash
from ..proposer import HypothesisProposalCallError, StructuredHypothesisProposer
from ..splits import BenchmarkItem, SplitManifest
from ..typed_operator_grammar import (
    MAX_REGISTERED_ARTIFACTS_PER_FAMILY,
    FamilyCapabilityGraph,
    TypedProgramBindingRegistry,
    build_family_capability_graph,
    freeze_typed_recipe_selection_snapshot,
    freeze_typed_selection_snapshot_ledger,
)
from . import typed_task_capability as _portable_capability
from .skilllearn_compiler import SkillLearnProgramCompiler
from .skilllearn_lifecycle import (
    PRE_AGENT_TASK_CAPABILITY_RUNTIME_VERSION,
    PortableTaskCapabilityRuntimeContext,
    SkillLearnSubprocessBackend,
    SkillLearnTrialRequest,
    TrialVariant,
)
from .train_proposal_diagnostic import reconstruct_v315_train_evidence
from .typed_operator_feasibility import _extract_all_train_trials
from .typed_selection_integration import (
    TYPED_SELECTION_INTEGRATION_VERSION,
    FrozenTypedSelectionLedger,
    _FixedResponseSelector,
    _OfflineRecipeSelector,
    _canonical_compiler_event_rows,
    _kernel,
    _validation_context,
)


TYPED_PORTABLE_INTEGRATION_VERSION = (
    "v315_train_projected_typed_portable_production_integration_v1"
)
TYPED_PORTABLE_RESULT_RECEIPT_VERSION = (
    "typed_portable_integration_result_receipt_v1"
)
TYPED_PORTABLE_EVALUATOR_EPOCH = "typed-portable-integration-v1"
PORTABLE_GRAPH_PROJECTION_POLICY = (
    "receipt_bound_portable_artifact_evidence_role_projection_v1"
)
REQUIRED_SLOT_COUNT = 3
MINIMUM_FAMILY_SUPPORT = 2

_TAMPER_PROBE_IDS = (
    "selector_extra_field",
    "selector_full_graph_only_recipe",
    "projected_ledger_rebinding",
    "compiled_metadata_mutation",
)

_DIAGNOSTIC_TYPED_SOURCE_FIELDS = frozenset(
    {
        "preregistration",
        "preregistration_file_sha256",
        "source_run_root",
        "source_train_receipt",
        "source_train_receipt_file_sha256",
        "snapshot_ledger_hash",
    }
)

_ACCEPTANCE_PREDICATES: Mapping[str, str] = {
    "historical_feasibility_evidence_passed": (
        "the old feasibility preregistration, result receipt, report, event "
        "ledger and completed decision lock match externally preregistered "
        "hashes and are used only as historical evidence"
    ),
    "current_train_reconstruction_passed": (
        "the bound TRAIN source reconstructs the exact trial ledger and full "
        "family graphs under the current implementation without agent replay"
    ),
    "portable_projection_closure_passed": (
        "each projected selector catalog is a deterministic child of its "
        "historical full graph and every selectable fixed typed plan has a "
        "registered read-only portable artifact-evidence role; this does not "
        "claim that its recipe operators are implemented"
    ),
    "production_authorization_loader_route_passed": (
        "the production v3.20 typed-selection authorization loader returns "
        "the same projected diagnostic-only ledger without requiring a "
        "result receipt, while the legacy integration policy and legacy full "
        "snapshot ledger route both fail closed"
    ),
    "opaque_production_selection_passed": (
        "the real StructuredHypothesisProposer and EvolutionKernel select only "
        "opaque recipe_id values from the projected three-slot ledger"
    ),
    "full_bundle_compile_passed": (
        "the real typed compiler binds all three selected typed plans and "
        "compiles their receipt-bound portable evidence-role metadata for "
        "every preregistered canary, without claiming operator execution"
    ),
    "pre_agent_runtime_passed": (
        "each exact frozen validation image executes the production injection "
        "hook with network none and verifies only the read-only evidence "
        "profile or inventory sidecar before any agent starts; no task or "
        "recipe-operator effect is claimed"
    ),
    "fixed_tamper_probes_passed": (
        "all preregistered selector, projection-ledger and compiled-metadata "
        "tamper probes fail closed"
    ),
    "offline_non_scoring_boundary_passed": (
        "the diagnostic uses stored TRAIN evidence and public validation inputs "
        "only, with no model, task run, evaluator, verifier, score or promotion"
    ),
}
_ACCEPTANCE_KEYS = frozenset(_ACCEPTANCE_PREDICATES)

_REQUIRED_IMPLEMENTATION_FILES = frozenset(
    {
        "assumption_agent/archive.py",
        "assumption_agent/events.py",
        "assumption_agent/evolution.py",
        "assumption_agent/models.py",
        "assumption_agent/proposer.py",
        "assumption_agent/splits.py",
        "assumption_agent/typed_operator_grammar.py",
        "assumption_agent/validation.py",
        "assumption_agent/benchmarks/skilllearn_compiler.py",
        "assumption_agent/benchmarks/skilllearn_experiment.py",
        "assumption_agent/benchmarks/skilllearn_lifecycle.py",
        "assumption_agent/benchmarks/skilllearnbench.py",
        "assumption_agent/benchmarks/paper_protocol.py",
        "assumption_agent/benchmarks/train_proposal_diagnostic.py",
        "assumption_agent/benchmarks/typed_operator_feasibility.py",
        "assumption_agent/benchmarks/typed_portable_integration.py",
        "assumption_agent/benchmarks/typed_selection_integration.py",
        "assumption_agent/benchmarks/typed_task_capability.py",
    }
)

_BOUNDARY_FLAGS: Mapping[str, bool] = {
    "source_agent_trials_reexecuted": False,
    "stored_offline_train_outcomes_used": True,
    "local_deterministic_selector_used": True,
    "validation_public_input_accessed": True,
    "validation_outcome_accessed": False,
    "live_model_invoked": False,
    "task_backend_run_task_invoked": False,
    "live_evaluator_invoked": False,
    "verifier_content_accessed": False,
    "test_split_accessed": False,
    "sealed_split_accessed": False,
    "promotion_policy_evaluated": False,
    "score_computed": False,
    "task_agent_started": False,
    "container_runtime_invoked": True,
    "container_network_enabled": False,
    "secret_value_persisted": False,
    "raw_content_persisted": False,
}


@dataclass(frozen=True)
class ReconstructedTypedGraphMaterial:
    evidence: Any
    trials: tuple[Any, ...]
    full_graphs: tuple[FamilyCapabilityGraph, ...]
    trial_evidence_hash: str
    full_graph_set_hash: str
    full_model_catalog_set_hash: str


@dataclass(frozen=True)
class HistoricalFeasibilityEvidence:
    preregistration_hash: str
    result_receipt_hash: str
    result_receipt_file_sha256: str
    decision_hash: str
    report_hash: str
    binding_hash: str
    report: Mapping[str, Any]

    def safe_payload(self) -> dict[str, Any]:
        return {
            "historical_preregistration_hash": self.preregistration_hash,
            "historical_result_receipt_hash": self.result_receipt_hash,
            "historical_result_receipt_file_sha256": (
                self.result_receipt_file_sha256
            ),
            "historical_decision_hash": self.decision_hash,
            "historical_report_hash": self.report_hash,
            "historical_binding_hash": self.binding_hash,
            "historical_evidence_only": True,
            "current_execution_authority_granted": False,
            "raw_content_persisted": False,
        }


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
        self.events.append(audited.to_dict())
        self.delegate.emit(audited)


def reconstruct_current_full_graph_material(
    *,
    root: str | Path,
    manifest_path: str | Path,
    source_run_root: str | Path,
    source_train_receipt: str | Path,
    commitments: Mapping[str, Any],
) -> ReconstructedTypedGraphMaterial:
    """Rebuild TRAIN graphs without producing any authorization object."""

    manifest = SplitManifest.read(manifest_path)
    if manifest.manifest_hash != commitments.get("manifest_hash"):
        raise PermissionError("typed portable manifest binding mismatch")
    evidence = reconstruct_v315_train_evidence(
        root=root,
        manifest=manifest,
        source_run_root=source_run_root,
        source_train_receipt=source_train_receipt,
        event_sink=MemoryEventSink(),
    )
    if evidence.source_train_receipt_hash != commitments.get(
        "source_train_receipt_hash"
    ):
        raise PermissionError("typed portable TRAIN receipt binding mismatch")
    trials = tuple(
        _extract_all_train_trials(
            evidence=evidence,
            source_run_root=source_run_root,
        )
    )
    trial_evidence_hash = stable_hash(
        {"trial_evidence_hashes": [row.evidence_hash for row in trials]}
    )
    if trial_evidence_hash != commitments.get("trial_evidence_hash"):
        raise PermissionError("typed portable trial evidence drifted")

    target_hashes = tuple(commitments.get("target_family_hashes") or ())
    expected_graph_hashes = tuple(commitments.get("graph_hashes") or ())
    expected_catalog_hashes = tuple(
        commitments.get("model_catalog_hashes") or ()
    )
    if not (
        len(target_hashes)
        == len(expected_graph_hashes)
        == len(expected_catalog_hashes)
        == REQUIRED_SLOT_COUNT
    ):
        raise PermissionError("typed portable full graph commitments malformed")
    family_by_hash = {
        stable_hash({"family": family}): family
        for family in {row.family for row in evidence.failures}
    }
    trial_by_id = {row.trial_id_hash: row for row in trials}
    graphs: list[FamilyCapabilityGraph] = []
    for target_hash in target_hashes:
        family = family_by_hash.get(target_hash)
        if family is None:
            raise PermissionError(
                "typed portable target family absent from TRAIN failures"
            )
        graphs.append(
            build_family_capability_graph(
                target_family=family,
                failures=evidence.failures,
                action_profiles=evidence.action_profiles,
                trial_evidence=trial_by_id,
                minimum_support=MINIMUM_FAMILY_SUPPORT,
                maximum_artifacts=MAX_REGISTERED_ARTIFACTS_PER_FAMILY,
            )
        )
    full_graphs = tuple(graphs)
    actual_graph_hashes = tuple(row.graph_hash for row in full_graphs)
    actual_catalog_hashes = tuple(
        stable_hash(row.model_catalog()) for row in full_graphs
    )
    full_graph_set_hash = stable_hash(
        {
            "outcomes": [
                {
                    "target_family_hash": row.target_family_hash,
                    "graph_hash": row.graph_hash,
                    "availability_error_hash": None,
                }
                for row in full_graphs
            ]
        }
    )
    full_model_catalog_set_hash = stable_hash(
        {"catalog_hashes": list(actual_catalog_hashes)}
    )
    if (
        tuple(row.target_family_hash for row in full_graphs) != target_hashes
        or actual_graph_hashes != expected_graph_hashes
        or actual_catalog_hashes != expected_catalog_hashes
        or full_graph_set_hash != commitments.get("graph_set_hash")
        or full_model_catalog_set_hash
        != commitments.get("model_catalog_set_hash")
    ):
        raise PermissionError(
            "typed portable full graphs drifted from historical commitments"
        )
    return ReconstructedTypedGraphMaterial(
        evidence=evidence,
        trials=trials,
        full_graphs=full_graphs,
        trial_evidence_hash=trial_evidence_hash,
        full_graph_set_hash=full_graph_set_hash,
        full_model_catalog_set_hash=full_model_catalog_set_hash,
    )


def load_frozen_portable_typed_selection_ledger(
    *,
    root: str | Path,
    manifest_path: str | Path,
    source_run_root: str | Path,
    source_train_receipt: str | Path,
    preregistration_path: str | Path,
) -> FrozenTypedSelectionLedger:
    ledger, _, _, _ = _load_portable_typed_selection_material(
        root=root,
        manifest_path=manifest_path,
        source_run_root=source_run_root,
        source_train_receipt=source_train_receipt,
        preregistration_path=preregistration_path,
    )
    return ledger


def _load_portable_typed_selection_material(
    *,
    root: str | Path,
    manifest_path: str | Path,
    source_run_root: str | Path,
    source_train_receipt: str | Path,
    preregistration_path: str | Path,
) -> tuple[
    FrozenTypedSelectionLedger,
    ReconstructedTypedGraphMaterial,
    HistoricalFeasibilityEvidence,
    tuple[dict[str, Any], ...],
]:
    """Load a diagnostic-only projected ledger under the new formal manifest."""

    preregistration = _read_preregistration(preregistration_path)
    project_root = _project_root(preregistration_path)
    _require_declared_directory(
        root,
        project_root=project_root,
        declared=preregistration["benchmark_root"],
        label="benchmark root",
    )
    _require_declared_file(
        manifest_path,
        project_root=project_root,
        declared=preregistration["manifest"],
        label="manifest",
    )
    _require_declared_file(
        source_train_receipt,
        project_root=project_root,
        declared=preregistration["source_train_receipt"],
        label="source TRAIN receipt",
    )
    _require_declared_directory(
        source_run_root,
        project_root=project_root,
        declared=preregistration["source_run_root"],
        label="source run root",
    )
    historical = _verify_historical_feasibility(
        preregistration,
        preregistration_path=preregistration_path,
    )
    full_commitments = dict(preregistration["full_graph_commitments"])
    full_commitments.update(
        {
            "manifest_hash": preregistration["manifest_hash"],
            "source_train_receipt_hash": preregistration[
                "source_train_receipt_hash"
            ],
        }
    )
    material = reconstruct_current_full_graph_material(
        root=root,
        manifest_path=manifest_path,
        source_run_root=source_run_root,
        source_train_receipt=source_train_receipt,
        commitments=full_commitments,
    )
    projected_graphs = tuple(
        _project_portable_graph(row) for row in material.full_graphs
    )
    projection_rows = tuple(
        _projection_row(full, projected)
        for full, projected in zip(material.full_graphs, projected_graphs)
    )
    projection = preregistration["portable_projection"]
    if list(projection_rows) != projection["rows"]:
        raise PermissionError("typed portable projection rows drifted")
    snapshots = tuple(
        freeze_typed_recipe_selection_snapshot(row)
        for row in projected_graphs
    )
    graph_set_hash = stable_hash(
        {
            "outcomes": [
                {
                    "target_family_hash": row.graph.target_family_hash,
                    "graph_hash": row.expected_graph_hash,
                    "availability_error_hash": None,
                }
                for row in snapshots
            ]
        }
    )
    model_catalog_set_hash = stable_hash(
        {
            "catalog_hashes": [
                row.expected_model_catalog_hash for row in snapshots
            ]
        }
    )
    if (
        graph_set_hash != projection["graph_set_hash"]
        or model_catalog_set_hash != projection["model_catalog_set_hash"]
    ):
        raise PermissionError("typed portable projected graph set drifted")
    manifest = SplitManifest.read(manifest_path)
    ledger = freeze_typed_selection_snapshot_ledger(
        snapshots,
        feasibility_preregistration_hash=historical.preregistration_hash,
        feasibility_result_receipt_sha256=(
            historical.result_receipt_file_sha256
        ),
        feasibility_decision_hash=historical.decision_hash,
        feasibility_report_hash=historical.report_hash,
        manifest_hash=manifest.manifest_hash,
        source_train_receipt_hash=(
            material.evidence.source_train_receipt_hash
        ),
        expected_graph_set_hash=graph_set_hash,
        expected_model_catalog_set_hash=model_catalog_set_hash,
        expected_target_family_hashes=tuple(
            row.target_family_hash for row in projected_graphs
        ),
    )
    if ledger.ledger_hash != projection["ledger_hash"]:
        raise PermissionError("typed portable projected ledger drifted")
    frozen = FrozenTypedSelectionLedger(
        evidence=material.evidence,
        trials=material.trials,
        snapshots=snapshots,
        production_snapshot_ledger=ledger,
        upstream_binding_hash=historical.binding_hash,
        trial_evidence_hash=material.trial_evidence_hash,
        graph_set_hash=graph_set_hash,
        model_catalog_set_hash=model_catalog_set_hash,
    )
    return frozen, material, historical, projection_rows


def _project_portable_graph(
    full_graph: FamilyCapabilityGraph,
) -> FamilyCapabilityGraph:
    projector = getattr(
        _portable_capability,
        "project_portable_family_capability_graph",
        None,
    )
    if not callable(projector):
        raise PermissionError("portable capability graph projector is missing")
    projected = projector(full_graph)
    if isinstance(projected, FamilyCapabilityGraph):
        graph = projected
    else:
        graph = getattr(projected, "projected_graph", None)
    portable_graph_type = getattr(
        _portable_capability,
        "PortableFamilyCapabilityGraph",
        None,
    )
    if (
        not isinstance(portable_graph_type, type)
        or not isinstance(graph, portable_graph_type)
        or graph.validate()
    ):
        raise PermissionError("portable capability graph projection is invalid")
    if graph.target_family != full_graph.target_family:
        raise PermissionError("portable capability projection changed family")
    full_artifacts = {row.artifact_id for row in full_graph.artifacts}
    full_recipes = {row.recipe_id for row in full_graph.recipes}
    if not {row.artifact_id for row in graph.artifacts} <= full_artifacts or not {
        row.recipe_id for row in graph.recipes
    } <= full_recipes:
        raise PermissionError("portable capability projection is not a subset")
    if not graph.recipes:
        raise PermissionError("portable capability projection is empty")
    role_builder = getattr(
        _portable_capability,
        "portable_role_spec_for_recipe",
        None,
    )
    if not callable(role_builder):
        raise PermissionError("portable capability role registry is missing")
    for recipe in graph.recipes:
        role_builder(graph, recipe.recipe_id)
    _verify_portable_evidence_catalog_semantics(graph)
    return graph


def _verify_portable_evidence_catalog_semantics(
    graph: FamilyCapabilityGraph,
) -> None:
    catalog = graph.model_catalog()
    selector_context = catalog.get("portable_selector_context")
    if selector_context != {
        "target_family": graph.target_family,
        "target_family_scope": "frozen_train_routing_label",
        "target_family_used_as_task_content": False,
        "selector_output_fields": ["recipe_id"],
        "opaque_recipe_id_only_output": True,
    }:
        raise PermissionError(
            "portable selector context semantics drifted"
        )
    projection = catalog.get("portable_capability_projection")
    if not isinstance(projection, Mapping) or (
        projection.get("all_selectable_recipes_artifact_evidence_backed")
        is not True
        or projection.get(
            "pre_agent_evidence_argument_surface_restricted"
        )
        is not True
        or projection.get(
            "capability_execution_covers_full_recipe_operator_plan"
        )
        is not False
        or projection.get("non_access_operators_are_fixed_agent_plan")
        is not True
        or projection.get("behavioral_aliases_deduplicated") is not True
        or projection.get("diversity_counts_behavioral_signature_only")
        is not True
        or projection.get("all_selectable_recipes_implementation_backed")
        is True
    ):
        raise PermissionError(
            "portable projection overclaims recipe operator implementation"
        )
    capabilities = catalog.get("capabilities")
    if not isinstance(capabilities, list) or not capabilities:
        raise PermissionError(
            "portable projection capability catalog is malformed"
        )
    for capability in capabilities:
        if not isinstance(capability, Mapping) or (
            capability.get("capability_implementation_verified") is not False
            or capability.get("runtime_agent_argument_surface_restricted")
            is not False
            or capability.get(
                "pre_agent_artifact_evidence_capability_verified"
            )
            is not True
            or capability.get(
                "pre_agent_evidence_argument_surface_restricted"
            )
            is not True
            or capability.get("runtime_agent_capability_invocation_available")
            is not False
            or capability.get("full_recipe_operator_execution_by_capability")
            is not False
        ):
            raise PermissionError(
                "portable evidence capability catalog semantics drifted"
            )


def _projection_row(
    full_graph: FamilyCapabilityGraph,
    projected_graph: FamilyCapabilityGraph,
) -> dict[str, Any]:
    artifact_ids = sorted(row.artifact_id for row in projected_graph.artifacts)
    recipe_ids = sorted(row.recipe_id for row in projected_graph.recipes)
    catalog = projected_graph.model_catalog()
    projection = catalog["portable_capability_projection"]
    selector_context = catalog["portable_selector_context"]
    workflow_values = sorted(
        {row.workflow.value for row in projected_graph.recipes}
    )
    treatment_signatures = list(
        projection["behavioral_treatment_signature_hashes"]
    )
    return {
        "target_family_hash": full_graph.target_family_hash,
        "parent_graph_hash": full_graph.graph_hash,
        "projected_graph_hash": projected_graph.graph_hash,
        "projected_model_catalog_hash": stable_hash(
            projected_graph.model_catalog()
        ),
        "projected_artifact_count": len(artifact_ids),
        "projected_recipe_count": len(recipe_ids),
        "projected_artifact_set_hash": stable_hash(
            {"artifact_ids": artifact_ids}
        ),
        "projected_recipe_set_hash": stable_hash(
            {"recipe_ids": recipe_ids}
        ),
        "projected_workflow_set_hash": stable_hash(
            {"workflows": workflow_values}
        ),
        "portable_selector_context_hash": stable_hash(selector_context),
        "selectable_recipe_set_hash": projection[
            "selectable_recipe_set_hash"
        ],
        "source_executable_artifact_count": projection[
            "source_executable_artifact_count"
        ],
        "retained_behavioral_artifact_count": projection[
            "retained_behavioral_artifact_count"
        ],
        "behavioral_alias_artifact_count": projection[
            "behavioral_alias_artifact_count"
        ],
        "behavioral_alias_set_hash": projection[
            "behavioral_alias_set_hash"
        ],
        "behavioral_treatment_signature_hashes": treatment_signatures,
        "behavioral_treatment_signature_set_hash": stable_hash(
            {"hashes": treatment_signatures}
        ),
        "behavioral_aliases_deduplicated": True,
        "diversity_counts_behavioral_signature_only": True,
        "every_selectable_fixed_plan_has_portable_artifact_evidence_role": True,
        "capability_execution_covers_full_recipe_operator_plan": False,
        "non_access_operators_are_fixed_agent_plan": True,
        "pre_agent_sidecar_is_task_effect": False,
        "raw_artifact_locators_persisted": False,
    }


def _verify_historical_feasibility(
    preregistration: Mapping[str, Any],
    *,
    preregistration_path: str | Path,
) -> HistoricalFeasibilityEvidence:
    """Verify immutable old artifacts without treating their code hash as current."""

    project_root = _project_root(preregistration_path)
    declared = preregistration["historical_feasibility"]
    paths = {
        key: _resolve_bound_file(project_root, declared[key]["path"])
        for key in (
            "preregistration",
            "result_receipt",
            "report",
            "events",
            "decision_lock",
        )
    }
    actual_sha = {key: _sha256_file(path) for key, path in paths.items()}
    for key, value in actual_sha.items():
        if value != declared[key]["sha256"]:
            raise PermissionError(
                f"historical feasibility {key} file hash drifted"
            )
    upstream_prereg = _read_json_object(paths["preregistration"])
    result_receipt = _read_json_object(paths["result_receipt"])
    report = _read_json_object(paths["report"])
    decision_lock = _read_json_object(paths["decision_lock"])
    events = _read_event_rows(paths["events"])
    if (
        stable_hash(upstream_prereg)
        != declared["preregistration"]["stable_hash"]
        or stable_hash(result_receipt)
        != declared["result_receipt"]["stable_hash"]
    ):
        raise PermissionError("historical feasibility stable hash drifted")
    report_without_hash = dict(report)
    report_hash = report_without_hash.pop("report_hash", None)
    if stable_hash(report_without_hash) != report_hash:
        raise PermissionError("historical feasibility report self hash drifted")
    decision_hash = report.get("decision_hash")
    if (
        report_hash != declared["report_hash"]
        or decision_hash != declared["decision_hash"]
        or report.get("feasibility_passed") is not True
        or result_receipt.get("feasibility_passed") is not True
        or result_receipt.get("report_hash") != report_hash
        or result_receipt.get("decision_hash") != decision_hash
        or result_receipt.get("exact_replay_verified") is not True
        or decision_lock.get("state") != "completed"
        or decision_lock.get("report_hash") != report_hash
        or decision_lock.get("decision_hash") != decision_hash
        or decision_lock.get("preregistration_hash")
        != stable_hash(upstream_prereg)
        or not events
        or events[-1].get("event")
        != "typed_operator_feasibility_completed"
        or events[-1].get("payload", {}).get("report_hash") != report_hash
        or events[-1].get("payload", {}).get("decision_hash")
        != decision_hash
    ):
        raise PermissionError("historical feasibility decision drifted")
    canonical = result_receipt.get("canonical_artifacts")
    if not isinstance(canonical, Mapping):
        raise PermissionError("historical feasibility receipt malformed")
    for key in ("report", "events", "decision_lock"):
        if (
            canonical.get(key) != declared[key]["path"]
            or canonical.get(f"{key}_sha256") != actual_sha[key]
        ):
            raise PermissionError(
                "historical feasibility receipt artifact drifted"
            )
    binding_hash = stable_hash(
        {
            "historical_preregistration_hash": stable_hash(upstream_prereg),
            "historical_result_receipt_hash": stable_hash(result_receipt),
            "artifact_hashes": actual_sha,
            "decision_hash": decision_hash,
            "report_hash": report_hash,
            "historical_evidence_only": True,
        }
    )
    return HistoricalFeasibilityEvidence(
        preregistration_hash=stable_hash(upstream_prereg),
        result_receipt_hash=stable_hash(result_receipt),
        result_receipt_file_sha256=actual_sha["result_receipt"],
        decision_hash=str(decision_hash),
        report_hash=str(report_hash),
        binding_hash=binding_hash,
        report=report,
    )


def _exercise_opaque_production_selection(
    *,
    ledger: FrozenTypedSelectionLedger,
    full_graphs: Sequence[FamilyCapabilityGraph],
    event_sink: EventSink,
    trace_id: str,
) -> tuple[
    tuple[HypothesisProgram, ...],
    StructuredHypothesisProposer,
    dict[str, Any],
    dict[str, bool],
]:
    """Exercise production proposal/evolution with a deterministic local model."""

    selector = _OfflineRecipeSelector(
        ledger.production_snapshot_ledger.ledger_hash
    )
    proposer = StructuredHypothesisProposer(
        selector,
        event_sink=event_sink,
        typed_program_registry=TypedProgramBindingRegistry(
            snapshot_ledger=ledger.production_snapshot_ledger
        ),
    )
    context = replace(
        _validation_context(ledger),
        evaluator_epoch=TYPED_PORTABLE_EVALUATOR_EPOCH,
    )
    kernel = _kernel(
        proposer=proposer,
        ledger=ledger,
        event_sink=event_sink,
    )
    kernel.counterfactual_runner.evaluator.epoch = (
        TYPED_PORTABLE_EVALUATOR_EPOCH
    )
    programs = kernel.propose_candidates(
        ledger.evidence.residuals,
        validation_context=context,
        trace_id=f"{trace_id}:generation-1",
    )
    if len(programs) != REQUIRED_SLOT_COUNT or selector.call_count != (
        REQUIRED_SLOT_COUNT
    ):
        raise PermissionError(
            "typed portable production selector coverage is incomplete"
        )
    bindings = tuple(
        proposer.typed_program_registry.require(program)
        for program in programs
    )
    for snapshot, binding in zip(ledger.snapshots, bindings):
        if binding.recipe_id not in {
            recipe.recipe_id for recipe in snapshot.graph.recipes
        }:
            raise PermissionError(
                "typed portable selector escaped the projected catalog"
            )
        _portable_capability.portable_role_spec_for_recipe(
            snapshot.graph,
            binding.recipe_id,
        )

    replay_before = selector.call_count
    replay = kernel.propose_candidates(
        ledger.evidence.residuals,
        validation_context=context,
        trace_id=f"{trace_id}:generation-1-replay",
    )
    if replay != programs or selector.call_count != replay_before:
        raise PermissionError(
            "typed portable production selection did not replay exactly"
        )
    shared = kernel.validate_typed_shared_proposal_candidates(
        programs,
        trace_id=f"{trace_id}:shared-bundle",
    )
    if tuple(row["binding_hash"] for row in shared) != tuple(
        row.binding_hash for row in bindings
    ):
        raise PermissionError(
            "typed portable shared proposal receipts drifted"
        )

    kernel.record_typed_selection_attempts(
        programs,
        trace_id=f"{trace_id}:record-generation-1",
    )
    generation_two_before = selector.call_count
    generation_two = kernel.propose_candidates(
        ledger.evidence.residuals,
        validation_context=context,
        trace_id=f"{trace_id}:generation-2",
    )
    if selector.call_count - generation_two_before != REQUIRED_SLOT_COUNT:
        raise PermissionError(
            "typed portable multi-generation selection did not advance"
        )
    generation_two_bindings = tuple(
        proposer.typed_program_registry.require(program)
        for program in generation_two
    )
    if any(
        current.recipe_id == previous.recipe_id
        for previous, current in zip(bindings, generation_two_bindings)
    ):
        raise PermissionError(
            "typed portable multi-generation exclusions were not applied"
        )
    if any(
        next(
            recipe.workflow
            for recipe in snapshot.graph.recipes
            if recipe.recipe_id == previous.recipe_id
        )
        == next(
            recipe.workflow
            for recipe in snapshot.graph.recipes
            if recipe.recipe_id == current.recipe_id
        )
        for snapshot, previous, current in zip(
            ledger.snapshots,
            bindings,
            generation_two_bindings,
        )
    ):
        raise PermissionError(
            "typed portable diversity did not change fixed plan treatment"
        )

    tamper = _selection_tamper_probes(
        ledger=ledger,
        full_graphs=full_graphs,
        proposer=proposer,
    )
    request_hashes = tuple(stable_hash(row) for row in selector.requests)
    response_hashes = tuple(stable_hash(row) for row in selector.responses)
    receipt = {
        "selector_kind": "deterministic_local_recipe_id_only",
        "production_proposer_class": type(proposer).__name__,
        "production_kernel_class": type(kernel).__name__,
        "root_program_count": len(programs),
        "root_program_set_hash": stable_hash(
            {"program_hashes": [row.payload_hash for row in programs]}
        ),
        "root_binding_set_hash": stable_hash(
            {"binding_hashes": [row.binding_hash for row in bindings]}
        ),
        "generation_two_program_set_hash": stable_hash(
            {
                "program_hashes": [
                    row.payload_hash for row in generation_two
                ]
            }
        ),
        "projected_behavioral_alias_artifact_count": sum(
            int(snapshot.graph.behavioral_alias_count)
            for snapshot in ledger.snapshots
        ),
        "projected_behavioral_alias_set_hash": stable_hash(
            {
                "hashes": [
                    snapshot.graph.behavioral_alias_set_hash
                    for snapshot in ledger.snapshots
                ]
            }
        ),
        "behavioral_alias_recipe_ids_expressible": False,
        "diversity_counts_deduplicated_recipe_treatment_only": True,
        "generation_two_changed_fixed_plan_treatment_per_slot": True,
        "request_count": len(request_hashes),
        "request_set_hash": stable_hash(
            {"request_hashes": list(request_hashes)}
        ),
        "response_set_hash": stable_hash(
            {"response_hashes": list(response_hashes)}
        ),
        "exact_replay_new_selector_calls": 0,
        "shared_bundle_new_selector_calls": 0,
        "model_authored_primitive_count": 0,
        "model_authored_locator_count": 0,
        "live_model_invoked": False,
        "raw_content_persisted": False,
    }
    return programs, proposer, receipt, tamper


def _selection_tamper_probes(
    *,
    ledger: FrozenTypedSelectionLedger,
    full_graphs: Sequence[FamilyCapabilityGraph],
    proposer: StructuredHypothesisProposer,
) -> dict[str, bool]:
    snapshot = ledger.snapshots[0]
    ledger_hash = ledger.production_snapshot_ledger.ledger_hash

    def rejects_response(response: Mapping[str, Any]) -> bool:
        probe = StructuredHypothesisProposer(
            _FixedResponseSelector(
                response,
                expected_ledger_hash=ledger_hash,
            ),
            typed_program_registry=TypedProgramBindingRegistry(
                snapshot_ledger=ledger.production_snapshot_ledger
            ),
        )
        try:
            probe.select_typed_recipe(
                snapshot=snapshot,
                evaluator_epoch=TYPED_PORTABLE_EVALUATOR_EPOCH,
                trace_id="typed-portable-tamper",
            )
        except (HypothesisProposalCallError, PermissionError, ValueError):
            return True
        return False

    projected_ids = {
        recipe.recipe_id for recipe in snapshot.graph.recipes
    }
    full_only_ids = sorted(
        recipe.recipe_id
        for graph in full_graphs
        if graph.target_family_hash == snapshot.graph.target_family_hash
        for recipe in graph.recipes
        if recipe.recipe_id not in projected_ids
    )
    if not full_only_ids:
        raise PermissionError(
            "typed portable full-graph-only tamper canary is unavailable"
        )
    tampered_ledger = replace(
        ledger.production_snapshot_ledger,
        feasibility_report_hash=stable_hash(
            {"tampered": "portable-projected-ledger"}
        ),
    )
    rebinding_rejected = False
    try:
        proposer.typed_program_registry.bind_snapshot_ledger(
            tampered_ledger
        )
    except (PermissionError, ValueError):
        rebinding_rejected = True
    return {
        "selector_extra_field": rejects_response(
            {
                "recipe_id": snapshot.graph.recipes[0].recipe_id,
                "forbidden_free_text": "not executable",
            }
        ),
        "selector_full_graph_only_recipe": rejects_response(
            {"recipe_id": full_only_ids[0]}
        ),
        "projected_ledger_rebinding": rebinding_rejected,
    }


def _compile_full_bundle_and_run_canaries(
    *,
    root: str | Path,
    manifest_path: str | Path,
    programs: Sequence[HypothesisProgram],
    proposer: StructuredHypothesisProposer,
    preregistration: Mapping[str, Any],
    event_sink: EventSink,
    trace_id: str,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Compile all three programs, then run one pre-agent canary per family."""

    manifest = SplitManifest.read(manifest_path)
    benchmark_root = Path(root).expanduser().resolve(strict=True)
    canaries = _validated_runtime_canaries(
        preregistration,
        manifest=manifest,
    )
    target_ids = tuple(row["item_id"] for row in canaries)
    compiler_events = MemoryEventSink()
    compiler = SkillLearnProgramCompiler(
        event_sink=compiler_events,
        typed_program_registry=proposer.typed_program_registry,
        require_typed_bindings=True,
        portable_capability_compiler_mode=(
            _portable_capability.PORTABLE_TASK_CAPABILITY_COMPILER_VERSION
        ),
    )
    metadata_mutation_rejected = False
    with tempfile.TemporaryDirectory(
        prefix="typed-portable-integration-compile-"
    ) as temporary_root:
        compile_result = compiler.compile(
            programs=programs,
            items=_public_canary_items(benchmark_root, canaries),
            split_manifest=manifest,
            output_root=temporary_root,
            method_name="typed-portable-integration",
            allowed_statuses={HypothesisStatus.CANDIDATE},
            target_item_ids=target_ids,
            target_split="validation",
            trace_id=f"{trace_id}:compile",
        )
        receipts = tuple(
            (row, compile_result.source_receipt_for(row["item_id"]))
            for row in canaries
        )
        if any(
            receipt.source_route is None
            or not receipt.portable_capability_role_spec_hashes
            or not receipt.portable_capability_metadata_file_hashes
            for _, receipt in receipts
        ):
            raise PermissionError(
                "typed portable compiler did not route every canary"
            )
        if len(compile_result.typed_binding_hashes) != REQUIRED_SLOT_COUNT:
            raise PermissionError(
                "typed portable compiler did not bind all three typed plans"
            )

        mutation_row, mutation_receipt = receipts[0]
        metadata_relative = (
            mutation_receipt.portable_capability_metadata_file_hashes[0][0]
        )
        metadata_path = compile_result.output_root / metadata_relative
        original = metadata_path.read_bytes()
        try:
            metadata_path.write_bytes(original + b"\n")
            try:
                compile_result.source_receipt_for(mutation_row["item_id"])
            except PermissionError:
                metadata_mutation_rejected = True
        finally:
            metadata_path.write_bytes(original)
        if not metadata_mutation_rejected or (
            compile_result.source_receipt_for(mutation_row["item_id"])
            != mutation_receipt
        ):
            raise PermissionError(
                "typed portable metadata mutation did not fail closed"
            )

        runtime_rows = _run_pre_agent_docker_canaries(
            benchmark_root=benchmark_root,
            compile_result=compile_result,
            programs=programs,
            canaries=canaries,
            manifest_hash=manifest.manifest_hash,
            trace_id=trace_id,
        )
        if len(runtime_rows) != REQUIRED_SLOT_COUNT or any(
            row.get("passed") is not True
            or row.get("container_cleanup_verified") is not True
            for row in runtime_rows
        ):
            raise PermissionError(
                "typed portable pre-agent runtime coverage is incomplete"
            )
        compile_manifest = _read_json_object(
            compile_result.output_root / "compile_manifest.json"
        )
        receipt_rows = tuple(
            {
                "item_id_hash": receipt.item_id_hash,
                "receipt_hash": receipt.receipt_hash,
                "source_tree_hash": receipt.source_tree_hash,
                "typed_binding_set_hash": receipt.typed_binding_set_hash,
                "typed_snapshot_ledger_hash": (
                    receipt.typed_snapshot_ledger_hash
                ),
                "portable_role_spec_hashes": list(
                    receipt.portable_capability_role_spec_hashes
                ),
                "portable_metadata_tree_hash": stable_hash(
                    {
                        "files": [
                            {"path": path, "sha256": sha256}
                            for path, sha256 in (
                                receipt.portable_capability_metadata_file_hashes
                            )
                        ]
                    }
                ),
                "raw_content_persisted": False,
            }
            for _, receipt in receipts
        )

    canonical_compiler_events = _canonical_compiler_event_rows(
        compiler_events.events
    )
    compiler_receipt = {
        "compiler_class": type(compiler).__name__,
        "compiler_mode": (
            _portable_capability.PORTABLE_TASK_CAPABILITY_COMPILER_VERSION
        ),
        "program_count": len(programs),
        "target_split": "validation",
        "target_item_count": len(canaries),
        "compile_manifest_hash": stable_hash(compile_manifest),
        "typed_binding_hashes": list(compile_result.typed_binding_hashes),
        "typed_binding_set_hash": compile_result.typed_binding_set_hash,
        "typed_snapshot_ledger_hash": (
            compile_result.typed_snapshot_ledger_hash
        ),
        "portable_role_spec_set_hash": (
            compile_result.portable_capability_role_spec_set_hash
        ),
        "source_receipt_set_hash": stable_hash(
            {"receipts": list(receipt_rows)}
        ),
        "source_receipt_count": len(receipt_rows),
        "compiler_event_set_hash": stable_hash(
            {"events": list(canonical_compiler_events)}
        ),
        "compiler_event_count": len(canonical_compiler_events),
        "all_three_typed_plan_bindings_compiled": True,
        "portable_metadata_scope": (
            "pre_agent_read_only_artifact_evidence_sidecar"
        ),
        "recipe_operator_execution_compiled": False,
        "task_effect_compiled": False,
        "validation_outcome_accessed": False,
        "raw_content_persisted": False,
    }
    runtime_receipt = {
        "runtime_policy": PRE_AGENT_TASK_CAPABILITY_RUNTIME_VERSION,
        "canary_count": len(runtime_rows),
        "canary_set_hash": stable_hash({"canaries": list(runtime_rows)}),
        "canaries": list(runtime_rows),
        "container_network": "none",
        "container_cleanup_verified": True,
        "agent_started": False,
        "model_invoked": False,
        "evaluator_invoked": False,
        "verifier_invoked": False,
        "raw_content_persisted": False,
    }
    return compiler_receipt, runtime_receipt, metadata_mutation_rejected


def _validated_runtime_canaries(
    preregistration: Mapping[str, Any],
    *,
    manifest: SplitManifest,
) -> tuple[dict[str, Any], ...]:
    section = preregistration.get("runtime_canaries")
    if not isinstance(section, Mapping) or set(section) != {
        "rows",
        "set_hash",
    }:
        raise PermissionError("typed portable runtime canaries are malformed")
    raw_rows = section.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != REQUIRED_SLOT_COUNT:
        raise PermissionError("typed portable runtime canary count drifted")
    rows: list[dict[str, Any]] = []
    required_keys = {
        "family",
        "family_hash",
        "item_id",
        "item_id_hash",
        "prebuilt_image_id",
        "prebuilt_image_key",
        "task_input_closure_hash",
        "task_input_integrity_receipt_hash",
        "task_input_integrity_container_network",
        "public_instruction_sha256",
        "dockerfile_sha256",
    }
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or set(raw) != required_keys:
            raise PermissionError("typed portable runtime canary row drifted")
        row = dict(raw)
        if (
            row["family_hash"] != stable_hash({"family": row["family"]})
            or row["item_id_hash"]
            != stable_hash({"item_id": row["item_id"]})
            or row["item_id"] not in manifest.validation_ids
            or manifest.family_by_id.get(row["item_id"]) != row["family"]
            or not str(row["prebuilt_image_id"]).startswith("sha256:")
            or len(str(row["prebuilt_image_id"])) != 71
            or not _is_sha256_text(row["prebuilt_image_key"])
            or not _is_sha256_text(row["public_instruction_sha256"])
            or not _is_sha256_text(row["dockerfile_sha256"])
        ):
            raise PermissionError(
                "typed portable runtime canary identity is invalid"
            )
        for optional_hash in (
            "task_input_closure_hash",
            "task_input_integrity_receipt_hash",
        ):
            if row[optional_hash] is not None and not _is_sha256_text(
                row[optional_hash]
            ):
                raise PermissionError(
                    "typed portable runtime canary receipt is invalid"
                )
        if row["task_input_integrity_container_network"] not in {
            None,
            "none",
        }:
            raise PermissionError(
                "typed portable runtime canary network receipt is invalid"
            )
        rows.append(row)
    if (
        len({row["family_hash"] for row in rows}) != REQUIRED_SLOT_COUNT
        or len({row["item_id_hash"] for row in rows}) != REQUIRED_SLOT_COUNT
        or stable_hash({"rows": rows}) != section["set_hash"]
    ):
        raise PermissionError("typed portable runtime canary set drifted")
    return tuple(rows)


def _public_canary_items(
    benchmark_root: Path,
    canaries: Sequence[Mapping[str, Any]],
) -> tuple[BenchmarkItem, ...]:
    """Build the compiler inventory from only the three public canary inputs."""

    tasks_root = (benchmark_root / "tasks").resolve(strict=True)
    items: list[BenchmarkItem] = []
    for canary in canaries:
        family = str(canary["family"])
        item_id = str(canary["item_id"])
        relative = Path(family) / item_id
        if relative.is_absolute() or ".." in relative.parts:
            raise PermissionError("typed portable public canary path is unsafe")
        current = tasks_root
        for component in relative.parts:
            current = current / component
            if current.is_symlink():
                raise PermissionError(
                    "typed portable public canary path contains a symlink"
                )
        item_root = current.resolve(strict=True)
        try:
            item_root.relative_to(tasks_root)
        except ValueError as exc:
            raise PermissionError(
                "typed portable public canary escaped benchmark root"
            ) from exc
        instruction = item_root / "instruction.md"
        dockerfile = item_root / "environment" / "Dockerfile"
        if (
            instruction.is_symlink()
            or not instruction.is_file()
            or dockerfile.is_symlink()
            or not dockerfile.is_file()
        ):
            raise PermissionError(
                "typed portable public canary input is incomplete"
            )
        if (
            _sha256_file(instruction) != canary["public_instruction_sha256"]
            or _sha256_file(dockerfile) != canary["dockerfile_sha256"]
        ):
            raise PermissionError(
                "typed portable public canary input hash drifted"
            )
        items.append(
            BenchmarkItem(
                id=item_id,
                family=family,
                features={
                    "benchmark": "skilllearnbench",
                    "family": family,
                    "public_input_canary": True,
                },
                content_ref=instruction.relative_to(benchmark_root).as_posix(),
                verifier_ref_hash=stable_hash(
                    {
                        "item_id_hash": canary["item_id_hash"],
                        "verifier_content_accessed": False,
                    }
                ),
            )
        )
    return tuple(items)


def _run_pre_agent_docker_canaries(
    *,
    benchmark_root: Path,
    compile_result: Any,
    programs: Sequence[HypothesisProgram],
    canaries: Sequence[Mapping[str, Any]],
    manifest_hash: str,
    trace_id: str,
) -> tuple[dict[str, Any], ...]:
    """Run only the production pre-agent injection hook in exact frozen images.

    This is the deliberate runner-injection seam used by unit tests.  Formal
    callers do not receive a skip flag: without monkeypatching this private
    function, the real Docker path is mandatory.
    """

    backend = SkillLearnSubprocessBackend(
        benchmark_root,
        model="gpt-5.4-mini",
        provider_mode="openai_compatible",
        event_sink=MemoryEventSink(),
    )
    runner = backend._load_runner()
    program_by_hash = {
        stable_hash({"program_id": program.id}): program
        for program in programs
    }
    rows: list[dict[str, Any]] = []
    for index, canary in enumerate(canaries, start=1):
        item_id = str(canary["item_id"])
        family = str(canary["family"])
        source = compile_result.source_for(item_id)
        if source is None:
            raise PermissionError(
                "typed portable runtime canary has no compiled source"
            )
        receipt = compile_result.source_receipt_for(item_id)
        metadata_paths = receipt.portable_capability_metadata_file_hashes
        if len(metadata_paths) != 1:
            raise PermissionError(
                "typed portable runtime canary role is not unique"
            )
        metadata_payload = _read_json_object(
            compile_result.output_root / metadata_paths[0][0]
        )
        program = program_by_hash.get(
            str(metadata_payload.get("program_id_hash") or "")
        )
        if program is None:
            raise PermissionError(
                "typed portable runtime metadata program is unbound"
            )
        request = SkillLearnTrialRequest(
            item_id=item_id,
            family=family,
            split=SplitName.VALIDATION,
            variant=TrialVariant.POLICY_ON,
            evaluator_epoch=TYPED_PORTABLE_EVALUATOR_EPOCH,
            pair_id=f"typed-portable-canary-{index}",
            repeat=1,
            agent_id="codex",
            model="gpt-5.4-mini",
            max_steps=100,
            manifest_hash=manifest_hash,
            program_id=program.id,
            program_set_hash=compile_result.program_set_hash,
            treatment_hash=compile_result.treatment_hash_for(item_id),
            compile_manifest_hash=compile_result.manifest_hash,
            skill_source_receipt_hash=receipt.receipt_hash,
            compile_root=compile_result.output_root,
            typed_binding_set_hash=compile_result.typed_binding_set_hash,
            typed_snapshot_hashes=compile_result.typed_snapshot_hashes,
            typed_snapshot_ledger_hash=(
                compile_result.typed_snapshot_ledger_hash
            ),
            portable_capability_compiler_mode=(
                compile_result.portable_capability_compiler_mode
            ),
            portable_capability_role_spec_set_hash=(
                compile_result.portable_capability_role_spec_set_hash
            ),
            portable_capability_role_spec_hashes=(
                receipt.portable_capability_role_spec_hashes
            ),
        )
        context = backend._load_portable_task_capability_context(
            request=request,
            source_receipt=receipt,
            compile_root=compile_result.output_root,
        )
        if not isinstance(context, PortableTaskCapabilityRuntimeContext):
            raise PermissionError(
                "typed portable runtime context was not constructed"
            )

        expected_image_id = str(canary["prebuilt_image_id"])
        inspected_image_id = _docker_stdout(
            runner.subprocess,
            [
                "docker",
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                expected_image_id,
            ],
        ).strip()
        if inspected_image_id != expected_image_id:
            raise PermissionError(
                "typed portable runtime image identity drifted"
            )
        image_labels = json.loads(
            _docker_stdout(
                runner.subprocess,
                [
                    "docker",
                    "image",
                    "inspect",
                    "--format",
                    "{{json .Config.Labels}}",
                    expected_image_id,
                ],
            ).strip()
        )
        if not isinstance(image_labels, Mapping) or image_labels.get(
            "org.assumption-agent.prebuild.key"
        ) != canary["prebuilt_image_key"]:
            raise PermissionError(
                "typed portable runtime prebuilt key drifted"
            )
        closure_hash = canary["task_input_closure_hash"]
        if closure_hash is not None and image_labels.get(
            "org.assumption-agent.prebuild.task-input-closure"
        ) != closure_hash:
            raise PermissionError(
                "typed portable runtime task-input closure drifted"
            )
        container_name = (
            f"aa-typed-portable-{trace_id[:12]}-{index}"
        )
        if _docker_container_name_present(
            runner.subprocess,
            container_name,
        ):
            raise PermissionError(
                "typed portable runtime canary container name is already "
                "in use"
            )
        created = False
        runtime_row: dict[str, Any] | None = None
        try:
            _docker_stdout(
                runner.subprocess,
                [
                    "docker",
                    "create",
                    "--name",
                    container_name,
                    "--network",
                    "none",
                    "--entrypoint",
                    "/bin/sh",
                    expected_image_id,
                    "-c",
                    "while :; do sleep 60; done",
                ],
            )
            created = True
            _docker_stdout(
                runner.subprocess,
                ["docker", "start", container_name],
            )
            dockerfile = (
                benchmark_root
                / "tasks"
                / family
                / item_id
                / "environment"
                / "Dockerfile"
            )
            copies = runner._parse_skill_copies(dockerfile)
            if not copies:
                raise PermissionError(
                    "typed portable runtime has no declared injection route"
                )
            runner._assumption_v2_task_capability_context = context
            runner._inject_skills_runtime(
                container_name,
                source,
                copies,
            )
            installed = getattr(
                runner,
                "_assumption_v2_installed_skill_receipt",
                None,
            )
            effects = getattr(
                runner,
                "_assumption_v2_task_capability_effects",
                None,
            )
            if (
                not isinstance(installed, Mapping)
                or installed.get("agent_started") is not False
                or not isinstance(effects, tuple)
                or len(effects) != 1
            ):
                raise PermissionError(
                    "typed portable pre-agent effect receipt is missing"
                )
            network_mode = json.loads(
                _docker_stdout(
                    runner.subprocess,
                    [
                        "docker",
                        "inspect",
                        "--format",
                        "{{json .HostConfig.NetworkMode}}",
                        container_name,
                    ],
                ).strip()
            )
            container_image_id = _docker_stdout(
                runner.subprocess,
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{.Image}}",
                    container_name,
                ],
            ).strip()
            if network_mode != "none" or container_image_id != expected_image_id:
                raise PermissionError(
                    "typed portable runtime isolation drifted"
                )
            runtime_row = {
                "family_hash": canary["family_hash"],
                "item_id_hash": canary["item_id_hash"],
                "prebuilt_image_id": expected_image_id,
                "prebuilt_image_key": canary["prebuilt_image_key"],
                "source_receipt_hash": receipt.receipt_hash,
                "runtime_context_hash": context.context_hash,
                "installed_tree_hash": installed.get(
                    "installed_tree_hash"
                ),
                "effect_count": len(effects),
                "artifact_evidence_sidecar_effect_set_hash": stable_hash(
                    {"effect_hashes": [row.effect_hash for row in effects]}
                ),
                "artifact_evidence_sidecar_verified": True,
                "recipe_operator_effect_claimed": False,
                "task_effect_claimed": False,
                "container_network": "none",
                "exact_image_verified": True,
                "agent_started": False,
                "model_invoked": False,
                "task_backend_run_task_invoked": False,
                "validation_outcome_accessed": False,
                "passed": True,
                "raw_content_persisted": False,
            }
        finally:
            runner._assumption_v2_task_capability_context = None
            runner._assumption_v2_task_capability_effects = None
            runner._assumption_v2_task_capability_agent_payloads = None
            runner._assumption_v2_installed_skill_receipt = None
            cleanup_command_failed = False
            if created:
                try:
                    _docker_stdout(
                        runner.subprocess,
                        ["docker", "rm", "-f", container_name],
                    )
                except RuntimeError:
                    cleanup_command_failed = True
            container_still_present = _docker_container_name_present(
                runner.subprocess,
                container_name,
            )
            if cleanup_command_failed or container_still_present:
                raise RuntimeError(
                    "typed_portable_docker_cleanup_not_verified"
                )
        if runtime_row is None:
            raise RuntimeError(
                "typed_portable_runtime_receipt_missing_after_cleanup"
            )
        rows.append(
            {
                **runtime_row,
                "container_cleanup_verified": True,
            }
        )
    return tuple(rows)


def _docker_stdout(delegate: Any, command: Sequence[str]) -> str:
    result = delegate.run(
        list(command),
        capture_output=True,
        text=True,
    )
    if getattr(result, "returncode", 1) != 0:
        raise RuntimeError("typed_portable_docker_command_failed")
    stdout = getattr(result, "stdout", "")
    if not isinstance(stdout, str):
        raise RuntimeError("typed_portable_docker_output_invalid")
    return stdout


def _docker_container_name_present(delegate: Any, name: str) -> bool:
    output = _docker_stdout(
        delegate,
        [
            "docker",
            "container",
            "ls",
            "--all",
            "--filter",
            f"name=^/{name}$",
            "--format",
            "{{.Names}}",
        ],
    )
    return bool([line for line in output.splitlines() if line.strip()])


def _verify_production_authorization_loader_route(
    *,
    root: str | Path,
    manifest_path: str | Path,
    source_run_root: str | Path,
    source_train_receipt: str | Path,
    preregistration: Mapping[str, Any],
    preregistration_path: str | Path,
    expected_ledger: FrozenTypedSelectionLedger,
    material: ReconstructedTypedGraphMaterial,
    historical: HistoricalFeasibilityEvidence,
) -> dict[str, Any]:
    """Traverse the production v3.20 loader without an authority receipt."""

    # Local imports avoid the production experiment module's import of this
    # diagnostic becoming a module-import cycle.
    from .paper_protocol import (
        PORTABLE_TASK_CAPABILITY_PROTOCOL_VERSION,
        PaperProtocol,
    )
    from .skilllearn_experiment import (
        _load_typed_selection_for_execution,
    )

    project_root = _project_root(preregistration_path)
    preregistration_file = Path(preregistration_path).expanduser().resolve(
        strict=True
    )
    source_receipt_file = _resolve_bound_file(
        project_root,
        preregistration["source_train_receipt"],
    )
    source = {
        "preregistration": preregistration_file.relative_to(
            project_root
        ).as_posix(),
        "preregistration_file_sha256": _sha256_file(
            preregistration_file
        ),
        "source_run_root": preregistration["source_run_root"],
        "source_train_receipt": preregistration[
            "source_train_receipt"
        ],
        "source_train_receipt_file_sha256": _sha256_file(
            source_receipt_file
        ),
        "snapshot_ledger_hash": (
            expected_ledger.production_snapshot_ledger.ledger_hash
        ),
    }
    if set(source) != _DIAGNOSTIC_TYPED_SOURCE_FIELDS:
        raise PermissionError(
            "typed portable diagnostic loader source fields drifted"
        )
    protocol_payload = {
        "protocol_id": "typed-portable-integration-loader-probe",
        "protocol_version": PORTABLE_TASK_CAPABILITY_PROTOCOL_VERSION,
    }
    protocol = PaperProtocol(
        path=(
            project_root
            / "manifests"
            / ".typed-portable-integration-loader-probe.json"
        ),
        payload=protocol_payload,
    )
    execution_contract = {"typed_selection_snapshot_source": source}
    loaded = _load_typed_selection_for_execution(
        root=root,
        manifest_path=manifest_path,
        protocol=protocol,
        execution_contract=execution_contract,
        proposal_formation_policy=(
            TYPED_RECIPE_PROPOSAL_FORMATION_POLICY_VERSION
        ),
        integration_diagnostic_policy=(
            TYPED_PORTABLE_INTEGRATION_VERSION
        ),
    )
    expected_snapshot_hash = (
        expected_ledger.production_snapshot_ledger.ledger_hash
    )
    if (
        loaded.production_snapshot_ledger.ledger_hash
        != expected_snapshot_hash
        or loaded.graph_set_hash != expected_ledger.graph_set_hash
        or loaded.model_catalog_set_hash
        != expected_ledger.model_catalog_set_hash
    ):
        raise PermissionError(
            "v3.20 production loader reconstructed a different projected "
            "ledger"
        )
    if loaded.freeze_authorization is not None:
        raise PermissionError(
            "v3.20 diagnostic production loader leaked freeze authority"
        )
    diagnostic_freeze_authority_rejected = False
    try:
        loaded.require_freeze_authorization()
    except PermissionError:
        diagnostic_freeze_authority_rejected = True
    if not diagnostic_freeze_authority_rejected:
        raise PermissionError(
            "v3.20 diagnostic ledger crossed the freeze boundary"
        )

    legacy_policy_rejected = False
    try:
        _load_typed_selection_for_execution(
            root=root,
            manifest_path=manifest_path,
            protocol=protocol,
            execution_contract=execution_contract,
            proposal_formation_policy=(
                TYPED_RECIPE_PROPOSAL_FORMATION_POLICY_VERSION
            ),
            integration_diagnostic_policy=(
                TYPED_SELECTION_INTEGRATION_VERSION
            ),
        )
    except PermissionError as exc:
        legacy_policy_rejected = "diagnostic policy" in str(exc)
    if not legacy_policy_rejected:
        raise PermissionError(
            "v3.20 production loader accepted the legacy integration policy"
        )

    full_snapshots = tuple(
        freeze_typed_recipe_selection_snapshot(graph)
        for graph in material.full_graphs
    )
    manifest = SplitManifest.read(manifest_path)
    legacy_full_ledger = freeze_typed_selection_snapshot_ledger(
        full_snapshots,
        feasibility_preregistration_hash=historical.preregistration_hash,
        feasibility_result_receipt_sha256=(
            historical.result_receipt_file_sha256
        ),
        feasibility_decision_hash=historical.decision_hash,
        feasibility_report_hash=historical.report_hash,
        manifest_hash=manifest.manifest_hash,
        source_train_receipt_hash=(
            material.evidence.source_train_receipt_hash
        ),
        expected_graph_set_hash=material.full_graph_set_hash,
        expected_model_catalog_set_hash=(
            material.full_model_catalog_set_hash
        ),
        expected_target_family_hashes=tuple(
            graph.target_family_hash for graph in material.full_graphs
        ),
    )
    legacy_full_ledger_hash = legacy_full_ledger.ledger_hash
    if legacy_full_ledger_hash == expected_snapshot_hash:
        raise PermissionError(
            "legacy full and projected snapshot ledgers unexpectedly match"
        )
    legacy_source = {
        **source,
        "snapshot_ledger_hash": legacy_full_ledger_hash,
    }
    legacy_full_ledger_rejected = False
    try:
        _load_typed_selection_for_execution(
            root=root,
            manifest_path=manifest_path,
            protocol=protocol,
            execution_contract={
                "typed_selection_snapshot_source": legacy_source
            },
            proposal_formation_policy=(
                TYPED_RECIPE_PROPOSAL_FORMATION_POLICY_VERSION
            ),
            integration_diagnostic_policy=(
                TYPED_PORTABLE_INTEGRATION_VERSION
            ),
        )
    except PermissionError as exc:
        legacy_full_ledger_rejected = (
            "production snapshot ledger drifted" in str(exc)
        )
    if not legacy_full_ledger_rejected:
        raise PermissionError(
            "v3.20 production loader accepted the legacy full snapshot ledger"
        )

    source_fields = sorted(source)
    return {
        "protocol_version": PORTABLE_TASK_CAPABILITY_PROTOCOL_VERSION,
        "integration_diagnostic_policy": (
            TYPED_PORTABLE_INTEGRATION_VERSION
        ),
        "legacy_integration_policy": (
            TYPED_SELECTION_INTEGRATION_VERSION
        ),
        "proposal_formation_policy": (
            TYPED_RECIPE_PROPOSAL_FORMATION_POLICY_VERSION
        ),
        "protocol_contract_hash": stable_hash(protocol_payload),
        "diagnostic_source_fields": source_fields,
        "diagnostic_source_field_set_hash": stable_hash(
            {"fields": source_fields}
        ),
        "diagnostic_source_omits_result_receipt_fields": True,
        "integration_result_receipt_required": False,
        "integration_result_receipt_used": False,
        "same_projected_ledger_loaded": True,
        "projected_snapshot_ledger_hash": expected_snapshot_hash,
        "legacy_full_snapshot_ledger_hash": legacy_full_ledger_hash,
        "legacy_integration_policy_rejected": legacy_policy_rejected,
        "legacy_full_snapshot_ledger_rejected": (
            legacy_full_ledger_rejected
        ),
        "freeze_authorization_present": False,
        "diagnostic_freeze_authority_rejected": (
            diagnostic_freeze_authority_rejected
        ),
        "fresh_development_protocol_freeze_eligible": False,
        "development_task_execution_authorized": False,
        "live_model_invoked": False,
        "task_backend_run_task_invoked": False,
        "evaluator_invoked": False,
        "score_computed": False,
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
    preregistration = _read_preregistration(preregistration_path)
    preregistration_hash = stable_hash(preregistration)
    implementation_hash = _implementation_file_set_hash(
        preregistration,
        preregistration_path=preregistration_path,
    )
    trace_id = stable_hash(
        {
            "policy": TYPED_PORTABLE_INTEGRATION_VERSION,
            "preregistration_hash": preregistration_hash,
            "implementation_file_set_hash": implementation_hash,
        }
    )[:24]
    audit_sink = _BoundaryAuditEventSink(event_sink)
    _emit(
        audit_sink,
        event="typed_portable_integration_started",
        trace_id=trace_id,
        payload={
            "integration_policy": TYPED_PORTABLE_INTEGRATION_VERSION,
            "preregistration_hash": preregistration_hash,
            "implementation_file_set_hash": implementation_hash,
            "decision_budget": 1,
        },
    )
    ledger, material, historical, projection_rows = (
        _load_portable_typed_selection_material(
            root=root,
            manifest_path=manifest_path,
            source_run_root=source_run_root,
            source_train_receipt=source_train_receipt,
            preregistration_path=preregistration_path,
        )
    )
    _emit(
        audit_sink,
        event="typed_portable_historical_feasibility_verified",
        trace_id=trace_id,
        payload=historical.safe_payload(),
    )
    _emit(
        audit_sink,
        event="typed_portable_train_graphs_reconstructed",
        trace_id=trace_id,
        payload={
            "trial_count": len(material.trials),
            "failure_count": len(material.evidence.failures),
            "trial_evidence_hash": material.trial_evidence_hash,
            "full_graph_set_hash": material.full_graph_set_hash,
            "full_model_catalog_set_hash": (
                material.full_model_catalog_set_hash
            ),
            "source_agent_trials_reexecuted": False,
        },
    )
    _emit(
        audit_sink,
        event="typed_portable_graphs_projected",
        trace_id=trace_id,
        payload={
            "projection_policy": PORTABLE_GRAPH_PROJECTION_POLICY,
            "projection_count": len(projection_rows),
            "projection_set_hash": stable_hash(
                {"rows": list(projection_rows)}
            ),
            "projected_graph_set_hash": ledger.graph_set_hash,
            "projected_model_catalog_set_hash": (
                ledger.model_catalog_set_hash
            ),
            "projected_snapshot_ledger_hash": (
                ledger.production_snapshot_ledger.ledger_hash
            ),
            "every_selectable_fixed_plan_has_portable_artifact_evidence_role": True,
            "capability_execution_covers_full_recipe_operator_plan": False,
            "non_access_operators_are_fixed_agent_plan": True,
            "pre_agent_sidecar_is_task_effect": False,
        },
    )
    production_loader = _verify_production_authorization_loader_route(
        root=root,
        manifest_path=manifest_path,
        source_run_root=source_run_root,
        source_train_receipt=source_train_receipt,
        preregistration=preregistration,
        preregistration_path=preregistration_path,
        expected_ledger=ledger,
        material=material,
        historical=historical,
    )
    _emit(
        audit_sink,
        event="typed_portable_production_authorization_loader_verified",
        trace_id=trace_id,
        payload=production_loader,
    )
    programs, proposer, selection, selection_tamper = (
        _exercise_opaque_production_selection(
            ledger=ledger,
            full_graphs=material.full_graphs,
            event_sink=audit_sink,
            trace_id=trace_id,
        )
    )
    _emit(
        audit_sink,
        event="typed_portable_production_selection_verified",
        trace_id=trace_id,
        payload=selection,
    )
    compiler, runtime, metadata_mutation_rejected = (
        _compile_full_bundle_and_run_canaries(
            root=root,
            manifest_path=manifest_path,
            programs=programs,
            proposer=proposer,
            preregistration=preregistration,
            event_sink=audit_sink,
            trace_id=trace_id,
        )
    )
    _emit(
        audit_sink,
        event="typed_portable_all_three_evidence_roles_compiled",
        trace_id=trace_id,
        payload=compiler,
    )
    _emit(
        audit_sink,
        event="typed_portable_pre_agent_canaries_verified",
        trace_id=trace_id,
        payload={
            "runtime_policy": runtime["runtime_policy"],
            "canary_count": runtime["canary_count"],
            "canary_set_hash": runtime["canary_set_hash"],
            "container_network": "none",
            "container_cleanup_verified": runtime[
                "container_cleanup_verified"
            ],
            "agent_started": False,
            "model_invoked": False,
            "task_backend_run_task_invoked": False,
        },
    )
    tamper = {
        **selection_tamper,
        "compiled_metadata_mutation": metadata_mutation_rejected,
    }
    if tuple(sorted(tamper)) != tuple(sorted(_TAMPER_PROBE_IDS)):
        raise PermissionError("typed portable tamper probe set drifted")
    tamper_rows = [
        {"probe_id": probe_id, "rejected": bool(tamper[probe_id])}
        for probe_id in _TAMPER_PROBE_IDS
    ]
    _emit(
        audit_sink,
        event="typed_portable_tamper_probes_completed",
        trace_id=trace_id,
        payload={
            "probe_count": len(tamper_rows),
            "rejection_count": sum(row["rejected"] for row in tamper_rows),
            "probe_set_hash": stable_hash({"probes": tamper_rows}),
        },
    )

    acceptance = {
        "historical_feasibility_evidence_passed": True,
        "current_train_reconstruction_passed": (
            len(material.full_graphs) == REQUIRED_SLOT_COUNT
            and bool(material.trials)
        ),
        "portable_projection_closure_passed": (
            len(projection_rows) == REQUIRED_SLOT_COUNT
            and all(
                row[
                    "every_selectable_fixed_plan_has_portable_artifact_evidence_role"
                ]
                and row[
                    "capability_execution_covers_full_recipe_operator_plan"
                ]
                is False
                and row["non_access_operators_are_fixed_agent_plan"] is True
                and row["pre_agent_sidecar_is_task_effect"] is False
                for row in projection_rows
            )
        ),
        "production_authorization_loader_route_passed": (
            production_loader["protocol_version"] == "3.20.0"
            and production_loader["integration_diagnostic_policy"]
            == TYPED_PORTABLE_INTEGRATION_VERSION
            and production_loader["same_projected_ledger_loaded"] is True
            and production_loader["projected_snapshot_ledger_hash"]
            == ledger.production_snapshot_ledger.ledger_hash
            and production_loader[
                "diagnostic_source_omits_result_receipt_fields"
            ]
            is True
            and production_loader["diagnostic_source_fields"]
            == sorted(_DIAGNOSTIC_TYPED_SOURCE_FIELDS)
            and production_loader["integration_result_receipt_required"]
            is False
            and production_loader["integration_result_receipt_used"]
            is False
            and production_loader["legacy_integration_policy_rejected"]
            is True
            and production_loader["legacy_full_snapshot_ledger_rejected"]
            is True
            and production_loader["freeze_authorization_present"] is False
            and production_loader[
                "diagnostic_freeze_authority_rejected"
            ]
            is True
            and production_loader[
                "fresh_development_protocol_freeze_eligible"
            ]
            is False
            and production_loader[
                "development_task_execution_authorized"
            ]
            is False
        ),
        "opaque_production_selection_passed": (
            selection["root_program_count"] == REQUIRED_SLOT_COUNT
            and selection["model_authored_primitive_count"] == 0
            and selection["live_model_invoked"] is False
            and selection["behavioral_alias_recipe_ids_expressible"] is False
            and selection[
                "diversity_counts_deduplicated_recipe_treatment_only"
            ]
            is True
        ),
        "full_bundle_compile_passed": (
            compiler["program_count"] == REQUIRED_SLOT_COUNT
            and compiler["all_three_typed_plan_bindings_compiled"] is True
            and compiler["recipe_operator_execution_compiled"] is False
            and compiler["task_effect_compiled"] is False
        ),
        "pre_agent_runtime_passed": (
            runtime["canary_count"] == REQUIRED_SLOT_COUNT
            and runtime["container_network"] == "none"
            and runtime["container_cleanup_verified"] is True
            and all(
                row.get("container_cleanup_verified") is True
                for row in runtime["canaries"]
            )
            and runtime["agent_started"] is False
        ),
        "fixed_tamper_probes_passed": all(
            row["rejected"] for row in tamper_rows
        ),
        "offline_non_scoring_boundary_passed": True,
    }
    if set(acceptance) != _ACCEPTANCE_KEYS:
        raise PermissionError("typed portable acceptance vector drifted")
    integration_passed = all(acceptance.values())
    predecision_events = list(audit_sink.events)
    boundary_issues = _offline_boundary_issues(predecision_events)
    if boundary_issues:
        acceptance["offline_non_scoring_boundary_passed"] = False
        integration_passed = False
    event_counts = dict(
        sorted(
            Counter(
                str(row.get("event") or "") for row in predecision_events
            ).items()
        )
    )
    decision_hash = stable_hash(
        {
            "integration_policy": TYPED_PORTABLE_INTEGRATION_VERSION,
            "preregistration_hash": preregistration_hash,
            "implementation_file_set_hash": implementation_hash,
            "historical_binding_hash": historical.binding_hash,
            "full_graph_set_hash": material.full_graph_set_hash,
            "projected_ledger_hash": (
                ledger.production_snapshot_ledger.ledger_hash
            ),
            "production_authorization_loader_receipt_hash": stable_hash(
                production_loader
            ),
            "selection_receipt_hash": stable_hash(selection),
            "compiler_receipt_hash": stable_hash(compiler),
            "runtime_receipt_hash": stable_hash(runtime),
            "tamper_probe_set_hash": stable_hash({"probes": tamper_rows}),
            "acceptance": acceptance,
        }
    )
    report: dict[str, Any] = {
        "integration_policy": TYPED_PORTABLE_INTEGRATION_VERSION,
        "decision_budget": 1,
        "decision_ordinal": 1,
        "preregistration": {
            "stable_hash": preregistration_hash,
            "implementation_file_set_hash": implementation_hash,
        },
        "historical_feasibility_evidence": historical.safe_payload(),
        "current_train_reconstruction": {
            "trial_count": len(material.trials),
            "failure_count": len(material.evidence.failures),
            "trial_evidence_hash": material.trial_evidence_hash,
            "full_graph_set_hash": material.full_graph_set_hash,
            "full_model_catalog_set_hash": (
                material.full_model_catalog_set_hash
            ),
            "full_graph_hashes": [
                row.graph_hash for row in material.full_graphs
            ],
            "raw_content_persisted": False,
        },
        "portable_projection": {
            "policy": PORTABLE_GRAPH_PROJECTION_POLICY,
            "rows": list(projection_rows),
            "projected_graph_set_hash": ledger.graph_set_hash,
            "projected_model_catalog_set_hash": (
                ledger.model_catalog_set_hash
            ),
            "projected_snapshot_ledger_hash": (
                ledger.production_snapshot_ledger.ledger_hash
            ),
            "current_execution_authority_granted": False,
        },
        "production_authorization_loader": production_loader,
        "production_selection": selection,
        "compiler_provenance": compiler,
        "pre_agent_runtime": runtime,
        "tamper_probes": tamper_rows,
        "offline_boundary_contract": {
            "flags": dict(_BOUNDARY_FLAGS),
            "issues": list(boundary_issues),
            "predecision_event_count": len(predecision_events),
            "predecision_event_counts": event_counts,
            "predecision_event_set_hash": stable_hash(
                {"events": predecision_events}
            ),
            "validation_public_input_accessed": True,
            "validation_outcome_accessed": False,
            "task_backend_run_task_invoked": False,
            "live_model_invoked": False,
            "live_evaluator_invoked": False,
            "verifier_content_accessed": False,
            "score_computed": False,
        },
        "acceptance": acceptance,
        "integration_passed": integration_passed,
        "fresh_development_protocol_freeze_eligible_if_passed": (
            integration_passed
        ),
        "development_task_execution_currently_authorized": False,
        "decision_hash": decision_hash,
        "model_call_count": 0,
        "task_backend_run_task_call_count": 0,
        "evaluator_call_count": 0,
        "verifier_call_count": 0,
        "score_count": 0,
        "raw_content_persisted": False,
    }
    report["report_hash"] = stable_hash(report)
    _emit(
        audit_sink,
        event="typed_portable_integration_completed",
        trace_id=trace_id,
        payload={
            "integration_policy": TYPED_PORTABLE_INTEGRATION_VERSION,
            "decision_hash": decision_hash,
            "report_hash": report["report_hash"],
            "integration_passed": integration_passed,
            "acceptance_hash": stable_hash(acceptance),
            "projected_snapshot_ledger_hash": (
                ledger.production_snapshot_ledger.ledger_hash
            ),
            "promotion_gate_or_score": False,
        },
    )
    return report


def _offline_boundary_issues(
    events: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    issues: list[str] = []
    forbidden_event_fragments = (
        "counterfactual",
        "promotion",
        "evaluator",
        "verifier",
        "backend_invoked",
        "sealed",
    )
    for row in events:
        payload = row.get("payload")
        if not isinstance(payload, Mapping):
            issues.append("event_payload_missing")
            continue
        for key, expected in _BOUNDARY_FLAGS.items():
            if payload.get(key) is not expected:
                issues.append(f"boundary_flag_mismatch:{key}")
        name = str(row.get("event") or "")
        if any(fragment in name for fragment in forbidden_event_fragments):
            issues.append("forbidden_event_observed")
    return tuple(sorted(set(issues)))


def run_typed_portable_integration(
    *,
    root: str | Path,
    manifest_path: str | Path,
    source_run_root: str | Path,
    source_train_receipt: str | Path,
    preregistration_path: str | Path,
    report_path: str | Path,
    events_path: str | Path,
) -> dict[str, Any]:
    """Consume the sole formal decision and write report/events/lock."""

    preregistration = _read_preregistration(preregistration_path)
    canonical = _canonical_decision_paths(
        preregistration,
        preregistration_path=preregistration_path,
    )
    report = Path(report_path).expanduser().resolve()
    events = Path(events_path).expanduser().resolve()
    if report != canonical["report"] or events != canonical["events"]:
        raise PermissionError(
            "formal typed portable integration requires canonical paths"
        )
    if report.exists() or events.exists() or canonical["result_receipt"].exists():
        raise FileExistsError(
            "typed portable integration output already exists"
        )
    _reserve_decision_lock(
        canonical["decision_lock"],
        preregistration_hash=stable_hash(preregistration),
    )
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
        _completed_decision_lock(
            report=result,
            preregistration_hash=stable_hash(preregistration),
        ),
    )
    return result


def verify_existing_typed_portable_integration(
    *,
    root: str | Path,
    manifest_path: str | Path,
    source_run_root: str | Path,
    source_train_receipt: str | Path,
    preregistration_path: str | Path,
    report_path: str | Path,
    events_path: str | Path,
) -> dict[str, Any]:
    """Recompute the diagnostic exactly before minting its result receipt."""

    preregistration = _read_preregistration(preregistration_path)
    canonical = _canonical_decision_paths(
        preregistration,
        preregistration_path=preregistration_path,
    )
    report_path_resolved = Path(report_path).expanduser().resolve(strict=True)
    events_path_resolved = Path(events_path).expanduser().resolve(strict=True)
    if (
        report_path_resolved != canonical["report"]
        or events_path_resolved != canonical["events"]
    ):
        raise PermissionError(
            "typed portable replay paths are not canonical"
        )
    declared_report = _read_json_object(report_path_resolved)
    replay_events = MemoryEventSink()
    expected_report = _compute_integration(
        root=root,
        manifest_path=manifest_path,
        source_run_root=source_run_root,
        source_train_receipt=source_train_receipt,
        preregistration_path=preregistration_path,
        event_sink=replay_events,
    )
    if dict(declared_report) != expected_report:
        raise PermissionError(
            "typed portable integration report does not replay exactly"
        )
    declared_events = _read_event_rows(events_path_resolved)
    if declared_events != replay_events.events:
        raise PermissionError(
            "typed portable integration events do not replay exactly"
        )
    lock = _read_json_object(
        canonical["decision_lock"].resolve(strict=True)
    )
    _verify_decision_lock(
        lock,
        report=expected_report,
        preregistration_hash=stable_hash(preregistration),
    )
    expected_receipt = _build_result_receipt(
        preregistration=preregistration,
        preregistration_path=preregistration_path,
        report=expected_report,
        events=declared_events,
        decision_lock=lock,
        canonical=canonical,
    )
    receipt_path = canonical["result_receipt"]
    if receipt_path.exists():
        if dict(_read_json_object(receipt_path.resolve(strict=True))) != (
            expected_receipt
        ):
            raise PermissionError(
                "typed portable integration result receipt drifted"
            )
    else:
        _write_json_atomic(receipt_path, expected_receipt)
    verify_typed_portable_integration_result_receipt(
        preregistration_path=preregistration_path,
        result_receipt_path=receipt_path,
    )
    return {
        **expected_report,
        "integration_reuse_verified": True,
        "result_receipt_path": str(receipt_path),
        "result_receipt_file_sha256": _sha256_file(receipt_path),
    }


def verify_typed_portable_integration_result_receipt(
    *,
    preregistration_path: str | Path,
    result_receipt_path: str | Path,
) -> Mapping[str, Any]:
    preregistration = _read_preregistration(preregistration_path)
    canonical = _canonical_decision_paths(
        preregistration,
        preregistration_path=preregistration_path,
    )
    receipt_path = Path(result_receipt_path).expanduser().resolve(strict=True)
    if receipt_path != canonical["result_receipt"]:
        raise PermissionError(
            "typed portable result receipt path is not canonical"
        )
    declared = _read_json_object(receipt_path)
    report = _read_json_object(canonical["report"].resolve(strict=True))
    events = _read_event_rows(canonical["events"].resolve(strict=True))
    decision_lock = _read_json_object(
        canonical["decision_lock"].resolve(strict=True)
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
            "typed portable result receipt or artifact drifted"
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
        project_root,
        preregistration["manifest"],
    )
    source_receipt_file = _resolve_bound_file(
        project_root,
        preregistration["source_train_receipt"],
    )
    _resolve_bound_directory(project_root, preregistration["benchmark_root"])
    _resolve_bound_directory(project_root, preregistration["source_run_root"])
    paths = preregistration["canonical_decision_paths"]
    return {
        "result_receipt_version": TYPED_PORTABLE_RESULT_RECEIPT_VERSION,
        "integration_policy": TYPED_PORTABLE_INTEGRATION_VERSION,
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
            "benchmark_root": preregistration["benchmark_root"],
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
            "historical_binding_hash": report[
                "historical_feasibility_evidence"
            ]["historical_binding_hash"],
            "full_graph_set_hash": report["current_train_reconstruction"][
                "full_graph_set_hash"
            ],
            "projected_snapshot_ledger_hash": report[
                "portable_projection"
            ]["projected_snapshot_ledger_hash"],
        },
        "decision_hash": report["decision_hash"],
        "report_hash": report["report_hash"],
        "canonical_artifacts": {
            "report": {
                "path": paths["report"],
                "sha256": _sha256_file(canonical["report"]),
            },
            "events": {
                "path": paths["events"],
                "sha256": _sha256_file(canonical["events"]),
                "event_count": len(events),
                "event_counts": dict(
                    sorted(
                        Counter(
                            str(row.get("event") or "") for row in events
                        ).items()
                    )
                ),
                "event_set_hash": stable_hash({"events": list(events)}),
            },
            "decision_lock": {
                "path": paths["decision_lock"],
                "sha256": _sha256_file(canonical["decision_lock"]),
                "stable_hash": stable_hash(decision_lock),
            },
        },
        "portable_projection": {
            "graph_set_hash": report["portable_projection"][
                "projected_graph_set_hash"
            ],
            "model_catalog_set_hash": report["portable_projection"][
                "projected_model_catalog_set_hash"
            ],
            "snapshot_ledger_hash": report["portable_projection"][
                "projected_snapshot_ledger_hash"
            ],
        },
        "production_authorization_loader": {
            key: report["production_authorization_loader"][key]
            for key in (
                "protocol_version",
                "integration_diagnostic_policy",
                "legacy_integration_policy",
                "protocol_contract_hash",
                "diagnostic_source_fields",
                "diagnostic_source_field_set_hash",
                "diagnostic_source_omits_result_receipt_fields",
                "integration_result_receipt_required",
                "integration_result_receipt_used",
                "same_projected_ledger_loaded",
                "projected_snapshot_ledger_hash",
                "legacy_full_snapshot_ledger_hash",
                "legacy_integration_policy_rejected",
                "legacy_full_snapshot_ledger_rejected",
                "freeze_authorization_present",
                "diagnostic_freeze_authority_rejected",
                "fresh_development_protocol_freeze_eligible",
                "development_task_execution_authorized",
            )
        },
        "compiler_provenance": {
            key: report["compiler_provenance"][key]
            for key in (
                "compile_manifest_hash",
                "typed_binding_set_hash",
                "typed_snapshot_ledger_hash",
                "portable_role_spec_set_hash",
                "source_receipt_set_hash",
                "compiler_event_set_hash",
            )
        },
        "pre_agent_runtime": {
            "runtime_policy": report["pre_agent_runtime"][
                "runtime_policy"
            ],
            "canary_count": report["pre_agent_runtime"]["canary_count"],
            "canary_set_hash": report["pre_agent_runtime"][
                "canary_set_hash"
            ],
            "container_network": "none",
            "container_cleanup_verified": report["pre_agent_runtime"][
                "container_cleanup_verified"
            ],
            "agent_started": False,
        },
        "integration_passed": True,
        "acceptance": dict(report["acceptance"]),
        "exact_replay_verified": True,
        "fresh_development_protocol_freeze_eligible": True,
        "development_task_execution_authorized": False,
        "promotion_gate_or_score": False,
        "raw_content_persisted": False,
    }


def _validate_completed_result_artifacts(
    *,
    preregistration: Mapping[str, Any],
    report: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    decision_lock: Mapping[str, Any],
) -> None:
    report_without_hash = dict(report)
    report_hash = report_without_hash.pop("report_hash", None)
    if (
        report_hash != stable_hash(report_without_hash)
        or report.get("integration_policy")
        != TYPED_PORTABLE_INTEGRATION_VERSION
        or report.get("integration_passed") is not True
        or report.get("acceptance") != preregistration["acceptance"]
        or report.get("fresh_development_protocol_freeze_eligible_if_passed")
        is not True
        or report.get("development_task_execution_currently_authorized")
        is not False
    ):
        raise PermissionError(
            "typed portable completed report is not authoritative"
        )
    production_loader = report.get("production_authorization_loader")
    projection = report.get("portable_projection")
    projected_ledger_hash = (
        projection.get("projected_snapshot_ledger_hash")
        if isinstance(projection, Mapping)
        else None
    )
    source_fields = sorted(_DIAGNOSTIC_TYPED_SOURCE_FIELDS)
    if (
        not isinstance(production_loader, Mapping)
        or production_loader.get("protocol_version") != "3.20.0"
        or production_loader.get("integration_diagnostic_policy")
        != TYPED_PORTABLE_INTEGRATION_VERSION
        or production_loader.get("legacy_integration_policy")
        != TYPED_SELECTION_INTEGRATION_VERSION
        or production_loader.get("diagnostic_source_fields")
        != source_fields
        or production_loader.get("diagnostic_source_field_set_hash")
        != stable_hash({"fields": source_fields})
        or production_loader.get(
            "diagnostic_source_omits_result_receipt_fields"
        )
        is not True
        or production_loader.get("integration_result_receipt_required")
        is not False
        or production_loader.get("integration_result_receipt_used")
        is not False
        or production_loader.get("same_projected_ledger_loaded") is not True
        or not _is_sha256_text(projected_ledger_hash)
        or production_loader.get("projected_snapshot_ledger_hash")
        != projected_ledger_hash
        or not _is_sha256_text(
            production_loader.get("legacy_full_snapshot_ledger_hash")
        )
        or production_loader.get("legacy_full_snapshot_ledger_hash")
        == projected_ledger_hash
        or production_loader.get("legacy_integration_policy_rejected")
        is not True
        or production_loader.get("legacy_full_snapshot_ledger_rejected")
        is not True
        or production_loader.get("freeze_authorization_present") is not False
        or production_loader.get("diagnostic_freeze_authority_rejected")
        is not True
        or production_loader.get(
            "fresh_development_protocol_freeze_eligible"
        )
        is not False
        or production_loader.get("development_task_execution_authorized")
        is not False
    ):
        raise PermissionError(
            "typed portable production authorization loader receipt drifted"
        )
    runtime = report.get("pre_agent_runtime")
    runtime_canaries = (
        runtime.get("canaries") if isinstance(runtime, Mapping) else None
    )
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("container_cleanup_verified") is not True
        or not isinstance(runtime_canaries, list)
        or len(runtime_canaries) != REQUIRED_SLOT_COUNT
        or any(
            not isinstance(row, Mapping)
            or row.get("container_cleanup_verified") is not True
            for row in runtime_canaries
        )
        or runtime.get("canary_set_hash")
        != stable_hash({"canaries": runtime_canaries})
    ):
        raise PermissionError(
            "typed portable runtime container cleanup receipt drifted"
        )
    _verify_decision_lock(
        decision_lock,
        report=report,
        preregistration_hash=stable_hash(preregistration),
    )
    if not events or events[-1].get("event") != (
        "typed_portable_integration_completed"
    ):
        raise PermissionError(
            "typed portable completion event is missing"
        )
    predecision = list(events[:-1])
    production_loader_events = [
        row
        for row in predecision
        if row.get("event")
        == "typed_portable_production_authorization_loader_verified"
    ]
    if len(production_loader_events) != 1 or not isinstance(
        production_loader_events[0].get("payload"), Mapping
    ) or any(
        production_loader_events[0]["payload"].get(key) != value
        for key, value in production_loader.items()
    ):
        raise PermissionError(
            "typed portable production authorization loader event drifted"
        )
    boundary = report.get("offline_boundary_contract")
    if not isinstance(boundary, Mapping) or (
        boundary.get("issues") != []
        or boundary.get("predecision_event_count") != len(predecision)
        or boundary.get("predecision_event_set_hash")
        != stable_hash({"events": predecision})
        or boundary.get("predecision_event_counts")
        != dict(
            sorted(
                Counter(
                    str(row.get("event") or "") for row in predecision
                ).items()
            )
        )
    ):
        raise PermissionError(
            "typed portable event commitment drifted"
        )
    payload = events[-1].get("payload")
    if not isinstance(payload, Mapping) or (
        payload.get("decision_hash") != report["decision_hash"]
        or payload.get("report_hash") != report_hash
        or payload.get("integration_passed") is not True
    ):
        raise PermissionError(
            "typed portable completion event drifted"
        )


def _read_preregistration(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    payload = dict(_read_json_object(resolved))
    fixed = {
        "integration_policy": TYPED_PORTABLE_INTEGRATION_VERSION,
        "decision_budget": 1,
        "decision_scope": (
            "offline_non_scoring_portable_integration_only"
        ),
        "evaluator_epoch": TYPED_PORTABLE_EVALUATOR_EPOCH,
        "slot_count": REQUIRED_SLOT_COUNT,
        "minimum_family_support": MINIMUM_FAMILY_SUPPORT,
        "portable_projection_policy": PORTABLE_GRAPH_PROJECTION_POLICY,
        "portable_compiler_mode": (
            _portable_capability.PORTABLE_TASK_CAPABILITY_COMPILER_VERSION
        ),
        "fixed_tamper_probe_ids": list(_TAMPER_PROBE_IDS),
        "acceptance_predicates": dict(_ACCEPTANCE_PREDICATES),
        "acceptance": {key: True for key in _ACCEPTANCE_PREDICATES},
        "boundary_contract": dict(_BOUNDARY_FLAGS),
    }
    for key, expected in fixed.items():
        if payload.get(key) != expected:
            raise PermissionError(
                f"typed portable preregistration drifted: {key}"
            )
    if set(payload.get("acceptance", {})) != _ACCEPTANCE_KEYS:
        raise PermissionError("typed portable acceptance vector drifted")

    project_root = _project_root(resolved)
    for key in (
        "benchmark_root",
        "manifest",
        "source_run_root",
        "source_train_receipt",
    ):
        _safe_relative_path(payload.get(key))
    _resolve_bound_directory(project_root, payload["benchmark_root"])
    manifest_file = _resolve_bound_file(project_root, payload["manifest"])
    source_receipt_file = _resolve_bound_file(
        project_root,
        payload["source_train_receipt"],
    )
    _resolve_bound_directory(project_root, payload["source_run_root"])
    for key in (
        "manifest_hash",
        "manifest_file_sha256",
        "source_train_receipt_hash",
        "source_train_receipt_file_sha256",
        "expected_implementation_file_set_hash",
    ):
        if not _is_sha256_text(payload.get(key)):
            raise PermissionError(
                f"typed portable hash binding is malformed: {key}"
            )
    manifest = SplitManifest.read(manifest_file)
    if (
        manifest.manifest_hash != payload["manifest_hash"]
        or _sha256_file(manifest_file) != payload["manifest_file_sha256"]
        or _sha256_file(source_receipt_file)
        != payload["source_train_receipt_file_sha256"]
    ):
        raise PermissionError("typed portable source file binding drifted")

    full = payload.get("full_graph_commitments")
    full_keys = {
        "target_family_hashes",
        "graph_hashes",
        "model_catalog_hashes",
        "trial_evidence_hash",
        "graph_set_hash",
        "model_catalog_set_hash",
    }
    if not isinstance(full, Mapping) or set(full) != full_keys:
        raise PermissionError(
            "typed portable full graph commitments are malformed"
        )
    for key in (
        "target_family_hashes",
        "graph_hashes",
        "model_catalog_hashes",
    ):
        values = full.get(key)
        if (
            not isinstance(values, list)
            or len(values) != REQUIRED_SLOT_COUNT
            or len(set(values)) != REQUIRED_SLOT_COUNT
            or not all(_is_sha256_text(value) for value in values)
        ):
            raise PermissionError(
                f"typed portable full graph order malformed: {key}"
            )
    for key in (
        "trial_evidence_hash",
        "graph_set_hash",
        "model_catalog_set_hash",
    ):
        if not _is_sha256_text(full.get(key)):
            raise PermissionError(
                f"typed portable full graph hash malformed: {key}"
            )

    projection = payload.get("portable_projection")
    if not isinstance(projection, Mapping) or set(projection) != {
        "rows",
        "graph_set_hash",
        "model_catalog_set_hash",
        "ledger_hash",
    }:
        raise PermissionError(
            "typed portable projection commitment is malformed"
        )
    projection_rows = projection.get("rows")
    expected_projection_row_keys = {
        "target_family_hash",
        "parent_graph_hash",
        "projected_graph_hash",
        "projected_model_catalog_hash",
        "projected_artifact_count",
        "projected_recipe_count",
        "projected_artifact_set_hash",
        "projected_recipe_set_hash",
        "projected_workflow_set_hash",
        "portable_selector_context_hash",
        "selectable_recipe_set_hash",
        "source_executable_artifact_count",
        "retained_behavioral_artifact_count",
        "behavioral_alias_artifact_count",
        "behavioral_alias_set_hash",
        "behavioral_treatment_signature_hashes",
        "behavioral_treatment_signature_set_hash",
        "behavioral_aliases_deduplicated",
        "diversity_counts_behavioral_signature_only",
        "every_selectable_fixed_plan_has_portable_artifact_evidence_role",
        "capability_execution_covers_full_recipe_operator_plan",
        "non_access_operators_are_fixed_agent_plan",
        "pre_agent_sidecar_is_task_effect",
        "raw_artifact_locators_persisted",
    }
    if (
        not isinstance(projection_rows, list)
        or len(projection_rows) != REQUIRED_SLOT_COUNT
    ):
        raise PermissionError("typed portable projection row count drifted")
    for index, row in enumerate(projection_rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_projection_row_keys
            or row.get("target_family_hash")
            != full["target_family_hashes"][index]
            or row.get("parent_graph_hash") != full["graph_hashes"][index]
            or row.get(
                "every_selectable_fixed_plan_has_portable_artifact_evidence_role"
            )
            is not True
            or row.get(
                "capability_execution_covers_full_recipe_operator_plan"
            )
            is not False
            or row.get("non_access_operators_are_fixed_agent_plan") is not True
            or row.get("pre_agent_sidecar_is_task_effect") is not False
            or row.get("raw_artifact_locators_persisted") is not False
            or not isinstance(row.get("projected_artifact_count"), int)
            or isinstance(row.get("projected_artifact_count"), bool)
            or row.get("projected_artifact_count", 0) < 1
            or not isinstance(row.get("projected_recipe_count"), int)
            or isinstance(row.get("projected_recipe_count"), bool)
            or row.get("projected_recipe_count") != 3
            or row.get("retained_behavioral_artifact_count")
            != row.get("projected_artifact_count")
            or row.get("source_executable_artifact_count")
            != row.get("projected_artifact_count")
            + row.get("behavioral_alias_artifact_count", -1)
            or not isinstance(row.get("behavioral_alias_artifact_count"), int)
            or isinstance(row.get("behavioral_alias_artifact_count"), bool)
            or row.get("behavioral_alias_artifact_count", -1) < 0
            or row.get("behavioral_aliases_deduplicated") is not True
            or row.get("diversity_counts_behavioral_signature_only")
            is not True
        ):
            raise PermissionError("typed portable projection row drifted")
        for key in (
            "target_family_hash",
            "parent_graph_hash",
            "projected_graph_hash",
            "projected_model_catalog_hash",
            "projected_artifact_set_hash",
            "projected_recipe_set_hash",
            "projected_workflow_set_hash",
            "portable_selector_context_hash",
            "selectable_recipe_set_hash",
            "behavioral_alias_set_hash",
            "behavioral_treatment_signature_set_hash",
        ):
            if not _is_sha256_text(row.get(key)):
                raise PermissionError(
                    "typed portable projection hash is malformed"
                )
        signatures = row.get("behavioral_treatment_signature_hashes")
        if (
            not isinstance(signatures, list)
            or signatures != sorted(set(signatures))
            or len(signatures) != row["projected_artifact_count"]
            or not all(_is_sha256_text(value) for value in signatures)
            or stable_hash({"hashes": signatures})
            != row["behavioral_treatment_signature_set_hash"]
        ):
            raise PermissionError(
                "typed portable behavioral treatment signatures drifted"
            )
    for key in ("graph_set_hash", "model_catalog_set_hash", "ledger_hash"):
        if not _is_sha256_text(projection.get(key)):
            raise PermissionError(
                "typed portable projected ledger hash is malformed"
            )

    historical = payload.get("historical_feasibility")
    if not isinstance(historical, Mapping) or set(historical) != {
        "preregistration",
        "result_receipt",
        "report",
        "events",
        "decision_lock",
        "decision_hash",
        "report_hash",
    }:
        raise PermissionError(
            "typed portable historical feasibility ledger is malformed"
        )
    for key in ("decision_hash", "report_hash"):
        if not _is_sha256_text(historical.get(key)):
            raise PermissionError(
                "typed portable historical decision hash is malformed"
            )
    for key in (
        "preregistration",
        "result_receipt",
        "report",
        "events",
        "decision_lock",
    ):
        row = historical.get(key)
        expected_keys = (
            {"path", "sha256", "stable_hash"}
            if key in {"preregistration", "result_receipt"}
            else {"path", "sha256"}
        )
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_keys
            or not _is_sha256_text(row.get("sha256"))
            or (
                "stable_hash" in expected_keys
                and not _is_sha256_text(row.get("stable_hash"))
            )
        ):
            raise PermissionError(
                "typed portable historical artifact binding is malformed"
            )
        _safe_relative_path(row["path"])

    canaries = _validated_runtime_canaries(payload, manifest=manifest)
    if [row["family_hash"] for row in canaries] != full[
        "target_family_hashes"
    ]:
        raise PermissionError(
            "typed portable canary order does not match selector slots"
        )
    _canonical_decision_paths(payload, preregistration_path=resolved)
    actual_implementation_hash = _implementation_file_set_hash(
        payload,
        preregistration_path=resolved,
    )
    if actual_implementation_hash != payload[
        "expected_implementation_file_set_hash"
    ]:
        raise PermissionError(
            "typed portable implementation binding drifted"
        )
    return payload


def build_implementation_file_binding(
    preregistration_path: str | Path,
) -> dict[str, Any]:
    """Compute explicit implementation rows for preregistration preparation."""

    resolved = Path(preregistration_path).expanduser().resolve(strict=True)
    payload = _read_json_object(resolved)
    project_root = _project_root(resolved)
    raw_rows = payload.get("implementation_files")
    if not isinstance(raw_rows, list):
        raise PermissionError(
            "typed portable implementation file list is missing"
        )
    rows: list[dict[str, str]] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256"}:
            raise PermissionError(
                "typed portable implementation file row is malformed"
            )
        relative = _safe_relative_path(raw["path"]).as_posix()
        path = _resolve_bound_file(project_root, relative)
        rows.append({"path": relative, "sha256": _sha256_file(path)})
    rows.sort(key=lambda row: row["path"])
    if len({row["path"] for row in rows}) != len(rows):
        raise PermissionError(
            "typed portable implementation files are not unique"
        )
    if not _REQUIRED_IMPLEMENTATION_FILES <= {
        row["path"] for row in rows
    }:
        raise PermissionError(
            "typed portable implementation file coverage is incomplete"
        )
    return {
        "implementation_files": rows,
        "expected_implementation_file_set_hash": stable_hash(
            {"files": rows}
        ),
    }


def _implementation_file_set_hash(
    preregistration: Mapping[str, Any],
    *,
    preregistration_path: str | Path,
) -> str:
    project_root = _project_root(preregistration_path)
    raw_rows = preregistration.get("implementation_files")
    if not isinstance(raw_rows, list):
        raise PermissionError(
            "typed portable implementation binding is missing"
        )
    declared_rows: list[dict[str, str]] = []
    for raw in raw_rows:
        if (
            not isinstance(raw, Mapping)
            or set(raw) != {"path", "sha256"}
            or not _is_sha256_text(raw.get("sha256"))
        ):
            raise PermissionError(
                "typed portable implementation row is malformed"
            )
        relative = _safe_relative_path(raw["path"]).as_posix()
        path = _resolve_bound_file(project_root, relative)
        if _sha256_file(path) != raw["sha256"]:
            raise PermissionError(
                "typed portable implementation file hash drifted"
            )
        declared_rows.append({"path": relative, "sha256": raw["sha256"]})
    if declared_rows != sorted(declared_rows, key=lambda row: row["path"]):
        raise PermissionError(
            "typed portable implementation rows are not canonical"
        )
    if (
        len({row["path"] for row in declared_rows}) != len(declared_rows)
        or not _REQUIRED_IMPLEMENTATION_FILES
        <= {row["path"] for row in declared_rows}
    ):
        raise PermissionError(
            "typed portable implementation file coverage is incomplete"
        )
    return stable_hash({"files": declared_rows})


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
        raise PermissionError("typed portable canonical paths are missing")
    project_root = _project_root(preregistration_path)
    paths: dict[str, Path] = {}
    for key in ("report", "events", "decision_lock", "result_receipt"):
        relative = _safe_relative_path(declared[key])
        current = project_root
        for part in relative.parts:
            current = current / part
            if current.exists() and current.is_symlink():
                raise PermissionError(
                    "typed portable canonical path contains a symlink"
                )
        resolved = (project_root / relative).resolve()
        try:
            resolved.relative_to(project_root)
        except ValueError as exc:
            raise PermissionError(
                "typed portable canonical path escaped project root"
            ) from exc
        paths[key] = resolved
    if len(set(paths.values())) != len(paths):
        raise PermissionError("typed portable canonical paths overlap")
    return paths


def _project_root(preregistration_path: str | Path) -> Path:
    return (
        Path(preregistration_path)
        .expanduser()
        .resolve(strict=True)
        .parent.parent
    )


def _safe_relative_path(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PermissionError("typed portable bound path is malformed")
    relative = Path(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise PermissionError("typed portable bound path is unsafe")
    return relative


def _resolve_bound_file(project_root: Path, value: Any) -> Path:
    relative = _safe_relative_path(value)
    current = project_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise PermissionError("typed portable bound symlink is forbidden")
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise PermissionError(
            "typed portable bound file escaped project root"
        ) from exc
    if not resolved.is_file():
        raise PermissionError("typed portable bound file is missing")
    return resolved


def _resolve_bound_directory(project_root: Path, value: Any) -> Path:
    relative = _safe_relative_path(value)
    current = project_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise PermissionError("typed portable bound symlink is forbidden")
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise PermissionError(
            "typed portable bound directory escaped project root"
        ) from exc
    if not resolved.is_dir():
        raise PermissionError("typed portable bound directory is missing")
    return resolved


def _require_declared_file(
    actual: str | Path,
    *,
    project_root: Path,
    declared: Any,
    label: str,
) -> Path:
    expected = _resolve_bound_file(project_root, declared)
    resolved = Path(actual).expanduser().resolve(strict=True)
    if resolved != expected:
        raise PermissionError(f"typed portable {label} path is not canonical")
    return resolved


def _require_declared_directory(
    actual: str | Path,
    *,
    project_root: Path,
    declared: Any,
    label: str,
) -> Path:
    expected = _resolve_bound_directory(project_root, declared)
    resolved = Path(actual).expanduser().resolve(strict=True)
    if resolved != expected:
        raise PermissionError(f"typed portable {label} path is not canonical")
    return resolved


def _completed_decision_lock(
    *,
    report: Mapping[str, Any],
    preregistration_hash: str,
) -> dict[str, Any]:
    return {
        "lock_version": "typed_portable_integration_decision_lock_v1",
        "decision_ordinal": 1,
        "state": "completed",
        "preregistration_hash": preregistration_hash,
        "decision_hash": report["decision_hash"],
        "report_hash": report["report_hash"],
        "integration_passed": report["integration_passed"],
        "raw_content_persisted": False,
    }


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
            "typed portable integration decision budget is consumed"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "lock_version": (
                    "typed_portable_integration_decision_lock_v1"
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
    if dict(lock) != _completed_decision_lock(
        report=report,
        preregistration_hash=preregistration_hash,
    ):
        raise PermissionError(
            "typed portable decision lock does not match"
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
            stage="benchmark.skilllearn.typed_portable_integration",
            trace_id=trace_id,
            payload=dict(payload),
        )
    )


def _read_json_object(path: str | Path) -> Mapping[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PermissionError("typed portable JSON artifact is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise PermissionError("typed portable JSON artifact is malformed")
    return payload


def _read_event_rows(path: str | Path) -> list[Mapping[str, Any]]:
    resolved = Path(path).expanduser().resolve(strict=True)
    rows: list[Mapping[str, Any]] = []
    for line_number, raw in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PermissionError(
                f"typed portable event is unreadable at line {line_number}"
            ) from exc
        if not isinstance(row, Mapping):
            raise PermissionError(
                f"typed portable event is malformed at line {line_number}"
            )
        rows.append(row)
    if not rows:
        raise PermissionError("typed portable event ledger is empty")
    return rows


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
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
            "Run or exactly replay the preregistered offline typed-portable "
            "production integration diagnostic."
        )
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-run-root", type=Path)
    parser.add_argument("--source-train-receipt", type=Path)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--result-receipt", type=Path)
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--verify-result-receipt", action="store_true")
    parser.add_argument("--print-implementation-binding", action="store_true")
    args = parser.parse_args(argv)

    if args.print_implementation_binding:
        print(
            json.dumps(
                build_implementation_file_binding(args.preregistration),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.verify_result_receipt:
        if args.result_receipt is None:
            parser.error(
                "--verify-result-receipt requires --result-receipt"
            )
        receipt = verify_typed_portable_integration_result_receipt(
            preregistration_path=args.preregistration,
            result_receipt_path=args.result_receipt,
        )
        print(
            json.dumps(
                {
                    "decision_hash": receipt["decision_hash"],
                    "integration_passed": receipt["integration_passed"],
                    "exact_replay_verified": receipt[
                        "exact_replay_verified"
                    ],
                },
                sort_keys=True,
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
        report = verify_existing_typed_portable_integration(**kwargs)
    else:
        report = run_typed_portable_integration(**kwargs)
    print(
        json.dumps(
            {
                "decision_hash": report["decision_hash"],
                "integration_passed": report["integration_passed"],
                "integration_reuse_verified": report.get(
                    "integration_reuse_verified",
                    False,
                ),
                "model_call_count": report["model_call_count"],
                "task_backend_run_task_call_count": report[
                    "task_backend_run_task_call_count"
                ],
                "evaluator_call_count": report["evaluator_call_count"],
            },
            sort_keys=True,
        )
    )
    if not report["integration_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
