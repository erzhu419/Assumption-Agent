from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..events import Event, EventSink, JsonlEventSink, MemoryEventSink, NullEventSink
from ..evolution import (
    PROPOSAL_FORMATION_POLICY_V2,
    _family_slot_event_row,
    _rank_family_proposal_slots,
)
from ..models import HypothesisProgram, stable_hash
from ..splits import SplitManifest
from ..typed_operator_grammar import (
    ALLOWED_OPERATOR_KINDS,
    CAUSAL_ACTION_SPAN_EVIDENCE_VERSION,
    FORBIDDEN_OPERATOR_KINDS,
    MAX_REGISTERED_ARTIFACTS_PER_FAMILY,
    MAX_TRACE_ACTION_SPANS,
    TYPED_OPERATOR_GRAMMAR_VERSION,
    TYPED_OPERATOR_LOWERING_VERSION,
    TYPED_RECIPE_SELECTION_VERSION,
    PrimitiveAssessment,
    SpanOutcome,
    TypedGraphUnavailableError,
    TrialTraceEvidence,
    assess_observed_primitives,
    build_family_capability_graph,
    canonical_recipe,
    extract_trial_trace_evidence,
    materialize_recipe_selection,
    selection_schema,
)
from ..validation import backend_action_contract_issues
from .skilllearn_compiler import (
    SKILLLEARN_ALLOWED_ACTION_OPERATIONS,
    _lower_skilllearn_program,
)
from .train_proposal_diagnostic import (
    _read_json_object,
    _sha256_file,
    _source_train_artifact_paths,
    reconstruct_v315_train_evidence,
)


TYPED_OPERATOR_FEASIBILITY_VERSION = (
    "v315_train_closed_typed_operator_feasibility_v1"
)
REQUIRED_SLOT_COUNT = 3
MINIMUM_FAMILY_SUPPORT = 2

_ACCEPTANCE_PREDICATES: Mapping[str, str] = {
    "source_provenance_passed": (
        "manifest and V3.15 TRAIN receipt hashes match preregistration and "
        "reconstruction yields exactly 38 TRAIN observations, 32 failures and "
        "6 success controls"
    ),
    "v317_target_plan_replay_passed": (
        "the frozen V3.17 production family-slot algorithm reconstructs the "
        "preregistered slot-plan hash and ordered three target-family hashes"
    ),
    "chronological_span_evidence_passed": (
        "38 complete receipt-bound TRAIN traces yield exactly 429 chronological "
        "allowlisted command occurrences, 70 failed occurrences, 208 explicitly "
        "discarded commands, 25 trials with a failed span and 4 successful trials "
        "with a failed span; no allowlisted occurrence is deduplicated or truncated "
        "and every item.started count/identity matches its self-hashed receipt and "
        "stays within the frozen 100-action budget"
    ),
    "failure_causality_separation_passed": (
        "at least one failed allowlisted span has a later scope-matched recovery, "
        "at least one successful trial contains a failed span, no failed span is "
        "the last allowlisted span, and observational inadmissibility count is zero"
    ),
    "typed_graph_coverage_passed": (
        "the three frozen target families each have a valid closed graph with at "
        "least one TRAIN-supported artifact and recipe and minimum support two"
    ),
    "closed_selection_surface_passed": (
        "model output is exactly one registered recipe_id; raw primitive values "
        "and artifact locators are absent; nine unknown/extra/type tamper probes "
        "fail closed; allowed and forbidden operator sets are disjoint"
    ),
    "prompt_lowering_compatibility_passed": (
        "three canonical registered recipes materialize valid harness-owned "
        "HypothesisPrograms and lower through the existing prompt-directive "
        "compiler with zero contract issues; no capability implementation or "
        "restricted runtime executor is claimed"
    ),
    "deterministic_replay_passed": (
        "a second in-process graph build has identical ordered graph hashes while "
        "the three graph and model-catalog identities remain distinct"
    ),
    "offline_boundary_contract_passed": (
        "the bounded event ledger before decision completion contains only one "
        "start, one TRAIN span reconstruction, three graph outcome events and "
        "three materialization outcome events; stored offline TRAIN outcomes, "
        "local contract checks and unit-test source hashing are declared, while "
        "no live model/task-backend/evaluator invocation, validation/test/sealed "
        "split access, verifier-content access or promotion evaluation occurs"
    ),
}
_ACCEPTANCE_KEYS = frozenset(_ACCEPTANCE_PREDICATES)

_BOUNDARY_FLAGS: Mapping[str, bool] = {
    "source_agent_trials_reexecuted": False,
    "stored_offline_train_outcomes_used": True,
    "local_contract_validation_used": True,
    "unit_test_source_hashed": True,
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


class _BoundaryAuditEventSink:
    def __init__(self, delegate: EventSink) -> None:
        self.delegate = delegate
        self.events: list[dict[str, Any]] = []

    def emit(self, event: Event) -> None:
        row = event.to_dict()
        self.events.append(row)
        self.delegate.emit(event)


def run_typed_operator_feasibility(
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
            "formal typed feasibility must use preregistered canonical paths"
        )
    if report.exists() or events.exists():
        raise FileExistsError("formal typed feasibility output already exists")
    _reserve_decision_lock(
        canonical["decision_lock"],
        preregistration_hash=stable_hash(preregistration),
    )
    try:
        result = _compute_feasibility(
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
                "lock_version": "typed_operator_feasibility_decision_lock_v1",
                "decision_ordinal": 1,
                "state": "completed",
                "preregistration_hash": stable_hash(preregistration),
                "decision_hash": result["decision_hash"],
                "report_hash": result["report_hash"],
                "feasibility_passed": result["feasibility_passed"],
                "raw_content_persisted": False,
            },
        )
        return result
    except Exception:
        # The exclusive lock deliberately remains reserved.  A crash or failed
        # decision cannot be retried under a fresh output path.
        raise


def verify_existing_typed_operator_feasibility(
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
    if Path(report_path).expanduser().resolve(strict=True) != canonical["report"]:
        raise PermissionError("typed feasibility report path is not canonical")
    if Path(events_path).expanduser().resolve(strict=True) != canonical["events"]:
        raise PermissionError("typed feasibility events path is not canonical")
    declared = _read_json_object(
        Path(report_path).expanduser().resolve(strict=True),
        label="typed operator feasibility report",
    )
    memory = MemoryEventSink()
    expected = _compute_feasibility(
        root=root,
        manifest_path=manifest_path,
        source_run_root=source_run_root,
        source_train_receipt=source_train_receipt,
        preregistration_path=preregistration_path,
        event_sink=memory,
    )
    if dict(declared) != expected:
        raise PermissionError(
            "typed operator feasibility report does not replay exactly"
        )
    persisted_events = _read_event_rows(
        Path(events_path).expanduser().resolve(strict=True)
    )
    if persisted_events != memory.events:
        raise PermissionError(
            "typed operator feasibility event ledger does not replay exactly"
        )
    decision_lock = _read_json_object(
        canonical["decision_lock"].resolve(strict=True),
        label="typed operator feasibility decision lock",
    )
    _verify_decision_lock(
        decision_lock,
        report=expected,
        preregistration_hash=stable_hash(preregistration),
    )
    return {**expected, "feasibility_reuse_verified": True}


def _compute_feasibility(
    *,
    root: str | Path,
    manifest_path: str | Path,
    source_run_root: str | Path,
    source_train_receipt: str | Path,
    preregistration_path: str | Path,
    event_sink: EventSink,
) -> dict[str, Any]:
    audit_sink = _BoundaryAuditEventSink(event_sink)
    manifest = SplitManifest.read(manifest_path)
    preregistration = _read_preregistration(preregistration_path)
    preregistration_hash = stable_hash(preregistration)
    implementation_file_set_hash = _implementation_file_set_hash(
        preregistration,
        preregistration_path=preregistration_path,
    )
    trace_id = stable_hash(
        {
            "policy": TYPED_OPERATOR_FEASIBILITY_VERSION,
            "manifest_hash": manifest.manifest_hash,
            "preregistration_hash": preregistration_hash,
            "implementation_file_set_hash": implementation_file_set_hash,
        }
    )[:24]
    _emit(
        audit_sink,
        event="typed_operator_feasibility_started",
        trace_id=trace_id,
        payload={
            "feasibility_policy": TYPED_OPERATOR_FEASIBILITY_VERSION,
            "preregistration_hash": preregistration_hash,
            "manifest_hash": manifest.manifest_hash,
            "decision_budget": preregistration["decision_budget"],
            "offline_only": True,
        },
    )

    evidence = reconstruct_v315_train_evidence(
        root=root,
        manifest=manifest,
        source_run_root=source_run_root,
        source_train_receipt=source_train_receipt,
        event_sink=NullEventSink(),
    )
    _validate_source_binding(
        preregistration,
        manifest=manifest,
        source_train_receipt_hash=evidence.source_train_receipt_hash,
    )
    trials = _extract_all_train_trials(
        evidence=evidence,
        source_run_root=source_run_root,
    )
    trial_evidence_hash = stable_hash(
        {"trial_evidence_hashes": [row.evidence_hash for row in trials]}
    )
    span_rows = tuple(span for trial in trials for span in trial.spans)
    assessments = assess_observed_primitives(trials)
    assessment_hash = stable_hash(
        {"assessments": [row.safe_payload() for row in assessments]}
    )
    span_summary = _span_summary(trials, assessments)
    _emit(
        audit_sink,
        event="typed_action_span_evidence_reconstructed",
        trace_id=trace_id,
        payload={
            "evidence_policy": CAUSAL_ACTION_SPAN_EVIDENCE_VERSION,
            "train_trial_count": len(trials),
            "complete_trace_count": sum(row.trace_complete for row in trials),
            "command_span_count": len(span_rows),
            "failed_command_span_count": span_summary[
                "failed_command_span_count"
            ],
            "recovered_failed_span_count": span_summary[
                "recovered_failed_span_count"
            ],
            "observationally_inadmissible_primitive_count": 0,
            "trial_evidence_hash": trial_evidence_hash,
            "primitive_assessment_hash": assessment_hash,
            "allowlisted_span_chronology_preserved": True,
            "allowlisted_command_occurrences_deduplicated_or_truncated": False,
            "full_raw_command_coverage_claimed": False,
        },
    )

    expected_target_hashes = preregistration["expected_target_family_hashes"]
    v317_ranked = _rank_family_proposal_slots(
        evidence.residuals,
        profiles=evidence.action_profiles,
        family_use_counts={},
    )
    v317_selected = tuple(
        row for row in v317_ranked if row.recommended_artifact is not None
    )[:REQUIRED_SLOT_COUNT]
    v317_plan_rows = [
        _family_slot_event_row(slot, slot_index=index)
        for index, slot in enumerate(v317_selected, start=1)
    ]
    reconstructed_v317_slot_plan_hash = stable_hash(
        {
            "policy": PROPOSAL_FORMATION_POLICY_V2,
            "slots": v317_plan_rows,
        }
    )
    reconstructed_v317_target_hashes = [
        stable_hash({"family": row.target_family}) for row in v317_selected
    ]
    family_by_hash = {
        stable_hash({"family": family}): family
        for family in {row.family for row in evidence.failures}
    }
    if any(value not in family_by_hash for value in expected_target_hashes):
        raise PermissionError(
            "preregistered V3.17 target family is absent from TRAIN failures"
        )
    selected_families = tuple(
        family_by_hash[value] for value in expected_target_hashes
    )
    trial_by_id = {row.trial_id_hash: row for row in trials}
    graph_list: list[Any] = []
    graph_rows: list[dict[str, Any]] = []
    replay_outcome_matches: list[bool] = []
    for family, target_family_hash in zip(
        selected_families,
        expected_target_hashes,
    ):
        graph = None
        graph_error_code: str | None = None
        try:
            graph = build_family_capability_graph(
                target_family=family,
                failures=evidence.failures,
                action_profiles=evidence.action_profiles,
                trial_evidence=trial_by_id,
                minimum_support=MINIMUM_FAMILY_SUPPORT,
                maximum_artifacts=MAX_REGISTERED_ARTIFACTS_PER_FAMILY,
            )
        except TypedGraphUnavailableError as exc:
            graph_error_code = exc.reason_code
        replay_graph = None
        replay_error_code: str | None = None
        try:
            replay_graph = build_family_capability_graph(
                target_family=family,
                failures=evidence.failures,
                action_profiles=evidence.action_profiles,
                trial_evidence=trial_by_id,
                minimum_support=MINIMUM_FAMILY_SUPPORT,
                maximum_artifacts=MAX_REGISTERED_ARTIFACTS_PER_FAMILY,
            )
        except TypedGraphUnavailableError as exc:
            replay_error_code = exc.reason_code
        primary_outcome = (
            {"graph_hash": graph.graph_hash}
            if graph is not None
            else {"unavailable_code": graph_error_code}
        )
        replay_outcome = (
            {"graph_hash": replay_graph.graph_hash}
            if replay_graph is not None
            else {"unavailable_code": replay_error_code}
        )
        replay_outcome_matches.append(primary_outcome == replay_outcome)
        if graph is None:
            row = {
                "target_family_hash": target_family_hash,
                "graph_available": False,
                "availability_error_code": graph_error_code,
                "availability_error_hash": stable_hash(
                    {"error_code": graph_error_code}
                ),
                "graph_hash": None,
                "source_evidence_hash": None,
                "artifact_count": 0,
                "artifact_set_hash": None,
                "capability_count": 0,
                "capability_set_hash": None,
                "recipe_count": 0,
                "recipe_set_hash": None,
                "model_catalog_hash": None,
                "graph_validation_issues": [graph_error_code],
                "raw_content_persisted": False,
            }
            event_name = "typed_capability_snapshot_unavailable"
        else:
            graph_list.append(graph)
            row = {
                "target_family_hash": graph.target_family_hash,
                "graph_available": True,
                "availability_error_code": None,
                "availability_error_hash": None,
                "graph_hash": graph.graph_hash,
                "source_evidence_hash": graph.source_evidence_hash,
                "artifact_count": len(graph.artifacts),
                "artifact_set_hash": stable_hash(
                    {
                        "artifacts": [
                            artifact.safe_payload()
                            for artifact in graph.artifacts
                        ]
                    }
                ),
                "capability_count": len(graph.capabilities),
                "capability_set_hash": stable_hash(
                    {
                        "capabilities": [
                            capability.payload()
                            for capability in graph.capabilities
                        ]
                    }
                ),
                "recipe_count": len(graph.recipes),
                "recipe_set_hash": stable_hash(
                    {
                        "recipes": [
                            recipe.payload() for recipe in graph.recipes
                        ]
                    }
                ),
                "model_catalog_hash": stable_hash(graph.model_catalog()),
                "graph_validation_issues": list(graph.validate()),
                "raw_content_persisted": False,
            }
            event_name = "typed_capability_snapshot_created"
        graph_rows.append(row)
        _emit(
            audit_sink,
            event=event_name,
            trace_id=trace_id,
            payload={
                "grammar_version": TYPED_OPERATOR_GRAMMAR_VERSION,
                "selection_contract": TYPED_RECIPE_SELECTION_VERSION,
                "target_family_hash": row["target_family_hash"],
                "graph_available": row["graph_available"],
                "availability_error_hash": row["availability_error_hash"],
                "graph_hash": row["graph_hash"],
                "source_evidence_hash": row["source_evidence_hash"],
                "artifact_count": row["artifact_count"],
                "capability_count": row["capability_count"],
                "recipe_count": row["recipe_count"],
                "model_catalog_hash": row["model_catalog_hash"],
                "model_authored_primitive_count": 0,
                "unknown_ref_count": 0,
            },
        )

    graphs = tuple(graph_list)
    graph_hashes = [row.graph_hash for row in graphs]
    target_family_hashes = [row.target_family_hash for row in graphs]
    model_catalog_hashes = [
        stable_hash(row.model_catalog()) for row in graphs
    ]
    snapshot_commitments = {
        str(row["target_family_hash"]): {
            "graph_hash": row["graph_hash"],
            "model_catalog_hash": row["model_catalog_hash"],
            "availability_error_hash": row["availability_error_hash"],
        }
        for row in graph_rows
    }

    primitive_values = _all_primitive_values(trials)
    raw_disclosure_count = sum(
        value in _leaf_strings(graph.model_catalog())
        for graph in graphs
        for value in primitive_values
    )
    locator_disclosure_count = sum(
        artifact.locator in _leaf_strings(graph.model_catalog())
        for graph in graphs
        for artifact in graph.artifacts
    )
    tamper_rejection_count = sum(
        _selection_tamper_rejected(graph) for graph in graphs
    )
    programs: list[HypothesisProgram] = []
    lowered_hashes: list[str] = []
    materialization_issue_count = 0
    materialization_failure_count = 0
    materialization_rows: list[dict[str, Any]] = []
    graph_by_target = {row.target_family_hash: row for row in graphs}
    graph_attempt_by_target = {
        str(row["target_family_hash"]): row for row in graph_rows
    }
    for target_family_hash in expected_target_hashes:
        graph = graph_by_target.get(target_family_hash)
        snapshot = snapshot_commitments[target_family_hash]
        if graph is None:
            materialization_failure_count += 1
            materialization_issue_count += 1
            availability = graph_attempt_by_target[target_family_hash]
            row = {
                "target_family_hash": target_family_hash,
                "materialized": False,
                "skip_reason": "graph_unavailable",
                "error_type_hash": availability[
                    "availability_error_hash"
                ],
                "snapshot_graph_hash": snapshot["graph_hash"],
                "snapshot_model_catalog_hash": snapshot[
                    "model_catalog_hash"
                ],
                "snapshot_availability_error_hash": snapshot[
                    "availability_error_hash"
                ],
                "graph_hash": None,
                "program_hash": None,
                "lowered_action_hash": None,
            }
            materialization_rows.append(row)
            _emit(
                audit_sink,
                event="typed_action_graph_materialization_skipped",
                trace_id=trace_id,
                payload={
                    **row,
                    "model_authored_primitive_count": 0,
                    "unknown_ref_count": 0,
                    "typed_graph_valid": False,
                },
            )
            continue
        try:
            recipe = canonical_recipe(graph)
            program = materialize_recipe_selection(
                {"recipe_id": recipe.recipe_id},
                graph=graph,
                evaluator_epoch="typed-operator-feasibility-v1",
                expected_graph_hash=str(snapshot["graph_hash"]),
                expected_model_catalog_hash=str(
                    snapshot["model_catalog_hash"]
                ),
            )
            contract_issues = backend_action_contract_issues(
                program,
                allowed_operations=SKILLLEARN_ALLOWED_ACTION_OPERATIONS,
                external_evidence_is_hidden=True,
            )
            program_issues = program.validate()
            lowered = _lower_skilllearn_program(program)
        except (PermissionError, ValueError) as exc:
            materialization_failure_count += 1
            materialization_issue_count += 1
            error_type = type(exc).__name__
            row = {
                "target_family_hash": target_family_hash,
                "materialized": False,
                "skip_reason": "materialization_contract_failure",
                "error_type_hash": stable_hash(
                    {"error_type": error_type}
                ),
                "snapshot_graph_hash": snapshot["graph_hash"],
                "snapshot_model_catalog_hash": snapshot[
                    "model_catalog_hash"
                ],
                "snapshot_availability_error_hash": snapshot[
                    "availability_error_hash"
                ],
                "graph_hash": graph.graph_hash,
                "program_hash": None,
                "lowered_action_hash": None,
            }
            materialization_rows.append(row)
            _emit(
                audit_sink,
                event="typed_action_graph_materialization_skipped",
                trace_id=trace_id,
                payload={
                    **row,
                    "model_authored_primitive_count": 0,
                    "unknown_ref_count": 0,
                    "typed_graph_valid": False,
                },
            )
            continue
        programs.append(program)
        issue_count = len(program_issues) + len(contract_issues)
        materialization_issue_count += issue_count
        lowered_hash = stable_hash(
            {"actions": [row.to_dict() for row in lowered]}
        )
        lowered_hashes.append(lowered_hash)
        row = {
            "target_family_hash": target_family_hash,
            "materialized": True,
            "skip_reason": None,
            "error_type_hash": None,
            "snapshot_graph_hash": snapshot["graph_hash"],
            "snapshot_model_catalog_hash": snapshot[
                "model_catalog_hash"
            ],
            "snapshot_availability_error_hash": snapshot[
                "availability_error_hash"
            ],
            "graph_hash": graph.graph_hash,
            "program_hash": program.payload_hash,
            "lowered_action_hash": lowered_hash,
        }
        materialization_rows.append(row)
        _emit(
            audit_sink,
            event="typed_action_graph_materialized",
            trace_id=trace_id,
            payload={
                **row,
                "selection_graph_hash": stable_hash(
                    selection_schema(graph)
                ),
                "selected_recipe_hash": stable_hash(
                    {"recipe_id": recipe.recipe_id}
                ),
                "model_authored_primitive_count": 0,
                "unknown_ref_count": 0,
                "typed_graph_valid": issue_count == 0,
            },
        )

    expected_counts = preregistration["expected_span_evidence"]
    source_provenance_passed = bool(
        evidence.source_train_receipt_hash
        == preregistration["source_train_receipt_hash"]
        and manifest.manifest_hash == preregistration["manifest_hash"]
        and len(evidence.residuals) == 38
        and len(evidence.failures) == 32
        and len(evidence.success_controls) == 6
    )
    v317_target_plan_replay_passed = bool(
        len(v317_selected) == REQUIRED_SLOT_COUNT
        and reconstructed_v317_slot_plan_hash
        == preregistration["expected_v317_slot_plan_hash"]
        and reconstructed_v317_target_hashes == expected_target_hashes
    )
    chronological_span_evidence_passed = bool(
        len(trials) == expected_counts["train_trial_count"]
        and sum(row.trace_complete for row in trials)
        == expected_counts["complete_trace_count"]
        and len(span_rows) == expected_counts["command_span_count"]
        and span_summary["failed_command_span_count"]
        == expected_counts["failed_command_span_count"]
        and span_summary["trials_with_failed_span_count"]
        == expected_counts["trials_with_failed_span_count"]
        and span_summary["successful_trials_with_failed_span_count"]
        == expected_counts["successful_trials_with_failed_span_count"]
        and span_summary["discarded_command_count"]
        == expected_counts["discarded_command_count"]
        and all(
            row.action_start_count <= row.action_budget_limit
            for row in trials
        )
    )
    failure_causality_separation_passed = bool(
        span_summary["recovered_failed_span_count"] > 0
        and span_summary["successful_trials_with_failed_span_count"] > 0
        and span_summary["last_allowlisted_failed_span_count"] == 0
        and all(not row.observationally_inadmissible for row in assessments)
    )
    typed_graph_coverage_passed = bool(
        len(graphs) == REQUIRED_SLOT_COUNT
        and target_family_hashes == expected_target_hashes
        and all(not graph.validate() for graph in graphs)
        and all(graph.artifacts and graph.recipes for graph in graphs)
        and all(
            artifact.support_count >= MINIMUM_FAMILY_SUPPORT
            for graph in graphs
            for artifact in graph.artifacts
        )
    )
    closed_selection_surface_passed = bool(
        len(graphs) == REQUIRED_SLOT_COUNT
        and raw_disclosure_count == 0
        and locator_disclosure_count == 0
        and tamper_rejection_count == 3 * REQUIRED_SLOT_COUNT
        and ALLOWED_OPERATOR_KINDS.isdisjoint(FORBIDDEN_OPERATOR_KINDS)
        and all(
            schema == selection_schema(graph)
            and schema.get("additionalProperties") is False
            and schema.get("required") == ["recipe_id"]
            for graph, schema in (
                (graph, selection_schema(graph)) for graph in graphs
            )
        )
    )
    prompt_lowering_compatibility_passed = bool(
        len(programs) == REQUIRED_SLOT_COUNT
        and materialization_failure_count == 0
        and materialization_issue_count == 0
        and all(program.action_graph for program in programs)
        and all(
            program.trigger.all_of
            and program.trigger.all_of[0].key == "family"
            and program.trigger.all_of[0].op == "eq"
            for program in programs
        )
    )
    deterministic_replay_passed = bool(
        len(graphs) == REQUIRED_SLOT_COUNT
        and all(replay_outcome_matches)
        and len(set(graph_hashes)) == len(graph_hashes)
        and len(set(model_catalog_hashes)) == len(model_catalog_hashes)
    )
    offline_boundary_issues = _offline_boundary_contract_issues(
        audit_sink.events
    )
    offline_boundary_contract_passed = not offline_boundary_issues
    acceptance = {
        "source_provenance_passed": source_provenance_passed,
        "v317_target_plan_replay_passed": v317_target_plan_replay_passed,
        "chronological_span_evidence_passed": (
            chronological_span_evidence_passed
        ),
        "failure_causality_separation_passed": (
            failure_causality_separation_passed
        ),
        "typed_graph_coverage_passed": typed_graph_coverage_passed,
        "closed_selection_surface_passed": closed_selection_surface_passed,
        "prompt_lowering_compatibility_passed": (
            prompt_lowering_compatibility_passed
        ),
        "deterministic_replay_passed": deterministic_replay_passed,
        "offline_boundary_contract_passed": offline_boundary_contract_passed,
    }
    if set(acceptance) != _ACCEPTANCE_KEYS:
        raise RuntimeError("typed feasibility acceptance contract drifted")
    feasibility_passed = all(acceptance.values())
    decision_evidence = {
        "preregistration_hash": preregistration_hash,
        "implementation_file_set_hash": implementation_file_set_hash,
        "source_train_receipt_hash": evidence.source_train_receipt_hash,
        "trial_evidence_hash": trial_evidence_hash,
        "primitive_assessment_hash": assessment_hash,
        "reconstructed_v317_slot_plan_hash": (
            reconstructed_v317_slot_plan_hash
        ),
        "graph_outcomes": [
            {
                "target_family_hash": row["target_family_hash"],
                "graph_hash": row["graph_hash"],
                "availability_error_hash": row[
                    "availability_error_hash"
                ],
            }
            for row in graph_rows
        ],
        "program_hashes": [row.payload_hash for row in programs],
        "lowered_action_hashes": lowered_hashes,
        "materialization_outcomes": materialization_rows,
        "acceptance": acceptance,
        "feasibility_passed": feasibility_passed,
    }
    decision_hash = stable_hash(decision_evidence)
    report: dict[str, Any] = {
        "feasibility_policy": TYPED_OPERATOR_FEASIBILITY_VERSION,
        "feasibility_only": True,
        "feasibility_passed": feasibility_passed,
        "decision_hash": decision_hash,
        "preregistration": {
            "preregistration_hash": preregistration_hash,
            "implementation_file_set_hash": implementation_file_set_hash,
            "decision_budget": preregistration["decision_budget"],
            "decision_ordinal": 1,
            "acceptance_contract_hash": stable_hash(
                preregistration["acceptance"]
            ),
            "acceptance_predicate_hash": stable_hash(
                preregistration["acceptance_predicates"]
            ),
            "no_acceptance_adaptation_after_decision": True,
        },
        "source": {
            "manifest_hash": manifest.manifest_hash,
            "source_train_receipt_hash": evidence.source_train_receipt_hash,
            "source_protocol_hash": evidence.source_protocol_hash,
            "source_protocol_lock_hash": evidence.source_protocol_lock_hash,
            "source_observation_set_hash": evidence.source_observation_set_hash,
            "source_row_set_hash": evidence.source_row_set_hash,
            "source_checkpoint_hash": evidence.source_checkpoint_hash,
            "train_observation_count": len(evidence.residuals),
            "train_failure_count": len(evidence.failures),
            "train_success_control_count": len(evidence.success_controls),
            "source_agent_trials_reexecuted": 0,
        },
        "historical_v317_target_plan": {
            "proposal_formation_policy": PROPOSAL_FORMATION_POLICY_V2,
            "expected_slot_plan_hash": preregistration[
                "expected_v317_slot_plan_hash"
            ],
            "reconstructed_slot_plan_hash": (
                reconstructed_v317_slot_plan_hash
            ),
            "reconstructed_target_family_hashes": (
                reconstructed_v317_target_hashes
            ),
            "target_plan_replay_passed": v317_target_plan_replay_passed,
            "used_only_to_bind_frozen_target_families": True,
            "used_for_typed_primitive_admissibility": False,
            "raw_content_persisted": False,
        },
        "action_span_evidence": {
            "policy": CAUSAL_ACTION_SPAN_EVIDENCE_VERSION,
            "trial_evidence_hash": trial_evidence_hash,
            "primitive_assessment_hash": assessment_hash,
            **span_summary,
            "assessment_count": len(assessments),
            "assessment_class_counts": dict(
                sorted(
                    Counter(
                        row.classification.value for row in assessments
                    ).items()
                )
            ),
            "observationally_inadmissible_primitive_count": 0,
            "allowlisted_span_chronology_preserved": True,
            "allowlisted_command_occurrences_deduplicated_or_truncated": False,
            "signature_task_path_limit": 12,
            "full_raw_command_coverage_claimed": False,
            "failed_span_implies_primitive_inadmissibility": False,
            "raw_content_persisted": False,
        },
        "typed_operator_graph": {
            "grammar_version": TYPED_OPERATOR_GRAMMAR_VERSION,
            "selection_contract": TYPED_RECIPE_SELECTION_VERSION,
            "lowering_version": TYPED_OPERATOR_LOWERING_VERSION,
            "legacy_v317_failed_primitive_filter_used_for_admissibility": False,
            "graph_attempt_count": len(graph_rows),
            "graph_count": len(graphs),
            "graph_unavailable_count": sum(
                not row["graph_available"] for row in graph_rows
            ),
            "graph_set_hash": stable_hash(
                {
                    "outcomes": [
                        {
                            "target_family_hash": row[
                                "target_family_hash"
                            ],
                            "graph_hash": row["graph_hash"],
                            "availability_error_hash": row[
                                "availability_error_hash"
                            ],
                        }
                        for row in graph_rows
                    ]
                }
            ),
            "expected_target_family_hashes": expected_target_hashes,
            "available_target_family_hashes": target_family_hashes,
            "graphs": graph_rows,
            "model_catalog_set_hash": stable_hash(
                {"catalog_hashes": model_catalog_hashes}
            ),
            "raw_primitive_value_disclosure_count": raw_disclosure_count,
            "raw_artifact_locator_disclosure_count": (
                locator_disclosure_count
            ),
            "model_authored_primitive_count": 0,
            "unknown_ref_count": 0,
            "tamper_rejection_count": tamper_rejection_count,
            "allowed_operator_set_hash": stable_hash(
                {"operators": sorted(ALLOWED_OPERATOR_KINDS)}
            ),
            "forbidden_operator_set_hash": stable_hash(
                {"operators": sorted(FORBIDDEN_OPERATOR_KINDS)}
            ),
            "allowed_and_forbidden_disjoint": (
                ALLOWED_OPERATOR_KINDS.isdisjoint(FORBIDDEN_OPERATOR_KINDS)
            ),
            "raw_content_persisted": False,
        },
        "materialization": {
            "attempt_count": len(materialization_rows),
            "program_count": len(programs),
            "program_set_hash": stable_hash(
                {"program_hashes": [row.payload_hash for row in programs]}
            ),
            "lowered_action_set_hash": stable_hash(
                {"lowered_action_hashes": lowered_hashes}
            ),
            "materialization_failure_count": materialization_failure_count,
            "materialization_issue_count": materialization_issue_count,
            "attempts": materialization_rows,
            "statement_owned_by_harness": True,
            "trigger_owned_by_harness": True,
            "action_text_owned_by_harness": True,
            "capability_requirement_derived_from_artifact_format": True,
            "capability_implementation_verified": False,
            "runtime_agent_still_receives_prompt_directives": True,
            "restricted_runtime_executor_claimed": False,
            "raw_content_persisted": False,
        },
        "offline_boundary_contract": {
            "predecision_event_count": len(audit_sink.events),
            "predecision_event_set_hash": stable_hash(
                {"events": audit_sink.events}
            ),
            "issues": list(offline_boundary_issues),
            "ledger_scope": "declared_operation_boundary_and_event_shape",
            "network_isolation_proved_by_ledger": False,
            "stored_offline_train_outcomes_used": True,
            "local_contract_validation_used": True,
            "unit_test_source_hashed": True,
            "live_model_invoked": False,
            "live_task_backend_invoked": False,
            "live_evaluator_invoked": False,
            "validation_split_accessed": False,
            "test_split_accessed": False,
            "sealed_split_accessed": False,
            "verifier_content_accessed": False,
            "raw_content_persisted": False,
        },
        "acceptance": acceptance,
        "selection_integration_diagnostic_freeze_eligible_if_passed": (
            feasibility_passed
        ),
        "typed_production_integration_currently_present": False,
        "development_protocol_freeze_currently_authorized": False,
        "development_task_execution_currently_authorized": False,
        "development_requires_frozen_typed_selection_integration": True,
        "promotion_gate_or_score": False,
        "failure_blocks_future_trial_spend_only": True,
        "model_call_count": 0,
        "backend_call_count": 0,
        "evaluator_call_count": 0,
        "validation_task_count": 0,
        **_BOUNDARY_FLAGS,
    }
    report["report_hash"] = stable_hash(report)
    _emit(
        audit_sink,
        event="typed_operator_feasibility_completed",
        trace_id=trace_id,
        payload={
            "feasibility_policy": TYPED_OPERATOR_FEASIBILITY_VERSION,
            "decision_hash": decision_hash,
            "report_hash": report["report_hash"],
            "feasibility_passed": feasibility_passed,
            "acceptance_hash": stable_hash(acceptance),
            "graph_set_hash": report["typed_operator_graph"][
                "graph_set_hash"
            ],
            "trial_evidence_hash": trial_evidence_hash,
            "primitive_assessment_hash": assessment_hash,
            "promotion_gate_or_score": False,
        },
    )
    return report


def _extract_all_train_trials(
    *,
    evidence: Any,
    source_run_root: str | Path,
) -> tuple[TrialTraceEvidence, ...]:
    source_root = Path(source_run_root).expanduser().resolve(strict=True)
    upstream = (
        source_root
        / "development_recursive"
        / "upstream_trials"
        / "no_skill"
    ).resolve(strict=True)
    rows: list[TrialTraceEvidence] = []
    for residual in evidence.residuals:
        _, trace_path, receipt_path = _source_train_artifact_paths(
            upstream,
            family=residual.family,
            item_id=residual.task_id,
        )
        receipt = _read_json_object(receipt_path, label="action budget receipt")
        trace_hash = _sha256_file(trace_path)
        if receipt.get("trace_sha256") != trace_hash:
            raise PermissionError("action-span trace hash drifted from receipt")
        receipt_hash = str(receipt.get("receipt_hash") or "")
        receipt_without_hash = dict(receipt)
        receipt_without_hash.pop("receipt_hash", None)
        if (
            not _is_sha256_text(receipt_hash)
            or stable_hash(receipt_without_hash) != receipt_hash
        ):
            raise PermissionError("action-span receipt hash is invalid")
        observed_steps = receipt.get("observed_steps")
        action_event_hash = receipt.get("action_event_hash")
        if (
            not isinstance(observed_steps, int)
            or isinstance(observed_steps, bool)
            or not _is_sha256_text(action_event_hash)
        ):
            raise PermissionError("action-span receipt binding is malformed")
        rows.append(
            extract_trial_trace_evidence(
                trace_path,
                containment_root=upstream,
                trial_id_hash=stable_hash({"item_id": residual.task_id}),
                family_hash=stable_hash({"family": residual.family}),
                trace_hash=trace_hash,
                action_budget_receipt_hash=receipt_hash,
                expected_action_start_count=observed_steps,
                expected_action_event_hash=action_event_hash,
                baseline_success=residual.baseline_success,
                expected_action_budget_limit=MAX_TRACE_ACTION_SPANS,
            )
        )
    return tuple(rows)


def _span_summary(
    trials: Sequence[TrialTraceEvidence],
    assessments: Sequence[PrimitiveAssessment],
) -> dict[str, Any]:
    spans = [span for trial in trials for span in trial.spans]
    failed = [span for span in spans if span.outcome is SpanOutcome.FAILED]
    return {
        "train_trial_count": len(trials),
        "complete_trace_count": sum(row.trace_complete for row in trials),
        "command_span_count": len(spans),
        "successful_command_span_count": sum(
            span.outcome is SpanOutcome.SUCCEEDED for span in spans
        ),
        "failed_command_span_count": len(failed),
        "unknown_command_span_count": sum(
            span.outcome is SpanOutcome.UNKNOWN for span in spans
        ),
        "discarded_command_count": sum(
            row.discarded_command_count for row in trials
        ),
        "action_start_count": sum(
            row.action_start_count for row in trials
        ),
        "maximum_trial_action_start_count": max(
            (row.action_start_count for row in trials), default=0
        ),
        "trials_with_failed_span_count": sum(
            any(span.outcome is SpanOutcome.FAILED for span in row.spans)
            for row in trials
        ),
        "successful_trials_with_failed_span_count": sum(
            row.baseline_success
            and any(span.outcome is SpanOutcome.FAILED for span in row.spans)
            for row in trials
        ),
        "recovered_failed_span_count": sum(span.recovered for span in failed),
        "later_exact_success_count": sum(
            bool(span.later_exact_success_span_hashes) for span in failed
        ),
        "later_same_executable_success_count": sum(
            bool(span.later_same_executable_success_span_hashes)
            for span in failed
        ),
        "later_shared_artifact_success_count": sum(
            bool(span.later_shared_artifact_success_span_hashes)
            for span in failed
        ),
        "last_allowlisted_failed_span_count": sum(
            span.is_last_allowlisted_span for span in failed
        ),
        "exact_signature_do_not_recommend_count": sum(
            row.scope == "exact_command"
            and row.classification.value
            == "do_not_recommend_exact_signature"
            for row in assessments
        ),
        "observationally_inadmissible_primitive_count": 0,
        "span_set_hash": stable_hash(
            {"span_hashes": [span.span_hash for span in spans]}
        ),
    }


def _all_primitive_values(
    trials: Sequence[TrialTraceEvidence],
) -> frozenset[str]:
    values: set[str] = set()
    for trial in trials:
        for span in trial.spans:
            values.add(span.executable.value)
            values.update(row.value for row in span.flags)
            values.update(row.value for row in span.artifacts)
        values.update(row.value for row in trial.changed_artifacts)
    return frozenset(value for value in values if value)


def _leaf_strings(value: Any) -> frozenset[str]:
    if isinstance(value, str):
        return frozenset({value})
    if isinstance(value, Mapping):
        return frozenset(
            child
            for key, item in value.items()
            for child in (*_leaf_strings(key), *_leaf_strings(item))
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return frozenset(
            child for item in value for child in _leaf_strings(item)
        )
    return frozenset()


def _selection_tamper_rejected(graph: Any) -> int:
    rejected = 0
    probes = (
        {"recipe_id": "recipe_unknown"},
        {
            "recipe_id": graph.recipes[0].recipe_id,
            "command": "arbitrary shell text",
        },
        {"recipe_id": {"operator": "network_fetch"}},
    )
    for probe in probes:
        try:
            materialize_recipe_selection(
                probe,
                graph=graph,
                evaluator_epoch="typed-operator-feasibility-v1",
                expected_graph_hash=graph.graph_hash,
                expected_model_catalog_hash=stable_hash(
                    graph.model_catalog()
                ),
            )
        except PermissionError:
            rejected += 1
    return rejected


def _read_preregistration(path: str | Path) -> Mapping[str, Any]:
    resolved_path = Path(path).expanduser().resolve(strict=True)
    payload = _read_json_object(
        resolved_path,
        label="typed operator feasibility preregistration",
    )
    expected = {
        "feasibility_policy": TYPED_OPERATOR_FEASIBILITY_VERSION,
        "decision_budget": 1,
        "decision_scope": "offline_representation_feasibility_only",
        "evidence_policy": CAUSAL_ACTION_SPAN_EVIDENCE_VERSION,
        "grammar_version": TYPED_OPERATOR_GRAMMAR_VERSION,
        "selection_contract": TYPED_RECIPE_SELECTION_VERSION,
        "lowering_version": TYPED_OPERATOR_LOWERING_VERSION,
        "slot_count": REQUIRED_SLOT_COUNT,
        "minimum_family_support": MINIMUM_FAMILY_SUPPORT,
        "maximum_artifacts_per_family": MAX_REGISTERED_ARTIFACTS_PER_FAMILY,
        "maximum_trace_action_spans": MAX_TRACE_ACTION_SPANS,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise PermissionError(
                f"typed feasibility preregistration drifted: {key}"
            )
    if set(payload.get("acceptance", {})) != _ACCEPTANCE_KEYS or not all(
        value is True for value in payload["acceptance"].values()
    ):
        raise PermissionError("typed feasibility acceptance contract drifted")
    if payload.get("acceptance_predicates") != dict(_ACCEPTANCE_PREDICATES):
        raise PermissionError("typed feasibility acceptance predicates drifted")
    if payload.get("boundary_contract") != dict(_BOUNDARY_FLAGS):
        raise PermissionError("typed feasibility boundary contract drifted")
    if payload.get("allowed_operator_kinds") != sorted(ALLOWED_OPERATOR_KINDS):
        raise PermissionError("typed feasibility allowed operators drifted")
    if payload.get("forbidden_operator_kinds") != sorted(
        FORBIDDEN_OPERATOR_KINDS
    ):
        raise PermissionError("typed feasibility forbidden operators drifted")
    expected_counts = payload.get("expected_span_evidence")
    expected_count_keys = {
        "train_trial_count",
        "complete_trace_count",
        "command_span_count",
        "failed_command_span_count",
        "discarded_command_count",
        "trials_with_failed_span_count",
        "successful_trials_with_failed_span_count",
    }
    if (
        not isinstance(expected_counts, Mapping)
        or set(expected_counts) != expected_count_keys
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in expected_counts.values()
        )
    ):
        raise PermissionError("typed feasibility expected span evidence missing")
    target_hashes = payload.get("expected_target_family_hashes")
    if (
        not isinstance(target_hashes, list)
        or len(target_hashes) != REQUIRED_SLOT_COUNT
        or len(set(target_hashes)) != REQUIRED_SLOT_COUNT
        or not all(_is_sha256_text(value) for value in target_hashes)
    ):
        raise PermissionError("typed feasibility target family binding missing")
    for key in (
        "manifest_hash",
        "source_train_receipt_hash",
        "expected_v317_slot_plan_hash",
        "expected_implementation_file_set_hash",
    ):
        if not _is_sha256_text(payload.get(key)):
            raise PermissionError(
                f"typed feasibility hash binding is malformed: {key}"
            )
    _canonical_decision_paths(payload, preregistration_path=resolved_path)
    implementation_hash = _implementation_file_set_hash(
        payload,
        preregistration_path=resolved_path,
    )
    if implementation_hash != payload["expected_implementation_file_set_hash"]:
        raise PermissionError("typed feasibility implementation binding drifted")
    return payload


def _validate_source_binding(
    preregistration: Mapping[str, Any],
    *,
    manifest: SplitManifest,
    source_train_receipt_hash: str,
) -> None:
    if preregistration.get("manifest_hash") != manifest.manifest_hash:
        raise PermissionError("typed feasibility manifest binding mismatch")
    if preregistration.get("source_train_receipt_hash") != (
        source_train_receipt_hash
    ):
        raise PermissionError("typed feasibility TRAIN receipt binding mismatch")


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
    }:
        raise PermissionError("typed feasibility canonical paths are missing")
    project_root = (
        Path(preregistration_path).expanduser().resolve(strict=True).parent.parent
    )
    result: dict[str, Path] = {}
    for key in ("report", "events", "decision_lock"):
        relative = Path(str(declared[key]))
        if relative.is_absolute() or ".." in relative.parts:
            raise PermissionError("typed feasibility canonical path is unsafe")
        resolved = (project_root / relative).resolve()
        try:
            resolved.relative_to(project_root)
        except ValueError as exc:
            raise PermissionError(
                "typed feasibility canonical path escaped project root"
            ) from exc
        result[key] = resolved
    if len(set(result.values())) != 3:
        raise PermissionError("typed feasibility canonical paths overlap")
    return result


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
            "typed feasibility decision budget is already consumed"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "lock_version": "typed_operator_feasibility_decision_lock_v1",
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
        "lock_version": "typed_operator_feasibility_decision_lock_v1",
        "decision_ordinal": 1,
        "state": "completed",
        "preregistration_hash": preregistration_hash,
        "decision_hash": report["decision_hash"],
        "report_hash": report["report_hash"],
        "feasibility_passed": report["feasibility_passed"],
        "raw_content_persisted": False,
    }
    if dict(lock) != expected:
        raise PermissionError("typed feasibility decision lock does not match")


def _offline_boundary_contract_issues(
    events: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    issues: list[str] = []
    actual_counts = Counter(str(row.get("event") or "") for row in events)
    allowed_events = {
        "typed_operator_feasibility_started",
        "typed_action_span_evidence_reconstructed",
        "typed_capability_snapshot_created",
        "typed_capability_snapshot_unavailable",
        "typed_action_graph_materialized",
        "typed_action_graph_materialization_skipped",
    }
    snapshot_count = (
        actual_counts["typed_capability_snapshot_created"]
        + actual_counts["typed_capability_snapshot_unavailable"]
    )
    materialization_count = (
        actual_counts["typed_action_graph_materialized"]
        + actual_counts["typed_action_graph_materialization_skipped"]
    )
    if (
        set(actual_counts).difference(allowed_events)
        or actual_counts["typed_operator_feasibility_started"] != 1
        or actual_counts["typed_action_span_evidence_reconstructed"] != 1
        or snapshot_count != REQUIRED_SLOT_COUNT
        or materialization_count != REQUIRED_SLOT_COUNT
        or len(events) != 2 + (2 * REQUIRED_SLOT_COUNT)
    ):
        issues.append("offline_event_type_or_count_mismatch")
    snapshot_targets: Counter[str] = Counter()
    materialization_targets: Counter[str] = Counter()
    snapshot_rows: dict[str, tuple[str, Mapping[str, Any]]] = {}
    materialization_rows: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for row in events:
        if row.get("stage") != "benchmark.skilllearn.typed_operator_feasibility":
            issues.append("offline_event_stage_mismatch")
        payload = row.get("payload")
        if not isinstance(payload, Mapping):
            issues.append("offline_event_payload_missing")
            continue
        for key, expected in _BOUNDARY_FLAGS.items():
            if payload.get(key) is not expected:
                issues.append(f"offline_boundary_flag_mismatch:{key}")
        event_name = str(row.get("event") or "")
        if event_name.startswith("typed_capability_snapshot_"):
            target_hash = str(payload.get("target_family_hash") or "")
            snapshot_targets[target_hash] += 1
            snapshot_rows[target_hash] = (event_name, payload)
        if event_name.startswith("typed_action_graph_material"):
            target_hash = str(payload.get("target_family_hash") or "")
            materialization_targets[target_hash] += 1
            materialization_rows[target_hash] = (event_name, payload)
    if (
        snapshot_targets != materialization_targets
        or len(snapshot_targets) != REQUIRED_SLOT_COUNT
        or any(not key or count != 1 for key, count in snapshot_targets.items())
    ):
        issues.append("offline_event_target_binding_mismatch")
    for target_hash in set(snapshot_rows).intersection(materialization_rows):
        snapshot_event, snapshot = snapshot_rows[target_hash]
        materialization_event, materialization = materialization_rows[
            target_hash
        ]
        commitment = (
            snapshot.get("graph_hash"),
            snapshot.get("model_catalog_hash"),
            snapshot.get("availability_error_hash"),
        )
        replayed_commitment = (
            materialization.get("snapshot_graph_hash"),
            materialization.get("snapshot_model_catalog_hash"),
            materialization.get("snapshot_availability_error_hash"),
        )
        if commitment != replayed_commitment:
            issues.append("offline_snapshot_commitment_mismatch")
        if snapshot_event == "typed_capability_snapshot_created":
            if (
                snapshot.get("graph_available") is not True
                or not _is_sha256_text(commitment[0])
                or not _is_sha256_text(commitment[1])
                or commitment[2] is not None
            ):
                issues.append("offline_snapshot_created_payload_mismatch")
        elif snapshot_event == "typed_capability_snapshot_unavailable":
            if (
                snapshot.get("graph_available") is not False
                or commitment[0] is not None
                or commitment[1] is not None
                or not _is_sha256_text(commitment[2])
            ):
                issues.append("offline_snapshot_unavailable_payload_mismatch")
        if materialization_event == "typed_action_graph_materialized":
            if (
                materialization.get("materialized") is not True
                or materialization.get("skip_reason") is not None
                or materialization.get("graph_hash") != commitment[0]
                or commitment[2] is not None
            ):
                issues.append("offline_materialized_payload_mismatch")
        elif materialization_event == (
            "typed_action_graph_materialization_skipped"
        ):
            skip_reason = materialization.get("skip_reason")
            if materialization.get("materialized") is not False:
                issues.append("offline_materialization_skip_payload_mismatch")
            elif skip_reason == "graph_unavailable":
                if (
                    commitment[0] is not None
                    or materialization.get("graph_hash") is not None
                    or materialization.get("error_type_hash")
                    != commitment[2]
                ):
                    issues.append(
                        "offline_materialization_skip_payload_mismatch"
                    )
            elif skip_reason == "materialization_contract_failure":
                if (
                    materialization.get("graph_hash") != commitment[0]
                    or commitment[2] is not None
                    or not _is_sha256_text(
                        materialization.get("error_type_hash")
                    )
                ):
                    issues.append(
                        "offline_materialization_skip_payload_mismatch"
                    )
            else:
                issues.append("offline_materialization_skip_payload_mismatch")
    return tuple(sorted(set(issues)))


def _implementation_file_set_hash(
    preregistration: Mapping[str, Any],
    *,
    preregistration_path: str | Path,
) -> str:
    declared_roots = preregistration.get("implementation_roots")
    declared_files = preregistration.get("implementation_files")
    if not isinstance(declared_roots, list) or not declared_roots:
        raise PermissionError(
            "typed feasibility implementation-root binding missing"
        )
    if not isinstance(declared_files, list):
        raise PermissionError("typed feasibility implementation binding missing")
    project_root = (
        Path(preregistration_path)
        .expanduser()
        .resolve(strict=True)
        .parent.parent
    )
    relative_files: set[Path] = set()
    for value in declared_roots:
        relative_root = _safe_bound_relative_path(value)
        root = _resolve_bound_project_path(
            project_root,
            relative_root,
            expected_kind="directory",
        )
        root_files = tuple(sorted(root.rglob("*.py")))
        if not root_files:
            raise PermissionError(
                "typed feasibility implementation root has no Python files"
            )
        for candidate in root_files:
            relative = candidate.relative_to(project_root)
            _resolve_bound_project_path(
                project_root,
                relative,
                expected_kind="file",
            )
            relative_files.add(relative)
    for value in declared_files:
        relative = _safe_bound_relative_path(value)
        _resolve_bound_project_path(
            project_root,
            relative,
            expected_kind="file",
        )
        relative_files.add(relative)
    if not relative_files:
        raise PermissionError("typed feasibility implementation set is empty")
    rows: list[dict[str, str]] = []
    for relative in sorted(relative_files, key=lambda row: row.as_posix()):
        resolved = _resolve_bound_project_path(
            project_root,
            relative,
            expected_kind="file",
        )
        rows.append(
            {
                "path": relative.as_posix(),
                "sha256": _sha256_file(resolved),
            }
        )
    return stable_hash({"implementation_files": rows})


def _safe_bound_relative_path(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PermissionError("typed feasibility implementation path malformed")
    relative = Path(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise PermissionError("typed feasibility implementation path unsafe")
    return relative


def _resolve_bound_project_path(
    project_root: Path,
    relative: Path,
    *,
    expected_kind: str,
) -> Path:
    current = project_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise PermissionError(
                "typed feasibility implementation symlink forbidden"
            )
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise PermissionError(
            "typed feasibility implementation escaped project root"
        ) from exc
    if expected_kind == "file" and not resolved.is_file():
        raise PermissionError("typed feasibility implementation file missing")
    if expected_kind == "directory" and not resolved.is_dir():
        raise PermissionError("typed feasibility implementation root missing")
    return resolved


def _is_sha256_text(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
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
            stage="benchmark.skilllearn.typed_operator_feasibility",
            trace_id=trace_id,
            payload={**dict(payload), **_BOUNDARY_FLAGS},
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
                f"typed feasibility event is malformed at line {line_number}"
            )
        rows.append(parsed)
    if not rows:
        raise PermissionError("typed feasibility event ledger is empty")
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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run or replay one preregistered offline typed-operator feasibility "
            "decision from stored TRAIN outcomes; no live model, task backend, "
            "evaluator, held-out split, verifier-content or promotion access."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--source-train-receipt", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args(argv)

    if args.verify_existing:
        report = verify_existing_typed_operator_feasibility(
            root=args.root,
            manifest_path=args.manifest,
            source_run_root=args.source_run_root,
            source_train_receipt=args.source_train_receipt,
            preregistration_path=args.preregistration,
            report_path=args.out,
            events_path=args.events,
        )
    else:
        report = run_typed_operator_feasibility(
            root=args.root,
            manifest_path=args.manifest,
            source_run_root=args.source_run_root,
            source_train_receipt=args.source_train_receipt,
            preregistration_path=args.preregistration,
            report_path=args.out,
            events_path=args.events,
        )
    print(
        json.dumps(
            {
                "decision_hash": report["decision_hash"],
                "feasibility_passed": report["feasibility_passed"],
                "feasibility_reuse_verified": report.get(
                    "feasibility_reuse_verified", False
                ),
                "model_call_count": report["model_call_count"],
                "backend_call_count": report["backend_call_count"],
                "evaluator_call_count": report["evaluator_call_count"],
            },
            sort_keys=True,
        )
    )
    if not report["feasibility_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
