from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pytest

from assumption_agent.benchmarks.skilllearn_compiler import (
    _lower_skilllearn_program,
)
from assumption_agent.benchmarks.typed_operator_feasibility import (
    _ACCEPTANCE_PREDICATES,
    _BOUNDARY_FLAGS,
    _canonical_decision_paths,
    _implementation_file_set_hash,
    _offline_boundary_contract_issues,
    _read_preregistration,
    _reserve_decision_lock,
    _selection_tamper_rejected,
    run_typed_operator_feasibility,
)
from assumption_agent.models import ResidualExample, SplitName, stable_hash
from assumption_agent.typed_operator_grammar import (
    ALLOWED_OPERATOR_KINDS,
    FORBIDDEN_OPERATOR_KINDS,
    PrimitiveAssessmentClass,
    SpanOutcome,
    TypedGraphUnavailableError,
    TrialTraceEvidence,
    assess_observed_primitives,
    build_family_capability_graph,
    canonical_recipe,
    extract_trial_trace_evidence,
    materialize_recipe_selection,
    selection_schema,
    _canonical_task_locator,
)


def test_chronological_span_evidence_preserves_recovery_and_all_occurrences(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "source" / "trial" / "agent" / "codex.txt"
    trace.parent.mkdir(parents=True)
    rows: list[Mapping[str, Any]] = [
        {
            "type": "assumption.action_budget.started",
            "policy": "codex_jsonl_action_start_budget_v1",
            "unit": "codex_action_start_v1",
            "limit": 100,
        }
    ]
    commands = [
        ("python /root/data.json", "failed", 1),
        ("python /root/data.json", "completed", 0),
        ("file /root/data.json", "failed", 1),
        ("python3 /root/data.json", "completed", 0),
        *(
            (f"sed -n /root/file-{index}.txt", "completed", 0)
            for index in range(13)
        ),
    ]
    for index, (command, status, exit_code) in enumerate(commands):
        rows.extend(
            _command_events(
                item_id=f"item-{index}",
                command=command,
                status=status,
                exit_code=exit_code,
            )
        )
    rows.extend(
        [
            {
                "type": "item.started",
                "item": {
                    "id": "change-1",
                    "type": "file_change",
                    "changes": [
                        {"path": "/root/output.json", "kind": "add"}
                    ],
                    "status": "in_progress",
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "id": "change-1",
                    "type": "file_change",
                    "changes": [
                        {"path": "/root/output.json", "kind": "add"}
                    ],
                    "status": "completed",
                },
            },
            {"type": "turn.completed", "usage": {}},
        ]
    )
    _write_trace(trace, rows)

    evidence = extract_trial_trace_evidence(
        trace,
        containment_root=tmp_path / "source",
        trial_id_hash="a" * 64,
        family_hash="b" * 64,
        trace_hash=hashlib.sha256(trace.read_bytes()).hexdigest(),
        action_budget_receipt_hash=stable_hash({"receipt": "fixture"}),
        **_action_receipt_binding(rows),
        baseline_success=True,
    )

    assert evidence.trace_complete is True
    assert evidence.action_start_count == 18
    assert evidence.command_span_count == 17
    assert [row.span_index for row in evidence.spans] == list(range(17))
    assert [row.completion_event_index for row in evidence.spans] == sorted(
        row.completion_event_index for row in evidence.spans
    )
    assert evidence.safe_payload()[
        "allowlisted_command_occurrences_deduplicated_or_truncated"
    ] is False
    assert evidence.safe_payload()["full_raw_command_coverage_claimed"] is False
    assert evidence.spans[0].outcome is SpanOutcome.FAILED
    assert evidence.spans[0].later_exact_success_span_hashes
    assert evidence.spans[2].later_shared_artifact_success_span_hashes
    assert evidence.spans[0].span_id in {
        row.span_id for row in evidence.spans
    }
    assert all(
        reference in {row.span_id for row in evidence.spans}
        for failed in evidence.spans
        for reference in (
            *failed.later_exact_success_span_hashes,
            *failed.later_same_executable_success_span_hashes,
            *failed.later_shared_artifact_success_span_hashes,
        )
    )
    assert len(evidence.changed_artifacts) == 1
    assert "python /root/data.json" not in json.dumps(
        evidence.safe_payload(), sort_keys=True
    )

    assessments = assess_observed_primitives((evidence,))
    assert any(
        row.scope == "executable"
        and row.classification
        is PrimitiveAssessmentClass.RECOVERED_AFTER_FAILURE
        for row in assessments
    )
    assert all(not row.observationally_inadmissible for row in assessments)


def test_repeated_terminal_exact_failure_is_not_primitive_inadmissibility(
    tmp_path: Path,
) -> None:
    trials = tuple(
        _single_command_trial(
            tmp_path / f"trial-{index}",
            trial_id_hash=stable_hash({"trial": index}),
            command="python /root/data.json",
            status="failed",
            exit_code=1,
        )
        for index in range(2)
    )
    assessments = assess_observed_primitives(trials)
    exact = next(row for row in assessments if row.scope == "exact_command")
    executable = next(row for row in assessments if row.scope == "executable")
    artifact = next(row for row in assessments if row.scope == "artifact")

    assert exact.classification is (
        PrimitiveAssessmentClass.DO_NOT_RECOMMEND_EXACT_SIGNATURE
    )
    assert executable.classification is (
        PrimitiveAssessmentClass.FAILURE_COOCCURRENCE_ONLY
    )
    assert artifact.classification is (
        PrimitiveAssessmentClass.FAILURE_COOCCURRENCE_ONLY
    )
    assert all(not row.observationally_inadmissible for row in assessments)


def test_cross_trial_or_earlier_success_is_not_labeled_recovery(
    tmp_path: Path,
) -> None:
    success = _single_command_trial(
        tmp_path / "success-first",
        trial_id_hash=stable_hash({"trial": "success"}),
        command="python /root/data.json",
        status="completed",
        exit_code=0,
    )
    failure = _single_command_trial(
        tmp_path / "failure-second",
        trial_id_hash=stable_hash({"trial": "failure"}),
        command="python /root/data.json",
        status="failed",
        exit_code=1,
    )
    assessments = assess_observed_primitives((success, failure))
    for row in assessments:
        assert row.recovery_contradiction_count == 0
        assert row.classification is (
            PrimitiveAssessmentClass.OPERATIONALLY_OBSERVED
        )


def test_trace_extraction_fails_closed_on_symlink_and_budget_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    trace = root / "trial" / "codex.txt"
    trace.parent.mkdir(parents=True)
    rows = [
        {"type": "assumption.action_budget.started", "limit": 99},
        *_command_events(
            item_id="item-1",
            command="python /root/data.json",
            status="completed",
            exit_code=0,
        ),
        {"type": "turn.completed", "usage": {}},
    ]
    _write_trace(trace, rows)
    with pytest.raises(PermissionError, match="budget identity"):
        extract_trial_trace_evidence(
            trace,
            containment_root=root,
            trial_id_hash="a" * 64,
            family_hash="b" * 64,
            trace_hash=hashlib.sha256(trace.read_bytes()).hexdigest(),
            action_budget_receipt_hash="d" * 64,
            **_action_receipt_binding(rows),
            baseline_success=False,
        )

    trace.unlink()
    target = root / "real.txt"
    _write_trace(
        target,
        [
            {"type": "assumption.action_budget.started", "limit": 100},
            {"type": "turn.completed", "usage": {}},
        ],
    )
    trace.symlink_to(target)
    with pytest.raises(PermissionError, match="symlink"):
        extract_trial_trace_evidence(
            trace,
            containment_root=root,
            trial_id_hash="a" * 64,
            family_hash="b" * 64,
            trace_hash="c" * 64,
            action_budget_receipt_hash="d" * 64,
            **_action_receipt_binding([]),
            baseline_success=False,
        )


def test_trace_state_machine_rejects_orphan_completion(tmp_path: Path) -> None:
    root = tmp_path / "source"
    trace = root / "trial" / "codex.txt"
    trace.parent.mkdir(parents=True)
    rows = [
            {
                "type": "assumption.action_budget.started",
                "policy": "codex_jsonl_action_start_budget_v1",
                "unit": "codex_action_start_v1",
                "limit": 100,
            },
            {
                "type": "item.completed",
                "item": {
                    "id": "orphan",
                    "type": "command_execution",
                    "command": "python /root/data.json",
                    "status": "completed",
                    "exit_code": 0,
                },
            },
            {"type": "turn.completed", "usage": {}},
        ]
    _write_trace(trace, rows)
    with pytest.raises(PermissionError, match="unpaired"):
        extract_trial_trace_evidence(
            trace,
            containment_root=root,
            trial_id_hash="a" * 64,
            family_hash="b" * 64,
            trace_hash=hashlib.sha256(trace.read_bytes()).hexdigest(),
            action_budget_receipt_hash="d" * 64,
            **_action_receipt_binding(rows),
            baseline_success=False,
        )


@pytest.mark.parametrize(
    ("rows", "message"),
    (
        (
            [
                {
                    "type": "item.started",
                    "item": {
                        "id": "open",
                        "type": "command_execution",
                        "command": "python /root/data.json",
                        "status": "in_progress",
                    },
                },
                {"type": "turn.completed", "usage": {}},
            ],
            "unclosed",
        ),
        (
            [
                {
                    "type": "item.started",
                    "item": {
                        "id": "changed",
                        "type": "command_execution",
                        "command": "python /root/data.json",
                        "status": "in_progress",
                    },
                },
                {
                    "type": "item.completed",
                    "item": {
                        "id": "changed",
                        "type": "command_execution",
                        "command": "python /root/other.json",
                        "status": "completed",
                        "exit_code": 0,
                    },
                },
                {"type": "turn.completed", "usage": {}},
            ],
            "command changed",
        ),
        (
            [
                {"type": "turn.completed", "usage": {}},
                {
                    "type": "item.started",
                    "item": {
                        "id": "late",
                        "type": "web_search",
                        "query": "must still count toward action budget",
                        "status": "in_progress",
                    },
                },
            ],
            "starts after terminal",
        ),
    ),
)
def test_trace_state_machine_rejects_temporal_drift(
    tmp_path: Path,
    rows: list[Mapping[str, Any]],
    message: str,
) -> None:
    root = tmp_path / message.replace(" ", "-")
    trace = root / "trial" / "codex.txt"
    trace.parent.mkdir(parents=True)
    trace_rows = [
            {
                "type": "assumption.action_budget.started",
                "policy": "codex_jsonl_action_start_budget_v1",
                "unit": "codex_action_start_v1",
                "limit": 100,
            },
            *rows,
        ]
    _write_trace(trace, trace_rows)
    with pytest.raises(PermissionError, match=message):
        extract_trial_trace_evidence(
            trace,
            containment_root=root,
            trial_id_hash="a" * 64,
            family_hash="b" * 64,
            trace_hash=hashlib.sha256(trace.read_bytes()).hexdigest(),
            action_budget_receipt_hash="d" * 64,
            **_action_receipt_binding(trace_rows),
            baseline_success=False,
        )


def test_every_item_start_counts_toward_action_budget(tmp_path: Path) -> None:
    root = tmp_path / "source"
    trace = root / "trial" / "codex.txt"
    trace.parent.mkdir(parents=True)
    rows: list[Mapping[str, Any]] = [
        {
            "type": "assumption.action_budget.started",
            "policy": "codex_jsonl_action_start_budget_v1",
            "unit": "codex_action_start_v1",
            "limit": 100,
        },
        *(
            {
                "type": "item.started",
                "item": {
                    "id": f"web-{index}",
                    "type": "web_search",
                    "query": "offline probe",
                    "status": "in_progress",
                },
            }
            for index in range(101)
        ),
        {"type": "turn.completed", "usage": {}},
    ]
    _write_trace(trace, rows)
    frozen_receipt_rows = rows[:101]
    with pytest.raises(PermissionError, match="exceeds frozen action budget"):
        extract_trial_trace_evidence(
            trace,
            containment_root=root,
            trial_id_hash="a" * 64,
            family_hash="b" * 64,
            trace_hash=hashlib.sha256(trace.read_bytes()).hexdigest(),
            action_budget_receipt_hash="d" * 64,
            **_action_receipt_binding(frozen_receipt_rows),
            baseline_success=False,
        )


def test_typed_artifact_locator_rejects_escape_and_sensitive_roots() -> None:
    assert _canonical_task_locator("/root/data/input.csv") == (
        "/root/data/input.csv"
    )
    for unsafe in (
        "/root/../etc/passwd",
        "/root/./data.json",
        "/root/verifier.json",
        "/root/tests/case.json",
        "/root/auth.json",
        "/root/credentials/token.txt",
        "/tmp/data.json",
    ):
        assert _canonical_task_locator(unsafe) is None


def test_closed_typed_graph_materializes_only_registered_recipe_ids(
    tmp_path: Path,
) -> None:
    family = "family-a"
    profile = {
        "runtime_environment": {
            "declared_task_local_paths": ["/root/data.csv"],
            "copied_task_files": ["/root/data.csv"],
            "environment_source_files": ["data.csv"],
        },
        "baseline_action_trace": {},
    }
    profile_hash = stable_hash(profile)
    failures = tuple(
        _residual(
            family=family,
            task_id=f"failure-{index}",
            profile_hash=profile_hash,
        )
        for index in range(2)
    )
    trial_evidence = {
        stable_hash({"item_id": residual.task_id}): _single_command_trial(
            tmp_path / residual.task_id,
            trial_id_hash=stable_hash({"item_id": residual.task_id}),
            family_hash=stable_hash({"family": family}),
            command="python3 /root/data.csv",
            status="completed",
            exit_code=0,
        )
        for residual in failures
    }
    graph = build_family_capability_graph(
        target_family=family,
        failures=failures,
        action_profiles={profile_hash: profile},
        trial_evidence=trial_evidence,
    )
    replay = build_family_capability_graph(
        target_family=family,
        failures=failures,
        action_profiles={profile_hash: profile},
        trial_evidence=trial_evidence,
    )

    assert graph.graph_hash == replay.graph_hash
    assert graph.validate() == ()
    assert ALLOWED_OPERATOR_KINDS.isdisjoint(FORBIDDEN_OPERATOR_KINDS)
    assert graph.artifacts
    assert graph.recipes
    catalog_text = json.dumps(graph.model_catalog(), sort_keys=True)
    assert "/root/data.csv" not in catalog_text
    assert '"python3"' not in catalog_text
    schema = selection_schema(graph)
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["recipe_id"]
    assert set(schema["properties"]["recipe_id"]["enum"]) == {
        row.recipe_id for row in graph.recipes
    }
    assert _selection_tamper_rejected(graph) == 3

    recipe = canonical_recipe(graph)
    program = materialize_recipe_selection(
        {"recipe_id": recipe.recipe_id},
        graph=graph,
        evaluator_epoch="typed-fixture",
        expected_graph_hash=graph.graph_hash,
        expected_model_catalog_hash=stable_hash(graph.model_catalog()),
    )
    assert program.validate() == []
    assert program.trigger.all_of[0].value == family
    assert program.anti_trigger.is_empty
    lowered = _lower_skilllearn_program(program)
    assert lowered
    assert {row.semantics for row in lowered}.issubset(
        {"prompt_directive", "agent_local_self_check"}
    )
    assert any(row.semantics == "prompt_directive" for row in lowered)
    assert all(
        action.target.startswith("typed operator ")
        for action in program.action_graph
    )

    for invalid in (
        {"recipe_id": "unknown"},
        {"recipe_id": recipe.recipe_id, "command": "python /tmp/pwn"},
        {"recipe_id": {"operator": "network_fetch"}},
    ):
        with pytest.raises(PermissionError):
            materialize_recipe_selection(
                invalid,
                graph=graph,
                evaluator_epoch="typed-fixture",
                expected_graph_hash=graph.graph_hash,
                expected_model_catalog_hash=stable_hash(
                    graph.model_catalog()
                ),
            )

    forged_artifact = replace(graph.artifacts[0], locator="/tmp/pwn")
    forged_graph = replace(
        graph,
        artifacts=(forged_artifact, *graph.artifacts[1:]),
    )
    assert "artifact_locator_not_canonical" in forged_graph.validate()
    with pytest.raises(PermissionError, match="graph is invalid"):
        materialize_recipe_selection(
            {"recipe_id": forged_graph.recipes[0].recipe_id},
            graph=forged_graph,
            evaluator_epoch="typed-fixture",
            expected_graph_hash=graph.graph_hash,
            expected_model_catalog_hash=stable_hash(graph.model_catalog()),
        )

    same_format_artifact = replace(
        graph.artifacts[0],
        locator="/root/other.csv",
    )
    same_format_graph = replace(
        graph,
        artifacts=(same_format_artifact, *graph.artifacts[1:]),
    )
    assert "artifact_id_not_canonical" in same_format_graph.validate()

    unknown_relation_artifact = replace(
        graph.artifacts[0],
        evidence_relations=("invented_positive",),
    )
    unknown_relation_graph = replace(
        graph,
        artifacts=(unknown_relation_artifact, *graph.artifacts[1:]),
    )
    assert "artifact_relation_unknown" in unknown_relation_graph.validate()

    changed_family_graph = replace(graph, target_family="family-b")
    assert "artifact_id_not_canonical" in changed_family_graph.validate()

    changed_capability = replace(
        graph.capabilities[0],
        provenance_hash="e" * 64,
    )
    changed_capability_graph = replace(
        graph,
        capabilities=(changed_capability, *graph.capabilities[1:]),
    )
    assert (
        "capability_registry_not_canonical"
        in changed_capability_graph.validate()
    )

    changed_workflow = replace(
        graph.recipes[0],
        workflow=next(
            row.workflow
            for row in graph.recipes
            if row.workflow is not graph.recipes[0].workflow
        ),
    )
    changed_workflow_graph = replace(
        graph,
        recipes=(changed_workflow, *graph.recipes[1:]),
    )
    assert "recipe_registry_not_canonical" in changed_workflow_graph.validate()

    for tampered_graph in (
        same_format_graph,
        unknown_relation_graph,
        changed_family_graph,
        changed_capability_graph,
        changed_workflow_graph,
    ):
        with pytest.raises(PermissionError):
            materialize_recipe_selection(
                {"recipe_id": graph.recipes[0].recipe_id},
                graph=tampered_graph,
                evaluator_epoch="typed-fixture",
                expected_graph_hash=graph.graph_hash,
                expected_model_catalog_hash=stable_hash(
                    graph.model_catalog()
                ),
            )

    empty_recipe = replace(graph.recipes[0], nodes=())
    empty_graph = replace(
        graph,
        recipes=(empty_recipe, *graph.recipes[1:]),
    )
    assert "recipe_nodes_empty" in empty_graph.validate()


def test_graph_without_supported_safe_artifact_is_semantic_unavailable(
    tmp_path: Path,
) -> None:
    family = "family-a"
    profile = {
        "runtime_environment": {
            "declared_task_local_paths": ["/root/tests/case.csv"],
            "copied_task_files": [],
            "environment_source_files": [],
        },
        "baseline_action_trace": {},
    }
    profile_hash = stable_hash(profile)
    failures = tuple(
        _residual(
            family=family,
            task_id=f"unsafe-{index}",
            profile_hash=profile_hash,
        )
        for index in range(2)
    )
    trial_evidence = {
        stable_hash({"item_id": residual.task_id}): _single_command_trial(
            tmp_path / residual.task_id,
            trial_id_hash=stable_hash({"item_id": residual.task_id}),
            family_hash=stable_hash({"family": family}),
            command="python3 --version",
            status="completed",
            exit_code=0,
        )
        for residual in failures
    }
    with pytest.raises(
        TypedGraphUnavailableError,
        match="no_supported_task_local_artifact",
    ):
        build_family_capability_graph(
            target_family=family,
            failures=failures,
            action_profiles={profile_hash: profile},
            trial_evidence=trial_evidence,
        )


def test_preregistration_is_single_decision_and_matches_code_constants() -> None:
    root = Path(__file__).resolve().parents[1]
    preregistration_path = (
        root / "manifests" / "skilllearn_typed_operator_feasibility_v1.json"
    )
    preregistration = _read_preregistration(
        preregistration_path
    )
    assert preregistration["decision_budget"] == 1
    assert preregistration["model_surface"][
        "model_authored_primitive_fields"
    ] == []
    assert preregistration["causal_rules"][
        "observational_task_failure_can_mark_primitive_inadmissible"
    ] is False
    assert preregistration["runtime_scope"][
        "restricted_runtime_executor_claimed"
    ] is False
    assert preregistration["acceptance_predicates"] == dict(
        _ACCEPTANCE_PREDICATES
    )
    assert all(preregistration["acceptance"].values())
    assert _implementation_file_set_hash(
        preregistration,
        preregistration_path=preregistration_path,
    ) == preregistration["expected_implementation_file_set_hash"]
    canonical = _canonical_decision_paths(
        preregistration,
        preregistration_path=preregistration_path,
    )
    assert canonical["report"].name == (
        "typed_operator_feasibility.report.json"
    )
    assert canonical["events"].name == (
        "typed_operator_feasibility.events.jsonl"
    )
    assert canonical["decision_lock"].name == (
        "typed_operator_feasibility.decision.lock.json"
    )


def test_decision_lock_is_exclusive_and_noncanonical_run_fails_closed(
    tmp_path: Path,
) -> None:
    decision_lock = tmp_path / "decision.lock.json"
    _reserve_decision_lock(decision_lock, preregistration_hash="a" * 64)
    with pytest.raises(FileExistsError, match="already consumed"):
        _reserve_decision_lock(decision_lock, preregistration_hash="a" * 64)
    assert json.loads(decision_lock.read_text(encoding="utf-8"))["state"] == (
        "reserved"
    )

    root = Path(__file__).resolve().parents[1]
    with pytest.raises(PermissionError, match="canonical paths"):
        run_typed_operator_feasibility(
            root=tmp_path,
            manifest_path=tmp_path / "unused-manifest.json",
            source_run_root=tmp_path / "unused-run",
            source_train_receipt=tmp_path / "unused-receipt.json",
            preregistration_path=(
                root
                / "manifests"
                / "skilllearn_typed_operator_feasibility_v1.json"
            ),
            report_path=tmp_path / "noncanonical.report.json",
            events_path=tmp_path / "noncanonical.events.jsonl",
        )


def test_boundary_ledger_allows_completed_semantic_graph_failure() -> None:
    target_hashes = ("a" * 64, "b" * 64, "c" * 64)
    events: list[dict[str, Any]] = []
    commitments = {
        target_hashes[0]: {
            "graph_hash": stable_hash({"graph": 0}),
            "model_catalog_hash": stable_hash({"catalog": 0}),
            "availability_error_hash": None,
        },
        target_hashes[1]: {
            "graph_hash": None,
            "model_catalog_hash": None,
            "availability_error_hash": stable_hash(
                {"error": "unavailable"}
            ),
        },
        target_hashes[2]: {
            "graph_hash": stable_hash({"graph": 2}),
            "model_catalog_hash": stable_hash({"catalog": 2}),
            "availability_error_hash": None,
        },
    }

    def append(
        event: str,
        target_hash: str | None = None,
        **extra: Any,
    ) -> None:
        payload: dict[str, Any] = dict(_BOUNDARY_FLAGS)
        if target_hash is not None:
            payload["target_family_hash"] = target_hash
        payload.update(extra)
        events.append(
            {
                "event": event,
                "stage": "benchmark.skilllearn.typed_operator_feasibility",
                "payload": payload,
            }
        )

    append("typed_operator_feasibility_started")
    append("typed_action_span_evidence_reconstructed")
    for index, target_hash in enumerate(target_hashes):
        commitment = commitments[target_hash]
        append(
            (
                "typed_capability_snapshot_unavailable"
                if index == 1
                else "typed_capability_snapshot_created"
            ),
            target_hash,
            graph_available=index != 1,
            **commitment,
        )
    for index, target_hash in enumerate(target_hashes):
        commitment = commitments[target_hash]
        common = {
            "snapshot_graph_hash": commitment["graph_hash"],
            "snapshot_model_catalog_hash": commitment[
                "model_catalog_hash"
            ],
            "snapshot_availability_error_hash": commitment[
                "availability_error_hash"
            ],
            "graph_hash": commitment["graph_hash"],
        }
        if index == 1:
            append(
                "typed_action_graph_materialization_skipped",
                target_hash,
                materialized=False,
                skip_reason="graph_unavailable",
                error_type_hash=commitment["availability_error_hash"],
                **common,
            )
        else:
            append(
                "typed_action_graph_materialized",
                target_hash,
                materialized=True,
                skip_reason=None,
                error_type_hash=None,
                **common,
            )

    assert _offline_boundary_contract_issues(events) == ()

    tampered = [dict(row) for row in events]
    tampered[-1] = {
        **tampered[-1],
        "payload": {
            **dict(tampered[-1]["payload"]),
            "snapshot_graph_hash": "d" * 64,
        },
    }
    assert "offline_snapshot_commitment_mismatch" in (
        _offline_boundary_contract_issues(tampered)
    )


def _residual(
    *,
    family: str,
    task_id: str,
    profile_hash: str,
) -> ResidualExample:
    return ResidualExample(
        transition_id="transition-" + task_id,
        task_id=task_id,
        family=family,
        split=SplitName.TRAIN,
        features={"family": family},
        failure_type="fixture_failure",
        evaluator_feedback=("offline TRAIN failure",),
        baseline_success=False,
        context={"action_context_profile_hash": profile_hash},
    )


def _single_command_trial(
    root: Path,
    *,
    trial_id_hash: str,
    family_hash: str = "f" * 64,
    command: str,
    status: str,
    exit_code: int,
) -> TrialTraceEvidence:
    trace = root / "codex.txt"
    trace.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "type": "assumption.action_budget.started",
            "policy": "codex_jsonl_action_start_budget_v1",
            "unit": "codex_action_start_v1",
            "limit": 100,
        },
        *_command_events(
            item_id="item-1",
            command=command,
            status=status,
            exit_code=exit_code,
        ),
        {"type": "turn.completed", "usage": {}},
    ]
    _write_trace(trace, rows)
    return extract_trial_trace_evidence(
        trace,
        containment_root=root,
        trial_id_hash=trial_id_hash,
        family_hash=family_hash,
        trace_hash=hashlib.sha256(trace.read_bytes()).hexdigest(),
        action_budget_receipt_hash=stable_hash({"receipt": trial_id_hash}),
        **_action_receipt_binding(rows),
        baseline_success=False,
    )


def _command_events(
    *,
    item_id: str,
    command: str,
    status: str,
    exit_code: int,
) -> list[Mapping[str, Any]]:
    return [
        {
            "type": "item.started",
            "item": {
                "id": item_id,
                "type": "command_execution",
                "command": command,
                "status": "in_progress",
                "exit_code": None,
                "aggregated_output": "MUST-NOT-BE-USED",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": item_id,
                "type": "command_execution",
                "command": command,
                "status": status,
                "exit_code": exit_code,
                "aggregated_output": "VERIFIER-SECRET-MUST-NOT-BE-USED",
            },
        },
    ]


def _write_trace(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _action_receipt_binding(
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for row in rows:
        if row.get("type") != "item.started":
            continue
        item = row.get("item")
        item_id = str(item.get("id") or "") if isinstance(item, Mapping) else ""
        item_type = (
            str(item.get("type") or "") if isinstance(item, Mapping) else ""
        )
        events.append(
            {
                "event_index": len(events) + 1,
                "item_id": item_id,
                "item_type": item_type,
                "malformed": not item_id or not item_type,
            }
        )
    return {
        "expected_action_start_count": len(events),
        "expected_action_event_hash": stable_hash(events),
    }
