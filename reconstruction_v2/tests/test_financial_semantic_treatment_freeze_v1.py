from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from assumption_agent.benchmarks.financial_semantic_operator_v1 import (
    FINANCIAL_QA_RUNTIME_ASSET_VERSION,
    FINANCIAL_SEMANTIC_ASSET_VERSION,
    FINANCIAL_SEMANTIC_OPERATOR_VERSION,
)
from assumption_agent.benchmarks.financial_semantic_treatment_freeze_v1 import (
    FRESH_FINANCIAL_ITEM_ID,
    PARENT_INSTANCE_MANIFEST_RELATIVE_PATH,
    TASK_INPUT_PREPARATION_RECEIPT_RELATIVE_PATH,
    V320_PREWARM_RECEIPT_RELATIVE_PATH,
    V320_PROTOCOL_RELATIVE_PATH,
    FinancialSemanticTreatmentFreezeError,
    build_financial_semantic_treatment_freeze_v1,
    validate_financial_semantic_treatment_freeze_v1,
)
from assumption_agent.benchmarks.semantic_assignment_operator_v1 import (
    RUNTIME_ASSET_VERSION,
)
from assumption_agent.models import stable_hash


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _with_self_hash(
    payload: dict[str, object], field: str = "manifest_hash"
) -> dict[str, object]:
    payload[field] = stable_hash(payload)
    return payload


def _git_repository(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "BENCHMARK_METADATA.txt").write_text("fixture\n", encoding="utf-8")
    subprocess.run(("git", "init", "-q"), cwd=root, check=True)
    subprocess.run(
        ("git", "config", "user.email", "fixture@example.invalid"),
        cwd=root,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Fixture"), cwd=root, check=True
    )
    subprocess.run(("git", "add", "-A"), cwd=root, check=True)
    subprocess.run(("git", "commit", "-qm", "fixture"), cwd=root, check=True)
    return root


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    project = tmp_path / "project"
    source_root = project / "assumption_agent" / "benchmarks"
    source_root.mkdir(parents=True)
    source_paths: dict[str, Path] = {}
    for role, filename in (
        ("operator_source_path", "financial_semantic_operator_v1.py"),
        ("integration_source_path", "financial_semantic_integration_v1.py"),
        ("prospective_runner_source_path", "financial_semantic_fresh_runner_v1.py"),
        ("lifecycle_source_path", "skilllearn_lifecycle.py"),
        ("offline_verifier_source_path", "offline_verifier.py"),
        ("codex_execution_policy_source_path", "codex_execution_policy.py"),
        ("codex_action_budget_source_path", "codex_action_budget.py"),
        (
            "treatment_freeze_source_path",
            "financial_semantic_treatment_freeze_v1.py",
        ),
    ):
        path = source_root / filename
        path.write_text(f"# {role}\n", encoding="utf-8")
        source_paths[role] = path

    skill = project / "candidates" / "financial_semantic_operator_v1"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# frozen candidate\n", encoding="utf-8")
    source_paths["candidate_skill_source"] = skill

    manifests = project / "manifests"
    minilm = _with_self_hash(
        {
            "asset_version": RUNTIME_ASSET_VERSION,
            "execution": {
                "device": "cpu",
                "local_files_only": True,
                "network_calls": 0,
            },
        }
    )
    minilm_path = manifests / "minilm.json"
    _write_json(minilm_path, minilm)
    source_paths["minilm_runtime_asset_path"] = minilm_path

    qa = _with_self_hash(
        {
            "asset_version": FINANCIAL_QA_RUNTIME_ASSET_VERSION,
            "execution": {
                "device": "cpu",
                "local_files_only": True,
                "network_calls": 0,
            },
        }
    )
    qa_path = manifests / "qa.json"
    _write_json(qa_path, qa)
    source_paths["qa_runtime_asset_path"] = qa_path

    operator_source_sha256 = hashlib.sha256(
        source_paths["operator_source_path"].read_bytes()
    ).hexdigest()
    candidate_material = {
        "operator_version": FINANCIAL_SEMANTIC_OPERATOR_VERSION,
        "formation_source_set_hash": "1" * 64,
        "train_example_set_hash": "2" * 64,
        "configuration_hash": "3" * 64,
        "minilm_runtime_asset_manifest_hash": minilm["manifest_hash"],
        "qa_runtime_asset_manifest_hash": qa["manifest_hash"],
        "operator_source_sha256": operator_source_sha256,
    }
    operator = {
        "asset_version": FINANCIAL_SEMANTIC_ASSET_VERSION,
        "operator_version": FINANCIAL_SEMANTIC_OPERATOR_VERSION,
        **candidate_material,
        "candidate_id": stable_hash(candidate_material),
        "online_calls": 0,
        "prospective_measurement_performed": False,
        "raw_instruction_logged_by_operator": False,
        "excluded_split_access": {
            "fresh_validation_content": False,
            "prior_validation_content": False,
            "residual_sealed_content": False,
        },
    }
    _with_self_hash(operator)
    operator_path = manifests / "operator.json"
    _write_json(operator_path, operator)
    source_paths["operator_asset_path"] = operator_path

    split = _with_self_hash(
        {
            "manifest_version": "skilllearn_fresh_provenance_split_v1",
            "counts": {
                "all": 3,
                "formation": 1,
                "fresh_validation": 1,
                "historical_trial_contaminated": 0,
                "local_content_contaminated": 0,
                "residual_sealed": 1,
            },
            "family_by_id": {
                "financial-analysis-1": "financial-analysis",
                FRESH_FINANCIAL_ITEM_ID: "financial-analysis",
                "financial-analysis-6": "financial-analysis",
            },
            "formation_ids": ["financial-analysis-1"],
            "fresh_validation_ids": [FRESH_FINANCIAL_ITEM_ID],
            "historical_trial_contaminated_ids": [],
            "local_content_contaminated_ids": [],
            "residual_sealed_ids": ["financial-analysis-6"],
            "fresh_validation_content_accessed": False,
            "residual_sealed_content_accessed": False,
            "raw_content_persisted": False,
            "sealed_test": True,
        }
    )
    split_path = manifests / "fresh_split.json"
    _write_json(split_path, split)
    source_paths["fresh_split_manifest_path"] = split_path

    report = {
        "report_version": "financial_semantic_consumed_train_offline_diagnostic_v1",
        "candidate_id": operator["candidate_id"],
        "candidate_manifest_hash": operator["manifest_hash"],
        "operator_source_sha256": operator_source_sha256,
        "cross_fit": False,
        "in_sample_formation_replay": True,
        "retrospective_formation_replay_gain": True,
        "prospective_claim_authorized": False,
        "causal_gain_claim_authorized": False,
        "online_calls": 0,
        "online_judge_calls": 0,
        "offline_verifier_only": True,
        "validation_content_accessed": False,
        "sealed_content_accessed": False,
    }
    _with_self_hash(report, "report_hash")
    report_path = project / "artifacts" / "formation" / "report.json"
    _write_json(report_path, report)
    source_paths["formation_diagnostic_report_path"] = report_path

    for argument, relative_path, marker in (
        (
            "v320_protocol_path",
            V320_PROTOCOL_RELATIVE_PATH,
            "v3.20 protocol",
        ),
        (
            "parent_instance_manifest_path",
            PARENT_INSTANCE_MANIFEST_RELATIVE_PATH,
            "parent instance manifest",
        ),
        (
            "task_input_preparation_receipt_path",
            TASK_INPUT_PREPARATION_RECEIPT_RELATIVE_PATH,
            "task input preparation receipt",
        ),
        (
            "v320_development_prewarm_receipt_path",
            V320_PREWARM_RECEIPT_RELATIVE_PATH,
            "v3.20 development prewarm receipt",
        ),
    ):
        path = project / relative_path
        _write_json(path, {"fixture": marker})
        source_paths[argument] = path

    _git_repository(project)
    benchmark = _git_repository(tmp_path / "benchmark")
    return project, benchmark, source_paths


def _build(
    project: Path, benchmark: Path, paths: dict[str, Path]
) -> dict[str, object]:
    return build_financial_semantic_treatment_freeze_v1(
        project_root=project,
        benchmark_root=benchmark,
        max_steps=100,
        **paths,
    )


def test_build_freezes_opaque_treatment_without_performance_gate(
    tmp_path: Path,
) -> None:
    project, benchmark, paths = _fixture(tmp_path)
    payload = _build(project, benchmark, paths)

    assert len(payload["recipe_id"]) == 64
    assert len(payload["treatment_id"]) == 64
    assert payload["program_set_hash"] == stable_hash(
        {"recipe_ids": [payload["recipe_id"]]}
    )
    assert payload["candidate_skill_source"] == (
        "candidates/financial_semantic_operator_v1"
    )
    assert payload["fresh_item_id"] == FRESH_FINANCIAL_ITEM_ID
    assert payload["official_hipporag"] is False
    assert payload["hipporag_status"] == "not_applicable_nonexecuted"
    assert payload["measurement"] == {
        "policy": "single_fresh_item_paired_offline_measurement_v1",
        "prospective_measurement_performed": False,
        "performance_gate_bound": False,
        "performance_thresholds_bound": False,
        "raw_content_persisted": False,
    }
    roles = {row["role"] for row in payload["source_closure"]["files"]}
    assert {
        "operator_asset",
        "minilm_runtime_asset",
        "qa_runtime_asset",
        "formation_diagnostic_report",
        "prospective_runner_source",
        "v320_protocol",
        "parent_instance_manifest",
        "task_input_preparation_receipt",
        "v320_development_prewarm_receipt",
    }.issubset(roles)
    validate_financial_semantic_treatment_freeze_v1(
        payload,
        project_root=project,
        benchmark_root=benchmark,
    )


def test_live_validation_rejects_runner_source_drift(tmp_path: Path) -> None:
    project, benchmark, paths = _fixture(tmp_path)
    payload = _build(project, benchmark, paths)
    paths["prospective_runner_source_path"].write_text(
        "# changed after freeze\n", encoding="utf-8"
    )

    with pytest.raises(
        FinancialSemanticTreatmentFreezeError,
        match="runtime source scope changed after implementation freeze",
    ):
        validate_financial_semantic_treatment_freeze_v1(
            payload,
            project_root=project,
            benchmark_root=benchmark,
        )


def test_builder_rejects_crossfit_claim_on_in_sample_report(
    tmp_path: Path,
) -> None:
    project, benchmark, paths = _fixture(tmp_path)
    report_path = paths["formation_diagnostic_report_path"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["cross_fit"] = True
    report.pop("report_hash")
    _with_self_hash(report, "report_hash")
    _write_json(report_path, report)

    with pytest.raises(
        FinancialSemanticTreatmentFreezeError,
        match="non-gating replay evidence",
    ):
        _build(project, benchmark, paths)


def test_structural_validation_rejects_added_performance_gate(
    tmp_path: Path,
) -> None:
    project, benchmark, paths = _fixture(tmp_path)
    payload = _build(project, benchmark, paths)
    payload["performance_gate"] = {"minimum_gain": 1}
    payload["manifest_hash"] = stable_hash(
        {key: value for key, value in payload.items() if key != "manifest_hash"}
    )

    with pytest.raises(
        FinancialSemanticTreatmentFreezeError,
        match="top-level fields drifted",
    ):
        validate_financial_semantic_treatment_freeze_v1(payload)


def test_live_validation_rejects_execution_asset_drift(tmp_path: Path) -> None:
    project, benchmark, paths = _fixture(tmp_path)
    payload = _build(project, benchmark, paths)
    paths["v320_protocol_path"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        FinancialSemanticTreatmentFreezeError,
        match="source closure role v320_protocol changed after freeze",
    ):
        validate_financial_semantic_treatment_freeze_v1(
            payload,
            project_root=project,
            benchmark_root=benchmark,
        )
