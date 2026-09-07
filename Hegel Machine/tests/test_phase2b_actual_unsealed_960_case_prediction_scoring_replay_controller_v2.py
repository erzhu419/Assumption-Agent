from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(
    "/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine"
)
TOOL_PATH = (
    ROOT
    / "tools/phase2b_actual_unsealed_960_case_prediction_scoring_replay_v2.py"
)
TOOL_SPEC = importlib.util.spec_from_file_location(
    "phase2b_actual_unsealed_960_case_prediction_scoring_replay_tool_test_v2",
    TOOL_PATH,
)
assert TOOL_SPEC is not None and TOOL_SPEC.loader is not None
tool = importlib.util.module_from_spec(TOOL_SPEC)
sys.modules[TOOL_SPEC.name] = tool
TOOL_SPEC.loader.exec_module(tool)


def _write_contract(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    source = json.loads(
        (
            ROOT
            / "config/phase2b_actual_unsealed_960_case_prediction_scoring_replay_actual_v2.json"
        ).read_text(encoding="utf-8")
    )
    paths = {
        "campaign_lock_path": tmp_path / "campaign.lock",
        "run_root": tmp_path / "run",
        "ledger_root": tmp_path / "ledger",
        "capsule_path": tmp_path / "run/capsule.json",
        "authorization_path": tmp_path / "ledger/authorization.json",
        "consumption_path": tmp_path / "ledger/consumption.json",
        "pending_path": tmp_path / "run/latest.pending.json",
        "pending_history_path": tmp_path / "run/pending-history.jsonl",
        "checkpoint_path": tmp_path / "ledger/checkpoint.json",
        "terminal_path": tmp_path / "ledger/terminal.json",
        "artifact_path": tmp_path / "artifact.cbor",
        "trajectory_path": tmp_path / "run/trajectory.jsonl",
        "upstream_admission_manifest_path": tmp_path / "future/admission.json",
        "scorer_bundle_path": tmp_path / "future/scorer.bundle",
    }
    source["preregistration_path"] = str(
        ROOT
        / "config/phase2b_actual_unsealed_960_case_prediction_scoring_replay_v2.json"
    )
    source["fixed_paths"] = {
        "project_root": str(ROOT),
        **{key: str(value) for key, value in paths.items()},
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(source, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="ascii",
    )
    return contract_path, paths


def _prepare_and_authorize(
    tmp_path: Path,
) -> tuple[Path, dict[str, Path], dict[str, object]]:
    contract_path, paths = _write_contract(tmp_path)
    prepared = tool.prepare_capsule(
        contract_path,
        source_commit="1" * 40,
        run_id="2" * 32,
        created_at_unix_seconds=1,
    )
    tool.record_authorization(contract_path, prepared["capsule_digest"])
    return contract_path, paths, prepared


def _manifest(capsule_digest: str, scorer_payload: bytes) -> dict[str, object]:
    return {
        "schema_version": tool.UPSTREAM_ADMISSION_SCHEMA_V2,
        "capsule_digest": capsule_digest,
        "authoritative_r16_bundle_id": "r16:authoritative-sealed-bundle",
        "campaign_id": "campaign:upstream-bound",
        "attempt_id": "attempt:upstream-bound",
        "recognizer_result_id": "recognizer:sealed-result",
        "baseline_result_ids": [
            "baseline:embedding-nearest-prototype",
            "baseline:frozen-llm-semantic-only",
            "baseline:flat-learned-typed",
        ],
        "prediction_archive_id": "prediction:sealed-archive",
        "answer_manifest_id": "answer:sealed-manifest",
        "scorer_bundle_id": "scorer:sealed-bundle",
        "scorer_bundle_raw_sha256": tool.state.raw_sha256(scorer_payload),
        "authoritative_schema_verifier_available": True,
        "authoritative_r01_r16_bundle_verified": True,
        "external_custodian_checkpoint_verified": True,
        "three_baseline_outputs_verified": True,
        "exact_scale_bridge_verified": True,
        "recognizer_executor_binding_verified": True,
        "scorer_executor_binding_verified": True,
    }


def _fake_evaluation(manifest: dict[str, object]):
    decoded = SimpleNamespace(
        bundle_id=manifest["scorer_bundle_id"],
        campaign_id=manifest["campaign_id"],
        attempt_id=manifest["attempt_id"],
        structural_receipt=SimpleNamespace(
            receipt_id=manifest["recognizer_result_id"]
        ),
        decoded_prediction_archive=SimpleNamespace(
            archive_id=manifest["prediction_archive_id"]
        ),
        answer_manifest=SimpleNamespace(
            answer_manifest_id=manifest["answer_manifest_id"]
        ),
    )
    scoring = SimpleNamespace(
        total_prediction_count=960,
        main_row_result_count=720,
        semantic_conflict_excluded_count=240,
        metric_results=tuple(range(9)),
    )
    gates = SimpleNamespace(
        overall_result_count=10,
        slice_result_count=24,
        unavailable_result_count=2,
        available_overall_gate_mechanics_results=tuple(range(10)),
        available_slice_gate_mechanics_results=tuple(range(24)),
        unavailable_gate_mechanics=(0, 1),
    )
    evaluated = SimpleNamespace(
        score_report_id="score:mechanics-report",
        available_gate_report_id="gate:available-mechanics-report",
        scoring_mechanics=scoring,
        available_gate_mechanics=gates,
    )
    return decoded, evaluated


def _trajectory(paths: dict[str, Path]) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in paths["trajectory_path"].read_text(encoding="ascii").splitlines()
    ]


def test_controller_has_no_subprocess_docker_network_or_producer_call_surface() -> None:
    source = TOOL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TOOL_PATH))
    imports: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add((node.module or "").split(".", 1)[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert not {"subprocess", "socket", "urllib", "requests", "httpx"}.intersection(
        imports
    )
    assert not {
        "Popen",
        "system",
        "execute_phase2b_recognizer_docker_v3",
        "execute_phase2b_scorer_docker_v1",
    }.intersection(calls)
    assert "docker run" not in source.lower()


def test_preflight_and_prepare_never_stat_future_actual_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path, paths = _write_contract(tmp_path)
    future = {
        paths["upstream_admission_manifest_path"],
        paths["scorer_bundle_path"],
    }
    original_lstat = Path.lstat

    def guarded_lstat(path: Path):
        if path in future:
            raise AssertionError(f"future actual input was statted: {path}")
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", guarded_lstat)
    report = tool.preflight(contract_path)
    prepared = tool.prepare_capsule(
        contract_path,
        source_commit="1" * 40,
        run_id="2" * 32,
        created_at_unix_seconds=1,
    )

    assert report["future_actual_input_stat_or_read_count"] == 0
    assert prepared["future_actual_input_stat_or_read_count"] == 0
    assert prepared["logical_campaign_count"] == 0
    assert prepared["authorization_consumption_count"] == 0
    assert not paths["authorization_path"].exists()
    assert not paths["consumption_path"].exists()


def test_missing_upstream_manifest_is_same_digest_pending_without_consumption(
    tmp_path: Path,
) -> None:
    contract_path, paths, prepared = _prepare_and_authorize(tmp_path)
    result = tool.run_v2(contract_path, prepared["capsule_digest"])

    assert result["status"] == tool.state.PRECONSUMPTION_PENDING_V2
    assert result["capsule_digest"] == prepared["capsule_digest"]
    assert result["authorization_consumption_count"] == 0
    assert result["logical_campaign_count"] == 0
    assert not paths["consumption_path"].exists()
    assert not paths["artifact_path"].exists()
    assert not paths["terminal_path"].exists()
    assert [row["event"] for row in _trajectory(paths)] == [
        "CAPSULE_PREPARED",
        "AUTHORIZATION_RECORDED",
        "PRECONSUMPTION_PENDING",
    ]


def test_bundle_content_is_read_once_only_after_consumption_and_mismatch_invalidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path, paths, prepared = _prepare_and_authorize(tmp_path)
    scorer_payload = b"one-shot-scorer-bundle"
    manifest = _manifest(prepared["capsule_digest"], b"different-bound-bytes")
    paths["upstream_admission_manifest_path"].parent.mkdir(parents=True)
    paths["upstream_admission_manifest_path"].write_bytes(
        tool.state.canonical_json_bytes(manifest)
    )
    paths["scorer_bundle_path"].write_bytes(scorer_payload)

    reads = 0
    original_read = tool._read_regular_once

    def counted_read(path: Path, label: str, maximum_bytes: int) -> bytes:
        nonlocal reads
        if path == paths["scorer_bundle_path"]:
            reads += 1
            assert paths["consumption_path"].exists()
        return original_read(path, label, maximum_bytes)

    monkeypatch.setattr(tool, "_read_regular_once", counted_read)
    monkeypatch.setattr(
        tool,
        "_decode_and_evaluate_once",
        lambda payload: pytest.fail("raw mismatch must precede decoder"),
    )
    terminal = tool.run_v2(contract_path, prepared["capsule_digest"])

    assert reads == 1
    assert terminal["status"] == tool.state.INVALIDATED_TERMINAL_NO_RESUME_V2
    assert terminal["outcome"]["failure_code"] == "POST_CONSUMPTION_REPLAY_FAILED"
    assert [row["event"] for row in _trajectory(paths)][-3:] == [
        "AUTHORIZATION_CONSUMED",
        "SCORER_BUNDLE_READ",
        "INVALIDATED",
    ]


def test_success_and_artifact_only_partial_finalization_never_repeat_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path, paths, prepared = _prepare_and_authorize(tmp_path)
    scorer_payload = b"synthetic-test-double-scorer-bundle"
    manifest = _manifest(prepared["capsule_digest"], scorer_payload)
    paths["upstream_admission_manifest_path"].parent.mkdir(parents=True)
    paths["upstream_admission_manifest_path"].write_bytes(
        tool.state.canonical_json_bytes(manifest)
    )
    paths["scorer_bundle_path"].write_bytes(scorer_payload)

    calls = 0

    def evaluate_once(payload: bytes):
        nonlocal calls
        calls += 1
        assert payload == scorer_payload
        return _fake_evaluation(manifest)

    monkeypatch.setattr(tool, "_decode_and_evaluate_once", evaluate_once)
    original_publish = tool._publish_new_or_exact
    terminal_failures = 0

    def interrupt_terminal(path: Path, payload: bytes, label: str) -> bool:
        nonlocal terminal_failures
        if path == paths["terminal_path"] and terminal_failures == 0:
            terminal_failures += 1
            raise tool.ControllerPendingV2(
                "LOCAL_PUBLICATION", "TEST_TERMINAL_INTERRUPTION", "after artifact"
            )
        return original_publish(path, payload, label)

    monkeypatch.setattr(tool, "_publish_new_or_exact", interrupt_terminal)
    with pytest.raises(tool.ControllerPendingV2):
        tool.run_v2(contract_path, prepared["capsule_digest"])
    assert calls == 1
    assert paths["artifact_path"].exists()
    assert not paths["terminal_path"].exists()

    monkeypatch.setattr(tool, "_publish_new_or_exact", original_publish)
    terminal = tool.run_v2(contract_path, prepared["capsule_digest"])
    assert terminal["status"] == tool.core.MECHANICS_COMPLETE_STATUS_V2
    assert calls == 1
    assert terminal["outcome"]["total_predictions"] == 960
    assert terminal["outcome"]["main_scored_rows"] == 720
    assert terminal["outcome"]["semantic_conflict_excluded"] == 240
    assert terminal["outcome"]["metric_results"] == 9
    assert terminal["outcome"]["available_overall_gate_mechanics"] == 10
    assert terminal["outcome"]["available_slice_gate_mechanics"] == 24
    assert terminal["outcome"]["unavailable_formal_overall_gates"] == 2
    assert terminal["outcome"]["formal_gate_evaluation_performed"] is False
    assert terminal["outcome"]["c1_pass"] is False

    events = [row["event"] for row in _trajectory(paths)]
    assert events == [
        "CAPSULE_PREPARED",
        "AUTHORIZATION_RECORDED",
        "AUTHORIZATION_CONSUMED",
        "SCORER_BUNDLE_READ",
        "SCORER_REPLAY_COMPLETED",
        "ARTIFACT_PUBLISHED",
        "TERMINAL_PUBLISHED",
    ]
