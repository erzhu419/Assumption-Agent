from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import hegel_machine.phase2b_scorer_runtime_cli_v1 as runtime
from hegel_machine.phase2b_c1_container_actor_profile_v2 import (
    PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256,
    PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID,
)
from hegel_machine.phase2b_scorer_bundle_v1 import (
    DecodedPhase2BScorerBundleV1,
    EvaluatedPhase2BScorerBundleV1,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SOURCE = Path(runtime.__file__).read_text(encoding="utf-8")


def _private(cls: type[object], **values: object) -> object:
    value = object.__new__(cls)
    for name, item in values.items():
        object.__setattr__(value, name, item)
    return value


def _install_success(
    monkeypatch: pytest.MonkeyPatch,
    bundle_payload: bytes,
) -> tuple[bytes, bytes]:
    decoded = _private(
        DecodedPhase2BScorerBundleV1,
        bundle_id="phase2b_scorer_bundle_v1_" + hashlib.sha256(bundle_payload).hexdigest(),
        authority_requirement_id=(
            "phase2b_c1_owner_accepted_container_authority_requirement_v1_"
            "53631d563f8da8f26c7202b7dd0681be5423fe2ca5c15e9a9f5deaa2920d9940"
        ),
        actor_profile_content_id=PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID,
        actor_profile_canonical_sha256=(
            PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256
        ),
        campaign_id="phase2b_actual_unsealed_960_campaign_v2_" + SHA_A,
        attempt_id="phase2b_actual_unsealed_960_attempt_v2_" + SHA_A,
        execution_registry_id=(
            "phase2b_c1_owner_accepted_execution_registry_v1_" + SHA_A
        ),
        scorer_image_digest="sha256:" + SHA_B,
        scorer_source_root_sha256=SHA_B,
        protocol_id="phase2b_protocol_" + SHA_A,
        exact_freeze_id="phase2b_exact_freeze_" + SHA_A,
        execution_freeze_manifest_id="phase2b_execution_freeze_" + SHA_A,
        scoring_source_sha256=(
            "c3c12e9cd72a930b9ca8667aab22c4eb3675ceab6249b0b26ba940bf794e7f11"
        ),
        run_manifest_id="phase2b_scorer_run_manifest_v1_" + SHA_A,
    )
    scoring = SimpleNamespace(result_id="score_result_" + SHA_A)
    replay = SimpleNamespace(result_id="replay_result_" + SHA_A)
    gate = SimpleNamespace(result_id="gate_result_" + SHA_A)
    score_report = b'{"report_kind":"score"}\n'
    gate_report = b'{"report_kind":"gate"}\n'
    score_report_id, gate_report_id = runtime._report_ids_v1(
        score_report,
        gate_report,
    )
    evaluation = _private(
        EvaluatedPhase2BScorerBundleV1,
        scoring_mechanics=scoring,
        replay_input_contract=replay,
        available_gate_mechanics=gate,
        score_report_bytes=score_report,
        available_gate_report_bytes=gate_report,
        score_report_id=score_report_id,
        available_gate_report_id=gate_report_id,
    )
    monkeypatch.setattr(runtime, "decode_phase2b_scorer_bundle_v1", lambda value: decoded)
    monkeypatch.setattr(
        runtime,
        "evaluate_phase2b_scorer_bundle_v1",
        lambda value: evaluation,
    )
    return score_report, gate_report


def test_fixed_cli_surface_and_exact_filenames() -> None:
    assert runtime.PHASE2B_SCORER_BUNDLE_V1_FILENAME == "phase2b_scorer_bundle_v1.bin"
    assert runtime.PHASE2B_SCORER_SCORE_MECHANICS_REPORT_V1_FILENAME == (
        "score_mechanics_report_v1.json"
    )
    assert runtime.PHASE2B_SCORER_AVAILABLE_GATE_MECHANICS_REPORT_V1_FILENAME == (
        "available_gate_mechanics_report_v1.json"
    )
    assert runtime.SCORER_RUNTIME_CLAIM_LEVEL_V1 == (
        "NON_AUTHORITATIVE_LOCAL_CONTAINER_SCORER_MECHANICS_ONLY"
    )


def test_success_publishes_exact_two_reports_and_stdout_only_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    input_directory.mkdir()
    output_directory.mkdir()
    bundle_payload = b"bounded-scorer-bundle"
    bundle = input_directory / runtime.PHASE2B_SCORER_BUNDLE_V1_FILENAME
    bundle.write_bytes(bundle_payload)
    score_report, gate_report = _install_success(monkeypatch, bundle_payload)

    assert runtime.main(
        ("--input", str(bundle), "--output", str(output_directory))
    ) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.endswith("\n") and captured.out.count("\n") == 1
    receipt = json.loads(captured.out)
    assert captured.out == json.dumps(
        receipt,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"
    assert sorted(item.name for item in output_directory.iterdir()) == [
        runtime.PHASE2B_SCORER_AVAILABLE_GATE_MECHANICS_REPORT_V1_FILENAME,
        runtime.PHASE2B_SCORER_SCORE_MECHANICS_REPORT_V1_FILENAME,
    ]
    assert (
        output_directory / runtime.PHASE2B_SCORER_SCORE_MECHANICS_REPORT_V1_FILENAME
    ).read_bytes() == score_report
    assert (
        output_directory
        / runtime.PHASE2B_SCORER_AVAILABLE_GATE_MECHANICS_REPORT_V1_FILENAME
    ).read_bytes() == gate_report
    # The test temp root may be DrvFS and cannot represent Unix chmod bits; the
    # container extractor independently requires exact 0400.  Keep the runtime
    # operation itself structurally pinned here.
    assert "os.fchmod(descriptor, 0o400)" in SOURCE
    assert receipt["actor_profile_content_id"] == (
        PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID
    )
    assert receipt["actor_profile_canonical_sha256"] == (
        PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256
    )
    assert receipt["scoring_source_sha256"] == (
        "c3c12e9cd72a930b9ca8667aab22c4eb3675ceab6249b0b26ba940bf794e7f11"
    )
    assert {
        name
        for name, value in receipt.items()
        if value is True
    } == {
        "local_container_mechanics_complete",
        "strict_input_prediction_replayed",
        "public_partition_rebuilt",
        "nine_metric_mechanics_materialized",
        "ten_overall_gate_mechanics_materialized",
        "twenty_four_slice_gate_mechanics_materialized",
        "two_unavailable_gate_inputs_retained",
    }
    for name in (
        "reveal_authority_verified",
        "prediction_commit_before_reveal_verified",
        "scoring_performed",
        "prediction_scored",
        "formal_gate_evaluation_performed",
        "actual_960_case_run_verified",
        "effect_evidence",
        "formal_c1_report_verified",
        "c1_exit_evidence",
    ):
        assert receipt[name] is False
    assert not (output_directory / "receipt.json").exists()


def test_fail_closed_on_foreign_output_without_replacing_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    input_directory.mkdir()
    output_directory.mkdir()
    bundle_payload = b"bounded-scorer-bundle"
    bundle = input_directory / runtime.PHASE2B_SCORER_BUNDLE_V1_FILENAME
    bundle.write_bytes(bundle_payload)
    foreign = output_directory / "foreign"
    foreign.write_bytes(b"preserve")
    _install_success(monkeypatch, bundle_payload)

    assert runtime.main(
        ("--input", str(bundle), "--output", str(output_directory))
    ) == 2
    assert capsys.readouterr().out == ""
    assert foreign.read_bytes() == b"preserve"
    assert tuple(output_directory.iterdir()) == (foreign,)


def test_receipt_construction_failure_precedes_report_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    input_directory.mkdir()
    output_directory.mkdir()
    bundle_payload = b"bounded-scorer-bundle"
    bundle = input_directory / runtime.PHASE2B_SCORER_BUNDLE_V1_FILENAME
    bundle.write_bytes(bundle_payload)
    _install_success(monkeypatch, bundle_payload)
    monkeypatch.setattr(
        runtime,
        "_receipt_v1",
        lambda **values: (_ for _ in ()).throw(ValueError("injected receipt failure")),
    )

    assert runtime.main(
        ("--input", str(bundle), "--output", str(output_directory))
    ) == 2
    assert capsys.readouterr().out == ""
    assert list(output_directory.iterdir()) == []


def test_symlink_bundle_and_argument_aliases_fail_closed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    input_directory.mkdir()
    output_directory.mkdir()
    target = input_directory / "target"
    target.write_bytes(b"bundle")
    bundle = input_directory / runtime.PHASE2B_SCORER_BUNDLE_V1_FILENAME
    bundle.symlink_to(target)
    assert runtime.main(
        ("--input", str(bundle), "--output", str(output_directory))
    ) == 2
    assert runtime.main(
        ("--input", str(target), "--input", str(output_directory))
    ) == 2
    assert capsys.readouterr().out == ""


def test_main_has_exact_single_decode_and_evaluate_calls() -> None:
    tree = ast.parse(SOURCE)
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls = [
        node.func.id
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls.count("decode_phase2b_scorer_bundle_v1") == 1
    assert calls.count("evaluate_phase2b_scorer_bundle_v1") == 1
    assert "canonicalize_score_mechanics_report_v1" not in calls
    assert "canonicalize_available_gate_mechanics_report_v1" not in calls
    assert "docker" not in SOURCE.casefold()
    assert "formal_gate_evaluation_performed\": True" not in SOURCE
