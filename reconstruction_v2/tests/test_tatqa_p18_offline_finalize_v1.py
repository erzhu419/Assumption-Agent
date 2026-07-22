from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile

import pytest

from assumption_agent.benchmarks import tatqa_p18_formal_study_v1 as formal_study
from assumption_agent.benchmarks import tatqa_p18_offline_finalize_v1 as offline


@pytest.fixture
def local_tmp() -> Path:
    # DrvFS does not preserve the finalizer's required 0600 output mode.
    root = Path(tempfile.mkdtemp(prefix="tatqa-p18-offline-finalize-", dir="/tmp"))
    try:
        yield root
    finally:
        if root.exists():
            for directory, _children, _files in os.walk(root):
                Path(directory).chmod(0o700)
            shutil.rmtree(root)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _semantic(value: object) -> str:
    return hashlib.sha256(_canonical(value).rstrip(b"\n")).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value))


def _controller_test_module():
    name = "_p18_controller_fixture_for_offline_finalize"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = Path(__file__).with_name("test_tatqa_p18_formal_controller_v1.py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _formal_value(*, promote: bool = True) -> dict[str, object]:
    fixture = _controller_test_module()
    disposition, *_rest = fixture._run(promote=promote)
    return formal_study._terminal_envelope(
        disposition=disposition,
        runtime_fingerprint_self_sha256="a" * 64,
        network_preflight={
            "network_properties": [
                "IPAddressDeny=any",
                "RestrictAddressFamilies=AF_UNIX",
            ],
            "returncode": 0,
            "stdout_sha256": "0" * 64,
            "stderr_sha256": "1" * 64,
        },
    )


def _rehash_formal(value: dict[str, object]) -> None:
    controller = value["controller_disposition"]
    assert isinstance(controller, dict)
    value["controller_disposition_sha256"] = _semantic(controller)
    body = dict(value)
    body.pop("final_disposition_sha256", None)
    value["final_disposition_sha256"] = _semantic(body)


def _reseal_a_hold_transport_change(formal: dict[str, object]) -> None:
    """Propagate an adversarially modified A_hold receipt through outer hashes."""

    artifacts = formal["offline_artifacts"]
    controller = formal["controller_disposition"]
    assert isinstance(artifacts, dict) and isinstance(controller, dict)
    archive = artifacts["A_hold_archive"]
    inference = archive["actual_inference_preparation"]
    inference_body = dict(inference)
    inference_body.pop("preparation_inference_receipt_sha256")
    inference["preparation_inference_receipt_sha256"] = _semantic(inference_body)
    archive_sha = _semantic(archive)
    score = artifacts["A_hold_score"]
    score["archive_sha256"] = archive_sha
    score_sha = _semantic(score)
    epoch = artifacts["epoch_authorization"]
    epoch["A_hold_score_sha256"] = score_sha
    controller["A_hold_archive_sha256"] = archive_sha
    controller["A_hold_score_sha256"] = score_sha
    controller["epoch_authorization_sha256"] = _semantic(epoch)
    _rehash_formal(formal)


def _postflight(block: str, archive_sha256: str) -> dict[str, object]:
    return {
        "schema": "tatqa_p18_formal_controller_v1_runtime_postflight_v1",
        "block": block,
        "archive_sha256": archive_sha256,
        "runtime_ok": True,
        "external_network_calls": 0,
        "api_or_online_evaluator_calls": 0,
        "retry_replay_resample_provider_switch": 0,
        "controller_or_worker_source_reads": 0,
        "controller_or_worker_label_reads": 0,
        "maximum_cpu_threads_per_hippo_process": (
            2 if block in {"A_hold", "M_search"} else 0
        ),
    }


def _control_root(tmp_path: Path, formal: dict[str, object]) -> Path:
    root = tmp_path / "control"
    root.mkdir(mode=0o700)
    _write(root / "formal.disposition.json", formal)
    artifacts = formal["offline_artifacts"]
    assert isinstance(artifacts, dict)
    runtime = "a" * 64
    canary = "b" * 64
    evidence = {
        "A_form_fit": artifacts["A_form_fit"],
        "F_search_policy_freeze": artifacts["policy_freeze"],
        "A_hold_score": artifacts["A_hold_score"],
        "M_search_score": artifacts["M_search_score"],
    }
    for name, payload in evidence.items():
        body = {
            "evidence_sha256": _semantic(payload),
            "name": name,
            "payload": payload,
            "production_canary_sha256": canary,
            "runtime_fingerprint_sha256": runtime,
            "schema": "tatqa_p18_formal_adapters_v1_durable_offline_evidence_v1",
        }
        _write(
            root / "evidence" / f"{name}.json",
            {**body, "durable_evidence_receipt_sha256": _semantic(body)},
        )

    stage_artifacts = {
        "A_form": artifacts["A_form_archive"],
        "F_search": artifacts["F_search_archive"],
        "A_hold": artifacts["A_hold_archive"],
        "M_search": artifacts["M_search_archive"],
    }
    item_counts = {"A_form": 48, "F_search": 36, "A_hold": 30, "M_search": 30}
    expected_postflight = {
        "A_form": artifacts["A_form_fit"]["postflight_sha256"],
        "F_search": artifacts["policy_freeze"]["F_search_postflight_sha256"],
        "A_hold": artifacts["A_hold_score"]["postflight_sha256"],
        "M_search": artifacts["M_search_score"]["postflight_sha256"],
    }
    for block, archive in stage_artifacts.items():
        stage = root / "stages" / block
        inference = archive["actual_inference_preparation"]
        arm_count = 2 if block in {"A_form", "F_search"} else 4
        item_order = [
            archive["logical_action_results"][index * arm_count][
                "item_commitment_sha256"
            ]
            for index in range(item_counts[block])
        ]
        hippo_count = item_counts[block] if block in {"A_hold", "M_search"} else 0
        prep_body = {
            "schema": "tatqa_p18_formal_adapters_v1_block_preparation_receipt_v1",
            "block": block,
            "item_count": item_counts[block],
            "actual_model_future_expected_count": 1 + hippo_count,
            "actual_model_future_submit_count_before_first_join": 1 + hippo_count,
            "all_actual_model_futures_submitted_before_first_join": True,
            "block_view_sha256": "2" * 64,
            "hippo_actual_inference_cap": 8 if hippo_count else 0,
            "hippo_actual_submitted_count": hippo_count,
            "items": [
                {
                    "item_commitment_sha256": item,
                    "prompt_receipt_sha256": "3" * 64,
                    "raw_behavior_sha256": "4" * 64,
                    "tensor_sha256": "5" * 64,
                }
                for item in item_order
            ],
            "minilm_raw_compiled_item_count": item_counts[block],
            "production_canary_sha256": canary,
            "qwen_hippo_dedicated_inference_executors": bool(hippo_count),
            "qwen_hippo_overlap_observed": bool(hippo_count),
            "retry_replay_resample_provider_switch": 0,
            "runtime_fingerprint_sha256": runtime,
            "typed_plan_input_sha256": inference["qwen_transport_receipt"][
                "input_sha256"
            ],
            "typed_plan_output_sha256": inference["qwen_transport_receipt"][
                "output_sha256"
            ],
            "typed_plan_transport_receipt_sha256": inference[
                "qwen_transport_receipt_sha256"
            ],
            "typed_plan_worker_receipt_sha256": inference[
                "qwen_worker_receipt_sha256"
            ],
            "typed_plan_worker_pid": inference["qwen_worker_pid"],
        }
        prep = {**prep_body, "preparation_receipt_sha256": _semantic(prep_body)}
        _write(stage / "block.preparation.json", prep)
        _write(stage / "preparation.inference.json", inference)
        transport_hashes = [
            inference["qwen_transport_receipt_sha256"],
            *inference["hippo_transport_receipt_sha256s"],
        ]
        transport_receipts = [
            inference["qwen_transport_receipt"],
            *inference["hippo_transport_receipts"],
        ]
        worker_pids = [inference["qwen_worker_pid"], *inference["hippo_worker_pids"]]
        shared = {
            "production_canary_sha256": canary,
            "runtime_fingerprint_sha256": runtime,
            "transport_receipt_aggregate_sha256": _semantic(
                {
                    "transport_receipts": transport_receipts,
                    "transport_receipt_sha256s": transport_hashes,
                    "worker_pids": worker_pids,
                }
            ),
            "transport_receipts": transport_receipts,
            "transport_receipt_sha256s": transport_hashes,
            "worker_pids": worker_pids,
        }
        archive_body = {
            "schema": "tatqa_p18_formal_adapters_v1_durable_action_archive_v1",
            "block": block,
            "archive": archive,
            "archive_sha256": _semantic(archive),
            "block_preparation_receipt_sha256": prep["preparation_receipt_sha256"],
            "hippo_worker_receipt_sha256s": inference[
                "hippo_worker_receipt_sha256s"
            ],
            "inference_executors_closed_after_terminal_validation": True,
            "preparation_inference_receipt_sha256": inference[
                "preparation_inference_receipt_sha256"
            ],
            **shared,
        }
        _write(
            stage / "action.archive.json",
            {**archive_body, "durable_archive_receipt_sha256": _semantic(archive_body)},
        )
        postflight = _postflight(block, _semantic(archive))
        assert _semantic(postflight) == expected_postflight[block]
        postflight_body = {
            "schema": "tatqa_p18_formal_adapters_v1_durable_runtime_postflight_v1",
            "block": block,
            "postflight": postflight,
            "postflight_sha256": _semantic(postflight),
            "inference_executors_closed_after_terminal_validation": True,
            "preparation_inference_receipt_sha256": inference[
                "preparation_inference_receipt_sha256"
            ],
            **shared,
        }
        _write(
            stage / "runtime.postflight.json",
            {
                **postflight_body,
                "durable_postflight_receipt_sha256": _semantic(postflight_body),
            },
        )
    return root


def test_exact_recompute_writes_canonical_exclusive_mode_0600(local_tmp: Path) -> None:
    formal = _formal_value()
    disposition = local_tmp / "formal.disposition.json"
    report = local_tmp / "offline.final.json"
    _write(disposition, formal)

    result = offline.finalize_offline(disposition, report)

    assert result["recomputed_status"] == "valid_primary_true"
    assert result["recomputed_A_hold_promoted"] is True
    assert result["recomputed_primary_value"] is True
    assert result["external_network_calls"] == 0
    assert result["formal_source_files_opened"] == 0
    assert result["runtime_receipt_recomputation"]["A_form"][
        "transport_receipt_count"
    ] == 1
    assert result["runtime_receipt_recomputation"]["A_hold"][
        "hippo_transport_receipt_count"
    ] == 30
    assert result["runtime_receipt_recomputation"]["A_hold"][
        "maximum_observed_process_thread_peak"
    ] <= 2
    assert result["runtime_receipt_recomputation"]["A_hold"][
        "overlap_witness_item_commitments"
    ]
    assert result["self_sha256"] == _semantic(
        {key: value for key, value in result.items() if key != "self_sha256"}
    )
    assert report.read_bytes() == _canonical(result)
    assert stat.S_IMODE(report.stat().st_mode) == 0o600
    with pytest.raises(offline.TatqaP18OfflineFinalizeError):
        offline.finalize_offline(disposition, report)


def test_recomputes_valid_nonpromotion_without_releasing_m_search(tmp_path: Path) -> None:
    formal = _formal_value(promote=False)
    disposition = tmp_path / "formal.disposition.json"
    _write(disposition, formal)

    result = offline.recompute_final_disposition(disposition)

    assert result["recomputed_status"] == "valid_nonpromotion"
    assert result["recomputed_A_hold_promoted"] is False
    assert result["recomputed_primary_value"] is False
    assert result["recomputed_hashes"]["M_search_archive_sha256"] is None


def test_rejects_score_tamper_even_after_outer_hashes_are_resealed(tmp_path: Path) -> None:
    formal = _formal_value()
    artifacts = formal["offline_artifacts"]
    controller = formal["controller_disposition"]
    assert isinstance(artifacts, dict) and isinstance(controller, dict)
    score = artifacts["A_hold_score"]
    score["item_exact_utility_rows"][0]["arm_utilities"]["RAW"] = {
        "numerator": 0,
        "denominator": 1,
    }
    controller["A_hold_score_sha256"] = _semantic(score)
    _rehash_formal(formal)
    disposition = tmp_path / "formal.disposition.json"
    _write(disposition, formal)

    with pytest.raises(
        offline.TatqaP18OfflineFinalizeError,
        match="item utility replay drifted",
    ):
        offline.recompute_final_disposition(disposition)


def test_control_root_replays_evidence_stage_and_final_chains_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    formal = _formal_value()
    control = _control_root(tmp_path, formal)
    disposition = control / "formal.disposition.json"

    result = offline.recompute_final_disposition(
        disposition, control_root=control
    )

    assert result["control_root_audit"]["verified"] is True
    assert set(result["control_root_audit"]["evidence_receipts"]) == {
        "A_form_fit",
        "F_search_policy_freeze",
        "A_hold_score",
        "M_search_score",
    }
    assert set(result["control_root_audit"]["stage_receipts"]) == {
        "A_form",
        "F_search",
        "A_hold",
        "M_search",
    }

    evidence_path = control / "evidence" / "A_hold_score.json"
    evidence = json.loads(evidence_path.read_text(encoding="ascii"))
    evidence["production_canary_sha256"] = "e" * 64
    _write(evidence_path, evidence)
    with pytest.raises(offline.TatqaP18OfflineFinalizeError):
        offline.recompute_final_disposition(disposition, control_root=control)


@pytest.mark.parametrize(
    "tamper",
    (
        "qwen_interval",
        "qwen_input_hash_shape",
        "qwen_item_count_float",
        "qwen_batch_size_float",
        "qwen_closure_bool",
        "qwen_closure_float",
        "hippo_thread_peak",
        "hippo_cpu_threads_float",
        "hippo_tasks_float",
        "hippo_reservation_bool",
        "hippo_worker_threads_float",
        "hippo_policy_tasks_float",
        "hippo_policy_pid_float",
        "hippo_policy_group",
        "hippo_closure_threads_float",
        "hippo_isolation",
        "hippo_full_hash",
        "overlap_witness",
    ),
)
def test_rejects_full_transport_interval_thread_hash_and_overlap_tamper_after_reseal(
    tmp_path: Path,
    tamper: str,
) -> None:
    formal = _formal_value()
    artifacts = formal["offline_artifacts"]
    assert isinstance(artifacts, dict)
    inference = artifacts["A_hold_archive"]["actual_inference_preparation"]
    if tamper == "qwen_interval":
        qwen = inference["qwen_transport_receipt"]
        qwen["model_execution_finished_monotonic_ns"] = qwen[
            "model_execution_started_monotonic_ns"
        ]
        inference["qwen_transport_receipt_sha256"] = _semantic(qwen)
    elif tamper == "qwen_input_hash_shape":
        qwen = inference["qwen_transport_receipt"]
        qwen["input_sha256"] = "not-a-sha256"
        inference["qwen_transport_receipt_sha256"] = _semantic(qwen)
    elif tamper == "qwen_item_count_float":
        qwen = inference["qwen_transport_receipt"]
        qwen["item_count"] = float(qwen["item_count"])
        inference["qwen_transport_receipt_sha256"] = _semantic(qwen)
    elif tamper == "qwen_batch_size_float":
        qwen = inference["qwen_transport_receipt"]
        qwen["batch_size"] = 4.0
        inference["qwen_transport_receipt_sha256"] = _semantic(qwen)
    elif tamper == "qwen_closure_bool":
        qwen = inference["qwen_transport_receipt"]
        qwen["systemd_unit_closure"]["main_pid"] = False
        inference["qwen_transport_receipt_sha256"] = _semantic(qwen)
    elif tamper == "qwen_closure_float":
        qwen = inference["qwen_transport_receipt"]
        qwen["systemd_unit_closure"]["control_group_process_count"] = 0.0
        inference["qwen_transport_receipt_sha256"] = _semantic(qwen)
    elif tamper == "hippo_thread_peak":
        hippo = inference["hippo_transport_receipts"][0]
        hippo["observed_process_thread_peak"] = 3
        inference["hippo_transport_receipt_sha256s"][0] = _semantic(hippo)
    elif tamper == "hippo_cpu_threads_float":
        hippo = inference["hippo_transport_receipts"][0]
        hippo["CPU_threads"] = 2.0
        inference["hippo_transport_receipt_sha256s"][0] = _semantic(hippo)
    elif tamper == "hippo_tasks_float":
        hippo = inference["hippo_transport_receipts"][0]
        hippo["systemd_tasks_max"] = 3.0
        inference["hippo_transport_receipt_sha256s"][0] = _semantic(hippo)
    elif tamper == "hippo_reservation_bool":
        hippo = inference["hippo_transport_receipts"][0]
        hippo["thread_monitor_process_reservation"] = True
        inference["hippo_transport_receipt_sha256s"][0] = _semantic(hippo)
    elif tamper == "hippo_worker_threads_float":
        hippo = inference["hippo_transport_receipts"][0]
        hippo["maximum_worker_process_threads"] = 2.0
        inference["hippo_transport_receipt_sha256s"][0] = _semantic(hippo)
    elif tamper == "hippo_policy_tasks_float":
        hippo = inference["hippo_transport_receipts"][0]
        hippo["systemd_start_policy"]["tasks_max"] = 3.0
        hippo["systemd_start_policy_sha256"] = _semantic(
            hippo["systemd_start_policy"]
        )
        inference["hippo_transport_receipt_sha256s"][0] = _semantic(hippo)
    elif tamper == "hippo_policy_pid_float":
        hippo = inference["hippo_transport_receipts"][0]
        hippo["systemd_start_policy"]["main_pid"] = float(hippo["worker_pid"])
        hippo["systemd_start_policy_sha256"] = _semantic(
            hippo["systemd_start_policy"]
        )
        inference["hippo_transport_receipt_sha256s"][0] = _semantic(hippo)
    elif tamper == "hippo_policy_group":
        hippo = inference["hippo_transport_receipts"][0]
        hippo["systemd_start_policy"]["control_group_sha256"] = "f" * 64
        hippo["systemd_start_policy_sha256"] = _semantic(
            hippo["systemd_start_policy"]
        )
        inference["hippo_transport_receipt_sha256s"][0] = _semantic(hippo)
    elif tamper == "hippo_closure_threads_float":
        hippo = inference["hippo_transport_receipts"][0]
        hippo["systemd_unit_closure"]["control_group_thread_count"] = 0.0
        inference["hippo_transport_receipt_sha256s"][0] = _semantic(hippo)
    elif tamper == "hippo_isolation":
        hippo = inference["hippo_transport_receipts"][0]
        hippo["visible_GPU"] = "0"
        inference["hippo_transport_receipt_sha256s"][0] = _semantic(hippo)
    elif tamper == "hippo_full_hash":
        inference["hippo_transport_receipt_sha256s"][0] = "f" * 64
    else:
        hippo = inference["hippo_transport_receipts"][0]
        hippo["model_execution_started_monotonic_ns"] = 11_000
        hippo["model_execution_finished_monotonic_ns"] = 12_000
        inference["hippo_transport_receipt_sha256s"][0] = _semantic(hippo)
        # Keep the old declared witnesses: independent interval intersection
        # must detect that this item is no longer an overlap witness.
    _reseal_a_hold_transport_change(formal)
    disposition = tmp_path / f"{tamper}.formal.disposition.json"
    _write(disposition, formal)

    with pytest.raises(offline.TatqaP18OfflineFinalizeError):
        offline.recompute_final_disposition(disposition)


def test_rejects_resealed_durable_full_receipt_drift_from_inference(
    tmp_path: Path,
) -> None:
    formal = _formal_value()
    control = _control_root(tmp_path, formal)
    stage = control / "stages" / "A_hold"
    for filename, self_field in (
        ("action.archive.json", "durable_archive_receipt_sha256"),
        ("runtime.postflight.json", "durable_postflight_receipt_sha256"),
    ):
        path = stage / filename
        envelope = json.loads(path.read_text(encoding="ascii"))
        envelope["transport_receipts"][1]["input_file_sha256"] = "e" * 64
        envelope["transport_receipt_sha256s"][1] = _semantic(
            envelope["transport_receipts"][1]
        )
        envelope["transport_receipt_aggregate_sha256"] = _semantic(
            {
                "transport_receipts": envelope["transport_receipts"],
                "transport_receipt_sha256s": envelope[
                    "transport_receipt_sha256s"
                ],
                "worker_pids": envelope["worker_pids"],
            }
        )
        body = dict(envelope)
        body.pop(self_field)
        envelope[self_field] = _semantic(body)
        _write(path, envelope)

    with pytest.raises(offline.TatqaP18OfflineFinalizeError):
        offline.recompute_final_disposition(
            control / "formal.disposition.json", control_root=control
        )


def test_cli_has_no_source_network_provider_or_retry_surface() -> None:
    options = {action.dest for action in offline._parser()._actions}
    assert options == {"help", "formal_disposition", "output", "control_root"}
    source = Path(offline.__file__).read_text(encoding="utf-8")
    assert "requests" not in source
    assert "urllib" not in source
    assert "socket" not in source
    assert "source-data loader" in source
