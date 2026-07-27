from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Mapping

import pytest

from replication_runtime.bioasq_coordinate_scorer_v1 import (
    contract as coordinate_contract,
)
from replication_runtime.bioasq_p1_formal_v1 import contract as runtime
from replication_runtime.dstc9_official_hipporag_v1 import runtime_binding


HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
HEX_D = "d" * 64


@pytest.fixture
def tmp_path() -> Path:
    """Use the Linux filesystem because NTFS does not preserve Unix modes."""

    root = Path(tempfile.mkdtemp(prefix="bioasq-runtime-", dir="/tmp"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _write_canonical(
    path: Path,
    value: object,
    *,
    mode: int = 0o600,
) -> str:
    raw = runtime.canonical_bytes(value, newline=True)
    path.write_bytes(raw)
    os.chmod(path, mode)
    return hashlib.sha256(raw).hexdigest()


def _coordinate_config(tmp_path: Path) -> runtime.CoordinateCanaryConfig:
    return runtime.CoordinateCanaryConfig(
        canary_root=(tmp_path / "canary").absolute(),
        canary_binding_sha256=HEX_A,
        runtime_python=(tmp_path / "python").absolute(),
        project_root=(tmp_path / "project").absolute(),
        minilm_asset_manifest=(tmp_path / "minilm.json").absolute(),
        minilm_model_root=(tmp_path / "minilm").absolute(),
        cross_encoder_model_root=(tmp_path / "cross").absolute(),
        timeout_seconds=123,
    )


def _fake_coordinate_run(**kwargs: object) -> dict[str, object]:
    scorer_input = coordinate_contract.validate_input(kwargs["input_value"])
    zeros = tuple(0 for _ in range(coordinate_contract.CORPUS_SIZE))
    score_rows = [
        {
            name: zeros
            for name in coordinate_contract.SCORE_NAMES
        }
        for _ in scorer_input.queries
    ]
    return coordinate_contract.make_output(
        scorer_input=scorer_input,
        score_rows=score_rows,
        model_binding_sha256=HEX_B,
    )


def _fake_hardware_capture(**kwargs: object) -> dict[str, object]:
    return runtime_binding.make_current_study_hardware_binding(
        study_id=str(kwargs["study_id"]),
        capture_id=str(kwargs["capture_id"]),
        gpus=runtime_binding.EXPECTED_GPU_ROWS,
        nvidia_driver_version="595.84",
        kernel_release="7.0.0-28-generic",
    )


def test_coordinate_canary_runs_once_and_publishes_no_vectors(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def run(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return _fake_coordinate_run(**kwargs)

    config = _coordinate_config(tmp_path)
    receipt = runtime.run_source_free_coordinate_canary_once(
        config,
        run_callable=run,
        hardware_capture_callable=_fake_hardware_capture,
    )

    assert len(calls) == 1
    scorer_input = coordinate_contract.validate_input(
        calls[0]["input_value"]
    )
    assert len(scorer_input.passages) == 2_900
    assert len(scorer_input.queries) == 1
    assert len({row.text for row in scorer_input.passages}) == 5
    assert receipt["formal_source_access_count"] == 0
    assert receipt["formal_action_count"] == 0
    assert receipt["formal_score_count"] == 0
    assert receipt["formal_evaluator_count"] == 0
    assert receipt["current_hardware_capture_count"] == 1
    assert receipt["current_hardware_binding_study_id"] == runtime.STUDY_ID
    assert receipt["minilm_constructor_canary_encode_call_count"] == 2
    assert receipt["minilm_formal_batch_encode_call_count"] == 1
    assert receipt["minilm_total_encode_call_count"] == 3
    assert receipt["retry_count"] == 0
    assert receipt["private_vector_values_published"] is False
    assert "rows" not in receipt
    assert "vectors" not in json.dumps(receipt)
    persisted = (
        config.canary_root / runtime.CANARY_RECEIPT_FILENAME
    ).read_bytes()
    assert persisted == runtime.canonical_bytes(receipt, newline=True)
    assert stat.S_IMODE(
        (config.canary_root / runtime.CANARY_RECEIPT_FILENAME)
        .stat()
        .st_mode
    ) == 0o600
    hardware_path = (
        config.canary_root / runtime.CURRENT_HARDWARE_FILENAME
    )
    assert hardware_path.is_file()
    assert stat.S_IMODE(hardware_path.stat().st_mode) == 0o600
    assert (
        hashlib.sha256(hardware_path.read_bytes()).hexdigest()
        == receipt["current_hardware_binding_file_sha256"]
    )


def test_coordinate_canary_is_one_shot(tmp_path: Path) -> None:
    config = _coordinate_config(tmp_path)
    runtime.run_source_free_coordinate_canary_once(
        config,
        run_callable=_fake_coordinate_run,
        hardware_capture_callable=_fake_hardware_capture,
    )
    with pytest.raises(runtime.BioasqP1FormalRuntimeError):
        runtime.run_source_free_coordinate_canary_once(
            config,
            run_callable=_fake_coordinate_run,
            hardware_capture_callable=_fake_hardware_capture,
        )


def test_coordinate_canary_rejects_worker_count_drift(
    tmp_path: Path,
) -> None:
    def drift(**kwargs: object) -> dict[str, object]:
        output = _fake_coordinate_run(**kwargs)
        output["receipt"]["minilm_total_encode_call_count"] = 4
        return output

    config = _coordinate_config(tmp_path)
    with pytest.raises(
        runtime.BioasqP1FormalRuntimeError,
        match="worker counters",
    ):
        runtime.run_source_free_coordinate_canary_once(
            config,
            run_callable=drift,
            hardware_capture_callable=_fake_hardware_capture,
        )
    failure = json.loads(
        (
            config.canary_root / runtime.CANARY_RECEIPT_FILENAME
        ).read_text("ascii")
    )
    assert failure["status"] == (
        "failed_source_free_coordinate_canary_no_retry"
    )
    assert failure["formal_source_access_count"] == 0
    assert failure["retry_count"] == 0


def test_strict_canonical_config_round_trip_and_extra_key_rejection(
    tmp_path: Path,
) -> None:
    paths = {
        "canary_root": str((tmp_path / "canary").absolute()),
        "cross_encoder_model_root": str((tmp_path / "cross").absolute()),
        "minilm_asset_manifest": str((tmp_path / "minilm.json").absolute()),
        "minilm_model_root": str((tmp_path / "minilm").absolute()),
        "project_root": str((tmp_path / "project").absolute()),
        "runtime_python": str((tmp_path / "python").absolute()),
    }
    body = {
        **paths,
        "canary_binding_sha256": HEX_A,
        "schema": runtime.COORDINATE_CANARY_CONFIG_SCHEMA,
        "timeout_seconds": 123,
    }
    value = runtime.with_self_hash(body)
    config_path = tmp_path / "canary.config.json"
    _write_canonical(config_path, value)
    loaded = runtime.load_runtime_config(config_path.absolute())
    assert isinstance(loaded, runtime.CoordinateCanaryConfig)
    assert loaded.canary_binding_sha256 == HEX_A

    invalid_body = {
        **body,
        "api_key": "must-never-be-configurable",
    }
    invalid = runtime.with_self_hash(invalid_body)
    invalid_path = tmp_path / "invalid.config.json"
    _write_canonical(invalid_path, invalid)
    with pytest.raises(
        runtime.BioasqP1FormalRuntimeError,
        match="schema drifted",
    ):
        runtime.load_runtime_config(invalid_path.absolute())

    noncanonical_path = tmp_path / "noncanonical.config.json"
    noncanonical_path.write_text(
        json.dumps(value, indent=2),
        encoding="ascii",
    )
    with pytest.raises(
        runtime.BioasqP1FormalRuntimeError,
        match="not canonical",
    ):
        runtime.load_runtime_config(noncanonical_path.absolute())


def test_config_rejects_relative_paths_and_bad_hashes() -> None:
    with pytest.raises(runtime.BioasqP1FormalRuntimeError):
        runtime.CoordinateCanaryConfig(
            canary_root=Path("relative"),
            canary_binding_sha256=HEX_A,
            runtime_python=Path("/python"),
            project_root=Path("/project"),
            minilm_asset_manifest=Path("/manifest"),
            minilm_model_root=Path("/minilm"),
            cross_encoder_model_root=Path("/cross"),
        )
    with pytest.raises(runtime.BioasqP1FormalRuntimeError):
        runtime.FormalPreflightConfig(
            execution_binding_sha256="bad",
            coordinate_canary_binding_sha256=HEX_A,
            coordinate_canary_receipt_path=Path("/coord"),
            coordinate_canary_receipt_file_sha256=HEX_B,
            coordinate_canary_receipt_self_sha256=HEX_C,
            bioasq_hardware_binding_path=Path("/bioasq-hardware"),
            bioasq_hardware_binding_file_sha256=HEX_B,
            bioasq_hardware_binding_self_sha256=HEX_C,
            legacy_hippo_canary_receipt_path=Path("/hippo"),
            legacy_hardware_binding_path=Path("/hardware"),
            coordinate_project_root=Path("/coordinate"),
            hippo_worker_project_root=Path("/hippo-project"),
            hippo_runtime_python=Path("/hippo-python"),
            hippo_local_llm_model=Path("/hippo-llm"),
            hippo_local_embedding_model=Path("/hippo-embedding"),
            hippo_runtime_fingerprint_path=Path("/fingerprint"),
        )


def test_current_generic_hippo_backend_matches_frozen_hashes() -> None:
    project_root = Path(__file__).resolve().parents[1]
    observed = runtime.verify_generic_hippo_backend(project_root)
    assert observed == runtime.GENERIC_HIPPO_BACKEND_SHA256
    coordinate = runtime.verify_coordinate_backend(project_root)
    assert coordinate == runtime.COORDINATE_BACKEND_SHA256


def _coordinate_safe_receipt(
    binding: str,
    *,
    hardware_file_sha256: str = HEX_B,
    hardware_self_sha256: str = HEX_C,
) -> dict[str, object]:
    return runtime.with_self_hash(
        {
            "aggregate_only_public_receipt": True,
            "canary_attempt_self_sha256": HEX_A,
            "canary_binding_sha256": binding,
            "coordinate_gpu": 1,
            "coordinate_worker_count": 1,
            "current_hardware_binding_file_sha256": (
                hardware_file_sha256
            ),
            "current_hardware_binding_self_sha256": (
                hardware_self_sha256
            ),
            "current_hardware_binding_study_id": runtime.STUDY_ID,
            "current_hardware_capture_count": 1,
            "formal_action_count": 0,
            "formal_evaluator_count": 0,
            "formal_score_count": 0,
            "formal_source_access_count": 0,
            "minilm_constructor_canary_encode_call_count": 2,
            "minilm_formal_batch_encode_call_count": 1,
            "minilm_total_encode_call_count": 3,
            "model_binding_sha256": HEX_B,
            "online_or_API_evaluator_calls": 0,
            "private_output_self_sha256": HEX_C,
            "private_score_bundle_sha256": HEX_D,
            "private_vector_values_published": False,
            "query_count": 1,
            "retry_count": 0,
            "schema": runtime.COORDINATE_CANARY_SCHEMA,
            "status": "passed_source_free_coordinate_canary_once",
            "study_id": runtime.STUDY_ID,
            "synthetic_corpus_count": 2_900,
            "synthetic_unique_passage_text_count": 5,
            "worker_receipt_sha256": HEX_A,
        }
    )


def _legacy_hardware_receipt() -> dict[str, object]:
    return runtime.with_self_hash(
        {
            "capture_id": "dstc9-v5",
            "hardware": {
                "GPUs": [{"index": 0}, {"index": 1}],
                "NVIDIA_driver_version": "595.84",
                "kernel_release": "7.0.0-28",
            },
            "schema": runtime_binding.CURRENT_HARDWARE_SCHEMA,
            "source_free_boundary": {
                "capture_scope": (
                    "hardware_only_no_model_source_or_evaluator_action_v1"
                ),
                "external_network_call_count": 0,
                "formal_source_open_count": 0,
                "old_P17_driver_or_kernel_used_as_requirement": False,
            },
            "status": runtime_binding.CURRENT_HARDWARE_STATUS,
            "study_id": runtime.LEGACY_HIPPO_STUDY_ID,
        }
    )


def _bioasq_hardware_receipt(
    legacy: Mapping[str, object],
) -> dict[str, object]:
    return runtime.with_self_hash(
        {
            "capture_id": "bioasq-p1",
            "hardware": legacy["hardware"],
            "schema": runtime_binding.CURRENT_HARDWARE_SCHEMA,
            "source_free_boundary": {
                "capture_scope": (
                    "hardware_only_no_model_source_or_evaluator_action_v1"
                ),
                "external_network_call_count": 0,
                "formal_source_open_count": 0,
                "old_P17_driver_or_kernel_used_as_requirement": False,
            },
            "status": runtime_binding.CURRENT_HARDWARE_STATUS,
            "study_id": runtime.STUDY_ID,
        }
    )


def _legacy_canary_receipt() -> dict[str, object]:
    return runtime.with_self_hash(
        {
            "current_hardware_binding_file_sha256": (
                runtime.LEGACY_HARDWARE_FILE_SHA256
            ),
            "current_hardware_binding_self_sha256": (
                runtime.LEGACY_HARDWARE_SELF_SHA256
            ),
            "formal_source_access_count": 0,
            "hipporag_build_count": 1,
            "hipporag_retrieve_count": 1,
            "online_or_API_evaluator_calls": 0,
            "retry_count": 0,
            "schema": (
                "dstc9_p1_source_free_infrastructure_canary_receipt_v2"
            ),
            "status": "passed_source_free_two_lane_canary_once",
            "study_id": runtime.LEGACY_HIPPO_STUDY_ID,
        }
    )


def test_formal_preflight_reuses_exact_legacy_canary_without_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hardware = _legacy_hardware_receipt()
    hardware_path = tmp_path / "hardware.json"
    hardware_file_hash = _write_canonical(hardware_path, hardware)
    bioasq_hardware = _bioasq_hardware_receipt(hardware)
    bioasq_hardware_path = tmp_path / "bioasq-hardware.json"
    bioasq_hardware_file_hash = _write_canonical(
        bioasq_hardware_path, bioasq_hardware
    )
    coordinate = _coordinate_safe_receipt(
        HEX_A,
        hardware_file_sha256=bioasq_hardware_file_hash,
        hardware_self_sha256=bioasq_hardware["self_sha256"],
    )
    coordinate_path = tmp_path / "coordinate.json"
    coordinate_file_hash = _write_canonical(coordinate_path, coordinate)
    monkeypatch.setattr(
        runtime, "LEGACY_HARDWARE_FILE_SHA256", hardware_file_hash
    )
    monkeypatch.setattr(
        runtime,
        "LEGACY_HARDWARE_SELF_SHA256",
        hardware["self_sha256"],
    )

    legacy = _legacy_canary_receipt()
    # The helper reads the monkeypatched hardware constants.
    legacy["current_hardware_binding_file_sha256"] = hardware_file_hash
    legacy["current_hardware_binding_self_sha256"] = hardware["self_sha256"]
    legacy.pop("self_sha256")
    legacy = runtime.with_self_hash(legacy)
    legacy_path = tmp_path / "legacy.json"
    legacy_file_hash = _write_canonical(legacy_path, legacy)
    monkeypatch.setattr(
        runtime, "LEGACY_HIPPO_CANARY_FILE_SHA256", legacy_file_hash
    )
    monkeypatch.setattr(
        runtime,
        "LEGACY_HIPPO_CANARY_SELF_SHA256",
        legacy["self_sha256"],
    )

    calls: list[dict[str, object]] = []
    live_binding = {
        "hardware": bioasq_hardware["hardware"],
        "receipt_file_sha256": bioasq_hardware_file_hash,
        "receipt_self_sha256": bioasq_hardware["self_sha256"],
        "study_id": runtime.STUDY_ID,
    }

    def verify_hardware(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return dict(live_binding)

    def verify_closure(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return runtime.with_self_hash(
            {
                "current_hardware_binding": dict(live_binding),
                "schema": runtime_binding.SCHEMA,
                "status": (
                    "verified_P17_reused_dependency_closure_with_"
                    "separate_current_hardware_binding"
                ),
            }
        )

    project_root = Path(__file__).resolve().parents[1]
    config = runtime.FormalPreflightConfig(
        execution_binding_sha256=HEX_D,
        coordinate_canary_binding_sha256=HEX_A,
        coordinate_canary_receipt_path=coordinate_path.absolute(),
        coordinate_canary_receipt_file_sha256=coordinate_file_hash,
        coordinate_canary_receipt_self_sha256=coordinate["self_sha256"],
        bioasq_hardware_binding_path=bioasq_hardware_path.absolute(),
        bioasq_hardware_binding_file_sha256=(
            bioasq_hardware_file_hash
        ),
        bioasq_hardware_binding_self_sha256=(
            bioasq_hardware["self_sha256"]
        ),
        legacy_hippo_canary_receipt_path=legacy_path.absolute(),
        legacy_hardware_binding_path=hardware_path.absolute(),
        coordinate_project_root=project_root,
        hippo_worker_project_root=project_root,
        hippo_runtime_python=(tmp_path / "hippo-python").absolute(),
        hippo_local_llm_model=(tmp_path / "hippo-llm").absolute(),
        hippo_local_embedding_model=(
            tmp_path / "hippo-embedding"
        ).absolute(),
        hippo_runtime_fingerprint_path=(
            tmp_path / "fingerprint.json"
        ).absolute(),
    )
    receipt = runtime.verify_formal_preflight(
        config,
        hardware_verify_callable=verify_hardware,
        closure_verify_callable=verify_closure,
    )
    assert len(calls) == 2
    assert receipt["status"] == (
        "passed_offline_preformal_infrastructure_binding"
    )
    assert receipt["formal_source_access_count"] == 0
    assert receipt["legacy_hippo_canary_rerun_count"] == 0
    assert receipt[
        "generic_backend_reused_despite_legacy_benchmark_label"
    ] is True
    assert receipt["coordinate_model_binding_sha256"] == HEX_B
    assert receipt[
        "bioasq_hardware_matches_legacy_canary_hardware"
    ] is True
    assert receipt["live_hardware_matches_qualified_binding"] is True


def test_formal_preflight_rejects_live_hardware_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hardware = _legacy_hardware_receipt()
    hardware_path = tmp_path / "hardware.json"
    hardware_file_hash = _write_canonical(hardware_path, hardware)
    bioasq_hardware = _bioasq_hardware_receipt(hardware)
    bioasq_hardware_path = tmp_path / "bioasq-hardware.json"
    bioasq_hardware_file_hash = _write_canonical(
        bioasq_hardware_path, bioasq_hardware
    )
    coordinate = _coordinate_safe_receipt(
        HEX_A,
        hardware_file_sha256=bioasq_hardware_file_hash,
        hardware_self_sha256=bioasq_hardware["self_sha256"],
    )
    coordinate_path = tmp_path / "coordinate.json"
    coordinate_file_hash = _write_canonical(coordinate_path, coordinate)
    monkeypatch.setattr(
        runtime, "LEGACY_HARDWARE_FILE_SHA256", hardware_file_hash
    )
    monkeypatch.setattr(
        runtime,
        "LEGACY_HARDWARE_SELF_SHA256",
        hardware["self_sha256"],
    )
    legacy = _legacy_canary_receipt()
    legacy["current_hardware_binding_file_sha256"] = hardware_file_hash
    legacy["current_hardware_binding_self_sha256"] = hardware["self_sha256"]
    legacy.pop("self_sha256")
    legacy = runtime.with_self_hash(legacy)
    legacy_path = tmp_path / "legacy.json"
    legacy_file_hash = _write_canonical(legacy_path, legacy)
    monkeypatch.setattr(
        runtime, "LEGACY_HIPPO_CANARY_FILE_SHA256", legacy_file_hash
    )
    monkeypatch.setattr(
        runtime,
        "LEGACY_HIPPO_CANARY_SELF_SHA256",
        legacy["self_sha256"],
    )

    project_root = Path(__file__).resolve().parents[1]
    config = runtime.FormalPreflightConfig(
        execution_binding_sha256=HEX_D,
        coordinate_canary_binding_sha256=HEX_A,
        coordinate_canary_receipt_path=coordinate_path.absolute(),
        coordinate_canary_receipt_file_sha256=coordinate_file_hash,
        coordinate_canary_receipt_self_sha256=coordinate["self_sha256"],
        bioasq_hardware_binding_path=bioasq_hardware_path.absolute(),
        bioasq_hardware_binding_file_sha256=(
            bioasq_hardware_file_hash
        ),
        bioasq_hardware_binding_self_sha256=(
            bioasq_hardware["self_sha256"]
        ),
        legacy_hippo_canary_receipt_path=legacy_path.absolute(),
        legacy_hardware_binding_path=hardware_path.absolute(),
        coordinate_project_root=project_root,
        hippo_worker_project_root=project_root,
        hippo_runtime_python=(tmp_path / "hippo-python").absolute(),
        hippo_local_llm_model=(tmp_path / "hippo-llm").absolute(),
        hippo_local_embedding_model=(
            tmp_path / "hippo-embedding"
        ).absolute(),
        hippo_runtime_fingerprint_path=(
            tmp_path / "fingerprint.json"
        ).absolute(),
    )
    with pytest.raises(
        runtime.BioasqP1FormalRuntimeError,
        match="hardware verification drifted",
    ):
        runtime.verify_formal_preflight(
            config,
            hardware_verify_callable=lambda **_: {
                "hardware": {"GPUs": []},
                "receipt_file_sha256": bioasq_hardware_file_hash,
                "receipt_self_sha256": bioasq_hardware["self_sha256"],
                "study_id": runtime.STUDY_ID,
            },
        )


def test_formal_preflight_rejects_coordinate_canary_binding_drift(
    tmp_path: Path,
) -> None:
    coordinate = _coordinate_safe_receipt(HEX_A)
    path = tmp_path / "coordinate.json"
    file_hash = _write_canonical(path, coordinate)
    loaded = runtime._load_exact_receipt(
        path.absolute(),
        expected_file_sha256=file_hash,
        expected_self_sha256=coordinate["self_sha256"],
        field="coordinate",
    )
    with pytest.raises(
        runtime.BioasqP1FormalRuntimeError,
        match="counters drifted",
    ):
        runtime._validate_coordinate_canary_receipt(
            loaded,
            binding_sha256=HEX_B,
            hardware_file_sha256=HEX_B,
            hardware_self_sha256=HEX_C,
        )
