from __future__ import annotations

import json
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import mmqa_p1_local_runtime_preflight_v1 as preflight


def _inventory_payload() -> dict[str, object]:
    return {
        "cuda_device_count": 2,
        "cuda_devices": [
            {
                "index": 0,
                "memory_total_bytes": 8_164_474_880,
                "name": "NVIDIA GeForce RTX 2080",
            },
            {
                "index": 1,
                "memory_total_bytes": 8_164_474_880,
                "name": "NVIDIA GeForce RTX 2080",
            },
        ],
        "executable_resolved_file_sha256": (
            preflight.EXPECTED_TYPED_PYTHON_RESOLVED_SHA256
        ),
        "mode": "inventory",
        "pyvenv_cfg_sha256": preflight.EXPECTED_TYPED_PYVENV_CFG_SHA256,
        "python_version": preflight.EXPECTED_PYTHON_VERSION,
        "runtime_versions": dict(preflight.EXPECTED_RUNTIME_VERSIONS),
        "schema": preflight.WORKER_SCHEMA,
        "torch_cuda_version": preflight.EXPECTED_TORCH_CUDA_VERSION,
        "transformers_import_version": preflight.EXPECTED_RUNTIME_VERSIONS[
            "transformers"
        ],
    }


def _minilm_payload() -> dict[str, object]:
    return {
        "all_finite": True,
        "all_rows_l2_normalized": True,
        "coordinates_float64_hex": [
            value.hex() for value in (0.93, 0.86, 0.42, 0.31, 0.90, 0.82, 0.37, 0.22)
        ],
        "fixture_sha256": preflight.PUBLIC_SYNTHETIC_FIXTURE_SHA256,
        "matrix_little_endian_float32_sha256": "a" * 64,
        "matrix_shape": [9, 384],
        "mode": "minilm",
        "process_concurrency": 1,
        "repeat_count": 2,
        "repeat_exact": True,
        "schema": preflight.WORKER_SCHEMA,
    }


def _ce_payload() -> dict[str, object]:
    return {
        "all_finite": True,
        "coordinates_float64_hex": [
            value.hex() for value in (0.96, 0.88, 0.44, 0.28, 0.92, 0.84, 0.33, 0.17)
        ],
        "fixture_sha256": preflight.PUBLIC_SYNTHETIC_FIXTURE_SHA256,
        "logit_little_endian_float32_sha256": "b" * 64,
        "logit_shape": [8, 1],
        "mode": "cross_encoder",
        "process_concurrency": 1,
        "repeat_count": 2,
        "repeat_exact": True,
        "schema": preflight.WORKER_SCHEMA,
    }


def _typed_binding() -> dict[str, object]:
    return {
        "executable_resolved_file_sha256": (
            preflight.EXPECTED_TYPED_PYTHON_RESOLVED_SHA256
        ),
        "lexical_path_sha256": "c" * 64,
        "pyvenv_cfg_sha256": preflight.EXPECTED_TYPED_PYVENV_CFG_SHA256,
    }


def _asset_binding(tree: str) -> dict[str, object]:
    return {
        "auxiliary_file_count": 2,
        "auxiliary_tree_sha256": "d" * 64,
        "required_file_count": 1,
        "required_size_bytes": 1,
        "required_tree_sha256": tree,
    }


def _install_fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        preflight,
        "production_address_family_isolation_probe",
        lambda: {
            "AF_INET6_socket_creation_errno": "EAFNOSUPPORT",
            "AF_INET_socket_creation_errno": "EAFNOSUPPORT",
            "address_family_isolation_contract": (
                preflight.ADDRESS_FAMILY_ISOLATION_CONTRACT
            ),
            "denied_family_count": 2,
            "probe_count": 2,
            "status": "AF_INET_and_AF_INET6_socket_creation_denied",
        },
    )
    monkeypatch.setattr(preflight, "_verify_typed_python", lambda _path: _typed_binding())
    monkeypatch.setattr(
        preflight,
        "_verify_minilm_asset",
        lambda _path: _asset_binding(preflight.MINILM_REQUIRED_TREE_SHA256),
    )
    monkeypatch.setattr(
        preflight,
        "_verify_ce_asset",
        lambda _path: _asset_binding(preflight.CE_REQUIRED_TREE_SHA256),
    )
    monkeypatch.setattr(
        preflight,
        "_probe_gpu_rows",
        lambda _path: tuple(dict(row) for row in preflight.EXPECTED_GPU_ROWS),
    )

    def fake_worker(**kwargs: object) -> dict[str, object]:
        mode = kwargs["mode"]
        if mode == "inventory":
            assert kwargs["physical_gpu"] == "0,1"
            assert kwargs.get("model_root") is None
            return _inventory_payload()
        if mode == "minilm":
            assert kwargs["physical_gpu"] == "0"
            return _minilm_payload()
        assert mode == "cross_encoder"
        assert kwargs["physical_gpu"] == "1"
        return _ce_payload()

    monkeypatch.setattr(preflight, "_invoke_worker", fake_worker)


def test_static_asset_contracts_and_fixture_hash_are_exact() -> None:
    preflight._validate_static_contract()
    assert preflight._semantic_hash(
        preflight._asset_rows(preflight.MINILM_REQUIRED_FILES)
    ) == preflight.MINILM_REQUIRED_TREE_SHA256
    assert preflight._semantic_hash(
        preflight._asset_rows(preflight.CE_REQUIRED_FILES)
    ) == preflight.CE_REQUIRED_TREE_SHA256
    assert len(preflight.PUBLIC_SYNTHETIC_FIXTURE_SHA256) == 64


def test_run_preflight_returns_only_sanitized_hash_shape_and_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_runtime(monkeypatch)

    receipt = preflight.run_preflight(
        typed_python="/typed/venv/bin/python",
        minilm_model="/models/minilm",
        cross_encoder_model="/models/ce",
        nvidia_smi="/usr/bin/nvidia-smi",
    )

    assert receipt["schema"] == preflight.RECEIPT_SCHEMA
    assert receipt["status"] == (
        "passed_public_synthetic_non_scoring_runtime_action_preflight"
    )
    assert receipt["study_design_self_sha256"] == (
        "eefa61986bd2f58efa26564dc0709728e0323660f23ae532819f4fa98f0601b3"
    )
    assert receipt["claim_boundary"] == {
        "api_or_provider_call_count": 0,
        "formal_HippoRAG_call_count": 0,
        "formal_MMQA_source_or_row_access_count": 0,
        "label_or_score_access_count": 0,
        "online_evaluator_call_count": 0,
        "retry_replay_or_resample_count": 0,
        "synthetic_fixture_count": 1,
    }
    assert receipt["concurrency"] == {
        "cross_encoder_physical_gpu_1_process_cap": 1,
        "minilm_physical_gpu_0_process_cap": 1,
        "model_process_co_residency": False,
        "typed_gpu_process_cap": 2,
    }
    assert receipt["address_family_isolation_probe"][
        "status"
    ] == "AF_INET_and_AF_INET6_socket_creation_denied"
    assert receipt["address_family_isolation_probe"]["probe_count"] == 2
    assert receipt["address_family_isolation_probe_sha256"] == (
        preflight._semantic_hash(
            receipt["address_family_isolation_probe"]
        )
    )
    assert receipt["core_action_preflight"]["proof_graph_shape"] == [8, 14]
    assert receipt["core_action_preflight"]["training_shape"] == [3, 23, 11]
    assert receipt["model_canaries"]["minilm"]["output_shape"] == [9, 384]
    assert receipt["model_canaries"]["cross_encoder"]["output_shape"] == [8, 1]
    body = dict(receipt)
    declared = body.pop("self_sha256")
    assert declared == preflight._semantic_hash(body)

    serialized = preflight._canonical_json_bytes(receipt).decode("ascii")
    for forbidden in (
        "coordinates_float64_hex",
        "coefficient_float64_hex",
        "energy_float64_hex",
        "Synthetic row",
        "Synthetic text",
        "/typed/venv",
        "/models/",
        '"runtime_versions"',
        '"uuid"',
    ):
        assert forbidden not in serialized


def test_address_family_probe_requires_both_errno97_denials() -> None:
    observed: list[int] = []

    def denied(family: int, _kind: int) -> object:
        observed.append(family)
        raise OSError(97, "Address family not supported")

    receipt = preflight.production_address_family_isolation_probe(
        socket_factory=denied
    )
    assert observed == [
        preflight.socket.AF_INET,
        preflight.socket.AF_INET6,
    ]
    assert receipt["probe_count"] == 2
    assert receipt["address_family_isolation_contract"] == (
        preflight.ADDRESS_FAMILY_ISOLATION_CONTRACT
    )


def test_address_family_probe_rejects_available_inet_socket() -> None:
    class OpenSocket:
        def close(self) -> None:
            pass

    with pytest.raises(
        preflight.MmqaP1LocalRuntimePreflightError,
        match="isolation is absent",
    ):
        preflight.production_address_family_isolation_probe(
            socket_factory=lambda _family, _kind: OpenSocket()
        )


def test_core_action_preflight_is_deterministic_and_coordinate_bound() -> None:
    minilm = (0.93, 0.86, 0.42, 0.31, 0.90, 0.82, 0.37, 0.22)
    cross = (0.96, 0.88, 0.44, 0.28, 0.92, 0.84, 0.33, 0.17)
    first = preflight._build_core_receipt(minilm, cross)
    second = preflight._build_core_receipt(minilm, cross)
    changed = preflight._build_core_receipt(
        minilm, (0.90, *cross[1:])
    )

    assert first == second
    assert first["proof_graph_shape"] == [8, 14]
    assert first["closure_shapes"] == [[3, 4], [5, 8], [5, 8]]
    assert first["bundle_registry_shapes"] == [[3, 11], [10, 11], [10, 11]]
    assert first["training_shape"] == [3, 23, 11]
    assert changed["core_output_sha256"] != first["core_output_sha256"]


def test_runtime_inventory_rejects_any_exact_version_or_device_drift() -> None:
    payload = _inventory_payload()
    assert len(
        preflight._validate_inventory_worker(payload, _typed_binding())
    ) == 64

    drifted = json.loads(json.dumps(payload))
    drifted["runtime_versions"]["torch"] = "2.8.1+cu128"
    with pytest.raises(
        preflight.MmqaP1LocalRuntimePreflightError,
        match="identity or versions drifted",
    ):
        preflight._validate_inventory_worker(drifted, _typed_binding())

    missing_gpu = json.loads(json.dumps(payload))
    missing_gpu["cuda_device_count"] = 1
    missing_gpu["cuda_devices"] = missing_gpu["cuda_devices"][:1]
    with pytest.raises(
        preflight.MmqaP1LocalRuntimePreflightError,
        match="CUDA visibility drifted",
    ):
        preflight._validate_inventory_worker(missing_gpu, _typed_binding())


def test_model_worker_rejects_nonrepeat_nonfinite_or_coordinate_leak_shape() -> None:
    coordinates, public = preflight._validate_model_worker(
        _minilm_payload(), role="minilm"
    )
    assert len(coordinates) == 8
    assert "coordinates_float64_hex" not in public

    nonrepeat = _minilm_payload()
    nonrepeat["repeat_exact"] = False
    with pytest.raises(preflight.MmqaP1LocalRuntimePreflightError):
        preflight._validate_model_worker(nonrepeat, role="minilm")

    nonfinite = _ce_payload()
    nonfinite["coordinates_float64_hex"][0] = "inf"
    with pytest.raises(
        preflight.MmqaP1LocalRuntimePreflightError,
        match="escaped",
    ):
        preflight._validate_model_worker(nonfinite, role="cross_encoder")

    extra = _ce_payload()
    extra["raw_logits"] = [1.0]
    with pytest.raises(
        preflight.MmqaP1LocalRuntimePreflightError,
        match="schema drifted",
    ):
        preflight._validate_model_worker(extra, role="cross_encoder")


def test_nvidia_smi_parser_binds_both_exact_311_gpu_uuids() -> None:
    raw = (
        b"0, GPU-32d6e292-70cd-50a0-405b-e344d2da8d39, "
        b"NVIDIA GeForce RTX 2080, 8192\n"
        b"1, GPU-db2137c8-0f6b-b790-a698-6bfbbd5dc9eb, "
        b"NVIDIA GeForce RTX 2080, 8192\n"
    )
    assert preflight._parse_gpu_probe_output(raw) == preflight.EXPECTED_GPU_ROWS

    wrong = raw.replace(b"GPU-db2137", b"GPU-deadbe")
    with pytest.raises(
        preflight.MmqaP1LocalRuntimePreflightError,
        match="UUID binding drifted",
    ):
        preflight._parse_gpu_probe_output(wrong)


def test_generic_asset_verifier_binds_required_files_and_hashes_cache(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    root.mkdir()
    payload = b"frozen model bytes"
    model_file = root / "model.safetensors"
    model_file.write_bytes(payload)
    cache = root / ".cache"
    cache.mkdir()
    (cache / "metadata.json").write_bytes(b"{}")
    row = preflight.AssetFile(
        "model.safetensors",
        len(payload),
        preflight.hashlib.sha256(payload).hexdigest(),
    )
    expected_tree = preflight._semantic_hash(preflight._asset_rows((row,)))

    receipt = preflight._verify_asset_tree(
        root,
        files=(row,),
        expected_tree_sha256=expected_tree,
        allowed_top_level_directories=frozenset({".cache"}),
        role="test",
    )

    assert receipt["required_tree_sha256"] == expected_tree
    assert receipt["auxiliary_file_count"] == 1
    assert len(receipt["auxiliary_tree_sha256"]) == 64

    model_file.write_bytes(b"drift")
    with pytest.raises(
        preflight.MmqaP1LocalRuntimePreflightError,
        match="required model file drifted",
    ):
        preflight._verify_asset_tree(
            root,
            files=(row,),
            expected_tree_sha256=expected_tree,
            allowed_top_level_directories=frozenset({".cache"}),
            role="test",
        )


def test_worker_output_must_be_single_canonical_json_line() -> None:
    payload = _minilm_payload()
    raw = preflight._canonical_json_bytes(payload, newline=True)
    assert preflight._parse_canonical_worker_output(raw) == payload

    with pytest.raises(
        preflight.MmqaP1LocalRuntimePreflightError,
        match="envelope drifted",
    ):
        preflight._parse_canonical_worker_output(
            json.dumps(payload, indent=2, sort_keys=True).encode("ascii")
        )


def test_output_is_exclusive_mode_0600_and_cannot_be_replayed(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    payload = {"schema": preflight.RECEIPT_SCHEMA, "status": "synthetic"}
    digest = preflight._write_exclusive(path, payload)

    assert len(digest) == 64
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.read_bytes() == preflight._canonical_json_bytes(payload, newline=True)
    with pytest.raises(
        preflight.MmqaP1LocalRuntimePreflightError,
        match="already exists",
    ):
        preflight._write_exclusive(path, payload)


def test_cli_has_no_source_api_evaluator_hippo_or_retry_surface() -> None:
    controller_options = {
        action.dest
        for action in preflight._controller_parser()._actions
        if action.dest != "help"
    }
    assert controller_options == {
        "cross_encoder_model",
        "minilm_model",
        "nvidia_smi",
        "output",
        "typed_python",
    }
    worker_options = {
        action.dest for action in preflight._worker_parser()._actions
    }
    assert worker_options == {"mode", "model"}
    forbidden = {
        "api",
        "dataset",
        "evaluator",
        "family",
        "gold",
        "hippo",
        "label",
        "qrel",
        "retry",
        "source",
    }
    assert not any(
        token in option
        for option in controller_options | worker_options
        for token in forbidden
    )


def test_gpu_probe_uses_only_fixed_query_and_sanitized_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "nvidia-smi"
    executable.write_bytes(b"tool")
    executable.chmod(0o700)
    observed: dict[str, object] = {}
    raw = (
        b"0, GPU-32d6e292-70cd-50a0-405b-e344d2da8d39, "
        b"NVIDIA GeForce RTX 2080, 8192\n"
        b"1, GPU-db2137c8-0f6b-b790-a698-6bfbbd5dc9eb, "
        b"NVIDIA GeForce RTX 2080, 8192\n"
    )

    def fake_execute(
        command: object,
        *,
        cwd: Path,
        environment: object,
        timeout: int,
    ) -> SimpleNamespace:
        observed.update(
            command=command,
            cwd=cwd,
            environment=environment,
            timeout=timeout,
        )
        return SimpleNamespace(returncode=0, stdout=raw, stderr=b"")

    monkeypatch.setattr(preflight, "_execute_subprocess", fake_execute)
    assert preflight._probe_gpu_rows(executable) == preflight.EXPECTED_GPU_ROWS
    assert observed["command"] == [
        str(executable),
        "--query-gpu=index,uuid,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    assert set(observed["environment"]) == {"HOME", "LANG", "LC_ALL", "PATH"}
    assert observed["timeout"] == 30
