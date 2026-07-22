from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from assumption_agent.benchmarks import tatqa_p19_acquisition_v1 as acquisition
from assumption_agent.benchmarks import tatqa_p19_implementation_freeze_v1 as freeze
from assumption_agent.benchmarks import tatqa_p19_public_canary_v1 as canary
from replication_runtime.tatqa_p19_v1 import hipporag_contract, typed_plan_contract


def _assets(tmp_path: Path) -> dict[str, Path]:
    result = {}
    for name in freeze.ASSET_NAMES:
        root = tmp_path / name
        root.mkdir()
        (root / "immutable.bin").write_bytes(name.encode("ascii"))
        result[name] = root
    return result


def _fingerprint(tmp_path: Path) -> Path:
    path = tmp_path / "runtime.fingerprint.json"
    typed_subfingerprint = acquisition.self_hashed(
        {
            "schema": (
                "tatqa_p19_typed_minilm_runtime_python_subfingerprint_v1"
            ),
            "capability_id": "TEST_TYPED_MINILM_RUNTIME",
        }
    )
    hippo_subfingerprint = acquisition.self_hashed(
        {
            "schema": "tatqa_p19_hipporag_runtime_python_subfingerprint_v1",
            "capability_id": "TEST_HIPPORAG_RUNTIME",
        }
    )
    freeze.build_runtime_fingerprint(
        output_path=path,
        asset_roots=_assets(tmp_path),
        runtime_inventory={
            "host": "public-test-host",
            "python": "3.11",
            "torch": "2.test",
            "runtime_python_subfingerprints": {
                "typed_plan_minilm_runtime_python": typed_subfingerprint,
                "hipporag_runtime_python": hippo_subfingerprint,
            },
        },
        systemd_network_preflight={
            "network_properties": [
                "IPAddressDeny=any",
                "RestrictAddressFamilies=AF_UNIX",
            ],
            "returncode": 0,
            "stdout_sha256": hashlib.sha256(b"").hexdigest(),
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
        },
    )
    return path


def _completion() -> str:
    return json.dumps(
        {
            "entity_facets": ["Northwind"],
            "metric_facets": ["renewable electricity share"],
            "time_facets": ["2022", "2024"],
            "operation": "DIFFERENCE",
            "relation_query": (
                "increase in renewable electricity share and explanation"
            ),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _unit_closure(
    *, unit_name_sha256: str, control_group_sha256: str, scope: str
) -> dict[str, object]:
    digest = lambda suffix: hashlib.sha256(f"{scope}-{suffix}".encode()).hexdigest()
    return {
        "active_state": "inactive",
        "control_group_process_count": 0,
        "control_group_sha256": control_group_sha256,
        "control_group_thread_count": 0,
        "load_state": "not-found",
        "main_pid": 0,
        "schema": "tatqa_p19_formal_runtime_v1_systemd_unit_closure_v1",
        "sub_state": "dead",
        "systemctl_reset_failed_returncode": 0,
        "systemctl_reset_failed_stderr_sha256": digest("reset-stderr"),
        "systemctl_reset_failed_stdout_sha256": digest("reset-stdout"),
        "systemctl_show_returncode": 0,
        "systemctl_show_stderr_sha256": digest("closed-show-stderr"),
        "systemctl_show_stdout_sha256": digest("closed-show-stdout"),
        "unit_name_sha256": unit_name_sha256,
    }


def _start_policy(
    *,
    unit_name_sha256: str,
    control_group_sha256: str,
    worker_pid: int,
    scope: str,
) -> dict[str, object]:
    digest = lambda suffix: hashlib.sha256(f"{scope}-{suffix}".encode()).hexdigest()
    return {
        "active_state": "active",
        "control_group_sha256": control_group_sha256,
        "kill_mode": "control-group",
        "load_state": "loaded",
        "main_pid": worker_pid,
        "schema": "tatqa_p19_formal_runtime_v1_systemd_start_policy_v1",
        "sub_state": "running",
        "systemctl_show_returncode": 0,
        "systemctl_show_stderr_sha256": digest("start-show-stderr"),
        "systemctl_show_stdout_sha256": digest("start-show-stdout"),
        "tasks_max": 3,
        "unit_name_sha256": unit_name_sha256,
    }


class _TypedRunner:
    def __init__(self, *, drift_second: bool = False) -> None:
        self.drift_second = drift_second
        self.calls: list[tuple[str, bytes]] = []
        self.receipts: dict[str, dict[str, object]] = {}

    def __call__(self, block: str, canonical_input: bytes) -> bytes:
        items = typed_plan_contract.parse_input(canonical_input)
        assert len(items) == 1
        index = len(self.calls)
        self.calls.append((block, canonical_input))
        row = typed_plan_contract.build_output_item(
            item=items[0],
            completion=_completion(),
            completion_token_count=32,
            prompt_sha256=("b" if self.drift_second and index == 1 else "a") * 64,
            prompt_token_count=120,
            prompt_projection_sha256="c" * 64,
        )
        raw = typed_plan_contract.canonical_json_bytes(
            typed_plan_contract.output_payload((row,))
        )
        worker_pid = 100 + index
        scope = f"typed-{block}-{worker_pid}"
        unit_name_sha256 = hashlib.sha256(f"{scope}-unit".encode()).hexdigest()
        control_group_sha256 = hashlib.sha256(
            f"{scope}-control-group".encode()
        ).hexdigest()
        self.receipts[block] = {
            "schema": "tatqa_p19_formal_runtime_v1_typed_plan_transport_receipt_v1",
            "block": block,
            "item_count": 1,
            "input_sha256": hashlib.sha256(canonical_input).hexdigest(),
            "model_execution_finished_monotonic_ns": 500 + index,
            "model_execution_started_monotonic_ns": 100 + index,
            "output_sha256": hashlib.sha256(raw).hexdigest(),
            "stdout_sha256": "d" * 64,
            "stderr_sha256": "e" * 64,
            "batch_size": 4,
            "physical_GPU": "1",
            "worker_pid": worker_pid,
            "model_context_tokens": 32768,
            "filesystem_isolation": (
                "systemd_InaccessiblePaths_official_source_and_acquisition_v1"
            ),
            "network_properties": [
                "IPAddressDeny=any",
                "RestrictAddressFamilies=AF_UNIX",
            ],
            "systemd_unit_closure": _unit_closure(
                unit_name_sha256=unit_name_sha256,
                control_group_sha256=control_group_sha256,
                scope=scope,
            ),
            "systemd_unit_name_sha256": unit_name_sha256,
        }
        return raw


def _normalized(rows: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(rows, dtype=np.float32)
    matrix = np.pad(matrix, ((0, 0), (0, 384 - matrix.shape[1])))
    matrix /= np.linalg.norm(
        matrix.astype(np.float64), axis=1, keepdims=True
    ).astype(np.float32)
    return matrix


def _matrix() -> np.ndarray:
    # question, entity, metric, 2022, 2024, relation, then eight units.
    return _normalized(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 1],
            [0.5, 0.3, 0.95, 0.1, 0.1, 0.1],
            [0.4, 0.85, 0.2, 0.95, 0.3, 0.1],
            [0.6, 0.6, 0.6, 0.6, 0.6, 0.1],
            [0.95, 0.2, 0.1, 0.1, 0.1, 0.1],
            [0.2, 0.8, 0.1, 0.1, 0.95, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.9],
            [0.1, 0.3, 0.4, 0.4, 0.1, 0.8],
        ]
    )


class _Encoder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.runtime_receipt = {
            "asset_file_sha256": "1" * 64,
            "asset_manifest_path": "/sanitized/by-canary/manifest",
            "asset_sha256": "2" * 64,
            "embedding_dimension": 384,
            "maximum_sequence_length": 512,
            "model_root": "/sanitized/by-canary/model",
            "model_tree_sha256": "3" * 64,
            "runtime_versions": {"python": "test"},
            "status": "verified_offline_immutable_qasper_minilm_runtime",
            "weights_sha256": "4" * 64,
        }
        self.canary_receipt = {
            "float32_bytes_sha256": "5" * 64,
            "quantized_embedding_matrix_sha256": "6" * 64,
            "qasper_rows_or_archives_accessed_by_canary": False,
            "repeat_count": 2,
            "repeat_exact": True,
            "sentence_count": 256,
            "status": "passed_exact_row_free_synthetic_canary",
            "text_vector_sha256": "7" * 64,
        }

    def encode(self, texts):
        self.calls.append(tuple(texts))
        assert len(texts) == 14
        return _matrix().copy()


class _HippoRunner:
    def __init__(self) -> None:
        self.receipts = []

    def __call__(self, block: str, item_id: str, canonical_input: bytes) -> bytes:
        query, units = hipporag_contract.parse_input(canonical_input)
        input_sha = hipporag_contract.input_binding_sha256(query, units)
        value = hipporag_contract.output_payload(
            top_unit_ids=[row.unit_id for row in units[:5]],
            graph_nodes=9,
            graph_edges=8,
            unit_count=len(units),
            input_sha256=input_sha,
        )
        raw = hipporag_contract.canonical_json_bytes(value)
        worker_pid = 201
        scope = f"hippo-{block}-{item_id}-{worker_pid}"
        unit_name_sha256 = hashlib.sha256(f"{scope}-unit".encode()).hexdigest()
        control_group_sha256 = hashlib.sha256(
            f"{scope}-control-group".encode()
        ).hexdigest()
        start_policy = _start_policy(
            unit_name_sha256=unit_name_sha256,
            control_group_sha256=control_group_sha256,
            worker_pid=worker_pid,
            scope=scope,
        )
        self.receipts.append(
            {
                "schema": "tatqa_p19_formal_runtime_v1_hippo_transport_receipt_v1",
                "block": block,
                "item_commitment_sha256": item_id,
                "input_file_sha256": hashlib.sha256(canonical_input).hexdigest(),
                "input_semantic_sha256": input_sha,
                "output_file_sha256": hashlib.sha256(raw).hexdigest(),
                "stdout_sha256": "8" * 64,
                "stderr_sha256": "9" * 64,
                "CPU_threads": 2,
                "configured_torch_interop_threads": 1,
                "configured_torch_intraop_threads": 1,
                "model_execution_finished_monotonic_ns": 300,
                "model_execution_started_monotonic_ns": 200,
                "observed_process_thread_peak": 1,
                "worker_pid": worker_pid,
                "filesystem_isolation": (
                    "systemd_InaccessiblePaths_official_source_and_acquisition_v1"
                ),
                "visible_GPU": "",
                "network_properties": [
                    "IPAddressDeny=any",
                    "RestrictAddressFamilies=AF_UNIX",
                ],
                "maximum_worker_process_threads": 2,
                "systemd_start_policy": start_policy,
                "systemd_start_policy_sha256": acquisition.stable_hash(
                    start_policy
                ),
                "systemd_tasks_max": 3,
                "systemd_unit_closure": _unit_closure(
                    unit_name_sha256=unit_name_sha256,
                    control_group_sha256=control_group_sha256,
                    scope=scope,
                ),
                "systemd_unit_name_sha256": unit_name_sha256,
                "thread_monitor_process_reservation": 1,
            }
        )
        return raw


def test_public_canary_crosses_full_path_twice_and_binds_optional_hippo(
    tmp_path: Path,
) -> None:
    runner = _TypedRunner()
    encoder = _Encoder()
    hippo = _HippoRunner()
    fingerprint_path = _fingerprint(tmp_path)
    fingerprint = json.loads(fingerprint_path.read_text(encoding="ascii"))
    output = tmp_path / "canary.json"
    receipt = canary.run_public_production_canary(
        runtime_fingerprint_path=fingerprint_path,
        output_path=output,
        typed_plan_runner=runner,
        encoder=encoder,
        hippo_runner=hippo,
    )
    assert receipt["qualified"] is True
    assert receipt["typed_plan_output_exact_repeat"] is True
    assert receipt["embedding_matrix_exact_repeat"] is True
    assert receipt["compiled_tensor_exact_repeat"] is True
    assert receipt["public_synthetic_distinct_rankings"] is True
    assert receipt["P1_retains_ordered_P0_top3"] is True
    assert receipt["P1_outside_P0_unit_count"] >= 1
    assert receipt["hippo_canary_ran"] is True
    assert receipt["hippo_canary_input_semantic_sha256"]
    assert receipt["hippo_canary_output_file_sha256"]
    expected_subfingerprints = {
        key: value["self_sha256"]
        for key, value in fingerprint["runtime_inventory"][
            "runtime_python_subfingerprints"
        ].items()
    }
    assert receipt[canary.RUNTIME_SUBFINGERPRINT_HASHES_FIELD] == (
        expected_subfingerprints
    )
    assert receipt["filesystem_isolation"] == (
        "systemd_InaccessiblePaths_official_source_and_acquisition_v1"
    )
    assert len(runner.calls) == len(encoder.calls) == 2
    assert runner.calls[0][1] == runner.calls[1][1]
    assert output.read_bytes().endswith(b"\n")
    acquisition.validate_production_canary_capability_receipts(
        receipt,
        runtime_fingerprint=fingerprint,
    )
    fallback = json.loads(json.dumps(receipt))
    fallback["typed_plan_worker_receipt_source"] = "canary_transport_fallback"
    with pytest.raises(acquisition.TatqaP19AcquisitionError, match="fallback"):
        acquisition.validate_production_canary_capability_receipts(
            fallback,
            runtime_fingerprint=fingerprint,
        )
    cross_binding_tamper = json.loads(json.dumps(receipt))
    cross_binding_tamper[canary.RUNTIME_SUBFINGERPRINT_HASHES_FIELD][
        "hipporag_runtime_python"
    ] = "0" * 64
    with pytest.raises(
        acquisition.TatqaP19AcquisitionError, match="cross-binding"
    ):
        acquisition.validate_production_canary_capability_receipts(
            cross_binding_tamper,
            runtime_fingerprint=fingerprint,
        )


def test_public_canary_fails_closed_on_second_run_byte_drift(tmp_path: Path) -> None:
    with pytest.raises(canary.TatqaP19PublicCanaryError, match="not exact"):
        canary.run_public_production_canary(
            runtime_fingerprint_path=_fingerprint(tmp_path),
            output_path=tmp_path / "never.json",
            typed_plan_runner=_TypedRunner(drift_second=True),
            encoder=_Encoder(),
        )
    assert not (tmp_path / "never.json").exists()


def test_public_canary_rejects_invalid_nested_runtime_subfingerprint(
    tmp_path: Path,
) -> None:
    fingerprint_path = _fingerprint(tmp_path)
    fingerprint = json.loads(fingerprint_path.read_text(encoding="ascii"))
    fingerprint["runtime_inventory"]["runtime_python_subfingerprints"][
        "typed_plan_minilm_runtime_python"
    ]["capability_id"] = "DRIFTED_WITHOUT_NESTED_REHASH"
    body = dict(fingerprint)
    body.pop("self_sha256")
    fingerprint["self_sha256"] = acquisition.stable_hash(body)
    drifted_path = tmp_path / "runtime.drifted.json"
    drifted_path.write_bytes(
        (
            json.dumps(
                fingerprint,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    )
    with pytest.raises(canary.TatqaP19PublicCanaryError, match="binding drifted"):
        canary.run_public_production_canary(
            runtime_fingerprint_path=drifted_path,
            output_path=tmp_path / "never-drifted.json",
            typed_plan_runner=_TypedRunner(),
            encoder=_Encoder(),
        )


@pytest.mark.parametrize("drifted_pid", (True, 100.5))
def test_acquisition_rejects_bool_or_float_canary_transport_pid_tamper(
    tmp_path: Path, drifted_pid: object
) -> None:
    fingerprint_path = _fingerprint(tmp_path)
    fingerprint = json.loads(fingerprint_path.read_text(encoding="ascii"))
    receipt = canary.run_public_production_canary(
        runtime_fingerprint_path=fingerprint_path,
        output_path=tmp_path / "canary.json",
        typed_plan_runner=_TypedRunner(),
        encoder=_Encoder(),
        hippo_runner=_HippoRunner(),
    )
    tampered = json.loads(json.dumps(receipt))
    tampered["typed_plan_worker_receipt_snapshot"]["receipts"][0][
        "worker_pid"
    ] = drifted_pid
    with pytest.raises(acquisition.TatqaP19AcquisitionError, match="typed capability"):
        acquisition.validate_production_canary_capability_receipts(
            tampered,
            runtime_fingerprint=fingerprint,
        )


def test_public_fixture_is_source_and_label_free() -> None:
    raw = json.dumps(canary.public_fixture_payload(), sort_keys=True).casefold()
    for forbidden in (
        "question_uid",
        "table_uid",
        "answer_from",
        "gold_unit",
        "tatqa_dataset",
    ):
        assert forbidden not in raw
    assert canary.public_runtime_item().item_id == hashlib.sha256(
        json.dumps(
            canary.public_fixture_payload(),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
