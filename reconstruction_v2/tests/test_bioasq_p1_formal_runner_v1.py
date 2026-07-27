from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterator, Mapping

import pytest

from assumption_agent.benchmarks import (
    bioasq_p1_formal_controller_v1 as formal_controller,
)
from assumption_agent.benchmarks import (
    bioasq_p1_formal_source_v2 as formal_source,
)
from replication_runtime.bioasq_p1_formal_v1 import acquisition
from replication_runtime.bioasq_p1_formal_v1 import contract
from replication_runtime.bioasq_p1_formal_v1 import lanes
from replication_runtime.bioasq_p1_formal_v1 import runner


HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
HEX_D = "d" * 64


@pytest.fixture
def posix_tmp() -> Iterator[Path]:
    """Use the Linux filesystem because the runtime enforces POSIX custody."""

    root = Path(
        tempfile.mkdtemp(prefix="bioasq-runner-", dir="/tmp")
    ).absolute()
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _write_canonical(path: Path, value: object) -> str:
    raw = contract.canonical_bytes(value, newline=True)
    path.write_bytes(raw)
    os.chmod(path, 0o600)
    return hashlib.sha256(raw).hexdigest()


def _preflight_value(root: Path) -> dict[str, object]:
    return contract.with_self_hash(
        {
            "bioasq_hardware_binding_file_sha256": HEX_B,
            "bioasq_hardware_binding_path": str(
                root / "bioasq-hardware.safe.json"
            ),
            "bioasq_hardware_binding_self_sha256": HEX_C,
            "coordinate_canary_binding_sha256": HEX_B,
            "coordinate_canary_receipt_file_sha256": HEX_C,
            "coordinate_canary_receipt_path": str(
                root / "coordinate-canary.safe.json"
            ),
            "coordinate_canary_receipt_self_sha256": HEX_D,
            "coordinate_project_root": str(root / "project"),
            "execution_binding_sha256": HEX_A,
            "hippo_local_embedding_model": str(
                root / "hippo-embedding"
            ),
            "hippo_local_llm_model": str(root / "hippo-llm"),
            "hippo_runtime_fingerprint_path": str(
                root / "runtime-fingerprint.json"
            ),
            "hippo_runtime_python": str(root / "hippo-python"),
            "hippo_worker_project_root": str(root / "project"),
            "legacy_hardware_binding_path": str(
                root / "hardware.safe.json"
            ),
            "legacy_hippo_canary_receipt_path": str(
                root / "hippo-canary.safe.json"
            ),
            "schema": contract.FORMAL_PREFLIGHT_CONFIG_SCHEMA,
        }
    )


def _formal_value(
    root: Path,
    *,
    formal_root: Path | None = None,
) -> tuple[dict[str, object], Path]:
    preflight_path = root / "preflight.config.json"
    preflight = _preflight_value(root)
    preflight_file_sha256 = _write_canonical(
        preflight_path, preflight
    )
    value = contract.with_self_hash(
        {
            "coordinate_lane": {
                "cross_encoder_model_root": str(root / "cross"),
                "minilm_asset_manifest": str(root / "minilm.json"),
                "minilm_model_root": str(root / "minilm"),
                "project_root": str(root / "project"),
                "runtime_python": str(root / "coordinate-python"),
                "timeout_seconds": 120,
            },
            "execution_binding_sha256": HEX_A,
            "formal_root": str(formal_root or (root / "formal")),
            "hippo_lane": {
                "build_timeout_seconds": 120,
                "current_hardware_binding_path": str(
                    root / "bioasq-hardware.safe.json"
                ),
                "local_embedding_model": str(
                    root / "hippo-embedding"
                ),
                "local_llm_model": str(root / "hippo-llm"),
                "retrieve_timeout_seconds": 120,
                "runtime_fingerprint_path": str(
                    root / "runtime-fingerprint.json"
                ),
                "runtime_python": str(root / "hippo-python"),
                "worker_project_root": str(root / "project"),
            },
            "preflight_config_file_sha256": preflight_file_sha256,
            "preflight_config_path": str(preflight_path),
            "preflight_config_self_sha256": preflight["self_sha256"],
            "schema": runner.FORMAL_CONFIG_SCHEMA,
            "source_inputs": {
                "p0_private_manifest_path": str(
                    root / "p0.private.json"
                ),
                "p0_receipt_path": str(root / "p0.safe.json"),
                "source_path": str(root / "source" / "training11b.json"),
            },
        }
    )
    return value, preflight_path


def _loaded_config(
    root: Path,
    *,
    formal_root: Path | None = None,
) -> runner.FormalRuntimeConfig:
    value, _ = _formal_value(root, formal_root=formal_root)
    path = root / "formal.config.json"
    _write_canonical(path, value)
    return runner.load_formal_runtime_config(path)


def test_formal_config_is_strict_canonical_and_self_hashed(
    posix_tmp: Path,
) -> None:
    value, _ = _formal_value(posix_tmp)
    path = posix_tmp / "formal.config.json"
    _write_canonical(path, value)

    loaded = runner.load_formal_runtime_config(path)
    assert loaded.execution_binding_sha256 == HEX_A
    assert loaded.formal_root == posix_tmp / "formal"
    assert loaded.coordinate.timeout_seconds == 120
    assert loaded.hippo.retrieve_timeout_seconds == 120

    extra = dict(value)
    extra["api_key"] = "forbidden"
    extra = contract.with_self_hash(
        {key: item for key, item in extra.items() if key != "self_sha256"}
    )
    extra_path = posix_tmp / "extra.config.json"
    _write_canonical(extra_path, extra)
    with pytest.raises(
        contract.BioasqP1FormalRuntimeError,
        match="schema drifted",
    ):
        runner.load_formal_runtime_config(extra_path)

    tampered = dict(value)
    tampered["execution_binding_sha256"] = HEX_B
    tampered_path = posix_tmp / "tampered.config.json"
    _write_canonical(tampered_path, tampered)
    with pytest.raises(
        contract.BioasqP1FormalRuntimeError,
        match="self hash",
    ):
        runner.load_formal_runtime_config(tampered_path)

    noncanonical_path = posix_tmp / "noncanonical.config.json"
    noncanonical_path.write_text(
        json.dumps(value, indent=2),
        encoding="ascii",
    )
    with pytest.raises(
        contract.BioasqP1FormalRuntimeError,
        match="not canonical",
    ):
        runner.load_formal_runtime_config(noncanonical_path)


def test_canary_cli_dispatches_only_to_source_free_path(
    posix_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = posix_tmp / "canary.config.json"
    canary_config = contract.CoordinateCanaryConfig(
        canary_root=posix_tmp / "canary",
        canary_binding_sha256=HEX_A,
        runtime_python=posix_tmp / "python",
        project_root=posix_tmp / "project",
        minilm_asset_manifest=posix_tmp / "minilm.json",
        minilm_model_root=posix_tmp / "minilm",
        cross_encoder_model_root=posix_tmp / "cross",
        timeout_seconds=120,
    )
    calls: list[object] = []
    receipt = contract.with_self_hash(
        {
            "schema": "synthetic_source_free_receipt",
            "status": "passed",
        }
    )
    monkeypatch.setattr(
        contract,
        "load_runtime_config",
        lambda path: calls.append(("load", path)) or canary_config,
    )
    monkeypatch.setattr(
        contract,
        "run_source_free_coordinate_canary_once",
        lambda value: calls.append(("run", value)) or receipt,
    )
    writes: list[tuple[int, bytes]] = []
    monkeypatch.setattr(
        runner.os,
        "write",
        lambda fd, raw: writes.append((fd, raw)) or len(raw),
    )

    assert (
        runner.main(
            [
                "--source-free-coordinate-canary",
                "--config",
                str(config_path),
            ]
        )
        == 0
    )
    assert calls == [
        ("load", config_path),
        ("run", canary_config),
    ]
    assert writes == [
        (1, contract.canonical_bytes(receipt, newline=True))
    ]


class _FakeHippoLane:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.build_call_count = 1
        self.retrieve_call_count = 1
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeCoordinateLane:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.worker_call_count = 2


class _FakeAcquisition:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def _patch_successful_formal_components(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> tuple[list[dict[str, object]], list[_FakeHippoLane]]:
    compile_calls: list[dict[str, object]] = []
    hippo_instances: list[_FakeHippoLane] = []
    selection = contract.with_self_hash(
        {
            "schema": "synthetic_selection_receipt",
            "status": "passed",
        }
    )
    controller_terminal = contract.with_self_hash(
        {
            "schema": "synthetic_controller_terminal",
            "status": "terminal_complete",
        }
    )

    def compile_once(**kwargs: object) -> Mapping[str, object]:
        events.append("source_compile")
        compile_calls.append(dict(kwargs))
        return selection

    def make_hippo(**kwargs: object) -> _FakeHippoLane:
        value = _FakeHippoLane(**kwargs)
        hippo_instances.append(value)
        return value

    def run_controller(**kwargs: object) -> Mapping[str, object]:
        events.append("controller")
        assert isinstance(kwargs["acquisition"], _FakeAcquisition)
        assert isinstance(
            kwargs["coordinate_scorer"], _FakeCoordinateLane
        )
        assert kwargs["hippo_runner"] is hippo_instances[0]
        return controller_terminal

    monkeypatch.setattr(
        formal_source, "compile_formal_source", compile_once
    )
    monkeypatch.setattr(lanes, "OfficialHippoLane", make_hippo)
    monkeypatch.setattr(
        lanes, "CoordinateScorerLane", _FakeCoordinateLane
    )
    monkeypatch.setattr(
        acquisition,
        "SealedSourceAcquisitionBoundary",
        _FakeAcquisition,
    )
    monkeypatch.setattr(
        formal_controller, "run_formal_controller", run_controller
    )
    return compile_calls, hippo_instances


def test_formal_preflight_precedes_exactly_one_source_compile_and_safe_terminal(
    posix_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _loaded_config(posix_tmp)
    events: list[str] = []
    compile_calls, hippo_instances = _patch_successful_formal_components(
        monkeypatch, events
    )
    preflight = contract.with_self_hash(
        {
            "aggregate_only_public_receipt": True,
            "coordinate_model_binding_sha256": HEX_D,
            "schema": "synthetic_preflight_receipt",
            "status": "passed",
        }
    )

    def verify(
        value: contract.FormalPreflightConfig,
    ) -> Mapping[str, object]:
        events.append("preflight")
        assert value.execution_binding_sha256 == HEX_A
        assert not (config.formal_root / "formal_source").exists()
        return preflight

    monkeypatch.setattr(contract, "verify_formal_preflight", verify)
    terminal = runner.run_formal_once(config)

    assert events == ["preflight", "source_compile", "controller"]
    assert len(compile_calls) == 1
    assert compile_calls[0]["source_path"] == config.source.source_path
    assert terminal["status"] == "terminal_complete"
    assert terminal["source_compiler_invocation_count"] == 1
    assert terminal["formal_source_access_count"] == 1
    assert terminal["coordinate_worker_call_count"] == 2
    assert terminal["hipporag_build_call_count"] == 1
    assert terminal["hipporag_retrieve_call_count"] == 1
    assert terminal["online_or_API_evaluator_calls"] == 0
    assert (
        terminal[
            "item_query_document_qrel_action_or_per_item_score_values_published"
        ]
        is False
    )
    assert hippo_instances[0].closed is True
    persisted = json.loads(
        (
            config.formal_root / runner.FORMAL_TERMINAL_FILENAME
        ).read_text("ascii")
    )
    assert persisted == terminal
    assert not (
        config.formal_root / runner.FORMAL_FAILURE_FILENAME
    ).exists()


def test_preflight_failure_never_invokes_source_compiler(
    posix_tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _loaded_config(posix_tmp)
    compile_calls: list[object] = []
    monkeypatch.setattr(
        formal_source,
        "compile_formal_source",
        lambda **kwargs: compile_calls.append(kwargs),
    )

    def fail_preflight(
        value: contract.FormalPreflightConfig,
    ) -> Mapping[str, object]:
        del value
        raise contract.BioasqP1FormalRuntimeError(
            "synthetic source-free preflight failure"
        )

    monkeypatch.setattr(
        contract, "verify_formal_preflight", fail_preflight
    )
    with pytest.raises(
        contract.BioasqP1FormalRuntimeError,
        match="source-free preflight failure",
    ):
        runner.run_formal_once(config)

    assert compile_calls == []
    assert not (config.formal_root / "formal_source").exists()
    failure = json.loads(
        (
            config.formal_root / runner.FORMAL_FAILURE_FILENAME
        ).read_text("ascii")
    )
    assert (
        failure["failure_stage"]
        == "offline_preflight_before_source_access"
    )
    assert failure["source_compiler_invocation_count"] == 0
    assert failure["online_or_API_evaluator_calls"] == 0
    assert failure["retry_count"] == 0
    assert failure["status"] == "failed_closed_no_retry"
    assert not (
        config.formal_root / runner.FORMAL_TERMINAL_FILENAME
    ).exists()
