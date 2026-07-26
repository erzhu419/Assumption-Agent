from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Mapping, Sequence

import numpy as np
import pytest

from assumption_agent.benchmarks import ebmnlp_p1_runtime_v1 as runtime
from replication_runtime.ebmnlp_p1_official_v1 import contract as hippo


def _payload(identity: str, count: int = 3) -> dict[str, object]:
    return hippo.input_payload(
        abstract_work_id=hashlib.sha256(
            f"abstract-{identity}".encode("ascii")
        ).hexdigest(),
        documents=[
            {
                "ordinal": ordinal,
                "text": f"Public synthetic evidence window {identity} {ordinal}.",
                "window_id": (
                    f"W:{ordinal * 24:08d}:"
                    f"{ordinal * 24 + 48:08d}"
                ),
            }
            for ordinal in range(count)
        ],
        queries=[
            {
                "ordinal": ordinal,
                "role": role,
                "text": hippo.ROLE_QUERIES[role],
                "work_id": hashlib.sha256(
                    f"{identity}-{role}".encode("ascii")
                ).hexdigest(),
            }
            for ordinal, role in enumerate(hippo.ROLE_ORDER)
        ],
    )


@dataclass
class _Solution:
    docs: list[str]
    doc_scores: list[float]


class _Graph:
    def vcount(self) -> int:
        return 3

    def ecount(self) -> int:
        return 2


class _Core:
    def __init__(self) -> None:
        self.graph = _Graph()
        self.documents: list[str] = []

    def index(self, documents: list[str]) -> None:
        self.documents = list(documents)

    def retrieve(
        self, queries: list[str], *, num_to_retrieve: int
    ) -> list[_Solution]:
        assert num_to_retrieve == len(self.documents)
        return [
            _Solution(
                docs=list(self.documents),
                doc_scores=[
                    float(len(self.documents) - ordinal)
                    for ordinal in range(len(self.documents))
                ],
            )
            for _query in queries
        ]


def _paths(tmp_path: Path, root_name: str = "runtime") -> runtime.RuntimePaths:
    project = tmp_path / "project"
    minilm = tmp_path / "minilm"
    llm = tmp_path / "llm"
    for path in (project, minilm, llm):
        path.mkdir(exist_ok=True)
    manifest = tmp_path / "asset.json"
    manifest.write_text("{}\n", encoding="ascii")
    return runtime.RuntimePaths(
        project_root=project,
        runtime_root=tmp_path / root_name,
        minilm_asset_manifest=manifest,
        minilm_model=minilm,
        hippo_llm_model=llm,
        hippo_embedding_model=minilm,
        hipporag_source=project,
        hippo_python=Path(sys.executable),
        strace_executable=Path("/usr/bin/true"),
        env_executable=Path("/usr/bin/env"),
    )


class _FakeModel:
    def encode(self, texts, **_kwargs):
        matrix = np.zeros(
            (len(texts), runtime.EMBEDDING_DIMENSION),
            dtype=np.float32,
        )
        for ordinal in range(len(texts)):
            matrix[ordinal, ordinal % runtime.EMBEDDING_DIMENSION] = 1.0
        return matrix


def test_local_minilm_validates_float32_shape_norm_and_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "CUBLAS_WORKSPACE_CONFIG", runtime.CUBLAS_WORKSPACE_CONFIG
    )
    monkeypatch.setattr(
        runtime,
        "_verify_minilm_asset",
        lambda **_kwargs: {
            "model_tree_sha256": runtime.MINILM_NORMATIVE_TREE_SHA256,
            "weights_sha256": runtime.MINILM_WEIGHTS_SHA256,
        },
    )
    monkeypatch.setattr(
        runtime, "_load_local_minilm_model", lambda _root: _FakeModel()
    )
    embedder = runtime.LocalMiniLMEmbedder(
        asset_manifest=tmp_path / "asset.json",
        model_root=tmp_path / "model",
    )
    rows = embedder(("alpha", "beta"))
    assert len(rows) == 2
    assert len(rows[0]) == runtime.EMBEDDING_DIMENSION
    assert all(isinstance(value, float) for value in rows[0])
    receipt = embedder.safe_runtime_receipt()
    assert receipt["call_count"] == 1
    assert receipt["encoded_text_count"] == 2
    assert receipt["external_network_call_count"] == 0


def test_two_lane_launcher_is_bounded_audited_and_nonreplayable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    active = 0
    peak = 0
    by_gpu = {"0": 0, "1": 0}
    by_gpu_peak = {"0": 0, "1": 0}
    lock = threading.Lock()

    def fake_run(command, **_kwargs):
        nonlocal active, peak
        gpu = next(
            value.split("=", 1)[1]
            for value in command
            if value.startswith("CUDA_VISIBLE_DEVICES=")
        )
        assert (
            f"CUBLAS_WORKSPACE_CONFIG={runtime.CUBLAS_WORKSPACE_CONFIG}"
            in command
        )
        python_path = next(
            value.split("=", 1)[1]
            for value in command
            if value.startswith("PYTHONPATH=")
        )
        assert str(paths.project_root) in python_path.split(":")
        assert str(paths.hipporag_source) in python_path.split(":")
        assert (
            Path(command[command.index("--hipporag-source-root") + 1])
            == paths.hipporag_source
        )
        assert (
            Path(command[command.index("--project-root") + 1])
            == paths.project_root
        )
        with lock:
            active += 1
            by_gpu[gpu] += 1
            peak = max(peak, active)
            by_gpu_peak[gpu] = max(by_gpu_peak[gpu], by_gpu[gpu])
        try:
            input_path = Path(
                command[command.index("--input") + 1]
            )
            output_path = Path(
                command[command.index("--output") + 1]
            )
            index_root = Path(
                command[command.index("--index-root") + 1]
            )
            audit_path = Path(command[command.index("-o") + 1])
            payload = json.loads(input_path.read_text("ascii"))
            result = hippo.retrieve_abstract_with_core(
                core=_Core(), payload=payload
            )
            index_root.mkdir(mode=0o700)
            output_path.write_bytes(hippo.canonical_json_bytes(result))
            output_path.chmod(0o600)
            audit_path.write_text(
                "socket(AF_INET, SOCK_STREAM, 0) = -1 EPERM "
                "(Operation not permitted)\n",
                encoding="utf-8",
            )
            time.sleep(0.03)
            return subprocess.CompletedProcess(command, 0, b"", b"")
        finally:
            with lock:
                active -= 1
                by_gpu[gpu] -= 1

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    launcher = runtime.OfficialHippoBatchLauncher(paths)
    payloads = tuple(_payload(str(index)) for index in range(4))
    outputs = launcher(payloads)
    assert set(outputs) == {
        payload["abstract_work_id"] for payload in payloads
    }
    assert peak == 2
    assert by_gpu_peak == {"0": 1, "1": 1}
    receipt = launcher.safe_runtime_receipt()
    assert receipt["worker_attempt_count"] == 4
    assert receipt["worker_completed_count"] == 4
    assert receipt["index_destroyed_count"] == 4
    assert receipt["attempted_network_syscall_count"] == 4
    assert receipt["denied_network_syscall_count"] == 4
    assert not list(paths.runtime_root.rglob("index"))
    with pytest.raises(runtime.EbmNlpP1RuntimeError, match="retry|replay"):
        launcher(payloads)


def test_worker_failure_terminally_consumes_launcher_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def failed_run(command, **_kwargs):
        nonlocal calls
        calls += 1
        audit_path = Path(command[command.index("-o") + 1])
        audit_path.write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(command, 3, b"", b"")

    monkeypatch.setattr(runtime.subprocess, "run", failed_run)
    launcher = runtime.OfficialHippoBatchLauncher(
        _paths(tmp_path, "failed-runtime")
    )
    with pytest.raises(runtime.EbmNlpP1RuntimeError, match="process failed"):
        launcher((_payload("first"),))
    with pytest.raises(runtime.EbmNlpP1RuntimeError, match="consumed"):
        launcher((_payload("second"),))
    assert calls == 1


class _CanaryEmbedder:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, texts: Sequence[str]):
        self.calls.append(len(texts))
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            raw = [float(value + 1) for value in digest[:4]]
            norm = math.sqrt(sum(value * value for value in raw))
            rows.append(tuple(value / norm for value in raw))
        return tuple(rows)

    def safe_runtime_receipt(self) -> Mapping[str, object]:
        return {
            "schema": runtime.EMBEDDER_RECEIPT_SCHEMA,
            "status": "complete_offline_deterministic_embeddings",
            "model_tree_sha256": runtime.MINILM_NORMATIVE_TREE_SHA256,
            "weights_sha256": runtime.MINILM_WEIGHTS_SHA256,
            "device": runtime.EMBEDDING_DEVICE,
            "dtype": "float32",
            "embedding_dimension": runtime.EMBEDDING_DIMENSION,
            "batch_size": runtime.EMBEDDING_BATCH_SIZE,
            "maximum_sequence_length": 256,
            "CUBLAS_WORKSPACE_CONFIG": (
                runtime.CUBLAS_WORKSPACE_CONFIG
            ),
            "torch_deterministic_algorithms": True,
            "torch_manual_seed": 0,
            "call_count": len(self.calls),
            "encoded_text_count": sum(self.calls),
            "external_network_call_count": 0,
            "online_or_api_evaluator_call_count": 0,
            "retry_or_replay_count": 0,
        }


class _CanaryHippo:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, payloads):
        self.count += len(payloads)
        return {
            payload["abstract_work_id"]: hippo.retrieve_abstract_with_core(
                core=_Core(), payload=payload
            )
            for payload in payloads
        }

    def safe_runtime_receipt(self) -> Mapping[str, object]:
        return {
            "schema": runtime.HIPPO_RECEIPT_SCHEMA,
            "status": (
                "complete_offline_outputs_verified_indexes_destroyed"
            ),
            "gpu_assignment": ["0", "1"],
            "CUBLAS_WORKSPACE_CONFIG": (
                runtime.CUBLAS_WORKSPACE_CONFIG
            ),
            "maximum_process_count": 2,
            "maximum_processes_per_gpu": 1,
            "configured_cpu_threads_per_process": 1,
            "worker_completed_count": self.count,
            "worker_completed_count_by_gpu": {"0": 1, "1": 1},
            "worker_attempt_count": self.count,
            "observed_process_peak": 2,
            "observed_process_peak_by_gpu": {"0": 1, "1": 1},
            "index_destroyed_count": self.count,
            "batch_invocation_count": 1,
            "HippoRAG_source_tree_sha256": (
                runtime.HIPPORAG_SOURCE_TREE_SHA256
            ),
            "HippoRAG_import_origin_verified_worker_count": self.count,
            "attempted_network_syscall_count": 0,
            "denied_network_syscall_count": 0,
            "external_network_call_count": 0,
            "online_or_api_evaluator_call_count": 0,
            "retry_or_replay_count": 0,
        }


def test_source_free_canary_exercises_full_local_path_without_source() -> None:
    receipt = runtime.run_source_free_full_path_canary(
        embedder=_CanaryEmbedder(),
        hippo_launcher=_CanaryHippo(),
        implementation_freeze_sha256="1" * 64,
        runtime_fingerprint_sha256="2" * 64,
    )
    assert receipt["status"] == "passed_source_free_synthetic_full_path"
    assert receipt["EBM_NLP_archive_path_or_member_access_count"] == 0
    assert receipt["external_network_call_count"] == 0
    assert receipt["synthetic_canary_used_as_efficacy_gate"] is False
    assert receipt["HippoRAG_complete_rank_permutation_count"] == 6


def _write_self_hashed_json(
    path: Path, body: Mapping[str, object], *, canonical: bool
) -> dict[str, object]:
    value = runtime.self_hashed(body)
    if canonical:
        path.write_bytes(runtime.canonical_json_bytes(value))
    else:
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
    return value


def test_implementation_freeze_verifies_actual_files_and_detects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runtime, "_verify_live_module_origins", lambda _root: None
    )
    project = tmp_path / "project"
    project.mkdir()
    bindings = []
    for ordinal, relative in enumerate(
        sorted(runtime._REQUIRED_IMPLEMENTATION_PATHS)
    ):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = f"frozen-{ordinal}-{relative}\n".encode("ascii")
        path.write_bytes(raw)
        bindings.append(
            {
                "relative_path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        )
    manifest_path = tmp_path / "implementation.json"
    manifest = _write_self_hashed_json(
        manifest_path,
        {
            "schema": runtime.IMPLEMENTATION_FREEZE_SCHEMA,
            "status": runtime.IMPLEMENTATION_FREEZE_STATUS,
            "study_id": runtime.core.STUDY_ID,
            "implementation_bindings": bindings,
        },
        canonical=False,
    )
    runtime.verify_implementation_freeze(
        project_root=project,
        manifest_path=manifest_path,
        expected_self_sha256=str(manifest["self_sha256"]),
    )
    drifted = project / sorted(
        runtime._REQUIRED_IMPLEMENTATION_PATHS
    )[0]
    drifted.write_bytes(b"drift\n")
    with pytest.raises(
        runtime.EbmNlpP1RuntimeError,
        match="implementation file drifted",
    ):
        runtime.verify_implementation_freeze(
            project_root=project,
            manifest_path=manifest_path,
            expected_self_sha256=str(manifest["self_sha256"]),
        )


def test_live_module_origins_reject_a_shadow_project(
    tmp_path: Path,
) -> None:
    project = Path(runtime.__file__).resolve().parents[2]
    runtime._verify_live_module_origins(project)
    with pytest.raises(
        runtime.EbmNlpP1RuntimeError,
        match="module origin drifted",
    ):
        runtime._verify_live_module_origins(tmp_path)


def test_receipt_chain_binds_semantic_formal_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    implementation_hash = "1" * 64
    fingerprint_path = tmp_path / "fingerprint.json"
    fingerprint = _write_self_hashed_json(
        fingerprint_path,
        {
            "schema": runtime.FINGERPRINT_SCHEMA,
            "status": "verified_before_formal_source_open",
            "study_id": runtime.core.STUDY_ID,
            "implementation_freeze_sha256": implementation_hash,
            "EBM_NLP_archive_path_or_member_access_count": 0,
            "model_inference_call_count": 0,
            "external_network_call_count": 0,
        },
        canonical=True,
    )
    canary_path = tmp_path / "canary.json"
    canary = runtime.run_source_free_full_path_canary(
        embedder=_CanaryEmbedder(),
        hippo_launcher=_CanaryHippo(),
        implementation_freeze_sha256=implementation_hash,
        runtime_fingerprint_sha256=str(fingerprint["self_sha256"]),
    )
    canary_path.write_bytes(runtime.canonical_json_bytes(canary))
    config = {
        key: f"/bound/{key}"
        for key in runtime._FORMAL_SEMANTIC_CONFIG_KEYS
        if not key.endswith("_sha256")
    }
    config.update(
        {
            "implementation_freeze_sha256": implementation_hash,
            "runtime_fingerprint_receipt": str(fingerprint_path),
            "runtime_fingerprint_sha256": fingerprint["self_sha256"],
            "source_free_canary_receipt": str(canary_path),
            "source_free_canary_sha256": canary["self_sha256"],
            "formal_hostname_sha256": "3" * 64,
            "formal_unit_file_sha256": "4" * 64,
            "systemctl_executable_sha256": "5" * 64,
        }
    )
    monkeypatch.setattr(
        runtime,
        "build_source_free_runtime_fingerprint",
        lambda _paths, *, implementation_freeze_sha256: fingerprint,
    )
    semantic_hash = runtime.formal_semantic_config_sha256(config)
    execution_path = tmp_path / "execution.json"
    execution = _write_self_hashed_json(
        execution_path,
        {
            "schema": runtime.EXECUTION_FREEZE_SCHEMA,
            "status": runtime.EXECUTION_FREEZE_STATUS,
            "study_id": runtime.core.STUDY_ID,
            "implementation_freeze_sha256": implementation_hash,
            "runtime_fingerprint_sha256": fingerprint["self_sha256"],
            "source_free_canary_sha256": canary["self_sha256"],
            "execution_config_sha256": semantic_hash,
            "formal_unit_contract": {
                "unit": config["formal_unit_name"],
                "CPUQuota_percent": 800,
                "MemoryMax_bytes": 40 * 1024**3,
                "TasksMax": 64,
                "Restart": "no",
                "KillMode": "control-group",
                "GPU_assignment": ["0", "1"],
                "CUBLAS_WORKSPACE_CONFIG": (
                    runtime.CUBLAS_WORKSPACE_CONFIG
                ),
                "external_network": "denied",
            },
            "formal_hostname_sha256": config[
                "formal_hostname_sha256"
            ],
            "formal_unit_file_sha256": config[
                "formal_unit_file_sha256"
            ],
            "systemctl_executable_sha256": config[
                "systemctl_executable_sha256"
            ],
        },
        canonical=False,
    )
    config.update(
        {
            "execution_config_sha256": semantic_hash,
            "execution_freeze_manifest": str(execution_path),
            "execution_freeze_sha256": execution["self_sha256"],
        }
    )
    assert runtime._verify_fingerprint_receipt(
        config,
        implementation_freeze_sha256=implementation_hash,
        paths=_paths(tmp_path, "unused-fingerprint-runtime"),
    ) == fingerprint
    assert runtime._verify_canary_receipt(
        config,
        implementation_freeze_sha256=implementation_hash,
        runtime_fingerprint_sha256=str(fingerprint["self_sha256"]),
    ) == canary
    verified, observed_hash = runtime._verify_execution_freeze(
        config,
        implementation_freeze_sha256=implementation_hash,
        runtime_fingerprint_sha256=str(fingerprint["self_sha256"]),
        source_free_canary_sha256=str(canary["self_sha256"]),
    )
    assert verified == execution
    assert observed_hash == semantic_hash
    config["formal_work_root"] = "/drifted/formal_work_root"
    with pytest.raises(
        runtime.EbmNlpP1RuntimeError,
        match="execution freeze prerequisite drifted",
    ):
        runtime._verify_execution_freeze(
            config,
            implementation_freeze_sha256=implementation_hash,
            runtime_fingerprint_sha256=str(
                fingerprint["self_sha256"]
            ),
            source_free_canary_sha256=str(canary["self_sha256"]),
        )


def test_proc_reader_and_live_formal_envelope_are_effective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert runtime._read_proc_regular(
        Path("/proc/self/cmdline"), maximum_bytes=64 * 1024
    )
    paths = _paths(tmp_path, "live-runtime")
    config_path = tmp_path / "formal.config.json"
    config_path.write_text("{}\n", encoding="ascii")
    unit_file = tmp_path / "ebmnlp-p1-formal-v1.service"
    unit_file.write_text("[Service]\nRestart=no\n", encoding="ascii")
    systemctl = Path("/usr/bin/true")
    unit_name = "ebmnlp-p1-formal-v1.service"
    live_output = tmp_path / "live.json"
    config = {
        "formal_unit_name": unit_name,
        "formal_hostname_sha256": hashlib.sha256(
            os.uname().nodename.encode("utf-8")
        ).hexdigest(),
        "formal_unit_file": str(unit_file),
        "formal_unit_file_sha256": runtime.file_sha256(unit_file),
        "systemctl_executable": str(systemctl),
        "systemctl_executable_sha256": runtime.file_sha256(systemctl),
        "live_attestation_output": str(live_output),
    }
    expected_cmdline = (
        str(paths.hippo_python).encode("utf-8")
        + b"\0-m\0"
        + b"assumption_agent.benchmarks.ebmnlp_p1_runtime_v1"
        + b"\0formal\0--config\0"
        + str(config_path.absolute()).encode("utf-8")
        + b"\0"
    )
    control_group = (
        "/user.slice/user-1000.slice/user@1000.service/app.slice/"
        + unit_name
    )

    def fake_proc(path: Path, *, maximum_bytes: int) -> bytes:
        assert maximum_bytes == 64 * 1024
        if path.name == "cmdline":
            return expected_cmdline
        if path.name == "cgroup":
            return f"0::{control_group}\n".encode("utf-8")
        assert path.name == "environ"
        return (
            b"\0".join(
                f"{key}={value}".encode("utf-8")
                for key, value in runtime.FORMAL_CLEAN_ENVIRONMENT.items()
            )
            + b"\0"
        )

    properties = {
        "ActiveState": "active",
        "CPUQuotaPerSecUSec": "8s",
        "ControlGroup": control_group,
        "ExecStart": (
            f"{paths.hippo_python} -m "
            "assumption_agent.benchmarks.ebmnlp_p1_runtime_v1 "
            f"formal --config {config_path.absolute()}"
        ),
        "FragmentPath": str(unit_file),
        "IPAddressDeny": "0.0.0.0/0 ::/0",
        "KillMode": "control-group",
        "MainPID": str(os.getpid()),
        "MemoryMax": str(40 * 1024**3),
        "Restart": "no",
        "RestrictAddressFamilies": "AF_UNIX",
        "SubState": "running",
        "TasksMax": "64",
    }
    monkeypatch.setattr(runtime, "_read_proc_regular", fake_proc)
    receipt = runtime.verify_live_formal_execution(
        config=config,
        config_path=config_path,
        paths=paths,
        command_runner=lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [],
            0,
            "\n".join(
                f"{key}={value}" for key, value in properties.items()
            )
            + "\n",
            "",
        ),
    )
    assert receipt["status"].startswith("verified_effective")
    assert json.loads(live_output.read_text("ascii")) == receipt
