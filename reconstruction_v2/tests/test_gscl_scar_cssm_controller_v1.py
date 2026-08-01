from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

import pytest

from replication_runtime.gscl_scar_cssm_v1 import controller


STUDY_ID = "SCAR_CSSM_CONTROLLER_FIXTURE_V1"
ACTION_COMMITMENT = hashlib.sha256(b"fixture-action").hexdigest()
QWEN_RUNTIME = hashlib.sha256(b"fixture-qwen-runtime").hexdigest()
QWEN_CANARY = hashlib.sha256(b"fixture-qwen-canary").hexdigest()
ENCODER_BINDING = hashlib.sha256(b"fixture-encoder").hexdigest()
IMPLEMENTATION_CLOSURE = hashlib.sha256(b"fixture-implementation").hexdigest()
EXECUTION_FREEZE = hashlib.sha256(b"fixture-execution-freeze").hexdigest()
GPU_UUIDS = (
    "GPU-00000000-0000-0000-0000-000000000001",
    "GPU-00000000-0000-0000-0000-000000000002",
)
POLICY = controller.MeasurementPolicy(
    action_item_count=4,
    primary_item_count=3,
    ambiguous_item_count=1,
)


@pytest.fixture
def linux_tmp_path() -> Path:
    root = Path(tempfile.mkdtemp(prefix="scar-cssm-controller-", dir="/tmp"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _token(index: int) -> str:
    return f"scar-item-v1-{index:064x}"


def _slot(index: int) -> str:
    return f"scar-slot-v1-{index:064x}"


def _side(index: int, *, collision: bool = False) -> dict[str, Any]:
    surfaces = ("K", "K") if collision else (f"left {index}", f"right {index}")
    return {
        "background": f"background {index}",
        "slots": [
            {"opaque_slot_id": _slot(index * 10 + 1), "surface": surfaces[0]},
            {"opaque_slot_id": _slot(index * 10 + 2), "surface": surfaces[1]},
        ],
        "system": f"system {index}",
    }


def _action_item(index: int, *, collision: bool = False) -> dict[str, Any]:
    left = _side(index * 2, collision=collision)
    right = _side(index * 2 + 1)
    return {
        "item_token": _token(index),
        "variants": {
            "base": {"left": left, "right": right},
            "system_swap": {"left": right, "right": left},
        },
    }


def _write_private(path: Path, raw: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o600)


def _fixture_config(tmp_path: Path) -> tuple[controller.FormalConfig, dict[str, Any]]:
    inputs = tmp_path / "inputs"
    inputs.mkdir(mode=0o700)
    action_pack = {
        "action_commitment_sha256": ACTION_COMMITMENT,
        "items": [
            _action_item(1),
            _action_item(2, collision=True),
            _action_item(3),
            _action_item(4),
        ],
        "study_id": STUDY_ID,
        "variant_names": list(controller.VARIANT_NAMES),
    }
    label_pack = {
        "fixture": "late labels",
        "study_id": STUDY_ID,
    }
    action_path = inputs / "action.private.json"
    label_path = inputs / "label.private.json"
    secret_path = inputs / "secret.private.bin"
    qwen_manifest = inputs / "qwen.manifest.json"
    minilm_manifest = inputs / "minilm.manifest.json"
    sandbox_path = inputs / "sandbox.safe.json"
    _write_private(action_path, _canonical(action_pack))
    _write_private(label_path, _canonical(label_pack))
    _write_private(secret_path, bytes(range(32)))
    _write_private(qwen_manifest, _canonical({"fixture": "qwen"}))
    _write_private(minilm_manifest, _canonical({"fixture": "minilm"}))
    sandbox_body = {
        "action_external_network_denied": True,
        "action_label_path_denied": True,
        "ip_address_deny": "any",
        "restrict_address_families": "AF_UNIX",
        "schema": "gscl_scar_cssm_sandbox_freeze_v1",
        "status": "frozen",
        "study_id": STUDY_ID,
    }
    sandbox = {**sandbox_body, "self_sha256": _hash(sandbox_body)}
    _write_private(sandbox_path, _canonical(sandbox))
    qwen_root = inputs / "qwen"
    minilm_root = inputs / "minilm"
    qwen_root.mkdir()
    minilm_root.mkdir()

    project = Path(controller.__file__).parents[2]
    implementation_paths = {
        "action_implementation_file_sha256": (
            project / "assumption_agent/benchmarks/gscl_scar_cssm_action_v1.py"
        ),
        "controller_file_sha256": Path(controller.__file__),
        "scorer_implementation_file_sha256": (
            project / "assumption_agent/benchmarks/gscl_scar_cssm_score_v1.py"
        ),
        "source_implementation_file_sha256": (
            project / "assumption_agent/benchmarks/gscl_scar_cssm_source_v1.py"
        ),
        "worker_file_sha256": (
            project / "replication_runtime/gscl_scar_cssm_v1/worker.py"
        ),
        "qwen_manifest_file_sha256": qwen_manifest,
        "minilm_manifest_file_sha256": minilm_manifest,
        "python_executable_file_sha256": Path(os.path.realpath(os.sys.executable)),
        "sandbox_receipt_file_sha256": sandbox_path,
    }
    bindings = {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in implementation_paths.items()
    }
    bindings.update(
        {
            "action_pack_file_sha256": hashlib.sha256(action_path.read_bytes()).hexdigest(),
            "action_pack_commitment_sha256": ACTION_COMMITMENT,
            "execution_freeze_sha256": EXECUTION_FREEZE,
            "implementation_closure_sha256": IMPLEMENTATION_CLOSURE,
            "label_pack_file_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
            "secret_file_sha256": hashlib.sha256(secret_path.read_bytes()).hexdigest(),
            "qwen_runtime_commitment": QWEN_RUNTIME,
            "qwen_canary_self_sha256": QWEN_CANARY,
            "encoder_binding_sha256": ENCODER_BINDING,
            "sandbox_receipt_self_sha256": sandbox["self_sha256"],
        }
    )
    initial = controller.FormalConfig(
        study_id=STUDY_ID,
        mutable_root=tmp_path / "formal",
        project_root=project,
        action_pack_path=action_path,
        label_pack_path=label_path,
        secret_path=secret_path,
        python_executable=Path(os.path.realpath(os.sys.executable)),
        qwen_model_root=qwen_root,
        qwen_manifest_path=qwen_manifest,
        minilm_model_root=minilm_root,
        minilm_manifest_path=minilm_manifest,
        sandbox_receipt_path=sandbox_path,
        nvidia_smi_path=Path(os.path.realpath(os.sys.executable)),
        gpu_uuids=GPU_UUIDS,
        lock_path=tmp_path / "locks" / "scar.lock",
        minimum_gpu_free_mib=1,
        minimum_host_available_bytes=1,
        bindings=bindings,
        self_sha256="0" * 64,
    )
    config = replace(initial, self_sha256=_hash(initial.body()))
    return config, action_pack


def _action_row(item: dict[str, Any], *, corrupt: bool = False) -> dict[str, Any]:
    collision = item["item_token"] == _token(2)
    if corrupt and item["item_token"] == _token(1):
        collision = True
    execution = (
        {
            "document_call_count": 0,
            "error_code": "SLOT_BINDER_TYPED_FAILURE",
            "structural_status": "TYPED_FAILURE",
        }
        if collision
        else {
            "document_call_count": 2,
            "error_code": None,
            "structural_status": "EXECUTED_WITHOUT_TYPED_FAILURE",
        }
    )
    variants = {
        variant_name: {"arms": {arm_id: {} for arm_id in controller.ARM_IDS}}
        for variant_name in controller.VARIANT_NAMES
    }
    diagnostics = {
        variant_name: {
            "arms": {arm_id: {} for arm_id in controller.ARM_IDS},
            "left_binder": None,
            "left_graph_receipt_sha256": None,
            "mapping_receipt_sha256_by_arm": {
                arm_id: None for arm_id in controller.ARM_IDS
            },
            "right_binder": None,
            "right_graph_receipt_sha256": None,
            "structural_diagnostics_available": False,
            "target_color_shuffle_effective": None,
        }
        for variant_name in controller.VARIANT_NAMES
    }
    pools = {
        variant_name: {"semantic_kbest": [], "structure_kbest": []}
        for variant_name in controller.VARIANT_NAMES
    }
    return {
        "diagnostics": diagnostics,
        "execution": execution,
        "item_token": item["item_token"],
        "proposal_pools": pools,
        "variants": variants,
    }


class _FakePopenFactory:
    def __init__(
        self,
        *,
        config: controller.FormalConfig,
        action_pack: dict[str, Any],
        corrupt: bool = False,
    ) -> None:
        self.config = config
        self.action_pack = action_pack
        self.corrupt = corrupt
        self.launches: list[dict[str, Any]] = []
        self.waits: list[int] = []

    def __call__(self, argv, **kwargs):
        shard = int(argv[argv.index("--shard-index") + 1])
        output_root = Path(argv[argv.index("--output-root") + 1])
        self.launches.append({"argv": list(argv), "kwargs": kwargs, "shard": shard})
        sandbox = json.loads(self.config.sandbox_receipt_path.read_text(encoding="ascii"))
        runtime_body = {
            "execution": {
                "cublas_workspace_config": ":4096:8",
                "cudnn_benchmark": False,
                "cudnn_tf32": False,
                "deterministic_algorithms": True,
                "hf_hub_offline": "1",
                "hf_hub_disable_telemetry": "1",
                "matmul_tf32": False,
                "python": {
                    "executable_sha256": self.config.bindings[
                        "python_executable_file_sha256"
                    ]
                },
                "python_no_user_site": "1",
                "supervisor_landlock_direct_parent_authority": (
                    "97ff3a77c33a3113712a4c11a9fd347902a12b45f76935023d2ac66377936c35"
                ),
                "transformers_offline": "1",
            },
            "execution_freeze_sha256": EXECUTION_FREEZE,
            "forbidden_label_negative_canary": {
                "errno": 13,
                "open_denied": True,
                "read_count": 0,
            },
            "gpu": {
                "cuda_visible_devices": GPU_UUIDS[shard],
                "logical_current_device": 0,
                "parameter_devices": ["cuda:0"],
                "physical_uuid": GPU_UUIDS[shard],
                "visible_device_count": 1,
            },
            "implementation_closure": {"self_sha256": IMPLEMENTATION_CLOSURE},
            "minilm": {"encoder_binding_sha256": ENCODER_BINDING},
            "network_negative_canary": {
                "AF_INET": {"creation_denied": True, "errno": 1},
                "AF_INET6": {"creation_denied": True, "errno": 1},
                "external_connect_attempt_count": 0,
            },
            "process_sandbox": {"no_new_privileges": True},
            "qwen": {
                "qualification_canary": {"self_sha256": QWEN_CANARY},
                "runtime_commitment": QWEN_RUNTIME,
            },
            "sandbox_freeze": sandbox,
            "sandbox_freeze_file": {
                "sha256": self.config.bindings["sandbox_receipt_file_sha256"]
            },
            "schema": "gscl_scar_cssm_worker_v1.runtime.safe_receipt.v1",
            "shard_count": 2,
            "shard_index": shard,
            "status": "qualified_before_action_pack_open",
            "study_id": STUDY_ID,
            "version": "gscl_scar_cssm_worker_v1",
        }
        runtime = {**runtime_body, "self_sha256": _hash(runtime_body)}
        runtime_raw = _canonical(runtime)
        _write_private(output_root / f"shard{shard}.runtime.safe.json", runtime_raw)
        sentinel = {
            "expected_action_commitment_sha256": ACTION_COMMITMENT,
            "expected_action_file_sha256": self.config.bindings[
                "action_pack_file_sha256"
            ],
            "expected_execution_freeze_sha256": EXECUTION_FREEZE,
            "runtime_receipt_sha256": hashlib.sha256(runtime_raw).hexdigest(),
            "shard_count": 2,
            "shard_index": shard,
            "study_id": STUDY_ID,
            "version": "gscl_scar_cssm_worker_v1",
        }
        _write_private(output_root / f"shard{shard}.attempt.sentinel", _canonical(sentinel))
        factory = self

        class Process:
            returncode: int | None = None

            def poll(self) -> int | None:
                return self.returncode

            def terminate(self) -> None:
                self.returncode = -15

            def wait(self) -> int:
                # The first wait is legal only after both logical actions were
                # submitted and the controller released both runtime-qualified
                # shards together.
                assert len(factory.launches) == 2
                if self.returncode is not None:
                    factory.waits.append(shard)
                    return self.returncode
                release_path = Path(
                    argv[argv.index("--action-release") + 1]
                )
                assert release_path.exists()
                release_raw = release_path.read_bytes()
                release = json.loads(release_raw.decode("ascii"))
                selected = [
                    item
                    for ordinal, item in enumerate(factory.action_pack["items"])
                    if ordinal % 2 == shard
                ]
                predictions = [
                    _action_row(item, corrupt=factory.corrupt) for item in selected
                ]
                records = []
                for ordinal, prediction in enumerate(predictions):
                    evidence = {
                        "availability": (
                            "PREMODEL_TYPED_FAILURE"
                            if prediction["execution"]["structural_status"]
                            == "TYPED_FAILURE"
                            else "COMPLETE"
                        ),
                        "error_code": prediction["execution"]["error_code"],
                        "semantic_matrix": None,
                        "sides": {"left": None, "right": None},
                        "variants": {
                            variant_name: None
                            for variant_name in controller.VARIANT_NAMES
                        },
                    }
                    record_body = {
                        "evidence": evidence,
                        "item_token": prediction["item_token"],
                        "ordinal_within_shard": ordinal,
                        "prediction": prediction,
                    }
                    records.append(
                        {**record_body, "self_sha256": _hash(record_body)}
                    )
                private_raw = b"".join(
                    _canonical(record) + b"\n" for record in records
                )
                private_path = output_root / f"shard{shard}.records.private.jsonl"
                _write_private(private_path, private_raw)
                calls = sum(
                    row["execution"]["document_call_count"] for row in predictions
                )
                failures = sum(
                    row["execution"]["structural_status"] == "TYPED_FAILURE"
                    for row in predictions
                )
                body = {
                    "action_commitment_sha256": ACTION_COMMITMENT,
                    "action_pack_file_receipt": {
                        "mode_octal": "0600",
                        "sha256": factory.config.bindings[
                            "action_pack_file_sha256"
                        ],
                    },
                    "action_release_file_receipt": {
                        "mode_octal": "0600",
                        "sha256": hashlib.sha256(release_raw).hexdigest(),
                    },
                    "action_release_self_sha256": release["self_sha256"],
                    "arm_ids": list(controller.ARM_IDS),
                    "document_call_count": calls,
                    "encoder_binding_sha256": ENCODER_BINDING,
                    "external_network_call_count": 0,
                    "formal_label_pack_access_count": 0,
                    "formal_scorer_access_count": 0,
                    "item_count": len(predictions),
                    "mechanism_resource_totals": {"fixture_count": 0},
                    "output_root_receipt": {
                        "filesystem_type": "ext4",
                        "mode_octal": "0700",
                    },
                    "private_records_file_sha256": hashlib.sha256(
                        private_raw
                    ).hexdigest(),
                    "private_records_file_size_bytes": len(private_raw),
                    "runtime_receipt_file_sha256": hashlib.sha256(
                        runtime_raw
                    ).hexdigest(),
                    "runtime_receipt_self_sha256": runtime["self_sha256"],
                    "schema": controller.SHARD_TERMINAL_SCHEMA,
                    "shard_count": 2,
                    "shard_index": shard,
                    "status": "complete",
                    "structural_error_code_counts": (
                        {"SLOT_BINDER_TYPED_FAILURE": failures}
                        if failures
                        else {}
                    ),
                    "structural_typed_failure_count": failures,
                    "study_id": STUDY_ID,
                    "variant_names": list(controller.VARIANT_NAMES),
                    "version": "gscl_scar_cssm_worker_v1",
                }
                terminal = {**body, "self_sha256": _hash(body)}
                _write_private(
                    output_root / f"shard{shard}.terminal.safe.json",
                    _canonical(terminal),
                )
                self.returncode = 0
                factory.waits.append(shard)
                return self.returncode

        return Process()


def _score_result(study_id: str) -> SimpleNamespace:
    private_body = {
        "per_item": [{"private": True}],
        "schema": "fixture.private.v1",
        "status": "SCORED_OFFLINE_ONCE",
        "study_id": study_id,
    }
    safe_body = {
        "aggregate": {"case_count": 4},
        "schema": "fixture.safe.v1",
        "status": "SCORED_OFFLINE_ONCE",
        "study_id": study_id,
    }
    return SimpleNamespace(
        private_result={**private_body, "self_sha256": _hash(private_body)},
        safe_aggregate={**safe_body, "self_sha256": _hash(safe_body)},
    )


def _dependencies(
    config: controller.FormalConfig,
    action_pack: dict[str, Any],
    *,
    events: list[str],
    corrupt: bool = False,
    admission: str = "ADMITTED_SHARED_RESOURCE",
) -> tuple[controller.ControllerDependencies, _FakePopenFactory]:
    popen = _FakePopenFactory(
        config=config,
        action_pack=action_pack,
        corrupt=corrupt,
    )

    def validate(pack, study_id):
        events.append("validate_action")
        assert pack == action_pack
        assert study_id == STUDY_ID

    def score(action, labels, prediction, *, secret, study_id):
        events.append("score")
        assert (config.mutable_root / "control/action_barrier.safe.json").exists()
        assert action == action_pack
        assert labels["fixture"] == "late labels"
        assert len(prediction["items"]) == 4
        assert secret == bytes(range(32))
        assert study_id == STUDY_ID
        return _score_result(study_id)

    deps = controller.ControllerDependencies(
        filesystem_type=lambda _path: "ext4",
        resource_probe=lambda _config: controller.AdmissionDecision(
            status=admission,
            reason_codes=("BUSY",) if admission != "ADMITTED_SHARED_RESOURCE" else (),
            host_mem_available_bytes=10_000,
            selected_gpu_free_mib=(20_000, 20_000),
        ),
        popen_factory=popen,
        validate_action_pack=validate,
        score_once=score,
    )
    return deps, popen


def test_formal_lifecycle_is_two_shard_label_late_and_exactly_once(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, action_pack = _fixture_config(linux_tmp_path)
    events: list[str] = []
    deps, popen = _dependencies(config, action_pack, events=events)
    original_read = controller._read_regular_once
    formal_reads: list[Path] = []

    def observed_read(path: Path, **kwargs):
        if path in {config.action_pack_path, config.label_pack_path, config.secret_path}:
            formal_reads.append(path)
        if path in {config.label_pack_path, config.secret_path}:
            assert (config.mutable_root / "control/action_barrier.safe.json").exists()
        return original_read(path, **kwargs)

    monkeypatch.setattr(controller, "_read_regular_once", observed_read)
    terminal = controller.run_formal_once(config, dependencies=deps, policy=POLICY)

    assert terminal["status"] == "completed_protocol_valid"
    assert terminal["action_child_launch_count"] == 2
    assert terminal["action_release_count"] == 1
    assert terminal["action_barrier_count"] == 1
    assert terminal["label_pack_access_count"] == 1
    assert terminal["secret_access_count"] == 1
    assert terminal["offline_scorer_call_count"] == 1
    assert events == ["validate_action", "score"]
    assert formal_reads.count(config.action_pack_path) == 1
    assert formal_reads.count(config.label_pack_path) == 1
    assert formal_reads.count(config.secret_path) == 1
    assert [row["shard"] for row in popen.launches] == [0, 1]
    assert set(popen.waits) == {0, 1}
    release_path = (
        config.mutable_root / "control/two_shard_action_release.safe.json"
    )
    release = json.loads(release_path.read_text(encoding="ascii"))
    assert release["gpu_uuid_by_shard"] == {
        "0": GPU_UUIDS[0],
        "1": GPU_UUIDS[1],
    }
    assert set(release["runtime_receipt_file_sha256_by_shard"]) == {"0", "1"}
    for shard, (launch, gpu) in enumerate(
        zip(popen.launches, GPU_UUIDS, strict=True)
    ):
        assert launch["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == gpu
        assert launch["kwargs"]["env"]["HF_HUB_OFFLINE"] == "1"
        assert launch["kwargs"]["env"]["CUBLAS_WORKSPACE_CONFIG"] == (
            ":4096:8"
        )
        assert launch["kwargs"]["env"][
            "GSCL_SUPERVISOR_LANDLOCK_DIRECT_PARENT_AUTHORITY_V1"
        ] == (
            "97ff3a77c33a3113712a4c11a9fd347902a12b45f76935023d2ac66377936c35"
        )
        assert launch["kwargs"]["env"]["TRANSFORMERS_OFFLINE"] == "1"
        assert callable(launch["kwargs"]["preexec_fn"])
        argv = launch["argv"]
        assert argv[argv.index("--action-release") + 1] == str(release_path)
        assert argv[argv.index("--forbidden-label-probe") + 1] == str(
            config.label_pack_path
        )
        assert argv.count(str(config.label_pack_path)) == 1
        assert argv[argv.index("--expected-gpu-uuid") + 1] == gpu
        assert argv[argv.index("--expected-peer-gpu-uuid") + 1] == GPU_UUIDS[
            1 - shard
        ]
        assert str(config.secret_path) not in argv
    for name in (
        "two_shard_action_release.safe.json",
        "action_barrier.safe.json",
        "prediction_pack.private.json",
        "score.private.json",
        "score.safe.json",
        "formal_terminal.private.json",
        "formal_terminal.safe.json",
    ):
        path = config.mutable_root / "control" / name
        assert path.exists()
        assert stat_mode(path) == 0o600

    with pytest.raises(controller.ScarCssmControllerAlreadyConsumed):
        controller.run_formal_once(config, dependencies=deps, policy=POLICY)
    assert events.count("score") == 1


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_shared_node_deferral_is_zero_attempt_and_reads_no_formal_input(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, action_pack = _fixture_config(linux_tmp_path)
    events: list[str] = []
    deps, popen = _dependencies(
        config,
        action_pack,
        events=events,
        admission="DEFERRED_SHARED_RESOURCE",
    )
    original_read = controller._read_regular_once
    formal_reads: list[Path] = []

    def observed_read(path: Path, **kwargs):
        if path in {config.action_pack_path, config.label_pack_path, config.secret_path}:
            formal_reads.append(path)
        return original_read(path, **kwargs)

    monkeypatch.setattr(controller, "_read_regular_once", observed_read)
    result = controller.run_formal_once(config, dependencies=deps, policy=POLICY)

    assert result["status"] == "DEFERRED_SHARED_RESOURCE"
    assert result["effect_study_attempt_count"] == 0
    assert result["formal_input_access_count"] == 0
    assert not config.mutable_root.exists()
    assert formal_reads == []
    assert popen.launches == []
    assert events == []


def test_unexpected_primary_typed_failure_terminates_before_label_or_scorer(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, action_pack = _fixture_config(linux_tmp_path)
    events: list[str] = []
    deps, _popen = _dependencies(
        config,
        action_pack,
        events=events,
        corrupt=True,
    )
    original_read = controller._read_regular_once
    late_reads: list[Path] = []

    def observed_read(path: Path, **kwargs):
        if path in {config.label_pack_path, config.secret_path}:
            late_reads.append(path)
        return original_read(path, **kwargs)

    monkeypatch.setattr(controller, "_read_regular_once", observed_read)
    terminal = controller.run_formal_once(config, dependencies=deps, policy=POLICY)

    assert terminal["status"] == "failed_after_formal_attempt"
    assert terminal["failure_stage"] == "validate_and_seal_action_closure"
    assert terminal["action_child_launch_count"] == 2
    assert terminal["action_release_count"] == 1
    assert terminal["action_barrier_count"] == 0
    assert terminal["label_pack_access_count"] == 0
    assert terminal["secret_access_count"] == 0
    assert terminal["offline_scorer_call_count"] == 0
    assert late_reads == []
    assert events == ["validate_action"]
    assert not (config.mutable_root / "control/action_barrier.safe.json").exists()


def test_config_round_trip_is_canonical_and_self_bound(linux_tmp_path: Path) -> None:
    config, _action_pack = _fixture_config(linux_tmp_path)
    value = {**config.body(), "self_sha256": config.self_sha256}
    path = linux_tmp_path / "formal.config.json"
    _write_private(path, _canonical(value))

    loaded = controller.load_config(path)

    assert loaded == config
    value["minimum_gpu_free_mib"] += 1
    _write_private(linux_tmp_path / "tampered.config.json", _canonical(value))
    with pytest.raises(controller.ScarCssmControllerError):
        controller.load_config(linux_tmp_path / "tampered.config.json")


@pytest.mark.skipif(sys.platform != "linux", reason="Landlock is Linux-only")
def test_landlock_child_can_read_bound_action_but_denies_label(
    linux_tmp_path: Path,
) -> None:
    allowed = linux_tmp_path / "action.private.json"
    forbidden = linux_tmp_path / "label.private.json"
    _write_private(allowed, b"action")
    _write_private(forbidden, b"label")
    project_root = Path(controller.__file__).parents[2]
    script = """
import os
from pathlib import Path
import sys
from replication_runtime.gscl_scar_cssm_v1 import controller

allowed = Path(sys.argv[1])
forbidden = Path(sys.argv[2])
controller._apply_landlock(read_paths=(allowed,), write_paths=(), device_paths=())
allowed_ok = allowed.read_bytes() == b"action"
try:
    forbidden.read_bytes()
except PermissionError:
    denied = True
else:
    denied = False
os.write(1, b"allowed_and_denied" if allowed_ok and denied else b"sandbox_failed")
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root)
    completed = subprocess.run(
        [sys.executable, "-c", script, str(allowed), str(forbidden)],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8", errors="replace"
    )
    assert completed.stdout == b"allowed_and_denied"


def test_action_landlock_grants_only_process_local_cuda_thread_metadata_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        controller,
        "_action_landlock_paths",
        lambda _config, _paths, _shard: ((), (), ()),
    )
    monkeypatch.setattr(
        controller,
        "_apply_landlock",
        lambda **kwargs: captured.update(kwargs),
    )

    config = SimpleNamespace(
        qwen_manifest_path=Path("/assets/qwen.json"),
        minilm_manifest_path=Path("/minilm/minilm.json"),
        action_pack_path=Path("/inputs/action.json"),
        sandbox_receipt_path=Path("/sandbox/sandbox.json"),
    )
    paths = SimpleNamespace(control=Path("/control"))
    controller._apply_action_landlock(config, paths, 0)

    assert captured == {
        "read_paths": (),
        "write_paths": (),
        "device_paths": (),
        "read_directory_paths": (
            Path("/assets"),
            Path("/minilm"),
            Path("/inputs"),
            Path("/sandbox"),
            Path("/control"),
        ),
        "write_file_paths": (Path("/proc/self/task"),),
    }
    assert controller._landlock_write_file_rights(2) == (  # noqa: SLF001
        controller._LL_WRITE_FILE  # noqa: SLF001
    )
    assert controller._landlock_write_file_rights(3) == (  # noqa: SLF001
        controller._LL_WRITE_FILE  # noqa: SLF001
        | controller._LL_TRUNCATE  # noqa: SLF001
    )


def test_canonical_worker_record_roundtrip_is_accepted_by_frozen_scorer() -> None:
    from assumption_agent.benchmarks import gscl_scar_cssm_score_v1 as scorer
    from tests import test_gscl_scar_cssm_score_v1 as score_fixture

    action_pack, label_pack, sealed_fixture = score_fixture._fixture()
    original_rows = sealed_fixture["items"]
    restored_rows: list[dict[str, Any]] = []
    for ordinal, original in enumerate(original_rows):
        prediction = {
            key: value
            for key, value in original.items()
            if key != "private_mechanism_receipts"
        }
        body = {
            "evidence": original["private_mechanism_receipts"],
            "item_token": original["item_token"],
            "ordinal_within_shard": ordinal,
            "prediction": prediction,
        }
        worker_record = {**body, "self_sha256": _hash(body)}
        decoded = controller._strict_json(
            _canonical(worker_record), field="CONTROLLER_TEST_WORKER_RECORD"
        )
        # Canonical JSON transport alphabetizes ARM_IDS and reproduces the
        # exact production incompatibility this regression guards.
        assert tuple(
            decoded["prediction"]["variants"]["base"]["arms"]
        ) != scorer.ARM_IDS
        restored = controller._restore_scorer_wire_order(
            decoded["prediction"], decoded["evidence"]
        )
        assert tuple(restored["variants"]["base"]["arms"]) == scorer.ARM_IDS
        assert tuple(
            restored["diagnostics"]["base"]["mapping_receipt_sha256_by_arm"]
        ) == scorer.ARM_IDS
        assert tuple(restored["diagnostics"]["base"]["arms"]) == scorer.ARM_IDS
        restored_rows.append(restored)

    prediction_pack = controller._prediction_pack(
        action_commitment=action_pack["action_commitment_sha256"],
        rows=restored_rows,
        study_id=score_fixture._STUDY_ID,
    )
    result = scorer._score_scar_cssm_fixture_v1(
        action_pack,
        label_pack,
        prediction_pack,
        secret=score_fixture._SECRET,
        study_id=score_fixture._STUDY_ID,
        expected_primary_count=1,
        expected_ambiguous_count=1,
    )
    assert result.safe_aggregate["status"] == "SCORED_OFFLINE_ONCE"


def test_python_runtime_roots_include_declared_venv_base(
    linux_tmp_path: Path,
) -> None:
    runtime = linux_tmp_path / "runtime"
    environment = runtime / "typed_venv"
    executable = environment / "bin/python"
    base_bin = runtime / "python310/bin"
    executable.parent.mkdir(parents=True)
    base_bin.mkdir(parents=True)
    executable.write_bytes(b"python")
    (environment / "pyvenv.cfg").write_text(
        f"home = {base_bin}\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )

    assert controller._python_runtime_read_roots(  # noqa: SLF001
        executable
    ) == (environment, runtime / "python310")
