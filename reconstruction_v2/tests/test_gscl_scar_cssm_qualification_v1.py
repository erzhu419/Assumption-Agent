from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unicodedata
from typing import Any

import pytest

from replication_runtime.gscl_scar_cssm_v1 import qualification
from replication_runtime.gscl_scar_cssm_v1 import worker


_STUDY_ID = "SCAR_CSSM_PUBLIC_RUNTIME_QUALIFICATION_FIXTURE_V1"
_GPU0 = "GPU-32d6e292-70cd-50a0-405b-e344d2da8d39"
_GPU1 = "GPU-db2137c8-0f6b-b790-a698-6bfbbd5dc9eb"
_EXECUTION = hashlib.sha256(b"qualification-execution-freeze").hexdigest()


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


def _publish(path: Path, value: Any, *, newline: bool = False) -> str:
    raw = value if isinstance(value, bytes) else _canonical(value)
    if newline:
        raw += b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o600)
    return hashlib.sha256(raw).hexdigest()


def _config(tmp_path: Path) -> qualification.QualificationConfig:
    project = Path(qualification.__file__).resolve().parents[2]
    qwen = tmp_path / "assets/qwen"
    minilm = tmp_path / "assets/minilm"
    qwen.mkdir(parents=True)
    minilm.mkdir(parents=True)
    qwen_manifest = tmp_path / "assets/qwen.manifest.json"
    minilm_manifest = tmp_path / "assets/minilm.manifest.json"
    _publish(qwen_manifest, {"public": "qwen"})
    _publish(minilm_manifest, {"public": "minilm"})
    return qualification.QualificationConfig(
        study_id=_STUDY_ID,
        root=tmp_path / "qualification",
        project_root=project,
        python_executable=Path(os.path.realpath(sys.executable)),
        qwen_model_root=qwen,
        qwen_manifest_path=qwen_manifest,
        minilm_model_root=minilm,
        minilm_manifest_path=minilm_manifest,
        nvidia_smi_path=Path(os.path.realpath(sys.executable)),
        gpu_uuids=(_GPU0, _GPU1),
        execution_freeze_sha256=_EXECUTION,
        lock_path=tmp_path / "locks/qualification.lock",
        runtime_barrier_timeout_seconds=10,
    )


def _runtime_receipt(
    *,
    config: qualification.QualificationConfig,
    shard: int,
    implementation_hash: str,
    sandbox: dict[str, Any],
    sandbox_file_hash: str,
) -> dict[str, Any]:
    body = {
        "execution": {
            "cublas_workspace_config": ":4096:8",
            "cuda_runtime_available": True,
            "cudnn_benchmark": False,
            "cudnn_tf32": False,
            "deterministic_algorithms": True,
            "hf_hub_offline": "1",
            "hf_hub_disable_telemetry": "1",
            "matmul_tf32": False,
            "python_no_user_site": "1",
            "supervisor_landlock_direct_parent_authority": (
                "97ff3a77c33a3113712a4c11a9fd347902a12b45f76935023d2ac66377936c35"
            ),
            "tokenizers_parallelism": "false",
            "transformers_offline": "1",
        },
        "execution_freeze_sha256": config.execution_freeze_sha256,
        "forbidden_label_negative_canary": {
            "errno": 13,
            "open_denied": True,
            "read_count": 0,
        },
        "gpu": {
            "cuda_visible_devices": config.gpu_uuids[shard],
            "logical_current_device": 0,
            "parameter_devices": ["cuda:0"],
            "physical_uuid": config.gpu_uuids[shard],
            "visible_device_count": 1,
        },
        "implementation_closure": {"self_sha256": implementation_hash},
        "minilm": {"encoder_binding_sha256": "d" * 64},
        "network_negative_canary": {
            "AF_INET": {"creation_denied": True, "errno": 1},
            "AF_INET6": {"creation_denied": True, "errno": 1},
            "external_connect_attempt_count": 0,
        },
        "process_sandbox": {
            "no_new_privileges": True,
            "seccomp_filter_count": "1",
            "seccomp_mode": "2",
        },
        "qwen": {"runtime_commitment": "c" * 64},
        "sandbox_freeze": sandbox,
        "sandbox_freeze_file": {"sha256": sandbox_file_hash},
        "schema": worker.RUNTIME_RECEIPT_SCHEMA,
        "shard_count": 2,
        "shard_index": shard,
        "status": "qualified_before_action_pack_open",
        "study_id": config.study_id,
        "version": worker.VERSION,
    }
    return {**body, "self_sha256": _hash(body)}


def _complete_evidence() -> dict[str, Any]:
    resource = {
        "leaf_call_count": 0,
        "reported_success_candidate_count": 0,
        "reported_success_forward_batch_count": 0,
    }
    side = {
        "binder": {},
        "bounded_set": {},
        "document_envelope": {
            "leaf_records": [],
            "receipt": {"receipt": {"resource_summary": resource}},
        },
        "slot_graph": {},
    }
    mapping = {
        "semantic_mapping": {},
        "structural_mapping": {},
        "target_color_shuffle_mapping": {},
    }
    return {
        "availability": "COMPLETE",
        "error_code": None,
        "semantic_matrix": {},
        "sides": {"left": copy.deepcopy(side), "right": copy.deepcopy(side)},
        "variants": {
            "base": copy.deepcopy(mapping),
            "system_swap": copy.deepcopy(mapping),
        },
    }


def _premodel_evidence() -> dict[str, Any]:
    return {
        "availability": "PREMODEL_TYPED_FAILURE",
        "error_code": "SLOT_BINDER_TYPED_FAILURE",
        "semantic_matrix": None,
        "sides": {"left": None, "right": None},
        "variants": {"base": None, "system_swap": None},
    }


def _prediction(item_token: str, *, executable: bool) -> dict[str, Any]:
    execution = (
        {
            "document_call_count": 2,
            "error_code": None,
            "structural_status": "EXECUTED_WITHOUT_TYPED_FAILURE",
        }
        if executable
        else {
            "document_call_count": 0,
            "error_code": "SLOT_BINDER_TYPED_FAILURE",
            "structural_status": "TYPED_FAILURE",
        }
    )
    arm = (
        {"disposition": "ABSTAIN", "error_code": None, "pairs": None}
        if executable
        else {
            "disposition": "ERROR",
            "error_code": "SLOT_BINDER_TYPED_FAILURE",
            "pairs": None,
        }
    )
    return {
        "diagnostics": {
            variant: {"static": True}
            for variant in qualification.VARIANT_NAMES
        },
        "execution": execution,
        "item_token": item_token,
        "proposal_pools": {
            variant: {"semantic_kbest": [], "structure_kbest": []}
            for variant in qualification.VARIANT_NAMES
        },
        "variants": {
            variant: {
                "arms": {
                    arm_id: copy.deepcopy(arm)
                    for arm_id in qualification.ARM_IDS
                }
            }
            for variant in qualification.VARIANT_NAMES
        },
    }


class _FakePopenFactory:
    def __init__(self, config: qualification.QualificationConfig) -> None:
        self.config = config
        self.launches: list[dict[str, Any]] = []

    @staticmethod
    def _argument(argv: list[str], name: str) -> str:
        return argv[argv.index(name) + 1]

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        shard = int(self._argument(argv, "--shard-index"))
        output = Path(self._argument(argv, "--output-root"))
        action_path = Path(self._argument(argv, "--action-pack"))
        sandbox_path = Path(self._argument(argv, "--sandbox-receipt"))
        implementation_hash = self._argument(
            argv, "--expected-implementation-closure-sha256"
        )
        sandbox_hash = self._argument(
            argv, "--expected-sandbox-receipt-sha256"
        )
        sandbox = json.loads(sandbox_path.read_text())
        runtime = _runtime_receipt(
            config=self.config,
            shard=shard,
            implementation_hash=implementation_hash,
            sandbox=sandbox,
            sandbox_file_hash=sandbox_hash,
        )
        runtime_hash = _publish(
            output / f"shard{shard}.runtime.safe.json", runtime
        )
        action_pack = json.loads(action_path.read_text())
        sentinel = {
            "expected_action_commitment_sha256": action_pack[
                "action_commitment_sha256"
            ],
            "expected_action_file_sha256": hashlib.sha256(
                action_path.read_bytes()
            ).hexdigest(),
            "expected_execution_freeze_sha256": (
                self.config.execution_freeze_sha256
            ),
            "runtime_receipt_sha256": runtime_hash,
            "shard_count": 2,
            "shard_index": shard,
            "study_id": self.config.study_id,
            "version": worker.VERSION,
        }
        _publish(output / f"shard{shard}.attempt.sentinel", sentinel)
        launch = {
            "action_pack": action_pack,
            "argv": argv,
            "kwargs": kwargs,
            "output": output,
            "runtime": runtime,
            "runtime_hash": runtime_hash,
            "shard": shard,
        }
        self.launches.append(launch)
        factory = self

        class _Process:
            completed = False

            def poll(self) -> int | None:
                return 0 if self.completed else None

            def wait(self) -> int:
                assert len(factory.launches) == 2
                release_path = Path(
                    factory._argument(argv, "--action-release")
                )
                assert release_path.exists()
                release = json.loads(release_path.read_text())
                selected = action_pack["items"][shard::2]
                record_rows: list[dict[str, Any]] = []
                resources: dict[str, int] = {}
                document_calls = typed_failures = 0
                errors: dict[str, int] = {}
                for local, item in enumerate(selected):
                    global_ordinal = shard + local * 2
                    executable = global_ordinal in {0, 1}
                    evidence = (
                        _complete_evidence()
                        if executable
                        else _premodel_evidence()
                    )
                    prediction = _prediction(
                        item["item_token"], executable=executable
                    )
                    body = {
                        "evidence": evidence,
                        "item_token": item["item_token"],
                        "ordinal_within_shard": local,
                        "prediction": prediction,
                    }
                    record_rows.append(
                        {**body, "self_sha256": _hash(body)}
                    )
                    document_calls += prediction["execution"][
                        "document_call_count"
                    ]
                    typed_failures += int(not executable)
                    if not executable:
                        errors["SLOT_BINDER_TYPED_FAILURE"] = (
                            errors.get("SLOT_BINDER_TYPED_FAILURE", 0) + 1
                        )
                    counts = worker._mechanism_resource_counts(  # noqa: SLF001
                        evidence
                    )
                    for key, value in counts.items():
                        resources[key] = resources.get(key, 0) + value
                records_raw = b"".join(
                    _canonical(row) + b"\n" for row in record_rows
                )
                records_hash = _publish(
                    output / f"shard{shard}.records.private.jsonl",
                    records_raw,
                )
                body = {
                    "action_commitment_sha256": action_pack[
                        "action_commitment_sha256"
                    ],
                    "action_pack_file_receipt": {
                        "sha256": hashlib.sha256(
                            action_path.read_bytes()
                        ).hexdigest()
                    },
                    "action_release_file_receipt": {
                        "sha256": hashlib.sha256(
                            release_path.read_bytes()
                        ).hexdigest()
                    },
                    "action_release_self_sha256": release["self_sha256"],
                    "arm_ids": list(qualification.ARM_IDS),
                    "document_call_count": document_calls,
                    "encoder_binding_sha256": "d" * 64,
                    "external_network_call_count": 0,
                    "formal_label_pack_access_count": 0,
                    "formal_scorer_access_count": 0,
                    "item_count": len(record_rows),
                    "mechanism_resource_totals": resources,
                    "output_root_receipt": {"filesystem_type": "ext4"},
                    "private_records_file_sha256": records_hash,
                    "private_records_file_size_bytes": len(records_raw),
                    "runtime_receipt_file_sha256": runtime_hash,
                    "runtime_receipt_self_sha256": runtime["self_sha256"],
                    "schema": worker.SHARD_TERMINAL_SCHEMA,
                    "shard_count": 2,
                    "shard_index": shard,
                    "status": "complete",
                    "structural_error_code_counts": errors,
                    "structural_typed_failure_count": typed_failures,
                    "study_id": factory.config.study_id,
                    "variant_names": list(qualification.VARIANT_NAMES),
                    "version": worker.VERSION,
                }
                terminal = {**body, "self_sha256": _hash(body)}
                _publish(
                    output / f"shard{shard}.terminal.safe.json",
                    terminal,
                )
                self.completed = True
                return 0

        return _Process()


def test_public_synthetic_pack_is_deterministic_valid_and_source_free() -> None:
    first = qualification.build_public_synthetic_action_pack_v1(_STUDY_ID)
    second = qualification.build_public_synthetic_action_pack_v1(_STUDY_ID)
    assert first == second
    worker.source.validate_scar_cssm_action_pack_v1(  # noqa: SLF001
        first, study_id=_STUDY_ID
    )
    assert len(first["items"]) == qualification.SYNTHETIC_ITEM_COUNT
    collision_count = 0
    executable_by_shard = {0: 0, 1: 0}
    for ordinal, item in enumerate(first["items"]):
        surfaces = [
            row["surface"]
            for row in item["variants"]["base"]["left"]["slots"]
        ]
        normalized = [
            unicodedata.normalize("NFKC", value).casefold()
            for value in surfaces
        ]
        collision = len(set(normalized)) != len(normalized)
        collision_count += collision
        executable_by_shard[ordinal % 2] += int(not collision)
    assert collision_count == qualification.EXPECTED_PREMODEL_FAILURE_COUNT
    assert executable_by_shard == {0: 1, 1: 1}
    encoded = json.dumps(first, ensure_ascii=False)
    for forbidden in ('"gold_pairs"', '"label_pack"', '"secret"'):
        assert forbidden not in encoded


def test_worker_cli_is_exact_and_has_no_formal_late_capability(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    paths = qualification.QualificationPaths.for_root(config.root)
    argv = qualification._worker_argv(  # noqa: SLF001
        config,
        paths,
        shard=0,
        action_file_sha256="a" * 64,
        action_commitment_sha256="b" * 64,
        implementation_closure_sha256="c" * 64,
        sandbox_file_sha256="d" * 64,
    )
    expected_options = {
        action.option_strings[0]
        for action in worker._parser()._actions  # noqa: SLF001
        if action.required and action.option_strings
    }
    assert {value for value in argv if value.startswith("--")} == expected_options
    assert argv[1:3] == [
        "-m",
        "replication_runtime.gscl_scar_cssm_v1.worker",
    ]
    assert "--forbidden-label-probe" in argv
    for forbidden in ("--label-pack", "--secret", "--source", "--scorer"):
        assert forbidden not in argv


def test_runtime_receipt_requires_landlock_network_and_exact_gpu(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    sandbox = qualification._sandbox_receipt(config.study_id)  # noqa: SLF001
    value = _runtime_receipt(
        config=config,
        shard=0,
        implementation_hash="a" * 64,
        sandbox=sandbox,
        sandbox_file_hash="b" * 64,
    )
    qualification._validate_runtime_receipt(  # noqa: SLF001
        value,
        config=config,
        shard=0,
        implementation_closure_sha256="a" * 64,
        sandbox_receipt=sandbox,
        sandbox_file_sha256="b" * 64,
    )
    for path, replacement in (
        (("network_negative_canary", "AF_INET", "creation_denied"), False),
        (("forbidden_label_negative_canary", "open_denied"), False),
        (("gpu", "physical_uuid"), _GPU1),
    ):
        tampered = copy.deepcopy(value)
        target = tampered
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement
        with pytest.raises(
            qualification.ScarCssmQualificationError,
            match="QUALIFICATION_RUNTIME_RECEIPT_INVALID",
        ):
            qualification._validate_runtime_receipt(  # noqa: SLF001
                tampered,
                config=config,
                shard=0,
                implementation_closure_sha256="a" * 64,
                sandbox_receipt=sandbox,
                sandbox_file_sha256="b" * 64,
            )


def test_two_worker_release_records_barrier_and_safe_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    popen = _FakePopenFactory(config)
    dependencies = qualification.QualificationDependencies(
        filesystem_type=lambda _path: "ext4",
        resource_probe=lambda _config: qualification.AdmissionDecision(
            status="ADMITTED_SHARED_RESOURCE",
            reason_codes=(),
            host_mem_available_bytes=100_000_000_000,
            selected_gpu_free_mib=(7_000, 7_000),
        ),
        popen_factory=popen,
        monotonic=lambda: 0.0,
        sleep=lambda _seconds: None,
    )
    monkeypatch.setattr(
        worker,
        "_implementation_closure",
        lambda: {"self_sha256": "a" * 64},
    )
    receipt = qualification.run_qualification_once(
        config, dependencies=dependencies
    )
    assert receipt["status"] == "QUALIFIED_SOURCE_FREE_EXACT_RUNTIME"
    assert receipt["effect_study_attempt_count"] == 0
    assert receipt["formal_scar_source_access_count"] == 0
    assert receipt["formal_label_pack_access_count"] == 0
    assert receipt["hmac_secret_access_count"] == 0
    assert receipt["formal_scorer_access_count"] == 0
    assert receipt["gpu_uuid_by_shard"] == {"0": _GPU0, "1": _GPU1}
    assert receipt["records_barrier"]["document_call_count"] == 4
    assert receipt["records_barrier"][
        "structural_typed_failure_count"
    ] == qualification.EXPECTED_PREMODEL_FAILURE_COUNT
    assert [row["shard"] for row in popen.launches] == [0, 1]
    release_path = qualification.QualificationPaths.for_root(
        config.root
    ).action_release
    assert release_path.exists()
    for launch in popen.launches:
        assert launch["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == (
            config.gpu_uuids[launch["shard"]]
        )
        assert launch["kwargs"]["env"]["HF_HUB_OFFLINE"] == "1"
        assert launch["kwargs"]["env"]["CUBLAS_WORKSPACE_CONFIG"] == (
            ":4096:8"
        )
        assert launch["kwargs"]["env"]["PYTHONNOUSERSITE"] == "1"
        assert launch["kwargs"]["env"][
            "GSCL_SUPERVISOR_LANDLOCK_DIRECT_PARENT_AUTHORITY_V1"
        ] == (
            "97ff3a77c33a3113712a4c11a9fd347902a12b45f76935023d2ac66377936c35"
        )
        assert callable(launch["kwargs"]["preexec_fn"])
    safe_encoded = json.dumps(receipt, sort_keys=True)
    for forbidden in (
        "scar-item-v1-",
        "scar-slot-v1-",
        '"background"',
        '"evidence"',
        '"item_token"',
        '"opaque_slot_id"',
        '"prediction"',
        '"surface"',
    ):
        assert forbidden not in safe_encoded


def test_shared_node_deferral_creates_no_qualification_root(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    dependencies = qualification.QualificationDependencies(
        filesystem_type=lambda _path: "ext4",
        resource_probe=lambda _config: qualification.AdmissionDecision(
            status="DEFERRED_SHARED_RESOURCE",
            reason_codes=("SELECTED_GPU_HAS_COMPUTE_PROCESS",),
            selected_gpu_free_mib=(1_000, 1_000),
        ),
        popen_factory=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("worker must not launch")
        ),
        monotonic=lambda: 0.0,
        sleep=lambda _seconds: None,
    )
    receipt = qualification.run_qualification_once(
        config, dependencies=dependencies
    )
    assert receipt["status"] == "DEFERRED_SHARED_RESOURCE"
    assert receipt["effect_study_attempt_count"] == 0
    assert not config.root.exists()


def test_module_has_no_source_label_secret_or_scorer_capability() -> None:
    source_text = inspect.getsource(qualification)
    for forbidden in (
        "gscl_scar_cssm_score_v1",
        "score_scar_cssm_predictions_v1",
        "label_pack_path",
        "secret_path",
        "validate_scar_cssm_pack_binding_v1",
    ):
        assert forbidden not in source_text


def test_python_runtime_roots_include_declared_venv_base(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
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

    assert qualification._python_runtime_read_roots(  # noqa: SLF001
        executable
    ) == (environment, runtime / "python310")


def test_child_landlock_grants_only_process_local_cuda_thread_metadata_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        qualification,
        "_landlock_paths",
        lambda _config, _paths, _shard: ((), (), ()),
    )
    monkeypatch.setattr(
        qualification,
        "_apply_landlock",
        lambda **kwargs: captured.update(kwargs),
    )

    config = SimpleNamespace(
        qwen_manifest_path=Path("/assets/qwen.json"),
        minilm_manifest_path=Path("/assets/minilm.json"),
    )
    paths = SimpleNamespace(
        action_pack=Path("/inputs/action.json"),
        sandbox_receipt=Path("/inputs/sandbox.json"),
        action_release=Path("/release/action.json"),
    )
    qualification._apply_child_landlock(config, paths, 0)

    assert captured == {
        "read_paths": (),
        "write_paths": (),
        "device_paths": (),
        "read_directory_paths": (
            Path("/assets"),
            Path("/inputs"),
            Path("/release"),
        ),
        "write_file_paths": (Path("/proc/self/task"),),
    }
    assert qualification._landlock_write_file_rights(2) == (  # noqa: SLF001
        qualification._LL_WRITE_FILE  # noqa: SLF001
    )
    assert qualification._landlock_write_file_rights(3) == (  # noqa: SLF001
        qualification._LL_WRITE_FILE  # noqa: SLF001
        | qualification._LL_TRUNCATE  # noqa: SLF001
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Landlock is Linux-only")
def test_landlock_allows_future_thread_comm_write_but_denies_sibling_file(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    forbidden = tmp_path / "forbidden"
    allowed.write_bytes(b"allowed")
    forbidden.write_bytes(b"forbidden")
    allowed.chmod(0o600)
    forbidden.chmod(0o600)
    project_root = Path(qualification.__file__).parents[2]
    script = r"""
import os
from pathlib import Path
import sys
import threading
from replication_runtime.gscl_scar_cssm_v1 import qualification
from replication_runtime.gscl_narrative_extractor_v1 import contract

allowed = Path(sys.argv[1])
forbidden = Path(sys.argv[2])
os.environ[contract.SUPERVISOR_LANDLOCK_DIRECT_PARENT_ENVIRONMENT_KEY] = (
    contract.SUPERVISOR_LANDLOCK_DIRECT_PARENT_AUTHORITY
)
qualification._apply_landlock(
    read_paths=(Path("/proc"), allowed),
    write_paths=(),
    device_paths=(),
    read_directory_paths=(allowed.parent,),
    write_file_paths=(Path("/proc/self/task"),),
)
ready = threading.Event()
release = threading.Event()
native_id = []
def wait_for_comm_write():
    native_id.append(threading.get_native_id())
    ready.set()
    release.wait(5)
thread = threading.Thread(target=wait_for_comm_write)
thread.start()
if not ready.wait(5):
    raise RuntimeError("thread did not start")
descriptor = os.open(
    f"/proc/self/task/{native_id[0]}/comm",
    os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
)
try:
    os.write(descriptor, b"gscl-cuda-test\n")
finally:
    os.close(descriptor)
release.set()
thread.join(5)
secure = contract.secure_read_file(allowed, maximum=1024)
if thread.is_alive() or secure.raw != b"allowed":
    raise RuntimeError("allowed operations failed")
try:
    forbidden.read_bytes()
except PermissionError:
    denied = True
else:
    denied = False
os.write(1, b"comm_written_and_denied" if denied else b"sandbox_failed")
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
    assert completed.stdout == b"comm_written_and_denied"
