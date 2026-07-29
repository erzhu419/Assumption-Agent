from __future__ import annotations

import hashlib
import inspect
import json
import marshal
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Iterator, Mapping

import pytest

from replication_runtime.wikisql_uao_formal_v1 import runner as formal_v1
from replication_runtime.wikisql_uao_formal_v5 import prepare
from replication_runtime.wikisql_uao_formal_v5 import runner as subject
from replication_runtime.wikisql_uao_formal_v5 import source_custody
from replication_runtime.wikisql_uao_runtime_qualification import (
    alias_runtime,
    contract as qualification_contract,
    resource_admission,
)


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """Use a native Linux filesystem so receipt mode checks are meaningful."""

    with tempfile.TemporaryDirectory(
        prefix="wikisql-uao-formal-v5-", dir="/tmp"
    ) as value:
        yield Path(value)


class _PathBinding:
    def __init__(self, path: Path) -> None:
        self.path = path


class _ServiceConfig:
    def file(self, name: str) -> _PathBinding:
        if name != "python_executable":
            raise AssertionError(f"unexpected file binding: {name}")
        return _PathBinding(
            Path(
                "/home/erzhu419/p19_runtime_assets_20260723/"
                "typed_venv/bin/python"
            )
        )

    def tree(self, name: str) -> _PathBinding:
        paths = {
            "code_tree": subject.FORMAL_ROOT / "reconstruction_v2",
            "python_dependency_tree": Path(
                "/home/erzhu419/p19_runtime_assets_20260723/"
                "typed_venv/lib/python3.10/site-packages"
            ),
            "babel_dependency_tree": (
                subject.FORMAL_ROOT
                / "runtime_assets/babel_2_10_3_clean"
            ),
        }
        return _PathBinding(paths[name])


def _service_bytes() -> bytes:
    path = (
        Path(__file__).parents[1]
        / "manifests/wikisql-uao-p4-formal-v5.service"
    )
    return path.read_bytes()


def test_service_freezes_shared_caps_accounting_threads_and_module() -> None:
    raw = _service_bytes()
    text = raw.decode("utf-8")

    for line in (
        "CPUQuota=400%",
        "CPUWeight=25",
        "IOWeight=25",
        "IOSchedulingClass=idle",
        "Nice=10",
        "MemoryHigh=25769803776",
        "MemoryMax=34359738368",
        "MemorySwapMax=0",
        "TasksMax=96",
        "CPUAccounting=yes",
        "IOAccounting=yes",
        "MemoryAccounting=yes",
        "TasksAccounting=yes",
        "SuccessExitStatus=75",
        "TimeoutStartSec=6h",
        "MKL_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "VECLIB_MAXIMUM_THREADS=1",
        "-m replication_runtime.wikisql_uao_formal_v5.runner",
    ):
        assert line in text
    subject._verify_service_profile(raw, _ServiceConfig())


@pytest.mark.parametrize(
    "old,new",
    (
        (
            b"replication_runtime.wikisql_uao_formal_v5.runner",
            b"replication_runtime.wikisql_uao_formal_v4.runner",
        ),
        (b"CPUWeight=25\n", b"CPUWeight=26\n"),
        (b"MemoryMax=34359738368\n", b"MemoryMax=34359738369\n"),
        (b"VECLIB_MAXIMUM_THREADS=1 ", b"VECLIB_MAXIMUM_THREADS=2 "),
    ),
)
def test_service_rejects_retired_module_or_control_drift(
    old: bytes,
    new: bytes,
) -> None:
    raw = _service_bytes()
    assert old in raw

    with pytest.raises(
        subject.WikiSQLUAOFormalError,
        match="shared-node formal v5 service profile drifted",
    ):
        subject._verify_service_profile(
            raw.replace(old, new, 1), _ServiceConfig()
        )


class _ActionConfig:
    def __init__(self, llm: Path, encoder: Path) -> None:
        self._trees = {
            "hippo_llm_model_tree": _PathBinding(llm),
            "encoder_model_tree": _PathBinding(encoder),
        }

    def tree(self, name: str) -> _PathBinding:
        return self._trees[name]


def _command(
    name: str,
    root: Path,
    *,
    cuda: str,
    argv: tuple[str, ...] = ("runner",),
) -> object:
    return subject._base.CommandSpec(
        name=name,
        argv=argv,
        cwd=root,
        environment={"CUDA_VISIBLE_DEVICES": cuda},
        read_paths=(root,),
        write_paths=(root,),
    )


def _model_tree(path: Path, payload: bytes) -> None:
    path.mkdir(mode=0o700)
    model = path / "weights.bin"
    model.write_bytes(payload)
    model.chmod(0o600)


def test_action_commands_use_verified_short_aliases_and_full_gpu_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = tmp_path / "llm-model"
    encoder = tmp_path / "encoder-model"
    _model_tree(llm, b"llm")
    _model_tree(encoder, b"encoder")
    paths = subject._base.FormalPaths.for_root(tmp_path / "formal")
    paths.hippo_root.mkdir(mode=0o700, parents=True)
    commands = {
        "Agent": _command("Agent", tmp_path / "agent", cuda="1"),
        "RAW": _command("RAW", tmp_path / "raw", cuda=""),
        "HippoRAG": _command(
            "HippoRAG",
            paths.hippo_root,
            cuda="0",
            argv=(
                "official-runner",
                "--llm-model",
                str(llm),
                "--embedding-model",
                str(encoder),
            ),
        ),
    }
    monkeypatch.setattr(
        subject, "_original_action_commands", lambda *_: commands
    )
    devices = (Path("/dev/nvidia0"), Path("/dev/nvidia1"))
    monkeypatch.setattr(subject, "_all_gpu_device_paths", lambda: devices)

    observed = subject._action_commands(
        _ActionConfig(llm, encoder), paths, object()
    )

    hippo = observed["HippoRAG"]
    assert hippo.argv[hippo.argv.index("--llm-model") + 1] == "smollm2"
    assert (
        hippo.argv[hippo.argv.index("--embedding-model") + 1]
        == "minilm"
    )
    alias_root = paths.hippo_root / alias_runtime.ALIAS_DIRECTORY
    assert hippo.cwd == alias_root
    assert os.readlink(alias_root / "smollm2") == str(llm)
    assert os.readlink(alias_root / "minilm") == str(encoder)
    assert os.path.samefile(alias_root / "smollm2", llm)
    assert os.path.samefile(alias_root / "minilm", encoder)
    assert observed["Agent"].device_paths == devices
    assert observed["HippoRAG"].device_paths == devices
    assert Path("/proc/self/task") in observed["Agent"].write_paths
    assert Path("/proc/self/task") in observed["HippoRAG"].write_paths
    assert observed["RAW"] is commands["RAW"]

    receipt_path = paths.hippo_root / "model_alias.safe.json"
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    body = {
        key: value
        for key, value in receipt.items()
        if key != "self_sha256"
    }
    assert receipt["self_sha256"] == subject._base.semantic_sha256(body)
    assert receipt["model_content_changed"] is False
    assert receipt["aliases"]["smollm2"]["tree_identity"] == list(
        subject._base.tree_identity(llm)
    )
    assert receipt["aliases"]["minilm"]["tree_identity"] == list(
        subject._base.tree_identity(encoder)
    )


def test_action_commands_reject_model_tree_drift_during_alias_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = tmp_path / "llm-model"
    encoder = tmp_path / "encoder-model"
    _model_tree(llm, b"llm")
    _model_tree(encoder, b"encoder")
    paths = subject._base.FormalPaths.for_root(tmp_path / "formal")
    paths.hippo_root.mkdir(mode=0o700, parents=True)
    commands = {
        "Agent": _command("Agent", tmp_path / "agent", cuda="1"),
        "RAW": _command("RAW", tmp_path / "raw", cuda=""),
        "HippoRAG": _command(
            "HippoRAG",
            paths.hippo_root,
            cuda="0",
            argv=(
                "official-runner",
                "--llm-model",
                str(llm),
                "--embedding-model",
                str(encoder),
            ),
        ),
    }
    monkeypatch.setattr(
        subject, "_original_action_commands", lambda *_: commands
    )
    monkeypatch.setattr(subject, "_all_gpu_device_paths", lambda: ())
    calls = 0

    def drifting_identity(_path: Path) -> tuple[str, int]:
        nonlocal calls
        calls += 1
        return ("identity", calls)

    monkeypatch.setattr(subject._base, "tree_identity", drifting_identity)

    with pytest.raises(
        alias_runtime.WikiSQLUAOAliasRuntimeError,
        match="model tree identity changed",
    ):
        subject._action_commands(
            _ActionConfig(llm, encoder), paths, object()
        )


class _AdmissionConfig:
    gpu_uuids = {
        "0": "GPU-00000000-0000-0000-0000-000000000000",
        "1": "GPU-11111111-1111-1111-1111-111111111111",
    }

    def __init__(self) -> None:
        self.source_binding_accesses = 0

    def file(self, name: str) -> _PathBinding:
        if name == "nvidia_smi_executable":
            return _PathBinding(Path("/usr/bin/nvidia-smi"))
        self.source_binding_accesses += 1
        raise AssertionError(f"pre-attempt source access: {name}")


class _FakeLock:
    def __init__(
        self,
        events: list[str],
        *,
        acquired: bool = True,
        error: Exception | None = None,
    ) -> None:
        self.events = events
        self.acquired = acquired
        self.error = error
        self.held = False

    def acquire_nonblocking(self) -> bool:
        self.events.append("acquire")
        if self.error is not None:
            raise self.error
        self.held = self.acquired
        return self.acquired

    def release(self) -> None:
        self.events.append("release")
        self.held = False


def _patch_formal_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, object]:
    root = tmp_path / "formal-v5"
    (root / "control").mkdir(mode=0o700, parents=True)
    monkeypatch.setattr(subject, "FORMAL_ROOT", root)
    monkeypatch.setattr(
        subject, "ADMISSION_PATH", root / "control/admission.safe.json"
    )
    monkeypatch.setattr(
        subject,
        "ADMISSION_FAILURE_PATH",
        root / "control/admission_failure.safe.json",
    )
    monkeypatch.setattr(
        subject, "DEFERRAL_ROOT", root / "control/resource_deferrals"
    )
    return root, subject._base.FormalPaths.for_root(root)


def _decision(status: str, reason: str) -> object:
    return resource_admission.AdmissionDecision(
        status=status,
        reason_codes=(reason,),
    )


def _patch_admission_shell(
    monkeypatch: pytest.MonkeyPatch,
    config: _AdmissionConfig,
    lock: _FakeLock,
) -> None:
    monkeypatch.setattr(subject, "load_config", lambda _path: config)
    monkeypatch.setattr(
        subject,
        "_service_probe",
        lambda _config: SimpleNamespace(invocation_id="a" * 32),
    )
    monkeypatch.setattr(subject, "_resource_policy", lambda: object())
    monkeypatch.setattr(
        resource_admission, "QualificationFlock", lambda _path: lock
    )


def test_shared_resource_deferral_precedes_attempt_and_source_and_exits_75(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, paths = _patch_formal_root(tmp_path, monkeypatch)
    config = _AdmissionConfig()
    events: list[str] = []
    lock = _FakeLock(events)
    _patch_admission_shell(monkeypatch, config, lock)
    monkeypatch.setattr(
        resource_admission,
        "sample_and_decide",
        lambda *_: _decision(
            resource_admission.DEFERRED_SHARED_RESOURCE,
            "GPU_CAPACITY_BUSY",
        ),
    )
    monkeypatch.setattr(
        subject._base,
        "_run_with_dependencies",
        lambda *_: pytest.fail("effect runner opened after deferral"),
    )

    terminal = subject.run_formal_production(tmp_path / "config.json")

    assert terminal["status"] == "DEFERRED_SHARED_RESOURCE"
    assert terminal["effect_study_attempt_count"] == 0
    assert terminal["formal_source_access_count"] == 0
    assert terminal["API_or_online_evaluation_count"] == 0
    assert config.source_binding_accesses == 0
    assert not paths.attempt.exists()
    assert not paths.terminal.exists()
    assert not subject.ADMISSION_PATH.exists()
    assert not subject.ADMISSION_FAILURE_PATH.exists()
    deferrals = list(subject.DEFERRAL_ROOT.glob("*.safe.json"))
    assert len(deferrals) == 1
    assert events == ["acquire", "release"]

    monkeypatch.setattr(
        subject, "run_formal_production", lambda _path: terminal
    )
    assert (
        subject.main(["--config", str(tmp_path / "config.json")])
        == resource_admission.EX_TEMPFAIL
    )


def test_admission_receipt_is_durable_before_base_run_and_lock_spans_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, paths = _patch_formal_root(tmp_path, monkeypatch)
    config = _AdmissionConfig()
    events: list[str] = []
    lock = _FakeLock(events)
    _patch_admission_shell(monkeypatch, config, lock)

    def sample(*_args: object) -> object:
        events.append("sample")
        return _decision(resource_admission.ADMITTED, "WITHIN_LIMITS")

    monkeypatch.setattr(resource_admission, "sample_and_decide", sample)

    def base_run(_path: Path, dependencies: object) -> Mapping[str, object]:
        events.append("base")
        assert dependencies is subject._base.PRODUCTION_DEPENDENCIES
        assert lock.held is True
        assert subject.ADMISSION_PATH.exists()
        receipt = json.loads(
            subject.ADMISSION_PATH.read_text(encoding="ascii")
        )
        assert receipt["status"] == "ADMITTED_SHARED_RESOURCE"
        assert receipt["effect_study_attempt_count"] == 0
        assert receipt["formal_source_access_count"] == 0
        assert receipt["resource_policy_sha256"] == (
            subject.RESOURCE_POLICY_SHA256
        )
        assert not paths.attempt.exists()
        return {"status": "completed_protocol_valid"}

    monkeypatch.setattr(subject._base, "_run_with_dependencies", base_run)

    terminal = subject.run_formal_production(tmp_path / "config.json")

    assert terminal == {"status": "completed_protocol_valid"}
    assert events == ["acquire", "sample", "base", "release"]
    assert lock.held is False
    assert config.source_binding_accesses == 0


def test_failed_pre_attempt_infrastructure_writes_only_safe_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, paths = _patch_formal_root(tmp_path, monkeypatch)
    config = _AdmissionConfig()
    events: list[str] = []
    lock = _FakeLock(events)
    _patch_admission_shell(monkeypatch, config, lock)
    monkeypatch.setattr(
        resource_admission,
        "sample_and_decide",
        lambda *_: _decision(
            resource_admission.FAILED_INFRASTRUCTURE,
            "TELEMETRY_UNTRUSTWORTHY",
        ),
    )

    terminal = subject.run_formal_production(tmp_path / "config.json")

    assert terminal["status"] == "FAILED_INFRASTRUCTURE_PRE_ATTEMPT"
    assert terminal["effect_study_attempt_count"] == 0
    assert terminal["formal_source_access_count"] == 0
    assert terminal["API_or_online_evaluation_count"] == 0
    assert subject.ADMISSION_FAILURE_PATH.exists()
    assert not subject.ADMISSION_PATH.exists()
    assert not paths.attempt.exists()
    assert not paths.terminal.exists()
    assert config.source_binding_accesses == 0
    assert events == ["acquire", "release"]


@pytest.mark.parametrize(
    ("status", "expected"),
    (
        ("completed_protocol_valid", 0),
        ("DEFERRED_SHARED_RESOURCE", resource_admission.EX_TEMPFAIL),
        (
            "FAILED_INFRASTRUCTURE_PRE_ATTEMPT",
            resource_admission.EX_SOFTWARE,
        ),
        ("formal_failed_no_retry_efficacy_unknown", 1),
    ),
)
def test_main_exit_statuses(
    status: str,
    expected: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subject,
        "run_formal_production",
        lambda _path: {"status": status},
    )
    assert (
        subject.main(["--config", str(tmp_path / "config.json")])
        == expected
    )


def test_local_source_custody_hashes_blob_without_archive_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"opaque archive bytes; never interpreted"
    source = tmp_path / "wikisql.tar.bz2"
    source.write_bytes(payload)
    source.chmod(0o600)
    git_blob_sha1 = hashlib.sha1(
        f"blob {len(payload)}\0".encode("ascii") + payload
    ).hexdigest()
    monkeypatch.setattr(
        source_custody, "EXPECTED_SOURCE_BYTES", len(payload)
    )
    monkeypatch.setattr(
        source_custody, "EXPECTED_SOURCE_GIT_BLOB_SHA1", git_blob_sha1
    )

    receipt = source_custody.create_receipt(source)

    assert receipt["archive_sha256"] == hashlib.sha256(payload).hexdigest()
    assert receipt["archive_size_bytes"] == len(payload)
    assert receipt["archive_git_blob_sha1"] == git_blob_sha1
    assert receipt["formal_source_access_count"] == 0
    assert receipt["formal_source_member_open_count"] == 0
    assert receipt["local_acquisition_archive_read_count"] == 1
    custody_path = tmp_path / "source_custody.safe.json"
    subject._base._write_once(custody_path, receipt, mode=0o600)
    assert source_custody.load_receipt(custody_path) == receipt

    implementation = inspect.getsource(source_custody._archive_identity)
    assert "tarfile" not in implementation
    assert "extract" not in implementation
    assert "getmembers" not in implementation


def test_remote_prepare_checks_source_metadata_without_reading_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"opaque"
    source = tmp_path / "wikisql.tar.bz2"
    source.write_bytes(payload)
    monkeypatch.setattr(
        source_custody, "EXPECTED_SOURCE_BYTES", len(payload)
    )
    source.chmod(0o600)

    assert prepare._opaque_source_metadata(source) == len(payload)
    implementation = inspect.getsource(prepare._opaque_source_metadata)
    assert ".open(" not in implementation
    assert "read_bytes" not in implementation
    assert "tarfile" not in implementation

    source.chmod(0o644)
    with pytest.raises(
        prepare.WikiSQLUAOFormalV5PrepareError, match="mode drifted"
    ):
        prepare._opaque_source_metadata(source)

    source.chmod(0o600)
    monkeypatch.setattr(
        source_custody, "EXPECTED_SOURCE_BYTES", len(payload) + 1
    )
    with pytest.raises(
        prepare.WikiSQLUAOFormalV5PrepareError,
        match="archive size drifted",
    ):
        prepare._opaque_source_metadata(source)


def _write_synthetic_file(
    path: Path,
    payload: bytes,
    *,
    mode: int = 0o600,
) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(mode)


def _seed_synthetic_tree(
    root: Path,
    relative: str,
    payload: bytes,
) -> Path:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    child = root / relative
    _write_synthetic_file(child, payload)
    return child


def _synthetic_file_binding(path: Path) -> dict[str, object]:
    digest, size = subject._base._file_sha256(path)
    return {
        "mode_octal": f"{path.stat().st_mode & 0o7777:04o}",
        "path": str(path),
        "sha256": digest,
        "size_bytes": size,
    }


def _synthetic_tree_binding(path: Path) -> dict[str, object]:
    digest, count, size = subject._base.tree_identity(path)
    return {
        "file_count": count,
        "path": str(path),
        "sha256": digest,
        "total_bytes": size,
    }


def _synthetic_build_config_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    qualification_root = tmp_path / "qualification"
    formal_root = tmp_path / "formal-v5"
    qualified_python = (
        qualification_root / "runtime_assets/python310_clean"
    )
    qualified_babel = (
        qualification_root / "runtime_assets/babel_2_10_3_clean"
    )
    qualified_hippo = (
        qualification_root / "runtime_assets/hipporag_clean"
    )
    qualified_base = (
        qualification_root / "runtime_assets/base_import_clean"
    )
    formal_python = formal_root / "runtime_assets/python310_clean"
    formal_babel = (
        formal_root / "runtime_assets/babel_2_10_3_clean"
    )

    monkeypatch.setattr(
        qualification_contract, "QUALIFICATION_ROOT", qualification_root
    )
    monkeypatch.setattr(
        qualification_contract, "PYTHONHOME_ROOT", qualified_python
    )
    monkeypatch.setattr(
        qualification_contract, "BABEL_ROOT", qualified_babel
    )
    monkeypatch.setattr(
        qualification_contract, "OFFICIAL_HIPPORAG_ROOT", qualified_hippo
    )
    monkeypatch.setattr(
        qualification_contract, "OFFICIAL_BASE_ROOT", qualified_base
    )
    monkeypatch.setattr(subject, "FORMAL_ROOT", formal_root)
    monkeypatch.setattr(subject._base, "FORMAL_ROOT", formal_root)
    monkeypatch.setattr(subject, "PYTHONHOME_ROOT", formal_python)
    monkeypatch.setattr(
        subject,
        "ADMISSION_PATH",
        formal_root / "control/resource_admission.safe.json",
    )
    monkeypatch.setattr(
        subject,
        "ADMISSION_FAILURE_PATH",
        formal_root / "control/resource_admission_failure.safe.json",
    )
    monkeypatch.setattr(
        subject,
        "DEFERRAL_ROOT",
        formal_root / "control/resource_deferrals",
    )

    for directory in (
        formal_root,
        formal_root / "control",
        formal_root / "control/home",
        formal_root / "control/tmp",
        formal_root / "work",
        formal_root / "source",
        formal_root / "runtime_assets",
    ):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.chmod(0o700)

    python_relative = "lib/python3.10/runtime.bin"
    babel_relative = "babel/core.py"
    _seed_synthetic_tree(
        qualified_python, python_relative, b"qualified-python-runtime"
    )
    formal_python_payload = _seed_synthetic_tree(
        formal_python, python_relative, b"qualified-python-runtime"
    )
    _seed_synthetic_tree(
        qualified_babel, babel_relative, b"qualified-babel-runtime"
    )
    formal_babel_payload = _seed_synthetic_tree(
        formal_babel, babel_relative, b"qualified-babel-runtime"
    )
    _seed_synthetic_tree(
        qualified_hippo, "hipporag/__init__.py", b"hipporag"
    )
    _seed_synthetic_tree(
        qualified_base, "base/__init__.py", b"base-import"
    )

    qualification_code = qualification_root / "reconstruction_v2"
    qualification_service = (
        qualification_code / qualification_contract.SERVICE_RELATIVE_PATH
    )
    _write_synthetic_file(
        qualification_service, b"[Service]\nType=oneshot\n"
    )
    tree_paths: dict[str, Path] = {
        "code_tree": qualification_code,
        "python_runtime_tree": qualified_python,
        "official_python_runtime_tree": qualified_python,
        "babel_dependency_tree": qualified_babel,
        "official_hipporag_tree": qualified_hippo,
        "official_base_dependency_tree": qualified_base,
    }
    for name in sorted(
        qualification_contract.REQUIRED_TREES - set(tree_paths)
    ):
        tree = qualification_root / "assets" / name
        _seed_synthetic_tree(
            tree, "payload.bin", f"{name}-payload".encode("ascii")
        )
        tree_paths[name] = tree

    executable_root = qualification_root / "bin"
    file_paths: dict[str, Path] = {
        "service_unit": qualification_service,
    }
    for name in sorted(
        qualification_contract.REQUIRED_FILES - {"service_unit"}
    ):
        executable = executable_root / name
        _write_synthetic_file(
            executable, f"{name}-bytes".encode("ascii"), mode=0o700
        )
        file_paths[name] = executable

    qualification_body: dict[str, object] = {
        "bindings": {
            "files": {
                name: _synthetic_file_binding(file_paths[name])
                for name in sorted(file_paths)
            },
            "trees": {
                name: _synthetic_tree_binding(tree_paths[name])
                for name in sorted(tree_paths)
            },
        },
        "capability_boundary": dict(
            qualification_contract.CAPABILITY_BOUNDARY
        ),
        "encoder_model_semantic_sha256": "e" * 64,
        "expected_babel_version": (
            qualification_contract.EXPECTED_BABEL_VERSION
        ),
        "gpu_uuids": {
            "0": "GPU-00000000-0000-0000-0000-000000000000",
            "1": "GPU-11111111-1111-1111-1111-111111111111",
        },
        "pythonpath_order": dict(
            qualification_contract.PYTHONPATH_ORDER
        ),
        "qualification_id": qualification_contract.QUALIFICATION_ID,
        "qualification_root": str(qualification_root),
        "resource_policy": {"schema": "synthetic-source-free-policy"},
        "schema": qualification_contract.CONFIG_SCHEMA,
        "unit_name": qualification_contract.UNIT_NAME,
    }
    qualification_payload = qualification_contract.addressed(
        qualification_body
    )
    qualification_config = (
        qualification_root / qualification_contract.CONFIG_RELATIVE_PATH
    )
    _write_synthetic_file(
        qualification_config,
        qualification_contract.canonical_json_bytes(
            qualification_payload
        ),
    )

    formal_code = formal_root / "reconstruction_v2"
    design_path = formal_code / subject._base.DESIGN_RELATIVE_PATH
    design_body = {
        "schema": "wikisql_uao_p4_study_design_v1",
        "study_id": subject.STUDY_ID,
    }
    design_self = subject._base.semantic_sha256(design_body)
    monkeypatch.setattr(
        prepare, "EXPECTED_DESIGN_SELF_SHA256", design_self
    )
    _write_synthetic_file(
        design_path,
        subject._base.canonical_json_bytes(
            {**design_body, "self_sha256": design_self}
        ),
    )
    _write_synthetic_file(
        formal_code / subject.SERVICE_RELATIVE_PATH,
        b"[Service]\nType=oneshot\n",
    )

    committed_payload = b"A" * 37
    remote_payload = b"B" * len(committed_payload)
    source_archive = formal_root / subject._base.SOURCE_RELATIVE_PATH
    _write_synthetic_file(source_archive, remote_payload)
    committed_git_blob = hashlib.sha1(
        (
            f"blob {len(committed_payload)}\0".encode("ascii")
            + committed_payload
        )
    ).hexdigest()
    committed_sha256 = hashlib.sha256(committed_payload).hexdigest()
    monkeypatch.setattr(
        source_custody, "EXPECTED_SOURCE_BYTES", len(committed_payload)
    )
    monkeypatch.setattr(
        source_custody,
        "EXPECTED_SOURCE_GIT_BLOB_SHA1",
        committed_git_blob,
    )
    custody = subject._base._self_hashed(
        {
            "API_or_online_evaluation_count": 0,
            "archive_git_blob_sha1": committed_git_blob,
            "archive_sha256": committed_sha256,
            "archive_size_bytes": len(committed_payload),
            "formal_source_access_count": 0,
            "formal_source_member_open_count": 0,
            "local_acquisition_archive_read_count": 1,
            "official_repository_commit": (
                source_custody.OFFICIAL_REPOSITORY_COMMIT
            ),
            "schema": source_custody.CUSTODY_SCHEMA,
            "source_payload_read_context": (
                "local_acquisition_only_not_formal_runtime"
            ),
            "study_id": subject.STUDY_ID,
        }
    )
    custody_path = formal_root / "control/source_custody.safe.json"
    _write_synthetic_file(
        custody_path, subject._base.canonical_json_bytes(custody)
    )
    return SimpleNamespace(
        qualification_config=qualification_config,
        formal_root=formal_root,
        formal_python_payload=formal_python_payload,
        formal_babel_payload=formal_babel_payload,
        source_archive=source_archive,
        custody_path=custody_path,
        custody_sha256=committed_sha256,
        remote_sha256=hashlib.sha256(remote_payload).hexdigest(),
        carried_file=file_paths["python_executable"],
        carried_tree=tree_paths["encoder_model_tree"],
    )


def _forbid_remote_payload_open(
    monkeypatch: pytest.MonkeyPatch,
    source_archive: Path,
) -> None:
    original_open = Path.open

    def guarded_open(
        path: Path, *args: object, **kwargs: object
    ) -> object:
        if path == source_archive:
            raise AssertionError("remote source payload was opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)


def test_full_synthetic_build_config_uses_custody_and_complete_tree_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _synthetic_build_config_inputs(tmp_path, monkeypatch)
    _forbid_remote_payload_open(monkeypatch, inputs.source_archive)

    config = prepare.build_config(
        inputs.qualification_config,
        inputs.source_archive,
        inputs.custody_path,
    )
    qualified = qualification_contract.load_config(
        inputs.qualification_config
    )
    bindings = config["bindings"]
    assert isinstance(bindings, dict)
    files = bindings["files"]
    trees = bindings["trees"]
    assert isinstance(files, dict)
    assert isinstance(trees, dict)
    assert inputs.custody_sha256 != inputs.remote_sha256
    assert files["source_archive"]["sha256"] == inputs.custody_sha256
    assert files["source_archive"]["size_bytes"] == (
        source_custody.EXPECTED_SOURCE_BYTES
    )
    assert files["python_executable"]["size_bytes"] == (
        inputs.carried_file.stat().st_size
    )
    assert trees["encoder_model_tree"]["total_bytes"] == (
        subject._base.tree_identity(inputs.carried_tree)[2]
    )
    assert (
        trees["python_runtime_tree"]["sha256"],
        trees["python_runtime_tree"]["file_count"],
        trees["python_runtime_tree"]["total_bytes"],
    ) == (
        qualified.tree("python_runtime_tree").sha256,
        qualified.tree("python_runtime_tree").file_count,
        qualified.tree("python_runtime_tree").total_bytes,
    )
    assert (
        trees["babel_dependency_tree"]["sha256"],
        trees["babel_dependency_tree"]["file_count"],
        trees["babel_dependency_tree"]["total_bytes"],
    ) == (
        qualified.tree("babel_dependency_tree").sha256,
        qualified.tree("babel_dependency_tree").file_count,
        qualified.tree("babel_dependency_tree").total_bytes,
    )

    config_path = inputs.formal_root / "control/formal_config.json"
    subject._base._write_once(config_path, config, mode=0o600)
    observed = subject.load_config(config_path)
    assert observed.file("python_executable").size_bytes == (
        inputs.carried_file.stat().st_size
    )
    assert observed.tree("encoder_model_tree").total_bytes == (
        subject._base.tree_identity(inputs.carried_tree)[2]
    )


@pytest.mark.parametrize(
    "payload_attribute",
    ("formal_python_payload", "formal_babel_payload"),
)
def test_full_synthetic_build_config_rejects_python_or_babel_copy_drift(
    payload_attribute: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _synthetic_build_config_inputs(tmp_path, monkeypatch)
    drifted = getattr(inputs, payload_attribute)
    drifted.write_bytes(b"runtime-tree-drift")
    drifted.chmod(0o600)
    _forbid_remote_payload_open(monkeypatch, inputs.source_archive)

    with pytest.raises(
        prepare.WikiSQLUAOFormalV5PrepareError,
        match="formal copied qualification runtime tree drifted",
    ):
        prepare.build_config(
            inputs.qualification_config,
            inputs.source_archive,
            inputs.custody_path,
        )


def _code_sha256(function: object) -> str:
    code = getattr(function, "__code__")
    return hashlib.sha256(marshal.dumps(code)).hexdigest()


def test_v5_preserves_study_and_effect_contract_from_frozen_v1() -> None:
    assert subject.STUDY_ID == formal_v1.STUDY_ID
    assert subject.CONFIG_SCHEMA == formal_v1.CONFIG_SCHEMA
    assert _code_sha256(subject._base._run_with_dependencies) == (
        _code_sha256(formal_v1._run_with_dependencies)
    )
    for field in ("source_compile", "label_projector", "scorer_command"):
        assert _code_sha256(
            getattr(subject._base.PRODUCTION_DEPENDENCIES, field)
        ) == _code_sha256(
            getattr(formal_v1.PRODUCTION_DEPENDENCIES, field)
        )

    implementation = inspect.getsource(
        subject._base._run_with_dependencies
    )
    assert implementation.index(
        'state.stage = "validate_and_durably_seal_common_actions"'
    ) < implementation.index(
        'state.stage = "post_barrier_project_minimal_A_hold_labels"'
    )
    assert implementation.index(
        'state.stage = "post_barrier_project_minimal_A_hold_labels"'
    ) < implementation.index(
        'state.stage = "launch_independent_offline_scorer"'
    )
    assert (
        '"API_or_online_evaluation_count": 0' in implementation
    )


def test_three_actions_all_launch_before_any_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Process:
        def __init__(self, name: str) -> None:
            self.name = name

        def wait(self) -> int:
            events.append(f"wait:{self.name}")
            return 0

    def launch(command: object, *, child_landlock: object) -> Process:
        del child_landlock
        name = getattr(command, "name")
        events.append(f"launch:{name}")
        return Process(name)

    monkeypatch.setattr(subject._base, "_launch_one", launch)
    commands = {
        name: SimpleNamespace(name=name)
        for name in ("Agent", "RAW", "HippoRAG")
    }

    statuses = subject._base._launch_actions_concurrently(
        commands,
        child_landlock=lambda **_kwargs: None,
        on_launch=lambda: events.append("on_launch"),
    )

    assert statuses == {"Agent": 0, "RAW": 0, "HippoRAG": 0}
    assert events == [
        "launch:Agent",
        "on_launch",
        "launch:RAW",
        "on_launch",
        "launch:HippoRAG",
        "on_launch",
        "wait:Agent",
        "wait:RAW",
        "wait:HippoRAG",
    ]
