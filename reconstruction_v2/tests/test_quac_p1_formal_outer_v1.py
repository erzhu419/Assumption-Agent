from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import (
    quac_p1_formal_controller_v1 as controller,
)
from assumption_agent.benchmarks import (
    quac_p1_formal_runner_v1 as core,
)
from assumption_agent.benchmarks import quac_p1_runtime_v1 as runtime
from replication_runtime import quac_p1_source_free_canary_v1 as canary
from replication_runtime.quac_p1_formal_v1 import runner as subject


def _write(path: Path, raw: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)


def _bindings(root: Path) -> runtime.RuntimeBindings:
    python0 = root / "runtime0/bin/python"
    python1 = root / "runtime1/bin/python"
    _write(python0, b"python zero\n", 0o755)
    _write(python1, b"python one\n", 0o755)
    site0 = root / "runtime0/site"
    site1 = root / "runtime1/site"
    overlay1 = root / "runtime1/overlay"
    base1 = root / "runtime1/base"
    _write(site0 / "typed.dist-info/METADATA", b"typed\n")
    _write(site1 / "hippo.dist-info/METADATA", b"hippo\n")
    _write(overlay1 / "overlay.dist-info/METADATA", b"overlay\n")
    _write(base1 / "base.dist-info/METADATA", b"base\n")
    minilm = root / "assets/minilm"
    llm = root / "assets/llm"
    hippo = root / "assets/hipporag"
    _write(minilm / "model.bin", b"minilm")
    _write(llm / "model.bin", b"llm")
    _write(hippo / "hipporag/__init__.py", b"# hippo\n")
    return runtime.RuntimeBindings(
        gpu0_python=runtime.PythonRuntimeBinding.capture(
            executable=python0,
            import_tree=site0,
        ),
        gpu1_python=runtime.PythonRuntimeBinding.capture(
            executable=python1,
            import_tree=site1,
        ),
        gpu1_overlay_import_tree=runtime.FrozenTreeBinding.capture(
            overlay1
        ),
        gpu1_base_import_tree=runtime.FrozenTreeBinding.capture(base1),
        minilm_asset=runtime.FrozenTreeBinding.capture(minilm),
        llm_asset=runtime.FrozenTreeBinding.capture(llm),
        hipporag_source=runtime.FrozenTreeBinding.capture(hippo),
    )


def _config_payload(
    tmp_path: Path,
    bindings: runtime.RuntimeBindings,
) -> tuple[dict[str, object], Path]:
    control = tmp_path / "control"
    config_path = control / subject.FORMAL_CONFIG_FILENAME
    source_root = tmp_path / "source"
    body = {
        "canary_binding": {
            "config_file_sha256": "1" * 64,
            "config_path": str(tmp_path / "canary/config.json"),
            "config_self_sha256": "2" * 64,
            "safe_terminal_file_sha256": "3" * 64,
            "safe_terminal_path": str(
                tmp_path / "canary/terminal.json"
            ),
            "safe_terminal_self_sha256": "4" * 64,
        },
        "config_path": str(config_path),
        "control_root": str(control),
        "core_binding": {
            "file_sha256": "5" * 64,
            "relative_path": subject.FORMAL_CORE_RELATIVE_PATH,
        },
        "design_binding": {
            "file_sha256": "6" * 64,
            "path": str(tmp_path / "design.json"),
            "schema": "quac_p1_effect_execution_design_v1",
            "self_sha256": subject.EXPECTED_DESIGN_SELF_SHA256,
        },
        "execution_freeze_binding": {
            "file_sha256": "7" * 64,
            "path": str(
                control / subject.EXECUTION_FREEZE_FILENAME
            ),
            "schema": subject.EXECUTION_FREEZE_SCHEMA,
            "self_sha256": "8" * 64,
        },
        "global_attempt_marker_path": str(
            control / subject.GLOBAL_ATTEMPT_FILENAME
        ),
        "implementation_freeze_binding": {
            "file_sha256": "9" * 64,
            "path": str(
                control / subject.IMPLEMENTATION_FREEZE_FILENAME
            ),
            "schema": subject.IMPLEMENTATION_FREEZE_SCHEMA,
            "self_sha256": "a" * 64,
        },
        "incident_binding": {
            "file_sha256": subject.EXPECTED_INCIDENT_FILE_SHA256,
            "path": str(
                subject._PROJECT_ROOT / subject.INCIDENT_RELATIVE_PATH
            ),
            "schema": subject.INCIDENT_SCHEMA,
            "self_sha256": subject.EXPECTED_INCIDENT_SELF_SHA256,
        },
        "outer_safe_terminal_path": str(
            control / subject.OUTER_SAFE_TERMINAL_FILENAME
        ),
        "project_binding": {
            "file_count": 1,
            "path": str(subject._PROJECT_ROOT),
            "total_bytes": 1,
            "tree_sha256": "b" * 64,
        },
        "runtime_bindings": canary.runtime_bindings_payload(bindings),
        "schema": subject.CONFIG_SCHEMA,
        "service_unit_binding": {
            "env_executable_file_sha256": "c" * 64,
            "file_sha256": "d" * 64,
            "installed_path": str(
                subject.INSTALLED_USER_UNIT_DIRECTORY
                / Path(subject.FORMAL_UNIT_RELATIVE_PATH).name
            ),
            "path": str(
                subject._PROJECT_ROOT / subject.FORMAL_UNIT_RELATIVE_PATH
            ),
            "systemctl_executable_file_sha256": "e" * 64,
            "systemctl_executable_path": str(subject.SYSTEMCTL_PATH),
            "unit_name": Path(subject.FORMAL_UNIT_RELATIVE_PATH).name,
        },
        "source_bindings": {
            "dev": {
                "mode_octal": "0600",
                "path": str(source_root / "dev.json"),
                "sha256": subject.EXPECTED_DEV_SHA256,
                "size_bytes": subject.EXPECTED_DEV_SIZE_BYTES,
            },
            "train": {
                "mode_octal": "0600",
                "path": str(source_root / "train.json"),
                "sha256": subject.EXPECTED_TRAIN_SHA256,
                "size_bytes": subject.EXPECTED_TRAIN_SIZE_BYTES,
            },
        },
        "work_root": str(control / subject.CORE_WORK_ROOT_NAME),
    }
    return (
        {**body, "self_sha256": subject.stable_hash(body)},
        config_path,
    )


def _config(
    tmp_path: Path,
) -> tuple[subject.SourceFreeFormalConfig, runtime.RuntimeBindings]:
    bindings = _bindings(tmp_path / "bindings")
    payload, _path = _config_payload(tmp_path, bindings)
    return subject.parse_config(payload), bindings


def _pre_source_receipt(
    config: subject.SourceFreeFormalConfig,
) -> subject.PreSourceReceipt:
    return subject.PreSourceReceipt(
        project_tree_sha256=config.project_binding.tree_sha256,
        core_file_sha256=config.core_binding.file_sha256,
        design_self_sha256=config.design_binding.self_sha256,
        incident_self_sha256=config.incident_binding.self_sha256,
        implementation_freeze_self_sha256=(
            config.implementation_freeze_binding.self_sha256
        ),
        execution_freeze_self_sha256=(
            config.execution_freeze_binding.self_sha256
        ),
        service_unit_file_sha256=(
            config.service_unit_binding.file_sha256
        ),
        canary_terminal_self_sha256=(
            config.canary_binding.safe_terminal_self_sha256
        ),
        runtime_binding_sha256=runtime.stable_hash(
            config.runtime_bindings.semantic_payload()
        ),
    )


def _live_receipt() -> subject.LiveServiceReceipt:
    return subject.LiveServiceReceipt(
        attestation_sha256="f" * 64,
        invocation_id_sha256="0" * 64,
        main_pid=os.getpid(),
        restart_count=0,
    )


def _write_core_terminal(
    config: subject.SourceFreeFormalConfig,
    status: str,
) -> core.FormalRunResult:
    inner_body = {
        "execution_design_self_sha256": (
            subject.EXPECTED_DESIGN_SELF_SHA256
        ),
        "online_or_API_evaluation_count": 0,
        "retry_replay_resample_repair_count": 0,
        "schema": f"{controller.VERSION}_safe_terminal_v1",
        "status": status,
        "study_id": subject.STUDY_ID,
    }
    inner = {
        **inner_body,
        "terminal_self_sha256": controller.stable_hash(inner_body),
    }
    body = {
        "API_or_online_evaluation_call_count": 0,
        "effect_design_self_sha256": (
            subject.EXPECTED_DESIGN_SELF_SHA256
        ),
        "inner_scientific_terminal": inner,
        "retry_replay_resample_repair_or_fallback_count": 0,
        "schema": core.SAFE_TERMINAL_SCHEMA,
        "secret_generation_count": 1,
        "status": status,
        "study_id": subject.STUDY_ID,
    }
    terminal = {**body, "self_sha256": core.stable_hash(body)}
    config.work_root.mkdir(mode=0o700)
    path = config.work_root / core.TERMINAL_FILENAME
    _write(path, core.canonical_bytes(terminal), 0o400)
    return core.FormalRunResult(terminal=terminal, terminal_path=path)


def test_config_is_exact_canonical_and_cli_has_only_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = _bindings(tmp_path / "bindings")
    payload, path = _config_payload(tmp_path, bindings)
    _write(path, subject.canonical_bytes(payload), 0o400)
    loaded = subject.load_config(path)
    assert loaded.payload() == payload

    extra = {**payload, "provider": "forbidden"}
    with pytest.raises(subject.QuacP1FormalOuterError, match="shape"):
        subject.parse_config(extra)

    wrong_size = dict(payload)
    wrong_sources = dict(payload["source_bindings"])
    wrong_train = dict(wrong_sources["train"])
    wrong_train["size_bytes"] = subject.EXPECTED_TRAIN_SIZE_BYTES + 1
    wrong_sources["train"] = wrong_train
    wrong_size["source_bindings"] = wrong_sources
    wrong_body = {
        key: value
        for key, value in wrong_size.items()
        if key != "self_sha256"
    }
    wrong_size["self_sha256"] = subject.stable_hash(wrong_body)
    with pytest.raises(
        subject.QuacP1FormalOuterError,
        match="frozen study identity drifted",
    ):
        subject.parse_config(wrong_size)

    with pytest.raises(SystemExit):
        subject.main(["--config", str(path), "--retry"])

    called = []
    monkeypatch.setattr(
        subject,
        "run_formal_production",
        lambda config: called.append(config),
    )
    assert subject.main(["--config", str(path)]) == 0
    assert called == [loaded]


def test_internal_seam_orders_all_freezes_before_exact_source_reads(
    tmp_path: Path,
) -> None:
    config, _bindings_value = _config(tmp_path)
    config.control_root.mkdir(mode=0o700)
    events: list[str] = []

    def preflight(received):
        assert received.global_attempt_marker_path.is_file()
        events.append("preflight")
        return _pre_source_receipt(received)

    def live(received):
        assert received is config
        events.append("live")
        return _live_receipt()

    token = object()

    def verify(bindings, *, source_access_count):
        assert bindings is config.runtime_bindings
        assert source_access_count == 0
        events.append("runtime")
        return token

    executor = object()
    ops = object()

    def build(received, verified):
        assert received is config and verified is token
        events.append("build")
        return executor, ops

    def read(binding, *, field, open_counts):
        events.append(f"read:{field}")
        open_counts[field] += 1
        return {"opaque": field}

    def run_core(**kwargs):
        events.append("core")
        assert kwargs["train_obj"] == {"opaque": "TRAIN"}
        assert kwargs["dev_obj"] == {"opaque": "DEV"}
        assert kwargs["block_executor"] is executor
        assert kwargs["scientific_ops"] is ops
        return _write_core_terminal(
            config,
            "VALID_NONPROMOTION_M_UNOPENED",
        )

    terminal = subject._run_with_dependencies(
        config,
        pre_source_verifier=preflight,
        live_service_attestor=live,
        runtime_verifier=verify,
        dependency_builder=build,
        source_reader=read,
        core_callable=run_core,
    )
    assert events == [
        "preflight",
        "live",
        "runtime",
        "build",
        "read:TRAIN",
        "read:DEV",
        "core",
    ]
    assert terminal["formal_source_fd_open_counts"] == {
        "DEV": 1,
        "TRAIN": 1,
    }
    assert terminal["prior_postqualification_hash_only_operation_count"] == 1
    assert terminal["prior_postqualification_hash_only_member_read_count"] == 2
    assert terminal["preformal_semantic_source_decode_count"] == 0
    assert terminal["live_service_attestation_count"] == 1
    assert (
        config.outer_safe_terminal_path.read_bytes()
        == subject.canonical_bytes(terminal)
    )
    with pytest.raises(
        subject.QuacP1FormalOuterError,
        match="attempt already exists",
    ):
        subject._run_with_dependencies(
            config,
            pre_source_verifier=preflight,
            live_service_attestor=live,
            runtime_verifier=verify,
            dependency_builder=build,
            source_reader=read,
            core_callable=run_core,
        )


def test_failure_after_global_claim_is_aggregate_and_never_retried(
    tmp_path: Path,
) -> None:
    config, _bindings_value = _config(tmp_path)
    config.control_root.mkdir(mode=0o700)
    source_calls = []

    def fail(_config):
        raise RuntimeError("private value must not escape")

    with pytest.raises(RuntimeError, match="private value"):
        subject._run_with_dependencies(
            config,
            pre_source_verifier=fail,
            live_service_attestor=lambda _config: _live_receipt(),
            runtime_verifier=lambda *_args, **_kwargs: object(),
            dependency_builder=lambda *_args: (object(), object()),
            source_reader=lambda *_args, **_kwargs: source_calls.append(1),
            core_callable=lambda **_kwargs: None,
        )
    failure = json.loads(config.outer_safe_terminal_path.read_text("ascii"))
    assert failure["status"] == (
        "IMPLEMENTATION_OR_INFRASTRUCTURE_INVALID_NO_RETRY"
    )
    assert failure["formal_source_fd_open_counts"] == {
        "DEV": 0,
        "TRAIN": 0,
    }
    assert failure["preformal_semantic_source_decode_count"] == 0
    assert "private value" not in config.outer_safe_terminal_path.read_text(
        "ascii"
    )
    assert source_calls == []


def test_source_fd_reader_checks_exact_identity_and_strict_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.json"
    raw = b'{"rows":[1,2]}\n'
    _write(path, raw, 0o600)
    binding = subject.SourceFileBinding(
        path=path,
        size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    counts = {"TEST": 0}
    assert subject._read_source_once(
        binding,
        field="TEST",
        open_counts=counts,
    ) == {"rows": [1, 2]}
    assert counts == {"TEST": 1}

    duplicate = tmp_path / "duplicate.json"
    duplicate_raw = b'{"x":1,"x":2}\n'
    _write(duplicate, duplicate_raw, 0o600)
    with pytest.raises(
        subject.QuacP1FormalOuterError,
        match="strict decode",
    ):
        subject._read_source_once(
            subject.SourceFileBinding(
                path=duplicate,
                size_bytes=len(duplicate_raw),
                sha256=hashlib.sha256(duplicate_raw).hexdigest(),
            ),
            field="DUP",
            open_counts={"DUP": 0},
        )


def test_required_implementation_paths_are_exact_and_exist() -> None:
    assert len(subject.REQUIRED_IMPLEMENTATION_RELATIVE_PATHS) == len(
        set(subject.REQUIRED_IMPLEMENTATION_RELATIVE_PATHS)
    )
    assert {
        "assumption_agent/__init__.py",
        "assumption_agent/models.py",
        "assumption_agent/benchmarks/__init__.py",
        "replication_runtime/__init__.py",
        "replication_runtime/maud_extraction_p2_official_v1/__init__.py",
        "replication_runtime/maud_extraction_p2_official_v1/worker.py",
        "replication_runtime/quac_p1_official_v1/__init__.py",
        "assumption_agent/benchmarks/quac_p1_formal_acquisition_v1.py",
    }.issubset(subject.REQUIRED_IMPLEMENTATION_RELATIVE_PATHS)
    assert all(
        (subject._PROJECT_ROOT / relative).is_file()
        for relative in subject.REQUIRED_IMPLEMENTATION_RELATIVE_PATHS
    )


def test_production_dependency_builder_consumes_verified_token(
    tmp_path: Path,
) -> None:
    config, _bindings_value = _config(tmp_path)
    verified = runtime.verify_runtime_bindings_once(
        config.runtime_bindings,
        source_access_count=0,
    )
    executor, ops = subject._build_production_dependencies(
        config,
        verified,
    )
    assert isinstance(executor, core.BoundRuntimeExecutor)
    assert executor.bindings is config.runtime_bindings
    assert executor.verified_bindings is verified
    assert isinstance(executor.action_adapter, runtime.FrozenActionAdapter)
    assert isinstance(ops, core.FrozenScientificOps)


def test_canary_terminal_cannot_rebind_asset_freeze_even_if_rehashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = _bindings(tmp_path / "bindings")
    payload, _config_path_value = _config_payload(tmp_path, bindings)
    canary_config_path = tmp_path / "canary/config.json"
    canary_config_raw = b"bound canary config\n"
    _write(canary_config_path, canary_config_raw, 0o400)
    expected_asset_self = "1" * 64
    rebound_asset_self = "2" * 64
    runtime_binding_sha256 = runtime.stable_hash(
        bindings.semantic_payload()
    )
    safe_body = {
        "API_or_online_evaluation_call_count": 0,
        "aggregate_only_public_receipt": True,
        "asset_freeze_self_sha256": rebound_asset_self,
        "canary_attempt_file_sha256": "3" * 64,
        "config_self_sha256": "4" * 64,
        "effect_execution_design_self_sha256": (
            subject.EXPECTED_DESIGN_SELF_SHA256
        ),
        "formal_source_access_count": 0,
        "max_concurrent_physical_model_lanes": 2,
        "minilm_encode_call_count": 1,
        "official_index_call_count": 1,
        "official_retrieve_call_count": 1,
        "parallel_submission_barrier_passed": True,
        "project_binding_sha256": "5" * 64,
        "retry_replay_resample_or_fallback_count": 0,
        "runtime_binding_sha256": runtime_binding_sha256,
        "runtime_safe_terminal_self_sha256": "6" * 64,
        "runtime_verification_token_sha256": "7" * 64,
        "schema": canary.SAFE_TERMINAL_SCHEMA,
        "source_path_loader_label_qrel_answer_input_count": 0,
        "status": "passed_source_free_two_lane_single_index_canary",
        "study_id": subject.STUDY_ID,
        "synthetic_document_count": canary.SYNTHETIC_DOCUMENT_COUNT,
        "synthetic_query_count": canary.SYNTHETIC_QUERY_COUNT,
    }
    safe = {
        **safe_body,
        "self_sha256": canary.stable_hash(safe_body),
    }
    safe_path = tmp_path / "canary/terminal.json"
    safe_raw = canary.canonical_bytes(safe)
    _write(safe_path, safe_raw, 0o400)
    payload["canary_binding"] = {
        "config_file_sha256": hashlib.sha256(
            canary_config_raw
        ).hexdigest(),
        "config_path": str(canary_config_path),
        "config_self_sha256": "4" * 64,
        "safe_terminal_file_sha256": hashlib.sha256(
            safe_raw
        ).hexdigest(),
        "safe_terminal_path": str(safe_path),
        "safe_terminal_self_sha256": safe["self_sha256"],
    }
    body = {
        key: value
        for key, value in payload.items()
        if key != "self_sha256"
    }
    payload["self_sha256"] = subject.stable_hash(body)
    config = subject.parse_config(payload)

    class _Project:
        @staticmethod
        def verify() -> None:
            return None

    asset_state = {"calls": 0, "fail": False}

    class _Asset:
        self_sha256 = expected_asset_self

        @staticmethod
        def verify(received_bindings) -> None:
            assert received_bindings is config.runtime_bindings
            asset_state["calls"] += 1
            if asset_state["fail"]:
                raise canary.QuacP1SourceFreeCanaryError(
                    "asset freeze missing or drifted"
                )

    fake_canary_config = SimpleNamespace(
        self_sha256="4" * 64,
        runtime_bindings=config.runtime_bindings,
        project_binding=_Project(),
        asset_freeze_binding=_Asset(),
    )
    monkeypatch.setattr(
        canary,
        "load_config",
        lambda path: (
            fake_canary_config
            if path == canary_config_path
            else pytest.fail("unexpected canary path")
        ),
    )
    with pytest.raises(
        subject.QuacP1FormalOuterError,
        match="safe terminal binding drifted",
    ):
        subject._verify_canary(config)
    assert asset_state["calls"] == 1

    asset_state["fail"] = True
    with pytest.raises(
        canary.QuacP1SourceFreeCanaryError,
        match="asset freeze missing or drifted",
    ):
        subject._verify_canary(config)
    assert asset_state["calls"] == 2


def test_service_unit_and_live_fragment_bind_exact_closed_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "deployed/reconstruction_v2"
    installed_root = tmp_path / "real_home/.config/systemd/user"
    monkeypatch.setattr(subject, "_PROJECT_ROOT", project_root)
    monkeypatch.setattr(
        subject,
        "INSTALLED_USER_UNIT_DIRECTORY",
        installed_root,
    )
    bindings = _bindings(tmp_path / "bindings")
    payload, _path = _config_payload(tmp_path, bindings)
    preliminary = subject.parse_config(payload)
    unit_path = project_root / subject.FORMAL_UNIT_RELATIVE_PATH
    installed_path = (
        installed_root / Path(subject.FORMAL_UNIT_RELATIVE_PATH).name
    )
    argv = subject._expected_service_argv(preliminary)
    directives = subject._expected_service_directives(preliminary)
    unit_text = "\n".join(
        (
            "[Unit]",
            "Description=QuAC P1 formal one-shot",
            "",
            "[Service]",
            f"ExecStart={subject.shlex.join(argv)}",
            *(
                f"{key}={value}"
                for key, value in sorted(directives.items())
            ),
            "",
        )
    ).encode("utf-8")
    _write(unit_path, unit_text, 0o400)
    installed_path.parent.mkdir(parents=True)
    installed_path.symlink_to(unit_path)
    service = dict(payload["service_unit_binding"])
    service.update(
        {
            "env_executable_file_sha256": hashlib.sha256(
                subject.ENV_PATH.read_bytes()
            ).hexdigest(),
            "file_sha256": hashlib.sha256(unit_text).hexdigest(),
            "installed_path": str(installed_path),
            "path": str(unit_path),
            "systemctl_executable_file_sha256": hashlib.sha256(
                subject.SYSTEMCTL_PATH.read_bytes()
            ).hexdigest(),
        }
    )
    payload["service_unit_binding"] = service
    body = {
        key: value
        for key, value in payload.items()
        if key != "self_sha256"
    }
    payload["self_sha256"] = subject.stable_hash(body)
    config = subject.parse_config(payload)
    subject._verify_service_unit(config)

    environment = subject._expected_service_environment(config)
    assert environment["HOME"] == str(config.control_root / "home")
    assert Path(environment["HOME"]) != installed_path.parents[3]
    assert environment["PYTHONPATH"] == os.pathsep.join(
        (
            str(project_root),
            config.runtime_bindings.gpu0_python.import_tree.path,
        )
    )
    python_index = argv.index(
        config.runtime_bindings.gpu0_python.executable.path
    )
    assert argv[python_index + 1 : python_index + 5] == [
        "-S",
        "-B",
        "-s",
        "-m",
    ]
    assert "-P" not in argv

    tampered_argv = list(argv)
    pythonpath_index = next(
        index
        for index, value in enumerate(tampered_argv)
        if value.startswith("PYTHONPATH=")
    )
    tampered_argv[pythonpath_index] = f"PYTHONPATH={project_root}"
    tampered_text = unit_text.replace(
        f"ExecStart={subject.shlex.join(argv)}".encode("utf-8"),
        (
            f"ExecStart={subject.shlex.join(tampered_argv)}"
        ).encode("utf-8"),
    )
    unit_path.chmod(0o600)
    unit_path.write_bytes(tampered_text)
    unit_path.chmod(0o400)
    tampered_payload = dict(payload)
    tampered_service = dict(service)
    tampered_service["file_sha256"] = hashlib.sha256(
        tampered_text
    ).hexdigest()
    tampered_payload["service_unit_binding"] = tampered_service
    tampered_body = {
        key: value
        for key, value in tampered_payload.items()
        if key != "self_sha256"
    }
    tampered_payload["self_sha256"] = subject.stable_hash(
        tampered_body
    )
    tampered_config = subject.parse_config(tampered_payload)
    with pytest.raises(
        subject.QuacP1FormalOuterError,
        match="service unit semantic binding drifted",
    ):
        subject._verify_service_unit(tampered_config)

    unit_path.chmod(0o600)
    unit_path.write_bytes(unit_text)
    unit_path.chmod(0o400)

    def systemctl_result(fragment: Path) -> SimpleNamespace:
        rows = {
            "ActiveState": "activating",
            "ExecMainPID": str(os.getpid()),
            "FragmentPath": str(fragment),
            "InvocationID": "1" * 32,
            "MainPID": str(os.getpid()),
            "NRestarts": "0",
            "Restart": "no",
            "SubState": "start",
            "Type": "oneshot",
        }
        stdout = "".join(
            f"{key}={value}\n" for key, value in rows.items()
        ).encode("ascii")
        return SimpleNamespace(returncode=0, stderr=b"", stdout=stdout)

    monkeypatch.setattr(
        subject.subprocess,
        "run",
        lambda *_args, **_kwargs: systemctl_result(installed_path),
    )
    receipt = subject._verify_live_service_attestation(config)
    assert receipt.main_pid == os.getpid()
    assert receipt.restart_count == 0

    monkeypatch.setattr(
        subject.subprocess,
        "run",
        lambda *_args, **_kwargs: systemctl_result(
            tmp_path / "wrong.service"
        ),
    )
    with pytest.raises(
        subject.QuacP1FormalOuterError,
        match="fragment path",
    ):
        subject._verify_live_service_attestation(config)
