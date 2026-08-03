from __future__ import annotations

import fcntl
import hashlib
import json
import io
import os
from pathlib import Path
import stat
import tarfile
from types import SimpleNamespace

import pytest

import hegel_machine.phase3_m3_implementation_qualification_v1 as qualification
import hegel_machine.phase3_dsl_v1 as full_dsl
import hegel_machine.phase3_m3_dsl_core_v1 as m3_dsl_core
import hegel_machine.phase3_m3_shrink1_core_v1 as m3_shrink_core
import hegel_machine.phase3_shrink1_registry_v1 as full_shrink
from hegel_machine.phase3_m25_wire_v1 import build_formal_object, id_digest_v1
from hegel_machine.phase3_m3_record_wire_v1 import build_m3_record_object_v1
from hegel_machine.phase3_m3_bounded_enumerator_cli_v1 import result_report, write_artifacts
from hegel_machine.phase3_m3_bounded_enumerator_v1 import (
    EnumerationBindingsV1,
    enumerate_bounded_closure_v1,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent


def _working_blob(_repository: Path, _commit: str, path: str) -> bytes:
    return (REPOSITORY_ROOT / path).read_bytes()


def _golden():
    value = json.loads(
        (PROJECT_ROOT / "golden_vectors/phase3_m3_bounded_dual_agreement_v1.json").read_text()
    )
    return qualification.validate_dual_golden_v1(value)


def _daemon_receipt() -> dict[str, object]:
    body: dict[str, object] = {
        "schema": "hegel-phase3-local-docker-daemon-identity/1",
        "local_linux_daemon": True,
    }
    encoded = json.dumps(body, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
        "ascii"
    )
    body["receipt_sha256"] = qualification.hashlib.sha256(encoded).hexdigest()
    return body


def _full_report(implementation: str) -> dict[str, object]:
    golden, _, _ = _golden()
    expected = golden["expected"]
    bindings = golden["binding_roots"]
    assert isinstance(expected, dict) and isinstance(bindings, dict)
    identity = {
        "python": (
            "hegel-m3-python-closure-enumerator-report/1",
            1,
            "hegel-python-m3-bounded-closure-enumerator-v1",
        ),
        "rust": (
            "hegel-m3-rust-closure-enumerator-report/1",
            2,
            "hegel-rust-m3-bounded-closure-enumerator-v1",
        ),
    }[implementation]
    report: dict[str, object] = {
        "schema_version": identity[0],
        "claim_level": "FORMAL_PROFILE_CANDIDATE_NOT_AUTHORITY",
        "authoritative_claim_allowed": False,
        "implementation": implementation,
        "implementation_id": identity[1],
        "implementation_machine_id": identity[2],
        "raw_expansion_limit_hit": False,
        "wall_clock_abort_hit": False,
        "target_roles_evaluated": False,
        "split_material_accessed": False,
        "secrets_accessed": False,
        **bindings,
    }
    renamed = {
        "bucket_accounting_root": "bucket_accounting_root_or_null",
        "canonical_program_archive_root": "canonical_program_archive_root_or_null",
        "first_out_of_budget_program_cbor_hex": "first_out_of_budget_program_cbor_hex_or_null",
        "first_out_of_budget_program_hash": "first_out_of_budget_program_hash_or_null",
        "program_chunk_manifest_root": "program_chunk_manifest_root_or_null",
    }
    for key, value in expected.items():
        report[renamed.get(key, key)] = value
    assert set(report) == qualification.REPORT_FIELDS
    return report


def _implementation_receipt(
    implementation_id: int, binding_root: bytes
) -> dict[str, object]:
    byte = "11" if implementation_id == 1 else "22"
    return {
        "implementation_id": implementation_id,
        "implementation_machine_id": (
            "hegel-python-m3-bounded-closure-enumerator-v1"
            if implementation_id == 1
            else "hegel-rust-m3-bounded-closure-enumerator-v1"
        ),
        "source_root": byte * 32,
        "source_file_count": 9 if implementation_id == 1 else 13,
        "dependency_lock_root": "33" * 32,
        "dependency_snapshot_root_or_null": None if implementation_id == 1 else "dd" * 32,
        "dependency_snapshot_file_count": 0 if implementation_id == 1 else 1054,
        "execution_environment_spec_root": "44" * 32,
        "image_ref": "python@sha256:" + "55" * 32 if implementation_id == 1 else "rust@sha256:" + "66" * 32,
        "bound_executable_locator": (
            "oci://python@sha256:" + "55" * 32 + "/usr/local/bin/python3.12"
            if implementation_id == 1
            else "generated-target://rust/m3_closure_enumerator/target/"
            + "m3_qualification/"
            + "12" * 20
            + "/hegel-m3-closure-enumerator"
        ),
        "binary_digest": "77" * 32,
        "compiler_or_interpreter_version_digest": "88" * 32,
        "entrypoint_id_digest": (
            id_digest_v1(qualification.PYTHON_ENTRYPOINT_ID).hex()
            if implementation_id == 1
            else id_digest_v1(qualification.RUST_ENTRYPOINT_ID).hex()
        ),
        "implementation_binding_root": binding_root.hex(),
        "canonical_report_sha256": "99" * 32,
        "execution_stdout_sha256": "aa" * 32,
        "runtime_container_environment_sha256": qualification._container_environment_digest(
            qualification.PYTHON_RUNTIME_ENVIRONMENT
            if implementation_id == 1
            else qualification.RUST_RUNTIME_ENVIRONMENT
        ).hex(),
        "build_container_environment_sha256_or_null": (
            None
            if implementation_id == 1
            else qualification._container_environment_digest(
                qualification.RUST_BUILD_ENVIRONMENT
            ).hex()
        ),
        "canonical_program_records_stream_sha256": "ad" * 32,
        "program_chunk_manifests_stream_sha256": "ae" * 32,
        "bucket_accounting_records_stream_sha256": "af" * 32,
        "build_stdout_sha256_or_null": (
            None if implementation_id == 1 else hashlib.sha256(b"").hexdigest()
        ),
        "build_stderr_sha256_or_null": (
            None if implementation_id == 1 else hashlib.sha256(b"").hexdigest()
        ),
        "input_snapshot_target_free": True,
        "archive_file_set_verified": True,
        "host_strict_archive_replay_verified": True,
        "witness_adjacency_verified": True,
    }


def test_target_free_python_and_rust_source_closures_are_exact(monkeypatch) -> None:
    monkeypatch.setattr(qualification, "_git_blob", _working_blob)
    python = qualification.validate_python_source_closure_v1(
        REPOSITORY_ROOT, "12" * 20
    )
    rust = qualification.validate_rust_source_closure_v1(
        REPOSITORY_ROOT, "12" * 20
    )
    assert tuple(python) == qualification.PYTHON_SOURCE_PATHS
    assert set(rust) == set(qualification.RUST_SOURCE_PATHS)
    assert all(b"ODD_REDUCTION_UNIVERSE" not in payload for payload in python.values())
    assert all(b"formal_bridge_m25" not in payload for payload in rust.values())


def test_target_free_core_is_exact_projection_of_frozen_full_dsl() -> None:
    for name in (
        "AGGREGATE_MAP_IDS",
        "CONTEXT_IDS",
        "QUANTITY_IDS",
        "SCOPE_IDS",
        "TASK_IDS",
    ):
        assert getattr(m3_dsl_core, name) == getattr(full_dsl, name)
    assert tuple(
        (atom.numerator, atom.denominator)
        for atom in m3_dsl_core.RATIONAL_PARAMETER_GRID
    ) == tuple(
        (atom.numerator, atom.denominator)
        for atom in full_dsl.RATIONAL_PARAMETER_GRID
    )
    for field in m3_dsl_core.StructuralLimits.__dataclass_fields__:
        assert getattr(m3_dsl_core.STRUCTURAL_LIMITS, field) == getattr(
            full_dsl.STRUCTURAL_LIMITS, field
        )
    for name in (
        "DSL_VERSION",
        "REGISTRY_WIDTH",
        "ACTIVE_AGGREGATE_IDS",
        "TOMBSTONED_AGGREGATE_IDS",
        "TOMBSTONED_AGGREGATE_NAMES",
        "REMOVED_AGGREGATE_ERROR",
    ):
        assert getattr(m3_shrink_core, name) == getattr(full_shrink, name)


def test_target_free_record_encoder_is_bit_exact_with_full_registry() -> None:
    fields = {
        "chunk_index": 0,
        "first_program_index": 0,
        "last_program_index": 3,
        "record_count": 4,
        "canonical_program_record_subtree_root": b"a" * 32,
        "compressed_program_blob_hash": b"b" * 32,
        "uncompressed_program_byte_length": 123,
    }
    assert build_m3_record_object_v1("ProgramChunkManifestV2", fields) == build_formal_object(
        "ProgramChunkManifestV2", fields
    )


def test_runtime_environment_and_build_profiles_do_not_impersonate_formal_actors() -> None:
    environment = qualification._environment_fields(
        "python@sha256:" + "55" * 32, "python", b"l" * 32
    )
    assert environment["container_or_host_profile_id_digest"] == id_digest_v1(
        qualification.RUNTIME_DOCKER_POLICY_ID
    )
    assert environment["container_or_host_profile_id_digest"] != id_digest_v1(
        "hegel-owner-accepted-container-technical-actors-v1"
    )
    rust = qualification._implementation_binding_fields(
        implementation_id=2,
        source_root=b"s" * 32,
        binary_digest=b"b" * 32,
        environment_root=b"e" * 32,
        version_digest=b"v" * 32,
        dependency_lock_root=b"l" * 32,
        entrypoint=qualification.RUST_ENTRYPOINT_ID,
        golden_root=b"g" * 32,
        commit_wire=(1, b"c" * 20),
    )
    assert rust["build_profile_id_digest"] == id_digest_v1(
        qualification.RUST_BUILD_DOCKER_POLICY_ID
    )


def test_golden_rejects_extra_fields_and_authority_promotion() -> None:
    path = PROJECT_ROOT / "golden_vectors/phase3_m3_bounded_dual_agreement_v1.json"
    value = json.loads(path.read_text())
    value["unexpected"] = False
    with pytest.raises(qualification.M3ImplementationQualificationError):
        qualification.validate_dual_golden_v1(value)
    value = json.loads(path.read_text())
    value["authoritative_claim_allowed"] = True
    with pytest.raises(qualification.M3ImplementationQualificationError):
        qualification.validate_dual_golden_v1(value)


def test_bare_cli_report_cannot_claim_formal_authority() -> None:
    golden, _, _ = _golden()
    report = _full_report("python")
    qualification.validate_enumerator_report_v1(
        report, implementation="python", golden=golden
    )
    report["claim_level"] = "FORMAL_ENUMERATION_OUTPUT"
    report["authoritative_claim_allowed"] = True
    with pytest.raises(qualification.M3ImplementationQualificationError) as captured:
        qualification.validate_enumerator_report_v1(
            report, implementation="python", golden=golden
        )
    assert captured.value.code == qualification.FAIL_REPORT


def test_receipt_is_typed_exact_field_and_non_authoritative() -> None:
    golden, _, golden_root = _golden()
    receipt = dict(
        qualification._build_receipt(
            basis_commit="12" * 20,
            golden=golden,
            golden_root=golden_root,
            python_fields=_implementation_receipt(1, b"p" * 32),
            rust_fields=_implementation_receipt(2, b"r" * 32),
            runtime_seccomp_digest=b"s" * 32,
            build_seccomp_digest=b"b" * 32,
            docker_daemon_receipt_binding=b"d" * 32,
            cargo_bootstrap_record_digest=b"c" * 32,
        )
    )
    root = qualification.validate_qualification_receipt_v1(
        receipt, golden=golden, basis_commit="12" * 20
    )
    assert root.hex() == receipt["receipt_root"]
    assert receipt["authoritative_claim_allowed"] is False
    assert receipt["formal_m3_output_roots_generated"] is False
    assert receipt["m3_state"] == "NOT_RUN"
    receipt["unexpected"] = None
    with pytest.raises(qualification.M3ImplementationQualificationError):
        qualification.validate_qualification_receipt_v1(
            receipt, golden=golden, basis_commit="12" * 20
        )


@pytest.mark.parametrize(
    ("implementation", "field", "value"),
    [
        ("python", "implementation_machine_id", "arbitrary-machine"),
        ("python", "source_file_count", -7),
        ("python", "image_ref", "not-even-an-image"),
        ("rust", "dependency_snapshot_file_count", 0),
        ("rust", "build_stdout_sha256_or_null", "bb" * 32),
        ("rust", "build_stderr_sha256_or_null", "cc" * 32),
    ],
)
def test_receipt_rejects_semantically_invalid_rehashed_rows(
    implementation: str, field: str, value: object
) -> None:
    golden, _, golden_root = _golden()
    python = _implementation_receipt(1, b"p" * 32)
    rust = _implementation_receipt(2, b"r" * 32)
    {"python": python, "rust": rust}[implementation][field] = value
    with pytest.raises(qualification.M3ImplementationQualificationError) as captured:
        qualification._build_receipt(
            basis_commit="12" * 20,
            golden=golden,
            golden_root=golden_root,
            python_fields=python,
            rust_fields=rust,
            runtime_seccomp_digest=b"s" * 32,
            build_seccomp_digest=b"b" * 32,
            docker_daemon_receipt_binding=b"d" * 32,
            cargo_bootstrap_record_digest=b"c" * 32,
        )
    assert captured.value.code == qualification.FAIL_RECEIPT


def test_receipt_rejects_caller_supplied_golden_root_substitution() -> None:
    golden, _, _ = _golden()
    with pytest.raises(qualification.M3ImplementationQualificationError) as captured:
        qualification._build_receipt(
            basis_commit="12" * 20,
            golden=golden,
            golden_root=b"X" * 32,
            python_fields=_implementation_receipt(1, b"p" * 32),
            rust_fields=_implementation_receipt(2, b"r" * 32),
            runtime_seccomp_digest=b"s" * 32,
            build_seccomp_digest=b"b" * 32,
            docker_daemon_receipt_binding=b"d" * 32,
            cargo_bootstrap_record_digest=b"c" * 32,
        )
    assert captured.value.code == qualification.FAIL_RECEIPT


def test_cargo_dependency_snapshot_is_lock_exact_and_private(
    monkeypatch, tmp_path: Path
) -> None:
    archive_path = tmp_path / "demo-1.0.0.crate"
    payload = b"[package]\nname='demo'\nversion='1.0.0'\n"
    info = tarfile.TarInfo("demo-1.0.0/Cargo.toml")
    info.size = len(payload)
    info.mode = 0o644
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(payload))
    checksum = qualification.hashlib.sha256(archive_path.read_bytes()).hexdigest()
    lock = (
        "version = 4\n\n[[package]]\n"
        "name = \"demo\"\nversion = \"1.0.0\"\n"
        "source = \"registry+https://github.com/rust-lang/crates.io-index\"\n"
        f"checksum = \"{checksum}\"\n"
    ).encode()
    monkeypatch.setattr(
        qualification,
        "_cached_crate_path",
        lambda name, version, digest: archive_path,
    )
    vendor = tmp_path / "vendor"
    root, count = qualification._build_cargo_dependency_snapshot(lock, vendor)
    assert len(root) == 32
    assert count == 2
    assert vendor.stat().st_mode & 0o777 == 0o700
    checksum_record = json.loads(
        (vendor / "demo-1.0.0/.cargo-checksum.json").read_text()
    )
    assert checksum_record["package"] == checksum
    assert checksum_record["files"]["Cargo.toml"] == qualification.hashlib.sha256(
        payload
    ).hexdigest()


def test_rust_build_mounts_only_private_vendor_snapshot_and_local_control_plane(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    lock = source / "Hegel Machine/rust/m3_closure_enumerator/Cargo.lock"
    lock.parent.mkdir(parents=True)
    lock.write_bytes(b"version = 4\n")
    persisted = tmp_path / "persist"
    (persisted / "rust/m3_closure_enumerator").mkdir(parents=True)
    monkeypatch.setattr(qualification, "PROJECT_ROOT", persisted)

    def fake_snapshot(_lock_payload: bytes, vendor: Path):
        vendor.mkdir(mode=0o700)
        return b"v" * 32, 7

    captured: dict[str, object] = {}

    def fake_run(command, *, code, timeout, environment):
        captured.update(
            command=list(command), code=code, timeout=timeout, environment=dict(environment)
        )
        output_mount = next(
            value for value in command if isinstance(value, str) and value.endswith(":/output:rw")
        )
        target = Path(output_mount.removesuffix(":/output:rw"))
        binary = target / "release/hegel-m3-closure-enumerator"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"test-rust-binary")
        return qualification.subprocess.CompletedProcess(command, 0, b"", b"")

    control = SimpleNamespace(
        environment={
            "DOCKER_CONFIG": "/tmp/private-config",
            "DOCKER_HOST": "unix:///var/run/docker.sock",
            "HOME": "/tmp/private-home",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
        command=lambda *arguments: [
            "/usr/bin/docker",
            "--host=unix:///var/run/docker.sock",
            *arguments,
        ],
    )
    monkeypatch.setattr(
        qualification, "_build_cargo_dependency_snapshot", fake_snapshot
    )
    monkeypatch.setattr(qualification, "_run", fake_run)
    binary, _, _, snapshot_root, file_count = qualification._build_rust(
        control,
        "rust@sha256:" + "66" * 32,
        source,
        seccomp_path=PROJECT_ROOT / "config/phase3_m3_offline_build_seccomp_v1.json",
        basis_commit="12" * 20,
        repository_root=REPOSITORY_ROOT,
    )
    command = captured["command"]
    assert isinstance(command, list)
    assert command[:2] == [
        "/usr/bin/docker",
        "--host=unix:///var/run/docker.sock",
    ]
    assert "--pull=never" in command and "--network=none" in command
    assert "--offline" in command and "/vendor" in " ".join(command)
    assert command.count("--quiet") == 1
    assert ".cargo/registry" not in " ".join(command)
    assert not any(key.endswith("_PROXY") for key in captured["environment"])
    assert binary.read_bytes() == b"test-rust-binary"
    assert snapshot_root == b"v" * 32 and file_count == 7


@pytest.mark.parametrize(
    ("stdout", "stderr"),
    [(b"unexpected cargo stdout\n", b""), (b"", b"unexpected cargo stderr\n")],
)
def test_quiet_cargo_success_with_output_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdout: bytes,
    stderr: bytes,
) -> None:
    source = tmp_path / "source"
    crate = source / "Hegel Machine/rust/m3_closure_enumerator"
    crate.mkdir(parents=True)
    (crate / "Cargo.lock").write_bytes(b"version = 4\n")
    persisted = tmp_path / "persist"
    (persisted / "rust/m3_closure_enumerator").mkdir(parents=True)
    monkeypatch.setattr(qualification, "PROJECT_ROOT", persisted)

    def fake_snapshot(_lock_payload: bytes, vendor: Path):
        vendor.mkdir(mode=0o700)
        return b"v" * 32, 7

    def fake_run(command, *, code, timeout, environment):
        return qualification.subprocess.CompletedProcess(command, 0, stdout, stderr)

    control = SimpleNamespace(
        environment={
            "DOCKER_CONFIG": "/tmp/private-config",
            "DOCKER_HOST": "unix:///var/run/docker.sock",
            "HOME": "/tmp/private-home",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
        command=lambda *arguments: [
            "/usr/bin/docker",
            "--host=unix:///var/run/docker.sock",
            *arguments,
        ],
    )
    monkeypatch.setattr(qualification, "_build_cargo_dependency_snapshot", fake_snapshot)
    monkeypatch.setattr(qualification, "_run", fake_run)

    with pytest.raises(qualification.M3ImplementationQualificationError) as captured:
        qualification._build_rust(
            control,
            "rust@sha256:" + "66" * 32,
            source,
            seccomp_path=PROJECT_ROOT / "config/phase3_m3_offline_build_seccomp_v1.json",
            basis_commit="12" * 20,
            repository_root=REPOSITORY_ROOT,
        )
    assert captured.value.code == qualification.FAIL_BUILD
    assert "successful quiet offline Cargo build emitted output" in captured.value.detail


def test_failed_quiet_cargo_build_preserves_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = qualification.subprocess.CompletedProcess(
        ["cargo", "build", "--quiet"],
        101,
        b"",
        b"error: expected expression\ncould not compile `hegel`\n",
    )
    monkeypatch.setattr(
        qualification.subprocess, "run", lambda *_args, **_kwargs: completed
    )
    with pytest.raises(qualification.M3ImplementationQualificationError) as captured:
        qualification._run(
            completed.args,
            code=qualification.FAIL_BUILD,
            timeout=1,
            environment={"PATH": "/usr/bin:/bin"},
        )
    assert captured.value.code == qualification.FAIL_BUILD
    assert "expected expression" in captured.value.detail
    assert "could not compile" in captured.value.detail


def test_host_replay_decodes_every_record_and_rejects_stream_tampering(
    tmp_path: Path,
) -> None:
    roots = (b"a" * 32, b"b" * 32, b"c" * 32)
    bindings = EnumerationBindingsV1(*roots)
    result = enumerate_bounded_closure_v1(
        bindings, canonical_budget=10, raw_application_cap=100_000
    )
    report = result_report(
        result, bindings, canonical_budget=10, raw_cap=100_000
    )
    write_artifacts(tmp_path / "archive", result, report)
    replay = qualification._host_validate_enumerator_archive_v1(
        tmp_path,
        implementation="python",
        stdout_report=report,
        roots=roots,
    )
    assert replay["witness_adjacency_verified"] is True
    assert replay["residual_out_of_budget_canonical_programs"] == 4
    stream = tmp_path / "archive/canonical_program_records.cborframed"
    corrupted = bytearray(stream.read_bytes())
    corrupted[-1] ^= 1
    stream.write_bytes(corrupted)
    with pytest.raises(qualification.M3ImplementationQualificationError):
        qualification._host_validate_enumerator_archive_v1(
            tmp_path,
            implementation="python",
            stdout_report=report,
            roots=roots,
        )


def test_builder_input_assembly_carries_runtime_and_build_policy_before_ready(
    tmp_path: Path,
) -> None:
    golden, _, golden_root = _golden()
    python_binding_root = b"p" * 32
    rust_binding_root = b"r" * 32
    runtime_seccomp = b"s" * 32
    build_seccomp = b"b" * 32
    daemon_receipt = _daemon_receipt()
    daemon_binding = qualification.local_docker_daemon_receipt_binding_v1(
        daemon_receipt
    )
    receipt = qualification._build_receipt(
        basis_commit="12" * 20,
        golden=golden,
        golden_root=golden_root,
        python_fields=_implementation_receipt(1, python_binding_root),
        rust_fields=_implementation_receipt(2, rust_binding_root),
        runtime_seccomp_digest=runtime_seccomp,
        build_seccomp_digest=build_seccomp,
        docker_daemon_receipt_binding=daemon_binding,
        cargo_bootstrap_record_digest=b"c" * 32,
    )
    inputs = qualification._qualified_implementation_inputs_v1(
        {"m3_execution_implementation_bindings_ready": False},
        receipt=receipt,
        receipt_root=bytes.fromhex(receipt["receipt_root"]),
        golden_root=golden_root,
        runtime_seccomp_digest=runtime_seccomp,
        build_seccomp_digest=build_seccomp,
        docker_daemon_receipt=daemon_receipt,
        docker_daemon_receipt_binding=daemon_binding,
        cargo_bootstrap_record_digest=b"c" * 32,
        python_source_root=b"1" * 32,
        rust_source_root=b"2" * 32,
        python_image="python@sha256:" + "55" * 32,
        rust_image="rust@sha256:" + "66" * 32,
        python_binary_digest=b"3" * 32,
        rust_binary_path=tmp_path / "hegel-m3-closure-enumerator",
        rust_binary_digest=b"4" * 32,
        rust_dependency_snapshot_root=b"\xdd" * 32,
        rust_dependency_snapshot_file_count=1054,
        python_binding_root=python_binding_root,
        rust_binding_root=rust_binding_root,
    )
    assert inputs["m3_execution_implementation_bindings_ready"] is True
    assert inputs["m3_runtime_seccomp_sha256"] == runtime_seccomp
    assert inputs["m3_build_seccomp_sha256"] == build_seccomp
    assert inputs["m3_runtime_docker_policy_id"] == qualification.RUNTIME_DOCKER_POLICY_ID
    assert inputs["m3_rust_build_docker_policy_id"] == qualification.RUST_BUILD_DOCKER_POLICY_ID


def test_cargo_bootstrap_record_uses_portable_machine_path_bindings() -> None:
    artifact = (
        PROJECT_ROOT / "artifacts/phase3_m3_cargo_offline_bootstrap_record_v1.json"
    )
    lock_payload = (
        PROJECT_ROOT / "rust/m3_closure_enumerator/Cargo.lock"
    ).read_bytes()
    payload = artifact.read_bytes()
    value = json.loads(payload)
    command = value["successful_exact_command"]

    assert "/home/" not in command
    assert "${HEGEL_BOOTSTRAP_CARGO_REGISTRY:" in command
    assert "${HEGEL_BOOTSTRAP_PROJECT_ROOT:" in command
    qualification._validate_cargo_bootstrap_record_v1(
        payload,
        cargo_lock_payload=lock_payload,
    )

    value["successful_exact_command"] = command.replace(
        "${HEGEL_BOOTSTRAP_PROJECT_ROOT:?set absolute Asumption Agent project root}",
        "/home/example/project",
    )
    attacked = json.dumps(value, ensure_ascii=True, sort_keys=True).encode("ascii")
    with pytest.raises(qualification.M3ImplementationQualificationError) as captured:
        qualification._validate_cargo_bootstrap_record_v1(
            attacked,
            cargo_lock_payload=lock_payload,
        )
    assert captured.value.code == qualification.FAIL_BUILD


def test_qualified_rust_binary_install_is_atomic_idempotent_and_no_follow(
    tmp_path: Path,
) -> None:
    trusted = tmp_path / "crate"
    trusted.mkdir(mode=0o700)
    built = tmp_path / "built"
    built.write_bytes(b"qualified-rust-binary")
    built.chmod(0o700)
    destination = trusted / "target/m3_qualification" / ("12" * 20) / "binary"

    installed = qualification._install_qualified_rust_binary_v1(
        built,
        destination,
        trusted_base=trusted,
    )
    assert installed == destination.absolute()
    assert destination.read_bytes() == built.read_bytes()
    assert stat.S_IMODE(destination.stat().st_mode) == 0o555

    victim = tmp_path / "victim"
    victim.write_bytes(b"do-not-overwrite")
    legacy_pending = destination.with_suffix(".pending")
    legacy_pending.symlink_to(victim)
    qualification._install_qualified_rust_binary_v1(
        built,
        destination,
        trusted_base=trusted,
    )
    assert victim.read_bytes() == b"do-not-overwrite"
    assert legacy_pending.is_symlink()


def test_qualified_rust_binary_install_rejects_destination_symlink_and_lock_race(
    tmp_path: Path,
) -> None:
    trusted = tmp_path / "crate"
    trusted.mkdir(mode=0o700)
    built = tmp_path / "built"
    built.write_bytes(b"qualified-rust-binary")
    built.chmod(0o700)
    parent = trusted / "target/m3_qualification" / ("34" * 20)
    parent.mkdir(parents=True, mode=0o700)
    destination = parent / "binary"
    victim = tmp_path / "victim"
    victim.write_bytes(b"untouched")
    destination.symlink_to(victim)
    with pytest.raises(qualification.M3ImplementationQualificationError) as captured:
        qualification._install_qualified_rust_binary_v1(
            built,
            destination,
            trusted_base=trusted,
        )
    assert captured.value.code == qualification.FAIL_BUILD
    assert victim.read_bytes() == b"untouched"

    destination.unlink()
    lock_path = parent / ".hegel-m3-qualification.lock"
    lock_descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(lock_descriptor, 0o600)
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(qualification.M3ImplementationQualificationError) as captured:
            qualification._install_qualified_rust_binary_v1(
                built,
                destination,
                trusted_base=trusted,
            )
        assert captured.value.code == qualification.FAIL_BUILD
        assert not destination.exists()
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def test_qualified_rust_binary_install_rejects_symlinked_base_ancestor(
    tmp_path: Path,
) -> None:
    real_ancestor = tmp_path / "real-ancestor"
    real_ancestor.mkdir(mode=0o700)
    real_trusted = real_ancestor / "crate"
    real_trusted.mkdir(mode=0o700)
    alias_ancestor = tmp_path / "alias-ancestor"
    alias_ancestor.symlink_to(real_ancestor, target_is_directory=True)
    aliased_trusted = alias_ancestor / "crate"

    built = tmp_path / "built"
    built.write_bytes(b"qualified-rust-binary")
    built.chmod(0o700)
    destination = (
        aliased_trusted / "target/m3_qualification" / ("56" * 20) / "binary"
    )

    with pytest.raises(qualification.M3ImplementationQualificationError) as captured:
        qualification._install_qualified_rust_binary_v1(
            built,
            destination,
            trusted_base=aliased_trusted,
        )
    assert captured.value.code == qualification.FAIL_BUILD
    assert not (real_trusted / "target").exists()


def test_qualified_rust_binary_install_detects_ancestor_replacement(
    monkeypatch, tmp_path: Path
) -> None:
    trusted = tmp_path / "crate"
    trusted.mkdir(mode=0o700)
    parent = trusted / "target/m3_qualification" / ("78" * 20)
    parent.mkdir(parents=True, mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)

    built = tmp_path / "built"
    built.write_bytes(b"qualified-rust-binary")
    built.chmod(0o700)
    destination = parent / "binary"
    real_link = os.link
    replaced = False

    def replace_after_link(*args, **kwargs):
        nonlocal replaced
        result = real_link(*args, **kwargs)
        if not replaced:
            replaced = True
            (trusted / "target").rename(trusted / "target-detached")
            (trusted / "target").symlink_to(outside, target_is_directory=True)
        return result

    monkeypatch.setattr(qualification.os, "link", replace_after_link)
    with pytest.raises(qualification.M3ImplementationQualificationError) as captured:
        qualification._install_qualified_rust_binary_v1(
            built,
            destination,
            trusted_base=trusted,
        )
    assert captured.value.code == qualification.FAIL_BUILD
    assert replaced is True
    assert not (outside / "m3_qualification").exists()
    detached_binary = (
        trusted
        / "target-detached/m3_qualification"
        / ("78" * 20)
        / "binary"
    )
    assert detached_binary.read_bytes() == built.read_bytes()


def test_runtime_guard_rejects_static_replayer_path_substitution(
    monkeypatch, tmp_path: Path
) -> None:
    commit = "12" * 20
    golden, _, golden_root = _golden()
    bootstrap_payload = (
        PROJECT_ROOT / "artifacts/phase3_m3_cargo_offline_bootstrap_record_v1.json"
    ).read_bytes()
    lock_payload = (
        PROJECT_ROOT / "rust/m3_closure_enumerator/Cargo.lock"
    ).read_bytes()
    (
        dependency_snapshot_root,
        dependency_snapshot_file_count,
        bootstrap_digest,
    ) = qualification._validate_cargo_bootstrap_record_v1(
        bootstrap_payload, cargo_lock_payload=lock_payload
    )
    monkeypatch.setattr(qualification, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        qualification,
        "load_committed_dual_golden_v1",
        lambda _repository, _commit: (golden, b"golden", golden_root),
    )
    monkeypatch.setattr(
        qualification,
        "_git_blob",
        lambda _repository, _commit, path: {
            "Hegel Machine/rust/m3_closure_enumerator/Cargo.lock": lock_payload,
            qualification.BOOTSTRAP_RECORD_PATH: bootstrap_payload,
        }[path],
    )
    binary = (
        tmp_path
        / "rust/m3_closure_enumerator/target/m3_qualification"
        / commit
        / "hegel-m3-closure-enumerator"
    )
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"bound-rust-enumerator")
    binary.chmod(0o755)
    binary_digest = qualification.hashlib.sha256(binary.read_bytes()).digest()
    python_binary_digest = b"\x77" * 32
    python_image = "python@sha256:" + "55" * 32
    rust_image = "rust@sha256:" + "66" * 32

    def source_rows(count: int, prefix: str):
        return tuple(
            {
                "path_alias_id_digest": id_digest_v1(f"test-path:{prefix}:{index}"),
                "raw_path_bytes": f"{prefix}/{index:02d}".encode(),
                "git_blob_algorithm_id": 1,
                "git_blob_digest": bytes([index + 1]) * 20,
                "file_mode": 0o100644,
                "byte_length": index + 1,
            }
            for index in range(count)
        )

    python_sources = source_rows(len(qualification.PYTHON_SOURCE_PATHS), "python")
    rust_sources = source_rows(len(qualification.RUST_SOURCE_PATHS), "rust")
    python_source_root = qualification.candidate_record_tree_root(
        "SourceFileRecordV1", python_sources
    )
    rust_source_root = qualification.candidate_record_tree_root(
        "SourceFileRecordV1", rust_sources
    )
    python_lock = ()
    rust_lock = ()
    python_lock_root = qualification.candidate_record_tree_root(
        "DependencyLockRecordV1", python_lock
    )
    rust_lock_root = qualification.candidate_record_tree_root(
        "DependencyLockRecordV1", rust_lock
    )
    python_environment = qualification._environment_fields(
        python_image, "python", python_lock_root
    )
    rust_environment = qualification._environment_fields(
        rust_image, "rust", rust_lock_root
    )
    python_environment_root = qualification.candidate_content_root(
        "ExecutionEnvironmentSpecV1", python_environment
    )
    rust_environment_root = qualification.candidate_content_root(
        "ExecutionEnvironmentSpecV1", rust_environment
    )
    commit_wire = qualification.git_sha1_commit_id(bytes.fromhex(commit))
    python_binding = qualification._implementation_binding_fields(
        implementation_id=1,
        source_root=python_source_root,
        binary_digest=python_binary_digest,
        environment_root=python_environment_root,
        version_digest=b"\x88" * 32,
        dependency_lock_root=python_lock_root,
        entrypoint=qualification.PYTHON_ENTRYPOINT_ID,
        golden_root=golden_root,
        commit_wire=commit_wire,
    )
    rust_binding = qualification._implementation_binding_fields(
        implementation_id=2,
        source_root=rust_source_root,
        binary_digest=binary_digest,
        environment_root=rust_environment_root,
        version_digest=b"\x88" * 32,
        dependency_lock_root=rust_lock_root,
        entrypoint=qualification.RUST_ENTRYPOINT_ID,
        golden_root=golden_root,
        commit_wire=commit_wire,
    )
    binding_roots = {
        "python_implementation_binding_root": qualification.candidate_content_root(
            "ImplementationBindingV1", python_binding
        ),
        "rust_implementation_binding_root": qualification.candidate_content_root(
            "ImplementationBindingV1", rust_binding
        ),
    }
    python_receipt = _implementation_receipt(
        1, binding_roots["python_implementation_binding_root"]
    )
    python_receipt.update(
        {
            "source_root": python_source_root.hex(),
            "dependency_lock_root": python_lock_root.hex(),
            "execution_environment_spec_root": python_environment_root.hex(),
            "binary_digest": python_binary_digest.hex(),
        }
    )
    rust_receipt = _implementation_receipt(
        2, binding_roots["rust_implementation_binding_root"]
    )
    rust_receipt.update(
        {
            "source_root": rust_source_root.hex(),
            "dependency_lock_root": rust_lock_root.hex(),
            "execution_environment_spec_root": rust_environment_root.hex(),
            "binary_digest": binary_digest.hex(),
            "dependency_snapshot_root_or_null": dependency_snapshot_root.hex(),
            "dependency_snapshot_file_count": dependency_snapshot_file_count,
        }
    )
    daemon_receipt = _daemon_receipt()
    daemon_binding = qualification.local_docker_daemon_receipt_binding_v1(
        daemon_receipt
    )
    receipt = qualification._build_receipt(
        basis_commit=commit,
        golden=golden,
        golden_root=golden_root,
        python_fields=python_receipt,
        rust_fields=rust_receipt,
        runtime_seccomp_digest=b"s" * 32,
        build_seccomp_digest=b"b" * 32,
        docker_daemon_receipt_binding=daemon_binding,
        cargo_bootstrap_record_digest=bootstrap_digest,
    )
    receipt_root = bytes.fromhex(receipt["receipt_root"])
    inputs = {
        "m3_execution_implementation_bindings_ready": True,
        "m3_execution_implementation_binding_roots": binding_roots,
        "python_m3_source_paths": qualification.PYTHON_SOURCE_PATHS,
        "rust_m3_source_paths": qualification.RUST_SOURCE_PATHS,
        "python_m3_source_root": python_source_root,
        "rust_m3_source_root": rust_source_root,
        "rust_m3_enumerator_binary_path": str(binary),
        "rust_m3_enumerator_binary_sha256": binary_digest,
        "python_m3_enumerator_binary_sha256": python_binary_digest,
        "python_m3_enumerator_image_ref": python_image,
        "rust_m3_enumerator_image_ref": rust_image,
        "python_image_ref": python_image,
        "rust_image_ref": rust_image,
        "python_m3_entrypoint_id_digest": id_digest_v1(qualification.PYTHON_ENTRYPOINT_ID),
        "rust_m3_entrypoint_id_digest": id_digest_v1(qualification.RUST_ENTRYPOINT_ID),
        "m3_dual_golden_vector_root": golden_root,
        "m3_implementation_qualification_receipt": receipt,
        "m3_implementation_qualification_receipt_root": receipt_root,
        "m3_runtime_seccomp_sha256": b"s" * 32,
        "m3_build_seccomp_sha256": b"b" * 32,
        "m3_local_docker_daemon_identity_receipt": daemon_receipt,
        "m3_local_docker_daemon_receipt_binding": daemon_binding,
        "m3_runtime_docker_policy_id": qualification.RUNTIME_DOCKER_POLICY_ID,
        "m3_rust_build_docker_policy_id": qualification.RUST_BUILD_DOCKER_POLICY_ID,
        "rust_m3_dependency_snapshot_root": dependency_snapshot_root,
        "rust_m3_dependency_snapshot_file_count": dependency_snapshot_file_count,
        "m3_cargo_offline_bootstrap_record_sha256": bootstrap_digest,
    }
    basis = SimpleNamespace(
        basis_commit=commit,
        repository_root=tmp_path,
        implementation_inputs=inputs,
        roots={
            **binding_roots,
            "python_m3_source_root": python_source_root,
            "rust_m3_source_root": rust_source_root,
            "python_m3_dependency_lock_root": python_lock_root,
            "rust_m3_dependency_lock_root": rust_lock_root,
            "python_m3_execution_environment_root": python_environment_root,
            "rust_m3_execution_environment_root": rust_environment_root,
            "rust_m3_dependency_snapshot_root": dependency_snapshot_root,
            "m3_cargo_offline_bootstrap_record_sha256": bootstrap_digest,
            "m3_local_docker_daemon_receipt_binding": daemon_binding,
            "m3_implementation_qualification_receipt_root": receipt_root,
        },
        m3_candidate_static_fields=binding_roots,
        objects={
            "python_m3_execution_environment": python_environment,
            "rust_m3_execution_environment": rust_environment,
            "python_m3_implementation_binding": python_binding,
            "rust_m3_implementation_binding": rust_binding,
        },
        record_sets={
            "python_m3_implementation_sources": python_sources,
            "rust_m3_implementation_sources": rust_sources,
            "python_m3_dependency_lock": python_lock,
            "rust_m3_dependency_lock": rust_lock,
        },
    )
    qualification.validate_m3_execution_implementation_bindings_v1(
        basis, live_python_probe=False
    )
    inputs["rust_m3_enumerator_binary_path"] = "/tmp/formal-static-replayer"
    with pytest.raises(qualification.M3ImplementationQualificationError) as captured:
        qualification.validate_m3_execution_implementation_bindings_v1(
            basis, live_python_probe=False
        )
    assert captured.value.code == qualification.FAIL_BINDING
