from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess

import pytest

import hegel_machine.phase3_m25_formal_static_basis_v1 as formal_static

from hegel_machine.phase3_local_runtime_v1 import (
    LinuxLocalTemporaryDirectoryV1,
    LocalDockerControlPlaneV1,
    prepare_local_docker_control_plane_v1,
)

from hegel_machine.phase3_m25_formal_static_basis_v1 import (
    DEFAULT_RUST_BINARY,
    FAIL_DUAL_RECEIPT,
    FAIL_RUST_REPLAY_POLICY,
    FormalStaticBasisError,
    GATE19_ROOT_NAMES,
    build_formal_static_basis_v1,
    build_identifier_registry_rows_v1,
    build_operator_semantics_rows_v1,
    build_python_static_replay_receipt_v1,
    run_rust_static_replay_receipt_v1,
    validate_dual_static_replay_receipts_v1,
)
from hegel_machine.phase3_m25_wire_v1 import candidate_record_tree_root


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent

COMMITTED_BASIS_PATHS = (
    "Hegel Machine/config/phase3_container_actor_profile_v1.json",
    "Hegel Machine/config/phase3_internal_actor_seccomp_v1.json",
    "Hegel Machine/artifacts/phase3_dual_strict_capacity_replay_v1.json",
    "Hegel Machine/artifacts/phase3_shrink1_dual_capacity_replay_v1.json",
    "Hegel Machine/docs/Hegel_Machine_Phase3A_M25_Bit_Exact_Wire_Completion_Amendment.md",
    "Hegel Machine/docs/Hegel_Machine_Phase3A_M25_Exact_Wire_Errata_Resolution.md",
    "Hegel Machine/docs/Hegel_Machine_Phase3A_M25_Implementation_Closure_Addendum_v1.md",
    "Hegel Machine/docs/Hegel_Machine_Phase3A_M25_Formal_Static_Basis_Engineering_Freeze_v1.md",
    "Hegel Machine/docs/Hegel_Machine_Phase3_Shrink_Step1_Freeze_Decisions.md",
    "Hegel Machine/src/hegel_machine/phase3_dsl_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_dsl_core_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_shrink1_core_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_formal_static_basis_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_rows_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_wire_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_shrink1_registry_v1.py",
    "Hegel Machine/src/hegel_machine/strict_ast_shrink1_v1.py",
    "Hegel Machine/src/hegel_machine/strict_ast_v1.py",
    "Hegel Machine/src/hegel_machine/strict_cbor_v1.py",
    "Hegel Machine/rust/formal_bridge_m25/Cargo.lock",
    "Hegel Machine/rust/formal_bridge_m25/Cargo.toml",
    "Hegel Machine/rust/formal_bridge_m25/src/lib.rs",
    "Hegel Machine/rust/formal_bridge_m25/src/main.rs",
)


def _code(action, *args, **kwargs) -> str:
    with pytest.raises(FormalStaticBasisError) as captured:
        action(*args, **kwargs)
    return captured.value.code


def _two_commit_git_fixture(repository: Path) -> tuple[str, str]:
    repository.mkdir()
    subprocess.run(["/usr/bin/git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["/usr/bin/git", "config", "user.email", "git-read@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["/usr/bin/git", "config", "user.name", "formal-git-read-test"],
        cwd=repository,
        check=True,
    )
    payload = repository / "payload.txt"
    payload.write_bytes(b"original\n")
    subprocess.run(["/usr/bin/git", "add", "payload.txt"], cwd=repository, check=True)
    subprocess.run(["/usr/bin/git", "commit", "-qm", "original"], cwd=repository, check=True)
    original = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    payload.write_bytes(b"replacement\n")
    subprocess.run(["/usr/bin/git", "commit", "-qam", "replacement"], cwd=repository, check=True)
    replacement = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return original, replacement


def test_formal_git_read_ignores_real_replace_ref(tmp_path: Path) -> None:
    repository = tmp_path / "replace-repository"
    original, replacement = _two_commit_git_fixture(repository)
    subprocess.run(
        ["/usr/bin/git", "replace", original, replacement],
        cwd=repository,
        check=True,
    )
    ambient = subprocess.run(
        ["/usr/bin/git", "show", f"{original}:payload.txt"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    assert ambient == b"replacement\n"
    assert formal_static._git_blob(repository, original, "payload.txt") == b"original\n"


def test_formal_git_read_uses_absolute_binary_and_noninheriting_environment(
    tmp_path: Path, monkeypatch,
) -> None:
    repository = tmp_path / "hostile-environment-repository"
    original, _replacement = _two_commit_git_fixture(repository)
    hostile = {
        "PATH": "/tmp/hostile-bin",
        "HOME": "/tmp/hostile-home",
        "GIT_DIR": "/tmp/hostile-git-dir",
        "GIT_WORK_TREE": "/tmp/hostile-work-tree",
        "GIT_OBJECT_DIRECTORY": "/tmp/hostile-objects",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": "/tmp/hostile-alternates",
        "GIT_REPLACE_REF_BASE": "refs/hostile-replace/",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.pager",
        "GIT_CONFIG_VALUE_0": "/tmp/hostile-pager",
        "GIT_NO_REPLACE_OBJECTS": "0",
        "GIT_NO_LAZY_FETCH": "0",
        "GIT_PROTOCOL_FROM_USER": "1",
        "GIT_SSH_COMMAND": "/tmp/hostile-ssh",
    }
    for key, value in hostile.items():
        monkeypatch.setenv(key, value)
    original_run = formal_static.subprocess.run
    observed: dict[str, object] = {}

    def recording_run(command, **kwargs):
        observed["command"] = tuple(command)
        observed["environment"] = dict(kwargs.get("env", {}))
        return original_run(command, **kwargs)

    monkeypatch.setattr(formal_static.subprocess, "run", recording_run)
    assert formal_static._git_blob(repository, original, "payload.txt") == b"original\n"
    command = observed["command"]
    environment = observed["environment"]
    assert isinstance(command, tuple) and command[0] == "/usr/bin/git"
    assert environment == formal_static.formal_git_environment_v1()
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert environment["GIT_NO_LAZY_FETCH"] == "1"
    assert environment["GIT_PROTOCOL_FROM_USER"] == "0"
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert environment["GIT_CONFIG_SYSTEM"] == "/dev/null"
    assert not (
        set(hostile)
        - {
            "PATH", "HOME", "GIT_NO_REPLACE_OBJECTS", "GIT_NO_LAZY_FETCH",
            "GIT_PROTOCOL_FROM_USER", "GIT_SSH_COMMAND",
        }
    ) & set(environment)


@pytest.fixture(scope="module")
def committed_basis(tmp_path_factory: pytest.TempPathFactory):
    repository = tmp_path_factory.mktemp("formal-static-basis-repo")
    for relative in COMMITTED_BASIS_PATHS:
        source = REPOSITORY_ROOT / relative
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "static-basis@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "static-basis-test"],
        cwd=repository,
        check=True,
    )
    subprocess.run(["git", "add", "--", "Hegel Machine"], cwd=repository, check=True)
    environment = dict(os.environ)
    environment.update(
        {
            "GIT_AUTHOR_DATE": "2026-08-02T00:00:00+00:00",
            "GIT_COMMITTER_DATE": "2026-08-02T00:00:00+00:00",
        }
    )
    subprocess.run(
        ["git", "commit", "-qm", "formal static fixture"],
        cwd=repository,
        env=environment,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return build_formal_static_basis_v1(commit, repository_root=repository)


@pytest.fixture(scope="module")
def dual_receipts(committed_basis):
    if not DEFAULT_RUST_BINARY.is_file():
        pytest.skip("Rust bridge binary absent")
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker unavailable")
    probe = subprocess.run(
        [docker, "info"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=20,
    )
    if probe.returncode != 0:
        pytest.skip("Docker daemon unavailable")
    with LinuxLocalTemporaryDirectoryV1(
        prefix="hegel-static-test-",
        repository_root=REPOSITORY_ROOT,
    ) as runtime:
        control_plane = prepare_local_docker_control_plane_v1(
            Path(runtime),
            repository_root=REPOSITORY_ROOT,
        )
        return (
            build_python_static_replay_receipt_v1(committed_basis),
            run_rust_static_replay_receipt_v1(
                committed_basis,
                control_plane=control_plane,
                daemon_receipt_binding=b"d" * 32,
            ),
        )


def test_identifier_registry_is_complete_sparse_and_stable() -> None:
    parent, _ = build_identifier_registry_rows_v1(parent=True)
    child, preimages = build_identifier_registry_rows_v1(parent=False)
    assert len(parent) == len(child) == 55
    assert len(preimages) == 55
    assert [(row["numeric_id"], row["entry_state_id"]) for row in child if row["registry_kind_id"] == 8] == [
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 2),
        (5, 1),
    ]
    assert all(row["entry_state_id"] == 1 for row in parent)
    assert [row["numeric_id"] for row in child if row["registry_kind_id"] == 10] == list(range(19))
    assert candidate_record_tree_root("IdentifierRegistryEntryV1", child) != candidate_record_tree_root(
        "IdentifierRegistryEntryV1", parent
    )


def test_operator_rows_preserve_dispatcher_gap_and_sparse_maps() -> None:
    child, preimages = build_operator_semantics_rows_v1(
        executable_semantics_root=bytes.fromhex("42" * 32), parent=False
    )
    assert len(child) == 28
    assert len(preimages) == 56
    assert [row["operator_id"] for row in child if row["operator_class_id"] == 1] == [
        0,
        1,
        2,
        4,
        5,
    ]
    assert [(row["operator_id"], row["admission_state_id"]) for row in child if row["operator_class_id"] == 6] == [
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 2),
        (5, 1),
    ]


def test_full_basis_has_exact_preimages_and_keeps_m3_implementation_gap(
    committed_basis,
) -> None:
    basis = committed_basis
    assert [entry.root_name for entry in basis.gate19_plan] == list(GATE19_ROOT_NAMES)
    assert len(basis.record_sets["identifier_registry"]) == 55
    assert len(basis.record_sets["operator_semantics"]) == 28
    assert len(basis.record_sets["diagnostic_formal_bridge"]) == 12
    assert all(type(value) is bytes and len(value) == 32 for value in basis.roots.values())
    assert all(value for value in basis.ordinary_digest_preimages.values())
    assert all(value for value in basis.diagnostic_preimages.values())
    assert "python_implementation_binding_root" not in basis.m3_candidate_static_fields
    assert "rust_implementation_binding_root" not in basis.m3_candidate_static_fields
    assert basis.implementation_inputs["m3_execution_implementation_bindings_ready"] is False
    assert basis.implementation_inputs["m3_execution_implementation_binding_roots"] is None
    assert basis.blocking_gaps == ("M3_EXECUTION_IMPLEMENTATION_BINDINGS_NOT_READY",)
    assert "python_static_replay_implementation_binding_root" in basis.roots
    assert "rust_static_replay_implementation_binding_root" in basis.roots
    assert basis.roots["approval_manifest_root"] == basis.roots[
        "normative_approval_manifest_root"
    ]
    assert basis.objects["split_spec_freeze"]["seed_state_id"] == 1
    assert basis.preseed_manifest_static_fields["SeedContinuityManifestV1"][
        "parent_seed_commitment_manifest_root_or_null"
    ] is None
    assert basis.preseed_manifest_required_dynamic_fields[
        "SeedContinuityManifestV1"
    ] == (
        "current_seed_commitment_manifest_root",
        "parent_manifest_absence_attestation_root",
        "hidden_access_ledger_genesis_root",
        "custodian_binding_core_root",
        "instantiated_at_unix_seconds",
    )
    shrink_static = basis.preseed_manifest_static_fields[
        "DslShrinkTransitionFormalV1"
    ]
    assert shrink_static["parent_execution_evidence_root"] == basis.roots[
        "parent_execution_evidence_root"
    ]
    assert shrink_static["shrink1_subset_replay_root"] == basis.roots[
        "shrink1_subset_replay_root"
    ]
    split_static = basis.preseed_manifest_static_fields["SplitBindingManifestV1"]
    assert len(split_static["split_algorithm_id_digest"]) == 32
    for role in ("OUTSIDE_TARGET", "IN_LANGUAGE_NULL"):
        role_static = basis.preseed_manifest_static_fields[
            f"DslRoleBindingManifestV1/{role}"
        ]
        assert len(role_static["semantic_spec_diagnostic_id_digest"]) == 32
        assert len(role_static["universe_diagnostic_id_digest"]) == 32
        assert len(role_static["truth_diagnostic_id_digest"]) == 32


def test_python_receipt_replays_every_exact_preimage(committed_basis) -> None:
    basis = committed_basis
    python_receipt = build_python_static_replay_receipt_v1(basis)
    assert python_receipt["container_image_ref_or_null"] is None
    assert python_receipt["network_mode_none"] is False
    assert python_receipt["generator_performs_network_io"] is False
    assert [entry["root_name"] for entry in python_receipt["entries"]] == list(
        GATE19_ROOT_NAMES
    )


def test_live_rust_offline_container_receipt_matches_python(
    committed_basis, dual_receipts
) -> None:
    basis = committed_basis
    python_receipt, rust_receipt = dual_receipts
    roots = validate_dual_static_replay_receipts_v1(
        basis, python_receipt, rust_receipt
    )
    assert tuple(roots) == GATE19_ROOT_NAMES
    assert rust_receipt["network_mode_none"] is True
    assert rust_receipt["pull_policy_never"] is True
    assert all("--pull=never" in row["normalized_command"] for row in rust_receipt["executions"])
    assert all("--network=none" in row["normalized_command"] for row in rust_receipt["executions"])

    tampered = deepcopy(rust_receipt)
    tampered["executions"][0]["stdout_hex"] = "7b7d0a"
    # Deliberately leave the receipt digest stale: either the envelope or the
    # exact public stdout binding must make the forged receipt fail closed.
    assert _code(
        validate_dual_static_replay_receipts_v1, basis, python_receipt, tampered
    ) == FAIL_DUAL_RECEIPT


@pytest.mark.skipif(not DEFAULT_RUST_BINARY.is_file(), reason="Rust bridge binary absent")
def test_rust_replay_rejects_binary_drift(committed_basis, tmp_path: Path) -> None:
    drifted = tmp_path / "formal-bridge-drifted"
    drifted.write_bytes(DEFAULT_RUST_BINARY.read_bytes() + b"drift")
    drifted.chmod(0o755)
    control_plane = LocalDockerControlPlaneV1(
        executable=Path("/usr/bin/docker"),
        socket_path=Path("/var/run/docker.sock"),
        config_directory=tmp_path / "docker-config",
        environment={},
        binding={},
    )
    assert _code(
        run_rust_static_replay_receipt_v1,
        committed_basis,
        control_plane=control_plane,
        daemon_receipt_binding=b"d" * 32,
        rust_binary=drifted,
    ) == FAIL_RUST_REPLAY_POLICY
