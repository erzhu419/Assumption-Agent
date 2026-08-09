from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import threading

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools/phase3_shrink3_dual_complete_diagnostic_v1.py"
SPEC = importlib.util.spec_from_file_location("shrink3_dual_complete_tool", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def _assert_hardened(command: list[str]) -> None:
    assert command[:3] == [
        "/usr/bin/docker",
        "--host=unix:///var/run/docker.sock",
        "run",
    ]
    assert "--pull=never" in command
    assert "--network=none" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "--read-only" in command
    assert "--pids-limit=64" in command
    assert "--memory=512m" in command
    assert "--memory-swap=512m" in command
    assert "--ulimit=nofile=128:128" in command


def _host_receipt() -> dict[str, object]:
    return {
        "schema_version": "hegel-m3-shrink3-dual-diagnostic-validation-receipt/1",
        "claim_level": tool.CLAIM_LEVEL,
        "qualification_level": "DIAGNOSTIC_ONLY",
        "diagnostic_only": True,
        "authoritative_claim_allowed": False,
        "execution_state": "NOT_RUN",
        "formal_roots_generated": False,
        "formal_roots": None,
        "formal_state_transition_allowed": False,
        "dual_reports_equal": True,
        "dual_archive_bytes_equal": True,
        "host_strict_archive_replay_verified": True,
        "host_target_free_isolation_verified": True,
        "host_target_or_split_modules_loaded": False,
        "independence_scope": (
            "INDEPENDENT_OF_ENDPOINT_REPORTED_WITNESS_NOT_A_THIRD_IMPLEMENTATION"
        ),
        "typed_language_boundary_independently_derived": True,
        "archive_prefix_exact": True,
        "program_indices_verified": True,
        "program_binding_roots_verified": True,
        "binary_operator_registry_verified": True,
        "removed_binary_operator_absent_from_archive": True,
        "operator_id_compaction_performed": False,
        "automatic_operator_migration_performed": False,
        "chunk_framing_and_blob_hashes_verified": True,
        "bucket_accounting_verified": True,
        "canonical_program_count": 50_000,
        "raw_operator_application_count_scope": (
            "THROUGH_FULLY_CLOSED_BOUNDARY_BUCKET"
        ),
        "witness_adjacency_verified": True,
        "witness_closed_bucket_rank_verified": True,
        "post_witness_traversal_buckets_untouched": True,
        "closure_status": "DSL_TOO_LARGE",
        "target_roles_evaluated": False,
        "split_material_accessed": False,
        "secrets_accessed": False,
    }


def test_enumerator_build_profile_is_literal_and_nonformal() -> None:
    receipt = tool._load_build_profile(PROJECT_ROOT)
    payload = json.loads(
        (
            PROJECT_ROOT
            / "config/phase3_shrink3_enumerator_offline_build_profile_v1.json"
        ).read_bytes()
    )
    assert receipt["profile_id"] == "hegel-shrink3-enumerator-offline-build-v1"
    assert payload["image"] == tool.RUST_IMAGE
    assert payload["network"] == "none"
    assert payload["pull_policy"] == "never"
    assert payload["root_filesystem_read_only"] is True
    assert payload["cargo_dependency_seed_file_set_root_required"] is True
    assert payload["target_or_split_inputs_allowed"] is False
    assert payload["seed_key_signature_or_formal_root_access_allowed"] is False


def test_build_command_is_offline_pinned_and_uses_fresh_target_rw() -> None:
    command = tool.rust_build_command(
        Path("/snapshot"), Path("/cargo-registry"), "fresh-target", 8
    )
    _assert_hardened(command)
    assert tool.RUST_IMAGE in command
    assert "CARGO_NET_OFFLINE=true" in command
    assert "CARGO_BUILD_JOBS=8" in command
    assert "CARGO_HOME=/tmp/cargo-home" in command
    assert "/cargo-registry/cache:/cargo-seed/cache:ro" in command
    assert "/cargo-registry/index:/cargo-seed/index:ro" in command
    assert not any("/cargo-seed/registry" in argument for argument in command)
    assert "/snapshot/rust:/workspace/rust:ro" in command
    assert "fresh-target:/cargo-target:rw" in command
    assert "/workspace/rust/m3_closure_enumerator_shrink3" in command
    assert command[-5:] == [
        "--release",
        "--locked",
        "--offline",
        "--bin",
        tool.RUST_BINARY,
    ]
    assert any(
        "test ! -e /tmp/cargo-home/registry/src" in argument
        for argument in command
    )


def test_dependency_seed_matches_lock_and_excludes_preunpacked_src() -> None:
    receipt = tool.verify_cargo_dependency_seed_v1(
        tool.DEFAULT_CARGO_REGISTRY,
        PROJECT_ROOT / "rust/m3_closure_enumerator_shrink3/Cargo.lock",
    )
    assert receipt["locked_registry_package_count"] == 21
    assert receipt["verified_crate_count"] == 21
    assert receipt["src_subtree_included"] is False
    assert receipt["fresh_tmpfs_cargo_home"] is True
    assert str(receipt["file_set_root"]).startswith("sha256:")
    assert all(
        not str(row["path"]).startswith("src/")
        for row in receipt["files"]
    )


def test_dependency_snapshot_is_reverified_and_mounts_only_two_subtrees(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "snapshot"
    receipt = tool.create_verified_cargo_seed_snapshot_v1(
        tool.DEFAULT_CARGO_REGISTRY,
        destination,
        PROJECT_ROOT / "rust/m3_closure_enumerator_shrink3/Cargo.lock",
    )
    try:
        assert receipt["snapshot_reverified_after_copy"] is True
        assert receipt["docker_bind_mounts_exact"] == ["cache", "index"]
        assert sorted(path.name for path in destination.iterdir()) == [
            "cache",
            "index",
        ]
        assert not (destination / "src").exists()
    finally:
        tool._restore_owner_writable_tree(destination)


@pytest.mark.parametrize("implementation", ["python", "rust"])
def test_endpoint_commands_are_parallel_ready_and_target_free(
    implementation: str,
) -> None:
    if implementation == "python":
        command = tool.python_endpoint_command(Path("/snapshot"), Path("/output"))
        assert tool.PYTHON_IMAGE in command
        assert tool.PYTHON_ENTRYPOINT in command
        assert "/snapshot:/workspace:ro" in command
        assert "/output:/output:rw" in command
        assert "/output/result" in command
    else:
        command = tool.rust_endpoint_command(
            Path("/snapshot"), Path("/output"), "fresh-target"
        )
        assert tool.RUST_IMAGE in command
        assert tool.RUST_BINARY_PATH in command
        assert "fresh-target:/cargo-target:ro" in command
        assert "/snapshot:/workspace:ro" in command
        assert "/output:/output:rw" in command
        assert "/output/result" in command
    _assert_hardened(command)
    assert "--enumerate-diagnostic" in command
    assert tool.CHILD_DSL_SPEC_ROOT in command
    assert tool.OPERATOR_SEMANTICS_ROOT in command
    assert tool.IDENTIFIER_REGISTRY_ROOT in command
    joined = " ".join(command)
    assert "--target" not in joined
    assert "--split" not in joined
    assert "--seed" not in joined
    assert "--signature" not in joined


def test_host_replay_runs_after_read_only_endpoint_archives() -> None:
    command = tool.host_replay_command(
        Path("/snapshot"), Path("/python-output"), Path("/rust-output")
    )
    _assert_hardened(command)
    assert tool.PYTHON_IMAGE in command
    assert tool.HOST_ENTRYPOINT in command
    assert "/snapshot:/workspace:ro" in command
    assert "/python-output:/evidence/python:ro" in command
    assert "/rust-output:/evidence/rust:ro" in command
    assert "/evidence/python" in command
    assert "/evidence/rust" in command
    assert "--validate-dual" in command


def test_endpoint_runner_really_uses_two_parallel_workers() -> None:
    barrier = threading.Barrier(2)
    thread_ids: set[int] = set()
    lock = threading.Lock()

    def runner(
        implementation: str, command: list[str], timeout: int
    ) -> tool.EndpointResult:
        assert command == [implementation]
        assert timeout == 123
        with lock:
            thread_ids.add(threading.get_ident())
        barrier.wait(timeout=2)
        payload = json.dumps({"implementation": implementation}).encode()
        return tool.EndpointResult(
            implementation, payload, {"implementation": implementation}
        )

    result = tool.run_endpoints_parallel(
        {"python": ["python"], "rust": ["rust"]},
        timeout=123,
        runner=runner,
    )
    assert set(result) == {"python", "rust"}
    assert len(thread_ids) == 2


def test_host_receipt_guards_keep_m3_not_run() -> None:
    receipt = _host_receipt()
    tool.validate_host_receipt(receipt)
    receipt["formal_roots"] = {"forbidden": "root"}
    with pytest.raises(tool.SupervisorError) as raised:
        tool.validate_host_receipt(receipt)
    assert raised.value.code == tool.FAIL_AUTHORITY


def test_archive_surface_is_commit_bound_and_excludes_target_inputs() -> None:
    assert tool.SUPERVISOR_PATH in tool.ARCHIVE_PATHS
    assert tool.HARDENED_SUPERVISOR_PATH in tool.ARCHIVE_PATHS
    assert tool.BUILD_PROFILE_PATH in tool.ARCHIVE_PATHS
    assert "Hegel Machine/rust/m3_closure_enumerator_shrink3" in tool.ARCHIVE_PATHS
    assert (
        "Hegel Machine/src/hegel_machine/phase3_m3_shrink3_dual_diagnostic_entrypoint_v1.py"
        in tool.ARCHIVE_PATHS
    )
    assert not any(
        fragment in path
        for path in tool.PYTHON_SOURCE_PATHS
        for fragment in ("_target", "_split_", "_seed", "_role", "_evaluator")
    )


def test_retained_output_must_be_external_and_new(tmp_path: Path) -> None:
    candidate = tmp_path / "new-evidence"
    assert tool._validate_external_output(candidate) == candidate
    candidate.mkdir()
    with pytest.raises(tool.SupervisorError) as raised:
        tool._validate_external_output(candidate)
    assert raised.value.code == tool.FAIL_ARGUMENT


def test_endpoint_mount_permission_is_not_narrowed_by_umask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    previous = os.umask(0o077)
    try:
        path = tool._fresh_endpoint_mount(tmp_path, "endpoint")
    finally:
        os.umask(previous)
    assert path.stat().st_mode & 0o777 == 0o777


def test_fail_closed_output_is_retained_and_requested_path_reusable(
    tmp_path: Path,
) -> None:
    requested = tmp_path / "diagnostic"
    requested.mkdir()
    (requested / "partial.log").write_text("partial", encoding="utf-8")
    retained = tool._preserve_failed_output(
        requested, tool.SupervisorError(tool.FAIL_ENDPOINT, "endpoint failed")
    )
    assert retained is not None and retained.is_dir()
    assert not requested.exists()
    receipt = json.loads(
        (retained / "failure_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "FAIL_CLOSED"
    assert receipt["failure_code"] == tool.FAIL_ENDPOINT
    assert receipt["execution_state"] == "NOT_RUN"
    assert receipt["formal_roots"] is None
    assert (retained / "partial.log").read_text(encoding="utf-8") == "partial"


def test_deterministic_archive_uses_retained_outer_mount_layout(
    tmp_path: Path,
) -> None:
    for role in ("python", "rust"):
        result = tmp_path / role / "result"
        result.mkdir(parents=True)
        (result / "report.json").write_text(role, encoding="utf-8")
    digest = tool._deterministic_output_archive(
        tmp_path, tmp_path, b'{"host":true}\n'
    )
    payload = (tmp_path / "diagnostic_outputs.tar").read_bytes()
    assert len(digest) == 64
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        assert archive.getnames() == [
            "python/result/report.json",
            "rust/result/report.json",
            "host_replay_receipt.json",
        ]
