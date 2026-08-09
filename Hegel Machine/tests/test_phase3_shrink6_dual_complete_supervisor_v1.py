from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import threading

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools/phase3_shrink6_dual_complete_diagnostic_v1.py"
SPEC = importlib.util.spec_from_file_location("shrink6_dual_complete_tool", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def _assert_hardened(command: list[str], *, expected_pids: int = 64) -> None:
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
    pids_arguments = [
        argument for argument in command if argument.startswith("--pids-limit=")
    ]
    assert pids_arguments == [f"--pids-limit={expected_pids}"]
    assert "--memory=512m" in command
    assert "--memory-swap=512m" in command
    assert "--ulimit=nofile=128:128" in command


def _host_receipt() -> dict[str, object]:
    # Synthetic schema/guard fixture only; no value below is an observed run.
    return {
        "schema_version": "hegel-m3-shrink6-dual-diagnostic-validation-receipt/1",
        "claim_level": tool.CLAIM_LEVEL,
        "qualification_level": "DIAGNOSTIC_ONLY",
        "profile_id": "hegel-m3-shrink6-dual-diagnostic-profile-v1",
        "binding_profile_id": "NON_FORMAL_SYNTHETIC_CHILD_BINDINGS_V1",
        "diagnostic_only": True,
        "authoritative_claim_allowed": False,
        "execution_state": "NOT_RUN",
        "formal_roots_generated": False,
        "formal_roots": None,
        "formal_state_transition_allowed": False,
        **tool.STRICT_QUALIFICATION_BINDING,
        "maximum_ast_depth": 3,
        "maximum_ast_node_count": 6,
        "maximum_top_level_clauses": 2,
        "formal_bucket_count": 120,
        "and3_generator_attempts_allowed": False,
        "and3_raw_operator_application_count": 0,
        "dual_reports_equal": True,
        "dual_archive_bytes_equal": True,
        "host_strict_archive_replay_verified": True,
        "host_loaded_hegel_modules": tool.EXPECTED_HOST_MODULES,
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
        "canonical_program_count": tool.EXPECTED_CANONICAL_PROGRAM_COUNT,
        "raw_operator_application_count": (
            tool.EXPECTED_RAW_OPERATOR_APPLICATION_COUNT
        ),
        "raw_operator_application_count_scope": (
            "THROUGH_FULLY_CLOSED_BOUNDARY_BUCKET"
        ),
        "witness_adjacency_verified": True,
        "witness_closed_bucket_rank_verified": True,
        "post_witness_traversal_buckets_untouched": True,
        "residual_out_of_budget_canonical_programs": (
            tool.EXPECTED_RESIDUAL_OUT_OF_BUDGET_COUNT
        ),
        "closure_status": "DSL_TOO_LARGE",
        "canonical_program_archive_root": "01" * 32,
        "program_chunk_manifest_root": "02" * 32,
        "bucket_accounting_root": "03" * 32,
        "first_out_of_budget_program_hash": tool.EXPECTED_WITNESS_HASH,
        "first_out_of_budget_program_cbor_hex": tool.EXPECTED_WITNESS_CBOR_HEX,
        "first_out_of_budget_program_ordinal": (
            tool.EXPECTED_FIRST_OUT_OF_BUDGET_ORDINAL
        ),
        "witness_bucket_index": tool.EXPECTED_WITNESS_BUCKET_INDEX,
        "witness_output_sort_id": tool.EXPECTED_WITNESS_OUTPUT_SORT_ID,
        "witness_ast_depth": tool.EXPECTED_WITNESS_AST_DEPTH,
        "witness_ast_node_count": tool.EXPECTED_WITNESS_AST_NODE_COUNT,
        "prefix_preservation_expectation_id": tool.PRESERVATION_EXPECTATION_ID,
        "prefix_preservation_verified": True,
        "preregistered_shrink_order_total_steps": 6,
        "preregistered_shrink_order_consumed_through_step": 6,
        "next_preregistered_shrink_step_or_null": None,
        "terminal_route": tool.DSL_TOO_LARGE_ROUTE,
        "budget_change_authorized": False,
        "additional_shrink_authorized": False,
        "new_dsl_version_authorized": False,
        "python_report_sha256": "05" * 32,
        "rust_report_sha256": "06" * 32,
        "stream_sha256": {
            "canonical_program_records": "07" * 32,
            "program_chunk_manifests": "08" * 32,
            "bucket_accounting_records": "09" * 32,
        },
        "target_roles_evaluated": False,
        "split_material_accessed": False,
        "secrets_accessed": False,
    }


def _complete_host_receipt() -> dict[str, object]:
    receipt = _host_receipt()
    receipt.update(
        {
            "canonical_program_count": 1,
            "raw_operator_application_count": 1,
            "raw_operator_application_count_scope": (
                "THROUGH_FULLY_CLOSED_FRONTIER"
            ),
            "witness_adjacency_verified": None,
            "witness_closed_bucket_rank_verified": None,
            "post_witness_traversal_buckets_untouched": None,
            "residual_out_of_budget_canonical_programs": 0,
            "closure_status": "COMPLETE",
            "first_out_of_budget_program_hash": None,
            "first_out_of_budget_program_cbor_hex": None,
            "first_out_of_budget_program_ordinal": None,
            "witness_bucket_index": None,
            "witness_output_sort_id": None,
            "witness_ast_depth": None,
            "witness_ast_node_count": None,
            "prefix_preservation_verified": None,
            "terminal_route": tool.COMPLETE_ROUTE,
        }
    )
    return receipt


def test_enumerator_build_profile_is_literal_and_nonformal() -> None:
    receipt = tool._load_build_profile(PROJECT_ROOT)
    payload = json.loads(
        (
            PROJECT_ROOT
            / "config/phase3_shrink6_enumerator_offline_build_profile_v1.json"
        ).read_bytes()
    )
    assert receipt["profile_id"] == "hegel-shrink6-enumerator-offline-build-v1"
    assert payload["image"] == tool.RUST_IMAGE
    assert payload["network"] == "none"
    assert payload["pull_policy"] == "never"
    assert payload["root_filesystem_read_only"] is True
    assert payload["pids_limit"] == tool.BUILD_PIDS_LIMIT == 256
    assert payload["pids_limit_scope"] == "build-only"
    assert payload["runtime_actor_pids_limit"] == (
        tool.RUNTIME_ACTOR_PIDS_LIMIT
    ) == 64
    assert payload["source_y_archive_only"] is True
    assert payload["strict_qualification_source_commit"] == tool.SOURCE_W_COMMIT
    assert payload["strict_qualification_evidence_commit"] == tool.EVIDENCE_X_COMMIT
    assert payload["strict_qualification_artifact_sha256"] == (
        tool.EVIDENCE_X_ARTIFACT_SHA256
    )
    assert payload["strict_qualification_diagnostic_report_hash"] == (
        tool.EVIDENCE_X_DIAGNOSTIC_REPORT_HASH
    )
    assert payload["cargo_dependency_seed_file_set_root_required"] is True
    assert payload["cargo_dependency_seed_pre_post_identity_required"] is True
    assert payload["result_evidence_label"] == "Evidence Z"
    assert payload["preregistered_prefix_preservation_expectation"] == (
        tool.PRESERVATION_EXPECTATION_ID
    )
    assert payload["preregistered_shrink_order_total_steps"] == 6
    assert payload["preregistered_shrink_order_consumed_through_step"] == 6
    assert payload["next_preregistered_shrink_step_or_null"] is None
    assert payload["automatic_budget_change_allowed"] is False
    assert payload["automatic_additional_shrink_allowed"] is False
    assert payload["automatic_dsl_v1_7_allowed"] is False
    assert payload["target_or_split_inputs_allowed"] is False
    assert payload["seed_key_signature_or_formal_root_access_allowed"] is False


def test_source_w_evidence_x_roots_and_topology_are_exact() -> None:
    assert tool.SOURCE_W_COMMIT == "a69bf6d9746e302a07019f122047ac0bc74aa1c1"
    assert tool.EVIDENCE_X_COMMIT == "f9218e28740953c9ac15a2ada70a8616e92c378b"
    assert tool.EVIDENCE_X_ARTIFACT_SHA256 == (
        "d5417639c651ea5d8dfbc224c79b0af56f1eb9d8705ee244f19dc9d95e6f2d08"
    )
    assert tool.EVIDENCE_X_DIAGNOSTIC_REPORT_HASH == (
        "sha256:3d2a6f06daa47b34aa56ae0d318cc818ba211859063d7a6b81271bc6bf1f8287"
    )
    assert tool.CHILD_DSL_SPEC_ROOT == (
        "da5ed2db33a88a0912d5003999f787cc26ba18564876615773a82bb742d9f8ae"
    )
    assert tool.OPERATOR_SEMANTICS_ROOT == (
        "922e48ada22dfa8621a4d516e07ec9aa7dc8fc10c165d1cafc963575aed5ec03"
    )
    assert tool.IDENTIFIER_REGISTRY_ROOT == (
        "64c9415f7759eec140e439030c5a5374851b9024d7d4849b52b995704ba76ff1"
    )
    assert len(tool.PYTHON_SOURCE_PATHS) == 27
    assert len(tool.RUST_SOURCE_DIRS) == 8
    assert len(tool.RUST_REQUIRED_PATHS) == 19
    assert len(tool.EXPECTED_HOST_MODULES) == 24


def test_preregistered_expectation_is_not_an_observed_source_result() -> None:
    assert tool.EXPECTED_CANONICAL_PROGRAM_COUNT == 50_000
    assert tool.EXPECTED_FIRST_OUT_OF_BUDGET_ORDINAL == 50_001
    assert tool.EXPECTED_RAW_OPERATOR_APPLICATION_COUNT == 3_120_719
    assert tool.EXPECTED_RESIDUAL_OUT_OF_BUDGET_COUNT == 2_237
    assert tool.EXPECTED_WITNESS_BUCKET_INDEX == 63
    assert tool.EXPECTED_WITNESS_CBOR_HEX == (
        "820183010384020183000001860003050200818203f5"
    )
    assert tool.EXPECTED_WITNESS_HASH == (
        "31320fc9f8926792aaf1416a4963df46a2300d87db8096f42e574a62272a68ee"
    )
    assert tool.PREREGISTERED_SHRINK_ORDER_TOTAL_STEPS == 6
    assert tool.PREREGISTERED_SHRINK_ORDER_CONSUMED_THROUGH_STEP == 6
    assert tool.DSL_TOO_LARGE_ROUTE == (
        "HALT_NO_PREREGISTERED_SHRINK_REMAINING_NEEDS_NEW_NORMATIVE_DECISION"
    )


def test_build_command_is_offline_pinned_and_uses_fresh_target_rw() -> None:
    command = tool.rust_build_command(
        Path("/snapshot"), Path("/cargo-registry"), "fresh-target", 8
    )
    _assert_hardened(command, expected_pids=tool.BUILD_PIDS_LIMIT)
    assert tool.BUILD_PIDS_LIMIT == 256
    assert tool.RUNTIME_ACTOR_PIDS_LIMIT == 64
    assert tool.RUST_IMAGE in command
    assert "CARGO_NET_OFFLINE=true" in command
    assert "CARGO_BUILD_JOBS=8" in command
    assert "CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1" in command
    assert "CARGO_HOME=/tmp/cargo-home" in command
    assert "/cargo-registry/cache:/cargo-seed/cache:ro" in command
    assert "/cargo-registry/index:/cargo-seed/index:ro" in command
    assert not any("/cargo-seed/registry" in argument for argument in command)
    assert "/snapshot/rust:/workspace/rust:ro" in command
    assert "fresh-target:/cargo-target:rw" in command
    assert "/workspace/rust/m3_closure_enumerator_shrink6" in command
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
    assert any(
        "cp -a --no-preserve=ownership /cargo-seed/cache" in argument
        and "cp -a --no-preserve=ownership /cargo-seed/index" in argument
        for argument in command
    )


def test_dependency_seed_matches_lock_and_excludes_preunpacked_src() -> None:
    receipt = tool.verify_cargo_dependency_seed_v1(
        tool.DEFAULT_CARGO_REGISTRY,
        PROJECT_ROOT / "rust/m3_closure_enumerator_shrink6/Cargo.lock",
    )
    assert receipt["locked_registry_package_count"] == 21
    assert receipt["verified_crate_count"] == 21
    assert receipt["src_subtree_included"] is False
    assert receipt["fresh_tmpfs_cargo_home"] is True
    assert str(receipt["manifest_root"]).startswith("sha256:")
    assert receipt["hash_domain"] == "HEGEL/SHRINK6/CARGO_SEED_MANIFEST/V1"
    assert all(
        not str(row["path"]).startswith("src/")
        for row in receipt["files"]
    )
    assert all(set(row) == {"path", "mode", "size", "sha256"} for row in receipt["files"])


def test_dependency_snapshot_is_reverified_and_mounts_only_two_subtrees() -> None:
    with tempfile.TemporaryDirectory(
        prefix="hegel-shrink6-seed-test-", dir="/tmp"
    ) as temporary:
        destination = Path(temporary) / "snapshot"
        receipt = tool.create_verified_cargo_seed_snapshot_v1(
            tool.DEFAULT_CARGO_REGISTRY,
            destination,
            PROJECT_ROOT / "rust/m3_closure_enumerator_shrink6/Cargo.lock",
        )
        try:
            assert receipt["snapshot_reverified_after_copy"] is True
            assert receipt["permission_frozen_reverified"] is True
            assert receipt["docker_bind_mounts_exact"] == ["cache", "index"]
            assert sorted(path.name for path in destination.iterdir()) == [
                "cache",
                "index",
            ]
            assert not (destination / "src").exists()
            assert all(row["mode"] == "0444" for row in receipt["files"])
            post_freeze = tool.verify_cargo_dependency_seed_v1(
                destination,
                PROJECT_ROOT / "rust/m3_closure_enumerator_shrink6/Cargo.lock",
            )
            assert all(
                receipt.get(field) == value
                for field, value in post_freeze.items()
            )
        finally:
            tool._restore_owner_writable_tree(destination)


@pytest.mark.skipif(
    os.environ.get("HEGEL_RUN_SHRINK6_DOCKER_SMOKE") != "1",
    reason="explicit opt-in for the offline hardened Docker build smoke",
)
def test_hardened_offline_build_really_runs_without_chown_capability() -> None:
    """Exercise the exact cap-dropped build, without enumeration or network."""

    volume = (
        f"hegel-shrink6-build-smoke-{os.getpid()}-"
        f"{__import__('secrets').token_hex(4)}"
    )
    with (
        tempfile.TemporaryDirectory(
            prefix="hegel-shrink6-build-control-", dir="/tmp"
        ) as control,
        tempfile.TemporaryDirectory(
            prefix="hegel-shrink6-build-seed-", dir="/tmp"
        ) as seed_parent,
    ):
        seed = Path(seed_parent) / "cargo-seed"
        created = False
        try:
            tool.hardened._initialize_docker_environment(Path(control))
            receipt = tool.create_verified_cargo_seed_snapshot_v1(
                tool.DEFAULT_CARGO_REGISTRY,
                seed,
                PROJECT_ROOT / "rust/m3_closure_enumerator_shrink6/Cargo.lock",
            )
            tool._create_fresh_volume(volume, "source-y-offline-build-smoke")
            created = True
            digest = tool._build_rust(PROJECT_ROOT, seed, volume, 2)
            assert len(digest) == 64
            assert all(character in "0123456789abcdef" for character in digest)
            post_build = tool.verify_cargo_dependency_seed_v1(
                seed,
                PROJECT_ROOT / "rust/m3_closure_enumerator_shrink6/Cargo.lock",
            )
            assert all(
                receipt.get(field) == value
                for field, value in post_build.items()
            )
        finally:
            if created:
                tool.hardened._remove_volume(volume)
            tool.hardened._DOCKER_ENV = None
            if seed.exists():
                tool._restore_owner_writable_tree(seed)


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
    assert receipt["maximum_ast_depth"] == 3
    assert receipt["maximum_ast_node_count"] == 6
    assert receipt["formal_bucket_count"] == 120
    assert receipt["terminal_route"] == tool.DSL_TOO_LARGE_ROUTE
    assert receipt["next_preregistered_shrink_step_or_null"] is None
    receipt["formal_roots"] = {"forbidden": "root"}
    with pytest.raises(tool.SupervisorError) as raised:
        tool.validate_host_receipt(receipt)
    assert raised.value.code == tool.FAIL_AUTHORITY


def test_host_receipt_schema_is_recursive_and_exact() -> None:
    receipt = _host_receipt()
    receipt["unknown"] = False
    with pytest.raises(tool.SupervisorError) as raised:
        tool.validate_host_receipt(receipt)
    assert raised.value.code == tool.FAIL_AUTHORITY

    receipt = _host_receipt()
    receipt["stream_sha256"]["unknown"] = "00" * 32  # type: ignore[index]
    with pytest.raises(tool.SupervisorError) as raised:
        tool.validate_host_receipt(receipt)
    assert raised.value.code == tool.FAIL_HOST


@pytest.mark.parametrize(
    "field",
    [
        "witness_adjacency_verified",
        "witness_closed_bucket_rank_verified",
        "post_witness_traversal_buckets_untouched",
    ],
)
def test_host_receipt_overflow_booleans_are_type_strict(field: str) -> None:
    receipt = _host_receipt()
    receipt[field] = 1
    with pytest.raises(tool.SupervisorError) as raised:
        tool.validate_host_receipt(receipt)
    assert raised.value.code == tool.FAIL_PRESERVATION


def test_host_receipt_rejects_raw_count_below_canonical_boundary() -> None:
    receipt = _host_receipt()
    receipt["raw_operator_application_count"] = 50_000
    with pytest.raises(tool.SupervisorError) as raised:
        tool.validate_host_receipt(receipt)
    assert raised.value.code == tool.FAIL_HOST

    receipt = _host_receipt()
    receipt["residual_out_of_budget_canonical_programs"] = 0
    with pytest.raises(tool.SupervisorError) as raised:
        tool.validate_host_receipt(receipt)
    assert raised.value.code == tool.FAIL_PRESERVATION

    receipt = _host_receipt()
    receipt["closure_status"] = "COMPLETE"
    receipt["canonical_program_count"] = 0
    receipt["raw_operator_application_count"] = 0
    receipt["raw_operator_application_count_scope"] = (
        "THROUGH_FULLY_CLOSED_FRONTIER"
    )
    receipt["witness_adjacency_verified"] = None
    receipt["witness_closed_bucket_rank_verified"] = None
    receipt["post_witness_traversal_buckets_untouched"] = None
    receipt["residual_out_of_budget_canonical_programs"] = 0
    receipt["first_out_of_budget_program_hash"] = None
    receipt["first_out_of_budget_program_cbor_hex"] = None
    receipt["first_out_of_budget_program_ordinal"] = None
    receipt["witness_bucket_index"] = None
    receipt["witness_output_sort_id"] = None
    receipt["witness_ast_depth"] = None
    receipt["witness_ast_node_count"] = None
    receipt["prefix_preservation_verified"] = None
    receipt["terminal_route"] = tool.COMPLETE_ROUTE
    with pytest.raises(tool.SupervisorError) as raised:
        tool.validate_host_receipt(receipt)
    assert raised.value.code == tool.FAIL_HOST


def test_complete_route_is_only_eligibility_and_never_starts_m3() -> None:
    receipt = _complete_host_receipt()
    tool.validate_host_receipt(receipt)
    assert receipt["terminal_route"] == tool.COMPLETE_ROUTE
    assert receipt["formal_state_transition_allowed"] is False
    assert receipt["execution_state"] == "NOT_RUN"
    assert receipt["prefix_preservation_verified"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("raw_operator_application_count", 3_120_718),
        ("residual_out_of_budget_canonical_programs", 2_236),
        ("first_out_of_budget_program_ordinal", 50_000),
        ("witness_bucket_index", 64),
        ("witness_output_sort_id", 2),
        ("witness_ast_depth", 3),
        ("witness_ast_node_count", 5),
        ("first_out_of_budget_program_hash", "00" * 32),
        ("first_out_of_budget_program_cbor_hex", "00"),
        ("prefix_preservation_verified", False),
    ],
)
def test_preregistered_prefix_mismatch_is_inconclusive(
    field: str, value: object
) -> None:
    receipt = _host_receipt()
    receipt[field] = value
    with pytest.raises(tool.SupervisorError) as raised:
        tool.validate_host_receipt(receipt)
    assert raised.value.code == tool.FAIL_PRESERVATION


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("preregistered_shrink_order_total_steps", 7),
        ("preregistered_shrink_order_consumed_through_step", 5),
        ("next_preregistered_shrink_step_or_null", 7),
        ("terminal_route", "SHRINK_STEP_7"),
        ("budget_change_authorized", True),
        ("additional_shrink_authorized", True),
        ("new_dsl_version_authorized", True),
    ],
)
def test_exhausted_shrink_order_cannot_invent_continuation(
    field: str, value: object
) -> None:
    receipt = _host_receipt()
    receipt[field] = value
    with pytest.raises(tool.SupervisorError) as raised:
        tool.validate_host_receipt(receipt)
    assert raised.value.code in {tool.FAIL_AUTHORITY, tool.FAIL_PRESERVATION}


def test_endpoint_stdout_must_match_retained_report(tmp_path: Path) -> None:
    def wire(value: object) -> bytes:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")

    output = tmp_path / "result"
    output.mkdir()
    retained = {"implementation": "python", "status": "COMPLETE"}
    (output / "report.json").write_bytes(wire(retained))
    endpoint = tool.EndpointResult(
        "python",
        wire(retained),
        retained,
    )
    tool._check_endpoint_stdout_matches_disk_report(output, endpoint)
    endpoint = tool.EndpointResult(
        "python",
        wire({"implementation": "python", "status": "DSL_TOO_LARGE"}),
        {"implementation": "python", "status": "DSL_TOO_LARGE"},
    )
    with pytest.raises(tool.SupervisorError) as raised:
        tool._check_endpoint_stdout_matches_disk_report(output, endpoint)
    assert raised.value.code == tool.FAIL_OUTPUT

    for retained_value, stdout_value in ((True, 1), (1, 1.0)):
        (output / "report.json").write_bytes(wire({"value": retained_value}))
        endpoint = tool.EndpointResult(
            "python",
            wire({"value": stdout_value}),
            {"value": stdout_value},
        )
        with pytest.raises(tool.SupervisorError) as raised:
            tool._check_endpoint_stdout_matches_disk_report(output, endpoint)
        assert raised.value.code == tool.FAIL_OUTPUT


def test_canonical_json_wire_is_recursive_sorted_compact_ascii_and_lf_ready() -> None:
    value = {
        "z": 1,
        "emoji": "\U0001f600",
        "del": "\u007f",
        "a": {"z": "\u4e2d", "a": True},
    }
    assert tool._canonical_json(value) + b"\n" == (
        b'{"a":{"a":true,"z":"\\u4e2d"},'
        b'"del":"\\u007f","emoji":"\\ud83d\\ude00","z":1}\n'
    )


def test_strict_json_rejects_duplicate_names_and_nonfinite_values() -> None:
    for payload in (b'{"a":1,"a":1}', b'{"a":NaN}'):
        with pytest.raises(tool.SupervisorError) as raised:
            tool._strict_json_object(payload, "probe", tool.FAIL_ENDPOINT)
        assert raised.value.code == tool.FAIL_ENDPOINT


def test_evidence_x_is_the_only_byte_exact_engineering_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = PROJECT_ROOT / tool.EVIDENCE_X_ARTIFACT_PATH.removeprefix(
        "Hegel Machine/"
    )
    payload = artifact.read_bytes()
    assert __import__("hashlib").sha256(payload).hexdigest() == (
        tool.EVIDENCE_X_ARTIFACT_SHA256
    )

    def fake_git(*arguments: str, binary: bool = False) -> bytes | str:
        assert arguments[0] == "show"
        assert binary is True
        return payload

    monkeypatch.setattr(tool, "_git", fake_git)
    receipt = tool._validate_evidence_x_admission(
        "a" * 40,
        [tool.EVIDENCE_X_COMMIT],
        [
            {
                "path": tool.EVIDENCE_X_ARTIFACT_PATH,
                "sha256": tool.EVIDENCE_X_ARTIFACT_SHA256,
            }
        ],
    )
    assert receipt == tool.STRICT_QUALIFICATION_BINDING

    with pytest.raises(tool.SupervisorError) as raised:
        tool._validate_evidence_x_admission(
            "a" * 40,
            ["b" * 40],
            [],
        )
    assert raised.value.code == tool.FAIL_GIT


def test_archive_surface_is_commit_bound_and_excludes_target_inputs() -> None:
    assert tool.SUPERVISOR_PATH in tool.ARCHIVE_PATHS
    assert tool.HARDENED_SUPERVISOR_PATH in tool.ARCHIVE_PATHS
    assert tool.SUPERVISOR_TEST_PATH in tool.ARCHIVE_PATHS
    assert set(tool.IMPLEMENTATION_TEST_PATHS).issubset(tool.ARCHIVE_PATHS)
    assert tool.PROFILE_DOC_PATH in tool.ARCHIVE_PATHS
    assert tool.EVIDENCE_X_ARTIFACT_PATH in tool.ARCHIVE_PATHS
    assert tool.BUILD_PROFILE_PATH in tool.ARCHIVE_PATHS
    assert "Hegel Machine/rust/m3_closure_enumerator_shrink6" in tool.ARCHIVE_PATHS
    assert (
        "Hegel Machine/src/hegel_machine/phase3_m3_shrink6_dual_diagnostic_entrypoint_v1.py"
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
