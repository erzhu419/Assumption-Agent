from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import pytest

from hegel_machine import phase3_q1_full_node6_host_replay_v1 as host


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase3_q1_full_node6_dual_supervisor_v1",
    ROOT / "tools" / "phase3_q1_full_node6_dual_supervisor_v1.py",
)
assert SPEC is not None and SPEC.loader is not None
supervisor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = supervisor
SPEC.loader.exec_module(supervisor)

CAPSULE = "sha256:" + "1" * 64
COMMIT = "2" * 40


@pytest.fixture
def native_tmp_path() -> Path:
    path = Path(tempfile.mkdtemp(prefix="hegel-node6-host-test-", dir="/tmp"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _projection(seed: int) -> dict[str, object]:
    streams = []
    combined = sha256()
    combined.update(host.PROJECTION_STATE_DOMAIN)
    total = 0
    for ordinal, kind in enumerate(host.STREAM_KINDS, 1):
        count = (10 + seed) * ordinal
        payload = (20 + seed) * ordinal
        framed = payload + 8 * count
        digest = f"{seed + ordinal:064x}"
        streams.append(
            {
                "counting_discard_sha256": digest,
                "framed_bytes": framed,
                "payload_bytes": payload,
                "record_count": count,
                "stream_kind": kind,
            }
        )
        total += framed
        combined.update(kind.encode("ascii"))
        combined.update(count.to_bytes(8, "big"))
        combined.update(framed.to_bytes(8, "big"))
        combined.update(digest.encode("ascii"))
    return {
        "combined_state_projection_sha256": combined.hexdigest(),
        "diagnostic_only": True,
        "formal_archive_materialized": False,
        "projected_diagnostic_scratch_bytes": 0,
        "projected_diagnostic_state_bytes": total,
        "projection_kind": "DIAGNOSTIC_STATE_AGREEMENT_KAT_NOT_FORMAL_ARCHIVE",
        "schema_version": "hegel-phase3a-q1-capacity-counting-discard-projection/1",
        "streams": streams,
    }


def _formal_projection(seed: int) -> dict[str, object]:
    streams = []
    total = 0
    for stream_id, kind in enumerate(host.STREAM_KINDS, 1):
        count = 10 * stream_id
        payload = (30 + seed) * stream_id
        framed = payload + 4 * count
        total += framed
        streams.append(
            {
                "archive_root": "sha256:" + f"{seed + stream_id:064x}",
                "canonical_record_payload_bytes": payload,
                "chunk_count": 1,
                "chunk_manifest_subtree_root": "sha256:" + f"{10 + seed + stream_id:064x}",
                "counting_discard_commitment": "sha256:" + f"{20 + seed + stream_id:064x}",
                "external_sort_projection_root": "sha256:" + f"{30 + seed + stream_id:064x}",
                "framed_bytes": framed,
                "record_count": count,
                "retained_framed_blob_bytes": 0,
                "retained_framed_blob_count": 0,
                "stream_kind": kind,
                "stream_kind_id": stream_id,
            }
        )
    chunk_manifest = 100 + seed
    fixed_point = 200 + seed
    signature_manifest = 300 + seed
    return {
        "diagnostic_only": True,
        "formal_archive_materialized": False,
        "projected_chunk_manifest_stream_bytes": chunk_manifest,
        "projected_fixed_point_frame_bytes": fixed_point,
        "projected_partition_payload_bytes": (
            total + chunk_manifest + fixed_point + signature_manifest
        ),
        "projected_peak_formal_scratch_bytes": 4096,
        "projected_record_stream_bytes": total,
        "projected_signature_manifest_frame_bytes": signature_manifest,
        "projection_kind": "FROZEN_FORMAL_WIRE_COUNTING_DISCARD_NOT_MATERIALIZED",
        "schema_version": "hegel-phase3a-q1-full-node6-formal-wire-projection/1",
        "streams": streams,
    }


def _partition(signature_id: int, elapsed: int) -> dict[str, object]:
    raw = 1000 + signature_id
    strict = 900 + signature_id
    rewrite = 20 + signature_id
    return {
        "diagnostic_state_projection": _projection(signature_id),
        "behavior_class_count": 40 + signature_id,
        "complete_claim_allowed": False,
        "continuation_bank_point_count": 100 + signature_id,
        "coverage": {
            "all_eligible_covered": True,
            "coverage_record_count": 846,
            "eligible_application_count": raw,
            "processed_application_count": raw,
            "rewrite_collapse_count": rewrite,
            "strict_admitted_application_count": strict,
            "zero_eligible_record_count": 5,
        },
        "depth_barriers": [
            {
                "barrier_kind": "STRUCTURAL_BOUNDARY",
                "behavior_class_count_after_barrier": 40 + signature_id,
                "continuation_bank_mutation_count": 0,
                "continuation_bank_point_count_after_barrier": 100 + signature_id,
                "depth": 4,
                "eligible_raw_application_count": 0,
                "new_behavior_class_count": 0,
                "new_signature_cohort_count": 0,
                "rewrite_collapse_count": 0,
                "signature_cohort_count_after_barrier": 80 + signature_id,
                "strict_admitted_application_count": 0,
                "visible_frontier_point_count_after_barrier": 50 + signature_id,
            }
        ],
        "diagnostic_only": True,
        "elapsed_monotonic_ns": elapsed,
        "fixed_point": {
            "all_eligible_operator_and_bank_tuple_expansions_covered": True,
            "final_continuation_bank_delta": 0,
            "final_new_behavior_class_delta": 0,
            "final_visible_frontier_delta": 0,
            "no_resource_guard_hit": True,
            "saturated_diagnostic_only": True,
            "work_queue_empty": True,
            "zero_delta_structural_boundary": True,
        },
        "formal_roots": None,
        "formal_roots_generated": False,
        "formal_wire_projection": _formal_projection(signature_id),
        "frozen_leaf_count": 810,
        "full_v16_structural_limits_applied": True,
        "input_signature_id": signature_id,
        "input_signature_name": "OddInputV1" if signature_id == 1 else "SinkInputV1",
        "limits": {
            "maximum_ast_depth": 3,
            "maximum_ast_node_count": 6,
            "maximum_behavior_classes": 16777216,
            "maximum_continuation_bank_points": 67108864,
            "maximum_continuation_bank_points_per_class": 65536,
            "maximum_raw_operator_applications": 4294967295,
            "maximum_saturation_rounds": 16,
            "maximum_visible_frontier_points": 33554432,
            "maximum_visible_frontier_points_per_class": 65536,
            "maximum_wall_time_seconds": 172800,
            "maximum_work_queue_points": 67108864,
        },
        "maximum_bank_points_per_class": 4,
        "maximum_frontier_points_per_class": 2,
        "peak_behavior_class_count": 40 + signature_id,
        "peak_continuation_bank_point_count": 100 + signature_id,
        "peak_continuation_bank_points_per_class": 4,
        "peak_raw_operator_application_count": raw,
        "peak_saturation_round_count": 5,
        "peak_visible_frontier_point_count": 50 + signature_id,
        "peak_visible_frontier_points_per_class": 2,
        "peak_work_queue_points": 700,
        "raw_operator_application_count": raw,
        "resource_guard_id": None,
        "resource_guard_name": None,
        "same_run_record_set_root": "sha256:" + f"{signature_id + 70:064x}",
        "rewrite_collapse_count": rewrite,
        "role_evaluation_performed": False,
        "signature_cohort_count": 80 + signature_id,
        "split_accessed": False,
        "strict_admitted_application_count": strict,
        "target_truth_accessed": False,
        "terminal_status": host.SUCCESS_STATUS,
        "traversal_closed": True,
        "universe_root": "sha256:" + ("a" if signature_id == 1 else "b") * 64,
        "universe_row_count": 480 if signature_id == 1 else 85,
        "vector_cache_entry_count": 100,
        "vector_cache_hit_count": 20,
        "vector_cache_miss_count": 80,
        "visible_frontier_point_count": 50 + signature_id,
    }


def _diagnostic(actor_id: str, elapsed_delta: int = 0) -> dict[str, object]:
    partitions = [_partition(1, 10 + elapsed_delta), _partition(2, 20 + elapsed_delta)]
    closure_frame = 777
    archive_payload = (
        sum(
            row["formal_wire_projection"]["projected_partition_payload_bytes"]
            for row in partitions
        )
        + closure_frame
    )
    return {
        "action_id": host.ACTION_ID,
        "active_transition_allowed": False,
        "actor_id": actor_id,
        "actual_preflight_peak_scratch_bytes": (
            1000 if actor_id == "PYTHON_ENDPOINT" else 2000
        ),
        "actual_preflight_diagnostic_output_bytes": (
            3000 if actor_id == "PYTHON_ENDPOINT" else 4000
        ),
        "actual_preflight_peak_rss_bytes": (
            5000 if actor_id == "PYTHON_ENDPOINT" else 6000
        ),
        "actual_preflight_terminal_live_scratch_bytes": 0,
        "actual_preflight_wall_time_ns": (
            7000 if actor_id == "PYTHON_ENDPOINT" else 8000
        ),
        "capsule_digest": CAPSULE,
        "closure_semantics_version": "hegel-quotient-closure-v1.0.1",
        "diagnostic_only": True,
        "dsl_version": "hegel-old-dsl-v1.6.0",
        "endpoint_schema_version": (
            "hegel-phase3a-q1-full-node6-python-endpoint/1"
            if actor_id == "PYTHON_ENDPOINT"
            else "hegel-phase3a-q1-full-node6-rust-endpoint/1"
        ),
        "implementation_id": "python-v1" if actor_id == "PYTHON_ENDPOINT" else "rust-v1",
        "partitions": partitions,
        "projected_closure_bundle_frame_bytes": closure_frame,
        "projected_formal_archive_payload_bytes": archive_payload,
        "projected_formal_total_output_bytes": archive_payload + 1_048_576,
        "projected_peak_formal_scratch_bytes": 4096,
        "q1_execution_started": False,
        "q1_formal_roots": None,
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_output_slots": [None] * 8,
        "q1_receipt": None,
        "q1_state": "NOT_RUN",
        "q2_state": "NOT_RUN",
        "role_evaluation_performed": False,
        "schema_version": "hegel-phase3a-q1-full-node6-capacity-diagnostic/1",
        "source_commit": COMMIT,
        "split_accessed": False,
        "target_truth_accessed": False,
        "terminal_status": host.SUCCESS_STATUS,
    }


def _guard_diagnostic(actor_id: str) -> dict[str, object]:
    value = _diagnostic(actor_id)
    value["actual_preflight_peak_scratch_bytes"] = 0
    for field in (
        "projected_closure_bundle_frame_bytes",
        "projected_formal_archive_payload_bytes",
        "projected_formal_total_output_bytes",
        "projected_peak_formal_scratch_bytes",
    ):
        value[field] = None
    value["terminal_status"] = host.GUARD_STATUS
    value["partitions"] = value["partitions"][:1]
    row = value["partitions"][0]
    row["terminal_status"] = host.GUARD_STATUS
    row["traversal_closed"] = False
    row["resource_guard_id"] = 1
    row["resource_guard_name"] = "RAW_OPERATOR_APPLICATIONS"
    row["same_run_record_set_root"] = None
    row["coverage"] = None
    row["diagnostic_state_projection"] = None
    row["formal_wire_projection"] = None
    row["fixed_point"] = None
    return value


def _sink_guard_diagnostic(actor_id: str) -> dict[str, object]:
    value = _diagnostic(actor_id)
    value["actual_preflight_peak_scratch_bytes"] = 0
    for field in (
        "projected_closure_bundle_frame_bytes",
        "projected_formal_archive_payload_bytes",
        "projected_formal_total_output_bytes",
        "projected_peak_formal_scratch_bytes",
    ):
        value[field] = None
    value["terminal_status"] = host.GUARD_STATUS
    for row in value["partitions"]:
        row["coverage"] = None
        row["diagnostic_state_projection"] = None
        row["fixed_point"] = None
        row["formal_wire_projection"] = None
        row["same_run_record_set_root"] = None
    sink = value["partitions"][1]
    sink["terminal_status"] = host.GUARD_STATUS
    sink["traversal_closed"] = False
    sink["resource_guard_id"] = 1
    sink["resource_guard_name"] = "RAW_OPERATOR_APPLICATIONS"
    return value


def _write_bundle(root: Path, diagnostic: dict[str, object]) -> bytes:
    root.mkdir(exist_ok=True)
    assert not any(root.iterdir())
    stdout = host.canonical_json_line_v1(diagnostic)
    payloads = {
        "diagnostic-summary-v1.json": stdout,
        "projection-summary-v1.json": host.canonical_json_line_v1(
            host._expected_sidecar(diagnostic, "projection")
        ),
        "coverage-summary-v1.json": host.canonical_json_line_v1(
            host._expected_sidecar(diagnostic, "coverage")
        ),
        "fixed-point-summary-v1.json": host.canonical_json_line_v1(
            host._expected_sidecar(diagnostic, "fixed-point")
        ),
    }
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
    manifest = {
        "actor_id": diagnostic["actor_id"],
        "capsule_digest": CAPSULE,
        "diagnostic_only": True,
        "implementation_id": diagnostic["implementation_id"],
        "q1_formal_roots": None,
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_state": "NOT_RUN",
        "rows": [
            {
                "byte_count": len(payloads[name]),
                "filename": name,
                "sha256": sha256(payloads[name]).hexdigest(),
            }
            for name in sorted(payloads)
        ],
        "schema_version": host.SIDECAR_MANIFEST_SCHEMA_VERSION,
        "source_commit": COMMIT,
    }
    (root / host.MANIFEST_PATH).write_bytes(host.canonical_json_line_v1(manifest))
    return stdout


def _run_spec(tmp_path: Path) -> supervisor.SupervisorRunSpecV1:
    python_snapshot = (tmp_path / "python-snapshot").resolve()
    rust_snapshot = (tmp_path / "rust-snapshot").resolve()
    host_replay_snapshot = (tmp_path / "host-replay-snapshot").resolve()
    for snapshot, paths in (
        (python_snapshot, supervisor.PYTHON_ENDPOINT_SOURCE_PATHS),
        (rust_snapshot, supervisor.RUST_ENDPOINT_SOURCE_PATHS),
        (host_replay_snapshot, supervisor.HOST_REPLAY_SOURCE_PATHS),
    ):
        snapshot.mkdir()
        for relative in paths:
            path = snapshot / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture\n")
            path.chmod(0o444)
        supervisor._make_tree_readonly(snapshot)
    binary = tmp_path / "engine"
    binary.write_bytes(b"fixture")
    binary.chmod(0o555)
    admission = tmp_path / "admission.json"
    admission.write_bytes(b"{}\n")
    admission.chmod(0o444)
    seccomp = tmp_path / "seccomp-profile-v1.json"
    seccomp.write_bytes((ROOT / supervisor.SECCOMP_RELATIVE_PATH).read_bytes())
    seccomp.chmod(0o444)
    return supervisor.SupervisorRunSpecV1(
        capsule_digest=CAPSULE,
        source_commit=COMMIT,
        python_snapshot=python_snapshot,
        rust_snapshot=rust_snapshot,
        host_replay_snapshot=host_replay_snapshot,
        rust_binary=binary,
        admission_json=admission,
        seccomp_profile=seccomp,
        run_root=tmp_path / "run",
        trajectory_root=tmp_path / "trajectory",
    )


def test_host_replays_sidecars_and_ignores_only_endpoint_identity_and_time(tmp_path: Path) -> None:
    python_root = tmp_path / "python"
    rust_root = tmp_path / "rust"
    python_stdout = _write_bundle(python_root, _diagnostic("PYTHON_ENDPOINT"))
    rust_stdout = _write_bundle(rust_root, _diagnostic("RUST_ENDPOINT", 1000))
    result = host.replay_dual_endpoints_v1(
        python_stdout,
        python_root,
        rust_stdout,
        rust_root,
        expected_capsule_digest=CAPSULE,
        expected_source_commit=COMMIT,
    )
    assert result["terminal_status"] == host.SUCCESS_STATUS
    assert result["q1_state"] == "NOT_RUN"
    assert result["q1_gate_count"] == 0
    assert result["q1_output_slots"] == [None] * 8
    assert [row["actual_diagnostic_output_bytes"] for row in result["endpoint_rows"]] != [
        3000,
        4000,
    ]
    assert result["projected_host_replay_output_bytes"] == 1_048_576
    assert result["projected_peak_host_replay_scratch_bytes"] == 0


def test_host_rejects_one_semantic_count_difference(tmp_path: Path) -> None:
    python_root = tmp_path / "python"
    rust_root = tmp_path / "rust"
    python_stdout = _write_bundle(python_root, _diagnostic("PYTHON_ENDPOINT"))
    rust_diagnostic = _diagnostic("RUST_ENDPOINT")
    rust_diagnostic["partitions"][0]["behavior_class_count"] += 1
    rust_stdout = _write_bundle(rust_root, rust_diagnostic)
    with pytest.raises(host.FullNode6HostReplayError) as caught:
        host.replay_dual_endpoints_v1(
            python_stdout,
            python_root,
            rust_stdout,
            rust_root,
            expected_capsule_digest=CAPSULE,
            expected_source_commit=COMMIT,
        )
    assert caught.value.code == "FAIL_FULL_NODE6_HOST_DUAL_DISAGREEMENT"


def test_host_recomputes_partition_and_aggregate_formal_projection(tmp_path: Path) -> None:
    root = tmp_path / "python"
    diagnostic = _diagnostic("PYTHON_ENDPOINT")
    diagnostic["projected_formal_total_output_bytes"] += 1
    stdout = _write_bundle(root, diagnostic)
    with pytest.raises(host.FullNode6HostReplayError) as caught:
        host.replay_endpoint_v1(
            "PYTHON_ENDPOINT",
            stdout,
            root,
            expected_capsule_digest=CAPSULE,
            expected_source_commit=COMMIT,
        )
    assert caught.value.code == "FAIL_FULL_NODE6_HOST_FORMAL_WIRE"


def test_host_accepts_exact_registered_guard_prefix_and_does_not_require_sink(
    tmp_path: Path,
) -> None:
    python_root = tmp_path / "python"
    rust_root = tmp_path / "rust"
    python_stdout = _write_bundle(python_root, _guard_diagnostic("PYTHON_ENDPOINT"))
    rust_stdout = _write_bundle(rust_root, _guard_diagnostic("RUST_ENDPOINT"))
    result = host.replay_dual_endpoints_v1(
        python_stdout,
        python_root,
        rust_stdout,
        rust_root,
        expected_capsule_digest=CAPSULE,
        expected_source_commit=COMMIT,
    )
    assert result["terminal_status"] == host.GUARD_STATUS


def test_host_accepts_saturated_odd_then_sink_guard_only_when_entire_prefix_is_null(
    tmp_path: Path,
) -> None:
    python_root = tmp_path / "python"
    rust_root = tmp_path / "rust"
    python_stdout = _write_bundle(
        python_root, _sink_guard_diagnostic("PYTHON_ENDPOINT")
    )
    rust_stdout = _write_bundle(
        rust_root, _sink_guard_diagnostic("RUST_ENDPOINT")
    )
    result = host.replay_dual_endpoints_v1(
        python_stdout,
        python_root,
        rust_stdout,
        rust_root,
        expected_capsule_digest=CAPSULE,
        expected_source_commit=COMMIT,
    )
    assert result["terminal_status"] == host.GUARD_STATUS

    bad_root = tmp_path / "bad"
    bad = _sink_guard_diagnostic("PYTHON_ENDPOINT")
    bad["partitions"][0]["coverage"] = _diagnostic("PYTHON_ENDPOINT")[
        "partitions"
    ][0]["coverage"]
    bad_stdout = _write_bundle(bad_root, bad)
    with pytest.raises(host.FullNode6HostReplayError) as caught:
        host.replay_endpoint_v1(
            "PYTHON_ENDPOINT",
            bad_stdout,
            bad_root,
            expected_capsule_digest=CAPSULE,
            expected_source_commit=COMMIT,
        )
    assert caught.value.code == "FAIL_FULL_NODE6_HOST_GUARD_PREFIX"
    assert result["q1_state"] == "NOT_RUN"


def test_host_rejects_sidecar_byte_change(tmp_path: Path) -> None:
    root = tmp_path / "python"
    stdout = _write_bundle(root, _diagnostic("PYTHON_ENDPOINT"))
    (root / "projection-summary-v1.json").write_bytes(b"{}\n")
    with pytest.raises(host.FullNode6HostReplayError) as caught:
        host.replay_endpoint_v1(
            "PYTHON_ENDPOINT",
            stdout,
            root,
            expected_capsule_digest=CAPSULE,
            expected_source_commit=COMMIT,
        )
    assert caught.value.code == "FAIL_FULL_NODE6_HOST_MANIFEST"


def test_supervisor_launches_strict_sequence_then_replays_without_docker(
    native_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = native_tmp_path
    spec = _run_spec(tmp_path)
    run_root = spec.run_root
    trajectory_root = spec.trajectory_root
    observed: dict[str, object] = {}

    def launch_sequence(
        python_command,
        rust_command,
        _python_cid,
        _rust_cid,
        on_actor_started,
        on_actor_completed,
    ):
        observed["python"] = list(python_command)
        observed["rust"] = list(rust_command)
        on_actor_started("PYTHON_ENDPOINT")
        (run_root / "python-scratch" / "transient.bin").write_bytes(b"python")
        python_stdout = _write_bundle(
            run_root / "python-output", _diagnostic("PYTHON_ENDPOINT")
        )
        python = supervisor.EndpointProcessResultV1(
            "PYTHON_ENDPOINT",
            0,
            python_stdout,
            b"",
            {
                "actor_id": "PYTHON_ENDPOINT",
                "cgroup": {"cgroup_path": "/python.scope"},
                "container_id": "a" * 64,
                "measurement_wall_time_ns": 1_000_000_000,
                "memory_peak_bytes": 1_000_000,
            },
        )
        on_actor_completed(python)
        on_actor_started("RUST_ENDPOINT")
        (run_root / "rust-scratch" / "transient.bin").write_bytes(b"rust")
        rust_stdout = _write_bundle(
            run_root / "rust-output", _diagnostic("RUST_ENDPOINT", 7)
        )
        rust = supervisor.EndpointProcessResultV1(
            "RUST_ENDPOINT",
            0,
            rust_stdout,
            b"",
            {
                "actor_id": "RUST_ENDPOINT",
                "cgroup": {"cgroup_path": "/rust.scope"},
                "container_id": "b" * 64,
                "measurement_wall_time_ns": 2_000_000_000,
                "memory_peak_bytes": 2_000_000,
            },
        )
        on_actor_completed(rust)
        return python, rust

    def launch_host_replay(command, _cidfile):
        observed["host"] = list(command)
        replay = host.replay_dual_endpoints_v1(
            (run_root / "python.stdout").read_bytes(),
            run_root / "python-output",
            (run_root / "rust.stdout").read_bytes(),
            run_root / "rust-output",
            expected_capsule_digest=CAPSULE,
            expected_source_commit=COMMIT,
        )
        return supervisor.EndpointProcessResultV1(
            "TRUSTED_HOST_REPLAY",
            0,
            host.canonical_json_line_v1(replay),
            b"",
            {
                "actor_id": "TRUSTED_HOST_REPLAY",
                "cgroup": {"cgroup_path": "/host.scope"},
                "container_id": "c" * 64,
                "measurement_wall_time_ns": 500_000_000,
                "memory_peak_bytes": 500_000,
            },
        )

    result = supervisor.run_supervisor_v1(
        spec,
        launch_sequence=launch_sequence,
        launch_host_replay=launch_host_replay,
        check_local_images=False,
    )
    assert result["terminal_status"] == host.SUCCESS_STATUS
    assert result["resource_accounting"]["projected_host_replay_output_bytes"] == 1048576
    assert result["resource_accounting"]["projected_peak_host_replay_scratch_bytes"] == 0
    assert all(
        row["actual_diagnostic_output_bytes"] > 0
        for row in result["resource_accounting"][
            "endpoint_actual_diagnostic_output_bytes"
        ]
    )
    for command in (observed["python"], observed["rust"]):
        assert command[command.index("--network") + 1] == "none"
        assert "--read-only" in command
        assert command[command.index("--pull") + 1] == "never"
        assert command[command.index("--user") + 1] == "65534:65534"
        assert command[command.index("--ipc") + 1] == "private"
        assert command[command.index("--pids-limit") + 1] == "128"
        assert command[command.index("--ulimit") + 1] == "nofile=256:256"
        assert command[command.index("--tmpfs") + 1] == supervisor.TMPFS_SPEC
    assert observed["python"][observed["python"].index("--cpuset-cpus") + 1] == "0-11"
    assert observed["rust"][observed["rust"].index("--cpuset-cpus") + 1] == "12-23"
    assert observed["python"].count("--mount") == 4
    assert observed["rust"].count("--mount") == 4
    for command in (observed["python"], observed["rust"]):
        assert command[command.index("--scratch-dir") + 1] == "/scratch"
    host_command = observed["host"]
    assert host_command[host_command.index("--cpuset-cpus") + 1] == "0-11"
    assert host_command[host_command.index("--user") + 1] == "65534:65534"
    assert host_command.count("--mount") == 4
    assert any("dst=/python-output,readonly" in row for row in host_command)
    assert any("dst=/rust-output,readonly" in row for row in host_command)
    scratch_receipt = json.loads((run_root / "scratch-cleanup-v1.json").read_text())
    assert scratch_receipt["terminal_absence_proved"] is True
    assert not (run_root / "python-scratch").exists()
    assert not (run_root / "rust-scratch").exists()
    assert not (run_root / "host-replay-scratch").exists()
    assert (run_root / "host-replay-scratch-cleanup-v1.json").is_file()
    assert (run_root / "endpoint-resource-evidence-v1.json").is_file()
    assert (run_root / "python-resource-evidence-v1.json").is_file()
    assert (run_root / "rust-resource-evidence-v1.json").is_file()
    assert (run_root / "host-replay-resource-evidence-v1.json").is_file()
    assert (run_root / "python-output").stat().st_mode & 0o777 == 0o500
    assert (run_root / "rust-output").stat().st_mode & 0o777 == 0o500
    assert (trajectory_root / "supervisor-trajectory-terminal-v1.json").is_file()
    trajectory_events = [
        json.loads(row)
        for row in (
            trajectory_root / "supervisor-trajectory-v1.jsonl"
        ).read_text().splitlines()
    ]
    endpoint_events = [
        (row["phase"], row["status"])
        for row in trajectory_events
        if row["phase"] in ("PYTHON_ENDPOINT", "RUST_ENDPOINT")
    ]
    assert endpoint_events == [
        ("PYTHON_ENDPOINT", "STARTED"),
        ("PYTHON_ENDPOINT", "COMPLETED"),
        ("RUST_ENDPOINT", "STARTED"),
        ("RUST_ENDPOINT", "COMPLETED"),
    ]
    execution_plan = json.loads((run_root / "execution-plan-v1.json").read_text())
    assert execution_plan["endpoint_execution_order"] == [
        "PYTHON_ENDPOINT",
        "RUST_ENDPOINT",
    ]
    assert execution_plan["python_rust_concurrent"] is False
    assert execution_plan["python_then_rust_run_strictly_sequentially"] is True
    assert execution_plan["maximum_concurrently_live_endpoint_containers"] == 1
    assert (
        execution_plan["endpoint_process_deadline_seconds"]
        == supervisor.ENDPOINT_WALL_TIME_SECONDS
        == 518400
    )
    assert (
        execution_plan["endpoint_sequence_process_budget_seconds"]
        == supervisor.ENDPOINT_SEQUENCE_WALL_TIME_SECONDS
        == 1036800
    )


def test_public_supervisor_prepares_before_measurement_and_publishes_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = (tmp_path / "project").resolve()
    project.mkdir()
    run_root = tmp_path / "run"
    trajectory_root = tmp_path / "trajectory"
    artifact = tmp_path / "artifact.json"
    events: list[str] = []

    monkeypatch.setattr(
        supervisor,
        "_replay_clean_commit",
        lambda *_args: events.append("source"),
    )
    monkeypatch.setattr(
        supervisor,
        "_require_local_image",
        lambda image: events.append("image:" + image.split("@", 1)[0]),
    )

    def materialize(_project, _commit, target, paths=supervisor.HOST_REPLAY_SOURCE_PATHS):
        events.append("snapshot:" + target.name)
        target.mkdir()
        return tuple(
            {"byte_count": 0, "relative_path": path, "sha256": "0" * 64}
            for path in paths
        )

    monkeypatch.setattr(supervisor, "_materialize_snapshot", materialize)
    monkeypatch.setattr(supervisor, "_materialize_host_replay_snapshot", materialize)

    def seccomp(_project, path):
        events.append("seccomp")
        path.write_bytes((ROOT / supervisor.SECCOMP_RELATIVE_PATH).read_bytes())

    monkeypatch.setattr(supervisor, "_prepare_seccomp_profile", seccomp)

    def admission(_project, path, _capsule, _commit):
        events.append("admission")
        path.write_bytes(b"{}\n")

    monkeypatch.setattr(supervisor, "_write_admission", admission)

    def build(_snapshot, build_root, cargo_home):
        events.append("build")
        build_root.mkdir()
        cargo_home.mkdir()
        binary = build_root / "engine"
        binary.write_bytes(b"fixture")
        binary.chmod(0o700)
        return binary

    monkeypatch.setattr(supervisor, "_build_rust_binary", build)
    monkeypatch.setattr(supervisor, "_make_tree_readonly", lambda _root: None)
    monkeypatch.setattr(supervisor, "_verify_snapshot_manifest_v1", lambda *_args: None)

    def measured(spec, **_kwargs):
        events.append("measure")
        assert spec.rust_binary.exists()
        return {
            "host_replay_passed": True,
            "terminal_status": host.SUCCESS_STATUS,
            "q1_state": "NOT_RUN",
        }

    monkeypatch.setattr(supervisor, "run_supervisor_v1", measured)
    result = supervisor.run_public_supervisor_v1(
        project_root=project,
        source_commit=COMMIT,
        capsule_digest=CAPSULE,
        artifact=artifact,
        run_root=run_root,
        trajectory_root=trajectory_root,
    )
    assert events.index("build") < events.index("measure")
    assert result["terminal_status"] == host.SUCCESS_STATUS
    assert result["artifact_sha256"].startswith("sha256:")
    assert artifact.is_file()
    assert result["trajectory_event_count"] > 1
    assert Path(result["trajectory_event_log_path"]).parent == trajectory_root
    assert Path(result["trajectory_terminal_path"]).parent == trajectory_root
    assert not (run_root / "supervisor-trajectory-v1.jsonl").exists()
    terminal = json.loads(Path(result["trajectory_terminal_path"]).read_text())
    assert terminal["terminal_status"] == host.SUCCESS_STATUS
    events_on_disk = [
        json.loads(row)
        for row in Path(result["trajectory_event_log_path"]).read_text().splitlines()
    ]
    assert [row["event_ordinal"] for row in events_on_disk] == list(
        range(1, len(events_on_disk) + 1)
    )
    assert any(
        row["phase"] == "RUST_OFFLINE_BUILD" and row["status"] == "COMPLETED"
        for row in events_on_disk
    )
    assert events_on_disk[-1]["phase"] == "TERMINAL"


def test_public_supervisor_persists_exact_failed_phase_before_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = (tmp_path / "project").resolve()
    project.mkdir()
    run_root = tmp_path / "run"
    trajectory_root = tmp_path / "trajectory"
    artifact = tmp_path / "artifact.json"
    monkeypatch.setattr(supervisor, "_replay_clean_commit", lambda *_args: None)
    monkeypatch.setattr(supervisor, "_require_local_image", lambda _image: None)

    def materialize(_project, _commit, target, paths=supervisor.HOST_REPLAY_SOURCE_PATHS):
        target.mkdir()
        return tuple(
            {"byte_count": 0, "relative_path": path, "sha256": "0" * 64}
            for path in paths
        )

    monkeypatch.setattr(supervisor, "_materialize_snapshot", materialize)
    monkeypatch.setattr(supervisor, "_materialize_host_replay_snapshot", materialize)
    monkeypatch.setattr(
        supervisor,
        "_prepare_seccomp_profile",
        lambda _project, path: path.write_bytes(
            (ROOT / supervisor.SECCOMP_RELATIVE_PATH).read_bytes()
        ),
    )

    def admission(_project, path, _capsule, _commit):
        path.write_bytes(b"{}\n")

    monkeypatch.setattr(supervisor, "_write_admission", admission)

    def failed_build(_snapshot, _build_root, _cargo_home):
        raise supervisor.FullNode6SupervisorError(
            "FAIL_FULL_NODE6_SUPERVISOR_BUILD",
            "fixture offline build failure",
        )

    monkeypatch.setattr(supervisor, "_build_rust_binary", failed_build)
    with pytest.raises(supervisor.FullNode6SupervisorError) as caught:
        supervisor.run_public_supervisor_v1(
            project_root=project,
            source_commit=COMMIT,
            capsule_digest=CAPSULE,
            artifact=artifact,
            run_root=run_root,
            trajectory_root=trajectory_root,
        )
    assert caught.value.code == "FAIL_FULL_NODE6_SUPERVISOR_BUILD"
    terminal = json.loads(
        (trajectory_root / "supervisor-trajectory-terminal-v1.json").read_text()
    )
    assert terminal["error_code_or_null"] == "FAIL_FULL_NODE6_SUPERVISOR_BUILD"
    events_on_disk = [
        json.loads(row)
        for row in (trajectory_root / "supervisor-trajectory-v1.jsonl").read_text().splitlines()
    ]
    assert any(
        row["phase"] == "RUST_OFFLINE_BUILD" and row["status"] == "FAILED"
        for row in events_on_disk
    )
    assert events_on_disk[-1]["phase"] == "TERMINAL"
    assert not artifact.exists()


def test_rust_build_command_is_single_workdir_offline_and_not_30_second_killed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    build_root = tmp_path / "rust-build"
    cargo_home = tmp_path / "cargo-home"
    observed: dict[str, object] = {}

    def copytree(_source, destination):
        destination.mkdir()

    def run_capture(command, label, **kwargs):
        observed["command"] = list(command)
        observed["label"] = label
        observed.update(kwargs)
        binary = build_root / "target/release/hegel-q1-full-capacity-preflight-engine"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"fixture")
        binary.chmod(0o700)
        return b""

    monkeypatch.setattr(supervisor.shutil, "copytree", copytree)
    monkeypatch.setattr(supervisor, "_run_capture", run_capture)
    binary = supervisor._build_rust_binary(snapshot, build_root, cargo_home)
    command = observed["command"]
    assert binary.is_file()
    assert command.count("--workdir") == 1
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--cpuset-cpus") + 1] == "12-23"
    assert command[command.index("--memory") + 1] == str(supervisor.MEMORY_BYTES)
    assert command[command.index("--pids-limit") + 1] == str(supervisor.PIDS_LIMIT)
    assert "CARGO_NET_OFFLINE=true" in command
    assert "--offline" in command and "--locked" in command
    assert command[-6:] == [
        "build",
        "--release",
        "--offline",
        "--locked",
        "--target-dir",
        "/build/target",
    ]
    assert observed["timeout_seconds"] == supervisor.RUST_BUILD_TIMEOUT_SECONDS
    assert observed["timeout_seconds"] > 30
    assert observed["evidence_root"] == tmp_path


def test_exact_source_snapshots_bind_34_21_55_rows_and_reject_extra_file(
    native_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert len(supervisor.PYTHON_ENDPOINT_SOURCE_PATHS) == 34
    assert len(supervisor.RUST_ENDPOINT_SOURCE_PATHS) == 21
    assert (
        len(supervisor.PYTHON_ENDPOINT_SOURCE_PATHS)
        + len(supervisor.RUST_ENDPOINT_SOURCE_PATHS)
        == supervisor.EXECUTION_CLOSURE_ROW_COUNT
        == 55
    )
    assert not any(
        forbidden in path
        for path in (
            *supervisor.PYTHON_ENDPOINT_SOURCE_PATHS,
            *supervisor.RUST_ENDPOINT_SOURCE_PATHS,
            *supervisor.HOST_REPLAY_SOURCE_PATHS,
        )
        for forbidden in ("phase3_m25_split", "phase3_m25_rows", "phase3_dsl_v1")
    )

    def git_blob(root, arguments, _label):
        if arguments == ["rev-parse", "--show-toplevel"]:
            return (str(root) + "\n").encode("utf-8")
        assert arguments[:2] == ["cat-file", "blob"]
        relative = arguments[2].split(":", 1)[1]
        return (relative + "\n").encode("ascii")

    monkeypatch.setattr(supervisor, "_git_output", git_blob)
    snapshot = native_tmp_path / "snapshot"
    paths = supervisor.HOST_REPLAY_SOURCE_PATHS
    rows = supervisor._materialize_snapshot(
        native_tmp_path,
        COMMIT,
        snapshot,
        paths,
    )
    supervisor._make_tree_readonly(snapshot)
    supervisor._verify_snapshot_manifest_v1(snapshot, rows)
    extra = snapshot / "forbidden.py"
    snapshot.chmod(0o700)
    extra.write_bytes(b"forbidden\n")
    extra.chmod(0o444)
    supervisor._make_tree_readonly(snapshot)
    with pytest.raises(supervisor.FullNode6SupervisorError) as caught:
        supervisor._verify_snapshot_manifest_v1(snapshot, rows)
    assert caught.value.code == "FAIL_FULL_NODE6_SUPERVISOR_SOURCE"


def test_readonly_git_timeout_is_frozen_and_maps_to_source_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def timeout(command, **kwargs):
        observed["command"] = command
        observed["timeout"] = kwargs.get("timeout")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(supervisor.subprocess, "run", timeout)
    with pytest.raises(supervisor.FullNode6SupervisorError) as caught:
        supervisor._git_output(
            tmp_path,
            ["rev-parse", "--show-toplevel"],
            "repository top-level",
        )
    assert caught.value.code == "FAIL_FULL_NODE6_SUPERVISOR_SOURCE"
    assert observed["timeout"] == supervisor.GIT_READONLY_TIMEOUT_SECONDS == 60
    assert observed["command"] == [
        "/usr/bin/git",
        "-C",
        str(tmp_path),
        "rev-parse",
        "--show-toplevel",
    ]


def test_snapshot_materialization_uses_workspace_repository_prefix_but_relative_outputs(
    native_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = native_tmp_path / "repo"
    project = repository / "Hegel Machine"
    project.mkdir(parents=True)
    observed: list[tuple[str, ...]] = []

    def git_blob(root, arguments, _label):
        assert root == project
        observed.append(tuple(arguments))
        if arguments == ["rev-parse", "--show-toplevel"]:
            return (str(repository) + "\n").encode("utf-8")
        assert arguments[:2] == ["cat-file", "blob"]
        return (arguments[2] + "\n").encode("utf-8")

    monkeypatch.setattr(supervisor, "_git_output", git_blob)
    python_target = native_tmp_path / "python-snapshot"
    rust_target = native_tmp_path / "rust-snapshot"
    host_target = native_tmp_path / "host-snapshot"
    python_rows = supervisor._materialize_snapshot(
        project,
        COMMIT,
        python_target,
        supervisor.PYTHON_ENDPOINT_SOURCE_PATHS,
    )
    rust_rows = supervisor._materialize_snapshot(
        project,
        COMMIT,
        rust_target,
        supervisor.RUST_ENDPOINT_SOURCE_PATHS,
    )
    host_rows = supervisor._materialize_host_replay_snapshot(
        project,
        COMMIT,
        host_target,
    )
    assert (len(python_rows), len(rust_rows), len(host_rows)) == (34, 21, 1)
    all_paths = (
        *supervisor.PYTHON_ENDPOINT_SOURCE_PATHS,
        *supervisor.RUST_ENDPOINT_SOURCE_PATHS,
        *supervisor.HOST_REPLAY_SOURCE_PATHS,
    )
    blob_commands = [row for row in observed if row[:2] == ("cat-file", "blob")]
    assert blob_commands == [
        ("cat-file", "blob", f"{COMMIT}:Hegel Machine/{relative}")
        for relative in all_paths
    ]
    assert observed.count(("rev-parse", "--show-toplevel")) == 3
    for target, paths, rows in (
        (python_target, supervisor.PYTHON_ENDPOINT_SOURCE_PATHS, python_rows),
        (rust_target, supervisor.RUST_ENDPOINT_SOURCE_PATHS, rust_rows),
        (host_target, supervisor.HOST_REPLAY_SOURCE_PATHS, host_rows),
    ):
        assert tuple(row["relative_path"] for row in rows) == tuple(paths)
        assert all((target / relative).is_file() for relative in paths)
        assert not (target / "Hegel Machine").exists()


def test_snapshot_materialization_reads_real_nested_repository_commit_blob(
    native_tmp_path: Path,
) -> None:
    repository = native_tmp_path / "repository"
    project = repository / "Hegel Machine"
    source = project / "tools/example.py"
    source.parent.mkdir(parents=True)
    payload = b"print('exact nested commit blob')\n"
    source.write_bytes(payload)
    for command in (
        ["/usr/bin/git", "init", "-q", str(repository)],
        ["/usr/bin/git", "-C", str(repository), "config", "user.name", "Test"],
        [
            "/usr/bin/git",
            "-C",
            str(repository),
            "config",
            "user.email",
            "test@example.invalid",
        ],
        ["/usr/bin/git", "-C", str(repository), "add", "Hegel Machine/tools/example.py"],
        ["/usr/bin/git", "-C", str(repository), "commit", "-q", "-m", "fixture"],
    ):
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    commit = subprocess.run(
        ["/usr/bin/git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    target = native_tmp_path / "snapshot"
    rows = supervisor._materialize_snapshot(
        project,
        commit,
        target,
        ("tools/example.py",),
    )
    supervisor._verify_snapshot_manifest_v1(target, rows)
    assert (target / "tools/example.py").read_bytes() == payload
    assert not (target / "Hegel Machine").exists()
    assert rows[0]["relative_path"] == "tools/example.py"


def test_public_cli_requires_fixed_trajectory_root(tmp_path: Path) -> None:
    arguments = [
        "--run",
        "--capsule-digest",
        CAPSULE,
        "--source-commit",
        COMMIT,
        "--project-root",
        str(tmp_path / "project"),
        "--artifact",
        str(tmp_path / "artifact.json"),
        "--run-root",
        str(tmp_path / "run"),
    ]
    with pytest.raises(SystemExit):
        supervisor._parser().parse_args(arguments)
    parsed = supervisor._parser().parse_args(
        [*arguments, "--trajectory-root", str(tmp_path / "trajectory")]
    )
    assert parsed.trajectory_root == tmp_path / "trajectory"


def test_contract_per_signature_budget_derives_exact_sequence_deadline() -> None:
    contract = json.loads(
        (ROOT / supervisor.TECHNICAL_CONTRACT_RELATIVE_PATH).read_text()
    )
    supervisor._validate_contract_sequence_deadline(contract)
    assert supervisor.PER_SIGNATURE_WALL_TIME_SECONDS == 172800
    assert supervisor.ORDERED_SIGNATURE_COUNT == 2
    assert supervisor.BOUNDED_PROJECTION_WALL_TIME_SECONDS == 172800
    assert supervisor.ENDPOINT_WALL_TIME_SECONDS == 518400
    assert supervisor.ENDPOINT_SEQUENCE_WALL_TIME_SECONDS == 1036800
    assert supervisor.HOST_REPLAY_WALL_TIME_SECONDS == 172800
    assert supervisor.OUTER_SUPERVISOR_WALL_TIME_SECONDS == 1212000
    bad = deepcopy(contract)
    bad["resource_envelope"]["python_endpoint"]["wall_time_seconds"] = 518400
    with pytest.raises(supervisor.FullNode6SupervisorError) as caught:
        supervisor._validate_contract_sequence_deadline(bad)
    assert caught.value.code == "FAIL_FULL_NODE6_SUPERVISOR_SOURCE"


def test_contract_execution_source_closure_exactly_binds_34_21_55_and_host1(
) -> None:
    contract = json.loads(
        (ROOT / supervisor.TECHNICAL_CONTRACT_RELATIVE_PATH).read_text()
    )
    supervisor._validate_contract_execution_source_closure(contract)
    closure = contract["execution_source_closure"]
    assert closure["python_endpoint_row_count"] == 34
    assert closure["rust_release_build_row_count"] == 21
    assert closure["endpoint_execution_union_row_count"] == 55
    assert closure["host_replay_row_count"] == 1
    bad = deepcopy(contract)
    bad["execution_source_closure"][
        "python_endpoint_relative_paths_in_exact_order"
    ] = list(reversed(supervisor.PYTHON_ENDPOINT_SOURCE_PATHS))
    with pytest.raises(supervisor.FullNode6SupervisorError) as caught:
        supervisor._validate_contract_execution_source_closure(bad)
    assert caught.value.code == "FAIL_FULL_NODE6_SUPERVISOR_SOURCE"


def test_admission_publication_replays_execution_source_closure_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = json.loads(
        (ROOT / supervisor.TECHNICAL_CONTRACT_RELATIVE_PATH).read_text()
    )
    contract["execution_source_closure"]["endpoint_execution_union_row_count"] = 53
    monkeypatch.setattr(supervisor, "_load_technical_contract", lambda _root: contract)
    admission = tmp_path / "admission.json"
    with pytest.raises(supervisor.FullNode6SupervisorError) as caught:
        supervisor._write_admission(tmp_path, admission, CAPSULE, COMMIT)
    assert caught.value.code == "FAIL_FULL_NODE6_SUPERVISOR_SOURCE"
    assert not admission.exists()


def test_actual_sequence_timeout_removes_python_and_never_starts_rust_without_docker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HangingProcess:
        def __init__(self, actor: str) -> None:
            self.actor = actor
            self.returncode = None
            self.stopped = threading.Event()

        def communicate(self):
            self.stopped.wait(timeout=2)
            return b"", b""

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15
            self.stopped.set()

        def wait(self, timeout=None):
            if not self.stopped.wait(timeout=timeout):
                raise subprocess.TimeoutExpired(self.actor, timeout)
            return self.returncode

        def kill(self):
            self.returncode = -9
            self.stopped.set()

    processes = [HangingProcess("python")]
    process_iterator = iter(processes)
    popen_count = 0

    def popen(*_args, **_kwargs):
        nonlocal popen_count
        popen_count += 1
        return next(process_iterator)

    monkeypatch.setattr(supervisor.subprocess, "Popen", popen)
    monkeypatch.setattr(
        supervisor,
        "_wait_cid",
        lambda _path, process: "a" * 64 if process.actor == "python" else "b" * 64,
    )
    removed: list[str] = []

    def force_remove(targets):
        removed.extend(targets)
        return []

    monkeypatch.setattr(supervisor, "_force_remove_endpoint_containers_v1", force_remove)
    python_scratch = tmp_path / "python-scratch"
    rust_scratch = tmp_path / "rust-scratch"
    python_scratch.mkdir()
    rust_scratch.mkdir()
    seccomp = tmp_path / "seccomp.json"
    started: list[str] = []
    completed: list[str] = []
    with pytest.raises(supervisor.FullNode6SupervisorError) as caught:
        supervisor._launch_sequence_actual_v1(
            [
                supervisor.DOCKER,
                "run",
                "--name",
                "python-name",
                "--security-opt",
                f"seccomp={seccomp}",
                "--mount",
                f"type=bind,src={python_scratch},dst=/scratch",
            ],
            [
                supervisor.DOCKER,
                "run",
                "--name",
                "rust-name",
                "--security-opt",
                f"seccomp={seccomp}",
                "--mount",
                f"type=bind,src={rust_scratch},dst=/scratch",
            ],
            tmp_path / "python.cid",
            tmp_path / "rust.cid",
            started.append,
            lambda result: completed.append(result.actor_id),
            endpoint_wall_time_seconds=0.01,
        )
    assert caught.value.code == "FULL_NODE6_ENDPOINT_SEQUENCE_TIMEOUT"
    assert removed == ["a" * 64]
    assert popen_count == 1
    assert started == ["PYTHON_ENDPOINT"]
    assert completed == []
    assert all(process.returncode == -15 for process in processes)


def test_sequence_timeout_is_persisted_terminal_no_resume(
    native_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = native_tmp_path
    spec = _run_spec(tmp_path)
    run_root = spec.run_root
    trajectory_root = spec.trajectory_root

    def timeout(*_args):
        (run_root / "python-scratch" / "partial.bin").write_bytes(b"python")
        raise supervisor.FullNode6SupervisorError(
            "FULL_NODE6_ENDPOINT_SEQUENCE_TIMEOUT",
            "endpoint sequence exceeded its active PYTHON_ENDPOINT budget",
        )

    with pytest.raises(supervisor.FullNode6SupervisorError):
        supervisor.run_supervisor_v1(
            spec,
            launch_sequence=timeout,
            check_local_images=False,
        )
    terminal = json.loads(
        (trajectory_root / "supervisor-trajectory-terminal-v1.json").read_text()
    )
    assert terminal["error_code_or_null"] == "FULL_NODE6_ENDPOINT_SEQUENCE_TIMEOUT"
    assert (
        terminal["terminal_status"]
        == "FULL_NODE6_DIAGNOSTIC_FAILED_TERMINAL_NO_RESUME"
    )
    scratch_receipt = json.loads((run_root / "scratch-cleanup-v1.json").read_text())
    assert scratch_receipt["terminal_absence_proved"] is True
    assert not (run_root / "python-scratch").exists()
    assert not (run_root / "rust-scratch").exists()


def test_rust_started_callback_failure_retains_durable_python_evidence(
    native_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _run_spec(native_tmp_path)
    run_root = spec.run_root
    trajectory_root = spec.trajectory_root
    endpoint_launches: list[str] = []

    def launch_endpoint(_command, _cidfile, actor_id, *, wall_time_seconds):
        assert wall_time_seconds == supervisor.ENDPOINT_WALL_TIME_SECONDS
        endpoint_launches.append(actor_id)
        assert actor_id == "PYTHON_ENDPOINT"
        stdout = _write_bundle(
            run_root / "python-output",
            _diagnostic("PYTHON_ENDPOINT"),
        )
        return supervisor.EndpointProcessResultV1(
            actor_id,
            0,
            stdout,
            b"python-stderr\n",
            {
                "actor_id": actor_id,
                "container_id": "a" * 64,
                "container_exit_monotonic_ns": 200,
                "measurement_interval_start_monotonic_ns": 100,
                "memory_peak_bytes": 123,
            },
        )

    monkeypatch.setattr(
        supervisor,
        "_launch_endpoint_actual_v1",
        launch_endpoint,
    )
    original_emit = supervisor.SupervisorTrajectoryV1.emit

    def emit(self, phase, status, **kwargs):
        if phase == "RUST_ENDPOINT" and status == "STARTED":
            raise supervisor.FullNode6SupervisorError(
                "FAIL_TEST_RUST_START_CALLBACK",
                "fixture durable STARTED callback failure",
            )
        return original_emit(self, phase, status, **kwargs)

    monkeypatch.setattr(supervisor.SupervisorTrajectoryV1, "emit", emit)
    with pytest.raises(supervisor.FullNode6SupervisorError) as caught:
        supervisor.run_supervisor_v1(spec, check_local_images=False)
    assert caught.value.code == "FAIL_TEST_RUST_START_CALLBACK"
    assert endpoint_launches == ["PYTHON_ENDPOINT"]
    assert (run_root / "python.stdout").is_file()
    assert (run_root / "python.stderr").read_bytes() == b"python-stderr\n"
    assert (run_root / "python-resource-evidence-v1.json").is_file()
    assert not (run_root / "rust.stdout").exists()
    assert not (run_root / "rust-resource-evidence-v1.json").exists()
    assert not (run_root / "endpoint-resource-evidence-v1.json").exists()
    events = [
        json.loads(row)
        for row in (
            trajectory_root / "supervisor-trajectory-v1.jsonl"
        ).read_text().splitlines()
    ]
    assert any(
        row["phase"] == "PYTHON_ENDPOINT" and row["status"] == "STARTED"
        for row in events
    )
    assert any(
        row["phase"] == "PYTHON_ENDPOINT" and row["status"] == "COMPLETED"
        for row in events
    )
    terminal = json.loads(
        (trajectory_root / "supervisor-trajectory-terminal-v1.json").read_text()
    )
    assert terminal["error_code_or_null"] == "FAIL_TEST_RUST_START_CALLBACK"
    assert (
        terminal["terminal_status"]
        == "FULL_NODE6_DIAGNOSTIC_FAILED_TERMINAL_NO_RESUME"
    )


def _minimal_held_command(name: str, scratch: Path, seccomp: Path) -> list[str]:
    return [
        supervisor.DOCKER,
        "run",
        "--name",
        name,
        "--security-opt",
        f"seccomp={seccomp}",
        "--mount",
        f"type=bind,src={scratch},dst=/scratch",
    ]


def test_sequence_gives_each_endpoint_the_full_independent_deadline(
    native_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seccomp = native_tmp_path / "seccomp.json"
    budgets: list[tuple[str, float]] = []

    def launch(_command, _cidfile, actor_id, *, wall_time_seconds):
        budgets.append((actor_id, wall_time_seconds))
        if actor_id == "PYTHON_ENDPOINT":
            start, end, cid = 100, 200, "a" * 64
        else:
            start, end, cid = 200, 300, "b" * 64
        return supervisor.EndpointProcessResultV1(
            actor_id,
            0,
            b"{}\n",
            b"",
            {
                "actor_id": actor_id,
                "container_id": cid,
                "container_exit_monotonic_ns": end,
                "measurement_interval_start_monotonic_ns": start,
            },
        )

    monkeypatch.setattr(supervisor, "_launch_endpoint_actual_v1", launch)
    supervisor._launch_sequence_actual_v1(
        _minimal_held_command(
            "python",
            native_tmp_path / "python-scratch",
            seccomp,
        ),
        _minimal_held_command(
            "rust",
            native_tmp_path / "rust-scratch",
            seccomp,
        ),
        native_tmp_path / "python.cid",
        native_tmp_path / "rust.cid",
        lambda _actor_id: None,
        lambda _result: None,
    )
    assert budgets == [
        ("PYTHON_ENDPOINT", 518400),
        ("RUST_ENDPOINT", 518400),
    ]


def test_sequence_resource_capture_and_measurement_intervals_do_not_overlap_without_docker(
    native_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python_scratch = native_tmp_path / "python-scratch"
    rust_scratch = native_tmp_path / "rust-scratch"
    python_scratch.mkdir()
    rust_scratch.mkdir()
    seccomp = native_tmp_path / "seccomp.json"

    class HeldProcess:
        def __init__(self, actor: str, scratch: Path) -> None:
            self.actor = actor
            self.scratch = scratch
            self.returncode = None
            (scratch / supervisor.ACTOR_RETURN_CODE_FILENAME).write_text("0\n")
            (scratch / supervisor.ACTOR_READY_FILENAME).write_bytes(b"")

        def communicate(self):
            deadline = time.monotonic() + 2
            while not (self.scratch / supervisor.SUPERVISOR_RELEASE_FILENAME).exists():
                assert time.monotonic() < deadline
                time.sleep(0.001)
            self.returncode = 0
            return self.actor.encode("ascii") + b"\n", b""

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    python_process = HeldProcess("python", python_scratch)
    rust_process = HeldProcess("rust", rust_scratch)
    processes = iter((python_process, rust_process))
    popen_order: list[str] = []

    def popen(*_args, **_kwargs):
        process = next(processes)
        if process.actor == "rust":
            assert python_process.returncode == 0
            assert (python_scratch / supervisor.SUPERVISOR_RELEASE_FILENAME).is_file()
        popen_order.append(process.actor)
        return process

    monkeypatch.setattr(supervisor.subprocess, "Popen", popen)
    monkeypatch.setattr(
        supervisor,
        "_wait_cid",
        lambda _path, process: "a" * 64 if process.actor == "python" else "b" * 64,
    )
    captures: list[str] = []

    def capture(cid, actor, _seccomp):
        scratch = python_scratch if actor == "PYTHON_ENDPOINT" else rust_scratch
        assert (scratch / supervisor.ACTOR_READY_FILENAME).is_file()
        assert not (scratch / supervisor.SUPERVISOR_RELEASE_FILENAME).exists()
        captures.append(actor)
        return {
            "actor_id": actor,
            "container_id": cid,
            "memory_peak_bytes": 123,
        }

    monkeypatch.setattr(supervisor, "_capture_resource_evidence", capture)
    python, rust = supervisor._launch_sequence_actual_v1(
        _minimal_held_command("python", python_scratch, seccomp),
        _minimal_held_command("rust", rust_scratch, seccomp),
        native_tmp_path / "python.cid",
        native_tmp_path / "rust.cid",
        lambda _actor_id: None,
        lambda _result: None,
        endpoint_wall_time_seconds=1,
    )
    assert popen_order == ["python", "rust"]
    assert captures == ["PYTHON_ENDPOINT", "RUST_ENDPOINT"]
    for result in (python, rust):
        evidence = result.resource_evidence
        assert evidence["memory_peak_bytes"] == 123
        assert (
            evidence["measurement_interval_start_monotonic_ns"]
            <= evidence["actor_exit_after_output_monotonic_ns"]
            <= evidence["cgroup_capture_monotonic_ns"]
            <= evidence["container_hold_release_monotonic_ns"]
            <= evidence["container_exit_monotonic_ns"]
        )
        assert evidence["measurement_wall_time_ns"] == (
            evidence["container_exit_monotonic_ns"]
            - evidence["measurement_interval_start_monotonic_ns"]
        )
    assert (
        python.resource_evidence["container_exit_monotonic_ns"]
        <= rust.resource_evidence["measurement_interval_start_monotonic_ns"]
    )
    assert python.resource_evidence["endpoint_sequence_ordinal"] == 1
    assert rust.resource_evidence["endpoint_sequence_ordinal"] == 2
    assert rust.resource_evidence["predecessor_container_exit_monotonic_ns"] == (
        python.resource_evidence["container_exit_monotonic_ns"]
    )
    assert all(
        row.resource_evidence["maximum_concurrently_live_endpoint_containers"]
        == 1
        for row in (python, rust)
    )


def test_host_replay_resource_capture_uses_same_completed_workload_handshake(
    native_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = native_tmp_path / "host-scratch"
    scratch.mkdir()
    seccomp = native_tmp_path / "seccomp.json"

    class HeldHostProcess:
        returncode = None

        def __init__(self) -> None:
            (scratch / supervisor.ACTOR_RETURN_CODE_FILENAME).write_text("0\n")
            (scratch / supervisor.ACTOR_READY_FILENAME).write_bytes(b"")

        def communicate(self):
            deadline = time.monotonic() + 2
            while not (scratch / supervisor.SUPERVISOR_RELEASE_FILENAME).exists():
                assert time.monotonic() < deadline
                time.sleep(0.001)
            self.returncode = 0
            return b"{}\n", b""

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    process = HeldHostProcess()
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(supervisor, "_wait_cid", lambda *_args: "c" * 64)

    def capture(cid, actor, _seccomp):
        assert actor == "TRUSTED_HOST_REPLAY"
        assert (scratch / supervisor.ACTOR_READY_FILENAME).exists()
        assert not (scratch / supervisor.SUPERVISOR_RELEASE_FILENAME).exists()
        return {"actor_id": actor, "container_id": cid, "memory_peak_bytes": 456}

    monkeypatch.setattr(supervisor, "_capture_resource_evidence", capture)
    result = supervisor._launch_host_replay_actual_v1(
        _minimal_held_command("host-replay", scratch, seccomp),
        native_tmp_path / "host.cid",
        wall_time_seconds=1,
    )
    assert result.return_code == 0
    assert result.resource_evidence["memory_peak_bytes"] == 456
    assert result.resource_evidence["actor_id"] == "TRUSTED_HOST_REPLAY"
    assert result.resource_evidence["measurement_wall_time_ns"] == (
        result.resource_evidence["container_exit_monotonic_ns"]
        - result.resource_evidence["measurement_interval_start_monotonic_ns"]
    )


def test_resource_evidence_binds_frozen_isolation_and_fresh_memory_peak(
    native_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seccomp = native_tmp_path / "seccomp.json"
    inspect = [
        {
            "Config": {"Image": supervisor.PYTHON_IMAGE, "User": "65534:65534"},
            "HostConfig": {
                "CapDrop": ["ALL"],
                "CpusetCpus": "0-11",
                "IpcMode": "private",
                "Memory": supervisor.MEMORY_BYTES,
                "MemorySwap": supervisor.MEMORY_BYTES,
                "NetworkMode": "none",
                "PidsLimit": 128,
                "ReadonlyRootfs": True,
                "SecurityOpt": ["no-new-privileges", f"seccomp={seccomp}"],
                "Tmpfs": {"/tmp": supervisor.TMPFS_SPEC.split(":", 1)[1]},
                "Ulimits": [{"Name": "nofile", "Hard": 256, "Soft": 256}],
            },
            "Mounts": [
                {"Destination": "/admission/admission.json", "RW": False, "Type": "bind"},
                {"Destination": "/output", "RW": True, "Type": "bind"},
                {"Destination": "/scratch", "RW": True, "Type": "bind"},
                {"Destination": "/snapshot", "RW": False, "Type": "bind"},
            ],
            "State": {"Pid": 1234},
        }
    ]
    monkeypatch.setattr(
        supervisor,
        "_run_capture",
        lambda *_args, **_kwargs: json.dumps(inspect).encode("ascii"),
    )
    monkeypatch.setattr(
        supervisor,
        "_read_cgroup_evidence",
        lambda _pid: {
            "cgroup_path": "/fresh-container.scope",
            "rows": {"memory.peak": "987654\n"},
        },
    )
    evidence = supervisor._capture_resource_evidence(
        "d" * 64,
        "PYTHON_ENDPOINT",
        seccomp,
    )
    assert evidence["memory_peak_bytes"] == 987654
    assert evidence["cgroup_capture_phase"] == "AFTER_ACTOR_OUTPUT_BEFORE_CONTAINER_RELEASE"
    assert evidence["runtime_seccomp_sha256"] == supervisor.SECCOMP_SHA256
    assert evidence["tmpfs"] == supervisor.TMPFS_SPEC


def _resource_process(actor: str, ordinal: int, memory: int, wall_ns: int):
    return supervisor.EndpointProcessResultV1(
        actor,
        0,
        b"{}\n",
        b"",
        {
            "actor_id": actor,
            "cgroup": {"cgroup_path": f"/{actor.lower()}-{ordinal}.scope"},
            "container_id": f"{ordinal:064x}",
            "measurement_wall_time_ns": wall_ns,
            "memory_peak_bytes": memory,
        },
    )


def _resource_replay() -> dict[str, object]:
    return {
        "endpoint_rows": [
            {
                "actor_id": "PYTHON_ENDPOINT",
                "actual_diagnostic_output_bytes": 1000,
                "actual_preflight_peak_scratch_bytes": 2000,
            },
            {
                "actor_id": "RUST_ENDPOINT",
                "actual_diagnostic_output_bytes": 1200,
                "actual_preflight_peak_scratch_bytes": 2200,
            },
        ],
        "formal_resource_projection": {
            "projected_formal_total_output_bytes": 1_000_000,
            "projected_peak_formal_scratch_bytes": 2_000_000,
        },
        "projected_host_replay_output_bytes": 1_048_576,
        "projected_peak_host_replay_scratch_bytes": 0,
        "registered_resource_guard_id": None,
        "terminal_status": host.SUCCESS_STATUS,
    }


def test_resource_adjudication_uses_lowest_direct_guard_id() -> None:
    replay = _resource_replay()
    replay["endpoint_rows"][0]["actual_diagnostic_output_bytes"] = (
        supervisor.ENDPOINT_OUTPUT_CEILING_BYTES + 1
    )
    replay["endpoint_rows"][1]["actual_preflight_peak_scratch_bytes"] = (
        supervisor.ENDPOINT_SCRATCH_CEILING_BYTES + 1
    )
    result = supervisor._adjudicate_resources_v1(
        replay,
        _resource_process("PYTHON_ENDPOINT", 1, 1000, 1_000_000_000),
        _resource_process("RUST_ENDPOINT", 2, 1000, 1_000_000_000),
        _resource_process("TRUSTED_HOST_REPLAY", 3, 1000, 1_000_000_000),
    )
    assert result["terminal_status"] == host.GUARD_STATUS
    assert result["first_resource_guard_id"] == 9
    assert result["violated_direct_resource_guard_ids"] == [9, 10]


def test_resource_adjudication_routes_derived_ceiling_to_infeasible() -> None:
    replay = _resource_replay()
    replay["formal_resource_projection"]["projected_formal_total_output_bytes"] = (
        20_000_000_000
    )
    result = supervisor._adjudicate_resources_v1(
        replay,
        _resource_process("PYTHON_ENDPOINT", 1, 1000, 1_000_000_000),
        _resource_process("RUST_ENDPOINT", 2, 1000, 1_000_000_000),
        _resource_process("TRUSTED_HOST_REPLAY", 3, 1000, 1_000_000_000),
    )
    assert result["terminal_status"] == supervisor.INFEASIBLE_STATUS
    assert "endpoint_output_guard_bytes" in result["infeasible_derived_fields"]
    assert result["host_replay_passed"] is True
    assert result["q1_state"] == "NOT_RUN"
    assert supervisor._terminal_exit_code_v1(result["terminal_status"]) == 3
    assert supervisor._terminal_exit_code_v1(host.SUCCESS_STATUS) == 0
    assert supervisor._terminal_exit_code_v1(host.GUARD_STATUS) == 0


def test_standalone_host_replay_cli_is_target_blind_and_leaves_scratch_empty(
    native_tmp_path: Path,
) -> None:
    python_root = native_tmp_path / "python-output"
    rust_root = native_tmp_path / "rust-output"
    _write_bundle(python_root, _diagnostic("PYTHON_ENDPOINT"))
    _write_bundle(rust_root, _diagnostic("RUST_ENDPOINT", 5))
    scratch = native_tmp_path / "scratch"
    scratch.mkdir()
    completed = subprocess.run(
        [
            "/usr/bin/python3.10",
            "-I",
            "-S",
            "-B",
            str(ROOT / supervisor.HOST_REPLAY_RELATIVE_PATH),
            "--run",
            "--python-output",
            str(python_root),
            "--rust-output",
            str(rust_root),
            "--capsule-digest",
            CAPSULE,
            "--source-commit",
            COMMIT,
            "--scratch-dir",
            str(scratch),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert json.loads(completed.stdout)["terminal_status"] == host.SUCCESS_STATUS
    assert not any(scratch.iterdir())
