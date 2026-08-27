from __future__ import annotations

from copy import deepcopy
import dis
from hashlib import sha256
import importlib
import importlib.util
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import threading
from typing import Mapping, Sequence

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CORE = importlib.import_module(
    "hegel_machine.phase3_q05b_shadow_retry_campaign_v2"
)
TRAJECTORY = importlib.import_module("hegel_machine.phase3_q05b_trajectory_v2")

TOOL_PATH = ROOT / "tools/phase3_q05b_shadow_retry_campaign_v2.py"
SPEC = importlib.util.spec_from_file_location(
    "phase3_q05b_shadow_retry_campaign_v2_tool_test",
    TOOL_PATH,
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)

_REAL_REQUIRE_SINGLE_THREADED_SPAWN = (
    TOOL._require_single_threaded_spawn_controller_v2
)


@pytest.fixture(autouse=True)
def _permit_pytest_worker_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The production CLI is single-threaded; this pytest host is not."""

    monkeypatch.setattr(
        TOOL, "_require_single_threaded_spawn_controller_v2", lambda: None
    )


_STAGES = (
    "FREEZE_SOURCE_IMAGES_AND_COMMAND_REGISTRY",
    "MATERIALIZE_AND_SEAL_THREE_SOURCE_SNAPSHOTS_AND_CARGO",
    "OFFLINE_RUST_TEST_AND_RELEASE_BUILD_AND_SEAL_BINARY",
    "RUN_TWO_HELD_ENDPOINTS_IN_PARALLEL",
    "SEAL_ENDPOINT_STDOUT_SIDECARS_CONTROLS_AND_RESOURCES",
    "RUN_HELD_TRUSTED_HOST_SEMANTIC_REPLAY",
    "CLOSE_HOST_RESOURCE_ISOLATION_NEGATIVE_VECTORS_AND_TOCTOU",
    "BUILD_AND_STRICTLY_DECODE_NINETEEN_ROW_CANDIDATE",
    "ADD_PREDICATE20_AND_BUILD_CANONICAL_ARTIFACT_IN_MEMORY",
    "STRICTLY_REPLAY_COMPLETE_ARTIFACT_BEFORE_PUBLICATION",
)

_SAMPLE_FIELDS = (
    "role_id_or_null",
    "sample_ordinal_or_null",
    "previous_finish_ns_or_null",
    "scheduled_start_ns_or_null",
    "actual_start_ns_or_null",
    "finish_ns_or_null",
    "collection_duration_ns_or_null",
    "scheduler_lateness_ns_or_null",
    "target_wait_ns_or_null",
    "gap_ns_or_null",
    "maximum_gap_ns_or_null",
    "gap_exceeded_or_null",
)


def _raw(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _tagged(label: str) -> str:
    return "sha256:" + _raw(label)


def _contract() -> dict[str, object]:
    raw = TOOL.CONFIG_PATH.read_bytes()
    value = TOOL._pretty_json(raw, "campaign test contract")
    assert type(value) is dict
    contract = deepcopy(value)
    contract["contract_digest"] = _tagged("contract")
    binding = contract["campaign_source_binding"]
    assert type(binding) is dict
    binding["source_binding_digest"] = _tagged("campaign-source")
    binding["core_raw_sha256"] = sha256(
        (SRC / "hegel_machine/phase3_q05b_shadow_retry_campaign_v2.py")
        .read_bytes()
    ).hexdigest()
    binding["tool_raw_sha256"] = _raw("tool-raw")
    target = contract["target_commit_a_binding"]
    assert type(target) is dict
    target["target_supervisor_commit"] = "12" * 20
    required = target["required_raw_sha256_by_path"]
    assert type(required) is dict
    for index, relative in enumerate(required, start=1):
        required[relative] = _raw(f"target-{index}")
    retry = contract["retry_policy"]
    success = contract["success_policy"]
    forbidden = contract["forbidden_scope"]
    assert type(retry) is dict and type(success) is dict
    assert type(forbidden) is dict
    retry["retry_predicate_digest"] = _tagged("retry")
    success["success_predicate_digest"] = _tagged("success")
    forbidden["forbidden_scope_digest"] = _tagged("forbidden")
    return contract


def _slot(
    contract: Mapping[str, object],
    ordinal: int = 1,
) -> dict[str, object]:
    slots = contract["slot_registry"]
    assert type(slots) is list
    value = slots[ordinal - 1]
    assert type(value) is dict
    return deepcopy(value)


def test_prepare_clone_rebuilds_reachable_graph_without_source_only_objects(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    temporary = tempfile.TemporaryDirectory(
        prefix="hegel-q05b-clone-graph-kat-",
        dir="/tmp",
    )
    request.addfinalizer(temporary.cleanup)
    tmp_path = Path(temporary.name)
    git = "/usr/bin/git"
    source = tmp_path / "source"
    subprocess.run(
        [git, "init", "--initial-branch=main", os.fspath(source)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    def source_git(*arguments: str) -> bytes:
        result = subprocess.run(
            [git, "-C", os.fspath(source), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout

    source_git("config", "user.name", "Q05b KAT")
    source_git("config", "user.email", "q05b-kat@example.invalid")
    project = source / ROOT.name
    contract = _contract()
    target = contract["target_commit_a_binding"]
    execution = contract["execution_policy"]
    assert type(target) is dict and type(execution) is dict
    required = target["required_raw_sha256_by_path"]
    assert type(required) is dict
    for index, relative in enumerate(required, start=1):
        path = project / str(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture-{index}-parent\n".encode("ascii"))
    source_git("add", ROOT.name)
    source_git("commit", "--quiet", "-m", "fixture parent")
    parent = source_git("rev-parse", "HEAD").decode("ascii").strip()
    first_required = project / str(next(iter(required)))
    first_required.write_bytes(b"fixture-target\n")
    source_git("add", ROOT.name)
    source_git("commit", "--quiet", "-m", "fixture target")
    target_commit = source_git("rev-parse", "HEAD").decode("ascii").strip()
    target_tree = source_git("rev-parse", "HEAD^{tree}").decode("ascii").strip()

    source_git("checkout", "--quiet", "-b", "source-only-ref")
    first_required.write_bytes(b"fixture-source-only-ref\n")
    source_git("add", ROOT.name)
    source_git("commit", "--quiet", "-m", "source-only ref commit")
    source_only_ref = source_git("rev-parse", "HEAD").decode("ascii").strip()
    source_git("checkout", "--quiet", "main")
    first_required.write_bytes(b"fixture-stash-only\n")
    source_git("add", ROOT.name)
    source_git("stash", "push", "--quiet", "-m", "source-only stash")
    source_only_stash = source_git("rev-parse", "refs/stash").decode(
        "ascii"
    ).strip()
    source_git("commit-graph", "write", "--reachable")
    source_git("commit-graph", "verify")
    source_graph = (source / ".git/objects/info/commit-graph").read_bytes()
    assert bytes.fromhex(source_only_ref) in source_graph
    assert bytes.fromhex(source_only_stash) in source_graph
    assert source_git("status", "--porcelain=v1", "-z") == b""

    execution["clone_source_repository"] = os.fspath(source)
    execution["clone_source_project_relative_path"] = ROOT.name
    target["target_supervisor_commit"] = target_commit
    target["target_supervisor_tree"] = target_tree
    target["target_supervisor_parent_commit"] = parent
    for relative in required:
        required[str(relative)] = sha256(
            (project / str(relative)).read_bytes()
        ).hexdigest()

    slot = contract["slot_registry"][0]
    assert type(slot) is dict
    clone = tmp_path / "slot-01-clone"
    clone_project = clone / ROOT.name
    artifact_parent = tmp_path / "slot-01-artifact"
    evidence = tmp_path / "slot-01-evidence"
    trace = tmp_path / "slot-01-trace"
    slot.update(
        {
            "clone_root": os.fspath(clone),
            "clone_project_root": os.fspath(clone_project),
            "cargo_cache_root": os.fspath(tmp_path / "slot-01-cargo"),
            "supervisor_artifact_path": os.fspath(
                clone_project
                / "artifacts/phase3_q05b_dual_qualification_v2.json"
            ),
            "artifact_parent_root": os.fspath(artifact_parent),
            "artifact_path": os.fspath(
                artifact_parent / "phase3_q05b_dual_qualification_v2.json"
            ),
            "evidence_root": os.fspath(evidence),
            "trace_root": os.fspath(trace),
            "stdout_path": os.fspath(evidence / "supervisor.stdout"),
            "stderr_path": os.fspath(evidence / "supervisor.stderr"),
            "external_trajectory_binding_path": os.fspath(
                evidence / "trajectory-binding.json"
            ),
            "slot_result_path": os.fspath(evidence / "slot-result.json"),
        }
    )

    calls: list[tuple[str, list[str], dict[str, str]]] = []
    real_run_exact_command = TOOL._run_exact_command

    def recording_run_exact_command(
        argv: Sequence[str],
        **kwargs: object,
    ) -> bytes:
        environment = kwargs["env"]
        assert type(environment) is dict
        calls.append(
            (
                str(kwargs["label"]),
                list(argv),
                dict(environment),
            )
        )
        return real_run_exact_command(argv, **kwargs)

    monkeypatch.setattr(TOOL, "_run_exact_command", recording_run_exact_command)
    identity = TOOL._prepare_clone(contract, slot)
    assert identity["head"] == target_commit
    assert identity["tree"] == target_tree
    assert identity["parent"] == parent
    expected_target_ref = str(execution["clone_target_ref_name"])
    exact_ref = subprocess.run(
        [
            git,
            "-C",
            os.fspath(clone),
            "for-each-ref",
            "--format=%(refname)%00%(objectname)",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert exact_ref.stdout == (
        expected_target_ref + "\x00" + target_commit + "\n"
    ).encode("ascii")
    assert exact_ref.stderr == b""
    replay_identity = TOOL._verify_clone(contract, slot, artifact_state="ABSENT")
    assert replay_identity == identity

    clone_graph = (clone / ".git/objects/info/commit-graph").read_bytes()
    assert bytes.fromhex(source_only_ref) not in clone_graph
    assert bytes.fromhex(source_only_stash) not in clone_graph
    for stale_oid in (source_only_ref, source_only_stash):
        absent = subprocess.run(
            [git, "-C", os.fspath(clone), "cat-file", "-e", stale_oid],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert absent.returncode != 0
        assert absent.stdout == b""
    for command in (
        (git, "-C", os.fspath(clone), "commit-graph", "verify"),
        (
            git,
            "-C",
            os.fspath(clone),
            "fsck",
            "--full",
            "--strict",
            "--unreachable",
            "--no-reflogs",
        ),
        (
            git,
            "-c",
            "core.commitGraph=false",
            "-C",
            os.fspath(clone),
            "fsck",
            "--full",
            "--strict",
            "--unreachable",
            "--no-reflogs",
        ),
    ):
        replay = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert replay.returncode == 0
        assert replay.stdout == b""
        assert replay.stderr == b""

    assert calls
    for label, argv, environment in calls:
        assert TOOL._bounded_command_binding_is_exact_v2(contract, label, argv)
        assert environment["GIT_ALLOW_PROTOCOL"] == "file"
        assert environment["GIT_PROTOCOL_FROM_USER"] == "0"
        assert environment["GIT_SSH_COMMAND"] == "/bin/false"
    clone_call = next(argv for label, argv, _env in calls if label == "slot no-local clone")
    assert "--no-local" in clone_call
    assert "--local" not in clone_call


def _success_wire(
    contract: Mapping[str, object],
    slot: Mapping[str, object],
) -> dict[str, object]:
    stage_roots = [_raw(f"stage-{index}") for index in range(1, 11)]
    result: dict[str, object] = {
        "artifact_path": slot["supervisor_artifact_path"],
        "artifact_sha256": _raw("artifact"),
        "artifact_set_root": _raw("artifact-set"),
        "candidate_receipt_root": _raw("candidate-receipt"),
        "final_receipt_root": _raw("final-receipt"),
        "qualification_count": 20,
        "qualification_mask": 0xFFFFF,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_formal_output_roots": [None] * 8,
        "actual_admission_prepublication_live_replay_root": _raw(
            "prepublication-live"
        ),
        "actual_admission_pre_artifact_live_replay_root": _raw(
            "pre-artifact-live"
        ),
        "actual_admission_stage10_live_replay_root": _raw("stage10-live"),
        "actual_admission_postpublication_live_replay_root": _raw(
            "postpublication-live"
        ),
        "actual_final_delivery_identity": None,
        "actual_final_delivery_root": "0" * 64,
        "publisher_handle_close_status": "CLOSED",
        "stage_roots": stage_roots,
    }
    target = contract["target_commit_a_binding"]
    assert type(target) is dict
    identity = {
        "schema_version": "hegel-phase3a-q05b-final-delivery-identity/2",
        "handoff_point": (
            "FINAL_ANCHORED_REPLAY_AFTER_POSTPUBLICATION_LIVE_"
            "BEFORE_HANDLE_CLOSE_AND_RETURN"
        ),
        "source_commit": target["target_supervisor_commit"],
        "artifact_path": result["artifact_path"],
        "ordered_stage_root_rows": [
            [index, name, stage_roots[index - 1]]
            for index, name in enumerate(_STAGES, start=1)
        ],
        "actual_admission_evidence_root": _raw("actual-admission"),
        "pre_artifact_live_marker_replay_root": result[
            "actual_admission_pre_artifact_live_replay_root"
        ],
        "stage_10_live_marker_replay_root": result[
            "actual_admission_stage10_live_replay_root"
        ],
        "prepublication_live_marker_replay_root": result[
            "actual_admission_prepublication_live_replay_root"
        ],
        "postpublication_live_marker_replay_root": result[
            "actual_admission_postpublication_live_replay_root"
        ],
        "artifact_set_root": result["artifact_set_root"],
        "candidate_receipt_root": result["candidate_receipt_root"],
        "final_receipt_root": result["final_receipt_root"],
        "publisher_parent_device": 2049,
        "publisher_parent_inode": 100,
        "publisher_file_device": 2049,
        "publisher_file_inode": 101,
        "publisher_file_mode": 0o444,
        "publisher_file_nlink": 1,
        "payload_length": 4096,
        "payload_sha256": result["artifact_sha256"],
        "first_anchored_replay_sha256": result["artifact_sha256"],
        "final_path_matches_publisher_owned_inode": True,
        "same_uid_mutation_after_handoff_outside_claim": True,
    }
    result["actual_final_delivery_identity"] = identity
    result["actual_final_delivery_root"] = sha256(
        b"HEGEL/Q05B/ACTUAL/FINAL_DELIVERY/V1\x00"
        + CORE.canonical_campaign_json_bytes_v2(identity)
    ).hexdigest()
    return result


def _error_wire(detail: str) -> dict[str, object]:
    return {
        "schema_version": CORE.ERROR_WIRE_SCHEMA_V2,
        "status": CORE.GAP_ERROR_CODE_V2,
        "detail": detail,
        "qualification_predicate_count": 0,
        "qualification_predicate_mask": 0,
        "qualification_predicate_total": 20,
        "qualification_candidate_receipt": None,
        "qualification_final_receipt": None,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_formal_output_roots": [None] * 8,
        "q1_receipt": None,
        "q2_state": "NOT_RUN",
        "m3_formal_roots": None,
        "receipt_created": False,
        "artifact_written": False,
    }


def _propagation_row(
    *,
    sequence: int,
    timestamp: int,
    operation: str,
    span_id: str,
    stage: str | None,
    substep: str | None,
    actor: str | None,
    detail: str,
    error_detail: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "sequence": sequence,
        "timestamp_ns": timestamp,
        "span_id": span_id,
        "operation": operation,
        "stage_or_null": stage,
        "substep_or_null": substep,
        "actor_or_null": actor,
        "detail_or_null": detail,
        "error_code": CORE.GAP_ERROR_CODE_V2,
        "error_detail": error_detail,
        "record_digest": _tagged(f"failure-{sequence}"),
    }
    row.update({field: None for field in _SAMPLE_FIELDS})
    return row


def _held_final_gap_rows() -> tuple[str, list[dict[str, object]]]:
    previous = 1_000_000_000
    scheduled = previous + CORE.TARGET_WAIT_NS_V2
    actual = previous + CORE.MAXIMUM_GAP_NS_V2 + 1
    ordinal = 2
    detail = (
        "held final sampling gap exceeded bound: "
        "actor_id=RUST_ENDPOINT; role_id=2; sample_ordinal=2; "
        f"previous_finish_ns={previous}; actual_start_ns={actual}; "
        f"gap_ns={actual - previous}; "
        f"maximum_gap_ns={CORE.MAXIMUM_GAP_NS_V2}; "
        f"target_wait_ns={CORE.TARGET_WAIT_NS_V2}; "
        "previous_duration_ns=1000000"
    )
    origin: dict[str, object] = {
        "sequence": 5,
        "timestamp_ns": actual + 5,
        "span_id": f"held-final-sample-RUST_ENDPOINT-{ordinal:06d}",
        "operation": "HELD_FINAL_RESOURCE_SAMPLE",
        "stage_or_null": "ACTOR_LIFECYCLE",
        "substep_or_null": "COLLECT_BOUND_HELD_FINAL_RESOURCES",
        "actor_or_null": "RUST_ENDPOINT",
        "detail_or_null": "MAXIMUM_UNSAMPLED_GAP_EXCEEDED",
        "error_code": CORE.GAP_ERROR_CODE_V2,
        "error_detail": detail,
        "record_digest": _tagged("failure-5"),
        "role_id_or_null": "2",
        "sample_ordinal_or_null": ordinal,
        "previous_finish_ns_or_null": previous,
        "scheduled_start_ns_or_null": scheduled,
        "actual_start_ns_or_null": actual,
        "finish_ns_or_null": actual,
        "collection_duration_ns_or_null": 0,
        "scheduler_lateness_ns_or_null": actual - scheduled,
        "target_wait_ns_or_null": CORE.TARGET_WAIT_NS_V2,
        "gap_ns_or_null": actual - previous,
        "maximum_gap_ns_or_null": CORE.MAXIMUM_GAP_NS_V2,
        "gap_exceeded_or_null": True,
    }
    rows = [origin]
    rows.append(
        _propagation_row(
            sequence=6,
            timestamp=actual + 6,
            operation="HELD_ACTOR_CLOSE",
            span_id="actor-close-RUST_ENDPOINT",
            stage="ACTOR_LIFECYCLE",
            substep="DONE_FINAL_SAMPLE_RELEASE_EXIT_REMOVE_ABSENCE",
            actor="RUST_ENDPOINT",
            detail="ACTOR_CLOSE_FAILED_BEFORE_CLEANUP",
            error_detail=detail,
        )
    )
    rows.append(
        _propagation_row(
            sequence=9,
            timestamp=actual + 9,
            operation="SUPERVISOR_STAGE",
            span_id="supervisor-stage-05",
            stage="STAGE_05",
            substep="SEAL_ENDPOINT_STDOUT_SIDECARS_CONTROLS_AND_RESOURCES",
            actor=None,
            detail="FAILED",
            error_detail=detail,
        )
    )
    rows.append(
        _propagation_row(
            sequence=10,
            timestamp=actual + 10,
            operation="SUPERVISOR_RUN",
            span_id="supervisor-run",
            stage="RUN",
            substep=None,
            actor=None,
            detail="supervisor attempt failed",
            error_detail=detail,
        )
    )
    return detail, rows


def _continuous_gap_rows() -> tuple[str, list[dict[str, object]]]:
    previous = 1_000_000_000
    scheduled = previous + CORE.TARGET_WAIT_NS_V2
    actual = previous + CORE.MAXIMUM_GAP_NS_V2 + 1
    raw_detail = (
        "held actor sampling gap exceeded bound: "
        "actor_id=RUST_ENDPOINT; role_id=2; sample_ordinal=1; "
        f"previous_finish_ns={previous}; scheduled_start_ns={scheduled}; "
        f"actual_start_ns={actual}; gap_ns={actual - previous}; "
        f"maximum_gap_ns={CORE.MAXIMUM_GAP_NS_V2}; "
        f"target_wait_ns={CORE.TARGET_WAIT_NS_V2}; "
        "previous_duration_ns=1000000"
    )
    propagated_detail = "held actor sampler failed: " + raw_detail
    origin: dict[str, object] = {
        "sequence": 6,
        "timestamp_ns": actual + 6,
        "span_id": "sample-RUST_ENDPOINT-000001",
        "operation": "LIVE_RESOURCE_SAMPLE",
        "stage_or_null": "ACTOR_LIFECYCLE",
        "substep_or_null": "COLLECT_BOUND_LIVE_RESOURCES",
        "actor_or_null": "RUST_ENDPOINT",
        "detail_or_null": "MAXIMUM_UNSAMPLED_GAP_EXCEEDED",
        "error_code": CORE.GAP_ERROR_CODE_V2,
        "error_detail": raw_detail,
        "record_digest": _tagged("failure-6"),
        "role_id_or_null": "2",
        "sample_ordinal_or_null": 1,
        "previous_finish_ns_or_null": previous,
        "scheduled_start_ns_or_null": scheduled,
        "actual_start_ns_or_null": actual,
        "finish_ns_or_null": actual,
        "collection_duration_ns_or_null": 0,
        "scheduler_lateness_ns_or_null": actual - scheduled,
        "target_wait_ns_or_null": CORE.TARGET_WAIT_NS_V2,
        "gap_ns_or_null": actual - previous,
        "maximum_gap_ns_or_null": CORE.MAXIMUM_GAP_NS_V2,
        "gap_exceeded_or_null": True,
    }
    contexts = (
        (
            "LIVE_RESOURCE_SAMPLER",
            "sampler-RUST_ENDPOINT",
            "ACTOR_LIFECYCLE",
            "CID_TO_HELD_FINAL_CONTINUOUS_SAMPLING",
            "RUST_ENDPOINT",
            "MAXIMUM_UNSAMPLED_GAP_EXCEEDED; "
            "sample_span=sample-RUST_ENDPOINT-000001",
            raw_detail,
        ),
        (
            "HELD_ACTOR_CLOSE",
            "actor-close-RUST_ENDPOINT",
            "ACTOR_LIFECYCLE",
            "DONE_FINAL_SAMPLE_RELEASE_EXIT_REMOVE_ABSENCE",
            "RUST_ENDPOINT",
            "ACTOR_CLOSE_FAILED_BEFORE_CLEANUP",
            propagated_detail,
        ),
        (
            "SUPERVISOR_STAGE",
            "supervisor-stage-05",
            "STAGE_05",
            "SEAL_ENDPOINT_STDOUT_SIDECARS_CONTROLS_AND_RESOURCES",
            None,
            "FAILED",
            propagated_detail,
        ),
        (
            "SUPERVISOR_RUN",
            "supervisor-run",
            "RUN",
            None,
            None,
            "supervisor attempt failed",
            propagated_detail,
        ),
    )
    rows = [origin]
    for sequence, context in zip((7, 8, 11, 12), contexts, strict=True):
        operation, span_id, stage, substep, actor, event_detail, error_detail = (
            context
        )
        rows.append(
            _propagation_row(
                sequence=sequence,
                timestamp=actual + sequence,
                operation=operation,
                span_id=span_id,
                stage=stage,
                substep=substep,
                actor=actor,
                detail=event_detail,
                error_detail=error_detail,
            )
        )
    return propagated_detail, rows


def _event_record(
    *,
    trace_id: str,
    sequence: int,
    timestamp: int,
    event_type: str,
    span_id: str,
    operation: str,
    stage: str | None,
    substep: str | None,
    actor: str | None,
    detail: str | None,
    error_code: str | None = None,
    error_detail: str | None = None,
    record_digest: str | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": TRAJECTORY.EVENT_SCHEMA_V2,
        "trace_id": trace_id,
        "producer_id": "Q05B_SUPERVISOR_V2",
        "clock_id": "CLOCK_MONOTONIC",
        "sequence": sequence,
        "timestamp_ns": timestamp,
        "event_type": event_type,
        "span_id": span_id,
        "operation": operation,
        "stage_or_null": stage,
        "substep_or_null": substep,
        "actor_or_null": actor,
        "detail_or_null": detail,
        "error_code_or_null": error_code,
        "error_detail_or_null": error_detail,
        "record_digest": record_digest
        or _tagged(f"event-record-{sequence}-{event_type}"),
    }
    value.update({field: None for field in _SAMPLE_FIELDS})
    return value


def _event_records_for_failures(
    trace_id: str,
    failure_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = [deepcopy(dict(row)) for row in failure_rows]
    assert rows and rows[0]["sequence"] == len(rows) + 1
    origin = rows[0]
    timestamp_base = int(origin["timestamp_ns"]) - int(origin["sequence"])
    events: list[dict[str, object]] = []
    for sequence, failure in enumerate(reversed(rows), start=1):
        events.append(
            _event_record(
                trace_id=trace_id,
                sequence=sequence,
                timestamp=timestamp_base + sequence,
                event_type="BEGIN",
                span_id=str(failure["span_id"]),
                operation=str(failure["operation"]),
                stage=failure["stage_or_null"],
                substep=failure["substep_or_null"],
                actor=failure["actor_or_null"],
                detail="ENTER",
            )
        )
    failures_by_sequence = {int(row["sequence"]): row for row in rows}
    cleanup_sequence = int(origin["sequence"]) + 1 + len(
        [
            row
            for row in rows[1:]
            if row["operation"]
            in {"LIVE_RESOURCE_SAMPLER", "HELD_ACTOR_CLOSE"}
        ]
    )
    cleanup_span = f"actor-cleanup-{origin['actor_or_null']}-000001"
    for sequence in range(int(origin["sequence"]), int(rows[-1]["sequence"]) + 1):
        failure = failures_by_sequence.get(sequence)
        if failure is not None:
            event = _event_record(
                trace_id=trace_id,
                    sequence=sequence,
                    timestamp=int(failure["timestamp_ns"]),
                    event_type="FAIL",
                    span_id=str(failure["span_id"]),
                    operation=str(failure["operation"]),
                    stage=failure["stage_or_null"],
                    substep=failure["substep_or_null"],
                    actor=failure["actor_or_null"],
                    detail=failure["detail_or_null"],
                    error_code=str(failure["error_code"]),
                    error_detail=str(failure["error_detail"]),
                    record_digest=str(failure["record_digest"]),
                )
            for field in _SAMPLE_FIELDS:
                event[field] = deepcopy(failure[field])
            events.append(event)
        elif sequence == cleanup_sequence:
            events.append(
                _event_record(
                    trace_id=trace_id,
                    sequence=sequence,
                    timestamp=timestamp_base + sequence,
                    event_type="BEGIN",
                    span_id=cleanup_span,
                    operation="HELD_ACTOR_ABORT_CLEANUP",
                    stage="ACTOR_LIFECYCLE",
                    substep="STOP_TERMINATE_DRAIN_REMOVE_ABSENCE",
                    actor=str(origin["actor_or_null"]),
                    detail="ENTER",
                )
            )
        elif sequence == cleanup_sequence + 1:
            events.append(
                _event_record(
                    trace_id=trace_id,
                    sequence=sequence,
                    timestamp=timestamp_base + sequence,
                    event_type="OK",
                    span_id=cleanup_span,
                    operation="HELD_ACTOR_ABORT_CLEANUP",
                    stage="ACTOR_LIFECYCLE",
                    substep="STOP_TERMINATE_DRAIN_REMOVE_ABSENCE",
                    actor=str(origin["actor_or_null"]),
                    detail="cleanup_error_count=0",
                )
            )
        else:  # pragma: no cover - fixture grammar is exact.
            raise AssertionError(f"unfilled event sequence {sequence}")
    return events


def _minimal_event_records(
    trace_id: str,
    outcome: str,
    failure_rows: Sequence[Mapping[str, object]],
    terminal_error_detail: str | None,
) -> list[dict[str, object]]:
    if failure_rows:
        return _event_records_for_failures(trace_id, failure_rows)
    begin = _event_record(
        trace_id=trace_id,
        sequence=1,
        timestamp=1,
        event_type="BEGIN",
        span_id="supervisor-run",
        operation="SUPERVISOR_RUN",
        stage="RUN",
        substep=None,
        actor=None,
        detail="ENTER",
    )
    end = _event_record(
        trace_id=trace_id,
        sequence=2,
        timestamp=2,
        event_type=outcome,
        span_id="supervisor-run",
        operation="SUPERVISOR_RUN",
        stage="RUN",
        substep=None,
        actor=None,
        detail="COMPLETE" if outcome == "OK" else "supervisor attempt failed",
        error_code=None if outcome == "OK" else "FAIL_TEST_SUPERVISOR",
        error_detail=None
        if outcome == "OK"
        else (terminal_error_detail or "test supervisor failure"),
    )
    return [begin, end]


def _trajectory(
    slot: Mapping[str, object],
    *,
    outcome: str,
    failure_rows: Sequence[Mapping[str, object]] = (),
    terminal_error_detail: str | None = None,
) -> dict[str, object]:
    status = f"COMPLETE_{outcome}"
    effective_failures = [deepcopy(dict(row)) for row in failure_rows]
    if outcome == "FAIL" and not effective_failures:
        effective_failures = [
            {
                "sequence": 2,
                "timestamp_ns": 2,
                "span_id": "supervisor-run",
                "operation": "SUPERVISOR_RUN",
                "stage_or_null": "RUN",
                "substep_or_null": None,
                "actor_or_null": None,
                "detail_or_null": "supervisor attempt failed",
                "error_code": "FAIL_TEST_SUPERVISOR",
                "error_detail": terminal_error_detail or "test supervisor failure",
                **{field: None for field in _SAMPLE_FIELDS},
                "record_digest": _tagged("generic-supervisor-fail"),
            }
        ]
    event_records = _minimal_event_records(
        str(slot["trajectory_trace_id"]),
        outcome,
        effective_failures,
        terminal_error_detail,
    )
    contract = _contract()
    payloads = [TOOL._canonical_json_bytes(event) for event in event_records]
    partition = TOOL._trajectory_partition_evidence(
        contract,
        payloads,
        trajectory_status=status,
        failure_rows_in_sequence_order=effective_failures,
    )
    return {
        "verification_status": status,
        "complete": True,
        "complete_verification_passed": True,
        "verification_error_or_null": None,
        "trace_id_or_null": slot["trajectory_trace_id"],
        "producer_id_or_null": "Q05B_SUPERVISOR_V2",
        "event_count": len(event_records),
        "first_event_record_digest_or_null": event_records[0]["record_digest"],
        "final_event_record_digest_or_null": event_records[-1]["record_digest"],
        "terminal_outcome_or_null": outcome,
        "terminal_manifest_digest_or_null": _tagged("terminal-manifest"),
        "terminal_manifest_raw_sha256_or_null": _raw("terminal-manifest-raw"),
        "clock_id_or_null": "CLOCK_MONOTONIC",
        "closed_span_count_or_null": 4,
        "terminal_event_type_or_null": outcome,
        "terminal_error_code_or_null": (
            CORE.GAP_ERROR_CODE_V2 if outcome == "FAIL" else None
        ),
        "terminal_error_detail_or_null": terminal_error_detail,
        "torn_tail_staging_basename_or_null": None,
        "open_span_ids_in_begin_order": [],
        "open_span_contexts_in_begin_order": [],
        "failure_rows_retained_in_sequence_order": deepcopy(
            partition["failure_rows_retained_in_sequence_order"]
        ),
        "cleanup_rows_retained_in_sequence_order": deepcopy(
            partition["cleanup_rows_retained_in_sequence_order"]
        ),
        "partition_certificate_or_null": deepcopy(
            partition["partition_certificate_or_null"]
        ),
        "last_durable_event_row_or_null": deepcopy(
            partition["last_durable_event_row_or_null"]
        ),
        "canonical_event_payloads_digest": partition[
            "canonical_event_payloads_digest"
        ],
        "success_trace_bindings_or_null": (
            {
                "stage_roots": [
                    _raw(f"stage-{index}") for index in range(1, 11)
                ],
                "adapter_artifact_sha256": _raw("artifact"),
                "prepublication_live_marker_replay_root": _raw(
                    "prepublication-live"
                ),
                "publication_artifact_sha256": _raw("artifact"),
            }
            if outcome == "OK"
            else None
        ),
    }


def _process(stdout: bytes, return_code: int) -> dict[str, object]:
    return {
        "return_code": return_code,
        "stdout": stdout,
        "stderr": b"",
        "stdout_overflow": False,
        "stderr_overflow": False,
        "timed_out": False,
        "child_reaped": True,
        "process_group_absent": True,
        "wrapper_resources_closed": True,
    }


def _valid_artifacts(
    wire: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    identity = wire["actual_final_delivery_identity"]
    assert type(identity) is dict
    supervisor = {
        "state": "VALID",
        "raw_sha256_or_null": wire["artifact_sha256"],
        "size_or_null": 4096,
        "replay_passed": True,
        "summary_or_null": {
            "canonical_sha256": wire["artifact_sha256"],
            "artifact_set_root": wire["artifact_set_root"],
            "candidate_receipt_root": wire["candidate_receipt_root"],
            "final_receipt_root": wire["final_receipt_root"],
            "qualification_count": wire["qualification_count"],
            "qualification_mask": wire["qualification_mask"],
            "closed_q1_authority": {
                "state": wire["q1_state"],
                "gate_count": wire["q1_gate_count"],
                "gate_mask": wire["q1_gate_mask"],
                "formal_output_roots": wire["q1_formal_output_roots"],
            },
            "actual_admission_source_commit": identity["source_commit"],
            "actual_admission_artifact_path": wire["artifact_path"],
            "pre_artifact_live_marker_replay_root": wire[
                "actual_admission_pre_artifact_live_replay_root"
            ],
            "actual_admission_evidence_root": identity[
                "actual_admission_evidence_root"
            ],
        },
        "error_or_null": None,
        "parent_identity_or_null": {
            "dev": 2049,
            "ino": 100,
            "mount_id": 7,
            "mode": stat.S_IFDIR | 0o755,
            "uid": os.geteuid(),
            "nlink": 2,
        },
        "file_identity_or_null": {
            "dev": 2049,
            "ino": 101,
            "mode": 0o444,
            "nlink": 1,
            "uid": os.geteuid(),
            "size": 4096,
        },
    }
    external = {
        "state": "VALID",
        "raw_sha256_or_null": wire["artifact_sha256"],
        "size_or_null": 4096,
        "replay_passed": True,
        "error_or_null": None,
        "parent_identity_or_null": {
            "dev": 2049,
            "ino": 200,
            "mount_id": 7,
            "mode": stat.S_IFDIR | 0o700,
            "uid": os.geteuid(),
            "nlink": 2,
        },
        "file_identity_or_null": {
            "dev": 2049,
            "ino": 201,
            "mode": 0o444,
            "nlink": 1,
            "uid": os.geteuid(),
            "size": 4096,
        },
    }
    return supervisor, external


def _absent_artifacts() -> tuple[dict[str, object], dict[str, object]]:
    supervisor = {
        "state": "ABSENT",
        "raw_sha256_or_null": None,
        "size_or_null": None,
        "replay_passed": False,
        "summary_or_null": None,
        "error_or_null": None,
        "parent_identity_or_null": None,
        "file_identity_or_null": None,
    }
    external = {
        "state": "ABSENT",
        "raw_sha256_or_null": None,
        "size_or_null": None,
        "replay_passed": False,
        "error_or_null": None,
        "parent_identity_or_null": None,
        "file_identity_or_null": None,
    }
    return supervisor, external


def _source_replay(label: str) -> dict[str, object]:
    evidence = {"label": label}
    return {
        "certain": True,
        "error_or_null": None,
        "evidence": evidence,
        "root": TOOL._domain_digest(
            "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R1/SOURCE_REPLAY/V2",
            evidence,
        ),
    }


def _future_freshness() -> dict[str, object]:
    value: dict[str, object] = {
        "label": "future slots",
        "rows": [],
        "error_or_null": None,
        "certain": True,
        "root": None,
    }
    unsigned = deepcopy(value)
    unsigned.pop("root")
    value["root"] = TOOL._domain_digest(
        "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R1/SLOT_ROOT_ABSENCE/V2",
        unsigned,
    )
    return value


def _semantic_capsule(contract: Mapping[str, object]) -> dict[str, object]:
    parents: dict[str, dict[str, int]] = {}
    slots = contract["slot_registry"]
    assert type(slots) is list
    for slot in slots:
        assert type(slot) is dict
        for field in (
            "clone_root",
            "cargo_cache_root",
            "artifact_parent_root",
            "evidence_root",
            "trace_root",
        ):
            parent = os.fspath(Path(str(slot[field])).parent)
            parents.setdefault(
                parent,
                {
                    "dev": 2049,
                    "ino": 10_000 + len(parents),
                    "mount_id": 7,
                    "mode": stat.S_IFDIR | 0o700,
                    "uid": os.geteuid(),
                },
            )
    return {
        "capsule_digest": _tagged("semantic-capsule"),
        "freshness_parent_identities": parents,
    }


def _slot_absence_evidence(
    capsule: Mapping[str, object],
    slots: Sequence[Mapping[str, object]],
    label: str,
) -> dict[str, object]:
    registry = capsule["freshness_parent_identities"]
    assert type(registry) is dict
    rows: list[dict[str, object]] = []
    for slot in slots:
        for field in (
            "clone_root",
            "cargo_cache_root",
            "artifact_parent_root",
            "evidence_root",
            "trace_root",
        ):
            target = Path(str(slot[field]))
            parent = os.fspath(target.parent)
            stable = deepcopy(registry[parent])
            assert type(stable) is dict
            full = {**stable, "nlink": 2}
            rows.append(
                {
                    "slot_id": slot["slot_id"],
                    "field": field,
                    "target_path": os.fspath(target),
                    "state_parent_path": parent,
                    "expected_state_parent_stable_identity": deepcopy(stable),
                    "observed_state_parent_identity": deepcopy(full),
                    "observed_state_parent_stable_identity": deepcopy(stable),
                    "final_state_parent_identity": deepcopy(full),
                    "final_state_parent_stable_identity": deepcopy(stable),
                    "target_present_identity_or_null": None,
                    "absent_and_anchored": True,
                }
            )
    value: dict[str, object] = {
        "label": label,
        "rows": rows,
        "error_or_null": None,
        "certain": True,
        "root": None,
    }
    unsigned = deepcopy(value)
    unsigned.pop("root")
    value["root"] = TOOL._domain_digest(
        "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R1/SLOT_ROOT_ABSENCE/V2",
        unsigned,
    )
    return value


def _cleanup_replay() -> dict[str, object]:
    preimage = {"cleanup": True}
    return {
        "preimage": preimage,
        "root": TOOL._domain_digest(
            "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R1/CLEANUP_REPLAY/V2",
            preimage,
        ),
    }


def test_domain_digest_excludes_transport_lf_and_agrees_with_pure_core() -> None:
    domain = "HEGEL/TEST/Q05B/CAMPAIGN/V2"
    value = {"a": [True, None, 7], "z": "snow"}
    observed = TOOL._domain_digest(domain, value)
    expected = CORE.campaign_record_digest_v2(domain + "\x00", value)
    including_lf = "sha256:" + sha256(
        domain.encode("ascii")
        + b"\x00"
        + TOOL._canonical_json_bytes(value)
    ).hexdigest()
    assert observed == expected
    assert observed != including_lf


def test_fixed_system_executables_are_root_owned_read_only_capsule_inputs() -> None:
    contract = _contract()
    execution = contract["execution_policy"]
    assert type(execution) is dict
    for field in (
        "supervisor_interpreter",
        "git_executable",
        "docker_executable",
    ):
        identity = TOOL._executable_identity(str(execution[field]), field)
        assert identity["uid"] == 0
        assert int(identity["mode"]) & 0o022 == 0
        assert int(identity["nlink"]) == 1


def test_target_preflight_requires_local_object_closure_fsck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "1" * 40
    tree = "2" * 40
    parent = "3" * 40
    commit_object = b"tree " + tree.encode("ascii") + b"\n"
    contract = {
        "target_commit_a_binding": {
            "target_supervisor_commit": commit,
            "target_supervisor_tree": tree,
            "target_supervisor_parent_commit": parent,
            "target_supervisor_commit_object_raw_sha256": sha256(
                commit_object
            ).hexdigest(),
        },
        "execution_policy": {
            "clone_source_repository": "/bound/repository"
        },
    }
    observed_fsck_argv: list[str] = []

    def fake_git(
        _contract: Mapping[str, object],
        arguments: Sequence[str],
        label: str,
        **_kwargs: object,
    ) -> bytes:
        if "fsck" in arguments:
            observed_fsck_argv.extend(arguments)
            raise TOOL.ShadowRetryCampaignErrorV2(
                "FAIL_CAMPAIGN_PROCESS",
                "injected missing reachable object",
                terminal=True,
                receipt={"label": label, "argv": list(arguments)},
            )
        if arguments[-1] == f"{commit}^{{tree}}":
            return tree.encode("ascii") + b"\n"
        if arguments[-1] == f"{commit}^":
            return parent.encode("ascii") + b"\n"
        if arguments[-3:-1] == ["cat-file", "commit"]:
            return commit_object
        raise AssertionError(arguments)

    monkeypatch.setattr(
        TOOL,
        "_verify_git_source_self_contained",
        lambda *_args, **_kwargs: {"self_contained": True},
    )
    monkeypatch.setattr(TOOL, "_git", fake_git)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._verify_target_git_objects(contract)
    assert error.value.code == "FAIL_CAMPAIGN_PROCESS"
    assert "injected missing reachable object" in error.value.detail
    assert observed_fsck_argv == [
        "-C",
        "/bound/repository",
        "fsck",
        "--full",
        "--strict",
        "--no-dangling",
        "--no-reflogs",
        commit,
    ]


@pytest.mark.parametrize(
    "forbidden_case",
    (
        "alternates",
        "http-alternates",
        "shallow",
        "grafts",
        "promisor-pack",
        "partial-clone-config",
        "promisor-config",
        "fsck-config",
        "include-config",
        "includeif-config",
    ),
)
def test_git_source_self_containment_rejects_external_or_partial_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    forbidden_case: str,
) -> None:
    common = tmp_path / "common.git"
    objects = common / "objects"
    git_dir = common / "worktrees" / "bound-worktree"
    (objects / "info").mkdir(parents=True, mode=0o700)
    (objects / "pack").mkdir(mode=0o700)
    (common / "info").mkdir(mode=0o700)
    git_dir.mkdir(parents=True, mode=0o700)
    config_raw = b"core.repositoryformatversion\n0\x00"
    if forbidden_case == "alternates":
        (objects / "info" / "alternates").write_bytes(b"/external/objects\n")
    elif forbidden_case == "http-alternates":
        (objects / "info" / "http-alternates").write_bytes(
            b"https://invalid.example/objects\n"
        )
    elif forbidden_case == "shallow":
        (common / "shallow").write_bytes(b"1" * 40 + b"\n")
    elif forbidden_case == "grafts":
        (common / "info" / "grafts").write_bytes(b"1" * 40 + b"\n")
    elif forbidden_case == "promisor-pack":
        (objects / "pack" / ("pack-" + "1" * 40 + ".promisor")).write_bytes(
            b""
        )
    elif forbidden_case == "partial-clone-config":
        config_raw += b"extensions.partialclone\norigin\x00"
    elif forbidden_case == "promisor-config":
        config_raw += b"remote.origin.promisor\ntrue\x00"
    elif forbidden_case == "fsck-config":
        config_raw += b"fsck.skiplist\n/external/skip-list\x00"
    elif forbidden_case == "include-config":
        config_raw += b"include.path\n/external/config\x00"
    elif forbidden_case == "includeif-config":
        config_raw += (
            b"includeif.gitdir:/bound/repository.path\n/external/config\x00"
        )

    def fake_git(
        _contract: Mapping[str, object],
        arguments: Sequence[str],
        _label: str,
        **_kwargs: object,
    ) -> bytes:
        if arguments[-1] == "--git-common-dir":
            return os.fsencode(common) + b"\n"
        if arguments[-1] == "--absolute-git-dir":
            return os.fsencode(git_dir) + b"\n"
        if arguments[-2:] == ["--git-path", "objects"]:
            return os.fsencode(objects) + b"\n"
        if arguments[-1] == "--is-shallow-repository":
            return b"false\n"
        if arguments[-1] == "--show-object-format":
            return b"sha1\n"
        if "config" in arguments:
            return config_raw
        raise AssertionError(arguments)

    monkeypatch.setattr(TOOL, "_git", fake_git)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._verify_git_source_self_contained({}, "/bound/repository")
    assert error.value.code == "FAIL_CAMPAIGN_SOURCE"
    assert any(
        token in error.value.detail
        for token in ("forbidden", "partial", "promisor")
    )


def test_git_source_self_containment_binds_common_object_and_config_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    common = tmp_path / "common.git"
    objects = common / "objects"
    git_dir = common / "worktrees" / "bound-worktree"
    (objects / "info").mkdir(parents=True, mode=0o700)
    (objects / "pack").mkdir(mode=0o700)
    (common / "info").mkdir(mode=0o700)
    git_dir.mkdir(parents=True, mode=0o700)
    config_raw = b"core.repositoryformatversion\n0\x00core.filemode\ntrue\x00"

    def fake_git(
        _contract: Mapping[str, object],
        arguments: Sequence[str],
        _label: str,
        **_kwargs: object,
    ) -> bytes:
        if arguments[-1] == "--git-common-dir":
            return os.fsencode(common) + b"\n"
        if arguments[-1] == "--absolute-git-dir":
            return os.fsencode(git_dir) + b"\n"
        if arguments[-2:] == ["--git-path", "objects"]:
            return os.fsencode(objects) + b"\n"
        if arguments[-1] == "--is-shallow-repository":
            return b"false\n"
        if arguments[-1] == "--show-object-format":
            return b"sha1\n"
        if "config" in arguments:
            return config_raw
        raise AssertionError(arguments)

    monkeypatch.setattr(TOOL, "_git", fake_git)
    evidence = TOOL._verify_git_source_self_contained({}, "/bound/repository")
    assert evidence["git_common_dir"] == os.fspath(common)
    assert evidence["git_object_dir"] == os.fspath(objects)
    assert evidence["git_worktree_dir"] == os.fspath(git_dir)
    assert evidence["local_config_listing_sha256"] == sha256(config_raw).hexdigest()
    assert evidence["local_config_row_count"] == 2
    assert evidence["repository_format_version"] == 0
    assert evidence["object_format"] == "sha1"
    assert evidence["alternates_present"] is False
    assert evidence["promisor_metadata_present"] is False


def test_fixed_cargo_source_accepts_bound_root_owned_read_only_entries() -> None:
    contract = _contract()
    execution = contract["execution_policy"]
    assert type(execution) is dict
    identity, rows = TOOL._scan_tree(
        Path(str(execution["cargo_copy_source"])),
        label="fixed Cargo source owner-policy KAT",
        allowed_uids=(os.geteuid(), 0),
        reject_group_or_world_writable=True,
    )
    assert rows
    assert {int(row["uid"]) for row in rows}.issubset(
        {os.geteuid(), 0}
    )
    assert any(int(row["uid"]) == 0 for row in rows)
    assert all(int(row["mode"]) & 0o022 == 0 for row in rows)
    assert identity["content_manifest_digest"].startswith("sha256:")


def test_freshness_parent_ignores_nlink_growth_but_rejects_inode_substitution(
    tmp_path: Path,
) -> None:
    state_parent = tmp_path / "state"
    state_parent.mkdir(mode=0o700)
    slot = {
        "slot_id": "SLOT_01",
        "clone_root": os.fspath(state_parent / "slot-clone"),
        "cargo_cache_root": os.fspath(state_parent / "slot-cargo"),
        "artifact_parent_root": os.fspath(state_parent / "slot-artifact"),
        "evidence_root": os.fspath(state_parent / "slot-evidence"),
        "trace_root": os.fspath(state_parent / "slot-trace"),
    }
    contract = {
        "fixed_external_roots": {
            "ledger_root": os.fspath(state_parent / "campaign-ledger")
        }
    }
    expected = TOOL._verify_fresh_targets(contract, [slot])
    parent_key = os.fspath(state_parent)
    assert "nlink" not in expected[parent_key]
    nlink_before = state_parent.stat().st_nlink

    (state_parent / "owner-authorization").mkdir(mode=0o700)
    (state_parent / "campaign-ledger").mkdir(mode=0o700)
    assert state_parent.stat().st_nlink > nlink_before
    context = {"freshness_parent_identities": expected}
    stable = TOOL._slot_roots_absence_evidence(
        context,
        [slot],
        label="nlink growth freshness KAT",
    )
    assert stable["certain"] is True
    assert all(row["absent_and_anchored"] is True for row in stable["rows"])
    assert any(
        row["observed_state_parent_identity"]["nlink"] != nlink_before
        for row in stable["rows"]
    )
    slot_with_artifact = {
        **slot,
        "artifact_path": os.fspath(
            Path(str(slot["artifact_parent_root"])) / "artifact.json"
        ),
    }
    absent_artifact = TOOL._absent_external_artifact(
        context,
        slot_with_artifact,
    )
    assert absent_artifact["state"] == "ABSENT"
    assert absent_artifact["error_or_null"] is None

    replaced_parent = tmp_path / "state-replaced"
    state_parent.rename(replaced_parent)
    state_parent.mkdir(mode=0o700)
    substituted = TOOL._slot_roots_absence_evidence(
        context,
        [slot],
        label="inode substitution freshness KAT",
    )
    assert substituted["certain"] is False
    assert "parent drifted" in str(substituted["error_or_null"])


def test_preflight_names_only_ledger_and_slot_freshness_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    capsule = {"capsule_digest": _tagged("capsule")}
    context = {
        "capsule": capsule,
        "authorization_template": {"authorization_digest": _tagged("authorization")},
    }
    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_preflight_context", lambda _contract: context)
    monkeypatch.setattr(TOOL.os.path, "lexists", lambda _path: True)
    receipt = TOOL.preflight_phase3_q05b_shadow_retry_campaign_v2()
    assert receipt["owner_authorization_present"] is True
    assert receipt["ledger_and_slot_freshness_targets_absent"] is True
    assert "freshness_targets_absent" not in receipt


def test_tree_manifest_and_closure_reject_ignored_file_and_empty_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "clone"
    root.mkdir(mode=0o700)
    (root / ".gitignore").write_text("ignored.tmp\n", encoding="ascii")
    (root / ".git").mkdir(mode=0o700)
    (root / ".git/config").write_text("baseline\n", encoding="ascii")
    before_identity, before_rows = TOOL._scan_tree(
        root,
        label="clone closure before",
        manifest_domain="HEGEL/TEST/CLONE-CLOSURE/V2",
    )
    (root / "ignored.tmp").write_bytes(b"residual\n")
    (root / "empty-residual-directory").mkdir(mode=0o700)
    (root / ".git/config").write_text("changed\n", encoding="ascii")
    after_identity, after_rows = TOOL._scan_tree(
        root,
        label="clone closure after",
        manifest_domain="HEGEL/TEST/CLONE-CLOSURE/V2",
    )
    assert before_identity["manifest_digest"] != after_identity["manifest_digest"]
    assert {row["relative_path"] for row in after_rows} - {
        row["relative_path"] for row in before_rows
    } == {"ignored.tmp", "empty-residual-directory"}

    source_before = {
        "certain": True,
        "error_or_null": None,
        "evidence": {
            "clone": {
                "clone_closure_without_artifact_digest": before_identity[
                    "manifest_digest"
                ],
                "clone_tree_entry_count": len(before_rows),
            }
        },
        "root": _tagged("before"),
        "clone_closure_rows_internal": before_rows,
    }
    source_after = {
        "certain": True,
        "error_or_null": None,
        "evidence": {
            "clone": {
                "clone_closure_without_artifact_digest": after_identity[
                    "manifest_digest"
                ],
                "clone_tree_entry_count": len(after_rows),
            }
        },
        "root": _tagged("after"),
        "clone_closure_rows_internal": after_rows,
    }
    closed = TOOL._enforce_clone_closure(
        {"slot_id": "SLOT_01"},
        source_before,
        source_after,
    )
    assert closed["certain"] is False
    assert "outside the single allowed artifact path" in str(
        closed["error_or_null"]
    )
    difference = closed["evidence"]["clone_closure_comparison"][
        "difference"
    ]
    assert {row["relative_path"] for row in difference["added_rows"]} == {
        "ignored.tmp",
        "empty-residual-directory",
    }
    assert [
        row["relative_path"] for row in difference["changed_rows"]
    ] == [".git/config"]


def test_slot_registry_accepts_exact_contract_and_rejects_path_drift() -> None:
    contract = _contract()
    slots = TOOL._slot_registry(contract)
    assert [slot["slot_id"] for slot in slots] == [
        "SLOT_01",
        "SLOT_02",
        "SLOT_03",
    ]
    assert [slot["slot_ordinal"] for slot in slots] == [1, 2, 3]

    drifted = deepcopy(contract)
    drifted_slots = drifted["slot_registry"]
    assert type(drifted_slots) is list and type(drifted_slots[0]) is dict
    drifted_slots[0]["artifact_parent_initial_state"] = "ABSENT"
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._slot_registry(drifted)
    assert error.value.code == "FAIL_CAMPAIGN_CONTRACT"

    aliased = deepcopy(contract)
    aliased_slots = aliased["slot_registry"]
    assert type(aliased_slots) is list
    assert type(aliased_slots[0]) is dict and type(aliased_slots[1]) is dict
    aliased_slots[1]["cargo_cache_root"] = aliased_slots[0]["cargo_cache_root"]
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._slot_registry(aliased)
    assert error.value.code == "FAIL_CAMPAIGN_CONTRACT"


def test_v2r1_instance_isolation_rejects_prior_v2_authority_and_roots() -> None:
    raw_contract = TOOL._pretty_json(
        TOOL.CONFIG_PATH.read_bytes(),
        "v2r1 target Git object closure",
    )
    assert type(raw_contract) is dict
    target_git_object_closure = raw_contract["target_commit_a_binding"]
    assert type(target_git_object_closure) is dict
    assert target_git_object_closure["target_supervisor_commit"] == (
        "7bc2044212e278443b1ca703115c7cb17591ac14"
    )

    contract = _contract()
    TOOL._require_v2r1_instance_isolation_v2(contract)
    protected = contract["protected_state_policy"]
    assert type(protected) is dict
    protected_roots = protected["protected_roots_in_exact_order"]
    assert type(protected_roots) is list
    assert set(TOOL._PRIOR_V2_PROTECTED_ROOTS).issubset(protected_roots)
    slots = contract["slot_registry"]
    assert type(slots) is list
    for ordinal, slot in enumerate(slots, start=1):
        assert type(slot) is dict
        assert slot["trajectory_trace_id"] == (
            f"q05b-shadow-retry-campaign-v2r1-slot-{ordinal:02d}"
        )

    mutations = []
    prior_id = deepcopy(contract)
    prior_id["contract_id"] = TOOL._PRIOR_V2_CONTRACT_ID
    mutations.append(prior_id)

    reusable_prior = deepcopy(contract)
    reusable_prior["protected_state_policy"][
        "prior_v2_capsule_authorization_or_ledger_reuse_allowed"
    ] = True
    mutations.append(reusable_prior)

    prior_authorization_root = deepcopy(contract)
    prior_authorization_root["fixed_external_roots"]["authorization_root"] = (
        f"{TOOL._PRIOR_V2_STATE_PREFIX}-authorization"
    )
    mutations.append(prior_authorization_root)

    prior_clone_root = deepcopy(contract)
    prior_clone_root["slot_registry"][0]["clone_root"] = (
        f"{TOOL._PRIOR_V2_STATE_PREFIX}-slot-01-clone"
    )
    mutations.append(prior_clone_root)

    prior_trace_id = deepcopy(contract)
    prior_trace_id["slot_registry"][0]["trajectory_trace_id"] = (
        "q05b-shadow-retry-campaign-v2-slot-01"
    )
    mutations.append(prior_trace_id)

    for mutation in mutations:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._require_v2r1_instance_isolation_v2(mutation)
        assert error.value.code == "FAIL_CAMPAIGN_CONTRACT"


@pytest.mark.parametrize(
    "domain_name",
    tuple(TOOL._EXPECTED_HASH_DOMAINS_V2R1),
)
def test_contract_loader_rejects_rehashed_v2r1_hash_domain_drift(
    monkeypatch: pytest.MonkeyPatch,
    domain_name: str,
) -> None:
    base = _contract()
    expected_domain = TOOL._EXPECTED_HASH_DOMAINS_V2R1[domain_name]
    prior_v2_domain = expected_domain.replace(
        "SHADOW_RETRY_CAMPAIGN_V2R1",
        "SHADOW_RETRY_CAMPAIGN",
    )
    for forged_domain in (
        prior_v2_domain,
        f"HEGEL/FORGED/Q05B/{domain_name.upper()}/V2",
    ):
        contract = deepcopy(base)
        domains = contract["hash_domains"]
        assert type(domains) is dict
        domains[domain_name] = forged_domain
        unsigned_contract = deepcopy(contract)
        unsigned_contract.pop("contract_digest")
        contract["contract_digest"] = TOOL._domain_digest(
            domains["contract"],
            unsigned_contract,
        )
        monkeypatch.setattr(
            TOOL,
            "_pretty_json",
            lambda _raw, _label, value=contract: deepcopy(value),
        )
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL.load_phase3_q05b_shadow_retry_campaign_contract_v2()
        assert error.value.code == "FAIL_CAMPAIGN_CONTRACT"
        assert error.value.detail == "campaign hash-domain registry differs"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("clone_target_ref_name", "refs/hegel/drifted-target"),
        ("clone_ref_policy", "TEMPORARY_OR_MULTIPLE_REFS_ALLOWED"),
    ),
)
def test_contract_loader_rejects_clone_target_ref_policy_drift(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    contract = _contract()
    execution = contract["execution_policy"]
    assert type(execution) is dict
    execution[field] = value
    monkeypatch.setattr(
        TOOL,
        "_pretty_json",
        lambda _raw, _label: deepcopy(contract),
    )
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL.load_phase3_q05b_shadow_retry_campaign_contract_v2()
    assert error.value.code == "FAIL_CAMPAIGN_CONTRACT"
    assert error.value.detail == "campaign authority boundary differs"


def test_prior_v2_capsule_authorization_and_rehashed_ledger_are_not_v2r1_authority(
    tmp_path: Path,
) -> None:
    contract, capsule, authorization, ledger_object, _result = (
        _completed_real_ledger(tmp_path)
    )
    ledger = ledger_object
    try:
        copied_prior_capsule = {
            "capsule_digest": TOOL._PRIOR_V2_CAPSULE_DIGEST,
        }
        copied_prior_authorization = {
            "authorization_digest": TOOL._PRIOR_V2_AUTHORIZATION_DIGEST,
        }
        rows = deepcopy(
            list(zip(ledger.published_names, ledger.published_records))
        )
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._verify_campaign_completion_chain_v2(
                contract,
                copied_prior_capsule,
                copied_prior_authorization,
                rows,
            )
        assert error.value.code == "FAIL_CAMPAIGN_LEDGER"

        prior_capsule_preimage = {
            "campaign_id": TOOL._PRIOR_V2_CONTRACT_ID,
            "authorization_root": (
                f"{TOOL._PRIOR_V2_STATE_PREFIX}-authorization"
            ),
            "ledger_root": f"{TOOL._PRIOR_V2_STATE_PREFIX}-ledger",
        }
        rehashed_prior_capsule = {
            **prior_capsule_preimage,
            "capsule_digest": TOOL._domain_digest(
                contract["hash_domains"]["capsule"],
                prior_capsule_preimage,
            ),
        }
        rehashed_prior_authorization = deepcopy(authorization)
        rehashed_prior_authorization.update(
            {
                "campaign_id": TOOL._PRIOR_V2_CONTRACT_ID,
                "contract_digest": _tagged("prior-v2-contract"),
                "capsule_digest": rehashed_prior_capsule["capsule_digest"],
                "authorization_root": (
                    f"{TOOL._PRIOR_V2_STATE_PREFIX}-authorization"
                ),
                "ledger_root": f"{TOOL._PRIOR_V2_STATE_PREFIX}-ledger",
            }
        )
        confirmation_fields = (
            "campaign_id",
            "contract_digest",
            "target_supervisor_commit",
            "target_source_binding_digest",
            "campaign_source_binding_digest",
            "capsule_digest",
            "authorization_root",
            "ledger_root",
            "three_exact_slot_path_records",
            "maximum_supervisor_invocations",
            "strictly_sequential_one_shot_slots",
            "retry_predicate_digest",
            "success_predicate_digest",
            "terminal_no_resume_policy",
            "forbidden_scope_digest",
        )
        rehashed_prior_authorization["owner_confirmation_digest"] = (
            TOOL._domain_digest(
                "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R1/"
                "OWNER_CONFIRMATION/V2",
                {
                    field: deepcopy(rehashed_prior_authorization[field])
                    for field in confirmation_fields
                },
            )
        )
        authorization_unsigned = deepcopy(rehashed_prior_authorization)
        authorization_unsigned.pop("authorization_digest")
        rehashed_prior_authorization["authorization_digest"] = (
            TOOL._domain_digest(
                contract["hash_domains"]["authorization"],
                authorization_unsigned,
            )
        )
        auth_contract = deepcopy(contract)
        auth_root = tmp_path / "rehashed-prior-v2-authorization"
        auth_root.mkdir(mode=0o700)
        auth_root.chmod(0o700)
        auth_contract["fixed_external_roots"]["authorization_root"] = (
            os.fspath(auth_root)
        )
        auth_contract["authorization_policy"]["authorization_root"] = (
            os.fspath(auth_root)
        )
        auth_basename = str(
            auth_contract["authorization_policy"][
                "authorization_record_name"
            ]
        )
        auth_root_fd = os.open(
            auth_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
        )
        try:
            TOOL._publish_immutable_at(
                auth_root_fd,
                auth_basename,
                rehashed_prior_authorization,
            )
        finally:
            os.close(auth_root_fd)
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._load_authorization(auth_contract, capsule)
        assert error.value.code == "FAIL_CAMPAIGN_AUTHORIZATION"

        rehashed_rows = deepcopy(rows)
        previous_digest: str | None = None
        for _basename, record in rehashed_rows:
            record["campaign_id"] = TOOL._PRIOR_V2_CONTRACT_ID
            record["contract_digest"] = _tagged("prior-v2-contract")
            record["campaign_source_binding_digest"] = _tagged(
                "prior-v2-source-binding"
            )
            record["capsule_digest"] = TOOL._PRIOR_V2_CAPSULE_DIGEST
            record["authorization_digest"] = (
                TOOL._PRIOR_V2_AUTHORIZATION_DIGEST
            )
            record["previous_record_digest_or_null"] = previous_digest
            unsigned = deepcopy(record)
            unsigned.pop("record_digest")
            record["record_digest"] = TOOL._domain_digest(
                contract["hash_domains"]["ledger_record"],
                unsigned,
            )
            previous_digest = str(record["record_digest"])
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._verify_campaign_completion_chain_v2(
                contract,
                copied_prior_capsule,
                copied_prior_authorization,
                rehashed_rows,
            )
        assert error.value.code == "FAIL_CAMPAIGN_LEDGER"
    finally:
        ledger.close()


def test_external_binding_exact_fields_digest_and_disk_replay(
    tmp_path: Path,
) -> None:
    contract = _contract()
    slot = _slot(contract)
    trace_root = tmp_path / "trace"
    trace_root.mkdir(mode=0o700)
    trace_root.chmod(0o700)
    slot["trace_root"] = str(trace_root)
    process = _process(b'{"result":"test"}\n', 1)
    trajectory = _trajectory(
        slot,
        outcome="FAIL",
        terminal_error_detail="test terminal failure",
    )
    supervisor, external = _absent_artifacts()
    capsule = {"capsule_digest": _tagged("capsule")}
    binding = TOOL._binding_record(
        contract,
        capsule,
        slot,
        process,
        {},
        trajectory,
        supervisor,
        external,
        True,
        [],
        None,
        _source_replay("source-before"),
        _source_replay("source-after"),
        _future_freshness(),
        _cleanup_replay(),
    )
    policy = contract["trajectory_binding_policy"]
    assert type(policy) is dict
    assert set(binding) == set(policy["external_binding_exact_fields"])
    unsigned = deepcopy(binding)
    unsigned.pop("binding_digest")
    assert binding["binding_digest"] == TOOL._domain_digest(
        contract["hash_domains"]["trajectory_binding"],
        unsigned,
    )
    payload = TOOL._canonical_json_bytes(binding)
    assert TOOL._strict_json(payload, "binding replay") == binding

    drifted = deepcopy(contract)
    drifted_policy = drifted["trajectory_binding_policy"]
    assert type(drifted_policy) is dict
    drifted_policy["external_binding_exact_fields"] = list(
        drifted_policy["external_binding_exact_fields"]
    )[:-1]
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._binding_record(
            drifted,
            capsule,
            slot,
            process,
            {},
            trajectory,
            supervisor,
            external,
            True,
            [],
            None,
            _source_replay("source-before"),
            _source_replay("source-after"),
            _future_freshness(),
            _cleanup_replay(),
        )
    assert error.value.code == "FAIL_CAMPAIGN_BINDING"


def test_external_binding_trace_identity_failure_is_never_downgraded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    slot = _slot(contract)
    trace_root = tmp_path / "trace"
    trace_root.mkdir(mode=0o700)
    trace_root.chmod(0o700)
    slot["trace_root"] = os.fspath(trace_root)
    process = _process(b'{"result":"test"}\n', 1)
    trajectory = _trajectory(
        slot,
        outcome="FAIL",
        terminal_error_detail="test terminal failure",
    )
    supervisor, external = _absent_artifacts()

    def fail_mount_identity(*_args: object, **_kwargs: object) -> int:
        raise TOOL.ShadowRetryCampaignErrorV2(
            "FAIL_CAMPAIGN_FILESYSTEM",
            "injected trace mount-id failure",
            terminal=True,
        )

    monkeypatch.setattr(TOOL, "_fd_mount_id", fail_mount_identity)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._binding_record(
            contract,
            {"capsule_digest": _tagged("capsule")},
            slot,
            process,
            {},
            trajectory,
            supervisor,
            external,
            True,
            [],
            None,
            _source_replay("source-before"),
            _source_replay("source-after"),
            _future_freshness(),
            _cleanup_replay(),
        )
    assert error.value.code == "FAIL_CAMPAIGN_FILESYSTEM"
    assert "injected trace mount-id failure" in error.value.detail


def test_external_binding_persists_all_postprocess_failure_preimages(
    tmp_path: Path,
) -> None:
    contract = _contract()
    slot = _slot(contract)
    trace_root = tmp_path / "trace"
    trace_root.mkdir(mode=0o700)
    slot["trace_root"] = os.fspath(trace_root)
    process = _process(b"not-success\n", 1)
    trajectory = _trajectory(
        slot,
        outcome="FAIL",
        terminal_error_detail="trajectory exact reason",
    )
    supervisor, external = _absent_artifacts()
    supervisor["state"] = "RESIDUAL_INVALID"
    supervisor["error_or_null"] = "artifact replay exact reason"
    external["state"] = "RESIDUAL_INVALID"
    external["error_or_null"] = "artifact export exact reason"
    source_before = _source_replay("source-before")
    source_after = _source_replay("source-after")
    source_after["certain"] = False
    source_after["error_or_null"] = "source replay exact reason"
    source_after["evidence"] = {
        "error": "source replay exact reason",
        "changed_path": "target/ignored.tmp",
    }
    future = _future_freshness()
    future["certain"] = False
    future["error_or_null"] = "future slot exact reason"
    cleanup = {
        "preimage": {
            "docker_rows": [{"container_id": "ab" * 32}],
            "docker_error_or_null": "docker exact reason",
        },
        "root": _tagged("cleanup-errors"),
    }
    binding = TOOL._binding_record(
        contract,
        {"capsule_digest": _tagged("capsule")},
        slot,
        process,
        {},
        trajectory,
        supervisor,
        external,
        False,
        [{"container_id": "ab" * 32}],
        "docker exact reason",
        source_before,
        source_after,
        future,
        cleanup,
    )
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir(mode=0o700)
    descriptor = os.open(
        evidence_root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
    )
    try:
        TOOL._publish_immutable_at(descriptor, "binding.json", binding)
    finally:
        os.close(descriptor)
    raw, _status = TOOL._read_regular_nofollow(
        evidence_root / "binding.json",
        "durable binding test",
    )
    replay = TOOL._strict_json(raw, "durable binding test")
    assert replay == binding
    for reason in (
        "trajectory exact reason",
        "artifact replay exact reason",
        "artifact export exact reason",
        "source replay exact reason",
        "future slot exact reason",
        "docker exact reason",
        "target/ignored.tmp",
    ):
        assert reason.encode("ascii") in raw


def test_slot_observation_success_gap_and_uncertainty_use_real_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    slot = _slot(contract)
    monkeypatch.setattr(TOOL, "_load_core", lambda _contract: CORE)

    success_wire = _success_wire(contract, slot)
    success_stdout = CORE.canonical_campaign_json_bytes_v2(success_wire)
    success_process = _process(success_stdout, 0)
    success_supervisor, success_external = _valid_artifacts(success_wire)
    success = TOOL._build_slot_observation(
        contract,
        slot,
        success_process,
        CORE.classify_supervisor_stdout_v2(success_stdout),
        _trajectory(slot, outcome="OK"),
        success_supervisor,
        success_external,
        source_before_certain=True,
        source_after_certain=True,
        docker_absent=True,
        external_binding_certain=True,
    )
    assert success["record_digest"] == CORE.slot_observation_record_digest_v2(
        success
    )
    assert CORE.evaluate_slot_observation_v2(success) == CORE.SUCCESS_V2

    gap_detail, gap_rows = _held_final_gap_rows()
    gap_stdout = CORE.canonical_campaign_json_bytes_v2(
        _error_wire(gap_detail)
    )
    gap_process = _process(gap_stdout, 1)
    gap_supervisor, gap_external = _absent_artifacts()
    gap = TOOL._build_slot_observation(
        contract,
        slot,
        gap_process,
        CORE.classify_supervisor_stdout_v2(gap_stdout),
        _trajectory(
            slot,
            outcome="FAIL",
            failure_rows=gap_rows,
            terminal_error_detail=gap_detail,
        ),
        gap_supervisor,
        gap_external,
        source_before_certain=True,
        source_after_certain=True,
        docker_absent=True,
        external_binding_certain=True,
    )
    assert CORE.evaluate_slot_observation_v2(gap) == CORE.RETRY_EXACT_GAP_V2

    uncertain_process = deepcopy(gap_process)
    uncertain_process["timed_out"] = True
    uncertain = TOOL._build_slot_observation(
        contract,
        slot,
        uncertain_process,
        CORE.classify_supervisor_stdout_v2(gap_stdout),
        _trajectory(
            slot,
            outcome="FAIL",
            failure_rows=gap_rows,
            terminal_error_detail=gap_detail,
        ),
        gap_supervisor,
        gap_external,
        source_before_certain=True,
        source_after_certain=True,
        docker_absent=True,
        external_binding_certain=True,
    )
    assert (
        CORE.evaluate_slot_observation_v2(uncertain)
        == CORE.TERMINAL_NONRETRYABLE_V2
    )

    slot_three = _slot(contract, 3)
    exhausted = TOOL._build_slot_observation(
        contract,
        slot_three,
        gap_process,
        CORE.classify_supervisor_stdout_v2(gap_stdout),
        _trajectory(
            slot_three,
            outcome="FAIL",
            failure_rows=gap_rows,
            terminal_error_detail=gap_detail,
        ),
        gap_supervisor,
        gap_external,
        source_before_certain=True,
        source_after_certain=True,
        docker_absent=True,
        external_binding_certain=True,
    )
    assert (
        CORE.evaluate_slot_observation_v2(exhausted)
        == CORE.EXHAUSTED_RETRYABLE_GAP_V2
    )


def test_continuous_sampler_gap_preserves_real_two_tier_detail_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    slot = _slot(contract)
    monkeypatch.setattr(TOOL, "_load_core", lambda _contract: CORE)
    terminal_detail, rows = _continuous_gap_rows()
    stdout = CORE.canonical_campaign_json_bytes_v2(
        _error_wire(terminal_detail)
    )
    supervisor, external = _absent_artifacts()
    observation = TOOL._build_slot_observation(
        contract,
        slot,
        _process(stdout, 1),
        CORE.classify_supervisor_stdout_v2(stdout),
        _trajectory(
            slot,
            outcome="FAIL",
            failure_rows=rows,
            terminal_error_detail=terminal_detail,
        ),
        supervisor,
        external,
        source_before_certain=True,
        source_after_certain=True,
        docker_absent=True,
        external_binding_certain=True,
    )
    assert (
        CORE.evaluate_slot_observation_v2(observation)
        == CORE.RETRY_EXACT_GAP_V2
    )


def test_real_stage4_to_stage5_benign_partition_and_cleanup_drive_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    slot = _slot(contract)
    trace_parent = tmp_path / "trajectory-parent"
    trace_parent.mkdir(mode=0o700)
    trace_parent.chmod(0o700)
    trace_root = trace_parent / "trajectory"
    slot["trace_root"] = os.fspath(trace_root)
    previous = 100
    scheduled = previous + CORE.TARGET_WAIT_NS_V2
    actual = previous + CORE.MAXIMUM_GAP_NS_V2 + 1
    raw_detail = (
        "held actor sampling gap exceeded bound: "
        "actor_id=PYTHON_ENDPOINT; role_id=1; sample_ordinal=1; "
        f"previous_finish_ns={previous}; scheduled_start_ns={scheduled}; "
        f"actual_start_ns={actual}; gap_ns={actual - previous}; "
        f"maximum_gap_ns={CORE.MAXIMUM_GAP_NS_V2}; "
        f"target_wait_ns={CORE.TARGET_WAIT_NS_V2}; previous_duration_ns=0"
    )
    terminal_detail = "held actor sampler failed: " + raw_detail
    peer_previous = 1_000
    peer_scheduled = peer_previous + CORE.TARGET_WAIT_NS_V2
    peer_actual = peer_scheduled
    peer_finish = peer_actual + 10
    ticks = iter(range(1_000_000_000, 1_000_000_100))
    writer = TRAJECTORY.create_durable_trajectory_writer_v2(
        trace_root,
        expected_parent=trace_parent,
        trace_id=str(slot["trajectory_trace_id"]),
        producer_id="Q05B_SUPERVISOR_V2",
        clock_ns=lambda: next(ticks),
    )
    try:
        writer.begin(
            "supervisor-run",
            "SUPERVISOR_RUN",
            stage="RUN",
            detail="ENTER",
        )
        writer.begin(
            "supervisor-stage-04",
            "SUPERVISOR_STAGE",
            stage="STAGE_04",
            substep="RUN_TWO_HELD_ENDPOINTS_IN_PARALLEL",
            detail="ENTER",
        )
        writer.begin(
            "sampler-PYTHON_ENDPOINT",
            "LIVE_RESOURCE_SAMPLER",
            stage="ACTOR_LIFECYCLE",
            substep="CID_TO_HELD_FINAL_CONTINUOUS_SAMPLING",
            actor="PYTHON_ENDPOINT",
            detail="ENTER",
        )
        writer.begin(
            "sample-PYTHON_ENDPOINT-000001",
            "LIVE_RESOURCE_SAMPLE",
            stage="ACTOR_LIFECYCLE",
            substep="COLLECT_BOUND_LIVE_RESOURCES",
            actor="PYTHON_ENDPOINT",
            detail="sample_ordinal=1",
        )
        writer.begin(
            "sampler-RUST_ENDPOINT",
            "LIVE_RESOURCE_SAMPLER",
            stage="ACTOR_LIFECYCLE",
            substep="CID_TO_HELD_FINAL_CONTINUOUS_SAMPLING",
            actor="RUST_ENDPOINT",
            detail="ENTER",
        )
        writer.begin(
            "sample-RUST_ENDPOINT-000001",
            "LIVE_RESOURCE_SAMPLE",
            stage="ACTOR_LIFECYCLE",
            substep="COLLECT_BOUND_LIVE_RESOURCES",
            actor="RUST_ENDPOINT",
            detail="sample_ordinal=1",
        )
        writer.fail(
            "sample-PYTHON_ENDPOINT-000001",
            error_code=CORE.GAP_ERROR_CODE_V2,
            error_detail=raw_detail,
            detail="MAXIMUM_UNSAMPLED_GAP_EXCEEDED",
            role_id="1",
            sample_ordinal=1,
            previous_finish_ns=previous,
            scheduled_start_ns=scheduled,
            actual_start_ns=actual,
            finish_ns=actual,
            collection_duration_ns=0,
            scheduler_lateness_ns=actual - scheduled,
            target_wait_ns=CORE.TARGET_WAIT_NS_V2,
            gap_ns=actual - previous,
            maximum_gap_ns=CORE.MAXIMUM_GAP_NS_V2,
            gap_exceeded=True,
        )
        writer.fail(
            "sampler-PYTHON_ENDPOINT",
            error_code=CORE.GAP_ERROR_CODE_V2,
            error_detail=raw_detail,
            detail=(
                "MAXIMUM_UNSAMPLED_GAP_EXCEEDED; "
                "sample_span=sample-PYTHON_ENDPOINT-000001"
            ),
        )
        writer.begin(
            "actor-cleanup-PYTHON_ENDPOINT-000001",
            "HELD_ACTOR_ABORT_CLEANUP",
            stage="ACTOR_LIFECYCLE",
            substep="STOP_TERMINATE_DRAIN_REMOVE_ABSENCE",
            actor="PYTHON_ENDPOINT",
            detail="ENTER",
        )
        writer.ok(
            "actor-cleanup-PYTHON_ENDPOINT-000001",
            detail="cleanup_error_count=0",
        )
        writer.begin(
            "actor-cleanup-RUST_ENDPOINT-000002",
            "HELD_ACTOR_ABORT_CLEANUP",
            stage="ACTOR_LIFECYCLE",
            substep="STOP_TERMINATE_DRAIN_REMOVE_ABSENCE",
            actor="RUST_ENDPOINT",
            detail="ENTER",
        )
        writer.ok(
            "sample-RUST_ENDPOINT-000001",
            detail="LIVE_RESOURCE_COLLECTION_COMPLETE",
            role_id="2",
            sample_ordinal=1,
            previous_finish_ns=peer_previous,
            scheduled_start_ns=peer_scheduled,
            actual_start_ns=peer_actual,
            finish_ns=peer_finish,
            collection_duration_ns=peer_finish - peer_actual,
            scheduler_lateness_ns=peer_actual - peer_scheduled,
            target_wait_ns=CORE.TARGET_WAIT_NS_V2,
            gap_ns=peer_actual - peer_previous,
            maximum_gap_ns=CORE.MAXIMUM_GAP_NS_V2,
            gap_exceeded=False,
        )
        writer.ok(
            "sampler-RUST_ENDPOINT",
            detail="sample_count=1",
        )
        writer.ok(
            "actor-cleanup-RUST_ENDPOINT-000002",
            detail="cleanup_error_count=0",
        )
        writer.begin(
            "actor-cleanup-PYTHON_ENDPOINT-000003",
            "HELD_ACTOR_ABORT_CLEANUP",
            stage="ACTOR_LIFECYCLE",
            substep="STOP_TERMINATE_DRAIN_REMOVE_ABSENCE",
            actor="PYTHON_ENDPOINT",
            detail="ENTER",
        )
        writer.ok(
            "actor-cleanup-PYTHON_ENDPOINT-000003",
            detail="cleanup_error_count=0",
        )
        writer.ok(
            "supervisor-stage-04",
            detail="stage_evidence_root=" + _raw("real-stage-04"),
        )
        writer.begin(
            "supervisor-stage-05",
            "SUPERVISOR_STAGE",
            stage="STAGE_05",
            substep="SEAL_ENDPOINT_STDOUT_SIDECARS_CONTROLS_AND_RESOURCES",
            detail="ENTER",
        )
        writer.begin(
            "actor-close-PYTHON_ENDPOINT",
            "HELD_ACTOR_CLOSE",
            stage="ACTOR_LIFECYCLE",
            substep="DONE_FINAL_SAMPLE_RELEASE_EXIT_REMOVE_ABSENCE",
            actor="PYTHON_ENDPOINT",
            detail="ENTER",
        )
        writer.fail(
            "actor-close-PYTHON_ENDPOINT",
            error_code=CORE.GAP_ERROR_CODE_V2,
            error_detail=terminal_detail,
            detail="ACTOR_CLOSE_FAILED_BEFORE_CLEANUP",
        )
        writer.fail(
            "supervisor-stage-05",
            error_code=CORE.GAP_ERROR_CODE_V2,
            error_detail=terminal_detail,
            detail="FAILED",
        )
        writer.fail(
            "supervisor-run",
            error_code=CORE.GAP_ERROR_CODE_V2,
            error_detail=terminal_detail,
            detail="supervisor attempt failed",
        )
        writer.seal_fail()
    finally:
        writer.close()

    monkeypatch.setattr(
        TOOL,
        "_trajectory_module",
        lambda _contract, _slot: TRAJECTORY,
    )
    monkeypatch.setattr(TOOL, "_load_core", lambda _contract: CORE)
    trajectory = TOOL._trajectory_evidence(contract, slot)
    certificate = trajectory["partition_certificate_or_null"]
    assert type(certificate) is dict
    assert certificate["origin_sequence"] == 7
    assert certificate["suffix_event_count"] == 16
    assert certificate["failure_event_count"] == 5
    assert certificate["failure_rows_retained_count"] == 5
    assert certificate["failure_rows_truncated"] is False
    assert certificate["cleanup_nonfailure_event_count"] == 6
    assert certificate["cleanup_rows_retained_count"] == 6
    assert certificate["cleanup_rows_truncated"] is False
    assert certificate["other_nonfailure_event_count"] == 5
    assert certificate["partitioned_event_count"] == 16
    assert certificate["unpartitioned_event_count"] == 0
    assert [
        row["operation"]
        for row in trajectory["failure_rows_retained_in_sequence_order"]
    ] == [
        "LIVE_RESOURCE_SAMPLE",
        "LIVE_RESOURCE_SAMPLER",
        "HELD_ACTOR_CLOSE",
        "SUPERVISOR_STAGE",
        "SUPERVISOR_RUN",
    ]
    cleanups = trajectory["cleanup_rows_retained_in_sequence_order"]
    assert [
        row["actor_or_null"]
        for row in cleanups
        if row["event_type"] == "OK"
    ] == ["PYTHON_ENDPOINT", "RUST_ENDPOINT", "PYTHON_ENDPOINT"]
    assert trajectory["last_durable_event_row_or_null"]["operation"] == (
        "SUPERVISOR_RUN"
    )
    stdout = CORE.canonical_campaign_json_bytes_v2(_error_wire(terminal_detail))
    process = _process(stdout, 1)
    supervisor, external = _absent_artifacts()
    binding = TOOL._binding_record(
        contract,
        {"capsule_digest": _tagged("capsule")},
        slot,
        process,
        CORE.classify_supervisor_stdout_v2(stdout),
        trajectory,
        supervisor,
        external,
        True,
        [],
        None,
        _source_replay("source-before"),
        _source_replay("source-after"),
        _future_freshness(),
        _cleanup_replay(),
    )
    assert binding["partition_certificate_or_null"] == certificate
    assert binding["cleanup_rows_retained_in_sequence_order"] == cleanups
    assert binding["failure_rows_retained_in_sequence_order"] == trajectory[
        "failure_rows_retained_in_sequence_order"
    ]
    observation = TOOL._build_slot_observation(
        contract,
        slot,
        process,
        CORE.classify_supervisor_stdout_v2(stdout),
        trajectory,
        supervisor,
        external,
        source_before_certain=True,
        source_after_certain=True,
        docker_absent=True,
        external_binding_certain=True,
    )
    assert observation["trajectory_partition_certificate_or_null"] == certificate
    assert observation["trajectory_cleanup_rows"] == cleanups
    assert observation["trajectory_failure_rows"] == trajectory[
        "failure_rows_retained_in_sequence_order"
    ]
    assert CORE.evaluate_slot_observation_v2(observation) == (
        CORE.RETRY_EXACT_GAP_V2
    )


def test_real_incomplete_prefix_preserves_last_durable_event_certificate_surface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    slot = _slot(contract)
    trace_parent = tmp_path / "incomplete-trajectory-parent"
    trace_parent.mkdir(mode=0o700)
    trace_parent.chmod(0o700)
    trace_root = trace_parent / "trajectory"
    slot["trace_root"] = os.fspath(trace_root)
    writer = TRAJECTORY.create_durable_trajectory_writer_v2(
        trace_root,
        expected_parent=trace_parent,
        trace_id=str(slot["trajectory_trace_id"]),
        producer_id="Q05B_SUPERVISOR_V2",
    )
    try:
        writer.begin(
            "supervisor-run",
            "SUPERVISOR_RUN",
            stage="RUN",
            detail="source_commit=" + "12" * 20,
        )
    finally:
        writer.close()
    monkeypatch.setattr(
        TOOL,
        "_trajectory_module",
        lambda _contract, _slot: TRAJECTORY,
    )
    evidence = TOOL._trajectory_evidence(contract, slot)
    assert evidence["verification_status"] == "INCOMPLETE_PREFIX"
    assert evidence["complete"] is False
    assert evidence["complete_verification_passed"] is False
    assert type(evidence["verification_error_or_null"]) is str
    assert evidence["terminal_error_code_or_null"] is None
    assert evidence["terminal_error_detail_or_null"] is None
    assert evidence["open_span_ids_in_begin_order"] == ["supervisor-run"]
    assert evidence["open_span_contexts_in_begin_order"][0]["operation"] == (
        "SUPERVISOR_RUN"
    )
    assert evidence["partition_certificate_or_null"] is None
    assert evidence["failure_rows_retained_in_sequence_order"] == []
    assert evidence["cleanup_rows_retained_in_sequence_order"] == []
    assert evidence["last_durable_event_row_or_null"] == {
        "sequence": 1,
        "timestamp_ns": evidence["last_durable_event_row_or_null"][
            "timestamp_ns"
        ],
        "event_type": "BEGIN",
        "span_id": "supervisor-run",
        "operation": "SUPERVISOR_RUN",
        "stage_or_null": "RUN",
        "substep_or_null": None,
        "actor_or_null": None,
        "detail_or_null": "source_commit=" + "12" * 20,
        "error_code_or_null": None,
        "error_detail_or_null": None,
        "record_digest": evidence["final_event_record_digest_or_null"],
    }
    assert type(evidence["canonical_event_payloads_digest"]) is str
    assert TOOL._trajectory_complete_and_bound(contract, slot, evidence) is False
    stdout = CORE.canonical_campaign_json_bytes_v2(
        _error_wire("controller observed incomplete trajectory")
    )
    process = _process(stdout, 1)
    supervisor, external = _absent_artifacts()
    binding = TOOL._binding_record(
        contract,
        {"capsule_digest": _tagged("capsule")},
        slot,
        process,
        CORE.classify_supervisor_stdout_v2(stdout),
        evidence,
        supervisor,
        external,
        True,
        [],
        None,
        _source_replay("source-before"),
        _source_replay("source-after"),
        _future_freshness(),
        _cleanup_replay(),
    )
    assert binding["partition_certificate_or_null"] is None
    assert binding["failure_rows_retained_in_sequence_order"] == []
    assert binding["cleanup_rows_retained_in_sequence_order"] == []
    assert binding["last_durable_event_row_or_null"] == evidence[
        "last_durable_event_row_or_null"
    ]
    monkeypatch.setattr(TOOL, "_load_core", lambda _contract: CORE)
    observation = TOOL._build_slot_observation(
        contract,
        slot,
        process,
        CORE.classify_supervisor_stdout_v2(stdout),
        evidence,
        supervisor,
        external,
        source_before_certain=True,
        source_after_certain=True,
        docker_absent=True,
        external_binding_certain=True,
    )
    assert observation["trajectory_partition_certificate_or_null"] is None
    assert observation["trajectory_failure_rows"] == []
    assert observation["trajectory_cleanup_rows"] == []
    assert CORE.evaluate_slot_observation_v2(observation) == (
        CORE.TERMINAL_NONRETRYABLE_V2
    )


def test_large_verified_prefix_uses_bounded_certificate_without_fsync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def unexpected_fsync(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("synthetic bounded-certificate KAT must not fsync")

    monkeypatch.setattr(TOOL.os, "fsync", unexpected_fsync)
    contract = _contract()
    slot = _slot(contract)
    trace_root = tmp_path / "large-prefix-trace"
    trace_root.mkdir(mode=0o700)
    trace_root.chmod(0o700)
    slot["trace_root"] = os.fspath(trace_root)
    trace_id = str(slot["trajectory_trace_id"])

    prefix_count = 8_192
    events: list[dict[str, object]] = []
    for sequence in range(1, prefix_count + 1):
        span_ordinal = (sequence + 1) // 2
        events.append(
            _event_record(
                trace_id=trace_id,
                sequence=sequence,
                timestamp=sequence,
                event_type="BEGIN" if sequence % 2 else "OK",
                span_id=f"bulk-nonfailure-{span_ordinal:06d}",
                operation="BULK_NONFAILURE_EVENT",
                stage="RUN",
                substep=None,
                actor=None,
                detail="x" * 256,
                record_digest="sha256:" + f"{sequence:064x}",
            )
        )

    terminal_detail, failure_rows = _held_final_gap_rows()
    suffix = _event_records_for_failures(trace_id, failure_rows)
    failures_by_digest = {
        row["record_digest"]: row for row in failure_rows
    }
    for event in suffix:
        old_sequence = int(event["sequence"])
        event["sequence"] = prefix_count + old_sequence
        if event["event_type"] != "FAIL":
            event["record_digest"] = _tagged(
                f"large-suffix-{old_sequence}-{event['event_type']}"
            )
        failure = failures_by_digest.get(event["record_digest"])
        if failure is not None:
            failure["sequence"] = event["sequence"]
            failure["timestamp_ns"] = event["timestamp_ns"]
    events.extend(suffix)

    old_projection = [
        {
            field: deepcopy(event[field])
            for field in TOOL._TRAJECTORY_MINIMAL_EVENT_FIELDS
        }
        for event in events
    ]
    old_bytes = TOOL._canonical_json_bytes(
        {"event_skeleton_rows_in_sequence_order": old_projection}
    )
    assert len(old_bytes) > TOOL._MAX_LEDGER_BYTES

    payloads = [TOOL._canonical_json_bytes(event) for event in events]
    partition = TOOL._trajectory_partition_evidence(
        contract,
        payloads,
        trajectory_status="COMPLETE_FAIL",
        failure_rows_in_sequence_order=failure_rows,
    )
    certificate = partition["partition_certificate_or_null"]
    assert type(certificate) is dict
    assert len(TOOL._canonical_json_bytes(certificate)) <= (
        TOOL._MAX_PARTITION_CERTIFICATE_BYTES
    )
    assert certificate["origin_sequence"] == prefix_count + 5
    assert certificate["suffix_event_count"] == 6
    assert certificate["failure_event_count"] == 4
    assert certificate["failure_rows_retained_count"] == 4
    assert certificate["failure_rows_truncated"] is False
    assert certificate["cleanup_nonfailure_event_count"] == 2
    assert certificate["cleanup_rows_retained_count"] == 2
    assert certificate["cleanup_rows_truncated"] is False
    assert certificate["other_nonfailure_event_count"] == 0
    assert certificate["partitioned_event_count"] == 6
    assert certificate["unpartitioned_event_count"] == 0

    trajectory = _trajectory(
        slot,
        outcome="FAIL",
        failure_rows=_held_final_gap_rows()[1],
        terminal_error_detail=terminal_detail,
    )
    trajectory.update(
        {
            "event_count": len(events),
            "first_event_record_digest_or_null": events[0]["record_digest"],
            "final_event_record_digest_or_null": events[-1]["record_digest"],
            "failure_rows_retained_in_sequence_order": deepcopy(
                partition["failure_rows_retained_in_sequence_order"]
            ),
            "cleanup_rows_retained_in_sequence_order": deepcopy(
                partition["cleanup_rows_retained_in_sequence_order"]
            ),
            "partition_certificate_or_null": deepcopy(certificate),
            "last_durable_event_row_or_null": deepcopy(
                partition["last_durable_event_row_or_null"]
            ),
            "canonical_event_payloads_digest": partition[
                "canonical_event_payloads_digest"
            ],
        }
    )
    assert TOOL._trajectory_complete_and_bound(contract, slot, trajectory) is True
    process = _process(
        CORE.canonical_campaign_json_bytes_v2(_error_wire(terminal_detail)),
        1,
    )
    supervisor, external = _absent_artifacts()
    binding = TOOL._binding_record(
        contract,
        {"capsule_digest": _tagged("capsule")},
        slot,
        process,
        {},
        trajectory,
        supervisor,
        external,
        True,
        [],
        None,
        _source_replay("source-before"),
        _source_replay("source-after"),
        _future_freshness(),
        _cleanup_replay(),
    )
    binding_bytes = TOOL._canonical_json_bytes(binding)
    assert len(binding_bytes) <= TOOL._MAX_LEDGER_BYTES
    assert TOOL._strict_json(
        binding_bytes,
        "large-prefix bounded binding",
        TOOL._MAX_LEDGER_BYTES,
    ) == binding


def test_authorization_digest_exact_load_and_no_replace(
    tmp_path: Path,
) -> None:
    contract = _contract()
    capsule = {"capsule_digest": _tagged("capsule")}
    prefix = f"q05b-{tmp_path.name}-"
    with tempfile.TemporaryDirectory(prefix=prefix, dir="/tmp") as raw_root:
        root = Path(raw_root) / "authorization"
        root.mkdir(mode=0o700)
        root.chmod(0o700)
        contract["fixed_external_roots"]["authorization_root"] = str(root)
        contract["authorization_policy"]["authorization_root"] = str(root)
        authorization = TOOL._authorization_template(contract, capsule)
        unsigned = deepcopy(authorization)
        unsigned.pop("authorization_digest")
        assert authorization["authorization_digest"] == TOOL._domain_digest(
            contract["hash_domains"]["authorization"],
            unsigned,
        )
        basename = contract["authorization_policy"][
            "authorization_record_name"
        ]
        assert type(basename) is str
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            payload = TOOL._publish_immutable_at(
                root_fd, basename, authorization
            )
            with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
                TOOL._publish_immutable_at(root_fd, basename, authorization)
        finally:
            os.close(root_fd)
        assert error.value.code == "FAIL_CAMPAIGN_LEDGER"
        path = root / basename
        assert path.read_bytes() == payload
        status = path.stat(follow_symlinks=False)
        assert stat.S_IMODE(status.st_mode) == 0o444
        assert status.st_nlink == 1
        assert TOOL._load_authorization(contract, capsule) == authorization


def test_authorization_postpublication_result_failure_is_uncertain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    root = tmp_path / "owner-authorization"
    contract["fixed_external_roots"]["authorization_root"] = os.fspath(root)
    contract["authorization_policy"]["authorization_root"] = os.fspath(root)
    capsule = {"capsule_digest": _tagged("capsule")}
    authorization = TOOL._authorization_template(contract, capsule)
    context = {
        "capsule": capsule,
        "authorization_template": authorization,
    }
    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_preflight_context", lambda _contract: context)
    real_publish = TOOL._publish_immutable_at
    real_fspath = TOOL.os.fspath
    helper_succeeded = False

    def publish_then_mark(*args: object, **kwargs: object) -> bytes:
        nonlocal helper_succeeded
        result = real_publish(*args, **kwargs)
        helper_succeeded = True
        return result

    def fail_receipt_path(value: object) -> str:
        if helper_succeeded and isinstance(value, Path) and value.parent == root:
            raise MemoryError("injected authorization receipt construction failure")
        return real_fspath(value)

    monkeypatch.setattr(TOOL, "_publish_immutable_at", publish_then_mark)
    monkeypatch.setattr(TOOL.os, "fspath", fail_receipt_path)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL.create_phase3_q05b_shadow_retry_authorization_v2(
            capsule["capsule_digest"]
        )
    assert error.value.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
    assert "receipt construction failure" in error.value.detail
    basename = str(contract["authorization_policy"]["authorization_record_name"])
    assert (root / basename).is_file()


def test_authorization_postpublication_close_failure_is_uncertain_and_closes_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    root = tmp_path / "owner-authorization"
    contract["fixed_external_roots"]["authorization_root"] = os.fspath(root)
    contract["authorization_policy"]["authorization_root"] = os.fspath(root)
    capsule = {"capsule_digest": _tagged("capsule")}
    authorization = TOOL._authorization_template(contract, capsule)
    context = {
        "capsule": capsule,
        "authorization_template": authorization,
    }
    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_preflight_context", lambda _contract: context)
    real_create = TOOL._create_private_root
    real_publish = TOOL._publish_immutable_at
    real_close = TOOL.os.close
    captured_fds: list[int] = []
    helper_succeeded = False
    close_failed = False
    parent_closed = False

    def create_and_capture(path: Path, label: str) -> tuple[int, int]:
        result = real_create(path, label)
        captured_fds.extend(result)
        return result

    def publish_then_mark(*args: object, **kwargs: object) -> bytes:
        nonlocal helper_succeeded
        result = real_publish(*args, **kwargs)
        helper_succeeded = True
        return result

    def fail_root_close_once(descriptor: int) -> None:
        nonlocal close_failed, parent_closed
        if (
            helper_succeeded
            and captured_fds
            and descriptor == captured_fds[1]
            and not close_failed
        ):
            close_failed = True
            raise OSError("injected authorization root close failure")
        if captured_fds and descriptor == captured_fds[0]:
            parent_closed = True
        real_close(descriptor)

    monkeypatch.setattr(TOOL, "_create_private_root", create_and_capture)
    monkeypatch.setattr(TOOL, "_publish_immutable_at", publish_then_mark)
    monkeypatch.setattr(TOOL.os, "close", fail_root_close_once)
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL.create_phase3_q05b_shadow_retry_authorization_v2(
                capsule["capsule_digest"]
            )
        assert error.value.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
        assert "root close failure" in error.value.detail
        assert close_failed is True
        assert parent_closed is True
    finally:
        if captured_fds and close_failed:
            real_close(captured_fds[1])


@pytest.mark.parametrize(
    "fault_stage",
    (
        "write",
        "file_fsync",
        "directory_fsync",
        "replay_open",
        "replay_read",
        "pathname_stat",
    ),
)
def test_publication_fault_after_create_is_always_uncertain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fault_stage: str,
) -> None:
    root = tmp_path / fault_stage
    root.mkdir(mode=0o700)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    real_open = TOOL.os.open
    real_write = TOOL.os.write
    real_fsync = TOOL.os.fsync
    real_read = TOOL.os.read
    real_stat = TOOL.os.stat
    fsync_count = 0
    open_count = 0

    def injected_open(*args: object, **kwargs: object) -> int:
        nonlocal open_count
        open_count += 1
        if fault_stage == "replay_open" and open_count == 2:
            raise OSError("injected replay open failure")
        return real_open(*args, **kwargs)

    def injected_write(*args: object, **kwargs: object) -> int:
        if fault_stage == "write":
            raise OSError("injected write failure")
        return real_write(*args, **kwargs)

    def injected_fsync(*args: object, **kwargs: object) -> None:
        nonlocal fsync_count
        fsync_count += 1
        if fault_stage == "file_fsync" and fsync_count == 1:
            raise OSError("injected file fsync failure")
        if fault_stage == "directory_fsync" and fsync_count == 2:
            raise OSError("injected directory fsync failure")
        real_fsync(*args, **kwargs)

    def injected_read(*args: object, **kwargs: object) -> bytes:
        if fault_stage == "replay_read":
            raise OSError("injected replay read failure")
        return real_read(*args, **kwargs)

    def injected_stat(*args: object, **kwargs: object) -> os.stat_result:
        if fault_stage == "pathname_stat":
            raise OSError("injected pathname stat failure")
        return real_stat(*args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(TOOL.os, "open", injected_open)
            patch.setattr(TOOL.os, "write", injected_write)
            patch.setattr(TOOL.os, "fsync", injected_fsync)
            patch.setattr(TOOL.os, "read", injected_read)
            patch.setattr(TOOL.os, "stat", injected_stat)
            with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
                TOOL._publish_immutable_at(root_fd, "record.json", b"x\n")
    finally:
        os.close(root_fd)
    assert error.value.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
    assert (root / "record.json").exists()


@pytest.mark.parametrize(
    ("basename", "publication_label"),
    (
        ("supervisor.stdout", "slot-supervisor-stdout"),
        ("supervisor.stderr", "slot-supervisor-stderr"),
    ),
)
def test_immutable_publication_diagnostic_binds_exact_stream_step(
    tmp_path: Path,
    basename: str,
    publication_label: str,
) -> None:
    root = tmp_path / publication_label
    root.mkdir(mode=0o700)
    (root / basename).write_bytes(b"existing")
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._publish_immutable_at(
                root_fd,
                basename,
                b"new",
                publication_label=publication_label,
                known_failure_code="FAIL_CAMPAIGN_EVIDENCE",
            )
    finally:
        os.close(root_fd)
    assert error.value.code == "FAIL_CAMPAIGN_EVIDENCE"
    assert error.value.receipt == {
        "schema_version": TOOL._PUBLICATION_RECEIPT_SCHEMA_V2,
        "publication_outcome": "KNOWN_NO_PUBLICATION",
        "publication_label": publication_label,
        "publication_basename": basename,
    }
    assert f"publication_label={publication_label}" in error.value.detail
    assert f"basename={basename}" in error.value.detail


def test_external_artifact_posthelper_replay_failure_is_publication_uncertain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "supervisor-artifact.json"
    raw = b'{"status":"valid"}\n'
    source.write_bytes(raw)
    source.chmod(0o444)
    source_status = source.stat()
    target_parent = tmp_path / "external-artifact"
    target = target_parent / "artifact.json"
    slot = {
        "supervisor_artifact_path": os.fspath(source),
        "artifact_parent_root": os.fspath(target_parent),
        "artifact_path": os.fspath(target),
    }
    evidence = {
        "state": "VALID",
        "replay_passed": True,
        "raw_sha256_or_null": sha256(raw).hexdigest(),
        "file_identity_or_null": {
            "dev": source_status.st_dev,
            "ino": source_status.st_ino,
            "mode": stat.S_IMODE(source_status.st_mode),
            "nlink": source_status.st_nlink,
            "uid": source_status.st_uid,
            "size": source_status.st_size,
        },
    }
    receipt = {"argv": ["post-helper-read"], "return_code": 7}

    def fail_posthelper_read(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[bytes, os.stat_result]:
        raise TOOL.ShadowRetryCampaignErrorV2(
            "FAIL_CAMPAIGN_FILESYSTEM",
            "injected external post-helper replay failure",
            terminal=True,
            receipt=receipt,
        )

    monkeypatch.setattr(TOOL, "_read_regular_at_nofollow", fail_posthelper_read)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._publish_external_artifact(slot, evidence)
    assert target.is_file()
    assert error.value.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
    assert "injected external post-helper replay failure" in error.value.detail
    assert type(error.value.receipt) is dict
    assert error.value.receipt["origin_error_or_null"][
        "error_receipt_or_null"
    ] == receipt


def test_external_artifact_double_fault_preserves_primary_publication_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "supervisor-artifact.json"
    raw = b'{"status":"valid"}\n'
    source.write_bytes(raw)
    source.chmod(0o444)
    source_status = source.stat()
    target_parent = tmp_path / "external-artifact"
    target = target_parent / "artifact.json"
    slot = {
        "supervisor_artifact_path": os.fspath(source),
        "artifact_parent_root": os.fspath(target_parent),
        "artifact_path": os.fspath(target),
    }
    evidence = {
        "state": "VALID",
        "replay_passed": True,
        "raw_sha256_or_null": sha256(raw).hexdigest(),
        "file_identity_or_null": {
            "dev": source_status.st_dev,
            "ino": source_status.st_ino,
            "mode": stat.S_IMODE(source_status.st_mode),
            "nlink": source_status.st_nlink,
            "uid": source_status.st_uid,
            "size": source_status.st_size,
        },
    }
    primary_receipt = {"primary": "post-helper replay"}
    real_create = TOOL._create_private_root
    real_close = TOOL.os.close
    root_fd = -1
    injected_close = False

    def capture_root(path: Path, label: str) -> tuple[int, int]:
        nonlocal root_fd
        pair = real_create(path, label)
        root_fd = pair[1]
        return pair

    def fail_primary(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[bytes, os.stat_result]:
        raise TOOL.ShadowRetryCampaignErrorV2(
            "FAIL_CAMPAIGN_FILESYSTEM",
            "injected primary external replay failure",
            terminal=True,
            receipt=primary_receipt,
        )

    def fail_root_close(descriptor: int) -> None:
        nonlocal injected_close
        if descriptor == root_fd and target.exists() and not injected_close:
            injected_close = True
            raise OSError("injected secondary external root close failure")
        real_close(descriptor)

    monkeypatch.setattr(TOOL, "_create_private_root", capture_root)
    monkeypatch.setattr(TOOL, "_read_regular_at_nofollow", fail_primary)
    monkeypatch.setattr(TOOL.os, "close", fail_root_close)
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._publish_external_artifact(slot, evidence)
        assert error.value.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
        assert "primary external replay failure" in error.value.detail
        assert "secondary external root close" not in error.value.detail
        assert type(error.value.receipt) is dict
        assert error.value.receipt["origin_error_or_null"][
            "error_receipt_or_null"
        ] == primary_receipt
        assert injected_close is True
    finally:
        if injected_close and root_fd >= 0:
            real_close(root_fd)


def test_ledger_posthelper_root_failure_is_publication_uncertain_without_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    root.mkdir(mode=0o700)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    contract = _contract()
    ledger = object.__new__(TOOL._LedgerV2)
    ledger.contract = contract
    ledger.capsule = {"capsule_digest": _tagged("capsule")}
    ledger.authorization = {"authorization_digest": _tagged("authorization")}
    ledger.root_path = root
    ledger.root_fd = root_fd
    ledger.root_identity = TOOL._verify_held_directory_path(
        root,
        root_fd,
        "ledger post-helper KAT baseline",
    )
    ledger.previous_record_digest = None
    ledger.sequence = 0
    ledger.published_names = []
    ledger.published_records = []
    ledger.completion_commit_verified = False
    ledger.allowed_names = ["campaign.intent.json"]
    ledger.last_name_index = -1
    original_verify = TOOL._verify_held_directory_path
    verify_count = 0

    def fail_second_verify(
        path: Path,
        descriptor: int,
        label: str,
    ) -> dict[str, object]:
        nonlocal verify_count
        verify_count += 1
        if verify_count == 2:
            raise TOOL.ShadowRetryCampaignErrorV2(
                "FAIL_CAMPAIGN_FILESYSTEM",
                "injected ledger post-helper root failure",
                terminal=True,
                receipt={"root_replay": "failed"},
            )
        return original_verify(path, descriptor, label)

    def publish_then_commit(
        *_args: object,
        **kwargs: object,
    ) -> bytes:
        callback = kwargs["commit_callback"]
        assert callable(callback)
        callback()
        return b"publication helper reported success"

    monkeypatch.setattr(TOOL, "_publish_immutable_at", publish_then_commit)
    monkeypatch.setattr(TOOL, "_verify_held_directory_path", fail_second_verify)
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            ledger.publish(
                "campaign.intent.json",
                "CAMPAIGN_INTENT",
                "INTENT",
                {"purpose": "post-helper fault KAT"},
            )
        assert error.value.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
        assert "injected ledger post-helper root failure" in error.value.detail
        assert type(error.value.receipt) is dict
        assert error.value.receipt["origin_error_or_null"][
            "error_receipt_or_null"
        ] == {"root_replay": "failed"}
        assert ledger.sequence == 0
        assert ledger.published_names == []
    finally:
        os.close(root_fd)


def test_ledger_next_state_is_prebuilt_before_immutable_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class ExplodingRecordList(list[dict[str, object]]):
        def __iter__(self) -> object:
            raise MemoryError("injected ledger next-state prebuild failure")

    root = tmp_path / "ledger"
    root.mkdir(mode=0o700)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    contract = _contract()
    ledger = object.__new__(TOOL._LedgerV2)
    ledger.contract = contract
    ledger.capsule = {"capsule_digest": _tagged("capsule")}
    ledger.authorization = {"authorization_digest": _tagged("authorization")}
    ledger.root_path = root
    ledger.root_fd = root_fd
    ledger.root_identity = TOOL._verify_held_directory_path(
        root,
        root_fd,
        "ledger state-commit KAT baseline",
    )
    ledger.previous_record_digest = None
    ledger.sequence = 0
    ledger.published_names = []
    ledger.published_records = ExplodingRecordList()
    ledger.completion_commit_verified = False
    ledger.allowed_names = ["campaign.intent.json"]
    ledger.last_name_index = -1
    helper_calls = 0

    def mark_publication(*_args: object, **_kwargs: object) -> bytes:
        nonlocal helper_calls
        helper_calls += 1
        return b"publication helper reported success"

    monkeypatch.setattr(
        TOOL,
        "_publish_immutable_at",
        mark_publication,
    )
    try:
        with pytest.raises(MemoryError) as error:
            ledger.publish(
                "campaign.intent.json",
                "CAMPAIGN_INTENT",
                "INTENT",
                {"purpose": "state-commit fault KAT"},
            )
        assert "next-state prebuild failure" in str(error.value)
        assert helper_calls == 0
        assert ledger.last_name_index == -1
        assert ledger.published_names == []
    finally:
        os.close(root_fd)


def test_ledger_close_failure_after_publication_is_uncertain_and_closes_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        tmp_path / "ledger-root"
    )
    ledger = TOOL._LedgerV2(
        contract,
        {"capsule_digest": _tagged("capsule")},
        {"authorization_digest": _tagged("authorization")},
    )
    ledger.publish(
        "campaign.intent.json",
        "CAMPAIGN_INTENT",
        "AUTHORIZED_CAMPAIGN_INTENT_STAGED",
        {"test": True},
    )
    root_fd = ledger.root_fd
    parent_fd = ledger.parent_fd
    real_close = TOOL.os.close
    root_failed = False
    parent_closed = False

    def fail_root_close_once(descriptor: int) -> None:
        nonlocal root_failed, parent_closed
        if descriptor == root_fd and not root_failed:
            root_failed = True
            raise OSError("injected ledger root close failure")
        if descriptor == parent_fd:
            parent_closed = True
        real_close(descriptor)

    monkeypatch.setattr(TOOL.os, "close", fail_root_close_once)
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            ledger.close()
        assert error.value.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
        assert "ledger root close failure" in error.value.detail
        assert root_failed is True
        assert parent_closed is True
        assert ledger.root_fd is None
        assert ledger.parent_fd is None
    finally:
        if root_failed:
            real_close(root_fd)


def test_ledger_close_double_fault_preserves_spawn_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        tmp_path / "ledger-root"
    )
    ledger = TOOL._LedgerV2(
        contract,
        {"capsule_digest": _tagged("capsule")},
        {"authorization_digest": _tagged("authorization")},
    )
    ledger.publish(
        "campaign.intent.json",
        "CAMPAIGN_INTENT",
        "AUTHORIZED_CAMPAIGN_INTENT_STAGED",
        {"test": True},
    )
    root_fd = ledger.root_fd
    real_close = TOOL.os.close
    root_failed = False

    def fail_root_close_once(descriptor: int) -> None:
        nonlocal root_failed
        if descriptor == root_fd and not root_failed:
            root_failed = True
            raise OSError("injected ledger close double fault")
        real_close(descriptor)

    monkeypatch.setattr(TOOL.os, "close", fail_root_close_once)
    original = TOOL.ShadowRetryCampaignErrorV2(
        "FAIL_CAMPAIGN_SPAWN_UNCERTAIN",
        "injected spawn identity uncertainty",
        terminal=True,
    )
    try:
        try:
            raise original
        except TOOL.ShadowRetryCampaignErrorV2:
            with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
                ledger.close()
        assert error.value is original
        assert root_failed is True
    finally:
        if root_failed:
            real_close(root_fd)


def test_known_evidence_precreate_failure_is_not_ledger_uncertainty(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    (root / "stdout").write_bytes(b"preexisting")
    descriptor = os.open(
        root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
    )
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._publish_immutable_at(
                descriptor,
                "stdout",
                b"new",
                known_failure_code="FAIL_CAMPAIGN_EVIDENCE",
            )
    finally:
        os.close(descriptor)
    assert error.value.code == "FAIL_CAMPAIGN_EVIDENCE"
    assert (root / "stdout").read_bytes() == b"preexisting"


def test_o_creat_return_gap_is_publication_uncertain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "publication-root"
    root.mkdir(mode=0o700)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    real_open = TOOL.os.open
    real_close = TOOL.os.close

    def create_then_interrupt(
        path: object,
        flags: int,
        *args: object,
        **kwargs: object,
    ) -> int:
        descriptor = real_open(path, flags, *args, **kwargs)
        if path == "record.json" and flags & os.O_CREAT:
            real_close(descriptor)
            raise KeyboardInterrupt("injected after kernel O_CREAT")
        return descriptor

    monkeypatch.setattr(TOOL.os, "open", create_then_interrupt)
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._publish_immutable_at(
                root_fd,
                "record.json",
                {"value": 1},
            )
        assert error.value.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
        assert (root / "record.json").exists()
    finally:
        real_close(root_fd)


def test_prepared_ledger_mapping_is_frozen_before_check_and_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        tmp_path / "frozen-ledger"
    )
    ledger = TOOL._LedgerV2(
        contract,
        {"capsule_digest": _tagged("capsule")},
        {"authorization_digest": _tagged("authorization")},
    )
    prepared = ledger.prepare(
        "campaign.intent.json",
        "CAMPAIGN_INTENT",
        "AUTHORIZED_CAMPAIGN_INTENT_STAGED",
        {"nested": {"value": "checked"}},
    )
    frozen = deepcopy(prepared)
    real_publish = TOOL._publish_immutable_at

    def mutate_original_then_publish(
        directory_fd: int,
        basename: str,
        value_or_bytes: Mapping[str, object] | bytes,
        **kwargs: object,
    ) -> bytes:
        assert type(value_or_bytes) is bytes
        prepared["body"]["nested"]["value"] = "mutated-after-freeze"
        return real_publish(
            directory_fd,
            basename,
            value_or_bytes,
            **kwargs,
        )

    monkeypatch.setattr(
        TOOL,
        "_publish_immutable_at",
        mutate_original_then_publish,
    )
    try:
        published = ledger.publish_prepared("campaign.intent.json", prepared)
        assert published == frozen
        raw = Path(
            str(contract["fixed_external_roots"]["ledger_root"])
        ).joinpath("campaign.intent.json").read_bytes()
        assert TOOL._strict_json(raw, "frozen ledger KAT") == frozen
    finally:
        ledger.close()


@pytest.mark.parametrize(
    "fault_stage",
    (
        "external_artifact_return",
        "postbinding_freshness",
        "postoutcome_evidence_close",
        "spawn_uncertain_evidence_close",
    ),
)
def test_slot_postbinding_failures_never_fall_through_to_a_second_terminal_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fault_stage: str,
) -> None:
    contract = _contract()
    slot = _slot(contract)
    slot.update(
        {
            "clone_root": os.fspath(tmp_path / "clone"),
            "clone_project_root": os.fspath(tmp_path / "clone" / "Hegel Machine"),
            "cargo_cache_root": os.fspath(tmp_path / "cargo"),
            "trace_root": os.fspath(tmp_path / "trace"),
            "evidence_root": os.fspath(tmp_path / "evidence"),
            "stdout_path": os.fspath(tmp_path / "evidence" / "stdout.bin"),
            "stderr_path": os.fspath(tmp_path / "evidence" / "stderr.bin"),
            "supervisor_artifact_path": os.fspath(
                tmp_path / "clone" / "Hegel Machine" / "artifact.json"
            ),
            "artifact_parent_root": os.fspath(tmp_path / "artifact-parent"),
            "artifact_path": os.fspath(tmp_path / "artifact-parent" / "artifact.json"),
            "external_trajectory_binding_path": os.fspath(
                tmp_path / "evidence" / "trajectory-binding.json"
            ),
            "slot_result_path": os.fspath(tmp_path / "evidence" / "outcome.json"),
        }
    )
    capsule = {"capsule_digest": _tagged("capsule")}
    context = {
        "slots": [slot],
        "cargo_rows": [],
        "cargo_identity": {"content_manifest_digest": _tagged("cargo")},
    }
    ledger = _FakeLedger()
    invocation_counter = [0]
    freshness = _future_freshness()
    source_before = _source_replay("source-before")
    source_before["clone_closure_rows_internal"] = []
    source_after = _source_replay("source-after")
    supervisor, external = _absent_artifacts()
    trajectory = _trajectory(
        slot,
        outcome=("OK" if fault_stage == "external_artifact_return" else "FAIL"),
        terminal_error_detail=(
            None
            if fault_stage == "external_artifact_return"
            else "injected supervisor failure"
        ),
    )
    binding_published = False
    outcome_published = False
    evidence_fd: int | None = None
    close_failed = False
    real_create_evidence = TOOL._create_evidence_root
    real_close = TOOL.os.close

    class FakeCore:
        TERMINAL_NONRETRYABLE_V2 = CORE.TERMINAL_NONRETRYABLE_V2

        @staticmethod
        def classify_supervisor_stdout_v2(_raw: bytes) -> dict[str, object]:
            return {}

        @staticmethod
        def evaluate_slot_observation_v2(
            _observation: Mapping[str, object],
        ) -> str:
            return CORE.TERMINAL_NONRETRYABLE_V2

    def freshness_replay(*_args: object, **_kwargs: object) -> dict[str, object]:
        if (
            fault_stage == "postbinding_freshness"
            and binding_published
            and not outcome_published
        ):
            raise TOOL.ShadowRetryCampaignErrorV2(
                "FAIL_CAMPAIGN_FRESHNESS",
                "injected postbinding freshness failure",
                terminal=True,
            )
        return deepcopy(freshness)

    def bounded_process(*_args: object, **kwargs: object) -> dict[str, object]:
        callback = kwargs.get("before_popen")
        assert callable(callback)
        callback()
        if fault_stage == "spawn_uncertain_evidence_close":
            raise TOOL.ShadowRetryCampaignErrorV2(
                "FAIL_CAMPAIGN_SPAWN_UNCERTAIN",
                "injected spawn uncertainty",
                terminal=True,
                receipt={"cleanup_certain": False},
            )
        return _process(
            b"{}\n",
            0 if fault_stage == "external_artifact_return" else 1,
        )

    def create_evidence(
        slot_value: Mapping[str, object],
    ) -> tuple[int, int]:
        nonlocal evidence_fd
        result = real_create_evidence(slot_value)
        evidence_fd = result[1]
        return result

    def publish_evidence(
        _directory_fd: int,
        basename: str,
        _value: object,
        **_kwargs: object,
    ) -> bytes:
        nonlocal binding_published, outcome_published
        if basename == "trajectory-binding.json":
            binding_published = True
        elif basename == "outcome.json":
            outcome_published = True
        return b"published"

    def close_with_injected_outcome_fault(descriptor: int) -> None:
        nonlocal close_failed
        if (
            fault_stage
            in {
                "postoutcome_evidence_close",
                "spawn_uncertain_evidence_close",
            }
            and (
                outcome_published
                or fault_stage == "spawn_uncertain_evidence_close"
            )
            and evidence_fd is not None
            and descriptor == evidence_fd
            and not close_failed
        ):
            close_failed = True
            raise OSError("injected postoutcome evidence close failure")
        real_close(descriptor)

    monkeypatch.setattr(TOOL, "_load_core", lambda _contract: FakeCore)
    monkeypatch.setattr(TOOL, "_slot_argv", lambda *_args: ["/usr/bin/true"])
    monkeypatch.setattr(TOOL, "_runtime_environment", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(TOOL, "_slot_roots_absence_evidence", freshness_replay)
    monkeypatch.setattr(TOOL, "_prepare_clone", lambda *_args: None)
    monkeypatch.setattr(
        TOOL,
        "_copy_tree_exact",
        lambda *_args, **_kwargs: {
            "content_manifest_digest": context["cargo_identity"][
                "content_manifest_digest"
            ]
        },
    )
    monkeypatch.setattr(
        TOOL,
        "_source_state_replay",
        lambda *_args, **_kwargs: deepcopy(source_before),
    )
    monkeypatch.setattr(TOOL, "_safe_docker_inventory", lambda _contract: (True, [], None))
    monkeypatch.setattr(TOOL, "_create_evidence_root", create_evidence)
    monkeypatch.setattr(
        TOOL,
        "_assert_existing_directory_roots_nonaliased",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        TOOL,
        "_augment_source_replay_with_alias_rows",
        lambda value, _rows: deepcopy(dict(value)),
    )
    monkeypatch.setattr(TOOL, "_bounded_process", bounded_process)
    monkeypatch.setattr(TOOL, "_publish_raw_streams", lambda *_args: None)
    monkeypatch.setattr(
        TOOL,
        "_postpublication_directory_identity",
        lambda _path, _descriptor, _label, expected: deepcopy(dict(expected)),
    )
    monkeypatch.setattr(
        TOOL,
        "_safe_source_state_replay",
        lambda *_args, **_kwargs: deepcopy(source_after),
    )
    monkeypatch.setattr(
        TOOL,
        "_enforce_clone_closure",
        lambda _slot, _before, after: deepcopy(dict(after)),
    )
    monkeypatch.setattr(
        TOOL,
        "_safe_trajectory_evidence",
        lambda *_args, **_kwargs: deepcopy(trajectory),
    )
    monkeypatch.setattr(TOOL, "_artifact_replay", lambda *_args: deepcopy(supervisor))
    monkeypatch.setattr(
        TOOL,
        "_trajectory_complete_and_bound",
        lambda *_args: fault_stage == "external_artifact_return",
    )
    monkeypatch.setattr(
        TOOL,
        "_artifact_matches_success_wire",
        lambda *_args: fault_stage == "external_artifact_return",
    )

    def external_artifact_then_return_fault(
        *_args: object,
        **kwargs: object,
    ) -> dict[str, object]:
        callback = kwargs["commit_callback"]
        assert callable(callback)
        callback({"state": "VALID", "replay_passed": True})
        raise KeyboardInterrupt("injected external artifact return gap")

    if fault_stage == "external_artifact_return":
        monkeypatch.setattr(
            TOOL,
            "_publish_external_artifact",
            external_artifact_then_return_fault,
        )
    monkeypatch.setattr(TOOL, "_absent_external_artifact", lambda *_args: deepcopy(external))
    monkeypatch.setattr(TOOL, "_cleanup_replay_evidence", lambda *_args: _cleanup_replay())
    monkeypatch.setattr(
        TOOL,
        "_binding_record",
        lambda *_args: {"binding_digest": _tagged("binding")},
    )
    monkeypatch.setattr(TOOL, "_publish_immutable_at", publish_evidence)
    monkeypatch.setattr(TOOL, "_build_slot_observation", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        TOOL,
        "_slot_outcome_record",
        lambda *_args: {"record_digest": _tagged("outcome")},
    )
    monkeypatch.setattr(TOOL.os, "close", close_with_injected_outcome_fault)
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._run_one_slot(
                contract,
                context,
                capsule,
                ledger,  # type: ignore[arg-type]
                slot,
                invocation_counter,
            )
        assert error.value.code == (
            "FAIL_CAMPAIGN_SPAWN_UNCERTAIN"
            if fault_stage == "spawn_uncertain_evidence_close"
            else "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
        )
        if fault_stage == "external_artifact_return":
            assert binding_published is False
            assert outcome_published is False
            assert "external artifact return gap" in error.value.detail
        elif fault_stage == "postbinding_freshness":
            assert binding_published is True
            assert outcome_published is False
            assert "postbinding freshness failure" in error.value.detail
        elif fault_stage == "postoutcome_evidence_close":
            assert binding_published is True
            assert outcome_published is True
            assert close_failed is True
            assert "postoutcome evidence close failure" in error.value.detail
        else:
            assert binding_published is False
            assert outcome_published is False
            assert close_failed is True
            assert "injected spawn uncertainty" in error.value.detail
    finally:
        if close_failed and evidence_fd is not None:
            real_close(evidence_fd)


def test_consumption_callback_failure_never_spawns_and_preserves_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        TOOL, "_require_single_threaded_spawn_controller_v2", lambda: None
    )
    popen_calls = 0

    def forbidden_popen(*_args: object, **_kwargs: object) -> object:
        nonlocal popen_calls
        popen_calls += 1
        raise AssertionError("Popen must not be reached")

    monkeypatch.setattr(TOOL.subprocess, "Popen", forbidden_popen)

    def typed_callback() -> None:
        raise TOOL.ShadowRetryCampaignErrorV2(
            "FAIL_CAMPAIGN_LEDGER",
            "injected known no-publication failure",
            terminal=True,
        )

    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as typed_error:
        TOOL._bounded_process(
            ["/usr/bin/true"],
            cwd=os.fspath(tmp_path),
            env={},
            timeout_seconds=1,
            stdout_max_bytes=1,
            stderr_max_bytes=1,
            before_popen=typed_callback,
        )
    assert typed_error.value.code == "FAIL_CAMPAIGN_LEDGER"

    def untyped_callback() -> None:
        raise OSError("injected ambiguous callback failure")

    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as untyped_error:
        TOOL._bounded_process(
            ["/usr/bin/true"],
            cwd=os.fspath(tmp_path),
            env={},
            timeout_seconds=1,
            stdout_max_bytes=1,
            stderr_max_bytes=1,
            before_popen=untyped_callback,
        )
    assert (
        untyped_error.value.code
        == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
    )
    assert popen_calls == 0


def test_registered_spawn_restores_child_signal_mask_and_leaks_no_fd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        TOOL, "_require_single_threaded_spawn_controller_v2", lambda: None
    )
    program = (
        "import json,os,signal;"
        "m=sorted(int(x) for x in signal.pthread_sigmask(signal.SIG_BLOCK,set()));"
        "f={};"
        "[(f.__setitem__(n,os.readlink('/proc/self/fd/'+n)) "
        "if os.path.exists('/proc/self/fd/'+n) else None) "
        "for n in os.listdir('/proc/self/fd')];"
        "print(json.dumps({'mask':m,'fds':f,'pid':os.getpid(),"
        "'pgrp':os.getpgrp(),'sid':os.getsid(0)},"
        "sort_keys=True,separators=(',',':')))"
    )
    result = TOOL._bounded_process(
        ["/usr/bin/python3.10", "-I", "-S", "-B", "-c", program],
        cwd=os.fspath(tmp_path),
        env={"PATH": "/usr/bin:/bin"},
        timeout_seconds=5,
        stdout_max_bytes=4096,
        stderr_max_bytes=4096,
    )
    decoded = TOOL._strict_json(
        result["stdout"], "registered spawn child state", 4096
    )
    assert decoded["mask"] == []
    assert set(decoded["fds"]) <= {"0", "1", "2"}
    assert decoded["pid"] == decoded["pgrp"] == decoded["sid"]
    assert result["return_code"] == 0
    assert result["process_group_absent"] is True


def test_spawn_uses_source_bound_launcher_without_preexec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[tuple[list[str], dict[str, object]]] = []
    masks_at_popen: list[set[object]] = []
    real_init = TOOL.subprocess.Popen.__init__
    parent_mask_before = TOOL.signal.pthread_sigmask(TOOL.signal.SIG_BLOCK, set())

    def inspect_init(process: object, args: object, **kwargs: object) -> None:
        assert type(args) is list
        observed.append((list(args), dict(kwargs)))
        masks_at_popen.append(
            TOOL.signal.pthread_sigmask(TOOL.signal.SIG_BLOCK, set())
        )
        real_init(process, args, **kwargs)

    monkeypatch.setattr(TOOL.subprocess.Popen, "__init__", inspect_init)
    result = TOOL._bounded_process(
        ["/usr/bin/true"],
        cwd=os.fspath(tmp_path),
        env={"PATH": "/usr/bin:/bin"},
        timeout_seconds=5,
        stdout_max_bytes=4096,
        stderr_max_bytes=4096,
    )
    assert result["return_code"] == 0
    assert len(observed) == 1
    launcher_argv, kwargs = observed[0]
    assert launcher_argv[:7] == [
        TOOL._SPAWN_LAUNCHER_INTERPRETER_V2,
        "-I",
        "-S",
        "-B",
        "-X",
        "utf8",
        "-c",
    ]
    assert launcher_argv[7] == TOOL._SPAWN_LAUNCHER_SOURCE_V2
    assert launcher_argv[-1] == "/usr/bin/true"
    assert "preexec_fn" not in kwargs
    assert TOOL.signal.SIGINT in masks_at_popen[0]
    assert (
        TOOL.signal.pthread_sigmask(TOOL.signal.SIG_BLOCK, set())
        == parent_mask_before
    )


def test_sigchld_ignore_is_rejected_before_consumption_or_spawn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    callback_calls = 0
    popen_calls = 0
    original = TOOL.signal.getsignal(TOOL.signal.SIGCHLD)

    def callback() -> None:
        nonlocal callback_calls
        callback_calls += 1

    def forbidden_init(*_args: object, **_kwargs: object) -> None:
        nonlocal popen_calls
        popen_calls += 1
        raise AssertionError("Popen must not be reached")

    monkeypatch.setattr(TOOL.subprocess.Popen, "__init__", forbidden_init)
    TOOL.signal.signal(TOOL.signal.SIGCHLD, TOOL.signal.SIG_IGN)
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._bounded_process(
                ["/usr/bin/true"],
                cwd=os.fspath(tmp_path),
                env={},
                timeout_seconds=1,
                stdout_max_bytes=1,
                stderr_max_bytes=1,
                before_popen=callback,
            )
    finally:
        TOOL.signal.signal(TOOL.signal.SIGCHLD, original)
    assert error.value.code == "FAIL_CAMPAIGN_PROCESS"
    assert "SIGCHLD=SIG_DFL" in error.value.detail
    assert callback_calls == 0
    assert popen_calls == 0


def test_thread_created_by_consumption_callback_is_rejected_before_spawn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    release = threading.Event()
    started = threading.Event()
    worker: list[threading.Thread] = []
    gate_calls = 0
    popen_calls = 0

    def gated_proof() -> None:
        nonlocal gate_calls
        gate_calls += 1
        if gate_calls == 2:
            _REAL_REQUIRE_SINGLE_THREADED_SPAWN()

    def callback() -> None:
        thread = threading.Thread(
            target=lambda: (started.set(), release.wait(5)),
        )
        worker.append(thread)
        thread.start()
        assert started.wait(1)

    def forbidden_init(*_args: object, **_kwargs: object) -> None:
        nonlocal popen_calls
        popen_calls += 1
        raise AssertionError("Popen must not be reached")

    monkeypatch.setattr(
        TOOL, "_require_single_threaded_spawn_controller_v2", gated_proof
    )
    monkeypatch.setattr(TOOL.subprocess.Popen, "__init__", forbidden_init)
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._bounded_process(
                ["/usr/bin/true"],
                cwd=os.fspath(tmp_path),
                env={},
                timeout_seconds=1,
                stdout_max_bytes=1,
                stderr_max_bytes=1,
                before_popen=callback,
            )
    finally:
        release.set()
        for thread in worker:
            thread.join(timeout=2)
    assert error.value.code == "FAIL_CAMPAIGN_PROCESS"
    assert "not exactly single-threaded" in error.value.detail
    assert gate_calls == 2
    assert popen_calls == 0


def test_launcher_preserves_actual_exit_and_maps_actual_exec_failure(
    tmp_path: Path,
) -> None:
    exited = TOOL._bounded_process(
        [
            "/usr/bin/python3.10",
            "-I",
            "-S",
            "-B",
            "-c",
            "import sys;sys.exit(37)",
        ],
        cwd=os.fspath(tmp_path),
        env={},
        timeout_seconds=5,
        stdout_max_bytes=4096,
        stderr_max_bytes=4096,
    )
    assert exited["return_code"] == 37
    assert exited["child_reaped"] is True
    missing = TOOL._bounded_process(
        ["/definitely/missing/q05b-actual-executable"],
        cwd=os.fspath(tmp_path),
        env={},
        timeout_seconds=5,
        stdout_max_bytes=4096,
        stderr_max_bytes=4096,
    )
    assert missing["return_code"] == 127
    assert missing["stderr"] == b"HEGEL_Q05B_LAUNCHER_ACTUAL_EXEC_FAILED\n"
    assert missing["child_reaped"] is True


def test_missing_initial_launcher_never_signals_or_probes_reaped_pid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    kill_calls: list[tuple[object, object]] = []
    probe_calls: list[object] = []

    def forbidden_killpg(pid: object, value: object) -> None:
        kill_calls.append((pid, value))
        raise AssertionError("already-reaped PID must not be signalled")

    def forbidden_probe(pid: object) -> bool:
        probe_calls.append(pid)
        raise AssertionError("already-reaped PID must not be probed")

    monkeypatch.setattr(
        TOOL,
        "_SPAWN_LAUNCHER_INTERPRETER_V2",
        "/definitely/missing/q05b-launcher",
    )
    monkeypatch.setattr(TOOL.os, "killpg", forbidden_killpg)
    monkeypatch.setattr(TOOL, "_process_group_present", forbidden_probe)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._bounded_process(
            ["/usr/bin/true"],
            cwd=os.fspath(tmp_path),
            env={},
            timeout_seconds=5,
            stdout_max_bytes=4096,
            stderr_max_bytes=4096,
        )
    assert error.value.code == "FAIL_CAMPAIGN_SPAWN_UNCERTAIN"
    assert error.value.receipt["registration_state"] == "MISSING"
    assert error.value.receipt["cleanup_kill_attempted"] is False
    assert error.value.receipt["cleanup_direct_child_reaped"] is True
    assert error.value.receipt["cleanup_process_group_absent"] is False
    assert error.value.receipt["later_ledger_write_permitted"] is False
    assert kill_calls == []
    assert probe_calls == []


def test_spawn_controller_rejects_unproven_or_multiple_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(TOOL.os, "listdir", lambda _path: ["11", "12"])
    monkeypatch.setattr(TOOL.os, "getpid", lambda: 11)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        _REAL_REQUIRE_SINGLE_THREADED_SPAWN()
    assert error.value.code == "FAIL_CAMPAIGN_PROCESS"
    assert "not exactly single-threaded" in error.value.detail


@pytest.mark.parametrize(
    "store_name",
    ("pid", "_child_created"),
)
def test_registered_spawn_recovers_popen_internal_store_windows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    store_name: str,
) -> None:
    monkeypatch.setattr(
        TOOL, "_require_single_threaded_spawn_controller_v2", lambda: None
    )
    code = TOOL.subprocess.Popen._execute_child.__code__
    target_offset = next(
        instruction.offset
        for instruction in dis.get_instructions(code)
        if instruction.opname == "STORE_ATTR"
        and instruction.argval == store_name
    )
    injected = False

    def trace(frame: object, event: str, _arg: object) -> object:
        nonlocal injected
        if getattr(frame, "f_code") is code:
            frame.f_trace_opcodes = True
            if event == "opcode" and frame.f_lasti == target_offset:
                injected = True
                sys.settrace(None)
                raise KeyboardInterrupt(f"injected before {store_name} store")
        return trace

    sys.settrace(trace)
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._bounded_process(
                [
                    "/usr/bin/python3.10", "-I", "-S", "-B", "-c",
                    "import time;time.sleep(60)",
                ],
                cwd=os.fspath(tmp_path),
                env={"PATH": "/usr/bin:/bin"},
                timeout_seconds=5,
                stdout_max_bytes=4096,
                stderr_max_bytes=4096,
                reap_seconds=2,
            )
    finally:
        sys.settrace(None)
    assert injected is True
    assert error.value.code == "FAIL_CAMPAIGN_PROCESS"
    receipt = error.value.receipt
    assert type(receipt) is dict
    assert receipt["registration_state"] == "VALID"
    assert receipt["cleanup_direct_child_reaped"] is True
    assert receipt["cleanup_process_group_absent"] is True
    assert receipt["cleanup_certain"] is True


def test_registered_spawn_recovers_popen_return_to_caller_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        TOOL, "_require_single_threaded_spawn_controller_v2", lambda: None
    )
    real_init = TOOL.subprocess.Popen.__init__

    def init_then_interrupt(process: object, *args: object, **kwargs: object) -> None:
        real_init(process, *args, **kwargs)
        raise KeyboardInterrupt("injected after Popen init return")

    monkeypatch.setattr(TOOL.subprocess.Popen, "__init__", init_then_interrupt)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._bounded_process(
            [
                "/usr/bin/python3.10", "-I", "-S", "-B", "-c",
                "import time;time.sleep(60)",
            ],
            cwd=os.fspath(tmp_path),
            env={"PATH": "/usr/bin:/bin"},
            timeout_seconds=5,
            stdout_max_bytes=4096,
            stderr_max_bytes=4096,
            reap_seconds=2,
        )
    assert error.value.code == "FAIL_CAMPAIGN_PROCESS"
    assert error.value.receipt["pid_match_or_null"] is True
    assert error.value.receipt["cleanup_certain"] is True


@pytest.mark.parametrize(
    ("raw", "expected_state"),
    (
        (b"", "MISSING"),
        (b"123", "TORN"),
        (b"123\nextra", "MALFORMED"),
        (b"2147483647\n", "VALID"),
    ),
)
def test_invalid_or_mismatched_spawn_receipt_fails_uncertain_after_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw: bytes,
    expected_state: str,
) -> None:
    monkeypatch.setattr(
        TOOL, "_require_single_threaded_spawn_controller_v2", lambda: None
    )
    monkeypatch.setattr(
        TOOL,
        "_read_spawn_registration_receipt_v2",
        lambda _fd: (raw, True),
    )
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._bounded_process(
            [
                "/usr/bin/python3.10", "-I", "-S", "-B", "-c",
                "import time;time.sleep(60)",
            ],
            cwd=os.fspath(tmp_path),
            env={"PATH": "/usr/bin:/bin"},
            timeout_seconds=5,
            stdout_max_bytes=4096,
            stderr_max_bytes=4096,
            reap_seconds=2,
        )
    assert error.value.code == "FAIL_CAMPAIGN_SPAWN_UNCERTAIN"
    assert error.value.receipt["registration_state"] == expected_state
    assert error.value.receipt["later_ledger_write_permitted"] is False


def test_spawn_cleanup_double_fault_is_uncertain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        TOOL, "_require_single_threaded_spawn_controller_v2", lambda: None
    )
    spawned: list[object] = []
    real_init = TOOL.subprocess.Popen.__init__

    def remember_then_interrupt(process: object, *args: object, **kwargs: object) -> None:
        real_init(process, *args, **kwargs)
        spawned.append(process)
        raise KeyboardInterrupt("injected post-spawn fault")

    monkeypatch.setattr(TOOL.subprocess.Popen, "__init__", remember_then_interrupt)
    monkeypatch.setattr(
        TOOL,
        "_cleanup_registered_spawn_v2",
        lambda *_args, **_kwargs: {
            "cleanup_target_pid_or_null": getattr(spawned[0], "pid", None),
            "cleanup_kill_attempted": True,
            "cleanup_direct_child_reaped": False,
            "cleanup_process_group_absent": False,
            "cleanup_identity_consistent": True,
            "cleanup_errors": ["injected cleanup double fault"],
            "cleanup_certain": False,
        },
    )
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._bounded_process(
                [
                    "/usr/bin/python3.10", "-I", "-S", "-B", "-c",
                    "import time;time.sleep(60)",
                ],
                cwd=os.fspath(tmp_path),
                env={"PATH": "/usr/bin:/bin"},
                timeout_seconds=5,
                stdout_max_bytes=4096,
                stderr_max_bytes=4096,
                reap_seconds=2,
            )
        assert error.value.code == "FAIL_CAMPAIGN_SPAWN_UNCERTAIN"
        assert error.value.receipt["cleanup_certain"] is False
        assert error.value.receipt["later_ledger_write_permitted"] is False
    finally:
        if spawned:
            process = spawned[0]
            try:
                TOOL.os.killpg(process.pid, TOOL.signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2)


class _FakeLedger:
    def __init__(self) -> None:
        self.authorization = {"authorization_digest": _tagged("authorization")}
        self.published_names: list[str] = []
        self.published_records: list[dict[str, object]] = []
        self.closed = False
        self.rows: list[tuple[str, str, str, dict[str, object]]] = []
        self.completion_commit_verified = False
        self.completion_result_candidate_or_null: dict[str, object] | None = None
        self.authoritative_completion_result_or_null: dict[str, object] | None = None

    def prepare(
        self,
        basename: str,
        record_type: str,
        status_text: str,
        body: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            "sequence": len(self.rows) + 1,
            "record_type": record_type,
            "status": status_text,
            "body": deepcopy(dict(body)),
            "record_digest": _tagged(f"ledger-{len(self.rows) + 1}"),
        }

    def publish_prepared(
        self,
        basename: str,
        prepared_record: Mapping[str, object] | bytes,
    ) -> dict[str, object]:
        if type(prepared_record) is bytes:
            decoded = TOOL._strict_json(
                prepared_record,
                "fake prepared ledger record",
            )
            assert type(decoded) is dict
            prepared_record = decoded
        record_type = prepared_record["record_type"]
        status_text = prepared_record["status"]
        body = prepared_record["body"]
        assert type(record_type) is str
        assert type(status_text) is str
        assert type(body) is dict
        self.published_names.append(basename)
        self.published_records.append(deepcopy(dict(prepared_record)))
        self.rows.append(
            (basename, record_type, status_text, deepcopy(body))
        )
        if basename == "campaign.completion.json":
            assert type(self.completion_result_candidate_or_null) is dict
            self.authoritative_completion_result_or_null = deepcopy(
                self.completion_result_candidate_or_null
            )
            self.completion_commit_verified = True
        return deepcopy(dict(prepared_record))

    def prepare_bytes(
        self,
        basename: str,
        record_type: str,
        status_text: str,
        body: Mapping[str, object],
    ) -> bytes:
        return TOOL._canonical_json_bytes(
            self.prepare(basename, record_type, status_text, body)
        )

    def cache_completion_result_candidate(
        self,
        result: Mapping[str, object],
        _completion_ledger_record: Mapping[str, object],
    ) -> dict[str, object]:
        self.completion_result_candidate_or_null = deepcopy(dict(result))
        return deepcopy(dict(result))

    def publish(
        self,
        basename: str,
        record_type: str,
        status_text: str,
        body: Mapping[str, object],
    ) -> dict[str, object]:
        return self.publish_prepared(
            basename,
            self.prepare(basename, record_type, status_text, body),
        )

    def close(self) -> None:
        self.closed = True


def _completed_real_ledger(
    tmp_path: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    object,
    dict[str, object],
]:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        tmp_path / "completed-ledger"
    )
    capsule = {"capsule_digest": _tagged("capsule")}
    authorization = TOOL._authorization_template(contract, capsule)
    ledger = TOOL._LedgerV2(contract, capsule, authorization)
    ledger.publish(
        "campaign.intent.json",
        "CAMPAIGN_INTENT",
        "AUTHORIZED_CAMPAIGN_INTENT_STAGED",
        {
            "capsule": deepcopy(capsule),
            "authorization_digest": authorization["authorization_digest"],
        },
    )
    ledger.publish(
        "campaign.authorization-consumed.json",
        "CAMPAIGN_AUTHORIZATION_CONSUMED",
        "CAMPAIGN_AUTHORIZATION_CONSUMED",
        {
            "authorization": deepcopy(authorization),
            "consumed_once": True,
        },
    )
    result = TOOL._publish_terminal(
        ledger,
        contract,
        capsule,
        status_text=contract["terminal_failure_policy"]["crash_disposition"],
        invocation_count=0,
        last_slot_id_or_null=None,
        last_slot_outcome_digest_or_null=None,
        error_code_or_null="KeyboardInterrupt",
        error_detail_or_null="injected legal pre-slot terminal",
    )
    return contract, capsule, authorization, ledger, result


def _completed_real_success_ledger(
    tmp_path: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    object,
    dict[str, object],
]:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        tmp_path / "success-ledger"
    )
    slots = contract["slot_registry"]
    assert type(slots) is list
    slot = slots[0]
    assert type(slot) is dict
    trace_root = tmp_path / "slot-01-trace"
    trace_root.mkdir(mode=0o700)
    trace_root.chmod(0o700)
    slot["trace_root"] = os.fspath(trace_root)
    capsule = _semantic_capsule(contract)
    authorization = TOOL._authorization_template(contract, capsule)
    wire = _success_wire(contract, slot)
    stdout = CORE.canonical_campaign_json_bytes_v2(wire)
    stdout_facts = CORE.classify_supervisor_stdout_v2(stdout)
    process = _process(stdout, 0)
    trajectory = _trajectory(slot, outcome="OK")
    supervisor_artifact, external_artifact = _valid_artifacts(wire)
    source_before = _source_replay("real-success-source-before")
    source_after = _source_replay("real-success-source-after")
    future = _slot_absence_evidence(
        capsule,
        slots[1:],
        "SLOT_01 future slots before outcome",
    )
    cleanup = TOOL._cleanup_replay_evidence(
        process,
        trajectory,
        True,
        [],
        None,
    )
    binding = TOOL._binding_record(
        contract,
        capsule,
        slot,
        process,
        stdout_facts,
        trajectory,
        supervisor_artifact,
        external_artifact,
        True,
        [],
        None,
        source_before,
        source_after,
        future,
        cleanup,
    )
    observation = TOOL._build_slot_observation(
        contract,
        slot,
        process,
        stdout_facts,
        trajectory,
        supervisor_artifact,
        external_artifact,
        source_before_certain=True,
        source_after_certain=True,
        docker_absent=True,
        external_binding_certain=True,
    )
    decision = CORE.evaluate_slot_observation_v2(observation)
    assert decision == CORE.SUCCESS_V2
    outcome = TOOL._slot_outcome_record(
        contract,
        capsule,
        slot,
        binding,
        observation,
        decision,
    )
    ledger = TOOL._LedgerV2(contract, capsule, authorization)
    ledger.publish(
        "campaign.intent.json",
        "CAMPAIGN_INTENT",
        "AUTHORIZED_CAMPAIGN_INTENT_STAGED",
        {
            "capsule": deepcopy(capsule),
            "authorization_digest": authorization["authorization_digest"],
        },
    )
    ledger.publish(
        "campaign.authorization-consumed.json",
        "CAMPAIGN_AUTHORIZATION_CONSUMED",
        "CAMPAIGN_AUTHORIZATION_CONSUMED",
        {
            "authorization": deepcopy(authorization),
            "consumed_once": True,
        },
    )
    ledger.publish(
        "slot-01.intent.json",
        "SLOT_INTENT",
        "SLOT_01_INTENT_STAGED",
        {
            "slot": deepcopy(slot),
            "argv": TOOL._slot_argv(contract, slot),
            "environment": TOOL._runtime_environment(
                contract,
                git_worktree=str(slot["clone_root"]),
            ),
            "previous_slot_id_or_null": None,
            "slot_freshness_before_intent": _slot_absence_evidence(
                capsule,
                [slot],
                "SLOT_01 before intent",
            ),
        },
    )
    ledger.publish(
        "slot-01.consumed.json",
        "SLOT_CONSUMED",
        "SLOT_01_CONSUMED_OUTCOME_UNKNOWN",
        {
            "slot_id": "SLOT_01",
            "slot_ordinal": 1,
            "argv": TOOL._slot_argv(contract, slot),
            "source_replay_before_root": source_before["root"],
            "docker_absence_before_spawn": True,
        },
    )
    ledger.publish(
        "slot-01.trajectory-binding.json",
        "EXTERNAL_TRAJECTORY_BINDING",
        "COMPLETE_OK",
        binding,
    )
    ledger.publish(
        "slot-01.outcome.json",
        "SLOT_OUTCOME",
        decision,
        outcome,
    )
    result = TOOL._publish_terminal(
        ledger,
        contract,
        capsule,
        status_text=contract["success_policy"]["terminal_status"],
        invocation_count=999,
        last_slot_id_or_null=None,
        last_slot_outcome_digest_or_null=None,
    )
    return contract, capsule, authorization, ledger, result


def _full_real_gap_ledger_prefix(
    tmp_path: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    object,
]:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        tmp_path / "full-gap-prefix-ledger"
    )
    slots = contract["slot_registry"]
    assert type(slots) is list
    for ordinal, slot in enumerate(slots, start=1):
        assert type(slot) is dict
        trace_root = tmp_path / f"slot-{ordinal:02d}-trace"
        trace_root.mkdir(mode=0o700)
        trace_root.chmod(0o700)
        slot["trace_root"] = os.fspath(trace_root)
    capsule = _semantic_capsule(contract)
    authorization = TOOL._authorization_template(contract, capsule)
    ledger = TOOL._LedgerV2(contract, capsule, authorization)
    ledger.publish(
        "campaign.intent.json",
        "CAMPAIGN_INTENT",
        "AUTHORIZED_CAMPAIGN_INTENT_STAGED",
        {
            "capsule": deepcopy(capsule),
            "authorization_digest": authorization["authorization_digest"],
        },
    )
    ledger.publish(
        "campaign.authorization-consumed.json",
        "CAMPAIGN_AUTHORIZATION_CONSUMED",
        "CAMPAIGN_AUTHORIZATION_CONSUMED",
        {
            "authorization": deepcopy(authorization),
            "consumed_once": True,
        },
    )
    for ordinal, slot in enumerate(slots, start=1):
        assert type(slot) is dict
        detail, failure_rows = _held_final_gap_rows()
        stdout = CORE.canonical_campaign_json_bytes_v2(_error_wire(detail))
        stdout_facts = CORE.classify_supervisor_stdout_v2(stdout)
        process = _process(stdout, 1)
        trajectory = _trajectory(
            slot,
            outcome="FAIL",
            failure_rows=failure_rows,
            terminal_error_detail=detail,
        )
        supervisor_artifact, external_artifact = _absent_artifacts()
        source_before = _source_replay(f"gap-{ordinal}-source-before")
        source_after = _source_replay(f"gap-{ordinal}-source-after")
        future = _slot_absence_evidence(
            capsule,
            slots[ordinal:],
            f"{slot['slot_id']} future slots before outcome",
        )
        cleanup = TOOL._cleanup_replay_evidence(
            process,
            trajectory,
            True,
            [],
            None,
        )
        binding = TOOL._binding_record(
            contract,
            capsule,
            slot,
            process,
            stdout_facts,
            trajectory,
            supervisor_artifact,
            external_artifact,
            True,
            [],
            None,
            source_before,
            source_after,
            future,
            cleanup,
        )
        observation = TOOL._build_slot_observation(
            contract,
            slot,
            process,
            stdout_facts,
            trajectory,
            supervisor_artifact,
            external_artifact,
            source_before_certain=True,
            source_after_certain=True,
            docker_absent=True,
            external_binding_certain=True,
        )
        decision = CORE.evaluate_slot_observation_v2(observation)
        assert decision == (
            CORE.RETRY_EXACT_GAP_V2
            if ordinal < 3
            else CORE.EXHAUSTED_RETRYABLE_GAP_V2
        )
        outcome = TOOL._slot_outcome_record(
            contract,
            capsule,
            slot,
            binding,
            observation,
            decision,
        )
        stem = f"slot-{ordinal:02d}"
        ledger.publish(
            f"{stem}.intent.json",
            "SLOT_INTENT",
            f"{slot['slot_id']}_INTENT_STAGED",
            {
                "slot": deepcopy(slot),
                "argv": TOOL._slot_argv(contract, slot),
                "environment": TOOL._runtime_environment(
                    contract,
                    git_worktree=str(slot["clone_root"]),
                ),
                "previous_slot_id_or_null": slot["previous_slot_id"],
                "slot_freshness_before_intent": _slot_absence_evidence(
                    capsule,
                    [slot],
                    f"{slot['slot_id']} before intent",
                ),
            },
        )
        ledger.publish(
            f"{stem}.consumed.json",
            "SLOT_CONSUMED",
            f"{slot['slot_id']}_CONSUMED_OUTCOME_UNKNOWN",
            {
                "slot_id": slot["slot_id"],
                "slot_ordinal": ordinal,
                "argv": TOOL._slot_argv(contract, slot),
                "source_replay_before_root": source_before["root"],
                "docker_absence_before_spawn": True,
            },
        )
        ledger.publish(
            f"{stem}.trajectory-binding.json",
            "EXTERNAL_TRAJECTORY_BINDING",
            "COMPLETE_FAIL",
            binding,
        )
        ledger.publish(
            f"{stem}.outcome.json",
            "SLOT_OUTCOME",
            decision,
            outcome,
        )
    return contract, capsule, authorization, ledger


def _append_test_ledger_record(
    contract: Mapping[str, object],
    capsule: Mapping[str, object],
    authorization: Mapping[str, object],
    records: list[tuple[str, dict[str, object]]],
    basename: str,
    record_type: str,
    status_text: str,
    body: Mapping[str, object],
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": (
            "hegel-phase3a-q05b-shadow-retry-campaign-ledger-record/2"
        ),
        "campaign_id": contract["contract_id"],
        "contract_digest": contract["contract_digest"],
        "campaign_source_binding_digest": contract[
            "campaign_source_binding"
        ]["source_binding_digest"],
        "capsule_digest": capsule["capsule_digest"],
        "authorization_digest": authorization["authorization_digest"],
        "sequence": len(records) + 1,
        "record_type": record_type,
        "status": status_text,
        "previous_record_digest_or_null": (
            records[-1][1]["record_digest"] if records else None
        ),
        "body": deepcopy(dict(body)),
        "record_digest": None,
    }
    unsigned = deepcopy(value)
    unsigned.pop("record_digest")
    value["record_digest"] = TOOL._domain_digest(
        contract["hash_domains"]["ledger_record"],
        unsigned,
    )
    records.append((basename, value))
    return value


def _rewrite_terminal_completion_chain(
    contract: Mapping[str, object],
    capsule: Mapping[str, object],
    authorization: Mapping[str, object],
    rows: list[tuple[str, dict[str, object]]],
    *,
    error_code: str,
    error_detail: str,
    error_receipt: Mapping[str, object],
) -> None:
    terminal_record = rows[-2][1]
    terminal = terminal_record["body"]
    assert type(terminal) is dict
    terminal["status"] = TOOL._error_code_terminal_disposition(
        contract, error_code
    )
    terminal["error_code_or_null"] = error_code
    terminal["error_detail_or_null"] = error_detail
    terminal["error_receipt_or_null"] = deepcopy(dict(error_receipt))
    terminal_unsigned = deepcopy(terminal)
    terminal_unsigned.pop("record_digest")
    terminal["record_digest"] = TOOL._domain_digest(
        contract["hash_domains"]["campaign_terminal"],
        terminal_unsigned,
    )
    terminal_record["status"] = terminal["status"]
    terminal_record_unsigned = deepcopy(terminal_record)
    terminal_record_unsigned.pop("record_digest")
    terminal_record["record_digest"] = TOOL._domain_digest(
        contract["hash_domains"]["ledger_record"],
        terminal_record_unsigned,
    )
    completion_record = rows[-1][1]
    completion_record["previous_record_digest_or_null"] = terminal_record[
        "record_digest"
    ]
    completion = TOOL._campaign_completion_body(
        contract,
        capsule,
        authorization_digest=str(authorization["authorization_digest"]),
        terminal=terminal,
        terminal_ledger_record=terminal_record,
        preceding_ledger_names=[name for name, _record in rows[:-1]],
    )
    completion_record["body"] = completion
    completion_record_unsigned = deepcopy(completion_record)
    completion_record_unsigned.pop("record_digest")
    completion_record["record_digest"] = TOOL._domain_digest(
        contract["hash_domains"]["ledger_record"],
        completion_record_unsigned,
    )


def test_completion_record_is_authoritative_and_offline_verifiable(
    tmp_path: Path,
) -> None:
    contract, capsule, authorization, ledger_object, result = (
        _completed_real_ledger(tmp_path)
    )
    ledger = ledger_object
    try:
        assert ledger.published_names[-2:] == [
            "campaign.terminal.json",
            "campaign.completion.json",
        ]
        assert ledger.completion_commit_verified is True
        assert result["completion_commit_verified"] is True
        assert result["authoritative_completion_record_name"] == (
            "campaign.completion.json"
        )
        rows = list(zip(ledger.published_names, ledger.published_records))
        verification = TOOL._verify_campaign_completion_chain_v2(
            contract,
            capsule,
            authorization,
            rows,
        )
        assert verification["status"] == (
            "AUTHORITATIVE_CAMPAIGN_COMPLETION_VERIFIED"
        )
        assert verification["ledger_completion_record_digest"] == result[
            "ledger_completion_record_digest"
        ]
        assert verification["campaign_completion_digest"] == result[
            "campaign_completion_digest"
        ]
        root = Path(str(contract["fixed_external_roots"]["ledger_root"]))
        assert (root / "campaign.terminal.json").is_file()
        assert (root / "campaign.completion.json").is_file()
        disk_verification = TOOL._verify_campaign_completion_ledger_at_v2(
            contract,
            capsule,
            authorization,
            root,
        )
        assert disk_verification == verification
    finally:
        ledger.close()


def test_offline_verifier_rejects_rehashed_unknown_receipt_schema(
    tmp_path: Path,
) -> None:
    contract, capsule, authorization, ledger_object, _result = (
        _completed_real_ledger(tmp_path)
    )
    ledger = ledger_object
    try:
        rows = deepcopy(list(zip(ledger.published_names, ledger.published_records)))
        _rewrite_terminal_completion_chain(
            contract,
            capsule,
            authorization,
            rows,
            error_code="KeyboardInterrupt",
            error_detail="injected legal pre-slot terminal",
            error_receipt={
                "schema_version": "forged-receipt/1",
                "argv": ["/not/the/command"],
                "label": "forged label",
                "phase": "FAKE",
            },
        )
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._verify_campaign_completion_chain_v2(
                contract,
                capsule,
                authorization,
                rows,
            )
        assert error.value.code == "FAIL_CAMPAIGN_LEDGER"
        assert "terminal error disposition" in error.value.detail
    finally:
        ledger.close()


@pytest.mark.parametrize(
    "mutation",
    (
        "unknown-field",
        "argv",
        "label",
        "phase",
        "terminal-detail-crosslink",
        "recursive-prior",
    ),
)
def test_offline_verifier_rejects_rehashed_receipt_semantic_forgery(
    tmp_path: Path,
    mutation: str,
) -> None:
    contract, capsule, authorization, ledger_object, _result = (
        _completed_real_ledger(tmp_path)
    )
    ledger = ledger_object
    try:
        receipt = {
            "schema_version": (
                "hegel-phase3a-q05b-shadow-retry-bounded-command-exception-"
                "diagnostic/2"
            ),
            "label": "target tree replay",
            "argv": [
                contract["execution_policy"]["git_executable"],
                "-C",
                contract["execution_policy"]["clone_source_repository"],
                "rev-parse",
                f"{contract['target_commit_a_binding']['target_supervisor_commit']}^{{tree}}",
            ],
            "phase": "BOUNDED_PROCESS_RAISED_BEFORE_RESULT",
            "error_code": "OSError",
            "error_detail": "injected selector failure",
            "return_code": None,
            "timed_out": None,
            "stdout_overflow": None,
            "stderr_overflow": None,
            "child_reaped": None,
            "process_group_absent": None,
            "wrapper_resources_closed": None,
            "stdout": {"available": False},
            "stderr": {"available": False},
            "prior_receipt_or_null": None,
        }
        detail = (
            "target tree replay bounded process raised before a "
            "result: OSError:injected selector failure"
        )
        if mutation == "unknown-field":
            receipt["forged"] = True
        elif mutation == "argv":
            receipt["argv"] = ["/usr/bin/docker", "ps", "--forged"]
        elif mutation == "label":
            receipt["label"] = "forged label"
        elif mutation == "phase":
            receipt["phase"] = "FORGED_PHASE"
        elif mutation == "terminal-detail-crosslink":
            detail += " forged"
        else:
            nested = deepcopy(receipt)
            nested["prior_receipt_or_null"] = None
            receipt["prior_receipt_or_null"] = nested
        rows = deepcopy(list(zip(ledger.published_names, ledger.published_records)))
        _rewrite_terminal_completion_chain(
            contract,
            capsule,
            authorization,
            rows,
            error_code="FAIL_CAMPAIGN_PROCESS",
            error_detail=detail,
            error_receipt=receipt,
        )
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._verify_campaign_completion_chain_v2(
                contract,
                capsule,
                authorization,
                rows,
            )
        assert error.value.code == "FAIL_CAMPAIGN_LEDGER"
        assert "terminal error disposition" in error.value.detail
    finally:
        ledger.close()


def test_offline_verifier_accepts_each_registered_authoritative_receipt(
    tmp_path: Path,
) -> None:
    contract, capsule, authorization, ledger_object, _result = (
        _completed_real_ledger(tmp_path)
    )
    ledger = ledger_object
    try:
        base_rows = deepcopy(
            list(zip(ledger.published_names, ledger.published_records))
        )
        docker_argv = TOOL._docker_inventory_argv_v2(contract)
        empty_stream = {
            "byte_count": 0,
            "raw_sha256": sha256(b"").hexdigest(),
            "excerpt_hex": "",
            "excerpt_byte_count": 0,
            "excerpt_truncated": False,
        }
        diagnostic = {
            "schema_version": (
                "hegel-phase3a-q05b-shadow-retry-bounded-command-diagnostic/2"
            ),
            "label": "Docker read-only inventory",
            "argv": docker_argv,
            "return_code": 7,
            "timed_out": False,
            "stdout_overflow": False,
            "stderr_overflow": False,
            "child_reaped": True,
            "process_group_absent": True,
            "wrapper_resources_closed": True,
            "stdout": deepcopy(empty_stream),
            "stderr": deepcopy(empty_stream),
            "classification": {
                "accepted": False,
                "reason": "injected Docker failure",
            },
        }
        diagnostic_detail = (
            "Docker read-only inventory failed closed: return_code=7; "
            "timed_out=False; stdout_overflow=False; stderr_overflow=False; "
            "stderr_excerpt_hex="
        )
        git_argv = [
            contract["execution_policy"]["git_executable"],
            "-C",
            contract["execution_policy"]["clone_source_repository"],
            "rev-parse",
            f"{contract['target_commit_a_binding']['target_supervisor_commit']}^{{tree}}",
        ]
        git_diagnostic = deepcopy(diagnostic)
        git_diagnostic["label"] = "target tree replay"
        git_diagnostic["argv"] = git_argv
        git_diagnostic_detail = diagnostic_detail.replace(
            "Docker read-only inventory", "target tree replay", 1
        )
        exception = {
            "schema_version": (
                "hegel-phase3a-q05b-shadow-retry-bounded-command-exception-"
                "diagnostic/2"
            ),
            "label": "target tree replay",
            "argv": git_argv,
            "phase": "BOUNDED_PROCESS_RAISED_BEFORE_RESULT",
            "error_code": "OSError",
            "error_detail": "injected selector failure",
            "return_code": None,
            "timed_out": None,
            "stdout_overflow": None,
            "stderr_overflow": None,
            "child_reaped": None,
            "process_group_absent": None,
            "wrapper_resources_closed": None,
            "stdout": {"available": False},
            "stderr": {"available": False},
            "prior_receipt_or_null": None,
        }
        exception_detail = (
            "target tree replay bounded process raised before a "
            "result: OSError:injected selector failure"
        )
        spawn = {
            "schema_version": TOOL._SPAWN_REGISTRATION_SCHEMA_V2,
            "phase": "SPAWN_REGISTRATION",
            "argv": TOOL._slot_argv(contract, contract["slot_registry"][0]),
            "registration_state": "VALID",
            "registration_raw_hex": b"123\n".hex(),
            "registered_pid_or_null": 123,
            "popen_pid_or_null": 123,
            "popen_child_created": True,
            "pid_match_or_null": True,
            "constructor_error_or_null": {
                "error_code": "OSError",
                "error_detail": "injected post-registration constructor error",
                "error_terminal": False,
                "error_receipt_or_null": None,
            },
            "cleanup_target_pid_or_null": 123,
            "cleanup_kill_attempted": True,
            "cleanup_direct_child_reaped": True,
            "cleanup_process_group_absent": True,
            "cleanup_identity_consistent": True,
            "cleanup_errors": [],
            "cleanup_certain": True,
            "later_ledger_write_permitted": True,
        }
        spawn_detail = (
            "registered Popen did not cross its commit boundary: "
            "registration=VALID; pid_match=True; restore_errors=[]; "
            "error=OSError:injected post-registration constructor error"
        )
        spawn_exception = {
            "schema_version": (
                "hegel-phase3a-q05b-shadow-retry-bounded-command-exception-"
                "diagnostic/2"
            ),
            "label": "SLOT_01 bound supervisor",
            "argv": TOOL._slot_argv(contract, contract["slot_registry"][0]),
            "phase": "BOUNDED_PROCESS_RAISED_BEFORE_RESULT",
            "error_code": "FAIL_CAMPAIGN_PROCESS",
            "error_detail": spawn_detail,
            "return_code": None,
            "timed_out": None,
            "stdout_overflow": None,
            "stderr_overflow": None,
            "child_reaped": None,
            "process_group_absent": None,
            "wrapper_resources_closed": None,
            "stdout": {"available": False},
            "stderr": {"available": False},
            "prior_receipt_or_null": spawn,
        }
        spawn_exception_detail = (
            "SLOT_01 bound supervisor bounded process raised before a result: "
            "FAIL_CAMPAIGN_PROCESS:" + spawn_detail
        )
        docker_error = TOOL._canonical_json_bytes(diagnostic).decode(
            "ascii"
        ).rstrip("\n")
        prespawn = {
            "schema_version": TOOL._PRESPAWN_DOCKER_RECEIPT_SCHEMA_V2,
            "docker_rows": [],
            "docker_error_or_null": docker_error,
            "bounded_command_diagnostic_or_null": deepcopy(diagnostic),
        }
        prespawn_detail = (
            "SLOT_01 pre-spawn Docker absence is uncertain: " + docker_error
        )
        publication = {
            "schema_version": TOOL._PUBLICATION_RECEIPT_SCHEMA_V2,
            "publication_outcome": "KNOWN_NO_PUBLICATION",
            "publication_label": "slot-supervisor-stdout",
            "publication_basename": "supervisor.stdout",
        }
        publication_detail = (
            "publication_label=slot-supervisor-stdout; "
            "basename=supervisor.stdout; exclusive publication failed"
        )
        omission = {
            "schema_version": (
                "hegel-phase3a-q05b-shadow-retry-bounded-receipt-omission/2"
            ),
            "status": "RECEIPT_OMITTED_OVER_BOUND",
        }
        cases = (
            (
                "FAIL_CAMPAIGN_PROCESS",
                git_diagnostic_detail,
                git_diagnostic,
            ),
            ("FAIL_CAMPAIGN_PROCESS", exception_detail, exception),
            (
                "FAIL_CAMPAIGN_PROCESS",
                spawn_exception_detail,
                spawn_exception,
            ),
            ("FAIL_CAMPAIGN_EVIDENCE", publication_detail, publication),
            ("KeyboardInterrupt", "bounded omission", omission),
        )
        for code, detail, receipt in cases:
            rows = deepcopy(base_rows)
            _rewrite_terminal_completion_chain(
                contract,
                capsule,
                authorization,
                rows,
                error_code=code,
                error_detail=detail,
                error_receipt=receipt,
            )
            verification = TOOL._verify_campaign_completion_chain_v2(
                contract,
                capsule,
                authorization,
                rows,
            )
            assert verification["status"] == (
                "AUTHORITATIVE_CAMPAIGN_COMPLETION_VERIFIED"
            )

        prespawn_root = tmp_path / "prespawn"
        prespawn_root.mkdir()
        full_contract, full_capsule, full_authorization, full_ledger = (
            _full_real_gap_ledger_prefix(prespawn_root)
        )
        try:
            prespawn_rows = deepcopy(
                list(zip(full_ledger.published_names, full_ledger.published_records))[
                    :3
                ]
            )
            progress = TOOL._ledger_progress_snapshot_v2(
                [record for _name, record in prespawn_rows]
            )
            status = TOOL._error_code_terminal_disposition(
                full_contract, "FAIL_CAMPAIGN_DOCKER"
            )
            terminal = TOOL._campaign_terminal_body(
                full_contract,
                full_capsule,
                status_text=status,
                invocation_count=int(progress["supervisor_invocation_count"]),
                last_slot_id_or_null=progress["last_slot_id_or_null"],
                last_slot_outcome_digest_or_null=progress[
                    "last_slot_outcome_digest_or_null"
                ],
                error_code_or_null="FAIL_CAMPAIGN_DOCKER",
                error_detail_or_null=prespawn_detail,
                error_receipt_or_null=prespawn,
            )
            terminal_record = _append_test_ledger_record(
                full_contract,
                full_capsule,
                full_authorization,
                prespawn_rows,
                "campaign.terminal.json",
                "CAMPAIGN_TERMINAL",
                status,
                terminal,
            )
            completion = TOOL._campaign_completion_body(
                full_contract,
                full_capsule,
                authorization_digest=str(
                    full_authorization["authorization_digest"]
                ),
                terminal=terminal,
                terminal_ledger_record=terminal_record,
                preceding_ledger_names=[
                    name for name, _record in prespawn_rows
                ],
            )
            _append_test_ledger_record(
                full_contract,
                full_capsule,
                full_authorization,
                prespawn_rows,
                "campaign.completion.json",
                "CAMPAIGN_COMPLETION",
                "CAMPAIGN_COMPLETION_COMMITTED",
                completion,
            )
            assert TOOL._verify_campaign_completion_chain_v2(
                full_contract,
                full_capsule,
                full_authorization,
                prespawn_rows,
            )["status"] == "AUTHORITATIVE_CAMPAIGN_COMPLETION_VERIFIED"
            wrong_slot_rows = deepcopy(prespawn_rows)
            _rewrite_terminal_completion_chain(
                full_contract,
                full_capsule,
                full_authorization,
                wrong_slot_rows,
                error_code="FAIL_CAMPAIGN_DOCKER",
                error_detail=prespawn_detail.replace(
                    "SLOT_01", "SLOT_02", 1
                ),
                error_receipt=prespawn,
            )
            with pytest.raises(TOOL.ShadowRetryCampaignErrorV2):
                TOOL._verify_campaign_completion_chain_v2(
                    full_contract,
                    full_capsule,
                    full_authorization,
                    wrong_slot_rows,
                )

            outer_exception = TOOL._bounded_process_exception_diagnostic(
                "Docker read-only inventory",
                TOOL._docker_inventory_argv_v2(full_contract),
                RuntimeError("injected outer inventory failure"),
            )
            outer_error = TOOL._canonical_json_bytes(
                outer_exception
            ).decode("ascii").rstrip("\n")
            outer_receipt = {
                "schema_version": TOOL._PRESPAWN_DOCKER_RECEIPT_SCHEMA_V2,
                "docker_rows": [],
                "docker_error_or_null": outer_error,
                "bounded_command_diagnostic_or_null": outer_exception,
            }
            outer_rows = deepcopy(prespawn_rows)
            _rewrite_terminal_completion_chain(
                full_contract,
                full_capsule,
                full_authorization,
                outer_rows,
                error_code="FAIL_CAMPAIGN_DOCKER",
                error_detail=(
                    "SLOT_01 pre-spawn Docker absence is uncertain: "
                    + outer_error
                ),
                error_receipt=outer_receipt,
            )
            assert TOOL._verify_campaign_completion_chain_v2(
                full_contract,
                full_capsule,
                full_authorization,
                outer_rows,
            )["status"] == "AUTHORITATIVE_CAMPAIGN_COMPLETION_VERIFIED"

            oversized_exception = TOOL._bounded_process_exception_diagnostic(
                "Docker read-only inventory",
                TOOL._docker_inventory_argv_v2(full_contract),
                TOOL.ShadowRetryCampaignErrorV2(
                    "FAIL_CAMPAIGN_PROCESS",
                    "injected oversized diagnostic",
                    receipt={"opaque_diagnostic": "x" * (20 * 1024)},
                ),
            )
            oversized_error = TOOL._canonical_json_bytes(
                oversized_exception
            ).decode("ascii").rstrip("\n")
            assert len(oversized_error.encode("ascii") + b"\n") > 16 * 1024
            omission_receipt = {
                "schema_version": TOOL._PRESPAWN_DOCKER_RECEIPT_SCHEMA_V2,
                "docker_rows": [],
                "docker_error_or_null": oversized_error,
                "bounded_command_diagnostic_or_null": deepcopy(omission),
            }
            oversized_rows = deepcopy(prespawn_rows)
            _rewrite_terminal_completion_chain(
                full_contract,
                full_capsule,
                full_authorization,
                oversized_rows,
                error_code="FAIL_CAMPAIGN_DOCKER",
                error_detail=(
                    "SLOT_01 pre-spawn Docker absence is uncertain: "
                    + oversized_error
                )[:4096],
                error_receipt=omission_receipt,
            )
            assert TOOL._verify_campaign_completion_chain_v2(
                full_contract,
                full_capsule,
                full_authorization,
                oversized_rows,
            )["status"] == "AUTHORITATIVE_CAMPAIGN_COMPLETION_VERIFIED"

            forged_small_omission = deepcopy(prespawn)
            forged_small_omission[
                "bounded_command_diagnostic_or_null"
            ] = deepcopy(omission)
            forged_small_rows = deepcopy(prespawn_rows)
            _rewrite_terminal_completion_chain(
                full_contract,
                full_capsule,
                full_authorization,
                forged_small_rows,
                error_code="FAIL_CAMPAIGN_DOCKER",
                error_detail=prespawn_detail,
                error_receipt=forged_small_omission,
            )
            with pytest.raises(TOOL.ShadowRetryCampaignErrorV2):
                TOOL._verify_campaign_completion_chain_v2(
                    full_contract,
                    full_capsule,
                    full_authorization,
                    forged_small_rows,
                )
        finally:
            full_ledger.close()
    finally:
        ledger.close()


def test_safe_docker_inventory_normalizes_outer_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()

    def raise_outer(_contract: Mapping[str, object]) -> object:
        raise RuntimeError("injected outer Docker inventory failure")

    monkeypatch.setattr(TOOL, "_docker_inventory", raise_outer)
    absent, rows, error = TOOL._safe_docker_inventory(contract)
    assert absent is False
    assert rows == []
    assert type(error) is str
    receipt = TOOL._strict_json(
        error.encode("ascii") + b"\n",
        "outer Docker inventory exception KAT",
        16 * 1024,
    )
    assert receipt["schema_version"] == (
        TOOL._BOUNDED_COMMAND_EXCEPTION_DIAGNOSTIC_SCHEMA_V2
    )
    assert receipt["label"] == "Docker read-only inventory"
    assert receipt["argv"] == TOOL._docker_inventory_argv_v2(contract)
    assert receipt["phase"] == "BOUNDED_PROCESS_RAISED_BEFORE_RESULT"
    assert receipt["error_code"] == "RuntimeError"
    assert receipt["error_detail"] == (
        "injected outer Docker inventory failure"
    )


def test_offline_verifier_rejects_rehashed_spawn_uncertain_completion(
    tmp_path: Path,
) -> None:
    contract, capsule, authorization, ledger_object, _result = (
        _completed_real_ledger(tmp_path)
    )
    ledger = ledger_object
    try:
        inner_detail = "injected missing spawn registration"
        receipt = {
            "schema_version": TOOL._BOUNDED_COMMAND_EXCEPTION_DIAGNOSTIC_SCHEMA_V2,
            "label": "SLOT_01 bound supervisor",
            "argv": TOOL._slot_argv(contract, contract["slot_registry"][0]),
            "phase": "BOUNDED_PROCESS_RAISED_BEFORE_RESULT",
            "error_code": "FAIL_CAMPAIGN_SPAWN_UNCERTAIN",
            "error_detail": inner_detail,
            "return_code": None,
            "timed_out": None,
            "stdout_overflow": None,
            "stderr_overflow": None,
            "child_reaped": None,
            "process_group_absent": None,
            "wrapper_resources_closed": None,
            "stdout": {"available": False},
            "stderr": {"available": False},
            "prior_receipt_or_null": {
                "schema_version": TOOL._SPAWN_REGISTRATION_SCHEMA_V2,
                "phase": "SPAWN_REGISTRATION",
                "registration_state": "PROCESS_OBJECT_MISSING",
                "later_ledger_write_permitted": False,
            },
        }
        detail = (
            "SLOT_01 bound supervisor bounded process raised before a result: "
            "FAIL_CAMPAIGN_SPAWN_UNCERTAIN:" + inner_detail
        )
        rows = deepcopy(list(zip(ledger.published_names, ledger.published_records)))
        _rewrite_terminal_completion_chain(
            contract,
            capsule,
            authorization,
            rows,
            error_code="FAIL_CAMPAIGN_SPAWN_UNCERTAIN",
            error_detail=detail,
            error_receipt=receipt,
        )
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._verify_campaign_completion_chain_v2(
                contract,
                capsule,
                authorization,
                rows,
            )
        assert error.value.code == "FAIL_CAMPAIGN_LEDGER"
    finally:
        ledger.close()


def test_real_legal_success_chain_is_authoritative_and_progress_is_frozen(
    tmp_path: Path,
) -> None:
    contract, capsule, authorization, ledger_object, result = (
        _completed_real_success_ledger(tmp_path)
    )
    ledger = ledger_object
    try:
        assert result["status"] == contract["success_policy"]["terminal_status"]
        assert result["supervisor_invocation_count"] == 1
        assert result["last_slot_id_or_null"] == "SLOT_01"
        verification = TOOL._verify_campaign_completion_chain_v2(
            contract,
            capsule,
            authorization,
            list(zip(ledger.published_names, ledger.published_records)),
        )
        assert verification["status"] == (
            "AUTHORITATIVE_CAMPAIGN_COMPLETION_VERIFIED"
        )
    finally:
        ledger.close()


@pytest.mark.parametrize("prefix_length", (2, 3, 4, 6, 7, 8, 10, 11, 12, 14))
def test_offline_verifier_accepts_each_legal_error_prefix(
    tmp_path: Path,
    prefix_length: int,
) -> None:
    contract, capsule, authorization, ledger_object = (
        _full_real_gap_ledger_prefix(tmp_path)
    )
    ledger = ledger_object
    try:
        records = deepcopy(
            list(zip(ledger.published_names, ledger.published_records))[
                :prefix_length
            ]
        )
        progress = TOOL._ledger_progress_snapshot_v2(
            [record for _name, record in records]
        )
        status_text = contract["terminal_failure_policy"]["crash_disposition"]
        terminal = TOOL._campaign_terminal_body(
            contract,
            capsule,
            status_text=status_text,
            invocation_count=int(progress["supervisor_invocation_count"]),
            last_slot_id_or_null=progress["last_slot_id_or_null"],
            last_slot_outcome_digest_or_null=progress[
                "last_slot_outcome_digest_or_null"
            ],
            error_code_or_null="KeyboardInterrupt",
            error_detail_or_null=(
                f"injected legal terminal at prefix {prefix_length}"
            ),
        )
        terminal_record = _append_test_ledger_record(
            contract,
            capsule,
            authorization,
            records,
            "campaign.terminal.json",
            "CAMPAIGN_TERMINAL",
            status_text,
            terminal,
        )
        completion = TOOL._campaign_completion_body(
            contract,
            capsule,
            authorization_digest=str(authorization["authorization_digest"]),
            terminal=terminal,
            terminal_ledger_record=terminal_record,
            preceding_ledger_names=[name for name, _record in records],
        )
        _append_test_ledger_record(
            contract,
            capsule,
            authorization,
            records,
            "campaign.completion.json",
            "CAMPAIGN_COMPLETION",
            "CAMPAIGN_COMPLETION_COMMITTED",
            completion,
        )
        verification = TOOL._verify_campaign_completion_chain_v2(
            contract,
            capsule,
            authorization,
            records,
        )
        assert verification["status"] == (
            "AUTHORITATIVE_CAMPAIGN_COMPLETION_VERIFIED"
        )
    finally:
        ledger.close()


def test_offline_verifier_rejects_rehashed_semantic_progress_forgery(
    tmp_path: Path,
) -> None:
    contract, capsule, authorization, ledger_object, _result = (
        _completed_real_success_ledger(tmp_path)
    )
    ledger = ledger_object
    try:
        rows = deepcopy(list(zip(ledger.published_names, ledger.published_records)))
        terminal_record = rows[-2][1]
        terminal = terminal_record["body"]
        assert type(terminal) is dict
        terminal["supervisor_invocation_count"] = 0
        terminal_unsigned = deepcopy(terminal)
        terminal_unsigned.pop("record_digest")
        terminal["record_digest"] = TOOL._domain_digest(
            contract["hash_domains"]["campaign_terminal"],
            terminal_unsigned,
        )
        terminal_record_unsigned = deepcopy(terminal_record)
        terminal_record_unsigned.pop("record_digest")
        terminal_record["record_digest"] = TOOL._domain_digest(
            contract["hash_domains"]["ledger_record"],
            terminal_record_unsigned,
        )
        completion_record = rows[-1][1]
        completion_record["previous_record_digest_or_null"] = terminal_record[
            "record_digest"
        ]
        completion = TOOL._campaign_completion_body(
            contract,
            capsule,
            authorization_digest=str(authorization["authorization_digest"]),
            terminal=terminal,
            terminal_ledger_record=terminal_record,
            preceding_ledger_names=[name for name, _record in rows[:-1]],
        )
        completion_record["body"] = completion
        completion_record_unsigned = deepcopy(completion_record)
        completion_record_unsigned.pop("record_digest")
        completion_record["record_digest"] = TOOL._domain_digest(
            contract["hash_domains"]["ledger_record"],
            completion_record_unsigned,
        )
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._verify_campaign_completion_chain_v2(
                contract,
                capsule,
                authorization,
                rows,
            )
        assert error.value.code == "FAIL_CAMPAIGN_LEDGER"
        assert "terminal progress" in error.value.detail
    finally:
        ledger.close()


@pytest.mark.parametrize("corruption", ("missing", "semantic-link", "order"))
def test_offline_completion_verifier_rejects_non_authoritative_chain(
    tmp_path: Path,
    corruption: str,
) -> None:
    contract, capsule, authorization, ledger_object, _result = (
        _completed_real_ledger(tmp_path)
    )
    ledger = ledger_object
    try:
        rows = deepcopy(list(zip(ledger.published_names, ledger.published_records)))
        if corruption == "missing":
            rows = rows[:-1]
        elif corruption == "order":
            rows[-2], rows[-1] = rows[-1], rows[-2]
        else:
            completion_record = rows[-1][1]
            completion_body = completion_record["body"]
            assert type(completion_body) is dict
            completion_body["terminal_body_digest"] = _tagged("wrong-terminal")
            completion_unsigned = deepcopy(completion_body)
            completion_unsigned.pop("record_digest")
            completion_body["record_digest"] = TOOL._domain_digest(
                contract["hash_domains"]["campaign_completion"],
                completion_unsigned,
            )
            completion_record_unsigned = deepcopy(completion_record)
            completion_record_unsigned.pop("record_digest")
            completion_record["record_digest"] = TOOL._domain_digest(
                contract["hash_domains"]["ledger_record"],
                completion_record_unsigned,
            )
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            TOOL._verify_campaign_completion_chain_v2(
                contract,
                capsule,
                authorization,
                rows,
            )
        assert error.value.code == "FAIL_CAMPAIGN_LEDGER"
    finally:
        ledger.close()


def test_completion_record_cannot_be_prepared_without_terminal_predecessor(
    tmp_path: Path,
) -> None:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        tmp_path / "ledger"
    )
    ledger = TOOL._LedgerV2(
        contract,
        {"capsule_digest": _tagged("capsule")},
        {"authorization_digest": _tagged("authorization")},
    )
    try:
        with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
            ledger.prepare(
                "campaign.completion.json",
                "CAMPAIGN_COMPLETION",
                "CAMPAIGN_COMPLETION_COMMITTED",
                {"record_digest": _tagged("completion")},
            )
        assert error.value.code == "FAIL_CAMPAIGN_LEDGER"
        assert "immediately follow terminal" in error.value.detail
    finally:
        ledger.close()


def test_verified_completion_makes_session_close_faults_nonrevoking(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _contract_value, _capsule, _authorization, ledger_object, _result = (
        _completed_real_ledger(tmp_path)
    )
    ledger = ledger_object
    root_fd = ledger.root_fd
    parent_fd = ledger.parent_fd
    real_flock = TOOL.fcntl.flock
    real_close = TOOL.os.close
    unlock_attempted = False
    close_attempts: list[int] = []

    def fail_unlock(descriptor: int, operation: int) -> None:
        nonlocal unlock_attempted
        if descriptor == root_fd and operation == TOOL.fcntl.LOCK_UN:
            unlock_attempted = True
            raise OSError("injected post-completion unlock failure")
        real_flock(descriptor, operation)

    def fail_session_close(descriptor: int) -> None:
        if descriptor in {root_fd, parent_fd}:
            close_attempts.append(descriptor)
            raise OSError("injected post-completion descriptor close failure")
        real_close(descriptor)

    monkeypatch.setattr(TOOL.fcntl, "flock", fail_unlock)
    monkeypatch.setattr(TOOL.os, "close", fail_session_close)
    try:
        ledger.close()
        assert unlock_attempted is True
        assert close_attempts == [root_fd, parent_fd]
        assert ledger.completion_commit_verified is True
        assert ledger.root_fd is None
        assert ledger.parent_fd is None
    finally:
        real_close(root_fd)
        real_close(parent_fd)


def test_completion_publication_fault_forbids_any_later_ledger_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        tmp_path / "ledger"
    )
    capsule = {"capsule_digest": _tagged("capsule")}
    authorization = TOOL._authorization_template(contract, capsule)
    ledger = TOOL._LedgerV2(contract, capsule, authorization)
    ledger.publish(
        "campaign.intent.json",
        "CAMPAIGN_INTENT",
        "AUTHORIZED_CAMPAIGN_INTENT_STAGED",
        {
            "capsule": deepcopy(capsule),
            "authorization_digest": authorization["authorization_digest"],
        },
    )
    ledger.publish(
        "campaign.authorization-consumed.json",
        "CAMPAIGN_AUTHORIZATION_CONSUMED",
        "CAMPAIGN_AUTHORIZATION_CONSUMED",
        {
            "authorization": deepcopy(authorization),
            "consumed_once": True,
        },
    )
    real_publish = TOOL._publish_immutable_at

    def publish_then_fault(
        directory_fd: int,
        basename: str,
        value_or_bytes: Mapping[str, object] | bytes,
        **kwargs: object,
    ) -> bytes:
        payload = real_publish(
            directory_fd,
            basename,
            value_or_bytes,
            **kwargs,
        )
        if basename == "campaign.completion.json":
            raise OSError("injected post-helper completion fault")
        return payload

    monkeypatch.setattr(TOOL, "_publish_immutable_at", publish_then_fault)
    try:
        result = TOOL._publish_terminal(
            ledger,
            contract,
            capsule,
            status_text=contract["terminal_failure_policy"][
                "crash_disposition"
            ],
            invocation_count=0,
            last_slot_id_or_null=None,
            last_slot_outcome_digest_or_null=None,
            error_code_or_null="KeyboardInterrupt",
            error_detail_or_null="injected pre-slot terminal",
        )
        assert result["completion_commit_verified"] is True
        assert ledger.published_names.count("campaign.terminal.json") == 1
        assert ledger.published_names.count("campaign.completion.json") == 1
        assert ledger.completion_commit_verified is True
        root = Path(str(contract["fixed_external_roots"]["ledger_root"]))
        assert (root / "campaign.completion.json").is_file()
    finally:
        ledger.close()


def test_ledger_precreate_failure_does_not_advance_chain_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    ledger_root = tmp_path / "ledger-root"
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        ledger_root
    )
    capsule = {"capsule_digest": _tagged("capsule")}
    authorization = {"authorization_digest": _tagged("authorization")}
    ledger = TOOL._LedgerV2(contract, capsule, authorization)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                TOOL,
                "_publish_immutable_at",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    MemoryError("injected before O_CREAT")
                ),
            )
            with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
                ledger.publish(
                    "campaign.intent.json",
                    "CAMPAIGN_INTENT",
                    "AUTHORIZED_CAMPAIGN_INTENT_STAGED",
                    {"test": True},
                )
            assert error.value.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
        assert ledger.sequence == 0
        assert ledger.previous_record_digest is None
        assert ledger.published_names == []
        assert ledger.last_name_index == -1
        assert not ledger_root.joinpath("campaign.intent.json").exists()
    finally:
        ledger.close()


@pytest.mark.parametrize(
    ("decisions", "expected_calls", "status_key"),
    [
        ((CORE.SUCCESS_V2,), 1, "success"),
        ((CORE.RETRY_EXACT_GAP_V2, CORE.SUCCESS_V2), 2, "success"),
        (
            (
                CORE.RETRY_EXACT_GAP_V2,
                CORE.RETRY_EXACT_GAP_V2,
                CORE.EXHAUSTED_RETRYABLE_GAP_V2,
            ),
            3,
            "exhausted",
        ),
        ((CORE.TERMINAL_NONRETRYABLE_V2,), 1, "terminal"),
        (
            (
                CORE.RETRY_EXACT_GAP_V2,
                CORE.RETRY_EXACT_GAP_V2,
                CORE.RETRY_EXACT_GAP_V2,
            ),
            3,
            "crash",
        ),
    ],
)
def test_run_state_machine_is_finite_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    decisions: tuple[str, ...],
    expected_calls: int,
    status_key: str,
) -> None:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = str(tmp_path / "ledger")
    capsule = {"capsule_digest": _tagged("capsule")}
    authorization = {"authorization_digest": _tagged("authorization")}
    slots = [
        {"slot_id": f"SLOT_{index:02d}", "slot_ordinal": index}
        for index in range(1, 4)
    ]
    context = {"capsule": capsule, "slots": slots}
    ledger = _FakeLedger()
    calls: list[str] = []

    def run_one_slot(
        _contract_value: Mapping[str, object],
        _context: Mapping[str, object],
        _capsule: Mapping[str, object],
        _ledger: object,
        slot: Mapping[str, object],
        invocation_counter: list[int],
    ) -> tuple[dict[str, object], str]:
        index = len(calls)
        assert index < len(decisions)
        calls.append(str(slot["slot_id"]))
        _ledger.publish(
            f"slot-{index + 1:02d}.consumed.json",
            "SLOT_CONSUMED",
            f"{slot['slot_id']}_CONSUMED_OUTCOME_UNKNOWN",
            {"slot_id": slot["slot_id"]},
        )
        invocation_counter[0] += 1
        outcome = {
            "slot_id": slot["slot_id"],
            "record_digest": _tagged(f"outcome-{index + 1}"),
        }
        _ledger.publish(
            f"slot-{index + 1:02d}.outcome.json",
            "SLOT_OUTCOME",
            decisions[index],
            outcome,
        )
        return (
            outcome,
            decisions[index],
        )

    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_preflight_context", lambda _contract: context)
    monkeypatch.setattr(
        TOOL,
        "_load_authorization",
        lambda _contract, _capsule: authorization,
    )
    monkeypatch.setattr(TOOL, "_LedgerV2", lambda *_args: ledger)
    monkeypatch.setattr(TOOL, "_run_one_slot", run_one_slot)
    monkeypatch.setattr(TOOL, "_load_core", lambda _contract: CORE)

    result = TOOL.run_phase3_q05b_shadow_retry_campaign_v2(
        capsule["capsule_digest"]
    )
    expected_status = {
        "success": contract["success_policy"]["terminal_status"],
        "exhausted": contract["terminal_failure_policy"][
            "slot_03_exact_gap_disposition"
        ],
        "terminal": contract["terminal_failure_policy"][
            "generic_nonretry_disposition"
        ],
        "crash": contract["terminal_failure_policy"]["crash_disposition"],
    }[status_key]
    assert result["status"] == expected_status
    assert result["supervisor_invocation_count"] == expected_calls
    assert calls == [f"SLOT_{index:02d}" for index in range(1, expected_calls + 1)]
    assert len(calls) <= 3
    assert ledger.published_names[:2] == [
        "campaign.intent.json",
        "campaign.authorization-consumed.json",
    ]
    assert ledger.published_names[-2:] == [
        "campaign.terminal.json",
        "campaign.completion.json",
    ]
    assert ledger.published_names.count("campaign.terminal.json") == 1
    assert ledger.published_names.count("campaign.completion.json") == 1
    assert result["completion_commit_verified"] is True
    assert ledger.closed is True


def test_publication_uncertainty_forbids_terminal_second_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        tmp_path / "ledger"
    )
    capsule = {"capsule_digest": _tagged("capsule")}
    authorization = {"authorization_digest": _tagged("authorization")}
    context = {
        "capsule": capsule,
        "slots": [{"slot_id": "SLOT_01", "slot_ordinal": 1}],
    }
    ledger = _FakeLedger()

    def uncertain_slot(*_args: object, **_kwargs: object) -> object:
        raise TOOL.ShadowRetryCampaignErrorV2(
            "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN",
            "injected consumed publication uncertainty",
            terminal=True,
        )

    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_preflight_context", lambda _contract: context)
    monkeypatch.setattr(
        TOOL,
        "_load_authorization",
        lambda _contract, _capsule: authorization,
    )
    monkeypatch.setattr(TOOL, "_LedgerV2", lambda *_args: ledger)
    monkeypatch.setattr(TOOL, "_run_one_slot", uncertain_slot)

    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL.run_phase3_q05b_shadow_retry_campaign_v2(
            capsule["capsule_digest"]
        )
    assert error.value.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
    assert ledger.published_names == [
        "campaign.intent.json",
        "campaign.authorization-consumed.json",
    ]
    assert "campaign.terminal.json" not in ledger.published_names
    assert ledger.closed is True


def test_spawn_uncertainty_forbids_terminal_completion_or_later_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        tmp_path / "ledger"
    )
    capsule = {"capsule_digest": _tagged("capsule")}
    authorization = {"authorization_digest": _tagged("authorization")}
    context = {
        "capsule": capsule,
        "slots": [{"slot_id": "SLOT_01", "slot_ordinal": 1}],
    }
    ledger = _FakeLedger()

    def uncertain_spawn(*_args: object, **_kwargs: object) -> object:
        raise TOOL.ShadowRetryCampaignErrorV2(
            "FAIL_CAMPAIGN_SPAWN_UNCERTAIN",
            "injected missing registration receipt",
            terminal=True,
            receipt={"cleanup_certain": False},
        )

    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_preflight_context", lambda _contract: context)
    monkeypatch.setattr(
        TOOL,
        "_load_authorization",
        lambda _contract, _capsule: authorization,
    )
    monkeypatch.setattr(TOOL, "_LedgerV2", lambda *_args: ledger)
    monkeypatch.setattr(TOOL, "_run_one_slot", uncertain_spawn)

    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL.run_phase3_q05b_shadow_retry_campaign_v2(
            capsule["capsule_digest"]
        )
    assert error.value.code == "FAIL_CAMPAIGN_SPAWN_UNCERTAIN"
    assert ledger.published_names == [
        "campaign.intent.json",
        "campaign.authorization-consumed.json",
    ]
    assert "campaign.terminal.json" not in ledger.published_names
    assert "campaign.completion.json" not in ledger.published_names
    assert ledger.closed is True


def test_terminal_publication_uncertainty_supersedes_original_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    capsule = {"capsule_digest": _tagged("capsule")}
    authorization = {"authorization_digest": _tagged("authorization")}
    context = {
        "capsule": capsule,
        "slots": [{"slot_id": "SLOT_01", "slot_ordinal": 1}],
    }

    class TerminalUncertainLedger(_FakeLedger):
        def publish(
            self,
            basename: str,
            record_type: str,
            status_text: str,
            body: Mapping[str, object],
        ) -> dict[str, object]:
            if basename == "campaign.terminal.json":
                raise TOOL.ShadowRetryCampaignErrorV2(
                    "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN",
                    "injected terminal fsync uncertainty",
                    terminal=True,
                )
            return super().publish(basename, record_type, status_text, body)

    ledger = TerminalUncertainLedger()

    def source_failure(*_args: object, **_kwargs: object) -> object:
        raise TOOL.ShadowRetryCampaignErrorV2(
            "FAIL_CAMPAIGN_SOURCE",
            "original source error",
            terminal=True,
        )

    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_preflight_context", lambda _contract: context)
    monkeypatch.setattr(
        TOOL,
        "_load_authorization",
        lambda _contract, _capsule: authorization,
    )
    monkeypatch.setattr(TOOL, "_LedgerV2", lambda *_args: ledger)
    monkeypatch.setattr(TOOL, "_run_one_slot", source_failure)

    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL.run_phase3_q05b_shadow_retry_campaign_v2(
            capsule["capsule_digest"]
        )
    assert error.value.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
    assert "original source error" not in error.value.detail
    assert "injected terminal fsync uncertainty" in error.value.detail
    assert type(error.value.receipt) is dict
    assert error.value.receipt["schema_version"] == (
        "hegel-phase3a-q05b-shadow-retry-terminalization-failure/2"
    )
    assert error.value.receipt["original_error"]["error_code"] == (
        "FAIL_CAMPAIGN_SOURCE"
    )
    assert error.value.receipt["terminalization_error"]["error_code"] == (
        "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
    )
    assert len(TOOL._canonical_json_bytes(error.value.receipt)) <= (
        TOOL._MAX_ERROR_RECEIPT_BYTES
    )
    assert ledger.published_names == [
        "campaign.intent.json",
        "campaign.authorization-consumed.json",
    ]
    assert ledger.closed is True


def test_result_construction_failure_after_terminal_staging_leaves_no_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    capsule = {"capsule_digest": _tagged("capsule")}
    ledger = _FakeLedger()

    def fail_result(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise MemoryError("injected completion result construction failure")

    monkeypatch.setattr(
        TOOL,
        "_campaign_result_from_prepared_completion",
        fail_result,
    )
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._publish_terminal(
            ledger,  # type: ignore[arg-type]
            contract,
            capsule,
            status_text="SUCCESS",
            invocation_count=1,
            last_slot_id_or_null="SLOT_01",
            last_slot_outcome_digest_or_null=_tagged("outcome"),
        )
    assert error.value.code == "FAIL_CAMPAIGN_PUBLICATION_UNCERTAIN"
    assert "completion result construction failure" in error.value.detail
    assert ledger.published_names == ["campaign.terminal.json"]
    assert ledger.completion_commit_verified is False


def test_cli_result_is_fully_built_before_completion_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_built = False
    real_builder = TOOL._campaign_result_from_prepared_completion

    class GuardedLedger(_FakeLedger):
        def publish_prepared(
            self,
            basename: str,
            prepared_record: Mapping[str, object],
        ) -> dict[str, object]:
            if basename == "campaign.completion.json":
                assert result_built is True
            return super().publish_prepared(basename, prepared_record)

    def build_then_mark(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal result_built
        result = real_builder(*args, **kwargs)
        result_built = True
        return result

    monkeypatch.setattr(
        TOOL,
        "_campaign_result_from_prepared_completion",
        build_then_mark,
    )
    ledger = GuardedLedger()
    result = TOOL._publish_terminal(
        ledger,  # type: ignore[arg-type]
        _contract(),
        {"capsule_digest": _tagged("capsule")},
        status_text="SUCCESS",
        invocation_count=1,
        last_slot_id_or_null="SLOT_01",
        last_slot_outcome_digest_or_null=_tagged("outcome"),
    )
    assert result_built is True
    assert result["completion_commit_verified"] is True
    assert ledger.published_names == [
        "campaign.terminal.json",
        "campaign.completion.json",
    ]


def test_verified_completion_supersedes_postcommit_async_fault() -> None:
    class PostcommitAsyncLedger(_FakeLedger):
        def publish_prepared(
            self,
            basename: str,
            prepared_record: Mapping[str, object],
        ) -> dict[str, object]:
            published = super().publish_prepared(basename, prepared_record)
            if basename == "campaign.completion.json":
                assert self.completion_commit_verified is True
                raise KeyboardInterrupt("injected after verified completion")
            return published

    ledger = PostcommitAsyncLedger()
    result = TOOL._publish_terminal(
        ledger,  # type: ignore[arg-type]
        _contract(),
        {"capsule_digest": _tagged("capsule")},
        status_text="SUCCESS",
        invocation_count=1,
        last_slot_id_or_null="SLOT_01",
        last_slot_outcome_digest_or_null=_tagged("outcome"),
    )
    assert result["status"] == "SUCCESS"
    assert result["completion_commit_verified"] is True
    assert ledger.completion_commit_verified is True
    assert ledger.published_names[-1] == "campaign.completion.json"


def test_completion_store_attr_gap_recovers_from_offline_verified_disk(
    tmp_path: Path,
) -> None:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        tmp_path / "opcode-gap-ledger"
    )
    capsule = {"capsule_digest": _tagged("capsule")}
    authorization = TOOL._authorization_template(contract, capsule)
    ledger = TOOL._LedgerV2(contract, capsule, authorization)
    ledger.publish(
        "campaign.intent.json",
        "CAMPAIGN_INTENT",
        "AUTHORIZED_CAMPAIGN_INTENT_STAGED",
        {
            "capsule": deepcopy(capsule),
            "authorization_digest": authorization["authorization_digest"],
        },
    )
    ledger.publish(
        "campaign.authorization-consumed.json",
        "CAMPAIGN_AUTHORIZATION_CONSUMED",
        "CAMPAIGN_AUTHORIZATION_CONSUMED",
        {
            "authorization": deepcopy(authorization),
            "consumed_once": True,
        },
    )
    callback_code = next(
        constant
        for constant in TOOL._LedgerV2.publish_prepared.__code__.co_consts
        if type(constant) is type(TOOL._LedgerV2.publish_prepared.__code__)
        and constant.co_name == "commit_verified_publication"
    )
    store_offset = next(
        instruction.offset
        for instruction in dis.get_instructions(callback_code)
        if instruction.opname == "STORE_ATTR"
        and instruction.argval == "authoritative_completion_result_or_null"
    )
    injected = False

    def tracer(frame: object, event: str, _arg: object) -> object:
        nonlocal injected
        if getattr(frame, "f_code", None) is callback_code:
            if event == "call":
                frame.f_trace_opcodes = True
            elif (
                event == "opcode"
                and frame.f_lasti == store_offset
                and not injected
            ):
                injected = True
                raise KeyboardInterrupt(
                    "injected before authoritative STORE_ATTR"
                )
        return tracer

    sys.settrace(tracer)
    try:
        result = TOOL._publish_terminal(
            ledger,
            contract,
            capsule,
            status_text=contract["terminal_failure_policy"][
                "crash_disposition"
            ],
            invocation_count=0,
            last_slot_id_or_null=None,
            last_slot_outcome_digest_or_null=None,
            error_code_or_null="KeyboardInterrupt",
            error_detail_or_null="legal pre-slot opcode-gap terminal",
        )
    finally:
        sys.settrace(None)
    try:
        assert injected is True
        assert result["completion_commit_verified"] is True
        assert ledger.completion_commit_verified is True
        assert ledger.published_names.count("campaign.terminal.json") == 1
        assert Path(
            str(contract["fixed_external_roots"]["ledger_root"])
        ).joinpath("campaign.completion.json").is_file()
    finally:
        ledger.close()


def test_run_recovers_verified_completion_across_caller_return_gap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract()
    contract["fixed_external_roots"]["ledger_root"] = os.fspath(
        tmp_path / "run-return-gap-ledger"
    )
    capsule = {"capsule_digest": _tagged("capsule")}
    authorization = {"authorization_digest": _tagged("authorization")}
    context = {
        "capsule": capsule,
        "slots": [{"slot_id": "SLOT_01", "slot_ordinal": 1}],
    }
    ledger = _FakeLedger()

    def successful_slot(
        _contract_value: Mapping[str, object],
        _context: Mapping[str, object],
        _capsule: Mapping[str, object],
        ledger_value: _FakeLedger,
        slot: Mapping[str, object],
        invocation_counter: list[int],
    ) -> tuple[dict[str, object], str]:
        ledger_value.publish(
            "slot-01.consumed.json",
            "SLOT_CONSUMED",
            "SLOT_01_CONSUMED_OUTCOME_UNKNOWN",
            {"slot_id": "SLOT_01"},
        )
        invocation_counter[0] += 1
        outcome = {
            "slot_id": slot["slot_id"],
            "record_digest": _tagged("run-return-outcome"),
        }
        ledger_value.publish(
            "slot-01.outcome.json",
            "SLOT_OUTCOME",
            CORE.SUCCESS_V2,
            outcome,
        )
        return outcome, CORE.SUCCESS_V2

    real_terminal = TOOL._publish_terminal

    def terminal_then_return_fault(*args: object, **kwargs: object) -> object:
        real_terminal(*args, **kwargs)
        raise KeyboardInterrupt("injected caller RETURN-to-STORE gap")

    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_preflight_context", lambda _contract: context)
    monkeypatch.setattr(
        TOOL,
        "_load_authorization",
        lambda _contract, _capsule: authorization,
    )
    monkeypatch.setattr(TOOL, "_LedgerV2", lambda *_args: ledger)
    monkeypatch.setattr(TOOL, "_run_one_slot", successful_slot)
    monkeypatch.setattr(TOOL, "_load_core", lambda _contract: CORE)
    monkeypatch.setattr(TOOL, "_publish_terminal", terminal_then_return_fault)
    result = TOOL.run_phase3_q05b_shadow_retry_campaign_v2(
        capsule["capsule_digest"]
    )
    assert result["completion_commit_verified"] is True
    assert ledger.published_names.count("campaign.terminal.json") == 1
    assert ledger.published_names.count("campaign.completion.json") == 1


def test_bounded_command_failure_receipt_is_durable_in_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    argv = [
        "/usr/bin/python3.10",
        "-I",
        "-S",
        "-B",
        "-c",
        "import sys;sys.stderr.write('exact clone reason\\n');sys.exit(7)",
    ]
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as command_error:
        TOOL._run_exact_command(
            argv,
            cwd=os.fspath(tmp_path),
            env={"PATH": "/usr/bin:/bin"},
            timeout_seconds=5,
            maximum_bytes=4096,
            label="injected clone command",
        )
    captured = command_error.value
    assert captured.code == "FAIL_CAMPAIGN_PROCESS"
    assert type(captured.receipt) is dict
    assert captured.receipt["argv"] == argv
    assert captured.receipt["return_code"] == 7
    assert bytes.fromhex(
        str(captured.receipt["stderr"]["excerpt_hex"])
    ) == b"exact clone reason\n"

    contract = _contract()
    capsule = {"capsule_digest": _tagged("capsule")}
    authorization = {"authorization_digest": _tagged("authorization")}
    context = {
        "capsule": capsule,
        "slots": [{"slot_id": "SLOT_01", "slot_ordinal": 1}],
    }
    ledger = _FakeLedger()
    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_preflight_context", lambda _contract: context)
    monkeypatch.setattr(
        TOOL,
        "_load_authorization",
        lambda _contract, _capsule: authorization,
    )
    monkeypatch.setattr(TOOL, "_LedgerV2", lambda *_args: ledger)
    monkeypatch.setattr(
        TOOL,
        "_run_one_slot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(captured),
    )
    result = TOOL.run_phase3_q05b_shadow_retry_campaign_v2(
        capsule["capsule_digest"]
    )
    assert result["error_receipt_or_null"] == captured.receipt
    terminal_body = ledger.rows[-2][3]
    assert terminal_body["error_receipt_or_null"] == captured.receipt
    assert terminal_body["error_receipt_or_null"]["stderr"][
        "raw_sha256"
    ] == sha256(b"exact clone reason\n").hexdigest()


def _registered_certain_git_failure(
    contract: Mapping[str, object],
) -> tuple[list[str], str, dict[str, object], str]:
    execution = contract["execution_policy"]
    target = contract["target_commit_a_binding"]
    assert type(execution) is dict and type(target) is dict
    arguments = [
        "-C",
        str(execution["clone_source_repository"]),
        "rev-parse",
        f"{target['target_supervisor_commit']}^{{tree}}",
    ]
    label = "target tree replay"
    argv = [str(execution["git_executable"]), *arguments]
    receipt = TOOL._bounded_process_diagnostic(
        label,
        argv,
        {
            "return_code": 7,
            "timed_out": False,
            "stdout_overflow": False,
            "stderr_overflow": False,
            "child_reaped": True,
            "process_group_absent": True,
            "wrapper_resources_closed": True,
            "stdout": b"",
            "stderr": b"fatal: exact Git source failure\n",
        },
        classification={
            "require_empty_stdout": False,
            "require_empty_stderr": True,
            "accepted": False,
        },
    )
    detail = (
        f"{label} failed closed: return_code=7; timed_out=False; "
        "stdout_overflow=False; stderr_overflow=False; "
        f"stderr_excerpt_hex={receipt['stderr']['excerpt_hex']}"
    )
    return arguments, label, receipt, detail


def test_certain_registered_git_failure_is_source_not_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    arguments, label, receipt, detail = _registered_certain_git_failure(
        contract
    )
    failure = TOOL.ShadowRetryCampaignErrorV2(
        "FAIL_CAMPAIGN_PROCESS",
        detail,
        terminal=True,
        receipt=receipt,
    )

    def fail_exact_command(*_args: object, **_kwargs: object) -> bytes:
        raise failure

    monkeypatch.setattr(TOOL, "_run_exact_command", fail_exact_command)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._git(contract, arguments, label)
    captured = error.value
    assert captured.code == "FAIL_CAMPAIGN_SOURCE"
    assert captured.detail == detail
    assert captured.receipt == receipt
    assert captured.terminal is True
    assert TOOL._error_code_terminal_disposition(
        contract,
        captured.code,
    ) == contract["terminal_failure_policy"][
        "source_or_binding_uncertainty_disposition"
    ]
    assert TOOL._terminal_error_receipt_semantics_v2(
        contract,
        receipt,
        terminal_error_code="FAIL_CAMPAIGN_SOURCE",
        terminal_error_detail=detail,
        top_level=True,
    )
    assert not TOOL._terminal_error_receipt_semantics_v2(
        contract,
        receipt,
        terminal_error_code="FAIL_CAMPAIGN_PROCESS",
        terminal_error_detail=detail,
        top_level=True,
    )


@pytest.mark.parametrize(
    "uncertainty",
    (
        "spawn",
        "exception",
        "close",
        "cleanup",
        "timeout",
        "overflow",
        "unregistered",
    ),
)
def test_git_never_reclassifies_process_or_receipt_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
    uncertainty: str,
) -> None:
    contract = _contract()
    arguments, label, base_receipt, detail = _registered_certain_git_failure(
        contract
    )
    receipt = deepcopy(base_receipt)
    if uncertainty == "spawn":
        receipt = {
            "schema_version": TOOL._SPAWN_REGISTRATION_SCHEMA_V2,
            "registration_state": "INVALID",
        }
    elif uncertainty == "exception":
        receipt["schema_version"] = (
            TOOL._BOUNDED_COMMAND_EXCEPTION_DIAGNOSTIC_SCHEMA_V2
        )
    elif uncertainty == "close":
        receipt["wrapper_resources_closed"] = False
        receipt["classification"] = {
            "accepted": False,
            "close_errors": ["injected close uncertainty"],
            "prior_error_or_null": None,
        }
    elif uncertainty == "cleanup":
        receipt["child_reaped"] = False
    elif uncertainty == "timeout":
        receipt["timed_out"] = True
    elif uncertainty == "overflow":
        receipt["stderr_overflow"] = True
    else:
        receipt["label"] = "unregistered Git command"
    failure = TOOL.ShadowRetryCampaignErrorV2(
        "FAIL_CAMPAIGN_PROCESS",
        detail,
        terminal=True,
        receipt=receipt,
    )

    def fail_exact_command(*_args: object, **_kwargs: object) -> bytes:
        raise failure

    monkeypatch.setattr(TOOL, "_run_exact_command", fail_exact_command)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._git(contract, arguments, label)
    assert error.value.code == "FAIL_CAMPAIGN_PROCESS"
    assert error.value.receipt == receipt


def test_bounded_process_exception_receipts_preserve_command_phase(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def selector_failure(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise OSError("injected selector setup failure")

    monkeypatch.setattr(TOOL, "_bounded_process", selector_failure)
    argv = ["/usr/bin/git", "status", "--porcelain=v1"]
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as command_error:
        TOOL._run_exact_command(
            argv,
            cwd=os.fspath(tmp_path),
            env={"PATH": "/usr/bin:/bin"},
            timeout_seconds=5,
            maximum_bytes=4096,
            label="post-run Git replay",
        )
    assert command_error.value.code == "FAIL_CAMPAIGN_PROCESS"
    receipt = command_error.value.receipt
    assert type(receipt) is dict
    assert receipt["label"] == "post-run Git replay"
    assert receipt["argv"] == argv
    assert receipt["phase"] == "BOUNDED_PROCESS_RAISED_BEFORE_RESULT"
    assert receipt["error_code"] == "OSError"
    assert receipt["stdout"] == {"available": False}
    assert "injected selector setup failure" in str(receipt["error_detail"])

    absent, rows, docker_error = TOOL._docker_inventory(_contract())
    assert absent is False
    assert rows == []
    assert type(docker_error) is str
    docker_receipt = TOOL._strict_json(
        docker_error.encode("ascii") + b"\n",
        "Docker exception receipt KAT",
        64 * 1024,
    )
    assert docker_receipt["label"] == "Docker read-only inventory"
    assert docker_receipt["argv"][2] == "ps"
    assert docker_receipt["phase"] == "BOUNDED_PROCESS_RAISED_BEFORE_RESULT"
    assert "injected selector setup failure" in docker_receipt["error_detail"]


def test_bounded_process_never_signals_reused_group_after_child_reap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    kill_calls: list[int] = []

    def forbidden_postreap_kill(
        process: object,
        *,
        first_signal: int,
        grace_seconds: float,
        reap_seconds: float,
    ) -> bool:
        del first_signal, grace_seconds, reap_seconds
        kill_calls.append(int(getattr(process, "pid")))
        raise AssertionError("numeric PGID must not be signalled after reap")

    monkeypatch.setattr(TOOL, "_process_group_present", lambda _pid: True)
    monkeypatch.setattr(TOOL, "_kill_and_reap", forbidden_postreap_kill)
    result = TOOL._bounded_process(
        [
            "/usr/bin/python3.10",
            "-I",
            "-S",
            "-B",
            "-c",
            "pass",
        ],
        cwd=os.fspath(tmp_path),
        env={"PATH": "/usr/bin:/bin"},
        timeout_seconds=5,
        stdout_max_bytes=4096,
        stderr_max_bytes=4096,
    )
    assert result["return_code"] == 0
    assert result["child_reaped"] is True
    assert result["process_group_absent"] is False
    assert result["wrapper_resources_closed"] is True
    assert kill_calls == []


def test_selector_setup_failure_reaps_live_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cleanup_rows: list[tuple[int, bool, int | None]] = []
    real_kill_and_reap = TOOL._kill_and_reap

    def tracked_kill_and_reap(
        process: object,
        *,
        first_signal: int,
        grace_seconds: float,
        reap_seconds: float,
    ) -> bool:
        reaped = real_kill_and_reap(
            process,
            first_signal=first_signal,
            grace_seconds=grace_seconds,
            reap_seconds=reap_seconds,
        )
        cleanup_rows.append(
            (int(getattr(process, "pid")), reaped, process.poll())
        )
        return reaped

    def fail_selector_setup() -> object:
        raise OSError("injected selector construction failure")

    monkeypatch.setattr(TOOL, "_kill_and_reap", tracked_kill_and_reap)
    monkeypatch.setattr(TOOL.selectors, "DefaultSelector", fail_selector_setup)
    with pytest.raises(OSError, match="selector construction failure"):
        TOOL._bounded_process(
            [
                "/usr/bin/python3.10",
                "-I",
                "-S",
                "-B",
                "-c",
                "import time;time.sleep(60)",
            ],
            cwd=os.fspath(tmp_path),
            env={"PATH": "/usr/bin:/bin"},
            timeout_seconds=5,
            stdout_max_bytes=4096,
            stderr_max_bytes=4096,
        )
    assert len(cleanup_rows) == 1
    assert cleanup_rows[0][1] is True
    assert cleanup_rows[0][2] is not None


def test_selector_close_failure_is_typed_and_retains_process_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_factory = TOOL.selectors.DefaultSelector

    class CloseFailSelector:
        def __init__(self) -> None:
            self.delegate = real_factory()

        def __getattr__(self, name: str) -> object:
            return getattr(self.delegate, name)

        def close(self) -> None:
            self.delegate.close()
            raise OSError("injected selector close failure")

    monkeypatch.setattr(TOOL.selectors, "DefaultSelector", CloseFailSelector)
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._bounded_process(
            [
                "/usr/bin/python3.10",
                "-I",
                "-S",
                "-B",
                "-c",
                "pass",
            ],
            cwd=os.fspath(tmp_path),
            env={"PATH": "/usr/bin:/bin"},
            timeout_seconds=5,
            stdout_max_bytes=4096,
            stderr_max_bytes=4096,
        )
    assert error.value.code == "FAIL_CAMPAIGN_PROCESS"
    assert "injected selector close failure" in error.value.detail
    assert type(error.value.receipt) is dict
    assert error.value.receipt["return_code"] == 0
    assert error.value.receipt["child_reaped"] is True
    assert error.value.receipt["wrapper_resources_closed"] is False
    assert error.value.receipt["classification"]["accepted"] is False


def test_safe_postrun_source_replay_preserves_bounded_command_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = {
        "schema_version": (
            "hegel-phase3a-q05b-shadow-retry-bounded-command-diagnostic/2"
        ),
        "argv": ["/usr/bin/git", "status", "--porcelain=v1"],
        "return_code": 7,
        "stderr": {
            "byte_count": 19,
            "raw_sha256": sha256(b"post-run git failed").hexdigest(),
            "excerpt_hex": b"post-run git failed".hex(),
            "excerpt_byte_count": 19,
            "excerpt_truncated": False,
        },
    }
    failure = TOOL.ShadowRetryCampaignErrorV2(
        "FAIL_CAMPAIGN_PROCESS",
        "post-run source Git replay failed closed",
        terminal=True,
        receipt=receipt,
    )

    def fail_replay(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise failure

    monkeypatch.setattr(TOOL, "_source_state_replay", fail_replay)
    result = TOOL._safe_source_state_replay(
        {},
        {},
        {},
        artifact_presence="ABSENT",
    )
    assert result["certain"] is False
    assert result["error_or_null"] == (
        "FAIL_CAMPAIGN_PROCESS:post-run source Git replay failed closed"
    )
    assert result["evidence"]["error_receipt_or_null"] == receipt
    assert result["evidence"]["error_receipt_or_null"]["argv"] == receipt["argv"]
    assert result["root"] == TOOL._domain_digest(
        "HEGEL/PHASE3/Q05B/SHADOW_RETRY_CAMPAIGN_V2R1/SOURCE_REPLAY/V2",
        result["evidence"],
    )


@pytest.mark.parametrize(
    ("updates", "expected_policy_field"),
    (
        (
            {"trajectory_status": "INVALID"},
            "trajectory_incomplete_or_unverifiable_disposition",
        ),
        (
            {"cleanup_certain": False},
            "cleanup_or_survivor_uncertainty_disposition",
        ),
        (
            {"source_binding_certain": False},
            "source_or_binding_uncertainty_disposition",
        ),
    ),
)
def test_terminal_disposition_uses_explicit_observation_class(
    updates: Mapping[str, object],
    expected_policy_field: str,
) -> None:
    contract = _contract()
    observation: dict[str, object] = {
        "trajectory_status": "COMPLETE_FAIL",
        "trajectory_torn_tail_or_null": None,
        "trajectory_open_span_ids": [],
        "trajectory_trace_id": "trace",
        "expected_trace_id": "trace",
        "trajectory_producer_id": "producer",
        "expected_producer_id": "producer",
        "cleanup_certain": True,
        "docker_absence_certain": True,
        "supervisor_reaped": True,
        "process_group_absent": True,
        "clone_clean_certain": True,
        "cargo_isolation_certain": True,
        "source_binding_certain": True,
        "external_binding_certain": True,
        "replay_certain": True,
    }
    observation.update(updates)
    assert TOOL._observation_terminal_disposition(
        contract,
        observation,
    ) == contract["terminal_failure_policy"][expected_policy_field]


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--pre"],
        ["--preflight", "--run"],
        ["--run", "--unknown"],
    ],
)
def test_cli_parser_rejects_missing_ambiguous_or_abbreviated_modes(
    arguments: list[str],
) -> None:
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._parser().parse_args(arguments)
    assert error.value.code == "FAIL_CAMPAIGN_ARGUMENT"


def test_cli_runtime_rejects_nonisolated_test_interpreter() -> None:
    contract = _contract()
    contract["execution_policy"]["supervisor_interpreter"] = sys.executable
    with pytest.raises(TOOL.ShadowRetryCampaignErrorV2) as error:
        TOOL._require_exact_cli_runtime(contract)
    assert error.value.code == "FAIL_CAMPAIGN_RUNTIME"


def test_cli_preflight_rejects_capsule_argument_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    contract = _contract()
    called = False

    def forbidden_preflight() -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(
        TOOL,
        "load_phase3_q05b_shadow_retry_campaign_contract_v2",
        lambda: contract,
    )
    monkeypatch.setattr(TOOL, "_require_exact_cli_runtime", lambda _contract: None)
    monkeypatch.setattr(
        TOOL,
        "preflight_phase3_q05b_shadow_retry_campaign_v2",
        forbidden_preflight,
    )
    assert TOOL.main(
        ["--preflight", "--capsule-digest", _tagged("capsule")]
    ) == 1
    captured = capsysbinary.readouterr()
    assert captured.err == b""
    value = TOOL._strict_json(captured.out, "CLI error")
    assert type(value) is dict
    assert value["status"] == "CAMPAIGN_COMMAND_FAILED_CLOSED"
    assert value["error_code"] == "FAIL_CAMPAIGN_ARGUMENT"
    assert value["resumable"] is False
    assert called is False
