from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile

import pytest

from hegel_machine import phase3_q05b_trajectory_v2 as trajectory


@pytest.fixture
def private_parent() -> Path:
    # The configured Windows TEMP mount cannot represent POSIX modes.  The
    # durable-record contract is POSIX-specific, so exercise it on /tmp.
    parent = Path(tempfile.mkdtemp(prefix="hegel-q05b-trajectory-v2-", dir="/tmp"))
    parent.chmod(0o700)
    try:
        yield parent
    finally:
        shutil.rmtree(parent)


def _ticks(start: int = 100):
    return itertools.count(start).__next__


def _writer(parent: Path, name: str = "trace-v2", *, clock=None):
    return trajectory.create_durable_trajectory_writer_v2(
        parent / name,
        expected_parent=parent,
        trace_id=f"trace-{name}",
        producer_id="q05b-supervisor-v2",
        clock_ns=clock or _ticks(10_000_000_000),
    )


def _complete_ok(parent: Path, name: str = "trace-v2") -> Path:
    root = parent / name
    with _writer(parent, name) as writer:
        writer.begin(
            "root-span",
            "dual-qualification",
            stage="VALIDATE",
            substep="source",
            actor="supervisor",
        )
        writer.ok("root-span", detail="validated")
        writer.seal_ok()
    return root


def _replace_bytes(path: Path, payload: bytes, mode: int = 0o444) -> None:
    path.chmod(0o600)
    path.write_bytes(payload)
    path.chmod(mode)


def _sample_metrics(
    gap: int = 250_000_000,
    exceeded: bool = False,
    *,
    ordinal: int = 1,
    previous: int = 1_000_000_000,
    role_id: str = "1",
) -> dict[str, object]:
    target_wait = trajectory.TARGET_WAIT_NS_V2
    scheduled = previous + target_wait
    actual = previous + gap
    finish = actual + 1_000_000
    return {
        "role_id": role_id,
        "sample_ordinal": ordinal,
        "previous_finish_ns": previous,
        "scheduled_start_ns": scheduled,
        "actual_start_ns": actual,
        "finish_ns": finish,
        "collection_duration_ns": 1_000_000,
        "scheduler_lateness_ns": actual - scheduled,
        "target_wait_ns": target_wait,
        "gap_ns": gap,
        "maximum_gap_ns": trajectory.MAXIMUM_GAP_NS_V2,
        "gap_exceeded": exceeded,
    }


def test_fixed_clock_exact_byte_known_answer_and_complete_ok(private_parent: Path) -> None:
    root = private_parent / "kat-v2"
    ticks = iter((100, 101, 102))
    with trajectory.create_durable_trajectory_writer_v2(
        root,
        expected_parent=private_parent,
        trace_id="trace-1",
        producer_id="supervisor",
        clock_ns=lambda: next(ticks),
    ) as writer:
        writer.begin("span-1", "validate", stage="S1")
        writer.ok("span-1")
        writer.seal_ok()

    assert (root / "event-00000000000000000001-v2.json").read_bytes() == (
        b'{"actor_or_null":null,"actual_start_ns_or_null":null,"clock_id":"CLOCK_MONOTONIC",'
        b'"collection_duration_ns_or_null":null,"detail_or_null":null,"error_code_or_null":null,'
        b'"error_detail_or_null":null,"event_type":"BEGIN","finish_ns_or_null":null,'
        b'"gap_exceeded_or_null":null,"gap_ns_or_null":null,"maximum_gap_ns_or_null":null,'
        b'"operation":"validate","previous_finish_ns_or_null":null,'
        b'"previous_record_digest_or_null":null,"producer_id":"supervisor",'
        b'"record_digest":"sha256:3810f4e45bec8ca53942c4f0aa5e58c8dc1a5b431c8725140083166121aee906",'
        b'"role_id_or_null":null,"sample_ordinal_or_null":null,"scheduled_start_ns_or_null":null,'
        b'"scheduler_lateness_ns_or_null":null,'
        b'"schema_version":"hegel-phase3-q05b-durable-trajectory-event/2","sequence":1,'
        b'"span_id":"span-1","stage_or_null":"S1","substep_or_null":null,'
        b'"target_wait_ns_or_null":null,"timestamp_ns":100,"trace_id":"trace-1"}\n'
    )
    assert (root / "event-00000000000000000002-v2.json").read_bytes() == (
        b'{"actor_or_null":null,"actual_start_ns_or_null":null,"clock_id":"CLOCK_MONOTONIC",'
        b'"collection_duration_ns_or_null":null,"detail_or_null":null,"error_code_or_null":null,'
        b'"error_detail_or_null":null,"event_type":"OK","finish_ns_or_null":null,'
        b'"gap_exceeded_or_null":null,"gap_ns_or_null":null,"maximum_gap_ns_or_null":null,'
        b'"operation":"validate","previous_finish_ns_or_null":null,'
        b'"previous_record_digest_or_null":"sha256:3810f4e45bec8ca53942c4f0aa5e58c8dc1a5b431c8725140083166121aee906",'
        b'"producer_id":"supervisor",'
        b'"record_digest":"sha256:e0ee8c879cc6cc8d55d4c643e18e42198fd12005561c32b5c48def47f6a31578",'
        b'"role_id_or_null":null,"sample_ordinal_or_null":null,"scheduled_start_ns_or_null":null,'
        b'"scheduler_lateness_ns_or_null":null,'
        b'"schema_version":"hegel-phase3-q05b-durable-trajectory-event/2","sequence":2,'
        b'"span_id":"span-1","stage_or_null":"S1","substep_or_null":null,'
        b'"target_wait_ns_or_null":null,"timestamp_ns":101,"trace_id":"trace-1"}\n'
    )
    assert (root / trajectory.TERMINAL_MANIFEST_BASENAME_V2).read_bytes() == (
        b'{"clock_id":"CLOCK_MONOTONIC","closed_span_count":1,"event_count":2,'
        b'"final_event_record_digest":"sha256:e0ee8c879cc6cc8d55d4c643e18e42198fd12005561c32b5c48def47f6a31578",'
        b'"first_event_record_digest":"sha256:3810f4e45bec8ca53942c4f0aa5e58c8dc1a5b431c8725140083166121aee906",'
        b'"manifest_digest":"sha256:806e8d394aa00de90e149dc0057d80ee32594a8f850b804028ce0ef677ab48c0",'
        b'"open_span_ids_in_begin_order":[],"outcome":"OK","producer_id":"supervisor",'
        b'"schema_version":"hegel-phase3-q05b-durable-trajectory-terminal-manifest/2",'
        b'"terminal_at_ns":102,"terminal_event_type":"OK","trace_id":"trace-1"}\n'
    )
    verified = trajectory.verify_durable_trajectory_v2(
        root, expected_parent=private_parent
    )
    assert verified.status == "COMPLETE_OK"
    assert verified.complete is True
    assert verified.event_count == 2
    assert verified.producer_id_or_null == "supervisor"


def test_fixed_absolute_parent_root_and_record_metadata(private_parent: Path) -> None:
    root = _complete_ok(private_parent)
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    for record in root.iterdir():
        metadata = record.stat(follow_symlinks=False)
        assert stat.S_ISREG(metadata.st_mode)
        assert stat.S_IMODE(metadata.st_mode) == 0o444
        assert metadata.st_nlink == 1

    with pytest.raises(trajectory.TrajectoryErrorV2):
        trajectory.create_durable_trajectory_writer_v2(
            Path("relative-root"),
            expected_parent=private_parent,
            trace_id="relative",
            producer_id="test",
        )
    with pytest.raises(trajectory.TrajectoryErrorV2):
        trajectory.create_durable_trajectory_writer_v2(
            private_parent / "wrong-parent",
            expected_parent=private_parent.parent,
            trace_id="wrong-parent",
            producer_id="test",
        )
    outside = private_parent / "real-parent"
    outside.mkdir(mode=0o700)
    alias = private_parent / "parent-alias"
    alias.symlink_to(outside, target_is_directory=True)
    with pytest.raises(trajectory.TrajectoryErrorV2):
        trajectory.create_durable_trajectory_writer_v2(
            alias / "trace",
            expected_parent=alias,
            trace_id="symlink-parent",
            producer_id="test",
        )


def test_incomplete_begin_prefix_is_explicit_and_strict_mode_rejects(
    private_parent: Path,
) -> None:
    writer = _writer(private_parent, "incomplete")
    writer.begin("stage-5", "continuous-sampling", actor="rust")
    writer.close()
    verified = trajectory.verify_durable_trajectory_v2(
        private_parent / "incomplete", expected_parent=private_parent
    )
    assert verified.status == "INCOMPLETE_PREFIX"
    assert verified.complete is False
    assert verified.open_span_ids_in_begin_order == ("stage-5",)
    assert len(verified.canonical_event_payloads_in_order) == 1
    assert verified.canonical_event_payloads_in_order[0].endswith(b"\n")
    assert verified.open_span_contexts_in_begin_order == (
        trajectory.VerifiedOpenSpanContextV2(
            begin_sequence=1,
            span_id="stage-5",
            operation="continuous-sampling",
            stage_or_null=None,
            substep_or_null=None,
            actor_or_null="rust",
            detail_or_null=None,
            begin_record_digest=verified.first_event_record_digest_or_null,
        ),
    )
    assert verified.failure_rows_in_sequence_order == ()
    with pytest.raises(trajectory.TrajectoryErrorV2) as caught:
        trajectory.verify_durable_trajectory_v2(
            private_parent / "incomplete",
            expected_parent=private_parent,
            allow_incomplete=False,
        )
    assert caught.value.code == trajectory.FAIL_STATE_V2


def test_complete_fail_persists_actor_and_over_250ms_gap_before_return(
    private_parent: Path,
) -> None:
    root = private_parent / "gap-fail"
    with _writer(private_parent, "gap-fail") as writer:
        writer.begin("trial", "dual-qualification", stage="ACTUAL")
        writer.begin(
            "rust-sampler",
            "LIVE_RESOURCE_SAMPLER",
            stage="STAGE_5",
            actor="RUST_ENDPOINT",
        )
        writer.begin(
            "rust-sample-1",
            "LIVE_RESOURCE_SAMPLE",
            stage="STAGE_5",
            substep="continuous-sampling",
            actor="RUST_ENDPOINT",
        )
        failure = writer.fail(
            "rust-sample-1",
            error_code=trajectory.GAP_EXCEEDED_ERROR_CODE_V2,
            error_detail="held actor sampling gap exceeded 250ms",
            **_sample_metrics(
                gap=250_000_001,
                exceeded=True,
                role_id="2",
            ),
        )
        # fail() is synchronous: immutable evidence is readable before the
        # caller raises its supervisor exception.
        failure_path = root / "event-00000000000000000004-v2.json"
        assert failure_path.exists()
        assert stat.S_IMODE(failure_path.stat().st_mode) == 0o444
        assert failure["role_id_or_null"] == "2"
        assert failure["gap_ns_or_null"] == 250_000_001
        assert failure["maximum_gap_ns_or_null"] == 250_000_000
        assert failure["gap_exceeded_or_null"] is True
        writer.seal_fail()

    verified = trajectory.verify_durable_trajectory_v2(
        root, expected_parent=private_parent
    )
    assert verified.status == "COMPLETE_FAIL"
    assert verified.open_span_ids_in_begin_order == ("trial", "rust-sampler")
    assert len(verified.canonical_event_payloads_in_order) == 4
    assert tuple(row.span_id for row in verified.open_span_contexts_in_begin_order) == (
        "trial",
        "rust-sampler",
    )
    assert len(verified.failure_rows_in_sequence_order) == 1
    failure_row = verified.failure_rows_in_sequence_order[0]
    assert (
        failure_row.sequence,
        failure_row.operation,
        failure_row.stage_or_null,
        failure_row.substep_or_null,
        failure_row.actor_or_null,
        failure_row.error_code,
        failure_row.gap_ns_or_null,
        failure_row.maximum_gap_ns_or_null,
    ) == (
        4,
        "LIVE_RESOURCE_SAMPLE",
        "STAGE_5",
        "continuous-sampling",
        "RUST_ENDPOINT",
        trajectory.GAP_EXCEEDED_ERROR_CODE_V2,
        250_000_001,
        250_000_000,
    )
    assert verified.terminal_error_code_or_null == (
        trajectory.GAP_EXCEEDED_ERROR_CODE_V2
    )
    assert verified.terminal_error_detail_or_null == (
        "held actor sampling gap exceeded 250ms"
    )


def test_exact_250ms_boundary_is_not_exceeded(private_parent: Path) -> None:
    assert trajectory.TARGET_WAIT_NS_V2 == 125_000_000
    root = private_parent / "boundary"
    with _writer(private_parent, "boundary") as writer:
        writer.begin(
            "python-sampler", "LIVE_RESOURCE_SAMPLER", actor="PYTHON_ENDPOINT"
        )
        writer.begin(
            "sample-1", "LIVE_RESOURCE_SAMPLE", actor="PYTHON_ENDPOINT"
        )
        boundary = writer.ok("sample-1", **_sample_metrics())
        writer.ok("python-sampler")
        writer.seal_ok()
    assert boundary["gap_ns_or_null"] == 250_000_000
    assert boundary["maximum_gap_ns_or_null"] == 250_000_000
    assert boundary["gap_exceeded_or_null"] is False
    assert (
        boundary["scheduled_start_ns_or_null"]
        - boundary["previous_finish_ns_or_null"]
        == 125_000_000
    )
    assert trajectory.verify_durable_trajectory_v2(
        root, expected_parent=private_parent
    ).status == "COMPLETE_OK"


def test_uint64_monotonic_clock_values_above_ijson_limit_are_exact(
    private_parent: Path,
) -> None:
    root = private_parent / "uint64-clock"
    maximum = (1 << 64) - 1
    ticks = iter((maximum - 2, maximum - 1, maximum))
    with trajectory.create_durable_trajectory_writer_v2(
        root,
        expected_parent=private_parent,
        trace_id="uint64-clock",
        producer_id="supervisor",
        clock_ns=lambda: next(ticks),
    ) as writer:
        writer.begin("root", "trial")
        writer.ok("root")
        writer.seal_ok()
    first = json.loads(
        (root / "event-00000000000000000001-v2.json").read_bytes()
    )
    second = json.loads(
        (root / "event-00000000000000000002-v2.json").read_bytes()
    )
    manifest = json.loads(
        (root / trajectory.TERMINAL_MANIFEST_BASENAME_V2).read_bytes()
    )
    assert first["timestamp_ns"] == maximum - 2
    assert second["timestamp_ns"] == maximum - 1
    assert manifest["terminal_at_ns"] == maximum
    assert trajectory.verify_durable_trajectory_v2(
        root, expected_parent=private_parent
    ).status == "COMPLETE_OK"


def test_frozen_sample_policy_role_chain_and_exact_begin_terminal_duplicate(
    private_parent: Path,
) -> None:
    root = private_parent / "sample-chain"
    first = _sample_metrics(previous=(1 << 53) + 100_000_000)
    second = _sample_metrics(
        gap=125_000_000,
        ordinal=2,
        previous=int(first["finish_ns"]),
    )
    with _writer(
        private_parent,
        "sample-chain",
        clock=_ticks(int(second["finish_ns"]) + 1_000),
    ) as writer:
        writer.begin(
            "python-sampler", "LIVE_RESOURCE_SAMPLER", actor="PYTHON_ENDPOINT"
        )
        # Exact BEGIN/terminal duplication is deliberately one observation.
        writer.begin(
            "python-sample-1",
            "LIVE_RESOURCE_SAMPLE",
            actor="PYTHON_ENDPOINT",
            **first,
        )
        writer.ok("python-sample-1", **first)
        # The normal supervisor shape has metrics only on the terminal event.
        writer.begin(
            "python-sample-2",
            "LIVE_RESOURCE_SAMPLE",
            actor="PYTHON_ENDPOINT",
        )
        writer.ok("python-sample-2", **second)
        writer.ok("python-sampler")
        writer.seal_ok()
    verified = trajectory.verify_durable_trajectory_v2(
        root, expected_parent=private_parent
    )
    assert verified.status == "COMPLETE_OK"
    assert verified.event_count == 6


def test_sample_fail_before_finish_may_have_all_null_metrics(
    private_parent: Path,
) -> None:
    root = private_parent / "sample-fail-before-finish"
    with _writer(private_parent, "sample-fail-before-finish") as writer:
        writer.begin(
            "python-sampler", "LIVE_RESOURCE_SAMPLER", actor="PYTHON_ENDPOINT"
        )
        writer.begin(
            "python-sample-1",
            "LIVE_RESOURCE_SAMPLE",
            actor="PYTHON_ENDPOINT",
        )
        writer.fail(
            "python-sample-1",
            error_code="RESOURCE_COLLECTION_FAILED",
            error_detail="collection failed before a finish timestamp existed",
        )
        writer.seal_fail()
    assert trajectory.verify_durable_trajectory_v2(
        root, expected_parent=private_parent
    ).status == "COMPLETE_FAIL"


def test_begin_sample_metrics_require_exact_terminal_duplicate(
    private_parent: Path,
) -> None:
    writer = _writer(private_parent, "sample-duplicate-mismatch")
    metrics = _sample_metrics()
    writer.begin(
        "python-sampler", "LIVE_RESOURCE_SAMPLER", actor="PYTHON_ENDPOINT"
    )
    writer.begin(
        "sample-1",
        "LIVE_RESOURCE_SAMPLE",
        actor="PYTHON_ENDPOINT",
        **metrics,
    )
    with pytest.raises(trajectory.TrajectoryErrorV2) as caught:
        writer.fail(
            "sample-1",
            error_code="RESOURCE_COLLECTION_FAILED",
            error_detail="terminal omitted already-recorded metrics",
        )
    assert caught.value.code == trajectory.FAIL_CHAIN_V2
    assert writer.poisoned is True
    writer.close()


def test_sample_actor_mapping_enclosure_and_same_clock_are_mandatory(
    private_parent: Path,
) -> None:
    standalone = _writer(private_parent, "standalone-sample")
    with pytest.raises(trajectory.TrajectoryErrorV2) as missing_outer:
        standalone.begin(
            "sample-1", "LIVE_RESOURCE_SAMPLE", actor="PYTHON_ENDPOINT"
        )
    assert missing_outer.value.code == trajectory.FAIL_CHAIN_V2
    standalone.close()

    missing_actor = _writer(private_parent, "missing-sample-actor")
    missing_actor.begin(
        "python-sampler", "LIVE_RESOURCE_SAMPLER", actor="PYTHON_ENDPOINT"
    )
    with pytest.raises(trajectory.TrajectoryErrorV2) as absent:
        missing_actor.begin("sample-1", "LIVE_RESOURCE_SAMPLE")
    assert absent.value.code == trajectory.FAIL_SCHEMA_V2
    missing_actor.close()

    switched = _writer(private_parent, "switched-sample-actor")
    switched.begin(
        "rust-sampler", "LIVE_RESOURCE_SAMPLER", actor="RUST_ENDPOINT"
    )
    switched.begin("sample-1", "LIVE_RESOURCE_SAMPLE", actor="RUST_ENDPOINT")
    with pytest.raises(trajectory.TrajectoryErrorV2) as wrong_map:
        switched.ok("sample-1", **_sample_metrics(role_id="1"))
    assert wrong_map.value.code == trajectory.FAIL_SCHEMA_V2
    switched.close()

    early_clock = _writer(
        private_parent, "early-sample-clock", clock=_ticks(100)
    )
    early_clock.begin(
        "python-sampler", "LIVE_RESOURCE_SAMPLER", actor="PYTHON_ENDPOINT"
    )
    early_clock.begin(
        "sample-1", "LIVE_RESOURCE_SAMPLE", actor="PYTHON_ENDPOINT"
    )
    with pytest.raises(trajectory.TrajectoryErrorV2) as before_finish:
        early_clock.ok("sample-1", **_sample_metrics())
    assert before_finish.value.code == trajectory.FAIL_SCHEMA_V2
    early_clock.close()


def test_held_final_sample_requires_matching_open_actor_close(
    private_parent: Path,
) -> None:
    root = private_parent / "held-final-enclosure"
    with _writer(private_parent, "held-final-enclosure") as writer:
        writer.begin(
            "rust-close", "HELD_ACTOR_CLOSE", actor="RUST_ENDPOINT"
        )
        writer.begin(
            "rust-held-final-1",
            "HELD_FINAL_RESOURCE_SAMPLE",
            actor="RUST_ENDPOINT",
        )
        writer.ok("rust-held-final-1", **_sample_metrics(role_id="2"))
        writer.ok("rust-close")
        writer.seal_ok()
    assert trajectory.verify_durable_trajectory_v2(
        root, expected_parent=private_parent
    ).status == "COMPLETE_OK"


def test_ok_cannot_carry_exceeded_gap_and_gap_code_is_bidirectional(
    private_parent: Path,
) -> None:
    writer = _writer(private_parent, "ok-exceeded-gap")
    writer.begin(
        "python-sampler", "LIVE_RESOURCE_SAMPLER", actor="PYTHON_ENDPOINT"
    )
    writer.begin(
        "sample-1", "LIVE_RESOURCE_SAMPLE", actor="PYTHON_ENDPOINT"
    )
    with pytest.raises(trajectory.TrajectoryErrorV2) as exceeded_ok:
        writer.ok(
            "sample-1",
            **_sample_metrics(gap=250_000_001, exceeded=True),
        )
    assert exceeded_ok.value.code == trajectory.FAIL_SCHEMA_V2
    writer.close()

    wrong_code = _writer(private_parent, "wrong-gap-code")
    wrong_code.begin(
        "python-sampler", "LIVE_RESOURCE_SAMPLER", actor="PYTHON_ENDPOINT"
    )
    wrong_code.begin(
        "sample-1", "LIVE_RESOURCE_SAMPLE", actor="PYTHON_ENDPOINT"
    )
    with pytest.raises(trajectory.TrajectoryErrorV2) as nonexact:
        wrong_code.fail(
            "sample-1",
            error_code="SOME_OTHER_FAILURE",
            error_detail="gap exceeded under a noncanonical code",
            **_sample_metrics(gap=250_000_001, exceeded=True),
        )
    assert nonexact.value.code == trajectory.FAIL_SCHEMA_V2
    wrong_code.close()


def test_any_prior_fail_taints_trace_and_forbids_complete_ok(
    private_parent: Path,
) -> None:
    root = private_parent / "failure-taint"
    writer = _writer(private_parent, "failure-taint")
    writer.begin("bad", "first-step")
    writer.fail(
        "bad",
        error_code="FIRST_STEP_FAILED",
        error_detail="the first step failed",
    )
    writer.begin("later", "later-step")
    writer.ok("later")
    with pytest.raises(trajectory.TrajectoryErrorV2) as seal:
        writer.seal_ok()
    assert seal.value.code == trajectory.FAIL_STATE_V2
    writer.close()

    # Build the exact fraudulent manifest the old verifier accepted, with all
    # event-chain and manifest digests valid.  Producer-free replay must still
    # reject because one earlier FAIL taints the entire attempt.
    events = [
        json.loads((root / f"event-{index:020d}-v2.json").read_bytes())
        for index in range(1, 5)
    ]
    manifest: dict[str, object] = {
        "schema_version": trajectory.TERMINAL_MANIFEST_SCHEMA_V2,
        "trace_id": "trace-failure-taint",
        "producer_id": "q05b-supervisor-v2",
        "clock_id": "CLOCK_MONOTONIC",
        "outcome": "OK",
        "event_count": 4,
        "first_event_record_digest": events[0]["record_digest"],
        "final_event_record_digest": events[-1]["record_digest"],
        "terminal_event_type": "OK",
        "terminal_at_ns": int(events[-1]["timestamp_ns"]) + 1,
        "open_span_ids_in_begin_order": [],
        "closed_span_count": 2,
        "manifest_digest": None,
    }
    manifest["manifest_digest"] = trajectory._manifest_digest(manifest)  # noqa: SLF001
    path = root / trajectory.TERMINAL_MANIFEST_BASENAME_V2
    path.write_bytes(trajectory.canonical_trajectory_json_bytes_v2(manifest))
    path.chmod(0o444)
    with pytest.raises(trajectory.TrajectoryErrorV2) as replay:
        trajectory.verify_durable_trajectory_v2(root, expected_parent=private_parent)
    assert replay.value.code == trajectory.FAIL_CHAIN_V2


def test_nested_gap_code_propagation_without_parent_sample_metrics_completes_fail(
    private_parent: Path,
) -> None:
    root = private_parent / "nested-gap-propagation"
    with _writer(private_parent, "nested-gap-propagation") as writer:
        writer.begin("run", "DUAL_QUALIFICATION_RUN")
        writer.begin("stage", "ACTUAL_STAGE")
        writer.begin(
            "rust-sampler", "LIVE_RESOURCE_SAMPLER", actor="RUST_ENDPOINT"
        )
        writer.begin(
            "rust-sample-1",
            "LIVE_RESOURCE_SAMPLE",
            actor="RUST_ENDPOINT",
        )
        writer.fail(
            "rust-sample-1",
            error_code=trajectory.GAP_EXCEEDED_ERROR_CODE_V2,
            error_detail="rust sample gap exceeded",
            **_sample_metrics(
                gap=250_000_001,
                exceeded=True,
                role_id="2",
            ),
        )
        for span in ("rust-sampler", "stage", "run"):
            writer.fail(
                span,
                error_code=trajectory.GAP_EXCEEDED_ERROR_CODE_V2,
                error_detail=f"propagated from rust sample through {span}",
            )
        writer.seal_fail()
    verified = trajectory.verify_durable_trajectory_v2(
        root, expected_parent=private_parent
    )
    assert verified.status == "COMPLETE_FAIL"
    assert len(verified.failure_rows_in_sequence_order) == 4
    assert verified.failure_rows_in_sequence_order[0].gap_exceeded_or_null is True
    assert all(
        row.gap_exceeded_or_null is None
        for row in verified.failure_rows_in_sequence_order[1:]
    )
    assert verified.open_span_contexts_in_begin_order == ()


def test_online_append_guard_is_constant_readback_work_at_404_event_scale(
    private_parent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = private_parent / "scale-404"
    writer = _writer(private_parent, "scale-404", clock=_ticks(20_000_000_000))
    real_read = trajectory._read_secure_file  # noqa: SLF001
    readback_calls = 0

    def count_readback(root_fd: int, basename: str) -> bytes:
        nonlocal readback_calls
        readback_calls += 1
        return real_read(root_fd, basename)

    with monkeypatch.context() as scoped:
        scoped.setattr(trajectory, "_read_secure_file", count_readback)
        for index in range(202):
            span = f"step-{index:03d}"
            writer.begin(span, "SCALE_STEP")
            writer.ok(span)
        writer.seal_ok()
    assert writer.event_count == 404
    # One readback of the just-published file plus, after the first event,
    # one readback of only the latest committed event.  This bound is linear;
    # the former implementation performed ~N^2/2 historical rereads.
    assert readback_calls <= writer.event_count * 2 + 2
    writer.close()
    assert trajectory.verify_durable_trajectory_v2(
        root, expected_parent=private_parent
    ).status == "COMPLETE_OK"


def test_constant_time_online_guard_still_detects_namespace_generation_change(
    private_parent: Path,
) -> None:
    root = private_parent / "namespace-generation"
    writer = _writer(private_parent, "namespace-generation")
    writer.begin("root", "trial")
    extra = root / "external-entry"
    extra.write_bytes(b"unexpected\n")
    extra.chmod(0o444)
    with pytest.raises(trajectory.TrajectoryErrorV2) as changed:
        writer.ok("root")
    assert changed.value.code == trajectory.FAIL_FILESYSTEM_V2
    assert writer.poisoned is True
    writer.close()


def _complete_two_sample_trace(parent: Path, name: str) -> Path:
    root = parent / name
    first = _sample_metrics()
    second = _sample_metrics(
        gap=125_000_000,
        ordinal=2,
        previous=int(first["finish_ns"]),
    )
    with _writer(parent, name) as writer:
        writer.begin(
            "python-sampler", "LIVE_RESOURCE_SAMPLER", actor="PYTHON_ENDPOINT"
        )
        writer.begin(
            "sample-1", "LIVE_RESOURCE_SAMPLE", actor="PYTHON_ENDPOINT"
        )
        writer.ok("sample-1", **first)
        writer.begin(
            "sample-2", "LIVE_RESOURCE_SAMPLE", actor="PYTHON_ENDPOINT"
        )
        writer.ok("sample-2", **second)
        writer.ok("python-sampler")
        writer.seal_ok()
    return root


def _rewrite_final_event_and_manifest(
    root: Path, updates: dict[str, object]
) -> None:
    manifest_path = root / trajectory.TERMINAL_MANIFEST_BASENAME_V2
    manifest = json.loads(manifest_path.read_bytes())
    prior_digest: str | None = None
    final_digest: str | None = None
    for sequence in range(1, int(manifest["event_count"]) + 1):
        event_path = root / f"event-{sequence:020d}-v2.json"
        event = json.loads(event_path.read_bytes())
        if sequence == 5:
            event.update(updates)
        if sequence >= 5:
            event["previous_record_digest_or_null"] = prior_digest
            event["record_digest"] = trajectory._event_digest(event)  # noqa: SLF001
            _replace_bytes(
                event_path, trajectory.canonical_trajectory_json_bytes_v2(event)
            )
        prior_digest = str(event["record_digest"])
        final_digest = prior_digest
    assert final_digest is not None
    manifest["final_event_record_digest"] = final_digest
    manifest["manifest_digest"] = trajectory._manifest_digest(manifest)  # noqa: SLF001
    _replace_bytes(
        manifest_path, trajectory.canonical_trajectory_json_bytes_v2(manifest)
    )


@pytest.mark.parametrize(
    "mutation",
    ["weaken-policy", "duplicate-ordinal", "break-chain", "ok-exceeded-gap"],
)
def test_producer_free_verifier_rejects_sample_policy_and_chain_weakening(
    private_parent: Path, mutation: str
) -> None:
    root = _complete_two_sample_trace(private_parent, f"replay-{mutation}")
    final = json.loads(
        (root / "event-00000000000000000005-v2.json").read_bytes()
    )
    if mutation == "weaken-policy":
        previous = int(final["previous_finish_ns_or_null"])
        actual = int(final["actual_start_ns_or_null"])
        _rewrite_final_event_and_manifest(
            root,
            {
                "target_wait_ns_or_null": 1,
                "scheduled_start_ns_or_null": previous + 1,
                "scheduler_lateness_ns_or_null": actual - previous - 1,
                "maximum_gap_ns_or_null": 1_000_000_000,
                "gap_exceeded_or_null": False,
            },
        )
    elif mutation == "duplicate-ordinal":
        _rewrite_final_event_and_manifest(root, {"sample_ordinal_or_null": 1})
    elif mutation == "break-chain":
        previous = int(final["previous_finish_ns_or_null"]) + 1
        actual = previous + 125_000_000
        finish = actual + 1_000_000
        _rewrite_final_event_and_manifest(
            root,
            {
                "previous_finish_ns_or_null": previous,
                "scheduled_start_ns_or_null": previous + 125_000_000,
                "actual_start_ns_or_null": actual,
                "finish_ns_or_null": finish,
                "scheduler_lateness_ns_or_null": 0,
                "gap_ns_or_null": 125_000_000,
                "collection_duration_ns_or_null": 1_000_000,
            },
        )
    else:
        previous = int(final["previous_finish_ns_or_null"])
        actual = previous + 250_000_001
        _rewrite_final_event_and_manifest(
            root,
            {
                "actual_start_ns_or_null": actual,
                "finish_ns_or_null": actual + 1_000_000,
                "scheduler_lateness_ns_or_null": 200_000_001,
                "gap_ns_or_null": 250_000_001,
                "gap_exceeded_or_null": True,
            },
        )
    with pytest.raises(trajectory.TrajectoryErrorV2):
        trajectory.verify_durable_trajectory_v2(root, expected_parent=private_parent)


@pytest.mark.parametrize(
    "override",
    [
        {"gap_exceeded": True},
        {"gap_ns": True},
        {"scheduler_lateness_ns": 124_999_999},
        {"collection_duration_ns": 999_999},
        {"target_wait_ns": 124_999_999},
        {"role_id": None},
    ],
)
def test_sample_metric_field_and_250ms_boundary_errors_fail_closed(
    private_parent: Path, override: dict[str, object]
) -> None:
    name = f"invalid-{len(list(private_parent.iterdir()))}"
    writer = _writer(private_parent, name)
    writer.begin(
        "python-sampler", "LIVE_RESOURCE_SAMPLER", actor="PYTHON_ENDPOINT"
    )
    writer.begin(
        "sample", "LIVE_RESOURCE_SAMPLE", actor="PYTHON_ENDPOINT"
    )
    values = _sample_metrics()
    values.update(override)
    with pytest.raises(trajectory.TrajectoryErrorV2) as caught:
        writer.ok("sample", **values)
    assert caught.value.code == trajectory.FAIL_SCHEMA_V2
    assert writer.poisoned is True
    with pytest.raises(trajectory.TrajectoryErrorV2) as second:
        writer.ok("sample")
    assert second.value.code == trajectory.FAIL_STATE_V2
    writer.close()


@pytest.mark.parametrize("fault", ["write", "fsync"])
def test_write_and_fsync_faults_permanently_poison_writer(
    private_parent: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    writer = _writer(private_parent, f"fault-{fault}")

    def injected(*_arguments: object, **_keywords: object) -> int:
        raise OSError(f"injected {fault} failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(trajectory.os, fault, injected)
        with pytest.raises(trajectory.TrajectoryErrorV2) as caught:
            writer.begin("root", "trial")
        assert caught.value.code == trajectory.FAIL_FILESYSTEM_V2
    assert writer.poisoned is True
    with pytest.raises(trajectory.TrajectoryErrorV2) as second:
        writer.begin("later", "forbidden-retry")
    assert second.value.code == trajectory.FAIL_STATE_V2
    writer.close()
    verified = trajectory.verify_durable_trajectory_v2(
        private_parent / f"fault-{fault}", expected_parent=private_parent
    )
    assert verified.status == "INCOMPLETE_TORN_TAIL"
    assert verified.event_count == 0
    assert verified.torn_tail_staging_basename_or_null == (
        ".event-00000000000000000001-v2.json.staging-v2"
    )


def test_publication_uses_exact_single_file_fsync_sequence(
    private_parent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _writer(private_parent, "publication-sequence")
    root_fd = writer._root_fd
    operations: list[str] = []
    real_write = os.write
    real_fchmod = os.fchmod
    real_fsync = os.fsync
    real_link = os.link
    real_unlink = os.unlink

    def record_write(descriptor: int, payload: object) -> int:
        operations.append("write")
        return real_write(descriptor, payload)

    def record_fchmod(descriptor: int, mode: int) -> None:
        operations.append(f"fchmod:{mode:o}")
        real_fchmod(descriptor, mode)

    def record_fsync(descriptor: int) -> None:
        operations.append("fsync:directory" if descriptor == root_fd else "fsync:file")
        real_fsync(descriptor)

    def record_link(*args: object, **kwargs: object) -> None:
        operations.append("link")
        real_link(*args, **kwargs)

    def record_unlink(*args: object, **kwargs: object) -> None:
        operations.append("unlink")
        real_unlink(*args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(trajectory.os, "write", record_write)
        scoped.setattr(trajectory.os, "fchmod", record_fchmod)
        scoped.setattr(trajectory.os, "fsync", record_fsync)
        scoped.setattr(trajectory.os, "link", record_link)
        scoped.setattr(trajectory.os, "unlink", record_unlink)
        writer.begin("root", "trial")
    writer.close()

    assert operations == [
        "write",
        "fchmod:444",
        "fsync:file",
        "fsync:directory",
        "link",
        "fsync:directory",
        "unlink",
        "fsync:directory",
    ]


def test_torn_event_preserves_preceding_durable_prefix(
    private_parent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = private_parent / "torn-after-prefix"
    writer = _writer(private_parent, "torn-after-prefix")
    writer.begin("root", "trial")
    committed = writer.ok("root")

    def fail_write(*_arguments: object, **_keywords: object) -> int:
        raise OSError("injected tail write failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(trajectory.os, "write", fail_write)
        with pytest.raises(trajectory.TrajectoryErrorV2):
            writer.begin("next", "next-step")
    writer.close()
    verified = trajectory.verify_durable_trajectory_v2(
        root, expected_parent=private_parent
    )
    assert verified.status == "INCOMPLETE_TORN_TAIL"
    assert verified.event_count == 2
    assert verified.final_event_record_digest_or_null == committed["record_digest"]
    with pytest.raises(trajectory.TrajectoryErrorV2):
        trajectory.verify_durable_trajectory_v2(
            root, expected_parent=private_parent, allow_incomplete=False
        )


def test_post_link_fsync_fault_is_recognized_as_two_link_torn_tail(
    private_parent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = private_parent / "post-link-fault"
    writer = _writer(private_parent, "post-link-fault")
    real_fsync = os.fsync
    calls = 0

    def fail_third_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected post-link directory fsync failure")
        real_fsync(descriptor)

    with monkeypatch.context() as scoped:
        scoped.setattr(trajectory.os, "fsync", fail_third_fsync)
        with pytest.raises(trajectory.TrajectoryErrorV2):
            writer.begin("root", "trial")
    writer.close()
    staging = root / ".event-00000000000000000001-v2.json.staging-v2"
    final = root / "event-00000000000000000001-v2.json"
    assert staging.stat().st_ino == final.stat().st_ino
    assert staging.stat().st_nlink == final.stat().st_nlink == 2
    verified = trajectory.verify_durable_trajectory_v2(
        root, expected_parent=private_parent
    )
    assert verified.status == "INCOMPLETE_TORN_TAIL"
    assert verified.event_count == 0


def test_torn_terminal_manifest_preserves_all_committed_events(
    private_parent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = private_parent / "torn-manifest"
    writer = _writer(private_parent, "torn-manifest")
    writer.begin("root", "trial")
    final_event = writer.ok("root")

    def fail_write(*_arguments: object, **_keywords: object) -> int:
        raise OSError("injected terminal-manifest write failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(trajectory.os, "write", fail_write)
        with pytest.raises(trajectory.TrajectoryErrorV2):
            writer.seal_ok()
    writer.close()
    verified = trajectory.verify_durable_trajectory_v2(
        root, expected_parent=private_parent
    )
    assert verified.status == "INCOMPLETE_TORN_TAIL"
    assert verified.event_count == 2
    assert verified.final_event_record_digest_or_null == final_event["record_digest"]
    assert verified.torn_tail_staging_basename_or_null == (
        ".terminal-manifest-v2.json.staging-v2"
    )


def test_torn_staging_external_hardlink_is_not_ignored(
    private_parent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = private_parent / "unsafe-torn"
    writer = _writer(private_parent, "unsafe-torn")

    def fail_write(*_arguments: object, **_keywords: object) -> int:
        raise OSError("injected write failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(trajectory.os, "write", fail_write)
        with pytest.raises(trajectory.TrajectoryErrorV2):
            writer.begin("root", "trial")
    writer.close()
    staging = root / ".event-00000000000000000001-v2.json.staging-v2"
    os.link(staging, private_parent / "external-staging-alias")
    with pytest.raises(trajectory.TrajectoryErrorV2):
        trajectory.verify_durable_trajectory_v2(root, expected_parent=private_parent)


def test_concurrent_producers_receive_one_contiguous_hash_chain(
    private_parent: Path,
) -> None:
    count = 12
    with _writer(private_parent, "concurrent", clock=_ticks(10_000)) as writer:
        def produce(index: int) -> None:
            span = f"worker-{index:02d}"
            writer.begin(span, "concurrent-step", actor=span)
            writer.ok(span)

        with ThreadPoolExecutor(max_workers=count) as executor:
            list(executor.map(produce, range(count)))
        writer.seal_ok()
    verified = trajectory.verify_durable_trajectory_v2(
        private_parent / "concurrent", expected_parent=private_parent
    )
    assert verified.status == "COMPLETE_OK"
    assert verified.event_count == count * 2
    assert verified.final_sequence == count * 2


@pytest.mark.parametrize("mutation", ["tamper", "reorder", "drop", "duplicate"])
def test_verifier_rejects_tamper_reorder_drop_and_duplicate(
    private_parent: Path, mutation: str
) -> None:
    root = _complete_ok(private_parent, f"chain-{mutation}")
    first = root / "event-00000000000000000001-v2.json"
    second = root / "event-00000000000000000002-v2.json"
    if mutation == "tamper":
        payload = first.read_bytes().replace(b'"stage_or_null":"VALIDATE"', b'"stage_or_null":"TAMPERED"')
        _replace_bytes(first, payload)
    elif mutation == "reorder":
        one, two = first.read_bytes(), second.read_bytes()
        _replace_bytes(first, two)
        _replace_bytes(second, one)
    elif mutation == "drop":
        first.unlink()
    else:
        duplicate = root / "event-00000000000000000003-v2.json"
        duplicate.write_bytes(first.read_bytes())
        duplicate.chmod(0o444)
    with pytest.raises(trajectory.TrajectoryErrorV2):
        trajectory.verify_durable_trajectory_v2(root, expected_parent=private_parent)


@pytest.mark.parametrize("mutation", ["symlink", "hardlink", "mode", "field"])
def test_verifier_rejects_symlink_hardlink_mode_and_field_errors(
    private_parent: Path, mutation: str
) -> None:
    root = _complete_ok(private_parent, f"metadata-{mutation}")
    first = root / "event-00000000000000000001-v2.json"
    if mutation == "symlink":
        outside = private_parent / "outside-event.json"
        outside.write_bytes(first.read_bytes())
        outside.chmod(0o444)
        first.unlink()
        first.symlink_to(outside)
    elif mutation == "hardlink":
        os.link(first, private_parent / "outside-hardlink.json")
    elif mutation == "mode":
        first.chmod(0o640)
    else:
        decoded = json.loads(first.read_bytes())
        decoded["clock_id"] = "CLOCK_REALTIME"
        decoded["record_digest"] = trajectory._event_digest(decoded)  # noqa: SLF001
        _replace_bytes(first, trajectory.canonical_trajectory_json_bytes_v2(decoded))
    with pytest.raises(trajectory.TrajectoryErrorV2):
        trajectory.verify_durable_trajectory_v2(root, expected_parent=private_parent)


def test_terminal_manifest_tamper_and_extra_namespace_are_rejected(
    private_parent: Path,
) -> None:
    root = _complete_ok(private_parent, "manifest-tamper")
    manifest = root / trajectory.TERMINAL_MANIFEST_BASENAME_V2
    decoded = json.loads(manifest.read_bytes())
    decoded["closed_span_count"] = 0
    decoded["manifest_digest"] = trajectory._manifest_digest(decoded)  # noqa: SLF001
    _replace_bytes(manifest, trajectory.canonical_trajectory_json_bytes_v2(decoded))
    with pytest.raises(trajectory.TrajectoryErrorV2):
        trajectory.verify_durable_trajectory_v2(root, expected_parent=private_parent)

    second = _complete_ok(private_parent, "extra-entry")
    extra = second / "unregistered.log"
    extra.write_bytes(b"forbidden\n")
    extra.chmod(0o444)
    with pytest.raises(trajectory.TrajectoryErrorV2):
        trajectory.verify_durable_trajectory_v2(second, expected_parent=private_parent)
