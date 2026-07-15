from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from assumption_agent.models import stable_hash
from replication_runtime.financial_semantic_v2.durable_state import (
    DurableStateError,
    atomic_write_hashed_json_v2,
    load_durable_stage_chain_v2,
    read_hashed_json_v2,
    transition_durable_stage_v2,
)
from replication_runtime.financial_semantic_v2.terminal_audit import (
    audit_codex_terminal_trace_v2,
)


def _trace(*event_types: str) -> str:
    return "\n".join(
        json.dumps({"type": event_type, "message": f"private-{index}"})
        for index, event_type in enumerate(event_types)
    )


def test_terminal_audit_recovers_only_preterminal_errors() -> None:
    clean = audit_codex_terminal_trace_v2(_trace("turn.completed"))
    recovered = audit_codex_terminal_trace_v2(
        _trace("error", "turn.started", "turn.completed")
    )

    assert clean.valid
    assert not clean.recovered_transient_error
    assert recovered.valid
    assert recovered.error_type is None
    assert recovered.error_before_terminal_count == 1
    assert recovered.error_after_terminal_count == 0
    assert recovered.recovered_transient_error
    assert "private" not in json.dumps(recovered.to_dict())
    body = recovered.to_dict()
    declared = body.pop("audit_hash")
    assert declared == stable_hash(body)


@pytest.mark.parametrize(
    ("events", "issue"),
    (
        (("turn.failed",), "codex_turn_failed_observed"),
        (
            ("turn.completed", "turn.completed"),
            "codex_multiple_terminal_events",
        ),
        (
            ("turn.completed", "error"),
            "codex_error_after_terminal",
        ),
        (("error",), "codex_turn_completed_missing"),
    ),
)
def test_terminal_audit_rejects_nonunique_or_missing_completion(
    events: tuple[str, ...],
    issue: str,
) -> None:
    audit = audit_codex_terminal_trace_v2(_trace(*events))

    assert not audit.valid
    assert issue in audit.issue_types
    assert not audit.recovered_transient_error


def test_durable_stage_chain_requires_exact_predecessor_and_order(
    tmp_path: Path,
) -> None:
    order = ("planned", "agent_completed", "verifier_completed")
    work_hash = "a" * 64
    request_hash = "b" * 64

    planned = transition_durable_stage_v2(
        tmp_path,
        stage_order=order,
        work_unit_hash=work_hash,
        request_hash=request_hash,
        stage="planned",
        predecessor_stage_hash=None,
        payload={"model_calls": 0},
    )
    with pytest.raises(DurableStateError, match="stale"):
        transition_durable_stage_v2(
            tmp_path,
            stage_order=order,
            work_unit_hash=work_hash,
            request_hash=request_hash,
            stage="agent_completed",
            predecessor_stage_hash="c" * 64,
            payload={},
        )
    with pytest.raises(DurableStateError, match="skipped"):
        transition_durable_stage_v2(
            tmp_path,
            stage_order=order,
            work_unit_hash=work_hash,
            request_hash=request_hash,
            stage="verifier_completed",
            predecessor_stage_hash=planned.stage_hash,
            payload={},
        )

    agent = transition_durable_stage_v2(
        tmp_path,
        stage_order=order,
        work_unit_hash=work_hash,
        request_hash=request_hash,
        stage="agent_completed",
        predecessor_stage_hash=planned.stage_hash,
        payload={"model_calls": 1},
    )
    verifier = transition_durable_stage_v2(
        tmp_path,
        stage_order=order,
        work_unit_hash=work_hash,
        request_hash=request_hash,
        stage="verifier_completed",
        predecessor_stage_hash=agent.stage_hash,
        payload={"offline": True},
    )
    chain = load_durable_stage_chain_v2(
        tmp_path,
        stage_order=order,
        work_unit_hash=work_hash,
        request_hash=request_hash,
    )

    assert [row.stage for row in chain] == list(order)
    assert chain[-1].stage_hash == verifier.stage_hash
    assert chain[1].predecessor_stage_hash == chain[0].stage_hash
    with pytest.raises(DurableStateError, match="skipped or repeated"):
        transition_durable_stage_v2(
            tmp_path,
            stage_order=order,
            work_unit_hash=work_hash,
            request_hash=request_hash,
            stage="verifier_completed",
            predecessor_stage_hash=agent.stage_hash,
            payload={},
        )


def test_atomic_hashed_json_fsyncs_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "receipt.json"
    calls: list[int] = []
    real_fsync = os.fsync

    def recording_fsync(descriptor: int) -> None:
        calls.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(
        "replication_runtime.financial_semantic_v2.durable_state.os.fsync",
        recording_fsync,
    )
    written = atomic_write_hashed_json_v2(
        target,
        {"stage": "planned"},
        hash_field="receipt_hash",
    )

    assert len(calls) >= 2
    assert read_hashed_json_v2(
        target,
        hash_field="receipt_hash",
    ) == written
    with pytest.raises(FileExistsError):
        atomic_write_hashed_json_v2(
            target,
            {"stage": "changed"},
            hash_field="receipt_hash",
        )


def test_durable_chain_rejects_tamper_and_gaps(tmp_path: Path) -> None:
    order = ("planned", "agent_completed", "verifier_completed")
    work_hash = "d" * 64
    request_hash = "e" * 64
    planned = transition_durable_stage_v2(
        tmp_path,
        stage_order=order,
        work_unit_hash=work_hash,
        request_hash=request_hash,
        stage="planned",
        predecessor_stage_hash=None,
        payload={},
    )
    agent = transition_durable_stage_v2(
        tmp_path,
        stage_order=order,
        work_unit_hash=work_hash,
        request_hash=request_hash,
        stage="agent_completed",
        predecessor_stage_hash=planned.stage_hash,
        payload={},
    )
    transition_durable_stage_v2(
        tmp_path,
        stage_order=order,
        work_unit_hash=work_hash,
        request_hash=request_hash,
        stage="verifier_completed",
        predecessor_stage_hash=agent.stage_hash,
        payload={},
    )

    agent_path = tmp_path / "001_agent_completed.stage.json"
    agent_path.unlink()
    with pytest.raises(DurableStateError, match="gap"):
        load_durable_stage_chain_v2(
            tmp_path,
            stage_order=order,
            work_unit_hash=work_hash,
            request_hash=request_hash,
        )
    # Restore a self-hash-invalid middle receipt to exercise tamper detection.
    agent_path.write_text(
        json.dumps({**agent.to_dict(), "payload": {"tampered": True}}),
        encoding="utf-8",
    )
    with pytest.raises(DurableStateError, match="self-hash"):
        load_durable_stage_chain_v2(
            tmp_path,
            stage_order=order,
            work_unit_hash=work_hash,
            request_hash=request_hash,
        )
