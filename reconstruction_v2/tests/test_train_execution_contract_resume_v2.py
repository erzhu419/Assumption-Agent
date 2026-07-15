from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from assumption_agent.events import Event
from assumption_agent.models import stable_hash
from assumption_agent.benchmarks.execution_contract_prompt_v2 import (
    ExecutionContractPromptInjectionReceiptV2,
)
from assumption_agent.benchmarks.train_execution_contract_resume_v2 import (
    TrainExecutionContractResumeError,
    _prompt_receipt_from_event,
    _verified_event_ledger,
)


def _digest(label: str) -> str:
    return stable_hash({"label": label})


def test_event_recovery_ledger_requires_exact_event_envelopes(
    tmp_path: Path,
) -> None:
    row = Event(
        event="example",
        stage="test",
        trace_id="trace-1",
        payload={"request_hash": _digest("request")},
    ).to_dict()
    path = tmp_path / "events.jsonl"
    raw = json.dumps(row, sort_keys=True) + "\n"
    path.write_text(raw, encoding="utf-8")

    rows, ledger_sha256 = _verified_event_ledger(path)
    assert rows == (row,)
    assert ledger_sha256 == hashlib.sha256(raw.encode()).hexdigest()

    row["payload_hash"] = _digest("tampered")
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(
        TrainExecutionContractResumeError,
        match="integrity check failed",
    ):
        _verified_event_ledger(path)


def test_prompt_receipt_is_reconstructed_without_prompt_content() -> None:
    receipt = ExecutionContractPromptInjectionReceiptV2(
        capsule_hash=_digest("capsule"),
        request_hash=_digest("request"),
        base_runtime_context_hash=_digest("context"),
        source_receipt_hash=_digest("source"),
        typed_binding_set_hash=_digest("bindings"),
        public_instruction_hash=_digest("instruction"),
        bundle_manifest_hash=_digest("bundle"),
        profile_set_hash=_digest("profiles"),
        profile_count=1,
        effect_receipt_hashes=(_digest("effect"),),
        profile_output_sha256s=(_digest("profile-output"),),
        contract_set_hash=_digest("contracts"),
        contract_hashes=(_digest("contract"),),
        profile_contract_binding_set_hash=_digest("profile-bindings"),
        profile_contract_binding_hashes=(_digest("profile-binding"),),
        fragment_sha256=_digest("fragment"),
        fragment_size=97,
        container_path_hash=_digest("container-path"),
        container_readback_sha256=_digest("readback"),
        run_template_before_hash=_digest("template-before"),
        run_template_after_hash=_digest("template-after"),
        effective_prompt_sha256=_digest("effective-prompt"),
    )
    payload = {
        **receipt.safe_payload(),
        "receipt_hash": receipt.receipt_hash,
    }

    assert _prompt_receipt_from_event(payload) == receipt

    payload["fragment_size"] = 98
    with pytest.raises(
        TrainExecutionContractResumeError,
        match="failed reconstruction",
    ):
        _prompt_receipt_from_event(payload)
