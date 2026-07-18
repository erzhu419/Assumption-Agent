from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from assumption_agent.benchmarks import feverous_nli_runtime_v1 as runtime
from replication_runtime.qasc_nli_v1.contract import encode_request
from replication_runtime.qasc_nli_v1.worker import canonical_canary_pairs


PROJECT = Path(__file__).parents[1]


def test_design_binds_exact_feverous_eight_worker_profile() -> None:
    receipt = runtime.verify_feverous_design(PROJECT)
    assert receipt["design_sha256"] == runtime.DESIGN_SHA256
    assert receipt["NLI_worker_processes"] == 8


def test_canary_requires_every_worker_twice_and_exact_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = (11, -7, 0)
    monkeypatch.setattr(
        runtime,
        "CANARY_SCORE_VECTOR_SHA256",
        runtime.stable_hash(list(vector)),
    )
    receipt = runtime._verify_canary_vectors(
        (vector,) * 16,
        worker_count=8,
    )
    assert receipt["per_worker_startup_repeat_exact"] is True
    with pytest.raises(runtime.FeverousNLIRuntimeError):
        runtime._verify_canary_vectors((vector,) * 15, worker_count=8)
    with pytest.raises(runtime.FeverousNLIRuntimeError):
        runtime._verify_canary_vectors(
            (*((vector,) * 15), (11, -6, 0)), worker_count=8
        )


def test_worker_wire_has_pairs_only() -> None:
    raw = encode_request(canonical_canary_pairs())
    assert b"family" not in raw
    assert b"evidence" not in raw
    assert b"item" not in raw
    assert b"label" not in raw


def test_pool_receipt_rejects_consistently_rehashed_wrong_canary() -> None:
    design = runtime.verify_feverous_design(PROJECT)
    runtime_payload = {
        "asset_sha256": runtime.qasc_binding.ASSET_SELF_SHA256,
        "status": "verified_offline_immutable_runtime",
    }
    canary = {
        "integer_score_vector_sha256": runtime.CANARY_SCORE_VECTOR_SHA256,
        "per_worker_startup_repeat_exact": True,
    }
    body = {
        "schema": f"{runtime.VERSION}_receipt",
        "version": runtime.VERSION,
        "design": dict(design),
        "runtime": runtime_payload,
        "canary": canary,
        "worker_count": 8,
        "network_calls": 0,
        "online_evaluator_calls": 0,
    }
    receipt = runtime.FeverousNLIPoolReceipt(
        design=design,
        runtime=runtime_payload,
        canary=canary,
        receipt_sha256=runtime.stable_hash(body),
    )
    assert runtime.verify_pool_receipt(receipt) == receipt.receipt_sha256

    forged_canary = dict(canary)
    forged_canary["integer_score_vector_sha256"] = "0" * 64
    forged_body = {**body, "canary": forged_canary}
    forged = replace(
        receipt,
        canary=forged_canary,
        receipt_sha256=runtime.stable_hash(forged_body),
    )
    with pytest.raises(runtime.FeverousNLIRuntimeError):
        runtime.verify_pool_receipt(forged)
