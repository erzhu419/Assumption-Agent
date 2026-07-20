from __future__ import annotations

import json
from pathlib import Path

import pytest

from assumption_agent.benchmarks import bright_reasoning_retrieval_core_v1 as core
from assumption_agent.benchmarks import bright_reasoning_retrieval_study_v1 as study


def _item() -> study.ViewItem:
    return study.ViewItem(
        ordinal=0,
        family="BIOLOGY",
        commitment="a" * 64,
        query="synthetic biology question",
        excluded_ids=(),
    )


def _action_pack() -> dict:
    candidates = [f"d{index}" for index in range(core.POOL_SIZE)]
    rows = list(range(core.POOL_SIZE))
    body = {
        "item_count": 1,
        "items": [
            {
                "candidate_document_ids": candidates,
                "candidate_rows": rows,
                "family": "BIOLOGY",
                "generation_valid": True,
                "item_commitment_sha256": "a" * 64,
                "ordinal": 0,
                "raw_document_ids": candidates[:10],
                "raw_rows": rows[:10],
                "recipe_document_ids": {
                    recipe: candidates[index : index + 10]
                    for index, recipe in enumerate(core.RECIPE_ORDER)
                },
                "recipe_rows": {
                    recipe: rows[index : index + 10]
                    for index, recipe in enumerate(core.RECIPE_ORDER)
                },
            }
        ],
        "recipe_order": list(core.RECIPE_ORDER),
        "schema": study.ACTION_SCHEMA,
    }
    return study.self_hashed(body, "pack_sha256")


def test_design_binding_and_action_pack_contract() -> None:
    design = Path(study.__file__).parents[2] / study.DESIGN_RELATIVE
    value = json.loads(design.read_text())
    assert study.file_sha256(design) == study.DESIGN_FILE_SHA256
    assert study.verify_self_hash(value, "self_sha256") == study.DESIGN_SELF_SHA256
    rows = study._validate_action_pack(_action_pack(), (_item(),))
    assert len(rows) == 1


def test_action_pack_rejects_candidate_escape() -> None:
    pack = _action_pack()
    body = dict(pack)
    body.pop("pack_sha256")
    body["items"][0]["recipe_document_ids"][core.RECIPE_ORDER[0]][0] = "outside"
    tampered = study.self_hashed(body, "pack_sha256")
    with pytest.raises(study.BrightStudyError, match="recipe action"):
        study._validate_action_pack(tampered, (_item(),))


def test_network_audit_allows_only_loopback_bind(tmp_path: Path) -> None:
    (tmp_path / "trace.1").write_text(
        'socket(AF_INET6, SOCK_STREAM, IPPROTO_IP) = 3\n'
        'bind(3, {sa_family=AF_INET6, inet_pton(AF_INET6, "::1", &sin6_addr)}, 28) = 0\n'
        '+++ exited with 0 +++\n',
        encoding="ascii",
    )
    receipt = study._network_trace_receipt(tmp_path, "trace.")
    assert receipt["loopback_bind_count"] == 1
    assert receipt["external_connect_syscall_count"] == 0
    (tmp_path / "bad.1").write_text(
        'connect(3, {sa_family=AF_INET, sin_port=htons(443)}, 16) = 0\n',
        encoding="ascii",
    )
    with pytest.raises(study.BrightStudyError, match="outbound"):
        study._network_trace_receipt(tmp_path, "bad.")


def test_concurrency_counter_tracks_peak_and_fails_underflow() -> None:
    counter = study._ConcurrencyCounter()
    counter.enter()
    counter.enter()
    assert counter.peak == 2
    counter.leave()
    counter.leave()
    with pytest.raises(study.BrightStudyError, match="underflow"):
        counter.leave()
