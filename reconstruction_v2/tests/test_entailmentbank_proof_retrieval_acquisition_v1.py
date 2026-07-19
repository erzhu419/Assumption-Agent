from __future__ import annotations

import json
from pathlib import Path
import stat
import tempfile
from typing import Any

import pytest

from assumption_agent.benchmarks import (
    entailmentbank_proof_retrieval_acquisition_v1 as acquisition,
)


def _row(
    item_id: str,
    leaf_count: int,
    *,
    question: str | None = None,
    hypothesis: str | None = None,
) -> dict[str, Any]:
    triples = {
        f"sent{index}": f"exact private fact {item_id} {index}"
        for index in range(1, 26)
    }
    leaves = " & ".join(f"sent{index}" for index in range(1, leaf_count + 1))
    return {
        "id": item_id,
        "question": question or f"private question {item_id}",
        "answer": f"private answer {item_id}",
        "hypothesis": hypothesis or f"private hypothesis {item_id}",
        "proof": f"{leaves} -> hypothesis;",
        "meta": {
            "triples": triples,
            "intermediate_conclusions": {},
            "distractors": [
                f"sent{index}" for index in range(leaf_count + 1, 26)
            ],
        },
    }


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(
            row,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=False,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
        for row in rows
    )


def _candidate(split: str, item_id: str, leaf_count: int, **kwargs: str):
    parsed, audit = acquisition.parse_source(
        _jsonl([_row(item_id, leaf_count, **kwargs)]), split=split
    )
    assert audit["candidate_count"] == 1
    return parsed[0]


def _pool() -> tuple[tuple[acquisition.Candidate, ...], tuple[acquisition.Candidate, ...]]:
    train_rows = []
    dev_rows = []
    for leaf_count in (2, 3, 4):
        for index in range(55):
            train_rows.append(_row(f"train-{leaf_count}-{index}", leaf_count))
        for index in range(12):
            dev_rows.append(_row(f"dev-{leaf_count}-{index}", leaf_count))
    train, _ = acquisition.parse_source(_jsonl(train_rows), split="train")
    dev, _ = acquisition.parse_source(_jsonl(dev_rows), split="dev")
    return train, dev


def test_strict_reader_preserves_triple_insertion_order_and_proof_leaf_ordinals() -> None:
    row = _row("item-one", 3)
    row["meta"]["triples"] = {
        f"sent{index}": f"text {index}" for index in range(25, 0, -1)
    }
    row["meta"]["distractors"] = [
        f"sent{index}" for index in range(25, 3, -1)
    ]
    parsed, audit = acquisition.parse_source(_jsonl([row]), split="train")
    assert audit == {
        "line_count": 1,
        "candidate_count": 1,
        "formal_ineligible_count": 0,
    }
    candidate = parsed[0]
    assert candidate.triples[0] == ("sent25", "text 25")
    train, dev = _pool()
    selected, _ = acquisition.select_blocks(train, dev, secret=b"o" * 32)
    mutable = {block: list(rows) for block, rows in selected.items()}
    replacement_ordinal = next(
        index
        for index, row_value in enumerate(mutable["G_form"])
        if row_value.family == "THREE_LEAF"
    )
    mutable["G_form"][replacement_ordinal] = candidate
    payloads = acquisition.build_private_pack_payloads(
        {block: tuple(rows) for block, rows in mutable.items()}
    )
    labels = payloads["G_form.labels.private.json"]["items"]
    assert labels[replacement_ordinal]["gold_ordinals"] == [22, 23, 24]


def test_component_graph_excludes_cross_split_and_documentation_components() -> None:
    train = (
        _candidate("train", "shared-train", 2, question="same normalized question"),
        _candidate("train", acquisition.DOCUMENTATION_EXAMPLE_ID, 3),
        _candidate("train", "clean-train", 4),
    )
    dev = (
        _candidate("dev", "shared-dev", 3, question="  SAME normalized   question "),
        _candidate("dev", "clean-dev", 2),
    )
    clean, audit = acquisition.build_clean_components(train, dev)
    assert len(clean["train"]) == len(clean["dev"]) == 1
    assert audit["cross_split_component_count"] == 1
    assert audit["documentation_example_component_count"] == 1
    assert audit["clean_candidate_counts"]["train"]["FOUR_FIVE_LEAF"] == 1


def test_min_cost_flow_assigns_each_component_once_and_saturates_family_demands() -> None:
    shared_question = "one multi-family private component"
    candidates = (
        _candidate("train", "multi-two", 2, question=shared_question),
        _candidate("train", "multi-three", 3, question=shared_question),
        _candidate("train", "only-two", 2),
        _candidate("train", "only-three", 3),
        _candidate("train", "only-four", 4),
    )
    clean, _audit = acquisition.build_clean_components(candidates, ())
    assigned = acquisition.assign_components(
        clean["train"],
        {"TWO_LEAF": 1, "THREE_LEAF": 1, "FOUR_FIVE_LEAF": 1},
        secret=b"s" * 32,
    )
    tokens = [component.token for rows in assigned.values() for component in rows]
    assert len(tokens) == len(set(tokens)) == 3
    assert all(len(assigned[family]) == 1 for family in acquisition.FAMILY_ORDER)


def test_full_selection_is_balanced_deterministic_and_secret_keyed() -> None:
    train, dev = _pool()
    first, first_audit = acquisition.select_blocks(train, dev, secret=b"a" * 32)
    repeated, repeated_audit = acquisition.select_blocks(train, dev, secret=b"a" * 32)
    second, _ = acquisition.select_blocks(train, dev, secret=b"b" * 32)
    assert first == repeated
    assert first_audit == repeated_audit
    assert {
        block: len(rows) for block, rows in first.items()
    } == acquisition.BLOCK_COUNTS
    assert first_audit["selected_family_counts"] == acquisition.BLOCK_FAMILY_COUNTS
    selected_keys = [row.item_key for rows in first.values() for row in rows]
    assert len(selected_keys) == len(set(selected_keys)) == 186
    assert any(
        [row.item_key for row in first[block]]
        != [row.item_key for row in second[block]]
        for block in acquisition.BLOCK_ORDER
    )


def test_private_packs_exclude_labels_and_source_metadata_from_views() -> None:
    train, dev = _pool()
    blocks, _ = acquisition.select_blocks(train, dev, secret=b"p" * 32)
    payloads = acquisition.build_private_pack_payloads(blocks)
    assert len(payloads) == 9
    assert "F_search.labels.private.json" not in payloads
    for block in acquisition.BLOCK_ORDER:
        view = payloads[f"{block}.view.private.json"]
        assert acquisition.verify_self_hash(view, "pack_sha256")
        for row in view["items"]:
            assert set(row) == {
                "ordinal",
                "item_commitment_sha256",
                "question",
                "answer",
                "hypothesis",
                "node_texts",
            }
            assert len(row["node_texts"]) == 25
    with tempfile.TemporaryDirectory(prefix="entbank-acq-", dir=Path.home()) as raw_root:
        pack_root = Path(raw_root) / "private"
        hashes = acquisition.write_private_pack_payloads(pack_root, payloads)
        assert set(hashes) == set(payloads)
        assert stat.S_IMODE(pack_root.stat().st_mode) == 0o700
        assert all(
            stat.S_IMODE((pack_root / name).stat().st_mode) == 0o600
            for name in payloads
        )
        with pytest.raises(
            acquisition.EntailmentBankAcquisitionError, match="not pristine"
        ):
            acquisition.write_private_pack_payloads(pack_root, payloads)


def test_reader_fails_closed_on_unknown_proof_reference_and_skips_leaf_size() -> None:
    malformed = _row("bad", 2)
    malformed["proof"] = "sent1 & missing -> hypothesis;"
    with pytest.raises(acquisition.EntailmentBankAcquisitionError, match="unknown"):
        acquisition.parse_source(_jsonl([malformed]), split="train")
    ineligible = _row("one-leaf", 2)
    ineligible["proof"] = "sent1 -> hypothesis;"
    parsed, audit = acquisition.parse_source(_jsonl([ineligible]), split="train")
    assert parsed == ()
    assert audit["formal_ineligible_count"] == 1
