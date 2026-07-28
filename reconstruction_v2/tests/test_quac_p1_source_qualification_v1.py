from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from assumption_agent.benchmarks import quac_p1_source_qualification_v1 as q


TEST_QUOTAS = {"A_form": 1, "A_hold": 1, "M_search": 1}


def _paragraph(seed: str, *, title: str | None = None) -> dict[str, object]:
    tokens = [f"{seed}_token_{index}" for index in range(140)]
    context = " ".join(tokens)
    qas: list[dict[str, object]] = []
    for index, (token_index, followup) in enumerate(
        ((2, "y"), (20, "m"), (70, "n"), (139, "y"))
    ):
        text = tokens[token_index]
        qas.append(
            {
                "id": f"{seed}_private_id_{index}",
                "question": f"{seed} private question {index}",
                "followup": followup,
                "orig_answer": {
                    "answer_start": context.index(text),
                    "text": text,
                    "answer_end": context.index(text) + len(text),
                },
                # Extra official fields are accepted: this qualifier freezes
                # a required-key subset rather than guessing an exact keyset.
                "answers": [],
                "yesno": "x",
            }
        )
    return {
        "title": title or f"{seed}_private_title",
        "section_title": f"{seed}_private_section",
        "background": f"{seed}_private_background",
        "paragraphs": [{"context": context, "qas": qas}],
    }


def _payload(prefix: str, count: int) -> dict[str, object]:
    return {
        "version": "v0.2",
        "data": [_paragraph(f"{prefix}_{index}") for index in range(count)],
    }


def _fixture() -> tuple[dict[str, object], dict[str, object]]:
    # Global component capacity=1 needs three TRAIN components for the three
    # A_form families and six DEV components for A_hold/M across three families.
    return _payload("train_private", 3), _payload("dev_private", 6)


def _raw(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def _contract(train_raw: bytes, dev_raw: bytes) -> q.QualificationContract:
    return q.QualificationContract(
        train=q.SourceFileContract(
            len(train_raw), hashlib.sha256(train_raw).hexdigest()
        ),
        dev=q.SourceFileContract(
            len(dev_raw), hashlib.sha256(dev_raw).hexdigest()
        ),
        quotas=TEST_QUOTAS,
    )


def test_canonical_windows_stop_at_first_stride_aligned_tail() -> None:
    expected_counts = {
        95: 1,
        96: 1,
        97: 2,
        140: 2,
        145: 3,
    }
    for token_count, expected_count in expected_counts.items():
        context = " ".join(f"w{index}" for index in range(token_count))
        windows = q.canonical_window_spans(context)
        assert len(windows) == expected_count
        assert windows[0][0] == 0
        assert windows[-1][1] == len(context)


def test_capacity_flow_can_reverse_an_early_flexible_assignment() -> None:
    component_families = {
        0: {
            "train": {"FOLLOW", "MAYBE_FOLLOW"},
            "dev": set(),
        },
        1: {
            "train": {"FOLLOW"},
            "dev": set(),
        },
    }
    expected = q._deterministic_capacity_flow(
        component_families,
        TEST_QUOTAS,
    )
    assert expected[0] == 2
    assert expected[1]["A_form"]["FOLLOW"] == 1
    assert expected[1]["A_form"]["MAYBE_FOLLOW"] == 1
    assert (
        q._deterministic_capacity_flow(
            component_families,
            TEST_QUOTAS,
        )
        == expected
    )


def test_capacity_is_proved_by_global_component_max_flow_and_is_aggregate_only() -> None:
    train, dev = _fixture()
    result = q.qualify_decoded_sources(train, dev, quotas=TEST_QUOTAS)

    assert q.QREL_ROLE_ORDER == (
        "previous_turn_orig_answer",
        "current_turn_orig_answer",
    )
    assert q.QREL_FALLBACK_ALLOWED is False
    assert q.QREL_SAME_WINDOW_ALLOWED is True
    assert result["passed"] is True
    assert result["capacity_flow"] == {
        "component_global_capacity": 1,
        "required_flow": 9,
        "achieved_flow": 9,
        "aggregate_slack": 0,
        "slot_flow": {
            block: {family: 1 for family in q.FAMILY_ORDER}
            for block in q.PARTITION_ORDER
        },
        "slot_slack": {
            block: {family: 0 for family in q.FAMILY_ORDER}
            for block in q.PARTITION_ORDER
        },
        "all_nine_slots_saturated": True,
        "assignment_witness_output_count": 0,
    }
    assert result["source_aggregates"]["train"][
        "family_eligible_item_counts"
    ] == {family: 3 for family in q.FAMILY_ORDER}
    assert result["source_aggregates"]["dev"][
        "family_eligible_item_counts"
    ] == {family: 6 for family in q.FAMILY_ORDER}

    serialized = json.dumps(result, sort_keys=True)
    for private in (
        "train_private",
        "dev_private",
        "private_id",
        "private question",
        "private_title",
        "private_section",
        "token_",
    ):
        assert private not in serialized


def test_equal_page_title_or_equal_context_unions_components_and_caps_supply() -> None:
    train, dev = _fixture()
    dev["data"][1]["title"] = dev["data"][0]["title"]
    by_title = q.qualify_decoded_sources(train, dev, quotas=TEST_QUOTAS)
    assert by_title["global_component_count"] == 8
    assert by_title["capacity_flow"]["achieved_flow"] == 8
    assert by_title["passed"] is False

    train, dev = _fixture()
    dev["data"][1]["paragraphs"][0]["context"] = (
        dev["data"][0]["paragraphs"][0]["context"]
    )
    # Keep spans exact after replacing the context.
    for qa in dev["data"][1]["paragraphs"][0]["qas"]:
        index = int(qa["id"].rsplit("_", 1)[1])
        source_qa = dev["data"][0]["paragraphs"][0]["qas"][index]
        qa["orig_answer"] = deepcopy(source_qa["orig_answer"])
    by_context = q.qualify_decoded_sources(train, dev, quotas=TEST_QUOTAS)
    assert by_context["global_component_count"] == 8
    assert by_context["capacity_flow"]["achieved_flow"] == 8


def test_cannotanswer_and_overwide_window_have_no_fallback() -> None:
    train, dev = _fixture()
    train["data"][0]["paragraphs"][0]["qas"][1]["orig_answer"] = {
        "answer_start": 0,
        "text": q.CANNOTANSWER,
    }
    cannot = q.qualify_decoded_sources(train, dev, quotas=TEST_QUOTAS)
    assert cannot["source_aggregates"]["train"][
        "family_eligible_item_counts"
    ]["FOLLOW"] == 2
    assert cannot["source_aggregates"]["train"][
        "role_ineligibility_reason_counts"
    ]["current_CANNOTANSWER"] == 1
    assert cannot["source_aggregates"]["train"][
        "role_ineligibility_reason_counts"
    ]["previous_CANNOTANSWER"] == 1

    train, dev = _fixture()
    paragraph = train["data"][0]["paragraphs"][0]
    context = paragraph["context"]
    first = context.index("train_private_0_token_0")
    last_text = "train_private_0_token_100"
    last_end = context.index(last_text) + len(last_text)
    paragraph["qas"][1]["orig_answer"] = {
        "answer_start": first,
        "text": context[first:last_end],
    }
    overwide = q.qualify_decoded_sources(train, dev, quotas=TEST_QUOTAS)
    assert overwide["source_aggregates"]["train"][
        "family_eligible_item_counts"
    ]["FOLLOW"] == 2


def test_exact_codepoint_offset_fails_closed_but_id_is_not_an_extra_gate() -> None:
    train, dev = _fixture()
    answer = train["data"][0]["paragraphs"][0]["qas"][1]["orig_answer"]
    answer["answer_start"] += 1
    with pytest.raises(
        q.QuacP1SourceQualificationError,
        match="original span is not exact",
    ):
        q.qualify_decoded_sources(train, dev, quotas=TEST_QUOTAS)

    train, dev = _fixture()
    dev["data"][0]["paragraphs"][0]["qas"][0]["id"] = (
        train["data"][0]["paragraphs"][0]["qas"][0]["id"]
    )
    result = q.qualify_decoded_sources(train, dev, quotas=TEST_QUOTAS)
    assert result["passed"] is True


def test_required_subset_does_not_turn_empty_optional_topology_into_a_gate() -> None:
    train, dev = _fixture()
    train["data"].append(
        {
            "title": "",
            "section_title": "",
            "paragraphs": [
                {
                    "context": "",
                    "qas": [],
                }
            ],
        }
    )
    result = q.qualify_decoded_sources(train, dev, quotas=TEST_QUOTAS)
    assert result["passed"] is True
    assert result["activity_counts"] == {
        "selection": 0,
        "model": 0,
        "action": 0,
        "score": 0,
        "online_or_API_evaluation": 0,
    }

    empty = q.qualify_decoded_sources(
        {"data": []},
        {"data": []},
        quotas=TEST_QUOTAS,
    )
    assert empty["status"] == "STOP_QUAC_FAMILY_CAPACITY"
    assert empty["capacity_flow"]["achieved_flow"] == 0


def test_exact_file_size_and_sha_are_verified_before_decode(
    tmp_path: Path,
) -> None:
    train, dev = _fixture()
    train_raw = _raw(train)
    dev_raw = _raw(dev)
    train_path = tmp_path / "train.json"
    dev_path = tmp_path / "dev.json"
    train_path.write_bytes(train_raw)
    dev_path.write_bytes(dev_raw)
    train_path.chmod(0o600)
    dev_path.chmod(0o600)
    contract = _contract(train_raw, dev_raw)

    result = q.qualify_source_files(
        train_path,
        dev_path,
        contract=contract,
    )
    assert result["passed"] is True

    train_path.chmod(0o644)
    with pytest.raises(
        q.QuacP1SourceQualificationError,
        match="identity drifted",
    ):
        q.qualify_source_files(train_path, dev_path, contract=contract)

    train_path.chmod(0o600)
    train_path.write_bytes(train_raw + b" ")
    with pytest.raises(
        q.QuacP1SourceQualificationError,
        match="identity drifted",
    ):
        q.qualify_source_files(train_path, dev_path, contract=contract)
