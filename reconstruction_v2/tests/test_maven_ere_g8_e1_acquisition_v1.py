from __future__ import annotations

import json

import pytest

from assumption_agent.benchmarks import maven_ere_g8_e1_acquisition_v1 as acquisition


def _document(index: int, family: str, *, split: str, duplicate_text: bool = False) -> dict[str, object]:
    text_index = 0 if duplicate_text else index
    temporal = {label: [] for label in acquisition.TEMPORAL_LABELS}
    causal = {label: [] for label in acquisition.CAUSAL_LABELS}
    subevent: list[list[str]] = []
    if family == "TEMPORAL":
        temporal["BEFORE"] = [["EVENT_0", "EVENT_1"]]
    elif family == "CAUSAL":
        causal["CAUSE"] = [["EVENT_0", "EVENT_1"]]
    elif family == "SUBEVENT":
        subevent = [["EVENT_0", "EVENT_1"]]
    else:
        raise AssertionError(family)
    # Generic two-edge path remains after the queried pair is erased.
    temporal["OVERLAP"] = [["EVENT_0", "EVENT_2"], ["EVENT_2", "EVENT_1"]]
    events = []
    for event_id, sentence in enumerate((0, 1, 4)):
        events.append(
            {
                "id": f"EVENT_{event_id}",
                "mention": [
                    {
                        "id": f"MENTION_{event_id}",
                        "offset": [0, 1],
                        "sent_id": sentence,
                        "trigger_word": f"trigger_{event_id}_{index}",
                    }
                ],
                "type": f"Type_{event_id}",
            }
        )
    return {
        "TIMEX": [],
        "causal_relations": causal,
        "events": events,
        "id": f"{split}-document-{index}",
        "subevent_relations": subevent,
        "temporal_relations": temporal,
        "title": f"Synthetic title {text_index}",
        "tokens": [
            [f"sentence_{text_index}_{sentence}", "content"] for sentence in range(6)
        ],
    }


def _jsonl(rows: list[dict[str, object]]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True).encode("ascii") + b"\n"
        for row in rows
    )


def _sources() -> tuple[bytes, bytes]:
    train: list[dict[str, object]] = []
    valid: list[dict[str, object]] = []
    for family_index, family in enumerate(acquisition.FAMILY_ORDER):
        train.extend(
            _document(family_index * 10 + offset, family, split="train")
            for offset in range(2)
        )
        valid.append(_document(100 + family_index, family, split="valid"))
    return _jsonl(train), _jsonl(valid)


def test_private_assignment_is_repeat_exact_document_disjoint_and_label_separated() -> None:
    train, valid = _sources()
    kwargs = {
        "train_raw": train,
        "valid_raw": valid,
        "secret": b"s" * 32,
        "block_specs": (
            ("G_form", "train", 2, True),
            ("A_hold", "valid", 1, True),
        ),
        "expected_line_counts": {"train": 6, "valid": 3},
    }
    first = acquisition.build_acquisition_materials(**kwargs)
    second = acquisition.build_acquisition_materials(**kwargs)
    assert first == second
    assert first.selected_item_count == 9
    assert first.collision_component_count == 9
    assert set(first.view_packs) == {"G_form", "A_hold"}
    assert set(first.label_packs) == {"G_form", "A_hold"}
    item_ids: set[str] = set()
    for block, pack in first.view_packs.items():
        assert pack["item_count"] == (6 if block == "G_form" else 3)
        for item in pack["items"]:
            assert "family" not in item
            assert "fine_label" not in item
            query_pair = sorted((item["head_event"], item["tail_event"]))
            assert query_pair not in item["generic_relations"]
            assert item["item_id"] not in item_ids
            item_ids.add(item["item_id"])
    assert len(item_ids) == 9
    for pack in first.label_packs.values():
        assert {row["family"] for row in pack["items"]} == set(acquisition.FAMILY_ORDER)
        assert all(set(row) == {"family", "item_id"} for row in pack["items"])


def test_F_search_never_materializes_a_label_pack() -> None:
    train, valid = _sources()
    materials = acquisition.build_acquisition_materials(
        train_raw=train,
        valid_raw=valid,
        secret=b"f" * 32,
        block_specs=(("F_search", "train", 2, False),),
        expected_line_counts={"train": 6, "valid": 3},
    )
    assert set(materials.view_packs) == {"F_search"}
    assert materials.label_packs == {}


def test_collision_component_cap_can_make_assignment_shortfall() -> None:
    rows = [
        _document(0, "CAUSAL", split="train", duplicate_text=True),
        _document(1, "CAUSAL", split="train", duplicate_text=True),
    ]
    # Force all three collision keys equal while retaining distinct source IDs.
    rows[1]["title"] = rows[0]["title"]
    rows[1]["tokens"] = rows[0]["tokens"]
    with pytest.raises(acquisition.AssignmentShortfall):
        acquisition.build_acquisition_materials(
            train_raw=_jsonl(rows),
            valid_raw=b"",
            secret=b"c" * 32,
            block_specs=(("G_form", "train", 2, True),),
            expected_line_counts={"train": 2, "valid": 0},
        )


def test_min_cost_assignment_maximizes_flow_before_cost() -> None:
    target_a = ("block", "A")
    target_b = ("block", "B")
    choices = {
        0: {
            target_a: acquisition._EdgeChoice(1, (), "0A"),
            target_b: acquisition._EdgeChoice(5, (), "0B"),
        },
        1: {target_a: acquisition._EdgeChoice(2, (), "1A")},
    }
    result = acquisition.deterministic_min_cost_assignment(
        choices, {target_a: 1, target_b: 1}
    )
    assert result.assigned_count == result.required_count == 2
    assert result.selected[target_a] == ("1A",)
    assert result.selected[target_b] == ("0B",)


def test_source_reader_rejects_line_count_drift() -> None:
    train, valid = _sources()
    with pytest.raises(acquisition.MavenEreAcquisitionError, match="line count"):
        acquisition.parse_released_members(
            train,
            valid,
            expected_line_counts={"train": 7, "valid": 3},
        )
