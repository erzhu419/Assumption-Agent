from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from assumption_agent.benchmarks import maven_ere_g8_e1_core_v1 as core
from assumption_agent.benchmarks import maven_ere_local_runtime_v1 as runtime


def _view(item_id: str = "a" * 64) -> runtime.ItemView:
    events = (
        runtime.EventView(0, "Attack", (("attacked", 0), ("ATTACKED", 2))),
        runtime.EventView(1, "Response", (("responded", 1),)),
        runtime.EventView(2, "Process", (("mediation", 4),)),
    )
    core_events = tuple(
        core.Event(
            row.event_id,
            row.event_type,
            tuple(core.Mention(surface, ordinal) for surface, ordinal in row.mentions),
        )
        for row in events
    )
    return runtime.ItemView(
        item_id=item_id,
        common_query=core.serialize_common_query(core_events[0], core_events[1]),
        sentences=tuple(f"Synthetic sentence {index}." for index in range(6)),
        events=events,
        head_event=0,
        tail_event=1,
        generic_relations=((0, 2), (1, 2)),
    )


def _pack(view: runtime.ItemView) -> dict[str, object]:
    item = {
        "common_query": view.common_query,
        "events": [
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "mentions": [
                    {"sentence_ordinal": ordinal, "surface": surface}
                    for surface, ordinal in event.mentions
                ],
            }
            for event in view.events
        ],
        "generic_relations": [list(row) for row in view.generic_relations],
        "head_event": view.head_event,
        "item_id": view.item_id,
        "sentences": list(view.sentences),
        "tail_event": view.tail_event,
    }
    body = {
        "block": "G_form",
        "item_count": 1,
        "items": [item],
        "schema": "maven_ere_g8_e1_action_view_pack_v1",
        "version": "v1",
    }
    return {**body, "pack_sha256": runtime.stable_hash(body)}


def test_view_pack_validation_and_query_projection(tmp_path: Path) -> None:
    expected = _view()
    path = tmp_path / "view.json"
    path.write_text(json.dumps(_pack(expected), sort_keys=True), encoding="ascii")
    assert runtime.load_view_pack(path, block="G_form") == (expected,)
    drifted = _pack(expected)
    drifted["items"][0]["family"] = "CAUSAL"  # type: ignore[index]
    path.write_text(json.dumps(drifted, sort_keys=True), encoding="ascii")
    with pytest.raises(runtime.MavenEreLocalRuntimeError):
        runtime.load_view_pack(path, block="G_form")


def test_fixed_hypotheses_and_score_collapse_follow_family_order() -> None:
    view = _view()
    hypotheses = runtime.fixed_hypotheses(view)
    assert len(hypotheses) == 10
    assert [family for family, _ in hypotheses[:4]] == [
        "CAUSAL",
        "CAUSAL",
        "SUBEVENT",
        "SUBEVENT",
    ]
    assert all("ATTACKED" not in text for _family, text in hypotheses)
    pairs = runtime.nli_pairs(view)
    assert len(pairs) == len(view.sentences) * 10
    scores: list[int] = []
    for sentence in range(len(view.sentences)):
        scores.extend(
            (
                sentence,
                sentence + 10,
                sentence + 20,
                sentence + 30,
                sentence + 40,
                sentence + 41,
                sentence + 42,
                sentence + 43,
                sentence + 44,
                sentence + 45,
            )
        )
    collapsed = runtime.collapse_nli_scores(view, scores)
    assert collapsed[0] == (10, 30, 45)
    assert collapsed[-1] == (15, 35, 50)


def test_prepare_block_builds_label_free_validated_action_items() -> None:
    views = (_view("a" * 64), _view("b" * 64))

    class FakeEncoder:
        def encode(self, texts: tuple[str, ...]) -> np.ndarray:
            matrix = np.zeros((len(texts), 384), dtype=np.float32)
            for index in range(len(texts)):
                matrix[index, 0] = 1.0
                matrix[index, 1] = index / 1000.0
                matrix[index] /= np.linalg.norm(matrix[index])
            return matrix

    class FakeNLI:
        def score_items(self, rows):
            result = {}
            for item_id, pairs in rows:
                values = []
                for index, _pair in enumerate(pairs):
                    values.append((index % 10) * 1000)
                result[item_id] = tuple(values)
            return result

    prepared = runtime.prepare_block(
        block="G_form",
        views=views,
        encoder=FakeEncoder(),  # type: ignore[arg-type]
        nli_pool=FakeNLI(),  # type: ignore[arg-type]
    )
    assert prepared.block == "G_form"
    assert len(prepared.items) == 2
    assert prepared.preparation_sha256 == runtime.stable_hash(
        {
            "block": "G_form",
            "items": [
                {
                    "item_id": row.view.item_id,
                    "semantic_receipt_sha256": row.semantic_receipt_sha256,
                }
                for row in prepared.items
            ],
        }
    )
    assert all(isinstance(row.item, core.ValidatedActionItem) for row in prepared.items)
    assert all("family" not in row.item.__dataclass_fields__ for row in prepared.items)
