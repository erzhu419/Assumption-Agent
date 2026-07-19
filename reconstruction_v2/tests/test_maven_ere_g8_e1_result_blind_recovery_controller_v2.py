from __future__ import annotations

import hashlib
import json
import math

import pytest

from assumption_agent.benchmarks import maven_ere_g8_e1_core_v1 as core
from assumption_agent.benchmarks import maven_ere_g8_e1_formal_controller_v1 as v1
from assumption_agent.benchmarks import maven_ere_g8_e1_result_blind_recovery_controller_v2 as recovery
from assumption_agent.benchmarks import maven_ere_local_runtime_v1 as runtime


def _normalized(values: tuple[float, ...]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in values))
    return tuple(value / norm for value in values)


def _prepared(index: int, family: str) -> runtime.PreparedItem:
    sentences = tuple(f"Recovery synthetic {index} sentence {ordinal}." for ordinal in range(8))
    embeddings = tuple(
        _normalized((1.0, (ordinal + 1) / 20.0, 0.1, 0.2))
        for ordinal in range(8)
    )
    events = (
        core.Event(0, "Attack", (core.Mention("attacked", 0), core.Mention("assault", 2))),
        core.Event(1, "Response", (core.Mention("responded", 1), core.Mention("reply", 3))),
        core.Event(2, "Process", (core.Mention("mediation", 4),)),
    )
    family_index = core.FAMILY_ORDER.index(family)
    wrong = (family_index + 1) % 3
    scores = []
    for ordinal in range(8):
        row = [-3_000_000] * 3
        row[wrong] = 2_000_000
        if ordinal == 4:
            row[family_index] = 7_000_000
            row[wrong] = -2_000_000
        scores.append(tuple(row))
    query = core.serialize_common_query(events[0], events[1])
    item = core.validate_action_item(
        sentences=sentences,
        sentence_embeddings=embeddings,
        events=events,
        head_event=0,
        tail_event=1,
        generic_relations=(core.GenericRelation(0, 2), core.GenericRelation(1, 2)),
        common_query=query,
        query_embedding=_normalized((1.0, 0.0, 0.1, 0.2)),
        sentence_family_nli_scores=scores,
    )
    view = runtime.ItemView(
        item_id=f"{index + 1:064x}",
        common_query=query,
        sentences=sentences,
        events=tuple(
            runtime.EventView(
                event.event_id,
                event.event_type,
                tuple(
                    (mention.surface, mention.sentence_ordinal)
                    for mention in event.mentions
                ),
            )
            for event in events
        ),
        head_event=0,
        tail_event=1,
        generic_relations=((0, 2), (1, 2)),
    )
    return runtime.PreparedItem(view, item, f"{index + 1000:064x}")


def _g8() -> core.G8Model:
    weights = [0.0] * len(core.G8_FEATURE_ORDER)
    weights[
        core.G8_FEATURE_ORDER.index("generic_two_edge_path_terminal_fraction")
    ] = 10.0
    return core.G8Model(
        tuple(weights),
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
        "5" * 64,
    )


def _execution() -> tuple[runtime.PreparedBlock, core.G8Model, v1.BlockExecution]:
    items = tuple(_prepared(i, core.FAMILY_ORDER[i % 3]) for i in range(6))
    prepared = runtime.PreparedBlock("A_form", items, "b" * 64)
    g8 = _g8()

    class FakeHippo:
        def retrieve(self, *, block: str, view: runtime.ItemView):
            assert block == "A_form"
            assert len(view.sentences) == 8
            return (0, 1, 2)

    execution = v1.execute_block(
        prepared=prepared,
        g8_model=g8,
        e1_model=None,
        hippo=FakeHippo(),  # type: ignore[arg-type]
        causal_audit=False,
    )
    return prepared, g8, execution


def test_real_action_archive_reproduces_v1_bug_then_normalizes_and_validates(
    tmp_path,
) -> None:
    prepared, g8, execution = _execution()
    broken = v1._action_archive(execution)
    with pytest.raises(v1.MavenEreFormalControllerError, match="semantic drifted"):
        v1._durable_roundtrip(tmp_path / "v1-broken.json", broken)

    normalized = recovery.normalized_action_archive(execution)
    assert normalized["archive_sha256"] == broken["archive_sha256"]
    assert isinstance(normalized["items"][0]["raw_selected"], list)
    assert isinstance(normalized["items"][0]["agent"]["e0_selected"], list)
    file_sha = v1._durable_roundtrip(tmp_path / "normalized.json", normalized)
    receipt = recovery.validate_reused_a_form_archive(
        tmp_path / "normalized.json",
        prepared,
        g8,
        expected_file_sha256=file_sha,
        expected_item_count=6,
    )
    assert receipt["validated_item_count"] == 6
    assert receipt["logical_task_count"] == 18
    assert receipt["A_form_three_arm_actions_reexecuted"] == 0
    assert receipt["raw3_recomputed_item_count"] == 6
    assert receipt["e0_frontier_and_behavior_recomputed_item_count"] == 6


def test_reused_action_validator_rejects_self_consistent_wrong_e0(tmp_path) -> None:
    prepared, g8, execution = _execution()
    archive = recovery.normalized_action_archive(execution)
    selected = archive["items"][0]["agent"]["e0_selected"]
    archive["items"][0]["agent"]["e0_selected"] = list(reversed(selected))
    body = dict(archive)
    body.pop("archive_sha256")
    archive["archive_sha256"] = v1.stable_hash(body)
    path = tmp_path / "wrong-e0.json"
    raw = v1._canonical_bytes(archive)
    path.write_bytes(raw)
    with pytest.raises(recovery.MavenEreRecoveryError, match="E0 recomputation"):
        recovery.validate_reused_a_form_archive(
            path,
            prepared,
            g8,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
            expected_item_count=6,
        )


def test_reused_action_validator_rejects_file_hash_drift(tmp_path) -> None:
    prepared, g8, execution = _execution()
    archive = recovery.normalized_action_archive(execution)
    path = tmp_path / "archive.json"
    path.write_text(json.dumps(archive), encoding="ascii")
    with pytest.raises(recovery.MavenEreRecoveryError, match="file hash"):
        recovery.validate_reused_a_form_archive(
            path,
            prepared,
            g8,
            expected_file_sha256="0" * 64,
            expected_item_count=6,
        )
