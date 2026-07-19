from __future__ import annotations

import math

from assumption_agent.benchmarks import maven_ere_g8_e1_core_v1 as core
from assumption_agent.benchmarks import maven_ere_g8_e1_formal_controller_v1 as controller
from assumption_agent.benchmarks import maven_ere_local_runtime_v1 as runtime


def _normalized(values: tuple[float, ...]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in values))
    return tuple(value / norm for value in values)


def _prepared(index: int, family: str) -> runtime.PreparedItem:
    sentences = tuple(f"Controller synthetic {index} sentence {ordinal}." for ordinal in range(8))
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
                tuple((mention.surface, mention.sentence_ordinal) for mention in event.mentions),
            )
            for event in events
        ),
        head_event=0,
        tail_event=1,
        generic_relations=((0, 2), (1, 2)),
    )
    return runtime.PreparedItem(view, item, f"{index + 1000:064x}")


def _manual_models() -> tuple[core.G8Model, core.E1Model]:
    g_weights = [0.0] * len(core.G8_FEATURE_ORDER)
    g_weights[core.G8_FEATURE_ORDER.index("generic_two_edge_path_terminal_fraction")] = 10.0
    g8 = core.G8Model(
        tuple(g_weights),
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
        "5" * 64,
    )
    e1 = core.E1Model(
        tuple(0.0 for _ in core.E1_FEATURE_ORDER),
        tuple(1.0 for _ in core.E1_FEATURE_ORDER),
        "6" * 64,
        "7" * 64,
        "8" * 64,
        "9" * 64,
        "a" * 64,
    )
    return g8, e1


def test_execute_block_submits_three_tasks_per_item_with_bounded_physical_pools() -> None:
    prepared_items = tuple(
        _prepared(index, core.FAMILY_ORDER[index % 3]) for index in range(6)
    )
    block = runtime.PreparedBlock("F_search", prepared_items, "b" * 64)
    g8, e1 = _manual_models()

    class FakeHippo:
        def retrieve(self, *, block: str, view: runtime.ItemView):
            assert block == "F_search"
            return (0, 1, 2)

    result = controller.execute_block(
        prepared=block,
        g8_model=g8,
        e1_model=e1,
        hippo=FakeHippo(),  # type: ignore[arg-type]
        causal_audit=False,
    )
    assert result.logical_task_count == 18
    assert result.all_3n_tasks_submitted_before_first_result is True
    assert result.local_physical_cap == 16
    assert result.hipporag_physical_cap == 2
    assert len(result.items) == 6


def test_score_promotion_and_real_domain_primary_rules() -> None:
    executions = []
    labels = {}
    index = 0
    for family in core.FAMILY_ORDER:
        for within_family in range(10):
            prepared = _prepared(index, family)
            good = (0, 1, 4)
            bad = (0, 1, 2)
            e0 = bad if within_family < 2 else good
            labels[prepared.view.item_id] = family
            executions.append(
                controller.ItemExecution(
                    prepared=prepared,
                    raw=controller.RawArtifact(prepared.view.item_id, bad),
                    hippo=controller.HippoArtifact(prepared.view.item_id, bad),
                    agent=controller.AgentArtifact(
                        item_id=prepared.view.item_id,
                        e0_selected=e0,
                        e1_selected=good,
                        e0_behavior_sha256=f"{index + 2000:064x}",
                        e1_behavior_sha256=f"{index + 3000:064x}",
                        frontier_sha256=f"{index + 4000:064x}",
                        edge_deletion_witness_count=1,
                        edge_deletion_action_change_count=1,
                    ),
                )
            )
            index += 1
    block = controller.BlockExecution(
        block="A_hold",
        items=tuple(executions),
        all_3n_tasks_submitted_before_first_result=True,
        logical_task_count=90,
        local_physical_cap=16,
        hipporag_physical_cap=2,
    )
    score = controller.score_block(block, labels)
    assert score["arm_correct_count"] == {
        "RAW": 0,
        "HippoRAG": 0,
        "E0": 24,
        "E1": 30,
    }
    assert score["comparisons"]["E1_minus_E0"]["net_utility"] == 6
    assert score["comparisons"]["E1_minus_HippoRAG"]["family_net"] == {
        family: 10 for family in core.FAMILY_ORDER
    }
    assert controller.evaluator_promoted(score) is True
    assert controller.real_domain_primary_passed(score) is True


def test_promotion_rejects_a_family_harm_even_with_positive_total() -> None:
    score = {
        "behavior_distinct_E1_vs_E0_count": 8,
        "comparisons": {
            "E1_minus_E0": {
                "family_net": {"CAUSAL": 5, "SUBEVENT": 4, "TEMPORAL": -1},
                "net_utility": 8,
                "p_value": {"numerator": 1, "denominator": 256},
            }
        },
    }
    assert controller.evaluator_promoted(score) is False
