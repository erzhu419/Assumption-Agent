from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import pickle

import pytest

from assumption_agent.benchmarks.multihoprag_typed_operator_v2 import (
    ACTION_IDS,
    ActionTrace,
    ArticleRecord,
    CausalSignature,
    CoverageSignature,
    EvaluationObservation,
    MultiHopRAGTypedOperatorV2Error,
    build_typed_corpus_graph,
    causal_signature,
    compile_query_plan,
    coverage_signature,
    exact_magnitude_signflip_p,
    item_utility,
    make_entity_key,
    paired_utility_summary,
    parse_date_ordinals,
    policies_identifiable,
    recompute_action_trace_sha256,
    run_action,
    run_all_actions,
    select_global_policy,
)


def _fixture_articles() -> tuple[ArticleRecord, ...]:
    acme = make_entity_key("ORG", "Acme Corporation")
    beta = make_entity_key("ORG", "Beta Labs")
    london = make_entity_key("LOC", "London")
    paris = make_entity_key("LOC", "Paris")
    return (
        ArticleRecord(0, "source a", "business", 20200101, (acme, london), (1,)),
        ArticleRecord(1, "source b", "business", 20210101, (acme, paris), (0, 2)),
        ArticleRecord(2, "source c", "technology", 20220101, (acme, beta), (1, 3)),
        ArticleRecord(3, "source a", "technology", 20230101, (beta, paris), (2,)),
        ArticleRecord(4, "source b", "science", 20200202, (beta, london), (5,)),
        ArticleRecord(5, "source c", "science", 20210202, (beta, paris), (4,)),
        ArticleRecord(6, "source d", "sports", 20200303, (london,), (7,)),
        ArticleRecord(7, "source e", "sports", 20210303, (paris,), (6,)),
    )


def _graph():
    return build_typed_corpus_graph(_fixture_articles())


def _plan(graph=None):
    graph = _graph() if graph is None else graph
    acme = make_entity_key("ORG", "Acme Corporation")
    return compile_query_plan(
        graph=graph,
        query="How do Source A and Source B compare Acme Corporation in 2020 and 2021?",
        capability_similarity_ints={
            "comparison_query": 900_000,
            "inference_query": 100_000,
            "temporal_query": 800_000,
        },
        query_entities=(acme,),
    )


def test_graph_is_typed_reciprocal_and_deterministic() -> None:
    first = _graph()
    second = _graph()
    assert first.graph_sha256 == second.graph_sha256
    assert len(first.articles) == 8
    assert any(edge.family == "RECIPROCAL_TOPIC_KNN" for edge in first.edges)
    assert any(edge.family == "CROSS_SOURCE_TYPED_ENTITY" for edge in first.edges)
    restored = pickle.loads(pickle.dumps(first))
    assert restored.graph_sha256 == first.graph_sha256
    assert tuple(restored.neighbors) == tuple(first.neighbors)


def test_graph_rejects_nonreciprocal_topic_neighbor() -> None:
    rows = list(_fixture_articles())
    rows[1] = ArticleRecord(1, "source b", "business", 20210101, rows[1].entities, (2,))
    with pytest.raises(MultiHopRAGTypedOperatorV2Error, match="not reciprocal"):
        build_typed_corpus_graph(tuple(rows))


def test_capability_router_uses_only_frozen_scores_and_tie_order() -> None:
    graph = _graph()
    plan = _plan(graph)
    assert plan.capability == "comparison_query"
    tie = compile_query_plan(
        graph=graph,
        query="Acme Corporation across Source A and Source B",
        capability_similarity_ints={name: 0 for name in (
            "comparison_query",
            "inference_query",
            "temporal_query",
        )},
        query_entities=(make_entity_key("ORG", "Acme Corporation"),),
    )
    assert tie.capability == "comparison_query"
    assert not hasattr(tie, "question_type")


def test_date_parser_pairs_mentions_without_cartesian_or_iso_duplicates() -> None:
    assert parse_date_ordinals("January 2020 and February 2021") == (20200100, 20210200)
    assert parse_date_ordinals("On 2020-01-02, after March 3, 2021 and before 2019") == (
        20200102,
        20210303,
        20190000,
    )
    assert parse_date_ordinals("Invalid 2021-02-31") == ()
    assert parse_date_ordinals("Malformed 2021-13") == ()
    assert parse_date_ordinals("Full width ２０２０-０１-０２") == (20200102,)


def test_all_actions_scan_the_whole_corpus_and_do_not_use_hippo_candidates() -> None:
    graph = _graph()
    relevance = (950_000, 900_000, 850_000, 100_000, 80_000, 60_000, 40_000, 20_000)
    traces = run_all_actions(graph=graph, plan=_plan(graph), relevance_ints=relevance)
    assert tuple(trace.action_id for trace in traces) == ACTION_IDS
    for trace in traces:
        assert trace.ordered_pair_scan_count == 8 * 7
        assert trace.extension_scan_count == (8 - 2) + (8 - 3)
        assert len(trace.core) == 4
        assert len(set(trace.output_top5)) == 5
        assert trace.output_top5[:4] == trace.core
        assert len(trace.trace_sha256) == 64
        assert trace.graph_sha256 == graph.graph_sha256
        assert trace.plan_sha256 == _plan(graph).plan_sha256
        assert len(trace.relevance_sha256) == 64
    entity_trace = next(trace for trace in traces if trace.action_id == "P2_ENTITY_BRIDGE")
    assert entity_trace.core_quality[0] > 0
    assert entity_trace.output_top5 != traces[0].output_top5


def test_required_slots_are_query_derived_and_common_to_every_action() -> None:
    graph = _graph()
    traces = run_all_actions(
        graph=graph,
        plan=_plan(graph),
        relevance_ints=(950_000, 900_000, 850_000, 100_000, 80_000, 60_000, 40_000, 20_000),
    )
    slot_sets = {trace.coverage.slot_keys for trace in traces}
    assert len(slot_sets) == 1
    slots = next(iter(slot_sets))
    assert any(value.startswith("source:source a") for value in slots)
    assert any(value.startswith("entity:ORG:acme corporation") for value in slots)


def _dummy_trace(
    action_id: str,
    *,
    e0: tuple[Fraction | int, ...],
    e1: tuple[Fraction | int, ...],
    output: tuple[int, int, int, int, int],
) -> ActionTrace:
    coverage = CoverageSignature(1, 1, Fraction(1), ("slot",), ("slot",))
    assert all(isinstance(value, Fraction) for value in e1[:3])
    causal = CausalSignature(
        int(e1[0] * 4), e1[0], e1[1], Fraction(0), e1[2]  # type: ignore[arg-type]
    )
    trace = ActionTrace(
        action_id=action_id,
        output_top5=output,
        core=output[:4],
        core_quality=(1,),
        coverage=coverage,
        causal=causal,
        e0_key=e0,
        e1_key=e1,
        ordered_pair_scan_count=56,
        extension_scan_count=11,
        graph_sha256="a" * 64,
        plan_sha256="b" * 64,
        query_sha256="c" * 64,
        relevance_sha256="d" * 64,
        trace_sha256="0" * 64,
    )
    return replace(trace, trace_sha256=recompute_action_trace_sha256(trace))


def test_fixed_e0_and_causal_e1_can_select_identifiable_policies_without_labels() -> None:
    observations = []
    for offset in range(3):
        traces = {}
        for action_i, action_id in enumerate(ACTION_IDS):
            e0 = (Fraction(0), 100 - action_i, 0)
            # E1's first component makes P2 the unique global choice.
            e1 = (
                Fraction(1) if action_id == "P2_ENTITY_BRIDGE" else Fraction(0),
                Fraction(0),
                Fraction(0),
                Fraction(0),
                100 - action_i,
                0,
            )
            output = tuple((action_i + offset + step) % 8 for step in range(5))
            traces[action_id] = _dummy_trace(action_id, e0=e0, e1=e1, output=output)
        observations.append(EvaluationObservation(traces_by_action=traces))
    e0 = select_global_policy(evaluator_id="E0_INDEPENDENT_V2", observations=observations)
    e1 = select_global_policy(evaluator_id="E1_CAUSAL_NECESSITY_V2", observations=observations)
    assert e0.action_id == "P0_IND_SUM"
    assert e1.action_id == "P2_ENTITY_BRIDGE"
    assert policies_identifiable(e0, e1, observations)
    assert len(e0.input_receipt_sha256) == 64
    assert e0.input_receipt_sha256 == e1.input_receipt_sha256


def test_stale_graph_plan_mutation_and_trace_tampering_are_rejected() -> None:
    graph = _graph()
    with pytest.raises(TypeError):
        graph.neighbors["SAME_SOURCE"] = ()  # type: ignore[index]
    stale_plan = _plan(graph)
    rows = list(_fixture_articles())
    rows[7] = ArticleRecord(
        7,
        "source z",
        rows[7].normalized_category,
        rows[7].published_ordinal,
        rows[7].entities,
        rows[7].reciprocal_topic_neighbors,
    )
    other_graph = build_typed_corpus_graph(tuple(rows))
    with pytest.raises(MultiHopRAGTypedOperatorV2Error, match="another graph"):
        run_action(
            action_id="P0_IND_SUM",
            graph=other_graph,
            plan=stale_plan,
            relevance_ints=(8, 7, 6, 5, 4, 3, 2, 1),
        )

    traces = run_all_actions(
        graph=graph,
        plan=stale_plan,
        relevance_ints=(950_000, 900_000, 850_000, 100_000, 80_000, 60_000, 40_000, 20_000),
    )
    tampered = dict((trace.action_id, trace) for trace in traces)
    tampered["P5_FAMILY_UNION"] = replace(
        tampered["P5_FAMILY_UNION"], e0_key=(Fraction(0), 10**12, 0)
    )
    with pytest.raises(MultiHopRAGTypedOperatorV2Error, match="does not match"):
        select_global_policy(
            evaluator_id="E0_INDEPENDENT_V2",
            observations=(EvaluationObservation(traces_by_action=tampered),),
        )


def test_policy_receipt_must_match_the_supplied_observation_matrix() -> None:
    graph = _graph()
    traces = run_all_actions(
        graph=graph,
        plan=_plan(graph),
        relevance_ints=(950_000, 900_000, 850_000, 100_000, 80_000, 60_000, 40_000, 20_000),
    )
    observations = (EvaluationObservation({trace.action_id: trace for trace in traces}),)
    e0 = select_global_policy(evaluator_id="E0_INDEPENDENT_V2", observations=observations)
    e1 = select_global_policy(evaluator_id="E1_CAUSAL_NECESSITY_V2", observations=observations)
    with pytest.raises(MultiHopRAGTypedOperatorV2Error, match="bind"):
        policies_identifiable(replace(e0, input_receipt_sha256="f" * 64), e1, observations)


def test_temporal_edges_are_directed_and_meta_actions_use_assignment_and_order() -> None:
    graph = _graph()
    assert 1 in graph.temporal_successors[0]
    assert 0 not in graph.temporal_successors[1]
    plan = compile_query_plan(
        graph=graph,
        query="Acme Corporation in 2020 and 2021",
        capability_similarity_ints={
            "comparison_query": 0,
            "inference_query": 0,
            "temporal_query": 900_000,
        },
        query_entities=(make_entity_key("ORG", "Acme Corporation"),),
    )
    relevance = (950_000, 900_000, 850_000, 100_000, 80_000, 60_000, 40_000, 20_000)
    p4 = run_action(
        action_id="P4_META_ASSIGN", graph=graph, plan=plan, relevance_ints=relevance
    )
    p5 = run_action(
        action_id="P5_FAMILY_UNION", graph=graph, plan=plan, relevance_ints=relevance
    )
    assert p4.core_quality[0] == 2  # the two declared date operands are one-to-one matched
    assert p4.core_quality[1] > 0  # directed typed temporal closure
    assert p4.core_quality[2] > 0  # selected tuple has chronological order consistency
    assert p5.core_quality[0] == 2
    assert p5.core_quality[1] > 0


def test_causal_connectivity_is_query_grounded_and_inference_excludes_same_source_only() -> None:
    anchor = make_entity_key("ORG", "Anchor")
    island = make_entity_key("ORG", "Island")
    rows = (
        ArticleRecord(0, "source a", "x", 20200101, (anchor,), (1,)),
        ArticleRecord(1, "source b", "x", 20210101, (anchor,), (0,)),
        ArticleRecord(2, "source c", "x", 20220101, (island,), (3,)),
        ArticleRecord(3, "source d", "x", 20230101, (island,), (2,)),
        ArticleRecord(4, "source e", "x", 20240101, (), ()),
    )
    graph = build_typed_corpus_graph(rows)
    plan = compile_query_plan(
        graph=graph,
        query="Infer the Anchor connection",
        capability_similarity_ints={
            "comparison_query": 0,
            "inference_query": 900_000,
            "temporal_query": 0,
        },
        query_entities=(anchor,),
    )
    causal = causal_signature(graph, plan, (0, 1, 2, 3), (5, 4, 3, 2, 1))
    assert causal.path_connectivity == Fraction(1, 2)

    same_source_rows = tuple(
        ArticleRecord(index, "source a", "x", 20200101 + index, (), ())
        for index in range(5)
    )
    same_source_graph = build_typed_corpus_graph(same_source_rows)
    same_source_plan = compile_query_plan(
        graph=same_source_graph,
        query="Infer across Source A",
        capability_similarity_ints={
            "comparison_query": 0,
            "inference_query": 900_000,
            "temporal_query": 0,
        },
        query_entities=(),
    )
    signature = coverage_signature(same_source_graph, same_source_plan, (0, 1, 2, 3))
    assert "relation:entity_or_topic_connected" not in signature.covered_slot_keys


def test_replacement_loss_is_diagnostic_not_in_e1_key() -> None:
    graph = _graph()
    trace = run_all_actions(
        graph=graph,
        plan=_plan(graph),
        relevance_ints=(950_000, 900_000, 850_000, 100_000, 80_000, 60_000, 40_000, 20_000),
    )[0]
    assert isinstance(trace.causal.minimum_replacement_loss, Fraction)
    assert len(trace.e0_key) == 3
    assert len(trace.e1_key) == 6
    assert trace.e1_key[:3] == (
        trace.causal.necessary_fraction,
        trace.causal.minimum_leave_one_out_loss,
        trace.causal.path_connectivity,
    )


def test_late_utility_deduplicates_articles_and_exact_test_is_deterministic() -> None:
    assert item_utility((0, 1, 2, 3, 4), (0, 2)) == 2
    assert item_utility((0, 1, 2, 3, 4), (0, 6)) == Fraction(1, 2)
    with pytest.raises(MultiHopRAGTypedOperatorV2Error, match="gold"):
        item_utility((0, 1, 2, 3, 4), (0, 0))
    deltas = (Fraction(1, 2), Fraction(1, 3), Fraction(0))
    assert exact_magnitude_signflip_p(deltas) == Fraction(1, 4)
    summary = paired_utility_summary(
        (Fraction(2), Fraction(1), Fraction(1, 2)),
        (Fraction(1), Fraction(1), Fraction(1)),
    )
    assert summary.delta_total == Fraction(1, 2)
    assert (summary.gains, summary.harms, summary.ties) == (1, 1, 1)


def test_design_manifest_self_hash_matches_canonical_contract() -> None:
    path = Path(__file__).resolve().parents[1] / "manifests" / "multihoprag_joint_graph_evaluator_design_v2.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("design_sha256")
    canonical = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == claimed
