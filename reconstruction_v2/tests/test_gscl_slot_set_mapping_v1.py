"""Source-free contract tests for categorical slot-set correspondence."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from assumption_agent.gscl_narrative_correspondence_v1 import (
    NarrativeExtraction,
    NarrativeSource,
    parse_untrusted_generator_completion,
)
from assumption_agent.gscl_slot_set_mapping_v1 import (
    K_BEST_PER_POOL,
    MAXIMUM_ASSIGNMENT_SUBPROBLEMS,
    MAXIMUM_PROPOSALS,
    MAXIMUM_SLOTS_PER_SIDE,
    MappingArm,
    OPERATOR_CLOSURE,
    SemanticSlotScoreMatrixV1,
    SlotRelationInputV1,
    SlotSetMappingError,
    build_slot_graph_v1,
    map_slot_graphs_v1,
    qualify_exact_bounded_slot_ownership,
)
from replication_runtime.gscl_narrative_extractor_v1.closed_choice_worker import (
    PromptAnswer,
)
from replication_runtime.gscl_narrative_extractor_v2 import (
    bounded_set_consumer,
    closed_choice,
    document_envelope,
)


def _h(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _graph(
    prefix: str,
    count: int,
    edges: tuple[
        tuple[int, int, str, str, str, str], ...
    ],
    *,
    labels: tuple[str, ...] | None = None,
    coverage_complete: bool = True,
):
    slot_ids = tuple(f"{prefix}.s{index}" for index in range(count))
    if labels is None:
        labels = tuple(f"{prefix}-label-{index}" for index in range(count))
    slot_labels = dict(zip(slot_ids, labels, strict=True))
    evidence = {slot_id: _h(f"{prefix}.slot.{index}") for index, slot_id in enumerate(slot_ids)}
    relations = tuple(
        SlotRelationInputV1(
            relation_id=f"{prefix}.r{index}",
            slot0_id=slot_ids[left],
            slot1_id=slot_ids[right],
            generator_kind=kind,
            polarity=polarity,
            temporal_orientation=temporal,
            causal_orientation=causal,
            evidence_binding_sha256=_h(f"{prefix}.relation.{index}"),
        )
        for index, (left, right, kind, polarity, temporal, causal) in enumerate(edges)
    )
    return build_slot_graph_v1(
        slot_labels=slot_labels,
        slot_evidence_bindings=evidence,
        relations=relations,
        extractor_binding_sha256=_h(f"{prefix}.extractor"),
        coverage_complete=coverage_complete,
    )


def _scores(source, target, function):
    return SemanticSlotScoreMatrixV1.from_mapping(
        {
            (left.slot_id, right.slot_id): function(source_index, target_index)
            for source_index, left in enumerate(source.slots)
            for target_index, right in enumerate(target.slots)
        }
    )


def _selected(result, arm: MappingArm):
    choice = result.choice(arm)
    assert choice.proposal_hash is not None
    return next(row for row in result.proposals if row.proposal_hash == choice.proposal_hash)


def test_fixed_polynomial_contract_and_claim_boundary() -> None:
    assert len(OPERATOR_CLOSURE) == 8
    assert K_BEST_PER_POOL == 4
    assert MAXIMUM_PROPOSALS == 64
    assert MAXIMUM_ASSIGNMENT_SUBPROBLEMS == 9 * (1 + 3 * 16)
    assert len({row.operator_id for row in OPERATOR_CLOSURE}) == 8

    graph = _graph(
        "contract",
        2,
        ((0, 1, "causal", "positive", "forward", "forward"),),
    )
    receipt = graph.receipt
    assert receipt["internal_graph_authority_established"] is True
    assert receipt["external_extractor_semantic_truth_established"] is False
    assert receipt["positional_slot0_slot1_only"] is True
    assert receipt["directed_endpoint_semantics_established"] is False
    assert receipt["formal_law_binding_count"] == 0
    assert receipt["effect_authority_established"] is False


def test_duplicate_nfkc_casefold_slots_and_resource_overflow_typed_fail() -> None:
    with pytest.raises(SlotSetMappingError) as collision:
        build_slot_graph_v1(
            slot_labels={"slot.a": "Ｋ", "slot.b": "k"},
            slot_evidence_bindings={"slot.a": _h("a"), "slot.b": _h("b")},
            relations=(),
            extractor_binding_sha256=_h("extractor"),
            coverage_complete=True,
        )
    assert collision.value.issue_id == "SCAR_NORMALIZED_SLOT_AMBIGUOUS"

    labels = {
        f"slot.s{index}": f"label-{index}"
        for index in range(MAXIMUM_SLOTS_PER_SIDE + 1)
    }
    with pytest.raises(SlotSetMappingError) as resource:
        build_slot_graph_v1(
            slot_labels=labels,
            slot_evidence_bindings={key: _h(key) for key in labels},
            relations=(),
            extractor_binding_sha256=_h("extractor"),
            coverage_complete=True,
        )
    assert resource.value.issue_id == "SCAR_RESOURCE_BOUND_EXCEEDED"


def test_complete_integer_matrix_is_mandatory() -> None:
    source = _graph("matrix.source", 2, ())
    target = _graph("matrix.target", 2, ())
    incomplete = SemanticSlotScoreMatrixV1.from_mapping(
        {(source.slots[0].slot_id, target.slots[0].slot_id): 1}
    )
    with pytest.raises(SlotSetMappingError) as error:
        map_slot_graphs_v1(source, target, incomplete)
    assert error.value.issue_id == "SCAR_SCORE_MATRIX_INCOMPLETE"

    with pytest.raises(SlotSetMappingError) as boolean_score:
        SemanticSlotScoreMatrixV1.from_mapping(
            {(source.slots[0].slot_id, target.slots[0].slot_id): True}
        )
    assert boolean_score.value.issue_id == "SCAR_SCORE_MATRIX_INVALID"


def test_structure_pool_recalls_mapping_outside_semantic_top_k() -> None:
    colors = (
        ("relation", "positive", "none", "none"),
        ("state_change", "negative", "forward", "none"),
        ("temporal", "neutral", "reverse", "none"),
        ("causal", "positive", "forward", "reverse"),
    )
    edges = tuple((index, index, *color) for index, color in enumerate(colors))
    source = _graph("pool.source", 4, edges)
    target = _graph("pool.target", 4, edges)
    # Semantic k-best strongly prefers the rotation 0->1->2->3->0 and its
    # near variants; the structurally exact identity has semantic score zero.
    scores = _scores(
        source,
        target,
        lambda left, right: 100 if right == (left + 1) % 4 else 0,
    )
    result = map_slot_graphs_v1(source, target, scores)
    identity = tuple(range(4))
    identity_rows = tuple(
        row
        for row in result.proposals
        if row.target_indices == identity
        and row.operator_id == OPERATOR_CLOSURE[0].operator_id
    )
    assert len(identity_rows) == 1
    assert identity_rows[0].origins == ("structure_kbest",)
    assert identity_rows[0].typed_incidence_verified is True
    assert identity_rows[0].length2_composition_verified is True
    assert result.receipt["structure_only_pool_uses_semantic_scores"] is False
    assert result.receipt["candidate_pool_union"] == [
        "semantic_kbest",
        "structure_kbest",
    ]
    semantic_only = _selected(result, MappingArm.SEMANTIC_ONLY)
    assert semantic_only.target_indices == (1, 2, 3, 0)
    assert "semantic_kbest" in semantic_only.origins


def test_semantic_only_ignores_structure_pool_scores_and_verification() -> None:
    first_edges = (
        (0, 0, "relation", "positive", "none", "none"),
        (1, 1, "causal", "negative", "forward", "reverse"),
        (2, 2, "temporal", "neutral", "reverse", "none"),
    )
    second_edges = (
        (0, 1, "state_change", "positive", "forward", "none"),
        (1, 2, "state_change", "positive", "forward", "none"),
        (2, 0, "state_change", "positive", "forward", "none"),
    )
    source = _graph("semantic.only.source", 3, first_edges)
    first_target = _graph("semantic.only.target", 3, first_edges)
    # Same slot IDs and score matrix, but a categorically different target
    # graph changes structural ranks and certificates.
    second_target = _graph("semantic.only.target", 3, second_edges)
    preferred = (2, 0, 1)
    matrix = _scores(
        source,
        first_target,
        lambda left, right: 50 if right == preferred[left] else 0,
    )
    first = map_slot_graphs_v1(source, first_target, matrix)
    second = map_slot_graphs_v1(source, second_target, matrix)
    first_choice = _selected(first, MappingArm.SEMANTIC_ONLY)
    second_choice = _selected(second, MappingArm.SEMANTIC_ONLY)

    assert first_choice.target_indices == preferred
    assert second_choice.target_indices == preferred
    assert first_choice.semantic_score == second_choice.semantic_score == 150
    assert "semantic_kbest" in first_choice.origins
    assert "semantic_kbest" in second_choice.origins
    assert first.receipt["semantic_only_uses_structural_scores"] is False
    assert (
        first.receipt["semantic_only_uses_structure_only_proposals"]
        is False
    )
    assert first.receipt["semantic_only_operator_id"] == (
        OPERATOR_CLOSURE[0].operator_id
    )
    # Structural evidence changes, but cannot alter the semantic-only arm.
    assert (
        first.receipt["proposal_set_commitment"]
        != second.receipt["proposal_set_commitment"]
    )


def test_length2_composition_separates_locally_tied_candidate() -> None:
    # This regular directed graph gives every node the same local typed
    # incidence.  Swapping nodes 1 and 2 is therefore locally tied but does
    # not preserve its length-two paths.
    edges = tuple(
        (index, (index + offset) % 5, "relation", "positive", "none", "none")
        for index in range(5)
        for offset in (1, 2)
    )
    source = _graph("compose.source", 5, edges)
    target = _graph("compose.target", 5, edges)
    wrong = (0, 2, 1, 3, 4)
    scores = _scores(
        source,
        target,
        lambda left, right: 100 if right == wrong[left] else 0,
    )
    result = map_slot_graphs_v1(source, target, scores)
    local = _selected(result, MappingArm.FULL_NO_COMPOSITION)
    composed = _selected(result, MappingArm.FULL_WITH_LENGTH2_COMPOSITION)

    assert local.target_indices == wrong
    assert local.typed_incidence_verified is True
    assert local.length2_composition_verified is False
    assert composed.typed_incidence_verified is True
    assert composed.length2_composition_verified is True
    assert composed.target_indices != wrong


def test_one_sided_target_color_shuffle_lowers_fixed_semantic_alignment() -> None:
    colors = (
        ("relation", "positive", "none", "none"),
        ("state_change", "positive", "none", "none"),
        ("temporal", "positive", "none", "none"),
        ("causal", "positive", "none", "none"),
    )
    edges = tuple((index, index, *color) for index, color in enumerate(colors))
    source = _graph("shuffle.source", 4, edges)
    target = _graph("shuffle.target", 4, edges)
    scores = _scores(source, target, lambda left, right: 100 if left == right else 0)

    normal = map_slot_graphs_v1(source, target, scores)
    shuffled = map_slot_graphs_v1(
        source, target, scores, target_color_shuffle=True
    )
    normal_selected = _selected(
        normal, MappingArm.FULL_WITH_LENGTH2_COMPOSITION
    )
    shuffled_selected = _selected(
        shuffled, MappingArm.FULL_WITH_LENGTH2_COMPOSITION
    )
    assert shuffled.target_color_shuffle_effective is True
    assert shuffled.receipt["target_color_shuffle_requested"] is True
    assert shuffled_selected.semantic_score < normal_selected.semantic_score
    assert shuffled.receipt["score_matrix_commitment"] == normal.receipt[
        "score_matrix_commitment"
    ]


def test_shape_and_decisions_are_slot_rename_invariant() -> None:
    edges = (
        (0, 1, "causal", "positive", "forward", "forward"),
        (1, 2, "temporal", "neutral", "forward", "none"),
        (2, 0, "relation", "negative", "none", "none"),
    )
    first_source = _graph("rename.a.source", 3, edges)
    first_target = _graph("rename.a.target", 3, edges)
    second_source = _graph(
        "rename.z.source", 3, edges, labels=("One", "Two", "Three")
    )
    second_target = _graph(
        "rename.z.target", 3, edges, labels=("Uno", "Dos", "Tres")
    )
    first = map_slot_graphs_v1(
        first_source,
        first_target,
        _scores(first_source, first_target, lambda left, right: 7 if left == right else 0),
    )
    second = map_slot_graphs_v1(
        second_source,
        second_target,
        _scores(second_source, second_target, lambda left, right: 7 if left == right else 0),
    )
    assert first_source.receipt["rename_invariant_shape_sha256"] == (
        second_source.receipt["rename_invariant_shape_sha256"]
    )
    assert sorted(
        (
            row.semantic_score,
            row.flat_structural_score,
            row.typed_incidence_verified,
            row.length2_composition_verified,
            row.origins,
        )
        for row in first.proposals
    ) == sorted(
        (
            row.semantic_score,
            row.flat_structural_score,
            row.typed_incidence_verified,
            row.length2_composition_verified,
            row.origins,
        )
        for row in second.proposals
    )
    assert [row.disposition for row in first.choices] == [
        row.disposition for row in second.choices
    ]


class _NeverCalled:
    def select_story(self, story_text: str):  # pragma: no cover - must not run
        raise AssertionError("short context unexpectedly invoked leaf")


class _FakeLeafBackend:
    @property
    def runtime_commitment(self) -> str:
        return _h("slot-mapping-fake-leaf")

    def score_batch(
        self, pairs: tuple[PromptAnswer, ...]
    ) -> tuple[closed_choice.TeacherForcedScore, ...]:
        rows = []
        for pair in pairs:
            preferred = int(pair.candidate_key.endswith(".plan.one_relation"))
            token_count = max(1, len(pair.answer.split()))
            rows.append(
                closed_choice.TeacherForcedScore(
                    total_logprob_microunits=(
                        preferred * 1_000_000 * token_count
                    ),
                    answer_token_count=token_count,
                    context_and_answer_token_count=token_count + 80,
                )
            )
        return tuple(rows)

    def count_program_owned_completion_tokens(self, completion: str) -> int:
        return max(1, len(completion.encode("utf-8")) // 4)


def _leaf_parser(story: str, completion: str) -> NarrativeExtraction:
    return parse_untrusted_generator_completion(
        NarrativeSource("slot.mapping.fake.source", story), completion
    )


class _OneRelationLeaf:
    def select_story(self, story_text: str):
        return closed_choice.select_hierarchical_qualification_only(
            story_text,
            backend=_FakeLeafBackend(),
            narrative_parser=_leaf_parser,
        )


def test_exact_bounded_binder_is_always_partial_and_never_mapping_authority() -> None:
    envelope = document_envelope.select_document_qualification_only(
        "Tiny context.", leaf_selector=_NeverCalled()
    )
    bounded = bounded_set_consumer.consume_document_envelope(envelope)
    qualification = qualify_exact_bounded_slot_ownership(
        bounded, ("Tiny", "context")
    )
    receipt = qualification.receipt
    assert receipt["disposition"] == "PARTIAL_EXACT_OWNERSHIP_ONLY"
    assert receipt["partial_graph_authority_established"] is False
    assert receipt["mapping_eligible"] is False
    assert receipt["effect_authority_established"] is False
    assert receipt["partial_edge_count"] == 0
    assert receipt["exact_endpoint_coverage_complete"] is False

    with pytest.raises(SlotSetMappingError) as collision:
        qualify_exact_bounded_slot_ownership(bounded, ("Ｋ", "k"))
    assert collision.value.issue_id == "SCAR_NORMALIZED_SLOT_AMBIGUOUS"


def test_exact_binder_merges_only_owned_endpoints_and_never_invents_edge() -> None:
    story = " ".join(f"Exact{index:02d}" for index in range(17)) + "."
    envelope = document_envelope.select_document_qualification_only(
        story, leaf_selector=_OneRelationLeaf()
    )
    bounded = bounded_set_consumer.consume_document_envelope(envelope)
    assert len(bounded.units) == 1
    episode = bounded.structural_episode
    assert episode is not None
    spans = {row.span_id: row for row in episode.evidence_spans}
    source_bytes = story.encode("utf-8")
    unit = bounded.units[0]
    labels = tuple(
        source_bytes[spans[span_id].start_byte : spans[span_id].end_byte].decode(
            "utf-8"
        )
        for span_id in (unit.slot0_span_id, unit.slot1_span_id)
    )

    complete = qualify_exact_bounded_slot_ownership(bounded, labels).receipt
    missing = qualify_exact_bounded_slot_ownership(bounded, labels[:1]).receipt
    assert complete["matched_endpoint_count"] == 2
    assert complete["partial_edge_count"] == 1
    assert complete["exact_endpoint_coverage_complete"] is True
    assert complete["mapping_eligible"] is False
    assert missing["matched_endpoint_count"] == 1
    assert missing["partial_edge_count"] == 0
    assert missing["missing_endpoint_count"] == 1


def test_partial_coverage_still_verifies_selected_graph_without_recall_claim() -> None:
    edges = (
        (0, 1, "causal", "positive", "forward", "forward"),
        (1, 0, "causal", "positive", "forward", "forward"),
    )
    source = _graph("tamper.source", 2, edges)
    partial_target = _graph(
        "tamper.target", 2, edges, coverage_complete=False
    )
    scores = _scores(source, partial_target, lambda left, right: int(left == right))
    result = map_slot_graphs_v1(source, partial_target, scores)
    assert result.choice(MappingArm.FLAT_STRUCTURAL).proposal_hash is not None
    assert result.choice(MappingArm.FULL_NO_COMPOSITION).proposal_hash is not None
    assert result.choice(
        MappingArm.FULL_WITH_LENGTH2_COMPOSITION
    ).proposal_hash is not None
    assert result.receipt["source_input_coverage_complete"] is True
    assert result.receipt["target_input_coverage_complete"] is False
    assert result.receipt["relation_recall_total"] is False
    assert (
        result.receipt[
            "selected_graph_verification_requires_global_coverage"
        ]
        is False
    )
    assert result.receipt["selected_graph_verification_scope"] == (
        "finite_supplied_categorical_graph_only"
    )


def test_authority_tamper_fails_closed() -> None:
    edges = (
        (0, 1, "causal", "positive", "forward", "forward"),
        (1, 0, "causal", "positive", "forward", "forward"),
    )
    source = _graph("tamper.source", 2, edges)
    target = _graph("tamper.target", 2, edges)
    scores = _scores(source, target, lambda left, right: int(left == right))
    result = map_slot_graphs_v1(source, target, scores)
    with pytest.raises(SlotSetMappingError) as proposal_tamper:
        replace(
            result.proposals[0],
            semantic_score=result.proposals[0].semantic_score + 1,
        )
    assert proposal_tamper.value.issue_id == "SCAR_MAPPING_AUTHORITY_INVALID"
    with pytest.raises(SlotSetMappingError) as result_tamper:
        replace(result, proposals=result.proposals[:-1])
    assert result_tamper.value.issue_id == "SCAR_MAPPING_AUTHORITY_INVALID"
    with pytest.raises(SlotSetMappingError) as graph_tamper:
        replace(source, graph_evidence_binding_sha256=_h("forged"))
    assert graph_tamper.value.issue_id == "SCAR_GRAPH_AUTHORITY_INVALID"


def test_safe_receipt_discloses_no_labels_and_no_effect_claim() -> None:
    source = _graph(
        "safe.source",
        2,
        ((0, 1, "relation", "positive", "none", "none"),),
        labels=("Private Alpha", "Private Beta"),
    )
    target = _graph(
        "safe.target",
        2,
        ((0, 1, "relation", "positive", "none", "none"),),
        labels=("Secret One", "Secret Two"),
    )
    result = map_slot_graphs_v1(
        source,
        target,
        _scores(source, target, lambda left, right: int(left == right)),
    )
    wire = result.receipt_bytes.decode("ascii")
    for forbidden in (
        "Private Alpha",
        "Private Beta",
        "Secret One",
        "Secret Two",
    ):
        assert forbidden not in wire
    receipt = json.loads(wire)
    assert receipt["effect_authority_established"] is False
    assert receipt["pair_label_or_gold_access_count"] == 0
    assert receipt["scorer_access_count"] == 0
    assert receipt["online_evaluator_access_count"] == 0
    assert receipt["formal_law_binding_count"] == 0
    assert receipt["positional_slot0_slot1_only"] is True
