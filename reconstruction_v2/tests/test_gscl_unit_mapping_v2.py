from __future__ import annotations

from dataclasses import replace
import inspect
import itertools
import json
import math

import pytest

import assumption_agent.gscl_unit_mapping_v2 as unit_mapping
from assumption_agent.gscl_narrative_correspondence_v1 import (
    CertificateDisposition,
    ChoiceDisposition,
    GlobalOperator,
    NarrativeContractError,
    NarrativeSource,
    OrientationMode,
    SCHEMA_VERSION,
    SemanticScoreTable,
    SlotPermutation,
    choose_flat_arm,
    choose_full_arm,
    parse_untrusted_generator_completion,
    verify_correspondence,
)
from assumption_agent.gscl_unit_mapping_v2 import (
    K_BEST_ASSIGNMENTS_PER_OPERATOR,
    MAX_CONSTRAINED_ASSIGNMENT_SUBPROBLEMS,
    MAX_EXCLUSIVE_UNITS,
    UNIT_OPERATOR_CLOSURE,
    UnitMappingSearchConfigV2,
    generate_unit_mapping_proposals_v2,
)


def _exclusive_extraction(
    prefix: str,
    units: int,
    *,
    polarity: str = "positive",
    temporal: str = "forward",
    causal: str = "forward",
    generator_kinds: tuple[str, ...] | None = None,
):
    if generator_kinds is not None and len(generator_kinds) != units:
        raise ValueError("generator_kinds length must equal units")
    sentences: list[str] = []
    mentions: list[dict[str, object]] = []
    generators: list[dict[str, object]] = []
    for index in range(units):
        left = f"{prefix}left{index}"
        anchor = f"{prefix}rel{index}"
        right = f"{prefix}right{index}"
        sentences.append(f"{left} {anchor} {right}.")
        left_id = f"{prefix}.u{index}.left"
        anchor_id = f"{prefix}.u{index}.anchor"
        right_id = f"{prefix}.u{index}.right"
        mentions.extend(
            (
                {
                    "mention_id": left_id,
                    "kind": "object",
                    "quote": left,
                    "occurrence": 0,
                },
                {
                    "mention_id": anchor_id,
                    "kind": "generator",
                    "quote": anchor,
                    "occurrence": 0,
                },
                {
                    "mention_id": right_id,
                    "kind": "object",
                    "quote": right,
                    "occurrence": 0,
                },
            )
        )
        generators.append(
            {
                "generator_id": f"{prefix}.u{index}.generator",
                "anchor_mention_id": anchor_id,
                "slot_mention_ids": [left_id, right_id],
                "generator_kind": (
                    "causal"
                    if generator_kinds is None
                    else generator_kinds[index]
                ),
                "polarity": polarity,
                "temporal_orientation": temporal,
                "causal_orientation": causal,
            }
        )
    source = NarrativeSource(f"source.{prefix}", " ".join(sentences))
    completion = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "mentions": mentions,
            "generators": generators,
        },
        separators=(",", ":"),
    )
    return parse_untrusted_generator_completion(source, completion)


def _shared_endpoint_extraction(prefix: str, units: int):
    left = f"{prefix}left"
    right = f"{prefix}right"
    anchors = [f"{prefix}rel{index}" for index in range(units)]
    text = " ".join((left, *anchors, right)) + "."
    left_id = f"{prefix}.left"
    right_id = f"{prefix}.right"
    mentions: list[dict[str, object]] = [
        {
            "mention_id": left_id,
            "kind": "object",
            "quote": left,
            "occurrence": 0,
        },
        {
            "mention_id": right_id,
            "kind": "object",
            "quote": right,
            "occurrence": 0,
        },
    ]
    generators: list[dict[str, object]] = []
    for index, anchor in enumerate(anchors):
        anchor_id = f"{prefix}.anchor{index}"
        mentions.append(
            {
                "mention_id": anchor_id,
                "kind": "generator",
                "quote": anchor,
                "occurrence": 0,
            }
        )
        generators.append(
            {
                "generator_id": f"{prefix}.generator{index}",
                "anchor_mention_id": anchor_id,
                "slot_mention_ids": [left_id, right_id],
                "generator_kind": "causal",
                "polarity": "positive",
                "temporal_orientation": "forward",
                "causal_orientation": "forward",
            }
        )
    return parse_untrusted_generator_completion(
        NarrativeSource(f"source.{prefix}", text),
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "mentions": mentions,
                "generators": generators,
            },
            separators=(",", ":"),
        ),
    )


def _complete_scores(source, target, unit_weights) -> SemanticScoreTable:
    object_scores = {
        (source_id, target_id): 0
        for source_id in source.hypergraph.object_mention_ids
        for target_id in target.hypergraph.object_mention_ids
    }
    generator_scores = {
        (source_generator.generator_id, target_generator.generator_id): (
            unit_weights[source_index][target_index]
        )
        for source_index, source_generator in enumerate(source.generators)
        for target_index, target_generator in enumerate(target.generators)
    }
    return SemanticScoreTable.from_mappings(
        object_scores=object_scores,
        generator_scores=generator_scores,
    )


def _sparse_scores(
    source,
    target,
    allowed: set[tuple[int, int]],
) -> SemanticScoreTable:
    object_scores: dict[tuple[str, str], int] = {}
    generator_scores: dict[tuple[str, str], int] = {}
    for source_index, target_index in allowed:
        source_generator = source.generators[source_index]
        target_generator = target.generators[target_index]
        generator_scores[
            (source_generator.generator_id, target_generator.generator_id)
        ] = 0
        for source_endpoint in source_generator.slot_mention_ids:
            for target_endpoint in target_generator.slot_mention_ids:
                object_scores[(source_endpoint, target_endpoint)] = 0
    return SemanticScoreTable.from_mappings(
        object_scores=object_scores,
        generator_scores=generator_scores,
    )


def _proposal_for(
    result,
    *,
    orientation: OrientationMode,
    invert: bool,
    permutation: SlotPermutation,
):
    matches = tuple(
        proposal
        for proposal in result.proposals
        if proposal.mapping.operator
        == GlobalOperator(
            orientation_mode=orientation,
            invert_polarity=invert,
            slot_permutation=permutation,
        )
    )
    return min(
        matches,
        key=lambda proposal: (
            -proposal.semantic_score_micros,
            proposal.proposal_hash,
        ),
    )


def _target_assignment(proposal, target) -> tuple[int, ...]:
    target_index = {
        generator.generator_id: index
        for index, generator in enumerate(target.generators)
    }
    return tuple(
        target_index[target_id]
        for _, target_id in proposal.mapping.generator_mapping
    )


def test_config_is_fixed_eight_operator_polynomial_contract() -> None:
    config = UnitMappingSearchConfigV2()
    assert len(UNIT_OPERATOR_CLOSURE) == 8
    assert config.operators == UNIT_OPERATOR_CLOSURE
    assert config.max_units == 21
    assert config.k_best_per_operator == 4
    assert K_BEST_ASSIGNMENTS_PER_OPERATOR == 4
    assert config.max_assignments == 8 * (1 + 3 * 21)
    assert config.max_assignments == (
        MAX_CONSTRAINED_ASSIGNMENT_SUBPROBLEMS
    )
    assert config.max_assignments <= 100_000
    assert "top_k" not in json.dumps(config.safe_payload())
    assert "max_assignments" not in inspect.signature(
        UnitMappingSearchConfigV2
    ).parameters


def test_result_seal_binds_v2_config_and_subproblem_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _exclusive_extraction("sealsource", 1)
    target = _exclusive_extraction("sealtarget", 1)
    config = UnitMappingSearchConfigV2()
    calls = 0
    original = unit_mapping._solve_assignment_subproblem

    def spy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        unit_mapping, "_solve_assignment_subproblem", spy
    )
    result = generate_unit_mapping_proposals_v2(
        source,
        target,
        _complete_scores(source, target, ((0,),)),
        config=config,
    )
    result.validate_internal()
    assert result.config_hash == config.config_hash
    # Per operator: one feasible root and one infeasible child subproblem.
    assert result.assignments_explored == 16
    assert result.assignments_explored == calls
    assert result.budget == config.max_assignments
    assert config.safe_payload()["assignments_explored_semantics"] == (
        "constrained_hungarian_assignment_subproblems_solved"
    )
    with pytest.raises(NarrativeContractError) as error:
        replace(result, config_hash="0" * 64)
    assert error.value.issue_id == "search_result_binding_mismatch"
    with pytest.raises(NarrativeContractError) as error:
        replace(result, proposals=result.proposals[:-1])
    assert error.value.issue_id == "search_result_binding_mismatch"


@pytest.mark.parametrize("units", (1, 2, 4, 8, 16, 21))
def test_capacity_scale_is_injective_and_polynomial(units: int) -> None:
    source = _exclusive_extraction(f"scale{units}s", units)
    target = _exclusive_extraction(f"scale{units}t", units)
    # The explicit shape avoids any dependence on candidate truncation.
    weights = tuple(
        tuple(
            1000 if source_index == target_index else -abs(
                source_index - target_index
            )
            for target_index in range(units)
        )
        for source_index in range(units)
    )
    result = generate_unit_mapping_proposals_v2(
        source, target, _complete_scores(source, target, weights)
    )
    per_operator = min(
        K_BEST_ASSIGNMENTS_PER_OPERATOR, math.factorial(units)
    )
    assert len(result.proposals) == 8 * per_operator
    assert not result.budget_exhausted
    assert result.assignments_explored <= 8 * (1 + 3 * units)
    for operator in UNIT_OPERATOR_CLOSURE:
        operator_proposals = sorted(
            (
                proposal
                for proposal in result.proposals
                if proposal.mapping.operator == operator
            ),
            key=lambda proposal: (
                -proposal.semantic_score_micros,
                _target_assignment(proposal, target),
            ),
        )
        assert len(operator_proposals) == per_operator
        assert _target_assignment(
            operator_proposals[0], target
        ) == tuple(range(units))
    for proposal in result.proposals:
        assert len(
            {target_id for _, target_id in proposal.mapping.generator_mapping}
        ) == units
        assert len(
            {target_id for _, target_id in proposal.mapping.object_mapping}
        ) == 2 * units


@pytest.mark.parametrize(("rows", "columns"), ((1, 3), (2, 4), (3, 5), (4, 6)))
def test_rectangular_optimum_matches_bruteforce_oracle(
    rows: int, columns: int
) -> None:
    source = _exclusive_extraction(f"oracle{rows}s", rows)
    target = _exclusive_extraction(f"oracle{rows}t", columns)
    weights = tuple(
        tuple(
            ((source_index + 3) * 17 + (target_index + 5) * 11) % 23 - 9
            for target_index in range(columns)
        )
        for source_index in range(rows)
    )
    scored = tuple(
        (
            sum(weights[row][column] for row, column in enumerate(choice)),
            choice,
        )
        for choice in itertools.permutations(range(columns), rows)
    )
    expected = tuple(
        sorted(scored, key=lambda row: (-row[0], row[1]))[
            :K_BEST_ASSIGNMENTS_PER_OPERATOR
        ]
    )
    result = generate_unit_mapping_proposals_v2(
        source, target, _complete_scores(source, target, weights)
    )
    assert len(result.proposals) == (
        8 * min(K_BEST_ASSIGNMENTS_PER_OPERATOR, len(scored))
    )
    for operator in UNIT_OPERATOR_CLOSURE:
        observed = tuple(
            sorted(
                (
                    (
                        proposal.semantic_score_micros,
                        _target_assignment(proposal, target),
                    )
                    for proposal in result.proposals
                    if proposal.mapping.operator == operator
                ),
                key=lambda row: (-row[0], row[1]),
            )
        )
        assert observed == expected


def test_exact_ties_are_byte_stable_and_lexicographic() -> None:
    source = _exclusive_extraction("tiesource", 3)
    target = _exclusive_extraction("tietarget", 5)
    scores = _complete_scores(
        source, target, tuple(tuple(0 for _ in range(5)) for _ in range(3))
    )
    first = generate_unit_mapping_proposals_v2(source, target, scores)
    second = generate_unit_mapping_proposals_v2(source, target, scores)
    assert first.result_binding_hash == second.result_binding_hash
    assert first.proposal_set_hash == second.proposal_set_hash
    assert [
        proposal.safe_payload() for proposal in first.proposals
    ] == [proposal.safe_payload() for proposal in second.proposals]
    expected = {
        choice
        for choice in itertools.islice(
            itertools.permutations(range(5), 3),
            K_BEST_ASSIGNMENTS_PER_OPERATOR,
        )
    }
    assert {
        _target_assignment(proposal, target)
        for proposal in first.proposals
    } == expected


def test_slot_reverse_maps_owned_endpoints_without_breaking_injection() -> None:
    source = _exclusive_extraction("slotsource", 2)
    target = _exclusive_extraction("slottarget", 3)
    weights = ((0, 1, 50), (40, 2, 0))
    result = generate_unit_mapping_proposals_v2(
        source, target, _complete_scores(source, target, weights)
    )
    proposal = _proposal_for(
        result,
        orientation=OrientationMode.INVERTING,
        invert=False,
        permutation=SlotPermutation.REVERSE,
    )
    assert _target_assignment(proposal, target) == (2, 0)
    object_mapping = dict(proposal.mapping.object_mapping)
    for source_index, target_index in enumerate((2, 0)):
        source_slots = source.generators[source_index].slot_mention_ids
        target_slots = target.generators[target_index].slot_mention_ids
        assert object_mapping[source_slots[0]] == target_slots[1]
        assert object_mapping[source_slots[1]] == target_slots[0]
    assert len(set(object_mapping.values())) == len(object_mapping)


def test_hall_deficiency_and_empty_domain_fail_closed() -> None:
    source = _exclusive_extraction("halls", 3)
    target = _exclusive_extraction("hallt", 3)
    deficient = generate_unit_mapping_proposals_v2(
        source,
        target,
        _sparse_scores(source, target, {(0, 0), (1, 0), (2, 1), (2, 2)}),
    )
    assert deficient.proposals == ()
    assert deficient.reason_ids == ("unit_injective_assignment_empty",)
    assert not deficient.budget_exhausted

    empty = generate_unit_mapping_proposals_v2(
        source,
        target,
        _sparse_scores(source, target, {(0, 0), (1, 1)}),
    )
    assert empty.proposals == ()
    assert empty.reason_ids == ("unit_edge_domain_empty",)


def test_schema_capacity_and_exclusive_ownership_reject_before_search() -> None:
    target = _exclusive_extraction("wiretarget", 2)
    no_scores = SemanticScoreTable(object_scores=(), generator_scores=())

    over_capacity = generate_unit_mapping_proposals_v2(
        _shared_endpoint_extraction("capacity", MAX_EXCLUSIVE_UNITS + 1),
        target,
        no_scores,
    )
    assert over_capacity.proposals == ()
    assert over_capacity.reason_ids == ("source_unit_capacity_exceeded",)
    assert over_capacity.assignments_explored == 0

    shared = generate_unit_mapping_proposals_v2(
        _shared_endpoint_extraction("shared", 2),
        target,
        no_scores,
    )
    assert shared.proposals == ()
    assert shared.reason_ids == ("source_endpoint_ownership_invalid",)

    target_over_capacity = generate_unit_mapping_proposals_v2(
        _exclusive_extraction("smallsource", 1),
        _shared_endpoint_extraction(
            "targetcapacity", MAX_EXCLUSIVE_UNITS + 1
        ),
        no_scores,
    )
    assert target_over_capacity.reason_ids == (
        "target_unit_capacity_exceeded",
    )

    target_shared = generate_unit_mapping_proposals_v2(
        _exclusive_extraction("ownedsource", 2),
        _shared_endpoint_extraction("targetshared", 2),
        no_scores,
    )
    assert target_shared.reason_ids == (
        "target_endpoint_ownership_invalid",
    )

    too_many_source_units = generate_unit_mapping_proposals_v2(
        _exclusive_extraction("rectsource", 3),
        _exclusive_extraction("recttarget", 2),
        no_scores,
    )
    assert too_many_source_units.reason_ids == (
        "unit_injection_impossible",
    )


def test_flat_can_prefer_semantic_contradiction_while_full_switches_operator() -> None:
    source = _exclusive_extraction(
        "armssource",
        1,
        polarity="positive",
        temporal="forward",
        causal="forward",
    )
    target = _exclusive_extraction(
        "armstarget",
        1,
        polarity="negative",
        temporal="reverse",
        causal="reverse",
    )
    source_slots = source.generators[0].slot_mention_ids
    target_slots = target.generators[0].slot_mention_ids
    scores = SemanticScoreTable.from_mappings(
        object_scores={
            (source_slots[0], target_slots[0]): 100,
            (source_slots[1], target_slots[1]): 100,
            (source_slots[0], target_slots[1]): 1,
            (source_slots[1], target_slots[0]): 1,
        },
        generator_scores={
            (
                source.generators[0].generator_id,
                target.generators[0].generator_id,
            ): 100
        },
    )
    result = generate_unit_mapping_proposals_v2(source, target, scores)
    flat = choose_flat_arm(result)
    full = choose_full_arm(source, target, result)
    assert flat.disposition is ChoiceDisposition.SELECTED
    assert full.disposition is ChoiceDisposition.SELECTED
    assert flat.selected_proposal_hash != full.selected_proposal_hash

    flat_proposal = next(
        proposal
        for proposal in result.proposals
        if proposal.proposal_hash == flat.selected_proposal_hash
    )
    flat_certificate = verify_correspondence(
        flat_proposal.mapping, source, target
    )
    assert flat_certificate.disposition is (
        CertificateDisposition.PROPOSAL_STRUCTURALLY_CONTRADICTED
    )
    assert flat_proposal.mapping.operator == GlobalOperator()
    assert full.certificate is not None
    assert full.certificate.disposition is (
        CertificateDisposition.PROPOSAL_INTERNALLY_CONSISTENT
    )
    full_proposal = next(
        proposal
        for proposal in result.proposals
        if proposal.proposal_hash == full.selected_proposal_hash
    )
    assert full_proposal.mapping.operator.orientation_mode is (
        OrientationMode.INVERTING
    )
    assert full_proposal.mapping.operator.invert_polarity
    assert full_proposal.mapping.operator.slot_permutation is (
        SlotPermutation.IDENTITY
    )


def test_kbest_exposes_same_operator_structurally_consistent_runner_up() -> None:
    source = _exclusive_extraction("kbestsource", 1)
    target = _exclusive_extraction(
        "kbesttarget",
        2,
        generator_kinds=("relation", "causal"),
    )
    source_generator = source.generators[0]
    source_slots = source_generator.slot_mention_ids
    object_scores: dict[tuple[str, str], int] = {}
    generator_scores: dict[tuple[str, str], int] = {}
    for target_index, target_generator in enumerate(target.generators):
        target_slots = target_generator.slot_mention_ids
        object_scores[(source_slots[0], target_slots[0])] = 100
        object_scores[(source_slots[1], target_slots[1])] = 100
        object_scores[(source_slots[0], target_slots[1])] = 0
        object_scores[(source_slots[1], target_slots[0])] = 0
        generator_scores[
            (source_generator.generator_id, target_generator.generator_id)
        ] = 1000 - 100 * target_index
    result = generate_unit_mapping_proposals_v2(
        source,
        target,
        SemanticScoreTable.from_mappings(
            object_scores=object_scores,
            generator_scores=generator_scores,
        ),
    )
    flat = choose_flat_arm(result)
    full = choose_full_arm(source, target, result)
    assert flat.disposition is ChoiceDisposition.SELECTED
    assert full.disposition is ChoiceDisposition.SELECTED
    assert flat.selected_proposal_hash != full.selected_proposal_hash

    flat_proposal = next(
        proposal
        for proposal in result.proposals
        if proposal.proposal_hash == flat.selected_proposal_hash
    )
    full_proposal = next(
        proposal
        for proposal in result.proposals
        if proposal.proposal_hash == full.selected_proposal_hash
    )
    assert flat_proposal.mapping.operator == GlobalOperator()
    assert full_proposal.mapping.operator == GlobalOperator()
    assert _target_assignment(flat_proposal, target) == (0,)
    assert _target_assignment(full_proposal, target) == (1,)
    assert flat_proposal.semantic_score_micros > (
        full_proposal.semantic_score_micros
    )
    assert verify_correspondence(
        flat_proposal.mapping, source, target
    ).disposition is (
        CertificateDisposition.PROPOSAL_STRUCTURALLY_CONTRADICTED
    )
    assert full.certificate is not None
    assert full.certificate.disposition is (
        CertificateDisposition.PROPOSAL_INTERNALLY_CONSISTENT
    )


def test_search_has_no_recursive_or_product_enumeration_path() -> None:
    source = inspect.getsource(unit_mapping)
    assert "itertools" not in source
    assert "def visit" not in source
    assert "cartesian" not in source.lower()
    assert "dfs" not in source.lower()
    assert "_hungarian_maximum_injection(" in source
    assert "_k_best_maximum_injections(" in source
    assert "heapq" in source


def test_score_references_remain_fail_closed() -> None:
    source = _exclusive_extraction("refs", 1)
    target = _exclusive_extraction("reft", 1)
    scores = SemanticScoreTable.from_mappings(
        object_scores={("missing.object", "missing.target"): 1},
        generator_scores={},
    )
    with pytest.raises(NarrativeContractError) as error:
        generate_unit_mapping_proposals_v2(source, target, scores)
    assert error.value.issue_id == "object_score_ref_invalid"
