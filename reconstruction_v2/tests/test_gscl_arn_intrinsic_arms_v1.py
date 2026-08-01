from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
import re

import pytest

import assumption_agent.gscl_narrative_correspondence_v1 as narrative_core
from assumption_agent.gscl_arn_intrinsic_arms_v1 import (
    IntrinsicArm,
    IntrinsicContractError,
    PredictionDisposition,
    evaluate_intrinsic_item,
    prepare_structural_candidates,
    select_flat_prediction,
    select_full_prediction,
)
from assumption_agent.gscl_narrative_correspondence_v1 import (
    GlobalOperator,
    MappingSearchConfig,
    NarrativeSource,
    SCHEMA_VERSION,
    SemanticScoreTable,
    parse_untrusted_generator_completion,
    verify_correspondence,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


COMMITMENTS = {
    "raw_text_scorer_commitment": _sha("raw scorer v1"),
    "legacy_vectorizer_commitment": _sha("legacy vectorizer v1"),
    "structural_scorer_commitment": _sha("structural scorer v1"),
}
FEATURES = ("pattern.alpha", "pattern.beta", "pattern.gamma")
IDENTITY_CONFIG = MappingSearchConfig(
    object_top_k=1,
    generator_top_k=1,
    max_assignments=64,
    operators=(GlobalOperator(),),
)


def _extraction(
    prefix: str,
    *,
    names: tuple[str, str, str],
    verbs: tuple[str, str],
    reverse_second_slots: bool = False,
    second_temporal: str = "forward",
    second_causal: str = "forward",
    preface: str = "",
):
    first, middle, last = names
    verb_one, verb_two = verbs
    text = (
        f"{preface}{first} {verb_one} {middle}. "
        f"Later {middle} {verb_two} {last}."
    )
    object_rows = [
        {
            "mention_id": f"{prefix}.first",
            "kind": "object",
            "quote": first,
            "occurrence": 0,
        },
        {
            "mention_id": f"{prefix}.middle",
            "kind": "object",
            "quote": middle,
            "occurrence": 0,
        },
        {
            "mention_id": f"{prefix}.last",
            "kind": "object",
            "quote": last,
            "occurrence": 0,
        },
        {
            "mention_id": f"{prefix}.anchor.g1",
            "kind": "generator",
            "quote": verb_one,
            "occurrence": 0,
        },
        {
            "mention_id": f"{prefix}.anchor.g2",
            "kind": "generator",
            "quote": verb_two,
            "occurrence": 0,
        },
    ]
    second_slots = [f"{prefix}.middle", f"{prefix}.last"]
    if reverse_second_slots:
        second_slots.reverse()
    completion = {
        "schema_version": SCHEMA_VERSION,
        "mentions": object_rows,
        "generators": [
            {
                "generator_id": f"{prefix}.g1",
                "anchor_mention_id": f"{prefix}.anchor.g1",
                "slot_mention_ids": [
                    f"{prefix}.first",
                    f"{prefix}.middle",
                ],
                "generator_kind": "causal",
                "polarity": "positive",
                "temporal_orientation": "forward",
                "causal_orientation": "forward",
            },
            {
                "generator_id": f"{prefix}.g2",
                "anchor_mention_id": f"{prefix}.anchor.g2",
                "slot_mention_ids": second_slots,
                "generator_kind": "causal",
                "polarity": "positive",
                "temporal_orientation": second_temporal,
                "causal_orientation": second_causal,
            },
        ],
    }
    return parse_untrusted_generator_completion(
        NarrativeSource(f"source.{prefix}", text),
        json.dumps(completion, separators=(",", ":")),
    )


def _aligned_structural_scorer(
    query,
    candidate,
    *,
    base: int,
) -> SemanticScoreTable:
    return SemanticScoreTable.from_mappings(
        object_scores={
            (left, right): base - index
            for index, (left, right) in enumerate(
                zip(
                    query.hypergraph.object_mention_ids,
                    candidate.hypergraph.object_mention_ids,
                )
            )
        },
        generator_scores={
            (left.generator_id, right.generator_id): base - 10 - index
            for index, (left, right) in enumerate(
                zip(query.generators, candidate.generators)
            )
        },
    )


def _word_overlap(left: bytes, right: bytes) -> int:
    left_words = set(re.findall(rb"[A-Za-z]+", left.lower()))
    right_words = set(re.findall(rb"[A-Za-z]+", right.lower()))
    return len(left_words.intersection(right_words))


def _legacy_vectorizer(extraction, feature_ids):
    assert feature_ids == FEATURES
    source_id = extraction.source.source_id
    if ".correct" in source_id:
        return (0, 1, 0)
    return (1, 0, 0)


def _fixture_triplet():
    query = _extraction(
        "item.query",
        names=("Ada", "Beo", "Cyra"),
        verbs=("causes", "enables"),
    )
    high_overlap_wrong = _extraction(
        "item.wrong",
        names=("Ada", "Beo", "Cyra"),
        verbs=("causes", "enables"),
        reverse_second_slots=True,
        second_temporal="reverse",
        second_causal="reverse",
        preface="Today ",
    )
    low_overlap_correct = _extraction(
        "item.correct",
        names=("Rin", "Sava", "Tov"),
        verbs=("propels", "guides"),
    )
    return query, high_overlap_wrong, low_overlap_correct


def _biased_structural_scorer(query, candidate):
    base = 900 if ".wrong" in candidate.source.source_id else 500
    return _aligned_structural_scorer(query, candidate, base=base)


def _evaluate(query, candidates, **overrides):
    kwargs = {
        "opaque_item_id": _sha("opaque synthetic item"),
        "query": query,
        "candidates": candidates,
        "raw_text_scorer": _word_overlap,
        "legacy_vectorizer": _legacy_vectorizer,
        "legacy_feature_ids": FEATURES,
        "structural_scorer": _biased_structural_scorer,
        "mapping_config": IDENTITY_CONFIG,
        **COMMITMENTS,
    }
    kwargs.update(overrides)
    return evaluate_intrinsic_item(**kwargs)


def _prediction_map(result):
    return {prediction.arm: prediction for prediction in result.predictions}


def test_four_arms_separate_surface_similarity_from_verified_structure() -> None:
    query, wrong, correct = _fixture_triplet()
    result = _evaluate(query, (wrong, correct))
    predictions = _prediction_map(result)
    assert predictions[IntrinsicArm.SEMANTIC_ONLY].predicted_ordinal == 0
    assert predictions[IntrinsicArm.LEGACY].predicted_ordinal == 0
    assert predictions[IntrinsicArm.FLAT].predicted_ordinal == 0
    assert predictions[IntrinsicArm.FULL].predicted_ordinal == 1
    assert all(
        prediction.disposition is PredictionDisposition.PREDICTED
        for prediction in predictions.values()
    )
    assert len(result.candidate_receipts) == 2
    assert all(
        receipt.flat_proposal_set_hash
        == receipt.full_proposal_set_hash
        for receipt in result.candidate_receipts
    )
    assert (
        result.private_payload()["privacy_class"]
        == "private_dictionary_linkable_item_evidence"
    )
    safe_text = json.dumps(result.private_payload(), sort_keys=True)
    for private_token in (
        "Ada",
        "Beo",
        "Cyra",
        "Rin",
        "propels",
        "causes",
    ):
        assert private_token not in safe_text


def test_choice_swap_is_equivariant_for_every_nonabstaining_arm() -> None:
    query, wrong, correct = _fixture_triplet()
    forward = _prediction_map(_evaluate(query, (wrong, correct)))
    swapped = _prediction_map(_evaluate(query, (correct, wrong)))
    for arm in IntrinsicArm:
        assert forward[arm].disposition is PredictionDisposition.PREDICTED
        assert swapped[arm].disposition is PredictionDisposition.PREDICTED
        assert swapped[arm].predicted_ordinal == 1 - forward[arm].predicted_ordinal


def test_full_is_invariant_to_synthetic_rename_and_paraphrase() -> None:
    query, wrong, correct = _fixture_triplet()
    first = _prediction_map(_evaluate(query, (wrong, correct)))[IntrinsicArm.FULL]

    renamed_query = _extraction(
        "renamed.query",
        names=("Uma", "Vic", "Wes"),
        verbs=("initiates", "sustains"),
    )
    renamed_wrong = _extraction(
        "renamed.wrong",
        names=("Uma", "Vic", "Wes"),
        verbs=("initiates", "sustains"),
        reverse_second_slots=True,
        second_temporal="reverse",
        second_causal="reverse",
        preface="Now ",
    )
    renamed_correct = _extraction(
        "renamed.correct",
        names=("Xia", "Yui", "Zed"),
        verbs=("launches", "maintains"),
    )
    second = _prediction_map(
        _evaluate(renamed_query, (renamed_wrong, renamed_correct))
    )[IntrinsicArm.FULL]
    assert first.predicted_ordinal == second.predicted_ordinal == 1


def test_flat_calls_no_checker_and_full_calls_once_per_shared_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query, wrong, correct = _fixture_triplet()
    prepared, receipts = prepare_structural_candidates(
        query,
        (wrong, correct),
        _biased_structural_scorer,
        mapping_config=IDENTITY_CONFIG,
    )
    input_commitment = _sha("pair input")
    calls = 0

    original = narrative_core.verify_correspondence

    def spy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(narrative_core, "verify_correspondence", spy)
    flat = select_flat_prediction(
        prepared, input_commitment=input_commitment
    )
    assert calls == 0
    full, _ = select_full_prediction(
        query,
        prepared,
        input_commitment=input_commitment,
    )
    assert calls == sum(
        len(item.search_result.proposals) for item in prepared
    )
    assert flat.predicted_ordinal == 0
    assert full.predicted_ordinal == 1
    assert all(
        receipt.flat_proposal_set_hash
        == receipt.full_proposal_set_hash
        for receipt in receipts
    )


def test_full_never_uses_semantic_score_to_break_item_tie() -> None:
    query = _extraction(
        "tie.query",
        names=("Ari", "Bex", "Cal"),
        verbs=("forms", "sends"),
    )
    first = _extraction(
        "tie.first",
        names=("Dax", "Eve", "Fox"),
        verbs=("shapes", "routes"),
    )
    second = _extraction(
        "tie.second",
        names=("Gia", "Han", "Ira"),
        verbs=("builds", "carries"),
    )

    def scorer(query_extraction, candidate_extraction):
        base = 900 if ".first" in candidate_extraction.source.source_id else 400
        return _aligned_structural_scorer(
            query_extraction, candidate_extraction, base=base
        )

    result = _evaluate(
        query,
        (first, second),
        structural_scorer=scorer,
    )
    predictions = _prediction_map(result)
    assert predictions[IntrinsicArm.FLAT].predicted_ordinal == 0
    assert predictions[IntrinsicArm.FULL].disposition is PredictionDisposition.ABSTAIN
    assert predictions[IntrinsicArm.FULL].reason_ids == (
        "full_item_exact_tie",
    )


def test_ties_invalid_scorers_and_budget_exhaustion_abstain_by_arm() -> None:
    query, wrong, correct = _fixture_triplet()
    counter = 0

    def nondeterministic_raw(_left, _right):
        nonlocal counter
        counter += 1
        return counter

    def invalid_legacy(_extraction, _features):
        return (1,)

    exhausted_config = MappingSearchConfig(
        object_top_k=1,
        generator_top_k=1,
        max_assignments=1,
        operators=(GlobalOperator(),),
    )
    result = _evaluate(
        query,
        (wrong, correct),
        raw_text_scorer=nondeterministic_raw,
        legacy_vectorizer=invalid_legacy,
        mapping_config=exhausted_config,
    )
    predictions = _prediction_map(result)
    assert all(
        predictions[arm].disposition is PredictionDisposition.ABSTAIN
        for arm in IntrinsicArm
    )
    assert {
        receipt.status for receipt in result.candidate_receipts
    } == {"mapping_budget_exhausted"}


def test_raw_and_legacy_exact_ties_abstain_without_thresholds() -> None:
    query, wrong, correct = _fixture_triplet()

    def tied_raw(_left, _right):
        return 7

    def tied_legacy(_extraction, _features):
        return (1, 2, 3)

    result = _evaluate(
        query,
        (wrong, correct),
        raw_text_scorer=tied_raw,
        legacy_vectorizer=tied_legacy,
    )
    predictions = _prediction_map(result)
    assert predictions[
        IntrinsicArm.SEMANTIC_ONLY
    ].disposition is PredictionDisposition.ABSTAIN
    assert predictions[
        IntrinsicArm.LEGACY
    ].disposition is PredictionDisposition.ABSTAIN


def test_api_exposes_no_reference_or_relation_family_parameter() -> None:
    parameters = tuple(inspect.signature(evaluate_intrinsic_item).parameters)
    forbidden = ("gold", "answer", "label", "law")
    assert not any(
        token in parameter.lower()
        for parameter in parameters
        for token in forbidden
    )
    query, wrong, correct = _fixture_triplet()
    result = _evaluate(query, (wrong, correct))
    payload_keys = json.dumps(result.safe_payload(), sort_keys=True).lower()
    assert not any(
        f'"{token}"' in payload_keys
        for token in forbidden
    )


def test_exactly_two_independent_candidates_are_required() -> None:
    query, wrong, correct = _fixture_triplet()
    with pytest.raises(IntrinsicContractError) as error:
        evaluate_intrinsic_item(
            opaque_item_id=_sha("bad arity"),
            query=query,
            candidates=(wrong,),  # type: ignore[arg-type]
            raw_text_scorer=_word_overlap,
            legacy_vectorizer=_legacy_vectorizer,
            legacy_feature_ids=FEATURES,
            structural_scorer=_biased_structural_scorer,
            mapping_config=IDENTITY_CONFIG,
            **COMMITMENTS,
        )
    assert error.value.issue_id == "intrinsic_inputs_invalid"

    with pytest.raises(IntrinsicContractError) as error:
        _evaluate(query, (correct, correct))
    assert error.value.issue_id == "candidate_independence_invalid"


def test_result_commitment_replays_exactly() -> None:
    query, wrong, correct = _fixture_triplet()
    first = _evaluate(query, (wrong, correct))
    second = _evaluate(query, (wrong, correct))
    assert first.safe_payload() == second.safe_payload()
    assert first.result_hash == second.result_hash


def test_result_rejects_cross_field_forgery() -> None:
    query, wrong, correct = _fixture_triplet()
    result = _evaluate(query, (wrong, correct))
    with pytest.raises(IntrinsicContractError) as error:
        replace(
            result,
            candidate_receipts=(
                result.candidate_receipts[0],
                result.candidate_receipts[0],
            ),
        )
    assert error.value.issue_id == "candidate_receipt_count_invalid"

    forged_prediction = replace(
        result.predictions[0],
        input_commitment=_sha("different item"),
    )
    with pytest.raises(IntrinsicContractError) as error:
        replace(
            result,
            predictions=(forged_prediction, *result.predictions[1:]),
        )
    assert error.value.issue_id == "prediction_input_cross_binding_invalid"

    with pytest.raises(IntrinsicContractError) as error:
        replace(result.candidate_receipts[0], status="scorer_invalid")
    assert error.value.issue_id == "scorer_invalid_receipt_has_outputs"
