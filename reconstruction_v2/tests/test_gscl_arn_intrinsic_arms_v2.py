from __future__ import annotations

import hashlib
import inspect
import json

import assumption_agent.gscl_arn_intrinsic_arms_v2 as arms_v2
from assumption_agent import gscl_arn_intrinsic_arms_v1 as arms_v1
from assumption_agent.gscl_narrative_correspondence_v1 import (
    NarrativeSource,
    SCHEMA_VERSION,
    SemanticScoreTable,
    parse_untrusted_generator_completion,
)
from assumption_agent.gscl_unit_mapping_v2 import (
    MAX_CONSTRAINED_ASSIGNMENT_SUBPROBLEMS,
    UnitMappingSearchConfigV2,
)
from replication_runtime.gscl_narrative_extractor_v2 import (
    closed_choice as extractor_v2,
)


def _exclusive_extraction(prefix: str, units: int):
    mentions: list[dict[str, object]] = []
    generators: list[dict[str, object]] = []
    sentences: list[str] = []
    for index in range(units):
        left = f"{prefix}left{index}"
        anchor = f"{prefix}rel{index}"
        right = f"{prefix}right{index}"
        left_id = f"{prefix}.u{index}.left"
        anchor_id = f"{prefix}.u{index}.anchor"
        right_id = f"{prefix}.u{index}.right"
        sentences.append(f"{left} {anchor} {right}.")
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
                "generator_kind": "causal",
                "polarity": "positive",
                "temporal_orientation": "forward",
                "causal_orientation": "forward",
            }
        )
    source = NarrativeSource(
        f"source.{prefix}", " ".join(sentences)
    )
    completion = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "mentions": mentions,
            "generators": generators,
        },
        separators=(",", ":"),
    )
    return parse_untrusted_generator_completion(source, completion)


def _scores(source, target) -> SemanticScoreTable:
    source_object_index = {
        object_id: index
        for index, generator in enumerate(source.generators)
        for object_id in generator.slot_mention_ids
    }
    target_object_index = {
        object_id: index
        for index, generator in enumerate(target.generators)
        for object_id in generator.slot_mention_ids
    }
    return SemanticScoreTable.from_mappings(
        object_scores={
            (left, right): (
                100
                if source_object_index[left]
                == target_object_index[right]
                else 0
            )
            for left in source.hypergraph.object_mention_ids
            for right in target.hypergraph.object_mention_ids
        },
        generator_scores={
            (left.generator_id, right.generator_id): (
                100 if left_index == right_index else 0
            )
            for left_index, left in enumerate(source.generators)
            for right_index, right in enumerate(target.generators)
        },
)

TeacherForcedScore = extractor_v2.TeacherForcedScore


def _sha(label: bytes) -> str:
    return hashlib.sha256(label).hexdigest()


class _ExtractorBackend:
    runtime_commitment = _sha(b"v2 extractor backend")

    def score_batch(self, pairs):
        rows = []
        for pair in pairs:
            preferred = (
                ".plan.one_relation" in pair.candidate_key
                or ".episode.e00" in pair.candidate_key
            )
            rows.append(
                TeacherForcedScore(
                    total_logprob_microunits=(
                        100 if preferred else 0
                    ),
                    answer_token_count=1,
                    context_and_answer_token_count=2,
                )
            )
        return tuple(rows)

    def count_program_owned_completion_tokens(
        self, _completion: str
    ) -> int:
        return 1


def _extract_v2(prefix: str):
    story = " ".join(
        f"{prefix}token{index}" for index in range(17)
    ) + "."
    source = NarrativeSource(f"source.{prefix}", story)
    decision = extractor_v2.select_hierarchical_qualification_only(
        story,
        backend=_ExtractorBackend(),
        narrative_parser=lambda text, completion: (
            parse_untrusted_generator_completion(
                NarrativeSource(source.source_id, text),
                completion,
            )
        ),
    )
    return decision


def test_four_arms_use_one_shared_polynomial_proposal_set_at_u21() -> None:
    query = _exclusive_extraction("query", 21)
    candidates = (
        _exclusive_extraction("first", 21),
        _exclusive_extraction("second", 21),
    )
    result = arms_v2.evaluate_intrinsic_item_v2(
        opaque_item_id=_sha(b"v2 item"),
        query=query,
        candidates=candidates,
        raw_text_scorer=lambda _left, right: (
            2 if b"first" in right else 1
        ),
        legacy_vectorizer=lambda extraction, _features: (
            len(extraction.source.text),
            len(extraction.generators),
        ),
        legacy_feature_ids=("text_length", "generator_count"),
        structural_scorer=_scores,
        mapping_config=UnitMappingSearchConfigV2(),
        raw_text_scorer_commitment=_sha(b"raw"),
        legacy_vectorizer_commitment=_sha(b"legacy"),
        structural_scorer_commitment=_sha(b"structural"),
    )
    assert {prediction.arm for prediction in result.predictions} == set(
        arms_v1.IntrinsicArm
    )
    assert len(result.candidate_receipts) == 2
    assert all(
        receipt.status == "complete"
        and receipt.flat_proposal_set_hash
        == receipt.full_proposal_set_hash
        for receipt in result.candidate_receipts
    )
    assert dict(result.implementation_commitments)[
        "mapping_config"
    ] == UnitMappingSearchConfigV2().config_hash


def test_real_v2_extractor_output_reaches_all_four_consumers() -> None:
    query = _extract_v2("query")
    first = _extract_v2("first")
    second = _extract_v2("second")
    result = arms_v2.evaluate_intrinsic_item_v2(
        opaque_item_id=_sha(b"extractor integration item"),
        query=query.extraction,
        candidates=(first.extraction, second.extraction),
        raw_text_scorer=lambda _left, right: (
            2 if b"first" in right else 1
        ),
        legacy_vectorizer=lambda extraction, _features: (
            len(extraction.source.text),
        ),
        legacy_feature_ids=("text_length",),
        structural_scorer=_scores,
        mapping_config=UnitMappingSearchConfigV2(),
        raw_text_scorer_commitment=_sha(b"raw integration"),
        legacy_vectorizer_commitment=_sha(b"legacy integration"),
        structural_scorer_commitment=_sha(
            b"structural integration"
        ),
    )
    assert {prediction.arm for prediction in result.predictions} == set(
        arms_v1.IntrinsicArm
    )
    assert all(
        decision.receipt["exclusive_endpoint_ownership"] is True
        for decision in (query, first, second)
    )
    assert all(
        receipt.flat_proposal_set_hash
        == receipt.full_proposal_set_hash
        for receipt in result.candidate_receipts
    )


def test_preparation_never_calls_v1_exponential_proposer(
    monkeypatch,
) -> None:
    query = _exclusive_extraction("query", 4)
    candidates = (
        _exclusive_extraction("first", 4),
        _exclusive_extraction("second", 4),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("v1 DFS proposer must not run")

    monkeypatch.setattr(
        arms_v1, "generate_pair_mapping_proposals", forbidden
    )
    prepared, receipts = (
        arms_v2.prepare_structural_candidates_v2(
            query,
            candidates,
            _scores,
            mapping_config=UnitMappingSearchConfigV2(),
        )
    )
    assert len(prepared) == len(receipts) == 2
    assert all(receipt.status == "complete" for receipt in receipts)
    assert all(
        candidate.search_result.assignments_explored
        <= MAX_CONSTRAINED_ASSIGNMENT_SUBPROBLEMS
        for candidate in prepared
    )


def test_v2_entrypoint_has_no_labels_or_online_evaluator_surface() -> None:
    parameters = set(
        inspect.signature(
            arms_v2.evaluate_intrinsic_item_v2
        ).parameters
    )
    assert not parameters.intersection(
        {
            "answer",
            "gold",
            "label",
            "online_evaluator",
            "reference",
            "score",
        }
    )
    source = inspect.getsource(arms_v2)
    assert "generate_pair_mapping_proposals(" not in source
    assert "generate_unit_mapping_proposals_v2(" in source
