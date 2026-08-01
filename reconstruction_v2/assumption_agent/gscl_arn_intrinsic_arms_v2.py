"""Source-free four-arm intrinsic comparison for exclusive GSCL v2 units.

This module keeps the semantic-only and legacy controls, the shared
proposal-set invariant, and the score-free structural verifier from v1.  Its
only mechanism change is proposal formation: the exponential per-mention DFS
is replaced by the fixed unit-level polynomial proposer from
``gscl_unit_mapping_v2``.

The injected entry point below is qualification-only.  It consumes no labels,
reference answers, benchmark source, network service, or online evaluator.
"""

from __future__ import annotations

from typing import Callable, Sequence

from . import gscl_arn_intrinsic_arms_v1 as v1
from .gscl_narrative_correspondence_v1 import (
    NarrativeExtraction,
    SemanticScoreTable,
)
from .gscl_unit_mapping_v2 import (
    UnitMappingSearchConfigV2,
    generate_unit_mapping_proposals_v2,
)


ARMS_CORE_VERSION = "gscl.arn.intrinsic.arms.v2.unit_mapping"


RawTextScorer = Callable[[bytes, bytes], int]
LegacyVectorizer = Callable[
    [NarrativeExtraction, tuple[str, ...]], Sequence[int]
]
StructuralSemanticScorer = Callable[
    [NarrativeExtraction, NarrativeExtraction], SemanticScoreTable
]


def prepare_structural_candidates_v2(
    query: NarrativeExtraction,
    candidates: tuple[NarrativeExtraction, NarrativeExtraction],
    structural_scorer: StructuralSemanticScorer,
    *,
    mapping_config: UnitMappingSearchConfigV2,
) -> tuple[
    tuple[v1.PreparedStructuralCandidate, ...],
    tuple[v1.CandidateProposalReceipt, ...],
]:
    """Form each candidate's sole shared polynomial proposal set once."""

    if (
        not isinstance(query, NarrativeExtraction)
        or not isinstance(candidates, tuple)
        or len(candidates) != 2
        or any(
            not isinstance(candidate, NarrativeExtraction)
            for candidate in candidates
        )
        or not callable(structural_scorer)
        or type(mapping_config) is not UnitMappingSearchConfigV2
    ):
        raise v1.IntrinsicContractError(
            "v2_structural_preparation_inputs_invalid"
        )
    prepared: list[v1.PreparedStructuralCandidate] = []
    receipts: list[v1.CandidateProposalReceipt] = []
    config_commitment = mapping_config.config_hash
    for ordinal, candidate in enumerate(candidates):
        base_search_payload = {
            "query_extraction_hash": query.extraction_hash,
            "query_provenance_hash": query.provenance_hash,
            "candidate_extraction_hash": candidate.extraction_hash,
            "candidate_provenance_hash": candidate.provenance_hash,
            "mapping_config": config_commitment,
            "proposal_algorithm": mapping_config.algorithm,
        }
        try:
            first_scores = structural_scorer(query, candidate)
            second_scores = structural_scorer(query, candidate)
            if (
                not isinstance(first_scores, SemanticScoreTable)
                or not isinstance(second_scores, SemanticScoreTable)
                or first_scores.safe_payload()
                != second_scores.safe_payload()
            ):
                raise v1.IntrinsicContractError(
                    "structural_scorer_nondeterministic"
                )
            score_commitment = v1._content_hash(  # noqa: SLF001
                first_scores.safe_payload()
            )
            search = generate_unit_mapping_proposals_v2(
                query,
                candidate,
                first_scores,
                config=mapping_config,
            )
        except Exception:
            receipts.append(
                v1.CandidateProposalReceipt(
                    candidate_ordinal=ordinal,
                    candidate_extraction_hash=candidate.extraction_hash,
                    candidate_provenance_hash=candidate.provenance_hash,
                    score_table_commitment=None,
                    flat_proposal_set_hash=None,
                    full_proposal_set_hash=None,
                    flat_choice_commitment=None,
                    full_choice_commitment=None,
                    search_commitment=v1._content_hash(  # noqa: SLF001
                        {
                            **base_search_payload,
                            "status": "scorer_invalid",
                        }
                    ),
                    status="scorer_invalid",
                )
            )
            continue

        flat = v1.choose_flat_arm(search)
        status = "complete"
        if search.budget_exhausted:
            status = "mapping_budget_exhausted"
        elif not search.proposals:
            status = "mapping_empty"
        prepared.append(
            v1.PreparedStructuralCandidate(
                candidate_ordinal=ordinal,
                candidate=candidate,
                score_table_commitment=score_commitment,
                search_result=search,
                flat_choice=flat,
            )
        )
        receipts.append(
            v1.CandidateProposalReceipt(
                candidate_ordinal=ordinal,
                candidate_extraction_hash=candidate.extraction_hash,
                candidate_provenance_hash=candidate.provenance_hash,
                score_table_commitment=score_commitment,
                flat_proposal_set_hash=search.proposal_set_hash,
                full_proposal_set_hash=search.proposal_set_hash,
                flat_choice_commitment=flat.choice_hash,
                full_choice_commitment=None,
                search_commitment=v1._content_hash(  # noqa: SLF001
                    {
                        **base_search_payload,
                        "status": status,
                        "proposal_set_hash": search.proposal_set_hash,
                        "assignments_explored": (
                            search.assignments_explored
                        ),
                        "budget_exhausted": search.budget_exhausted,
                        "reason_ids": list(search.reason_ids),
                    }
                ),
                status=status,
            )
        )
    return tuple(prepared), tuple(receipts)


def evaluate_intrinsic_item_v2(
    *,
    opaque_item_id: str,
    query: NarrativeExtraction,
    candidates: tuple[NarrativeExtraction, NarrativeExtraction],
    raw_text_scorer: RawTextScorer,
    legacy_vectorizer: LegacyVectorizer,
    legacy_feature_ids: tuple[str, ...],
    structural_scorer: StructuralSemanticScorer,
    mapping_config: UnitMappingSearchConfigV2,
    raw_text_scorer_commitment: str,
    legacy_vectorizer_commitment: str,
    structural_scorer_commitment: str,
) -> v1.IntrinsicItemResult:
    """Evaluate the fixed four controls without opening any reference label."""

    v1._require_sha256(  # noqa: SLF001
        opaque_item_id, "opaque_item_id_invalid"
    )
    if (
        not isinstance(query, NarrativeExtraction)
        or not isinstance(candidates, tuple)
        or len(candidates) != 2
        or any(
            not isinstance(candidate, NarrativeExtraction)
            for candidate in candidates
        )
    ):
        raise v1.IntrinsicContractError("intrinsic_inputs_invalid")
    if (
        candidates[0].extraction_hash
        == candidates[1].extraction_hash
        or candidates[0].source.source_sha256
        == candidates[1].source.source_sha256
        or query.extraction_hash
        in {candidate.extraction_hash for candidate in candidates}
    ):
        raise v1.IntrinsicContractError(
            "candidate_independence_invalid"
        )
    if (
        not callable(raw_text_scorer)
        or not callable(legacy_vectorizer)
        or not callable(structural_scorer)
    ):
        raise v1.IntrinsicContractError("scorer_callable_invalid")
    for value, issue in (
        (raw_text_scorer_commitment, "raw_scorer_commitment_invalid"),
        (
            legacy_vectorizer_commitment,
            "legacy_vectorizer_commitment_invalid",
        ),
        (
            structural_scorer_commitment,
            "structural_scorer_commitment_invalid",
        ),
    ):
        v1._require_sha256(value, issue)  # noqa: SLF001
    if (
        not isinstance(legacy_feature_ids, tuple)
        or not 1
        <= len(legacy_feature_ids)
        <= v1.MAX_LEGACY_FEATURES
        or len(set(legacy_feature_ids))
        != len(legacy_feature_ids)
        or any(
            not isinstance(item, str)
            or v1._FEATURE_ID.fullmatch(item) is None  # noqa: SLF001
            for item in legacy_feature_ids
        )
    ):
        raise v1.IntrinsicContractError(
            "legacy_feature_registry_invalid"
        )
    if type(mapping_config) is not UnitMappingSearchConfigV2:
        raise v1.IntrinsicContractError("mapping_config_invalid")

    input_commitment = v1._input_commitment(  # noqa: SLF001
        opaque_item_id, query, candidates
    )
    raw_scores = v1._deterministic_raw_scores(  # noqa: SLF001
        query, candidates, raw_text_scorer
    )
    if raw_scores is None:
        semantic_prediction = v1._abstain(  # noqa: SLF001
            v1.IntrinsicArm.SEMANTIC_ONLY,
            input_commitment,
            "raw_text_scorer_invalid",
            {
                "scorer_commitment": raw_text_scorer_commitment,
                "input_commitment": input_commitment,
            },
        )
    else:
        raw_evidence = {
            "scorer_commitment": raw_text_scorer_commitment,
            "score_commitment": v1._content_hash(  # noqa: SLF001
                list(raw_scores)
            ),
        }
        if raw_scores[0] == raw_scores[1]:
            semantic_prediction = v1._abstain(  # noqa: SLF001
                v1.IntrinsicArm.SEMANTIC_ONLY,
                input_commitment,
                "semantic_only_exact_tie",
                raw_evidence,
            )
        else:
            semantic_prediction = v1._predicted(  # noqa: SLF001
                v1.IntrinsicArm.SEMANTIC_ONLY,
                input_commitment,
                0 if raw_scores[0] > raw_scores[1] else 1,
                raw_evidence,
            )

    legacy_registry_commitment = v1._content_hash(  # noqa: SLF001
        list(legacy_feature_ids)
    )
    legacy_vectors = v1._deterministic_legacy_vectors(  # noqa: SLF001
        query,
        candidates,
        legacy_vectorizer,
        legacy_feature_ids,
    )
    if legacy_vectors is None:
        legacy_prediction = v1._abstain(  # noqa: SLF001
            v1.IntrinsicArm.LEGACY,
            input_commitment,
            "legacy_vectorizer_invalid",
            {
                "vectorizer_commitment": (
                    legacy_vectorizer_commitment
                ),
                "registry_commitment": (
                    legacy_registry_commitment
                ),
            },
        )
    else:
        legacy_ordinal = v1._cosine_order(  # noqa: SLF001
            legacy_vectors[0],
            legacy_vectors[1],
            legacy_vectors[2],
        )
        legacy_evidence = {
            "vectorizer_commitment": legacy_vectorizer_commitment,
            "registry_commitment": legacy_registry_commitment,
            "vector_commitment": v1._content_hash(  # noqa: SLF001
                [list(vector) for vector in legacy_vectors]
            ),
        }
        if legacy_ordinal is None:
            legacy_prediction = v1._abstain(  # noqa: SLF001
                v1.IntrinsicArm.LEGACY,
                input_commitment,
                "legacy_exact_tie_or_zero_vector",
                legacy_evidence,
            )
        else:
            legacy_prediction = v1._predicted(  # noqa: SLF001
                v1.IntrinsicArm.LEGACY,
                input_commitment,
                legacy_ordinal,
                legacy_evidence,
            )

    prepared, initial_receipts = prepare_structural_candidates_v2(
        query,
        candidates,
        structural_scorer,
        mapping_config=mapping_config,
    )
    flat_prediction = v1.select_flat_prediction(
        prepared, input_commitment=input_commitment
    )
    full_prediction, full_choices = v1.select_full_prediction(
        query,
        prepared,
        input_commitment=input_commitment,
    )
    receipts: list[v1.CandidateProposalReceipt] = []
    for receipt in initial_receipts:
        choice = full_choices.get(receipt.candidate_ordinal)
        receipts.append(
            v1.CandidateProposalReceipt(
                candidate_ordinal=receipt.candidate_ordinal,
                candidate_extraction_hash=(
                    receipt.candidate_extraction_hash
                ),
                candidate_provenance_hash=(
                    receipt.candidate_provenance_hash
                ),
                score_table_commitment=(
                    receipt.score_table_commitment
                ),
                flat_proposal_set_hash=(
                    receipt.flat_proposal_set_hash
                ),
                full_proposal_set_hash=(
                    receipt.full_proposal_set_hash
                ),
                flat_choice_commitment=(
                    receipt.flat_choice_commitment
                ),
                full_choice_commitment=(
                    None if choice is None else choice.choice_hash
                ),
                search_commitment=receipt.search_commitment,
                status=receipt.status,
            )
        )

    implementation_commitments = tuple(
        sorted(
            {
                "raw_text_scorer": raw_text_scorer_commitment,
                "legacy_vectorizer": (
                    legacy_vectorizer_commitment
                ),
                "structural_scorer": (
                    structural_scorer_commitment
                ),
                "proposal_consistency_checker": (
                    v1.FIXED_CHECKER_COMMITMENT
                ),
                "legacy_registry": legacy_registry_commitment,
                "mapping_config": mapping_config.config_hash,
            }.items()
        )
    )
    return v1.IntrinsicItemResult(
        opaque_item_id=opaque_item_id,
        query_extraction_hash=query.extraction_hash,
        query_provenance_hash=query.provenance_hash,
        candidate_extraction_hashes=(
            candidates[0].extraction_hash,
            candidates[1].extraction_hash,
        ),
        candidate_provenance_hashes=(
            candidates[0].provenance_hash,
            candidates[1].provenance_hash,
        ),
        implementation_commitments=implementation_commitments,
        candidate_receipts=tuple(receipts),
        predictions=(
            semantic_prediction,
            legacy_prediction,
            flat_prediction,
            full_prediction,
        ),
    )


__all__ = [
    "ARMS_CORE_VERSION",
    "evaluate_intrinsic_item_v2",
    "prepare_structural_candidates_v2",
]
