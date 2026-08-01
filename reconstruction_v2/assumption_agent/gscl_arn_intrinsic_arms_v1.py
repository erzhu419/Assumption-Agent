"""Source-free item-level four-arm intrinsic comparison.

The core consumes one query extraction and exactly two independently parsed
candidate extractions.  It emits only an opaque item commitment, ordinal
predictions (or abstentions), and content commitments.  It never consumes or
emits a reference answer.

All potentially learned components are injected:

* ``raw_text_scorer`` supplies deterministic full-text similarity;
* ``legacy_vectorizer`` supplies the complete frozen feature vector; and
* ``structural_scorer`` supplies mention/generator semantic scores.

The last score table is used only to form bounded mapping domains and rank the
flat arm.  Each candidate's proposal set is generated exactly once and shared
by flat and full.  Only full invokes the fixed proposal-consistency checker.
Its ordinal reflects the exact grounded proposal-internal consistency tuple;
it is not a claim that either candidate narrative is true.
No threshold in this module is an effect or efficacy gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Any, Callable, Mapping, Sequence

from .gscl_narrative_correspondence_v1 import (
    ArmChoice,
    ChoiceDisposition,
    CORE_VERSION,
    MappingSearchConfig,
    MappingSearchResult,
    NarrativeExtraction,
    PairMappingProposal,
    SemanticScoreTable,
    choose_flat_arm,
    choose_full_arm,
    generate_pair_mapping_proposals,
)


ARMS_CORE_VERSION = "gscl.arn.intrinsic.arms.v1"
MAX_EXACT_SCORE_ABS = 10**18
MAX_LEGACY_FEATURES = 512
MAX_LEGACY_VALUE = 10**12

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FEATURE_ID = re.compile(r"[a-z][a-z0-9_.-]{1,127}\Z")


class IntrinsicContractError(ValueError):
    def __init__(self, issue_id: str) -> None:
        self.issue_id = issue_id
        super().__init__(issue_id)


class IntrinsicArm(str, Enum):
    SEMANTIC_ONLY = "semantic_only"
    LEGACY = "legacy"
    FLAT = "flat"
    FULL = "full"


class PredictionDisposition(str, Enum):
    PREDICTED = "predicted"
    ABSTAIN = "abstain"


RawTextScorer = Callable[[bytes, bytes], int]
LegacyVectorizer = Callable[
    [NarrativeExtraction, tuple[str, ...]], Sequence[int]
]
StructuralSemanticScorer = Callable[
    [NarrativeExtraction, NarrativeExtraction], SemanticScoreTable
]
FIXED_CHECKER_COMMITMENT = hashlib.sha256(
    f"{CORE_VERSION}:fixed_proposal_consistency_checker".encode()
).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    def check(item: Any) -> None:
        if item is None or type(item) in {bool, int, str}:
            return
        if isinstance(item, list):
            for child in item:
                check(child)
            return
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise TypeError("safe payload key must be a string")
            for child in item.values():
                check(child)
            return
        raise TypeError(f"non-strict safe payload type: {type(item).__name__}")

    check(value)
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, issue_id: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IntrinsicContractError(issue_id)
    return value


@dataclass(frozen=True)
class ArmPrediction:
    arm: IntrinsicArm
    disposition: PredictionDisposition
    predicted_ordinal: int | None
    input_commitment: str
    evidence_commitment: str
    reason_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.arm, IntrinsicArm):
            raise IntrinsicContractError("prediction_arm_invalid")
        if not isinstance(self.disposition, PredictionDisposition):
            raise IntrinsicContractError("prediction_disposition_invalid")
        if self.disposition is PredictionDisposition.PREDICTED:
            if (
                not isinstance(self.predicted_ordinal, int)
                or isinstance(self.predicted_ordinal, bool)
                or self.predicted_ordinal not in {0, 1}
            ):
                raise IntrinsicContractError("prediction_ordinal_invalid")
        elif self.predicted_ordinal is not None:
            raise IntrinsicContractError("abstention_ordinal_present")
        _require_sha256(
            self.input_commitment, "prediction_input_commitment_invalid"
        )
        _require_sha256(
            self.evidence_commitment,
            "prediction_evidence_commitment_invalid",
        )
        if (
            not isinstance(self.reason_ids, tuple)
            or any(not isinstance(item, str) or not item for item in self.reason_ids)
            or (
                self.disposition is PredictionDisposition.PREDICTED
                and self.reason_ids
            )
            or (
                self.disposition is PredictionDisposition.ABSTAIN
                and not self.reason_ids
            )
        ):
            raise IntrinsicContractError("prediction_reasons_invalid")

    def safe_payload(self) -> dict[str, Any]:
        return {
            "arm": self.arm.value,
            "disposition": self.disposition.value,
            "predicted_ordinal": self.predicted_ordinal,
            "input_commitment": self.input_commitment,
            "evidence_commitment": self.evidence_commitment,
            "reason_ids": list(self.reason_ids),
        }


@dataclass(frozen=True)
class CandidateProposalReceipt:
    candidate_ordinal: int
    candidate_extraction_hash: str
    candidate_provenance_hash: str
    score_table_commitment: str | None
    flat_proposal_set_hash: str | None
    full_proposal_set_hash: str | None
    flat_choice_commitment: str | None
    full_choice_commitment: str | None
    search_commitment: str
    status: str

    def __post_init__(self) -> None:
        if self.candidate_ordinal not in {0, 1}:
            raise IntrinsicContractError("candidate_ordinal_invalid")
        _require_sha256(
            self.candidate_extraction_hash,
            "candidate_extraction_hash_invalid",
        )
        _require_sha256(
            self.candidate_provenance_hash,
            "candidate_provenance_hash_invalid",
        )
        for value, issue in (
            (self.score_table_commitment, "score_table_commitment_invalid"),
            (
                self.flat_proposal_set_hash,
                "flat_proposal_set_hash_invalid",
            ),
            (
                self.full_proposal_set_hash,
                "full_proposal_set_hash_invalid",
            ),
            (
                self.flat_choice_commitment,
                "flat_choice_commitment_invalid",
            ),
            (
                self.full_choice_commitment,
                "full_choice_commitment_invalid",
            ),
        ):
            if value is not None:
                _require_sha256(value, issue)
        _require_sha256(
            self.search_commitment, "search_commitment_invalid"
        )
        if self.status not in {
            "complete",
            "scorer_invalid",
            "mapping_budget_exhausted",
            "mapping_empty",
        }:
            raise IntrinsicContractError("candidate_receipt_status_invalid")
        if (
            self.flat_proposal_set_hash is not None
            and self.flat_proposal_set_hash != self.full_proposal_set_hash
        ):
            raise IntrinsicContractError("proposal_set_not_shared")
        optional_outputs = (
            self.score_table_commitment,
            self.flat_proposal_set_hash,
            self.full_proposal_set_hash,
            self.flat_choice_commitment,
            self.full_choice_commitment,
        )
        if self.status == "scorer_invalid":
            if any(value is not None for value in optional_outputs):
                raise IntrinsicContractError(
                    "scorer_invalid_receipt_has_outputs"
                )
        elif (
            self.score_table_commitment is None
            or self.flat_proposal_set_hash is None
            or self.full_proposal_set_hash is None
            or self.flat_choice_commitment is None
        ):
            raise IntrinsicContractError(
                "candidate_receipt_outputs_incomplete"
            )

    def safe_payload(self) -> dict[str, Any]:
        return {
            "candidate_ordinal": self.candidate_ordinal,
            "candidate_extraction_hash": self.candidate_extraction_hash,
            "candidate_provenance_hash": self.candidate_provenance_hash,
            "score_table_commitment": self.score_table_commitment,
            "flat_proposal_set_hash": self.flat_proposal_set_hash,
            "full_proposal_set_hash": self.full_proposal_set_hash,
            "flat_choice_commitment": self.flat_choice_commitment,
            "full_choice_commitment": self.full_choice_commitment,
            "search_commitment": self.search_commitment,
            "status": self.status,
        }


@dataclass(frozen=True)
class PreparedStructuralCandidate:
    candidate_ordinal: int
    candidate: NarrativeExtraction
    score_table_commitment: str
    search_result: MappingSearchResult
    flat_choice: ArmChoice

    def __post_init__(self) -> None:
        if self.candidate_ordinal not in {0, 1}:
            raise IntrinsicContractError(
                "prepared_candidate_ordinal_invalid"
            )
        if not isinstance(self.candidate, NarrativeExtraction):
            raise IntrinsicContractError(
                "prepared_candidate_extraction_invalid"
            )
        self.candidate.__post_init__()
        _require_sha256(
            self.score_table_commitment,
            "prepared_score_commitment_invalid",
        )
        if not isinstance(self.search_result, MappingSearchResult):
            raise IntrinsicContractError(
                "prepared_search_result_invalid"
            )
        self.search_result.validate_internal()
        if (
            self.search_result.target_semantic_hash
            != self.candidate.semantic_hash
            or self.search_result.score_table_hash
            != self.score_table_commitment
            or not isinstance(self.flat_choice, ArmChoice)
            or self.flat_choice.arm.value != "flat"
            or self.flat_choice.checker_called
            or self.flat_choice.pair_input_hash
            != self.search_result.pair_input_hash
            or self.flat_choice.proposal_set_hash
            != self.search_result.proposal_set_hash
            or self.flat_choice.search_result_binding_hash
            != self.search_result.result_binding_hash
        ):
            raise IntrinsicContractError(
                "prepared_candidate_cross_binding_invalid"
            )


@dataclass(frozen=True)
class IntrinsicItemResult:
    opaque_item_id: str
    query_extraction_hash: str
    query_provenance_hash: str
    candidate_extraction_hashes: tuple[str, str]
    candidate_provenance_hashes: tuple[str, str]
    implementation_commitments: tuple[tuple[str, str], ...]
    candidate_receipts: tuple[CandidateProposalReceipt, ...]
    predictions: tuple[ArmPrediction, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.opaque_item_id, "opaque_item_id_invalid")
        _require_sha256(
            self.query_extraction_hash, "query_extraction_hash_invalid"
        )
        _require_sha256(
            self.query_provenance_hash, "query_provenance_hash_invalid"
        )
        if (
            not isinstance(self.candidate_extraction_hashes, tuple)
            or len(self.candidate_extraction_hashes) != 2
            or len(set(self.candidate_extraction_hashes)) != 2
        ):
            raise IntrinsicContractError(
                "candidate_extraction_commitments_invalid"
            )
        for value in self.candidate_extraction_hashes:
            _require_sha256(value, "candidate_extraction_hash_invalid")
        if (
            not isinstance(self.candidate_provenance_hashes, tuple)
            or len(self.candidate_provenance_hashes) != 2
        ):
            raise IntrinsicContractError(
                "candidate_provenance_commitments_invalid"
            )
        for value in self.candidate_provenance_hashes:
            _require_sha256(value, "candidate_provenance_hash_invalid")
        expected_arms = tuple(IntrinsicArm)
        if (
            not isinstance(self.predictions, tuple)
            or any(
                not isinstance(prediction, ArmPrediction)
                for prediction in self.predictions
            )
            or tuple(
                prediction.arm for prediction in self.predictions
            )
            != expected_arms
        ):
            raise IntrinsicContractError("prediction_arm_order_invalid")
        if (
            not isinstance(self.candidate_receipts, tuple)
            or len(self.candidate_receipts) != 2
            or any(
                not isinstance(receipt, CandidateProposalReceipt)
                for receipt in self.candidate_receipts
            )
            or tuple(
                receipt.candidate_ordinal
                for receipt in self.candidate_receipts
            )
            != (0, 1)
        ):
            raise IntrinsicContractError("candidate_receipt_count_invalid")
        for ordinal, receipt in enumerate(self.candidate_receipts):
            if (
                receipt.candidate_extraction_hash
                != self.candidate_extraction_hashes[ordinal]
                or receipt.candidate_provenance_hash
                != self.candidate_provenance_hashes[ordinal]
            ):
                raise IntrinsicContractError(
                    "candidate_receipt_cross_binding_invalid"
                )
        if (
            not isinstance(self.implementation_commitments, tuple)
            or any(
                not isinstance(row, tuple)
                or len(row) != 2
                or not all(isinstance(item, str) for item in row)
                for row in self.implementation_commitments
            )
            or len({row[0] for row in self.implementation_commitments})
            != len(self.implementation_commitments)
        ):
            raise IntrinsicContractError(
                "implementation_commitments_invalid"
            )
        commitment_map = dict(self.implementation_commitments)
        if set(commitment_map) != {
            "raw_text_scorer",
            "legacy_vectorizer",
            "structural_scorer",
            "proposal_consistency_checker",
            "legacy_registry",
            "mapping_config",
        }:
            raise IntrinsicContractError(
                "implementation_commitment_fields_invalid"
            )
        for value in commitment_map.values():
            _require_sha256(value, "implementation_commitment_invalid")
        expected_input = _content_hash(
            {
                "opaque_item_id": self.opaque_item_id,
                "query_extraction_hash": self.query_extraction_hash,
                "query_provenance_hash": self.query_provenance_hash,
                "candidate_extraction_hashes": list(
                    self.candidate_extraction_hashes
                ),
                "candidate_provenance_hashes": list(
                    self.candidate_provenance_hashes
                ),
            }
        )
        if any(
            prediction.input_commitment != expected_input
            for prediction in self.predictions
        ):
            raise IntrinsicContractError(
                "prediction_input_cross_binding_invalid"
            )

    def private_payload(self) -> dict[str, Any]:
        """Return dictionary-linkable item evidence for private archives only."""

        return {
            "version": ARMS_CORE_VERSION,
            "privacy_class": "private_dictionary_linkable_item_evidence",
            "opaque_item_id": self.opaque_item_id,
            "query_extraction_hash": self.query_extraction_hash,
            "query_provenance_hash": self.query_provenance_hash,
            "candidate_extraction_hashes": list(
                self.candidate_extraction_hashes
            ),
            "candidate_provenance_hashes": list(
                self.candidate_provenance_hashes
            ),
            "implementation_commitments": dict(
                self.implementation_commitments
            ),
            "candidate_receipts": [
                receipt.safe_payload() for receipt in self.candidate_receipts
            ],
            "predictions": [
                prediction.safe_payload() for prediction in self.predictions
            ],
        }

    @property
    def result_hash(self) -> str:
        return _content_hash(self.private_payload())

    def safe_payload(self) -> dict[str, Any]:
        """Compatibility alias; the returned item payload is not public-safe."""

        return self.private_payload()


def _mapping_config_payload(config: MappingSearchConfig) -> dict[str, Any]:
    return {
        "object_top_k": config.object_top_k,
        "generator_top_k": config.generator_top_k,
        "minimum_score_micros": config.minimum_score_micros,
        "max_assignments": config.max_assignments,
        "operators": [
            operator.safe_payload() for operator in config.operators
        ],
    }


def _input_commitment(
    opaque_item_id: str,
    query: NarrativeExtraction,
    candidates: tuple[NarrativeExtraction, NarrativeExtraction],
) -> str:
    return _content_hash(
        {
            "opaque_item_id": opaque_item_id,
            "query_extraction_hash": query.extraction_hash,
            "query_provenance_hash": query.provenance_hash,
            "candidate_extraction_hashes": [
                candidate.extraction_hash for candidate in candidates
            ],
            "candidate_provenance_hashes": [
                candidate.provenance_hash for candidate in candidates
            ],
        }
    )


def _abstain(
    arm: IntrinsicArm,
    input_commitment: str,
    reason_id: str,
    evidence_payload: Mapping[str, Any],
) -> ArmPrediction:
    return ArmPrediction(
        arm=arm,
        disposition=PredictionDisposition.ABSTAIN,
        predicted_ordinal=None,
        input_commitment=input_commitment,
        evidence_commitment=_content_hash(dict(evidence_payload)),
        reason_ids=(reason_id,),
    )


def _predicted(
    arm: IntrinsicArm,
    input_commitment: str,
    ordinal: int,
    evidence_payload: Mapping[str, Any],
) -> ArmPrediction:
    return ArmPrediction(
        arm=arm,
        disposition=PredictionDisposition.PREDICTED,
        predicted_ordinal=ordinal,
        input_commitment=input_commitment,
        evidence_commitment=_content_hash(dict(evidence_payload)),
        reason_ids=(),
    )


def _exact_score(value: object) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or abs(value) > MAX_EXACT_SCORE_ABS
    ):
        raise IntrinsicContractError("scorer_value_invalid")
    return value


def _deterministic_raw_scores(
    query: NarrativeExtraction,
    candidates: tuple[NarrativeExtraction, NarrativeExtraction],
    scorer: RawTextScorer,
) -> tuple[int, int] | None:
    rows: list[int] = []
    try:
        for candidate in candidates:
            first = _exact_score(
                scorer(query.source.utf8_bytes, candidate.source.utf8_bytes)
            )
            second = _exact_score(
                scorer(query.source.utf8_bytes, candidate.source.utf8_bytes)
            )
            if first != second:
                return None
            rows.append(first)
    except Exception:
        return None
    return rows[0], rows[1]


def _normalize_vector(
    value: object, expected_dimensions: int
) -> tuple[int, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != expected_dimensions
    ):
        raise IntrinsicContractError("legacy_vector_shape_invalid")
    result: list[int] = []
    for item in value:
        if (
            not isinstance(item, int)
            or isinstance(item, bool)
            or item < 0
            or item > MAX_LEGACY_VALUE
        ):
            raise IntrinsicContractError("legacy_vector_value_invalid")
        result.append(item)
    return tuple(result)


def _deterministic_legacy_vectors(
    query: NarrativeExtraction,
    candidates: tuple[NarrativeExtraction, NarrativeExtraction],
    vectorizer: LegacyVectorizer,
    feature_ids: tuple[str, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None:
    rows: list[tuple[int, ...]] = []
    try:
        for extraction in (query, *candidates):
            first = _normalize_vector(
                vectorizer(extraction, feature_ids), len(feature_ids)
            )
            second = _normalize_vector(
                vectorizer(extraction, feature_ids), len(feature_ids)
            )
            if first != second:
                return None
            rows.append(first)
    except Exception:
        return None
    return rows[0], rows[1], rows[2]


def _cosine_order(
    query: tuple[int, ...],
    first: tuple[int, ...],
    second: tuple[int, ...],
) -> int | None:
    query_norm = sum(value * value for value in query)
    first_norm = sum(value * value for value in first)
    second_norm = sum(value * value for value in second)
    if query_norm == 0 or first_norm == 0 or second_norm == 0:
        return None
    first_dot = sum(left * right for left, right in zip(query, first))
    second_dot = sum(left * right for left, right in zip(query, second))
    # Feature vectors are nonnegative.  Squared cosine therefore preserves
    # order and permits an exact integer cross-product comparison.
    left = first_dot * first_dot * second_norm
    right = second_dot * second_dot * first_norm
    if left == right:
        return None
    return 0 if left > right else 1


def prepare_structural_candidates(
    query: NarrativeExtraction,
    candidates: tuple[NarrativeExtraction, NarrativeExtraction],
    structural_scorer: StructuralSemanticScorer,
    *,
    mapping_config: MappingSearchConfig,
) -> tuple[
    tuple[PreparedStructuralCandidate, ...],
    tuple[CandidateProposalReceipt, ...],
]:
    """Generate each pair's sole proposal set for both structural arms."""

    prepared: list[PreparedStructuralCandidate] = []
    receipts: list[CandidateProposalReceipt] = []
    config_commitment = _content_hash(_mapping_config_payload(mapping_config))
    for ordinal, candidate in enumerate(candidates):
        base_search_payload = {
            "query_extraction_hash": query.extraction_hash,
            "query_provenance_hash": query.provenance_hash,
            "candidate_extraction_hash": candidate.extraction_hash,
            "candidate_provenance_hash": candidate.provenance_hash,
            "mapping_config": config_commitment,
        }
        try:
            first_scores = structural_scorer(query, candidate)
            second_scores = structural_scorer(query, candidate)
            if (
                not isinstance(first_scores, SemanticScoreTable)
                or not isinstance(second_scores, SemanticScoreTable)
                or first_scores.safe_payload() != second_scores.safe_payload()
            ):
                raise IntrinsicContractError("structural_scorer_nondeterministic")
            score_commitment = _content_hash(first_scores.safe_payload())
            search = generate_pair_mapping_proposals(
                query,
                candidate,
                first_scores,
                config=mapping_config,
            )
        except Exception:
            receipts.append(
                CandidateProposalReceipt(
                    candidate_ordinal=ordinal,
                    candidate_extraction_hash=candidate.extraction_hash,
                    candidate_provenance_hash=candidate.provenance_hash,
                    score_table_commitment=None,
                    flat_proposal_set_hash=None,
                    full_proposal_set_hash=None,
                    flat_choice_commitment=None,
                    full_choice_commitment=None,
                    search_commitment=_content_hash(
                        {**base_search_payload, "status": "scorer_invalid"}
                    ),
                    status="scorer_invalid",
                )
            )
            continue

        flat = choose_flat_arm(search)
        status = "complete"
        if search.budget_exhausted:
            status = "mapping_budget_exhausted"
        elif not search.proposals:
            status = "mapping_empty"
        prepared.append(
            PreparedStructuralCandidate(
                candidate_ordinal=ordinal,
                candidate=candidate,
                score_table_commitment=score_commitment,
                search_result=search,
                flat_choice=flat,
            )
        )
        # Full is intentionally not called in preparation.  Both fields bind
        # to the same already-materialized proposal set.
        receipts.append(
            CandidateProposalReceipt(
                candidate_ordinal=ordinal,
                candidate_extraction_hash=candidate.extraction_hash,
                candidate_provenance_hash=candidate.provenance_hash,
                score_table_commitment=score_commitment,
                flat_proposal_set_hash=search.proposal_set_hash,
                full_proposal_set_hash=search.proposal_set_hash,
                flat_choice_commitment=flat.choice_hash,
                full_choice_commitment=None,
                search_commitment=_content_hash(
                    {
                        **base_search_payload,
                        "status": status,
                        "proposal_set_hash": search.proposal_set_hash,
                        "assignments_explored": search.assignments_explored,
                        "budget_exhausted": search.budget_exhausted,
                        "reason_ids": list(search.reason_ids),
                    }
                ),
                status=status,
            )
        )
    return tuple(prepared), tuple(receipts)


def select_flat_prediction(
    prepared: tuple[PreparedStructuralCandidate, ...],
    *,
    input_commitment: str,
) -> ArmPrediction:
    """Compare shared proposals without any consistency-checker call."""

    if len(prepared) != 2 or {
        item.candidate_ordinal for item in prepared
    } != {0, 1}:
        return _abstain(
            IntrinsicArm.FLAT,
            input_commitment,
            "structural_pair_invalid",
            {"prepared_count": len(prepared)},
        )
    ordered = tuple(sorted(prepared, key=lambda item: item.candidate_ordinal))
    if any(
        item.search_result.budget_exhausted
        or not item.search_result.proposals
        or item.flat_choice.disposition is ChoiceDisposition.ABSTAIN
        for item in ordered
    ):
        return _abstain(
            IntrinsicArm.FLAT,
            input_commitment,
            "flat_candidate_abstained",
            {
                "proposal_set_hashes": [
                    item.search_result.proposal_set_hash for item in ordered
                ],
                "flat_choice_hashes": [
                    item.flat_choice.choice_hash for item in ordered
                ],
            },
        )
    selected: list[PairMappingProposal] = []
    for item in ordered:
        proposal = next(
            (
                proposal
                for proposal in item.search_result.proposals
                if proposal.proposal_hash
                == item.flat_choice.selected_proposal_hash
            ),
            None,
        )
        if proposal is None:
            return _abstain(
                IntrinsicArm.FLAT,
                input_commitment,
                "flat_selection_commitment_invalid",
                {
                    "proposal_set_hashes": [
                        entry.search_result.proposal_set_hash
                        for entry in ordered
                    ]
                },
            )
        selected.append(proposal)
    scores = (
        selected[0].semantic_score_micros,
        selected[1].semantic_score_micros,
    )
    evidence = {
        "proposal_set_hashes": [
            item.search_result.proposal_set_hash for item in ordered
        ],
        "flat_choice_hashes": [
            item.flat_choice.choice_hash for item in ordered
        ],
        "selected_proposal_hashes": [
            proposal.proposal_hash for proposal in selected
        ],
        "score_commitment": _content_hash(list(scores)),
    }
    if scores[0] == scores[1]:
        return _abstain(
            IntrinsicArm.FLAT,
            input_commitment,
            "flat_item_exact_tie",
            evidence,
        )
    return _predicted(
        IntrinsicArm.FLAT,
        input_commitment,
        0 if scores[0] > scores[1] else 1,
        evidence,
    )


def select_full_prediction(
    query: NarrativeExtraction,
    prepared: tuple[PreparedStructuralCandidate, ...],
    *,
    input_commitment: str,
) -> tuple[ArmPrediction, Mapping[int, ArmChoice]]:
    """Compare candidates only by independently verified exact tuples."""

    if len(prepared) != 2 or {
        item.candidate_ordinal for item in prepared
    } != {0, 1}:
        return (
            _abstain(
                IntrinsicArm.FULL,
                input_commitment,
                "structural_pair_invalid",
                {"prepared_count": len(prepared)},
            ),
            {},
        )
    ordered = tuple(sorted(prepared, key=lambda item: item.candidate_ordinal))
    choices: dict[int, ArmChoice] = {}
    for item in ordered:
        if item.search_result.budget_exhausted or not item.search_result.proposals:
            return (
                _abstain(
                    IntrinsicArm.FULL,
                    input_commitment,
                    "full_candidate_invalid",
                    {
                        "proposal_set_hashes": [
                            entry.search_result.proposal_set_hash
                            for entry in ordered
                        ]
                    },
                ),
                choices,
            )
        choices[item.candidate_ordinal] = choose_full_arm(
            query,
            item.candidate,
            item.search_result,
        )
    if any(
        choice.disposition is ChoiceDisposition.ABSTAIN
        or choice.certificate is None
        for choice in choices.values()
    ):
        return (
            _abstain(
                IntrinsicArm.FULL,
                input_commitment,
                "full_candidate_abstained",
                {
                    "full_choice_hashes": [
                        choices[index].choice_hash for index in sorted(choices)
                    ]
                },
            ),
            choices,
        )
    scores = (
        choices[0].certificate.lexicographic_score,
        choices[1].certificate.lexicographic_score,
    )
    evidence = {
        "proposal_set_hashes": [
            item.search_result.proposal_set_hash for item in ordered
        ],
        "full_choice_hashes": [
            choices[index].choice_hash for index in (0, 1)
        ],
        "certificate_hashes": [
            choices[index].certificate.certificate_hash
            for index in (0, 1)
        ],
        "lexicographic_score_commitment": _content_hash(
            [list(score) for score in scores]
        ),
    }
    if scores[0] == scores[1]:
        return (
            _abstain(
                IntrinsicArm.FULL,
                input_commitment,
                "full_item_exact_tie",
                evidence,
            ),
            choices,
        )
    return (
        _predicted(
            IntrinsicArm.FULL,
            input_commitment,
            0 if scores[0] < scores[1] else 1,
            evidence,
        ),
        choices,
    )


def evaluate_intrinsic_item(
    *,
    opaque_item_id: str,
    query: NarrativeExtraction,
    candidates: tuple[NarrativeExtraction, NarrativeExtraction],
    raw_text_scorer: RawTextScorer,
    legacy_vectorizer: LegacyVectorizer,
    legacy_feature_ids: tuple[str, ...],
    structural_scorer: StructuralSemanticScorer,
    mapping_config: MappingSearchConfig,
    raw_text_scorer_commitment: str,
    legacy_vectorizer_commitment: str,
    structural_scorer_commitment: str,
) -> IntrinsicItemResult:
    """Qualification-only injected four-arm path without reference data.

    Formal execution uses :func:`evaluate_frozen_intrinsic_item`; arbitrary
    callables and caller-declared commitments are not formal provenance.
    """

    _require_sha256(opaque_item_id, "opaque_item_id_invalid")
    if (
        not isinstance(query, NarrativeExtraction)
        or not isinstance(candidates, tuple)
        or len(candidates) != 2
        or any(
            not isinstance(candidate, NarrativeExtraction)
            for candidate in candidates
        )
    ):
        raise IntrinsicContractError("intrinsic_inputs_invalid")
    if (
        candidates[0].extraction_hash == candidates[1].extraction_hash
        or candidates[0].source.source_sha256
        == candidates[1].source.source_sha256
        or query.extraction_hash
        in {candidate.extraction_hash for candidate in candidates}
    ):
        raise IntrinsicContractError("candidate_independence_invalid")
    if (
        not callable(raw_text_scorer)
        or not callable(legacy_vectorizer)
        or not callable(structural_scorer)
    ):
        raise IntrinsicContractError("scorer_callable_invalid")
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
        _require_sha256(value, issue)
    if (
        not isinstance(legacy_feature_ids, tuple)
        or not 1 <= len(legacy_feature_ids) <= MAX_LEGACY_FEATURES
        or len(set(legacy_feature_ids)) != len(legacy_feature_ids)
        or any(
            not isinstance(item, str)
            or _FEATURE_ID.fullmatch(item) is None
            for item in legacy_feature_ids
        )
    ):
        raise IntrinsicContractError("legacy_feature_registry_invalid")
    if not isinstance(mapping_config, MappingSearchConfig):
        raise IntrinsicContractError("mapping_config_invalid")

    input_commitment = _input_commitment(
        opaque_item_id, query, candidates
    )
    raw_scores = _deterministic_raw_scores(
        query, candidates, raw_text_scorer
    )
    if raw_scores is None:
        semantic_prediction = _abstain(
            IntrinsicArm.SEMANTIC_ONLY,
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
            "score_commitment": _content_hash(list(raw_scores)),
        }
        if raw_scores[0] == raw_scores[1]:
            semantic_prediction = _abstain(
                IntrinsicArm.SEMANTIC_ONLY,
                input_commitment,
                "semantic_only_exact_tie",
                raw_evidence,
            )
        else:
            semantic_prediction = _predicted(
                IntrinsicArm.SEMANTIC_ONLY,
                input_commitment,
                0 if raw_scores[0] > raw_scores[1] else 1,
                raw_evidence,
            )

    legacy_registry_commitment = _content_hash(
        list(legacy_feature_ids)
    )
    legacy_vectors = _deterministic_legacy_vectors(
        query, candidates, legacy_vectorizer, legacy_feature_ids
    )
    if legacy_vectors is None:
        legacy_prediction = _abstain(
            IntrinsicArm.LEGACY,
            input_commitment,
            "legacy_vectorizer_invalid",
            {
                "vectorizer_commitment": legacy_vectorizer_commitment,
                "registry_commitment": legacy_registry_commitment,
            },
        )
    else:
        legacy_ordinal = _cosine_order(
            legacy_vectors[0], legacy_vectors[1], legacy_vectors[2]
        )
        legacy_evidence = {
            "vectorizer_commitment": legacy_vectorizer_commitment,
            "registry_commitment": legacy_registry_commitment,
            "vector_commitment": _content_hash(
                [list(vector) for vector in legacy_vectors]
            ),
        }
        if legacy_ordinal is None:
            legacy_prediction = _abstain(
                IntrinsicArm.LEGACY,
                input_commitment,
                "legacy_exact_tie_or_zero_vector",
                legacy_evidence,
            )
        else:
            legacy_prediction = _predicted(
                IntrinsicArm.LEGACY,
                input_commitment,
                legacy_ordinal,
                legacy_evidence,
            )

    prepared, initial_receipts = prepare_structural_candidates(
        query,
        candidates,
        structural_scorer,
        mapping_config=mapping_config,
    )
    flat_prediction = select_flat_prediction(
        prepared, input_commitment=input_commitment
    )
    full_prediction, full_choices = select_full_prediction(
        query,
        prepared,
        input_commitment=input_commitment,
    )
    receipts: list[CandidateProposalReceipt] = []
    for receipt in initial_receipts:
        choice = full_choices.get(receipt.candidate_ordinal)
        receipts.append(
            CandidateProposalReceipt(
                candidate_ordinal=receipt.candidate_ordinal,
                candidate_extraction_hash=receipt.candidate_extraction_hash,
                candidate_provenance_hash=receipt.candidate_provenance_hash,
                score_table_commitment=receipt.score_table_commitment,
                flat_proposal_set_hash=receipt.flat_proposal_set_hash,
                full_proposal_set_hash=receipt.full_proposal_set_hash,
                flat_choice_commitment=receipt.flat_choice_commitment,
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
                "legacy_vectorizer": legacy_vectorizer_commitment,
                "structural_scorer": structural_scorer_commitment,
                "proposal_consistency_checker": FIXED_CHECKER_COMMITMENT,
                "legacy_registry": legacy_registry_commitment,
                "mapping_config": _content_hash(
                    _mapping_config_payload(mapping_config)
                ),
            }.items()
        )
    )
    return IntrinsicItemResult(
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


def evaluate_frozen_intrinsic_item(
    *,
    opaque_item_id: str,
    query: NarrativeExtraction,
    candidates: tuple[NarrativeExtraction, NarrativeExtraction],
    scorers: object,
) -> IntrinsicItemResult:
    """Authoritative factory for the trusted formal-supervisor process.

    The function accepts no caller-selected lane callable, implementation
    commitment, mapping result, prepared candidate, or prediction.  Concrete
    scorers, the fixed mapping configuration, proposal generation, and both
    structural choices are constructed and consumed within this call.
    """

    from .gscl_arn_intrinsic_scorers_v1 import (  # noqa: PLC0415
        FrozenNarrativeScorers,
        LEGACY_FEATURE_IDS,
        SCORER_CONTRACT_HASH,
    )

    if type(scorers) is not FrozenNarrativeScorers:
        raise IntrinsicContractError("frozen_scorer_bundle_invalid")
    try:
        scorers.validate_internal()
    except Exception as exc:
        raise IntrinsicContractError(
            "frozen_scorer_bundle_invalid"
        ) from exc
    if (
        scorers.receipt.get("construction_domain")
        != "formal_exact_gscl_target_local_portable_minilm_v1"
    ):
        raise IntrinsicContractError(
            "frozen_scorer_not_formal"
        )
    scorer_receipt_hash = scorers.receipt.get("self_hash")
    _require_sha256(
        scorer_receipt_hash, "frozen_scorer_receipt_invalid"
    )
    mapping_config = MappingSearchConfig()

    def lane_commitment(lane: str) -> str:
        return _content_hash(
            {
                "lane": lane,
                "scorer_contract_hash": SCORER_CONTRACT_HASH,
                "scorer_receipt_hash": scorer_receipt_hash,
            }
        )

    return evaluate_intrinsic_item(
        opaque_item_id=opaque_item_id,
        query=query,
        candidates=candidates,
        raw_text_scorer=scorers.raw_text_scorer,
        legacy_vectorizer=scorers.legacy_vectorizer,
        legacy_feature_ids=LEGACY_FEATURE_IDS,
        structural_scorer=scorers.structural_scorer,
        mapping_config=mapping_config,
        raw_text_scorer_commitment=lane_commitment("semantic_only"),
        legacy_vectorizer_commitment=lane_commitment("legacy_keyword"),
        structural_scorer_commitment=lane_commitment(
            "structural_semantic_proposal"
        ),
    )


__all__ = [
    "ARMS_CORE_VERSION",
    "ArmPrediction",
    "CandidateProposalReceipt",
    "IntrinsicArm",
    "IntrinsicContractError",
    "IntrinsicItemResult",
    "LegacyVectorizer",
    "PredictionDisposition",
    "PreparedStructuralCandidate",
    "RawTextScorer",
    "StructuralSemanticScorer",
    "evaluate_frozen_intrinsic_item",
    "evaluate_intrinsic_item",
    "prepare_structural_candidates",
    "select_flat_prediction",
    "select_full_prediction",
]
