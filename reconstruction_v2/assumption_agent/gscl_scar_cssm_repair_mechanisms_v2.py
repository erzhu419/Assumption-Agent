"""Pure mechanism primitives for the same-study SCAR repair development.

This module has no filesystem, network, model, source, label, or scorer I/O.
It only transforms caller-supplied, already archived values.  In particular,
the null package never generates a candidate: it validates the fixed archived
proposal set, transforms only the target graph, and recomputes four structural
features for those same proposal hashes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import math
import re
from typing import Any, Mapping, Sequence

from assumption_agent.gscl_scar_cssm_repair_contract_v2 import content_hash


VERSION = "gscl_scar_cssm_repair_mechanisms_v2"
FOLD_COUNT = 5
NULL_REPLICATE_COUNT = 32
FOLD_HASH_DOMAIN = (
    "ASSUMPTION_AGENT/GSCL_SCAR_CSSM/"
    "SAME_STUDY_REPAIR_FOLD_ASSIGNMENT/V2"
)
NULL_RELATION_ORDER_HASH_DOMAIN = (
    "ASSUMPTION_AGENT/GSCL_SCAR_CSSM/"
    "U1_NULL_PACKAGE_RELATION_ORDER/V2"
)
NULL_STAGES = ("COLOR", "ROLE", "SIGN")

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_OPERATOR = re.compile(
    r"ori_(?P<orientation>keep|inv)\."
    r"pol_(?P<polarity>keep|inv)\."
    r"slots_(?P<slots>identity|reverse)\Z"
)
_DOMAIN_RELATIONS = frozenset({"cross_domain", "intra_domain"})
_KINDS = frozenset({"relation", "state_change", "temporal", "causal"})
_POLARITIES = frozenset({"negative", "neutral", "positive"})
_ORIENTATIONS = frozenset({"none", "forward", "reverse"})
_GRAPH_KEYS = frozenset(
    {
        "slots",
        "relations",
        "coverage_complete",
        "extractor_binding_sha256",
        "graph_evidence_binding_sha256",
        "receipt",
    }
)
_SLOT_KEYS = frozenset(
    {"slot_id", "normalized_label_sha256", "evidence_binding_sha256"}
)
_RELATION_KEYS = frozenset(
    {
        "relation_id",
        "slot0_id",
        "slot1_id",
        "generator_kind",
        "polarity",
        "temporal_orientation",
        "causal_orientation",
        "evidence_binding_sha256",
    }
)
_PROPOSAL_KEYS = frozenset(
    {
        "flat_structural_score",
        "injective_verified",
        "length2_composition_verified",
        "length2_path_matched",
        "length2_path_total",
        "operator_id",
        "origins",
        "semantic_score",
        "target_indices",
        "typed_incidence_matched",
        "typed_incidence_total",
        "typed_incidence_verified",
        "proposal_hash",
    }
)
_ORIGINS = frozenset({"semantic_kbest", "structure_kbest"})
_MAX_SLOTS = 16
_MAX_RELATIONS = 64
_MAX_SCORE_ABS = 1_000_000_000
_MAX_DIAGNOSTIC_COUNT = 4_096


class ScarRepairMechanismError(ValueError):
    """Stable, content-free rejection from the pure mechanism boundary."""

    _KNOWN = frozenset(
        {
            "SCAR_REPAIR_CANDIDATE_RANK_INVALID",
            "SCAR_REPAIR_FOLD_WIRE_INVALID",
            "SCAR_REPAIR_NULL_ARCHIVE_MISMATCH",
            "SCAR_REPAIR_NULL_GRAPH_INVALID",
            "SCAR_REPAIR_NULL_PROPOSAL_INVALID",
            "SCAR_REPAIR_NULL_TRANSFORM_INVALID",
        }
    )

    def __init__(self, issue_id: str) -> None:
        if issue_id not in self._KNOWN:
            raise ValueError("scar_repair_mechanism_issue_unknown")
        self.issue_id = issue_id
        super().__init__(issue_id)


def _is_hash(value: object) -> bool:
    return type(value) is str and _HEX64.fullmatch(value) is not None


@dataclass(frozen=True, slots=True)
class StratifiedFoldRow:
    canonical_item_id: str
    domain_relation: str
    arity: int


@dataclass(frozen=True, slots=True)
class FoldAssignment:
    canonical_item_id: str
    stratum: str
    assignment_digest: str
    fold_index: int


def _arity_bucket(arity: object) -> str:
    if type(arity) is not int or not 2 <= arity <= 14:
        raise ScarRepairMechanismError("SCAR_REPAIR_FOLD_WIRE_INVALID")
    if arity <= 4:
        return f"ARITY_{arity}"
    return "ARITY_5_PLUS"


def fold_assignment_digest(
    canonical_item_id: str, *, formal_result_self_sha256: str
) -> str:
    """Return the exact frozen digest body for one canonical item group."""

    if (
        type(canonical_item_id) is not str
        or not canonical_item_id
        or not _is_hash(formal_result_self_sha256)
    ):
        raise ScarRepairMechanismError("SCAR_REPAIR_FOLD_WIRE_INVALID")
    return content_hash(
        {
            "canonical_item_id": canonical_item_id,
            "formal_result_self_sha256": formal_result_self_sha256,
            "hash_domain": FOLD_HASH_DOMAIN,
        }
    )


def assign_stratified_folds(
    rows: Sequence[StratifiedFoldRow],
    *,
    formal_result_self_sha256: str,
) -> tuple[FoldAssignment, ...]:
    """Hash-sort item groups within domain-relation x arity strata modulo 5.

    Duplicate rows represent multiple views of one item group.  They receive
    the same fold when their stratum metadata agrees; conflicting duplicates
    fail closed.
    """

    if (
        isinstance(rows, (str, bytes))
        or not isinstance(rows, Sequence)
        or not rows
        or not _is_hash(formal_result_self_sha256)
    ):
        raise ScarRepairMechanismError("SCAR_REPAIR_FOLD_WIRE_INVALID")

    unique: dict[str, tuple[str, str]] = {}
    normalized: list[tuple[str, str, str]] = []
    for row in rows:
        if (
            not isinstance(row, StratifiedFoldRow)
            or type(row.canonical_item_id) is not str
            or not row.canonical_item_id
            or row.domain_relation not in _DOMAIN_RELATIONS
        ):
            raise ScarRepairMechanismError("SCAR_REPAIR_FOLD_WIRE_INVALID")
        bucket = _arity_bucket(row.arity)
        stratum = f"{row.domain_relation}|{bucket}"
        digest = fold_assignment_digest(
            row.canonical_item_id,
            formal_result_self_sha256=formal_result_self_sha256,
        )
        prior = unique.setdefault(row.canonical_item_id, (stratum, digest))
        if prior != (stratum, digest):
            raise ScarRepairMechanismError("SCAR_REPAIR_FOLD_WIRE_INVALID")
        normalized.append((row.canonical_item_id, stratum, digest))

    by_stratum: dict[str, list[tuple[str, str]]] = {}
    for item_id, (stratum, digest) in unique.items():
        by_stratum.setdefault(stratum, []).append((digest, item_id))
    fold_by_item: dict[str, int] = {}
    for members in by_stratum.values():
        for ordinal, (_, item_id) in enumerate(sorted(members)):
            fold_by_item[item_id] = ordinal % FOLD_COUNT

    return tuple(
        FoldAssignment(
            canonical_item_id=item_id,
            stratum=stratum,
            assignment_digest=digest,
            fold_index=fold_by_item[item_id],
        )
        for item_id, stratum, digest in normalized
    )


@dataclass(frozen=True, slots=True)
class CandidateRankInput:
    payload: object
    predicted_delta: float
    semantic_score: int
    proposal_hash: str


def rank_candidates(
    candidates: Sequence[CandidateRankInput],
) -> tuple[CandidateRankInput, ...]:
    """Rank by clipped delta desc, semantic score desc, proposal hash asc."""

    if (
        isinstance(candidates, (str, bytes))
        or not isinstance(candidates, Sequence)
        or not candidates
    ):
        raise ScarRepairMechanismError("SCAR_REPAIR_CANDIDATE_RANK_INVALID")
    hashes: set[str] = set()
    for row in candidates:
        if (
            not isinstance(row, CandidateRankInput)
            or type(row.predicted_delta) not in {int, float}
            or isinstance(row.predicted_delta, bool)
            or not math.isfinite(float(row.predicted_delta))
            or type(row.semantic_score) is not int
            or isinstance(row.semantic_score, bool)
            or abs(row.semantic_score) > _MAX_SCORE_ABS
            or not _is_hash(row.proposal_hash)
            or row.proposal_hash in hashes
        ):
            raise ScarRepairMechanismError(
                "SCAR_REPAIR_CANDIDATE_RANK_INVALID"
            )
        hashes.add(row.proposal_hash)

    def key(row: CandidateRankInput) -> tuple[float, int, str]:
        clipped = max(-1.0, min(1.0, float(row.predicted_delta)))
        return (-clipped, -row.semantic_score, row.proposal_hash)

    return tuple(sorted(candidates, key=key))


@dataclass(frozen=True, slots=True)
class StageRelationOrder:
    relation_id: str
    digest: str


@dataclass(frozen=True, slots=True)
class StructuralFeatureValues:
    flat_structural_score: int
    typed_incidence_matched: int
    typed_incidence_total: int
    f04_flat_structural_score_per_slot: Fraction
    f05_typed_incidence_match_rate: Fraction
    f06_typed_incidence_total_per_slot: Fraction
    f07_zero_incidence_support: Fraction


@dataclass(frozen=True, slots=True)
class NullProposalMean:
    proposal_hash: str
    f04_flat_structural_score_per_slot: Fraction
    f05_typed_incidence_match_rate: Fraction
    f06_typed_incidence_total_per_slot: Fraction
    f07_zero_incidence_support: Fraction


@dataclass(frozen=True, slots=True)
class _Relation:
    relation_id: str
    slot0_id: str
    slot1_id: str
    generator_kind: str
    polarity: str
    temporal_orientation: str
    causal_orientation: str
    evidence_binding_sha256: str

    def as_dict(self) -> dict[str, str]:
        return {
            "relation_id": self.relation_id,
            "slot0_id": self.slot0_id,
            "slot1_id": self.slot1_id,
            "generator_kind": self.generator_kind,
            "polarity": self.polarity,
            "temporal_orientation": self.temporal_orientation,
            "causal_orientation": self.causal_orientation,
            "evidence_binding_sha256": self.evidence_binding_sha256,
        }


@dataclass(frozen=True, slots=True)
class _Graph:
    slot_ids: tuple[str, ...]
    relations: tuple[_Relation, ...]


@dataclass(frozen=True, slots=True)
class _Edge:
    slot0: int
    slot1: int
    color: tuple[str, str, str, str]


def _normalize_graph(value: object) -> _Graph:
    issue = "SCAR_REPAIR_NULL_GRAPH_INVALID"
    if type(value) is not dict or set(value) != _GRAPH_KEYS:
        raise ScarRepairMechanismError(issue)
    if (
        type(value["slots"]) is not list
        or not 1 <= len(value["slots"]) <= _MAX_SLOTS
        or type(value["relations"]) is not list
        or len(value["relations"]) > _MAX_RELATIONS
        or type(value["coverage_complete"]) is not bool
        or value["coverage_complete"] is not False
        or not _is_hash(value["extractor_binding_sha256"])
        or not _is_hash(value["graph_evidence_binding_sha256"])
        or type(value["receipt"]) is not dict
    ):
        raise ScarRepairMechanismError(issue)

    slots: list[str] = []
    for slot in value["slots"]:
        if (
            type(slot) is not dict
            or set(slot) != _SLOT_KEYS
            or type(slot["slot_id"]) is not str
            or not slot["slot_id"]
            or not _is_hash(slot["normalized_label_sha256"])
            or not _is_hash(slot["evidence_binding_sha256"])
        ):
            raise ScarRepairMechanismError(issue)
        slots.append(slot["slot_id"])
    if len(slots) != len(set(slots)):
        raise ScarRepairMechanismError(issue)
    slot_set = frozenset(slots)

    relations: list[_Relation] = []
    relation_ids: set[str] = set()
    for relation in value["relations"]:
        if (
            type(relation) is not dict
            or set(relation) != _RELATION_KEYS
            or type(relation["relation_id"]) is not str
            or not relation["relation_id"]
            or relation["relation_id"] in relation_ids
            or relation["slot0_id"] not in slot_set
            or relation["slot1_id"] not in slot_set
            or relation["generator_kind"] not in _KINDS
            or relation["polarity"] not in _POLARITIES
            or relation["temporal_orientation"] not in _ORIENTATIONS
            or relation["causal_orientation"] not in _ORIENTATIONS
            or not _is_hash(relation["evidence_binding_sha256"])
        ):
            raise ScarRepairMechanismError(issue)
        relation_ids.add(relation["relation_id"])
        relations.append(_Relation(**relation))
    return _Graph(tuple(slots), tuple(relations))


def _normalize_item_and_replicate(item_token: object, replicate_index: object) -> None:
    if (
        type(item_token) is not str
        or not item_token
        or type(replicate_index) is not int
        or not 0 <= replicate_index < NULL_REPLICATE_COUNT
    ):
        raise ScarRepairMechanismError("SCAR_REPAIR_NULL_TRANSFORM_INVALID")


def _stage_order(
    item_token: str,
    relations: Sequence[_Relation],
    *,
    replicate_index: int,
    stage: str,
) -> tuple[StageRelationOrder, ...]:
    _normalize_item_and_replicate(item_token, replicate_index)
    if stage not in NULL_STAGES:
        raise ScarRepairMechanismError("SCAR_REPAIR_NULL_TRANSFORM_INVALID")
    rows = tuple(
        StageRelationOrder(
            relation_id=relation.relation_id,
            digest=content_hash(
                {
                    "hash_domain": NULL_RELATION_ORDER_HASH_DOMAIN,
                    "item_token": item_token,
                    "relation_id": relation.relation_id,
                    "replicate_index": replicate_index,
                    "stage": stage,
                }
            ),
        )
        for relation in relations
    )
    return tuple(sorted(rows, key=lambda row: (row.digest, row.relation_id)))


def stage_relation_order(
    item_token: str,
    target_graph: Mapping[str, object],
    *,
    replicate_index: int,
    stage: str,
) -> tuple[StageRelationOrder, ...]:
    """Expose the exact stage-specific SHA-256 order for golden auditing."""

    graph = _normalize_graph(target_graph)
    return _stage_order(
        item_token,
        graph.relations,
        replicate_index=replicate_index,
        stage=stage,
    )


def _rotate_values(
    relations: dict[str, _Relation],
    order: tuple[StageRelationOrder, ...],
    fields: tuple[str, ...],
) -> None:
    if len(order) < 2:
        return
    values = [
        tuple(getattr(relations[row.relation_id], field) for field in fields)
        for row in order
    ]
    rotated = values[1:] + values[:1]
    for order_row, replacement in zip(order, rotated, strict=True):
        prior = relations[order_row.relation_id]
        body = prior.as_dict()
        body.update(dict(zip(fields, replacement, strict=True)))
        relations[order_row.relation_id] = _Relation(**body)


def _apply_null_transform_graph(
    item_token: str, graph: _Graph, *, replicate_index: int
) -> _Graph:
    _normalize_item_and_replicate(item_token, replicate_index)
    by_id = {row.relation_id: row for row in graph.relations}

    color_order = _stage_order(
        item_token,
        graph.relations,
        replicate_index=replicate_index,
        stage="COLOR",
    )
    _rotate_values(by_id, color_order, ("generator_kind",))

    role_order = _stage_order(
        item_token,
        graph.relations,
        replicate_index=replicate_index,
        stage="ROLE",
    )
    half_edges_per_replicate = (len(role_order) + 1) // 2
    for order_row in role_order[:half_edges_per_replicate]:
        prior = by_id[order_row.relation_id]
        body = prior.as_dict()
        body["slot0_id"], body["slot1_id"] = (
            body["slot1_id"],
            body["slot0_id"],
        )
        by_id[order_row.relation_id] = _Relation(**body)

    sign_order = _stage_order(
        item_token,
        graph.relations,
        replicate_index=replicate_index,
        stage="SIGN",
    )
    _rotate_values(
        by_id,
        sign_order,
        ("polarity", "temporal_orientation", "causal_orientation"),
    )
    return _Graph(
        graph.slot_ids,
        tuple(by_id[row.relation_id] for row in graph.relations),
    )


def apply_null_package_transform(
    item_token: str,
    target_graph: Mapping[str, object],
    *,
    replicate_index: int,
) -> tuple[dict[str, str], ...]:
    """Return transformed relation rows in the archived relation-list order."""

    graph = _normalize_graph(target_graph)
    transformed = _apply_null_transform_graph(
        item_token, graph, replicate_index=replicate_index
    )
    return tuple(row.as_dict() for row in transformed.relations)


def _normalize_proposal(value: object, *, arity: int) -> dict[str, object]:
    issue = "SCAR_REPAIR_NULL_PROPOSAL_INVALID"
    if type(value) is not dict or set(value) != _PROPOSAL_KEYS:
        raise ScarRepairMechanismError(issue)
    body = dict(value)
    proposal_hash = body.pop("proposal_hash")
    if (
        not _is_hash(proposal_hash)
        or proposal_hash != content_hash(body)
        or type(value["operator_id"]) is not str
        or _OPERATOR.fullmatch(value["operator_id"]) is None
        or type(value["origins"]) is not list
        or value["origins"] != sorted(set(value["origins"]))
        or not value["origins"]
        or not set(value["origins"]).issubset(_ORIGINS)
        or type(value["target_indices"]) is not list
        or sorted(value["target_indices"]) != list(range(arity))
        or any(
            type(value[key]) is not bool
            for key in (
                "injective_verified",
                "typed_incidence_verified",
                "length2_composition_verified",
            )
        )
        or value["injective_verified"] is not True
    ):
        raise ScarRepairMechanismError(issue)
    for key in (
        "flat_structural_score",
        "semantic_score",
        "length2_path_matched",
        "length2_path_total",
        "typed_incidence_matched",
        "typed_incidence_total",
    ):
        if type(value[key]) is not int or isinstance(value[key], bool):
            raise ScarRepairMechanismError(issue)
    if (
        abs(value["flat_structural_score"]) > _MAX_SCORE_ABS
        or abs(value["semantic_score"]) > _MAX_SCORE_ABS
        or not 0
        <= value["length2_path_matched"]
        <= value["length2_path_total"]
        <= _MAX_DIAGNOSTIC_COUNT
        or not 0
        <= value["typed_incidence_matched"]
        <= value["typed_incidence_total"]
        <= _MAX_DIAGNOSTIC_COUNT
    ):
        raise ScarRepairMechanismError(issue)
    return dict(value)


def _invert_orientation(value: str) -> str:
    return {"none": "none", "forward": "reverse", "reverse": "forward"}[
        value
    ]


def _edges(graph: _Graph, *, operator_id: str | None = None) -> tuple[_Edge, ...]:
    slot_indices = {slot_id: index for index, slot_id in enumerate(graph.slot_ids)}
    operator = None if operator_id is None else _OPERATOR.fullmatch(operator_id)
    if operator_id is not None and operator is None:
        raise ScarRepairMechanismError("SCAR_REPAIR_NULL_PROPOSAL_INVALID")
    result: list[_Edge] = []
    for relation in graph.relations:
        slot0 = slot_indices[relation.slot0_id]
        slot1 = slot_indices[relation.slot1_id]
        kind = relation.generator_kind
        polarity = relation.polarity
        temporal = relation.temporal_orientation
        causal = relation.causal_orientation
        if operator is not None:
            if operator.group("slots") == "reverse":
                slot0, slot1 = slot1, slot0
            if operator.group("polarity") == "inv":
                polarity = {
                    "negative": "positive",
                    "neutral": "neutral",
                    "positive": "negative",
                }[polarity]
            if operator.group("orientation") == "inv":
                temporal = _invert_orientation(temporal)
                causal = _invert_orientation(causal)
        result.append(_Edge(slot0, slot1, (kind, polarity, temporal, causal)))
    return tuple(result)


def _profiles(
    slot_count: int, edges: tuple[_Edge, ...]
) -> tuple[Counter[tuple[object, ...]], ...]:
    rows = [Counter() for _ in range(slot_count)]
    for edge in edges:
        rows[edge.slot0][("slot0", *edge.color)] += 1
        rows[edge.slot1][("slot1", *edge.color)] += 1
    return tuple(rows)


def _profile_score(
    source: Counter[tuple[object, ...]],
    target: Counter[tuple[object, ...]],
) -> int:
    overlap = sum((source & target).values())
    return 2 * overlap - sum(source.values()) - sum(target.values())


def _recompute_features(
    source_graph: _Graph, target_graph: _Graph, proposal: Mapping[str, object]
) -> StructuralFeatureValues:
    arity = len(source_graph.slot_ids)
    source_profiles = _profiles(
        arity, _edges(source_graph, operator_id=str(proposal["operator_id"]))
    )
    target_profiles = _profiles(arity, _edges(target_graph))
    assignment = tuple(int(row) for row in proposal["target_indices"])
    flat = sum(
        _profile_score(source_profiles[index], target_profiles[target_index])
        for index, target_index in enumerate(assignment)
    )
    matched = 0
    total = 0
    for index, target_index in enumerate(assignment):
        source_profile = source_profiles[index]
        target_profile = target_profiles[target_index]
        matched += sum((source_profile & target_profile).values())
        total += sum(source_profile.values())
    return StructuralFeatureValues(
        flat_structural_score=flat,
        typed_incidence_matched=matched,
        typed_incidence_total=total,
        f04_flat_structural_score_per_slot=Fraction(flat, arity),
        f05_typed_incidence_match_rate=(
            Fraction(matched, total) if total else Fraction(0)
        ),
        f06_typed_incidence_total_per_slot=Fraction(total, arity),
        f07_zero_incidence_support=Fraction(int(total == 0), 1),
    )


def recompute_structural_features(
    source_graph: Mapping[str, object],
    target_graph: Mapping[str, object],
    proposal: Mapping[str, object],
) -> StructuralFeatureValues:
    """Recompute f04-f07 with the archived categorical mapping semantics."""

    source = _normalize_graph(source_graph)
    target = _normalize_graph(target_graph)
    if len(source.slot_ids) != len(target.slot_ids):
        raise ScarRepairMechanismError("SCAR_REPAIR_NULL_GRAPH_INVALID")
    normalized_proposal = _normalize_proposal(proposal, arity=len(source.slot_ids))
    return _recompute_features(source, target, normalized_proposal)


def build_null_package_mean(
    item_token: str,
    source_graph: Mapping[str, object],
    target_graph: Mapping[str, object],
    proposals: Sequence[Mapping[str, object]],
) -> tuple[NullProposalMean, ...]:
    """Validate fixed candidates, apply 32 target nulls, and average f04-f07."""

    _normalize_item_and_replicate(item_token, 0)
    source = _normalize_graph(source_graph)
    target = _normalize_graph(target_graph)
    if len(source.slot_ids) != len(target.slot_ids):
        raise ScarRepairMechanismError("SCAR_REPAIR_NULL_GRAPH_INVALID")
    if isinstance(proposals, (str, bytes)) or not isinstance(proposals, Sequence):
        raise ScarRepairMechanismError("SCAR_REPAIR_NULL_PROPOSAL_INVALID")
    normalized = tuple(
        _normalize_proposal(row, arity=len(source.slot_ids)) for row in proposals
    )
    proposal_hashes = tuple(str(row["proposal_hash"]) for row in normalized)
    if len(proposal_hashes) != len(set(proposal_hashes)):
        raise ScarRepairMechanismError("SCAR_REPAIR_NULL_PROPOSAL_INVALID")

    for proposal in normalized:
        archived = _recompute_features(source, target, proposal)
        if (
            archived.flat_structural_score != proposal["flat_structural_score"]
            or archived.typed_incidence_matched
            != proposal["typed_incidence_matched"]
            or archived.typed_incidence_total != proposal["typed_incidence_total"]
        ):
            raise ScarRepairMechanismError(
                "SCAR_REPAIR_NULL_ARCHIVE_MISMATCH"
            )

    replicates = tuple(
        _apply_null_transform_graph(item_token, target, replicate_index=index)
        for index in range(NULL_REPLICATE_COUNT)
    )
    outputs: list[NullProposalMean] = []
    for proposal in sorted(normalized, key=lambda row: str(row["proposal_hash"])):
        rows = tuple(
            _recompute_features(source, transformed, proposal)
            for transformed in replicates
        )
        outputs.append(
            NullProposalMean(
                proposal_hash=str(proposal["proposal_hash"]),
                f04_flat_structural_score_per_slot=sum(
                    (row.f04_flat_structural_score_per_slot for row in rows),
                    Fraction(0),
                )
                / NULL_REPLICATE_COUNT,
                f05_typed_incidence_match_rate=sum(
                    (row.f05_typed_incidence_match_rate for row in rows),
                    Fraction(0),
                )
                / NULL_REPLICATE_COUNT,
                f06_typed_incidence_total_per_slot=sum(
                    (row.f06_typed_incidence_total_per_slot for row in rows),
                    Fraction(0),
                )
                / NULL_REPLICATE_COUNT,
                f07_zero_incidence_support=sum(
                    (row.f07_zero_incidence_support for row in rows),
                    Fraction(0),
                )
                / NULL_REPLICATE_COUNT,
            )
        )
    return tuple(outputs)


__all__ = [
    "CandidateRankInput",
    "FOLD_COUNT",
    "FOLD_HASH_DOMAIN",
    "FoldAssignment",
    "NULL_RELATION_ORDER_HASH_DOMAIN",
    "NULL_REPLICATE_COUNT",
    "NULL_STAGES",
    "NullProposalMean",
    "ScarRepairMechanismError",
    "StageRelationOrder",
    "StratifiedFoldRow",
    "StructuralFeatureValues",
    "VERSION",
    "apply_null_package_transform",
    "assign_stratified_folds",
    "build_null_package_mean",
    "fold_assignment_digest",
    "rank_candidates",
    "recompute_structural_features",
    "stage_relation_order",
]
