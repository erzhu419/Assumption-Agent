"""Deterministic typed-candidate and E4 evaluator core for BIRCO P1.

This module implements the label-free algebra frozen by
``birco_p1_typed_constraint_e4_study_design_v1``.  It deliberately has no
dataset reader, filesystem access, network client, model client, credential
handling, or retry path.  The only candidate identity admitted by the core is
the zero-based source ordinal within one already-authorized candidate pool.

The public surface is split into four parts:

* strict typed-facet DAG and candidate/facet/evidence schemas, together with
  deterministic one-pass totalizers for synthetic model output;
* the four complete-permutation recipes and the frozen E0 selector;
* the twelve content-free action features and a shared, recipe-ID-free linear
  listwise-softmax E4 model with lambda-one L2 and Laplace uncertainty; and
* offline linear-gain nDCG@10, Recall@5, integer utility, descriptive binomial
  reference tails, and the predeclared promotion decisions.

Score reports contain aggregate numbers only.  They never serialize candidate
ordinals, plan text, evidence, source IDs, query IDs, or document IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import itertools
import math
import re
import unicodedata
from typing import Any, Hashable, Mapping, Sequence

import numpy as np


VERSION = "birco_p1_typed_constraint_e4_core_v1"

FACET_TYPES = (
    "REQUIRED",
    "EXCLUDED",
    "PREFERRED",
    "ELIGIBILITY",
    "TEMPORAL",
    "RELATIONAL",
)
EDGE_TYPES = ("REQUIRES", "REFINES", "CONTRASTS_WITH")

# Deterministic defaults are used only by the invalid-output totalizer.  A
# strictly valid planner facet carries its own integer weight in [1, 4].
FACET_TYPE_DEFAULT_WEIGHTS: Mapping[str, int] = {
    "REQUIRED": 4,
    "EXCLUDED": 4,
    "PREFERRED": 1,
    "ELIGIBILITY": 4,
    "TEMPORAL": 2,
    "RELATIONAL": 2,
}

R1_WEIGHTED_MASS = "R1_WEIGHTED_MASS"
R2_BOTTLENECK = "R2_BOTTLENECK"
R3_DEPENDENCY_FLOW = "R3_DEPENDENCY_FLOW"
R4_CAPACITY_MATCH = "R4_CAPACITY_MATCH"
RECIPE_IDS = (
    R1_WEIGHTED_MASS,
    R2_BOTTLENECK,
    R3_DEPENDENCY_FLOW,
    R4_CAPACITY_MATCH,
)

FEATURE_ORDER = (
    "plan_facet_count",
    "required_facet_fraction",
    "exclusion_or_eligibility_fraction",
    "dependency_edge_fraction",
    "top10_mean_support",
    "top10_minimum_required_support",
    "top10_satisfied_facet_fraction",
    "top10_contradiction_negative",
    "top1_to_top2_margin",
    "score_entropy_negative",
    "top10_distinct_evidence_assignment_fraction",
    "single-facet-removal_rank_stability",
)

FORBIDDEN_E4_FEATURES = frozenset(
    {
        "recipe_id",
        "family",
        "family_id",
        "query_id",
        "candidate_id",
        "document_id",
        "document_text",
        "RAW_rank",
        "HippoRAG_rank",
        "qrel",
        "relevance",
    }
)

MIN_FACETS = 2
MAX_FACETS = 12
SUPPORT_MIN = 0
SUPPORT_MAX = 4
SATISFIED_SUPPORT_MIN = 1
NDCG_CUTOFF = 10
RECALL_CUTOFF = 5
RECALL_RELEVANCE_THRESHOLD = 1.0
INTEGER_UTILITY_SCALE = 1_000_000_000
E4_L2 = 1.0
E4_TARGET_TEMPERATURE = 1.0 / 20.0
E4_LAPLACE_PENALTY = 0.5
E4_MAX_ITER = 256
E4_A_FORM_ITEM_COUNT = 30
PROMOTION_ALPHA = Fraction(1, 10)

_PLAN_FIELDS = frozenset({"facets", "edges"})
_FACET_FIELDS = frozenset({"ordinal", "facet_type", "text", "weight"})
_FACET_SHORT_FIELDS = frozenset({"ordinal", "type", "text", "weight"})
_EDGE_FIELDS = frozenset(
    {"source_facet_ordinal", "target_facet_ordinal", "edge_type"}
)
_EDGE_SHORT_FIELDS = frozenset({"source", "target", "type"})
_MATRIX_FIELDS = frozenset({"candidates"})
_CANDIDATE_FIELDS = frozenset(
    {"candidate_ordinal", "evidence_unit_count", "facet_evidence"}
)
_EVIDENCE_FIELDS = frozenset(
    {"facet_ordinal", "support", "contradiction", "evidence_unit_ordinal"}
)
_WHITESPACE = re.compile(r"\s+", flags=re.UNICODE)


class BircoP1CoreError(ValueError):
    """Fail-closed error for malformed plans, matrices, slates, or scores."""


def _strict_int(value: object, field: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise BircoP1CoreError(f"{field} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise BircoP1CoreError(f"{field} must be at least {minimum}")
    return result


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise BircoP1CoreError(f"{field} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise BircoP1CoreError(f"{field} must be a finite real number")
    return 0.0 if result == 0.0 else result


def _canonical_text(value: object) -> str:
    if not isinstance(value, str):
        raise BircoP1CoreError("facet text must be a string")
    result = unicodedata.normalize("NFKC", value)
    result = _WHITESPACE.sub(" ", result).strip()
    if not result:
        raise BircoP1CoreError("facet text must be nonempty")
    if len(result) > 512:
        raise BircoP1CoreError("facet text exceeds 512 canonical characters")
    return result


def _require_exact_fields(value: Mapping[str, object], expected: frozenset[str], name: str) -> None:
    supplied = set(value)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise BircoP1CoreError(
            f"{name} schema drifted; missing={missing}, extra={extra}"
        )


@dataclass(frozen=True)
class TypedFacet:
    """One canonical planner facet.

    ``ordinal`` is local to a plan and must equal the facet's array position.
    ``weight`` is the frozen planner importance and must be an integer from one
    through four.
    """

    ordinal: int
    facet_type: str
    text: str
    weight: int

    def __post_init__(self) -> None:
        ordinal = _strict_int(self.ordinal, "facet ordinal", minimum=0)
        if not isinstance(self.facet_type, str) or self.facet_type not in FACET_TYPES:
            raise BircoP1CoreError("facet type is outside the frozen registry")
        text = _canonical_text(self.text)
        weight = _strict_int(self.weight, "facet weight", minimum=1)
        if weight > 4:
            raise BircoP1CoreError("facet weight must be an integer from 1 to 4")
        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "weight", weight)

    def payload(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "facet_type": self.facet_type,
            "text": self.text,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class TypedFacetEdge:
    """One directed edge in the facet DAG.

    For ``REQUIRES``, ``source_facet_ordinal`` is the prerequisite and
    ``target_facet_ordinal`` is the dependent facet.  The same orientation is
    retained for the other edge types so the complete typed graph is a DAG.
    """

    source_facet_ordinal: int
    target_facet_ordinal: int
    edge_type: str

    def __post_init__(self) -> None:
        source = _strict_int(
            self.source_facet_ordinal, "edge source facet ordinal", minimum=0
        )
        target = _strict_int(
            self.target_facet_ordinal, "edge target facet ordinal", minimum=0
        )
        if source == target:
            raise BircoP1CoreError("typed facet edges cannot be self-loops")
        if not isinstance(self.edge_type, str) or self.edge_type not in EDGE_TYPES:
            raise BircoP1CoreError("edge type is outside the frozen registry")
        object.__setattr__(self, "source_facet_ordinal", source)
        object.__setattr__(self, "target_facet_ordinal", target)

    def payload(self) -> dict[str, object]:
        return {
            "source_facet_ordinal": self.source_facet_ordinal,
            "target_facet_ordinal": self.target_facet_ordinal,
            "edge_type": self.edge_type,
        }


def _topological_ordinals(
    facet_ordinals: Sequence[int], edges: Sequence[TypedFacetEdge]
) -> tuple[int, ...]:
    nodes = tuple(facet_ordinals)
    node_set = set(nodes)
    incoming = {node: 0 for node in nodes}
    outgoing: dict[int, list[int]] = {node: [] for node in nodes}
    for edge in edges:
        if (
            edge.source_facet_ordinal not in node_set
            or edge.target_facet_ordinal not in node_set
        ):
            raise BircoP1CoreError("typed edge refers to a missing facet ordinal")
        outgoing[edge.source_facet_ordinal].append(edge.target_facet_ordinal)
        incoming[edge.target_facet_ordinal] += 1
    ready = sorted(node for node, count in incoming.items() if count == 0)
    order: list[int] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for target in sorted(outgoing[node]):
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
                ready.sort()
    if len(order) != len(nodes):
        raise BircoP1CoreError("typed facet graph must be acyclic")
    return tuple(order)


@dataclass(frozen=True)
class TypedFacetPlan:
    """Exact two-to-twelve-facet typed DAG emitted by the planner."""

    facets: tuple[TypedFacet, ...]
    edges: tuple[TypedFacetEdge, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.facets, (str, bytes)) or not isinstance(self.facets, Sequence):
            raise BircoP1CoreError("facets must be an array")
        facets = tuple(
            facet if isinstance(facet, TypedFacet) else _facet_from_mapping(facet)
            for facet in self.facets
        )
        if not MIN_FACETS <= len(facets) <= MAX_FACETS:
            raise BircoP1CoreError(
                f"typed plan must contain between {MIN_FACETS} and {MAX_FACETS} facets"
            )
        if tuple(facet.ordinal for facet in facets) != tuple(range(len(facets))):
            raise BircoP1CoreError(
                "facet ordinals must be canonical, contiguous, and match array order"
            )
        folded = tuple(facet.text.casefold() for facet in facets)
        if len(set(folded)) != len(folded):
            raise BircoP1CoreError("typed plan contains duplicate canonical facet text")

        if isinstance(self.edges, (str, bytes)) or not isinstance(self.edges, Sequence):
            raise BircoP1CoreError("edges must be an array")
        edges = tuple(
            edge if isinstance(edge, TypedFacetEdge) else _edge_from_mapping(edge)
            for edge in self.edges
        )
        canonical = tuple(
            sorted(
                edges,
                key=lambda edge: (
                    edge.source_facet_ordinal,
                    edge.target_facet_ordinal,
                    EDGE_TYPES.index(edge.edge_type),
                ),
            )
        )
        if edges != canonical:
            raise BircoP1CoreError("typed edges must be in canonical ordinal/type order")
        pairs = tuple(
            (edge.source_facet_ordinal, edge.target_facet_ordinal) for edge in edges
        )
        if len(set(pairs)) != len(pairs):
            raise BircoP1CoreError("a directed facet pair may carry only one typed edge")
        _topological_ordinals(tuple(range(len(facets))), edges)
        object.__setattr__(self, "facets", facets)
        object.__setattr__(self, "edges", edges)

    def payload(self) -> dict[str, object]:
        return {
            "facets": [facet.payload() for facet in self.facets],
            "edges": [edge.payload() for edge in self.edges],
        }


# Short aliases are convenient for controllers without weakening the schema.
Facet = TypedFacet
FacetEdge = TypedFacetEdge
FacetPlan = TypedFacetPlan


def _facet_from_mapping(value: object) -> TypedFacet:
    if not isinstance(value, Mapping):
        raise BircoP1CoreError("each facet must be a mapping or TypedFacet")
    supplied = set(value)
    if supplied == _FACET_FIELDS:
        facet_type = value["facet_type"]
    elif supplied == _FACET_SHORT_FIELDS:
        facet_type = value["type"]
    else:
        missing = sorted(_FACET_FIELDS - supplied)
        extra = sorted(supplied - _FACET_FIELDS)
        raise BircoP1CoreError(
            f"facet schema drifted; missing={missing}, extra={extra}"
        )
    return TypedFacet(
        ordinal=value["ordinal"],  # type: ignore[arg-type]
        facet_type=facet_type,  # type: ignore[arg-type]
        text=value["text"],  # type: ignore[arg-type]
        weight=value["weight"],  # type: ignore[arg-type]
    )


def _edge_from_mapping(value: object) -> TypedFacetEdge:
    if not isinstance(value, Mapping):
        raise BircoP1CoreError("each typed edge must be a mapping or TypedFacetEdge")
    supplied = set(value)
    if supplied == _EDGE_FIELDS:
        source = value["source_facet_ordinal"]
        target = value["target_facet_ordinal"]
        edge_type = value["edge_type"]
    elif supplied == _EDGE_SHORT_FIELDS:
        source = value["source"]
        target = value["target"]
        edge_type = value["type"]
    else:
        missing = sorted(_EDGE_FIELDS - supplied)
        extra = sorted(supplied - _EDGE_FIELDS)
        raise BircoP1CoreError(
            f"typed edge schema drifted; missing={missing}, extra={extra}"
        )
    return TypedFacetEdge(
        source_facet_ordinal=source,  # type: ignore[arg-type]
        target_facet_ordinal=target,  # type: ignore[arg-type]
        edge_type=edge_type,  # type: ignore[arg-type]
    )


def validate_typed_facet_plan(
    value: TypedFacetPlan | Mapping[str, object],
) -> TypedFacetPlan:
    """Validate the exact plan schema without silently repairing it."""

    if isinstance(value, TypedFacetPlan):
        return TypedFacetPlan(tuple(value.facets), tuple(value.edges))
    if not isinstance(value, Mapping):
        raise BircoP1CoreError("typed facet plan must be a mapping or TypedFacetPlan")
    _require_exact_fields(value, _PLAN_FIELDS, "typed facet plan")
    facets = value["facets"]
    edges = value["edges"]
    if isinstance(facets, (str, bytes)) or not isinstance(facets, Sequence):
        raise BircoP1CoreError("facets must be an array")
    if isinstance(edges, (str, bytes)) or not isinstance(edges, Sequence):
        raise BircoP1CoreError("edges must be an array")
    return TypedFacetPlan(
        tuple(_facet_from_mapping(row) for row in facets),
        tuple(_edge_from_mapping(row) for row in edges),
    )


validate_typed_plan = validate_typed_facet_plan


def _candidate_facet_type(value: object) -> str:
    if isinstance(value, str):
        result = value.strip().upper()
        if result in FACET_TYPES:
            return result
    return "REQUIRED"


def _would_be_acyclic(facet_count: int, edges: Sequence[TypedFacetEdge]) -> bool:
    try:
        _topological_ordinals(tuple(range(facet_count)), edges)
    except BircoP1CoreError:
        return False
    return True


def totalize_typed_facet_plan(
    candidate: object,
    *,
    fallback_clauses: object = (),
) -> TypedFacetPlan:
    """Return one valid typed DAG for every possible planner output.

    Valid facet rows are retained in input order, Unicode/whitespace is
    canonicalized, duplicates retain their first occurrence, and ordinals are
    reassigned contiguously.  Caller-supplied deterministic query clauses fill
    missing facets; two inert placeholders are the final totalizer.  Edges are
    retained only when their endpoints survive and adding them preserves the
    DAG.  There is no retry, resampling, source access, or model call.
    """

    if isinstance(candidate, TypedFacetPlan):
        return validate_typed_facet_plan(candidate)
    raw = candidate if isinstance(candidate, Mapping) else {}
    raw_facets = raw.get("facets", ())
    if isinstance(raw_facets, (str, bytes)) or not isinstance(raw_facets, Sequence):
        raw_facets = ()

    accepted: list[tuple[str, str, int, int | None]] = []
    seen_text: set[str] = set()
    old_to_new: dict[int, int] = {}

    def add(
        text_value: object,
        facet_type_value: object,
        weight_value: object,
        old_ordinal: object,
    ) -> None:
        if len(accepted) >= MAX_FACETS:
            return
        try:
            text = _canonical_text(text_value)
        except BircoP1CoreError:
            return
        folded = text.casefold()
        if folded in seen_text:
            return
        facet_type = _candidate_facet_type(facet_type_value)
        weight = (
            int(weight_value)
            if type(weight_value) is int and 1 <= weight_value <= 4
            else FACET_TYPE_DEFAULT_WEIGHTS[facet_type]
        )
        old = old_ordinal if type(old_ordinal) is int and old_ordinal >= 0 else None
        new = len(accepted)
        accepted.append((text, facet_type, weight, old))
        seen_text.add(folded)
        if old is not None and old not in old_to_new:
            old_to_new[old] = new

    for position, row in enumerate(raw_facets):
        if isinstance(row, Mapping):
            add(
                row.get("text"),
                row.get("facet_type", row.get("type")),
                row.get("weight"),
                row.get("ordinal", position),
            )
        elif isinstance(row, str):
            add(row, "REQUIRED", None, position)

    fallback_values: Sequence[object]
    if isinstance(fallback_clauses, (str, bytes)):
        fallback_values = (fallback_clauses,)
    elif isinstance(fallback_clauses, Sequence):
        fallback_values = fallback_clauses
    else:
        fallback_values = ()
    for row in fallback_values:
        if len(accepted) >= MIN_FACETS:
            break
        add(row, "REQUIRED", None, None)
    for placeholder in ("unspecified query facet", "unspecified query context"):
        if len(accepted) >= MIN_FACETS:
            break
        add(placeholder, "REQUIRED", None, None)

    facets = tuple(
        TypedFacet(index, facet_type, text, weight)
        for index, (text, facet_type, weight, _old) in enumerate(accepted[:MAX_FACETS])
    )

    raw_edges = raw.get("edges", ())
    if isinstance(raw_edges, (str, bytes)) or not isinstance(raw_edges, Sequence):
        raw_edges = ()
    retained: list[TypedFacetEdge] = []
    used_pairs: set[tuple[int, int]] = set()
    for row in raw_edges:
        if not isinstance(row, Mapping):
            continue
        source_old = row.get("source_facet_ordinal", row.get("source"))
        target_old = row.get("target_facet_ordinal", row.get("target"))
        edge_type_value = row.get("edge_type", row.get("type"))
        if (
            type(source_old) is not int
            or type(target_old) is not int
            or source_old not in old_to_new
            or target_old not in old_to_new
            or not isinstance(edge_type_value, str)
        ):
            continue
        edge_type = edge_type_value.strip().upper()
        if edge_type not in EDGE_TYPES:
            continue
        source = old_to_new[source_old]
        target = old_to_new[target_old]
        if source == target or (source, target) in used_pairs:
            continue
        edge = TypedFacetEdge(source, target, edge_type)
        proposed = tuple(retained) + (edge,)
        if not _would_be_acyclic(len(facets), proposed):
            continue
        retained.append(edge)
        used_pairs.add((source, target))
    edges = tuple(
        sorted(
            retained,
            key=lambda edge: (
                edge.source_facet_ordinal,
                edge.target_facet_ordinal,
                EDGE_TYPES.index(edge.edge_type),
            ),
        )
    )
    return TypedFacetPlan(facets, edges)


totalize_typed_plan = totalize_typed_facet_plan


@dataclass(frozen=True)
class FacetEvidence:
    """Quantized support/contradiction and at most one evidence-unit ordinal."""

    facet_ordinal: int
    support: int
    contradiction: int
    evidence_unit_ordinal: int | None

    def __post_init__(self) -> None:
        facet = _strict_int(self.facet_ordinal, "evidence facet ordinal", minimum=0)
        support = _strict_int(self.support, "facet support", minimum=SUPPORT_MIN)
        contradiction = _strict_int(
            self.contradiction, "facet contradiction", minimum=SUPPORT_MIN
        )
        if support > SUPPORT_MAX or contradiction > SUPPORT_MAX:
            raise BircoP1CoreError("support and contradiction must be integers from 0 to 4")
        evidence = self.evidence_unit_ordinal
        if evidence is not None:
            evidence = _strict_int(evidence, "evidence unit ordinal", minimum=0)
        object.__setattr__(self, "facet_ordinal", facet)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "contradiction", contradiction)
        object.__setattr__(self, "evidence_unit_ordinal", evidence)

    def payload(self) -> dict[str, object]:
        return {
            "facet_ordinal": self.facet_ordinal,
            "support": self.support,
            "contradiction": self.contradiction,
            "evidence_unit_ordinal": self.evidence_unit_ordinal,
        }


def _evidence_from_mapping(value: object) -> FacetEvidence:
    if not isinstance(value, Mapping):
        raise BircoP1CoreError("each facet-evidence row must be a mapping")
    _require_exact_fields(value, _EVIDENCE_FIELDS, "facet-evidence row")
    return FacetEvidence(
        facet_ordinal=value["facet_ordinal"],  # type: ignore[arg-type]
        support=value["support"],  # type: ignore[arg-type]
        contradiction=value["contradiction"],  # type: ignore[arg-type]
        evidence_unit_ordinal=value["evidence_unit_ordinal"],  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class CandidateFacetEvidence:
    """One candidate's complete row over every plan facet."""

    candidate_ordinal: int
    evidence_unit_count: int
    facet_evidence: tuple[FacetEvidence, ...]

    def __post_init__(self) -> None:
        candidate = _strict_int(self.candidate_ordinal, "candidate ordinal", minimum=0)
        unit_count = _strict_int(
            self.evidence_unit_count, "evidence unit count", minimum=0
        )
        if isinstance(self.facet_evidence, (str, bytes)) or not isinstance(
            self.facet_evidence, Sequence
        ):
            raise BircoP1CoreError("facet_evidence must be an array")
        rows = tuple(
            row if isinstance(row, FacetEvidence) else _evidence_from_mapping(row)
            for row in self.facet_evidence
        )
        ordinals = tuple(row.facet_ordinal for row in rows)
        if ordinals != tuple(sorted(ordinals)) or len(set(ordinals)) != len(ordinals):
            raise BircoP1CoreError(
                "facet-evidence rows must have unique increasing facet ordinals"
            )
        for row in rows:
            if (
                row.evidence_unit_ordinal is not None
                and row.evidence_unit_ordinal >= unit_count
            ):
                raise BircoP1CoreError(
                    "evidence unit ordinal is outside the candidate segmentation"
                )
        object.__setattr__(self, "candidate_ordinal", candidate)
        object.__setattr__(self, "evidence_unit_count", unit_count)
        object.__setattr__(self, "facet_evidence", rows)

    def payload(self) -> dict[str, object]:
        return {
            "candidate_ordinal": self.candidate_ordinal,
            "evidence_unit_count": self.evidence_unit_count,
            "facet_evidence": [row.payload() for row in self.facet_evidence],
        }


def _candidate_from_mapping(value: object) -> CandidateFacetEvidence:
    if not isinstance(value, Mapping):
        raise BircoP1CoreError("each candidate matrix row must be a mapping")
    _require_exact_fields(value, _CANDIDATE_FIELDS, "candidate matrix row")
    evidence = value["facet_evidence"]
    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
        raise BircoP1CoreError("facet_evidence must be an array")
    return CandidateFacetEvidence(
        candidate_ordinal=value["candidate_ordinal"],  # type: ignore[arg-type]
        evidence_unit_count=value["evidence_unit_count"],  # type: ignore[arg-type]
        facet_evidence=tuple(_evidence_from_mapping(row) for row in evidence),
    )


@dataclass(frozen=True)
class CandidateFacetEvidenceMatrix:
    """A complete, ordinal-only candidate/facet/evidence tensor for one query."""

    candidates: tuple[CandidateFacetEvidence, ...]

    def __post_init__(self) -> None:
        if isinstance(self.candidates, (str, bytes)) or not isinstance(
            self.candidates, Sequence
        ):
            raise BircoP1CoreError("candidates must be an array")
        rows = tuple(
            row if isinstance(row, CandidateFacetEvidence) else _candidate_from_mapping(row)
            for row in self.candidates
        )
        if not rows:
            raise BircoP1CoreError("candidate matrix must contain at least one candidate")
        if tuple(row.candidate_ordinal for row in rows) != tuple(range(len(rows))):
            raise BircoP1CoreError(
                "candidate ordinals must be canonical, contiguous, and match array order"
            )
        object.__setattr__(self, "candidates", rows)

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    def validate_against(self, plan: TypedFacetPlan) -> "CandidateFacetEvidenceMatrix":
        expected = tuple(range(len(plan.facets)))
        for row in self.candidates:
            observed = tuple(value.facet_ordinal for value in row.facet_evidence)
            if observed != expected:
                raise BircoP1CoreError(
                    "every candidate must contain every plan facet exactly once"
                )
        return self

    def payload(self) -> dict[str, object]:
        return {"candidates": [row.payload() for row in self.candidates]}


CandidateEvidenceMatrix = CandidateFacetEvidenceMatrix


def validate_candidate_facet_evidence_matrix(
    value: CandidateFacetEvidenceMatrix | Mapping[str, object],
    plan: TypedFacetPlan | Mapping[str, object],
) -> CandidateFacetEvidenceMatrix:
    """Validate exact candidate and facet completeness against ``plan``."""

    checked_plan = validate_typed_facet_plan(plan)
    if isinstance(value, CandidateFacetEvidenceMatrix):
        matrix = CandidateFacetEvidenceMatrix(tuple(value.candidates))
    else:
        if not isinstance(value, Mapping):
            raise BircoP1CoreError("candidate matrix must be a mapping or matrix")
        _require_exact_fields(value, _MATRIX_FIELDS, "candidate matrix")
        candidates = value["candidates"]
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
            raise BircoP1CoreError("candidates must be an array")
        matrix = CandidateFacetEvidenceMatrix(
            tuple(_candidate_from_mapping(row) for row in candidates)
        )
    return matrix.validate_against(checked_plan)


validate_candidate_matrix = validate_candidate_facet_evidence_matrix


def _totalized_quantized(value: object) -> int:
    if type(value) is not int:
        return 0
    return min(SUPPORT_MAX, max(SUPPORT_MIN, int(value)))


def totalize_candidate_facet_evidence(
    output: object,
    *,
    plan: TypedFacetPlan | Mapping[str, object],
    candidate_ordinal: int,
    evidence_unit_count: int,
) -> CandidateFacetEvidence:
    """Totalize one arbitrary scorer row into a complete zero-safe facet row."""

    checked_plan = validate_typed_facet_plan(plan)
    candidate = _strict_int(candidate_ordinal, "candidate ordinal", minimum=0)
    unit_count = _strict_int(evidence_unit_count, "evidence unit count", minimum=0)
    raw = output if isinstance(output, Mapping) else {}
    raw_rows = raw.get("facet_evidence", raw.get("facets", ()))
    if isinstance(raw_rows, (str, bytes)) or not isinstance(raw_rows, Sequence):
        raw_rows = ()
    retained: dict[int, FacetEvidence] = {}
    for row in raw_rows:
        if not isinstance(row, Mapping):
            continue
        facet = row.get("facet_ordinal")
        if type(facet) is not int or not 0 <= facet < len(checked_plan.facets):
            continue
        if facet in retained:
            continue
        evidence = row.get("evidence_unit_ordinal")
        if type(evidence) is not int or not 0 <= evidence < unit_count:
            evidence = None
        retained[facet] = FacetEvidence(
            facet,
            _totalized_quantized(row.get("support")),
            _totalized_quantized(row.get("contradiction")),
            evidence,
        )
    rows = tuple(
        retained.get(facet.ordinal, FacetEvidence(facet.ordinal, 0, 0, None))
        for facet in checked_plan.facets
    )
    return CandidateFacetEvidence(candidate, unit_count, rows)


def totalize_candidate_facet_evidence_matrix(
    output: object,
    *,
    plan: TypedFacetPlan | Mapping[str, object],
    evidence_unit_counts: Sequence[int],
) -> CandidateFacetEvidenceMatrix:
    """Totalize one candidate batch exactly once in canonical source order."""

    checked_plan = validate_typed_facet_plan(plan)
    if isinstance(evidence_unit_counts, (str, bytes)) or not isinstance(
        evidence_unit_counts, Sequence
    ):
        raise BircoP1CoreError("evidence_unit_counts must be an integer array")
    counts = tuple(
        _strict_int(value, "evidence unit count", minimum=0)
        for value in evidence_unit_counts
    )
    if not counts:
        raise BircoP1CoreError("candidate matrix must contain at least one candidate")
    raw = output if isinstance(output, Mapping) else {}
    raw_candidates = raw.get("candidates", ())
    if isinstance(raw_candidates, (str, bytes)) or not isinstance(
        raw_candidates, Sequence
    ):
        raw_candidates = ()
    by_ordinal: dict[int, object] = {}
    for position, row in enumerate(raw_candidates):
        if not isinstance(row, Mapping):
            continue
        ordinal = row.get("candidate_ordinal", position)
        if type(ordinal) is int and 0 <= ordinal < len(counts) and ordinal not in by_ordinal:
            by_ordinal[ordinal] = row
    rows = tuple(
        totalize_candidate_facet_evidence(
            by_ordinal.get(ordinal),
            plan=checked_plan,
            candidate_ordinal=ordinal,
            evidence_unit_count=counts[ordinal],
        )
        for ordinal in range(len(counts))
    )
    return CandidateFacetEvidenceMatrix(rows).validate_against(checked_plan)


totalize_candidate_matrix = totalize_candidate_facet_evidence_matrix


def _is_satisfied(row: FacetEvidence) -> bool:
    return row.support >= SATISFIED_SUPPORT_MIN and row.support > row.contradiction


def _rows_by_facet(candidate: CandidateFacetEvidence) -> dict[int, FacetEvidence]:
    return {row.facet_ordinal: row for row in candidate.facet_evidence}


def _active_facets(
    plan: TypedFacetPlan, active: frozenset[int] | None
) -> tuple[TypedFacet, ...]:
    if active is None:
        return plan.facets
    return tuple(facet for facet in plan.facets if facet.ordinal in active)


def _weighted_mass(
    plan: TypedFacetPlan,
    candidate: CandidateFacetEvidence,
    active: frozenset[int] | None = None,
) -> float:
    facets = _active_facets(plan, active)
    rows = _rows_by_facet(candidate)
    denominator = sum(facet.weight for facet in facets)
    return math.fsum(
        facet.weight
        * (rows[facet.ordinal].support - rows[facet.ordinal].contradiction)
        for facet in facets
    ) / denominator


def _contradiction_cost(
    plan: TypedFacetPlan,
    candidate: CandidateFacetEvidence,
    active: frozenset[int] | None = None,
) -> float:
    facets = _active_facets(plan, active)
    rows = _rows_by_facet(candidate)
    denominator = sum(facet.weight for facet in facets)
    return math.fsum(
        facet.weight * rows[facet.ordinal].contradiction for facet in facets
    ) / denominator


def _dependency_flow_mass(
    plan: TypedFacetPlan,
    candidate: CandidateFacetEvidence,
    active: frozenset[int] | None = None,
) -> float:
    facets = _active_facets(plan, active)
    active_ordinals = frozenset(facet.ordinal for facet in facets)
    edges = tuple(
        edge
        for edge in plan.edges
        if edge.edge_type == "REQUIRES"
        and edge.source_facet_ordinal in active_ordinals
        and edge.target_facet_ordinal in active_ordinals
    )
    order = _topological_ordinals(tuple(sorted(active_ordinals)), edges)
    incoming: dict[int, list[int]] = {ordinal: [] for ordinal in active_ordinals}
    for edge in edges:
        incoming[edge.target_facet_ordinal].append(edge.source_facet_ordinal)
    rows = _rows_by_facet(candidate)
    effective: dict[int, int] = {}
    for ordinal in order:
        row = rows[ordinal]
        predecessors = incoming[ordinal]
        if not _is_satisfied(row):
            effective[ordinal] = 0
        elif predecessors and not all(
            _is_satisfied(rows[source]) and effective[source] > 0
            for source in predecessors
        ):
            effective[ordinal] = 0
        elif predecessors:
            effective[ordinal] = min(
                row.support, *(effective[source] for source in predecessors)
            )
        else:
            effective[ordinal] = row.support
    denominator = sum(facet.weight for facet in facets)
    return math.fsum(
        facet.weight * effective[facet.ordinal] for facet in facets
    ) / denominator


@dataclass(frozen=True)
class CapacityAssignment:
    """Canonical optimal capacity-one assignment, containing ordinals only."""

    facet_to_evidence: tuple[tuple[int, int], ...]
    bottleneck_support: int
    satisfied_facet_count: int
    assignment_mass: float
    contradiction_cost: float

    @property
    def assigned_facet_count(self) -> int:
        return len(self.facet_to_evidence)


def solve_capacity_assignment(
    plan: TypedFacetPlan,
    candidate: CandidateFacetEvidence,
    *,
    active_facets: frozenset[int] | None = None,
) -> CapacityAssignment:
    """Solve the frozen capacity-one facet/evidence assignment exactly.

    Because each facet names at most one evidence unit, the feasible choices
    are one facet per occupied evidence ordinal.  At twelve facets the complete
    Cartesian enumeration is tiny (the worst product is below 7,000), making
    the result deterministic and independent of an external solver.
    """

    facets = _active_facets(plan, active_facets)
    active = frozenset(facet.ordinal for facet in facets)
    rows = _rows_by_facet(candidate)
    grouped: dict[int, list[int]] = {}
    for facet in facets:
        row = rows[facet.ordinal]
        evidence = row.evidence_unit_ordinal
        # A capacity assignment is evidence for a satisfied facet, not merely
        # a reference emitted alongside a zero-support or contradicted row.
        if evidence is not None and _is_satisfied(row):
            grouped.setdefault(evidence, []).append(facet.ordinal)
    evidence_ordinals = tuple(sorted(grouped))
    # Matching is partial: an occupied evidence unit has capacity at most one,
    # not an obligation to accept a facet.  ``None`` wins only when every
    # substantive coordinate ties.
    choices = tuple(
        (None, *tuple(sorted(grouped[evidence]))) for evidence in evidence_ordinals
    )
    combinations = itertools.product(*choices) if choices else ((),)
    total_weight = sum(facet.weight for facet in facets)
    facet_by_ordinal = {facet.ordinal: facet for facet in facets}
    contradiction = _contradiction_cost(plan, candidate, active)

    best_key: tuple[float, ...] | None = None
    best_choices: tuple[int | None, ...] = ()
    for selected_value in combinations:
        chosen = tuple(selected_value)
        selected = tuple(value for value in chosen if value is not None)
        selected_set = frozenset(selected)
        supports = tuple(
            rows[facet.ordinal].support if facet.ordinal in selected_set else 0
            for facet in facets
        )
        bottleneck = min(supports) if supports else 0
        satisfied = sum(_is_satisfied(rows[ordinal]) for ordinal in selected)
        mass = math.fsum(
            facet_by_ordinal[ordinal].weight * rows[ordinal].support
            for ordinal in selected
        ) / total_weight
        # The final coordinates select a unique canonical assignment if all
        # substantive coordinates tie: lower facet ordinals win in evidence
        # ordinal order.
        canonical_tail = tuple(
            0.0 if ordinal is None else float(-(ordinal + 1))
            for ordinal in chosen
        )
        key = (float(bottleneck), float(satisfied), mass, -contradiction, *canonical_tail)
        if best_key is None or key > best_key:
            best_key = key
            best_choices = chosen

    pairs = tuple(
        sorted(
            (
                (facet_ordinal, evidence_ordinal)
                for facet_ordinal, evidence_ordinal in zip(
                    best_choices, evidence_ordinals
                )
                if facet_ordinal is not None
            ),
            key=lambda row: row[0],
        )
    )
    best_selected = tuple(
        ordinal for ordinal in best_choices if ordinal is not None
    )
    selected_set = frozenset(best_selected)
    supports = tuple(
        rows[facet.ordinal].support if facet.ordinal in selected_set else 0
        for facet in facets
    )
    mass = math.fsum(
        facet_by_ordinal[ordinal].weight * rows[ordinal].support
        for ordinal in best_selected
    ) / total_weight
    return CapacityAssignment(
        facet_to_evidence=pairs,
        bottleneck_support=min(supports) if supports else 0,
        satisfied_facet_count=sum(_is_satisfied(rows[row]) for row in best_selected),
        assignment_mass=mass,
        contradiction_cost=contradiction,
    )


@dataclass(frozen=True)
class CandidateRecipeScore:
    """Content-free ordinal score used to construct one complete permutation."""

    candidate_ordinal: int
    rank_key: tuple[float, ...]
    scalar_score: float
    weighted_mass: float
    contradiction_cost: float
    assignment: CapacityAssignment

    def __post_init__(self) -> None:
        _strict_int(self.candidate_ordinal, "candidate ordinal", minimum=0)
        if not self.rank_key or any(not math.isfinite(float(value)) for value in self.rank_key):
            raise BircoP1CoreError("candidate rank key must be finite and nonempty")
        for value in (self.scalar_score, self.weighted_mass, self.contradiction_cost):
            if not math.isfinite(float(value)):
                raise BircoP1CoreError("candidate recipe scores must be finite")


@dataclass(frozen=True)
class RecipeRanking:
    """One recipe's full source-candidate permutation and numeric score rows."""

    recipe_id: str
    candidate_ordinals: tuple[int, ...]
    scores: tuple[CandidateRecipeScore, ...]

    def __post_init__(self) -> None:
        if self.recipe_id not in RECIPE_IDS:
            raise BircoP1CoreError("recipe is outside the frozen registry")
        permutation = validate_full_permutation(
            self.candidate_ordinals, len(self.candidate_ordinals)
        )
        if len(self.scores) != len(permutation):
            raise BircoP1CoreError("ranking score rows must cover the full permutation")
        if tuple(score.candidate_ordinal for score in self.scores) != permutation:
            raise BircoP1CoreError("ranking score rows must follow permutation order")
        object.__setattr__(self, "candidate_ordinals", permutation)

    def score_by_ordinal(self) -> dict[int, CandidateRecipeScore]:
        return {score.candidate_ordinal: score for score in self.scores}


def _candidate_recipe_score(
    plan: TypedFacetPlan,
    candidate: CandidateFacetEvidence,
    recipe_id: str,
    active: frozenset[int] | None = None,
) -> CandidateRecipeScore:
    facets = _active_facets(plan, active)
    rows = _rows_by_facet(candidate)
    mass = _weighted_mass(plan, candidate, active)
    contradiction = _contradiction_cost(plan, candidate, active)
    assignment = solve_capacity_assignment(plan, candidate, active_facets=active)

    if recipe_id == R1_WEIGHTED_MASS:
        key = (mass,)
        scalar = mass
    elif recipe_id == R2_BOTTLENECK:
        required = tuple(facet for facet in facets if facet.facet_type == "REQUIRED")
        minimum = min((rows[facet.ordinal].support for facet in required), default=4)
        satisfied = sum(_is_satisfied(rows[facet.ordinal]) for facet in required)
        key = (float(minimum), float(satisfied), mass)
        scalar = minimum * 1_000.0 + satisfied * 10.0 + mass
    elif recipe_id == R3_DEPENDENCY_FLOW:
        flow = _dependency_flow_mass(plan, candidate, active)
        key = (flow, -contradiction, mass)
        scalar = flow * 1_000.0 - contradiction * 10.0 + mass
    elif recipe_id == R4_CAPACITY_MATCH:
        key = (
            float(assignment.bottleneck_support),
            float(assignment.satisfied_facet_count),
            assignment.assignment_mass,
            -assignment.contradiction_cost,
        )
        scalar = (
            assignment.bottleneck_support * 10_000.0
            + assignment.satisfied_facet_count * 100.0
            + assignment.assignment_mass * 10.0
            - assignment.contradiction_cost
        )
    else:
        raise BircoP1CoreError("recipe is outside the frozen registry")
    return CandidateRecipeScore(
        candidate_ordinal=candidate.candidate_ordinal,
        rank_key=tuple(float(value) for value in key),
        scalar_score=float(scalar),
        weighted_mass=mass,
        contradiction_cost=contradiction,
        assignment=assignment,
    )


def _rank_with_active(
    plan: TypedFacetPlan,
    matrix: CandidateFacetEvidenceMatrix,
    recipe_id: str,
    active: frozenset[int] | None = None,
) -> RecipeRanking:
    if recipe_id not in RECIPE_IDS:
        raise BircoP1CoreError("recipe is outside the frozen registry")
    if active is not None and not active:
        raise BircoP1CoreError("a recipe cannot be scored with zero active facets")
    scores = tuple(
        _candidate_recipe_score(plan, candidate, recipe_id, active)
        for candidate in matrix.candidates
    )
    ordered = tuple(
        sorted(
            scores,
            key=lambda score: (
                *( -value for value in score.rank_key),
                score.candidate_ordinal,
            ),
        )
    )
    permutation = tuple(score.candidate_ordinal for score in ordered)
    validate_full_permutation(permutation, matrix.candidate_count)
    return RecipeRanking(recipe_id, permutation, ordered)


def rank_candidates(
    plan: TypedFacetPlan | Mapping[str, object],
    matrix: CandidateFacetEvidenceMatrix | Mapping[str, object],
    recipe_id: str,
) -> RecipeRanking:
    """Rank every candidate exactly once, with source ordinal as the final tie."""

    checked_plan = validate_typed_facet_plan(plan)
    checked_matrix = validate_candidate_facet_evidence_matrix(matrix, checked_plan)
    return _rank_with_active(checked_plan, checked_matrix, recipe_id)


def rank_r1_weighted_mass(plan: object, matrix: object) -> RecipeRanking:
    return rank_candidates(plan, matrix, R1_WEIGHTED_MASS)  # type: ignore[arg-type]


def rank_r2_bottleneck(plan: object, matrix: object) -> RecipeRanking:
    return rank_candidates(plan, matrix, R2_BOTTLENECK)  # type: ignore[arg-type]


def rank_r3_dependency_flow(plan: object, matrix: object) -> RecipeRanking:
    return rank_candidates(plan, matrix, R3_DEPENDENCY_FLOW)  # type: ignore[arg-type]


def rank_r4_capacity_match(plan: object, matrix: object) -> RecipeRanking:
    return rank_candidates(plan, matrix, R4_CAPACITY_MATCH)  # type: ignore[arg-type]


def build_recipe_rankings(
    plan: TypedFacetPlan | Mapping[str, object],
    matrix: CandidateFacetEvidenceMatrix | Mapping[str, object],
) -> dict[str, RecipeRanking]:
    checked_plan = validate_typed_facet_plan(plan)
    checked_matrix = validate_candidate_facet_evidence_matrix(matrix, checked_plan)
    return {
        recipe: _rank_with_active(checked_plan, checked_matrix, recipe)
        for recipe in RECIPE_IDS
    }


def select_e0_recipe(plan: TypedFacetPlan | Mapping[str, object]) -> str:
    """Apply the frozen qrel-blind E0 recipe policy."""

    checked = validate_typed_facet_plan(plan)
    if any(edge.edge_type == "REQUIRES" for edge in checked.edges):
        return R3_DEPENDENCY_FLOW
    if any(
        facet.facet_type in {"EXCLUDED", "ELIGIBILITY"}
        for facet in checked.facets
    ):
        return R2_BOTTLENECK
    return R4_CAPACITY_MATCH


def select_e0_ranking(
    plan: TypedFacetPlan | Mapping[str, object],
    rankings: Mapping[str, RecipeRanking],
) -> RecipeRanking:
    if set(rankings) != set(RECIPE_IDS):
        raise BircoP1CoreError("E0 requires exactly the four recipe rankings")
    recipe = select_e0_recipe(plan)
    ranking = rankings[recipe]
    if ranking.recipe_id != recipe:
        raise BircoP1CoreError("recipe-ranking registry drifted")
    return ranking


def validate_full_permutation(
    permutation: Sequence[int], expected: int | Sequence[int] | Mapping[int, object]
) -> tuple[int, ...]:
    """Require one occurrence of every expected canonical candidate ordinal."""

    if isinstance(permutation, (str, bytes)) or not isinstance(permutation, Sequence):
        raise BircoP1CoreError("candidate ranking must be an array")
    values = tuple(
        _strict_int(value, "candidate permutation ordinal", minimum=0)
        for value in permutation
    )
    if type(expected) is int:
        if expected < 0:
            raise BircoP1CoreError("expected candidate count cannot be negative")
        expected_values = tuple(range(expected))
    elif isinstance(expected, Mapping):
        expected_values = tuple(expected.keys())
    elif isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        expected_values = tuple(expected)
    else:
        raise BircoP1CoreError("expected candidate universe is malformed")
    checked_expected = tuple(
        _strict_int(value, "expected candidate ordinal", minimum=0)
        for value in expected_values
    )
    if len(set(checked_expected)) != len(checked_expected):
        raise BircoP1CoreError("expected candidate universe contains duplicates")
    if len(values) != len(checked_expected) or set(values) != set(checked_expected):
        raise BircoP1CoreError(
            "ranking must be a full permutation of the candidate universe"
        )
    return values


def _softmax(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1 or values.size == 0:
        raise BircoP1CoreError("softmax requires a nonempty vector")
    shifted = values - float(np.max(values))
    exponents = np.exp(shifted)
    denominator = float(np.sum(exponents))
    if not math.isfinite(denominator) or denominator <= 0:
        raise BircoP1CoreError("softmax normalization is nonfinite")
    return exponents / denominator


def _pairwise_rank_stability(
    baseline: Sequence[int], perturbed: Sequence[int]
) -> float:
    if len(baseline) != len(perturbed) or set(baseline) != set(perturbed):
        raise BircoP1CoreError("rank-stability permutations differ in membership")
    if len(baseline) < 2:
        return 1.0
    positions = {ordinal: index for index, ordinal in enumerate(perturbed)}
    agreeing = 0
    total = 0
    for left_index, left in enumerate(baseline):
        for right in baseline[left_index + 1 :]:
            total += 1
            agreeing += positions[left] < positions[right]
    return agreeing / total


def validate_action_features(value: Mapping[str, object] | Sequence[object]) -> tuple[float, ...]:
    """Validate exactly the twelve frozen, content-free E4 coordinates."""

    if isinstance(value, Mapping):
        supplied = set(value)
        forbidden = supplied.intersection(FORBIDDEN_E4_FEATURES)
        if forbidden:
            raise BircoP1CoreError(
                "forbidden E4 feature(s): " + ", ".join(sorted(forbidden))
            )
        expected = set(FEATURE_ORDER)
        if supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            raise BircoP1CoreError(
                f"fixed action feature schema drifted; missing={missing}, extra={extra}"
            )
        raw = tuple(value[name] for name in FEATURE_ORDER)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = tuple(value)
        if len(raw) != len(FEATURE_ORDER):
            raise BircoP1CoreError("action feature vector must have exactly 12 values")
    else:
        raise BircoP1CoreError("action features must be a mapping or numeric vector")
    return tuple(
        _finite_float(feature, f"action feature {FEATURE_ORDER[index]}")
        for index, feature in enumerate(raw)
    )


feature_vector = validate_action_features


def compute_action_features(
    plan: TypedFacetPlan | Mapping[str, object],
    matrix: CandidateFacetEvidenceMatrix | Mapping[str, object],
    ranking: RecipeRanking,
) -> tuple[float, ...]:
    """Compute the frozen twelve label-free action coordinates."""

    checked_plan = validate_typed_facet_plan(plan)
    checked_matrix = validate_candidate_facet_evidence_matrix(matrix, checked_plan)
    if not isinstance(ranking, RecipeRanking):
        raise BircoP1CoreError("action ranking must be a RecipeRanking")
    validate_full_permutation(ranking.candidate_ordinals, checked_matrix.candidate_count)
    recomputed = _rank_with_active(checked_plan, checked_matrix, ranking.recipe_id)
    if ranking != recomputed:
        raise BircoP1CoreError("action ranking does not match the supplied plan and matrix")

    facet_count = len(checked_plan.facets)
    required = tuple(
        facet for facet in checked_plan.facets if facet.facet_type == "REQUIRED"
    )
    exclusion_or_eligibility = sum(
        facet.facet_type in {"EXCLUDED", "ELIGIBILITY"}
        for facet in checked_plan.facets
    )
    top = ranking.scores[:NDCG_CUTOFF]
    top_rows = tuple(
        checked_matrix.candidates[score.candidate_ordinal] for score in top
    )

    if top_rows:
        mean_support = math.fsum(
            row.support
            for candidate in top_rows
            for row in candidate.facet_evidence
        ) / (len(top_rows) * facet_count * SUPPORT_MAX)
        if required:
            minimum_required = math.fsum(
                min(
                    candidate.facet_evidence[facet.ordinal].support
                    for facet in required
                )
                for candidate in top_rows
            ) / (len(top_rows) * SUPPORT_MAX)
        else:
            minimum_required = 1.0
        satisfied_fraction = math.fsum(
            _is_satisfied(row)
            for candidate in top_rows
            for row in candidate.facet_evidence
        ) / (len(top_rows) * facet_count)
        contradiction_negative = -math.fsum(
            row.contradiction
            for candidate in top_rows
            for row in candidate.facet_evidence
        ) / (len(top_rows) * facet_count * SUPPORT_MAX)
        assignment_fraction = math.fsum(
            score.assignment.assigned_facet_count / facet_count for score in top
        ) / len(top)
    else:  # RecipeRanking cannot be empty, retained as a defensive totalizer.
        mean_support = 0.0
        minimum_required = 1.0 if not required else 0.0
        satisfied_fraction = 0.0
        contradiction_negative = 0.0
        assignment_fraction = 0.0

    margin = (
        max(0.0, top[0].scalar_score - top[1].scalar_score)
        if len(top) >= 2
        else 0.0
    )
    scalar = np.asarray([score.scalar_score for score in ranking.scores], dtype=np.float64)
    if len(scalar) <= 1:
        entropy_negative = 0.0
    else:
        scale = float(np.std(scalar, ddof=0))
        logits = scalar - float(np.mean(scalar))
        if scale > 0:
            logits = logits / scale
        probabilities = _softmax(logits)
        entropy = -math.fsum(
            float(probability) * math.log(float(probability))
            for probability in probabilities
            if probability > 0
        )
        entropy_negative = -entropy / math.log(len(probabilities))

    baseline = ranking.candidate_ordinals
    stabilities = []
    all_ordinals = frozenset(range(facet_count))
    for removed in range(facet_count):
        active = all_ordinals - {removed}
        perturbed = _rank_with_active(
            checked_plan, checked_matrix, ranking.recipe_id, active
        ).candidate_ordinals
        stabilities.append(_pairwise_rank_stability(baseline, perturbed))

    values = (
        float(facet_count),
        len(required) / facet_count,
        exclusion_or_eligibility / facet_count,
        len(checked_plan.edges) / facet_count,
        mean_support,
        minimum_required,
        satisfied_fraction,
        contradiction_negative,
        margin,
        entropy_negative,
        assignment_fraction,
        math.fsum(stabilities) / len(stabilities),
    )
    return validate_action_features(values)


def action_feature_mapping(values: Mapping[str, object] | Sequence[object]) -> dict[str, float]:
    checked = validate_action_features(values)
    return dict(zip(FEATURE_ORDER, checked))


@dataclass(frozen=True)
class E4ActionTrainingRow:
    """One recipe action in one complete A_form slate; no item identity exists."""

    recipe_id: str
    features: tuple[float, ...]
    integer_utility: int

    def __post_init__(self) -> None:
        if self.recipe_id not in RECIPE_IDS:
            raise BircoP1CoreError("training recipe is outside the frozen registry")
        features = validate_action_features(self.features)
        utility = _strict_int(self.integer_utility, "integer utility", minimum=0)
        if utility > INTEGER_UTILITY_SCALE:
            raise BircoP1CoreError("integer utility cannot exceed 1,000,000,000")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "integer_utility", utility)

    @classmethod
    def from_mapping(
        cls,
        recipe_id: str,
        features: Mapping[str, object] | Sequence[object],
        integer_utility: int,
    ) -> "E4ActionTrainingRow":
        return cls(recipe_id, validate_action_features(features), integer_utility)


@dataclass(frozen=True)
class E4TrainingSlate:
    """All four recipe actions for one anonymous A_form item."""

    actions: tuple[E4ActionTrainingRow, ...]

    def __post_init__(self) -> None:
        if isinstance(self.actions, (str, bytes)) or not isinstance(self.actions, Sequence):
            raise BircoP1CoreError("E4 slate actions must be an array")
        actions = tuple(self.actions)
        if not all(isinstance(action, E4ActionTrainingRow) for action in actions):
            raise BircoP1CoreError("E4 slate contains a foreign action type")
        by_recipe: dict[str, E4ActionTrainingRow] = {}
        for action in actions:
            if action.recipe_id in by_recipe:
                raise BircoP1CoreError("E4 slate contains a duplicate recipe")
            by_recipe[action.recipe_id] = action
        if set(by_recipe) != set(RECIPE_IDS):
            raise BircoP1CoreError("each E4 slate must contain all four recipes once")
        object.__setattr__(
            self, "actions", tuple(by_recipe[recipe] for recipe in RECIPE_IDS)
        )


def make_e4_training_slate(
    features_by_recipe: Mapping[str, Mapping[str, object] | Sequence[object]],
    integer_utilities_by_recipe: Mapping[str, int],
) -> E4TrainingSlate:
    if set(features_by_recipe) != set(RECIPE_IDS) or set(integer_utilities_by_recipe) != set(
        RECIPE_IDS
    ):
        raise BircoP1CoreError("training slate mappings must contain all four recipes")
    return E4TrainingSlate(
        tuple(
            E4ActionTrainingRow.from_mapping(
                recipe,
                features_by_recipe[recipe],
                integer_utilities_by_recipe[recipe],
            )
            for recipe in RECIPE_IDS
        )
    )


@dataclass(frozen=True)
class E4Model:
    """Shared standardized linear utility model and Laplace covariance."""

    population_mean: tuple[float, ...]
    population_std: tuple[float, ...]
    coefficients: tuple[float, ...]
    laplace_covariance: tuple[tuple[float, ...], ...]
    solver: str
    iterations: int
    converged: bool
    objective: float

    def __post_init__(self) -> None:
        width = len(FEATURE_ORDER)
        mean = validate_action_features(self.population_mean)
        std = validate_action_features(self.population_std)
        beta = validate_action_features(self.coefficients)
        if any(value < 0 for value in std):
            raise BircoP1CoreError("E4 population standard deviation is negative")
        covariance = tuple(tuple(float(value) for value in row) for row in self.laplace_covariance)
        if len(covariance) != width or any(len(row) != width for row in covariance):
            raise BircoP1CoreError("E4 Laplace covariance width drifted")
        array = np.asarray(covariance, dtype=np.float64)
        if not np.all(np.isfinite(array)) or not np.allclose(
            array, array.T, rtol=1.0e-10, atol=1.0e-12
        ):
            raise BircoP1CoreError("E4 Laplace covariance must be finite and symmetric")
        if not isinstance(self.solver, str) or not self.solver:
            raise BircoP1CoreError("E4 solver name is missing")
        _strict_int(self.iterations, "E4 iteration count", minimum=0)
        if type(self.converged) is not bool:
            raise BircoP1CoreError("E4 converged flag must be boolean")
        objective = _finite_float(self.objective, "E4 objective")
        object.__setattr__(self, "population_mean", mean)
        object.__setattr__(self, "population_std", std)
        object.__setattr__(self, "coefficients", beta)
        object.__setattr__(self, "laplace_covariance", covariance)
        object.__setattr__(self, "objective", objective)

    def standardize(
        self, features: Mapping[str, object] | Sequence[object]
    ) -> tuple[float, ...]:
        values = validate_action_features(features)
        return tuple(
            0.0 if std == 0.0 else (value - mean) / std
            for value, mean, std in zip(
                values, self.population_mean, self.population_std
            )
        )

    def predict_mean(self, features: Mapping[str, object] | Sequence[object]) -> float:
        z = self.standardize(features)
        return float(math.fsum(value * beta for value, beta in zip(z, self.coefficients)))

    def predict_standard_error(
        self, features: Mapping[str, object] | Sequence[object]
    ) -> float:
        z = np.asarray(self.standardize(features), dtype=np.float64)
        covariance = np.asarray(self.laplace_covariance, dtype=np.float64)
        variance = float(z @ covariance @ z)
        if variance < 0 and abs(variance) <= 1.0e-12:
            variance = 0.0
        if variance < 0 or not math.isfinite(variance):
            raise BircoP1CoreError("E4 posterior predictive variance is invalid")
        return math.sqrt(variance)

    def lower_confidence_utility(
        self, features: Mapping[str, object] | Sequence[object]
    ) -> float:
        return self.predict_mean(features) - E4_LAPLACE_PENALTY * self.predict_standard_error(
            features
        )

    def payload(self) -> dict[str, object]:
        return {
            "version": VERSION,
            "feature_order": list(FEATURE_ORDER),
            "population_mean": list(self.population_mean),
            "population_std": list(self.population_std),
            "coefficients": list(self.coefficients),
            "laplace_covariance": [list(row) for row in self.laplace_covariance],
            "L2": E4_L2,
            "target_temperature": E4_TARGET_TEMPERATURE,
            "laplace_standard_error_multiplier": E4_LAPLACE_PENALTY,
            "solver": self.solver,
            "max_iter": E4_MAX_ITER,
            "iterations": self.iterations,
            "converged": self.converged,
            "objective": self.objective,
            "recipe_id_used_as_feature": False,
        }


def _listwise_loss_gradient_hessian(
    beta: np.ndarray,
    standardized_slates: Sequence[np.ndarray],
    target_probabilities: Sequence[np.ndarray],
    *,
    include_hessian: bool,
) -> tuple[float, np.ndarray, np.ndarray | None]:
    width = len(FEATURE_ORDER)
    loss = 0.5 * E4_L2 * float(beta @ beta)
    gradient = E4_L2 * beta.copy()
    hessian = E4_L2 * np.eye(width, dtype=np.float64) if include_hessian else None
    # A_form is one thirty-item likelihood.  Summing per-item cross-entropies
    # before adding the lambda-one prior is essential: averaging would silently
    # multiply the effective regularization and would prevent Laplace
    # information from growing with the number of observed slates.
    item_weight = 1.0
    for features, target in zip(standardized_slates, target_probabilities):
        # The predeclared 1/20 temperature belongs to the scaled nDCG target
        # distribution.  The shared linear model emits ordinary listwise
        # logits; applying the target temperature again here would silently
        # change the lambda-one regularized objective.
        logits = features @ beta
        probabilities = _softmax(logits)
        log_normalizer = float(np.max(logits)) + math.log(
            math.fsum(math.exp(float(value - np.max(logits))) for value in logits)
        )
        loss += item_weight * (
            -float(target @ logits) + log_normalizer
        )
        gradient += item_weight * (features.T @ (probabilities - target))
        if hessian is not None:
            probability_hessian = np.diag(probabilities) - np.outer(
                probabilities, probabilities
            )
            hessian += item_weight * (
                features.T @ probability_hessian @ features
            )
    return float(loss), gradient, hessian


def _numpy_lbfgs(
    objective_gradient,
    width: int,
    *,
    max_iter: int = E4_MAX_ITER,
    memory: int = 10,
) -> tuple[np.ndarray, float, int, bool]:
    """Small deterministic L-BFGS with Armijo backtracking."""

    x = np.zeros(width, dtype=np.float64)
    value, gradient = objective_gradient(x)
    s_history: list[np.ndarray] = []
    y_history: list[np.ndarray] = []
    rho_history: list[float] = []
    converged = float(np.max(np.abs(gradient))) <= 1.0e-9
    iterations = 0
    for iteration in range(max_iter):
        if converged:
            break
        q = gradient.copy()
        alphas: list[float] = []
        for s_value, y_value, rho in zip(
            reversed(s_history), reversed(y_history), reversed(rho_history)
        ):
            alpha = rho * float(s_value @ q)
            alphas.append(alpha)
            q -= alpha * y_value
        if s_history:
            latest_s = s_history[-1]
            latest_y = y_history[-1]
            yy = float(latest_y @ latest_y)
            gamma = float(latest_s @ latest_y) / yy if yy > 0 else 1.0
        else:
            gamma = 1.0
        direction = gamma * q
        for index, (s_value, y_value, rho) in enumerate(
            zip(s_history, y_history, rho_history)
        ):
            beta_value = rho * float(y_value @ direction)
            alpha = alphas[len(alphas) - 1 - index]
            direction += s_value * (alpha - beta_value)
        direction = -direction
        directional = float(gradient @ direction)
        if not math.isfinite(directional) or directional >= 0:
            direction = -gradient
            directional = -float(gradient @ gradient)

        step = 1.0
        accepted = False
        for _line_search in range(64):
            candidate = x + step * direction
            candidate_value, candidate_gradient = objective_gradient(candidate)
            if math.isfinite(candidate_value) and candidate_value <= value + 1.0e-4 * step * directional:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break

        s_value = candidate - x
        y_value = candidate_gradient - gradient
        curvature = float(s_value @ y_value)
        if curvature > 1.0e-12 * max(1.0, float(np.linalg.norm(s_value)) * float(np.linalg.norm(y_value))):
            if len(s_history) == memory:
                s_history.pop(0)
                y_history.pop(0)
                rho_history.pop(0)
            s_history.append(s_value)
            y_history.append(y_value)
            rho_history.append(1.0 / curvature)
        x = candidate
        value = float(candidate_value)
        gradient = candidate_gradient
        iterations = iteration + 1
        converged = float(np.max(np.abs(gradient))) <= 1.0e-9
        if float(np.max(np.abs(s_value))) <= 1.0e-13 * max(
            1.0, float(np.max(np.abs(x)))
        ):
            converged = float(np.max(np.abs(gradient))) <= 1.0e-7
            break
    return x, value, iterations, converged


def fit_e4_listwise_softmax(slates: Sequence[E4TrainingSlate]) -> E4Model:
    """Fit the single lambda-one listwise E4 model and Laplace covariance."""

    if isinstance(slates, (str, bytes)) or not isinstance(slates, Sequence):
        raise BircoP1CoreError("E4 fit requires the formal training slate array")
    supplied = tuple(slates)
    if len(supplied) != E4_A_FORM_ITEM_COUNT:
        raise BircoP1CoreError("E4 A_form fit requires exactly 30 complete slates")
    if not all(isinstance(slate, E4TrainingSlate) for slate in supplied):
        raise BircoP1CoreError("E4 fit accepts only E4TrainingSlate rows")
    # No query/item identity is admitted to E4.  Canonical content ordering
    # makes the same anonymous slate multiset bit-stable under caller order.
    checked = tuple(
        sorted(
            supplied,
            key=lambda slate: tuple(
                coordinate
                for action in slate.actions
                for coordinate in (*action.features, float(action.integer_utility))
            ),
        )
    )
    all_features = np.asarray(
        [action.features for slate in checked for action in slate.actions],
        dtype=np.float64,
    )
    population_mean_array = np.mean(all_features, axis=0)
    population_std_array = np.std(all_features, axis=0, ddof=0)
    safe_std = np.where(population_std_array == 0.0, 1.0, population_std_array)
    standardized_slates = tuple(
        (np.asarray([action.features for action in slate.actions], dtype=np.float64) - population_mean_array)
        / safe_std
        for slate in checked
    )
    for values in standardized_slates:
        values[:, population_std_array == 0.0] = 0.0
    target_probabilities = tuple(
        _softmax(
            np.asarray(
                [action.integer_utility / INTEGER_UTILITY_SCALE for action in slate.actions],
                dtype=np.float64,
            )
            / E4_TARGET_TEMPERATURE
        )
        for slate in checked
    )

    def objective_gradient(beta: np.ndarray) -> tuple[float, np.ndarray]:
        loss, gradient, _ = _listwise_loss_gradient_hessian(
            beta,
            standardized_slates,
            target_probabilities,
            include_hessian=False,
        )
        return loss, gradient

    beta, objective, iterations, converged = _numpy_lbfgs(
        objective_gradient, len(FEATURE_ORDER), max_iter=E4_MAX_ITER
    )
    objective, gradient, hessian = _listwise_loss_gradient_hessian(
        beta,
        standardized_slates,
        target_probabilities,
        include_hessian=True,
    )
    if hessian is None:  # pragma: no cover - guarded by include_hessian=True.
        raise BircoP1CoreError("E4 Laplace Hessian was not computed")
    if not converged and float(np.max(np.abs(gradient))) > 1.0e-6:
        raise BircoP1CoreError("deterministic E4 L-BFGS did not converge")
    covariance = np.linalg.inv(hessian)
    covariance = (covariance + covariance.T) / 2.0
    return E4Model(
        population_mean=tuple(float(value) for value in population_mean_array),
        population_std=tuple(float(value) for value in population_std_array),
        coefficients=tuple(float(value) for value in beta),
        laplace_covariance=tuple(
            tuple(float(value) for value in row) for row in covariance
        ),
        solver="numpy_deterministic_lbfgs_m10_v1",
        iterations=iterations,
        converged=converged,
        objective=float(objective),
    )


fit_e4_model = fit_e4_listwise_softmax
fit_e4 = fit_e4_listwise_softmax


def fit_listwise_softmax_from_mappings(
    feature_slates: Sequence[
        Mapping[str, Mapping[str, object] | Sequence[object]]
    ],
    integer_utility_slates: Sequence[Mapping[str, int]],
) -> E4Model:
    if len(feature_slates) != len(integer_utility_slates):
        raise BircoP1CoreError("E4 feature and utility slate counts differ")
    return fit_e4_listwise_softmax(
        tuple(
            make_e4_training_slate(features, utilities)
            for features, utilities in zip(feature_slates, integer_utility_slates)
        )
    )


@dataclass(frozen=True)
class E4Selection:
    selected_recipe_id: str
    e0_recipe_id: str
    lower_confidence_scores: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if self.selected_recipe_id not in RECIPE_IDS or self.e0_recipe_id not in RECIPE_IDS:
            raise BircoP1CoreError("E4 selection recipe is outside the registry")
        if tuple(name for name, _ in self.lower_confidence_scores) != RECIPE_IDS:
            raise BircoP1CoreError("E4 score registry is incomplete or noncanonical")
        if any(not math.isfinite(value) for _, value in self.lower_confidence_scores):
            raise BircoP1CoreError("E4 lower-confidence score is nonfinite")

    def payload(self) -> dict[str, object]:
        return {
            "selected_recipe_id": self.selected_recipe_id,
            "E0_recipe_id": self.e0_recipe_id,
            "lower_confidence_scores": {
                recipe: score for recipe, score in self.lower_confidence_scores
            },
            "tie_policy": "argmax_then_E0_then_recipe_name",
        }


def select_e4_recipe(
    model: E4Model,
    features_by_recipe: Mapping[
        str, Mapping[str, object] | Sequence[object]
    ],
    *,
    e0_recipe_id: str,
) -> E4Selection:
    """Select argmax posterior mean-minus-half-SE, then E0, then recipe name."""

    if not isinstance(model, E4Model):
        raise BircoP1CoreError("E4 selector requires an E4Model")
    if e0_recipe_id not in RECIPE_IDS:
        raise BircoP1CoreError("E0 tie recipe is outside the frozen registry")
    if set(features_by_recipe) != set(RECIPE_IDS):
        raise BircoP1CoreError("E4 selection requires all four action feature vectors")
    scores = tuple(
        (recipe, model.lower_confidence_utility(features_by_recipe[recipe]))
        for recipe in RECIPE_IDS
    )
    maximum = max(score for _, score in scores)
    tied = tuple(recipe for recipe, score in scores if score == maximum)
    if e0_recipe_id in tied:
        selected = e0_recipe_id
    else:
        selected = min(tied)
    return E4Selection(selected, e0_recipe_id, scores)


def select_e4_ranking(
    model: E4Model,
    features_by_recipe: Mapping[
        str, Mapping[str, object] | Sequence[object]
    ],
    rankings: Mapping[str, RecipeRanking],
    *,
    e0_recipe_id: str,
) -> RecipeRanking:
    if set(rankings) != set(RECIPE_IDS):
        raise BircoP1CoreError("E4 ranking selection requires all four recipes")
    selection = select_e4_recipe(
        model, features_by_recipe, e0_recipe_id=e0_recipe_id
    )
    ranking = rankings[selection.selected_recipe_id]
    if ranking.recipe_id != selection.selected_recipe_id:
        raise BircoP1CoreError("recipe-ranking registry drifted")
    return ranking


def _validated_relevance(
    relevance: Mapping[int, object], permutation: Sequence[int]
) -> dict[int, float]:
    if not isinstance(relevance, Mapping) or not relevance:
        raise BircoP1CoreError("relevance must be a nonempty ordinal-to-score mapping")
    validate_full_permutation(permutation, relevance)
    result: dict[int, float] = {}
    for ordinal, value in relevance.items():
        checked_ordinal = _strict_int(ordinal, "qrel candidate ordinal", minimum=0)
        score = _finite_float(value, "linear relevance score")
        if score < 0:
            raise BircoP1CoreError("linear relevance scores cannot be negative")
        result[checked_ordinal] = score
    return result


def linear_gain_ndcg_at_10(
    permutation: Sequence[int], relevance: Mapping[int, object]
) -> float:
    """Official BIRCO-style nDCG@10 with linear source relevance gain."""

    qrels = _validated_relevance(relevance, permutation)
    ranking = tuple(permutation)
    dcg = math.fsum(
        qrels[ordinal] / math.log2(rank + 1)
        for rank, ordinal in enumerate(ranking[:NDCG_CUTOFF], start=1)
    )
    ideal = tuple(sorted(qrels, key=lambda ordinal: (-qrels[ordinal], ordinal)))
    idcg = math.fsum(
        qrels[ordinal] / math.log2(rank + 1)
        for rank, ordinal in enumerate(ideal[:NDCG_CUTOFF], start=1)
    )
    if idcg == 0.0:
        return 0.0
    value = dcg / idcg
    if value > 1.0 and value <= 1.0 + 1.0e-15:
        value = 1.0
    if not 0.0 <= value <= 1.0:
        raise BircoP1CoreError("computed nDCG is outside [0, 1]")
    return value


ndcg_at_10 = linear_gain_ndcg_at_10


def recall_at_5(
    permutation: Sequence[int], relevance: Mapping[int, object]
) -> float:
    """Official-style Recall@5 using source relevance >= 1 as relevant."""

    qrels = _validated_relevance(relevance, permutation)
    relevant = frozenset(
        ordinal for ordinal, value in qrels.items() if value >= RECALL_RELEVANCE_THRESHOLD
    )
    if not relevant:
        return 0.0
    hits = sum(ordinal in relevant for ordinal in tuple(permutation)[:RECALL_CUTOFF])
    return hits / len(relevant)


def integer_utility_from_ndcg(ndcg: object) -> int:
    value = _finite_float(ndcg, "nDCG")
    if not 0.0 <= value <= 1.0:
        raise BircoP1CoreError("nDCG must lie in [0, 1]")
    return math.floor(INTEGER_UTILITY_SCALE * value)


def integer_ndcg_at_10(
    permutation: Sequence[int], relevance: Mapping[int, object]
) -> int:
    return integer_utility_from_ndcg(linear_gain_ndcg_at_10(permutation, relevance))


@dataclass(frozen=True)
class PublicScoreReport:
    """Aggregate-only score report: no plan text, IDs, qrels, or permutations."""

    candidate_count: int
    relevant_candidate_count: int
    ndcg_at_10: float
    recall_at_5: float
    integer_utility: int

    def __post_init__(self) -> None:
        candidate_count = _strict_int(
            self.candidate_count, "score-report candidate count", minimum=1
        )
        relevant_count = _strict_int(
            self.relevant_candidate_count,
            "score-report relevant candidate count",
            minimum=0,
        )
        if relevant_count > candidate_count:
            raise BircoP1CoreError("relevant count exceeds candidate count")
        ndcg = _finite_float(self.ndcg_at_10, "score-report nDCG")
        recall = _finite_float(self.recall_at_5, "score-report Recall")
        if not (0.0 <= ndcg <= 1.0 and 0.0 <= recall <= 1.0):
            raise BircoP1CoreError("public score metrics must lie in [0, 1]")
        utility = _strict_int(self.integer_utility, "integer utility", minimum=0)
        if utility != integer_utility_from_ndcg(ndcg):
            raise BircoP1CoreError("public score integer utility is inconsistent")

    def payload(self) -> dict[str, object]:
        return {
            "schema": f"{VERSION}_public_score_report",
            "candidate_count": self.candidate_count,
            "relevant_candidate_count": self.relevant_candidate_count,
            "nDCG_at_10": self.ndcg_at_10,
            "Recall_at_5": self.recall_at_5,
            "integer_utility": self.integer_utility,
            "gain": "linear_source_relevance",
            "recall_relevance_threshold": RECALL_RELEVANCE_THRESHOLD,
        }


ScoreReport = PublicScoreReport


def score_full_permutation(
    permutation: Sequence[int], relevance: Mapping[int, object]
) -> PublicScoreReport:
    qrels = _validated_relevance(relevance, permutation)
    ndcg = linear_gain_ndcg_at_10(permutation, qrels)
    recall = recall_at_5(permutation, qrels)
    return PublicScoreReport(
        candidate_count=len(qrels),
        relevant_candidate_count=sum(
            value >= RECALL_RELEVANCE_THRESHOLD for value in qrels.values()
        ),
        ndcg_at_10=ndcg,
        recall_at_5=recall,
        integer_utility=integer_utility_from_ndcg(ndcg),
    )


score_ranking = score_full_permutation


def descriptive_binomial_tail(gains: int, harms: int) -> Fraction:
    """One-sided gain-vs-harm Binomial(1/2) tail, excluding ties."""

    gains_checked = _strict_int(gains, "gain count", minimum=0)
    harms_checked = _strict_int(harms, "harm count", minimum=0)
    nonzero = gains_checked + harms_checked
    if nonzero == 0:
        return Fraction(1)
    numerator = sum(
        math.comb(nonzero, value) for value in range(gains_checked, nonzero + 1)
    )
    return Fraction(numerator, 2**nonzero)


binomial_gain_vs_harm_tail = descriptive_binomial_tail


@dataclass(frozen=True)
class PairedUtilitySummary:
    item_count: int
    total_integer_delta: int
    gains: int
    harms: int
    ties: int
    descriptive_reference_tail: Fraction

    def __post_init__(self) -> None:
        for field, value in (
            ("item count", self.item_count),
            ("gain count", self.gains),
            ("harm count", self.harms),
            ("tie count", self.ties),
        ):
            _strict_int(value, field, minimum=0)
        if type(self.total_integer_delta) is not int:
            raise BircoP1CoreError("total integer delta must be an integer")
        if self.gains + self.harms + self.ties != self.item_count:
            raise BircoP1CoreError("paired utility counts do not sum to item count")
        if self.descriptive_reference_tail != descriptive_binomial_tail(
            self.gains, self.harms
        ):
            raise BircoP1CoreError("descriptive binomial tail is inconsistent")

    @property
    def tail_at_most_alpha(self) -> bool:
        return self.descriptive_reference_tail <= PROMOTION_ALPHA

    def payload(self) -> dict[str, object]:
        tail = self.descriptive_reference_tail
        return {
            "item_count": self.item_count,
            "total_integer_delta": self.total_integer_delta,
            "gains": self.gains,
            "harms": self.harms,
            "ties": self.ties,
            "ties_excluded_from_tail": True,
            "descriptive_binomial_tail": {
                "numerator": tail.numerator,
                "denominator": tail.denominator,
            },
            "descriptive_reference_only": True,
        }


def paired_utility_summary(
    challenger: Sequence[int], incumbent: Sequence[int]
) -> PairedUtilitySummary:
    if len(challenger) != len(incumbent) or not challenger:
        raise BircoP1CoreError(
            "paired utility vectors must be nonempty and have equal length"
        )
    challenger_values = tuple(
        _strict_int(value, "challenger integer utility", minimum=0)
        for value in challenger
    )
    incumbent_values = tuple(
        _strict_int(value, "incumbent integer utility", minimum=0)
        for value in incumbent
    )
    if any(value > INTEGER_UTILITY_SCALE for value in challenger_values + incumbent_values):
        raise BircoP1CoreError("paired integer utility exceeds its frozen scale")
    deltas = tuple(
        left - right for left, right in zip(challenger_values, incumbent_values)
    )
    gains = sum(value > 0 for value in deltas)
    harms = sum(value < 0 for value in deltas)
    ties = len(deltas) - gains - harms
    return PairedUtilitySummary(
        item_count=len(deltas),
        total_integer_delta=sum(deltas),
        gains=gains,
        harms=harms,
        ties=ties,
        descriptive_reference_tail=descriptive_binomial_tail(gains, harms),
    )


@dataclass(frozen=True)
class FIdentifiabilityResult:
    item_count: int
    differing_ranking_count: int
    differing_family_count: int
    passed: bool

    @property
    def differing_recipe_count(self) -> int:
        """Compatibility label; the value counts rankings, never recipe IDs."""

        return self.differing_ranking_count


def _formal_permutation(
    value: RecipeRanking | Sequence[int], *, label: str
) -> tuple[int, ...]:
    if isinstance(value, RecipeRanking):
        permutation = value.candidate_ordinals
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        permutation = tuple(value)
    else:
        raise BircoP1CoreError(f"{label} must be a complete candidate permutation")
    if not permutation:
        raise BircoP1CoreError(f"{label} cannot be empty")
    return validate_full_permutation(permutation, len(permutation))


def assess_f_identifiability(
    e4_permutations: Sequence[RecipeRanking | Sequence[int]],
    e0_permutations: Sequence[RecipeRanking | Sequence[int]],
    families: Sequence[Hashable],
    *,
    require_formal_count: bool = True,
) -> FIdentifiabilityResult:
    """Assess label-free behavior from complete selected permutations.

    Recipe names are intentionally not accepted.  Selecting different recipe
    IDs with the same complete ordering is the same observable behavior and
    therefore does not contribute to the preregistered F_search threshold.
    """

    if not (len(e4_permutations) == len(e0_permutations) == len(families)):
        raise BircoP1CoreError("F_search permutation and family vectors differ in length")
    if require_formal_count and len(e4_permutations) != 30:
        raise BircoP1CoreError("formal F_search identifiability requires exactly 30 items")
    if not e4_permutations:
        raise BircoP1CoreError("F_search identifiability vectors are empty")
    differing_rows: list[int] = []
    for index, (e4_value, e0_value) in enumerate(
        zip(e4_permutations, e0_permutations)
    ):
        e4 = _formal_permutation(e4_value, label="E4 F_search ranking")
        e0 = _formal_permutation(e0_value, label="E0 F_search ranking")
        if len(e4) != len(e0) or set(e4) != set(e0):
            raise BircoP1CoreError(
                "paired F_search rankings must share one candidate universe"
            )
        if e4 != e0:
            differing_rows.append(index)
    differing = tuple(differing_rows)
    family_count = len({families[index] for index in differing})
    return FIdentifiabilityResult(
        item_count=len(e4_permutations),
        differing_ranking_count=len(differing),
        differing_family_count=family_count,
        passed=len(differing) >= 3 and family_count >= 2,
    )


@dataclass(frozen=True)
class E4PromotionDecision:
    comparison: PairedUtilitySummary
    f_identifiability_passed: bool
    promoted: bool

    def payload(self) -> dict[str, object]:
        return {
            "schema": f"{VERSION}_A_hold_E4_promotion",
            "comparison": self.comparison.payload(),
            "F_identifiability_passed": self.f_identifiability_passed,
            "positive_total_integer_utility": self.comparison.total_integer_delta > 0,
            "tail_at_most_one_tenth": self.comparison.tail_at_most_alpha,
            "promoted": self.promoted,
        }


def decide_a_hold_e4_promotion(
    e4_integer_utilities: Sequence[int],
    e0_integer_utilities: Sequence[int],
    *,
    f_identifiability_passed: bool,
) -> E4PromotionDecision:
    if type(f_identifiability_passed) is not bool:
        raise BircoP1CoreError("F identifiability decision must be boolean")
    comparison = paired_utility_summary(e4_integer_utilities, e0_integer_utilities)
    promoted = (
        f_identifiability_passed
        and comparison.total_integer_delta > 0
        and comparison.tail_at_most_alpha
    )
    return E4PromotionDecision(comparison, f_identifiability_passed, promoted)


decide_a_hold_promotion = decide_a_hold_e4_promotion


def _family_delta_sums(
    challenger: Sequence[int], incumbent: Sequence[int], families: Sequence[Hashable]
) -> tuple[int, ...]:
    if not (len(challenger) == len(incumbent) == len(families)):
        raise BircoP1CoreError("family comparison vectors differ in length")
    grouped: dict[Hashable, int] = {}
    for left, right, family in zip(challenger, incumbent, families):
        left_checked = _strict_int(left, "challenger integer utility", minimum=0)
        right_checked = _strict_int(right, "incumbent integer utility", minimum=0)
        try:
            grouped[family] = grouped.get(family, 0) + left_checked - right_checked
        except TypeError as exc:
            raise BircoP1CoreError("family labels must be hashable") from exc
    if len(grouped) != 3:
        raise BircoP1CoreError("formal BIRCO decision requires exactly three families")
    # Family labels never enter the returned/public decision.  Sorting by repr
    # merely makes the numeric tuple stable for arbitrary hashable labels.
    return tuple(grouped[key] for key in sorted(grouped, key=repr))


@dataclass(frozen=True)
class RealityPrimaryDecision:
    agent_minus_raw: PairedUtilitySummary
    agent_minus_hipporag: PairedUtilitySummary
    raw_family_integer_deltas: tuple[int, int, int]
    hipporag_family_integer_deltas: tuple[int, int, int]
    passed: bool


def decide_a_hold_reality_primary(
    agent_e0_integer_utilities: Sequence[int],
    raw_integer_utilities: Sequence[int],
    hipporag_integer_utilities: Sequence[int],
    families: Sequence[Hashable],
) -> RealityPrimaryDecision:
    raw = paired_utility_summary(agent_e0_integer_utilities, raw_integer_utilities)
    hippo = paired_utility_summary(
        agent_e0_integer_utilities, hipporag_integer_utilities
    )
    raw_families = _family_delta_sums(
        agent_e0_integer_utilities, raw_integer_utilities, families
    )
    hippo_families = _family_delta_sums(
        agent_e0_integer_utilities, hipporag_integer_utilities, families
    )
    passed = (
        raw.total_integer_delta > 0
        and hippo.total_integer_delta > 0
        and all(value > 0 for value in raw_families + hippo_families)
        and raw.tail_at_most_alpha
        and hippo.tail_at_most_alpha
    )
    return RealityPrimaryDecision(raw, hippo, raw_families, hippo_families, passed)


@dataclass(frozen=True)
class MSearchDecision:
    comparison: PairedUtilitySummary
    family_integer_deltas: tuple[int, int, int]
    passed: bool


def decide_m_search_e4_improvement(
    e4_integer_utilities: Sequence[int],
    e0_integer_utilities: Sequence[int],
    families: Sequence[Hashable],
) -> MSearchDecision:
    comparison = paired_utility_summary(e4_integer_utilities, e0_integer_utilities)
    family_deltas = _family_delta_sums(
        e4_integer_utilities, e0_integer_utilities, families
    )
    passed = (
        comparison.total_integer_delta > 0
        and all(value >= 0 for value in family_deltas)
        and sum(value > 0 for value in family_deltas) >= 2
        and comparison.tail_at_most_alpha
    )
    return MSearchDecision(comparison, family_deltas, passed)


decide_m_search = decide_m_search_e4_improvement


__all__ = [
    "VERSION",
    "FACET_TYPES",
    "EDGE_TYPES",
    "FACET_TYPE_DEFAULT_WEIGHTS",
    "RECIPE_IDS",
    "R1_WEIGHTED_MASS",
    "R2_BOTTLENECK",
    "R3_DEPENDENCY_FLOW",
    "R4_CAPACITY_MATCH",
    "FEATURE_ORDER",
    "FORBIDDEN_E4_FEATURES",
    "INTEGER_UTILITY_SCALE",
    "E4_A_FORM_ITEM_COUNT",
    "PROMOTION_ALPHA",
    "BircoP1CoreError",
    "TypedFacet",
    "TypedFacetEdge",
    "TypedFacetPlan",
    "Facet",
    "FacetEdge",
    "FacetPlan",
    "validate_typed_facet_plan",
    "validate_typed_plan",
    "totalize_typed_facet_plan",
    "totalize_typed_plan",
    "FacetEvidence",
    "CandidateFacetEvidence",
    "CandidateFacetEvidenceMatrix",
    "CandidateEvidenceMatrix",
    "validate_candidate_facet_evidence_matrix",
    "validate_candidate_matrix",
    "totalize_candidate_facet_evidence",
    "totalize_candidate_facet_evidence_matrix",
    "totalize_candidate_matrix",
    "CapacityAssignment",
    "solve_capacity_assignment",
    "CandidateRecipeScore",
    "RecipeRanking",
    "rank_candidates",
    "rank_r1_weighted_mass",
    "rank_r2_bottleneck",
    "rank_r3_dependency_flow",
    "rank_r4_capacity_match",
    "build_recipe_rankings",
    "select_e0_recipe",
    "select_e0_ranking",
    "validate_full_permutation",
    "validate_action_features",
    "feature_vector",
    "compute_action_features",
    "action_feature_mapping",
    "E4ActionTrainingRow",
    "E4TrainingSlate",
    "make_e4_training_slate",
    "E4Model",
    "fit_e4_listwise_softmax",
    "fit_e4_model",
    "fit_e4",
    "fit_listwise_softmax_from_mappings",
    "E4Selection",
    "select_e4_recipe",
    "select_e4_ranking",
    "linear_gain_ndcg_at_10",
    "ndcg_at_10",
    "recall_at_5",
    "integer_utility_from_ndcg",
    "integer_ndcg_at_10",
    "PublicScoreReport",
    "ScoreReport",
    "score_full_permutation",
    "score_ranking",
    "descriptive_binomial_tail",
    "binomial_gain_vs_harm_tail",
    "PairedUtilitySummary",
    "paired_utility_summary",
    "FIdentifiabilityResult",
    "assess_f_identifiability",
    "E4PromotionDecision",
    "decide_a_hold_e4_promotion",
    "decide_a_hold_promotion",
    "RealityPrimaryDecision",
    "decide_a_hold_reality_primary",
    "MSearchDecision",
    "decide_m_search_e4_improvement",
    "decide_m_search",
]
