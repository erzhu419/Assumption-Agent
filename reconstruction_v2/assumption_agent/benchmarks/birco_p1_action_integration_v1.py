"""Source-free glue between BIRCO selector, semantic actions, and core algebra.

The private selector exposes one label-free action item containing an opaque
work ID and the exact objective/query/common candidate projection.  This module
reopens only that projection, constructs the frozen 24-candidate semantic
batches, verifies terminal action objects, converts compact semantic matrix
rows to the strict typed core, and produces the four Agent actions plus E0/E4
selection.

There is deliberately no qrel parameter, source row, file path, network client,
provider credential, retry, or persistence operation in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Sequence

from replication_runtime.birco_gpt54_semantic_v1 import contract as semantic

from . import birco_p1_typed_constraint_e4_core_v1 as core


VERSION = "birco_p1_action_integration_v1"
SELECTOR_ACTION_ITEM_SCHEMA = (
    "birco_p1_private_selection_v1_label_free_action_item_v1"
)
HIPPORAG_INPUT_SCHEMA = "birco_official_hipporag_candidate_retrieval_v1_input"
SEMANTIC_TRANSPORT_ID = "urllib_openai_compatible_chat_completions_one_request_v1"
PROVIDER_ORIGIN = "https://ruoli.dev"
KEY_COMMITMENT_VERSION = "birco_p1_provider_key_hmac_sha256_v1"

_ACTION_ITEM_FIELDS = frozenset(
    {
        "schema",
        "block_ordinal",
        "work_id",
        "candidate_count",
        "common_projection_sha256",
        "hipporag_input",
    }
)
_HIPPORAG_INPUT_FIELDS = frozenset(
    {
        "schema",
        "work_id",
        "objective",
        "query",
        "documents",
        "common_projection_sha256",
    }
)
_DOCUMENT_FIELDS = frozenset({"ordinal", "text"})
_TERMINAL_COMMON_FIELDS = frozenset(
    {
        "action",
        "attempt_count",
        "generation_valid",
        "input_sha256",
        "mode",
        "model_request_sha256",
        "provider",
        "raw_completion_persisted",
        "response_sha256",
        "retry_replay_resample_or_provider_switch_count",
        "schema",
        "terminal_category",
        "transport",
        "transport_succeeded",
        "work_id",
        "self_sha256",
    }
)
_TERMINAL_BATCH_FIELDS = frozenset(
    {
        "batch_count",
        "batch_ordinal",
        "batch_common_projection_sha256",
        "pool_candidate_count",
        "pool_common_projection_sha256",
    }
)
_PROVIDER_FIELDS = frozenset(
    {
        "api_key_hmac_sha256",
        "api_origin",
        "key_commitment_version",
        "model",
        "provider_label",
        "secret_persisted",
    }
)
_WORK_ID = re.compile(r"birco-work-v1-[0-9a-f]{64}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PROVIDER_LABEL = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")


class BircoP1ActionIntegrationError(ValueError):
    """Fail-closed error for selector, batch, terminal, or merge drift."""


def _exact_fields(
    value: Mapping[str, object], expected: frozenset[str], label: str
) -> None:
    supplied = set(value)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise BircoP1ActionIntegrationError(
            f"{label} schema drifted; missing={missing}, extra={extra}"
        )


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise BircoP1ActionIntegrationError(f"{label} must be a lowercase SHA-256")
    return value


def _full_pool_projection_hash(
    *,
    objective: str,
    query: str,
    candidates: Sequence[semantic.CandidateProjection],
) -> str:
    """Hash the selector-common full pool without the semantic batch cap."""

    return semantic.semantic_hash(
        {
            "documents": [
                {"ordinal": row.ordinal, "text": row.projection_text}
                for row in candidates
            ],
            "objective": objective,
            "query": query,
        }
    )


@dataclass(frozen=True)
class ValidatedActionItem:
    """Exact label-free selector projection, with source identities absent."""

    block_ordinal: int
    work_id: str
    objective: str
    query: str
    candidates: tuple[semantic.CandidateProjection, ...]
    common_projection_sha256: str

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def batch_count(self) -> int:
        return math.ceil(
            self.candidate_count / semantic.MAXIMUM_CANDIDATES_PER_BATCH
        )

    def batches(self) -> tuple[tuple[semantic.CandidateProjection, ...], ...]:
        width = semantic.MAXIMUM_CANDIDATES_PER_BATCH
        return tuple(
            self.candidates[start : start + width]
            for start in range(0, self.candidate_count, width)
        )


def validate_selector_action_item(value: Mapping[str, object]) -> ValidatedActionItem:
    """Validate one exact private-selector action item without label access."""

    if not isinstance(value, Mapping):
        raise BircoP1ActionIntegrationError("selector action item must be a mapping")
    _exact_fields(value, _ACTION_ITEM_FIELDS, "selector action item")
    if value.get("schema") != SELECTOR_ACTION_ITEM_SCHEMA:
        raise BircoP1ActionIntegrationError("selector action item schema is invalid")
    block_ordinal = value.get("block_ordinal")
    if type(block_ordinal) is not int or not 0 <= block_ordinal < 30:
        raise BircoP1ActionIntegrationError("selector block ordinal is invalid")
    work_id = value.get("work_id")
    if not isinstance(work_id, str) or _WORK_ID.fullmatch(work_id) is None:
        raise BircoP1ActionIntegrationError("selector opaque work ID is invalid")
    nested = value.get("hipporag_input")
    if not isinstance(nested, Mapping):
        raise BircoP1ActionIntegrationError("selector common projection is absent")
    _exact_fields(nested, _HIPPORAG_INPUT_FIELDS, "selector common projection")
    if (
        nested.get("schema") != HIPPORAG_INPUT_SCHEMA
        or nested.get("work_id") != work_id
    ):
        raise BircoP1ActionIntegrationError("selector work binding drifted")
    objective = nested.get("objective")
    query = nested.get("query")
    if not isinstance(objective, str) or not isinstance(query, str):
        raise BircoP1ActionIntegrationError("selector objective/query is malformed")
    # The public semantic constructor supplies the exact text bounds.
    semantic.planner_input(work_id=work_id, objective=objective, query=query)

    documents = nested.get("documents")
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise BircoP1ActionIntegrationError("selector documents must be an array")
    candidate_count = value.get("candidate_count")
    if (
        type(candidate_count) is not int
        or candidate_count != len(documents)
        or not semantic.MINIMUM_POOL_CANDIDATES
        <= candidate_count
        <= semantic.MAXIMUM_POOL_CANDIDATES
    ):
        raise BircoP1ActionIntegrationError("selector candidate count drifted")
    candidates: list[semantic.CandidateProjection] = []
    for ordinal, document in enumerate(documents):
        if not isinstance(document, Mapping):
            raise BircoP1ActionIntegrationError("selector document must be a mapping")
        _exact_fields(document, _DOCUMENT_FIELDS, "selector document")
        if document.get("ordinal") != ordinal or not isinstance(
            document.get("text"), str
        ):
            raise BircoP1ActionIntegrationError(
                "selector document ordinals/text drifted"
            )
        try:
            candidates.append(
                semantic.candidate_projection_from_text(
                    str(document["text"]), candidate_ordinal=ordinal
                )
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise BircoP1ActionIntegrationError(
                "selector candidate projection is not canonical"
            ) from exc

    claimed_top = _sha256(
        value.get("common_projection_sha256"),
        "selector common projection commitment",
    )
    claimed_nested = _sha256(
        nested.get("common_projection_sha256"),
        "nested common projection commitment",
    )
    expected = _full_pool_projection_hash(
        objective=objective, query=query, candidates=candidates
    )
    if claimed_top != claimed_nested or claimed_top != expected:
        raise BircoP1ActionIntegrationError(
            "selector full-pool common projection commitment drifted"
        )
    return ValidatedActionItem(
        block_ordinal=block_ordinal,
        work_id=work_id,
        objective=objective,
        query=query,
        candidates=tuple(candidates),
        common_projection_sha256=expected,
    )


@dataclass(frozen=True)
class CanonicalActionInputs:
    """One planner input and all frozen full-pool RAW batch inputs."""

    action_item: ValidatedActionItem
    planner_input: Mapping[str, object]
    raw_inputs: tuple[Mapping[str, object], ...]
    batch_candidate_ordinals: tuple[tuple[int, ...], ...]

    @property
    def pool_common_projection_sha256(self) -> str:
        return self.action_item.common_projection_sha256


def prepare_canonical_action_inputs(
    value: Mapping[str, object] | ValidatedActionItem,
) -> CanonicalActionInputs:
    """Build canonical planner/RAW inputs and exact 24-candidate slices."""

    action = (
        value
        if isinstance(value, ValidatedActionItem)
        else validate_selector_action_item(value)
    )
    planner = semantic.planner_input(
        work_id=action.work_id,
        objective=action.objective,
        query=action.query,
    )
    batches = action.batches()
    raw_inputs = tuple(
        semantic.raw_input(
            work_id=action.work_id,
            objective=action.objective,
            query=action.query,
            candidates=batch,
            batch_ordinal=batch_ordinal,
            batch_count=action.batch_count,
            pool_candidate_count=action.candidate_count,
            pool_common_projection_sha256=action.common_projection_sha256,
        )
        for batch_ordinal, batch in enumerate(batches)
    )
    ordinals = tuple(
        tuple(candidate.ordinal for candidate in batch) for batch in batches
    )
    expected = tuple(range(action.candidate_count))
    if tuple(ordinal for batch in ordinals for ordinal in batch) != expected:
        raise BircoP1ActionIntegrationError("fixed candidate slicing drifted")
    return CanonicalActionInputs(action, planner, raw_inputs, ordinals)


def _validate_provider(value: object) -> None:
    if not isinstance(value, Mapping):
        raise BircoP1ActionIntegrationError("semantic provider receipt is absent")
    _exact_fields(value, _PROVIDER_FIELDS, "semantic provider receipt")
    if (
        _SHA256.fullmatch(str(value.get("api_key_hmac_sha256"))) is None
        or value.get("api_origin") != PROVIDER_ORIGIN
        or value.get("key_commitment_version") != KEY_COMMITMENT_VERSION
        or value.get("model") != semantic.MODEL_ID
        or not isinstance(value.get("provider_label"), str)
        or _PROVIDER_LABEL.fullmatch(str(value["provider_label"])) is None
        or value.get("secret_persisted") is not False
    ):
        raise BircoP1ActionIntegrationError("semantic provider receipt drifted")


def _validate_semantic_terminal(
    value: Mapping[str, object],
    *,
    mode: str,
    expected_input: Mapping[str, object],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BircoP1ActionIntegrationError("semantic terminal must be a mapping")
    expected_fields = _TERMINAL_COMMON_FIELDS | (
        _TERMINAL_BATCH_FIELDS if mode in {"matrix", "raw"} else frozenset()
    )
    _exact_fields(value, expected_fields, "semantic terminal")
    body = dict(value)
    claimed_self = _sha256(body.pop("self_sha256", None), "semantic terminal self hash")
    if semantic.semantic_hash(body) != claimed_self:
        raise BircoP1ActionIntegrationError("semantic terminal self hash drifted")
    if (
        mode not in {"plan", "matrix", "raw"}
        or value.get("schema") != semantic.TERMINAL_OUTPUT_SCHEMA
        or value.get("mode") != mode
        or value.get("work_id") != expected_input.get("work_id")
        or value.get("attempt_count") != 1
        or value.get("retry_replay_resample_or_provider_switch_count") != 0
        or value.get("raw_completion_persisted") is not False
        or value.get("transport") != SEMANTIC_TRANSPORT_ID
        or type(value.get("generation_valid")) is not bool
        or type(value.get("transport_succeeded")) is not bool
    ):
        raise BircoP1ActionIntegrationError("semantic terminal control fields drifted")
    if value.get("input_sha256") != semantic.semantic_hash(expected_input):
        raise BircoP1ActionIntegrationError("semantic terminal input binding drifted")
    _sha256(value.get("model_request_sha256"), "semantic model request hash")
    response = value.get("response_sha256")
    if response is not None:
        _sha256(response, "semantic response hash")
    category = value.get("terminal_category")
    if category not in {
        "success",
        "output_totalized",
        "transport_unavailable",
        "provider_protocol_totalized",
    }:
        raise BircoP1ActionIntegrationError("semantic terminal category drifted")
    if (category == "success") is not bool(value["generation_valid"]):
        raise BircoP1ActionIntegrationError(
            "semantic generation-valid/category binding drifted"
        )
    _validate_provider(value.get("provider"))
    action = value.get("action")
    if not isinstance(action, Mapping):
        raise BircoP1ActionIntegrationError("semantic terminal action is absent")

    if mode in {"matrix", "raw"}:
        for field in _TERMINAL_BATCH_FIELDS:
            if value.get(field) != expected_input.get(field):
                raise BircoP1ActionIntegrationError(
                    f"semantic terminal {field} binding drifted"
                )
    return action


def _semantic_plan_from_terminal(
    terminal: Mapping[str, object],
    *,
    expected_input: Mapping[str, object],
) -> semantic.Plan:
    action = _validate_semantic_terminal(
        terminal, mode="plan", expected_input=expected_input
    )
    _exact_fields(action, frozenset({"plan"}), "semantic planner action")
    value = action.get("plan")
    if not isinstance(value, Mapping):
        raise BircoP1ActionIntegrationError("semantic planner payload is absent")
    _exact_fields(
        value,
        frozenset({"facets", "edges", "generation_valid"}),
        "semantic plan",
    )
    raw_facets = value.get("facets")
    raw_edges = value.get("edges")
    if (
        isinstance(raw_facets, (str, bytes))
        or not isinstance(raw_facets, Sequence)
        or isinstance(raw_edges, (str, bytes))
        or not isinstance(raw_edges, Sequence)
    ):
        raise BircoP1ActionIntegrationError("semantic plan rows are malformed")
    try:
        facets = tuple(
            semantic.Facet(
                ordinal=row["ordinal"],
                facet_type=row["type"],
                text=row["text"],
                weight=row["weight"],
            )
            for row in raw_facets
            if isinstance(row, Mapping)
            and set(row) == {"ordinal", "type", "text", "weight"}
        )
        edges = tuple(
            semantic.PlanEdge(
                source=row["source"], target=row["target"], edge_type=row["type"]
            )
            for row in raw_edges
            if isinstance(row, Mapping) and set(row) == {"source", "target", "type"}
        )
        if len(facets) != len(raw_facets) or len(edges) != len(raw_edges):
            raise BircoP1ActionIntegrationError("semantic plan row shape drifted")
        plan = semantic.Plan(facets, edges, value.get("generation_valid"))
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise BircoP1ActionIntegrationError("semantic plan is invalid") from exc
    if plan.payload() != dict(value):
        raise BircoP1ActionIntegrationError("semantic plan is not canonical")
    if plan.generation_valid is not terminal.get("generation_valid"):
        raise BircoP1ActionIntegrationError("planner validity binding drifted")
    return plan


def _core_plan(plan: semantic.Plan) -> core.TypedFacetPlan:
    """Convert one already-canonical semantic plan to canonical core order."""

    try:
        facets = tuple(
            core.TypedFacet(row.ordinal, row.facet_type, row.text, row.weight)
            for row in plan.facets
        )
        edges = tuple(
            sorted(
                (
                    core.TypedFacetEdge(row.source, row.target, row.edge_type)
                    for row in plan.edges
                ),
                key=lambda row: (
                    row.source_facet_ordinal,
                    row.target_facet_ordinal,
                    core.EDGE_TYPES.index(row.edge_type),
                ),
            )
        )
        return core.TypedFacetPlan(facets, edges)
    except (TypeError, ValueError) as exc:
        raise BircoP1ActionIntegrationError(
            "semantic plan is incompatible with the strict core plan"
        ) from exc


@dataclass(frozen=True)
class CanonicalMatrixInputs:
    """Canonical semantic/core plan and every frozen matrix batch input."""

    prepared: CanonicalActionInputs
    semantic_plan: semantic.Plan
    core_plan: core.TypedFacetPlan
    matrix_inputs: tuple[Mapping[str, object], ...]
    plan_terminal_self_sha256: str


def build_canonical_matrix_inputs(
    prepared: CanonicalActionInputs,
    plan_terminal: Mapping[str, object],
) -> CanonicalMatrixInputs:
    if not isinstance(prepared, CanonicalActionInputs):
        raise BircoP1ActionIntegrationError("canonical action inputs are required")
    plan = _semantic_plan_from_terminal(
        plan_terminal, expected_input=prepared.planner_input
    )
    typed_plan = _core_plan(plan)
    action = prepared.action_item
    matrix_inputs = tuple(
        semantic.matrix_input(
            work_id=action.work_id,
            objective=action.objective,
            query=action.query,
            plan=plan,
            candidates=batch,
            batch_ordinal=batch_ordinal,
            batch_count=action.batch_count,
            pool_candidate_count=action.candidate_count,
            pool_common_projection_sha256=action.common_projection_sha256,
        )
        for batch_ordinal, batch in enumerate(action.batches())
    )
    return CanonicalMatrixInputs(
        prepared=prepared,
        semantic_plan=plan,
        core_plan=typed_plan,
        matrix_inputs=matrix_inputs,
        plan_terminal_self_sha256=_sha256(
            plan_terminal.get("self_sha256"), "planner terminal self hash"
        ),
    )


def _compact_matrix_candidate(
    value: object,
    *,
    candidate: semantic.CandidateProjection,
    facet_count: int,
) -> core.CandidateFacetEvidence:
    if not isinstance(value, Mapping):
        raise BircoP1ActionIntegrationError("matrix candidate row must be a mapping")
    _exact_fields(value, frozenset({"ordinal", "rows"}), "matrix candidate row")
    if value.get("ordinal") != candidate.ordinal:
        raise BircoP1ActionIntegrationError("matrix candidate ordinal drifted")
    compact = value.get("rows")
    if isinstance(compact, (str, bytes)) or not isinstance(compact, Sequence):
        raise BircoP1ActionIntegrationError("compact matrix rows are absent")
    if len(compact) != facet_count:
        raise BircoP1ActionIntegrationError("compact matrix facet width drifted")
    converted: list[core.FacetEvidence] = []
    for facet_ordinal, cell in enumerate(compact):
        if (
            isinstance(cell, (str, bytes))
            or not isinstance(cell, Sequence)
            or len(cell) != 3
        ):
            raise BircoP1ActionIntegrationError("compact facet row must have width three")
        support, contradiction, evidence = cell
        if (
            type(support) is not int
            or type(contradiction) is not int
            or not 0 <= support <= 4
            or not 0 <= contradiction <= 4
            or (
                evidence is not None
                and (
                    type(evidence) is not int
                    or not 0 <= evidence < len(candidate.evidence_units)
                )
            )
        ):
            raise BircoP1ActionIntegrationError("compact facet value is invalid")
        # Facet array index is the only conversion authority.  The semantic
        # plan is canonical before matrix inputs exist, so no raw-plan ordinal
        # or text remapping is possible here.
        converted.append(
            core.FacetEvidence(
                facet_ordinal=facet_ordinal,
                support=support,
                contradiction=contradiction,
                evidence_unit_ordinal=evidence,
            )
        )
    return core.CandidateFacetEvidence(
        candidate_ordinal=candidate.ordinal,
        evidence_unit_count=len(candidate.evidence_units),
        facet_evidence=tuple(converted),
    )


def merge_matrix_terminals(
    stage: CanonicalMatrixInputs,
    terminals: Sequence[Mapping[str, object]],
) -> core.CandidateFacetEvidenceMatrix:
    """Strictly merge all terminal batches into one complete core matrix."""

    if not isinstance(stage, CanonicalMatrixInputs):
        raise BircoP1ActionIntegrationError("canonical matrix stage is required")
    if isinstance(terminals, (str, bytes)) or not isinstance(terminals, Sequence):
        raise BircoP1ActionIntegrationError("matrix terminals must be an array")
    if len(terminals) != len(stage.matrix_inputs):
        raise BircoP1ActionIntegrationError("matrix terminal batch count drifted")
    by_batch: dict[int, Mapping[str, object]] = {}
    for terminal in terminals:
        if not isinstance(terminal, Mapping):
            raise BircoP1ActionIntegrationError("matrix terminal must be a mapping")
        batch_ordinal = terminal.get("batch_ordinal")
        if (
            type(batch_ordinal) is not int
            or not 0 <= batch_ordinal < len(stage.matrix_inputs)
            or batch_ordinal in by_batch
        ):
            raise BircoP1ActionIntegrationError(
                "matrix terminal batch ordinal is invalid or duplicated"
            )
        by_batch[batch_ordinal] = terminal

    candidate_rows: list[core.CandidateFacetEvidence] = []
    batches = stage.prepared.action_item.batches()
    for batch_ordinal, expected_input in enumerate(stage.matrix_inputs):
        terminal = by_batch[batch_ordinal]
        action = _validate_semantic_terminal(
            terminal, mode="matrix", expected_input=expected_input
        )
        _exact_fields(action, frozenset({"matrix"}), "semantic matrix action")
        matrix = action.get("matrix")
        if isinstance(matrix, (str, bytes)) or not isinstance(matrix, Sequence):
            raise BircoP1ActionIntegrationError("semantic matrix payload is absent")
        expected_candidates = batches[batch_ordinal]
        if len(matrix) != len(expected_candidates):
            raise BircoP1ActionIntegrationError("semantic matrix batch width drifted")
        candidate_rows.extend(
            _compact_matrix_candidate(
                value,
                candidate=candidate,
                facet_count=len(stage.core_plan.facets),
            )
            for value, candidate in zip(matrix, expected_candidates)
        )

    result = core.CandidateFacetEvidenceMatrix(tuple(candidate_rows))
    return core.validate_candidate_matrix(result, stage.core_plan)


@dataclass(frozen=True)
class RawActionResult:
    """Complete RAW integer scores and their ordinal-tied full permutation."""

    scores_by_candidate_ordinal: tuple[int, ...]
    candidate_ordinals: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.scores_by_candidate_ordinal:
            raise BircoP1ActionIntegrationError("RAW score vector is empty")
        if any(type(value) is not int or not 0 <= value <= 100 for value in self.scores_by_candidate_ordinal):
            raise BircoP1ActionIntegrationError("RAW score is outside integer 0-100")
        core.validate_full_permutation(
            self.candidate_ordinals, len(self.scores_by_candidate_ordinal)
        )
        expected = tuple(
            sorted(
                range(len(self.scores_by_candidate_ordinal)),
                key=lambda ordinal: (
                    -self.scores_by_candidate_ordinal[ordinal], ordinal
                ),
            )
        )
        if self.candidate_ordinals != expected:
            raise BircoP1ActionIntegrationError("RAW ranking/score binding drifted")


def merge_raw_terminals(
    prepared: CanonicalActionInputs,
    terminals: Sequence[Mapping[str, object]],
) -> RawActionResult:
    """Strictly merge every RAW batch and rank the entire candidate pool."""

    if not isinstance(prepared, CanonicalActionInputs):
        raise BircoP1ActionIntegrationError("canonical action inputs are required")
    if isinstance(terminals, (str, bytes)) or not isinstance(terminals, Sequence):
        raise BircoP1ActionIntegrationError("RAW terminals must be an array")
    if len(terminals) != len(prepared.raw_inputs):
        raise BircoP1ActionIntegrationError("RAW terminal batch count drifted")
    by_batch: dict[int, Mapping[str, object]] = {}
    for terminal in terminals:
        if not isinstance(terminal, Mapping):
            raise BircoP1ActionIntegrationError("RAW terminal must be a mapping")
        batch_ordinal = terminal.get("batch_ordinal")
        if (
            type(batch_ordinal) is not int
            or not 0 <= batch_ordinal < len(prepared.raw_inputs)
            or batch_ordinal in by_batch
        ):
            raise BircoP1ActionIntegrationError(
                "RAW terminal batch ordinal is invalid or duplicated"
            )
        by_batch[batch_ordinal] = terminal

    scores: list[int] = []
    batches = prepared.action_item.batches()
    for batch_ordinal, expected_input in enumerate(prepared.raw_inputs):
        action = _validate_semantic_terminal(
            by_batch[batch_ordinal], mode="raw", expected_input=expected_input
        )
        _exact_fields(action, frozenset({"scores"}), "semantic RAW action")
        rows = action.get("scores")
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise BircoP1ActionIntegrationError("semantic RAW scores are absent")
        expected_candidates = batches[batch_ordinal]
        if len(rows) != len(expected_candidates):
            raise BircoP1ActionIntegrationError("semantic RAW batch width drifted")
        for row, candidate in zip(rows, expected_candidates):
            if not isinstance(row, Mapping):
                raise BircoP1ActionIntegrationError("RAW score row must be a mapping")
            _exact_fields(row, frozenset({"ordinal", "score"}), "RAW score row")
            score = row.get("score")
            if (
                row.get("ordinal") != candidate.ordinal
                or type(score) is not int
                or not 0 <= score <= 100
            ):
                raise BircoP1ActionIntegrationError(
                    "RAW candidate ordinal or integer score drifted"
                )
            scores.append(score)
    if len(scores) != prepared.action_item.candidate_count:
        raise BircoP1ActionIntegrationError("RAW merge did not cover the full pool")
    ranking = tuple(
        sorted(range(len(scores)), key=lambda ordinal: (-scores[ordinal], ordinal))
    )
    return RawActionResult(tuple(scores), ranking)


@dataclass(frozen=True)
class AgentActionEvaluation:
    """Content-free four-action result with frozen E0 and E4 choices."""

    plan: core.TypedFacetPlan
    matrix: core.CandidateFacetEvidenceMatrix
    rankings: tuple[core.RecipeRanking, ...]
    action_features: tuple[tuple[str, tuple[float, ...]], ...]
    e0_recipe_id: str
    e0_ranking: core.RecipeRanking
    e4_selection: core.E4Selection
    e4_ranking: core.RecipeRanking

    def ranking_by_recipe(self) -> dict[str, core.RecipeRanking]:
        return {ranking.recipe_id: ranking for ranking in self.rankings}

    def features_by_recipe(self) -> dict[str, tuple[float, ...]]:
        return dict(self.action_features)


def produce_e0_e4_evaluation(
    stage: CanonicalMatrixInputs,
    matrix_terminals: Sequence[Mapping[str, object]],
    *,
    e4_model: core.E4Model,
) -> AgentActionEvaluation:
    """Produce four full rankings/features and the frozen E0/E4 selections."""

    if not isinstance(e4_model, core.E4Model):
        raise BircoP1ActionIntegrationError("a frozen E4 model is required")
    matrix = merge_matrix_terminals(stage, matrix_terminals)
    ranking_map = core.build_recipe_rankings(stage.core_plan, matrix)
    features = {
        recipe: core.compute_action_features(
            stage.core_plan, matrix, ranking_map[recipe]
        )
        for recipe in core.RECIPE_IDS
    }
    e0_recipe = core.select_e0_recipe(stage.core_plan)
    e0_ranking = ranking_map[e0_recipe]
    e4_selection = core.select_e4_recipe(
        e4_model, features, e0_recipe_id=e0_recipe
    )
    return AgentActionEvaluation(
        plan=stage.core_plan,
        matrix=matrix,
        rankings=tuple(ranking_map[recipe] for recipe in core.RECIPE_IDS),
        action_features=tuple((recipe, features[recipe]) for recipe in core.RECIPE_IDS),
        e0_recipe_id=e0_recipe,
        e0_ranking=e0_ranking,
        e4_selection=e4_selection,
        e4_ranking=ranking_map[e4_selection.selected_recipe_id],
    )


__all__ = [
    "VERSION",
    "SELECTOR_ACTION_ITEM_SCHEMA",
    "HIPPORAG_INPUT_SCHEMA",
    "BircoP1ActionIntegrationError",
    "ValidatedActionItem",
    "validate_selector_action_item",
    "CanonicalActionInputs",
    "prepare_canonical_action_inputs",
    "CanonicalMatrixInputs",
    "build_canonical_matrix_inputs",
    "merge_matrix_terminals",
    "RawActionResult",
    "merge_raw_terminals",
    "AgentActionEvaluation",
    "produce_e0_e4_evaluation",
]
