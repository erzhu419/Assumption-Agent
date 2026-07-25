"""Source-free action integration for ``MMQA_P1_LOCAL_PROOF_E5_V1``.

The integration accepts one anonymous, label-free work item and externally
computed MiniLM/cross-encoder coordinates plus frozen typed-anchor flags.  It
does not read a source, load or call a model, access a network, retry work, or
accept qids, question families/types, answers, support annotations, or source
metadata IDs.

Every arm receives the same immutable closure units and byte-identical local
ordinal vector.  The module forms the core graph, reciprocal exact structural
links, bounded closure and proof bundles, then emits E0, optional E5, and RAW
rankings plus a label-free action-feature archive.  Gold is admitted only by
the separate late-scoring function after action formation is complete.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence
import unicodedata

from . import mmqa_p1_typed_proof_e5_core_v1 as core


VERSION = "mmqa_p1_action_integration_v1"
STUDY_ID = core.STUDY_ID
STUDY_DESIGN_SELF_SHA256 = (
    "eefa61986bd2f58efa26564dc0709728e0323660f23ae532819f4fa98f0601b3"
)
ANONYMOUS_WORK_ITEM_SCHEMA = f"{VERSION}_anonymous_work_item"
UNIT_COORDINATES_SCHEMA = f"{VERSION}_unit_coordinates"
ACTION_FEATURE_ARCHIVE_SCHEMA = f"{VERSION}_action_feature_archive"

MAXIMUM_ROWS_BEFORE_SELECTION = 1024
MAXIMUM_ROW_NODES = 48
MAXIMUM_TEXT_NODES = 48
MINIMUM_CLOSURE_NODES = 5

_WORK_ITEM_FIELDS = frozenset(
    {"schema", "question", "rows", "texts", "exact_row_text_links"}
)
_UNIT_FIELDS = frozenset({"ordinal", "serialized_content"})
_LINK_FIELDS = frozenset({"row_ordinal", "text_ordinal"})
_COORDINATE_FIELDS = frozenset(
    {
        "schema",
        "ordinal",
        "minilm_similarity",
        "cross_encoder_relevance",
        "entity_anchor",
        "relation_anchor",
        "numeric_or_temporal_anchor",
    }
)

FORBIDDEN_ACTION_INPUT_FIELDS = frozenset(
    {
        "answer",
        "answers",
        "family",
        "family_id",
        "gold",
        "gold_row",
        "gold_text",
        "intermediate_answer",
        "intermediate_answers",
        "entities",
        "image_doc_ids",
        "metadata",
        "metadata_id",
        "modalities",
        "qid",
        "question_id",
        "question_type",
        "row_id",
        "source_id",
        "support",
        "support_id",
        "supporting_context",
        "table_id",
        "text_doc_ids",
        "type",
    }
)


class MmqaP1ActionIntegrationError(ValueError):
    """The anonymous action, coordinates, closure, or late score drifted."""


def _exact_fields(
    value: Mapping[str, object], expected: frozenset[str], label: str
) -> None:
    supplied = set(value)
    if supplied != expected:
        forbidden = sorted(supplied.intersection(FORBIDDEN_ACTION_INPUT_FIELDS))
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise MmqaP1ActionIntegrationError(
            f"{label} schema drifted; forbidden={forbidden}, "
            f"missing={missing}, extra={extra}"
        )


def _strict_int(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise MmqaP1ActionIntegrationError(
            f"{field} must be an exact integer at least {minimum}"
        )
    return int(value)


def _canonical_text(value: object, field: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise MmqaP1ActionIntegrationError(f"{field} must be exact text")
    if (
        not value
        or len(value) > maximum
        or value != value.strip()
        or "\x00" in value
        or "\r" in value
        or unicodedata.normalize("NFKC", value) != value
    ):
        raise MmqaP1ActionIntegrationError(
            f"{field} is empty, noncanonical, or outside its frozen bound"
        )
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MmqaP1ActionIntegrationError(
            "action value is not canonical JSON"
        ) from exc


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class SerializedUnit:
    """One source-local ordinal and its already-serialized anonymous content."""

    ordinal: int
    serialized_content: str
    node_type: str

    def __post_init__(self) -> None:
        ordinal = _strict_int(self.ordinal, "unit ordinal")
        content = _canonical_text(
            self.serialized_content, "serialized unit content", maximum=16_384
        )
        if self.node_type not in core.NODE_TYPES:
            raise MmqaP1ActionIntegrationError("unit node type drifted")
        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(self, "serialized_content", content)

    def anonymous_payload(self) -> dict[str, object]:
        # node_type is conveyed by the enclosing rows/texts array and is not an
        # accepted source metadata field.
        return {
            "ordinal": self.ordinal,
            "serialized_content": self.serialized_content,
        }


@dataclass(frozen=True, order=True)
class ExactRowTextLink:
    row_ordinal: int
    text_ordinal: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "row_ordinal", _strict_int(self.row_ordinal, "link row ordinal")
        )
        object.__setattr__(
            self,
            "text_ordinal",
            _strict_int(self.text_ordinal, "link text ordinal"),
        )


@dataclass(frozen=True)
class AnonymousWorkItem:
    question: str
    rows: tuple[SerializedUnit, ...]
    texts: tuple[SerializedUnit, ...]
    exact_row_text_links: tuple[ExactRowTextLink, ...]

    def __post_init__(self) -> None:
        question = _canonical_text(self.question, "question", maximum=4096)
        rows = tuple(self.rows)
        texts = tuple(self.texts)
        if (
            not rows
            or len(rows) > MAXIMUM_ROWS_BEFORE_SELECTION
            or not all(
                isinstance(row, SerializedUnit) and row.node_type == core.ROW
                for row in rows
            )
        ):
            raise MmqaP1ActionIntegrationError(
                "anonymous rows are empty, oversized, or malformed"
            )
        if (
            not texts
            or len(texts) > MAXIMUM_TEXT_NODES
            or not all(
                isinstance(row, SerializedUnit) and row.node_type == core.TEXT
                for row in texts
            )
        ):
            raise MmqaP1ActionIntegrationError(
                "anonymous texts are empty, oversized, or malformed"
            )
        row_ordinals = tuple(row.ordinal for row in rows)
        text_ordinals = tuple(row.ordinal for row in texts)
        if (
            row_ordinals != tuple(sorted(row_ordinals))
            or text_ordinals != tuple(sorted(text_ordinals))
            or len(set((*row_ordinals, *text_ordinals)))
            != len(row_ordinals) + len(text_ordinals)
        ):
            raise MmqaP1ActionIntegrationError(
                "source-local unit ordinals must be globally distinct and sorted"
            )
        links = tuple(self.exact_row_text_links)
        if (
            not links
            or not all(isinstance(link, ExactRowTextLink) for link in links)
            or links != tuple(sorted(links))
            or len(set(links)) != len(links)
        ):
            raise MmqaP1ActionIntegrationError(
                "exact row-text links must be nonempty, unique, and sorted"
            )
        row_set = set(row_ordinals)
        text_set = set(text_ordinals)
        if any(
            link.row_ordinal not in row_set or link.text_ordinal not in text_set
            for link in links
        ):
            raise MmqaP1ActionIntegrationError(
                "an exact structural link refers outside the anonymous units"
            )
        object.__setattr__(self, "question", question)

    @property
    def units(self) -> tuple[SerializedUnit, ...]:
        return tuple(sorted((*self.rows, *self.texts), key=lambda row: row.ordinal))

    @property
    def anonymous_projection_sha256(self) -> str:
        return _semantic_hash(self.anonymous_payload())

    def anonymous_payload(self) -> dict[str, object]:
        return {
            "schema": ANONYMOUS_WORK_ITEM_SCHEMA,
            "question": self.question,
            "rows": [row.anonymous_payload() for row in self.rows],
            "texts": [row.anonymous_payload() for row in self.texts],
            "exact_row_text_links": [
                {
                    "row_ordinal": link.row_ordinal,
                    "text_ordinal": link.text_ordinal,
                }
                for link in self.exact_row_text_links
            ],
        }


def _serialized_unit_from_mapping(
    value: object, *, node_type: str, label: str
) -> SerializedUnit:
    if not isinstance(value, Mapping):
        raise MmqaP1ActionIntegrationError(f"{label} unit must be a mapping")
    _exact_fields(value, _UNIT_FIELDS, f"{label} unit")
    return SerializedUnit(
        ordinal=value.get("ordinal"),  # type: ignore[arg-type]
        serialized_content=value.get("serialized_content"),  # type: ignore[arg-type]
        node_type=node_type,
    )


def validate_anonymous_work_item(
    value: Mapping[str, object] | AnonymousWorkItem,
) -> AnonymousWorkItem:
    if isinstance(value, AnonymousWorkItem):
        return AnonymousWorkItem(
            value.question,
            tuple(value.rows),
            tuple(value.texts),
            tuple(value.exact_row_text_links),
        )
    if not isinstance(value, Mapping):
        raise MmqaP1ActionIntegrationError(
            "anonymous work item must be a mapping"
        )
    _exact_fields(value, _WORK_ITEM_FIELDS, "anonymous work item")
    if value.get("schema") != ANONYMOUS_WORK_ITEM_SCHEMA:
        raise MmqaP1ActionIntegrationError("anonymous work item schema drifted")
    raw_rows = value.get("rows")
    raw_texts = value.get("texts")
    raw_links = value.get("exact_row_text_links")
    if (
        isinstance(raw_rows, (str, bytes))
        or not isinstance(raw_rows, Sequence)
        or isinstance(raw_texts, (str, bytes))
        or not isinstance(raw_texts, Sequence)
        or isinstance(raw_links, (str, bytes))
        or not isinstance(raw_links, Sequence)
    ):
        raise MmqaP1ActionIntegrationError(
            "anonymous rows, texts, and exact links must be arrays"
        )
    links = []
    for raw in raw_links:
        if not isinstance(raw, Mapping):
            raise MmqaP1ActionIntegrationError(
                "exact structural link must be a mapping"
            )
        _exact_fields(raw, _LINK_FIELDS, "exact structural link")
        links.append(
            ExactRowTextLink(
                raw.get("row_ordinal"),  # type: ignore[arg-type]
                raw.get("text_ordinal"),  # type: ignore[arg-type]
            )
        )
    return AnonymousWorkItem(
        question=value.get("question"),  # type: ignore[arg-type]
        rows=tuple(
            _serialized_unit_from_mapping(row, node_type=core.ROW, label="row")
            for row in raw_rows
        ),
        texts=tuple(
            _serialized_unit_from_mapping(row, node_type=core.TEXT, label="text")
            for row in raw_texts
        ),
        exact_row_text_links=tuple(links),
    )


@dataclass(frozen=True)
class UnitCoordinates:
    """Outer-computed scores and frozen parser flags; never source metadata."""

    ordinal: int
    minilm_similarity: float
    cross_encoder_relevance: float
    entity_anchor: int
    relation_anchor: int
    numeric_or_temporal_anchor: int

    def __post_init__(self) -> None:
        try:
            node = core.ProofNode(
                ordinal=self.ordinal,
                node_type=core.ROW,
                minilm_similarity=self.minilm_similarity,
                cross_encoder_relevance=self.cross_encoder_relevance,
                entity_anchor=self.entity_anchor,
                relation_anchor=self.relation_anchor,
                numeric_or_temporal_anchor=self.numeric_or_temporal_anchor,
            )
        except core.MmqaP1CoreError as exc:
            raise MmqaP1ActionIntegrationError(
                "unit coordinate row drifted"
            ) from exc
        object.__setattr__(self, "ordinal", node.ordinal)
        object.__setattr__(self, "minilm_similarity", node.minilm_similarity)
        object.__setattr__(
            self, "cross_encoder_relevance", node.cross_encoder_relevance
        )
        object.__setattr__(self, "entity_anchor", node.entity_anchor)
        object.__setattr__(self, "relation_anchor", node.relation_anchor)
        object.__setattr__(
            self,
            "numeric_or_temporal_anchor",
            node.numeric_or_temporal_anchor,
        )


def _coordinate_from_mapping(value: object) -> UnitCoordinates:
    if not isinstance(value, Mapping):
        raise MmqaP1ActionIntegrationError("unit coordinate must be a mapping")
    _exact_fields(value, _COORDINATE_FIELDS, "unit coordinate")
    if value.get("schema") != UNIT_COORDINATES_SCHEMA:
        raise MmqaP1ActionIntegrationError("unit coordinate schema drifted")
    return UnitCoordinates(
        ordinal=value.get("ordinal"),  # type: ignore[arg-type]
        minilm_similarity=value.get("minilm_similarity"),  # type: ignore[arg-type]
        cross_encoder_relevance=value.get("cross_encoder_relevance"),  # type: ignore[arg-type]
        entity_anchor=value.get("entity_anchor"),  # type: ignore[arg-type]
        relation_anchor=value.get("relation_anchor"),  # type: ignore[arg-type]
        numeric_or_temporal_anchor=value.get(  # type: ignore[arg-type]
            "numeric_or_temporal_anchor"
        ),
    )


def validate_unit_coordinates(
    work_item: AnonymousWorkItem,
    value: Sequence[UnitCoordinates | Mapping[str, object]],
) -> tuple[UnitCoordinates, ...]:
    if not isinstance(work_item, AnonymousWorkItem):
        raise MmqaP1ActionIntegrationError(
            "coordinate validation requires an anonymous work item"
        )
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MmqaP1ActionIntegrationError("unit coordinates must be an array")
    coordinates = tuple(
        row if isinstance(row, UnitCoordinates) else _coordinate_from_mapping(row)
        for row in value
    )
    coordinate_ordinals = tuple(row.ordinal for row in coordinates)
    expected = tuple(row.ordinal for row in work_item.units)
    if coordinate_ordinals != expected:
        raise MmqaP1ActionIntegrationError(
            "unit coordinates must exactly follow all source-local ordinals"
        )
    return coordinates


@dataclass(frozen=True)
class ActionRanking:
    policy_id: str
    top5_ordinals: tuple[int, ...]
    selected_bundle_ordinals: tuple[int, ...] | None
    selected_bundle_energy: float | None

    def __post_init__(self) -> None:
        if self.policy_id not in {"E0", "E5", "RAW"}:
            raise MmqaP1ActionIntegrationError("action policy ID drifted")
        top5 = tuple(self.top5_ordinals)
        if (
            len(top5) != core.TOP_K
            or len(set(top5)) != core.TOP_K
            or any(type(value) is not int or value < 0 for value in top5)
        ):
            raise MmqaP1ActionIntegrationError(
                "each action ranking must be exactly five unique local ordinals"
            )
        bundle = self.selected_bundle_ordinals
        energy = self.selected_bundle_energy
        if self.policy_id == "RAW":
            if bundle is not None or energy is not None:
                raise MmqaP1ActionIntegrationError(
                    "RAW cannot carry a proof bundle or energy"
                )
        else:
            if (
                bundle is None
                or tuple(sorted(set(bundle))) != bundle
                or not 2 <= len(bundle) <= core.MAX_BUNDLE_SIZE
                or energy is None
                or not math.isfinite(energy)
            ):
                raise MmqaP1ActionIntegrationError(
                    "Agent action bundle or energy drifted"
                )


@dataclass(frozen=True)
class SharedClosureContract:
    units: tuple[SerializedUnit, ...]
    agent_ordinal_bytes: bytes
    raw_ordinal_bytes: bytes
    hipporag_ordinal_bytes: bytes

    def __post_init__(self) -> None:
        ordinals = tuple(unit.ordinal for unit in self.units)
        expected = _canonical_json_bytes(list(ordinals))
        if (
            not self.units
            or self.agent_ordinal_bytes != expected
            or self.raw_ordinal_bytes != expected
            or self.hipporag_ordinal_bytes != expected
        ):
            raise MmqaP1ActionIntegrationError(
                "Agent/RAW/HippoRAG closure ordinals are not byte-identical"
            )

    @property
    def ordinals(self) -> tuple[int, ...]:
        return tuple(unit.ordinal for unit in self.units)

    @property
    def ordinal_bytes_sha256(self) -> str:
        return hashlib.sha256(self.agent_ordinal_bytes).hexdigest()


@dataclass(frozen=True)
class ActionFeatureArchive:
    anonymous_projection_sha256: str
    closure_ordinals: tuple[int, ...]
    closure_ordinal_bytes_sha256: str
    node_feature_rows: tuple[tuple[object, ...], ...]
    directed_edge_rows: tuple[tuple[object, ...], ...]
    bundle_feature_rows: tuple[tuple[object, ...], ...]
    e0_ranking: ActionRanking
    e5_ranking: ActionRanking | None
    raw_ranking: ActionRanking

    def payload(self) -> dict[str, object]:
        closure_hashes = {
            "AGENT": self.closure_ordinal_bytes_sha256,
            "RAW": self.closure_ordinal_bytes_sha256,
            "HIPPORAG": self.closure_ordinal_bytes_sha256,
        }
        return {
            "schema": ACTION_FEATURE_ARCHIVE_SCHEMA,
            "study_id": STUDY_ID,
            "study_design_self_sha256": STUDY_DESIGN_SELF_SHA256,
            "anonymous_projection_sha256": self.anonymous_projection_sha256,
            "closure_ordinals": list(self.closure_ordinals),
            "three_arm_closure_ordinal_bytes_sha256": closure_hashes,
            "three_arm_closure_ordinals_byte_identical": (
                len(set(closure_hashes.values())) == 1
            ),
            "node_feature_order": [
                "ordinal_alignment_only_not_a_feature",
                "node_type",
                "minilm_similarity",
                "cross_encoder_relevance",
                "entity_anchor",
                "relation_anchor",
                "numeric_or_temporal_anchor",
            ],
            "node_feature_rows": [list(row) for row in self.node_feature_rows],
            "directed_edge_rows": [list(row) for row in self.directed_edge_rows],
            "bundle_feature_order": list(core.FEATURE_ORDER),
            "bundle_feature_rows": [
                {
                    "bundle_ordinals": list(row[0]),
                    "features": list(row[1]),
                    "e0_energy": row[2],
                    "e5_energy": row[3],
                }
                for row in self.bundle_feature_rows
            ],
            "E0": _ranking_payload(self.e0_ranking),
            "E5": (
                None if self.e5_ranking is None else _ranking_payload(self.e5_ranking)
            ),
            "RAW": _ranking_payload(self.raw_ranking),
            "gold_or_support_read_count": 0,
            "network_call_count": 0,
            "model_call_count": 0,
            "source_reader_call_count": 0,
            "retry_replay_resample_count": 0,
            "source_metadata_id_feature_count": 0,
            "family_or_question_type_feature_count": 0,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.payload())


def _ranking_payload(value: ActionRanking) -> dict[str, object]:
    return {
        "policy_id": value.policy_id,
        "top5_ordinals": list(value.top5_ordinals),
        "selected_bundle_ordinals": (
            None
            if value.selected_bundle_ordinals is None
            else list(value.selected_bundle_ordinals)
        ),
        "selected_bundle_energy": value.selected_bundle_energy,
    }


@dataclass(frozen=True)
class IntegratedActions:
    work_item: AnonymousWorkItem
    unit_coordinates: tuple[UnitCoordinates, ...]
    shared_closure: SharedClosureContract
    core_closure: core.ProofClosure
    bundles: tuple[core.ProofBundle, ...]
    e0_ranking: ActionRanking
    e5_ranking: ActionRanking | None
    raw_ranking: ActionRanking
    action_feature_archive: ActionFeatureArchive

    def __post_init__(self) -> None:
        closure = tuple(node.ordinal for node in self.core_closure.graph.nodes)
        if closure != self.shared_closure.ordinals:
            raise MmqaP1ActionIntegrationError(
                "integrated core/common closure membership drifted"
            )
        closure_set = set(closure)
        for ranking in (self.e0_ranking, self.raw_ranking, self.e5_ranking):
            if ranking is not None and not set(ranking.top5_ordinals).issubset(
                closure_set
            ):
                raise MmqaP1ActionIntegrationError(
                    "an action ranking escaped the common closure"
                )


def _direct_ce_order(
    graph: core.ProofGraph, *, exclude: frozenset[int] = frozenset()
) -> tuple[int, ...]:
    return tuple(
        node.ordinal
        for node in sorted(
            (node for node in graph.nodes if node.ordinal not in exclude),
            key=lambda node: (-node.cross_encoder_relevance, node.ordinal),
        )
    )


def _bundle_first_top5(
    graph: core.ProofGraph, bundle: core.ProofBundle
) -> tuple[int, ...]:
    first = core.rank_bundle_evidence(graph, bundle)
    remainder = _direct_ce_order(graph, exclude=frozenset(first))
    ranking = tuple((*first, *remainder)[: core.TOP_K])
    if len(ranking) != core.TOP_K:
        raise MmqaP1ActionIntegrationError(
            "closure cannot project a complete top-five action"
        )
    return ranking


def _core_edges(links: Sequence[ExactRowTextLink]) -> tuple[core.TypedLinkEdge, ...]:
    edges = []
    for link in links:
        edges.append(
            core.TypedLinkEdge(
                link.row_ordinal, link.text_ordinal, core.ROW_TO_TEXT
            )
        )
        edges.append(
            core.TypedLinkEdge(
                link.text_ordinal, link.row_ordinal, core.TEXT_TO_ROW
            )
        )
    return tuple(
        sorted(
            edges,
            key=lambda edge: (
                edge.source_ordinal,
                edge.target_ordinal,
                core.EDGE_TYPES.index(edge.edge_type),
            ),
        )
    )


def form_actions(
    work_item: Mapping[str, object] | AnonymousWorkItem,
    unit_coordinates: Sequence[UnitCoordinates | Mapping[str, object]],
    *,
    e5_model: core.E5Model | None = None,
) -> IntegratedActions:
    """Form all label-free actions; there is intentionally no gold argument."""

    item = validate_anonymous_work_item(work_item)
    coordinates = validate_unit_coordinates(item, unit_coordinates)
    by_coordinate = {row.ordinal: row for row in coordinates}

    retained_rows = tuple(
        sorted(
            item.rows,
            key=lambda unit: (
                -by_coordinate[unit.ordinal].cross_encoder_relevance,
                -by_coordinate[unit.ordinal].minilm_similarity,
                unit.ordinal,
            ),
        )[:MAXIMUM_ROW_NODES]
    )
    retained_units = tuple(
        sorted((*retained_rows, *item.texts), key=lambda unit: unit.ordinal)
    )
    if not MINIMUM_CLOSURE_NODES <= len(retained_units) <= core.MAX_CLOSURE_NODES:
        raise MmqaP1ActionIntegrationError(
            "selected common closure must contain five through 96 units"
        )
    retained_ordinals = frozenset(unit.ordinal for unit in retained_units)
    retained_links = tuple(
        link
        for link in item.exact_row_text_links
        if link.row_ordinal in retained_ordinals
        and link.text_ordinal in retained_ordinals
    )
    if not retained_links:
        raise MmqaP1ActionIntegrationError(
            "query-ranked row cap removed every exact row-text link"
        )

    nodes = tuple(
        core.ProofNode(
            ordinal=unit.ordinal,
            node_type=unit.node_type,
            minilm_similarity=by_coordinate[unit.ordinal].minilm_similarity,
            cross_encoder_relevance=by_coordinate[
                unit.ordinal
            ].cross_encoder_relevance,
            entity_anchor=by_coordinate[unit.ordinal].entity_anchor,
            relation_anchor=by_coordinate[unit.ordinal].relation_anchor,
            numeric_or_temporal_anchor=by_coordinate[
                unit.ordinal
            ].numeric_or_temporal_anchor,
        )
        for unit in retained_units
    )
    graph = core.ProofGraph(nodes, _core_edges(retained_links))
    # The source item is already query-local.  Every retained text and each
    # frozen top-ranked row is an explicit closure seed; exact links remain the
    # sole authority for bundle connectivity, never for hidden arm-specific
    # candidate expansion.
    closure = core.build_query_local_closure(
        graph,
        tuple(unit.ordinal for unit in retained_units),
        hop_limit=core.MAX_CLOSURE_HOPS,
        max_nodes=core.MAX_CLOSURE_NODES,
    )
    closure_ordinals = tuple(node.ordinal for node in closure.graph.nodes)
    if closure_ordinals != tuple(unit.ordinal for unit in retained_units):
        raise MmqaP1ActionIntegrationError(
            "the common retained-unit closure unexpectedly changed membership"
        )
    bundles = core.enumerate_connected_bundles(closure)
    if not bundles:
        raise MmqaP1ActionIntegrationError(
            "no connected row-text proof bundle was generated"
        )

    e0_bundle = core.select_e0_bundle(closure.graph, bundles)
    e0_ranking = ActionRanking(
        "E0",
        _bundle_first_top5(closure.graph, e0_bundle),
        e0_bundle.node_ordinals,
        core.e0_proof_energy(closure.graph, e0_bundle),
    )
    raw_ranking = ActionRanking(
        "RAW", _direct_ce_order(closure.graph)[: core.TOP_K], None, None
    )
    if e5_model is None:
        e5_ranking = None
    else:
        if not isinstance(e5_model, core.E5Model):
            raise MmqaP1ActionIntegrationError(
                "optional E5 model must be the frozen core type"
            )
        e5_bundle = core.select_e5_bundle(e5_model, closure.graph, bundles)
        e5_ranking = ActionRanking(
            "E5",
            _bundle_first_top5(closure.graph, e5_bundle),
            e5_bundle.node_ordinals,
            e5_model.energy(core.bundle_feature_vector(closure.graph, e5_bundle)),
        )

    ordinal_bytes = _canonical_json_bytes(list(closure_ordinals))
    shared = SharedClosureContract(
        retained_units, ordinal_bytes, ordinal_bytes, ordinal_bytes
    )
    bundle_rows = tuple(
        (
            bundle.node_ordinals,
            core.bundle_feature_vector(closure.graph, bundle),
            core.e0_proof_energy(closure.graph, bundle),
            (
                None
                if e5_model is None
                else e5_model.energy(
                    core.bundle_feature_vector(closure.graph, bundle)
                )
            ),
        )
        for bundle in bundles
    )
    archive = ActionFeatureArchive(
        anonymous_projection_sha256=item.anonymous_projection_sha256,
        closure_ordinals=closure_ordinals,
        closure_ordinal_bytes_sha256=shared.ordinal_bytes_sha256,
        node_feature_rows=tuple(
            (
                node.ordinal,
                node.node_type,
                node.minilm_similarity,
                node.cross_encoder_relevance,
                node.entity_anchor,
                node.relation_anchor,
                node.numeric_or_temporal_anchor,
            )
            for node in closure.graph.nodes
        ),
        directed_edge_rows=tuple(
            (edge.source_ordinal, edge.target_ordinal, edge.edge_type)
            for edge in closure.graph.edges
        ),
        bundle_feature_rows=bundle_rows,
        e0_ranking=e0_ranking,
        e5_ranking=e5_ranking,
        raw_ranking=raw_ranking,
    )
    return IntegratedActions(
        work_item=item,
        unit_coordinates=coordinates,
        shared_closure=shared,
        core_closure=closure,
        bundles=bundles,
        e0_ranking=e0_ranking,
        e5_ranking=e5_ranking,
        raw_ranking=raw_ranking,
        action_feature_archive=archive,
    )


@dataclass(frozen=True)
class OfflineRankingScore:
    ndcg_at_5: float
    integer_utility: int
    recall_at_5: float
    connected_gold_row_text_pair_recovered: bool


@dataclass(frozen=True)
class LateGoldScores:
    e0: OfflineRankingScore
    e5: OfflineRankingScore | None
    raw: OfflineRankingScore
    hipporag: OfflineRankingScore | None


def _validate_complete_top5(
    value: Sequence[int], closure_ordinals: frozenset[int], label: str
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MmqaP1ActionIntegrationError(f"{label} ranking must be an array")
    checked = tuple(_strict_int(row, f"{label} ordinal") for row in value)
    if (
        len(checked) != core.TOP_K
        or len(set(checked)) != core.TOP_K
        or not set(checked).issubset(closure_ordinals)
    ):
        raise MmqaP1ActionIntegrationError(
            f"{label} must be five unique common-closure ordinals"
        )
    return checked


def _score_one_ranking(
    actions: IntegratedActions,
    ranking: Sequence[int],
    gold: tuple[int, ...],
    exact_gold_pairs: tuple[ExactRowTextLink, ...],
) -> OfflineRankingScore:
    closure_set = frozenset(actions.shared_closure.ordinals)
    checked = _validate_complete_top5(ranking, closure_set, "late-score")
    ndcg = core.binary_evidence_ndcg_at_5(checked, gold)
    gold_set = frozenset(gold)
    hits = len(set(checked).intersection(gold_set))
    selected = frozenset(checked)
    recovered = any(
        pair.row_ordinal in selected and pair.text_ordinal in selected
        for pair in exact_gold_pairs
    )
    return OfflineRankingScore(
        ndcg_at_5=ndcg,
        integer_utility=core.integer_utility_from_ndcg(ndcg),
        recall_at_5=hits / len(gold),
        connected_gold_row_text_pair_recovered=recovered,
    )


def _validate_exact_gold_pairs(
    actions: IntegratedActions,
    gold: tuple[int, ...],
    value: Sequence[ExactRowTextLink | Mapping[str, object]] | None,
) -> tuple[ExactRowTextLink, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise MmqaP1ActionIntegrationError(
            "late exact gold pairs must be a nonempty array"
        )
    pairs = []
    for raw in value:
        if isinstance(raw, ExactRowTextLink):
            pair = raw
        else:
            if not isinstance(raw, Mapping):
                raise MmqaP1ActionIntegrationError(
                    "late exact gold pair must be a mapping"
                )
            _exact_fields(raw, _LINK_FIELDS, "late exact gold pair")
            pair = ExactRowTextLink(
                raw.get("row_ordinal"),  # type: ignore[arg-type]
                raw.get("text_ordinal"),  # type: ignore[arg-type]
            )
        pairs.append(pair)
    checked = tuple(pairs)
    if len(set(checked)) != len(checked):
        raise MmqaP1ActionIntegrationError(
            "late exact gold pairs must be unique"
        )
    gold_set = frozenset(gold)
    row_set = frozenset(row.ordinal for row in actions.work_item.rows)
    text_set = frozenset(text.ordinal for text in actions.work_item.texts)
    structural_links = frozenset(actions.work_item.exact_row_text_links)
    if any(
        pair.row_ordinal not in row_set
        or pair.text_ordinal not in text_set
        or pair.row_ordinal not in gold_set
        or pair.text_ordinal not in gold_set
        or pair not in structural_links
        for pair in checked
    ):
        raise MmqaP1ActionIntegrationError(
            "late exact gold pair escaped the original item, gold set, "
            "or exact structural links"
        )
    return checked


def score_late_gold(
    actions: IntegratedActions,
    gold_evidence_ordinals: Sequence[int],
    *,
    exact_gold_pairs: (
        Sequence[ExactRowTextLink | Mapping[str, object]] | None
    ) = None,
    hipporag_top5_ordinals: Sequence[int] | None = None,
) -> LateGoldScores:
    """Score sealed rankings after gold release; never reform an action."""

    if not isinstance(actions, IntegratedActions):
        raise MmqaP1ActionIntegrationError(
            "late scoring requires completed integrated actions"
        )
    if isinstance(gold_evidence_ordinals, (str, bytes)) or not isinstance(
        gold_evidence_ordinals, Sequence
    ):
        raise MmqaP1ActionIntegrationError("late gold must be an ordinal array")
    gold = tuple(
        _strict_int(row, "late gold evidence ordinal")
        for row in gold_evidence_ordinals
    )
    item_set = frozenset(unit.ordinal for unit in actions.work_item.units)
    if not gold or len(set(gold)) != len(gold) or not set(gold).issubset(item_set):
        raise MmqaP1ActionIntegrationError(
            "late gold must be nonempty, unique, and inside the original "
            "anonymous item universe"
        )
    pairs = _validate_exact_gold_pairs(actions, gold, exact_gold_pairs)
    e0 = _score_one_ranking(
        actions, actions.e0_ranking.top5_ordinals, gold, pairs
    )
    raw = _score_one_ranking(
        actions, actions.raw_ranking.top5_ordinals, gold, pairs
    )
    e5 = (
        None
        if actions.e5_ranking is None
        else _score_one_ranking(
            actions, actions.e5_ranking.top5_ordinals, gold, pairs
        )
    )
    hipporag = (
        None
        if hipporag_top5_ordinals is None
        else _score_one_ranking(actions, hipporag_top5_ordinals, gold, pairs)
    )
    return LateGoldScores(e0=e0, e5=e5, raw=raw, hipporag=hipporag)


__all__ = [
    "VERSION",
    "STUDY_ID",
    "STUDY_DESIGN_SELF_SHA256",
    "ANONYMOUS_WORK_ITEM_SCHEMA",
    "UNIT_COORDINATES_SCHEMA",
    "ACTION_FEATURE_ARCHIVE_SCHEMA",
    "MAXIMUM_ROWS_BEFORE_SELECTION",
    "MAXIMUM_ROW_NODES",
    "MAXIMUM_TEXT_NODES",
    "MINIMUM_CLOSURE_NODES",
    "FORBIDDEN_ACTION_INPUT_FIELDS",
    "MmqaP1ActionIntegrationError",
    "SerializedUnit",
    "ExactRowTextLink",
    "AnonymousWorkItem",
    "UnitCoordinates",
    "ActionRanking",
    "SharedClosureContract",
    "ActionFeatureArchive",
    "IntegratedActions",
    "OfflineRankingScore",
    "LateGoldScores",
    "validate_anonymous_work_item",
    "validate_unit_coordinates",
    "form_actions",
    "score_late_gold",
]
