"""Outcome-disciplined execution core for the frozen FEVEROUS P6/E2 study.

This module owns no source opener and no filesystem capability.  It converts
the already-acquired, label-free 8,192-unit view into the exact semantic
sidecars, executes every local recipe for a block with eager bounded
parallelism, and performs the late exact-utility calculations.  Gold labels
are accepted only by the explicitly late scoring functions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from types import MappingProxyType
from typing import Any

from assumption_agent.benchmarks import feverous_e2_evaluator_v1 as evaluator
from assumption_agent.benchmarks import feverous_e2_feature_producer_v1 as features
from assumption_agent.benchmarks import feverous_offline_semantic_tensor_v1 as semantic
from assumption_agent.benchmarks import feverous_p6_e2_acquisition_v1 as acquisition
from assumption_agent.benchmarks import feverous_p6_query_anchored_operator_v1 as operator
from assumption_agent.benchmarks.feverous_wikipedia_source_qualification_v1 import (
    FeverousWikipediaQualificationError,
    parse_element_id,
)


VERSION = "feverous_p6_e2_formal_runner_v1"
LOCAL_ITEM_WORKERS = 64
TOP_K = 5
BLOCK_COUNTS = dict(acquisition.BLOCK_COUNTS)
RECIPE_IDS = evaluator.RECIPE_IDS

_CORPUS_VIEW_KEYS = frozenset(
    {
        "schema",
        "version",
        "unit_count",
        "gold_origin_or_membership_included",
        "units",
        "corpus_view_sha256",
    }
)
_BLOCK_VIEW_KEYS = frozenset(
    {
        "schema",
        "version",
        "item_count",
        "late_label_fields_included",
        "items",
        "block_view_sha256",
    }
)
_BLOCK_LABEL_KEYS = frozenset(
    {
        "schema",
        "version",
        "block",
        "item_count",
        "items",
        "block_labels_sha256",
    }
)
_SIDECAR_KEYS = frozenset(
    {
        "linearizer_version",
        "page",
        "local_id",
        "unit_type",
        "coordinates",
        "section_ids",
        "section_path",
        "official_ordinal",
        "previous_atomic_local_id",
        "next_atomic_local_id",
        "table_id",
        "table_kind",
        "table_caption",
        "row_span",
        "column_span",
        "applicable_row_header_ids",
        "applicable_column_header_ids",
        "list_id",
        "list_ancestor_ids",
    }
)


class FeverousFormalRunnerError(RuntimeError):
    """A label-free execution or exact late-scoring invariant drifted."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise FeverousFormalRunnerError("value is not canonical JSON") from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FeverousFormalRunnerError(f"{field} is not a lowercase SHA-256")
    return value


def _fraction_payload(value: Fraction) -> list[int]:
    if not isinstance(value, Fraction):
        raise FeverousFormalRunnerError("exact utility is not a Fraction")
    return [value.numerator, value.denominator]


def _opaque_component(kind: str, *values: str) -> str:
    return stable_hash([kind, *values])


def _sidecar_tuple(sidecar: Mapping[str, object], field: str) -> tuple[object, ...]:
    value = sidecar.get(field, ())
    if not isinstance(value, (list, tuple)):
        raise FeverousFormalRunnerError(f"corpus sidecar {field} is not a sequence")
    return tuple(value)


def _safe_optional_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or "\x00" in value:
        raise FeverousFormalRunnerError(f"corpus sidecar {field} is malformed")
    return value


def _parsed_local_identity(
    *, page: str, local_id: str, expected_kind: str
) -> object:
    try:
        parsed = parse_element_id(f"{page}_{local_id}")
    except FeverousWikipediaQualificationError as exc:
        raise FeverousFormalRunnerError(
            "corpus sidecar local identity is noncanonical"
        ) from exc
    if (
        parsed.page != page
        or parsed.local_id != local_id
        or parsed.kind != expected_kind
    ):
        raise FeverousFormalRunnerError(
            "corpus sidecar local identity disagrees with its type"
        )
    return parsed


def _validate_atomic_sidecar(
    *, sidecar: Mapping[str, object], unit_type: str
) -> tuple[str, str, int]:
    """Validate the exact full-compiler sidecar before deriving opaque keys."""

    if set(sidecar) != _SIDECAR_KEYS:
        raise FeverousFormalRunnerError("corpus sidecar schema drifted")
    if sidecar.get("linearizer_version") != acquisition.ATOMIC_LINEARIZER_VERSION:
        raise FeverousFormalRunnerError("corpus sidecar linearizer drifted")
    page = sidecar.get("page")
    local_id = sidecar.get("local_id")
    if (
        not isinstance(page, str)
        or not page
        or "\x00" in page
        or not isinstance(local_id, str)
        or not local_id
        or "\x00" in local_id
        or sidecar.get("unit_type") != unit_type
    ):
        raise FeverousFormalRunnerError("corpus identity sidecar is malformed")
    parsed = _parsed_local_identity(
        page=page, local_id=local_id, expected_kind=unit_type
    )

    official_order = sidecar.get("official_ordinal")
    if (
        isinstance(official_order, bool)
        or not isinstance(official_order, int)
        or official_order < 0
    ):
        raise FeverousFormalRunnerError("official corpus order is malformed")
    coordinates = _sidecar_tuple(sidecar, "coordinates")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in coordinates
    ):
        raise FeverousFormalRunnerError("atomic coordinates are malformed")
    expected_arity = {
        "sentence": 1,
        "item": 2,
        "cell": 3,
        "header_cell": 3,
        "table_caption": 1,
    }.get(unit_type)
    if expected_arity is None or len(coordinates) != expected_arity:
        raise FeverousFormalRunnerError("atomic coordinate arity drifted")
    parsed_indices = tuple(parsed.indices)  # type: ignore[attr-defined]
    if unit_type in {"cell", "header_cell"}:
        # Span normalization may move the physical row/column, but never the
        # top-level table identity encoded by the first coordinate.
        if not parsed_indices or coordinates[0] != parsed_indices[0]:
            raise FeverousFormalRunnerError("cell table coordinate drifted")
    elif tuple(coordinates) != parsed_indices:
        raise FeverousFormalRunnerError("atomic coordinates disagree with identity")

    section_ids = _sidecar_tuple(sidecar, "section_ids")
    section_path = _sidecar_tuple(sidecar, "section_path")
    if (
        len(section_ids) != len(section_path)
        or any(
            not isinstance(value, str)
            or not value
            or "\x00" in value
            for value in (*section_ids, *section_path)
        )
    ):
        raise FeverousFormalRunnerError("section topology is malformed")
    for field in ("previous_atomic_local_id", "next_atomic_local_id"):
        adjacent = _safe_optional_text(sidecar.get(field), field)
        if adjacent is not None:
            if not adjacent or adjacent == local_id:
                raise FeverousFormalRunnerError("atomic adjacency is malformed")
            try:
                adjacent_parsed = parse_element_id(f"{page}_{adjacent}")
            except FeverousWikipediaQualificationError as exc:
                raise FeverousFormalRunnerError(
                    "atomic adjacency identity is noncanonical"
                ) from exc
            if adjacent_parsed.page != page or adjacent_parsed.local_id != adjacent:
                raise FeverousFormalRunnerError(
                    "atomic adjacency identity leaves its page"
                )

    table_id = _safe_optional_text(sidecar.get("table_id"), "table_id")
    table_kind = _safe_optional_text(sidecar.get("table_kind"), "table_kind")
    table_caption = _safe_optional_text(
        sidecar.get("table_caption"), "table_caption"
    )
    row_span = sidecar.get("row_span")
    column_span = sidecar.get("column_span")
    row_headers = _sidecar_tuple(sidecar, "applicable_row_header_ids")
    column_headers = _sidecar_tuple(sidecar, "applicable_column_header_ids")
    header_ids = (*row_headers, *column_headers)
    if any(not isinstance(value, str) or not value for value in header_ids):
        raise FeverousFormalRunnerError("applicable header identity is malformed")
    if len(set(header_ids)) != len(header_ids):
        raise FeverousFormalRunnerError("applicable header identity is duplicated")

    if unit_type in {"cell", "header_cell", "table_caption"}:
        expected_table = f"table_{coordinates[0]}"
        if table_id != expected_table or table_kind is None:
            raise FeverousFormalRunnerError("table identity sidecar drifted")
        if unit_type in {"cell", "header_cell"}:
            if (
                type(row_span) is not int
                or row_span < 1
                or type(column_span) is not int
                or column_span < 1
            ):
                raise FeverousFormalRunnerError("cell span sidecar is malformed")
        elif row_span is not None or column_span is not None or header_ids:
            raise FeverousFormalRunnerError("table caption carries cell topology")
        for header_id in header_ids:
            header = _parsed_local_identity(
                page=page,
                local_id=str(header_id),
                expected_kind="header_cell",
            )
            if not header.indices or header.indices[0] != coordinates[0]:
                raise FeverousFormalRunnerError(
                    "applicable header leaves the exact table"
                )
    elif any(
        value is not None
        for value in (table_id, table_kind, table_caption, row_span, column_span)
    ) or header_ids:
        raise FeverousFormalRunnerError("non-table unit carries table topology")

    list_id = _safe_optional_text(sidecar.get("list_id"), "list_id")
    ancestors = _sidecar_tuple(sidecar, "list_ancestor_ids")
    if any(not isinstance(value, str) or not value for value in ancestors):
        raise FeverousFormalRunnerError("list ancestor path is malformed")
    if len(set(ancestors)) != len(ancestors):
        raise FeverousFormalRunnerError("list ancestor path is duplicated")
    if unit_type == "item":
        expected_list = f"list_{coordinates[0]}"
        if list_id != expected_list:
            raise FeverousFormalRunnerError("list identity sidecar drifted")
        for ancestor_id in ancestors:
            ancestor = _parsed_local_identity(
                page=page,
                local_id=str(ancestor_id),
                expected_kind="item",
            )
            if (
                ancestor.indices[0] != coordinates[0]
                or ancestor.indices[1] >= parsed_indices[1]
            ):
                raise FeverousFormalRunnerError(
                    "list ancestor leaves its exact prior list path"
                )
    elif list_id is not None or ancestors:
        raise FeverousFormalRunnerError("non-item unit carries list topology")
    return page, local_id, official_order


def corpus_view_to_semantic_units(
    corpus_view: Mapping[str, Any],
) -> tuple[semantic.SemanticCorpusUnit, ...]:
    """Convert the exact acquired view without introducing text or graph drift."""

    if not isinstance(corpus_view, Mapping):
        raise FeverousFormalRunnerError("corpus view is not an object")
    try:
        acquisition.verify_self_hash(corpus_view, "corpus_view_sha256")
    except acquisition.FeverousP6E2AcquisitionError as exc:
        raise FeverousFormalRunnerError("corpus view self-hash drifted") from exc
    rows = corpus_view.get("units")
    if (
        set(corpus_view) != _CORPUS_VIEW_KEYS
        or
        corpus_view.get("schema") != acquisition.CORPUS_VIEW_SCHEMA
        or corpus_view.get("version") != acquisition.VERSION
        or corpus_view.get("unit_count") != acquisition.CORPUS_UNIT_COUNT
        or corpus_view.get("gold_origin_or_membership_included") is not False
        or not isinstance(rows, list)
        or len(rows) != acquisition.CORPUS_UNIT_COUNT
    ):
        raise FeverousFormalRunnerError("corpus view shape drifted")

    key_to_ordinal: dict[tuple[str, str], int] = {}
    normalized_rows: list[tuple[int, str, str, dict[str, object]]] = []
    page_orders: set[tuple[str, int]] = set()
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "unit_i",
            "text",
            "unit_type",
            "sidecar",
        }:
            raise FeverousFormalRunnerError("corpus row schema drifted")
        if row.get("unit_i") != ordinal:
            raise FeverousFormalRunnerError("corpus row ordinal drifted")
        text = row.get("text")
        unit_type = row.get("unit_type")
        sidecar_value = row.get("sidecar")
        if (
            not isinstance(text, str)
            or not text
            or not isinstance(unit_type, str)
            or not isinstance(sidecar_value, Mapping)
        ):
            raise FeverousFormalRunnerError("corpus row content is malformed")
        sidecar = dict(sidecar_value)
        page, local_id, official_order = _validate_atomic_sidecar(
            sidecar=sidecar, unit_type=unit_type
        )
        unit_key = (page, local_id)
        if unit_key in key_to_ordinal:
            raise FeverousFormalRunnerError("corpus contains a duplicate atomic identity")
        if (page, official_order) in page_orders:
            raise FeverousFormalRunnerError(
                "official corpus order is duplicated within a page"
            )
        key_to_ordinal[unit_key] = ordinal
        page_orders.add((page, official_order))
        normalized_rows.append((ordinal, text, unit_type, sidecar))

    output: list[semantic.SemanticCorpusUnit] = []
    for ordinal, text, unit_type, sidecar in normalized_rows:
        page = str(sidecar["page"])
        official_order = int(sidecar["official_ordinal"])
        section_path_raw = _sidecar_tuple(sidecar, "section_path")
        if any(not isinstance(value, str) for value in section_path_raw):
            raise FeverousFormalRunnerError("section path contains a non-string")
        section_path = tuple(str(value) for value in section_path_raw)

        table_id = sidecar.get("table_id")
        table_key = None
        table_row = None
        if table_id is not None:
            if not isinstance(table_id, str) or not table_id:
                raise FeverousFormalRunnerError("table identity is malformed")
            table_key = _opaque_component("table", page, table_id)
        if unit_type in {"cell", "header_cell"}:
            coordinates = _sidecar_tuple(sidecar, "coordinates")
            if (
                len(coordinates) != 3
                or any(isinstance(value, bool) or not isinstance(value, int) for value in coordinates)
                or int(coordinates[1]) < 0
            ):
                raise FeverousFormalRunnerError("cell coordinates are malformed")
            table_row = int(coordinates[1])

        applicable_headers: tuple[int, ...] = ()
        if unit_type == "cell":
            header_ids = (
                *_sidecar_tuple(sidecar, "applicable_row_header_ids"),
                *_sidecar_tuple(sidecar, "applicable_column_header_ids"),
            )
            if any(not isinstance(value, str) or not value for value in header_ids):
                raise FeverousFormalRunnerError("applicable header identity is malformed")
            applicable_headers = tuple(
                sorted(
                    {
                        key_to_ordinal[(page, str(header_id))]
                        for header_id in header_ids
                        if (page, str(header_id)) in key_to_ordinal
                    }
                )
            )
            for header_ordinal in applicable_headers:
                _header_index, _header_text, header_type, header_sidecar = (
                    normalized_rows[header_ordinal]
                )
                if (
                    header_type != "header_cell"
                    or header_sidecar.get("table_id") != table_id
                ):
                    raise FeverousFormalRunnerError(
                        "applicable header does not map to a same-table header_cell"
                    )

        list_parent_path: tuple[str, ...] = ()
        if unit_type == "item":
            list_id = sidecar.get("list_id")
            if not isinstance(list_id, str) or not list_id:
                raise FeverousFormalRunnerError("list identity is malformed")
            ancestors = _sidecar_tuple(sidecar, "list_ancestor_ids")
            if any(not isinstance(value, str) or not value for value in ancestors):
                raise FeverousFormalRunnerError("list ancestor path is malformed")
            # The list root is an explicit first component, so top-level
            # siblings share one real parent path instead of all mapping to ().
            list_parent_path = (
                _opaque_component("list", page, list_id),
                *(
                    _opaque_component("list_ancestor", page, list_id, str(value))
                    for value in ancestors
                ),
            )

        try:
            output.append(
                semantic.SemanticCorpusUnit(
                    corpus_ordinal=ordinal,
                    linearized_text=text,
                    unit_type=unit_type,
                    page_key=page,
                    official_order=official_order,
                    section_path=section_path,
                    table_key=table_key,
                    table_row=table_row,
                    applicable_header_ordinals=applicable_headers,
                    list_parent_path=list_parent_path,
                )
            )
        except (semantic.FeverousSemanticTensorError, operator.FeverousP6OperatorError) as exc:
            raise FeverousFormalRunnerError("semantic corpus conversion failed") from exc
    return tuple(output)


def item_commitment(*, block: str, ordinal: int, claim: str) -> str:
    if block not in BLOCK_COUNTS or type(ordinal) is not int or ordinal < 0:
        raise FeverousFormalRunnerError("item identity is invalid")
    if not isinstance(claim, str) or not claim or "\x00" in claim:
        raise FeverousFormalRunnerError("claim is invalid")
    return stable_hash(
        {
            "block": block,
            "claim_sha256": hashlib.sha256(claim.encode("utf-8")).hexdigest(),
            "ordinal": ordinal,
            "schema": f"{VERSION}_item_commitment",
        }
    )


def claims_from_block_view(
    block_view: Mapping[str, Any], *, block: str
) -> tuple[str, ...]:
    """Open only the frozen claim-only view for one already-authorized block."""

    if block not in BLOCK_COUNTS or not isinstance(block_view, Mapping):
        raise FeverousFormalRunnerError("block view identity is invalid")
    try:
        acquisition.verify_self_hash(block_view, "block_view_sha256")
    except acquisition.FeverousP6E2AcquisitionError as exc:
        raise FeverousFormalRunnerError("block view self-hash drifted") from exc
    rows = block_view.get("items")
    if (
        set(block_view) != _BLOCK_VIEW_KEYS
        or
        block_view.get("schema") != acquisition.BLOCK_VIEW_SCHEMA
        or block_view.get("version") != acquisition.VERSION
        or block_view.get("item_count") != BLOCK_COUNTS[block]
        or block_view.get("late_label_fields_included") is not False
        or not isinstance(rows, list)
        or len(rows) != BLOCK_COUNTS[block]
    ):
        raise FeverousFormalRunnerError("block view shape drifted")
    claims: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"claim"}:
            raise FeverousFormalRunnerError("block view contains a forbidden field")
        claim = row.get("claim")
        if not isinstance(claim, str) or not claim or "\x00" in claim:
            raise FeverousFormalRunnerError("block view claim is malformed")
        claims.append(claim)
    return tuple(claims)


def action_trace_content_free_receipt(
    trace: operator.ActionTrace,
) -> dict[str, object]:
    """Serialize the complete auditable action seal without corpus/query text."""

    try:
        operator.verify_action_trace(trace)
    except operator.FeverousP6OperatorError as exc:
        raise FeverousFormalRunnerError("action trace self-seal drifted") from exc
    body: dict[str, object] = {
        "candidate_scan_sha256": trace.candidate_scan_sha256,
        "candidate_score_evaluations": trace.candidate_score_evaluations,
        "candidate_universe_size": trace.candidate_universe_size,
        "graph_sha256": trace.graph_sha256,
        "hipporag_candidate_or_feature_count": (
            trace.hipporag_candidate_or_feature_count
        ),
        "output_top5": list(trace.output_top5),
        "query_sha256": trace.query_sha256,
        "raw_dense_order_sha256": trace.raw_dense_order_sha256,
        "reachability_sha256": trace.reachability_sha256,
        "recipe_id": trace.recipe_id,
        "retained_raw_top3": list(trace.retained_raw_top3),
        "selection_steps": [
            [
                step.output_slot,
                step.selected_unit_ordinal,
                step.disposition,
                step.residual_facet_coverage_gain_int,
                step.direct_anchor,
                step.path_length,
                step.path_strength_int,
            ]
            for step in trace.selection_steps
        ],
        "semantic_cell_scan_count": trace.semantic_cell_scan_count,
        "semantic_tensor_sha256": trace.semantic_tensor_sha256,
        "version": operator.VERSION,
    }
    if operator.stable_hash(body) != trace.trace_sha256:
        raise FeverousFormalRunnerError("action trace serialization drifted")
    return {**body, "trace_sha256": trace.trace_sha256}


@dataclass(frozen=True)
class ItemExecution:
    block: str
    ordinal: int
    item_commitment_sha256: str
    semantic_build: semantic.SemanticTensorBuild
    action_traces: tuple[operator.ActionTrace, ...]
    feature_traces: tuple[features.FeatureProductionTrace, ...]
    operator_receipt_sha256: str

    @property
    def recipe_traces(self) -> tuple[evaluator.RecipeTrace, ...]:
        return tuple(trace.recipe_trace for trace in self.feature_traces)

    @property
    def outputs(self) -> Mapping[str, tuple[int, ...]]:
        return MappingProxyType(
            {trace.recipe_id: tuple(trace.output_top5) for trace in self.action_traces}
        )

    def public_payload(self) -> dict[str, object]:
        action_receipts = [
            action_trace_content_free_receipt(trace)
            for trace in self.action_traces
        ]
        return {
            "block": self.block,
            "ordinal": self.ordinal,
            "item_commitment_sha256": self.item_commitment_sha256,
            "semantic_receipt": dict(self.semantic_build.receipt),
            "operator_receipt_sha256": self.operator_receipt_sha256,
            "action_trace_sha256s": {
                trace.recipe_id: trace.trace_sha256 for trace in self.action_traces
            },
            "ordered_top5": {
                trace.recipe_id: list(trace.output_top5) for trace in self.action_traces
            },
            "complete_action_trace_receipts": action_receipts,
            "feature_traces": [
                {
                    **trace.payload_body(),
                    "production_trace_sha256": trace.production_trace_sha256,
                }
                for trace in self.feature_traces
            ],
        }


@dataclass(frozen=True)
class BlockExecution:
    block: str
    items: tuple[ItemExecution, ...]
    feature_receipt: Mapping[str, Any]
    receipt: Mapping[str, Any]

    @property
    def recipe_traces(self) -> tuple[evaluator.RecipeTrace, ...]:
        return tuple(trace for item in self.items for trace in item.recipe_traces)


@dataclass(frozen=True)
class FormationExecution:
    """The inseparable A_form/F_search local execution barrier."""

    A_form: BlockExecution
    F_search: BlockExecution
    receipt: Mapping[str, Any]


def execute_local_item(
    *,
    block: str,
    ordinal: int,
    claim: str,
    prepared_corpus: semantic.PreparedSemanticCorpus,
    minilm_backend: semantic.MiniLMBackend,
    ner_backend: semantic.NERBackend,
    nli_backend: semantic.NLIBackend,
) -> ItemExecution:
    """Execute one claim without labels, family, gold, RAW, or Hippo inputs."""

    commitment = item_commitment(block=block, ordinal=ordinal, claim=claim)
    build = semantic.build_prepared_offline_semantic_tensor(
        claim_text=claim,
        prepared_corpus=prepared_corpus,
        minilm_backend=minilm_backend,
        ner_backend=ner_backend,
        nli_backend=nli_backend,
    )
    action_traces = operator.run_all_recipes(
        graph=build.graph,
        semantic_tensor=build.tensor,
    )
    operator_receipt = stable_hash(
        {
            "graph_sha256": build.graph.graph_sha256,
            "item_commitment_sha256": commitment,
            "query_sha256": build.tensor.query_sha256,
            "recipe_trace_sha256s": [trace.trace_sha256 for trace in action_traces],
            "schema": f"{VERSION}_operator_matrix_receipt",
        }
    )
    semantic_receipt = _require_sha256(
        build.receipt.get("semantic_receipt_sha256"), "semantic receipt"
    )
    produced = features.produce_complete_e2_recipe_matrix(
        item_commitment_sha256=commitment,
        graph=build.graph,
        semantic_tensor=build.tensor,
        action_traces=action_traces,
        external_operator_receipt_sha256=operator_receipt,
        external_semantic_receipt_sha256=semantic_receipt,
    )
    return ItemExecution(
        block=block,
        ordinal=ordinal,
        item_commitment_sha256=commitment,
        semantic_build=build,
        action_traces=action_traces,
        feature_traces=produced,
        operator_receipt_sha256=operator_receipt,
    )


def _validate_local_inputs(
    *, block: str, claims: Sequence[str], worker_count: int
) -> None:
    if block not in BLOCK_COUNTS or len(claims) != BLOCK_COUNTS[block]:
        raise FeverousFormalRunnerError("local block shape drifted")
    if type(worker_count) is not int or not 1 <= worker_count <= LOCAL_ITEM_WORKERS:
        raise FeverousFormalRunnerError("local worker count exceeds the frozen cap")
    if any(
        not isinstance(claim, str) or not claim or "\x00" in claim
        for claim in claims
    ):
        raise FeverousFormalRunnerError("block contains a malformed claim")


def _assemble_block_execution(
    *,
    block: str,
    items: Sequence[ItemExecution],
    worker_count: int,
    execution_scope: str,
    formation_total_items_eager_submitted: int | None,
) -> BlockExecution:
    item_rows = tuple(items)
    if (
        len(item_rows) != BLOCK_COUNTS[block]
        or tuple((item.block, item.ordinal) for item in item_rows)
        != tuple((block, ordinal) for ordinal in range(BLOCK_COUNTS[block]))
    ):
        raise FeverousFormalRunnerError("local block result order drifted")
    traces = tuple(trace for item in item_rows for trace in item.recipe_traces)
    if block in evaluator.BLOCK_ITEM_COUNTS:
        feature_receipt = evaluator.build_feature_receipt(
            block=block, traces=traces
        )
    else:
        # A_hold/M use the already-frozen policies and never refit or select an
        # evaluator.  Retain a content-free trace seal without pretending that
        # either anchor is an evaluator-formation block.
        anchor_feature_body = {
            "schema": f"{VERSION}_anchor_feature_receipt",
            "version": VERSION,
            "block": block,
            "receipt_purpose": "anchor_action_audit_only_not_evaluator_formation",
            "item_count": len(item_rows),
            "recipe_count_per_item": len(RECIPE_IDS),
            "recipe_registry": list(RECIPE_IDS),
            "trace_matrix_sha256": stable_hash(
                [trace.payload() for trace in traces]
            ),
            "evaluator_fit_or_policy_selection_authorized": False,
            "labels_family_gold_or_utility_accessed": False,
            "online_evaluator_calls": 0,
        }
        feature_receipt = {
            **anchor_feature_body,
            "feature_receipt_sha256": stable_hash(anchor_feature_body),
        }
    receipt_body = {
        "schema": f"{VERSION}_label_free_block_receipt",
        "version": VERSION,
        "block": block,
        "item_count": len(item_rows),
        "recipe_count_per_item": len(RECIPE_IDS),
        "logical_RAW_Hippo_Agent_work_units": 3 * len(item_rows),
        "local_worker_cap": LOCAL_ITEM_WORKERS,
        "local_worker_count": worker_count,
        "execution_scope": execution_scope,
        "shared_A_form_F_search_pool": (
            execution_scope == "shared_A_form_F_search_pool"
        ),
        "formation_total_items_eager_submitted": (
            formation_total_items_eager_submitted
        ),
        "all_items_submitted_before_join": True,
        "feature_receipt_sha256": feature_receipt["feature_receipt_sha256"],
        "item_commitment_vector_sha256": stable_hash(
            [item.item_commitment_sha256 for item in item_rows]
        ),
        "semantic_receipt_vector_sha256": stable_hash(
            [
                item.semantic_build.receipt["semantic_receipt_sha256"]
                for item in item_rows
            ]
        ),
        "operator_receipt_vector_sha256": stable_hash(
            [item.operator_receipt_sha256 for item in item_rows]
        ),
        "feature_production_vector_sha256": stable_hash(
            [
                trace.production_trace_sha256
                for item in item_rows
                for trace in item.feature_traces
            ]
        ),
        "labels_family_gold_or_Hippo_accessed": False,
        "online_evaluator_calls": 0,
    }
    receipt = {
        **receipt_body,
        "block_receipt_sha256": stable_hash(receipt_body),
    }
    return BlockExecution(
        block=block,
        items=item_rows,
        feature_receipt=MappingProxyType(dict(feature_receipt)),
        receipt=MappingProxyType(receipt),
    )


def execute_local_block(
    *,
    block: str,
    claims: Sequence[str],
    prepared_corpus: semantic.PreparedSemanticCorpus,
    minilm_backend: semantic.MiniLMBackend,
    ner_backend: semantic.NERBackend,
    nli_backend: semantic.NLIBackend,
    worker_count: int = LOCAL_ITEM_WORKERS,
) -> BlockExecution:
    """Execute one untouched anchor; formation blocks use the shared barrier."""

    _validate_local_inputs(block=block, claims=claims, worker_count=worker_count)
    if block in {"A_form", "F_search"}:
        raise FeverousFormalRunnerError(
            "formation blocks require execute_formation_blocks"
        )
    futures: list[Future[ItemExecution]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        for ordinal, claim in enumerate(claims):
            futures.append(
                pool.submit(
                    execute_local_item,
                    block=block,
                    ordinal=ordinal,
                    claim=claim,
                    prepared_corpus=prepared_corpus,
                    minilm_backend=minilm_backend,
                    ner_backend=ner_backend,
                    nli_backend=nli_backend,
                )
            )
        items = tuple(future.result() for future in futures)
    return _assemble_block_execution(
        block=block,
        items=items,
        worker_count=worker_count,
        execution_scope="single_anchor_block_pool",
        formation_total_items_eager_submitted=None,
    )


def execute_formation_blocks(
    *,
    A_form_claims: Sequence[str],
    F_search_claims: Sequence[str],
    prepared_corpus: semantic.PreparedSemanticCorpus,
    minilm_backend: semantic.MiniLMBackend,
    ner_backend: semantic.NERBackend,
    nli_backend: semantic.NLIBackend,
    worker_count: int = LOCAL_ITEM_WORKERS,
) -> FormationExecution:
    """Interleave all 144 formation items in one eager, at-most-64 pool."""

    _validate_local_inputs(
        block="A_form", claims=A_form_claims, worker_count=worker_count
    )
    _validate_local_inputs(
        block="F_search", claims=F_search_claims, worker_count=worker_count
    )
    claims_by_block = {
        "A_form": A_form_claims,
        "F_search": F_search_claims,
    }
    schedule = tuple(
        (block, ordinal)
        for ordinal in range(max(map(len, claims_by_block.values())))
        for block in ("A_form", "F_search")
        if ordinal < len(claims_by_block[block])
    )
    if len(schedule) != BLOCK_COUNTS["A_form"] + BLOCK_COUNTS["F_search"]:
        raise FeverousFormalRunnerError("formation submission schedule drifted")

    futures: list[tuple[str, int, Future[ItemExecution]]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        for block, ordinal in schedule:
            futures.append(
                (
                    block,
                    ordinal,
                    pool.submit(
                        execute_local_item,
                        block=block,
                        ordinal=ordinal,
                        claim=claims_by_block[block][ordinal],
                        prepared_corpus=prepared_corpus,
                        minilm_backend=minilm_backend,
                        ner_backend=ner_backend,
                        nli_backend=nli_backend,
                    ),
                )
            )
        collected: dict[str, list[ItemExecution]] = {
            "A_form": [],
            "F_search": [],
        }
        for expected_block, expected_ordinal, future in futures:
            item = future.result()
            if (item.block, item.ordinal) != (expected_block, expected_ordinal):
                raise FeverousFormalRunnerError("formation result identity drifted")
            collected[expected_block].append(item)

    total = len(schedule)
    a_form = _assemble_block_execution(
        block="A_form",
        items=collected["A_form"],
        worker_count=worker_count,
        execution_scope="shared_A_form_F_search_pool",
        formation_total_items_eager_submitted=total,
    )
    f_search = _assemble_block_execution(
        block="F_search",
        items=collected["F_search"],
        worker_count=worker_count,
        execution_scope="shared_A_form_F_search_pool",
        formation_total_items_eager_submitted=total,
    )
    receipt_body: dict[str, Any] = {
        "schema": f"{VERSION}_shared_formation_execution_receipt",
        "version": VERSION,
        "blocks": ["A_form", "F_search"],
        "item_counts": {
            "A_form": len(a_form.items),
            "F_search": len(f_search.items),
        },
        "total_item_count": total,
        "local_worker_cap": LOCAL_ITEM_WORKERS,
        "local_worker_count": worker_count,
        "single_shared_thread_pool": True,
        "interleaved_submission": True,
        "all_144_items_submitted_before_first_join": True,
        "submission_schedule_sha256": stable_hash(
            [[block, ordinal] for block, ordinal in schedule]
        ),
        "block_receipt_sha256s": {
            "A_form": a_form.receipt["block_receipt_sha256"],
            "F_search": f_search.receipt["block_receipt_sha256"],
        },
        "A_form_labels_accessed": False,
        "F_search_labels_created_or_accessed": False,
        "online_evaluator_calls": 0,
    }
    formation_receipt = {
        **receipt_body,
        "formation_execution_receipt_sha256": stable_hash(receipt_body),
    }
    return FormationExecution(
        A_form=a_form,
        F_search=f_search,
        receipt=MappingProxyType(formation_receipt),
    )


def _validated_labels(
    labels: Mapping[str, Any], *, block: str
) -> tuple[tuple[int, tuple[int, ...], str, str], ...]:
    if block not in {"A_form", "A_hold", "M_search"}:
        raise FeverousFormalRunnerError("labels requested for a label-free block")
    try:
        acquisition.verify_self_hash(labels, "block_labels_sha256")
    except acquisition.FeverousP6E2AcquisitionError as exc:
        raise FeverousFormalRunnerError("late label pack self-hash drifted") from exc
    rows = labels.get("items")
    if (
        set(labels) != _BLOCK_LABEL_KEYS
        or
        labels.get("schema") != acquisition.BLOCK_LABEL_SCHEMA
        or labels.get("version") != acquisition.VERSION
        or labels.get("block") != block
        or labels.get("item_count") != BLOCK_COUNTS[block]
        or not isinstance(rows, list)
        or len(rows) != BLOCK_COUNTS[block]
    ):
        raise FeverousFormalRunnerError("late label pack shape drifted")
    output: list[tuple[int, tuple[int, ...], str, str]] = []
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "ordinal",
            "gold_unit_indices",
            "family",
            "verdict",
        }:
            raise FeverousFormalRunnerError("late label row schema drifted")
        gold = row.get("gold_unit_indices")
        family = row.get("family")
        verdict = row.get("verdict")
        if (
            row.get("ordinal") != ordinal
            or not isinstance(gold, list)
            or not 2 <= len(gold) <= 5
            or any(type(value) is not int or not 0 <= value < acquisition.CORPUS_UNIT_COUNT for value in gold)
            or tuple(sorted(set(gold))) != tuple(gold)
            or family not in acquisition.FAMILIES
            or verdict not in acquisition.VERDICTS
        ):
            raise FeverousFormalRunnerError("late label row content drifted")
        output.append((ordinal, tuple(gold), str(family), str(verdict)))
    return tuple(output)


def _validated_item_outputs(
    item: ItemExecution, *, block: str, ordinal: int
) -> Mapping[str, tuple[int, ...]]:
    if (
        not isinstance(item, ItemExecution)
        or item.block != block
        or item.ordinal != ordinal
    ):
        raise FeverousFormalRunnerError("executed item identity drifted")
    _require_sha256(item.item_commitment_sha256, "item commitment")
    if tuple(trace.recipe_id for trace in item.action_traces) != RECIPE_IDS:
        raise FeverousFormalRunnerError("executed recipe matrix drifted")
    for trace in item.action_traces:
        try:
            operator.verify_action_trace(trace)
        except operator.FeverousP6OperatorError as exc:
            raise FeverousFormalRunnerError("executed action seal drifted") from exc
        if (
            len(trace.output_top5) != TOP_K
            or len(set(trace.output_top5)) != TOP_K
            or any(
                type(value) is not int
                or not 0 <= value < acquisition.CORPUS_UNIT_COUNT
                for value in trace.output_top5
            )
        ):
            raise FeverousFormalRunnerError("executed action output drifted")
    return item.outputs


def a_form_utility_matrix(
    *, block: BlockExecution, labels: Mapping[str, Any]
) -> dict[tuple[str, str], Fraction]:
    """Open A_form labels only after its complete feature matrix is sealed."""

    if block.block != "A_form" or len(block.items) != BLOCK_COUNTS["A_form"]:
        raise FeverousFormalRunnerError("A_form execution is incomplete")
    label_rows = _validated_labels(labels, block="A_form")
    utilities: dict[tuple[str, str], Fraction] = {}
    for ordinal, (item, (_label_ordinal, gold, _family, _verdict)) in enumerate(
        zip(block.items, label_rows)
    ):
        outputs = _validated_item_outputs(
            item, block="A_form", ordinal=ordinal
        )
        for recipe_id, output in outputs.items():
            utilities[(item.item_commitment_sha256, recipe_id)] = evaluator.item_utility(
                output, gold
            ).value
    return utilities


def _comparison_payload(deltas: Sequence[Fraction]) -> dict[str, object]:
    exact = evaluator.exact_magnitude_preserving_sign_flip(deltas)
    return exact.payload()


def score_anchor_block(
    *,
    block: BlockExecution,
    labels: Mapping[str, Any],
    hippo_top5: Sequence[Sequence[int]],
    e0_recipe_id: str,
    e2_recipe_id: str,
    evaluator_comparison_identifiable: bool,
) -> dict[str, Any]:
    """Score A_hold or M_search once all three logical arms are terminal."""

    if (
        block.block not in {"A_hold", "M_search"}
        or len(block.items) != BLOCK_COUNTS.get(block.block, -1)
    ):
        raise FeverousFormalRunnerError("anchor scorer received the wrong block")
    if e0_recipe_id not in RECIPE_IDS or e2_recipe_id not in RECIPE_IDS:
        raise FeverousFormalRunnerError("frozen policy recipe is invalid")
    if len(hippo_top5) != len(block.items):
        raise FeverousFormalRunnerError("Hippo result count drifted")
    if type(evaluator_comparison_identifiable) is not bool:
        raise FeverousFormalRunnerError("evaluator identifiability flag is not Boolean")
    hippo = tuple(tuple(row) for row in hippo_top5)
    if any(
        len(row) != TOP_K
        or len(set(row)) != TOP_K
        or any(type(value) is not int or not 0 <= value < acquisition.CORPUS_UNIT_COUNT for value in row)
        for row in hippo
    ):
        raise FeverousFormalRunnerError("Hippo output is not an exact top five")
    label_rows = _validated_labels(labels, block=block.block)

    rows: list[dict[str, object]] = []
    e2_minus_e0: list[Fraction] = []
    e2_minus_hippo: list[Fraction] = []
    e2_minus_raw: list[Fraction] = []
    family_deltas: dict[str, list[Fraction]] = {
        family: [] for family in acquisition.FAMILIES
    }
    complete_counts = {"E0": 0, "E2": 0, "HippoRAG": 0, "RAW": 0}
    recipe_total_u = {recipe_id: Fraction(0) for recipe_id in RECIPE_IDS}
    recipe_complete_counts = {recipe_id: 0 for recipe_id in RECIPE_IDS}
    for ordinal, (item, hippo_row, (_label_ordinal, gold, family, verdict)) in enumerate(
        zip(block.items, hippo, label_rows)
    ):
        outputs = _validated_item_outputs(
            item, block=block.block, ordinal=ordinal
        )
        for recipe_id in RECIPE_IDS:
            recipe_utility = evaluator.item_utility(outputs[recipe_id], gold)
            recipe_total_u[recipe_id] += recipe_utility.value
            recipe_complete_counts[recipe_id] += int(recipe_utility.complete)
        raw = outputs["R0_DENSE5"]
        e0 = outputs[e0_recipe_id]
        e2 = outputs[e2_recipe_id]
        scored = {
            "E0": evaluator.item_utility(e0, gold),
            "E2": evaluator.item_utility(e2, gold),
            "HippoRAG": evaluator.item_utility(hippo_row, gold),
            "RAW": evaluator.item_utility(raw, gold),
        }
        for arm, utility in scored.items():
            complete_counts[arm] += int(utility.complete)
        delta_e0 = scored["E2"].value - scored["E0"].value
        delta_hippo = scored["E2"].value - scored["HippoRAG"].value
        delta_raw = scored["E2"].value - scored["RAW"].value
        e2_minus_e0.append(delta_e0)
        e2_minus_hippo.append(delta_hippo)
        e2_minus_raw.append(delta_raw)
        family_deltas[family].append(delta_hippo)
        rows.append(
            {
                "item_commitment_sha256": item.item_commitment_sha256,
                "family": family,
                "verdict": verdict,
                "utilities": {
                    arm: _fraction_payload(utility.value)
                    for arm, utility in scored.items()
                },
                "complete": {arm: utility.complete for arm, utility in scored.items()},
            }
        )

    evaluator_test = _comparison_payload(e2_minus_e0)
    hippo_test = _comparison_payload(e2_minus_hippo)
    raw_test = _comparison_payload(e2_minus_raw)
    family_sums = {
        family: sum(values, Fraction(0)) for family, values in family_deltas.items()
    }
    primary_passed = (
        bool(hippo_test["promoted"])
        and all(value > 0 for value in family_sums.values())
    )
    promoted = bool(evaluator_test["promoted"]) and evaluator_comparison_identifiable
    raw_advantage_overcome = (
        bool(raw_test["promoted"])
        and complete_counts["E2"] >= complete_counts["RAW"]
    )
    body: dict[str, Any] = {
        "schema": f"{VERSION}_{block.block}_score_receipt",
        "version": VERSION,
        "block": block.block,
        "item_count": len(rows),
        "E0_recipe_id": e0_recipe_id,
        "E2_recipe_id": e2_recipe_id,
        "evaluator_comparison_identifiable": evaluator_comparison_identifiable,
        "E2_minus_E0": evaluator_test,
        "E2_minus_HippoRAG": hippo_test,
        "E2_minus_RAW": raw_test,
        "E2_minus_HippoRAG_family_sums": {
            family: _fraction_payload(value) for family, value in family_sums.items()
        },
        "complete_counts": complete_counts,
        "all_four_recipe_aggregates": {
            recipe_id: {
                "total_U": _fraction_payload(recipe_total_u[recipe_id]),
                "complete_count": recipe_complete_counts[recipe_id],
            }
            for recipe_id in RECIPE_IDS
        },
        "A_hold_real_domain_primary_passed": (
            primary_passed if block.block == "A_hold" else None
        ),
        "evaluator_promoted": promoted if block.block == "A_hold" else None,
        "M_L5_passed": promoted if block.block == "M_search" else None,
        "RAW_complete_advantage_overcome": raw_advantage_overcome,
        "item_utility_matrix_sha256": stable_hash(rows),
        "item_level_utility_values_persisted": False,
        "aggregate_utility_values_persisted": True,
        "official_verdict_or_FEVEROUS_score_used": False,
        "online_evaluator_calls": 0,
    }
    return {**body, "score_receipt_sha256": stable_hash(body)}


__all__ = [
    "BLOCK_COUNTS",
    "BlockExecution",
    "FormationExecution",
    "FeverousFormalRunnerError",
    "ItemExecution",
    "LOCAL_ITEM_WORKERS",
    "RECIPE_IDS",
    "VERSION",
    "action_trace_content_free_receipt",
    "a_form_utility_matrix",
    "claims_from_block_view",
    "corpus_view_to_semantic_units",
    "execute_local_block",
    "execute_formation_blocks",
    "execute_local_item",
    "item_commitment",
    "score_anchor_block",
    "stable_hash",
]
