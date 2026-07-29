"""Process-isolated late-label scorer for the frozen WikiSQL UAO study.

This module is the only study component that joins A_hold labels to all three
already-durable action packs.  It deliberately imports only the pure reality
algebra: no source compiler, action runtime, archive, model, network client, or
online evaluator is importable through this boundary.

The scorer validates five content-addressed private packs, constructs
``reality.ItemMeasurement`` objects, evaluates ``aggregate_primary`` offline,
and returns:

* a private per-item score pack; and
* a content-free safe aggregate receipt containing only aggregate effects,
  exact sign-flip fractions, pass decisions, and input/output commitments.

Formal WikiSQL tables contain at least eleven rows.  Consequently every arm
must provide exactly five distinct integer ordinals; ``None`` padding is never
accepted by this formal scorer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks import wikisql_uao_reality_v1 as reality


VERSION = "wikisql_uao_scorer_v1"
STUDY_ID = "UAO_P4_WIKISQL_TYPED_RELATIONAL_ROW_EVIDENCE_V1"
BLOCK = "A_hold"
ITEM_COUNT = 72
FAMILY_COUNT = 24
ARMS = ("Agent", "RAW", "HippoRAG")

ACTION_VIEW_PACK_SCHEMA = (
    "wikisql_uao_source_compiler_v1_private_action_view_pack_v1"
)
MINIMAL_LABEL_PACK_SCHEMA = f"{VERSION}_private_minimal_label_pack_v1"
ACTION_PACK_SCHEMA = "wikisql_uao_action_runtime_v1_private_action_pack_v1"
PRIVATE_SCORE_PACK_SCHEMA = f"{VERSION}_private_per_item_score_pack_v1"
SAFE_AGGREGATE_SCHEMA = f"{VERSION}_safe_aggregate_receipt_v1"
SAFE_TERMINAL_SCHEMA = f"{VERSION}_safe_terminal_v1"
LABEL_RELEASE_POLICY = "after_all_A_hold_three_arm_actions_are_sealed"

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_VIEW_PACK_KEYS = frozenset(
    {
        "block",
        "contains_labels",
        "item_count",
        "items",
        "schema",
        "self_sha256",
        "study_id",
    }
)
_VIEW_ITEM_KEYS = frozenset(
    {
        "opaque_item_id",
        "physical_rows",
        "question",
        "table_header",
        "table_types",
    }
)
_LABEL_PACK_KEYS = frozenset(
    {
        "action_view_pack_sha256",
        "block",
        "item_count",
        "items",
        "release_policy",
        "schema",
        "self_sha256",
        "study_id",
    }
)
_LABEL_ITEM_KEYS = frozenset(
    {
        "action_view_sha256",
        "family",
        "gold_row_ids",
        "item_commitment_sha256",
        "opaque_item_id",
        "sqlite_rowid_cross_checked",
        "table_row_count",
    }
)
_ACTION_PACK_KEYS = frozenset(
    {
        "action_view_pack_sha256",
        "arm",
        "block",
        "item_count",
        "items",
        "schema",
        "self_sha256",
        "study_id",
    }
)
_ACTION_ITEM_KEYS = frozenset({"opaque_item_id", "top5_row_ids"})


class WikiSQLUAOScorerError(RuntimeError):
    """A late-label pack, join, score, or durable output drifted."""


Cell = str | int | float


def canonical_json_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise WikiSQLUAOScorerError("value is not canonical JSON") from exc
    return raw + (b"\n" if newline else b"")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _content_addressed(value: Mapping[str, object]) -> dict[str, object]:
    if "self_sha256" in value:
        raise WikiSQLUAOScorerError("payload already contains self_sha256")
    return {**value, "self_sha256": canonical_sha256(value)}


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise WikiSQLUAOScorerError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise WikiSQLUAOScorerError(f"{field} must be an array")
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise WikiSQLUAOScorerError(f"{field} is not a SHA-256 commitment")
    return value


def _text(
    value: object,
    *,
    field: str,
    allow_empty: bool = False,
    maximum: int = 1_000_000,
) -> str:
    if (
        not isinstance(value, str)
        or "\x00" in value
        or len(value) > maximum
        or (not allow_empty and not value.strip())
    ):
        raise WikiSQLUAOScorerError(f"{field} is invalid")
    return value


def _cell(value: object, *, field: str) -> Cell:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise WikiSQLUAOScorerError(f"{field} is not a scalar table cell")
    if isinstance(value, str):
        _text(value, field=field, allow_empty=True)
    if isinstance(value, float) and not math.isfinite(value):
        raise WikiSQLUAOScorerError(f"{field} is non-finite")
    return value


def _verify_self_hash(
    value: Mapping[str, object],
    *,
    exact_keys: frozenset[str],
    field: str,
) -> str:
    if set(value) != exact_keys:
        raise WikiSQLUAOScorerError(f"{field} fields drifted")
    supplied = _sha256(value.get("self_sha256"), field=f"{field} self hash")
    base = {
        key: child
        for key, child in value.items()
        if key != "self_sha256"
    }
    if canonical_sha256(base) != supplied:
        raise WikiSQLUAOScorerError(f"{field} content hash drifted")
    return supplied


@dataclass(frozen=True, slots=True)
class ActionView:
    item_id: str
    payload_sha256: str
    row_count: int


@dataclass(frozen=True, slots=True)
class MinimalLabel:
    item_id: str
    item_commitment_sha256: str
    action_view_sha256: str
    family: str
    gold_row_ids: tuple[int, ...]
    table_row_count: int


def _decode_action_view_item(value: object, *, ordinal: int) -> ActionView:
    row = _mapping(value, field=f"action view item[{ordinal}]")
    if set(row) != _VIEW_ITEM_KEYS:
        raise WikiSQLUAOScorerError(
            f"action view item[{ordinal}] fields drifted"
        )
    item_id = _sha256(
        row.get("opaque_item_id"),
        field=f"action view item[{ordinal}] ID",
    )
    question = _text(
        row.get("question"),
        field=f"action view item[{ordinal}] question",
        maximum=32_000,
    )
    header = tuple(
        _text(
            child,
            field=f"action view item[{ordinal}] header[{index}]",
            maximum=16_000,
        )
        for index, child in enumerate(
            _sequence(
                row.get("table_header"),
                field=f"action view item[{ordinal}] header",
            )
        )
    )
    types = tuple(
        _text(
            child,
            field=f"action view item[{ordinal}] type[{index}]",
            maximum=16,
        )
        for index, child in enumerate(
            _sequence(
                row.get("table_types"),
                field=f"action view item[{ordinal}] types",
            )
        )
    )
    physical_rows = tuple(
        tuple(
            _cell(
                child,
                field=(
                    f"action view item[{ordinal}]"
                    f" row[{row_index}][{column_index}]"
                ),
            )
            for column_index, child in enumerate(
                _sequence(
                    raw_row,
                    field=f"action view item[{ordinal}] row[{row_index}]",
                )
            )
        )
        for row_index, raw_row in enumerate(
            _sequence(
                row.get("physical_rows"),
                field=f"action view item[{ordinal}] rows",
            )
        )
    )
    try:
        table = reality.WikiSQLTable(
            table_id=item_id,
            header=header,
            types=types,
            rows=physical_rows,
        )
    except reality.WikiSQLUAORealityError as exc:
        raise WikiSQLUAOScorerError(
            f"action view item[{ordinal}] table drifted"
        ) from exc
    if (
        not reality.MIN_TABLE_ROWS
        <= len(table.rows)
        <= reality.MAX_TABLE_ROWS
    ):
        raise WikiSQLUAOScorerError(
            f"action view item[{ordinal}] row count drifted"
        )
    # Reading the question is validation only; no source/model action occurs.
    del question
    return ActionView(
        item_id=item_id,
        payload_sha256=canonical_sha256(row),
        row_count=len(table.rows),
    )


def _decode_action_view_pack(
    value: Mapping[str, object],
) -> tuple[str, tuple[ActionView, ...]]:
    pack_hash = _verify_self_hash(
        value,
        exact_keys=_VIEW_PACK_KEYS,
        field="A_hold action view pack",
    )
    if (
        value.get("schema") != ACTION_VIEW_PACK_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("block") != BLOCK
        or value.get("contains_labels") is not False
        or value.get("item_count") != ITEM_COUNT
    ):
        raise WikiSQLUAOScorerError("A_hold action view pack envelope drifted")
    rows = _sequence(value.get("items"), field="A_hold action view items")
    if len(rows) != ITEM_COUNT:
        raise WikiSQLUAOScorerError("A_hold action view item count drifted")
    decoded = tuple(
        _decode_action_view_item(row, ordinal=index)
        for index, row in enumerate(rows)
    )
    identifiers = tuple(row.item_id for row in decoded)
    if (
        len(set(identifiers)) != ITEM_COUNT
        or identifiers != tuple(sorted(identifiers))
    ):
        raise WikiSQLUAOScorerError(
            "A_hold action view IDs repeat or are not canonically sorted"
        )
    return pack_hash, decoded


def _decode_minimal_label_item(
    value: object, *, ordinal: int
) -> MinimalLabel:
    row = _mapping(value, field=f"minimal label item[{ordinal}]")
    if set(row) != _LABEL_ITEM_KEYS:
        raise WikiSQLUAOScorerError(
            f"minimal label item[{ordinal}] fields drifted"
        )
    item_id = _sha256(
        row.get("opaque_item_id"),
        field=f"minimal label item[{ordinal}] ID",
    )
    item_commitment = _sha256(
        row.get("item_commitment_sha256"),
        field=f"minimal label item[{ordinal}] source commitment",
    )
    action_view_sha256 = _sha256(
        row.get("action_view_sha256"),
        field=f"minimal label item[{ordinal}] action view commitment",
    )
    family = row.get("family")
    if not isinstance(family, str) or family not in reality.FAMILY_ORDER:
        raise WikiSQLUAOScorerError(
            f"minimal label item[{ordinal}] family drifted"
        )
    raw_gold = tuple(
        _sequence(
            row.get("gold_row_ids"),
            field=f"minimal label item[{ordinal}] gold rows",
        )
    )
    if (
        not 1 <= len(raw_gold) <= reality.MAX_GOLD_ROWS
        or any(
            isinstance(child, bool)
            or not isinstance(child, int)
            or child < 0
            for child in raw_gold
        )
        or tuple(sorted(set(raw_gold))) != raw_gold
    ):
        raise WikiSQLUAOScorerError(
            f"minimal label item[{ordinal}] gold rows drifted"
        )
    row_count = row.get("table_row_count")
    if (
        isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or not reality.MIN_TABLE_ROWS
        <= row_count
        <= reality.MAX_TABLE_ROWS
        or row.get("sqlite_rowid_cross_checked") is not True
    ):
        raise WikiSQLUAOScorerError(
            f"minimal label item[{ordinal}] row binding drifted"
        )
    return MinimalLabel(
        item_id=item_id,
        item_commitment_sha256=item_commitment,
        action_view_sha256=action_view_sha256,
        family=family,
        gold_row_ids=raw_gold,  # type: ignore[arg-type]
        table_row_count=row_count,
    )


def build_minimal_label_pack(
    *,
    action_view_pack_sha256: str,
    items: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build the exact late-release projection; no SQL/source row is accepted."""

    view_hash = _sha256(
        action_view_pack_sha256,
        field="minimal label action-view pack commitment",
    )
    raw_items = tuple(
        _sequence(items, field="minimal label items")
    )
    if len(raw_items) != ITEM_COUNT:
        raise WikiSQLUAOScorerError("minimal label item count drifted")
    decoded = tuple(
        _decode_minimal_label_item(row, ordinal=index)
        for index, row in enumerate(raw_items)
    )
    identifiers = tuple(row.item_id for row in decoded)
    if (
        len(set(identifiers)) != ITEM_COUNT
        or identifiers != tuple(sorted(identifiers))
    ):
        raise WikiSQLUAOScorerError(
            "minimal label IDs repeat or are not canonically sorted"
        )
    family_counts = Counter(row.family for row in decoded)
    if any(
        family_counts[family] != FAMILY_COUNT
        for family in reality.FAMILY_ORDER
    ):
        raise WikiSQLUAOScorerError(
            "minimal labels are not balanced 24xEQ/GT/LT"
        )
    return _content_addressed(
        {
            "action_view_pack_sha256": view_hash,
            "block": BLOCK,
            "item_count": ITEM_COUNT,
            "items": [dict(row) for row in raw_items],
            "release_policy": LABEL_RELEASE_POLICY,
            "schema": MINIMAL_LABEL_PACK_SCHEMA,
            "study_id": STUDY_ID,
        }
    )


def _decode_minimal_label_pack(
    value: Mapping[str, object],
    *,
    expected_action_view_pack_sha256: str,
) -> tuple[str, tuple[MinimalLabel, ...]]:
    pack_hash = _verify_self_hash(
        value,
        exact_keys=_LABEL_PACK_KEYS,
        field="minimal A_hold label pack",
    )
    if (
        value.get("schema") != MINIMAL_LABEL_PACK_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("block") != BLOCK
        or value.get("item_count") != ITEM_COUNT
        or value.get("release_policy") != LABEL_RELEASE_POLICY
        or value.get("action_view_pack_sha256")
        != expected_action_view_pack_sha256
    ):
        raise WikiSQLUAOScorerError("minimal A_hold label envelope drifted")
    rows = _sequence(value.get("items"), field="minimal A_hold labels")
    if len(rows) != ITEM_COUNT:
        raise WikiSQLUAOScorerError("minimal A_hold label count drifted")
    decoded = tuple(
        _decode_minimal_label_item(row, ordinal=index)
        for index, row in enumerate(rows)
    )
    identifiers = tuple(row.item_id for row in decoded)
    if (
        len(set(identifiers)) != ITEM_COUNT
        or identifiers != tuple(sorted(identifiers))
    ):
        raise WikiSQLUAOScorerError(
            "minimal A_hold label IDs repeat or are not sorted"
        )
    family_counts = Counter(row.family for row in decoded)
    if any(
        family_counts[family] != FAMILY_COUNT
        for family in reality.FAMILY_ORDER
    ):
        raise WikiSQLUAOScorerError(
            "minimal labels are not balanced 24xEQ/GT/LT"
        )
    return pack_hash, decoded


def _decode_action_pack(
    value: Mapping[str, object],
    *,
    expected_arm: str,
    expected_action_view_pack_sha256: str,
) -> tuple[str, dict[str, tuple[int, ...]]]:
    pack_hash = _verify_self_hash(
        value,
        exact_keys=_ACTION_PACK_KEYS,
        field=f"{expected_arm} action pack",
    )
    if expected_arm not in ARMS:
        raise WikiSQLUAOScorerError("expected action arm is outside registry")
    if (
        value.get("schema") != ACTION_PACK_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("block") != BLOCK
        or value.get("arm") != expected_arm
        or value.get("action_view_pack_sha256")
        != expected_action_view_pack_sha256
        or value.get("item_count") != ITEM_COUNT
    ):
        raise WikiSQLUAOScorerError(
            f"{expected_arm} action pack envelope drifted"
        )
    raw_rows = _sequence(
        value.get("items"), field=f"{expected_arm} action items"
    )
    if len(raw_rows) != ITEM_COUNT:
        raise WikiSQLUAOScorerError(
            f"{expected_arm} action item count drifted"
        )
    result: dict[str, tuple[int, ...]] = {}
    ordered_ids: list[str] = []
    for ordinal, raw in enumerate(raw_rows):
        row = _mapping(raw, field=f"{expected_arm} action item[{ordinal}]")
        if set(row) != _ACTION_ITEM_KEYS:
            raise WikiSQLUAOScorerError(
                f"{expected_arm} action item[{ordinal}] fields drifted"
            )
        item_id = _sha256(
            row.get("opaque_item_id"),
            field=f"{expected_arm} action item[{ordinal}] ID",
        )
        top5 = tuple(
            _sequence(
                row.get("top5_row_ids"),
                field=f"{expected_arm} action item[{ordinal}] top5",
            )
        )
        if (
            len(top5) != reality.TOP_K
            or len(set(top5)) != reality.TOP_K
            or any(
                isinstance(child, bool)
                or not isinstance(child, int)
                or child < 0
                for child in top5
            )
        ):
            raise WikiSQLUAOScorerError(
                f"{expected_arm} top5 must be five distinct integer ordinals"
            )
        if item_id in result:
            raise WikiSQLUAOScorerError(
                f"{expected_arm} action IDs repeat"
            )
        result[item_id] = top5  # type: ignore[assignment]
        ordered_ids.append(item_id)
    if tuple(ordered_ids) != tuple(sorted(ordered_ids)):
        raise WikiSQLUAOScorerError(
            f"{expected_arm} action IDs are not canonically sorted"
        )
    return pack_hash, result


def _arm_score_payload(
    top5: tuple[int, ...],
    gold: tuple[int, ...],
) -> dict[str, object]:
    scored = reality.score_item(top5, gold)
    return {
        "complete": scored.complete,
        "hits": scored.hits,
        "top5_row_ids": list(top5),
        "utility": scored.utility,
    }


def _comparison_payload(
    comparison: reality.BaselineComparison,
) -> dict[str, object]:
    return {
        "baseline": comparison.baseline,
        "observed_net_u": comparison.observed_net_u,
        "family_net_u": {
            family: value for family, value in comparison.family_net_u
        },
        "exact_p_numerator": comparison.sign_flip.p_value.numerator,
        "exact_p_denominator": comparison.sign_flip.p_value.denominator,
        "nonzero_pair_count": comparison.sign_flip.nonzero_pair_count,
        "passed": comparison.passed,
    }


@dataclass(frozen=True, slots=True)
class ScoringArtifacts:
    private_score_pack: Mapping[str, object]
    safe_aggregate_receipt: Mapping[str, object]


def score_late_labels(
    *,
    action_view_pack: Mapping[str, object],
    minimal_label_pack: Mapping[str, object],
    agent_action_pack: Mapping[str, object],
    raw_action_pack: Mapping[str, object],
    hipporag_action_pack: Mapping[str, object],
) -> ScoringArtifacts:
    """Join late labels only after all three exact action packs validate."""

    view_pack_hash, views = _decode_action_view_pack(action_view_pack)
    # Validate all three durable action commitments before opening/decoding the
    # label payload.  Function argument evaluation cannot expose labels to an
    # action runtime because this module imports no such runtime.
    agent_hash, agent_actions = _decode_action_pack(
        agent_action_pack,
        expected_arm="Agent",
        expected_action_view_pack_sha256=view_pack_hash,
    )
    raw_hash, raw_actions = _decode_action_pack(
        raw_action_pack,
        expected_arm="RAW",
        expected_action_view_pack_sha256=view_pack_hash,
    )
    hipporag_hash, hipporag_actions = _decode_action_pack(
        hipporag_action_pack,
        expected_arm="HippoRAG",
        expected_action_view_pack_sha256=view_pack_hash,
    )
    label_pack_hash, labels = _decode_minimal_label_pack(
        minimal_label_pack,
        expected_action_view_pack_sha256=view_pack_hash,
    )

    view_by_id = {row.item_id: row for row in views}
    label_by_id = {row.item_id: row for row in labels}
    identifier_sets = (
        set(view_by_id),
        set(label_by_id),
        set(agent_actions),
        set(raw_actions),
        set(hipporag_actions),
    )
    if any(len(rows) != ITEM_COUNT for rows in identifier_sets) or any(
        rows != identifier_sets[0] for rows in identifier_sets[1:]
    ):
        raise WikiSQLUAOScorerError(
            "view, label, and three action ID sets are not identical 72-item sets"
        )

    measurements: list[reality.ItemMeasurement] = []
    private_items: list[dict[str, object]] = []
    for item_id in sorted(view_by_id):
        view = view_by_id[item_id]
        label = label_by_id[item_id]
        if (
            label.action_view_sha256 != view.payload_sha256
            or label.table_row_count != view.row_count
        ):
            raise WikiSQLUAOScorerError(
                "label action_view_sha256 or table_row_count binding drifted"
            )
        if any(row >= view.row_count for row in label.gold_row_ids):
            raise WikiSQLUAOScorerError("gold row ordinal exceeds table row count")
        top5_by_arm = {
            "Agent": agent_actions[item_id],
            "RAW": raw_actions[item_id],
            "HippoRAG": hipporag_actions[item_id],
        }
        if any(
            any(row >= view.row_count for row in top5)
            for top5 in top5_by_arm.values()
        ):
            raise WikiSQLUAOScorerError(
                "action row ordinal exceeds table row count"
            )
        measurement = reality.ItemMeasurement(
            item_commitment_sha256=label.item_commitment_sha256,
            family=label.family,
            gold_row_ids=label.gold_row_ids,
            agent_top5=top5_by_arm["Agent"],
            raw_top5=top5_by_arm["RAW"],
            hipporag_top5=top5_by_arm["HippoRAG"],
        )
        measurements.append(measurement)
        agent_score = _arm_score_payload(
            top5_by_arm["Agent"], label.gold_row_ids
        )
        raw_score = _arm_score_payload(
            top5_by_arm["RAW"], label.gold_row_ids
        )
        hipporag_score = _arm_score_payload(
            top5_by_arm["HippoRAG"], label.gold_row_ids
        )
        private_items.append(
            {
                "Agent": agent_score,
                "HippoRAG": hipporag_score,
                "RAW": raw_score,
                "agent_minus_hipporag": (
                    agent_score["utility"] - hipporag_score["utility"]
                ),
                "agent_minus_raw": (
                    agent_score["utility"] - raw_score["utility"]
                ),
                "family": label.family,
                "gold_row_ids": list(label.gold_row_ids),
                "item_commitment_sha256": label.item_commitment_sha256,
                "opaque_item_id": item_id,
                "table_row_count": view.row_count,
            }
        )

    try:
        aggregate = reality.aggregate_primary(tuple(measurements))
    except reality.WikiSQLUAORealityError as exc:
        raise WikiSQLUAOScorerError("offline primary aggregation failed") from exc
    if (
        aggregate.item_count != ITEM_COUNT
        or aggregate.family_counts
        != tuple((family, FAMILY_COUNT) for family in reality.FAMILY_ORDER)
    ):
        raise WikiSQLUAOScorerError("offline aggregate cohort shape drifted")

    input_commitments = {
        "Agent_action_pack_sha256": agent_hash,
        "HippoRAG_action_pack_sha256": hipporag_hash,
        "RAW_action_pack_sha256": raw_hash,
        "action_view_pack_sha256": view_pack_hash,
        "minimal_label_pack_sha256": label_pack_hash,
    }
    private_score_pack = _content_addressed(
        {
            "block": BLOCK,
            "input_commitments": input_commitments,
            "item_count": ITEM_COUNT,
            "items": private_items,
            "schema": PRIVATE_SCORE_PACK_SCHEMA,
            "study_id": STUDY_ID,
        }
    )
    raw_comparison = _comparison_payload(aggregate.agent_vs_raw)
    hipporag_comparison = _comparison_payload(
        aggregate.agent_vs_hipporag
    )
    safe_aggregate = _content_addressed(
        {
            "alpha_denominator": reality.PROMOTION_ALPHA.denominator,
            "alpha_numerator": reality.PROMOTION_ALPHA.numerator,
            "block": BLOCK,
            "family_counts": {
                family: count for family, count in aggregate.family_counts
            },
            "input_commitments": input_commitments,
            "item_count": ITEM_COUNT,
            "offline_aggregate_primary_call_count": 1,
            "online_evaluation_count": 0,
            "primary_passed": aggregate.passed,
            "private_score_pack_sha256": private_score_pack["self_sha256"],
            "schema": SAFE_AGGREGATE_SCHEMA,
            "status": (
                "PASS_REALITY_PRIMARY"
                if aggregate.passed
                else "FAIL_REALITY_PRIMARY"
            ),
            "study_id": STUDY_ID,
            "Agent_vs_RAW": raw_comparison,
            "Agent_vs_HippoRAG": hipporag_comparison,
        }
    )
    return ScoringArtifacts(
        private_score_pack=private_score_pack,
        safe_aggregate_receipt=safe_aggregate,
    )


def _read_canonical_private_pack(path: Path, *, field: str) -> dict[str, object]:
    try:
        metadata = path.lstat()
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WikiSQLUAOScorerError(f"{field} is unreadable") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or not isinstance(value, dict)
        or canonical_json_bytes(value, newline=True) != raw
    ):
        raise WikiSQLUAOScorerError(
            f"{field} metadata or canonical encoding drifted"
        )
    return value


def _write_exclusive(
    path: Path,
    value: Mapping[str, object],
) -> str:
    raw = canonical_json_bytes(value, newline=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise WikiSQLUAOScorerError(
            "exclusive scorer output creation failed"
        ) from exc
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise WikiSQLUAOScorerError("scorer output write stalled")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if (
        path.is_symlink()
        or stat.S_IMODE(path.lstat().st_mode) != 0o600
        or path.read_bytes() != raw
    ):
        raise WikiSQLUAOScorerError("scorer output verification failed")
    return hashlib.sha256(raw).hexdigest()


def run_cli(
    *,
    action_view_path: Path,
    minimal_label_path: Path,
    agent_action_path: Path,
    raw_action_path: Path,
    hipporag_action_path: Path,
    private_score_output: Path,
    safe_receipt_output: Path,
    terminal_output: Path,
) -> Mapping[str, object]:
    """Read canonical private inputs and write private, safe, terminal in order."""

    action_view_value = _read_canonical_private_pack(
        action_view_path, field="action view pack"
    )
    view_pack_hash, _ = _decode_action_view_pack(action_view_value)
    agent_action_value = _read_canonical_private_pack(
        agent_action_path, field="Agent action pack"
    )
    raw_action_value = _read_canonical_private_pack(
        raw_action_path, field="RAW action pack"
    )
    hipporag_action_value = _read_canonical_private_pack(
        hipporag_action_path, field="HippoRAG action pack"
    )
    # Validate all durable label-free actions before even opening the minimal
    # A_hold label file.  score_late_labels repeats these checks at the join.
    _decode_action_pack(
        agent_action_value,
        expected_arm="Agent",
        expected_action_view_pack_sha256=view_pack_hash,
    )
    _decode_action_pack(
        raw_action_value,
        expected_arm="RAW",
        expected_action_view_pack_sha256=view_pack_hash,
    )
    _decode_action_pack(
        hipporag_action_value,
        expected_arm="HippoRAG",
        expected_action_view_pack_sha256=view_pack_hash,
    )
    minimal_label_value = _read_canonical_private_pack(
        minimal_label_path, field="minimal label pack"
    )
    artifacts = score_late_labels(
        action_view_pack=action_view_value,
        minimal_label_pack=minimal_label_value,
        agent_action_pack=agent_action_value,
        raw_action_pack=raw_action_value,
        hipporag_action_pack=hipporag_action_value,
    )
    private_file_sha256 = _write_exclusive(
        private_score_output, artifacts.private_score_pack
    )
    safe_file_sha256 = _write_exclusive(
        safe_receipt_output, artifacts.safe_aggregate_receipt
    )
    terminal = _content_addressed(
        {
            "block": BLOCK,
            "primary_passed": artifacts.safe_aggregate_receipt[
                "primary_passed"
            ],
            "private_score_file_sha256": private_file_sha256,
            "private_score_pack_sha256": artifacts.private_score_pack[
                "self_sha256"
            ],
            "safe_aggregate_file_sha256": safe_file_sha256,
            "safe_aggregate_receipt_sha256": artifacts.safe_aggregate_receipt[
                "self_sha256"
            ],
            "schema": SAFE_TERMINAL_SCHEMA,
            "status": "completed",
            "study_id": STUDY_ID,
        }
    )
    _write_exclusive(terminal_output, terminal)
    return terminal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-view-pack", type=Path, required=True)
    parser.add_argument("--minimal-label-pack", type=Path, required=True)
    parser.add_argument("--agent-action-pack", type=Path, required=True)
    parser.add_argument("--raw-action-pack", type=Path, required=True)
    parser.add_argument("--hipporag-action-pack", type=Path, required=True)
    parser.add_argument("--private-score-output", type=Path, required=True)
    parser.add_argument("--safe-receipt-output", type=Path, required=True)
    parser.add_argument("--terminal-output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    run_cli(
        action_view_path=arguments.action_view_pack,
        minimal_label_path=arguments.minimal_label_pack,
        agent_action_path=arguments.agent_action_pack,
        raw_action_path=arguments.raw_action_pack,
        hipporag_action_path=arguments.hipporag_action_pack,
        private_score_output=arguments.private_score_output,
        safe_receipt_output=arguments.safe_receipt_output,
        terminal_output=arguments.terminal_output,
    )
    return 0


__all__ = [
    "ACTION_PACK_SCHEMA",
    "ACTION_VIEW_PACK_SCHEMA",
    "ARMS",
    "BLOCK",
    "FAMILY_COUNT",
    "ITEM_COUNT",
    "LABEL_RELEASE_POLICY",
    "MINIMAL_LABEL_PACK_SCHEMA",
    "PRIVATE_SCORE_PACK_SCHEMA",
    "SAFE_AGGREGATE_SCHEMA",
    "SAFE_TERMINAL_SCHEMA",
    "STUDY_ID",
    "ScoringArtifacts",
    "VERSION",
    "WikiSQLUAOScorerError",
    "build_minimal_label_pack",
    "canonical_json_bytes",
    "canonical_sha256",
    "main",
    "run_cli",
    "score_late_labels",
]


if __name__ == "__main__":  # pragma: no cover - exercised by formal service.
    raise SystemExit(main())
