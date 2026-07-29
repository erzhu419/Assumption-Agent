"""Offline action runtime for the frozen WikiSQL UAO reality study.

Three canonical pack languages form the complete data boundary:

* a view item contains exactly the opaque ID, question, table schema, and
  physical rows;
* an A_form label item contains only the frozen training and join fields; and
* an action item contains exactly ``opaque_item_id/top5_row_ids``.

RAW sees one view pack and applies the pure reality-core BM25 implementation.
Agent formation sees the A_form view and labels plus the A_hold *view only*.
It batches label-free question/row serializations through one injected local
encoder, forms TRAIN ``TrainingItem`` objects, compiles the fixed UAO policy,
and applies that policy to A_hold.  All three action arms use one exact pack
language bound to the shared view-pack commitment.  There is no held-out
label, family, query program, answer, network, API, retry, replay, or
online-evaluator input.
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
from typing import Any, Mapping, Protocol, Sequence

from assumption_agent.benchmarks import wikisql_uao_policy_v1 as policy
from assumption_agent.benchmarks import wikisql_uao_reality_v1 as reality
from assumption_agent.benchmarks import (
    wikisql_uao_source_compiler_v1 as source_compiler,
)


VERSION = "wikisql_uao_action_runtime_v1"
STUDY_ID = source_compiler.STUDY_ID
VIEW_PACK_SCHEMA = (
    f"{source_compiler.VERSION}_private_action_view_pack_v1"
)
LABEL_PACK_SCHEMA = f"{source_compiler.VERSION}_private_label_pack_v1"
ACTION_PACK_SCHEMA = f"{VERSION}_private_action_pack_v1"
AGENT_RECEIPT_SCHEMA = f"{VERSION}_agent_safe_receipt"
EMBEDDING_RECEIPT_SCHEMA = f"{VERSION}_embedding_safe_receipt"

BLOCKS = ("A_form", "A_hold")
ARMS = ("Agent", "RAW", "HippoRAG")
FORMAL_BLOCK_COUNTS = {"A_form": 192, "A_hold": 72}
FORMAL_FORM_FAMILY_COUNTS = {"EQ": 64, "GT": 64, "LT": 64}
ENCODER_BATCH_SIZE = 128
MAX_QUESTION_CHARACTERS = source_compiler.MAX_QUESTION_CHARACTERS
MAX_HEADER_CHARACTERS = source_compiler.MAX_HEADER_OR_CELL_CHARACTERS
MAX_CELL_CHARACTERS = source_compiler.MAX_HEADER_OR_CELL_CHARACTERS
MAX_COLUMNS = source_compiler.MAX_COLUMNS
MAX_SERIALIZED_ROW_CHARACTERS = reality.MAX_SERIALIZED_ROW_CHARACTERS
MAX_ITEM_COUNT = 1_024

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_DEVICE = re.compile(r"(?:cpu|cuda(?::(?:0|[1-9][0-9]*))?)\Z")
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
_LABEL_PACK_KEYS = frozenset(
    {
        "block",
        "item_count",
        "items",
        "release_policy",
        "schema",
        "self_sha256",
        "study_id",
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
_VIEW_ITEM_KEYS = frozenset(
    {
        "opaque_item_id",
        "physical_rows",
        "question",
        "table_header",
        "table_types",
    }
)
_A_FORM_LABEL_ITEM_KEYS = frozenset(
    {
        "action_view_sha256",
        "family",
        "fold_index",
        "gold_row_ids",
        "item_commitment_sha256",
        "opaque_item_id",
        "sqlite_rowid_cross_checked",
        "table_row_count",
    }
)
_ACTION_ITEM_KEYS = frozenset(
    {"opaque_item_id", "top5_row_ids"}
)
_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
}


class WikiSQLUAOActionRuntimeError(RuntimeError):
    """A pack, embedding, action, or local runtime drifted."""


Cell = str | int | float


@dataclass(frozen=True, slots=True)
class ViewItem:
    item_id: str
    question: str
    header: tuple[str, ...]
    types: tuple[str, ...]
    rows: tuple[tuple[Cell, ...], ...]

    @property
    def table(self) -> reality.WikiSQLTable:
        return reality.WikiSQLTable(
            table_id=self.item_id,
            header=self.header,
            types=self.types,
            rows=self.rows,
        )


@dataclass(frozen=True, slots=True)
class LabelItem:
    item_id: str
    family: str
    gold_row_ids: tuple[int, ...]
    action_view_sha256: str
    table_row_count: int
    fold_index: int


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    by_item_id: Mapping[str, policy.PrecomputedEmbeddings]
    safe_receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class AgentRunArtifacts:
    action_pack: Mapping[str, object]
    compiled_policy_private: Mapping[str, object]
    safe_receipt: Mapping[str, object]


class OfflineEncoder(Protocol):
    model_sha256: str

    def encode(
        self, texts: Sequence[str], *, batch_size: int
    ) -> Sequence[Sequence[float]]: ...


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
        raise WikiSQLUAOActionRuntimeError(
            "value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _content_addressed(value: Mapping[str, object]) -> dict[str, object]:
    if "self_sha256" in value or "pack_sha256" in value:
        raise WikiSQLUAOActionRuntimeError(
            "content-addressed payload already has a self field"
        )
    return {**value, "self_sha256": canonical_sha256(value)}


def _pack(base: Mapping[str, object]) -> dict[str, object]:
    return _content_addressed(base)


def _sequence(value: object, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise WikiSQLUAOActionRuntimeError(f"{field} must be an array")
    return value


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise WikiSQLUAOActionRuntimeError(f"{field} must be an object")
    return value


def _text(
    value: object,
    field: str,
    *,
    maximum: int,
    allow_empty: bool = False,
) -> str:
    if (
        not isinstance(value, str)
        or "\x00" in value
        or len(value) > maximum
        or (not allow_empty and not value.strip())
    ):
        raise WikiSQLUAOActionRuntimeError(f"{field} is invalid")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise WikiSQLUAOActionRuntimeError(
            f"{field} is not a SHA-256 commitment"
        )
    return value


def _block(value: object) -> str:
    if not isinstance(value, str) or value not in BLOCKS:
        raise WikiSQLUAOActionRuntimeError("block is invalid")
    return value


def _arm(value: object) -> str:
    if not isinstance(value, str) or value not in ARMS:
        raise WikiSQLUAOActionRuntimeError("action arm is invalid")
    return value


def _cell(value: object, field: str) -> Cell:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise WikiSQLUAOActionRuntimeError(
            f"{field} is not a documented scalar cell"
        )
    if isinstance(value, str):
        _text(
            value,
            field,
            maximum=MAX_CELL_CHARACTERS,
            allow_empty=True,
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise WikiSQLUAOActionRuntimeError(f"{field} is non-finite")
    return value


def _view_item_payload(item: ViewItem) -> dict[str, object]:
    return {
        "opaque_item_id": item.item_id,
        "physical_rows": [list(row) for row in item.rows],
        "question": item.question,
        "table_header": list(item.header),
        "table_types": list(item.types),
    }


def _decode_view_item(value: object, ordinal: int) -> ViewItem:
    row = _mapping(value, f"view item[{ordinal}]")
    if set(row) != _VIEW_ITEM_KEYS:
        raise WikiSQLUAOActionRuntimeError(
            f"view item[{ordinal}] fields drifted"
        )
    item_id = _sha256(
        row.get("opaque_item_id"), f"view item[{ordinal}] ID"
    )
    question = _text(
        row.get("question"),
        f"view item[{ordinal}] question",
        maximum=MAX_QUESTION_CHARACTERS,
    )
    raw_header = _sequence(
        row.get("table_header"), f"view item[{ordinal}] header"
    )
    if not 1 <= len(raw_header) <= MAX_COLUMNS:
        raise WikiSQLUAOActionRuntimeError(
            f"view item[{ordinal}] column count is outside 1..64"
        )
    header = tuple(
        _text(
            value,
            f"view item[{ordinal}] header[{index}]",
            maximum=MAX_HEADER_CHARACTERS,
        )
        for index, value in enumerate(raw_header)
    )
    raw_types = _sequence(
        row.get("table_types"), f"view item[{ordinal}] types"
    )
    types = tuple(
        _text(
            value,
            f"view item[{ordinal}] types[{index}]",
            maximum=16,
        )
        for index, value in enumerate(raw_types)
    )
    raw_rows = _sequence(
        row.get("physical_rows"), f"view item[{ordinal}] rows"
    )
    rows = tuple(
        tuple(
            _cell(
                cell,
                f"view item[{ordinal}] rows[{row_index}][{column_index}]",
            )
            for column_index, cell in enumerate(
                _sequence(
                    raw_row,
                    f"view item[{ordinal}] row[{row_index}]",
                )
            )
        )
        for row_index, raw_row in enumerate(raw_rows)
    )
    try:
        table = reality.WikiSQLTable(
            table_id=item_id,
            header=header,
            types=types,
            rows=rows,
        )
    except reality.WikiSQLUAORealityError as exc:
        raise WikiSQLUAOActionRuntimeError(
            f"view item[{ordinal}] table drifted"
        ) from exc
    if not reality.MIN_TABLE_ROWS <= len(table.rows) <= reality.MAX_TABLE_ROWS:
        raise WikiSQLUAOActionRuntimeError(
            f"view item[{ordinal}] row count is outside 11..80"
        )
    try:
        reality.validated_retrieval_documents(table)
    except reality.WikiSQLUAORealityError as exc:
        raise WikiSQLUAOActionRuntimeError(
            f"view item[{ordinal}] is outside the shared row-document contract"
        ) from exc
    return ViewItem(
        item_id=item_id,
        question=question,
        header=table.header,
        types=table.types,
        rows=table.rows,
    )


def _label_item_payload(item: LabelItem) -> dict[str, object]:
    """Return only the four A_form fields authorized past pack validation."""

    return {
        "action_view_sha256": item.action_view_sha256,
        "family": item.family,
        "fold_index": item.fold_index,
        "gold_row_ids": list(item.gold_row_ids),
        "opaque_item_id": item.item_id,
        "table_row_count": item.table_row_count,
    }


def _decode_label_item(
    value: object, ordinal: int, *, block: str
) -> LabelItem:
    if block != "A_form":
        raise WikiSQLUAOActionRuntimeError(
            "action runtime cannot construct or read A_hold labels"
        )
    row = _mapping(value, f"label item[{ordinal}]")
    if set(row) != _A_FORM_LABEL_ITEM_KEYS:
        raise WikiSQLUAOActionRuntimeError(
            f"label item[{ordinal}] fields drifted"
        )
    item_id = _sha256(
        row.get("opaque_item_id"), f"label item[{ordinal}] ID"
    )
    action_view_sha256 = _sha256(
        row.get("action_view_sha256"),
        f"label item[{ordinal}] action view",
    )
    _sha256(
        row.get("item_commitment_sha256"),
        f"label item[{ordinal}] source item",
    )
    family = row.get("family")
    if not isinstance(family, str) or family not in policy.FAMILY_ORDER:
        raise WikiSQLUAOActionRuntimeError(
            f"label item[{ordinal}] family drifted"
        )
    raw_gold = _sequence(
        row.get("gold_row_ids"),
        f"label item[{ordinal}] gold rows",
    )
    gold = tuple(raw_gold)
    if (
        not 1 <= len(gold) <= policy.TOP_K
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < reality.MAX_TABLE_ROWS
            for value in gold
        )
        or tuple(sorted(set(gold))) != gold
    ):
        raise WikiSQLUAOActionRuntimeError(
            f"label item[{ordinal}] gold rows drifted"
        )
    if (
        row.get("sqlite_rowid_cross_checked") is not True
        or isinstance(row.get("table_row_count"), bool)
        or not isinstance(row.get("table_row_count"), int)
        or not reality.MIN_TABLE_ROWS
        <= row["table_row_count"]
        <= reality.MAX_TABLE_ROWS
    ):
        raise WikiSQLUAOActionRuntimeError(
            f"label item[{ordinal}] binding metadata drifted"
        )
    raw_fold = row.get("fold_index")
    if (
        isinstance(raw_fold, bool)
        or not isinstance(raw_fold, int)
        or not 0 <= raw_fold < source_compiler.FOLD_COUNT
    ):
        raise WikiSQLUAOActionRuntimeError(
            f"label item[{ordinal}] fold drifted"
        )
    return LabelItem(
        item_id=item_id,
        family=family,
        gold_row_ids=gold,  # type: ignore[arg-type]
        action_view_sha256=action_view_sha256,
        table_row_count=row["table_row_count"],  # type: ignore[arg-type]
        fold_index=raw_fold,
    )


def _verify_pack_hash(
    value: Mapping[str, object],
    *,
    exact_keys: frozenset[str],
    field: str,
) -> dict[str, object]:
    if set(value) != exact_keys:
        raise WikiSQLUAOActionRuntimeError(f"{field} fields drifted")
    supplied = _sha256(value.get("self_sha256"), f"{field} hash")
    base = {
        key: child
        for key, child in value.items()
        if key != "self_sha256"
    }
    if canonical_sha256(base) != supplied:
        raise WikiSQLUAOActionRuntimeError(
            f"{field} content hash drifted"
        )
    return base


def build_view_pack(
    *, block: str, items: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    checked_block = _block(block)
    if (
        isinstance(items, (str, bytes, bytearray))
        or not isinstance(items, Sequence)
        or not 1 <= len(items) <= MAX_ITEM_COUNT
    ):
        raise WikiSQLUAOActionRuntimeError(
            "view pack item count drifted"
        )
    checked = tuple(
        _decode_view_item(item, ordinal)
        for ordinal, item in enumerate(items)
    )
    if len({item.item_id for item in checked}) != len(checked):
        raise WikiSQLUAOActionRuntimeError(
            "view pack repeats an item ID"
        )
    checked = tuple(sorted(checked, key=lambda item: item.item_id))
    return _pack(
        {
            "block": checked_block,
            "contains_labels": False,
            "item_count": len(checked),
            "items": [_view_item_payload(item) for item in checked],
            "schema": VIEW_PACK_SCHEMA,
            "study_id": STUDY_ID,
        }
    )


def decode_view_pack(
    value: Mapping[str, object],
    *,
    expected_block: str | None = None,
    expected_count: int | None = None,
) -> tuple[ViewItem, ...]:
    _verify_pack_hash(
        value,
        exact_keys=_VIEW_PACK_KEYS,
        field="view pack",
    )
    if value.get("schema") != VIEW_PACK_SCHEMA:
        raise WikiSQLUAOActionRuntimeError(
            "view pack schema drifted"
        )
    checked_block = _block(value.get("block"))
    if expected_block is not None and checked_block != _block(expected_block):
        raise WikiSQLUAOActionRuntimeError(
            "view pack block drifted"
        )
    raw_items = _sequence(value.get("items"), "view pack items")
    if (
        value.get("study_id") != STUDY_ID
        or value.get("contains_labels") is not False
        or value.get("item_count") != len(raw_items)
        or isinstance(value.get("item_count"), bool)
        or not 1 <= len(raw_items) <= MAX_ITEM_COUNT
        or (
            expected_count is not None
            and len(raw_items) != expected_count
        )
    ):
        raise WikiSQLUAOActionRuntimeError(
            "view pack item count drifted"
        )
    items = tuple(
        _decode_view_item(item, ordinal)
        for ordinal, item in enumerate(raw_items)
    )
    if len({item.item_id for item in items}) != len(items):
        raise WikiSQLUAOActionRuntimeError(
            "view pack repeats an item ID"
        )
    if tuple(item.item_id for item in items) != tuple(
        sorted(item.item_id for item in items)
    ):
        raise WikiSQLUAOActionRuntimeError(
            "view pack opaque item order drifted"
        )
    return items


def build_label_pack(
    *, block: str, items: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    checked_block = _block(block)
    if checked_block != "A_form":
        raise WikiSQLUAOActionRuntimeError(
            "action runtime cannot construct or read A_hold labels"
        )
    if (
        isinstance(items, (str, bytes, bytearray))
        or not isinstance(items, Sequence)
        or not 1 <= len(items) <= MAX_ITEM_COUNT
    ):
        raise WikiSQLUAOActionRuntimeError(
            "label pack item count drifted"
        )
    raw_rows = tuple(
        dict(_mapping(item, f"label item[{ordinal}]"))
        for ordinal, item in enumerate(items)
    )
    checked = tuple(
        _decode_label_item(item, ordinal, block=checked_block)
        for ordinal, item in enumerate(raw_rows)
    )
    if len({item.item_id for item in checked}) != len(checked):
        raise WikiSQLUAOActionRuntimeError(
            "label pack repeats an item ID"
        )
    paired = sorted(
        zip(checked, raw_rows, strict=True),
        key=lambda pair: pair[0].item_id,
    )
    return _pack(
        {
            "block": checked_block,
            "item_count": len(paired),
            "items": [row for _item, row in paired],
            "release_policy": (
                "A_form_train_only"
                if checked_block == "A_form"
                else "after_all_A_hold_three_arm_actions_are_sealed"
            ),
            "schema": LABEL_PACK_SCHEMA,
            "study_id": STUDY_ID,
        }
    )


def decode_label_pack(
    value: Mapping[str, object],
    *,
    expected_block: str | None = None,
    expected_count: int | None = None,
) -> tuple[LabelItem, ...]:
    _verify_pack_hash(
        value,
        exact_keys=_LABEL_PACK_KEYS,
        field="label pack",
    )
    if value.get("schema") != LABEL_PACK_SCHEMA:
        raise WikiSQLUAOActionRuntimeError(
            "label pack schema drifted"
        )
    checked_block = _block(value.get("block"))
    if checked_block != "A_form":
        raise WikiSQLUAOActionRuntimeError(
            "action runtime cannot construct or read A_hold labels"
        )
    if expected_block is not None and checked_block != _block(expected_block):
        raise WikiSQLUAOActionRuntimeError(
            "label pack block drifted"
        )
    raw_items = _sequence(value.get("items"), "label pack items")
    expected_release = (
        "A_form_train_only"
        if checked_block == "A_form"
        else "after_all_A_hold_three_arm_actions_are_sealed"
    )
    if (
        value.get("study_id") != STUDY_ID
        or value.get("release_policy") != expected_release
        or value.get("item_count") != len(raw_items)
        or isinstance(value.get("item_count"), bool)
        or not 1 <= len(raw_items) <= MAX_ITEM_COUNT
        or (
            expected_count is not None
            and len(raw_items) != expected_count
        )
    ):
        raise WikiSQLUAOActionRuntimeError(
            "label pack item count drifted"
        )
    items = tuple(
        _decode_label_item(
            item, ordinal, block=checked_block
        )
        for ordinal, item in enumerate(raw_items)
    )
    if len({item.item_id for item in items}) != len(items):
        raise WikiSQLUAOActionRuntimeError(
            "label pack repeats an item ID"
        )
    if tuple(item.item_id for item in items) != tuple(
        sorted(item.item_id for item in items)
    ):
        raise WikiSQLUAOActionRuntimeError(
            "label pack opaque item order drifted"
        )
    return items


def _action_item(
    *,
    item_id: str,
    top5: Sequence[int],
    row_count: int,
) -> dict[str, object]:
    checked_id = _sha256(item_id, "action item ID")
    values = tuple(top5)
    if (
        len(values) != policy.TOP_K
        or len(set(values)) != policy.TOP_K
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < row_count
            for value in values
        )
    ):
        raise WikiSQLUAOActionRuntimeError(
            "action top five drifted"
        )
    return {
        "opaque_item_id": checked_id,
        "top5_row_ids": list(values),
    }


def build_action_pack(
    *,
    block: str,
    arm: str,
    action_view_pack_sha256: str,
    items: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    checked_block = _block(block)
    checked_arm = _arm(arm)
    checked_view_sha256 = _sha256(
        action_view_pack_sha256,
        "action view pack",
    )
    if (
        isinstance(items, (str, bytes, bytearray))
        or not isinstance(items, Sequence)
        or not 1 <= len(items) <= MAX_ITEM_COUNT
    ):
        raise WikiSQLUAOActionRuntimeError(
            "action pack item count drifted"
        )
    checked_items: list[dict[str, object]] = []
    identifiers: set[str] = set()
    for ordinal, raw in enumerate(items):
        row = _mapping(raw, f"action item[{ordinal}]")
        if set(row) != _ACTION_ITEM_KEYS:
            raise WikiSQLUAOActionRuntimeError(
                f"action item[{ordinal}] fields drifted"
            )
        item_id = _sha256(
            row.get("opaque_item_id"),
            f"action item[{ordinal}] ID",
        )
        raw_top5 = _sequence(
            row.get("top5_row_ids"),
            f"action item[{ordinal}] top five",
        )
        # Action pack construction can be used independently of the view, so
        # the frozen global maximum is the only available upper bound.
        checked = _action_item(
            item_id=item_id,
            top5=raw_top5,  # type: ignore[arg-type]
            row_count=reality.MAX_TABLE_ROWS,
        )
        if item_id in identifiers:
            raise WikiSQLUAOActionRuntimeError(
                "action pack repeats an item ID"
            )
        identifiers.add(item_id)
        checked_items.append(checked)
    checked_items.sort(key=lambda row: row["opaque_item_id"])
    return _pack(
        {
            "action_view_pack_sha256": checked_view_sha256,
            "arm": checked_arm,
            "block": checked_block,
            "item_count": len(checked_items),
            "items": checked_items,
            "schema": ACTION_PACK_SCHEMA,
            "study_id": STUDY_ID,
        }
    )


def decode_action_pack(
    value: Mapping[str, object],
    *,
    expected_block: str | None = None,
    expected_arm: str | None = None,
    expected_action_view_pack_sha256: str | None = None,
) -> tuple[Mapping[str, object], ...]:
    _verify_pack_hash(
        value,
        exact_keys=_ACTION_PACK_KEYS,
        field="action pack",
    )
    if value.get("schema") != ACTION_PACK_SCHEMA:
        raise WikiSQLUAOActionRuntimeError(
            "action pack schema drifted"
        )
    checked_block = _block(value.get("block"))
    checked_arm = _arm(value.get("arm"))
    checked_view_sha256 = _sha256(
        value.get("action_view_pack_sha256"),
        "action view pack",
    )
    if (
        value.get("study_id") != STUDY_ID
        or isinstance(value.get("item_count"), bool)
    ):
        raise WikiSQLUAOActionRuntimeError(
            "action pack study or count drifted"
        )
    if expected_block is not None and checked_block != _block(expected_block):
        raise WikiSQLUAOActionRuntimeError(
            "action pack block drifted"
        )
    if expected_arm is not None and checked_arm != _arm(expected_arm):
        raise WikiSQLUAOActionRuntimeError(
            "action pack arm drifted"
        )
    if (
        expected_action_view_pack_sha256 is not None
        and checked_view_sha256
        != _sha256(
            expected_action_view_pack_sha256,
            "expected action view pack",
        )
    ):
        raise WikiSQLUAOActionRuntimeError(
            "action view pack commitment drifted"
        )
    rows = _sequence(value.get("items"), "action pack items")
    if value.get("item_count") != len(rows):
        raise WikiSQLUAOActionRuntimeError(
            "action pack item count drifted"
        )
    rebuilt = build_action_pack(
        block=checked_block,
        arm=checked_arm,
        action_view_pack_sha256=checked_view_sha256,
        items=rows,  # type: ignore[arg-type]
    )
    if rebuilt != dict(value):
        raise WikiSQLUAOActionRuntimeError(
            "action pack reconstruction drifted"
        )
    return tuple(_mapping(row, "action item") for row in rows)


def _raw_top5(item: ViewItem) -> tuple[int, ...]:
    result = reality.raw_bm25_top5(item.question, item.table)
    if any(value is None for value in result):
        raise WikiSQLUAOActionRuntimeError(
            "eligible RAW action unexpectedly contains padding"
        )
    return result  # type: ignore[return-value]


def run_raw(
    *, view_pack: Mapping[str, object]
) -> dict[str, object]:
    """Apply the frozen reality-core BM25 action to one view pack."""

    items = decode_view_pack(view_pack)
    block = _block(view_pack.get("block"))
    action_items = [
        _action_item(
            item_id=item.item_id,
            top5=_raw_top5(item),
            row_count=len(item.rows),
        )
        for item in items
    ]
    return build_action_pack(
        block=block,
        arm="RAW",
        action_view_pack_sha256=_sha256(
            view_pack.get("self_sha256"),
            "RAW action view pack",
        ),
        items=action_items,
    )


def _finite_matrix(
    value: object, *, expected_rows: int
) -> tuple[tuple[float, ...], ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()  # type: ignore[union-attr]
    rows = _sequence(value, "encoder matrix")
    if len(rows) != expected_rows:
        raise WikiSQLUAOActionRuntimeError(
            "encoder row count drifted"
        )
    matrix: list[tuple[float, ...]] = []
    width: int | None = None
    for row_index, raw_row in enumerate(rows):
        values = _sequence(raw_row, f"encoder row[{row_index}]")
        converted: list[float] = []
        for column_index, raw_value in enumerate(values):
            if isinstance(raw_value, bool) or not isinstance(
                raw_value, (int, float)
            ):
                raise WikiSQLUAOActionRuntimeError(
                    f"encoder row[{row_index}][{column_index}] is not numeric"
                )
            numeric = float(raw_value)
            if not math.isfinite(numeric):
                raise WikiSQLUAOActionRuntimeError(
                    "encoder matrix is non-finite"
                )
            converted.append(numeric)
        if not converted:
            raise WikiSQLUAOActionRuntimeError(
                "encoder vector is empty"
            )
        if width is None:
            width = len(converted)
        elif len(converted) != width:
            raise WikiSQLUAOActionRuntimeError(
                "encoder width drifted"
            )
        matrix.append(tuple(converted))
    return tuple(matrix)


def encode_view_items(
    *,
    a_form_items: Sequence[ViewItem],
    a_hold_items: Sequence[ViewItem],
    encoder: OfflineEncoder,
    batch_size: int = ENCODER_BATCH_SIZE,
) -> EmbeddingBatch:
    """Bulk-encode questions and serialized rows with exact alignment."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise WikiSQLUAOActionRuntimeError(
            "encoder batch size drifted"
        )
    all_items = tuple(a_form_items) + tuple(a_hold_items)
    if (
        not all_items
        or any(not isinstance(item, ViewItem) for item in all_items)
        or len({item.item_id for item in all_items}) != len(all_items)
    ):
        raise WikiSQLUAOActionRuntimeError(
            "embedding item set is empty, malformed, or overlapping"
        )
    model_sha256 = _sha256(
        getattr(encoder, "model_sha256", None),
        "embedding model",
    )
    requests: list[str] = []
    row_counts: list[int] = []
    for item in all_items:
        serialized = reality.serialize_table_rows(item.table)
        requests.append(item.question)
        requests.extend(serialized)
        row_counts.append(len(serialized))
    try:
        encoded = encoder.encode(
            tuple(requests), batch_size=batch_size
        )
    except BaseException as exc:
        raise WikiSQLUAOActionRuntimeError(
            "offline encoder failed without retry"
        ) from exc
    matrix = _finite_matrix(encoded, expected_rows=len(requests))
    by_item: dict[str, policy.PrecomputedEmbeddings] = {}
    cursor = 0
    for item, row_count in zip(all_items, row_counts, strict=True):
        question_vector = matrix[cursor]
        cursor += 1
        row_vectors = matrix[cursor : cursor + row_count]
        cursor += row_count
        try:
            by_item[item.item_id] = policy.PrecomputedEmbeddings(
                model_sha256=model_sha256,
                question=question_vector,
                rows=row_vectors,
            )
        except policy.WikiSQLUAOPolicyError as exc:
            raise WikiSQLUAOActionRuntimeError(
                "aligned embedding bundle drifted"
            ) from exc
    if cursor != len(matrix) or len(by_item) != len(all_items):
        raise WikiSQLUAOActionRuntimeError(
            "embedding alignment did not consume the exact matrix"
        )
    request_hashes = [
        hashlib.sha256(text.encode("utf-8")).hexdigest()
        for text in requests
    ]
    receipt = _content_addressed(
        {
            "a_form_item_count": len(a_form_items),
            "a_hold_item_count": len(a_hold_items),
            "api_call_count": 0,
            "batch_size": batch_size,
            "dimension": len(matrix[0]),
            "encoder_call_count": 1,
            "matrix_sha256": canonical_sha256(
                [list(row) for row in matrix]
            ),
            "model_sha256": model_sha256,
            "network_call_count": 0,
            "request_count": len(requests),
            "request_set_sha256": canonical_sha256(request_hashes),
            "retry_count": 0,
            "schema": EMBEDDING_RECEIPT_SCHEMA,
        }
    )
    return EmbeddingBatch(
        by_item_id=by_item,
        safe_receipt=receipt,
    )


def build_training_items(
    *,
    a_form_items: Sequence[ViewItem],
    labels: Sequence[LabelItem],
    embeddings: Mapping[str, policy.PrecomputedEmbeddings],
) -> tuple[policy.TrainingItem, ...]:
    """Join A_form labels to their exact views; no A_hold input is accepted."""

    views = tuple(a_form_items)
    checked_labels = tuple(labels)
    _validate_a_form_bindings(views, checked_labels)
    result: list[policy.TrainingItem] = []
    for item, label in zip(views, checked_labels, strict=True):
        serialized = reality.serialize_table_rows(item.table)
        raw_top5 = reality.raw_bm25_top5(
            item.question, item.table
        )
        try:
            result.append(
                policy.TrainingItem(
                    item=policy.LabelFreeItem(
                        question=item.question,
                        headers=item.header,
                        types=item.types,
                        serialized_rows=serialized,
                        raw_top5=raw_top5,
                        embeddings=embeddings[item.item_id],
                    ),
                    gold_row_ids=label.gold_row_ids,
                    family=label.family,
                    fold_index=label.fold_index,
                )
            )
        except (KeyError, policy.WikiSQLUAOPolicyError) as exc:
            raise WikiSQLUAOActionRuntimeError(
                "A_form policy item construction drifted"
            ) from exc
    return tuple(result)


def _validate_a_form_bindings(
    views: Sequence[ViewItem],
    labels: Sequence[LabelItem],
) -> None:
    """Validate all private A_form joins before any model execution."""

    if (
        not views
        or len(views) != len(labels)
        or tuple(item.item_id for item in views)
        != tuple(item.item_id for item in labels)
    ):
        raise WikiSQLUAOActionRuntimeError(
            "A_form view/label identity order drifted"
        )
    for item, label in zip(views, labels, strict=True):
        if (
            label.action_view_sha256
            != canonical_sha256(_view_item_payload(item))
            or label.table_row_count != len(item.rows)
            or any(
                value >= len(item.rows)
                for value in label.gold_row_ids
            )
        ):
            raise WikiSQLUAOActionRuntimeError(
                "A_form label/view binding drifted"
            )


def _apply_agent(
    *,
    item: ViewItem,
    compiled_policy: policy.CompiledPolicy,
    embedding: policy.PrecomputedEmbeddings,
) -> tuple[int, ...]:
    serialized = reality.serialize_table_rows(item.table)
    raw_top5 = reality.raw_bm25_top5(item.question, item.table)
    try:
        action = policy.apply_uao_policy(
            compiled_policy,
            question=item.question,
            headers=item.header,
            types=item.types,
            serialized_rows=serialized,
            raw_top5=raw_top5,
            embeddings=embedding,
        )
    except policy.WikiSQLUAOPolicyError as exc:
        raise WikiSQLUAOActionRuntimeError(
            "A_hold label-free policy application failed"
        ) from exc
    if any(value is None for value in action):
        raise WikiSQLUAOActionRuntimeError(
            "eligible Agent action unexpectedly contains padding"
        )
    return action  # type: ignore[return-value]


def run_agent(
    *,
    a_form_view_pack: Mapping[str, object],
    a_form_label_pack: Mapping[str, object],
    a_hold_view_pack: Mapping[str, object],
    encoder: OfflineEncoder,
) -> AgentRunArtifacts:
    """Compile on A_form and act on A_hold without a held-out label input."""

    a_form = decode_view_pack(
        a_form_view_pack, expected_block="A_form"
    )
    labels = decode_label_pack(
        a_form_label_pack, expected_block="A_form"
    )
    a_hold = decode_view_pack(
        a_hold_view_pack, expected_block="A_hold"
    )
    if {item.item_id for item in a_form}.intersection(
        item.item_id for item in a_hold
    ):
        raise WikiSQLUAOActionRuntimeError(
            "A_form and A_hold item IDs overlap"
        )
    _validate_a_form_bindings(a_form, labels)
    embedding_batch = encode_view_items(
        a_form_items=a_form,
        a_hold_items=a_hold,
        encoder=encoder,
    )
    train_items = build_training_items(
        a_form_items=a_form,
        labels=labels,
        embeddings=embedding_batch.by_item_id,
    )
    try:
        formation = policy.fit_uao_policy(train_items)
    except policy.WikiSQLUAOPolicyError as exc:
        raise WikiSQLUAOActionRuntimeError(
            "UAO policy formation failed without retry"
        ) from exc
    action_rows = [
        _action_item(
            item_id=item.item_id,
            top5=_apply_agent(
                item=item,
                compiled_policy=formation.policy,
                embedding=embedding_batch.by_item_id[item.item_id],
            ),
            row_count=len(item.rows),
        )
        for item in a_hold
    ]
    action_pack = build_action_pack(
        block="A_hold",
        arm="Agent",
        action_view_pack_sha256=_sha256(
            a_hold_view_pack.get("self_sha256"),
            "Agent action view pack",
        ),
        items=action_rows,
    )
    private_policy = formation.policy.content_addressed_private_payload()
    probe_receipts = [
        receipt.safe_receipt()
        for receipt in formation.probe_receipts
    ]
    safe_receipt = _content_addressed(
        {
            "a_form_label_pack_sha256": a_form_label_pack[
                "self_sha256"
            ],
            "a_form_view_pack_sha256": a_form_view_pack[
                "self_sha256"
            ],
            "a_hold_action_pack_sha256": action_pack[
                "self_sha256"
            ],
            "a_hold_label_access_count": 0,
            "a_hold_view_pack_sha256": a_hold_view_pack[
                "self_sha256"
            ],
            "api_call_count": 0,
            "compiled_policy_private_sha256": private_policy[
                "self_sha256"
            ],
            "claim_selection_receipt": (
                formation.claim_selection_receipt.safe_receipt()
            ),
            "embedding_receipt": dict(
                embedding_batch.safe_receipt
            ),
            "formation_receipt": formation.safe_receipt(),
            "network_call_count": 0,
            "no_op_calibration_receipt": (
                formation.no_op_calibration_receipt.safe_receipt()
            ),
            "online_evaluator_call_count": 0,
            "policy_receipt": formation.policy.safe_receipt(),
            "probe_receipts": probe_receipts,
            "replay_count": 0,
            "retry_count": 0,
            "schema": AGENT_RECEIPT_SCHEMA,
        }
    )
    return AgentRunArtifacts(
        action_pack=action_pack,
        compiled_policy_private=private_policy,
        safe_receipt=safe_receipt,
    )


def require_formal_raw_counts(
    view_pack: Mapping[str, object],
) -> None:
    block = _block(view_pack.get("block"))
    decode_view_pack(
        view_pack,
        expected_block=block,
        expected_count=FORMAL_BLOCK_COUNTS[block],
    )


def require_formal_agent_counts(
    *,
    a_form_view_pack: Mapping[str, object],
    a_form_label_pack: Mapping[str, object],
    a_hold_view_pack: Mapping[str, object],
) -> None:
    a_form = decode_view_pack(
        a_form_view_pack,
        expected_block="A_form",
        expected_count=FORMAL_BLOCK_COUNTS["A_form"],
    )
    labels = decode_label_pack(
        a_form_label_pack,
        expected_block="A_form",
        expected_count=FORMAL_BLOCK_COUNTS["A_form"],
    )
    decode_view_pack(
        a_hold_view_pack,
        expected_block="A_hold",
        expected_count=FORMAL_BLOCK_COUNTS["A_hold"],
    )
    if tuple(item.item_id for item in a_form) != tuple(
        item.item_id for item in labels
    ):
        raise WikiSQLUAOActionRuntimeError(
            "formal A_form view/label order drifted"
        )
    for item, label in zip(a_form, labels, strict=True):
        if (
            label.action_view_sha256
            != canonical_sha256(_view_item_payload(item))
            or label.table_row_count != len(item.rows)
            or any(
                row_id >= len(item.rows)
                for row_id in label.gold_row_ids
            )
        ):
            raise WikiSQLUAOActionRuntimeError(
                "formal A_form label/view binding drifted"
            )
    if Counter(item.family for item in labels) != Counter(
        FORMAL_FORM_FAMILY_COUNTS
    ):
        raise WikiSQLUAOActionRuntimeError(
            "formal A_form family quotas drifted"
        )
    expected_folds = Counter(
        {
            (family, fold_index): 16
            for family in policy.FAMILY_ORDER
            for fold_index in range(source_compiler.FOLD_COUNT)
        }
    )
    if Counter(
        (item.family, item.fold_index) for item in labels
    ) != expected_folds:
        raise WikiSQLUAOActionRuntimeError(
            "formal A_form fold/family quotas drifted"
        )


def directory_tree_sha256(root: Path) -> str:
    """Content-address one direct, symlink-free local model tree."""

    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise WikiSQLUAOActionRuntimeError(
            "local encoder model is unavailable"
        ) from exc
    if root.is_symlink() or not stat.S_ISDIR(root_metadata.st_mode):
        raise WikiSQLUAOActionRuntimeError(
            "local encoder model root drifted"
        )
    entries: list[dict[str, object]] = []
    for path in sorted(
        root.rglob("*"),
        key=lambda row: row.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise WikiSQLUAOActionRuntimeError(
                "local encoder model entry is unavailable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise WikiSQLUAOActionRuntimeError(
                "local encoder model contains a symlink"
            )
        if stat.S_ISDIR(metadata.st_mode):
            entries.append({"kind": "directory", "path": relative})
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise WikiSQLUAOActionRuntimeError(
                "local encoder model contains a special file"
            )
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                while True:
                    block = handle.read(1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
        except OSError as exc:
            raise WikiSQLUAOActionRuntimeError(
                "local encoder model could not be hashed"
            ) from exc
        entries.append(
            {
                "kind": "file",
                "path": relative,
                "sha256": digest.hexdigest(),
                "size": metadata.st_size,
            }
        )
    return canonical_sha256(entries)


class LocalSentenceTransformerEncoder:
    """Lazy, local-files-only, normalized SentenceTransformer encoder."""

    def __init__(
        self,
        *,
        model_root: Path,
        expected_model_sha256: str,
        device: str,
    ) -> None:
        self.model_root = model_root
        self.model_sha256 = _sha256(
            expected_model_sha256, "expected encoder model"
        )
        if not isinstance(device, str) or _DEVICE.fullmatch(device) is None:
            raise WikiSQLUAOActionRuntimeError(
                "encoder device drifted"
            )
        self.device = device
        observed = directory_tree_sha256(model_root)
        if observed != self.model_sha256:
            raise WikiSQLUAOActionRuntimeError(
                "local encoder model tree drifted"
            )
        self._model: object | None = None

    def _load(self) -> object:
        if any(
            os.environ.get(key) != expected
            for key, expected in _OFFLINE_ENVIRONMENT.items()
        ):
            raise WikiSQLUAOActionRuntimeError(
                "local encoder offline environment drifted"
            )
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    str(self.model_root),
                    device=self.device,
                    local_files_only=True,
                    trust_remote_code=False,
                )
            except BaseException as exc:
                raise WikiSQLUAOActionRuntimeError(
                    "local SentenceTransformer load failed"
                ) from exc
        return self._model

    def encode(
        self, texts: Sequence[str], *, batch_size: int
    ) -> Sequence[Sequence[float]]:
        if (
            batch_size != ENCODER_BATCH_SIZE
            or isinstance(texts, (str, bytes, bytearray))
            or not isinstance(texts, Sequence)
            or not texts
            or any(not isinstance(value, str) for value in texts)
        ):
            raise WikiSQLUAOActionRuntimeError(
                "local encoder request drifted"
            )
        try:
            matrix = self._load().encode(  # type: ignore[union-attr]
                list(texts),
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.device,
            )
        except BaseException as exc:
            raise WikiSQLUAOActionRuntimeError(
                "local SentenceTransformer inference failed without retry"
            ) from exc
        return matrix  # type: ignore[return-value]


def _read_canonical_pack(path: Path, field: str) -> dict[str, object]:
    try:
        metadata = path.lstat()
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WikiSQLUAOActionRuntimeError(
            f"{field} is unreadable"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or not isinstance(value, dict)
        or canonical_json_bytes(value, newline=True) != raw
    ):
        raise WikiSQLUAOActionRuntimeError(
            f"{field} metadata or canonical encoding drifted"
        )
    return value


def _write_exclusive(path: Path, value: Mapping[str, object]) -> str:
    raw = canonical_json_bytes(value, newline=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise WikiSQLUAOActionRuntimeError(
            "exclusive output creation failed"
        ) from exc
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise WikiSQLUAOActionRuntimeError(
                    "exclusive output write stalled"
                )
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


def _require_fresh_distinct_outputs(paths: Sequence[Path]) -> None:
    normalized = tuple(path.absolute() for path in paths)
    if (
        len(set(normalized)) != len(normalized)
        or any(path.exists() or path.is_symlink() for path in paths)
    ):
        raise WikiSQLUAOActionRuntimeError(
            "output paths are not fresh and distinct"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    raw = subparsers.add_parser("raw")
    raw.add_argument("--view", required=True, type=Path)
    raw.add_argument("--action-output", required=True, type=Path)

    agent = subparsers.add_parser("agent")
    agent.add_argument("--a-form-view", required=True, type=Path)
    agent.add_argument("--a-form-labels", required=True, type=Path)
    agent.add_argument("--a-hold-view", required=True, type=Path)
    agent.add_argument("--action-output", required=True, type=Path)
    agent.add_argument("--policy-output", required=True, type=Path)
    agent.add_argument("--receipt-output", required=True, type=Path)
    agent.add_argument("--encoder-model", required=True, type=Path)
    agent.add_argument("--encoder-model-sha256", required=True)
    agent.add_argument("--device", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.mode == "raw":
        _require_fresh_distinct_outputs((arguments.action_output,))
        view = _read_canonical_pack(arguments.view, "RAW view pack")
        require_formal_raw_counts(view)
        action_pack = run_raw(view_pack=view)
        file_sha256 = _write_exclusive(
            arguments.action_output, action_pack
        )
        terminal = {
            "action_file_sha256": file_sha256,
            "action_pack_sha256": action_pack["self_sha256"],
            "arm": "RAW",
            "block": view["block"],
            "item_count": len(action_pack["items"]),
            "status": "passed",
        }
    else:
        _require_fresh_distinct_outputs(
            (
                arguments.action_output,
                arguments.policy_output,
                arguments.receipt_output,
            )
        )
        a_form_view = _read_canonical_pack(
            arguments.a_form_view, "A_form view pack"
        )
        a_form_labels = _read_canonical_pack(
            arguments.a_form_labels, "A_form label pack"
        )
        a_hold_view = _read_canonical_pack(
            arguments.a_hold_view, "A_hold view pack"
        )
        require_formal_agent_counts(
            a_form_view_pack=a_form_view,
            a_form_label_pack=a_form_labels,
            a_hold_view_pack=a_hold_view,
        )
        encoder = LocalSentenceTransformerEncoder(
            model_root=arguments.encoder_model,
            expected_model_sha256=arguments.encoder_model_sha256,
            device=arguments.device,
        )
        artifacts = run_agent(
            a_form_view_pack=a_form_view,
            a_form_label_pack=a_form_labels,
            a_hold_view_pack=a_hold_view,
            encoder=encoder,
        )
        action_file_sha256 = _write_exclusive(
            arguments.action_output, artifacts.action_pack
        )
        policy_file_sha256 = _write_exclusive(
            arguments.policy_output,
            artifacts.compiled_policy_private,
        )
        receipt_file_sha256 = _write_exclusive(
            arguments.receipt_output, artifacts.safe_receipt
        )
        terminal = {
            "action_file_sha256": action_file_sha256,
            "action_pack_sha256": artifacts.action_pack[
                "self_sha256"
            ],
            "arm": "Agent",
            "block": "A_hold",
            "item_count": len(artifacts.action_pack["items"]),
            "policy_file_sha256": policy_file_sha256,
            "policy_sha256": artifacts.compiled_policy_private[
                "self_sha256"
            ],
            "receipt_file_sha256": receipt_file_sha256,
            "receipt_sha256": artifacts.safe_receipt[
                "self_sha256"
            ],
            "status": "passed",
        }
    print(
        json.dumps(
            terminal,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTION_PACK_SCHEMA",
    "AGENT_RECEIPT_SCHEMA",
    "ARMS",
    "AgentRunArtifacts",
    "BLOCKS",
    "EMBEDDING_RECEIPT_SCHEMA",
    "ENCODER_BATCH_SIZE",
    "EmbeddingBatch",
    "FORMAL_BLOCK_COUNTS",
    "FORMAL_FORM_FAMILY_COUNTS",
    "LABEL_PACK_SCHEMA",
    "LabelItem",
    "LocalSentenceTransformerEncoder",
    "OfflineEncoder",
    "VERSION",
    "VIEW_PACK_SCHEMA",
    "ViewItem",
    "WikiSQLUAOActionRuntimeError",
    "build_action_pack",
    "build_label_pack",
    "build_training_items",
    "build_view_pack",
    "canonical_json_bytes",
    "canonical_sha256",
    "decode_action_pack",
    "decode_label_pack",
    "decode_view_pack",
    "directory_tree_sha256",
    "encode_view_items",
    "main",
    "require_formal_agent_counts",
    "require_formal_raw_counts",
    "run_agent",
    "run_raw",
]
