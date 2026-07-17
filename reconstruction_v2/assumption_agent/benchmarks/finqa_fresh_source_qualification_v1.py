"""Aggregate-only source qualification for the frozen official FinQA source.

The public entry point runs an isolated child process.  The child opens only
the exact official TRAIN and DEV archive members, validates their public
schema, and emits aggregate counts and byte commitments.  It never selects a
row, reads a selection secret, scores a retrieval result, opens TEST or
PRIVATE_TEST, or emits a question/report identifier or source content.

The parser intentionally follows the repaired official FinQA implementation:
``text_i`` indexes ``pre_text + post_text``; ``table_i`` directly indexes the
table including the ``table_0`` header candidate; and table rows are rendered
with the lower-case ``the ...`` template plus literal-ASCII-space collapse.
Ragged rows retain the official ``zip`` truncation semantics.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from typing import Any
import unicodedata


VERSION = "finqa_fresh_source_qualification_v1"
SCHEMA = VERSION
SOURCE_RELEASE = "FinQA_official_commit_0f16e286_train_dev_only"
QUALIFICATION_CLASS = "outcome_blind_aggregate_only_source_qualification"

OFFICIAL_COMMIT = "0f16e2867befa6840783e58be38c9efb9229d742"
ARCHIVE_ROOT = f"FinQA-{OFFICIAL_COMMIT}"
EXPECTED_MEMBERS = {
    "train": f"{ARCHIVE_ROOT}/dataset/train.json",
    "dev": f"{ARCHIVE_ROOT}/dataset/dev.json",
}
FORBIDDEN_DATA_MEMBERS = frozenset(
    {
        f"{ARCHIVE_ROOT}/dataset/test.json",
        f"{ARCHIVE_ROOT}/dataset/private_test.json",
        f"{ARCHIVE_ROOT}/code/evaluate/test.json",
    }
)

FORMAL_ARCHIVE_SHA256 = (
    "eec31eec72c4258ba80fa4575c1adc6cdca0ba8aff787f11ce58fab514f93876"
)
FORMAL_ARCHIVE_SIZE = 21_204_944
FORMAL_CUSTODY_MANIFEST_FILE_SHA256 = (
    "2ab949afad3ff17f8c4b130eebb1691dc855fd2216b711bf55bb6511af28c1a3"
)
FORMAL_CUSTODY_SHA256 = (
    "a35c48d2a81b785a44ff06d4b09013f6d0cfc13ee94008655c5e8512376bb0a0"
)
FORMAL_CUSTODY_COMMIT = "6321cfe2"
FORMAL_CUSTODY_SCHEMA = "finqa_graph_evaluator_source_custody_v1"

MIN_GOLD_UNITS = 2
MAX_GOLD_UNITS = 5
MIN_ADDRESSABLE_NODES = 5
MAX_ADDRESSABLE_NODES = 128
FORMAL_MINIMUM_DISTINCT_ELIGIBLE_REPORTS = {"train": 192, "dev": 64}
MAX_ARCHIVE_MEMBERS = 100_000
MAX_SPLIT_MEMBER_BYTES = 512 * 1024 * 1024

OFFICIAL_EXAMPLE_REPORT_DENYLIST = frozenset(
    {"ETR/2016/page_23.pdf", "INTC/2015/page_41.pdf"}
)
PAPER_FIGURE_COMPANY_ALIASES = frozenset({"garmin", "grmn"})
PAPER_FIGURE_YEAR = "2006"
PAPER_FIGURE_PAGE = "page_91.pdf"

ENTRY_FIELDS = frozenset({"id", "pre_text", "post_text", "table", "qa"})
QA_FIELDS = frozenset(
    {"question", "program", "gold_inds", "exe_ans", "program_re"}
)
REPORT_ID_RE = re.compile(r"(?s)^(.+)-(0|[1-9][0-9]*)$")
GOLD_ID_RE = re.compile(r"^(text|table)_(0|[1-9][0-9]*)$")
NORMALIZED_PROGRAM_FINGERPRINT_RE = re.compile(
    r"^divide\(\+?102400(?:\.0+)?,\+?619314(?:\.0+)?\)$"
)
FIRST_OPERAND_RE = re.compile(r"(?<![0-9])102(?:[, ]?)400(?:\.0+)?(?![0-9])")
SECOND_OPERAND_RE = re.compile(r"(?<![0-9])619(?:[, ]?)314(?:\.0+)?(?![0-9])")
DIVISION_OPERATOR_RE = re.compile(
    r"(?:\bdivide(?:d|s|ing)?\b|\bdivision\b|/|÷)", re.IGNORECASE
)
HEX_COMMITMENT_RE = re.compile(r"^[0-9a-f]{8}$|^[0-9a-f]{40}$|^[0-9a-f]{64}$")
PUBLIC_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")

NODE_ORDER = "pre_text_then_all_table_rows_including_table_0_then_post_text"
REPORT_GROUPING_RULE = "remove_rightmost_hyphen_plus_canonical_decimal_suffix"
TABLE_SERIALIZER = (
    "optional_header_0_prefix_then_lowercase_the_row0_of_head_is_cell_"
    "semicolon_clauses_then_literal_ASCII_space_collapse_and_strip"
)
CAPACITY_SHORTFALL_ACTION = (
    "terminal_source_infeasible_without_smaller_blocks_or_same_source_redesign"
)
STATUS_QUALIFIED = "source_qualified_for_frozen_report_disjoint_block_capacity"
STATUS_INFEASIBLE = "terminal_source_infeasible_for_frozen_block_capacity"
STATUS_DIAGNOSTIC = "synthetic_or_nonformal_aggregate_diagnostic"

PUBLIC_RECEIPT_STRINGS = frozenset(
    {
        SCHEMA,
        SOURCE_RELEASE,
        QUALIFICATION_CLASS,
        FORMAL_CUSTODY_SCHEMA,
        NODE_ORDER,
        REPORT_GROUPING_RULE,
        TABLE_SERIALIZER,
        CAPACITY_SHORTFALL_ACTION,
        STATUS_QUALIFIED,
        STATUS_INFEASIBLE,
        STATUS_DIAGNOSTIC,
        "not_performed",
        *EXPECTED_MEMBERS.values(),
    }
)


class FinqaQualificationError(RuntimeError):
    """Raised when source bytes violate the frozen qualification contract."""


@dataclass(frozen=True)
class EntryRecord:
    """Private row-level state; no instance is ever serialized."""

    split: str
    full_id: str
    report_id: str
    addressable_node_count: int
    gold_count: int
    context_fingerprint_sha256: str
    official_example_denylisted: bool
    paper_figure_denylisted: bool
    program_fingerprint_denylisted: bool
    row_fingerprint_denylisted: bool

    @property
    def structurally_eligible(self) -> bool:
        return (
            MIN_ADDRESSABLE_NODES
            <= self.addressable_node_count
            <= MAX_ADDRESSABLE_NODES
            and MIN_GOLD_UNITS <= self.gold_count <= MAX_GOLD_UNITS
        )


def remove_space(text_in: str) -> str:
    """Mirror official ``remove_space``: collapse literal ASCII spaces only."""

    return " ".join(part for part in text_in.split(" ") if part != "")


def table_row_to_text(header: Sequence[str], row: Sequence[str]) -> str:
    """Mirror the repaired official FinQA row serializer exactly."""

    result = ""
    if header[0]:
        result += header[0] + " "
    for head, cell in zip(header[1:], row[1:]):
        result += "the " + row[0] + " of " + head + " is " + cell + " ; "
    return remove_space(result).strip()


def parse_report_id(full_id: str) -> tuple[str, int]:
    """Return the opaque report prefix and canonical decimal question index."""

    match = REPORT_ID_RE.fullmatch(full_id)
    if match is None:
        raise FinqaQualificationError("entry id lacks canonical report suffix")
    return match.group(1), int(match.group(2))


def normalize_program(program: str) -> str:
    """Frozen normalization used only by the disclosed-content denylist."""

    normalized = unicodedata.normalize("NFKC", program).casefold()
    return re.sub(r"\s+", "", normalized)


def program_matches_disclosed_fingerprint(program: str) -> bool:
    return NORMALIZED_PROGRAM_FINGERPRINT_RE.fullmatch(
        normalize_program(program)
    ) is not None


def row_matches_disclosed_fingerprint(row: Sequence[str]) -> bool:
    normalized = unicodedata.normalize("NFKC", " ".join(row)).casefold()
    return bool(
        FIRST_OPERAND_RE.search(normalized)
        and SECOND_OPERAND_RE.search(normalized)
        and DIVISION_OPERATOR_RE.search(normalized)
    )


def report_matches_paper_figure_denylist(report_id: str) -> bool:
    parts = unicodedata.normalize("NFKC", report_id).casefold().split("/")
    return bool(
        parts
        and parts[-1] == PAPER_FIGURE_PAGE
        and PAPER_FIGURE_YEAR in parts
        and any(part in PAPER_FIGURE_COMPANY_ALIASES for part in parts)
    )


def _canonical_json(payload: Mapping[str, Any], *, ensure_ascii: bool = False) -> str:
    return json.dumps(
        payload,
        ensure_ascii=ensure_ascii,
        sort_keys=True,
        separators=(",", ":"),
    )


def _semantic_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_path(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _regular_file(path: str | Path, *, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise FinqaQualificationError(f"{label} must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise FinqaQualificationError(f"{label} is unavailable") from exc
    if not resolved.is_file():
        raise FinqaQualificationError(f"{label} must be a regular file")
    return resolved


def _json_no_duplicate_keys(raw: bytes, *, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FinqaQualificationError(f"{label} is not strict UTF-8") from exc

    def hook(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FinqaQualificationError("duplicate JSON object key")
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise FinqaQualificationError("JSON contains a non-finite constant")

    try:
        return json.loads(
            text,
            object_pairs_hook=hook,
            parse_constant=reject_constant,
        )
    except FinqaQualificationError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise FinqaQualificationError(f"{label} is not valid JSON") from exc


def _read_custody_manifest(path: Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FinqaQualificationError("custody manifest cannot be read") from exc
    file_hash = hashlib.sha256(raw).hexdigest()
    file_size = len(raw)
    payload = _json_no_duplicate_keys(raw, label="custody manifest")
    if not isinstance(payload, Mapping):
        raise FinqaQualificationError("custody manifest root must be an object")
    declared = payload.get("custody_sha256")
    if not isinstance(declared, str) or not re.fullmatch(r"[0-9a-f]{64}", declared):
        raise FinqaQualificationError("custody manifest lacks its body hash")
    body = dict(payload)
    body.pop("custody_sha256", None)
    actual = hashlib.sha256(
        _canonical_json(body, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    if actual != declared:
        raise FinqaQualificationError("custody manifest body hash mismatch")
    receipt = {
        "file_sha256": file_hash,
        "file_size": file_size,
        "declared_body_sha256": declared,
        "body_hash_verified": True,
        "schema": payload.get("schema")
        if isinstance(payload.get("schema"), str)
        else "",
    }
    return payload, receipt


def _validate_formal_custody(
    payload: Mapping[str, Any], receipt: Mapping[str, Any]
) -> None:
    if receipt.get("file_sha256") != FORMAL_CUSTODY_MANIFEST_FILE_SHA256:
        raise FinqaQualificationError("formal custody manifest file binding mismatch")
    if payload.get("custody_sha256") != FORMAL_CUSTODY_SHA256:
        raise FinqaQualificationError("formal custody body binding mismatch")
    if payload.get("schema") != FORMAL_CUSTODY_SCHEMA:
        raise FinqaQualificationError("formal custody schema binding mismatch")
    source = payload.get("official_source_contract")
    archive = source.get("source_archive") if isinstance(source, Mapping) else None
    required = archive.get("expected_required_members") if isinstance(archive, Mapping) else None
    if (
        not isinstance(source, Mapping)
        or source.get("fixed_commit") != OFFICIAL_COMMIT
        or not isinstance(required, list)
        or not all(isinstance(item, str) for item in required)
        or not set(EXPECTED_MEMBERS.values()).issubset(set(required))
    ):
        raise FinqaQualificationError("formal custody source contract mismatch")


def _read_archive(
    archive: Path,
    *,
    bound_hash_and_size: tuple[str, int],
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    selected: dict[str, tuple[tarfile.TarInfo, bytes]] = {}
    regular_count = 0
    directory_count = 0
    nonregular_count = 0
    member_count = 0
    path_to_split = {path: split for split, path in EXPECTED_MEMBERS.items()}
    try:
        with tarfile.open(archive, mode="r:gz") as bundle:
            for member in bundle:
                member_count += 1
                if member_count > MAX_ARCHIVE_MEMBERS:
                    raise FinqaQualificationError("archive member bound exceeded")
                if member.isdir():
                    directory_count += 1
                elif member.isfile():
                    regular_count += 1
                else:
                    nonregular_count += 1

                split = path_to_split.get(member.name)
                if split is None:
                    continue
                if split in selected:
                    raise FinqaQualificationError(
                        "archive contains duplicate exact official split member"
                    )
                if not member.isfile():
                    raise FinqaQualificationError(
                        "exact official split member is not a regular file"
                    )
                if member.size < 0 or member.size > MAX_SPLIT_MEMBER_BYTES:
                    raise FinqaQualificationError("official split member size is invalid")
                handle = bundle.extractfile(member)
                if handle is None:
                    raise FinqaQualificationError("official split member cannot be opened")
                raw = handle.read()
                if len(raw) != member.size:
                    raise FinqaQualificationError("official split member size mismatch")
                selected[split] = (member, raw)
    except FinqaQualificationError:
        raise
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise FinqaQualificationError("invalid FinQA source archive") from exc

    if set(selected) != set(EXPECTED_MEMBERS):
        raise FinqaQualificationError("archive lacks an exact official TRAIN or DEV member")

    datasets: dict[str, list[Any]] = {}
    member_receipts: dict[str, Any] = {}
    for split in ("train", "dev"):
        member, raw = selected[split]
        decoded = _json_no_duplicate_keys(raw, label="dataset member")
        if not isinstance(decoded, list):
            raise FinqaQualificationError("split root must be a JSON list")
        if not decoded:
            raise FinqaQualificationError("split root must not be empty")
        datasets[split] = decoded
        member_receipts[split] = {
            "expected_relative_path": EXPECTED_MEMBERS[split],
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "file_size": len(raw),
            "tar_declared_size": member.size,
        }

    after_hash_and_size = _sha256_path(archive)
    if after_hash_and_size != bound_hash_and_size:
        raise FinqaQualificationError("archive changed during qualification")

    archive_receipt = {
        "file_sha256": bound_hash_and_size[0],
        "file_size": bound_hash_and_size[1],
        "gzip_and_tar_integrity_passed": True,
        "regular_member_count": regular_count,
        "directory_member_count": directory_count,
        "nonregular_member_count": nonregular_count,
        "exact_train_dev_members_opened": 2,
        "test_private_or_evaluate_test_members_opened": 0,
        "official_members": member_receipts,
    }
    return datasets, archive_receipt


def _presence_template(expected: Sequence[str]) -> dict[str, Any]:
    return {
        "record_count": 0,
        "all_required_fields_present_count": 0,
        "required_field_presence_counts": {key: 0 for key in sorted(expected)},
        "unexpected_field_occurrence_count": 0,
    }


def _observe_schema(
    accumulator: dict[str, Any], record: Mapping[str, Any], expected: Sequence[str]
) -> None:
    expected_set = set(expected)
    keys = set(record)
    accumulator["record_count"] += 1
    if expected_set.issubset(keys):
        accumulator["all_required_fields_present_count"] += 1
    for key in expected_set:
        if key in record:
            accumulator["required_field_presence_counts"][key] += 1
    accumulator["unexpected_field_occurrence_count"] += len(keys - expected_set)


def _require_text(record: Mapping[str, Any], field: str, *, nonempty: bool) -> str:
    value = record.get(field)
    if not isinstance(value, str) or (nonempty and not value):
        raise FinqaQualificationError(f"required {field} must be text")
    return value


def _require_text_list(record: Mapping[str, Any], field: str) -> list[str]:
    value = record.get(field)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise FinqaQualificationError(f"required {field} must be a list of text")
    return list(value)


def _require_table(record: Mapping[str, Any]) -> list[list[str]]:
    value = record.get("table")
    if not isinstance(value, list) or not value:
        raise FinqaQualificationError("required table must be a nonempty list")
    table: list[list[str]] = []
    for row in value:
        if (
            not isinstance(row, list)
            or not row
            or not all(isinstance(cell, str) for cell in row)
        ):
            raise FinqaQualificationError(
                "each table row must be a nonempty list of text"
            )
        table.append(list(row))
    return table


def _execution_answer_is_supported(value: Any) -> bool:
    if type(value) in (int, float):
        try:
            return math.isfinite(float(value))
        except (OverflowError, ValueError):
            return False
    return type(value) is str and value in {"yes", "no"}


def _new_split_aggregates() -> tuple[Counter[str], Counter[str]]:
    nodes = Counter(
        {
            "entry_count": 0,
            "pre_text_occurrence_count": 0,
            "post_text_occurrence_count": 0,
            "table_row_occurrence_count_including_header": 0,
            "table_0_candidate_count": 0,
            "addressable_node_occurrence_count": 0,
            "duplicate_addressable_content_occurrence_count": 0,
            "empty_header_0_entry_count": 0,
            "nonempty_header_0_entry_count": 0,
            "ragged_nonheader_row_occurrence_count": 0,
            "entry_with_ragged_table_count": 0,
            "entry_below_5_nodes_count": 0,
            "entry_with_5_to_128_nodes_count": 0,
            "entry_above_128_nodes_count": 0,
            "minimum_addressable_node_count": 0,
            "maximum_addressable_node_count": 0,
        }
    )
    gold = Counter(
        {
            "entry_count": 0,
            "gold_unit_count": 0,
            "text_gold_unit_count": 0,
            "table_gold_unit_count": 0,
            "table_0_gold_unit_count": 0,
            "gold_value_exact_canonical_match_count": 0,
            "gold_value_canonical_mismatch_count": 0,
            "entry_below_2_gold_units_count": 0,
            "entry_with_2_to_5_gold_units_count": 0,
            "entry_above_5_gold_units_count": 0,
            "program_fingerprint_match_entry_count": 0,
            "table_row_fingerprint_match_entry_count": 0,
        }
    )
    return nodes, gold


def _parse_split(
    split: str,
    rows: list[Any],
    schema: dict[str, Any],
) -> tuple[list[EntryRecord], dict[str, int], dict[str, int]]:
    records: list[EntryRecord] = []
    node_counts, gold_counts = _new_split_aggregates()
    minimum_nodes: int | None = None
    maximum_nodes: int | None = None

    for row in rows:
        if not isinstance(row, Mapping):
            raise FinqaQualificationError("split entries must be JSON objects")
        _observe_schema(schema["entry"], row, ENTRY_FIELDS)
        full_id = _require_text(row, "id", nonempty=True)
        if "\x00" in full_id:
            raise FinqaQualificationError("entry id contains NUL")
        report_id, _question_index = parse_report_id(full_id)
        pre_text = _require_text_list(row, "pre_text")
        post_text = _require_text_list(row, "post_text")
        table = _require_table(row)
        qa = row.get("qa")
        if not isinstance(qa, Mapping):
            raise FinqaQualificationError("required qa must be an object")
        _observe_schema(schema["qa"], qa, QA_FIELDS)
        _require_text(qa, "question", nonempty=True)
        program = _require_text(qa, "program", nonempty=True)
        program_re = _require_text(qa, "program_re", nonempty=True)
        exe_ans = qa.get("exe_ans")
        if not _execution_answer_is_supported(exe_ans):
            raise FinqaQualificationError("exe_ans has an unsupported public type")
        gold_inds = qa.get("gold_inds")
        if not isinstance(gold_inds, Mapping):
            raise FinqaQualificationError("gold_inds must be an object")

        header = table[0]
        rendered_table = [table_row_to_text(header, table_row) for table_row in table]
        addressable = pre_text + rendered_table + post_text
        node_count = len(addressable)
        ragged_count = sum(len(table_row) != len(header) for table_row in table[1:])
        duplicate_content_count = len(addressable) - len(set(addressable))

        node_counts["entry_count"] += 1
        node_counts["pre_text_occurrence_count"] += len(pre_text)
        node_counts["post_text_occurrence_count"] += len(post_text)
        node_counts["table_row_occurrence_count_including_header"] += len(table)
        node_counts["table_0_candidate_count"] += 1
        node_counts["addressable_node_occurrence_count"] += node_count
        node_counts["duplicate_addressable_content_occurrence_count"] += (
            duplicate_content_count
        )
        node_counts[
            "empty_header_0_entry_count"
            if header[0] == ""
            else "nonempty_header_0_entry_count"
        ] += 1
        node_counts["ragged_nonheader_row_occurrence_count"] += ragged_count
        if ragged_count:
            node_counts["entry_with_ragged_table_count"] += 1
        if node_count < MIN_ADDRESSABLE_NODES:
            node_counts["entry_below_5_nodes_count"] += 1
        elif node_count <= MAX_ADDRESSABLE_NODES:
            node_counts["entry_with_5_to_128_nodes_count"] += 1
        else:
            node_counts["entry_above_128_nodes_count"] += 1
        minimum_nodes = node_count if minimum_nodes is None else min(minimum_nodes, node_count)
        maximum_nodes = node_count if maximum_nodes is None else max(maximum_nodes, node_count)

        gold_counts["entry_count"] += 1
        all_text = pre_text + post_text
        for raw_gold_id, gold_value in gold_inds.items():
            if not isinstance(raw_gold_id, str) or not isinstance(gold_value, str):
                raise FinqaQualificationError("gold_inds must bind text keys to text")
            match = GOLD_ID_RE.fullmatch(raw_gold_id)
            if match is None:
                raise FinqaQualificationError("gold key is not a canonical node id")
            node_type = match.group(1)
            index = int(match.group(2))
            if node_type == "text":
                if index >= len(all_text):
                    raise FinqaQualificationError("text gold key is out of bounds")
                canonical_value = all_text[index]
                gold_counts["text_gold_unit_count"] += 1
            else:
                if index >= len(table):
                    raise FinqaQualificationError("table gold key is out of bounds")
                canonical_value = rendered_table[index]
                gold_counts["table_gold_unit_count"] += 1
                if index == 0:
                    gold_counts["table_0_gold_unit_count"] += 1
            gold_counts[
                "gold_value_exact_canonical_match_count"
                if gold_value == canonical_value
                else "gold_value_canonical_mismatch_count"
            ] += 1

        gold_count = len(gold_inds)
        gold_counts["gold_unit_count"] += gold_count
        if gold_count < MIN_GOLD_UNITS:
            gold_counts["entry_below_2_gold_units_count"] += 1
        elif gold_count <= MAX_GOLD_UNITS:
            gold_counts["entry_with_2_to_5_gold_units_count"] += 1
        else:
            gold_counts["entry_above_5_gold_units_count"] += 1

        program_match = program_matches_disclosed_fingerprint(program) or (
            program_matches_disclosed_fingerprint(program_re)
        )
        row_match = any(row_matches_disclosed_fingerprint(item) for item in table)
        if program_match:
            gold_counts["program_fingerprint_match_entry_count"] += 1
        if row_match:
            gold_counts["table_row_fingerprint_match_entry_count"] += 1

        records.append(
            EntryRecord(
                split=split,
                full_id=full_id,
                report_id=report_id,
                addressable_node_count=node_count,
                gold_count=gold_count,
                context_fingerprint_sha256=_semantic_hash(
                    {"pre_text": pre_text, "table": table, "post_text": post_text}
                ),
                official_example_denylisted=(
                    report_id in OFFICIAL_EXAMPLE_REPORT_DENYLIST
                ),
                paper_figure_denylisted=report_matches_paper_figure_denylist(
                    report_id
                ),
                program_fingerprint_denylisted=program_match,
                row_fingerprint_denylisted=row_match,
            )
        )

    node_counts["minimum_addressable_node_count"] = minimum_nodes or 0
    node_counts["maximum_addressable_node_count"] = maximum_nodes or 0
    return records, dict(node_counts), dict(gold_counts)


def _context_duplicate_diagnostics(records: Sequence[EntryRecord]) -> dict[str, int]:
    by_fingerprint: dict[str, set[tuple[str, str]]] = defaultdict(set)
    by_report: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        by_fingerprint[record.context_fingerprint_sha256].add(
            (record.split, record.report_id)
        )
        by_report[(record.split, record.report_id)].add(
            record.context_fingerprint_sha256
        )
    duplicate_groups = [group for group in by_fingerprint.values() if len(group) > 1]
    cross_split = [
        group for group in duplicate_groups if len({split for split, _ in group}) > 1
    ]
    return {
        "distinct_label_free_context_fingerprint_count": len(by_fingerprint),
        "duplicate_context_fingerprint_class_count_across_distinct_reports": len(
            duplicate_groups
        ),
        "cross_split_duplicate_context_fingerprint_class_count": len(cross_split),
        "report_with_multiple_label_free_context_fingerprints_count": sum(
            len(fingerprints) > 1 for fingerprints in by_report.values()
        ),
        "private_context_fingerprints_emitted": 0,
    }


def _aggregate_population(
    records_by_split: Mapping[str, Sequence[EntryRecord]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    train_reports = {record.report_id for record in records_by_split["train"]}
    dev_reports = {record.report_id for record in records_by_split["dev"]}
    cross_split_overlap = train_reports & dev_reports

    flags_by_report: dict[tuple[str, str], dict[str, bool]] = defaultdict(
        lambda: {
            "official_example": False,
            "paper_figure": False,
            "program_fingerprint": False,
            "row_fingerprint": False,
        }
    )
    for split, records in records_by_split.items():
        for record in records:
            flags = flags_by_report[(split, record.report_id)]
            flags["official_example"] |= record.official_example_denylisted
            flags["paper_figure"] |= record.paper_figure_denylisted
            flags["program_fingerprint"] |= record.program_fingerprint_denylisted
            flags["row_fingerprint"] |= record.row_fingerprint_denylisted

    overlap = {
        "train_report_count": len(train_reports),
        "dev_report_count": len(dev_reports),
        "cross_split_exact_report_overlap_count": len(cross_split_overlap),
        "cross_split_overlapping_report_identifiers_emitted": 0,
    }
    exclusions: dict[str, Any] = {}
    eligibility: dict[str, Any] = {}
    for split in ("train", "dev"):
        records = records_by_split[split]
        report_to_questions: Counter[str] = Counter(record.report_id for record in records)
        exclusion_report_sets = {
            name: {
                report
                for (part, report), flags in flags_by_report.items()
                if part == split and flags[name]
            }
            for name in (
                "official_example",
                "paper_figure",
                "program_fingerprint",
                "row_fingerprint",
            )
        }
        union_excluded_reports = set().union(*exclusion_report_sets.values())
        union_excluded_entries = sum(
            record.report_id in union_excluded_reports for record in records
        )
        split_exclusions = {
            "source_report_count": len(report_to_questions),
            "source_entry_count": len(records),
            "official_example_denylist_report_count": len(
                exclusion_report_sets["official_example"]
            ),
            "paper_figure_semantic_denylist_report_count": len(
                exclusion_report_sets["paper_figure"]
            ),
            "program_content_fingerprint_report_count": len(
                exclusion_report_sets["program_fingerprint"]
            ),
            "table_row_content_fingerprint_report_count": len(
                exclusion_report_sets["row_fingerprint"]
            ),
            "union_custody_excluded_report_count": len(union_excluded_reports),
            "union_custody_excluded_entry_count": union_excluded_entries,
            "cross_split_overlap_report_count": len(
                set(report_to_questions) & cross_split_overlap
            ),
            "cross_split_overlap_entry_count": sum(
                record.report_id in cross_split_overlap for record in records
            ),
        }
        exclusions[split] = split_exclusions

        structural = [record for record in records if record.structurally_eligible]
        eligible = [
            record
            for record in structural
            if record.report_id not in union_excluded_reports
            and record.report_id not in cross_split_overlap
        ]
        eligible_reports = {record.report_id for record in eligible}
        eligibility[split] = {
            "source_entry_count": len(records),
            "source_report_count": len(report_to_questions),
            "report_with_multiple_questions_count": sum(
                count > 1 for count in report_to_questions.values()
            ),
            "maximum_question_count_on_one_report": max(
                report_to_questions.values(), default=0
            ),
            "node_range_candidate_entry_count": sum(
                MIN_ADDRESSABLE_NODES
                <= record.addressable_node_count
                <= MAX_ADDRESSABLE_NODES
                for record in records
            ),
            "gold_range_candidate_entry_count": sum(
                MIN_GOLD_UNITS <= record.gold_count <= MAX_GOLD_UNITS
                for record in records
            ),
            "joint_structural_candidate_entry_count": len(structural),
            "formal_eligible_entry_count_before_one_question_per_report_cap": len(
                eligible
            ),
            "formal_eligible_report_count_one_question_cap": len(eligible_reports),
            "one_question_per_report_cap_applied": True,
            "cross_split_overlapping_reports_removed_before_capacity": True,
        }
    return overlap, exclusions, eligibility


def _merge_integer_mappings(parts: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    output: Counter[str] = Counter()
    for part in parts:
        for key, value in part.items():
            if type(value) is int:
                output[key] += value
    return dict(output)


def build_qualification(
    archive_path: str | Path,
    custody_manifest_path: str | Path,
    *,
    enforce_formal_bindings: bool = False,
) -> dict[str, Any]:
    """Build one redacted aggregate receipt; intended for the clean worker."""

    archive = _regular_file(archive_path, label="source archive")
    custody_manifest = _regular_file(
        custody_manifest_path, label="custody manifest"
    )

    archive_hash_and_size = _sha256_path(archive)
    custody_payload, custody_receipt = _read_custody_manifest(custody_manifest)
    if enforce_formal_bindings:
        if archive_hash_and_size != (FORMAL_ARCHIVE_SHA256, FORMAL_ARCHIVE_SIZE):
            raise FinqaQualificationError(
                "formal archive byte binding mismatch before row parse"
            )
        _validate_formal_custody(custody_payload, custody_receipt)

    datasets, archive_receipt = _read_archive(
        archive, bound_hash_and_size=archive_hash_and_size
    )
    schema = {
        split: {
            "entry": _presence_template(ENTRY_FIELDS),
            "qa": _presence_template(QA_FIELDS),
        }
        for split in ("train", "dev")
    }
    records_by_split: dict[str, list[EntryRecord]] = {}
    node_counts: dict[str, dict[str, int]] = {}
    gold_counts: dict[str, dict[str, int]] = {}
    seen_full_ids: set[str] = set()
    for split in ("train", "dev"):
        records, split_nodes, split_gold = _parse_split(
            split, datasets[split], schema[split]
        )
        for record in records:
            if record.full_id in seen_full_ids:
                raise FinqaQualificationError("duplicate full entry id")
            seen_full_ids.add(record.full_id)
        records_by_split[split] = records
        node_counts[split] = split_nodes
        gold_counts[split] = split_gold

    all_records = records_by_split["train"] + records_by_split["dev"]
    context_duplicates = _context_duplicate_diagnostics(all_records)
    overlap, exclusions, eligibility = _aggregate_population(records_by_split)

    capacity_by_split = {
        split: {
            "minimum_distinct_eligible_reports": (
                FORMAL_MINIMUM_DISTINCT_ELIGIBLE_REPORTS[split]
            ),
            "observed_distinct_eligible_reports": eligibility[split][
                "formal_eligible_report_count_one_question_cap"
            ],
            "minimum_met": eligibility[split][
                "formal_eligible_report_count_one_question_cap"
            ]
            >= FORMAL_MINIMUM_DISTINCT_ELIGIBLE_REPORTS[split],
        }
        for split in ("train", "dev")
    }
    formal_capacity_met = all(
        part["minimum_met"] for part in capacity_by_split.values()
    )
    status = (
        STATUS_QUALIFIED if formal_capacity_met else STATUS_INFEASIBLE
    ) if enforce_formal_bindings else STATUS_DIAGNOSTIC

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "qualification_class": QUALIFICATION_CLASS,
        "source_release": SOURCE_RELEASE,
        "selection_status": "not_performed",
        "selection_secret_opened_or_generated": False,
        "performance_or_retrieval_scoring_performed": False,
        "formal_binding_mode": enforce_formal_bindings,
        "formal_public_bindings": {
            "archive_sha256": FORMAL_ARCHIVE_SHA256,
            "archive_size": FORMAL_ARCHIVE_SIZE,
            "custody_manifest_file_sha256": FORMAL_CUSTODY_MANIFEST_FILE_SHA256,
            "custody_sha256": FORMAL_CUSTODY_SHA256,
            "custody_commit": FORMAL_CUSTODY_COMMIT,
            "official_source_commit": OFFICIAL_COMMIT,
            "all_input_byte_bindings_verified_before_row_parse": (
                enforce_formal_bindings
            ),
        },
        "archive": archive_receipt,
        "custody_manifest": custody_receipt,
        "field_schema": schema,
        "addressable_graph": {
            "node_order": NODE_ORDER,
            "table_0_is_an_addressable_candidate": True,
            "duplicate_content_nodes_are_not_deduplicated": True,
            "ragged_rows_use_zip_truncation": True,
            "empty_header_0_is_supported": True,
            "table_serializer": TABLE_SERIALIZER,
            "split_aggregates": node_counts,
            "combined": _merge_integer_mappings(list(node_counts.values())),
        },
        "gold_mapping_diagnostics": {
            "gold_keys_are_item_local_bounded_node_ids": True,
            "global_identity_uses_full_entry_plus_local_node": True,
            "gold_values_used_only_for_exact_mapping_diagnostics": True,
            "gold_values_used_for_action_selector_or_score": False,
            "split_aggregates": gold_counts,
            "combined": _merge_integer_mappings(list(gold_counts.values())),
        },
        "report_grouping_and_duplicates": {
            "grouping_rule": REPORT_GROUPING_RULE,
            "report_prefix_is_otherwise_opaque": True,
            "cross_split_report_overlap": overlap,
            "label_free_context_fingerprint_diagnostics": context_duplicates,
        },
        "custody_exclusions": {
            "exclusions_propagated_to_entire_report_before_eligibility": True,
            "split_aggregates": exclusions,
        },
        "selection_eligibility": {
            "minimum_gold_units_inclusive": MIN_GOLD_UNITS,
            "maximum_gold_units_inclusive": MAX_GOLD_UNITS,
            "minimum_addressable_nodes_inclusive": MIN_ADDRESSABLE_NODES,
            "maximum_addressable_nodes_inclusive": MAX_ADDRESSABLE_NODES,
            "one_selected_question_per_report_across_all_blocks": True,
            "report_disjoint_across_train_and_dev_blocks": True,
            "split_aggregates": eligibility,
        },
        "formal_capacity_decision": {
            "distinct_eligible_report_minimums": capacity_by_split,
            "all_minimums_met": formal_capacity_met,
            "capacity_shortfall_action": CAPACITY_SHORTFALL_ACTION,
        },
        "qualification_operations": {
            "source_entries_programmatically_parsed": len(all_records),
            "selection_or_sampling_operations": 0,
            "concrete_item_or_report_identifiers_emitted": 0,
            "source_question_or_evidence_strings_emitted": 0,
            "table_cells_or_gold_mapping_values_emitted": 0,
            "program_or_answer_annotations_emitted": 0,
            "private_content_fingerprints_emitted": 0,
            "test_private_or_evaluate_test_members_opened": 0,
        },
        "status": status,
    }
    receipt["qualification_sha256"] = _semantic_hash(receipt)
    _validate_redacted_shape(receipt)
    return receipt


def _validate_redacted_shape(payload: Any) -> None:
    """Fail closed if a receipt contains row-shaped or arbitrary strings."""

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if not isinstance(key, str) or PUBLIC_KEY_RE.fullmatch(key) is None:
                raise FinqaQualificationError("aggregate receipt contains a private key")
            _validate_redacted_shape(value)
        return
    if isinstance(payload, (list, tuple)):
        raise FinqaQualificationError("aggregate receipt must not contain row arrays")
    if isinstance(payload, str):
        if payload not in PUBLIC_RECEIPT_STRINGS and HEX_COMMITMENT_RE.fullmatch(
            payload
        ) is None:
            raise FinqaQualificationError("aggregate receipt contains a private string")
        return
    if payload is None or type(payload) in (bool, int):
        return
    raise FinqaQualificationError("aggregate receipt contains a nonaggregate value")


def _validate_child_receipt(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise FinqaQualificationError("clean worker returned an invalid receipt")
    declared = payload.get("qualification_sha256")
    if not isinstance(declared, str) or re.fullmatch(r"[0-9a-f]{64}", declared) is None:
        raise FinqaQualificationError("clean worker receipt lacks a semantic hash")
    body = dict(payload)
    body.pop("qualification_sha256", None)
    if _semantic_hash(body) != declared:
        raise FinqaQualificationError("clean worker receipt hash mismatch")
    _validate_redacted_shape(payload)

    operations = payload.get("qualification_operations")
    zero_fields = (
        "selection_or_sampling_operations",
        "concrete_item_or_report_identifiers_emitted",
        "source_question_or_evidence_strings_emitted",
        "table_cells_or_gold_mapping_values_emitted",
        "program_or_answer_annotations_emitted",
        "private_content_fingerprints_emitted",
        "test_private_or_evaluate_test_members_opened",
    )
    if not isinstance(operations, Mapping) or any(
        operations.get(field) != 0 for field in zero_fields
    ):
        raise FinqaQualificationError("clean worker receipt violates redaction")

    if payload.get("formal_binding_mode") is True:
        bindings = payload.get("formal_public_bindings")
        archive = payload.get("archive")
        custody = payload.get("custody_manifest")
        if (
            not isinstance(bindings, Mapping)
            or not isinstance(archive, Mapping)
            or not isinstance(custody, Mapping)
            or bindings.get("archive_sha256") != FORMAL_ARCHIVE_SHA256
            or bindings.get("archive_size") != FORMAL_ARCHIVE_SIZE
            or bindings.get("custody_manifest_file_sha256")
            != FORMAL_CUSTODY_MANIFEST_FILE_SHA256
            or bindings.get("custody_sha256") != FORMAL_CUSTODY_SHA256
            or bindings.get("custody_commit") != FORMAL_CUSTODY_COMMIT
            or bindings.get("official_source_commit") != OFFICIAL_COMMIT
            or bindings.get("all_input_byte_bindings_verified_before_row_parse")
            is not True
            or archive.get("file_sha256") != FORMAL_ARCHIVE_SHA256
            or archive.get("file_size") != FORMAL_ARCHIVE_SIZE
            or custody.get("file_sha256") != FORMAL_CUSTODY_MANIFEST_FILE_SHA256
            or custody.get("declared_body_sha256") != FORMAL_CUSTODY_SHA256
        ):
            raise FinqaQualificationError("formal clean-worker binding drift")
    return payload


def run_clean_qualification(
    archive_path: str | Path,
    custody_manifest_path: str | Path,
    *,
    enforce_formal_bindings: bool = False,
) -> dict[str, Any]:
    """Run qualification under isolated Python with a minimal environment."""

    archive = _regular_file(archive_path, label="source archive")
    custody = _regular_file(custody_manifest_path, label="custody manifest")
    command = [
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        "--_aggregate-worker",
        "--archive",
        str(archive),
        "--custody-manifest",
        str(custody),
    ]
    if enforce_formal_bindings:
        command.append("--formal")
    environment = {
        "PATH": os.defpath,
        "PYTHONHASHSEED": "0",
        "LC_ALL": "C.UTF-8",
    }
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        close_fds=True,
        env=environment,
        cwd=str(Path(__file__).resolve().parent),
    )
    if completed.returncode != 0:
        raise FinqaQualificationError("clean aggregate worker failed")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise FinqaQualificationError("clean worker emitted non-JSON output") from exc
    return _validate_child_receipt(payload)


def _atomic_write_exclusive(destination: Path, raw: bytes, *, mode: int) -> None:
    """Publish complete bytes without replacing or exposing a partial output."""

    temporary = destination.parent / (
        f".{destination.name}.{os.urandom(12).hex()}.tmp"
    )
    descriptor: int | None = None
    published = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            mode,
        )
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination, follow_symlinks=False)
        published = True
        temporary.unlink()
        directory = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if published:
            destination.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
        raise


def _write_json_exclusive(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Persist one public receipt with exclusive creation and exact 0644 mode."""

    destination = Path(path).absolute()
    raw = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"
    _atomic_write_exclusive(destination, raw, mode=0o644)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--custody-manifest", required=True, type=Path)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="parent-process-only exclusive public receipt destination",
    )
    parser.add_argument("--_aggregate-worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments._aggregate_worker:
        if arguments.output is not None:
            parser.error("--output is parent-process-only")
        receipt = build_qualification(
            arguments.archive,
            arguments.custody_manifest,
            enforce_formal_bindings=arguments.formal,
        )
    else:
        if arguments.formal and arguments.output is None:
            parser.error("--formal requires --output")
        receipt = run_clean_qualification(
            arguments.archive,
            arguments.custody_manifest,
            enforce_formal_bindings=arguments.formal,
        )
        if arguments.output is not None:
            _write_json_exclusive(arguments.output, receipt)
    sys.stdout.write(json.dumps(receipt, ensure_ascii=True, sort_keys=True, indent=2))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
