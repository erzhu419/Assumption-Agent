"""One-shot parent-process acquisition for the frozen CUAD graph study.

The production entry point performs only archive/manifests/ZIP-central-directory
checks before durably consuming the attempt marker.  It opens exactly one
``train_separate_questions.json`` member after that marker, forms four
contract-component-disjoint private blocks, and publishes only aggregate
commitments.  It intentionally has no diagnostic, replay, worker, subprocess,
or prequalification mode.

Pure helpers are kept in this module so that the offset, grouping, and
selection contracts can be tested with synthetic fixtures.  They are not
source loaders; the only loader for the fixed official archive is the formal
CLI path.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any
import unicodedata
import zipfile
import zlib


VERSION = "cuad_direct_evaluator_acquisition_v1"
PUBLIC_SCHEMA = "cuad_graph_evaluator_acquisition_v1"
MARKER_SCHEMA = f"{VERSION}_attempt_marker"
FAILURE_SCHEMA = f"{VERSION}_terminal_failure"
LABEL_FREE_BLOCK_SCHEMA = "cuad_direct_v1_label_free_block"
LABEL_BLOCK_SCHEMA = "cuad_direct_v1_label_block"
LABEL_FREE_ITEM_SCHEMA = "cuad_direct_v1_label_free_item"
LABEL_ITEM_SCHEMA = "cuad_direct_v1_label_item"

BLOCK_ORDER = ("A_form", "F_search", "A_hold", "M_search")
BLOCK_SIZE = 64
SELECTED_COUNT = 256
MIN_NODES = 5
MAX_NODES = 128
MAX_NODE_CHARS = 1200
MIN_FALLBACK_CUT = 600
MIN_GOLD = 1
MAX_GOLD = 5
MAX_ARCHIVE_MEMBERS = 100_000
MAX_TRAIN_BYTES = 512 * 1024 * 1024

SOURCE_COMMIT = "67faa0e6023b04fcaae6cc09497ab00e5d63a2a2"
SOURCE_ARCHIVE_GIT_BLOB_SHA1 = "1ae94ff0a9b70b2e3b9b8d215737c8bfae460ddc"
SOURCE_ARCHIVE_SIZE = 18_309_308
TRAIN_BASENAME = "train_separate_questions.json"
SELECTION_SECRET_COMMITMENT_SHA256 = (
    "5446cb77c86051ee369da54831459daf5af5527e0f52fb5972d3c6fcb13cbe9b"
)

DESIGN_RELATIVE = "manifests/cuad_graph_evaluator_design_v1.json"
DESIGN_SCHEMA = "cuad_graph_evaluator_design_v1"
DESIGN_SHA256 = "2a651230838f51ca615fbf93cfc902800f0d1debfb184b8f1b552d4fc6893a15"
DESIGN_FILE_SHA256 = (
    "3c85a6949d18408013e2e8e9da0f140b16da434e63a7a053924532525163052c"
)
CUSTODY_RELATIVE = "manifests/cuad_graph_evaluator_source_custody_v1.json"
CUSTODY_SCHEMA = "cuad_graph_evaluator_source_custody_v1"
CUSTODY_SHA256 = "bd788f83f55daf974185b2a7fbf2f513210aa53288e0aaba5272fbdfb3b1e31e"
CUSTODY_FILE_SHA256 = (
    "53e83094731a3af3c8b3cabc72794015cc09de7f07a578ea9c4868e6f60d052e"
)
SOURCE_ACCESS_RELATIVE = "manifests/cuad_graph_evaluator_source_access_v1.json"
SOURCE_ACCESS_SCHEMA = "cuad_graph_evaluator_source_access_v1"
SOURCE_ACCESS_SHA256 = (
    "007daedd6965e61a4a931f955db23000e55484a05b2d4e73bf94e972c02a1ac1"
)
SOURCE_ACCESS_FILE_SHA256 = (
    "92f1a6aacb5449dab00ec8d275ca1cffbd9021f8e4e60d3c8cb2e09ceeb81fbb"
)
GRAPH_CORE_RELATIVE = (
    "assumption_agent/benchmarks/contractnli_typed_clause_graph_v1.py"
)
GRAPH_CORE_SHA256 = (
    "7aef388172c08eecd227033111ce0e92845bca0b514a8bacbff205566963460c"
)
ARCHIVE_RELATIVE = "artifacts/cuad_official_source_v1/data-67faa0e6023b.zip"
SECRET_RELATIVE = "artifacts/cuad_graph_evaluator_custody_v1/selection.key"
MARKER_RELATIVE = (
    "artifacts/cuad_graph_evaluator_custody_v1/acquisition_attempt_v1.marker"
)
FAILURE_RELATIVE = (
    "artifacts/cuad_graph_evaluator_custody_v1/acquisition_terminal_failure_v1.json"
)
PUBLIC_RECEIPT_RELATIVE = "manifests/cuad_graph_evaluator_acquisition_v1.json"

PRIVATE_RELATIVE_PATHS = {
    "A_form": (
        "artifacts/cuad_graph_evaluator_acquisition_v1/A_form.label_free.private.json",
        "artifacts/cuad_graph_evaluator_acquisition_v1/A_form.labels.private.json",
    ),
    "F_search": (
        "artifacts/cuad_graph_evaluator_acquisition_v1/F_search.label_free.private.json",
        "artifacts/cuad_graph_evaluator_acquisition_v1/F_search.labels.sealed.json",
    ),
    "A_hold": (
        "artifacts/cuad_graph_evaluator_acquisition_v1/A_hold.label_free.sealed.json",
        "artifacts/cuad_graph_evaluator_acquisition_v1/A_hold.labels.sealed.json",
    ),
    "M_search": (
        "artifacts/cuad_graph_evaluator_acquisition_v1/M_search.label_free.sealed.json",
        "artifacts/cuad_graph_evaluator_acquisition_v1/M_search.labels.sealed.json",
    ),
}

EXPOSED_TITLE_OR_ID_PREFIX = (
    "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT"
)
EXPOSED_CONTEXT_SIGNATURES = (
    "This Agreement shall be governed by the laws of the State of California "
    "without giving effect to conflict or choice of law principles.",
    "In addition, Company shall not now or in the future contest the validity "
    "of Investor's ownership of its Intellectual Property.",
    "Company grants to Investor a worldwide, royalty-free, exclusive, "
    "irrevocable license (with the right to grant sublicenses).",
)

LOCAL_REASON_ORDER = (
    "contract_entry_not_object",
    "contract_entry_schema",
    "paragraph_not_object",
    "paragraph_schema",
    "invalid_unicode",
    "duplicate_qa_id",
    "qa_not_object",
    "qa_schema",
    "qa_impossible",
    "answers_empty",
    "answer_not_object",
    "answer_schema",
    "answer_offset_mismatch",
    "answer_fragment_empty",
    "answer_fragment_unmapped",
    "omitted_alignment_missing",
    "omitted_alignment_ambiguous",
    "gold_cardinality",
    "node_cardinality",
    "exposure_title_or_id",
    "exposure_context_signature",
    "selected_context_has_no_eligible_qa",
)

_HEADING = re.compile(
    r"^\s*(?:section|clause|paragraph)\s+"
    r"([0-9]+(?:\.[0-9]+)*|[a-z])\b",
    flags=re.UNICODE,
)
_LIST_MARKER = re.compile(
    r"^\s*(?:(\((?:[a-z]|[ivxlcdm]+|[0-9]{1,3})\))|"
    r"((?:[a-z]|[ivxlcdm]+|[0-9]{1,3})[.)]))\s+",
    flags=re.UNICODE,
)
_PUNCTUATION_SOFT_CUT = re.compile(r"[.!?;:](?=\s)", flags=re.UNICODE)
_NEWLINE_SEQUENCE = re.compile(r"\r\n|\r|\n", flags=re.UNICODE)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_DRIVE_PREFIX = re.compile(r"[A-Za-z]:")
_FORMAL_ENTRY_ACTIVE = False


class CUADAcquisitionError(RuntimeError):
    """A frozen acquisition or source invariant was violated."""


class LocalRowError(ValueError):
    """A row-local exclusion that must be counted rather than aborting."""

    def __init__(self, reason: str):
        if reason not in LOCAL_REASON_ORDER:
            raise AssertionError(reason)
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class SourceNode:
    span_i: int
    start: int
    end: int
    identity_text: str


@dataclass(frozen=True)
class ParagraphRecord:
    ordinal: int
    title: str
    normalized_title: str
    normalized_title_sha256: str
    context: str
    normalized_context_sha256: str
    raw_context_sha256: str
    qas: tuple[Any, ...]


@dataclass(frozen=True)
class EligibleItem:
    component_commitment_sha256: str
    item_commitment_sha256: str
    exact_qa_id_sha256: str
    question: str
    nodes: tuple[SourceNode, ...]
    gold_node_indices: tuple[int, ...]


@dataclass(frozen=True)
class BoundZipMember:
    path: str
    crc32: str
    uncompressed_size: int
    compressed_size: int
    compression_type: int
    flag_bits: int
    create_system: int
    external_attr: int


@dataclass(frozen=True)
class ArchiveBinding:
    sha256: str
    byte_size: int
    git_blob_sha1: str


@dataclass(frozen=True)
class OutputPaths:
    marker: Path
    failure: Path
    public_receipt: Path
    private: Mapping[str, tuple[Path, Path]]


class _DSU:
    def __init__(self, count: int):
        self.parent = list(range(count))
        self.rank = [0] * count

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CUADAcquisitionError("value is not canonical JSON") from exc


def _semantic_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _strict_unicode(value: str, *, reason: str) -> str:
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise LocalRowError(reason) from exc
    return value


def exposure_normalize(value: str) -> str:
    """The frozen disclosure/content-group normalization."""

    value = unicodedata.normalize("NFKC", value)
    value = value.translate(
        {
            ord("\u2018"): "'",
            ord("\u2019"): "'",
            ord("\u201c"): '"',
            ord("\u201d"): '"',
        }
    )
    return " ".join(value.casefold().split())


_NORMALIZED_EXPOSED_PREFIX = exposure_normalize(EXPOSED_TITLE_OR_ID_PREFIX)
_NORMALIZED_CONTEXT_SIGNATURES = tuple(
    exposure_normalize(value) for value in EXPOSED_CONTEXT_SIGNATURES
)


def _trim_unicode_whitespace(value: str, start: int, end: int) -> tuple[int, int]:
    while start < end and value[start].isspace():
        start += 1
    while end > start and value[end - 1].isspace():
        end -= 1
    return start, end


def _line_ranges(text: str) -> Iterable[tuple[int, int, int]]:
    """Yield raw line start, body end, and end including the newline."""

    start = 0
    for match in _NEWLINE_SEQUENCE.finditer(text):
        yield start, match.start(), match.end()
        start = match.end()
    yield start, len(text), len(text)


def _hard_starts(text: str) -> tuple[int, ...]:
    starts = {0}
    previous_blank = False
    for line_start, body_end, _line_end in _line_ranges(text):
        body = text[line_start:body_end]
        stripped_offset = 0
        while stripped_offset < len(body) and body[stripped_offset].isspace():
            stripped_offset += 1
        if stripped_offset < len(body):
            candidate = line_start + stripped_offset
            if previous_blank:
                starts.add(candidate)
            line = text[line_start:body_end]
            if _HEADING.match(line) is not None or _LIST_MARKER.match(line) is not None:
                starts.add(candidate)
            previous_blank = False
        else:
            previous_blank = True
    return tuple(sorted(value for value in starts if 0 <= value < len(text)))


def _fallback_cut(text: str, start: int, block_end: int) -> int:
    upper = min(start + MAX_NODE_CHARS, block_end)
    lower = min(start + MIN_FALLBACK_CUT, upper)
    for offset in range(upper, lower - 1, -1):
        if offset < block_end and text[offset].isspace():
            return offset
        if offset > start and text[offset - 1].isspace():
            return offset
    return upper


def segment_context(context: str) -> tuple[SourceNode, ...]:
    """Partition a raw context without changing offsets or node identity."""

    if not isinstance(context, str):
        raise TypeError("context must be a string")
    _strict_unicode(context, reason="invalid_unicode")
    if not context:
        return ()
    hard = _hard_starts(context)
    block_starts = hard or (0,)
    boundaries = (*block_starts, len(context))
    raw_ranges: list[tuple[int, int]] = []
    for block_start, block_end in zip(boundaries, boundaries[1:]):
        if block_start >= block_end:
            continue
        soft = {block_end}
        for match in _NEWLINE_SEQUENCE.finditer(context, block_start, block_end):
            soft.add(match.end())
        for match in _PUNCTUATION_SOFT_CUT.finditer(
            context, block_start, block_end
        ):
            soft.add(match.end())
        ordered_soft = tuple(sorted(soft))
        current = block_start
        while current < block_end:
            limit = min(current + MAX_NODE_CHARS, block_end)
            feasible = [cut for cut in ordered_soft if current < cut <= limit]
            cut = max(feasible) if feasible else _fallback_cut(context, current, block_end)
            if cut <= current:
                raise CUADAcquisitionError("segmenter made no progress")
            raw_ranges.append((current, cut))
            current = cut

    nodes: list[SourceNode] = []
    for raw_start, raw_end in raw_ranges:
        start, end = _trim_unicode_whitespace(context, raw_start, raw_end)
        if start == end:
            continue
        if end - start > MAX_NODE_CHARS:
            raise CUADAcquisitionError("segment exceeds the frozen maximum")
        nodes.append(
            SourceNode(
                span_i=len(nodes),
                start=start,
                end=end,
                identity_text=context[start:end],
            )
        )
    return tuple(nodes)


def _overlap_nodes(
    context: str, nodes: Sequence[SourceNode], start: int, end: int
) -> frozenset[int]:
    start, end = _trim_unicode_whitespace(context, start, end)
    if start >= end:
        raise LocalRowError("answer_fragment_empty")
    indices = frozenset(
        node.span_i for node in nodes if node.start < end and start < node.end
    )
    if not indices:
        raise LocalRowError("answer_fragment_unmapped")
    return indices


def _all_occurrences(context: str, fragment: str, minimum: int) -> tuple[int, ...]:
    result: list[int] = []
    offset = context.find(fragment, minimum)
    while offset >= 0:
        result.append(offset)
        offset = context.find(fragment, offset + 1)
    return tuple(result)


def map_answer_to_nodes(
    *, context: str, nodes: Sequence[SourceNode], text: str, answer_start: int
) -> frozenset[int]:
    """Map one exact CUAD answer, including strict ``<omitted>`` handling."""

    if not isinstance(text, str) or not text:
        raise LocalRowError("answer_schema")
    _strict_unicode(text, reason="invalid_unicode")
    if isinstance(answer_start, bool) or not isinstance(answer_start, int):
        raise LocalRowError("answer_schema")
    if answer_start < 0 or answer_start > len(context):
        raise LocalRowError("answer_offset_mismatch")

    if "<omitted>" not in text:
        end = answer_start + len(text)
        if end > len(context) or context[answer_start:end] != text:
            raise LocalRowError("answer_offset_mismatch")
        return _overlap_nodes(context, nodes, answer_start, end)

    fragments = [(index, value) for index, value in enumerate(text.split("<omitted>")) if value]
    if not fragments:
        raise LocalRowError("answer_fragment_empty")
    first_fragment = fragments[0][1]
    first_end = answer_start + len(first_fragment)
    if first_end > len(context) or context[answer_start:first_end] != first_fragment:
        raise LocalRowError("answer_offset_mismatch")
    initial_gold = _overlap_nodes(context, nodes, answer_start, first_end)
    states: set[tuple[int, frozenset[int]]] = {(first_end, initial_gold)}
    for _fragment_index, fragment in fragments[1:]:
        next_states: set[tuple[int, frozenset[int]]] = set()
        occurrence_cache: dict[int, tuple[int, ...]] = {}
        for previous_end, previous_gold in states:
            occurrences = occurrence_cache.setdefault(
                previous_end, _all_occurrences(context, fragment, previous_end)
            )
            for occurrence in occurrences:
                fragment_end = occurrence + len(fragment)
                try:
                    contribution = _overlap_nodes(
                        context, nodes, occurrence, fragment_end
                    )
                except LocalRowError:
                    continue
                next_states.add((fragment_end, previous_gold | contribution))
        if not next_states:
            raise LocalRowError("omitted_alignment_missing")
        states = next_states
    gold_sets = {gold for _end, gold in states}
    if not gold_sets:
        raise LocalRowError("omitted_alignment_missing")
    if len(gold_sets) != 1:
        raise LocalRowError("omitted_alignment_ambiguous")
    return next(iter(gold_sets))


def _parse_qa_gold(
    qa: Any,
    *,
    context: str,
    nodes: Sequence[SourceNode],
) -> tuple[str, str, tuple[int, ...]]:
    if not isinstance(qa, Mapping):
        raise LocalRowError("qa_not_object")
    qa_id = qa.get("id")
    question = qa.get("question")
    impossible = qa.get("is_impossible")
    answers = qa.get("answers")
    if (
        not isinstance(qa_id, str)
        or not qa_id
        or not isinstance(question, str)
        or not question.strip()
        or type(impossible) is not bool
        or not isinstance(answers, list)
    ):
        raise LocalRowError("qa_schema")
    _strict_unicode(qa_id, reason="invalid_unicode")
    _strict_unicode(question, reason="invalid_unicode")
    if impossible:
        raise LocalRowError("qa_impossible")
    if not answers:
        raise LocalRowError("answers_empty")
    union: set[int] = set()
    for answer in answers:
        if not isinstance(answer, Mapping):
            raise LocalRowError("answer_not_object")
        text = answer.get("text")
        answer_start = answer.get("answer_start")
        if (
            not isinstance(text, str)
            or not text
            or isinstance(answer_start, bool)
            or not isinstance(answer_start, int)
        ):
            raise LocalRowError("answer_schema")
        union.update(
            map_answer_to_nodes(
                context=context,
                nodes=nodes,
                text=text,
                answer_start=answer_start,
            )
        )
    if not MIN_GOLD <= len(union) <= MAX_GOLD:
        raise LocalRowError("gold_cardinality")
    return qa_id, question, tuple(sorted(union))


def _hmac_array(secret: bytes, values: Sequence[str]) -> bytes:
    raw = json.dumps(
        list(values), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hmac.new(secret, raw, hashlib.sha256).digest()


def component_commitment(records: Sequence[ParagraphRecord]) -> str:
    title_hashes = sorted(
        {record.normalized_title_sha256 for record in records if record.normalized_title}
    )
    context_hashes = sorted({record.normalized_context_sha256 for record in records})
    raw = json.dumps(
        ["cuad_direct_v1", "component", title_hashes, context_hashes],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _records_from_root(
    payload: Any, reason_counts: Counter[str]
) -> tuple[list[ParagraphRecord], dict[str, int]]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), list):
        raise CUADAcquisitionError("TRAIN root is not the frozen SQuAD-v2 envelope")
    records: list[ParagraphRecord] = []
    contract_entries = payload["data"]
    paragraph_total = 0
    for entry in contract_entries:
        if not isinstance(entry, Mapping):
            reason_counts["contract_entry_not_object"] += 1
            continue
        title = entry.get("title")
        paragraphs = entry.get("paragraphs")
        if not isinstance(title, str) or not isinstance(paragraphs, list):
            reason_counts["contract_entry_schema"] += 1
            continue
        try:
            _strict_unicode(title, reason="invalid_unicode")
        except LocalRowError as exc:
            reason_counts[exc.reason] += 1
            continue
        normalized_title = exposure_normalize(title)
        title_hash = _sha256_bytes(normalized_title.encode("utf-8"))
        for paragraph in paragraphs:
            paragraph_total += 1
            if not isinstance(paragraph, Mapping):
                reason_counts["paragraph_not_object"] += 1
                continue
            context = paragraph.get("context")
            qas = paragraph.get("qas")
            if not isinstance(context, str) or not isinstance(qas, list):
                reason_counts["paragraph_schema"] += 1
                continue
            try:
                _strict_unicode(context, reason="invalid_unicode")
                raw = context.encode("utf-8", errors="strict")
            except LocalRowError as exc:
                reason_counts[exc.reason] += 1
                continue
            normalized_context = exposure_normalize(context)
            records.append(
                ParagraphRecord(
                    ordinal=len(records),
                    title=title,
                    normalized_title=normalized_title,
                    normalized_title_sha256=title_hash,
                    context=context,
                    normalized_context_sha256=_sha256_bytes(
                        normalized_context.encode("utf-8")
                    ),
                    raw_context_sha256=_sha256_bytes(raw),
                    qas=tuple(qas),
                )
            )
    return records, {
        "contract_entry_count": len(contract_entries),
        "paragraph_count_declared": paragraph_total,
        "valid_paragraph_record_count": len(records),
    }


def build_components(records: Sequence[ParagraphRecord]) -> tuple[tuple[ParagraphRecord, ...], ...]:
    """DSU by nonempty normalized title or normalized complete-context hash."""

    dsu = _DSU(len(records))
    title_owner: dict[str, int] = {}
    context_owner: dict[str, int] = {}
    for index, record in enumerate(records):
        if record.normalized_title:
            previous = title_owner.setdefault(record.normalized_title, index)
            dsu.union(index, previous)
        previous = context_owner.setdefault(record.normalized_context_sha256, index)
        dsu.union(index, previous)
    groups: dict[int, list[ParagraphRecord]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[dsu.find(index)].append(record)
    return tuple(
        tuple(sorted(group, key=lambda row: row.ordinal))
        for _minimum, group in sorted(
            ((min(row.ordinal for row in value), value) for value in groups.values()),
            key=lambda pair: pair[0],
        )
    )


def _component_exposure_reason(records: Sequence[ParagraphRecord]) -> str | None:
    if any(
        record.normalized_title == _NORMALIZED_EXPOSED_PREFIX
        or record.normalized_title.startswith(_NORMALIZED_EXPOSED_PREFIX)
        for record in records
    ):
        return "exposure_title_or_id"
    for record in records:
        for qa in record.qas:
            if isinstance(qa, Mapping) and isinstance(qa.get("id"), str):
                try:
                    normalized_id = exposure_normalize(qa["id"])
                except (TypeError, ValueError):
                    continue
                if normalized_id.startswith(_NORMALIZED_EXPOSED_PREFIX):
                    return "exposure_title_or_id"
    for record in records:
        normalized_context = exposure_normalize(record.context)
        if any(signature in normalized_context for signature in _NORMALIZED_CONTEXT_SIGNATURES):
            return "exposure_context_signature"
    return None


def _selected_context_records(
    records: Sequence[ParagraphRecord], *, secret: bytes, commitment: str
) -> tuple[ParagraphRecord, ...]:
    by_raw: dict[str, list[ParagraphRecord]] = defaultdict(list)
    exact_context_by_hash: dict[str, str] = {}
    for record in records:
        previous = exact_context_by_hash.setdefault(record.raw_context_sha256, record.context)
        if previous != record.context:
            raise CUADAcquisitionError("raw context SHA256 collision")
        by_raw[record.raw_context_sha256].append(record)
    selected_hash = min(
        by_raw,
        key=lambda raw_hash: (
            _hmac_array(
                secret,
                ["cuad_direct_v1", "context_variant", commitment, raw_hash],
            ),
            raw_hash,
        ),
    )
    return tuple(by_raw[selected_hash])


def _eligible_item_for_component(
    records: Sequence[ParagraphRecord],
    *,
    secret: bytes,
    commitment: str,
    reason_counts: Counter[str],
) -> EligibleItem | None:
    selected_records = _selected_context_records(
        records, secret=secret, commitment=commitment
    )
    context = selected_records[0].context
    if any(record.context != context for record in selected_records):
        raise CUADAcquisitionError("selected raw context hash collision")
    nodes = segment_context(context)
    if not MIN_NODES <= len(nodes) <= MAX_NODES:
        reason_counts["node_cardinality"] += 1
        return None

    raw_qas = [qa for record in selected_records for qa in record.qas]
    id_counts: Counter[str] = Counter(
        qa.get("id")
        for qa in raw_qas
        if isinstance(qa, Mapping) and isinstance(qa.get("id"), str)
    )
    duplicate_ids = {qa_id for qa_id, count in id_counts.items() if count > 1}
    parsed: list[tuple[str, str, tuple[int, ...]]] = []
    for qa in raw_qas:
        if isinstance(qa, Mapping) and isinstance(qa.get("id"), str) and qa["id"] in duplicate_ids:
            reason_counts["duplicate_qa_id"] += 1
            continue
        try:
            parsed.append(_parse_qa_gold(qa, context=context, nodes=nodes))
        except LocalRowError as exc:
            reason_counts[exc.reason] += 1
    if not parsed:
        reason_counts["selected_context_has_no_eligible_qa"] += 1
        return None
    ranked: list[tuple[bytes, str, str, str, tuple[int, ...]]] = []
    for qa_id, question, gold in parsed:
        qa_id_sha = _sha256_bytes(qa_id.encode("utf-8"))
        rank = _hmac_array(
            secret, ["cuad_direct_v1", "item", commitment, qa_id_sha]
        )
        ranked.append((rank, qa_id_sha, qa_id, question, gold))
    _rank, qa_id_sha, _qa_id, question, gold = min(
        ranked, key=lambda row: (row[0], row[1])
    )
    item_commitment = _semantic_hash(
        ["cuad_direct_v1", "selected_item", commitment, qa_id_sha]
    )
    return EligibleItem(
        component_commitment_sha256=commitment,
        item_commitment_sha256=item_commitment,
        exact_qa_id_sha256=qa_id_sha,
        question=question,
        nodes=tuple(nodes),
        gold_node_indices=gold,
    )


def select_blocks_from_payload(
    payload: Any, *, secret: bytes
) -> tuple[dict[str, tuple[EligibleItem, ...]], dict[str, Any]]:
    """Parse and select a synthetic or already-authorized in-memory TRAIN root."""

    if not isinstance(secret, bytes) or len(secret) != 32:
        raise CUADAcquisitionError("selection secret must be 32 raw bytes")
    reasons: Counter[str] = Counter()
    records, root_counts = _records_from_root(payload, reasons)
    components = build_components(records)
    candidates: list[EligibleItem] = []
    exposure_counts: Counter[str] = Counter()
    node_histogram: Counter[int] = Counter()
    gold_histogram: Counter[int] = Counter()
    for records_in_component in components:
        reason = _component_exposure_reason(records_in_component)
        if reason is not None:
            reasons[reason] += 1
            exposure_counts[reason] += 1
            continue
        commitment = component_commitment(records_in_component)
        item = _eligible_item_for_component(
            records_in_component,
            secret=secret,
            commitment=commitment,
            reason_counts=reasons,
        )
        if item is None:
            continue
        candidates.append(item)
        node_histogram[len(item.nodes)] += 1
        gold_histogram[len(item.gold_node_indices)] += 1
    candidates.sort(
        key=lambda item: (
            _hmac_array(
                secret,
                [
                    "cuad_direct_v1",
                    "contract",
                    item.component_commitment_sha256,
                ],
            ),
            item.component_commitment_sha256,
        )
    )
    selected = candidates[:SELECTED_COUNT]
    blocks = {
        block: tuple(selected[index * BLOCK_SIZE : (index + 1) * BLOCK_SIZE])
        for index, block in enumerate(BLOCK_ORDER)
    }
    stats = {
        "root_counts": root_counts,
        "component_counts": {
            "constructed": len(components),
            "exposure_excluded": sum(exposure_counts.values()),
            "eligible": len(candidates),
            "required": SELECTED_COUNT,
            "capacity_satisfied": len(candidates) >= SELECTED_COUNT,
        },
        "parser_reason_counts": {
            reason: reasons[reason] for reason in LOCAL_REASON_ORDER
        },
        "exposure_counts": {
            "title_or_id": exposure_counts["exposure_title_or_id"],
            "context_signature": exposure_counts["exposure_context_signature"],
        },
        "eligible_node_cardinality_histogram": {
            str(key): node_histogram[key] for key in sorted(node_histogram)
        },
        "eligible_gold_cardinality_histogram": {
            str(key): gold_histogram[key] for key in sorted(gold_histogram)
        },
    }
    return blocks, stats


def _validate_zip_info(info: zipfile.ZipInfo) -> None:
    name = info.filename
    if not name or "\x00" in name or "\\" in name or name.startswith("/"):
        raise CUADAcquisitionError("ZIP contains an unsafe member path")
    parts = name.split("/")
    nonterminal = parts[:-1] if info.is_dir() and parts[-1] == "" else parts
    if (
        not nonterminal
        or any(part in {"", ".", ".."} for part in nonterminal)
        or _DRIVE_PREFIX.match(nonterminal[0]) is not None
        or PurePosixPath(name).is_absolute()
    ):
        raise CUADAcquisitionError("ZIP contains an unsafe member path")
    if info.flag_bits & 0x1:
        raise CUADAcquisitionError("ZIP contains an encrypted member")
    if info.file_size < 0 or info.compress_size < 0:
        raise CUADAcquisitionError("ZIP member size is invalid")
    if info.create_system == 3:
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        if unix_mode:
            kind = stat.S_IFMT(unix_mode)
            expected = stat.S_IFDIR if info.is_dir() else stat.S_IFREG
            if kind not in {0, expected}:
                raise CUADAcquisitionError("ZIP contains a symlink or nonregular member")


def inspect_zip_central_directory(path: Path) -> BoundZipMember:
    """Read only ZIP metadata and bind the unique TRAIN basename."""

    try:
        with zipfile.ZipFile(path, "r", allowZip64=True) as archive:
            infos = archive.infolist()
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError) as exc:
        raise CUADAcquisitionError("ZIP central-directory read failed") from exc
    if len(infos) > MAX_ARCHIVE_MEMBERS:
        raise CUADAcquisitionError("ZIP member count exceeds the frozen limit")
    names: set[str] = set()
    matches: list[zipfile.ZipInfo] = []
    for info in infos:
        _validate_zip_info(info)
        if info.filename in names:
            raise CUADAcquisitionError("ZIP contains duplicate member paths")
        names.add(info.filename)
        if not info.is_dir() and PurePosixPath(info.filename).name == TRAIN_BASENAME:
            matches.append(info)
    if len(matches) != 1:
        raise CUADAcquisitionError("ZIP TRAIN basename is not unique")
    info = matches[0]
    if info.file_size > MAX_TRAIN_BYTES:
        raise CUADAcquisitionError("TRAIN member exceeds the frozen byte limit")
    return BoundZipMember(
        path=info.filename,
        crc32=f"{info.CRC & 0xFFFFFFFF:08x}",
        uncompressed_size=info.file_size,
        compressed_size=info.compress_size,
        compression_type=info.compress_type,
        flag_bits=info.flag_bits,
        create_system=info.create_system,
        external_attr=info.external_attr,
    )


def _require_regular_file(path: Path, *, label: str, mode: int | None = None) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CUADAcquisitionError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise CUADAcquisitionError(f"{label} must be a non-symlink regular file")
    if mode is not None and stat.S_IMODE(metadata.st_mode) != mode:
        raise CUADAcquisitionError(f"{label} must have mode {mode:04o}")


def hash_archive(path: Path) -> ArchiveBinding:
    _require_regular_file(path, label="source archive")
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1()
    git_blob.update(f"blob {size}\0".encode("ascii"))
    observed = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            sha256.update(chunk)
            git_blob.update(chunk)
            observed += len(chunk)
    if observed != size:
        raise CUADAcquisitionError("source archive changed while hashing")
    return ArchiveBinding(sha256.hexdigest(), size, git_blob.hexdigest())


def _read_bound_member(
    path: Path, bound: BoundZipMember, initial_archive: ArchiveBinding
) -> tuple[bytes, str]:
    try:
        with zipfile.ZipFile(path, "r", allowZip64=True) as archive:
            infos = [info for info in archive.infolist() if info.filename == bound.path]
            if len(infos) != 1:
                raise CUADAcquisitionError("bound TRAIN member changed after marker")
            info = infos[0]
            observed_bound = BoundZipMember(
                path=info.filename,
                crc32=f"{info.CRC & 0xFFFFFFFF:08x}",
                uncompressed_size=info.file_size,
                compressed_size=info.compress_size,
                compression_type=info.compress_type,
                flag_bits=info.flag_bits,
                create_system=info.create_system,
                external_attr=info.external_attr,
            )
            if observed_bound != bound:
                raise CUADAcquisitionError("bound TRAIN metadata changed after marker")
            chunks: list[bytes] = []
            digest = hashlib.sha256()
            crc = 0
            size = 0
            with archive.open(info, "r") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_TRAIN_BYTES:
                        raise CUADAcquisitionError("TRAIN member exceeds byte limit")
                    digest.update(chunk)
                    crc = zlib.crc32(chunk, crc)
                    chunks.append(chunk)
    except CUADAcquisitionError:
        raise
    except (
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        RuntimeError,
        NotImplementedError,
        EOFError,
        OSError,
    ) as exc:
        raise CUADAcquisitionError("bound TRAIN member read failed") from exc
    if size != bound.uncompressed_size or f"{crc & 0xFFFFFFFF:08x}" != bound.crc32:
        raise CUADAcquisitionError("bound TRAIN bytes fail central-directory checks")
    if hash_archive(path) != initial_archive:
        raise CUADAcquisitionError("source archive changed during acquisition")
    return b"".join(chunks), digest.hexdigest()


def _strict_json(raw: bytes, *, label: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CUADAcquisitionError(f"{label} contains duplicate JSON keys")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise CUADAcquisitionError(f"{label} contains non-finite JSON")

    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except CUADAcquisitionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CUADAcquisitionError(f"{label} is not strict JSON") from exc


def _read_self_hashed_manifest(
    path: Path, *, schema: str, hash_field: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_regular_file(path, label=schema)
    raw = path.read_bytes()
    payload = _strict_json(raw, label=schema)
    if not isinstance(payload, Mapping) or payload.get("schema") != schema:
        raise CUADAcquisitionError(f"{schema} schema mismatch")
    declared = payload.get(hash_field)
    if not isinstance(declared, str) or _HEX64.fullmatch(declared) is None:
        raise CUADAcquisitionError(f"{schema} self-hash is missing")
    body = dict(payload)
    del body[hash_field]
    observed = _semantic_hash(body)
    if observed != declared:
        raise CUADAcquisitionError(f"{schema} self-hash mismatch")
    return dict(payload), {
        "schema": schema,
        "semantic_sha256": observed,
        "file_sha256": _sha256_bytes(raw),
        "byte_size": len(raw),
    }


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise CUADAcquisitionError("manifest binding is incomplete")
        value = value[key]
    return value


def _git_repository(project: Path) -> tuple[Path, Path]:
    """Find the enclosing repository without invoking Git or another process."""

    candidate = project.resolve(strict=True)
    for repository_root in (candidate, *candidate.parents):
        marker = repository_root / ".git"
        if marker.is_dir() and not marker.is_symlink():
            return repository_root, marker
        if marker.is_file() and not marker.is_symlink():
            raw = marker.read_text(encoding="utf-8").strip()
            if not raw.startswith("gitdir: "):
                raise CUADAcquisitionError("worktree .git pointer is malformed")
            git_dir = Path(raw[8:])
            if not git_dir.is_absolute():
                git_dir = (repository_root / git_dir).resolve(strict=True)
            if not git_dir.is_dir() or git_dir.is_symlink():
                raise CUADAcquisitionError("worktree Git directory is unavailable")
            return repository_root, git_dir
    raise CUADAcquisitionError("formal project is not in a Git repository")


def _git_head_oid(git_dir: Path) -> str:
    raw = (git_dir / "HEAD").read_text(encoding="ascii").strip()
    if _HEX64.fullmatch(raw) is not None:
        # SHA-256 repositories are not supported by this frozen SHA-1 source
        # repository contract.
        raise CUADAcquisitionError("unexpected SHA-256 Git repository")
    if re.fullmatch(r"[0-9a-f]{40}", raw):
        return raw
    if not raw.startswith("ref: "):
        raise CUADAcquisitionError("Git HEAD is malformed")
    reference = raw[5:]
    if reference.startswith("/") or ".." in reference.split("/"):
        raise CUADAcquisitionError("Git HEAD reference is unsafe")
    loose = git_dir / reference
    if loose.is_file() and not loose.is_symlink():
        oid = loose.read_text(encoding="ascii").strip()
        if re.fullmatch(r"[0-9a-f]{40}", oid):
            return oid
    packed = git_dir / "packed-refs"
    if packed.is_file() and not packed.is_symlink():
        for line in packed.read_text(encoding="ascii").splitlines():
            if line.startswith(("#", "^")) or not line.strip():
                continue
            fields = line.split(" ", 1)
            if len(fields) == 2 and fields[1] == reference and re.fullmatch(
                r"[0-9a-f]{40}", fields[0]
            ):
                return fields[0]
    raise CUADAcquisitionError("Git HEAD reference cannot be resolved")


def _git_loose_object(git_dir: Path, oid: str) -> tuple[str, bytes]:
    path = git_dir / "objects" / oid[:2] / oid[2:]
    try:
        compressed = path.read_bytes()
        raw = zlib.decompress(compressed)
    except (OSError, zlib.error) as exc:
        raise CUADAcquisitionError(
            "required recent Git object is unavailable as a loose object"
        ) from exc
    header, separator, body = raw.partition(b"\0")
    if not separator:
        raise CUADAcquisitionError("Git object header is malformed")
    fields = header.split(b" ", 1)
    if len(fields) != 2:
        raise CUADAcquisitionError("Git object header is malformed")
    try:
        kind = fields[0].decode("ascii")
        declared_size = int(fields[1])
    except (UnicodeDecodeError, ValueError) as exc:
        raise CUADAcquisitionError("Git object header is malformed") from exc
    if declared_size != len(body):
        raise CUADAcquisitionError("Git object size is malformed")
    return kind, body


def _git_head_tree(git_dir: Path, head_oid: str) -> str:
    kind, body = _git_loose_object(git_dir, head_oid)
    if kind != "commit":
        raise CUADAcquisitionError("Git HEAD does not name a commit")
    first_line = body.split(b"\n", 1)[0]
    if not first_line.startswith(b"tree "):
        raise CUADAcquisitionError("Git commit tree is missing")
    tree_oid = first_line[5:].decode("ascii", errors="strict")
    if re.fullmatch(r"[0-9a-f]{40}", tree_oid) is None:
        raise CUADAcquisitionError("Git commit tree is malformed")
    return tree_oid


def _git_tree_entry(git_dir: Path, tree_oid: str, name: bytes) -> tuple[bytes, str]:
    kind, body = _git_loose_object(git_dir, tree_oid)
    if kind != "tree":
        raise CUADAcquisitionError("Git path traversed a non-tree object")
    offset = 0
    while offset < len(body):
        space = body.find(b" ", offset)
        nul = body.find(b"\0", space + 1)
        if space < 0 or nul < 0 or nul + 21 > len(body):
            raise CUADAcquisitionError("Git tree object is malformed")
        mode = body[offset:space]
        entry_name = body[space + 1 : nul]
        oid = body[nul + 1 : nul + 21].hex()
        if entry_name == name:
            return mode, oid
        offset = nul + 21
    raise CUADAcquisitionError("frozen protocol file is absent from Git HEAD")


def _git_head_blob_oid(
    git_dir: Path, tree_oid: str, relative_path: PurePosixPath
) -> str:
    parts = relative_path.parts
    current = tree_oid
    for index, part in enumerate(parts):
        try:
            encoded = part.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise CUADAcquisitionError("Git protocol path is not UTF-8") from exc
        mode, oid = _git_tree_entry(git_dir, current, encoded)
        if index + 1 < len(parts):
            if mode not in {b"40000", b"040000"}:
                raise CUADAcquisitionError("Git protocol path is not a directory")
            current = oid
        else:
            if mode in {b"40000", b"040000", b"120000", b"160000"}:
                raise CUADAcquisitionError("Git protocol path is not a regular blob")
            return oid
    raise CUADAcquisitionError("empty Git protocol path")


def _git_blob_oid(raw: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {len(raw)}\0".encode("ascii"))
    digest.update(raw)
    return digest.hexdigest()


def verify_protocol_files_at_head(project: Path) -> dict[str, Any]:
    """Verify the frozen acquisition surface is byte-identical to Git HEAD."""

    repository_root, git_dir = _git_repository(project)
    head_oid = _git_head_oid(git_dir)
    tree_oid = _git_head_tree(git_dir, head_oid)
    relative_project = project.resolve(strict=True).relative_to(repository_root)
    files = (
        DESIGN_RELATIVE,
        CUSTODY_RELATIVE,
        SOURCE_ACCESS_RELATIVE,
        GRAPH_CORE_RELATIVE,
        "assumption_agent/benchmarks/cuad_direct_evaluator_acquisition_v1.py",
        "assumption_agent/benchmarks/cuad_graph_evaluator_runner_v1.py",
        "tests/test_cuad_direct_evaluator_acquisition_v1.py",
        "tests/test_cuad_graph_evaluator_runner_v1.py",
    )
    for relative in files:
        path = project / relative
        _require_regular_file(path, label="frozen protocol file")
        raw = path.read_bytes()
        repository_relative = PurePosixPath(relative_project.as_posix()) / relative
        head_blob = _git_head_blob_oid(git_dir, tree_oid, repository_relative)
        if head_blob != _git_blob_oid(raw):
            raise CUADAcquisitionError(
                "frozen protocol file does not byte-match committed Git HEAD"
            )
    return {
        "head_commit": head_oid,
        "verified_file_count": len(files),
        "all_frozen_protocol_files_byte_match_HEAD": True,
        "subprocess_or_worker_used": False,
    }


def verify_formal_protocol(
    *, project: Path, archive: Path, selection_secret: Path
) -> tuple[ArchiveBinding, BoundZipMember, bytes, dict[str, Any]]:
    """Perform every operation allowed before the one-shot marker."""

    head_binding = verify_protocol_files_at_head(project)
    design, design_binding = _read_self_hashed_manifest(
        project / DESIGN_RELATIVE, schema=DESIGN_SCHEMA, hash_field="design_sha256"
    )
    custody, custody_binding = _read_self_hashed_manifest(
        project / CUSTODY_RELATIVE, schema=CUSTODY_SCHEMA, hash_field="custody_sha256"
    )
    access, access_binding = _read_self_hashed_manifest(
        project / SOURCE_ACCESS_RELATIVE,
        schema=SOURCE_ACCESS_SCHEMA,
        hash_field="source_access_sha256",
    )
    if (
        design_binding["semantic_sha256"] != DESIGN_SHA256
        or design_binding["file_sha256"] != DESIGN_FILE_SHA256
        or custody_binding["semantic_sha256"] != CUSTODY_SHA256
        or custody_binding["file_sha256"] != CUSTODY_FILE_SHA256
        or access_binding["semantic_sha256"] != SOURCE_ACCESS_SHA256
        or access_binding["file_sha256"] != SOURCE_ACCESS_FILE_SHA256
        or _sha256_file(project / GRAPH_CORE_RELATIVE) != GRAPH_CORE_SHA256
    ):
        raise CUADAcquisitionError("frozen design, custody, or graph-core binding drifted")
    if (
        _nested(custody, "official_source_contract", "repository_fixed_commit")
        != SOURCE_COMMIT
        or _nested(
            custody,
            "official_source_contract",
            "source_archive",
            "expected_git_blob_sha1",
        )
        != SOURCE_ARCHIVE_GIT_BLOB_SHA1
        or _nested(
            custody,
            "official_source_contract",
            "source_archive",
            "expected_size_bytes",
        )
        != SOURCE_ARCHIVE_SIZE
        or _nested(
            custody, "selection_custody", "selection_secret_commitment_sha256"
        )
        != SELECTION_SECRET_COMMITMENT_SHA256
        or _nested(custody, "mechanism_design_binding", "design_sha256")
        != DESIGN_SHA256
    ):
        raise CUADAcquisitionError("formal custody values drifted")
    expected_signatures = [
        exposure_normalize(value) for value in EXPOSED_CONTEXT_SIGNATURES
    ]
    if (
        _nested(
            custody,
            "exposure_and_exclusion_contract",
            "paper_whole_contract_normalized_context_substring_signatures_v1",
        )
        != expected_signatures
        or _nested(
            custody,
            "exposure_and_exclusion_contract",
            "public_dataset_card_contract_title_or_QA_ID_prefix_v1",
        )
        != EXPOSED_TITLE_OR_ID_PREFIX
    ):
        raise CUADAcquisitionError("formal exposure denylist drifted")

    _require_regular_file(selection_secret, label="selection secret", mode=0o600)
    secret = selection_secret.read_bytes()
    if len(secret) != 32 or _sha256_bytes(secret) != SELECTION_SECRET_COMMITMENT_SHA256:
        raise CUADAcquisitionError("selection secret commitment drifted")
    _require_regular_file(archive, label="source archive", mode=0o600)
    archive_binding = hash_archive(archive)
    if (
        archive_binding.byte_size != SOURCE_ARCHIVE_SIZE
        or archive_binding.git_blob_sha1 != SOURCE_ARCHIVE_GIT_BLOB_SHA1
    ):
        raise CUADAcquisitionError("official archive source pin drifted")

    # The addendum is deliberately created only after archive-byte hashing and
    # a metadata-only central-directory read.  Its exact field names are part
    # of the frozen v1 schema below.
    if (
        _nested(access, "archive_binding", "sha256") != archive_binding.sha256
        or _nested(access, "archive_binding", "observed_size_bytes")
        != archive_binding.byte_size
        or _nested(access, "archive_binding", "computed_git_blob_sha1")
        != archive_binding.git_blob_sha1
        or _nested(access, "custody_binding", "canonical_custody_sha256")
        != CUSTODY_SHA256
        or _nested(access, "design_binding", "design_sha256") != DESIGN_SHA256
    ):
        raise CUADAcquisitionError("source-access archive or preregistration binding drifted")
    member = inspect_zip_central_directory(archive)
    member_binding = _nested(
        access, "central_directory_only_binding", "target_member"
    )
    if (
        not isinstance(member_binding, Mapping)
        or member_binding.get("full_path") != member.path
        or member_binding.get("CRC32_lowercase_hex") != member.crc32
        or member_binding.get("uncompressed_size") != member.uncompressed_size
        or member_binding.get("compressed_size") != member.compressed_size
        or member_binding.get("compression_type") != member.compression_type
        or member_binding.get("flag_bits") != member.flag_bits
        or member_binding.get("create_system") != member.create_system
        or member_binding.get("external_attr") != member.external_attr
        or member_binding.get("member_content_bytes_decompressed") != 0
        or member_binding.get("member_content_bytes_read") != 0
        or member_binding.get("member_SHA256") is not None
        or _nested(
            access, "central_directory_only_binding", "target_basename"
        )
        != TRAIN_BASENAME
        or _nested(
            access, "central_directory_only_binding", "target_basename_match_count"
        )
        != 1
        or _nested(
            access, "source_byte_state", "target_TRAIN_member_content_opened_decompressed_read_or_hashed"
        )
        is not False
    ):
        raise CUADAcquisitionError("source-access TRAIN central-directory binding drifted")
    return archive_binding, member, secret, {
        "design": design_binding,
        "custody": custody_binding,
        "source_access": access_binding,
        "graph_core_file_sha256": GRAPH_CORE_SHA256,
        "acquisition_implementation_file_sha256": _sha256_file(Path(__file__)),
        "git_HEAD": head_binding,
        "protocol_surface_matches_frozen_files_before_marker": True,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_safe_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = path.parent.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise CUADAcquisitionError("output parent must be a non-symlink directory")


def _atomic_write_exclusive(path: Path, raw: bytes, *, mode: int) -> None:
    _ensure_safe_parent(path)
    temporary = path.parent / f".{path.name}.{os.urandom(12).hex()}.tmp"
    descriptor: int | None = None
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
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _self_hashed_payload(payload: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    body = dict(payload)
    body.pop(hash_field, None)
    body[hash_field] = _semantic_hash(body)
    return body


def _write_json_exclusive(
    path: Path, payload: Mapping[str, Any], *, hash_field: str, mode: int
) -> tuple[str, dict[str, Any]]:
    body = _self_hashed_payload(payload, hash_field)
    raw = json.dumps(body, ensure_ascii=True, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    _atomic_write_exclusive(path, raw, mode=mode)
    return _sha256_bytes(raw), body


def consume_attempt_marker(path: Path, *, preflight: Mapping[str, Any]) -> bytes:
    """Create and durably fsync the marker before any TRAIN member open."""

    payload = _self_hashed_payload(
        {
            "schema": MARKER_SCHEMA,
            "design_sha256": DESIGN_SHA256,
            "custody_sha256": CUSTODY_SHA256,
            "source_archive_sha256": preflight["source_archive_sha256"],
            "bound_train_member_metadata_sha256": preflight[
                "bound_train_member_metadata_sha256"
            ],
            "TRAIN_member_opened_before_marker": False,
            "retry_replay_resample_authorized": False,
        },
        "marker_sha256",
    )
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    _ensure_safe_parent(path)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        _fsync_directory(path.parent)
    except FileExistsError as exc:
        raise CUADAcquisitionError(
            "formal CUAD acquisition attempt is already consumed"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return raw


def _default_output_paths(project: Path) -> OutputPaths:
    return OutputPaths(
        marker=project / MARKER_RELATIVE,
        failure=project / FAILURE_RELATIVE,
        public_receipt=project / PUBLIC_RECEIPT_RELATIVE,
        private={
            block: (project / paths[0], project / paths[1])
            for block, paths in PRIVATE_RELATIVE_PATHS.items()
        },
    )


def _preflight_outputs(paths: OutputPaths) -> None:
    all_paths = [paths.marker, paths.failure, paths.public_receipt]
    for block in BLOCK_ORDER:
        all_paths.extend(paths.private[block])
    for path in all_paths:
        _ensure_safe_parent(path)
        if path.exists() or path.is_symlink():
            raise CUADAcquisitionError(f"one-shot output already exists: {path.name}")


def _view_row(block: str, ordinal: int, item: EligibleItem) -> dict[str, Any]:
    return {
        "schema": LABEL_FREE_ITEM_SCHEMA,
        "block": block,
        "ordinal": ordinal,
        "item_commitment_sha256": item.item_commitment_sha256,
        "component_commitment_sha256": item.component_commitment_sha256,
        "question": item.question,
        "title": "CUAD_contract",
        "nodes": [
            {
                "span_i": node.span_i,
                "start": node.start,
                "end": node.end,
                "identity_text": node.identity_text,
            }
            for node in item.nodes
        ],
    }


def _label_row(block: str, ordinal: int, item: EligibleItem) -> dict[str, Any]:
    return {
        "schema": LABEL_ITEM_SCHEMA,
        "block": block,
        "ordinal": ordinal,
        "item_commitment_sha256": item.item_commitment_sha256,
        "gold_node_indices": list(item.gold_node_indices),
    }


def _private_envelope(
    *, schema: str, block: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return _self_hashed_payload(
        {"schema": schema, "block": block, "count": len(rows), "rows": list(rows)},
        "block_sha256",
    )


def _assert_public_redacted(payload: Any) -> None:
    forbidden_keys = {
        "title",
        "id",
        "qa_id",
        "question",
        "context",
        "answer",
        "answers",
        "answer_start",
        "answer_text",
        "nodes",
        "identity_text",
        "gold_node_indices",
        "item_commitment_sha256",
        "component_commitment_sha256",
        "rows",
    }

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            if forbidden_keys & set(value):
                raise CUADAcquisitionError("public receipt contains a private field")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True).casefold()
    if _NORMALIZED_EXPOSED_PREFIX.casefold() in serialized:
        raise CUADAcquisitionError("public receipt contains exposure content")


def _persist_private_blocks(
    blocks: Mapping[str, Sequence[EligibleItem]], paths: OutputPaths
) -> list[dict[str, Any]]:
    commitments: list[dict[str, Any]] = []
    for block in BLOCK_ORDER:
        items = blocks[block]
        if len(items) != BLOCK_SIZE:
            raise CUADAcquisitionError("private block size drifted")
        views = [_view_row(block, index, item) for index, item in enumerate(items)]
        labels = [_label_row(block, index, item) for index, item in enumerate(items)]
        view_payload = _private_envelope(
            schema=LABEL_FREE_BLOCK_SCHEMA, block=block, rows=views
        )
        label_payload = _private_envelope(
            schema=LABEL_BLOCK_SCHEMA, block=block, rows=labels
        )
        view_raw = json.dumps(
            view_payload, ensure_ascii=True, sort_keys=True, indent=2
        ).encode("utf-8") + b"\n"
        label_raw = json.dumps(
            label_payload, ensure_ascii=True, sort_keys=True, indent=2
        ).encode("utf-8") + b"\n"
        view_path, label_path = paths.private[block]
        _atomic_write_exclusive(view_path, view_raw, mode=0o600)
        _atomic_write_exclusive(label_path, label_raw, mode=0o600)
        item_root = _semantic_hash(
            [item.item_commitment_sha256 for item in items]
        )
        commitments.append(
            {
                "block": block,
                "count": len(items),
                "item_commitment_root_sha256": item_root,
                "label_free_file_sha256": _sha256_bytes(view_raw),
                "label_file_sha256": _sha256_bytes(label_raw),
                "label_free_block_sha256": view_payload["block_sha256"],
                "label_block_sha256": label_payload["block_sha256"],
            }
        )
    return commitments


def _base_public_receipt(
    *,
    status: str,
    archive_binding: ArchiveBinding,
    member: BoundZipMember,
    member_sha256: str | None,
    marker_raw: bytes,
    protocol_bindings: Mapping[str, Any],
    stats: Mapping[str, Any] | None,
) -> dict[str, Any]:
    marker_payload = _strict_json(marker_raw, label="attempt marker")
    return {
        "schema": PUBLIC_SCHEMA,
        "status": status,
        "source": {
            "repository_fixed_commit": SOURCE_COMMIT,
            "archive_sha256": archive_binding.sha256,
            "archive_byte_size": archive_binding.byte_size,
            "archive_git_blob_sha1": archive_binding.git_blob_sha1,
            "train_member_path_sha256": _sha256_bytes(member.path.encode("utf-8")),
            "train_member_crc32": member.crc32,
            "train_member_uncompressed_size": member.uncompressed_size,
            "train_member_compressed_size": member.compressed_size,
            "train_member_compression_type": member.compression_type,
            "train_member_flag_bits": member.flag_bits,
            "train_member_create_system": member.create_system,
            "train_member_external_attr": member.external_attr,
            "train_member_sha256": member_sha256,
            "other_member_content_open_count": 0,
            "TEST_or_CUADv1_member_open_count": 0,
        },
        "attempt": {
            "marker_file_sha256": _sha256_bytes(marker_raw),
            "marker_sha256": marker_payload["marker_sha256"],
            "marker_durable_before_train_member_open": True,
            "formal_invocation_count": 1,
            "worker_or_subprocess_count": 0,
            "retry_replay_resample_authorized": False,
        },
        "protocol_bindings": dict(protocol_bindings),
        "aggregate": dict(stats or {}),
        "safety": {
            "selection_completed_before_action_or_score": status
            == "private_four_block_pack_formed",
            "performance_scores_computed": 0,
            "model_calls": 0,
            "network_calls": 0,
            "online_evaluator_calls": 0,
            "public_item_content_or_identifiers": 0,
        },
    }


def _record_terminal_failure(
    *,
    paths: OutputPaths,
    archive_binding: ArchiveBinding,
    member: BoundZipMember,
    marker_raw: bytes,
    protocol_bindings: Mapping[str, Any],
    stage: str,
    error: BaseException,
) -> None:
    private_failure = {
        "schema": FAILURE_SCHEMA,
        "stage": stage,
        "exception_type": f"{type(error).__module__}.{type(error).__qualname__}",
        "exception_message_sha256": _sha256_bytes(str(error).encode("utf-8")),
        "marker_file_sha256": _sha256_bytes(marker_raw),
        "same_source_replay_authorized": False,
    }
    try:
        _write_json_exclusive(
            paths.failure,
            private_failure,
            hash_field="failure_sha256",
            mode=0o600,
        )
    except BaseException:
        pass
    receipt = _base_public_receipt(
        status="terminal_infrastructure_invalid",
        archive_binding=archive_binding,
        member=member,
        member_sha256=None,
        marker_raw=marker_raw,
        protocol_bindings=protocol_bindings,
        stats={
            "failure_stage": stage,
            "exception_type_sha256": _sha256_bytes(
                f"{type(error).__module__}.{type(error).__qualname__}".encode("utf-8")
            ),
            "same_source_replay_authorized": False,
        },
    )
    try:
        _assert_public_redacted(receipt)
        _write_json_exclusive(
            paths.public_receipt,
            receipt,
            hash_field="acquisition_sha256",
            mode=0o644,
        )
    except BaseException:
        pass


def execute_acquisition_once(
    *,
    archive_path: Path,
    archive_binding: ArchiveBinding,
    bound_member: BoundZipMember,
    secret: bytes,
    protocol_bindings: Mapping[str, Any],
    paths: OutputPaths,
) -> dict[str, Any]:
    """Internal one-shot engine; callers must complete pre-marker authorization."""

    _preflight_outputs(paths)
    preflight = {
        "source_archive_sha256": archive_binding.sha256,
        "bound_train_member_metadata_sha256": _semantic_hash(
            {
                "path": bound_member.path,
                "crc32": bound_member.crc32,
                "uncompressed_size": bound_member.uncompressed_size,
                "compressed_size": bound_member.compressed_size,
                "compression_type": bound_member.compression_type,
                "flag_bits": bound_member.flag_bits,
                "create_system": bound_member.create_system,
                "external_attr": bound_member.external_attr,
            }
        ),
    }
    marker_raw = consume_attempt_marker(paths.marker, preflight=preflight)

    stage = "read_bound_TRAIN_member"
    try:
        member_raw, member_sha256 = _read_bound_member(
            archive_path, bound_member, archive_binding
        )
        stage = "parse_strict_SQuAD_v2_JSON"
        payload = _strict_json(member_raw, label="CUAD TRAIN member")
        stage = "form_component_disjoint_selection"
        blocks, stats = select_blocks_from_payload(payload, secret=secret)
        if not stats["component_counts"]["capacity_satisfied"]:
            receipt = _base_public_receipt(
                status="terminal_source_capacity_insufficient",
                archive_binding=archive_binding,
                member=bound_member,
                member_sha256=member_sha256,
                marker_raw=marker_raw,
                protocol_bindings=protocol_bindings,
                stats=stats,
            )
            receipt["blocks"] = {
                "private_files_created": 0,
                "selected_block_count": 0,
                "selected_item_count": 0,
                "smaller_blocks_or_resampling_authorized": False,
            }
            _assert_public_redacted(receipt)
            _write_json_exclusive(
                paths.public_receipt,
                receipt,
                hash_field="acquisition_sha256",
                mode=0o644,
            )
            return receipt

        stage = "persist_eight_private_block_files"
        commitments = _persist_private_blocks(blocks, paths)
        receipt = _base_public_receipt(
            status="private_four_block_pack_formed",
            archive_binding=archive_binding,
            member=bound_member,
            member_sha256=member_sha256,
            marker_raw=marker_raw,
            protocol_bindings=protocol_bindings,
            stats=stats,
        )
        receipt["blocks"] = {
            "block_order": list(BLOCK_ORDER),
            "block_size": BLOCK_SIZE,
            "selected_item_count": SELECTED_COUNT,
            "global_component_disjointness": True,
            "private_file_commitments": commitments,
        }
        _assert_public_redacted(receipt)
        stage = "persist_public_aggregate_receipt"
        _write_json_exclusive(
            paths.public_receipt,
            receipt,
            hash_field="acquisition_sha256",
            mode=0o644,
        )
        return receipt
    except BaseException as exc:
        _record_terminal_failure(
            paths=paths,
            archive_binding=archive_binding,
            member=bound_member,
            marker_raw=marker_raw,
            protocol_bindings=protocol_bindings,
            stage=stage,
            error=exc,
        )
        raise


def formal_acquire(
    *,
    project: Path,
    archive_path: Path,
    selection_secret_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run the unique official acquisition directly in this parent process."""

    if _FORMAL_ENTRY_ACTIVE is not True:
        raise CUADAcquisitionError("official row access is available only through --formal")
    root = project.resolve(strict=True)
    expected_archive = (root / ARCHIVE_RELATIVE).absolute()
    expected_secret = (root / SECRET_RELATIVE).absolute()
    expected_output = (root / PUBLIC_RECEIPT_RELATIVE).absolute()
    if (
        archive_path.absolute() != expected_archive
        or selection_secret_path.absolute() != expected_secret
        or output_path.absolute() != expected_output
    ):
        raise CUADAcquisitionError("formal inputs must use their canonical frozen paths")
    archive_binding, member, secret, protocol = verify_formal_protocol(
        project=root, archive=expected_archive, selection_secret=expected_secret
    )
    return execute_acquisition_once(
        archive_path=expected_archive,
        archive_binding=archive_binding,
        bound_member=member,
        secret=secret,
        protocol_bindings=protocol,
        paths=_default_output_paths(root),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", action="store_true", required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--selection-secret", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if not arguments.formal:
        raise CUADAcquisitionError("no nonformal source-loading mode exists")
    global _FORMAL_ENTRY_ACTIVE
    _FORMAL_ENTRY_ACTIVE = True
    try:
        formal_acquire(
            project=arguments.project,
            archive_path=arguments.archive,
            selection_secret_path=arguments.selection_secret,
            output_path=arguments.output,
        )
    finally:
        _FORMAL_ENTRY_ACTIVE = False
    return 0


__all__ = [
    "ArchiveBinding",
    "BLOCK_ORDER",
    "BLOCK_SIZE",
    "BoundZipMember",
    "CUADAcquisitionError",
    "EligibleItem",
    "LABEL_BLOCK_SCHEMA",
    "LABEL_FREE_BLOCK_SCHEMA",
    "MAX_NODE_CHARS",
    "OutputPaths",
    "ParagraphRecord",
    "SELECTED_COUNT",
    "SourceNode",
    "build_components",
    "component_commitment",
    "execute_acquisition_once",
    "exposure_normalize",
    "hash_archive",
    "inspect_zip_central_directory",
    "map_answer_to_nodes",
    "segment_context",
    "select_blocks_from_payload",
]


if __name__ == "__main__":
    raise SystemExit(main())
