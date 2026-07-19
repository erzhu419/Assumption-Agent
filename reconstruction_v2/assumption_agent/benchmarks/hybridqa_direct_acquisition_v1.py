"""One-shot TRAIN selection and private pack formation for HybridQA P6/E2.

The formal entrypoint first consumes the already-frozen implementation, then
creates an exclusive attempt root, exhausts the embedded aggregate source
validator, and only then creates a fresh selection secret.  Candidate
formation, block assignment, corpus fill and shuffle are all fixed private
HMAC operations.  Any failure leaves the root terminal and non-reusable.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import heapq
import hmac
import json
import os
from pathlib import Path
import re
import stat
import sys
import unicodedata
from typing import Any, Iterable, Iterator, Mapping, Sequence

from assumption_agent.benchmarks import hybridqa_source_qualification_v1 as source_qualification


VERSION = "hybridqa_direct_acquisition_v1"
CORPUS_UNIT_COUNT = 609
BLOCK_ORDER = ("A_form", "F_search", "A_hold", "M_search")
FAMILIES = ("PASSAGE_ONLY", "TABLE_ONLY", "DUAL_TABLE_PASSAGE")
PER_FAMILY_QUOTA = {"A_form": 16, "F_search": 12, "A_hold": 10, "M_search": 10}
BLOCK_COUNTS = {block: PER_FAMILY_QUOTA[block] * len(FAMILIES) for block in BLOCK_ORDER}

FORMAL_ROOT_RELATIVE = Path("artifacts/hybridqa_p6_e2_formal_v1")
ACQUISITION_RELATIVE = FORMAL_ROOT_RELATIVE / "acquisition"
SOURCE_RELATIVE = Path("artifacts/hybridqa_official_source_v1")
HYBRIDQA_RELATIVE = SOURCE_RELATIVE / "HybridQA"
WIKITABLES_RELATIVE = SOURCE_RELATIVE / "WikiTables-WithLinks"
TRAIN_RELATIVE = Path("released_data/train.json")
IMPLEMENTATION_FREEZE_RELATIVE = Path("manifests/hybridqa_p6_e2_implementation_freeze_v1.json")
DESIGN_RELATIVE = Path("manifests/hybridqa_p6_e2_design_v1.json")
DESIGN_SHA256 = "028f6a58b4e7809e6165cc04e1356aa1b7904dfbe3a8bee18e92ecf00360de34"

IMPLEMENTATION_FREEZE_REQUIRED_PATHS = tuple(
    sorted(
        {
            "assumption_agent/models.py",
            "assumption_agent/benchmarks/feverous_e2_evaluator_v1.py",
            "assumption_agent/benchmarks/hybridqa_direct_acquisition_v1.py",
            "assumption_agent/benchmarks/hybridqa_isolated_bootstrap_v1.py",
            "assumption_agent/benchmarks/hybridqa_local_runtime_v1.py",
            "assumption_agent/benchmarks/hybridqa_p6_e2_formal_controller_v1.py",
            "assumption_agent/benchmarks/hybridqa_query_anchored_formal_runner_v1.py",
            "assumption_agent/benchmarks/hybridqa_query_anchored_operator_v1.py",
            "assumption_agent/benchmarks/hybridqa_source_qualification_v1.py",
            "assumption_agent/benchmarks/musique_formal_runtime_binding_v2.py",
            "assumption_agent/benchmarks/musique_formal_runtime_binding_v3.py",
            "replication_runtime/hybridqa_official_hipporag_v1/__init__.py",
            "replication_runtime/hybridqa_official_hipporag_v1/adapter.py",
            "replication_runtime/hybridqa_official_hipporag_v1/contract.py",
            "replication_runtime/hybridqa_official_hipporag_v1/worker.py",
            "replication_runtime/multihoprag_minilm_v1/__init__.py",
            "replication_runtime/multihoprag_minilm_v1/adapter.py",
            "replication_runtime/qasper_minilm_v1/__init__.py",
            "replication_runtime/qasper_minilm_v1/binding.py",
            "replication_runtime/musique_official_hipporag_v1/__init__.py",
            "replication_runtime/musique_official_hipporag_v1/adapter.py",
            "replication_runtime/musique_official_hipporag_v1/adapter_v2.py",
            "replication_runtime/musique_official_hipporag_v1/adapter_v3.py",
            "replication_runtime/musique_official_hipporag_v1/binding.py",
            "replication_runtime/musique_official_hipporag_v1/contract.py",
            "replication_runtime/musique_official_hipporag_v1/runtime_attestation_v2.py",
            "replication_runtime/musique_official_hipporag_v1/runtime_attestation_v3.py",
            "replication_runtime/musique_official_hipporag_v1/worker.py",
            "manifests/hybridqa_official_hipporag_duplicate_compatibility_qualification_v1.json",
            "manifests/hybridqa_p6_e2_design_v1.json",
            "manifests/hybridqa_source_custody_v1.json",
            "manifests/musique_official_hipporag_retrieve_only_binding_v1.json",
            "manifests/musique_official_hipporag_runtime_attestation_v2.json",
            "manifests/musique_official_hipporag_runtime_attestation_v3.json",
            "manifests/qasper_minilm_runtime_asset_v1.json",
            "tests/test_hybridqa_direct_acquisition_v1.py",
            "tests/test_hybridqa_isolated_bootstrap_v1.py",
            "tests/test_hybridqa_local_runtime_v1.py",
            "tests/test_hybridqa_official_hipporag_v1.py",
            "tests/test_hybridqa_p6_e2_formal_controller_v1.py",
            "tests/test_hybridqa_query_anchored_formal_runner_v1.py",
            "tests/test_hybridqa_query_anchored_operator_v1.py",
            "tests/test_hybridqa_source_qualification_v1.py",
        }
    )
)
IMPLEMENTATION_FREEZE_SEMANTICS = {
    "dependency_site": "explicit_user_site_before_system_third_party_under_python_isolated_mode",
    "formal_python_version": [3, 10, 12],
    "offline_only": True,
    "one_shot_no_retry_replay_resample_or_threshold_change": True,
    "raw_hipporag_agent_logical_parallelism": "3_times_n",
    "selection_split": "official_TRAIN_only",
}

MARKER_FILENAME = "acquisition.one_shot_marker.json"
SECRET_FILENAME = "selection_secret.private.bin"
PUBLIC_FILENAME = "acquisition.public.json"
CORPUS_FILENAME = "corpus.private.json"
FAILURE_FILENAME = "acquisition.terminal_failure.json"

_ALNUM_RUN = re.compile(r"[^\W_]+", re.UNICODE)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class HybridQaAcquisitionError(RuntimeError):
    """The source, selection, corpus, pack or one-shot contract drifted."""


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise HybridQaAcquisitionError("value is not canonical JSON") from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise HybridQaAcquisitionError("self-hash field already exists")
    return {**dict(body), field: stable_hash(body)}


def verify_self_hash(receipt: Mapping[str, Any], field: str) -> str:
    if not isinstance(receipt, Mapping):
        raise HybridQaAcquisitionError("self-hashed receipt is not an object")
    body = dict(receipt)
    declared = body.pop(field, None)
    if not isinstance(declared, str) or _HEX64.fullmatch(declared) is None:
        raise HybridQaAcquisitionError("self-hash is absent or invalid")
    if stable_hash(body) != declared:
        raise HybridQaAcquisitionError("self-hash mismatch")
    return declared


def canonical_text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise HybridQaAcquisitionError(f"{field} is invalid")
    normalized = " ".join(unicodedata.normalize("NFKC", value).split())
    if not normalized and not allow_empty:
        raise HybridQaAcquisitionError(f"{field} is empty")
    return normalized


def canonical_lexical_tokens(value: object) -> tuple[str, ...]:
    text = canonical_text(value, field="lexical text", allow_empty=True).casefold()
    return tuple(_ALNUM_RUN.findall(text))


def normalized_question(value: object) -> str:
    return canonical_text(value, field="question").casefold()


def _contains_bounded_sequence(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    left = tuple(haystack)
    right = tuple(needle)
    if not right or len(right) > len(left):
        return False
    return any(left[index : index + len(right)] == right for index in range(len(left) - len(right) + 1))


@dataclass(frozen=True, order=True)
class UnitKey:
    unit_type: str
    table_id: str
    local_key: str

    def __post_init__(self) -> None:
        if self.unit_type not in {"table_row", "linked_passage"}:
            raise HybridQaAcquisitionError("unit key type is invalid")
        for value, field in ((self.table_id, "table ID"), (self.local_key, "unit local key")):
            canonical_text(value, field=field)
        if self.unit_type == "table_row":
            try:
                row = int(self.local_key)
            except ValueError as exc:
                raise HybridQaAcquisitionError("row unit key is invalid") from exc
            if str(row) != self.local_key or row < 0:
                raise HybridQaAcquisitionError("row unit key is noncanonical")

    def payload(self) -> list[str]:
        return [self.unit_type, self.table_id, self.local_key]


@dataclass(frozen=True)
class Candidate:
    source_ordinal: int
    question_id: str
    table_id: str
    question: str
    question_postag: str
    family: str
    gold_unit_keys: tuple[UnitKey, ...]

    def __post_init__(self) -> None:
        if type(self.source_ordinal) is not int or self.source_ordinal < 0:
            raise HybridQaAcquisitionError("candidate source ordinal is invalid")
        for value, field in (
            (self.question_id, "question ID"),
            (self.table_id, "table ID"),
            (self.question, "question"),
            (self.question_postag, "question POS tags"),
        ):
            canonical_text(value, field=field)
        if self.family not in FAMILIES:
            raise HybridQaAcquisitionError("candidate family is invalid")
        if not 1 <= len(self.gold_unit_keys) <= 3 or tuple(sorted(set(self.gold_unit_keys))) != self.gold_unit_keys:
            raise HybridQaAcquisitionError("candidate gold units are invalid")
        if any(key.table_id != self.table_id for key in self.gold_unit_keys):
            raise HybridQaAcquisitionError("candidate gold unit crosses tables")


@dataclass(frozen=True)
class CorpusUnit:
    key: UnitKey
    title: str
    body: str
    row_ordinal: int | None
    link_target_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        canonical_text(self.title, field="unit title")
        canonical_text(self.body, field="unit body")
        if tuple(sorted(set(self.link_target_keys))) != self.link_target_keys:
            raise HybridQaAcquisitionError("unit link targets are not canonical")
        if self.key.unit_type == "table_row":
            if type(self.row_ordinal) is not int or self.row_ordinal < 0 or str(self.row_ordinal) != self.key.local_key:
                raise HybridQaAcquisitionError("row unit sidecar is invalid")
        elif self.row_ordinal is not None or self.link_target_keys != (self.key.local_key,):
            raise HybridQaAcquisitionError("passage unit sidecar is invalid")


def _table_parts(table: Mapping[str, Any], request: Mapping[str, Any]) -> tuple[list[str], list[list[tuple[str, tuple[str, ...]]]], dict[str, str], str]:
    if not isinstance(table, Mapping) or not isinstance(request, Mapping):
        raise HybridQaAcquisitionError("table/request source is malformed")
    header_raw = table.get("header")
    data_raw = table.get("data")
    if not isinstance(header_raw, list) or not header_raw or not isinstance(data_raw, list) or not data_raw:
        raise HybridQaAcquisitionError("table header/data is malformed")

    def cell(value: object) -> tuple[str, tuple[str, ...]]:
        if not isinstance(value, list) or len(value) != 2 or not isinstance(value[1], list):
            raise HybridQaAcquisitionError("table cell is malformed")
        text = canonical_text(value[0], field="cell text", allow_empty=True)
        links = tuple(sorted(set(canonical_text(link, field="cell link") for link in value[1])))
        return text, links

    header_cells = [cell(value) for value in header_raw]
    headers = [value[0] for value in header_cells]
    rows: list[list[tuple[str, tuple[str, ...]]]] = []
    for raw_row in data_raw:
        if not isinstance(raw_row, list) or len(raw_row) != len(headers):
            raise HybridQaAcquisitionError("table row width drifted")
        rows.append([cell(value) for value in raw_row])
    passages = {
        canonical_text(key, field="request key"): canonical_text(
            value, field="request passage", allow_empty=True
        )
        for key, value in request.items()
    }
    # Empty titles are source-native and explicitly accepted by the embedded
    # source validator.  Classification does not use the title; corpus
    # projection supplies the stable table ID as a non-empty fallback.
    title = canonical_text(table.get("title"), field="table title", allow_empty=True)
    return headers, rows, passages, title


def classify_candidate(
    *,
    source_ordinal: int,
    row: Mapping[str, Any],
    table: Mapping[str, Any],
    request: Mapping[str, Any],
) -> tuple[Candidate | None, str]:
    """Apply the frozen clean T/P grammar to one decoded TRAIN record."""

    required = {"answer-text", "question", "question_id", "question_postag", "table_id"}
    if not isinstance(row, Mapping) or set(row) != required:
        raise HybridQaAcquisitionError("TRAIN row schema drifted")
    answer_tokens = canonical_lexical_tokens(row["answer-text"])
    if not answer_tokens:
        return None, "empty_answer_tokens"
    question = canonical_text(row["question"], field="question")
    question_postag = canonical_text(row["question_postag"], field="question POS tags")
    if len(question.split()) != len(question_postag.split()):
        return None, "question_postag_alignment"
    question_id = canonical_text(row["question_id"], field="question ID")
    table_id = canonical_text(row["table_id"], field="table ID")
    _headers, data, passages, _title = _table_parts(table, request)

    table_cells: list[tuple[int, int]] = []
    link_cells: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row_i, body_row in enumerate(data):
        for column_i, (surface, links) in enumerate(body_row):
            if canonical_lexical_tokens(surface) == answer_tokens:
                table_cells.append((row_i, column_i))
            for link in links:
                if link not in passages:
                    raise HybridQaAcquisitionError("body link is absent from request map")
                link_cells[link].append((row_i, column_i))
    passage_keys = sorted(
        link
        for link in link_cells
        if _contains_bounded_sequence(canonical_lexical_tokens(passages[link]), answer_tokens)
    )
    has_table = bool(table_cells)
    has_passage = bool(passage_keys)
    if not has_table and not has_passage:
        return None, "neither_locus"
    if has_table and len(table_cells) != 1:
        return None, "ambiguous_table_answer_cell"
    if has_passage and len(passage_keys) != 1:
        return None, "ambiguous_answer_passage"
    if has_passage and len(link_cells[passage_keys[0]]) != 1:
        return None, "ambiguous_passage_link_cell"

    if has_table and has_passage:
        family = "DUAL_TABLE_PASSAGE"
    elif has_table:
        family = "TABLE_ONLY"
    else:
        family = "PASSAGE_ONLY"
    gold: set[UnitKey] = set()
    if has_table:
        gold.add(UnitKey("table_row", table_id, str(table_cells[0][0])))
    if has_passage:
        link = passage_keys[0]
        bridge_row, _bridge_column = link_cells[link][0]
        gold.add(UnitKey("table_row", table_id, str(bridge_row)))
        gold.add(UnitKey("linked_passage", table_id, link))
    candidate = Candidate(
        source_ordinal=source_ordinal,
        question_id=question_id,
        table_id=table_id,
        question=question,
        question_postag=question_postag,
        family=family,
        gold_unit_keys=tuple(sorted(gold)),
    )
    return candidate, "eligible"


def _hmac_digest(secret: bytes, purpose: str, payload: object) -> bytes:
    if not isinstance(secret, bytes) or len(secret) != 32:
        raise HybridQaAcquisitionError("selection secret must contain exactly 32 bytes")
    return hmac.new(secret, purpose.encode("ascii") + b"\0" + _canonical_bytes(payload), hashlib.sha256).digest()


def _selection_secret_commitment(secret: bytes) -> str:
    """Return a domain-separated commitment without publishing the secret."""

    if not isinstance(secret, bytes) or len(secret) != 32:
        raise HybridQaAcquisitionError("selection secret must contain exactly 32 bytes")
    return hashlib.sha256(
        f"{VERSION}:selection-secret-commitment\0".encode("ascii") + secret
    ).hexdigest()


def select_blocks(
    candidates: Sequence[Candidate], *, secret: bytes
) -> dict[str, tuple[Candidate, ...]]:
    """Select all 144 globally question/table-disjoint records by private HMAC."""

    rows = tuple(candidates)
    if len({candidate.question_id for candidate in rows}) != len(rows):
        raise HybridQaAcquisitionError("candidate question IDs are not unique")
    by_family = {family: [candidate for candidate in rows if candidate.family == family] for family in FAMILIES}
    used_questions: set[str] = set()
    used_tables: set[str] = set()
    selected: dict[str, tuple[Candidate, ...]] = {}
    for block in BLOCK_ORDER:
        block_rows: list[Candidate] = []
        for family in FAMILIES:
            ordered = sorted(
                by_family[family],
                key=lambda candidate: (
                    _hmac_digest(secret, f"{VERSION}:select:{block}:{family}", candidate.question_id),
                    candidate.question_id,
                ),
            )
            family_rows: list[Candidate] = []
            for candidate in ordered:
                if candidate.question_id in used_questions or candidate.table_id in used_tables:
                    continue
                family_rows.append(candidate)
                used_questions.add(candidate.question_id)
                used_tables.add(candidate.table_id)
                if len(family_rows) == PER_FAMILY_QUOTA[block]:
                    break
            if len(family_rows) != PER_FAMILY_QUOTA[block]:
                raise HybridQaAcquisitionError("fixed family/block quota is unavailable")
            block_rows.extend(family_rows)
        selected[block] = tuple(block_rows)
    all_rows = tuple(candidate for block in BLOCK_ORDER for candidate in selected[block])
    if len(all_rows) != sum(BLOCK_COUNTS.values()) or len({row.table_id for row in all_rows}) != len(all_rows):
        raise HybridQaAcquisitionError("global selected block invariants drifted")
    return selected


def _row_body(headers: Sequence[str], row: Sequence[tuple[str, tuple[str, ...]]]) -> str:
    values = [
        f"{header if header else f'column_{index}'}: {cell_text}"
        for index, (header, (cell_text, _links)) in enumerate(zip(headers, row, strict=True))
    ]
    return " | ".join(values)


def _passage_title(link: str) -> str:
    value = link.rsplit("/", 1)[-1].replace("_", " ")
    return canonical_text(value or link, field="passage title")


def decoded_corpus_units(
    *, table_id: str, table: Mapping[str, Any], request: Mapping[str, Any]
) -> tuple[CorpusUnit, ...]:
    canonical_table_id = canonical_text(table_id, field="table ID")
    headers, data, passages, title = _table_parts(table, request)
    projected_title = title or canonical_table_id
    units: list[CorpusUnit] = []
    referenced_links: set[str] = set()
    for row_i, row in enumerate(data):
        links = tuple(sorted({link for _surface, cell_links in row for link in cell_links}))
        referenced_links.update(links)
        units.append(
            CorpusUnit(
                key=UnitKey("table_row", canonical_table_id, str(row_i)),
                title=projected_title,
                body=_row_body(headers, row),
                row_ordinal=row_i,
                link_target_keys=links,
            )
        )
    for link in sorted(referenced_links):
        if link not in passages:
            raise HybridQaAcquisitionError("linked passage is absent")
        passage_title = _passage_title(link)
        units.append(
            CorpusUnit(
                key=UnitKey("linked_passage", canonical_table_id, link),
                title=passage_title,
                # Some official request entries are source-native empty
                # placeholders.  Only body-referenced links become units; a
                # deterministic link-derived fallback keeps every shared-arm
                # document non-empty without inventing answer evidence.
                body=passages[link] or passage_title,
                row_ordinal=None,
                link_target_keys=(link,),
            )
        )
    return tuple(units)


def form_fixed_corpus(
    *,
    selected: Mapping[str, Sequence[Candidate]],
    unit_stream: Iterable[CorpusUnit],
    secret: bytes,
) -> tuple[tuple[CorpusUnit, ...], dict[UnitKey, int]]:
    gold_keys = {
        key
        for block in BLOCK_ORDER
        for candidate in selected[block]
        for key in candidate.gold_unit_keys
    }
    if not gold_keys or len(gold_keys) > CORPUS_UNIT_COUNT:
        raise HybridQaAcquisitionError("gold union is outside corpus capacity")
    needed = CORPUS_UNIT_COUNT - len(gold_keys)
    gold_units: dict[UnitKey, CorpusUnit] = {}
    seen: set[UnitKey] = set()
    # UnitKey is an explicit third tie-breaker so even adversarially forced
    # digest collisions never fall through to comparing CorpusUnit objects.
    heap: list[tuple[int, int, UnitKey, CorpusUnit]] = []
    for unit in unit_stream:
        if not isinstance(unit, CorpusUnit) or unit.key in seen:
            raise HybridQaAcquisitionError("corpus stream contains a duplicate or foreign unit")
        seen.add(unit.key)
        if unit.key in gold_keys:
            gold_units[unit.key] = unit
            continue
        if needed == 0:
            continue
        digest = int.from_bytes(_hmac_digest(secret, f"{VERSION}:distractor", unit.key.payload()), "big")
        tie = int(stable_hash(unit.key.payload()), 16)
        entry = (-digest, -tie, unit.key, unit)
        if len(heap) < needed:
            heapq.heappush(heap, entry)
        else:
            worst = (-heap[0][0], -heap[0][1])
            if (digest, tie) < worst:
                heapq.heapreplace(heap, entry)
    if set(gold_units) != gold_keys or len(heap) != needed:
        raise HybridQaAcquisitionError("exact fixed corpus cannot be materialized")
    units = [*gold_units.values(), *(entry[3] for entry in heap)]
    if len({unit.key for unit in units}) != CORPUS_UNIT_COUNT:
        raise HybridQaAcquisitionError("fixed corpus unit uniqueness drifted")
    units.sort(
        key=lambda unit: (
            _hmac_digest(secret, f"{VERSION}:shuffle", unit.key.payload()),
            unit.key,
        )
    )
    frozen = tuple(units)
    return frozen, {unit.key: index for index, unit in enumerate(frozen)}


def item_commitment(
    *,
    block: str,
    ordinal: int,
    question: str,
    question_postag: str,
) -> str:
    if (
        block not in BLOCK_COUNTS
        or type(ordinal) is not int
        or not 0 <= ordinal < BLOCK_COUNTS[block]
        or not isinstance(question, str)
        or not question.strip()
        or not isinstance(question_postag, str)
        or not question_postag.strip()
    ):
        raise HybridQaAcquisitionError("item commitment input drifted")
    return stable_hash(
        {
            "block": block,
            "ordinal": ordinal,
            "question_postag_sha256": hashlib.sha256(
                question_postag.encode("utf-8")
            ).hexdigest(),
            "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "version": VERSION,
        }
    )


def form_private_packs(
    *,
    selected: Mapping[str, Sequence[Candidate]],
    corpus: Sequence[CorpusUnit],
    unit_to_index: Mapping[UnitKey, int],
) -> dict[str, dict[str, Any]]:
    expected_index = {unit.key: index for index, unit in enumerate(corpus)}
    if (
        len(corpus) != CORPUS_UNIT_COUNT
        or len(expected_index) != CORPUS_UNIT_COUNT
        or dict(unit_to_index) != expected_index
    ):
        raise HybridQaAcquisitionError("private pack corpus binding drifted")
    documents = [f"{unit.title}\n\n{unit.body}" for unit in corpus]
    multiplicity = Counter(documents)
    duplicate_group_count = sum(count > 1 for count in multiplicity.values())
    duplicate_unit_count = sum(
        count for count in multiplicity.values() if count > 1
    )
    corpus_body = {
        "schema": f"{VERSION}_corpus_pack",
        "version": VERSION,
        "unit_count": CORPUS_UNIT_COUNT,
        "shared_arm_document_serialization": "title_plus_two_LF_plus_body",
        "duplicate_text_group_count": duplicate_group_count,
        "duplicate_text_unit_count": duplicate_unit_count,
        "duplicate_expansion_delegated_to_frozen_official_HippoRAG_adapter": True,
        "units": [
            {
                "idx": index,
                "unit_type": unit.key.unit_type,
                "title": unit.title,
                "body": unit.body,
                "sidecar": {
                    "table_key": unit.key.table_id,
                    "row_ordinal": unit.row_ordinal,
                    "link_target_keys": list(unit.link_target_keys),
                },
            }
            for index, unit in enumerate(corpus)
        ],
    }
    corpus_pack = self_hashed(corpus_body, "corpus_pack_sha256")
    packs: dict[str, dict[str, Any]] = {CORPUS_FILENAME: corpus_pack}
    for block in BLOCK_ORDER:
        candidates = tuple(selected[block])
        if len(candidates) != BLOCK_COUNTS[block]:
            raise HybridQaAcquisitionError("selected block count drifted")
        commitments = [
            item_commitment(
                block=block,
                ordinal=ordinal,
                question=candidate.question,
                question_postag=candidate.question_postag,
            )
            for ordinal, candidate in enumerate(candidates)
        ]
        view_body = {
            "schema": f"{VERSION}_block_view",
            "version": VERSION,
            "block": block,
            "item_count": len(candidates),
            "items": [
                {
                    "item_commitment_sha256": commitment,
                    "question": candidate.question,
                    "question_postag": candidate.question_postag,
                }
                for candidate, commitment in zip(candidates, commitments, strict=True)
            ],
            "labels_family_gold_or_table_included": False,
        }
        view_pack = self_hashed(view_body, "block_view_sha256")
        packs[f"{block}.view.private.json"] = view_pack
        if block != "F_search":
            labels_body = {
                "schema": f"{VERSION}_label_pack",
                "version": VERSION,
                "block": block,
                "item_count": len(candidates),
                "block_view_sha256": view_pack["block_view_sha256"],
                "corpus_pack_sha256": corpus_pack["corpus_pack_sha256"],
                "items": [
                    {
                        "item_commitment_sha256": commitment,
                        "family": candidate.family,
                        "gold_indices": sorted(unit_to_index[key] for key in candidate.gold_unit_keys),
                    }
                    for candidate, commitment in zip(candidates, commitments, strict=True)
                ],
            }
            packs[f"{block}.labels.sealed.json"] = self_hashed(labels_body, "label_pack_sha256")
    if "F_search.labels.sealed.json" in packs:
        raise HybridQaAcquisitionError("F_search label pack must not exist")
    return packs


def _read_json(path: Path, *, label: str) -> Any:
    try:
        raw = path.read_bytes()
        return json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HybridQaAcquisitionError(f"{label} is invalid") from exc


def _write_exclusive(path: Path, payload: object, *, mode: int = 0o600) -> str:
    raw = _canonical_bytes(payload, newline=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        os.fchmod(handle.fileno(), mode)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
        if stat.S_IMODE(os.fstat(handle.fileno()).st_mode) != mode:
            raise HybridQaAcquisitionError("private artifact permissions are unenforceable")
    return hashlib.sha256(raw).hexdigest()


def _write_secret(path: Path, secret: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(secret)
        handle.flush()
        os.fsync(handle.fileno())
        if stat.S_IMODE(os.fstat(handle.fileno()).st_mode) != 0o600:
            raise HybridQaAcquisitionError("selection secret permissions are unenforceable")


def _read_frozen_regular(project: Path, relative: str, *, label: str) -> bytes:
    if not isinstance(relative, str):
        raise HybridQaAcquisitionError(f"{label} path is unsafe")
    candidate = Path(relative)
    if (
        not relative
        or candidate.is_absolute()
        or candidate.as_posix() != relative
        or ".." in candidate.parts
    ):
        raise HybridQaAcquisitionError(f"{label} path is unsafe")
    cursor = project
    for component in candidate.parts:
        cursor = cursor / component
        try:
            metadata = cursor.lstat()
        except OSError as exc:
            raise HybridQaAcquisitionError(f"{label} is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise HybridQaAcquisitionError(f"{label} contains a symlink")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(project / candidate, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise HybridQaAcquisitionError(f"{label} is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            return handle.read()
    except HybridQaAcquisitionError:
        raise
    except OSError as exc:
        raise HybridQaAcquisitionError(f"{label} is unreadable") from exc


def _verify_implementation_freeze(project: Path) -> dict[str, Any]:
    path = project / IMPLEMENTATION_FREEZE_RELATIVE
    raw = _read_frozen_regular(
        project,
        IMPLEMENTATION_FREEZE_RELATIVE.as_posix(),
        label="implementation freeze",
    )
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HybridQaAcquisitionError("implementation freeze is invalid") from exc
    if (
        not isinstance(value, Mapping)
        or raw != _canonical_bytes(value, newline=True)
    ):
        raise HybridQaAcquisitionError("implementation freeze is not an object")
    verify_self_hash(value, "freeze_sha256")
    files = value.get("files")
    if (
        set(value)
        != {
            "schema",
            "version",
            "status",
            "design_sha256",
            "required_path_registry_sha256",
            "implementation_file_count",
            "freeze_semantics",
            "files",
            "freeze_sha256",
        }
        or value.get("schema") != "hybridqa_p6_e2_implementation_freeze_v1"
        or value.get("version") != "v1"
        or value.get("status") != "implementation_frozen"
        or value.get("design_sha256") != DESIGN_SHA256
        or value.get("required_path_registry_sha256")
        != stable_hash(list(IMPLEMENTATION_FREEZE_REQUIRED_PATHS))
        or value.get("implementation_file_count")
        != len(IMPLEMENTATION_FREEZE_REQUIRED_PATHS)
        or value.get("freeze_semantics") != IMPLEMENTATION_FREEZE_SEMANTICS
        or not isinstance(files, list)
        or len(files) != len(IMPLEMENTATION_FREEZE_REQUIRED_PATHS)
    ):
        raise HybridQaAcquisitionError("implementation freeze contract drifted")
    observed_paths: list[str] = []
    for row in files:
        if not isinstance(row, Mapping) or set(row) != {"relative_path", "sha256"}:
            raise HybridQaAcquisitionError("implementation freeze file row drifted")
        relative = row.get("relative_path")
        digest = row.get("sha256")
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or _HEX64.fullmatch(digest) is None
        ):
            raise HybridQaAcquisitionError("implementation freeze file identity is invalid")
        observed_paths.append(relative)
        file_raw = _read_frozen_regular(
            project, relative, label="implementation freeze member"
        )
        if hashlib.sha256(file_raw).hexdigest() != digest:
            raise HybridQaAcquisitionError("implementation freeze file hash drifted")
    if tuple(observed_paths) != IMPLEMENTATION_FREEZE_REQUIRED_PATHS:
        raise HybridQaAcquisitionError(
            "implementation freeze required path registry drifted"
        )
    return dict(value)


def _candidate_pool(project: Path) -> tuple[tuple[Candidate, ...], dict[str, int]]:
    hybrid_root = project / HYBRIDQA_RELATIVE
    wiki_root = project / WIKITABLES_RELATIVE
    rows = _read_json(hybrid_root / TRAIN_RELATIVE, label="official TRAIN")
    if not isinstance(rows, list) or len(rows) != source_qualification.FORMAL_QA_COUNTS["train"]:
        raise HybridQaAcquisitionError("official TRAIN count drifted")
    collision_counts = Counter(normalized_question(row.get("question")) for row in rows if isinstance(row, Mapping))
    duplicated = {key for key, count in collision_counts.items() if count > 1}
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise HybridQaAcquisitionError("official TRAIN contains a non-object")
        table_id = canonical_text(row.get("table_id"), field="table ID")
        grouped[table_id].append((ordinal, row))
    candidates: list[Candidate] = []
    exclusions: Counter[str] = Counter()
    for table_id in sorted(grouped):
        table = _read_json(wiki_root / "tables_tok" / f"{table_id}.json", label="TRAIN table")
        request = _read_json(wiki_root / "request_tok" / f"{table_id}.json", label="TRAIN request")
        for ordinal, row in grouped[table_id]:
            if normalized_question(row["question"]) in duplicated:
                exclusions["normalized_question_collision"] += 1
                continue
            candidate, disposition = classify_candidate(
                source_ordinal=ordinal,
                row=row,
                table=table,
                request=request,
            )
            if candidate is None:
                exclusions[disposition] += 1
            else:
                candidates.append(candidate)
    if len(candidates) + sum(exclusions.values()) != len(rows):
        raise HybridQaAcquisitionError("candidate accounting drifted")
    return tuple(candidates), dict(sorted(exclusions.items()))


def _official_unit_stream(project: Path) -> Iterator[CorpusUnit]:
    wiki_root = project / WIKITABLES_RELATIVE
    tables = wiki_root / "tables_tok"
    requests = wiki_root / "request_tok"
    paths = sorted(tables.glob("*.json"), key=lambda path: path.name)
    if len(paths) != source_qualification.FORMAL_CORPUS_COUNT:
        raise HybridQaAcquisitionError("official corpus table count drifted")
    for table_path in paths:
        table_id = table_path.stem
        request_path = requests / table_path.name
        table = _read_json(table_path, label="official table")
        request = _read_json(request_path, label="official request")
        yield from decoded_corpus_units(table_id=table_id, table=table, request=request)


def _terminal_failure(acquisition_root: Path, *, stage: str, exc: BaseException) -> None:
    body = {
        "schema": f"{VERSION}_terminal_failure",
        "version": VERSION,
        "status": "terminal_no_retry_or_resample",
        "failure_stage": stage,
        "exception_class": type(exc).__name__,
        "exception_message_sha256": hashlib.sha256(str(exc).encode("utf-8", errors="replace")).hexdigest(),
        "selection_or_score_result_persisted_publicly": False,
    }
    try:
        _write_exclusive(acquisition_root / FAILURE_FILENAME, self_hashed(body, "failure_sha256"))
    except BaseException:
        pass


def run_formal_acquisition(project_root: str | Path) -> dict[str, Any]:
    """Execute the sole formal source validation/selection/materialization pass."""

    project = Path(project_root).resolve(strict=True)
    if not project.is_dir():
        raise HybridQaAcquisitionError("project root is invalid")
    freeze = _verify_implementation_freeze(project)
    formal_root = project / FORMAL_ROOT_RELATIVE
    acquisition_root = project / ACQUISITION_RELATIVE
    # The workspace's generic artifact container is not itself part of the
    # one-shot state.  Only the formal root and its acquisition child must be
    # newly and exclusively created.
    formal_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        formal_root.mkdir(mode=0o700)
        acquisition_root.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise HybridQaAcquisitionError("formal acquisition root already exists and is nonreusable") from exc
    marker_body = {
        "schema": f"{VERSION}_one_shot_marker",
        "version": VERSION,
        "status": "formal_attempt_started",
        "design_sha256": DESIGN_SHA256,
        "implementation_freeze_sha256": freeze["freeze_sha256"],
        "source_validation_completed": False,
        "selection_secret_created": False,
    }
    _write_exclusive(acquisition_root / MARKER_FILENAME, self_hashed(marker_body, "marker_sha256"))
    try:
        qualification_receipt = source_qualification.qualify_official_source(project)
    except BaseException as exc:
        _terminal_failure(acquisition_root, stage="embedded_source_validation", exc=exc)
        raise HybridQaAcquisitionError("embedded source validation failed terminally") from exc
    try:
        secret = os.urandom(32)
        _write_secret(acquisition_root / SECRET_FILENAME, secret)
        candidates, exclusions = _candidate_pool(project)
        selected = select_blocks(candidates, secret=secret)
        corpus, unit_to_index = form_fixed_corpus(
            selected=selected,
            unit_stream=_official_unit_stream(project),
            secret=secret,
        )
        packs = form_private_packs(selected=selected, corpus=corpus, unit_to_index=unit_to_index)
        file_hashes: dict[str, str] = {}
        for filename, payload in packs.items():
            file_hashes[filename] = _write_exclusive(acquisition_root / filename, payload)
        family_pool = Counter(candidate.family for candidate in candidates)
        unit_types = Counter(unit.key.unit_type for unit in corpus)
        public_body = {
            "schema": f"{VERSION}_public_receipt",
            "version": VERSION,
            "status": "formal_acquisition_complete",
            "design_sha256": DESIGN_SHA256,
            "implementation_freeze_sha256": freeze["freeze_sha256"],
            "source_qualification_receipt": qualification_receipt,
            "selection_secret_commitment_sha256": _selection_secret_commitment(secret),
            "selection_secret_persisted_publicly": False,
            "candidate_counts_by_family": {family: family_pool[family] for family in FAMILIES},
            "typed_exclusion_counts": exclusions,
            "block_counts": dict(BLOCK_COUNTS),
            "per_family_quota": dict(PER_FAMILY_QUOTA),
            "selected_question_count": sum(BLOCK_COUNTS.values()),
            "selected_table_count": sum(BLOCK_COUNTS.values()),
            "question_and_table_disjoint": True,
            "corpus_unit_count": CORPUS_UNIT_COUNT,
            "corpus_unit_type_counts": {kind: unit_types[kind] for kind in ("table_row", "linked_passage")},
            "private_pack_file_sha256s": dict(sorted(file_hashes.items())),
            "F_search_label_pack_created": False,
            "raw_question_answer_table_or_unit_identity_persisted_publicly": False,
            "online_evaluator_calls": 0,
            "retry_replay_or_resample": 0,
        }
        public = self_hashed(public_body, "acquisition_receipt_sha256")
        _write_exclusive(acquisition_root / PUBLIC_FILENAME, public)
        try:
            acquisition_root.chmod(0o500)
            metadata = acquisition_root.lstat()
        except OSError as exc:
            raise HybridQaAcquisitionError(
                "completed acquisition root cannot be sealed"
            ) from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o500
        ):
            raise HybridQaAcquisitionError(
                "completed acquisition root seal drifted"
            )
        return public
    except BaseException as exc:
        _terminal_failure(acquisition_root, stage="selection_or_materialization", exc=exc)
        raise HybridQaAcquisitionError("formal selection/materialization failed terminally") from exc


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    arguments = parser.parse_args(argv)
    run_formal_acquisition(arguments.project)
    return 0


def main() -> int:
    from assumption_agent.benchmarks import hybridqa_isolated_bootstrap_v1 as bootstrap

    target = "assumption_agent.benchmarks.hybridqa_direct_acquisition_v1"
    bootstrap.reexec_isolated(target, tuple(sys.argv[1:]))
    bootstrap.assert_isolated(target)
    return _main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACQUISITION_RELATIVE",
    "BLOCK_COUNTS",
    "BLOCK_ORDER",
    "CORPUS_UNIT_COUNT",
    "Candidate",
    "CorpusUnit",
    "FAMILIES",
    "HybridQaAcquisitionError",
    "IMPLEMENTATION_FREEZE_REQUIRED_PATHS",
    "IMPLEMENTATION_FREEZE_SEMANTICS",
    "PER_FAMILY_QUOTA",
    "UnitKey",
    "canonical_lexical_tokens",
    "classify_candidate",
    "decoded_corpus_units",
    "form_fixed_corpus",
    "form_private_packs",
    "item_commitment",
    "normalized_question",
    "run_formal_acquisition",
    "select_blocks",
    "stable_hash",
    "verify_self_hash",
]
