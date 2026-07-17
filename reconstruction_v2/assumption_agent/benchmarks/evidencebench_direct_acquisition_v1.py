"""One-shot parent-process acquisition for EvidenceBench direct v1.

The formal entry point is deliberately unusable until the external self-hashed
implementation-freeze manifest binds the design, custody, source access,
implementation surface, source, and secret.  Once frozen, its only pre-marker
contact with the bound source is a complete byte/hash/Git-blob pass.  JSON
decoding, paper schema access, block formation, and label access all happen
after the durable attempt marker.

Pure in-memory helpers exist only for row-free synthetic tests.  This module
has no downloader, network client, parser worker, smoke-test, source
qualification, replay, or diagnostic source-loading entry point.  Formal
preflight permits exactly two fixed, read-only Git metadata commands; neither
receives a source, secret, private-pack, or output path.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from typing import Any
import unicodedata
import zlib


VERSION = "evidencebench_direct_acquisition_v1"
PUBLIC_SCHEMA = "evidencebench_direct_acquisition_v1"
MARKER_SCHEMA = f"{VERSION}_attempt_marker"
FAILURE_SCHEMA = f"{VERSION}_terminal_failure"
LABEL_FREE_BLOCK_SCHEMA = "evidencebench_direct_v1_label_free_block"
LABEL_BLOCK_SCHEMA = "evidencebench_direct_v1_label_block"
LABEL_FREE_ITEM_SCHEMA = "evidencebench_direct_v1_label_free_item"
LABEL_ITEM_SCHEMA = "evidencebench_direct_v1_label_item"

SOURCE_REPOSITORY = "EvidenceBench/EvidenceBench"
SOURCE_COMMIT = "bf1d9633c694381c7b016fd56ee9f95f48593cc3"
SOURCE_REPOSITORY_PATH = "datasets/evidencebench_test_set.json"
SOURCE_GIT_BLOB_SHA1 = "df380a1ba1359f9cea8bca2f2298dc9fd99e6513"
SOURCE_BYTE_SIZE = 12_735_397
SOURCE_RAW_URL = (
    "https://raw.githubusercontent.com/EvidenceBench/EvidenceBench/"
    f"{SOURCE_COMMIT}/{SOURCE_REPOSITORY_PATH}"
)

ROOT_RECORD_COUNT = 293
BLOCK_ORDER = ("A_form", "F_search", "A_hold", "M_search")
BLOCK_SIZE = 64
SELECTED_COUNT = 256
NODE_COUNT = 32
MIN_SENTENCES = 48
MAX_SOURCE_BYTES = 32 * 1024 * 1024

EXPOSED_PMCID = "PMC5533284"
EXPOSED_DOI = "10.1158/1055-9965.EPI-16-0219"
EXPOSED_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5533284/"
EXPOSURE_METADATA_FIELDS = (
    "paper_id",
    "pmcid",
    "doi",
    "url",
    "paper_url",
    "source_url",
)

DESIGN_RELATIVE = "manifests/evidencebench_graph_evaluator_design_v1.json"
DESIGN_SCHEMA = "evidencebench_graph_evaluator_design_v1"
CUSTODY_RELATIVE = "manifests/evidencebench_graph_evaluator_source_custody_v1.json"
CUSTODY_SCHEMA = "evidencebench_graph_evaluator_source_custody_v1"
SOURCE_ACCESS_RELATIVE = "manifests/evidencebench_graph_evaluator_source_access_v1.json"
SOURCE_ACCESS_SCHEMA = "evidencebench_graph_evaluator_source_access_v1"
IMPLEMENTATION_FREEZE_RELATIVE = "manifests/evidencebench_implementation_freeze_v1.json"
IMPLEMENTATION_FREEZE_SCHEMA = "evidencebench_implementation_freeze_v1"
GRAPH_CORE_RELATIVE = (
    "assumption_agent/benchmarks/evidencebench_typed_scientific_graph_v1.py"
)
SOURCE_RELATIVE = (
    "artifacts/evidencebench_official_source_v1/"
    "evidencebench_test_set-bf1d9633.json"
)
SECRET_RELATIVE = "artifacts/evidencebench_graph_evaluator_custody_v1/selection.key"
MARKER_RELATIVE = (
    "artifacts/evidencebench_graph_evaluator_custody_v1/"
    "acquisition_attempt_v1.marker"
)
FAILURE_RELATIVE = (
    "artifacts/evidencebench_graph_evaluator_custody_v1/"
    "acquisition_terminal_failure_v1.json"
)
PUBLIC_RECEIPT_RELATIVE = "manifests/evidencebench_direct_acquisition_v1.json"
ACQUISITION_RELATIVE = (
    "assumption_agent/benchmarks/evidencebench_direct_acquisition_v1.py"
)
ACQUISITION_TEST_RELATIVE = "tests/test_evidencebench_direct_acquisition_v1.py"
GRAPH_CORE_TEST_RELATIVE = "tests/test_evidencebench_typed_scientific_graph_v1.py"
EVALUATOR_RUNNER_RELATIVE = (
    "assumption_agent/benchmarks/evidencebench_graph_evaluator_runner_v1.py"
)
EVALUATOR_TEST_RELATIVE = "tests/test_evidencebench_graph_evaluator_runner_v1.py"

PRIVATE_RELATIVE_PATHS = {
    "A_form": (
        "artifacts/evidencebench_direct_acquisition_v1/A_form.label_free.private.json",
        "artifacts/evidencebench_direct_acquisition_v1/A_form.labels.private.json",
    ),
    "F_search": (
        "artifacts/evidencebench_direct_acquisition_v1/F_search.label_free.private.json",
        "artifacts/evidencebench_direct_acquisition_v1/F_search.labels.sealed.json",
    ),
    "A_hold": (
        "artifacts/evidencebench_direct_acquisition_v1/A_hold.label_free.sealed.json",
        "artifacts/evidencebench_direct_acquisition_v1/A_hold.labels.sealed.json",
    ),
    "M_search": (
        "artifacts/evidencebench_direct_acquisition_v1/M_search.label_free.sealed.json",
        "artifacts/evidencebench_direct_acquisition_v1/M_search.labels.sealed.json",
    ),
}

REQUIRED_FREEZE_ROLES = frozenset(
    {
        "design",
        "custody",
        "source_access",
        "graph_core",
        "acquisition_runner",
        "evaluator_runner",
        "acquisition_test",
        "graph_core_test",
        "evaluator_test",
    }
)
FIXED_ROLE_PATHS = {
    "design": DESIGN_RELATIVE,
    "custody": CUSTODY_RELATIVE,
    "source_access": SOURCE_ACCESS_RELATIVE,
    "graph_core": GRAPH_CORE_RELATIVE,
    "acquisition_runner": ACQUISITION_RELATIVE,
    "evaluator_runner": EVALUATOR_RUNNER_RELATIVE,
    "acquisition_test": ACQUISITION_TEST_RELATIVE,
    "graph_core_test": GRAPH_CORE_TEST_RELATIVE,
    "evaluator_test": EVALUATOR_TEST_RELATIVE,
}
EXPECTED_FREEZE_INTERFACES: dict[str, dict[str, str]] = {
    "design": {
        "relative_path": DESIGN_RELATIVE,
        "schema": DESIGN_SCHEMA,
    },
    "custody": {
        "relative_path": CUSTODY_RELATIVE,
        "schema": CUSTODY_SCHEMA,
    },
    "source_access": {
        "relative_path": SOURCE_ACCESS_RELATIVE,
        "schema": SOURCE_ACCESS_SCHEMA,
    },
    "graph_core": {
        "relative_path": GRAPH_CORE_RELATIVE,
        "version": "evidencebench_typed_scientific_graph_v1",
    },
    "graph_core_test": {"relative_path": GRAPH_CORE_TEST_RELATIVE},
    "acquisition_runner": {
        "relative_path": ACQUISITION_RELATIVE,
        "version": VERSION,
    },
    "acquisition_test": {"relative_path": ACQUISITION_TEST_RELATIVE},
    "evaluator_runner": {
        "relative_path": EVALUATOR_RUNNER_RELATIVE,
        "version": "evidencebench_graph_evaluator_runner_v1",
    },
    "evaluator_test": {"relative_path": EVALUATOR_TEST_RELATIVE},
}

LOCAL_REASON_ORDER = (
    "paper_not_object",
    "paper_schema",
    "invalid_unicode",
    "sentence_count",
    "node_empty",
    "exposure_identifier",
    "aspect_list_schema",
    "aspect_id_schema",
    "aspect_id_duplicate",
    "aspect_map_schema",
    "aspect_map_key_mismatch",
    "aspect_sentence_list_schema",
    "aspect_sentence_index_schema",
    "aspect_sentence_index_bounds",
)

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_EXPOSED_PMCID_RE = re.compile(r"(?<![a-z0-9])pmc5533284(?![a-z0-9])")
_EXPOSED_DOI_RE = re.compile(
    r"(?<![a-z0-9])10\.1158/1055-9965\.epi-16-0219(?![a-z0-9])"
)
_FORMAL_ENTRY_ACTIVE = False


class EvidenceBenchAcquisitionError(RuntimeError):
    """A frozen acquisition, source, or output invariant was violated."""


class LocalPaperError(ValueError):
    """A paper-local exclusion which is counted rather than root-aborting."""

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
class PaperIdentity:
    source_ordinal: int
    paper_group_key: str
    sentence_list_sha256: str
    paper_commitment_sha256: str
    item_commitment_sha256: str
    paper_id: str
    hypothesis: str
    sentences: tuple[str, ...]


@dataclass(frozen=True)
class LabelFreePaper:
    source_ordinal: int
    paper_group_key: str
    sentence_list_sha256: str
    paper_commitment_sha256: str
    item_commitment_sha256: str
    paper_id: str
    hypothesis: str
    nodes: tuple[SourceNode, ...]
    sentence_count: int


@dataclass(frozen=True)
class EligibleItem:
    component_commitment_sha256: str
    paper_commitment_sha256: str
    item_commitment_sha256: str
    hypothesis: str
    nodes: tuple[SourceNode, ...]
    gold_aspect_node_indices: tuple[tuple[int, ...], ...]
    sentence_count: int


@dataclass(frozen=True)
class SourceBinding:
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
        raise EvidenceBenchAcquisitionError("value is not canonical JSON") from exc


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


def _strict_unicode(value: str, *, reason: str = "invalid_unicode") -> str:
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise LocalPaperError(reason) from exc
    return value


def identifier_normalize(value: str) -> str:
    """Normalize only explicit identifier/URL metadata for exposure checks."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = " ".join(normalized.split())
    return normalized[:-1] if normalized.endswith("/") else normalized


def _paper_exposure_reason(record: Mapping[str, Any]) -> str | None:
    # Never scan paper text: another paper may merely cite the exposed DOI.
    for field in EXPOSURE_METADATA_FIELDS:
        value = record.get(field)
        if not isinstance(value, str):
            continue
        normalized = identifier_normalize(value)
        if (
            _EXPOSED_PMCID_RE.search(normalized) is not None
            or _EXPOSED_DOI_RE.search(normalized) is not None
            or normalized == identifier_normalize(EXPOSED_URL)
        ):
            return "exposure_identifier"
    return None


def balanced_nodes(sentences: Sequence[Any]) -> tuple[SourceNode, ...]:
    """Partition ordered sentences into exactly 32 balanced contiguous nodes."""

    if isinstance(sentences, (str, bytes)) or not isinstance(sentences, Sequence):
        raise LocalPaperError("paper_schema")
    count = len(sentences)
    if count < MIN_SENTENCES:
        raise LocalPaperError("sentence_count")
    exact: list[str] = []
    for sentence in sentences:
        if not isinstance(sentence, str):
            raise LocalPaperError("paper_schema")
        exact.append(_strict_unicode(sentence))
    nodes: list[SourceNode] = []
    for span_i in range(NODE_COUNT):
        start = span_i * count // NODE_COUNT
        end = (span_i + 1) * count // NODE_COUNT
        if end <= start:
            raise AssertionError("n>=48 must make every balanced bucket nonempty")
        identity_text = "\n".join(exact[start:end])
        if not identity_text.strip():
            raise LocalPaperError("node_empty")
        nodes.append(SourceNode(span_i, start, end, identity_text))
    return tuple(nodes)


def map_sentence_indices_to_nodes(
    sentence_indices: Sequence[Any], *, nodes: Sequence[SourceNode], sentence_count: int
) -> tuple[int, ...]:
    if isinstance(sentence_indices, (str, bytes)) or not isinstance(
        sentence_indices, Sequence
    ):
        raise LocalPaperError("aspect_sentence_list_schema")
    if not sentence_indices:
        raise LocalPaperError("aspect_sentence_list_schema")
    sentence_to_node = [-1] * sentence_count
    for node in nodes:
        for sentence_index in range(node.start, node.end):
            sentence_to_node[sentence_index] = node.span_i
    if any(value < 0 for value in sentence_to_node):
        raise AssertionError("balanced partition did not cover every sentence")
    result: set[int] = set()
    for index in sentence_indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise LocalPaperError("aspect_sentence_index_schema")
        if index < 0 or index >= sentence_count:
            raise LocalPaperError("aspect_sentence_index_bounds")
        result.add(sentence_to_node[index])
    if not result:
        raise LocalPaperError("aspect_sentence_list_schema")
    return tuple(sorted(result))


def _paper_group_key(paper_id: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", paper_id).casefold().split())
    if not normalized:
        raise LocalPaperError("paper_schema")
    return normalized


def _parse_paper_identity(
    record: Mapping[str, Any], *, source_ordinal: int
) -> PaperIdentity:
    """Parse label-blind grouping fields without applying node eligibility."""

    paper_id = record.get("paper_id")
    hypothesis = record.get("hypothesis")
    sentences = record.get("paper_as_candidate_pool")
    if (
        not isinstance(paper_id, str)
        or not paper_id.strip()
        or not isinstance(hypothesis, str)
        or not hypothesis.strip()
    ):
        raise LocalPaperError("paper_schema")
    _strict_unicode(paper_id)
    _strict_unicode(hypothesis)
    if isinstance(sentences, (str, bytes)) or not isinstance(sentences, Sequence):
        raise LocalPaperError("paper_schema")
    exact_sentences: list[str] = []
    for sentence in sentences:
        if not isinstance(sentence, str):
            raise LocalPaperError("paper_schema")
        exact_sentences.append(_strict_unicode(sentence))
    group_key = _paper_group_key(paper_id)
    sentence_commitment = _semantic_hash(exact_sentences)
    paper_commitment = _semantic_hash(
        [
            "evidencebench_direct_v1",
            "paper",
            _sha256_bytes(paper_id.encode("utf-8")),
            sentence_commitment,
        ]
    )
    item_commitment = _semantic_hash(
        [
            "evidencebench_direct_v1",
            "item",
            paper_commitment,
            _sha256_bytes(hypothesis.encode("utf-8")),
        ]
    )
    return PaperIdentity(
        source_ordinal=source_ordinal,
        paper_group_key=group_key,
        sentence_list_sha256=sentence_commitment,
        paper_commitment_sha256=paper_commitment,
        item_commitment_sha256=item_commitment,
        paper_id=paper_id,
        hypothesis=hypothesis,
        sentences=tuple(exact_sentences),
    )


def _materialize_label_free_paper(identity: PaperIdentity) -> LabelFreePaper:
    nodes = balanced_nodes(identity.sentences)
    return LabelFreePaper(
        source_ordinal=identity.source_ordinal,
        paper_group_key=identity.paper_group_key,
        sentence_list_sha256=identity.sentence_list_sha256,
        paper_commitment_sha256=identity.paper_commitment_sha256,
        item_commitment_sha256=identity.item_commitment_sha256,
        paper_id=identity.paper_id,
        hypothesis=identity.hypothesis,
        nodes=nodes,
        sentence_count=len(identity.sentences),
    )


def _parse_label_free_paper(
    record: Mapping[str, Any], *, source_ordinal: int
) -> LabelFreePaper:
    """Parse label-free paper/query/nodes without reading any aspect label."""

    return _materialize_label_free_paper(
        _parse_paper_identity(record, source_ordinal=source_ordinal)
    )


def _aspect_key(value: Any) -> str:
    if isinstance(value, bool):
        raise LocalPaperError("aspect_id_schema")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value:
        _strict_unicode(value)
        return value
    raise LocalPaperError("aspect_id_schema")


def _parse_aspect_labels(
    record: Mapping[str, Any], *, candidate: LabelFreePaper
) -> tuple[tuple[int, ...], ...]:
    """Isolated label controller; all official aspects must validate."""

    aspect_ids = record.get("aspect_list_ids")
    aspect_map = record.get("aspect2sentence_indices")
    if (
        isinstance(aspect_ids, (str, bytes))
        or not isinstance(aspect_ids, Sequence)
        or not aspect_ids
    ):
        raise LocalPaperError("aspect_list_schema")
    if not isinstance(aspect_map, Mapping):
        raise LocalPaperError("aspect_map_schema")
    keys = [_aspect_key(value) for value in aspect_ids]
    if len(keys) != len(set(keys)):
        raise LocalPaperError("aspect_id_duplicate")
    if any(not isinstance(key, str) for key in aspect_map):
        raise LocalPaperError("aspect_map_schema")
    if set(aspect_map) != set(keys):
        raise LocalPaperError("aspect_map_key_mismatch")
    result: list[tuple[int, ...]] = []
    for key in keys:
        result.append(
            map_sentence_indices_to_nodes(
                aspect_map[key],
                nodes=candidate.nodes,
                sentence_count=candidate.sentence_count,
            )
        )
    return tuple(result)


def _hmac_array(secret: bytes, values: Sequence[str]) -> bytes:
    raw = json.dumps(
        list(values), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hmac.new(secret, raw, hashlib.sha256).digest()


def build_label_free_components(
    rows: Sequence[tuple[PaperIdentity, Mapping[str, Any]]]
) -> tuple[tuple[tuple[PaperIdentity, Mapping[str, Any]], ...], ...]:
    """DSU by normalized paper_id or exact ordered-sentence-list SHA256."""

    dsu = _DSU(len(rows))
    id_owner: dict[str, int] = {}
    content_owner: dict[str, int] = {}
    for index, (candidate, _record) in enumerate(rows):
        previous = id_owner.setdefault(candidate.paper_group_key, index)
        dsu.union(index, previous)
        previous = content_owner.setdefault(candidate.sentence_list_sha256, index)
        dsu.union(index, previous)
    groups: dict[int, list[tuple[PaperIdentity, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[dsu.find(index)].append(row)
    return tuple(
        tuple(sorted(group, key=lambda pair: pair[0].source_ordinal))
        for _minimum, group in sorted(
            (
                (min(pair[0].source_ordinal for pair in value), value)
                for value in groups.values()
            ),
            key=lambda pair: pair[0],
        )
    )


def component_commitment(
    rows: Sequence[tuple[PaperIdentity | LabelFreePaper, Mapping[str, Any]]]
) -> str:
    id_hashes = sorted(
        {
            _sha256_bytes(candidate.paper_group_key.encode("utf-8"))
            for candidate, _record in rows
        }
    )
    content_hashes = sorted(
        {candidate.sentence_list_sha256 for candidate, _record in rows}
    )
    return _semantic_hash(
        ["evidencebench_direct_v1", "component", id_hashes, content_hashes]
    )


def select_blocks_from_payload(
    payload: Any, *, secret: bytes
) -> tuple[dict[str, tuple[EligibleItem, ...]], dict[str, Any]]:
    """Select four blocks from an authorized in-memory synthetic root."""

    if not isinstance(secret, bytes) or len(secret) != 32:
        raise EvidenceBenchAcquisitionError("selection secret must be 32 raw bytes")
    if not isinstance(payload, list) or len(payload) != ROOT_RECORD_COUNT:
        raise EvidenceBenchAcquisitionError(
            "EvidenceBench root must be a list of exactly 293 papers"
        )

    reasons: Counter[str] = Counter()
    identity_rows: list[tuple[PaperIdentity, Mapping[str, Any]]] = []
    for source_ordinal, record in enumerate(payload):
        if not isinstance(record, Mapping):
            reasons["paper_not_object"] += 1
            continue
        try:
            identity = _parse_paper_identity(
                record, source_ordinal=source_ordinal
            )
        except LocalPaperError as exc:
            reasons[exc.reason] += 1
            continue
        identity_rows.append((identity, record))

    components = build_label_free_components(identity_rows)
    candidates: list[EligibleItem] = []
    sentence_histogram: Counter[int] = Counter()
    aspect_histogram: Counter[int] = Counter()
    aspect_bucket_histogram: Counter[int] = Counter()
    multi_record_component_count = 0
    component_size_histogram: Counter[int] = Counter()
    for rows in components:
        component_size_histogram[len(rows)] += 1
        if len(rows) > 1:
            multi_record_component_count += 1
        if any(
            _paper_exposure_reason(record) is not None
            for _candidate, record in rows
        ):
            reasons["exposure_identifier"] += 1
            continue
        commitment = component_commitment(rows)
        representative, record = min(
            rows,
            key=lambda pair: (
                _hmac_array(
                    secret,
                    [
                        "evidencebench_direct_v1",
                        "component_representative",
                        commitment,
                        pair[0].item_commitment_sha256,
                    ],
                ),
                pair[0].item_commitment_sha256,
                pair[0].source_ordinal,
            ),
        )
        try:
            candidate = _materialize_label_free_paper(representative)
            aspect_gold = _parse_aspect_labels(record, candidate=candidate)
        except LocalPaperError as exc:
            # One invalid aspect excludes the representative and therefore the
            # whole component; no aspect dropping, alternate representative,
            # runner-up, or partial-gold item is formed.
            reasons[exc.reason] += 1
            continue
        item = EligibleItem(
            component_commitment_sha256=commitment,
            paper_commitment_sha256=candidate.paper_commitment_sha256,
            item_commitment_sha256=candidate.item_commitment_sha256,
            hypothesis=candidate.hypothesis,
            nodes=candidate.nodes,
            gold_aspect_node_indices=aspect_gold,
            sentence_count=candidate.sentence_count,
        )
        candidates.append(item)
        sentence_histogram[item.sentence_count] += 1
        aspect_histogram[len(aspect_gold)] += 1
        for bucket_set in aspect_gold:
            aspect_bucket_histogram[len(bucket_set)] += 1

    candidates.sort(
        key=lambda item: (
            _hmac_array(
                secret,
                [
                    "evidencebench_direct_v1",
                    "component",
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
        "root_counts": {
            "declared_paper_records": len(payload),
            "label_blind_identity_valid_records": len(identity_rows),
            "constructed_paper_components": len(components),
            "multi_record_paper_components": multi_record_component_count,
            "paper_component_size_histogram": {
                str(key): component_size_histogram[key]
                for key in sorted(component_size_histogram)
            },
        },
        "paper_counts": {
            "eligible": len(candidates),
            "required": SELECTED_COUNT,
            "unused_eligible_after_selection": max(0, len(candidates) - SELECTED_COUNT),
            "capacity_satisfied": len(candidates) >= SELECTED_COUNT,
        },
        "parser_reason_counts": {
            reason: reasons[reason] for reason in LOCAL_REASON_ORDER
        },
        "exposure_counts": {
            "identifier_excluded_paper_components": reasons[
                "exposure_identifier"
            ]
        },
        "eligible_sentence_count_histogram": {
            str(key): sentence_histogram[key] for key in sorted(sentence_histogram)
        },
        "eligible_aspect_count_histogram": {
            str(key): aspect_histogram[key] for key in sorted(aspect_histogram)
        },
        "eligible_per_aspect_gold_bucket_cardinality_histogram": {
            str(key): aspect_bucket_histogram[key]
            for key in sorted(aspect_bucket_histogram)
        },
    }
    return blocks, stats


def _require_regular_file(path: Path, *, label: str, mode: int | None = None) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise EvidenceBenchAcquisitionError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise EvidenceBenchAcquisitionError(
            f"{label} must be a non-symlink regular file"
        )
    if mode is not None and stat.S_IMODE(metadata.st_mode) != mode:
        raise EvidenceBenchAcquisitionError(f"{label} must have mode {mode:04o}")


def hash_source_file(path: Path) -> SourceBinding:
    """The sole allowed pre-marker pass over the complete source bytes."""

    _require_regular_file(path, label="source file")
    size = path.stat().st_size
    if size > MAX_SOURCE_BYTES:
        raise EvidenceBenchAcquisitionError("source file exceeds frozen byte limit")
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1()
    git_blob.update(f"blob {size}\0".encode("ascii"))
    observed = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            observed += len(chunk)
            sha256.update(chunk)
            git_blob.update(chunk)
    if observed != size:
        raise EvidenceBenchAcquisitionError("source file changed while hashing")
    return SourceBinding(sha256.hexdigest(), size, git_blob.hexdigest())


def _read_bound_source(path: Path, initial: SourceBinding) -> bytes:
    """Read source bytes after marker and prove they match the preflight pass."""

    _require_regular_file(path, label="source file")
    chunks: list[bytes] = []
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1()
    git_blob.update(f"blob {initial.byte_size}\0".encode("ascii"))
    observed = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            observed += len(chunk)
            if observed > MAX_SOURCE_BYTES:
                raise EvidenceBenchAcquisitionError("source file exceeds frozen byte limit")
            sha256.update(chunk)
            git_blob.update(chunk)
            chunks.append(chunk)
    raw = b"".join(chunks)
    observed_binding = SourceBinding(
        sha256.hexdigest(), observed, git_blob.hexdigest()
    )
    if observed_binding != initial or len(raw) != initial.byte_size:
        raise EvidenceBenchAcquisitionError("source file changed after marker")
    return raw


def _strict_json(raw: bytes, *, label: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EvidenceBenchAcquisitionError(
                    f"{label} contains duplicate JSON keys"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise EvidenceBenchAcquisitionError(f"{label} contains non-finite JSON")

    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except EvidenceBenchAcquisitionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise EvidenceBenchAcquisitionError(f"{label} is not strict JSON") from exc


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
        raise EvidenceBenchAcquisitionError(
            "output parent must be a non-symlink directory"
        )


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
    raw = json.dumps(body, ensure_ascii=True, sort_keys=True, indent=2).encode(
        "utf-8"
    ) + b"\n"
    _atomic_write_exclusive(path, raw, mode=mode)
    return _sha256_bytes(raw), body


def consume_attempt_marker(
    path: Path, *, source_binding: SourceBinding, protocol_bindings: Mapping[str, Any]
) -> bytes:
    payload = _self_hashed_payload(
        {
            "schema": MARKER_SCHEMA,
            "source_file_sha256": source_binding.sha256,
            "source_file_byte_size": source_binding.byte_size,
            "source_git_blob_sha1": source_binding.git_blob_sha1,
            "protocol_bindings_sha256": _semantic_hash(protocol_bindings),
            "source_JSON_opened_or_parsed_before_marker": False,
            "retry_replay_resample_authorized": False,
        },
        "marker_sha256",
    )
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2).encode(
        "utf-8"
    ) + b"\n"
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
        raise EvidenceBenchAcquisitionError(
            "formal EvidenceBench acquisition attempt is already consumed"
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
        if block not in paths.private:
            raise EvidenceBenchAcquisitionError("private output map is incomplete")
        all_paths.extend(paths.private[block])
    for path in all_paths:
        _ensure_safe_parent(path)
        if path.exists() or path.is_symlink():
            raise EvidenceBenchAcquisitionError(
                f"one-shot output already exists: {path.name}"
            )


def _view_row(block: str, ordinal: int, item: EligibleItem) -> dict[str, Any]:
    return {
        "schema": LABEL_FREE_ITEM_SCHEMA,
        "block": block,
        "ordinal": ordinal,
        "item_commitment_sha256": item.item_commitment_sha256,
        "component_commitment_sha256": item.component_commitment_sha256,
        "paper_commitment_sha256": item.paper_commitment_sha256,
        "hypothesis": item.hypothesis,
        "title": "EvidenceBench_paper",
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
        # The list order is the local aspect ordinal.  Official aspect IDs are
        # never persisted, exposed to a runner, or published.
        "gold_aspect_node_indices": [
            list(indices) for indices in item.gold_aspect_node_indices
        ],
    }


def _private_envelope(
    *, schema: str, block: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return _self_hashed_payload(
        {"schema": schema, "block": block, "count": len(rows), "rows": list(rows)},
        "block_sha256",
    )


def _persist_private_blocks(
    blocks: Mapping[str, Sequence[EligibleItem]], paths: OutputPaths
) -> list[dict[str, Any]]:
    commitments: list[dict[str, Any]] = []
    for block in BLOCK_ORDER:
        items = blocks[block]
        if len(items) != BLOCK_SIZE:
            raise EvidenceBenchAcquisitionError("private block size drifted")
        views = [_view_row(block, ordinal, item) for ordinal, item in enumerate(items)]
        labels = [
            _label_row(block, ordinal, item) for ordinal, item in enumerate(items)
        ]
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
        commitments.append(
            {
                "block": block,
                "count": len(items),
                "item_commitment_root_sha256": _semantic_hash(
                    [item.item_commitment_sha256 for item in items]
                ),
                "label_free_file_sha256": _sha256_bytes(view_raw),
                "label_file_sha256": _sha256_bytes(label_raw),
                "label_free_block_sha256": view_payload["block_sha256"],
                "label_block_sha256": label_payload["block_sha256"],
            }
        )
    return commitments


def _assert_public_redacted(payload: Any) -> None:
    forbidden_keys = {
        "paper_id",
        "hypothesis",
        "paper_as_candidate_pool",
        "aspect_list_ids",
        "aspect2sentence_indices",
        "sentence_index2aspects",
        "nodes",
        "identity_text",
        "gold_aspect_node_indices",
        "item_commitment_sha256",
        "paper_commitment_sha256",
        "component_commitment_sha256",
        "rows",
    }

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            if forbidden_keys & set(value):
                raise EvidenceBenchAcquisitionError(
                    "public receipt contains a private field"
                )
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True).casefold()
    for exposed in (EXPOSED_PMCID, EXPOSED_DOI, EXPOSED_URL):
        if exposed.casefold() in serialized:
            raise EvidenceBenchAcquisitionError(
                "public receipt contains an exposure identifier"
            )


def _base_public_receipt(
    *,
    status: str,
    source_binding: SourceBinding,
    marker_raw: bytes,
    protocol_bindings: Mapping[str, Any],
    stats: Mapping[str, Any] | None,
) -> dict[str, Any]:
    marker_payload = _strict_json(marker_raw, label="attempt marker")
    return {
        "schema": PUBLIC_SCHEMA,
        "status": status,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "repository_fixed_commit": SOURCE_COMMIT,
            "repository_path": SOURCE_REPOSITORY_PATH,
            "repository_git_blob_sha1": source_binding.git_blob_sha1,
            "source_file_sha256": source_binding.sha256,
            "source_file_byte_size": source_binding.byte_size,
            "pre_marker_whole_file_hash_only_pass_count": 1,
            "post_marker_source_content_open_count": 1,
        },
        "attempt": {
            "marker_file_sha256": _sha256_bytes(marker_raw),
            "marker_sha256": marker_payload["marker_sha256"],
            "marker_durable_before_source_JSON_open": True,
            "preregistered_formal_invocation_count": 1,
            "observed_marker_consuming_attempt_count": 1,
            "attempt_marker_creation_count": 1,
            "source_JSON_open_attempt_count": 1,
            "same_source_replay_count": 0,
            "resample_count": 0,
            "secret_rotation_count": 0,
            "parser_or_model_worker_count": 0,
            "readonly_git_metadata_subprocess_count": (
                protocol_bindings.get("git_HEAD", {}).get(
                    "readonly_git_metadata_subprocess_count", 0
                )
                if isinstance(protocol_bindings.get("git_HEAD"), Mapping)
                else 0
            ),
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
    source_binding: SourceBinding,
    marker_raw: bytes,
    protocol_bindings: Mapping[str, Any],
    stage: str,
    error: BaseException,
) -> None:
    failure = {
        "schema": FAILURE_SCHEMA,
        "stage": stage,
        "exception_type": f"{type(error).__module__}.{type(error).__qualname__}",
        "exception_message_sha256": _sha256_bytes(str(error).encode("utf-8")),
        "marker_file_sha256": _sha256_bytes(marker_raw),
        "same_source_replay_authorized": False,
    }
    try:
        _write_json_exclusive(
            paths.failure, failure, hash_field="failure_sha256", mode=0o600
        )
    except BaseException:
        pass
    receipt = _base_public_receipt(
        status="terminal_infrastructure_invalid",
        source_binding=source_binding,
        marker_raw=marker_raw,
        protocol_bindings=protocol_bindings,
        stats={
            "failure_stage": stage,
            "exception_type_sha256": _sha256_bytes(
                f"{type(error).__module__}.{type(error).__qualname__}".encode(
                    "utf-8"
                )
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
    source_path: Path,
    source_binding: SourceBinding,
    secret: bytes,
    protocol_bindings: Mapping[str, Any],
    paths: OutputPaths,
) -> dict[str, Any]:
    """One marker-consuming engine; source parsing starts after the marker."""

    _preflight_outputs(paths)
    marker_raw = consume_attempt_marker(
        paths.marker,
        source_binding=source_binding,
        protocol_bindings=protocol_bindings,
    )
    stage = "read_bound_source_file"
    try:
        raw = _read_bound_source(source_path, source_binding)
        stage = "parse_strict_EvidenceBench_JSON"
        payload = _strict_json(raw, label="EvidenceBench original-test source")
        stage = "form_paper_disjoint_selection"
        blocks, stats = select_blocks_from_payload(payload, secret=secret)
        if not stats["paper_counts"]["capacity_satisfied"]:
            receipt = _base_public_receipt(
                status="terminal_source_capacity_insufficient",
                source_binding=source_binding,
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
            source_binding=source_binding,
            marker_raw=marker_raw,
            protocol_bindings=protocol_bindings,
            stats=stats,
        )
        receipt["blocks"] = {
            "block_order": list(BLOCK_ORDER),
            "block_size": BLOCK_SIZE,
            "selected_item_count": SELECTED_COUNT,
            "global_paper_disjointness": True,
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
            source_binding=source_binding,
            marker_raw=marker_raw,
            protocol_bindings=protocol_bindings,
            stage=stage,
            error=exc,
        )
        raise


def _read_self_hashed_manifest(
    path: Path, *, schema: str, hash_field: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_regular_file(path, label=schema)
    raw = path.read_bytes()
    payload = _strict_json(raw, label=schema)
    if not isinstance(payload, Mapping) or payload.get("schema") != schema:
        raise EvidenceBenchAcquisitionError(f"{schema} schema mismatch")
    declared = payload.get(hash_field)
    if not isinstance(declared, str) or _HEX64.fullmatch(declared) is None:
        raise EvidenceBenchAcquisitionError(f"{schema} self-hash is missing")
    body = dict(payload)
    del body[hash_field]
    observed = _semantic_hash(body)
    if observed != declared:
        raise EvidenceBenchAcquisitionError(f"{schema} self-hash mismatch")
    return dict(payload), {
        "schema": schema,
        "semantic_sha256": observed,
        "file_sha256": _sha256_bytes(raw),
        "byte_size": len(raw),
    }


def _safe_protocol_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise EvidenceBenchAcquisitionError("freeze protocol path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise EvidenceBenchAcquisitionError("freeze protocol path is unsafe")
    return path.as_posix()


def _git_repository_root(project: Path) -> Path:
    candidate = project.resolve(strict=True)
    for root in (candidate, *candidate.parents):
        marker = root / ".git"
        if (marker.is_dir() or marker.is_file()) and not marker.is_symlink():
            return root
    raise EvidenceBenchAcquisitionError("formal project is not in a Git repository")


def _git_blob_oid(raw: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {len(raw)}\0".encode("ascii"))
    digest.update(raw)
    return digest.hexdigest()


def verify_protocol_files_at_head(
    project: Path, relative_paths: Sequence[str]
) -> dict[str, Any]:
    """Use only fixed read-only Git metadata commands on public protocol paths."""

    root = project.resolve(strict=True)
    repository_root = _git_repository_root(root)
    relative_project = root.relative_to(repository_root)
    safe_paths = tuple(_safe_protocol_relative_path(value) for value in relative_paths)
    if len(safe_paths) != len(set(safe_paths)):
        raise EvidenceBenchAcquisitionError("freeze protocol paths are duplicated")
    repository_paths = tuple(
        (PurePosixPath(relative_project.as_posix()) / path).as_posix()
        for path in safe_paths
    )
    try:
        head_result = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "--verify", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        tree_result = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "ls-tree",
                "-r",
                "-z",
                "HEAD",
                "--",
                *repository_paths,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise EvidenceBenchAcquisitionError(
            "read-only Git protocol verification failed"
        ) from exc
    try:
        head = head_result.stdout.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise EvidenceBenchAcquisitionError("Git HEAD output is malformed") from exc
    if _HEX40.fullmatch(head) is None:
        raise EvidenceBenchAcquisitionError("Git HEAD output is malformed")
    observed: dict[str, tuple[str, str]] = {}
    for raw_record in tree_result.stdout.split(b"\0"):
        if not raw_record:
            continue
        metadata, separator, raw_path = raw_record.partition(b"\t")
        fields = metadata.split(b" ")
        if not separator or len(fields) != 3:
            raise EvidenceBenchAcquisitionError("Git ls-tree output is malformed")
        mode, kind, oid = fields
        try:
            decoded_path = raw_path.decode("utf-8", errors="strict")
            decoded_oid = oid.decode("ascii", errors="strict")
        except UnicodeDecodeError as exc:
            raise EvidenceBenchAcquisitionError(
                "Git ls-tree output is malformed"
            ) from exc
        if (
            kind != b"blob"
            or mode in {b"120000", b"160000"}
            or _HEX40.fullmatch(decoded_oid) is None
            or decoded_path in observed
        ):
            raise EvidenceBenchAcquisitionError("Git protocol entry is not a regular blob")
        observed[decoded_path] = (mode.decode("ascii"), decoded_oid)
    if set(observed) != set(repository_paths):
        raise EvidenceBenchAcquisitionError(
            "freeze-listed protocol files are not exactly present at Git HEAD"
        )
    for relative, repository_relative in zip(
        safe_paths, repository_paths, strict=True
    ):
        path = root / relative
        _require_regular_file(path, label="freeze-listed protocol file")
        if observed[repository_relative][1] != _git_blob_oid(path.read_bytes()):
            raise EvidenceBenchAcquisitionError(
                "freeze-listed protocol file does not byte-match Git HEAD"
            )
    return {
        "head_commit": head,
        "verified_file_count": len(safe_paths),
        "all_freeze_listed_files_byte_match_HEAD": True,
        "readonly_git_metadata_subprocess_count": 2,
        "source_secret_private_or_output_path_passed_to_git": False,
    }


def _validate_implementation_freeze(
    payload: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], str, str]:
    expected_top_keys = {
        "schema",
        "bindings",
        "source_binding",
        "selection_secret_commitment",
        "freeze_hash_contract",
        "implementation_freeze_sha256",
    }
    if set(payload) != expected_top_keys:
        raise EvidenceBenchAcquisitionError(
            "implementation freeze top-level schema drifted"
        )
    hash_contract = payload.get("freeze_hash_contract")
    if not isinstance(hash_contract, Mapping) or (
        hash_contract.get("algorithm") != "sha256"
        or hash_contract.get("excluded_top_level_fields")
        != ["implementation_freeze_sha256"]
    ):
        raise EvidenceBenchAcquisitionError(
            "implementation freeze hash contract drifted"
        )
    source = payload.get("source_binding")
    expected_source = {
        "repository": SOURCE_REPOSITORY,
        "commit": SOURCE_COMMIT,
        "relative_path": SOURCE_REPOSITORY_PATH,
        "raw_url": SOURCE_RAW_URL,
        "git_blob_sha1": SOURCE_GIT_BLOB_SHA1,
        "byte_size": SOURCE_BYTE_SIZE,
    }
    if not isinstance(source, Mapping) or any(
        source.get(key) != value for key, value in expected_source.items()
    ):
        raise EvidenceBenchAcquisitionError("implementation freeze source identity drifted")
    source_sha256 = source.get("whole_file_sha256")
    secret_sha256 = payload.get("selection_secret_commitment")
    if (
        not isinstance(source_sha256, str)
        or _HEX64.fullmatch(source_sha256) is None
        or not isinstance(secret_sha256, str)
        or _HEX64.fullmatch(secret_sha256) is None
    ):
        raise EvidenceBenchAcquisitionError(
            "implementation freeze source or secret hash is missing"
        )
    files = payload.get("bindings")
    if not isinstance(files, Mapping):
        raise EvidenceBenchAcquisitionError("implementation freeze bindings are missing")
    if set(files) != REQUIRED_FREEZE_ROLES:
        raise EvidenceBenchAcquisitionError("implementation freeze role set is incomplete")
    entries: dict[str, dict[str, Any]] = {}
    for role, entry in files.items():
        if not isinstance(entry, Mapping):
            raise EvidenceBenchAcquisitionError("implementation freeze file entry is invalid")
        if not isinstance(role, str) or role in entries:
            raise EvidenceBenchAcquisitionError("implementation freeze file role is invalid")
        relative = _safe_protocol_relative_path(entry.get("relative_path"))
        file_sha256 = entry.get("file_sha256")
        git_blob_sha1 = entry.get("git_blob_sha1")
        if (
            not isinstance(file_sha256, str)
            or _HEX64.fullmatch(file_sha256) is None
            or not isinstance(git_blob_sha1, str)
            or _HEX40.fullmatch(git_blob_sha1) is None
        ):
            raise EvidenceBenchAcquisitionError(
                "implementation freeze file hashes or identity are invalid"
            )
        entries[role] = dict(entry)
        entries[role]["relative_path"] = relative
    for role, interface in EXPECTED_FREEZE_INTERFACES.items():
        entry = entries[role]
        if entry["relative_path"] != interface["relative_path"]:
            raise EvidenceBenchAcquisitionError(
                f"implementation freeze {role} path drifted"
            )
        if "schema" in interface and entry.get("schema") != interface["schema"]:
            raise EvidenceBenchAcquisitionError(
                f"implementation freeze {role} schema drifted"
            )
        if "version" in interface and entry.get("version") != interface["version"]:
            raise EvidenceBenchAcquisitionError(
                f"implementation freeze {role} version drifted"
            )
    return entries, source_sha256, secret_sha256


def verify_formal_protocol(
    *, project: Path, source: Path, selection_secret: Path
) -> tuple[SourceBinding, bytes, dict[str, Any]]:
    """Verify the external freeze, Git HEAD, secret, and source hashes."""

    freeze, freeze_binding = _read_self_hashed_manifest(
        project / IMPLEMENTATION_FREEZE_RELATIVE,
        schema=IMPLEMENTATION_FREEZE_SCHEMA,
        hash_field="implementation_freeze_sha256",
    )
    entries, source_sha256, secret_sha256 = _validate_implementation_freeze(freeze)
    head_binding = verify_protocol_files_at_head(
        project,
        (IMPLEMENTATION_FREEZE_RELATIVE,)
        + tuple(entries[role]["relative_path"] for role in sorted(entries)),
    )

    protocol_files: dict[str, dict[str, Any]] = {}
    for role, entry in entries.items():
        path = project / entry["relative_path"]
        raw = path.read_bytes()
        if (
            _sha256_bytes(raw) != entry["file_sha256"]
            or _git_blob_oid(raw) != entry["git_blob_sha1"]
        ):
            raise EvidenceBenchAcquisitionError(
                f"freeze-listed {role} file hash drifted"
            )
        protocol_files[role] = {
            "relative_path": entry["relative_path"],
            "file_sha256": entry["file_sha256"],
            "git_blob_sha1": entry["git_blob_sha1"],
        }
        if "schema" in entry:
            protocol_files[role]["schema"] = entry["schema"]
        if "version" in entry:
            protocol_files[role]["version"] = entry["version"]

    manifest_specs = {
        "design": (DESIGN_SCHEMA, "design_sha256"),
        "custody": (CUSTODY_SCHEMA, "custody_sha256"),
        "source_access": (SOURCE_ACCESS_SCHEMA, "source_access_sha256"),
    }
    for role, (schema, hash_field) in manifest_specs.items():
        _manifest, binding = _read_self_hashed_manifest(
            project / entries[role]["relative_path"],
            schema=schema,
            hash_field=hash_field,
        )
        declared_semantic = entries[role].get("semantic_sha256")
        if (
            not isinstance(declared_semantic, str)
            or declared_semantic != binding["semantic_sha256"]
        ):
            raise EvidenceBenchAcquisitionError(
                f"freeze-listed {role} semantic hash drifted"
            )
        protocol_files[role]["semantic_sha256"] = declared_semantic

    _require_regular_file(selection_secret, label="selection secret", mode=0o600)
    secret = selection_secret.read_bytes()
    if len(secret) != 32 or _sha256_bytes(secret) != secret_sha256:
        raise EvidenceBenchAcquisitionError("selection secret binding drifted")

    binding = hash_source_file(source)
    if (
        binding.byte_size != SOURCE_BYTE_SIZE
        or binding.git_blob_sha1 != SOURCE_GIT_BLOB_SHA1
        or binding.sha256 != source_sha256
    ):
        raise EvidenceBenchAcquisitionError("official source binding drifted")
    protocol = {
        "implementation_freeze": freeze_binding,
        "git_HEAD": head_binding,
        "protocol_files": protocol_files,
        "source_identity": {
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "path": SOURCE_REPOSITORY_PATH,
            "git_blob_sha1": SOURCE_GIT_BLOB_SHA1,
            "byte_size": SOURCE_BYTE_SIZE,
        },
        "pre_marker_real_row_or_JSON_access": False,
    }
    return binding, secret, protocol


def formal_acquire(
    *,
    project: Path,
    source_path: Path,
    selection_secret_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run the unique direct acquisition in this invoking parent process."""

    if _FORMAL_ENTRY_ACTIVE is not True:
        raise EvidenceBenchAcquisitionError(
            "official row access is available only through --formal"
        )
    root = project.resolve(strict=True)
    expected_source = (root / SOURCE_RELATIVE).absolute()
    expected_secret = (root / SECRET_RELATIVE).absolute()
    expected_output = (root / PUBLIC_RECEIPT_RELATIVE).absolute()
    if (
        source_path.absolute() != expected_source
        or selection_secret_path.absolute() != expected_secret
        or output_path.absolute() != expected_output
    ):
        raise EvidenceBenchAcquisitionError(
            "formal inputs must use their canonical frozen paths"
        )
    binding, secret, protocol = verify_formal_protocol(
        project=root,
        source=expected_source,
        selection_secret=expected_secret,
    )
    return execute_acquisition_once(
        source_path=expected_source,
        source_binding=binding,
        secret=secret,
        protocol_bindings=protocol,
        paths=_default_output_paths(root),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", action="store_true", required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--selection-secret", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if not arguments.formal:
        raise EvidenceBenchAcquisitionError("no nonformal source-loading mode exists")
    global _FORMAL_ENTRY_ACTIVE
    _FORMAL_ENTRY_ACTIVE = True
    try:
        formal_acquire(
            project=arguments.project,
            source_path=arguments.source,
            selection_secret_path=arguments.selection_secret,
            output_path=arguments.output,
        )
    finally:
        _FORMAL_ENTRY_ACTIVE = False
    return 0


__all__ = [
    "BLOCK_ORDER",
    "BLOCK_SIZE",
    "EligibleItem",
    "EvidenceBenchAcquisitionError",
    "EXPOSED_DOI",
    "EXPOSED_PMCID",
    "EXPOSED_URL",
    "IMPLEMENTATION_FREEZE_RELATIVE",
    "IMPLEMENTATION_FREEZE_SCHEMA",
    "LABEL_BLOCK_SCHEMA",
    "LABEL_FREE_BLOCK_SCHEMA",
    "NODE_COUNT",
    "OutputPaths",
    "ROOT_RECORD_COUNT",
    "SELECTED_COUNT",
    "SOURCE_BYTE_SIZE",
    "SOURCE_COMMIT",
    "SOURCE_GIT_BLOB_SHA1",
    "SOURCE_RAW_URL",
    "SOURCE_REPOSITORY",
    "SOURCE_REPOSITORY_PATH",
    "SourceBinding",
    "SourceNode",
    "balanced_nodes",
    "execute_acquisition_once",
    "formal_acquire",
    "hash_source_file",
    "identifier_normalize",
    "map_sentence_indices_to_nodes",
    "select_blocks_from_payload",
]


if __name__ == "__main__":
    raise SystemExit(main())
