"""Non-scoring public-source qualification for the BioASQ P1 study.

This module is intentionally source-only and stdlib-only.  It has no model,
retrieval, action, evaluator, score, API, network, or cohort-selection
surface.  A production qualification:

* opens the bound source file exactly once;
* hashes the bytes during that one read and verifies the frozen identity;
* decodes strict UTF-8 JSON exactly once while rejecting duplicate keys;
* validates the four source-native question families;
* joins questions through normalized query, gold-document, or gold-snippet
  commitments;
* proves simultaneous capacity for 56 component-disjoint questions per
  family without selecting a cohort; and
* emits only a safe aggregate receipt plus a private, non-cohort commitment
  manifest.

``run_source_free_canary`` constructs synthetic bytes in memory and invokes
the same ``_qualify_decoded_source`` production parser/component entrypoint.
It never accepts or opens a formal source path.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, BinaryIO, Callable, Mapping, Sequence
import unicodedata


VERSION = "bioasq_p0_public_source_qualification_v1"
STUDY_ID = "BIOASQ_P1_TYPED_QUESTION_EVIDENCE_EVALUATOR_L5_V1"
FAMILIES = ("yesno", "factoid", "list", "summary")
DEFAULT_MINIMUM_COMPONENTS_PER_FAMILY = 56
OFFICIAL_QUESTION_COUNT = 4_719

SAFE_RECEIPT_SCHEMA = f"{VERSION}_safe_receipt_v1"
PRIVATE_MANIFEST_SCHEMA = f"{VERSION}_private_noncohort_manifest_v1"
SOURCE_FREE_CANARY_SCHEMA = f"{VERSION}_source_free_canary_receipt_v1"
COMPONENT_RULE = (
    "union_by_normalized_query_or_gold_document_or_normalized_gold_snippet_v1"
)

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_SOURCE_BYTES = 1_000_000_000
_MAX_QUESTION_COUNT = 1_000_000
_MAX_TEXT_CHARACTERS = 10_000_000
_MAX_IDENTIFIER_CHARACTERS = 100_000


class BioasqP0QualificationError(RuntimeError):
    """The frozen source identity, schema, or output contract drifted."""


class _DuplicateJsonKey(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SourceFileContract:
    """Exact byte identity required before semantic JSON decoding."""

    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.size_bytes) is not int
            or not 1 <= self.size_bytes <= _MAX_SOURCE_BYTES
            or not isinstance(self.sha256, str)
            or _HEX64.fullmatch(self.sha256) is None
        ):
            raise BioasqP0QualificationError(
                "source file contract is invalid"
            )


@dataclass(frozen=True, slots=True)
class QualificationContract:
    """Frozen simultaneous component demand for the four families."""

    minimum_components_per_family: Mapping[str, int]
    expected_question_count: int

    def __post_init__(self) -> None:
        if (
            tuple(self.minimum_components_per_family) != FAMILIES
            or any(
                type(self.minimum_components_per_family[family]) is not int
                or self.minimum_components_per_family[family] < 1
                for family in FAMILIES
            )
            or type(self.expected_question_count) is not int
            or not 1 <= self.expected_question_count <= _MAX_QUESTION_COUNT
        ):
            raise BioasqP0QualificationError(
                "qualification family demand contract drifted"
            )


DEFAULT_CONTRACT = QualificationContract(
    {
        family: DEFAULT_MINIMUM_COMPONENTS_PER_FAMILY
        for family in FAMILIES
    },
    expected_question_count=OFFICIAL_QUESTION_COUNT,
)


@dataclass(frozen=True, slots=True)
class QualificationResult:
    safe_receipt: Mapping[str, object]
    private_manifest: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _Question:
    opaque_item_commitment: str
    family: str
    query_commitment: str
    document_commitments: tuple[str, ...]
    snippet_commitments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PrivateRow:
    opaque_item_commitment: str
    family: str
    component_commitment: str
    query_commitment: str
    snippet_commitments: tuple[str, ...]

    def payload(self) -> dict[str, object]:
        return {
            "component_commitment": self.component_commitment,
            "family": self.family,
            "opaque_item_commitment": self.opaque_item_commitment,
            "query_commitment": self.query_commitment,
            "snippet_commitments": list(self.snippet_commitments),
        }


@dataclass(slots=True)
class _FlowEdge:
    to: int
    reverse: int
    capacity: int


@dataclass(slots=True)
class _AccessAudit:
    stage: str = "validate_formal_arguments"
    source_open_count: int = 0
    source_hash_count: int = 0
    source_json_decode_count: int = 0
    real_source_access_count: int = 0

    def payload(self) -> dict[str, int]:
        return {
            "real_source_access_count": self.real_source_access_count,
            "source_hash_count": self.source_hash_count,
            "source_json_decode_count": self.source_json_decode_count,
            "source_open_count": self.source_open_count,
        }


def canonical_bytes(value: object, *, newline: bool = True) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise BioasqP0QualificationError(
            "qualification value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value, newline=False)).hexdigest()


def self_hashed(body: Mapping[str, object]) -> dict[str, object]:
    if "self_sha256" in body:
        raise BioasqP0QualificationError(
            "self hash was already present"
        )
    result = dict(body)
    result["self_sha256"] = stable_hash(result)
    return result


def _normalize_text(
    value: object,
    *,
    field: str,
    casefold: bool,
    allow_empty: bool = False,
    maximum_length: int = _MAX_TEXT_CHARACTERS,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) > maximum_length
        or "\x00" in value
    ):
        raise BioasqP0QualificationError(f"{field} schema drifted")
    normalized = unicodedata.normalize("NFKC", value)
    if casefold:
        normalized = normalized.casefold()
    normalized = " ".join(normalized.split())
    if not allow_empty and not normalized:
        raise BioasqP0QualificationError(f"{field} schema drifted")
    return normalized


def _commit(kind: str, value: str) -> str:
    return stable_hash(
        {
            "kind": kind,
            "normalized_value": value,
            "version": VERSION,
        }
    )


def _reject_constant(_value: str) -> None:
    raise BioasqP0QualificationError(
        "source JSON contains a non-finite number"
    )


def _no_duplicate_object(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey
        result[key] = value
    return result


def _decode_strict_json(raw: bytes) -> object:
    """Decode one strict JSON value; this is the sole production decoder."""

    if not isinstance(raw, bytes) or not raw:
        raise BioasqP0QualificationError("source bytes are invalid")
    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_constant,
        )
    except _DuplicateJsonKey as exc:
        raise BioasqP0QualificationError(
            "source JSON contains a duplicate object key"
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BioasqP0QualificationError(
            "source is not strict UTF-8 JSON"
        ) from exc


def _family(value: object) -> str:
    if not isinstance(value, str):
        raise BioasqP0QualificationError(
            "question family schema drifted"
        )
    normalized = " ".join(value.strip().casefold().split())
    if normalized not in FAMILIES:
        raise BioasqP0QualificationError(
            "question family is outside the frozen four-family registry"
        )
    return normalized


def _string_list(
    value: object,
    *,
    field: str,
    casefold: bool,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise BioasqP0QualificationError(f"{field} schema drifted")
    rows = tuple(
        _normalize_text(
            row,
            field=field,
            casefold=casefold,
            allow_empty=False,
            maximum_length=_MAX_IDENTIFIER_CHARACTERS,
        )
        for row in value
    )
    if len(set(rows)) != len(rows):
        raise BioasqP0QualificationError(
            f"{field} contains duplicate normalized values"
        )
    return rows


def _parse_questions(
    value: object,
    *,
    expected_question_count: int,
) -> tuple[tuple[_Question, ...], Mapping[str, int], Mapping[str, int]]:
    if not isinstance(value, Mapping) or set(value) != {"questions"}:
        raise BioasqP0QualificationError(
            "source root must contain only questions"
        )
    raw_questions = value.get("questions")
    if (
        not isinstance(raw_questions, list)
        or len(raw_questions) != expected_question_count
    ):
        raise BioasqP0QualificationError(
            "source question count drifted"
        )

    questions: list[_Question] = []
    source_family_counts: Counter[str] = Counter()
    ineligible: Counter[str] = Counter()
    item_commitments: set[str] = set()

    for raw_question in raw_questions:
        if not isinstance(raw_question, Mapping):
            raise BioasqP0QualificationError(
                "question row schema drifted"
            )
        required = {"body", "documents", "id", "snippets", "type"}
        if not required <= set(raw_question):
            raise BioasqP0QualificationError(
                "question row is missing a required public field"
            )
        family = _family(raw_question.get("type"))
        source_family_counts[family] += 1
        source_id = _normalize_text(
            raw_question.get("id"),
            field="question id",
            casefold=False,
            maximum_length=_MAX_IDENTIFIER_CHARACTERS,
        )
        opaque_item_commitment = _commit("question_id", source_id)
        if opaque_item_commitment in item_commitments:
            raise BioasqP0QualificationError(
                "question id registry contains a duplicate"
            )
        item_commitments.add(opaque_item_commitment)

        try:
            query = _normalize_text(
                raw_question.get("body"),
                field="question body",
                casefold=True,
            )
        except BioasqP0QualificationError:
            ineligible["empty_or_invalid_query"] += 1
            continue

        documents = _string_list(
            raw_question.get("documents"),
            field="question documents",
            casefold=False,
        )
        raw_snippets = raw_question.get("snippets")
        if not isinstance(raw_snippets, list):
            raise BioasqP0QualificationError(
                "question snippets schema drifted"
            )
        snippets: set[tuple[str, str]] = set()
        observed_snippet_documents: set[str] = set()
        for raw_snippet in raw_snippets:
            if (
                not isinstance(raw_snippet, Mapping)
                or not {"document", "text"} <= set(raw_snippet)
            ):
                raise BioasqP0QualificationError(
                    "snippet row schema drifted"
                )
            snippet_document = _normalize_text(
                raw_snippet.get("document"),
                field="snippet document",
                casefold=False,
                maximum_length=_MAX_IDENTIFIER_CHARACTERS,
            )
            if snippet_document not in set(documents):
                raise BioasqP0QualificationError(
                    "snippet document is absent from question documents"
                )
            observed_snippet_documents.add(snippet_document)
            try:
                snippet_text = _normalize_text(
                    raw_snippet.get("text"),
                    field="snippet text",
                    casefold=False,
                )
            except BioasqP0QualificationError:
                continue
            snippets.add((snippet_document, snippet_text))

        if not documents:
            ineligible["no_gold_document"] += 1
            continue
        if not snippets:
            ineligible["no_nonempty_gold_snippet"] += 1
            continue
        if not observed_snippet_documents:
            ineligible["no_snippet_document"] += 1
            continue

        questions.append(
            _Question(
                opaque_item_commitment=opaque_item_commitment,
                family=family,
                query_commitment=_commit("normalized_query", query),
                document_commitments=tuple(
                    sorted(
                        _commit("gold_document", document)
                        for document in documents
                    )
                ),
                snippet_commitments=tuple(
                    sorted(
                        _commit(
                            "normalized_gold_snippet",
                            document + "\0" + snippet,
                        )
                        for document, snippet in snippets
                    )
                ),
            )
        )

    if not questions:
        raise BioasqP0QualificationError(
            "source contains no formally eligible question"
        )
    return (
        tuple(questions),
        {family: source_family_counts[family] for family in FAMILIES},
        dict(sorted(ineligible.items())),
    )


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

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


def _component_rows(
    questions: Sequence[_Question],
) -> tuple[
    tuple[_PrivateRow, ...],
    Mapping[str, frozenset[str]],
    Mapping[str, object],
]:
    if not questions:
        raise BioasqP0QualificationError(
            "component input is empty"
        )
    union = _DisjointSet(len(questions))
    owners: dict[tuple[str, str], int] = {}
    for index, question in enumerate(questions):
        keys = (
            (("query", question.query_commitment),)
            + tuple(
                ("document", commitment)
                for commitment in question.document_commitments
            )
            + tuple(
                ("snippet", commitment)
                for commitment in question.snippet_commitments
            )
        )
        for key in keys:
            previous = owners.setdefault(key, index)
            union.union(index, previous)

    grouped: dict[int, list[int]] = {}
    for index in range(len(questions)):
        grouped.setdefault(union.find(index), []).append(index)

    component_families: dict[str, frozenset[str]] = {}
    private_rows: list[_PrivateRow] = []
    profiles: Counter[str] = Counter()
    multi_question_count = 0
    row_count_in_multi = 0
    for indices in grouped.values():
        ordered = tuple(
            sorted(
                (questions[index] for index in indices),
                key=lambda row: row.opaque_item_commitment,
            )
        )
        component_commitment = stable_hash(
            {
                "component_rule": COMPONENT_RULE,
                "opaque_item_commitments": [
                    row.opaque_item_commitment for row in ordered
                ],
            }
        )
        families = frozenset(row.family for row in ordered)
        component_families[component_commitment] = families
        profiles["+".join(family for family in FAMILIES if family in families)] += 1
        if len(ordered) > 1:
            multi_question_count += 1
            row_count_in_multi += len(ordered)
        private_rows.extend(
            _PrivateRow(
                opaque_item_commitment=row.opaque_item_commitment,
                family=row.family,
                component_commitment=component_commitment,
                query_commitment=row.query_commitment,
                snippet_commitments=row.snippet_commitments,
            )
            for row in ordered
        )

    private_rows.sort(
        key=lambda row: (
            row.component_commitment,
            FAMILIES.index(row.family),
            row.opaque_item_commitment,
        )
    )
    aggregate = {
        "component_count": len(grouped),
        "component_family_profile_counts": dict(sorted(profiles.items())),
        "multi_question_component_count": multi_question_count,
        "row_count_in_multi_question_components": row_count_in_multi,
    }
    return tuple(private_rows), component_families, aggregate


def _add_flow_edge(
    graph: list[list[_FlowEdge]],
    left: int,
    right: int,
    capacity: int,
) -> _FlowEdge:
    forward = _FlowEdge(right, len(graph[right]), capacity)
    reverse = _FlowEdge(left, len(graph[left]), 0)
    graph[left].append(forward)
    graph[right].append(reverse)
    return forward


def _simultaneous_family_capacity(
    component_families: Mapping[str, frozenset[str]],
    demands: Mapping[str, int],
) -> Mapping[str, object]:
    components = tuple(sorted(component_families))
    source = 0
    component_offset = 1
    family_offset = component_offset + len(components)
    sink = family_offset + len(FAMILIES)
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    family_nodes = {
        family: family_offset + index
        for index, family in enumerate(FAMILIES)
    }
    sink_edges: dict[str, _FlowEdge] = {}
    for index, component in enumerate(components):
        node = component_offset + index
        _add_flow_edge(graph, source, node, 1)
        for family in FAMILIES:
            if family in component_families[component]:
                _add_flow_edge(graph, node, family_nodes[family], 1)
    for family in FAMILIES:
        sink_edges[family] = _add_flow_edge(
            graph,
            family_nodes[family],
            sink,
            demands[family],
        )

    flow = 0
    required = sum(demands[family] for family in FAMILIES)
    while flow < required:
        previous: list[tuple[int, int] | None] = [None] * len(graph)
        previous[source] = (-1, -1)
        queue: deque[int] = deque((source,))
        while queue and previous[sink] is None:
            node = queue.popleft()
            for edge_index, edge in enumerate(graph[node]):
                if edge.capacity <= 0 or previous[edge.to] is not None:
                    continue
                previous[edge.to] = (node, edge_index)
                queue.append(edge.to)
        if previous[sink] is None:
            break
        node = sink
        while node != source:
            parent_edge = previous[node]
            if parent_edge is None:
                raise BioasqP0QualificationError(
                    "capacity flow predecessor drifted"
                )
            parent, edge_index = parent_edge
            edge = graph[parent][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = parent
        flow += 1

    assigned = {
        family: demands[family] - sink_edges[family].capacity
        for family in FAMILIES
    }
    assignable = {
        family: sum(
            family in families for families in component_families.values()
        )
        for family in FAMILIES
    }
    return {
        "assignable_component_count_by_family": assignable,
        "demand_by_family": {
            family: demands[family] for family in FAMILIES
        },
        "maximum_flow_assigned_count_by_family": assigned,
        "maximum_flow_assigned_total": flow,
        "required_total": required,
        "simultaneous_component_capacity_saturated": flow == required,
    }


def _source_binding(
    *,
    size_bytes: int,
    sha256: str,
    synthetic: bool,
) -> dict[str, object]:
    if (
        type(size_bytes) is not int
        or size_bytes < 1
        or not isinstance(sha256, str)
        or _HEX64.fullmatch(sha256) is None
    ):
        raise BioasqP0QualificationError(
            "source binding is invalid"
        )
    return {
        "file_sha256": sha256,
        "size_bytes": size_bytes,
        "synthetic_source_free_canary_input": synthetic,
    }


def _qualify_decoded_source(
    raw: bytes,
    *,
    source_binding: Mapping[str, object],
    contract: QualificationContract = DEFAULT_CONTRACT,
    source_open_count: int,
    real_source_access_count: int,
    audit: _AccessAudit | None = None,
) -> QualificationResult:
    """Shared production parser, schema, component, and capacity entrypoint."""

    if (
        not isinstance(contract, QualificationContract)
        or type(source_open_count) is not int
        or source_open_count not in {0, 1}
        or type(real_source_access_count) is not int
        or real_source_access_count not in {0, 1}
        or real_source_access_count > source_open_count
    ):
        raise BioasqP0QualificationError(
            "source access audit contract drifted"
        )
    if audit is not None:
        audit.stage = "decode_strict_json"
        audit.source_json_decode_count += 1
    decoded = _decode_strict_json(raw)
    if audit is not None:
        audit.stage = "validate_four_family_schema"
    questions, source_family_counts, ineligible = _parse_questions(
        decoded,
        expected_question_count=contract.expected_question_count,
    )
    if audit is not None:
        audit.stage = "construct_query_document_snippet_components"
    private_rows, component_families, component_aggregate = (
        _component_rows(questions)
    )
    if audit is not None:
        audit.stage = "prove_simultaneous_component_capacity"
    capacity = _simultaneous_family_capacity(
        component_families,
        contract.minimum_components_per_family,
    )
    eligible_family_counts = Counter(row.family for row in questions)
    qualified = bool(
        capacity["simultaneous_component_capacity_saturated"]
    )

    if audit is not None:
        audit.stage = "form_safe_qualification_receipts"
    private_manifest = self_hashed(
        {
            "claim_boundary": {
                "action_model_retrieval_evaluator_or_score_count": 0,
                "cohort_assignment_or_selection_secret_count": 0,
                "contains_source_text_document_identifier_or_qrel_value": False,
                "noncohort_eligibility_commitments_only": True,
            },
            "component_rule": COMPONENT_RULE,
            "family_order": list(FAMILIES),
            "rows": [row.payload() for row in private_rows],
            "schema": PRIVATE_MANIFEST_SCHEMA,
            "source_binding": dict(source_binding),
            "status": "private_noncohort_component_commitments",
            "study_id": STUDY_ID,
        }
    )
    private_raw = canonical_bytes(private_manifest)
    safe_receipt = self_hashed(
        {
            "access_boundary": {
                "action_model_retrieval_evaluator_or_score_count": 0,
                "cohort_assignment_or_selection_secret_count": 0,
                "individual_item_query_document_snippet_or_commitment_published": False,
                "online_or_API_evaluation_count": 0,
                "real_source_access_count": real_source_access_count,
                "source_hash_count": source_open_count,
                "source_json_decode_count": 1,
                "source_open_count": source_open_count,
            },
            "capacity": dict(capacity),
            "component_aggregate": dict(component_aggregate),
            "eligible_question_count_by_family": {
                family: eligible_family_counts[family]
                for family in FAMILIES
            },
            "formal_ineligible_reason_counts": dict(ineligible),
            "private_manifest_binding": {
                "file_sha256": hashlib.sha256(private_raw).hexdigest(),
                "row_count": len(private_rows),
                "self_sha256": private_manifest["self_sha256"],
            },
            "schema": SAFE_RECEIPT_SCHEMA,
            "source_binding": dict(source_binding),
            "source_question_count": sum(source_family_counts.values()),
            "source_question_count_by_family": dict(source_family_counts),
            "status": (
                "qualified_public_non_scoring_schema_component_capacity"
                if qualified
                else "terminal_public_source_component_capacity_insufficient"
            ),
            "study_id": STUDY_ID,
        }
    )
    return QualificationResult(
        safe_receipt=safe_receipt,
        private_manifest=private_manifest,
    )


def _open_binary(path: Path) -> BinaryIO:
    """Single indirection used to audit the one production source open."""

    return path.open("rb")


def _read_source_once(
    path: Path,
    contract: SourceFileContract,
    *,
    audit: _AccessAudit | None = None,
) -> tuple[bytes, Mapping[str, object]]:
    absolute = path.absolute()
    if audit is not None:
        audit.stage = "verify_source_metadata"
    try:
        metadata = absolute.lstat()
    except OSError as exc:
        raise BioasqP0QualificationError(
            "bound source file is unavailable"
        ) from exc
    if (
        absolute.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size != contract.size_bytes
    ):
        raise BioasqP0QualificationError(
            "bound source file metadata drifted"
        )

    digest = hashlib.sha256()
    raw = bytearray()
    if audit is not None:
        audit.stage = "open_hash_and_read_source_once"
    try:
        handle = _open_binary(absolute)
        if audit is not None:
            audit.source_open_count += 1
            audit.real_source_access_count += 1
        with handle:
            while True:
                chunk = handle.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                raw.extend(chunk)
    except OSError as exc:
        raise BioasqP0QualificationError(
            "bound source file read failed"
        ) from exc
    observed = bytes(raw)
    if audit is not None:
        audit.source_hash_count += 1
    if (
        len(observed) != contract.size_bytes
        or digest.hexdigest() != contract.sha256
    ):
        raise BioasqP0QualificationError(
            "bound source file identity drifted"
        )
    return observed, _source_binding(
        size_bytes=len(observed),
        sha256=digest.hexdigest(),
        synthetic=False,
    )


def _write_exclusive_json(
    path: Path,
    value: Mapping[str, object],
    *,
    mode: int,
) -> Mapping[str, object]:
    raw = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, mode)
    except OSError as exc:
        raise BioasqP0QualificationError(
            "qualification output cannot be written exclusively"
        ) from exc
    return {
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "mode_octal": format(mode, "04o"),
        "self_sha256": value["self_sha256"],
        "size_bytes": len(raw),
    }


def qualify_source_path(
    *,
    source_path: Path,
    source_contract: SourceFileContract,
    private_manifest_path: Path | None = None,
    safe_receipt_path: Path | None = None,
    contract: QualificationContract = DEFAULT_CONTRACT,
    _audit: _AccessAudit | None = None,
) -> QualificationResult:
    """Open, hash, decode, and qualify one exact source file once."""

    if (private_manifest_path is None) != (safe_receipt_path is None):
        raise BioasqP0QualificationError(
            "both qualification output paths must be supplied together"
        )
    audit = _audit if _audit is not None else _AccessAudit()
    raw, binding = _read_source_once(
        source_path,
        source_contract,
        audit=audit,
    )
    result = _qualify_decoded_source(
        raw,
        source_binding=binding,
        contract=contract,
        source_open_count=1,
        real_source_access_count=1,
        audit=audit,
    )
    if private_manifest_path is not None and safe_receipt_path is not None:
        private_binding = _write_exclusive_json(
            private_manifest_path,
            result.private_manifest,
            mode=0o600,
        )
        expected = result.safe_receipt["private_manifest_binding"]
        if (
            not isinstance(expected, Mapping)
            or private_binding.get("file_sha256")
            != expected.get("file_sha256")
            or private_binding.get("self_sha256")
            != expected.get("self_sha256")
        ):
            raise BioasqP0QualificationError(
                "private manifest post-write binding drifted"
            )
        _write_exclusive_json(
            safe_receipt_path,
            result.safe_receipt,
            mode=0o600,
        )
    return result


def _synthetic_source_bytes() -> bytes:
    questions: list[dict[str, object]] = []
    for family in FAMILIES:
        for index in range(DEFAULT_MINIMUM_COMPONENTS_PER_FAMILY):
            token = f"{family}-{index:03d}"
            document = f"https://synthetic.invalid/document/{token}"
            questions.append(
                {
                    "body": f"Synthetic source-free {family} question {index}",
                    "documents": [document],
                    "id": f"synthetic-{token}",
                    "snippets": [
                        {
                            "document": document,
                            "text": (
                                f"Synthetic source-free {family} evidence "
                                f"{index}"
                            ),
                        }
                    ],
                    "type": family,
                }
            )
    return canonical_bytes({"questions": questions}, newline=False)


def run_source_free_canary() -> Mapping[str, object]:
    """Exercise the production parser/component path without a source path."""

    raw = _synthetic_source_bytes()
    binding = _source_binding(
        size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        synthetic=True,
    )
    synthetic_contract = QualificationContract(
        {
            family: DEFAULT_MINIMUM_COMPONENTS_PER_FAMILY
            for family in FAMILIES
        },
        expected_question_count=(
            len(FAMILIES) * DEFAULT_MINIMUM_COMPONENTS_PER_FAMILY
        ),
    )
    result = _qualify_decoded_source(
        raw,
        source_binding=binding,
        contract=synthetic_contract,
        source_open_count=0,
        real_source_access_count=0,
    )
    if (
        result.safe_receipt.get("status")
        != "qualified_public_non_scoring_schema_component_capacity"
    ):
        raise BioasqP0QualificationError(
            "source-free production parser/component canary failed"
        )
    body = {
        "external_distribution_import_count": 0,
        "formal_source_access_count": 0,
        "parser_component_entrypoint": "_qualify_decoded_source",
        "private_manifest_self_sha256": (
            result.private_manifest["self_sha256"]
        ),
        "schema": SOURCE_FREE_CANARY_SCHEMA,
        "source_json_decode_count": 1,
        "source_open_count": 0,
        "status": "passed_source_free_production_parser_component_canary",
        "study_id": STUDY_ID,
        "synthetic_component_capacity_saturated": (
            result.safe_receipt["capacity"][
                "simultaneous_component_capacity_saturated"
            ]
        ),
        "synthetic_component_count": result.safe_receipt[
            "component_aggregate"
        ]["component_count"],
    }
    return self_hashed(body)


def _safe_failure_receipt(
    *,
    audit: _AccessAudit,
    source_contract: SourceFileContract,
    exc: BaseException,
) -> Mapping[str, object]:
    body = {
        "access_boundary": {
            **audit.payload(),
            "action_model_retrieval_evaluator_or_score_count": 0,
            "cohort_assignment_or_selection_secret_count": 0,
            "individual_item_query_document_snippet_or_commitment_published": False,
            "online_or_API_evaluation_count": 0,
        },
        "aggregate_only_public_receipt": True,
        "expected_source_binding": {
            "file_sha256": source_contract.sha256,
            "size_bytes": source_contract.size_bytes,
        },
        "failure_exception_message_sha256": hashlib.sha256(
            str(exc).encode("utf-8", errors="replace")
        ).hexdigest(),
        "failure_exception_type_sha256": hashlib.sha256(
            type(exc).__name__.encode("ascii", errors="replace")
        ).hexdigest(),
        "failure_stage": audit.stage,
        "schema": f"{VERSION}_safe_failure_receipt_v1",
        "status": "terminal_public_source_qualification_failure_no_retry",
        "study_id": STUDY_ID,
    }
    return self_hashed(body)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-free-canary", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--expected-size-bytes", type=int)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--private-manifest", type=Path)
    parser.add_argument("--safe-receipt", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    if arguments.source_free_canary:
        if any(
            value is not None
            for value in (
                arguments.source,
                arguments.expected_size_bytes,
                arguments.expected_sha256,
                arguments.private_manifest,
            )
        ):
            raise BioasqP0QualificationError(
                "source-free canary received a formal source capability"
            )
        receipt = run_source_free_canary()
        if arguments.safe_receipt is not None:
            _write_exclusive_json(
                arguments.safe_receipt,
                receipt,
                mode=0o600,
            )
    else:
        if any(
            value is None
            for value in (
                arguments.source,
                arguments.expected_size_bytes,
                arguments.expected_sha256,
                arguments.private_manifest,
                arguments.safe_receipt,
            )
        ):
            raise BioasqP0QualificationError(
                "formal qualification arguments are incomplete"
            )
        source_contract = SourceFileContract(
            size_bytes=arguments.expected_size_bytes,
            sha256=arguments.expected_sha256,
        )
        audit = _AccessAudit()
        try:
            result = qualify_source_path(
                source_path=arguments.source,
                source_contract=source_contract,
                private_manifest_path=arguments.private_manifest,
                safe_receipt_path=arguments.safe_receipt,
                _audit=audit,
            )
            receipt = result.safe_receipt
        except Exception as exc:
            receipt = _safe_failure_receipt(
                audit=audit,
                source_contract=source_contract,
                exc=exc,
            )
            try:
                _write_exclusive_json(
                    arguments.safe_receipt,
                    receipt,
                    mode=0o600,
                )
            except Exception:
                pass
            print(
                json.dumps(
                    {
                        "schema": receipt["schema"],
                        "self_sha256": receipt["self_sha256"],
                        "status": receipt["status"],
                    },
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            return 1
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "self_sha256": receipt["self_sha256"],
                "status": receipt["status"],
            },
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
    "BioasqP0QualificationError",
    "DEFAULT_CONTRACT",
    "DEFAULT_MINIMUM_COMPONENTS_PER_FAMILY",
    "FAMILIES",
    "PRIVATE_MANIFEST_SCHEMA",
    "QualificationContract",
    "QualificationResult",
    "SAFE_RECEIPT_SCHEMA",
    "SOURCE_FREE_CANARY_SCHEMA",
    "STUDY_ID",
    "SourceFileContract",
    "VERSION",
    "canonical_bytes",
    "qualify_source_path",
    "run_source_free_canary",
    "self_hashed",
    "stable_hash",
]
