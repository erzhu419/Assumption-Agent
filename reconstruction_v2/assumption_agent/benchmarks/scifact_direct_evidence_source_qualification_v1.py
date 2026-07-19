"""One-shot aggregate-only SciFact TRAIN/DEV source qualification.

This module verifies the pre-row source custody, extracts only the three
allowed archive members, and checks whether connected-component-disjoint
balanced cohorts exist.  It never opens the TEST payload, creates a selection
secret, selects an item, or runs a retrieval action or score.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from typing import Any, BinaryIO, Iterable, Mapping, Sequence


VERSION = "v1"
SCHEMA = "scifact_direct_evidence_source_qualification_result_v1"
ATTEMPT_SCHEMA = "scifact_direct_evidence_source_qualification_attempt_v1"
FREEZE_SCHEMA = (
    "scifact_direct_evidence_source_qualification_implementation_freeze_v1"
)

FAMILY_ORDER = (
    "CONTRADICT_SINGLE",
    "MULTI_SENTENCE",
    "SUPPORT_SINGLE",
)
SPLIT_ORDER = ("train", "dev")
LABELS = frozenset(("SUPPORT", "CONTRADICT"))

ARCHIVE_RELATIVE_PATH = Path("artifacts/scifact_official_source_v1/data.tar.gz")
ARCHIVE_SIZE = 3_115_079
ARCHIVE_SHA256 = (
    "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
)
MEMBER_SPECS = {
    "train": ("data/claims_train.jsonl", 175_616),
    "dev": ("data/claims_dev.jsonl", 65_007),
    "corpus": ("data/corpus.jsonl", 8_307_875),
}
TEST_MEMBER = "data/claims_test.jsonl"

DESIGN_RELATIVE_PATH = Path(
    "manifests/scifact_direct_evidence_source_qualification_design_v1.json"
)
DESIGN_FILE_SHA256 = (
    "7703c61dfe72bb03b80d76c0cbeacdc24b01b5646edf5dbc5d0107832f1da179"
)
DESIGN_SELF_SHA256 = (
    "186421fcb527ce60c0fe72888b0413cbddd8de3a2963f44c06ef7d13e7b5ee0b"
)
CUSTODY_RELATIVE_PATH = Path(
    "manifests/scifact_direct_evidence_source_custody_v1.json"
)
CUSTODY_FILE_SHA256 = (
    "5826bc56b1a807ec6617a36d3a162e9b97026c86ebb95d4f9ef98cc44f73759c"
)
CUSTODY_SELF_SHA256 = (
    "c7516174a85605d64b84800e7f52f4e2a422e5b90c09c85db04242aa3c82afad"
)
FREEZE_RELATIVE_PATH = Path(
    "manifests/"
    "scifact_direct_evidence_source_qualification_implementation_freeze_v1.json"
)
QUALIFIER_RELATIVE_PATH = Path(
    "assumption_agent/benchmarks/"
    "scifact_direct_evidence_source_qualification_v1.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/test_scifact_direct_evidence_source_qualification_v1.py"
)
FORMAL_ATTEMPT_RELATIVE_PATH = Path(
    "artifacts/scifact_direct_evidence_source_qualification_v1"
)
FORMAL_RESULT_RELATIVE_PATH = Path(
    "manifests/scifact_direct_evidence_source_qualification_result_v1.json"
)

TRAIN_DEMANDS = {family: 52 for family in FAMILY_ORDER}
DEV_DEMANDS = {family: 10 for family in FAMILY_ORDER}
DENY_CLAIM_IDS = frozenset((123, 263))
DENY_DOC_IDS = frozenset((4_883_040, 11_328_820, 14_853_989, 30_041_340))


class SciFactQualificationError(RuntimeError):
    """Fail-closed source qualification error with no item content."""


class FormalProvenanceError(SciFactQualificationError):
    """The committed implementation or preregistration drifted."""


class OneShotRefusal(SciFactQualificationError):
    """The formal source-qualification attempt is not pristine."""


class _RowInvalid(ValueError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise SciFactQualificationError("non-canonical public value") from exc


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    output = dict(body)
    output[field] = _sha256(_canonical_json(output))
    return output


def _no_duplicate_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate object key")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise ValueError("nonfinite JSON number")


def _strict_json(raw: bytes, *, public_label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SciFactQualificationError(
            f"invalid strict JSON in {public_label}"
        ) from exc


def _strict_json_line(raw: bytes) -> Mapping[str, Any]:
    if not raw:
        raise _RowInvalid("json_line")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise _RowInvalid("json_line") from exc
    if not isinstance(value, Mapping):
        raise _RowInvalid("row_root")
    return value


def _integer(value: Any, reason: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _RowInvalid(reason)
    return value


def _text(value: Any, reason: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise _RowInvalid(reason)
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise _RowInvalid(reason) from exc
    if nonempty and not value.strip():
        raise _RowInvalid(reason)
    return value


def _sequence(value: Any, reason: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _RowInvalid(reason)
    return value


def _mapping(value: Any, reason: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _RowInvalid(reason)
    return value


def _decimal_object_key(value: str, reason: str) -> int:
    if not isinstance(value, str) or not value or not value.isascii():
        raise _RowInvalid(reason)
    if not value.isdigit() or (len(value) > 1 and value.startswith("0")):
        raise _RowInvalid(reason)
    return int(value)


@dataclass(frozen=True)
class _CorpusDocument:
    doc_id: int
    title: str
    abstract: tuple[str, ...]
    structured: bool

    @property
    def candidate_abstract_eligible(self) -> bool:
        return (
            5 <= len(self.abstract) <= 64
            and all(sentence.strip() for sentence in self.abstract)
        )


@dataclass(frozen=True)
class _Rationale:
    label: str
    sentences: tuple[int, ...]


@dataclass(frozen=True)
class _Claim:
    split: str
    claim_id: int
    claim: str
    cited_doc_ids: tuple[int, ...]
    evidence: Mapping[int, tuple[_Rationale, ...]]


@dataclass(frozen=True)
class _Candidate:
    split: str
    claim_id: int
    doc_id: int
    family: str
    label: str
    rationale_sizes: tuple[int, ...]

    @property
    def key(self) -> str:
        return f"{self.split}:{self.claim_id}:{self.doc_id}"


@dataclass(frozen=True)
class _MemberBinding:
    member: str
    byte_size: int
    sha256: str
    line_count: int
    private_path: Path


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[str, int], tuple[str, int]] = {}
        self.rank: dict[tuple[str, int], int] = {}

    def add(self, value: tuple[str, int]) -> None:
        if value not in self.parent:
            self.parent[value] = value
            self.rank[value] = 0

    def find(self, value: tuple[str, int]) -> tuple[str, int]:
        self.add(value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: tuple[str, int], right: tuple[str, int]) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


@dataclass
class _FlowEdge:
    to: int
    reverse: int
    capacity: int


def _add_edge(
    graph: list[list[_FlowEdge]], left: int, right: int, capacity: int
) -> None:
    graph[left].append(_FlowEdge(right, len(graph[right]), capacity))
    graph[right].append(_FlowEdge(left, len(graph[left]) - 1, 0))


def _max_family_flow(
    component_families: Mapping[str, frozenset[str]],
    demands: Mapping[str, int],
) -> tuple[int, Mapping[str, int]]:
    components = tuple(sorted(component_families))
    families = tuple(family for family in FAMILY_ORDER if family in demands)
    source = 0
    component_offset = 1
    family_offset = component_offset + len(components)
    sink = family_offset + len(families)
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    component_nodes = {
        component: component_offset + index
        for index, component in enumerate(components)
    }
    family_nodes = {
        family: family_offset + index for index, family in enumerate(families)
    }
    sink_edges: dict[str, _FlowEdge] = {}
    for component in components:
        _add_edge(graph, source, component_nodes[component], 1)
        for family in families:
            if family in component_families[component]:
                _add_edge(
                    graph,
                    component_nodes[component],
                    family_nodes[family],
                    1,
                )
    for family in families:
        _add_edge(graph, family_nodes[family], sink, demands[family])
        sink_edges[family] = graph[family_nodes[family]][-1]

    flow = 0
    while True:
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
            parent, edge_index = previous[node]  # type: ignore[misc]
            edge = graph[parent][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = parent
        flow += 1

    assigned = {
        family: demands[family] - sink_edges[family].capacity
        for family in families
    }
    return flow, assigned


def _parse_corpus(raw: bytes) -> tuple[dict[int, _CorpusDocument], dict[str, Any]]:
    documents: dict[int, _CorpusDocument] = {}
    invalid = Counter()
    duplicate_count = 0
    length_histogram = Counter()
    candidate_eligible_count = 0
    structured_count = 0
    lines = raw.splitlines()
    for line in lines:
        try:
            value = _strict_json_line(line)
            doc_id = _integer(value.get("doc_id"), "corpus_doc_id")
            title = _text(value.get("title"), "corpus_title")
            raw_abstract = _sequence(value.get("abstract"), "corpus_abstract")
            abstract = tuple(
                _text(sentence, "corpus_sentence") for sentence in raw_abstract
            )
            structured = value.get("structured")
            if not isinstance(structured, bool):
                raise _RowInvalid("corpus_structured")
            if doc_id in documents:
                duplicate_count += 1
                continue
            document = _CorpusDocument(doc_id, title, abstract, structured)
            documents[doc_id] = document
            length_histogram[len(abstract)] += 1
            candidate_eligible_count += int(document.candidate_abstract_eligible)
            structured_count += int(structured)
        except _RowInvalid as exc:
            invalid[exc.reason] += 1
    return documents, {
        "line_count": len(lines),
        "valid_unique_document_count": len(documents),
        "duplicate_document_id_count": duplicate_count,
        "invalid_row_reason_counts": dict(sorted(invalid.items())),
        "candidate_abstract_eligible_document_count": candidate_eligible_count,
        "structured_document_count": structured_count,
        "abstract_sentence_count_histogram": {
            str(key): length_histogram[key] for key in sorted(length_histogram)
        },
    }


def _parse_claims(raw: bytes, *, split: str) -> tuple[list[_Claim], dict[str, Any]]:
    claims: list[_Claim] = []
    invalid = Counter()
    duplicate_within_split = 0
    seen: set[int] = set()
    evidence_pair_count = 0
    empty_evidence_claim_count = 0
    lines = raw.splitlines()
    for line in lines:
        try:
            value = _strict_json_line(line)
            claim_id = _integer(value.get("id"), "claim_id")
            claim_text = _text(value.get("claim"), "claim_text", nonempty=True)
            raw_cited = _sequence(value.get("cited_doc_ids"), "cited_doc_ids")
            cited = tuple(
                _integer(doc_id, "cited_doc_id") for doc_id in raw_cited
            )
            if len(set(cited)) != len(cited):
                raise _RowInvalid("duplicate_cited_doc_id")
            raw_evidence = _mapping(value.get("evidence"), "evidence")
            evidence: dict[int, tuple[_Rationale, ...]] = {}
            for raw_doc_id, raw_rationales in raw_evidence.items():
                doc_id = _decimal_object_key(raw_doc_id, "evidence_doc_id")
                rows = _sequence(raw_rationales, "rationales")
                if not rows:
                    raise _RowInvalid("empty_rationales")
                rationales: list[_Rationale] = []
                for raw_rationale in rows:
                    rationale = _mapping(raw_rationale, "rationale")
                    label = _text(
                        rationale.get("label"), "rationale_label", nonempty=True
                    )
                    if label not in LABELS:
                        raise _RowInvalid("rationale_label")
                    raw_sentences = _sequence(
                        rationale.get("sentences"), "rationale_sentences"
                    )
                    sentences = tuple(
                        _integer(position, "rationale_sentence")
                        for position in raw_sentences
                    )
                    rationales.append(_Rationale(label, sentences))
                evidence[doc_id] = tuple(rationales)
            if claim_id in seen:
                duplicate_within_split += 1
                continue
            seen.add(claim_id)
            claims.append(_Claim(split, claim_id, claim_text, cited, evidence))
            evidence_pair_count += len(evidence)
            empty_evidence_claim_count += int(not evidence)
        except _RowInvalid as exc:
            invalid[exc.reason] += 1
    return claims, {
        "line_count": len(lines),
        "valid_unique_claim_count": len(claims),
        "duplicate_claim_id_within_split_count": duplicate_within_split,
        "invalid_row_reason_counts": dict(sorted(invalid.items())),
        "evidence_pair_count": evidence_pair_count,
        "empty_evidence_claim_count": empty_evidence_claim_count,
    }


def _component_token(root: tuple[str, int]) -> str:
    return _sha256(_canonical_json([root[0], root[1]]))


def _candidate_audit(
    corpus: Mapping[int, _CorpusDocument],
    claims_by_split: Mapping[str, Sequence[_Claim]],
) -> tuple[dict[str, Any], Mapping[str, Mapping[str, frozenset[str]]], int]:
    union_find = _UnionFind()
    all_claims = [
        claim
        for split in SPLIT_ORDER
        for claim in claims_by_split.get(split, ())
    ]
    claim_occurrences: Counter[int] = Counter(claim.claim_id for claim in all_claims)
    duplicate_across_split_count = sum(
        count - 1 for count in claim_occurrences.values() if count > 1
    )
    for claim in all_claims:
        claim_node = ("claim", claim.claim_id)
        union_find.add(claim_node)
        doc_ids = set(claim.cited_doc_ids) | set(claim.evidence)
        for doc_id in doc_ids:
            union_find.union(claim_node, ("doc", doc_id))

    component_splits: dict[tuple[str, int], set[str]] = {}
    component_denied: set[tuple[str, int]] = set()
    for claim in all_claims:
        root = union_find.find(("claim", claim.claim_id))
        component_splits.setdefault(root, set()).add(claim.split)
    for claim_id in DENY_CLAIM_IDS:
        node = ("claim", claim_id)
        if node in union_find.parent:
            component_denied.add(union_find.find(node))
    for doc_id in DENY_DOC_IDS:
        node = ("doc", doc_id)
        if node in union_find.parent:
            component_denied.add(union_find.find(node))

    pre_candidates: list[tuple[_Candidate, tuple[str, int]]] = []
    ineligible = {split: Counter() for split in SPLIT_ORDER}
    mapping_errors = {split: Counter() for split in SPLIT_ORDER}
    rationale_size_histograms = {split: Counter() for split in SPLIT_ORDER}
    for claim in all_claims:
        root = union_find.find(("claim", claim.claim_id))
        cited = set(claim.cited_doc_ids)
        for doc_id, rationales in claim.evidence.items():
            reason: str | None = None
            document = corpus.get(doc_id)
            if doc_id not in cited:
                mapping_errors[claim.split]["evidence_doc_not_cited"] += 1
                reason = "mapping_error"
            elif document is None:
                mapping_errors[claim.split]["evidence_doc_missing_corpus"] += 1
                reason = "mapping_error"
            labels = {rationale.label for rationale in rationales}
            if len(labels) != 1:
                mapping_errors[claim.split]["inconsistent_rationale_labels"] += 1
                reason = "mapping_error"
            sizes = tuple(len(rationale.sentences) for rationale in rationales)
            for size in sizes:
                rationale_size_histograms[claim.split][size] += 1
            for rationale in rationales:
                positions = rationale.sentences
                if not positions:
                    mapping_errors[claim.split]["empty_rationale"] += 1
                    reason = "mapping_error"
                elif len(set(positions)) != len(positions):
                    mapping_errors[claim.split]["duplicate_rationale_position"] += 1
                    reason = "mapping_error"
                elif document is not None and any(
                    position < 0 or position >= len(document.abstract)
                    for position in positions
                ):
                    mapping_errors[claim.split]["rationale_position_out_of_bounds"] += 1
                    reason = "mapping_error"
            if reason is not None:
                ineligible[claim.split][reason] += 1
                continue
            assert document is not None
            if not document.candidate_abstract_eligible:
                ineligible[claim.split]["abstract_sentence_contract"] += 1
                continue
            if any(size > 3 for size in sizes):
                ineligible[claim.split]["rationale_size_above_Set3"] += 1
                continue
            minimum_size = min(sizes)
            label = next(iter(labels))
            if minimum_size >= 2:
                family = "MULTI_SENTENCE"
            elif label == "CONTRADICT":
                family = "CONTRADICT_SINGLE"
            else:
                family = "SUPPORT_SINGLE"
            pre_candidates.append(
                (
                    _Candidate(
                        claim.split,
                        claim.claim_id,
                        doc_id,
                        family,
                        label,
                        sizes,
                    ),
                    root,
                )
            )

    clean_by_split: dict[str, list[tuple[_Candidate, tuple[str, int]]]] = {
        split: [] for split in SPLIT_ORDER
    }
    excluded_cross = {split: Counter() for split in SPLIT_ORDER}
    excluded_deny = {split: Counter() for split in SPLIT_ORDER}
    pre_family = {split: Counter() for split in SPLIT_ORDER}
    for candidate, root in pre_candidates:
        pre_family[candidate.split][candidate.family] += 1
        if len(component_splits.get(root, ())) > 1:
            excluded_cross[candidate.split][candidate.family] += 1
        elif root in component_denied:
            excluded_deny[candidate.split][candidate.family] += 1
        else:
            clean_by_split[candidate.split].append((candidate, root))

    public_split: dict[str, Any] = {}
    flow_inputs: dict[str, dict[str, frozenset[str]]] = {}
    for split in SPLIT_ORDER:
        candidates = clean_by_split[split]
        by_component: dict[str, set[str]] = {}
        clean_family = Counter()
        keys_by_family: dict[str, list[str]] = {
            family: [] for family in FAMILY_ORDER
        }
        for candidate, root in candidates:
            token = _component_token(root)
            by_component.setdefault(token, set()).add(candidate.family)
            clean_family[candidate.family] += 1
            keys_by_family[candidate.family].append(candidate.key)
        profiles = Counter(
            "+".join(family for family in FAMILY_ORDER if family in families)
            for families in by_component.values()
        )
        flow_inputs[split] = {
            component: frozenset(families)
            for component, families in by_component.items()
        }
        public_split[split] = {
            "pre_component_eligible_candidate_counts": {
                family: pre_family[split][family] for family in FAMILY_ORDER
            },
            "source_ineligible_reason_counts": dict(sorted(ineligible[split].items())),
            "mapping_error_reason_counts": dict(
                sorted(mapping_errors[split].items())
            ),
            "rationale_size_histogram": {
                str(key): rationale_size_histograms[split][key]
                for key in sorted(rationale_size_histograms[split])
            },
            "cross_split_component_excluded_candidate_counts": {
                family: excluded_cross[split][family] for family in FAMILY_ORDER
            },
            "public_example_component_excluded_candidate_counts": {
                family: excluded_deny[split][family] for family in FAMILY_ORDER
            },
            "clean_candidate_counts": {
                family: clean_family[family] for family in FAMILY_ORDER
            },
            "clean_component_count": len(by_component),
            "clean_component_family_profile_counts": dict(sorted(profiles.items())),
            "clean_population_key_sha256_by_family": {
                family: _sha256(
                    _canonical_json(sorted(keys_by_family[family]))
                )
                for family in FAMILY_ORDER
            },
        }

    graph_roots = {union_find.find(node) for node in union_find.parent}
    public = {
        "component_graph": {
            "node_count": len(union_find.parent),
            "component_count": len(graph_roots),
            "cross_split_component_count": sum(
                len(splits) > 1 for splits in component_splits.values()
            ),
            "declared_public_example_intersecting_component_count": len(
                component_denied
            ),
        },
        "candidate_splits": public_split,
    }
    return public, flow_inputs, duplicate_across_split_count


def qualify_decoded_sources(
    corpus_raw: bytes,
    train_raw: bytes,
    dev_raw: bytes,
    *,
    source_binding: Mapping[str, Any],
    train_demands: Mapping[str, int] = TRAIN_DEMANDS,
    dev_demands: Mapping[str, int] = DEV_DEMANDS,
) -> dict[str, Any]:
    corpus, corpus_audit = _parse_corpus(corpus_raw)
    train_claims, train_audit = _parse_claims(train_raw, split="train")
    dev_claims, dev_audit = _parse_claims(dev_raw, split="dev")
    claims_by_split = {"train": train_claims, "dev": dev_claims}
    candidate_audit, flow_inputs, cross_split_claim_duplicates = _candidate_audit(
        corpus, claims_by_split
    )

    schema_error_count = (
        corpus_audit["duplicate_document_id_count"]
        + sum(corpus_audit["invalid_row_reason_counts"].values())
        + train_audit["duplicate_claim_id_within_split_count"]
        + sum(train_audit["invalid_row_reason_counts"].values())
        + dev_audit["duplicate_claim_id_within_split_count"]
        + sum(dev_audit["invalid_row_reason_counts"].values())
        + cross_split_claim_duplicates
    )
    mapping_error_count = sum(
        sum(
            candidate_audit["candidate_splits"][split][
                "mapping_error_reason_counts"
            ].values()
        )
        for split in SPLIT_ORDER
    )

    flows: dict[str, Any] = {}
    total_required = 0
    total_assigned = 0
    for split, demands in (("train", train_demands), ("dev", dev_demands)):
        assigned_total, assigned = _max_family_flow(flow_inputs[split], demands)
        required_total = sum(demands.values())
        total_required += required_total
        total_assigned += assigned_total
        flows[split] = {
            "demands": {family: demands[family] for family in FAMILY_ORDER},
            "assignable_component_counts": {
                family: sum(
                    family in families for families in flow_inputs[split].values()
                )
                for family in FAMILY_ORDER
            },
            "maximum_flow_assigned_counts": {
                family: assigned[family] for family in FAMILY_ORDER
            },
            "required_total": required_total,
            "maximum_flow_assigned_total": assigned_total,
            "simultaneous_family_capacity_saturated": assigned_total
            == required_total,
        }

    passed = (
        schema_error_count == 0
        and mapping_error_count == 0
        and total_assigned == total_required
    )
    body = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": (
            "qualified_source_capacity_no_selection"
            if passed
            else "terminal_source_infeasible_no_selection"
        ),
        "source_binding": dict(source_binding),
        "source_aggregates": {
            "corpus": corpus_audit,
            "claims": {"train": train_audit, "dev": dev_audit},
        },
        "candidate_and_component_aggregates": candidate_audit,
        "simultaneous_component_disjoint_capacity": flows,
        "terminal_reason_counts": {
            "schema_error_count": schema_error_count,
            "mapping_error_count": mapping_error_count,
            "unsatisfied_capacity_count": total_required - total_assigned,
        },
        "claim_boundary": {
            "qualification_only_no_efficacy_claim": True,
            "selection_secret_generated_or_opened": False,
            "item_selected_or_materialized": False,
            "retrieval_action_evaluator_classifier_or_score_run": False,
            "online_or_external_evaluation_used": False,
            "test_member_payload_opened": False,
            "item_ID_doc_ID_claim_title_abstract_rationale_text_or_per_item_record_emitted": False,
        },
    }
    return _self_hashed(body, "qualification_sha256")


def _copy_member(
    source: BinaryIO, destination: Path, *, expected_size: int
) -> tuple[str, int]:
    digest = hashlib.sha256()
    written = 0
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                digest.update(block)
                written += len(block)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    if written != expected_size:
        destination.unlink(missing_ok=True)
        raise SciFactQualificationError("allowed archive member size drifted")
    return digest.hexdigest(), written


def extract_allowed_members_once(
    archive_path: Path,
    destination_root: Path,
    *,
    expected_archive_size: int = ARCHIVE_SIZE,
    expected_archive_sha256: str = ARCHIVE_SHA256,
    member_specs: Mapping[str, tuple[str, int]] = MEMBER_SPECS,
) -> Mapping[str, _MemberBinding]:
    if archive_path.stat().st_size != expected_archive_size:
        raise SciFactQualificationError("source archive size drifted")
    if _file_sha256(archive_path) != expected_archive_sha256:
        raise SciFactQualificationError("source archive SHA256 drifted")
    if destination_root.exists():
        raise OneShotRefusal("private extraction root already exists")
    destination_root.mkdir(parents=True, mode=0o700)
    os.chmod(destination_root, 0o700)
    bindings: dict[str, _MemberBinding] = {}
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            by_name: dict[str, list[tarfile.TarInfo]] = {}
            for member in members:
                by_name.setdefault(member.name, []).append(member)
            if TEST_MEMBER not in by_name or len(by_name[TEST_MEMBER]) != 1:
                raise SciFactQualificationError("forbidden TEST header drifted")
            for role, (member_name, expected_size) in member_specs.items():
                matched = by_name.get(member_name, ())
                if len(matched) != 1:
                    raise SciFactQualificationError("allowed member header drifted")
                member = matched[0]
                if not member.isfile() or member.size != expected_size:
                    raise SciFactQualificationError("allowed member header drifted")
                source = archive.extractfile(member)
                if source is None:
                    raise SciFactQualificationError("allowed member unavailable")
                destination = destination_root / Path(member_name).name
                with source:
                    member_hash, byte_size = _copy_member(
                        source, destination, expected_size=expected_size
                    )
                with destination.open("rb") as handle:
                    line_count = sum(1 for _ in handle)
                bindings[role] = _MemberBinding(
                    member_name,
                    byte_size,
                    member_hash,
                    line_count,
                    destination,
                )
    except BaseException:
        shutil.rmtree(destination_root, ignore_errors=True)
        raise
    return bindings


def _verify_self_hash(
    value: Mapping[str, Any], *, field: str, expected: str
) -> None:
    body = dict(value)
    claimed = body.pop(field, None)
    actual = _sha256(_canonical_json(body))
    if claimed != expected or actual != expected:
        raise FormalProvenanceError("manifest self hash drifted")


def _load_bound_manifest(
    path: Path,
    *,
    expected_file_sha256: str,
    self_field: str,
    expected_self_sha256: str,
) -> Mapping[str, Any]:
    raw = path.read_bytes()
    if _sha256(raw) != expected_file_sha256:
        raise FormalProvenanceError("bound manifest file hash drifted")
    value = _strict_json(raw, public_label="bound manifest")
    if not isinstance(value, Mapping):
        raise FormalProvenanceError("bound manifest root drifted")
    _verify_self_hash(value, field=self_field, expected=expected_self_sha256)
    return value


def _git_output(project_root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=project_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def verify_formal_provenance(project_root: Path) -> Mapping[str, Any]:
    _load_bound_manifest(
        project_root / "reconstruction_v2" / DESIGN_RELATIVE_PATH,
        expected_file_sha256=DESIGN_FILE_SHA256,
        self_field="design_sha256",
        expected_self_sha256=DESIGN_SELF_SHA256,
    )
    _load_bound_manifest(
        project_root / "reconstruction_v2" / CUSTODY_RELATIVE_PATH,
        expected_file_sha256=CUSTODY_FILE_SHA256,
        self_field="source_custody_sha256",
        expected_self_sha256=CUSTODY_SELF_SHA256,
    )
    freeze_path = project_root / "reconstruction_v2" / FREEZE_RELATIVE_PATH
    freeze_raw = freeze_path.read_bytes()
    freeze = _strict_json(freeze_raw, public_label="implementation freeze")
    if not isinstance(freeze, Mapping) or freeze.get("schema") != FREEZE_SCHEMA:
        raise FormalProvenanceError("implementation freeze schema drifted")
    _verify_self_hash(
        freeze,
        field="implementation_freeze_sha256",
        expected=str(freeze.get("implementation_freeze_sha256", "")),
    )
    if freeze.get("status") != "frozen_before_first_TRAIN_or_DEV_row_parse":
        raise FormalProvenanceError("implementation freeze status drifted")
    implementation_commit = freeze.get("implementation_commit")
    if (
        not isinstance(implementation_commit, str)
        or len(implementation_commit) != 40
        or any(character not in "0123456789abcdef" for character in implementation_commit)
    ):
        raise FormalProvenanceError("implementation commit drifted")
    if _git_output(project_root, "cat-file", "-t", implementation_commit) != "commit":
        raise FormalProvenanceError("implementation commit is unavailable")
    required = (
        ("custody", CUSTODY_RELATIVE_PATH, CUSTODY_FILE_SHA256),
        ("design", DESIGN_RELATIVE_PATH, DESIGN_FILE_SHA256),
        ("qualifier", QUALIFIER_RELATIVE_PATH, None),
        ("qualifier_test", TEST_RELATIVE_PATH, None),
    )
    files = freeze.get("files")
    if not isinstance(files, list) or len(files) != len(required):
        raise FormalProvenanceError("implementation file bindings drifted")
    for row, (role, relative, fixed_sha256) in zip(files, required, strict=True):
        if not isinstance(row, Mapping):
            raise FormalProvenanceError("implementation file binding drifted")
        expected_sha256 = row.get("sha256")
        if (
            set(row) != {"role", "relative_path", "sha256"}
            or row.get("role") != role
            or row.get("relative_path") != relative.as_posix()
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or (fixed_sha256 is not None and expected_sha256 != fixed_sha256)
        ):
            raise FormalProvenanceError("implementation file binding drifted")
        path = project_root / "reconstruction_v2" / relative
        if _file_sha256(path) != expected_sha256:
            raise FormalProvenanceError("implementation working file drifted")
        repository_path = str(Path("reconstruction_v2") / relative)
        committed = subprocess.run(
            ("git", "show", f"{implementation_commit}:{repository_path}"),
            cwd=project_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        if _sha256(committed) != expected_sha256:
            raise FormalProvenanceError("implementation committed file drifted")
    subprocess.run(
        ("git", "merge-base", "--is-ancestor", implementation_commit, "HEAD"),
        cwd=project_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    dirty = _git_output(
        project_root,
        "status",
        "--porcelain",
        "--",
        str(Path("reconstruction_v2") / QUALIFIER_RELATIVE_PATH),
        str(Path("reconstruction_v2") / TEST_RELATIVE_PATH),
        str(Path("reconstruction_v2") / DESIGN_RELATIVE_PATH),
        str(Path("reconstruction_v2") / CUSTODY_RELATIVE_PATH),
        str(Path("reconstruction_v2") / FREEZE_RELATIVE_PATH),
    )
    if dirty:
        raise FormalProvenanceError("formal implementation paths are dirty")
    return freeze


def _write_json_once(path: Path, value: Mapping[str, Any], *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("ascii") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def run_formal(project_root: Path) -> Mapping[str, Any]:
    project_root = project_root.resolve()
    freeze = verify_formal_provenance(project_root)
    root = project_root / "reconstruction_v2"
    attempt_root = root / FORMAL_ATTEMPT_RELATIVE_PATH
    result_path = root / FORMAL_RESULT_RELATIVE_PATH
    if attempt_root.exists() or result_path.exists():
        raise OneShotRefusal("formal qualification path is not pristine")
    attempt_root.mkdir(parents=True, mode=0o700)
    os.chmod(attempt_root, 0o700)
    attempt = _self_hashed(
        {
            "schema": ATTEMPT_SCHEMA,
            "version": VERSION,
            "archive_sha256": ARCHIVE_SHA256,
            "design_sha256": DESIGN_SELF_SHA256,
            "custody_sha256": CUSTODY_SELF_SHA256,
            "implementation_freeze_sha256": freeze[
                "implementation_freeze_sha256"
            ],
            "selection_secret_generated_or_opened": False,
            "test_member_payload_opened": False,
        },
        "attempt_sha256",
    )
    marker_path = attempt_root / "attempt.json"
    _write_json_once(marker_path, attempt, mode=0o600)
    extracted_root = attempt_root / "train_dev_source"
    bindings = extract_allowed_members_once(
        root / ARCHIVE_RELATIVE_PATH, extracted_root
    )
    source_binding = {
        "archive": {
            "byte_size": ARCHIVE_SIZE,
            "sha256": ARCHIVE_SHA256,
        },
        "members": {
            role: {
                "member": binding.member,
                "byte_size": binding.byte_size,
                "sha256": binding.sha256,
                "line_count": binding.line_count,
            }
            for role, binding in sorted(bindings.items())
        },
        "design_sha256": DESIGN_SELF_SHA256,
        "custody_sha256": CUSTODY_SELF_SHA256,
        "implementation_freeze_sha256": freeze[
            "implementation_freeze_sha256"
        ],
        "test_member_payload_open_count": 0,
    }
    receipt = qualify_decoded_sources(
        bindings["corpus"].private_path.read_bytes(),
        bindings["train"].private_path.read_bytes(),
        bindings["dev"].private_path.read_bytes(),
        source_binding=source_binding,
    )
    _write_json_once(result_path, receipt, mode=0o644)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--formal", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.formal:
        raise SciFactQualificationError("only the frozen formal path is exposed")
    receipt = run_formal(args.project_root)
    print(receipt["status"])
    print(receipt["qualification_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
