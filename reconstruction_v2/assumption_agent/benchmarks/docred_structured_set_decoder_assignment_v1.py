"""One-shot deterministic private assignment for the frozen DocRED G8 study.

The formal controller is intentionally narrow.  It writes a durable attempt
marker before source access, opens only the three files already authorized by
the source qualifier, reuses those decoded objects for qualification and
assignment, and commits all five private blocks as one directory rename.  No
private source value is copied into its public receipt or terminal incident.

The decoded and non-formal entry points exist only for synthetic tests.  They
must be selected explicitly and never weaken the formal provenance checks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import heapq
import hmac
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

from assumption_agent.benchmarks import (
    docred_structured_set_decoder_source_qualification_v1 as qualifier,
)


VERSION = "v1"
ASSIGNMENT_RECEIPT_SCHEMA = "docred_structured_set_decoder_assignment_v1"
ATTEMPT_MARKER_SCHEMA = "docred_structured_set_decoder_assignment_attempt_v1"
TERMINAL_INCIDENT_SCHEMA = (
    "docred_structured_set_decoder_assignment_terminal_incident_v1"
)
PRIVATE_VIEW_SCHEMA = "docred_structured_set_decoder_private_view_v1"
PRIVATE_LABEL_SCHEMA = "docred_structured_set_decoder_private_labels_v1"

DESIGN_RELATIVE_PATH = Path(
    "manifests/docred_structured_set_decoder_g8_e1_design_v1.json"
)
QUALIFIER_RELATIVE_PATH = Path(
    "assumption_agent/benchmarks/"
    "docred_structured_set_decoder_source_qualification_v1.py"
)
QUALIFIER_TEST_RELATIVE_PATH = Path(
    "tests/test_docred_structured_set_decoder_source_qualification_v1.py"
)
ASSIGNMENT_RELATIVE_PATH = Path(
    "assumption_agent/benchmarks/docred_structured_set_decoder_assignment_v1.py"
)
ASSIGNMENT_TEST_RELATIVE_PATH = Path(
    "tests/test_docred_structured_set_decoder_assignment_v1.py"
)
CORE_RELATIVE_PATH = Path(
    "assumption_agent/benchmarks/docred_structured_set_decoder_g8_e1_v1.py"
)
CORE_TEST_RELATIVE_PATH = Path(
    "tests/test_docred_structured_set_decoder_g8_e1_v1.py"
)
IMPLEMENTATION_FREEZE_RELATIVE_PATH = Path(
    "manifests/docred_structured_set_decoder_pre_row_implementation_freeze_v1.json"
)
IMPLEMENTATION_FREEZE_SCHEMA = (
    "docred_structured_set_decoder_pre_row_implementation_freeze_v1"
)
FORMAL_OUTPUT_RELATIVE_PATH = Path(
    "artifacts/docred_structured_set_decoder_formal_v1"
)

FORMAL_DESIGN_COMMIT = "8fda35782ecf10d2c0f0045049d9944abf0c8c32"
FORMAL_DESIGN_SELF_SHA256 = (
    "67bbba4dc0cf62928e28f97f96cd757249400f95abddec3b3ec2f753053f3345"
)
FORMAL_DESIGN_FILE_SHA256 = (
    "3d6b5e44fa45ba05aa26912a7a73142999383d8515c72a2452d175c1eff98334"
)
FORMAL_QUALIFIER_COMMIT = "2b4ec38f6092d0972d701c59050ea5ed0dcc5788"
FORMAL_QUALIFIER_FILE_SHA256 = (
    "672629bd32fef49e11e87a32655f5e05c13dc94b13312c5ed807054d613e8df4"
)
FORMAL_QUALIFIER_TEST_FILE_SHA256 = (
    "04dc912258ab77db03ea9f043359cfeef7cbea7f7869b95e125e9409e949d744"
)

ATTEMPT_MARKER_NAME = "attempt_marker.json"
PUBLIC_RECEIPT_NAME = "assignment_receipt.json"
TERMINAL_INCIDENT_NAME = "terminal_incident.json"
SOURCE_QUALIFICATION_RECEIPT_NAME = "source_qualification_receipt.json"
PRIVATE_DIRECTORY_NAME = "private_assignment"
PRIVATE_STAGING_NAME = ".private_assignment.staging"

REQUIRED_IMPLEMENTATION_ROLE_PATHS: dict[str, Path] = {
    "design": DESIGN_RELATIVE_PATH,
    "qualifier": QUALIFIER_RELATIVE_PATH,
    "qualifier_test": QUALIFIER_TEST_RELATIVE_PATH,
    "assignment": ASSIGNMENT_RELATIVE_PATH,
    "assignment_test": ASSIGNMENT_TEST_RELATIVE_PATH,
    "g8_e1_core": CORE_RELATIVE_PATH,
    "g8_e1_core_test": CORE_TEST_RELATIVE_PATH,
}

# Insertion order is part of the frozen selector.  Family order is imported
# from the already-frozen relation-family manifest implementation.
BLOCK_SPECS: tuple[tuple[str, str, int, bool], ...] = (
    ("G_form", "train", 32, True),
    ("A_form", "train", 16, True),
    ("F_search", "train", 12, False),
    ("A_hold", "dev", 10, True),
    ("M_search", "dev", 10, True),
)
BLOCK_ORDER = tuple(spec[0] for spec in BLOCK_SPECS)
BLOCK_TO_SPLIT = {block: split for block, split, _quota, _label in BLOCK_SPECS}
BLOCK_TO_QUOTA = {block: quota for block, _split, quota, _label in BLOCK_SPECS}
BLOCK_HAS_LABELS = {
    block: has_labels for block, _split, _quota, has_labels in BLOCK_SPECS
}
TARGET_DEMANDS: dict[tuple[str, str], int] = {
    (block, family): quota
    for block, _split, quota, _label in BLOCK_SPECS
    for family in qualifier.FAMILIES
}
TOTAL_REQUIRED_ITEMS = sum(TARGET_DEMANDS.values())


class DocredAssignmentError(RuntimeError):
    """Base class for fixed assignment/controller failures."""


class OneShotRefusal(DocredAssignmentError):
    """The caller-specified output root is not pristine."""


class FormalProvenanceError(DocredAssignmentError):
    """A committed design or qualifier provenance check failed."""


class AssignmentShortfall(DocredAssignmentError):
    """The simultaneous component-capacity assignment is infeasible."""

    def __init__(self, assigned: int, required: int):
        self.assigned = assigned
        self.required = required
        super().__init__("simultaneous deterministic assignment shortfall")


class _QualificationTerminal(DocredAssignmentError):
    def __init__(self, receipt: Mapping[str, Any]):
        self.receipt = receipt
        super().__init__("aggregate source qualification did not pass")


@dataclass(frozen=True)
class _PreparedDocument:
    split: str
    split_index: int
    normalized_title_sha256: str
    document_identity: str


@dataclass(frozen=True)
class _PreparedCandidate:
    split: str
    split_index: int
    document_global_index: int
    family: str
    relation: str
    head: int
    tail: int
    label_ordinal: int
    gold_sentence_ordinals: tuple[int, ...]
    selection_digest: str
    item_id: str
    query: str
    corpus: tuple[str, ...]
    agent_sidecar: Mapping[str, Any]

    @property
    def cost(self) -> int:
        return int(self.selection_digest, 16)

    @property
    def tie_break(self) -> tuple[Any, ...]:
        return (
            self.relation,
            self.head,
            self.tail,
            self.label_ordinal,
            0 if self.split == "train" else 1,
            self.split_index,
        )


@dataclass(frozen=True)
class _EdgeChoice:
    """One deterministic component-to-target choice.

    ``payload`` is deliberately opaque so tiny synthetic graph tests can use
    the exact production min-cost implementation without private documents.
    """

    cost: int
    tie_break: tuple[Any, ...]
    payload: Any = field(compare=False)


@dataclass
class _FlowEdge:
    to: int
    reverse: int
    capacity: int
    cost: int
    original_capacity: int
    payload: Any = None
    target: tuple[str, str] | None = None


@dataclass(frozen=True)
class _FlowSolution:
    assigned_count: int
    required_count: int
    total_cost: int
    selected: Mapping[tuple[str, str], tuple[Any, ...]]


@dataclass(frozen=True)
class _AssignmentMaterials:
    view_bytes: Mapping[str, bytes]
    label_bytes: Mapping[str, bytes]
    block_aggregates: Mapping[str, Mapping[str, Any]]
    collision_component_count: int
    eligible_candidate_count: int
    min_cost_sum: int


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
        raise DocredAssignmentError("non-canonical assignment value") from exc


def _self_hashed(body: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    output = dict(body)
    output[field_name] = hashlib.sha256(_canonical_json(output)).hexdigest()
    return output


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hmac_hex(secret: bytes, domain: str, value: Any) -> str:
    return hmac.new(
        secret,
        domain.encode("ascii") + b"\x00" + _canonical_json(value),
        hashlib.sha256,
    ).hexdigest()


def _secret_commitment(secret: bytes) -> str:
    if type(secret) is not bytes or len(secret) != 32:
        raise DocredAssignmentError("selection secret must be exactly 32 bytes")
    return _sha256(b"docred-structured-set-selection-secret-v1\x00" + secret)


def _render_sentences(document: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(" ".join(sentence) for sentence in document["sents"])


def _deduplicated_aliases(cluster: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    aliases: list[str] = []
    seen: set[str] = set()
    for mention in cluster:
        surface = mention["name"]
        key = qualifier._normalize_text(surface)
        if key not in seen:
            seen.add(key)
            aliases.append(surface)
    return tuple(aliases)


def _agent_sidecar(
    vertex_set: Sequence[Sequence[Mapping[str, Any]]],
    *,
    head: int,
    tail: int,
) -> dict[str, Any]:
    entities: list[dict[str, Any]] = []
    for cluster in vertex_set:
        mentions: list[dict[str, Any]] = []
        for mention in cluster:
            mentions.append(
                {
                    "surface": mention["name"],
                    "entity_type": mention["type"],
                    "sentence_index": mention["sent_id"],
                    "token_span": [mention["pos"][0], mention["pos"][1]],
                }
            )
        entities.append({"mentions": mentions})
    return {
        "head_entity_index": head,
        "tail_entity_index": tail,
        "entities": entities,
    }


def _prepare_documents_and_candidates(
    train_payload: Sequence[Mapping[str, Any]],
    dev_payload: Sequence[Mapping[str, Any]],
    relation_metadata: Mapping[str, str],
    *,
    secret: bytes,
) -> tuple[tuple[_PreparedDocument, ...], tuple[_PreparedCandidate, ...]]:
    if type(secret) is not bytes or len(secret) != 32:
        raise DocredAssignmentError("selection secret must be exactly 32 bytes")
    documents: list[_PreparedDocument] = []
    candidates: list[_PreparedCandidate] = []
    for split, payload in (("train", train_payload), ("dev", dev_payload)):
        for split_index, document in enumerate(payload):
            rendered = _render_sentences(document)
            normalized_title = qualifier._normalize_text(document["title"])
            document_identity = qualifier._stable_hash(
                {
                    "normalized_title": normalized_title,
                    "rendered_sentences": rendered,
                }
            )
            prepared_document = _PreparedDocument(
                split=split,
                split_index=split_index,
                normalized_title_sha256=hashlib.sha256(
                    normalized_title.encode("utf-8")
                ).hexdigest(),
                document_identity=document_identity,
            )
            document_global_index = len(documents)
            documents.append(prepared_document)
            vertex_set = document["vertexSet"]
            aliases = tuple(_deduplicated_aliases(cluster) for cluster in vertex_set)
            if len(rendered) < qualifier.MIN_NONEMPTY_SENTENCE_COUNT:
                continue
            for label_ordinal, label in enumerate(document["labels"]):
                relation = label["r"]
                family = qualifier.PROPERTY_TO_FAMILY.get(relation)
                evidence = tuple(sorted(set(label["evidence"])))
                if (
                    family is None
                    or not (
                        qualifier.MIN_GOLD_SENTENCE_COUNT
                        <= len(evidence)
                        <= qualifier.MAX_GOLD_SENTENCE_COUNT
                    )
                ):
                    continue
                head = label["h"]
                tail = label["t"]
                query = (
                    "HEAD: "
                    + " | ".join(aliases[head])
                    + "\nRELATION: "
                    + relation_metadata[relation]
                    + "\nTAIL: "
                    + " | ".join(aliases[tail])
                )
                selection_value = {
                    "split": split,
                    "family": family,
                    "document_identity": document_identity,
                    "h": head,
                    "t": tail,
                    "r": relation,
                    "label_ordinal": label_ordinal,
                }
                selection_digest = _hmac_hex(
                    secret, "docred-assignment-candidate-v1", selection_value
                )
                item_id = "i_" + _hmac_hex(
                    secret, "docred-opaque-item-id-v1", selection_value
                )
                candidates.append(
                    _PreparedCandidate(
                        split=split,
                        split_index=split_index,
                        document_global_index=document_global_index,
                        family=family,
                        relation=relation,
                        head=head,
                        tail=tail,
                        label_ordinal=label_ordinal,
                        gold_sentence_ordinals=evidence,
                        selection_digest=selection_digest,
                        item_id=item_id,
                        query=query,
                        corpus=rendered,
                        agent_sidecar=_agent_sidecar(
                            vertex_set,
                            head=head,
                            tail=tail,
                        ),
                    )
                )
    return tuple(documents), tuple(candidates)


class _UnionFind:
    def __init__(self, size: int):
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


def _collision_components(
    documents: Sequence[_PreparedDocument],
) -> tuple[tuple[int, ...], dict[int, int]]:
    union_find = _UnionFind(len(documents))
    title_first: dict[str, int] = {}
    document_first: dict[str, int] = {}
    for index, document in enumerate(documents):
        for value, seen in (
            (document.normalized_title_sha256, title_first),
            (document.document_identity, document_first),
        ):
            previous = seen.setdefault(value, index)
            union_find.union(index, previous)
    grouped: dict[int, list[int]] = {}
    for index in range(len(documents)):
        grouped.setdefault(union_find.find(index), []).append(index)
    ordered_groups = sorted(
        grouped.values(),
        key=lambda indices: min(
            (
                documents[index].document_identity,
                documents[index].normalized_title_sha256,
                0 if documents[index].split == "train" else 1,
                documents[index].split_index,
            )
            for index in indices
        ),
    )
    document_to_component: dict[int, int] = {}
    for component, indices in enumerate(ordered_groups):
        for index in indices:
            document_to_component[index] = component
    return tuple(tuple(indices) for indices in ordered_groups), document_to_component


def _add_edge(
    graph: list[list[_FlowEdge]],
    left: int,
    right: int,
    capacity: int,
    cost: int,
    *,
    payload: Any = None,
    target: tuple[str, str] | None = None,
) -> None:
    forward = _FlowEdge(
        to=right,
        reverse=len(graph[right]),
        capacity=capacity,
        cost=cost,
        original_capacity=capacity,
        payload=payload,
        target=target,
    )
    reverse = _FlowEdge(
        to=left,
        reverse=len(graph[left]),
        capacity=0,
        cost=-cost,
        original_capacity=0,
    )
    graph[left].append(forward)
    graph[right].append(reverse)


def _deterministic_min_cost_assignment(
    component_choices: Mapping[
        int, Mapping[tuple[str, str], _EdgeChoice]
    ],
    demands: Mapping[tuple[str, str], int],
) -> _FlowSolution:
    """Solve component-cap-one assignment, maximizing flow before cost.

    The successive shortest augmenting path implementation is deterministic:
    components are sorted, targets retain the frozen mapping order, and equal
    distances retain the first discovered predecessor.
    """

    component_ids = tuple(sorted(component_choices))
    targets = tuple(demands)
    if any(type(demand) is not int or demand < 0 for demand in demands.values()):
        raise DocredAssignmentError("invalid target demand")
    source = 0
    component_offset = 1
    target_offset = component_offset + len(component_ids)
    sink = target_offset + len(targets)
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    component_node = {
        component: component_offset + ordinal
        for ordinal, component in enumerate(component_ids)
    }
    target_node = {
        target: target_offset + ordinal for ordinal, target in enumerate(targets)
    }
    for component in component_ids:
        node = component_node[component]
        _add_edge(graph, source, node, 1, 0)
        choices = component_choices[component]
        for target in targets:
            choice = choices.get(target)
            if choice is None:
                continue
            if type(choice.cost) is not int or choice.cost < 0:
                raise DocredAssignmentError("invalid assignment cost")
            _add_edge(
                graph,
                node,
                target_node[target],
                1,
                choice.cost,
                payload=choice.payload,
                target=target,
            )
    for target in targets:
        _add_edge(graph, target_node[target], sink, demands[target], 0)

    required = sum(demands.values())
    flow = 0
    total_cost = 0
    potential = [0] * len(graph)
    infinity: int | None = None
    while flow < required:
        distance: list[int | None] = [infinity] * len(graph)
        previous_node = [-1] * len(graph)
        previous_edge = [-1] * len(graph)
        distance[source] = 0
        queue: list[tuple[int, int]] = [(0, source)]
        while queue:
            current_distance, node = heapq.heappop(queue)
            if distance[node] != current_distance:
                continue
            for edge_index, edge in enumerate(graph[node]):
                if edge.capacity <= 0:
                    continue
                reduced_cost = edge.cost + potential[node] - potential[edge.to]
                candidate_distance = current_distance + reduced_cost
                if (
                    distance[edge.to] is None
                    or candidate_distance < distance[edge.to]
                ):
                    distance[edge.to] = candidate_distance
                    previous_node[edge.to] = node
                    previous_edge[edge.to] = edge_index
                    heapq.heappush(queue, (candidate_distance, edge.to))
        if distance[sink] is None:
            break
        for node, value in enumerate(distance):
            if value is not None:
                potential[node] += value
        node = sink
        path_cost = 0
        while node != source:
            parent = previous_node[node]
            if parent < 0:
                raise AssertionError("broken augmenting path")
            edge = graph[parent][previous_edge[node]]
            path_cost += edge.cost
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = parent
        flow += 1
        total_cost += path_cost

    selected: dict[tuple[str, str], list[Any]] = {
        target: [] for target in targets
    }
    for component in component_ids:
        for edge in graph[component_node[component]]:
            if (
                edge.payload is not None
                and edge.target is not None
                and edge.original_capacity == 1
                and edge.capacity == 0
            ):
                selected[edge.target].append(edge.payload)
    return _FlowSolution(
        assigned_count=flow,
        required_count=required,
        total_cost=total_cost,
        selected={target: tuple(selected[target]) for target in targets},
    )


def _component_target_choices(
    documents: Sequence[_PreparedDocument],
    candidates: Sequence[_PreparedCandidate],
) -> tuple[
    Mapping[int, Mapping[tuple[str, str], _EdgeChoice]],
    int,
]:
    components, document_to_component = _collision_components(documents)
    best: dict[int, dict[tuple[str, str], _PreparedCandidate]] = {
        component: {} for component in range(len(components))
    }
    for candidate in candidates:
        component = document_to_component[candidate.document_global_index]
        for block, split, _quota, _has_labels in BLOCK_SPECS:
            if split != candidate.split:
                continue
            target = (block, candidate.family)
            incumbent = best[component].get(target)
            if incumbent is None or (
                candidate.selection_digest,
                candidate.tie_break,
            ) < (
                incumbent.selection_digest,
                incumbent.tie_break,
            ):
                best[component][target] = candidate
    choices = {
        component: {
            target: _EdgeChoice(
                cost=candidate.cost,
                tie_break=candidate.tie_break,
                payload=candidate,
            )
            for target, candidate in targets.items()
        }
        for component, targets in best.items()
        if targets
    }
    return choices, len(components)


def _view_record(candidate: _PreparedCandidate) -> dict[str, Any]:
    return {
        "item_id": candidate.item_id,
        "query": candidate.query,
        "corpus": list(candidate.corpus),
        "agent_sidecar": candidate.agent_sidecar,
    }


def _label_record(candidate: _PreparedCandidate) -> dict[str, Any]:
    return {
        "item_id": candidate.item_id,
        "gold_sentence_ordinals": list(candidate.gold_sentence_ordinals),
    }


def _build_assignment_materials(
    train_payload: Sequence[Mapping[str, Any]],
    dev_payload: Sequence[Mapping[str, Any]],
    relation_metadata: Mapping[str, str],
    *,
    secret: bytes,
) -> _AssignmentMaterials:
    documents, candidates = _prepare_documents_and_candidates(
        train_payload,
        dev_payload,
        relation_metadata,
        secret=secret,
    )
    component_choices, component_count = _component_target_choices(
        documents, candidates
    )
    solution = _deterministic_min_cost_assignment(
        component_choices,
        TARGET_DEMANDS,
    )
    if solution.assigned_count != solution.required_count:
        raise AssignmentShortfall(
            solution.assigned_count,
            solution.required_count,
        )

    family_order = {family: index for index, family in enumerate(qualifier.FAMILIES)}
    view_bytes: dict[str, bytes] = {}
    label_bytes: dict[str, bytes] = {}
    block_aggregates: dict[str, dict[str, Any]] = {}
    for block in BLOCK_ORDER:
        selected: list[tuple[str, _PreparedCandidate]] = []
        family_counts: dict[str, int] = {}
        for family in qualifier.FAMILIES:
            values = solution.selected[(block, family)]
            if len(values) != BLOCK_TO_QUOTA[block]:
                raise AssertionError("min-cost flow violated an exact target quota")
            family_counts[family] = len(values)
            selected.extend((family, candidate) for candidate in values)
        selected.sort(key=lambda pair: (family_order[pair[0]], pair[1].item_id))
        view = {
            "schema": PRIVATE_VIEW_SCHEMA,
            "version": VERSION,
            "items": [_view_record(candidate) for _family, candidate in selected],
        }
        view_raw = _canonical_json(view) + b"\n"
        view_bytes[block] = view_raw
        aggregate: dict[str, Any] = {
            "view_item_count": len(selected),
            "label_item_count": len(selected) if BLOCK_HAS_LABELS[block] else 0,
            "family_item_counts": family_counts,
            "view_file_sha256": _sha256(view_raw),
            "labels_file_created": BLOCK_HAS_LABELS[block],
        }
        if BLOCK_HAS_LABELS[block]:
            labels = {
                "schema": PRIVATE_LABEL_SCHEMA,
                "version": VERSION,
                "items": [
                    _label_record(candidate) for _family, candidate in selected
                ],
            }
            label_raw = _canonical_json(labels) + b"\n"
            label_bytes[block] = label_raw
            aggregate["labels_file_sha256"] = _sha256(label_raw)
        block_aggregates[block] = aggregate
    return _AssignmentMaterials(
        view_bytes=view_bytes,
        label_bytes=label_bytes,
        block_aggregates=block_aggregates,
        collision_component_count=component_count,
        eligible_candidate_count=len(candidates),
        min_cost_sum=solution.total_cost,
    )


def _regular_public_bytes(path: Path, *, label: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise FormalProvenanceError(f"{label} unavailable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise FormalProvenanceError(f"{label} is not a regular file")
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            raw = handle.read()
            after_open = os.fstat(handle.fileno())
    except OSError as exc:
        raise FormalProvenanceError(f"{label} read failed") from exc
    try:
        after = path.lstat()
    except OSError as exc:
        raise FormalProvenanceError(f"{label} post-read stat failed") from exc
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size)
    if not (
        identity(before)
        == identity(opened)
        == identity(after_open)
        == identity(after)
        and stat.S_ISREG(after.st_mode)
        and not stat.S_ISLNK(after.st_mode)
        and len(raw) == after.st_size
    ):
        raise FormalProvenanceError(f"{label} changed during read")
    return raw


def _git(
    repository_root: Path,
    arguments: Sequence[str],
    *,
    expect_output: bool = True,
) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise FormalProvenanceError("git provenance command unavailable") from exc
    if completed.returncode != 0:
        raise FormalProvenanceError("git provenance check failed")
    return completed.stdout if expect_output else b""


def _validate_implementation_freeze(
    project_root: Path,
    repository_root: Path,
) -> dict[str, str]:
    freeze_path = project_root / IMPLEMENTATION_FREEZE_RELATIVE_PATH
    raw = _regular_public_bytes(
        freeze_path,
        label="pre-row implementation freeze",
    )
    payload = qualifier._strict_json(raw, label="pre-row implementation freeze")
    if not isinstance(payload, Mapping):
        raise FormalProvenanceError("implementation freeze root drifted")
    body = dict(payload)
    declared = body.pop("implementation_freeze_sha256", None)
    if (
        not isinstance(declared, str)
        or qualifier._SHA256_RE.fullmatch(declared) is None
        or _sha256(qualifier._canonical_json(body)) != declared
        or payload.get("schema") != IMPLEMENTATION_FREEZE_SCHEMA
        or payload.get("version") != VERSION
        or payload.get("status")
        != "frozen_before_formal_source_qualification_or_private_assignment"
    ):
        raise FormalProvenanceError("implementation freeze self-hash drifted")
    design_binding = payload.get("design_binding")
    if design_binding != {
        "commit": FORMAL_DESIGN_COMMIT,
        "file_sha256": FORMAL_DESIGN_FILE_SHA256,
        "self_sha256": FORMAL_DESIGN_SELF_SHA256,
    }:
        raise FormalProvenanceError("implementation freeze design binding drifted")
    qualifier_binding = payload.get("qualifier_binding")
    if qualifier_binding != {
        "commit": FORMAL_QUALIFIER_COMMIT,
        "file_sha256": FORMAL_QUALIFIER_FILE_SHA256,
        "test_file_sha256": FORMAL_QUALIFIER_TEST_FILE_SHA256,
    }:
        raise FormalProvenanceError(
            "implementation freeze qualifier binding drifted"
        )
    implementation_commit = payload.get("implementation_commit")
    if (
        not isinstance(implementation_commit, str)
        or len(implementation_commit) != 40
        or any(character not in "0123456789abcdef" for character in implementation_commit)
    ):
        raise FormalProvenanceError("implementation commit binding drifted")
    binding = payload.get("implementation_binding")
    if not isinstance(binding, Mapping):
        raise FormalProvenanceError("implementation file binding drifted")
    files = binding.get("files")
    if (
        binding.get("file_count") != len(REQUIRED_IMPLEMENTATION_ROLE_PATHS)
        or not isinstance(files, list)
        or len(files) != len(REQUIRED_IMPLEMENTATION_ROLE_PATHS)
    ):
        raise FormalProvenanceError("implementation file count drifted")
    expected_rows: list[tuple[str, Path, str]] = []
    for row, (expected_role, expected_path) in zip(
        files,
        REQUIRED_IMPLEMENTATION_ROLE_PATHS.items(),
        strict=True,
    ):
        if not isinstance(row, Mapping):
            raise FormalProvenanceError("implementation file row drifted")
        sha256 = row.get("sha256")
        if (
            set(row) != {"role", "relative_path", "sha256"}
            or row.get("role") != expected_role
            or row.get("relative_path") != expected_path.as_posix()
            or not isinstance(sha256, str)
            or qualifier._SHA256_RE.fullmatch(sha256) is None
        ):
            raise FormalProvenanceError("implementation role registry drifted")
        if _sha256(
            _regular_public_bytes(
                project_root / expected_path,
                label=f"implementation role {expected_role}",
            )
        ) != sha256:
            raise FormalProvenanceError("implementation working blob drifted")
        expected_rows.append((expected_role, expected_path, sha256))
    claim = payload.get("claim_boundary")
    if not isinstance(claim, Mapping) or claim != {
        "docred_train_or_dev_rows_opened": False,
        "formal_source_qualification_run": False,
        "private_selection_secret_generated": False,
        "private_cohort_selected": False,
        "retrieval_or_efficacy_score_run": False,
        "online_or_network_evaluation_used": False,
    }:
        raise FormalProvenanceError("implementation freeze claim boundary drifted")

    if _git(repository_root, ("cat-file", "-t", implementation_commit)).strip() != b"commit":
        raise FormalProvenanceError("implementation provenance is not a commit")
    _git(
        repository_root,
        ("merge-base", "--is-ancestor", implementation_commit, "HEAD"),
        expect_output=False,
    )
    repository_paths = [
        "reconstruction_v2/" + path.as_posix()
        for _role, path, _sha256_value in expected_rows
    ]
    freeze_repository_path = (
        "reconstruction_v2/" + IMPLEMENTATION_FREEZE_RELATIVE_PATH.as_posix()
    )
    for repository_path in (*repository_paths, freeze_repository_path):
        _git(
            repository_root,
            ("ls-files", "--error-unmatch", "--", repository_path),
        )
    _git(
        repository_root,
        ("diff", "--quiet", "HEAD", "--", *repository_paths, freeze_repository_path),
        expect_output=False,
    )
    for _role, relative_path, expected_sha256 in expected_rows:
        repository_path = "reconstruction_v2/" + relative_path.as_posix()
        if _sha256(
            _git(repository_root, ("show", f"{implementation_commit}:{repository_path}"))
        ) != expected_sha256:
            raise FormalProvenanceError("implementation committed blob drifted")
    return {
        "implementation_commit": implementation_commit,
        "implementation_freeze_file_sha256": _sha256(raw),
        "implementation_freeze_self_sha256": declared,
    }


def _validate_formal_provenance(project_root: Path) -> dict[str, str]:
    """Bind exact public blobs and committed ancestor provenance.

    This function opens no DocRED source file.  It is deliberately executed
    only after the durable attempt marker and before qualifier source access.
    """

    root = project_root.resolve(strict=True)
    design_raw = _regular_public_bytes(
        root / DESIGN_RELATIVE_PATH,
        label="frozen design",
    )
    qualifier_raw = _regular_public_bytes(
        root / QUALIFIER_RELATIVE_PATH,
        label="frozen qualifier implementation",
    )
    qualifier_test_raw = _regular_public_bytes(
        root / QUALIFIER_TEST_RELATIVE_PATH,
        label="frozen qualifier test",
    )
    if _sha256(design_raw) != FORMAL_DESIGN_FILE_SHA256:
        raise FormalProvenanceError("frozen design blob drifted")
    if _sha256(qualifier_raw) != FORMAL_QUALIFIER_FILE_SHA256:
        raise FormalProvenanceError("frozen qualifier blob drifted")
    if _sha256(qualifier_test_raw) != FORMAL_QUALIFIER_TEST_FILE_SHA256:
        raise FormalProvenanceError("frozen qualifier test blob drifted")
    design = qualifier._strict_json(design_raw, label="frozen design")
    if not isinstance(design, Mapping):
        raise FormalProvenanceError("frozen design root drifted")
    design_body = dict(design)
    declared_self_hash = design_body.pop("design_sha256", None)
    if (
        declared_self_hash != FORMAL_DESIGN_SELF_SHA256
        or _sha256(qualifier._canonical_json(design_body))
        != FORMAL_DESIGN_SELF_SHA256
    ):
        raise FormalProvenanceError("frozen design self-hash drifted")

    repository_raw = _git(root, ("rev-parse", "--show-toplevel"))
    try:
        repository_root = Path(repository_raw.decode("utf-8").strip()).resolve(
            strict=True
        )
    except (UnicodeDecodeError, OSError) as exc:
        raise FormalProvenanceError("repository root provenance drifted") from exc
    if (repository_root / "reconstruction_v2").resolve() != root:
        raise FormalProvenanceError("formal project root is not reconstruction_v2")
    for commit in (FORMAL_DESIGN_COMMIT, FORMAL_QUALIFIER_COMMIT):
        if _git(repository_root, ("cat-file", "-t", commit)).strip() != b"commit":
            raise FormalProvenanceError("bound provenance object is not a commit")
        _git(
            repository_root,
            ("merge-base", "--is-ancestor", commit, "HEAD"),
            expect_output=False,
        )
    _git(
        repository_root,
        (
            "merge-base",
            "--is-ancestor",
            FORMAL_DESIGN_COMMIT,
            FORMAL_QUALIFIER_COMMIT,
        ),
        expect_output=False,
    )
    committed_blobs = (
        (
            FORMAL_DESIGN_COMMIT,
            "reconstruction_v2/" + DESIGN_RELATIVE_PATH.as_posix(),
            FORMAL_DESIGN_FILE_SHA256,
        ),
        (
            FORMAL_QUALIFIER_COMMIT,
            "reconstruction_v2/" + QUALIFIER_RELATIVE_PATH.as_posix(),
            FORMAL_QUALIFIER_FILE_SHA256,
        ),
        (
            FORMAL_QUALIFIER_COMMIT,
            "reconstruction_v2/" + QUALIFIER_TEST_RELATIVE_PATH.as_posix(),
            FORMAL_QUALIFIER_TEST_FILE_SHA256,
        ),
    )
    for commit, path, expected_sha256 in committed_blobs:
        if _sha256(_git(repository_root, ("show", f"{commit}:{path}"))) != (
            expected_sha256
        ):
            raise FormalProvenanceError("committed blob binding drifted")
    freeze_binding = _validate_implementation_freeze(root, repository_root)
    return {
        "design_commit": FORMAL_DESIGN_COMMIT,
        "design_file_sha256": FORMAL_DESIGN_FILE_SHA256,
        "design_self_sha256": FORMAL_DESIGN_SELF_SHA256,
        "qualifier_commit": FORMAL_QUALIFIER_COMMIT,
        "qualifier_file_sha256": FORMAL_QUALIFIER_FILE_SHA256,
        "qualifier_test_file_sha256": FORMAL_QUALIFIER_TEST_FILE_SHA256,
        **freeze_binding,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short durable write")
        view = view[written:]


def _exclusive_write(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, raw)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        try:
            path.unlink()
        except OSError:
            pass
        raise
    else:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _atomic_write(path: Path, raw: bytes) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise OneShotRefusal("durable output already exists")
    _exclusive_write(temporary, raw)
    os.rename(temporary, path)
    _fsync_directory(path.parent)


def _require_empty_output_root(
    output_root: str | Path,
    *,
    require_mode_0700: bool = False,
) -> Path:
    path = Path(output_root)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise OneShotRefusal("caller output root must already exist") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise OneShotRefusal("caller output root must be a real directory")
    if require_mode_0700 and stat.S_IMODE(metadata.st_mode) != 0o700:
        raise OneShotRefusal("formal output root must have mode 0700")
    root = path.resolve(strict=True)
    try:
        if next(root.iterdir(), None) is not None:
            raise OneShotRefusal("caller output root is not empty")
    except OSError as exc:
        raise OneShotRefusal("caller output root cannot be inspected") from exc
    return root


def _attempt_marker(*, formal_identity_enforced: bool) -> dict[str, Any]:
    body = {
        "schema": ATTEMPT_MARKER_SCHEMA,
        "version": VERSION,
        "status": "attempt_consumed_no_replay",
        "formal_identity_enforced": formal_identity_enforced,
        "design_commit": FORMAL_DESIGN_COMMIT,
        "design_file_sha256": FORMAL_DESIGN_FILE_SHA256,
        "design_self_sha256": FORMAL_DESIGN_SELF_SHA256,
        "qualifier_commit": FORMAL_QUALIFIER_COMMIT,
        "qualifier_file_sha256": FORMAL_QUALIFIER_FILE_SHA256,
        "qualifier_test_file_sha256": FORMAL_QUALIFIER_TEST_FILE_SHA256,
    }
    return _self_hashed(body, "attempt_marker_sha256")


def _commit_private_outputs(
    output_root: Path,
    materials: _AssignmentMaterials,
    *,
    secret: bytes,
) -> None:
    staging = output_root / PRIVATE_STAGING_NAME
    destination = output_root / PRIVATE_DIRECTORY_NAME
    if staging.exists() or destination.exists():
        raise OneShotRefusal("private output already exists")
    staging.mkdir(mode=0o700)
    try:
        _exclusive_write(staging / "selection_secret.bin", secret)
        for block in BLOCK_ORDER:
            block_root = staging / block
            block_root.mkdir(mode=0o700)
            _exclusive_write(block_root / "view.json", materials.view_bytes[block])
            if BLOCK_HAS_LABELS[block]:
                _exclusive_write(
                    block_root / "labels.json",
                    materials.label_bytes[block],
                )
            _fsync_directory(block_root)
        _fsync_directory(staging)
        os.rename(staging, destination)
        _fsync_directory(output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        _fsync_directory(output_root)
        raise


def _source_binding(
    specs: Mapping[str, Any],
    manifest_binding: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **manifest_binding,
        "official_git_commit": qualifier.FORMAL_OFFICIAL_GIT_COMMIT,
        "source_freeze_commit": qualifier.FORMAL_SOURCE_FREEZE_COMMIT,
        "train_file_sha256": specs["train"].sha256,
        "train_file_size": specs["train"].size,
        "dev_file_sha256": specs["dev"].sha256,
        "dev_file_size": specs["dev"].size,
        "relation_metadata_file_sha256": specs["relation_metadata"].sha256,
        "relation_metadata_file_size": specs["relation_metadata"].size,
    }


def _success_receipt(
    *,
    formal_identity_enforced: bool,
    qualification_receipt: Mapping[str, Any],
    qualification_receipt_file_sha256: str,
    implementation_binding: Mapping[str, Any],
    materials: _AssignmentMaterials,
    selection_secret_commitment_sha256: str,
) -> dict[str, Any]:
    private_commitment = {
        block: dict(materials.block_aggregates[block]) for block in BLOCK_ORDER
    }
    body: dict[str, Any] = {
        "schema": ASSIGNMENT_RECEIPT_SCHEMA,
        "version": VERSION,
        "status": "passed_assignment_committed_no_private_values",
        "formal_identity_enforced": formal_identity_enforced,
        "implementation_binding": dict(implementation_binding),
        "qualification_receipt_sha256": qualification_receipt[
            "qualification_sha256"
        ],
        "qualification_receipt_file_sha256": qualification_receipt_file_sha256,
        "assignment_aggregate": {
            "assigned_item_count": TOTAL_REQUIRED_ITEMS,
            "eligible_candidate_count": materials.eligible_candidate_count,
            "collision_component_count": materials.collision_component_count,
            "block_aggregates": private_commitment,
            "private_output_commitment_sha256": _sha256(
                _canonical_json(private_commitment)
            ),
            "selection_secret_commitment_sha256": (
                selection_secret_commitment_sha256
            ),
        },
        "opened_content_boundary": {
            "train_annotated_open_count": 1,
            "dev_open_count": 1,
            "relation_metadata_open_count": 1,
            "official_test_open_count": 0,
            "train_distant_open_count": 0,
            "source_decode_pass_count": 1,
            "source_reopen_for_assignment_count": 0,
            "m_search_private_view_open_count": 0,
            "m_search_private_label_open_count": 0,
        },
        "private_output_boundary": {
            "five_views_committed_together": True,
            "four_label_packs_committed_together": True,
            "F_search_label_pack_created": False,
            "M_search_view_precommitted_unopened": True,
            "M_search_labels_precommitted_unopened": True,
            "selection_secret_emitted_publicly": False,
            "selection_secret_commitment_emitted_publicly": True,
            "selection_secret_persisted_private_mode_0600": True,
            "item_document_title_text_alias_triple_ordinal_or_per_document_hash_emitted_publicly": False,
        },
    }
    return _self_hashed(body, "assignment_receipt_sha256")


def _incident_category(exception: Exception) -> str:
    if isinstance(exception, FormalProvenanceError):
        return "formal_provenance_invalid"
    if isinstance(exception, _QualificationTerminal):
        return "source_qualification_terminal"
    if isinstance(exception, AssignmentShortfall):
        return "deterministic_assignment_shortfall"
    if isinstance(exception, qualifier.DocredSourceQualificationError):
        return "source_or_manifest_contract_invalid"
    if isinstance(exception, OneShotRefusal):
        return "preexisting_output_refusal"
    return "implementation_or_infrastructure_exception"


def _terminal_incident(
    *,
    formal_identity_enforced: bool,
    stage: str,
    exception: Exception,
    opened_counts: Mapping[str, int],
    qualification_receipt: Mapping[str, Any] | None,
    qualification_receipt_file_sha256: str | None,
    private_output_committed: bool,
) -> dict[str, Any]:
    qualification_aggregate: dict[str, Any] = {
        "qualification_receipt_available": qualification_receipt is not None
    }
    if qualification_receipt is not None:
        qualification_aggregate.update(
            {
                "qualification_status": qualification_receipt["status"],
                "qualification_receipt_sha256": qualification_receipt[
                    "qualification_sha256"
                ],
                "terminal_reason_counts": qualification_receipt[
                    "terminal_reason_counts"
                ],
                "qualification_receipt_file_sha256": (
                    qualification_receipt_file_sha256
                ),
            }
        )
    failure_aggregate: dict[str, Any] = {}
    if isinstance(exception, AssignmentShortfall):
        failure_aggregate = {
            "assigned_item_count": exception.assigned,
            "required_item_count": exception.required,
            "shortfall_item_count": exception.required - exception.assigned,
        }
    body = {
        "schema": TERMINAL_INCIDENT_SCHEMA,
        "version": VERSION,
        "status": "terminal_no_replay_no_private_values",
        "formal_identity_enforced": formal_identity_enforced,
        "failure_stage": stage,
        "failure_category": _incident_category(exception),
        "opened_content_counts": dict(opened_counts),
        "qualification_aggregate": qualification_aggregate,
        "failure_aggregate": failure_aggregate,
        "private_output_committed_before_failure": private_output_committed,
        "secret_item_document_title_text_alias_triple_ordinal_or_per_document_hash_emitted": False,
        "same_source_retry_authorized": False,
    }
    return _self_hashed(body, "terminal_incident_sha256")


def _run_one_shot_controller(
    project_root: str | Path,
    output_root: str | Path,
    *,
    formal_identity_enforced: bool,
    secret_factory: Callable[[int], bytes] | None = None,
) -> dict[str, Any]:
    if type(formal_identity_enforced) is not bool:
        raise TypeError("formal_identity_enforced must be an exact boolean")
    if formal_identity_enforced and secret_factory is not None:
        raise FormalProvenanceError(
            "formal execution forbids a caller-supplied secret factory"
        )
    root = Path(project_root).resolve(strict=True)
    output = _require_empty_output_root(
        output_root,
        require_mode_0700=formal_identity_enforced,
    )
    if formal_identity_enforced:
        expected_output = root / FORMAL_OUTPUT_RELATIVE_PATH
        if output != expected_output:
            raise OneShotRefusal("formal output root is not the frozen path")
    marker = _attempt_marker(
        formal_identity_enforced=formal_identity_enforced
    )
    _exclusive_write(
        output / ATTEMPT_MARKER_NAME,
        _canonical_json(marker) + b"\n",
    )

    stage = "attempt_marker_committed"
    opened_counts = {
        "train_annotated_open_count": 0,
        "dev_open_count": 0,
        "relation_metadata_open_count": 0,
        "official_test_open_count": 0,
        "train_distant_open_count": 0,
    }
    qualification_receipt: Mapping[str, Any] | None = None
    qualification_receipt_file_sha256: str | None = None
    private_output_committed = False
    implementation_binding: dict[str, Any] = {
        "design_commit": FORMAL_DESIGN_COMMIT,
        "design_file_sha256": FORMAL_DESIGN_FILE_SHA256,
        "design_self_sha256": FORMAL_DESIGN_SELF_SHA256,
        "qualifier_commit": FORMAL_QUALIFIER_COMMIT,
        "qualifier_file_sha256": FORMAL_QUALIFIER_FILE_SHA256,
        "qualifier_test_file_sha256": FORMAL_QUALIFIER_TEST_FILE_SHA256,
        "pre_row_implementation_freeze_verified": False,
    }
    try:
        if formal_identity_enforced:
            stage = "formal_provenance_validation"
            implementation_binding.update(_validate_formal_provenance(root))
            implementation_binding["pre_row_implementation_freeze_verified"] = True

        stage = "frozen_source_contract_validation"
        specs, manifest_binding = qualifier._validate_frozen_contracts(
            root,
            enforce_formal_identity=formal_identity_enforced,
        )
        raw_sources: dict[str, bytes] = {}
        for key, public_count_key in (
            ("relation_metadata", "relation_metadata_open_count"),
            ("train", "train_annotated_open_count"),
            ("dev", "dev_open_count"),
        ):
            stage = "single_source_open"
            spec = specs[key]
            opened_counts[public_count_key] += 1
            raw_sources[key] = qualifier._read_bound_source(
                root / spec.relative_path,
                spec,
                label=f"authorized {key} source",
            )

        stage = "single_source_decode"
        train_payload = qualifier._strict_json(
            raw_sources["train"], label="authorized TRAIN source"
        )
        dev_payload = qualifier._strict_json(
            raw_sources["dev"], label="authorized DEV source"
        )
        relation_metadata_payload = qualifier._strict_json(
            raw_sources["relation_metadata"],
            label="authorized relation metadata",
        )
        del raw_sources

        stage = "aggregate_source_qualification"
        qualification_receipt = qualifier._qualify_decoded_sources(
            train_payload,
            dev_payload,
            relation_metadata_payload,
            source_binding=_source_binding(specs, manifest_binding),
            formal_identity_enforced=formal_identity_enforced,
        )
        qualification_raw = _canonical_json(qualification_receipt) + b"\n"
        _atomic_write(output / SOURCE_QUALIFICATION_RECEIPT_NAME, qualification_raw)
        qualification_receipt_file_sha256 = _sha256(qualification_raw)
        if qualification_receipt["status"] != (
            "passed_source_qualification_no_selection"
        ):
            raise _QualificationTerminal(qualification_receipt)

        stage = "one_time_secret_generation"
        factory = os.urandom if secret_factory is None else secret_factory
        secret = factory(32)
        if type(secret) is not bytes or len(secret) != 32:
            raise DocredAssignmentError(
                "secret factory did not return exactly 32 bytes"
            )
        secret_commitment = _secret_commitment(secret)

        stage = "deterministic_private_assignment"
        relation_metadata = qualifier._parse_relation_metadata(
            relation_metadata_payload
        )
        materials = _build_assignment_materials(
            train_payload,
            dev_payload,
            relation_metadata,
            secret=secret,
        )
        stage = "atomic_private_commit"
        _commit_private_outputs(output, materials, secret=secret)
        del secret
        private_output_committed = True

        stage = "aggregate_public_receipt_commit"
        receipt = _success_receipt(
            formal_identity_enforced=formal_identity_enforced,
            qualification_receipt=qualification_receipt,
            qualification_receipt_file_sha256=qualification_receipt_file_sha256,
            implementation_binding=implementation_binding,
            materials=materials,
            selection_secret_commitment_sha256=secret_commitment,
        )
        _atomic_write(
            output / PUBLIC_RECEIPT_NAME,
            _canonical_json(receipt) + b"\n",
        )
        return receipt
    except Exception as exception:
        staging = output / PRIVATE_STAGING_NAME
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
            _fsync_directory(output)
        incident = _terminal_incident(
            formal_identity_enforced=formal_identity_enforced,
            stage=stage,
            exception=exception,
            opened_counts=opened_counts,
            qualification_receipt=qualification_receipt,
            qualification_receipt_file_sha256=(
                qualification_receipt_file_sha256
            ),
            private_output_committed=private_output_committed,
        )
        _atomic_write(
            output / TERMINAL_INCIDENT_NAME,
            _canonical_json(incident) + b"\n",
        )
        return incident


def run_synthetic_one_shot(
    project_root: str | Path,
    output_root: str | Path,
    *,
    secret_factory: Callable[[int], bytes] | None = None,
) -> dict[str, Any]:
    """Explicit non-formal controller for synthetic fixtures only."""

    return _run_one_shot_controller(
        project_root,
        output_root,
        formal_identity_enforced=False,
        secret_factory=secret_factory,
    )


def run_formal_one_shot(
    project_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Consume the formal source epoch exactly once."""

    return _run_one_shot_controller(
        project_root,
        output_root,
        formal_identity_enforced=True,
        secret_factory=None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Formal one-shot DocRED structured-set assignment"
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = run_formal_one_shot(args.project_root, args.output_root)
    sys.stdout.buffer.write(_canonical_json(receipt) + b"\n")
    return int(receipt["status"] != "passed_assignment_committed_no_private_values")


if __name__ == "__main__":
    raise SystemExit(main())
