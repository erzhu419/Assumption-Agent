"""One-shot private-HMAC acquisition for the frozen MAVEN-ERE G8/E1 study."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import heapq
import hmac
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping, Sequence
import unicodedata

from assumption_agent.benchmarks import (
    maven_ere_relation_context_source_qualification_v1 as qualification,
)


VERSION = "v1"
SCHEMA = "maven_ere_g8_e1_acquisition_v1"
DESIGN_RELATIVE = Path("manifests/maven_ere_g8_e1_formal_design_v1.json")
DESIGN_FILE_SHA256 = "e8ae662809ead29f2a5c08fd0ca44970ef8916ccda3741f480b87b571f44ddf4"
DESIGN_SELF_SHA256 = "314a9804d32a3c3fb848e0100bc62bc693a468e8e3ac09c9baf018c7cfeee417"
DESIGN_COMMIT = "5b6232a927205e85ae78c9726d692819661ad3c2"

TRAIN_RELATIVE = qualification.TRAIN_RELATIVE_PATH
VALID_RELATIVE = qualification.VALID_RELATIVE_PATH
SOURCE_SPECS = qualification.SOURCE_SPECS
EXPECTED_LINE_COUNTS = qualification.EXPECTED_LINE_COUNTS
FAMILY_ORDER = ("CAUSAL", "SUBEVENT", "TEMPORAL")
FAMILY_PRIORITY = FAMILY_ORDER
TEMPORAL_LABELS = qualification.TEMPORAL_LABELS
CAUSAL_LABELS = qualification.CAUSAL_LABELS

BLOCK_SPECS = (
    ("G_form", "train", 32, True),
    ("A_form", "train", 16, True),
    ("F_search", "train", 12, False),
    ("A_hold", "valid", 10, True),
    ("M_search", "valid", 10, True),
)
FORMAL_ROOT_RELATIVE = Path("artifacts/maven_ere_g8_e1_formal_v1")
ACQUISITION_ROOT_NAME = "acquisition"


class MavenEreAcquisitionError(RuntimeError):
    """Fail-closed acquisition or private-pack error."""


class AssignmentShortfall(MavenEreAcquisitionError):
    """The frozen exact assignment cannot satisfy every target."""


class OneShotRefusal(MavenEreAcquisitionError):
    """The formal acquisition root is not pristine."""


class FormalProvenanceError(MavenEreAcquisitionError):
    """A frozen public implementation or source binding drifted."""


@dataclass(frozen=True)
class _EventRow:
    source_id: str = field(repr=False)
    event_type: str
    mentions: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class _Document:
    split: str
    split_index: int
    source_id: str = field(repr=False)
    title: str = field(repr=False)
    sentences: tuple[str, ...] = field(repr=False)
    events: tuple[_EventRow, ...] = field(repr=False)
    generic_relations: tuple[tuple[int, int], ...] = field(repr=False)
    family_pairs: Mapping[str, tuple[tuple[int, int], ...]] = field(repr=False)

    @property
    def collision_keys(self) -> tuple[str, str, str]:
        return (
            hashlib.sha256(self.source_id.encode("utf-8")).hexdigest(),
            hashlib.sha256(_normalize(self.title).encode("utf-8")).hexdigest(),
            stable_hash(self.sentences),
        )


@dataclass(frozen=True)
class _Candidate:
    document_index: int
    family: str
    head_event: int
    tail_event: int
    selection_digest: str
    item_id: str
    block: str

    @property
    def cost(self) -> int:
        return int(self.selection_digest, 16)

    @property
    def tie_break(self) -> tuple[object, ...]:
        return (
            self.family,
            self.head_event,
            self.tail_event,
            self.document_index,
            self.block,
        )


@dataclass(frozen=True)
class _EdgeChoice:
    cost: int
    tie_break: tuple[object, ...]
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
class AcquisitionMaterials:
    view_packs: Mapping[str, Mapping[str, Any]]
    label_packs: Mapping[str, Mapping[str, Any]]
    selected_item_count: int
    collision_component_count: int
    eligible_candidate_count: int
    min_cost_sum: int


def canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MavenEreAcquisitionError("value is not canonical JSON") from exc


def _canonical_body(value: object) -> bytes:
    return canonical_json(value)[:-1]


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_body(value)).hexdigest()


def _self_hashed(body: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    output = dict(body)
    output[field_name] = stable_hash(output)
    return output


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _hmac(secret: bytes, domain: str, value: object) -> str:
    if not isinstance(secret, bytes) or len(secret) != 32:
        raise MavenEreAcquisitionError("selection secret must be exactly 32 bytes")
    message = domain.encode("ascii") + b"\x00" + _canonical_body(value)
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def _secret_commitment(secret: bytes) -> str:
    return _hmac(secret, "maven_ere_g8_e1_secret_commitment_v1", "committed")


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def _collision_components(documents: Sequence[_Document]) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    union = _UnionFind(len(documents))
    seen: list[dict[str, int]] = [{}, {}, {}]
    for index, document in enumerate(documents):
        for table, key in zip(seen, document.collision_keys, strict=True):
            prior = table.setdefault(key, index)
            union.union(index, prior)
    groups: dict[int, list[int]] = {}
    for index in range(len(documents)):
        groups.setdefault(union.find(index), []).append(index)
    components = tuple(tuple(row) for row in sorted(groups.values(), key=min))
    mapping = [-1] * len(documents)
    for component, rows in enumerate(components):
        for index in rows:
            mapping[index] = component
    return components, tuple(mapping)


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
        right,
        len(graph[right]),
        capacity,
        cost,
        capacity,
        payload,
        target,
    )
    reverse = _FlowEdge(left, len(graph[left]), 0, -cost, 0)
    graph[left].append(forward)
    graph[right].append(reverse)


def deterministic_min_cost_assignment(
    component_choices: Mapping[int, Mapping[tuple[str, str], _EdgeChoice]],
    demands: Mapping[tuple[str, str], int],
) -> _FlowSolution:
    component_ids = tuple(sorted(component_choices))
    targets = tuple(demands)
    if any(type(value) is not int or value < 0 for value in demands.values()):
        raise MavenEreAcquisitionError("invalid target demand")
    source = 0
    component_offset = 1
    target_offset = component_offset + len(component_ids)
    sink = target_offset + len(targets)
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    component_nodes = {
        component: component_offset + index
        for index, component in enumerate(component_ids)
    }
    target_nodes = {
        target: target_offset + index for index, target in enumerate(targets)
    }
    for component in component_ids:
        node = component_nodes[component]
        _add_edge(graph, source, node, 1, 0)
        for target in targets:
            choice = component_choices[component].get(target)
            if choice is not None:
                _add_edge(
                    graph,
                    node,
                    target_nodes[target],
                    1,
                    choice.cost,
                    payload=choice.payload,
                    target=target,
                )
    for target, demand in demands.items():
        _add_edge(graph, target_nodes[target], sink, demand, 0)
    required = sum(demands.values())
    flow = 0
    total_cost = 0
    potential = [0] * len(graph)
    while flow < required:
        distance: list[int | None] = [None] * len(graph)
        previous_node = [-1] * len(graph)
        previous_edge = [-1] * len(graph)
        distance[source] = 0
        queue: list[tuple[int, int]] = [(0, source)]
        while queue:
            current, node = heapq.heappop(queue)
            if distance[node] != current:
                continue
            for edge_index, edge in enumerate(graph[node]):
                if edge.capacity <= 0:
                    continue
                candidate = current + edge.cost + potential[node] - potential[edge.to]
                if distance[edge.to] is None or candidate < distance[edge.to]:
                    distance[edge.to] = candidate
                    previous_node[edge.to] = node
                    previous_edge[edge.to] = edge_index
                    heapq.heappush(queue, (candidate, edge.to))
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
    selected: dict[tuple[str, str], list[Any]] = {target: [] for target in targets}
    for component in component_ids:
        for edge in graph[component_nodes[component]]:
            if (
                edge.payload is not None
                and edge.target is not None
                and edge.original_capacity == 1
                and edge.capacity == 0
            ):
                selected[edge.target].append(edge.payload)
    return _FlowSolution(
        flow,
        required,
        total_cost,
        {target: tuple(rows) for target, rows in selected.items()},
    )


def _parse_document(
    value: Mapping[str, Any], *, split: str, split_index: int
) -> _Document:
    # Reuse the exact source-qualification parser as the minimum reader proof.
    qualification._parse_document(value, split=split, line_index=split_index)
    source_id = str(value["id"])
    title = str(value["title"])
    sentences = tuple(" ".join(str(token) for token in row) for row in value["tokens"])
    source_to_local: dict[str, int] = {}
    events: list[_EventRow] = []
    for event_index, raw_event in enumerate(value["events"]):
        event_id = str(raw_event["id"])
        source_to_local[event_id] = event_index
        mentions = tuple(
            (
                str(raw_mention.get("trigger_word", raw_mention.get("mention"))),
                int(raw_mention["sent_id"]),
            )
            for raw_mention in raw_event["mention"]
        )
        events.append(_EventRow(event_id, str(raw_event["type"]), mentions))

    family_rows: dict[str, list[tuple[str, str, str]]] = {
        "TEMPORAL": [],
        "CAUSAL": [],
        "SUBEVENT": [],
    }
    for label in TEMPORAL_LABELS:
        for left, right in value["temporal_relations"].get(label, ()):
            if left in source_to_local and right in source_to_local:
                family_rows["TEMPORAL"].append((label, str(left), str(right)))
    for label in CAUSAL_LABELS:
        for left, right in value["causal_relations"].get(label, ()):
            if left in source_to_local and right in source_to_local:
                family_rows["CAUSAL"].append((label, str(left), str(right)))
    for left, right in value["subevent_relations"]:
        if left in source_to_local and right in source_to_local:
            family_rows["SUBEVENT"].append(("SUBEVENT", str(left), str(right)))

    memberships: dict[tuple[str, str], dict[str, set[str]]] = {}
    generic: set[tuple[int, int]] = set()
    for family, rows in family_rows.items():
        for fine, left, right in rows:
            memberships.setdefault((left, right), {}).setdefault(family, set()).add(fine)
            local_left, local_right = source_to_local[left], source_to_local[right]
            generic.add(tuple(sorted((local_left, local_right))))
    selected: dict[str, list[tuple[int, int]]] = {family: [] for family in FAMILY_ORDER}
    for (left, right), by_family in memberships.items():
        primary = next(family for family in FAMILY_PRIORITY if family in by_family)
        if len(by_family[primary]) == 1:
            selected[primary].append((source_to_local[left], source_to_local[right]))
    return _Document(
        split=split,
        split_index=split_index,
        source_id=source_id,
        title=title,
        sentences=sentences,
        events=tuple(events),
        generic_relations=tuple(sorted(generic)),
        family_pairs={family: tuple(sorted(set(rows))) for family, rows in selected.items()},
    )


def parse_released_members(
    train_raw: bytes,
    valid_raw: bytes,
    *,
    expected_line_counts: Mapping[str, int] = EXPECTED_LINE_COUNTS,
) -> tuple[_Document, ...]:
    documents: list[_Document] = []
    for split, raw in (("train", train_raw), ("valid", valid_raw)):
        lines = raw.splitlines()
        if len(lines) != expected_line_counts[split]:
            raise MavenEreAcquisitionError("released member line count drifted")
        for index, line in enumerate(lines):
            try:
                value = qualification._strict_json_line(line)
                documents.append(_parse_document(value, split=split, split_index=index))
            except Exception as exc:
                raise MavenEreAcquisitionError(
                    "source qualification reader equivalence failed"
                ) from exc
    return tuple(documents)


def _candidate_for_target(
    *,
    secret: bytes,
    document: _Document,
    document_index: int,
    block: str,
    family: str,
) -> _Candidate | None:
    rows = document.family_pairs[family]
    if not rows:
        return None
    ranked: list[tuple[str, tuple[int, int]]] = []
    for head, tail in rows:
        digest = _hmac(
            secret,
            "maven_ere_g8_e1_candidate_rank_v1",
            {
                "block": block,
                "document_id": document.source_id,
                "family": family,
                "head_source_event_id": document.events[head].source_id,
                "split": document.split,
                "tail_source_event_id": document.events[tail].source_id,
            },
        )
        ranked.append((digest, (head, tail)))
    digest, (head, tail) = min(ranked, key=lambda row: (row[0], row[1]))
    item_id = _hmac(
        secret,
        "maven_ere_g8_e1_private_item_id_v1",
        {
            "block": block,
            "document_id": document.source_id,
            "head_source_event_id": document.events[head].source_id,
            "tail_source_event_id": document.events[tail].source_id,
        },
    )
    return _Candidate(document_index, family, head, tail, digest, item_id, block)


def _candidate_view(candidate: _Candidate, document: _Document) -> dict[str, Any]:
    head, tail = candidate.head_event, candidate.tail_event
    query_pair = tuple(sorted((head, tail)))
    generic_relations = [
        [left, right]
        for left, right in document.generic_relations
        if (left, right) != query_pair
    ]
    events = [
        {
            "event_id": event_id,
            "event_type": event.event_type,
            "mentions": [
                {"sentence_ordinal": sentence, "surface": surface}
                for surface, sentence in event.mentions
            ],
        }
        for event_id, event in enumerate(document.events)
    ]
    head_aliases = _canonical_aliases(surface for surface, _ in document.events[head].mentions)
    tail_aliases = _canonical_aliases(surface for surface, _ in document.events[tail].mentions)
    query = (
        f"EVENT_A aliases: {' | '.join(head_aliases)}\n"
        f"EVENT_A type: {document.events[head].event_type}\n"
        f"EVENT_B aliases: {' | '.join(tail_aliases)}\n"
        f"EVENT_B type: {document.events[tail].event_type}\n"
        "Question: What is the relationship between event A and event B?"
    )
    return {
        "common_query": query,
        "events": events,
        "generic_relations": generic_relations,
        "head_event": head,
        "item_id": candidate.item_id,
        "sentences": list(document.sentences),
        "tail_event": tail,
    }


def _canonical_aliases(values: Any) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw)
        key = _normalize(value)
        if key not in seen:
            result.append(value)
            seen.add(key)
    if not result:
        raise MavenEreAcquisitionError("selected endpoint has no alias")
    return tuple(result)


def _pack(schema: str, block: str, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    body = {
        "block": block,
        "item_count": len(items),
        "items": list(items),
        "schema": schema,
        "version": VERSION,
    }
    return _self_hashed(body, "pack_sha256")


def build_acquisition_materials(
    *,
    train_raw: bytes,
    valid_raw: bytes,
    secret: bytes,
    block_specs: Sequence[tuple[str, str, int, bool]] = BLOCK_SPECS,
    expected_line_counts: Mapping[str, int] = EXPECTED_LINE_COUNTS,
) -> AcquisitionMaterials:
    documents = parse_released_members(
        train_raw,
        valid_raw,
        expected_line_counts=expected_line_counts,
    )
    components, document_to_component = _collision_components(documents)
    demands: dict[tuple[str, str], int] = {}
    best: dict[int, dict[tuple[str, str], _Candidate]] = {
        component: {} for component in range(len(components))
    }
    eligible_count = 0
    for block, split, quota, _has_labels in block_specs:
        for family in FAMILY_ORDER:
            demands[(block, family)] = quota
        for document_index, document in enumerate(documents):
            if document.split != split:
                continue
            component = document_to_component[document_index]
            for family in FAMILY_ORDER:
                candidate = _candidate_for_target(
                    secret=secret,
                    document=document,
                    document_index=document_index,
                    block=block,
                    family=family,
                )
                if candidate is None:
                    continue
                eligible_count += len(document.family_pairs[family])
                target = (block, family)
                incumbent = best[component].get(target)
                if incumbent is None or (
                    candidate.selection_digest,
                    candidate.tie_break,
                ) < (incumbent.selection_digest, incumbent.tie_break):
                    best[component][target] = candidate
    choices = {
        component: {
            target: _EdgeChoice(candidate.cost, candidate.tie_break, candidate)
            for target, candidate in rows.items()
        }
        for component, rows in best.items()
        if rows
    }
    solution = deterministic_min_cost_assignment(choices, demands)
    if solution.assigned_count != solution.required_count:
        raise AssignmentShortfall("frozen document-disjoint assignment shortfall")
    view_packs: dict[str, Mapping[str, Any]] = {}
    label_packs: dict[str, Mapping[str, Any]] = {}
    all_item_ids: set[str] = set()
    for block, _split, quota, has_labels in block_specs:
        selected: list[_Candidate] = []
        for family in FAMILY_ORDER:
            rows = list(solution.selected[(block, family)])
            if len(rows) != quota:
                raise AssignmentShortfall("target quota was not exactly satisfied")
            selected.extend(rows)
        selected.sort(key=lambda row: row.item_id)
        if any(row.item_id in all_item_ids for row in selected):
            raise MavenEreAcquisitionError("private item ID collision")
        all_item_ids.update(row.item_id for row in selected)
        views = [_candidate_view(row, documents[row.document_index]) for row in selected]
        view_packs[block] = _pack(
            "maven_ere_g8_e1_action_view_pack_v1", block, views
        )
        if has_labels:
            labels = [
                {"family": row.family, "item_id": row.item_id} for row in selected
            ]
            label_packs[block] = _pack(
                "maven_ere_g8_e1_family_label_pack_v1", block, labels
            )
    if "F_search" in label_packs:
        raise MavenEreAcquisitionError("F_search label pack is forbidden")
    return AcquisitionMaterials(
        view_packs=view_packs,
        label_packs=label_packs,
        selected_item_count=len(all_item_ids),
        collision_component_count=len(components),
        eligible_candidate_count=eligible_count,
        min_cost_sum=solution.total_cost,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_bound_source(path: Path, *, size: int, digest: str) -> bytes:
    absolute = path.absolute()
    if absolute.is_symlink() or not absolute.is_file():
        raise FormalProvenanceError("bound source is unavailable")
    metadata = absolute.stat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size != size
        or metadata.st_mode & 0o077
    ):
        raise FormalProvenanceError("bound source metadata drifted")
    raw = absolute.read_bytes()
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        raise FormalProvenanceError("bound source bytes drifted")
    return raw


def _exclusive_write(path: Path, raw: bytes, mode: int = 0o600) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write(path: Path, raw: bytes, mode: int = 0o600) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    _exclusive_write(temporary, raw, mode)
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _public_pack_binding(root: Path, path: Path, pack: Mapping[str, Any]) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "file": path.relative_to(root).as_posix(),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "item_count": pack["item_count"],
        "pack_sha256": pack["pack_sha256"],
        "size_bytes": len(raw),
    }


def _validate_public_design(project: Path) -> None:
    path = project / DESIGN_RELATIVE
    if not path.is_file() or _sha256_file(path) != DESIGN_FILE_SHA256:
        raise FormalProvenanceError("formal design file drifted")
    value = json.loads(path.read_text(encoding="utf-8"))
    body = dict(value)
    declared = body.pop("study_design_sha256", None)
    if declared != DESIGN_SELF_SHA256 or stable_hash(body) != DESIGN_SELF_SHA256:
        raise FormalProvenanceError("formal design self hash drifted")


def run_formal_acquisition(project_root: str | Path) -> dict[str, Any]:
    project = Path(project_root).resolve(strict=True)
    if project.name != "reconstruction_v2":
        raise FormalProvenanceError("project root must be reconstruction_v2")
    _validate_public_design(project)
    root = project / FORMAL_ROOT_RELATIVE / ACQUISITION_ROOT_NAME
    if os.path.lexists(root):
        raise OneShotRefusal("formal acquisition root already exists")
    root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.mkdir(mode=0o700)
    marker = _self_hashed(
        {
            "design_sha256": DESIGN_SELF_SHA256,
            "schema": "maven_ere_g8_e1_acquisition_authorization_consumed_v1",
            "status": "consumed_before_source_open",
            "version": VERSION,
        },
        "marker_sha256",
    )
    _exclusive_write(root / "authorization.consumed.json", canonical_json(marker))
    try:
        secret = os.urandom(32)
        _exclusive_write(root / "selection.secret", secret)
        train_path, train_size, train_sha = SOURCE_SPECS["train"]
        valid_path, valid_size, valid_sha = SOURCE_SPECS["valid"]
        train_raw = _read_bound_source(
            project / train_path, size=train_size, digest=train_sha
        )
        valid_raw = _read_bound_source(
            project / valid_path, size=valid_size, digest=valid_sha
        )
        materials = build_acquisition_materials(
            train_raw=train_raw,
            valid_raw=valid_raw,
            secret=secret,
        )
        private = root / "private_packs"
        private.mkdir(mode=0o700)
        view_bindings: dict[str, Any] = {}
        label_bindings: dict[str, Any] = {}
        for block, pack in materials.view_packs.items():
            path = private / f"{block}.view.json"
            _atomic_write(path, canonical_json(pack))
            view_bindings[block] = _public_pack_binding(root, path, pack)
        for block, pack in materials.label_packs.items():
            path = private / f"{block}.labels.json"
            _atomic_write(path, canonical_json(pack))
            label_bindings[block] = _public_pack_binding(root, path, pack)
        receipt = _self_hashed(
            {
                "block_item_counts": {
                    block: pack["item_count"]
                    for block, pack in materials.view_packs.items()
                },
                "claim_boundary": {
                    "action_model_retrieval_classifier_or_score_run": False,
                    "hidden_TEST_opened": False,
                    "online_or_external_evaluator_calls": 0,
                    "source_train_open_count": 1,
                    "source_valid_open_count": 1,
                },
                "collision_component_count": materials.collision_component_count,
                "design_sha256": DESIGN_SELF_SHA256,
                "eligible_candidate_occurrence_count": materials.eligible_candidate_count,
                "label_pack_bindings": label_bindings,
                "minimum_cost_sum_sha256": hashlib.sha256(
                    str(materials.min_cost_sum).encode("ascii")
                ).hexdigest(),
                "schema": SCHEMA,
                "secret_commitment": _secret_commitment(secret),
                "selected_item_count": materials.selected_item_count,
                "status": "passed_one_shot_private_document_disjoint_acquisition",
                "version": VERSION,
                "view_pack_bindings": view_bindings,
            },
            "acquisition_sha256",
        )
        _atomic_write(root / "acquisition.receipt.json", canonical_json(receipt))
        return receipt
    except BaseException as exc:
        failure = _self_hashed(
            {
                "category": type(exc).__name__,
                "design_sha256": DESIGN_SELF_SHA256,
                "message_sha256": hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
                "schema": "maven_ere_g8_e1_acquisition_failure_v1",
                "status": "terminal_no_retry",
                "version": VERSION,
            },
            "failure_sha256",
        )
        try:
            _atomic_write(root / "acquisition.failure.json", canonical_json(failure))
        except BaseException:
            pass
        raise


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise FormalProvenanceError("git provenance command failed")
    return completed.stdout.strip()


__all__ = [
    "AcquisitionMaterials",
    "AssignmentShortfall",
    "BLOCK_SPECS",
    "DESIGN_COMMIT",
    "DESIGN_FILE_SHA256",
    "DESIGN_SELF_SHA256",
    "FAMILY_ORDER",
    "FormalProvenanceError",
    "MavenEreAcquisitionError",
    "OneShotRefusal",
    "build_acquisition_materials",
    "canonical_json",
    "deterministic_min_cost_assignment",
    "parse_released_members",
    "run_formal_acquisition",
    "stable_hash",
]
