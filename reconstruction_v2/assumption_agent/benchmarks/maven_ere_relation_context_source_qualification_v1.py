"""Aggregate-only one-shot MAVEN-ERE source qualification.

The formal path reads only the two released labelled members bound before this
implementation existed.  It emits source-wide aggregates and a simultaneous
document-component capacity result; it never selects an item or creates a
selection secret.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping, Sequence
import unicodedata


VERSION = "v1"
SCHEMA = "maven_ere_relation_context_source_qualification_v1"
ATTEMPT_SCHEMA = "maven_ere_relation_context_source_qualification_attempt_v1"
IMPLEMENTATION_FREEZE_SCHEMA = (
    "maven_ere_relation_context_source_qualification_implementation_freeze_v1"
)

FAMILY_ORDER = ("TEMPORAL", "CAUSAL", "SUBEVENT")
TEMPORAL_LABELS = (
    "BEFORE",
    "OVERLAP",
    "CONTAINS",
    "SIMULTANEOUS",
    "ENDS-ON",
    "BEGINS-ON",
)
CAUSAL_LABELS = ("CAUSE", "PRECONDITION")
EXPECTED_LINE_COUNTS = {"train": 2913, "valid": 710}
REQUIRED_PER_SPLIT_FAMILY = {
    ("train", family): 60 for family in FAMILY_ORDER
} | {("valid", family): 20 for family in FAMILY_ORDER}

DESIGN_RELATIVE_PATH = Path(
    "manifests/maven_ere_relation_context_source_qualification_design_v1.json"
)
DESIGN_FILE_SHA256 = (
    "cd8c0f1d42763debe6eda7e520983f9c83fc04d5de3f14856f8db821c8fd6e2f"
)
DESIGN_SELF_SHA256 = (
    "abdf26e29a5a08f60d65bba27355bcc6993aaec6d0cd47c01fcfae19ae039def"
)
MEMBER_BINDING_RELATIVE_PATH = Path(
    "manifests/maven_ere_event_context_member_binding_v1.json"
)
MEMBER_BINDING_FILE_SHA256 = (
    "1d52b5835187882764fa2587e871e242955c1af9c831e096a4cb33b3e1c3326c"
)
MEMBER_BINDING_SELF_SHA256 = (
    "1bf7e0ff04ec0373effe93a3c194368006de14cdf68a207c5a8ed6fcc8b2e869"
)
AMENDMENT_RELATIVE_PATH = Path(
    "manifests/maven_ere_relation_context_pre_row_family_priority_amendment_v1.json"
)
AMENDMENT_FILE_SHA256 = (
    "ab1f8752fb0b097240e0878154a0141d73595e57005ef71bce2ec71611806d57"
)
AMENDMENT_SELF_SHA256 = (
    "ff4acc9982146f208157b7209e0606b6e5cbb6fa30cc1d8a7cba02fe76110006"
)
TRAIN_RELATIVE_PATH = Path(
    "artifacts/maven_ere_official_source_v1/released_labelled_v1/train.jsonl"
)
VALID_RELATIVE_PATH = Path(
    "artifacts/maven_ere_official_source_v1/released_labelled_v1/valid.jsonl"
)
SOURCE_SPECS = {
    "train": (
        TRAIN_RELATIVE_PATH,
        101_215_305,
        "6a5519fe7c30448690adb13d49217c50d474fc57480eae10aecb29df7eb638b7",
    ),
    "valid": (
        VALID_RELATIVE_PATH,
        24_406_071,
        "6faea0e4e16b4a2d5d9631e09ef6e1c6bac6e3f912490bfc48eeaceaf98c6153",
    ),
}
QUALIFIER_RELATIVE_PATH = Path(
    "assumption_agent/benchmarks/"
    "maven_ere_relation_context_source_qualification_v1.py"
)
QUALIFIER_TEST_RELATIVE_PATH = Path(
    "tests/test_maven_ere_relation_context_source_qualification_v1.py"
)
IMPLEMENTATION_FREEZE_RELATIVE_PATH = Path(
    "manifests/"
    "maven_ere_relation_context_source_qualification_implementation_freeze_v1.json"
)
FORMAL_OUTPUT_RELATIVE_PATH = Path(
    "artifacts/maven_ere_source_qualification_v1"
)


class MavenEreSourceQualificationError(RuntimeError):
    """Fail-closed public source qualification error."""


class FormalProvenanceError(MavenEreSourceQualificationError):
    """The pre-row public implementation binding is not exact."""


class OneShotRefusal(MavenEreSourceQualificationError):
    """The formal output root is not pristine and exact."""


class _DocumentInvalid(ValueError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class _DocumentRecord:
    split: str
    line_index: int
    document_id_hash: str
    normalized_title_hash: str
    rendered_document_hash: str
    sentence_count: int
    event_count: int
    timex_count: int
    raw_relation_counts: Mapping[str, int]
    eligible_candidate_counts: Mapping[str, int]


@dataclass(frozen=True)
class _SplitAudit:
    split: str
    line_count: int
    valid_documents: tuple[_DocumentRecord, ...]
    invalid_reason_counts: Mapping[str, int]


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
        raise MavenEreSourceQualificationError("non-canonical public value") from exc


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _stable_hash(value: Any) -> str:
    return _sha256(_canonical_json(value))


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


def _strict_json(raw: bytes, *, label: str) -> Any:
    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise MavenEreSourceQualificationError(f"invalid strict JSON: {label}") from exc


def _strict_json_line(raw: bytes) -> Mapping[str, Any]:
    if not raw:
        raise _DocumentInvalid("json_line")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise _DocumentInvalid("json_line") from exc
    if not isinstance(value, Mapping):
        raise _DocumentInvalid("document_root")
    return value


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _text(value: Any, *, reason: str, allow_empty_token: bool = False) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise _DocumentInvalid(reason)
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise _DocumentInvalid(reason) from exc
    if not allow_empty_token and not value.strip():
        raise _DocumentInvalid(reason)
    return value


def _integer(value: Any, *, reason: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _DocumentInvalid(reason)
    return value


def _sequence(value: Any, *, reason: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _DocumentInvalid(reason)
    return value


def _mapping(value: Any, *, reason: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _DocumentInvalid(reason)
    return value


def _mention(
    raw: Any,
    *,
    sentence_count: int,
    token_counts: Sequence[int],
    reason_prefix: str,
) -> tuple[str, int]:
    value = _mapping(raw, reason=reason_prefix)
    _text(value.get("id"), reason=reason_prefix + "_id")
    surface_key = "trigger_word" if "trigger_word" in value else "mention"
    surface = _text(value.get(surface_key), reason=reason_prefix + "_surface")
    sentence = _integer(value.get("sent_id"), reason=reason_prefix + "_sentence")
    if not 0 <= sentence < sentence_count:
        raise _DocumentInvalid(reason_prefix + "_sentence")
    offset = _sequence(value.get("offset"), reason=reason_prefix + "_offset")
    if len(offset) != 2:
        raise _DocumentInvalid(reason_prefix + "_offset")
    start = _integer(offset[0], reason=reason_prefix + "_offset")
    end = _integer(offset[1], reason=reason_prefix + "_offset")
    if not 0 <= start < end <= token_counts[sentence]:
        raise _DocumentInvalid(reason_prefix + "_offset")
    return surface, sentence


def _relation_pairs(
    raw: Any,
    *,
    allowed_labels: Sequence[str],
    known_ids: frozenset[str],
    reason: str,
) -> tuple[tuple[str, str, str], ...]:
    value = _mapping(raw, reason=reason)
    if any(key not in allowed_labels for key in value):
        raise _DocumentInvalid(reason + "_label")
    rows: list[tuple[str, str, str]] = []
    for label in allowed_labels:
        for raw_pair in _sequence(value.get(label, ()), reason=reason + "_pairs"):
            pair = _sequence(raw_pair, reason=reason + "_pair")
            if len(pair) != 2:
                raise _DocumentInvalid(reason + "_pair")
            left = _text(pair[0], reason=reason + "_endpoint")
            right = _text(pair[1], reason=reason + "_endpoint")
            if left == right or left not in known_ids or right not in known_ids:
                raise _DocumentInvalid(reason + "_endpoint")
            rows.append((label, left, right))
    return tuple(rows)


def _parse_document(
    value: Mapping[str, Any],
    *,
    split: str,
    line_index: int,
) -> _DocumentRecord:
    required = {
        "id",
        "title",
        "tokens",
        "events",
        "TIMEX",
        "temporal_relations",
        "causal_relations",
        "subevent_relations",
    }
    if not required.issubset(value):
        raise _DocumentInvalid("document_required_fields")
    document_id = _text(value["id"], reason="document_id")
    title = _text(value["title"], reason="title")
    raw_sentences = _sequence(value["tokens"], reason="tokens")
    rendered: list[str] = []
    token_counts: list[int] = []
    for raw_sentence in raw_sentences:
        sentence = _sequence(raw_sentence, reason="sentence")
        tokens = [
            _text(token, reason="sentence_token", allow_empty_token=True)
            for token in sentence
        ]
        rendered_sentence = " ".join(tokens)
        if not rendered_sentence.strip():
            raise _DocumentInvalid("empty_rendered_sentence")
        rendered.append(rendered_sentence)
        token_counts.append(len(tokens))
    if len(rendered) < 6:
        raise _DocumentInvalid("sentence_count")

    raw_events = _sequence(value["events"], reason="events")
    event_mentions: dict[str, tuple[tuple[str, int], ...]] = {}
    for raw_event in raw_events:
        event = _mapping(raw_event, reason="event")
        event_id = _text(event.get("id"), reason="event_id")
        if event_id in event_mentions or event_id.startswith("TIME"):
            raise _DocumentInvalid("event_id")
        _text(event.get("type"), reason="event_type")
        mentions = tuple(
            _mention(
                mention,
                sentence_count=len(rendered),
                token_counts=token_counts,
                reason_prefix="event_mention",
            )
            for mention in _sequence(event.get("mention"), reason="event_mentions")
        )
        if not mentions:
            raise _DocumentInvalid("event_mentions")
        event_mentions[event_id] = mentions

    raw_timex = _sequence(value["TIMEX"], reason="TIMEX")
    timex_ids: set[str] = set()
    for raw_time in raw_timex:
        time = _mapping(raw_time, reason="TIMEX_item")
        time_id = _text(time.get("id"), reason="TIMEX_id")
        if time_id in event_mentions or time_id in timex_ids:
            raise _DocumentInvalid("TIMEX_id")
        _mention(
            time,
            sentence_count=len(rendered),
            token_counts=token_counts,
            reason_prefix="TIMEX_mention",
        )
        timex_ids.add(time_id)

    all_ids = frozenset((*event_mentions, *timex_ids))
    event_ids = frozenset(event_mentions)
    temporal = _relation_pairs(
        value["temporal_relations"],
        allowed_labels=TEMPORAL_LABELS,
        known_ids=all_ids,
        reason="temporal_relations",
    )
    causal = _relation_pairs(
        value["causal_relations"],
        allowed_labels=CAUSAL_LABELS,
        known_ids=all_ids,
        reason="causal_relations",
    )
    subevent_rows: list[tuple[str, str, str]] = []
    for raw_pair in _sequence(value["subevent_relations"], reason="subevent_relations"):
        pair = _sequence(raw_pair, reason="subevent_pair")
        if len(pair) != 2:
            raise _DocumentInvalid("subevent_pair")
        left = _text(pair[0], reason="subevent_endpoint")
        right = _text(pair[1], reason="subevent_endpoint")
        if left == right or left not in all_ids or right not in all_ids:
            raise _DocumentInvalid("subevent_endpoint")
        subevent_rows.append(("SUBEVENT", left, right))

    family_rows = {
        "TEMPORAL": temporal,
        "CAUSAL": causal,
        "SUBEVENT": tuple(subevent_rows),
    }
    raw_counts = {family: len(rows) for family, rows in family_rows.items()}
    pair_family_labels: dict[
        tuple[str, str], dict[str, set[str]]
    ] = {}
    for family, rows in family_rows.items():
        for fine_label, left, right in rows:
            if left not in event_ids or right not in event_ids:
                continue
            pair_family_labels.setdefault((left, right), {}).setdefault(
                family, set()
            ).add(fine_label)
    eligible = {family: 0 for family in FAMILY_ORDER}
    family_priority = ("CAUSAL", "SUBEVENT", "TEMPORAL")
    for memberships in pair_family_labels.values():
        primary = next(
            family for family in family_priority if family in memberships
        )
        if len(memberships[primary]) == 1:
            eligible[primary] += 1

    return _DocumentRecord(
        split=split,
        line_index=line_index,
        document_id_hash=_sha256(document_id.encode("utf-8")),
        normalized_title_hash=_sha256(
            _normalize_text(title).encode("utf-8")
        ),
        rendered_document_hash=_stable_hash(rendered),
        sentence_count=len(rendered),
        event_count=len(event_mentions),
        timex_count=len(timex_ids),
        raw_relation_counts=raw_counts,
        eligible_candidate_counts=eligible,
    )


def _audit_split(raw: bytes, *, split: str) -> _SplitAudit:
    lines = raw.splitlines()
    invalid: Counter[str] = Counter()
    valid: list[_DocumentRecord] = []
    for index, line in enumerate(lines):
        try:
            value = _strict_json_line(line)
            valid.append(_parse_document(value, split=split, line_index=index))
        except _DocumentInvalid as exc:
            invalid[exc.reason] += 1
    return _SplitAudit(
        split=split,
        line_count=len(lines),
        valid_documents=tuple(valid),
        invalid_reason_counts=dict(sorted(invalid.items())),
    )


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


@dataclass
class _FlowEdge:
    to: int
    reverse: int
    capacity: int


def _add_edge(graph: list[list[_FlowEdge]], left: int, right: int, capacity: int) -> None:
    graph[left].append(_FlowEdge(right, len(graph[right]), capacity))
    graph[right].append(_FlowEdge(left, len(graph[left]) - 1, 0))


def _max_flow(
    component_targets: Mapping[int, frozenset[tuple[str, str]]],
    demands: Mapping[tuple[str, str], int],
) -> tuple[int, Mapping[tuple[str, str], int]]:
    components = tuple(sorted(component_targets))
    targets = tuple(demands)
    source = 0
    component_offset = 1
    target_offset = component_offset + len(components)
    sink = target_offset + len(targets)
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    component_nodes = {
        component: component_offset + index
        for index, component in enumerate(components)
    }
    target_nodes = {
        target: target_offset + index for index, target in enumerate(targets)
    }
    target_sink_edges: dict[tuple[str, str], _FlowEdge] = {}
    for component in components:
        _add_edge(graph, source, component_nodes[component], 1)
        for target in targets:
            if target in component_targets[component]:
                _add_edge(graph, component_nodes[component], target_nodes[target], 1)
    for target in targets:
        _add_edge(graph, target_nodes[target], sink, demands[target])
        target_sink_edges[target] = graph[target_nodes[target]][-1]
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
        target: demands[target] - target_sink_edges[target].capacity
        for target in targets
    }
    return flow, assigned


def _capacity_receipt(audits: Sequence[_SplitAudit]) -> dict[str, Any]:
    documents = tuple(
        document for audit in audits for document in audit.valid_documents
    )
    union_find = _UnionFind(len(documents))
    seen_by_kind: list[dict[str, int]] = [{}, {}, {}]
    for index, document in enumerate(documents):
        for seen, value in zip(
            seen_by_kind,
            (
                document.document_id_hash,
                document.normalized_title_hash,
                document.rendered_document_hash,
            ),
            strict=True,
        ):
            previous = seen.setdefault(value, index)
            union_find.union(index, previous)
    groups: dict[int, list[int]] = {}
    for index in range(len(documents)):
        groups.setdefault(union_find.find(index), []).append(index)
    component_targets: dict[int, set[tuple[str, str]]] = {}
    for component, indices in enumerate(sorted(groups.values(), key=min)):
        for index in indices:
            document = documents[index]
            for family in FAMILY_ORDER:
                if document.eligible_candidate_counts[family] > 0:
                    component_targets.setdefault(component, set()).add(
                        (document.split, family)
                    )
    flow, assigned = _max_flow(
        {
            component: frozenset(targets)
            for component, targets in component_targets.items()
        },
        REQUIRED_PER_SPLIT_FAMILY,
    )
    multi = [indices for indices in groups.values() if len(indices) > 1]
    per_target: dict[str, dict[str, Any]] = {"train": {}, "valid": {}}
    for split in ("train", "valid"):
        for family in FAMILY_ORDER:
            target = (split, family)
            assignable = sum(
                target in targets for targets in component_targets.values()
            )
            per_target[split][family] = {
                "assignable_component_count": assignable,
                "required_count": REQUIRED_PER_SPLIT_FAMILY[target],
                "max_flow_assigned_count": assigned[target],
                "requirement_met": assigned[target]
                == REQUIRED_PER_SPLIT_FAMILY[target],
            }
    required = sum(REQUIRED_PER_SPLIT_FAMILY.values())
    return {
        "document_count_after_reader_validity_exclusion": len(documents),
        "collision_component_count": len(groups),
        "multi_document_collision_component_count": len(multi),
        "document_occurrence_count_in_multi_document_components": sum(
            len(indices) for indices in multi
        ),
        "required_global_document_count": required,
        "deterministic_max_flow_assigned_document_count": flow,
        "simultaneous_document_disjoint_capacity_feasible": flow == required,
        "per_split_family": per_target,
        "private_collision_value_or_per_document_hash_emitted_count": 0,
    }


def _split_aggregate(audit: _SplitAudit) -> dict[str, Any]:
    documents = audit.valid_documents
    sentence_histogram = Counter(document.sentence_count for document in documents)
    candidate_counts = {
        family: sum(
            document.eligible_candidate_counts[family] for document in documents
        )
        for family in FAMILY_ORDER
    }
    assignable_documents = {
        family: sum(
            document.eligible_candidate_counts[family] > 0 for document in documents
        )
        for family in FAMILY_ORDER
    }
    raw_relation_counts = {
        family: sum(document.raw_relation_counts[family] for document in documents)
        for family in FAMILY_ORDER
    }
    return {
        "physical_line_count": audit.line_count,
        "expected_physical_line_count": EXPECTED_LINE_COUNTS[audit.split],
        "line_count_matches_frozen_source": audit.line_count
        == EXPECTED_LINE_COUNTS[audit.split],
        "reader_valid_document_count": len(documents),
        "reader_invalid_document_count": sum(audit.invalid_reason_counts.values()),
        "invalid_document_reason_counts": dict(audit.invalid_reason_counts),
        "sentence_count_histogram": {
            str(key): sentence_histogram[key] for key in sorted(sentence_histogram)
        },
        "event_count_total": sum(document.event_count for document in documents),
        "TIMEX_count_total": sum(document.timex_count for document in documents),
        "raw_relation_instance_counts": raw_relation_counts,
        "eligible_unique_family_pair_candidate_counts": candidate_counts,
        "assignable_document_counts_before_collision_components": (
            assignable_documents
        ),
    }


def qualify_decoded_jsonl(
    train_raw: bytes,
    valid_raw: bytes,
    *,
    formal_identity_enforced: bool,
    source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    audits = (
        _audit_split(train_raw, split="train"),
        _audit_split(valid_raw, split="valid"),
    )
    capacity = _capacity_receipt(audits)
    line_count_drift = sum(
        audit.line_count != EXPECTED_LINE_COUNTS[audit.split] for audit in audits
    )
    capacity_shortfall = int(
        not capacity["simultaneous_document_disjoint_capacity_feasible"]
    )
    terminal = line_count_drift + capacity_shortfall
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": (
            "passed_source_qualification_no_selection"
            if terminal == 0
            else "terminal_source_incompatible_no_selection"
        ),
        "formal_identity_enforced": formal_identity_enforced,
        "source_binding": dict(source_binding),
        "split_aggregates": {
            audit.split: _split_aggregate(audit) for audit in audits
        },
        "simultaneous_document_assignment_capacity": capacity,
        "terminal_reason_counts": {
            "split_line_count_drift_count": line_count_drift,
            "simultaneous_assignment_shortfall_count": capacity_shortfall,
        },
        "claim_boundary": {
            "qualification_only_no_efficacy_claim": True,
            "selection_secret_generated_or_opened": False,
            "cohort_selected_or_materialized": False,
            "retrieval_action_evaluator_classifier_or_score_run": False,
            "online_or_external_evaluation_used": False,
            "hidden_TEST_member_opened": False,
            "document_item_title_alias_trigger_relation_pair_ordinal_or_per_document_hash_emitted": False,
        },
    }
    return _self_hashed(body, "qualification_sha256")


def _read_public_file(path: Path, *, label: str) -> bytes:
    try:
        before = path.lstat()
        raw = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise FormalProvenanceError(f"missing public binding: {label}") from exc
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or identity(before) != identity(after)
        or len(raw) != after.st_size
    ):
        raise FormalProvenanceError(f"unstable public binding: {label}")
    return raw


def _git(repository_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(repository_root), *arguments),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise FormalProvenanceError("git provenance check failed")
    return completed.stdout


def _validate_self_hashed_manifest(
    raw: bytes,
    *,
    self_field: str,
    expected_self: str,
    label: str,
) -> Mapping[str, Any]:
    value = _strict_json(raw, label=label)
    if not isinstance(value, Mapping):
        raise FormalProvenanceError(f"invalid manifest root: {label}")
    body = dict(value)
    declared = body.pop(self_field, None)
    if declared != expected_self or _sha256(_canonical_json(body)) != expected_self:
        raise FormalProvenanceError(f"manifest self hash drifted: {label}")
    return value


def validate_formal_provenance(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    repository_root = Path(
        _git(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    ).resolve(strict=True)
    if (repository_root / "reconstruction_v2").resolve() != root:
        raise FormalProvenanceError("project root is not reconstruction_v2")
    design_raw = _read_public_file(root / DESIGN_RELATIVE_PATH, label="design")
    member_raw = _read_public_file(
        root / MEMBER_BINDING_RELATIVE_PATH, label="member binding"
    )
    amendment_raw = _read_public_file(
        root / AMENDMENT_RELATIVE_PATH, label="family priority amendment"
    )
    if _sha256(design_raw) != DESIGN_FILE_SHA256:
        raise FormalProvenanceError("design file drifted")
    if _sha256(member_raw) != MEMBER_BINDING_FILE_SHA256:
        raise FormalProvenanceError("member binding file drifted")
    if _sha256(amendment_raw) != AMENDMENT_FILE_SHA256:
        raise FormalProvenanceError("family priority amendment file drifted")
    _validate_self_hashed_manifest(
        design_raw,
        self_field="source_qualification_design_sha256",
        expected_self=DESIGN_SELF_SHA256,
        label="design",
    )
    _validate_self_hashed_manifest(
        member_raw,
        self_field="source_member_binding_sha256",
        expected_self=MEMBER_BINDING_SELF_SHA256,
        label="member binding",
    )
    _validate_self_hashed_manifest(
        amendment_raw,
        self_field="source_qualification_amendment_sha256",
        expected_self=AMENDMENT_SELF_SHA256,
        label="family priority amendment",
    )

    freeze_raw = _read_public_file(
        root / IMPLEMENTATION_FREEZE_RELATIVE_PATH,
        label="implementation freeze",
    )
    freeze = _strict_json(freeze_raw, label="implementation freeze")
    if not isinstance(freeze, Mapping):
        raise FormalProvenanceError("implementation freeze root invalid")
    freeze_body = dict(freeze)
    declared = freeze_body.pop("implementation_freeze_sha256", None)
    if (
        not isinstance(declared, str)
        or _sha256(_canonical_json(freeze_body)) != declared
        or freeze.get("schema") != IMPLEMENTATION_FREEZE_SCHEMA
        or freeze.get("status")
        != "frozen_before_first_formal_train_or_valid_JSONL_decode"
        or freeze.get("claim_boundary")
        != {
            "train_or_valid_JSONL_text_decoded": False,
            "formal_source_qualification_run": False,
            "selection_secret_generated": False,
            "private_cohort_selected": False,
            "retrieval_action_evaluator_classifier_or_score_run": False,
            "hidden_TEST_opened": False,
            "online_or_external_evaluation_used": False,
        }
    ):
        raise FormalProvenanceError("implementation freeze drifted")
    implementation_commit = freeze.get("implementation_commit")
    if not isinstance(implementation_commit, str) or len(implementation_commit) != 40:
        raise FormalProvenanceError("implementation commit invalid")
    if any(character not in "0123456789abcdef" for character in implementation_commit):
        raise FormalProvenanceError("implementation commit invalid")
    if _git(repository_root, "cat-file", "-t", implementation_commit).strip() != b"commit":
        raise FormalProvenanceError("implementation commit object invalid")
    required_roles = (
        ("design", DESIGN_RELATIVE_PATH, DESIGN_FILE_SHA256),
        (
            "member_binding",
            MEMBER_BINDING_RELATIVE_PATH,
            MEMBER_BINDING_FILE_SHA256,
        ),
        (
            "family_priority_amendment",
            AMENDMENT_RELATIVE_PATH,
            AMENDMENT_FILE_SHA256,
        ),
        ("qualifier", QUALIFIER_RELATIVE_PATH, None),
        ("qualifier_test", QUALIFIER_TEST_RELATIVE_PATH, None),
    )
    files = freeze.get("files")
    if not isinstance(files, list) or len(files) != len(required_roles):
        raise FormalProvenanceError("implementation role count drifted")
    for row, (role, relative_path, fixed_sha) in zip(
        files, required_roles, strict=True
    ):
        if not isinstance(row, Mapping):
            raise FormalProvenanceError("implementation role invalid")
        expected_sha = row.get("sha256")
        if (
            set(row) != {"role", "relative_path", "sha256"}
            or row.get("role") != role
            or row.get("relative_path") != relative_path.as_posix()
            or not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or (fixed_sha is not None and expected_sha != fixed_sha)
        ):
            raise FormalProvenanceError("implementation role drifted")
        current_raw = _read_public_file(root / relative_path, label=role)
        if _sha256(current_raw) != expected_sha:
            raise FormalProvenanceError("implementation working blob drifted")
        repository_path = "reconstruction_v2/" + relative_path.as_posix()
        committed_raw = _git(
            repository_root,
            "show",
            f"{implementation_commit}:{repository_path}",
        )
        if _sha256(committed_raw) != expected_sha:
            raise FormalProvenanceError("implementation committed blob drifted")
    _git(repository_root, "merge-base", "--is-ancestor", implementation_commit, "HEAD")
    tracked_paths = [
        "reconstruction_v2/" + relative_path.as_posix()
        for _role, relative_path, _fixed in required_roles
    ] + [
        "reconstruction_v2/" + IMPLEMENTATION_FREEZE_RELATIVE_PATH.as_posix()
    ]
    for tracked_path in tracked_paths:
        _git(repository_root, "ls-files", "--error-unmatch", "--", tracked_path)
    _git(repository_root, "diff", "--quiet", "HEAD", "--", *tracked_paths)
    return {
        "design_file_sha256": DESIGN_FILE_SHA256,
        "design_self_sha256": DESIGN_SELF_SHA256,
        "member_binding_file_sha256": MEMBER_BINDING_FILE_SHA256,
        "member_binding_self_sha256": MEMBER_BINDING_SELF_SHA256,
        "family_priority_amendment_file_sha256": AMENDMENT_FILE_SHA256,
        "family_priority_amendment_self_sha256": AMENDMENT_SELF_SHA256,
        "implementation_commit": implementation_commit,
        "implementation_freeze_file_sha256": _sha256(freeze_raw),
        "implementation_freeze_self_sha256": declared,
    }


def _read_bound_source(path: Path, *, size: int, sha256: str) -> bytes:
    try:
        before = path.lstat()
        raw = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise MavenEreSourceQualificationError("bound source unavailable") from exc
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o600
        or identity(before) != identity(after)
        or len(raw) != size
        or _sha256(raw) != sha256
    ):
        raise MavenEreSourceQualificationError("bound source identity drifted")
    return raw


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_write(path: Path, raw: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
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


def _require_formal_output(project_root: Path, output_root: Path) -> Path:
    expected = (project_root / FORMAL_OUTPUT_RELATIVE_PATH).resolve()
    try:
        metadata = output_root.lstat()
        output = output_root.resolve(strict=True)
    except OSError as exc:
        raise OneShotRefusal("formal output root must already exist") from exc
    if (
        output != expected
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or next(output.iterdir(), None) is not None
    ):
        raise OneShotRefusal("formal output root is not exact pristine mode-0700")
    return output


def build_formal_qualification(
    project_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve(strict=True)
    output = _require_formal_output(root, Path(output_root))
    provenance = validate_formal_provenance(root)
    marker = _self_hashed(
        {
            "schema": ATTEMPT_SCHEMA,
            "version": VERSION,
            "status": "formal_attempt_consumed_no_replay",
            "provenance": provenance,
        },
        "attempt_sha256",
    )
    _exclusive_write(output / "attempt.json", _canonical_json(marker) + b"\n")
    opened = {"train": 0, "valid": 0, "test": 0}
    try:
        source_raw: dict[str, bytes] = {}
        for split in ("train", "valid"):
            relative_path, size, sha256 = SOURCE_SPECS[split]
            opened[split] += 1
            source_raw[split] = _read_bound_source(
                root / relative_path,
                size=size,
                sha256=sha256,
            )
        source_binding = {
            **provenance,
            "train_file_sha256": SOURCE_SPECS["train"][2],
            "train_file_size": SOURCE_SPECS["train"][1],
            "valid_file_sha256": SOURCE_SPECS["valid"][2],
            "valid_file_size": SOURCE_SPECS["valid"][1],
            "train_open_count": opened["train"],
            "valid_open_count": opened["valid"],
            "hidden_TEST_open_count": opened["test"],
        }
        receipt = qualify_decoded_jsonl(
            source_raw["train"],
            source_raw["valid"],
            formal_identity_enforced=True,
            source_binding=source_binding,
        )
        _exclusive_write(
            output / "qualification.json",
            _canonical_json(receipt) + b"\n",
        )
        return receipt
    except Exception as exc:
        incident = _self_hashed(
            {
                "schema": "maven_ere_relation_context_source_qualification_incident_v1",
                "version": VERSION,
                "status": "terminal_no_replay_no_private_values",
                "failure_category": type(exc).__name__,
                "opened_content_counts": opened,
                "selection_secret_generated": False,
                "cohort_selected": False,
                "retrieval_action_evaluator_classifier_or_score_run": False,
                "hidden_TEST_opened": False,
                "same_source_retry_authorized": False,
            },
            "incident_sha256",
        )
        _exclusive_write(output / "incident.json", _canonical_json(incident) + b"\n")
        return incident


__all__ = [
    "CAUSAL_LABELS",
    "DESIGN_FILE_SHA256",
    "DESIGN_SELF_SHA256",
    "EXPECTED_LINE_COUNTS",
    "FAMILY_ORDER",
    "FORMAL_OUTPUT_RELATIVE_PATH",
    "FormalProvenanceError",
    "MavenEreSourceQualificationError",
    "OneShotRefusal",
    "REQUIRED_PER_SPLIT_FAMILY",
    "SCHEMA",
    "SOURCE_SPECS",
    "TEMPORAL_LABELS",
    "build_formal_qualification",
    "qualify_decoded_jsonl",
    "validate_formal_provenance",
]
