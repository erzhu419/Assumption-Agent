"""Aggregate-only, one-source qualification for the pinned QuAC v0.2 files.

The formal caller is allowed to read the two QuAC payloads exactly once only
after a separate source-free architecture qualification has passed.  This
module verifies the frozen byte identities before decoding either file, then
checks only the schema subset required by the prospective study:

* article ``title``, ``section_title`` and ``paragraphs``;
* paragraph ``context`` and ordered ``qas``;
* QA ``id``, ``question``, ``followup`` and ``orig_answer``;
* original-answer ``answer_start`` and ``text``.

No exact object key-set is guessed.  Additional official fields are ignored,
but every required value has a strict type.  Non-``CANNOTANSWER`` original
answers must exactly equal their context slice and map to at least one
96-whitespace-token window at stride 48.  A current turn is eligible only when
it is not the first turn and both its own and its immediately previous answer
map.  Its family is the *previous* turn's frozen ``followup`` value.

Leakage components are formed globally: equal page titles are one component,
and components are additionally unioned when any contexts have equal SHA-256
commitments.  A component may supply at most one future formal item across all
blocks and families.  A deterministic stdlib max-flow proves all nine
block-by-family quotas simultaneously; its assignment witness is discarded.
The returned value contains aggregate family, eligible-item, component,
flow/slack, and pass/fail facts only.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence


VERSION = "quac_p1_source_qualification_v1"
STUDY_ID = "QUAC_P1_RJMC_DIALOGUE_EVIDENCE_L5_V1"
SCHEMA = VERSION

TRAIN_EXPECTED_SIZE_BYTES = 68_114_819
TRAIN_EXPECTED_SHA256 = (
    "ff5cca5a2e4b4d1cb5b5ced68b9fce88394ef6d93117426d6d4baafbcc05c56a"
)
DEV_EXPECTED_SIZE_BYTES = 8_929_167
DEV_EXPECTED_SHA256 = (
    "09e622916280ba04c9352acb1bc5bbe80f11a2598f6f34e934c51d9e6570f378"
)

WINDOW_TOKEN_COUNT = 96
WINDOW_TOKEN_STRIDE = 48
CANNOTANSWER = "CANNOTANSWER"
QREL_ROLE_ORDER = ("previous_turn_orig_answer", "current_turn_orig_answer")
QREL_FALLBACK_ALLOWED = False
QREL_SAME_WINDOW_ALLOWED = True
FOLLOWUP_TO_FAMILY = {
    "y": "FOLLOW",
    "m": "MAYBE_FOLLOW",
    "n": "DONT_FOLLOW",
}
FAMILY_ORDER = ("FOLLOW", "MAYBE_FOLLOW", "DONT_FOLLOW")
PARTITION_ORDER = ("A_form", "A_hold", "M_search")
FORMAL_QUOTAS = {
    "A_form": 64,
    "A_hold": 32,
    "M_search": 32,
}

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_WHITESPACE_TOKEN = re.compile(r"\S+")


class QuacP1SourceQualificationError(RuntimeError):
    """The pinned source identity or required source topology failed closed."""


@dataclass(frozen=True)
class SourceFileContract:
    """Exact byte identity for one official source file."""

    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.size_bytes) is not int
            or self.size_bytes <= 0
            or not isinstance(self.sha256, str)
            or _HEX64.fullmatch(self.sha256) is None
        ):
            raise QuacP1SourceQualificationError(
                "source file contract is invalid"
            )


@dataclass(frozen=True)
class QualificationContract:
    """Exact sources and prospectively fixed aggregate quotas."""

    train: SourceFileContract
    dev: SourceFileContract
    quotas: Mapping[str, int]

    def __post_init__(self) -> None:
        if set(self.quotas) != set(PARTITION_ORDER):
            raise QuacP1SourceQualificationError(
                "qualification quota partition set is invalid"
            )
        if any(type(value) is not int or value <= 0 for value in self.quotas.values()):
            raise QuacP1SourceQualificationError(
                "qualification quota is invalid"
            )


FORMAL_CONTRACT = QualificationContract(
    train=SourceFileContract(
        TRAIN_EXPECTED_SIZE_BYTES,
        TRAIN_EXPECTED_SHA256,
    ),
    dev=SourceFileContract(
        DEV_EXPECTED_SIZE_BYTES,
        DEV_EXPECTED_SHA256,
    ),
    quotas=FORMAL_QUOTAS,
)


@dataclass(frozen=True)
class _AnswerSpan:
    """Transient span topology; never serialized."""

    maps_to_window: bool
    ineligibility_reason: str | None


@dataclass(frozen=True)
class _ParagraphRecord:
    """Private paragraph topology retained only during qualification."""

    official_split: str
    title: str
    context_sha256: str
    family_item_counts: Mapping[str, int]
    nonfirst_turn_count: int
    role_ineligibility_reason_counts: Mapping[str, int]


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise QuacP1SourceQualificationError(
            "qualification value is not canonical JSON"
        ) from exc


def stable_hash(value: object) -> str:
    """Return the frozen canonical semantic hash."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_text(
    value: object,
    *,
    label: str,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise QuacP1SourceQualificationError(f"{label} must be text")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise QuacP1SourceQualificationError(
            f"{label} contains invalid Unicode"
        ) from exc
    if not allow_empty and not value.strip():
        raise QuacP1SourceQualificationError(f"{label} must be nonempty")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QuacP1SourceQualificationError(f"{label} must be an object")
    return value


def _list(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise QuacP1SourceQualificationError(f"{label} must be an array")
    return value


def _require_keys(
    value: Mapping[str, Any],
    required: frozenset[str],
    *,
    label: str,
) -> None:
    if not required.issubset(value):
        raise QuacP1SourceQualificationError(
            f"{label} required fields are absent"
        )


def _object_without_duplicate_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QuacP1SourceQualificationError(
                "source JSON contains duplicate object keys"
            )
        result[key] = value
    return result


def _reject_nonfinite(_value: str) -> None:
    raise QuacP1SourceQualificationError(
        "source JSON contains a non-finite number"
    )


def _decode_strict_json(raw: bytes) -> object:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise QuacP1SourceQualificationError(
            "source is not strict UTF-8"
        ) from exc
    if text.startswith("\ufeff"):
        raise QuacP1SourceQualificationError(
            "source contains a forbidden UTF-8 BOM"
        )
    try:
        return json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except QuacP1SourceQualificationError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise QuacP1SourceQualificationError(
            "source is not strict JSON"
        ) from exc


def _read_exact_source(
    path: Path,
    contract: SourceFileContract,
) -> bytes:
    """Read one non-symlink descriptor and verify its exact frozen identity."""

    try:
        before = path.lstat()
    except OSError as exc:
        raise QuacP1SourceQualificationError(
            "pinned source file is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) != 0o600
        or before.st_size != contract.size_bytes
    ):
        raise QuacP1SourceQualificationError(
            "pinned source file identity drifted"
        )
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ):
            raise QuacP1SourceQualificationError(
                "pinned source changed during open"
            )
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
        after_open = os.fstat(descriptor)
    except OSError as exc:
        raise QuacP1SourceQualificationError(
            "pinned source read failed"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise QuacP1SourceQualificationError(
            "pinned source disappeared after read"
        ) from exc
    if (
        after_open.st_dev,
        after_open.st_ino,
        after_open.st_size,
        after_open.st_mtime_ns,
    ) != (
        after_path.st_dev,
        after_path.st_ino,
        after_path.st_size,
        after_path.st_mtime_ns,
    ):
        raise QuacP1SourceQualificationError(
            "pinned source changed while being read"
        )
    raw = b"".join(chunks)
    if (
        len(raw) != contract.size_bytes
        or digest.hexdigest() != contract.sha256
    ):
        raise QuacP1SourceQualificationError(
            "pinned source byte identity drifted"
        )
    return raw


def canonical_window_spans(context: str) -> tuple[tuple[int, int], ...]:
    """Return frozen 96/48 stride-aligned codepoint windows.

    The final window may be shorter than 96 tokens.  Once a stride-aligned
    window reaches the end of the context no redundant suffix-only windows
    are materialized.
    """

    tokens = tuple(_WHITESPACE_TOKEN.finditer(context))
    if not tokens:
        return ()
    windows: list[tuple[int, int]] = []
    for window_start in range(0, len(tokens), WINDOW_TOKEN_STRIDE):
        window_tokens = tokens[
            window_start : window_start + WINDOW_TOKEN_COUNT
        ]
        windows.append(
            (window_tokens[0].start(), window_tokens[-1].end())
        )
        if window_start + WINDOW_TOKEN_COUNT >= len(tokens):
            break
    return tuple(windows)


def _span_maps_to_window(
    context: str,
    *,
    answer_start: int,
    answer_text: str,
) -> bool:
    answer_end = answer_start + len(answer_text)
    # QuAC offsets and Python slicing are Unicode-codepoint based here.
    # Full [start,end) containment is required; no fuzzy or fallback map.
    return any(
        window_start <= answer_start and answer_end <= window_end
        for window_start, window_end in canonical_window_spans(context)
    )


def _validate_answer(
    raw_answer: object,
    *,
    context: str,
) -> _AnswerSpan:
    answer = _mapping(raw_answer, label="orig_answer")
    _require_keys(
        answer,
        frozenset({"answer_start", "text"}),
        label="orig_answer",
    )
    start = answer["answer_start"]
    text = _strict_text(
        answer["text"],
        label="orig_answer text",
        allow_empty=False,
    )
    if type(start) is not int:
        raise QuacP1SourceQualificationError(
            "orig_answer answer_start is invalid"
        )
    if text == CANNOTANSWER:
        return _AnswerSpan(
            maps_to_window=False,
            ineligibility_reason="CANNOTANSWER",
        )
    if start < 0:
        raise QuacP1SourceQualificationError(
            "non-CANNOTANSWER original span is not exact"
        )
    end = start + len(text)
    if end > len(context) or context[start:end] != text:
        raise QuacP1SourceQualificationError(
            "non-CANNOTANSWER original span is not exact"
        )
    maps_to_window = _span_maps_to_window(
        context,
        answer_start=start,
        answer_text=text,
    )
    return _AnswerSpan(
        maps_to_window=maps_to_window,
        ineligibility_reason=(
            None if maps_to_window else "NOT_CONTAINED_IN_FROZEN_WINDOW"
        ),
    )


def _parse_split(
    payload: object,
    *,
    official_split: str,
) -> list[_ParagraphRecord]:
    root = _mapping(payload, label="source root")
    _require_keys(root, frozenset({"data"}), label="source root")
    articles = _list(root["data"], label="source data")
    records: list[_ParagraphRecord] = []
    for raw_article in articles:
        article = _mapping(raw_article, label="article")
        _require_keys(
            article,
            frozenset({"title", "section_title", "paragraphs"}),
            label="article",
        )
        title = _strict_text(
            article["title"],
            label="article title",
            allow_empty=True,
        )
        _strict_text(
            article["section_title"],
            label="article section_title",
            allow_empty=True,
        )
        paragraphs = _list(article["paragraphs"], label="article paragraphs")
        for raw_paragraph in paragraphs:
            paragraph = _mapping(raw_paragraph, label="paragraph")
            _require_keys(
                paragraph,
                frozenset({"context", "qas"}),
                label="paragraph",
            )
            context = _strict_text(
                paragraph["context"],
                label="context",
                allow_empty=True,
            )
            qas = _list(paragraph["qas"], label="qas")
            # Section title is required and strictly typed because it is part
            # of the prospective item topology, but leakage grouping is the
            # more conservative page-title union context-hash relation.
            answers: list[_AnswerSpan] = []
            followups: list[str] = []
            for raw_qa in qas:
                qa = _mapping(raw_qa, label="QA")
                _require_keys(
                    qa,
                    frozenset(
                        {
                            "id",
                            "question",
                            "followup",
                            "orig_answer",
                        }
                    ),
                    label="QA",
                )
                _strict_text(
                    qa["id"],
                    label="QA id",
                    allow_empty=True,
                )
                _strict_text(
                    qa["question"],
                    label="QA question",
                    allow_empty=True,
                )
                followup = qa["followup"]
                if (
                    not isinstance(followup, str)
                    or followup not in FOLLOWUP_TO_FAMILY
                ):
                    raise QuacP1SourceQualificationError(
                        "invalid followup"
                    )
                followups.append(followup)
                answers.append(
                    _validate_answer(qa["orig_answer"], context=context)
                )
            family_counts: Counter[str] = Counter()
            role_ineligibility_counts: Counter[str] = Counter()
            for turn_index in range(1, len(qas)):
                if not (
                    answers[turn_index - 1].maps_to_window
                    and answers[turn_index].maps_to_window
                ):
                    for role, answer in (
                        ("previous", answers[turn_index - 1]),
                        ("current", answers[turn_index]),
                    ):
                        if answer.ineligibility_reason is not None:
                            role_ineligibility_counts[
                                f"{role}_{answer.ineligibility_reason}"
                            ] += 1
                    continue
                family = FOLLOWUP_TO_FAMILY[followups[turn_index - 1]]
                family_counts[family] += 1
            records.append(
                _ParagraphRecord(
                    official_split=official_split,
                    title=title,
                    context_sha256=hashlib.sha256(
                        context.encode("utf-8")
                    ).hexdigest(),
                    family_item_counts={
                        family: int(family_counts[family])
                        for family in FAMILY_ORDER
                    },
                    nonfirst_turn_count=max(0, len(qas) - 1),
                    role_ineligibility_reason_counts={
                        role_reason: int(
                            role_ineligibility_counts[role_reason]
                        )
                        for role_reason in (
                            "previous_CANNOTANSWER",
                            "current_CANNOTANSWER",
                            "previous_NOT_CONTAINED_IN_FROZEN_WINDOW",
                            "current_NOT_CONTAINED_IN_FROZEN_WINDOW",
                        )
                    },
                )
            )
    return records


class _UnionFind:
    """Small deterministic union-find for private source components."""

    def __init__(self, count: int) -> None:
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


def _global_components(
    records: Sequence[_ParagraphRecord],
) -> tuple[
    dict[int, set[str]],
    dict[int, dict[str, set[str]]],
    dict[str, Counter[str]],
    Counter[str],
    dict[str, Counter[str]],
    dict[str, set[int]],
    dict[str, dict[str, set[int]]],
]:
    union = _UnionFind(len(records))
    first_by_title: dict[str, int] = {}
    first_by_context: dict[str, int] = {}
    for index, record in enumerate(records):
        prior_title = first_by_title.setdefault(record.title, index)
        union.union(index, prior_title)
        prior_context = first_by_context.setdefault(
            record.context_sha256,
            index,
        )
        union.union(index, prior_context)

    availability: dict[int, set[str]] = {}
    by_split_family: dict[int, dict[str, set[str]]] = {}
    split_items = {
        split: Counter() for split in ("train", "dev")
    }
    split_nonfirst_turns = Counter()
    split_role_ineligibility_reasons = {
        split: Counter() for split in ("train", "dev")
    }
    split_components = {
        split: set() for split in ("train", "dev")
    }
    split_family_components = {
        split: {family: set() for family in FAMILY_ORDER}
        for split in ("train", "dev")
    }
    for index, record in enumerate(records):
        component = union.find(index)
        split_nonfirst_turns[record.official_split] += (
            record.nonfirst_turn_count
        )
        split_role_ineligibility_reasons[record.official_split].update(
            record.role_ineligibility_reason_counts
        )
        split_components[record.official_split].add(component)
        availability.setdefault(component, set()).add(
            record.official_split
        )
        family_map = by_split_family.setdefault(
            component,
            {"train": set(), "dev": set()},
        )
        for family in FAMILY_ORDER:
            count = int(record.family_item_counts[family])
            split_items[record.official_split][family] += count
            if count > 0:
                family_map[record.official_split].add(family)
                split_family_components[record.official_split][family].add(
                    component
                )
    return (
        availability,
        by_split_family,
        split_items,
        split_nonfirst_turns,
        split_role_ineligibility_reasons,
        split_components,
        split_family_components,
    )


def _deterministic_capacity_flow(
    component_families: Mapping[int, Mapping[str, set[str]]],
    quotas: Mapping[str, int],
) -> tuple[int, dict[str, dict[str, int]]]:
    """Solve the frozen component-capacity problem and discard its witness."""

    source = "SOURCE"
    sink = "SINK"
    adjacency: dict[str, list[str]] = {}
    capacity: dict[tuple[str, str], int] = {}

    def add_edge(left: str, right: str, amount: int) -> None:
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, []).append(left)
        capacity[(left, right)] = amount
        capacity[(right, left)] = 0

    slot_nodes: dict[tuple[str, str], str] = {}
    for block in PARTITION_ORDER:
        for family in FAMILY_ORDER:
            node = f"SLOT:{block}:{family}"
            slot_nodes[(block, family)] = node
            add_edge(node, sink, int(quotas[block]))

    for component in sorted(component_families):
        node = f"COMPONENT:{component}"
        add_edge(source, node, 1)
        split_map = component_families[component]
        for family in sorted(split_map["train"], key=FAMILY_ORDER.index):
            add_edge(node, slot_nodes[("A_form", family)], 1)
        for family in sorted(split_map["dev"], key=FAMILY_ORDER.index):
            add_edge(node, slot_nodes[("A_hold", family)], 1)
            add_edge(node, slot_nodes[("M_search", family)], 1)

    total_flow = 0
    while True:
        predecessor: dict[str, str | None] = {source: None}
        queue: deque[str] = deque([source])
        while queue and sink not in predecessor:
            node = queue.popleft()
            for neighbor in adjacency.get(node, ()):
                if (
                    neighbor not in predecessor
                    and capacity.get((node, neighbor), 0) > 0
                ):
                    predecessor[neighbor] = node
                    queue.append(neighbor)
                    if neighbor == sink:
                        break
        if sink not in predecessor:
            break
        node = sink
        while predecessor[node] is not None:
            prior = predecessor[node]
            assert prior is not None
            capacity[(prior, node)] -= 1
            capacity[(node, prior)] += 1
            node = prior
        total_flow += 1

    slot_flow: dict[str, dict[str, int]] = {}
    for block in PARTITION_ORDER:
        slot_flow[block] = {}
        for family in FAMILY_ORDER:
            node = slot_nodes[(block, family)]
            remaining = capacity[(node, sink)]
            slot_flow[block][family] = int(quotas[block]) - remaining
    return total_flow, slot_flow


def qualify_decoded_sources(
    train_payload: object,
    dev_payload: object,
    *,
    quotas: Mapping[str, int] = FORMAL_QUOTAS,
) -> dict[str, Any]:
    """Validate decoded sources and return aggregate-only capacity facts."""

    if (
        set(quotas) != set(PARTITION_ORDER)
        or any(type(value) is not int or value <= 0 for value in quotas.values())
    ):
        raise QuacP1SourceQualificationError(
            "qualification quotas are invalid"
        )
    train_records = _parse_split(
        train_payload,
        official_split="train",
    )
    dev_records = _parse_split(
        dev_payload,
        official_split="dev",
    )
    records = [*train_records, *dev_records]
    (
        _component_splits,
        component_families,
        split_items,
        split_nonfirst_turns,
        split_role_ineligibility_reasons,
        split_components,
        split_family_components,
    ) = _global_components(records)
    overlap_count = len(
        split_components["train"].intersection(split_components["dev"])
    )
    flow, slot_flow = _deterministic_capacity_flow(
        component_families,
        quotas,
    )
    required_flow = sum(
        int(quotas[block]) * len(FAMILY_ORDER)
        for block in PARTITION_ORDER
    )
    slot_slack = {
        block: {
            family: int(quotas[block]) - slot_flow[block][family]
            for family in FAMILY_ORDER
        }
        for block in PARTITION_ORDER
    }
    overall_pass = flow == required_flow
    source_aggregates = {
        split: {
            "component_count": len(split_components[split]),
            "eligible_component_count": len(
                set().union(
                    *split_family_components[split].values()
                )
            ),
            "eligible_item_count": sum(
                int(split_items[split][family])
                for family in FAMILY_ORDER
            ),
            "nonfirst_turn_count": int(split_nonfirst_turns[split]),
            "role_ineligibility_reason_counts": {
                reason: int(
                    split_role_ineligibility_reasons[split][reason]
                )
                for reason in (
                    "previous_CANNOTANSWER",
                    "current_CANNOTANSWER",
                    "previous_NOT_CONTAINED_IN_FROZEN_WINDOW",
                    "current_NOT_CONTAINED_IN_FROZEN_WINDOW",
                )
            },
            "family_eligible_component_counts": {
                family: len(split_family_components[split][family])
                for family in FAMILY_ORDER
            },
            "family_eligible_item_counts": {
                family: int(split_items[split][family])
                for family in FAMILY_ORDER
            },
        }
        for split in ("train", "dev")
    }
    return {
        "schema": SCHEMA,
        "status": (
            "PASS_QUAC_SCHEMA_TOPOLOGY_AND_FAMILY_CAPACITY"
            if overall_pass
            else "STOP_QUAC_FAMILY_CAPACITY"
        ),
        "passed": overall_pass,
        "source_identity_pass": {"train": True, "dev": True},
        "required_schema_subset_pass": True,
        "train_dev_component_overlap_count": overlap_count,
        "global_component_count": len(component_families),
        "source_aggregates": source_aggregates,
        "activity_counts": {
            "selection": 0,
            "model": 0,
            "action": 0,
            "score": 0,
            "online_or_API_evaluation": 0,
        },
        "capacity_flow": {
            "component_global_capacity": 1,
            "required_flow": required_flow,
            "achieved_flow": flow,
            "aggregate_slack": required_flow - flow,
            "slot_flow": slot_flow,
            "slot_slack": slot_slack,
            "all_nine_slots_saturated": overall_pass,
            "assignment_witness_output_count": 0,
        },
    }


def qualify_source_files(
    train_path: Path,
    dev_path: Path,
    *,
    contract: QualificationContract = FORMAL_CONTRACT,
) -> dict[str, Any]:
    """Verify exact bytes, decode once, and return aggregate-only capacity."""

    if train_path == dev_path:
        raise QuacP1SourceQualificationError(
            "TRAIN and DEV source paths must differ"
        )
    train_raw = _read_exact_source(train_path, contract.train)
    dev_raw = _read_exact_source(dev_path, contract.dev)
    return qualify_decoded_sources(
        _decode_strict_json(train_raw),
        _decode_strict_json(dev_raw),
        quotas=contract.quotas,
    )


__all__ = [
    "DEV_EXPECTED_SHA256",
    "DEV_EXPECTED_SIZE_BYTES",
    "FORMAL_CONTRACT",
    "FORMAL_QUOTAS",
    "QREL_FALLBACK_ALLOWED",
    "QREL_ROLE_ORDER",
    "QREL_SAME_WINDOW_ALLOWED",
    "QualificationContract",
    "QuacP1SourceQualificationError",
    "SourceFileContract",
    "TRAIN_EXPECTED_SHA256",
    "TRAIN_EXPECTED_SIZE_BYTES",
    "canonical_window_spans",
    "qualify_decoded_sources",
    "qualify_source_files",
    "stable_hash",
]
