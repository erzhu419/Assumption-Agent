"""Trusted, source-decoded acquisition boundary for the QuAC RJMC study.

This module deliberately performs no filesystem or network I/O.  A formal
controller supplies the two already decoded, byte-qualified QuAC objects and
one 32-byte study secret.  The module then:

* reconstructs the exact source-qualification eligibility semantics;
* unions paragraph components globally by exact title or exact context hash;
* solves all nine block/family quotas in one component-capacity-one max-flow;
* chooses one item per assigned component with domain-separated HMACs;
* forms label-free views and separate two-role late-label packs; and
* keeps ``M_search`` rows behind an authenticated, one-use promotion
  capability.

The public/action-facing view never contains the official split, family,
source coordinate, native context, answer span, qrel, or a query-to-native-
context association.  Native source records stay inside the trusted broker.
Before promotion, the only M_search surface is an opaque reservation
commitment whose materialization and path counters are both zero.

All persisted payload helpers use exact canonical JSON and strict schemas.
The module accepts decoded objects solely to make the acquisition logic
source-free-testable; opening and byte-verifying the official files remains
the formal controller's responsibility.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks import (
    quac_p1_source_qualification_v1 as source_qualification,
)


VERSION = "quac_p1_formal_acquisition_v1"
STUDY_ID = source_qualification.STUDY_ID

BLOCK_ORDER = ("A_form", "A_hold", "M_search")
FAMILY_ORDER = source_qualification.FAMILY_ORDER
BLOCK_SPLIT = {
    "A_form": "train",
    "A_hold": "dev",
    "M_search": "dev",
}
FORMAL_QUOTAS = dict(source_qualification.FORMAL_QUOTAS)

HMAC_SECRET_BYTES = 32
FOLD_COUNT = 5
FORMAL_A_FORM_COUNT = FORMAL_QUOTAS["A_form"] * len(FAMILY_ORDER)
FORMAL_FOLD_SIZES = (39, 39, 38, 38, 38)

VIEW_SCHEMA = "quac_p1_label_free_view_pack_v1"
LABEL_SCHEMA = "quac_p1_private_late_label_pack_v1"
RESERVATION_SCHEMA = "quac_p1_m_search_opaque_reservation_v1"
PROMOTION_PROOF_SCHEMA = "quac_p1_a_hold_promotion_proof_v1"
M_CAPABILITY_SCHEMA = "quac_p1_m_search_capability_v1"
LABEL_CAPABILITY_SCHEMA = "quac_p1_late_label_capability_v1"
A_FORM_MODEL_SEAL_SCHEMA = "quac_p1_a_form_model_seal_v1"
M_MATERIALIZED_REGISTRY_SCHEMA = (
    "quac_p1_m_search_materialized_registry_v1"
)

QREL_ROLE_ORDER = source_qualification.QREL_ROLE_ORDER
FOLLOWUP_TO_FAMILY = source_qualification.FOLLOWUP_TO_FAMILY
CANNOTANSWER = source_qualification.CANNOTANSWER

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_WORK_ID = re.compile(r"quac-work-v1-[0-9a-f]{64}\Z")

_HMAC_DOMAINS = frozenset(
    {
        "component-order-v1",
        "component-slot-order-v1",
        "context-id-v1",
        "block-id-v1",
        "item-order-v1",
        "work-id-v1",
        "window-unit-id-v1",
        "a-form-fold-order-v1",
        "a-form-model-seal-v1",
        "m-reservation-v1",
        "m-capability-v1",
        "late-label-capability-v1",
    }
)

FORBIDDEN_VIEW_KEYS = frozenset(
    {
        "answer",
        "answer_start",
        "answer_text",
        "article_ordinal",
        "component_commitment",
        "context",
        "context_sha256",
        "family",
        "followup",
        "gold",
        "label",
        "native_context",
        "native_context_id",
        "paragraph_ordinal",
        "qrel",
        "qrels",
        "source_coordinate",
        "source_ordinal",
        "split",
        "official_split",
        "section_title",
        "title",
        "turn_index",
    }
)


class QuacP1FormalAcquisitionError(RuntimeError):
    """A decoded source, selection, pack, or capability failed closed."""


def _assert_json_types(value: object) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise QuacP1FormalAcquisitionError(
                "canonical JSON contains a non-finite number"
            )
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _assert_json_types(child)
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise QuacP1FormalAcquisitionError(
                "canonical JSON object key is not text"
            )
        for child in value.values():
            _assert_json_types(child)
        return
    raise QuacP1FormalAcquisitionError(
        "value contains a non-exact JSON type"
    )


def canonical_bytes(value: object) -> bytes:
    """Encode exact newline-free ASCII canonical JSON."""

    _assert_json_types(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise QuacP1FormalAcquisitionError(
            "value is not canonical JSON"
        ) from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise QuacP1FormalAcquisitionError(
            f"{field_name} must be a lowercase SHA256"
        )
    return value


def _strict_text(
    value: object,
    *,
    field_name: str,
    allow_empty: bool,
) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise QuacP1FormalAcquisitionError(f"{field_name} must be text")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise QuacP1FormalAcquisitionError(
            f"{field_name} contains invalid Unicode"
        ) from exc
    if not allow_empty and not value.strip():
        raise QuacP1FormalAcquisitionError(
            f"{field_name} must be nonempty"
        )
    return value


def _mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QuacP1FormalAcquisitionError(
            f"{field_name} must be an object"
        )
    if any(not isinstance(key, str) for key in value):
        raise QuacP1FormalAcquisitionError(
            f"{field_name} contains a non-text key"
        )
    return value


def _array(value: object, *, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise QuacP1FormalAcquisitionError(
            f"{field_name} must be an array"
        )
    return value


def _require_keys(
    value: Mapping[str, Any],
    required: frozenset[str],
    *,
    field_name: str,
) -> None:
    if not required.issubset(value):
        raise QuacP1FormalAcquisitionError(
            f"{field_name} required fields are absent"
        )


@dataclass(frozen=True, slots=True)
class SelectionSecret:
    """The one whole-study secret; raw bytes have no serialization method."""

    _value: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if type(self._value) is not bytes or len(self._value) != HMAC_SECRET_BYTES:
            raise QuacP1FormalAcquisitionError(
                "selection secret must be exactly 32 bytes"
            )

    @property
    def commitment(self) -> str:
        return hashlib.sha256(self._value).hexdigest()

    def digest(self, domain: str, *parts: object) -> str:
        if domain not in _HMAC_DOMAINS:
            raise QuacP1FormalAcquisitionError(
                "HMAC domain is outside the frozen registry"
            )
        framed = bytearray(b"QUAC-P1-FORMAL-HMAC\x00")
        domain_bytes = domain.encode("ascii")
        framed.extend(len(domain_bytes).to_bytes(2, "big"))
        framed.extend(domain_bytes)
        for part in parts:
            raw = canonical_bytes(part)
            framed.extend(len(raw).to_bytes(8, "big"))
            framed.extend(raw)
        return hmac.new(
            self._value,
            bytes(framed),
            hashlib.sha256,
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class AnswerSpan:
    """One singular official ``orig_answer`` mapped without fallback."""

    answer_start: int
    answer_text: str
    window_ordinals: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            type(self.answer_start) is not int
            or self.answer_start < 0
            or not isinstance(self.answer_text, str)
            or not self.answer_text
            or self.answer_text == CANNOTANSWER
            or not self.window_ordinals
            or any(
                type(value) is not int or value < 0
                for value in self.window_ordinals
            )
            or tuple(sorted(set(self.window_ordinals)))
            != self.window_ordinals
        ):
            raise QuacP1FormalAcquisitionError(
                "eligible original-answer span is invalid"
            )


@dataclass(frozen=True, slots=True)
class EligibleItem:
    """Trusted private source record; never used as an action-facing view."""

    item_commitment: str
    component_commitment: str
    official_split: str
    article_ordinal: int
    paragraph_ordinal: int
    turn_index: int
    title: str
    section_title: str
    context: str
    context_sha256: str
    question_text: str
    recent_questions: tuple[str, ...]
    family: str
    previous_answer: AnswerSpan
    current_answer: AnswerSpan

    def __post_init__(self) -> None:
        _require_sha256(self.item_commitment, "item commitment")
        _require_sha256(self.component_commitment, "component commitment")
        _require_sha256(self.context_sha256, "context SHA256")
        if self.official_split not in {"train", "dev"}:
            raise QuacP1FormalAcquisitionError(
                "eligible item split is invalid"
            )
        if (
            any(
                type(value) is not int or value < 0
                for value in (
                    self.article_ordinal,
                    self.paragraph_ordinal,
                    self.turn_index,
                )
            )
            or self.turn_index < 1
        ):
            raise QuacP1FormalAcquisitionError(
                "eligible item source coordinate is invalid"
            )
        for value, name in (
            (self.title, "title"),
            (self.section_title, "section title"),
            (self.context, "context"),
            (self.question_text, "question"),
        ):
            _strict_text(value, field_name=name, allow_empty=True)
        if (
            hashlib.sha256(self.context.encode("utf-8")).hexdigest()
            != self.context_sha256
        ):
            raise QuacP1FormalAcquisitionError(
                "eligible item context commitment drifted"
            )
        if (
            not 1 <= len(self.recent_questions) <= 4
            or self.recent_questions[-1] != self.question_text
            or any(not isinstance(value, str) for value in self.recent_questions)
        ):
            raise QuacP1FormalAcquisitionError(
                "recent dialogue questions are invalid"
            )
        if self.family not in FAMILY_ORDER:
            raise QuacP1FormalAcquisitionError(
                "eligible item family is invalid"
            )


@dataclass(frozen=True, slots=True)
class SourceIndex:
    """All private eligible items after global component reconstruction."""

    items: tuple[EligibleItem, ...]
    paragraph_count: int
    component_count: int

    def __post_init__(self) -> None:
        if (
            type(self.paragraph_count) is not int
            or self.paragraph_count < 0
            or type(self.component_count) is not int
            or self.component_count < 0
            or self.component_count > self.paragraph_count
        ):
            raise QuacP1FormalAcquisitionError(
                "source index aggregate is invalid"
            )
        commitments = tuple(item.item_commitment for item in self.items)
        if (
            len(set(commitments)) != len(commitments)
            or commitments != tuple(sorted(commitments))
        ):
            raise QuacP1FormalAcquisitionError(
                "source index items are not canonical and unique"
            )


@dataclass(frozen=True, slots=True)
class _ProvisionalItem:
    item_commitment: str
    paragraph_index: int
    official_split: str
    article_ordinal: int
    paragraph_ordinal: int
    turn_index: int
    title: str
    section_title: str
    context: str
    context_sha256: str
    question_text: str
    recent_questions: tuple[str, ...]
    family: str
    previous_answer: AnswerSpan
    current_answer: AnswerSpan


@dataclass(frozen=True, slots=True)
class _Paragraph:
    official_split: str
    article_ordinal: int
    paragraph_ordinal: int
    title: str
    section_title: str
    context: str
    context_sha256: str
    paragraph_commitment: str


class _UnionFind:
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


def _parse_answer(
    raw_answer: object,
    *,
    context: str,
) -> AnswerSpan | None:
    answer = _mapping(raw_answer, field_name="orig_answer")
    _require_keys(
        answer,
        frozenset({"answer_start", "text"}),
        field_name="orig_answer",
    )
    start = answer["answer_start"]
    text = _strict_text(
        answer["text"],
        field_name="orig_answer text",
        allow_empty=False,
    )
    if type(start) is not int:
        raise QuacP1FormalAcquisitionError(
            "orig_answer answer_start is invalid"
        )
    if text == CANNOTANSWER:
        return None
    if start < 0:
        raise QuacP1FormalAcquisitionError(
            "non-CANNOTANSWER original span is not exact"
        )
    end = start + len(text)
    if end > len(context) or context[start:end] != text:
        raise QuacP1FormalAcquisitionError(
            "non-CANNOTANSWER original span is not exact"
        )
    windows = source_qualification.canonical_window_spans(context)
    ordinals = tuple(
        ordinal
        for ordinal, (window_start, window_end) in enumerate(windows)
        if window_start <= start and end <= window_end
    )
    if not ordinals:
        return None
    return AnswerSpan(
        answer_start=start,
        answer_text=text,
        window_ordinals=ordinals,
    )


def _parse_split(
    payload: object,
    *,
    official_split: str,
    paragraph_offset: int,
) -> tuple[list[_Paragraph], list[_ProvisionalItem]]:
    root = _mapping(payload, field_name="source root")
    _require_keys(root, frozenset({"data"}), field_name="source root")
    articles = _array(root["data"], field_name="source data")
    paragraphs: list[_Paragraph] = []
    items: list[_ProvisionalItem] = []
    for article_ordinal, raw_article in enumerate(articles):
        article = _mapping(raw_article, field_name="article")
        _require_keys(
            article,
            frozenset({"title", "section_title", "paragraphs"}),
            field_name="article",
        )
        title = _strict_text(
            article["title"],
            field_name="article title",
            allow_empty=True,
        )
        section_title = _strict_text(
            article["section_title"],
            field_name="article section_title",
            allow_empty=True,
        )
        raw_paragraphs = _array(
            article["paragraphs"],
            field_name="article paragraphs",
        )
        for paragraph_ordinal, raw_paragraph in enumerate(raw_paragraphs):
            paragraph = _mapping(
                raw_paragraph,
                field_name="paragraph",
            )
            _require_keys(
                paragraph,
                frozenset({"context", "qas"}),
                field_name="paragraph",
            )
            context = _strict_text(
                paragraph["context"],
                field_name="context",
                allow_empty=True,
            )
            context_sha256 = hashlib.sha256(
                context.encode("utf-8")
            ).hexdigest()
            paragraph_commitment = stable_hash(
                {
                    "article_ordinal": article_ordinal,
                    "context_sha256": context_sha256,
                    "official_split": official_split,
                    "paragraph_ordinal": paragraph_ordinal,
                    "schema": "quac_p1_private_paragraph_identity_v1",
                    "title_sha256": hashlib.sha256(
                        title.encode("utf-8")
                    ).hexdigest(),
                }
            )
            paragraph_index = paragraph_offset + len(paragraphs)
            paragraphs.append(
                _Paragraph(
                    official_split=official_split,
                    article_ordinal=article_ordinal,
                    paragraph_ordinal=paragraph_ordinal,
                    title=title,
                    section_title=section_title,
                    context=context,
                    context_sha256=context_sha256,
                    paragraph_commitment=paragraph_commitment,
                )
            )

            qas = _array(paragraph["qas"], field_name="qas")
            questions: list[str] = []
            followups: list[str] = []
            answers: list[AnswerSpan | None] = []
            for raw_qa in qas:
                qa = _mapping(raw_qa, field_name="QA")
                _require_keys(
                    qa,
                    frozenset(
                        {"id", "question", "followup", "orig_answer"}
                    ),
                    field_name="QA",
                )
                _strict_text(
                    qa["id"],
                    field_name="QA id",
                    allow_empty=True,
                )
                questions.append(
                    _strict_text(
                        qa["question"],
                        field_name="QA question",
                        allow_empty=True,
                    )
                )
                followup = qa["followup"]
                if (
                    not isinstance(followup, str)
                    or followup not in FOLLOWUP_TO_FAMILY
                ):
                    raise QuacP1FormalAcquisitionError(
                        "invalid followup"
                    )
                followups.append(followup)
                answers.append(
                    _parse_answer(qa["orig_answer"], context=context)
                )
            for turn_index in range(1, len(qas)):
                previous = answers[turn_index - 1]
                current = answers[turn_index]
                if previous is None or current is None:
                    continue
                family = FOLLOWUP_TO_FAMILY[followups[turn_index - 1]]
                item_commitment = stable_hash(
                    {
                        "article_ordinal": article_ordinal,
                        "context_sha256": context_sha256,
                        "official_split": official_split,
                        "paragraph_ordinal": paragraph_ordinal,
                        "question_sha256": hashlib.sha256(
                            questions[turn_index].encode("utf-8")
                        ).hexdigest(),
                        "schema": "quac_p1_private_item_identity_v1",
                        "turn_index": turn_index,
                    }
                )
                items.append(
                    _ProvisionalItem(
                        item_commitment=item_commitment,
                        paragraph_index=paragraph_index,
                        official_split=official_split,
                        article_ordinal=article_ordinal,
                        paragraph_ordinal=paragraph_ordinal,
                        turn_index=turn_index,
                        title=title,
                        section_title=section_title,
                        context=context,
                        context_sha256=context_sha256,
                        question_text=questions[turn_index],
                        recent_questions=tuple(
                            questions[max(0, turn_index - 3) : turn_index + 1]
                        ),
                        family=family,
                        previous_answer=previous,
                        current_answer=current,
                    )
                )
    return paragraphs, items


def build_source_index(
    train_payload: object,
    dev_payload: object,
) -> SourceIndex:
    """Reconstruct all eligible items and exact global leakage components."""

    train_paragraphs, train_items = _parse_split(
        train_payload,
        official_split="train",
        paragraph_offset=0,
    )
    dev_paragraphs, dev_items = _parse_split(
        dev_payload,
        official_split="dev",
        paragraph_offset=len(train_paragraphs),
    )
    paragraphs = [*train_paragraphs, *dev_paragraphs]
    provisional = [*train_items, *dev_items]

    union = _UnionFind(len(paragraphs))
    first_by_title: dict[str, int] = {}
    first_by_context: dict[str, int] = {}
    for index, paragraph in enumerate(paragraphs):
        prior_title = first_by_title.setdefault(paragraph.title, index)
        union.union(index, prior_title)
        prior_context = first_by_context.setdefault(
            paragraph.context_sha256,
            index,
        )
        union.union(index, prior_context)

    members_by_root: dict[int, list[str]] = {}
    for index, paragraph in enumerate(paragraphs):
        members_by_root.setdefault(union.find(index), []).append(
            paragraph.paragraph_commitment
        )
    component_by_root = {
        root: stable_hash(
            {
                "member_paragraph_commitments": sorted(members),
                "schema": "quac_p1_global_title_or_context_component_v1",
            }
        )
        for root, members in members_by_root.items()
    }

    items = tuple(
        sorted(
            (
                EligibleItem(
                    item_commitment=item.item_commitment,
                    component_commitment=component_by_root[
                        union.find(item.paragraph_index)
                    ],
                    official_split=item.official_split,
                    article_ordinal=item.article_ordinal,
                    paragraph_ordinal=item.paragraph_ordinal,
                    turn_index=item.turn_index,
                    title=item.title,
                    section_title=item.section_title,
                    context=item.context,
                    context_sha256=item.context_sha256,
                    question_text=item.question_text,
                    recent_questions=item.recent_questions,
                    family=item.family,
                    previous_answer=item.previous_answer,
                    current_answer=item.current_answer,
                )
                for item in provisional
            ),
            key=lambda row: row.item_commitment,
        )
    )
    return SourceIndex(
        items=items,
        paragraph_count=len(paragraphs),
        component_count=len(members_by_root),
    )


@dataclass(slots=True)
class _FlowEdge:
    to: int
    reverse: int
    capacity: int


def _add_edge(
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


def _normalize_quotas(
    quotas: Mapping[str, int],
) -> dict[str, int]:
    if (
        not isinstance(quotas, Mapping)
        or set(quotas) != set(BLOCK_ORDER)
        or any(type(quotas[block]) is not int or quotas[block] <= 0 for block in BLOCK_ORDER)
    ):
        raise QuacP1FormalAcquisitionError(
            "block quotas are invalid"
        )
    return {block: int(quotas[block]) for block in BLOCK_ORDER}


def _simultaneous_assignment(
    items: Sequence[EligibleItem],
    *,
    secret: SelectionSecret,
    quotas: Mapping[str, int],
) -> dict[str, tuple[str, str]]:
    """Return component -> (block, family) from one global residual flow."""

    normalized = _normalize_quotas(quotas)
    items_by_component: dict[str, list[EligibleItem]] = {}
    for item in items:
        items_by_component.setdefault(
            item.component_commitment,
            [],
        ).append(item)

    availability: dict[str, set[tuple[str, str]]] = {}
    for component, rows in items_by_component.items():
        slots: set[tuple[str, str]] = set()
        for row in rows:
            for block in BLOCK_ORDER:
                if row.official_split == BLOCK_SPLIT[block]:
                    slots.add((block, row.family))
        availability[component] = slots

    component_order = sorted(
        availability,
        key=lambda component: (
            secret.digest(
                "component-order-v1",
                {"component_commitment": component},
            ),
            component,
        ),
    )
    slots = tuple(
        (block, family)
        for block in BLOCK_ORDER
        for family in FAMILY_ORDER
    )
    source = 0
    component_nodes = {
        component: index + 1
        for index, component in enumerate(component_order)
    }
    slot_offset = 1 + len(component_order)
    slot_nodes = {
        slot: slot_offset + index for index, slot in enumerate(slots)
    }
    sink = slot_offset + len(slots)
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    assignment_edges: dict[
        tuple[str, tuple[str, str]], _FlowEdge
    ] = {}

    for component in component_order:
        component_node = component_nodes[component]
        _add_edge(graph, source, component_node, 1)
        ordered_slots = sorted(
            availability[component],
            key=lambda slot: (
                secret.digest(
                    "component-slot-order-v1",
                    {
                        "block": slot[0],
                        "component_commitment": component,
                        "family": slot[1],
                    },
                ),
                BLOCK_ORDER.index(slot[0]),
                FAMILY_ORDER.index(slot[1]),
            ),
        )
        for slot in ordered_slots:
            assignment_edges[(component, slot)] = _add_edge(
                graph,
                component_node,
                slot_nodes[slot],
                1,
            )
    for block, family in slots:
        _add_edge(
            graph,
            slot_nodes[(block, family)],
            sink,
            normalized[block],
        )

    required = sum(normalized.values()) * len(FAMILY_ORDER)
    achieved = 0
    while achieved < required:
        predecessor: list[tuple[int, int] | None] = [None] * len(graph)
        predecessor[source] = (source, -1)
        queue: deque[int] = deque([source])
        while queue and predecessor[sink] is None:
            node = queue.popleft()
            for edge_index, edge in enumerate(graph[node]):
                if edge.capacity <= 0 or predecessor[edge.to] is not None:
                    continue
                predecessor[edge.to] = (node, edge_index)
                queue.append(edge.to)
                if edge.to == sink:
                    break
        if predecessor[sink] is None:
            break
        node = sink
        while node != source:
            prior = predecessor[node]
            if prior is None:
                raise QuacP1FormalAcquisitionError(
                    "max-flow predecessor drifted"
                )
            prior_node, edge_index = prior
            edge = graph[prior_node][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = prior_node
        achieved += 1
    if achieved != required:
        raise QuacP1FormalAcquisitionError(
            "global nine-slot component capacity is insufficient"
        )

    result: dict[str, tuple[str, str]] = {}
    for component in component_order:
        used = tuple(
            slot
            for slot in availability[component]
            if assignment_edges[(component, slot)].capacity == 0
        )
        if len(used) > 1:
            raise QuacP1FormalAcquisitionError(
                "component capacity-one assignment drifted"
            )
        if used:
            result[component] = used[0]
    if len(result) != required:
        raise QuacP1FormalAcquisitionError(
            "global assignment witness count drifted"
        )
    return result


@dataclass(frozen=True, slots=True)
class SelectedItem:
    """One trusted selected item; source content never enters ``ViewPack``."""

    block: str
    family: str
    work_id: str
    selection_order_sha256: str
    source: EligibleItem = field(repr=False)

    def __post_init__(self) -> None:
        if (
            self.block not in BLOCK_ORDER
            or self.family not in FAMILY_ORDER
            or self.family != self.source.family
            or self.source.official_split != BLOCK_SPLIT[self.block]
            or not isinstance(self.work_id, str)
            or _WORK_ID.fullmatch(self.work_id) is None
        ):
            raise QuacP1FormalAcquisitionError(
                "selected item binding is invalid"
            )
        _require_sha256(
            self.selection_order_sha256,
            "selection order SHA256",
        )


@dataclass(frozen=True, slots=True)
class SelectionPlan:
    """Canonical whole-study assignment with one component per item."""

    secret_commitment: str
    selected: tuple[SelectedItem, ...] = field(repr=False)
    quotas: Mapping[str, int]
    selection_commitment: str

    def __post_init__(self) -> None:
        _require_sha256(self.secret_commitment, "secret commitment")
        _require_sha256(
            self.selection_commitment,
            "selection commitment",
        )
        normalized = _normalize_quotas(self.quotas)
        counts = {
            (block, family): 0
            for block in BLOCK_ORDER
            for family in FAMILY_ORDER
        }
        components: set[str] = set()
        work_ids: set[str] = set()
        for row in self.selected:
            counts[(row.block, row.family)] += 1
            if row.source.component_commitment in components:
                raise QuacP1FormalAcquisitionError(
                    "selection reuses a global component"
                )
            components.add(row.source.component_commitment)
            if row.work_id in work_ids:
                raise QuacP1FormalAcquisitionError(
                    "selection contains duplicate work IDs"
                )
            work_ids.add(row.work_id)
        if any(
            counts[(block, family)] != normalized[block]
            for block in BLOCK_ORDER
            for family in FAMILY_ORDER
        ):
            raise QuacP1FormalAcquisitionError(
                "selection quota counts drifted"
            )
        expected_commitment = stable_hash(
            {
                "rows": [
                    {
                        "block": row.block,
                        "component_commitment": (
                            row.source.component_commitment
                        ),
                        "family": row.family,
                        "item_commitment": row.source.item_commitment,
                        "selection_order_sha256": (
                            row.selection_order_sha256
                        ),
                        "work_id": row.work_id,
                    }
                    for row in self.selected
                ],
                "schema": "quac_p1_private_selection_plan_v1",
                "secret_commitment": self.secret_commitment,
            }
        )
        if expected_commitment != self.selection_commitment:
            raise QuacP1FormalAcquisitionError(
                "selection commitment drifted"
            )

    def rows(self, block: str) -> tuple[SelectedItem, ...]:
        if block not in BLOCK_ORDER:
            raise QuacP1FormalAcquisitionError("block is invalid")
        return tuple(row for row in self.selected if row.block == block)


def select_study(
    source_index: SourceIndex,
    secret: SelectionSecret,
    *,
    quotas: Mapping[str, int] = FORMAL_QUOTAS,
) -> SelectionPlan:
    """Select every block/family jointly, then choose items deterministically."""

    normalized = _normalize_quotas(quotas)
    assignment = _simultaneous_assignment(
        source_index.items,
        secret=secret,
        quotas=normalized,
    )
    by_component: dict[str, list[EligibleItem]] = {}
    for item in source_index.items:
        by_component.setdefault(item.component_commitment, []).append(item)

    selected: list[SelectedItem] = []
    for component, (block, family) in assignment.items():
        candidates = [
            item
            for item in by_component[component]
            if item.official_split == BLOCK_SPLIT[block]
            and item.family == family
        ]
        ordered = sorted(
            candidates,
            key=lambda item: (
                secret.digest(
                    "item-order-v1",
                    {
                        "block": block,
                        "component_commitment": component,
                        "family": family,
                        "item_commitment": item.item_commitment,
                    },
                ),
                item.item_commitment,
            ),
        )
        if not ordered:
            raise QuacP1FormalAcquisitionError(
                "assigned component has no eligible item"
            )
        chosen = ordered[0]
        order_digest = secret.digest(
            "item-order-v1",
            {
                "block": block,
                "component_commitment": component,
                "family": family,
                "item_commitment": chosen.item_commitment,
            },
        )
        work_id = "quac-work-v1-" + secret.digest(
            "work-id-v1",
            {"item_commitment": chosen.item_commitment},
        )
        selected.append(
            SelectedItem(
                block=block,
                family=family,
                work_id=work_id,
                selection_order_sha256=order_digest,
                source=chosen,
            )
        )
    selected_rows = tuple(
        sorted(
            selected,
            key=lambda row: (
                BLOCK_ORDER.index(row.block),
                FAMILY_ORDER.index(row.family),
                row.selection_order_sha256,
                row.work_id,
            ),
        )
    )
    commitment = stable_hash(
        {
            "rows": [
                {
                    "block": row.block,
                    "component_commitment": row.source.component_commitment,
                    "family": row.family,
                    "item_commitment": row.source.item_commitment,
                    "selection_order_sha256": row.selection_order_sha256,
                    "work_id": row.work_id,
                }
                for row in selected_rows
            ],
            "schema": "quac_p1_private_selection_plan_v1",
            "secret_commitment": secret.commitment,
        }
    )
    return SelectionPlan(
        secret_commitment=secret.commitment,
        selected=selected_rows,
        quotas=normalized,
        selection_commitment=commitment,
    )


@dataclass(frozen=True, slots=True)
class ViewRow:
    """The only item information available to label-free action code."""

    work_id: str
    query_text: str
    recent_questions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.work_id, str) or _WORK_ID.fullmatch(self.work_id) is None:
            raise QuacP1FormalAcquisitionError("view work ID is invalid")
        _strict_text(
            self.query_text,
            field_name="view query",
            allow_empty=True,
        )
        if (
            not 1 <= len(self.recent_questions) <= 4
            or self.recent_questions[-1] != self.query_text
            or any(not isinstance(value, str) for value in self.recent_questions)
        ):
            raise QuacP1FormalAcquisitionError(
                "view recent questions are invalid"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "query_text": self.query_text,
            "recent_questions": list(self.recent_questions),
            "work_id": self.work_id,
        }


@dataclass(frozen=True, slots=True)
class ViewPack:
    block: str
    selection_commitment: str
    rows: tuple[ViewRow, ...]

    def __post_init__(self) -> None:
        if self.block not in BLOCK_ORDER:
            raise QuacP1FormalAcquisitionError("view block is invalid")
        _require_sha256(
            self.selection_commitment,
            "view selection commitment",
        )
        work_ids = tuple(row.work_id for row in self.rows)
        if len(set(work_ids)) != len(work_ids) or work_ids != tuple(
            sorted(work_ids)
        ):
            raise QuacP1FormalAcquisitionError(
                "view rows are not canonical and unique"
            )

    def payload(self) -> dict[str, Any]:
        result = {
            "block": self.block,
            "rows": [row.payload() for row in self.rows],
            "schema": VIEW_SCHEMA,
            "selection_commitment": self.selection_commitment,
            "study_id": STUDY_ID,
        }
        assert_view_is_label_free(result)
        return result

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.payload())


@dataclass(frozen=True, slots=True)
class LabelRow:
    work_id: str
    family: str
    previous_turn_orig_answer: tuple[str, ...]
    current_turn_orig_answer: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.work_id, str) or _WORK_ID.fullmatch(self.work_id) is None:
            raise QuacP1FormalAcquisitionError("label work ID is invalid")
        if self.family not in FAMILY_ORDER:
            raise QuacP1FormalAcquisitionError(
                "label family is invalid"
            )
        for role_rows in (
            self.previous_turn_orig_answer,
            self.current_turn_orig_answer,
        ):
            if (
                not role_rows
                or tuple(sorted(set(role_rows))) != role_rows
                or any(_HEX64.fullmatch(value) is None for value in role_rows)
            ):
                raise QuacP1FormalAcquisitionError(
                    "label qrel role is invalid"
                )

    def payload(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "qrel_roles": {
                QREL_ROLE_ORDER[0]: list(
                    self.previous_turn_orig_answer
                ),
                QREL_ROLE_ORDER[1]: list(
                    self.current_turn_orig_answer
                ),
            },
            "work_id": self.work_id,
        }


@dataclass(frozen=True, slots=True)
class LabelPack:
    block: str
    selection_commitment: str
    action_seal_sha256: str
    rows: tuple[LabelRow, ...]

    def __post_init__(self) -> None:
        if self.block not in BLOCK_ORDER:
            raise QuacP1FormalAcquisitionError("label block is invalid")
        _require_sha256(
            self.selection_commitment,
            "label selection commitment",
        )
        _require_sha256(
            self.action_seal_sha256,
            "label action seal SHA256",
        )
        work_ids = tuple(row.work_id for row in self.rows)
        if len(set(work_ids)) != len(work_ids) or work_ids != tuple(
            sorted(work_ids)
        ):
            raise QuacP1FormalAcquisitionError(
                "label rows are not canonical and unique"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "action_seal_sha256": self.action_seal_sha256,
            "block": self.block,
            "rows": [row.payload() for row in self.rows],
            "schema": LABEL_SCHEMA,
            "selection_commitment": self.selection_commitment,
            "study_id": STUDY_ID,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.payload())


def assert_view_is_label_free(payload: Mapping[str, Any]) -> None:
    """Fail if a view contains any frozen forbidden source/label key."""

    def walk(value: object) -> None:
        if isinstance(value, Mapping):
            overlap = set(value).intersection(FORBIDDEN_VIEW_KEYS)
            if overlap:
                raise QuacP1FormalAcquisitionError(
                    "view contains a forbidden source or label field"
                )
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)

    walk(payload)


def _window_unit_id(
    secret: SelectionSecret,
    source: EligibleItem,
    window_ordinal: int,
) -> str:
    windows = source_qualification.canonical_window_spans(source.context)
    if (
        type(window_ordinal) is not int
        or not 0 <= window_ordinal < len(windows)
    ):
        raise QuacP1FormalAcquisitionError(
            "qrel window ordinal is invalid"
        )
    start, end = windows[window_ordinal]
    commitment = stable_hash(
        {
            "context_sha256": source.context_sha256,
            "end": end,
            "schema": "quac_p1_canonical_window_commitment_v1",
            "start": start,
            "window_ordinal": window_ordinal,
            "window_text_sha256": hashlib.sha256(
                source.context[start:end].encode("utf-8")
            ).hexdigest(),
        }
    )
    return secret.digest(
        "window-unit-id-v1",
        {"window_commitment": commitment},
    )


def _label_row(
    selected: SelectedItem,
    secret: SelectionSecret,
) -> LabelRow:
    previous = tuple(
        sorted(
            _window_unit_id(secret, selected.source, ordinal)
            for ordinal in selected.source.previous_answer.window_ordinals
        )
    )
    current = tuple(
        sorted(
            _window_unit_id(secret, selected.source, ordinal)
            for ordinal in selected.source.current_answer.window_ordinals
        )
    )
    return LabelRow(
        work_id=selected.work_id,
        family=selected.family,
        previous_turn_orig_answer=previous,
        current_turn_orig_answer=current,
    )


def _view_pack(plan: SelectionPlan, block: str) -> ViewPack:
    rows = tuple(
        sorted(
            (
                ViewRow(
                    work_id=row.work_id,
                    query_text=row.source.question_text,
                    recent_questions=row.source.recent_questions,
                )
                for row in plan.rows(block)
            ),
            key=lambda row: row.work_id,
        )
    )
    return ViewPack(
        block=block,
        selection_commitment=plan.selection_commitment,
        rows=rows,
    )


def assign_a_form_folds(
    plan: SelectionPlan,
    secret: SelectionSecret,
) -> dict[str, int]:
    """Assign the 192 A_form items to fixed 39/39/38/38/38 folds."""

    rows = plan.rows("A_form")
    if len(rows) != FORMAL_A_FORM_COUNT:
        raise QuacP1FormalAcquisitionError(
            "formal A_form fold assignment requires exactly 192 items"
        )
    ordered = sorted(
        rows,
        key=lambda row: (
            secret.digest(
                "a-form-fold-order-v1",
                {"work_id": row.work_id},
            ),
            row.work_id,
        ),
    )
    result = {
        row.work_id: rank % FOLD_COUNT
        for rank, row in enumerate(ordered)
    }
    sizes = tuple(
        sum(fold == expected for fold in result.values())
        for expected in range(FOLD_COUNT)
    )
    if sizes != FORMAL_FOLD_SIZES:
        raise QuacP1FormalAcquisitionError(
            "formal five-fold balance drifted"
        )
    return result


@dataclass(frozen=True, slots=True)
class PromotionProof:
    """Exact A_hold promotion decision that may authorize M_search."""

    selection_commitment: str
    a_hold_score_receipt_sha256: str
    aggregate_e1_minus_e0: int
    p_numerator: int
    p_denominator: int
    promoted: bool
    self_sha256: str

    @classmethod
    def create(
        cls,
        *,
        selection_commitment: str,
        a_hold_score_receipt_sha256: str,
        aggregate_e1_minus_e0: int,
        p_numerator: int,
        p_denominator: int,
        promoted: bool,
    ) -> "PromotionProof":
        body = {
            "a_hold_score_receipt_sha256": a_hold_score_receipt_sha256,
            "aggregate_e1_minus_e0": aggregate_e1_minus_e0,
            "p_denominator": p_denominator,
            "p_numerator": p_numerator,
            "promoted": promoted,
            "schema": PROMOTION_PROOF_SCHEMA,
            "selection_commitment": selection_commitment,
            "study_id": STUDY_ID,
        }
        return cls(
            selection_commitment=selection_commitment,
            a_hold_score_receipt_sha256=a_hold_score_receipt_sha256,
            aggregate_e1_minus_e0=aggregate_e1_minus_e0,
            p_numerator=p_numerator,
            p_denominator=p_denominator,
            promoted=promoted,
            self_sha256=stable_hash(body),
        )

    def body(self) -> dict[str, Any]:
        return {
            "a_hold_score_receipt_sha256": (
                self.a_hold_score_receipt_sha256
            ),
            "aggregate_e1_minus_e0": self.aggregate_e1_minus_e0,
            "p_denominator": self.p_denominator,
            "p_numerator": self.p_numerator,
            "promoted": self.promoted,
            "schema": PROMOTION_PROOF_SCHEMA,
            "selection_commitment": self.selection_commitment,
            "study_id": STUDY_ID,
        }

    def verify_for(self, selection_commitment: str) -> None:
        _require_sha256(
            self.selection_commitment,
            "promotion selection commitment",
        )
        _require_sha256(
            self.a_hold_score_receipt_sha256,
            "A_hold score receipt SHA256",
        )
        _require_sha256(
            self.self_sha256,
            "promotion proof self SHA256",
        )
        if (
            self.selection_commitment != selection_commitment
            or stable_hash(self.body()) != self.self_sha256
            or type(self.aggregate_e1_minus_e0) is not int
            or type(self.p_numerator) is not int
            or type(self.p_denominator) is not int
            or self.aggregate_e1_minus_e0 <= 0
            or self.p_numerator < 0
            or self.p_denominator <= 0
            or self.p_numerator * 10 > self.p_denominator
            or self.promoted is not True
        ):
            raise QuacP1FormalAcquisitionError(
                "A_hold proof does not satisfy frozen promotion"
            )


@dataclass(frozen=True, slots=True)
class MSearchCapability:
    reservation_commitment: str
    selection_commitment: str
    promotion_proof_sha256: str
    capability_mac: str

    def payload(self) -> dict[str, Any]:
        return {
            "capability_mac": self.capability_mac,
            "promotion_proof_sha256": self.promotion_proof_sha256,
            "reservation_commitment": self.reservation_commitment,
            "schema": M_CAPABILITY_SCHEMA,
            "selection_commitment": self.selection_commitment,
            "study_id": STUDY_ID,
        }


@dataclass(frozen=True, slots=True)
class LateLabelCapability:
    block: str
    selection_commitment: str
    action_seal_sha256: str
    capability_mac: str

    def payload(self) -> dict[str, Any]:
        return {
            "action_seal_sha256": self.action_seal_sha256,
            "block": self.block,
            "capability_mac": self.capability_mac,
            "schema": LABEL_CAPABILITY_SCHEMA,
            "selection_commitment": self.selection_commitment,
            "study_id": STUDY_ID,
        }


@dataclass(frozen=True, slots=True)
class AFormModelSeal:
    """Broker-issued fit seal bound to the exact A_form action and labels."""

    selection_commitment: str
    action_seal_sha256: str
    label_pack_sha256: str
    model_parameter_sha256: str
    seal_mac: str

    def payload(self) -> dict[str, Any]:
        return {
            "action_seal_sha256": self.action_seal_sha256,
            "label_pack_sha256": self.label_pack_sha256,
            "model_parameter_sha256": self.model_parameter_sha256,
            "schema": A_FORM_MODEL_SEAL_SCHEMA,
            "seal_mac": self.seal_mac,
            "selection_commitment": self.selection_commitment,
            "study_id": STUDY_ID,
        }


@dataclass(frozen=True, slots=True)
class MaterializedMSearch:
    """One successful M opening without any selected/source row surface."""

    view_pack: ViewPack


@dataclass(frozen=True, slots=True)
class RuntimeMaterialDocument:
    """One label-free corpus window for the runtime bridge."""

    unit_id: str
    context_id: str
    title: str
    section_title: str
    context_window_ordinal: int
    text: str

    def __post_init__(self) -> None:
        _require_sha256(self.unit_id, "runtime material unit ID")
        _require_sha256(self.context_id, "runtime material context ID")
        _strict_text(self.title, field_name="runtime title", allow_empty=True)
        _strict_text(
            self.section_title,
            field_name="runtime section title",
            allow_empty=True,
        )
        _strict_text(
            self.text,
            field_name="runtime window text",
            allow_empty=False,
        )
        if (
            type(self.context_window_ordinal) is not int
            or self.context_window_ordinal < 0
        ):
            raise QuacP1FormalAcquisitionError(
                "runtime window ordinal is invalid"
            )


@dataclass(frozen=True, slots=True)
class RuntimeMaterialQuery:
    """One opaque query with current, previous-1, ..., previous-3 order."""

    query_id: str
    question_turns: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.query_id, "runtime material query ID")
        if not 1 <= len(self.question_turns) <= 4:
            raise QuacP1FormalAcquisitionError(
                "runtime material question count is invalid"
            )
        for question in self.question_turns:
            _strict_text(
                question,
                field_name="runtime question",
                allow_empty=False,
            )


@dataclass(frozen=True, slots=True)
class RuntimeMaterial:
    """Separated block corpus and queries; no native query-context mapping."""

    block: str
    block_id: str
    documents: tuple[RuntimeMaterialDocument, ...]
    queries: tuple[RuntimeMaterialQuery, ...]

    def __post_init__(self) -> None:
        if self.block not in BLOCK_ORDER:
            raise QuacP1FormalAcquisitionError(
                "runtime material block is invalid"
            )
        _require_sha256(self.block_id, "runtime material block ID")
        document_ids = tuple(row.unit_id for row in self.documents)
        query_ids = tuple(row.query_id for row in self.queries)
        if (
            len(self.documents) < 5
            or not self.queries
            or document_ids != tuple(sorted(document_ids))
            or len(set(document_ids)) != len(document_ids)
            or query_ids != tuple(sorted(query_ids))
            or len(set(query_ids)) != len(query_ids)
        ):
            raise QuacP1FormalAcquisitionError(
                "runtime material registries are invalid"
            )


def _durable_action_identity(
    path: Path,
    *,
    expected_payload: Mapping[str, Any],
) -> tuple[Path, str, int, int, int]:
    """Verify one direct mode-0400 canonical action barrier."""

    if not isinstance(path, Path) or not path.is_absolute():
        raise QuacP1FormalAcquisitionError(
            "durable action barrier path must be absolute"
        )
    if not hasattr(os, "O_NOFOLLOW"):
        raise QuacP1FormalAcquisitionError(
            "durable action barriers require O_NOFOLLOW"
        )
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        )
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        pathname = path.lstat()
    except OSError as exc:
        raise QuacP1FormalAcquisitionError(
            "durable action barrier is unavailable"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    raw = b"".join(chunks)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) != 0o400
        or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        or not stat.S_ISREG(pathname.st_mode)
        or pathname.st_nlink != 1
        or pathname.st_uid != os.getuid()
        or stat.S_IMODE(pathname.st_mode) != 0o400
        or (
            pathname.st_dev,
            pathname.st_ino,
            pathname.st_size,
            pathname.st_mtime_ns,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
    ):
        raise QuacP1FormalAcquisitionError(
            "durable action barrier identity drifted"
        )
    expected = canonical_bytes(dict(expected_payload))
    if raw not in {expected, expected + b"\n"}:
        raise QuacP1FormalAcquisitionError(
            "durable action barrier payload drifted"
        )
    return (
        path,
        hashlib.sha256(raw).hexdigest(),
        before.st_dev,
        before.st_ino,
        before.st_mtime_ns,
    )


def _reverify_durable_action_identity(
    identity: tuple[Path, str, int, int, int],
) -> None:
    path, expected_sha256, device, inode, mtime_ns = identity
    if not hasattr(os, "O_NOFOLLOW"):
        raise QuacP1FormalAcquisitionError(
            "durable action barriers require O_NOFOLLOW"
        )
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        )
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        metadata = os.fstat(descriptor)
        pathname = path.lstat()
    except OSError as exc:
        raise QuacP1FormalAcquisitionError(
            "registered action barrier disappeared"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    raw = b"".join(chunks)
    if (
        not stat.S_ISREG(before.st_mode)
        or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        != (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        )
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o400
        or metadata.st_dev != device
        or metadata.st_ino != inode
        or metadata.st_mtime_ns != mtime_ns
        or not stat.S_ISREG(pathname.st_mode)
        or pathname.st_nlink != 1
        or pathname.st_uid != os.getuid()
        or stat.S_IMODE(pathname.st_mode) != 0o400
        or pathname.st_dev != metadata.st_dev
        or pathname.st_ino != metadata.st_ino
        or pathname.st_size != metadata.st_size
        or pathname.st_mtime_ns != metadata.st_mtime_ns
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise QuacP1FormalAcquisitionError(
            "registered action barrier changed before label opening"
        )


def _validate_action_barrier_payload(
    block: str,
    payload: Mapping[str, Any],
    *,
    expected_block_id: str,
    expected_query_ids: frozenset[str],
    corpus_unit_ids: frozenset[str],
) -> None:
    if not isinstance(payload, Mapping):
        raise QuacP1FormalAcquisitionError(
            "action barrier payload must be an object"
        )
    expected_schema = (
        "quac_p1_runtime_v1_private_action_pack_v1"
        if block == "A_form"
        else (
            "quac_p1_formal_controller_v1_"
            "sealed_stage_actions_v1"
        )
    )
    if payload.get("schema") != expected_schema:
        raise QuacP1FormalAcquisitionError(
            "action barrier schema does not match the block"
        )
    if block == "A_form":
        if (
            set(payload) != {"block_id", "rows", "schema"}
            or payload.get("block_id") != expected_block_id
        ):
            raise QuacP1FormalAcquisitionError(
                "A_form action barrier block binding drifted"
            )
    else:
        if (
            set(payload)
            != {
                "arms",
                "block",
                "corpus_unit_ids_sha256",
                "label_or_family_present",
                "rows",
                "schema",
            }
            or payload.get("block") != block
            or payload.get("arms")
            != ["E0", "E1", "RAW", "official_HippoRAG"]
            or payload.get("label_or_family_present") is not False
            or payload.get("corpus_unit_ids_sha256")
            != stable_hash(sorted(corpus_unit_ids))
        ):
            raise QuacP1FormalAcquisitionError(
                "measurement action barrier block/corpus binding drifted"
            )
    assert_view_is_label_free(payload)
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise QuacP1FormalAcquisitionError(
            "action barrier rows are invalid"
        )
    id_field = "query_id" if block == "A_form" else "item_id"
    observed_ids = []
    for row in raw_rows:
        if not isinstance(row, Mapping):
            raise QuacP1FormalAcquisitionError(
                "action barrier row is invalid"
            )
        item_id = row.get(id_field)
        if not isinstance(item_id, str):
            raise QuacP1FormalAcquisitionError(
                "action barrier item ID is invalid"
            )
        observed_ids.append(item_id)
        if block == "A_form":
            if set(row) != {"action", "action_sha256", "query_id"}:
                raise QuacP1FormalAcquisitionError(
                    "A_form action row key set drifted"
                )
            action_payload = row.get("action")
            if not isinstance(action_payload, Mapping):
                raise QuacP1FormalAcquisitionError(
                    "A_form action graph payload is invalid"
                )
            graph = action_payload.get("graph")
            if not isinstance(graph, Mapping):
                raise QuacP1FormalAcquisitionError(
                    "A_form action graph is absent"
                )
            units = graph.get("units")
            raw_top5 = action_payload.get("raw_top5")
            if (
                not isinstance(units, list)
                or not isinstance(raw_top5, list)
            ):
                raise QuacP1FormalAcquisitionError(
                    "A_form graph units/RAW are invalid"
                )
            graph_ids = {
                unit.get("unit_id")
                for unit in units
                if isinstance(unit, Mapping)
            }
            if (
                row.get("action_sha256") != stable_hash(action_payload)
                or len(graph_ids) != len(units)
                or not graph_ids.issubset(corpus_unit_ids)
                or len(raw_top5) != 5
                or len(set(raw_top5)) != 5
                or not set(raw_top5).issubset(graph_ids)
            ):
                raise QuacP1FormalAcquisitionError(
                    "A_form graph/RAW detaches from the block corpus"
                )
        else:
            if set(row) != {
                "E0",
                "E1",
                "RAW",
                "item_id",
                "official_HippoRAG",
            }:
                raise QuacP1FormalAcquisitionError(
                    "measurement action row key set drifted"
                )
            for arm in ("E0", "E1", "RAW", "official_HippoRAG"):
                selected = row.get(arm)
                if (
                    not isinstance(selected, list)
                    or len(selected) != 5
                    or len(set(selected)) != 5
                    or not set(selected).issubset(corpus_unit_ids)
                ):
                    raise QuacP1FormalAcquisitionError(
                        "measurement action detaches from the block corpus"
                    )
    if (
        len(set(observed_ids)) != len(observed_ids)
        or frozenset(observed_ids) != expected_query_ids
    ):
        raise QuacP1FormalAcquisitionError(
            "action barrier has missing, extra, or duplicate cohort rows"
        )


class TrustedAcquisitionBroker:
    """In-memory capability broker around one immutable selection epoch."""

    def __init__(
        self,
        *,
        secret: SelectionSecret,
        plan: SelectionPlan,
    ) -> None:
        if secret.commitment != plan.secret_commitment:
            raise QuacP1FormalAcquisitionError(
                "broker secret does not bind the selection plan"
            )
        self._secret = secret
        self._plan = plan
        m_rows = plan.rows("M_search")
        self._reservation_commitment = secret.digest(
            "m-reservation-v1",
            {
                "item_commitments": [
                    row.source.item_commitment for row in m_rows
                ],
                "selection_commitment": plan.selection_commitment,
            },
        )
        self._m_materialization_count = 0
        self._m_materialized_path_count = 0
        self._issued_m_capability: str | None = None
        self._consumed_m_capability: str | None = None
        self._issued_label_blocks: set[str] = set()
        self._opened_label_blocks: set[str] = set()
        self._opened_label_pack_sha256: dict[str, str] = {}
        self._consumed_label_capabilities: set[str] = set()
        self._action_barriers: dict[
            str, tuple[Path, str, int, int, int]
        ] = {}
        self._measurement_actions: dict[str, object] = {}
        self._registered_a_hold_score_sha256: str | None = None
        self._issued_a_form_model_seal: AFormModelSeal | None = None
        self._registered_a_form_model_seal: tuple[
            Path, str, int, int, int
        ] | None = None
        self._m_materialized_registry: tuple[
            Path, str, int, int, int
        ] | None = None

    @property
    def selection_commitment(self) -> str:
        return self._plan.selection_commitment

    def safe_selection_receipt(self) -> dict[str, Any]:
        a_form_count = len(self._plan.rows("A_form"))
        return {
            "component_capacity_per_whole_study": 1,
            "fold_sizes": (
                list(FORMAL_FOLD_SIZES)
                if a_form_count == FORMAL_A_FORM_COUNT
                else None
            ),
            "global_simultaneous_max_flow": True,
            "item_counts": {
                block: len(self._plan.rows(block))
                for block in BLOCK_ORDER
            },
            "per_family_quotas": {
                block: {
                    family: self._plan.quotas[block]
                    for family in FAMILY_ORDER
                }
                for block in BLOCK_ORDER
            },
            "schema": "quac_p1_safe_selection_receipt_v1",
            "secret_commitment": self._plan.secret_commitment,
            "selection_commitment": self._plan.selection_commitment,
            "study_id": STUDY_ID,
        }

    def m_reservation_receipt(self) -> dict[str, Any]:
        """Return the only M_search surface allowed before promotion."""

        return {
            "block": "M_search",
            "item_count": len(self._plan.rows("M_search")),
            "materialization_count": self._m_materialization_count,
            "materialized_path_count": self._m_materialized_path_count,
            "opaque_reservation_commitment": self._reservation_commitment,
            "schema": RESERVATION_SCHEMA,
            "selection_commitment": self._plan.selection_commitment,
            "study_id": STUDY_ID,
        }

    def private_rows(self, block: str) -> tuple[SelectedItem, ...]:
        """Selected/source rows are intentionally not a public broker surface."""

        del block
        raise QuacP1FormalAcquisitionError(
            "private selected rows are sealed inside the broker"
        )

    def _require_registered_a_form_model_seal(self) -> None:
        if self._registered_a_form_model_seal is None:
            raise QuacP1FormalAcquisitionError(
                "A_hold is sealed until the exact A_form model seal is "
                "durably registered"
            )
        _reverify_durable_action_identity(
            self._registered_a_form_model_seal
        )

    def view_pack(self, block: str) -> ViewPack:
        if block == "M_search" and self._m_materialization_count != 1:
            raise QuacP1FormalAcquisitionError(
                "M_search view is not materialized"
            )
        if block in {"A_hold", "M_search"}:
            self._require_registered_a_form_model_seal()
        return _view_pack(self._plan, block)

    def runtime_material(self, block: str) -> RuntimeMaterial:
        """Project one action block without labels or native query linkage."""

        if block not in BLOCK_ORDER:
            raise QuacP1FormalAcquisitionError("block is invalid")
        if block == "M_search" and self._m_materialization_count != 1:
            raise QuacP1FormalAcquisitionError(
                "M_search runtime material is not materialized"
            )
        if block in {"A_hold", "M_search"}:
            self._require_registered_a_form_model_seal()
        selected = self._plan.rows(block)
        documents_by_id: dict[str, RuntimeMaterialDocument] = {}
        for row in selected:
            source = row.source
            context_id = self._secret.digest(
                "context-id-v1",
                {"context_sha256": source.context_sha256},
            )
            windows = source_qualification.canonical_window_spans(
                source.context
            )
            for ordinal, (start, end) in enumerate(windows):
                document = RuntimeMaterialDocument(
                    unit_id=_window_unit_id(
                        self._secret,
                        source,
                        ordinal,
                    ),
                    context_id=context_id,
                    title=source.title,
                    section_title=source.section_title,
                    context_window_ordinal=ordinal,
                    text=source.context[start:end],
                )
                prior = documents_by_id.setdefault(
                    document.unit_id,
                    document,
                )
                if prior != document:
                    raise QuacP1FormalAcquisitionError(
                        "runtime unit ID collision drifted"
                    )
        queries = tuple(
            sorted(
                (
                    RuntimeMaterialQuery(
                        query_id=row.work_id.removeprefix(
                            "quac-work-v1-"
                        ),
                        question_turns=tuple(
                            reversed(row.source.recent_questions)
                        ),
                    )
                    for row in selected
                ),
                key=lambda row: row.query_id,
            )
        )
        return RuntimeMaterial(
            block=block,
            block_id=self._secret.digest(
                "block-id-v1",
                {
                    "block": block,
                    "selection_commitment": (
                        self._plan.selection_commitment
                    ),
                },
            ),
            documents=tuple(
                sorted(
                    documents_by_id.values(),
                    key=lambda row: row.unit_id,
                )
            ),
            queries=queries,
        )

    def a_form_folds(self) -> dict[str, int]:
        return {
            work_id.removeprefix("quac-work-v1-"): fold
            for work_id, fold in assign_a_form_folds(
                self._plan,
                self._secret,
            ).items()
        }

    def issue_a_form_model_seal(
        self,
        *,
        model_parameter_sha256: str,
    ) -> AFormModelSeal:
        """Issue one fit seal only after the exact A_form labels opened."""

        model_parameter_sha256 = _require_sha256(
            model_parameter_sha256,
            "A_form model parameter SHA256",
        )
        if self._issued_a_form_model_seal is not None:
            raise QuacP1FormalAcquisitionError(
                "A_form model seal has already been issued"
            )
        if (
            "A_form" not in self._opened_label_blocks
            or "A_form" not in self._action_barriers
            or "A_form" not in self._opened_label_pack_sha256
        ):
            raise QuacP1FormalAcquisitionError(
                "A_form model seal requires the exact opened A_form "
                "action/label epoch"
            )
        barrier = self._action_barriers["A_form"]
        _reverify_durable_action_identity(barrier)
        body = {
            "action_seal_sha256": barrier[1],
            "label_pack_sha256": self._opened_label_pack_sha256[
                "A_form"
            ],
            "model_parameter_sha256": model_parameter_sha256,
            "selection_commitment": self._plan.selection_commitment,
        }
        seal = AFormModelSeal(
            selection_commitment=self._plan.selection_commitment,
            action_seal_sha256=barrier[1],
            label_pack_sha256=self._opened_label_pack_sha256[
                "A_form"
            ],
            model_parameter_sha256=model_parameter_sha256,
            seal_mac=self._secret.digest(
                "a-form-model-seal-v1",
                body,
            ),
        )
        self._issued_a_form_model_seal = seal
        return seal

    def register_durable_a_form_model_seal(
        self,
        *,
        seal: AFormModelSeal,
        seal_path: Path,
    ) -> None:
        """Advance to A_hold only after the broker-issued seal is durable."""

        if type(seal) is not AFormModelSeal:
            raise QuacP1FormalAcquisitionError(
                "A_form model seal type is not exact"
            )
        if self._registered_a_form_model_seal is not None:
            raise QuacP1FormalAcquisitionError(
                "A_form model seal has already been registered"
            )
        issued = self._issued_a_form_model_seal
        if issued is None or seal != issued:
            raise QuacP1FormalAcquisitionError(
                "A_form model seal is forged or detached"
            )
        for value, name in (
            (seal.selection_commitment, "model-seal selection"),
            (seal.action_seal_sha256, "model-seal action"),
            (seal.label_pack_sha256, "model-seal label pack"),
            (seal.model_parameter_sha256, "model parameter"),
            (seal.seal_mac, "model-seal MAC"),
        ):
            _require_sha256(value, name)
        body = {
            "action_seal_sha256": seal.action_seal_sha256,
            "label_pack_sha256": seal.label_pack_sha256,
            "model_parameter_sha256": seal.model_parameter_sha256,
            "selection_commitment": seal.selection_commitment,
        }
        expected_mac = self._secret.digest(
            "a-form-model-seal-v1",
            body,
        )
        if (
            seal.selection_commitment
            != self._plan.selection_commitment
            or seal.action_seal_sha256
            != self._action_barriers["A_form"][1]
            or seal.label_pack_sha256
            != self._opened_label_pack_sha256["A_form"]
            or not hmac.compare_digest(seal.seal_mac, expected_mac)
        ):
            raise QuacP1FormalAcquisitionError(
                "A_form model seal is forged or detached"
            )
        self._registered_a_form_model_seal = (
            _durable_action_identity(
                seal_path,
                expected_payload=seal.payload(),
            )
        )

    def issue_m_search_capability(
        self,
        proof: PromotionProof,
    ) -> MSearchCapability:
        del proof
        raise QuacP1FormalAcquisitionError(
            "external promotion proofs are not accepted"
        )

    def authorize_m_search_from_stage_score(
        self,
        *,
        stage_score: object,
        score_receipt_path: Path,
    ) -> MSearchCapability:
        """Authorize M only from a durable, actual controller StageScore."""

        # Lazy import keeps decoded-source acquisition independent of torch
        # until the formal controller has already produced its score object.
        from assumption_agent.benchmarks import (  # noqa: PLC0415
            quac_p1_formal_controller_v1 as formal_controller,
        )

        if self._m_materialization_count != 0:
            raise QuacP1FormalAcquisitionError(
                "M_search has already been materialized"
            )
        if self._issued_m_capability is not None:
            raise QuacP1FormalAcquisitionError(
                "M_search capability has already been issued"
            )
        self._require_registered_a_form_model_seal()
        if (
            "A_hold" not in self._action_barriers
            or "A_hold" not in self._opened_label_blocks
            or type(stage_score) is not formal_controller.StageScore
        ):
            raise QuacP1FormalAcquisitionError(
                "M_search requires the registered A_hold barrier, labels, "
                "and an actual promoted controller StageScore"
            )
        try:
            sealed_actions = self._measurement_actions["A_hold"]
        except KeyError as exc:
            raise QuacP1FormalAcquisitionError(
                "registered A_hold controller action barrier is absent"
            ) from exc
        internal_labels = tuple(
            sorted(
                (
                    formal_controller.LateLabelRow(
                        item_id=row.work_id.removeprefix(
                            "quac-work-v1-"
                        ),
                        family=row.family,
                        previous_qrel=(
                            _label_row(
                                row,
                                self._secret,
                            ).previous_turn_orig_answer
                        ),
                        current_qrel=(
                            _label_row(
                                row,
                                self._secret,
                            ).current_turn_orig_answer
                        ),
                    )
                    for row in self._plan.rows("A_hold")
                ),
                key=lambda row: row.item_id,
            )
        )
        recomputed_score = formal_controller.score_sealed_stage(
            sealed_actions,
            internal_labels,
            block_corpus_unit_ids=tuple(
                row.unit_id
                for row in self.runtime_material("A_hold").documents
            ),
        )
        if (
            stage_score != recomputed_score
            or recomputed_score.block != "A_hold"
            or recomputed_score.promotion is not True
        ):
            raise QuacP1FormalAcquisitionError(
                "A_hold StageScore is not the exact registered action/label "
                "controller result"
            )
        if self._registered_a_hold_score_sha256 is not None:
            raise QuacP1FormalAcquisitionError(
                "A_hold score authorization has already been registered"
            )
        score_identity = _durable_action_identity(
            score_receipt_path,
            expected_payload=recomputed_score.safe_payload(),
        )
        score_file_sha256 = score_identity[1]
        comparison = recomputed_score.comparison("E0")
        proof = PromotionProof.create(
            selection_commitment=self._plan.selection_commitment,
            a_hold_score_receipt_sha256=score_file_sha256,
            aggregate_e1_minus_e0=comparison.net,
            p_numerator=comparison.exact.numerator,
            p_denominator=comparison.exact.denominator,
            promoted=True,
        )
        proof.verify_for(self._plan.selection_commitment)
        body = {
            "promotion_proof_sha256": proof.self_sha256,
            "reservation_commitment": self._reservation_commitment,
            "selection_commitment": self._plan.selection_commitment,
        }
        capability = MSearchCapability(
            reservation_commitment=self._reservation_commitment,
            selection_commitment=self._plan.selection_commitment,
            promotion_proof_sha256=proof.self_sha256,
            capability_mac=self._secret.digest(
                "m-capability-v1",
                body,
            ),
        )
        self._issued_m_capability = capability.capability_mac
        self._registered_a_hold_score_sha256 = score_file_sha256
        return capability

    def materialize_m_search(
        self,
        capability: MSearchCapability,
    ) -> MaterializedMSearch:
        if type(capability) is not MSearchCapability:
            raise QuacP1FormalAcquisitionError(
                "M_search capability type is not exact"
            )
        if self._m_materialization_count != 0:
            raise QuacP1FormalAcquisitionError(
                "M_search capability replay is forbidden"
            )
        for value, name in (
            (
                capability.reservation_commitment,
                "M reservation commitment",
            ),
            (
                capability.selection_commitment,
                "M selection commitment",
            ),
            (
                capability.promotion_proof_sha256,
                "promotion proof SHA256",
            ),
            (capability.capability_mac, "M capability MAC"),
        ):
            _require_sha256(value, name)
        body = {
            "promotion_proof_sha256": (
                capability.promotion_proof_sha256
            ),
            "reservation_commitment": (
                capability.reservation_commitment
            ),
            "selection_commitment": capability.selection_commitment,
        }
        expected = self._secret.digest("m-capability-v1", body)
        if (
            capability.reservation_commitment
            != self._reservation_commitment
            or capability.selection_commitment
            != self._plan.selection_commitment
            or capability.capability_mac != self._issued_m_capability
            or not hmac.compare_digest(
                capability.capability_mac,
                expected,
            )
        ):
            raise QuacP1FormalAcquisitionError(
                "M_search capability is forged or detached"
            )
        if capability.capability_mac == self._consumed_m_capability:
            raise QuacP1FormalAcquisitionError(
                "M_search capability replay is forbidden"
            )
        self._consumed_m_capability = capability.capability_mac
        self._m_materialization_count = 1
        return MaterializedMSearch(
            view_pack=_view_pack(self._plan, "M_search"),
        )

    def m_search_materialized_registry_payload(self) -> dict[str, Any]:
        """Return the exact opaque registry that must be durably archived."""

        if self._m_materialization_count != 1:
            raise QuacP1FormalAcquisitionError(
                "M_search registry cannot precede materialization"
            )
        material = self.runtime_material("M_search")
        return {
            "block": "M_search",
            "block_id": material.block_id,
            "corpus_unit_ids_sha256": stable_hash(
                [row.unit_id for row in material.documents]
            ),
            "item_count": len(material.queries),
            "query_ids_sha256": stable_hash(
                [row.query_id for row in material.queries]
            ),
            "schema": M_MATERIALIZED_REGISTRY_SCHEMA,
            "selection_commitment": self._plan.selection_commitment,
            "study_id": STUDY_ID,
            "view_pack_sha256": stable_hash(
                _view_pack(self._plan, "M_search").payload()
            ),
        }

    def register_durable_m_search_materialized_registry(
        self,
        *,
        registry_path: Path,
        expected_payload: Mapping[str, Any],
    ) -> None:
        """Count one M path only after its exact registry is durable."""

        if (
            self._m_materialization_count != 1
            or self._m_materialized_path_count != 0
            or self._m_materialized_registry is not None
        ):
            raise QuacP1FormalAcquisitionError(
                "M_search materialized registry is invalid or replayed"
            )
        exact = self.m_search_materialized_registry_payload()
        if dict(expected_payload) != exact:
            raise QuacP1FormalAcquisitionError(
                "M_search materialized registry drifted"
            )
        self._m_materialized_registry = _durable_action_identity(
            registry_path,
            expected_payload=exact,
        )
        self._m_materialized_path_count = 1

    def record_m_search_materialized_paths_once(self, count: int) -> None:
        del count
        raise QuacP1FormalAcquisitionError(
            "arbitrary M_search materialized-path counts are not accepted"
        )

    def issue_late_label_capability(
        self,
        *,
        block: str,
        action_seal_sha256: str,
    ) -> LateLabelCapability:
        del block, action_seal_sha256
        raise QuacP1FormalAcquisitionError(
            "arbitrary action-seal hashes are not accepted"
        )

    def register_durable_action_barrier(
        self,
        *,
        block: str,
        action_path: Path,
        expected_payload: Mapping[str, Any],
    ) -> LateLabelCapability:
        """Register and authenticate one exact durable label-free barrier."""

        if block not in BLOCK_ORDER:
            raise QuacP1FormalAcquisitionError("label block is invalid")
        if block == "M_search" and self._m_materialization_count != 1:
            raise QuacP1FormalAcquisitionError(
                "M_search labels cannot precede promotion"
            )
        if block == "M_search" and self._m_materialized_registry is None:
            raise QuacP1FormalAcquisitionError(
                "M_search actions require the durable materialized registry"
            )
        if block in self._issued_label_blocks:
            raise QuacP1FormalAcquisitionError(
                "late-label capability has already been issued"
            )
        material = self.runtime_material(block)
        _validate_action_barrier_payload(
            block,
            expected_payload,
            expected_block_id=material.block_id,
            expected_query_ids=frozenset(
                row.query_id for row in material.queries
            ),
            corpus_unit_ids=frozenset(
                row.unit_id for row in material.documents
            ),
        )
        identity = _durable_action_identity(
            action_path,
            expected_payload=expected_payload,
        )
        if block != "A_form":
            from assumption_agent.benchmarks import (  # noqa: PLC0415
                quac_p1_formal_controller_v1 as formal_controller,
            )

            raw_rows = expected_payload.get("rows")
            if not isinstance(raw_rows, list):
                raise QuacP1FormalAcquisitionError(
                    "measurement action rows are invalid"
                )
            try:
                native_actions = formal_controller.SealedStageActions(
                    block=block,
                    corpus_unit_ids_sha256=expected_payload[
                        "corpus_unit_ids_sha256"
                    ],
                    rows=tuple(
                        sorted(
                            (
                                formal_controller.ActionRow(
                                    item_id=row["item_id"],
                                    E0=tuple(row["E0"]),
                                    E1=tuple(row["E1"]),
                                    RAW=tuple(row["RAW"]),
                                    official_HippoRAG=tuple(
                                        row["official_HippoRAG"]
                                    ),
                                )
                                for row in raw_rows
                            ),
                            key=lambda row: row.item_id,
                        )
                    ),
                )
            except (KeyError, TypeError) as exc:
                raise QuacP1FormalAcquisitionError(
                    "measurement action barrier cannot be reconstructed"
                ) from exc
            if native_actions.payload() != dict(expected_payload):
                raise QuacP1FormalAcquisitionError(
                    "measurement action barrier semantic payload drifted"
                )
            self._measurement_actions[block] = native_actions
        action_seal_sha256 = identity[1]
        body = {
            "action_seal_sha256": action_seal_sha256,
            "block": block,
            "selection_commitment": self._plan.selection_commitment,
        }
        capability = LateLabelCapability(
            block=block,
            selection_commitment=self._plan.selection_commitment,
            action_seal_sha256=action_seal_sha256,
            capability_mac=self._secret.digest(
                "late-label-capability-v1",
                body,
            ),
        )
        self._action_barriers[block] = identity
        self._issued_label_blocks.add(block)
        return capability

    def open_late_labels(
        self,
        capability: LateLabelCapability,
    ) -> LabelPack:
        if type(capability) is not LateLabelCapability:
            raise QuacP1FormalAcquisitionError(
                "late-label capability type is not exact"
            )
        if capability.block not in BLOCK_ORDER:
            raise QuacP1FormalAcquisitionError("label block is invalid")
        if capability.block in {"A_hold", "M_search"}:
            self._require_registered_a_form_model_seal()
        if (
            capability.block == "M_search"
            and self._m_materialization_count != 1
        ):
            raise QuacP1FormalAcquisitionError(
                "M_search labels cannot precede promotion"
            )
        for value, name in (
            (
                capability.selection_commitment,
                "label selection commitment",
            ),
            (capability.action_seal_sha256, "action seal SHA256"),
            (capability.capability_mac, "label capability MAC"),
        ):
            _require_sha256(value, name)
        body = {
            "action_seal_sha256": capability.action_seal_sha256,
            "block": capability.block,
            "selection_commitment": capability.selection_commitment,
        }
        expected = self._secret.digest(
            "late-label-capability-v1",
            body,
        )
        if (
            capability.selection_commitment
            != self._plan.selection_commitment
            or capability.block not in self._issued_label_blocks
            or not hmac.compare_digest(
                capability.capability_mac,
                expected,
            )
        ):
            raise QuacP1FormalAcquisitionError(
                "late-label capability is forged or detached"
            )
        if (
            capability.block in self._opened_label_blocks
            or capability.capability_mac
            in self._consumed_label_capabilities
        ):
            raise QuacP1FormalAcquisitionError(
                "late-label capability replay is forbidden"
            )
        try:
            barrier = self._action_barriers[capability.block]
        except KeyError as exc:
            raise QuacP1FormalAcquisitionError(
                "late-label action barrier was not registered"
            ) from exc
        if barrier[1] != capability.action_seal_sha256:
            raise QuacP1FormalAcquisitionError(
                "late-label capability detached from its barrier"
            )
        _reverify_durable_action_identity(barrier)
        rows = tuple(
            sorted(
                (
                    _label_row(row, self._secret)
                    for row in self._plan.rows(capability.block)
                ),
                key=lambda row: row.work_id,
            )
        )
        pack = LabelPack(
            block=capability.block,
            selection_commitment=self._plan.selection_commitment,
            action_seal_sha256=capability.action_seal_sha256,
            rows=rows,
        )
        self._opened_label_blocks.add(capability.block)
        self._opened_label_pack_sha256[capability.block] = stable_hash(
            pack.payload()
        )
        self._consumed_label_capabilities.add(
            capability.capability_mac
        )
        return pack


def acquire_decoded_sources_once(
    train_obj: object,
    dev_obj: object,
    secret: bytes | SelectionSecret,
    *,
    quotas: Mapping[str, int] = FORMAL_QUOTAS,
) -> TrustedAcquisitionBroker:
    """Build one trusted in-memory acquisition epoch from decoded sources.

    ``secret`` is consumed as the sole whole-study HMAC key.  This function
    never creates, rotates, writes, or reads a secret and never performs file
    I/O.  The formal caller must generate and persist exactly one 32-byte
    secret before calling it.
    """

    study_secret = (
        secret
        if type(secret) is SelectionSecret
        else SelectionSecret(secret)
    )
    source_index = build_source_index(train_obj, dev_obj)
    plan = select_study(
        source_index,
        study_secret,
        quotas=quotas,
    )
    return TrustedAcquisitionBroker(
        secret=study_secret,
        plan=plan,
    )


def _strict_object_pairs(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QuacP1FormalAcquisitionError(
                "pack contains duplicate JSON keys"
            )
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise QuacP1FormalAcquisitionError(
        f"pack contains non-finite constant {value}"
    )


def decode_strict_pack(
    raw: bytes,
    *,
    expected_schema: str,
) -> dict[str, Any]:
    """Decode only an exact canonical payload with a registered strict schema."""

    if type(raw) is not bytes:
        raise QuacP1FormalAcquisitionError("pack bytes are invalid")
    try:
        text = raw.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise QuacP1FormalAcquisitionError(
            "pack is not canonical ASCII JSON"
        ) from exc
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_nonfinite,
        )
    except QuacP1FormalAcquisitionError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise QuacP1FormalAcquisitionError(
            "pack is not strict JSON"
        ) from exc
    if not isinstance(payload, dict) or canonical_bytes(payload) != raw:
        raise QuacP1FormalAcquisitionError(
            "pack is not exact canonical JSON"
        )
    _validate_pack_schema(payload, expected_schema=expected_schema)
    return payload


def _validate_pack_schema(
    payload: Mapping[str, Any],
    *,
    expected_schema: str,
) -> None:
    registered = {
        VIEW_SCHEMA,
        LABEL_SCHEMA,
        RESERVATION_SCHEMA,
        M_CAPABILITY_SCHEMA,
        LABEL_CAPABILITY_SCHEMA,
    }
    if expected_schema not in registered:
        raise QuacP1FormalAcquisitionError(
            "expected pack schema is outside the registry"
        )
    if payload.get("schema") != expected_schema:
        raise QuacP1FormalAcquisitionError("pack schema drifted")
    if payload.get("study_id") != STUDY_ID:
        raise QuacP1FormalAcquisitionError(
            "pack study binding drifted"
        )

    if expected_schema == VIEW_SCHEMA:
        top = {
            "block",
            "rows",
            "schema",
            "selection_commitment",
            "study_id",
        }
        row_keys = {"query_text", "recent_questions", "work_id"}
    elif expected_schema == LABEL_SCHEMA:
        top = {
            "action_seal_sha256",
            "block",
            "rows",
            "schema",
            "selection_commitment",
            "study_id",
        }
        row_keys = {"family", "qrel_roles", "work_id"}
    elif expected_schema == RESERVATION_SCHEMA:
        top = {
            "block",
            "item_count",
            "materialization_count",
            "materialized_path_count",
            "opaque_reservation_commitment",
            "schema",
            "selection_commitment",
            "study_id",
        }
        row_keys = set()
    elif expected_schema == M_CAPABILITY_SCHEMA:
        top = {
            "capability_mac",
            "promotion_proof_sha256",
            "reservation_commitment",
            "schema",
            "selection_commitment",
            "study_id",
        }
        row_keys = set()
    else:
        top = {
            "action_seal_sha256",
            "block",
            "capability_mac",
            "schema",
            "selection_commitment",
            "study_id",
        }
        row_keys = set()
    if set(payload) != top:
        raise QuacP1FormalAcquisitionError(
            "pack top-level key set drifted"
        )
    selection_commitment = _require_sha256(
        payload.get("selection_commitment"),
        "pack selection commitment",
    )
    if expected_schema in {VIEW_SCHEMA, LABEL_SCHEMA}:
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise QuacP1FormalAcquisitionError(
                "pack rows are invalid"
            )
        if any(
            not isinstance(row, Mapping) or set(row) != row_keys
            for row in rows
        ):
            raise QuacP1FormalAcquisitionError(
                "pack row key set drifted"
            )
        if expected_schema == VIEW_SCHEMA:
            assert_view_is_label_free(payload)
            if any(
                not isinstance(row["recent_questions"], list)
                for row in rows
            ):
                raise QuacP1FormalAcquisitionError(
                    "view recent_questions must be an array"
                )
            try:
                view_rows = tuple(
                    ViewRow(
                        work_id=row["work_id"],
                        query_text=row["query_text"],
                        recent_questions=tuple(row["recent_questions"]),
                    )
                    for row in rows
                )
                ViewPack(
                    block=payload["block"],
                    selection_commitment=selection_commitment,
                    rows=view_rows,
                )
            except (KeyError, TypeError) as exc:
                raise QuacP1FormalAcquisitionError(
                    "view pack value schema drifted"
                ) from exc
        else:
            for row in rows:
                roles = row["qrel_roles"]
                if (
                    not isinstance(roles, Mapping)
                    or set(roles) != set(QREL_ROLE_ORDER)
                    or any(
                        not isinstance(roles[role], list)
                        or not roles[role]
                        or any(
                            not isinstance(value, str)
                            or _HEX64.fullmatch(value) is None
                            for value in roles[role]
                        )
                        for role in QREL_ROLE_ORDER
                    )
                ):
                    raise QuacP1FormalAcquisitionError(
                        "label qrel role schema drifted"
                    )
            try:
                label_rows = tuple(
                    LabelRow(
                        work_id=row["work_id"],
                        family=row["family"],
                        previous_turn_orig_answer=tuple(
                            row["qrel_roles"][QREL_ROLE_ORDER[0]]
                        ),
                        current_turn_orig_answer=tuple(
                            row["qrel_roles"][QREL_ROLE_ORDER[1]]
                        ),
                    )
                    for row in rows
                )
                LabelPack(
                    block=payload["block"],
                    selection_commitment=selection_commitment,
                    action_seal_sha256=payload["action_seal_sha256"],
                    rows=label_rows,
                )
            except (KeyError, TypeError) as exc:
                raise QuacP1FormalAcquisitionError(
                    "label pack value schema drifted"
                ) from exc
    elif expected_schema == RESERVATION_SCHEMA:
        _require_sha256(
            payload.get("opaque_reservation_commitment"),
            "opaque reservation commitment",
        )
        if (
            payload.get("block") != "M_search"
            or type(payload.get("item_count")) is not int
            or payload["item_count"] <= 0
            or type(payload.get("materialization_count")) is not int
            or payload["materialization_count"] not in (0, 1)
            or type(payload.get("materialized_path_count")) is not int
            or payload["materialized_path_count"] < 0
            or (
                payload["materialization_count"] == 0
                and payload["materialized_path_count"] != 0
            )
        ):
            raise QuacP1FormalAcquisitionError(
                "M_search reservation value schema drifted"
            )
    elif expected_schema == M_CAPABILITY_SCHEMA:
        for key in (
            "capability_mac",
            "promotion_proof_sha256",
            "reservation_commitment",
        ):
            _require_sha256(payload.get(key), key)
    else:
        if payload.get("block") not in BLOCK_ORDER:
            raise QuacP1FormalAcquisitionError(
                "late-label capability block drifted"
            )
        _require_sha256(
            payload.get("action_seal_sha256"),
            "action seal SHA256",
        )
        _require_sha256(
            payload.get("capability_mac"),
            "late-label capability MAC",
        )
