"""One-shot private-HMAC acquisition for the EntailmentBank G1/E1 study.

The formal entry point reads each bound Task2 source member exactly once,
selects every experimental block before any action exists, and emits separate
mode-600 label-free views and late label packs.  F_search has no label pack.
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
import subprocess
from typing import Any, Mapping, Sequence
import unicodedata


VERSION = "entailmentbank_proof_retrieval_acquisition_v2"
FAMILY_ORDER = ("TWO_LEAF", "THREE_LEAF", "FOUR_FIVE_LEAF")
BLOCK_ORDER = ("G_form", "A_form", "F_search", "A_hold", "M_search")
TRAIN_BLOCKS = BLOCK_ORDER[:4]
BLOCK_FAMILY_COUNTS = {
    "G_form": {family: 20 for family in FAMILY_ORDER},
    "A_form": {family: 12 for family in FAMILY_ORDER},
    "F_search": {family: 10 for family in FAMILY_ORDER},
    "A_hold": {family: 10 for family in FAMILY_ORDER},
    "M_search": {family: 10 for family in FAMILY_ORDER},
}
BLOCK_COUNTS = {
    block: sum(BLOCK_FAMILY_COUNTS[block].values()) for block in BLOCK_ORDER
}
SPLIT_DEMANDS = {
    "train": {
        family: sum(BLOCK_FAMILY_COUNTS[block][family] for block in TRAIN_BLOCKS)
        for family in FAMILY_ORDER
    },
    "dev": dict(BLOCK_FAMILY_COUNTS["M_search"]),
}
DOCUMENTATION_EXAMPLE_ID = "Mercury_SC_401371"
DESIGN_SHA256 = "95c42921f0fdbc62902234b8ed911c20253dd33cd9df620a0cd4382bbca1e1f6"
SOURCE_REPOSITORY_RELATIVE_PATH = Path(
    "reference/entailment_bank_official_repo_daac2fdb"
)
SOURCE_REPOSITORY_COMMIT = "daac2fdb7ab52ec3ef8f2953f59288c1edd7c2f0"
SOURCE_SPECS = {
    "train": {
        "relative_path": Path(
            "data/public_dataset/entailment_trees_emnlp2021_data_v2/"
            "dataset/task_2/train.jsonl"
        ),
        "byte_size": 10_867_722,
        "sha256": "36cdb362c24755b9640ed54e671fc9c72427b6c918f79429551a0800e9055a1b",
        "line_count": 1_313,
        "candidate_count": 996,
    },
    "dev": {
        "relative_path": Path(
            "data/public_dataset/entailment_trees_emnlp2021_data_v2/"
            "dataset/task_2/dev.jsonl"
        ),
        "byte_size": 1_537_951,
        "sha256": "3271adc67c65149780adbd3729f6b19404ff288e1849905fc16c1c22814a28f7",
        "line_count": 187,
        "candidate_count": 144,
    },
}
FORMAL_COMPONENT_AGGREGATES = {
    "component_count": 1_091,
    "multi_row_component_count": 40,
    "row_count_in_multi_row_components": 89,
    "cross_split_component_count": 15,
    "documentation_example_component_count": 1,
    "clean_component_counts": {"train": 947, "dev": 129},
    "clean_candidate_counts": {
        "train": {
            "TWO_LEAF": 265,
            "THREE_LEAF": 294,
            "FOUR_FIVE_LEAF": 415,
        },
        "dev": {
            "TWO_LEAF": 37,
            "THREE_LEAF": 44,
            "FOUR_FIVE_LEAF": 48,
        },
    },
}
PRIVATE_ROOT_RELATIVE_PATH = Path(
    "artifacts/entailmentbank_proof_retrieval_g1_e1_v2"
)
SECRET_RELATIVE_PATH = PRIVATE_ROOT_RELATIVE_PATH / "selection_secret.bin"
ATTEMPT_RELATIVE_PATH = PRIVATE_ROOT_RELATIVE_PATH / "acquisition_attempt.json"
PACK_ROOT_RELATIVE_PATH = PRIVATE_ROOT_RELATIVE_PATH / "private_packs"
CUSTODY_RELATIVE_PATH = Path(
    "manifests/entailmentbank_proof_retrieval_selection_secret_custody_v2.json"
)
RECEIPT_RELATIVE_PATH = Path(
    "manifests/entailmentbank_proof_retrieval_acquisition_receipt_v2.json"
)
FREEZE_RELATIVE_PATH = Path(
    "manifests/entailmentbank_proof_retrieval_g1_e1_implementation_freeze_v2.json"
)
_HMAC_DOMAIN = b"entailmentbank_proof_retrieval_acquisition_v2/hmac-sha256/v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class EntailmentBankAcquisitionError(RuntimeError):
    """A source, secret, component, selection, or pack invariant drifted."""


class FormalAcquisitionRefusal(EntailmentBankAcquisitionError):
    """The one-shot formal acquisition is no longer pristine."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise EntailmentBankAcquisitionError("value is not canonical JSON") from exc


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise EntailmentBankAcquisitionError("self-hash field already exists")
    output = dict(body)
    output[field] = stable_hash(body)
    return output


def verify_self_hash(value: Mapping[str, Any], field: str) -> str:
    claimed = value.get(field)
    if not isinstance(claimed, str) or _SHA256.fullmatch(claimed) is None:
        raise EntailmentBankAcquisitionError("self-hash is absent")
    body = dict(value)
    del body[field]
    if stable_hash(body) != claimed:
        raise EntailmentBankAcquisitionError("self-hash drifted")
    return claimed


def _no_duplicate_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate object key")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number {value}")


def strict_json_bytes(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise EntailmentBankAcquisitionError(f"invalid strict JSON in {label}") from exc


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise EntailmentBankAcquisitionError(f"{field} is invalid")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise EntailmentBankAcquisitionError(f"{field} is invalid") from exc
    return value


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EntailmentBankAcquisitionError(f"{field} is invalid")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EntailmentBankAcquisitionError(f"{field} is invalid")
    return value


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _frame(raw: bytes) -> bytes:
    return len(raw).to_bytes(8, "big") + raw


def hmac_digest(secret: bytes, purpose: str, *parts: str) -> bytes:
    if not isinstance(secret, bytes) or len(secret) != 32:
        raise EntailmentBankAcquisitionError("selection secret must be 32 bytes")
    rows = (_HMAC_DOMAIN, _text(purpose, "HMAC purpose").encode()) + tuple(
        _text(part, "HMAC part").encode() for part in parts
    )
    return hmac.new(secret, b"".join(_frame(row) for row in rows), hashlib.sha256).digest()


def _parse_text_mapping(value: Any, field: str) -> tuple[tuple[str, str], ...]:
    raw = _mapping(value, field)
    output: list[tuple[str, str]] = []
    for key, text in raw.items():
        output.append((_text(key, f"{field} ID"), _text(text, f"{field} text")))
    if len({key for key, _text_value in output}) != len(output):
        raise EntailmentBankAcquisitionError(f"{field} IDs are duplicated")
    return tuple(output)


def _proof_leaves(
    proof: str,
    *,
    triple_ids: frozenset[str],
    intermediate_ids: frozenset[str],
) -> tuple[str, ...]:
    leaves: list[str] = []
    step_count = 0
    for raw_step in proof.split(";"):
        step = raw_step.strip()
        if not step:
            continue
        step_count += 1
        if step.count("->") != 1:
            raise EntailmentBankAcquisitionError("proof step is invalid")
        left, right = step.split("->", 1)
        if not left.strip() or not right.strip():
            raise EntailmentBankAcquisitionError("proof step is invalid")
        references = tuple(part.strip() for part in left.split("&"))
        if not references or any(not reference for reference in references):
            raise EntailmentBankAcquisitionError("proof references are invalid")
        for reference in references:
            if reference in triple_ids:
                leaves.append(reference)
            elif reference not in intermediate_ids:
                raise EntailmentBankAcquisitionError("proof reference is unknown")
    if step_count == 0 or not leaves:
        raise EntailmentBankAcquisitionError("proof is empty")
    return tuple(dict.fromkeys(leaves))


@dataclass(frozen=True)
class Candidate:
    source_split: str
    source_line_ordinal: int
    item_id: str
    question: str
    answer: str
    hypothesis: str
    triples: tuple[tuple[str, str], ...]
    gold_leaf_ids: tuple[str, ...]
    family: str

    def __post_init__(self) -> None:
        if self.source_split not in {"train", "dev"} or self.family not in FAMILY_ORDER:
            raise EntailmentBankAcquisitionError("candidate split or family drifted")
        if (
            isinstance(self.source_line_ordinal, bool)
            or not isinstance(self.source_line_ordinal, int)
            or self.source_line_ordinal <= 0
        ):
            raise EntailmentBankAcquisitionError("candidate source line ordinal drifted")
        _text(self.item_id, "item ID")
        _text(self.question, "question")
        _text(self.answer, "answer")
        _text(self.hypothesis, "hypothesis")
        if len(self.triples) != 25 or len({key for key, _ in self.triples}) != 25:
            raise EntailmentBankAcquisitionError("candidate must contain 25 triples")
        triple_ids = {key for key, _ in self.triples}
        if (
            not 2 <= len(self.gold_leaf_ids) <= 5
            or len(set(self.gold_leaf_ids)) != len(self.gold_leaf_ids)
            or not set(self.gold_leaf_ids).issubset(triple_ids)
        ):
            raise EntailmentBankAcquisitionError("candidate gold leaves drifted")
        expected = (
            "TWO_LEAF"
            if len(self.gold_leaf_ids) == 2
            else "THREE_LEAF"
            if len(self.gold_leaf_ids) == 3
            else "FOUR_FIVE_LEAF"
        )
        if self.family != expected:
            raise EntailmentBankAcquisitionError("candidate family drifted")

    @property
    def normalized_question(self) -> str:
        return normalize_text(self.question)

    @property
    def normalized_hypothesis(self) -> str:
        return normalize_text(self.hypothesis)

    @property
    def item_key(self) -> str:
        return f"{self.source_split}:{self.source_line_ordinal:08d}:{self.item_id}"

    @property
    def item_commitment_sha256(self) -> str:
        return stable_hash(
            {
                "schema": f"{VERSION}_source_item_commitment",
                "source_split": self.source_split,
                "source_line_ordinal": self.source_line_ordinal,
                "item_id": self.item_id,
                "question": self.question,
                "answer": self.answer,
                "hypothesis": self.hypothesis,
                "triples": [list(row) for row in self.triples],
                "gold_leaf_ids": list(self.gold_leaf_ids),
            }
        )


def _parse_row(
    value: Mapping[str, Any], *, split: str, source_line_ordinal: int
) -> Candidate | None:
    item_id = _text(value.get("id"), "item ID")
    question = _text(value.get("question"), "question")
    answer = _text(value.get("answer"), "answer")
    hypothesis = _text(value.get("hypothesis"), "hypothesis")
    proof = _text(value.get("proof"), "proof")
    meta = _mapping(value.get("meta"), "meta")
    triples = _parse_text_mapping(meta.get("triples"), "triples")
    intermediates = _parse_text_mapping(
        meta.get("intermediate_conclusions"), "intermediate conclusions"
    )
    distractors = tuple(
        _text(row, "distractor ID")
        for row in _sequence(meta.get("distractors", ()), "distractors")
    )
    triple_ids = frozenset(key for key, _ in triples)
    if len(set(distractors)) != len(distractors) or not set(distractors).issubset(triple_ids):
        raise EntailmentBankAcquisitionError("distractor registry drifted")
    leaves = _proof_leaves(
        proof,
        triple_ids=triple_ids,
        intermediate_ids=frozenset(key for key, _ in intermediates),
    )
    if set(leaves).intersection(distractors):
        raise EntailmentBankAcquisitionError("gold leaf is marked distractor")
    if not 8 <= len(triples) <= 64 or not 2 <= len(leaves) <= 5:
        return None
    if len(triples) != 25:
        raise EntailmentBankAcquisitionError("qualified Task2 context width drifted")
    family = (
        "TWO_LEAF"
        if len(leaves) == 2
        else "THREE_LEAF"
        if len(leaves) == 3
        else "FOUR_FIVE_LEAF"
    )
    return Candidate(
        split,
        source_line_ordinal,
        item_id,
        question,
        answer,
        hypothesis,
        triples,
        leaves,
        family,
    )


def parse_source(raw: bytes, *, split: str) -> tuple[tuple[Candidate, ...], dict[str, int]]:
    if split not in {"train", "dev"}:
        raise EntailmentBankAcquisitionError("source split drifted")
    candidates: list[Candidate] = []
    ineligible = 0
    lines = raw.splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise EntailmentBankAcquisitionError("blank source line")
        value = strict_json_bytes(line, label=f"{split} line {line_number}")
        candidate = _parse_row(
            _mapping(value, "row root"),
            split=split,
            source_line_ordinal=line_number,
        )
        if candidate is None:
            ineligible += 1
        else:
            candidates.append(candidate)
    return tuple(candidates), {
        "line_count": len(lines),
        "candidate_count": len(candidates),
        "formal_ineligible_count": ineligible,
    }


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
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


@dataclass(frozen=True)
class Component:
    token: str
    split: str
    candidates: tuple[Candidate, ...]

    @property
    def families(self) -> frozenset[str]:
        return frozenset(row.family for row in self.candidates)


def build_clean_components(
    train: Sequence[Candidate], dev: Sequence[Candidate]
) -> tuple[Mapping[str, tuple[Component, ...]], Mapping[str, Any]]:
    candidates = tuple(train) + tuple(dev)
    if not candidates or len({row.item_key for row in candidates}) != len(candidates):
        raise EntailmentBankAcquisitionError("candidate identity registry drifted")
    union = _UnionFind(len(candidates))
    seen_id: dict[str, int] = {}
    seen_question: dict[str, int] = {}
    seen_hypothesis: dict[str, int] = {}
    for index, candidate in enumerate(candidates):
        for registry, key in (
            (seen_id, candidate.item_id),
            (seen_question, candidate.normalized_question),
            (seen_hypothesis, candidate.normalized_hypothesis),
        ):
            previous = registry.setdefault(key, index)
            union.union(index, previous)
    groups: dict[int, list[Candidate]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        groups[union.find(index)].append(candidate)
    clean: dict[str, list[Component]] = {"train": [], "dev": []}
    cross_split = 0
    documentation = 0
    clean_candidate_counts = {
        split: Counter() for split in ("train", "dev")
    }
    for rows in groups.values():
        splits = {row.source_split for row in rows}
        is_cross = len(splits) > 1
        is_documentation = any(row.item_id == DOCUMENTATION_EXAMPLE_ID for row in rows)
        cross_split += int(is_cross)
        documentation += int(is_documentation)
        if is_cross or is_documentation:
            continue
        split = rows[0].source_split
        ordered = tuple(sorted(rows, key=lambda row: row.item_key))
        token = stable_hash([row.item_key for row in ordered])
        clean[split].append(Component(token, split, ordered))
        clean_candidate_counts[split].update(row.family for row in ordered)
    output = {
        split: tuple(sorted(rows, key=lambda row: row.token))
        for split, rows in clean.items()
    }
    multi = [rows for rows in groups.values() if len(rows) > 1]
    audit = {
        "component_count": len(groups),
        "multi_row_component_count": len(multi),
        "row_count_in_multi_row_components": sum(len(rows) for rows in multi),
        "cross_split_component_count": cross_split,
        "documentation_example_component_count": documentation,
        "clean_component_counts": {split: len(output[split]) for split in output},
        "clean_candidate_counts": {
            split: {
                family: clean_candidate_counts[split][family]
                for family in FAMILY_ORDER
            }
            for split in output
        },
    }
    return output, audit


@dataclass
class _Edge:
    to: int
    reverse: int
    capacity: int
    cost: int


def _add_edge(graph: list[list[_Edge]], left: int, right: int, capacity: int, cost: int) -> _Edge:
    forward = _Edge(right, len(graph[right]), capacity, cost)
    reverse = _Edge(left, len(graph[left]), 0, -cost)
    graph[left].append(forward)
    graph[right].append(reverse)
    return forward


def assign_components(
    components: Sequence[Component],
    demands: Mapping[str, int],
    *,
    secret: bytes,
) -> Mapping[str, tuple[Component, ...]]:
    rows = tuple(sorted(components, key=lambda row: row.token))
    if set(demands) != set(FAMILY_ORDER) or any(
        type(demands[family]) is not int or demands[family] < 0 for family in FAMILY_ORDER
    ):
        raise EntailmentBankAcquisitionError("family demands drifted")
    source = 0
    component_offset = 1
    family_offset = component_offset + len(rows)
    sink = family_offset + len(FAMILY_ORDER)
    graph: list[list[_Edge]] = [[] for _ in range(sink + 1)]
    family_nodes = {
        family: family_offset + index for index, family in enumerate(FAMILY_ORDER)
    }
    assignment_edges: dict[tuple[str, str], _Edge] = {}
    for index, component in enumerate(rows):
        node = component_offset + index
        _add_edge(graph, source, node, 1, 0)
        for family in FAMILY_ORDER:
            if family in component.families:
                cost = int.from_bytes(
                    hmac_digest(secret, "component_family_cost", component.split, component.token, family),
                    "big",
                )
                assignment_edges[(component.token, family)] = _add_edge(
                    graph, node, family_nodes[family], 1, cost
                )
    for family in FAMILY_ORDER:
        _add_edge(graph, family_nodes[family], sink, demands[family], 0)
    required = sum(demands.values())
    potential = [0] * len(graph)
    flow = 0
    while flow < required:
        distance: list[int | None] = [None] * len(graph)
        previous: list[tuple[int, int] | None] = [None] * len(graph)
        distance[source] = 0
        heap: list[tuple[int, int]] = [(0, source)]
        while heap:
            current, node = heapq.heappop(heap)
            if distance[node] != current:
                continue
            for edge_index, edge in enumerate(graph[node]):
                if edge.capacity <= 0:
                    continue
                reduced = edge.cost + potential[node] - potential[edge.to]
                if reduced < 0:
                    raise EntailmentBankAcquisitionError("flow reduced cost became negative")
                candidate_distance = current + reduced
                old = distance[edge.to]
                predecessor = (node, edge_index)
                if old is None or candidate_distance < old:
                    distance[edge.to] = candidate_distance
                    previous[edge.to] = predecessor
                    heapq.heappush(heap, (candidate_distance, edge.to))
        if distance[sink] is None:
            raise EntailmentBankAcquisitionError("component capacity is below frozen demand")
        for node, value in enumerate(distance):
            if value is not None:
                potential[node] += value
        node = sink
        while node != source:
            parent_edge = previous[node]
            if parent_edge is None:
                raise EntailmentBankAcquisitionError("flow predecessor is absent")
            parent, edge_index = parent_edge
            edge = graph[parent][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = parent
        flow += 1
    assigned: dict[str, list[Component]] = {family: [] for family in FAMILY_ORDER}
    by_token = {component.token: component for component in rows}
    for (token, family), edge in assignment_edges.items():
        if edge.capacity == 0:
            assigned[family].append(by_token[token])
    if any(len(assigned[family]) != demands[family] for family in FAMILY_ORDER):
        raise EntailmentBankAcquisitionError("flow family demand was not saturated")
    return {
        family: tuple(sorted(assigned[family], key=lambda row: row.token))
        for family in FAMILY_ORDER
    }


def _choose_candidate(component: Component, family: str, secret: bytes) -> Candidate:
    eligible = tuple(row for row in component.candidates if row.family == family)
    if not eligible:
        raise EntailmentBankAcquisitionError("assigned component lacks its family")
    return min(
        eligible,
        key=lambda row: (
            hmac_digest(secret, "within_component_candidate", component.token, family, row.item_key),
            row.item_key,
        ),
    )


def select_blocks(
    train: Sequence[Candidate],
    dev: Sequence[Candidate],
    *,
    secret: bytes,
) -> tuple[Mapping[str, tuple[Candidate, ...]], Mapping[str, Any]]:
    clean, component_audit = build_clean_components(train, dev)
    assigned_by_split = {
        split: assign_components(clean[split], SPLIT_DEMANDS[split], secret=secret)
        for split in ("train", "dev")
    }
    blocks: dict[str, list[Candidate]] = {block: [] for block in BLOCK_ORDER}
    for split in ("train", "dev"):
        split_blocks = TRAIN_BLOCKS if split == "train" else ("M_search",)
        for family in FAMILY_ORDER:
            selected = [
                _choose_candidate(component, family, secret)
                for component in assigned_by_split[split][family]
            ]
            selected.sort(
                key=lambda row: (
                    hmac_digest(secret, "family_partition_order", split, family, row.item_key),
                    row.item_key,
                )
            )
            cursor = 0
            for block in split_blocks:
                take = BLOCK_FAMILY_COUNTS[block][family]
                partition = selected[cursor : cursor + take]
                if len(partition) != take:
                    raise EntailmentBankAcquisitionError("family partition is short")
                blocks[block].extend(partition)
                cursor += take
            if cursor != len(selected):
                raise EntailmentBankAcquisitionError("family partition left unused rows")
    finalized: dict[str, tuple[Candidate, ...]] = {}
    for block in BLOCK_ORDER:
        ordered = sorted(
            blocks[block],
            key=lambda row: (
                hmac_digest(secret, "within_block_order", block, row.item_key),
                row.item_key,
            ),
        )
        finalized[block] = tuple(ordered)
    selected = tuple(row for block in BLOCK_ORDER for row in finalized[block])
    if (
        any(len(finalized[block]) != BLOCK_COUNTS[block] for block in BLOCK_ORDER)
        or len({row.item_key for row in selected}) != len(selected)
    ):
        raise EntailmentBankAcquisitionError("selected block shape drifted")
    family_counts = {
        block: {
            family: sum(row.family == family for row in finalized[block])
            for family in FAMILY_ORDER
        }
        for block in BLOCK_ORDER
    }
    if family_counts != BLOCK_FAMILY_COUNTS:
        raise EntailmentBankAcquisitionError("selected family balance drifted")
    return finalized, {
        "component_aggregates": component_audit,
        "selected_block_counts": dict(BLOCK_COUNTS),
        "selected_family_counts": family_counts,
        "selected_unique_component_bound_item_count": len(selected),
    }


def build_private_pack_payloads(
    blocks: Mapping[str, Sequence[Candidate]],
) -> Mapping[str, Mapping[str, Any]]:
    if set(blocks) != set(BLOCK_ORDER):
        raise EntailmentBankAcquisitionError("private pack block registry drifted")
    payloads: dict[str, Mapping[str, Any]] = {}
    all_item_keys: list[str] = []
    all_commitments: list[str] = []
    for block in BLOCK_ORDER:
        rows = tuple(blocks[block])
        if len(rows) != BLOCK_COUNTS[block]:
            raise EntailmentBankAcquisitionError("private pack block count drifted")
        family_counts = Counter(row.family for row in rows)
        if {
            family: family_counts[family] for family in FAMILY_ORDER
        } != BLOCK_FAMILY_COUNTS[block]:
            raise EntailmentBankAcquisitionError("private pack family balance drifted")
        source_split = "dev" if block == "M_search" else "train"
        views = []
        labels = []
        for ordinal, row in enumerate(rows):
            if row.source_split != source_split:
                raise EntailmentBankAcquisitionError("block source split drifted")
            commitment = row.item_commitment_sha256
            all_item_keys.append(row.item_key)
            all_commitments.append(commitment)
            views.append(
                {
                    "ordinal": ordinal,
                    "item_commitment_sha256": commitment,
                    "question": row.question,
                    "answer": row.answer,
                    "hypothesis": row.hypothesis,
                    "node_texts": [text for _key, text in row.triples],
                }
            )
            if block != "F_search":
                ordinal_by_id = {key: index for index, (key, _text_value) in enumerate(row.triples)}
                gold_ordinals = sorted(ordinal_by_id[key] for key in row.gold_leaf_ids)
                labels.append(
                    {
                        "ordinal": ordinal,
                        "item_commitment_sha256": commitment,
                        "family": row.family,
                        "gold_ordinals": gold_ordinals,
                    }
                )
        view_body = {
            "schema": f"{VERSION}_block_view",
            "block": block,
            "source_split": source_split,
            "item_count": len(views),
            "items": views,
            "excluded_fields": [
                "proof",
                "meta.distractors",
                "meta.intermediate_conclusions",
                "gold_leaf_IDs",
                "family",
                "source_item_ID",
            ],
        }
        payloads[f"{block}.view.private.json"] = self_hashed(view_body, "pack_sha256")
        if block != "F_search":
            label_body = {
                "schema": f"{VERSION}_block_labels",
                "block": block,
                "source_split": source_split,
                "item_count": len(labels),
                "items": labels,
            }
            payloads[f"{block}.labels.private.json"] = self_hashed(
                label_body, "pack_sha256"
            )
    if "F_search.labels.private.json" in payloads or len(payloads) != 9:
        raise EntailmentBankAcquisitionError("F_search label isolation drifted")
    if (
        len(set(all_item_keys)) != sum(BLOCK_COUNTS.values())
        or len(set(all_commitments)) != sum(BLOCK_COUNTS.values())
    ):
        raise EntailmentBankAcquisitionError("private pack item uniqueness drifted")
    return dict(sorted(payloads.items()))


def _atomic_private_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    raw = canonical_json_bytes(value) + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_private_pack_payloads(
    pack_root: Path, payloads: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, str]:
    expected = {
        f"{block}.view.private.json" for block in BLOCK_ORDER
    }.union(
        f"{block}.labels.private.json" for block in BLOCK_ORDER if block != "F_search"
    )
    if set(payloads) != expected or pack_root.exists():
        raise EntailmentBankAcquisitionError("private pack destination is not pristine")
    pack_root.mkdir(parents=True, mode=0o700)
    os.chmod(pack_root, 0o700)
    hashes: dict[str, str] = {}
    for name in sorted(payloads):
        path = pack_root / name
        _atomic_private_write(path, payloads[name])
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _git_output(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args), cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return result.stdout.strip()


def _write_public_manifest(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FormalAcquisitionRefusal("public manifest already exists")
    raw = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _load_json_file(path: Path, *, label: str) -> Mapping[str, Any]:
    value = strict_json_bytes(path.read_bytes(), label=label)
    return _mapping(value, label)


def generate_selection_secret(project_root: Path) -> Mapping[str, Any]:
    root = project_root / "reconstruction_v2"
    custody_path = root / CUSTODY_RELATIVE_PATH
    secret_path = root / SECRET_RELATIVE_PATH
    freeze_path = root / FREEZE_RELATIVE_PATH
    if custody_path.exists() or secret_path.exists():
        raise FormalAcquisitionRefusal("selection secret custody is not pristine")
    freeze = _load_json_file(freeze_path, label="implementation freeze")
    freeze_sha = verify_self_hash(freeze, "implementation_freeze_sha256")
    if (
        freeze.get("schema") != "entailmentbank_proof_retrieval_g1_e1_implementation_freeze_v2"
        or freeze.get("design_sha256") != DESIGN_SHA256
        or freeze.get("status") != "full_implementation_frozen_before_selection_secret"
    ):
        raise FormalAcquisitionRefusal("implementation freeze is invalid")
    relative_freeze = FREEZE_RELATIVE_PATH.as_posix()
    committed_blob = _git_output(
        project_root, "rev-parse", f"HEAD:reconstruction_v2/{relative_freeze}"
    )
    working_blob = _git_output(project_root, "hash-object", str(freeze_path))
    if committed_blob != working_blob:
        raise FormalAcquisitionRefusal("implementation freeze is not committed at HEAD")
    secret_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(secret_path.parent, 0o700)
    secret = os.urandom(32)
    descriptor = os.open(secret_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(secret)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(secret_path, 0o600)
    body = {
        "schema": "entailmentbank_proof_retrieval_selection_secret_custody_v2",
        "status": "fresh_private_selection_secret_generated_no_source_open",
        "design_sha256": DESIGN_SHA256,
        "implementation_freeze_sha256": freeze_sha,
        "implementation_freeze_git_commit": _git_output(project_root, "rev-parse", "HEAD"),
        "secret_byte_count": 32,
        "secret_file_sha256": hashlib.sha256(secret).hexdigest(),
        "secret_path": SECRET_RELATIVE_PATH.as_posix(),
        "source_payload_opened_or_parsed": False,
        "item_selected_or_action_run": False,
    }
    custody = self_hashed(body, "selection_secret_custody_sha256")
    _write_public_manifest(custody_path, custody)
    return custody


def _read_formal_sources_once(root: Path) -> tuple[bytes, bytes, Mapping[str, Any]]:
    source_root = root / SOURCE_REPOSITORY_RELATIVE_PATH
    if _git_output(source_root, "rev-parse", "HEAD") != SOURCE_REPOSITORY_COMMIT:
        raise FormalAcquisitionRefusal("official source commit drifted")
    if _git_output(source_root, "status", "--short"):
        raise FormalAcquisitionRefusal("official source worktree drifted")
    values: dict[str, bytes] = {}
    binding: dict[str, Any] = {}
    for split in ("train", "dev"):
        spec = SOURCE_SPECS[split]
        path = source_root / spec["relative_path"]
        if path.stat().st_size != spec["byte_size"]:
            raise FormalAcquisitionRefusal("official source size drifted")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != spec["sha256"]:
            raise FormalAcquisitionRefusal("official source hash drifted")
        values[split] = raw
        binding[split] = {
            "byte_size": len(raw),
            "sha256": digest,
            "read_count": 1,
        }
    return values["train"], values["dev"], binding


def run_formal_acquisition(project_root: Path) -> Mapping[str, Any]:
    root = project_root / "reconstruction_v2"
    custody = _load_json_file(root / CUSTODY_RELATIVE_PATH, label="secret custody")
    custody_sha = verify_self_hash(custody, "selection_secret_custody_sha256")
    secret_path = root / SECRET_RELATIVE_PATH
    secret = secret_path.read_bytes()
    if len(secret) != 32 or hashlib.sha256(secret).hexdigest() != custody.get("secret_file_sha256"):
        raise FormalAcquisitionRefusal("private selection secret drifted")
    attempt_path = root / ATTEMPT_RELATIVE_PATH
    receipt_path = root / RECEIPT_RELATIVE_PATH
    pack_root = root / PACK_ROOT_RELATIVE_PATH
    if attempt_path.exists() or receipt_path.exists() or pack_root.exists():
        raise FormalAcquisitionRefusal("formal acquisition attempt is not pristine")
    marker = self_hashed(
        {
            "schema": f"{VERSION}_attempt",
            "status": "started_before_source_payload_open",
            "selection_secret_custody_sha256": custody_sha,
            "source_parse_attempts_authorized": 1,
        },
        "attempt_sha256",
    )
    _atomic_private_write(attempt_path, marker)
    train_raw, dev_raw, source_binding = _read_formal_sources_once(root)
    train, train_audit = parse_source(train_raw, split="train")
    dev, dev_audit = parse_source(dev_raw, split="dev")
    for split, audit in (("train", train_audit), ("dev", dev_audit)):
        spec = SOURCE_SPECS[split]
        if audit["line_count"] != spec["line_count"] or audit["candidate_count"] != spec["candidate_count"]:
            raise FormalAcquisitionRefusal("formal source aggregate drifted")
    blocks, selection_audit = select_blocks(train, dev, secret=secret)
    if selection_audit["component_aggregates"] != FORMAL_COMPONENT_AGGREGATES:
        raise FormalAcquisitionRefusal("formal component aggregate drifted")
    payloads = build_private_pack_payloads(blocks)
    file_hashes = write_private_pack_payloads(pack_root, payloads)
    body = {
        "schema": "entailmentbank_proof_retrieval_acquisition_receipt_v2",
        "status": "formal_186_item_private_cohort_acquired_before_any_action",
        "design_sha256": DESIGN_SHA256,
        "selection_secret_custody_sha256": custody_sha,
        "source_binding": source_binding,
        "source_parse_counts": {"train": train_audit, "dev": dev_audit},
        "selection_aggregates": selection_audit,
        "private_pack_file_sha256s": dict(sorted(file_hashes.items())),
        "private_pack_count": len(file_hashes),
        "F_search_label_pack_created": False,
        "all_blocks_selected_before_any_item_action": True,
        "source_payload_read_count_per_member": 1,
        "item_action_model_score_or_evaluator_calls": 0,
        "external_network_calls": 0,
        "test_split_opened_hashed_or_parsed": False,
    }
    receipt = self_hashed(body, "acquisition_receipt_sha256")
    _write_public_manifest(receipt_path, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--generate-secret", action="store_true")
    parser.add_argument("--acquire", action="store_true")
    args = parser.parse_args(argv)
    if args.generate_secret == args.acquire:
        parser.error("choose exactly one of --generate-secret or --acquire")
    value = (
        generate_selection_secret(args.project_root.resolve())
        if args.generate_secret
        else run_formal_acquisition(args.project_root.resolve())
    )
    print(json.dumps(value, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BLOCK_COUNTS",
    "BLOCK_FAMILY_COUNTS",
    "BLOCK_ORDER",
    "Candidate",
    "EntailmentBankAcquisitionError",
    "FAMILY_ORDER",
    "FormalAcquisitionRefusal",
    "assign_components",
    "build_clean_components",
    "build_private_pack_payloads",
    "canonical_json_bytes",
    "generate_selection_secret",
    "hmac_digest",
    "parse_source",
    "run_formal_acquisition",
    "select_blocks",
    "self_hashed",
    "stable_hash",
    "strict_json_bytes",
    "verify_self_hash",
    "write_private_pack_payloads",
]
