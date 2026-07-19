"""Aggregate-only one-shot EntailmentBank Task2 source qualification.

The formal path decodes only the fixed official Task2 TRAIN and DEV files.  It
checks proof-leaf, context, exposure, collision-component, and balanced-family
capacity contracts without selecting an item or running an action or score.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence
import unicodedata


VERSION = "v1"
SCHEMA = "entailmentbank_proof_retrieval_source_qualification_result_v1"
ATTEMPT_SCHEMA = "entailmentbank_proof_retrieval_source_qualification_attempt_v1"
FREEZE_SCHEMA = (
    "entailmentbank_proof_retrieval_source_qualification_implementation_freeze_v1"
)

FAMILY_ORDER = ("TWO_LEAF", "THREE_LEAF", "FOUR_FIVE_LEAF")
SPLIT_ORDER = ("train", "dev")
TRAIN_DEMANDS = {family: 52 for family in FAMILY_ORDER}
DEV_DEMANDS = {family: 10 for family in FAMILY_ORDER}
DOCUMENTATION_EXAMPLE_ID = "Mercury_SC_401371"

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
        "git_blob_sha1": "c508dea8c9256e7dfa78ce4c100177a815734839",
        "sha256": "36cdb362c24755b9640ed54e671fc9c72427b6c918f79429551a0800e9055a1b",
    },
    "dev": {
        "relative_path": Path(
            "data/public_dataset/entailment_trees_emnlp2021_data_v2/"
            "dataset/task_2/dev.jsonl"
        ),
        "byte_size": 1_537_951,
        "git_blob_sha1": "fc909f2f155892e863ba0becfa03e243477c88f5",
        "sha256": "3271adc67c65149780adbd3729f6b19404ff288e1849905fc16c1c22814a28f7",
    },
}

CUSTODY_RELATIVE_PATH = Path(
    "manifests/entailmentbank_proof_retrieval_source_custody_v1.json"
)
CUSTODY_FILE_SHA256 = (
    "458717659938c93201a71c7f28684a459e61240d3b15342b5c53211aadcbb7f7"
)
CUSTODY_SELF_SHA256 = (
    "99ccb112e7bfb5f326c13a3928d58d10a9ea341a9ca49d58c41fd4954fb018b3"
)
ACCESS_RELATIVE_PATH = Path(
    "manifests/entailmentbank_proof_retrieval_source_access_v1.json"
)
ACCESS_FILE_SHA256 = (
    "155789a2e581f8e43f622021b1adb7839bb10fefd35ce6523e87e7650bf12a7f"
)
ACCESS_SELF_SHA256 = (
    "ecb0d0c6508373e5df3564a60231d3011728d4350412a69b5b94492cfa24ce40"
)
DESIGN_RELATIVE_PATH = Path(
    "manifests/entailmentbank_proof_retrieval_source_qualification_design_v1.json"
)
DESIGN_FILE_SHA256 = (
    "68d9eb733a133c99a79a328058b2f3bd48576dead466253cd011dc27c8c2f29c"
)
DESIGN_SELF_SHA256 = (
    "9ada3f1760e8bcc46a5a78ef09f5c0621cc89fa52902c74eaf8c0e3bb4f6d1b4"
)
QUALIFIER_RELATIVE_PATH = Path(
    "assumption_agent/benchmarks/"
    "entailmentbank_proof_retrieval_source_qualification_v1.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/test_entailmentbank_proof_retrieval_source_qualification_v1.py"
)
FREEZE_RELATIVE_PATH = Path(
    "manifests/"
    "entailmentbank_proof_retrieval_source_qualification_implementation_freeze_v1.json"
)
ATTEMPT_RELATIVE_PATH = Path(
    "artifacts/entailmentbank_proof_retrieval_source_qualification_v1"
)
RESULT_RELATIVE_PATH = Path(
    "manifests/entailmentbank_proof_retrieval_source_qualification_result_v1.json"
)


class EntailmentBankQualificationError(RuntimeError):
    """Fail-closed qualification error with no item content."""


class FormalProvenanceError(EntailmentBankQualificationError):
    """A committed source or implementation binding drifted."""


class OneShotRefusal(EntailmentBankQualificationError):
    """The formal qualification attempt is not pristine."""


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
        raise EntailmentBankQualificationError(
            "non-canonical public value"
        ) from exc


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
        raise EntailmentBankQualificationError(
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


def _text(value: Any, reason: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise _RowInvalid(reason)
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise _RowInvalid(reason) from exc
    if nonempty and not value.strip():
        raise _RowInvalid(reason)
    return value


def _mapping(value: Any, reason: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _RowInvalid(reason)
    return value


def _sequence(value: Any, reason: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _RowInvalid(reason)
    return value


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _parse_text_mapping(value: Any, reason: str) -> dict[str, str]:
    raw = _mapping(value, reason)
    parsed: dict[str, str] = {}
    for key, text in raw.items():
        identifier = _text(key, reason + "_id")
        if identifier in parsed:
            raise _RowInvalid(reason + "_id")
        parsed[identifier] = _text(text, reason + "_text")
    return parsed


def _extract_proof_leaves(
    proof: str,
    *,
    triples: Mapping[str, str],
    intermediates: Mapping[str, str],
) -> tuple[str, ...]:
    leaves: list[str] = []
    step_count = 0
    for raw_step in proof.split(";"):
        step = raw_step.strip()
        if not step:
            continue
        step_count += 1
        if step.count("->") != 1:
            raise _RowInvalid("proof_step_arrow")
        raw_left, raw_right = step.split("->", 1)
        if not raw_left.strip() or not raw_right.strip():
            raise _RowInvalid("proof_step_shape")
        references = [reference.strip() for reference in raw_left.split("&")]
        if not references or any(not reference for reference in references):
            raise _RowInvalid("proof_LHS_shape")
        for reference in references:
            if reference in triples:
                leaves.append(reference)
            elif reference not in intermediates:
                raise _RowInvalid("proof_LHS_unknown_reference")
    if step_count == 0 or not leaves:
        raise _RowInvalid("proof_empty")
    return tuple(dict.fromkeys(leaves))


@dataclass(frozen=True)
class _Candidate:
    split: str
    item_id: str
    normalized_question: str
    normalized_hypothesis: str
    family: str
    context_size: int
    gold_leaf_count: int
    proof_step_count: int
    distractor_count: int

    @property
    def item_key(self) -> str:
        return f"{self.split}:{self.item_id}"


@dataclass(frozen=True)
class _SplitAudit:
    split: str
    line_count: int
    reader_valid_row_count: int
    candidates: tuple[_Candidate, ...]
    invalid_reason_counts: Mapping[str, int]
    ineligible_reason_counts: Mapping[str, int]
    context_size_histogram: Mapping[int, int]
    proof_leaf_count_histogram: Mapping[int, int]
    proof_step_count_histogram: Mapping[int, int]
    distractor_count_histogram: Mapping[int, int]


def _parse_row(value: Mapping[str, Any], *, split: str) -> _Candidate:
    item_id = _text(value.get("id"), "item_id")
    question = _text(value.get("question"), "question")
    _text(value.get("answer"), "answer")
    hypothesis = _text(value.get("hypothesis"), "hypothesis")
    proof = _text(value.get("proof"), "proof")
    meta = _mapping(value.get("meta"), "meta")
    triples = _parse_text_mapping(meta.get("triples"), "triples")
    intermediates = _parse_text_mapping(
        meta.get("intermediate_conclusions"), "intermediate_conclusions"
    )
    raw_distractors = meta.get("distractors", ())
    distractors = tuple(
        _text(identifier, "distractor_id")
        for identifier in _sequence(raw_distractors, "distractors")
    )
    if len(set(distractors)) != len(distractors):
        raise _RowInvalid("duplicate_distractor_id")
    if any(identifier not in triples for identifier in distractors):
        raise _RowInvalid("unknown_distractor_id")
    leaves = _extract_proof_leaves(
        proof,
        triples=triples,
        intermediates=intermediates,
    )
    if any(identifier in set(distractors) for identifier in leaves):
        raise _RowInvalid("gold_leaf_marked_distractor")
    context_size = len(triples)
    if not 8 <= context_size <= 64:
        raise _RowInvalid("formal_context_size")
    leaf_count = len(leaves)
    if not 2 <= leaf_count <= 5:
        raise _RowInvalid("formal_gold_leaf_size")
    if leaf_count == 2:
        family = "TWO_LEAF"
    elif leaf_count == 3:
        family = "THREE_LEAF"
    else:
        family = "FOUR_FIVE_LEAF"
    step_count = sum(bool(step.strip()) for step in proof.split(";"))
    return _Candidate(
        split,
        item_id,
        _normalize(question),
        _normalize(hypothesis),
        family,
        context_size,
        leaf_count,
        step_count,
        len(distractors),
    )


def _audit_split(raw: bytes, *, split: str) -> _SplitAudit:
    invalid = Counter()
    ineligible = Counter()
    candidates: list[_Candidate] = []
    reader_valid = 0
    context_histogram = Counter()
    leaf_histogram = Counter()
    step_histogram = Counter()
    distractor_histogram = Counter()
    lines = raw.splitlines()
    formal_reasons = frozenset(("formal_context_size", "formal_gold_leaf_size"))
    for line in lines:
        try:
            value = _strict_json_line(line)
            candidate = _parse_row(value, split=split)
            reader_valid += 1
            candidates.append(candidate)
            context_histogram[candidate.context_size] += 1
            leaf_histogram[candidate.gold_leaf_count] += 1
            step_histogram[candidate.proof_step_count] += 1
            distractor_histogram[candidate.distractor_count] += 1
        except _RowInvalid as exc:
            if exc.reason in formal_reasons:
                ineligible[exc.reason] += 1
            else:
                invalid[exc.reason] += 1
    return _SplitAudit(
        split,
        len(lines),
        reader_valid,
        tuple(candidates),
        dict(sorted(invalid.items())),
        dict(sorted(ineligible.items())),
        dict(sorted(context_histogram.items())),
        dict(sorted(leaf_histogram.items())),
        dict(sorted(step_histogram.items())),
        dict(sorted(distractor_histogram.items())),
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
    source = 0
    component_offset = 1
    family_offset = component_offset + len(components)
    sink = family_offset + len(FAMILY_ORDER)
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    component_nodes = {
        component: component_offset + index
        for index, component in enumerate(components)
    }
    family_nodes = {
        family: family_offset + index
        for index, family in enumerate(FAMILY_ORDER)
    }
    sink_edges: dict[str, _FlowEdge] = {}
    for component in components:
        _add_edge(graph, source, component_nodes[component], 1)
        for family in FAMILY_ORDER:
            if family in component_families[component]:
                _add_edge(
                    graph,
                    component_nodes[component],
                    family_nodes[family],
                    1,
                )
    for family in FAMILY_ORDER:
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
        for family in FAMILY_ORDER
    }
    return flow, assigned


def _component_audit(
    audits: Sequence[_SplitAudit],
) -> tuple[dict[str, Any], Mapping[str, Mapping[str, frozenset[str]]]]:
    candidates = tuple(
        candidate for audit in audits for candidate in audit.candidates
    )
    union_find = _UnionFind(len(candidates))
    seen_id: dict[str, int] = {}
    seen_question: dict[str, int] = {}
    seen_hypothesis: dict[str, int] = {}
    for index, candidate in enumerate(candidates):
        for seen, value in (
            (seen_id, candidate.item_id),
            (seen_question, candidate.normalized_question),
            (seen_hypothesis, candidate.normalized_hypothesis),
        ):
            previous = seen.setdefault(value, index)
            union_find.union(index, previous)
    groups: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        groups.setdefault(union_find.find(index), []).append(index)

    cross_split_roots: set[int] = set()
    documentation_roots: set[int] = set()
    for root, indices in groups.items():
        if len({candidates[index].split for index in indices}) > 1:
            cross_split_roots.add(root)
        if any(
            candidates[index].item_id == DOCUMENTATION_EXAMPLE_ID
            for index in indices
        ):
            documentation_roots.add(root)

    split_output: dict[str, Any] = {}
    flow_inputs: dict[str, dict[str, frozenset[str]]] = {}
    for split in SPLIT_ORDER:
        pre_family = Counter(
            candidate.family
            for candidate in candidates
            if candidate.split == split
        )
        cross_family = Counter()
        documentation_family = Counter()
        clean_family = Counter()
        component_families: dict[str, set[str]] = {}
        item_keys = {family: [] for family in FAMILY_ORDER}
        for root, indices in groups.items():
            for index in indices:
                candidate = candidates[index]
                if candidate.split != split:
                    continue
                if root in cross_split_roots:
                    cross_family[candidate.family] += 1
                elif root in documentation_roots:
                    documentation_family[candidate.family] += 1
                else:
                    token = _sha256(_canonical_json([root, split]))
                    component_families.setdefault(token, set()).add(candidate.family)
                    clean_family[candidate.family] += 1
                    item_keys[candidate.family].append(candidate.item_key)
        profiles = Counter(
            "+".join(family for family in FAMILY_ORDER if family in families)
            for families in component_families.values()
        )
        flow_inputs[split] = {
            token: frozenset(families)
            for token, families in component_families.items()
        }
        split_output[split] = {
            "pre_component_candidate_counts": {
                family: pre_family[family] for family in FAMILY_ORDER
            },
            "cross_split_component_excluded_candidate_counts": {
                family: cross_family[family] for family in FAMILY_ORDER
            },
            "documentation_example_component_excluded_candidate_counts": {
                family: documentation_family[family] for family in FAMILY_ORDER
            },
            "clean_candidate_counts": {
                family: clean_family[family] for family in FAMILY_ORDER
            },
            "clean_component_count": len(component_families),
            "clean_component_family_profile_counts": dict(sorted(profiles.items())),
            "clean_population_key_sha256_by_family": {
                family: _sha256(_canonical_json(sorted(item_keys[family])))
                for family in FAMILY_ORDER
            },
        }
    multi = [indices for indices in groups.values() if len(indices) > 1]
    return {
        "component_graph": {
            "candidate_row_count": len(candidates),
            "component_count": len(groups),
            "multi_row_component_count": len(multi),
            "row_count_in_multi_row_components": sum(len(indices) for indices in multi),
            "cross_split_component_count": len(cross_split_roots),
            "documentation_example_component_count": len(documentation_roots),
        },
        "candidate_splits": split_output,
    }, flow_inputs


def _audit_public(audit: _SplitAudit) -> dict[str, Any]:
    return {
        "line_count": audit.line_count,
        "reader_and_formal_valid_candidate_count": audit.reader_valid_row_count,
        "reader_incompatible_reason_counts": dict(audit.invalid_reason_counts),
        "formal_ineligible_reason_counts": dict(audit.ineligible_reason_counts),
        "context_size_histogram": {
            str(key): value for key, value in audit.context_size_histogram.items()
        },
        "proof_leaf_count_histogram": {
            str(key): value for key, value in audit.proof_leaf_count_histogram.items()
        },
        "proof_step_count_histogram": {
            str(key): value for key, value in audit.proof_step_count_histogram.items()
        },
        "distractor_count_histogram": {
            str(key): value for key, value in audit.distractor_count_histogram.items()
        },
    }


def qualify_decoded_sources(
    train_raw: bytes,
    dev_raw: bytes,
    *,
    source_binding: Mapping[str, Any],
    train_demands: Mapping[str, int] = TRAIN_DEMANDS,
    dev_demands: Mapping[str, int] = DEV_DEMANDS,
) -> dict[str, Any]:
    audits = (
        _audit_split(train_raw, split="train"),
        _audit_split(dev_raw, split="dev"),
    )
    components, flow_inputs = _component_audit(audits)
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
    passed = total_assigned == total_required
    body = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": (
            "qualified_source_capacity_no_selection"
            if passed
            else "terminal_source_infeasible_no_selection"
        ),
        "source_binding": dict(source_binding),
        "split_aggregates": {
            audit.split: _audit_public(audit) for audit in audits
        },
        "candidate_and_component_aggregates": components,
        "simultaneous_component_disjoint_capacity": flows,
        "terminal_reason_counts": {
            "unsatisfied_capacity_count": total_required - total_assigned,
        },
        "claim_boundary": {
            "qualification_only_no_efficacy_claim": True,
            "selection_secret_generated_or_opened": False,
            "item_selected_or_materialized": False,
            "retrieval_action_evaluator_classifier_or_score_run": False,
            "online_or_external_evaluation_used": False,
            "test_payload_hashed_opened_or_parsed": False,
            "item_ID_question_answer_hypothesis_fact_proof_or_per_item_record_emitted": False,
        },
    }
    return _self_hashed(body, "qualification_sha256")


def _git_output(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _verify_self_hash(
    value: Mapping[str, Any], *, field: str, expected: str
) -> None:
    body = dict(value)
    claimed = body.pop(field, None)
    if claimed != expected or _sha256(_canonical_json(body)) != expected:
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


def verify_source_identity(root: Path) -> Mapping[str, Any]:
    source_root = root / SOURCE_REPOSITORY_RELATIVE_PATH
    if _git_output(source_root, "rev-parse", "HEAD") != SOURCE_REPOSITORY_COMMIT:
        raise FormalProvenanceError("official source commit drifted")
    if _git_output(source_root, "status", "--short"):
        raise FormalProvenanceError("official source worktree drifted")
    bindings: dict[str, Any] = {}
    for split, spec in SOURCE_SPECS.items():
        path = source_root / spec["relative_path"]
        if path.stat().st_size != spec["byte_size"]:
            raise FormalProvenanceError("official source size drifted")
        if _file_sha256(path) != spec["sha256"]:
            raise FormalProvenanceError("official source SHA256 drifted")
        blob = _git_output(source_root, "hash-object", str(spec["relative_path"]))
        if blob != spec["git_blob_sha1"]:
            raise FormalProvenanceError("official source Git blob drifted")
        bindings[split] = {
            "byte_size": spec["byte_size"],
            "git_blob_sha1": spec["git_blob_sha1"],
            "sha256": spec["sha256"],
            "relative_path_in_fixed_repository": spec[
                "relative_path"
            ].as_posix(),
        }
    return bindings


def verify_formal_provenance(project_root: Path) -> Mapping[str, Any]:
    root = project_root / "reconstruction_v2"
    for relative, file_hash, self_field, self_hash in (
        (
            CUSTODY_RELATIVE_PATH,
            CUSTODY_FILE_SHA256,
            "source_custody_sha256",
            CUSTODY_SELF_SHA256,
        ),
        (
            ACCESS_RELATIVE_PATH,
            ACCESS_FILE_SHA256,
            "source_access_sha256",
            ACCESS_SELF_SHA256,
        ),
        (
            DESIGN_RELATIVE_PATH,
            DESIGN_FILE_SHA256,
            "design_sha256",
            DESIGN_SELF_SHA256,
        ),
    ):
        _load_bound_manifest(
            root / relative,
            expected_file_sha256=file_hash,
            self_field=self_field,
            expected_self_sha256=self_hash,
        )
    freeze_raw = (root / FREEZE_RELATIVE_PATH).read_bytes()
    freeze = _strict_json(freeze_raw, public_label="implementation freeze")
    if not isinstance(freeze, Mapping) or freeze.get("schema") != FREEZE_SCHEMA:
        raise FormalProvenanceError("implementation freeze schema drifted")
    claimed = freeze.get("implementation_freeze_sha256")
    if not isinstance(claimed, str):
        raise FormalProvenanceError("implementation freeze self hash absent")
    _verify_self_hash(
        freeze,
        field="implementation_freeze_sha256",
        expected=claimed,
    )
    if freeze.get("status") != "frozen_before_first_Task2_TRAIN_or_DEV_row_parse":
        raise FormalProvenanceError("implementation freeze status drifted")
    implementation_commit = freeze.get("implementation_commit")
    if (
        not isinstance(implementation_commit, str)
        or len(implementation_commit) != 40
        or any(character not in "0123456789abcdef" for character in implementation_commit)
    ):
        raise FormalProvenanceError("implementation commit drifted")
    if _git_output(project_root, "cat-file", "-t", implementation_commit) != "commit":
        raise FormalProvenanceError("implementation commit unavailable")
    required = (
        ("custody", CUSTODY_RELATIVE_PATH, CUSTODY_FILE_SHA256),
        ("access", ACCESS_RELATIVE_PATH, ACCESS_FILE_SHA256),
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
        path = root / relative
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
    tracked = [
        str(Path("reconstruction_v2") / relative)
        for _role, relative, _fixed in required
    ] + [str(Path("reconstruction_v2") / FREEZE_RELATIVE_PATH)]
    if _git_output(project_root, "status", "--porcelain", "--", *tracked):
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
    source_bindings = verify_source_identity(root)
    attempt_root = root / ATTEMPT_RELATIVE_PATH
    result_path = root / RESULT_RELATIVE_PATH
    if attempt_root.exists() or result_path.exists():
        raise OneShotRefusal("formal qualification path is not pristine")
    attempt_root.mkdir(parents=True, mode=0o700)
    os.chmod(attempt_root, 0o700)
    marker = _self_hashed(
        {
            "schema": ATTEMPT_SCHEMA,
            "version": VERSION,
            "custody_sha256": CUSTODY_SELF_SHA256,
            "source_access_sha256": ACCESS_SELF_SHA256,
            "design_sha256": DESIGN_SELF_SHA256,
            "implementation_freeze_sha256": freeze[
                "implementation_freeze_sha256"
            ],
            "selection_secret_generated_or_opened": False,
            "test_payload_hashed_opened_or_parsed": False,
        },
        "attempt_sha256",
    )
    _write_json_once(attempt_root / "attempt.json", marker, mode=0o600)
    source_root = root / SOURCE_REPOSITORY_RELATIVE_PATH
    source_binding = {
        "official_repository_commit": SOURCE_REPOSITORY_COMMIT,
        "source_files": source_bindings,
        "custody_sha256": CUSTODY_SELF_SHA256,
        "source_access_sha256": ACCESS_SELF_SHA256,
        "design_sha256": DESIGN_SELF_SHA256,
        "implementation_freeze_sha256": freeze[
            "implementation_freeze_sha256"
        ],
        "test_payload_hash_open_parse_count": 0,
    }
    receipt = qualify_decoded_sources(
        (source_root / SOURCE_SPECS["train"]["relative_path"]).read_bytes(),
        (source_root / SOURCE_SPECS["dev"]["relative_path"]).read_bytes(),
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
        raise EntailmentBankQualificationError(
            "only the frozen formal path is exposed"
        )
    receipt = run_formal(args.project_root)
    print(receipt["status"])
    print(receipt["qualification_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
