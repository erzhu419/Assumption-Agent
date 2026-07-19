"""Embedded, aggregate-only validator for the frozen HybridQA source.

This module is deliberately a reader/validator rather than a standalone
qualification gate.  A future direct acquisition may call
``qualify_official_source`` once, before creating or reading a selection
secret, and embed the returned receipt in its own acquisition receipt.  The
module has no CLI, attempt marker, persistence path, retry mechanism, or
selection/scoring behavior.

Only fixed-schema aggregates and code/file-set hashes leave the validator.
Question IDs, table IDs, questions, answers, titles, passage text, cell text,
URLs, per-row hashes, and linkable row diagnostics remain transient.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from typing import Any


VERSION = "hybridqa_source_qualification_v1"
SCHEMA = VERSION
SOURCE_RELEASE = "HybridQA_official_table_linked_passage_source_v1"
QUALIFICATION_CLASS = "embedded_pre_secret_aggregate_source_validator"

FORMAL_SOURCE_ROOT_RELATIVE = Path("artifacts/hybridqa_official_source_v1")
FORMAL_HYBRIDQA_RELATIVE = Path("HybridQA")
FORMAL_WIKITABLES_RELATIVE = Path("WikiTables-WithLinks")

FORMAL_HYBRIDQA_COMMIT = "db22fda8c5951438fade3c69d75b350335ba93b3"
FORMAL_HYBRIDQA_TREE = "1e1ef6a6168ef6c6cf362264d8f7b75859ce8fdf"
FORMAL_WIKITABLES_COMMIT = "dc066e1a6d5281511d8b73a6107d5ad2824cc2b2"
FORMAL_WIKITABLES_TREE = "b4f2d5e0eeb2d18cf95bf6e6a583bc499c53b68c"

QA_RELATIVE_PATHS = {
    "train": PurePosixPath("released_data/train.json"),
    "train_traced": PurePosixPath("released_data/train.traced.json"),
    "dev": PurePosixPath("released_data/dev.json"),
    "dev_traced": PurePosixPath("released_data/dev.traced.json"),
    "test": PurePosixPath("released_data/test.json"),
    "dev_reference": PurePosixPath("released_data/dev_reference.json"),
}

FORMAL_QA_COUNTS = {"train": 62_682, "dev": 3_466, "test": 3_463}
FORMAL_CORPUS_COUNT = 15_316
FORMAL_DEV_REFERENCE_PARTITION = {
    "table": 1_349,
    "passage": 2_025,
    "computed": 92,
}

RAW_LABELED_FIELDS = frozenset(
    {"answer-text", "question", "question_id", "question_postag", "table_id"}
)
RAW_TEST_FIELDS = frozenset(
    {"question", "question_id", "question_postag", "table_id"}
)
TRACED_FIELDS = RAW_LABELED_FIELDS | {"answer-node"}
TABLE_FIELDS = frozenset(
    {
        "data",
        "header",
        "intro",
        "section_text",
        "section_title",
        "title",
        "uid",
        "url",
    }
)
DEV_REFERENCE_FIELDS = frozenset({"reference", "table", "passage"})

_HEX40_RE = re.compile(r"[0-9a-f]{40}\Z")
_HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")


class HybridQaSourceQualificationError(RuntimeError):
    """The frozen source or aggregate-only validation contract drifted."""


@dataclass(frozen=True)
class _AnswerNode:
    """Transient private node; never serialized into the public receipt."""

    answer: str
    row: int
    column: int
    link: str | None
    source: str


@dataclass(frozen=True)
class _QaAudit:
    qa_counts: dict[str, int]
    qid_count: int
    referenced_table_ids: frozenset[str]
    answer_nodes_by_table: Mapping[str, tuple[_AnswerNode, ...]]
    answer_node_source_counts: dict[str, int]
    empty_answer_node_rows: dict[str, int]
    dev_rows_by_qid: Mapping[str, Mapping[str, Any]]


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise HybridQaSourceQualificationError(
            "value is not canonical JSON"
        ) from exc


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_regular_file(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise HybridQaSourceQualificationError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise HybridQaSourceQualificationError(
            f"{label} must be a non-symlink regular file"
        )
    return metadata


def _require_directory(path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise HybridQaSourceQualificationError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise HybridQaSourceQualificationError(
            f"{label} must be a non-symlink directory"
        )


def _sha256_file(path: Path, *, label: str) -> str:
    _require_regular_file(path, label=label)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise HybridQaSourceQualificationError(f"{label} hashing failed") from exc
    return digest.hexdigest()


def _decode_strict_json(raw: bytes, *, label: str) -> Any:
    def reject_constant(_value: str) -> None:
        raise HybridQaSourceQualificationError(
            f"{label} contains a non-finite number"
        )

    def pairs_hook(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise HybridQaSourceQualificationError(
                    f"{label} contains duplicate JSON keys"
                )
            result[key] = value
        return result

    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except HybridQaSourceQualificationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise HybridQaSourceQualificationError(f"{label} is not strict JSON") from exc


def _read_strict_json(path: Path, *, label: str) -> Any:
    before = _require_regular_file(path, label=label)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise HybridQaSourceQualificationError(f"{label} read failed") from exc
    after = _require_regular_file(path, label=label)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or len(raw) != before.st_size:
        raise HybridQaSourceQualificationError(f"{label} changed while being read")
    return _decode_strict_json(raw, label=label)


def _git_command(repo: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", "-C", os.fspath(repo), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise HybridQaSourceQualificationError("git is unavailable") from exc


def _git_text(repo: Path, *arguments: str, label: str) -> str:
    result = _git_command(repo, *arguments)
    if result.returncode != 0:
        raise HybridQaSourceQualificationError(f"{label} failed")
    try:
        return result.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise HybridQaSourceQualificationError(f"{label} output is invalid") from exc


def _tracked_paths(repo: Path) -> tuple[PurePosixPath, ...]:
    result = _git_command(repo, "ls-files", "-z", "--cached")
    if result.returncode != 0:
        raise HybridQaSourceQualificationError("tracked file enumeration failed")
    raw_paths = result.stdout.split(b"\0")
    if raw_paths and raw_paths[-1] == b"":
        raw_paths.pop()
    paths: list[PurePosixPath] = []
    for raw_path in raw_paths:
        try:
            value = os.fsdecode(raw_path)
        except UnicodeError as exc:
            raise HybridQaSourceQualificationError(
                "tracked file path is not decodable"
            ) from exc
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise HybridQaSourceQualificationError("tracked file path is unsafe")
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise HybridQaSourceQualificationError("tracked file paths are duplicated")
    return tuple(sorted(paths, key=lambda value: value.as_posix()))


def _verify_git_checkout(
    repo: Path,
    *,
    expected_commit: str,
    expected_tree: str,
    repository_label: str,
) -> dict[str, Any]:
    """Verify a pinned clean checkout and return aggregate custody metadata."""

    if _HEX40_RE.fullmatch(expected_commit) is None:
        raise HybridQaSourceQualificationError("expected commit is invalid")
    if _HEX40_RE.fullmatch(expected_tree) is None:
        raise HybridQaSourceQualificationError("expected tree is invalid")
    _require_directory(repo, label=f"{repository_label} checkout")
    resolved = repo.resolve(strict=True)
    top = _git_text(
        repo,
        "rev-parse",
        "--show-toplevel",
        label=f"{repository_label} repository-root verification",
    )
    if Path(top).resolve(strict=True) != resolved:
        raise HybridQaSourceQualificationError(
            f"{repository_label} checkout is not the repository root"
        )
    commit = _git_text(
        repo,
        "rev-parse",
        "--verify",
        "HEAD",
        label=f"{repository_label} commit verification",
    )
    tree = _git_text(
        repo,
        "rev-parse",
        "--verify",
        "HEAD^{tree}",
        label=f"{repository_label} tree verification",
    )
    if commit != expected_commit or tree != expected_tree:
        raise HybridQaSourceQualificationError(
            f"{repository_label} pinned identity drifted"
        )
    status = _git_command(
        repo,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    if status.returncode != 0 or status.stdout:
        raise HybridQaSourceQualificationError(
            f"{repository_label} checkout is not clean"
        )
    unstaged = _git_command(repo, "diff", "--quiet", "--no-ext-diff", "HEAD", "--")
    staged = _git_command(repo, "diff", "--cached", "--quiet", "--no-ext-diff", "HEAD", "--")
    if unstaged.returncode != 0 or staged.returncode != 0:
        raise HybridQaSourceQualificationError(
            f"{repository_label} checkout content drifted"
        )
    tracked = _tracked_paths(repo)
    return {
        "commit": commit,
        "tree": tree,
        "tracked_file_count": len(tracked),
        "tracked_file_set_sha256": _stable_hash(
            [value.as_posix() for value in tracked]
        ),
    }


def _require_text(value: Any, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise HybridQaSourceQualificationError(f"{label} must be text")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise HybridQaSourceQualificationError(f"{label} is invalid Unicode") from exc
    if not allow_empty and not value.strip():
        raise HybridQaSourceQualificationError(f"{label} must be nonempty")
    return value


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HybridQaSourceQualificationError(f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        raise HybridQaSourceQualificationError(f"{label} fields drifted")


def _validate_raw_rows(
    payload: Any,
    *,
    split: str,
    expected_count: int,
    labeled: bool,
) -> tuple[list[Mapping[str, Any]], set[str], set[str]]:
    if not isinstance(payload, list) or len(payload) != expected_count:
        raise HybridQaSourceQualificationError(f"{split} row count drifted")
    expected_fields = RAW_LABELED_FIELDS if labeled else RAW_TEST_FIELDS
    rows: list[Mapping[str, Any]] = []
    qids: set[str] = set()
    table_ids: set[str] = set()
    for row in payload:
        mapping = _require_mapping(row, label=f"{split} row")
        _require_exact_keys(mapping, expected_fields, label=f"{split} row")
        for field in expected_fields:
            text = _require_text(mapping.get(field), label=f"{split} {field}")
            if field == "question_id":
                if text in qids:
                    raise HybridQaSourceQualificationError(
                        f"{split} question IDs are not unique"
                    )
                qids.add(text)
            elif field == "table_id":
                if "/" in text or text in {".", ".."}:
                    raise HybridQaSourceQualificationError(
                        f"{split} table ID is not a safe filename stem"
                    )
                table_ids.add(text)
        rows.append(mapping)
    return rows, qids, table_ids


def _parse_answer_node(value: Any) -> _AnswerNode:
    if not isinstance(value, list) or len(value) != 4:
        raise HybridQaSourceQualificationError("answer node shape drifted")
    answer = _require_text(value[0], label="answer node answer")
    coordinate = value[1]
    if (
        not isinstance(coordinate, list)
        or len(coordinate) != 2
        or type(coordinate[0]) is not int
        or type(coordinate[1]) is not int
        or coordinate[0] < 0
        or coordinate[1] < 0
    ):
        raise HybridQaSourceQualificationError("answer node coordinate is invalid")
    source = value[3]
    if source == "table":
        if value[2] is not None:
            raise HybridQaSourceQualificationError(
                "table answer node must not declare a passage link"
            )
        link = None
    elif source == "passage":
        link = _require_text(value[2], label="passage answer node link")
    else:
        raise HybridQaSourceQualificationError("answer node source is invalid")
    return _AnswerNode(
        answer=answer,
        row=coordinate[0],
        column=coordinate[1],
        link=link,
        source=source,
    )


def _validate_traced_rows(
    payload: Any,
    *,
    split: str,
    raw_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[_AnswerNode]], Counter[str], int]:
    if not isinstance(payload, list) or len(payload) != len(raw_rows):
        raise HybridQaSourceQualificationError(f"{split} traced count drifted")
    by_table: dict[str, list[_AnswerNode]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    empty_answer_node_rows = 0
    for raw_row, traced_row in zip(raw_rows, payload, strict=True):
        mapping = _require_mapping(traced_row, label=f"{split} traced row")
        _require_exact_keys(mapping, TRACED_FIELDS, label=f"{split} traced row")
        raw_projection = {key: mapping[key] for key in RAW_LABELED_FIELDS}
        if raw_projection != dict(raw_row):
            raise HybridQaSourceQualificationError(
                f"{split} traced/raw fields disagree"
            )
        nodes = mapping.get("answer-node")
        if not isinstance(nodes, list):
            raise HybridQaSourceQualificationError(
                f"{split} traced answer nodes must be a list"
            )
        # Empty traces are source-native for computed/unresolved answers.  They
        # are counted but never made eligible by the downstream acquisition.
        if not nodes:
            empty_answer_node_rows += 1
        table_id = raw_row["table_id"]
        for value in nodes:
            node = _parse_answer_node(value)
            by_table[table_id].append(node)
            source_counts[node.source] += 1
    return by_table, source_counts, empty_answer_node_rows


def _validate_dev_reference(
    payload: Any,
    *,
    dev_rows: Sequence[Mapping[str, Any]],
    expected_partition: Mapping[str, int],
) -> tuple[dict[str, int], Mapping[str, Mapping[str, Any]]]:
    mapping = _require_mapping(payload, label="dev reference")
    _require_exact_keys(mapping, DEV_REFERENCE_FIELDS, label="dev reference")
    reference = _require_mapping(mapping.get("reference"), label="dev reference map")
    dev_by_qid = {row["question_id"]: row for row in dev_rows}
    if set(reference) != set(dev_by_qid):
        raise HybridQaSourceQualificationError("dev reference question IDs drifted")
    for qid, answer in reference.items():
        _require_text(qid, label="dev reference question ID")
        _require_text(answer, label="dev reference answer")
        if answer != dev_by_qid[qid]["answer-text"]:
            raise HybridQaSourceQualificationError("dev reference answer drifted")
    partitions: dict[str, set[str]] = {}
    for name in ("table", "passage"):
        values = mapping.get(name)
        if not isinstance(values, list):
            raise HybridQaSourceQualificationError(
                f"dev reference {name} partition must be a list"
            )
        observed: set[str] = set()
        for value in values:
            qid = _require_text(value, label=f"dev reference {name} question ID")
            if qid not in dev_by_qid or qid in observed:
                raise HybridQaSourceQualificationError(
                    f"dev reference {name} partition is invalid"
                )
            observed.add(qid)
        partitions[name] = observed
    if partitions["table"].intersection(partitions["passage"]):
        raise HybridQaSourceQualificationError("dev reference partitions overlap")
    computed = set(dev_by_qid).difference(
        partitions["table"], partitions["passage"]
    )
    observed_counts = {
        "table": len(partitions["table"]),
        "passage": len(partitions["passage"]),
        "computed": len(computed),
    }
    expected = {name: int(expected_partition[name]) for name in observed_counts}
    if observed_counts != expected:
        raise HybridQaSourceQualificationError(
            "dev reference partition counts drifted"
        )
    return observed_counts, dev_by_qid


def _audit_qa(
    payloads: Mapping[str, Any],
    *,
    expected_counts: Mapping[str, int],
    expected_dev_reference_partition: Mapping[str, int],
) -> tuple[_QaAudit, dict[str, int]]:
    if set(payloads) != set(QA_RELATIVE_PATHS):
        raise HybridQaSourceQualificationError("QA payload file set drifted")
    raw_rows: dict[str, list[Mapping[str, Any]]] = {}
    qid_sets: dict[str, set[str]] = {}
    table_ids: set[str] = set()
    for split in ("train", "dev", "test"):
        rows, qids, split_table_ids = _validate_raw_rows(
            payloads[split],
            split=split,
            expected_count=int(expected_counts[split]),
            labeled=split != "test",
        )
        raw_rows[split] = rows
        qid_sets[split] = qids
        table_ids.update(split_table_ids)
    if (
        qid_sets["train"].intersection(qid_sets["dev"])
        or qid_sets["train"].intersection(qid_sets["test"])
        or qid_sets["dev"].intersection(qid_sets["test"])
    ):
        raise HybridQaSourceQualificationError("QA split question IDs overlap")
    nodes_by_table: dict[str, list[_AnswerNode]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    empty_answer_node_rows: dict[str, int] = {}
    for split in ("train", "dev"):
        observed_nodes, observed_sources, empty_count = _validate_traced_rows(
            payloads[f"{split}_traced"],
            split=split,
            raw_rows=raw_rows[split],
        )
        for table_id, nodes in observed_nodes.items():
            nodes_by_table[table_id].extend(nodes)
        source_counts.update(observed_sources)
        empty_answer_node_rows[split] = empty_count
    dev_partition, dev_by_qid = _validate_dev_reference(
        payloads["dev_reference"],
        dev_rows=raw_rows["dev"],
        expected_partition=expected_dev_reference_partition,
    )
    qa_counts = {split: len(raw_rows[split]) for split in ("train", "dev", "test")}
    return (
        _QaAudit(
            qa_counts=qa_counts,
            qid_count=sum(qa_counts.values()),
            referenced_table_ids=frozenset(table_ids),
            answer_nodes_by_table={
                key: tuple(value) for key, value in nodes_by_table.items()
            },
            answer_node_source_counts={
                name: source_counts[name] for name in ("table", "passage")
            },
            empty_answer_node_rows=empty_answer_node_rows,
            dev_rows_by_qid=dev_by_qid,
        ),
        dev_partition,
    )


def _validate_cell(value: Any, *, label: str) -> tuple[str, tuple[str, ...]]:
    if not isinstance(value, list) or len(value) != 2:
        raise HybridQaSourceQualificationError(f"{label} shape drifted")
    text = _require_text(value[0], label=f"{label} text", allow_empty=True)
    raw_links = value[1]
    if not isinstance(raw_links, list):
        raise HybridQaSourceQualificationError(f"{label} links must be a list")
    links = tuple(
        _require_text(link, label=f"{label} link") for link in raw_links
    )
    return text, links


def _validate_request_map(value: Any) -> Mapping[str, str]:
    mapping = _require_mapping(value, label="request map")
    output: dict[str, str] = {}
    for key, passage in mapping.items():
        link = _require_text(key, label="request map link")
        output[link] = _require_text(passage, label="request map passage")
    return output


def _validated_source_custody(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != {
        "hybridqa",
        "wikitables_with_links",
        "clean_checkout_verified_before_and_after",
    } or value.get("clean_checkout_verified_before_and_after") is not True:
        raise HybridQaSourceQualificationError("formal source custody fields drifted")
    output: dict[str, Any] = {
        "clean_checkout_verified_before_and_after": True,
    }
    expected_identity = {
        "hybridqa": (FORMAL_HYBRIDQA_COMMIT, FORMAL_HYBRIDQA_TREE),
        "wikitables_with_links": (
            FORMAL_WIKITABLES_COMMIT,
            FORMAL_WIKITABLES_TREE,
        ),
    }
    expected_fields = {
        "commit",
        "tree",
        "tracked_file_count",
        "tracked_file_set_sha256",
    }
    for label, (expected_commit, expected_tree) in expected_identity.items():
        record = _require_mapping(value.get(label), label=f"{label} custody")
        if set(record) != expected_fields:
            raise HybridQaSourceQualificationError(
                f"{label} custody fields drifted"
            )
        commit = record.get("commit")
        tree = record.get("tree")
        count = record.get("tracked_file_count")
        file_set_hash = record.get("tracked_file_set_sha256")
        if (
            commit != expected_commit
            or tree != expected_tree
            or type(count) is not int
            or count <= 0
            or not isinstance(file_set_hash, str)
            or _HEX64_RE.fullmatch(file_set_hash) is None
        ):
            raise HybridQaSourceQualificationError(
                f"{label} custody identity is invalid"
            )
        output[label] = {
            "commit": commit,
            "tree": tree,
            "tracked_file_count": count,
            "tracked_file_set_sha256": file_set_hash,
        }
    return output


def _validate_table_and_nodes(
    *,
    table_id: str,
    table_value: Any,
    request_value: Any,
    answer_nodes: Sequence[_AnswerNode],
) -> dict[str, int]:
    table = _require_mapping(table_value, label="table")
    _require_exact_keys(table, TABLE_FIELDS, label="table")
    for field in ("intro", "section_text", "section_title", "title", "url"):
        _require_text(table.get(field), label=f"table {field}", allow_empty=True)
    uid = _require_text(table.get("uid"), label="table uid")
    if uid != table_id:
        raise HybridQaSourceQualificationError("table uid/filename stem drifted")
    request = _validate_request_map(request_value)

    raw_header = table.get("header")
    if not isinstance(raw_header, list) or not raw_header:
        raise HybridQaSourceQualificationError("table header is invalid")
    header = [
        _validate_cell(cell, label="table header cell") for cell in raw_header
    ]
    raw_data = table.get("data")
    if not isinstance(raw_data, list) or not raw_data:
        raise HybridQaSourceQualificationError("table data is invalid")
    data: list[list[tuple[str, tuple[str, ...]]]] = []
    for raw_row in raw_data:
        if not isinstance(raw_row, list) or len(raw_row) != len(header):
            raise HybridQaSourceQualificationError("table row width drifted")
        data.append(
            [_validate_cell(cell, label="table data cell") for cell in raw_row]
        )
    link_reference_count = 0
    for _text, links in [*header, *(cell for row in data for cell in row)]:
        link_reference_count += len(links)
        if any(link not in request for link in links):
            raise HybridQaSourceQualificationError(
                "table link is not exactly resolvable in request map"
            )
    source_counts: Counter[str] = Counter()
    for node in answer_nodes:
        if node.row >= len(data) or node.column >= len(data[node.row]):
            raise HybridQaSourceQualificationError(
                "answer node coordinate is outside table data"
            )
        cell_text, cell_links = data[node.row][node.column]
        if node.source == "table":
            if node.link is not None or node.answer != cell_text:
                raise HybridQaSourceQualificationError(
                    "table answer node does not resolve exactly to its cell"
                )
        elif (
            node.source == "passage"
            and (
                node.link is None
                or node.link not in cell_links
                or node.link not in request
            )
        ):
            raise HybridQaSourceQualificationError(
                "passage answer node link does not resolve at its cell"
            )
        source_counts[node.source] += 1
    return {
        "row_count": len(data),
        "cell_count": len(header) + sum(len(row) for row in data),
        "link_reference_count": link_reference_count,
        "request_entry_count": len(request),
        "table_answer_node_count": source_counts["table"],
        "passage_answer_node_count": source_counts["passage"],
    }


def qualify_decoded_sources(
    qa_payloads: Mapping[str, Any],
    corpus: Iterable[tuple[str, Any, Any]],
    *,
    expected_qa_counts: Mapping[str, int] = FORMAL_QA_COUNTS,
    expected_corpus_count: int = FORMAL_CORPUS_COUNT,
    expected_dev_reference_partition: Mapping[str, int] = (
        FORMAL_DEV_REFERENCE_PARTITION
    ),
    formal_identity_enforced: bool = False,
    source_custody: Mapping[str, Any] | None = None,
    qualification_code_sha256: str,
    qa_file_set_sha256: str,
    corpus_file_set_sha256: str,
) -> dict[str, Any]:
    """Validate decoded sources and return a row-free aggregate receipt.

    ``corpus`` yields ``(table_id, table_json, request_json)`` triples.  The
    identifiers and payloads are consumed only for exact reconciliation and
    are never copied into the returned receipt.
    """

    for value, label in (
        (qualification_code_sha256, "qualification code hash"),
        (qa_file_set_sha256, "QA file-set hash"),
        (corpus_file_set_sha256, "corpus file-set hash"),
    ):
        if _HEX64_RE.fullmatch(value) is None:
            raise HybridQaSourceQualificationError(f"{label} is invalid")
    if expected_corpus_count <= 0:
        raise HybridQaSourceQualificationError("expected corpus count is invalid")
    qa_audit, dev_partition = _audit_qa(
        qa_payloads,
        expected_counts=expected_qa_counts,
        expected_dev_reference_partition=expected_dev_reference_partition,
    )
    seen_table_ids: set[str] = set()
    aggregate: Counter[str] = Counter()
    for table_id_value, table_value, request_value in corpus:
        table_id = _require_text(table_id_value, label="corpus table ID")
        if table_id in seen_table_ids:
            raise HybridQaSourceQualificationError("corpus table IDs are duplicated")
        seen_table_ids.add(table_id)
        observed = _validate_table_and_nodes(
            table_id=table_id,
            table_value=table_value,
            request_value=request_value,
            answer_nodes=qa_audit.answer_nodes_by_table.get(table_id, ()),
        )
        aggregate.update(observed)
    if len(seen_table_ids) != expected_corpus_count:
        raise HybridQaSourceQualificationError("corpus table/request count drifted")
    if not qa_audit.referenced_table_ids.issubset(seen_table_ids):
        raise HybridQaSourceQualificationError(
            "a dataset table ID is absent from table/request corpus"
        )
    expected_node_counts = qa_audit.answer_node_source_counts
    if (
        aggregate["table_answer_node_count"] != expected_node_counts["table"]
        or aggregate["passage_answer_node_count"] != expected_node_counts["passage"]
    ):
        raise HybridQaSourceQualificationError(
            "not all answer nodes were reconciled against the corpus"
        )

    status = (
        "source_qualified_for_embedded_pre_secret_acquisition"
        if formal_identity_enforced
        else "synthetic_or_nonformal_aggregate_diagnostic"
    )
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "source_release": SOURCE_RELEASE,
        "qualification_class": QUALIFICATION_CLASS,
        "status": status,
        "formal_identity_enforced": formal_identity_enforced,
        "qualification_code_sha256": qualification_code_sha256,
        "file_sets": {
            "qa_required_file_count": len(QA_RELATIVE_PATHS),
            "qa_required_file_set_sha256": qa_file_set_sha256,
            "table_request_pair_count": len(seen_table_ids),
            "table_request_file_set_sha256": corpus_file_set_sha256,
        },
        "qa": {
            "train_row_count": qa_audit.qa_counts["train"],
            "dev_row_count": qa_audit.qa_counts["dev"],
            "test_row_count": qa_audit.qa_counts["test"],
            "question_id_count": qa_audit.qid_count,
            "question_ids_unique_within_splits": True,
            "question_id_splits_pairwise_disjoint": True,
            "train_traced_raw_exact_match": True,
            "dev_traced_raw_exact_match": True,
            "train_empty_answer_node_row_count": qa_audit.empty_answer_node_rows[
                "train"
            ],
            "dev_empty_answer_node_row_count": qa_audit.empty_answer_node_rows[
                "dev"
            ],
            "referenced_table_count": len(qa_audit.referenced_table_ids),
        },
        "dev_reference": {
            "question_id_count": qa_audit.qa_counts["dev"],
            "reference_answer_exact_match": True,
            "table_partition_count": dev_partition["table"],
            "passage_partition_count": dev_partition["passage"],
            "computed_partition_count": dev_partition["computed"],
            "partition_complete_and_disjoint": True,
        },
        "corpus": {
            "table_json_count": len(seen_table_ids),
            "request_json_count": len(seen_table_ids),
            "table_request_filename_sets_equal": True,
            "dataset_table_ids_exactly_resolved": True,
            "unused_table_count": len(seen_table_ids)
            - len(qa_audit.referenced_table_ids),
            "data_row_count": aggregate["row_count"],
            "header_and_data_cell_count": aggregate["cell_count"],
            "link_reference_count": aggregate["link_reference_count"],
            "request_entry_count": aggregate["request_entry_count"],
            "all_links_exactly_resolved": True,
        },
        "answer_nodes": {
            "answer_node_count": sum(expected_node_counts.values()),
            "table_source_count": expected_node_counts["table"],
            "passage_source_count": expected_node_counts["passage"],
            "sources_coordinates_and_links_valid": True,
        },
        "safeguards": {
            "pre_design_programmatic_audit_occurred": True,
            "pre_design_programmatic_audit_raw_output_count": 0,
            "raw_record_output_count": 0,
            "per_row_or_linkable_hash_output_count": 0,
            "selection_secret_created_or_read_count": 0,
            "selection_or_hmac_count": 0,
            "action_or_retrieval_count": 0,
            "score_or_utility_count": 0,
            "dev_test_online_evaluator_count": 0,
            "standalone_qualification_manifest_persisted_count": 0,
        },
    }
    if formal_identity_enforced:
        if source_custody is None:
            raise HybridQaSourceQualificationError(
                "formal receipt requires source custody"
            )
        body["source_custody"] = _validated_source_custody(source_custody)
    elif source_custody is not None:
        raise HybridQaSourceQualificationError(
            "nonformal receipt must not assert formal source custody"
        )
    receipt = dict(body)
    receipt["receipt_sha256"] = _stable_hash(body)
    # Construction is fixed-schema and contains no source-derived string.  A
    # successful canonicalization is retained as a final fail-closed check.
    _canonical_json(receipt)
    return receipt


def _official_corpus_items(
    *,
    wiki_root: Path,
    table_paths: Mapping[str, Path],
    request_paths: Mapping[str, Path],
) -> Iterable[tuple[str, Any, Any]]:
    del wiki_root  # Documents that only the paired, pinned paths are consumed.
    for table_id in sorted(table_paths):
        yield (
            table_id,
            _read_strict_json(table_paths[table_id], label="table JSON"),
            _read_strict_json(request_paths[table_id], label="request JSON"),
        )


def _official_data_paths(
    *,
    wiki_root: Path,
    tracked_paths: Sequence[PurePosixPath],
) -> tuple[dict[str, Path], dict[str, Path], str]:
    _require_directory(wiki_root / "tables_tok", label="tables_tok directory")
    _require_directory(wiki_root / "request_tok", label="request_tok directory")
    table_relatives = {
        path
        for path in tracked_paths
        if len(path.parts) == 2
        and path.parts[0] == "tables_tok"
        and path.suffix == ".json"
    }
    request_relatives = {
        path
        for path in tracked_paths
        if len(path.parts) == 2
        and path.parts[0] == "request_tok"
        and path.suffix == ".json"
    }
    if (
        len(table_relatives) != FORMAL_CORPUS_COUNT
        or len(request_relatives) != FORMAL_CORPUS_COUNT
    ):
        raise HybridQaSourceQualificationError(
            "tracked table/request JSON counts drifted"
        )
    table_ids = {path.name[:-5] for path in table_relatives}
    request_ids = {path.name[:-5] for path in request_relatives}
    if table_ids != request_ids or len(table_ids) != FORMAL_CORPUS_COUNT:
        raise HybridQaSourceQualificationError(
            "tracked table/request filename sets drifted"
        )
    filesystem_json = {
        PurePosixPath(path.relative_to(wiki_root).as_posix())
        for directory in (wiki_root / "tables_tok", wiki_root / "request_tok")
        for path in directory.rglob("*.json")
    }
    expected_json = table_relatives | request_relatives
    if filesystem_json != expected_json:
        raise HybridQaSourceQualificationError(
            "working-tree table/request JSON file set drifted"
        )
    tables: dict[str, Path] = {}
    requests: dict[str, Path] = {}
    for table_id in table_ids:
        table_path = wiki_root / "tables_tok" / f"{table_id}.json"
        request_path = wiki_root / "request_tok" / f"{table_id}.json"
        _require_regular_file(table_path, label="table JSON")
        _require_regular_file(request_path, label="request JSON")
        tables[table_id] = table_path
        requests[table_id] = request_path
    file_set_hash = _stable_hash(
        sorted(path.as_posix() for path in expected_json)
    )
    return tables, requests, file_set_hash


def qualify_official_source(project_root: Path) -> dict[str, Any]:
    """Exhaust the pinned official source and return an embeddable receipt.

    The caller must invoke this before creating or reading any acquisition
    secret.  This function performs no persistence and should not be exposed
    as a separately repeatable formal CLI.
    """

    project = Path(project_root).resolve(strict=True)
    source_root = project / FORMAL_SOURCE_ROOT_RELATIVE
    hybrid_root = source_root / FORMAL_HYBRIDQA_RELATIVE
    wiki_root = source_root / FORMAL_WIKITABLES_RELATIVE
    before_hybrid = _verify_git_checkout(
        hybrid_root,
        expected_commit=FORMAL_HYBRIDQA_COMMIT,
        expected_tree=FORMAL_HYBRIDQA_TREE,
        repository_label="HybridQA",
    )
    before_wiki = _verify_git_checkout(
        wiki_root,
        expected_commit=FORMAL_WIKITABLES_COMMIT,
        expected_tree=FORMAL_WIKITABLES_TREE,
        repository_label="WikiTables-WithLinks",
    )
    hybrid_tracked = set(_tracked_paths(hybrid_root))
    required_qa_paths = set(QA_RELATIVE_PATHS.values())
    if not required_qa_paths.issubset(hybrid_tracked):
        raise HybridQaSourceQualificationError(
            "required official QA files are not tracked"
        )
    qa_payloads: dict[str, Any] = {}
    for label, relative in QA_RELATIVE_PATHS.items():
        qa_payloads[label] = _read_strict_json(
            hybrid_root / Path(relative.as_posix()),
            label=f"official {label} JSON",
        )
    wiki_tracked = _tracked_paths(wiki_root)
    table_paths, request_paths, corpus_file_set_sha256 = _official_data_paths(
        wiki_root=wiki_root,
        tracked_paths=wiki_tracked,
    )
    source_custody = {
        "hybridqa": before_hybrid,
        "wikitables_with_links": before_wiki,
        "clean_checkout_verified_before_and_after": True,
    }
    receipt = qualify_decoded_sources(
        qa_payloads,
        _official_corpus_items(
            wiki_root=wiki_root,
            table_paths=table_paths,
            request_paths=request_paths,
        ),
        formal_identity_enforced=True,
        source_custody=source_custody,
        qualification_code_sha256=_sha256_file(
            Path(__file__).resolve(strict=True), label="qualification code"
        ),
        qa_file_set_sha256=_stable_hash(
            sorted(path.as_posix() for path in required_qa_paths)
        ),
        corpus_file_set_sha256=corpus_file_set_sha256,
    )
    after_hybrid = _verify_git_checkout(
        hybrid_root,
        expected_commit=FORMAL_HYBRIDQA_COMMIT,
        expected_tree=FORMAL_HYBRIDQA_TREE,
        repository_label="HybridQA",
    )
    after_wiki = _verify_git_checkout(
        wiki_root,
        expected_commit=FORMAL_WIKITABLES_COMMIT,
        expected_tree=FORMAL_WIKITABLES_TREE,
        repository_label="WikiTables-WithLinks",
    )
    if before_hybrid != after_hybrid or before_wiki != after_wiki:
        raise HybridQaSourceQualificationError(
            "source custody changed during qualification"
        )
    return receipt


__all__ = [
    "FORMAL_CORPUS_COUNT",
    "FORMAL_DEV_REFERENCE_PARTITION",
    "FORMAL_HYBRIDQA_COMMIT",
    "FORMAL_HYBRIDQA_TREE",
    "FORMAL_QA_COUNTS",
    "FORMAL_WIKITABLES_COMMIT",
    "FORMAL_WIKITABLES_TREE",
    "HybridQaSourceQualificationError",
    "qualify_decoded_sources",
    "qualify_official_source",
]
