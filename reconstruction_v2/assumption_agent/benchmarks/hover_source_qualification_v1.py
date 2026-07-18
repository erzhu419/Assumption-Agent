"""One-shot aggregate-only qualification for the frozen HoVer TRAIN source.

The formal CLI consumes an exclusive marker before parsing either the TRAIN
JSON or the official SQLite corpus.  Its only public output is an aggregate
receipt: no UID, hpqa_id, claim, title, body, support sentence, per-row hash,
or selected cohort member is serialized.  Synthetic tests call
``qualify_payload`` directly and never touch the formal source.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import stat
import subprocess
from typing import Any
import unicodedata


VERSION = "hover_source_qualification_v1"
SCHEMA = VERSION

FORMAL_TRAIN_RELATIVE = Path(
    "artifacts/hover_official_source_v1/"
    "hover_train_release_v1.1-39b84697.json"
)
FORMAL_DB_RELATIVE = Path(
    "artifacts/hover_official_source_v1/wiki_wo_links.db"
)
FORMAL_MARKER_RELATIVE = Path(
    "artifacts/hover_official_source_v1/"
    "source_qualification_attempt_v1.marker"
)
FORMAL_OUTPUT_RELATIVE = Path("manifests/hover_source_qualification_v1.json")
FORMAL_DESIGN_RELATIVE = Path(
    "manifests/hover_joint_graph_evaluator_design_v1.json"
)

FORMAL_TRAIN_SIZE = 9_205_582
FORMAL_TRAIN_SHA256 = (
    "1f1cd57abd616fa00c70bdc575ce77c16fc6cf1a6cffd5ff87c208030a336bb6"
)
FORMAL_TRAIN_GIT_BLOB_SHA1 = "49a36d7eb2f319329264b546cb687f54b8f1990f"
FORMAL_DB_SIZE = 2_156_273_664
FORMAL_DB_SHA256 = (
    "c37ee397916ec0bffacfe8902db454a5cda88a7a188409217b2e15231fe5ee2f"
)
FORMAL_GIT_COMMIT = "39b84697f196308f398a251a7aea9b82ae0f0562"
FORMAL_DESIGN_SHA256 = (
    "e558d5305af5a31953a9d87ef92d7cc8d6c4ee48fc82d89eb52e4355826ca818"
)
FORMAL_DESIGN_FILE_SHA256 = (
    "6e8a493cdbd1662eef7af7158d3275fdaef2c8f48b089f19888a11f25b98c9ad"
)
FORMAL_DESIGN_GIT_BLOB_SHA1 = "6868c7cda2209b0c9bcae6628d8820aee6b962e4"

FORMAL_ROW_COUNT = 18_171
HOP_ORDER = (2, 3, 4)
FORMAL_HOP_COUNTS = {2: 9_052, 3: 6_084, 4: 3_035}
TARGET_GROUPS_PER_HOP = 48
MINIMUM_CORPUS_ROWS = 609
REQUIRED_FIELDS = frozenset(
    {"uid", "hpqa_id", "claim", "num_hops", "supporting_facts"}
)

_WHITESPACE_RE = re.compile(r"\s+")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class HoVerSourceQualificationError(RuntimeError):
    """The frozen source or one-shot qualification contract drifted."""


@dataclass(frozen=True)
class _Candidate:
    uid: str
    hpqa_id: str
    normalized_claim: str
    hop: int
    gold_rowids: tuple[int, ...]


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
        raise HoVerSourceQualificationError("value is not canonical JSON") from exc


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {len(raw)}\0".encode("ascii"))
    digest.update(raw)
    return digest.hexdigest()


def _decode_strict_json(raw: bytes) -> object:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise HoVerSourceQualificationError("TRAIN is not strict UTF-8") from exc

    def object_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise HoVerSourceQualificationError("duplicate JSON object key")
            output[key] = value
        return output

    def reject_constant(_value: str) -> None:
        raise HoVerSourceQualificationError("JSON contains a nonfinite constant")

    try:
        return json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except HoVerSourceQualificationError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise HoVerSourceQualificationError("TRAIN is not strict JSON") from exc


def _normalize_claim(value: str) -> str:
    return _WHITESPACE_RE.sub(
        " ", unicodedata.normalize("NFKC", value)
    ).strip().casefold()


def _nfd_title(value: str) -> str:
    return unicodedata.normalize("NFD", value)


def _counter_payload(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter, key=str)}


def _require_sqlite_schema(connection: sqlite3.Connection) -> dict[str, int | str]:
    try:
        table_rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        columns = connection.execute("PRAGMA table_info(documents)").fetchall()
        aggregate = connection.execute(
            "SELECT COUNT(*), MIN(rowid), MAX(rowid) FROM documents"
        ).fetchone()
    except sqlite3.Error as exc:
        raise HoVerSourceQualificationError("SQLite schema is unavailable") from exc
    table_names = [row[0] for row in table_rows if len(row) == 1]
    if "documents" not in table_names or not columns or aggregate is None:
        raise HoVerSourceQualificationError("documents table is unavailable")
    names = [row[1] for row in columns if len(row) >= 2]
    if "id" not in names or "text" not in names:
        raise HoVerSourceQualificationError("documents id/text columns are absent")
    row_count, minimum_rowid, maximum_rowid = aggregate
    if (
        type(row_count) is not int
        or row_count < MINIMUM_CORPUS_ROWS
        or type(minimum_rowid) is not int
        or type(maximum_rowid) is not int
        or not 1 <= minimum_rowid <= maximum_rowid
    ):
        raise HoVerSourceQualificationError("documents rowid aggregate is invalid")
    return {
        "column_count": len(names),
        "column_name_set_sha256": _stable_hash(sorted(names)),
        "maximum_rowid": maximum_rowid,
        "minimum_rowid": minimum_rowid,
        "row_count": row_count,
        "table_count": len(table_names),
        "table_name_set_sha256": _stable_hash(sorted(table_names)),
    }


def _resolve_title(
    connection: sqlite3.Connection,
    title: str,
) -> tuple[str, tuple[int, str, str] | None]:
    key = _nfd_title(title)
    try:
        rows = connection.execute(
            "SELECT rowid, id, text FROM documents WHERE id = ?", (key,)
        ).fetchall()
    except sqlite3.Error as exc:
        raise HoVerSourceQualificationError("gold title lookup failed") from exc
    if not rows:
        return "missing", None
    if len(rows) != 1:
        return "ambiguous", None
    rowid, observed_title, body = rows[0]
    if (
        type(rowid) is not int
        or not isinstance(observed_title, str)
        or observed_title != key
        or not observed_title.strip()
        or "\x00" in observed_title
        or not isinstance(body, str)
        or not body.strip()
        or "\x00" in body
    ):
        return "invalid_document", None
    return "resolved", (rowid, observed_title, body)


def _hall_capacity(
    candidates: Sequence[_Candidate],
) -> tuple[bool, dict[str, int], dict[str, int], int]:
    group_options: dict[str, set[int]] = defaultdict(set)
    for candidate in candidates:
        group_options[candidate.hpqa_id].add(candidate.hop)
    neighborhood_counts: dict[str, int] = {}
    shortfalls: dict[str, int] = {}
    for width in range(1, len(HOP_ORDER) + 1):
        for subset in itertools.combinations(HOP_ORDER, width):
            label = "+".join(str(value) for value in subset)
            neighborhood = sum(
                bool(options.intersection(subset))
                for options in group_options.values()
            )
            required = TARGET_GROUPS_PER_HOP * len(subset)
            neighborhood_counts[label] = neighborhood
            shortfalls[label] = max(0, required - neighborhood)
    return (
        all(value == 0 for value in shortfalls.values()),
        neighborhood_counts,
        shortfalls,
        len(group_options),
    )


def qualify_payload(
    payload: object,
    connection: sqlite3.Connection,
    *,
    expected_row_count: int = FORMAL_ROW_COUNT,
    expected_hop_counts: Mapping[int, int] = FORMAL_HOP_COUNTS,
    source_size: int,
    source_sha256: str,
    source_git_blob_sha1: str,
    db_size: int,
    db_sha256: str,
    formal_identity_enforced: bool,
) -> dict[str, Any]:
    """Qualify one decoded source and an already read-only SQLite connection."""

    if not isinstance(payload, list) or len(payload) != expected_row_count:
        raise HoVerSourceQualificationError("TRAIN root count drifted")
    db_schema = _require_sqlite_schema(connection)
    hop_counts: Counter[int] = Counter()
    keyset_counts: Counter[str] = Counter()
    extra_field_names: set[str] = set()
    seen_uids: set[str] = set()
    normalized_claim_counts: Counter[str] = Counter()
    candidates_before_claim_exclusion: list[_Candidate] = []
    structural_exclusions: Counter[str] = Counter()
    resolution_counts: Counter[str] = Counter()
    gold_cardinality_counts: Counter[int] = Counter()
    support_pair_count = 0
    title_cache: dict[str, tuple[str, tuple[int, str, str] | None]] = {}
    serialization_by_rowid: dict[int, str] = {}

    for raw in payload:
        if not isinstance(raw, Mapping) or not REQUIRED_FIELDS <= set(raw):
            raise HoVerSourceQualificationError("TRAIN required fields drifted")
        keys = sorted(str(key) for key in raw)
        if len(keys) != len(raw) or any(not isinstance(key, str) for key in raw):
            raise HoVerSourceQualificationError("TRAIN object key type drifted")
        keyset_counts[_stable_hash(keys)] += 1
        extra_field_names.update(set(raw) - REQUIRED_FIELDS)
        uid = raw.get("uid")
        hpqa_id = raw.get("hpqa_id")
        claim = raw.get("claim")
        hop = raw.get("num_hops")
        supporting = raw.get("supporting_facts")
        if (
            not isinstance(uid, str)
            or not isinstance(hpqa_id, str)
            or not isinstance(claim, str)
            or type(hop) is not int
            or hop not in HOP_ORDER
            or not isinstance(supporting, list)
        ):
            raise HoVerSourceQualificationError("TRAIN consumed field type drifted")
        if uid in seen_uids:
            raise HoVerSourceQualificationError("TRAIN uid is duplicated")
        seen_uids.add(uid)
        hop_counts[hop] += 1
        normalized_claim = _normalize_claim(claim)
        normalized_claim_counts[normalized_claim] += 1

        invalid_reason: str | None = None
        if not uid.strip():
            invalid_reason = "empty_uid"
        elif not hpqa_id.strip():
            invalid_reason = "empty_hpqa_id"
        elif not normalized_claim:
            invalid_reason = "empty_claim"
        elif not supporting:
            invalid_reason = "empty_supporting_facts"

        titles: list[str] = []
        if invalid_reason is None:
            for pair in supporting:
                if (
                    not isinstance(pair, list)
                    or len(pair) != 2
                    or not isinstance(pair[0], str)
                    or type(pair[1]) is not int
                ):
                    raise HoVerSourceQualificationError(
                        "supporting_facts pair schema drifted"
                    )
                support_pair_count += 1
                title, sentence_index = pair
                if not title.strip() or "\x00" in title or sentence_index < 0:
                    invalid_reason = "invalid_support_pair"
                    break
                titles.append(_nfd_title(title))

        distinct_titles = tuple(dict.fromkeys(titles))
        if invalid_reason is None and len(distinct_titles) != hop:
            invalid_reason = "distinct_support_title_count_mismatch"

        gold_rows: list[int] = []
        if invalid_reason is None:
            for title in distinct_titles:
                result = title_cache.get(title)
                if result is None:
                    result = _resolve_title(connection, title)
                    title_cache[title] = result
                status, document = result
                resolution_counts[status] += 1
                if status != "resolved" or document is None:
                    invalid_reason = f"gold_{status}"
                    break
                rowid, observed_title, body = document
                serialization = hashlib.sha256(
                    (observed_title + "\n\n" + body).encode("utf-8")
                ).hexdigest()
                prior = serialization_by_rowid.setdefault(rowid, serialization)
                if prior != serialization:
                    raise HoVerSourceQualificationError(
                        "one documents rowid changed serialization"
                    )
                gold_rows.append(rowid)
        if invalid_reason is None and len(set(gold_rows)) != hop:
            invalid_reason = "distinct_gold_document_count_mismatch"
        if invalid_reason is not None:
            structural_exclusions[invalid_reason] += 1
            continue
        gold_cardinality_counts[len(gold_rows)] += 1
        candidates_before_claim_exclusion.append(
            _Candidate(
                uid=uid,
                hpqa_id=hpqa_id,
                normalized_claim=normalized_claim,
                hop=hop,
                gold_rowids=tuple(sorted(gold_rows)),
            )
        )

    expected_counter = Counter(dict(expected_hop_counts))
    if hop_counts != expected_counter:
        raise HoVerSourceQualificationError("official TRAIN hop counts drifted")
    collision_claims = {
        claim for claim, count in normalized_claim_counts.items() if count > 1
    }
    eligible = tuple(
        candidate
        for candidate in candidates_before_claim_exclusion
        if candidate.normalized_claim not in collision_claims
    )
    collision_excluded_count = (
        len(candidates_before_claim_exclusion) - len(eligible)
    )
    capacity_ok, neighborhoods, shortfalls, hpqa_group_count = _hall_capacity(eligible)
    if not capacity_ok:
        raise HoVerSourceQualificationError(
            "exact hpqa_id by hop b-matching capacity is absent"
        )

    body: dict[str, Any] = {
        "schema": SCHEMA,
        "version": "v1",
        "status": "passed_source_qualification_no_selection",
        "recorded_date": "2026-07-18",
        "claim_boundary": {
            "source_rows_titles_bodies_or_per_row_hashes_serialized": False,
            "selection_secret_or_cohort_created": False,
            "development_or_test_opened": False,
            "retrieval_action_evaluator_or_score_run": False,
        },
        "source_binding": {
            "formal_identity_enforced": formal_identity_enforced,
            "training_path": FORMAL_TRAIN_RELATIVE.as_posix(),
            "training_size": source_size,
            "training_sha256": source_sha256,
            "training_git_blob_sha1": source_git_blob_sha1,
            "corpus_path": FORMAL_DB_RELATIVE.as_posix(),
            "corpus_size": db_size,
            "corpus_sha256": db_sha256,
            "official_git_commit": FORMAL_GIT_COMMIT,
            "design_sha256": FORMAL_DESIGN_SHA256,
        },
        "parser_and_schema": {
            "strict_utf8_duplicate_keys_and_nonfinite_constants_rejected": True,
            "observed_row_count": len(payload),
            "required_consumed_field_count": len(REQUIRED_FIELDS),
            "keyset_hash_counts": dict(sorted(keyset_counts.items())),
            "extra_field_name_count": len(extra_field_names),
            "extra_field_name_set_sha256": _stable_hash(sorted(extra_field_names)),
        },
        "identity_and_grouping": {
            "unique_uid_count": len(seen_uids),
            "unique_normalized_claim_count": len(normalized_claim_counts),
            "normalized_claim_collision_group_count": len(collision_claims),
            "whole_collision_group_excluded_item_count": collision_excluded_count,
            "eligible_unique_hpqa_id_group_count": hpqa_group_count,
            "global_one_uid_per_hpqa_id_selection_required": True,
        },
        "hop_and_structure": {
            "hop_counts": _counter_payload(hop_counts),
            "support_pair_count": support_pair_count,
            "structural_exclusion_counts": _counter_payload(structural_exclusions),
            "eligible_item_count_before_claim_collision_exclusion": len(
                candidates_before_claim_exclusion
            ),
            "eligible_item_count_after_claim_collision_exclusion": len(eligible),
            "gold_cardinality_counts": _counter_payload(gold_cardinality_counts),
        },
        "sqlite_and_gold_resolution": {
            **db_schema,
            "distinct_support_title_lookup_count": len(title_cache),
            "support_title_resolution_occurrence_counts": _counter_payload(
                resolution_counts
            ),
            "distinct_resolved_gold_rowid_count": len(serialization_by_rowid),
            "title_codec": "Unicode_NFD_then_exact_documents_id_equality",
            "fuzzy_casefold_substring_or_underscore_rewrite_used": False,
        },
        "capacity": {
            "target_distinct_hpqa_groups_per_hop": TARGET_GROUPS_PER_HOP,
            "hall_neighborhood_group_counts": neighborhoods,
            "hall_shortfall_counts": shortfalls,
            "exact_three_hop_b_matching_capacity_met": True,
            "maximum_selected_gold_occurrences": 432,
            "fixed_corpus_article_count": 609,
            "minimum_filler_slots": 177,
        },
    }
    body["qualification_sha256"] = _stable_hash(body)
    return body


def _regular_file(path: Path, *, size: int, mode: int, field: str) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise HoVerSourceQualificationError(f"{field} is unavailable") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size != size
        or stat.S_IMODE(metadata.st_mode) != mode
    ):
        raise HoVerSourceQualificationError(f"{field} identity drifted")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise HoVerSourceQualificationError(f"{field} cannot be read") from exc


def _canonical_project(value: str | Path) -> Path:
    path = Path(value)
    if path.is_symlink():
        raise HoVerSourceQualificationError("project root is a symlink")
    try:
        root = path.resolve(strict=True)
    except OSError as exc:
        raise HoVerSourceQualificationError("project root is unavailable") from exc
    if not root.is_dir():
        raise HoVerSourceQualificationError("project root is not a directory")
    return root


def _repository_root(project: Path) -> Path:
    try:
        result = subprocess.run(
            ["git", "-C", str(project), "rev-parse", "--show-toplevel"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise HoVerSourceQualificationError("Git repository is unavailable") from exc
    return Path(result.stdout.strip()).resolve(strict=True)


def _require_private_paths_ignored(project: Path) -> None:
    repository = _repository_root(project)
    prefix = project.relative_to(repository)
    relatives = (FORMAL_TRAIN_RELATIVE, FORMAL_DB_RELATIVE, FORMAL_MARKER_RELATIVE)
    paths = tuple(
        (PurePosixPath(prefix.as_posix()) / relative.as_posix()).as_posix()
        for relative in relatives
    )
    try:
        tracked = subprocess.run(
            ["git", "-C", str(repository), "ls-files", "-z", "--", *paths],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        ignored = subprocess.run(
            ["git", "-C", str(repository), "check-ignore", "--no-index", "-z", "--stdin"],
            input=b"\0".join(path.encode("utf-8") for path in paths) + b"\0",
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise HoVerSourceQualificationError("Git ignore check failed") from exc
    observed = {row.decode("utf-8") for row in ignored.stdout.split(b"\0") if row}
    if tracked.stdout or ignored.returncode != 0 or observed != set(paths):
        raise HoVerSourceQualificationError("private formal paths are not ignored")


def _verify_design(project: Path) -> None:
    path = project / FORMAL_DESIGN_RELATIVE
    raw = _regular_file(
        path,
        size=path.stat().st_size if path.exists() else -1,
        mode=0o644,
        field="design",
    )
    if (
        hashlib.sha256(raw).hexdigest() != FORMAL_DESIGN_FILE_SHA256
        or _git_blob_sha1(raw) != FORMAL_DESIGN_GIT_BLOB_SHA1
    ):
        raise HoVerSourceQualificationError("design bytes drifted")
    try:
        payload = _decode_strict_json(raw)
    except HoVerSourceQualificationError as exc:
        raise HoVerSourceQualificationError("design JSON drifted") from exc
    if not isinstance(payload, Mapping) or payload.get("design_sha256") != FORMAL_DESIGN_SHA256:
        raise HoVerSourceQualificationError("design self binding drifted")


def _consume_marker(project: Path) -> None:
    marker = project / FORMAL_MARKER_RELATIVE
    parent = marker.parent
    if parent.is_symlink() or not parent.is_dir():
        raise HoVerSourceQualificationError("attempt marker parent is unsafe")
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(f"{SCHEMA}\nformal_attempt_consumed\n".encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise HoVerSourceQualificationError("attempt marker already exists") from exc


def _write_json_exclusive(path: Path, payload: object) -> None:
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise HoVerSourceQualificationError("output parent is unsafe")
    raw = _canonical_json(payload) + b"\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise HoVerSourceQualificationError("qualification output exists") from exc


def run_formal(project: str | Path) -> dict[str, Any]:
    root = _canonical_project(project)
    output = root / FORMAL_OUTPUT_RELATIVE
    if output.exists() or output.is_symlink():
        raise HoVerSourceQualificationError("qualification output already exists")
    _require_private_paths_ignored(root)
    _verify_design(root)
    train_path = root / FORMAL_TRAIN_RELATIVE
    db_path = root / FORMAL_DB_RELATIVE
    train_raw = _regular_file(
        train_path,
        size=FORMAL_TRAIN_SIZE,
        mode=0o600,
        field="TRAIN source",
    )
    try:
        db_metadata = db_path.lstat()
    except OSError as exc:
        raise HoVerSourceQualificationError("SQLite source is unavailable") from exc
    if (
        stat.S_ISLNK(db_metadata.st_mode)
        or not stat.S_ISREG(db_metadata.st_mode)
        or db_metadata.st_size != FORMAL_DB_SIZE
        or stat.S_IMODE(db_metadata.st_mode) != 0o600
    ):
        raise HoVerSourceQualificationError("SQLite source identity drifted")
    if (
        hashlib.sha256(train_raw).hexdigest() != FORMAL_TRAIN_SHA256
        or _git_blob_sha1(train_raw) != FORMAL_TRAIN_GIT_BLOB_SHA1
        or _sha256_file(db_path) != FORMAL_DB_SHA256
    ):
        raise HoVerSourceQualificationError("formal source bytes drifted")
    _consume_marker(root)
    try:
        connection = sqlite3.connect(
            f"file:{db_path.as_posix()}?mode=ro&immutable=1",
            uri=True,
        )
        connection.execute("PRAGMA query_only = ON")
        receipt = qualify_payload(
            _decode_strict_json(train_raw),
            connection,
            source_size=len(train_raw),
            source_sha256=FORMAL_TRAIN_SHA256,
            source_git_blob_sha1=FORMAL_TRAIN_GIT_BLOB_SHA1,
            db_size=FORMAL_DB_SIZE,
            db_sha256=FORMAL_DB_SHA256,
            formal_identity_enforced=True,
        )
    except sqlite3.Error as exc:
        raise HoVerSourceQualificationError("formal SQLite open failed") from exc
    finally:
        if "connection" in locals():
            connection.close()
    _write_json_exclusive(output, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    receipt = run_formal(arguments.project)
    print(
        json.dumps(
            {
                "qualification_sha256": receipt["qualification_sha256"],
                "status": receipt["status"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FORMAL_DB_RELATIVE",
    "FORMAL_HOP_COUNTS",
    "FORMAL_MARKER_RELATIVE",
    "FORMAL_OUTPUT_RELATIVE",
    "FORMAL_ROW_COUNT",
    "FORMAL_TRAIN_RELATIVE",
    "HOP_ORDER",
    "HoVerSourceQualificationError",
    "SCHEMA",
    "TARGET_GROUPS_PER_HOP",
    "VERSION",
    "qualify_payload",
    "run_formal",
]
