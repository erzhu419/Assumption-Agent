"""Private one-shot selection and late-qrel custody for BIRCO P1.

The formal BIRCO JSON is deliberately not imported or opened at module load.
``run_formal_selection`` first verifies the committed custody, study design and
aggregate-only qualification result.  The one-shot acquisition then creates
exactly one 32-byte secret and publishes its commitment *before* decoding the
formal source.  Query selection is the ascending HMAC-SHA256 order of one
unambiguous, length-framed ``(study_id, family, qid)`` namespace.  The four
blocks are consecutive ten-query slices of that single order.

Action packs contain the task objective, query and the complete source qrel-key
candidate projection, but contain neither source identifiers nor numeric qrel
values.  Every block's numeric values live in its own sealed mode-0600 pack.
The only loader for those values consumes an exact, externally bound block-open
authorization after all authorization checks have passed.  ``F_search`` values
are formed and committed but can never be opened by this implementation.

This module contains a generic, dependency-free ``acquire_once`` entry point so
the custody logic can be tested with wholly synthetic JSON.  It is not a replay
or a second formal route: the formal wrapper fixes all official paths, byte
identities, family counts and committed manifest self-hashes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Mapping, Sequence

from replication_runtime.birco_gpt54_semantic_v1.contract import (
    project_candidate_text,
)


VERSION = "birco_p1_private_selection_v1"
STUDY_ID = "BIRCO_P1_TYPED_CONSTRAINT_E4_V1"
BIRCO_REPOSITORY_COMMIT = "84b2b05c862cff6f6a80c06a6a11e6a7cd12e838"
TASK_OBJECTIVES_MANIFEST_SCHEMA = "birco_p1_task_objectives_v1"
FAMILIES = ("doris-mae", "clinical-trial", "wtb")
BLOCK_ORDER = ("A_form", "F_search", "A_hold", "M_search")
PER_FAMILY_QUOTA = 10
MAX_OBJECTIVE_CHARACTERS = 8_192
BLOCK_WINDOWS = {
    block: (ordinal * PER_FAMILY_QUOTA, (ordinal + 1) * PER_FAMILY_QUOTA)
    for ordinal, block in enumerate(BLOCK_ORDER)
}
SELECTED_PER_FAMILY = len(BLOCK_ORDER) * PER_FAMILY_QUOTA
SELECTED_TOTAL = len(FAMILIES) * SELECTED_PER_FAMILY

SOURCE_SIZE_BYTES = 20_134_244
SOURCE_MD5 = "548cad5d25ce8c0714274ba0ec17fa78"
SOURCE_SHA256 = "0c30d86924479c0255ecf6101892388a3547312fd09c3882446bab527a1d1f34"
SOURCE_CUSTODY_SELF_SHA256 = (
    "190cddaf78d807d791713301cdaa95fe6239c7c541a385429f2cb7973599af12"
)
STUDY_DESIGN_SELF_SHA256 = (
    "47f88edd3c322ad602f8d3ed4bbe64dc9a94acb6fe20a78791f93ce8e6d747c4"
)
QUALIFICATION_SELF_SHA256 = (
    "c708a02d6fe0fa59b3c4942a17319173e490903dfc07a1890bee34e3431d6618"
)

SOURCE_RELATIVE = Path("artifacts/birco_p1_official_source_v1/BIRCO_dataset.json")
CUSTODY_RELATIVE = Path("manifests/birco_p1_source_custody_v1.json")
DESIGN_RELATIVE = Path("manifests/birco_p1_typed_constraint_e4_study_design_v1.json")
QUALIFICATION_RELATIVE = Path(
    "manifests/birco_p1_source_qualification_result_v1.json"
)
OUTPUT_RELATIVE = Path("artifacts/birco_p1_private_selection_v1")

ATTEMPT_MARKER_FILENAME = "selection.one_shot.private.json"
SECRET_FILENAME = "selection_secret.private.bin"
COMMITMENT_FILENAME = "selection_commitment.public.json"
PUBLIC_RECEIPT_FILENAME = "selection_receipt.public.json"
FAILURE_FILENAME = "selection.terminal_failure.public.json"

ACTION_PACK_FILENAMES = {
    block: f"{block}.action.label_free.private.json" for block in BLOCK_ORDER
}
QREL_PACK_FILENAMES = {
    block: f"{block}.qrels.sealed.private.json" for block in BLOCK_ORDER
}
QREL_OPEN_MARKER_FILENAMES = {
    block: f"{block}.qrels.opened.private.json" for block in BLOCK_ORDER
}

# The order message contains only this domain and the three explicitly framed
# fields below.  In particular, block name, source ordinal and qrel value never
# enter selection.
ORDER_HMAC_DOMAIN = b"BIRCO_P1_PRIVATE_QUERY_ORDER_HMAC_SHA256_V1\x00"
WORK_ID_HMAC_DOMAIN = b"BIRCO_P1_OPAQUE_WORK_ID_HMAC_SHA256_V1\x00"

_HEX32 = re.compile(r"[0-9a-f]{32}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_WORK_ID = re.compile(r"birco-work-v1-[0-9a-f]{64}\Z")
_FORMAL_EXECUTION_CAPABILITY = object()


class BircoP1PrivateSelectionError(RuntimeError):
    """The fixed source, selection, persistence or authorization contract failed."""


@dataclass(frozen=True)
class FamilyContract:
    """Pinned semantic constraints for one source family."""

    name: str
    query_count: int
    corpus_count: int
    allowed_scores: tuple[float, ...] | None
    minimum_score: float = 0.0
    maximum_score: float = 2.0

    def __post_init__(self) -> None:
        if self.name not in FAMILIES:
            raise BircoP1PrivateSelectionError("family contract name is invalid")
        if (
            type(self.query_count) is not int
            or self.query_count < SELECTED_PER_FAMILY
            or type(self.corpus_count) is not int
            or self.corpus_count < PER_FAMILY_QUOTA
        ):
            raise BircoP1PrivateSelectionError("family contract capacity is invalid")
        if (
            not math.isfinite(self.minimum_score)
            or not math.isfinite(self.maximum_score)
            or self.minimum_score > self.maximum_score
        ):
            raise BircoP1PrivateSelectionError("family score interval is invalid")
        if self.allowed_scores is not None:
            if (
                not self.allowed_scores
                or len(set(self.allowed_scores)) != len(self.allowed_scores)
                or any(
                    not math.isfinite(value)
                    or not self.minimum_score <= value <= self.maximum_score
                    for value in self.allowed_scores
                )
            ):
                raise BircoP1PrivateSelectionError(
                    "family discrete score domain is invalid"
                )


@dataclass(frozen=True)
class SourceContract:
    """Byte identity and source shape needed by the one-shot selector."""

    source_size_bytes: int
    source_md5: str
    source_sha256: str
    families: tuple[FamilyContract, ...]
    minimum_pool_size: int = 10
    maximum_id_characters: int = 1_024
    maximum_query_characters: int = 250_000
    maximum_document_characters: int = 2_000_000

    def __post_init__(self) -> None:
        if (
            type(self.source_size_bytes) is not int
            or self.source_size_bytes <= 0
            or _HEX32.fullmatch(self.source_md5) is None
            or _HEX64.fullmatch(self.source_sha256) is None
            or tuple(family.name for family in self.families) != FAMILIES
            or type(self.minimum_pool_size) is not int
            or self.minimum_pool_size < 1
            or self.minimum_pool_size > min(
                family.corpus_count for family in self.families
            )
        ):
            raise BircoP1PrivateSelectionError("source contract is invalid")
        for limit in (
            self.maximum_id_characters,
            self.maximum_query_characters,
            self.maximum_document_characters,
        ):
            if type(limit) is not int or limit < 1:
                raise BircoP1PrivateSelectionError("source text limit is invalid")

    def family(self, name: str) -> FamilyContract:
        for family in self.families:
            if family.name == name:
                return family
        raise BircoP1PrivateSelectionError("source family is outside the contract")


FORMAL_CONTRACT = SourceContract(
    source_size_bytes=SOURCE_SIZE_BYTES,
    source_md5=SOURCE_MD5,
    source_sha256=SOURCE_SHA256,
    families=(
        FamilyContract("doris-mae", 60, 5_543, None),
        FamilyContract("clinical-trial", 50, 3_256, (0.0, 1.0, 2.0)),
        FamilyContract("wtb", 100, 1_767, (0.0, 1.0)),
    ),
)


@dataclass(frozen=True)
class QueryRecord:
    """Validated source record retained only inside the private selector."""

    family: str
    qid: str
    query: str
    candidate_ids: tuple[str, ...]
    candidate_texts: tuple[str, ...]
    qrel_values: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            self.family not in FAMILIES
            or not isinstance(self.qid, str)
            or not self.qid
            or self.qid != self.qid.strip()
            or "\x00" in self.qid
            or not isinstance(self.query, str)
            or not self.query.strip()
            or "\x00" in self.query
        ):
            raise BircoP1PrivateSelectionError("query record identity is invalid")
        width = len(self.candidate_ids)
        if (
            width < 1
            or len(set(self.candidate_ids)) != width
            or len(self.candidate_texts) != width
            or len(self.qrel_values) != width
            or any(
                not isinstance(candidate_id, str)
                or not candidate_id
                or candidate_id != candidate_id.strip()
                or "\x00" in candidate_id
                for candidate_id in self.candidate_ids
            )
            or any(
                not isinstance(text, str) or not text.strip() or "\x00" in text
                for text in self.candidate_texts
            )
            or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                for value in self.qrel_values
            )
        ):
            raise BircoP1PrivateSelectionError("query candidate projection is invalid")


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise BircoP1PrivateSelectionError("value is not canonical JSON") from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    """Return the SHA-256 of the canonical semantic JSON representation."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise BircoP1PrivateSelectionError("self-hash field already exists")
    return {**dict(body), field: stable_hash(body)}


def verify_self_hash(value: Mapping[str, Any], field: str) -> str:
    if not isinstance(value, Mapping):
        raise BircoP1PrivateSelectionError("self-hashed value is not an object")
    body = dict(value)
    claimed = body.pop(field, None)
    if not isinstance(claimed, str) or _HEX64.fullmatch(claimed) is None:
        raise BircoP1PrivateSelectionError("self-hash is missing or invalid")
    if not hmac.compare_digest(stable_hash(body), claimed):
        raise BircoP1PrivateSelectionError("self-hash mismatch")
    return claimed


def _duplicate_rejecting_object(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BircoP1PrivateSelectionError(
                "JSON contains a duplicate object key"
            )
        result[key] = value
    return result


def _reject_nonfinite_constant(_value: str) -> None:
    raise BircoP1PrivateSelectionError("JSON contains a non-finite number")


def _strict_json(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_nonfinite_constant,
        )
    except BircoP1PrivateSelectionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BircoP1PrivateSelectionError(f"{label} JSON is invalid") from exc


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise BircoP1PrivateSelectionError("durable directory is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise BircoP1PrivateSelectionError("durable path is not a directory")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_durable_directory(path: Path) -> None:
    missing: list[Path] = []
    cursor = path
    while True:
        try:
            metadata = cursor.lstat()
        except FileNotFoundError:
            if cursor.parent == cursor:
                raise BircoP1PrivateSelectionError(
                    "durable directory parent is unavailable"
                )
            missing.append(cursor)
            cursor = cursor.parent
            continue
        except OSError as exc:
            raise BircoP1PrivateSelectionError(
                "durable directory cannot be inspected"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise BircoP1PrivateSelectionError("durable directory path is unsafe")
        break
    for directory in reversed(missing):
        try:
            os.mkdir(directory, 0o700)
            os.chmod(directory, 0o700)
        except OSError as exc:
            raise BircoP1PrivateSelectionError(
                "durable directory cannot be created"
            ) from exc
        _fsync_directory(directory)
        _fsync_directory(directory.parent)


def _create_one_shot_root(path: Path) -> None:
    _ensure_durable_directory(path.parent)
    try:
        os.mkdir(path, 0o700)
        os.chmod(path, 0o700)
    except FileExistsError as exc:
        raise BircoP1PrivateSelectionError(
            "selection root already exists; replay is forbidden"
        ) from exc
    except OSError as exc:
        raise BircoP1PrivateSelectionError(
            "selection root cannot be created"
        ) from exc
    _fsync_directory(path)
    _fsync_directory(path.parent)


def _require_regular_file(
    path: Path,
    *,
    label: str,
    expected_mode: int | None = None,
    require_single_link: bool = True,
) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise BircoP1PrivateSelectionError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise BircoP1PrivateSelectionError(f"{label} is not a regular file")
    if require_single_link and metadata.st_nlink != 1:
        raise BircoP1PrivateSelectionError(f"{label} link count is unsafe")
    if expected_mode is not None and stat.S_IMODE(metadata.st_mode) != expected_mode:
        raise BircoP1PrivateSelectionError(f"{label} mode drifted")
    return metadata


def _write_exclusive_bytes(path: Path, raw: bytes, *, mode: int) -> str:
    """Write a fixed path with O_CREAT|O_EXCL, fsync it and fsync its parent."""

    if not isinstance(raw, bytes):
        raise BircoP1PrivateSelectionError("exclusive payload is not bytes")
    _ensure_durable_directory(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as exc:
        raise BircoP1PrivateSelectionError(
            "exclusive output already exists or is unavailable"
        ) from exc
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            metadata = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != mode
            ):
                raise BircoP1PrivateSelectionError(
                    "exclusive output mode is unenforceable"
                )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_directory(path.parent)
    _require_regular_file(path, label="exclusive output", expected_mode=mode)
    return hashlib.sha256(raw).hexdigest()


def _atomic_write_exclusive(path: Path, raw: bytes, *, mode: int) -> str:
    """Atomically publish complete bytes without an overwrite race.

    A fixed sibling staging path is created with ``O_CREAT|O_EXCL``.  After the
    bytes and mode are durable, ``link`` creates the final path atomically and
    refuses an existing destination; the staging link is then removed.  No
    random filename or second secret is generated.
    """

    if not isinstance(raw, bytes):
        raise BircoP1PrivateSelectionError("atomic payload is not bytes")
    _ensure_durable_directory(path.parent)
    staging = path.with_name(f".{path.name}.part")
    if path.exists() or path.is_symlink() or staging.exists() or staging.is_symlink():
        raise BircoP1PrivateSelectionError(
            "atomic output or fixed staging path already exists"
        )
    _write_exclusive_bytes(staging, raw, mode=mode)
    try:
        os.link(staging, path, follow_symlinks=False)
        _fsync_directory(path.parent)
        os.unlink(staging)
        _fsync_directory(path.parent)
    except OSError as exc:
        raise BircoP1PrivateSelectionError(
            "atomic exclusive output publication failed"
        ) from exc
    _require_regular_file(path, label="atomic output", expected_mode=mode)
    return hashlib.sha256(raw).hexdigest()


def _atomic_write_json(path: Path, value: object, *, mode: int) -> dict[str, Any]:
    raw = _canonical_bytes(value, newline=True)
    file_sha256 = _atomic_write_exclusive(path, raw, mode=mode)
    return {
        "file_sha256": file_sha256,
        "size_bytes": len(raw),
        "mode_octal": f"{mode:04o}",
    }


def _read_stable_regular_bytes(
    path: Path,
    *,
    label: str,
    expected_size: int | None = None,
    expected_md5: str | None = None,
    expected_sha256: str | None = None,
    expected_mode: int | None = None,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise BircoP1PrivateSelectionError(f"{label} is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise BircoP1PrivateSelectionError(f"{label} is not a private regular file")
        if expected_mode is not None and stat.S_IMODE(before.st_mode) != expected_mode:
            raise BircoP1PrivateSelectionError(f"{label} mode drifted")
        if expected_size is not None and before.st_size != expected_size:
            raise BircoP1PrivateSelectionError(f"{label} size drifted")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise BircoP1PrivateSelectionError(f"{label} changed during read")
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if expected_size is not None and len(raw) != expected_size:
        raise BircoP1PrivateSelectionError(f"{label} size drifted")
    if expected_md5 is not None and not hmac.compare_digest(
        hashlib.md5(raw).hexdigest(), expected_md5  # nosec B303: source identity
    ):
        raise BircoP1PrivateSelectionError(f"{label} MD5 identity drifted")
    if expected_sha256 is not None and not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), expected_sha256
    ):
        raise BircoP1PrivateSelectionError(f"{label} SHA256 identity drifted")
    return raw


def _load_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_stable_regular_bytes(path, label=label)
    value = _strict_json(raw, label=label)
    if not isinstance(value, Mapping):
        raise BircoP1PrivateSelectionError(f"{label} is not an object")
    return dict(value), raw


def _verify_manifest(
    path: Path,
    *,
    expected_self_sha256: str,
    expected_schema: str,
) -> dict[str, Any]:
    value, _raw = _load_json_object(path, label=expected_schema)
    observed = verify_self_hash(value, "self_sha256")
    if (
        not hmac.compare_digest(observed, expected_self_sha256)
        or value.get("schema") != expected_schema
        or value.get("study_id") != STUDY_ID
    ):
        raise BircoP1PrivateSelectionError("bound manifest identity drifted")
    return value


def verify_qualification_result(
    path: Path,
    *,
    expected_self_sha256: str,
    contract: SourceContract,
) -> dict[str, Any]:
    """Verify the aggregate-only qualification without opening source rows."""

    value, _raw = _load_json_object(path, label="source qualification result")
    observed = verify_self_hash(value, "self_sha256")
    if (
        _HEX64.fullmatch(expected_self_sha256) is None
        or not hmac.compare_digest(observed, expected_self_sha256)
        or value.get("schema") != "birco_p1_source_qualification_v1_result_v1"
        or value.get("status") != "qualified_aggregate_only"
        or value.get("qualified") is not True
        or value.get("model_action_or_score_count") != 0
        or value.get("online_evaluator_call_count") != 0
        or value.get("qrel_value_output_count") != 0
    ):
        raise BircoP1PrivateSelectionError("source qualification binding drifted")
    if value.get("source_identity") != {
        "md5": contract.source_md5,
        "sha256": contract.source_sha256,
        "size_bytes": contract.source_size_bytes,
    }:
        raise BircoP1PrivateSelectionError(
            "qualification source identity drifted"
        )
    aggregates = value.get("family_aggregates")
    if not isinstance(aggregates, Mapping):
        raise BircoP1PrivateSelectionError("qualification family aggregates drifted")
    for family in contract.families:
        aggregate = aggregates.get(family.name)
        if not isinstance(aggregate, Mapping):
            raise BircoP1PrivateSelectionError(
                "qualification family aggregate is absent"
            )
        membership = aggregate.get("candidate_membership")
        if (
            aggregate.get("query_count") != family.query_count
            or aggregate.get("corpus_count") != family.corpus_count
            or aggregate.get("query_disjoint_selected_capacity")
            != SELECTED_PER_FAMILY
            or not isinstance(membership, Mapping)
            or membership.get("distinct_candidate_count") != family.corpus_count
            or not isinstance(membership.get("minimum_pool_size"), int)
            or membership["minimum_pool_size"] < contract.minimum_pool_size
        ):
            raise BircoP1PrivateSelectionError(
                "qualification family capacity drifted"
            )
    return value


def _verify_formal_manifests(project: Path) -> None:
    custody = _verify_manifest(
        project / CUSTODY_RELATIVE,
        expected_self_sha256=SOURCE_CUSTODY_SELF_SHA256,
        expected_schema="birco_p1_source_custody_v1",
    )
    design = _verify_manifest(
        project / DESIGN_RELATIVE,
        expected_self_sha256=STUDY_DESIGN_SELF_SHA256,
        expected_schema="birco_p1_typed_constraint_e4_study_design_v1",
    )
    formal_source = custody.get("formal_source")
    formal_scope = custody.get("formal_scope")
    block_contract = design.get("block_contract")
    source_binding = design.get("source_binding")
    execution_contract = design.get("execution_contract")
    if (
        not isinstance(formal_source, Mapping)
        or formal_source.get("expected_MD5") != SOURCE_MD5
        or formal_source.get("expected_size_bytes") != SOURCE_SIZE_BYTES
        or not isinstance(formal_scope, Mapping)
        or tuple(formal_scope.get("families", ())) != FAMILIES
        or formal_scope.get("qrel_values_visible_to_action_or_online_evaluator")
        is not False
        or not isinstance(block_contract, Mapping)
        or block_contract.get("query_disjoint_across_blocks") is not True
        or block_contract.get("selected_total") != SELECTED_TOTAL
        or not isinstance(source_binding, Mapping)
        or source_binding.get("formal_file_expected_MD5") != SOURCE_MD5
        or source_binding.get("formal_file_expected_size_bytes")
        != SOURCE_SIZE_BYTES
        or source_binding.get("source_custody_self_sha256")
        != SOURCE_CUSTODY_SELF_SHA256
        or not isinstance(execution_contract, Mapping)
        or execution_contract.get("qrel_release_order")
        != (
            "candidate_membership_keys_may_be_released_label-free;numeric_qrel_"
            "values_for_a_block_remain_sealed_until_all_authorized_action_archives_"
            "for_that_block_are_complete_and_immutable"
        )
    ):
        raise BircoP1PrivateSelectionError("formal custody/design contract drifted")
    for block in BLOCK_ORDER:
        row = block_contract.get(block)
        if (
            not isinstance(row, Mapping)
            or row.get("per_family") != PER_FAMILY_QUOTA
            or (
                "total" in row
                and row.get("total") != len(FAMILIES) * PER_FAMILY_QUOTA
            )
        ):
            raise BircoP1PrivateSelectionError("formal block quota drifted")
    if block_contract["M_search"].get("open_only_after_A_hold_promotion") is not True:
        raise BircoP1PrivateSelectionError("formal M_search open boundary drifted")


def _safe_id(value: object, *, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise BircoP1PrivateSelectionError("source identifier schema drifted")
    return value


def _safe_text(value: object, *, maximum: int, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise BircoP1PrivateSelectionError(f"source {field} schema drifted")
    return value


def _validated_score(value: object, family: FamilyContract) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise BircoP1PrivateSelectionError("qrel value is not numeric")
    score = float(value)
    if (
        not math.isfinite(score)
        or not family.minimum_score <= score <= family.maximum_score
        or (
            family.allowed_scores is not None
            and score not in family.allowed_scores
        )
    ):
        raise BircoP1PrivateSelectionError("qrel value is outside the frozen domain")
    return score


def validate_source_payload(
    source: object,
    *,
    contract: SourceContract,
) -> dict[str, tuple[QueryRecord, ...]]:
    """Validate every query, qrel membership and corpus text join."""

    if not isinstance(source, Mapping):
        raise BircoP1PrivateSelectionError("source top level is not an object")
    records: dict[str, tuple[QueryRecord, ...]] = {}
    for family_contract in contract.families:
        family_name = family_contract.name
        family_value = source.get(family_name)
        if not isinstance(family_value, Mapping):
            raise BircoP1PrivateSelectionError("selected source family is absent")
        if not {"query", "corpus", "qrel"}.issubset(family_value):
            raise BircoP1PrivateSelectionError("source family fields are incomplete")
        queries = family_value.get("query")
        corpus = family_value.get("corpus")
        qrels = family_value.get("qrel")
        if (
            not isinstance(queries, Mapping)
            or not isinstance(corpus, Mapping)
            or not isinstance(qrels, Mapping)
            or len(queries) != family_contract.query_count
            or len(corpus) != family_contract.corpus_count
        ):
            raise BircoP1PrivateSelectionError("source family mappings drifted")

        # Validate the complete corpus, including entries not encountered until
        # a later query.  The final union equality below forbids silent extras.
        corpus_text: dict[str, str] = {}
        for raw_cid, raw_text in corpus.items():
            cid = _safe_id(raw_cid, maximum=contract.maximum_id_characters)
            corpus_text[cid] = _safe_text(
                raw_text,
                maximum=contract.maximum_document_characters,
                field="document text",
            )
        query_ids = tuple(
            _safe_id(raw_qid, maximum=contract.maximum_id_characters)
            for raw_qid in queries
        )
        if set(qrels) != set(query_ids):
            raise BircoP1PrivateSelectionError("query/qrel identity sets drifted")

        used_candidate_ids: set[str] = set()
        family_records: list[QueryRecord] = []
        for qid in query_ids:
            query = _safe_text(
                queries[qid],
                maximum=contract.maximum_query_characters,
                field="query text",
            )
            row = qrels.get(qid)
            if not isinstance(row, Mapping) or len(row) < contract.minimum_pool_size:
                raise BircoP1PrivateSelectionError(
                    "qrel candidate membership or capacity drifted"
                )
            candidate_ids: list[str] = []
            candidate_texts: list[str] = []
            values: list[float] = []
            positive = False
            for raw_cid, raw_value in row.items():
                cid = _safe_id(
                    raw_cid, maximum=contract.maximum_id_characters
                )
                if cid not in corpus_text:
                    raise BircoP1PrivateSelectionError(
                        "qrel candidate is absent from the family corpus"
                    )
                score = _validated_score(raw_value, family_contract)
                candidate_ids.append(cid)
                candidate_texts.append(corpus_text[cid])
                values.append(score)
                positive = positive or score > 0.0
                used_candidate_ids.add(cid)
            if not positive:
                raise BircoP1PrivateSelectionError(
                    "qrel candidate pool has no positive gain"
                )
            family_records.append(
                QueryRecord(
                    family=family_name,
                    qid=qid,
                    query=query,
                    candidate_ids=tuple(candidate_ids),
                    candidate_texts=tuple(candidate_texts),
                    qrel_values=tuple(values),
                )
            )
        if used_candidate_ids != set(corpus_text):
            raise BircoP1PrivateSelectionError(
                "candidate membership does not cover the complete family corpus"
            )
        records[family_name] = tuple(family_records)
    return records


def parse_source_bytes(
    raw: bytes,
    *,
    contract: SourceContract,
) -> dict[str, tuple[QueryRecord, ...]]:
    """Strictly decode and validate source bytes (called only after commitment)."""

    return validate_source_payload(
        _strict_json(raw, label="BIRCO source"), contract=contract
    )


def _frame(name: bytes, value: str) -> bytes:
    if not isinstance(name, bytes) or not name or b"\x00" in name:
        raise BircoP1PrivateSelectionError("HMAC frame name is invalid")
    if not isinstance(value, str) or not value or "\x00" in value:
        raise BircoP1PrivateSelectionError("HMAC frame value is invalid")
    raw = value.encode("utf-8")
    return name + b"\x00" + len(raw).to_bytes(8, "big") + raw


def selection_hmac_message(family: str, qid: str) -> bytes:
    """Exact order namespace: domain, study ID, family, then source qid."""

    if family not in FAMILIES:
        raise BircoP1PrivateSelectionError("selection family is invalid")
    return (
        ORDER_HMAC_DOMAIN
        + _frame(b"study", STUDY_ID)
        + _frame(b"family", family)
        + _frame(b"qid", qid)
    )


def selection_hmac_digest(secret: bytes, family: str, qid: str) -> bytes:
    if not isinstance(secret, bytes) or len(secret) != 32:
        raise BircoP1PrivateSelectionError(
            "selection secret must contain exactly 32 bytes"
        )
    return hmac.new(
        secret, selection_hmac_message(family, qid), hashlib.sha256
    ).digest()


def opaque_work_id(secret: bytes, *, family: str, qid: str, block: str) -> str:
    if not isinstance(secret, bytes) or len(secret) != 32:
        raise BircoP1PrivateSelectionError(
            "selection secret must contain exactly 32 bytes"
        )
    if family not in FAMILIES or block not in BLOCK_ORDER:
        raise BircoP1PrivateSelectionError("work ID namespace is invalid")
    message = (
        WORK_ID_HMAC_DOMAIN
        + _frame(b"study", STUDY_ID)
        + _frame(b"family", family)
        + _frame(b"qid", qid)
        + _frame(b"block", block)
    )
    return "birco-work-v1-" + hmac.new(secret, message, hashlib.sha256).hexdigest()


def selection_secret_commitment(secret: bytes) -> str:
    if not isinstance(secret, bytes) or len(secret) != 32:
        raise BircoP1PrivateSelectionError(
            "selection secret must contain exactly 32 bytes"
        )
    return hashlib.sha256(secret).hexdigest()


def common_projection_sha256(
    *, objective: str, query: str, documents: Sequence[Mapping[str, Any]]
) -> str:
    """Hash the exact no-newline common three-arm text projection."""

    if (
        not isinstance(objective, str)
        or not objective.strip()
        or "\x00" in objective
        or not isinstance(query, str)
        or not query.strip()
        or "\x00" in query
        or not isinstance(documents, (list, tuple))
        or not documents
    ):
        raise BircoP1PrivateSelectionError("common projection is invalid")
    frozen_documents: list[dict[str, Any]] = []
    for ordinal, document in enumerate(documents):
        if (
            not isinstance(document, Mapping)
            or set(document) != {"ordinal", "text"}
            or document.get("ordinal") != ordinal
            or not isinstance(document.get("text"), str)
            or not str(document["text"]).strip()
            or "\x00" in str(document["text"])
        ):
            raise BircoP1PrivateSelectionError(
                "common projection document drifted"
            )
        frozen_documents.append(
            {"ordinal": ordinal, "text": str(document["text"])}
        )
    return stable_hash(
        {
            "documents": frozen_documents,
            "objective": objective,
            "query": query,
        }
    )


def select_private_blocks(
    records: Mapping[str, Sequence[QueryRecord]],
    *,
    secret: bytes,
) -> dict[str, tuple[QueryRecord, ...]]:
    """Take four fixed slices from one HMAC order in each family."""

    if not isinstance(secret, bytes) or len(secret) != 32:
        raise BircoP1PrivateSelectionError(
            "selection secret must contain exactly 32 bytes"
        )
    selected: dict[str, list[QueryRecord]] = {block: [] for block in BLOCK_ORDER}
    selected_identities: set[tuple[str, str]] = set()
    for family in FAMILIES:
        rows = tuple(records.get(family, ()))
        if len(rows) < SELECTED_PER_FAMILY:
            raise BircoP1PrivateSelectionError(
                "family has insufficient query-disjoint selection capacity"
            )
        if (
            any(not isinstance(row, QueryRecord) or row.family != family for row in rows)
            or len({row.qid for row in rows}) != len(rows)
        ):
            raise BircoP1PrivateSelectionError("selection records are invalid")
        ordered = tuple(
            sorted(
                rows,
                key=lambda row: (
                    selection_hmac_digest(secret, family, row.qid),
                    row.qid.encode("utf-8"),
                ),
            )
        )
        for block in BLOCK_ORDER:
            start, stop = BLOCK_WINDOWS[block]
            block_rows = ordered[start:stop]
            if len(block_rows) != PER_FAMILY_QUOTA:
                raise BircoP1PrivateSelectionError("fixed HMAC window is incomplete")
            for row in block_rows:
                identity = (family, row.qid)
                if identity in selected_identities:
                    raise BircoP1PrivateSelectionError(
                        "selected blocks are not query-disjoint"
                    )
                selected_identities.add(identity)
            selected[block].extend(block_rows)
    frozen = {block: tuple(selected[block]) for block in BLOCK_ORDER}
    expected_block_count = len(FAMILIES) * PER_FAMILY_QUOTA
    if (
        any(len(frozen[block]) != expected_block_count for block in BLOCK_ORDER)
        or len(selected_identities) != SELECTED_TOTAL
    ):
        raise BircoP1PrivateSelectionError("selected block quota drifted")
    return frozen


def _validated_objectives(objectives: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(objectives, Mapping) or set(objectives) != set(FAMILIES):
        raise BircoP1PrivateSelectionError("task objective family set drifted")
    result: dict[str, str] = {}
    for family in FAMILIES:
        objective = objectives.get(family)
        if (
            not isinstance(objective, str)
            or not objective.strip()
            or "\x00" in objective
            or len(objective) > MAX_OBJECTIVE_CHARACTERS
        ):
            raise BircoP1PrivateSelectionError("task objective text is invalid")
        result[family] = objective
    return result


def load_task_objectives_manifest(path: Path) -> dict[str, str]:
    """Load the formal README/objective binding and return only its objectives.

    The manifest remains a public pre-source binding.  Its self-hash is the
    canonical semantic hash of exactly ``schema``, ``repository_commit``,
    ``readme_sha256`` and ``objectives``; additional provenance or mutable
    controller fields are rejected.
    """

    value, _raw = _load_json_object(Path(path), label="task objectives manifest")
    if set(value) != {
        "schema",
        "repository_commit",
        "readme_sha256",
        "objectives",
        "self_sha256",
    }:
        raise BircoP1PrivateSelectionError(
            "task objectives manifest exact shape drifted"
        )
    observed = verify_self_hash(value, "self_sha256")
    readme_sha256 = value.get("readme_sha256")
    if (
        value.get("schema") != TASK_OBJECTIVES_MANIFEST_SCHEMA
        or value.get("repository_commit") != BIRCO_REPOSITORY_COMMIT
        or not isinstance(readme_sha256, str)
        or _HEX64.fullmatch(readme_sha256) is None
        or _HEX64.fullmatch(observed) is None
    ):
        raise BircoP1PrivateSelectionError(
            "task objectives manifest binding drifted"
        )
    objectives = value.get("objectives")
    if not isinstance(objectives, Mapping):
        raise BircoP1PrivateSelectionError(
            "task objectives manifest payload drifted"
        )
    return _validated_objectives(objectives)


def build_private_packs(
    selected: Mapping[str, Sequence[QueryRecord]],
    *,
    secret: bytes,
    task_objectives: Mapping[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build ID-free action packs and distinct numeric qrel packs."""

    objectives = _validated_objectives(task_objectives)
    actions: dict[str, dict[str, Any]] = {}
    qrels: dict[str, dict[str, Any]] = {}
    all_work_ids: set[str] = set()
    for block in BLOCK_ORDER:
        rows = tuple(selected.get(block, ()))
        if len(rows) != len(FAMILIES) * PER_FAMILY_QUOTA:
            raise BircoP1PrivateSelectionError("private block count drifted")
        action_items: list[dict[str, Any]] = []
        label_items: list[dict[str, Any]] = []
        family_counts = {family: 0 for family in FAMILIES}
        for ordinal, row in enumerate(rows):
            if not isinstance(row, QueryRecord):
                raise BircoP1PrivateSelectionError("private block row is invalid")
            family_counts[row.family] += 1
            work_id = opaque_work_id(
                secret, family=row.family, qid=row.qid, block=block
            )
            if work_id in all_work_ids:
                raise BircoP1PrivateSelectionError("opaque work ID collision")
            all_work_ids.add(work_id)
            documents: list[dict[str, Any]] = []
            for candidate_ordinal, source_text in enumerate(row.candidate_texts):
                try:
                    projection = project_candidate_text(
                        source_text, candidate_ordinal=candidate_ordinal
                    )
                except (TypeError, ValueError, RuntimeError) as exc:
                    raise BircoP1PrivateSelectionError(
                        "frozen semantic candidate projection failed"
                    ) from exc
                documents.append(
                    {
                        "ordinal": candidate_ordinal,
                        "text": projection.projection_text,
                    }
                )
            projection_sha256 = common_projection_sha256(
                objective=objectives[row.family],
                query=row.query,
                documents=documents,
            )
            action_items.append(
                {
                    "schema": f"{VERSION}_label_free_action_item_v1",
                    "block_ordinal": ordinal,
                    "work_id": work_id,
                    "candidate_count": len(documents),
                    "common_projection_sha256": projection_sha256,
                    "hipporag_input": {
                        "schema": (
                            "birco_official_hipporag_candidate_retrieval_v1_input"
                        ),
                        "work_id": work_id,
                        "objective": objectives[row.family],
                        "query": row.query,
                        "documents": documents,
                        "common_projection_sha256": projection_sha256,
                    },
                }
            )
            label_items.append(
                {
                    "block_ordinal": ordinal,
                    "work_id": work_id,
                    "family": row.family,
                    "qrel_values": [
                        {
                            "candidate_ordinal": candidate_ordinal,
                            "value": value,
                        }
                        for candidate_ordinal, value in enumerate(row.qrel_values)
                    ],
                }
            )
        if family_counts != {family: PER_FAMILY_QUOTA for family in FAMILIES}:
            raise BircoP1PrivateSelectionError("private block family quota drifted")
        action_body = {
            "schema": f"{VERSION}_label_free_action_pack_v1",
            "version": VERSION,
            "study_id": STUDY_ID,
            "block": block,
            "item_count": len(action_items),
            "common_action_projection_fields": [
                "hipporag_input.objective",
                "hipporag_input.query",
                "hipporag_input.documents.ordinal",
                "hipporag_input.documents.text",
                "hipporag_input.common_projection_sha256",
            ],
            "hipporag_exact_input_field": "hipporag_input",
            "source_qid_or_candidate_id_included": False,
            "numeric_qrel_value_included": False,
            "items": action_items,
        }
        action_pack = self_hashed(action_body, "action_pack_sha256")
        qrel_body = {
            "schema": f"{VERSION}_sealed_qrel_pack_v1",
            "version": VERSION,
            "study_id": STUDY_ID,
            "block": block,
            "item_count": len(label_items),
            "action_pack_sha256": action_pack["action_pack_sha256"],
            "source_qid_or_candidate_id_included": False,
            "numeric_qrel_values_sealed_separately": True,
            "items": label_items,
        }
        qrel_pack = self_hashed(qrel_body, "qrel_pack_sha256")
        actions[block] = action_pack
        qrels[block] = qrel_pack
    if len(all_work_ids) != SELECTED_TOTAL:
        raise BircoP1PrivateSelectionError("opaque work ID cardinality drifted")
    return actions, qrels


def _validate_action_pack(value: Mapping[str, Any], *, block: str) -> str:
    digest = verify_self_hash(value, "action_pack_sha256")
    items = value.get("items")
    if (
        block not in BLOCK_ORDER
        or value.get("schema") != f"{VERSION}_label_free_action_pack_v1"
        or value.get("version") != VERSION
        or value.get("study_id") != STUDY_ID
        or value.get("block") != block
        or value.get("item_count") != len(FAMILIES) * PER_FAMILY_QUOTA
        or value.get("source_qid_or_candidate_id_included") is not False
        or value.get("numeric_qrel_value_included") is not False
        or not isinstance(items, list)
        or len(items) != len(FAMILIES) * PER_FAMILY_QUOTA
    ):
        raise BircoP1PrivateSelectionError("label-free action pack drifted")
    work_ids: set[str] = set()
    for ordinal, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise BircoP1PrivateSelectionError("action item is invalid")
        work_id = item.get("work_id")
        hipporag_input = item.get("hipporag_input")
        documents = (
            hipporag_input.get("documents")
            if isinstance(hipporag_input, Mapping)
            else None
        )
        if (
            item.get("schema") != f"{VERSION}_label_free_action_item_v1"
            or item.get("block_ordinal") != ordinal
            or not isinstance(work_id, str)
            or _WORK_ID.fullmatch(work_id) is None
            or work_id in work_ids
            or not isinstance(hipporag_input, Mapping)
            or set(hipporag_input)
            != {
                "schema",
                "work_id",
                "objective",
                "query",
                "documents",
                "common_projection_sha256",
            }
            or hipporag_input.get("schema")
            != "birco_official_hipporag_candidate_retrieval_v1_input"
            or hipporag_input.get("work_id") != work_id
            or not isinstance(hipporag_input.get("objective"), str)
            or not isinstance(hipporag_input.get("query"), str)
            or not isinstance(documents, list)
            or len(documents) < 1
            or item.get("candidate_count") != len(documents)
            or item.get("common_projection_sha256")
            != hipporag_input.get("common_projection_sha256")
        ):
            raise BircoP1PrivateSelectionError("action item projection drifted")
        work_ids.add(work_id)
        for candidate_ordinal, candidate in enumerate(documents):
            if (
                not isinstance(candidate, Mapping)
                or set(candidate) != {"ordinal", "text"}
                or candidate.get("ordinal") != candidate_ordinal
                or not isinstance(candidate.get("text"), str)
            ):
                raise BircoP1PrivateSelectionError(
                    "action candidate projection drifted"
                )
        observed_projection = common_projection_sha256(
            objective=str(hipporag_input["objective"]),
            query=str(hipporag_input["query"]),
            documents=documents,
        )
        if not hmac.compare_digest(
            observed_projection, str(item["common_projection_sha256"])
        ):
            raise BircoP1PrivateSelectionError(
                "action common projection commitment drifted"
            )
    return digest


def _validate_qrel_pack(
    value: Mapping[str, Any],
    *,
    block: str,
    expected_action_pack_sha256: str,
) -> str:
    digest = verify_self_hash(value, "qrel_pack_sha256")
    items = value.get("items")
    if (
        block not in BLOCK_ORDER
        or value.get("schema") != f"{VERSION}_sealed_qrel_pack_v1"
        or value.get("version") != VERSION
        or value.get("study_id") != STUDY_ID
        or value.get("block") != block
        or value.get("item_count") != len(FAMILIES) * PER_FAMILY_QUOTA
        or value.get("action_pack_sha256") != expected_action_pack_sha256
        or value.get("source_qid_or_candidate_id_included") is not False
        or value.get("numeric_qrel_values_sealed_separately") is not True
        or not isinstance(items, list)
        or len(items) != len(FAMILIES) * PER_FAMILY_QUOTA
    ):
        raise BircoP1PrivateSelectionError("sealed qrel pack drifted")
    work_ids: set[str] = set()
    family_counts = {family: 0 for family in FAMILIES}
    for ordinal, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise BircoP1PrivateSelectionError("sealed qrel item is invalid")
        work_id = item.get("work_id")
        family = item.get("family")
        values = item.get("qrel_values")
        if (
            item.get("block_ordinal") != ordinal
            or not isinstance(work_id, str)
            or _WORK_ID.fullmatch(work_id) is None
            or work_id in work_ids
            or family not in FAMILIES
            or not isinstance(values, list)
            or len(values) < 1
        ):
            raise BircoP1PrivateSelectionError("sealed qrel item drifted")
        work_ids.add(work_id)
        family_counts[str(family)] += 1
        for candidate_ordinal, row in enumerate(values):
            if (
                not isinstance(row, Mapping)
                or set(row) != {"candidate_ordinal", "value"}
                or row.get("candidate_ordinal") != candidate_ordinal
                or isinstance(row.get("value"), bool)
                or not isinstance(row.get("value"), Real)
                or not math.isfinite(float(row["value"]))
            ):
                raise BircoP1PrivateSelectionError("sealed qrel value drifted")
    if family_counts != {family: PER_FAMILY_QUOTA for family in FAMILIES}:
        raise BircoP1PrivateSelectionError("sealed qrel family quota drifted")
    return digest


def _persist_private_packs(
    output_root: Path,
    *,
    actions: Mapping[str, Mapping[str, Any]],
    qrels: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    bindings: dict[str, dict[str, dict[str, Any]]] = {}
    for block in BLOCK_ORDER:
        action = actions.get(block)
        qrel = qrels.get(block)
        if not isinstance(action, Mapping) or not isinstance(qrel, Mapping):
            raise BircoP1PrivateSelectionError("private pack set is incomplete")
        action_semantic = _validate_action_pack(action, block=block)
        qrel_semantic = _validate_qrel_pack(
            qrel, block=block, expected_action_pack_sha256=action_semantic
        )
        action_name = ACTION_PACK_FILENAMES[block]
        qrel_name = QREL_PACK_FILENAMES[block]
        action_binding = _atomic_write_json(
            output_root / action_name, action, mode=0o600
        )
        qrel_binding = _atomic_write_json(
            output_root / qrel_name, qrel, mode=0o600
        )
        bindings[block] = {
            "action": {
                "relative_path": action_name,
                "semantic_sha256": action_semantic,
                **action_binding,
            },
            "qrels": {
                "relative_path": qrel_name,
                "semantic_sha256": qrel_semantic,
                **qrel_binding,
            },
        }
    return bindings


def _source_counts(
    records: Mapping[str, Sequence[QueryRecord]],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for family in FAMILIES:
        rows = tuple(records[family])
        candidate_ids = {cid for row in rows for cid in row.candidate_ids}
        result[family] = {
            "query_count": len(rows),
            "distinct_candidate_count": len(candidate_ids),
            "pool_entry_count": sum(len(row.candidate_ids) for row in rows),
        }
    return result


def _assert_public_receipt_shape(value: Mapping[str, Any]) -> None:
    """Structural guard against item content entering a public artifact."""

    forbidden_exact_keys = {
        "qid",
        "cid",
        "query",
        "document",
        "text",
        "task_objective",
        "qrel_values",
        "value",
        "candidate_ids",
        "candidate_texts",
        "items",
        "work_id",
    }

    def visit(node: object) -> None:
        if isinstance(node, Mapping):
            if forbidden_exact_keys.intersection(node):
                raise BircoP1PrivateSelectionError(
                    "public receipt contains a forbidden item field"
                )
            for child in node.values():
                visit(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(value)


def _terminal_failure(
    output_root: Path,
    *,
    stage: str,
    exc: BaseException,
    commitment_sha256: str | None,
) -> None:
    body = {
        "schema": f"{VERSION}_terminal_failure_v1",
        "version": VERSION,
        "study_id": STUDY_ID,
        "status": "terminal_no_retry_replay_resample_or_secret_rotation",
        "failure_stage": stage,
        "exception_type_sha256": hashlib.sha256(
            f"{type(exc).__module__}.{type(exc).__qualname__}".encode("utf-8")
        ).hexdigest(),
        "exception_message_sha256": hashlib.sha256(
            str(exc).encode("utf-8", errors="replace")
        ).hexdigest(),
        "selection_secret_commitment_sha256": commitment_sha256,
        "raw_qid_cid_query_document_or_qrel_value_published": False,
    }
    payload = self_hashed(body, "failure_sha256")
    try:
        _assert_public_receipt_shape(payload)
        _atomic_write_json(output_root / FAILURE_FILENAME, payload, mode=0o644)
    except BaseException:
        pass


def acquire_once(
    *,
    source_path: Path,
    qualification_path: Path,
    output_root: Path,
    contract: SourceContract,
    expected_qualification_self_sha256: str,
    task_objectives: Mapping[str, str],
    random_bytes: Callable[[int], bytes] | None = None,
    require_source_mode_0600: bool = True,
    _formal_capability: object | None = None,
) -> dict[str, Any]:
    """Perform one private selection from a pinned source.

    ``random_bytes`` exists solely for deterministic synthetic qualification.
    The formal wrapper does not accept or inject a caller-selected secret.
    Irrespective of the callable, it is invoked exactly once with ``32``.
    """

    if contract == FORMAL_CONTRACT:
        if (
            _formal_capability is not _FORMAL_EXECUTION_CAPABILITY
            or random_bytes is not None
            or expected_qualification_self_sha256 != QUALIFICATION_SELF_SHA256
            or require_source_mode_0600 is not True
        ):
            raise BircoP1PrivateSelectionError(
                "official source access is restricted to the fixed formal wrapper"
            )
    elif _formal_capability is not None:
        raise BircoP1PrivateSelectionError("formal capability used outside formal source")

    source = Path(source_path)
    qualification = Path(qualification_path)
    root = Path(output_root)
    objectives = _validated_objectives(task_objectives)
    qualification_result = verify_qualification_result(
        qualification,
        expected_self_sha256=expected_qualification_self_sha256,
        contract=contract,
    )
    _create_one_shot_root(root)
    stage = "write_one_shot_attempt_marker"
    commitment: str | None = None
    try:
        objective_hashes = {
            family: hashlib.sha256(objectives[family].encode("utf-8")).hexdigest()
            for family in FAMILIES
        }
        attempt_body = {
            "schema": f"{VERSION}_one_shot_marker_v1",
            "version": VERSION,
            "study_id": STUDY_ID,
            "status": "consumed_before_selection_secret_creation_or_source_parse",
            "source_identity": {
                "size_bytes": contract.source_size_bytes,
                "md5": contract.source_md5,
                "sha256": contract.source_sha256,
            },
            "qualification_self_sha256": expected_qualification_self_sha256,
            "task_objective_sha256s": objective_hashes,
            "retry_replay_resample_or_secret_rotation_authorized": False,
        }
        attempt = self_hashed(attempt_body, "attempt_marker_sha256")
        _atomic_write_json(
            root / ATTEMPT_MARKER_FILENAME, attempt, mode=0o600
        )

        stage = "create_exactly_one_selection_secret"
        generator = os.urandom if random_bytes is None else random_bytes
        secret = generator(32)
        if not isinstance(secret, bytes) or len(secret) != 32:
            raise BircoP1PrivateSelectionError(
                "the sole 32-byte random request returned an invalid secret"
            )
        _write_exclusive_bytes(root / SECRET_FILENAME, secret, mode=0o600)
        commitment = selection_secret_commitment(secret)

        stage = "publish_preparse_selection_commitment"
        commitment_body = {
            "schema": f"{VERSION}_selection_commitment_v1",
            "version": VERSION,
            "study_id": STUDY_ID,
            "status": "one_32_byte_secret_committed_before_source_parse",
            "attempt_marker_sha256": attempt["attempt_marker_sha256"],
            "selection_secret_commitment_sha256": commitment,
            "selection_secret_size_bytes": 32,
            "selection_secret_file_mode": "0600",
            "os_random_call_count": 1,
            "os_random_requested_bytes": 32,
            "source_identity": attempt_body["source_identity"],
            "qualification_self_sha256": expected_qualification_self_sha256,
            "selection_order": (
                "ascending_HMAC_SHA256_of_length_framed_study_family_qid_then_"
                "fixed_windows_0_10_20_30_40"
            ),
            "selection_secret_published": False,
            "raw_qid_cid_query_document_or_qrel_value_published": False,
        }
        commitment_marker = self_hashed(
            commitment_body, "commitment_marker_sha256"
        )
        _assert_public_receipt_shape(commitment_marker)
        commitment_file = _atomic_write_json(
            root / COMMITMENT_FILENAME, commitment_marker, mode=0o644
        )

        # This is the first source-file access.  It occurs only after the sole
        # secret and its public commitment are durable.  Byte hashing is
        # followed by the unique semantic decode below.
        stage = "verify_exact_source_bytes_after_commitment"
        source_raw = _read_stable_regular_bytes(
            source,
            label="pinned BIRCO source",
            expected_size=contract.source_size_bytes,
            expected_md5=contract.source_md5,
            expected_sha256=contract.source_sha256,
            expected_mode=0o600 if require_source_mode_0600 else None,
        )
        stage = "strict_source_parse_and_full_join_validation"
        records = parse_source_bytes(source_raw, contract=contract)
        stage = "fixed_HMAC_selection"
        selected = select_private_blocks(records, secret=secret)
        stage = "separate_label_free_and_numeric_qrel_packs"
        actions, qrels = build_private_packs(
            selected, secret=secret, task_objectives=objectives
        )
        stage = "persist_private_packs"
        pack_bindings = _persist_private_packs(
            root, actions=actions, qrels=qrels
        )

        stage = "persist_public_receipt"
        receipt_body = {
            "schema": f"{VERSION}_public_receipt_v1",
            "version": VERSION,
            "study_id": STUDY_ID,
            "status": "private_query_disjoint_four_block_selection_complete",
            "source_identity": attempt_body["source_identity"],
            "custody_binding": {
                "qualification_self_sha256": expected_qualification_self_sha256,
                "qualification_result_semantic_sha256": qualification_result[
                    "self_sha256"
                ],
                "selection_commitment_marker_sha256": commitment_marker[
                    "commitment_marker_sha256"
                ],
                "selection_commitment_file_sha256": commitment_file["file_sha256"],
                "selection_secret_commitment_sha256": commitment,
            },
            "selection_contract": {
                "family_order": list(FAMILIES),
                "block_order": list(BLOCK_ORDER),
                "per_family_per_block": PER_FAMILY_QUOTA,
                "selected_total": SELECTED_TOTAL,
                "query_disjoint_across_blocks": True,
                "HMAC_algorithm": "HMAC-SHA256",
                "HMAC_fields_in_exact_order": ["study_id", "family", "qid"],
            },
            "source_aggregates": _source_counts(records),
            "selected_counts_by_block_and_family": {
                block: {family: PER_FAMILY_QUOTA for family in FAMILIES}
                for block in BLOCK_ORDER
            },
            "task_objective_sha256s": objective_hashes,
            "private_pack_bindings": pack_bindings,
            "qrel_isolation": {
                "candidate_membership_from_source_qrel_keys": True,
                "numeric_values_absent_from_label_free_action_packs": True,
                "one_distinct_sealed_numeric_pack_per_block": True,
                "F_search_numeric_open_authorized": False,
            },
            "selection_secret_published": False,
            "raw_qid_cid_query_document_or_qrel_value_published": False,
            "online_evaluator_calls": 0,
            "retry_replay_resample_or_secret_rotation": 0,
        }
        receipt = self_hashed(receipt_body, "acquisition_sha256")
        _assert_public_receipt_shape(receipt)
        _atomic_write_json(root / PUBLIC_RECEIPT_FILENAME, receipt, mode=0o644)
        return receipt
    except BaseException as exc:
        _terminal_failure(
            root,
            stage=stage,
            exc=exc,
            commitment_sha256=commitment,
        )
        if isinstance(exc, BircoP1PrivateSelectionError):
            raise
        raise BircoP1PrivateSelectionError(
            "private one-shot selection failed terminally"
        ) from exc


def run_formal_selection(
    project_root: str | Path,
    *,
    task_objectives: Mapping[str, str],
) -> dict[str, Any]:
    """Run the sole formal route with all committed official identities fixed."""

    project = Path(project_root).resolve(strict=True)
    if not project.is_dir():
        raise BircoP1PrivateSelectionError("formal project root is invalid")
    _verify_formal_manifests(project)
    # No random injection, alternate path, source contract or qualification
    # identity is accepted by the formal wrapper.
    return acquire_once(
        source_path=project / SOURCE_RELATIVE,
        qualification_path=project / QUALIFICATION_RELATIVE,
        output_root=project / OUTPUT_RELATIVE,
        contract=FORMAL_CONTRACT,
        expected_qualification_self_sha256=QUALIFICATION_SELF_SHA256,
        task_objectives=task_objectives,
        random_bytes=None,
        require_source_mode_0600=True,
        _formal_capability=_FORMAL_EXECUTION_CAPABILITY,
    )


def _load_public_receipt(output_root: Path) -> dict[str, Any]:
    value, raw = _load_json_object(
        output_root / PUBLIC_RECEIPT_FILENAME, label="selection public receipt"
    )
    if raw != _canonical_bytes(value, newline=True):
        raise BircoP1PrivateSelectionError("selection public receipt is noncanonical")
    verify_self_hash(value, "acquisition_sha256")
    if (
        value.get("schema") != f"{VERSION}_public_receipt_v1"
        or value.get("version") != VERSION
        or value.get("study_id") != STUDY_ID
        or value.get("status")
        != "private_query_disjoint_four_block_selection_complete"
        or value.get("raw_qid_cid_query_document_or_qrel_value_published")
        is not False
    ):
        raise BircoP1PrivateSelectionError("selection public receipt drifted")
    return value


def _pack_binding(
    receipt: Mapping[str, Any], *, block: str, role: str
) -> Mapping[str, Any]:
    packs = receipt.get("private_pack_bindings")
    if not isinstance(packs, Mapping):
        raise BircoP1PrivateSelectionError("private pack bindings are absent")
    block_row = packs.get(block)
    if not isinstance(block_row, Mapping):
        raise BircoP1PrivateSelectionError("private block binding is absent")
    binding = block_row.get(role)
    if not isinstance(binding, Mapping):
        raise BircoP1PrivateSelectionError("private pack role binding is absent")
    expected_name = (
        ACTION_PACK_FILENAMES[block]
        if role == "action"
        else QREL_PACK_FILENAMES[block]
    )
    if (
        binding.get("relative_path") != expected_name
        or binding.get("mode_octal") != "0600"
        or _HEX64.fullmatch(str(binding.get("file_sha256"))) is None
        or _HEX64.fullmatch(str(binding.get("semantic_sha256"))) is None
        or type(binding.get("size_bytes")) is not int
        or binding["size_bytes"] <= 0
    ):
        raise BircoP1PrivateSelectionError("private pack binding drifted")
    return binding


def _read_bound_private_pack(
    output_root: Path,
    *,
    binding: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    path = output_root / str(binding["relative_path"])
    raw = _read_stable_regular_bytes(
        path,
        label=label,
        expected_size=int(binding["size_bytes"]),
        expected_sha256=str(binding["file_sha256"]),
        expected_mode=0o600,
    )
    value = _strict_json(raw, label=label)
    if not isinstance(value, Mapping) or raw != _canonical_bytes(value, newline=True):
        raise BircoP1PrivateSelectionError(f"{label} is noncanonical")
    return dict(value)


def write_block_open_authorization(
    path: Path,
    *,
    output_root: Path,
    block: str,
    action_archive_sha256s: Sequence[str],
    promotion_sha256: str | None = None,
) -> dict[str, Any]:
    """Create the controller's explicit, O_EXCL block-qrel capability.

    The controller must supply hashes of already complete immutable action
    archives.  ``M_search`` additionally requires the frozen valid A_hold
    promotion hash.  Creating or using any ``F_search`` qrel capability is
    forbidden.
    """

    if block == "F_search":
        raise BircoP1PrivateSelectionError(
            "F_search numeric qrels are never authorized to open"
        )
    if block not in BLOCK_ORDER:
        raise BircoP1PrivateSelectionError("block-open authorization is invalid")
    archives = tuple(action_archive_sha256s)
    if (
        not archives
        or len(set(archives)) != len(archives)
        or any(not isinstance(value, str) or _HEX64.fullmatch(value) is None for value in archives)
    ):
        raise BircoP1PrivateSelectionError("action archive bindings are invalid")
    archives = tuple(sorted(archives))
    if block == "M_search":
        if not isinstance(promotion_sha256, str) or _HEX64.fullmatch(
            promotion_sha256
        ) is None:
            raise BircoP1PrivateSelectionError(
                "M_search requires a valid A_hold promotion binding"
            )
    elif promotion_sha256 is not None:
        raise BircoP1PrivateSelectionError(
            "promotion binding is valid only for M_search"
        )
    root = Path(output_root)
    receipt = _load_public_receipt(root)
    action_binding = _pack_binding(receipt, block=block, role="action")
    qrel_binding = _pack_binding(receipt, block=block, role="qrels")
    body = {
        "schema": f"{VERSION}_block_qrel_open_authorization_v1",
        "version": VERSION,
        "study_id": STUDY_ID,
        "status": "numeric_qrel_open_authorized_after_immutable_action_archives",
        "block": block,
        "acquisition_sha256": receipt["acquisition_sha256"],
        "action_pack_semantic_sha256": action_binding["semantic_sha256"],
        "qrel_pack_semantic_sha256": qrel_binding["semantic_sha256"],
        "action_archive_sha256s": list(archives),
        "action_archives_complete_and_immutable": True,
        "numeric_qrel_open_authorized": True,
        "A_hold_promotion_sha256": promotion_sha256,
        "same_block_replay_authorized": False,
    }
    authorization = self_hashed(body, "authorization_sha256")
    _atomic_write_json(Path(path), authorization, mode=0o600)
    return authorization


def _validate_block_authorization(
    value: Mapping[str, Any],
    *,
    expected_authorization_sha256: str,
    receipt: Mapping[str, Any],
    block: str,
    action_binding: Mapping[str, Any],
    qrel_binding: Mapping[str, Any],
) -> str:
    observed = verify_self_hash(value, "authorization_sha256")
    archives = value.get("action_archive_sha256s")
    promotion = value.get("A_hold_promotion_sha256")
    if (
        _HEX64.fullmatch(expected_authorization_sha256) is None
        or not hmac.compare_digest(observed, expected_authorization_sha256)
        or set(value)
        != {
            "schema",
            "version",
            "study_id",
            "status",
            "block",
            "acquisition_sha256",
            "action_pack_semantic_sha256",
            "qrel_pack_semantic_sha256",
            "action_archive_sha256s",
            "action_archives_complete_and_immutable",
            "numeric_qrel_open_authorized",
            "A_hold_promotion_sha256",
            "same_block_replay_authorized",
            "authorization_sha256",
        }
        or value.get("schema")
        != f"{VERSION}_block_qrel_open_authorization_v1"
        or value.get("version") != VERSION
        or value.get("study_id") != STUDY_ID
        or value.get("status")
        != "numeric_qrel_open_authorized_after_immutable_action_archives"
        or value.get("block") != block
        or value.get("acquisition_sha256") != receipt.get("acquisition_sha256")
        or value.get("action_pack_semantic_sha256")
        != action_binding.get("semantic_sha256")
        or value.get("qrel_pack_semantic_sha256")
        != qrel_binding.get("semantic_sha256")
        or not isinstance(archives, list)
        or not archives
        or archives != sorted(set(archives))
        or any(not isinstance(row, str) or _HEX64.fullmatch(row) is None for row in archives)
        or value.get("action_archives_complete_and_immutable") is not True
        or value.get("numeric_qrel_open_authorized") is not True
        or value.get("same_block_replay_authorized") is not False
    ):
        raise BircoP1PrivateSelectionError("block-open authorization drifted")
    if block == "M_search":
        if not isinstance(promotion, str) or _HEX64.fullmatch(promotion) is None:
            raise BircoP1PrivateSelectionError(
                "M_search promotion authorization drifted"
            )
    elif promotion is not None:
        raise BircoP1PrivateSelectionError("unexpected promotion authorization")
    return observed


def open_block_qrels(
    *,
    output_root: Path,
    block: str,
    authorization_path: Path,
    expected_authorization_sha256: str,
) -> dict[str, Any]:
    """Consume one authorization before touching the corresponding qrel pack."""

    # This gate deliberately precedes receipt, authorization and pack I/O.
    if block == "F_search":
        raise BircoP1PrivateSelectionError(
            "F_search numeric qrels are permanently sealed"
        )
    if block not in BLOCK_ORDER:
        raise BircoP1PrivateSelectionError("qrel-open block is invalid")
    root = Path(output_root)
    receipt = _load_public_receipt(root)
    action_binding = _pack_binding(receipt, block=block, role="action")
    qrel_binding = _pack_binding(receipt, block=block, role="qrels")

    # Authorization is fully read, self-hash checked and matched to the
    # externally supplied capability hash before the qrel path is statted.
    authorization_raw = _read_stable_regular_bytes(
        Path(authorization_path),
        label=f"{block} qrel-open authorization",
        expected_mode=0o600,
    )
    authorization_value = _strict_json(
        authorization_raw, label=f"{block} qrel-open authorization"
    )
    if (
        not isinstance(authorization_value, Mapping)
        or authorization_raw
        != _canonical_bytes(authorization_value, newline=True)
    ):
        raise BircoP1PrivateSelectionError("block-open authorization is noncanonical")
    authorization = dict(authorization_value)
    authorization_sha256 = _validate_block_authorization(
        authorization,
        expected_authorization_sha256=expected_authorization_sha256,
        receipt=receipt,
        block=block,
        action_binding=action_binding,
        qrel_binding=qrel_binding,
    )

    open_body = {
        "schema": f"{VERSION}_qrel_open_marker_v1",
        "version": VERSION,
        "study_id": STUDY_ID,
        "status": "authorization_consumed_immediately_before_numeric_qrel_open",
        "block": block,
        "acquisition_sha256": receipt["acquisition_sha256"],
        "authorization_sha256": authorization_sha256,
        "same_block_second_open_authorized": False,
    }
    open_marker = self_hashed(open_body, "open_marker_sha256")
    _atomic_write_json(
        root / QREL_OPEN_MARKER_FILENAMES[block], open_marker, mode=0o600
    )

    qrel_pack = _read_bound_private_pack(
        root, binding=qrel_binding, label=f"{block} sealed qrel pack"
    )
    observed = _validate_qrel_pack(
        qrel_pack,
        block=block,
        expected_action_pack_sha256=str(action_binding["semantic_sha256"]),
    )
    if not hmac.compare_digest(observed, str(qrel_binding["semantic_sha256"])):
        raise BircoP1PrivateSelectionError("authorized qrel pack commitment drifted")
    return qrel_pack


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-acquire", action="store_true")
    parser.add_argument("--project", type=Path)
    parser.add_argument(
        "--task-objectives-json",
        type=Path,
        help="pre-frozen private JSON mapping of the three family names to objective text",
    )
    arguments = parser.parse_args(argv)
    if (
        not arguments.formal_acquire
        or arguments.project is None
        or arguments.task_objectives_json is None
    ):
        parser.error(
            "formal execution requires --formal-acquire, --project and "
            "--task-objectives-json"
        )
    objectives_value = load_task_objectives_manifest(
        arguments.task_objectives_json
    )
    run_formal_selection(arguments.project, task_objectives=objectives_value)
    return 0


def main() -> int:
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTION_PACK_FILENAMES",
    "ATTEMPT_MARKER_FILENAME",
    "BLOCK_ORDER",
    "BLOCK_WINDOWS",
    "BircoP1PrivateSelectionError",
    "BIRCO_REPOSITORY_COMMIT",
    "COMMITMENT_FILENAME",
    "FAILURE_FILENAME",
    "FAMILIES",
    "FORMAL_CONTRACT",
    "FamilyContract",
    "MAX_OBJECTIVE_CHARACTERS",
    "ORDER_HMAC_DOMAIN",
    "OUTPUT_RELATIVE",
    "PER_FAMILY_QUOTA",
    "PUBLIC_RECEIPT_FILENAME",
    "QREL_PACK_FILENAMES",
    "QREL_OPEN_MARKER_FILENAMES",
    "QUALIFICATION_RELATIVE",
    "QUALIFICATION_SELF_SHA256",
    "QueryRecord",
    "SELECTED_PER_FAMILY",
    "SELECTED_TOTAL",
    "SOURCE_MD5",
    "SOURCE_SHA256",
    "SOURCE_SIZE_BYTES",
    "SECRET_FILENAME",
    "STUDY_ID",
    "TASK_OBJECTIVES_MANIFEST_SCHEMA",
    "SourceContract",
    "VERSION",
    "acquire_once",
    "build_private_packs",
    "common_projection_sha256",
    "load_task_objectives_manifest",
    "opaque_work_id",
    "open_block_qrels",
    "parse_source_bytes",
    "run_formal_selection",
    "select_private_blocks",
    "selection_hmac_digest",
    "selection_hmac_message",
    "selection_secret_commitment",
    "self_hashed",
    "stable_hash",
    "validate_source_payload",
    "verify_qualification_result",
    "verify_self_hash",
    "write_block_open_authorization",
]
