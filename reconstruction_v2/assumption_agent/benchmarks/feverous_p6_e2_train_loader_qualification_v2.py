"""Real, aggregate-only TRAIN loader qualification for source epoch v2.

This qualification is deliberately earlier and narrower than acquisition.  It
streams the one frozen TRAIN annotation file, uses the production strict JSON
decoder and exact six-empty-string sentinel predicate, and retains no records.
It never creates or reads a selection secret, opens the Wikipedia database,
forms a cohort, retrieves, scores, or accesses DEV/TEST.  The deterministic
expected receipt may be preregistered; a manifest exists only after the real
stream reaches normal exhaustion and matches it exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
from typing import Any, Callable

from assumption_agent.benchmarks import feverous_p6_e2_formal_source_v1 as formal_source
from assumption_agent.benchmarks import feverous_p6_e2_source_adapter_v1 as source_adapter


VERSION = "feverous_p6_e2_train_loader_qualification_v2"
SCHEMA = f"{VERSION}_receipt"
MANIFEST_RELATIVE = Path(
    "manifests/feverous_p6_e2_train_loader_qualification_v2.json"
)
ANNOTATION_RELATIVE = Path("artifacts/feverous_official_source_v1") / (
    formal_source.FROZEN_ANNOTATION_BASENAME
)
FORMAL_SOURCE_CODE_RELATIVE = Path(
    "assumption_agent/benchmarks/feverous_p6_e2_formal_source_v1.py"
)
SOURCE_ADAPTER_CODE_RELATIVE = Path(
    "assumption_agent/benchmarks/feverous_p6_e2_source_adapter_v1.py"
)


class FeverousTrainLoaderQualificationError(RuntimeError):
    """The real aggregate loader qualification or its code binding drifted."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise FeverousTrainLoaderQualificationError(
            "qualification value is not canonical JSON"
        ) from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise FeverousTrainLoaderQualificationError(
            "qualification code file cannot be hashed"
        ) from exc
    return digest.hexdigest()


def _callable_source_sha256(function: Callable[..., object]) -> str:
    try:
        source = inspect.getsource(function).encode("utf-8", errors="strict")
    except (OSError, TypeError, UnicodeEncodeError) as exc:
        raise FeverousTrainLoaderQualificationError(
            "qualification predicate source is unavailable"
        ) from exc
    return hashlib.sha256(source).hexdigest()


def _canonical_project(project: str | Path) -> Path:
    try:
        root = Path(project).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FeverousTrainLoaderQualificationError(
            "project root is unavailable"
        ) from exc
    if not root.is_dir() or root.is_symlink():
        raise FeverousTrainLoaderQualificationError("project root is unsafe")
    return root


def _code_bindings(project: Path) -> dict[str, str]:
    return {
        "formal_source_file_sha256": _sha256_file(
            project / FORMAL_SOURCE_CODE_RELATIVE
        ),
        "source_adapter_file_sha256": _sha256_file(
            project / SOURCE_ADAPTER_CODE_RELATIVE
        ),
        "strict_json_decoder_source_sha256": _callable_source_sha256(
            formal_source._decode_json_line
        ),
        "exact_blank_sentinel_predicate_source_sha256": (
            _callable_source_sha256(source_adapter._is_blank_sentinel)
        ),
    }


def expected_qualification_receipt(project: str | Path) -> dict[str, Any]:
    """Return the receipt a successful real scan must produce exactly."""

    root = _canonical_project(project)
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "real_train_loader_aggregate_qualified_before_v2_secret",
        "source_split": "TRAIN",
        "annotation_relative": ANNOTATION_RELATIVE.as_posix(),
        "annotation_basename": formal_source.FROZEN_ANNOTATION_BASENAME,
        "annotation_size_bytes": formal_source.FROZEN_ANNOTATION_SIZE_BYTES,
        "annotation_file_sha256": formal_source.FROZEN_ANNOTATION_SHA256,
        "annotation_physical_rows": (
            formal_source.FROZEN_ANNOTATION_NONBLANK_ROWS
            + formal_source.FROZEN_ANNOTATION_BLANK_SENTINEL_ROWS
        ),
        "annotation_nonblank_rows": (
            formal_source.FROZEN_ANNOTATION_NONBLANK_ROWS
        ),
        "annotation_blank_sentinel_rows": (
            formal_source.FROZEN_ANNOTATION_BLANK_SENTINEL_ROWS
        ),
        "exact_count_identity": "71292=71291+1",
        **_code_bindings(root),
        "annotation_file_read_count": 1,
        "annotation_records_retained_after_scan": 0,
        "selection_secret_generated_or_read": False,
        "candidate_adapter_invoked": False,
        "cohort_or_block_selection_invoked": False,
        "wikipedia_database_stated_hashed_opened_or_queried": False,
        "retrieval_action_utility_evaluator_or_scoring_calls": 0,
        "claim_corpus_gold_label_or_outcome_rows_persisted": False,
        "development_or_test_source_accessed": False,
        "online_evaluator_calls": 0,
    }
    return {**body, "qualification_sha256": stable_hash(body)}


def run_real_train_loader_qualification(project: str | Path) -> Mapping[str, Any]:
    """Stream and qualify the exact TRAIN annotation without retaining rows."""

    root = _canonical_project(project)
    path = root / ANNOTATION_RELATIVE
    try:
        before = path.lstat()
    except OSError as exc:
        raise FeverousTrainLoaderQualificationError(
            "frozen TRAIN annotation is unavailable"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != formal_source.FROZEN_TRAIN_SOURCE_SPEC.required_mode
        or before.st_size != formal_source.FROZEN_ANNOTATION_SIZE_BYTES
    ):
        raise FeverousTrainLoaderQualificationError(
            "frozen TRAIN annotation metadata drifted"
        )

    digest = hashlib.sha256()
    total_bytes = 0
    physical_rows = 0
    nonblank_rows = 0
    blank_rows = 0
    try:
        with path.open("rb", buffering=1024 * 1024) as handle:
            opened = os.fstat(handle.fileno())
            if (opened.st_dev, opened.st_ino, opened.st_size) != (
                before.st_dev,
                before.st_ino,
                before.st_size,
            ):
                raise FeverousTrainLoaderQualificationError(
                    "opened TRAIN annotation identity drifted"
                )
            for raw_line in handle:
                total_bytes += len(raw_line)
                digest.update(raw_line)
                physical_rows += 1
                content = raw_line[:-1] if raw_line.endswith(b"\n") else raw_line
                if content.endswith(b"\r"):
                    content = content[:-1]
                if not content:
                    raise FeverousTrainLoaderQualificationError(
                        "TRAIN JSONL contains an empty physical line"
                    )
                record = formal_source._decode_json_line(content)
                if source_adapter._is_blank_sentinel(record):
                    blank_rows += 1
                else:
                    nonblank_rows += 1
    except OSError as exc:
        raise FeverousTrainLoaderQualificationError(
            "real TRAIN loader qualification read failed"
        ) from exc
    try:
        after = path.lstat()
    except OSError as exc:
        raise FeverousTrainLoaderQualificationError(
            "TRAIN annotation changed during qualification"
        ) from exc
    if (
        path.is_symlink()
        or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        or total_bytes != formal_source.FROZEN_ANNOTATION_SIZE_BYTES
        or digest.hexdigest() != formal_source.FROZEN_ANNOTATION_SHA256
        or physical_rows
        != formal_source.FROZEN_ANNOTATION_NONBLANK_ROWS
        + formal_source.FROZEN_ANNOTATION_BLANK_SENTINEL_ROWS
        or nonblank_rows != formal_source.FROZEN_ANNOTATION_NONBLANK_ROWS
        or blank_rows != formal_source.FROZEN_ANNOTATION_BLANK_SENTINEL_ROWS
    ):
        raise FeverousTrainLoaderQualificationError(
            "real TRAIN loader aggregate differs from preregistration"
        )
    return expected_qualification_receipt(root)


def verify_train_loader_qualification(project: str | Path) -> Mapping[str, Any]:
    """Verify the persisted aggregate receipt without reopening TRAIN data."""

    root = _canonical_project(project)
    path = root / MANIFEST_RELATIVE
    if path.is_symlink() or not path.is_file():
        raise FeverousTrainLoaderQualificationError(
            "TRAIN loader qualification receipt is unavailable"
        )
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeverousTrainLoaderQualificationError(
            "TRAIN loader qualification receipt is invalid"
        ) from exc
    expected = expected_qualification_receipt(root)
    if (
        not isinstance(value, Mapping)
        or raw != _canonical_bytes(value) + b"\n"
        or dict(value) != expected
    ):
        raise FeverousTrainLoaderQualificationError(
            "TRAIN loader qualification receipt drifted"
        )
    return value


__all__ = [
    "ANNOTATION_RELATIVE",
    "FeverousTrainLoaderQualificationError",
    "FORMAL_SOURCE_CODE_RELATIVE",
    "MANIFEST_RELATIVE",
    "SCHEMA",
    "SOURCE_ADAPTER_CODE_RELATIVE",
    "VERSION",
    "expected_qualification_receipt",
    "run_real_train_loader_qualification",
    "stable_hash",
    "verify_train_loader_qualification",
]
