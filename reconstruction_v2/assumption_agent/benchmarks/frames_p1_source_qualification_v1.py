"""One-shot aggregate-only qualification for the pinned FRAMES source.

This is a source-integrity and capacity check, not a retrieval, model, or
performance gate.  The sole formal call consumes a marker before validating
the frozen custody and implementation receipts.  It reads the source through
one no-follow file descriptor, verifies that descriptor before and after the
parse, excludes all public-viewer and exact-content collisions, and emits only
aggregate schema/capacity facts.  No source, family, quota, marker, or result
path is configurable.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Iterator, Mapping, Sequence, TextIO
import unicodedata
from urllib.parse import quote, unquote_to_bytes, urlsplit


VERSION = "frames_p1_source_qualification_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GIT_TOP = PROJECT_ROOT.parent
GIT_PROJECT_PREFIX = "reconstruction_v2"
GIT_EXECUTABLE = Path("/usr/bin/git")
QUALIFIER_PATH = Path(__file__).resolve()
TEST_PATH = PROJECT_ROOT / "tests/test_frames_p1_source_qualification_v1.py"
SOURCE_PATH = PROJECT_ROOT / "artifacts/frames_p1_official_source_v1/test.tsv"
DOWNLOAD_RECEIPT_PATH = (
    PROJECT_ROOT / "manifests/frames_p1_source_download_receipt_v1.json"
)
CUSTODY_PATH = PROJECT_ROOT / "manifests/frames_p1_source_custody_v1.json"
BRIGHT_DISPOSITION_PATH = (
    PROJECT_ROOT
    / "manifests/bright_p17_postterminal_view_exposure_disposition_v1.json"
)
FREEZE_PATH = (
    PROJECT_ROOT / "manifests/frames_p1_source_qualification_freeze_v1.json"
)
MARKER_PATH = (
    PROJECT_ROOT
    / "artifacts/frames_p1_source_qualification_v1/qualification.one_shot_marker.json"
)
FAILURE_PATH = (
    PROJECT_ROOT
    / "artifacts/frames_p1_source_qualification_v1/qualification.terminal_failure.json"
)
RESULT_PATH = PROJECT_ROOT / "manifests/frames_p1_source_qualification_result_v1.json"

EXPECTED_REPOSITORY = "google/frames-benchmark"
EXPECTED_REVISION = "58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef"
EXPECTED_DATASET_FILE = "test.tsv"
EXPECTED_DOWNLOAD_URL = (
    "https://huggingface.co/datasets/google/frames-benchmark/resolve/"
    f"{EXPECTED_REVISION}/{EXPECTED_DATASET_FILE}"
)
EXPECTED_SOURCE_GIT_BLOB_SHA1 = "cea20270ebb661d0ee1cdb15598c2c8fcba31025"
EXPECTED_SOURCE_SIZE_BYTES = 484887
EXPECTED_CUSTODY_SELF_SHA256 = (
    "47e1b3b70bdef530db46b4e25b85af08240e713467e4c0609b0a83d57b923263"
)
EXPECTED_BRIGHT_DISPOSITION_SELF_SHA256 = (
    "b84ab5f7e35a2286f2ead074d24e60582679c7b6f7e1022fa10c3f420c2c0dd0"
)

EXPECTED_ROW_COUNT = 824
EXPOSED_FORMATION_INTERVAL = (0, 100)
VIEWER_SCHEMA_VALIDATION_INTERVAL = (0, 90)
MEASUREMENT_MINIMUM_ID = 100
BLOCK_NAMES = ("A_form", "F_search", "A_hold", "M_search")
FAMILY_QUOTA_PER_BLOCK = 12
REQUIRED_ELIGIBLE_PER_FAMILY = len(BLOCK_NAMES) * FAMILY_QUOTA_PER_BLOCK
FAMILIES = ("constraint_postprocess", "structured", "temporal")
POOL_ORDER_DOMAIN = "frames-p1-deterministic-page-disjoint-capacity-v1"
ALLOWED_REASONING_TYPES = frozenset(
    {
        "Multiple constraints",
        "Numerical reasoning",
        "Post processing",
        "Tabular reasoning",
        "Temporal reasoning",
    }
)
EXPECTED_COLUMNS = (
    "Unnamed: 0",
    "Prompt",
    "Answer",
    "wikipedia_link_1",
    "wikipedia_link_2",
    "wikipedia_link_3",
    "wikipedia_link_4",
    "wikipedia_link_5",
    "wikipedia_link_6",
    "wikipedia_link_7",
    "wikipedia_link_8",
    "wikipedia_link_9",
    "wikipedia_link_10",
    "wikipedia_link_11+",
    "reasoning_types",
    "wiki_links",
)

DOWNLOAD_RECEIPT_SCHEMA = "frames_p1_source_download_receipt_v1"
CUSTODY_SCHEMA = "frames_p1_source_custody_v1"
BRIGHT_DISPOSITION_SCHEMA = "bright_p17_postterminal_view_exposure_disposition_v1"
FREEZE_SCHEMA = "frames_p1_source_qualification_freeze_v1"
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_BAD_PERCENT = re.compile(r"%(?![0-9A-Fa-f]{2})")


class FramesP1SourceQualificationError(RuntimeError):
    """The fixed source or aggregate-only contract failed closed."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise FramesP1SourceQualificationError(
            "qualification value is not canonical JSON"
        ) from exc


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise FramesP1SourceQualificationError("bound file is unreadable") from exc
    return digest.hexdigest()


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    raw = _canonical_bytes(value)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise FramesP1SourceQualificationError(
            "one-shot qualification path is already consumed"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if path.is_symlink() or path.read_bytes() != raw:
        raise FramesP1SourceQualificationError(
            "qualification receipt reopen verification failed"
        )
    return hashlib.sha256(raw).hexdigest()


def _load_canonical(path: Path, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise FramesP1SourceQualificationError(f"{field} is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FramesP1SourceQualificationError(f"{field} is invalid") from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise FramesP1SourceQualificationError(f"{field} is not canonical JSON")
    body = dict(value)
    declared = body.pop("self_sha256", None)
    if (
        not isinstance(declared, str)
        or _HEX64.fullmatch(declared) is None
        or _semantic_hash(body) != declared
    ):
        raise FramesP1SourceQualificationError(f"{field} self hash drifted")
    return value


def _validate_custody() -> dict[str, Any]:
    value = _load_canonical(CUSTODY_PATH, "source custody")
    if (
        value.get("schema") != CUSTODY_SCHEMA
        or value.get("self_sha256") != EXPECTED_CUSTODY_SELF_SHA256
        or value.get("dataset_repository") != EXPECTED_REPOSITORY
        or value.get("dataset_revision") != EXPECTED_REVISION
        or value.get("dataset_file") != EXPECTED_DATASET_FILE
        or value.get("pinned_download_url") != EXPECTED_DOWNLOAD_URL
        or value.get("official_git_blob_sha1") != EXPECTED_SOURCE_GIT_BLOB_SHA1
        or value.get("official_source_size_bytes") != EXPECTED_SOURCE_SIZE_BYTES
        or value.get("conservative_exposed_row_id_interval_half_open")
        != list(EXPOSED_FORMATION_INTERVAL)
        or value.get("viewer_schema_validation_row_id_interval_half_open")
        != list(VIEWER_SCHEMA_VALIDATION_INTERVAL)
        or value.get("viewer_schema_validation_output_question_answer_or_URL_count")
        != 0
        or value.get("measurement_row_minimum_id") != MEASUREMENT_MINIMUM_ID
        or value.get("measurement_use_of_rows_0_through_99_allowed") is not False
        or value.get("strict_source_download_before_freeze") is not False
        or value.get("pre_freeze_nonsemantic_source_byte_stream_attempt_count") != 1
        or value.get("pre_freeze_source_stream_persisted") is not False
        or value.get("pre_freeze_source_row_semantically_parsed") is not False
        or value.get("pre_freeze_source_content_output_count") != 0
        or value.get("bright_p17_exposure_disposition_self_sha256")
        != EXPECTED_BRIGHT_DISPOSITION_SELF_SHA256
        or value.get("online_evaluator_or_API_calls_allowed") is not False
    ):
        raise FramesP1SourceQualificationError("source custody contract drifted")
    return value


def _validate_bright_disposition() -> dict[str, Any]:
    value = _load_canonical(BRIGHT_DISPOSITION_PATH, "BRIGHT disposition")
    if (
        value.get("schema") != BRIGHT_DISPOSITION_SCHEMA
        or value.get("self_sha256")
        != EXPECTED_BRIGHT_DISPOSITION_SELF_SHA256
        or value.get("P17_C_confirm_question_blind_after_exposure") is not False
        or value.get("P17_or_BRIGHT_reuse_for_candidate_or_measurement_allowed")
        is not False
        or value.get("label_gold_or_score_semantic_read_count") != 0
    ):
        raise FramesP1SourceQualificationError("BRIGHT disposition drifted")
    return value


def _validate_freeze() -> dict[str, Any]:
    value = _load_canonical(FREEZE_PATH, "qualification freeze")
    required = {
        "schema",
        "status",
        "implementation_commit",
        "file_bindings",
        "source_repository_object",
        "pre_freeze_public_viewer_validation",
        "pre_freeze_nonsemantic_source_stream",
        "formal_source_file_present_at_freeze",
        "source_download_receipt_present_at_freeze",
        "formal_qualification_attempt_count_at_freeze",
        "model_action_or_score_count_at_freeze",
        "online_evaluator_or_API_calls_at_freeze",
        "self_sha256",
    }
    bindings = value.get("file_bindings")
    source_object = value.get("source_repository_object")
    viewer = value.get("pre_freeze_public_viewer_validation")
    source_stream = value.get("pre_freeze_nonsemantic_source_stream")
    expected_bindings = {
        "qualifier": (
            QUALIFIER_PATH,
            "assumption_agent/benchmarks/frames_p1_source_qualification_v1.py",
        ),
        "tests": (TEST_PATH, "tests/test_frames_p1_source_qualification_v1.py"),
        "source_custody": (
            CUSTODY_PATH,
            "manifests/frames_p1_source_custody_v1.json",
        ),
        "bright_exposure_disposition": (
            BRIGHT_DISPOSITION_PATH,
            "manifests/bright_p17_postterminal_view_exposure_disposition_v1.json",
        ),
    }
    if (
        set(value) != required
        or value.get("schema") != FREEZE_SCHEMA
        or value.get("status")
        != "frozen_before_persisted_source_download_and_formal_qualification"
        or not isinstance(value.get("implementation_commit"), str)
        or _HEX40.fullmatch(str(value.get("implementation_commit"))) is None
        or value.get("implementation_commit") == "0" * 40
        or not isinstance(bindings, dict)
        or set(bindings) != set(expected_bindings)
        or source_object
        != {
            "dataset_repository": EXPECTED_REPOSITORY,
            "dataset_revision": EXPECTED_REVISION,
            "dataset_file": EXPECTED_DATASET_FILE,
            "git_blob_sha1": EXPECTED_SOURCE_GIT_BLOB_SHA1,
            "size_bytes": EXPECTED_SOURCE_SIZE_BYTES,
        }
        or viewer
        != {
            "row_id_interval_half_open": list(VIEWER_SCHEMA_VALIDATION_INTERVAL),
            "header_matches_expected": True,
            "reasoning_types_pipe_delimited_scalar_rows": 90,
            "wiki_links_python_list_string_rows": 90,
            "first_10_plus_11th_scalar_columns_match_rows": 90,
            "question_answer_or_URL_values_output_count": 0,
        }
        or source_stream
        != {
            "attempt_count": 1,
            "reason": "raw_URL_returned_content_instead_of_expected_git_pointer",
            "persisted": False,
            "row_or_cell_semantically_parsed": False,
            "question_answer_URL_or_row_value_output_count": 0,
            "candidate_metric_quota_or_parser_adaptation_from_content": False,
        }
        or value.get("formal_source_file_present_at_freeze") is not False
        or value.get("source_download_receipt_present_at_freeze") is not False
        or value.get("formal_qualification_attempt_count_at_freeze") != 0
        or value.get("model_action_or_score_count_at_freeze") != 0
        or value.get("online_evaluator_or_API_calls_at_freeze") != 0
    ):
        raise FramesP1SourceQualificationError("qualification freeze drifted")
    for role, (path, relative) in expected_bindings.items():
        row = bindings.get(role)
        if (
            not isinstance(row, dict)
            or set(row) != {"relative_path", "sha256"}
            or row.get("relative_path") != relative
            or not isinstance(row.get("sha256"), str)
            or _HEX64.fullmatch(str(row.get("sha256"))) is None
            or _file_sha256(path) != row.get("sha256")
        ):
            raise FramesP1SourceQualificationError(
                f"qualification freeze file binding drifted: {role}"
            )
    _verify_git_freeze_bindings(
        str(value["implementation_commit"]), bindings, expected_bindings
    )
    return value


def _git_call(arguments: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    if not GIT_EXECUTABLE.is_file():
        raise FramesP1SourceQualificationError("fixed Git executable is unavailable")
    environment = {
        "HOME": "/nonexistent",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    try:
        return subprocess.run(
            [
                str(GIT_EXECUTABLE),
                "--no-replace-objects",
                "-c",
                "core.hooksPath=/dev/null",
                "-C",
                str(GIT_TOP),
                *arguments,
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FramesP1SourceQualificationError("fixed Git verification failed") from exc


def _verify_git_freeze_bindings(
    commit: str,
    bindings: Mapping[str, Any],
    expected_bindings: Mapping[str, tuple[Path, str]],
) -> None:
    commit_check = _git_call(["cat-file", "-e", f"{commit}^{{commit}}"])
    ancestor_check = _git_call(["merge-base", "--is-ancestor", commit, "HEAD"])
    if commit_check.returncode != 0 or ancestor_check.returncode != 0:
        raise FramesP1SourceQualificationError(
            "implementation commit is not a real current-history ancestor"
        )
    for role, (_, relative) in expected_bindings.items():
        object_path = f"{GIT_PROJECT_PREFIX}/{relative}"
        blob = _git_call(["cat-file", "blob", f"{commit}:{object_path}"])
        row = bindings[role]
        if (
            blob.returncode != 0
            or hashlib.sha256(blob.stdout).hexdigest() != row["sha256"]
        ):
            raise FramesP1SourceQualificationError(
                f"implementation commit blob binding drifted: {role}"
            )


def _validate_download_receipt() -> dict[str, Any]:
    value = _load_canonical(DOWNLOAD_RECEIPT_PATH, "source download receipt")
    required = {
        "schema",
        "status",
        "dataset_repository",
        "dataset_revision",
        "dataset_file",
        "pinned_download_url",
        "fixed_local_source_path",
        "source_file_sha256",
        "source_git_blob_sha1",
        "source_size_bytes",
        "source_content_semantically_opened_during_download",
        "external_network_use",
        "online_evaluator_or_API_calls",
        "self_sha256",
    }
    source_hash = value.get("source_file_sha256")
    if (
        set(value) != required
        or value.get("schema") != DOWNLOAD_RECEIPT_SCHEMA
        or value.get("status") != "downloaded_exact_pinned_official_source"
        or value.get("dataset_repository") != EXPECTED_REPOSITORY
        or value.get("dataset_revision") != EXPECTED_REVISION
        or value.get("dataset_file") != EXPECTED_DATASET_FILE
        or value.get("pinned_download_url") != EXPECTED_DOWNLOAD_URL
        or value.get("fixed_local_source_path")
        != "artifacts/frames_p1_official_source_v1/test.tsv"
        or not isinstance(source_hash, str)
        or _HEX64.fullmatch(source_hash) is None
        or value.get("source_git_blob_sha1") != EXPECTED_SOURCE_GIT_BLOB_SHA1
        or value.get("source_size_bytes") != EXPECTED_SOURCE_SIZE_BYTES
        or value.get("source_content_semantically_opened_during_download") is not False
        or value.get("external_network_use") != "pinned_source_download_only"
        or value.get("online_evaluator_or_API_calls") != 0
    ):
        raise FramesP1SourceQualificationError("source download receipt drifted")
    return value


def _hash_descriptor(descriptor: int, size: int) -> tuple[str, str, int]:
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1()
    git_blob.update(f"blob {size}\0".encode("ascii"))
    observed = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        sha256.update(block)
        git_blob.update(block)
        observed += len(block)
    return sha256.hexdigest(), git_blob.hexdigest(), observed


def _source_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


@contextmanager
def _open_bound_source(download: Mapping[str, Any]) -> Iterator[TextIO]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(SOURCE_PATH, flags)
        before = os.fstat(descriptor)
        path_before = os.stat(SOURCE_PATH, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or _source_identity(before) != _source_identity(path_before)
            or before.st_size != EXPECTED_SOURCE_SIZE_BYTES
        ):
            raise FramesP1SourceQualificationError(
                "downloaded source is not one bound regular file"
            )
        sha256, git_blob, observed = _hash_descriptor(descriptor, before.st_size)
        if (
            observed != before.st_size
            or sha256 != download.get("source_file_sha256")
            or git_blob != EXPECTED_SOURCE_GIT_BLOB_SHA1
        ):
            raise FramesP1SourceQualificationError("downloaded source binding drifted")

        os.lseek(descriptor, 0, os.SEEK_SET)
        binary = os.fdopen(os.dup(descriptor), "rb", closefd=True)
        text = io.TextIOWrapper(binary, encoding="utf-8", newline="")
        try:
            yield text
        finally:
            text.close()

        after = os.fstat(descriptor)
        path_after = os.stat(SOURCE_PATH, follow_symlinks=False)
        post_sha256, post_git_blob, post_observed = _hash_descriptor(
            descriptor, after.st_size
        )
        if (
            _source_identity(before) != _source_identity(after)
            or _source_identity(after) != _source_identity(path_after)
            or post_observed != before.st_size
            or post_sha256 != sha256
            or post_git_blob != git_blob
        ):
            raise FramesP1SourceQualificationError(
                "downloaded source changed during qualification"
            )
    except OSError as exc:
        raise FramesP1SourceQualificationError(
            "downloaded source descriptor validation failed"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _reasoning_types(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(" | "))
    if (
        not values
        or any(not value for value in values)
        or " | ".join(values) != raw
        or len(values) != len(set(values))
        or any(value not in ALLOWED_REASONING_TYPES for value in values)
    ):
        raise FramesP1SourceQualificationError("reasoning type grammar drifted")
    return values


def _family(types: Sequence[str]) -> str:
    values = set(types)
    if "Temporal reasoning" in values:
        return "temporal"
    if values & {"Numerical reasoning", "Tabular reasoning"}:
        return "structured"
    return "constraint_postprocess"


def _parsed_links(raw: str) -> tuple[str, ...]:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise FramesP1SourceQualificationError("wiki_links grammar drifted") from exc
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(link, str) or not link for link in value)
    ):
        raise FramesP1SourceQualificationError("wiki_links grammar drifted")
    return tuple(value)


def _normalized_text_identity(value: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    if not normalized:
        raise FramesP1SourceQualificationError("empty normalized text identity")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _normalized_canonical_links(links: Sequence[str]) -> tuple[str, ...] | None:
    normalized: list[str] = []
    for link in links:
        if (
            link != link.strip()
            or any(character.isspace() or ord(character) < 32 for character in link)
            or any(ord(character) == 127 for character in link)
            or _BAD_PERCENT.search(link) is not None
            or "\\" in link
        ):
            return None
        try:
            split = urlsplit(link)
            port = split.port
            decoded_path = unquote_to_bytes(split.path).decode("utf-8", "strict")
        except (UnicodeDecodeError, ValueError):
            return None
        decoded_path = unicodedata.normalize("NFC", decoded_path)
        if (
            split.scheme != "https"
            or split.netloc != "en.wikipedia.org"
            or split.hostname != "en.wikipedia.org"
            or not decoded_path.startswith("/wiki/")
            or decoded_path == "/wiki/"
            or split.query
            or split.username is not None
            or split.password is not None
            or port is not None
            or "\\" in decoded_path
            or any(ord(character) < 32 or ord(character) == 127 for character in decoded_path)
        ):
            return None
        title = decoded_path[len("/wiki/") :].replace(" ", "_")
        if (
            not title
            or title.casefold().startswith("special:")
            or any(part in {".", ".."} for part in title.split("/"))
        ):
            return None
        encoded_path = "/wiki/" + quote(title, safe="/:()!,'-._~")
        normalized.append("https://en.wikipedia.org" + encoded_path)
    if len(normalized) != len(set(normalized)):
        return None
    return tuple(normalized)


def _validate_redundant_link_columns(
    row: Mapping[str, str], links: Sequence[str]
) -> None:
    for index in range(1, 11):
        expected = links[index - 1] if index <= len(links) else ""
        if row[f"wikipedia_link_{index}"] != expected:
            raise FramesP1SourceQualificationError(
                "redundant wikipedia link column drifted"
            )
    tail = row["wikipedia_link_11+"]
    if len(links) <= 10 and tail:
        raise FramesP1SourceQualificationError("11+ link column drifted")
    if len(links) == 11 and tail != links[10]:
        raise FramesP1SourceQualificationError("11th link column drifted")
    if len(links) > 11:
        residual = tail
        for link in sorted(links[10:], key=len, reverse=True):
            if residual.count(link) != 1:
                raise FramesP1SourceQualificationError(
                    "11+ aggregate link column drifted"
                )
            residual = residual.replace(link, "", 1)
        if any(character not in " \t\r\n[](),'\"|;" for character in residual):
            raise FramesP1SourceQualificationError(
                "11+ aggregate link column has unbound content"
            )


def _pool_order(row_id: int) -> bytes:
    return hashlib.sha256(
        f"{POOL_ORDER_DOMAIN}\0{row_id}".encode("ascii")
    ).digest()


def _deterministic_disjoint_pool(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    counts = {family: 0 for family in FAMILIES}
    collision_skips = {family: 0 for family in FAMILIES}
    prompts: set[str] = set()
    answers: set[str] = set()
    pages: set[str] = set()
    for candidate in sorted(candidates, key=lambda row: _pool_order(int(row["row_id"]))):
        family = str(candidate["family"])
        if counts[family] >= REQUIRED_ELIGIBLE_PER_FAMILY:
            continue
        links = set(candidate["links"])
        if (
            candidate["prompt_identity"] in prompts
            or candidate["answer_identity"] in answers
            or not pages.isdisjoint(links)
        ):
            collision_skips[family] += 1
            continue
        prompts.add(str(candidate["prompt_identity"]))
        answers.add(str(candidate["answer_identity"]))
        pages.update(links)
        counts[family] += 1
    return counts, collision_skips


def _terminal_failure(stage: str, exc: BaseException) -> None:
    body = {
        "schema": f"{VERSION}_terminal_failure_v1",
        "status": "terminal_FRAMES_source_route_no_retry",
        "failure_stage": stage,
        "failure_type_sha256": hashlib.sha256(
            f"{type(exc).__module__}.{type(exc).__qualname__}".encode("utf-8")
        ).hexdigest(),
        "question_answer_url_or_row_id_values_output_count": 0,
        "model_action_or_score_count": 0,
        "external_network_calls": 0,
        "online_evaluator_or_API_calls": 0,
        "retry_replay_resample_or_quota_revision": 0,
    }
    try:
        _write_exclusive(FAILURE_PATH, {**body, "self_sha256": _semantic_hash(body)})
    except BaseException:
        pass


def run_source_qualification() -> dict[str, object]:
    """Consume the sole fixed aggregate-only FRAMES source qualification."""

    if MARKER_PATH.exists() or MARKER_PATH.is_symlink():
        raise FramesP1SourceQualificationError(
            "FRAMES source qualification marker is already consumed"
        )
    if (
        FAILURE_PATH.exists()
        or FAILURE_PATH.is_symlink()
        or RESULT_PATH.exists()
        or RESULT_PATH.is_symlink()
    ):
        raise FramesP1SourceQualificationError(
            "FRAMES source qualification terminal path is already consumed"
        )
    marker_body = {
        "schema": f"{VERSION}_one_shot_marker_v1",
        "status": "started_before_frozen_receipt_validation_or_source_row_parse",
        "expected_source_custody_self_sha256": EXPECTED_CUSTODY_SELF_SHA256,
        "expected_bright_disposition_self_sha256": (
            EXPECTED_BRIGHT_DISPOSITION_SELF_SHA256
        ),
        "fixed_freeze_relative_path": (
            "manifests/frames_p1_source_qualification_freeze_v1.json"
        ),
        "fixed_source_relative_path": (
            "artifacts/frames_p1_official_source_v1/test.tsv"
        ),
        "official_git_blob_sha1": EXPECTED_SOURCE_GIT_BLOB_SHA1,
        "official_source_size_bytes": EXPECTED_SOURCE_SIZE_BYTES,
        "conservative_exposed_row_id_interval_half_open": list(
            EXPOSED_FORMATION_INTERVAL
        ),
        "question_answer_url_or_row_id_values_output_count": 0,
        "model_action_or_score_count": 0,
        "retry_replay_resample_or_quota_revision": 0,
    }
    _write_exclusive(
        MARKER_PATH,
        {**marker_body, "self_sha256": _semantic_hash(marker_body)},
    )

    stage = "frozen_custody_and_implementation_binding"
    try:
        custody = _validate_custody()
        bright = _validate_bright_disposition()
        freeze = _validate_freeze()
        stage = "download_receipt_and_source_descriptor_binding"
        download = _validate_download_receipt()
        stage = "header_cell_grammar_and_aggregate_disjoint_capacity"

        family_counts = {family: 0 for family in FAMILIES}
        type_counts = {value: 0 for value in sorted(ALLOWED_REASONING_TYPES)}
        combination_counts: dict[str, int] = {}
        link_count_histogram: dict[str, int] = {}
        ineligible_counts = {
            "exposed_formation_interval": 0,
            "gold_link_count_outside_2_through_5": 0,
            "noncanonical_or_duplicate_normalized_gold_link": 0,
            "formation_prompt_collision": 0,
            "formation_answer_collision": 0,
            "formation_gold_page_collision": 0,
        }
        formation_prompts: set[str] = set()
        formation_answers: set[str] = set()
        formation_pages: set[str] = set()
        candidates: list[dict[str, Any]] = []
        observed_ids: set[int] = set()
        prompt_length_minimum: int | None = None
        prompt_length_maximum = 0
        answer_length_minimum: int | None = None
        answer_length_maximum = 0

        with _open_bound_source(download) as handle:
            reader = csv.DictReader(handle, delimiter="\t", strict=True)
            if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
                raise FramesP1SourceQualificationError("official TSV header drifted")
            for expected_row_id, row in enumerate(reader):
                if set(row) != set(EXPECTED_COLUMNS) or any(
                    value is None for value in row.values()
                ):
                    raise FramesP1SourceQualificationError("official TSV row drifted")
                raw_row_id = row["Unnamed: 0"]
                if raw_row_id != str(expected_row_id):
                    raise FramesP1SourceQualificationError(
                        "row id canonical grammar or order drifted"
                    )
                row_id = expected_row_id
                observed_ids.add(row_id)
                prompt = row["Prompt"]
                answer = row["Answer"]
                if not prompt.strip() or not answer.strip():
                    raise FramesP1SourceQualificationError(
                        "prompt or answer nonempty schema drifted"
                    )
                prompt_length_minimum = (
                    len(prompt)
                    if prompt_length_minimum is None
                    else min(prompt_length_minimum, len(prompt))
                )
                prompt_length_maximum = max(prompt_length_maximum, len(prompt))
                answer_length_minimum = (
                    len(answer)
                    if answer_length_minimum is None
                    else min(answer_length_minimum, len(answer))
                )
                answer_length_maximum = max(answer_length_maximum, len(answer))
                prompt_identity = _normalized_text_identity(prompt)
                answer_identity = _normalized_text_identity(answer)

                types = _reasoning_types(row["reasoning_types"])
                for value in types:
                    type_counts[value] += 1
                combination = " | ".join(types)
                combination_counts[combination] = (
                    combination_counts.get(combination, 0) + 1
                )
                links = _parsed_links(row["wiki_links"])
                if not 2 <= len(links) <= 15:
                    raise FramesP1SourceQualificationError(
                        "official gold-link cardinality drifted"
                    )
                _validate_redundant_link_columns(row, links)
                link_count_histogram[str(len(links))] = (
                    link_count_histogram.get(str(len(links)), 0) + 1
                )
                normalized_links = _normalized_canonical_links(links)

                if row_id < MEASUREMENT_MINIMUM_ID:
                    ineligible_counts["exposed_formation_interval"] += 1
                    if normalized_links is None:
                        raise FramesP1SourceQualificationError(
                            "formation page identity is noncanonical"
                        )
                    formation_prompts.add(prompt_identity)
                    formation_answers.add(answer_identity)
                    formation_pages.update(normalized_links)
                    continue
                if not 2 <= len(links) <= 5:
                    ineligible_counts[
                        "gold_link_count_outside_2_through_5"
                    ] += 1
                    continue
                if normalized_links is None:
                    ineligible_counts[
                        "noncanonical_or_duplicate_normalized_gold_link"
                    ] += 1
                    continue
                if prompt_identity in formation_prompts:
                    ineligible_counts["formation_prompt_collision"] += 1
                    continue
                if answer_identity in formation_answers:
                    ineligible_counts["formation_answer_collision"] += 1
                    continue
                if not formation_pages.isdisjoint(normalized_links):
                    ineligible_counts["formation_gold_page_collision"] += 1
                    continue
                family = _family(types)
                family_counts[family] += 1
                candidates.append(
                    {
                        "row_id": row_id,
                        "family": family,
                        "prompt_identity": prompt_identity,
                        "answer_identity": answer_identity,
                        "links": normalized_links,
                    }
                )

        if observed_ids != set(range(EXPECTED_ROW_COUNT)):
            raise FramesP1SourceQualificationError(
                "official source row-id closure drifted"
            )
        disjoint_counts, disjoint_collision_skips = _deterministic_disjoint_pool(
            candidates
        )
        capacity_pass = all(
            disjoint_counts[family] == REQUIRED_ELIGIBLE_PER_FAMILY
            for family in FAMILIES
        )
        body: dict[str, object] = {
            "schema": f"{VERSION}_result_v1",
            "status": (
                "qualified_aggregate_source_and_disjoint_capacity"
                if capacity_pass
                else "terminal_FRAMES_aggregate_disjoint_capacity_failed"
            ),
            "qualified": capacity_pass,
            "dataset_repository": EXPECTED_REPOSITORY,
            "dataset_revision": EXPECTED_REVISION,
            "dataset_file": EXPECTED_DATASET_FILE,
            "source_file_sha256": download["source_file_sha256"],
            "source_git_blob_sha1": download["source_git_blob_sha1"],
            "source_size_bytes": download["source_size_bytes"],
            "source_single_descriptor_pre_parse_and_post_parse_binding": True,
            "source_custody_self_sha256": custody["self_sha256"],
            "bright_disposition_self_sha256": bright["self_sha256"],
            "qualification_freeze_self_sha256": freeze["self_sha256"],
            "row_count": len(observed_ids),
            "row_id_sequence_exact_canonical_0_through_823": True,
            "TSV_header": list(EXPECTED_COLUMNS),
            "TSV_header_sha256": _semantic_hash(list(EXPECTED_COLUMNS)),
            "authoritative_wiki_links_cell_grammar": "Python_list_of_nonempty_strings",
            "redundant_link_column_validation": (
                "links_1_through_10_exact_11_exact_and_each_12_to_15_tail_link_once"
            ),
            "allowed_reasoning_types": sorted(ALLOWED_REASONING_TYPES),
            "reasoning_type_counts": type_counts,
            "reasoning_combination_counts": dict(
                sorted(combination_counts.items())
            ),
            "gold_link_count_histogram": dict(
                sorted(link_count_histogram.items(), key=lambda item: int(item[0]))
            ),
            "prompt_length_range": [prompt_length_minimum, prompt_length_maximum],
            "answer_length_range": [answer_length_minimum, answer_length_maximum],
            "conservative_exposed_row_id_interval_half_open": list(
                EXPOSED_FORMATION_INTERVAL
            ),
            "measurement_minimum_row_id": MEASUREMENT_MINIMUM_ID,
            "measurement_eligibility": {
                "gold_link_count_minimum": 2,
                "gold_link_count_maximum": 5,
                "normalized_link_scheme": "https",
                "normalized_link_host": "en.wikipedia.org",
                "normalized_link_path_prefix": "/wiki/",
                "fragments_removed_before_qrel_identity": True,
                "special_search_query_dot_segment_or_backslash_URL_allowed": False,
                "duplicate_normalized_gold_page_allowed": False,
                "exact_normalized_prompt_answer_or_gold_page_collision_with_formation_allowed": False,
            },
            "family_precedence": [
                "temporal_if_Temporal_reasoning_present",
                "structured_if_Numerical_or_Tabular_present",
                "constraint_postprocess_otherwise",
            ],
            "eligible_pre_disjoint_counts_by_family": family_counts,
            "ineligible_counts": ineligible_counts,
            "deterministic_page_prompt_answer_disjoint_pool_order_domain": (
                POOL_ORDER_DOMAIN
            ),
            "deterministic_disjoint_pool_counts_by_family": disjoint_counts,
            "deterministic_disjoint_pool_collision_skips_by_family": (
                disjoint_collision_skips
            ),
            "formal_block_names": list(BLOCK_NAMES),
            "family_quota_per_block": FAMILY_QUOTA_PER_BLOCK,
            "required_disjoint_pool_per_family": REQUIRED_ELIGIBLE_PER_FAMILY,
            "future_block_assignment_requires_new_private_HMAC_secret": True,
            "question_answer_url_or_row_id_values_output_count": 0,
            "answers_read_only_for_nonempty_length_and_collision_hash_validation": True,
            "model_action_or_score_count": 0,
            "external_network_calls": 0,
            "online_evaluator_or_API_calls": 0,
            "retry_replay_resample_or_quota_revision": 0,
        }
        result = {**body, "self_sha256": _semantic_hash(body)}
        _write_exclusive(RESULT_PATH, result)
        return result
    except BaseException as exc:
        _terminal_failure(stage, exc)
        raise FramesP1SourceQualificationError(
            "FRAMES source qualification failed terminally"
        ) from exc


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__)


def _main(argv: Sequence[str] | None = None) -> int:
    _parser().parse_args(argv)
    result = run_source_qualification()
    print(_canonical_bytes(result).decode("ascii"), end="")
    return 0 if result["qualified"] is True else 2


def main() -> int:
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FAILURE_PATH",
    "FREEZE_PATH",
    "MARKER_PATH",
    "RESULT_PATH",
    "SOURCE_PATH",
    "FramesP1SourceQualificationError",
    "VERSION",
    "main",
    "run_source_qualification",
]
