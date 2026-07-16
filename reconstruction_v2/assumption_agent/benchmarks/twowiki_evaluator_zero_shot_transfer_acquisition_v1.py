"""One-shot acquisition for the fresh 2Wiki zero-shot transfer study.

Preregistration reads only committed public protocol lineage, a newly generated
private selection key, and the already-consumed historical HippoRAG query
artifact.  It does not open the official 2Wiki archive.  Formal acquisition
first exercises every persistence location, then durably consumes a one-shot
authorization marker, and only afterwards opens ``train.json`` and
``dev.json`` from the hash-locked official archive.

Selection is independently stratified by exact source member and official
question type.  Every member of an internal or cross-split normalized-question
collision is excluded, as is every row matching any of four historical
deny-list identities.  The public receipt contains counts and commitments only;
questions, answers, contexts, labels, item IDs, and private paths remain in the
ignored private pack.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Sequence
import unicodedata
import zipfile

from ..models import stable_hash
from .hotpot_family_out_acquisition_v1 import committed_public_file_receipt
from .musique_official_core_comparison_v1 import (
    _assert_git_ignored_private_path,
    _canonical_bytes,
    _read_selection_secret,
    _selection_secret_commitment,
    _sha256_bytes,
    _sha256_file,
    normalize_answer_primary,
)
from . import musique_evaluator_portfolio_acquisition_v1 as musique_portfolio


VERSION = "twowiki_evaluator_zero_shot_transfer_acquisition_v1"
PREREGISTRATION_SCHEMA = f"{VERSION}_preregistration"
ACQUISITION_SCHEMA = f"{VERSION}_acquisition"
PRIVATE_ROW_SCHEMA = f"{VERSION}_private_row"
PRIVATE_LOCATOR_SCHEMA = f"{VERSION}_private_locator"
CONSUMPTION_SCHEMA = f"{VERSION}_consumption"

QUESTION_TYPES = (
    "bridge_comparison",
    "comparison",
    "compositional",
    "inference",
)
BLOCK_ORDER = ("A_hold", "M_search")
BLOCK_SOURCE_MEMBERS = {"A_hold": "train.json", "M_search": "dev.json"}
BLOCK_PER_TYPE_COUNTS = {"A_hold": 12, "M_search": 6}
BLOCK_COUNTS = {
    block: BLOCK_PER_TYPE_COUNTS[block] * len(QUESTION_TYPES)
    for block in BLOCK_ORDER
}
SELECTED_COUNT = sum(BLOCK_COUNTS.values())
SELECTION_DOMAIN_SEPARATOR = VERSION

OFFICIAL_ARCHIVE_SHA256 = (
    "95df2bf56fdabe034e27aebc580e02264232203cf52552f9efe8a919e5529eef"
)
SOURCE_MEMBER_SHA256S = {
    "train.json": "b318dbafbfed51a8029718fa59be8b616600cbff675a3b587694b28c5eedfc13",
    "dev.json": "79f77ae104088ea8e25b1a65dbece768d45771194663bc5660ec9a98070dadf5",
}
SOURCE_MEMBER_ROW_COUNTS = {"train.json": 167454, "dev.json": 12576}
COLLISION_ONLY_MEMBER_SHA256S = {
    "test.json": "48b196d4ba8557343abb9bd1ad03566bc02762ecd734617ff910027c33821b04",
}
COLLISION_ONLY_MEMBER_ROW_COUNTS = {"test.json": 12576}
ARCHIVE_MEMBER_SHA256S = {
    **SOURCE_MEMBER_SHA256S,
    **COLLISION_ONLY_MEMBER_SHA256S,
}
ARCHIVE_MEMBER_ROW_COUNTS = {
    **SOURCE_MEMBER_ROW_COUNTS,
    **COLLISION_ONLY_MEMBER_ROW_COUNTS,
}

HISTORICAL_QUERY_SHA256 = (
    "895cba294064df0c3302c76847b1fc08d99b5619f7663dfaa3b65cd780f1cac4"
)
HISTORICAL_ID_SET_SHA256 = (
    "c4954bfe1fd51d0113d86f6d26f935faec28be930b5b875886c888690fda6006"
)
HISTORICAL_QUERY_COUNT = 1000
HIPPORAG_COMMIT = "d437bfb1805278b81e20c82357ed3f7d90f14901"
HIPPORAG_QUERY_GIT_BLOB_SHA1 = "c87b01db53166b2b85b82d8773c6ed685bab2c16"
HISTORICAL_QUERY_REPO_RELATIVE = Path("reproduce/dataset/2wikimultihopqa.json")
HISTORICAL_QUERY_WORKSPACE_RELATIVE = (
    Path("reference/repos/HippoRAG") / HISTORICAL_QUERY_REPO_RELATIVE
)

DESIGN_RELATIVE = "manifests/twowiki_evaluator_zero_shot_transfer_design_v1.json"
DESIGN_SCHEMA = "twowiki_evaluator_zero_shot_transfer_design_v1"
DESIGN_FILE_SHA256 = (
    "1a5ab0d806324c721ff7ddc48ac7b22de94abadf12e2887b182a1af76db755ba"
)
DESIGN_SHA256 = (
    "903cf6dee77dedab34894330b1ae54b3893d6a2648392fb0cdd6f7569c354754"
)
SOURCE_QUALIFICATION_RELATIVE = "manifests/twowiki_fresh_source_qualification_v2.json"
SOURCE_QUALIFICATION_SCHEMA = "twowiki_fresh_source_qualification_v2"
SOURCE_QUALIFICATION_FILE_SHA256 = (
    "65c1d20a67288b4811a19d6cfca857ba034c04805499f316dfdabd2180b000e4"
)
SOURCE_QUALIFICATION_SHA256 = (
    "6b171908aa10884f1cf23ea5bcff26c85d38ddb1c6314f0251b69f51df13949d"
)
SOURCE_CUSTODY_RELATIVE = "manifests/twowiki_fresh_source_custody_v1.json"
SOURCE_CUSTODY_SCHEMA = "twowiki_fresh_source_custody_v1"
SOURCE_CUSTODY_FILE_SHA256 = (
    "1c9b11eea9eaef31eea31c315e898c3702d33fc13b400e7690397e8d8c30392c"
)
SOURCE_CUSTODY_SHA256 = (
    "b7f4f8c30ce543a05ec78a8894586f9c04ead707cfc3eb21e9bebe4f87906194"
)
SOURCE_ACCESS_ADDENDUM_RELATIVE = "manifests/twowiki_source_access_addendum_v3.json"
SOURCE_ACCESS_ADDENDUM_SCHEMA = "twowiki_source_access_addendum_v3"
SOURCE_ACCESS_ADDENDUM_FILE_SHA256 = (
    "509468a3b979daedfc5de70c8a6f08733d6b55874305c5f93af8d8100ac56379"
)
SOURCE_ACCESS_ADDENDUM_SHA256 = (
    "e10bc1980bf508a5cd9155bcfd1e81b685f544d7b83d17975876e9d387422958"
)

PREREGISTRATION_RELATIVE = (
    "manifests/twowiki_evaluator_zero_shot_transfer_acquisition_v1_"
    "preregistration.json"
)
ACQUISITION_RELATIVE = (
    "manifests/twowiki_evaluator_zero_shot_transfer_acquisition_v1_"
    "acquisition.json"
)
SELECTION_SECRET_RELATIVE = (
    "artifacts/twowiki_evaluator_transfer_custody_v1/selection.key"
)
SOURCE_ARCHIVE_RELATIVE = "artifacts/twowiki_official_source_v1/data_ids_april7.zip"
PRIVATE_PACK_ROOT_RELATIVE = (
    "artifacts/twowiki_evaluator_zero_shot_transfer_v1/private_pack"
)
PRIVATE_LOCATOR_RELATIVE = (
    "artifacts/twowiki_evaluator_zero_shot_transfer_v1/private_pack.locator.json"
)
CONSUMPTION_RELATIVE = (
    "artifacts/twowiki_evaluator_zero_shot_transfer_acquisition_v1/"
    "authorization.consumed.json"
)

# Production preregistration intentionally becomes possible only after the
# acquisition and execution implementations are clean committed HEAD blobs.
IMPLEMENTATION_RELATIVE_FILES = tuple(
    dict.fromkeys(
        (
            *musique_portfolio.IMPLEMENTATION_RELATIVE_FILES,
            "assumption_agent/benchmarks/"
            "twowiki_evaluator_zero_shot_transfer_acquisition_v1.py",
            "assumption_agent/benchmarks/"
            "twowiki_evaluator_zero_shot_transfer_v1.py",
        )
    )
)

PRIVATE_BLOCK_ROW_KEYS = frozenset(
    {
        "schema",
        "block",
        "source_member",
        "question_type",
        "item_id",
        "question",
        "corpus",
        "answers",
        "normalized_answers",
        "support_indices",
        "source_row_sha256",
        "normalized_question_sha256",
        "canonical_question_plus_ordered_context_sha256",
        "canonical_row_sha256",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_MIN_FREE_BYTES = 256 * 1024 * 1024


class TwoWikiAcquisitionError(RuntimeError):
    """Raised when source, selection, custody, or persistence drifts."""


@dataclass(frozen=True)
class BlockCommitment:
    block: str
    source_member: str
    question_type_counts: Mapping[str, int]
    count: int
    file_sha256: str
    item_commitment_set_sha256: str

    def to_dict(self) -> dict[str, Any]:
        body = asdict(self)
        body["question_type_counts"] = dict(self.question_type_counts)
        return body


@dataclass(frozen=True)
class _CandidateIdentity:
    source_member: str
    question_type: str
    item_id: str
    normalized_question_sha256: str
    canonical_question_plus_ordered_context_sha256: str
    canonical_row_sha256: str
    identity_commitment_sha256: str


@dataclass(frozen=True)
class _CollisionIdentity:
    source_member: str
    item_id: str | None
    normalized_question_sha256: str | None


@dataclass(frozen=True)
class _HistoricalDenylist:
    item_ids: frozenset[str]
    normalized_question_sha256s: frozenset[str]
    canonical_question_context_sha256s: frozenset[str]
    canonical_row_sha256s: frozenset[str]
    binding: Mapping[str, Any]


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TwoWikiAcquisitionError(f"{field} must be a lowercase SHA-256")
    return value


def _read_json_object(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise TwoWikiAcquisitionError(f"{field} is unavailable")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TwoWikiAcquisitionError(f"{field} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise TwoWikiAcquisitionError(f"{field} must be one object")
    return value, raw


def _assert_public_safe(payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    forbidden = (
        '"answers"',
        '"corpus"',
        '"item_id"',
        '"normalized_answers"',
        '"paragraph_text"',
        '"private_root"',
        '"question"',
        '"selection_secret_path"',
        '"support_indices"',
    )
    if any(token in serialized for token in forbidden):
        raise TwoWikiAcquisitionError(
            "public artifact contains private content or a private locator"
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_exclusive(path: Path, raw: bytes, *, mode: int) -> None:
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise TwoWikiAcquisitionError("output parent is unavailable")
    temporary = path.parent / f".{path.name}.{os.urandom(12).hex()}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        finally:
            temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_exclusive(
    path: Path,
    payload: Mapping[str, Any],
    *,
    hash_field: str,
    mode: int,
) -> None:
    body = dict(payload)
    body.pop(hash_field, None)
    body[hash_field] = stable_hash(body)
    raw = (json.dumps(body, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _atomic_write_exclusive(path, raw, mode=mode)


def _write_jsonl_exclusive(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> tuple[str, str]:
    raw = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    _atomic_write_exclusive(path, raw, mode=0o600)
    return _sha256_bytes(raw), stable_hash([stable_hash(row) for row in rows])


def _committed_binding(*, project: Path, path: Path, field: str) -> dict[str, Any]:
    try:
        receipt = committed_public_file_receipt(project=project, path=path)
    except Exception as exc:
        raise TwoWikiAcquisitionError(
            f"{field} must be the clean tracked HEAD blob"
        ) from exc
    file_sha256 = receipt["preregistration_file_sha256"]
    if file_sha256 != receipt["preregistration_head_blob_sha256"]:
        raise TwoWikiAcquisitionError(f"{field} HEAD binding drifted")
    return {
        "file_sha256": file_sha256,
        "head_blob_sha256": file_sha256,
        "clean_tracked_HEAD_blob": True,
    }


def _canonical_public_path(
    *, project: Path, supplied: Path, relative: str, field: str
) -> Path:
    root = project.resolve(strict=True)
    candidate = supplied if supplied.is_absolute() else root / supplied
    expected = root / relative
    try:
        actual = candidate.resolve(strict=True)
        canonical = expected.resolve(strict=True)
    except FileNotFoundError as exc:
        raise TwoWikiAcquisitionError(f"canonical {field} is unavailable") from exc
    if actual != canonical or candidate.is_symlink():
        raise TwoWikiAcquisitionError(f"{field} must use its fixed canonical path")
    return canonical


def _canonical_private_path(
    *,
    project: Path,
    supplied: Path,
    relative: str,
    require_file: bool | None,
    field: str,
) -> Path:
    expected = (project / relative).absolute()
    candidate = supplied if supplied.is_absolute() else project / supplied
    if candidate.absolute() != expected:
        raise TwoWikiAcquisitionError(f"{field} must use its fixed canonical path")
    try:
        return _assert_git_ignored_private_path(
            project=project, path=expected, require_file=require_file
        )
    except Exception as exc:
        raise TwoWikiAcquisitionError(f"{field} private custody drifted") from exc


def _load_self_hashed_public_binding(
    *,
    project: Path,
    relative: str,
    schema: str,
    file_sha256: str,
    semantic_field: str,
    semantic_sha256: str,
    field: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = project / relative
    payload, raw = _read_json_object(path, field)
    body = dict(payload)
    declared = _require_sha256(body.pop(semantic_field, None), semantic_field)
    custody = _committed_binding(project=project, path=path, field=field)
    if (
        payload.get("schema") != schema
        or _sha256_bytes(raw) != file_sha256
        or custody["file_sha256"] != file_sha256
        or declared != semantic_sha256
        or stable_hash(body) != declared
    ):
        raise TwoWikiAcquisitionError(f"{field} binding drifted")
    return payload, {
        "relative_path": relative,
        "schema": schema,
        "file_sha256": file_sha256,
        semantic_field: semantic_sha256,
        "committed_custody": custody,
    }


def public_protocol_bindings(project: Path) -> dict[str, Any]:
    root = project.resolve(strict=True)
    design, design_binding = _load_self_hashed_public_binding(
        project=root,
        relative=DESIGN_RELATIVE,
        schema=DESIGN_SCHEMA,
        file_sha256=DESIGN_FILE_SHA256,
        semantic_field="design_sha256",
        semantic_sha256=DESIGN_SHA256,
        field="zero-shot transfer design",
    )
    qualification, qualification_binding = _load_self_hashed_public_binding(
        project=root,
        relative=SOURCE_QUALIFICATION_RELATIVE,
        schema=SOURCE_QUALIFICATION_SCHEMA,
        file_sha256=SOURCE_QUALIFICATION_FILE_SHA256,
        semantic_field="qualification_sha256",
        semantic_sha256=SOURCE_QUALIFICATION_SHA256,
        field="source qualification",
    )
    custody, custody_binding = _load_self_hashed_public_binding(
        project=root,
        relative=SOURCE_CUSTODY_RELATIVE,
        schema=SOURCE_CUSTODY_SCHEMA,
        file_sha256=SOURCE_CUSTODY_FILE_SHA256,
        semantic_field="receipt_sha256",
        semantic_sha256=SOURCE_CUSTODY_SHA256,
        field="source custody",
    )
    addendum, addendum_binding = _load_self_hashed_public_binding(
        project=root,
        relative=SOURCE_ACCESS_ADDENDUM_RELATIVE,
        schema=SOURCE_ACCESS_ADDENDUM_SCHEMA,
        file_sha256=SOURCE_ACCESS_ADDENDUM_FILE_SHA256,
        semantic_field="addendum_sha256",
        semantic_sha256=SOURCE_ACCESS_ADDENDUM_SHA256,
        field="source access addendum",
    )
    if (
        design.get("status")
        != "fixed_zero_shot_transfer_before_private_selection_or_any_retrieval_score"
        or design.get("selection", {}).get("domain_separator")
        != SELECTION_DOMAIN_SEPARATOR
        or design.get("selection", {}).get("selected_count") != SELECTED_COUNT
        or design.get("source_binding", {}).get("archive_sha256")
        != OFFICIAL_ARCHIVE_SHA256
        or design.get("source_binding", {}).get("TRAIN_member_sha256")
        != SOURCE_MEMBER_SHA256S["train.json"]
        or design.get("source_binding", {}).get("DEV_member_sha256")
        != SOURCE_MEMBER_SHA256S["dev.json"]
        or design.get("source_binding", {}).get("historical_1000_ID_set_sha256")
        != HISTORICAL_ID_SET_SHA256
        or qualification.get("selection_status") != "not_performed"
        or qualification.get("archive", {}).get("file_sha256")
        != OFFICIAL_ARCHIVE_SHA256
        or qualification.get("historical_consumption", {}).get(
            "historical_query_file_sha256"
        )
        != HISTORICAL_QUERY_SHA256
        or custody.get("selection_secret_commitment_sha256")
        != design.get("selection", {}).get("selection_secret_commitment_sha256")
        or addendum.get("selection_performed") is not False
        or addendum.get("clarifications", {}).get(
            "unique_QA_rows_previously_programmatically_parsed"
        )
        != 192606
        or addendum.get("clarifications", {}).get(
            "alias_records_previously_programmatically_parsed"
        )
        != 203297
        or addendum.get("qualification_v2_binding", {}).get("file_sha256")
        != SOURCE_QUALIFICATION_FILE_SHA256
    ):
        raise TwoWikiAcquisitionError("public protocol dependency closure drifted")
    return {
        "design": design_binding,
        "source_qualification": qualification_binding,
        "source_custody": custody_binding,
        "source_access_addendum": addendum_binding,
    }


def implementation_binding(project: Path) -> dict[str, Any]:
    root = project.resolve(strict=True)
    rows: list[dict[str, Any]] = []
    for relative in IMPLEMENTATION_RELATIVE_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise TwoWikiAcquisitionError(
                f"implementation file missing or symlinked: {relative}"
            )
        custody = _committed_binding(
            project=root, path=path, field=f"implementation {relative}"
        )
        live = _sha256_file(path)
        if custody["file_sha256"] != live:
            raise TwoWikiAcquisitionError(f"implementation file drifted: {relative}")
        rows.append(
            {
                "path": relative,
                "sha256": live,
                "head_blob_sha256": live,
                "clean_tracked_HEAD_blob": True,
            }
        )
    return {"files": rows, "set_sha256": stable_hash(rows)}


def normalize_question(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"\w+", normalized, flags=re.UNICODE))


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TwoWikiAcquisitionError("value is not canonical-JSON serializable") from exc
    return serialized.encode("utf-8")


def _identity_hashes(
    raw: Mapping[str, Any], *, source_member: str
) -> tuple[str, str, str, str]:
    item_id = raw.get("_id")
    question = raw.get("question")
    context = raw.get("context")
    if (
        not isinstance(item_id, str)
        or not isinstance(question, str)
        or not isinstance(context, list)
        or source_member not in SOURCE_MEMBER_SHA256S
    ):
        raise TwoWikiAcquisitionError("row identity fields are malformed")
    normalized_question = normalize_question(question)
    normalized_question_sha256 = _sha256_bytes(normalized_question.encode("utf-8"))
    question_context_sha256 = _sha256_bytes(
        _canonical_json_bytes({"question": question, "context": context})
    )
    canonical_row_sha256 = _sha256_bytes(_canonical_json_bytes(raw))
    identity_body = {
        "member": source_member,
        "item_id_sha256": _sha256_bytes(item_id.encode("utf-8")),
        "normalized_question_sha256": normalized_question_sha256,
        "canonical_question_plus_ordered_context_sha256": question_context_sha256,
        "canonical_row_sha256": canonical_row_sha256,
    }
    return (
        normalized_question_sha256,
        question_context_sha256,
        canonical_row_sha256,
        _sha256_bytes(_canonical_json_bytes(identity_body)),
    )


def _collision_identity(
    raw: object, *, source_member: str
) -> _CollisionIdentity | None:
    """Return only the metadata needed to exclude cross-split collisions."""

    if not isinstance(raw, Mapping) or source_member not in ARCHIVE_MEMBER_SHA256S:
        return None
    raw_item_id = raw.get("_id")
    question = raw.get("question")
    item_id = (
        raw_item_id
        if isinstance(raw_item_id, str) and raw_item_id.strip()
        else None
    )
    normalized = (
        normalize_question(question)
        if isinstance(question, str) and question.strip()
        else ""
    )
    normalized_hash = (
        _sha256_bytes(normalized.encode("utf-8")) if normalized else None
    )
    if item_id is None and normalized_hash is None:
        return None
    return _CollisionIdentity(
        source_member=source_member,
        item_id=item_id,
        normalized_question_sha256=normalized_hash,
    )


def _normalize_source_row(
    raw: object, *, source_member: str
) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping) or source_member not in SOURCE_MEMBER_SHA256S:
        return None
    item_id = raw.get("_id")
    question_type = raw.get("type")
    question = raw.get("question")
    context = raw.get("context")
    supporting = raw.get("supporting_facts")
    answer = raw.get("answer")
    if (
        not isinstance(item_id, str)
        or not item_id.strip()
        or question_type not in QUESTION_TYPES
        or not isinstance(question, str)
        or not question.strip()
        or not normalize_question(question)
        or not isinstance(context, list)
        or len(context) != 10
        or not isinstance(supporting, list)
        or not supporting
        or not isinstance(answer, str)
        or not answer.strip()
    ):
        return None

    titles: list[str] = []
    sentence_rows: list[list[str]] = []
    for document in context:
        if (
            not isinstance(document, list)
            or len(document) != 2
            or not isinstance(document[0], str)
            or not document[0].strip()
            or not isinstance(document[1], list)
            or not document[1]
            or any(
                not isinstance(sentence, str) or not sentence.strip()
                for sentence in document[1]
            )
        ):
            return None
        titles.append(document[0])
        sentence_rows.append(document[1])
    if len(set(titles)) != len(titles):
        return None
    title_to_index = {title: index for index, title in enumerate(titles)}
    support_titles: set[str] = set()
    for fact in supporting:
        if (
            not isinstance(fact, list)
            or len(fact) != 2
            or not isinstance(fact[0], str)
            or fact[0] not in title_to_index
            or type(fact[1]) is not int
            or not 0 <= fact[1] < len(sentence_rows[title_to_index[fact[0]]])
        ):
            return None
        support_titles.add(fact[0])
    if not support_titles:
        return None
    support_indices = [
        index for index, title in enumerate(titles) if title in support_titles
    ]
    if len(support_indices) != len(support_titles):
        return None
    (
        normalized_question_hash,
        question_context_hash,
        canonical_row_hash,
        _identity_commitment,
    ) = (
        _identity_hashes(raw, source_member=source_member)
    )
    corpus = [
        {
            "paragraph_idx": index,
            "paragraph_title": title,
            "paragraph_text": " ".join(sentence_rows[index]),
        }
        for index, title in enumerate(titles)
    ]
    return {
        "schema": PRIVATE_ROW_SCHEMA,
        "block": "",
        "source_member": source_member,
        "question_type": question_type,
        "item_id": item_id,
        "question": question,
        "corpus": corpus,
        "answers": [answer],
        "normalized_answers": [normalize_answer_primary(answer)],
        "support_indices": support_indices,
        "source_row_sha256": stable_hash(raw),
        "normalized_question_sha256": normalized_question_hash,
        "canonical_question_plus_ordered_context_sha256": question_context_hash,
        "canonical_row_sha256": canonical_row_hash,
    }


def _git(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise TwoWikiAcquisitionError(f"git command failed: {arguments[0]}")
    return completed.stdout


def _historical_query_path(project: Path, supplied: Path) -> tuple[Path, Path]:
    workspace = project.parent.resolve(strict=True)
    repository = workspace / "reference/repos/HippoRAG"
    expected = repository / HISTORICAL_QUERY_REPO_RELATIVE
    candidate = supplied if supplied.is_absolute() else project / supplied
    try:
        actual = candidate.resolve(strict=True)
        canonical = expected.resolve(strict=True)
    except FileNotFoundError as exc:
        raise TwoWikiAcquisitionError("historical query artifact is unavailable") from exc
    if actual != canonical or candidate.is_symlink():
        raise TwoWikiAcquisitionError(
            "historical queries must use the fixed HippoRAG checkout path"
        )
    if _git(repository, "rev-parse", "HEAD").decode().strip() != HIPPORAG_COMMIT:
        raise TwoWikiAcquisitionError("HippoRAG historical checkout commit drifted")
    if _git(repository, "status", "--porcelain", "--", str(HISTORICAL_QUERY_REPO_RELATIVE)):
        raise TwoWikiAcquisitionError("historical query artifact is dirty")
    index = _git(
        repository, "ls-files", "-s", "--", str(HISTORICAL_QUERY_REPO_RELATIVE)
    ).decode().strip().split()
    if len(index) < 2 or index[1] != HIPPORAG_QUERY_GIT_BLOB_SHA1:
        raise TwoWikiAcquisitionError("historical query HEAD blob drifted")
    return canonical, repository


def load_historical_denylist(
    *, project: Path, path: Path
) -> _HistoricalDenylist:
    canonical, _repository = _historical_query_path(project, path)
    raw = canonical.read_bytes()
    if _sha256_bytes(raw) != HISTORICAL_QUERY_SHA256:
        raise TwoWikiAcquisitionError("historical query bytes drifted")
    try:
        rows = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TwoWikiAcquisitionError("historical queries are invalid JSON") from exc
    if not isinstance(rows, list) or len(rows) != HISTORICAL_QUERY_COUNT:
        raise TwoWikiAcquisitionError("historical query count drifted")
    ids: set[str] = set()
    normalized_questions: set[str] = set()
    question_contexts: set[str] = set()
    canonical_rows: set[str] = set()
    for raw_row in rows:
        if (
            not isinstance(raw_row, Mapping)
            or not isinstance(raw_row.get("_id"), str)
            or not raw_row["_id"].strip()
            or not isinstance(raw_row.get("question"), str)
            or not isinstance(raw_row.get("context"), list)
        ):
            raise TwoWikiAcquisitionError("historical row identity drifted")
        normalized_question, question_context, canonical_row, _identity = (
            _identity_hashes(raw_row, source_member="dev.json")
        )
        ids.add(raw_row["_id"])
        normalized_questions.add(normalized_question)
        question_contexts.add(question_context)
        canonical_rows.add(canonical_row)
    if len(ids) != HISTORICAL_QUERY_COUNT or stable_hash(sorted(ids)) != HISTORICAL_ID_SET_SHA256:
        raise TwoWikiAcquisitionError("historical ID deny-list drifted")
    set_commitments = {
        "item_id_set_sha256": stable_hash(sorted(ids)),
        "normalized_question_sha256_set_sha256": stable_hash(
            sorted(normalized_questions)
        ),
        "canonical_question_plus_ordered_context_sha256_set_sha256": stable_hash(
            sorted(question_contexts)
        ),
        "canonical_row_sha256_set_sha256": stable_hash(sorted(canonical_rows)),
    }
    binding = {
        "workspace_relative_path": HISTORICAL_QUERY_WORKSPACE_RELATIVE.as_posix(),
        "file_sha256": HISTORICAL_QUERY_SHA256,
        "hipporag_commit": HIPPORAG_COMMIT,
        "git_blob_sha1": HIPPORAG_QUERY_GIT_BLOB_SHA1,
        "clean_tracked_HEAD_blob": True,
        "row_count": HISTORICAL_QUERY_COUNT,
        "set_counts": {
            "item_ids": len(ids),
            "normalized_questions": len(normalized_questions),
            "canonical_question_plus_ordered_contexts": len(question_contexts),
            "canonical_rows": len(canonical_rows),
        },
        "set_commitments": set_commitments,
        "item_level_content_persisted_publicly": False,
    }
    return _HistoricalDenylist(
        item_ids=frozenset(ids),
        normalized_question_sha256s=frozenset(normalized_questions),
        canonical_question_context_sha256s=frozenset(question_contexts),
        canonical_row_sha256s=frozenset(canonical_rows),
        binding=binding,
    )


def _canonical_selection_secret(project: Path, supplied: Path) -> tuple[Path, bytes]:
    path = _canonical_private_path(
        project=project,
        supplied=supplied,
        relative=SELECTION_SECRET_RELATIVE,
        require_file=True,
        field="selection secret",
    )
    secret = _read_selection_secret(project=project, path=path)
    if _selection_secret_commitment(secret) != (
        "fc1589f1c5453a2c115f89b315e11e0c9182e65e741afc53fc552ca4d5733d26"
    ):
        raise TwoWikiAcquisitionError("selection secret commitment drifted")
    return path, secret


def _selection_key(
    *,
    source_member: str,
    question_type: str,
    identity_commitment_sha256: str,
    secret: bytes,
) -> str:
    if source_member not in SOURCE_MEMBER_SHA256S or question_type not in QUESTION_TYPES:
        raise TwoWikiAcquisitionError("selection stratum is invalid")
    commitment = _require_sha256(identity_commitment_sha256, "identity commitment")
    message = (
        f"{SELECTION_DOMAIN_SEPARATOR}\0{source_member}\0{question_type}\0{commitment}"
    ).encode("utf-8")
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def selection_runtime_binding() -> dict[str, Any]:
    return {
        "python_implementation": sys.implementation.name,
        "python_version_info": list(sys.version_info[:5]),
        "unicode_database_version": unicodedata.unidata_version,
        "json_canonicalization": (
            "ensure_ascii_true_sort_keys_true_separators_comma_colon_allow_nan_false"
        ),
    }


def build_preregistration(
    *, project: Path, selection_secret_path: Path, historical_queries_path: Path
) -> dict[str, Any]:
    """Build the complete zero-official-row acquisition preregistration."""

    root = project.resolve(strict=True)
    _secret_path, secret = _canonical_selection_secret(root, selection_secret_path)
    historical = load_historical_denylist(
        project=root, path=historical_queries_path
    )
    payload: dict[str, Any] = {
        "schema": PREREGISTRATION_SCHEMA,
        "decision": "fresh_two_block_acquisition_only_no_measurement_authority",
        "public_protocol_bindings": public_protocol_bindings(root),
        "implementation": implementation_binding(root),
        "selection_runtime": selection_runtime_binding(),
        "source": {
            "archive_sha256": OFFICIAL_ARCHIVE_SHA256,
            "members": {
                member: {
                    "file_sha256": ARCHIVE_MEMBER_SHA256S[member],
                    "row_count": ARCHIVE_MEMBER_ROW_COUNTS[member],
                    "selection_role": (
                        "candidate" if member in SOURCE_MEMBER_SHA256S
                        else "collision_metadata_only_not_candidate"
                    ),
                }
                for member in ARCHIVE_MEMBER_SHA256S
            },
            "official_archive_rows_opened": 0,
            "qualification_rows_previously_parsed_outcome_blind": 192606,
            "alias_records_previously_parsed_outcome_blind": 203297,
        },
        "historical_denylist": dict(historical.binding),
        "eligibility": {
            "allowed_question_types": list(QUESTION_TYPES),
            "context_count": 10,
            "context_titles_unique": True,
            "nonempty_question_answer_context_and_support": True,
            "support_sentence_indices_in_range": True,
            "support_title_maps_to_exactly_one_context": True,
            "variable_support_document_count_supported": True,
            "question_normalization": (
                "Unicode_NFKC_then_casefold_then_Unicode_word_tokens_joined_by_one_space"
            ),
            "internal_or_cross_split_normalized_question_collision_policy": (
                "exclude_every_member_of_any_collision_class"
            ),
            "internal_or_cross_split_item_id_collision_policy": (
                "exclude_every_member_of_any_collision_class"
            ),
            "collision_scan_members": list(ARCHIVE_MEMBER_SHA256S),
            "collision_only_members_never_eligible_for_selection": list(
                COLLISION_ONLY_MEMBER_SHA256S
            ),
            "historical_exclusion_fields": [
                "_id",
                "normalized_question_sha256",
                "canonical_question_plus_ordered_context_sha256",
                "canonical_row_sha256",
            ],
        },
        "selection": {
            "method": "private_HMAC_rank_within_exact_source_member_and_question_type",
            "domain_separator": SELECTION_DOMAIN_SEPARATOR,
            "selection_secret_commitment_sha256": _selection_secret_commitment(secret),
            "selection_secret_persisted_publicly": False,
            "block_order": list(BLOCK_ORDER),
            "block_source_members": dict(BLOCK_SOURCE_MEMBERS),
            "question_type_order": list(QUESTION_TYPES),
            "per_type_counts": dict(BLOCK_PER_TYPE_COUNTS),
            "block_counts": dict(BLOCK_COUNTS),
            "selected_count": SELECTED_COUNT,
            "replacement": False,
            "manual_or_outcome_conditioned_selection": False,
        },
        "access_contract": {
            "both_blocks_formed_together": True,
            "persistence_preflight_precedes_consumption": True,
            "one_shot_marker_precedes_source_archive_open": True,
            "A_hold_requires_separate_committed_pre_run_freeze": True,
            "M_search_open_only_after_A_hold_promotion": True,
            "retry_replay_resample": 0,
        },
        "safety": {
            "official_archive_rows_read": 0,
            "historical_consumed_rows_read": HISTORICAL_QUERY_COUNT,
            "model_calls": 0,
            "network_calls": 0,
            "scores_computed": 0,
            "online_evaluator_calls": 0,
        },
    }
    _assert_public_safe(payload)
    payload["preregistration_sha256"] = stable_hash(payload)
    return payload


def verify_preregistration(
    *,
    path: Path,
    project: Path,
    selection_secret_path: Path,
    historical_queries_path: Path,
) -> dict[str, Any]:
    canonical = _canonical_public_path(
        project=project,
        supplied=path,
        relative=PREREGISTRATION_RELATIVE,
        field="preregistration",
    )
    payload, _raw = _read_json_object(canonical, "preregistration")
    body = dict(payload)
    declared = _require_sha256(
        body.pop("preregistration_sha256", None), "preregistration hash"
    )
    if payload.get("schema") != PREREGISTRATION_SCHEMA or stable_hash(body) != declared:
        raise TwoWikiAcquisitionError("preregistration self-hash drifted")
    expected = build_preregistration(
        project=project,
        selection_secret_path=selection_secret_path,
        historical_queries_path=historical_queries_path,
    )
    if payload != expected:
        raise TwoWikiAcquisitionError(
            "preregistration differs from the complete live protocol"
        )
    return payload


def _persistence_canary(directory: Path) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise TwoWikiAcquisitionError("persistence directory is unsafe")
    target = directory / f".{VERSION}.{os.urandom(12).hex()}.canary"
    expected = b"twowiki-acquisition-persistence-canary\n"
    try:
        _atomic_write_exclusive(target, expected, mode=0o600)
        if target.read_bytes() != expected or stat.S_IMODE(target.stat().st_mode) & 0o077:
            raise TwoWikiAcquisitionError("persistence canary verification failed")
    finally:
        target.unlink(missing_ok=True)
        _fsync_directory(directory)


def _preflight_persistence(
    *, pack_root: Path, locator: Path, marker: Path, public_receipt: Path
) -> None:
    if marker.exists():
        raise FileExistsError("2Wiki acquisition authorization was already consumed")
    if pack_root.exists() or locator.exists() or public_receipt.exists():
        raise FileExistsError("2Wiki acquisition output already exists")
    paths = (pack_root, locator, marker, public_receipt)
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise TwoWikiAcquisitionError("private and public outputs must be disjoint")
    directories = {pack_root.parent, locator.parent, marker.parent, public_receipt.parent}
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if directory.is_symlink() or not directory.is_dir():
            raise TwoWikiAcquisitionError("output parent is unsafe")
    created = False
    try:
        os.mkdir(pack_root, 0o700)
        created = True
        os.chmod(pack_root, 0o700)
        if stat.S_IMODE(pack_root.stat().st_mode) != 0o700:
            raise TwoWikiAcquisitionError("private pack permissions are unsafe")
        _fsync_directory(pack_root.parent)
        for directory in {*directories, pack_root}:
            _persistence_canary(directory)
            if shutil.disk_usage(directory).free < _MIN_FREE_BYTES:
                raise TwoWikiAcquisitionError(
                    "insufficient free space for one-shot acquisition"
                )
    except BaseException:
        if created:
            try:
                pack_root.rmdir()
                _fsync_directory(pack_root.parent)
            except OSError:
                pass
        raise


def _iter_json_array_stream(handle: BinaryIO, *, chunk_size: int = 1 << 20) -> Iterator[Any]:
    """Incrementally decode one UTF-8 JSON array without loading a member whole."""

    decoder = json.JSONDecoder()
    utf8 = __import__("codecs").getincrementaldecoder("utf-8")()
    buffer = ""
    position = 0
    eof = False
    started = False
    expect_value = True
    just_saw_comma = False
    while True:
        if position > (1 << 20):
            buffer = buffer[position:]
            position = 0
        while position < len(buffer) and buffer[position].isspace():
            position += 1
        if not started:
            if position < len(buffer):
                if buffer[position] != "[":
                    raise TwoWikiAcquisitionError("source member must be a JSON array")
                position += 1
                started = True
                continue
        elif position < len(buffer):
            character = buffer[position]
            if character == "]":
                if just_saw_comma:
                    raise TwoWikiAcquisitionError("source array has a trailing comma")
                position += 1
                tail_text = buffer[position:]
                while not eof:
                    tail = handle.read(chunk_size)
                    if tail:
                        try:
                            tail_text += utf8.decode(tail, final=False)
                        except UnicodeDecodeError as exc:
                            raise TwoWikiAcquisitionError(
                                "source member is not UTF-8"
                            ) from exc
                    else:
                        try:
                            tail_text += utf8.decode(b"", final=True)
                        except UnicodeDecodeError as exc:
                            raise TwoWikiAcquisitionError(
                                "source member is not UTF-8"
                            ) from exc
                        eof = True
                if tail_text.strip():
                    raise TwoWikiAcquisitionError("source member has trailing content")
                return
            if not expect_value:
                if character != ",":
                    raise TwoWikiAcquisitionError("source array separator is malformed")
                position += 1
                expect_value = True
                just_saw_comma = True
                continue
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise TwoWikiAcquisitionError("source member JSON is truncated")
            else:
                position = end
                expect_value = False
                just_saw_comma = False
                yield value
                continue
        if eof:
            raise TwoWikiAcquisitionError("source member JSON is truncated")
        chunk = handle.read(chunk_size)
        if chunk:
            try:
                buffer += utf8.decode(chunk, final=False)
            except UnicodeDecodeError as exc:
                raise TwoWikiAcquisitionError("source member is not UTF-8") from exc
        else:
            try:
                buffer += utf8.decode(b"", final=True)
            except UnicodeDecodeError as exc:
                raise TwoWikiAcquisitionError("source member is not UTF-8") from exc
            eof = True


def _exact_zip_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    result: dict[str, zipfile.ZipInfo] = {}
    for expected in ARCHIVE_MEMBER_SHA256S:
        matches = [
            info
            for info in archive.infolist()
            if not info.is_dir() and PurePosixPath(info.filename).name == expected
        ]
        if len(matches) != 1:
            raise TwoWikiAcquisitionError(f"exact {expected} archive member is ambiguous")
        result[expected] = matches[0]
    if len({info.filename for info in result.values()}) != len(result):
        raise TwoWikiAcquisitionError("source archive member identities overlap")
    return result


def _hash_zip_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    digest = hashlib.sha256()
    with archive.open(info, "r") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _scan_source_metadata(
    archive: zipfile.ZipFile, members: Mapping[str, zipfile.ZipInfo]
) -> tuple[
    list[_CandidateIdentity],
    list[_CollisionIdentity],
    dict[str, int],
    dict[str, int],
    dict[str, int],
]:
    identities: list[_CandidateIdentity] = []
    collision_identities: list[_CollisionIdentity] = []
    row_counts: dict[str, int] = {}
    integrity_rejected: dict[str, int] = {}
    collision_metadata_rejected: dict[str, int] = {}
    for source_member in ARCHIVE_MEMBER_SHA256S:
        if _hash_zip_member(archive, members[source_member]) != ARCHIVE_MEMBER_SHA256S[source_member]:
            raise TwoWikiAcquisitionError(f"{source_member} hash drifted")
        count = 0
        rejected = 0
        collision_rejected = 0
        with archive.open(members[source_member], "r") as handle:
            for raw_row in _iter_json_array_stream(handle):
                count += 1
                collision = _collision_identity(
                    raw_row, source_member=source_member
                )
                if collision is None:
                    collision_rejected += 1
                else:
                    collision_identities.append(collision)
                if source_member not in SOURCE_MEMBER_SHA256S:
                    continue
                normalized = _normalize_source_row(raw_row, source_member=source_member)
                if normalized is None:
                    rejected += 1
                    continue
                identities.append(
                    _CandidateIdentity(
                        source_member=source_member,
                        question_type=normalized["question_type"],
                        item_id=normalized["item_id"],
                        normalized_question_sha256=normalized[
                            "normalized_question_sha256"
                        ],
                        canonical_question_plus_ordered_context_sha256=normalized[
                            "canonical_question_plus_ordered_context_sha256"
                        ],
                        canonical_row_sha256=normalized["canonical_row_sha256"],
                        identity_commitment_sha256=_identity_hashes(
                            raw_row, source_member=source_member
                        )[3],
                    )
                )
        if count != ARCHIVE_MEMBER_ROW_COUNTS[source_member]:
            raise TwoWikiAcquisitionError(f"{source_member} row count drifted")
        row_counts[source_member] = count
        collision_metadata_rejected[source_member] = collision_rejected
        if source_member in SOURCE_MEMBER_SHA256S:
            integrity_rejected[source_member] = rejected
    return (
        identities,
        collision_identities,
        row_counts,
        integrity_rejected,
        collision_metadata_rejected,
    )


def _select_identities(
    identities: Sequence[_CandidateIdentity],
    *,
    collision_identities: Sequence[_CollisionIdentity],
    historical: _HistoricalDenylist,
    secret: bytes,
) -> tuple[dict[str, tuple[_CandidateIdentity, ...]], dict[str, Any]]:
    normalized_question_counts = Counter(
        row.normalized_question_sha256
        for row in collision_identities
        if row.normalized_question_sha256 is not None
    )
    item_id_counts = Counter(
        row.item_id for row in collision_identities if row.item_id is not None
    )
    eligible_by_stratum: dict[tuple[str, str], list[_CandidateIdentity]] = {
        (member, question_type): []
        for member in SOURCE_MEMBER_SHA256S
        for question_type in QUESTION_TYPES
    }
    exclusions = {
        member: {"historical": 0, "question_collision": 0, "duplicate_item_id": 0}
        for member in SOURCE_MEMBER_SHA256S
    }
    for row in identities:
        historical_match = (
            row.item_id in historical.item_ids
            or row.normalized_question_sha256
            in historical.normalized_question_sha256s
            or row.canonical_question_plus_ordered_context_sha256
            in historical.canonical_question_context_sha256s
            or row.canonical_row_sha256 in historical.canonical_row_sha256s
        )
        if historical_match:
            exclusions[row.source_member]["historical"] += 1
            continue
        if normalized_question_counts[row.normalized_question_sha256] != 1:
            exclusions[row.source_member]["question_collision"] += 1
            continue
        if item_id_counts[row.item_id] != 1:
            exclusions[row.source_member]["duplicate_item_id"] += 1
            continue
        eligible_by_stratum[(row.source_member, row.question_type)].append(row)

    selected: dict[str, tuple[_CandidateIdentity, ...]] = {}
    eligible_counts: dict[str, dict[str, int]] = {
        member: {} for member in SOURCE_MEMBER_SHA256S
    }
    for block in BLOCK_ORDER:
        member = BLOCK_SOURCE_MEMBERS[block]
        per_type = BLOCK_PER_TYPE_COUNTS[block]
        block_rows: list[_CandidateIdentity] = []
        for question_type in QUESTION_TYPES:
            rows = eligible_by_stratum[(member, question_type)]
            eligible_counts[member][question_type] = len(rows)
            rows.sort(
                key=lambda row: (
                    _selection_key(
                        source_member=member,
                        question_type=question_type,
                        identity_commitment_sha256=row.identity_commitment_sha256,
                        secret=secret,
                    ),
                    row.identity_commitment_sha256,
                )
            )
            if len(rows) < per_type:
                raise TwoWikiAcquisitionError(
                    f"insufficient eligible {member}/{question_type} rows"
                )
            block_rows.extend(rows[:per_type])
        selected[block] = tuple(block_rows)
    selected_keys = [row.item_id for block in BLOCK_ORDER for row in selected[block]]
    if len(selected_keys) != SELECTED_COUNT or len(set(selected_keys)) != SELECTED_COUNT:
        raise TwoWikiAcquisitionError("selected source identities overlap")
    return selected, {
        "eligible_counts_by_member_and_type": eligible_counts,
        "exclusion_counts_by_member": exclusions,
        "normalized_question_collision_class_count": sum(
            count > 1 for count in normalized_question_counts.values()
        ),
        "item_id_collision_class_count": sum(
            count > 1 for count in item_id_counts.values()
        ),
        "collision_scan_member_counts": dict(
            (
                member,
                sum(row.source_member == member for row in collision_identities),
            )
            for member in ARCHIVE_MEMBER_SHA256S
        ),
    }


def _materialize_selected_rows(
    archive: zipfile.ZipFile,
    members: Mapping[str, zipfile.ZipInfo],
    selected: Mapping[str, Sequence[_CandidateIdentity]],
) -> dict[str, tuple[dict[str, Any], ...]]:
    lookup = {
        (row.source_member, row.item_id): (block, row)
        for block in BLOCK_ORDER
        for row in selected[block]
    }
    materialized: dict[tuple[str, str], dict[str, Any]] = {}
    for source_member in SOURCE_MEMBER_SHA256S:
        with archive.open(members[source_member], "r") as handle:
            for raw_row in _iter_json_array_stream(handle):
                if not isinstance(raw_row, Mapping):
                    continue
                item_id = raw_row.get("_id")
                key = (source_member, item_id) if isinstance(item_id, str) else None
                if key not in lookup:
                    continue
                block, identity = lookup[key]
                normalized = _normalize_source_row(raw_row, source_member=source_member)
                if normalized is None:
                    raise TwoWikiAcquisitionError("selected row integrity changed")
                if (
                    normalized["question_type"] != identity.question_type
                    or normalized["normalized_question_sha256"]
                    != identity.normalized_question_sha256
                    or normalized["canonical_question_plus_ordered_context_sha256"]
                    != identity.canonical_question_plus_ordered_context_sha256
                    or normalized["canonical_row_sha256"] != identity.canonical_row_sha256
                    or _identity_hashes(raw_row, source_member=source_member)[3]
                    != identity.identity_commitment_sha256
                ):
                    raise TwoWikiAcquisitionError("selected row identity changed")
                normalized["block"] = block
                if set(normalized) != PRIVATE_BLOCK_ROW_KEYS:
                    raise TwoWikiAcquisitionError("private row schema drifted")
                if key in materialized:
                    raise TwoWikiAcquisitionError("selected row materialized twice")
                materialized[key] = normalized
    result: dict[str, tuple[dict[str, Any], ...]] = {}
    for block in BLOCK_ORDER:
        rows = tuple(
            materialized[(identity.source_member, identity.item_id)]
            for identity in selected[block]
            if (identity.source_member, identity.item_id) in materialized
        )
        if len(rows) != BLOCK_COUNTS[block]:
            raise TwoWikiAcquisitionError("selected row materialization is incomplete")
        result[block] = rows
    return result


def load_private_block(
    path: str | Path,
    *,
    commitment: BlockCommitment,
    expected_block: str | None = None,
) -> tuple[dict[str, Any], ...]:
    block = commitment.block if expected_block is None else expected_block
    if (
        block != commitment.block
        or block not in BLOCK_ORDER
        or commitment.source_member != BLOCK_SOURCE_MEMBERS[block]
        or commitment.count != BLOCK_COUNTS[block]
        or dict(commitment.question_type_counts)
        != {question_type: BLOCK_PER_TYPE_COUNTS[block] for question_type in QUESTION_TYPES}
    ):
        raise TwoWikiAcquisitionError("private block identity drifted")
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise TwoWikiAcquisitionError("private block is unavailable")
    raw = candidate.read_bytes()
    if _sha256_bytes(raw) != commitment.file_sha256 or not raw.endswith(b"\n"):
        raise TwoWikiAcquisitionError("private block file hash drifted")
    rows: list[dict[str, Any]] = []
    try:
        for line in raw.splitlines():
            row = json.loads(line.decode("utf-8"))
            if not isinstance(row, dict):
                raise TwoWikiAcquisitionError("private row is malformed")
            rows.append(row)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TwoWikiAcquisitionError("private block JSONL is invalid") from exc
    type_counts = Counter(row.get("question_type") for row in rows)
    if (
        len(rows) != commitment.count
        or any(set(row) != PRIVATE_BLOCK_ROW_KEYS for row in rows)
        or any(row.get("schema") != PRIVATE_ROW_SCHEMA for row in rows)
        or any(row.get("block") != block for row in rows)
        or any(row.get("source_member") != commitment.source_member for row in rows)
        or type_counts != Counter(commitment.question_type_counts)
        or b"".join(_canonical_bytes(row) + b"\n" for row in rows) != raw
        or stable_hash([stable_hash(row) for row in rows])
        != commitment.item_commitment_set_sha256
    ):
        raise TwoWikiAcquisitionError("private block schema or commitment drifted")
    ids = [row["item_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise TwoWikiAcquisitionError("private block IDs are not unique")
    return tuple(rows)


def acquire_private_blocks(
    *,
    project: Path,
    preregistration_path: Path,
    selection_secret_path: Path,
    historical_queries_path: Path,
    source_archive_path: Path,
    private_root: Path,
    private_locator_path: Path,
    public_receipt_path: Path,
) -> dict[str, Any]:
    """Consume authorization and create both blocks in one indivisible study."""

    root = project.resolve(strict=True)
    preregistration = verify_preregistration(
        path=preregistration_path,
        project=root,
        selection_secret_path=selection_secret_path,
        historical_queries_path=historical_queries_path,
    )
    canonical_prereg = root / PREREGISTRATION_RELATIVE
    prereg_custody = _committed_binding(
        project=root, path=canonical_prereg, field="preregistration"
    )
    _secret_path, secret = _canonical_selection_secret(root, selection_secret_path)
    historical = load_historical_denylist(project=root, path=historical_queries_path)
    source = _canonical_private_path(
        project=root,
        supplied=source_archive_path,
        relative=SOURCE_ARCHIVE_RELATIVE,
        require_file=True,
        field="source archive",
    )
    pack_root = _canonical_private_path(
        project=root,
        supplied=private_root,
        relative=PRIVATE_PACK_ROOT_RELATIVE,
        require_file=False,
        field="private pack root",
    )
    locator = _canonical_private_path(
        project=root,
        supplied=private_locator_path,
        relative=PRIVATE_LOCATOR_RELATIVE,
        require_file=None,
        field="private locator",
    )
    marker = _canonical_private_path(
        project=root,
        supplied=root / CONSUMPTION_RELATIVE,
        relative=CONSUMPTION_RELATIVE,
        require_file=None,
        field="consumption marker",
    )
    public_receipt = (root / ACQUISITION_RELATIVE).absolute()
    supplied_public = (
        public_receipt_path
        if public_receipt_path.is_absolute()
        else root / public_receipt_path
    ).absolute()
    if supplied_public != public_receipt:
        raise TwoWikiAcquisitionError("public receipt must use its canonical path")
    _preflight_persistence(
        pack_root=pack_root,
        locator=locator,
        marker=marker,
        public_receipt=public_receipt,
    )

    protocol = preregistration["public_protocol_bindings"]
    marker_body = {
        "schema": CONSUMPTION_SCHEMA,
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "preregistration_file_sha256": prereg_custody["file_sha256"],
        "design_file_sha256": protocol["design"]["file_sha256"],
        "source_qualification_file_sha256": protocol["source_qualification"][
            "file_sha256"
        ],
        "source_custody_file_sha256": protocol["source_custody"]["file_sha256"],
        "source_access_addendum_file_sha256": protocol[
            "source_access_addendum"
        ]["file_sha256"],
        "historical_query_file_sha256": HISTORICAL_QUERY_SHA256,
        "historical_denylist_set_commitments": historical.binding[
            "set_commitments"
        ],
        "source_archive_sha256": OFFICIAL_ARCHIVE_SHA256,
        "source_member_sha256s": dict(ARCHIVE_MEMBER_SHA256S),
        "selection_secret_commitment_sha256": _selection_secret_commitment(secret),
        "private_pack_path_hash": stable_hash(
            {"absolute_private_pack": str(pack_root)}
        ),
        "private_locator_path_hash": stable_hash(
            {"absolute_private_locator": str(locator)}
        ),
        "public_receipt_path_hash": stable_hash(
            {"absolute_public_receipt": str(public_receipt)}
        ),
        "persistence_preflight_complete": True,
        "source_archive_opened_before_consumption": False,
        "retry_replay_resample_authorized": False,
    }
    try:
        _write_json_exclusive(
            marker, marker_body, hash_field="consumption_sha256", mode=0o600
        )
    except BaseException:
        if not marker.exists():
            try:
                pack_root.rmdir()
                _fsync_directory(pack_root.parent)
            except OSError:
                pass
        raise
    marker_raw = marker.read_bytes()

    # Formal source bytes are first opened only after the durable marker above.
    if _sha256_file(source) != OFFICIAL_ARCHIVE_SHA256:
        raise TwoWikiAcquisitionError("official archive hash drifted")
    with zipfile.ZipFile(source) as archive:
        members = _exact_zip_members(archive)
        (
            identities,
            collision_identities,
            source_rows,
            integrity_rejected,
            collision_metadata_rejected,
        ) = _scan_source_metadata(archive, members)
        selected, selection_stats = _select_identities(
            identities,
            collision_identities=collision_identities,
            historical=historical,
            secret=secret,
        )
        private_rows = _materialize_selected_rows(archive, members, selected)

    block_commitments: list[BlockCommitment] = []
    for block in BLOCK_ORDER:
        file_hash, item_set_hash = _write_jsonl_exclusive(
            pack_root / f"{block}.jsonl", private_rows[block]
        )
        block_commitments.append(
            BlockCommitment(
                block=block,
                source_member=BLOCK_SOURCE_MEMBERS[block],
                question_type_counts={
                    question_type: BLOCK_PER_TYPE_COUNTS[block]
                    for question_type in QUESTION_TYPES
                },
                count=BLOCK_COUNTS[block],
                file_sha256=file_hash,
                item_commitment_set_sha256=item_set_hash,
            )
        )
    locator_body = {
        "schema": PRIVATE_LOCATOR_SCHEMA,
        "private_root": str(pack_root),
        "blocks": [
            {**row.to_dict(), "relative_file": f"{row.block}.jsonl"}
            for row in block_commitments
        ],
        "private_pack_sha256": stable_hash(
            [row.to_dict() for row in block_commitments]
        ),
        "selection_secret_included": False,
    }
    _write_json_exclusive(
        locator, locator_body, hash_field="locator_sha256", mode=0o600
    )
    receipt: dict[str, Any] = {
        "schema": ACQUISITION_SCHEMA,
        "decision": "fresh_two_block_private_pack_formed_no_measurement_authority",
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "preregistration_custody": prereg_custody,
        "public_protocol_bindings": preregistration["public_protocol_bindings"],
        "implementation": preregistration["implementation"],
        "selection_runtime": preregistration["selection_runtime"],
        "historical_denylist": preregistration["historical_denylist"],
        "source": {
            "archive_sha256": OFFICIAL_ARCHIVE_SHA256,
            "member_sha256s": dict(ARCHIVE_MEMBER_SHA256S),
            "source_row_counts": source_rows,
            "data_integrity_rejected_counts": integrity_rejected,
            "collision_metadata_rejected_counts": collision_metadata_rejected,
            "collision_only_members_never_selected": list(
                COLLISION_ONLY_MEMBER_SHA256S
            ),
        },
        "selection": {
            "method": "private_HMAC_rank_within_exact_source_member_and_question_type",
            "domain_separator": SELECTION_DOMAIN_SEPARATOR,
            "selection_secret_commitment_sha256": _selection_secret_commitment(secret),
            "question_type_order": list(QUESTION_TYPES),
            "block_source_members": dict(BLOCK_SOURCE_MEMBERS),
            "block_counts": dict(BLOCK_COUNTS),
            "selected_count": SELECTED_COUNT,
            **selection_stats,
        },
        "commitments": {
            "block_files": [row.to_dict() for row in block_commitments],
            "private_pack_sha256": stable_hash(
                [row.to_dict() for row in block_commitments]
            ),
            "private_locator_file_sha256": _sha256_file(locator),
            "private_row_key_set_sha256": stable_hash(
                sorted(PRIVATE_BLOCK_ROW_KEYS)
            ),
            "item_ids_persisted_publicly": False,
            "private_paths_persisted_publicly": False,
        },
        "prospective_ordering": {
            "preregistration_committed_before_consumption": True,
            "persistence_preflight_complete_before_consumption": True,
            "pack_root_created_before_consumption": True,
            "consumption_persisted_before_source_archive_open": True,
            "source_rows_opened_before_consumption": 0,
            "acquisition_consumption_file_sha256": _sha256_bytes(marker_raw),
            "acquisition_consumption_sha256": json.loads(marker_raw)[
                "consumption_sha256"
            ],
            "retry_replay_resample_authorized": False,
        },
        "safety": {
            "formation_executed": False,
            "measurement_executed": False,
            "model_calls": 0,
            "network_calls": 0,
            "online_evaluator_calls": 0,
            "scores_computed": 0,
        },
    }
    _assert_public_safe(receipt)
    return receipt


def _parse_block_commitments(receipt: Mapping[str, Any]) -> tuple[BlockCommitment, ...]:
    rows = receipt.get("commitments", {}).get("block_files")
    if not isinstance(rows, list) or len(rows) != len(BLOCK_ORDER):
        raise TwoWikiAcquisitionError("block commitments are malformed")
    result: list[BlockCommitment] = []
    for block, row in zip(BLOCK_ORDER, rows):
        expected_type_counts = {
            question_type: BLOCK_PER_TYPE_COUNTS[block]
            for question_type in QUESTION_TYPES
        }
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "block",
                "source_member",
                "question_type_counts",
                "count",
                "file_sha256",
                "item_commitment_set_sha256",
            }
            or row.get("block") != block
            or row.get("source_member") != BLOCK_SOURCE_MEMBERS[block]
            or row.get("question_type_counts") != expected_type_counts
            or row.get("count") != BLOCK_COUNTS[block]
        ):
            raise TwoWikiAcquisitionError("block commitment drifted")
        result.append(
            BlockCommitment(
                block=block,
                source_member=BLOCK_SOURCE_MEMBERS[block],
                question_type_counts=expected_type_counts,
                count=BLOCK_COUNTS[block],
                file_sha256=_require_sha256(row.get("file_sha256"), "block file"),
                item_commitment_set_sha256=_require_sha256(
                    row.get("item_commitment_set_sha256"), "block item set"
                ),
            )
        )
    return tuple(result)


def _valid_committed_custody(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value)
        == {"clean_tracked_HEAD_blob", "file_sha256", "head_blob_sha256"}
        and value.get("clean_tracked_HEAD_blob") is True
        and _SHA256_RE.fullmatch(str(value.get("file_sha256"))) is not None
        and value.get("head_blob_sha256") == value.get("file_sha256")
    )


def _valid_implementation(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"files", "set_sha256"}:
        return False
    rows = value.get("files")
    return (
        isinstance(rows, list)
        and value.get("set_sha256") == stable_hash(rows)
        and [row.get("path") for row in rows if isinstance(row, Mapping)]
        == list(IMPLEMENTATION_RELATIVE_FILES)
        and all(
            isinstance(row, Mapping)
            and set(row)
            == {
                "clean_tracked_HEAD_blob",
                "head_blob_sha256",
                "path",
                "sha256",
            }
            and row.get("clean_tracked_HEAD_blob") is True
            and row.get("head_blob_sha256") == row.get("sha256")
            and _SHA256_RE.fullmatch(str(row.get("sha256"))) is not None
            for row in rows
        )
    )


def _valid_public_protocol_bindings(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "design",
        "source_qualification",
        "source_custody",
        "source_access_addendum",
    }:
        return False
    configurations = {
        "design": (DESIGN_RELATIVE, DESIGN_SCHEMA, DESIGN_FILE_SHA256, "design_sha256", DESIGN_SHA256),
        "source_qualification": (
            SOURCE_QUALIFICATION_RELATIVE,
            SOURCE_QUALIFICATION_SCHEMA,
            SOURCE_QUALIFICATION_FILE_SHA256,
            "qualification_sha256",
            SOURCE_QUALIFICATION_SHA256,
        ),
        "source_custody": (
            SOURCE_CUSTODY_RELATIVE,
            SOURCE_CUSTODY_SCHEMA,
            SOURCE_CUSTODY_FILE_SHA256,
            "receipt_sha256",
            SOURCE_CUSTODY_SHA256,
        ),
        "source_access_addendum": (
            SOURCE_ACCESS_ADDENDUM_RELATIVE,
            SOURCE_ACCESS_ADDENDUM_SCHEMA,
            SOURCE_ACCESS_ADDENDUM_FILE_SHA256,
            "addendum_sha256",
            SOURCE_ACCESS_ADDENDUM_SHA256,
        ),
    }
    for role, (relative, schema, file_hash, semantic_field, semantic_hash) in configurations.items():
        row = value.get(role)
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "relative_path",
                "schema",
                "file_sha256",
                semantic_field,
                "committed_custody",
            }
            or row.get("relative_path") != relative
            or row.get("schema") != schema
            or row.get("file_sha256") != file_hash
            or row.get(semantic_field) != semantic_hash
            or not _valid_committed_custody(row.get("committed_custody"))
            or row["committed_custody"]["file_sha256"] != file_hash
        ):
            return False
    return True


def _valid_historical_binding(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "workspace_relative_path",
        "file_sha256",
        "hipporag_commit",
        "git_blob_sha1",
        "clean_tracked_HEAD_blob",
        "row_count",
        "set_counts",
        "set_commitments",
        "item_level_content_persisted_publicly",
    }:
        return False
    counts = value.get("set_counts")
    commitments = value.get("set_commitments")
    count_keys = {
        "item_ids",
        "normalized_questions",
        "canonical_question_plus_ordered_contexts",
        "canonical_rows",
    }
    commitment_keys = {
        "item_id_set_sha256",
        "normalized_question_sha256_set_sha256",
        "canonical_question_plus_ordered_context_sha256_set_sha256",
        "canonical_row_sha256_set_sha256",
    }
    return (
        value.get("workspace_relative_path")
        == HISTORICAL_QUERY_WORKSPACE_RELATIVE.as_posix()
        and value.get("file_sha256") == HISTORICAL_QUERY_SHA256
        and value.get("hipporag_commit") == HIPPORAG_COMMIT
        and value.get("git_blob_sha1") == HIPPORAG_QUERY_GIT_BLOB_SHA1
        and value.get("clean_tracked_HEAD_blob") is True
        and value.get("row_count") == HISTORICAL_QUERY_COUNT
        and value.get("item_level_content_persisted_publicly") is False
        and isinstance(counts, Mapping)
        and set(counts) == count_keys
        and counts.get("item_ids") == HISTORICAL_QUERY_COUNT
        and all(type(counts.get(key)) is int and 0 < counts[key] <= HISTORICAL_QUERY_COUNT for key in count_keys)
        and isinstance(commitments, Mapping)
        and set(commitments) == commitment_keys
        and commitments.get("item_id_set_sha256") == HISTORICAL_ID_SET_SHA256
        and all(_SHA256_RE.fullmatch(str(commitments.get(key))) is not None for key in commitment_keys)
    )


def load_acquisition_binding(
    path: str | Path,
) -> tuple[dict[str, Any], tuple[BlockCommitment, ...]]:
    receipt, _raw = _read_json_object(Path(path), "acquisition receipt")
    body = dict(receipt)
    declared = _require_sha256(body.pop("acquisition_sha256", None), "acquisition hash")
    if receipt.get("schema") != ACQUISITION_SCHEMA or stable_hash(body) != declared:
        raise TwoWikiAcquisitionError("acquisition receipt self-hash drifted")
    expected_top = {
        "schema",
        "decision",
        "preregistration_sha256",
        "preregistration_custody",
        "public_protocol_bindings",
        "implementation",
        "selection_runtime",
        "historical_denylist",
        "source",
        "selection",
        "commitments",
        "prospective_ordering",
        "safety",
        "acquisition_sha256",
    }
    selection = receipt.get("selection")
    source = receipt.get("source")
    safety = receipt.get("safety")
    commitments = receipt.get("commitments")
    prospective = receipt.get("prospective_ordering")
    eligible_counts = (
        selection.get("eligible_counts_by_member_and_type")
        if isinstance(selection, Mapping)
        else None
    )
    exclusion_counts = (
        selection.get("exclusion_counts_by_member")
        if isinstance(selection, Mapping)
        else None
    )
    if (
        set(receipt) != expected_top
        or receipt.get("decision")
        != "fresh_two_block_private_pack_formed_no_measurement_authority"
        or not isinstance(selection, Mapping)
        or selection.get("method")
        != "private_HMAC_rank_within_exact_source_member_and_question_type"
        or selection.get("domain_separator") != SELECTION_DOMAIN_SEPARATOR
        or set(selection)
        != {
            "method",
            "domain_separator",
            "selection_secret_commitment_sha256",
            "question_type_order",
            "block_source_members",
            "block_counts",
            "selected_count",
            "eligible_counts_by_member_and_type",
            "exclusion_counts_by_member",
            "normalized_question_collision_class_count",
            "item_id_collision_class_count",
            "collision_scan_member_counts",
        }
        or selection.get("selection_secret_commitment_sha256")
        != "fc1589f1c5453a2c115f89b315e11e0c9182e65e741afc53fc552ca4d5733d26"
        or selection.get("question_type_order") != list(QUESTION_TYPES)
        or selection.get("block_source_members") != BLOCK_SOURCE_MEMBERS
        or selection.get("block_counts") != BLOCK_COUNTS
        or selection.get("selected_count") != SELECTED_COUNT
        or not isinstance(eligible_counts, Mapping)
        or set(eligible_counts) != set(SOURCE_MEMBER_SHA256S)
        or any(
            not isinstance(eligible_counts.get(member), Mapping)
            or set(eligible_counts[member]) != set(QUESTION_TYPES)
            or any(
                type(eligible_counts[member].get(question_type)) is not int
                or eligible_counts[member][question_type]
                < BLOCK_PER_TYPE_COUNTS[
                    "A_hold" if member == "train.json" else "M_search"
                ]
                for question_type in QUESTION_TYPES
            )
            for member in SOURCE_MEMBER_SHA256S
        )
        or not isinstance(exclusion_counts, Mapping)
        or set(exclusion_counts) != set(SOURCE_MEMBER_SHA256S)
        or any(
            not isinstance(exclusion_counts.get(member), Mapping)
            or set(exclusion_counts[member])
            != {"historical", "question_collision", "duplicate_item_id"}
            or any(
                type(exclusion_counts[member].get(reason)) is not int
                or exclusion_counts[member][reason] < 0
                for reason in ("historical", "question_collision", "duplicate_item_id")
            )
            for member in SOURCE_MEMBER_SHA256S
        )
        or type(selection.get("normalized_question_collision_class_count")) is not int
        or selection["normalized_question_collision_class_count"] < 0
        or type(selection.get("item_id_collision_class_count")) is not int
        or selection["item_id_collision_class_count"] < 0
        or not isinstance(selection.get("collision_scan_member_counts"), Mapping)
        or set(selection["collision_scan_member_counts"])
        != set(ARCHIVE_MEMBER_SHA256S)
        or any(
            type(selection["collision_scan_member_counts"].get(member)) is not int
            or not 0
            <= selection["collision_scan_member_counts"][member]
            <= ARCHIVE_MEMBER_ROW_COUNTS[member]
            for member in ARCHIVE_MEMBER_SHA256S
        )
        or not isinstance(source, Mapping)
        or set(source)
        != {
            "archive_sha256",
            "member_sha256s",
            "source_row_counts",
            "data_integrity_rejected_counts",
            "collision_metadata_rejected_counts",
            "collision_only_members_never_selected",
        }
        or source.get("archive_sha256") != OFFICIAL_ARCHIVE_SHA256
        or source.get("member_sha256s") != ARCHIVE_MEMBER_SHA256S
        or source.get("source_row_counts") != ARCHIVE_MEMBER_ROW_COUNTS
        or not isinstance(source.get("data_integrity_rejected_counts"), Mapping)
        or set(source["data_integrity_rejected_counts"])
        != set(SOURCE_MEMBER_SHA256S)
        or any(
            type(source["data_integrity_rejected_counts"].get(member)) is not int
            or source["data_integrity_rejected_counts"][member] < 0
            for member in SOURCE_MEMBER_SHA256S
        )
        or any(
            sum(eligible_counts[member].values())
            + sum(exclusion_counts[member].values())
            + source["data_integrity_rejected_counts"][member]
            != SOURCE_MEMBER_ROW_COUNTS[member]
            for member in SOURCE_MEMBER_SHA256S
        )
        or not isinstance(source.get("collision_metadata_rejected_counts"), Mapping)
        or set(source["collision_metadata_rejected_counts"])
        != set(ARCHIVE_MEMBER_SHA256S)
        or any(
            type(source["collision_metadata_rejected_counts"].get(member)) is not int
            or source["collision_metadata_rejected_counts"][member] < 0
            for member in ARCHIVE_MEMBER_SHA256S
        )
        or any(
            selection["collision_scan_member_counts"][member]
            + source["collision_metadata_rejected_counts"][member]
            != ARCHIVE_MEMBER_ROW_COUNTS[member]
            for member in ARCHIVE_MEMBER_SHA256S
        )
        or source.get("collision_only_members_never_selected")
        != list(COLLISION_ONLY_MEMBER_SHA256S)
        or not _valid_committed_custody(receipt.get("preregistration_custody"))
        or not _valid_public_protocol_bindings(receipt.get("public_protocol_bindings"))
        or not _valid_implementation(receipt.get("implementation"))
        or receipt.get("selection_runtime") != selection_runtime_binding()
        or not _valid_historical_binding(receipt.get("historical_denylist"))
        or not isinstance(commitments, Mapping)
        or set(commitments)
        != {
            "block_files",
            "private_pack_sha256",
            "private_locator_file_sha256",
            "private_row_key_set_sha256",
            "item_ids_persisted_publicly",
            "private_paths_persisted_publicly",
        }
        or _SHA256_RE.fullmatch(str(commitments.get("private_locator_file_sha256")))
        is None
        or commitments.get("private_row_key_set_sha256")
        != stable_hash(sorted(PRIVATE_BLOCK_ROW_KEYS))
        or commitments.get("item_ids_persisted_publicly") is not False
        or commitments.get("private_paths_persisted_publicly") is not False
        or not isinstance(prospective, Mapping)
        or set(prospective)
        != {
            "preregistration_committed_before_consumption",
            "persistence_preflight_complete_before_consumption",
            "pack_root_created_before_consumption",
            "consumption_persisted_before_source_archive_open",
            "source_rows_opened_before_consumption",
            "acquisition_consumption_file_sha256",
            "acquisition_consumption_sha256",
            "retry_replay_resample_authorized",
        }
        or any(
            prospective.get(field) is not True
            for field in (
                "preregistration_committed_before_consumption",
                "persistence_preflight_complete_before_consumption",
                "pack_root_created_before_consumption",
                "consumption_persisted_before_source_archive_open",
            )
        )
        or prospective.get("source_rows_opened_before_consumption") != 0
        or _SHA256_RE.fullmatch(
            str(prospective.get("acquisition_consumption_file_sha256"))
        )
        is None
        or _SHA256_RE.fullmatch(
            str(prospective.get("acquisition_consumption_sha256"))
        )
        is None
        or prospective.get("retry_replay_resample_authorized") is not False
        or not isinstance(safety, Mapping)
        or set(safety)
        != {
            "formation_executed",
            "measurement_executed",
            "model_calls",
            "network_calls",
            "online_evaluator_calls",
            "scores_computed",
        }
        or any(safety.get(field) != 0 for field in ("model_calls", "network_calls", "online_evaluator_calls", "scores_computed"))
        or safety.get("formation_executed") is not False
        or safety.get("measurement_executed") is not False
    ):
        raise TwoWikiAcquisitionError("acquisition receipt contract drifted")
    blocks = _parse_block_commitments(receipt)
    if receipt.get("commitments", {}).get("private_pack_sha256") != stable_hash(
        [row.to_dict() for row in blocks]
    ):
        raise TwoWikiAcquisitionError("private pack commitment drifted")
    _assert_public_safe(receipt)
    return receipt, blocks


def _load_consumption_marker(project: Path) -> tuple[dict[str, Any], bytes]:
    path = project / CONSUMPTION_RELATIVE
    payload, raw = _read_json_object(path, "consumption marker")
    body = dict(payload)
    declared = _require_sha256(body.pop("consumption_sha256", None), "consumption hash")
    if (
        payload.get("schema") != CONSUMPTION_SCHEMA
        or set(payload)
        != {
            "schema",
            "preregistration_sha256",
            "preregistration_file_sha256",
            "design_file_sha256",
            "source_qualification_file_sha256",
            "source_custody_file_sha256",
            "source_access_addendum_file_sha256",
            "historical_query_file_sha256",
            "historical_denylist_set_commitments",
            "source_archive_sha256",
            "source_member_sha256s",
            "selection_secret_commitment_sha256",
            "private_pack_path_hash",
            "private_locator_path_hash",
            "public_receipt_path_hash",
            "persistence_preflight_complete",
            "source_archive_opened_before_consumption",
            "retry_replay_resample_authorized",
            "consumption_sha256",
        }
        or stable_hash(body) != declared
        or stat.S_IMODE(path.stat().st_mode) & 0o077
    ):
        raise TwoWikiAcquisitionError("consumption marker drifted")
    return payload, raw


def load_acquisition_binding_live(
    *, project: Path, path: str | Path, selection_secret_path: Path
) -> tuple[dict[str, Any], tuple[BlockCommitment, ...]]:
    """Validate the canonical committed receipt/preregistration/marker chain."""

    root = project.resolve(strict=True)
    canonical = _canonical_public_path(
        project=root,
        supplied=Path(path),
        relative=ACQUISITION_RELATIVE,
        field="acquisition receipt",
    )
    receipt, blocks = load_acquisition_binding(canonical)
    receipt_raw = canonical.read_bytes()
    receipt_custody = _committed_binding(
        project=root, path=canonical, field="acquisition receipt"
    )
    prereg_path = root / PREREGISTRATION_RELATIVE
    prereg, prereg_raw = _read_json_object(prereg_path, "preregistration")
    prereg_custody = _committed_binding(
        project=root, path=prereg_path, field="preregistration"
    )
    prereg_body = dict(prereg)
    prereg_hash = _require_sha256(
        prereg_body.pop("preregistration_sha256", None), "preregistration hash"
    )
    if stable_hash(prereg_body) != prereg_hash:
        raise TwoWikiAcquisitionError("committed preregistration self-hash drifted")
    _secret_path, secret = _canonical_selection_secret(root, selection_secret_path)
    verified_prereg = verify_preregistration(
        path=prereg_path,
        project=root,
        selection_secret_path=selection_secret_path,
        historical_queries_path=root.parent / HISTORICAL_QUERY_WORKSPACE_RELATIVE,
    )
    if verified_prereg != prereg:
        raise TwoWikiAcquisitionError("live preregistration contract drifted")
    marker, marker_raw = _load_consumption_marker(root)
    live_protocol = public_protocol_bindings(root)
    live_implementation = implementation_binding(root)
    live_historical = load_historical_denylist(
        project=root,
        path=root.parent / HISTORICAL_QUERY_WORKSPACE_RELATIVE,
    )
    selection = receipt["selection"]
    prospective = receipt["prospective_ordering"]
    if (
        receipt_custody["file_sha256"] != _sha256_bytes(receipt_raw)
        or receipt["preregistration_sha256"] != prereg_hash
        or receipt["preregistration_custody"] != prereg_custody
        or prereg_custody["file_sha256"] != _sha256_bytes(prereg_raw)
        or receipt["public_protocol_bindings"]
        != prereg["public_protocol_bindings"]
        or receipt["public_protocol_bindings"] != live_protocol
        or receipt["implementation"] != prereg["implementation"]
        or receipt["implementation"] != live_implementation
        or receipt["selection_runtime"] != prereg["selection_runtime"]
        or receipt["selection_runtime"] != selection_runtime_binding()
        or receipt["historical_denylist"] != prereg["historical_denylist"]
        or receipt["historical_denylist"] != live_historical.binding
        or selection["selection_secret_commitment_sha256"]
        != prereg["selection"]["selection_secret_commitment_sha256"]
        or selection["selection_secret_commitment_sha256"]
        != _selection_secret_commitment(secret)
        or marker["preregistration_sha256"] != prereg_hash
        or marker["preregistration_file_sha256"] != _sha256_bytes(prereg_raw)
        or marker["selection_secret_commitment_sha256"]
        != _selection_secret_commitment(secret)
        or marker["historical_query_file_sha256"] != HISTORICAL_QUERY_SHA256
        or marker["historical_denylist_set_commitments"]
        != live_historical.binding["set_commitments"]
        or marker["source_archive_sha256"] != OFFICIAL_ARCHIVE_SHA256
        or marker["source_member_sha256s"] != ARCHIVE_MEMBER_SHA256S
        or marker["design_file_sha256"]
        != prereg["public_protocol_bindings"]["design"]["file_sha256"]
        or marker["source_qualification_file_sha256"]
        != prereg["public_protocol_bindings"]["source_qualification"][
            "file_sha256"
        ]
        or marker["source_custody_file_sha256"]
        != prereg["public_protocol_bindings"]["source_custody"]["file_sha256"]
        or marker["source_access_addendum_file_sha256"]
        != prereg["public_protocol_bindings"]["source_access_addendum"][
            "file_sha256"
        ]
        or marker["public_receipt_path_hash"]
        != stable_hash({"absolute_public_receipt": str(canonical)})
        or marker["private_pack_path_hash"]
        != stable_hash(
            {"absolute_private_pack": str(root / PRIVATE_PACK_ROOT_RELATIVE)}
        )
        or marker["private_locator_path_hash"]
        != stable_hash(
            {"absolute_private_locator": str(root / PRIVATE_LOCATOR_RELATIVE)}
        )
        or marker["persistence_preflight_complete"] is not True
        or marker["source_archive_opened_before_consumption"] is not False
        or marker["retry_replay_resample_authorized"] is not False
        or prospective["acquisition_consumption_file_sha256"]
        != _sha256_bytes(marker_raw)
        or prospective["acquisition_consumption_sha256"]
        != marker["consumption_sha256"]
    ):
        raise TwoWikiAcquisitionError(
            "canonical acquisition, preregistration, marker, or secret drifted"
        )
    return receipt, blocks


__all__ = [
    "ACQUISITION_SCHEMA",
    "BLOCK_COUNTS",
    "BLOCK_ORDER",
    "BLOCK_PER_TYPE_COUNTS",
    "BLOCK_SOURCE_MEMBERS",
    "BlockCommitment",
    "PRIVATE_BLOCK_ROW_KEYS",
    "PRIVATE_LOCATOR_SCHEMA",
    "PREREGISTRATION_SCHEMA",
    "QUESTION_TYPES",
    "SELECTED_COUNT",
    "TwoWikiAcquisitionError",
    "acquire_private_blocks",
    "build_preregistration",
    "implementation_binding",
    "load_acquisition_binding",
    "load_acquisition_binding_live",
    "load_historical_denylist",
    "load_private_block",
    "normalize_question",
    "public_protocol_bindings",
    "verify_preregistration",
]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preregister = commands.add_parser("preregister")
    acquire = commands.add_parser("acquire")
    for command in (preregister, acquire):
        command.add_argument("--project", type=Path, required=True)
        command.add_argument("--selection-secret", type=Path, required=True)
        command.add_argument("--historical-queries", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    acquire.add_argument("--preregistration", type=Path, required=True)
    acquire.add_argument("--source-archive", type=Path, required=True)
    acquire.add_argument("--private-root", type=Path, required=True)
    acquire.add_argument("--private-locator", type=Path, required=True)
    arguments = parser.parse_args(argv)

    root = arguments.project.resolve(strict=True)
    expected_output = root / (
        PREREGISTRATION_RELATIVE
        if arguments.command == "preregister"
        else ACQUISITION_RELATIVE
    )
    if arguments.output.resolve(strict=False) != expected_output.resolve(strict=False):
        raise TwoWikiAcquisitionError("production CLI output must be canonical")
    if arguments.output.exists():
        raise FileExistsError("public acquisition output already exists")
    common = {
        "project": root,
        "selection_secret_path": arguments.selection_secret,
        "historical_queries_path": arguments.historical_queries,
    }
    if arguments.command == "preregister":
        payload = build_preregistration(**common)
        _write_json_exclusive(
            arguments.output,
            payload,
            hash_field="preregistration_sha256",
            mode=0o644,
        )
        return 0
    if arguments.preregistration.resolve(strict=True) != (
        root / PREREGISTRATION_RELATIVE
    ).resolve(strict=True):
        raise TwoWikiAcquisitionError("production CLI preregistration must be canonical")
    payload = acquire_private_blocks(
        **common,
        preregistration_path=arguments.preregistration,
        source_archive_path=arguments.source_archive,
        private_root=arguments.private_root,
        private_locator_path=arguments.private_locator,
        public_receipt_path=arguments.output,
    )
    _write_json_exclusive(
        arguments.output, payload, hash_field="acquisition_sha256", mode=0o644
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
