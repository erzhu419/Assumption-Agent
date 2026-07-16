"""One-shot, label-blind acquisition for the frozen QASC L5 study.

The preregistration path opens no QASC source bytes.  Formal acquisition may
verify and extract the unlabeled fact corpus before authorization, but it
durably consumes a one-shot marker before opening the labeled TRAIN or DEV
members.  TEST is never reopened.  All four blocks are selected together and
stored as separate gold-free views and private label envelopes.

This module deliberately does not import or modify any earlier frozen study.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tarfile
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Sequence
import unicodedata

from ..models import stable_hash


VERSION = "qasc_evaluator_direct_action_coevolution_acquisition_v1"
PREREGISTRATION_SCHEMA = f"{VERSION}_preregistration"
ACQUISITION_SCHEMA = f"{VERSION}_acquisition"
CONSUMPTION_SCHEMA = f"{VERSION}_consumption"
PRIVATE_VIEW_SCHEMA = "qasc_evaluator_direct_action_acquisition_v1_private_view"
PRIVATE_LABEL_SCHEMA = (
    "qasc_evaluator_direct_action_acquisition_v1_private_label_envelope"
)
PRIVATE_LOCATOR_SCHEMA = f"{VERSION}_private_locator"

BLOCK_ORDER = ("A_form", "F_search", "A_hold", "M_search")
BLOCK_COUNTS = {block: 64 for block in BLOCK_ORDER}
BLOCK_SOURCE_MEMBERS = {
    "A_form": "TRAIN",
    "F_search": "TRAIN",
    "A_hold": "TRAIN",
    "M_search": "DEV",
}
SELECTED_COUNT = sum(BLOCK_COUNTS.values())
_CLEAN_MODULE_CLI_ACTIVE = False
SOURCE_MEMBER_PATHS = {
    "TRAIN": "QASC_Dataset/train.jsonl",
    "DEV": "QASC_Dataset/dev.jsonl",
}
SOURCE_MEMBER_SHA256S = {
    "TRAIN": "6d8094cd4291ac06b1d26b3a7e28639035d83449a317ef1dc51e6a80ad11ec73",
    "DEV": "b3c60e18b1ff0aa67b868f9a06921dc7e0c11f211e6601eecf9baee1c1a0df55",
}
SOURCE_MEMBER_ROW_COUNTS = {"TRAIN": 8134, "DEV": 926}
FORMAL_STEM_CANDIDATE_COUNTS = {"TRAIN": 7175, "DEV": 865}

DATASET_ARCHIVE_SHA256 = (
    "a7b3f2244f768974c609fd621346c931a72715609f171cb5544fc1da2a2ad55c"
)
CORPUS_ARCHIVE_SHA256 = (
    "370df36f9449241f7d06f6d461866723f40a7ee23e5f200e926dc8e7e1522cc2"
)
CORPUS_MEMBER_PATH = "QASC_Corpus/QASC_Corpus.txt"
CORPUS_MEMBER_SHA256 = (
    "8853c3906e4912cebe09e889b222db42ab5477192cb73243afee53b21b7e2905"
)
CORPUS_MEMBER_SIZE = 1_156_126_443
CORPUS_LINE_COUNT = 16_987_130

SELECTION_DOMAIN_SEPARATOR = VERSION
SELECTION_SECRET_COMMITMENT_SHA256 = (
    "4cd68e425cfdf5e8577c2df18099f1236ff205f19e2a63f08c6692c511bae329"
)

DESIGN_RELATIVE = "manifests/qasc_evaluator_direct_action_coevolution_design_v1.json"
DESIGN_SCHEMA = "qasc_evaluator_direct_action_coevolution_design_v1"
DESIGN_FILE_SHA256 = (
    "fdd1bd1d088cee851a20015227d1f3dea1d086bcaf5c0f435f1bf52e943ab003"
)
DESIGN_SHA256 = (
    "7c52b7e43d02ffa986683c49ca61863c3f36985b97a1a4677a40b6cddef8c150"
)
DESIGN_COMMIT = "ac95a656b7bd1c4c0078f3d8f54a8f5579209aff"

SOURCE_QUALIFICATION_RELATIVE = "manifests/qasc_fresh_source_qualification_v1.json"
SOURCE_QUALIFICATION_SCHEMA = "qasc_fresh_source_qualification_v1"
SOURCE_QUALIFICATION_FILE_SHA256 = (
    "a108b4fe58c2c09a33bcdad36d760d1dc18b26e8dc642bf29d332ffc1b6b5001"
)
SOURCE_QUALIFICATION_SHA256 = (
    "a927ea40ce94eb9428146fdea35712fa6a60d97e19e9b3ce37ee2d49fe9425a2"
)
SOURCE_QUALIFICATION_COMMIT = "38f77d800775ec7c39ea0d4f167af7729a019d65"

SOURCE_CUSTODY_RELATIVE = "manifests/qasc_fresh_source_custody_v1.json"
SOURCE_CUSTODY_SCHEMA = "qasc_fresh_source_custody_v1"
SOURCE_CUSTODY_FILE_SHA256 = (
    "b60d190610cbf6d7f83d581f9ed29b204897183babbfc67688e9e020eec8a7a5"
)
SOURCE_CUSTODY_SHA256 = (
    "acd02819cef73af89aba8a47c058d1f37abc1fae5847262e27a0bd4e6df57b61"
)
SOURCE_CUSTODY_COMMIT = "6a54d371d1fa8870897ef4c2917a078bb2d17e43"

SOURCE_ADDENDUM_RELATIVE = "manifests/qasc_source_access_addendum_v2.json"
SOURCE_ADDENDUM_SCHEMA = "qasc_source_access_addendum_v2"
SOURCE_ADDENDUM_FILE_SHA256 = (
    "e4fb3db6fca8d678f7aa88493c016c9b62066556eac6038c03495e8c3b05764d"
)
SOURCE_ADDENDUM_SHA256 = (
    "2541a7ef700714d3c88f726c22abf0d38658d3ea837f6068fba4585db30162fe"
)
SOURCE_ADDENDUM_COMMIT = "099620ec728043967d52f2f10294a31fecd87e00"

NLI_ASSET_RELATIVE = "manifests/qasc_nli_runtime_asset_v1.json"
NLI_ASSET_SCHEMA = "qasc_nli_runtime_asset_v1"
NLI_ASSET_FILE_SHA256 = (
    "7abe0922a800739cdea06a269310681d50cb00d73e6d9b995d8a147e45d7c961"
)
NLI_ASSET_SHA256 = (
    "d64f4403e7603ea71e622e7e7124eae466cbf67bf4c758979b54c4ccf9bb5fe8"
)
NLI_ASSET_COMMIT = "acce2ebd46c46abfe197aa0241b5748e0ccec2e2"
NLI_RUNTIME_COMMIT = "a248dc8ea3345a036a27a1d4aca652dfbb6cee55"

INFRASTRUCTURE_DIAGNOSTIC_RELATIVE = (
    "manifests/qasc_evaluator_direct_action_infrastructure_diagnostic_v1.json"
)
INFRASTRUCTURE_DIAGNOSTIC_SCHEMA = (
    "qasc_evaluator_direct_action_coevolution_v1_infrastructure_diagnostic"
)
INFRASTRUCTURE_DIAGNOSTIC_STATUS = (
    "passed_row_free_QASC_infrastructure_diagnostic"
)

PREREGISTRATION_RELATIVE = (
    "manifests/qasc_evaluator_direct_action_acquisition_v1_preregistration.json"
)
ACQUISITION_RELATIVE = (
    "manifests/qasc_evaluator_direct_action_acquisition_v1_acquisition.json"
)
SELECTION_SECRET_RELATIVE = "artifacts/qasc_evaluator_custody_v1/selection_v2.key"
SOURCE_ASSET_ROOT_RELATIVE = "artifacts/qasc_official_source_v1"
DATASET_ARCHIVE_RELATIVE = f"{SOURCE_ASSET_ROOT_RELATIVE}/qasc_dataset.tar.gz"
CORPUS_ARCHIVE_RELATIVE = f"{SOURCE_ASSET_ROOT_RELATIVE}/qasc_corpus.tar.gz"
EXTRACTED_CORPUS_RELATIVE = f"{SOURCE_ASSET_ROOT_RELATIVE}/QASC_Corpus.txt"
PRIVATE_PACK_ROOT_RELATIVE = (
    "artifacts/qasc_evaluator_direct_action_coevolution_v1/private_pack"
)
PRIVATE_LOCATOR_RELATIVE = (
    "artifacts/qasc_evaluator_direct_action_coevolution_v1/private_pack.locator.json"
)
CONSUMPTION_RELATIVE = (
    "artifacts/qasc_evaluator_direct_action_acquisition_v1/"
    "authorization.consumed.json"
)
FAILURE_RELATIVE = (
    "artifacts/qasc_evaluator_direct_action_acquisition_v1/"
    "post_marker_failure.json"
)

IMPLEMENTATION_RELATIVE_FILES = (
    "assumption_agent/models.py",
    "assumption_agent/benchmarks/qasc_evaluator_direct_action_acquisition_v1.py",
    "assumption_agent/benchmarks/qasc_counterfactual_chain_margin_v1.py",
    "assumption_agent/benchmarks/qasc_evaluator_direct_action_coevolution_v1.py",
    "replication_runtime/qasc_nli_v1/__init__.py",
    "replication_runtime/qasc_nli_v1/contract.py",
    "replication_runtime/qasc_nli_v1/worker.py",
    "replication_runtime/qasc_nli_v1/binding.py",
)

VIEW_KEYS = frozenset(
    {
        "schema",
        "block",
        "source_member",
        "formatted_question",
        "choices",
        "documents",
        "raw_ranking",
    }
)
LABEL_KEYS = frozenset(
    {
        "schema",
        "block",
        "source_member",
        "identity_commitment_sha256",
        "view_sha256",
        "answerKey",
        "gold_document_ids",
        "fact1_document_id",
        "fact2_document_id",
    }
)
CHOICE_KEYS = frozenset({"label", "text"})
DOCUMENT_KEYS = frozenset({"doc_id", "text", "bm25_score_int"})

TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
BM25_K1 = 1.2
BM25_B = 0.75
BM25_QUANTIZATION = 1_000_000_000_000
HARD_DISTRACTOR_COUNT = 30
LOCAL_DISTRIBUTED_RETAIN_COUNT = 31
LOCAL_DISTRIBUTED_TRACK_COUNT = 32
DOCUMENT_COUNT = 32
RAW_COUNT = 5
DEFAULT_CORPUS_WORKERS = 16
PASS2_BATCH_SIZE = 4096
PASS2_SCREENING_SAFETY_QUANTA = 16
_MIN_FREE_BYTES = 256 * 1024 * 1024
_CORPUS_EXTRACTION_SAFETY_MARGIN_BYTES = 2 * 1024 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SHA1_RE = re.compile(r"[0-9a-f]{40}")


class QASCAcquisitionError(RuntimeError):
    """Raised when a frozen QASC acquisition contract drifts."""


@dataclass(frozen=True)
class Candidate:
    source_member: str
    source_row_ordinal: int
    item_id: str
    normalized_question_sha256: str
    normalized_fact1: str
    normalized_fact2: str
    normalized_fact1_sha256: str
    normalized_fact2_sha256: str
    identity_commitment_sha256: str
    label_free_row_sha256: str
    formatted_question: str
    choices: tuple[tuple[str, str], ...]
    fact1: str
    fact2: str
    answer_key: object


@dataclass(frozen=True)
class BM25Candidate:
    score_int: int
    normalized_fact: str
    exact_fact: str
    source_ordinal: int


@dataclass(frozen=True)
class CorpusStatistics:
    raw_line_count: int
    eligible_document_count: int
    total_token_count: int
    average_document_length: float
    document_frequency: Mapping[str, int]
    chunk_count: int


@dataclass(frozen=True)
class BlockCommitment:
    block: str
    source_member: str
    count: int
    view_file_sha256: str
    label_file_sha256: str
    view_commitment_set_sha256: str
    label_commitment_set_sha256: str
    joined_commitment_set_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise QASCAcquisitionError("value is not canonical JSON") from exc
    return payload.encode("utf-8")


def normalize_text(value: str) -> str:
    if not isinstance(value, str):
        raise QASCAcquisitionError("normalization input must be text")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def tokenize(value: str) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise QASCAcquisitionError("tokenizer input must be text")
    return tuple(TOKEN_RE.findall(unicodedata.normalize("NFKC", value).casefold()))


def canonical_query(formatted_question: str, choices: Sequence[Sequence[str]]) -> str:
    rendered: list[str] = []
    for choice in choices:
        if (
            len(choice) != 2
            or not isinstance(choice[0], str)
            or not isinstance(choice[1], str)
        ):
            raise QASCAcquisitionError("choice is malformed")
        rendered.append(f"{choice[0]}: {choice[1]}")
    return f"{formatted_question} [CHOICES] {' '.join(rendered)}"


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise QASCAcquisitionError(f"{field} must be a lowercase SHA-256")
    return value


def _read_json_object(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise QASCAcquisitionError(f"{field} is unavailable")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QASCAcquisitionError(f"{field} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise QASCAcquisitionError(f"{field} must be one object")
    return payload, raw


def _git(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise QASCAcquisitionError(f"git command failed: {arguments[0]}")
    return completed.stdout


def _committed_binding(
    *, project: Path, path: Path, introducing_commit: str, field: str
) -> dict[str, Any]:
    root = Path(_git(project, "rev-parse", "--show-toplevel").decode().strip())
    actual = path.resolve(strict=True)
    try:
        relative = actual.relative_to(root).as_posix()
    except ValueError as exc:
        raise QASCAcquisitionError(f"{field} is outside the repository") from exc
    live = actual.read_bytes()
    head = _git(root, "show", f"HEAD:{relative}")
    introduced = _git(root, "show", f"{introducing_commit}:{relative}")
    if live != head or live != introduced:
        raise QASCAcquisitionError(f"{field} is not the clean committed blob")
    if _git(root, "status", "--porcelain", "--", relative):
        raise QASCAcquisitionError(f"{field} is dirty")
    ancestor = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", introducing_commit, "HEAD"],
        check=False,
        capture_output=True,
        timeout=30,
    )
    if ancestor.returncode != 0:
        raise QASCAcquisitionError(f"{field} introducing commit is not in HEAD")
    digest = _sha256_bytes(live)
    return {
        "relative_path": Path(relative).relative_to("reconstruction_v2").as_posix(),
        "file_sha256": digest,
        "head_blob_sha256": digest,
        "introducing_commit": introducing_commit,
        "clean_tracked_HEAD_blob": True,
    }


def _load_public_binding(
    *,
    project: Path,
    relative: str,
    schema: str,
    file_sha256: str,
    semantic_field: str,
    semantic_sha256: str,
    commit: str,
    field: str,
    schema_field: str = "schema",
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = project / relative
    payload, raw = _read_json_object(path, field)
    body = dict(payload)
    declared = _require_sha256(body.pop(semantic_field, None), semantic_field)
    custody = _committed_binding(
        project=project, path=path, introducing_commit=commit, field=field
    )
    if (
        payload.get(schema_field) != schema
        or _sha256_bytes(raw) != file_sha256
        or custody["file_sha256"] != file_sha256
        or declared != semantic_sha256
        or stable_hash(body) != declared
    ):
        raise QASCAcquisitionError(f"{field} binding drifted")
    binding = dict(custody)
    binding.update({"schema": schema, semantic_field: semantic_sha256})
    return payload, binding


def public_protocol_bindings(project: Path) -> dict[str, Any]:
    root = project.resolve(strict=True)
    design, design_binding = _load_public_binding(
        project=root,
        relative=DESIGN_RELATIVE,
        schema=DESIGN_SCHEMA,
        file_sha256=DESIGN_FILE_SHA256,
        semantic_field="design_sha256",
        semantic_sha256=DESIGN_SHA256,
        commit=DESIGN_COMMIT,
        field="QASC design",
    )
    qualification, qualification_binding = _load_public_binding(
        project=root,
        relative=SOURCE_QUALIFICATION_RELATIVE,
        schema=SOURCE_QUALIFICATION_SCHEMA,
        file_sha256=SOURCE_QUALIFICATION_FILE_SHA256,
        semantic_field="qualification_sha256",
        semantic_sha256=SOURCE_QUALIFICATION_SHA256,
        commit=SOURCE_QUALIFICATION_COMMIT,
        field="QASC source qualification",
    )
    custody, custody_binding = _load_public_binding(
        project=root,
        relative=SOURCE_CUSTODY_RELATIVE,
        schema=SOURCE_CUSTODY_SCHEMA,
        file_sha256=SOURCE_CUSTODY_FILE_SHA256,
        semantic_field="custody_sha256",
        semantic_sha256=SOURCE_CUSTODY_SHA256,
        commit=SOURCE_CUSTODY_COMMIT,
        field="QASC source custody",
    )
    addendum, addendum_binding = _load_public_binding(
        project=root,
        relative=SOURCE_ADDENDUM_RELATIVE,
        schema=SOURCE_ADDENDUM_SCHEMA,
        file_sha256=SOURCE_ADDENDUM_FILE_SHA256,
        semantic_field="addendum_sha256",
        semantic_sha256=SOURCE_ADDENDUM_SHA256,
        commit=SOURCE_ADDENDUM_COMMIT,
        field="QASC access addendum",
    )
    nli, nli_binding = _load_public_binding(
        project=root,
        relative=NLI_ASSET_RELATIVE,
        schema=NLI_ASSET_SCHEMA,
        file_sha256=NLI_ASSET_FILE_SHA256,
        semantic_field="asset_sha256",
        semantic_sha256=NLI_ASSET_SHA256,
        commit=NLI_ASSET_COMMIT,
        field="QASC NLI asset",
        schema_field="asset_version",
    )
    diagnostic_path = root / INFRASTRUCTURE_DIAGNOSTIC_RELATIVE
    diagnostic, diagnostic_raw = _read_json_object(
        diagnostic_path, "QASC infrastructure diagnostic"
    )
    diagnostic_body = dict(diagnostic)
    diagnostic_sha256 = _require_sha256(
        diagnostic_body.pop("diagnostic_sha256", None), "diagnostic hash"
    )
    diagnostic_custody = _head_binding(
        project=root,
        path=diagnostic_path,
        field="QASC infrastructure diagnostic",
    )
    diagnostic_synthetic = diagnostic.get("synthetic_recipe")
    diagnostic_canary = diagnostic.get("nli_canary")
    if (
        diagnostic.get("schema") != INFRASTRUCTURE_DIAGNOSTIC_SCHEMA
        or stable_hash(diagnostic_body) != diagnostic_sha256
        or diagnostic_custody["file_sha256"] != _sha256_bytes(diagnostic_raw)
        or diagnostic.get("status") != INFRASTRUCTURE_DIAGNOSTIC_STATUS
        or diagnostic.get("design_binding", {}).get("design_sha256")
        != DESIGN_SHA256
        or diagnostic.get("nli_runtime_commit") != NLI_RUNTIME_COMMIT
        or _SHA256_RE.fullmatch(
            str(diagnostic.get("nli_runtime_binding_sha256", ""))
        )
        is None
        or not isinstance(diagnostic_synthetic, Mapping)
        or diagnostic_synthetic.get("document_count") != DOCUMENT_COUNT
        or diagnostic_synthetic.get("choice_count") != 8
        or diagnostic_synthetic.get("view_count") != 1
        or diagnostic_synthetic.get("recipe_count_per_view") != 16
        or diagnostic_synthetic.get("first_wave_item_terminal_count") != 1
        or diagnostic_synthetic.get("second_wave_item_terminal_count") != 1
        or diagnostic_synthetic.get("recipe_action_terminal_count") != 16
        or diagnostic_synthetic.get("two_score_waves_exact") is not True
        or not isinstance(diagnostic_canary, Mapping)
        or diagnostic_canary.get("worker_count") != 8
        or diagnostic_canary.get("torch_threads_per_worker") != 4
        or diagnostic_canary.get("status")
        != "passed_exact_shape_8_worker_repeat_equality_and_capacity"
        or diagnostic.get("formal_QA_rows_read") != 0
        or diagnostic.get("labels_opened") != 0
        or diagnostic.get("network_calls") != 0
        or diagnostic.get("online_evaluator_calls") != 0
        or diagnostic.get("raw_content_persisted") is not False
        or diagnostic.get("implementation_binding") != implementation_binding(root)
    ):
        raise QASCAcquisitionError("QASC infrastructure diagnostic drifted")
    diagnostic_binding = {
        **diagnostic_custody,
        "schema": INFRASTRUCTURE_DIAGNOSTIC_SCHEMA,
        "diagnostic_sha256": diagnostic_sha256,
    }
    if (
        design.get("status")
        != "row_and_outcome_blind_single_final_QASC_mechanism_fixed_before_private_selection_or_retrieval"
        or design.get("selection_contract", {}).get("domain_separator")
        != SELECTION_DOMAIN_SEPARATOR
        or design.get("selection_contract", {}).get(
            "selection_secret_commitment_sha256"
        )
        != SELECTION_SECRET_COMMITMENT_SHA256
        or design.get("cohort_contract", {}).get(
            "private_blocks_selected_and_acquired_before_formation"
        )
        != SELECTED_COUNT
        or design.get("source_mapping_and_eligibility", {})
        .get("source_byte_expectations", {})
        .get("corpus_archive_sha256")
        != CORPUS_ARCHIVE_SHA256
        or design.get("nli_binding", {}).get("asset_sha256") != NLI_ASSET_SHA256
        or design.get("selection_contract", {}).get(
            "formal_candidate_counts_before_fact_group_rules"
        )
        != FORMAL_STEM_CANDIDATE_COUNTS
        or qualification.get("selection_status") != "not_performed"
        or qualification.get("claim_boundary", {}).get(
            "absolute_family_blind_or_family_row_zero_claim"
        )
        is not False
        or custody.get("access_boundary", {}).get("selection_performed") is not False
        or addendum.get("selection_status") != "not_performed"
        or addendum.get("rotated_selection_custody", {}).get(
            "selection_secret_commitment_sha256"
        )
        != SELECTION_SECRET_COMMITMENT_SHA256
        or nli.get("scope", {}).get("formal_qasc_rows_accessed") is not False
    ):
        raise QASCAcquisitionError("public protocol dependency closure drifted")
    return {
        "design": design_binding,
        "source_qualification": qualification_binding,
        "source_custody": custody_binding,
        "source_access_addendum": addendum_binding,
        "nli_asset": nli_binding,
        "infrastructure_diagnostic": diagnostic_binding,
    }


def implementation_binding(project: Path) -> dict[str, Any]:
    root = project.resolve(strict=True)
    repository = Path(_git(root, "rev-parse", "--show-toplevel").decode().strip())
    rows: list[dict[str, Any]] = []
    for relative in IMPLEMENTATION_RELATIVE_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise QASCAcquisitionError(f"implementation file unavailable: {relative}")
        repository_relative = path.resolve().relative_to(repository).as_posix()
        live = path.read_bytes()
        if live != _git(repository, "show", f"HEAD:{repository_relative}"):
            raise QASCAcquisitionError(f"implementation is not committed: {relative}")
        if _git(repository, "status", "--porcelain", "--", repository_relative):
            raise QASCAcquisitionError(f"implementation is dirty: {relative}")
        digest = _sha256_bytes(live)
        rows.append(
            {
                "path": relative,
                "sha256": digest,
                "head_blob_sha256": digest,
                "clean_tracked_HEAD_blob": True,
            }
        )
    return {"files": rows, "set_sha256": stable_hash(rows)}


def _git_ignored(project: Path, path: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(project), "check-ignore", "-q", "--", str(path)],
        check=False,
        capture_output=True,
        timeout=30,
    )
    return completed.returncode == 0


def _canonical_private_path(
    *,
    project: Path,
    supplied: Path,
    relative: str,
    require_file: bool | None,
    field: str,
) -> Path:
    root = project.resolve(strict=True)
    expected = (root / relative).absolute()
    candidate = supplied if supplied.is_absolute() else root / supplied
    if candidate.absolute() != expected or candidate.is_symlink():
        raise QASCAcquisitionError(f"{field} must use its fixed canonical path")
    if not _git_ignored(root, expected):
        raise QASCAcquisitionError(f"{field} must remain git ignored")
    if require_file is True and (not expected.is_file() or expected.is_symlink()):
        raise QASCAcquisitionError(f"{field} is unavailable")
    if require_file is False and expected.exists():
        raise QASCAcquisitionError(f"{field} must not exist yet")
    return expected


def _canonical_public_path(
    *, project: Path, supplied: Path, relative: str, field: str
) -> Path:
    root = project.resolve(strict=True)
    expected = (root / relative).absolute()
    candidate = supplied if supplied.is_absolute() else root / supplied
    try:
        actual = candidate.resolve(strict=True)
        canonical = expected.resolve(strict=True)
    except FileNotFoundError as exc:
        raise QASCAcquisitionError(f"{field} is unavailable") from exc
    if actual != canonical or candidate.is_symlink():
        raise QASCAcquisitionError(f"{field} must use its fixed canonical path")
    return canonical


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_exclusive(path: Path, raw: bytes, *, mode: int) -> None:
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise QASCAcquisitionError("output parent is unavailable")
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
    path: Path, payload: Mapping[str, Any], *, hash_field: str, mode: int
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


def _canonical_selection_secret(project: Path, supplied: Path) -> tuple[Path, bytes]:
    path = _canonical_private_path(
        project=project,
        supplied=supplied,
        relative=SELECTION_SECRET_RELATIVE,
        require_file=True,
        field="selection secret",
    )
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise QASCAcquisitionError("selection secret mode must be 0600")
    raw = path.read_bytes()
    if re.fullmatch(rb"[0-9a-f]{64}\n", raw) is None:
        raise QASCAcquisitionError(
            "selection secret must be lowercase hex with one trailing newline"
        )
    secret = bytes.fromhex(raw[:-1].decode("ascii"))
    if len(secret) != 32 or _sha256_bytes(secret) != SELECTION_SECRET_COMMITMENT_SHA256:
        raise QASCAcquisitionError("selection secret commitment drifted")
    return path, secret


def load_selection_secret(*, project: Path, selection_secret_path: Path) -> bytes:
    """Return the strictly decoded private selection secret."""

    _path, secret = _canonical_selection_secret(project, selection_secret_path)
    return secret


def selection_runtime_binding() -> dict[str, Any]:
    import numpy

    return {
        "python_implementation": sys.implementation.name,
        "python_version_info": list(sys.version_info[:5]),
        "unicode_database_version": unicodedata.unidata_version,
        "json_canonicalization": (
            "ensure_ascii_true_sort_keys_true_separators_comma_colon_allow_nan_false"
        ),
        "BM25_math": "Python_binary64_math_log1p_round_ties_to_even",
        "numpy_version": numpy.__version__,
        "numpy_score_dtype": "float64",
        "numpy_quantized_dtype": "int64",
        "numpy_rint_contract": "IEEE_754_round_to_nearest_ties_to_even",
        "pass2_batch_size": PASS2_BATCH_SIZE,
        "pass2_screening_safety_quanta": PASS2_SCREENING_SAFETY_QUANTA,
        "pass2_retained_and_frontier_scores_recomputed_normatively": True,
    }


def build_preregistration(
    *, project: Path, selection_secret_path: Path
) -> dict[str, Any]:
    """Build a complete preregistration without opening any QASC source bytes."""

    root = project.resolve(strict=True)
    _secret_path, secret = _canonical_selection_secret(root, selection_secret_path)
    payload: dict[str, Any] = {
        "schema": PREREGISTRATION_SCHEMA,
        "decision": "one_shot_four_block_QASC_acquisition_only_no_measurement_authority",
        "public_protocol_bindings": public_protocol_bindings(root),
        "implementation": implementation_binding(root),
        "selection_runtime": selection_runtime_binding(),
        "source": {
            "dataset_archive_sha256": DATASET_ARCHIVE_SHA256,
            "candidate_members": {
                member: {
                    "path": SOURCE_MEMBER_PATHS[member],
                    "file_sha256": SOURCE_MEMBER_SHA256S[member],
                    "row_count": SOURCE_MEMBER_ROW_COUNTS[member],
                }
                for member in SOURCE_MEMBER_PATHS
            },
            "corpus_archive_sha256": CORPUS_ARCHIVE_SHA256,
            "corpus_member_path": CORPUS_MEMBER_PATH,
            "corpus_member_sha256": CORPUS_MEMBER_SHA256,
            "corpus_line_count": CORPUS_LINE_COUNT,
            "TEST_reopened_by_formal_acquisition": False,
            "formal_QA_rows_opened": 0,
        },
        "selection": {
            "method": "post_exposure_private_HMAC_rank_greedy_frozen_constraints",
            "domain_separator": SELECTION_DOMAIN_SEPARATOR,
            "selection_secret_commitment_sha256": _sha256_bytes(secret),
            "secret_persisted_publicly": False,
            "block_order": list(BLOCK_ORDER),
            "block_counts": dict(BLOCK_COUNTS),
            "block_source_members": dict(BLOCK_SOURCE_MEMBERS),
            "selected_count": SELECTED_COUNT,
            "replacement": False,
            "answerKey_used": False,
            "retry_replay_resample": 0,
        },
        "eligibility": {
            "question_collision_members": ["TRAIN", "DEV"],
            "question_collision_source_field": "question.stem",
            "exclude_all_within_member_normalized_question_duplicate_classes": True,
            "exclude_all_TRAIN_DEV_normalized_question_overlap_classes": True,
            "TEST_used_for_formal_collision_filter": False,
            "formal_candidate_counts_before_fact_group_rules": dict(
                FORMAL_STEM_CANDIDATE_COUNTS
            ),
            "formal_counts_enforced_after_marker_without_TEST_reopen": True,
            "TRAIN_selected_fact1_groups_unique_across_three_blocks": True,
            "A_hold_fact2_disjoint_from_formation_union": True,
            "M_search_fact1_groups_unique_within_block": True,
            "M_search_fact2_disjoint_from_all_selected_TRAIN": True,
        },
        "distractor_mining": {
            "schema": "qasc_full_official_corpus_two_pass_targeted_BM25_distractor_miner_v1",
            "multiprocess_byte_chunks": DEFAULT_CORPUS_WORKERS,
            "full_corpus_passes": 2,
            "hard_distractors_per_item": HARD_DISTRACTOR_COUNT,
            "gold_facts_injected_per_item": 2,
            "final_documents_per_item": DOCUMENT_COUNT,
            "answerKey_used": False,
            "performance_scores_computed": 0,
        },
        "access_order": {
            "preregistration_precedes_any_source_open": True,
            "unlabeled_corpus_hash_verify_and_extract_precedes_marker": True,
            "all_output_persistence_canaries_precede_marker": True,
            "durable_marker_precedes_TRAIN_or_DEV_open": True,
            "TEST_never_reopened": True,
            "all_four_blocks_sealed_before_formation": True,
            "post_marker_failure_burns_all_blocks": True,
            "batched_BM25_common_token_exactness_and_throughput_diagnostic_precedes_marker": True,
        },
        "private_schema": {
            "gold_free_view_keys_sha256": stable_hash(sorted(VIEW_KEYS)),
            "label_envelope_keys_sha256": stable_hash(sorted(LABEL_KEYS)),
            "separate_view_and_label_files": True,
            "item_content_or_ID_persisted_publicly": False,
        },
        "safety": {
            "formal_QA_rows_read": 0,
            "TEST_rows_read": 0,
            "model_calls": 0,
            "network_calls": 0,
            "online_evaluator_calls": 0,
            "performance_scores_computed": 0,
        },
    }
    payload["preregistration_sha256"] = stable_hash(payload)
    return payload


def verify_preregistration(
    *, path: Path, project: Path, selection_secret_path: Path
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
        raise QASCAcquisitionError("preregistration self-hash drifted")
    expected = build_preregistration(
        project=project, selection_secret_path=selection_secret_path
    )
    if payload != expected:
        raise QASCAcquisitionError("preregistration differs from live frozen protocol")
    return payload


def _choice_rows(raw: object) -> tuple[tuple[str, str], ...] | None:
    if not isinstance(raw, list) or len(raw) != 8:
        return None
    result: list[tuple[str, str]] = []
    for row in raw:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"label", "text"}
            or not isinstance(row.get("label"), str)
            or not row["label"].strip()
            or not isinstance(row.get("text"), str)
            or not row["text"].strip()
        ):
            return None
        result.append((row["label"], row["text"]))
    if (
        len({label for label, _text in result}) != 8
        or len({text for _label, text in result}) != 8
    ):
        return None
    return tuple(result)


def _parse_candidate(
    raw: object, *, source_member: str, source_row_ordinal: int
) -> Candidate | None:
    if (
        not isinstance(raw, Mapping)
        or source_member not in SOURCE_MEMBER_PATHS
        or type(source_row_ordinal) is not int
        or source_row_ordinal < 0
        or set(raw)
        != {
            "answerKey",
            "combinedfact",
            "fact1",
            "fact2",
            "formatted_question",
            "id",
            "question",
        }
    ):
        return None
    question = raw.get("question")
    if (
        not isinstance(raw.get("id"), str)
        or not raw["id"].strip()
        or not isinstance(raw.get("formatted_question"), str)
        or not raw["formatted_question"].strip()
        or not isinstance(question, Mapping)
        or set(question) != {"stem", "choices"}
        or not isinstance(question.get("stem"), str)
        or not question["stem"].strip()
    ):
        return None
    choices = _choice_rows(question.get("choices"))
    fact1 = raw.get("fact1")
    fact2 = raw.get("fact2")
    combined = raw.get("combinedfact")
    if (
        choices is None
        or not isinstance(fact1, str)
        or not fact1.strip()
        or not isinstance(fact2, str)
        or not fact2.strip()
        or not isinstance(combined, str)
        or not combined.strip()
    ):
        return None
    # Deliberately opaque until after all four HMAC-ranked blocks are fixed.
    # Eligibility, identity, and selection must not depend on the gold label.
    answer_key = raw.get("answerKey")
    normalized_fact1 = normalize_text(fact1)
    normalized_fact2 = normalize_text(fact2)
    if not normalized_fact1 or not normalized_fact2 or normalized_fact1 == normalized_fact2:
        return None
    normalized_formatted = normalize_text(raw["formatted_question"])
    normalized_stem = normalize_text(question["stem"])
    if not normalized_formatted or not normalized_stem:
        return None
    choice_objects = [{"label": label, "text": text} for label, text in choices]
    label_free_row = {
        "source_member": source_member,
        "zero_based_source_row_ordinal": source_row_ordinal,
        "formatted_question": raw["formatted_question"],
        "ordered_choices": choice_objects,
        "fact1": fact1,
        "fact2": fact2,
    }
    label_free_row_sha256 = _sha256_bytes(_canonical_bytes(label_free_row))
    identity_body = {
        "source_member": source_member,
        "zero_based_source_row_ordinal": source_row_ordinal,
        "normalized_formatted_question_sha256": _sha256_bytes(
            normalized_formatted.encode("utf-8")
        ),
        "ordered_choice_labels_and_texts_sha256": _sha256_bytes(
            _canonical_bytes(choice_objects)
        ),
        "normalized_fact1_sha256": _sha256_bytes(
            normalized_fact1.encode("utf-8")
        ),
        "normalized_fact2_sha256": _sha256_bytes(
            normalized_fact2.encode("utf-8")
        ),
        "label_free_row_sha256": label_free_row_sha256,
    }
    return Candidate(
        source_member=source_member,
        source_row_ordinal=source_row_ordinal,
        item_id=raw["id"],
        normalized_question_sha256=_sha256_bytes(normalized_stem.encode("utf-8")),
        normalized_fact1=normalized_fact1,
        normalized_fact2=normalized_fact2,
        normalized_fact1_sha256=identity_body["normalized_fact1_sha256"],
        normalized_fact2_sha256=identity_body["normalized_fact2_sha256"],
        identity_commitment_sha256=_sha256_bytes(_canonical_bytes(identity_body)),
        label_free_row_sha256=label_free_row_sha256,
        formatted_question=raw["formatted_question"],
        choices=choices,
        fact1=fact1,
        fact2=fact2,
        answer_key=answer_key,
    )


def _selection_digest(candidate: Candidate, *, block: str, secret: bytes) -> bytes:
    if block not in BLOCK_ORDER or candidate.source_member != BLOCK_SOURCE_MEMBERS[block]:
        raise QASCAcquisitionError("selection block/source mismatch")
    message = (
        f"{SELECTION_DOMAIN_SEPARATOR}\0select\0{block}\0"
        f"{candidate.source_member}\0{candidate.identity_commitment_sha256}"
    ).encode("utf-8")
    return hmac.new(secret, message, hashlib.sha256).digest()


def _select_candidates(
    candidates: Sequence[Candidate], *, secret: bytes, enforce_formal_counts: bool = False
) -> tuple[dict[str, tuple[Candidate, ...]], dict[str, Any]]:
    question_counts = Counter(row.normalized_question_sha256 for row in candidates)
    id_counts = Counter(row.item_id for row in candidates)
    eligible: dict[str, list[Candidate]] = {"TRAIN": [], "DEV": []}
    excluded = {
        member: {"normalized_question_collision": 0, "duplicate_item_id": 0}
        for member in eligible
    }
    for row in candidates:
        if question_counts[row.normalized_question_sha256] != 1:
            excluded[row.source_member]["normalized_question_collision"] += 1
            continue
        if id_counts[row.item_id] != 1:
            excluded[row.source_member]["duplicate_item_id"] += 1
            continue
        eligible[row.source_member].append(row)
    eligible_counts = {member: len(rows) for member, rows in eligible.items()}
    if (
        enforce_formal_counts
        and eligible_counts != FORMAL_STEM_CANDIDATE_COUNTS
    ):
        raise QASCAcquisitionError("formal TRAIN/DEV candidate counts drifted")

    selected: dict[str, tuple[Candidate, ...]] = {}
    train_fact1_groups: set[str] = set()
    formation_fact2: set[str] = set()
    selected_train_fact2: set[str] = set()
    m_fact1_groups: set[str] = set()
    constraint_rejections: dict[str, Counter[str]] = {
        block: Counter() for block in BLOCK_ORDER
    }
    for block in BLOCK_ORDER:
        member = BLOCK_SOURCE_MEMBERS[block]
        ordered = sorted(
            eligible[member],
            key=lambda row: (
                _selection_digest(row, block=block, secret=secret),
                row.identity_commitment_sha256,
            ),
        )
        accepted: list[Candidate] = []
        for row in ordered:
            if member == "TRAIN" and row.normalized_fact1 in train_fact1_groups:
                constraint_rejections[block]["TRAIN_fact1_group_used"] += 1
                continue
            if block == "A_hold" and row.normalized_fact2 in formation_fact2:
                constraint_rejections[block]["fact2_overlaps_formation"] += 1
                continue
            if block == "M_search" and row.normalized_fact1 in m_fact1_groups:
                constraint_rejections[block]["M_fact1_group_used"] += 1
                continue
            if block == "M_search" and row.normalized_fact2 in selected_train_fact2:
                constraint_rejections[block]["fact2_overlaps_selected_TRAIN"] += 1
                continue
            accepted.append(row)
            if member == "TRAIN":
                train_fact1_groups.add(row.normalized_fact1)
                selected_train_fact2.add(row.normalized_fact2)
                if block in {"A_form", "F_search"}:
                    formation_fact2.add(row.normalized_fact2)
            else:
                m_fact1_groups.add(row.normalized_fact1)
            if len(accepted) == BLOCK_COUNTS[block]:
                break
        if len(accepted) != BLOCK_COUNTS[block]:
            raise QASCAcquisitionError(f"unable to fill frozen block {block}")
        selected[block] = tuple(accepted)

    commitments = [
        row.identity_commitment_sha256
        for block in BLOCK_ORDER
        for row in selected[block]
    ]
    if len(commitments) != SELECTED_COUNT or len(set(commitments)) != SELECTED_COUNT:
        raise QASCAcquisitionError("selected identities overlap")
    return selected, {
        "eligible_candidate_counts": eligible_counts,
        "preselection_exclusions": excluded,
        "constraint_rejections_by_block": {
            block: dict(constraint_rejections[block]) for block in BLOCK_ORDER
        },
        "selected_TRAIN_fact1_group_count": len(train_fact1_groups),
        "selected_M_search_fact1_group_count": len(m_fact1_groups),
        "formation_fact2_normalization_count": len(formation_fact2),
        "selected_TRAIN_fact2_normalization_count": len(selected_train_fact2),
    }


def _exact_tar_member(archive: tarfile.TarFile, expected: str) -> tarfile.TarInfo:
    matches = [member for member in archive.getmembers() if member.name == expected]
    if len(matches) != 1 or not matches[0].isfile():
        raise QASCAcquisitionError(f"exact tar member is ambiguous: {expected}")
    return matches[0]


def _extract_stream_exclusive(
    *, source: BinaryIO, destination: Path, expected_sha256: str, expected_size: int
) -> None:
    if destination.exists():
        raise FileExistsError(f"extracted corpus already exists: {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = destination.parent / (
        f".{destination.name}.{os.urandom(12).hex()}.tmp"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            while True:
                chunk = source.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if size != expected_size or digest.hexdigest() != expected_sha256:
            raise QASCAcquisitionError("extracted corpus bytes drifted")
        try:
            os.link(temporary, destination, follow_symlinks=False)
        finally:
            temporary.unlink(missing_ok=True)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def prepare_unlabeled_corpus(
    *, corpus_archive: Path, extracted_corpus: Path
) -> dict[str, Any]:
    """Hash-verify and safely extract only the unlabeled official corpus."""

    if corpus_archive.is_symlink() or not corpus_archive.is_file():
        raise QASCAcquisitionError("corpus archive is unavailable")
    if _sha256_file(corpus_archive) != CORPUS_ARCHIVE_SHA256:
        raise QASCAcquisitionError("corpus archive hash drifted")
    if extracted_corpus.exists():
        if (
            extracted_corpus.is_symlink()
            or not extracted_corpus.is_file()
            or extracted_corpus.stat().st_size != CORPUS_MEMBER_SIZE
            or _sha256_file(extracted_corpus) != CORPUS_MEMBER_SHA256
        ):
            raise QASCAcquisitionError("existing extracted corpus drifted")
        return {
            "archive_sha256": CORPUS_ARCHIVE_SHA256,
            "member_sha256": CORPUS_MEMBER_SHA256,
            "member_size": CORPUS_MEMBER_SIZE,
            "reused_verified_extraction": True,
        }
    if shutil.disk_usage(extracted_corpus.parent).free < (
        CORPUS_MEMBER_SIZE + _CORPUS_EXTRACTION_SAFETY_MARGIN_BYTES
    ):
        raise QASCAcquisitionError("insufficient space for corpus extraction")
    with tarfile.open(corpus_archive, "r:gz") as archive:
        member = _exact_tar_member(archive, CORPUS_MEMBER_PATH)
        if member.size != CORPUS_MEMBER_SIZE:
            raise QASCAcquisitionError("corpus member size drifted")
        source = archive.extractfile(member)
        if source is None:
            raise QASCAcquisitionError("corpus member is unavailable")
        with source:
            _extract_stream_exclusive(
                source=source,
                destination=extracted_corpus,
                expected_sha256=CORPUS_MEMBER_SHA256,
                expected_size=CORPUS_MEMBER_SIZE,
            )
    return {
        "archive_sha256": CORPUS_ARCHIVE_SHA256,
        "member_sha256": CORPUS_MEMBER_SHA256,
        "member_size": CORPUS_MEMBER_SIZE,
        "reused_verified_extraction": False,
    }


def _load_qa_member(
    archive: tarfile.TarFile, *, source_member: str
) -> tuple[Candidate, ...]:
    member = _exact_tar_member(archive, SOURCE_MEMBER_PATHS[source_member])
    source = archive.extractfile(member)
    if source is None:
        raise QASCAcquisitionError(f"{source_member} member is unavailable")
    digest = hashlib.sha256()
    rows: list[Candidate] = []
    ids: set[str] = set()
    with source:
        for ordinal, raw_line in enumerate(source):
            digest.update(raw_line)
            if not raw_line.endswith(b"\n") or not raw_line.strip():
                raise QASCAcquisitionError(f"{source_member} JSONL framing drifted")
            try:
                raw = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise QASCAcquisitionError(
                    f"{source_member} contains invalid JSON"
                ) from exc
            candidate = _parse_candidate(
                raw, source_member=source_member, source_row_ordinal=ordinal
            )
            if candidate is None:
                raise QASCAcquisitionError(
                    f"{source_member} label-free schema drifted"
                )
            if candidate.item_id in ids:
                raise QASCAcquisitionError(f"{source_member} duplicate ID drifted")
            ids.add(candidate.item_id)
            rows.append(candidate)
    if (
        len(rows) != SOURCE_MEMBER_ROW_COUNTS[source_member]
        or digest.hexdigest() != SOURCE_MEMBER_SHA256S[source_member]
    ):
        raise QASCAcquisitionError(f"{source_member} count or hash drifted")
    return tuple(rows)


def load_formal_candidates(dataset_archive: Path) -> tuple[Candidate, ...]:
    """Open exactly TRAIN and DEV after the caller has consumed authorization."""

    if dataset_archive.is_symlink() or not dataset_archive.is_file():
        raise QASCAcquisitionError("dataset archive is unavailable")
    if _sha256_file(dataset_archive) != DATASET_ARCHIVE_SHA256:
        raise QASCAcquisitionError("dataset archive hash drifted")
    with tarfile.open(dataset_archive, "r:gz") as archive:
        train = _load_qa_member(archive, source_member="TRAIN")
        dev = _load_qa_member(archive, source_member="DEV")
    ids = [row.item_id for row in (*train, *dev)]
    if len(ids) != len(set(ids)):
        raise QASCAcquisitionError("TRAIN/DEV item IDs overlap")
    return (*train, *dev)


def _byte_chunks(file_size: int, workers: int) -> tuple[tuple[int, int], ...]:
    if file_size <= 0 or workers <= 0:
        raise QASCAcquisitionError("corpus chunk shape is invalid")
    count = min(workers, file_size)
    boundaries = [(file_size * index) // count for index in range(count + 1)]
    return tuple(
        (boundaries[index], boundaries[index + 1]) for index in range(count)
    )


def _iter_chunk_lines(
    path: str, start: int, end: int
) -> Iterator[tuple[int, str, tuple[str, ...]]]:
    """Yield each whole line whose first byte belongs to ``[start,end)``."""

    with open(path, "rb") as handle:
        if start:
            handle.seek(start - 1)
            previous = handle.read(1)
            if previous != b"\n":
                handle.readline()
        else:
            handle.seek(0)
        local_ordinal = 0
        while handle.tell() < end:
            raw = handle.readline()
            if not raw:
                break
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise QASCAcquisitionError("corpus line is not UTF-8") from exc
            if text.endswith("\n"):
                text = text[:-1]
                if text.endswith("\r"):
                    text = text[:-1]
            tokens = tokenize(text)
            yield local_ordinal, text, tokens
            local_ordinal += 1


def _pass1_worker(
    arguments: tuple[str, int, int, tuple[str, ...]]
) -> dict[str, Any]:
    path, start, end, union_tokens = arguments
    union = frozenset(union_tokens)
    raw_count = 0
    eligible_count = 0
    total_length = 0
    df: Counter[str] = Counter()
    for _local, _text, tokens in _iter_chunk_lines(path, start, end):
        raw_count += 1
        if not tokens:
            continue
        eligible_count += 1
        total_length += len(tokens)
        df.update(set(tokens) & union)
    return {
        "raw_line_count": raw_count,
        "eligible_document_count": eligible_count,
        "total_token_count": total_length,
        "document_frequency": dict(df),
    }


def _idf(document_count: int, document_frequency: int) -> float:
    if (
        type(document_count) is not int
        or type(document_frequency) is not int
        or document_count <= 0
        or not 0 <= document_frequency <= document_count
    ):
        raise QASCAcquisitionError("BM25 document frequency is invalid")
    return math.log1p(
        (document_count - document_frequency + 0.5)
        / (document_frequency + 0.5)
    )


def bm25_score_int(
    *,
    query_tokens: Sequence[str],
    document_tokens: Sequence[str],
    document_count: int,
    total_token_count: int,
    document_frequency: Mapping[str, int],
) -> int:
    if document_count <= 0 or total_token_count <= 0 or not document_tokens:
        raise QASCAcquisitionError("BM25 corpus statistics are invalid")
    average_length = total_token_count / document_count
    frequencies = Counter(document_tokens)
    score = 0.0
    for token in sorted(set(query_tokens)):
        tf = frequencies.get(token, 0)
        if not tf:
            continue
        denominator = tf + BM25_K1 * (
            1.0 - BM25_B + BM25_B * len(document_tokens) / average_length
        )
        score += (
            _idf(document_count, int(document_frequency.get(token, 0)))
            * tf
            * (BM25_K1 + 1.0)
            / denominator
        )
    return int(round(score * BM25_QUANTIZATION))


def _bm25_term_contribution(
    *,
    token: str,
    term_frequency: int,
    document_length: int,
    document_count: int,
    total_token_count: int,
    document_frequency: Mapping[str, int],
) -> float:
    average_length = total_token_count / document_count
    denominator = term_frequency + BM25_K1 * (
        1.0 - BM25_B + BM25_B * document_length / average_length
    )
    return (
        _idf(document_count, int(document_frequency.get(token, 0)))
        * term_frequency
        * (BM25_K1 + 1.0)
        / denominator
    )


def _bm25_candidate_key(row: BM25Candidate) -> tuple[Any, ...]:
    return (
        -row.score_int,
        row.normalized_fact.encode("utf-8"),
        row.exact_fact.encode("utf-8"),
        row.source_ordinal,
    )


def _update_top_unique(
    rows: dict[str, BM25Candidate], candidate: BM25Candidate, *, limit: int
) -> None:
    if candidate.score_int <= 0 or not candidate.normalized_fact:
        return
    previous = rows.get(candidate.normalized_fact)
    if previous is not None:
        if _bm25_candidate_key(candidate) < _bm25_candidate_key(previous):
            rows[candidate.normalized_fact] = candidate
        return
    if len(rows) < limit:
        rows[candidate.normalized_fact] = candidate
        return
    worst_normalized, worst = max(
        rows.items(), key=lambda pair: _bm25_candidate_key(pair[1])
    )
    if _bm25_candidate_key(candidate) < _bm25_candidate_key(worst):
        del rows[worst_normalized]
        rows[candidate.normalized_fact] = candidate


def batched_bm25_score_matrix(
    *,
    document_token_rows: Sequence[Sequence[str]],
    query_token_rows: Sequence[Sequence[str]],
    document_count: int,
    total_token_count: int,
    document_frequency: Mapping[str, int],
) -> Any:
    """Return int64 scores with the exact frozen ascending-token sum order."""

    import numpy as np

    normalized_queries = [tuple(sorted(set(row))) for row in query_token_rows]
    inverted: dict[str, list[int]] = {}
    for query_index, tokens in enumerate(normalized_queries):
        for token in tokens:
            inverted.setdefault(token, []).append(query_index)
    scores = np.zeros(
        (len(document_token_rows), len(normalized_queries)), dtype=np.float64
    )
    occurrences: dict[str, list[tuple[int, int, int]]] = {}
    for row_index, document_tokens in enumerate(document_token_rows):
        frequencies = Counter(document_tokens)
        for token, term_frequency in frequencies.items():
            if token in inverted:
                occurrences.setdefault(token, []).append(
                    (row_index, term_frequency, len(document_tokens))
                )
    for token in sorted(occurrences):
        rows = occurrences[token]
        row_indices = np.fromiter((row[0] for row in rows), dtype=np.int64)
        contributions = np.fromiter(
            (
                _bm25_term_contribution(
                    token=token,
                    term_frequency=row[1],
                    document_length=row[2],
                    document_count=document_count,
                    total_token_count=total_token_count,
                    document_frequency=document_frequency,
                )
                for row in rows
            ),
            dtype=np.float64,
        )
        query_columns = np.asarray(inverted[token], dtype=np.int64)
        scores[np.ix_(row_indices, query_columns)] += contributions[:, None]
    return np.rint(scores * BM25_QUANTIZATION).astype(np.int64)


def run_synthetic_bm25_batch_diagnostic() -> dict[str, Any]:
    """Certify dense common-token batching against every scalar score."""

    import time

    documents = [
        tuple(
            ["the"] * (1 + index % 4)
            + [f"group{index % 17}", f"signal{index % 31}"]
            + (["café", "１２"] if index % 7 == 0 else ["plain"])
        )
        for index in range(1024)
    ]
    queries = [
        tuple(sorted({"the", f"group{index % 17}", f"signal{index % 31}"}))
        for index in range(32)
    ]
    document_frequency: Counter[str] = Counter()
    for row in documents:
        document_frequency.update(set(row))
    total_token_count = sum(len(row) for row in documents)
    start = time.perf_counter()
    first = batched_bm25_score_matrix(
        document_token_rows=documents,
        query_token_rows=queries,
        document_count=len(documents),
        total_token_count=total_token_count,
        document_frequency=document_frequency,
    )
    second = batched_bm25_score_matrix(
        document_token_rows=documents,
        query_token_rows=queries,
        document_count=len(documents),
        total_token_count=total_token_count,
        document_frequency=document_frequency,
    )
    if first.tolist() != second.tolist():
        raise QASCAcquisitionError("batched BM25 repeat equality failed")
    for document_index, document_tokens in enumerate(documents):
        for query_index, query_tokens in enumerate(queries):
            scalar = bm25_score_int(
                query_tokens=query_tokens,
                document_tokens=document_tokens,
                document_count=len(documents),
                total_token_count=total_token_count,
                document_frequency=document_frequency,
            )
            if int(first[document_index, query_index]) != scalar:
                raise QASCAcquisitionError("batched BM25 differs from scalar contract")
    elapsed = time.perf_counter() - start
    pair_count = len(documents) * len(queries)
    rate = pair_count / elapsed
    if rate < 10_000:
        raise QASCAcquisitionError("batched BM25 synthetic throughput is insufficient")
    return {
        "schema": "qasc_batched_BM25_common_token_diagnostic_v1",
        "document_count": len(documents),
        "query_count": len(queries),
        "pair_count": pair_count,
        "common_token_present_in_every_document_and_query": True,
        "repeat_exact": True,
        "scalar_equality_checked_pair_count": pair_count,
        "integer_matrix_sha256": stable_hash(first.tolist()),
        "minimum_pairs_per_second": 10_000,
        "observed_pairs_per_second_rounded_down": int(rate),
        "formal_QA_rows_read": 0,
    }


def _pass2_worker(arguments: tuple[Any, ...]) -> dict[str, Any]:
    (
        path,
        start,
        end,
        ordinal_offset,
        query_specs,
        document_count,
        total_token_count,
        document_frequency,
    ) = arguments
    queries: dict[int, tuple[tuple[str, ...], frozenset[str]]] = {}
    item_order: list[int] = []
    for item_index, query_tokens, excluded in query_specs:
        tokens = tuple(query_tokens)
        queries[item_index] = (tokens, frozenset(excluded))
        item_order.append(item_index)
    top: dict[int, dict[str, BM25Candidate]] = {
        item_index: {} for item_index in queries
    }
    iterator = iter(_iter_chunk_lines(path, start, end))
    while True:
        batch: list[tuple[int, str, tuple[str, ...]]] = []
        try:
            for _ in range(PASS2_BATCH_SIZE):
                batch.append(next(iterator))
        except StopIteration:
            pass
        if not batch:
            break
        quantized = batched_bm25_score_matrix(
            document_token_rows=[row[2] for row in batch],
            query_token_rows=[queries[item_index][0] for item_index in item_order],
            document_count=document_count,
            total_token_count=total_token_count,
            document_frequency=document_frequency,
        )
        import numpy as np
        for query_column, item_index in enumerate(item_order):
            current = top[item_index]
            threshold = (
                max(current.values(), key=_bm25_candidate_key).score_int
                if len(current) >= LOCAL_DISTRIBUTED_TRACK_COUNT
                else 1
            )
            candidate_indices = np.flatnonzero(
                quantized[:, query_column]
                >= threshold - PASS2_SCREENING_SAFETY_QUANTA
            )
            query_tokens, excluded = queries[item_index]
            for row_index_raw in candidate_indices:
                row_index = int(row_index_raw)
                local_ordinal, exact_fact, document_tokens = batch[row_index]
                if not document_tokens:
                    continue
                normalized = normalize_text(exact_fact)
                if normalized in excluded:
                    continue
                # Retained/frontier rows are always recomputed by the normative
                # ascending-token Python-binary64 implementation.
                score = bm25_score_int(
                    query_tokens=query_tokens,
                    document_tokens=document_tokens,
                    document_count=document_count,
                    total_token_count=total_token_count,
                    document_frequency=document_frequency,
                )
                _update_top_unique(
                    current,
                    BM25Candidate(
                        score_int=score,
                        normalized_fact=normalized,
                        exact_fact=exact_fact,
                        source_ordinal=ordinal_offset + local_ordinal,
                    ),
                    limit=LOCAL_DISTRIBUTED_TRACK_COUNT,
                )
        if len(batch) < PASS2_BATCH_SIZE:
            break
    retained: dict[int, tuple[BM25Candidate, ...]] = {}
    frontier: dict[int, BM25Candidate | None] = {}
    for item_index, rows in top.items():
        ordered = tuple(sorted(rows.values(), key=_bm25_candidate_key))
        retained[item_index] = ordered[:LOCAL_DISTRIBUTED_RETAIN_COUNT]
        frontier[item_index] = (
            ordered[LOCAL_DISTRIBUTED_RETAIN_COUNT]
            if len(ordered) > LOCAL_DISTRIBUTED_RETAIN_COUNT
            else None
        )
    return {"retained": retained, "frontier": frontier}


def mine_distractors(
    *,
    corpus_path: Path,
    selected: Mapping[str, Sequence[Candidate]],
    workers: int = DEFAULT_CORPUS_WORKERS,
) -> tuple[
    dict[str, tuple[tuple[BM25Candidate, ...], ...]], CorpusStatistics, dict[str, Any]
]:
    """Run the exact selected-query two-pass full-corpus BM25 miner."""

    flat: list[tuple[str, Candidate]] = [
        (block, row) for block in BLOCK_ORDER for row in selected[block]
    ]
    query_tokens = [
        tuple(sorted(set(tokenize(canonical_query(row.formatted_question, row.choices)))))
        for _block, row in flat
    ]
    if any(not tokens for tokens in query_tokens):
        raise QASCAcquisitionError("selected retrieval query has no tokens")
    union_tokens = tuple(sorted({token for tokens in query_tokens for token in tokens}))
    chunks = _byte_chunks(corpus_path.stat().st_size, workers)
    pass1_arguments = [
        (str(corpus_path), start, end, union_tokens) for start, end in chunks
    ]
    if len(chunks) == 1:
        pass1 = [_pass1_worker(pass1_arguments[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            pass1 = list(pool.map(_pass1_worker, pass1_arguments))
    raw_counts = [int(row["raw_line_count"]) for row in pass1]
    raw_line_count = sum(raw_counts)
    eligible_count = sum(int(row["eligible_document_count"]) for row in pass1)
    total_token_count = sum(int(row["total_token_count"]) for row in pass1)
    if raw_line_count != CORPUS_LINE_COUNT or eligible_count <= 0 or total_token_count <= 0:
        raise QASCAcquisitionError("full corpus pass-1 count drifted")
    document_frequency: Counter[str] = Counter()
    for row in pass1:
        document_frequency.update(row["document_frequency"])
    offsets: list[int] = []
    running = 0
    for count in raw_counts:
        offsets.append(running)
        running += count
    query_specs = tuple(
        (
            index,
            query_tokens[index],
            (row.normalized_fact1, row.normalized_fact2),
        )
        for index, (_block, row) in enumerate(flat)
    )
    pass2_arguments = [
        (
            str(corpus_path),
            start,
            end,
            offsets[index],
            query_specs,
            eligible_count,
            total_token_count,
            dict(document_frequency),
        )
        for index, (start, end) in enumerate(chunks)
    ]
    if len(chunks) == 1:
        pass2 = [_pass2_worker(pass2_arguments[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            pass2 = list(pool.map(_pass2_worker, pass2_arguments))
    merged: dict[int, dict[str, BM25Candidate]] = {
        index: {} for index in range(len(flat))
    }
    for worker_rows in pass2:
        for item_index, rows in worker_rows["retained"].items():
            for row in rows:
                _update_top_unique(
                    merged[item_index], row, limit=HARD_DISTRACTOR_COUNT
                )
    ordered: list[tuple[BM25Candidate, ...]] = []
    for index in range(len(flat)):
        rows = tuple(sorted(merged[index].values(), key=_bm25_candidate_key))
        if len(rows) != HARD_DISTRACTOR_COUNT or any(row.score_int <= 0 for row in rows):
            raise QASCAcquisitionError(
                "fewer than 30 positive unique non-gold distractors"
            )
        ordered.append(rows)
    for worker_rows in pass2:
        for item_index, frontier in worker_rows["frontier"].items():
            if frontier is None:
                continue
            final_worst = ordered[item_index][-1]
            if _bm25_candidate_key(frontier) < _bm25_candidate_key(final_worst):
                raise QASCAcquisitionError(
                    "distributed top-k omission frontier certification failed"
                )
    by_block: dict[str, tuple[tuple[BM25Candidate, ...], ...]] = {}
    cursor = 0
    for block in BLOCK_ORDER:
        count = len(selected[block])
        by_block[block] = tuple(ordered[cursor : cursor + count])
        cursor += count
    statistics = CorpusStatistics(
        raw_line_count=raw_line_count,
        eligible_document_count=eligible_count,
        total_token_count=total_token_count,
        average_document_length=total_token_count / eligible_count,
        document_frequency=dict(document_frequency),
        chunk_count=len(chunks),
    )
    public_stats = {
        "raw_line_count": raw_line_count,
        "eligible_document_count": eligible_count,
        "total_token_count": total_token_count,
        "average_document_length_hex": statistics.average_document_length.hex(),
        "union_query_token_count": len(union_tokens),
        "document_frequency_commitment_sha256": stable_hash(
            sorted(document_frequency.items())
        ),
        "chunk_count": len(chunks),
        "full_corpus_pass_count": 2,
        "distributed_unique_topk_retained_per_chunk_item": (
            LOCAL_DISTRIBUTED_RETAIN_COUNT
        ),
        "first_omitted_unique_candidate_ordinal": (
            LOCAL_DISTRIBUTED_RETAIN_COUNT + 1
        ),
        "all_omission_frontiers_certified_against_global_top30": True,
    }
    return by_block, statistics, public_stats


def _doc_order_digest(
    *, identity_commitment_sha256: str, exact_fact: str, secret: bytes
) -> bytes:
    fact_sha256 = _sha256_bytes(exact_fact.encode("utf-8"))
    message = (
        f"{SELECTION_DOMAIN_SEPARATOR}\0doc_order\0"
        f"{identity_commitment_sha256}\0{fact_sha256}"
    ).encode("utf-8")
    return hmac.new(secret, message, hashlib.sha256).digest()


def _validate_private_pair(
    view: Mapping[str, Any], label: Mapping[str, Any], *, expected_block: str
) -> None:
    if set(view) != VIEW_KEYS or set(label) != LABEL_KEYS:
        raise QASCAcquisitionError("private view or label key set drifted")
    if (
        view.get("schema") != PRIVATE_VIEW_SCHEMA
        or label.get("schema") != PRIVATE_LABEL_SCHEMA
        or view.get("block") != expected_block
        or label.get("block") != expected_block
        or view.get("source_member") != BLOCK_SOURCE_MEMBERS[expected_block]
        or label.get("source_member") != BLOCK_SOURCE_MEMBERS[expected_block]
    ):
        raise QASCAcquisitionError("private view or label identity drifted")
    _require_sha256(label.get("identity_commitment_sha256"), "identity")
    choices = view.get("choices")
    documents = view.get("documents")
    ranking = view.get("raw_ranking")
    if (
        not isinstance(choices, list)
        or len(choices) != 8
        or any(not isinstance(row, Mapping) or set(row) != CHOICE_KEYS for row in choices)
        or len({row["label"] for row in choices}) != 8
        or len({row["text"] for row in choices}) != 8
        or any(
            not isinstance(row["label"], str)
            or not row["label"].strip()
            or not isinstance(row["text"], str)
            or not row["text"].strip()
            for row in choices
        )
        or not isinstance(documents, list)
        or len(documents) != DOCUMENT_COUNT
        or any(
            not isinstance(row, Mapping)
            or set(row) != DOCUMENT_KEYS
            or type(row.get("doc_id")) is not int
            or not isinstance(row.get("text"), str)
            or not row["text"]
            or type(row.get("bm25_score_int")) is not int
            for row in documents
        )
        or [row["doc_id"] for row in documents] != list(range(DOCUMENT_COUNT))
        or len({normalize_text(row["text"]) for row in documents}) != DOCUMENT_COUNT
        or not isinstance(ranking, list)
        or len(ranking) != RAW_COUNT
        or any(type(index) is not int for index in ranking)
        or len(set(ranking)) != RAW_COUNT
        or any(not 0 <= index < DOCUMENT_COUNT for index in ranking)
        or ranking
        != [
            row["doc_id"]
            for row in sorted(
                documents,
                key=lambda row: (-row["bm25_score_int"], row["doc_id"]),
            )[:RAW_COUNT]
        ]
    ):
        raise QASCAcquisitionError("gold-free view payload drifted")
    if label.get("view_sha256") != stable_hash(view):
        raise QASCAcquisitionError("view join hash drifted")
    labels = {row["label"] for row in choices}
    gold_ids = label.get("gold_document_ids")
    fact1_id = label.get("fact1_document_id")
    fact2_id = label.get("fact2_document_id")
    if (
        label.get("answerKey") not in labels
        or not isinstance(gold_ids, list)
        or len(gold_ids) != 2
        or len(set(gold_ids)) != 2
        or sorted(gold_ids) != gold_ids
        or type(fact1_id) is not int
        or type(fact2_id) is not int
        or fact1_id == fact2_id
        or not 0 <= fact1_id < DOCUMENT_COUNT
        or not 0 <= fact2_id < DOCUMENT_COUNT
        or sorted([fact1_id, fact2_id]) != gold_ids
    ):
        raise QASCAcquisitionError("private label envelope drifted")


def build_private_pair(
    *,
    candidate: Candidate,
    block: str,
    distractors: Sequence[BM25Candidate],
    statistics: CorpusStatistics,
    secret: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if candidate.source_member != BLOCK_SOURCE_MEMBERS.get(block):
        raise QASCAcquisitionError("private pair block/source mismatch")
    if len(distractors) != HARD_DISTRACTOR_COUNT:
        raise QASCAcquisitionError("private pair distractor count drifted")
    exact_facts = [candidate.fact1, candidate.fact2, *[row.exact_fact for row in distractors]]
    normalized_facts = [normalize_text(fact) for fact in exact_facts]
    if len(exact_facts) != DOCUMENT_COUNT or len(set(normalized_facts)) != DOCUMENT_COUNT:
        raise QASCAcquisitionError("private document normalized facts are not unique")
    query_tokens = tuple(
        sorted(
            set(
                tokenize(
                    canonical_query(candidate.formatted_question, candidate.choices)
                )
            )
        )
    )
    ordered = sorted(
        zip(exact_facts, normalized_facts),
        key=lambda pair: (
            _doc_order_digest(
                identity_commitment_sha256=candidate.identity_commitment_sha256,
                exact_fact=pair[0],
                secret=secret,
            ),
            pair[1].encode("utf-8"),
            pair[0].encode("utf-8"),
        ),
    )
    documents: list[dict[str, Any]] = []
    fact1_id: int | None = None
    fact2_id: int | None = None
    for doc_id, (exact_fact, _normalized) in enumerate(ordered):
        if exact_fact == candidate.fact1:
            fact1_id = doc_id
        if exact_fact == candidate.fact2:
            fact2_id = doc_id
        documents.append(
            {
                "doc_id": doc_id,
                "text": exact_fact,
                "bm25_score_int": bm25_score_int(
                    query_tokens=query_tokens,
                    document_tokens=tokenize(exact_fact),
                    document_count=statistics.eligible_document_count,
                    total_token_count=statistics.total_token_count,
                    document_frequency=statistics.document_frequency,
                ),
            }
        )
    if fact1_id is None or fact2_id is None or fact1_id == fact2_id:
        raise QASCAcquisitionError("gold document mapping failed")
    raw_ranking = [
        row["doc_id"]
        for row in sorted(
            documents, key=lambda row: (-row["bm25_score_int"], row["doc_id"])
        )[:RAW_COUNT]
    ]
    view: dict[str, Any] = {
        "schema": PRIVATE_VIEW_SCHEMA,
        "block": block,
        "source_member": candidate.source_member,
        "formatted_question": candidate.formatted_question,
        "choices": [
            {"label": label, "text": text} for label, text in candidate.choices
        ],
        "documents": documents,
        "raw_ranking": raw_ranking,
    }
    if not isinstance(candidate.answer_key, str):
        raise QASCAcquisitionError("selected answerKey is malformed")
    label = {
        "schema": PRIVATE_LABEL_SCHEMA,
        "block": block,
        "source_member": candidate.source_member,
        "identity_commitment_sha256": candidate.identity_commitment_sha256,
        "view_sha256": stable_hash(view),
        "answerKey": candidate.answer_key,
        "gold_document_ids": sorted([fact1_id, fact2_id]),
        "fact1_document_id": fact1_id,
        "fact2_document_id": fact2_id,
    }
    _validate_private_pair(view, label, expected_block=block)
    return view, label


def build_private_blocks(
    *,
    selected: Mapping[str, Sequence[Candidate]],
    distractors: Mapping[str, Sequence[Sequence[BM25Candidate]]],
    statistics: CorpusStatistics,
    secret: bytes,
) -> tuple[
    dict[str, tuple[dict[str, Any], ...]],
    dict[str, tuple[dict[str, Any], ...]],
]:
    views: dict[str, tuple[dict[str, Any], ...]] = {}
    labels: dict[str, tuple[dict[str, Any], ...]] = {}
    for block in BLOCK_ORDER:
        if (
            len(selected[block]) != BLOCK_COUNTS[block]
            or len(distractors[block]) != BLOCK_COUNTS[block]
        ):
            raise QASCAcquisitionError("private block materialization shape drifted")
        pairs = [
            build_private_pair(
                candidate=candidate,
                block=block,
                distractors=item_distractors,
                statistics=statistics,
                secret=secret,
            )
            for candidate, item_distractors in zip(
                selected[block], distractors[block], strict=True
            )
        ]
        views[block] = tuple(pair[0] for pair in pairs)
        labels[block] = tuple(pair[1] for pair in pairs)
    return views, labels


def _joined_commitment(
    views: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]]
) -> str:
    label_by_view = {row["view_sha256"]: row for row in labels}
    rows: list[dict[str, Any]] = []
    for view in views:
        view_sha256 = stable_hash(view)
        label = label_by_view.get(view_sha256)
        if label is None:
            raise QASCAcquisitionError("private join is incomplete")
        rows.append(
            {
                "identity_commitment_sha256": label[
                    "identity_commitment_sha256"
                ],
                "view_sha256": view_sha256,
                "label_envelope_sha256": stable_hash(label),
            }
        )
    if len(rows) != len(labels) or len(label_by_view) != len(labels):
        raise QASCAcquisitionError("private join cardinality drifted")
    return stable_hash(rows)


def _validate_commitment_identity(
    commitment: BlockCommitment, *, expected_block: str
) -> None:
    if (
        commitment.block != expected_block
        or expected_block not in BLOCK_ORDER
        or commitment.source_member != BLOCK_SOURCE_MEMBERS[expected_block]
        or commitment.count != BLOCK_COUNTS[expected_block]
    ):
        raise QASCAcquisitionError("private block commitment identity drifted")


def _read_canonical_jsonl(path: Path, *, digest: str, field: str) -> tuple[dict[str, Any], ...]:
    if path.is_symlink() or not path.is_file() or _sha256_file(path) != digest:
        raise QASCAcquisitionError(f"private {field} file hash drifted")
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise QASCAcquisitionError("private JSONL framing drifted")
    try:
        rows = tuple(json.loads(line) for line in raw.splitlines())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QASCAcquisitionError("private JSONL is invalid") from exc
    if any(not isinstance(row, dict) for row in rows) or b"".join(
        _canonical_bytes(row) + b"\n" for row in rows
    ) != raw:
        raise QASCAcquisitionError("private JSONL is noncanonical")
    return rows


def _validate_view_only(view: Mapping[str, Any], *, expected_block: str) -> None:
    if (
        set(view) != VIEW_KEYS
        or view.get("schema") != PRIVATE_VIEW_SCHEMA
        or view.get("block") != expected_block
        or view.get("source_member") != BLOCK_SOURCE_MEMBERS[expected_block]
    ):
        raise QASCAcquisitionError("private view identity drifted")
    choices = view.get("choices")
    documents = view.get("documents")
    ranking = view.get("raw_ranking")
    if (
        not isinstance(view.get("formatted_question"), str)
        or not view["formatted_question"]
        or not isinstance(choices, list)
        or len(choices) != 8
        or any(not isinstance(row, Mapping) or set(row) != CHOICE_KEYS for row in choices)
        or len({row["label"] for row in choices}) != 8
        or len({row["text"] for row in choices}) != 8
        or any(
            not isinstance(row["label"], str)
            or not row["label"].strip()
            or not isinstance(row["text"], str)
            or not row["text"].strip()
            for row in choices
        )
        or not isinstance(documents, list)
        or len(documents) != DOCUMENT_COUNT
        or any(
            not isinstance(row, Mapping)
            or set(row) != DOCUMENT_KEYS
            or type(row.get("doc_id")) is not int
            or not isinstance(row.get("text"), str)
            or not row["text"]
            or type(row.get("bm25_score_int")) is not int
            for row in documents
        )
        or [row["doc_id"] for row in documents] != list(range(DOCUMENT_COUNT))
        or len({normalize_text(row["text"]) for row in documents}) != DOCUMENT_COUNT
        or not isinstance(ranking, list)
        or len(ranking) != RAW_COUNT
        or any(type(index) is not int for index in ranking)
        or len(set(ranking)) != RAW_COUNT
        or any(not 0 <= index < DOCUMENT_COUNT for index in ranking)
        or ranking
        != [
            row["doc_id"]
            for row in sorted(
                documents,
                key=lambda row: (-row["bm25_score_int"], row["doc_id"]),
            )[:RAW_COUNT]
        ]
    ):
        raise QASCAcquisitionError("gold-free view payload drifted")


def load_private_views(
    *, view_path: Path, commitment: BlockCommitment, expected_block: str
) -> tuple[dict[str, Any], ...]:
    """Load gold-free views without statting or opening any label path."""

    _validate_commitment_identity(commitment, expected_block=expected_block)
    views = _read_canonical_jsonl(
        view_path, digest=commitment.view_file_sha256, field="view"
    )
    if len(views) != commitment.count:
        raise QASCAcquisitionError("private view row count drifted")
    for view in views:
        _validate_view_only(view, expected_block=expected_block)
    if (
        len({stable_hash(view) for view in views}) != len(views)
        or stable_hash([stable_hash(row) for row in views])
        != commitment.view_commitment_set_sha256
    ):
        raise QASCAcquisitionError("private view commitment drifted")
    return views


def load_private_labels(
    *, label_path: Path, commitment: BlockCommitment, expected_block: str
) -> tuple[dict[str, Any], ...]:
    """Load labels only after the runner's explicit label authorization."""

    _validate_commitment_identity(commitment, expected_block=expected_block)
    labels = _read_canonical_jsonl(
        label_path, digest=commitment.label_file_sha256, field="label"
    )
    if len(labels) != commitment.count:
        raise QASCAcquisitionError("private label row count drifted")
    identities: set[str] = set()
    view_hashes: set[str] = set()
    for label in labels:
        if (
            set(label) != LABEL_KEYS
            or label.get("schema") != PRIVATE_LABEL_SCHEMA
            or label.get("block") != expected_block
            or label.get("source_member") != BLOCK_SOURCE_MEMBERS[expected_block]
        ):
            raise QASCAcquisitionError("private label identity drifted")
        identities.add(_require_sha256(label.get("identity_commitment_sha256"), "identity"))
        view_hashes.add(_require_sha256(label.get("view_sha256"), "view hash"))
    if (
        len(identities) != len(labels)
        or len(view_hashes) != len(labels)
        or stable_hash([stable_hash(row) for row in labels])
        != commitment.label_commitment_set_sha256
    ):
        raise QASCAcquisitionError("private label commitment drifted")
    return labels


def join_private_block(
    *,
    views: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    commitment: BlockCommitment,
    expected_block: str,
) -> tuple[tuple[Mapping[str, Any], Mapping[str, Any]], ...]:
    """Join already-loaded views and authorized labels by ``view_sha256``."""

    _validate_commitment_identity(commitment, expected_block=expected_block)
    label_map = {row.get("view_sha256"): row for row in labels}
    if len(label_map) != len(labels):
        raise QASCAcquisitionError("private label view hashes overlap")
    joined: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for view in views:
        label = label_map.get(stable_hash(view))
        if label is None:
            raise QASCAcquisitionError("private view has no authorized label envelope")
        _validate_private_pair(view, label, expected_block=expected_block)
        joined.append((view, label))
    if (
        len(joined) != commitment.count
        or len(labels) != commitment.count
        or _joined_commitment(views, labels)
        != commitment.joined_commitment_set_sha256
    ):
        raise QASCAcquisitionError("private joined commitment drifted")
    return tuple(joined)


def _persistence_canary(directory: Path) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise QASCAcquisitionError("persistence directory is unsafe")
    target = directory / f".{VERSION}.{os.urandom(12).hex()}.canary"
    expected = b"qasc-acquisition-persistence-canary\n"
    try:
        _atomic_write_exclusive(target, expected, mode=0o600)
        if target.read_bytes() != expected or stat.S_IMODE(target.stat().st_mode) & 0o077:
            raise QASCAcquisitionError("persistence canary verification failed")
    finally:
        target.unlink(missing_ok=True)
        _fsync_directory(directory)


def _preflight_persistence(
    *,
    pack_root: Path,
    locator: Path,
    marker: Path,
    failure: Path,
    public_receipt: Path,
) -> None:
    if marker.exists():
        raise FileExistsError("QASC acquisition authorization was already consumed")
    if pack_root.exists() or locator.exists() or failure.exists() or public_receipt.exists():
        raise FileExistsError("QASC acquisition output already exists")
    paths = (pack_root, locator, marker, failure, public_receipt)
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise QASCAcquisitionError("private and public outputs overlap")
    directories = {
        pack_root.parent,
        locator.parent,
        marker.parent,
        failure.parent,
        public_receipt.parent,
    }
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if directory.is_symlink() or not directory.is_dir():
            raise QASCAcquisitionError("output parent is unsafe")
    created = False
    try:
        os.mkdir(pack_root, 0o700)
        created = True
        os.chmod(pack_root, 0o700)
        _fsync_directory(pack_root.parent)
        for directory in {*directories, pack_root}:
            _persistence_canary(directory)
            if shutil.disk_usage(directory).free < _MIN_FREE_BYTES:
                raise QASCAcquisitionError("insufficient acquisition free space")
    except BaseException:
        if created:
            try:
                pack_root.rmdir()
                _fsync_directory(pack_root.parent)
            except OSError:
                pass
        raise


def _head_binding(*, project: Path, path: Path, field: str) -> dict[str, Any]:
    repository = Path(_git(project, "rev-parse", "--show-toplevel").decode().strip())
    actual = path.resolve(strict=True)
    relative = actual.relative_to(repository).as_posix()
    live = actual.read_bytes()
    if live != _git(repository, "show", f"HEAD:{relative}"):
        raise QASCAcquisitionError(f"{field} is not the HEAD blob")
    if _git(repository, "status", "--porcelain", "--", relative):
        raise QASCAcquisitionError(f"{field} is dirty")
    digest = _sha256_bytes(live)
    commit = _git(repository, "log", "-1", "--format=%H", "--", relative).decode().strip()
    if _SHA1_RE.fullmatch(commit) is None:
        raise QASCAcquisitionError(f"{field} commit must be a lowercase SHA-1")
    return {
        "relative_path": actual.relative_to(project.resolve(strict=True)).as_posix(),
        "file_sha256": digest,
        "head_blob_sha256": digest,
        "commit": commit,
        "clean_tracked_HEAD_blob": True,
    }


def _assert_public_safe(payload: Mapping[str, Any]) -> None:
    forbidden_keys = {
        "formatted_question",
        "choices",
        "documents",
        "raw_ranking",
        "answerKey",
        "gold_document_ids",
        "fact1_document_id",
        "fact2_document_id",
        "identity_commitment_sha256",
        "item_id",
        "private_root",
    }

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            if forbidden_keys & set(value):
                raise QASCAcquisitionError("public artifact contains private fields")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)


def _record_post_marker_failure(
    *, failure_path: Path, marker_raw: bytes, stage: str, error: BaseException
) -> None:
    payload = {
        "schema": f"{VERSION}_post_marker_failure",
        "stage": stage,
        "exception_type": f"{type(error).__module__}.{type(error).__qualname__}",
        "exception_message_sha256": _sha256_bytes(str(error).encode("utf-8")),
        "consumption_file_sha256": _sha256_bytes(marker_raw),
        "consumption_sha256": json.loads(marker_raw)["consumption_sha256"],
        "item_content_or_ID_persisted": False,
        "retry_replay_resample_authorized": False,
        "study_burned": True,
    }
    try:
        _write_json_exclusive(
            failure_path, payload, hash_field="failure_sha256", mode=0o600
        )
    except BaseException:
        # The durable marker remains the authoritative burn receipt even if a
        # secondary failure prevents this diagnostic receipt.
        pass


def acquire_private_blocks(
    *,
    project: Path,
    preregistration_path: Path,
    selection_secret_path: Path,
    dataset_archive_path: Path,
    corpus_archive_path: Path,
    extracted_corpus_path: Path,
    private_root: Path,
    private_locator_path: Path,
    public_receipt_path: Path,
    corpus_workers: int = DEFAULT_CORPUS_WORKERS,
) -> dict[str, Any]:
    """Consume authorization and form all four blocks exactly once."""

    if _CLEAN_MODULE_CLI_ACTIVE is not True:
        raise QASCAcquisitionError(
            "formal acquisition is available only through clean CLI"
        )
    root = project.resolve(strict=True)
    if corpus_workers != DEFAULT_CORPUS_WORKERS:
        raise QASCAcquisitionError(
            f"formal corpus worker count must be {DEFAULT_CORPUS_WORKERS}"
        )
    preregistration = verify_preregistration(
        path=preregistration_path,
        project=root,
        selection_secret_path=selection_secret_path,
    )
    prereg_path = root / PREREGISTRATION_RELATIVE
    prereg_custody = _head_binding(
        project=root, path=prereg_path, field="QASC acquisition preregistration"
    )
    _secret_path, secret = _canonical_selection_secret(root, selection_secret_path)
    dataset_archive = _canonical_private_path(
        project=root,
        supplied=dataset_archive_path,
        relative=DATASET_ARCHIVE_RELATIVE,
        require_file=True,
        field="dataset archive",
    )
    corpus_archive = _canonical_private_path(
        project=root,
        supplied=corpus_archive_path,
        relative=CORPUS_ARCHIVE_RELATIVE,
        require_file=True,
        field="corpus archive",
    )
    extracted_corpus = _canonical_private_path(
        project=root,
        supplied=extracted_corpus_path,
        relative=EXTRACTED_CORPUS_RELATIVE,
        require_file=None,
        field="extracted corpus",
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
    failure = _canonical_private_path(
        project=root,
        supplied=root / FAILURE_RELATIVE,
        relative=FAILURE_RELATIVE,
        require_file=None,
        field="post-marker failure receipt",
    )
    public_receipt = (root / ACQUISITION_RELATIVE).absolute()
    supplied_public = (
        public_receipt_path
        if public_receipt_path.is_absolute()
        else root / public_receipt_path
    ).absolute()
    if supplied_public != public_receipt:
        raise QASCAcquisitionError("public receipt must use its canonical path")

    dataset_archive_preflight_sha256 = _sha256_file(dataset_archive)
    if dataset_archive_preflight_sha256 != DATASET_ARCHIVE_SHA256:
        raise QASCAcquisitionError("dataset archive pre-marker hash drifted")
    bm25_batch_diagnostic = run_synthetic_bm25_batch_diagnostic()
    # This is the only source open allowed before authorization: it contains
    # unlabeled facts and no QA rows.
    corpus_preflight = prepare_unlabeled_corpus(
        corpus_archive=corpus_archive, extracted_corpus=extracted_corpus
    )
    _preflight_persistence(
        pack_root=pack_root,
        locator=locator,
        marker=marker,
        failure=failure,
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
        "source_access_addendum_file_sha256": protocol["source_access_addendum"][
            "file_sha256"
        ],
        "nli_asset_file_sha256": protocol["nli_asset"]["file_sha256"],
        "infrastructure_diagnostic_file_sha256": protocol[
            "infrastructure_diagnostic"
        ]["file_sha256"],
        "infrastructure_diagnostic_sha256": protocol[
            "infrastructure_diagnostic"
        ]["diagnostic_sha256"],
        "selection_secret_commitment_sha256": _sha256_bytes(secret),
        "dataset_archive_expected_sha256": DATASET_ARCHIVE_SHA256,
        "dataset_archive_preflight_sha256": dataset_archive_preflight_sha256,
        "dataset_archive_byte_hash_verified_before_consumption": True,
        "corpus_preflight": corpus_preflight,
        "bm25_batch_diagnostic": bm25_batch_diagnostic,
        "private_pack_path_sha256": stable_hash(
            {"absolute_private_pack": str(pack_root)}
        ),
        "private_locator_path_sha256": stable_hash(
            {"absolute_private_locator": str(locator)}
        ),
        "public_receipt_path_sha256": stable_hash(
            {"absolute_public_receipt": str(public_receipt)}
        ),
        "persistence_preflight_complete": True,
        "unlabeled_corpus_ready_before_consumption": True,
        "TRAIN_or_DEV_opened_before_consumption": False,
        "TEST_reopened": False,
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

    # Every operation below is post-marker.  Any failure leaves the durable
    # marker and permanently burns this cohort.
    def guarded(stage: str, operation: Any) -> Any:
        try:
            return operation()
        except BaseException as exc:
            _record_post_marker_failure(
                failure_path=failure, marker_raw=marker_raw, stage=stage, error=exc
            )
            raise

    candidates = guarded(
        "load_and_validate_TRAIN_DEV",
        lambda: load_formal_candidates(dataset_archive),
    )
    selected, selection_stats = guarded(
        "select_all_four_blocks",
        lambda: _select_candidates(
            candidates, secret=secret, enforce_formal_counts=True
        ),
    )
    distractors, corpus_statistics, mining_stats = guarded(
        "two_pass_full_corpus_BM25",
        lambda: mine_distractors(
            corpus_path=extracted_corpus,
            selected=selected,
            workers=corpus_workers,
        ),
    )
    views, labels = guarded(
        "construct_gold_free_views_and_label_envelopes",
        lambda: build_private_blocks(
            selected=selected,
            distractors=distractors,
            statistics=corpus_statistics,
            secret=secret,
        ),
    )

    commitments: list[BlockCommitment] = []
    for block in BLOCK_ORDER:
        view_hash, view_set_hash = guarded(
            f"persist_{block}_views",
            lambda block=block: _write_jsonl_exclusive(
                pack_root / f"{block}.views.jsonl", views[block]
            ),
        )
        label_hash, label_set_hash = guarded(
            f"persist_{block}_labels",
            lambda block=block: _write_jsonl_exclusive(
                pack_root / f"{block}.labels.jsonl", labels[block]
            ),
        )
        commitments.append(
            BlockCommitment(
                block=block,
                source_member=BLOCK_SOURCE_MEMBERS[block],
                count=BLOCK_COUNTS[block],
                view_file_sha256=view_hash,
                label_file_sha256=label_hash,
                view_commitment_set_sha256=view_set_hash,
                label_commitment_set_sha256=label_set_hash,
                joined_commitment_set_sha256=_joined_commitment(
                    views[block], labels[block]
                ),
            )
        )
    locator_body = {
        "schema": PRIVATE_LOCATOR_SCHEMA,
        "private_root": str(pack_root),
        "blocks": [
            {
                **commitment.to_dict(),
                "view_relative_file": f"{commitment.block}.views.jsonl",
                "label_relative_file": f"{commitment.block}.labels.jsonl",
            }
            for commitment in commitments
        ],
        "private_pack_sha256": stable_hash(
            [commitment.to_dict() for commitment in commitments]
        ),
        "selection_secret_included": False,
    }
    guarded(
        "persist_private_locator",
        lambda: _write_json_exclusive(
            locator, locator_body, hash_field="locator_sha256", mode=0o600
        ),
    )

    receipt: dict[str, Any] = {
        "schema": ACQUISITION_SCHEMA,
        "decision": "one_shot_four_block_private_pack_formed_no_measurement_authority",
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "preregistration_custody": prereg_custody,
        "public_protocol_bindings": preregistration["public_protocol_bindings"],
        "implementation": preregistration["implementation"],
        "selection_runtime": preregistration["selection_runtime"],
        "source": {
            "dataset_archive_sha256": DATASET_ARCHIVE_SHA256,
            "member_sha256s": dict(SOURCE_MEMBER_SHA256S),
            "member_row_counts": dict(SOURCE_MEMBER_ROW_COUNTS),
            "corpus_archive_sha256": CORPUS_ARCHIVE_SHA256,
            "corpus_member_sha256": CORPUS_MEMBER_SHA256,
            "corpus_member_size": CORPUS_MEMBER_SIZE,
            "TEST_reopened": False,
        },
        "selection": {
            "method": "post_exposure_private_HMAC_rank_greedy_frozen_constraints",
            "domain_separator": SELECTION_DOMAIN_SEPARATOR,
            "selection_secret_commitment_sha256": _sha256_bytes(secret),
            "block_counts": dict(BLOCK_COUNTS),
            "block_source_members": dict(BLOCK_SOURCE_MEMBERS),
            "selected_count": SELECTED_COUNT,
            **selection_stats,
        },
        "distractor_mining": mining_stats,
        "pre_marker_bm25_batch_diagnostic": bm25_batch_diagnostic,
        "commitments": {
            "block_files": [commitment.to_dict() for commitment in commitments],
            "private_pack_sha256": stable_hash(
                [commitment.to_dict() for commitment in commitments]
            ),
            "private_locator_file_sha256": _sha256_file(locator),
            "view_key_set_sha256": stable_hash(sorted(VIEW_KEYS)),
            "label_key_set_sha256": stable_hash(sorted(LABEL_KEYS)),
            "view_and_label_files_are_separate": True,
            "item_IDs_or_content_persisted_publicly": False,
            "private_paths_persisted_publicly": False,
        },
        "prospective_ordering": {
            "preregistration_committed_before_consumption": True,
            "dataset_archive_byte_hash_verified_before_consumption": True,
            "unlabeled_corpus_verified_and_extracted_before_consumption": True,
            "persistence_preflight_complete_before_consumption": True,
            "pack_root_created_before_consumption": True,
            "consumption_persisted_before_TRAIN_or_DEV_open": True,
            "TEST_reopened": False,
            "consumption_file_sha256": _sha256_bytes(marker_raw),
            "consumption_sha256": json.loads(marker_raw)["consumption_sha256"],
            "retry_replay_resample_authorized": False,
        },
        "safety": {
            "formation_executed": False,
            "measurement_executed": False,
            "BM25_retrieval_metadata_computed": True,
            "performance_scores_computed": 0,
            "model_calls": 0,
            "network_calls": 0,
            "online_evaluator_calls": 0,
        },
    }
    guarded("validate_public_receipt_safety", lambda: _assert_public_safe(receipt))
    return receipt


def parse_block_commitments(
    receipt: Mapping[str, Any]
) -> tuple[BlockCommitment, ...]:
    rows = receipt.get("commitments", {}).get("block_files")
    if not isinstance(rows, list) or len(rows) != len(BLOCK_ORDER):
        raise QASCAcquisitionError("public block commitments are malformed")
    parsed: list[BlockCommitment] = []
    for expected_block, row in zip(BLOCK_ORDER, rows, strict=True):
        if not isinstance(row, Mapping):
            raise QASCAcquisitionError("public block commitment is malformed")
        try:
            commitment = BlockCommitment(**row)
        except TypeError as exc:
            raise QASCAcquisitionError("public block commitment keys drifted") from exc
        _validate_commitment_identity(commitment, expected_block=expected_block)
        for field in (
            "view_file_sha256",
            "label_file_sha256",
            "view_commitment_set_sha256",
            "label_commitment_set_sha256",
            "joined_commitment_set_sha256",
        ):
            _require_sha256(getattr(commitment, field), field)
        parsed.append(commitment)
    return tuple(parsed)


def load_acquisition_binding_live(
    *, project: Path, receipt_path: Path, selection_secret_path: Path
) -> tuple[dict[str, Any], tuple[BlockCommitment, ...]]:
    """Verify the committed public acquisition without touching private blocks."""

    root = project.resolve(strict=True)
    canonical = _canonical_public_path(
        project=root,
        supplied=receipt_path,
        relative=ACQUISITION_RELATIVE,
        field="acquisition receipt",
    )
    receipt, _raw = _read_json_object(canonical, "acquisition receipt")
    body = dict(receipt)
    declared = _require_sha256(
        body.pop("acquisition_sha256", None), "acquisition hash"
    )
    if receipt.get("schema") != ACQUISITION_SCHEMA or stable_hash(body) != declared:
        raise QASCAcquisitionError("acquisition receipt self-hash drifted")
    custody = _head_binding(
        project=root, path=canonical, field="QASC acquisition receipt"
    )
    _secret_path, secret = _canonical_selection_secret(root, selection_secret_path)
    commitments = parse_block_commitments(receipt)
    commitment_rows = [row.to_dict() for row in commitments]
    if (
        receipt.get("decision")
        != "one_shot_four_block_private_pack_formed_no_measurement_authority"
        or receipt.get("public_protocol_bindings") != public_protocol_bindings(root)
        or receipt.get("implementation") != implementation_binding(root)
        or receipt.get("selection", {}).get(
            "selection_secret_commitment_sha256"
        )
        != _sha256_bytes(secret)
        or receipt.get("selection", {}).get("selected_count") != SELECTED_COUNT
        or receipt.get("source", {}).get("TEST_reopened") is not False
        or receipt.get("commitments", {}).get("private_pack_sha256")
        != stable_hash(commitment_rows)
        or receipt.get("commitments", {}).get("view_and_label_files_are_separate")
        is not True
        or receipt.get("prospective_ordering", {}).get(
            "consumption_persisted_before_TRAIN_or_DEV_open"
        )
        is not True
        or receipt.get("prospective_ordering", {}).get("TEST_reopened") is not False
        or receipt.get("prospective_ordering", {}).get(
            "retry_replay_resample_authorized"
        )
        is not False
        or receipt.get("safety", {}).get("performance_scores_computed") != 0
    ):
        raise QASCAcquisitionError("live acquisition binding drifted")
    if custody["file_sha256"] != _sha256_file(canonical):
        raise QASCAcquisitionError("acquisition committed custody drifted")
    return receipt, commitments


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preregister = commands.add_parser("preregister")
    acquire = commands.add_parser("acquire")
    for command in (preregister, acquire):
        command.add_argument("--project", type=Path, required=True)
        command.add_argument("--selection-secret", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    acquire.add_argument("--preregistration", type=Path, required=True)
    acquire.add_argument("--dataset-archive", type=Path, required=True)
    acquire.add_argument("--corpus-archive", type=Path, required=True)
    acquire.add_argument("--extracted-corpus", type=Path, required=True)
    acquire.add_argument("--private-root", type=Path, required=True)
    acquire.add_argument("--private-locator", type=Path, required=True)
    acquire.add_argument(
        "--corpus-workers", type=int, default=DEFAULT_CORPUS_WORKERS
    )
    arguments = parser.parse_args(argv)
    root = arguments.project.resolve(strict=True)
    expected_output = root / (
        PREREGISTRATION_RELATIVE
        if arguments.command == "preregister"
        else ACQUISITION_RELATIVE
    )
    if arguments.output.resolve(strict=False) != expected_output.resolve(strict=False):
        raise QASCAcquisitionError("production CLI output must be canonical")
    if arguments.output.exists():
        raise FileExistsError("public output already exists")
    if arguments.command == "preregister":
        payload = build_preregistration(
            project=root, selection_secret_path=arguments.selection_secret
        )
        _write_json_exclusive(
            arguments.output,
            payload,
            hash_field="preregistration_sha256",
            mode=0o644,
        )
        return 0
    global _CLEAN_MODULE_CLI_ACTIVE
    _CLEAN_MODULE_CLI_ACTIVE = True
    try:
        payload = acquire_private_blocks(
            project=root,
            preregistration_path=arguments.preregistration,
            selection_secret_path=arguments.selection_secret,
            dataset_archive_path=arguments.dataset_archive,
            corpus_archive_path=arguments.corpus_archive,
            extracted_corpus_path=arguments.extracted_corpus,
            private_root=arguments.private_root,
            private_locator_path=arguments.private_locator,
            public_receipt_path=arguments.output,
            corpus_workers=arguments.corpus_workers,
        )
        try:
            _write_json_exclusive(
                arguments.output,
                payload,
                hash_field="acquisition_sha256",
                mode=0o644,
            )
        except BaseException as exc:
            marker = root / CONSUMPTION_RELATIVE
            failure = root / FAILURE_RELATIVE
            if marker.is_file():
                _record_post_marker_failure(
                    failure_path=failure,
                    marker_raw=marker.read_bytes(),
                    stage="persist_public_acquisition_receipt",
                    error=exc,
                )
            raise
    finally:
        _CLEAN_MODULE_CLI_ACTIVE = False
    return 0


__all__ = [
    "ACQUISITION_SCHEMA",
    "BLOCK_COUNTS",
    "BLOCK_ORDER",
    "BLOCK_SOURCE_MEMBERS",
    "BlockCommitment",
    "LABEL_KEYS",
    "PRIVATE_LABEL_SCHEMA",
    "PRIVATE_VIEW_SCHEMA",
    "QASCAcquisitionError",
    "SELECTED_COUNT",
    "VIEW_KEYS",
    "acquire_private_blocks",
    "build_preregistration",
    "join_private_block",
    "load_acquisition_binding_live",
    "load_private_labels",
    "load_private_views",
    "load_selection_secret",
    "parse_block_commitments",
    "public_protocol_bindings",
    "verify_preregistration",
]


if __name__ == "__main__":
    raise SystemExit(main())
