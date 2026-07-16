"""Preregister and acquire a fresh HotpotQA family-out retrieval cohort.

The canonical measurement source is an immutable Hugging Face-hosted Parquet
conversion of the HotpotQA distractor validation split.  The protocol does not
claim byte or row equivalence with the original CMU JSON.  Preregistration
opens no dataset row.  Acquisition then performs one mechanical eligibility
scan and one private-HMAC selection, writing questions, documents, labels, and
IDs only to a git-ignored private pack.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import hmac
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
from pathlib import Path
import platform
import re
import subprocess
from typing import Any, Mapping, Sequence

from ..models import stable_hash
from .musique_official_core_comparison_v1 import (
    SELECTION_SECRET_BYTES,
    _assert_git_ignored_private_path,
    _read_selection_secret,
    _selection_secret_commitment,
    generate_selection_secret,
)
from .musique_recursive_study_blocks_v1 import load_study_frozen_program


VERSION = "hotpot_family_out_retrieval_v1"
PREREGISTRATION_SCHEMA = f"{VERSION}_preregistration"
ACQUISITION_SCHEMA = f"{VERSION}_acquisition"
PRIVATE_ROW_SCHEMA = f"{VERSION}_private_row"
PRIVATE_LOCATOR_SCHEMA = f"{VERSION}_private_locator"
SOURCE_URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/"
    "14f0ace3c3fac7bd86149c616b5b05d8282e5c6a/"
    "distractor/validation/0000.parquet"
)
HF_REPOSITORY = "hotpotqa/hotpot_qa"
HF_REPOSITORY_COMMIT = "14f0ace3c3fac7bd86149c616b5b05d8282e5c6a"
SOURCE_SHA256 = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"
SOURCE_SIZE = 27_452_575
SOURCE_ROW_COUNT = 7_405
OFFICIAL_REPOSITORY = "https://github.com/hotpotqa/hotpot.git"
OFFICIAL_REPOSITORY_COMMIT = "3635853403a8735609ee997664e1528f4480762a"
OFFICIAL_README_SHA256 = (
    "2a0a1758bb9a2e52b52e6a6528bbf72d46a5a01e0318cdea0473d0660d555998"
)
ORIGINAL_DECLARED_URL = (
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
)
SAMPLE_COUNT = 12
TOP_K = 5
ACQUISITION_CONSUMPTION_RELATIVE = (
    "artifacts/hotpotqa_family_out_v1/acquisition.authorization.consumed.json"
)
BWRAP_SHA256 = "d78807229d616606e339c5988392b9e0ab4a6a6998fa51e4590837f426a12fca"
BWRAP_VERSION = "bubblewrap 0.6.1"
BWRAP_PROBE_TEMPLATE_ARGS = (
    "--unshare-net",
    "--die-with-parent",
    "--new-session",
    "--ro-bind",
    "/",
    "/",
    "--dev",
    "/dev",
    "--bind",
    "<writable_root>",
    "<writable_root>",
    "/bin/true",
)
EXPECTED_SOURCE_FIELDS = (
    "id",
    "question",
    "answer",
    "type",
    "level",
    "supporting_facts",
    "context",
)
IMPLEMENTATION_RELATIVE_FILES = (
    "assumption_agent/benchmarks/hotpot_family_out_acquisition_v1.py",
    "assumption_agent/benchmarks/hotpot_family_out_runner_v1.py",
    "assumption_agent/benchmarks/musique_formal_runtime_binding_v2.py",
    "assumption_agent/benchmarks/musique_official_core_comparison_v1.py",
    "assumption_agent/benchmarks/musique_recursive_study_acquisition_v1.py",
    "assumption_agent/benchmarks/musique_recursive_study_blocks_v1.py",
    "assumption_agent/benchmarks/musique_typed_retriever_formation_v1.py",
    "assumption_agent/models.py",
    "replication_runtime/musique_official_hipporag_v1/adapter.py",
    "replication_runtime/musique_official_hipporag_v1/adapter_v2.py",
    "replication_runtime/musique_official_hipporag_v1/binding.py",
    "replication_runtime/musique_official_hipporag_v1/contract.py",
    "replication_runtime/musique_official_hipporag_v1/runtime_attestation_v2.py",
    "replication_runtime/musique_official_hipporag_v1/worker.py",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class HotpotFamilyOutAcquisitionError(RuntimeError):
    """The prospective family-out source or custody contract drifted."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise HotpotFamilyOutAcquisitionError("required file is unavailable")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise HotpotFamilyOutAcquisitionError(
            f"{field_name} must be lowercase SHA-256"
        )
    return value


def _write_json_exclusive(
    path: Path, payload: Mapping[str, Any], *, hash_field: str, mode: int
) -> None:
    body = dict(payload)
    body.pop(hash_field, None)
    body[hash_field] = stable_hash(body)
    raw = json.dumps(body, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "wb") as handle:
        for row in rows:
            raw = _canonical_bytes(row) + b"\n"
            handle.write(raw)
            digest.update(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return digest.hexdigest()


def _assert_public_safe(payload: Mapping[str, Any]) -> None:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    forbidden = (
        '"answer"',
        '"context"',
        '"corpus"',
        '"item_id"',
        '"paragraph_text"',
        '"private_pack_path"',
        '"question"',
        '"selection_secret_path"',
        '"support_indices"',
    )
    if any(value in raw for value in forbidden):
        raise HotpotFamilyOutAcquisitionError(
            "public artifact contains private content or a locator"
        )


def _git(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise HotpotFamilyOutAcquisitionError("official repository check failed")
    return completed.stdout


def official_repository_receipt(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    commit = _git(repository, "rev-parse", "HEAD^{commit}").decode().strip()
    remote = _git(repository, "remote", "get-url", "origin").decode().strip()
    if (
        commit != OFFICIAL_REPOSITORY_COMMIT
        or remote.rstrip("/") != OFFICIAL_REPOSITORY.rstrip("/")
        or _git(repository, "status", "--porcelain", "--untracked-files=no").strip()
    ):
        raise HotpotFamilyOutAcquisitionError("official repository identity drifted")
    readme = repository / "README.md"
    if _sha256_file(readme) != OFFICIAL_README_SHA256:
        raise HotpotFamilyOutAcquisitionError("official README drifted")
    text = readme.read_text(encoding="utf-8")
    if ORIGINAL_DECLARED_URL not in text:
        raise HotpotFamilyOutAcquisitionError(
            "official README no longer declares distractor dev source"
        )
    return {
        "repository": OFFICIAL_REPOSITORY,
        "commit": commit,
        "readme_sha256": OFFICIAL_README_SHA256,
        "declared_original_url": ORIGINAL_DECLARED_URL,
        "role": "schema_and_original_source_declaration_only",
    }


def implementation_binding(project: Path) -> dict[str, Any]:
    project = project.resolve(strict=True)
    rows: list[dict[str, str]] = []
    for relative in IMPLEMENTATION_RELATIVE_FILES:
        path = project / relative
        if path.is_symlink() or not path.is_file():
            raise HotpotFamilyOutAcquisitionError(
                f"implementation file missing: {relative}"
            )
        rows.append({"path": relative, "sha256": _sha256_file(path)})
    return {"files": rows, "set_sha256": stable_hash(rows)}


def _load_self_hashed(path: Path, schema: str, hash_field: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise HotpotFamilyOutAcquisitionError("public receipt is unavailable")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HotpotFamilyOutAcquisitionError("public receipt is invalid") from exc
    if not isinstance(payload, dict):
        raise HotpotFamilyOutAcquisitionError("public receipt must be an object")
    body = dict(payload)
    declared = body.pop(hash_field, None)
    if payload.get("schema") != schema or stable_hash(body) != declared:
        raise HotpotFamilyOutAcquisitionError("public receipt self-hash drifted")
    return payload, raw


def capability_binding(path: Path) -> dict[str, str]:
    payload, raw = _load_self_hashed(
        path,
        f"{VERSION}_bubblewrap_capability",
        "receipt_sha256",
    )
    expected_keys = {
        "benchmark_rows_read",
        "bwrap_file_sha256",
        "bwrap_version",
        "decision",
        "execution_context_requirement",
        "model_calls",
        "network_connectivity_attempted",
        "performance_gate",
        "probe_contract_sha256",
        "probe_returncode",
        "probe_stderr_sha256",
        "probe_stdout_sha256",
        "receipt_sha256",
        "schema",
    }
    if (
        set(payload) != expected_keys
        or payload.get("decision") != "bubblewrap_network_namespace_available"
        or payload.get("probe_returncode") != 0
        or payload.get("benchmark_rows_read") != 0
        or payload.get("model_calls") != 0
        or payload.get("network_connectivity_attempted") is not False
        or payload.get("performance_gate") is not False
        or payload.get("bwrap_file_sha256") != BWRAP_SHA256
        or payload.get("bwrap_version") != BWRAP_VERSION
        or payload.get("probe_contract_sha256")
        != stable_hash({"argv_without_binary": list(BWRAP_PROBE_TEMPLATE_ARGS)})
    ):
        raise HotpotFamilyOutAcquisitionError("bubblewrap capability did not pass")
    for key in (
        "bwrap_file_sha256",
        "probe_contract_sha256",
        "probe_stderr_sha256",
        "probe_stdout_sha256",
        "receipt_sha256",
    ):
        _require_sha256(payload.get(key), key)
    return {
        "file_sha256": _sha256_bytes(raw),
        "receipt_sha256": payload["receipt_sha256"],
        "bwrap_file_sha256": payload["bwrap_file_sha256"],
        "probe_contract_sha256": payload["probe_contract_sha256"],
    }


def acquisition_runtime_binding() -> dict[str, str]:
    try:
        pyarrow_version = package_version("pyarrow")
    except PackageNotFoundError as exc:
        raise HotpotFamilyOutAcquisitionError("pyarrow is unavailable") from exc
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "pyarrow_version": pyarrow_version,
    }


def committed_public_file_receipt(*, project: Path, path: Path) -> dict[str, str]:
    project = project.resolve(strict=True)
    raw_candidate = path.absolute()
    for component in (*reversed(raw_candidate.parents), raw_candidate):
        if component.is_symlink():
            raise HotpotFamilyOutAcquisitionError(
                "preregistration may not traverse a symbolic link"
            )
    candidate = raw_candidate.resolve(strict=True)
    try:
        relative = candidate.relative_to(project)
    except ValueError as exc:
        raise HotpotFamilyOutAcquisitionError(
            "preregistration must be inside the project"
        ) from exc
    if not candidate.is_file():
        raise HotpotFamilyOutAcquisitionError("preregistration is unavailable")
    git_top = Path(
        _git(project, "rev-parse", "--show-toplevel").decode().strip()
    ).resolve(strict=True)
    try:
        repository_relative_text = candidate.relative_to(git_top).as_posix()
    except ValueError as exc:
        raise HotpotFamilyOutAcquisitionError(
            "preregistration is outside the containing Git repository"
        ) from exc
    tracked = subprocess.run(
        [
            "git",
            "-C",
            str(git_top),
            "ls-files",
            "--error-unmatch",
            repository_relative_text,
        ],
        check=False,
        capture_output=True,
    )
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(git_top),
            "diff",
            "--quiet",
            "HEAD",
            "--",
            repository_relative_text,
        ],
        check=False,
        capture_output=True,
    )
    if tracked.returncode != 0 or dirty.returncode != 0:
        raise HotpotFamilyOutAcquisitionError(
            "preregistration must be a clean tracked HEAD blob"
        )
    head_raw = _git(git_top, "show", f"HEAD:{repository_relative_text}")
    live_raw = candidate.read_bytes()
    if head_raw != live_raw:
        raise HotpotFamilyOutAcquisitionError(
            "preregistration differs from its committed HEAD blob"
        )
    commit = _git(git_top, "rev-parse", "HEAD^{commit}").decode().strip()
    if _GIT_COMMIT_RE.fullmatch(commit) is None:
        raise HotpotFamilyOutAcquisitionError("project commit identity is invalid")
    return {
        "repository_commit": commit,
        "preregistration_file_sha256": _sha256_bytes(live_raw),
        "preregistration_head_blob_sha256": _sha256_bytes(head_raw),
    }


def p_program_binding(
    *, project: Path, formation_receipt_path: Path, frozen_program_path: Path
) -> dict[str, str]:
    program, receipt, envelope = load_study_frozen_program(
        frozen_program_path=frozen_program_path,
        formation_receipt_path=formation_receipt_path,
        verify_live=True,
        implementation_root=project,
    )
    expected = stable_hash({"block": "F1"})
    if receipt.get("formation_block_id_hash") != expected:
        raise HotpotFamilyOutAcquisitionError("P was not formed on exact F1")
    return {
        "formation_receipt_file_sha256": _sha256_file(formation_receipt_path),
        "formation_receipt_hash": receipt["receipt_hash"],
        "frozen_program_file_sha256": _sha256_file(frozen_program_path),
        "frozen_program_envelope_hash": envelope["envelope_hash"],
        "program_hash": program.program_hash,
        "formed_on_block_id_hash": expected,
    }


def build_preregistration(
    *,
    project: Path,
    official_repository: Path,
    selection_secret_path: Path,
    capability_receipt_path: Path,
    p_formation_receipt_path: Path,
    p_frozen_program_path: Path,
) -> dict[str, Any]:
    project = project.resolve(strict=True)
    secret = _read_selection_secret(project=project, path=selection_secret_path)
    payload: dict[str, Any] = {
        "schema": PREREGISTRATION_SCHEMA,
        "decision": "acquisition_only_family_out_measurement_not_authorized",
        "source": {
            "canonical_role": "immutable_HF_hosted_HotpotQA_distractor_validation_conversion",
            "url": SOURCE_URL,
            "hf_repository": HF_REPOSITORY,
            "hf_repository_commit": HF_REPOSITORY_COMMIT,
            "file_sha256": SOURCE_SHA256,
            "file_size": SOURCE_SIZE,
            "row_count": SOURCE_ROW_COUNT,
            "expected_field_count": len(EXPECTED_SOURCE_FIELDS),
            "expected_field_order_sha256": stable_hash(
                list(EXPECTED_SOURCE_FIELDS)
            ),
            "official_repository_receipt": official_repository_receipt(
                official_repository
            ),
            "parquet_equivalent_to_original_CMU_JSON_claim": False,
        },
        "eligibility": {
            "item_id_nonempty_and_globally_unique": True,
            "question_nonempty": True,
            "minimum_unique_nonempty_context_titles": TOP_K,
            "duplicate_context_titles_eligible": False,
            "context_sentences_are_nonempty_strings": True,
            "empty_context_sentence_lists_eligible": False,
            "support_title_must_exist_in_context": True,
            "support_sentence_id_integer_and_in_range": True,
            "exact_unique_gold_title_count": 2,
            "answer_type_level_text_or_score_filtering": False,
        },
        "selection": {
            "algorithm": "ascending_hmac_sha256_private_secret_and_item_id_v1",
            "domain_separator": VERSION,
            "selection_secret_commitment_sha256": _selection_secret_commitment(
                secret
            ),
            "selection_secret_persisted_publicly": False,
            "sample_count": SAMPLE_COUNT,
            "replacement": False,
            "manual_or_outcome_conditioned_selection": False,
        },
        "frozen_treatment": {
            **p_program_binding(
                project=project,
                formation_receipt_path=p_formation_receipt_path,
                frozen_program_path=p_frozen_program_path,
            ),
            "adaptation_or_reformation_on_HotpotQA": False,
        },
        "execution_capability": capability_binding(capability_receipt_path),
        "measurement": {
            "arms": ["canonical_RAW", "frozen_P", "official_HippoRAG"],
            "same_question_corpus_top_k_and_source_provided_support_labels": True,
            "top_k": TOP_K,
            "primary_metric": "offline_micro_support_recall_at_5",
            "maximum_concurrency": 3 * SAMPLE_COUNT,
            "generator_calls": 0,
            "online_evaluator_calls": 0,
            "retry_replay_resample": 0,
        },
        "implementation": implementation_binding(project),
        "acquisition_runtime": acquisition_runtime_binding(),
        "claim_boundary": {
            "new_family": "HotpotQA_distractor_validation_conversion",
            "answer_generation_claim": False,
            "original_CMU_JSON_equivalence_claim": False,
            "performance_claim_before_measurement": False,
        },
        "safety": {
            "dataset_rows_read": 0,
            "model_calls": 0,
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
    official_repository: Path,
    selection_secret_path: Path,
    capability_receipt_path: Path,
    p_formation_receipt_path: Path,
    p_frozen_program_path: Path,
) -> dict[str, Any]:
    payload, _raw = _load_self_hashed(
        path, PREREGISTRATION_SCHEMA, "preregistration_sha256"
    )
    expected = build_preregistration(
        project=project,
        official_repository=official_repository,
        selection_secret_path=selection_secret_path,
        capability_receipt_path=capability_receipt_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
    )
    if payload != expected:
        raise HotpotFamilyOutAcquisitionError(
            "preregistration differs from complete live protocol"
        )
    return payload


def _normalize_source_row(raw: object) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping) or set(raw) != set(EXPECTED_SOURCE_FIELDS):
        return None
    item_id = raw.get("id")
    question = raw.get("question")
    context = raw.get("context")
    supporting = raw.get("supporting_facts")
    if (
        not isinstance(item_id, str)
        or not item_id.strip()
        or not isinstance(question, str)
        or not question.strip()
        or not isinstance(context, Mapping)
        or not isinstance(supporting, Mapping)
    ):
        return None
    titles = context.get("title")
    sentence_rows = context.get("sentences")
    support_titles = supporting.get("title")
    support_sentence_ids = supporting.get("sent_id")
    if (
        not isinstance(titles, list)
        or not isinstance(sentence_rows, list)
        or len(titles) != len(sentence_rows)
        or len(titles) < TOP_K
        or any(not isinstance(title, str) or not title.strip() for title in titles)
        or len(set(titles)) != len(titles)
        or not isinstance(support_titles, list)
        or not isinstance(support_sentence_ids, list)
        or len(support_titles) != len(support_sentence_ids)
        or not support_titles
    ):
        return None
    documents: list[str] = []
    for sentences in sentence_rows:
        if (
            not isinstance(sentences, list)
            or not sentences
            or any(not isinstance(sentence, str) or not sentence.strip() for sentence in sentences)
        ):
            return None
        documents.append(" ".join(sentence.strip() for sentence in sentences))
    title_to_index = {title: index for index, title in enumerate(titles)}
    for title, sentence_id in zip(support_titles, support_sentence_ids):
        if (
            not isinstance(title, str)
            or title not in title_to_index
            or type(sentence_id) is not int
            or not 0 <= sentence_id < len(sentence_rows[title_to_index[title]])
        ):
            return None
    unique_support_titles = set(support_titles)
    if len(unique_support_titles) != 2:
        return None
    support_indices = tuple(
        index for index, title in enumerate(titles) if title in unique_support_titles
    )
    if len(support_indices) != 2:
        return None
    corpus = [
        {
            "idx": index,
            "title": title,
            "text": documents[index],
            "is_supporting": index in support_indices,
        }
        for index, title in enumerate(titles)
    ]
    return {
        "schema": PRIVATE_ROW_SCHEMA,
        "item_id": item_id,
        "question": question,
        "corpus": corpus,
        "support_indices": list(support_indices),
        "source_row_sha256": stable_hash(raw),
    }


def _selection_key(item_id: str, secret: bytes) -> str:
    return hmac.new(
        secret,
        f"{VERSION}:{item_id}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def acquire_private_pack(
    *,
    project: Path,
    preregistration_path: Path,
    official_repository: Path,
    selection_secret_path: Path,
    capability_receipt_path: Path,
    p_formation_receipt_path: Path,
    p_frozen_program_path: Path,
    source_parquet_path: Path,
    private_pack_path: Path,
    private_locator_path: Path,
) -> dict[str, Any]:
    project = project.resolve(strict=True)
    preregistration = verify_preregistration(
        path=preregistration_path,
        project=project,
        official_repository=official_repository,
        selection_secret_path=selection_secret_path,
        capability_receipt_path=capability_receipt_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
    )
    preregistration_custody = committed_public_file_receipt(
        project=project,
        path=preregistration_path,
    )
    source = _assert_git_ignored_private_path(
        project=project, path=source_parquet_path, require_file=True
    )
    private_pack = _assert_git_ignored_private_path(
        project=project, path=private_pack_path, require_file=False
    )
    private_locator = _assert_git_ignored_private_path(
        project=project, path=private_locator_path, require_file=False
    )
    consumption_path = _assert_git_ignored_private_path(
        project=project,
        path=project / ACQUISITION_CONSUMPTION_RELATIVE,
        require_file=None,
    )
    output_paths = (private_pack, private_locator, consumption_path)
    if len(set(output_paths)) != len(output_paths):
        raise HotpotFamilyOutAcquisitionError(
            "family-out private output paths must be distinct"
        )
    if private_pack.exists() or private_locator.exists():
        raise FileExistsError("family-out private output already exists")
    if consumption_path.exists():
        raise FileExistsError("family-out acquisition was already consumed")
    if source.stat().st_size != SOURCE_SIZE or _sha256_file(source) != SOURCE_SHA256:
        raise HotpotFamilyOutAcquisitionError("family-out source identity mismatch")
    secret = _read_selection_secret(project=project, path=selection_secret_path)
    if not hmac.compare_digest(
        _selection_secret_commitment(secret),
        preregistration["selection"]["selection_secret_commitment_sha256"],
    ):
        raise HotpotFamilyOutAcquisitionError("selection secret drifted")
    if preregistration.get("acquisition_runtime") != acquisition_runtime_binding():
        raise HotpotFamilyOutAcquisitionError("acquisition runtime drifted")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - workspace dependency
        raise HotpotFamilyOutAcquisitionError("pyarrow is unavailable") from exc
    consumption_body = {
        "schema": f"{VERSION}_acquisition_consumption",
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "source_file_sha256": SOURCE_SHA256,
        "private_pack_path_hash": stable_hash(
            {"absolute_private_pack_path": str(private_pack)}
        ),
        "private_locator_path_hash": stable_hash(
            {"absolute_private_locator_path": str(private_locator)}
        ),
        "retry_replay_resample_authorized": False,
        "source_rows_opened_before_consumption": 0,
    }
    _write_json_exclusive(
        consumption_path,
        consumption_body,
        hash_field="consumption_sha256",
        mode=0o600,
    )
    consumption_raw = consumption_path.read_bytes()
    parquet = pq.ParquetFile(source)
    if parquet.metadata.num_rows != SOURCE_ROW_COUNT:
        raise HotpotFamilyOutAcquisitionError("source row count drifted")
    if tuple(parquet.schema_arrow.names) != EXPECTED_SOURCE_FIELDS:
        raise HotpotFamilyOutAcquisitionError("source schema drifted")
    source_rows = parquet.read().to_pylist()
    normalized = [_normalize_source_row(row) for row in source_rows]
    # Global uniqueness is evaluated over every source row carrying a nonempty
    # string ID, not only over rows that pass the remaining structural rules.
    # Otherwise a malformed duplicate could silently make its valid twin
    # eligible, contrary to the preregistered global-ID contract.
    counts = Counter(
        row.get("id")
        for row in source_rows
        if isinstance(row, Mapping)
        and isinstance(row.get("id"), str)
        and row.get("id").strip()
    )
    eligible = [
        row
        for row in normalized
        if row is not None and counts[row["item_id"]] == 1
    ]
    if len(eligible) < SAMPLE_COUNT:
        raise HotpotFamilyOutAcquisitionError("insufficient eligible source rows")
    eligible.sort(
        key=lambda row: (
            _selection_key(str(row["item_id"]), secret),
            str(row["item_id"]),
        )
    )
    selected = eligible[:SAMPLE_COUNT]
    pack_hash = _write_jsonl_exclusive(private_pack, selected)
    item_commitment_set = stable_hash([stable_hash(row) for row in selected])
    locator_body = {
        "schema": PRIVATE_LOCATOR_SCHEMA,
        "private_pack_path": str(private_pack),
        "private_pack_file_sha256": pack_hash,
        "item_commitment_set_sha256": item_commitment_set,
        "item_count": SAMPLE_COUNT,
        "selection_secret_included": False,
    }
    _write_json_exclusive(
        private_locator,
        locator_body,
        hash_field="locator_sha256",
        mode=0o600,
    )
    receipt: dict[str, Any] = {
        "schema": ACQUISITION_SCHEMA,
        "decision": "fresh_family_out_pack_formed_measurement_not_authorized",
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "preregistration_custody": preregistration_custody,
        "source": {
            "file_sha256": SOURCE_SHA256,
            "file_size": SOURCE_SIZE,
            "row_count": SOURCE_ROW_COUNT,
            "hf_repository_commit": HF_REPOSITORY_COMMIT,
            "original_CMU_JSON_equivalence_claim": False,
        },
        "counts": {
            "source_rows": len(source_rows),
            "structurally_valid_rows": sum(row is not None for row in normalized),
            "eligible_unique_id_rows": len(eligible),
            "selected_rows": len(selected),
        },
        "acquisition_runtime": acquisition_runtime_binding(),
        "prospective_ordering": {
            "preregistration_committed_before_source_row_open": True,
            "acquisition_consumed_before_source_row_open": True,
            "source_rows_opened_before_consumption": 0,
            "acquisition_consumption_file_sha256": _sha256_bytes(
                consumption_raw
            ),
            "acquisition_consumption_sha256": json.loads(consumption_raw)[
                "consumption_sha256"
            ],
            "retry_replay_resample_authorized": False,
        },
        "commitments": {
            "private_pack_file_sha256": pack_hash,
            "item_commitment_set_sha256": item_commitment_set,
            "selection_secret_commitment_sha256": _selection_secret_commitment(
                secret
            ),
            "item_ids_persisted_publicly": False,
            "private_paths_persisted_publicly": False,
        },
        "safety": {
            "model_calls": 0,
            "scores_computed": 0,
            "online_evaluator_calls": 0,
            "measurement_executed": False,
        },
    }
    _assert_public_safe(receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    secret = sub.add_parser("generate-secret")
    preregister = sub.add_parser("preregister")
    acquire = sub.add_parser("acquire")
    secret.add_argument("--project", type=Path, required=True)
    secret.add_argument("--output", type=Path, required=True)
    for command in (preregister, acquire):
        command.add_argument("--project", type=Path, required=True)
        command.add_argument("--official-repository", type=Path, required=True)
        command.add_argument("--selection-secret", type=Path, required=True)
        command.add_argument("--capability-receipt", type=Path, required=True)
        command.add_argument("--p-formation-receipt", type=Path, required=True)
        command.add_argument("--p-frozen-program", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    acquire.add_argument("--preregistration", type=Path, required=True)
    acquire.add_argument("--source-parquet", type=Path, required=True)
    acquire.add_argument("--private-pack", type=Path, required=True)
    acquire.add_argument("--private-locator", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "generate-secret":
        commitment = generate_selection_secret(
            project=arguments.project, output=arguments.output
        )
        print(json.dumps({"selection_secret_commitment_sha256": commitment}))
        return 0
    common = {
        "project": arguments.project,
        "official_repository": arguments.official_repository,
        "selection_secret_path": arguments.selection_secret,
        "capability_receipt_path": arguments.capability_receipt,
        "p_formation_receipt_path": arguments.p_formation_receipt,
        "p_frozen_program_path": arguments.p_frozen_program,
    }
    if arguments.output.exists():
        raise FileExistsError("public family-out output already exists")
    if arguments.command == "preregister":
        payload = build_preregistration(**common)
        _write_json_exclusive(
            arguments.output,
            payload,
            hash_field="preregistration_sha256",
            mode=0o644,
        )
        return 0
    payload = acquire_private_pack(
        **common,
        preregistration_path=arguments.preregistration,
        source_parquet_path=arguments.source_parquet,
        private_pack_path=arguments.private_pack,
        private_locator_path=arguments.private_locator,
    )
    _write_json_exclusive(
        arguments.output,
        payload,
        hash_field="acquisition_sha256",
        mode=0o644,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
