"""Acquire and preregister a private MuSiQue retrieval-comparison pack.

Only the official MuSiQue source repository and a user-supplied official data
archive are read.  Acquisition performs no model call, answer scoring, or
online judging.  Item content is written only below a caller-selected private
root; public manifests contain source evidence, counts, and commitments.
"""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import stat
import string
import subprocess
from typing import Any, Iterable, Mapping, Sequence
import zipfile

from ..models import stable_hash


VERSION = "musique-official-core-comparison-v1"
PREREGISTRATION_SCHEMA = f"{VERSION}-preregistration"
ACQUISITION_SCHEMA = f"{VERSION}-acquisition"
PRIVATE_PACK_SCHEMA = f"{VERSION}-private-pack"
OFFICIAL_REPOSITORY = "https://github.com/StonyBrookNLP/musique.git"
OFFICIAL_SOURCE_COMMIT = "922ac98f19a201998dbdae6d7f2887a5258dbdeb"
OFFICIAL_DOWNLOAD_FILE_ID = "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h"
OFFICIAL_TRAIN_MEMBER_BASENAME = "musique_ans_v1.0_train.jsonl"
SPLIT_COUNTS = {"train": 12, "development": 6, "residual_sealed": 6}
SHARED_RETRIEVAL_TOP_K = 5
SELECTION_SECRET_BYTES = 32


class MuSiQueAcquisitionError(RuntimeError):
    """Raised when the public acquisition contract cannot be audited."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _selection_secret_commitment(secret: bytes) -> str:
    if len(secret) != SELECTION_SECRET_BYTES:
        raise MuSiQueAcquisitionError("selection secret length mismatch")
    return hashlib.sha256(b"musique-selection-secret-v1\0" + secret).hexdigest()


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MuSiQueAcquisitionError(f"{field} must be non-empty text")
    return value.strip()


# Oracle A follows the official MuSiQue/SQuAD-style implementation directly.
def normalize_answer_primary(value: str) -> str:
    lowered = value.lower()
    unpunctuated = "".join(character for character in lowered if character not in set(string.punctuation))
    without_articles = re.sub(r"\b(a|an|the)\b", " ", unpunctuated, flags=re.UNICODE)
    return " ".join(without_articles.split())


def evaluate_aliases_primary(prediction: str, aliases: Sequence[str]) -> tuple[int, Fraction]:
    predicted_tokens = normalize_answer_primary(prediction).split()
    exact_scores: list[int] = []
    f1_scores: list[Fraction] = []
    for alias in aliases:
        normalized_alias = normalize_answer_primary(alias)
        alias_tokens = normalized_alias.split()
        exact_scores.append(int(" ".join(predicted_tokens) == normalized_alias))
        common = Counter(predicted_tokens) & Counter(alias_tokens)
        overlap = sum(common.values())
        if not predicted_tokens or not alias_tokens:
            f1_scores.append(Fraction(int(predicted_tokens == alias_tokens), 1))
        elif overlap == 0:
            f1_scores.append(Fraction(0, 1))
        else:
            precision = Fraction(overlap, len(predicted_tokens))
            recall = Fraction(overlap, len(alias_tokens))
            f1_scores.append(2 * precision * recall / (precision + recall))
    if not exact_scores:
        raise MuSiQueAcquisitionError("at least one answer alias is required")
    return max(exact_scores), max(f1_scores)


# Oracle B is intentionally implemented independently: translate, tokenize,
# filter, and multiset intersection do not reuse Oracle A's helpers.
def normalize_answer_secondary(value: str) -> str:
    translated = value.lower().translate(str.maketrans("", "", string.punctuation))
    kept_tokens = [
        token
        for token in translated.split()
        if token not in frozenset(("a", "an", "the"))
    ]
    return " ".join(kept_tokens)


def evaluate_aliases_secondary(prediction: str, aliases: Sequence[str]) -> tuple[int, Fraction]:
    prediction_normalized = normalize_answer_secondary(prediction)
    prediction_tokens = prediction_normalized.split()
    best_exact = 0
    best_f1 = Fraction(0, 1)
    saw_alias = False
    for alias in aliases:
        saw_alias = True
        alias_normalized = normalize_answer_secondary(alias)
        alias_tokens = alias_normalized.split()
        best_exact = max(best_exact, int(prediction_normalized == alias_normalized))
        available: dict[str, int] = {}
        for token in alias_tokens:
            available[token] = available.get(token, 0) + 1
        overlap = 0
        for token in prediction_tokens:
            if available.get(token, 0) > 0:
                overlap += 1
                available[token] -= 1
        if not prediction_tokens or not alias_tokens:
            candidate = Fraction(int(prediction_tokens == alias_tokens), 1)
        elif overlap == 0:
            candidate = Fraction(0, 1)
        else:
            candidate = Fraction(2 * overlap, len(prediction_tokens) + len(alias_tokens))
        best_f1 = max(best_f1, candidate)
    if not saw_alias:
        raise MuSiQueAcquisitionError("at least one answer alias is required")
    return best_exact, best_f1


def evaluate_support_primary(predicted: Sequence[int], accepted: Sequence[int]) -> Fraction:
    accepted_set = set(accepted)
    if not accepted_set:
        return Fraction(1 if not predicted else 0, 1)
    return Fraction(len(set(predicted) & accepted_set), len(accepted_set))


def evaluate_support_secondary(predicted: Sequence[int], accepted: Sequence[int]) -> Fraction:
    unique_accepted = tuple(dict.fromkeys(accepted))
    if not unique_accepted:
        return Fraction(int(len(predicted) == 0), 1)
    hits = sum(any(candidate == value for candidate in predicted) for value in unique_accepted)
    return Fraction(hits, len(unique_accepted))


def _oracle_conformance_receipt() -> dict[str, Any]:
    cases = (
        ("The Alpha", ("alpha", "the alpha")),
        ("A-B", ("a b", "ab")),
        ("two two", ("two", "two two")),
        ("", ("", "none")),
        ("AN example!", ("example",)),
        ("x, y", ("x y",)),
        ("the", ("",)),
        ("One", ("one", "1")),
    )
    rows = []
    for prediction, aliases in cases:
        primary = evaluate_aliases_primary(prediction, aliases)
        secondary = evaluate_aliases_secondary(prediction, aliases)
        if primary != secondary:
            raise MuSiQueAcquisitionError("independent answer oracles disagree")
        rows.append((primary[0], primary[1].numerator, primary[1].denominator))
    support_cases = (((0, 2), (0, 1, 2)), ((), ()), ((2, 2), (1, 2)))
    support_rows = []
    for predicted, accepted in support_cases:
        primary = evaluate_support_primary(predicted, accepted)
        secondary = evaluate_support_secondary(predicted, accepted)
        if primary != secondary:
            raise MuSiQueAcquisitionError("independent support oracles disagree")
        support_rows.append((primary.numerator, primary.denominator))
    return {
        "answer_case_count": len(cases),
        "support_case_count": len(support_cases),
        "conformance_sha256": stable_hash({"answer": rows, "support": support_rows}),
        "oracle_disagreement_count": 0,
    }


def _git(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise MuSiQueAcquisitionError(f"official source git command failed: {arguments[0]}")
    return completed.stdout


def _assert_git_ignored_private_path(
    *,
    project: Path,
    path: Path,
    require_file: bool | None,
) -> Path:
    """Require a private path below this project's ignored ``artifacts/`` tree."""

    project = project.resolve()
    raw_candidate = path.absolute()
    try:
        raw_relative = raw_candidate.relative_to(project)
    except ValueError as exc:
        raise PermissionError("private path must be below the project root") from exc
    if not raw_relative.parts or raw_relative.parts[0] != "artifacts":
        raise PermissionError("private path must be below the ignored artifacts tree")
    current = project
    for component in raw_relative.parts:
        current = current / component
        if current.exists() and current.is_symlink():
            raise PermissionError("private path may not traverse a symlink")
    candidate = raw_candidate.resolve(strict=False)
    try:
        relative = candidate.relative_to(project)
    except ValueError as exc:
        raise PermissionError("private path resolves outside the project root") from exc
    if not relative.parts or relative.parts[0] != "artifacts":
        raise PermissionError("private path resolves outside the ignored artifacts tree")
    ignored = subprocess.run(
        ["git", "-C", str(project), "check-ignore", "--no-index", "-q", relative.as_posix()],
        check=False,
        capture_output=True,
        timeout=30,
    )
    if ignored.returncode != 0:
        raise PermissionError("private path is not git-ignored")
    tracked = subprocess.run(
        ["git", "-C", str(project), "ls-files", "--error-unmatch", relative.as_posix()],
        check=False,
        capture_output=True,
        timeout=30,
    )
    if tracked.returncode == 0:
        raise PermissionError("private path is tracked")
    if require_file is True and not candidate.is_file():
        raise MuSiQueAcquisitionError("required private file is missing")
    if require_file is False and candidate.exists() and not candidate.is_dir():
        raise MuSiQueAcquisitionError("private root is not a directory")
    return candidate


def generate_selection_secret(*, project: Path, output: Path) -> str:
    output = _assert_git_ignored_private_path(
        project=project,
        path=output,
        require_file=None,
    )
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    secret = secrets.token_bytes(SELECTION_SECRET_BYTES)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(secret)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(output, 0o600)
    if stat.S_IMODE(output.stat().st_mode) & 0o077:
        output.unlink()
        raise PermissionError("selection secret permissions cannot be restricted")
    return _selection_secret_commitment(secret)


def _read_selection_secret(*, project: Path, path: Path) -> bytes:
    path = _assert_git_ignored_private_path(project=project, path=path, require_file=True)
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise PermissionError("selection secret permissions are too broad")
    secret = path.read_bytes()
    if len(secret) != SELECTION_SECRET_BYTES:
        raise MuSiQueAcquisitionError("selection secret length mismatch")
    return secret


def official_source_receipt(repository: Path) -> dict[str, Any]:
    repository = repository.resolve()
    commit = _git(repository, "rev-parse", "HEAD^{commit}").decode().strip()
    if commit != OFFICIAL_SOURCE_COMMIT:
        raise MuSiQueAcquisitionError("official MuSiQue source commit mismatch")
    remote = _git(repository, "remote", "get-url", "origin").decode().strip()
    if remote.rstrip("/") != OFFICIAL_REPOSITORY.rstrip("/"):
        raise MuSiQueAcquisitionError("official MuSiQue remote mismatch")
    if _git(repository, "status", "--porcelain", "--untracked-files=no").strip():
        raise MuSiQueAcquisitionError("official MuSiQue checkout is modified")
    files = {}
    for relative in ("LICENSE", "README.md", "download_data.sh", "evaluate_v1.0.py", "metrics/answer.py"):
        path = repository / relative
        files[relative] = _sha256_file(path)
    download_script = (repository / "download_data.sh").read_text(encoding="utf-8")
    if OFFICIAL_DOWNLOAD_FILE_ID not in download_script:
        raise MuSiQueAcquisitionError("official download file ID drifted")
    readme = (repository / "README.md").read_text(encoding="utf-8")
    if "CC BY 4.0" not in readme or "MuSiQue is distributed" not in readme:
        raise MuSiQueAcquisitionError("official license declaration drifted")
    return {
        "repository": remote,
        "commit": commit,
        "tracked_checkout_clean": True,
        "file_sha256": files,
        "download": {
            "google_drive_file_id": OFFICIAL_DOWNLOAD_FILE_ID,
            "declared_archive_name": "musique_v1.0.zip",
            "official_script": "download_data.sh",
        },
        "license": {
            "spdx": "CC-BY-4.0",
            "license_file_sha256": files["LICENSE"],
            "official_repository_url": "https://github.com/StonyBrookNLP/musique",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "attribution_required": True,
        },
    }


def _implementation_binding(project: Path) -> dict[str, Any]:
    relative_files = (
        "assumption_agent/benchmarks/musique_official_core_comparison_v1.py",
        "tests/test_musique_official_core_comparison_v1.py",
    )
    rows = []
    for relative in relative_files:
        path = project / relative
        if not path.is_file():
            raise MuSiQueAcquisitionError(f"implementation file missing: {relative}")
        rows.append({"path": relative, "sha256": _sha256_file(path)})
    return {"files": rows, "set_sha256": stable_hash(rows)}


def _runtime_qualification_binding(project: Path) -> dict[str, Any]:
    relative = "manifests/official_hipporag_runtime_adapter_qualification_v1.json"
    path = project / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    declared = payload.get("qualification_sha256")
    body = dict(payload)
    body.pop("qualification_sha256", None)
    if not isinstance(declared, str) or stable_hash(body) != declared or payload.get("qualified") is not True:
        raise MuSiQueAcquisitionError("official HippoRAG runtime qualification is invalid")
    return {
        "path": relative,
        "file_sha256": _sha256_file(path),
        "qualification_sha256": declared,
        "official_commit": payload["source_binding"]["commit"],
        "claim_scope": "official_core_plus_frozen_custom_adapter_only",
    }


def build_preregistration(
    *,
    project: Path,
    official_repository: Path,
    selection_secret_path: Path,
) -> dict[str, Any]:
    source = official_source_receipt(official_repository)
    implementation = _implementation_binding(project)
    runtime = _runtime_qualification_binding(project)
    oracle = _oracle_conformance_receipt()
    selection_secret = _read_selection_secret(project=project, path=selection_secret_path)
    selection_secret_commitment = _selection_secret_commitment(selection_secret)
    payload: dict[str, Any] = {
        "schema": PREREGISTRATION_SCHEMA,
        "decision": "acquisition_authorized_model_execution_not_authorized",
        "source": source,
        "dataset_contract": {
            "dataset": "MuSiQue-Answerable v1.0",
            "claim_scope": "multi_alias_eligible_subset_of_official_train_not_full_musique",
            "source_split": "official_train_only",
            "archive_member_basename": OFFICIAL_TRAIN_MEMBER_BASENAME,
            "required_item_fields": [
                "id", "question", "answer", "answer_aliases", "answerable", "paragraphs"
            ],
            "required_paragraph_fields": [
                "idx", "title", "paragraph_text", "is_supporting"
            ],
            "eligibility": {
                "answerable": True,
                "minimum_paragraph_count": SHARED_RETRIEVAL_TOP_K,
                "minimum_supporting_paragraph_count": 2,
                "minimum_non_supporting_paragraph_count": 1,
                "minimum_distinct_normalized_answers": 2,
                "empty_normalized_answers_do_not_count": True,
                "independent_normalizers_must_agree": True,
                "paragraph_idx_namespace": "official_contiguous_zero_based_idx",
            },
        },
        "selection": {
            "algorithm": "ascending_hmac_sha256_of_private_secret_and_official_item_id_v1",
            "selection_secret_commitment_sha256": selection_secret_commitment,
            "selection_secret_persisted_publicly": False,
            "selection_secret_path_persisted_publicly": False,
            "selected_count": sum(SPLIT_COUNTS.values()),
            "split_order": list(SPLIT_COUNTS),
            "split_counts": SPLIT_COUNTS,
            "manual_item_selection": False,
            "content_conditioned_selection_after_eligibility": False,
        },
        "private_boundary": {
            "raw_archive_private": True,
            "corpus_private": True,
            "questions_private": True,
            "answers_and_aliases_private": True,
            "support_labels_private": True,
            "item_ids_persisted_publicly": False,
            "private_paths_persisted_publicly": False,
            "residual_sealed_must_be_a_separate_file": True,
        },
        "split_access": {
            "train": "formation_and_infrastructure_only",
            "development": "one_shot_after_complete_execution_freeze",
            "residual_sealed": "no_access_before_post_development_authorization",
            "preregistration_authorizes_sealed_access": False,
        },
        "oracles": {
            "answer_primary": "normalize_answer_primary_plus_evaluate_aliases_primary",
            "answer_secondary": "normalize_answer_secondary_plus_evaluate_aliases_secondary",
            "support_primary": "evaluate_support_primary",
            "support_secondary": "evaluate_support_secondary",
            "official_reference_evaluator_sha256": source["file_sha256"]["metrics/answer.py"],
            "dual_oracle_consensus_required": True,
            "synthetic_conformance": oracle,
            "online_judge": False,
        },
        "homologous_three_arm_draft": {
            "status": "draft_no_execution_authority",
            "arms": [
                {
                    "arm_id": "canonical_order_top_k_context_baseline",
                    "retriever": "official_paragraph_idx_order_first_k_no_learned_retrieval",
                },
                {
                    "arm_id": "assumption_retrieval",
                    "retriever": "train_only_typed_retrieval_treatment_to_be_frozen",
                    "current_status": "not_yet_frozen",
                },
                {
                    "arm_id": "official_hipporag_retrieval",
                    "retriever": "official_core_plus_frozen_custom_adapter_only",
                    "runtime_binding": runtime,
                },
            ],
            "shared_generator": {
                "model_family": "gpt-5.4",
                "same_provider_and_model_within_every_paired_block": True,
                "temperature": 0,
                "max_output_tokens": 256,
                "one_call_per_arm_item": True,
                "retry_replay_resample": 0,
            },
            "shared_context": {
                "retrieval_top_k": SHARED_RETRIEVAL_TOP_K,
                "maximum_context_tokens": 8192,
                "same_document_serialization": True,
                "same_prompt_template": True,
                "same_document_count": True,
                "candidate_corpus_exact_same_across_arms": True,
                "candidate_paragraph_idx_namespace_frozen": True,
                "gold_support_not_exposed_to_any_arm": True,
                "answers_aliases_and_labels_not_exposed_to_retrievers": True,
                "answers_aliases_and_labels_not_exposed_to_generator": True,
            },
            "offline_evaluation": {
                "answer_exact_match": True,
                "answer_token_f1": True,
                "support_recall_at_k": True,
                "dual_oracle_consensus_required": True,
                "online_evaluator_calls": 0,
            },
            "planned_contrasts": [
                "assumption_retrieval_minus_canonical_order_top_k_context_baseline",
                "official_hipporag_retrieval_minus_canonical_order_top_k_context_baseline",
                "assumption_retrieval_minus_official_hipporag_retrieval",
            ],
            "automatic_performance_gate": False,
            "result_policy": "paired_descriptive_estimates_with_item_level_receipts",
            "development_max_parallel_generator_calls": 18,
        },
        "implementation": implementation,
        "safety": {
            "dataset_rows_read_during_preregistration": 0,
            "model_calls": 0,
            "scores_computed": 0,
            "online_judge_calls": 0,
            "hipporag_bundled_data_accessed": False,
            "skilllearn_residual_sealed_accessed": False,
            "noaa_private_or_split_data_accessed": False,
        },
    }
    payload["preregistration_sha256"] = stable_hash(payload)
    return payload


def _verify_preregistration(path: Path, *, project: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    declared = payload.get("preregistration_sha256")
    body = dict(payload)
    body.pop("preregistration_sha256", None)
    if payload.get("schema") != PREREGISTRATION_SCHEMA or stable_hash(body) != declared:
        raise MuSiQueAcquisitionError("preregistration self-hash mismatch")
    if payload.get("implementation") != _implementation_binding(project):
        raise MuSiQueAcquisitionError("implementation drifted after preregistration")
    return payload


def _find_train_member(archive: zipfile.ZipFile) -> str:
    matches = [
        name
        for name in archive.namelist()
        if Path(name).name == OFFICIAL_TRAIN_MEMBER_BASENAME and not name.endswith("/")
    ]
    if len(matches) != 1:
        raise MuSiQueAcquisitionError("official train member is missing or ambiguous")
    return matches[0]


def _iter_source_rows(raw: bytes) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise MuSiQueAcquisitionError(f"source row {line_number} is not an object")
        yield value


def _normalize_source_row(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    if raw.get("answerable") is not True:
        return None
    item_id = _require_text(raw.get("id"), "id")
    question = _require_text(raw.get("question"), "question")
    answer = _require_text(raw.get("answer"), "answer")
    aliases_raw = raw.get("answer_aliases")
    paragraphs_raw = raw.get("paragraphs")
    if not isinstance(aliases_raw, list) or any(not isinstance(value, str) for value in aliases_raw):
        return None
    if not isinstance(paragraphs_raw, list):
        return None
    answers = [answer, *aliases_raw]
    normalized_primary = tuple(
        dict.fromkeys(
            normalized
            for value in answers
            if (normalized := normalize_answer_primary(value))
        )
    )
    normalized_secondary = tuple(
        dict.fromkeys(
            normalized
            for value in answers
            if (normalized := normalize_answer_secondary(value))
        )
    )
    if normalized_primary != normalized_secondary or len(set(normalized_primary)) < 2:
        return None
    paragraphs = []
    seen_indices: set[int] = set()
    for paragraph in paragraphs_raw:
        if not isinstance(paragraph, Mapping):
            return None
        index = paragraph.get("idx")
        supporting = paragraph.get("is_supporting")
        if type(index) is not int or index in seen_indices or type(supporting) is not bool:
            return None
        seen_indices.add(index)
        paragraphs.append(
            {
                "idx": index,
                "title": _require_text(paragraph.get("title"), "paragraph.title"),
                "text": _require_text(paragraph.get("paragraph_text"), "paragraph.text"),
                "is_supporting": supporting,
            }
        )
    paragraphs.sort(key=lambda row: row["idx"])
    if [row["idx"] for row in paragraphs] != list(range(len(paragraphs))):
        return None
    support_count = sum(row["is_supporting"] for row in paragraphs)
    if (
        len(paragraphs) < SHARED_RETRIEVAL_TOP_K
        or support_count < 2
        or len(paragraphs) - support_count < 1
    ):
        return None
    return {
        "item_id": item_id,
        "question": question,
        "corpus": paragraphs,
        "answers": answers,
        "normalized_answers": list(normalized_primary),
        "support_indices": [row["idx"] for row in paragraphs if row["is_supporting"]],
        "source_row_sha256": stable_hash(raw),
    }


def _selection_key(item_id: str, selection_secret: bytes) -> str:
    if len(selection_secret) != SELECTION_SECRET_BYTES:
        raise MuSiQueAcquisitionError("selection secret length mismatch")
    message = f"{VERSION}:{item_id}".encode("utf-8")
    return hmac.new(selection_secret, message, hashlib.sha256).hexdigest()


def _write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256_bytes(raw)


def acquire_private_pack(
    *,
    project: Path,
    preregistration_path: Path,
    source_archive: Path,
    private_root: Path,
    selection_secret_path: Path,
) -> dict[str, Any]:
    project = project.resolve()
    preregistration = _verify_preregistration(preregistration_path, project=project)
    source_archive = _assert_git_ignored_private_path(
        project=project,
        path=source_archive,
        require_file=True,
    )
    private_root = _assert_git_ignored_private_path(
        project=project,
        path=private_root,
        require_file=False,
    )
    selection_secret = _read_selection_secret(project=project, path=selection_secret_path)
    selection_secret_commitment = _selection_secret_commitment(selection_secret)
    expected_secret_commitment = preregistration.get("selection", {}).get(
        "selection_secret_commitment_sha256"
    )
    if not hmac.compare_digest(selection_secret_commitment, str(expected_secret_commitment)):
        raise MuSiQueAcquisitionError("selection secret does not match preregistration")
    preregistration_mtime_ns = preregistration_path.stat().st_mtime_ns
    if private_root.exists():
        raise FileExistsError(private_root)
    private_root.mkdir(parents=True, mode=0o700)
    if stat.S_IMODE(private_root.stat().st_mode) != 0o700:
        os.chmod(private_root, 0o700)
    with zipfile.ZipFile(source_archive) as archive:
        train_member = _find_train_member(archive)
        source_train_raw = archive.read(train_member)

    seen_ids: set[str] = set()
    eligible: list[dict[str, Any]] = []
    source_row_count = 0
    for raw in _iter_source_rows(source_train_raw):
        source_row_count += 1
        row = _normalize_source_row(raw)
        if row is None:
            continue
        if row["item_id"] in seen_ids:
            raise MuSiQueAcquisitionError("duplicate eligible item ID")
        seen_ids.add(row["item_id"])
        eligible.append(row)
    selected_count = sum(SPLIT_COUNTS.values())
    if len(eligible) < selected_count:
        raise MuSiQueAcquisitionError("insufficient eligible multi-answer MuSiQue items")
    eligible.sort(
        key=lambda row: (
            _selection_key(row["item_id"], selection_secret),
            row["item_id"],
        )
    )
    selected = eligible[:selected_count]

    split_rows: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    for split_name, count in SPLIT_COUNTS.items():
        rows = []
        for row in selected[cursor : cursor + count]:
            rows.append({"schema": PRIVATE_PACK_SCHEMA, "split": split_name, **row})
        split_rows[split_name] = rows
        cursor += count

    file_rows = []
    split_commitments = {}
    for split_name, rows in split_rows.items():
        path = private_root / f"{split_name}.jsonl"
        file_sha256 = _write_jsonl_exclusive(path, rows)
        item_commitments = [stable_hash(row) for row in rows]
        split_commitments[split_name] = stable_hash(item_commitments)
        file_rows.append(
            {
                "split": split_name,
                "count": len(rows),
                "file_sha256": file_sha256,
                "item_commitment_set_sha256": split_commitments[split_name],
            }
        )
    pack_files_follow_preregistration = all(
        preregistration_mtime_ns < (private_root / f"{split_name}.jsonl").stat().st_mtime_ns
        for split_name in SPLIT_COUNTS
    )
    if not pack_files_follow_preregistration:
        raise MuSiQueAcquisitionError(
            "local filesystem ordering does not show preregistration before pack formation"
        )
    pack_commitment = stable_hash(file_rows)
    return {
        "schema": ACQUISITION_SCHEMA,
        "decision": "private_pack_formed_no_model_execution_authorized",
        "source": {
            "repository": preregistration["source"]["repository"],
            "commit": preregistration["source"]["commit"],
            "license": preregistration["source"]["license"],
            "dataset": "MuSiQue-Answerable v1.0",
            "claim_scope": "multi_alias_eligible_subset_of_official_train_not_full_musique",
            "archive_sha256": _sha256_file(source_archive),
            "official_train_member_sha256": _sha256_bytes(source_train_raw),
        },
        "ordering": {
            "claim": "preregistration_preceded_pack_formation_local_filesystem_evidence",
            "evidence_scope": "local_filesystem_only_not_source_provenance",
            "preregistration_preceded_pack_files_local_mtime": True,
            "archive_mtime_used_as_source_provenance_evidence": False,
            "archive_acquisition_order_claimed_from_mtime": False,
            "preregistration_file_sha256": _sha256_file(preregistration_path),
            "preregistration_sha256": preregistration["preregistration_sha256"],
        },
        "counts": {
            "source_train_rows": source_row_count,
            "eligible_rows": len(eligible),
            "selected_rows": len(selected),
            "splits": SPLIT_COUNTS,
            "oracle_disagreements": 0,
        },
        "commitments": {
            "private_pack_sha256": pack_commitment,
            "selection_secret_commitment_sha256": selection_secret_commitment,
            "split_files": file_rows,
            "item_ids_persisted_publicly": False,
            "private_paths_persisted_publicly": False,
        },
        "private_boundary": {
            "source_archive_git_ignored": True,
            "selection_secret_git_ignored": True,
            "private_pack_root_git_ignored": True,
            "all_private_paths_below_project_artifacts": True,
            "source_archive_path_persisted_publicly": False,
            "selection_secret_persisted_publicly": False,
            "selection_secret_path_persisted_publicly": False,
            "private_pack_path_persisted_publicly": False,
        },
        "oracles": {
            "dual_answer_normalization": True,
            "dual_answer_evaluator": True,
            "dual_support_evaluator": True,
            "synthetic_conformance": preregistration["oracles"]["synthetic_conformance"],
        },
        "safety": {
            "model_calls": 0,
            "scores_computed": 0,
            "online_judge_calls": 0,
            "hipporag_bundled_data_accessed": False,
            "skilllearn_residual_sealed_accessed": False,
            "noaa_private_or_split_data_accessed": False,
            "residual_sealed_content_persisted_publicly": False,
        },
    }


def _write_json_exclusive(path: Path, payload: Mapping[str, Any], hash_field: str) -> None:
    body = dict(payload)
    body[hash_field] = stable_hash(body)
    raw = json.dumps(body, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_secret = subparsers.add_parser("generate-secret")
    generate_secret.add_argument("--project", type=Path, required=True)
    generate_secret.add_argument("--output", type=Path, required=True)
    preregister = subparsers.add_parser("preregister")
    preregister.add_argument("--project", type=Path, required=True)
    preregister.add_argument("--official-repository", type=Path, required=True)
    preregister.add_argument("--selection-secret", type=Path, required=True)
    preregister.add_argument("--output", type=Path, required=True)
    acquire = subparsers.add_parser("acquire")
    acquire.add_argument("--project", type=Path, required=True)
    acquire.add_argument("--preregistration", type=Path, required=True)
    acquire.add_argument("--source-archive", type=Path, required=True)
    acquire.add_argument("--private-root", type=Path, required=True)
    acquire.add_argument("--selection-secret", type=Path, required=True)
    acquire.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args(argv)

    if arguments.command == "generate-secret":
        commitment = generate_selection_secret(
            project=arguments.project.resolve(),
            output=arguments.output,
        )
        print(json.dumps({"selection_secret_commitment_sha256": commitment}, sort_keys=True))
        return 0

    if arguments.command == "preregister":
        payload = build_preregistration(
            project=arguments.project.resolve(),
            official_repository=arguments.official_repository,
            selection_secret_path=arguments.selection_secret,
        )
        raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(arguments.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(raw)
        print(json.dumps({"preregistration_sha256": payload["preregistration_sha256"]}, sort_keys=True))
        return 0

    payload = acquire_private_pack(
        project=arguments.project.resolve(),
        preregistration_path=arguments.preregistration.resolve(),
        source_archive=arguments.source_archive.resolve(),
        private_root=arguments.private_root,
        selection_secret_path=arguments.selection_secret,
    )
    _write_json_exclusive(arguments.receipt, payload, "acquisition_sha256")
    safe = {
        "decision": payload["decision"],
        "private_pack_sha256": payload["commitments"]["private_pack_sha256"],
        "selected_rows": payload["counts"]["selected_rows"],
    }
    print(json.dumps(safe, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
