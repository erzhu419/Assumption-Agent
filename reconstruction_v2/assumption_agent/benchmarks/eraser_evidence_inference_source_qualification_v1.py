"""Aggregate-only qualification of the frozen ERASER Evidence Inference archive.

The formal entry point is intentionally narrow.  It byte-binds one official
archive, the pinned official ``annotations/prompts_merged.csv`` sidecar, and
their already public custody manifests.  It scans tar headers only and opens
exactly ``train.jsonl``, ``val.jsonl``, and the documents referenced by those
two files.  It never opens ``test.jsonl`` or an unreferenced document.  After
collecting the two authorized PromptID sets, it streams the sidecar and retains
only exact matches; Intervention/Comparator/Outcome are never reverse-parsed
from the ERASER query string.

The receipt contains schema, class, completeness, capacity, and hash *counts*
only.  Annotation identifiers, queries, document identifiers, document text,
evidence text, evidence coordinates, and per-row hashes never leave private
memory.  No cohort is selected by this module.

The JSON shape follows the official pinned ERASER implementation at commit
``36467f1662812cbd4fbdd66879946cd7338e08ec``:
``rationale_benchmark.utils.Annotation`` contains alternative evidence groups,
and each ``Evidence`` contains token and sentence half-open spans.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
from typing import Any
import unicodedata


VERSION = "eraser_evidence_inference_source_qualification_v1"
SCHEMA = VERSION

FORMAL_ARCHIVE_RELATIVE_PATH = Path(
    "artifacts/eraser_evidence_inference_official_source_v1/"
    "evidence_inference.tar.gz"
)
FORMAL_CUSTODY_RELATIVE_PATH = Path(
    "manifests/eraser_evidence_inference_source_custody_v1.json"
)
FORMAL_ACCESS_RELATIVE_PATH = Path(
    "manifests/eraser_evidence_inference_source_access_v1.json"
)
FORMAL_PROMPT_SIDECAR_RELATIVE_PATH = Path(
    "reference/evidence_inference_official_v1/annotations/prompts_merged.csv"
)
FORMAL_PROMPT_ACCESS_RELATIVE_PATH = Path(
    "manifests/eraser_evidence_inference_prompt_sidecar_access_v1.json"
)
FORMAL_TAR_HEADER_AMENDMENT_RELATIVE_PATH = Path(
    "manifests/eraser_evidence_inference_tar_header_access_amendment_v1.json"
)
FORMAL_DESIGN_AMENDMENT_RELATIVE_PATH = Path(
    "manifests/eraser_evidence_inference_r7_e3_design_amendment_v1.json"
)
FORMAL_HIPPORAG_FREEZE_RELATIVE_PATH = Path(
    "manifests/eraser_evidence_inference_hipporag_implementation_freeze_v1.json"
)

FORMAL_ARCHIVE_SIZE = 21_903_872
FORMAL_ARCHIVE_SHA256 = (
    "6b748f359d745df79f05e0ed4edf57c19279955126c3b603cb46dc23d36e0555"
)
FORMAL_CUSTODY_FILE_SHA256 = (
    "980284110ba8d9cbe4d86e1391154f9c94d9a74eeccda3e1ade572c835a1f32f"
)
FORMAL_CUSTODY_SELF_SHA256 = (
    "cab365516e9d9534dfc114f5097cbc113ba39f183ac3b823089a432625a86eb6"
)
FORMAL_CUSTODY_COMMIT = "b308e3bbaa40ffa08d4dcf2d4d1c9ca345c75534"
FORMAL_ACCESS_FILE_SHA256 = (
    "fc1bc2f2c46e179334405da2acf4a50939da084ad898e0285e7bcaabe6763f63"
)
FORMAL_ACCESS_SELF_SHA256 = (
    "8eab92003caf5fb0bf668f28d19279f2366d892d39be50f44cc03b4f55748f3e"
)
FORMAL_PROMPT_SIDECAR_SIZE = 1_498_509
FORMAL_PROMPT_SIDECAR_SHA256 = (
    "0d70d9e1e78d113fdbbd4919310eab7768d6a328368f7d8f9d5a3b48f99c25d3"
)
FORMAL_PROMPT_SIDECAR_GIT_BLOB_SHA1 = (
    "a689ff9316aa89255eb780738157ab109181e2d8"
)
FORMAL_PROMPT_SIDECAR_GIT_COMMIT = (
    "a661e8c14f973398380c8865cf2f27a535aaaf6d"
)
FORMAL_PROMPT_ACCESS_FILE_SHA256 = (
    "30fa43114314b3f4074f883d0ef84b7c2d92ab3b83dd3d958040722f50f94fc6"
)
FORMAL_PROMPT_ACCESS_SELF_SHA256 = (
    "a212aa3eb8f39558c05408c2aaa7d83352c450c36703612d84a5a0199bd4dc7c"
)
FORMAL_TAR_HEADER_AMENDMENT_FILE_SHA256 = (
    "8c8b171b1ff1d07518fcc52f81334a9f89c43aeaf7a5b53ee3aa8f9e485f7310"
)
FORMAL_TAR_HEADER_AMENDMENT_SELF_SHA256 = (
    "9487440f175459243658d6333f5307040fc111870d6244032d5a4012d2594f69"
)
FORMAL_TAR_HEADER_AMENDMENT_COMMIT = (
    "13fd17cedfda6ffb5ac747ccbe7a3bb77463e6c4"
)
FORMAL_DESIGN_AMENDMENT_FILE_SHA256 = (
    "7341cc713d717f5f120e03043522fb47881c3023e7c4d42c7cc2ba89e8383515"
)
FORMAL_DESIGN_AMENDMENT_SELF_SHA256 = (
    "6d7b76a444bd940c2c4f488d6f3bac45f2634462f1c077e1e4feae66a0485fe9"
)
FORMAL_DESIGN_AMENDMENT_COMMIT = (
    "e52f21f47a53be6a5cf1f9d8c90394edc2b2291d"
)
FORMAL_BASE_DESIGN_SELF_SHA256 = (
    "49920ccaa8e3f52eeb95fa86d64ecab577971fb8d0cc50d2bd93e0d5baaa2196"
)
FORMAL_HIPPORAG_FREEZE_FILE_SHA256 = (
    "66e116f3d49343922a3e50f6186447f09aea8b8fb6192478a0d0f04726903a51"
)
FORMAL_HIPPORAG_FREEZE_SELF_SHA256 = (
    "17911e90909bb447ce94b097e139694595c436624ccc9b81ec22b39378349219"
)
FORMAL_HIPPORAG_IMPLEMENTATION_COMMIT = (
    "23487651662e15ce225a63208c03e6f0aabf8a0e"
)

FORMAL_EXPECTED_ANNOTATION_COUNTS = {"train": 7_958, "val": 972}
FORMAL_EXPECTED_ARTICLE_COUNTS = {"train": 1_924, "val": 247}

OFFICIAL_CLASSIFICATION_TO_FAMILY = {
    "significantly decreased": "SIGNIFICANTLY_DECREASED",
    "no significant difference": "NO_SIGNIFICANT_DIFFERENCE",
    "significantly increased": "SIGNIFICANTLY_INCREASED",
}
RELATION_FAMILIES = tuple(OFFICIAL_CLASSIFICATION_TO_FAMILY.values())
FAMILY_TO_OFFICIAL_CLASSIFICATION = {
    family: official
    for official, family in OFFICIAL_CLASSIFICATION_TO_FAMILY.items()
}
FORMAL_COHORT_DEMANDS = {
    "train": {"A_form": 16, "F_search": 12},
    "val": {"A_hold": 10, "M_search": 10},
}

_SPLIT_BASENAMES = {"train.jsonl": "train", "val.jsonl": "val"}
_ANNOTATION_REQUIRED_FIELDS = frozenset(
    {"annotation_id", "query", "evidences", "classification"}
)
_ANNOTATION_ALLOWED_FIELDS = frozenset(
    {
        "annotation_id",
        "query",
        "evidences",
        "classification",
        "query_type",
        "docids",
    }
)
_EVIDENCE_REQUIRED_FIELDS = frozenset({"text", "docid"})
_EVIDENCE_ALLOWED_FIELDS = frozenset(
    {
        "text",
        "docid",
        "start_token",
        "end_token",
        "start_sentence",
        "end_sentence",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")
_PROMPT_SIDECAR_REQUIRED_FIELDS = frozenset(
    {"PromptID", "PMCID", "Outcome", "Intervention", "Comparator"}
)


class EraserEvidenceInferenceQualificationError(RuntimeError):
    """The frozen source, manifest, archive, or schema contract drifted."""


@dataclass(frozen=True)
class _EvidenceSpan:
    docid: str
    start_token: int
    end_token: int
    start_sentence: int
    end_sentence: int
    text_tokens: tuple[str, ...] | None


@dataclass(frozen=True)
class _PrivateAnnotation:
    prompt_id: str
    normalized_query_sha256: str
    classification: str
    article_docid: str | None
    evidence_groups: tuple[tuple[_EvidenceSpan, ...], ...]


@dataclass(frozen=True)
class _ParsedSplit:
    annotations: tuple[_PrivateAnnotation, ...]
    referenced_docids: frozenset[str]
    annotation_keyset_hash_counts: Mapping[str, int]
    evidence_keyset_hash_counts: Mapping[str, int]
    annotation_identity_hashes: frozenset[str]
    query_hashes: frozenset[str]
    annotation_identity_duplicate_count: int
    query_duplicate_count: int
    class_counts: Mapping[str, int]
    evidence_text_representation_counts: Mapping[str, int]
    evidence_group_cardinality_counts: Mapping[int, int]


@dataclass(frozen=True)
class _PrivateDocument:
    sentences: tuple[tuple[str, ...], ...]
    flattened_tokens: tuple[str, ...]
    sentence_token_boundaries: tuple[int, ...]
    content_sha256: str
    member_content_sha256: str


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
        raise EraserEvidenceInferenceQualificationError(
            "value is not canonical JSON"
        ) from exc


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
                size += len(block)
    except OSError as exc:
        raise EraserEvidenceInferenceQualificationError(
            "source binding cannot be read"
        ) from exc
    return digest.hexdigest(), size


def _strict_json(raw: bytes, *, context: str) -> Any:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise EraserEvidenceInferenceQualificationError(
            f"{context} is not strict UTF-8"
        ) from exc

    def object_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise EraserEvidenceInferenceQualificationError(
                    f"{context} contains a duplicate JSON object key"
                )
            output[key] = value
        return output

    def reject_constant(_value: str) -> None:
        raise EraserEvidenceInferenceQualificationError(
            f"{context} contains a nonfinite JSON constant"
        )

    try:
        return json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except EraserEvidenceInferenceQualificationError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise EraserEvidenceInferenceQualificationError(
            f"{context} is not strict JSON"
        ) from exc


def _load_manifest(
    path: Path,
    *,
    schema: str,
    self_hash_field: str,
) -> tuple[dict[str, Any], str, str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EraserEvidenceInferenceQualificationError(
            "public source manifest cannot be read"
        ) from exc
    payload = _strict_json(raw, context="public source manifest")
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise EraserEvidenceInferenceQualificationError(
            "public source manifest schema drifted"
        )
    declared = payload.get(self_hash_field)
    if not isinstance(declared, str) or not _SHA256_RE.fullmatch(declared):
        raise EraserEvidenceInferenceQualificationError(
            "public source manifest self hash is invalid"
        )
    body = dict(payload)
    body.pop(self_hash_field)
    if _stable_hash(body) != declared:
        raise EraserEvidenceInferenceQualificationError(
            "public source manifest self hash drifted"
        )
    return payload, hashlib.sha256(raw).hexdigest(), declared


def _require_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EraserEvidenceInferenceQualificationError(f"{context} drifted")
    return value


def _validate_manifests(
    custody_path: Path,
    access_path: Path,
    *,
    expected_archive_sha256: str,
    expected_archive_size: int,
    enforce_formal_manifest_identity: bool,
) -> dict[str, str]:
    custody, custody_file_hash, custody_self_hash = _load_manifest(
        custody_path,
        schema="eraser_evidence_inference_source_custody_v1",
        self_hash_field="source_custody_sha256",
    )
    access, access_file_hash, access_self_hash = _load_manifest(
        access_path,
        schema="eraser_evidence_inference_source_access_v1",
        self_hash_field="source_access_sha256",
    )

    if enforce_formal_manifest_identity and (
        custody_file_hash != FORMAL_CUSTODY_FILE_SHA256
        or custody_self_hash != FORMAL_CUSTODY_SELF_SHA256
        or access_file_hash != FORMAL_ACCESS_FILE_SHA256
        or access_self_hash != FORMAL_ACCESS_SELF_SHA256
    ):
        raise EraserEvidenceInferenceQualificationError(
            "formal public source manifest identity drifted"
        )

    archive_binding = _require_mapping(
        access.get("archive_binding"), context="access archive binding"
    )
    custody_archive = _require_mapping(
        custody.get("archive_metadata"), context="custody archive metadata"
    )
    custody_binding = _require_mapping(
        access.get("custody_binding"), context="access custody binding"
    )
    pre_access = _require_mapping(
        access.get("pre_member_access_state"), context="pre-member access state"
    )
    custody_claim = _require_mapping(
        custody.get("claim_boundary"), context="custody claim boundary"
    )
    split_policy = _require_mapping(
        custody.get("prospective_split_policy"), context="split policy"
    )
    terminal_policy = _require_mapping(
        custody.get("terminal_policy"), context="terminal policy"
    )

    if (
        archive_binding.get("sha256") != expected_archive_sha256
        or archive_binding.get("byte_size") != expected_archive_size
        or custody_archive.get("content_length") != expected_archive_size
    ):
        raise EraserEvidenceInferenceQualificationError(
            "public manifest archive binding drifted"
        )
    custody_path_value = custody_archive.get("local_ignored_relative_path")
    access_path_value = archive_binding.get("local_relative_path")
    if (
        not isinstance(custody_path_value, str)
        or custody_path_value != access_path_value
    ):
        raise EraserEvidenceInferenceQualificationError(
            "public manifest archive path binding drifted"
        )
    if (
        custody_binding.get("custody_file_sha256") != custody_file_hash
        or custody_binding.get("custody_self_sha256") != custody_self_hash
    ):
        raise EraserEvidenceInferenceQualificationError(
            "access-to-custody binding drifted"
        )
    if enforce_formal_manifest_identity and (
        custody_binding.get("custody_commit") != FORMAL_CUSTODY_COMMIT
        or access_path_value != FORMAL_ARCHIVE_RELATIVE_PATH.as_posix()
    ):
        raise EraserEvidenceInferenceQualificationError(
            "formal source custody binding drifted"
        )

    forbidden_pre_access_truths = (
        "archive_member_content_opened_or_extracted",
        "archive_member_list_created",
        "dataset_member_individually_hashed",
        "source_schema_or_row_parsed",
        "test_member_name_query_document_label_or_content_opened",
    )
    if any(pre_access.get(key) is not False for key in forbidden_pre_access_truths):
        raise EraserEvidenceInferenceQualificationError(
            "pre-member access boundary drifted"
        )
    forbidden_custody_truths = (
        "archive_body_downloaded_or_opened",
        "dataset_member_or_row_listed_parsed_or_hashed",
        "retrieval_action_evaluator_or_score_run",
        "selection_secret_or_cohort_created",
        "test_query_document_or_label_opened",
    )
    if any(custody_claim.get(key) is not False for key in forbidden_custody_truths):
        raise EraserEvidenceInferenceQualificationError(
            "custody claim boundary drifted"
        )
    if (
        not isinstance(split_policy.get("test"), str)
        or terminal_policy.get("test_use_authorized") is not False
        or terminal_policy.get("online_evaluation_fallback") is not False
    ):
        raise EraserEvidenceInferenceQualificationError(
            "test or online policy drifted"
        )
    return {
        "custody_file_sha256": custody_file_hash,
        "custody_self_sha256": custody_self_hash,
        "access_file_sha256": access_file_hash,
        "access_self_sha256": access_self_hash,
    }


def _validate_prompt_access_manifest(
    path: Path,
    *,
    custody_self_sha256: str,
    expected_sidecar_sha256: str,
    expected_sidecar_size: int,
    expected_git_blob_sha1: str,
    expected_git_commit: str,
    enforce_formal_manifest_identity: bool,
) -> dict[str, str]:
    payload, file_hash, self_hash = _load_manifest(
        path,
        schema="eraser_evidence_inference_prompt_sidecar_access_v1",
        self_hash_field="prompt_sidecar_access_sha256",
    )
    if enforce_formal_manifest_identity and (
        file_hash != FORMAL_PROMPT_ACCESS_FILE_SHA256
        or self_hash != FORMAL_PROMPT_ACCESS_SELF_SHA256
    ):
        raise EraserEvidenceInferenceQualificationError(
            "formal prompt sidecar access manifest identity drifted"
        )
    binding = _require_mapping(
        payload.get("binding"), context="prompt sidecar binding"
    )
    boundary = _require_mapping(
        payload.get("access_boundary"), context="prompt sidecar access boundary"
    )
    contract = _require_mapping(
        payload.get("sidecar_contract"), context="prompt sidecar contract"
    )
    custody_binding = _require_mapping(
        payload.get("source_custody_binding"),
        context="prompt sidecar custody binding",
    )
    if (
        binding.get("sha256") != expected_sidecar_sha256
        or binding.get("byte_size") != expected_sidecar_size
        or binding.get("git_blob_sha1") != expected_git_blob_sha1
        or binding.get("git_commit") != expected_git_commit
        or binding.get("repository_path") != "annotations/prompts_merged.csv"
    ):
        raise EraserEvidenceInferenceQualificationError(
            "prompt sidecar byte or git binding drifted"
        )
    if (
        custody_binding.get("custody_self_sha256") != custody_self_sha256
        or boundary.get("content_rows_listed_parsed_or_printed") is not False
        or boundary.get("prompt_or_article_values_opened") is not False
        or boundary.get("test_prompt_values_opened_or_used") is not False
        or boundary.get("exact_file_stat_git_blob_and_whole_file_sha256_only")
        is not True
    ):
        raise EraserEvidenceInferenceQualificationError(
            "prompt sidecar pre-row-access boundary drifted"
        )
    fields = contract.get("label_free_fields")
    if (
        not isinstance(fields, list)
        or set(fields) != {"Intervention", "Comparator", "Outcome"}
        or not isinstance(contract.get("binding_key"), str)
    ):
        raise EraserEvidenceInferenceQualificationError(
            "prompt sidecar structured-field contract drifted"
        )
    return {"file_sha256": file_hash, "self_sha256": self_hash}


def _validate_container_amendments(
    tar_header_amendment_path: Path,
    design_amendment_path: Path,
    *,
    archive_sha256: str,
    custody_self_sha256: str,
    access_self_sha256: str,
    enforce_formal_manifest_identity: bool,
) -> dict[str, str]:
    tar_payload, tar_file_hash, tar_self_hash = _load_manifest(
        tar_header_amendment_path,
        schema="eraser_evidence_inference_tar_header_access_amendment_v1",
        self_hash_field="tar_header_access_amendment_sha256",
    )
    design_payload, design_file_hash, design_self_hash = _load_manifest(
        design_amendment_path,
        schema="eraser_evidence_inference_r7_e3_design_amendment_v1",
        self_hash_field="design_amendment_sha256",
    )
    if enforce_formal_manifest_identity and (
        tar_file_hash != FORMAL_TAR_HEADER_AMENDMENT_FILE_SHA256
        or tar_self_hash != FORMAL_TAR_HEADER_AMENDMENT_SELF_SHA256
        or design_file_hash != FORMAL_DESIGN_AMENDMENT_FILE_SHA256
        or design_self_hash != FORMAL_DESIGN_AMENDMENT_SELF_SHA256
    ):
        raise EraserEvidenceInferenceQualificationError(
            "formal container amendment identity drifted"
        )

    base = _require_mapping(
        tar_payload.get("base_bindings"), context="tar amendment base binding"
    )
    boundary = _require_mapping(
        tar_payload.get("authorized_header_boundary"),
        context="authorized tar header boundary",
    )
    tar_claim = _require_mapping(
        tar_payload.get("claim_boundary"), context="tar amendment claim boundary"
    )
    supersession = _require_mapping(
        tar_payload.get("narrow_supersession"),
        context="tar amendment narrow supersession",
    )
    if (
        base.get("archive_sha256") != archive_sha256
        or base.get("custody_self_sha256") != custody_self_sha256
        or base.get("source_access_self_sha256") != access_self_sha256
        or boundary.get("member_name_persistence_output_or_hash") is not False
        or boundary.get("test_member_content_extract_open_read_hash_or_parse")
        is not False
        or boundary.get(
            "test_only_document_content_extract_open_read_hash_or_parse"
        )
        is not False
        or not isinstance(boundary.get("in_memory_routing"), str)
        or not isinstance(supersession.get("unchanged_clause"), str)
        or tar_payload.get("status")
        != "prospective_container_routing_correction_before_any_archive_member_header_access"
    ):
        raise EraserEvidenceInferenceQualificationError(
            "container header access boundary drifted"
        )
    if any(
        tar_claim.get(key) is not False
        for key in (
            "action_evaluator_retrieval_or_score_changed",
            "archive_member_header_or_content_access_before_this_amendment",
            "cohort_family_quota_or_selection_changed",
            "online_evaluation_authorized",
            "test_query_document_label_or_content_authorized",
        )
    ):
        raise EraserEvidenceInferenceQualificationError(
            "container amendment prospective claim drifted"
        )

    design_binding = _require_mapping(
        design_payload.get("base_design_binding"),
        context="design amendment base binding",
    )
    access_binding = _require_mapping(
        design_payload.get("tar_header_access_binding"),
        context="design amendment tar binding",
    )
    scope = _require_mapping(
        design_payload.get("change_scope"), context="design amendment scope"
    )
    prospective = _require_mapping(
        design_payload.get("prospective_state"),
        context="design amendment prospective state",
    )
    if (
        access_binding.get("amendment_file_sha256") != tar_file_hash
        or access_binding.get("amendment_self_sha256") != tar_self_hash
        or not isinstance(scope.get("source_qualification_test_policy"), str)
        or scope.get("action_operator_feature_evaluator_or_score_change")
        is not False
        or scope.get("cohort_family_quota_split_or_selection_change") is not False
        or scope.get("new_gate_threshold_retry_or_online_fallback") is not False
    ):
        raise EraserEvidenceInferenceQualificationError(
            "design amendment scope or tar binding drifted"
        )
    if any(
        prospective.get(key) is not False
        for key in (
            "archive_member_header_or_content_access_before_this_amendment",
            "private_assignment_or_secret_created",
            "retrieval_action_evaluator_or_score_run",
        )
    ):
        raise EraserEvidenceInferenceQualificationError(
            "design amendment prospective state drifted"
        )
    if enforce_formal_manifest_identity and (
        access_binding.get("amendment_commit")
        != FORMAL_TAR_HEADER_AMENDMENT_COMMIT
        or design_binding.get("design_self_sha256")
        != FORMAL_BASE_DESIGN_SELF_SHA256
    ):
        raise EraserEvidenceInferenceQualificationError(
            "formal container design binding drifted"
        )
    return {
        "tar_header_amendment_file_sha256": tar_file_hash,
        "tar_header_amendment_self_sha256": tar_self_hash,
        "design_amendment_file_sha256": design_file_hash,
        "design_amendment_self_sha256": design_self_hash,
    }


def _validate_hipporag_implementation_freeze(
    path: Path,
    *,
    enforce_formal_manifest_identity: bool,
) -> dict[str, str]:
    payload, file_hash, self_hash = _load_manifest(
        path,
        schema="eraser_evidence_inference_hipporag_implementation_freeze_v1",
        self_hash_field="implementation_freeze_sha256",
    )
    if enforce_formal_manifest_identity and (
        file_hash != FORMAL_HIPPORAG_FREEZE_FILE_SHA256
        or self_hash != FORMAL_HIPPORAG_FREEZE_SELF_SHA256
    ):
        raise EraserEvidenceInferenceQualificationError(
            "formal ERASER HippoRAG freeze identity drifted"
        )
    design = _require_mapping(
        payload.get("design_binding"), context="HippoRAG freeze design binding"
    )
    runtime = _require_mapping(
        payload.get("base_runtime_binding"),
        context="HippoRAG freeze runtime binding",
    )
    implementation = _require_mapping(
        payload.get("implementation_binding"),
        context="HippoRAG freeze implementation binding",
    )
    formal_tests = _require_mapping(
        payload.get("synthetic_formal_tests"),
        context="HippoRAG freeze synthetic tests",
    )
    integration = _require_mapping(
        payload.get("synthetic_official_core_integration"),
        context="HippoRAG freeze integration",
    )
    files = implementation.get("files")
    if not isinstance(files, list) or not files:
        raise EraserEvidenceInferenceQualificationError(
            "HippoRAG implementation file binding drifted"
        )
    required_suffixes = {
        "__init__.py",
        "adapter.py",
        "contract.py",
        "worker.py",
        "test_eraser_evidence_inference_official_hipporag_v1.py",
    }
    observed_suffixes: set[str] = set()
    for row in files:
        row_mapping = _require_mapping(
            row, context="HippoRAG implementation file row"
        )
        file_path = row_mapping.get("path")
        file_sha256 = row_mapping.get("sha256")
        if (
            not isinstance(file_path, str)
            or not isinstance(file_sha256, str)
            or not _SHA256_RE.fullmatch(file_sha256)
        ):
            raise EraserEvidenceInferenceQualificationError(
                "HippoRAG implementation file row drifted"
            )
        observed_suffixes.add(PurePosixPath(file_path).name)
    if observed_suffixes != required_suffixes:
        raise EraserEvidenceInferenceQualificationError(
            "HippoRAG implementation file set drifted"
        )
    if (
        payload.get("status")
        != "frozen_offline_item_local_implementation_after_non_scoring_integration_pass"
        or design.get("base_design_self_sha256")
        != FORMAL_BASE_DESIGN_SELF_SHA256
        or design.get("container_design_amendment_self_sha256")
        != FORMAL_DESIGN_AMENDMENT_SELF_SHA256
        or runtime.get("base_binding_receipt_sha256")
        != "522d31926df70f983ae2f644f05c9f3ee45fcd08e0d847642e144652df5a45d0"
        or runtime.get("runtime_attestation_receipt_sha256")
        != "23996f9f41f494e2fd032b285039ec9420f6a893c24081e59c1ec79f229c2c60"
        or not isinstance(implementation.get("frozen_contract"), str)
        or formal_tests.get("real_source_or_benchmark_item_read") is not False
        or formal_tests.get("passed_case_count")
        != formal_tests.get("collected_case_count")
        or type(formal_tests.get("passed_case_count")) is not int
        or formal_tests.get("passed_case_count", 0) <= 0
        or integration.get("passed") is not True
        or integration.get("external_network_transport_possible") is not False
        or type(integration.get("logical_sentence_count")) is not int
        or integration.get("logical_sentence_count", 0) <= 128
        or type(integration.get("unique_exact_text_count")) is not int
        or not 0
        < integration.get("unique_exact_text_count", 0)
        < integration.get("logical_sentence_count", 0)
        or integration.get("output_count") != 5
        or integration.get("work_root_removed_before_return") is not True
    ):
        raise EraserEvidenceInferenceQualificationError(
            "ERASER HippoRAG implementation freeze contract drifted"
        )
    if enforce_formal_manifest_identity and (
        implementation.get("commit") != FORMAL_HIPPORAG_IMPLEMENTATION_COMMIT
    ):
        raise EraserEvidenceInferenceQualificationError(
            "formal ERASER HippoRAG implementation commit drifted"
        )
    return {
        "hipporag_implementation_freeze_file_sha256": file_hash,
        "hipporag_implementation_freeze_self_sha256": self_hash,
    }


def _safe_tar_parts(name: str) -> tuple[str, ...]:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise EraserEvidenceInferenceQualificationError(
            "archive contains an unsafe member header"
        )
    path = PurePosixPath(name)
    parts = tuple(part for part in path.parts if part not in {"", "."})
    if path.is_absolute() or not parts or any(part == ".." for part in parts):
        raise EraserEvidenceInferenceQualificationError(
            "archive contains an unsafe member header"
        )
    return parts


def _read_tar_member(bundle: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    if member.size < 0 or member.size > FORMAL_ARCHIVE_SIZE:
        raise EraserEvidenceInferenceQualificationError(
            "archive member declared size is invalid"
        )
    handle = bundle.extractfile(member)
    if handle is None:
        raise EraserEvidenceInferenceQualificationError(
            "authorized archive member cannot be opened"
        )
    raw = handle.read(member.size + 1)
    if len(raw) != member.size:
        raise EraserEvidenceInferenceQualificationError(
            "authorized archive member size drifted"
        )
    return raw


def _read_split_members(
    archive_path: Path,
) -> tuple[dict[str, bytes], tuple[str, ...], dict[str, int]]:
    split_raw: dict[str, bytes] = {}
    split_root: tuple[str, ...] | None = None
    counts = Counter({"regular": 0, "directory": 0, "other": 0})
    try:
        with tarfile.open(archive_path, mode="r:gz", errorlevel=2) as bundle:
            for member in bundle:
                parts = _safe_tar_parts(member.name)
                if member.isdir():
                    counts["directory"] += 1
                    continue
                if not member.isfile():
                    counts["other"] += 1
                    continue
                counts["regular"] += 1
                split = _SPLIT_BASENAMES.get(parts[-1])
                if split is None:
                    continue
                if split in split_raw:
                    raise EraserEvidenceInferenceQualificationError(
                        "archive contains a duplicate authorized split member"
                    )
                current_root = parts[:-1]
                if split_root is None:
                    split_root = current_root
                elif split_root != current_root:
                    raise EraserEvidenceInferenceQualificationError(
                        "authorized split members do not share one dataset root"
                    )
                split_raw[split] = _read_tar_member(bundle, member)
    except EraserEvidenceInferenceQualificationError:
        raise
    except (OSError, tarfile.TarError, EOFError) as exc:
        raise EraserEvidenceInferenceQualificationError(
            "official archive tar/gzip integrity failed"
        ) from exc
    if set(split_raw) != {"train", "val"} or split_root is None:
        raise EraserEvidenceInferenceQualificationError(
            "archive lacks exactly one authorized train and validation split"
        )
    return split_raw, split_root, dict(counts)


def _query_hash(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise EraserEvidenceInferenceQualificationError(
            "annotation query type or value drifted"
        )
    normalized = " ".join(unicodedata.normalize("NFKC", value).split()).casefold()
    if not normalized:
        raise EraserEvidenceInferenceQualificationError(
            "annotation normalized query is empty"
        )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _evidence_text(value: Any) -> tuple[str, tuple[str, ...] | None]:
    if isinstance(value, str):
        if not value.strip() or "\x00" in value:
            raise EraserEvidenceInferenceQualificationError(
                "evidence text representation drifted"
            )
        return "string", tuple(token for token in value.split() if token)
    if isinstance(value, list) and value:
        if all(isinstance(token, str) for token in value):
            if any(not token or "\x00" in token for token in value):
                raise EraserEvidenceInferenceQualificationError(
                    "evidence text representation drifted"
                )
            return "string_sequence", tuple(value)
        if all(type(token) is int for token in value):
            return "integer_sequence", None
    raise EraserEvidenceInferenceQualificationError(
        "evidence text representation drifted"
    )


def _parse_jsonl(raw: bytes, *, split: str) -> _ParsedSplit:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise EraserEvidenceInferenceQualificationError(
            "authorized split is not strict UTF-8"
        ) from exc
    lines = text.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise EraserEvidenceInferenceQualificationError(
            "authorized split JSONL line structure drifted"
        )

    annotations: list[_PrivateAnnotation] = []
    referenced_docids: set[str] = set()
    annotation_keysets: Counter[str] = Counter()
    evidence_keysets: Counter[str] = Counter()
    identity_hashes: set[str] = set()
    query_hashes: set[str] = set()
    duplicate_identity_count = 0
    duplicate_query_count = 0
    class_counts = Counter({family: 0 for family in RELATION_FAMILIES})
    text_kind_counts: Counter[str] = Counter()
    group_cardinalities: Counter[int] = Counter()

    for line in lines:
        row = _strict_json(line.encode("utf-8"), context="authorized split row")
        if not isinstance(row, Mapping):
            raise EraserEvidenceInferenceQualificationError(
                "annotation row is not an object"
            )
        keys = set(row)
        if (
            not _ANNOTATION_REQUIRED_FIELDS <= keys
            or not keys <= _ANNOTATION_ALLOWED_FIELDS
            or any(not isinstance(key, str) for key in keys)
        ):
            raise EraserEvidenceInferenceQualificationError(
                "annotation field schema drifted"
            )
        annotation_keysets[_stable_hash(sorted(keys))] += 1

        annotation_id = row.get("annotation_id")
        if (
            not isinstance(annotation_id, str)
            or not annotation_id.strip()
            or "\x00" in annotation_id
        ):
            raise EraserEvidenceInferenceQualificationError(
                "annotation identity type or value drifted"
            )
        identity_hash = hashlib.sha256(annotation_id.encode("utf-8")).hexdigest()
        if identity_hash in identity_hashes:
            duplicate_identity_count += 1
        identity_hashes.add(identity_hash)

        query_digest = _query_hash(row.get("query"))
        if query_digest in query_hashes:
            duplicate_query_count += 1
        query_hashes.add(query_digest)

        query_type = row.get("query_type")
        if query_type is not None and not isinstance(query_type, str):
            raise EraserEvidenceInferenceQualificationError(
                "annotation query_type drifted"
            )
        official_classification = row.get("classification")
        if official_classification not in OFFICIAL_CLASSIFICATION_TO_FAMILY:
            raise EraserEvidenceInferenceQualificationError(
                "annotation relation family drifted"
            )
        classification = OFFICIAL_CLASSIFICATION_TO_FAMILY[
            official_classification
        ]
        class_counts[classification] += 1

        declared_docids_raw = row.get("docids")
        if declared_docids_raw is None:
            declared_docids: set[str] = set()
        elif isinstance(declared_docids_raw, list) and all(
            isinstance(docid, str) for docid in declared_docids_raw
        ):
            declared_docids = set(declared_docids_raw)
        else:
            raise EraserEvidenceInferenceQualificationError(
                "annotation docids field drifted"
            )
        for docid in declared_docids:
            _validate_docid(docid)

        raw_groups = row.get("evidences")
        if not isinstance(raw_groups, list):
            raise EraserEvidenceInferenceQualificationError(
                "annotation evidences field drifted"
            )
        groups: list[tuple[_EvidenceSpan, ...]] = []
        evidence_docids: set[str] = set()
        for raw_group in raw_groups:
            if not isinstance(raw_group, list):
                raise EraserEvidenceInferenceQualificationError(
                    "alternative evidence group schema drifted"
                )
            group_cardinalities[len(raw_group)] += 1
            group: list[_EvidenceSpan] = []
            for raw_evidence in raw_group:
                if not isinstance(raw_evidence, Mapping):
                    raise EraserEvidenceInferenceQualificationError(
                        "evidence entry is not an object"
                    )
                evidence_keys = set(raw_evidence)
                if (
                    not _EVIDENCE_REQUIRED_FIELDS <= evidence_keys
                    or not evidence_keys <= _EVIDENCE_ALLOWED_FIELDS
                    or any(not isinstance(key, str) for key in evidence_keys)
                ):
                    raise EraserEvidenceInferenceQualificationError(
                        "evidence field schema drifted"
                    )
                evidence_keysets[_stable_hash(sorted(evidence_keys))] += 1
                kind, text_tokens = _evidence_text(raw_evidence.get("text"))
                text_kind_counts[kind] += 1
                docid = raw_evidence.get("docid")
                if not isinstance(docid, str):
                    raise EraserEvidenceInferenceQualificationError(
                        "evidence document identity type drifted"
                    )
                _validate_docid(docid)
                offsets: list[int] = []
                for field in (
                    "start_token",
                    "end_token",
                    "start_sentence",
                    "end_sentence",
                ):
                    value = raw_evidence.get(field, -1)
                    if type(value) is not int:
                        raise EraserEvidenceInferenceQualificationError(
                            "evidence span coordinate type drifted"
                        )
                    offsets.append(value)
                group.append(
                    _EvidenceSpan(docid, *offsets, text_tokens=text_tokens)
                )
                evidence_docids.add(docid)
            groups.append(tuple(group))

        all_docids = declared_docids | evidence_docids
        referenced_docids.update(all_docids)
        article_docid = next(iter(all_docids)) if len(all_docids) == 1 else None
        annotations.append(
            _PrivateAnnotation(
                prompt_id=annotation_id,
                normalized_query_sha256=query_digest,
                classification=classification,
                article_docid=article_docid,
                evidence_groups=tuple(groups),
            )
        )

    if duplicate_identity_count:
        raise EraserEvidenceInferenceQualificationError(
            "annotation identities are not unique"
        )
    return _ParsedSplit(
        annotations=tuple(annotations),
        referenced_docids=frozenset(referenced_docids),
        annotation_keyset_hash_counts=dict(annotation_keysets),
        evidence_keyset_hash_counts=dict(evidence_keysets),
        annotation_identity_hashes=frozenset(identity_hashes),
        query_hashes=frozenset(query_hashes),
        annotation_identity_duplicate_count=duplicate_identity_count,
        query_duplicate_count=duplicate_query_count,
        class_counts=dict(class_counts),
        evidence_text_representation_counts=dict(text_kind_counts),
        evidence_group_cardinality_counts=dict(group_cardinalities),
    )


def _validate_docid(docid: str) -> None:
    if (
        not docid
        or not docid.strip()
        or "\x00" in docid
        or "\\" in docid
        or PurePosixPath(docid).name != docid
        or docid in {".", ".."}
    ):
        raise EraserEvidenceInferenceQualificationError(
            "document identity is unsafe or empty"
        )


def _canonical_pmcid(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("PMC"):
        stripped = stripped[3:]
    if not stripped or not stripped.isascii() or not stripped.isdigit():
        raise EraserEvidenceInferenceQualificationError(
            "referenced prompt-to-article PMCID binding drifted"
        )
    return f"PMC{int(stripped)}"


def _stream_prompt_sidecar(
    path: Path,
    *,
    prompt_to_article: Mapping[str, str],
    prompt_to_split: Mapping[str, str],
) -> dict[str, Any]:
    """Retain only exact train/val PromptID matches while streaming the CSV."""

    seen: set[str] = set()
    structured_prompt_hashes: set[str] = set()
    matched_by_split = Counter({"train": 0, "val": 0})
    try:
        with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames
            if (
                headers is None
                or len(headers) != len(set(headers))
                or not _PROMPT_SIDECAR_REQUIRED_FIELDS <= set(headers)
            ):
                raise EraserEvidenceInferenceQualificationError(
                    "prompt sidecar CSV header drifted"
                )
            for row in reader:
                prompt_id = row.get("PromptID")
                if prompt_id not in prompt_to_article:
                    continue
                if prompt_id in seen:
                    raise EraserEvidenceInferenceQualificationError(
                        "referenced PromptID is duplicated in the prompt sidecar"
                    )
                if None in row:
                    raise EraserEvidenceInferenceQualificationError(
                        "referenced prompt sidecar row has excess CSV columns"
                    )
                facets: dict[str, str] = {}
                for field in ("Intervention", "Comparator", "Outcome"):
                    value = row.get(field)
                    if (
                        not isinstance(value, str)
                        or not value.strip()
                        or "\x00" in value
                    ):
                        raise EraserEvidenceInferenceQualificationError(
                            "referenced prompt sidecar ICO field is incomplete"
                        )
                    facets[field] = value.strip()
                pmcid = row.get("PMCID")
                if not isinstance(pmcid, str) or _canonical_pmcid(
                    pmcid
                ) != _canonical_pmcid(prompt_to_article[prompt_id]):
                    raise EraserEvidenceInferenceQualificationError(
                        "referenced PromptID has an ambiguous article binding"
                    )
                seen.add(prompt_id)
                matched_by_split[prompt_to_split[prompt_id]] += 1
                structured_prompt_hashes.add(_stable_hash(facets))
    except EraserEvidenceInferenceQualificationError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise EraserEvidenceInferenceQualificationError(
            "prompt sidecar streaming failed"
        ) from exc

    missing = set(prompt_to_article) - seen
    if missing:
        raise EraserEvidenceInferenceQualificationError(
            "one or more authorized PromptIDs are absent from the prompt sidecar"
        )
    return {
        "authorized_prompt_id_count": len(prompt_to_article),
        "exact_one_to_one_match_count": len(seen),
        "missing_match_count": 0,
        "duplicate_or_ambiguous_match_count": 0,
        "matched_prompt_counts_by_split": _string_counter(matched_by_split),
        "independent_ico_field_count_per_prompt": 3,
        "unique_structured_ico_hash_count": len(structured_prompt_hashes),
        "duplicate_structured_ico_hash_occurrence_count": (
            len(seen) - len(structured_prompt_hashes)
        ),
        "query_string_reverse_parsing_used": False,
        "unreferenced_or_test_row_persisted_or_emitted_count": 0,
    }


def _decode_document(raw: bytes) -> _PrivateDocument:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise EraserEvidenceInferenceQualificationError(
            "referenced document is not strict UTF-8"
        ) from exc
    lines = [line.strip() for line in text.splitlines()]
    sentences = tuple(
        tuple(token for token in line.split(" ") if token)
        for line in lines
        if line
    )
    if not sentences or any(not sentence for sentence in sentences):
        raise EraserEvidenceInferenceQualificationError(
            "referenced document sentence/token structure drifted"
        )
    flattened = tuple(token for sentence in sentences for token in sentence)
    boundaries = [0]
    for sentence in sentences:
        boundaries.append(boundaries[-1] + len(sentence))
    return _PrivateDocument(
        sentences=sentences,
        flattened_tokens=flattened,
        sentence_token_boundaries=tuple(boundaries),
        content_sha256=_stable_hash(sentences),
        member_content_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _read_referenced_documents(
    archive_path: Path,
    *,
    split_root: tuple[str, ...],
    referenced_docids: frozenset[str],
) -> dict[str, _PrivateDocument]:
    expected_paths = {
        split_root + ("docs", docid): docid for docid in referenced_docids
    }
    documents: dict[str, _PrivateDocument] = {}
    try:
        with tarfile.open(archive_path, mode="r:gz", errorlevel=2) as bundle:
            for member in bundle:
                parts = _safe_tar_parts(member.name)
                docid = expected_paths.get(parts)
                if docid is None:
                    continue
                if not member.isfile() or docid in documents:
                    raise EraserEvidenceInferenceQualificationError(
                        "referenced document member is missing, duplicate, or nonregular"
                    )
                documents[docid] = _decode_document(
                    _read_tar_member(bundle, member)
                )
    except EraserEvidenceInferenceQualificationError:
        raise
    except (OSError, tarfile.TarError, EOFError) as exc:
        raise EraserEvidenceInferenceQualificationError(
            "official archive document scan failed"
        ) from exc
    if set(documents) != set(referenced_docids):
        raise EraserEvidenceInferenceQualificationError(
            "one or more referenced documents are absent"
        )
    return documents


def _string_counter(counter: Mapping[Any, int]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=str)}


def _verify_split(
    parsed: _ParsedSplit,
    documents: Mapping[str, _PrivateDocument],
    *,
    excluded_normalized_query_hashes: frozenset[str],
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    span_counts = Counter(
        {
            "evidence_span_count": 0,
            "token_span_valid_count": 0,
            "sentence_span_valid_count": 0,
            "token_span_contained_by_sentence_span_count": 0,
            "fully_valid_span_count": 0,
            "evidence_text_exact_token_slice_match_count": 0,
        }
    )
    group_counts = Counter(
        {
            "alternative_group_count": 0,
            "nonempty_group_count": 0,
            "fully_complete_group_count": 0,
            "duplicate_span_occurrence_count": 0,
            "duplicate_alternative_group_occurrence_count": 0,
        }
    )
    annotation_counts = Counter(
        {
            "annotation_count": len(parsed.annotations),
            "single_article_annotation_count": 0,
            "at_least_one_complete_group_annotation_count": 0,
            "all_alternative_groups_complete_annotation_count": 0,
            "whole_duplicate_query_group_excluded_annotation_count": 0,
            "fewer_than_five_sentence_annotation_count": 0,
            "capacity_eligible_annotation_count": 0,
        }
    )
    article_families: dict[str, set[str]] = {}
    eligible_by_family = Counter({family: 0 for family in RELATION_FAMILIES})
    flattened_union_cardinality: Counter[int] = Counter()
    flattened_union_sentence_occurrence_count = 0

    for annotation in parsed.annotations:
        duplicate_query_excluded = (
            annotation.normalized_query_sha256
            in excluded_normalized_query_hashes
        )
        if duplicate_query_excluded:
            annotation_counts[
                "whole_duplicate_query_group_excluded_annotation_count"
            ] += 1
        if annotation.article_docid is not None:
            annotation_counts["single_article_annotation_count"] += 1
        complete_groups = 0
        group_signatures: set[str] = set()
        all_groups_nonduplicate = True
        for group in annotation.evidence_groups:
            group_counts["alternative_group_count"] += 1
            if group:
                group_counts["nonempty_group_count"] += 1
            signature_parts: list[tuple[str, int, int, int, int]] = []
            seen_spans: set[tuple[str, int, int, int, int]] = set()
            all_spans_valid = bool(group)
            for evidence in group:
                span_counts["evidence_span_count"] += 1
                document = documents[evidence.docid]
                token_valid = (
                    0 <= evidence.start_token < evidence.end_token
                    <= len(document.flattened_tokens)
                )
                sentence_valid = (
                    0 <= evidence.start_sentence < evidence.end_sentence
                    <= len(document.sentences)
                )
                contained = False
                if token_valid:
                    span_counts["token_span_valid_count"] += 1
                if sentence_valid:
                    span_counts["sentence_span_valid_count"] += 1
                if token_valid and sentence_valid:
                    contained = (
                        document.sentence_token_boundaries[
                            evidence.start_sentence
                        ]
                        <= evidence.start_token
                        and evidence.end_token
                        <= document.sentence_token_boundaries[
                            evidence.end_sentence
                        ]
                    )
                if contained:
                    span_counts[
                        "token_span_contained_by_sentence_span_count"
                    ] += 1
                    span_counts["fully_valid_span_count"] += 1
                else:
                    all_spans_valid = False
                if (
                    token_valid
                    and evidence.text_tokens is not None
                    and evidence.text_tokens
                    == document.flattened_tokens[
                        evidence.start_token : evidence.end_token
                    ]
                ):
                    span_counts[
                        "evidence_text_exact_token_slice_match_count"
                    ] += 1
                span_key = (
                    evidence.docid,
                    evidence.start_token,
                    evidence.end_token,
                    evidence.start_sentence,
                    evidence.end_sentence,
                )
                if span_key in seen_spans:
                    group_counts["duplicate_span_occurrence_count"] += 1
                    all_spans_valid = False
                seen_spans.add(span_key)
                signature_parts.append(span_key)
            group_signature = _stable_hash(sorted(signature_parts))
            if group_signature in group_signatures:
                group_counts["duplicate_alternative_group_occurrence_count"] += 1
                all_groups_nonduplicate = False
            group_signatures.add(group_signature)
            if all_spans_valid:
                complete_groups += 1
                group_counts["fully_complete_group_count"] += 1

        if complete_groups:
            annotation_counts[
                "at_least_one_complete_group_annotation_count"
            ] += 1
        all_groups_complete = (
            bool(annotation.evidence_groups)
            and complete_groups == len(annotation.evidence_groups)
            and all_groups_nonduplicate
        )
        if all_groups_complete:
            annotation_counts[
                "all_alternative_groups_complete_annotation_count"
            ] += 1
            flattened_sentence_union = {
                (evidence.docid, sentence_index)
                for group in annotation.evidence_groups
                for evidence in group
                for sentence_index in range(
                    evidence.start_sentence, evidence.end_sentence
                )
            }
            flattened_union_cardinality[len(flattened_sentence_union)] += 1
            flattened_union_sentence_occurrence_count += len(
                flattened_sentence_union
            )
        candidate_sentence_count_eligible = (
            annotation.article_docid is not None
            and len(documents[annotation.article_docid].sentences) >= 5
        )
        if (
            annotation.article_docid is not None
            and not candidate_sentence_count_eligible
        ):
            annotation_counts["fewer_than_five_sentence_annotation_count"] += 1
        if (
            all_groups_complete
            and annotation.article_docid is not None
            and not duplicate_query_excluded
            and candidate_sentence_count_eligible
        ):
            annotation_counts["capacity_eligible_annotation_count"] += 1
            eligible_by_family[annotation.classification] += 1
            article_families.setdefault(annotation.article_docid, set()).add(
                annotation.classification
            )

    content_hashes = {document.content_sha256 for document in documents.values()}
    member_hashes = {
        document.member_content_sha256 for document in documents.values()
    }
    sentence_cardinality = Counter(
        len(document.sentences) for document in documents.values()
    )
    token_cardinality = Counter(
        len(document.flattened_tokens) for document in documents.values()
    )
    article_counts_by_family = {
        family: sum(family in options for options in article_families.values())
        for family in RELATION_FAMILIES
    }
    option_cardinality = Counter(len(options) for options in article_families.values())
    aggregate = {
        "annotation_and_class_counts": {
            "annotation_count": len(parsed.annotations),
            "relation_family_counts": _string_counter(parsed.class_counts),
            "unique_annotation_identity_hash_count": len(
                parsed.annotation_identity_hashes
            ),
            "duplicate_annotation_identity_hash_count": (
                parsed.annotation_identity_duplicate_count
            ),
            "unique_query_hash_count": len(parsed.query_hashes),
            "duplicate_query_hash_count": parsed.query_duplicate_count,
        },
        "schema_hash_counts": {
            "annotation_keyset_sha256_counts": _string_counter(
                parsed.annotation_keyset_hash_counts
            ),
            "evidence_keyset_sha256_counts": _string_counter(
                parsed.evidence_keyset_hash_counts
            ),
            "evidence_text_representation_counts": _string_counter(
                parsed.evidence_text_representation_counts
            ),
            "evidence_group_cardinality_counts": _string_counter(
                parsed.evidence_group_cardinality_counts
            ),
        },
        "referenced_document_aggregates": {
            "referenced_document_count": len(documents),
            "unique_document_content_sha256_count": len(content_hashes),
            "duplicate_document_content_hash_occurrence_count": (
                len(documents) - len(content_hashes)
            ),
            "unique_document_member_sha256_count": len(member_hashes),
            "duplicate_document_member_hash_occurrence_count": (
                len(documents) - len(member_hashes)
            ),
            "sentence_cardinality_counts": _string_counter(sentence_cardinality),
            "token_cardinality_counts": _string_counter(token_cardinality),
        },
        "evidence_span_completeness": dict(span_counts),
        "alternative_group_completeness": dict(group_counts),
        "annotation_completeness": dict(annotation_counts),
        "gold_flattened_rationale_semantics": {
            "all_alternative_evidence_groups_sentence_span_union_used": True,
            "best_group_or_single_group_selection_used": False,
            "complete_annotation_union_sentence_occurrence_count": (
                flattened_union_sentence_occurrence_count
            ),
            "complete_annotation_union_sentence_cardinality_counts": (
                _string_counter(flattened_union_cardinality)
            ),
        },
        "capacity_inputs": {
            "eligible_annotation_counts_by_family": _string_counter(
                eligible_by_family
            ),
            "eligible_unique_article_counts_by_family": _string_counter(
                article_counts_by_family
            ),
            "eligible_article_family_option_cardinality_counts": _string_counter(
                option_cardinality
            ),
            "eligible_unique_article_count": len(article_families),
        },
    }
    return aggregate, article_families


class _FlowNetwork:
    def __init__(self, node_count: int) -> None:
        self.graph: list[list[list[int]]] = [[] for _ in range(node_count)]

    def add_edge(self, source: int, target: int, capacity: int) -> None:
        forward = [target, capacity, len(self.graph[target])]
        reverse = [source, 0, len(self.graph[source])]
        self.graph[source].append(forward)
        self.graph[target].append(reverse)

    def maximum_flow(self, source: int, sink: int) -> int:
        total = 0
        while True:
            levels = [-1] * len(self.graph)
            levels[source] = 0
            queue: deque[int] = deque([source])
            while queue:
                node = queue.popleft()
                for target, capacity, _reverse in self.graph[node]:
                    if capacity > 0 and levels[target] < 0:
                        levels[target] = levels[node] + 1
                        queue.append(target)
            if levels[sink] < 0:
                return total
            positions = [0] * len(self.graph)

            def augment(node: int, available: int) -> int:
                if node == sink:
                    return available
                while positions[node] < len(self.graph[node]):
                    edge = self.graph[node][positions[node]]
                    target, capacity, reverse_index = edge
                    if capacity > 0 and levels[target] == levels[node] + 1:
                        pushed = augment(target, min(available, capacity))
                        if pushed:
                            edge[1] -= pushed
                            self.graph[target][reverse_index][1] += pushed
                            return pushed
                    positions[node] += 1
                return 0

            while True:
                pushed = augment(source, 1 << 60)
                if not pushed:
                    break
                total += pushed


def _capacity_receipt(
    article_families: Mapping[str, set[str]],
    cohort_demands: Mapping[str, int],
) -> dict[str, Any]:
    buckets = [
        (family, cohort)
        for family in RELATION_FAMILIES
        for cohort in sorted(cohort_demands)
    ]
    articles = sorted(article_families)
    source = 0
    article_offset = 1
    bucket_offset = article_offset + len(articles)
    sink = bucket_offset + len(buckets)
    network = _FlowNetwork(sink + 1)
    for index, article in enumerate(articles):
        node = article_offset + index
        network.add_edge(source, node, 1)
        for bucket_index, (family, _cohort) in enumerate(buckets):
            if family in article_families[article]:
                network.add_edge(node, bucket_offset + bucket_index, 1)
    for bucket_index, (_family, cohort) in enumerate(buckets):
        network.add_edge(
            bucket_offset + bucket_index,
            sink,
            int(cohort_demands[cohort]),
        )
    maximum = network.maximum_flow(source, sink)
    total_per_family = sum(cohort_demands.values())
    total_demand = total_per_family * len(RELATION_FAMILIES)

    neighborhood_counts: dict[str, int] = {}
    hall_shortfalls: dict[str, int] = {}
    for width in range(1, len(RELATION_FAMILIES) + 1):
        for subset in itertools.combinations(RELATION_FAMILIES, width):
            label = "+".join(subset)
            neighborhood = sum(
                bool(options.intersection(subset))
                for options in article_families.values()
            )
            required = total_per_family * len(subset)
            neighborhood_counts[label] = neighborhood
            hall_shortfalls[label] = max(0, required - neighborhood)

    return {
        "cohort_demands_per_relation_family": _string_counter(cohort_demands),
        "relation_family_count": len(RELATION_FAMILIES),
        "total_demand_per_relation_family": total_per_family,
        "total_article_disjoint_demand": total_demand,
        "maximum_article_disjoint_assignment_count": maximum,
        "article_disjoint_assignment_shortfall_count": total_demand - maximum,
        "hall_neighborhood_article_counts": neighborhood_counts,
        "hall_shortfall_counts": hall_shortfalls,
        "exact_article_disjoint_capacity_met": maximum == total_demand,
    }


def qualify_archive(
    archive_path: Path,
    custody_manifest_path: Path,
    access_manifest_path: Path,
    prompt_sidecar_path: Path,
    prompt_access_manifest_path: Path,
    tar_header_amendment_path: Path,
    design_amendment_path: Path,
    hipporag_implementation_freeze_path: Path,
    *,
    expected_archive_sha256: str,
    expected_archive_size: int,
    expected_prompt_sidecar_sha256: str,
    expected_prompt_sidecar_size: int,
    expected_prompt_sidecar_git_blob_sha1: str,
    expected_prompt_sidecar_git_commit: str,
    expected_annotation_counts: Mapping[str, int],
    expected_article_counts: Mapping[str, int],
    cohort_demands: Mapping[str, Mapping[str, int]] = FORMAL_COHORT_DEMANDS,
    enforce_formal_manifest_identity: bool = False,
) -> dict[str, Any]:
    """Qualify one byte-bound archive without selecting or exposing any row."""

    if not _SHA256_RE.fullmatch(expected_archive_sha256):
        raise EraserEvidenceInferenceQualificationError(
            "expected whole-archive SHA-256 is invalid"
        )
    if type(expected_archive_size) is not int or expected_archive_size <= 0:
        raise EraserEvidenceInferenceQualificationError(
            "expected whole-archive size is invalid"
        )
    if (
        not _SHA256_RE.fullmatch(expected_prompt_sidecar_sha256)
        or type(expected_prompt_sidecar_size) is not int
        or expected_prompt_sidecar_size <= 0
        or not _SHA1_RE.fullmatch(expected_prompt_sidecar_git_blob_sha1)
        or not re.fullmatch(r"[0-9a-f]{40}", expected_prompt_sidecar_git_commit)
    ):
        raise EraserEvidenceInferenceQualificationError(
            "expected prompt sidecar binding is invalid"
        )
    if set(expected_annotation_counts) != {"train", "val"} or set(
        expected_article_counts
    ) != {"train", "val"}:
        raise EraserEvidenceInferenceQualificationError(
            "expected aggregate split registry drifted"
        )
    if set(cohort_demands) != {"train", "val"} or any(
        not demands
        or any(type(value) is not int or value <= 0 for value in demands.values())
        for demands in cohort_demands.values()
    ):
        raise EraserEvidenceInferenceQualificationError(
            "cohort demand registry drifted"
        )

    manifest_bindings = _validate_manifests(
        custody_manifest_path,
        access_manifest_path,
        expected_archive_sha256=expected_archive_sha256,
        expected_archive_size=expected_archive_size,
        enforce_formal_manifest_identity=enforce_formal_manifest_identity,
    )
    prompt_manifest_binding = _validate_prompt_access_manifest(
        prompt_access_manifest_path,
        custody_self_sha256=manifest_bindings["custody_self_sha256"],
        expected_sidecar_sha256=expected_prompt_sidecar_sha256,
        expected_sidecar_size=expected_prompt_sidecar_size,
        expected_git_blob_sha1=expected_prompt_sidecar_git_blob_sha1,
        expected_git_commit=expected_prompt_sidecar_git_commit,
        enforce_formal_manifest_identity=enforce_formal_manifest_identity,
    )
    container_amendment_binding = _validate_container_amendments(
        tar_header_amendment_path,
        design_amendment_path,
        archive_sha256=expected_archive_sha256,
        custody_self_sha256=manifest_bindings["custody_self_sha256"],
        access_self_sha256=manifest_bindings["access_self_sha256"],
        enforce_formal_manifest_identity=enforce_formal_manifest_identity,
    )
    hipporag_freeze_binding = _validate_hipporag_implementation_freeze(
        hipporag_implementation_freeze_path,
        enforce_formal_manifest_identity=enforce_formal_manifest_identity,
    )
    observed_archive_hash, observed_archive_size = _sha256_file(archive_path)
    if (
        observed_archive_hash != expected_archive_sha256
        or observed_archive_size != expected_archive_size
    ):
        raise EraserEvidenceInferenceQualificationError(
            "whole official archive identity drifted"
        )

    split_raw, split_root, header_counts = _read_split_members(archive_path)
    parsed = {
        split: _parse_jsonl(split_raw[split], split=split)
        for split in ("train", "val")
    }
    for split in ("train", "val"):
        if len(parsed[split].annotations) != expected_annotation_counts[split]:
            raise EraserEvidenceInferenceQualificationError(
                "authorized split annotation count drifted"
            )
        if len(parsed[split].referenced_docids) != expected_article_counts[split]:
            raise EraserEvidenceInferenceQualificationError(
                "authorized split referenced article count drifted"
            )
    prompt_to_article: dict[str, str] = {}
    prompt_to_split: dict[str, str] = {}
    for split in ("train", "val"):
        for annotation in parsed[split].annotations:
            if annotation.article_docid is None:
                raise EraserEvidenceInferenceQualificationError(
                    "authorized PromptID lacks one unambiguous article binding"
                )
            if annotation.prompt_id in prompt_to_article:
                raise EraserEvidenceInferenceQualificationError(
                    "PromptID overlaps authorized train and validation splits"
                )
            prompt_to_article[annotation.prompt_id] = annotation.article_docid
            prompt_to_split[annotation.prompt_id] = split

    observed_sidecar_hash, observed_sidecar_size = _sha256_file(
        prompt_sidecar_path
    )
    if (
        observed_sidecar_hash != expected_prompt_sidecar_sha256
        or observed_sidecar_size != expected_prompt_sidecar_size
    ):
        raise EraserEvidenceInferenceQualificationError(
            "whole prompt sidecar identity drifted"
        )
    structured_prompt_receipt = _stream_prompt_sidecar(
        prompt_sidecar_path,
        prompt_to_article=prompt_to_article,
        prompt_to_split=prompt_to_split,
    )
    cross_split_overlap = len(
        parsed["train"].referenced_docids.intersection(
            parsed["val"].referenced_docids
        )
    )

    all_referenced = frozenset(
        parsed["train"].referenced_docids
        | parsed["val"].referenced_docids
    )
    normalized_query_counts = Counter(
        annotation.normalized_query_sha256
        for split in ("train", "val")
        for annotation in parsed[split].annotations
    )
    excluded_normalized_query_hashes = frozenset(
        digest for digest, count in normalized_query_counts.items() if count > 1
    )
    excluded_normalized_query_annotation_count = sum(
        normalized_query_counts[digest]
        for digest in excluded_normalized_query_hashes
    )
    documents = _read_referenced_documents(
        archive_path,
        split_root=split_root,
        referenced_docids=all_referenced,
    )
    split_receipts: dict[str, Any] = {}
    split_capacity_inputs: dict[str, dict[str, set[str]]] = {}
    for split in ("train", "val"):
        split_documents = {
            docid: documents[docid] for docid in parsed[split].referenced_docids
        }
        split_receipts[split], split_capacity_inputs[split] = _verify_split(
            parsed[split],
            split_documents,
            excluded_normalized_query_hashes=excluded_normalized_query_hashes,
        )

    capacities = {
        split: _capacity_receipt(
            split_capacity_inputs[split], cohort_demands[split]
        )
        for split in ("train", "val")
    }
    passed = (
        cross_split_overlap == 0
        and capacities["train"]["exact_article_disjoint_capacity_met"]
        and capacities["val"]["exact_article_disjoint_capacity_met"]
    )
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": (
            "passed_source_qualification_no_selection"
            if passed
            else "terminal_source_infeasible_no_selection"
        ),
        "source_binding": {
            "whole_archive_sha256": observed_archive_hash,
            "whole_archive_size": observed_archive_size,
            "custody_manifest_file_sha256": manifest_bindings[
                "custody_file_sha256"
            ],
            "custody_manifest_self_sha256": manifest_bindings[
                "custody_self_sha256"
            ],
            "access_manifest_file_sha256": manifest_bindings[
                "access_file_sha256"
            ],
            "access_manifest_self_sha256": manifest_bindings[
                "access_self_sha256"
            ],
            "prompt_sidecar_sha256": observed_sidecar_hash,
            "prompt_sidecar_size": observed_sidecar_size,
            "prompt_sidecar_git_blob_sha1": (
                expected_prompt_sidecar_git_blob_sha1
            ),
            "prompt_sidecar_git_commit": expected_prompt_sidecar_git_commit,
            "prompt_access_manifest_file_sha256": prompt_manifest_binding[
                "file_sha256"
            ],
            "prompt_access_manifest_self_sha256": prompt_manifest_binding[
                "self_sha256"
            ],
            **container_amendment_binding,
            **hipporag_freeze_binding,
            "formal_manifest_identity_enforced": enforce_formal_manifest_identity,
        },
        "archive_header_aggregates": {
            "regular_member_count": header_counts["regular"],
            "directory_member_count": header_counts["directory"],
            "nonregular_nondirectory_member_count": header_counts["other"],
        },
        "opened_content_boundary": {
            "authorized_split_member_count": 2,
            "nonpersistent_in_memory_tar_header_routing_used": True,
            "referenced_document_member_count": len(all_referenced),
            "test_member_content_open_count": 0,
            "unreferenced_document_content_open_count": 0,
            "member_name_or_path_emitted_count": 0,
        },
        "cross_split_article_disjointness": {
            "train_referenced_article_count": len(
                parsed["train"].referenced_docids
            ),
            "validation_referenced_article_count": len(
                parsed["val"].referenced_docids
            ),
            "train_validation_article_overlap_count": cross_split_overlap,
            "article_disjoint": cross_split_overlap == 0,
        },
        "duplicate_normalized_query_group_exclusion": {
            "normalization": "Unicode_NFKC_then_whitespace_collapse_then_casefold",
            "duplicate_group_count": len(excluded_normalized_query_hashes),
            "excluded_annotation_count": excluded_normalized_query_annotation_count,
            "excluded_group_or_query_value_emitted": False,
        },
        "independent_structured_prompt_binding": structured_prompt_receipt,
        "split_aggregates": split_receipts,
        "article_disjoint_capacity": capacities,
        "claim_boundary": {
            "selection_secret_opened_or_generated": False,
            "cohort_selected": False,
            "retrieval_action_evaluator_or_score_run": False,
            "online_or_network_evaluation_used": False,
            "test_member_query_document_label_or_content_opened": False,
            "per_row_or_per_item_hash_emitted": False,
            "private_identifier_query_document_or_evidence_value_emitted": False,
            "query_string_heuristically_parsed_into_ico_fields": False,
            "official_prompt_sidecar_exact_binding_used": True,
        },
    }
    body["qualification_sha256"] = _stable_hash(body)
    return body


def build_formal_qualification(project: Path) -> dict[str, Any]:
    """Run the fixed formal binding relative to the reconstruction_v2 root."""

    root = project.resolve(strict=True)
    archive = root / FORMAL_ARCHIVE_RELATIVE_PATH
    custody = root / FORMAL_CUSTODY_RELATIVE_PATH
    access = root / FORMAL_ACCESS_RELATIVE_PATH
    prompt_sidecar = root / FORMAL_PROMPT_SIDECAR_RELATIVE_PATH
    prompt_access = root / FORMAL_PROMPT_ACCESS_RELATIVE_PATH
    tar_header_amendment = root / FORMAL_TAR_HEADER_AMENDMENT_RELATIVE_PATH
    design_amendment = root / FORMAL_DESIGN_AMENDMENT_RELATIVE_PATH
    hipporag_freeze = root / FORMAL_HIPPORAG_FREEZE_RELATIVE_PATH
    for path in (
        archive,
        custody,
        access,
        prompt_sidecar,
        prompt_access,
        tar_header_amendment,
        design_amendment,
        hipporag_freeze,
    ):
        if path.is_symlink():
            raise EraserEvidenceInferenceQualificationError(
                "formal source binding may not be a symlink"
            )
        try:
            mode = path.stat().st_mode
        except OSError as exc:
            raise EraserEvidenceInferenceQualificationError(
                "formal source binding is unavailable"
            ) from exc
        if not stat.S_ISREG(mode):
            raise EraserEvidenceInferenceQualificationError(
                "formal source binding is not a regular file"
            )
    if stat.S_IMODE(archive.stat().st_mode) != 0o600:
        raise EraserEvidenceInferenceQualificationError(
            "formal archive mode must remain 0600"
        )
    return qualify_archive(
        archive,
        custody,
        access,
        prompt_sidecar,
        prompt_access,
        tar_header_amendment,
        design_amendment,
        hipporag_freeze,
        expected_archive_sha256=FORMAL_ARCHIVE_SHA256,
        expected_archive_size=FORMAL_ARCHIVE_SIZE,
        expected_prompt_sidecar_sha256=FORMAL_PROMPT_SIDECAR_SHA256,
        expected_prompt_sidecar_size=FORMAL_PROMPT_SIDECAR_SIZE,
        expected_prompt_sidecar_git_blob_sha1=(
            FORMAL_PROMPT_SIDECAR_GIT_BLOB_SHA1
        ),
        expected_prompt_sidecar_git_commit=FORMAL_PROMPT_SIDECAR_GIT_COMMIT,
        expected_annotation_counts=FORMAL_EXPECTED_ANNOTATION_COUNTS,
        expected_article_counts=FORMAL_EXPECTED_ARTICLE_COUNTS,
        cohort_demands=FORMAL_COHORT_DEMANDS,
        enforce_formal_manifest_identity=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate-only ERASER Evidence Inference source qualifier"
    )
    parser.add_argument(
        "--project",
        type=Path,
        required=True,
        help="reconstruction_v2 project root",
    )
    args = parser.parse_args(argv)
    receipt = build_formal_qualification(args.project)
    sys.stdout.buffer.write(_canonical_json(receipt) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
