"""One-shot, aggregate-only qualification of the frozen MoreHopQA source.

The formal CLI accepts only ``--project``.  It reads one canonical ignored
source, consumes one ignored attempt marker before parsing, and exclusively
publishes one canonical public manifest.  No row, identifier, question,
answer, title, paragraph, support title, or per-row digest is serialized.

Tests must use :func:`qualify_payload` or :func:`build_synthetic_aggregate`;
they must never open the formal source.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Any
import unicodedata


VERSION = "morehopqa_source_qualification_v1"
SCHEMA = VERSION
QUALIFICATION_CLASS = "aggregate_only_morehopqa_source_qualification"

FORMAL_SOURCE_RELATIVE_PATH = Path(
    "artifacts/morehopqa_official_source_v1/"
    "with_human_verification-e839079e.json"
)
FORMAL_ATTEMPT_MARKER_RELATIVE_PATH = Path(
    "artifacts/morehopqa_official_source_v1/"
    "source_qualification_attempt_v1.marker"
)
FORMAL_EXCLUSION_RELATIVE_PATH = Path(
    "artifacts/morehopqa_official_exclusion_v1/"
    "morehopqa_final_150samples-27c72b1a.json"
)
FORMAL_OUTPUT_RELATIVE_PATH = Path(
    "manifests/morehopqa_source_qualification_v1.json"
)

FORMAL_SOURCE_SIZE = 4_230_947
FORMAL_SOURCE_SHA256 = (
    "41b1c31af2546f005fd699148bc1ae68968349179941f0218ef7975596489f4a"
)
FORMAL_SOURCE_GIT_BLOB_SHA1 = "7b596112e906217dcc096c199642f3bca2299fb9"
FORMAL_ROOT_COUNT = 1_118
FORMAL_EXCLUSION_SIZE = 603_971
FORMAL_EXCLUSION_SHA256 = (
    "6b7f67153c6fc63425f4da3025bdb775d5265ca3a2107cd6838b01d41d6c05dc"
)
FORMAL_EXCLUSION_GIT_BLOB_SHA1 = "847ea6804062d720b83e56cf1a7d97372ecf2357"
FORMAL_EXCLUSION_ROOT_COUNT = 150
FORMAL_EXCLUSION_MANIFEST_SELF_SHA256 = (
    "0cfc2ce644cc4ff9fa7b2c558540c8b4967f725d99b03523e853945ee033f1b5"
)
FORMAL_HUGGINGFACE_COMMIT = "e839079eb8eb686ecf42c6d334a87964f1b90eef"
FORMAL_GITHUB_COMMIT = "27c72b1a220255093266a61f9e70af6ae981dc0b"
TARGET_PER_REASONING_FAMILY = 72

REASONING_TOKEN_ORDER = ("Symbolic", "Arithmetic", "Commonsense")
REASONING_TOKEN_SET_LABELS = (
    "Symbolic",
    "Arithmetic",
    "Commonsense",
    "Symbolic+Arithmetic",
    "Symbolic+Commonsense",
    "Arithmetic+Commonsense",
    "Symbolic+Arithmetic+Commonsense",
)
ANSWER_TYPES = ("date", "number", "string", "letter")

PUBLIC_ROOT_FIELDS = frozenset(
    {
        "_id",
        "question",
        "answer",
        "previous_question",
        "previous_answer",
        "question_decomposition",
        "context",
        "answer_type",
        "previous_answer_type",
        "no_of_hops",
        "reasoning_type",
        "pattern",
        "subquestion_patterns",
        "cutted_question",
        "ques_on_last_hop",
    }
)
PUBLIC_DECOMPOSITION_FIELDS = frozenset(
    {"sub_id", "question", "answer", "paragraph_support_title"}
)

_STRING_ROOT_FIELDS = (
    "_id",
    "question",
    "answer",
    "previous_question",
    "previous_answer",
    "answer_type",
    "previous_answer_type",
    "reasoning_type",
    "pattern",
    "cutted_question",
    "ques_on_last_hop",
)
_REASONING_RE = re.compile(
    r"(?<![0-9A-Za-z_])(?:Symbolic|Arithmetic|Commonsense)"
    r"(?![0-9A-Za-z_])"
)
_WHITESPACE_RE = re.compile(r"\s+")


class MoreHopQASourceQualificationError(RuntimeError):
    """The frozen source or qualification lifecycle violated its contract."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {len(raw)}\0".encode("ascii"))
    digest.update(raw)
    return digest.hexdigest()


def _decode_strict_json(raw: bytes) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MoreHopQASourceQualificationError(
            "source is not strict UTF-8"
        ) from exc

    def object_hook(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MoreHopQASourceQualificationError(
                    "duplicate JSON object key"
                )
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise MoreHopQASourceQualificationError(
            "JSON contains a non-finite constant"
        )

    try:
        return json.loads(
            text,
            object_pairs_hook=object_hook,
            parse_constant=reject_constant,
        )
    except MoreHopQASourceQualificationError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise MoreHopQASourceQualificationError(
            "source is not strict JSON"
        ) from exc


def _normalize_text(value: str) -> str:
    return _WHITESPACE_RE.sub(
        " ", unicodedata.normalize("NFKC", value)
    ).strip()


def normalize_question(value: str) -> str:
    """Frozen normalized-question identity used only for uniqueness counts."""

    return _normalize_text(value).casefold()


def _normalize_title_key(value: str) -> str:
    return _normalize_text(value).casefold()


def _document_identity(title: str, paragraphs: Sequence[str]) -> str:
    return _sha256_json(
        {
            "title": _normalize_text(title),
            "paragraphs": [_normalize_text(value) for value in paragraphs],
        }
    )


def _reasoning_parse(value: str) -> tuple[tuple[str, ...] | None, int, int]:
    matches = list(_REASONING_RE.finditer(value))
    tokens = [match.group(0) for match in matches]
    masked = list(value)
    for match in matches:
        masked[match.start() : match.end()] = " " * (match.end() - match.start())

    unknown_residue_count = 0
    for character in masked:
        if character.isspace():
            continue
        category = unicodedata.category(character)
        if category[:1] not in {"P", "S", "Z"}:
            unknown_residue_count += 1

    duplicate_count = len(tokens) - len(set(tokens))
    if not tokens:
        return None, unknown_residue_count, duplicate_count
    ordered = tuple(token for token in REASONING_TOKEN_ORDER if token in tokens)
    return ordered, unknown_residue_count, duplicate_count


def _token_set_label(tokens: Sequence[str]) -> str:
    return "+".join(tokens)


def _require_string(value: Any) -> str:
    if not isinstance(value, str):
        raise MoreHopQASourceQualificationError(
            "public source field has the wrong type"
        )
    return value


def _require_string_list(value: Any) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(element, str) for element in value
    ):
        raise MoreHopQASourceQualificationError(
            "public source field has the wrong type"
        )
    return value


def _counter_mapping(counter: Counter[int]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter)}


def _name_set_hash(names: set[str]) -> str:
    return _sha256_json({"names": sorted(names)})


def _strict_exclusion_ids(payload: Any) -> frozenset[str]:
    if not isinstance(payload, list) or len(payload) != FORMAL_EXCLUSION_ROOT_COUNT:
        raise MoreHopQASourceQualificationError(
            "official exclusion root must contain exactly 150 entries"
        )
    identifiers: set[str] = set()
    for row in payload:
        if not isinstance(row, Mapping):
            raise MoreHopQASourceQualificationError(
                "official exclusion entry must be an object"
            )
        identifier = _require_string(row.get("_id"))
        if not identifier or identifier in identifiers:
            raise MoreHopQASourceQualificationError(
                "official exclusion identifiers must be nonempty and unique"
            )
        identifiers.add(identifier)
    return frozenset(identifiers)


def qualify_payload(
    payload: Any,
    *,
    source_size: int,
    source_sha256: str,
    source_git_blob_sha1: str,
    formal_source_identity_enforced: bool,
    exclusion_ids: frozenset[str],
    exclusion_size: int,
    exclusion_sha256: str,
    exclusion_git_blob_sha1: str,
    formal_exclusion_identity_enforced: bool,
) -> dict[str, Any]:
    """Validate public raw schema and return aggregate-only statistics."""

    if not isinstance(payload, list):
        raise MoreHopQASourceQualificationError("source root must be a JSON array")
    if len(payload) != FORMAL_ROOT_COUNT:
        raise MoreHopQASourceQualificationError(
            "source root count is not the frozen 1118"
        )

    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    duplicate_id_count = 0
    duplicate_question_count = 0
    reasoning_counts = Counter({label: 0 for label in REASONING_TOKEN_SET_LABELS})
    answer_type_counts = Counter({label: 0 for label in ANSWER_TYPES})
    unknown_answer_type_count = 0
    reasoning_unknown_residue_count = 0
    reasoning_duplicate_token_count = 0
    reasoning_missing_token_count = 0
    context_cardinality = Counter()
    decomposition_cardinality = Counter()
    gold_cardinality = Counter()
    all_document_identities: set[str] = set()
    all_normalized_titles: set[str] = set()
    total_context_documents = 0
    total_decomposition_entries = 0
    nonempty_support_reference_count = 0
    empty_support_reference_count = 0
    exactly_resolved_support_reference_count = 0
    missing_support_reference_count = 0
    ambiguous_support_reference_count = 0
    fully_resolved_item_count = 0
    zero_nonempty_support_item_count = 0
    item_with_duplicate_normalized_context_title_count = 0
    root_rows_with_extra_fields = 0
    root_extra_field_occurrences = 0
    root_extra_field_names: set[str] = set()
    decomposition_rows_with_extra_fields = 0
    decomposition_extra_field_occurrences = 0
    decomposition_extra_field_names: set[str] = set()
    eligible_token_sets_by_question: dict[str, list[frozenset[str]]] = defaultdict(list)
    matched_exclusion_ids: set[str] = set()
    excluded_structurally_eligible_item_count = 0

    for row in payload:
        if not isinstance(row, Mapping):
            raise MoreHopQASourceQualificationError(
                "source array entry must be an object"
            )
        if not PUBLIC_ROOT_FIELDS.issubset(row):
            raise MoreHopQASourceQualificationError(
                "source entry is missing a public root field"
            )

        extras = set(row) - PUBLIC_ROOT_FIELDS
        if extras:
            root_rows_with_extra_fields += 1
            root_extra_field_occurrences += len(extras)
            root_extra_field_names.update(extras)

        for field in _STRING_ROOT_FIELDS:
            _require_string(row[field])
        _require_string_list(row["subquestion_patterns"])
        if not isinstance(row["no_of_hops"], int) or isinstance(
            row["no_of_hops"], bool
        ):
            raise MoreHopQASourceQualificationError(
                "public source field has the wrong type"
            )

        identifier = row["_id"]
        is_officially_excluded = identifier in exclusion_ids
        if is_officially_excluded:
            matched_exclusion_ids.add(identifier)
        normalized = normalize_question(row["question"])
        if not identifier or not normalized:
            raise MoreHopQASourceQualificationError(
                "source identity field is empty"
            )
        if identifier in seen_ids:
            duplicate_id_count += 1
        else:
            seen_ids.add(identifier)
        if normalized in seen_questions:
            duplicate_question_count += 1
        else:
            seen_questions.add(normalized)

        answer_type = row["answer_type"]
        if answer_type in answer_type_counts:
            answer_type_counts[answer_type] += 1
        else:
            unknown_answer_type_count += 1

        tokens, residue_count, token_duplicate_count = _reasoning_parse(
            row["reasoning_type"]
        )
        reasoning_unknown_residue_count += residue_count
        reasoning_duplicate_token_count += token_duplicate_count
        if tokens is None:
            reasoning_missing_token_count += 1
        elif residue_count == 0 and token_duplicate_count == 0:
            reasoning_counts[_token_set_label(tokens)] += 1

        context = row["context"]
        if not isinstance(context, list) or not context:
            raise MoreHopQASourceQualificationError(
                "context must be a nonempty sequence"
            )
        context_cardinality[len(context)] += 1
        total_context_documents += len(context)
        local_title_to_documents: dict[str, list[str]] = defaultdict(list)
        for document in context:
            if not isinstance(document, list) or len(document) != 2:
                raise MoreHopQASourceQualificationError(
                    "context document must be [title, paragraphs]"
                )
            title = _require_string(document[0])
            paragraphs = _require_string_list(document[1])
            if not _normalize_title_key(title) or not paragraphs:
                raise MoreHopQASourceQualificationError(
                    "context title and paragraphs must be nonempty"
                )
            identity = _document_identity(title, paragraphs)
            title_key = _normalize_title_key(title)
            local_title_to_documents[title_key].append(identity)
            all_document_identities.add(identity)
            all_normalized_titles.add(title_key)
        if any(len(documents) > 1 for documents in local_title_to_documents.values()):
            item_with_duplicate_normalized_context_title_count += 1

        decomposition = row["question_decomposition"]
        if not isinstance(decomposition, list) or not decomposition:
            raise MoreHopQASourceQualificationError(
                "question_decomposition must be a nonempty sequence"
            )
        decomposition_cardinality[len(decomposition)] += 1
        total_decomposition_entries += len(decomposition)
        resolved_document_identities: set[str] = set()
        item_missing = False
        item_ambiguous = False
        item_nonempty_support_count = 0
        for step in decomposition:
            if not isinstance(step, Mapping):
                raise MoreHopQASourceQualificationError(
                    "decomposition entry must be an object"
                )
            if not PUBLIC_DECOMPOSITION_FIELDS.issubset(step):
                raise MoreHopQASourceQualificationError(
                    "decomposition entry is missing a public field"
                )
            step_extras = set(step) - PUBLIC_DECOMPOSITION_FIELDS
            if step_extras:
                decomposition_rows_with_extra_fields += 1
                decomposition_extra_field_occurrences += len(step_extras)
                decomposition_extra_field_names.update(step_extras)
            for field in PUBLIC_DECOMPOSITION_FIELDS:
                _require_string(step[field])

            support_key = _normalize_title_key(step["paragraph_support_title"])
            if not support_key:
                empty_support_reference_count += 1
                continue
            item_nonempty_support_count += 1
            nonempty_support_reference_count += 1
            matches = local_title_to_documents.get(support_key, [])
            if len(matches) == 1:
                exactly_resolved_support_reference_count += 1
                resolved_document_identities.add(matches[0])
            elif not matches:
                missing_support_reference_count += 1
                item_missing = True
            else:
                ambiguous_support_reference_count += 1
                item_ambiguous = True

        if item_nonempty_support_count == 0:
            zero_nonempty_support_item_count += 1
        exactly_resolved_item = (
            item_nonempty_support_count > 0
            and not item_missing
            and not item_ambiguous
            and bool(resolved_document_identities)
        )
        if exactly_resolved_item:
            fully_resolved_item_count += 1
            gold_cardinality[len(resolved_document_identities)] += 1
            if is_officially_excluded:
                excluded_structurally_eligible_item_count += 1
            elif tokens is not None and residue_count == 0 and token_duplicate_count == 0:
                eligible_token_sets_by_question[normalized].append(frozenset(tokens))

    if duplicate_id_count:
        raise MoreHopQASourceQualificationError(
            "source identity uniqueness failed: "
            f"duplicate_id_count={duplicate_id_count}"
        )
    if unknown_answer_type_count:
        raise MoreHopQASourceQualificationError(
            "unknown answer type count is nonzero"
        )
    unmatched_exclusion_count = len(exclusion_ids - matched_exclusion_ids)
    if unmatched_exclusion_count:
        raise MoreHopQASourceQualificationError(
            "official exclusion contains identifiers absent from the source: "
            f"unmatched_count={unmatched_exclusion_count}"
        )
    if (
        reasoning_unknown_residue_count
        or reasoning_duplicate_token_count
        or reasoning_missing_token_count
    ):
        raise MoreHopQASourceQualificationError(
            "reasoning registry validation failed: "
            f"unknown_residue_count={reasoning_unknown_residue_count}; "
            f"duplicate_token_count={reasoning_duplicate_token_count}; "
            f"missing_token_count={reasoning_missing_token_count}"
        )

    # Cohort selection retains one private-HMAC representative per normalized
    # question before matching.  Qualification therefore counts each question
    # once and uses the intersection of duplicate rows' token sets.  This is a
    # conservative guarantee: every possible retained representative supports
    # each token in the intersection.  Exact three-family b-matching capacity is
    # then characterized by the seven Hall neighborhoods.
    eligible_token_set_counts: Counter[str] = Counter()
    for token_sets in eligible_token_sets_by_question.values():
        conservative = set(token_sets[0])
        for token_set in token_sets[1:]:
            conservative.intersection_update(token_set)
        if conservative:
            ordered = tuple(
                token for token in REASONING_TOKEN_ORDER if token in conservative
            )
            eligible_token_set_counts[_token_set_label(ordered)] += 1
    hall_neighborhood_counts: dict[str, int] = {}
    hall_shortfalls: dict[str, int] = {}
    for width in range(1, len(REASONING_TOKEN_ORDER) + 1):
        for subset in itertools.combinations(REASONING_TOKEN_ORDER, width):
            subset_set = set(subset)
            label = _token_set_label(subset)
            neighborhood = sum(
                count
                for token_label, count in eligible_token_set_counts.items()
                if set(token_label.split("+")) & subset_set
            )
            hall_neighborhood_counts[label] = neighborhood
            hall_shortfalls[label] = max(
                0, TARGET_PER_REASONING_FAMILY * width - neighborhood
            )
    capacity_ok = all(value == 0 for value in hall_shortfalls.values())
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "version": "v1",
        "qualification_class": QUALIFICATION_CLASS,
        "qualification_status": (
            "source_qualified_for_exact_three_family_b_matching_capacity"
            if capacity_ok
            else "terminal_source_infeasible_for_three_family_b_matching_capacity"
        ),
        "aggregate_only": {
            "item_or_field_value_emitted": False,
            "per_item_hash_or_identifier_emitted": False,
            "selection_secret_or_cohort_marker_read_or_generated": False,
            "retrieval_model_action_or_score_performed": False,
        },
        "source_binding": {
            "formal_source_identity_enforced": formal_source_identity_enforced,
            "local_relative_path": FORMAL_SOURCE_RELATIVE_PATH.as_posix(),
            "size": source_size,
            "sha256": source_sha256,
            "git_blob_sha1": source_git_blob_sha1,
            "huggingface_commit": FORMAL_HUGGINGFACE_COMMIT,
            "github_commit": FORMAL_GITHUB_COMMIT,
        },
        "official_public_example_exclusion": {
            "formal_exclusion_identity_enforced": (
                formal_exclusion_identity_enforced
            ),
            "local_relative_path": FORMAL_EXCLUSION_RELATIVE_PATH.as_posix(),
            "size": exclusion_size,
            "sha256": exclusion_sha256,
            "git_blob_sha1": exclusion_git_blob_sha1,
            "official_git_commit": FORMAL_GITHUB_COMMIT,
            "public_manifest_self_sha256": (
                FORMAL_EXCLUSION_MANIFEST_SELF_SHA256
            ),
            "deny_id_count": len(exclusion_ids),
            "matched_source_item_count": len(matched_exclusion_ids),
            "unmatched_exclusion_id_count": 0,
            "deny_id_set_sha256": _sha256_json(
                {"ids": sorted(exclusion_ids)}
            ),
            "excluded_structurally_eligible_item_count": (
                excluded_structurally_eligible_item_count
            ),
            "capacity_counts_exclude_every_deny_id": True,
        },
        "parser_and_schema": {
            "strict_utf8_json_duplicate_keys_and_nonfinite_constants_rejected": True,
            "expected_root_count": FORMAL_ROOT_COUNT,
            "observed_root_count": len(payload),
            "public_root_field_count": len(PUBLIC_ROOT_FIELDS),
            "public_decomposition_field_count": len(PUBLIC_DECOMPOSITION_FIELDS),
            "root_rows_with_extra_fields": root_rows_with_extra_fields,
            "root_extra_field_occurrences": root_extra_field_occurrences,
            "root_distinct_extra_field_name_count": len(root_extra_field_names),
            "root_extra_field_name_set_sha256": _name_set_hash(
                root_extra_field_names
            ),
            "decomposition_rows_with_extra_fields": (
                decomposition_rows_with_extra_fields
            ),
            "decomposition_extra_field_occurrences": (
                decomposition_extra_field_occurrences
            ),
            "decomposition_distinct_extra_field_name_count": len(
                decomposition_extra_field_names
            ),
            "decomposition_extra_field_name_set_sha256": _name_set_hash(
                decomposition_extra_field_names
            ),
        },
        "identity_uniqueness": {
            "unique_id_count": len(seen_ids),
            "duplicate_id_count": 0,
            "unique_normalized_question_count": len(seen_questions),
            "duplicate_normalized_question_count": duplicate_question_count,
        },
        "reasoning": {
            "token_set_counts": dict(reasoning_counts),
            "unknown_residue_count": 0,
            "duplicate_token_count": 0,
            "missing_token_count": 0,
        },
        "answer_types": {
            "counts": dict(answer_type_counts),
            "unknown_answer_type_count": 0,
        },
        "contexts": {
            "item_context_cardinality_counts": _counter_mapping(
                context_cardinality
            ),
            "total_context_document_occurrences": total_context_documents,
            "unique_normalized_title_and_body_document_count": len(
                all_document_identities
            ),
            "duplicate_normalized_title_and_body_document_occurrences": (
                total_context_documents - len(all_document_identities)
            ),
            "unique_normalized_title_count": len(all_normalized_titles),
            "item_with_duplicate_normalized_context_title_count": (
                item_with_duplicate_normalized_context_title_count
            ),
        },
        "support_title_resolution": {
            "item_decomposition_cardinality_counts": _counter_mapping(
                decomposition_cardinality
            ),
            "total_decomposition_entries": total_decomposition_entries,
            "nonempty_support_reference_count": nonempty_support_reference_count,
            "empty_support_reference_count": empty_support_reference_count,
            "exactly_resolved_support_reference_count": (
                exactly_resolved_support_reference_count
            ),
            "missing_support_reference_count": missing_support_reference_count,
            "ambiguous_support_reference_count": ambiguous_support_reference_count,
            "fully_exactly_resolved_item_count": fully_resolved_item_count,
            "zero_nonempty_support_item_count": zero_nonempty_support_item_count,
        },
        "gold_and_capacity": {
            "fully_resolved_distinct_gold_document_cardinality_counts": (
                _counter_mapping(gold_cardinality)
            ),
            "minimum_gold_document_cardinality_required": 1,
            "fixed_gold_document_cardinality_assumed": False,
            "fixed_corpus_size_assumed": False,
            "target_per_reasoning_family": TARGET_PER_REASONING_FAMILY,
            "total_required_distinct_items": (
                TARGET_PER_REASONING_FAMILY * len(REASONING_TOKEN_ORDER)
            ),
            "eligible_normalized_question_count_after_public_exclusion": sum(
                eligible_token_set_counts.values()
            ),
            "conservative_eligible_token_set_counts": {
                label: eligible_token_set_counts[label]
                for label in REASONING_TOKEN_SET_LABELS
            },
            "hall_neighborhood_counts": hall_neighborhood_counts,
            "hall_shortfall_counts": hall_shortfalls,
            "exact_three_family_b_matching_capacity_met": capacity_ok,
        },
    }
    body["qualification_sha256"] = _sha256_json(body)
    return body


def build_synthetic_aggregate(
    raw: bytes,
    exclusion_raw: bytes,
) -> dict[str, Any]:
    """Synthetic-only entry point; preserves the frozen 1118/schema contract."""

    exclusion_ids = _strict_exclusion_ids(_decode_strict_json(exclusion_raw))
    return qualify_payload(
        _decode_strict_json(raw),
        source_size=len(raw),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        source_git_blob_sha1=_git_blob_sha1(raw),
        formal_source_identity_enforced=False,
        exclusion_ids=exclusion_ids,
        exclusion_size=len(exclusion_raw),
        exclusion_sha256=hashlib.sha256(exclusion_raw).hexdigest(),
        exclusion_git_blob_sha1=_git_blob_sha1(exclusion_raw),
        formal_exclusion_identity_enforced=False,
    )


def _canonical_project(project: str | Path) -> Path:
    supplied = Path(project)
    if supplied.is_symlink():
        raise MoreHopQASourceQualificationError("project root must not be a symlink")
    try:
        root = supplied.resolve(strict=True)
    except OSError as exc:
        raise MoreHopQASourceQualificationError("project root is unavailable") from exc
    if not root.is_dir():
        raise MoreHopQASourceQualificationError("project root must be a directory")
    return root


def _repository_root(project: Path) -> Path:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project), "rev-parse", "--show-toplevel"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MoreHopQASourceQualificationError(
            "project Git repository is unavailable"
        ) from exc
    return Path(completed.stdout.strip()).resolve(strict=True)


def _require_private_paths_ignored(project: Path) -> None:
    repository = _repository_root(project)
    try:
        prefix = project.relative_to(repository)
    except ValueError as exc:
        raise MoreHopQASourceQualificationError(
            "project escaped its Git repository"
        ) from exc
    relatives = (
        FORMAL_SOURCE_RELATIVE_PATH,
        FORMAL_EXCLUSION_RELATIVE_PATH,
        FORMAL_ATTEMPT_MARKER_RELATIVE_PATH,
    )
    repository_paths = tuple(
        (PurePosixPath(prefix.as_posix()) / path.as_posix()).as_posix()
        for path in relatives
    )
    stdin = b"\0".join(path.encode("utf-8") for path in repository_paths) + b"\0"
    try:
        index = subprocess.run(
            ["git", "-C", str(repository), "ls-files", "-z", "--", *repository_paths],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        head = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "ls-tree",
                "-r",
                "-z",
                "HEAD",
                "--",
                *repository_paths,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        ignored = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "check-ignore",
                "--no-index",
                "-z",
                "--stdin",
            ],
            input=stdin,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MoreHopQASourceQualificationError(
            "private source ignore verification failed"
        ) from exc
    if index.stdout or head.stdout:
        raise MoreHopQASourceQualificationError(
            "private source or marker is tracked"
        )
    returned = {row for row in ignored.stdout.split(b"\0") if row}
    expected = {path.encode("utf-8") for path in repository_paths}
    if ignored.returncode != 0 or returned != expected:
        raise MoreHopQASourceQualificationError(
            "private source or marker is not git-ignored"
        )


def _read_bound_private_json(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    expected_git_blob_sha1: str,
    label: str,
) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise MoreHopQASourceQualificationError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise MoreHopQASourceQualificationError(
            f"{label} must be a non-symlink regular file"
        )
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise MoreHopQASourceQualificationError(f"{label} mode must be 0600")
    if metadata.st_size != expected_size:
        raise MoreHopQASourceQualificationError(f"{label} size mismatch")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        raw = b""
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise MoreHopQASourceQualificationError(f"{label} changed while reading")
    if (
        len(raw) != expected_size
        or hashlib.sha256(raw).hexdigest() != expected_sha256
        or _git_blob_sha1(raw) != expected_git_blob_sha1
    ):
        raise MoreHopQASourceQualificationError(f"{label} identity mismatch")
    return raw


def build_aggregate(project: str | Path) -> dict[str, Any]:
    """Open only the fixed ignored source and build its aggregate receipt."""

    root = _canonical_project(project)
    _require_private_paths_ignored(root)
    exclusion_raw = _read_bound_private_json(
        root / FORMAL_EXCLUSION_RELATIVE_PATH,
        expected_size=FORMAL_EXCLUSION_SIZE,
        expected_sha256=FORMAL_EXCLUSION_SHA256,
        expected_git_blob_sha1=FORMAL_EXCLUSION_GIT_BLOB_SHA1,
        label="official exclusion",
    )
    exclusion_ids = _strict_exclusion_ids(_decode_strict_json(exclusion_raw))
    raw = _read_bound_private_json(
        root / FORMAL_SOURCE_RELATIVE_PATH,
        expected_size=FORMAL_SOURCE_SIZE,
        expected_sha256=FORMAL_SOURCE_SHA256,
        expected_git_blob_sha1=FORMAL_SOURCE_GIT_BLOB_SHA1,
        label="formal source",
    )
    return qualify_payload(
        _decode_strict_json(raw),
        source_size=len(raw),
        source_sha256=FORMAL_SOURCE_SHA256,
        source_git_blob_sha1=FORMAL_SOURCE_GIT_BLOB_SHA1,
        formal_source_identity_enforced=True,
        exclusion_ids=exclusion_ids,
        exclusion_size=FORMAL_EXCLUSION_SIZE,
        exclusion_sha256=FORMAL_EXCLUSION_SHA256,
        exclusion_git_blob_sha1=FORMAL_EXCLUSION_GIT_BLOB_SHA1,
        formal_exclusion_identity_enforced=True,
    )


def _consume_attempt_marker(project: Path) -> None:
    marker = project / FORMAL_ATTEMPT_MARKER_RELATIVE_PATH
    parent = marker.parent
    try:
        parent_metadata = parent.lstat()
    except OSError as exc:
        raise MoreHopQASourceQualificationError(
            "attempt marker parent is unavailable"
        ) from exc
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(
        parent_metadata.st_mode
    ):
        raise MoreHopQASourceQualificationError(
            "attempt marker parent is unsafe"
        )
    try:
        descriptor = os.open(
            marker,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError as exc:
        raise MoreHopQASourceQualificationError(
            "source qualification attempt is already consumed"
        ) from exc
    try:
        os.fchmod(descriptor, 0o600)
        raw = f"{SCHEMA}\nformal_attempt_consumed\n".encode("ascii")
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    parent = path.parent
    metadata = parent.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise MoreHopQASourceQualificationError("output parent is unsafe")
    raw = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"
    temporary = parent / f".{path.name}.{os.urandom(12).hex()}.tmp"
    descriptor: int | None = None
    published = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        published = True
        temporary.unlink()
        directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if published:
            path.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    project = _canonical_project(arguments.project)
    output = project / FORMAL_OUTPUT_RELATIVE_PATH
    if output.exists() or output.is_symlink():
        raise FileExistsError("source qualification manifest already exists")
    _require_private_paths_ignored(project)
    _consume_attempt_marker(project)
    receipt = build_aggregate(project)
    _write_json_exclusive(output, receipt)
    sys.stdout.write(
        json.dumps(
            {
                "qualification_sha256": receipt["qualification_sha256"],
                "qualification_status": receipt["qualification_status"],
                "schema": SCHEMA,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
