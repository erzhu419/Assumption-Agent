"""Public, non-scoring WiCE claim schema/topology qualification.

This module is deliberately limited to the three public claim-level JSONL
files.  It verifies exact byte size, Git blob identity, and whole-file SHA-256
for all three.  Only TRAIN and DEV are decoded for aggregate schema, topology,
capacity, and component-collision counts.  TEST remains identity-only: its raw
newline count is recorded, but no JSON value or field is decoded before a
later study authorizes first access.  Claim text, evidence text, metadata
values, identifiers, supporting-sentence indices, and content-derived
fingerprints never leave private memory.

No cohort, secret, action, model, evaluator, qrel, quota, or score is created
by this qualification.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any
import unicodedata


class WiceP0QualificationError(RuntimeError):
    """Raised when the pinned public source cannot be qualified safely."""


OFFICIAL_COMMIT = "ddeb6c183665e2a20c5f03c5aa07f03888b9870f"
OFFICIAL_FILES: dict[str, dict[str, Any]] = {
    "train": {
        "relative_path": "data/entailment_retrieval/claim/train.jsonl",
        "size_bytes": 12_039_943,
        "git_blob_sha1": "d1dba6f6cd6f24ea929f46c7ed1a176c6e0d63c1",
    },
    "dev": {
        "relative_path": "data/entailment_retrieval/claim/dev.jsonl",
        "size_bytes": 3_490_479,
        "git_blob_sha1": "7a0add2baa9c9b0eadbf4ad8940e2f5676e56166",
    },
    "test": {
        "relative_path": "data/entailment_retrieval/claim/test.jsonl",
        "size_bytes": 3_624_529,
        "git_blob_sha1": "f601848e944e55b4c580b381e1d0db5feb52839e",
    },
}

_SPLITS = frozenset({"train", "dev", "test"})
_ROW_KEYS = frozenset(
    {"label", "supporting_sentences", "claim", "evidence", "meta"}
)
_META_KEYS = frozenset(
    {"id", "claim_title", "claim_section", "claim_context"}
)
_ELIGIBLE_LABELS = frozenset({"supported", "partially_supported"})
_LABEL_REGISTRY = {
    "supported": "SUPPORTED",
    "partially_supported": "PARTIALLY_SUPPORTED",
    "not_supported": "NOT_SUPPORTED",
}
_SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class _DuplicateJsonKey(ValueError):
    """Private strict-JSON signal; object keys are never included."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise WiceP0QualificationError(
            "aggregate receipt is not canonical JSON"
        ) from exc


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _json_type(value: object) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unsupported"


def _git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()  # noqa: S324


def _histogram(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _collision_counts(counter: Counter[str]) -> dict[str, int]:
    colliding = [count for count in counter.values() if count > 1]
    return {
        "collision_group_count": len(colliding),
        "collision_member_count": sum(colliding),
        "collision_row_count": sum(colliding),
        "collision_excess_member_count": sum(count - 1 for count in colliding),
        "unique_component_count": len(counter),
    }


def _key_contract(value: Mapping[str, Any], expected: frozenset[str]) -> str:
    actual = set(value)
    missing = bool(expected - actual)
    extra = bool(actual - expected)
    if missing and extra:
        return "MISSING_AND_EXTRA"
    if missing:
        return "MISSING_ONLY"
    if extra:
        return "EXTRA_ONLY"
    return "EXACT"


def _strict_json_line(raw: str) -> object:
    def object_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _DuplicateJsonKey
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise ValueError("nonfinite JSON constant")

    return json.loads(
        raw,
        object_pairs_hook=object_pairs,
        parse_constant=reject_constant,
    )


def _validate_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise WiceP0QualificationError("source relative path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise WiceP0QualificationError("source relative path is unsafe")
    return value


def _validate_bindings(
    expected_files: Mapping[str, Mapping[str, Any]],
    expected_sha256s: Mapping[str, str] | None,
) -> dict[str, dict[str, Any]]:
    if set(expected_files) != _SPLITS:
        raise WiceP0QualificationError(
            "expected file bindings must contain train, dev, and test"
        )
    supplied_sha256s = dict(expected_sha256s or {})
    if any(split not in _SPLITS for split in supplied_sha256s):
        raise WiceP0QualificationError(
            "whole-file SHA-256 binding contains an unknown split"
        )

    validated: dict[str, dict[str, Any]] = {}
    for split in sorted(_SPLITS):
        expected = expected_files[split]
        relative_path = _validate_relative_path(expected.get("relative_path"))
        size = expected.get("size_bytes")
        blob = expected.get("git_blob_sha1")
        embedded_sha256 = expected.get(
            "file_sha256", expected.get("sha256")
        )
        supplied_sha256 = supplied_sha256s.get(split)
        if (
            type(size) is not int
            or size < 0
            or not isinstance(blob, str)
            or _SHA1_RE.fullmatch(blob) is None
        ):
            raise WiceP0QualificationError(
                "source size or Git blob binding is invalid"
            )
        if (
            embedded_sha256 is not None
            and supplied_sha256 is not None
            and embedded_sha256 != supplied_sha256
        ):
            raise WiceP0QualificationError(
                "whole-file SHA-256 bindings disagree"
            )
        file_sha256 = (
            supplied_sha256
            if supplied_sha256 is not None
            else embedded_sha256
        )
        if (
            not isinstance(file_sha256, str)
            or _SHA256_RE.fullmatch(file_sha256) is None
        ):
            raise WiceP0QualificationError(
                "whole-file SHA-256 binding is required for every split"
            )
        validated[split] = {
            "file_sha256": file_sha256,
            "git_blob_sha1": blob,
            "relative_path": relative_path,
            "size_bytes": size,
        }
    return validated


def _read_bound_file(path: Path, expected: Mapping[str, Any]) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise WiceP0QualificationError(
            "pinned public source file is unavailable"
        ) from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size != expected["size_bytes"]
    ):
        raise WiceP0QualificationError(
            "pinned public source is not the expected single regular file"
        )

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
        try:
            opened = os.fstat(fd)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or opened.st_size != expected["size_bytes"]
                or opened.st_dev != before.st_dev
                or opened.st_ino != before.st_ino
            ):
                raise WiceP0QualificationError(
                    "pinned public source changed before read"
                )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(fd, 1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(fd)
        finally:
            os.close(fd)
    except WiceP0QualificationError:
        raise
    except OSError as exc:
        raise WiceP0QualificationError(
            "pinned public source file could not be read safely"
        ) from exc

    raw = b"".join(chunks)
    if (
        after.st_dev != opened.st_dev
        or after.st_ino != opened.st_ino
        or after.st_size != opened.st_size
        or after.st_mtime_ns != opened.st_mtime_ns
        or len(raw) != expected["size_bytes"]
    ):
        raise WiceP0QualificationError(
            "pinned public source changed during read"
        )
    if _git_blob_sha1(raw) != expected["git_blob_sha1"]:
        raise WiceP0QualificationError(
            "pinned public source Git blob identity does not match"
        )
    if hashlib.sha256(raw).hexdigest() != expected["file_sha256"]:
        raise WiceP0QualificationError(
            "pinned public source whole-file SHA-256 does not match"
        )
    return raw


def _family_for_minimum(size: int) -> str:
    if size == 1:
        return "MIN_SUPPORTING_SET_SIZE_1"
    if size == 2:
        return "MIN_SUPPORTING_SET_SIZE_2"
    return "MIN_SUPPORTING_SET_SIZE_GE_3"


def _observe_split(
    *,
    split: str,
    raw: bytes,
) -> tuple[dict[str, Any], dict[str, Counter[str]]]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise WiceP0QualificationError(
            "pinned public source is not strict UTF-8"
        ) from exc

    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()

    anomaly_counts: Counter[str] = Counter()
    field_types: Counter[str] = Counter()
    row_key_counts: Counter[str] = Counter()
    row_key_contracts: Counter[str] = Counter()
    meta_key_counts: Counter[str] = Counter()
    meta_key_contracts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    evidence_sizes: Counter[str] = Counter()
    alternative_counts: Counter[str] = Counter()
    raw_supporting_set_sizes: Counter[str] = Counter()
    valid_supporting_set_sizes: Counter[str] = Counter()
    valid_alternative_counts: Counter[str] = Counter()
    minimum_sizes: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    family_evidence_components: dict[str, set[str]] = {}

    claim_components: Counter[str] = Counter()
    evidence_list_components: Counter[str] = Counter()
    claim_evidence_pair_components: Counter[str] = Counter()
    identifier_components: Counter[str] = Counter()
    evidence_sentence_components: Counter[str] = Counter()

    parsed_json_count = 0
    json_decode_attempt_count = 0
    object_row_count = 0
    valid_evidence_sentence_link_count = 0
    eligible_label_row_count = 0
    eligible_family_row_count = 0
    empty_supporting_set_count = 0
    duplicate_alternative_count = 0
    within_row_evidence_sentence_collision_excess_link_count = 0

    for line in lines:
        if not line.strip():
            anomaly_counts["jsonl_blank_record"] += 1
            continue
        json_decode_attempt_count += 1
        try:
            row = _strict_json_line(line)
        except _DuplicateJsonKey:
            anomaly_counts["row_duplicate_json_key"] += 1
            continue
        except (json.JSONDecodeError, ValueError, RecursionError):
            anomaly_counts["row_not_strict_json"] += 1
            continue
        parsed_json_count += 1
        if not isinstance(row, dict):
            anomaly_counts["row_not_object"] += 1
            continue
        object_row_count += 1

        row_key_counts[str(len(row))] += 1
        row_contract = _key_contract(row, _ROW_KEYS)
        row_key_contracts[row_contract] += 1
        if row_contract != "EXACT":
            anomaly_counts["row_keyset_not_exact"] += 1

        for field in sorted(_ROW_KEYS):
            field_types[f"{field}:{_json_type(row.get(field))}"] += 1

        label = row.get("label")
        canonical_label: str | None = None
        if isinstance(label, str):
            canonical_label = _LABEL_REGISTRY.get(label)
            label_counts[canonical_label or "UNKNOWN_STRING"] += 1
            if canonical_label is None:
                anomaly_counts["label_not_in_official_registry"] += 1
        else:
            label_counts[
                f"NON_STRING_{_json_type(label).upper()}"
            ] += 1
            anomaly_counts["label_not_string"] += 1

        normalized_claim: str | None = None
        claim = row.get("claim")
        if isinstance(claim, str) and _normalize_text(claim):
            normalized_claim = _normalize_text(claim)
            claim_hash = hashlib.sha256(
                normalized_claim.encode("utf-8")
            ).hexdigest()
            claim_components[claim_hash] += 1
        else:
            anomaly_counts["claim_not_nonempty_string"] += 1

        evidence = row.get("evidence")
        evidence_length: int | None = None
        evidence_fingerprint: str | None = None
        if isinstance(evidence, list):
            evidence_length = len(evidence)
            evidence_sizes[str(evidence_length)] += 1
            normalized_evidence: list[str] = []
            evidence_elements_valid = True
            per_row_sentence_counter: Counter[str] = Counter()
            for sentence in evidence:
                if not isinstance(sentence, str):
                    anomaly_counts["evidence_sentence_not_string"] += 1
                    evidence_elements_valid = False
                    continue
                normalized_sentence = _normalize_text(sentence)
                if not normalized_sentence:
                    anomaly_counts[
                        "evidence_sentence_not_nonempty_string"
                    ] += 1
                    evidence_elements_valid = False
                    continue
                sentence_hash = hashlib.sha256(
                    normalized_sentence.encode("utf-8")
                ).hexdigest()
                normalized_evidence.append(normalized_sentence)
                per_row_sentence_counter[sentence_hash] += 1
                evidence_sentence_components[sentence_hash] += 1
                valid_evidence_sentence_link_count += 1
            repeated_within = sum(
                count - 1
                for count in per_row_sentence_counter.values()
                if count > 1
            )
            within_row_evidence_sentence_collision_excess_link_count += (
                repeated_within
            )
            if evidence_elements_valid:
                evidence_fingerprint = hashlib.sha256(
                    _canonical_bytes(normalized_evidence)
                ).hexdigest()
                evidence_list_components[evidence_fingerprint] += 1
        else:
            anomaly_counts["evidence_not_array"] += 1

        meta = row.get("meta")
        if isinstance(meta, dict):
            meta_key_counts[str(len(meta))] += 1
            meta_contract = _key_contract(meta, _META_KEYS)
            meta_key_contracts[meta_contract] += 1
            if meta_contract != "EXACT":
                anomaly_counts["meta_keyset_not_exact"] += 1
            for field in sorted(_META_KEYS):
                field_types[
                    f"meta.{field}:{_json_type(meta.get(field))}"
                ] += 1
            identifier = meta.get("id")
            if isinstance(identifier, str) and _normalize_text(identifier):
                identifier_hash = hashlib.sha256(
                    _normalize_text(identifier).encode("utf-8")
                ).hexdigest()
                identifier_components[identifier_hash] += 1
            else:
                anomaly_counts["meta_id_not_nonempty_string"] += 1
            title = meta.get("claim_title")
            if not isinstance(title, str) or not _normalize_text(title):
                anomaly_counts[
                    "meta_claim_title_not_nonempty_string"
                ] += 1
            for field in ("claim_section", "claim_context"):
                if not isinstance(meta.get(field), str):
                    anomaly_counts[f"meta_{field}_not_string"] += 1
        else:
            anomaly_counts["meta_not_object"] += 1

        supporting = row.get("supporting_sentences")
        valid_sets: list[tuple[int, ...]] = []
        canonical_alternatives: Counter[tuple[int, ...]] = Counter()
        if isinstance(supporting, list):
            alternative_counts[str(len(supporting))] += 1
            for alternative in supporting:
                if not isinstance(alternative, list):
                    anomaly_counts["supporting_alternative_not_array"] += 1
                    continue
                raw_supporting_set_sizes[str(len(alternative))] += 1
                if not alternative:
                    empty_supporting_set_count += 1
                indices_valid = True
                integer_indices: list[int] = []
                for index in alternative:
                    if type(index) is not int:
                        anomaly_counts[
                            "supporting_index_not_integer"
                        ] += 1
                        indices_valid = False
                        continue
                    integer_indices.append(index)
                    if (
                        evidence_length is None
                        or index < 0
                        or index >= evidence_length
                    ):
                        anomaly_counts[
                            "supporting_index_out_of_range"
                        ] += 1
                        indices_valid = False
                if len(integer_indices) != len(set(integer_indices)):
                    anomaly_counts[
                        "supporting_index_duplicate_within_set"
                    ] += 1
                    indices_valid = False
                if indices_valid:
                    canonical = tuple(sorted(integer_indices))
                    canonical_alternatives[canonical] += 1
                    if canonical:
                        valid_sets.append(canonical)
                        valid_supporting_set_sizes[
                            str(len(canonical))
                        ] += 1
            duplicate_alternatives = sum(
                count - 1
                for count in canonical_alternatives.values()
                if count > 1
            )
            if duplicate_alternatives:
                anomaly_counts[
                    "supporting_alternative_duplicate"
                ] += duplicate_alternatives
                duplicate_alternative_count += duplicate_alternatives
        else:
            anomaly_counts["supporting_sentences_not_array"] += 1

        unique_valid_sets = set(valid_sets)
        valid_alternative_counts[str(len(unique_valid_sets))] += 1
        is_eligible_label = label in _ELIGIBLE_LABELS
        if is_eligible_label:
            eligible_label_row_count += 1
            if not unique_valid_sets:
                anomaly_counts[
                    "eligible_label_without_valid_nonempty_supporting_set"
                ] += 1
            else:
                minimum_size = min(len(value) for value in unique_valid_sets)
                family = _family_for_minimum(minimum_size)
                minimum_sizes[str(minimum_size)] += 1
                family_counts[family] += 1
                eligible_family_row_count += 1
                if evidence_fingerprint is not None:
                    family_evidence_components.setdefault(
                        family, set()
                    ).add(evidence_fingerprint)
            if any(not value for value in canonical_alternatives):
                anomaly_counts[
                    "eligible_label_has_empty_supporting_set"
                ] += canonical_alternatives[()]
        elif label == "not_supported" and unique_valid_sets:
            anomaly_counts[
                "not_supported_label_has_nonempty_supporting_set"
            ] += 1

        if normalized_claim is not None and evidence_fingerprint is not None:
            pair_hash = hashlib.sha256(
                (
                    hashlib.sha256(
                        normalized_claim.encode("utf-8")
                    ).hexdigest()
                    + "\0"
                    + evidence_fingerprint
                ).encode("ascii")
            ).hexdigest()
            claim_evidence_pair_components[pair_hash] += 1

    private_components = {
        "claim": claim_components,
        "claim_evidence_pair": claim_evidence_pair_components,
        "evidence_list": evidence_list_components,
        "evidence_sentence": evidence_sentence_components,
        "identifier": identifier_components,
    }
    receipt = {
        "aggregate_counts": {
            "duplicate_supporting_alternative_count": (
                duplicate_alternative_count
            ),
            "eligible_family_row_count": eligible_family_row_count,
            "eligible_label_row_count": eligible_label_row_count,
            "empty_supporting_set_count": empty_supporting_set_count,
            "jsonl_record_count": len(lines),
            "json_decode_attempt_count": json_decode_attempt_count,
            "object_row_count": object_row_count,
            "parsed_json_count": parsed_json_count,
            "valid_evidence_sentence_link_count": (
                valid_evidence_sentence_link_count
            ),
            "within_row_evidence_sentence_collision_excess_link_count": (
                within_row_evidence_sentence_collision_excess_link_count
            ),
        },
        "alternative_count_histogram": _histogram(alternative_counts),
        "component_collision_count": {
            name: _collision_counts(counter)
            for name, counter in sorted(private_components.items())
        },
        "eligible_family_capacity_count": _histogram(family_counts),
        "eligible_family_unique_evidence_component_count": {
            family: len(values)
            for family, values in sorted(
                family_evidence_components.items()
            )
        },
        "evidence_list_size_histogram": _histogram(evidence_sizes),
        "field_type_count": _histogram(field_types),
        "label_count": _histogram(label_counts),
        "meta_key_contract_count": _histogram(meta_key_contracts),
        "meta_key_count_histogram": _histogram(meta_key_counts),
        "minimum_valid_supporting_set_size_histogram": _histogram(
            minimum_sizes
        ),
        "raw_supporting_set_size_histogram": _histogram(
            raw_supporting_set_sizes
        ),
        "row_key_contract_count": _histogram(row_key_contracts),
        "row_key_count_histogram": _histogram(row_key_counts),
        "schema_anomaly_count": _histogram(anomaly_counts),
        "split": split,
        "valid_nonempty_alternative_count_histogram": _histogram(
            valid_alternative_counts
        ),
        "valid_supporting_set_size_histogram": _histogram(
            valid_supporting_set_sizes
        ),
    }
    return receipt, private_components


def _cross_split_collision_counts(
    components: Mapping[str, Mapping[str, Counter[str]]],
) -> dict[str, dict[str, int]]:
    names = sorted(next(iter(components.values())))
    output: dict[str, dict[str, int]] = {}
    for name in names:
        by_value: dict[str, list[int]] = {}
        for split in sorted(components):
            for fingerprint, count in components[split][name].items():
                by_value.setdefault(fingerprint, []).append(count)
        colliding = [
            counts for counts in by_value.values() if len(counts) > 1
        ]
        output[name] = {
            "cross_split_collision_group_count": len(colliding),
            "cross_split_collision_member_count": sum(
                sum(counts) for counts in colliding
            ),
            "cross_split_collision_row_count": sum(
                sum(counts) for counts in colliding
            ),
            "cross_split_collision_excess_member_count": sum(
                sum(counts) - 1 for counts in colliding
            ),
        }
    return output


def qualify_source(
    *,
    source_root: Path,
    expected_files: Mapping[str, Mapping[str, Any]] = OFFICIAL_FILES,
    expected_sha256s: Mapping[str, str] | None = None,
    semantic_splits: Sequence[str] = ("train", "dev"),
) -> dict[str, Any]:
    """Verify all files and aggregate TRAIN/DEV without decoding TEST."""

    root = source_root.resolve()
    if not root.is_dir():
        raise WiceP0QualificationError(
            "pinned public source root is unavailable"
        )
    bindings = _validate_bindings(expected_files, expected_sha256s)
    if (
        tuple(semantic_splits) != ("train", "dev")
        or len(set(semantic_splits)) != 2
    ):
        raise WiceP0QualificationError(
            "semantic split contract must be exactly train then dev"
        )
    semantic_split_set = set(semantic_splits)

    split_receipts: dict[str, Any] = {}
    private_components: dict[str, dict[str, Counter[str]]] = {}
    source_files: dict[str, Any] = {}
    for split in sorted(bindings):
        expected = bindings[split]
        path = root / expected["relative_path"]
        raw = _read_bound_file(path, expected)
        source_files[split] = {
            "file_sha256": expected["file_sha256"],
            "git_blob_sha1": expected["git_blob_sha1"],
            "json_decode_count": 0,
            "raw_newline_count": raw.count(b"\n"),
            "relative_path": expected["relative_path"],
            "size_bytes": expected["size_bytes"],
        }
        if split in semantic_split_set:
            split_receipts[split], private_components[split] = _observe_split(
                split=split,
                raw=raw,
            )
            source_files[split]["json_decode_count"] = split_receipts[split][
                "aggregate_counts"
            ]["json_decode_attempt_count"]

    anomaly_total = sum(
        sum(receipt["schema_anomaly_count"].values())
        for receipt in split_receipts.values()
    )
    body: dict[str, Any] = {
        "access_boundary": {
            "action_model_evaluator_qrel_or_score_count": 0,
            "individual_claim_evidence_meta_identifier_or_support_index_output_count": 0,
            "private_cohort_or_secret_count": 0,
            "public_source_file_identity_read_count": len(source_files),
            "public_source_split_json_decode_count": len(split_receipts),
            "test_json_decode_count": source_files["test"][
                "json_decode_count"
            ],
        },
        "cross_split_component_collision_count": (
            _cross_split_collision_counts(private_components)
        ),
        "official_commit": OFFICIAL_COMMIT,
        "p1_quota_or_cohort_assumption_count": 0,
        "recorded_date": "2026-07-26",
        "schema": "wice_p0_public_schema_qualification_v1",
        "source_files": source_files,
        "split_receipts": split_receipts,
        "split_semantic_access_policy": {
            "identity_only_splits": ["test"],
            "semantic_aggregate_splits": list(semantic_splits),
        },
        "status": (
            "qualified_public_non_scoring_schema_topology"
            if anomaly_total == 0
            else "not_qualified_public_schema_anomalies"
        ),
        "study_id": "WICE_P0_PUBLIC_SCHEMA_TOPOLOGY_V1",
        "total_schema_anomaly_count": anomaly_total,
    }
    body["self_sha256"] = _stable_hash(body)
    return body


def write_receipt_exclusive(
    path: Path, receipt: Mapping[str, Any]
) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    data = _canonical_bytes(receipt) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
        try:
            offset = 0
            while offset < len(data):
                offset += os.write(fd, data[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        raise WiceP0QualificationError(
            "aggregate receipt could not be written exclusively"
        ) from exc
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise WiceP0QualificationError(
            "aggregate receipt mode is not 0600"
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    for split in sorted(_SPLITS):
        parser.add_argument(
            f"--{split}-sha256",
            required=True,
            dest=f"{split}_sha256",
        )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    receipt = qualify_source(
        source_root=args.source_root,
        expected_sha256s={
            split: getattr(args, f"{split}_sha256")
            for split in sorted(_SPLITS)
        },
    )
    write_receipt_exclusive(args.output, receipt)
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "self_sha256": receipt["self_sha256"],
                "status": receipt["status"],
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
