"""Public, non-scoring AVeriTeC schema/topology qualification.

This module intentionally runs before any private secret, cohort, action, model,
evaluator, qrel pack, or score exists.  It emits aggregate schema and capacity
facts only; claim, question, answer, URL, justification, and item identifiers
are never included in the receipt.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence
import unicodedata


class AveritecP0QualificationError(RuntimeError):
    """Raised when the pinned public source cannot be qualified safely."""


OFFICIAL_COMMIT = "7c62d1ec8df3fb560d6efe2b85fa191135636f81"
OFFICIAL_FILES: dict[str, dict[str, Any]] = {
    "train": {
        "relative_path": "data/train.json",
        "size_bytes": 10_184_813,
        "git_blob_sha1": "0f190e115cf2ee23416e8a539c8d6ac043d7cc83",
    },
    "dev": {
        "relative_path": "data/dev.json",
        "size_bytes": 1_785_475,
        "git_blob_sha1": "40974243267f395dc583d805d10f043812419249",
    },
}

_LABEL_REGISTRY = {
    "supported": "SUPPORTED",
    "refuted": "REFUTED",
    "not enough evidence": "NOT_ENOUGH_EVIDENCE",
    "conflicting evidence/cherrypicking": "CONFLICTING_EVIDENCE_CHERRYPICKING",
}
_CLAIM_TYPE_REGISTRY = {
    "position statement": "POSITION_STATEMENT",
    "numerical claim": "NUMERICAL_CLAIM",
    "event/property claim": "EVENT_OR_PROPERTY_CLAIM",
    "quote verification": "QUOTE_VERIFICATION",
    "causal claim": "CAUSAL_CLAIM",
}
_ANSWER_TYPE_REGISTRY = {
    "abstractive": "ABSTRACTIVE",
    "extractive": "EXTRACTIVE",
    "boolean": "BOOLEAN",
    "unanswerable": "UNANSWERABLE",
}
_FAMILY_PRIORITY = (
    "CAUSAL_CLAIM",
    "QUOTE_VERIFICATION",
    "NUMERICAL_CLAIM",
)


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
        raise AveritecP0QualificationError(
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


def _safe_vocab_bucket(value: object, registry: Mapping[str, str]) -> str:
    if not isinstance(value, str):
        return f"NON_STRING_{_json_type(value).upper()}"
    normalized = _normalize_text(value)
    known = registry.get(normalized)
    if known is not None:
        return known
    return "UNKNOWN_SHA256_" + hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()  # noqa: S324 - Git identity.


def _read_bound_file(path: Path, expected: Mapping[str, Any]) -> bytes:
    try:
        info = path.lstat()
    except OSError as exc:
        raise AveritecP0QualificationError(
            "pinned public source file is unavailable"
        ) from exc
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise AveritecP0QualificationError(
            "pinned public source path is not a single regular file"
        )
    if info.st_size != expected["size_bytes"]:
        raise AveritecP0QualificationError(
            "pinned public source size does not match Git topology"
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
        try:
            raw = b""
            while True:
                chunk = os.read(fd, 1 << 20)
                if not chunk:
                    break
                raw += chunk
        finally:
            os.close(fd)
    except OSError as exc:
        raise AveritecP0QualificationError(
            "pinned public source file could not be read safely"
        ) from exc
    if _git_blob_sha1(raw) != expected["git_blob_sha1"]:
        raise AveritecP0QualificationError(
            "pinned public source Git blob identity does not match"
        )
    return raw


def _histogram(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _keyset_histogram(counter: Counter[tuple[str, ...]]) -> list[dict[str, Any]]:
    return [
        {"count": counter[keys], "keys": list(keys)}
        for keys in sorted(counter)
    ]


def _exclusive_family(claim_types: set[str]) -> str | None:
    for family in _FAMILY_PRIORITY:
        if family in claim_types:
            return family
    return None


def _observe_split(
    *,
    split: str,
    rows: object,
    exposure_claim_sha256s: set[str],
) -> tuple[dict[str, Any], Counter[str], Counter[str]]:
    if not isinstance(rows, list):
        raise AveritecP0QualificationError(
            "pinned public split root is not an array"
        )

    row_keysets: Counter[tuple[str, ...]] = Counter()
    question_keysets: Counter[tuple[str, ...]] = Counter()
    answer_keysets: Counter[tuple[str, ...]] = Counter()
    field_types: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    claim_type_counts: Counter[str] = Counter()
    answer_type_counts: Counter[str] = Counter()
    anomaly_counts: Counter[str] = Counter()
    evidence_cardinality: Counter[str] = Counter()
    family_rows: Counter[str] = Counter()
    family_minimum_one: Counter[str] = Counter()
    family_minimum_two: Counter[str] = Counter()
    normalized_claim_counts: Counter[str] = Counter()
    evidence_unit_counts: Counter[str] = Counter()
    exposure_match_count = 0
    total_question_count = 0
    total_answer_count = 0
    total_usable_evidence_count = 0

    for row in rows:
        if not isinstance(row, dict):
            anomaly_counts["row_not_object"] += 1
            continue
        if any(not isinstance(key, str) for key in row):
            anomaly_counts["row_key_not_string"] += 1
            continue
        row_keysets[tuple(sorted(row))] += 1
        for key in (
            "claim",
            "label",
            "claim_types",
            "questions",
            "justification",
            "claim_date",
            "speaker",
        ):
            field_types[f"{key}:{_json_type(row.get(key))}"] += 1

        claim = row.get("claim")
        claim_hash: str | None = None
        if isinstance(claim, str) and _normalize_text(claim):
            claim_hash = hashlib.sha256(
                _normalize_text(claim).encode("utf-8")
            ).hexdigest()
            normalized_claim_counts[claim_hash] += 1
            if claim_hash in exposure_claim_sha256s:
                exposure_match_count += 1
        else:
            anomaly_counts["claim_not_nonempty_string"] += 1

        label_counts[
            _safe_vocab_bucket(row.get("label"), _LABEL_REGISTRY)
        ] += 1

        canonical_claim_types: set[str] = set()
        claim_types = row.get("claim_types")
        if not isinstance(claim_types, list):
            anomaly_counts["claim_types_not_array"] += 1
        else:
            for value in claim_types:
                bucket = _safe_vocab_bucket(value, _CLAIM_TYPE_REGISTRY)
                claim_type_counts[bucket] += 1
                canonical_claim_types.add(bucket)
        family = _exclusive_family(canonical_claim_types)
        if family is not None:
            family_rows[family] += 1

        item_evidence_hashes: set[str] = set()
        questions = row.get("questions")
        if not isinstance(questions, list):
            anomaly_counts["questions_not_array"] += 1
            questions = []
        for question in questions:
            total_question_count += 1
            if not isinstance(question, dict):
                anomaly_counts["question_not_object"] += 1
                continue
            if any(not isinstance(key, str) for key in question):
                anomaly_counts["question_key_not_string"] += 1
                continue
            question_keysets[tuple(sorted(question))] += 1
            question_text = question.get("question")
            normalized_question = (
                _normalize_text(question_text)
                if isinstance(question_text, str)
                else ""
            )
            if not normalized_question:
                anomaly_counts["question_not_nonempty_string"] += 1
            answers = question.get("answers")
            if not isinstance(answers, list):
                anomaly_counts["answers_not_array"] += 1
                continue
            for answer in answers:
                total_answer_count += 1
                if not isinstance(answer, dict):
                    anomaly_counts["answer_not_object"] += 1
                    continue
                if any(not isinstance(key, str) for key in answer):
                    anomaly_counts["answer_key_not_string"] += 1
                    continue
                answer_keysets[tuple(sorted(answer))] += 1
                answer_type = _safe_vocab_bucket(
                    answer.get("answer_type"), _ANSWER_TYPE_REGISTRY
                )
                answer_type_counts[answer_type] += 1
                answer_text = answer.get("answer")
                normalized_answer = (
                    _normalize_text(answer_text)
                    if isinstance(answer_text, str)
                    else ""
                )
                if not normalized_answer:
                    anomaly_counts["answer_not_nonempty_string"] += 1
                    continue
                if not normalized_question or answer_type == "UNANSWERABLE":
                    continue
                evidence_hash = hashlib.sha256(
                    (
                        normalized_question
                        + "\0"
                        + normalized_answer
                    ).encode("utf-8")
                ).hexdigest()
                item_evidence_hashes.add(evidence_hash)

        for evidence_hash in item_evidence_hashes:
            evidence_unit_counts[evidence_hash] += 1
        cardinality = len(item_evidence_hashes)
        evidence_cardinality[str(cardinality)] += 1
        total_usable_evidence_count += cardinality
        if family is not None and claim_hash is not None:
            if cardinality >= 1:
                family_minimum_one[family] += 1
            if cardinality >= 2:
                family_minimum_two[family] += 1

    claim_collision_groups = sum(
        count > 1 for count in normalized_claim_counts.values()
    )
    claim_collision_rows = sum(
        count for count in normalized_claim_counts.values() if count > 1
    )
    repeated_evidence_groups = sum(
        count > 1 for count in evidence_unit_counts.values()
    )
    repeated_evidence_links = sum(
        count for count in evidence_unit_counts.values() if count > 1
    )
    receipt = {
        "aggregate_counts": {
            "answer_count": total_answer_count,
            "claim_exposure_match_count": exposure_match_count,
            "normalized_claim_collision_group_count": claim_collision_groups,
            "normalized_claim_collision_row_count": claim_collision_rows,
            "question_count": total_question_count,
            "repeated_evidence_unit_group_count": repeated_evidence_groups,
            "repeated_evidence_unit_link_count": repeated_evidence_links,
            "row_count": len(rows),
            "unique_normalized_claim_count": len(normalized_claim_counts),
            "unique_usable_evidence_unit_count": len(evidence_unit_counts),
            "usable_evidence_unit_link_count": total_usable_evidence_count,
        },
        "answer_keyset_histogram": _keyset_histogram(answer_keysets),
        "answer_type_count": _histogram(answer_type_counts),
        "claim_type_count": _histogram(claim_type_counts),
        "exclusive_family_count": _histogram(family_rows),
        "exclusive_family_minimum_one_evidence_count": _histogram(
            family_minimum_one
        ),
        "exclusive_family_minimum_two_evidence_count": _histogram(
            family_minimum_two
        ),
        "field_type_count": _histogram(field_types),
        "label_count": _histogram(label_counts),
        "question_keyset_histogram": _keyset_histogram(question_keysets),
        "row_keyset_histogram": _keyset_histogram(row_keysets),
        "schema_anomaly_count": _histogram(anomaly_counts),
        "split": split,
        "usable_evidence_cardinality_histogram": _histogram(
            evidence_cardinality
        ),
    }
    return receipt, normalized_claim_counts, evidence_unit_counts


def qualify_source(
    *,
    source_root: Path,
    expected_files: Mapping[str, Mapping[str, Any]] = OFFICIAL_FILES,
    exposure_claim_sha256s: Sequence[str] = (),
) -> dict[str, Any]:
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise AveritecP0QualificationError(
            "pinned public source root is unavailable"
        )
    exposure_hashes = set(exposure_claim_sha256s)
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for value in exposure_hashes
    ):
        raise AveritecP0QualificationError(
            "viewer exposure commitment is not a SHA-256 set"
        )

    split_receipts: dict[str, Any] = {}
    claim_counters: dict[str, Counter[str]] = {}
    evidence_counters: dict[str, Counter[str]] = {}
    source_files: dict[str, Any] = {}
    for split in sorted(expected_files):
        expected = expected_files[split]
        path = source_root / expected["relative_path"]
        raw = _read_bound_file(path, expected)
        try:
            rows = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AveritecP0QualificationError(
                "pinned public source is not strict UTF-8 JSON"
            ) from exc
        (
            split_receipts[split],
            claim_counters[split],
            evidence_counters[split],
        ) = _observe_split(
            split=split,
            rows=rows,
            exposure_claim_sha256s=exposure_hashes,
        )
        source_files[split] = {
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "git_blob_sha1": expected["git_blob_sha1"],
            "relative_path": expected["relative_path"],
            "size_bytes": len(raw),
        }

    split_names = sorted(claim_counters)
    cross_split_claim_hashes: set[str] = set()
    cross_split_evidence_hashes: set[str] = set()
    if len(split_names) == 2:
        left, right = split_names
        cross_split_claim_hashes = (
            set(claim_counters[left]) & set(claim_counters[right])
        )
        cross_split_evidence_hashes = (
            set(evidence_counters[left]) & set(evidence_counters[right])
        )

    body: dict[str, Any] = {
        "access_boundary": {
            "action_model_evaluator_qrel_or_score_count": 0,
            "individual_claim_question_answer_url_justification_or_identifier_output_count": 0,
            "private_cohort_or_secret_count": 0,
            "public_source_split_parse_count": len(split_receipts),
        },
        "cross_split_aggregate": {
            "normalized_claim_collision_group_count": len(
                cross_split_claim_hashes
            ),
            "usable_evidence_unit_overlap_group_count": len(
                cross_split_evidence_hashes
            ),
        },
        "official_commit": OFFICIAL_COMMIT,
        "recorded_date": "2026-07-26",
        "schema": "averitec_p0_public_schema_qualification_v1",
        "source_files": source_files,
        "split_receipts": split_receipts,
        "status": "qualified_public_non_scoring_schema_topology",
        "study_id": "AVERITEC_P0_PUBLIC_SCHEMA_TOPOLOGY_V1",
    }
    body["self_sha256"] = _stable_hash(body)
    return body


def write_receipt_exclusive(path: Path, receipt: Mapping[str, Any]) -> None:
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
        raise AveritecP0QualificationError(
            "aggregate receipt could not be written exclusively"
        ) from exc
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise AveritecP0QualificationError(
            "aggregate receipt mode is not 0600"
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--exposure-claim-sha256",
        action="append",
        default=[],
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    receipt = qualify_source(
        source_root=args.source_root,
        exposure_claim_sha256s=args.exposure_claim_sha256,
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
