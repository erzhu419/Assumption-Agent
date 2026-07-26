"""One-shot private cohort acquisition for AVeriTeC P1.

The public source has already passed the independent P0 schema/topology study.
This module creates one HMAC-selected, component-disjoint four-block cohort,
label-free action views, and late-open qrel packs.  It has no model, retrieval,
evaluator, score, API, or online surface.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence
import unicodedata


STUDY_ID = "AVERITEC_P1_TYPED_QA_SET_EVALUATOR_V1"
VERSION = "averitec_p1_acquisition_v1"
SOURCE_FILES: dict[str, dict[str, object]] = {
    "train": {
        "relative_path": "data/train.json",
        "size_bytes": 10_184_813,
        "file_sha256": "ae5eda7c42ddf1695ef185a7ba1bc716928f5adf57103e4f78aae5f9afe00f9c",
        "git_blob_sha1": "0f190e115cf2ee23416e8a539c8d6ac043d7cc83",
    },
    "dev": {
        "relative_path": "data/dev.json",
        "size_bytes": 1_785_475,
        "file_sha256": "499793726b4a5406780928a3d9dedc48d6dd53de778f22437d129cacdb08e300",
        "git_blob_sha1": "40974243267f395dc583d805d10f043812419249",
    },
}
P0_RECEIPT_FILE_SHA256 = (
    "618727c173ea3f03fb756e301e4d5f55f8189830f4de427841a98f01e5010686"
)
P0_RECEIPT_SELF_SHA256 = (
    "8e6fdf432bd3f79b8f9ddc11574ddaa8dc12ecfb1691477d2ab0b11691f1f272"
)
KNOWN_EXPOSURE_CLAIM_SHA256S = frozenset(
    {"24fc11fe8d979bd072bab11baa4aec69fa5d5838345b3a690ce951e6c4ba1d13"}
)

CAUSAL = "CAUSAL_CLAIM"
QUOTE = "QUOTE_VERIFICATION"
NUMERICAL = "NUMERICAL_CLAIM"
FAMILIES = (CAUSAL, QUOTE, NUMERICAL)
FAMILY_SOURCE_VALUES = {
    "causal claim": CAUSAL,
    "quote verification": QUOTE,
    "numerical claim": NUMERICAL,
}
FAMILY_PRIORITY = FAMILIES

A_FORM = "A_form"
F_SEARCH = "F_search"
A_HOLD = "A_hold"
M_SEARCH = "M_search"
BLOCK_ORDER = (A_FORM, F_SEARCH, A_HOLD, M_SEARCH)
BLOCK_SPLIT = {
    A_FORM: "train",
    F_SEARCH: "train",
    A_HOLD: "dev",
    M_SEARCH: "dev",
}
DEFAULT_BLOCK_QUOTAS: dict[str, dict[str, int]] = {
    A_FORM: {family: 36 for family in FAMILIES},
    F_SEARCH: {family: 12 for family in FAMILIES},
    A_HOLD: {family: 12 for family in FAMILIES},
    M_SEARCH: {family: 12 for family in FAMILIES},
}

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class AveritecP1AcquisitionError(RuntimeError):
    """The frozen source, selection, or private pack contract drifted."""


def canonical_bytes(value: object, *, newline: bool = True) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise AveritecP1AcquisitionError(
            "acquisition value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value, newline=False)).hexdigest()


def self_hashed(body: Mapping[str, object]) -> dict[str, object]:
    result = dict(body)
    if "self_sha256" in result:
        raise AveritecP1AcquisitionError("self hash field was already present")
    result["self_sha256"] = stable_hash(result)
    return result


def normalize_text(value: str) -> str:
    if not isinstance(value, str):
        raise AveritecP1AcquisitionError("source text is not a string")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise AveritecP1AcquisitionError(
            "bound source or receipt file cannot be hashed"
        ) from exc
    return digest.hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(  # noqa: S324 - Git object identity.
        f"blob {len(raw)}\0".encode("ascii") + raw
    ).hexdigest()


def _read_bound_source(path: Path, expected: Mapping[str, object]) -> list[object]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise AveritecP1AcquisitionError("bound source file is unavailable") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_size != expected["size_bytes"]
    ):
        raise AveritecP1AcquisitionError("bound source file metadata drifted")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AveritecP1AcquisitionError("bound source file cannot be read") from exc
    if (
        hashlib.sha256(raw).hexdigest() != expected["file_sha256"]
        or _git_blob_sha1(raw) != expected["git_blob_sha1"]
    ):
        raise AveritecP1AcquisitionError("bound source file identity drifted")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AveritecP1AcquisitionError("bound source is not strict JSON") from exc
    if not isinstance(value, list):
        raise AveritecP1AcquisitionError("bound source split root drifted")
    return value


def _read_secret(path: Path) -> bytes:
    try:
        info = path.lstat()
    except OSError as exc:
        raise AveritecP1AcquisitionError("selection secret is unavailable") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_size != 32
    ):
        raise AveritecP1AcquisitionError("selection secret metadata drifted")
    raw = path.read_bytes()
    if len(raw) != 32:
        raise AveritecP1AcquisitionError("selection secret length drifted")
    return raw


def _verify_p0_receipt(path: Path) -> dict[str, object]:
    if _sha256_file(path) != P0_RECEIPT_FILE_SHA256:
        raise AveritecP1AcquisitionError("P0 receipt file identity drifted")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AveritecP1AcquisitionError("P0 receipt is unavailable") from exc
    if not isinstance(value, dict) or canonical_bytes(value) != raw:
        raise AveritecP1AcquisitionError("P0 receipt is not canonical")
    body = dict(value)
    self_sha256 = body.pop("self_sha256", None)
    if (
        self_sha256 != P0_RECEIPT_SELF_SHA256
        or self_sha256 != stable_hash(body)
        or value.get("status") != "qualified_public_non_scoring_schema_topology"
        or value.get("study_id") != "AVERITEC_P0_PUBLIC_SCHEMA_TOPOLOGY_V1"
    ):
        raise AveritecP1AcquisitionError("P0 qualification binding drifted")
    return value


def _write_exclusive(path: Path, payload: Mapping[str, object]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = canonical_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise AveritecP1AcquisitionError(
            "private acquisition file could not be written once"
        ) from exc
    info = path.stat()
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise AveritecP1AcquisitionError("private acquisition mode drifted")
    return {
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "self_sha256": payload.get("self_sha256"),
        "size_bytes": len(raw),
    }


def _hmac_hex(secret: bytes, *parts: str) -> str:
    message = "\0".join(parts).encode("utf-8")
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def _family(claim_types: object) -> str | None:
    if not isinstance(claim_types, list):
        return None
    observed: set[str] = set()
    for value in claim_types:
        if isinstance(value, str):
            mapped = FAMILY_SOURCE_VALUES.get(normalize_text(value))
            if mapped is not None:
                observed.add(mapped)
    for family in FAMILY_PRIORITY:
        if family in observed:
            return family
    return None


@dataclass(frozen=True)
class EvidenceUnit:
    evidence_hash: str
    question: str
    answer: str


@dataclass(frozen=True)
class Candidate:
    split: str
    row_index: int
    row_hash: str
    claim_hash: str
    query_text: str
    family: str
    evidence: tuple[EvidenceUnit, ...]
    component_id: int


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


@dataclass(frozen=True)
class _ObservedRow:
    split: str
    row_index: int
    row_hash: str
    claim_hash: str | None
    query_text: str | None
    family: str | None
    evidence: tuple[EvidenceUnit, ...]


def _evidence_units(row: Mapping[str, object]) -> tuple[EvidenceUnit, ...]:
    questions = row.get("questions")
    if not isinstance(questions, list):
        return ()
    by_hash: dict[str, EvidenceUnit] = {}
    for question in questions:
        if not isinstance(question, Mapping):
            continue
        question_value = question.get("question")
        if not isinstance(question_value, str):
            continue
        normalized_question = normalize_text(question_value)
        if not normalized_question:
            continue
        answers = question.get("answers")
        if not isinstance(answers, list):
            continue
        for answer in answers:
            if not isinstance(answer, Mapping):
                continue
            answer_type = answer.get("answer_type")
            if (
                isinstance(answer_type, str)
                and normalize_text(answer_type) == "unanswerable"
            ):
                continue
            answer_value = answer.get("answer")
            if not isinstance(answer_value, str):
                continue
            normalized_answer = normalize_text(answer_value)
            if not normalized_answer:
                continue
            evidence_hash = hashlib.sha256(
                (
                    normalized_question
                    + "\0"
                    + normalized_answer
                ).encode("utf-8")
            ).hexdigest()
            by_hash.setdefault(
                evidence_hash,
                EvidenceUnit(
                    evidence_hash=evidence_hash,
                    question=" ".join(question_value.split()),
                    answer=" ".join(answer_value.split()),
                ),
            )
    return tuple(by_hash[key] for key in sorted(by_hash))


def _observe_rows(
    split_rows: Mapping[str, Sequence[object]],
) -> tuple[list[_ObservedRow], dict[str, int]]:
    observed: list[_ObservedRow] = []
    anomaly = {
        "ineligible_claim": 0,
        "ineligible_family": 0,
        "ineligible_evidence_cardinality": 0,
        "known_exposure": 0,
        "row_not_object": 0,
    }
    for split in sorted(split_rows):
        for row_index, raw_row in enumerate(split_rows[split]):
            if not isinstance(raw_row, Mapping):
                anomaly["row_not_object"] += 1
                observed.append(
                    _ObservedRow(
                        split, row_index, stable_hash(raw_row), None, None, None, ()
                    )
                )
                continue
            row = dict(raw_row)
            row_hash = stable_hash(row)
            claim = row.get("claim")
            query_text: str | None = None
            claim_hash: str | None = None
            if isinstance(claim, str) and normalize_text(claim):
                query_text = " ".join(claim.split())
                if len(query_text) <= 4_000 and "\x00" not in query_text:
                    claim_hash = hashlib.sha256(
                        normalize_text(claim).encode("utf-8")
                    ).hexdigest()
            if claim_hash is None:
                anomaly["ineligible_claim"] += 1
            if claim_hash in KNOWN_EXPOSURE_CLAIM_SHA256S:
                anomaly["known_exposure"] += 1
            family = _family(row.get("claim_types"))
            if family is None:
                anomaly["ineligible_family"] += 1
            evidence = _evidence_units(row)
            if len(evidence) < 2:
                anomaly["ineligible_evidence_cardinality"] += 1
            observed.append(
                _ObservedRow(
                    split=split,
                    row_index=row_index,
                    row_hash=row_hash,
                    claim_hash=claim_hash,
                    query_text=query_text,
                    family=family,
                    evidence=evidence,
                )
            )
    return observed, anomaly


def _component_candidates(
    split_rows: Mapping[str, Sequence[object]],
) -> tuple[list[Candidate], dict[str, object]]:
    observed, anomaly = _observe_rows(split_rows)
    dsu = _DisjointSet(len(observed))
    owner: dict[tuple[str, str], int] = {}
    for index, row in enumerate(observed):
        keys: list[tuple[str, str]] = []
        if row.claim_hash is not None:
            keys.append(("claim", row.claim_hash))
        keys.extend(("evidence", unit.evidence_hash) for unit in row.evidence)
        for key in keys:
            prior = owner.get(key)
            if prior is None:
                owner[key] = index
            else:
                dsu.union(index, prior)
    component_splits: dict[int, set[str]] = {}
    for index, row in enumerate(observed):
        component_splits.setdefault(dsu.find(index), set()).add(row.split)
    cross_split_components = {
        component
        for component, splits in component_splits.items()
        if len(splits) > 1
    }
    candidates: list[Candidate] = []
    eligible_counts = {
        split: {family: 0 for family in FAMILIES}
        for split in split_rows
    }
    excluded_cross_split_count = 0
    for index, row in enumerate(observed):
        component = dsu.find(index)
        if component in cross_split_components:
            excluded_cross_split_count += 1
            continue
        if (
            row.claim_hash is None
            or row.query_text is None
            or row.claim_hash in KNOWN_EXPOSURE_CLAIM_SHA256S
            or row.family not in FAMILIES
            or len(row.evidence) < 2
        ):
            continue
        candidate = Candidate(
            split=row.split,
            row_index=row.row_index,
            row_hash=row.row_hash,
            claim_hash=row.claim_hash,
            query_text=row.query_text,
            family=row.family,
            evidence=row.evidence,
            component_id=component,
        )
        candidates.append(candidate)
        eligible_counts[row.split][row.family] += 1
    aggregate = {
        "cross_split_component_count": len(cross_split_components),
        "cross_split_row_exclusion_count": excluded_cross_split_count,
        "eligible_row_count_by_split_and_family": eligible_counts,
        "observed_component_count": len(
            {dsu.find(index) for index in range(len(observed))}
        ),
        "source_anomaly_or_ineligibility_count": anomaly,
        "source_row_count": {
            split: len(rows) for split, rows in sorted(split_rows.items())
        },
    }
    return candidates, aggregate


def _validate_quotas(
    block_quotas: Mapping[str, Mapping[str, int]],
) -> None:
    if tuple(block_quotas) != BLOCK_ORDER:
        raise AveritecP1AcquisitionError("block quota order drifted")
    for block in BLOCK_ORDER:
        if tuple(block_quotas[block]) != FAMILIES or any(
            type(value) is not int or value <= 0
            for value in block_quotas[block].values()
        ):
            raise AveritecP1AcquisitionError("block family quota drifted")


def _select(
    *,
    candidates: Sequence[Candidate],
    secret: bytes,
    block_quotas: Mapping[str, Mapping[str, int]],
) -> dict[str, tuple[Candidate, ...]]:
    _validate_quotas(block_quotas)
    used_components: set[int] = set()
    selected: dict[str, tuple[Candidate, ...]] = {}
    for block in BLOCK_ORDER:
        split = BLOCK_SPLIT[block]
        block_rows: list[Candidate] = []
        for family in FAMILIES:
            eligible = [
                row
                for row in candidates
                if row.split == split
                and row.family == family
                and row.component_id not in used_components
            ]
            eligible.sort(
                key=lambda row: (
                    _hmac_hex(
                        secret,
                        "selection_order",
                        block,
                        family,
                        row.row_hash,
                    ),
                    row.row_index,
                )
            )
            quota = block_quotas[block][family]
            chosen: list[Candidate] = []
            local_components: set[int] = set()
            for row in eligible:
                if row.component_id in local_components:
                    continue
                chosen.append(row)
                local_components.add(row.component_id)
                if len(chosen) == quota:
                    break
            if len(chosen) != quota:
                raise AveritecP1AcquisitionError(
                    "frozen component-disjoint family quota is infeasible"
                )
            block_rows.extend(chosen)
            used_components.update(local_components)
        block_rows.sort(
            key=lambda row: (
                _hmac_hex(secret, "query_order", block, row.row_hash),
                row.row_index,
            )
        )
        selected[block] = tuple(block_rows)
    return selected


def _block_payloads(
    *,
    block: str,
    rows: Sequence[Candidate],
    secret: bytes,
) -> tuple[dict[str, object], dict[str, object] | None]:
    evidence_by_hash: dict[str, EvidenceUnit] = {}
    for row in rows:
        for unit in row.evidence:
            evidence_by_hash.setdefault(unit.evidence_hash, unit)
    evidence_order = sorted(
        evidence_by_hash,
        key=lambda evidence_hash: (
            _hmac_hex(secret, "document_order", block, evidence_hash),
            evidence_hash,
        ),
    )
    ordinal_by_hash = {
        evidence_hash: ordinal
        for ordinal, evidence_hash in enumerate(evidence_order)
    }
    corpus = []
    for ordinal, evidence_hash in enumerate(evidence_order):
        unit = evidence_by_hash[evidence_hash]
        corpus.append(
            {
                "body": unit.answer,
                "document_id": _hmac_hex(
                    secret, "document_id", block, evidence_hash
                ),
                "ordinal": ordinal,
                "title": unit.question,
            }
        )
    queries = []
    qrel_rows = []
    for ordinal, row in enumerate(rows):
        item_id = _hmac_hex(
            secret, "item_id", block, row.split, str(row.row_index), row.row_hash
        )
        queries.append(
            {
                "item_id": item_id,
                "ordinal": ordinal,
                "text": row.query_text,
            }
        )
        qrel_rows.append(
            {
                "family": row.family,
                "item_id": item_id,
                "qrel_document_ordinals": sorted(
                    ordinal_by_hash[unit.evidence_hash]
                    for unit in row.evidence
                ),
            }
        )
    view = self_hashed(
        {
            "block": block,
            "corpus": corpus,
            "queries": queries,
            "schema": f"{VERSION}_label_free_action_view_v1",
            "study_id": STUDY_ID,
        }
    )
    if block == F_SEARCH:
        return view, None
    qrels = self_hashed(
        {
            "block": block,
            "rows": qrel_rows,
            "schema": f"{VERSION}_late_qrel_pack_v1",
            "study_id": STUDY_ID,
        }
    )
    return view, qrels


def acquire_from_rows(
    *,
    train_rows: Sequence[object],
    dev_rows: Sequence[object],
    secret: bytes,
    block_quotas: Mapping[str, Mapping[str, int]] = DEFAULT_BLOCK_QUOTAS,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    if not isinstance(secret, bytes) or len(secret) != 32:
        raise AveritecP1AcquisitionError("selection secret drifted")
    candidates, qualification = _component_candidates(
        {"train": train_rows, "dev": dev_rows}
    )
    selected = _select(
        candidates=candidates,
        secret=secret,
        block_quotas=block_quotas,
    )
    payloads: dict[str, dict[str, object]] = {}
    block_aggregate: dict[str, object] = {}
    for block in BLOCK_ORDER:
        view, qrels = _block_payloads(
            block=block, rows=selected[block], secret=secret
        )
        payloads[f"{block}.view"] = view
        if qrels is not None:
            payloads[f"{block}.qrels"] = qrels
        family_counts = {family: 0 for family in FAMILIES}
        for row in selected[block]:
            family_counts[row.family] += 1
        block_aggregate[block] = {
            "corpus_document_count": len(view["corpus"]),  # type: ignore[arg-type]
            "family_count": family_counts,
            "qrel_pack_created": qrels is not None,
            "query_count": len(view["queries"]),  # type: ignore[arg-type]
        }
    aggregate = {
        "block_aggregate": block_aggregate,
        "qualification": qualification,
        "selected_component_count": sum(
            len(rows) for rows in selected.values()
        ),
    }
    return payloads, aggregate


def run_acquisition(
    *,
    source_root: Path,
    p0_receipt_path: Path,
    secret_path: Path,
    attempt_marker_path: Path,
    output_root: Path,
    execution_binding_sha256: str,
) -> dict[str, object]:
    if not _HEX64.fullmatch(execution_binding_sha256):
        raise AveritecP1AcquisitionError("execution binding is invalid")
    if output_root.exists():
        raise AveritecP1AcquisitionError("acquisition output root is not fresh")
    _verify_p0_receipt(p0_receipt_path)
    secret = _read_secret(secret_path)
    secret_commitment = hashlib.sha256(secret).hexdigest()
    attempt = self_hashed(
        {
            "execution_binding_sha256": execution_binding_sha256,
            "schema": f"{VERSION}_attempt_marker_v1",
            "secret_commitment_sha256": secret_commitment,
            "study_id": STUDY_ID,
        }
    )
    _write_exclusive(attempt_marker_path, attempt)

    split_rows = {
        split: _read_bound_source(
            source_root / str(expected["relative_path"]), expected
        )
        for split, expected in sorted(SOURCE_FILES.items())
    }
    payloads, aggregate = acquire_from_rows(
        train_rows=split_rows["train"],
        dev_rows=split_rows["dev"],
        secret=secret,
    )
    output_root.mkdir(parents=True, mode=0o700)
    files: dict[str, object] = {}
    for name in sorted(payloads):
        files[name] = _write_exclusive(
            output_root / f"{name}.json", payloads[name]
        )
    receipt = self_hashed(
        {
            "access_boundary": {
                "action_model_retrieval_evaluator_or_score_count": 0,
                "F_search_qrel_pack_created": False,
                "individual_source_or_selected_item_value_published": False,
                "private_secret_output_count": 0,
                "source_split_parse_count": 2,
            },
            "aggregate": aggregate,
            "execution_binding_sha256": execution_binding_sha256,
            "files": files,
            "p0_qualification_receipt_self_sha256": P0_RECEIPT_SELF_SHA256,
            "recorded_date": "2026-07-26",
            "schema": f"{VERSION}_safe_receipt_v1",
            "secret_commitment_sha256": secret_commitment,
            "source_files": {
                split: {
                    "file_sha256": expected["file_sha256"],
                    "git_blob_sha1": expected["git_blob_sha1"],
                    "size_bytes": expected["size_bytes"],
                }
                for split, expected in sorted(SOURCE_FILES.items())
            },
            "status": "selected_component_disjoint_four_block_private_cohort",
            "study_id": STUDY_ID,
        }
    )
    _write_exclusive(output_root / "acquisition.safe_receipt.json", receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--p0-receipt", required=True, type=Path)
    parser.add_argument("--secret", required=True, type=Path)
    parser.add_argument("--attempt-marker", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--execution-binding-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    receipt = run_acquisition(
        source_root=arguments.source_root,
        p0_receipt_path=arguments.p0_receipt,
        secret_path=arguments.secret,
        attempt_marker_path=arguments.attempt_marker,
        output_root=arguments.output_root,
        execution_binding_sha256=arguments.execution_binding_sha256,
    )
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "self_sha256": receipt["self_sha256"],
                "status": receipt["status"],
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
