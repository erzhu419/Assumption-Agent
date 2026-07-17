"""One-shot FEVER fixed-P item-local acquisition.

The formal CLI has three canonical phases. ``freeze`` verifies only committed
public protocol files and opaque whole-file source bindings before writing an
external implementation freeze. ``secret`` consumes its own durable marker,
creates one 32-byte secret, and publishes only its commitment. ``acquire``
verifies the now-committed freeze and secret custody, consumes the unique
acquisition marker, and only then parses the official labelled paper-test file
and the official preprocessed Wikipedia archive.

Unit tests use synthetic JSONL/ZIP fixtures.  Importing this module never
locates or opens FEVER assets.
"""

from __future__ import annotations

import argparse
from bisect import insort
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import stat
import subprocess
from typing import Any, Iterable, Iterator, Mapping, Sequence
import unicodedata
import zipfile


VERSION = "fever_fixed_p_itemlocal_acquisition_v1"
DOMAIN_SEPARATOR = "fever_fixed_p_itemlocal_reranking_v1"

SOURCE_CUSTODY_RELATIVE = Path(
    "manifests/fever_official_fixed_transfer_source_custody_v1.json"
)
SOURCE_ACCESS_RELATIVE = Path(
    "manifests/fever_official_fixed_transfer_source_access_v1.json"
)
DESIGN_RELATIVE = Path(
    "manifests/fever_fixed_p_itemlocal_reranking_design_v1.json"
)
IMPLEMENTATION_FREEZE_RELATIVE = Path(
    "manifests/fever_official_fixed_transfer_implementation_freeze_v1.json"
)
SELECTION_CUSTODY_RELATIVE = Path(
    "manifests/fever_official_fixed_transfer_selection_custody_v1.json"
)
SELECTION_FAILURE_RELATIVE = Path(
    "manifests/fever_official_fixed_transfer_selection_failure_v1.json"
)
ACQUISITION_RECEIPT_RELATIVE = Path(
    "manifests/fever_official_fixed_transfer_acquisition_v1.json"
)
ACQUISITION_FAILURE_RELATIVE = Path(
    "manifests/fever_official_fixed_transfer_acquisition_failure_v1.json"
)

ARTIFACT_ROOT_RELATIVE = Path("artifacts/fever_official_fixed_transfer_v1")
ACTION_PACK_RELATIVE = ARTIFACT_ROOT_RELATIVE / "action_pack.json"
LABEL_PACK_RELATIVE = ARTIFACT_ROOT_RELATIVE / "label_pack.json"
SELECTION_SECRET_RELATIVE = ARTIFACT_ROOT_RELATIVE / "selection.key"
SECRET_MARKER_RELATIVE = ARTIFACT_ROOT_RELATIVE / "secret.attempt.marker"
ACQUISITION_MARKER_RELATIVE = ARTIFACT_ROOT_RELATIVE / "acquisition.attempt.marker"
IDENTITY_LEDGER_RELATIVE = ARTIFACT_ROOT_RELATIVE / "wiki_identity_ledger.sqlite3"
WORK_ROOT_RELATIVE = ARTIFACT_ROOT_RELATIVE / "work"

PAPER_TEST_RELATIVE = Path("artifacts/fever_official_source_v1/paper_test.jsonl")
WIKI_ZIP_RELATIVE = Path("artifacts/fever_official_source_v1/wiki-pages.zip")
LICENSE_RELATIVE = Path("artifacts/fever_official_source_v1/license.html")

SOURCE_CUSTODY_SCHEMA = "fever_official_fixed_transfer_source_custody_v1"
SOURCE_CUSTODY_FILE_SHA256 = (
    "5595339d3b089b9cd278382381225946f6c0409ba6c64395d376dae3cb0c7a9e"
)
SOURCE_CUSTODY_SHA256 = (
    "ba0a4aca54a06d8b29e851120a68fc2ca0fb28bc58f3066ffb1fcfcad4957050"
)
SOURCE_ACCESS_SCHEMA = "fever_official_fixed_transfer_source_access_v1"
SOURCE_ACCESS_FILE_SHA256 = (
    "a982c7104329b3ae35cbe9d5698b9597bf66208423a32bcd55115532aae45596"
)
SOURCE_ACCESS_SHA256 = (
    "05d7ef2a16f30349d145fd84c775a61d870e9fc15e34e23b37b9967b0586c0b8"
)
DESIGN_SCHEMA = "fever_fixed_p_itemlocal_reranking_design_v1"
DESIGN_FILE_SHA256 = (
    "b9b04f607b78a6b24d678de5dea1eff8c097c6a7f393100ceffd8dfe179c822b"
)
DESIGN_SHA256 = (
    "d000802fdc2a56aa8d91991abd013101a33aa147d89225d292b31b60b4d014aa"
)

ASSET_EXPECTATIONS: dict[str, dict[str, object]] = {
    "paper_test": {
        "relative_path": PAPER_TEST_RELATIVE,
        "size_bytes": 2_181_168,
        "file_sha256": (
            "fb7b0280a0adc2302bbb29bfb7af37274fa585de3171bcf908f180642d11d88e"
        ),
    },
    "wiki_pages": {
        "relative_path": WIKI_ZIP_RELATIVE,
        "size_bytes": 1_713_485_474,
        "file_sha256": (
            "4b06d95da6adf7fe02d2796176c670dacccb21348da89cba4c50676ab99665f2"
        ),
    },
    "license": {
        "relative_path": LICENSE_RELATIVE,
        "size_bytes": 670,
        "file_sha256": (
            "eaddbd6e2854b45e6925bac9171fb6a2438492dc0efd4fda80144e836ae1ad05"
        ),
    },
}

REQUIRED_FREEZE_PATHS = (
    SOURCE_CUSTODY_RELATIVE.as_posix(),
    SOURCE_ACCESS_RELATIVE.as_posix(),
    DESIGN_RELATIVE.as_posix(),
    "assumption_agent/benchmarks/fever_fixed_p_itemlocal_acquisition_v1.py",
    "tests/test_fever_fixed_p_itemlocal_acquisition_v1.py",
    "assumption_agent/benchmarks/fever_fixed_p_itemlocal_runner_v1.py",
    "tests/test_fever_fixed_p_itemlocal_runner_v1.py",
    "manifests/musique_recursive_study_f1_formation_v1/frozen_program.json",
    "manifests/musique_recursive_study_f1_formation_v1/formation.receipt.json",
    "assumption_agent/benchmarks/musique_typed_retriever_formation_v1.py",
    "assumption_agent/benchmarks/musique_recursive_study_blocks_v1.py",
    "manifests/musique_official_hipporag_runtime_attestation_v2.json",
    "manifests/musique_official_hipporag_retrieve_only_binding_v1.json",
    "assumption_agent/benchmarks/musique_formal_runtime_binding_v2.py",
    "replication_runtime/musique_official_hipporag_v1/adapter_v2.py",
    "replication_runtime/musique_official_hipporag_v1/runtime_attestation_v2.py",
    "replication_runtime/musique_official_hipporag_v1/contract.py",
    "replication_runtime/musique_official_hipporag_v1/adapter.py",
    "replication_runtime/musique_official_hipporag_v1/binding.py",
    "replication_runtime/musique_official_hipporag_v1/worker.py",
    "assumption_agent/models.py",
)

IMPLEMENTATION_FREEZE_SCHEMA = "fever_official_fixed_transfer_implementation_freeze_v1"
SELECTION_CUSTODY_SCHEMA = "fever_official_fixed_transfer_selection_custody_v1"
ACQUISITION_SCHEMA = "fever_official_fixed_transfer_acquisition_v1"
FAILURE_SCHEMA = "fever_official_fixed_transfer_terminal_failure_v1"
ACTION_PACK_SCHEMA = "fever_fixed_p_itemlocal_action_pack_v1"
ACTION_ITEM_SCHEMA = "fever_fixed_p_itemlocal_action_item_v1"
LABEL_PACK_SCHEMA = "fever_fixed_p_itemlocal_label_pack_v1"
LABEL_ITEM_SCHEMA = "fever_fixed_p_itemlocal_label_item_v1"
ACQUISITION_STATUS = "formal_itemlocal_pack_acquired"

LABEL_ORDER = ("SUPPORTS", "REFUTES")
LABEL_COUNT = 64
COHORT_COUNT = 128
DOCUMENT_COUNT = 32
BM25_K1 = 1.2
BM25_B = 0.75
BM25_QUANTIZATION = 1_000_000_000_000
TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")
MAX_PUBLIC_BYTES = 4 * 1024 * 1024
MAX_JSONL_LINE_BYTES = 16 * 1024 * 1024


class FeverAcquisitionError(RuntimeError):
    """A frozen FEVER acquisition contract drifted."""


@dataclass(frozen=True, order=True)
class EvidenceRef:
    page_id: str
    line_number: int


@dataclass(frozen=True)
class PaperCandidate:
    source_row_ordinal: int
    exact_id: int | str
    item_id_hash: str
    source_label: str
    claim: str
    identity_commitment_sha256: str
    eligible_sets: tuple[tuple[EvidenceRef, ...], ...]
    all_annotated_refs: frozenset[EvidenceRef]


@dataclass(frozen=True)
class SelectedCandidate:
    candidate: PaperCandidate
    selected_set: tuple[EvidenceRef, ...]


@dataclass(frozen=True)
class WikiSentence:
    page_id: str
    line_number: int
    sentence_text: str


@dataclass(frozen=True)
class BM25Statistics:
    document_count: int
    total_token_count: int
    average_document_length: float
    document_frequency: Mapping[str, int]


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise FeverAcquisitionError("value is not canonical JSON") from exc


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise FeverAcquisitionError(f"{field} must be lowercase SHA256")
    return value


def _canonical_project(project: Path) -> Path:
    absolute = project.expanduser().absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise FeverAcquisitionError("project path contains a symlink")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise FeverAcquisitionError("project is unavailable") from exc
    if resolved != absolute or not resolved.is_dir():
        raise FeverAcquisitionError("project path is not canonical")
    return resolved


def _safe_path(project: Path, relative: Path, field: str) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise FeverAcquisitionError(f"{field} relative path is unsafe")
    path = project / relative
    for component in (*reversed(path.parents), path):
        if component.is_symlink():
            raise FeverAcquisitionError(f"{field} path contains a symlink")
    return path


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, raw: bytes, *, mode: int) -> None:
    if path.exists() or path.is_symlink():
        raise FeverAcquisitionError("canonical output already exists; replay is forbidden")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    os.chmod(path, mode)
    _fsync_directory(path.parent)


def _with_hash(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise FeverAcquisitionError("self-hash field already exists")
    normalized = dict(body)
    return {**normalized, field: _semantic_hash(normalized)}


def _write_json(path: Path, payload: Mapping[str, Any], *, mode: int) -> str:
    raw = _canonical_bytes(payload) + b"\n"
    _write_exclusive(path, raw, mode=mode)
    return _sha256_bytes(raw)


def _write_private_json(path: Path, payload: Mapping[str, Any]) -> str:
    raw = _canonical_bytes(payload)
    _write_exclusive(path, raw, mode=0o600)
    return _sha256_bytes(raw)


def normalize_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def tokenize(value: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(normalize_text(value)))


def _hmac_digest(secret: bytes, *parts: str) -> bytes:
    if len(secret) != 32:
        raise FeverAcquisitionError("selection secret must contain exactly 32 bytes")
    message = "\0".join((DOMAIN_SEPARATOR, *parts)).encode("utf-8")
    return hmac.new(secret, message, hashlib.sha256).digest()


def _canonical_evidence_bytes(refs: Sequence[EvidenceRef]) -> bytes:
    return _canonical_bytes(
        [[ref.page_id, ref.line_number] for ref in refs]
    )


def _parse_evidence(
    raw: object,
) -> tuple[tuple[tuple[EvidenceRef, ...], ...], frozenset[EvidenceRef]]:
    if not isinstance(raw, list):
        raise FeverAcquisitionError("paper evidence must be a list")
    eligible: set[tuple[EvidenceRef, ...]] = set()
    annotated: set[EvidenceRef] = set()
    for raw_set in raw:
        if not isinstance(raw_set, list):
            raise FeverAcquisitionError("paper evidence set must be a list")
        complete = True
        refs: set[EvidenceRef] = set()
        for entry in raw_set:
            if not isinstance(entry, list) or len(entry) != 4:
                raise FeverAcquisitionError("paper evidence entry schema drifted")
            annotation_id, evidence_id, page_id, line_number = entry
            if (
                annotation_id is not None
                and type(annotation_id) is not int
            ) or (evidence_id is not None and type(evidence_id) is not int):
                raise FeverAcquisitionError("paper evidence IDs drifted")
            if page_id is None or line_number is None:
                complete = False
                continue
            if (
                not isinstance(page_id, str)
                or not page_id
                or "\x00" in page_id
                or type(line_number) is not int
                or line_number < 0
            ):
                raise FeverAcquisitionError("paper evidence reference drifted")
            ref = EvidenceRef(page_id, line_number)
            refs.add(ref)
            annotated.add(ref)
        canonical = tuple(sorted(refs))
        if complete and 1 <= len(canonical) <= 5:
            eligible.add(canonical)
    return (
        tuple(sorted(eligible, key=_canonical_evidence_bytes)),
        frozenset(annotated),
    )


def _parse_paper_row(raw: object, ordinal: int) -> PaperCandidate | None:
    if not isinstance(raw, Mapping) or set(raw) != {
        "id",
        "verifiable",
        "label",
        "claim",
        "evidence",
    }:
        raise FeverAcquisitionError("paper row schema drifted")
    exact_id = raw.get("id")
    if not (
        type(exact_id) is int
        or (isinstance(exact_id, str) and exact_id and "\x00" not in exact_id)
    ):
        raise FeverAcquisitionError("paper row ID drifted")
    label = raw.get("label")
    claim = raw.get("claim")
    if label not in {*LABEL_ORDER, "NOT ENOUGH INFO"}:
        raise FeverAcquisitionError("paper label drifted")
    if (
        raw.get("verifiable") not in {"VERIFIABLE", "NOT VERIFIABLE"}
        or not isinstance(claim, str)
        or not claim.strip()
        or "\x00" in claim
    ):
        raise FeverAcquisitionError("paper claim or verifiability drifted")
    eligible_sets, annotated = _parse_evidence(raw.get("evidence"))
    if label not in LABEL_ORDER or not eligible_sets:
        return None
    identity_body = {
        "source_member": "official_labelled_paper_test",
        "source_row_ordinal": ordinal,
        "exact_id": exact_id,
        "exact_claim": claim,
        "full_canonical_source_row": dict(raw),
    }
    return PaperCandidate(
        source_row_ordinal=ordinal,
        exact_id=exact_id,
        item_id_hash=_semantic_hash(exact_id),
        source_label=str(label),
        claim=claim,
        identity_commitment_sha256=_semantic_hash(identity_body),
        eligible_sets=eligible_sets,
        all_annotated_refs=annotated,
    )


def load_paper_candidates(path: Path) -> tuple[tuple[PaperCandidate, ...], int]:
    candidates: list[PaperCandidate] = []
    row_count = 0
    seen_ids: set[tuple[type, int | str]] = set()
    with path.open("rb") as handle:
        for ordinal, line in enumerate(handle):
            if not line.endswith(b"\n") or not line.strip() or len(line) > MAX_JSONL_LINE_BYTES:
                raise FeverAcquisitionError("paper JSONL framing drifted")
            try:
                raw = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise FeverAcquisitionError("paper JSONL decode failed") from exc
            candidate = _parse_paper_row(raw, ordinal)
            exact_id = raw.get("id") if isinstance(raw, Mapping) else None
            identity = (type(exact_id), exact_id)
            if identity in seen_ids:
                raise FeverAcquisitionError("paper row IDs are not unique")
            seen_ids.add(identity)
            if candidate is not None:
                candidates.append(candidate)
            row_count += 1
    if row_count == 0:
        raise FeverAcquisitionError("paper source is empty")
    return tuple(candidates), row_count


def select_candidates(
    candidates: Sequence[PaperCandidate], secret: bytes
) -> tuple[tuple[SelectedCandidate, ...], dict[str, object]]:
    by_label = {label: [] for label in LABEL_ORDER}
    for row in candidates:
        by_label[row.source_label].append(row)
    selected: list[SelectedCandidate] = []
    eligible_counts: dict[str, int] = {}
    evidence_histogram: Counter[int] = Counter()
    for label in LABEL_ORDER:
        eligible_counts[label] = len(by_label[label])
        ordered = sorted(
            by_label[label],
            key=lambda row: (
                _hmac_digest(
                    secret,
                    "row_rank",
                    label,
                    row.identity_commitment_sha256,
                ),
                row.identity_commitment_sha256,
            ),
        )
        if len(ordered) < LABEL_COUNT:
            raise FeverAcquisitionError(
                f"source capacity insufficient for exact {label} cohort"
            )
        for row in ordered[:LABEL_COUNT]:
            chosen = min(
                row.eligible_sets,
                key=lambda refs: (
                    _hmac_digest(
                        secret,
                        "evidence_set",
                        row.identity_commitment_sha256,
                        _sha256_bytes(_canonical_evidence_bytes(refs)),
                    ),
                    _canonical_evidence_bytes(refs),
                ),
            )
            selected.append(SelectedCandidate(row, chosen))
            evidence_histogram[len(chosen)] += 1
    if len(selected) != COHORT_COUNT or len(
        {row.candidate.identity_commitment_sha256 for row in selected}
    ) != COHORT_COUNT:
        raise FeverAcquisitionError("selected paper identities overlap")
    return tuple(selected), {
        "eligible_counts": eligible_counts,
        "selected_label_counts": {label: LABEL_COUNT for label in LABEL_ORDER},
        "selected_evidence_set_cardinality_histogram": {
            str(key): evidence_histogram[key] for key in sorted(evidence_histogram)
        },
    }


def bm25_score_int(
    query_tokens: Sequence[str],
    document_tokens: Sequence[str],
    statistics: BM25Statistics,
) -> int:
    if statistics.document_count <= 0 or statistics.average_document_length <= 0:
        raise FeverAcquisitionError("BM25 corpus statistics are invalid")
    frequencies = Counter(document_tokens)
    document_length = len(document_tokens)
    score = 0.0
    for token in sorted(set(query_tokens)):
        frequency = frequencies.get(token, 0)
        if frequency == 0:
            continue
        document_frequency = statistics.document_frequency.get(token, 0)
        idf = math.log1p(
            (statistics.document_count - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )
        denominator = frequency + BM25_K1 * (
            1.0
            - BM25_B
            + BM25_B * document_length / statistics.average_document_length
        )
        score += idf * frequency * (BM25_K1 + 1.0) / denominator
    return int(round(score * BM25_QUANTIZATION))


def _safe_wiki_members(archive: zipfile.ZipFile) -> tuple[zipfile.ZipInfo, ...]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise FeverAcquisitionError("wiki ZIP member names are not unique")
    accepted: list[zipfile.ZipInfo] = []
    for info in infos:
        path = PurePosixPath(info.filename)
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        if (
            info.flag_bits & 0x1
            or path.is_absolute()
            or ".." in path.parts
            or stat.S_ISLNK(unix_mode)
        ):
            raise FeverAcquisitionError("wiki ZIP member is unsafe")
        if info.is_dir():
            continue
        if path.suffix != ".jsonl":
            raise FeverAcquisitionError("wiki ZIP contains an unexpected member")
        accepted.append(info)
    if not accepted:
        raise FeverAcquisitionError("wiki ZIP has no JSONL members")
    return tuple(sorted(accepted, key=lambda info: info.filename.encode("utf-8")))


def _wiki_row_sentences(raw: object) -> tuple[WikiSentence, ...]:
    if not isinstance(raw, Mapping) or set(raw) != {"id", "text", "lines"}:
        raise FeverAcquisitionError("wiki page schema drifted")
    page_id = raw.get("id")
    lines = raw.get("lines")
    if (
        not isinstance(page_id, str)
        or not page_id
        or "\x00" in page_id
        or not isinstance(raw.get("text"), str)
        or not isinstance(lines, str)
    ):
        raise FeverAcquisitionError("wiki page fields drifted")
    result: list[WikiSentence] = []
    seen: set[int] = set()
    for raw_line in lines.splitlines():
        fields = raw_line.split("\t")
        if len(fields) < 2:
            raise FeverAcquisitionError("wiki sentence line schema drifted")
        try:
            line_number = int(fields[0], 10)
        except ValueError as exc:
            raise FeverAcquisitionError("wiki line number drifted") from exc
        sentence_text = fields[1]
        if line_number < 0 or line_number in seen or "\x00" in sentence_text:
            raise FeverAcquisitionError("wiki sentence identity drifted")
        seen.add(line_number)
        if sentence_text.strip():
            result.append(WikiSentence(page_id, line_number, sentence_text))
    return tuple(result)


def iter_wiki_sentences(path: Path) -> Iterator[WikiSentence]:
    """Stream every eligible sentence in deterministic member/row order."""

    with zipfile.ZipFile(path, "r") as archive:
        for info in _safe_wiki_members(archive):
            with archive.open(info, "r") as handle:
                for line in handle:
                    if (
                        not line.endswith(b"\n")
                        or not line.strip()
                        or len(line) > MAX_JSONL_LINE_BYTES
                    ):
                        raise FeverAcquisitionError("wiki JSONL framing drifted")
                    try:
                        raw = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise FeverAcquisitionError("wiki JSONL decode failed") from exc
                    yield from _wiki_row_sentences(raw)


def scan_wiki_pass1(
    *,
    wiki_zip: Path,
    selected: Sequence[SelectedCandidate],
    identity_ledger_path: Path,
) -> tuple[BM25Statistics, dict[EvidenceRef, WikiSentence], dict[str, int]]:
    """First full wiki pass: corpus BM25 statistics and chosen-gold resolve."""

    query_union = sorted(
        set().union(*(set(tokenize(row.candidate.claim)) for row in selected))
    )
    if not query_union:
        raise FeverAcquisitionError("selected claims have no BM25 query tokens")
    query_token_set = set(query_union)
    required_refs = {
        ref for row in selected for ref in row.selected_set
    }
    resolved: dict[EvidenceRef, WikiSentence] = {}
    document_frequency: Counter[str] = Counter()
    document_count = 0
    total_token_count = 0
    if identity_ledger_path.exists() or identity_ledger_path.is_symlink():
        raise FeverAcquisitionError("wiki identity ledger already exists")
    identity_ledger_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        identity_ledger_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    os.close(descriptor)
    connection = sqlite3.connect(identity_ledger_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(
            "CREATE TABLE identities (page_id TEXT NOT NULL, "
            "line_number INTEGER NOT NULL, PRIMARY KEY(page_id,line_number)) "
            "WITHOUT ROWID"
        )
        connection.execute("BEGIN")
        for sentence in iter_wiki_sentences(wiki_zip):
            tokens = tokenize(sentence.sentence_text)
            if not tokens:
                continue
            try:
                connection.execute(
                    "INSERT INTO identities(page_id,line_number) VALUES(?,?)",
                    (sentence.page_id, sentence.line_number),
                )
            except sqlite3.IntegrityError as exc:
                raise FeverAcquisitionError(
                    "wiki sentence identity is duplicated"
                ) from exc
            document_count += 1
            total_token_count += len(tokens)
            for token in set(tokens) & query_token_set:
                document_frequency[token] += 1
            ref = EvidenceRef(sentence.page_id, sentence.line_number)
            if ref in required_refs:
                resolved[ref] = sentence
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()
        os.chmod(identity_ledger_path, 0o600)
    if document_count == 0 or total_token_count == 0:
        raise FeverAcquisitionError("wiki has no eligible sentences")
    missing = required_refs - set(resolved)
    if missing:
        raise FeverAcquisitionError(
            "chosen evidence reference is unresolved; replacement is forbidden"
        )
    average = total_token_count / document_count
    statistics = BM25Statistics(
        document_count=document_count,
        total_token_count=total_token_count,
        average_document_length=average,
        document_frequency={
            token: document_frequency[token] for token in query_union
        },
    )
    return statistics, resolved, {
        "document_count": document_count,
        "total_token_count": total_token_count,
        "query_union_token_count": len(query_union),
        "chosen_reference_count": len(required_refs),
        "chosen_reference_resolved_count": len(resolved),
        "identity_ledger_row_count": document_count,
    }


def mine_hard_negatives_pass2(
    *,
    wiki_zip: Path,
    selected: Sequence[SelectedCandidate],
    statistics: BM25Statistics,
) -> dict[int, tuple[tuple[int, WikiSentence], ...]]:
    """Second full wiki pass retaining exact top non-annotated BM25 negatives."""

    queries = [tuple(sorted(set(tokenize(row.candidate.claim)))) for row in selected]
    by_token: dict[str, set[int]] = defaultdict(set)
    for item_i, query in enumerate(queries):
        for token in query:
            by_token[token].add(item_i)
    exclusions = [row.candidate.all_annotated_refs for row in selected]
    required = [DOCUMENT_COUNT - len(row.selected_set) for row in selected]
    retained: list[list[tuple[tuple[object, ...], int, WikiSentence]]] = [
        [] for _ in selected
    ]
    for sentence in iter_wiki_sentences(wiki_zip):
        tokens = tokenize(sentence.sentence_text)
        if not tokens:
            continue
        impacted: set[int] = set()
        for token in set(tokens):
            impacted.update(by_token.get(token, ()))
        if not impacted:
            continue
        ref = EvidenceRef(sentence.page_id, sentence.line_number)
        for item_i in impacted:
            if ref in exclusions[item_i]:
                continue
            score = bm25_score_int(queries[item_i], tokens, statistics)
            if score <= 0:
                continue
            rank: tuple[object, ...] = (
                -score,
                sentence.page_id,
                sentence.line_number,
                sentence.sentence_text,
            )
            insort(retained[item_i], (rank, score, sentence))
            if len(retained[item_i]) > required[item_i]:
                retained[item_i].pop()
    result: dict[int, tuple[tuple[int, WikiSentence], ...]] = {}
    for item_i, rows in enumerate(retained):
        if len(rows) != required[item_i]:
            raise FeverAcquisitionError(
                "full-wiki BM25 hard negatives are insufficient; replacement is forbidden"
            )
        result[item_i] = tuple((score, sentence) for _rank, score, sentence in rows)
    return result


def build_private_packs(
    *,
    selected: Sequence[SelectedCandidate],
    resolved: Mapping[EvidenceRef, WikiSentence],
    hard_negatives: Mapping[int, Sequence[tuple[int, WikiSentence]]],
    statistics: BM25Statistics,
    secret: bytes,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, object]]:
    action_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    gold_cardinality: Counter[int] = Counter()
    for ordinal, selected_row in enumerate(selected):
        candidate = selected_row.candidate
        gold_sentences = [resolved[ref] for ref in selected_row.selected_set]
        negatives = [sentence for _score, sentence in hard_negatives[ordinal]]
        all_sentences = [*gold_sentences, *negatives]
        if len(all_sentences) != DOCUMENT_COUNT or len(
            {(row.page_id, row.line_number) for row in all_sentences}
        ) != DOCUMENT_COUNT:
            raise FeverAcquisitionError("final item-local document identities drifted")
        ordered = sorted(
            all_sentences,
            key=lambda sentence: (
                _hmac_digest(
                    secret,
                    "doc_order",
                    candidate.identity_commitment_sha256,
                    sentence.page_id,
                    str(sentence.line_number),
                    _sha256_bytes(sentence.sentence_text.encode("utf-8")),
                ),
                sentence.page_id,
                sentence.line_number,
                sentence.sentence_text,
            ),
        )
        documents = [
            {
                "doc_id": doc_id,
                "page_id": sentence.page_id,
                "line_number": sentence.line_number,
                "sentence_text": sentence.sentence_text,
            }
            for doc_id, sentence in enumerate(ordered)
        ]
        scores = [
            bm25_score_int(tokenize(candidate.claim), tokenize(sentence.sentence_text), statistics)
            for sentence in ordered
        ]
        rank = sorted(range(DOCUMENT_COUNT), key=lambda doc_id: (-scores[doc_id], doc_id))
        action_body = {
            "schema": ACTION_ITEM_SCHEMA,
            "ordinal": ordinal,
            "item_id_hash": candidate.item_id_hash,
            "claim": candidate.claim,
            "documents": documents,
            "bm25_scores": scores,
            "bm25_rank": rank,
        }
        action_item = _with_hash(action_body, "action_item_sha256")
        gold_refs = set(selected_row.selected_set)
        gold_indices = sorted(
            doc_id
            for doc_id, sentence in enumerate(ordered)
            if EvidenceRef(sentence.page_id, sentence.line_number) in gold_refs
        )
        if len(gold_indices) != len(gold_refs):
            raise FeverAcquisitionError("gold injection drifted")
        label_body = {
            "schema": LABEL_ITEM_SCHEMA,
            "ordinal": ordinal,
            "item_id_hash": candidate.item_id_hash,
            "action_item_sha256": action_item["action_item_sha256"],
            "gold_indices": gold_indices,
            "source_label": candidate.source_label,
        }
        label_rows.append(_with_hash(label_body, "label_item_sha256"))
        action_rows.append(action_item)
        label_counts[candidate.source_label] += 1
        gold_cardinality[len(gold_indices)] += 1
    if label_counts != Counter({"SUPPORTS": LABEL_COUNT, "REFUTES": LABEL_COUNT}):
        raise FeverAcquisitionError("private label strata drifted")
    action_body = {
        "schema": ACTION_PACK_SCHEMA,
        "version": "v1",
        "item_count": COHORT_COUNT,
        "document_count_per_item": DOCUMENT_COUNT,
        "labels_included": False,
        "items": action_rows,
    }
    label_body = {
        "schema": LABEL_PACK_SCHEMA,
        "version": "v1",
        "item_count": COHORT_COUNT,
        "gold_contract": "one_preselected_gold_evidence_set_per_item",
        "items": label_rows,
    }
    return (
        _with_hash(action_body, "pack_sha256"),
        _with_hash(label_body, "pack_sha256"),
        {
            "selected_label_counts": dict(label_counts),
            "gold_cardinality_histogram": {
                str(key): gold_cardinality[key] for key in sorted(gold_cardinality)
            },
        },
    )


def _run_git(project: Path, arguments: Sequence[str]) -> bytes:
    if not arguments or arguments[0] not in {
        "rev-parse",
        "ls-tree",
        "check-ignore",
    }:
        raise FeverAcquisitionError("Git command is not allowlisted")
    completed = subprocess.run(
        ("git", "-C", str(project), *arguments),
        check=False,
        capture_output=True,
        timeout=30,
        shell=False,
    )
    if completed.returncode != 0:
        raise FeverAcquisitionError(f"Git {arguments[0]} verification failed")
    return completed.stdout


def _committed_bindings(
    project: Path, relative_paths: Sequence[str]
) -> tuple[str, tuple[dict[str, object], ...]]:
    root_raw = _run_git(project, ("rev-parse", "--show-toplevel"))
    try:
        repository = Path(root_raw.decode("utf-8").strip()).resolve(strict=True)
    except (UnicodeDecodeError, OSError) as exc:
        raise FeverAcquisitionError("Git repository root is invalid") from exc
    head = _run_git(repository, ("rev-parse", "HEAD")).decode("ascii").strip()
    if SHA1_RE.fullmatch(head) is None:
        raise FeverAcquisitionError("Git HEAD is malformed")
    prefix = project.relative_to(repository)
    repository_paths: list[str] = []
    by_repository_path: dict[str, str] = {}
    for relative in relative_paths:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise FeverAcquisitionError("freeze-listed path is unsafe")
        repository_relative = (prefix / candidate).as_posix()
        repository_paths.append(repository_relative)
        by_repository_path[repository_relative] = relative
    if len(repository_paths) != len(set(repository_paths)):
        raise FeverAcquisitionError("freeze-listed paths are not unique")
    raw_tree = _run_git(
        repository,
        ("ls-tree", "-r", head, "--", *repository_paths),
    )
    observed: dict[str, tuple[str, str]] = {}
    for line in raw_tree.decode("utf-8").splitlines():
        metadata, separator, repository_relative = line.partition("\t")
        fields = metadata.split(" ")
        if not separator or len(fields) != 3:
            raise FeverAcquisitionError("Git tree output is malformed")
        mode, kind, oid = fields
        if (
            repository_relative not in by_repository_path
            or mode not in {"100644", "100755"}
            or kind != "blob"
            or SHA1_RE.fullmatch(oid) is None
        ):
            raise FeverAcquisitionError("Git tree binding is invalid")
        observed[repository_relative] = (mode, oid)
    if set(observed) != set(repository_paths):
        raise FeverAcquisitionError("freeze-listed path is absent from HEAD")
    rows: list[dict[str, object]] = []
    for repository_relative in repository_paths:
        relative = by_repository_path[repository_relative]
        path = _safe_path(project, Path(relative), "freeze-listed file")
        if not path.is_file():
            raise FeverAcquisitionError("freeze-listed worktree file is unavailable")
        raw = path.read_bytes()
        mode, oid = observed[repository_relative]
        if _git_blob_sha1(raw) != oid:
            raise FeverAcquisitionError("freeze-listed worktree file drifted from HEAD")
        rows.append(
            {
                "relative_path": relative,
                "file_sha256": _sha256_bytes(raw),
                "git_blob_sha1": oid,
            }
        )
    return head, tuple(rows)


def _require_artifact_ignored(project: Path) -> None:
    completed = subprocess.run(
        (
            "git",
            "-C",
            str(project),
            "check-ignore",
            "-q",
            "--",
            ARTIFACT_ROOT_RELATIVE.as_posix(),
        ),
        check=False,
        capture_output=True,
        timeout=30,
        shell=False,
    )
    if completed.returncode != 0:
        raise FeverAcquisitionError("canonical FEVER artifact root is not Git ignored")


def _read_self_hashed_manifest(
    *,
    project: Path,
    relative: Path,
    schema: str,
    hash_field: str,
    expected_file_sha256: str | None = None,
    expected_semantic_sha256: str | None = None,
    require_committed: bool,
) -> tuple[dict[str, Any], bytes, dict[str, object] | None]:
    path = _safe_path(project, relative, "public manifest")
    if (
        not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != 0o644
        or not 1 <= path.stat().st_size <= MAX_PUBLIC_BYTES
    ):
        raise FeverAcquisitionError("public manifest mode or size drifted")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeverAcquisitionError("public manifest JSON drifted") from exc
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise FeverAcquisitionError("public manifest schema drifted")
    body = dict(payload)
    declared = _require_sha256(body.pop(hash_field, None), hash_field)
    if _semantic_hash(body) != declared:
        raise FeverAcquisitionError("public manifest self-hash drifted")
    file_sha = _sha256_bytes(raw)
    if expected_file_sha256 is not None and file_sha != expected_file_sha256:
        raise FeverAcquisitionError("public manifest file hash drifted")
    if expected_semantic_sha256 is not None and declared != expected_semantic_sha256:
        raise FeverAcquisitionError("public manifest semantic hash drifted")
    binding: dict[str, object] | None = None
    if require_committed:
        head, rows = _committed_bindings(project, (relative.as_posix(),))
        binding = {**rows[0], "verified_at_git_HEAD": head}
    return payload, raw, binding


def verify_public_source_bindings(project: Path) -> dict[str, object]:
    custody, _custody_raw, custody_binding = _read_self_hashed_manifest(
        project=project,
        relative=SOURCE_CUSTODY_RELATIVE,
        schema=SOURCE_CUSTODY_SCHEMA,
        hash_field="source_custody_sha256",
        expected_file_sha256=SOURCE_CUSTODY_FILE_SHA256,
        expected_semantic_sha256=SOURCE_CUSTODY_SHA256,
        require_committed=True,
    )
    access, _access_raw, access_binding = _read_self_hashed_manifest(
        project=project,
        relative=SOURCE_ACCESS_RELATIVE,
        schema=SOURCE_ACCESS_SCHEMA,
        hash_field="source_access_sha256",
        expected_file_sha256=SOURCE_ACCESS_FILE_SHA256,
        expected_semantic_sha256=SOURCE_ACCESS_SHA256,
        require_committed=True,
    )
    design, _design_raw, design_binding = _read_self_hashed_manifest(
        project=project,
        relative=DESIGN_RELATIVE,
        schema=DESIGN_SCHEMA,
        hash_field="design_sha256",
        expected_file_sha256=DESIGN_FILE_SHA256,
        expected_semantic_sha256=DESIGN_SHA256,
        require_committed=True,
    )
    if (
        access.get("custody_binding", {}).get("source_custody_sha256")
        != SOURCE_CUSTODY_SHA256
        or access.get("custody_binding", {}).get("file_sha256")
        != SOURCE_CUSTODY_FILE_SHA256
        or design.get("source_binding", {}).get("custody", {}).get(
            "source_custody_sha256"
        )
        != SOURCE_CUSTODY_SHA256
        or design.get("source_binding", {}).get("access", {}).get(
            "source_access_sha256"
        )
        != SOURCE_ACCESS_SHA256
    ):
        raise FeverAcquisitionError("public source dependency chain drifted")
    return {
        "source_custody": custody_binding,
        "source_access": access_binding,
        "design": design_binding,
    }


def verify_opaque_assets(project: Path) -> dict[str, dict[str, object]]:
    """Verify modes, sizes, and whole-file hashes without parsing content."""

    observed: dict[str, dict[str, object]] = {}
    for asset_id, expectation in ASSET_EXPECTATIONS.items():
        relative = expectation["relative_path"]
        if not isinstance(relative, Path):
            raise FeverAcquisitionError("asset expectation path drifted")
        path = _safe_path(project, relative, f"{asset_id} asset")
        if not path.is_file() or path.is_symlink():
            raise FeverAcquisitionError(f"{asset_id} asset is unavailable")
        info = path.stat()
        if (
            stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size != expectation["size_bytes"]
            or _sha256_file(path) != expectation["file_sha256"]
        ):
            raise FeverAcquisitionError(f"{asset_id} opaque binding drifted")
        observed[asset_id] = {
            "relative_path": relative.as_posix(),
            "mode": "0600",
            "size_bytes": info.st_size,
            "file_sha256": expectation["file_sha256"],
            "content_decoded_or_parsed": False,
        }
    return observed


def _preflight_absent(project: Path, relatives: Iterable[Path]) -> None:
    for relative in relatives:
        path = _safe_path(project, relative, "canonical phase output")
        if path.exists() or path.is_symlink():
            raise FeverAcquisitionError("canonical phase output exists; replay is forbidden")


def create_implementation_freeze(project: Path) -> dict[str, Any]:
    root = _canonical_project(project)
    _require_artifact_ignored(root)
    _preflight_absent(
        root,
        (
            IMPLEMENTATION_FREEZE_RELATIVE,
            SELECTION_CUSTODY_RELATIVE,
            SELECTION_FAILURE_RELATIVE,
            ACQUISITION_RECEIPT_RELATIVE,
            ACQUISITION_FAILURE_RELATIVE,
            SELECTION_SECRET_RELATIVE,
            SECRET_MARKER_RELATIVE,
            ACQUISITION_MARKER_RELATIVE,
            ACTION_PACK_RELATIVE,
            LABEL_PACK_RELATIVE,
            IDENTITY_LEDGER_RELATIVE,
            WORK_ROOT_RELATIVE,
        ),
    )
    public_bindings = verify_public_source_bindings(root)
    assets = verify_opaque_assets(root)
    head, rows = _committed_bindings(root, REQUIRED_FREEZE_PATHS)
    body = {
        "schema": IMPLEMENTATION_FREEZE_SCHEMA,
        "version": "v1",
        "status": "implementation_and_opaque_sources_frozen_before_secret_or_source_parse",
        "creation_HEAD": head,
        "git_HEAD": head,
        "fixed_P_program_sha256": (
            "0e9fea159e2dbcb302575f97954be8461c9921a91e11ef9b64a80ecab9640785"
        ),
        "bindings": list(rows),
        "binding_set_sha256": _semantic_hash(list(rows)),
        "public_protocol_bindings": public_bindings,
        "opaque_source_bindings": assets,
        "selection_secret_created": False,
        "source_member_listed_decoded_or_parsed": False,
    }
    payload = _with_hash(body, "implementation_freeze_sha256")
    _write_json(
        _safe_path(root, IMPLEMENTATION_FREEZE_RELATIVE, "implementation freeze"),
        payload,
        mode=0o644,
    )
    return payload


def load_committed_implementation_freeze(project: Path) -> tuple[dict[str, Any], dict[str, object]]:
    payload, raw, binding = _read_self_hashed_manifest(
        project=project,
        relative=IMPLEMENTATION_FREEZE_RELATIVE,
        schema=IMPLEMENTATION_FREEZE_SCHEMA,
        hash_field="implementation_freeze_sha256",
        require_committed=True,
    )
    if binding is None:
        raise FeverAcquisitionError("implementation freeze is not committed")
    current_head, current_rows = _committed_bindings(project, REQUIRED_FREEZE_PATHS)
    if (
        payload.get("bindings") != list(current_rows)
        or payload.get("binding_set_sha256") != _semantic_hash(list(current_rows))
        or payload.get("status")
        != "implementation_and_opaque_sources_frozen_before_secret_or_source_parse"
    ):
        raise FeverAcquisitionError("implementation freeze binding set drifted")
    return payload, {
        **binding,
        "file_sha256": _sha256_bytes(raw),
        "current_git_HEAD": current_head,
    }


def _phase_marker(
    *, project: Path, relative: Path, phase: str, bindings: Mapping[str, object]
) -> tuple[dict[str, Any], str]:
    body = {
        "schema": f"{VERSION}_one_shot_marker",
        "version": VERSION,
        "phase": phase,
        "replay_allowed": False,
        "bindings": dict(bindings),
    }
    payload = _with_hash(body, "marker_sha256")
    path = _safe_path(project, relative, f"{phase} marker")
    file_sha = _write_json(path, payload, mode=0o600)
    return payload, file_sha


def _write_terminal_failure(
    *, project: Path, relative: Path, phase: str, marker_sha256: str, failure_class: str
) -> None:
    allowed = {
        "source_schema_invalid",
        "source_capacity_insufficient",
        "chosen_evidence_unresolved",
        "hard_negative_capacity_insufficient",
        "persistence_invalid",
        "unexpected_infrastructure_invalid",
    }
    if failure_class not in allowed:
        failure_class = "unexpected_infrastructure_invalid"
    body = {
        "schema": FAILURE_SCHEMA,
        "version": "v1",
        "phase": phase,
        "status": "terminal_infrastructure_invalid_no_replay",
        "failure_class": failure_class,
        "marker_sha256": _require_sha256(marker_sha256, "failure marker hash"),
        "row_ids_claims_labels_evidence_or_sentences_persisted_publicly": False,
    }
    _write_json(
        _safe_path(project, relative, f"{phase} failure receipt"),
        _with_hash(body, "failure_sha256"),
        mode=0o644,
    )


def _classify_failure(exc: BaseException) -> str:
    message = str(exc).casefold()
    if "capacity insufficient for exact" in message:
        return "source_capacity_insufficient"
    if "unresolved" in message:
        return "chosen_evidence_unresolved"
    if "hard negatives are insufficient" in message:
        return "hard_negative_capacity_insufficient"
    if "schema" in message or "jsonl" in message or "zip" in message:
        return "source_schema_invalid"
    if "persist" in message or "output" in message:
        return "persistence_invalid"
    return "unexpected_infrastructure_invalid"


def create_selection_custody(project: Path) -> dict[str, Any]:
    root = _canonical_project(project)
    _require_artifact_ignored(root)
    freeze, freeze_binding = load_committed_implementation_freeze(root)
    _preflight_absent(
        root,
        (
            SELECTION_CUSTODY_RELATIVE,
            SELECTION_FAILURE_RELATIVE,
            SELECTION_SECRET_RELATIVE,
            SECRET_MARKER_RELATIVE,
            ACQUISITION_MARKER_RELATIVE,
            ACTION_PACK_RELATIVE,
            LABEL_PACK_RELATIVE,
            WORK_ROOT_RELATIVE,
            ACQUISITION_RECEIPT_RELATIVE,
            ACQUISITION_FAILURE_RELATIVE,
        ),
    )
    marker, marker_file_sha = _phase_marker(
        project=root,
        relative=SECRET_MARKER_RELATIVE,
        phase="selection_secret_creation",
        bindings={
            "implementation_freeze_sha256": freeze[
                "implementation_freeze_sha256"
            ],
            "implementation_freeze_file_sha256": freeze_binding["file_sha256"],
            "implementation_freeze_git_blob_sha1": freeze_binding["git_blob_sha1"],
        },
    )
    try:
        secret = os.urandom(32)
        if len(secret) != 32:
            raise FeverAcquisitionError("os.urandom did not return exactly 32 bytes")
        secret_path = _safe_path(root, SELECTION_SECRET_RELATIVE, "selection secret")
        _write_exclusive(secret_path, secret, mode=0o600)
        commitment = _sha256_bytes(secret)
        body = {
            "schema": SELECTION_CUSTODY_SCHEMA,
            "version": "v1",
            "status": "one_selection_secret_created_and_committed",
            "implementation_freeze_relative_path": (
                IMPLEMENTATION_FREEZE_RELATIVE.as_posix()
            ),
            "implementation_freeze_sha256": freeze[
                "implementation_freeze_sha256"
            ],
            "implementation_freeze_file_sha256": freeze_binding["file_sha256"],
            "implementation_freeze_git_blob_sha1": freeze_binding["git_blob_sha1"],
            "implementation_freeze_verified_at_git_HEAD": freeze_binding[
                "current_git_HEAD"
            ],
            "secret_marker_relative_path": SECRET_MARKER_RELATIVE.as_posix(),
            "secret_marker_sha256": marker["marker_sha256"],
            "secret_marker_file_sha256": marker_file_sha,
            "selection_secret_relative_path": SELECTION_SECRET_RELATIVE.as_posix(),
            "selection_secret_file_mode": "0600",
            "selection_secret_byte_count": 32,
            "selection_secret_commitment_sha256": commitment,
            "os_urandom_call_count": 1,
            "source_bytes_opened_listed_decoded_or_parsed": 0,
            "secret_replacement_regeneration_or_replay_authorized": False,
        }
        payload = _with_hash(body, "selection_custody_sha256")
        _write_json(
            _safe_path(root, SELECTION_CUSTODY_RELATIVE, "selection custody"),
            payload,
            mode=0o644,
        )
        return payload
    except BaseException as exc:
        _write_terminal_failure(
            project=root,
            relative=SELECTION_FAILURE_RELATIVE,
            phase="selection_secret_creation",
            marker_sha256=marker["marker_sha256"],
            failure_class=_classify_failure(exc),
        )
        raise


def load_committed_selection_custody(
    project: Path,
) -> tuple[dict[str, Any], dict[str, object], bytes]:
    freeze, freeze_binding = load_committed_implementation_freeze(project)
    custody, custody_raw, custody_binding = _read_self_hashed_manifest(
        project=project,
        relative=SELECTION_CUSTODY_RELATIVE,
        schema=SELECTION_CUSTODY_SCHEMA,
        hash_field="selection_custody_sha256",
        require_committed=True,
    )
    if custody_binding is None:
        raise FeverAcquisitionError("selection custody is not committed")
    if (
        custody.get("status") != "one_selection_secret_created_and_committed"
        or custody.get("implementation_freeze_relative_path")
        != IMPLEMENTATION_FREEZE_RELATIVE.as_posix()
        or custody.get("implementation_freeze_sha256")
        != freeze["implementation_freeze_sha256"]
        or custody.get("implementation_freeze_file_sha256")
        != freeze_binding["file_sha256"]
        or custody.get("implementation_freeze_git_blob_sha1")
        != freeze_binding["git_blob_sha1"]
        or custody.get("selection_secret_relative_path")
        != SELECTION_SECRET_RELATIVE.as_posix()
        or custody.get("selection_secret_file_mode") != "0600"
        or custody.get("selection_secret_byte_count") != 32
        or custody.get("os_urandom_call_count") != 1
        or custody.get("source_bytes_opened_listed_decoded_or_parsed") != 0
        or custody.get("secret_replacement_regeneration_or_replay_authorized")
        is not False
    ):
        raise FeverAcquisitionError("selection custody chain drifted")
    secret_path = _safe_path(project, SELECTION_SECRET_RELATIVE, "selection secret")
    if (
        not secret_path.is_file()
        or secret_path.is_symlink()
        or stat.S_IMODE(secret_path.stat().st_mode) != 0o600
        or secret_path.stat().st_size != 32
    ):
        raise FeverAcquisitionError("selection secret mode or size drifted")
    secret = secret_path.read_bytes()
    if _sha256_bytes(secret) != custody.get("selection_secret_commitment_sha256"):
        raise FeverAcquisitionError("selection secret commitment drifted")
    return custody, {
        **custody_binding,
        "file_sha256": _sha256_bytes(custody_raw),
    }, secret


def _acquisition_preflight(project: Path) -> None:
    _preflight_absent(
        project,
        (
            ACQUISITION_MARKER_RELATIVE,
            ACTION_PACK_RELATIVE,
            LABEL_PACK_RELATIVE,
            IDENTITY_LEDGER_RELATIVE,
            WORK_ROOT_RELATIVE,
            ACQUISITION_RECEIPT_RELATIVE,
            ACQUISITION_FAILURE_RELATIVE,
        ),
    )


def acquire(project: Path) -> dict[str, Any]:
    root = _canonical_project(project)
    _require_artifact_ignored(root)
    custody, custody_binding, secret = load_committed_selection_custody(root)
    _acquisition_preflight(root)
    assets = verify_opaque_assets(root)
    marker, marker_file_sha = _phase_marker(
        project=root,
        relative=ACQUISITION_MARKER_RELATIVE,
        phase="formal_acquisition",
        bindings={
            "selection_custody_sha256": custody["selection_custody_sha256"],
            "selection_custody_file_sha256": custody_binding["file_sha256"],
            "selection_custody_git_blob_sha1": custody_binding["git_blob_sha1"],
            "selection_secret_commitment_sha256": custody[
                "selection_secret_commitment_sha256"
            ],
        },
    )
    try:
        paper_path = _safe_path(root, PAPER_TEST_RELATIVE, "paper_test source")
        wiki_path = _safe_path(root, WIKI_ZIP_RELATIVE, "wiki source")
        candidates, source_row_count = load_paper_candidates(paper_path)
        selected, selection_stats = select_candidates(candidates, secret)
        statistics, resolved, wiki_stats = scan_wiki_pass1(
            wiki_zip=wiki_path,
            selected=selected,
            identity_ledger_path=_safe_path(
                root, IDENTITY_LEDGER_RELATIVE, "wiki identity ledger"
            ),
        )
        hard_negatives = mine_hard_negatives_pass2(
            wiki_zip=wiki_path,
            selected=selected,
            statistics=statistics,
        )
        action_pack, label_pack, pack_stats = build_private_packs(
            selected=selected,
            resolved=resolved,
            hard_negatives=hard_negatives,
            statistics=statistics,
            secret=secret,
        )
        action_file_sha = _write_private_json(
            _safe_path(root, ACTION_PACK_RELATIVE, "action pack"),
            action_pack,
        )
        label_file_sha = _write_private_json(
            _safe_path(root, LABEL_PACK_RELATIVE, "late label pack"),
            label_pack,
        )
        action_item_set_sha = _semantic_hash(
            [row["action_item_sha256"] for row in action_pack["items"]]
        )
        label_item_set_sha = _semantic_hash(
            [row["label_item_sha256"] for row in label_pack["items"]]
        )
        body = {
            "schema": ACQUISITION_SCHEMA,
            "version": "v1",
            "status": ACQUISITION_STATUS,
            "implementation_freeze_sha256": custody[
                "implementation_freeze_sha256"
            ],
            "selection_custody_relative_path": SELECTION_CUSTODY_RELATIVE.as_posix(),
            "selection_custody_sha256": custody["selection_custody_sha256"],
            "selection_custody_file_sha256": custody_binding["file_sha256"],
            "selection_custody_git_blob_sha1": custody_binding["git_blob_sha1"],
            "selection_custody_verified_at_git_HEAD": custody_binding[
                "verified_at_git_HEAD"
            ],
            "selection_secret_commitment_sha256": custody[
                "selection_secret_commitment_sha256"
            ],
            "acquisition_marker_relative_path": ACQUISITION_MARKER_RELATIVE.as_posix(),
            "acquisition_marker_sha256": marker["marker_sha256"],
            "acquisition_marker_file_sha256": marker_file_sha,
            "opaque_source_bindings": assets,
            "source_aggregate": {
                "paper_test_row_count": source_row_count,
                "syntactically_eligible_row_count": len(candidates),
                **selection_stats,
            },
            "wiki_aggregate": {
                **wiki_stats,
                "average_document_length": statistics.average_document_length,
                "bm25_document_frequency_set_sha256": _semantic_hash(
                    dict(statistics.document_frequency)
                ),
                "full_wiki_pass_count": 2,
                "frozen_full_wiki_BM25_two_pass_stream": True,
                "identity_ledger_relative_path": IDENTITY_LEDGER_RELATIVE.as_posix(),
                "identity_ledger_file_sha256": _sha256_file(
                    _safe_path(root, IDENTITY_LEDGER_RELATIVE, "wiki identity ledger")
                ),
                "identity_ledger_file_mode": "0600",
            },
            "private_packs": {
                "action_pack_relative_path": ACTION_PACK_RELATIVE.as_posix(),
                "action_pack_file_sha256": action_file_sha,
                "action_pack_sha256": action_pack["pack_sha256"],
                "label_pack_relative_path": LABEL_PACK_RELATIVE.as_posix(),
                "label_pack_file_sha256": label_file_sha,
                "label_pack_sha256": label_pack["pack_sha256"],
                "action_item_set_sha256": action_item_set_sha,
                "label_item_set_sha256": label_item_set_sha,
                **pack_stats,
                "file_modes": {"action_pack": "0600", "label_pack": "0600"},
            },
            "counts": {
                "document_count_per_item": DOCUMENT_COUNT,
                "item_count": COHORT_COUNT,
            },
            "commitments": {
                "action_item_commitment_set_sha256": action_item_set_sha,
                "action_pack_file_sha256": action_file_sha,
                "label_item_commitment_set_sha256": label_item_set_sha,
                "label_pack_file_sha256": label_file_sha,
            },
            "source_labels_and_evidence_read_only_by_acquisition": True,
            "label_pack_opened_by_action_runner_before_action_barrier": False,
            "safety": {
                "source_labels_and_evidence_read_only_by_acquisition": True,
                "row_selection_used_wiki_or_BM25_information": False,
                "row_or_evidence_set_replacement_after_wiki_resolution": False,
                "label_pack_opened_by_action_runner_before_action_barrier": False,
                "network_calls": 0,
                "online_evaluator_calls": 0,
                "model_calls": 0,
                "performance_scores_computed": 0,
                "replay_resample_retry_count": 0,
                "public_row_ids_claims_labels_evidence_or_sentences": 0,
            },
        }
        receipt = _with_hash(body, "acquisition_sha256")
        _write_json(
            _safe_path(root, ACQUISITION_RECEIPT_RELATIVE, "acquisition receipt"),
            receipt,
            mode=0o644,
        )
        return receipt
    except BaseException as exc:
        _write_terminal_failure(
            project=root,
            relative=ACQUISITION_FAILURE_RELATIVE,
            phase="formal_acquisition",
            marker_sha256=marker["marker_sha256"],
            failure_class=_classify_failure(exc),
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze", "secret", "acquire"):
        phase = subparsers.add_parser(command)
        phase.add_argument("--project", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "freeze":
        receipt = create_implementation_freeze(arguments.project)
        hash_field = "implementation_freeze_sha256"
    elif arguments.command == "secret":
        receipt = create_selection_custody(arguments.project)
        hash_field = "selection_custody_sha256"
    else:
        receipt = acquire(arguments.project)
        hash_field = "acquisition_sha256"
    print(
        json.dumps(
            {"status": receipt["status"], hash_field: receipt[hash_field]},
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ACQUISITION_RECEIPT_RELATIVE",
    "ACTION_PACK_RELATIVE",
    "ASSET_EXPECTATIONS",
    "BM25Statistics",
    "COHORT_COUNT",
    "DESIGN_RELATIVE",
    "DOCUMENT_COUNT",
    "EvidenceRef",
    "FeverAcquisitionError",
    "IMPLEMENTATION_FREEZE_RELATIVE",
    "LABEL_PACK_RELATIVE",
    "PaperCandidate",
    "REQUIRED_FREEZE_PATHS",
    "SELECTION_CUSTODY_RELATIVE",
    "SELECTION_SECRET_RELATIVE",
    "SelectedCandidate",
    "VERSION",
    "WikiSentence",
    "acquire",
    "bm25_score_int",
    "build_private_packs",
    "create_implementation_freeze",
    "create_selection_custody",
    "iter_wiki_sentences",
    "load_paper_candidates",
    "main",
    "mine_hard_negatives_pass2",
    "scan_wiki_pass1",
    "select_candidates",
    "tokenize",
    "verify_opaque_assets",
]


if __name__ == "__main__":
    raise SystemExit(main())
