"""One-shot private acquisition for the frozen ERASER EI experiment.

The aggregate source qualifier intentionally exposes no row API.  This module
therefore reproduces its narrow archive boundary for the *subsequent* private
acquisition stage: it opens exactly one ``train.jsonl``, one ``val.jsonl``, and
only documents referenced by those two members.  A TEST member or a TEST-only
document is routed past by header and is never extracted or read.

The first call creates a 32-byte selection secret and a burn marker before any
authorized archive member is opened.  It commits all four opaque blocks at
once, then persists label-free views for A_form and F_search only.  Later VAL
views and every label pack require independent, self-hashed capabilities.
There is deliberately no retry, replacement, resampling, or TEST API.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
from typing import Any
import unicodedata


VERSION = "eraser_evidence_inference_direct_acquisition_v1"

BLOCK_ORDER = ("A_form", "F_search", "A_hold", "M_search")
FAMILY_ORDER = (
    "SIGNIFICANTLY_DECREASED",
    "NO_SIGNIFICANT_DIFFERENCE",
    "SIGNIFICANTLY_INCREASED",
)
BLOCK_SPLITS = {
    "A_form": "train",
    "F_search": "train",
    "A_hold": "val",
    "M_search": "val",
}
BLOCK_FAMILY_QUOTAS = {
    "A_form": {family: 16 for family in FAMILY_ORDER},
    "F_search": {family: 12 for family in FAMILY_ORDER},
    "A_hold": {family: 10 for family in FAMILY_ORDER},
    "M_search": {family: 10 for family in FAMILY_ORDER},
}
BLOCK_COUNTS = {
    block: sum(quotas.values()) for block, quotas in BLOCK_FAMILY_QUOTAS.items()
}
TOTAL_ITEM_COUNT = sum(BLOCK_COUNTS.values())

OFFICIAL_CLASSIFICATION_TO_FAMILY = {
    "significantly decreased": "SIGNIFICANTLY_DECREASED",
    "no significant difference": "NO_SIGNIFICANT_DIFFERENCE",
    "significantly increased": "SIGNIFICANTLY_INCREASED",
}

MARKER_RELATIVE = Path("acquisition.marker.private.json")
ASSIGNMENT_RELATIVE = Path("assignment.private.json")
PUBLIC_RECEIPT_RELATIVE = Path("acquisition.receipt.json")
VIEW_DIRECTORY = Path("views")
LABEL_DIRECTORY = Path("labels")
STAGE_MARKER_DIRECTORY = Path("stage_markers")
AUTHORIZATION_DIRECTORY = Path("authorizations")

QUALIFICATION_SCHEMA = "eraser_evidence_inference_source_qualification_v1"
QUALIFICATION_SELF_HASH_FIELD = "qualification_sha256"
DESIGN_SELF_HASH_FIELD = "design_sha256"
FORMAL_DESIGN_SHA256 = (
    "49920ccaa8e3f52eeb95fa86d64ecab577971fb8d0cc50d2bd93e0d5baaa2196"
)
IMPLEMENTATION_FREEZE_SCHEMA = (
    "eraser_evidence_inference_full_implementation_freeze_v1"
)
IMPLEMENTATION_FREEZE_SELF_HASH_FIELD = "implementation_freeze_sha256"
REQUIRED_IMPLEMENTATION_ROLE_REGISTRY = (
    "source_qualifier",
    "direct_acquisition",
    "local_runtime",
    "three_arm_scheduler",
    "r7_operator",
    "exact_feature_bridge",
    "e3_runner",
    "formal_controller",
    "test_source_qualifier",
    "test_direct_acquisition",
    "test_local_runtime",
    "test_three_arm_scheduler",
    "test_r7_operator",
    "test_exact_feature_bridge",
    "test_e3_runner",
    "hipporag_freeze_manifest",
)

MAX_AUTHORIZED_MEMBER_BYTES = 64 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
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
_SIDECAR_REQUIRED_FIELDS = frozenset(
    {"PromptID", "PMCID", "Outcome", "Intervention", "Comparator"}
)


class EraserEvidenceInferenceDirectAcquisitionError(RuntimeError):
    """The source epoch, opaque assignment, capability, or state drifted."""


@dataclass(frozen=True)
class _EvidenceSpan:
    docid: str
    start_token: int
    end_token: int
    start_sentence: int
    end_sentence: int
    text_tokens: tuple[str, ...] | None


@dataclass(frozen=True)
class _Annotation:
    annotation_id: str
    query: str
    normalized_query_sha256: str
    family: str
    article_docid: str
    evidence_groups: tuple[tuple[_EvidenceSpan, ...], ...]
    source_split: str


@dataclass(frozen=True)
class _Document:
    sentences: tuple[tuple[str, ...], ...]
    flattened_tokens: tuple[str, ...]
    sentence_token_boundaries: tuple[int, ...]
    member_content_sha256: str


@dataclass(frozen=True)
class _PrivateItem:
    annotation: _Annotation
    document: _Document
    facets: tuple[str, str, str]
    flattened_gold_sentence_ordinals: tuple[int, ...]
    validated_groups: tuple[tuple[int, ...], ...]


def _assert_json_types(value: object) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _assert_json_types(child)
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "canonical JSON key is not text"
            )
        for child in value.values():
            _assert_json_types(child)
        return
    raise EraserEvidenceInferenceDirectAcquisitionError(
        "value contains a non-exact JSON type"
    )


def canonical_bytes(value: object) -> bytes:
    _assert_json_types(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "value is not canonical JSON"
        ) from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "self-hash field already exists"
        )
    return {**dict(body), field: stable_hash(dict(body))}


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            f"{field} is not a lowercase SHA256"
        )
    return value


def _verify_self_hash(
    payload: Mapping[str, Any], *, schema: str | None, field: str
) -> str:
    declared = _require_sha256(payload.get(field), field)
    body = dict(payload)
    body.pop(field)
    if schema is not None and payload.get("schema") != schema:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            f"{field} schema drifted"
        )
    if not hmac.compare_digest(stable_hash(body), declared):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            f"{field} self-hash drifted"
        )
    return declared


def _strict_json(raw: bytes, field: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            f"{field} is not strict UTF-8"
        ) from exc

    def object_pairs(rows: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise EraserEvidenceInferenceDirectAcquisitionError(
                    f"{field} contains duplicate JSON keys"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            f"{field} contains nonfinite constant {value}"
        )

    try:
        payload = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except EraserEvidenceInferenceDirectAcquisitionError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            f"{field} is not strict JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            f"{field} is not a JSON object"
        )
    _assert_json_types(payload)
    return payload


def _regular_nonsymlink(path: Path, field: str) -> None:
    if path.is_symlink():
        raise EraserEvidenceInferenceDirectAcquisitionError(
            f"{field} may not be a symlink"
        )
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            f"{field} is unavailable"
        ) from exc
    if not stat.S_ISREG(mode):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            f"{field} is not a regular file"
        )


def _read_json_path(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    _regular_nonsymlink(path, field)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            f"{field} cannot be read"
        ) from exc
    return _strict_json(raw, field), raw


def _sha256_file(path: Path, field: str) -> tuple[str, int]:
    _regular_nonsymlink(path, field)
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
                size += len(block)
    except OSError as exc:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            f"{field} cannot be hashed"
        ) from exc
    return digest.hexdigest(), size


def _safe_tar_parts(name: str) -> tuple[str, ...]:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "archive contains an unsafe member header"
        )
    path = PurePosixPath(name)
    parts = tuple(part for part in path.parts if part not in {"", "."})
    if path.is_absolute() or not parts or any(part == ".." for part in parts):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "archive contains an unsafe member header"
        )
    return parts


def _read_tar_member(bundle: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    if member.size < 0 or member.size > MAX_AUTHORIZED_MEMBER_BYTES:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "authorized archive member size is invalid"
        )
    handle = bundle.extractfile(member)
    if handle is None:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "authorized archive member cannot be opened"
        )
    raw = handle.read(member.size + 1)
    if len(raw) != member.size:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "authorized archive member size drifted"
        )
    return raw


def _read_authorized_split_members(
    archive_path: Path,
) -> tuple[dict[str, bytes], tuple[str, ...], dict[str, int]]:
    split_raw: dict[str, bytes] = {}
    split_root: tuple[str, ...] | None = None
    header_counts = Counter({"regular": 0, "directory": 0, "other": 0})
    opened = Counter({"train": 0, "val": 0, "test": 0})
    try:
        with tarfile.open(archive_path, mode="r:gz", errorlevel=2) as bundle:
            for member in bundle:
                parts = _safe_tar_parts(member.name)
                if member.isdir():
                    header_counts["directory"] += 1
                    continue
                if not member.isfile():
                    header_counts["other"] += 1
                    continue
                header_counts["regular"] += 1
                split = _SPLIT_BASENAMES.get(parts[-1])
                if split is None:
                    continue
                if split in split_raw:
                    raise EraserEvidenceInferenceDirectAcquisitionError(
                        "archive duplicates an authorized split member"
                    )
                root = parts[:-1]
                if split_root is None:
                    split_root = root
                elif root != split_root:
                    raise EraserEvidenceInferenceDirectAcquisitionError(
                        "authorized splits do not share one dataset root"
                    )
                split_raw[split] = _read_tar_member(bundle, member)
                opened[split] += 1
    except EraserEvidenceInferenceDirectAcquisitionError:
        raise
    except (OSError, tarfile.TarError, EOFError) as exc:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "official archive split scan failed"
        ) from exc
    if set(split_raw) != {"train", "val"} or split_root is None:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "archive lacks exactly one authorized train and validation split"
        )
    return split_raw, split_root, {
        **dict(header_counts),
        "authorized_split_member_open_count": opened["train"] + opened["val"],
        "test_member_content_open_count": opened["test"],
    }


def _normalized_query_sha256(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "annotation query type or value drifted"
        )
    normalized = " ".join(unicodedata.normalize("NFKC", value).split()).casefold()
    if not normalized:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "annotation normalized query is empty"
        )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _validate_docid(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not value.strip()
        or "\x00" in value
        or "\\" in value
        or PurePosixPath(value).name != value
        or value in {".", ".."}
    ):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "document identity is unsafe or empty"
        )
    return value


def _evidence_text_tokens(value: object) -> tuple[str, ...] | None:
    if isinstance(value, str):
        if not value.strip() or "\x00" in value:
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "evidence text representation drifted"
            )
        result = tuple(token for token in value.split() if token)
        return result if result else None
    if isinstance(value, list) and value:
        if all(isinstance(token, str) for token in value):
            if any(not token or "\x00" in token for token in value):
                raise EraserEvidenceInferenceDirectAcquisitionError(
                    "evidence text representation drifted"
                )
            return tuple(value)
        if all(type(token) is int for token in value):
            return None
    raise EraserEvidenceInferenceDirectAcquisitionError(
        "evidence text representation drifted"
    )


def _parse_split_jsonl(raw: bytes, *, split: str) -> tuple[_Annotation, ...]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "authorized split is not strict UTF-8"
        ) from exc
    lines = text.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "authorized split JSONL line structure drifted"
        )
    annotations: list[_Annotation] = []
    identities: set[str] = set()
    for line in lines:
        row = _strict_json(line.encode("utf-8"), "authorized split row")
        keys = set(row)
        if (
            not _ANNOTATION_REQUIRED_FIELDS <= keys
            or not keys <= _ANNOTATION_ALLOWED_FIELDS
        ):
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "annotation field schema drifted"
            )
        annotation_id = row.get("annotation_id")
        if (
            not isinstance(annotation_id, str)
            or not annotation_id.strip()
            or "\x00" in annotation_id
            or annotation_id in identities
        ):
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "annotation identity is empty or duplicated within split"
            )
        identities.add(annotation_id)
        query = row.get("query")
        query_digest = _normalized_query_sha256(query)
        assert isinstance(query, str)
        query_type = row.get("query_type")
        if query_type is not None and not isinstance(query_type, str):
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "annotation query_type drifted"
            )
        official_classification = row.get("classification")
        if official_classification not in OFFICIAL_CLASSIFICATION_TO_FAMILY:
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "annotation official lowercase classification drifted"
            )
        family = OFFICIAL_CLASSIFICATION_TO_FAMILY[str(official_classification)]

        declared_raw = row.get("docids")
        if declared_raw is None:
            declared_docids: set[str] = set()
        elif isinstance(declared_raw, list) and all(
            isinstance(value, str) for value in declared_raw
        ):
            declared_docids = {_validate_docid(value) for value in declared_raw}
        else:
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "annotation docids field drifted"
            )
        raw_groups = row.get("evidences")
        if not isinstance(raw_groups, list):
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "annotation evidence groups drifted"
            )
        evidence_docids: set[str] = set()
        groups: list[tuple[_EvidenceSpan, ...]] = []
        for raw_group in raw_groups:
            if not isinstance(raw_group, list):
                raise EraserEvidenceInferenceDirectAcquisitionError(
                    "alternative evidence group drifted"
                )
            group: list[_EvidenceSpan] = []
            for raw_evidence in raw_group:
                if not isinstance(raw_evidence, Mapping):
                    raise EraserEvidenceInferenceDirectAcquisitionError(
                        "evidence entry is not an object"
                    )
                evidence_keys = set(raw_evidence)
                if (
                    not _EVIDENCE_REQUIRED_FIELDS <= evidence_keys
                    or not evidence_keys <= _EVIDENCE_ALLOWED_FIELDS
                ):
                    raise EraserEvidenceInferenceDirectAcquisitionError(
                        "evidence field schema drifted"
                    )
                docid = _validate_docid(raw_evidence.get("docid"))
                offsets: list[int] = []
                for field in (
                    "start_token",
                    "end_token",
                    "start_sentence",
                    "end_sentence",
                ):
                    value = raw_evidence.get(field, -1)
                    if type(value) is not int:
                        raise EraserEvidenceInferenceDirectAcquisitionError(
                            "evidence span coordinate type drifted"
                        )
                    offsets.append(value)
                group.append(
                    _EvidenceSpan(
                        docid,
                        offsets[0],
                        offsets[1],
                        offsets[2],
                        offsets[3],
                        _evidence_text_tokens(raw_evidence.get("text")),
                    )
                )
                evidence_docids.add(docid)
            groups.append(tuple(group))
        all_docids = declared_docids | evidence_docids
        if len(all_docids) != 1:
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "authorized PromptID lacks one unambiguous article binding"
            )
        annotations.append(
            _Annotation(
                annotation_id=annotation_id,
                query=query,
                normalized_query_sha256=query_digest,
                family=family,
                article_docid=next(iter(all_docids)),
                evidence_groups=tuple(groups),
                source_split=split,
            )
        )
    return tuple(annotations)


def _decode_document(raw: bytes) -> _Document:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "referenced document is not strict UTF-8"
        ) from exc
    lines = [line.strip() for line in text.splitlines()]
    sentences = tuple(
        tuple(token for token in line.split(" ") if token)
        for line in lines
        if line
    )
    if not sentences or any(not sentence for sentence in sentences):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "referenced document sentence/token structure drifted"
        )
    flattened = tuple(token for sentence in sentences for token in sentence)
    boundaries = [0]
    for sentence in sentences:
        boundaries.append(boundaries[-1] + len(sentence))
    return _Document(
        sentences=sentences,
        flattened_tokens=flattened,
        sentence_token_boundaries=tuple(boundaries),
        member_content_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _read_referenced_documents(
    archive_path: Path,
    *,
    split_root: tuple[str, ...],
    referenced_docids: frozenset[str],
) -> tuple[dict[str, _Document], int]:
    expected = {split_root + ("docs", docid): docid for docid in referenced_docids}
    documents: dict[str, _Document] = {}
    try:
        with tarfile.open(archive_path, mode="r:gz", errorlevel=2) as bundle:
            for member in bundle:
                parts = _safe_tar_parts(member.name)
                docid = expected.get(parts)
                if docid is None:
                    continue
                if not member.isfile() or docid in documents:
                    raise EraserEvidenceInferenceDirectAcquisitionError(
                        "referenced document is duplicate or nonregular"
                    )
                documents[docid] = _decode_document(_read_tar_member(bundle, member))
    except EraserEvidenceInferenceDirectAcquisitionError:
        raise
    except (OSError, tarfile.TarError, EOFError) as exc:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "official archive document scan failed"
        ) from exc
    if set(documents) != set(referenced_docids):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "one or more referenced documents are absent"
        )
    return documents, len(documents)


def _canonical_pmcid(value: object) -> str:
    if not isinstance(value, str):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "sidecar PMCID type drifted"
        )
    stripped = value.strip()
    if stripped.startswith("PMC"):
        stripped = stripped[3:]
    if not stripped or not stripped.isascii() or not stripped.isdigit():
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "sidecar PMCID binding drifted"
        )
    return f"PMC{int(stripped)}"


def _read_bound_facets(
    sidecar_path: Path,
    annotations: Sequence[_Annotation],
) -> dict[str, tuple[str, str, str]]:
    article_by_prompt = {
        annotation.annotation_id: annotation.article_docid
        for annotation in annotations
    }
    if len(article_by_prompt) != len(annotations):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "PromptID overlaps train and validation"
        )
    facets_by_prompt: dict[str, tuple[str, str, str]] = {}
    try:
        with sidecar_path.open(
            "r", encoding="utf-8-sig", errors="strict", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames
            if (
                headers is None
                or len(headers) != len(set(headers))
                or not _SIDECAR_REQUIRED_FIELDS <= set(headers)
            ):
                raise EraserEvidenceInferenceDirectAcquisitionError(
                    "prompt sidecar CSV header drifted"
                )
            for row in reader:
                prompt_id = row.get("PromptID")
                if prompt_id not in article_by_prompt:
                    # The row may belong to another official split.  Do not
                    # inspect or retain its article, I/C/O, or any other value.
                    continue
                if prompt_id in facets_by_prompt or None in row:
                    raise EraserEvidenceInferenceDirectAcquisitionError(
                        "referenced sidecar PromptID is duplicate or ambiguous"
                    )
                if _canonical_pmcid(row.get("PMCID")) != _canonical_pmcid(
                    article_by_prompt[prompt_id]
                ):
                    raise EraserEvidenceInferenceDirectAcquisitionError(
                        "referenced PromptID article binding drifted"
                    )
                values: list[str] = []
                for field in ("Intervention", "Comparator", "Outcome"):
                    value = row.get(field)
                    if (
                        not isinstance(value, str)
                        or not value.strip()
                        or "\x00" in value
                    ):
                        raise EraserEvidenceInferenceDirectAcquisitionError(
                            "referenced PromptID has incomplete exact I/C/O"
                        )
                    values.append(value)
                facets_by_prompt[prompt_id] = (values[0], values[1], values[2])
    except EraserEvidenceInferenceDirectAcquisitionError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "prompt sidecar streaming failed"
        ) from exc
    if set(facets_by_prompt) != set(article_by_prompt):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "one or more authorized PromptIDs are absent from sidecar"
        )
    return facets_by_prompt


def _validated_private_item(
    annotation: _Annotation,
    document: _Document,
    facets: tuple[str, str, str],
) -> _PrivateItem | None:
    if len(document.sentences) < 5 or not annotation.evidence_groups:
        return None
    group_signatures: set[tuple[tuple[int, int, int, int], ...]] = set()
    validated_groups: list[tuple[int, ...]] = []
    for group in annotation.evidence_groups:
        if not group:
            return None
        span_signatures: set[tuple[int, int, int, int]] = set()
        group_ordinals: set[int] = set()
        for evidence in group:
            if evidence.docid != annotation.article_docid:
                return None
            token_valid = (
                0
                <= evidence.start_token
                < evidence.end_token
                <= len(document.flattened_tokens)
            )
            sentence_valid = (
                0
                <= evidence.start_sentence
                < evidence.end_sentence
                <= len(document.sentences)
            )
            if not token_valid or not sentence_valid:
                return None
            contained = (
                document.sentence_token_boundaries[evidence.start_sentence]
                <= evidence.start_token
                and evidence.end_token
                <= document.sentence_token_boundaries[evidence.end_sentence]
            )
            text_exact = (
                evidence.text_tokens is not None
                and evidence.text_tokens
                == document.flattened_tokens[
                    evidence.start_token : evidence.end_token
                ]
            )
            signature = (
                evidence.start_token,
                evidence.end_token,
                evidence.start_sentence,
                evidence.end_sentence,
            )
            if not contained or not text_exact or signature in span_signatures:
                return None
            span_signatures.add(signature)
            group_ordinals.update(
                range(evidence.start_sentence, evidence.end_sentence)
            )
        group_signature = tuple(sorted(span_signatures))
        if group_signature in group_signatures:
            return None
        group_signatures.add(group_signature)
        validated_groups.append(tuple(sorted(group_ordinals)))
    flattened = tuple(
        sorted({ordinal for group in validated_groups for ordinal in group})
    )
    if not flattened:
        return None
    return _PrivateItem(
        annotation=annotation,
        document=document,
        facets=facets,
        flattened_gold_sentence_ordinals=flattened,
        validated_groups=tuple(validated_groups),
    )


def _load_private_source(
    *,
    archive_path: Path,
    sidecar_path: Path,
    expected_archive_sha256: str,
    expected_archive_size: int,
    expected_sidecar_sha256: str,
    expected_sidecar_size: int,
) -> tuple[tuple[_PrivateItem, ...], dict[str, int]]:
    archive_hash, archive_size = _sha256_file(archive_path, "official archive")
    sidecar_hash, sidecar_size = _sha256_file(sidecar_path, "prompt sidecar")
    if (
        archive_hash != expected_archive_sha256
        or archive_size != expected_archive_size
    ):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "whole official archive identity drifted"
        )
    if sidecar_hash != expected_sidecar_sha256 or sidecar_size != expected_sidecar_size:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "whole prompt sidecar identity drifted"
        )
    split_raw, split_root, header_audit = _read_authorized_split_members(
        archive_path
    )
    parsed = {
        split: _parse_split_jsonl(split_raw[split], split=split)
        for split in ("train", "val")
    }
    all_annotations = parsed["train"] + parsed["val"]
    identities = [annotation.annotation_id for annotation in all_annotations]
    if len(identities) != len(set(identities)):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "PromptID overlaps train and validation"
        )
    referenced_docids = frozenset(
        annotation.article_docid for annotation in all_annotations
    )
    documents, document_open_count = _read_referenced_documents(
        archive_path,
        split_root=split_root,
        referenced_docids=referenced_docids,
    )
    facets = _read_bound_facets(sidecar_path, all_annotations)
    query_counts = Counter(
        annotation.normalized_query_sha256 for annotation in all_annotations
    )
    duplicate_query_hashes = {
        digest for digest, count in query_counts.items() if count > 1
    }
    eligible: list[_PrivateItem] = []
    incomplete_count = 0
    duplicate_excluded_count = 0
    for annotation in all_annotations:
        if annotation.normalized_query_sha256 in duplicate_query_hashes:
            duplicate_excluded_count += 1
            continue
        item = _validated_private_item(
            annotation,
            documents[annotation.article_docid],
            facets[annotation.annotation_id],
        )
        if item is None:
            incomplete_count += 1
        else:
            eligible.append(item)
    audit = {
        "authorized_split_member_open_count": header_audit[
            "authorized_split_member_open_count"
        ],
        "referenced_document_member_open_count": document_open_count,
        "test_member_content_open_count": header_audit[
            "test_member_content_open_count"
        ],
        "unreferenced_document_content_open_count": 0,
        "duplicate_normalized_query_group_count": len(duplicate_query_hashes),
        "duplicate_normalized_query_annotation_exclusion_count": (
            duplicate_excluded_count
        ),
        "incomplete_or_ineligible_annotation_count": incomplete_count,
        "eligible_annotation_count": len(eligible),
    }
    return tuple(eligible), audit


def _verify_qualification_receipt(payload: Mapping[str, Any]) -> str:
    declared = _verify_self_hash(
        payload,
        schema=QUALIFICATION_SCHEMA,
        field=QUALIFICATION_SELF_HASH_FIELD,
    )
    source = payload.get("source_binding")
    claim = payload.get("claim_boundary")
    opened = payload.get("opened_content_boundary")
    cross = payload.get("cross_split_article_disjointness")
    capacity = payload.get("article_disjoint_capacity")
    prompt = payload.get("independent_structured_prompt_binding")
    if (
        payload.get("status") != "passed_source_qualification_no_selection"
        or not isinstance(source, Mapping)
        or not isinstance(claim, Mapping)
        or not isinstance(opened, Mapping)
        or not isinstance(cross, Mapping)
        or not isinstance(capacity, Mapping)
        or not isinstance(prompt, Mapping)
    ):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "qualification receipt did not pass the frozen boundary"
        )
    for field in (
        "whole_archive_sha256",
        "prompt_sidecar_sha256",
        "custody_manifest_self_sha256",
        "access_manifest_self_sha256",
        "prompt_access_manifest_self_sha256",
    ):
        _require_sha256(source.get(field), f"qualification {field}")
    if (
        type(source.get("whole_archive_size")) is not int
        or source["whole_archive_size"] <= 0
        or type(source.get("prompt_sidecar_size")) is not int
        or source["prompt_sidecar_size"] <= 0
        or claim.get("selection_secret_opened_or_generated") is not False
        or claim.get("cohort_selected") is not False
        or claim.get("online_or_network_evaluation_used") is not False
        or claim.get("test_member_query_document_label_or_content_opened")
        is not False
        or opened.get("test_member_content_open_count") != 0
        or cross.get("article_disjoint") is not True
        or cross.get("train_validation_article_overlap_count") != 0
        or set(capacity) != {"train", "val"}
        or any(
            not isinstance(capacity[split], Mapping)
            or capacity[split].get("exact_article_disjoint_capacity_met")
            is not True
            for split in ("train", "val")
        )
        or prompt.get("missing_match_count") != 0
        or prompt.get("duplicate_or_ambiguous_match_count") != 0
        or prompt.get("query_string_reverse_parsing_used") is not False
    ):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "qualification receipt safety or capacity claim drifted"
        )
    return declared


def _verify_design(
    payload: Mapping[str, Any],
    qualification: Mapping[str, Any],
    *,
    enforce_formal_design_identity: bool,
) -> str:
    declared = _verify_self_hash(payload, schema=None, field=DESIGN_SELF_HASH_FIELD)
    if enforce_formal_design_identity and declared != FORMAL_DESIGN_SHA256:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "formal frozen design identity drifted"
        )
    acquisition = payload.get("acquisition_contract")
    blocks = payload.get("block_contract")
    gold = payload.get("gold_and_utility_contract")
    source = payload.get("source_binding")
    qualification_source = qualification["source_binding"]
    if (
        payload.get("version") != "v1"
        or not isinstance(acquisition, Mapping)
        or not isinstance(blocks, Mapping)
        or not isinstance(gold, Mapping)
        or not isinstance(source, Mapping)
        or acquisition.get("all_four_opaque_block_assignments_committed_before_any_action_or_outcome")
        is not True
        or acquisition.get("test_access") is not False
        or blocks.get("total_items") != TOTAL_ITEM_COUNT
        or source.get("test_member_query_document_label_or_content_access")
        is not False
    ):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "frozen acquisition design semantics drifted"
        )
    for block in BLOCK_ORDER:
        row = blocks.get(block)
        if (
            not isinstance(row, Mapping)
            or row.get("official_split") != BLOCK_SPLITS[block]
            or row.get("per_relation_family")
            != BLOCK_FAMILY_QUOTAS[block][FAMILY_ORDER[0]]
            or row.get("total") != BLOCK_COUNTS[block]
        ):
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "frozen block order, split, or quota drifted"
            )
    family_contract = gold.get("classification_relation_families")
    if not isinstance(family_contract, Mapping) or set(family_contract) != set(
        FAMILY_ORDER
    ):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "frozen relation family registry drifted"
        )
    binding_pairs = (
        ("archive_sha256", "whole_archive_sha256"),
        ("archive_size", "whole_archive_size"),
        ("custody_self_sha256", "custody_manifest_self_sha256"),
        ("source_access_self_sha256", "access_manifest_self_sha256"),
        ("prompt_sidecar_sha256", "prompt_sidecar_sha256"),
        ("prompt_sidecar_access_self_sha256", "prompt_access_manifest_self_sha256"),
    )
    if any(source.get(left) != qualification_source.get(right) for left, right in binding_pairs):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "design and qualification source binding drifted"
        )
    return declared


def _safe_project_relative(value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "implementation freeze relative path drifted"
        )
    path = PurePosixPath(value)
    parts = tuple(part for part in path.parts if part not in {"", "."})
    if path.is_absolute() or not parts or any(part == ".." for part in parts):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "implementation freeze relative path is unsafe"
        )
    return Path(*parts)


def _verify_general_implementation_freeze(
    payload: Mapping[str, Any],
    *,
    design_sha256: str,
    project_root: Path,
) -> str:
    declared = _verify_self_hash(
        payload,
        schema=IMPLEMENTATION_FREEZE_SCHEMA,
        field=IMPLEMENTATION_FREEZE_SELF_HASH_FIELD,
    )
    binding = payload.get("implementation_binding")
    files = binding.get("files") if isinstance(binding, Mapping) else None
    if (
        payload.get("status")
        != "frozen_before_source_qualification_or_private_assignment"
        or payload.get("design_sha256") != design_sha256
        or payload.get("required_role_registry")
        != list(REQUIRED_IMPLEMENTATION_ROLE_REGISTRY)
        or not isinstance(files, list)
        or len(files) != len(REQUIRED_IMPLEMENTATION_ROLE_REGISTRY)
    ):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "general implementation freeze semantics drifted"
        )
    root = project_root.resolve(strict=True)
    seen: set[str] = set()
    seen_roles: set[str] = set()
    for row in files:
        if not isinstance(row, Mapping):
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "implementation freeze file row drifted"
            )
        relative_value = row.get("relative_path", row.get("path"))
        digest_value = row.get("sha256", row.get("file_sha256"))
        role = row.get("role")
        if set(row) != {"relative_path", "role", "sha256"} or role not in REQUIRED_IMPLEMENTATION_ROLE_REGISTRY:
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "implementation freeze file role schema drifted"
            )
        relative = _safe_project_relative(relative_value)
        digest = _require_sha256(digest_value, "frozen implementation file")
        canonical_relative = relative.as_posix()
        if canonical_relative in seen:
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "implementation freeze file path is duplicated"
            )
        seen.add(canonical_relative)
        if str(role) in seen_roles:
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "implementation freeze role is duplicated"
            )
        seen_roles.add(str(role))
        candidate = root / relative
        observed, _size = _sha256_file(candidate, "frozen implementation file")
        if observed != digest:
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "listed implementation file SHA256 drifted"
            )
    if seen_roles != set(REQUIRED_IMPLEMENTATION_ROLE_REGISTRY):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "implementation freeze required role registry is incomplete"
        )
    return declared


def _load_and_verify_bindings(
    *,
    qualification_receipt_path: Path,
    design_path: Path,
    implementation_freeze_path: Path,
    project_root: Path,
    enforce_formal_design_identity: bool,
) -> tuple[
    dict[str, Any],
    bytes,
    dict[str, Any],
    bytes,
    dict[str, Any],
    bytes,
]:
    qualification, qualification_raw = _read_json_path(
        qualification_receipt_path, "qualification receipt"
    )
    design, design_raw = _read_json_path(design_path, "frozen design")
    implementation_freeze, implementation_freeze_raw = _read_json_path(
        implementation_freeze_path, "general implementation freeze"
    )
    _verify_qualification_receipt(qualification)
    _verify_design(
        design,
        qualification,
        enforce_formal_design_identity=enforce_formal_design_identity,
    )
    _verify_general_implementation_freeze(
        implementation_freeze,
        design_sha256=design["design_sha256"],
        project_root=project_root,
    )
    return (
        qualification,
        qualification_raw,
        design,
        design_raw,
        implementation_freeze,
        implementation_freeze_raw,
    )


def _ensure_private_directory(path: Path, *, create: bool) -> None:
    if create:
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        except OSError as exc:
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "private acquisition directory cannot be created"
            ) from exc
    if path.is_symlink():
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "private acquisition directory may not be a symlink"
        )
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "private acquisition directory is unavailable"
        ) from exc
    if not stat.S_ISDIR(mode) or stat.S_IMODE(mode) != 0o700:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "private acquisition directory mode must be 0700"
        )


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> bytes:
    _ensure_private_directory(path.parent, create=True)
    raw = canonical_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "exclusive acquisition artifact already exists"
        ) from exc
    except OSError as exc:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "exclusive acquisition artifact cannot be created"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "exclusive acquisition artifact write failed"
        ) from exc
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "private acquisition artifact mode must be 0600"
        )
    return raw


def _read_private_json(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    _regular_nonsymlink(path, field)
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            f"{field} mode must be 0600"
        )
    payload, raw = _read_json_path(path, field)
    if raw != canonical_bytes(payload):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            f"{field} is not byte-canonical"
        )
    return payload, raw


def _secret_commitment(secret: bytes) -> str:
    if not isinstance(secret, bytes) or len(secret) != 32:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "selection secret must be exactly 32 bytes"
        )
    return hashlib.sha256(
        b"eraser-ei-direct-acquisition-secret-v1\x00" + secret
    ).hexdigest()


def _private_item_body(item: _PrivateItem) -> dict[str, Any]:
    annotation = item.annotation
    return {
        "annotation_id": annotation.annotation_id,
        "article_docid": annotation.article_docid,
        "document_member_content_sha256": item.document.member_content_sha256,
        "family": annotation.family,
        "flattened_gold_sentence_ordinals": list(
            item.flattened_gold_sentence_ordinals
        ),
        "intervention": item.facets[0],
        "comparator": item.facets[1],
        "outcome": item.facets[2],
        "normalized_query_sha256": annotation.normalized_query_sha256,
        "query": annotation.query,
        "sentence_tokens": [list(sentence) for sentence in item.document.sentences],
        "source_split": annotation.source_split,
        "validated_groups": [list(group) for group in item.validated_groups],
    }


def _item_commitment(secret: bytes, item: _PrivateItem) -> str:
    return hmac.new(
        secret,
        b"item_commitment\x00" + canonical_bytes(_private_item_body(item)),
        hashlib.sha256,
    ).hexdigest()


def _assignment_hmac(secret: bytes, *, block: str, item: _PrivateItem) -> str:
    annotation = item.annotation
    message = canonical_bytes(
        {
            "block": block,
            "family": annotation.family,
            "annotation_id": annotation.annotation_id,
            "article_docid": annotation.article_docid,
            "normalized_query_sha256": annotation.normalized_query_sha256,
            "source_split": annotation.source_split,
            "version": VERSION,
        }
    )
    return hmac.new(secret, b"block_assignment\x00" + message, hashlib.sha256).hexdigest()


def _assignment_slots() -> tuple[tuple[str, str, int, int], ...]:
    slots: list[tuple[str, str, int, int]] = []
    for block in BLOCK_ORDER:
        block_ordinal = 0
        for family in FAMILY_ORDER:
            for family_rank in range(BLOCK_FAMILY_QUOTAS[block][family]):
                slots.append((block, family, family_rank, block_ordinal))
                block_ordinal += 1
    return tuple(slots)


def _completion_is_article_disjoint_feasible(
    items: Sequence[_PrivateItem],
    remaining_slots: Sequence[tuple[str, str, int, int]],
    used_articles: set[str],
) -> bool:
    """Exact bipartite feasibility for the still-unassigned fixed slots."""

    if not remaining_slots:
        return True
    article_options: dict[str, set[tuple[str, str]]] = {}
    for item in items:
        annotation = item.annotation
        if annotation.article_docid not in used_articles:
            article_options.setdefault(annotation.article_docid, set()).add(
                (annotation.source_split, annotation.family)
            )
    slot_options: list[tuple[str, ...]] = []
    for block, family, _family_rank, _block_ordinal in remaining_slots:
        slot_type = (BLOCK_SPLITS[block], family)
        options = tuple(
            article
            for article, supported in article_options.items()
            if slot_type in supported
        )
        if not options:
            return False
        slot_options.append(options)
    # Constrained slots first makes the standard augmenting-path matching
    # deterministic and cheap for the 144-slot frozen design.
    ordered_options = sorted(slot_options, key=lambda values: (len(values), values))
    matched_slot_by_article: dict[str, int] = {}

    def augment(slot_i: int, seen: set[str]) -> bool:
        for article in ordered_options[slot_i]:
            if article in seen:
                continue
            seen.add(article)
            previous = matched_slot_by_article.get(article)
            if previous is None or augment(previous, seen):
                matched_slot_by_article[article] = slot_i
                return True
        return False

    return all(augment(slot_i, set()) for slot_i in range(len(ordered_options)))


def _select_assignments(
    items: Sequence[_PrivateItem], secret: bytes
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, dict[str, int]]]:
    used_annotations: set[str] = set()
    used_articles: set[str] = set()
    assignments: list[dict[str, Any]] = []
    eligible_counts: dict[str, dict[str, int]] = {
        split: {
            family: sum(
                item.annotation.source_split == split
                and item.annotation.family == family
                for item in items
            )
            for family in FAMILY_ORDER
        }
        for split in ("train", "val")
    }
    article_conflict_skips = Counter({block: 0 for block in BLOCK_ORDER})
    slots = _assignment_slots()
    if not _completion_is_article_disjoint_feasible(items, slots, set()):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "terminal source epoch: fixed article-disjoint quota is infeasible"
        )
    for slot_i, (block, family, family_rank, block_ordinal) in enumerate(slots):
        split = BLOCK_SPLITS[block]
        ordered = sorted(
            (
                item
                for item in items
                if item.annotation.source_split == split
                and item.annotation.family == family
                and item.annotation.annotation_id not in used_annotations
            ),
            key=lambda item: (
                _assignment_hmac(secret, block=block, item=item),
                item.annotation.annotation_id,
            ),
        )
        selected: _PrivateItem | None = None
        infeasible_articles: set[str] = set()
        for item in ordered:
            article = item.annotation.article_docid
            if article in used_articles:
                article_conflict_skips[block] += 1
                continue
            if article in infeasible_articles:
                continue
            if _completion_is_article_disjoint_feasible(
                items,
                slots[slot_i + 1 :],
                used_articles | {article},
            ):
                selected = item
                break
            infeasible_articles.add(article)
        if selected is None:
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "terminal source epoch: no HMAC candidate preserves fixed completion"
            )
        annotation = selected.annotation
        assignment_digest = _assignment_hmac(
            secret, block=block, item=selected
        )
        assignments.append(
            {
                "annotation_id": annotation.annotation_id,
                "article_docid": annotation.article_docid,
                "assignment_hmac_sha256": assignment_digest,
                "block": block,
                "block_ordinal": block_ordinal,
                "family": family,
                "family_hmac_rank": family_rank,
                "item_commitment_sha256": _item_commitment(secret, selected),
                "normalized_query_sha256": annotation.normalized_query_sha256,
                "source_split": split,
            }
        )
        used_annotations.add(annotation.annotation_id)
        used_articles.add(annotation.article_docid)
    if (
        len(assignments) != TOTAL_ITEM_COUNT
        or len(used_annotations) != TOTAL_ITEM_COUNT
        or len(used_articles) != TOTAL_ITEM_COUNT
    ):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "global assignment uniqueness drifted"
        )
    return assignments, dict(article_conflict_skips), eligible_counts


def _marker_payload(
    *,
    secret: bytes,
    secret_generated_by_os_random: bool,
    qualification_sha256: str,
    qualification_file_sha256: str,
    design_sha256: str,
    design_file_sha256: str,
    implementation_freeze_sha256: str,
    implementation_freeze_file_sha256: str,
) -> dict[str, Any]:
    return _self_hashed(
        {
            "schema": f"{VERSION}_source_epoch_marker",
            "version": VERSION,
            "status": "source_epoch_started_failure_is_terminal",
            "selection_secret_hex": secret.hex(),
            "selection_secret_byte_count": 32,
            "selection_secret_generated_by_os_random": (
                secret_generated_by_os_random
            ),
            "selection_secret_commitment_sha256": _secret_commitment(secret),
            "qualification_sha256": qualification_sha256,
            "qualification_file_sha256": qualification_file_sha256,
            "design_sha256": design_sha256,
            "design_file_sha256": design_file_sha256,
            "implementation_freeze_sha256": implementation_freeze_sha256,
            "implementation_freeze_file_sha256": (
                implementation_freeze_file_sha256
            ),
            "test_access_authorized": False,
            "retry_replay_resample_or_secret_rotation_authorized": False,
        },
        "source_epoch_marker_sha256",
    )


def _assignment_payload(
    *,
    marker: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]],
    source_binding: Mapping[str, Any],
    source_audit: Mapping[str, int],
    article_conflict_skips: Mapping[str, int],
    eligible_counts: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    block_counts = Counter(str(row["block"]) for row in assignments)
    family_counts = {
        block: Counter(
            str(row["family"])
            for row in assignments
            if row["block"] == block
        )
        for block in BLOCK_ORDER
    }
    return _self_hashed(
        {
            "schema": f"{VERSION}_private_assignment",
            "version": VERSION,
            "status": "all_four_blocks_privately_committed_before_action",
            "source_epoch_marker_sha256": marker["source_epoch_marker_sha256"],
            "selection_secret_hex": marker["selection_secret_hex"],
            "selection_secret_commitment_sha256": marker[
                "selection_secret_commitment_sha256"
            ],
            "qualification_sha256": marker["qualification_sha256"],
            "design_sha256": marker["design_sha256"],
            "implementation_freeze_sha256": marker[
                "implementation_freeze_sha256"
            ],
            "source_binding": {
                "whole_archive_sha256": source_binding["whole_archive_sha256"],
                "whole_archive_size": source_binding["whole_archive_size"],
                "prompt_sidecar_sha256": source_binding["prompt_sidecar_sha256"],
                "prompt_sidecar_size": source_binding["prompt_sidecar_size"],
            },
            "block_order": list(BLOCK_ORDER),
            "family_order": list(FAMILY_ORDER),
            "selection_algorithm": (
                "fixed_slot_order_minimum_HMAC_preserving_exact_remaining_"
                "article_disjoint_bipartite_completion"
            ),
            "block_counts": dict(block_counts),
            "block_family_counts": {
                block: dict(family_counts[block]) for block in BLOCK_ORDER
            },
            "assignment_count": len(assignments),
            "assignments": [dict(row) for row in assignments],
            "eligible_counts_by_split_family": {
                split: dict(rows) for split, rows in eligible_counts.items()
            },
            "article_conflict_skip_counts_by_block": dict(article_conflict_skips),
            "source_access_audit": dict(source_audit),
            "one_annotation_per_article": True,
            "whole_duplicate_normalized_query_groups_excluded": True,
            "test_assignment_count": 0,
        },
        "private_assignment_sha256",
    )


def _items_by_identity(items: Sequence[_PrivateItem]) -> dict[tuple[str, str], _PrivateItem]:
    result = {
        (item.annotation.annotation_id, item.annotation.article_docid): item
        for item in items
    }
    if len(result) != len(items):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "private source identity registry drifted"
        )
    return result


def _view_payload(
    *,
    block: str,
    assignment: Mapping[str, Any],
    private_items: Sequence[_PrivateItem],
    authorization_field: str,
    authorization_sha256: str,
) -> dict[str, Any]:
    item_registry = _items_by_identity(private_items)
    secret = bytes.fromhex(str(assignment["selection_secret_hex"]))
    rows = [row for row in assignment["assignments"] if row["block"] == block]
    output: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row["annotation_id"]), str(row["article_docid"]))
        item = item_registry.get(key)
        if item is None or _item_commitment(secret, item) != row[
            "item_commitment_sha256"
        ]:
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "assigned item no longer reconstructs from source"
            )
        output.append(
            {
                "block_ordinal": row["block_ordinal"],
                "item_commitment_sha256": row["item_commitment_sha256"],
                "payload": {
                    "query": item.annotation.query,
                    "official_ico": {
                        "Intervention": item.facets[0],
                        "Comparator": item.facets[1],
                        "Outcome": item.facets[2],
                    },
                    "sentence_tokens": [
                        list(sentence) for sentence in item.document.sentences
                    ],
                },
            }
        )
    return _self_hashed(
        {
            "schema": f"{VERSION}_label_free_view",
            "version": VERSION,
            "status": "exact_query_ico_sentence_view_materialized_no_label",
            "block": block,
            "source_split": BLOCK_SPLITS[block],
            "item_count": BLOCK_COUNTS[block],
            "private_assignment_sha256": assignment[
                "private_assignment_sha256"
            ],
            "authorization_field": authorization_field,
            "authorization_sha256": authorization_sha256,
            "items": output,
            "family_gold_annotation_docid_or_test_value_included": False,
        },
        "label_free_view_sha256",
    )


def _view_path(root: Path, block: str) -> Path:
    return root / VIEW_DIRECTORY / f"{block}.private.json"


def _view_file_binding(path: Path, payload: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    return {
        "block": payload["block"],
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "item_count": payload["item_count"],
        "label_free_view_sha256": payload["label_free_view_sha256"],
        "relative_path": (VIEW_DIRECTORY / path.name).as_posix(),
    }


def _public_receipt_payload(
    *,
    marker: Mapping[str, Any],
    assignment: Mapping[str, Any],
    assignment_raw: bytes,
    initial_view_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    source_audit = assignment["source_access_audit"]
    return _self_hashed(
        {
            "schema": f"{VERSION}_public_receipt",
            "version": VERSION,
            "status": "private_assignment_complete_initial_label_free_views_materialized",
            "qualification_sha256": assignment["qualification_sha256"],
            "qualification_file_sha256": marker["qualification_file_sha256"],
            "design_sha256": assignment["design_sha256"],
            "design_file_sha256": marker["design_file_sha256"],
            "implementation_freeze_sha256": assignment[
                "implementation_freeze_sha256"
            ],
            "implementation_freeze_file_sha256": marker[
                "implementation_freeze_file_sha256"
            ],
            "source_epoch_marker_sha256": assignment[
                "source_epoch_marker_sha256"
            ],
            "selection_secret_commitment_sha256": assignment[
                "selection_secret_commitment_sha256"
            ],
            "private_assignment_sha256": assignment[
                "private_assignment_sha256"
            ],
            "private_assignment_file_sha256": hashlib.sha256(
                assignment_raw
            ).hexdigest(),
            "block_order": list(BLOCK_ORDER),
            "family_order": list(FAMILY_ORDER),
            "selection_algorithm": assignment["selection_algorithm"],
            "block_counts": dict(BLOCK_COUNTS),
            "block_family_quotas": {
                block: dict(BLOCK_FAMILY_QUOTAS[block]) for block in BLOCK_ORDER
            },
            "total_assignment_count": TOTAL_ITEM_COUNT,
            "initial_label_free_view_bindings": [
                dict(row) for row in initial_view_bindings
            ],
            "initially_materialized_blocks": ["A_form", "F_search"],
            "initially_sealed_blocks": ["A_hold", "M_search"],
            "duplicate_normalized_query_group_count": source_audit[
                "duplicate_normalized_query_group_count"
            ],
            "duplicate_normalized_query_annotation_exclusion_count": source_audit[
                "duplicate_normalized_query_annotation_exclusion_count"
            ],
            "source_access_safe_aggregates": {
                "authorized_split_member_open_count": source_audit[
                    "authorized_split_member_open_count"
                ],
                "referenced_document_member_open_count": source_audit[
                    "referenced_document_member_open_count"
                ],
                "test_member_content_open_count": 0,
                "unreferenced_document_content_open_count": 0,
            },
            "private_identifier_query_document_evidence_or_item_hash_emitted": False,
            "selection_secret_emitted": False,
            "family_or_gold_in_label_free_view": False,
            "F_search_label_pack_created": False,
            "test_access_authorized_or_performed": False,
            "online_or_network_evaluation_used": False,
            "retry_replay_resample_replacement_or_secret_rotation": 0,
        },
        "public_receipt_sha256",
    )


def _forbidden_label_free_key(value: object) -> bool:
    forbidden = {
        "family",
        "classification",
        "gold",
        "gold_ordinals",
        "flattened_gold_sentence_ordinals",
        "validated_groups",
        "evidence",
        "evidences",
        "annotation_id",
        "article_docid",
        "docid",
        "test",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).casefold()
            if lowered in forbidden or lowered.startswith("test_"):
                return True
            if _forbidden_label_free_key(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_forbidden_label_free_key(child) for child in value)
    return False


def _verify_marker(payload: Mapping[str, Any]) -> bytes:
    _verify_self_hash(
        payload,
        schema=f"{VERSION}_source_epoch_marker",
        field="source_epoch_marker_sha256",
    )
    secret_hex = payload.get("selection_secret_hex")
    try:
        secret = bytes.fromhex(secret_hex) if isinstance(secret_hex, str) else b""
    except ValueError as exc:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "selection secret encoding drifted"
        ) from exc
    if (
        payload.get("status") != "source_epoch_started_failure_is_terminal"
        or len(secret) != 32
        or payload.get("selection_secret_byte_count") != 32
        or payload.get("selection_secret_commitment_sha256")
        != _secret_commitment(secret)
        or type(payload.get("selection_secret_generated_by_os_random")) is not bool
        or payload.get("test_access_authorized") is not False
        or payload.get("retry_replay_resample_or_secret_rotation_authorized")
        is not False
    ):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "source epoch marker semantics drifted"
        )
    for field in (
        "qualification_sha256",
        "qualification_file_sha256",
        "design_sha256",
        "design_file_sha256",
        "implementation_freeze_sha256",
        "implementation_freeze_file_sha256",
    ):
        _require_sha256(payload.get(field), f"marker {field}")
    return secret


def _verify_assignment(payload: Mapping[str, Any], marker: Mapping[str, Any]) -> str:
    declared = _verify_self_hash(
        payload,
        schema=f"{VERSION}_private_assignment",
        field="private_assignment_sha256",
    )
    rows = payload.get("assignments")
    source_binding = payload.get("source_binding")
    if (
        payload.get("status")
        != "all_four_blocks_privately_committed_before_action"
        or payload.get("source_epoch_marker_sha256")
        != marker.get("source_epoch_marker_sha256")
        or payload.get("selection_secret_hex") != marker.get("selection_secret_hex")
        or payload.get("selection_secret_commitment_sha256")
        != marker.get("selection_secret_commitment_sha256")
        or payload.get("qualification_sha256") != marker.get("qualification_sha256")
        or payload.get("design_sha256") != marker.get("design_sha256")
        or payload.get("implementation_freeze_sha256")
        != marker.get("implementation_freeze_sha256")
        or payload.get("block_order") != list(BLOCK_ORDER)
        or payload.get("family_order") != list(FAMILY_ORDER)
        or payload.get("selection_algorithm")
        != "fixed_slot_order_minimum_HMAC_preserving_exact_remaining_article_disjoint_bipartite_completion"
        or payload.get("assignment_count") != TOTAL_ITEM_COUNT
        or payload.get("test_assignment_count") != 0
        or payload.get("one_annotation_per_article") is not True
        or payload.get("whole_duplicate_normalized_query_groups_excluded") is not True
        or not isinstance(rows, list)
        or len(rows) != TOTAL_ITEM_COUNT
        or not isinstance(source_binding, Mapping)
        or set(source_binding)
        != {
            "whole_archive_sha256",
            "whole_archive_size",
            "prompt_sidecar_sha256",
            "prompt_sidecar_size",
        }
    ):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "private assignment semantics drifted"
        )
    _require_sha256(source_binding["whole_archive_sha256"], "assignment archive")
    _require_sha256(source_binding["prompt_sidecar_sha256"], "assignment sidecar")
    if (
        type(source_binding["whole_archive_size"]) is not int
        or source_binding["whole_archive_size"] <= 0
        or type(source_binding["prompt_sidecar_size"]) is not int
        or source_binding["prompt_sidecar_size"] <= 0
    ):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "private assignment source size binding drifted"
        )
    secret = _verify_marker(marker)
    seen_ids: set[str] = set()
    seen_docs: set[str] = set()
    seen_commitments: set[str] = set()
    block_counts = Counter()
    family_counts: dict[str, Counter[str]] = {
        block: Counter() for block in BLOCK_ORDER
    }
    expected_block_ordinal = Counter()
    for row in rows:
        required = {
            "annotation_id",
            "article_docid",
            "assignment_hmac_sha256",
            "block",
            "block_ordinal",
            "family",
            "family_hmac_rank",
            "item_commitment_sha256",
            "normalized_query_sha256",
            "source_split",
        }
        if not isinstance(row, Mapping) or set(row) != required:
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "private assignment row schema drifted"
            )
        block = row.get("block")
        family = row.get("family")
        annotation_id = row.get("annotation_id")
        docid = row.get("article_docid")
        commitment = _require_sha256(
            row.get("item_commitment_sha256"), "item commitment"
        )
        _require_sha256(row.get("assignment_hmac_sha256"), "assignment HMAC")
        _require_sha256(row.get("normalized_query_sha256"), "normalized query")
        if (
            block not in BLOCK_ORDER
            or family not in FAMILY_ORDER
            or row.get("source_split") != BLOCK_SPLITS[str(block)]
            or not isinstance(annotation_id, str)
            or not annotation_id
            or not isinstance(docid, str)
            or not docid
            or type(row.get("block_ordinal")) is not int
            or row.get("block_ordinal") != expected_block_ordinal[str(block)]
            or type(row.get("family_hmac_rank")) is not int
            or not 0 <= row["family_hmac_rank"] < BLOCK_FAMILY_QUOTAS[str(block)][str(family)]
            or annotation_id in seen_ids
            or docid in seen_docs
            or commitment in seen_commitments
        ):
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "private assignment row semantics drifted"
            )
        # Recompute the HMAC from the stored private identity registry.
        message = canonical_bytes(
            {
                "block": block,
                "family": family,
                "annotation_id": annotation_id,
                "article_docid": docid,
                "normalized_query_sha256": row["normalized_query_sha256"],
                "source_split": row["source_split"],
                "version": VERSION,
            }
        )
        expected_hmac = hmac.new(
            secret, b"block_assignment\x00" + message, hashlib.sha256
        ).hexdigest()
        if row["assignment_hmac_sha256"] != expected_hmac:
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "private assignment HMAC drifted"
            )
        expected_block_ordinal[str(block)] += 1
        block_counts[str(block)] += 1
        family_counts[str(block)][str(family)] += 1
        seen_ids.add(annotation_id)
        seen_docs.add(docid)
        seen_commitments.add(commitment)
    if dict(block_counts) != BLOCK_COUNTS or any(
        dict(family_counts[block]) != BLOCK_FAMILY_QUOTAS[block]
        for block in BLOCK_ORDER
    ):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "private assignment fixed quota drifted"
        )
    return declared


def _verify_view(
    payload: Mapping[str, Any],
    assignment: Mapping[str, Any],
    *,
    block: str,
) -> str:
    _require_exact_keys(
        payload,
        {
            "schema",
            "version",
            "status",
            "block",
            "source_split",
            "item_count",
            "private_assignment_sha256",
            "authorization_field",
            "authorization_sha256",
            "items",
            "family_gold_annotation_docid_or_test_value_included",
            "label_free_view_sha256",
        },
        "label-free view",
    )
    declared = _verify_self_hash(
        payload,
        schema=f"{VERSION}_label_free_view",
        field="label_free_view_sha256",
    )
    rows = payload.get("items")
    assigned = [
        row for row in assignment["assignments"] if row["block"] == block
    ]
    if (
        payload.get("version") != VERSION
        or payload.get("status")
        != "exact_query_ico_sentence_view_materialized_no_label"
        or payload.get("block") != block
        or payload.get("source_split") != BLOCK_SPLITS[block]
        or payload.get("item_count") != BLOCK_COUNTS[block]
        or payload.get("private_assignment_sha256")
        != assignment["private_assignment_sha256"]
        or payload.get("family_gold_annotation_docid_or_test_value_included")
        is not False
        or not isinstance(rows, list)
        or len(rows) != BLOCK_COUNTS[block]
    ):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "label-free view semantics drifted"
        )
    for row, expected in zip(rows, assigned):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"block_ordinal", "item_commitment_sha256", "payload"}
            or row.get("block_ordinal") != expected["block_ordinal"]
            or row.get("item_commitment_sha256")
            != expected["item_commitment_sha256"]
            or not isinstance(row.get("payload"), Mapping)
            or set(row["payload"]) != {"official_ico", "query", "sentence_tokens"}
            or _forbidden_label_free_key(row["payload"])
        ):
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "label-free view item schema or ordinal drifted"
            )
        payload_row = row["payload"]
        ico = payload_row.get("official_ico")
        sentences = payload_row.get("sentence_tokens")
        if (
            not isinstance(payload_row.get("query"), str)
            or not payload_row["query"].strip()
            or not isinstance(ico, Mapping)
            or set(ico) != {"Intervention", "Comparator", "Outcome"}
            or any(not isinstance(value, str) or not value.strip() for value in ico.values())
            or not isinstance(sentences, list)
            or len(sentences) < 5
            or any(
                not isinstance(sentence, list)
                or not sentence
                or any(not isinstance(token, str) or not token for token in sentence)
                for sentence in sentences
            )
        ):
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "label-free exact query, I/C/O, or sentence view drifted"
            )
    _require_sha256(payload.get("authorization_sha256"), "view authorization")
    if not isinstance(payload.get("authorization_field"), str):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "view authorization field drifted"
        )
    return declared


def _verify_public_receipt(
    payload: Mapping[str, Any],
    *,
    marker: Mapping[str, Any],
    assignment: Mapping[str, Any],
    assignment_raw: bytes,
    view_bindings: Sequence[Mapping[str, Any]],
) -> str:
    _require_exact_keys(
        payload,
        {
            "schema",
            "version",
            "status",
            "qualification_sha256",
            "qualification_file_sha256",
            "design_sha256",
            "design_file_sha256",
            "implementation_freeze_sha256",
            "implementation_freeze_file_sha256",
            "source_epoch_marker_sha256",
            "selection_secret_commitment_sha256",
            "private_assignment_sha256",
            "private_assignment_file_sha256",
            "block_order",
            "family_order",
            "selection_algorithm",
            "block_counts",
            "block_family_quotas",
            "total_assignment_count",
            "initial_label_free_view_bindings",
            "initially_materialized_blocks",
            "initially_sealed_blocks",
            "duplicate_normalized_query_group_count",
            "duplicate_normalized_query_annotation_exclusion_count",
            "source_access_safe_aggregates",
            "private_identifier_query_document_evidence_or_item_hash_emitted",
            "selection_secret_emitted",
            "family_or_gold_in_label_free_view",
            "F_search_label_pack_created",
            "test_access_authorized_or_performed",
            "online_or_network_evaluation_used",
            "retry_replay_resample_replacement_or_secret_rotation",
            "public_receipt_sha256",
        },
        "public acquisition receipt",
    )
    declared = _verify_self_hash(
        payload,
        schema=f"{VERSION}_public_receipt",
        field="public_receipt_sha256",
    )
    if (
        payload.get("version") != VERSION
        or payload.get("status")
        != "private_assignment_complete_initial_label_free_views_materialized"
        or payload.get("qualification_sha256") != marker["qualification_sha256"]
        or payload.get("qualification_file_sha256")
        != marker["qualification_file_sha256"]
        or payload.get("design_sha256") != marker["design_sha256"]
        or payload.get("design_file_sha256") != marker["design_file_sha256"]
        or payload.get("implementation_freeze_sha256")
        != marker["implementation_freeze_sha256"]
        or payload.get("implementation_freeze_file_sha256")
        != marker["implementation_freeze_file_sha256"]
        or payload.get("source_epoch_marker_sha256")
        != marker["source_epoch_marker_sha256"]
        or payload.get("selection_secret_commitment_sha256")
        != marker["selection_secret_commitment_sha256"]
        or payload.get("private_assignment_sha256")
        != assignment["private_assignment_sha256"]
        or payload.get("private_assignment_file_sha256")
        != hashlib.sha256(assignment_raw).hexdigest()
        or payload.get("block_order") != list(BLOCK_ORDER)
        or payload.get("family_order") != list(FAMILY_ORDER)
        or payload.get("selection_algorithm") != assignment["selection_algorithm"]
        or payload.get("block_counts") != BLOCK_COUNTS
        or payload.get("block_family_quotas") != BLOCK_FAMILY_QUOTAS
        or payload.get("total_assignment_count") != TOTAL_ITEM_COUNT
        or payload.get("initially_materialized_blocks") != ["A_form", "F_search"]
        or payload.get("initially_sealed_blocks") != ["A_hold", "M_search"]
        or payload.get("initial_label_free_view_bindings")
        != [dict(row) for row in view_bindings]
        or payload.get("private_identifier_query_document_evidence_or_item_hash_emitted")
        is not False
        or payload.get("selection_secret_emitted") is not False
        or payload.get("family_or_gold_in_label_free_view") is not False
        or payload.get("F_search_label_pack_created") is not False
        or payload.get("test_access_authorized_or_performed") is not False
        or payload.get("online_or_network_evaluation_used") is not False
        or payload.get("retry_replay_resample_replacement_or_secret_rotation") != 0
    ):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "public acquisition receipt drifted"
        )
    return declared


def verify_acquisition_state(
    *,
    acquisition_root: Path,
    qualification_receipt_path: Path | None = None,
    design_path: Path | None = None,
    implementation_freeze_path: Path | None = None,
    project_root: Path | None = None,
    enforce_formal_design_identity: bool = False,
) -> dict[str, Any]:
    """Verify a complete state without reopening the source archive or sidecar."""

    root = acquisition_root.resolve(strict=True)
    _ensure_private_directory(root, create=False)
    marker, _marker_raw = _read_private_json(
        root / MARKER_RELATIVE, "source epoch marker"
    )
    _verify_marker(marker)
    assignment, assignment_raw = _read_private_json(
        root / ASSIGNMENT_RELATIVE, "private assignment"
    )
    _verify_assignment(assignment, marker)
    binding_options = (
        qualification_receipt_path,
        design_path,
        implementation_freeze_path,
        project_root,
    )
    if any(value is not None for value in binding_options) and any(
        value is None for value in binding_options
    ):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "state binding verification requires qualification, design, freeze, and project"
        )
    if all(value is not None for value in binding_options):
        assert qualification_receipt_path is not None
        assert design_path is not None
        assert implementation_freeze_path is not None
        assert project_root is not None
        (
            qualification,
            qualification_raw,
            design,
            design_raw,
            implementation_freeze,
            implementation_freeze_raw,
        ) = (
            _load_and_verify_bindings(
                qualification_receipt_path=qualification_receipt_path,
                design_path=design_path,
                implementation_freeze_path=implementation_freeze_path,
                project_root=project_root,
                enforce_formal_design_identity=enforce_formal_design_identity,
            )
        )
        if (
            qualification["qualification_sha256"]
            != marker["qualification_sha256"]
            or hashlib.sha256(qualification_raw).hexdigest()
            != marker["qualification_file_sha256"]
            or design["design_sha256"] != marker["design_sha256"]
            or hashlib.sha256(design_raw).hexdigest()
            != marker["design_file_sha256"]
            or implementation_freeze["implementation_freeze_sha256"]
            != marker["implementation_freeze_sha256"]
            or hashlib.sha256(implementation_freeze_raw).hexdigest()
            != marker["implementation_freeze_file_sha256"]
        ):
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "recovered prerequisite file binding drifted"
            )
    view_bindings: list[dict[str, Any]] = []
    for block in ("A_form", "F_search"):
        path = _view_path(root, block)
        view, raw = _read_private_json(path, f"{block} label-free view")
        _verify_view(view, assignment, block=block)
        view_bindings.append(_view_file_binding(path, view, raw))
    public, _public_raw = _read_private_json(
        root / PUBLIC_RECEIPT_RELATIVE, "public acquisition receipt"
    )
    _verify_public_receipt(
        public,
        marker=marker,
        assignment=assignment,
        assignment_raw=assignment_raw,
        view_bindings=view_bindings,
    )
    return public


def acquire_once(
    *,
    archive_path: Path,
    prompt_sidecar_path: Path,
    qualification_receipt_path: Path,
    design_path: Path,
    implementation_freeze_path: Path,
    project_root: Path,
    acquisition_root: Path,
    selection_secret: bytes | None = None,
    enforce_formal_design_identity: bool = False,
) -> dict[str, Any]:
    """Commit all 144 items once and materialize only A_form/F_search views.

    ``selection_secret`` exists for deterministic synthetic tests and controlled
    reproducibility audits.  Formal use must omit it and set
    ``enforce_formal_design_identity=True``; the secret is then generated once
    with ``os.urandom(32)`` and persisted before archive-member access.
    """

    if enforce_formal_design_identity and selection_secret is not None:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "formal acquisition forbids caller-supplied selection secret"
        )

    (
        qualification,
        qualification_raw,
        design,
        design_raw,
        implementation_freeze,
        implementation_freeze_raw,
    ) = _load_and_verify_bindings(
        qualification_receipt_path=qualification_receipt_path,
        design_path=design_path,
        implementation_freeze_path=implementation_freeze_path,
        project_root=project_root,
        enforce_formal_design_identity=enforce_formal_design_identity,
    )
    root = acquisition_root.absolute()
    if root.exists():
        if selection_secret is not None:
            recovered_marker, _raw = _read_private_json(
                root / MARKER_RELATIVE, "source epoch marker"
            )
            recovered_secret = _verify_marker(recovered_marker)
            if not hmac.compare_digest(recovered_secret, selection_secret):
                raise EraserEvidenceInferenceDirectAcquisitionError(
                    "recovery attempted selection secret rotation"
                )
        return verify_acquisition_state(
            acquisition_root=root,
            qualification_receipt_path=qualification_receipt_path,
            design_path=design_path,
            implementation_freeze_path=implementation_freeze_path,
            project_root=project_root,
            enforce_formal_design_identity=enforce_formal_design_identity,
        )
    if not root.parent.exists():
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "acquisition root parent must already exist"
        )
    try:
        root.mkdir(mode=0o700)
    except OSError as exc:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "private acquisition root cannot be created"
        ) from exc
    _ensure_private_directory(root, create=False)
    generated = selection_secret is None
    secret = os.urandom(32) if selection_secret is None else selection_secret
    if not isinstance(secret, bytes) or len(secret) != 32:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "selection secret must be exactly 32 bytes"
        )
    marker = _marker_payload(
        secret=secret,
        secret_generated_by_os_random=generated,
        qualification_sha256=qualification["qualification_sha256"],
        qualification_file_sha256=hashlib.sha256(qualification_raw).hexdigest(),
        design_sha256=design["design_sha256"],
        design_file_sha256=hashlib.sha256(design_raw).hexdigest(),
        implementation_freeze_sha256=implementation_freeze[
            "implementation_freeze_sha256"
        ],
        implementation_freeze_file_sha256=hashlib.sha256(
            implementation_freeze_raw
        ).hexdigest(),
    )
    _write_exclusive(root / MARKER_RELATIVE, marker)

    source = qualification["source_binding"]
    items, source_audit = _load_private_source(
        archive_path=archive_path,
        sidecar_path=prompt_sidecar_path,
        expected_archive_sha256=source["whole_archive_sha256"],
        expected_archive_size=source["whole_archive_size"],
        expected_sidecar_sha256=source["prompt_sidecar_sha256"],
        expected_sidecar_size=source["prompt_sidecar_size"],
    )
    if source_audit["test_member_content_open_count"] != 0:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "TEST member content was opened"
        )
    assignments, conflict_skips, eligible_counts = _select_assignments(
        items, secret
    )
    assignment = _assignment_payload(
        marker=marker,
        assignments=assignments,
        source_binding=source,
        source_audit=source_audit,
        article_conflict_skips=conflict_skips,
        eligible_counts=eligible_counts,
    )
    assignment_raw = _write_exclusive(root / ASSIGNMENT_RELATIVE, assignment)
    _verify_assignment(assignment, marker)

    view_bindings: list[dict[str, Any]] = []
    for block in ("A_form", "F_search"):
        view = _view_payload(
            block=block,
            assignment=assignment,
            private_items=items,
            authorization_field="source_epoch_marker_sha256",
            authorization_sha256=marker["source_epoch_marker_sha256"],
        )
        path = _view_path(root, block)
        raw = _write_exclusive(path, view)
        _verify_view(view, assignment, block=block)
        view_bindings.append(_view_file_binding(path, view, raw))
    public = _public_receipt_payload(
        marker=marker,
        assignment=assignment,
        assignment_raw=assignment_raw,
        initial_view_bindings=view_bindings,
    )
    _write_exclusive(root / PUBLIC_RECEIPT_RELATIVE, public)
    return verify_acquisition_state(
        acquisition_root=root,
        qualification_receipt_path=qualification_receipt_path,
        design_path=design_path,
        implementation_freeze_path=implementation_freeze_path,
        project_root=project_root,
        enforce_formal_design_identity=enforce_formal_design_identity,
    )


def _load_base_private_state(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _ensure_private_directory(root, create=False)
    marker, _marker_raw = _read_private_json(
        root / MARKER_RELATIVE, "source epoch marker"
    )
    _verify_marker(marker)
    assignment, assignment_raw = _read_private_json(
        root / ASSIGNMENT_RELATIVE, "private assignment"
    )
    _verify_assignment(assignment, marker)
    view_bindings: list[dict[str, Any]] = []
    for block in ("A_form", "F_search"):
        path = _view_path(root, block)
        view, raw = _read_private_json(path, f"{block} label-free view")
        _verify_view(view, assignment, block=block)
        view_bindings.append(_view_file_binding(path, view, raw))
    public, _public_raw = _read_private_json(
        root / PUBLIC_RECEIPT_RELATIVE, "public acquisition receipt"
    )
    _verify_public_receipt(
        public,
        marker=marker,
        assignment=assignment,
        assignment_raw=assignment_raw,
        view_bindings=view_bindings,
    )
    return marker, assignment, public


def derive_a_form_fold_key(*, acquisition_root: Path) -> bytes:
    """Return the 32-byte domain-separated A_form fold key.

    The unique persisted selection secret never leaves this module's public
    API.  Controllers receive only this one-purpose HMAC derivative.
    """

    root = acquisition_root.resolve(strict=True)
    marker, _assignment, public = _load_base_private_state(root)
    secret = _verify_marker(marker)
    message = canonical_bytes(
        {
            "domain": "A_form_four_fold_assignment_key",
            "private_assignment_sha256": public["private_assignment_sha256"],
            "public_receipt_sha256": public["public_receipt_sha256"],
            "version": VERSION,
        }
    )
    key = hmac.new(secret, b"fold_key\x00" + message, hashlib.sha256).digest()
    if len(key) != 32:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "A_form fold key length drifted"
        )
    return key


def _require_exact_keys(payload: Mapping[str, Any], keys: set[str], field: str) -> None:
    if set(payload) != keys:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            f"{field} exact key schema drifted"
        )


def build_label_capability(
    *,
    block: str,
    private_assignment_sha256: str,
    public_receipt_sha256: str,
    label_free_view_sha256: str,
    three_arm_execution_seal_sha256: str,
    feature_seal_sha256: str,
) -> dict[str, Any]:
    if block not in {"A_form", "A_hold", "M_search"}:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "label capability block is invalid or permanently label-free"
        )
    for value, field in (
        (private_assignment_sha256, "label assignment"),
        (public_receipt_sha256, "label public receipt"),
        (label_free_view_sha256, "label-free view"),
        (three_arm_execution_seal_sha256, "three-arm execution seal"),
        (feature_seal_sha256, "feature seal"),
    ):
        _require_sha256(value, field)
    return _self_hashed(
        {
            "schema": f"{VERSION}_label_capability",
            "version": VERSION,
            "status": "three_arm_execution_and_features_sealed_label_open_authorized",
            "authorized_block": block,
            "private_assignment_sha256": private_assignment_sha256,
            "public_receipt_sha256": public_receipt_sha256,
            "label_free_view_sha256": label_free_view_sha256,
            "three_arm_execution_seal_sha256": three_arm_execution_seal_sha256,
            "feature_seal_sha256": feature_seal_sha256,
            "upstream_typed_artifact_content_verified_by_acquisition": False,
            "label_materialization_authorized": True,
            "test_access_authorized": False,
        },
        "label_capability_sha256",
    )


def verify_label_capability(
    payload: Mapping[str, Any],
    *,
    block: str,
    private_assignment_sha256: str,
    public_receipt_sha256: str,
    label_free_view_sha256: str,
) -> str:
    keys = {
        "schema",
        "version",
        "status",
        "authorized_block",
        "private_assignment_sha256",
        "public_receipt_sha256",
        "label_free_view_sha256",
        "three_arm_execution_seal_sha256",
        "feature_seal_sha256",
        "upstream_typed_artifact_content_verified_by_acquisition",
        "label_materialization_authorized",
        "test_access_authorized",
        "label_capability_sha256",
    }
    _require_exact_keys(payload, keys, "label capability")
    declared = _verify_self_hash(
        payload,
        schema=f"{VERSION}_label_capability",
        field="label_capability_sha256",
    )
    if (
        block not in {"A_form", "A_hold", "M_search"}
        or payload.get("version") != VERSION
        or payload.get("status")
        != "three_arm_execution_and_features_sealed_label_open_authorized"
        or payload.get("authorized_block") != block
        or payload.get("private_assignment_sha256") != private_assignment_sha256
        or payload.get("public_receipt_sha256") != public_receipt_sha256
        or payload.get("label_free_view_sha256") != label_free_view_sha256
        or payload.get("label_materialization_authorized") is not True
        or payload.get("upstream_typed_artifact_content_verified_by_acquisition")
        is not False
        or payload.get("test_access_authorized") is not False
    ):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "label capability provenance drifted"
        )
    _require_sha256(
        payload.get("three_arm_execution_seal_sha256"), "three-arm execution seal"
    )
    _require_sha256(payload.get("feature_seal_sha256"), "feature seal")
    return declared


def build_f_policy_seal(
    *,
    private_assignment_sha256: str,
    public_receipt_sha256: str,
    a_form_label_free_view_sha256: str,
    f_search_label_free_view_sha256: str,
    a_form_three_arm_execution_seal_sha256: str,
    f_search_three_arm_execution_seal_sha256: str,
    a_form_feature_seal_sha256: str,
    f_search_feature_seal_sha256: str,
    a_form_label_pack_sha256: str,
    a_form_label_capability_sha256: str,
    a_form_label_capability_file_sha256: str,
    a_form_label_stage_marker_sha256: str,
    e3_fit_receipt_sha256: str,
    f_search_policy_receipt_sha256: str,
) -> dict[str, Any]:
    values = locals()
    for field, value in values.items():
        _require_sha256(value, field)
    return _self_hashed(
        {
            "schema": f"{VERSION}_f_policy_seal",
            "version": VERSION,
            "status": "F_policy_frozen_A_hold_authorized",
            "authorized_block": "A_hold",
            **values,
            "upstream_typed_artifact_content_verified_by_acquisition": False,
            "A_hold_materialization_authorized": True,
            "M_search_materialization_authorized": False,
            "test_access_authorized": False,
        },
        "f_policy_seal_sha256",
    )


def verify_f_policy_seal(
    payload: Mapping[str, Any],
    *,
    private_assignment_sha256: str,
    public_receipt_sha256: str,
    a_form_label_free_view_sha256: str,
    f_search_label_free_view_sha256: str,
) -> str:
    hash_fields = {
        "private_assignment_sha256",
        "public_receipt_sha256",
        "a_form_label_free_view_sha256",
        "f_search_label_free_view_sha256",
        "a_form_three_arm_execution_seal_sha256",
        "f_search_three_arm_execution_seal_sha256",
        "a_form_feature_seal_sha256",
        "f_search_feature_seal_sha256",
        "a_form_label_pack_sha256",
        "a_form_label_capability_sha256",
        "a_form_label_capability_file_sha256",
        "a_form_label_stage_marker_sha256",
        "e3_fit_receipt_sha256",
        "f_search_policy_receipt_sha256",
    }
    keys = hash_fields | {
        "schema",
        "version",
        "status",
        "authorized_block",
        "A_hold_materialization_authorized",
        "M_search_materialization_authorized",
        "test_access_authorized",
        "f_policy_seal_sha256",
        "upstream_typed_artifact_content_verified_by_acquisition",
    }
    _require_exact_keys(payload, keys, "F policy seal")
    declared = _verify_self_hash(
        payload,
        schema=f"{VERSION}_f_policy_seal",
        field="f_policy_seal_sha256",
    )
    if (
        payload.get("version") != VERSION
        or payload.get("status") != "F_policy_frozen_A_hold_authorized"
        or payload.get("authorized_block") != "A_hold"
        or payload.get("private_assignment_sha256") != private_assignment_sha256
        or payload.get("public_receipt_sha256") != public_receipt_sha256
        or payload.get("a_form_label_free_view_sha256")
        != a_form_label_free_view_sha256
        or payload.get("f_search_label_free_view_sha256")
        != f_search_label_free_view_sha256
        or payload.get("A_hold_materialization_authorized") is not True
        or payload.get("M_search_materialization_authorized") is not False
        or payload.get("test_access_authorized") is not False
        or payload.get("upstream_typed_artifact_content_verified_by_acquisition")
        is not False
    ):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "F policy seal provenance drifted"
        )
    for field in hash_fields:
        _require_sha256(payload.get(field), field)
    return declared


def build_a_hold_promotion_seal(
    *,
    private_assignment_sha256: str,
    public_receipt_sha256: str,
    a_hold_label_free_view_sha256: str,
    f_policy_seal_sha256: str,
    a_hold_three_arm_execution_seal_sha256: str,
    a_hold_feature_seal_sha256: str,
    a_hold_label_pack_sha256: str,
    a_hold_label_capability_sha256: str,
    a_hold_label_capability_file_sha256: str,
    a_hold_label_stage_marker_sha256: str,
    a_hold_score_receipt_sha256: str,
    promotion_decision_sha256: str,
) -> dict[str, Any]:
    values = locals()
    for field, value in values.items():
        _require_sha256(value, field)
    return _self_hashed(
        {
            "schema": f"{VERSION}_a_hold_promotion_seal",
            "version": VERSION,
            "status": "A_hold_promoted_M_search_authorized",
            "authorized_block": "M_search",
            **values,
            "upstream_typed_artifact_content_verified_by_acquisition": False,
            "evaluator_promoted": True,
            "M_search_materialization_authorized": True,
            "test_access_authorized": False,
        },
        "a_hold_promotion_seal_sha256",
    )


def verify_a_hold_promotion_seal(
    payload: Mapping[str, Any],
    *,
    private_assignment_sha256: str,
    public_receipt_sha256: str,
    a_hold_label_free_view_sha256: str,
) -> str:
    hash_fields = {
        "private_assignment_sha256",
        "public_receipt_sha256",
        "a_hold_label_free_view_sha256",
        "f_policy_seal_sha256",
        "a_hold_three_arm_execution_seal_sha256",
        "a_hold_feature_seal_sha256",
        "a_hold_label_pack_sha256",
        "a_hold_label_capability_sha256",
        "a_hold_label_capability_file_sha256",
        "a_hold_label_stage_marker_sha256",
        "a_hold_score_receipt_sha256",
        "promotion_decision_sha256",
    }
    keys = hash_fields | {
        "schema",
        "version",
        "status",
        "authorized_block",
        "evaluator_promoted",
        "M_search_materialization_authorized",
        "test_access_authorized",
        "a_hold_promotion_seal_sha256",
        "upstream_typed_artifact_content_verified_by_acquisition",
    }
    _require_exact_keys(payload, keys, "A_hold promotion seal")
    declared = _verify_self_hash(
        payload,
        schema=f"{VERSION}_a_hold_promotion_seal",
        field="a_hold_promotion_seal_sha256",
    )
    if (
        payload.get("version") != VERSION
        or payload.get("status") != "A_hold_promoted_M_search_authorized"
        or payload.get("authorized_block") != "M_search"
        or payload.get("private_assignment_sha256") != private_assignment_sha256
        or payload.get("public_receipt_sha256") != public_receipt_sha256
        or payload.get("a_hold_label_free_view_sha256")
        != a_hold_label_free_view_sha256
        or payload.get("evaluator_promoted") is not True
        or payload.get("M_search_materialization_authorized") is not True
        or payload.get("test_access_authorized") is not False
        or payload.get("upstream_typed_artifact_content_verified_by_acquisition")
        is not False
    ):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "A_hold promotion seal provenance drifted"
        )
    for field in hash_fields:
        _require_sha256(payload.get(field), field)
    return declared


def _load_authorization(
    path: Path,
    *,
    assignment_sha256: str,
    public_receipt_sha256: str,
    block: str,
    view_sha256: str | None = None,
    a_form_view_sha256: str | None = None,
    f_search_view_sha256: str | None = None,
) -> tuple[dict[str, Any], str, str]:
    payload, raw = _read_json_path(path, f"{block} authorization")
    schema = payload.get("schema")
    if schema == f"{VERSION}_label_capability":
        if view_sha256 is None:
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "label capability requires its exact view binding"
            )
        declared = verify_label_capability(
            payload,
            block=block,
            private_assignment_sha256=assignment_sha256,
            public_receipt_sha256=public_receipt_sha256,
            label_free_view_sha256=view_sha256,
        )
    elif schema == f"{VERSION}_f_policy_seal" and block == "A_hold":
        if a_form_view_sha256 is None or f_search_view_sha256 is None:
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "F policy seal requires both frozen TRAIN view bindings"
            )
        declared = verify_f_policy_seal(
            payload,
            private_assignment_sha256=assignment_sha256,
            public_receipt_sha256=public_receipt_sha256,
            a_form_label_free_view_sha256=a_form_view_sha256,
            f_search_label_free_view_sha256=f_search_view_sha256,
        )
    elif schema == f"{VERSION}_a_hold_promotion_seal" and block == "M_search":
        if view_sha256 is None:
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "promotion seal requires the exact A_hold view binding"
            )
        declared = verify_a_hold_promotion_seal(
            payload,
            private_assignment_sha256=assignment_sha256,
            public_receipt_sha256=public_receipt_sha256,
            a_hold_label_free_view_sha256=view_sha256,
        )
    else:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            f"{block} authorization exact schema drifted"
        )
    return payload, declared, hashlib.sha256(raw).hexdigest()


def _stage_marker_payload(
    *,
    block: str,
    stage: str,
    assignment_sha256: str,
    authorization_field: str,
    authorization_sha256: str,
    authorization_file_sha256: str,
    prerequisite_view_sha256: str | None = None,
) -> dict[str, Any]:
    if prerequisite_view_sha256 is not None:
        _require_sha256(prerequisite_view_sha256, "stage prerequisite view")
    return _self_hashed(
        {
            "schema": f"{VERSION}_stage_marker",
            "version": VERSION,
            "status": "authorized_stage_started_failure_is_terminal",
            "stage": stage,
            "block": block,
            "private_assignment_sha256": assignment_sha256,
            "authorization_field": authorization_field,
            "authorization_sha256": authorization_sha256,
            "authorization_file_sha256": authorization_file_sha256,
            "prerequisite_view_sha256": prerequisite_view_sha256,
            "retry_replay_resample_or_replacement_authorized": False,
            "test_access_authorized": False,
        },
        "stage_marker_sha256",
    )


def _stage_marker_path(root: Path, *, stage: str, block: str) -> Path:
    return root / STAGE_MARKER_DIRECTORY / f"{stage}.{block}.private.json"


def _verify_stage_marker(
    payload: Mapping[str, Any],
    *,
    stage: str,
    block: str,
    assignment_sha256: str,
    authorization_sha256: str,
    authorization_file_sha256: str | None = None,
    authorization_field: str | None = None,
    prerequisite_view_sha256: str | None = None,
) -> str:
    _require_exact_keys(
        payload,
        {
            "schema",
            "version",
            "status",
            "stage",
            "block",
            "private_assignment_sha256",
            "authorization_field",
            "authorization_sha256",
            "authorization_file_sha256",
            "prerequisite_view_sha256",
            "retry_replay_resample_or_replacement_authorized",
            "test_access_authorized",
            "stage_marker_sha256",
        },
        "authorized stage marker",
    )
    declared = _verify_self_hash(
        payload,
        schema=f"{VERSION}_stage_marker",
        field="stage_marker_sha256",
    )
    if (
        payload.get("version") != VERSION
        or payload.get("status") != "authorized_stage_started_failure_is_terminal"
        or payload.get("stage") != stage
        or payload.get("block") != block
        or payload.get("private_assignment_sha256") != assignment_sha256
        or payload.get("authorization_sha256") != authorization_sha256
        or (
            authorization_file_sha256 is not None
            and payload.get("authorization_file_sha256")
            != authorization_file_sha256
        )
        or (
            authorization_field is not None
            and payload.get("authorization_field") != authorization_field
        )
        or payload.get("prerequisite_view_sha256")
        != prerequisite_view_sha256
        or payload.get("retry_replay_resample_or_replacement_authorized")
        is not False
        or payload.get("test_access_authorized") is not False
    ):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "authorized stage marker drifted"
        )
    _require_sha256(payload.get("authorization_file_sha256"), "authorization file")
    return declared


def _verify_late_authorization(
    *,
    root: Path,
    assignment: Mapping[str, Any],
    public: Mapping[str, Any],
    block: str,
    authorization_path: Path,
) -> tuple[str, str, str]:
    if block == "A_hold":
        self_field = "f_policy_seal_sha256"
        a_form, _raw = _read_private_json(
            _view_path(root, "A_form"), "A_form label-free view"
        )
        f_search, _raw = _read_private_json(
            _view_path(root, "F_search"), "F_search label-free view"
        )
        a_form_sha = _verify_view(a_form, assignment, block="A_form")
        f_search_sha = _verify_view(f_search, assignment, block="F_search")
        a_hold_sha = None
    elif block == "M_search":
        self_field = "a_hold_promotion_seal_sha256"
        a_hold, _raw = _read_private_json(
            _view_path(root, "A_hold"), "A_hold label-free view"
        )
        a_hold_sha = _verify_view(a_hold, assignment, block="A_hold")
        a_form_sha = None
        f_search_sha = None
    else:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "late authorization block is invalid"
        )
    authorization, authorization_sha, authorization_file_sha = _load_authorization(
        authorization_path,
        assignment_sha256=assignment["private_assignment_sha256"],
        public_receipt_sha256=public["public_receipt_sha256"],
        block=block,
        view_sha256=a_hold_sha,
        a_form_view_sha256=a_form_sha,
        f_search_view_sha256=f_search_sha,
    )
    required_label_block = "A_form" if block == "A_hold" else "A_hold"
    label_state = _load_verified_label_state(
        root=root,
        block=required_label_block,
        assignment=assignment,
        public=public,
    )
    prefix = "a_form" if block == "A_hold" else "a_hold"
    capability = label_state["label_capability"]
    if (
        authorization.get(f"{prefix}_label_pack_sha256")
        != label_state["label_pack_sha256"]
        or authorization.get(f"{prefix}_label_capability_sha256")
        != label_state["label_capability_sha256"]
        or authorization.get(f"{prefix}_label_capability_file_sha256")
        != label_state["label_capability_file_sha256"]
        or authorization.get(f"{prefix}_label_stage_marker_sha256")
        != label_state["label_stage_marker_sha256"]
        or authorization.get(f"{prefix}_three_arm_execution_seal_sha256")
        != capability["three_arm_execution_seal_sha256"]
        or authorization.get(f"{prefix}_feature_seal_sha256")
        != capability["feature_seal_sha256"]
    ):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "late authorization prerequisite label chain drifted"
        )
    if block == "M_search":
        a_hold_stage, _raw = _read_private_json(
            _stage_marker_path(root, stage="view", block="A_hold"),
            "A_hold view stage marker",
        )
        _verify_stage_marker(
            a_hold_stage,
            stage="view",
            block="A_hold",
            assignment_sha256=assignment["private_assignment_sha256"],
            authorization_sha256=str(authorization["f_policy_seal_sha256"]),
            authorization_field="f_policy_seal_sha256",
        )
        if (
            a_hold.get("authorization_field") != "f_policy_seal_sha256"
            or a_hold.get("authorization_sha256")
            != authorization.get("f_policy_seal_sha256")
        ):
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "promotion-to-F-policy authorization chain drifted"
            )
    return self_field, authorization_sha, authorization_file_sha


def _recover_late_view(
    *,
    root: Path,
    assignment: Mapping[str, Any],
    block: str,
    authorization_sha256: str,
    authorization_file_sha256: str,
    authorization_field: str,
) -> dict[str, Any] | None:
    path = _view_path(root, block)
    marker_path = _stage_marker_path(root, stage="view", block=block)
    if not path.exists() and not marker_path.exists():
        return None
    if not path.exists() or not marker_path.exists():
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "authorized view stage is incomplete and terminal"
        )
    marker, _raw = _read_private_json(marker_path, f"{block} view stage marker")
    _verify_stage_marker(
        marker,
        stage="view",
        block=block,
        assignment_sha256=assignment["private_assignment_sha256"],
        authorization_sha256=authorization_sha256,
        authorization_file_sha256=authorization_file_sha256,
        authorization_field=authorization_field,
    )
    view, _view_raw = _read_private_json(path, f"{block} label-free view")
    _verify_view(view, assignment, block=block)
    if (
        view.get("authorization_sha256") != authorization_sha256
        or view.get("authorization_field") != authorization_field
    ):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "recovered view authorization drifted"
        )
    return view


def materialize_late_view_once(
    *,
    archive_path: Path,
    prompt_sidecar_path: Path,
    acquisition_root: Path,
    block: str,
    authorization_path: Path,
) -> dict[str, Any]:
    """Materialize A_hold from an F seal or M_search from a promoted seal."""

    if block not in {"A_hold", "M_search"}:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "late view API accepts only A_hold or M_search"
        )
    root = acquisition_root.resolve(strict=True)
    marker, assignment, public = _load_base_private_state(root)
    self_field, authorization_sha, authorization_file_sha = (
        _verify_late_authorization(
            root=root,
            assignment=assignment,
            public=public,
            block=block,
            authorization_path=authorization_path,
        )
    )
    recovered = _recover_late_view(
        root=root,
        assignment=assignment,
        block=block,
        authorization_sha256=authorization_sha,
        authorization_file_sha256=authorization_file_sha,
        authorization_field=self_field,
    )
    if recovered is not None:
        return recovered
    stage_marker = _stage_marker_payload(
        block=block,
        stage="view",
        assignment_sha256=assignment["private_assignment_sha256"],
        authorization_field=self_field,
        authorization_sha256=authorization_sha,
        authorization_file_sha256=authorization_file_sha,
    )
    _write_exclusive(
        _stage_marker_path(root, stage="view", block=block), stage_marker
    )
    # Exact source byte bindings were safely copied into the private assignment
    # audit only as aggregates, so recover them from the still-bound qualifier.
    # The caller supplies the same byte paths; hashes are taken from the public
    # qualification file via the acquisition marker's immutable file binding.
    # To avoid accepting caller metadata, persist these four source values in
    # the private assignment at creation and use them here.
    source_binding = assignment.get("source_binding")
    if not isinstance(source_binding, Mapping):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "private assignment source binding is absent"
        )
    items, source_audit = _load_private_source(
        archive_path=archive_path,
        sidecar_path=prompt_sidecar_path,
        expected_archive_sha256=str(source_binding["whole_archive_sha256"]),
        expected_archive_size=int(source_binding["whole_archive_size"]),
        expected_sidecar_sha256=str(source_binding["prompt_sidecar_sha256"]),
        expected_sidecar_size=int(source_binding["prompt_sidecar_size"]),
    )
    if source_audit["test_member_content_open_count"] != 0:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "TEST member content was opened"
        )
    view = _view_payload(
        block=block,
        assignment=assignment,
        private_items=items,
        authorization_field=self_field,
        authorization_sha256=authorization_sha,
    )
    _write_exclusive(_view_path(root, block), view)
    _verify_view(view, assignment, block=block)
    return view


def load_verified_block_view(
    *,
    acquisition_root: Path,
    block: str,
    authorization_path: Path | None = None,
) -> dict[str, Any]:
    """Load one block only after validating its complete authorization chain."""

    if block not in BLOCK_ORDER:
        raise EraserEvidenceInferenceDirectAcquisitionError("block is invalid")
    root = acquisition_root.resolve(strict=True)
    marker, assignment, public = _load_base_private_state(root)
    if block in {"A_form", "F_search"}:
        if authorization_path is not None:
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "initial TRAIN views use only the source epoch marker"
            )
        view, _raw = _read_private_json(
            _view_path(root, block), f"{block} label-free view"
        )
        _verify_view(view, assignment, block=block)
        if (
            view.get("authorization_field") != "source_epoch_marker_sha256"
            or view.get("authorization_sha256")
            != marker["source_epoch_marker_sha256"]
        ):
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "initial view source epoch authorization drifted"
            )
        return view
    if authorization_path is None:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "late block load requires its exact authorization seal"
        )
    _self_field, authorization_sha, authorization_file_sha = (
        _verify_late_authorization(
            root=root,
            assignment=assignment,
            public=public,
            block=block,
            authorization_path=authorization_path,
        )
    )
    recovered = _recover_late_view(
        root=root,
        assignment=assignment,
        block=block,
        authorization_sha256=authorization_sha,
        authorization_file_sha256=authorization_file_sha,
        authorization_field=_self_field,
    )
    if recovered is None:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "authorized late block has not been materialized"
        )
    return recovered


def _label_pack_path(root: Path, block: str) -> Path:
    return root / LABEL_DIRECTORY / f"{block}.private.json"


def _label_authorization_path(root: Path, block: str) -> Path:
    """Return the sole acquisition-owned capability path for one label stage."""

    return root / AUTHORIZATION_DIRECTORY / f"label.{block}.private.json"


def _label_pack_payload(
    *,
    block: str,
    assignment: Mapping[str, Any],
    private_items: Sequence[_PrivateItem],
    label_capability_sha256: str,
    label_free_view_sha256: str,
) -> dict[str, Any]:
    registry = _items_by_identity(private_items)
    secret = bytes.fromhex(str(assignment["selection_secret_hex"]))
    assigned = [row for row in assignment["assignments"] if row["block"] == block]
    rows: list[dict[str, Any]] = []
    for row in assigned:
        key = (str(row["annotation_id"]), str(row["article_docid"]))
        item = registry.get(key)
        if item is None or _item_commitment(secret, item) != row[
            "item_commitment_sha256"
        ]:
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "label item does not reconstruct from committed source"
            )
        rows.append(
            {
                "item_commitment_sha256": row["item_commitment_sha256"],
                "family": item.annotation.family,
                "flattened_gold_sentence_ordinals": list(
                    item.flattened_gold_sentence_ordinals
                ),
                "validated_groups": [
                    list(group) for group in item.validated_groups
                ],
            }
        )
    return _self_hashed(
        {
            "schema": f"{VERSION}_label_pack",
            "version": VERSION,
            "status": "authorized_exact_gold_label_pack_materialized",
            "block": block,
            "item_count": BLOCK_COUNTS[block],
            "private_assignment_sha256": assignment[
                "private_assignment_sha256"
            ],
            "label_free_view_sha256": label_free_view_sha256,
            "label_capability_sha256": label_capability_sha256,
            "items": rows,
            "item_field_registry": [
                "item_commitment_sha256",
                "family",
                "flattened_gold_sentence_ordinals",
                "validated_groups",
            ],
            "query_document_ico_annotation_docid_or_test_value_included": False,
        },
        "label_pack_sha256",
    )


def _verify_label_pack(
    payload: Mapping[str, Any],
    *,
    block: str,
    assignment: Mapping[str, Any],
    label_capability_sha256: str,
    label_free_view_sha256: str,
    label_free_view: Mapping[str, Any],
) -> str:
    _require_exact_keys(
        payload,
        {
            "schema",
            "version",
            "status",
            "block",
            "item_count",
            "private_assignment_sha256",
            "label_free_view_sha256",
            "label_capability_sha256",
            "items",
            "item_field_registry",
            "query_document_ico_annotation_docid_or_test_value_included",
            "label_pack_sha256",
        },
        "label pack",
    )
    declared = _verify_self_hash(
        payload,
        schema=f"{VERSION}_label_pack",
        field="label_pack_sha256",
    )
    rows = payload.get("items")
    view_rows = label_free_view.get("items")
    assigned = [row for row in assignment["assignments"] if row["block"] == block]
    allowed_fields = {
        "item_commitment_sha256",
        "family",
        "flattened_gold_sentence_ordinals",
        "validated_groups",
    }
    if (
        block == "F_search"
        or payload.get("version") != VERSION
        or payload.get("status")
        != "authorized_exact_gold_label_pack_materialized"
        or payload.get("block") != block
        or payload.get("item_count") != BLOCK_COUNTS[block]
        or payload.get("private_assignment_sha256")
        != assignment["private_assignment_sha256"]
        or payload.get("label_free_view_sha256") != label_free_view_sha256
        or payload.get("label_capability_sha256") != label_capability_sha256
        or payload.get("item_field_registry") != [
            "item_commitment_sha256",
            "family",
            "flattened_gold_sentence_ordinals",
            "validated_groups",
        ]
        or payload.get("query_document_ico_annotation_docid_or_test_value_included")
        is not False
        or not isinstance(rows, list)
        or len(rows) != BLOCK_COUNTS[block]
        or label_free_view.get("label_free_view_sha256")
        != label_free_view_sha256
        or not isinstance(view_rows, list)
        or len(view_rows) != BLOCK_COUNTS[block]
    ):
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "label pack semantics drifted"
        )
    for row, expected, view_row in zip(rows, assigned, view_rows):
        sentences = (
            view_row.get("payload", {}).get("sentence_tokens")
            if isinstance(view_row, Mapping)
            and isinstance(view_row.get("payload"), Mapping)
            else None
        )
        if (
            not isinstance(row, Mapping)
            or set(row) != allowed_fields
            or row.get("item_commitment_sha256")
            != expected["item_commitment_sha256"]
            or row.get("family") != expected["family"]
            or not isinstance(row.get("flattened_gold_sentence_ordinals"), list)
            or not row["flattened_gold_sentence_ordinals"]
            or any(type(value) is not int or value < 0 for value in row["flattened_gold_sentence_ordinals"])
            or row["flattened_gold_sentence_ordinals"]
            != sorted(set(row["flattened_gold_sentence_ordinals"]))
            or not isinstance(row.get("validated_groups"), list)
            or not row["validated_groups"]
            or any(
                not isinstance(group, list)
                or not group
                or any(type(value) is not int or value < 0 for value in group)
                or group != sorted(set(group))
                for group in row["validated_groups"]
            )
            or sorted(
                {
                    value
                    for group in row["validated_groups"]
                    for value in group
                }
            )
            != row["flattened_gold_sentence_ordinals"]
            or not isinstance(sentences, list)
            or any(
                value >= len(sentences)
                for value in row["flattened_gold_sentence_ordinals"]
            )
        ):
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "label pack item field registry or gold semantics drifted"
            )
    return declared


def _load_verified_label_state(
    *,
    root: Path,
    block: str,
    assignment: Mapping[str, Any],
    public: Mapping[str, Any],
    external_capability_path: Path | None = None,
) -> dict[str, Any]:
    """Close a label stage through its deterministic private capability copy."""

    if block not in {"A_form", "A_hold", "M_search"}:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "verified label state block is invalid or permanently label-free"
        )
    view, _view_raw = _read_private_json(
        _view_path(root, block), f"{block} label-free view"
    )
    view_sha = _verify_view(view, assignment, block=block)

    capability_path = _label_authorization_path(root, block)
    capability, capability_raw = _read_private_json(
        capability_path, f"{block} acquisition-owned label capability"
    )
    capability_sha = verify_label_capability(
        capability,
        block=block,
        private_assignment_sha256=assignment["private_assignment_sha256"],
        public_receipt_sha256=public["public_receipt_sha256"],
        label_free_view_sha256=view_sha,
    )
    capability_file_sha = hashlib.sha256(capability_raw).hexdigest()

    if external_capability_path is not None:
        external, external_sha, _external_file_sha = _load_authorization(
            external_capability_path,
            assignment_sha256=assignment["private_assignment_sha256"],
            public_receipt_sha256=public["public_receipt_sha256"],
            block=block,
            view_sha256=view_sha,
        )
        if (
            not hmac.compare_digest(external_sha, capability_sha)
            or external != capability
        ):
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "external label capability differs from its acquisition-owned copy"
            )

    stage_marker, _stage_raw = _read_private_json(
        _stage_marker_path(root, stage="label", block=block),
        f"{block} label stage marker",
    )
    stage_marker_sha = _verify_stage_marker(
        stage_marker,
        stage="label",
        block=block,
        assignment_sha256=assignment["private_assignment_sha256"],
        authorization_sha256=capability_sha,
        authorization_file_sha256=capability_file_sha,
        authorization_field="label_capability_sha256",
        prerequisite_view_sha256=view_sha,
    )

    pack, _pack_raw = _read_private_json(
        _label_pack_path(root, block), f"{block} label pack"
    )
    pack_sha = _verify_label_pack(
        pack,
        block=block,
        assignment=assignment,
        label_capability_sha256=capability_sha,
        label_free_view_sha256=view_sha,
        label_free_view=view,
    )
    return {
        "block": block,
        "label_free_view_sha256": view_sha,
        "label_capability": capability,
        "label_capability_sha256": capability_sha,
        "label_capability_file_sha256": capability_file_sha,
        "label_stage_marker_sha256": stage_marker_sha,
        "label_pack": pack,
        "label_pack_sha256": pack_sha,
        "upstream_typed_artifact_content_verified_by_acquisition": False,
    }


def load_verified_label_state(
    *,
    acquisition_root: Path,
    block: str,
    label_capability_path: Path | None = None,
) -> dict[str, Any]:
    """Load a label pack only after verifying its complete private hash chain."""

    root = acquisition_root.resolve(strict=True)
    _marker, assignment, public = _load_base_private_state(root)
    return _load_verified_label_state(
        root=root,
        block=block,
        assignment=assignment,
        public=public,
        external_capability_path=label_capability_path,
    )


def materialize_label_pack_once(
    *,
    archive_path: Path,
    prompt_sidecar_path: Path,
    acquisition_root: Path,
    block: str,
    label_capability_path: Path,
) -> dict[str, Any]:
    """Open one non-F label pack only behind its independent capability."""

    if block == "F_search":
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "F_search has no label capability or label pack"
        )
    if block not in {"A_form", "A_hold", "M_search"}:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "label pack block is invalid"
        )
    root = acquisition_root.resolve(strict=True)
    _marker, assignment, public = _load_base_private_state(root)
    view_path = _view_path(root, block)
    if not view_path.exists():
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "label capability cannot precede label-free block view"
        )
    view, _view_raw = _read_private_json(view_path, f"{block} label-free view")
    view_sha = _verify_view(view, assignment, block=block)
    capability, capability_sha, _external_capability_file_sha = _load_authorization(
        label_capability_path,
        assignment_sha256=assignment["private_assignment_sha256"],
        public_receipt_sha256=public["public_receipt_sha256"],
        block=block,
        view_sha256=view_sha,
    )
    capability_file_sha = hashlib.sha256(canonical_bytes(capability)).hexdigest()
    label_path = _label_pack_path(root, block)
    stage_marker_path = _stage_marker_path(root, stage="label", block=block)
    stored_capability_path = _label_authorization_path(root, block)
    existing = (
        label_path.exists(),
        stage_marker_path.exists(),
        stored_capability_path.exists(),
    )
    if any(existing):
        if not all(existing):
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "authorized label stage or capability copy is incomplete and terminal"
            )
        state = _load_verified_label_state(
            root=root,
            block=block,
            assignment=assignment,
            public=public,
            external_capability_path=label_capability_path,
        )
        return dict(state["label_pack"])
    stage_marker = _stage_marker_payload(
        block=block,
        stage="label",
        assignment_sha256=assignment["private_assignment_sha256"],
        authorization_field="label_capability_sha256",
        authorization_sha256=capability_sha,
        authorization_file_sha256=capability_file_sha,
        prerequisite_view_sha256=view_sha,
    )
    _write_exclusive(stage_marker_path, stage_marker)
    copied_raw = _write_exclusive(stored_capability_path, capability)
    if hashlib.sha256(copied_raw).hexdigest() != capability_file_sha:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "acquisition-owned label capability copy drifted while persisting"
        )
    source_binding = assignment["source_binding"]
    items, source_audit = _load_private_source(
        archive_path=archive_path,
        sidecar_path=prompt_sidecar_path,
        expected_archive_sha256=source_binding["whole_archive_sha256"],
        expected_archive_size=source_binding["whole_archive_size"],
        expected_sidecar_sha256=source_binding["prompt_sidecar_sha256"],
        expected_sidecar_size=source_binding["prompt_sidecar_size"],
    )
    if source_audit["test_member_content_open_count"] != 0:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "TEST member content was opened"
        )
    pack = _label_pack_payload(
        block=block,
        assignment=assignment,
        private_items=items,
        label_capability_sha256=capability_sha,
        label_free_view_sha256=view_sha,
    )
    _write_exclusive(label_path, pack)
    state = _load_verified_label_state(
        root=root,
        block=block,
        assignment=assignment,
        public=public,
        external_capability_path=label_capability_path,
    )
    return dict(state["label_pack"])


def verify_full_acquisition_state(
    *,
    acquisition_root: Path,
    late_authorization_paths: Mapping[str, Path] | None = None,
    label_capability_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Verify base state plus every materialized late view/label stage."""

    root = acquisition_root.resolve(strict=True)
    base = verify_acquisition_state(acquisition_root=root)
    _marker, assignment, public = _load_base_private_state(root)
    late_paths = dict(late_authorization_paths or {})
    label_paths = dict(label_capability_paths or {})
    if not set(late_paths) <= {"A_hold", "M_search"} or not set(label_paths) <= {
        "A_form",
        "A_hold",
        "M_search",
    }:
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "full verifier authorization registry contains an invalid block"
        )
    verified_views = ["A_form", "F_search"]
    for block in ("A_hold", "M_search"):
        view_exists = _view_path(root, block).exists()
        marker_exists = _stage_marker_path(root, stage="view", block=block).exists()
        if view_exists != marker_exists:
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "late view and stage marker completeness drifted"
            )
        if view_exists:
            authorization_path = late_paths.get(block)
            if authorization_path is None:
                raise EraserEvidenceInferenceDirectAcquisitionError(
                    "full verifier lacks a materialized late-view authorization"
                )
            load_verified_block_view(
                acquisition_root=root,
                block=block,
                authorization_path=authorization_path,
            )
            verified_views.append(block)
        elif block in late_paths:
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "authorization supplied for an unmaterialized late view"
            )
    f_label = _label_pack_path(root, "F_search")
    f_marker = _stage_marker_path(root, stage="label", block="F_search")
    f_capability = _label_authorization_path(root, "F_search")
    if f_label.exists() or f_marker.exists() or f_capability.exists():
        raise EraserEvidenceInferenceDirectAcquisitionError(
            "F_search label artifact exists despite permanent label prohibition"
        )
    verified_labels: list[str] = []
    for block in ("A_form", "A_hold", "M_search"):
        label_path = _label_pack_path(root, block)
        stage_path = _stage_marker_path(root, stage="label", block=block)
        capability_copy_path = _label_authorization_path(root, block)
        existing = (
            label_path.exists(),
            stage_path.exists(),
            capability_copy_path.exists(),
        )
        if any(existing) and not all(existing):
            raise EraserEvidenceInferenceDirectAcquisitionError(
                "label pack, stage marker, and capability copy completeness drifted"
            )
        if not any(existing):
            if block in label_paths:
                raise EraserEvidenceInferenceDirectAcquisitionError(
                    "capability supplied for an unmaterialized label pack"
                )
            continue
        capability_path = label_paths.get(block)
        _load_verified_label_state(
            root=root,
            block=block,
            assignment=assignment,
            public=public,
            external_capability_path=capability_path,
        )
        verified_labels.append(block)
    return {
        "public_receipt_sha256": base["public_receipt_sha256"],
        "verified_view_blocks": verified_views,
        "verified_label_blocks": verified_labels,
        "test_access_authorized_or_performed": False,
    }


__all__ = [
    "BLOCK_COUNTS",
    "BLOCK_FAMILY_QUOTAS",
    "BLOCK_ORDER",
    "BLOCK_SPLITS",
    "DESIGN_SELF_HASH_FIELD",
    "EraserEvidenceInferenceDirectAcquisitionError",
    "FAMILY_ORDER",
    "FORMAL_DESIGN_SHA256",
    "IMPLEMENTATION_FREEZE_SCHEMA",
    "OFFICIAL_CLASSIFICATION_TO_FAMILY",
    "QUALIFICATION_SCHEMA",
    "REQUIRED_IMPLEMENTATION_ROLE_REGISTRY",
    "TOTAL_ITEM_COUNT",
    "VERSION",
    "acquire_once",
    "build_a_hold_promotion_seal",
    "build_f_policy_seal",
    "build_label_capability",
    "canonical_bytes",
    "derive_a_form_fold_key",
    "load_verified_block_view",
    "load_verified_label_state",
    "materialize_label_pack_once",
    "materialize_late_view_once",
    "stable_hash",
    "verify_acquisition_state",
    "verify_a_hold_promotion_seal",
    "verify_f_policy_seal",
    "verify_full_acquisition_state",
    "verify_label_capability",
]
