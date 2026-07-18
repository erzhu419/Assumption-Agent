"""One-shot private acquisition for the frozen HoVer TRAIN lifecycle.

The formal source/SQLite entry point is intentionally not invoked by tests.
Pure functions accept decoded synthetic rows and an in-memory representation of
the official ``documents(rowid, id, text)`` table.  The formal integration may
only call the same functions after verifying the committed aggregate
qualification receipt and the two pinned source files.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import subprocess
import sys
from typing import Any
import unicodedata
from urllib.parse import quote
import argparse

from assumption_agent.benchmarks import hover_isolated_bootstrap_v1 as isolated_bootstrap


VERSION = "hover_direct_acquisition_v1"
DESIGN_SHA256 = "e558d5305af5a31953a9d87ef92d7cc8d6c4ee48fc82d89eb52e4355826ca818"
TRAINING_SHA256 = "1f1cd57abd616fa00c70bdc575ce77c16fc6cf1a6cffd5ff87c208030a336bb6"
CORPUS_SOURCE_SHA256 = "c37ee397916ec0bffacfe8902db454a5cda88a7a188409217b2e15231fe5ee2f"
TRAINING_SIZE = 9_205_582
CORPUS_SOURCE_SIZE = 2_156_273_664
FORMAL_TRAIN_COUNT = 18_171
FORMAL_HOP_COUNTS = {2: 9_052, 3: 6_084, 4: 3_035}

BLOCK_ORDER = ("A_form", "F_search", "A_hold", "M_search")
HOP_STRATA = ("2_hop", "3_hop", "4_hop")
FAMILIES = HOP_STRATA  # compatibility alias; these are strata, not relation families.
HOP_QUOTAS = {"A_form": 16, "F_search": 12, "A_hold": 10, "M_search": 10}
FAMILY_QUOTAS = HOP_QUOTAS
BLOCK_COUNTS = {block: HOP_QUOTAS[block] * 3 for block in BLOCK_ORDER}
TOTAL_SELECTED = 144
TOTAL_PER_HOP = 48
CORPUS_SIZE = 609
DISTRACTOR_ATTEMPT_CAP = 1_000_000

FORMAL_TRAIN_RELATIVE = Path(
    "artifacts/hover_official_source_v1/hover_train_release_v1.1-39b84697.json"
)
FORMAL_SQLITE_RELATIVE = Path("artifacts/hover_official_source_v1/wiki_wo_links.db")
FORMAL_QUALIFICATION_RELATIVE = Path("manifests/hover_source_qualification_v1.json")
FORMAL_OUTPUT_ROOT_RELATIVE = Path("artifacts/hover_direct_acquisition_v1")
FORMAL_MARKER_RELATIVE = FORMAL_OUTPUT_ROOT_RELATIVE / "acquisition.one_shot_marker.json"
FORMAL_SECRET_RELATIVE = FORMAL_OUTPUT_ROOT_RELATIVE / "selection_secret.bin"
CORPUS_VIEW_RELATIVE = FORMAL_OUTPUT_ROOT_RELATIVE / "private/corpus_view.json"
BLOCK_VIEW_RELATIVES = {
    block: FORMAL_OUTPUT_ROOT_RELATIVE / f"private/{block}.claim_view.json"
    for block in BLOCK_ORDER
}
BLOCK_LABEL_RELATIVES = {
    block: FORMAL_OUTPUT_ROOT_RELATIVE / f"private/{block}.utility_labels.json"
    for block in ("A_form", "A_hold", "M_search")
}
PUBLIC_RECEIPT_RELATIVE = Path("manifests/hover_direct_acquisition_v1_acquisition.json")

CORPUS_VIEW_SCHEMA = f"{VERSION}_corpus_view"
BLOCK_VIEW_SCHEMA = f"{VERSION}_block_view"
BLOCK_LABEL_SCHEMA = f"{VERSION}_block_utility_labels"
VIEW_ITEM_SCHEMA = f"{VERSION}_claim_view_item"
LABEL_ITEM_SCHEMA = f"{VERSION}_utility_label_item"
ONE_SHOT_MARKER_SCHEMA = f"{VERSION}_one_shot_marker"
PUBLIC_RECEIPT_SCHEMA = VERSION
QUALIFICATION_SCHEMA = "hover_source_qualification_v1"
QUALIFICATION_STATUS = "passed_source_qualification_no_selection"
QUALIFICATION_HASH_FIELD = "qualification_sha256"
QUALIFICATION_SHA256 = "b3ef9c012ba4cdcf63ec7837471c06de2d75537965de10b3f36afd4dab6e3fd0"
QUALIFICATION_REQUIRED_KEYS = frozenset(
    {
        "capacity",
        "claim_boundary",
        "hop_and_structure",
        "identity_and_grouping",
        "parser_and_schema",
        "recorded_date",
        "schema",
        "source_binding",
        "sqlite_and_gold_resolution",
        "version",
        "status",
        QUALIFICATION_HASH_FIELD,
    }
)

_SOURCE_REQUIRED_KEYS = frozenset(
    {"uid", "hpqa_id", "claim", "num_hops", "supporting_facts"}
)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HMAC_DOMAIN = b"hover_direct_acquisition_v1/hmac-sha256/v1"


class HoVerAcquisitionError(RuntimeError):
    """A frozen source, cohort, corpus, or isolation invariant drifted."""


@dataclass(frozen=True)
class DocumentRow:
    rowid: int
    exact_id: str
    exact_text: str
    identity_commitment_sha256: str

    def view_row(self, article_id: int) -> dict[str, Any]:
        return {"article_id": article_id, "title": self.exact_id, "body": self.exact_text}


@dataclass(frozen=True)
class DocumentCatalog:
    by_rowid: Mapping[int, DocumentRow]
    rowids_by_exact_nfd_id: Mapping[str, tuple[int, ...]]
    maximum_rowid: int
    catalog_sha256: str

    @property
    def row_count(self) -> int:
        return len(self.by_rowid)

    @property
    def binding_sha256(self) -> str:
        return self.catalog_sha256

    def resolve_exact_nfd_id(self, exact_nfd_id: str) -> tuple[DocumentRow, ...]:
        return tuple(
            self.by_rowid[rowid]
            for rowid in self.rowids_by_exact_nfd_id.get(exact_nfd_id, ())
        )

    def fetch_rowid(self, rowid: int) -> DocumentRow | None:
        return self.by_rowid.get(rowid)


@dataclass(frozen=True)
class QualificationBinding:
    qualification_sha256: str
    eligible_record_count: int
    normalized_claim_collision_member_count: int
    eligible_hpqa_group_count: int
    sqlite_document_row_count: int
    sqlite_maximum_rowid: int


class DocumentResolver:
    """Minimal lazy SQLite boundary used by parsing and filler sampling."""

    row_count: int
    maximum_rowid: int
    binding_sha256: str

    def resolve_exact_nfd_id(self, exact_nfd_id: str) -> tuple[DocumentRow, ...]:
        raise NotImplementedError

    def fetch_rowid(self, rowid: int) -> DocumentRow | None:
        raise NotImplementedError


class ImmutableSQLiteDocumentResolver(DocumentResolver):
    """Lazy, read-only exact-title/rowid gateway; never materializes the DB."""

    def __init__(
        self,
        *,
        path: Path,
        row_count: int,
        maximum_rowid: int,
        binding_sha256: str,
    ) -> None:
        absolute = path.absolute()
        try:
            metadata = absolute.lstat()
        except OSError as exc:
            raise HoVerAcquisitionError("SQLite corpus is unavailable") from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise HoVerAcquisitionError("SQLite corpus must be a mode-0600 regular file")
        if type(row_count) is not int or row_count < CORPUS_SIZE:
            raise HoVerAcquisitionError("SQLite row count binding is invalid")
        if type(maximum_rowid) is not int or maximum_rowid < row_count:
            raise HoVerAcquisitionError("SQLite maximum rowid binding is invalid")
        self.row_count = row_count
        self.maximum_rowid = maximum_rowid
        self.binding_sha256 = _require_sha256(binding_sha256, "SQLite binding")
        self._absolute_path = absolute
        self._connection: sqlite3.Connection | None = None
        self._title_cache: dict[str, tuple[DocumentRow, ...]] = {}
        self._row_cache: dict[int, DocumentRow | None] = {}

    def _ensure_open(self) -> sqlite3.Connection:
        if self._connection is not None:
            return self._connection
        uri = "file:" + quote(str(self._absolute_path), safe="/") + "?mode=ro&immutable=1"
        try:
            connection = sqlite3.connect(uri, uri=True, check_same_thread=True)
            connection.execute("PRAGMA query_only=ON")
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            columns = connection.execute("PRAGMA table_info(documents)").fetchall()
            observed_aggregates = connection.execute(
                "SELECT COUNT(*), MIN(rowid), MAX(rowid) FROM documents"
            ).fetchone()
        except sqlite3.Error as exc:
            try:
                connection.close()
            except (UnboundLocalError, sqlite3.Error):
                pass
            raise HoVerAcquisitionError("immutable SQLite open/schema validation failed") from exc
        if tables != [("documents",)] or [row[1] for row in columns] != ["id", "text"]:
            connection.close()
            raise HoVerAcquisitionError("SQLite documents schema drifted")
        if observed_aggregates != (self.row_count, 1, self.maximum_rowid):
            connection.close()
            raise HoVerAcquisitionError("SQLite row-count or rowid aggregate drifted")
        self._connection = connection
        return connection

    @staticmethod
    def _row(raw: Sequence[Any]) -> DocumentRow:
        if len(raw) != 3 or type(raw[0]) is not int or raw[0] <= 0:
            raise HoVerAcquisitionError("SQLite document row drifted")
        rowid = raw[0]
        exact_id = _text(raw[1], "documents.id", nonempty=True)
        exact_text = _text(raw[2], "documents.text", nonempty=True)
        identity = stable_hash(
            {
                "domain": f"{VERSION}/official-document-row/v1",
                "rowid": rowid,
                "id": exact_id,
                "text": exact_text,
            }
        )
        return DocumentRow(rowid, exact_id, exact_text, identity)

    def resolve_exact_nfd_id(self, exact_nfd_id: str) -> tuple[DocumentRow, ...]:
        title = normalize_support_title_nfd(exact_nfd_id)
        cached = self._title_cache.get(title)
        if cached is not None:
            return cached
        connection = self._ensure_open()
        try:
            rows = connection.execute(
                "SELECT rowid, id, text FROM documents WHERE id = ? ORDER BY rowid",
                (title,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise HoVerAcquisitionError("exact SQLite title lookup failed") from exc
        resolved = tuple(self._row(row) for row in rows)
        self._title_cache[title] = resolved
        for row in resolved:
            self._row_cache[row.rowid] = row
        return resolved

    def fetch_rowid(self, rowid: int) -> DocumentRow | None:
        if type(rowid) is not int or not 1 <= rowid <= self.maximum_rowid:
            raise HoVerAcquisitionError("SQLite rowid lookup is invalid")
        if rowid in self._row_cache:
            return self._row_cache[rowid]
        connection = self._ensure_open()
        try:
            rows = connection.execute(
                "SELECT rowid, id, text FROM documents WHERE rowid = ?", (rowid,)
            ).fetchall()
        except sqlite3.Error as exc:
            raise HoVerAcquisitionError("exact SQLite rowid lookup failed") from exc
        if len(rows) > 1:
            raise HoVerAcquisitionError("SQLite rowid lookup is non-unique")
        resolved = None if not rows else self._row(rows[0])
        self._row_cache[rowid] = resolved
        return resolved

    def close(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()
            self._connection = None

    def __enter__(self) -> "ImmutableSQLiteDocumentResolver":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


@dataclass(frozen=True)
class EligibleRecord:
    source_row_ordinal: int
    uid_sha256: str
    hpqa_id_sha256: str
    claim: str
    normalized_claim: str
    hop_stratum: str
    gold_document_rowids: tuple[int, ...]
    identity_commitment_sha256: str
    source_record_commitment_sha256: str


@dataclass(frozen=True)
class AssignedRecord:
    record: EligibleRecord
    block: str
    hop_stratum: str
    stratum_slot_ordinal: int


@dataclass(frozen=True)
class AcquisitionPaths:
    marker: Path
    secret: Path
    corpus_view: Path
    block_views: Mapping[str, Path]
    block_labels: Mapping[str, Path]
    public_receipt: Path


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise HoVerAcquisitionError("value is not canonical JSON") from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise HoVerAcquisitionError(f"{field} is not a SHA-256")
    return value


def _text(value: object, field: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise HoVerAcquisitionError(f"{field} is not valid text")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise HoVerAcquisitionError(f"{field} is not valid Unicode") from exc
    if nonempty and not value.strip():
        raise HoVerAcquisitionError(f"{field} is empty")
    return value


def normalize_claim(value: str) -> str:
    exact = _text(value, "claim", nonempty=True)
    normalized = " ".join(unicodedata.normalize("NFKC", exact).casefold().split())
    if not normalized:
        raise HoVerAcquisitionError("normalized claim is empty")
    return normalized


def normalize_support_title_nfd(value: str) -> str:
    return unicodedata.normalize("NFD", _text(value, "support title", nonempty=True))


def _frame(raw: bytes) -> bytes:
    return len(raw).to_bytes(8, "big", signed=False) + raw


def hmac_digest(secret: bytes, purpose: str, *parts: str) -> bytes:
    if not isinstance(secret, bytes) or len(secret) != 32:
        raise HoVerAcquisitionError("private selection secret must be exactly 32 bytes")
    rows = [_text(purpose, "HMAC purpose", nonempty=True).encode("utf-8")]
    rows.extend(_text(part, "HMAC part").encode("utf-8") for part in parts)
    message = _frame(_HMAC_DOMAIN) + b"".join(_frame(row) for row in rows)
    return hmac.new(secret, message, hashlib.sha256).digest()


def strict_json_loads(raw: bytes, *, label: str) -> Any:
    if not isinstance(raw, bytes):
        raise HoVerAcquisitionError(f"{label} bytes are invalid")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise HoVerAcquisitionError(f"{label} contains a duplicate object key")
            output[key] = value
        return output

    def reject_constant(value: str) -> None:
        raise HoVerAcquisitionError(f"{label} contains non-finite {value}")

    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HoVerAcquisitionError(f"{label} is invalid strict JSON") from exc


def with_self_hash(body: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    if hash_field in body:
        raise HoVerAcquisitionError("self-hash field already exists")
    payload = dict(body)
    payload[hash_field] = stable_hash(body)
    return payload


def verify_self_hash(
    payload: Mapping[str, Any], *, hash_field: str, schema: str | None = None
) -> str:
    if not isinstance(payload, Mapping):
        raise HoVerAcquisitionError("self-hashed payload is not an object")
    if schema is not None and payload.get("schema") != schema:
        raise HoVerAcquisitionError("self-hashed payload schema drifted")
    declared = _require_sha256(payload.get(hash_field), hash_field)
    body = dict(payload)
    del body[hash_field]
    observed = stable_hash(body)
    if not hmac.compare_digest(declared, observed):
        raise HoVerAcquisitionError(f"{hash_field} self-hash mismatch")
    return observed


def validate_qualification_manifest(
    manifest: Mapping[str, Any], *, require_committed_identity: bool = True
) -> QualificationBinding:
    """Validate the aggregate-only committed qualification input."""

    if not isinstance(manifest, Mapping) or set(manifest) != QUALIFICATION_REQUIRED_KEYS:
        raise HoVerAcquisitionError("qualification manifest width drifted")
    digest = verify_self_hash(
        manifest, hash_field=QUALIFICATION_HASH_FIELD, schema=QUALIFICATION_SCHEMA
    )
    source = manifest.get("source_binding")
    structure = manifest.get("hop_and_structure")
    grouping = manifest.get("identity_and_grouping")
    capacity = manifest.get("capacity")
    sqlite = manifest.get("sqlite_and_gold_resolution")
    parser = manifest.get("parser_and_schema")
    boundary = manifest.get("claim_boundary")
    if (
        manifest.get("version") != "v1"
        or manifest.get("status") != QUALIFICATION_STATUS
        or not all(
            isinstance(value, Mapping)
            for value in (source, structure, grouping, capacity, sqlite, parser, boundary)
        )
    ):
        raise HoVerAcquisitionError("qualification manifest binding drifted")
    assert isinstance(source, Mapping)
    assert isinstance(structure, Mapping)
    assert isinstance(grouping, Mapping)
    assert isinstance(capacity, Mapping)
    assert isinstance(sqlite, Mapping)
    assert isinstance(parser, Mapping)
    assert isinstance(boundary, Mapping)
    if (
        source.get("design_sha256") != DESIGN_SHA256
        or source.get("training_sha256") != TRAINING_SHA256
        or source.get("corpus_sha256") != CORPUS_SOURCE_SHA256
        or source.get("formal_identity_enforced") is not True
        or parser.get("observed_row_count") != FORMAL_TRAIN_COUNT
        or structure.get("hop_counts") != {"2": 9052, "3": 6084, "4": 3035}
        or structure.get("gold_cardinality_counts") != {"2": 9052, "3": 6084, "4": 3035}
        or structure.get("eligible_item_count_after_claim_collision_exclusion") != 17_905
        or grouping.get("eligible_unique_hpqa_id_group_count") != 6_103
        or grouping.get("whole_collision_group_excluded_item_count") != 266
        or grouping.get("global_one_uid_per_hpqa_id_selection_required") is not True
        or capacity.get("exact_three_hop_b_matching_capacity_met") is not True
        or capacity.get("target_distinct_hpqa_groups_per_hop") != TOTAL_PER_HOP
        or capacity.get("fixed_corpus_article_count") != CORPUS_SIZE
        or sqlite.get("row_count") != 5_233_329
        or sqlite.get("maximum_rowid") != 5_233_329
        or sqlite.get("title_codec") != "Unicode_NFD_then_exact_documents_id_equality"
        or sqlite.get("fuzzy_casefold_substring_or_underscore_rewrite_used") is not False
        or boundary.get("selection_secret_or_cohort_created") is not False
        or boundary.get("retrieval_action_evaluator_or_score_run") is not False
    ):
        raise HoVerAcquisitionError("qualification aggregate contract drifted")
    if require_committed_identity and digest != QUALIFICATION_SHA256:
        raise HoVerAcquisitionError("qualification committed identity drifted")
    return QualificationBinding(
        qualification_sha256=digest,
        eligible_record_count=17_905,
        normalized_claim_collision_member_count=266,
        eligible_hpqa_group_count=6_103,
        sqlite_document_row_count=5_233_329,
        sqlite_maximum_rowid=5_233_329,
    )


def build_document_catalog(rows: Sequence[Mapping[str, Any] | Sequence[Any]]) -> DocumentCatalog:
    """Build a private exact-rowid catalog from synthetic/verified SQLite rows."""

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise HoVerAcquisitionError("documents rows are empty or malformed")
    by_rowid: dict[int, DocumentRow] = {}
    by_id: dict[str, list[int]] = defaultdict(list)
    for raw in rows:
        if isinstance(raw, Mapping):
            if set(raw) != {"rowid", "id", "text"}:
                raise HoVerAcquisitionError("documents row width drifted")
            rowid, exact_id, exact_text = raw["rowid"], raw["id"], raw["text"]
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) == 3:
            rowid, exact_id, exact_text = raw
        else:
            raise HoVerAcquisitionError("documents row is malformed")
        if type(rowid) is not int or rowid <= 0 or rowid in by_rowid:
            raise HoVerAcquisitionError("documents rowid is invalid or duplicated")
        exact_id = _text(exact_id, "documents.id", nonempty=True)
        exact_text = _text(exact_text, "documents.text", nonempty=True)
        identity = stable_hash(
            {
                "domain": f"{VERSION}/official-document-row/v1",
                "rowid": rowid,
                "id": exact_id,
                "text": exact_text,
            }
        )
        row = DocumentRow(rowid, exact_id, exact_text, identity)
        by_rowid[rowid] = row
        by_id[exact_id].append(rowid)
    catalog_body = [
        [rowid, by_rowid[rowid].identity_commitment_sha256] for rowid in sorted(by_rowid)
    ]
    return DocumentCatalog(
        by_rowid=by_rowid,
        rowids_by_exact_nfd_id={key: tuple(sorted(value)) for key, value in by_id.items()},
        maximum_rowid=max(by_rowid),
        catalog_sha256=stable_hash(catalog_body),
    )


def _parse_supporting_facts(value: object) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, list) or not value:
        raise HoVerAcquisitionError("supporting_facts must be a nonempty list")
    output: list[tuple[str, int]] = []
    for raw in value:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise HoVerAcquisitionError("supporting_facts entry width drifted")
        title = normalize_support_title_nfd(raw[0])
        sentence = raw[1]
        if type(sentence) is not int or sentence < 0:
            raise HoVerAcquisitionError("support sentence index is invalid")
        output.append((title, sentence))
    return tuple(output)


def parse_train_payload(
    payload: Any,
    *,
    documents: DocumentResolver | DocumentCatalog,
    enforce_formal_counts: bool = False,
) -> tuple[tuple[EligibleRecord, ...], dict[str, Any]]:
    """Strictly re-qualify TRAIN rows and resolve exact-NFD gold documents."""

    if not isinstance(payload, list):
        raise HoVerAcquisitionError("TRAIN root must be a list")
    if enforce_formal_counts and len(payload) != FORMAL_TRAIN_COUNT:
        raise HoVerAcquisitionError("formal TRAIN row count drifted")
    parsed: list[EligibleRecord] = []
    uid_groups: dict[str, list[int]] = defaultdict(list)
    claim_groups: dict[str, list[int]] = defaultdict(list)
    raw_hops: Counter[int] = Counter()
    exclusions: Counter[str] = Counter()
    extra_keysets: Counter[str] = Counter()
    source_keysets: Counter[str] = Counter()
    extra_field_names: set[str] = set()
    support_pair_count = 0
    resolved_gold_rowids: set[int] = set()
    resolved_title_lookup_count = 0

    for ordinal, raw in enumerate(payload):
        if not isinstance(raw, Mapping) or not _SOURCE_REQUIRED_KEYS.issubset(raw):
            raise HoVerAcquisitionError("TRAIN row required width drifted")
        extras = sorted(set(raw) - _SOURCE_REQUIRED_KEYS)
        extra_keysets[stable_hash(extras)] += 1
        source_keysets[stable_hash(sorted(raw))] += 1
        extra_field_names.update(extras)
        uid = _text(raw.get("uid"), "uid", nonempty=True)
        hpqa_id = _text(raw.get("hpqa_id"), "hpqa_id", nonempty=True)
        claim = _text(raw.get("claim"), "claim", nonempty=True)
        num_hops = raw.get("num_hops")
        if type(num_hops) is not int:
            raise HoVerAcquisitionError("num_hops physical type drifted")
        raw_hops[num_hops] += 1
        supports = _parse_supporting_facts(raw.get("supporting_facts"))
        support_pair_count += len(supports)
        uid_sha = _sha256_text(uid)
        normalized = normalize_claim(claim)
        uid_groups[uid_sha].append(ordinal)
        claim_groups[normalized].append(ordinal)
        distinct_titles = tuple(sorted({title for title, _sentence in supports}))
        if num_hops not in (2, 3, 4):
            exclusions["num_hops_outside_2_3_4"] += 1
            continue
        if len(distinct_titles) != num_hops:
            exclusions["distinct_support_title_count_not_num_hops"] += 1
            continue
        gold_rowids: list[int] = []
        unresolved = False
        for title in distinct_titles:
            matches = documents.resolve_exact_nfd_id(title)
            if len(matches) != 1:
                unresolved = True
                break
            gold_rowids.append(matches[0].rowid)
            resolved_gold_rowids.add(matches[0].rowid)
            resolved_title_lookup_count += 1
        if unresolved:
            exclusions["support_title_not_exactly_one_SQLite_row"] += 1
            continue
        identity = stable_hash(
            {
                "domain": f"{VERSION}/candidate-identity/v1",
                "uid_sha256": uid_sha,
                "hpqa_id_sha256": _sha256_text(hpqa_id),
                "normalized_claim_sha256": _sha256_text(normalized),
                "hop_stratum": f"{num_hops}_hop",
                "gold_document_rowids": sorted(gold_rowids),
            }
        )
        parsed.append(
            EligibleRecord(
                source_row_ordinal=ordinal,
                uid_sha256=uid_sha,
                hpqa_id_sha256=_sha256_text(hpqa_id),
                claim=claim,
                normalized_claim=normalized,
                hop_stratum=f"{num_hops}_hop",
                gold_document_rowids=tuple(sorted(gold_rowids)),
                identity_commitment_sha256=identity,
                source_record_commitment_sha256=stable_hash(
                    {
                        "domain": f"{VERSION}/complete-source-record/v1",
                        "candidate_identity_sha256": identity,
                        "canonical_source_row_sha256": stable_hash(raw),
                    }
                ),
            )
        )

    if enforce_formal_counts and raw_hops != Counter(FORMAL_HOP_COUNTS):
        raise HoVerAcquisitionError("formal TRAIN hop counts drifted")
    duplicate_uids = {uid for uid, ordinals in uid_groups.items() if len(ordinals) > 1}
    collision_claims = {
        claim for claim, ordinals in claim_groups.items() if len(ordinals) > 1
    }
    eligible = tuple(
        row
        for row in parsed
        if row.uid_sha256 not in duplicate_uids
        and row.normalized_claim not in collision_claims
    )
    eligible_hpqa = {row.hpqa_id_sha256 for row in eligible}
    stats = {
        "training_row_count": len(payload),
        "raw_hop_counts": {str(key): raw_hops[key] for key in sorted(raw_hops)},
        "structural_exclusion_counts": {key: exclusions[key] for key in sorted(exclusions)},
        "duplicate_uid_group_count": len(duplicate_uids),
        "normalized_claim_collision_group_count": len(collision_claims),
        "normalized_claim_collision_member_count": sum(
            len(claim_groups[claim]) for claim in collision_claims
        ),
        "eligible_record_count": len(eligible),
        "eligible_hpqa_group_count": len(eligible_hpqa),
        "eligible_hop_membership_counts": dict(Counter(row.hop_stratum for row in eligible)),
        "extra_keyset_hash_histogram": dict(sorted(extra_keysets.items())),
        "source_keyset_hash_histogram": dict(sorted(source_keysets.items())),
        "extra_field_name_count": len(extra_field_names),
        "extra_field_name_set_sha256": stable_hash(sorted(extra_field_names)),
        "support_pair_count": support_pair_count,
        "resolved_gold_title_lookup_count": resolved_title_lookup_count,
        "distinct_resolved_gold_rowid_count": len(resolved_gold_rowids),
        "document_catalog_sha256": documents.binding_sha256,
        "item_uid_claim_title_body_or_support_text_emitted": False,
    }
    return eligible, stats


@dataclass(frozen=True)
class _HopSlot:
    hop_stratum: str
    ordinal: int

    @property
    def slot_id(self) -> str:
        return f"{self.hop_stratum}/{self.ordinal:04d}"


def _hop_slots() -> tuple[_HopSlot, ...]:
    return tuple(
        _HopSlot(hop_stratum, ordinal)
        for hop_stratum in HOP_STRATA
        for ordinal in range(TOTAL_PER_HOP)
    )


def _maximum_group_hop_matching(
    records: Sequence[EligibleRecord], *, secret: bytes
) -> tuple[dict[str, str], dict[tuple[str, str], tuple[EligibleRecord, ...]]]:
    """Match unique hpqa groups to exact hop slots without choosing labels."""

    grouped: dict[tuple[str, str], list[EligibleRecord]] = defaultdict(list)
    group_hops: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if not isinstance(record, EligibleRecord) or record.hop_stratum not in HOP_STRATA:
            raise HoVerAcquisitionError("eligible record type or hop stratum drifted")
        grouped[(record.hpqa_id_sha256, record.hop_stratum)].append(record)
        group_hops[record.hpqa_id_sha256].add(record.hop_stratum)
    slots = _hop_slots()
    slot_by_id = {slot.slot_id: slot for slot in slots}
    group_order = sorted(
        group_hops,
        key=lambda group: (hmac_digest(secret, "matching_group_order", group), group),
    )
    adjacency: dict[str, tuple[str, ...]] = {}
    for group in group_order:
        allowed = [
            slot_id
            for slot_id, slot in slot_by_id.items()
            if slot.hop_stratum in group_hops[group]
        ]
        adjacency[group] = tuple(
            sorted(
                allowed,
                key=lambda slot_id: (
                    hmac_digest(secret, "matching_edge_order", group, slot_id),
                    slot_id,
                ),
            )
        )
    owner_by_slot: dict[str, str] = {}

    def augment(group: str, seen_slots: set[str]) -> bool:
        for slot_id in adjacency[group]:
            if slot_id in seen_slots:
                continue
            seen_slots.add(slot_id)
            owner = owner_by_slot.get(slot_id)
            if owner is None or augment(owner, seen_slots):
                owner_by_slot[slot_id] = group
                return True
        return False

    for group in group_order:
        augment(group, set())
    frozen_grouped = {
        key: tuple(
            sorted(
                value,
                key=lambda row: (
                    hmac_digest(
                        secret,
                        "within_group_hop_candidate_order",
                        key[0],
                        key[1],
                        row.identity_commitment_sha256,
                    ),
                    row.identity_commitment_sha256,
                ),
            )
        )
        for key, value in grouped.items()
    }
    return owner_by_slot, frozen_grouped


def synthetic_qualification_binding(
    *, source_stats: Mapping[str, Any], documents: DocumentCatalog
) -> QualificationBinding:
    """Construct a nonformal binding for synthetic-only tests."""

    return QualificationBinding(
        qualification_sha256=stable_hash(
            {
                "domain": f"{VERSION}/synthetic-qualification/v1",
                "source_stats": dict(source_stats),
                "document_catalog_sha256": documents.catalog_sha256,
            }
        ),
        eligible_record_count=int(source_stats["eligible_record_count"]),
        normalized_claim_collision_member_count=int(
            source_stats["normalized_claim_collision_member_count"]
        ),
        eligible_hpqa_group_count=int(source_stats["eligible_hpqa_group_count"]),
        sqlite_document_row_count=documents.row_count,
        sqlite_maximum_rowid=documents.maximum_rowid,
    )


def select_private_blocks(
    records: Sequence[EligibleRecord],
    *,
    source_stats: Mapping[str, Any],
    qualification: QualificationBinding,
    secret: bytes,
) -> tuple[dict[str, tuple[AssignedRecord, ...]], dict[str, Any]]:
    """Run the single HMAC b-matching and assign its 144 records to four blocks."""

    if not isinstance(qualification, QualificationBinding):
        raise HoVerAcquisitionError("qualification binding is absent")
    rows = tuple(records)
    if (
        len(rows) != qualification.eligible_record_count
        or source_stats.get("eligible_record_count") != len(rows)
        or source_stats.get("normalized_claim_collision_member_count")
        != qualification.normalized_claim_collision_member_count
        or source_stats.get("eligible_hpqa_group_count")
        != qualification.eligible_hpqa_group_count
    ):
        raise HoVerAcquisitionError("private source requalification disagrees with manifest")
    owner_by_slot, grouped = _maximum_group_hop_matching(rows, secret=secret)
    slots = _hop_slots()
    if len(owner_by_slot) != len(slots):
        raise HoVerAcquisitionError("exact three-hop b-matching capacity is insufficient")
    chosen_by_hop: dict[str, list[EligibleRecord]] = {hop: [] for hop in HOP_STRATA}
    for slot in slots:
        group = owner_by_slot.get(slot.slot_id)
        if group is None:
            raise HoVerAcquisitionError("maximum matching omitted a required hop slot")
        candidates = grouped.get((group, slot.hop_stratum), ())
        if not candidates:
            raise HoVerAcquisitionError("matched group lacks its hop candidate")
        chosen_by_hop[slot.hop_stratum].append(candidates[0])
    for hop in HOP_STRATA:
        chosen_by_hop[hop].sort(
            key=lambda row: (
                hmac_digest(
                    secret,
                    "selected_hop_assignment_order",
                    hop,
                    row.identity_commitment_sha256,
                ),
                row.identity_commitment_sha256,
            )
        )
        if len(chosen_by_hop[hop]) != TOTAL_PER_HOP:
            raise HoVerAcquisitionError("selected hop count drifted")

    blocks: dict[str, list[AssignedRecord]] = {block: [] for block in BLOCK_ORDER}
    cursor_by_hop = {hop: 0 for hop in HOP_STRATA}
    for block in BLOCK_ORDER:
        for hop in HOP_STRATA:
            start = cursor_by_hop[hop]
            end = start + HOP_QUOTAS[block]
            for slot_ordinal, record in enumerate(chosen_by_hop[hop][start:end]):
                blocks[block].append(
                    AssignedRecord(record, block, hop, slot_ordinal)
                )
            cursor_by_hop[hop] = end
        blocks[block].sort(
            key=lambda assigned: (
                hmac_digest(
                    secret,
                    "block_unified_presentation_order",
                    block,
                    assigned.record.identity_commitment_sha256,
                ),
                assigned.record.identity_commitment_sha256,
            )
        )
    flattened = [assigned for block in BLOCK_ORDER for assigned in blocks[block]]
    if (
        len(flattened) != TOTAL_SELECTED
        or len({row.record.uid_sha256 for row in flattened}) != TOTAL_SELECTED
        or len({row.record.hpqa_id_sha256 for row in flattened}) != TOTAL_SELECTED
        or len({row.record.normalized_claim for row in flattened}) != TOTAL_SELECTED
    ):
        raise HoVerAcquisitionError("selected cohort global disjointness drifted")
    counts = {
        block: dict(Counter(row.hop_stratum for row in blocks[block]))
        for block in BLOCK_ORDER
    }
    for block in BLOCK_ORDER:
        if len(blocks[block]) != BLOCK_COUNTS[block] or counts[block] != {
            hop: HOP_QUOTAS[block] for hop in HOP_STRATA
        }:
            raise HoVerAcquisitionError("selected block hop quotas drifted")
    stats = {
        "qualification_sha256": qualification.qualification_sha256,
        "maximum_b_matching_cardinality": len(owner_by_slot),
        "required_b_matching_cardinality": TOTAL_SELECTED,
        "selected_block_counts": {block: len(blocks[block]) for block in BLOCK_ORDER},
        "selected_hop_stratum_counts": counts,
        "selected_unique_uid_count": len({row.record.uid_sha256 for row in flattened}),
        "selected_unique_hpqa_group_count": len(
            {row.record.hpqa_id_sha256 for row in flattened}
        ),
        "selected_unique_normalized_claim_count": len(
            {row.record.normalized_claim for row in flattened}
        ),
        "selection_contract": {
            "whole_normalized_claim_collision_groups_excluded": True,
            "hpqa_id_global_at_most_one": True,
            "private_HMAC_b_matching": True,
            "private_HMAC_hop_assignment": True,
            "private_HMAC_block_unified_order": True,
        },
    }
    return {block: tuple(blocks[block]) for block in BLOCK_ORDER}, stats


def build_fixed_corpus(
    *,
    blocks: Mapping[str, Sequence[AssignedRecord]],
    documents: DocumentResolver | DocumentCatalog,
    qualification: QualificationBinding,
    secret: bytes,
    attempt_cap: int = DISTRACTOR_ATTEMPT_CAP,
) -> tuple[tuple[DocumentRow, ...], dict[int, int], dict[str, Any]]:
    """Form gold union plus counter-to-rowid rejection fillers, then blind order."""

    if set(blocks) != set(BLOCK_ORDER):
        raise HoVerAcquisitionError("block set drifted during corpus formation")
    if (
        documents.row_count != qualification.sqlite_document_row_count
        or documents.maximum_rowid != qualification.sqlite_maximum_rowid
    ):
        raise HoVerAcquisitionError("SQLite aggregate differs from qualification")
    selected = [assigned for block in BLOCK_ORDER for assigned in blocks[block]]
    if len(selected) != TOTAL_SELECTED:
        raise HoVerAcquisitionError("corpus formation cohort count drifted")
    gold_rowids = {
        rowid
        for assigned in selected
        for rowid in assigned.record.gold_document_rowids
    }
    if not gold_rowids or len(gold_rowids) > 432:
        raise HoVerAcquisitionError("selected gold union violates frozen capacity proof")
    chosen = set(gold_rowids)
    documents_by_rowid: dict[int, DocumentRow] = {}
    serialized_documents: set[str] = set()
    for rowid in sorted(gold_rowids):
        row = documents.fetch_rowid(rowid)
        if row is None:
            raise HoVerAcquisitionError("selected gold rowid disappeared from SQLite")
        serialized = row.exact_id + "\n\n" + row.exact_text
        if serialized in serialized_documents:
            raise HoVerAcquisitionError("selected gold document serialization overlaps")
        serialized_documents.add(serialized)
        documents_by_rowid[rowid] = row
    counter = 0
    rejected_missing = 0
    rejected_duplicate = 0
    rejected_serialization = 0
    while len(chosen) < CORPUS_SIZE and counter < attempt_cap:
        digest = hmac_digest(secret, "distractor_counter_to_rowid", str(counter))
        counter += 1
        rowid = 1 + int.from_bytes(digest[:8], "big") % documents.maximum_rowid
        if rowid in chosen:
            rejected_duplicate += 1
            continue
        row = documents.fetch_rowid(rowid)
        if row is None:
            rejected_missing += 1
            continue
        serialized = row.exact_id + "\n\n" + row.exact_text
        if serialized in serialized_documents:
            rejected_serialization += 1
            continue
        chosen.add(rowid)
        serialized_documents.add(serialized)
        documents_by_rowid[rowid] = row
    if len(chosen) != CORPUS_SIZE:
        raise HoVerAcquisitionError("distractor rejection sampler exhausted its cap")
    ordered_rowids = tuple(
        sorted(
            chosen,
            key=lambda rowid: (
                hmac_digest(
                    secret,
                    "corpus_independent_unified_order",
                    str(rowid),
                    documents_by_rowid[rowid].identity_commitment_sha256,
                ),
                rowid,
            ),
        )
    )
    article_id_by_rowid = {rowid: index for index, rowid in enumerate(ordered_rowids)}
    corpus_rows = tuple(documents_by_rowid[rowid] for rowid in ordered_rowids)
    return corpus_rows, article_id_by_rowid, {
        "fixed_article_count": len(corpus_rows),
        "unique_selected_gold_article_count": len(gold_rowids),
        "filler_article_count": CORPUS_SIZE - len(gold_rowids),
        "distractor_counter_attempt_count": counter,
        "distractor_rejected_missing_rowid_count": rejected_missing,
        "distractor_rejected_duplicate_or_gold_count": rejected_duplicate,
        "distractor_rejected_duplicate_serialization_count": rejected_serialization,
        "all_selected_gold_included": gold_rowids.issubset(article_id_by_rowid),
        "origin_or_is_gold_in_corpus_view": False,
        "shared_all_blocks_and_methods": True,
        "corpus_order_independent_private_HMAC": True,
    }


_CORPUS_ARTICLE_KEYS = {"article_id", "title", "body"}
_VIEW_ITEM_KEYS = {"schema", "block", "ordinal", "claim"}
_LABEL_ITEM_KEYS = {
    "schema",
    "block",
    "ordinal",
    "view_sha256",
    "identity_commitment_sha256",
    "source_record_commitment_sha256",
    "hop_stratum",
    "gold_article_ids",
}


def validate_corpus_view(payload: Mapping[str, Any]) -> None:
    if set(payload) != {
        "schema",
        "version",
        "article_count",
        "origin_or_gold_membership_included",
        "articles",
        "corpus_view_sha256",
    }:
        raise HoVerAcquisitionError("corpus view envelope drifted")
    verify_self_hash(payload, hash_field="corpus_view_sha256", schema=CORPUS_VIEW_SCHEMA)
    articles = payload.get("articles")
    if (
        payload.get("version") != VERSION
        or payload.get("article_count") != CORPUS_SIZE
        or payload.get("origin_or_gold_membership_included") is not False
        or not isinstance(articles, list)
        or len(articles) != CORPUS_SIZE
    ):
        raise HoVerAcquisitionError("corpus view identity drifted")
    for article_id, row in enumerate(articles):
        if (
            not isinstance(row, Mapping)
            or set(row) != _CORPUS_ARTICLE_KEYS
            or row.get("article_id") != article_id
        ):
            raise HoVerAcquisitionError("corpus article row drifted")
        _text(row.get("title"), "corpus title", nonempty=True)
        _text(row.get("body"), "corpus body", nonempty=True)
    if len({(row["title"], row["body"]) for row in articles}) != CORPUS_SIZE:
        raise HoVerAcquisitionError("corpus title/body rows are not unique")


def validate_block_view(payload: Mapping[str, Any], *, expected_block: str) -> None:
    if set(payload) != {
        "schema",
        "version",
        "block",
        "item_count",
        "late_utility_fields_included",
        "items",
        "block_view_sha256",
    }:
        raise HoVerAcquisitionError("block view envelope drifted")
    verify_self_hash(payload, hash_field="block_view_sha256", schema=BLOCK_VIEW_SCHEMA)
    items = payload.get("items")
    if (
        expected_block not in BLOCK_ORDER
        or payload.get("version") != VERSION
        or payload.get("block") != expected_block
        or payload.get("late_utility_fields_included") is not False
        or not isinstance(items, list)
        or len(items) != BLOCK_COUNTS[expected_block]
        or payload.get("item_count") != len(items)
    ):
        raise HoVerAcquisitionError("block view identity drifted")
    hashes: set[str] = set()
    for ordinal, item in enumerate(items):
        if (
            not isinstance(item, Mapping)
            or set(item) != _VIEW_ITEM_KEYS
            or item.get("schema") != VIEW_ITEM_SCHEMA
            or item.get("block") != expected_block
            or item.get("ordinal") != ordinal
        ):
            raise HoVerAcquisitionError("claim view item drifted")
        _text(item.get("claim"), "claim view", nonempty=True)
        hashes.add(stable_hash(item))
    if len(hashes) != len(items):
        raise HoVerAcquisitionError("claim view item hashes overlap")


def validate_block_labels(payload: Mapping[str, Any], *, expected_block: str) -> None:
    if expected_block == "F_search":
        raise HoVerAcquisitionError("F_search utility label pack must not exist")
    if set(payload) != {
        "schema",
        "version",
        "block",
        "item_count",
        "source_or_verdict_payload_included",
        "items",
        "block_labels_sha256",
    }:
        raise HoVerAcquisitionError("utility label envelope drifted")
    verify_self_hash(payload, hash_field="block_labels_sha256", schema=BLOCK_LABEL_SCHEMA)
    items = payload.get("items")
    if (
        expected_block not in BLOCK_ORDER
        or payload.get("version") != VERSION
        or payload.get("block") != expected_block
        or payload.get("source_or_verdict_payload_included") is not False
        or not isinstance(items, list)
        or len(items) != BLOCK_COUNTS[expected_block]
        or payload.get("item_count") != len(items)
    ):
        raise HoVerAcquisitionError("utility label identity drifted")
    hop_counts: Counter[str] = Counter()
    identities: set[str] = set()
    source_records: set[str] = set()
    view_hashes: set[str] = set()
    for ordinal, item in enumerate(items):
        gold = item.get("gold_article_ids") if isinstance(item, Mapping) else None
        hop = item.get("hop_stratum") if isinstance(item, Mapping) else None
        if (
            not isinstance(item, Mapping)
            or set(item) != _LABEL_ITEM_KEYS
            or item.get("schema") != LABEL_ITEM_SCHEMA
            or item.get("block") != expected_block
            or item.get("ordinal") != ordinal
            or hop not in HOP_STRATA
            or not isinstance(gold, list)
            or len(gold) != int(str(hop)[0])
            or gold != sorted(set(gold))
            or any(type(value) is not int or not 0 <= value < CORPUS_SIZE for value in gold)
        ):
            raise HoVerAcquisitionError("utility label item drifted")
        identities.add(_require_sha256(item.get("identity_commitment_sha256"), "identity"))
        source_records.add(
            _require_sha256(item.get("source_record_commitment_sha256"), "source record")
        )
        view_hashes.add(_require_sha256(item.get("view_sha256"), "view item"))
        hop_counts[str(hop)] += 1
    if (
        len(identities) != len(items)
        or len(source_records) != len(items)
        or len(view_hashes) != len(items)
        or hop_counts != Counter({hop: HOP_QUOTAS[expected_block] for hop in HOP_STRATA})
    ):
        raise HoVerAcquisitionError("utility label uniqueness or quota drifted")


def materialize_private_payloads(
    *,
    blocks: Mapping[str, Sequence[AssignedRecord]],
    corpus_rows: Sequence[DocumentRow],
    article_id_by_rowid: Mapping[int, int],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Build the label-free corpus/views and separate late utility mappings."""

    if set(blocks) != set(BLOCK_ORDER) or len(corpus_rows) != CORPUS_SIZE:
        raise HoVerAcquisitionError("materialization input shape drifted")
    corpus = with_self_hash(
        {
            "schema": CORPUS_VIEW_SCHEMA,
            "version": VERSION,
            "article_count": CORPUS_SIZE,
            "origin_or_gold_membership_included": False,
            "articles": [row.view_row(article_id) for article_id, row in enumerate(corpus_rows)],
        },
        "corpus_view_sha256",
    )
    validate_corpus_view(corpus)
    views: dict[str, dict[str, Any]] = {}
    labels: dict[str, dict[str, Any]] = {}
    gold_histograms: dict[str, dict[str, int]] = {}
    for block in BLOCK_ORDER:
        assigned_rows = tuple(blocks[block])
        if len(assigned_rows) != BLOCK_COUNTS[block]:
            raise HoVerAcquisitionError("materialized block count drifted")
        view_items = [
            {
                "schema": VIEW_ITEM_SCHEMA,
                "block": block,
                "ordinal": ordinal,
                "claim": assigned.record.claim,
            }
            for ordinal, assigned in enumerate(assigned_rows)
        ]
        view = with_self_hash(
            {
                "schema": BLOCK_VIEW_SCHEMA,
                "version": VERSION,
                "block": block,
                "item_count": len(view_items),
                "late_utility_fields_included": False,
                "items": view_items,
            },
            "block_view_sha256",
        )
        validate_block_view(view, expected_block=block)
        views[block] = view
        histogram = Counter(len(row.record.gold_document_rowids) for row in assigned_rows)
        gold_histograms[block] = {str(key): histogram[key] for key in sorted(histogram)}
        if block == "F_search":
            continue
        label_items = []
        for ordinal, assigned in enumerate(assigned_rows):
            try:
                gold_article_ids = sorted(
                    article_id_by_rowid[rowid]
                    for rowid in assigned.record.gold_document_rowids
                )
            except KeyError as exc:
                raise HoVerAcquisitionError("selected gold is absent from fixed corpus") from exc
            label_items.append(
                {
                    "schema": LABEL_ITEM_SCHEMA,
                    "block": block,
                    "ordinal": ordinal,
                    "view_sha256": stable_hash(view_items[ordinal]),
                    "identity_commitment_sha256": assigned.record.identity_commitment_sha256,
                    "source_record_commitment_sha256": assigned.record.source_record_commitment_sha256,
                    "hop_stratum": assigned.hop_stratum,
                    "gold_article_ids": gold_article_ids,
                }
            )
        label = with_self_hash(
            {
                "schema": BLOCK_LABEL_SCHEMA,
                "version": VERSION,
                "block": block,
                "item_count": len(label_items),
                "source_or_verdict_payload_included": False,
                "items": label_items,
            },
            "block_labels_sha256",
        )
        validate_block_labels(label, expected_block=block)
        labels[block] = label
    if set(labels) != {"A_form", "A_hold", "M_search"}:
        raise HoVerAcquisitionError("late utility label pack set drifted")
    return corpus, views, labels, {
        "fixed_article_count": CORPUS_SIZE,
        "block_gold_cardinality_histograms": gold_histograms,
        "F_search_utility_label_pack_created": False,
        "claim_verdict_support_sentences_or_hop_in_view": False,
        "origin_or_is_gold_in_corpus_view": False,
    }


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_project(project: Path) -> Path:
    path = Path(project).absolute()
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise HoVerAcquisitionError("project root is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise HoVerAcquisitionError("project root must be a real directory")
    return path


def default_acquisition_paths(project: Path) -> AcquisitionPaths:
    root = _canonical_project(project)
    return AcquisitionPaths(
        marker=root / FORMAL_MARKER_RELATIVE,
        secret=root / FORMAL_SECRET_RELATIVE,
        corpus_view=root / CORPUS_VIEW_RELATIVE,
        block_views={block: root / path for block, path in BLOCK_VIEW_RELATIVES.items()},
        block_labels={block: root / path for block, path in BLOCK_LABEL_RELATIVES.items()},
        public_receipt=root / PUBLIC_RECEIPT_RELATIVE,
    )


def _all_output_paths(paths: AcquisitionPaths) -> tuple[Path, ...]:
    if set(paths.block_views) != set(BLOCK_ORDER) or set(paths.block_labels) != {
        "A_form",
        "A_hold",
        "M_search",
    }:
        raise HoVerAcquisitionError("acquisition output path set drifted")
    return (
        paths.marker,
        paths.secret,
        paths.corpus_view,
        *(paths.block_views[block] for block in BLOCK_ORDER),
        *(paths.block_labels[block] for block in ("A_form", "A_hold", "M_search")),
        paths.public_receipt,
    )


def _reject_symlink_ancestors(path: Path) -> None:
    absolute = path.absolute()
    for ancestor in (absolute.parent, *absolute.parent.parents):
        try:
            metadata = ancestor.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise HoVerAcquisitionError("output ancestor is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise HoVerAcquisitionError("output path has a symlink ancestor")


def _ensure_directory_durable(path: Path) -> None:
    """Create each missing directory and fsync its parent entry."""

    directory = path.absolute()
    missing: list[Path] = []
    cursor = directory
    while not os.path.lexists(cursor):
        missing.append(cursor)
        if cursor.parent == cursor:
            raise HoVerAcquisitionError("durable output ancestor is unavailable")
        cursor = cursor.parent
    try:
        metadata = cursor.lstat()
    except OSError as exc:
        raise HoVerAcquisitionError("durable output ancestor is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise HoVerAcquisitionError("durable output ancestor is unsafe")
    for directory_to_create in reversed(missing):
        try:
            directory_to_create.mkdir(mode=0o700)
            parent_descriptor = os.open(
                directory_to_create.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
        except OSError as exc:
            raise HoVerAcquisitionError(
                "durable output parent creation failed"
            ) from exc
        metadata = directory_to_create.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise HoVerAcquisitionError("durable output parent is unsafe")
    cursor = directory
    while cursor.parent != cursor:
        try:
            descriptor = os.open(
                cursor.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise HoVerAcquisitionError(
                "durable output ancestor fsync failed"
            ) from exc
        cursor = cursor.parent


def _prepare_output_parents(paths: AcquisitionPaths) -> None:
    for path in _all_output_paths(paths):
        _reject_symlink_ancestors(path)
        _ensure_directory_durable(path.parent)
        _reject_symlink_ancestors(path)
        metadata = path.parent.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise HoVerAcquisitionError("output parent is not a directory")


def _preflight_outputs(paths: AcquisitionPaths) -> None:
    targets = _all_output_paths(paths)
    if len({path.absolute() for path in targets}) != len(targets):
        raise HoVerAcquisitionError("acquisition output paths overlap")
    for path in targets:
        if path.exists() or path.is_symlink():
            raise HoVerAcquisitionError("acquisition output already exists; replay forbidden")


def _write_exclusive(path: Path, raw: bytes, *, mode: int) -> tuple[str, int]:
    if not isinstance(raw, bytes) or not raw:
        raise HoVerAcquisitionError("exclusive output bytes are empty")
    _reject_symlink_ancestors(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as exc:
        raise HoVerAcquisitionError("exclusive output creation failed") from exc
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            metadata = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != mode
            ):
                raise HoVerAcquisitionError("exclusive output mode drifted")
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    parent_descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return _sha256_bytes(raw), len(raw)


def _write_json_exclusive(
    path: Path, payload: Mapping[str, Any], *, mode: int
) -> tuple[str, int]:
    return _write_exclusive(path, canonical_json_bytes(payload) + b"\n", mode=mode)


def _require_regular_file(path: Path, *, label: str, mode: int | None = None) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise HoVerAcquisitionError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise HoVerAcquisitionError(f"{label} must be a real regular file")
    if mode is not None and stat.S_IMODE(metadata.st_mode) != mode:
        raise HoVerAcquisitionError(f"{label} mode drifted")
    return metadata


def _stream_file_sha256(path: Path, *, label: str, mode: int = 0o600) -> tuple[str, int]:
    metadata = _require_regular_file(path, label=label, mode=mode)
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            observed = os.fstat(handle.fileno())
    except OSError as exc:
        raise HoVerAcquisitionError(f"{label} hash read failed") from exc
    if (metadata.st_dev, metadata.st_ino, metadata.st_size) != (
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
    ):
        raise HoVerAcquisitionError(f"{label} identity changed while hashing")
    return digest.hexdigest(), metadata.st_size


def _read_bound_bytes(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    mode: int | None = None,
) -> bytes:
    metadata = _require_regular_file(path, label=label, mode=mode)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise HoVerAcquisitionError(f"{label} read failed") from exc
    if len(raw) != metadata.st_size:
        raise HoVerAcquisitionError(f"{label} changed while reading")
    if expected_size is not None and len(raw) != expected_size:
        raise HoVerAcquisitionError(f"{label} size binding drifted")
    if expected_sha256 is not None and not hmac.compare_digest(
        _sha256_bytes(raw), _require_sha256(expected_sha256, f"{label} binding")
    ):
        raise HoVerAcquisitionError(f"{label} SHA-256 binding drifted")
    return raw


def _read_bound_json(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    mode: int | None = None,
) -> tuple[Any, bytes]:
    raw = _read_bound_bytes(
        path,
        label=label,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
        mode=mode,
    )
    return strict_json_loads(raw, label=label), raw


def _file_binding(
    *, path: Path, file_sha256: str, byte_size: int, semantic_sha256: str | None, mode: int
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "relative_name": path.name,
        "file_sha256": _require_sha256(file_sha256, "file binding"),
        "byte_size": byte_size,
        "mode": f"{mode:04o}",
    }
    if semantic_sha256 is not None:
        payload["semantic_sha256"] = _require_sha256(
            semantic_sha256, "semantic binding"
        )
    return payload


def _assert_public_safe(payload: Mapping[str, Any]) -> None:
    forbidden = {
        "uid",
        "hpqa_id",
        "claim",
        "title",
        "body",
        "supporting_facts",
        "gold_article_ids",
        "items",
        "articles",
        "source_row_ordinal",
        "identity_commitment_sha256",
        "source_record_commitment_sha256",
    }

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if key in forbidden:
                    raise HoVerAcquisitionError(f"public receipt contains private field {key}")
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(payload)


def _formal_source_stats_match(
    stats: Mapping[str, Any], qualification: QualificationBinding
) -> None:
    if (
        stats.get("training_row_count") != FORMAL_TRAIN_COUNT
        or stats.get("raw_hop_counts") != {"2": 9052, "3": 6084, "4": 3035}
        or stats.get("structural_exclusion_counts") != {}
        or stats.get("duplicate_uid_group_count") != 0
        or stats.get("normalized_claim_collision_group_count") != 132
        or stats.get("normalized_claim_collision_member_count")
        != qualification.normalized_claim_collision_member_count
        or stats.get("eligible_record_count") != qualification.eligible_record_count
        or stats.get("eligible_hpqa_group_count") != qualification.eligible_hpqa_group_count
        or stats.get("source_keyset_hash_histogram")
        != {"d844b00049ef3f3668ef59c6866619f1addd64e6e70f20ceb89227d01b09d03e": 18171}
        or stats.get("extra_field_name_count") != 1
        or stats.get("extra_field_name_set_sha256")
        != "d9ff09800e17936c2de8910ec5dabeb73f634725585275a8b17ca508f0675c21"
        or stats.get("support_pair_count") != 55_816
        or stats.get("resolved_gold_title_lookup_count") != 48_496
        or stats.get("distinct_resolved_gold_rowid_count") != 14_507
        or stats.get("document_catalog_sha256") != CORPUS_SOURCE_SHA256
    ):
        raise HoVerAcquisitionError("formal source requalification differs from committed aggregates")


def _consume_marker(
    *, paths: AcquisitionPaths, qualification: QualificationBinding, source_bindings: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    marker = with_self_hash(
        {
            "schema": ONE_SHOT_MARKER_SCHEMA,
            "version": VERSION,
            "status": "one_shot_attempt_consumed_before_secret_or_selection",
            "design_sha256": DESIGN_SHA256,
            "qualification_sha256": qualification.qualification_sha256,
            "source_bindings_sha256": stable_hash(source_bindings),
            "same_source_replay_secret_rotation_resample_or_replacement_authorized": False,
        },
        "marker_sha256",
    )
    file_sha, size = _write_json_exclusive(paths.marker, marker, mode=0o600)
    return marker, _file_binding(
        path=paths.marker,
        file_sha256=file_sha,
        byte_size=size,
        semantic_sha256=str(marker["marker_sha256"]),
        mode=0o600,
    )


def create_one_shot_secret_once(
    *, path: Path, random_bytes: Callable[[int], bytes] = os.urandom
) -> tuple[bytes, dict[str, Any]]:
    try:
        secret = random_bytes(32)
    except BaseException as exc:
        raise HoVerAcquisitionError("private selection secret generation failed") from exc
    if not isinstance(secret, bytes) or len(secret) != 32:
        raise HoVerAcquisitionError("private selection secret generator returned invalid bytes")
    file_sha, size = _write_exclusive(path, secret, mode=0o600)
    return secret, _file_binding(
        path=path,
        file_sha256=file_sha,
        byte_size=size,
        semantic_sha256=None,
        mode=0o600,
    )


def persist_private_payloads(
    *,
    corpus: Mapping[str, Any],
    views: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    paths: AcquisitionPaths,
) -> dict[str, Any]:
    if set(views) != set(BLOCK_ORDER) or set(labels) != {"A_form", "A_hold", "M_search"}:
        raise HoVerAcquisitionError("private payload set drifted")
    validate_corpus_view(corpus)
    corpus_file_sha, corpus_size = _write_json_exclusive(
        paths.corpus_view, corpus, mode=0o600
    )
    block_bindings: dict[str, Any] = {}
    for block in BLOCK_ORDER:
        validate_block_view(views[block], expected_block=block)
        view_sha, view_size = _write_json_exclusive(
            paths.block_views[block], views[block], mode=0o600
        )
        block_binding: dict[str, Any] = {
            "item_count": BLOCK_COUNTS[block],
            "view": _file_binding(
                path=paths.block_views[block],
                file_sha256=view_sha,
                byte_size=view_size,
                semantic_sha256=str(views[block]["block_view_sha256"]),
                mode=0o600,
            ),
            "labels": {"created": False},
        }
        if block != "F_search":
            validate_block_labels(labels[block], expected_block=block)
            label_sha, label_size = _write_json_exclusive(
                paths.block_labels[block], labels[block], mode=0o600
            )
            block_binding["labels"] = {
                "created": True,
                **_file_binding(
                    path=paths.block_labels[block],
                    file_sha256=label_sha,
                    byte_size=label_size,
                    semantic_sha256=str(labels[block]["block_labels_sha256"]),
                    mode=0o600,
                ),
            }
        block_bindings[block] = block_binding
    return {
        "corpus_view": _file_binding(
            path=paths.corpus_view,
            file_sha256=corpus_file_sha,
            byte_size=corpus_size,
            semantic_sha256=str(corpus["corpus_view_sha256"]),
            mode=0o600,
        ),
        "blocks": block_bindings,
        "private_json_file_count": 8,
        "F_search_label_pack_created": False,
    }


def execute_acquisition_once(
    *,
    train_payload: Any,
    documents: DocumentResolver | DocumentCatalog,
    qualification: QualificationBinding,
    paths: AcquisitionPaths,
    source_bindings: Mapping[str, Any],
    enforce_formal_counts: bool = False,
    random_bytes: Callable[[int], bytes] = os.urandom,
    stability_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if not isinstance(qualification, QualificationBinding) or not isinstance(
        source_bindings, Mapping
    ):
        raise HoVerAcquisitionError("acquisition qualification or source binding is absent")
    if enforce_formal_counts and stability_check is None:
        raise HoVerAcquisitionError(
            "formal acquisition requires an implementation stability check"
        )
    _preflight_outputs(paths)
    _prepare_output_parents(paths)
    marker, marker_binding = _consume_marker(
        paths=paths, qualification=qualification, source_bindings=source_bindings
    )
    secret, secret_binding = create_one_shot_secret_once(
        path=paths.secret, random_bytes=random_bytes
    )
    decoded_train_payload = (
        strict_json_loads(train_payload, label="pinned HoVer TRAIN")
        if isinstance(train_payload, bytes)
        else train_payload
    )
    eligible, source_stats = parse_train_payload(
        decoded_train_payload,
        documents=documents,
        enforce_formal_counts=enforce_formal_counts,
    )
    if enforce_formal_counts:
        _formal_source_stats_match(source_stats, qualification)
    blocks, selection_stats = select_private_blocks(
        eligible,
        source_stats=source_stats,
        qualification=qualification,
        secret=secret,
    )
    corpus_rows, article_id_by_rowid, corpus_stats = build_fixed_corpus(
        blocks=blocks,
        documents=documents,
        qualification=qualification,
        secret=secret,
    )
    corpus, views, labels, materialization_stats = materialize_private_payloads(
        blocks=blocks,
        corpus_rows=corpus_rows,
        article_id_by_rowid=article_id_by_rowid,
    )
    commitments = persist_private_payloads(
        corpus=corpus, views=views, labels=labels, paths=paths
    )
    if stability_check is not None:
        try:
            stability_check()
        except HoVerAcquisitionError:
            raise
        except Exception as exc:
            raise HoVerAcquisitionError(
                "implementation closure drifted before public receipt"
            ) from exc
    body = {
        "schema": PUBLIC_RECEIPT_SCHEMA,
        "version": VERSION,
        "status": "private_four_block_pack_formed",
        "design_sha256": DESIGN_SHA256,
        "qualification_sha256": qualification.qualification_sha256,
        "attempt_marker": marker_binding,
        "selection_secret_commitment": secret_binding,
        "source_bindings": dict(source_bindings),
        "source_requalification": source_stats,
        "selection_qualification": selection_stats,
        "corpus_qualification": corpus_stats,
        "materialization_qualification": materialization_stats,
        "private_pack_commitments": commitments,
        "label_isolation": {
            "A_form": "separate_late_utility_pack",
            "F_search": "utility_pack_not_created",
            "A_hold": "separate_late_utility_pack",
            "M_search": "separate_late_utility_pack",
        },
        "public_uid_hpqa_claim_title_body_support_or_gold_payload_included": False,
        "same_source_replay_secret_rotation_resample_replacement_or_retry_authorized": False,
    }
    if marker.get("marker_sha256") != marker_binding["semantic_sha256"]:
        raise HoVerAcquisitionError("attempt marker binding drifted")
    receipt = with_self_hash(body, "acquisition_sha256")
    _assert_public_safe(receipt)
    _write_json_exclusive(paths.public_receipt, receipt, mode=0o644)
    return receipt


def _validated_private_binding(
    value: object, *, expected_name: str, expected_mode: str = "0600"
) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {
            "relative_name",
            "file_sha256",
            "byte_size",
            "mode",
            "semantic_sha256",
        }
        or value.get("relative_name") != expected_name
        or value.get("mode") != expected_mode
        or type(value.get("byte_size")) is not int
        or value["byte_size"] <= 0
    ):
        raise HoVerAcquisitionError("private file binding drifted")
    _require_sha256(value.get("file_sha256"), "private file hash")
    _require_sha256(value.get("semantic_sha256"), "private semantic hash")
    return dict(value)


def _validated_pack_commitments(receipt: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    packs = receipt.get("private_pack_commitments")
    if (
        not isinstance(packs, Mapping)
        or set(packs) != {
            "corpus_view",
            "blocks",
            "private_json_file_count",
            "F_search_label_pack_created",
        }
        or packs.get("private_json_file_count") != 8
        or packs.get("F_search_label_pack_created") is not False
    ):
        raise HoVerAcquisitionError("private pack commitment envelope drifted")
    corpus = _validated_private_binding(
        packs.get("corpus_view"), expected_name=Path(CORPUS_VIEW_RELATIVE).name
    )
    raw_blocks = packs.get("blocks")
    if not isinstance(raw_blocks, Mapping) or set(raw_blocks) != set(BLOCK_ORDER):
        raise HoVerAcquisitionError("private block commitment set drifted")
    blocks: dict[str, Any] = {}
    for block in BLOCK_ORDER:
        row = raw_blocks[block]
        if (
            not isinstance(row, Mapping)
            or set(row) != {"item_count", "view", "labels"}
            or row.get("item_count") != BLOCK_COUNTS[block]
        ):
            raise HoVerAcquisitionError("private block commitment drifted")
        view = _validated_private_binding(
            row.get("view"), expected_name=Path(BLOCK_VIEW_RELATIVES[block]).name
        )
        raw_labels = row.get("labels")
        if block == "F_search":
            if raw_labels != {"created": False}:
                raise HoVerAcquisitionError("F_search label commitment exists")
            labels = {"created": False}
        else:
            if not isinstance(raw_labels, Mapping) or raw_labels.get("created") is not True:
                raise HoVerAcquisitionError("late utility commitment is missing")
            label_body = dict(raw_labels)
            del label_body["created"]
            labels = {
                "created": True,
                **_validated_private_binding(
                    label_body,
                    expected_name=Path(BLOCK_LABEL_RELATIVES[block]).name,
                ),
            }
        blocks[block] = {"item_count": row["item_count"], "view": view, "labels": labels}
    return corpus, blocks


def _git_output(root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise HoVerAcquisitionError("Git receipt verification is unavailable") from exc
    if completed.returncode != 0:
        raise HoVerAcquisitionError("Git receipt verification failed")
    return completed.stdout


def _verify_regular_file_at_head(
    *, project: Path, path: Path, working_raw: bytes, label: str
) -> tuple[str, str]:
    root_raw = _git_output(project, "rev-parse", "--show-toplevel")
    try:
        git_root = Path(root_raw.decode("utf-8", errors="strict").strip()).absolute()
        relative = path.absolute().relative_to(git_root).as_posix()
    except (UnicodeDecodeError, ValueError) as exc:
        raise HoVerAcquisitionError(f"{label} is outside its Git worktree") from exc
    # Pathspecs passed to ``git ls-tree`` are interpreted relative to ``-C``.
    # ``relative`` is rooted at the worktree, so all subsequent Git commands
    # must run from that same root even when ``project`` is a subdirectory.
    head_before = _git_output(git_root, "rev-parse", "--verify", "HEAD").decode(
        "ascii", errors="strict"
    ).strip()
    if re.fullmatch(r"[0-9a-f]{40}", head_before) is None:
        raise HoVerAcquisitionError("Git HEAD identity drifted")
    tree = _git_output(git_root, "ls-tree", "-z", head_before, "--", relative)
    rows = [row for row in tree.split(b"\0") if row]
    if len(rows) != 1 or b"\t" not in rows[0]:
        raise HoVerAcquisitionError(f"{label} is not a unique Git HEAD blob")
    metadata, encoded_path = rows[0].split(b"\t", 1)
    fields = metadata.split()
    try:
        observed_path = encoded_path.decode("utf-8", errors="strict")
        object_id = fields[2].decode("ascii", errors="strict")
    except (IndexError, UnicodeDecodeError) as exc:
        raise HoVerAcquisitionError(f"{label} Git tree row drifted") from exc
    if (
        fields[:2] != [b"100644", b"blob"]
        or observed_path != relative
        or re.fullmatch(r"[0-9a-f]{40}", object_id) is None
    ):
        raise HoVerAcquisitionError(f"{label} Git tree binding drifted")
    committed_raw = _git_output(git_root, "cat-file", "blob", object_id)
    head_after = _git_output(git_root, "rev-parse", "--verify", "HEAD").decode(
        "ascii", errors="strict"
    ).strip()
    if head_after != head_before or not hmac.compare_digest(committed_raw, working_raw):
        raise HoVerAcquisitionError(f"{label} working bytes differ from stable Git HEAD")
    return head_before, object_id


def _validate_marker_and_secret_bindings(
    *, paths: AcquisitionPaths, receipt: Mapping[str, Any]
) -> None:
    marker_binding = _validated_private_binding(
        receipt.get("attempt_marker"), expected_name=paths.marker.name
    )
    marker = _read_private_payload(paths.marker, marker_binding, label="attempt marker")
    verify_self_hash(marker, hash_field="marker_sha256", schema=ONE_SHOT_MARKER_SCHEMA)
    if (
        marker.get("version") != VERSION
        or marker.get("status") != "one_shot_attempt_consumed_before_secret_or_selection"
        or marker.get("design_sha256") != DESIGN_SHA256
        or marker.get("qualification_sha256") != receipt.get("qualification_sha256")
        or marker.get("source_bindings_sha256") != stable_hash(receipt.get("source_bindings"))
        or marker.get(
            "same_source_replay_secret_rotation_resample_or_replacement_authorized"
        )
        is not False
        or marker.get("marker_sha256") != marker_binding["semantic_sha256"]
    ):
        raise HoVerAcquisitionError("attempt marker contract drifted")
    secret_binding = receipt.get("selection_secret_commitment")
    if (
        not isinstance(secret_binding, Mapping)
        or set(secret_binding) != {"relative_name", "file_sha256", "byte_size", "mode"}
        or secret_binding.get("relative_name") != paths.secret.name
        or secret_binding.get("byte_size") != 32
        or secret_binding.get("mode") != "0600"
    ):
        raise HoVerAcquisitionError("selection secret commitment drifted")
    expected_secret_sha = _require_sha256(
        secret_binding.get("file_sha256"), "selection secret file hash"
    )
    _require_regular_file(paths.secret, label="selection secret", mode=0o600)
    secret = paths.secret.read_bytes()
    if len(secret) != 32 or not hmac.compare_digest(_sha256_bytes(secret), expected_secret_sha):
        raise HoVerAcquisitionError("selection secret file binding drifted")


def load_committed_acquisition_receipt(
    project: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = default_acquisition_paths(project)
    payload, raw = _read_bound_json(
        paths.public_receipt, label="HoVer acquisition receipt", mode=0o644
    )
    if not isinstance(payload, dict):
        raise HoVerAcquisitionError("acquisition receipt root drifted")
    expected_keys = {
        "schema",
        "version",
        "status",
        "design_sha256",
        "qualification_sha256",
        "attempt_marker",
        "selection_secret_commitment",
        "source_bindings",
        "source_requalification",
        "selection_qualification",
        "corpus_qualification",
        "materialization_qualification",
        "private_pack_commitments",
        "label_isolation",
        "public_uid_hpqa_claim_title_body_support_or_gold_payload_included",
        "same_source_replay_secret_rotation_resample_replacement_or_retry_authorized",
        "acquisition_sha256",
    }
    if set(payload) != expected_keys:
        raise HoVerAcquisitionError("acquisition receipt schema drifted")
    verify_self_hash(payload, hash_field="acquisition_sha256", schema=PUBLIC_RECEIPT_SCHEMA)
    if (
        payload.get("version") != VERSION
        or payload.get("status") != "private_four_block_pack_formed"
        or payload.get("design_sha256") != DESIGN_SHA256
        or payload.get("public_uid_hpqa_claim_title_body_support_or_gold_payload_included")
        is not False
        or payload.get(
            "same_source_replay_secret_rotation_resample_replacement_or_retry_authorized"
        )
        is not False
    ):
        raise HoVerAcquisitionError("acquisition receipt contract drifted")
    _assert_public_safe(payload)
    corpus, blocks = _validated_pack_commitments(payload)
    _validate_marker_and_secret_bindings(paths=paths, receipt=payload)
    git_head, git_blob = _verify_regular_file_at_head(
        project=_canonical_project(project),
        path=paths.public_receipt,
        working_raw=raw,
        label="acquisition receipt",
    )
    return payload, {
        "receipt_file_sha256": _sha256_bytes(raw),
        "receipt_git_head": git_head,
        "receipt_git_blob_sha1": git_blob,
        "corpus": corpus,
        "blocks": blocks,
    }


def _validate_formal_receipt_aggregates(payload: Mapping[str, Any]) -> None:
    """Reject a committed but nonformal or semantically rewritten receipt."""

    qualification = QualificationBinding(
        qualification_sha256=QUALIFICATION_SHA256,
        eligible_record_count=17_905,
        normalized_claim_collision_member_count=266,
        eligible_hpqa_group_count=6_103,
        sqlite_document_row_count=5_233_329,
        sqlite_maximum_rowid=5_233_329,
    )
    expected_sources = {
        "training": {
            "relative_path": FORMAL_TRAIN_RELATIVE.as_posix(),
            "sha256": TRAINING_SHA256,
            "byte_size": TRAINING_SIZE,
            "mode": "0600",
        },
        "sqlite_corpus": {
            "relative_path": FORMAL_SQLITE_RELATIVE.as_posix(),
            "sha256": CORPUS_SOURCE_SHA256,
            "byte_size": CORPUS_SOURCE_SIZE,
            "mode": "0600",
        },
        "qualification_sha256": QUALIFICATION_SHA256,
    }
    source_bindings = payload.get("source_bindings")
    implementation = (
        source_bindings.get("implementation")
        if isinstance(source_bindings, Mapping)
        else None
    )
    source_stats = payload.get("source_requalification")
    if (
        payload.get("qualification_sha256") != QUALIFICATION_SHA256
        or not isinstance(source_bindings, Mapping)
        or set(source_bindings) != {*expected_sources, "implementation"}
        or any(source_bindings.get(key) != value for key, value in expected_sources.items())
        or not isinstance(implementation, Mapping)
        or set(implementation)
        != {
            "implementation_freeze_sha256",
            "implementation_freeze_file_sha256",
            "implementation_freeze_git_blob_sha1",
            "acquisition_execution_git_head",
            "all_frozen_python_origins_verified",
        }
        or _HEX64.fullmatch(
            str(implementation.get("implementation_freeze_sha256"))
        )
        is None
        or _HEX64.fullmatch(
            str(implementation.get("implementation_freeze_file_sha256"))
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{40}",
            str(implementation.get("implementation_freeze_git_blob_sha1")),
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{40}",
            str(implementation.get("acquisition_execution_git_head")),
        )
        is None
        or implementation.get("all_frozen_python_origins_verified") is not True
        or not isinstance(source_stats, Mapping)
    ):
        raise HoVerAcquisitionError("formal acquisition source binding drifted")
    _formal_source_stats_match(source_stats, qualification)
    if source_stats.get("item_uid_claim_title_body_or_support_text_emitted") is not False:
        raise HoVerAcquisitionError("formal source aggregate leaked item content")

    expected_hop_counts = {
        block: {hop: HOP_QUOTAS[block] for hop in HOP_STRATA}
        for block in BLOCK_ORDER
    }
    selection = payload.get("selection_qualification")
    if (
        not isinstance(selection, Mapping)
        or set(selection)
        != {
            "qualification_sha256",
            "maximum_b_matching_cardinality",
            "required_b_matching_cardinality",
            "selected_block_counts",
            "selected_hop_stratum_counts",
            "selected_unique_uid_count",
            "selected_unique_hpqa_group_count",
            "selected_unique_normalized_claim_count",
            "selection_contract",
        }
        or selection.get("qualification_sha256") != QUALIFICATION_SHA256
        or selection.get("maximum_b_matching_cardinality") != TOTAL_SELECTED
        or selection.get("required_b_matching_cardinality") != TOTAL_SELECTED
        or selection.get("selected_block_counts") != BLOCK_COUNTS
        or selection.get("selected_hop_stratum_counts") != expected_hop_counts
        or selection.get("selected_unique_uid_count") != TOTAL_SELECTED
        or selection.get("selected_unique_hpqa_group_count") != TOTAL_SELECTED
        or selection.get("selected_unique_normalized_claim_count") != TOTAL_SELECTED
        or selection.get("selection_contract")
        != {
            "whole_normalized_claim_collision_groups_excluded": True,
            "hpqa_id_global_at_most_one": True,
            "private_HMAC_b_matching": True,
            "private_HMAC_hop_assignment": True,
            "private_HMAC_block_unified_order": True,
        }
    ):
        raise HoVerAcquisitionError("formal acquisition selection aggregates drifted")

    corpus = payload.get("corpus_qualification")
    corpus_integer_fields = (
        "unique_selected_gold_article_count",
        "filler_article_count",
        "distractor_counter_attempt_count",
        "distractor_rejected_missing_rowid_count",
        "distractor_rejected_duplicate_or_gold_count",
        "distractor_rejected_duplicate_serialization_count",
    )
    if (
        not isinstance(corpus, Mapping)
        or set(corpus)
        != {
            "fixed_article_count",
            *corpus_integer_fields,
            "all_selected_gold_included",
            "origin_or_is_gold_in_corpus_view",
            "shared_all_blocks_and_methods",
            "corpus_order_independent_private_HMAC",
        }
        or corpus.get("fixed_article_count") != CORPUS_SIZE
        or any(type(corpus.get(field)) is not int for field in corpus_integer_fields)
        or not 1 <= int(corpus["unique_selected_gold_article_count"]) <= 432
        or corpus.get("filler_article_count")
        != CORPUS_SIZE - int(corpus["unique_selected_gold_article_count"])
        or int(corpus["distractor_counter_attempt_count"])
        > DISTRACTOR_ATTEMPT_CAP
        or any(int(corpus[field]) < 0 for field in corpus_integer_fields[2:])
        or int(corpus["distractor_counter_attempt_count"])
        != int(corpus["filler_article_count"])
        + int(corpus["distractor_rejected_missing_rowid_count"])
        + int(corpus["distractor_rejected_duplicate_or_gold_count"])
        + int(corpus["distractor_rejected_duplicate_serialization_count"])
        or corpus.get("all_selected_gold_included") is not True
        or corpus.get("origin_or_is_gold_in_corpus_view") is not False
        or corpus.get("shared_all_blocks_and_methods") is not True
        or corpus.get("corpus_order_independent_private_HMAC") is not True
    ):
        raise HoVerAcquisitionError("formal acquisition corpus aggregates drifted")

    materialization = payload.get("materialization_qualification")
    expected_histograms = {
        block: {
            str(hop): HOP_QUOTAS[block]
            for hop in (2, 3, 4)
        }
        for block in BLOCK_ORDER
    }
    if (
        not isinstance(materialization, Mapping)
        or materialization
        != {
            "fixed_article_count": CORPUS_SIZE,
            "block_gold_cardinality_histograms": expected_histograms,
            "F_search_utility_label_pack_created": False,
            "claim_verdict_support_sentences_or_hop_in_view": False,
            "origin_or_is_gold_in_corpus_view": False,
        }
        or payload.get("label_isolation")
        != {
            "A_form": "separate_late_utility_pack",
            "F_search": "utility_pack_not_created",
            "A_hold": "separate_late_utility_pack",
            "M_search": "separate_late_utility_pack",
        }
    ):
        raise HoVerAcquisitionError("formal acquisition materialization drifted")


def load_formal_committed_acquisition_receipt(
    project: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the Git-bound receipt and enforce every formal aggregate boundary."""

    root = _canonical_project(project)
    payload, binding = load_committed_acquisition_receipt(root)
    _validate_formal_receipt_aggregates(payload)
    implementation = _verify_formal_implementation(root)
    frozen = payload["source_bindings"]["implementation"]
    for field in (
        "implementation_freeze_sha256",
        "implementation_freeze_file_sha256",
        "implementation_freeze_git_blob_sha1",
        "all_frozen_python_origins_verified",
    ):
        if frozen.get(field) != implementation.get(field):
            raise HoVerAcquisitionError(
                "formal acquisition implementation binding drifted"
            )
    current_head = binding.get("receipt_git_head")
    execution_head = frozen.get("acquisition_execution_git_head")
    if implementation.get("acquisition_execution_git_head") != current_head:
        raise HoVerAcquisitionError("formal acquisition current HEAD drifted")
    try:
        _git_output(
            root,
            "merge-base",
            "--is-ancestor",
            str(execution_head),
            str(current_head),
        )
    except HoVerAcquisitionError as exc:
        raise HoVerAcquisitionError(
            "acquisition execution HEAD is not an ancestor of its receipt"
        ) from exc
    return payload, binding


def _verify_formal_implementation(project: Path) -> dict[str, Any]:
    """Close the executing acquisition module to the committed freeze."""

    from assumption_agent.benchmarks import hover_implementation_freeze_v1 as freeze

    try:
        current_target = os.environ.get(isolated_bootstrap.TARGET_ENV)
        if current_target not in isolated_bootstrap.TARGETS:
            raise HoVerAcquisitionError(
                "formal acquisition has no isolated interpreter target"
            )
        isolated_bootstrap.assert_isolated(current_target)
        receipt = freeze.verify_committed_implementation_freeze(project)
        freeze.import_and_verify_frozen_python_roles(
            project=project,
            implementation_receipt=receipt,
        )
    except Exception as exc:
        raise HoVerAcquisitionError(
            "formal acquisition implementation freeze verification failed"
        ) from exc
    role_paths = receipt.get("role_paths")
    acquisition_relative = (
        role_paths.get("acquisition") if isinstance(role_paths, Mapping) else None
    )
    if (
        not isinstance(acquisition_relative, str)
        or (project / acquisition_relative).resolve(strict=True)
        != Path(__file__).resolve(strict=True)
    ):
        raise HoVerAcquisitionError("executing acquisition module is not frozen")
    return {
        "implementation_freeze_sha256": receipt[freeze.HASH_FIELD],
        "implementation_freeze_file_sha256": receipt[
            "implementation_freeze_file_sha256"
        ],
        "implementation_freeze_git_blob_sha1": receipt[
            "implementation_freeze_git_blob_sha1"
        ],
        "acquisition_execution_git_head": receipt["verified_git_head"],
        "all_frozen_python_origins_verified": True,
    }


def _read_private_payload(path: Path, binding: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    metadata = _require_regular_file(path, label=label, mode=0o600)
    raw = path.read_bytes()
    if (
        len(raw) != metadata.st_size
        or len(raw) != binding.get("byte_size")
        or not hmac.compare_digest(_sha256_bytes(raw), str(binding.get("file_sha256")))
    ):
        raise HoVerAcquisitionError(f"{label} file binding drifted")
    payload = strict_json_loads(raw, label=label)
    if not isinstance(payload, dict):
        raise HoVerAcquisitionError(f"{label} root drifted")
    return payload


def load_corpus_view(*, project: Path) -> dict[str, Any]:
    paths = default_acquisition_paths(project)
    _receipt, bindings = load_committed_acquisition_receipt(project)
    binding = bindings["corpus"]
    payload = _read_private_payload(paths.corpus_view, binding, label="corpus view")
    validate_corpus_view(payload)
    if payload.get("corpus_view_sha256") != binding["semantic_sha256"]:
        raise HoVerAcquisitionError("corpus semantic binding drifted")
    return payload


def load_block_view(*, project: Path, expected_block: str) -> dict[str, Any]:
    if expected_block not in BLOCK_ORDER:
        raise HoVerAcquisitionError("claim view block is invalid")
    paths = default_acquisition_paths(project)
    _receipt, bindings = load_committed_acquisition_receipt(project)
    binding = bindings["blocks"][expected_block]["view"]
    payload = _read_private_payload(
        paths.block_views[expected_block], binding, label=f"{expected_block} claim view"
    )
    validate_block_view(payload, expected_block=expected_block)
    if payload.get("block_view_sha256") != binding["semantic_sha256"]:
        raise HoVerAcquisitionError("block view semantic binding drifted")
    return payload


def load_block_labels(*, project: Path, expected_block: str) -> dict[str, Any]:
    if expected_block == "F_search":
        raise HoVerAcquisitionError("F_search utility label pack does not exist")
    if expected_block not in {"A_form", "A_hold", "M_search"}:
        raise HoVerAcquisitionError("utility label block is invalid")
    paths = default_acquisition_paths(project)
    _receipt, bindings = load_committed_acquisition_receipt(project)
    binding = bindings["blocks"][expected_block]["labels"]
    if binding.get("created") is not True:
        raise HoVerAcquisitionError("late utility label commitment is absent")
    payload = _read_private_payload(
        paths.block_labels[expected_block], binding, label=f"{expected_block} utility labels"
    )
    validate_block_labels(payload, expected_block=expected_block)
    if payload.get("block_labels_sha256") != binding["semantic_sha256"]:
        raise HoVerAcquisitionError("utility label semantic binding drifted")
    return payload


def preflight_formal_private_pack_files(project: Path) -> dict[str, Any]:
    """Hash-bind every private pack file without decoding any view or label."""

    root = _canonical_project(project)
    _receipt, bindings = load_formal_committed_acquisition_receipt(root)
    paths = default_acquisition_paths(root)
    targets: list[tuple[Path, Mapping[str, Any], str]] = [
        (paths.corpus_view, bindings["corpus"], "corpus view"),
    ]
    for block in BLOCK_ORDER:
        block_binding = bindings["blocks"][block]
        targets.append(
            (
                paths.block_views[block],
                block_binding["view"],
                f"{block} claim view",
            )
        )
        if block != "F_search":
            targets.append(
                (
                    paths.block_labels[block],
                    block_binding["labels"],
                    f"{block} utility labels",
                )
            )
    total_bytes = 0
    for path, binding, label in targets:
        observed_sha, observed_size = _stream_file_sha256(
            path, label=label, mode=0o600
        )
        if (
            observed_size != binding.get("byte_size")
            or not hmac.compare_digest(
                observed_sha, str(binding.get("file_sha256"))
            )
        ):
            raise HoVerAcquisitionError(f"{label} preflight binding drifted")
        total_bytes += observed_size
    f_label = FORMAL_OUTPUT_ROOT_RELATIVE / "private/F_search.utility_labels.json"
    if os.path.lexists(root / f_label):
        raise HoVerAcquisitionError("F_search utility label residue exists")
    return {
        "private_pack_file_count": len(targets),
        "private_pack_total_bytes": total_bytes,
        "private_pack_json_payloads_decoded": 0,
        "claim_or_label_fields_exposed_to_controller": False,
        "all_private_file_hashes_match_committed_receipt": True,
    }


def formal_acquire(project: Path) -> dict[str, Any]:
    root = _canonical_project(project)
    isolated_bootstrap.assert_isolated(
        "assumption_agent.benchmarks.hover_direct_acquisition_v1"
    )
    implementation_binding = _verify_formal_implementation(root)
    qualification_path = root / FORMAL_QUALIFICATION_RELATIVE
    qualification_payload, qualification_raw = _read_bound_json(
        qualification_path,
        label="committed HoVer source qualification",
    )
    if not isinstance(qualification_payload, dict):
        raise HoVerAcquisitionError("source qualification root drifted")
    _verify_regular_file_at_head(
        project=root,
        path=qualification_path,
        working_raw=qualification_raw,
        label="source qualification manifest",
    )
    qualification = validate_qualification_manifest(qualification_payload)
    train_path = root / FORMAL_TRAIN_RELATIVE
    sqlite_path = root / FORMAL_SQLITE_RELATIVE
    train_raw = _read_bound_bytes(
        train_path,
        label="pinned HoVer TRAIN",
        expected_sha256=TRAINING_SHA256,
        expected_size=TRAINING_SIZE,
        mode=0o600,
    )
    sqlite_sha, sqlite_size = _stream_file_sha256(
        sqlite_path, label="pinned HoVer SQLite", mode=0o600
    )
    if sqlite_sha != CORPUS_SOURCE_SHA256 or sqlite_size != CORPUS_SOURCE_SIZE:
        raise HoVerAcquisitionError("pinned HoVer SQLite source binding drifted")
    source_bindings = {
        "training": {
            "relative_path": FORMAL_TRAIN_RELATIVE.as_posix(),
            "sha256": TRAINING_SHA256,
            "byte_size": TRAINING_SIZE,
            "mode": "0600",
        },
        "sqlite_corpus": {
            "relative_path": FORMAL_SQLITE_RELATIVE.as_posix(),
            "sha256": CORPUS_SOURCE_SHA256,
            "byte_size": CORPUS_SOURCE_SIZE,
            "mode": "0600",
        },
        "qualification_sha256": qualification.qualification_sha256,
        "implementation": implementation_binding,
    }

    def assert_implementation_stable() -> None:
        if _verify_formal_implementation(root) != implementation_binding:
            raise HoVerAcquisitionError(
                "formal acquisition implementation changed during execution"
            )

    with ImmutableSQLiteDocumentResolver(
        path=sqlite_path,
        row_count=qualification.sqlite_document_row_count,
        maximum_rowid=qualification.sqlite_maximum_rowid,
        binding_sha256=CORPUS_SOURCE_SHA256,
    ) as documents:
        return execute_acquisition_once(
            train_payload=train_raw,
            documents=documents,
            qualification=qualification,
            paths=default_acquisition_paths(root),
            source_bindings=source_bindings,
            enforce_formal_counts=True,
            stability_check=assert_implementation_stable,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_arguments = tuple(sys.argv[1:] if argv is None else argv)
    isolated_bootstrap.reexec_isolated(
        "assumption_agent.benchmarks.hover_direct_acquisition_v1",
        raw_arguments,
    )
    arguments = _parser().parse_args(raw_arguments)
    formal_acquire(arguments.project)
    return 0


__all__ = [
    "AcquisitionPaths",
    "AssignedRecord",
    "BLOCK_COUNTS",
    "BLOCK_LABEL_SCHEMA",
    "BLOCK_ORDER",
    "BLOCK_VIEW_SCHEMA",
    "CORPUS_SIZE",
    "CORPUS_VIEW_SCHEMA",
    "DESIGN_SHA256",
    "DocumentCatalog",
    "DocumentResolver",
    "DocumentRow",
    "EligibleRecord",
    "HOP_QUOTAS",
    "HOP_STRATA",
    "HoVerAcquisitionError",
    "ImmutableSQLiteDocumentResolver",
    "LABEL_ITEM_SCHEMA",
    "QualificationBinding",
    "VERSION",
    "VIEW_ITEM_SCHEMA",
    "build_document_catalog",
    "build_fixed_corpus",
    "canonical_json_bytes",
    "create_one_shot_secret_once",
    "default_acquisition_paths",
    "execute_acquisition_once",
    "formal_acquire",
    "hmac_digest",
    "load_block_labels",
    "load_block_view",
    "load_committed_acquisition_receipt",
    "load_formal_committed_acquisition_receipt",
    "load_corpus_view",
    "materialize_private_payloads",
    "normalize_claim",
    "normalize_support_title_nfd",
    "parse_train_payload",
    "preflight_formal_private_pack_files",
    "persist_private_payloads",
    "select_private_blocks",
    "stable_hash",
    "strict_json_loads",
    "synthetic_qualification_binding",
    "validate_block_labels",
    "validate_block_view",
    "validate_corpus_view",
    "validate_qualification_manifest",
    "verify_self_hash",
    "with_self_hash",
]


if __name__ == "__main__":
    from assumption_agent.benchmarks import hover_direct_acquisition_v1 as _canonical

    raise SystemExit(_canonical.main())
