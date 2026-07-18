"""Private-HMAC selection and pack formation for FEVEROUS P6/E2.

This module intentionally accepts only already-qualified candidate records and
already-linearized atomic units.  Reading the official annotations or SQLite
source belongs to a separate source adapter.  Consequently the selection
core is fully testable on synthetic data before any real cohort exists.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import heapq
import hmac
import json
import re
from types import MappingProxyType
from typing import Any
import unicodedata

from assumption_agent.benchmarks.feverous_atomic_corpus_v1 import (
    VERSION as ATOMIC_LINEARIZER_VERSION,
)
from assumption_agent.benchmarks.feverous_wikipedia_source_qualification_v1 import (
    FeverousWikipediaQualificationError,
    parse_element_id,
)


VERSION = "feverous_p6_e2_acquisition_v1"
BLOCK_ORDER = ("A_form", "F_search", "A_hold", "M_search")
FAMILIES = (
    "Combining Tables and Text",
    "Entity Disambiguation",
    "Multi-hop Reasoning",
    "Numerical Reasoning",
)
VERDICTS = ("SUPPORTS", "REFUTES")
PER_FAMILY = {
    "A_form": 24,
    "F_search": 12,
    "A_hold": 18,
    "M_search": 18,
}
BLOCK_COUNTS = {
    block: PER_FAMILY[block] * len(FAMILIES) for block in BLOCK_ORDER
}
CORPUS_UNIT_COUNT = 8192
MINIMUM_GOLD_ATOMS = 2
MAXIMUM_GOLD_ATOMS = 5

CORPUS_VIEW_SCHEMA = f"{VERSION}_corpus_view"
BLOCK_VIEW_SCHEMA = f"{VERSION}_block_view"
BLOCK_LABEL_SCHEMA = f"{VERSION}_block_labels"
IDENTITY_STREAM_RECEIPT_SCHEMA = (
    "feverous_p6_e2_source_adapter_v1_corpus_identity_stream_receipt"
)
MATERIALIZATION_RECEIPT_SCHEMA = (
    "feverous_p6_e2_source_adapter_v1_selected_corpus_materialization_receipt"
)
FORMAL_DATABASE_SHA256 = (
    "a980581f55d46a252090b29269954503735b6f00274d05225476a650ab940276"
)
FORMAL_DATABASE_SIZE_BYTES = 53_486_538_752
FORMAL_DATABASE_ROW_COUNT = 5_421_406

_HMAC_DOMAIN = b"feverous_p6_e2_acquisition_v1/hmac-sha256/v1"
_WHITESPACE = re.compile(r"\s+")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ATOMIC_SIDECAR_KEYS = frozenset(
    {
        "linearizer_version",
        "page",
        "local_id",
        "unit_type",
        "coordinates",
        "section_ids",
        "section_path",
        "official_ordinal",
        "previous_atomic_local_id",
        "next_atomic_local_id",
        "table_id",
        "table_kind",
        "table_caption",
        "row_span",
        "column_span",
        "applicable_row_header_ids",
        "applicable_column_header_ids",
        "list_id",
        "list_ancestor_ids",
    }
)
_REQUIRED_ATOMIC_SIDECAR_KEYS = frozenset(
    {"linearizer_version", "page", "local_id", "unit_type"}
)


class FeverousP6E2AcquisitionError(RuntimeError):
    """A candidate, selection, corpus, or private-pack invariant drifted."""


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise FeverousP6E2AcquisitionError(
            "value is not canonical JSON"
        ) from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise FeverousP6E2AcquisitionError("self-hash field already exists")
    output = dict(body)
    output[field] = stable_hash(body)
    return output


def verify_self_hash(payload: Mapping[str, Any], field: str) -> str:
    declared = payload.get(field)
    if not isinstance(declared, str) or _SHA256.fullmatch(declared) is None:
        raise FeverousP6E2AcquisitionError("self-hash is absent")
    body = dict(payload)
    del body[field]
    if stable_hash(body) != declared:
        raise FeverousP6E2AcquisitionError("self-hash mismatch")
    return declared


def normalize_claim(value: str) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise FeverousP6E2AcquisitionError("claim must be a safe string")
    normalized = _WHITESPACE.sub(
        " ", unicodedata.normalize("NFKC", value)
    ).strip().casefold()
    if not normalized:
        raise FeverousP6E2AcquisitionError("claim is empty")
    return normalized


def _safe_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise FeverousP6E2AcquisitionError(f"{field} must be nonempty")
    return value


def _frame(raw: bytes) -> bytes:
    return len(raw).to_bytes(8, "big") + raw


def hmac_digest(secret: bytes, purpose: str, *parts: str) -> bytes:
    if not isinstance(secret, bytes) or len(secret) != 32:
        raise FeverousP6E2AcquisitionError(
            "selection secret must contain exactly 32 bytes"
        )
    rows = [_HMAC_DOMAIN, _safe_text(purpose, "HMAC purpose").encode()]
    rows.extend(_safe_text(part, "HMAC part").encode() for part in parts)
    message = b"".join(_frame(row) for row in rows)
    return hmac.new(secret, message, hashlib.sha256).digest()


@dataclass(frozen=True)
class CandidateRecord:
    """One source-qualified claim; all fields remain inside acquisition."""

    source_key: str
    claim: str
    family: str
    verdict: str
    evidence_sets: tuple[tuple[str, ...], ...]
    all_official_evidence_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        _safe_text(self.source_key, "source key")
        normalize_claim(self.claim)
        if self.family not in FAMILIES or self.verdict not in VERDICTS:
            raise FeverousP6E2AcquisitionError(
                "candidate family or verdict is outside the frozen registry"
            )
        try:
            evidence_sets = tuple(
                sorted(
                    {tuple(sorted(row)) for row in self.evidence_sets},
                    key=lambda row: (stable_hash(list(row)), row),
                )
            )
            official_keys = tuple(sorted(self.all_official_evidence_keys))
        except (TypeError, AttributeError) as exc:
            raise FeverousP6E2AcquisitionError(
                "candidate evidence containers are malformed"
            ) from exc
        object.__setattr__(self, "evidence_sets", evidence_sets)
        object.__setattr__(self, "all_official_evidence_keys", official_keys)
        if not self.evidence_sets:
            raise FeverousP6E2AcquisitionError(
                "candidate has no eligible evidence set"
            )
        for row in self.evidence_sets:
            if (
                not MINIMUM_GOLD_ATOMS <= len(row) <= MAXIMUM_GOLD_ATOMS
                or len(row) != len(set(row))
                or any(not isinstance(key, str) or not key for key in row)
            ):
                raise FeverousP6E2AcquisitionError(
                    "candidate evidence set is malformed"
                )
        if (
            not self.all_official_evidence_keys
            or len(self.all_official_evidence_keys)
            != len(set(self.all_official_evidence_keys))
            or any(
                not isinstance(key, str) or not key
                for key in self.all_official_evidence_keys
            )
        ):
            raise FeverousP6E2AcquisitionError(
                "official evidence-key universe is malformed"
            )
        universe = set(self.all_official_evidence_keys)
        if any(not set(row).issubset(universe) for row in self.evidence_sets):
            raise FeverousP6E2AcquisitionError(
                "eligible evidence set is outside the official universe"
            )

    @property
    def normalized_claim(self) -> str:
        return normalize_claim(self.claim)


@dataclass(frozen=True)
class AssignedRecord:
    block: str
    ordinal: int
    record: CandidateRecord
    canonical_gold_keys: tuple[str, ...]


@dataclass(frozen=True)
class CorpusUnit:
    """One nonempty atomic unit already produced by the frozen linearizer."""

    unit_key: str
    text: str
    unit_type: str
    sidecar: Mapping[str, Any]

    def __post_init__(self) -> None:
        _safe_text(self.unit_key, "unit key")
        text = _safe_text(self.text, "unit text")
        first_line = text.split("\n", 1)[0]
        if not first_line.startswith("TARGET: ") or not normalize_claim(
            first_line[len("TARGET: ") :]
        ):
            raise FeverousP6E2AcquisitionError(
                "unit is not a nonempty target-first frozen serialization"
            )
        if self.unit_type not in {
            "sentence",
            "item",
            "cell",
            "header_cell",
            "table_caption",
        }:
            raise FeverousP6E2AcquisitionError("unit type is invalid")
        if not isinstance(self.sidecar, Mapping):
            raise FeverousP6E2AcquisitionError("unit sidecar is invalid")
        sidecar = dict(self.sidecar)
        supplied = set(sidecar)
        if not _REQUIRED_ATOMIC_SIDECAR_KEYS.issubset(supplied) or not supplied.issubset(
            _ATOMIC_SIDECAR_KEYS
        ):
            raise FeverousP6E2AcquisitionError(
                "unit sidecar is outside the frozen source-only schema"
            )
        if sidecar.get("linearizer_version") != ATOMIC_LINEARIZER_VERSION:
            raise FeverousP6E2AcquisitionError(
                "unit is not bound to the frozen atomic linearizer"
            )
        page = sidecar.get("page")
        local_id = sidecar.get("local_id")
        _safe_text(page, "unit sidecar page")
        _safe_text(local_id, "unit sidecar local id")
        if sidecar.get("unit_type") != self.unit_type:
            raise FeverousP6E2AcquisitionError(
                "unit type disagrees with its typed sidecar"
            )
        try:
            parsed = parse_element_id(self.unit_key)
        except FeverousWikipediaQualificationError as exc:
            raise FeverousP6E2AcquisitionError(
                "unit key is not an exact FEVEROUS atomic identity"
            ) from exc
        if (
            parsed.page != page
            or parsed.local_id != local_id
            or parsed.kind != self.unit_type
        ):
            raise FeverousP6E2AcquisitionError(
                "unit key disagrees with its exact page/local sidecar"
            )

        frozen_sidecar: dict[str, Any] = {}
        for key, value in sidecar.items():
            if isinstance(value, list):
                value = tuple(value)
            elif isinstance(value, Mapping):
                raise FeverousP6E2AcquisitionError(
                    "nested unit sidecar objects are not permitted"
                )
            frozen_sidecar[key] = value
        canonical_json_bytes(frozen_sidecar)
        object.__setattr__(self, "sidecar", MappingProxyType(frozen_sidecar))

    @property
    def page(self) -> str:
        value = self.sidecar["page"]
        assert isinstance(value, str)
        return value

    @property
    def local_id(self) -> str:
        value = self.sidecar["local_id"]
        assert isinstance(value, str)
        return value


@dataclass(frozen=True)
class CorpusIdentity:
    """Lightweight eligible atomic identity from the complete DB universe."""

    unit_key: str
    page: str
    local_id: str
    unit_type: str
    official_ordinal: int
    target_sha256: str

    def __post_init__(self) -> None:
        _safe_text(self.unit_key, "identity unit key")
        _safe_text(self.page, "identity page")
        _safe_text(self.local_id, "identity local id")
        if self.unit_type not in {
            "sentence",
            "item",
            "cell",
            "header_cell",
            "table_caption",
        }:
            raise FeverousP6E2AcquisitionError("identity unit type is invalid")
        if type(self.official_ordinal) is not int or self.official_ordinal < 0:
            raise FeverousP6E2AcquisitionError(
                "identity official ordinal is invalid"
            )
        if _SHA256.fullmatch(self.target_sha256) is None:
            raise FeverousP6E2AcquisitionError("identity target digest is invalid")
        try:
            parsed = parse_element_id(self.unit_key)
        except FeverousWikipediaQualificationError as exc:
            raise FeverousP6E2AcquisitionError(
                "identity key is not an exact FEVEROUS atomic identity"
            ) from exc
        if (
            parsed.page != self.page
            or parsed.local_id != self.local_id
            or parsed.kind != self.unit_type
        ):
            raise FeverousP6E2AcquisitionError(
                "identity key disagrees with page/local/type"
            )

    @property
    def commitment_row(self) -> list[Any]:
        return [
            self.page,
            self.local_id,
            self.unit_type,
            self.official_ordinal,
            self.target_sha256,
        ]


@dataclass(order=False)
class _ReverseIdentityRank:
    rank: tuple[bytes, str, str, str]
    identity: CorpusIdentity = field(compare=False)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _ReverseIdentityRank):
            return NotImplemented
        return self.rank > other.rank


@dataclass(frozen=True)
class CorpusSelectionPlan:
    """Private bounded result of the complete lightweight identity scan."""

    identities: tuple[CorpusIdentity, ...]
    selected_page_ids: tuple[str, ...]
    qualification_page_ids: tuple[str, ...]
    identity_stream_receipt: Mapping[str, Any]
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if len(self.identities) != CORPUS_UNIT_COUNT:
            raise FeverousP6E2AcquisitionError("identity plan shape drifted")
        if len({row.unit_key for row in self.identities}) != CORPUS_UNIT_COUNT:
            raise FeverousP6E2AcquisitionError("identity plan is not unique")
        expected_pages = tuple(
            sorted({row.page for row in self.identities}, key=lambda value: value.encode("utf-8"))
        )
        if self.selected_page_ids != expected_pages:
            raise FeverousP6E2AcquisitionError("identity plan page set drifted")
        if (
            tuple(
                sorted(
                    set(self.qualification_page_ids),
                    key=lambda value: value.encode("utf-8"),
                )
            )
            != self.qualification_page_ids
            or len(self.qualification_page_ids) > 64
        ):
            raise FeverousP6E2AcquisitionError(
                "identity/compiler qualification page set drifted"
            )

    @property
    def plan_sha256(self) -> str:
        value = self.receipt.get("corpus_identity_plan_sha256")
        if not isinstance(value, str):
            raise FeverousP6E2AcquisitionError("identity plan receipt is absent")
        return value

    @property
    def full_compile_page_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                set(self.selected_page_ids).union(self.qualification_page_ids),
                key=lambda value: value.encode("utf-8"),
            )
        )


@dataclass(order=False)
class _ReverseCorpusRank:
    """Heap wrapper whose smallest element is the worst retained HMAC rank."""

    rank: tuple[bytes, str, str, str]
    unit: CorpusUnit = field(compare=False)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _ReverseCorpusRank):
            return NotImplemented
        return self.rank > other.rank


def _choose_evidence_set(
    record: CandidateRecord, secret: bytes
) -> tuple[str, ...]:
    rows = []
    for evidence in record.evidence_sets:
        canonical = tuple(sorted(evidence))
        commitment = stable_hash(list(canonical))
        rows.append(
            (
                hmac_digest(
                    secret,
                    "canonical_evidence_set",
                    record.source_key,
                    commitment,
                ),
                commitment,
                canonical,
            )
        )
    return min(rows, key=lambda row: (row[0], row[1]))[2]


def select_private_blocks(
    candidates: Sequence[CandidateRecord], secret: bytes
) -> tuple[dict[str, tuple[AssignedRecord, ...]], dict[str, Any]]:
    """Select all four blocks in one family/verdict-balanced HMAC pass."""

    rows = tuple(candidates)
    if not rows:
        raise FeverousP6E2AcquisitionError("candidate pool is empty")
    source_counts = Counter(row.source_key for row in rows)
    if any(count != 1 for count in source_counts.values()):
        raise FeverousP6E2AcquisitionError("source key is not unique")

    claims: dict[str, list[CandidateRecord]] = defaultdict(list)
    for row in rows:
        claims[row.normalized_claim].append(row)
    collision_claims = {key for key, group in claims.items() if len(group) > 1}
    eligible = [
        row for row in rows if row.normalized_claim not in collision_claims
    ]

    grouped: dict[tuple[str, str], list[CandidateRecord]] = defaultdict(list)
    canonical: dict[str, tuple[str, ...]] = {}
    for row in eligible:
        canonical[row.source_key] = _choose_evidence_set(row, secret)
        grouped[(row.family, row.verdict)].append(row)

    blocks: dict[str, list[AssignedRecord]] = {
        block: [] for block in BLOCK_ORDER
    }
    required_per_verdict = {
        block: PER_FAMILY[block] // 2 for block in BLOCK_ORDER
    }
    for family in FAMILIES:
        for verdict in VERDICTS:
            pool = grouped[(family, verdict)]
            ordered = sorted(
                pool,
                key=lambda row: (
                    hmac_digest(
                        secret,
                        "candidate_order",
                        family,
                        verdict,
                        row.source_key,
                        stable_hash(row.normalized_claim),
                        stable_hash(list(canonical[row.source_key])),
                    ),
                    row.source_key,
                ),
            )
            needed = sum(required_per_verdict.values())
            if len(ordered) < needed:
                raise FeverousP6E2AcquisitionError(
                    "candidate capacity is below a frozen family/verdict quota"
                )
            cursor = 0
            for block in BLOCK_ORDER:
                take = required_per_verdict[block]
                for row in ordered[cursor : cursor + take]:
                    blocks[block].append(
                        AssignedRecord(
                            block=block,
                            ordinal=-1,
                            record=row,
                            canonical_gold_keys=canonical[row.source_key],
                        )
                    )
                cursor += take

    finalized: dict[str, tuple[AssignedRecord, ...]] = {}
    for block in BLOCK_ORDER:
        ordered = sorted(
            blocks[block],
            key=lambda assigned: (
                hmac_digest(
                    secret,
                    "block_order",
                    block,
                    assigned.record.source_key,
                ),
                assigned.record.source_key,
            ),
        )
        finalized[block] = tuple(
            AssignedRecord(
                block=block,
                ordinal=ordinal,
                record=row.record,
                canonical_gold_keys=row.canonical_gold_keys,
            )
            for ordinal, row in enumerate(ordered)
        )

    selected = [row for block in BLOCK_ORDER for row in finalized[block]]
    if (
        any(len(finalized[block]) != BLOCK_COUNTS[block] for block in BLOCK_ORDER)
        or len({row.record.source_key for row in selected}) != len(selected)
        or len({row.record.normalized_claim for row in selected})
        != len(selected)
    ):
        raise FeverousP6E2AcquisitionError("selected block invariants drifted")
    counts = {
        block: {
            family: {
                verdict: sum(
                    row.record.family == family
                    and row.record.verdict == verdict
                    for row in finalized[block]
                )
                for verdict in VERDICTS
            }
            for family in FAMILIES
        }
        for block in BLOCK_ORDER
    }
    return finalized, {
        "candidate_count": len(rows),
        "normalized_claim_collision_group_count": len(collision_claims),
        "normalized_claim_collision_record_count": sum(
            len(claims[key]) for key in collision_claims
        ),
        "post_collision_candidate_count": len(eligible),
        "selected_block_counts": dict(BLOCK_COUNTS),
        "selected_family_verdict_counts": counts,
        "selection_uses_private_HMAC_only": True,
        "score_model_or_online_evaluator_calls": 0,
    }


def _validated_selected_rows(
    blocks: Mapping[str, Sequence[AssignedRecord]],
) -> list[AssignedRecord]:
    if not isinstance(blocks, Mapping) or set(blocks) != set(BLOCK_ORDER):
        raise FeverousP6E2AcquisitionError("block set is incomplete")
    selected: list[AssignedRecord] = []
    for block in BLOCK_ORDER:
        rows = blocks[block]
        if (
            isinstance(rows, (str, bytes, bytearray))
            or len(rows) != BLOCK_COUNTS[block]
        ):
            raise FeverousP6E2AcquisitionError("selected block shape drifted")
        for ordinal, row in enumerate(rows):
            if (
                not isinstance(row, AssignedRecord)
                or row.block != block
                or row.ordinal != ordinal
            ):
                raise FeverousP6E2AcquisitionError(
                    "selected block identity or ordinal drifted"
                )
            selected.append(row)
    if (
        len({row.record.source_key for row in selected}) != len(selected)
        or len({row.record.normalized_claim for row in selected}) != len(selected)
    ):
        raise FeverousP6E2AcquisitionError("selected block uniqueness drifted")
    return selected


def build_fixed_corpus(
    *,
    blocks: Mapping[str, Sequence[AssignedRecord]],
    units: Iterable[CorpusUnit],
    secret: bytes,
) -> tuple[tuple[CorpusUnit, ...], dict[str, int], dict[str, Any]]:
    """Include canonical gold, exclude alternatives, then HMAC-fill units."""

    selected = _validated_selected_rows(blocks)
    gold_keys = {
        key for row in selected for key in row.canonical_gold_keys
    }
    official_keys = {
        key
        for row in selected
        for key in row.record.all_official_evidence_keys
    }
    forbidden_alternatives = official_keys.difference(gold_keys)
    needed = CORPUS_UNIT_COUNT - len(gold_keys)
    if needed < 0:
        raise FeverousP6E2AcquisitionError(
            "fixed corpus capacity is unavailable"
        )

    gold_by_key: dict[str, CorpusUnit] = {}
    distractor_heap: list[_ReverseCorpusRank] = []
    stream_hasher = hashlib.sha256()
    stream_count = 0
    previous_page: str | None = None
    previous_official_ordinal = -1
    for row in units:
        if not isinstance(row, CorpusUnit):
            raise FeverousP6E2AcquisitionError(
                "source stream contains a non-corpus unit"
            )
        official_ordinal = row.sidecar.get("official_ordinal")
        if type(official_ordinal) is not int or official_ordinal < 0:
            raise FeverousP6E2AcquisitionError(
                "source stream lacks an exact official ordinal"
            )
        if previous_page is not None and (
            row.page < previous_page
            or (
                row.page == previous_page
                and official_ordinal <= previous_official_ordinal
            )
        ):
            raise FeverousP6E2AcquisitionError(
                "source stream is not strict page/official order"
            )
        previous_official_ordinal = (
            official_ordinal if row.page == previous_page else official_ordinal
        )
        previous_page = row.page
        stream_count += 1
        stream_hasher.update(
            canonical_json_bytes(
                [
                    row.page,
                    row.local_id,
                    official_ordinal,
                    row.unit_type,
                    hashlib.sha256(row.text.encode("utf-8")).hexdigest(),
                    stable_hash(dict(row.sidecar)),
                ]
            )
        )
        stream_hasher.update(b"\n")
        if row.unit_key in gold_keys:
            if row.unit_key in gold_by_key:
                raise FeverousP6E2AcquisitionError(
                    "canonical gold unit is duplicated in the source stream"
                )
            gold_by_key[row.unit_key] = row
            continue
        if row.unit_key in forbidden_alternatives:
            continue
        rank = (
            hmac_digest(
                secret,
                "distractor_order",
                row.page,
                row.local_id,
            ),
            row.page,
            row.local_id,
            row.unit_key,
        )
        entry = _ReverseCorpusRank(rank=rank, unit=row)
        if len(distractor_heap) < needed:
            heapq.heappush(distractor_heap, entry)
        elif needed and rank < distractor_heap[0].rank:
            heapq.heapreplace(distractor_heap, entry)

    missing_gold = gold_keys.difference(gold_by_key)
    if missing_gold or len(distractor_heap) != needed:
        raise FeverousP6E2AcquisitionError(
            "fixed corpus capacity or canonical gold is unavailable"
        )
    distractors = [
        entry.unit for entry in sorted(distractor_heap, key=lambda row: row.rank)
    ]
    chosen = [gold_by_key[key] for key in sorted(gold_keys)] + distractors
    chosen.sort(
        key=lambda row: (
            hmac_digest(secret, "final_corpus_order", row.unit_key),
            row.unit_key,
        )
    )
    index = {row.unit_key: ordinal for ordinal, row in enumerate(chosen)}
    if len(chosen) != CORPUS_UNIT_COUNT or len(index) != CORPUS_UNIT_COUNT:
        raise FeverousP6E2AcquisitionError("fixed corpus shape drifted")
    return tuple(chosen), index, {
        "fixed_atomic_unit_count": len(chosen),
        "unique_canonical_gold_unit_count": len(gold_keys),
        "distractor_unit_count": needed,
        "source_atomic_unit_scan_count": stream_count,
        "source_atomic_unit_stream_sha256": stream_hasher.hexdigest(),
        "source_stream_order": "ascending_exact_page_then_official_ordinal",
        "known_noncanonical_official_evidence_excluded": len(
            forbidden_alternatives
        ),
        "all_selected_canonical_gold_included": True,
        "gold_origin_serialized_in_corpus": False,
        "shared_all_blocks_and_arms": True,
        # This legacy full-CorpusUnit stream remains useful for synthetic and
        # counterfactual diagnostics.  It is intentionally never a formal
        # 5.4M-page source proof: that path is the two-phase identity scan.
        "formal_source_stream_receipt_sha256": None,
        "formal_source_bound": False,
        "formal_acquisition_valid": False,
    }


def _receipt_from_exhausted_stream(
    stream: object, *, label: str
) -> Mapping[str, Any]:
    provider = getattr(stream, "aggregate_receipt", None)
    if not callable(provider):
        raise FeverousP6E2AcquisitionError(
            f"{label} does not expose an exhaustion receipt"
        )
    try:
        receipt = provider()
    except Exception as exc:
        raise FeverousP6E2AcquisitionError(
            f"{label} was not normally exhausted"
        ) from exc
    if not isinstance(receipt, Mapping):
        raise FeverousP6E2AcquisitionError(f"{label} receipt is not an object")
    return receipt


def _verify_identity_stream_receipt(
    receipt: Mapping[str, Any],
    *,
    observed_count: int,
    require_formal_source: bool,
) -> str:
    try:
        declared = verify_self_hash(receipt, "corpus_identity_stream_receipt_sha256")
    except FeverousP6E2AcquisitionError as exc:
        raise FeverousP6E2AcquisitionError(
            "identity stream receipt self-hash drifted"
        ) from exc
    required_sha_fields = (
        "database_page_stream_receipt_sha256",
        "formal_source_opener_source_sha256",
        "database_file_sha256",
        "logical_page_stream_sha256",
        "eligible_atomic_identity_stream_sha256",
        "atomic_compiler_source_sha256",
        "identity_enumerator_source_sha256",
        "source_adapter_source_sha256",
        "acquisition_source_sha256",
        "identity_full_compile_equivalence_qualification_sha256",
        "real_identity_compiler_sample_page_set_sha256",
    )
    if (
        receipt.get("schema") != IDENTITY_STREAM_RECEIPT_SCHEMA
        or receipt.get("status")
        != "complete_atomic_identity_universe_exhausted_no_selection"
        or receipt.get("source_split") != "TRAIN"
        or receipt.get("stream_fully_exhausted") is not True
        or receipt.get("eligible_atomic_identity_count") != observed_count
        or receipt.get("all_identities_or_pages_materialized") is not False
        or receipt.get("full_atomic_text_or_sidecar_linearized") is not False
        or receipt.get("real_identity_compiler_sample_policy")
        != "lowest_sha256_domain_page_id_then_binary_page_id"
        or type(receipt.get("real_identity_compiler_sample_page_count")) is not int
        or receipt["real_identity_compiler_sample_page_count"] < 0
        or receipt.get("cohort_canonical_set_or_fixed_corpus_selected") is not False
        or receipt.get("development_or_test_source_accessed") is not False
        or receipt.get("online_evaluator_calls") != 0
        or any(not isinstance(receipt.get(field), str) or _SHA256.fullmatch(receipt[field]) is None for field in required_sha_fields)
    ):
        raise FeverousP6E2AcquisitionError("identity stream receipt drifted")
    if require_formal_source and (
        receipt.get("formal_source") is not True
        or receipt.get("database_size_bytes") != FORMAL_DATABASE_SIZE_BYTES
        or receipt.get("database_file_sha256") != FORMAL_DATABASE_SHA256
        or receipt.get("expected_database_row_count") != FORMAL_DATABASE_ROW_COUNT
        or receipt.get("observed_database_row_count") != FORMAL_DATABASE_ROW_COUNT
        or receipt.get("real_identity_compiler_sample_page_count") != 64
    ):
        raise FeverousP6E2AcquisitionError(
            "partial or synthetic identity source is not formal-valid"
        )
    return declared


def plan_fixed_corpus_from_identity_stream(
    *,
    blocks: Mapping[str, Sequence[AssignedRecord]],
    identities: Iterable[CorpusIdentity],
    secret: bytes,
    require_formal_source: bool = True,
) -> CorpusSelectionPlan:
    """Scan the entire lightweight universe and retain only gold + heap IDs.

    The iterator must expose an aggregate receipt after normal exhaustion.  At
    most 8192 distractor identities and the selected gold identities remain
    resident; no complete page, unit text, sidecar, or all-ID set is retained.
    """

    selected = _validated_selected_rows(blocks)
    gold_keys = {key for row in selected for key in row.canonical_gold_keys}
    official_keys = {
        key
        for row in selected
        for key in row.record.all_official_evidence_keys
    }
    forbidden_alternatives = official_keys.difference(gold_keys)
    needed = CORPUS_UNIT_COUNT - len(gold_keys)
    if needed < 0:
        raise FeverousP6E2AcquisitionError("fixed corpus capacity is unavailable")

    source_object = identities
    gold_by_key: dict[str, CorpusIdentity] = {}
    distractor_heap: list[_ReverseIdentityRank] = []
    stream_count = 0
    previous_page: str | None = None
    previous_ordinal = -1
    for identity in identities:
        if not isinstance(identity, CorpusIdentity):
            raise FeverousP6E2AcquisitionError(
                "identity stream contains a non-identity row"
            )
        if identity.page == previous_page:
            if identity.official_ordinal <= previous_ordinal:
                raise FeverousP6E2AcquisitionError(
                    "identity stream is not strict within-page official order"
                )
        else:
            previous_ordinal = -1
        previous_page = identity.page
        previous_ordinal = identity.official_ordinal
        stream_count += 1
        if identity.unit_key in gold_keys:
            if identity.unit_key in gold_by_key:
                raise FeverousP6E2AcquisitionError(
                    "canonical gold identity is duplicated"
                )
            gold_by_key[identity.unit_key] = identity
            continue
        if identity.unit_key in forbidden_alternatives:
            continue
        rank = (
            hmac_digest(
                secret,
                "distractor_order",
                identity.page,
                identity.local_id,
            ),
            identity.page,
            identity.local_id,
            identity.unit_key,
        )
        entry = _ReverseIdentityRank(rank=rank, identity=identity)
        if len(distractor_heap) < needed:
            heapq.heappush(distractor_heap, entry)
        elif needed and rank < distractor_heap[0].rank:
            heapq.heapreplace(distractor_heap, entry)

    stream_receipt = _receipt_from_exhausted_stream(
        source_object, label="identity stream"
    )
    identity_receipt_sha256 = _verify_identity_stream_receipt(
        stream_receipt,
        observed_count=stream_count,
        require_formal_source=require_formal_source,
    )
    stream_commitment = str(
        stream_receipt["eligible_atomic_identity_stream_sha256"]
    )
    try:
        qualification_page_ids = tuple(source_object.qualification_page_ids)  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as exc:
        raise FeverousP6E2AcquisitionError(
            "identity stream does not expose its bounded real qualification sample"
        ) from exc
    if (
        len(qualification_page_ids)
        != stream_receipt.get("real_identity_compiler_sample_page_count")
        or stable_hash(list(qualification_page_ids))
        != stream_receipt.get("real_identity_compiler_sample_page_set_sha256")
    ):
        raise FeverousP6E2AcquisitionError(
            "real identity/compiler qualification sample receipt drifted"
        )
    missing_gold = gold_keys.difference(gold_by_key)
    if missing_gold or len(distractor_heap) != needed:
        raise FeverousP6E2AcquisitionError(
            "fixed corpus capacity or canonical gold identity is unavailable"
        )
    retained = [gold_by_key[key] for key in sorted(gold_keys)]
    retained.extend(
        entry.identity
        for entry in sorted(distractor_heap, key=lambda value: value.rank)
    )
    if len({identity.unit_key for identity in retained}) != CORPUS_UNIT_COUNT:
        raise FeverousP6E2AcquisitionError("retained identity plan is not unique")
    retained.sort(key=lambda identity: identity.unit_key)
    selected_pages = tuple(
        sorted({identity.page for identity in retained}, key=lambda value: value.encode("utf-8"))
    )
    plan_body: dict[str, Any] = {
        "schema": f"{VERSION}_corpus_identity_plan",
        "version": VERSION,
        "status": "complete_universe_scanned_bounded_identity_plan_formed",
        "identity_stream_receipt_sha256": identity_receipt_sha256,
        "formal_source_bound": bool(stream_receipt.get("formal_source")),
        "complete_identity_scan_count": stream_count,
        "complete_identity_stream_sha256": stream_commitment,
        "identity_stream_commitment_verified_from_exhausted_adapter_receipt": True,
        "selected_atomic_identity_count": len(retained),
        "selected_page_count": len(selected_pages),
        "real_identity_compiler_qualification_page_count": len(
            qualification_page_ids
        ),
        "real_identity_compiler_qualification_page_set_sha256": stable_hash(
            list(qualification_page_ids)
        ),
        "unique_canonical_gold_identity_count": len(gold_keys),
        "distractor_identity_count": needed,
        "known_noncanonical_official_evidence_excluded": len(
            forbidden_alternatives
        ),
        "maximum_retained_distractor_identities": CORPUS_UNIT_COUNT,
        "all_identity_keys_or_page_ids_serialized": False,
        "full_atomic_text_or_sidecar_linearized_during_scan": False,
        "development_or_test_source_accessed": False,
        "online_evaluator_calls": 0,
    }
    plan_receipt = MappingProxyType(
        self_hashed(plan_body, "corpus_identity_plan_sha256")
    )
    return CorpusSelectionPlan(
        identities=tuple(retained),
        selected_page_ids=selected_pages,
        qualification_page_ids=qualification_page_ids,
        identity_stream_receipt=MappingProxyType(dict(stream_receipt)),
        receipt=plan_receipt,
    )


def _verify_materialization_receipt(
    receipt: Mapping[str, Any],
    *,
    plan: CorpusSelectionPlan,
    selected_hit_count: int,
    observed_crosscheck_sha256: str,
    require_formal_source: bool,
) -> str:
    try:
        declared = verify_self_hash(
            receipt, "selected_corpus_materialization_receipt_sha256"
        )
    except FeverousP6E2AcquisitionError as exc:
        raise FeverousP6E2AcquisitionError(
            "selected materialization receipt self-hash drifted"
        ) from exc
    sha_fields = (
        "selected_page_lookup_receipt_sha256",
        "formal_source_opener_source_sha256",
        "database_page_stream_receipt_sha256",
        "corpus_identity_plan_sha256",
        "atomic_compiler_source_sha256",
        "selected_compiled_unit_stream_sha256",
        "selected_identity_compiler_crosscheck_sha256",
        "identity_full_compile_qualification_page_set_sha256",
    )
    if (
        receipt.get("schema") != MATERIALIZATION_RECEIPT_SCHEMA
        or receipt.get("status")
        != "selected_pages_fully_compiled_and_identity_crosschecked"
        or receipt.get("source_split") != "TRAIN"
        or receipt.get("corpus_identity_plan_sha256") != plan.plan_sha256
        or receipt.get("selected_identity_hit_count") != selected_hit_count
        or receipt.get("selected_identity_hit_count") != CORPUS_UNIT_COUNT
        or receipt.get("selected_identity_compiler_crosscheck_sha256")
        != observed_crosscheck_sha256
        or receipt.get("atomic_compiler_source_sha256")
        != plan.identity_stream_receipt.get("atomic_compiler_source_sha256")
        or receipt.get("formal_source_opener_source_sha256")
        != plan.identity_stream_receipt.get(
            "formal_source_opener_source_sha256"
        )
        or receipt.get("selected_page_count") != len(plan.full_compile_page_ids)
        or receipt.get("selected_identity_page_count")
        != len(plan.selected_page_ids)
        or receipt.get("identity_full_compile_qualification_page_count")
        != len(plan.qualification_page_ids)
        or receipt.get("identity_full_compile_qualification_page_set_sha256")
        != stable_hash(list(plan.qualification_page_ids))
        or receipt.get("identity_full_compile_crosschecked_page_count")
        != len(plan.full_compile_page_ids)
        or receipt.get("selected_page_lookup_fully_exhausted") is not True
        or receipt.get("full_database_rescan") is not False
        or receipt.get("development_or_test_source_accessed") is not False
        or receipt.get("online_evaluator_calls") != 0
        or any(not isinstance(receipt.get(field), str) or _SHA256.fullmatch(receipt[field]) is None for field in sha_fields)
    ):
        raise FeverousP6E2AcquisitionError("materialization receipt drifted")
    if require_formal_source and (
        receipt.get("formal_source") is not True
        or receipt.get("identity_full_compile_qualification_page_count") != 64
    ):
        raise FeverousP6E2AcquisitionError(
            "synthetic selected-page materialization is not formal-valid"
        )
    return declared


def materialize_fixed_corpus_from_selection_plan(
    *,
    plan: CorpusSelectionPlan,
    units: Iterable[CorpusUnit],
    secret: bytes,
    require_formal_source: bool = True,
) -> tuple[tuple[CorpusUnit, ...], dict[str, int], dict[str, Any]]:
    """Full-compile only selected pages and verify every retained identity."""

    if not isinstance(plan, CorpusSelectionPlan):
        raise FeverousP6E2AcquisitionError("identity selection plan is absent")
    identity_by_key = {row.unit_key: row for row in plan.identities}
    source_object = units
    materialized: dict[str, CorpusUnit] = {}
    selected_crosscheck_hasher = hashlib.sha256()
    for unit in units:
        if not isinstance(unit, CorpusUnit):
            raise FeverousP6E2AcquisitionError(
                "selected materialization contains a non-corpus unit"
            )
        expected = identity_by_key.get(unit.unit_key)
        if expected is None:
            continue
        if unit.unit_key in materialized:
            raise FeverousP6E2AcquisitionError(
                "selected corpus unit was materialized more than once"
            )
        official_ordinal = unit.sidecar.get("official_ordinal")
        first_line = unit.text.split("\n", 1)[0]
        if not first_line.startswith("TARGET: "):
            raise FeverousP6E2AcquisitionError("compiled target prefix drifted")
        target = first_line[len("TARGET: ") :]
        target_sha256 = hashlib.sha256(target.encode("utf-8")).hexdigest()
        observed = CorpusIdentity(
            unit_key=unit.unit_key,
            page=unit.page,
            local_id=unit.local_id,
            unit_type=unit.unit_type,
            official_ordinal=official_ordinal,
            target_sha256=target_sha256,
        )
        if observed != expected:
            raise FeverousP6E2AcquisitionError(
                "lightweight identity differs from frozen full compiler"
            )
        encoded = canonical_json_bytes(observed.commitment_row)
        selected_crosscheck_hasher.update(len(encoded).to_bytes(8, "big"))
        selected_crosscheck_hasher.update(encoded)
        materialized[unit.unit_key] = unit
    materialization_receipt = _receipt_from_exhausted_stream(
        source_object, label="selected full-compiler stream"
    )
    materialization_receipt_sha256 = _verify_materialization_receipt(
        materialization_receipt,
        plan=plan,
        selected_hit_count=len(materialized),
        observed_crosscheck_sha256=selected_crosscheck_hasher.hexdigest(),
        require_formal_source=require_formal_source,
    )
    if set(materialized) != set(identity_by_key):
        raise FeverousP6E2AcquisitionError(
            "selected full compiler did not materialize every identity"
        )
    chosen = [materialized[row.unit_key] for row in plan.identities]
    chosen.sort(
        key=lambda row: (
            hmac_digest(secret, "final_corpus_order", row.unit_key),
            row.unit_key,
        )
    )
    index = {row.unit_key: ordinal for ordinal, row in enumerate(chosen)}
    if len(chosen) != CORPUS_UNIT_COUNT or len(index) != CORPUS_UNIT_COUNT:
        raise FeverousP6E2AcquisitionError("fixed corpus shape drifted")
    corpus_hasher = hashlib.sha256()
    for unit in chosen:
        encoded = canonical_json_bytes(
            [
                unit.unit_key,
                hashlib.sha256(unit.text.encode("utf-8")).hexdigest(),
                stable_hash(dict(unit.sidecar)),
            ]
        )
        corpus_hasher.update(len(encoded).to_bytes(8, "big"))
        corpus_hasher.update(encoded)
    formal_bound = bool(
        plan.receipt.get("formal_source_bound")
        and materialization_receipt.get("formal_source")
    )
    stats = {
        "fixed_atomic_unit_count": len(chosen),
        "source_atomic_identity_scan_count": plan.receipt[
            "complete_identity_scan_count"
        ],
        "source_atomic_identity_stream_sha256": plan.receipt[
            "complete_identity_stream_sha256"
        ],
        "source_identity_stream_receipt_sha256": plan.receipt[
            "identity_stream_receipt_sha256"
        ],
        "corpus_identity_plan_sha256": plan.plan_sha256,
        "selected_corpus_materialization_receipt_sha256": materialization_receipt_sha256,
        "selected_identity_compiler_crosscheck_sha256": selected_crosscheck_hasher.hexdigest(),
        "fixed_compiled_corpus_sha256": corpus_hasher.hexdigest(),
        "formal_source_bound": formal_bound,
        "full_universe_linearized_or_sidecars_built": False,
        "selected_pages_only_full_compiled": True,
        "all_selected_canonical_gold_included": True,
        "gold_origin_serialized_in_corpus": False,
        "shared_all_blocks_and_arms": True,
        "formal_acquisition_valid": formal_bound and require_formal_source,
    }
    return tuple(chosen), index, stats


def verify_formal_corpus_acquisition(stats: Mapping[str, Any]) -> str:
    """Reject legacy, partial, synthetic, or unbound corpus acquisitions."""

    sha_fields = (
        "source_atomic_identity_stream_sha256",
        "source_identity_stream_receipt_sha256",
        "corpus_identity_plan_sha256",
        "selected_corpus_materialization_receipt_sha256",
        "selected_identity_compiler_crosscheck_sha256",
        "fixed_compiled_corpus_sha256",
    )
    if (
        not isinstance(stats, Mapping)
        or stats.get("fixed_atomic_unit_count") != CORPUS_UNIT_COUNT
        or stats.get("formal_source_bound") is not True
        or stats.get("formal_acquisition_valid") is not True
        or stats.get("full_universe_linearized_or_sidecars_built") is not False
        or stats.get("selected_pages_only_full_compiled") is not True
        or stats.get("all_selected_canonical_gold_included") is not True
        or any(not isinstance(stats.get(field), str) or _SHA256.fullmatch(stats[field]) is None for field in sha_fields)
    ):
        raise FeverousP6E2AcquisitionError(
            "corpus acquisition is not bound to the complete formal source"
        )
    return str(stats["fixed_compiled_corpus_sha256"])


def materialize_private_payloads(
    *,
    blocks: Mapping[str, Sequence[AssignedRecord]],
    corpus: Sequence[CorpusUnit],
    corpus_index: Mapping[str, int],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """Create one label-free corpus, claim views, and separate late labels."""

    if set(blocks) != set(BLOCK_ORDER) or len(corpus) != CORPUS_UNIT_COUNT:
        raise FeverousP6E2AcquisitionError("private materialization shape drifted")
    expected_index = {
        unit.unit_key: ordinal for ordinal, unit in enumerate(corpus)
    }
    if dict(corpus_index) != expected_index:
        raise FeverousP6E2AcquisitionError(
            "corpus index is not the exact corpus enumeration"
        )
    corpus_view = self_hashed(
        {
            "schema": CORPUS_VIEW_SCHEMA,
            "version": VERSION,
            "unit_count": len(corpus),
            "gold_origin_or_membership_included": False,
            "units": [
                {
                    "unit_i": ordinal,
                    "text": unit.text,
                    "unit_type": unit.unit_type,
                    "sidecar": dict(unit.sidecar),
                }
                for ordinal, unit in enumerate(corpus)
            ],
        },
        "corpus_view_sha256",
    )
    views: dict[str, dict[str, Any]] = {}
    labels: dict[str, dict[str, Any]] = {}
    for block in BLOCK_ORDER:
        assigned = tuple(blocks[block])
        if len(assigned) != BLOCK_COUNTS[block]:
            raise FeverousP6E2AcquisitionError("private block count drifted")
        views[block] = self_hashed(
            {
                "schema": BLOCK_VIEW_SCHEMA,
                "version": VERSION,
                "item_count": len(assigned),
                "late_label_fields_included": False,
                "items": [
                    {"claim": row.record.claim}
                    for row in assigned
                ],
            },
            "block_view_sha256",
        )
        if block == "F_search":
            continue
        label_rows = []
        for row in assigned:
            try:
                gold = sorted(corpus_index[key] for key in row.canonical_gold_keys)
            except KeyError as exc:
                raise FeverousP6E2AcquisitionError(
                    "gold unit is absent from corpus index"
                ) from exc
            label_rows.append(
                {
                    "ordinal": row.ordinal,
                    "gold_unit_indices": gold,
                    "family": row.record.family,
                    "verdict": row.record.verdict,
                }
            )
        labels[block] = self_hashed(
            {
                "schema": BLOCK_LABEL_SCHEMA,
                "version": VERSION,
                "block": block,
                "item_count": len(label_rows),
                "items": label_rows,
            },
            "block_labels_sha256",
        )
    if set(labels) != {"A_form", "A_hold", "M_search"}:
        raise FeverousP6E2AcquisitionError(
            "late-label pack set drifted"
        )
    return corpus_view, views, labels


__all__ = [
    "BLOCK_COUNTS",
    "BLOCK_LABEL_SCHEMA",
    "BLOCK_ORDER",
    "BLOCK_VIEW_SCHEMA",
    "CORPUS_UNIT_COUNT",
    "CORPUS_VIEW_SCHEMA",
    "CandidateRecord",
    "AssignedRecord",
    "CorpusUnit",
    "FAMILIES",
    "FeverousP6E2AcquisitionError",
    "PER_FAMILY",
    "VERDICTS",
    "build_fixed_corpus",
    "canonical_json_bytes",
    "hmac_digest",
    "materialize_private_payloads",
    "normalize_claim",
    "select_private_blocks",
    "self_hashed",
    "stable_hash",
    "verify_self_hash",
]
