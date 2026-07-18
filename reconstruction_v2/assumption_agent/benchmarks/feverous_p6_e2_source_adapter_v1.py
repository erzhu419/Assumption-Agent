"""Outcome-blind FEVEROUS TRAIN source adapter for the P6/E2 study.

The adapter is the only bridge from already-qualified official annotation
records and Wikipedia pages to the private acquisition data classes.  It does
not select a cohort, assign blocks, choose a canonical evidence set, score a
recipe, or accept any selection secret.  Its public receipt is aggregate-only;
claims, source ids, evidence ids, page ids, page text, and per-record digests
remain absent.

Every evidence reference is resolved through :class:`FeverousWikiResolver`
using the single exact title in its official context.  The same page is also
compiled by the frozen atomic compiler, and resolver target text must agree
with the compiler target after the frozen NFKC/whitespace normalization.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import heapq
import inspect
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

from assumption_agent.benchmarks import feverous_atomic_corpus_v1 as atomic_corpus
from assumption_agent.benchmarks import feverous_p6_e2_acquisition_v1 as acquisition_core
from assumption_agent.benchmarks.feverous_atomic_corpus_v1 import (
    VERSION as ATOMIC_COMPILER_VERSION,
    PageCompilation,
    compile_official_page,
    normalize_surface,
)
from assumption_agent.benchmarks.feverous_p6_e2_acquisition_v1 import (
    CORPUS_UNIT_COUNT,
    CORPUS_VIEW_SCHEMA,
    FAMILIES,
    VERDICTS,
    VERSION as ACQUISITION_VERSION,
    CandidateRecord,
    CorpusIdentity,
    CorpusSelectionPlan,
    CorpusUnit,
)
from assumption_agent.benchmarks.feverous_wikipedia_source_qualification_v1 import (
    CONTENT_KINDS,
    CONTEXT_KINDS,
    VERSION as WIKIPEDIA_RESOLVER_VERSION,
    FeverousWikiResolver,
    FeverousWikipediaQualificationError,
    parse_element_id,
)


VERSION = "feverous_p6_e2_source_adapter_v1"
RECEIPT_SCHEMA = f"{VERSION}_aggregate_receipt"
STREAM_RECEIPT_SCHEMA = f"{VERSION}_corpus_stream_receipt"
IDENTITY_STREAM_RECEIPT_SCHEMA = f"{VERSION}_corpus_identity_stream_receipt"
MATERIALIZATION_RECEIPT_SCHEMA = (
    f"{VERSION}_selected_corpus_materialization_receipt"
)
REAL_IDENTITY_COMPILER_SAMPLE_PAGE_COUNT = 64

DESIGN_SHA256 = "6193646baca9e35820a5d157bc248012fbd478c89a45db7d879295c4d64f0181"
ANNOTATION_QUALIFICATION_SHA256 = (
    "fa34114eb83fdd3b132799346be41aa0ebcf75f7c7e3344c3c59e9e2ae3b92e3"
)
WIKIPEDIA_QUALIFICATION_SHA256 = (
    "18c1ed8af7c2bef9232d70f2c6e5e5669c50fad0a071b851b8045dd09764a01d"
)

OFFICIAL_RECORD_FIELDS = frozenset(
    {
        "annotator_operations",
        "challenge",
        "claim",
        "evidence",
        "id",
        "label",
    }
)
STRUCTURED_TYPES = frozenset({"cell", "header_cell", "table_caption"})
TEXT_TYPES = frozenset({"sentence", "item"})
_SHA256_FIELDS = (
    "design_sha256",
    "annotation_qualification_sha256",
    "wikipedia_qualification_sha256",
)
_RECORD_STATUSES = frozenset(
    {
        "blank_sentinel",
        "unsupported_family",
        "unsupported_verdict",
        "no_eligible_canonical_set",
        "eligible_candidate",
    }
)
_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "version",
        "status",
        "source_split",
        "source_binding_sha256",
        "design_sha256",
        "annotation_qualification_sha256",
        "wikipedia_qualification_sha256",
        "atomic_compiler_version",
        "wikipedia_resolver_version",
        "acquisition_version",
        "downstream_corpus_view_schema",
        "downstream_fixed_atomic_unit_count",
        "input_record_count",
        "record_status_counts",
        "candidate_count",
        "official_evidence_set_count",
        "eligible_evidence_set_count",
        "official_evidence_reference_count",
        "excluded_empty_set_count",
        "excluded_cardinality_set_count",
        "excluded_family_structure_set_count",
        "adapted_page_count",
        "eligible_atomic_source_unit_count",
        "excluded_empty_atomic_source_unit_count",
        "raw_claim_page_or_evidence_serialized",
        "per_record_or_per_source_digest_serialized",
        "cohort_block_or_canonical_set_selected",
        "fixed_8192_corpus_formed",
        "utility_recipe_or_model_accessed",
        "development_or_test_source_accessed",
        "online_evaluator_calls",
        "adapter_receipt_sha256",
    }
)


class FeverousSourceAdapterError(RuntimeError):
    """The qualified-source boundary or exact topology drifted."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise FeverousSourceAdapterError(
            "adapter receipt is not canonical JSON"
        ) from exc


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _source_file_sha256(module: object) -> str:
    source = inspect.getsourcefile(module)
    if not isinstance(source, str):
        raise FeverousSourceAdapterError("runtime source file is unavailable")
    try:
        return hashlib.sha256(Path(source).read_bytes()).hexdigest()
    except OSError as exc:
        raise FeverousSourceAdapterError("runtime source file cannot be hashed") from exc


@dataclass(frozen=True)
class SourceQualificationBinding:
    """Exact public qualifications required before any private adaptation."""

    source_split: str
    design_sha256: str
    annotation_qualification_sha256: str
    wikipedia_qualification_sha256: str
    atomic_compiler_version: str
    wikipedia_resolver_version: str
    acquisition_version: str
    corpus_view_schema: str
    frozen_corpus_unit_count: int

    def __post_init__(self) -> None:
        expected = {
            "source_split": "TRAIN",
            "design_sha256": DESIGN_SHA256,
            "annotation_qualification_sha256": ANNOTATION_QUALIFICATION_SHA256,
            "wikipedia_qualification_sha256": WIKIPEDIA_QUALIFICATION_SHA256,
            "atomic_compiler_version": ATOMIC_COMPILER_VERSION,
            "wikipedia_resolver_version": WIKIPEDIA_RESOLVER_VERSION,
            "acquisition_version": ACQUISITION_VERSION,
            "corpus_view_schema": CORPUS_VIEW_SCHEMA,
            "frozen_corpus_unit_count": CORPUS_UNIT_COUNT,
        }
        if asdict(self) != expected:
            raise FeverousSourceAdapterError(
                "source qualification binding differs from the frozen TRAIN design"
            )
        for field in _SHA256_FIELDS:
            if not _is_sha256(getattr(self, field)):
                raise FeverousSourceAdapterError(
                    "source qualification binding has an invalid digest"
                )

    @property
    def binding_sha256(self) -> str:
        return _stable_hash(asdict(self))


FROZEN_TRAIN_BINDING = SourceQualificationBinding(
    source_split="TRAIN",
    design_sha256=DESIGN_SHA256,
    annotation_qualification_sha256=ANNOTATION_QUALIFICATION_SHA256,
    wikipedia_qualification_sha256=WIKIPEDIA_QUALIFICATION_SHA256,
    atomic_compiler_version=ATOMIC_COMPILER_VERSION,
    wikipedia_resolver_version=WIKIPEDIA_RESOLVER_VERSION,
    acquisition_version=ACQUISITION_VERSION,
    corpus_view_schema=CORPUS_VIEW_SCHEMA,
    frozen_corpus_unit_count=CORPUS_UNIT_COUNT,
)


@dataclass(frozen=True)
class AdaptedPage:
    """Private, compiler/resolver-agreeing atomic projection of one page."""

    page_id: str
    units: tuple[CorpusUnit, ...]
    excluded_empty_unit_keys: tuple[str, ...]
    source_binding_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.page_id, str) or not self.page_id:
            raise FeverousSourceAdapterError("adapted page id is invalid")
        if self.source_binding_sha256 != FROZEN_TRAIN_BINDING.binding_sha256:
            raise FeverousSourceAdapterError("adapted page source binding drifted")
        keys = [unit.unit_key for unit in self.units]
        if len(keys) != len(set(keys)):
            raise FeverousSourceAdapterError("adapted page has duplicate units")
        if any(unit.page != self.page_id for unit in self.units):
            raise FeverousSourceAdapterError("adapted page unit identity drifted")
        if len(self.excluded_empty_unit_keys) != len(
            set(self.excluded_empty_unit_keys)
        ):
            raise FeverousSourceAdapterError(
                "adapted page has duplicate empty exclusions"
            )

    @property
    def unit_by_key(self) -> Mapping[str, CorpusUnit]:
        return MappingProxyType({unit.unit_key: unit for unit in self.units})


@dataclass(frozen=True)
class RecordAdaptation:
    candidate: CandidateRecord | None
    status: str
    official_evidence_set_count: int
    eligible_evidence_set_count: int
    official_evidence_reference_count: int
    excluded_empty_set_count: int
    excluded_cardinality_set_count: int
    excluded_family_structure_set_count: int


@dataclass(frozen=True)
class AdapterBatch:
    """Private adapter outputs plus one aggregate, content-free receipt."""

    candidates: tuple[CandidateRecord, ...]
    corpus_units: tuple[CorpusUnit, ...]
    receipt: Mapping[str, Any]


class CorpusUnitStream(Iterator[CorpusUnit]):
    """One-pass, page-bounded compiler stream over a qualified page iterable.

    Only the current page's atomic tuple is resident.  A fresh resolver window
    is used for each page so the qualification resolver's page cache cannot
    retain millions of :class:`WikiElement` objects.  The aggregate receipt is
    available only after normal iterator exhaustion.
    """

    def __init__(
        self,
        page_rows: Iterable[tuple[str, Any]],
        *,
        resolver: FeverousWikiResolver,
        binding: SourceQualificationBinding,
    ) -> None:
        _require_binding(binding, source_split="TRAIN")
        if not isinstance(resolver, FeverousWikiResolver):
            raise FeverousSourceAdapterError("exact Wikipedia resolver is required")
        self._page_rows = iter(page_rows)
        self._connection = resolver.connection
        self._binding = binding
        self._current: Iterator[CorpusUnit] = iter(())
        self._previous_page_utf8: bytes | None = None
        self._page_count = 0
        self._unit_count = 0
        self._empty_count = 0
        self._unit_hasher = hashlib.sha256()
        self._atomic_compiler_source_sha256 = _source_file_sha256(atomic_corpus)
        self._complete = False
        self._receipt: Mapping[str, Any] | None = None

    def __iter__(self) -> "CorpusUnitStream":
        return self

    def __next__(self) -> CorpusUnit:
        while True:
            try:
                unit = next(self._current)
            except StopIteration:
                pass
            else:
                encoded = _canonical_json(
                    [
                        unit.page,
                        unit.local_id,
                        unit.sidecar.get("official_ordinal"),
                        unit.unit_type,
                        hashlib.sha256(unit.text.encode("utf-8")).hexdigest(),
                        _stable_hash(dict(unit.sidecar)),
                    ]
                )
                self._unit_hasher.update(len(encoded).to_bytes(8, "big"))
                self._unit_hasher.update(encoded)
                self._unit_count += 1
                return unit
            if self._complete:
                raise StopIteration
            try:
                row = next(self._page_rows)
            except StopIteration:
                self._complete = True
                self._receipt = MappingProxyType(
                    _corpus_stream_receipt(
                        page_count=self._page_count,
                        unit_count=self._unit_count,
                        empty_count=self._empty_count,
                        unit_stream_sha256=self._unit_hasher.hexdigest(),
                        atomic_compiler_source_sha256=(
                            self._atomic_compiler_source_sha256
                        ),
                        binding=self._binding,
                    )
                )
                raise
            if not isinstance(row, tuple) or len(row) != 2:
                raise FeverousSourceAdapterError(
                    "qualified page stream row must be a page-id/payload tuple"
                )
            page_id, raw_page = row
            if not isinstance(page_id, str) or not page_id:
                raise FeverousSourceAdapterError(
                    "qualified page stream contains an invalid page"
                )
            try:
                page_id_utf8 = page_id.encode("utf-8", errors="strict")
            except UnicodeEncodeError as exc:
                raise FeverousSourceAdapterError(
                    "qualified page id is not strict UTF-8"
                ) from exc
            if (
                self._previous_page_utf8 is not None
                and page_id_utf8 <= self._previous_page_utf8
            ):
                raise FeverousSourceAdapterError(
                    "qualified diagnostic page stream is not strict page-id order"
                )
            self._previous_page_utf8 = page_id_utf8
            # Page-local cache lifetime is bounded to this iteration step.
            window = FeverousWikiResolver(self._connection)
            adapted = adapt_qualified_page(
                page_id,
                raw_page,
                resolver=window,
                binding=self._binding,
            )
            self._page_count += 1
            self._empty_count += len(adapted.excluded_empty_unit_keys)
            self._current = iter(adapted.units)

    def aggregate_receipt(self) -> Mapping[str, Any]:
        if not self._complete or self._receipt is None:
            raise FeverousSourceAdapterError(
                "corpus stream receipt is unavailable before normal exhaustion"
            )
        return self._receipt


def _verified_upstream_receipt(
    source: object,
    *,
    hash_field: str,
    schema: str,
    status: str,
) -> Mapping[str, Any]:
    provider = getattr(source, "aggregate_receipt", None)
    if not callable(provider):
        raise FeverousSourceAdapterError(
            "formal upstream iterator does not expose an exhaustion receipt"
        )
    try:
        receipt = provider()
    except Exception as exc:
        raise FeverousSourceAdapterError(
            "formal upstream iterator was not normally exhausted"
        ) from exc
    if not isinstance(receipt, Mapping):
        raise FeverousSourceAdapterError("upstream receipt must be an object")
    body = dict(receipt)
    declared = body.pop(hash_field, None)
    if (
        not _is_sha256(declared)
        or body.get("schema") != schema
        or body.get("status") != status
        or body.get("source_split") != "TRAIN"
        or _stable_hash(body) != declared
    ):
        raise FeverousSourceAdapterError("upstream exhaustion receipt drifted")
    return receipt


@dataclass(order=False)
class _ReversePageSampleRank:
    rank: tuple[bytes, bytes]
    page_id: str

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _ReversePageSampleRank):
            return NotImplemented
        return self.rank > other.rank


class CorpusIdentityStream(Iterator[CorpusIdentity]):
    """Lightweight complete-universe stream for bounded two-phase selection.

    Unlike :class:`CorpusUnitStream`, this formal path does not build complete
    linearized text or topology sidecars for 5.4 million pages.  The frozen
    atomic enumerator emits only exact identity, official ordinal, normalized
    nonempty-target digest, and empty count.  Encounter order is physical
    consecutive SQLite rowid order as certified by the upstream receipt.
    """

    def __init__(
        self,
        page_rows: Iterable[tuple[str, Any]],
        *,
        binding: SourceQualificationBinding,
        identity_full_compile_equivalence_qualification_sha256: str,
    ) -> None:
        _require_binding(binding, source_split="TRAIN")
        if not _is_sha256(
            identity_full_compile_equivalence_qualification_sha256
        ):
            raise FeverousSourceAdapterError(
                "identity/full-compiler equivalence qualification is absent"
            )
        self._page_source = page_rows
        self._page_rows = iter(page_rows)
        self._binding = binding
        self._equivalence_sha256 = (
            identity_full_compile_equivalence_qualification_sha256
        )
        self._current: Iterator[CorpusIdentity] = iter(())
        self._previous_page_id: str | None = None
        self._page_count = 0
        self._identity_count = 0
        self._empty_count = 0
        self._identity_hasher = hashlib.sha256()
        self._qualification_page_heap: list[_ReversePageSampleRank] = []
        self._atomic_source_sha256 = _source_file_sha256(atomic_corpus)
        self._acquisition_source_sha256 = _source_file_sha256(acquisition_core)
        self._adapter_source_sha256 = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        self._complete = False
        self._receipt: Mapping[str, Any] | None = None

    def __iter__(self) -> "CorpusIdentityStream":
        return self

    def __next__(self) -> CorpusIdentity:
        while True:
            try:
                identity = next(self._current)
            except StopIteration:
                pass
            else:
                encoded = _canonical_json(identity.commitment_row)
                self._identity_hasher.update(len(encoded).to_bytes(8, "big"))
                self._identity_hasher.update(encoded)
                self._identity_count += 1
                return identity
            if self._complete:
                raise StopIteration
            try:
                row = next(self._page_rows)
            except StopIteration:
                self._finish()
                raise StopIteration
            if not isinstance(row, tuple) or len(row) != 2:
                raise FeverousSourceAdapterError(
                    "database page stream row must be a page-id/payload tuple"
                )
            page_id, raw_page = row
            if (
                not isinstance(page_id, str)
                or not page_id
                or "\x00" in page_id
                or page_id == self._previous_page_id
            ):
                raise FeverousSourceAdapterError(
                    "database page stream contains an invalid adjacent duplicate"
                )
            self._previous_page_id = page_id
            page_id_utf8 = page_id.encode("utf-8", errors="strict")
            sample_rank = (
                hashlib.sha256(
                    b"feverous_p6_e2/identity_compiler_real_sample/v1\x00"
                    + page_id_utf8
                ).digest(),
                page_id_utf8,
            )
            sample_entry = _ReversePageSampleRank(sample_rank, page_id)
            if (
                len(self._qualification_page_heap)
                < REAL_IDENTITY_COMPILER_SAMPLE_PAGE_COUNT
            ):
                heapq.heappush(self._qualification_page_heap, sample_entry)
            elif sample_rank < self._qualification_page_heap[0].rank:
                heapq.heapreplace(self._qualification_page_heap, sample_entry)
            enumerator = getattr(
                atomic_corpus,
                "enumerate_official_page_atomic_identities",
                None,
            )
            if not callable(enumerator):
                raise FeverousSourceAdapterError(
                    "frozen lightweight atomic identity enumerator is unavailable"
                )
            try:
                enumeration = enumerator(page_id, raw_page)
                raw_identities = tuple(enumeration.identities)
            except Exception as exc:
                raise FeverousSourceAdapterError(
                    "lightweight atomic identity enumeration failed"
                ) from exc
            excluded_empty_count = getattr(
                enumeration, "excluded_empty_count", None
            )
            if excluded_empty_count is None:
                excluded = getattr(
                    enumeration, "excluded_empty_local_ids", None
                )
                if isinstance(excluded, tuple):
                    excluded_empty_count = len(excluded)
            if type(excluded_empty_count) is not int or excluded_empty_count < 0:
                raise FeverousSourceAdapterError(
                    "identity enumeration empty count is invalid"
                )
            converted: list[CorpusIdentity] = []
            previous_ordinal = -1
            local_ids: set[str] = set()
            for raw_identity in raw_identities:
                normalized_target = getattr(
                    raw_identity, "normalized_target", None
                )
                target_sha256 = getattr(raw_identity, "target_sha256", None)
                if (
                    getattr(raw_identity, "page", None) != page_id
                    or not isinstance(normalized_target, str)
                    or not normalized_target
                    or hashlib.sha256(normalized_target.encode("utf-8")).hexdigest()
                    != target_sha256
                ):
                    raise FeverousSourceAdapterError(
                        "enumerated target digest is not exact"
                    )
                identity = CorpusIdentity(
                    unit_key=f"{page_id}_{raw_identity.local_id}",
                    page=page_id,
                    local_id=raw_identity.local_id,
                    unit_type=raw_identity.unit_type,
                    official_ordinal=raw_identity.official_ordinal,
                    target_sha256=target_sha256,
                )
                if (
                    identity.official_ordinal <= previous_ordinal
                    or identity.local_id in local_ids
                ):
                    raise FeverousSourceAdapterError(
                        "enumerated identities are not strict official order"
                    )
                previous_ordinal = identity.official_ordinal
                local_ids.add(identity.local_id)
                converted.append(identity)
            self._page_count += 1
            self._empty_count += excluded_empty_count
            self._current = iter(converted)

    def _finish(self) -> None:
        upstream = _verified_upstream_receipt(
            self._page_source,
            hash_field="database_page_stream_receipt_sha256",
            schema=(
                "feverous_p6_e2_formal_source_v1_"
                "database_page_stream_receipt"
            ),
            status="complete_database_page_stream_exhausted",
        )
        if (
            upstream.get("stream_fully_exhausted") is not True
            or upstream.get("observed_database_row_count") != self._page_count
            or upstream.get("expected_database_row_count") != self._page_count
            or not _is_sha256(upstream.get("database_file_sha256"))
            or not _is_sha256(upstream.get("logical_page_stream_sha256"))
        ):
            raise FeverousSourceAdapterError(
                "database receipt does not bind the exhausted page universe"
            )
        enumerator_version = getattr(
            atomic_corpus,
            "IDENTITY_ENUMERATOR_VERSION",
            f"{ATOMIC_COMPILER_VERSION}_lightweight_identity_v1",
        )
        qualification_page_ids = self._qualification_page_ids()
        body: dict[str, Any] = {
            "schema": IDENTITY_STREAM_RECEIPT_SCHEMA,
            "version": VERSION,
            "status": "complete_atomic_identity_universe_exhausted_no_selection",
            "source_split": "TRAIN",
            "source_binding_sha256": self._binding.binding_sha256,
            "formal_source": upstream.get("formal_source") is True,
            "source_spec_sha256": upstream.get("source_spec_sha256"),
            "formal_source_opener_source_sha256": upstream.get(
                "formal_source_opener_source_sha256"
            ),
            "database_page_stream_receipt_sha256": upstream.get(
                "database_page_stream_receipt_sha256"
            ),
            "database_size_bytes": upstream.get("database_size_bytes"),
            "database_file_sha256": upstream.get("database_file_sha256"),
            "expected_database_row_count": upstream.get(
                "expected_database_row_count"
            ),
            "observed_database_row_count": upstream.get(
                "observed_database_row_count"
            ),
            "logical_page_stream_sha256": upstream.get(
                "logical_page_stream_sha256"
            ),
            "atomic_compiler_version": ATOMIC_COMPILER_VERSION,
            "identity_enumerator_version": enumerator_version,
            "atomic_compiler_source_sha256": self._atomic_source_sha256,
            "identity_enumerator_source_sha256": self._atomic_source_sha256,
            "source_adapter_source_sha256": self._adapter_source_sha256,
            "acquisition_source_sha256": self._acquisition_source_sha256,
            "identity_full_compile_equivalence_qualification_sha256": (
                self._equivalence_sha256
            ),
            "real_identity_compiler_sample_policy": (
                "lowest_sha256_domain_page_id_then_binary_page_id"
            ),
            "real_identity_compiler_sample_page_count": len(
                qualification_page_ids
            ),
            "real_identity_compiler_sample_page_set_sha256": _stable_hash(
                list(qualification_page_ids)
            ),
            "stream_fully_exhausted": True,
            "adapted_page_count": self._page_count,
            "eligible_atomic_identity_count": self._identity_count,
            "excluded_empty_atomic_identity_count": self._empty_count,
            "eligible_atomic_identity_stream_sha256": (
                self._identity_hasher.hexdigest()
            ),
            "maximum_resident_enumerated_pages": 1,
            "all_identities_or_pages_materialized": False,
            "full_atomic_text_or_sidecar_linearized": False,
            "cohort_canonical_set_or_fixed_corpus_selected": False,
            "development_or_test_source_accessed": False,
            "online_evaluator_calls": 0,
        }
        body["corpus_identity_stream_receipt_sha256"] = _stable_hash(body)
        self._receipt = MappingProxyType(body)
        self._complete = True

    @property
    def qualification_page_ids(self) -> tuple[str, ...]:
        if not self._complete:
            raise FeverousSourceAdapterError(
                "qualification sample is unavailable before exhaustion"
            )
        return self._qualification_page_ids()

    def _qualification_page_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                (entry.page_id for entry in self._qualification_page_heap),
                key=lambda value: value.encode("utf-8"),
            )
        )

    def aggregate_receipt(self) -> Mapping[str, Any]:
        if not self._complete or self._receipt is None:
            raise FeverousSourceAdapterError(
                "identity stream receipt is unavailable before normal exhaustion"
            )
        return self._receipt


class SelectedCorpusUnitStream(Iterator[CorpusUnit]):
    """Full-compile only bounded selected pages after universe exhaustion."""

    def __init__(
        self,
        page_rows: Iterable[tuple[str, Any]],
        *,
        resolver: FeverousWikiResolver,
        binding: SourceQualificationBinding,
        plan: CorpusSelectionPlan,
    ) -> None:
        _require_binding(binding, source_split="TRAIN")
        if not isinstance(resolver, FeverousWikiResolver):
            raise FeverousSourceAdapterError("exact Wikipedia resolver is required")
        if not isinstance(plan, CorpusSelectionPlan):
            raise FeverousSourceAdapterError("identity selection plan is absent")
        self._page_source = page_rows
        self._page_rows = iter(page_rows)
        self._connection = resolver.connection
        self._binding = binding
        self._plan = plan
        self._selected_by_key = {row.unit_key: row for row in plan.identities}
        self._expected_pages = plan.full_compile_page_ids
        self._current: Iterator[CorpusUnit] = iter(())
        self._page_index = 0
        self._compiled_unit_count = 0
        self._selected_hits: set[str] = set()
        self._compiled_hasher = hashlib.sha256()
        self._crosscheck_hasher = hashlib.sha256()
        self._atomic_source_sha256 = _source_file_sha256(atomic_corpus)
        self._complete = False
        self._receipt: Mapping[str, Any] | None = None

    def __iter__(self) -> "SelectedCorpusUnitStream":
        return self

    def __next__(self) -> CorpusUnit:
        while True:
            try:
                unit = next(self._current)
            except StopIteration:
                pass
            else:
                official_ordinal = unit.sidecar.get("official_ordinal")
                encoded = _canonical_json(
                    [
                        unit.page,
                        unit.local_id,
                        official_ordinal,
                        unit.unit_type,
                        hashlib.sha256(unit.text.encode("utf-8")).hexdigest(),
                        _stable_hash(dict(unit.sidecar)),
                    ]
                )
                self._compiled_hasher.update(len(encoded).to_bytes(8, "big"))
                self._compiled_hasher.update(encoded)
                self._compiled_unit_count += 1
                expected = self._selected_by_key.get(unit.unit_key)
                if expected is not None:
                    target = unit.text.split("\n", 1)[0][len("TARGET: ") :]
                    observed = CorpusIdentity(
                        unit_key=unit.unit_key,
                        page=unit.page,
                        local_id=unit.local_id,
                        unit_type=unit.unit_type,
                        official_ordinal=official_ordinal,
                        target_sha256=hashlib.sha256(
                            target.encode("utf-8")
                        ).hexdigest(),
                    )
                    if observed != expected or unit.unit_key in self._selected_hits:
                        raise FeverousSourceAdapterError(
                            "selected identity/full-compiler crosscheck failed"
                        )
                    crosscheck = _canonical_json(observed.commitment_row)
                    self._crosscheck_hasher.update(
                        len(crosscheck).to_bytes(8, "big")
                    )
                    self._crosscheck_hasher.update(crosscheck)
                    self._selected_hits.add(unit.unit_key)
                return unit
            if self._complete:
                raise StopIteration
            try:
                row = next(self._page_rows)
            except StopIteration:
                self._finish()
                raise StopIteration
            if not isinstance(row, tuple) or len(row) != 2:
                raise FeverousSourceAdapterError(
                    "selected page row must be a page-id/payload tuple"
                )
            page_id, raw_page = row
            if (
                self._page_index >= len(self._expected_pages)
                or page_id != self._expected_pages[self._page_index]
            ):
                raise FeverousSourceAdapterError(
                    "selected page lookup differs from the identity plan"
                )
            self._page_index += 1
            window = FeverousWikiResolver(self._connection)
            try:
                enumeration = atomic_corpus.enumerate_official_page_atomic_identities(
                    page_id, raw_page
                )
                compilation = compile_official_page(page_id, raw_page)
                atomic_corpus.crosscheck_identity_enumeration(
                    enumeration, compilation
                )
            except Exception as exc:
                raise FeverousSourceAdapterError(
                    "selected page identity/full-compiler seam failed"
                ) from exc
            adapted = _adapt_qualified_compilation(
                page_id,
                compilation,
                resolver=window,
                binding=self._binding,
            )
            self._current = iter(adapted.units)

    def _finish(self) -> None:
        upstream = _verified_upstream_receipt(
            self._page_source,
            hash_field="selected_page_lookup_receipt_sha256",
            schema=(
                "feverous_p6_e2_formal_source_v1_"
                "selected_page_lookup_receipt"
            ),
            status=(
                "selected_pages_materialized_after_full_universe_exhaustion"
            ),
        )
        if (
            self._page_index != len(self._expected_pages)
            or upstream.get("selected_page_count") != self._page_index
            or set(self._selected_hits) != set(self._selected_by_key)
            or not _is_sha256(
                upstream.get("database_page_stream_receipt_sha256")
            )
        ):
            raise FeverousSourceAdapterError(
                "selected page materialization was partial"
            )
        body: dict[str, Any] = {
            "schema": MATERIALIZATION_RECEIPT_SCHEMA,
            "version": VERSION,
            "status": "selected_pages_fully_compiled_and_identity_crosschecked",
            "source_split": "TRAIN",
            "source_binding_sha256": self._binding.binding_sha256,
            "formal_source_opener_source_sha256": upstream.get(
                "formal_source_opener_source_sha256"
            ),
            "formal_source": upstream.get("formal_source") is True,
            "database_page_stream_receipt_sha256": upstream.get(
                "database_page_stream_receipt_sha256"
            ),
            "selected_page_lookup_receipt_sha256": upstream.get(
                "selected_page_lookup_receipt_sha256"
            ),
            "corpus_identity_plan_sha256": self._plan.plan_sha256,
            "atomic_compiler_version": ATOMIC_COMPILER_VERSION,
            "atomic_compiler_source_sha256": self._atomic_source_sha256,
            "selected_page_count": self._page_index,
            "selected_identity_page_count": len(self._plan.selected_page_ids),
            "identity_full_compile_qualification_page_count": len(
                self._plan.qualification_page_ids
            ),
            "identity_full_compile_qualification_page_set_sha256": _stable_hash(
                list(self._plan.qualification_page_ids)
            ),
            "identity_full_compile_crosschecked_page_count": self._page_index,
            "compiled_eligible_unit_count_on_full_compiled_pages": (
                self._compiled_unit_count
            ),
            "selected_identity_hit_count": len(self._selected_hits),
            "selected_compiled_unit_stream_sha256": (
                self._compiled_hasher.hexdigest()
            ),
            "selected_identity_compiler_crosscheck_sha256": (
                self._crosscheck_hasher.hexdigest()
            ),
            "selected_page_lookup_fully_exhausted": True,
            "maximum_resident_compiled_pages": 1,
            "full_database_rescan": False,
            "development_or_test_source_accessed": False,
            "online_evaluator_calls": 0,
        }
        body["selected_corpus_materialization_receipt_sha256"] = _stable_hash(body)
        self._receipt = MappingProxyType(body)
        self._complete = True

    def aggregate_receipt(self) -> Mapping[str, Any]:
        if not self._complete or self._receipt is None:
            raise FeverousSourceAdapterError(
                "selected materialization receipt is unavailable before exhaustion"
            )
        return self._receipt


@dataclass(frozen=True)
class _ResolvedEvidenceSet:
    unit_keys: tuple[str, ...]
    unit_types: tuple[str, ...]
    pages: tuple[str, ...]
    contains_empty_target: bool


def _full_key(page_id: str, local_id: str) -> str:
    return f"{page_id}_{local_id}"


def _require_binding(
    binding: SourceQualificationBinding, *, source_split: str
) -> None:
    if not isinstance(binding, SourceQualificationBinding):
        raise FeverousSourceAdapterError("source qualification binding is absent")
    if binding != FROZEN_TRAIN_BINDING or source_split != "TRAIN":
        raise FeverousSourceAdapterError(
            "only the source-qualified official TRAIN split is permitted"
        )


def _resolver_target(
    resolver: FeverousWikiResolver,
    full_id: str,
    *,
    context_page: str,
) -> str:
    resolution = resolver.resolve_exact(full_id, context_page=context_page)
    if resolution.status != "resolved" or resolution.element is None:
        raise FeverousSourceAdapterError(
            "qualified resolver did not resolve an exact source element"
        )
    return normalize_surface(resolution.element.value)


def adapt_qualified_page(
    page_id: str,
    raw_page: Any,
    *,
    resolver: FeverousWikiResolver,
    binding: SourceQualificationBinding,
) -> AdaptedPage:
    """Cross-check and convert one already-qualified official TRAIN page."""

    _require_binding(binding, source_split="TRAIN")
    if not isinstance(resolver, FeverousWikiResolver):
        raise FeverousSourceAdapterError("exact Wikipedia resolver is required")
    compilation: PageCompilation = compile_official_page(page_id, raw_page)
    return _adapt_qualified_compilation(
        page_id,
        compilation,
        resolver=resolver,
        binding=binding,
    )


def _adapt_qualified_compilation(
    page_id: str,
    compilation: PageCompilation,
    *,
    resolver: FeverousWikiResolver,
    binding: SourceQualificationBinding,
) -> AdaptedPage:
    """Convert an already full-compiled page after exact resolver checks."""

    if compilation.page != page_id:
        raise FeverousSourceAdapterError("compiled page identity drifted")
    title_key = _full_key(page_id, "title")
    if not _resolver_target(resolver, title_key, context_page=page_id):
        raise FeverousSourceAdapterError("qualified page title is empty")

    corpus_units: list[CorpusUnit] = []
    for unit in compilation.units:
        full_id = _full_key(page_id, unit.sidecar.local_id)
        resolved_target = _resolver_target(
            resolver, full_id, context_page=page_id
        )
        if resolved_target != unit.target:
            raise FeverousSourceAdapterError(
                "atomic compiler target differs from exact resolver target"
            )
        sidecar = asdict(unit.sidecar)
        corpus_units.append(
            CorpusUnit(
                unit_key=full_id,
                text=unit.text,
                unit_type=unit.sidecar.unit_type,
                sidecar=sidecar,
            )
        )

    excluded_keys: list[str] = []
    for local_id in compilation.excluded_empty_local_ids:
        full_id = _full_key(page_id, local_id)
        if _resolver_target(resolver, full_id, context_page=page_id):
            raise FeverousSourceAdapterError(
                "atomic compiler excluded a nonempty exact source target"
            )
        excluded_keys.append(full_id)
    return AdaptedPage(
        page_id=page_id,
        units=tuple(corpus_units),
        excluded_empty_unit_keys=tuple(excluded_keys),
        source_binding_sha256=binding.binding_sha256,
    )


def _source_key(source_id: object) -> str:
    if type(source_id) is int and source_id >= 0:
        return f"TRAIN:int:{source_id}"
    if isinstance(source_id, str) and source_id and "\x00" not in source_id:
        try:
            size = len(source_id.encode("utf-8", errors="strict"))
        except UnicodeEncodeError as exc:
            raise FeverousSourceAdapterError(
                "official TRAIN id is invalid"
            ) from exc
        return f"TRAIN:str:{size}:{source_id}"
    raise FeverousSourceAdapterError("official TRAIN id is invalid")


def _require_official_record(record: Mapping[str, Any]) -> None:
    if set(record) != OFFICIAL_RECORD_FIELDS:
        raise FeverousSourceAdapterError(
            "official TRAIN record root schema drifted"
        )
    _source_key(record.get("id"))
    claim = record.get("claim")
    if not isinstance(claim, str) or "\x00" in claim:
        raise FeverousSourceAdapterError("official TRAIN claim is invalid")
    if not isinstance(record.get("evidence"), list):
        raise FeverousSourceAdapterError("official TRAIN evidence is invalid")


def _context_page(
    content_id: str,
    context_values: object,
) -> tuple[str, tuple[str, ...]]:
    if not isinstance(context_values, list) or any(
        not isinstance(value, str) for value in context_values
    ):
        raise FeverousSourceAdapterError("official evidence context is invalid")
    parsed_context = []
    for context_id in context_values:
        try:
            parsed = parse_element_id(context_id)
        except FeverousWikipediaQualificationError as exc:
            raise FeverousSourceAdapterError(
                "official evidence context id is invalid"
            ) from exc
        if parsed.kind not in CONTEXT_KINDS:
            raise FeverousSourceAdapterError(
                "official evidence context kind is invalid"
            )
        parsed_context.append((context_id, parsed))
    titles = [parsed for _, parsed in parsed_context if parsed.kind == "title"]
    if len(titles) != 1:
        raise FeverousSourceAdapterError(
            "official evidence does not have one exact title authority"
        )
    page = titles[0].page
    try:
        parsed_content = parse_element_id(content_id)
    except FeverousWikipediaQualificationError as exc:
        raise FeverousSourceAdapterError(
            "official evidence content id is invalid"
        ) from exc
    if parsed_content.kind not in CONTENT_KINDS:
        raise FeverousSourceAdapterError(
            "official evidence content kind is invalid"
        )
    if parsed_content.page != page or any(
        parsed.page != page for _, parsed in parsed_context
    ):
        raise FeverousSourceAdapterError(
            "official evidence content/context page drifted"
        )
    return page, tuple(context_id for context_id, _ in parsed_context)


def _resolve_evidence_set(
    evidence_set: object,
    *,
    resolver: FeverousWikiResolver,
    pages: Mapping[str, AdaptedPage] | None,
) -> _ResolvedEvidenceSet:
    if not isinstance(evidence_set, Mapping):
        raise FeverousSourceAdapterError("official evidence set is invalid")
    content = evidence_set.get("content")
    context = evidence_set.get("context")
    if not isinstance(content, list) or any(
        not isinstance(value, str) for value in content
    ):
        raise FeverousSourceAdapterError("official evidence content is invalid")
    if len(content) != len(set(content)):
        raise FeverousSourceAdapterError(
            "official evidence content contains a duplicate"
        )
    if not isinstance(context, Mapping) or set(context) != set(content):
        raise FeverousSourceAdapterError(
            "official evidence context keys drifted from content"
        )

    unit_keys: list[str] = []
    unit_types: list[str] = []
    page_ids: list[str] = []
    contains_empty = False
    for content_id in content:
        page_id, context_ids = _context_page(content_id, context[content_id])
        resolved_target = _resolver_target(
            resolver, content_id, context_page=page_id
        )
        for context_id in context_ids:
            _resolver_target(resolver, context_id, context_page=page_id)
        if pages is None:
            parsed = parse_element_id(content_id)
            contains_empty = contains_empty or not resolved_target
            unit_keys.append(content_id)
            unit_types.append(parsed.kind)
            page_ids.append(page_id)
            continue
        page = pages.get(page_id)
        if not isinstance(page, AdaptedPage):
            raise FeverousSourceAdapterError(
                "an exact evidence page was not adapted before record screening"
            )
        if content_id in page.excluded_empty_unit_keys:
            contains_empty = True
            parsed = parse_element_id(content_id)
            unit_keys.append(content_id)
            unit_types.append(parsed.kind)
            page_ids.append(page_id)
            continue
        unit = page.unit_by_key.get(content_id)
        if unit is None:
            raise FeverousSourceAdapterError(
                "resolved evidence atom is absent from atomic page projection"
            )
        unit_keys.append(content_id)
        unit_types.append(unit.unit_type)
        page_ids.append(page_id)
    return _ResolvedEvidenceSet(
        unit_keys=tuple(unit_keys),
        unit_types=tuple(unit_types),
        pages=tuple(page_ids),
        contains_empty_target=contains_empty,
    )


def _family_eligible(
    family: str,
    resolved: _ResolvedEvidenceSet,
) -> bool:
    kinds = set(resolved.unit_types)
    if family == "Combining Tables and Text":
        return bool(kinds.intersection(TEXT_TYPES)) and bool(
            kinds.intersection(STRUCTURED_TYPES)
        )
    if family == "Entity Disambiguation":
        return True
    if family == "Multi-hop Reasoning":
        return len(set(resolved.pages)) >= 2
    if family == "Numerical Reasoning":
        return bool(kinds) and kinds.issubset(STRUCTURED_TYPES)
    return False


def adapt_train_record(
    record: Mapping[str, Any],
    *,
    source_split: str,
    resolver: FeverousWikiResolver,
    pages: Mapping[str, AdaptedPage] | None,
    binding: SourceQualificationBinding,
) -> RecordAdaptation:
    """Screen one TRAIN record and emit at most one acquisition candidate."""

    _require_binding(binding, source_split=source_split)
    if not isinstance(record, Mapping):
        raise FeverousSourceAdapterError("official TRAIN record must be an object")
    if not record:
        return RecordAdaptation(None, "blank_sentinel", 0, 0, 0, 0, 0, 0)
    _require_official_record(record)
    family = record["challenge"]
    verdict = record["label"]
    if family not in FAMILIES:
        return RecordAdaptation(None, "unsupported_family", 0, 0, 0, 0, 0, 0)
    if verdict not in VERDICTS:
        return RecordAdaptation(None, "unsupported_verdict", 0, 0, 0, 0, 0, 0)
    claim = record["claim"]
    try:
        normalized_claim = normalize_surface(claim)
    except (TypeError, ValueError) as exc:
        raise FeverousSourceAdapterError("official TRAIN claim is invalid") from exc
    if not normalized_claim:
        raise FeverousSourceAdapterError("official TRAIN claim is empty")

    official_sets = record["evidence"]
    all_official_keys: set[str] = set()
    eligible_sets: set[tuple[str, ...]] = set()
    empty_count = 0
    cardinality_count = 0
    structure_count = 0
    reference_count = 0
    for evidence_set in official_sets:
        resolved = _resolve_evidence_set(
            evidence_set,
            resolver=resolver,
            pages=pages,
        )
        reference_count += len(resolved.unit_keys)
        all_official_keys.update(resolved.unit_keys)
        if resolved.contains_empty_target:
            empty_count += 1
            continue
        if not 2 <= len(resolved.unit_keys) <= 5:
            cardinality_count += 1
            continue
        if not _family_eligible(str(family), resolved):
            structure_count += 1
            continue
        eligible_sets.add(tuple(sorted(resolved.unit_keys)))

    if not eligible_sets:
        return RecordAdaptation(
            None,
            "no_eligible_canonical_set",
            len(official_sets),
            0,
            reference_count,
            empty_count,
            cardinality_count,
            structure_count,
        )
    if not all_official_keys:
        raise FeverousSourceAdapterError(
            "eligible record has no official evidence-key universe"
        )
    candidate = CandidateRecord(
        source_key=_source_key(record["id"]),
        claim=claim,
        family=str(family),
        verdict=str(verdict),
        evidence_sets=tuple(sorted(eligible_sets)),
        all_official_evidence_keys=tuple(sorted(all_official_keys)),
    )
    return RecordAdaptation(
        candidate,
        "eligible_candidate",
        len(official_sets),
        len(eligible_sets),
        reference_count,
        empty_count,
        cardinality_count,
        structure_count,
    )


def _aggregate_receipt(
    *,
    decisions: Sequence[RecordAdaptation],
    pages: Sequence[AdaptedPage],
    corpus_units: Sequence[CorpusUnit],
    binding: SourceQualificationBinding,
) -> dict[str, Any]:
    statuses = Counter(decision.status for decision in decisions)
    body: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "version": VERSION,
        "status": "source_adapted_no_selection_no_scoring",
        "source_split": "TRAIN",
        "source_binding_sha256": binding.binding_sha256,
        "design_sha256": binding.design_sha256,
        "annotation_qualification_sha256": (
            binding.annotation_qualification_sha256
        ),
        "wikipedia_qualification_sha256": (
            binding.wikipedia_qualification_sha256
        ),
        "atomic_compiler_version": binding.atomic_compiler_version,
        "wikipedia_resolver_version": binding.wikipedia_resolver_version,
        "acquisition_version": binding.acquisition_version,
        "downstream_corpus_view_schema": binding.corpus_view_schema,
        "downstream_fixed_atomic_unit_count": (
            binding.frozen_corpus_unit_count
        ),
        "input_record_count": len(decisions),
        "record_status_counts": {
            key: statuses[key] for key in sorted(statuses)
        },
        "candidate_count": sum(
            decision.candidate is not None for decision in decisions
        ),
        "official_evidence_set_count": sum(
            decision.official_evidence_set_count for decision in decisions
        ),
        "eligible_evidence_set_count": sum(
            decision.eligible_evidence_set_count for decision in decisions
        ),
        "official_evidence_reference_count": sum(
            decision.official_evidence_reference_count for decision in decisions
        ),
        "excluded_empty_set_count": sum(
            decision.excluded_empty_set_count for decision in decisions
        ),
        "excluded_cardinality_set_count": sum(
            decision.excluded_cardinality_set_count for decision in decisions
        ),
        "excluded_family_structure_set_count": sum(
            decision.excluded_family_structure_set_count
            for decision in decisions
        ),
        "adapted_page_count": len(pages),
        "eligible_atomic_source_unit_count": len(corpus_units),
        "excluded_empty_atomic_source_unit_count": sum(
            len(page.excluded_empty_unit_keys) for page in pages
        ),
        "raw_claim_page_or_evidence_serialized": False,
        "per_record_or_per_source_digest_serialized": False,
        "cohort_block_or_canonical_set_selected": False,
        "fixed_8192_corpus_formed": False,
        "utility_recipe_or_model_accessed": False,
        "development_or_test_source_accessed": False,
        "online_evaluator_calls": 0,
    }
    body["adapter_receipt_sha256"] = _stable_hash(body)
    return body


def verify_adapter_receipt(receipt: Mapping[str, Any]) -> str:
    if not isinstance(receipt, Mapping):
        raise FeverousSourceAdapterError("adapter receipt must be an object")
    if set(receipt) != _RECEIPT_KEYS:
        raise FeverousSourceAdapterError("adapter receipt drifted")
    body = dict(receipt)
    declared = body.pop("adapter_receipt_sha256", None)
    status_counts = body.get("record_status_counts")
    count_fields = (
        "input_record_count",
        "candidate_count",
        "official_evidence_set_count",
        "eligible_evidence_set_count",
        "official_evidence_reference_count",
        "excluded_empty_set_count",
        "excluded_cardinality_set_count",
        "excluded_family_structure_set_count",
        "adapted_page_count",
        "eligible_atomic_source_unit_count",
        "excluded_empty_atomic_source_unit_count",
        "online_evaluator_calls",
    )
    counts_valid = all(
        type(body.get(field)) is int and body[field] >= 0
        for field in count_fields
    )
    statuses_valid = (
        isinstance(status_counts, Mapping)
        and set(status_counts).issubset(_RECORD_STATUSES)
        and all(type(value) is int and value >= 0 for value in status_counts.values())
        and sum(status_counts.values()) == body.get("input_record_count")
        and status_counts.get("eligible_candidate", 0)
        == body.get("candidate_count")
    )
    if (
        not _is_sha256(declared)
        or not counts_valid
        or not statuses_valid
        or body.get("schema") != RECEIPT_SCHEMA
        or body.get("version") != VERSION
        or body.get("status") != "source_adapted_no_selection_no_scoring"
        or body.get("source_split") != "TRAIN"
        or body.get("source_binding_sha256")
        != FROZEN_TRAIN_BINDING.binding_sha256
        or body.get("design_sha256") != DESIGN_SHA256
        or body.get("annotation_qualification_sha256")
        != ANNOTATION_QUALIFICATION_SHA256
        or body.get("wikipedia_qualification_sha256")
        != WIKIPEDIA_QUALIFICATION_SHA256
        or body.get("atomic_compiler_version") != ATOMIC_COMPILER_VERSION
        or body.get("wikipedia_resolver_version")
        != WIKIPEDIA_RESOLVER_VERSION
        or body.get("acquisition_version") != ACQUISITION_VERSION
        or body.get("downstream_corpus_view_schema") != CORPUS_VIEW_SCHEMA
        or body.get("downstream_fixed_atomic_unit_count")
        != CORPUS_UNIT_COUNT
        or body.get("per_record_or_per_source_digest_serialized") is not False
        or body.get("cohort_block_or_canonical_set_selected") is not False
        or body.get("fixed_8192_corpus_formed") is not False
        or body.get("utility_recipe_or_model_accessed") is not False
        or body.get("raw_claim_page_or_evidence_serialized") is not False
        or body.get("development_or_test_source_accessed") is not False
        or body.get("online_evaluator_calls") != 0
        or _stable_hash(body) != declared
    ):
        raise FeverousSourceAdapterError("adapter receipt drifted")
    return str(declared)


def _corpus_stream_receipt(
    *,
    page_count: int,
    unit_count: int,
    empty_count: int,
    unit_stream_sha256: str,
    atomic_compiler_source_sha256: str,
    binding: SourceQualificationBinding,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": STREAM_RECEIPT_SCHEMA,
        "version": VERSION,
        "status": "qualified_corpus_stream_exhausted_no_selection",
        "source_split": "TRAIN",
        "source_binding_sha256": binding.binding_sha256,
        "atomic_compiler_version": binding.atomic_compiler_version,
        "wikipedia_resolver_version": binding.wikipedia_resolver_version,
        "downstream_corpus_view_schema": binding.corpus_view_schema,
        "downstream_fixed_atomic_unit_count": binding.frozen_corpus_unit_count,
        "stream_fully_exhausted": True,
        "adapted_page_count": page_count,
        "emitted_eligible_atomic_unit_count": unit_count,
        "excluded_empty_atomic_unit_count": empty_count,
        "eligible_atomic_unit_stream_sha256": unit_stream_sha256,
        "atomic_compiler_source_sha256": atomic_compiler_source_sha256,
        "maximum_resident_compiled_pages": 1,
        "all_units_materialized_or_sorted": False,
        "cohort_canonical_set_or_fixed_corpus_selected": False,
        "raw_claim_page_or_evidence_serialized": False,
        "online_evaluator_calls": 0,
    }
    body["corpus_stream_receipt_sha256"] = _stable_hash(body)
    return body


def verify_corpus_stream_receipt(receipt: Mapping[str, Any]) -> str:
    if not isinstance(receipt, Mapping):
        raise FeverousSourceAdapterError("corpus stream receipt must be an object")
    body = dict(receipt)
    declared = body.pop("corpus_stream_receipt_sha256", None)
    count_fields = (
        "adapted_page_count",
        "emitted_eligible_atomic_unit_count",
        "excluded_empty_atomic_unit_count",
        "online_evaluator_calls",
    )
    if (
        set(body)
        != {
            "schema",
            "version",
            "status",
            "source_split",
            "source_binding_sha256",
            "atomic_compiler_version",
            "wikipedia_resolver_version",
            "downstream_corpus_view_schema",
            "downstream_fixed_atomic_unit_count",
            "stream_fully_exhausted",
            "adapted_page_count",
            "emitted_eligible_atomic_unit_count",
            "excluded_empty_atomic_unit_count",
            "eligible_atomic_unit_stream_sha256",
            "atomic_compiler_source_sha256",
            "maximum_resident_compiled_pages",
            "all_units_materialized_or_sorted",
            "cohort_canonical_set_or_fixed_corpus_selected",
            "raw_claim_page_or_evidence_serialized",
            "online_evaluator_calls",
        }
        or not _is_sha256(declared)
        or any(type(body.get(field)) is not int or body[field] < 0 for field in count_fields)
        or body.get("schema") != STREAM_RECEIPT_SCHEMA
        or body.get("version") != VERSION
        or body.get("status")
        != "qualified_corpus_stream_exhausted_no_selection"
        or body.get("source_split") != "TRAIN"
        or body.get("source_binding_sha256")
        != FROZEN_TRAIN_BINDING.binding_sha256
        or body.get("atomic_compiler_version") != ATOMIC_COMPILER_VERSION
        or body.get("wikipedia_resolver_version")
        != WIKIPEDIA_RESOLVER_VERSION
        or body.get("downstream_corpus_view_schema") != CORPUS_VIEW_SCHEMA
        or body.get("downstream_fixed_atomic_unit_count") != CORPUS_UNIT_COUNT
        or not _is_sha256(body.get("eligible_atomic_unit_stream_sha256"))
        or not _is_sha256(body.get("atomic_compiler_source_sha256"))
        or body.get("stream_fully_exhausted") is not True
        or body.get("maximum_resident_compiled_pages") != 1
        or body.get("all_units_materialized_or_sorted") is not False
        or body.get("cohort_canonical_set_or_fixed_corpus_selected") is not False
        or body.get("raw_claim_page_or_evidence_serialized") is not False
        or body.get("online_evaluator_calls") != 0
        or _stable_hash(body) != declared
    ):
        raise FeverousSourceAdapterError("corpus stream receipt drifted")
    return str(declared)


def iter_qualified_corpus_units(
    page_rows: Iterable[tuple[str, Any]],
    *,
    resolver: FeverousWikiResolver,
    binding: SourceQualificationBinding,
) -> CorpusUnitStream:
    """Return a one-pass bounded-memory official-page atomic-unit stream."""

    return CorpusUnitStream(
        page_rows,
        resolver=resolver,
        binding=binding,
    )


def verify_corpus_identity_stream_receipt(receipt: Mapping[str, Any]) -> str:
    if not isinstance(receipt, Mapping):
        raise FeverousSourceAdapterError("identity stream receipt must be an object")
    body = dict(receipt)
    declared = body.pop("corpus_identity_stream_receipt_sha256", None)
    sha_fields = (
        "source_binding_sha256",
        "formal_source_opener_source_sha256",
        "source_spec_sha256",
        "database_page_stream_receipt_sha256",
        "database_file_sha256",
        "logical_page_stream_sha256",
        "atomic_compiler_source_sha256",
        "identity_enumerator_source_sha256",
        "source_adapter_source_sha256",
        "acquisition_source_sha256",
        "identity_full_compile_equivalence_qualification_sha256",
        "real_identity_compiler_sample_page_set_sha256",
        "eligible_atomic_identity_stream_sha256",
    )
    count_fields = (
        "database_size_bytes",
        "expected_database_row_count",
        "observed_database_row_count",
        "adapted_page_count",
        "eligible_atomic_identity_count",
        "excluded_empty_atomic_identity_count",
        "real_identity_compiler_sample_page_count",
        "online_evaluator_calls",
    )
    if (
        not _is_sha256(declared)
        or body.get("schema") != IDENTITY_STREAM_RECEIPT_SCHEMA
        or body.get("version") != VERSION
        or body.get("status")
        != "complete_atomic_identity_universe_exhausted_no_selection"
        or body.get("source_split") != "TRAIN"
        or body.get("source_binding_sha256")
        != FROZEN_TRAIN_BINDING.binding_sha256
        or body.get("atomic_compiler_version") != ATOMIC_COMPILER_VERSION
        or body.get("stream_fully_exhausted") is not True
        or body.get("real_identity_compiler_sample_policy")
        != "lowest_sha256_domain_page_id_then_binary_page_id"
        or body.get("real_identity_compiler_sample_page_count")
        != min(
            body.get("adapted_page_count", -1),
            REAL_IDENTITY_COMPILER_SAMPLE_PAGE_COUNT,
        )
        or body.get("adapted_page_count")
        != body.get("observed_database_row_count")
        or body.get("expected_database_row_count")
        != body.get("observed_database_row_count")
        or body.get("maximum_resident_enumerated_pages") != 1
        or body.get("all_identities_or_pages_materialized") is not False
        or body.get("full_atomic_text_or_sidecar_linearized") is not False
        or body.get("cohort_canonical_set_or_fixed_corpus_selected") is not False
        or body.get("development_or_test_source_accessed") is not False
        or body.get("online_evaluator_calls") != 0
        or any(not _is_sha256(body.get(field)) for field in sha_fields)
        or any(
            type(body.get(field)) is not int or body[field] < 0
            for field in count_fields
        )
        or _stable_hash(body) != declared
    ):
        raise FeverousSourceAdapterError("identity stream receipt drifted")
    return str(declared)


def iter_qualified_corpus_identities(
    page_rows: Iterable[tuple[str, Any]],
    *,
    binding: SourceQualificationBinding,
    identity_full_compile_equivalence_qualification_sha256: str,
) -> CorpusIdentityStream:
    """Return the formal lightweight, page-bounded complete-universe stream."""

    return CorpusIdentityStream(
        page_rows,
        binding=binding,
        identity_full_compile_equivalence_qualification_sha256=(
            identity_full_compile_equivalence_qualification_sha256
        ),
    )


def verify_selected_corpus_materialization_receipt(
    receipt: Mapping[str, Any],
) -> str:
    if not isinstance(receipt, Mapping):
        raise FeverousSourceAdapterError(
            "selected materialization receipt must be an object"
        )
    body = dict(receipt)
    declared = body.pop(
        "selected_corpus_materialization_receipt_sha256", None
    )
    sha_fields = (
        "source_binding_sha256",
        "formal_source_opener_source_sha256",
        "database_page_stream_receipt_sha256",
        "selected_page_lookup_receipt_sha256",
        "corpus_identity_plan_sha256",
        "atomic_compiler_source_sha256",
        "selected_compiled_unit_stream_sha256",
        "selected_identity_compiler_crosscheck_sha256",
        "identity_full_compile_qualification_page_set_sha256",
    )
    if (
        not _is_sha256(declared)
        or body.get("schema") != MATERIALIZATION_RECEIPT_SCHEMA
        or body.get("version") != VERSION
        or body.get("status")
        != "selected_pages_fully_compiled_and_identity_crosschecked"
        or body.get("source_split") != "TRAIN"
        or body.get("source_binding_sha256")
        != FROZEN_TRAIN_BINDING.binding_sha256
        or body.get("atomic_compiler_version") != ATOMIC_COMPILER_VERSION
        or type(body.get("selected_page_count")) is not int
        or body["selected_page_count"] < 0
        or type(body.get("selected_identity_page_count")) is not int
        or body["selected_identity_page_count"] < 0
        or type(body.get("identity_full_compile_qualification_page_count"))
        is not int
        or body["identity_full_compile_qualification_page_count"] < 0
        or body["identity_full_compile_qualification_page_count"]
        > REAL_IDENTITY_COMPILER_SAMPLE_PAGE_COUNT
        or body.get("identity_full_compile_crosschecked_page_count")
        != body.get("selected_page_count")
        or type(body.get("compiled_eligible_unit_count_on_full_compiled_pages"))
        is not int
        or body["compiled_eligible_unit_count_on_full_compiled_pages"] < 0
        or body.get("selected_identity_hit_count") != CORPUS_UNIT_COUNT
        or body.get("selected_page_lookup_fully_exhausted") is not True
        or body.get("maximum_resident_compiled_pages") != 1
        or body.get("full_database_rescan") is not False
        or body.get("development_or_test_source_accessed") is not False
        or body.get("online_evaluator_calls") != 0
        or any(not _is_sha256(body.get(field)) for field in sha_fields)
        or _stable_hash(body) != declared
    ):
        raise FeverousSourceAdapterError("materialization receipt drifted")
    return str(declared)


def iter_selected_corpus_units(
    page_rows: Iterable[tuple[str, Any]],
    *,
    resolver: FeverousWikiResolver,
    binding: SourceQualificationBinding,
    plan: CorpusSelectionPlan,
) -> SelectedCorpusUnitStream:
    return SelectedCorpusUnitStream(
        page_rows,
        resolver=resolver,
        binding=binding,
        plan=plan,
    )


def adapt_train_candidate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    source_split: str,
    resolver: FeverousWikiResolver,
    binding: SourceQualificationBinding,
) -> AdapterBatch:
    """Screen candidates with page-local resolver windows and no page corpus."""

    _require_binding(binding, source_split=source_split)
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(
        records, Sequence
    ):
        raise FeverousSourceAdapterError("TRAIN records must be a sequence")
    decisions: list[RecordAdaptation] = []
    for record in records:
        window = FeverousWikiResolver(resolver.connection)
        decisions.append(
            adapt_train_record(
                record,
                source_split=source_split,
                resolver=window,
                pages=None,
                binding=binding,
            )
        )
    candidates = tuple(
        decision.candidate
        for decision in decisions
        if decision.candidate is not None
    )
    if len({candidate.source_key for candidate in candidates}) != len(candidates):
        raise FeverousSourceAdapterError("adapted TRAIN source id is not unique")
    receipt = _aggregate_receipt(
        decisions=decisions,
        pages=(),
        corpus_units=(),
        binding=binding,
    )
    verify_adapter_receipt(receipt)
    return AdapterBatch(
        candidates=candidates,
        corpus_units=(),
        receipt=MappingProxyType(receipt),
    )


def adapt_train_records(
    records: Sequence[Mapping[str, Any]],
    *,
    source_split: str,
    resolver: FeverousWikiResolver,
    pages: Mapping[str, AdaptedPage],
    binding: SourceQualificationBinding,
) -> AdapterBatch:
    """Adapt a qualified TRAIN batch without forming a cohort or corpus."""

    _require_binding(binding, source_split=source_split)
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(
        records, Sequence
    ):
        raise FeverousSourceAdapterError("TRAIN records must be a sequence")
    if not isinstance(pages, Mapping) or any(
        key != page.page_id
        or page.source_binding_sha256 != binding.binding_sha256
        for key, page in pages.items()
        if isinstance(page, AdaptedPage)
    ) or any(not isinstance(page, AdaptedPage) for page in pages.values()):
        raise FeverousSourceAdapterError("adapted page map is invalid")

    decisions = tuple(
        adapt_train_record(
            record,
            source_split=source_split,
            resolver=resolver,
            pages=pages,
            binding=binding,
        )
        for record in records
    )
    candidates = tuple(
        decision.candidate
        for decision in decisions
        if decision.candidate is not None
    )
    if len({candidate.source_key for candidate in candidates}) != len(candidates):
        raise FeverousSourceAdapterError("adapted TRAIN source id is not unique")

    corpus_units = tuple(
        sorted(
            (unit for page in pages.values() for unit in page.units),
            key=lambda unit: (unit.page, unit.local_id),
        )
    )
    if len({unit.unit_key for unit in corpus_units}) != len(corpus_units):
        raise FeverousSourceAdapterError(
            "adapted page map contains duplicate atomic identities"
        )
    receipt = _aggregate_receipt(
        decisions=decisions,
        pages=tuple(pages.values()),
        corpus_units=corpus_units,
        binding=binding,
    )
    verify_adapter_receipt(receipt)
    return AdapterBatch(
        candidates=candidates,
        corpus_units=corpus_units,
        receipt=MappingProxyType(receipt),
    )


__all__ = [
    "ACQUISITION_VERSION",
    "ANNOTATION_QUALIFICATION_SHA256",
    "ATOMIC_COMPILER_VERSION",
    "AdapterBatch",
    "AdaptedPage",
    "CorpusUnitStream",
    "DESIGN_SHA256",
    "FROZEN_TRAIN_BINDING",
    "OFFICIAL_RECORD_FIELDS",
    "RECEIPT_SCHEMA",
    "STREAM_RECEIPT_SCHEMA",
    "RecordAdaptation",
    "SourceQualificationBinding",
    "VERSION",
    "WIKIPEDIA_QUALIFICATION_SHA256",
    "WIKIPEDIA_RESOLVER_VERSION",
    "FeverousSourceAdapterError",
    "adapt_qualified_page",
    "adapt_train_candidate_records",
    "adapt_train_record",
    "adapt_train_records",
    "iter_qualified_corpus_units",
    "verify_adapter_receipt",
    "verify_corpus_stream_receipt",
]
