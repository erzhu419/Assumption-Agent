"""Schedule-neutral target-blind semantic coverage for Q0.5a snapshots.

This module projects the unique capacity-engine candidate semantics into the
frozen 846-row formal coverage record type.  Per-record application and strict
admission roots are replayable diagnostic evidence; no coverage archive root,
Q1 output slot, receipt, gate, target role, or certificate is produced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, NoReturn

from . import phase3_q0_input_adapter_v1 as _adapter
from . import phase3_q1_capacity_preflight_v1 as _capacity
from . import phase3_q1_formal_archive_contract_v1 as _formal
from .phase3_q1_partition_snapshot_v1 import (
    Q1PartitionSnapshotV1,
    validate_q1_partition_snapshot_v1,
)
from .phase3_q1_quotient_contract_v1 import OutputSortId
from .strict_cbor_v1 import canonical_cbor_encode, rfc6962_root


SEMANTIC_COVERAGE_DIAGNOSTIC_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05a-semantic-coverage-diagnostic/1"
)
SEMANTIC_COVERAGE_DIAGNOSTIC_ID: Final = (
    "hegel-phase3a-q05a-target-blind-semantic-coverage-v1"
)

REJECT_Q1_SEMANTIC_COVERAGE: Final = "REJECT_Q1_SEMANTIC_COVERAGE"
FAIL_SHA256_PREIMAGE_COLLISION: Final = "FAIL_SHA256_PREIMAGE_COLLISION"
FAIL_Q1_SEMANTIC_COVERAGE_REPLAY: Final = "FAIL_Q1_SEMANTIC_COVERAGE_REPLAY"

Q1SemanticCoverageRecordV1 = _formal.Q1SemanticCoverageRecordV1


class Q1SemanticCoverageError(ValueError):
    """Stable fail-closed error from the schedule-neutral coverage layer."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Q1SemanticCoverageError(code, detail)


def _canonical_order(values: tuple[tuple[object, ...], ...]) -> tuple[tuple[object, ...], ...]:
    return tuple(sorted(values, key=canonical_cbor_encode))


def _register_rfc6962_preimage_v1(
    seen_preimages: dict[bytes, bytes],
    *,
    root: bytes,
    rows: tuple[tuple[object, ...], ...],
    label: str,
) -> None:
    preimage = canonical_cbor_encode(rows)
    previous = seen_preimages.get(root)
    if previous is not None and previous != preimage:
        _fail(
            FAIL_SHA256_PREIMAGE_COLLISION,
            f"{label} RFC6962 root has different preimages",
        )
    seen_preimages[root] = preimage


def _register_digest_preimage_v1(
    seen_preimages: dict[bytes, bytes],
    *,
    digest: bytes,
    preimage: bytes,
    label: str,
) -> None:
    previous = seen_preimages.get(digest)
    if previous is not None and previous != preimage:
        _fail(
            FAIL_SHA256_PREIMAGE_COLLISION,
            f"{label} digest has different preimages",
        )
    seen_preimages[digest] = preimage


@dataclass(frozen=True, slots=True)
class Q1SemanticCoveragePreimageV1:
    construction_depth: int
    coverage_code: int
    eligible_application_keys: tuple[tuple[object, ...], ...]
    processed_application_keys: tuple[tuple[object, ...], ...]
    strict_admission_preimages: tuple[tuple[bytes, bytes], ...]

    def __post_init__(self) -> None:
        if type(self.construction_depth) is not int or not 0 <= self.construction_depth <= 3:
            _fail(REJECT_Q1_SEMANTIC_COVERAGE, "construction depth is invalid")
        if type(self.coverage_code) is not int or not 0 <= self.coverage_code <= 0xFFFF:
            _fail(REJECT_Q1_SEMANTIC_COVERAGE, "coverage code is invalid")
        for name in ("eligible_application_keys", "processed_application_keys"):
            material = getattr(self, name)
            if type(material) is not tuple or any(type(row) is not tuple for row in material):
                _fail(REJECT_Q1_SEMANTIC_COVERAGE, f"{name} must be tuple rows")
            if material != _canonical_order(material):
                _fail(REJECT_Q1_SEMANTIC_COVERAGE, f"{name} is not canonical")
            encoded = tuple(canonical_cbor_encode(row) for row in material)
            if len(set(encoded)) != len(encoded):
                _fail(REJECT_Q1_SEMANTIC_COVERAGE, f"{name} contains duplicates")
        if self.eligible_application_keys != self.processed_application_keys:
            _fail(REJECT_Q1_SEMANTIC_COVERAGE, "eligible and processed keys differ")
        application_id_preimages: dict[bytes, bytes] = {}
        for key in self.eligible_application_keys:
            _register_digest_preimage_v1(
                application_id_preimages,
                digest=_formal.semantic_application_id_v1(key),
                preimage=canonical_cbor_encode(key),
                label="semantic application ID",
            )
        strict_rows = self.strict_admission_preimages
        if type(strict_rows) is not tuple or any(
            type(row) is not tuple
            or len(row) != 2
            or type(row[0]) is not bytes
            or len(row[0]) != 32
            or type(row[1]) is not bytes
            or len(row[1]) != 32
            for row in strict_rows
        ):
            _fail(REJECT_Q1_SEMANTIC_COVERAGE, "strict preimages are malformed")
        if strict_rows != _canonical_order(strict_rows):
            _fail(REJECT_Q1_SEMANTIC_COVERAGE, "strict preimages are not canonical")
        if len({row[0] for row in strict_rows}) != len(strict_rows):
            _fail(REJECT_Q1_SEMANTIC_COVERAGE, "strict application IDs repeat")
        expected_ids = tuple(
            sorted(
                _formal.semantic_application_id_v1(key)
                for key in self.eligible_application_keys
            )
        )
        if tuple(sorted(row[0] for row in strict_rows)) != expected_ids:
            _fail(REJECT_Q1_SEMANTIC_COVERAGE, "strict/application ID sets differ")


@dataclass(frozen=True, slots=True)
class Q1SemanticCoverageArchiveV1:
    schema_version: str
    diagnostic_id: str
    input_signature_id: int
    universe_root: bytes
    coverage_records: tuple[Q1SemanticCoverageRecordV1, ...]
    coverage_preimages: tuple[Q1SemanticCoveragePreimageV1, ...]
    eligible_application_count: int
    processed_application_count: int
    strict_admitted_count: int
    rewrite_collapse_count: int
    formal_coverage_archive_root: None
    q1_state: str
    q1_gate_count: int
    q1_gate_mask: int
    target_truth_accessed: bool
    split_accessed: bool
    role_evaluation_performed: bool

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or type(self.diagnostic_id) is not str
            or type(self.q1_state) is not str
        ):
            _fail(REJECT_Q1_SEMANTIC_COVERAGE, "archive strings are not exact")
        if (
            self.schema_version != SEMANTIC_COVERAGE_DIAGNOSTIC_SCHEMA_VERSION
            or self.diagnostic_id != SEMANTIC_COVERAGE_DIAGNOSTIC_ID
        ):
            _fail(REJECT_Q1_SEMANTIC_COVERAGE, "archive identity differs")
        if type(self.input_signature_id) is not int or self.input_signature_id not in (1, 2):
            _fail(REJECT_Q1_SEMANTIC_COVERAGE, "input signature is invalid")
        if type(self.universe_root) is not bytes or len(self.universe_root) != 32:
            _fail(REJECT_Q1_SEMANTIC_COVERAGE, "universe root is invalid")
        if type(self.coverage_records) is not tuple or any(
            type(row) is not Q1SemanticCoverageRecordV1 for row in self.coverage_records
        ):
            _fail(REJECT_Q1_SEMANTIC_COVERAGE, "coverage records are malformed")
        if type(self.coverage_preimages) is not tuple or any(
            type(row) is not Q1SemanticCoveragePreimageV1
            for row in self.coverage_preimages
        ):
            _fail(REJECT_Q1_SEMANTIC_COVERAGE, "coverage preimages are malformed")
        if len(self.coverage_records) != _formal.EXPECTED_COVERAGE_RECORD_COUNT:
            _fail(REJECT_Q1_SEMANTIC_COVERAGE, "coverage record count is not 846")
        if len(self.coverage_preimages) != len(self.coverage_records):
            _fail(REJECT_Q1_SEMANTIC_COVERAGE, "coverage/preimage lengths differ")
        expected_registry = _formal.expected_coverage_registry_v1()
        if tuple(
            (row.construction_depth, row.coverage_code)
            for row in self.coverage_records
        ) != expected_registry or tuple(
            (row.construction_depth, row.coverage_code)
            for row in self.coverage_preimages
        ) != expected_registry:
            _fail(REJECT_Q1_SEMANTIC_COVERAGE, "coverage archive order differs")
        rfc6962_preimages: dict[bytes, bytes] = {}
        coverage_record_id_preimages: dict[bytes, bytes] = {}
        application_id_preimages: dict[bytes, bytes] = {}
        for record, preimage in zip(
            self.coverage_records,
            self.coverage_preimages,
            strict=True,
        ):
            eligible_root = rfc6962_root(preimage.eligible_application_keys)
            processed_root = rfc6962_root(preimage.processed_application_keys)
            strict_root = rfc6962_root(preimage.strict_admission_preimages)
            if (
                record.input_signature_id != self.input_signature_id
                or record.universe_root != self.universe_root
            ):
                _fail(REJECT_Q1_SEMANTIC_COVERAGE, "coverage binding differs")
            _register_digest_preimage_v1(
                coverage_record_id_preimages,
                digest=record.record_id,
                preimage=canonical_cbor_encode(record.canonical_object()),
                label="coverage record ID",
            )
            for key in preimage.eligible_application_keys:
                if len(key) != 8:
                    _fail(REJECT_Q1_SEMANTIC_COVERAGE, "application key arity differs")
                try:
                    replayed_key = _formal.semantic_application_key_v1(
                        key[2],
                        key[3],
                        key[4],
                        key[5],
                        key[6],
                        key[7],
                    )
                except (TypeError, ValueError) as error:
                    _fail(REJECT_Q1_SEMANTIC_COVERAGE, str(error))
                if replayed_key != key or (
                    key[2] != self.input_signature_id
                    or key[3] != self.universe_root
                    or key[4] != record.construction_depth
                    or key[5] != record.coverage_code
                ):
                    _fail(REJECT_Q1_SEMANTIC_COVERAGE, "application key binding differs")
                _register_digest_preimage_v1(
                    application_id_preimages,
                    digest=_formal.semantic_application_id_v1(key),
                    preimage=canonical_cbor_encode(key),
                    label="semantic application ID",
                )
            if (
                record.eligible_application_count
                != len(preimage.eligible_application_keys)
                or record.processed_application_count
                != len(preimage.processed_application_keys)
                or record.strict_admitted_count
                != len(preimage.strict_admission_preimages)
                or record.eligible_application_root
                != eligible_root
                or record.processed_application_root
                != processed_root
                or record.strict_admission_root
                != strict_root
                or record.unique_canonical_ast_count
                != len({row[1] for row in preimage.strict_admission_preimages})
            ):
                _fail(REJECT_Q1_SEMANTIC_COVERAGE, "coverage preimage replay differs")
            _register_rfc6962_preimage_v1(
                rfc6962_preimages,
                root=eligible_root,
                rows=preimage.eligible_application_keys,
                label="eligible application",
            )
            _register_rfc6962_preimage_v1(
                rfc6962_preimages,
                root=processed_root,
                rows=preimage.processed_application_keys,
                label="processed application",
            )
            _register_rfc6962_preimage_v1(
                rfc6962_preimages,
                root=strict_root,
                rows=preimage.strict_admission_preimages,
                label="strict admission",
            )
        for name in (
            "eligible_application_count",
            "processed_application_count",
            "strict_admitted_count",
            "rewrite_collapse_count",
            "q1_gate_count",
            "q1_gate_mask",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                _fail(REJECT_Q1_SEMANTIC_COVERAGE, f"{name} must be uint")
        if (
            self.eligible_application_count
            != sum(row.eligible_application_count for row in self.coverage_records)
            or self.processed_application_count
            != sum(row.processed_application_count for row in self.coverage_records)
            or self.strict_admitted_count
            != sum(row.strict_admitted_count for row in self.coverage_records)
            or self.rewrite_collapse_count
            != sum(row.rewrite_collapse_count for row in self.coverage_records)
        ):
            _fail(REJECT_Q1_SEMANTIC_COVERAGE, "coverage archive totals differ")
        if self.formal_coverage_archive_root is not None:
            _fail(REJECT_Q1_SEMANTIC_COVERAGE, "formal coverage root must stay null")
        if (
            self.q1_state != "NOT_RUN"
            or self.q1_gate_count != 0
            or self.q1_gate_mask != 0
            or type(self.target_truth_accessed) is not bool
            or self.target_truth_accessed is not False
            or type(self.split_accessed) is not bool
            or self.split_accessed is not False
            or type(self.role_evaluation_performed) is not bool
            or self.role_evaluation_performed is not False
        ):
            _fail(REJECT_Q1_SEMANTIC_COVERAGE, "downstream authority is not closed")


def _formal_program_id_by_ast(
    snapshot: Q1PartitionSnapshotV1,
) -> dict[bytes, bytes]:
    provisional: list[_formal.Q1RepresentativeProgramRecordV1] = []
    behavior_id_preimages: dict[bytes, bytes] = {}
    for class_row in snapshot.behavior_classes:
        output_sort = OutputSortId(class_row.output_sort_id)
        formal_cells: list[_formal.Q1BehaviorCellV1] = []
        for cell in class_row.behavior_cells:
            value = cell.runtime_value(class_row.output_sort_id)
            formal_cells.append(
                _formal.Q1BehaviorCellV1.bottom()
                if value is _adapter.BOTTOM
                else _formal.Q1BehaviorCellV1.exact(value)
            )
        behavior = _formal.Q1BehaviorBlobV1(
            snapshot.input_signature_id,
            snapshot.universe_root,
            output_sort,
            tuple(formal_cells),
        )
        prior_behavior = behavior_id_preimages.get(behavior.behavior_id)
        if prior_behavior is not None:
            if prior_behavior == behavior.canonical_bytes:
                _fail(
                    REJECT_Q1_SEMANTIC_COVERAGE,
                    "behavior identity occurs twice",
                )
            _fail(
                FAIL_SHA256_PREIMAGE_COLLISION,
                "behavior ID has different preimages",
            )
        behavior_id_preimages[behavior.behavior_id] = behavior.canonical_bytes
        for cohort in class_row.cohorts:
            for representative in cohort.representatives:
                provisional.append(
                    _formal.Q1RepresentativeProgramRecordV1(
                        snapshot.input_signature_id,
                        snapshot.universe_root,
                        0,
                        behavior.behavior_id,
                        representative.canonical_ast_cbor,
                        representative.canonical_ast_hash,
                        cohort.signature,
                    )
                )
    provisional.sort(key=lambda row: row.sort_key)
    output: dict[bytes, bytes] = {}
    id_preimages: dict[bytes, bytes] = {}
    for index, row in enumerate(provisional):
        record = _formal.Q1RepresentativeProgramRecordV1(
            row.input_signature_id,
            row.universe_root,
            index,
            row.class_id,
            row.canonical_ast_cbor,
            row.canonical_ast_hash,
            row.construction_signature,
        )
        prior = output.get(record.canonical_ast_cbor)
        if prior is not None:
            _fail(REJECT_Q1_SEMANTIC_COVERAGE, "AST occurs twice in program bank")
        identity_preimage = canonical_cbor_encode(
            (
                record.input_signature_id,
                record.universe_root,
                record.canonical_ast_cbor,
                record.canonical_ast_hash,
                _formal.construction_signature_object_v1(
                    record.construction_signature
                ),
            )
        )
        prior_preimage = id_preimages.get(record.program_id)
        if prior_preimage is not None:
            if prior_preimage == identity_preimage:
                _fail(REJECT_Q1_SEMANTIC_COVERAGE, "program identity occurs twice")
            _fail(
                FAIL_SHA256_PREIMAGE_COLLISION,
                "program ID has different preimages",
            )
        output[record.canonical_ast_cbor] = record.program_id
        id_preimages[record.program_id] = identity_preimage
    return output


def _build_q1_semantic_coverage_v1(
    snapshot: Q1PartitionSnapshotV1,
) -> Q1SemanticCoverageArchiveV1:
    validate_q1_partition_snapshot_v1(snapshot)
    program_ids = _formal_program_id_by_ast(snapshot)
    bank_ast_cbors = tuple(sorted(program_ids))
    candidates = _capacity.immutable_candidate_applications_v1(
        bank_ast_cbors,
        limits=snapshot.limits,
    )
    expected_registry = _formal.expected_coverage_registry_v1()
    grouped: dict[tuple[int, int], list[tuple[tuple[object, ...], bytes, bytes, bool]]] = {
        key: [] for key in expected_registry
    }
    application_id_preimages: dict[bytes, bytes] = {}
    ast_hash_preimages: dict[bytes, bytes] = {}
    for candidate in candidates:
        registry_key = (candidate.construction_depth, candidate.coverage_code)
        if registry_key not in grouped:
            _fail(FAIL_Q1_SEMANTIC_COVERAGE_REPLAY, "candidate coverage code is unknown")
        try:
            child_ids = tuple(
                program_ids[value]
                for value in candidate.ordered_child_canonical_ast_cbors
            )
        except KeyError:
            _fail(FAIL_Q1_SEMANTIC_COVERAGE_REPLAY, "candidate child is outside bank")
        application_key = _formal.semantic_application_key_v1(
            snapshot.input_signature_id,
            snapshot.universe_root,
            candidate.construction_depth,
            candidate.coverage_code,
            candidate.operator_parameters,
            child_ids,
        )
        if candidate.construction_depth == 0 and (
            candidate.operator_parameters or child_ids
        ):
            _fail(FAIL_Q1_SEMANTIC_COVERAGE_REPLAY, "leaf application binding differs")
        application_id = _formal.semantic_application_id_v1(application_key)
        application_bytes = canonical_cbor_encode(application_key)
        prior_application = application_id_preimages.get(application_id)
        if prior_application is not None:
            if prior_application != application_bytes:
                _fail(
                    FAIL_SHA256_PREIMAGE_COLLISION,
                    "application ID has different preimages",
                )
            _fail(REJECT_Q1_SEMANTIC_COVERAGE, "application key occurs twice")
        application_id_preimages[application_id] = application_bytes
        prior_ast = ast_hash_preimages.get(candidate.canonical_ast_hash)
        if prior_ast is not None and prior_ast != candidate.canonical_ast_cbor:
            _fail(
                FAIL_SHA256_PREIMAGE_COLLISION,
                "canonical AST digest has different preimages",
            )
        ast_hash_preimages[candidate.canonical_ast_hash] = candidate.canonical_ast_cbor
        grouped[registry_key].append(
            (
                application_key,
                application_id,
                candidate.canonical_ast_hash,
                candidate.rewrite_collapsed,
            )
        )

    records: list[Q1SemanticCoverageRecordV1] = []
    preimages: list[Q1SemanticCoveragePreimageV1] = []
    for construction_depth, coverage_code in expected_registry:
        material = grouped[(construction_depth, coverage_code)]
        application_keys = _canonical_order(tuple(row[0] for row in material))
        strict_rows = _canonical_order(tuple((row[1], row[2]) for row in material))
        application_root = rfc6962_root(application_keys)
        strict_root = rfc6962_root(strict_rows)
        record = Q1SemanticCoverageRecordV1(
            snapshot.input_signature_id,
            snapshot.universe_root,
            construction_depth,
            coverage_code,
            len(application_keys),
            application_root,
            len(application_keys),
            application_root,
            len(strict_rows),
            strict_root,
            len({row[2] for row in material}),
            sum(int(row[3]) for row in material),
        )
        records.append(record)
        preimages.append(
            Q1SemanticCoveragePreimageV1(
                construction_depth=construction_depth,
                coverage_code=coverage_code,
                eligible_application_keys=application_keys,
                processed_application_keys=application_keys,
                strict_admission_preimages=strict_rows,
            )
        )
    record_tuple = tuple(records)
    _formal.canonical_archive_order_v1(
        record_tuple,
        stream_kind_id=_formal.ArchiveStreamKindId.COVERAGE,
    )
    archive = Q1SemanticCoverageArchiveV1(
        schema_version=SEMANTIC_COVERAGE_DIAGNOSTIC_SCHEMA_VERSION,
        diagnostic_id=SEMANTIC_COVERAGE_DIAGNOSTIC_ID,
        input_signature_id=snapshot.input_signature_id,
        universe_root=snapshot.universe_root,
        coverage_records=record_tuple,
        coverage_preimages=tuple(preimages),
        eligible_application_count=sum(
            row.eligible_application_count for row in record_tuple
        ),
        processed_application_count=sum(
            row.processed_application_count for row in record_tuple
        ),
        strict_admitted_count=sum(row.strict_admitted_count for row in record_tuple),
        rewrite_collapse_count=sum(row.rewrite_collapse_count for row in record_tuple),
        formal_coverage_archive_root=None,
        q1_state="NOT_RUN",
        q1_gate_count=0,
        q1_gate_mask=0,
        target_truth_accessed=False,
        split_accessed=False,
        role_evaluation_performed=False,
    )
    if (
        archive.eligible_application_count != snapshot.raw_operator_application_count
        or archive.processed_application_count != snapshot.raw_operator_application_count
        or archive.strict_admitted_count != snapshot.strict_admitted_application_count
        or archive.rewrite_collapse_count != snapshot.rewrite_collapse_count
    ):
        _fail(FAIL_Q1_SEMANTIC_COVERAGE_REPLAY, "coverage totals differ from snapshot")
    barrier_by_depth = {
        row.depth: row
        for row in snapshot.depth_barriers
        if row.depth <= snapshot.limits.maximum_ast_depth
    }
    for depth in range(snapshot.limits.maximum_ast_depth + 1):
        depth_records = tuple(row for row in record_tuple if row.construction_depth == depth)
        barrier = barrier_by_depth[depth]
        if (
            sum(row.eligible_application_count for row in depth_records)
            != barrier.eligible_raw_application_count
            or sum(row.strict_admitted_count for row in depth_records)
            != barrier.strict_admitted_application_count
            or sum(row.rewrite_collapse_count for row in depth_records)
            != barrier.rewrite_collapse_count
        ):
            _fail(FAIL_Q1_SEMANTIC_COVERAGE_REPLAY, "coverage depth totals differ")
    return archive


def build_q1_semantic_coverage_v1(
    snapshot: Q1PartitionSnapshotV1,
) -> Q1SemanticCoverageArchiveV1:
    """Build 846 canonical diagnostic records from one immutable snapshot."""

    if type(snapshot) is not Q1PartitionSnapshotV1:
        raise TypeError("snapshot must be Q1PartitionSnapshotV1")
    return _build_q1_semantic_coverage_v1(snapshot)


def validate_q1_semantic_coverage_v1(
    archive: Q1SemanticCoverageArchiveV1,
    snapshot: Q1PartitionSnapshotV1,
) -> None:
    """Rebuild schedule-neutrally and require exact immutable equality."""

    if type(archive) is not Q1SemanticCoverageArchiveV1:
        raise TypeError("archive must be Q1SemanticCoverageArchiveV1")
    if type(snapshot) is not Q1PartitionSnapshotV1:
        raise TypeError("snapshot must be Q1PartitionSnapshotV1")
    expected = _build_q1_semantic_coverage_v1(snapshot)
    if archive != expected:
        _fail(FAIL_Q1_SEMANTIC_COVERAGE_REPLAY, "coverage archive replay differs")


__all__ = [
    "FAIL_SHA256_PREIMAGE_COLLISION",
    "FAIL_Q1_SEMANTIC_COVERAGE_REPLAY",
    "Q1SemanticCoverageArchiveV1",
    "Q1SemanticCoverageError",
    "Q1SemanticCoveragePreimageV1",
    "Q1SemanticCoverageRecordV1",
    "REJECT_Q1_SEMANTIC_COVERAGE",
    "SEMANTIC_COVERAGE_DIAGNOSTIC_ID",
    "SEMANTIC_COVERAGE_DIAGNOSTIC_SCHEMA_VERSION",
    "build_q1_semantic_coverage_v1",
    "validate_q1_semantic_coverage_v1",
]
