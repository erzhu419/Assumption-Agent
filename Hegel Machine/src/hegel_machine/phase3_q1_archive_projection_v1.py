"""Bounded local Q1 archive-wire projection for Q0.5 qualification.

The materialized path intentionally retains framed bytes so tests can tamper and
replay every preimage.  A separate bounded-node3 counting/discard path replays
the same frozen wire without retaining framed blobs and can be compared against
the materializer.  Neither path is the future admitted full-node6 endpoint and
neither can populate a formal Q1 output slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from typing import Callable, Final, Iterable, NoReturn, Sequence, TypeVar

from . import phase3_q0_input_adapter_v1 as _adapter
from .phase3_q0_evaluator_v1 import evaluate_canonical_ast_on_environments_v1
from .phase3_q1_formal_archive_contract_v1 import (
    ARCHIVE_CHUNK_MANIFEST_SCHEMA_ID,
    ArchiveStreamKindId,
    BEHAVIOR_ID_DOMAIN,
    BEHAVIOR_BLOB_SCHEMA_ID,
    CLASS_RECORD_ID_DOMAIN,
    COHORT_ID_DOMAIN,
    COHORT_RECORD_ID_DOMAIN,
    CONSTRUCTION_SIGNATURE_ID_DOMAIN,
    CONSTRUCTION_SIGNATURE_SCHEMA_ID,
    CONTINUATION_COHORT_SCHEMA_ID,
    COVERAGE_RECORD_ID_DOMAIN,
    FRAMED_BLOB_HASH_DOMAIN,
    FRAME_LENGTH_BYTES,
    MAX_CHUNK_FRAMED_BYTES,
    MAX_RECORDS_PER_CHUNK,
    PROGRAM_ID_DOMAIN,
    PROGRAM_RECORD_ID_DOMAIN,
    QUOTIENT_CLASS_SCHEMA_ID,
    REPRESENTATIVE_PROGRAM_SCHEMA_ID,
    SEMANTIC_COVERAGE_SCHEMA_ID,
    Q1_BEHAVIOR_BLOB_TAG,
    Q1_ARCHIVE_CHUNK_MANIFEST_TAG,
    Q1_CONSTRUCTION_SIGNATURE_TAG,
    Q1_CONTINUATION_COHORT_TAG,
    Q1_QUOTIENT_CLASS_TAG,
    Q1_REPRESENTATIVE_PROGRAM_TAG,
    Q1_SEMANTIC_COVERAGE_TAG,
    Q1ArchiveChunkManifestV1,
    Q1BehaviorBlobV1,
    Q1BehaviorCellV1,
    Q1CohortWitnessV1,
    Q1ContinuationCohortRecordV1,
    Q1QuotientClassRecordV1,
    Q1RepresentativeProgramRecordV1,
    Q1SemanticCoverageRecordV1,
    Q1StreamDescriptorV1,
    STREAM_DESCRIPTOR_SCHEMA_ID,
    archive_root_v1,
    canonical_archive_order_v1,
    construction_signature_object_v1,
    frame_canonical_record_v1,
    framed_blob_hash_v1,
    replay_framed_records_v1,
)
from .phase3_q1_external_sort_profile_v1 import (
    EXTERNAL_SORT_PROJECTION_SCHEMA_ID,
    Q1ExternalSortProjectionV1,
    project_external_sort_v1,
)
from .phase3_q1_partition_snapshot_v1 import (
    Q1BehaviorCellSnapshotV1,
    Q1PartitionSnapshotV1,
    validate_q1_partition_snapshot_v1,
)
from .phase3_q1_quotient_contract_v1 import (
    FutureAdmissibilitySignatureV1,
    NormalizationProfileId,
    OutputSortId,
)
from .phase3_q1_universe_v1 import production_universe_v1
from .strict_ast_shrink6_v1 import decode_shrink6_canonical_ast
from .strict_cbor_v1 import canonical_cbor_encode, content_hash, rfc6962_root


PROJECTED_STREAM_SCHEMA_ID: Final = b"hegel-q1-projected-record-stream/1"
COUNTING_DISCARD_STREAM_SCHEMA_ID: Final = (
    b"hegel-q05b-counting-discard-record-stream/1"
)
SNAPSHOT_RECORD_SET_SCHEMA_ID: Final = b"hegel-q1-snapshot-record-set/1"
PROJECTED_STREAM_ROOT_DOMAIN: Final = "HEGEL/Q1/PREFLIGHT/PROJECTED_STREAM/V1"
SNAPSHOT_RECORD_SET_ROOT_DOMAIN: Final = (
    "HEGEL/Q1/PREFLIGHT/SNAPSHOT_RECORD_SET/V1"
)
_RecordT = TypeVar("_RecordT")


class Q1ArchiveProjectionError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Q1ArchiveProjectionError(code, detail)


def _record_id(record: object) -> bytes:
    value = getattr(record, "record_id", None)
    if type(value) is not bytes or len(value) != 32:
        _fail("REJECT_Q1_PROJECTED_RECORD", "record has no exact 32-byte record ID")
    return value


def _program_identity_preimage_v1(
    row: Q1RepresentativeProgramRecordV1,
) -> bytes:
    return canonical_cbor_encode(
        (
            row.input_signature_id,
            row.universe_root,
            row.canonical_ast_cbor,
            row.canonical_ast_hash,
            construction_signature_object_v1(row.construction_signature),
        )
    )


def _cohort_identity_preimage_v1(
    row: Q1ContinuationCohortRecordV1,
) -> bytes:
    return canonical_cbor_encode(
        (
            row.input_signature_id,
            row.universe_root,
            row.class_id,
            row.signature_id,
        )
    )


def _class_identity_preimage_v1(row: Q1QuotientClassRecordV1) -> bytes:
    return row.behavior.canonical_bytes


def _unique_semantic_id_rows_v1(
    rows: tuple[_RecordT, ...],
    *,
    id_attribute: str,
    identity_preimage: Callable[[_RecordT], bytes],
    duplicate_code: str,
    label: str,
) -> dict[bytes, _RecordT]:
    """Reject exact duplicate identities separately from digest collisions."""

    seen_preimages: dict[bytes, bytes] = {}
    rows_by_id: dict[bytes, _RecordT] = {}
    for row in rows:
        identity = getattr(row, id_attribute)
        preimage = identity_preimage(row)
        previous = seen_preimages.get(identity)
        if previous is not None:
            if previous == preimage:
                _fail(duplicate_code, f"duplicate {label} semantic identity")
            _fail(
                "FAIL_SHA256_PREIMAGE_COLLISION",
                f"{label} semantic ID has different preimages",
            )
        seen_preimages[identity] = preimage
        rows_by_id[identity] = row
    return rows_by_id


def _require_same_id_preimage_v1(
    *,
    claimed_id: bytes,
    replayed_id: bytes,
    claimed_preimage: bytes,
    replayed_preimage: bytes,
    reject_detail: str,
    collision_detail: str,
) -> None:
    if replayed_id != claimed_id:
        _fail("REJECT_Q1_RECORD_SET", reject_detail)
    if replayed_preimage != claimed_preimage:
        _fail("FAIL_SHA256_PREIMAGE_COLLISION", collision_detail)


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
            "FAIL_SHA256_PREIMAGE_COLLISION",
            f"{label} digest has different preimages",
        )
    seen_preimages[digest] = preimage


def _register_unique_digest_preimage_v1(
    seen_preimages: dict[bytes, bytes],
    *,
    digest: bytes,
    preimage: bytes,
    duplicate_code: str,
    label: str,
) -> None:
    previous = seen_preimages.get(digest)
    if previous is not None:
        if previous == preimage:
            _fail(duplicate_code, f"duplicate {label} preimage")
        _fail(
            "FAIL_SHA256_PREIMAGE_COLLISION",
            f"{label} digest has different preimages",
        )
    seen_preimages[digest] = preimage


def _decode_signature_v1(value: object) -> FutureAdmissibilitySignatureV1:
    if (
        type(value) is not tuple
        or len(value) != 14
        or value[:3]
        != (1, Q1_CONSTRUCTION_SIGNATURE_TAG, CONSTRUCTION_SIGNATURE_SCHEMA_ID)
        or any(type(item) is not int for item in value[3:])
    ):
        _fail("REJECT_Q1_PROJECTED_RECORD", "construction signature wire differs")
    try:
        signature = FutureAdmissibilitySignatureV1(
            output_sort_id=OutputSortId(value[3]),
            ast_depth=value[4],
            ast_node_count=value[5],
            scalar_parameter_occurrence_count=value[6],
            aggregate_leaf_count=value[7],
            distinct_bit_slot_bitmap=value[8],
            scope_clause_count=value[9],
            top_level_clause_count=value[10],
            old_law_composition_depth=value[11],
            normalization_profile_id=NormalizationProfileId(value[12]),
            mdl_length_q32=value[13],
        )
    except (TypeError, ValueError) as error:
        _fail("REJECT_Q1_PROJECTED_RECORD", f"construction signature: {error}")
    if canonical_cbor_encode(construction_signature_object_v1(signature)) != (
        canonical_cbor_encode(value)
    ):
        _fail("REJECT_Q1_PROJECTED_RECORD", "construction signature replay differs")
    return signature


def _decode_behavior_cell_v1(
    value: object,
    output_sort_id: OutputSortId,
) -> Q1BehaviorCellV1:
    if type(value) is not tuple:
        _fail("REJECT_Q1_PROJECTED_RECORD", "behavior cell must be exact tuple")
    if len(value) == 1 and type(value[0]) is int and value[0] == 0:
        return Q1BehaviorCellV1.bottom()
    if len(value) != 2 or value[0] != 1 or type(value[0]) is not int:
        _fail("REJECT_Q1_PROJECTED_RECORD", "behavior cell wire differs")
    payload = value[1]
    if output_sort_id is OutputSortId.RATIONAL_VALUE:
        if (
            type(payload) is not tuple
            or len(payload) != 2
            or type(payload[0]) is not int
            or type(payload[1]) is not int
            or payload[1] == 0
        ):
            _fail("REJECT_Q1_PROJECTED_RECORD", "rational behavior cell differs")
        payload = Fraction(payload[0], payload[1])
    cell = Q1BehaviorCellV1.exact(payload)
    try:
        replay = cell.canonical_object(output_sort_id)
    except (TypeError, ValueError) as error:
        _fail("REJECT_Q1_PROJECTED_RECORD", f"behavior cell: {error}")
    if canonical_cbor_encode(replay) != canonical_cbor_encode(value):
        _fail("REJECT_Q1_PROJECTED_RECORD", "behavior cell replay differs")
    return cell


def _decode_behavior_blob_v1(value: object) -> Q1BehaviorBlobV1:
    if (
        type(value) is not tuple
        or len(value) != 8
        or value[:3] != (1, Q1_BEHAVIOR_BLOB_TAG, BEHAVIOR_BLOB_SCHEMA_ID)
        or type(value[3]) is not int
        or type(value[5]) is not int
        or type(value[6]) is not int
        or type(value[7]) is not tuple
    ):
        _fail("REJECT_Q1_PROJECTED_RECORD", "behavior blob wire differs")
    try:
        output_sort_id = OutputSortId(value[5])
        cells = tuple(
            _decode_behavior_cell_v1(cell, output_sort_id) for cell in value[7]
        )
        behavior = Q1BehaviorBlobV1(value[3], value[4], output_sort_id, cells)
    except (TypeError, ValueError) as error:
        _fail("REJECT_Q1_PROJECTED_RECORD", f"behavior blob: {error}")
    if value[6] != len(cells) or canonical_cbor_encode(
        behavior.canonical_object()
    ) != canonical_cbor_encode(value):
        _fail("REJECT_Q1_PROJECTED_RECORD", "behavior blob replay differs")
    return behavior


def _strict_decode_formal_record_v1(
    value: tuple[object, ...],
    stream_kind_id: ArchiveStreamKindId,
) -> object:
    """Decode through the authoritative dataclass and require exact re-encode."""

    if type(value) is not tuple:
        _fail("REJECT_Q1_PROJECTED_RECORD", "canonical record must be exact tuple")
    try:
        if stream_kind_id is ArchiveStreamKindId.PROGRAM:
            if len(value) != 12:
                raise ValueError("program record length differs")
            record: object = Q1RepresentativeProgramRecordV1(
                input_signature_id=value[3],
                universe_root=value[4],
                program_index=value[5],
                class_id=value[7],
                canonical_ast_cbor=value[8],
                canonical_ast_hash=value[9],
                construction_signature=_decode_signature_v1(value[10]),
            )
        elif stream_kind_id is ArchiveStreamKindId.COHORT:
            if len(value) != 14 or type(value[12]) is not tuple:
                raise ValueError("cohort record wire differs")
            witnesses = tuple(
                Q1CohortWitnessV1(*witness)
                if type(witness) is tuple and len(witness) == 3
                else (_fail("REJECT_Q1_PROJECTED_RECORD", "cohort witness differs"))
                for witness in value[12]
            )
            record = Q1ContinuationCohortRecordV1(
                input_signature_id=value[3],
                universe_root=value[4],
                cohort_index=value[5],
                class_id=value[7],
                construction_signature=_decode_signature_v1(value[8]),
                witnesses=witnesses,
                visible_frontier_cohort=value[13],
            )
        elif stream_kind_id is ArchiveStreamKindId.CLASS:
            if len(value) != 16:
                raise ValueError("class record length differs")
            record = Q1QuotientClassRecordV1(
                input_signature_id=value[3],
                universe_root=value[4],
                class_index=value[5],
                behavior=_decode_behavior_blob_v1(value[6]),
                first_cohort_index=value[8],
                cohort_count=value[9],
                class_cohort_subtree_root=value[10],
                bank_point_count=value[11],
                visible_cohort_count=value[12],
                visible_frontier_point_count=value[13],
                visible_frontier_subtree_root=value[14],
                minimum_mdl_q32=value[15],
            )
        elif stream_kind_id is ArchiveStreamKindId.COVERAGE:
            if len(value) != 15:
                raise ValueError("coverage record length differs")
            record = Q1SemanticCoverageRecordV1(
                input_signature_id=value[3],
                universe_root=value[4],
                construction_depth=value[5],
                coverage_code=value[6],
                eligible_application_count=value[7],
                eligible_application_root=value[8],
                processed_application_count=value[9],
                processed_application_root=value[10],
                strict_admitted_count=value[11],
                strict_admission_root=value[12],
                unique_canonical_ast_count=value[13],
                rewrite_collapse_count=value[14],
            )
        else:
            raise TypeError("stream_kind_id must be ArchiveStreamKindId")
    except Q1ArchiveProjectionError:
        raise
    except (IndexError, TypeError, ValueError) as error:
        _fail("REJECT_Q1_PROJECTED_RECORD", str(error))
    if canonical_cbor_encode(record.canonical_object()) != canonical_cbor_encode(  # type: ignore[attr-defined]
        value
    ):
        _fail("REJECT_Q1_PROJECTED_RECORD", "formal dataclass replay differs")
    return record


def _canonical_record_identity_and_sort_key_v1(
    value: tuple[object, ...],
    stream_kind_id: ArchiveStreamKindId,
) -> tuple[bytes, bytes]:
    """Replay one formal record object into its record ID and order-preserving key."""

    record = _strict_decode_formal_record_v1(value, stream_kind_id)
    if stream_kind_id is ArchiveStreamKindId.PROGRAM:
        key = _stream_sort_key(record, stream_kind_id)
        domain = PROGRAM_RECORD_ID_DOMAIN
    elif stream_kind_id is ArchiveStreamKindId.COHORT:
        key = _stream_sort_key(record, stream_kind_id)
        domain = COHORT_RECORD_ID_DOMAIN
    elif stream_kind_id is ArchiveStreamKindId.CLASS:
        key = _stream_sort_key(record, stream_kind_id)
        domain = CLASS_RECORD_ID_DOMAIN
    elif stream_kind_id is ArchiveStreamKindId.COVERAGE:
        key = _stream_sort_key(record, stream_kind_id)
        domain = COVERAGE_RECORD_ID_DOMAIN
    else:
        raise TypeError("stream_kind_id must be ArchiveStreamKindId")
    return content_hash(domain, value), key


def _stream_sort_key(record: object, stream_kind_id: ArchiveStreamKindId) -> bytes:
    if stream_kind_id is ArchiveStreamKindId.PROGRAM:
        depth, nodes, sort_id, root_operator_id, ast_cbor = record.sort_key  # type: ignore[attr-defined]
        if not (
            type(depth) is int
            and 0 <= depth <= 0xFF
            and type(nodes) is int
            and 0 <= nodes <= 0xFFFF
            and type(sort_id) is int
            and 0 <= sort_id <= 0xFF
            and type(root_operator_id) is int
            and 0 <= root_operator_id <= 0xFFFF
            and type(ast_cbor) is bytes
        ):
            _fail("REJECT_Q1_SORT_KEY", "program sort-key component is outside wire")
        return (
            bytes((depth,))
            + nodes.to_bytes(2, "big")
            + bytes((sort_id,))
            + root_operator_id.to_bytes(2, "big")
            + ast_cbor
        )
    elif stream_kind_id is ArchiveStreamKindId.COHORT:
        return (
            record.class_id  # type: ignore[attr-defined]
            + record.signature_id  # type: ignore[attr-defined]
            + canonical_cbor_encode(
                construction_signature_object_v1(record.construction_signature)  # type: ignore[attr-defined]
            )
        )
    elif stream_kind_id is ArchiveStreamKindId.CLASS:
        return (
            record.class_id  # type: ignore[attr-defined]
            + record.behavior.canonical_bytes  # type: ignore[attr-defined]
        )
    elif stream_kind_id is ArchiveStreamKindId.COVERAGE:
        depth = record.construction_depth  # type: ignore[attr-defined]
        coverage_code = record.coverage_code  # type: ignore[attr-defined]
        if not (
            type(depth) is int
            and 0 <= depth <= 0xFF
            and type(coverage_code) is int
            and 0 <= coverage_code <= 0xFFFF
        ):
            _fail("REJECT_Q1_SORT_KEY", "coverage sort-key component is outside wire")
        return bytes((depth,)) + coverage_code.to_bytes(2, "big")
    else:
        raise TypeError("stream_kind_id must be ArchiveStreamKindId")


@dataclass(frozen=True, slots=True)
class Q1ChunkProjectionV1:
    manifests: tuple[Q1ArchiveChunkManifestV1, ...]
    framed_blobs: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if type(self.manifests) is not tuple or type(self.framed_blobs) is not tuple:
            _fail("REJECT_Q1_CHUNK_PROJECTION", "chunk collections must be tuples")
        if len(self.manifests) != len(self.framed_blobs):
            _fail("REJECT_Q1_CHUNK_PROJECTION", "manifest/blob counts differ")
        next_record_index = 0
        all_records: list[tuple[object, ...]] = []
        all_record_ids: list[bytes] = []
        record_id_preimages: dict[bytes, bytes] = {}
        framed_blob_preimages: dict[bytes, bytes] = {}
        manifest_id_preimages: dict[bytes, bytes] = {}
        record_subtree_preimages: dict[bytes, bytes] = {}
        for index, (manifest, blob) in enumerate(
            zip(self.manifests, self.framed_blobs, strict=True)
        ):
            if type(manifest) is not Q1ArchiveChunkManifestV1 or type(blob) is not bytes:
                _fail("REJECT_Q1_CHUNK_PROJECTION", "chunk entry type differs")
            _register_digest_preimage_v1(
                manifest_id_preimages,
                digest=manifest.record_id,
                preimage=canonical_cbor_encode(manifest.canonical_object()),
                label="chunk manifest record",
            )
            if manifest.chunk_index != index:
                _fail("REJECT_Q1_CHUNK_PROJECTION", "chunk indices are not contiguous")
            if manifest.first_record_index != next_record_index:
                _fail(
                    "REJECT_Q1_CHUNK_PROJECTION",
                    "chunk record ranges are not contiguous",
                )
            if manifest.framed_blob_length != len(blob):
                _fail("REJECT_Q1_CHUNK_PROJECTION", "chunk byte length differs")
            if manifest.framed_blob_hash != framed_blob_hash_v1(blob):
                _fail("REJECT_Q1_CHUNK_PROJECTION", "chunk blob hash differs")
            _register_digest_preimage_v1(
                framed_blob_preimages,
                digest=manifest.framed_blob_hash,
                preimage=canonical_cbor_encode((blob,)),
                label="framed blob",
            )
            records = replay_framed_records_v1(blob)
            if len(records) != manifest.record_count:
                _fail("REJECT_Q1_CHUNK_PROJECTION", "chunk record count differs")
            if rfc6962_root(records) != manifest.record_subtree_root:
                _fail("REJECT_Q1_CHUNK_PROJECTION", "chunk record root differs")
            _register_digest_preimage_v1(
                record_subtree_preimages,
                digest=manifest.record_subtree_root,
                preimage=canonical_cbor_encode(records),
                label="chunk RFC6962 record subtree",
            )
            record_ids = tuple(
                _canonical_record_identity_and_sort_key_v1(
                    record,
                    manifest.stream_kind_id,
                )[0]
                for record in records
            )
            for record, record_id in zip(records, record_ids, strict=True):
                _register_unique_digest_preimage_v1(
                    record_id_preimages,
                    digest=record_id,
                    preimage=canonical_cbor_encode(record),
                    duplicate_code="REJECT_Q1_CHUNK_PROJECTION",
                    label="formal record ID",
                )
            if (
                manifest.first_record_id != record_ids[0]
                or manifest.last_record_id != record_ids[-1]
            ):
                _fail("REJECT_Q1_CHUNK_PROJECTION", "chunk record ID boundary differs")
            all_records.extend(records)
            all_record_ids.extend(record_ids)
            next_record_index += manifest.record_count
        if not self.manifests:
            return
        binding = (
            self.manifests[0].input_signature_id,
            self.manifests[0].universe_root,
            self.manifests[0].stream_kind_id,
        )
        if any(
            (
                manifest.input_signature_id,
                manifest.universe_root,
                manifest.stream_kind_id,
            )
            != binding
            for manifest in self.manifests
        ):
            _fail("REJECT_Q1_CHUNK_PROJECTION", "chunk bindings differ")

        frames = tuple(frame_canonical_record_v1(record) for record in all_records)
        expected_manifests: list[Q1ArchiveChunkManifestV1] = []
        expected_blobs: list[bytes] = []
        start = 0
        while start < len(all_records):
            end = start
            payload_bytes = 0
            while end < len(all_records):
                next_size = len(frames[end])
                if next_size > MAX_CHUNK_FRAMED_BYTES:
                    _fail(
                        "INCONCLUSIVE_RESOURCE_LIMIT",
                        "OUTPUT_BYTES: one record exceeds chunk",
                    )
                if end > start and (
                    end - start + 1 > MAX_RECORDS_PER_CHUNK
                    or payload_bytes + next_size > MAX_CHUNK_FRAMED_BYTES
                ):
                    break
                payload_bytes += next_size
                end += 1
                if end - start == MAX_RECORDS_PER_CHUNK:
                    break
            blob = b"".join(frames[start:end])
            subset = tuple(all_records[start:end])
            expected_manifests.append(
                Q1ArchiveChunkManifestV1(
                    binding[0],
                    binding[1],
                    binding[2],
                    len(expected_manifests),
                    start,
                    len(subset),
                    all_record_ids[start],
                    all_record_ids[end - 1],
                    rfc6962_root(subset),
                    framed_blob_hash_v1(blob),
                    len(blob),
                )
            )
            expected_blobs.append(blob)
            start = end
        if self.manifests != tuple(expected_manifests) or self.framed_blobs != tuple(
            expected_blobs
        ):
            _fail(
                "REJECT_Q1_CHUNK_PROJECTION",
                "chunk close policy differs from the canonical deterministic chunker",
            )


@dataclass(frozen=True, slots=True)
class Q1ProjectedStreamV1:
    input_signature_id: int
    universe_root: bytes
    stream_kind_id: ArchiveStreamKindId
    descriptor: Q1StreamDescriptorV1
    chunks: Q1ChunkProjectionV1
    external_sort_projection: Q1ExternalSortProjectionV1
    diagnostic_commitment: bytes

    def __post_init__(self) -> None:
        self.canonical_diagnostic_object()

    def _commitment_preimage(self) -> tuple[object, ...]:
        return (
            1,
            PROJECTED_STREAM_SCHEMA_ID,
            self.input_signature_id,
            self.universe_root,
            int(self.stream_kind_id),
            self.descriptor.canonical_object(),
            tuple(manifest.canonical_object() for manifest in self.chunks.manifests),
            self.external_sort_projection.canonical_object(),
        )

    def canonical_diagnostic_object(self) -> tuple[object, ...]:
        if type(self.input_signature_id) is not int or self.input_signature_id not in (1, 2):
            _fail("REJECT_Q1_PROJECTED_STREAM", "input signature differs")
        if type(self.universe_root) is not bytes or len(self.universe_root) != 32:
            _fail("REJECT_Q1_PROJECTED_STREAM", "universe root differs")
        if (
            self.universe_root
            != production_universe_v1(self.input_signature_id).universe_root
        ):
            _fail("REJECT_Q1_PROJECTED_STREAM", "universe binding differs")
        if not isinstance(self.stream_kind_id, ArchiveStreamKindId):
            _fail("REJECT_Q1_PROJECTED_STREAM", "stream kind differs")
        if type(self.descriptor) is not Q1StreamDescriptorV1:
            _fail("REJECT_Q1_PROJECTED_STREAM", "descriptor type differs")
        if type(self.chunks) is not Q1ChunkProjectionV1:
            _fail("REJECT_Q1_PROJECTED_STREAM", "chunks type differs")
        if type(self.external_sort_projection) is not Q1ExternalSortProjectionV1:
            _fail("REJECT_Q1_PROJECTED_STREAM", "sort projection type differs")
        if type(self.diagnostic_commitment) is not bytes or len(self.diagnostic_commitment) != 32:
            _fail("REJECT_Q1_PROJECTED_STREAM", "diagnostic commitment differs")
        if self.descriptor.stream_kind_id is not self.stream_kind_id:
            _fail("REJECT_Q1_PROJECTED_STREAM", "descriptor stream kind differs")
        manifests = self.chunks.manifests
        if any(
            manifest.input_signature_id != self.input_signature_id
            or manifest.universe_root != self.universe_root
            or manifest.stream_kind_id is not self.stream_kind_id
            for manifest in manifests
        ):
            _fail("REJECT_Q1_PROJECTED_STREAM", "chunk binding differs")
        records = tuple(
            record
            for blob in self.chunks.framed_blobs
            for record in replay_framed_records_v1(blob)
        )
        if not records:
            _fail("REJECT_Q1_PROJECTED_STREAM", "formal record stream must be nonempty")
        decoded_records = tuple(
            _strict_decode_formal_record_v1(record, self.stream_kind_id)
            for record in records
        )
        if any(
            record.input_signature_id != self.input_signature_id
            or record.universe_root != self.universe_root
            for record in decoded_records
        ):
            _fail("REJECT_Q1_PROJECTED_STREAM", "record/stream binding differs")
        if self.stream_kind_id in (
            ArchiveStreamKindId.PROGRAM,
            ArchiveStreamKindId.COHORT,
        ):
            signature_preimages: dict[bytes, bytes] = {}
            for record in decoded_records:
                _register_digest_preimage_v1(
                    signature_preimages,
                    digest=record.signature_id,
                    preimage=canonical_cbor_encode(
                        construction_signature_object_v1(
                            record.construction_signature
                        )
                    ),
                    label="construction signature",
                )
        if self.stream_kind_id is ArchiveStreamKindId.PROGRAM:
            ast_preimages: dict[bytes, bytes] = {}
            for record in decoded_records:
                _register_digest_preimage_v1(
                    ast_preimages,
                    digest=record.canonical_ast_hash,
                    preimage=record.canonical_ast_cbor,
                    label="strict AST",
                )
            _unique_semantic_id_rows_v1(
                decoded_records,
                id_attribute="program_id",
                identity_preimage=_program_identity_preimage_v1,
                duplicate_code="REJECT_Q1_PROJECTED_STREAM",
                label="program",
            )
        elif self.stream_kind_id is ArchiveStreamKindId.COHORT:
            _unique_semantic_id_rows_v1(
                decoded_records,
                id_attribute="cohort_id",
                identity_preimage=_cohort_identity_preimage_v1,
                duplicate_code="REJECT_Q1_PROJECTED_STREAM",
                label="cohort",
            )
        elif self.stream_kind_id is ArchiveStreamKindId.CLASS:
            _unique_semantic_id_rows_v1(
                decoded_records,
                id_attribute="class_id",
                identity_preimage=_class_identity_preimage_v1,
                duplicate_code="REJECT_Q1_PROJECTED_STREAM",
                label="class",
            )
        canonical_archive_order_v1(
            decoded_records,
            stream_kind_id=self.stream_kind_id,
        )
        manifest_objects = tuple(manifest.canonical_object() for manifest in manifests)
        if (
            self.descriptor.record_count != len(records)
            or self.descriptor.archive_root != rfc6962_root(records)
            or self.descriptor.framed_stream_bytes
            != sum(len(blob) for blob in self.chunks.framed_blobs)
            or self.descriptor.chunk_count != len(manifests)
            or self.descriptor.chunk_manifest_subtree_root
            != rfc6962_root(manifest_objects)
        ):
            _fail("REJECT_Q1_PROJECTED_STREAM", "stream descriptor replay differs")
        rfc6962_preimages: dict[bytes, bytes] = {}
        _register_digest_preimage_v1(
            rfc6962_preimages,
            digest=self.descriptor.archive_root,
            preimage=canonical_cbor_encode(records),
            label="full stream RFC6962 archive",
        )
        for manifest in manifests:
            start = manifest.first_record_index
            end = start + manifest.record_count
            _register_digest_preimage_v1(
                rfc6962_preimages,
                digest=manifest.record_subtree_root,
                preimage=canonical_cbor_encode(records[start:end]),
                label="chunk RFC6962 record subtree",
            )
        _register_digest_preimage_v1(
            rfc6962_preimages,
            digest=self.descriptor.chunk_manifest_subtree_root,
            preimage=canonical_cbor_encode(manifest_objects),
            label="chunk-manifest RFC6962 subtree",
        )
        sort_projection = self.external_sort_projection
        if (
            sort_projection.input_signature_id != self.input_signature_id
            or sort_projection.stream_kind_id is not self.stream_kind_id
            or sort_projection.record_count != len(records)
        ):
            _fail("REJECT_Q1_PROJECTED_STREAM", "external-sort binding differs")
        replayed_sort_rows = tuple(
            (
                _stream_sort_key(decoded, self.stream_kind_id),
                canonical_cbor_encode(record),
            )
            for record, decoded in zip(records, decoded_records, strict=True)
        )
        expected_sort_projection = project_external_sort_v1(
            replayed_sort_rows,
            input_signature_id=self.input_signature_id,
            stream_kind_id=self.stream_kind_id,
        )
        if sort_projection != expected_sort_projection:
            _fail("REJECT_Q1_PROJECTED_STREAM", "external-sort replay differs")
        expected_commitment = content_hash(
            PROJECTED_STREAM_ROOT_DOMAIN,
            self._commitment_preimage(),
        )
        if self.diagnostic_commitment != expected_commitment:
            _fail("REJECT_Q1_PROJECTED_STREAM", "diagnostic commitment preimage differs")
        return (
            1,
            PROJECTED_STREAM_SCHEMA_ID,
            self.input_signature_id,
            self.universe_root,
            int(self.stream_kind_id),
            self.descriptor.canonical_object(),
            tuple(manifest.canonical_object() for manifest in self.chunks.manifests),
            self.external_sort_projection.canonical_object(),
            self.diagnostic_commitment,
        )


@dataclass(frozen=True, slots=True)
class Q1CountingDiscardStreamV1:
    """Qualification-only stream projection that retains no framed blobs.

    This object commits the counters and metadata produced by the independent
    bounded-node3 counting sink.  Its comparison with ``Q1ProjectedStreamV1``
    is deliberately explicit; construction of either object alone is not
    evidence that the two encoder sinks agree.
    """

    input_signature_id: int
    universe_root: bytes
    stream_kind_id: ArchiveStreamKindId
    canonical_record_payload_bytes: int
    descriptor: Q1StreamDescriptorV1
    chunk_manifests: tuple[Q1ArchiveChunkManifestV1, ...]
    external_sort_projection: Q1ExternalSortProjectionV1
    diagnostic_commitment: bytes

    def __post_init__(self) -> None:
        if type(self.input_signature_id) is not int or self.input_signature_id not in (
            1,
            2,
        ):
            _fail("REJECT_Q1_COUNTING_DISCARD", "input signature differs")
        expected_universe_root = production_universe_v1(
            self.input_signature_id
        ).universe_root
        if type(self.universe_root) is not bytes or self.universe_root != expected_universe_root:
            _fail("REJECT_Q1_COUNTING_DISCARD", "universe binding differs")
        if not isinstance(self.stream_kind_id, ArchiveStreamKindId):
            _fail("REJECT_Q1_COUNTING_DISCARD", "stream kind differs")
        if (
            type(self.canonical_record_payload_bytes) is not int
            or self.canonical_record_payload_bytes < 1
        ):
            _fail(
                "REJECT_Q1_COUNTING_DISCARD",
                "canonical record payload byte count differs",
            )
        if type(self.descriptor) is not Q1StreamDescriptorV1:
            _fail("REJECT_Q1_COUNTING_DISCARD", "descriptor type differs")
        self.descriptor.canonical_object()
        if type(self.chunk_manifests) is not tuple or not self.chunk_manifests:
            _fail("REJECT_Q1_COUNTING_DISCARD", "chunk manifests differ")
        if any(
            type(manifest) is not Q1ArchiveChunkManifestV1
            for manifest in self.chunk_manifests
        ):
            _fail("REJECT_Q1_COUNTING_DISCARD", "chunk manifest type differs")
        if type(self.external_sort_projection) is not Q1ExternalSortProjectionV1:
            _fail("REJECT_Q1_COUNTING_DISCARD", "external sort type differs")
        self.external_sort_projection.canonical_object()
        if type(self.diagnostic_commitment) is not bytes or len(
            self.diagnostic_commitment
        ) != 32:
            _fail("REJECT_Q1_COUNTING_DISCARD", "diagnostic commitment differs")

        if self.descriptor.stream_kind_id is not self.stream_kind_id:
            _fail("REJECT_Q1_COUNTING_DISCARD", "descriptor kind differs")
        if (
            self.external_sort_projection.input_signature_id
            != self.input_signature_id
            or self.external_sort_projection.stream_kind_id is not self.stream_kind_id
            or self.external_sort_projection.record_count
            != self.descriptor.record_count
        ):
            _fail("REJECT_Q1_COUNTING_DISCARD", "external sort binding differs")
        next_record_index = 0
        for chunk_index, manifest in enumerate(self.chunk_manifests):
            if (
                manifest.input_signature_id != self.input_signature_id
                or manifest.universe_root != self.universe_root
                or manifest.stream_kind_id is not self.stream_kind_id
                or manifest.chunk_index != chunk_index
                or manifest.first_record_index != next_record_index
            ):
                _fail("REJECT_Q1_COUNTING_DISCARD", "chunk binding/order differs")
            next_record_index += manifest.record_count
        manifest_objects = tuple(
            manifest.canonical_object() for manifest in self.chunk_manifests
        )
        if (
            next_record_index != self.descriptor.record_count
            or self.descriptor.framed_stream_bytes
            != sum(manifest.framed_blob_length for manifest in self.chunk_manifests)
            or self.descriptor.chunk_count != len(self.chunk_manifests)
            or self.descriptor.chunk_manifest_subtree_root
            != rfc6962_root(manifest_objects)
        ):
            _fail("REJECT_Q1_COUNTING_DISCARD", "descriptor counter replay differs")
        expected_commitment = content_hash(
            PROJECTED_STREAM_ROOT_DOMAIN,
            (
                1,
                PROJECTED_STREAM_SCHEMA_ID,
                self.input_signature_id,
                self.universe_root,
                int(self.stream_kind_id),
                self.descriptor.canonical_object(),
                manifest_objects,
                self.external_sort_projection.canonical_object(),
            ),
        )
        if self.diagnostic_commitment != expected_commitment:
            _fail(
                "REJECT_Q1_COUNTING_DISCARD",
                "projected-stream commitment replay differs",
            )

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            COUNTING_DISCARD_STREAM_SCHEMA_ID,
            self.input_signature_id,
            self.universe_root,
            int(self.stream_kind_id),
            self.descriptor.record_count,
            self.canonical_record_payload_bytes,
            self.descriptor.framed_stream_bytes,
            self.descriptor.chunk_count,
            self.descriptor.canonical_object(),
            tuple(
                manifest.canonical_object() for manifest in self.chunk_manifests
            ),
            self.external_sort_projection.canonical_object(),
            self.diagnostic_commitment,
            0,  # retained framed blob count
            0,  # retained framed blob bytes
        )


@dataclass(frozen=True, slots=True)
class Q1SnapshotRecordSetV1:
    input_signature_id: int
    universe_root: bytes
    program_records: tuple[Q1RepresentativeProgramRecordV1, ...]
    cohort_records: tuple[Q1ContinuationCohortRecordV1, ...]
    class_records: tuple[Q1QuotientClassRecordV1, ...]
    diagnostic_root: bytes

    def __post_init__(self) -> None:
        if type(self.input_signature_id) is not int or self.input_signature_id not in (1, 2):
            _fail("REJECT_Q1_RECORD_SET", "input signature differs")
        expected_universe_root = production_universe_v1(
            self.input_signature_id
        ).universe_root
        if type(self.universe_root) is not bytes or self.universe_root != expected_universe_root:
            _fail("REJECT_Q1_RECORD_SET", "universe binding differs")
        for name, rows, expected_type, stream_kind in (
            (
                "program_records",
                self.program_records,
                Q1RepresentativeProgramRecordV1,
                ArchiveStreamKindId.PROGRAM,
            ),
            (
                "cohort_records",
                self.cohort_records,
                Q1ContinuationCohortRecordV1,
                ArchiveStreamKindId.COHORT,
            ),
            (
                "class_records",
                self.class_records,
                Q1QuotientClassRecordV1,
                ArchiveStreamKindId.CLASS,
            ),
        ):
            if type(rows) is not tuple or any(type(row) is not expected_type for row in rows):
                _fail("REJECT_Q1_RECORD_SET", f"{name} has wrong exact type")
            canonical_archive_order_v1(rows, stream_kind_id=stream_kind)
            if any(
                row.input_signature_id != self.input_signature_id
                or row.universe_root != self.universe_root
                for row in rows
            ):
                _fail("REJECT_Q1_RECORD_SET", f"{name} binding differs")
        if not self.program_records or not self.cohort_records or not self.class_records:
            _fail("REJECT_Q1_RECORD_SET", "partition record streams must be nonempty")

        for label, rows in (
            ("program record", self.program_records),
            ("cohort record", self.cohort_records),
            ("class record", self.class_records),
        ):
            record_id_preimages: dict[bytes, bytes] = {}
            for row in rows:
                _register_unique_digest_preimage_v1(
                    record_id_preimages,
                    digest=row.record_id,
                    preimage=canonical_cbor_encode(row.canonical_object()),
                    duplicate_code="REJECT_Q1_RECORD_SET",
                    label=label,
                )

        programs_by_id = _unique_semantic_id_rows_v1(
            self.program_records,
            id_attribute="program_id",
            identity_preimage=_program_identity_preimage_v1,
            duplicate_code="REJECT_Q1_RECORD_SET_DUPLICATE_PROGRAM",
            label="program",
        )
        witness_program_ids: list[bytes] = []
        classes_by_id = _unique_semantic_id_rows_v1(
            self.class_records,
            id_attribute="class_id",
            identity_preimage=_class_identity_preimage_v1,
            duplicate_code="REJECT_Q1_RECORD_SET_DUPLICATE_CLASS",
            label="class",
        )
        class_ids = set(classes_by_id)
        signature_preimages: dict[bytes, bytes] = {}
        for record in (*self.program_records, *self.cohort_records):
            _register_digest_preimage_v1(
                signature_preimages,
                digest=record.signature_id,
                preimage=canonical_cbor_encode(
                    construction_signature_object_v1(
                        record.construction_signature
                    )
                ),
                label="construction signature",
            )
        _unique_semantic_id_rows_v1(
            self.cohort_records,
            id_attribute="cohort_id",
            identity_preimage=_cohort_identity_preimage_v1,
            duplicate_code="REJECT_Q1_RECORD_SET_DUPLICATE_COHORT",
            label="cohort",
        )

        environments = production_universe_v1(
            self.input_signature_id
        ).observation_environments()
        ast_preimages: dict[bytes, bytes] = {}
        for program in self.program_records:
            _register_digest_preimage_v1(
                ast_preimages,
                digest=program.canonical_ast_hash,
                preimage=program.canonical_ast_cbor,
                label="strict AST",
            )
            ast = decode_shrink6_canonical_ast(program.canonical_ast_cbor)
            behavior_values = evaluate_canonical_ast_on_environments_v1(
                ast,
                environments,
            )
            behavior = Q1BehaviorBlobV1(
                self.input_signature_id,
                self.universe_root,
                program.construction_signature.output_sort_id,
                tuple(
                    Q1BehaviorCellV1.bottom()
                    if value is _adapter.BOTTOM
                    else Q1BehaviorCellV1.exact(value)
                    for value in behavior_values
                ),
            )
            class_row = classes_by_id.get(program.class_id)
            if class_row is None:
                _fail(
                    "REJECT_Q1_RECORD_SET",
                    "program references an absent quotient class",
                )
            _require_same_id_preimage_v1(
                claimed_id=program.class_id,
                replayed_id=behavior.behavior_id,
                claimed_preimage=class_row.behavior.canonical_bytes,
                replayed_preimage=behavior.canonical_bytes,
                reject_detail=(
                    "program behavior replay differs from its quotient class"
                ),
                collision_detail=(
                    "class ID has different replayed behavior preimages"
                ),
            )
        for cohort in self.cohort_records:
            if cohort.class_id not in class_ids:
                _fail("REJECT_Q1_RECORD_SET", "cohort references absent class")
            for witness in cohort.witnesses:
                program = programs_by_id.get(witness.program_id)
                if program is None:
                    _fail("REJECT_Q1_RECORD_SET", "cohort references absent program")
                if (
                    program.class_id != cohort.class_id
                    or program.canonical_ast_hash != witness.canonical_ast_hash
                ):
                    _fail("REJECT_Q1_RECORD_SET", "cohort/program witness differs")
                _require_same_id_preimage_v1(
                    claimed_id=cohort.signature_id,
                    replayed_id=program.signature_id,
                    claimed_preimage=canonical_cbor_encode(
                        construction_signature_object_v1(
                            cohort.construction_signature
                        )
                    ),
                    replayed_preimage=canonical_cbor_encode(
                        construction_signature_object_v1(
                            program.construction_signature
                        )
                    ),
                    reject_detail="cohort/program signature ID differs",
                    collision_detail=(
                        "signature ID has different construction preimages"
                    ),
                )
                witness_program_ids.append(witness.program_id)
        if (
            len(witness_program_ids) != len(self.program_records)
            or len(set(witness_program_ids)) != len(witness_program_ids)
            or set(witness_program_ids) != set(programs_by_id)
        ):
            _fail("REJECT_Q1_RECORD_SET", "program/cohort witness bijection differs")

        class_subtree_preimages: dict[bytes, bytes] = {}
        for class_row in self.class_records:
            start = class_row.first_cohort_index
            end = start + class_row.cohort_count
            cohorts = self.cohort_records[start:end]
            if len(cohorts) != class_row.cohort_count or any(
                row.class_id != class_row.class_id for row in cohorts
            ):
                _fail("REJECT_Q1_RECORD_SET", "class cohort range differs")
            expected_visibility = tuple(
                not any(
                    other.construction_signature.dominates(
                        cohort.construction_signature
                    )
                    and len(other.witnesses) >= len(cohort.witnesses)
                    for other_index, other in enumerate(cohorts)
                    if other_index != cohort_index
                )
                for cohort_index, cohort in enumerate(cohorts)
            )
            if tuple(row.visible_frontier_cohort for row in cohorts) != (
                expected_visibility
            ):
                _fail(
                    "REJECT_Q1_RECORD_SET",
                    "visible frontier flags differ from global Pareto replay",
                )
            cohort_objects = tuple(row.canonical_object() for row in cohorts)
            visible = tuple(row for row in cohorts if row.visible_frontier_cohort)
            visible_objects = tuple(row.canonical_object() for row in visible)
            cohort_subtree_root = rfc6962_root(cohort_objects)
            visible_subtree_root = rfc6962_root(visible_objects)
            if (
                class_row.class_cohort_subtree_root != cohort_subtree_root
                or class_row.bank_point_count
                != sum(len(row.witnesses) for row in cohorts)
                or class_row.visible_cohort_count != len(visible)
                or class_row.visible_frontier_point_count
                != sum(len(row.witnesses) for row in visible)
                or class_row.visible_frontier_subtree_root
                != visible_subtree_root
                or class_row.minimum_mdl_q32
                != min(row.construction_signature.mdl_length_q32 for row in cohorts)
            ):
                _fail("REJECT_Q1_RECORD_SET", "class aggregate replay differs")
            _register_digest_preimage_v1(
                class_subtree_preimages,
                digest=cohort_subtree_root,
                preimage=canonical_cbor_encode(cohort_objects),
                label="class cohort RFC6962 subtree",
            )
            _register_digest_preimage_v1(
                class_subtree_preimages,
                digest=visible_subtree_root,
                preimage=canonical_cbor_encode(visible_objects),
                label="visible frontier RFC6962 subtree",
            )
        covered_cohort_indices = tuple(
            index
            for class_row in self.class_records
            for index in range(
                class_row.first_cohort_index,
                class_row.first_cohort_index + class_row.cohort_count,
            )
        )
        if covered_cohort_indices != tuple(range(len(self.cohort_records))):
            _fail(
                "REJECT_Q1_RECORD_SET",
                "class ranges do not partition the cohort stream exactly once",
            )

        if type(self.diagnostic_root) is not bytes or len(self.diagnostic_root) != 32:
            _fail("REJECT_Q1_RECORD_SET", "diagnostic root must be 32 bytes")
        expected_root = content_hash(
            SNAPSHOT_RECORD_SET_ROOT_DOMAIN,
            self.canonical_diagnostic_object(),
        )
        if self.diagnostic_root != expected_root:
            _fail("REJECT_Q1_RECORD_SET", "diagnostic root preimage differs")

    def canonical_diagnostic_object(self) -> tuple[object, ...]:
        return (
            1,
            SNAPSHOT_RECORD_SET_SCHEMA_ID,
            self.input_signature_id,
            self.universe_root,
            tuple(record.canonical_object() for record in self.program_records),
            tuple(record.canonical_object() for record in self.cohort_records),
            tuple(record.canonical_object() for record in self.class_records),
        )


def chunk_canonical_records_v1(
    record_objects: Sequence[object],
    record_ids: Sequence[bytes],
    *,
    input_signature_id: int,
    universe_root: bytes,
    stream_kind_id: ArchiveStreamKindId,
) -> Q1ChunkProjectionV1:
    objects = tuple(record_objects)
    ids = tuple(record_ids)
    if len(objects) != len(ids):
        _fail("REJECT_Q1_CHUNK_INPUT", "record object/ID counts differ")
    if any(type(value) is not bytes or len(value) != 32 for value in ids):
        _fail("REJECT_Q1_CHUNK_INPUT", "record IDs must be 32-byte values")
    frames = tuple(frame_canonical_record_v1(record) for record in objects)
    if any(len(frame) > MAX_CHUNK_FRAMED_BYTES for frame in frames):
        _fail("INCONCLUSIVE_RESOURCE_LIMIT", "OUTPUT_BYTES: one record exceeds chunk")
    derived_ids = tuple(
        _canonical_record_identity_and_sort_key_v1(record, stream_kind_id)[0]
        for record in objects
    )
    if ids != derived_ids:
        _fail("REJECT_Q1_CHUNK_INPUT", "record IDs differ from strict replay")
    record_id_preimages: dict[bytes, bytes] = {}
    for record, record_id in zip(objects, ids, strict=True):
        _register_unique_digest_preimage_v1(
            record_id_preimages,
            digest=record_id,
            preimage=canonical_cbor_encode(record),
            duplicate_code="REJECT_Q1_CHUNK_INPUT",
            label="formal record ID",
        )
    manifests: list[Q1ArchiveChunkManifestV1] = []
    blobs: list[bytes] = []
    start = 0
    while start < len(objects):
        end = start
        payload_bytes = 0
        while end < len(objects):
            next_size = len(frames[end])
            if next_size > MAX_CHUNK_FRAMED_BYTES:
                _fail("INCONCLUSIVE_RESOURCE_LIMIT", "OUTPUT_BYTES: one record exceeds chunk")
            if end > start and (
                end - start + 1 > MAX_RECORDS_PER_CHUNK
                or payload_bytes + next_size > MAX_CHUNK_FRAMED_BYTES
            ):
                break
            payload_bytes += next_size
            end += 1
            if end - start == MAX_RECORDS_PER_CHUNK:
                break
        blob = b"".join(frames[start:end])
        subset = objects[start:end]
        manifest = Q1ArchiveChunkManifestV1(
            input_signature_id,
            universe_root,
            stream_kind_id,
            len(manifests),
            start,
            len(subset),
            ids[start],
            ids[end - 1],
            rfc6962_root(subset),
            framed_blob_hash_v1(blob),
            len(blob),
        )
        manifests.append(manifest)
        blobs.append(blob)
        start = end
    return Q1ChunkProjectionV1(tuple(manifests), tuple(blobs))


def _canonical_cbor_bstr_header_v1(length: int) -> bytes:
    if type(length) is not int or length < 0 or length > 0xFFFFFFFF:
        _fail("REJECT_Q1_COUNTING_DISCARD", "byte-string length is outside u32")
    if length <= 23:
        return bytes((0x40 + length,))
    if length <= 0xFF:
        return b"\x58" + length.to_bytes(1, "big")
    if length <= 0xFFFF:
        return b"\x59" + length.to_bytes(2, "big")
    return b"\x5a" + length.to_bytes(4, "big")


def _counting_framed_blob_hash_v1(
    objects: tuple[tuple[object, ...], ...],
) -> tuple[bytes, int]:
    """Hash one chunk as ``framed_blob_hash_v1`` without joining its frames."""

    encoded_lengths = tuple(len(canonical_cbor_encode(item)) for item in objects)
    framed_length = sum(FRAME_LENGTH_BYTES + length for length in encoded_lengths)
    digest = sha256()
    digest.update(FRAMED_BLOB_HASH_DOMAIN.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(b"\x81")  # canonical one-item array containing the blob
    digest.update(_canonical_cbor_bstr_header_v1(framed_length))
    for item, encoded_length in zip(objects, encoded_lengths, strict=True):
        payload = canonical_cbor_encode(item)
        if len(payload) != encoded_length:
            raise AssertionError("canonical Q1 record encoding changed within one pass")
        digest.update(encoded_length.to_bytes(FRAME_LENGTH_BYTES, "big"))
        digest.update(payload)
    return digest.digest(), framed_length


def counting_discard_record_stream_v1(
    records: Iterable[object],
    *,
    input_signature_id: int,
    universe_root: bytes,
    stream_kind_id: ArchiveStreamKindId,
) -> Q1CountingDiscardStreamV1:
    """Run the independent bounded-node3 counting sink.

    The sink re-encodes records while hashing each chunk incrementally.  It
    returns manifests and counters, but never constructs or retains a framed
    archive blob.  Formal Q1 authority remains outside this diagnostic API.
    """

    if type(input_signature_id) is not int or input_signature_id not in (1, 2):
        _fail("REJECT_Q1_COUNTING_DISCARD", "input signature differs")
    if (
        type(universe_root) is not bytes
        or universe_root != production_universe_v1(input_signature_id).universe_root
    ):
        _fail("REJECT_Q1_COUNTING_DISCARD", "universe binding differs")
    material = canonical_archive_order_v1(records, stream_kind_id=stream_kind_id)
    if not material:
        _fail("REJECT_Q1_COUNTING_DISCARD", "record stream must be nonempty")
    if any(
        record.input_signature_id != input_signature_id  # type: ignore[attr-defined]
        or record.universe_root != universe_root  # type: ignore[attr-defined]
        for record in material
    ):
        _fail("REJECT_Q1_COUNTING_DISCARD", "record/stream binding differs")
    objects = tuple(record.canonical_object() for record in material)  # type: ignore[attr-defined]
    ids = tuple(
        _canonical_record_identity_and_sort_key_v1(obj, stream_kind_id)[0]
        for obj in objects
    )
    record_id_preimages: dict[bytes, bytes] = {}
    for record, record_id in zip(objects, ids, strict=True):
        _register_unique_digest_preimage_v1(
            record_id_preimages,
            digest=record_id,
            preimage=canonical_cbor_encode(record),
            duplicate_code="REJECT_Q1_COUNTING_DISCARD",
            label="formal record ID",
        )

    encoded_lengths = tuple(len(canonical_cbor_encode(item)) for item in objects)
    frame_lengths = tuple(FRAME_LENGTH_BYTES + length for length in encoded_lengths)
    if any(length > MAX_CHUNK_FRAMED_BYTES for length in frame_lengths):
        _fail("INCONCLUSIVE_RESOURCE_LIMIT", "OUTPUT_BYTES: one record exceeds chunk")
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(objects):
        end = start
        payload_bytes = 0
        while end < len(objects):
            next_size = frame_lengths[end]
            if end > start and (
                end - start + 1 > MAX_RECORDS_PER_CHUNK
                or payload_bytes + next_size > MAX_CHUNK_FRAMED_BYTES
            ):
                break
            payload_bytes += next_size
            end += 1
            if end - start == MAX_RECORDS_PER_CHUNK:
                break
        ranges.append((start, end))
        start = end

    manifests: list[Q1ArchiveChunkManifestV1] = []
    for chunk_index, (start, end) in enumerate(ranges):
        subset = objects[start:end]
        framed_blob_hash, framed_blob_length = _counting_framed_blob_hash_v1(subset)
        manifests.append(
            Q1ArchiveChunkManifestV1(
                input_signature_id,
                universe_root,
                stream_kind_id,
                chunk_index,
                start,
                end - start,
                ids[start],
                ids[end - 1],
                rfc6962_root(subset),
                framed_blob_hash,
                framed_blob_length,
            )
        )
    manifest_objects = tuple(item.canonical_object() for item in manifests)
    descriptor = Q1StreamDescriptorV1(
        stream_kind_id,
        len(objects),
        archive_root_v1(objects),
        sum(item.framed_blob_length for item in manifests),
        len(manifests),
        rfc6962_root(manifest_objects),
    )
    sort_rows = tuple(
        (_stream_sort_key(record, stream_kind_id), canonical_cbor_encode(obj))
        for record, obj in zip(material, objects, strict=True)
    )
    sort_projection = project_external_sort_v1(
        sort_rows,
        input_signature_id=input_signature_id,
        stream_kind_id=stream_kind_id,
    )
    commitment = content_hash(
        PROJECTED_STREAM_ROOT_DOMAIN,
        (
            1,
            PROJECTED_STREAM_SCHEMA_ID,
            input_signature_id,
            universe_root,
            int(stream_kind_id),
            descriptor.canonical_object(),
            manifest_objects,
            sort_projection.canonical_object(),
        ),
    )
    return Q1CountingDiscardStreamV1(
        input_signature_id,
        universe_root,
        stream_kind_id,
        sum(encoded_lengths),
        descriptor,
        tuple(manifests),
        sort_projection,
        commitment,
    )


def validate_counting_discard_matches_materialized_v1(
    counting: Q1CountingDiscardStreamV1,
    materialized: Q1ProjectedStreamV1,
) -> None:
    if type(counting) is not Q1CountingDiscardStreamV1:
        _fail("REJECT_Q1_DUAL_ENCODER", "counting projection type differs")
    if type(materialized) is not Q1ProjectedStreamV1:
        _fail("REJECT_Q1_DUAL_ENCODER", "materialized projection type differs")
    counting.canonical_object()
    materialized_object = materialized.canonical_diagnostic_object()
    records = tuple(
        record
        for blob in materialized.chunks.framed_blobs
        for record in replay_framed_records_v1(blob)
    )
    if (
        counting.input_signature_id != materialized.input_signature_id
        or counting.universe_root != materialized.universe_root
        or counting.stream_kind_id is not materialized.stream_kind_id
        or counting.descriptor.canonical_object() != materialized_object[5]
        or tuple(item.canonical_object() for item in counting.chunk_manifests)
        != materialized_object[6]
        or counting.external_sort_projection.canonical_object()
        != materialized_object[7]
        or counting.diagnostic_commitment != materialized_object[8]
        or counting.canonical_record_payload_bytes
        != sum(len(canonical_cbor_encode(record)) for record in records)
    ):
        _fail(
            "REJECT_Q1_DUAL_ENCODER",
            "counting/discard and materialized encoder projections differ",
        )


def project_record_stream_v1(
    records: Iterable[object],
    *,
    input_signature_id: int,
    universe_root: bytes,
    stream_kind_id: ArchiveStreamKindId,
) -> Q1ProjectedStreamV1:
    if type(input_signature_id) is not int or input_signature_id not in (1, 2):
        _fail("REJECT_Q1_PROJECTED_STREAM", "input signature differs")
    if (
        type(universe_root) is not bytes
        or universe_root != production_universe_v1(input_signature_id).universe_root
    ):
        _fail("REJECT_Q1_PROJECTED_STREAM", "universe binding differs")
    material = canonical_archive_order_v1(
        records,
        stream_kind_id=stream_kind_id,
    )
    if not material:
        _fail("REJECT_Q1_PROJECTED_STREAM", "formal record stream must be nonempty")
    if any(
        record.input_signature_id != input_signature_id  # type: ignore[attr-defined]
        or record.universe_root != universe_root  # type: ignore[attr-defined]
        for record in material
    ):
        _fail("REJECT_Q1_PROJECTED_STREAM", "record/stream binding differs")
    objects = tuple(record.canonical_object() for record in material)  # type: ignore[attr-defined]
    ids = tuple(_record_id(record) for record in material)
    chunks = chunk_canonical_records_v1(
        objects,
        ids,
        input_signature_id=input_signature_id,
        universe_root=universe_root,
        stream_kind_id=stream_kind_id,
    )
    chunk_objects = tuple(manifest.canonical_object() for manifest in chunks.manifests)
    descriptor = Q1StreamDescriptorV1(
        stream_kind_id,
        len(material),
        archive_root_v1(objects),
        sum(len(blob) for blob in chunks.framed_blobs),
        len(chunks.manifests),
        rfc6962_root(chunk_objects),
    )
    sort_rows = tuple(
        (_stream_sort_key(record, stream_kind_id), canonical_cbor_encode(obj))
        for record, obj in zip(material, objects, strict=True)
    )
    sort_projection = project_external_sort_v1(
        sort_rows,
        input_signature_id=input_signature_id,
        stream_kind_id=stream_kind_id,
    )
    preimage = (
        1,
        PROJECTED_STREAM_SCHEMA_ID,
        input_signature_id,
        universe_root,
        int(stream_kind_id),
        descriptor.canonical_object(),
        chunk_objects,
        sort_projection.canonical_object(),
    )
    commitment = content_hash(PROJECTED_STREAM_ROOT_DOMAIN, preimage)
    return Q1ProjectedStreamV1(
        input_signature_id,
        universe_root,
        stream_kind_id,
        descriptor,
        chunks,
        sort_projection,
        commitment,
    )


def _formal_behavior_from_snapshot(
    snapshot: Q1PartitionSnapshotV1,
    behavior_class: object,
) -> Q1BehaviorBlobV1:
    output_sort_id = OutputSortId(behavior_class.output_sort_id)  # type: ignore[attr-defined]
    cells: list[Q1BehaviorCellV1] = []
    for cell in behavior_class.behavior_cells:  # type: ignore[attr-defined]
        if type(cell) is not Q1BehaviorCellSnapshotV1:
            _fail("REJECT_Q1_SNAPSHOT_CONVERSION", "behavior cell type differs")
        if cell.cell_tag == 0:
            cells.append(Q1BehaviorCellV1.bottom())
            continue
        runtime = cell.runtime_value(int(output_sort_id))
        cells.append(Q1BehaviorCellV1.exact(runtime))
    return Q1BehaviorBlobV1(
        snapshot.input_signature_id,
        snapshot.universe_root,
        output_sort_id,
        tuple(cells),
    )


def records_from_partition_snapshot_v1(
    snapshot: Q1PartitionSnapshotV1,
) -> Q1SnapshotRecordSetV1:
    """Convert one validated immutable bank snapshot to three formal streams."""

    validate_q1_partition_snapshot_v1(snapshot)
    class_material: list[tuple[bytes, object, Q1BehaviorBlobV1]] = []
    for row in snapshot.behavior_classes:
        behavior = _formal_behavior_from_snapshot(snapshot, row)
        class_material.append((behavior.behavior_id, row, behavior))
    class_material.sort(key=lambda item: (item[0], item[2].canonical_bytes))
    seen_behavior_preimages: dict[bytes, bytes] = {}
    for behavior_id, _row, behavior in class_material:
        previous = seen_behavior_preimages.get(behavior_id)
        if previous is not None:
            if previous == behavior.canonical_bytes:
                _fail(
                    "REJECT_Q1_RECORD_SET_DUPLICATE_CLASS",
                    "snapshot conversion repeated a behavior identity",
                )
            _fail(
                "FAIL_SHA256_PREIMAGE_COLLISION",
                "formal behavior ID has different preimages",
            )
        seen_behavior_preimages[behavior_id] = behavior.canonical_bytes

    temporary_programs: list[Q1RepresentativeProgramRecordV1] = []
    for class_id, row, _behavior in class_material:
        for cohort in row.cohorts:  # type: ignore[attr-defined]
            for representative in cohort.representatives:
                temporary_programs.append(
                    Q1RepresentativeProgramRecordV1(
                        snapshot.input_signature_id,
                        snapshot.universe_root,
                        0,
                        class_id,
                        representative.canonical_ast_cbor,
                        representative.canonical_ast_hash,
                        cohort.signature,
                    )
                )
    temporary_programs.sort(key=lambda row: row.sort_key)
    programs = tuple(
        Q1RepresentativeProgramRecordV1(
            row.input_signature_id,
            row.universe_root,
            index,
            row.class_id,
            row.canonical_ast_cbor,
            row.canonical_ast_hash,
            row.construction_signature,
        )
        for index, row in enumerate(temporary_programs)
    )
    if len({row.canonical_ast_cbor for row in programs}) != len(programs):
        _fail("REJECT_Q1_SNAPSHOT_CONVERSION", "one AST occurs in two classes")
    program_by_ast = {row.canonical_ast_cbor: row for row in programs}

    temporary_cohorts: list[Q1ContinuationCohortRecordV1] = []
    for class_id, row, _behavior in class_material:
        for cohort in row.cohorts:  # type: ignore[attr-defined]
            witnesses = tuple(
                Q1CohortWitnessV1(
                    rank,
                    program_by_ast[representative.canonical_ast_cbor].program_id,
                    representative.canonical_ast_hash,
                )
                for rank, representative in enumerate(cohort.representatives)
            )
            temporary_cohorts.append(
                Q1ContinuationCohortRecordV1(
                    snapshot.input_signature_id,
                    snapshot.universe_root,
                    0,
                    class_id,
                    cohort.signature,
                    witnesses,
                    cohort.visible_frontier_member,
                )
            )
    temporary_cohorts.sort(
        key=lambda row: (
            row.class_id,
            row.signature_id,
            canonical_cbor_encode(construction_signature_object_v1(row.construction_signature)),
        )
    )
    cohorts = tuple(
        Q1ContinuationCohortRecordV1(
            row.input_signature_id,
            row.universe_root,
            index,
            row.class_id,
            row.construction_signature,
            row.witnesses,
            row.visible_frontier_cohort,
        )
        for index, row in enumerate(temporary_cohorts)
    )

    classes: list[Q1QuotientClassRecordV1] = []
    for class_index, (class_id, row, behavior) in enumerate(class_material):
        class_cohorts = tuple(item for item in cohorts if item.class_id == class_id)
        first = class_cohorts[0].cohort_index
        cohort_objects = tuple(item.canonical_object() for item in class_cohorts)
        visible = tuple(item for item in class_cohorts if item.visible_frontier_cohort)
        visible_objects = tuple(item.canonical_object() for item in visible)
        classes.append(
            Q1QuotientClassRecordV1(
                snapshot.input_signature_id,
                snapshot.universe_root,
                class_index,
                behavior,
                first,
                len(class_cohorts),
                rfc6962_root(cohort_objects),
                sum(len(item.witnesses) for item in class_cohorts),
                len(visible),
                sum(len(item.witnesses) for item in visible),
                rfc6962_root(visible_objects),
                row.minimum_admitted_mdl_q32,  # type: ignore[attr-defined]
            )
        )
    class_records = tuple(classes)
    diagnostic_object = (
        1,
        SNAPSHOT_RECORD_SET_SCHEMA_ID,
        snapshot.input_signature_id,
        snapshot.universe_root,
        tuple(record.canonical_object() for record in programs),
        tuple(record.canonical_object() for record in cohorts),
        tuple(record.canonical_object() for record in class_records),
    )
    root = content_hash(
        SNAPSHOT_RECORD_SET_ROOT_DOMAIN,
        diagnostic_object,
    )
    return Q1SnapshotRecordSetV1(
        snapshot.input_signature_id,
        snapshot.universe_root,
        programs,
        cohorts,
        class_records,
        root,
    )


__all__ = [
    "COUNTING_DISCARD_STREAM_SCHEMA_ID",
    "PROJECTED_STREAM_SCHEMA_ID",
    "Q1ArchiveProjectionError",
    "Q1ChunkProjectionV1",
    "Q1CountingDiscardStreamV1",
    "Q1ProjectedStreamV1",
    "Q1SnapshotRecordSetV1",
    "chunk_canonical_records_v1",
    "counting_discard_record_stream_v1",
    "project_record_stream_v1",
    "records_from_partition_snapshot_v1",
    "validate_counting_discard_matches_materialized_v1",
]
