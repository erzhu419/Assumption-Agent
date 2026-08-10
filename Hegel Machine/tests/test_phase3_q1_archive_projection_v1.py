from __future__ import annotations

from dataclasses import replace
import importlib
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

capacity = importlib.import_module("hegel_machine.phase3_q1_capacity_preflight_v1")
contract = importlib.import_module("hegel_machine.phase3_q1_formal_archive_contract_v1")
projection = importlib.import_module("hegel_machine.phase3_q1_archive_projection_v1")
semantic = importlib.import_module("hegel_machine.phase3_q1_semantic_coverage_v1")
snapshot_module = importlib.import_module("hegel_machine.phase3_q1_partition_snapshot_v1")
strict_cbor = importlib.import_module("hegel_machine.strict_cbor_v1")


@pytest.fixture(scope="module")
def node3_record_sets():
    limits = capacity.PreflightLimitsV1(maximum_ast_node_count=3)
    snapshots = tuple(
        snapshot_module.build_q1_partition_snapshot_v1(signature_id, limits=limits)
        for signature_id in (1, 2)
    )
    return tuple(
        (snapshot, projection.records_from_partition_snapshot_v1(snapshot))
        for snapshot in snapshots
    )


@pytest.fixture(scope="module")
def odd_node3_coverage(node3_record_sets):
    snapshot, _records = node3_record_sets[0]
    return semantic.build_q1_semantic_coverage_v1(snapshot)


def test_chunk_boundaries_are_exact_and_replayable() -> None:
    odd = importlib.import_module(
        "hegel_machine.phase3_q1_universe_v1"
    ).production_universe_v1(1)
    strict_ast = importlib.import_module("hegel_machine.strict_ast_shrink6_v1")
    quotient = importlib.import_module("hegel_machine.phase3_q1_quotient_contract_v1")
    ast = strict_ast.canonicalize_shrink6_source_ast(("scalar_const", 3))
    signature = quotient.future_signature_from_ast_v1(ast)
    base = contract.Q1RepresentativeProgramRecordV1(
        1,
        odd.universe_root,
        0,
        b"\x33" * 32,
        ast.cbor_bytes,
        ast.digest,
        signature,
    ).canonical_object()
    objects = tuple(base[:5] + (index,) + base[6:] for index in range(4097))
    ids = tuple(
        strict_cbor.content_hash(contract.PROGRAM_RECORD_ID_DOMAIN, value)
        for value in objects
    )
    chunks = projection.chunk_canonical_records_v1(
        objects,
        ids,
        input_signature_id=1,
        universe_root=odd.universe_root,
        stream_kind_id=contract.ArchiveStreamKindId.PROGRAM,
    )
    assert tuple(row.record_count for row in chunks.manifests) == (4096, 1)
    assert tuple(row.first_record_index for row in chunks.manifests) == (0, 4096)
    assert tuple(
        record
        for blob in chunks.framed_blobs
        for record in contract.replay_framed_records_v1(blob)
    ) == objects

    with pytest.raises(projection.Q1ArchiveProjectionError) as error:
        projection.chunk_canonical_records_v1(
            ((b"x" * contract.MAX_CHUNK_FRAMED_BYTES,),),
            (b"\x01" * 32,),
            input_signature_id=1,
            universe_root=odd.universe_root,
            stream_kind_id=contract.ArchiveStreamKindId.PROGRAM,
        )
    assert error.value.code == "INCONCLUSIVE_RESOURCE_LIMIT"


def test_chunk_ids_and_close_policy_are_derived_not_caller_selected(
    node3_record_sets,
) -> None:
    snapshot, records = node3_record_sets[0]
    selected = records.program_records[:3]
    objects = tuple(record.canonical_object() for record in selected)
    ids = tuple(record.record_id for record in selected)
    with pytest.raises(projection.Q1ArchiveProjectionError) as caught:
        projection.chunk_canonical_records_v1(
            objects,
            (ids[0], b"\x77" * 32, ids[2]),
            input_signature_id=1,
            universe_root=snapshot.universe_root,
            stream_kind_id=contract.ArchiveStreamKindId.PROGRAM,
        )
    assert caught.value.code == "REJECT_Q1_CHUNK_INPUT"

    left = projection.chunk_canonical_records_v1(
        objects[:1],
        ids[:1],
        input_signature_id=1,
        universe_root=snapshot.universe_root,
        stream_kind_id=contract.ArchiveStreamKindId.PROGRAM,
    )
    right = projection.chunk_canonical_records_v1(
        objects[1:2],
        ids[1:2],
        input_signature_id=1,
        universe_root=snapshot.universe_root,
        stream_kind_id=contract.ArchiveStreamKindId.PROGRAM,
    )
    second_manifest = replace(
        right.manifests[0],
        chunk_index=1,
        first_record_index=1,
    )
    with pytest.raises(projection.Q1ArchiveProjectionError) as caught:
        projection.Q1ChunkProjectionV1(
            (left.manifests[0], second_manifest),
            (left.framed_blobs[0], right.framed_blobs[0]),
        )
    assert caught.value.code == "REJECT_Q1_CHUNK_PROJECTION"


def test_snapshot_conversion_preserves_complete_bank_bijection(node3_record_sets) -> None:
    expected = ((1, 110, 86, 40), (2, 144, 112, 28))
    for (snapshot, records), counts in zip(node3_record_sets, expected, strict=True):
        assert (
            snapshot.input_signature_id,
            len(records.program_records),
            len(records.cohort_records),
            len(records.class_records),
        ) == counts
        assert len(records.diagnostic_root) == 32
        assert len(records.program_records) == snapshot.continuation_bank_point_count
        assert len(records.cohort_records) == snapshot.signature_cohort_count
        assert len(records.class_records) == snapshot.behavior_class_count
        assert contract.canonical_archive_order_v1(
            records.program_records,
            stream_kind_id=contract.ArchiveStreamKindId.PROGRAM,
        ) == records.program_records
        assert contract.canonical_archive_order_v1(
            records.cohort_records,
            stream_kind_id=contract.ArchiveStreamKindId.COHORT,
        ) == records.cohort_records
        assert contract.canonical_archive_order_v1(
            records.class_records,
            stream_kind_id=contract.ArchiveStreamKindId.CLASS,
        ) == records.class_records
        with pytest.raises(projection.Q1ArchiveProjectionError):
            replace(records, diagnostic_root=b"\x00" * 32)


def test_record_set_replays_behavior_cohort_identity_and_pareto_visibility(
    node3_record_sets,
) -> None:
    _snapshot, records = node3_record_sets[0]
    wrong_class_program = replace(
        records.program_records[0],
        class_id=records.class_records[-1].class_id,
    )
    with pytest.raises(projection.Q1ArchiveProjectionError) as caught:
        replace(
            records,
            program_records=(wrong_class_program,) + records.program_records[1:],
        )
    assert caught.value.code == "REJECT_Q1_RECORD_SET"

    duplicate_program = records.program_records[0]
    inserted_programs = [
        duplicate_program,
        replace(duplicate_program, program_index=1),
    ]
    inserted_programs.extend(
        replace(row, program_index=index + 1)
        for index, row in enumerate(records.program_records[1:], start=1)
    )
    with pytest.raises(projection.Q1ArchiveProjectionError) as caught:
        replace(records, program_records=tuple(inserted_programs))
    assert caught.value.code == "REJECT_Q1_RECORD_SET_DUPLICATE_PROGRAM"

    duplicate_class = records.class_records[0]
    inserted_classes = [
        duplicate_class,
        replace(duplicate_class, class_index=1),
    ]
    inserted_classes.extend(
        replace(row, class_index=index + 1)
        for index, row in enumerate(records.class_records[1:], start=1)
    )
    with pytest.raises(projection.Q1ArchiveProjectionError) as caught:
        replace(records, class_records=tuple(inserted_classes))
    assert caught.value.code == "REJECT_Q1_RECORD_SET_DUPLICATE_CLASS"

    duplicate_source = records.cohort_records[0]
    inserted = [duplicate_source, replace(duplicate_source, cohort_index=1)]
    inserted.extend(
        replace(row, cohort_index=index + 1)
        for index, row in enumerate(records.cohort_records[1:], start=1)
    )
    with pytest.raises(projection.Q1ArchiveProjectionError) as caught:
        replace(records, cohort_records=tuple(inserted))
    assert caught.value.code == "REJECT_Q1_RECORD_SET_DUPLICATE_COHORT"

    first = records.cohort_records[0]
    wrong_visibility = replace(
        first,
        visible_frontier_cohort=not first.visible_frontier_cohort,
    )
    with pytest.raises(projection.Q1ArchiveProjectionError) as caught:
        replace(
            records,
            cohort_records=(wrong_visibility,) + records.cohort_records[1:],
        )
    assert caught.value.code == "REJECT_Q1_RECORD_SET"


def test_same_digest_different_identity_preimage_is_a_collision(
    node3_record_sets,
    monkeypatch,
) -> None:
    snapshot, records = node3_record_sets[0]
    with monkeypatch.context() as collision_patch:
        collision_patch.setattr(
            contract.Q1RepresentativeProgramRecordV1,
            "program_id",
            property(lambda _self: b"\xa5" * 32),
        )
        with pytest.raises(projection.Q1ArchiveProjectionError) as caught:
            replace(records)
    assert caught.value.code == "FAIL_SHA256_PREIMAGE_COLLISION"

    original_content_hash = projection.content_hash

    def collide_record_ids(domain, value):
        if domain == contract.PROGRAM_RECORD_ID_DOMAIN:
            return b"\xb6" * 32
        return original_content_hash(domain, value)

    selected = records.program_records[:2]
    selected_objects = tuple(row.canonical_object() for row in selected)
    canonical_chunks = projection.chunk_canonical_records_v1(
        selected_objects,
        tuple(row.record_id for row in selected),
        input_signature_id=1,
        universe_root=snapshot.universe_root,
        stream_kind_id=contract.ArchiveStreamKindId.PROGRAM,
    )
    with monkeypatch.context() as collision_patch:
        collision_patch.setattr(projection, "content_hash", collide_record_ids)
        with pytest.raises(projection.Q1ArchiveProjectionError) as caught:
            projection.chunk_canonical_records_v1(
                selected_objects,
                (b"\xb6" * 32, b"\xb6" * 32),
                input_signature_id=1,
                universe_root=snapshot.universe_root,
                stream_kind_id=contract.ArchiveStreamKindId.PROGRAM,
            )
    assert caught.value.code == "FAIL_SHA256_PREIMAGE_COLLISION"

    collided_manifest = replace(
        canonical_chunks.manifests[0],
        first_record_id=b"\xb6" * 32,
        last_record_id=b"\xb6" * 32,
    )
    with monkeypatch.context() as collision_patch:
        collision_patch.setattr(projection, "content_hash", collide_record_ids)
        with pytest.raises(projection.Q1ArchiveProjectionError) as caught:
            projection.Q1ChunkProjectionV1(
                (collided_manifest,),
                canonical_chunks.framed_blobs,
            )
    assert caught.value.code == "FAIL_SHA256_PREIMAGE_COLLISION"

    with pytest.raises(projection.Q1ArchiveProjectionError) as caught:
        projection._require_same_id_preimage_v1(
            claimed_id=b"\x11" * 32,
            replayed_id=b"\x11" * 32,
            claimed_preimage=b"first-preimage",
            replayed_preimage=b"second-preimage",
            reject_detail="unreachable ID mismatch",
            collision_detail="forced cross-object preimage collision",
        )
    assert caught.value.code == "FAIL_SHA256_PREIMAGE_COLLISION"

    class_left, class_right = records.class_records[:2]
    assert class_left.behavior.canonical_bytes != class_right.behavior.canonical_bytes
    with pytest.raises(projection.Q1ArchiveProjectionError) as caught:
        projection._require_same_id_preimage_v1(
            claimed_id=class_left.class_id,
            replayed_id=class_left.class_id,
            claimed_preimage=class_left.behavior.canonical_bytes,
            replayed_preimage=class_right.behavior.canonical_bytes,
            reject_detail="unreachable behavior ID mismatch",
            collision_detail="forced nested behavior preimage collision",
        )
    assert caught.value.code == "FAIL_SHA256_PREIMAGE_COLLISION"

    cohort_left = records.cohort_records[0]
    cohort_right = next(
        row
        for row in records.cohort_records[1:]
        if row.construction_signature != cohort_left.construction_signature
    )
    left_signature = projection.canonical_cbor_encode(
        contract.construction_signature_object_v1(
            cohort_left.construction_signature
        )
    )
    right_signature = projection.canonical_cbor_encode(
        contract.construction_signature_object_v1(
            cohort_right.construction_signature
        )
    )
    seen_signature_preimages = {cohort_left.signature_id: left_signature}
    with pytest.raises(projection.Q1ArchiveProjectionError) as caught:
        projection._register_digest_preimage_v1(
            seen_signature_preimages,
            digest=cohort_left.signature_id,
            preimage=right_signature,
            label="construction signature",
        )
    assert caught.value.code == "FAIL_SHA256_PREIMAGE_COLLISION"

    seen_ast_preimages = {b"\x22" * 32: b"first-ast"}
    with pytest.raises(projection.Q1ArchiveProjectionError) as caught:
        projection._register_digest_preimage_v1(
            seen_ast_preimages,
            digest=b"\x22" * 32,
            preimage=b"second-ast",
            label="strict AST",
        )
    assert caught.value.code == "FAIL_SHA256_PREIMAGE_COLLISION"


def test_three_snapshot_streams_replay_all_roots(node3_record_sets) -> None:
    for snapshot, records in node3_record_sets:
        for kind, rows in (
            (contract.ArchiveStreamKindId.PROGRAM, records.program_records),
            (contract.ArchiveStreamKindId.COHORT, records.cohort_records),
            (contract.ArchiveStreamKindId.CLASS, records.class_records),
        ):
            sort_keys = tuple(projection._stream_sort_key(row, kind) for row in rows)
            assert sort_keys == tuple(sorted(sort_keys))
            assert len(set(sort_keys)) == len(sort_keys)
            stream = projection.project_record_stream_v1(
                rows,
                input_signature_id=snapshot.input_signature_id,
                universe_root=snapshot.universe_root,
                stream_kind_id=kind,
            )
            objects = tuple(row.canonical_object() for row in rows)
            assert stream.descriptor.record_count == len(rows)
            assert stream.descriptor.archive_root == strict_cbor.rfc6962_root(objects)
            assert stream.external_sort_projection.record_count == len(rows)
            assert stream.external_sort_projection.charged_scratch_high_water_bytes > 0
            assert stream.canonical_diagnostic_object()[-1] == stream.diagnostic_commitment
            with pytest.raises(projection.Q1ArchiveProjectionError):
                replace(stream, diagnostic_commitment=b"\x00" * 32)
            with pytest.raises(projection.Q1ArchiveProjectionError):
                replace(
                    stream,
                    external_sort_projection=replace(
                        stream.external_sort_projection,
                        charged_scratch_high_water_bytes=(
                            stream.external_sort_projection.charged_scratch_high_water_bytes
                            + 1
                        ),
                    ),
                )
            first_blob = stream.chunks.framed_blobs[0]
            with pytest.raises(projection.Q1ArchiveProjectionError):
                replace(
                    stream.chunks,
                    framed_blobs=(first_blob[:-1] + bytes([first_blob[-1] ^ 1]),)
                    + stream.chunks.framed_blobs[1:],
                )
            with pytest.raises(projection.Q1ArchiveProjectionError):
                replace(
                    stream.chunks,
                    manifests=(
                        replace(
                            stream.chunks.manifests[0],
                            first_record_id=b"\x00" * 32,
                        ),
                    )
                    + stream.chunks.manifests[1:],
                )


def test_coverage_stream_projects_all_846_rows(
    node3_record_sets,
    odd_node3_coverage,
) -> None:
    snapshot, _records = node3_record_sets[0]
    stream = projection.project_record_stream_v1(
        odd_node3_coverage.coverage_records,
        input_signature_id=1,
        universe_root=snapshot.universe_root,
        stream_kind_id=contract.ArchiveStreamKindId.COVERAGE,
    )
    assert stream.descriptor.record_count == 846
    assert stream.external_sort_projection.record_count == 846
    assert stream.descriptor.chunk_count >= 1
    assert sum(
        manifest.record_count for manifest in stream.chunks.manifests
    ) == 846


@pytest.mark.parametrize("stream_kind", tuple(contract.ArchiveStreamKindId))
def test_direct_empty_projected_stream_is_rejected(stream_kind) -> None:
    odd = importlib.import_module(
        "hegel_machine.phase3_q1_universe_v1"
    ).production_universe_v1(1)
    descriptor = contract.Q1StreamDescriptorV1(
        stream_kind,
        0,
        strict_cbor.rfc6962_root(()),
        0,
        0,
        strict_cbor.rfc6962_root(()),
    )
    chunks = projection.Q1ChunkProjectionV1((), ())
    sort_projection = importlib.import_module(
        "hegel_machine.phase3_q1_external_sort_profile_v1"
    ).project_external_sort_v1(
        (),
        input_signature_id=1,
        stream_kind_id=stream_kind,
    )
    preimage = (
        1,
        projection.PROJECTED_STREAM_SCHEMA_ID,
        1,
        odd.universe_root,
        int(stream_kind),
        descriptor.canonical_object(),
        (),
        sort_projection.canonical_object(),
    )
    commitment = strict_cbor.content_hash(
        projection.PROJECTED_STREAM_ROOT_DOMAIN,
        preimage,
    )
    with pytest.raises(projection.Q1ArchiveProjectionError) as caught:
        projection.Q1ProjectedStreamV1(
            1,
            odd.universe_root,
            stream_kind,
            descriptor,
            chunks,
            sort_projection,
            commitment,
        )
    assert caught.value.code == "REJECT_Q1_PROJECTED_STREAM"


def test_noncanonical_stream_order_is_rejected(node3_record_sets) -> None:
    snapshot, records = node3_record_sets[0]
    with pytest.raises(contract.Q1ArchiveContractError) as error:
        projection.project_record_stream_v1(
            tuple(reversed(records.program_records)),
            input_signature_id=snapshot.input_signature_id,
            universe_root=snapshot.universe_root,
            stream_kind_id=contract.ArchiveStreamKindId.PROGRAM,
        )
    assert error.value.code == "REJECT_Q1_ARCHIVE_ORDER"


def test_stream_binding_and_bool_aliases_fail_closed(node3_record_sets) -> None:
    odd_snapshot, odd_records = node3_record_sets[0]
    _sink_snapshot, sink_records = node3_record_sets[1]
    with pytest.raises(projection.Q1ArchiveProjectionError) as caught:
        projection.project_record_stream_v1(
            sink_records.program_records,
            input_signature_id=odd_snapshot.input_signature_id,
            universe_root=odd_snapshot.universe_root,
            stream_kind_id=contract.ArchiveStreamKindId.PROGRAM,
        )
    assert caught.value.code == "REJECT_Q1_PROJECTED_STREAM"

    program_object = odd_records.program_records[0].canonical_object()
    aliased_program = (True,) + program_object[1:]
    with pytest.raises(projection.Q1ArchiveProjectionError) as caught:
        projection._canonical_record_identity_and_sort_key_v1(
            aliased_program,
            contract.ArchiveStreamKindId.PROGRAM,
        )
    assert caught.value.code == "REJECT_Q1_PROJECTED_RECORD"

    class_object = odd_records.class_records[0].canonical_object()
    behavior = class_object[6]
    cells = behavior[7]
    aliased_cells = ((False,),) + cells[1:]
    aliased_behavior = behavior[:7] + (aliased_cells,)
    aliased_class = class_object[:6] + (aliased_behavior,) + class_object[7:]
    with pytest.raises(projection.Q1ArchiveProjectionError) as caught:
        projection._canonical_record_identity_and_sort_key_v1(
            aliased_class,
            contract.ArchiveStreamKindId.CLASS,
        )
    assert caught.value.code == "REJECT_Q1_PROJECTED_RECORD"


def test_projection_source_has_no_target_truth_split_or_role_import() -> None:
    source = Path(projection.__file__).read_text(encoding="utf-8")
    forbidden = (
        "phase3_dsl_v1",
        "phase3_m25_rows_v1",
        "phase3_q0_quotient_contract_v1",
        "BOOL_BIT_EXACT_PREDICATE_MATCH_V1",
    )
    assert all(token not in source for token in forbidden)
