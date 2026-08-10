from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import importlib
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

contract = importlib.import_module(
    "hegel_machine.phase3_q1_formal_archive_contract_v1"
)
quotient = importlib.import_module("hegel_machine.phase3_q1_quotient_contract_v1")
strict = importlib.import_module("hegel_machine.strict_ast_shrink6_v1")
cbor = importlib.import_module("hegel_machine.strict_cbor_v1")
universe = importlib.import_module("hegel_machine.phase3_q1_universe_v1")


def _zero_behavior(input_signature_id: int) -> object:
    rows = universe.production_universe_v1(input_signature_id).rows
    return contract.Q1BehaviorBlobV1(
        input_signature_id,
        universe.production_universe_v1(input_signature_id).universe_root,
        quotient.OutputSortId.RATIONAL_VALUE,
        tuple(contract.Q1BehaviorCellV1.exact(Fraction(0)) for _ in rows),
    )


def _program_and_cohort(input_signature_id: int = 1):
    behavior = _zero_behavior(input_signature_id)
    ast = strict.canonicalize_shrink6_source_ast(("scalar_const", 3))
    signature = quotient.future_signature_from_ast_v1(ast)
    program = contract.Q1RepresentativeProgramRecordV1(
        input_signature_id,
        behavior.universe_root,
        0,
        behavior.behavior_id,
        ast.cbor_bytes,
        ast.digest,
        signature,
    )
    cohort = contract.Q1ContinuationCohortRecordV1(
        input_signature_id,
        behavior.universe_root,
        0,
        behavior.behavior_id,
        signature,
        (contract.Q1CohortWitnessV1(0, program.program_id, ast.digest),),
        True,
    )
    return behavior, program, cohort


def test_tag_registry_is_unique_contiguous_and_unoccupied_by_existing_python() -> None:
    tags = [row[0] for row in contract.Q1_TAG_REGISTRY]
    assert tags == list(range(0x3700, 0x370D))
    assert len({row[1] for row in contract.Q1_TAG_REGISTRY}) == 13
    for path in (ROOT / "src/hegel_machine").glob("*.py"):
        if path.name == "phase3_q1_formal_archive_contract_v1.py":
            continue
        source = path.read_text(encoding="utf-8")
        assert not any(f"0x{tag:04X}" in source for tag in tags)


def test_behavior_blob_is_sort_sensitive_and_bound_to_exact_universe() -> None:
    odd = universe.production_universe_v1(1)
    bool_blob = contract.Q1BehaviorBlobV1(
        1,
        odd.universe_root,
        quotient.OutputSortId.BOOL,
        tuple(contract.Q1BehaviorCellV1.exact(False) for _ in odd.rows),
    )
    bit_blob = contract.Q1BehaviorBlobV1(
        1,
        odd.universe_root,
        quotient.OutputSortId.BIT,
        tuple(contract.Q1BehaviorCellV1.exact(0) for _ in odd.rows),
    )
    assert bool_blob.behavior_id != bit_blob.behavior_id
    assert cbor.canonical_cbor_decode(bool_blob.canonical_bytes) == (
        bool_blob.canonical_object()
    )
    with pytest.raises(contract.Q1ArchiveContractError) as error:
        contract.Q1BehaviorBlobV1(
            True,
            odd.universe_root,
            quotient.OutputSortId.BOOL,
            tuple(contract.Q1BehaviorCellV1.exact(False) for _ in odd.rows),
        )
    assert error.value.code == "REJECT_Q1_INPUT_SIGNATURE"
    with pytest.raises(contract.Q1ArchiveContractError):
        contract.Q1BehaviorBlobV1(
            1,
            b"\x00" * 32,
            quotient.OutputSortId.BOOL,
            tuple(contract.Q1BehaviorCellV1.exact(False) for _ in odd.rows),
        )


def test_program_cohort_and_class_identities_replay_strict_ast() -> None:
    behavior, program, cohort = _program_and_cohort()
    assert program.signature_id == contract.construction_signature_id_v1(
        program.construction_signature
    )
    assert cbor.canonical_cbor_decode(
        cbor.canonical_cbor_encode(program.canonical_object())
    ) == program.canonical_object()
    cohort_root = cbor.rfc6962_root([cohort.canonical_object()])
    class_record = contract.Q1QuotientClassRecordV1(
        1,
        behavior.universe_root,
        0,
        behavior,
        0,
        1,
        cohort_root,
        1,
        1,
        1,
        cohort_root,
        program.construction_signature.mdl_length_q32,
    )
    assert class_record.class_id == behavior.behavior_id
    assert len({program.program_id, program.record_id, cohort.record_id, class_record.record_id}) == 4
    with pytest.raises(contract.Q1ArchiveContractError) as error:
        replace(program, canonical_ast_hash=b"\x00" * 32)
    assert error.value.code == "REJECT_Q1_PROGRAM_AST_HASH"
    with pytest.raises(contract.Q1ArchiveContractError):
        replace(
            cohort,
            witnesses=(
                contract.Q1CohortWitnessV1(1, program.program_id, program.canonical_ast_hash),
            ),
        )


def test_coverage_registry_has_846_rows_and_leaf_counts_are_exact() -> None:
    registry = contract.expected_coverage_registry_v1()
    assert len(registry) == contract.EXPECTED_COVERAGE_RECORD_COUNT == 846
    assert registry[:3] == ((0, 0), (0, 1), (0, 2))
    assert registry[809] == (0, 809)
    assert registry[810] == (1, 0x1000)
    assert registry[-1] == (3, 0x4002)
    odd = universe.production_universe_v1(1)
    application_key = contract.semantic_application_key_v1(
        1, odd.universe_root, 0, 0, (), ()
    )
    application_root = cbor.rfc6962_root([application_key])
    admission_root = cbor.rfc6962_root(
        [(contract.semantic_application_id_v1(application_key), b"\x11" * 32)]
    )
    row = contract.Q1SemanticCoverageRecordV1(
        1,
        odd.universe_root,
        0,
        0,
        1,
        application_root,
        1,
        application_root,
        1,
        admission_root,
        1,
        0,
    )
    assert row.canonical_object()[7:11] == (
        1,
        application_root,
        1,
        application_root,
    )
    with pytest.raises(contract.Q1ArchiveContractError):
        replace(row, processed_application_count=0)
    with pytest.raises(contract.Q1ArchiveContractError):
        replace(row, strict_admitted_count=0, unique_canonical_ast_count=0)
    coverage_root = cbor.rfc6962_root(registry)
    profile = contract.projection_profile_object_v1(
        semantic_binding_root=b"\x22" * 32,
        coverage_registry_root=coverage_root,
        resource_guard_registry=contract.Q1_RESOURCE_GUARD_REGISTRY,
    )
    assert profile[9] == coverage_root
    with pytest.raises(contract.Q1ArchiveContractError):
        contract.projection_profile_object_v1(
            semantic_binding_root=b"\x22" * 32,
            coverage_registry_root=b"\x00" * 32,
            resource_guard_registry=contract.Q1_RESOURCE_GUARD_REGISTRY,
        )
    with pytest.raises(contract.Q1ArchiveContractError):
        contract.projection_profile_object_v1(
            semantic_binding_root=b"\x22" * 32,
            coverage_registry_root=coverage_root,
            resource_guard_registry=((True, b"RAW_OPERATOR_APPLICATIONS"),),
        )


def test_framing_is_canonical_length_prefixed_and_replayable() -> None:
    records = ((1, b"a"), (2, b"bb"), (3, (False, None)))
    framed = b"".join(contract.frame_canonical_record_v1(row) for row in records)
    assert contract.replay_framed_records_v1(framed) == records
    assert contract.framed_blob_hash_v1(framed) == contract.framed_blob_hash_v1(framed)
    with pytest.raises(contract.Q1ArchiveContractError) as error:
        contract.replay_framed_records_v1(framed[:-1])
    assert error.value.code == "REJECT_Q1_FRAME"


def test_stream_manifest_state_bundle_and_projection_stay_fail_closed() -> None:
    odd = universe.production_universe_v1(1)
    sink = universe.production_universe_v1(2)
    empty = cbor.rfc6962_root([])
    descriptors = tuple(
        contract.Q1StreamDescriptorV1(kind, 0, empty, 0, 0, empty)
        for kind in contract.ArchiveStreamKindId
    )
    manifest = contract.Q1SignatureArchiveManifestV1(
        1,
        odd.universe_root,
        len(odd.rows),
        b"\x01" * 32,
        b"\x02" * 32,
        descriptors,
        b"\x03" * 32,
        b"\x04" * 32,
        0,
        empty,
    )
    bundle = contract.Q1ClosureBundleV1(
        b"\x01" * 32,
        b"\x02" * 32,
        (
            (1, odd.universe_root, manifest.manifest_root, b"\x04" * 32),
            (2, sink.universe_root, b"\x05" * 32, b"\x06" * 32),
        ),
    )
    assert len(bundle.bundle_root) == 32

    def partition_row(input_signature_id: int, universe_root: bytes):
        stream_roots = tuple(bytes([0x70 + index]) * 32 for index in range(4))
        sort_roots = tuple(bytes([0x80 + index]) * 32 for index in range(4))
        stream_commitment = contract.partition_stream_commitment_v1(
            input_signature_id=input_signature_id,
            universe_root=universe_root,
            raw_application_count=846,
            behavior_class_count=2,
            cohort_count=3,
            bank_point_count=4,
            frontier_point_count=2,
            maximum_bank_points_per_class=2,
            maximum_frontier_points_per_class=1,
            program_record_count=4,
            coverage_record_count=846,
            projected_record_stream_bytes=100,
            projected_chunk_manifest_stream_bytes=20,
            stream_diagnostic_commitments=stream_roots,
        )
        sort_root = contract.partition_external_sort_root_v1(
            input_signature_id=input_signature_id,
            universe_root=universe_root,
            external_sort_stream_roots=sort_roots,
            projected_peak_scratch_bytes=200,
        )
        return contract.Q1ProjectionPartitionRowV1(
            input_signature_id=input_signature_id,
            universe_root=universe_root,
            raw_application_count=846,
            behavior_class_count=2,
            cohort_count=3,
            bank_point_count=4,
            frontier_point_count=2,
            maximum_bank_points_per_class=2,
            maximum_frontier_points_per_class=1,
            peak_work_queue_points=10,
            program_record_count=4,
            coverage_record_count=846,
            projected_record_stream_bytes=100,
            projected_chunk_manifest_stream_bytes=20,
            projected_fixed_point_frame_bytes=10,
            projected_signature_manifest_frame_bytes=10,
            projected_partition_payload_bytes=140,
            projected_peak_scratch_bytes=200,
            stream_diagnostic_commitments=stream_roots,
            diagnostic_stream_commitment=stream_commitment,
            external_sort_stream_roots=sort_roots,
            external_sort_projection_root=sort_root,
        )

    semantic_binding_root = b"\x01" * 32
    projection_profile_root = contract.projection_profile_root_v1(
        semantic_binding_root=semantic_binding_root,
        coverage_registry_root=cbor.rfc6962_root(
            contract.expected_coverage_registry_v1()
        ),
        resource_guard_registry=contract.Q1_RESOURCE_GUARD_REGISTRY,
    )
    result = contract.Q1ArchiveProjectionResultV1(
        projection_profile_root,
        semantic_binding_root,
        (
            partition_row(1, odd.universe_root),
            partition_row(2, sink.universe_root),
        ),
        20,
        300,
        300 + contract.ENDPOINT_RUN_METADATA_RESERVATION_BYTES,
        200,
        contract.HOST_RUN_METADATA_RESERVATION_BYTES,
        0,
    )
    wire = result.canonical_object()
    assert wire[12:21] == (
        0,
        0,
        0,
        None,
        (None,) * 8,
        0,
        False,
        None,
        False,
    )
    assert len(result.diagnostic_root) == 32
    with pytest.raises(contract.Q1ArchiveContractError):
        replace(result, projection_profile_root=b"\x02" * 32)
    with pytest.raises(contract.Q1ArchiveContractError):
        replace(result, projected_endpoint_total_output_bytes=301)
    with pytest.raises(contract.Q1ArchiveContractError):
        replace(result, projected_closure_bundle_frame_bytes=21)
    with pytest.raises(contract.Q1ArchiveContractError):
        replace(result, projected_endpoint_peak_scratch_bytes=199)
    with pytest.raises(contract.Q1ArchiveContractError):
        replace(
            result.partition_rows[0],
            diagnostic_stream_commitment=b"\x00" * 32,
        )
    with pytest.raises(contract.Q1ArchiveContractError):
        replace(
            result.partition_rows[0],
            external_sort_projection_root=b"\x00" * 32,
        )


def test_formal_archive_contract_has_no_target_truth_split_or_role_import() -> None:
    source = Path(contract.__file__).read_text(encoding="utf-8")
    forbidden = (
        "phase3_dsl_v1",
        "phase3_m25_rows_v1",
        "phase3_q0_quotient_contract_v1",
        "BOOL_BIT_EXACT_PREDICATE_MATCH_V1",
    )
    assert all(token not in source for token in forbidden)
