from __future__ import annotations

import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

adapter = importlib.import_module("hegel_machine.phase3_q0_input_adapter_v1")
cbor = importlib.import_module("hegel_machine.strict_cbor_v1")
contract = importlib.import_module(
    "hegel_machine.phase3_q1_formal_archive_contract_v1"
)
evaluator = importlib.import_module("hegel_machine.phase3_q0_evaluator_v1")
host = importlib.import_module("hegel_machine.phase3_q1_formal_host_replay_v1")
projection = importlib.import_module("hegel_machine.phase3_q1_archive_projection_v1")
quotient = importlib.import_module("hegel_machine.phase3_q1_quotient_contract_v1")
strict_ast = importlib.import_module("hegel_machine.strict_ast_shrink6_v1")
universe = importlib.import_module("hegel_machine.phase3_q1_universe_v1")


CAPSULE_DIGEST = "sha256:" + "a1" * 32
SOURCE_COMMIT = "b2" * 20
RUN_GENESIS_ROOT = "sha256:" + "c3" * 32
PRESTART_RECEIPT_ROOT = "sha256:" + "d4" * 32
EXPECTED_HOST_MODULE_SOURCE_PATHS = (
    "src/hegel_machine/phase3_m3_bounded_enumerator_shrink2_v1.py",
    "src/hegel_machine/phase3_m3_bounded_enumerator_shrink3_v1.py",
    "src/hegel_machine/phase3_m3_bounded_enumerator_shrink4_v1.py",
    "src/hegel_machine/phase3_m3_bounded_enumerator_shrink5_v1.py",
    "src/hegel_machine/phase3_m3_bounded_enumerator_shrink6_v1.py",
    "src/hegel_machine/phase3_m3_bounded_enumerator_v1.py",
    "src/hegel_machine/phase3_m3_dsl_core_v1.py",
    "src/hegel_machine/phase3_m3_record_wire_v1.py",
    "src/hegel_machine/phase3_m3_shrink1_core_v1.py",
    "src/hegel_machine/phase3_m3_shrink2_core_v1.py",
    "src/hegel_machine/phase3_m3_shrink3_core_v1.py",
    "src/hegel_machine/phase3_m3_shrink4_core_v1.py",
    "src/hegel_machine/phase3_m3_shrink5_core_v1.py",
    "src/hegel_machine/phase3_m3_shrink6_core_v1.py",
    "src/hegel_machine/phase3_q0_evaluator_v1.py",
    "src/hegel_machine/phase3_q0_input_adapter_v1.py",
    "src/hegel_machine/phase3_q1_archive_projection_v1.py",
    "src/hegel_machine/phase3_q1_capacity_preflight_v1.py",
    "src/hegel_machine/phase3_q1_external_sort_profile_v1.py",
    "src/hegel_machine/phase3_q1_formal_archive_contract_v1.py",
    "src/hegel_machine/phase3_q1_formal_host_replay_v1.py",
    "src/hegel_machine/phase3_q1_partition_snapshot_v1.py",
    "src/hegel_machine/phase3_q1_quotient_contract_v1.py",
    "src/hegel_machine/phase3_q1_universe_v1.py",
    "src/hegel_machine/strict_ast_shrink1_v1.py",
    "src/hegel_machine/strict_ast_shrink2_v1.py",
    "src/hegel_machine/strict_ast_shrink3_v1.py",
    "src/hegel_machine/strict_ast_shrink4_v1.py",
    "src/hegel_machine/strict_ast_shrink5_v1.py",
    "src/hegel_machine/strict_ast_shrink6_v1.py",
    "src/hegel_machine/strict_ast_v1.py",
    "src/hegel_machine/strict_cbor_v1.py",
)
FORMAL_CAPACITY_LIMITS = [
    {
        "input_signature_id": 1,
        "maximum_ast_depth": 3,
        "maximum_ast_node_count": 6,
        "maximum_raw_operator_applications": 32_768,
        "maximum_behavior_classes": 1_024,
        "maximum_visible_frontier_points": 4_096,
        "maximum_visible_frontier_points_per_class": 32,
        "maximum_continuation_bank_points": 8_192,
        "maximum_continuation_bank_points_per_class": 4_096,
        "maximum_work_queue_points": 16_384,
        "maximum_saturation_rounds": 7,
        "maximum_wall_time_seconds": 3_600,
    },
    {
        "input_signature_id": 2,
        "maximum_ast_depth": 3,
        "maximum_ast_node_count": 6,
        "maximum_raw_operator_applications": 32_768,
        "maximum_behavior_classes": 256,
        "maximum_visible_frontier_points": 1_024,
        "maximum_visible_frontier_points_per_class": 256,
        "maximum_continuation_bank_points": 8_192,
        "maximum_continuation_bank_points_per_class": 4_096,
        "maximum_work_queue_points": 16_384,
        "maximum_saturation_rounds": 7,
        "maximum_wall_time_seconds": 3_600,
    },
]


def _root(value: bytes) -> str:
    return "sha256:" + value.hex()


def _one_record_set(input_signature_id: int):
    frozen_universe = universe.production_universe_v1(input_signature_id)
    ast = strict_ast.canonicalize_shrink6_source_ast(("scalar_const", 3))
    signature = quotient.future_signature_from_ast_v1(ast)
    environments = frozen_universe.observation_environments()
    constant_value = evaluator.evaluate_canonical_ast_on_environments_v1(
        ast,
        environments[:1],
    )[0]
    values = (constant_value,) * len(environments)
    behavior = contract.Q1BehaviorBlobV1(
        input_signature_id,
        frozen_universe.universe_root,
        signature.output_sort_id,
        tuple(
            contract.Q1BehaviorCellV1.bottom()
            if value is adapter.BOTTOM
            else contract.Q1BehaviorCellV1.exact(value)
            for value in values
        ),
    )
    program = contract.Q1RepresentativeProgramRecordV1(
        input_signature_id,
        frozen_universe.universe_root,
        0,
        behavior.behavior_id,
        ast.cbor_bytes,
        ast.digest,
        signature,
    )
    cohort = contract.Q1ContinuationCohortRecordV1(
        input_signature_id,
        frozen_universe.universe_root,
        0,
        behavior.behavior_id,
        signature,
        (contract.Q1CohortWitnessV1(0, program.program_id, ast.digest),),
        True,
    )
    cohort_object = cohort.canonical_object()
    class_record = contract.Q1QuotientClassRecordV1(
        input_signature_id,
        frozen_universe.universe_root,
        0,
        behavior,
        0,
        1,
        cbor.rfc6962_root((cohort_object,)),
        1,
        1,
        1,
        cbor.rfc6962_root((cohort_object,)),
        signature.mdl_length_q32,
    )
    empty = cbor.rfc6962_root(())
    coverage_records = []
    for depth, code in contract.expected_coverage_registry_v1():
        if depth == 0:
            one = cbor.rfc6962_root(((depth, code),))
            coverage_records.append(
                contract.Q1SemanticCoverageRecordV1(
                    input_signature_id,
                    frozen_universe.universe_root,
                    depth,
                    code,
                    1,
                    one,
                    1,
                    one,
                    1,
                    one,
                    1,
                    0,
                )
            )
        else:
            coverage_records.append(
                contract.Q1SemanticCoverageRecordV1(
                    input_signature_id,
                    frozen_universe.universe_root,
                    depth,
                    code,
                    0,
                    empty,
                    0,
                    empty,
                    0,
                    empty,
                    0,
                    0,
                )
            )
    return (
        frozen_universe,
        (program,),
        (cohort,),
        (class_record,),
        tuple(coverage_records),
    )


def _formal_tree_bytes(semantic_byte: int):
    semantic_binding_root = bytes((semantic_byte,)) * 32
    projection_profile_root = contract.projection_profile_root_v1(
        semantic_binding_root=semantic_binding_root,
        coverage_registry_root=cbor.rfc6962_root(
            contract.expected_coverage_registry_v1()
        ),
        resource_guard_registry=contract.Q1_RESOURCE_GUARD_REGISTRY,
    )
    files: dict[str, bytes] = {}
    signatures = []
    endpoint_partitions = []
    for input_signature_id in (1, 2):
        frozen_universe, program, cohort, classes, coverage = _one_record_set(
            input_signature_id
        )
        rows = (program, cohort, classes, coverage)
        stream_items = []
        for stream_kind, material in zip(
            contract.ArchiveStreamKindId,
            rows,
            strict=True,
        ):
            objects = tuple(row.canonical_object() for row in material)
            record_ids = tuple(row.record_id for row in material)
            blob = b"".join(
                contract.frame_canonical_record_v1(value) for value in objects
            )
            manifest = contract.Q1ArchiveChunkManifestV1(
                input_signature_id,
                frozen_universe.universe_root,
                stream_kind,
                0,
                0,
                len(objects),
                record_ids[0],
                record_ids[-1],
                cbor.rfc6962_root(objects),
                contract.framed_blob_hash_v1(blob),
                len(blob),
            )
            chunks = SimpleNamespace(
                manifests=(manifest,),
                framed_blobs=(blob,),
            )
            manifests = (manifest.canonical_object(),)
            descriptor = contract.Q1StreamDescriptorV1(
                stream_kind,
                len(objects),
                cbor.rfc6962_root(objects),
                sum(len(blob) for blob in chunks.framed_blobs),
                len(chunks.manifests),
                cbor.rfc6962_root(manifests),
            )
            diagnostic_commitment = cbor.content_hash(
                "hegel-test-q1-formal-counting-discard-v1",
                (input_signature_id, int(stream_kind), descriptor.canonical_object()),
            )
            external_sort_root = cbor.content_hash(
                "hegel-test-q1-formal-external-sort-v1",
                (input_signature_id, int(stream_kind), descriptor.archive_root),
            )
            stream_items.append(
                SimpleNamespace(
                    stream_kind_id=stream_kind,
                    descriptor=descriptor,
                    chunks=chunks,
                    canonical_record_payload_bytes=sum(
                        len(cbor.canonical_cbor_encode(value)) for value in objects
                    ),
                    diagnostic_commitment=diagnostic_commitment,
                    external_sort_root=external_sort_root,
                )
            )
        streams = tuple(stream_items)
        signature_name = f"signature-{input_signature_id:02d}"
        chunk_objects = []
        endpoint_streams = []
        for stream in streams:
            stream_name = f"stream-{int(stream.stream_kind_id):02d}"
            for manifest, blob in zip(
                stream.chunks.manifests,
                stream.chunks.framed_blobs,
                strict=True,
            ):
                files[
                    f"{signature_name}/{stream_name}/"
                    f"chunk-{manifest.chunk_index:06d}.framed"
                ] = blob
                chunk_objects.append(manifest.canonical_object())
            endpoint_streams.append(
                {
                    "archive_root": _root(stream.descriptor.archive_root),
                    "canonical_record_payload_bytes": (
                        stream.canonical_record_payload_bytes
                    ),
                    "chunk_count": stream.descriptor.chunk_count,
                    "chunk_manifest_subtree_root": _root(
                        stream.descriptor.chunk_manifest_subtree_root
                    ),
                    "counting_discard_commitment": _root(
                        stream.diagnostic_commitment
                    ),
                    "external_sort_projection_root": _root(
                        stream.external_sort_root
                    ),
                    "framed_bytes": stream.descriptor.framed_stream_bytes,
                    "record_count": stream.descriptor.record_count,
                    "stream_kind": stream.stream_kind_id.name,
                    "stream_kind_id": int(stream.stream_kind_id),
                }
            )
        descriptors = tuple(stream.descriptor for stream in streams)
        fixed_point = contract.Q1FixedPointRecordV1(
            input_signature_id=input_signature_id,
            universe_root=frozen_universe.universe_root,
            projection_profile_root=projection_profile_root,
            raw_application_count=810,
            strict_admitted_count=810,
            rewrite_collapse_count=0,
            maximum_depth=3,
            maximum_nodes=6,
            structural_boundary_depth=4,
            work_queue_empty=True,
            zero_delta_full_boundary=True,
            all_eligible_covered=True,
            final_class_delta=0,
            final_cohort_delta=0,
            final_frontier_delta=0,
            final_bank_delta=0,
            program_count=1,
            class_count=1,
            cohort_count=1,
            bank_point_count=1,
            frontier_point_count=1,
            maximum_bank_points_per_class=1,
            maximum_frontier_points_per_class=1,
            program_archive_root=descriptors[0].archive_root,
            bank_archive_root=descriptors[1].archive_root,
            class_archive_root=descriptors[2].archive_root,
            coverage_archive_root=descriptors[3].archive_root,
        )
        saturation_root = contract.signature_saturation_state_root_v1(
            input_signature_id=input_signature_id,
            universe_root=frozen_universe.universe_root,
            semantic_binding_root=semantic_binding_root,
            projection_profile_root=projection_profile_root,
            program_archive_root=descriptors[0].archive_root,
            cohort_archive_root=descriptors[1].archive_root,
            class_archive_root=descriptors[2].archive_root,
            coverage_archive_root=descriptors[3].archive_root,
            fixed_point_record_root=fixed_point.record_root,
        )
        signature_manifest = contract.Q1SignatureArchiveManifestV1(
            input_signature_id,
            frozen_universe.universe_root,
            len(frozen_universe.rows),
            semantic_binding_root,
            projection_profile_root,
            descriptors,
            fixed_point.record_root,
            saturation_root,
            len(chunk_objects),
            cbor.rfc6962_root(tuple(chunk_objects)),
        )
        files[f"{signature_name}/chunk-manifests.framed"] = b"".join(
            contract.frame_canonical_record_v1(value) for value in chunk_objects
        )
        files[f"{signature_name}/fixed-point.framed"] = (
            contract.frame_canonical_record_v1(fixed_point.canonical_object())
        )
        files[f"{signature_name}/signature-manifest.framed"] = (
            contract.frame_canonical_record_v1(
                signature_manifest.canonical_object()
            )
        )
        record_set_preimage = (
            1,
            projection.SNAPSHOT_RECORD_SET_SCHEMA_ID,
            input_signature_id,
            frozen_universe.universe_root,
            tuple(row.canonical_object() for row in program),
            tuple(row.canonical_object() for row in cohort),
            tuple(row.canonical_object() for row in classes),
        )
        record_set_root = cbor.content_hash(
            projection.SNAPSHOT_RECORD_SET_ROOT_DOMAIN,
            record_set_preimage,
        )
        partition_prefix = f"{signature_name}/"
        partition_payload = sum(
            len(payload)
            for path, payload in files.items()
            if path.startswith(partition_prefix)
        )
        record_stream_bytes = sum(
            item.descriptor.framed_stream_bytes for item in streams
        )
        chunk_manifest_stream_bytes = len(
            files[f"{signature_name}/chunk-manifests.framed"]
        )
        stream_commitment = contract.partition_stream_commitment_v1(
            input_signature_id=input_signature_id,
            universe_root=frozen_universe.universe_root,
            raw_application_count=fixed_point.raw_application_count,
            behavior_class_count=fixed_point.class_count,
            cohort_count=fixed_point.cohort_count,
            bank_point_count=fixed_point.bank_point_count,
            frontier_point_count=fixed_point.frontier_point_count,
            maximum_bank_points_per_class=(
                fixed_point.maximum_bank_points_per_class
            ),
            maximum_frontier_points_per_class=(
                fixed_point.maximum_frontier_points_per_class
            ),
            program_record_count=streams[0].descriptor.record_count,
            coverage_record_count=streams[3].descriptor.record_count,
            projected_record_stream_bytes=record_stream_bytes,
            projected_chunk_manifest_stream_bytes=chunk_manifest_stream_bytes,
            stream_diagnostic_commitments=tuple(
                item.diagnostic_commitment for item in streams
            ),
        )
        external_sort_root = contract.partition_external_sort_root_v1(
            input_signature_id=input_signature_id,
            universe_root=frozen_universe.universe_root,
            external_sort_stream_roots=tuple(
                item.external_sort_root for item in streams
            ),
            projected_peak_scratch_bytes=0,
        )
        endpoint_partitions.append(
            {
                "chunk_manifest_archive_root": _root(
                    signature_manifest.chunk_manifest_archive_root
                ),
                "chunk_manifest_count": signature_manifest.chunk_manifest_count,
                "fixed_point_record_root": _root(fixed_point.record_root),
                "input_signature_id": input_signature_id,
                "input_signature_name": (
                    "OddInputV1" if input_signature_id == 1 else "SinkInputV1"
                ),
                "partition_external_sort_root": _root(external_sort_root),
                "partition_stream_commitment_root": _root(stream_commitment),
                "projected_partition_payload_bytes": partition_payload,
                "same_run_record_set_root": _root(record_set_root),
                "saturation_state_root": _root(saturation_root),
                "signature_manifest_root": _root(
                    signature_manifest.manifest_root
                ),
                "streams": endpoint_streams,
                "universe_root": _root(frozen_universe.universe_root),
            }
        )
        signatures.append(
            (
                input_signature_id,
                frozen_universe.universe_root,
                signature_manifest.manifest_root,
                saturation_root,
            )
        )
    bundle = contract.Q1ClosureBundleV1(
        semantic_binding_root,
        projection_profile_root,
        tuple(signatures),
    )
    files["closure-bundle.framed"] = contract.frame_canonical_record_v1(
        bundle.canonical_object()
    )
    return files, bundle, endpoint_partitions


def _write_endpoint(
    root: Path,
    *,
    endpoint_kind: str,
    semantic_byte: int,
) -> None:
    files, bundle, partitions = _formal_tree_bytes(semantic_byte)
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    if endpoint_kind == "python":
        implementation = host.PYTHON_IMPLEMENTATION_ID
    else:
        implementation = host.RUST_IMPLEMENTATION_ID
    result = {
        "action_id": host.ENDPOINT_ACTION_ID,
        "actor_id": endpoint_kind,
        "actual_formal_peak_scratch_bytes": 4096,
        "actual_formal_terminal_live_scratch_bytes": 0,
        "capsule_digest": CAPSULE_DIGEST,
        "closure_bundle_root": _root(bundle.bundle_root),
        "closure_semantics_version": contract.CLOSURE_SEMANTICS_VERSION,
        "diagnostic_only": False,
        "dsl_version": contract.DSL_VERSION,
        "formal_archive_materialized": True,
        "formal_archive_payload_bytes": sum(
            len(payload) for payload in files.values()
        ),
        "formal_capacity_limits_by_signature": FORMAL_CAPACITY_LIMITS,
        "formal_output_bytes_limit": 16 * 1024 * 1024,
        "formal_scratch_bytes_limit": 128 * 1024 * 1024,
        "full_v16_structural_limits_applied": True,
        "implementation_id": implementation,
        "maximum_ast_depth": 3,
        "maximum_ast_node_count": 6,
        "partitions": partitions,
        "prestart_admission_receipt_root": PRESTART_RECEIPT_ROOT,
        "q1_gate_count": 14,
        "q1_gate_mask": 0x3FFF,
        "q1_gate_total": 20,
        "q1_output_slots": [None] * 8,
        "q1_state": "NOT_RUN",
        "role_evaluation_performed": False,
        "run_genesis_root": RUN_GENESIS_ROOT,
        "schema_version": host.ENDPOINT_RESULT_SCHEMA,
        "source_commit": SOURCE_COMMIT,
        "split_accessed": False,
        "status": host.ENDPOINT_RESULT_STATUS,
        "target_truth_accessed": False,
    }
    (root / "endpoint-result.json").write_bytes(host.canonical_json_bytes_v1(result))


@pytest.fixture(scope="module")
def exact_pair(tmp_path_factory):
    root = tmp_path_factory.mktemp("q1-formal-host")
    python_root = root / "python"
    rust_root = root / "rust"
    _write_endpoint(python_root, endpoint_kind="python", semantic_byte=0x91)
    _write_endpoint(rust_root, endpoint_kind="rust", semantic_byte=0x91)
    return python_root, rust_root


def _copy_pair(exact_pair, tmp_path: Path):
    python = tmp_path / "python"
    rust = tmp_path / "rust"
    shutil.copytree(exact_pair[0], python)
    shutil.copytree(exact_pair[1], rust)
    return python, rust


def _dual(python_root: Path, rust_root: Path, **overrides):
    expected = {
        "expected_action_id": host.ACTION_ID,
        "expected_capsule_digest": CAPSULE_DIGEST,
        "expected_source_commit": SOURCE_COMMIT,
        "expected_run_genesis_root": RUN_GENESIS_ROOT,
        "expected_prestart_admission_receipt_root": PRESTART_RECEIPT_ROOT,
    }
    expected.update(overrides)
    return host.dual_formal_host_replay_v1(
        python_root,
        rust_root,
        **expected,
    )


def test_dual_host_replays_every_formal_layer_and_attests_absence(
    exact_pair,
    monkeypatch,
) -> None:
    replayed_endpoints = []
    original_replay = host.replay_endpoint_formal_archive_v1

    def capture_replay(root: Path):
        endpoint = original_replay(root)
        replayed_endpoints.append(endpoint)
        return endpoint

    monkeypatch.setattr(host, "replay_endpoint_formal_archive_v1", capture_replay)
    result = _dual(*exact_pair)
    assert result["actor_id"] == "host"
    assert result["status"] == host.RESULT_STATUS
    assert result["q1_gate_count"] == 14
    assert result["q1_gate_total"] == 20
    assert result["q1_gate_mask"] == 0x3FFF
    assert result["q1_output_slots"] == [None] * 8
    assert result["formal_capacity_limits_by_signature"] == FORMAL_CAPACITY_LIMITS
    assert result["formal_output_bytes_limit"] == 16 * 1024 * 1024
    assert host.FORMAL_SCRATCH_BYTES_LIMIT == 128 * 1024 * 1024
    assert result["formal_scratch_bytes_limit"] == 128 * 1024 * 1024
    assert result["python_rust_formal_bytes_exact"] is True
    assert result["python_rust_replayed_roots_exact"] is True
    assert result["target_split_role_absence_attested"] is True
    assert result["q1_completion_receipt_issued"] is False
    assert [row["record_count_by_stream"] for row in result["signatures"]] == [
        [1, 1, 1, 846],
        [1, 1, 1, 846],
    ]
    assert len(replayed_endpoints) == 1
    for signature in replayed_endpoints[0].signatures:
        for stream in signature.streams:
            assert not hasattr(stream, "record_objects")
            assert not hasattr(stream, "decoded_records")
    assert set(host.Q1FormalStreamReplayV1.__slots__) == {
        "stream_kind_id",
        "descriptor",
        "canonical_record_payload_bytes",
        "manifests",
        "framed_relative_paths",
    }


def test_chunk_tamper_is_rejected_by_manifest_and_record_replay(
    exact_pair,
    tmp_path,
) -> None:
    python, _rust = _copy_pair(exact_pair, tmp_path)
    chunk = python / "signature-01/stream-01/chunk-000000.framed"
    payload = chunk.read_bytes()
    chunk.write_bytes(payload[:-1] + bytes((payload[-1] ^ 1,)))
    with pytest.raises(host.Q1FormalHostReplayError) as caught:
        host.replay_endpoint_formal_archive_v1(python)
    assert caught.value.code == "REJECT_Q1_FORMAL_STREAM"


def test_old_diagnostic_zero_of_twenty_is_not_a_formal_endpoint(
    exact_pair,
    tmp_path,
) -> None:
    python, _rust = _copy_pair(exact_pair, tmp_path)
    path = python / "endpoint-result.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["diagnostic_only"] = True
    value["q1_gate_count"] = 0
    value["q1_gate_mask"] = 0
    path.write_bytes(host.canonical_json_bytes_v1(value))
    with pytest.raises(host.Q1FormalHostReplayError) as caught:
        host.replay_endpoint_formal_archive_v1(python)
    assert caught.value.code == "REJECT_Q1_FORMAL_ENDPOINT_RESULT"


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-diagnostic",
        "bool-gate-count",
        "bool-formal-limit",
        "bool-capacity-limit",
        "scratch-over-limit",
        "extra-partition-field",
        "missing-stream-field",
        "extra-archive-alias",
    ),
)
def test_endpoint_metadata_requires_exact_closed_typed_schema(
    exact_pair,
    tmp_path,
    mutation: str,
) -> None:
    python, _rust = _copy_pair(exact_pair, tmp_path)
    path = python / "endpoint-result.json"
    value = json.loads(path.read_bytes())
    if mutation == "missing-diagnostic":
        del value["diagnostic_only"]
    elif mutation == "bool-gate-count":
        value["q1_gate_count"] = True
    elif mutation == "bool-formal-limit":
        value["formal_output_bytes_limit"] = True
    elif mutation == "bool-capacity-limit":
        value["formal_capacity_limits_by_signature"][0][
            "maximum_behavior_classes"
        ] = True
    elif mutation == "scratch-over-limit":
        value["actual_formal_peak_scratch_bytes"] = 128 * 1024 * 1024 + 1
    elif mutation == "extra-partition-field":
        value["partitions"][0]["partition_payload_bytes"] = value["partitions"][0][
            "projected_partition_payload_bytes"
        ]
    elif mutation == "missing-stream-field":
        del value["partitions"][0]["streams"][0]["framed_bytes"]
    else:
        value["archive_payload_bytes"] = value["formal_archive_payload_bytes"]
    path.write_bytes(host.canonical_json_bytes_v1(value))
    with pytest.raises(host.Q1FormalHostReplayError) as caught:
        host.replay_endpoint_formal_archive_v1(python)
    assert caught.value.code == "REJECT_Q1_FORMAL_ENDPOINT_RESULT"


def test_endpoint_result_requires_exact_canonical_one_line_json(
    exact_pair,
    tmp_path,
) -> None:
    python, _rust = _copy_pair(exact_pair, tmp_path)
    path = python / "endpoint-result.json"
    value = json.loads(path.read_bytes())
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(host.Q1FormalHostReplayError) as caught:
        host.replay_endpoint_formal_archive_v1(python)
    assert caught.value.code == "REJECT_Q1_FORMAL_ENDPOINT_RESULT"
    assert "canonical one-line JSON" in caught.value.detail


def test_two_individually_valid_but_different_archives_do_not_agree(
    exact_pair,
    tmp_path,
) -> None:
    alternate_rust = tmp_path / "alternate-rust"
    _write_endpoint(alternate_rust, endpoint_kind="rust", semantic_byte=0x92)
    with pytest.raises(host.Q1FormalHostReplayError) as caught:
        _dual(exact_pair[0], alternate_rust)
    assert caught.value.code == "REJECT_Q1_FORMAL_DUAL_AGREEMENT"


def test_host_invocation_requires_exact_action_before_replay(exact_pair) -> None:
    with pytest.raises(host.Q1FormalHostReplayError) as caught:
        _dual(
            *exact_pair,
            expected_action_id="replay-full-node6-formal-archive-v2",
        )
    assert caught.value.code == "REJECT_Q1_FORMAL_HOST_INVOCATION"


def test_direct_isolated_tool_replays_without_package_initializer(exact_pair) -> None:
    command = (
        "/usr/bin/python3.10",
        "-I",
        "-S",
        "-B",
        str(ROOT / "tools/phase3_q1_formal_host_replay_v1.py"),
        "--action",
        host.ACTION_ID,
        "--capsule-digest",
        CAPSULE_DIGEST,
        "--source-commit",
        SOURCE_COMMIT,
        "--run-genesis-root",
        RUN_GENESIS_ROOT,
        "--prestart-admission-receipt-root",
        PRESTART_RECEIPT_ROOT,
        "--python-root",
        str(exact_pair[0]),
        "--rust-root",
        str(exact_pair[1]),
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    value = json.loads(completed.stdout)
    assert value["actor_id"] == "host"
    assert value["implementation_id"] == host.IMPLEMENTATION_ID
    assert value["action_id"] == host.ACTION_ID
    assert value["status"] == host.RESULT_STATUS


def test_isolated_tool_bootstrap_has_exact_project_module_source_closure() -> None:
    tool = ROOT / "tools/phase3_q1_formal_host_replay_v1.py"
    probe = (
        "import json,runpy;"
        f"value=runpy.run_path({str(tool)!r});"
        "print(json.dumps({"
        "'paths':value['EXPECTED_MODULE_SOURCE_PATHS'],"
        "'modules':value['_loaded_project_modules']()}))"
    )
    completed = subprocess.run(
        ("/usr/bin/python3.10", "-I", "-S", "-B", "-c", probe),
        cwd=ROOT,
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    value = json.loads(completed.stdout)
    assert tuple(value["paths"]) == EXPECTED_HOST_MODULE_SOURCE_PATHS
    assert tuple(value["modules"]) == tuple(
        sorted(
            "hegel_machine." + Path(relative).stem
            for relative in EXPECTED_HOST_MODULE_SOURCE_PATHS
        )
    )


def test_host_replay_source_has_no_execution_or_target_input_dependencies() -> None:
    core_source = Path(host.__file__).read_text(encoding="utf-8")
    tool_source = (
        ROOT / "tools/phase3_q1_formal_host_replay_v1.py"
    ).read_text(encoding="utf-8")
    forbidden_core = (
        "phase3_dsl_v1",
        "phase3_m25_rows_v1",
        "phase3_m25_split_v1",
        "subprocess",
        "socket",
        "urllib",
    )
    assert all(token not in core_source for token in forbidden_core)
    assert "subprocess" not in tool_source
