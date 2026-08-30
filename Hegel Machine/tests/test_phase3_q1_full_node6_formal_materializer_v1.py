from __future__ import annotations

from dataclasses import asdict, replace
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hegel_machine import phase3_q1_capacity_preflight_v1 as capacity
from hegel_machine import phase3_q1_formal_archive_contract_v1 as formal
from hegel_machine import phase3_q1_full_node6_capacity_preflight_v1 as full
from hegel_machine import phase3_q1_full_node6_formal_materializer_v1 as materializer
from hegel_machine import phase3_q1_full_node6_projection_v1 as projection
from hegel_machine import phase3_q1_prestart_wire_v1 as prestart
from hegel_machine.strict_cbor_v1 import rfc6962_root


ENTRYPOINT = ROOT / "tools/phase3_q1_full_node6_formal_materializer_entrypoint_v1.py"


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "q1_formal_python_materializer_test",
        ENTRYPOINT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


entrypoint = _load_entrypoint()


def _expected_formal_limits_from_gate11():
    envelope = prestart.FORMAL_RESOURCE_ENVELOPE
    signature_resources = {
        signature_id: dict(resources)
        for signature_id, resources in envelope[2]
    }
    python_resources = next(
        dict(resources)
        for _actor_number, actor_id, resources in envelope[3]
        if actor_id == b"python"
    )
    return tuple(
        capacity.PreflightLimitsV1(
            maximum_ast_depth=3,
            maximum_ast_node_count=6,
            maximum_raw_operator_applications=signature_resources[signature_id][1],
            maximum_behavior_classes=signature_resources[signature_id][2],
            maximum_visible_frontier_points=signature_resources[signature_id][3],
            maximum_visible_frontier_points_per_class=(
                signature_resources[signature_id][4]
            ),
            maximum_continuation_bank_points=signature_resources[signature_id][5],
            maximum_continuation_bank_points_per_class=(
                signature_resources[signature_id][6]
            ),
            maximum_work_queue_points=signature_resources[signature_id][7],
            maximum_saturation_rounds=signature_resources[signature_id][8],
            maximum_wall_time_seconds=python_resources[12],
        )
        for signature_id in (1, 2)
    )


@pytest.fixture(scope="module")
def synthetic_full_node6_same_run():
    formal_limits = _expected_formal_limits_from_gate11()
    node3_limits = tuple(
        replace(limits, maximum_ast_node_count=3)
        for limits in formal_limits
    )
    measured = tuple(
        capacity._execute_q1_partition_capacity_preflight_state_v1(
            signature_id,
            limits=node3_limits[signature_id - 1],
        )
        for signature_id in (1, 2)
    )
    node3_snapshots = tuple(
        full._snapshot_from_result_and_material_v1(
            result,
            state.immutable_snapshot_material(),
        )
        for result, state in measured
    )
    formal_shape_snapshots = tuple(
        replace(
            snapshot,
            limits=formal_limits[index],
            terminal_status=capacity.PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY,
        )
        for index, snapshot in enumerate(node3_snapshots)
    )
    partitions = tuple(
        full.FullNode6PartitionDiagnosticV1.from_snapshot(snapshot)
        for snapshot in formal_shape_snapshots
    )
    diagnostic = full.FullNode6CapacityDiagnosticV1(
        schema_version=full.SCHEMA_VERSION,
        action_id=full.ACTION_ID,
        capsule_digest="sha256:" + "a" * 64,
        source_commit="b" * 40,
        terminal_status=full.TERMINAL_SATURATED,
        partitions=partitions,
    )
    return diagnostic, formal_shape_snapshots, partitions


def _formal_admission() -> materializer.Q1FormalArchiveMaterializerAdmissionV1:
    return materializer.Q1FormalArchiveMaterializerAdmissionV1(
        schema_version=materializer.ADMISSION_SCHEMA_VERSION,
        action_id=materializer.ACTION_ID,
        capsule_digest="sha256:" + "a" * 64,
        source_commit="b" * 40,
        run_genesis_root="sha256:" + "c" * 64,
        prestart_admission_receipt_root="sha256:" + "d" * 64,
        projection_profile_root=projection.FROZEN_PROJECTION_PROFILE_ROOT_ID,
        q1_state="NOT_RUN",
        q1_gate_count=14,
        q1_gate_total=20,
        q1_gate_mask=0x3FFF,
        q1_output_slots=(None,) * 8,
        diagnostic_only=False,
        target_truth_accessed=False,
        split_accessed=False,
        role_evaluation_performed=False,
    )


def _node3_projection_runner(
    diagnostic: full.FullNode6CapacityDiagnosticV1,
    snapshots: tuple[object, ...],
    *,
    scratch_root: Path,
    retained_chunk_consumer=None,
) -> projection.FullNode6ProjectionResultV1:
    bounded_snapshots = tuple(
        replace(
            snapshot,
            limits=replace(snapshot.limits, maximum_ast_node_count=3),
            terminal_status=capacity.LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED,
        )
        for snapshot in snapshots
    )
    tracker = projection._ScratchTrackerV1()
    try:
        partitions = tuple(
            projection._project_same_run_partition_with_tracker_v1(
                snapshot,
                scratch_root=scratch_root,
                tracker=tracker,
                retained_chunk_consumer=retained_chunk_consumer,
            )
            for snapshot in bounded_snapshots
        )
    finally:
        tracker.release_after_transient_tree_cleanup()
    closure, archive_result = projection._assemble_formal_aggregate_v1(partitions)
    return projection.FullNode6ProjectionResultV1(
        capsule_digest=diagnostic.capsule_digest,
        source_commit=diagnostic.source_commit,
        partitions=partitions,
        closure_bundle=closure,
        archive_projection_result=archive_result,
        actual_preflight_peak_scratch_bytes=tracker.peak_logical_bytes,
        actual_preflight_scratch_live_bytes_at_terminal=0,
    )


def _archive_projection(
    synthetic_full_node6_same_run,
    scratch_root: Path,
) -> projection.FullNode6ProjectionResultV1:
    diagnostic, snapshots, _rows = synthetic_full_node6_same_run
    return _node3_projection_runner(
        diagnostic,
        snapshots,
        scratch_root=scratch_root,
    )


def test_retained_node3_kat_replays_exact_frozen_roots_and_lengths(
    synthetic_full_node6_same_run,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_scratch = tmp_path / "baseline-scratch"
    baseline_scratch.mkdir()
    baseline = _archive_projection(synthetic_full_node6_same_run, baseline_scratch)
    assert tuple(baseline_scratch.iterdir()) == ()
    for partition in baseline.partitions:
        for stream in partition.streams:
            assert not hasattr(stream, "framed_blobs")
            assert not hasattr(stream, "chunks")

    output_root = tmp_path / "output"
    output_root.mkdir()
    materializer_scratch = tmp_path / "materializer-scratch"
    materializer_scratch.mkdir()
    diagnostic, snapshots, rows = synthetic_full_node6_same_run
    admission = _formal_admission()
    admission_path = tmp_path / "formal-admission.json"
    admission_path.write_text(
        json.dumps(asdict(admission), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    expected_formal_limits = _expected_formal_limits_from_gate11()
    assert all(
        limits != capacity.PreflightLimitsV1()
        for limits in expected_formal_limits
    )
    observed: list[tuple[int, capacity.PreflightLimitsV1]] = []
    fsync_modes: list[int] = []
    real_fsync = os.fsync

    def tracked_fsync(descriptor: int) -> None:
        fsync_modes.append(os.fstat(descriptor).st_mode)
        real_fsync(descriptor)

    monkeypatch.setattr(materializer.os, "fsync", tracked_fsync)

    def partition_runner(
        signature_id: int,
        limits: capacity.PreflightLimitsV1,
    ):
        assert limits == expected_formal_limits[signature_id - 1]
        observed.append((signature_id, limits))
        return rows[signature_id - 1], snapshots[signature_id - 1]

    stdout = entrypoint.run_endpoint_v1(
        [
            "--action",
            materializer.ACTION_ID,
            "--admission-json",
            str(admission_path),
            "--capsule-digest",
            admission.capsule_digest,
            "--source-commit",
            admission.source_commit,
            "--run-genesis-root",
            admission.run_genesis_root,
            "--prestart-admission-receipt-root",
            admission.prestart_admission_receipt_root,
            "--output-dir",
            str(output_root),
            "--scratch-dir",
            str(materializer_scratch),
        ],
        partition_runner=partition_runner,
        projection_runner=_node3_projection_runner,
        module_verifier=lambda: (),
    )
    assert observed == list(zip((1, 2), expected_formal_limits, strict=True))
    assert tuple(materializer_scratch.iterdir()) == ()

    expected_files = {"closure-bundle.framed", "endpoint-result.json"}
    for partition in baseline.partitions:
        signature_name = f"signature-{partition.input_signature_id:02d}"
        expected_files.update(
            {
                f"{signature_name}/chunk-manifests.framed",
                f"{signature_name}/fixed-point.framed",
                f"{signature_name}/signature-manifest.framed",
            }
        )
        all_manifest_objects: list[tuple[object, ...]] = []
        for stream in partition.streams:
            stream_name = f"stream-{int(stream.stream_kind_id):02d}"
            record_objects: list[tuple[object, ...]] = []
            framed_bytes = 0
            for manifest in stream.chunk_manifests:
                relative = (
                    f"{signature_name}/{stream_name}/"
                    f"chunk-{manifest.chunk_index:06d}.framed"
                )
                expected_files.add(relative)
                blob = (output_root / relative).read_bytes()
                assert len(blob) == manifest.framed_blob_length
                assert formal.framed_blob_hash_v1(blob) == manifest.framed_blob_hash
                chunk_records = formal.replay_framed_records_v1(blob)
                assert len(chunk_records) == manifest.record_count
                assert rfc6962_root(chunk_records) == manifest.record_subtree_root
                record_objects.extend(chunk_records)
                framed_bytes += len(blob)
                all_manifest_objects.append(manifest.canonical_object())
            assert len(record_objects) == stream.descriptor.record_count
            assert framed_bytes == stream.descriptor.framed_stream_bytes
            assert rfc6962_root(tuple(record_objects)) == stream.descriptor.archive_root

        manifest_frames = formal.replay_framed_records_v1(
            (output_root / signature_name / "chunk-manifests.framed").read_bytes()
        )
        assert manifest_frames == tuple(all_manifest_objects)
        assert rfc6962_root(manifest_frames) == (
            partition.signature_archive_manifest.chunk_manifest_archive_root
        )
        fixed_point = formal.replay_framed_records_v1(
            (output_root / signature_name / "fixed-point.framed").read_bytes()
        )
        signature_manifest = formal.replay_framed_records_v1(
            (output_root / signature_name / "signature-manifest.framed").read_bytes()
        )
        assert fixed_point == (partition.fixed_point_record.canonical_object(),)
        assert signature_manifest == (
            partition.signature_archive_manifest.canonical_object(),
        )

    closure = formal.replay_framed_records_v1(
        (output_root / "closure-bundle.framed").read_bytes()
    )
    assert closure == (baseline.closure_bundle.canonical_object(),)
    actual_files = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    assert actual_files == expected_files
    assert sum(stat.S_ISREG(mode) for mode in fsync_modes) == len(expected_files)
    assert sum(stat.S_ISDIR(mode) for mode in fsync_modes) == 11
    actual_archive_bytes = sum(
        path.stat().st_size for path in output_root.rglob("*.framed")
    )
    endpoint_payload = (output_root / "endpoint-result.json").read_bytes()
    endpoint = json.loads(endpoint_payload)
    assert set(endpoint) == {
        "action_id",
        "actor_id",
        "actual_formal_peak_scratch_bytes",
        "actual_formal_terminal_live_scratch_bytes",
        "capsule_digest",
        "closure_bundle_root",
        "closure_semantics_version",
        "diagnostic_only",
        "dsl_version",
        "formal_archive_materialized",
        "formal_archive_payload_bytes",
        "formal_capacity_limits_by_signature",
        "formal_output_bytes_limit",
        "formal_scratch_bytes_limit",
        "full_v16_structural_limits_applied",
        "implementation_id",
        "maximum_ast_depth",
        "maximum_ast_node_count",
        "partitions",
        "prestart_admission_receipt_root",
        "q1_gate_count",
        "q1_gate_mask",
        "q1_gate_total",
        "q1_output_slots",
        "q1_state",
        "role_evaluation_performed",
        "run_genesis_root",
        "schema_version",
        "source_commit",
        "split_accessed",
        "status",
        "target_truth_accessed",
    }
    assert all(
        set(partition) == {
            "chunk_manifest_archive_root",
            "chunk_manifest_count",
            "fixed_point_record_root",
            "input_signature_id",
            "input_signature_name",
            "partition_external_sort_root",
            "partition_stream_commitment_root",
            "projected_partition_payload_bytes",
            "same_run_record_set_root",
            "saturation_state_root",
            "signature_manifest_root",
            "streams",
            "universe_root",
        }
        for partition in endpoint["partitions"]
    )
    assert all(
        set(stream) == {
            "archive_root",
            "canonical_record_payload_bytes",
            "chunk_count",
            "chunk_manifest_subtree_root",
            "counting_discard_commitment",
            "external_sort_projection_root",
            "framed_bytes",
            "record_count",
            "stream_kind",
            "stream_kind_id",
        }
        for partition in endpoint["partitions"]
        for stream in partition["streams"]
    )
    assert actual_archive_bytes == endpoint["formal_archive_payload_bytes"]
    assert actual_archive_bytes == (
        baseline.archive_projection_result.projected_archive_payload_bytes_per_endpoint
    )
    assert endpoint["closure_bundle_root"] == (
        "sha256:" + baseline.closure_bundle.bundle_root.hex()
    )

    assert endpoint_payload == stdout
    assert endpoint_payload == projection.canonical_json_bytes_v1(endpoint)
    assert endpoint["implementation_id"] == materializer.IMPLEMENTATION_ID
    assert endpoint["action_id"] == materializer.ACTION_ID
    assert endpoint["actor_id"] == "python"
    assert endpoint["schema_version"] == (
        "hegel-phase3a-q1-full-node6-formal-archive-materializer/1"
    )
    assert endpoint["status"] == (
        "FULL_NODE6_FORMAL_ARCHIVE_MATERIALIZED_Q1_NOT_RUN"
    )
    assert endpoint["formal_archive_materialized"] is True
    assert endpoint["diagnostic_only"] is False
    limit_field_names = (
        "maximum_ast_depth",
        "maximum_ast_node_count",
        "maximum_raw_operator_applications",
        "maximum_behavior_classes",
        "maximum_visible_frontier_points",
        "maximum_visible_frontier_points_per_class",
        "maximum_continuation_bank_points",
        "maximum_continuation_bank_points_per_class",
        "maximum_work_queue_points",
        "maximum_saturation_rounds",
        "maximum_wall_time_seconds",
    )
    expected_limit_rows = [
        {
            "input_signature_id": signature_id,
            **{
                field_name: getattr(limits, field_name)
                for field_name in limit_field_names
            },
        }
        for signature_id, limits in zip(
            (1, 2),
            expected_formal_limits,
            strict=True,
        )
    ]
    assert endpoint["formal_capacity_limits_by_signature"] == expected_limit_rows
    assert all(
        set(row) == {
            "input_signature_id",
            "maximum_ast_depth",
            "maximum_ast_node_count",
            "maximum_raw_operator_applications",
            "maximum_behavior_classes",
            "maximum_visible_frontier_points",
            "maximum_visible_frontier_points_per_class",
            "maximum_continuation_bank_points",
            "maximum_continuation_bank_points_per_class",
            "maximum_work_queue_points",
            "maximum_saturation_rounds",
            "maximum_wall_time_seconds",
        }
        for row in endpoint["formal_capacity_limits_by_signature"]
    )
    assert endpoint["formal_output_bytes_limit"] == 16_777_216
    assert endpoint["formal_scratch_bytes_limit"] == 134_217_728
    assert actual_archive_bytes + len(endpoint_payload) <= 16_777_216
    assert endpoint["actual_formal_peak_scratch_bytes"] <= 134_217_728
    assert endpoint["run_genesis_root"] == _formal_admission().run_genesis_root
    assert endpoint["prestart_admission_receipt_root"] == (
        _formal_admission().prestart_admission_receipt_root
    )
    assert endpoint["q1_state"] == "NOT_RUN"
    assert endpoint["q1_gate_count"] == 14
    assert endpoint["q1_gate_total"] == 20
    assert endpoint["q1_gate_mask"] == 0x3FFF
    assert endpoint["q1_output_slots"] == [None] * 8
    assert endpoint["target_truth_accessed"] is False
    assert endpoint["split_accessed"] is False
    assert endpoint["role_evaluation_performed"] is False


def test_formal_python_resource_guards_are_inclusive_at_frozen_bounds() -> None:
    historical_python_limits = next(
        dict(rows)
        for _ordinal, actor_id, rows in prestart.FORMAL_RESOURCE_ENVELOPE[3]
        if actor_id == b"python"
    )
    assert historical_python_limits[10] == 4_194_304
    assert materializer.FORMAL_SCRATCH_BYTES_LIMIT == 134_217_728
    materializer._require_formal_scratch_limit_v1(134_217_728)
    materializer._require_formal_output_limit_v1(16_777_215, 1)
    with pytest.raises(
        materializer.Q1FormalArchiveMaterializationError,
        match="INCONCLUSIVE_Q1_FORMAL_RESOURCE_GUARD",
    ):
        materializer._require_formal_scratch_limit_v1(134_217_729)
    with pytest.raises(
        materializer.Q1FormalArchiveMaterializationError,
        match="INCONCLUSIVE_Q1_FORMAL_RESOURCE_GUARD",
    ):
        materializer._require_formal_output_limit_v1(16_777_216, 1)


def test_formal_cli_rejects_output_scratch_alias_before_measurement(
    tmp_path: Path,
) -> None:
    shared_root = tmp_path / "shared"
    shared_root.mkdir()
    admission = _formal_admission()
    admission_path = tmp_path / "formal-admission.json"
    admission_path.write_text(
        json.dumps(asdict(admission), sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    measurements: list[int] = []

    def forbidden_partition_runner(
        signature_id: int,
        _limits: capacity.PreflightLimitsV1,
    ) -> tuple[object, object]:
        measurements.append(signature_id)
        raise AssertionError("path alias must be rejected before measurement")

    with pytest.raises(
        entrypoint.EndpointError,
        match="output and scratch directories alias",
    ):
        entrypoint.run_endpoint_v1(
            [
                "--action",
                materializer.ACTION_ID,
                "--admission-json",
                str(admission_path),
                "--capsule-digest",
                admission.capsule_digest,
                "--source-commit",
                admission.source_commit,
                "--run-genesis-root",
                admission.run_genesis_root,
                "--prestart-admission-receipt-root",
                admission.prestart_admission_receipt_root,
                "--output-dir",
                str(shared_root),
                "--scratch-dir",
                str(shared_root),
            ],
            partition_runner=forbidden_partition_runner,
            projection_runner=lambda *_args, **_kwargs: None,
            module_verifier=lambda: (),
        )
    assert measurements == []
    assert tuple(shared_root.iterdir()) == ()


def test_formal_api_rejects_diagnostic_action_before_projection(
    synthetic_full_node6_same_run,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic, snapshots, _rows = synthetic_full_node6_same_run
    output_root = tmp_path / "output"
    output_root.mkdir()
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()

    diagnostic_admission = full.FullNode6AdmissionV1(
        schema_version=full.ADMISSION_SCHEMA_VERSION,
        action_id=full.ACTION_ID,
        capsule_digest=diagnostic.capsule_digest,
        source_commit=diagnostic.source_commit,
        predecessor_capsule_digest="sha256:" + "1" * 64,
        predecessor_terminal_receipt_digest="sha256:" + "2" * 64,
        predecessor_artifact_sha256="sha256:" + "3" * 64,
        projection_profile_root=projection.FROZEN_PROJECTION_PROFILE_ROOT_ID,
        diagnostic_only=True,
        q1_state="NOT_RUN",
        q1_gate_count=0,
        q1_gate_mask=0,
        q1_output_slots=(None,) * 8,
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("diagnostic action must not reach formal projection")

    monkeypatch.setattr(projection, "project_full_node6_same_run_v1", forbidden)
    with pytest.raises(
        materializer.Q1FormalArchiveMaterializationError,
        match="REJECT_Q1_FORMAL_MATERIALIZER_ADMISSION",
    ):
        materializer.materialize_full_node6_formal_archive_v1(
            diagnostic,
            snapshots,
            admission=diagnostic_admission,  # type: ignore[arg-type]
            output_root=output_root,
            scratch_root=scratch_root,
        )
    assert tuple(output_root.iterdir()) == ()
    assert tuple(scratch_root.iterdir()) == ()


def test_formal_cli_has_no_diagnostic_action_route() -> None:
    with pytest.raises(
        entrypoint.EndpointError,
        match="Q1_FORMAL_MATERIALIZER_ACTION_NOT_ADMITTED",
    ):
        entrypoint.run_endpoint_v1(["--action", full.ACTION_ID])
    source = ENTRYPOINT.read_text(encoding="utf-8")
    assert entrypoint.ACTION_ID == materializer.ACTION_ID
    assert entrypoint.IMPLEMENTATION_ID == materializer.IMPLEMENTATION_ID
    assert "--run-genesis-root" in source
    assert "--prestart-admission-receipt-root" in source
    assert "phase3_q1_prestart_wire_v1" in entrypoint.MODULE_NAMES
    assert len(entrypoint.MODULE_NAMES) == 35


@pytest.mark.parametrize(
    "changes",
    (
        {"q1_gate_count": 0, "q1_gate_mask": 0},
        {"run_genesis_root": None},
        {"prestart_admission_receipt_root": None},
        {"q1_output_slots": (None,) * 7 + (b"not-null",)},
    ),
)
def test_formal_admission_rejects_non_14_of_20_or_null_root(changes) -> None:
    with pytest.raises(
        materializer.Q1FormalArchiveMaterializationError,
        match="REJECT_Q1_FORMAL_MATERIALIZER_ADMISSION",
    ):
        replace(_formal_admission(), **changes)
