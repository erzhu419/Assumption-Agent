from __future__ import annotations

from dataclasses import asdict, replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hegel_machine import phase3_q1_archive_projection_v1 as archive
from hegel_machine import phase3_q1_capacity_preflight_v1 as capacity
from hegel_machine import phase3_q1_full_node6_capacity_preflight_v1 as full
from hegel_machine import phase3_q1_full_node6_projection_v1 as projection
from hegel_machine import phase3_q1_partition_snapshot_v1 as snapshots


ENTRYPOINT = ROOT / "tools/phase3_q1_full_node6_capacity_entrypoint_v1.py"


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location("q1_full_python_endpoint_test", ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


endpoint = _load_entrypoint()


@pytest.fixture(scope="module")
def node3_same_run_material(tmp_path_factory: pytest.TempPathFactory):
    limits = capacity.PreflightLimitsV1(maximum_ast_node_count=3)
    measured = tuple(
        capacity._execute_q1_partition_capacity_preflight_state_v1(
            signature_id,
            limits=limits,
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
    # Projection must consume only the already-returned immutable snapshots.
    # Each patched symbol would otherwise start a second quotient measurement
    # or a fresh representative behavior replay.
    with (
        patch.object(
            capacity,
            "_execute_q1_partition_capacity_preflight_state_v1",
            side_effect=AssertionError("second quotient measurement"),
        ),
        patch.object(
            snapshots,
            "validate_q1_partition_snapshot_v1",
            side_effect=AssertionError("public snapshot validator"),
        ),
        patch.object(
            snapshots,
            "_validate_q1_partition_snapshot_self_consistency_v1",
            side_effect=AssertionError("fresh behavior replay"),
        ),
        patch.object(
            capacity,
            "evaluate_canonical_ast_on_environments_v1",
            side_effect=AssertionError("fresh quotient behavior evaluation"),
        ),
        patch.object(
            archive,
            "evaluate_canonical_ast_on_environments_v1",
            side_effect=AssertionError("fresh record-set behavior evaluation"),
        ),
        patch.object(
            archive,
            "counting_discard_record_stream_v1",
            side_effect=AssertionError("materializing old public counting sink"),
        ),
    ):
        scratch_root = tmp_path_factory.mktemp("full-node6-projection-scratch")
        projected = tuple(
            projection.project_same_run_partition_v1(
                snapshot,
                scratch_root=scratch_root,
            )
            for snapshot in node3_snapshots
        )
        assert tuple(scratch_root.iterdir()) == ()
    return measured, node3_snapshots, projected


def _admission() -> full.FullNode6AdmissionV1:
    return full.FullNode6AdmissionV1(
        schema_version=full.ADMISSION_SCHEMA_VERSION,
        action_id=full.ACTION_ID,
        capsule_digest="sha256:" + "1" * 64,
        source_commit="2" * 40,
        predecessor_capsule_digest="sha256:" + "3" * 64,
        predecessor_terminal_receipt_digest="sha256:" + "4" * 64,
        predecessor_artifact_sha256="sha256:" + "5" * 64,
        projection_profile_root=projection.FROZEN_PROJECTION_PROFILE_ROOT_ID,
        diagnostic_only=True,
        q1_state="NOT_RUN",
        q1_gate_count=0,
        q1_gate_mask=0,
        q1_output_slots=(None,) * 8,
    )


def _mock_full_result(node3_same_run_material):
    measured, node3_snapshots, _projected = node3_same_run_material
    full_limits = capacity.PreflightLimitsV1()
    observed: list[int] = []

    def fake_execute(signature_id: int, *, limits: object):
        assert limits == full_limits
        observed.append(signature_id)
        result, _state = measured[signature_id - 1]
        return (
            replace(
                result,
                limits=full_limits,
                full_v16_structural_limits_applied=True,
                maximum_ast_depth=3,
                maximum_ast_node_count=6,
                terminal_status=capacity.PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY,
            ),
            object(),
        )

    def fake_transfer(result: object, _state: object):
        return replace(
            node3_snapshots[result.input_signature_id - 1],
            limits=full_limits,
            terminal_status=capacity.PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY,
        )

    with (
        patch.object(
            capacity,
            "_execute_q1_partition_capacity_preflight_state_v1",
            side_effect=fake_execute,
        ),
        patch.object(
            full,
            "_snapshot_from_result_and_state_transfer_v1",
            side_effect=fake_transfer,
        ),
    ):
        diagnostic, same_run_snapshots = full.run_full_node6_capacity_preflight_v1(
            _admission()
        )
    return observed, diagnostic, same_run_snapshots


def test_node3_library_golden_has_four_formal_counting_discard_streams(
    node3_same_run_material,
) -> None:
    _measured, _snapshots, projected = node3_same_run_material
    assert [
        [stream.descriptor.record_count for stream in row.streams]
        for row in projected
    ] == [[110, 86, 40, 846], [144, 112, 28, 846]]
    for row in projected:
        assert tuple(stream.stream_kind_id for stream in row.streams) == tuple(
            kind for _name, kind in projection.STREAM_REGISTRY
        )
        assert all(stream.descriptor.framed_stream_bytes > 0 for stream in row.streams)
        assert all(not hasattr(stream, "framed_blobs") for stream in row.streams)
        assert row.q1_state == "NOT_RUN"
        assert row.q1_output_slots == (None,) * 8
        assert row.target_truth_accessed is False
        assert row.split_accessed is False
        assert row.role_evaluation_performed is False
        assert row.fixed_point_record.projection_profile_root == (
            projection.FROZEN_PROJECTION_PROFILE_ROOT
        )
        assert row.signature_archive_manifest.saturation_state_root == (
            row.saturation_state_root
        )


def test_actual_scratch_tracker_counts_atomic_merge_high_water(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = projection._ScratchTrackerV1()
    monkeypatch.setattr(projection, "COVERAGE_SORT_BUFFER_BYTES", 22)
    sorter = projection._SpillSorter(tmp_path, "scratch-accounting-kat", tracker)
    for ordinal in range(3):
        # The serialized row is exactly 3 + 4 + 4 + 1 + 10 = 22 bytes.
        sorter.add(b"\x00\x00\x00", bytes((ordinal + 1,)), b"x" * 10)
    final_path = sorter.finish()
    assert final_path is not None
    # All three 22-byte inputs and the completed 66-byte merge output are
    # simultaneously live immediately before the atomic input release.
    assert tracker.peak_logical_bytes == 132
    projection._unlink_run_v1(final_path, tracker)
    tracker.require_terminal_empty()
    assert tracker.live == {}


def test_failed_partition_projection_clears_registered_scratch_ledger(
    node3_same_run_material,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[projection._ScratchTrackerV1] = []

    def fail_with_registered_file(
        _snapshot: object,
        scratch: Path,
        tracker: projection._ScratchTrackerV1,
    ) -> object:
        path = scratch / "registered-partial.run"
        with path.open("xb", buffering=0) as target:
            tracker.create(path)
            target.write(b"partial")
        captured.append(tracker)
        raise RuntimeError("injected tracked projection failure")

    monkeypatch.setattr(
        projection,
        "_bounded_project_same_run_partition_v1",
        fail_with_registered_file,
    )
    with pytest.raises(RuntimeError, match="injected tracked projection failure"):
        projection.project_same_run_partition_v1(
            node3_same_run_material[1][0],
            scratch_root=tmp_path,
        )
    assert tuple(tmp_path.iterdir()) == ()
    assert len(captured) == 1
    assert captured[0].peak_logical_bytes == len(b"partial")
    assert captured[0].live == {}


def test_bounded_formal_wire_matches_frozen_node3_oracle(
    node3_same_run_material,
) -> None:
    _measured, _snapshots, projected = node3_same_run_material
    expected_roots = (
        "555a6885e30d3584f544e7abaae4b30af192d88c8ddcccaa9b4e4073a92f212e",
        "96045ee6b9a39a84a6032c690efab33a880f0ac5fc557709600623a48cad843d",
    )
    expected_top = ((294548, 167936), (251526, 167936))
    expected_streams = (
        (
            (110, 31422, 31862, "c13a827f47949732a2cc252b0f81691e1c802d6ce5fe835826adfa378321c87b", "7771d4662573ac7f721f204dd0c0d26fb43e674d1f74c9df7f2484cf4eee9cb8", "0489c519c911180de18197ed0bac37207dbb1e9c4caf32e0aaddf5db7f702c4d", "dcfdd0cc2c6662bf0ede92975ec5c45d1e2fe679ffe786c5fabf524976d1f565"),
            (86, 28242, 28586, "2883f3e3c5cbf5bfca4a5e0cee3d32de5ab03e2ef39c967c0eb7937d57c8515c", "6ccf958e86e1265911f2ccfd8ab4d98f9753e3c6042a490fffdadf27b833af44", "006e820758c5a27696b10fd5ec34fae5b56cd5cbf1dd60000eb43d8c01252fa1", "86bca9fa0f3ef83f8a6a895080752ff1d5999f6e2811936f0b4d480b088aff7f"),
            (40, 77690, 77850, "76f0a88caebbb53ea1b386c3e281e7ab5b2036f201b959f6eda5165723417c30", "920f993599b92c9d05e48e2c71c838ad2a8c47cf8d0d041ce1d186cfd286fa50", "ff3bf80d7b5d883cdbbf3babdd172e15a5f4f9536a2a12b26e33fa9861a9b710", "c9dac32e2a5625e31b4d64a3487a0354e57f8dfa8cb65a460215e6aee68a98dd"),
            (846, 152866, 156250, "fe056f876706971d5bd15959325e4fdcc164a5ad66303d0a61bfbff88fae929c", "727797a029bdea79c7cb9ff5e85321520928056b630b25ed2e49215cb6b14a5a", "0558238849ab04470880a5e0c19b9bb8ccf1ea4d01bab54297d496dcbb2045fa", "8768eff30d94161db322730f9e936597c5de168cccfa06cf9694633ea43a460e"),
        ),
        (
            (144, 41210, 41786, "67a13e35ac7863f0348d2d1933685f27d09d45de5a2458b3c3185cd06bb078dd", "fd7d82fc8d3bb0f41c07a438a4a76292f6a1a8c9e84ab812008d48f3d233914a", "fa711a730619d7153d8c3b7e6fca24e2b89d1fa19761c27a08628286ca3c300b", "ea40c0b19d2ac27abcc6a0974b561cb4a126ed306e50e2a5936ba536390216ce"),
            (112, 36836, 37284, "3cbd765f4ff301df0fec87eee61f4b69bf0fe5af3483bf8f2b894379ad047972", "fe1a6d4c7712fd70f137473a5d40c512b8deaf5c308fc29e03bb894c7f0892a7", "0824b01e3652694649fc0b93f96dec6465c9b9de6ad3d57bde2b130eefbdbeb8", "352c7fedf23f2f9d89f63d9a1b874888608be1ee4a8cf9ca000e51d6ca954453"),
            (28, 16094, 16206, "7d35bad68c4768c6666e403d03d62f24c9aaec6414e1ab84513a05a7a661492d", "01b414e800c42803d92d48c11ec2c44b2509a1592bf532954cb11dd01a95515d", "4a289ec2eeb355dea073856c232d9638b3f177ab163e2987a47aca1b943e1aaa", "d52de0239fe1d1b3cadba59c32b7f7eb0fa76e9ece7e23c5f454102070e5537b"),
            (846, 152866, 156250, "bdc7821d3a96087c1f0b97d7d6e0317e953d00dbf748d2e774eb3040ff0dc6ea", "9cb8360549c9eeadc4fdb56545ec2358f2db940d3c902c90156de5131d9060c5", "77da55805ee2d3a90a3ab5bc20b250a61ac3c89ef112c928e266998de8b2e88b", "f447fa48e630aca20e2c92ffae491987bfd8ede863c0fa55ec0800906184ea01"),
        ),
    )
    for partition, root, top, frozen_streams in zip(
        projected,
        expected_roots,
        expected_top,
        expected_streams,
        strict=True,
    ):
        assert partition.same_run_record_set_root.hex() == root
        formal = projection.formal_wire_projection_json_v1(partition)
        assert (
            formal["projected_record_stream_bytes"],
            formal["projected_peak_formal_scratch_bytes"],
        ) == top
        assert "projected_formal_archive_payload_bytes" not in formal
        actual_streams = tuple(
            (
                stream.descriptor.record_count,
                stream.canonical_record_payload_bytes,
                stream.descriptor.framed_stream_bytes,
                stream.descriptor.archive_root.hex(),
                stream.descriptor.chunk_manifest_subtree_root.hex(),
                stream.diagnostic_commitment.hex(),
                stream.external_sort_projection.diagnostic_root.hex(),
            )
            for stream in partition.streams
        )
        assert actual_streams == frozen_streams


def test_strict_formal_projection_assembler_matches_cross_language_node3_kat(
    node3_same_run_material,
) -> None:
    projected = node3_same_run_material[2]
    expected = (
        (
            (294548, 897, 268, 699, 296412, 167936),
            "0730c1a71c6e6f30ee67a00637ca7704787456c890db18e6b0403068a18f17d4",
            "1395ffbd845769567674f6b517a1915575606ae3377b3cff2d9933e2383fec4f",
            "c05244f21720c60877bc27bc9dabb553c09cec194d951cf71d0435e64729e5dd",
            "27dcd6efdeea1986f3513dea0fc4e46d47d7024864cf1bde2b981fd04c52cd11",
            "a50027c7ae466cbd6dfb726babdddb1b1280a42fffb88054d1c85046eee12b98",
        ),
        (
            (251526, 895, 269, 696, 253386, 167936),
            "09139e258bdd9c1a59b15b561d0d3109e76c1b6770ea9bababc14a208726b29a",
            "114057d41a0510b688f603d5f3575037beceee601f65f694865278ee72e8942c",
            "04b48abf1c6e9db9223d12a39f40fa7517f918d99505b122165cfde6d2ed5fa3",
            "e8b9065a4b256aa22c3f76ccc6bacd3ea299178508cfe5c37ca471baa2e7998d",
            "10bec0234d380988bdab2880749e7ad6d4aef8fbe35c80f5173ee1d5c1c88004",
        ),
    )
    for partition, frozen in zip(projected, expected, strict=True):
        byte_counts, fixed_root, state_root, manifest_root, stream_root, sort_root = frozen
        row = partition.projection_partition_row
        assert (
            row.projected_record_stream_bytes,
            row.projected_chunk_manifest_stream_bytes,
            row.projected_fixed_point_frame_bytes,
            row.projected_signature_manifest_frame_bytes,
            row.projected_partition_payload_bytes,
            row.projected_peak_scratch_bytes,
        ) == byte_counts
        assert partition.fixed_point_record.record_root.hex() == fixed_root
        assert partition.saturation_state_root.hex() == state_root
        assert partition.signature_archive_manifest.manifest_root.hex() == manifest_root
        assert row.diagnostic_stream_commitment.hex() == stream_root
        assert row.external_sort_projection_root.hex() == sort_root
        formal = projection.formal_wire_projection_json_v1(partition)
        assert (
            formal["projected_record_stream_bytes"],
            formal["projected_chunk_manifest_stream_bytes"],
            formal["projected_fixed_point_frame_bytes"],
            formal["projected_signature_manifest_frame_bytes"],
            formal["projected_partition_payload_bytes"],
            formal["projected_peak_formal_scratch_bytes"],
        ) == byte_counts

    closure, archive_result = projection._assemble_formal_aggregate_v1(projected)
    assert len(
        archive.frame_canonical_record_v1(closure.canonical_object())
    ) == 314
    assert archive_result.projected_closure_bundle_frame_bytes == 314
    assert archive_result.projected_archive_payload_bytes_per_endpoint == 550112
    assert archive_result.projected_endpoint_total_output_bytes == 1598688
    assert archive_result.projected_endpoint_peak_scratch_bytes == 167936
    assert (
        closure.bundle_root.hex()
        == "c4e1f2d950f66f5530ae834d2769e6c65decf7f50842fc66f158d11fc8aa9511"
    )
    assert (
        archive_result.diagnostic_root.hex()
        == "b29b3ea0c2e0d5b608394458f74cb29c37e813b3a36d05e39975ba12652a7d00"
    )


def test_cross_language_diagnostic_state_projection_matches_rust_node3_kat(
    node3_same_run_material,
) -> None:
    _measured, _snapshots, projected = node3_same_run_material
    expected = (
        (
            (
                "PROGRAM",
                110,
                13724,
                14604,
                "e78f0b56ecdce07fbff2435068e91d74fdf9984d6f638c75767dc95c16a05caf",
            ),
            (
                "COHORT",
                86,
                15904,
                16592,
                "4f62693738c7a10240454aa99919337870560a09034f4a5031a9f91ce631c0e0",
            ),
            (
                "CLASS",
                40,
                133784,
                134104,
                "4ae8ffd5dcf86bcb6f77bb1c3a6ba2b59108e0287e3a4a2b1152cacb5cf523d8",
            ),
            (
                "COVERAGE",
                846,
                30456,
                37224,
                "be0e67ea10a388afdb3126d47a2fc0d759707347a529da400f5faba5cb449082",
            ),
            "c19d3ccf996292490f0243627f380ded5a0aa743041ee9a50a163c9fc70adbb7",
        ),
        (
            (
                "PROGRAM",
                144,
                18038,
                19190,
                "8839a25d1045130fc8fe80009d208fec559d7277c595053db953b9a25ee81986",
            ),
            (
                "COHORT",
                112,
                20736,
                21632,
                "2824952e491112e2a4ff68bf89cadd88089039256a8695b24ed6568953395240",
            ),
            (
                "CLASS",
                28,
                22898,
                23122,
                "b7a4cc9370a9049ce00cc94807a6138540b8746c4bb56c07fe93b4867b85ee7f",
            ),
            (
                "COVERAGE",
                846,
                30456,
                37224,
                "4d4b7def075077179568181a8bc428ab6fd3fd3ac41715e2cf9c147877f3f975",
            ),
            "0d9b8506d64051c8d89a25975c765630e30b53eeda1d6f5f32529933af246e44",
        ),
    )
    for actual, frozen in zip(projected, expected, strict=True):
        rows = tuple(
            (
                row.stream_kind,
                row.record_count,
                row.payload_bytes,
                row.framed_bytes,
                row.counting_discard_sha256,
            )
            for row in actual.capacity_state_projection.streams
        )
        assert rows == frozen[:4]
        assert (
            actual.capacity_state_projection.combined_state_projection_sha256
            == frozen[4]
        )


def test_full_runner_calls_internal_measurement_once_per_signature(
    node3_same_run_material,
) -> None:
    observed, diagnostic, same_run_snapshots = _mock_full_result(
        node3_same_run_material
    )
    assert observed == [1, 2]
    assert tuple(value.input_signature_id for value in same_run_snapshots) == (1, 2)
    assert diagnostic.terminal_status == full.TERMINAL_SATURATED
    assert diagnostic.q1_state == "NOT_RUN"
    assert diagnostic.q1_gate_count == 0
    assert diagnostic.q1_gate_mask == 0
    assert diagnostic.q1_output_slots == (None,) * 8


def test_full_projection_uses_one_tracker_across_both_partitions(
    node3_same_run_material,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _observed, diagnostic, same_run_snapshots = _mock_full_result(
        node3_same_run_material
    )
    observed_trackers: list[projection._ScratchTrackerV1] = []

    def fake_partition_projection(
        snapshot: object,
        *,
        scratch_root: Path,
        tracker: projection._ScratchTrackerV1,
    ) -> object:
        observed_trackers.append(tracker)
        path = scratch_root / f"partition-{snapshot.input_signature_id}.run"
        with path.open("xb", buffering=0) as target:
            tracker.create(path)
            target.write(b"x" * (7 * snapshot.input_signature_id))
        tracker.reconcile()
        projection._unlink_run_v1(path, tracker)
        return node3_same_run_material[2][snapshot.input_signature_id - 1]

    monkeypatch.setattr(
        projection,
        "_project_same_run_partition_with_tracker_v1",
        fake_partition_projection,
    )
    result = projection.project_full_node6_same_run_v1(
        diagnostic,
        same_run_snapshots,
        scratch_root=tmp_path,
    )
    assert len(observed_trackers) == 2
    assert observed_trackers[0] is observed_trackers[1]
    assert result.actual_preflight_peak_scratch_bytes == 14
    assert result.actual_preflight_scratch_live_bytes_at_terminal == 0
    assert observed_trackers[0].live == {}
    assert tuple(tmp_path.iterdir()) == ()


def test_mocked_full_endpoint_requires_exact_admission_and_writes_canonical_sidecars(
    node3_same_run_material,
    tmp_path: Path,
) -> None:
    _observed, diagnostic, same_run_snapshots = _mock_full_result(
        node3_same_run_material
    )
    closure_bundle, archive_projection_result = (
        projection._assemble_formal_aggregate_v1(node3_same_run_material[2])
    )
    projected = projection.FullNode6ProjectionResultV1(
        capsule_digest=diagnostic.capsule_digest,
        source_commit=diagnostic.source_commit,
        partitions=node3_same_run_material[2],
        closure_bundle=closure_bundle,
        archive_projection_result=archive_projection_result,
        actual_preflight_peak_scratch_bytes=123456,
        actual_preflight_scratch_live_bytes_at_terminal=0,
    )
    admission_path = tmp_path / "admission.json"
    admission_path.write_text(
        json.dumps(asdict(_admission()), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()

    def projected_result(
        _diagnostic: object,
        _snapshots: object,
        *,
        scratch_root: Path,
    ) -> object:
        assert scratch_root.parent == scratch_dir
        assert tuple(scratch_root.iterdir()) == ()
        return projected

    stdout = endpoint.run_endpoint_v1(
        [
            "--action",
            full.ACTION_ID,
            "--admission-json",
            str(admission_path),
            "--capsule-digest",
            _admission().capsule_digest,
            "--source-commit",
            _admission().source_commit,
            "--output-dir",
            str(output_dir),
            "--scratch-dir",
            str(scratch_dir),
        ],
        capacity_runner=lambda _admission: (diagnostic, same_run_snapshots),
        projection_runner=projected_result,
        module_verifier=lambda: (),
    )
    assert tuple(scratch_dir.iterdir()) == ()
    value = json.loads(stdout)
    assert stdout.endswith(b"\n") and stdout.count(b"\n") == 1
    assert value["schema_version"] == endpoint.SCHEMA_VERSION
    assert value["actor_id"] == "PYTHON_ENDPOINT"
    assert value["actual_preflight_peak_scratch_bytes"] == 123456
    assert value["actual_preflight_terminal_live_scratch_bytes"] == 0
    assert "actual_preflight_diagnostic_output_bytes" not in value
    assert value["projected_closure_bundle_frame_bytes"] == 314
    assert value["projected_formal_archive_payload_bytes"] == 550112
    assert value["projected_formal_total_output_bytes"] == 1598688
    assert value["projected_peak_formal_scratch_bytes"] == 167936
    assert value["q1_state"] == "NOT_RUN"
    assert value["q1_gate_count"] == 0
    assert value["q1_gate_mask"] == 0
    assert value["q1_output_slots"] == [None] * 8
    assert [row["input_signature_id"] for row in value["partitions"]] == [1, 2]
    assert all("formal_wire_projection" in row for row in value["partitions"])
    assert all("diagnostic_state_projection" in row for row in value["partitions"])
    assert tuple(sorted(path.name for path in output_dir.iterdir())) == tuple(
        sorted(projection.SIDECAR_PATHS)
    )
    for path in output_dir.iterdir():
        payload = path.read_bytes()
        assert payload.endswith(b"\n") and payload.count(b"\n") == 1
        assert json.dumps(
            json.loads(payload),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode() + b"\n" == payload
    manifest_path = output_dir / "endpoint-manifest-v1.json"
    manifest = json.loads(manifest_path.read_bytes())
    declared_sidecar_bytes = sum(
        row["byte_count"] for row in manifest["rows"]
    ) + manifest_path.stat().st_size
    assert declared_sidecar_bytes == sum(
        path.stat().st_size for path in output_dir.iterdir()
    )
    actual_preflight_diagnostic_output_bytes = len(stdout) + declared_sidecar_bytes
    assert actual_preflight_diagnostic_output_bytes > declared_sidecar_bytes


def test_guard_prefix_is_serialized_without_projection_or_later_measurement(
    node3_same_run_material,
    tmp_path: Path,
) -> None:
    _observed, saturated, _same_run_snapshots = _mock_full_result(
        node3_same_run_material
    )
    guard_row = replace(
        saturated.partitions[1],
        terminal_status=full.TERMINAL_GUARD_HIT,
        resource_guard_id=1,
        resource_guard_name="RAW_OPERATOR_APPLICATIONS",
        traversal_closed=False,
    )
    guard = replace(
        saturated,
        terminal_status=full.TERMINAL_GUARD_HIT,
        partitions=(saturated.partitions[0], guard_row),
    )
    admission_path = tmp_path / "admission.json"
    admission_path.write_text(
        json.dumps(asdict(_admission()), sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()

    def forbidden_projection(
        _diagnostic: object,
        _snapshots: object,
        **_kwargs: object,
    ) -> object:
        raise AssertionError("guard prefix must not start projection or measurement")

    stdout = endpoint.run_endpoint_v1(
        [
            "--action",
            full.ACTION_ID,
            "--admission-json",
            str(admission_path),
            "--capsule-digest",
            _admission().capsule_digest,
            "--source-commit",
            _admission().source_commit,
            "--output-dir",
            str(output_dir),
            "--scratch-dir",
            str(scratch_dir),
        ],
        capacity_runner=lambda _admission: (
            guard,
            (_same_run_snapshots[0], None),
        ),
        projection_runner=forbidden_projection,
        module_verifier=lambda: (),
    )
    value = json.loads(stdout)
    assert value["terminal_status"] == full.TERMINAL_GUARD_HIT
    assert value["actual_preflight_peak_scratch_bytes"] == 0
    assert value["actual_preflight_terminal_live_scratch_bytes"] == 0
    for field in (
        "projected_closure_bundle_frame_bytes",
        "projected_formal_archive_payload_bytes",
        "projected_formal_total_output_bytes",
        "projected_peak_formal_scratch_bytes",
    ):
        assert value[field] is None
    assert len(value["partitions"]) == 2
    assert value["partitions"][0]["terminal_status"] == full.TERMINAL_SATURATED
    assert value["partitions"][1]["resource_guard_id"] == 1
    assert value["partitions"][1]["resource_guard_name"] == "RAW_OPERATOR_APPLICATIONS"
    for row in value["partitions"]:
        assert row["coverage"] is None
        assert row["diagnostic_state_projection"] is None
        assert row["formal_wire_projection"] is None
        assert row["same_run_record_set_root"] is None
        assert row["fixed_point"] is None
    assert tuple(scratch_dir.iterdir()) == ()
    fixed_sidecar = json.loads(
        (output_dir / "fixed-point-summary-v1.json").read_bytes()
    )
    assert all(
        row["fixed_point"] is None
        and row["same_run_record_set_root"] is None
        for row in fixed_sidecar["partitions"]
    )


def test_projection_failure_removes_endpoint_owned_transient_scratch(
    node3_same_run_material,
    tmp_path: Path,
) -> None:
    _observed, diagnostic, same_run_snapshots = _mock_full_result(
        node3_same_run_material
    )
    admission_path = tmp_path / "admission.json"
    admission_path.write_text(
        json.dumps(asdict(_admission()), sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()

    def fail_after_transient_write(
        _diagnostic: object,
        _snapshots: object,
        *,
        scratch_root: Path,
    ) -> object:
        (scratch_root / "partial.run").write_bytes(b"partial")
        raise RuntimeError("injected projection failure")

    with pytest.raises(RuntimeError, match="injected projection failure"):
        endpoint.run_endpoint_v1(
            [
                "--action",
                full.ACTION_ID,
                "--admission-json",
                str(admission_path),
                "--capsule-digest",
                _admission().capsule_digest,
                "--source-commit",
                _admission().source_commit,
                "--output-dir",
                str(output_dir),
                "--scratch-dir",
                str(scratch_dir),
            ],
            capacity_runner=lambda _admission: (diagnostic, same_run_snapshots),
            projection_runner=fail_after_transient_write,
            module_verifier=lambda: (),
        )
    assert tuple(scratch_dir.iterdir()) == ()
    assert tuple(output_dir.iterdir()) == ()


def test_direct_cli_rejects_every_nonfull_or_ambiguous_action() -> None:
    environment = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for arguments in ((), ("--action", "bounded-node3-golden-v1")):
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-B", str(ENTRYPOINT), *arguments],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            timeout=30,
        )
        assert completed.returncode == 1
        assert completed.stderr == b""
        assert completed.stdout.count(b"\n") == 1
        value = json.loads(completed.stdout)
        assert value["error_code"] == "Q1_FULL_CAPACITY_ACTION_NOT_ADMITTED"
        assert value["q1_state"] == "NOT_RUN"
        assert value["q1_gate_count"] == 0
        assert value["q1_formal_roots"] is None


def test_source_closure_is_exact_isolated_and_offline() -> None:
    assert len(endpoint.MODULE_NAMES) == 33
    assert endpoint._recursive_allowlist_replay() == tuple(sorted(endpoint.MODULE_NAMES))
    assert "hegel_machine.__init__" in endpoint.FORBIDDEN_PROJECT_MODULES
    source = ENTRYPOINT.read_text(encoding="utf-8")
    assert "pinned CPython 3.11.15" in source
    assert "/usr/local/bin/python3 -I -S -B" in source
    assert "bounded-node3-golden-v1" not in source
    assert "--admission-json" in source
    assert "--scratch-dir" in source
    assert "FORBIDDEN_NETWORK_IMPORT_ROOTS" in source
    assert "os.environ" not in source
    assert "expanduser" not in source
    assert not hasattr(projection, "_record_streams_from_same_run_snapshot_v1")
