from __future__ import annotations

import ast
from collections import Counter
from dataclasses import fields, is_dataclass, replace
from hashlib import sha256
import importlib
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

capacity = importlib.import_module(
    "hegel_machine.phase3_q1_capacity_preflight_v1"
)
snapshot_module = importlib.import_module(
    "hegel_machine.phase3_q1_partition_snapshot_v1"
)
strict_cbor = importlib.import_module("hegel_machine.strict_cbor_v1")


@pytest.fixture(scope="module")
def node3_snapshots():
    limits = capacity.PreflightLimitsV1(maximum_ast_node_count=3)
    return tuple(
        snapshot_module.build_q1_partition_snapshot_v1(
            input_signature_id,
            limits=limits,
        )
        for input_signature_id in (1, 2)
    )


def _test_only_content_fingerprint(snapshot) -> str:
    digest = sha256()
    for class_row in snapshot.behavior_classes:
        digest.update(len(class_row.behavior_key).to_bytes(4, "big"))
        digest.update(class_row.behavior_key)
        for cohort in class_row.cohorts:
            signature = strict_cbor.canonical_cbor_encode(
                cohort.signature.canonical_object()
            )
            digest.update(len(signature).to_bytes(4, "big"))
            digest.update(signature)
            digest.update(bytes((int(cohort.visible_frontier_member),)))
            for representative in cohort.representatives:
                digest.update(representative.canonical_ast_hash)
    return digest.hexdigest()


def test_node3_odd_and_sink_snapshot_goldens(node3_snapshots) -> None:
    odd, sink = node3_snapshots
    assert (
        odd.input_signature_id,
        odd.universe_row_count,
        odd.behavior_class_count,
        odd.signature_cohort_count,
        odd.continuation_bank_point_count,
        odd.visible_frontier_point_count,
    ) == (1, 480, 40, 86, 110, 59)
    assert (
        sink.input_signature_id,
        sink.universe_row_count,
        sink.behavior_class_count,
        sink.signature_cohort_count,
        sink.continuation_bank_point_count,
        sink.visible_frontier_point_count,
    ) == (2, 85, 28, 112, 144, 84)
    assert Counter(row.output_sort_id for row in odd.behavior_classes) == {
        1: 3,
        2: 8,
        3: 12,
        4: 2,
        5: 15,
    }
    assert Counter(row.output_sort_id for row in sink.behavior_classes) == {
        1: 5,
        2: 1,
        3: 5,
        4: 3,
        5: 14,
    }
    assert _test_only_content_fingerprint(odd) == (
        "779ea49c57457622b0a4c4016ede29edd93be391aac3db3223d47997d4db3f5f"
    )
    assert _test_only_content_fingerprint(sink) == (
        "928feb24a2cb9f24cf72a7e1137b2e1775c6f7c255c2ee6d4a272273332bafd3"
    )


def test_snapshot_contains_complete_bank_and_visible_frontier(node3_snapshots) -> None:
    for snapshot in node3_snapshots:
        cohort_count = sum(len(row.cohorts) for row in snapshot.behavior_classes)
        bank_count = sum(
            len(cohort.representatives)
            for row in snapshot.behavior_classes
            for cohort in row.cohorts
        )
        frontier_count = sum(
            len(row.visible_frontier_representative_hashes)
            for row in snapshot.behavior_classes
        )
        assert cohort_count == snapshot.signature_cohort_count
        assert bank_count == snapshot.continuation_bank_point_count
        assert frontier_count == snapshot.visible_frontier_point_count
        assert bank_count > frontier_count
        assert all(
            cohort.representatives
            for row in snapshot.behavior_classes
            for cohort in row.cohorts
        )
        assert all(
            len(row.behavior_cells) == snapshot.universe_row_count
            for row in snapshot.behavior_classes
        )


def test_snapshot_ordering_barriers_high_water_and_authority(node3_snapshots) -> None:
    for snapshot in node3_snapshots:
        assert tuple(row.behavior_key for row in snapshot.behavior_classes) == tuple(
            sorted(row.behavior_key for row in snapshot.behavior_classes)
        )
        for class_row in snapshot.behavior_classes:
            assert tuple(
                cohort.canonical_signature_key for cohort in class_row.cohorts
            ) == tuple(
                sorted(
                    cohort.canonical_signature_key for cohort in class_row.cohorts
                )
            )
            for cohort in class_row.cohorts:
                assert tuple(
                    item.canonical_ast_cbor for item in cohort.representatives
                ) == tuple(
                    sorted(item.canonical_ast_cbor for item in cohort.representatives)
                )
        assert tuple(row.depth for row in snapshot.depth_barriers) == (0, 1, 2, 3, 4)
        assert snapshot.depth_barriers[-1].barrier_kind == "STRUCTURAL_BOUNDARY"
        assert snapshot.peak_raw_operator_application_count == (
            snapshot.raw_operator_application_count
        )
        assert snapshot.peak_behavior_class_count == snapshot.behavior_class_count
        assert snapshot.peak_continuation_bank_point_count == (
            snapshot.continuation_bank_point_count
        )
        assert snapshot.peak_saturation_round_count == 5
        assert snapshot.diagnostic_only is True
        assert snapshot.q1_state == "NOT_RUN"
        assert snapshot.q1_gate_count == 0
        assert snapshot.q1_gate_mask == 0
        assert snapshot.q1_formal_roots is None
        assert snapshot.q1_receipt is None
        assert snapshot.q2_state == "NOT_RUN"
        assert snapshot.m3_formal_roots is None
        assert snapshot.target_truth_accessed is False
        assert snapshot.split_accessed is False
        assert snapshot.role_evaluation_performed is False
        assert snapshot.outside_certificate_issued is False
        assert snapshot.active_transition_allowed is False


def test_snapshot_full_replay_passes_for_both_signatures(node3_snapshots) -> None:
    for snapshot in node3_snapshots:
        assert snapshot_module.validate_q1_partition_snapshot_v1(snapshot) is None


def _assert_no_mutable_dict(value: object) -> None:
    assert type(value) is not dict
    if type(value) is tuple:
        for item in value:
            _assert_no_mutable_dict(item)
    elif is_dataclass(value):
        for field in fields(value):
            _assert_no_mutable_dict(getattr(value, field.name))


def test_public_snapshot_graph_exposes_no_mutable_dictionary(node3_snapshots) -> None:
    for snapshot in node3_snapshots:
        _assert_no_mutable_dict(snapshot)


def test_ast_hash_tamper_fails_replay(node3_snapshots) -> None:
    snapshot = node3_snapshots[0]
    class_row = snapshot.behavior_classes[0]
    cohort = class_row.cohorts[0]
    representative = cohort.representatives[0]
    bad_representative = replace(representative, canonical_ast_hash=b"\x00" * 32)
    bad_cohort = replace(
        cohort,
        representatives=(bad_representative,) + cohort.representatives[1:],
    )
    bad_class = replace(
        class_row,
        cohorts=(bad_cohort,) + class_row.cohorts[1:],
    )
    bad_snapshot = replace(
        snapshot,
        behavior_classes=(bad_class,) + snapshot.behavior_classes[1:],
    )
    with pytest.raises(snapshot_module.Q1PartitionSnapshotError) as caught:
        snapshot_module.validate_q1_partition_snapshot_v1(bad_snapshot)
    assert caught.value.code == snapshot_module.FAIL_Q1_PARTITION_SNAPSHOT_REPLAY


def test_same_ast_digest_with_different_cbor_preimages_is_a_collision() -> None:
    collision = b"\x5a" * 32
    seen = {collision: b"first-canonical-ast"}
    snapshot_module._register_ast_digest_preimage_v1(
        seen,
        collision,
        b"first-canonical-ast",
    )
    with pytest.raises(snapshot_module.Q1PartitionSnapshotError) as caught:
        snapshot_module._register_ast_digest_preimage_v1(
            seen,
            collision,
            b"second-canonical-ast",
        )
    assert caught.value.code == snapshot_module.FAIL_SHA256_PREIMAGE_COLLISION


def test_ast_cbor_tamper_fails_with_stable_snapshot_error(node3_snapshots) -> None:
    snapshot = node3_snapshots[0]
    class_index, cohort_index = next(
        (class_index, cohort_index)
        for class_index, class_row in enumerate(snapshot.behavior_classes)
        for cohort_index, cohort in enumerate(class_row.cohorts)
        if len(cohort.representatives) == 1
    )
    class_row = snapshot.behavior_classes[class_index]
    cohort = class_row.cohorts[cohort_index]
    bad_representative = replace(
        cohort.representatives[0],
        canonical_ast_cbor=b"\x00",
    )
    bad_cohort = replace(cohort, representatives=(bad_representative,))
    bad_class = replace(
        class_row,
        cohorts=class_row.cohorts[:cohort_index]
        + (bad_cohort,)
        + class_row.cohorts[cohort_index + 1 :],
    )
    bad_snapshot = replace(
        snapshot,
        behavior_classes=snapshot.behavior_classes[:class_index]
        + (bad_class,)
        + snapshot.behavior_classes[class_index + 1 :],
    )
    with pytest.raises(snapshot_module.Q1PartitionSnapshotError) as caught:
        snapshot_module.validate_q1_partition_snapshot_v1(bad_snapshot)
    assert caught.value.code == snapshot_module.FAIL_Q1_PARTITION_SNAPSHOT_REPLAY


def test_behavior_cell_tamper_fails_identity_or_replay(node3_snapshots) -> None:
    snapshot = node3_snapshots[0]
    class_index = next(
        index
        for index, row in enumerate(snapshot.behavior_classes)
        if row.output_sort_id == 1
        and any(cell.boolean_value is not None for cell in row.behavior_cells)
    )
    class_row = snapshot.behavior_classes[class_index]
    cell_index = next(
        index
        for index, cell in enumerate(class_row.behavior_cells)
        if cell.boolean_value is not None
    )
    cell = class_row.behavior_cells[cell_index]
    bad_cell = replace(cell, boolean_value=not cell.boolean_value)
    bad_cells = (
        class_row.behavior_cells[:cell_index]
        + (bad_cell,)
        + class_row.behavior_cells[cell_index + 1 :]
    )
    bad_class = replace(class_row, behavior_cells=bad_cells)
    bad_classes = (
        snapshot.behavior_classes[:class_index]
        + (bad_class,)
        + snapshot.behavior_classes[class_index + 1 :]
    )
    bad_snapshot = replace(snapshot, behavior_classes=bad_classes)
    with pytest.raises(snapshot_module.Q1PartitionSnapshotError) as caught:
        snapshot_module.validate_q1_partition_snapshot_v1(bad_snapshot)
    assert caught.value.code == snapshot_module.FAIL_Q1_PARTITION_SNAPSHOT_REPLAY


def test_frontier_tamper_fails_replay(node3_snapshots) -> None:
    snapshot = node3_snapshots[1]
    class_index = next(
        index
        for index, row in enumerate(snapshot.behavior_classes)
        if row.visible_frontier_representative_hashes
    )
    class_row = snapshot.behavior_classes[class_index]
    hashes = class_row.visible_frontier_representative_hashes
    replacement_hash = b"\x00" * 32
    assert replacement_hash not in hashes
    bad_hashes = tuple(sorted((replacement_hash,) + hashes[1:]))
    bad_class = replace(
        class_row,
        visible_frontier_representative_hashes=bad_hashes,
    )
    bad_classes = (
        snapshot.behavior_classes[:class_index]
        + (bad_class,)
        + snapshot.behavior_classes[class_index + 1 :]
    )
    bad_snapshot = replace(snapshot, behavior_classes=bad_classes)
    with pytest.raises(snapshot_module.Q1PartitionSnapshotError) as caught:
        snapshot_module.validate_q1_partition_snapshot_v1(bad_snapshot)
    assert caught.value.code == snapshot_module.FAIL_Q1_PARTITION_SNAPSHOT_REPLAY


def test_accepted_barrier_stat_tamper_fails_replay(node3_snapshots) -> None:
    snapshot = node3_snapshots[0]
    barrier = snapshot.depth_barriers[1]
    bad_barrier = replace(
        barrier,
        eligible_raw_application_count=barrier.eligible_raw_application_count + 1,
    )
    bad_snapshot = replace(
        snapshot,
        depth_barriers=(
            snapshot.depth_barriers[:1]
            + (bad_barrier,)
            + snapshot.depth_barriers[2:]
        ),
    )
    with pytest.raises(snapshot_module.Q1PartitionSnapshotError) as caught:
        snapshot_module.validate_q1_partition_snapshot_v1(bad_snapshot)
    assert caught.value.code == snapshot_module.FAIL_Q1_PARTITION_SNAPSHOT_REPLAY


def test_snapshot_container_types_are_exact(node3_snapshots) -> None:
    snapshot = node3_snapshots[0]
    with pytest.raises(snapshot_module.Q1PartitionSnapshotError):
        replace(snapshot, behavior_classes=list(snapshot.behavior_classes))
    with pytest.raises(snapshot_module.Q1PartitionSnapshotError):
        snapshot_module.Q1BehaviorCellSnapshotV1(cell_tag=True)
    with pytest.raises(snapshot_module.Q1PartitionSnapshotError):
        replace(snapshot, behavior_classes=tuple(reversed(snapshot.behavior_classes)))


def test_resource_limited_partition_has_no_final_snapshot() -> None:
    limits = capacity.PreflightLimitsV1(
        maximum_ast_node_count=3,
        maximum_behavior_classes=1,
    )
    with pytest.raises(snapshot_module.Q1PartitionSnapshotError) as caught:
        snapshot_module.build_q1_partition_snapshot_v1(1, limits=limits)
    assert caught.value.code == (
        snapshot_module.REJECT_Q1_PARTITION_SNAPSHOT_INCOMPLETE
    )


def test_full_or_oversized_snapshot_cannot_bypass_admission() -> None:
    for limits in (
        None,
        capacity.PreflightLimitsV1(maximum_ast_node_count=4),
        capacity.PreflightLimitsV1(),
    ):
        with pytest.raises(snapshot_module.Q1PartitionSnapshotError) as caught:
            snapshot_module.build_q1_partition_snapshot_v1(1, limits=limits)
        assert caught.value.code == (
            snapshot_module.REJECT_Q1_LOCAL_SNAPSHOT_SCOPE_NOT_AUTHORIZED
        )


def test_snapshot_source_has_no_target_truth_split_or_role_match_import() -> None:
    path = Path(snapshot_module.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imported.add(node.module)
            imported.update(alias.name for alias in node.names)
    forbidden = {
        "phase3_dsl_v1",
        "phase3_m25_rows_v1",
        "phase3_m25_split_v1",
        "phase3_m25_formal_static_basis_v1",
        "phase3_q0_quotient_contract_v1",
    }
    assert imported.isdisjoint(forbidden)


def test_empty_package_bootstrap_loads_no_forbidden_module() -> None:
    package_path = SRC / "hegel_machine"
    code = f"""
import importlib
import sys
import types
package = types.ModuleType('hegel_machine')
package.__path__ = [{str(package_path)!r}]
package.__package__ = 'hegel_machine'
sys.modules['hegel_machine'] = package
importlib.import_module('hegel_machine.phase3_q1_partition_snapshot_v1')
forbidden = (
    'hegel_machine.phase3_dsl_v1',
    'hegel_machine.phase3_m25_rows_v1',
    'hegel_machine.phase3_m25_split_v1',
    'hegel_machine.phase3_m25_formal_static_basis_v1',
    'hegel_machine.phase3_q0_quotient_contract_v1',
)
loaded = [name for name in forbidden if name in sys.modules]
if loaded:
    raise SystemExit('forbidden modules loaded: ' + ','.join(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
