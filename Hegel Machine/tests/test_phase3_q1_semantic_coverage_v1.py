from __future__ import annotations

import ast
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
formal = importlib.import_module(
    "hegel_machine.phase3_q1_formal_archive_contract_v1"
)
semantic = importlib.import_module(
    "hegel_machine.phase3_q1_semantic_coverage_v1"
)
snapshot_module = importlib.import_module(
    "hegel_machine.phase3_q1_partition_snapshot_v1"
)
strict_cbor = importlib.import_module("hegel_machine.strict_cbor_v1")


@pytest.fixture(scope="module")
def node3_material():
    limits = capacity.PreflightLimitsV1(maximum_ast_node_count=3)
    snapshots = tuple(
        snapshot_module.build_q1_partition_snapshot_v1(
            input_signature_id,
            limits=limits,
        )
        for input_signature_id in (1, 2)
    )
    archives = tuple(
        semantic.build_q1_semantic_coverage_v1(snapshot)
        for snapshot in snapshots
    )
    return tuple(zip(snapshots, archives, strict=True))


def _test_only_content_fingerprint(archive) -> str:
    digest = sha256()
    for record, preimage in zip(
        archive.coverage_records,
        archive.coverage_preimages,
        strict=True,
    ):
        digest.update(
            strict_cbor.canonical_cbor_encode(
                (
                    record.canonical_object(),
                    preimage.construction_depth,
                    preimage.coverage_code,
                    preimage.eligible_application_keys,
                    preimage.strict_admission_preimages,
                )
            )
        )
    return digest.hexdigest()


def _record_map(archive):
    return {
        (row.construction_depth, row.coverage_code): row
        for row in archive.coverage_records
    }


def _bank_ast_cbors(snapshot) -> tuple[bytes, ...]:
    return tuple(
        sorted(
            representative.canonical_ast_cbor
            for class_row in snapshot.behavior_classes
            for cohort in class_row.cohorts
            for representative in cohort.representatives
        )
    )


def test_node3_odd_and_sink_846_row_goldens(node3_material) -> None:
    expected = (
        (1, 480, 846, 1_048, 1_048, 1_048, 22),
        (2, 85, 846, 1_101, 1_101, 1_101, 26),
    )
    observed = tuple(
        (
            snapshot.input_signature_id,
            snapshot.universe_row_count,
            len(archive.coverage_records),
            archive.eligible_application_count,
            archive.processed_application_count,
            archive.strict_admitted_count,
            archive.rewrite_collapse_count,
        )
        for snapshot, archive in node3_material
    )
    assert observed == expected


def test_all_846_rows_sum_to_snapshot_including_810_leaves(node3_material) -> None:
    for snapshot, archive in node3_material:
        assert len(archive.coverage_preimages) == 846
        assert sum(
            row.eligible_application_count for row in archive.coverage_records
        ) == snapshot.raw_operator_application_count
        assert sum(
            row.processed_application_count for row in archive.coverage_records
        ) == snapshot.raw_operator_application_count
        assert sum(
            row.strict_admitted_count for row in archive.coverage_records
        ) == snapshot.strict_admitted_application_count
        assert sum(
            row.rewrite_collapse_count for row in archive.coverage_records
        ) == snapshot.rewrite_collapse_count
        leaves = archive.coverage_records[:810]
        assert all(
            (
                row.construction_depth,
                row.coverage_code,
                row.eligible_application_count,
                row.processed_application_count,
                row.strict_admitted_count,
                row.unique_canonical_ast_count,
                row.rewrite_collapse_count,
            )
            == (0, index, 1, 1, 1, 1, 0)
            for index, row in enumerate(leaves)
        )


def test_registry_and_archive_order_are_exact(node3_material) -> None:
    registry = formal.expected_coverage_registry_v1()
    assert len(registry) == formal.EXPECTED_COVERAGE_RECORD_COUNT == 846
    for _snapshot, archive in node3_material:
        assert tuple(
            (row.construction_depth, row.coverage_code)
            for row in archive.coverage_records
        ) == registry
        assert tuple(
            (row.construction_depth, row.coverage_code)
            for row in archive.coverage_preimages
        ) == registry
        assert formal.canonical_archive_order_v1(
            archive.coverage_records,
            stream_kind_id=formal.ArchiveStreamKindId.COVERAGE,
        ) == archive.coverage_records


def test_application_and_strict_preimage_roots_replay_exactly(
    node3_material,
) -> None:
    for _snapshot, archive in node3_material:
        for record, preimage in zip(
            archive.coverage_records,
            archive.coverage_preimages,
            strict=True,
        ):
            assert (
                preimage.eligible_application_keys
                == preimage.processed_application_keys
            )
            assert record.eligible_application_root == strict_cbor.rfc6962_root(
                preimage.eligible_application_keys
            )
            assert record.processed_application_root == record.eligible_application_root
            assert record.strict_admission_root == strict_cbor.rfc6962_root(
                preimage.strict_admission_preimages
            )
            assert tuple(
                sorted(row[0] for row in preimage.strict_admission_preimages)
            ) == tuple(
                sorted(
                    formal.semantic_application_id_v1(key)
                    for key in preimage.eligible_application_keys
                )
            )


def test_leaf_codes_bind_manifest_index_with_empty_parameters(node3_material) -> None:
    for snapshot, archive in node3_material:
        candidates = capacity.immutable_candidate_applications_v1(
            _bank_ast_cbors(snapshot),
            limits=snapshot.limits,
        )
        leaves = candidates[:810]
        assert tuple(row.coverage_code for row in leaves) == tuple(range(810))
        assert all(
            row.construction_depth == 0
            and row.operator_parameters == ()
            and row.ordered_child_canonical_ast_cbors == ()
            and row.rewrite_collapsed is False
            for row in leaves
        )
        for candidate, preimage in zip(
            leaves,
            archive.coverage_preimages[:810],
            strict=True,
        ):
            assert len(preimage.eligible_application_keys) == 1
            application_key = preimage.eligible_application_keys[0]
            assert application_key[4:] == (0, candidate.coverage_code, (), ())
            assert preimage.strict_admission_preimages == (
                (
                    formal.semantic_application_id_v1(application_key),
                    candidate.canonical_ast_hash,
                ),
            )


def test_operator_keys_bind_ordered_formal_child_program_ids(node3_material) -> None:
    arities = {
        **{code: 1 for code in (0x1000, 0x1001, 0x1002, 0x1003)},
        **{
            code: 2
            for code in (
                0x2001,
                0x2002,
                0x2003,
                0x2005,
                0x2006,
                0x3001,
                0x3002,
                0x4002,
            )
        },
    }
    for snapshot, archive in node3_material:
        admitted_program_ids = set(
            semantic._formal_program_id_by_ast(snapshot).values()
        )
        for preimage in archive.coverage_preimages[810:]:
            for key in preimage.eligible_application_keys:
                assert key == formal.semantic_application_key_v1(
                    snapshot.input_signature_id,
                    snapshot.universe_root,
                    preimage.construction_depth,
                    preimage.coverage_code,
                    key[6],
                    key[7],
                )
                assert len(key[7]) == arities[preimage.coverage_code]
                assert set(key[7]).issubset(admitted_program_ids)
                assert key[6] == (
                    (preimage.coverage_code - 0x3000,)
                    if preimage.coverage_code in (0x3001, 0x3002)
                    else ()
                )


def test_node3_depth_operator_and_rewrite_goldens(node3_material) -> None:
    expected_depth_totals = (
        {(0, 810), (1, 202), (2, 36), (3, 0)},
        {(0, 810), (1, 249), (2, 42), (3, 0)},
    )
    expected_nonempty_operator_counts = (
        {
            (1, 0x1000): 8,
            (1, 0x1001): 4,
            (1, 0x1002): 9,
            (1, 0x1003): 9,
            (1, 0x2001): 45,
            (1, 0x2002): 24,
            (1, 0x2003): 45,
            (1, 0x3001): 24,
            (1, 0x3002): 24,
            (1, 0x4002): 10,
            (2, 0x1002): 18,
            (2, 0x1003): 18,
        },
        {
            (1, 0x1000): 8,
            (1, 0x1001): 5,
            (1, 0x1002): 11,
            (1, 0x1003): 11,
            (1, 0x2001): 57,
            (1, 0x2002): 30,
            (1, 0x2003): 57,
            (1, 0x3001): 30,
            (1, 0x3002): 30,
            (1, 0x4002): 10,
            (2, 0x1002): 21,
            (2, 0x1003): 21,
        },
    )
    for (_snapshot, archive), depth_expected, operator_expected in zip(
        node3_material,
        expected_depth_totals,
        expected_nonempty_operator_counts,
        strict=True,
    ):
        assert {
            (
                depth,
                sum(
                    row.eligible_application_count
                    for row in archive.coverage_records
                    if row.construction_depth == depth
                ),
            )
            for depth in range(4)
        } == depth_expected
        assert {
            (row.construction_depth, row.coverage_code): row.eligible_application_count
            for row in archive.coverage_records[810:]
            if row.eligible_application_count
        } == operator_expected
    odd_records = _record_map(node3_material[0][1])
    sink_records = _record_map(node3_material[1][1])
    assert {
        key: row.rewrite_collapse_count
        for key, row in odd_records.items()
        if row.rewrite_collapse_count
    } == {(1, 0x1002): 3, (1, 0x2001): 13, (2, 0x1002): 6}
    assert {
        key: row.rewrite_collapse_count
        for key, row in sink_records.items()
        if row.rewrite_collapse_count
    } == {(1, 0x1002): 3, (1, 0x2001): 15, (2, 0x1002): 8}


def test_node3_diagnostic_content_fingerprints(node3_material) -> None:
    assert tuple(
        _test_only_content_fingerprint(archive)
        for _snapshot, archive in node3_material
    ) == (
        "cc38cd4f789566357f1d0767c8def678ff496a174c1cbfcb4dfd036e30cc86f7",
        "7b32cf05a857f9b4d5824e24ef196b18965c1ee0445cbbc0a67e87d99832585d",
    )


def test_candidate_schedule_shuffle_is_neutral(node3_material, monkeypatch) -> None:
    snapshot, expected = node3_material[0]
    original = capacity.immutable_candidate_applications_v1

    def reversed_candidates(*args, **kwargs):
        return tuple(reversed(original(*args, **kwargs)))

    monkeypatch.setattr(
        capacity,
        "immutable_candidate_applications_v1",
        reversed_candidates,
    )
    assert semantic.build_q1_semantic_coverage_v1(snapshot) == expected


def test_record_or_preimage_shuffle_is_rejected(node3_material) -> None:
    _snapshot, archive = node3_material[0]
    with pytest.raises(semantic.Q1SemanticCoverageError) as records_error:
        replace(archive, coverage_records=tuple(reversed(archive.coverage_records)))
    assert records_error.value.code == semantic.REJECT_Q1_SEMANTIC_COVERAGE
    with pytest.raises(semantic.Q1SemanticCoverageError) as preimages_error:
        replace(archive, coverage_preimages=tuple(reversed(archive.coverage_preimages)))
    assert preimages_error.value.code == semantic.REJECT_Q1_SEMANTIC_COVERAGE


def test_root_and_strict_hash_tamper_fail_closed(node3_material) -> None:
    _snapshot, archive = node3_material[0]
    record = archive.coverage_records[0]
    bad_record = replace(
        record,
        eligible_application_root=b"\x00" * 32,
        processed_application_root=b"\x00" * 32,
    )
    with pytest.raises(semantic.Q1SemanticCoverageError) as root_error:
        replace(
            archive,
            coverage_records=(bad_record,) + archive.coverage_records[1:],
        )
    assert root_error.value.code == semantic.REJECT_Q1_SEMANTIC_COVERAGE

    preimage = archive.coverage_preimages[0]
    application_id, _ast_hash = preimage.strict_admission_preimages[0]
    bad_preimage = replace(
        preimage,
        strict_admission_preimages=((application_id, b"\x00" * 32),),
    )
    with pytest.raises(semantic.Q1SemanticCoverageError) as strict_error:
        replace(
            archive,
            coverage_preimages=(bad_preimage,) + archive.coverage_preimages[1:],
        )
    assert strict_error.value.code == semantic.REJECT_Q1_SEMANTIC_COVERAGE


def test_application_id_collision_is_detected(node3_material, monkeypatch) -> None:
    snapshot, _archive = node3_material[0]
    monkeypatch.setattr(
        semantic._formal,
        "semantic_application_id_v1",
        lambda _key: b"\x77" * 32,
    )
    with pytest.raises(semantic.Q1SemanticCoverageError) as caught:
        semantic.build_q1_semantic_coverage_v1(snapshot)
    assert caught.value.code == semantic.FAIL_SHA256_PREIMAGE_COLLISION


def test_rfc6962_same_root_different_preimages_is_a_collision() -> None:
    root = b"\x66" * 32
    seen = {root: semantic.canonical_cbor_encode(((1,),))}
    with pytest.raises(semantic.Q1SemanticCoverageError) as caught:
        semantic._register_rfc6962_preimage_v1(
            seen,
            root=root,
            rows=((2,),),
            label="forced",
        )
    assert caught.value.code == semantic.FAIL_SHA256_PREIMAGE_COLLISION


def test_exact_types_and_no_mutable_public_dictionary(node3_material) -> None:
    _snapshot, archive = node3_material[0]
    with pytest.raises(TypeError):
        semantic.build_q1_semantic_coverage_v1([archive])
    with pytest.raises(semantic.Q1SemanticCoverageError):
        replace(archive, coverage_records=list(archive.coverage_records))

    def assert_no_dict(value: object) -> None:
        assert type(value) is not dict
        if type(value) is tuple:
            for item in value:
                assert_no_dict(item)
        elif is_dataclass(value):
            for field in fields(value):
                assert_no_dict(getattr(value, field.name))

    assert_no_dict(archive)


def test_replay_passes_and_formal_authority_stays_closed(node3_material) -> None:
    for snapshot, archive in node3_material:
        assert semantic.validate_q1_semantic_coverage_v1(archive, snapshot) is None
        assert archive.schema_version == (
            "hegel-phase3a-q05a-semantic-coverage-diagnostic/1"
        )
        assert archive.formal_coverage_archive_root is None
        assert archive.q1_state == "NOT_RUN"
        assert archive.q1_gate_count == 0
        assert archive.q1_gate_mask == 0
        assert archive.target_truth_accessed is False
        assert archive.split_accessed is False
        assert archive.role_evaluation_performed is False


def test_source_import_graph_has_no_target_truth_split_or_q0_dependency() -> None:
    path = Path(semantic.__file__)
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
importlib.import_module('hegel_machine.phase3_q1_semantic_coverage_v1')
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
