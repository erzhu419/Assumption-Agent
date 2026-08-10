from __future__ import annotations

import ast as python_ast
from fractions import Fraction
from pathlib import Path
from time import monotonic

import pytest

from hegel_machine import phase3_q0_quotient_contract_v1 as contract
from hegel_machine import phase3_q0_quotient_oracle_v1 as oracle
from hegel_machine.phase3_q0_input_adapter_v1 import BOTTOM
from hegel_machine.strict_ast_shrink6_v1 import canonicalize_shrink6_source_ast
from hegel_machine.strict_cbor_v1 import content_hash


@pytest.fixture(scope="module")
def result() -> oracle.Q0OracleEndpointResultV1:
    return oracle.run_q0_python_oracle_v1()


def _ast(source: tuple[object, ...]):
    return canonicalize_shrink6_source_ast(source)


def _cell_values(blob: contract.BehaviorBlobV1) -> tuple[object, ...]:
    return tuple(cell.value if cell.defined else BOTTOM for cell in blob.cells)


def test_frozen_leaf_manifest_and_exact_probe_behaviors() -> None:
    assert tuple(seed.coverage_code for seed in oracle.Q0_FROZEN_LEAF_SEEDS) == tuple(
        range(15)
    )
    actual = tuple(
        (
            oracle.behavior_blob_for_ast_v1(_ast(seed.source_ast)).output_sort_id,
            _cell_values(oracle.behavior_blob_for_ast_v1(_ast(seed.source_ast))),
        )
        for seed in oracle.Q0_FROZEN_LEAF_SEEDS
    )
    bottom4 = (BOTTOM, BOTTOM, BOTTOM, BOTTOM)
    assert actual == (
        (contract.OutputSortId.RATIONAL_VALUE, (Fraction(-1),) * 4),
        (contract.OutputSortId.RATIONAL_VALUE, (Fraction(0),) * 4),
        (contract.OutputSortId.RATIONAL_VALUE, (Fraction(1),) * 4),
        (contract.OutputSortId.BIT, (0, 1, BOTTOM, BOTTOM)),
        (contract.OutputSortId.BIT, (1, 0, BOTTOM, BOTTOM)),
        (contract.OutputSortId.BOUNDED_INT, (5, 8, 4, 4)),
        (
            contract.OutputSortId.RATIONAL_VALUE,
            (BOTTOM, BOTTOM, Fraction(0), Fraction(10)),
        ),
        (contract.OutputSortId.BOUNDED_INT, (BOTTOM, BOTTOM, 0, 4)),
        (
            contract.OutputSortId.RATIONAL_VALUE,
            (BOTTOM, BOTTOM, Fraction(0), Fraction(0)),
        ),
        (contract.OutputSortId.RATIONAL_VALUE, bottom4),
        (contract.OutputSortId.RATIONAL_VALUE, bottom4),
        (contract.OutputSortId.RATIONAL_VALUE, bottom4),
        (contract.OutputSortId.BOUNDED_INT, bottom4),
        (contract.OutputSortId.BOOL, bottom4),
        (contract.OutputSortId.BOOL, bottom4),
    )


def test_recursive_evaluator_strict_bottom_and_exact_operator_semantics() -> None:
    difference = _ast(
        (
            "difference",
            ("bit_to_scalar", ("bit_at", 1)),
            ("scalar_const", 3),
        )
    )
    assert _cell_values(oracle.behavior_blob_for_ast_v1(difference)) == (
        Fraction(1),
        Fraction(0),
        BOTTOM,
        BOTTOM,
    )

    same_sign = _ast(
        (
            "same_sign",
            ("sign", ("scalar_const", 1)),
            ("sign", ("scalar_const", 5)),
        )
    )
    assert _cell_values(oracle.behavior_blob_for_ast_v1(same_sign)) == (False,) * 4

    opposite_sign = _ast(
        (
            "opposite_sign",
            ("sign", ("scalar_const", 1)),
            ("sign", ("scalar_const", 5)),
        )
    )
    assert _cell_values(oracle.behavior_blob_for_ast_v1(opposite_sign)) == (True,) * 4

    approximate = _ast(
        (
            "approx_equal",
            ("scalar_const", 3),
            ("scalar_const", 5),
            2,
        )
    )
    assert _cell_values(oracle.behavior_blob_for_ast_v1(approximate)) == (False,) * 4


def test_bottom_vectors_remain_sort_bound() -> None:
    universe_root = contract.Q0ProbeInputV1().universe_root
    cells = (contract.BehaviorCellV1.bottom(),) * 4
    bool_blob = contract.BehaviorBlobV1(
        contract.Q0_PROBE_INPUT_SIGNATURE_ID,
        universe_root,
        contract.OutputSortId.BOOL,
        cells,
    )
    bit_blob = contract.BehaviorBlobV1(
        contract.Q0_PROBE_INPUT_SIGNATURE_ID,
        universe_root,
        contract.OutputSortId.BIT,
        cells,
    )
    assert bool_blob.canonical_bytes != bit_blob.canonical_bytes
    assert bool_blob.behavior_id != bit_blob.behavior_id


def test_behavior_digest_collision_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    collision = b"\x9a" * 32
    monkeypatch.setattr(contract, "content_hash", lambda _domain, _value: collision)
    accumulator = oracle.QuotientAccumulatorV1()
    assert accumulator.add_ast(_ast(("scalar_const", 1))) == (1, 1, 1)
    with pytest.raises(oracle.Q0OracleError) as caught:
        accumulator.add_ast(_ast(("scalar_const", 3)))
    assert caught.value.code == "FAIL_SHA256_PREIMAGE_COLLISION"


def test_exact_signature_cohort_retains_two_distinct_bool_witnesses() -> None:
    accumulator = oracle.QuotientAccumulatorV1()
    context = _ast(("context_flag", 0))
    task = _ast(("task_flag", 0))
    assert oracle.behavior_blob_for_ast_v1(context).canonical_bytes == (
        oracle.behavior_blob_for_ast_v1(task).canonical_bytes
    )
    assert oracle.future_signature_for_ast_v1(context) == (
        oracle.future_signature_for_ast_v1(task)
    )
    assert accumulator.add_ast(context) == (1, 1, 1)
    # Class count is stable while the future-admissibility state grows.
    assert accumulator.add_ast(task) == (0, 1, 1)
    record = accumulator.records()[0]
    assert tuple(entry.normalization_witness_rank for entry in record.frontier) == (0, 1)
    assert {entry.representative_ast_cbor for entry in record.frontier} == {
        context.cbor_bytes,
        task.cbor_bytes,
    }

    and2 = _ast(("top_level_AND", ("context_flag", 0), ("task_flag", 0)))
    assert and2.cbor_bytes.hex() == "82018204828300040083000500"


def test_exhaustive_syntax_and_direct_quotient_golden_agreement(
    result: oracle.Q0OracleEndpointResultV1,
) -> None:
    assert result.endpoint_status == contract.Q0_ENDPOINT_PASS_STATUS
    assert (
        result.syntax_raw_application_count,
        result.quotient_raw_application_count,
        result.strict_admitted_syntax_application_count,
        result.strict_admitted_quotient_application_count,
        result.rewrite_collapse_syntax_count,
        result.rewrite_collapse_quotient_count,
        result.canonical_syntax_program_count,
        result.behavior_class_count,
        result.frontier_point_count,
        result.maximum_frontier_points_per_class,
        result.syntax_continuation_bank_point_count,
        result.quotient_continuation_bank_point_count,
        result.maximum_syntax_bank_points_per_class,
        result.maximum_quotient_bank_points_per_class,
        result.saturation_round_count,
    ) == (567, 545, 567, 545, 30, 30, 537, 69, 122, 4, 251, 251, 43, 43, 3)
    assert result.syntax_class_archive_root == result.direct_class_archive_root
    assert tuple(record.canonical_bytes for record in result.syntax_class_records) == tuple(
        record.canonical_bytes for record in result.direct_class_records
    )
    assert result.syntax_program_archive_root.hex() == (
        "bd1a59f816bd6648d0dd73b9a1622f2bb88bb9aeca1489a0d876fbc9dbf0c829"
    )
    assert result.syntax_class_archive_root.hex() == (
        "a2f0dacf4524fdb8725d29a2c3883a7ebd78fa686cb2030ac0d0608710176cf1"
    )


def test_fixed_point_requires_empty_queue_and_zero_complete_state_delta(
    result: oracle.Q0OracleEndpointResultV1,
) -> None:
    assert result.work_queue_empty is True
    assert result.zero_delta_full_round is True
    assert result.final_class_delta == 0
    assert result.final_frontier_mutation_delta == 0
    assert result.final_bank_mutation_delta == 0
    assert result.round_deltas[-1].new_canonical_program_count == 0
    assert tuple(
        (
            row.queued_application_count,
            row.new_canonical_program_count,
            row.new_behavior_class_count,
            row.frontier_mutation_count,
            row.bank_mutation_count,
            row.complete_state_changed,
        )
        for row in result.round_deltas
    ) == (
        (163, 148, 25, 46, 82, True),
        (367, 352, 32, 73, 222, True),
        (0, 0, 0, 0, 0, False),
    )
    # The middle round grows frontiers as well as classes; completion is not a
    # class-count-only test, and the final round fingerprints the full state.
    assert any(row.frontier_mutation_count > 0 for row in result.round_deltas[:-1])


def test_all_27_coverage_codes_are_attempted_and_roots_are_stable(
    result: oracle.Q0OracleEndpointResultV1,
) -> None:
    assert tuple(row[0] for row in result.syntax_coverage_records) == contract.Q0_COVERAGE_CODES
    assert tuple(row[0] for row in result.quotient_coverage_records) == contract.Q0_COVERAGE_CODES
    # Sign leaves require depth 1 / two nodes, so same_sign/opposite_sign need
    # five nodes and are structurally unreachable under the four-node Q0
    # projection.  Their registered zero rows are an exhaustion proof, not an
    # omitted operator.  Every eligible application for every other code is
    # strict-admitted.
    assert [row[0] for row in result.syntax_coverage_records if row[1] == 0] == [
        0x2005,
        0x2006,
    ]
    assert [row[0] for row in result.quotient_coverage_records if row[1] == 0] == [
        0x2005,
        0x2006,
    ]
    assert all(row[2] == row[1] for row in result.syntax_coverage_records)
    assert all(row[2] == row[1] for row in result.quotient_coverage_records)
    assert result.syntax_operator_coverage_root.hex() == (
        "6953f39dc97f17288850b524ca8b04dbb2f6ddd3d53eaf4cb8e4e6465bcd840c"
    )
    assert result.quotient_operator_coverage_root.hex() == (
        "a9a0b6fdc97c475323ccae31fba14a6df411307220efd8538c7971fe9c38c1fd"
    )


def test_endpoint_state_is_deterministic_non_authoritative_and_not_dual(
    result: oracle.Q0OracleEndpointResultV1,
) -> None:
    assert len(result.canonical_state_object()) == 43
    assert len(result.canonical_state_bytes) == 589
    assert result.endpoint_state_root == content_hash(
        contract.ENDPOINT_STATE_ROOT_DOMAIN,
        result.canonical_state_object(),
    )
    assert result.endpoint_state_root.hex() == (
        "d33e54dd99e6cbe8aacc541fc0877af9657a553be58523670cce5c474006d4d2"
    )
    assert result.projection_manifest_root.hex() == (
        "2f39aa248f1305eeaf20a724f6d690cf2b13003f86620d09d2753815831f7ad1"
    )
    assert result.semantic_binding_root.hex() == (
        "b7ec5e860a007469b8a1b3930f17c130f59a800d2a832dfd438d18a75538ff99"
    )
    assert result.syntax_state_root.hex() == (
        "7028819d133c4da6071c06a0bfca2d0b91622e106207d0b0f081148f41c0826a"
    )
    assert result.direct_state_root.hex() == (
        "d87ef33d9d7010ded284b55acfa71aab4d7d991e3d7703c30f1db2caf5893933"
    )
    assert result.target_truth_accessed is False
    assert result.split_accessed is False
    assert result.role_evaluation_performed is False
    assert result.formal_roots_generated is False
    assert result.authoritative_claim_allowed is False
    assert "DUAL" not in result.endpoint_status


def test_raw_resource_guard_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contract, "Q0_MAX_RAW_APPLICATIONS", 0)
    engine = oracle._Engine(direct=False, start_time=monotonic())
    seed = oracle.Q0_FROZEN_LEAF_SEEDS[0]
    with pytest.raises(oracle.Q0OracleError) as caught:
        engine._process(
            oracle._Candidate(seed.coverage_code, seed.source_ast, seed.canonical_node)
        )
    assert caught.value.code == "INCONCLUSIVE_RESOURCE_LIMIT"
    assert caught.value.guard_id is contract.Q0ResourceGuardId.RAW_OPERATOR_APPLICATIONS
    assert engine.raw_count == 1
    assert engine.strict_admitted_count == 0
    assert len(engine.programs) == 0
    assert engine.coverage[seed.coverage_code].new_canonical == 0


def test_canonical_and_class_max_plus_one_fail_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = oracle.Q0_FROZEN_LEAF_SEEDS[0]
    candidate = oracle._Candidate(seed.coverage_code, seed.source_ast, seed.canonical_node)
    monkeypatch.setattr(contract, "Q0_MAX_CANONICAL_SYNTAX", 0)
    engine = oracle._Engine(direct=False, start_time=monotonic())
    with pytest.raises(oracle.Q0OracleError) as caught_program:
        engine._process(candidate)
    assert (
        caught_program.value.guard_id
        is contract.Q0ResourceGuardId.CANONICAL_SYNTAX_PROGRAMS
    )
    assert len(engine.programs) == 0
    assert engine.accumulator.class_count == 0
    assert engine.accumulator.frontier_point_count == 0
    assert engine.accumulator.continuation_bank_point_count == 0
    assert engine.coverage[seed.coverage_code].new_canonical == 0

    monkeypatch.setattr(contract, "Q0_MAX_CANONICAL_SYNTAX", 2_000)
    monkeypatch.setattr(contract, "Q0_MAX_BEHAVIOR_CLASSES", 0)
    accumulator = oracle.QuotientAccumulatorV1()
    with pytest.raises(oracle.Q0OracleError) as caught_class:
        accumulator.add_ast(_ast(("scalar_const", 1)))
    assert caught_class.value.guard_id is contract.Q0ResourceGuardId.BEHAVIOR_CLASSES
    assert accumulator.class_count == 0
    assert accumulator.frontier_point_count == 0
    assert accumulator.continuation_bank_point_count == 0
    assert accumulator._digest_preimages == {}


def test_visible_frontier_and_latent_bank_have_independent_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregate = _ast(("aggregate", 0, 3, 0, ()))
    absolute_aggregate = _ast(("absolute", ("aggregate", 0, 3, 0, ())))
    latent_dominated = _ast(
        (
            "absolute",
            (
                "difference",
                ("scalar_const", 3),
                ("aggregate", 0, 3, 0, ()),
            ),
        )
    )

    monkeypatch.setattr(contract, "Q0_MAX_FRONTIER_POINTS", 1)
    visible = oracle.QuotientAccumulatorV1()
    visible.add_ast(aggregate)
    with pytest.raises(oracle.Q0OracleError) as caught_visible:
        visible.add_ast(absolute_aggregate)
    assert caught_visible.value.code == "INCONCLUSIVE_RESOURCE_LIMIT"
    assert caught_visible.value.guard_id is contract.Q0ResourceGuardId.TOTAL_FRONTIER_POINTS
    assert caught_visible.value.detail == "frontier-point guard reached"
    assert visible.frontier_point_count == 1
    assert visible.continuation_bank_point_count == 1

    monkeypatch.setattr(contract, "Q0_MAX_FRONTIER_POINTS", 2)
    monkeypatch.setattr(contract, "Q0_MAX_CONTINUATION_BANK_POINTS", 2)
    bank = oracle.QuotientAccumulatorV1()
    bank.add_ast(aggregate)
    bank.add_ast(absolute_aggregate)
    with pytest.raises(oracle.Q0OracleError) as caught_bank:
        bank.add_ast(latent_dominated)
    assert caught_bank.value.code == "INCONCLUSIVE_RESOURCE_LIMIT"
    assert (
        caught_bank.value.guard_id
        is contract.Q0ResourceGuardId.TOTAL_CONTINUATION_BANK_POINTS
    )
    assert caught_bank.value.detail == "continuation-bank guard reached"
    assert bank.frontier_point_count == 2
    assert bank.continuation_bank_point_count == 2


def test_target_truth_split_and_phase3_dsl_modules_are_not_imported() -> None:
    source_path = Path(oracle.__file__)
    tree = python_ast.parse(source_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in python_ast.walk(tree):
        if isinstance(node, python_ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, python_ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = (
        "phase3_dsl_v1",
        "target",
        "truth",
        "split",
    )
    assert not any(any(token in module for token in forbidden) for module in imported)
    assert not any(
        getattr(value, "__name__", "").endswith("phase3_dsl_v1")
        for value in vars(oracle).values()
    )
