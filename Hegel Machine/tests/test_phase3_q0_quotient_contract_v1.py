from __future__ import annotations

import ast
from fractions import Fraction
from pathlib import Path
import re

import pytest

import hegel_machine.phase3_q0_quotient_contract_v1 as q0
from hegel_machine.phase3_m25_wire_v1 import OBJECT_TAGS
from hegel_machine.phase3_m3_shadow_wire_v1 import SHADOW_OBJECT_TAGS
from hegel_machine.phase3_dsl_v1 import RATIONAL_VALUE_FRACTIONS
from hegel_machine.phase3_q0_input_adapter_v1 import (
    RATIONAL_VALUE_GRID as ADAPTER_RATIONAL_VALUE_GRID,
)
from hegel_machine.strict_ast_shrink6_v1 import (
    canonicalize_shrink6_source_ast,
    decode_shrink6_canonical_ast,
)
from hegel_machine.strict_cbor_v1 import canonical_cbor_encode, content_hash


PROJECT_ROOT = Path(__file__).resolve().parents[1]
Q0_CONTRACT_PATH = (
    PROJECT_ROOT / "src/hegel_machine/phase3_q0_quotient_contract_v1.py"
)
Q0_RUST_ROOT = PROJECT_ROOT / "rust/q0_quotient_oracle"
Q32 = 1 << 32


def _signature(
    *,
    output_sort_id: q0.OutputSortId = q0.OutputSortId.RATIONAL_VALUE,
    ast_depth: int = 0,
    ast_node_count: int = 1,
    scalar_parameter_occurrence_count: int = 0,
    aggregate_leaf_count: int = 0,
    distinct_bit_slot_bitmap: int = 0,
    scope_clause_count: int = 0,
    top_level_clause_count: int = 0,
    old_law_composition_depth: int = 0,
    normalization_profile_id: q0.NormalizationProfileId = (
        q0.NormalizationProfileId.GENERAL
    ),
    mdl_length_q32: int = 8 * Q32,
) -> q0.FutureAdmissibilitySignatureV1:
    return q0.FutureAdmissibilitySignatureV1(
        output_sort_id=output_sort_id,
        ast_depth=ast_depth,
        ast_node_count=ast_node_count,
        scalar_parameter_occurrence_count=scalar_parameter_occurrence_count,
        aggregate_leaf_count=aggregate_leaf_count,
        distinct_bit_slot_bitmap=distinct_bit_slot_bitmap,
        scope_clause_count=scope_clause_count,
        top_level_clause_count=top_level_clause_count,
        old_law_composition_depth=old_law_composition_depth,
        normalization_profile_id=normalization_profile_id,
        mdl_length_q32=mdl_length_q32,
    )


def _entry(
    signature: q0.FutureAdmissibilitySignatureV1,
    ast_cbor: bytes,
) -> q0.FrontierEntryV1:
    replay = decode_shrink6_canonical_ast(ast_cbor)
    return q0.FrontierEntryV1(
        signature=signature,
        normalization_witness_rank=0,
        representative_ast_cbor=ast_cbor,
        representative_ast_hash=replay.digest,
    )


def _derived_entry(source_ast: object) -> q0.FrontierEntryV1:
    ast = canonicalize_shrink6_source_ast(source_ast)
    return _entry(q0.future_signature_from_ast_v1(ast), ast.cbor_bytes)


def q0_ast_by_cbor() -> dict[bytes, object]:
    asts = tuple(
        canonicalize_shrink6_source_ast(source)
        for source in (
            ("scalar_const", 1),
            ("scalar_const", 3),
            ("scalar_const", 5),
            ("bit_at", 0),
            ("bit_at", 1),
            ("set_size",),
            ("context_flag", 0),
            ("context_flag", 1),
            ("task_flag", 0),
            ("task_flag", 1),
        )
    )
    return {ast.cbor_bytes: ast for ast in asts}


Q0_TEST_AST_CBORS = tuple(sorted(q0_ast_by_cbor()))


def test_behavior_cells_have_canonical_typed_encodings_and_explicit_bottom() -> None:
    bool_cell = q0.BehaviorCellV1.exact(True)
    bit_cell = q0.BehaviorCellV1.exact(1)
    bottom = q0.BehaviorCellV1.bottom()
    rational = q0.BehaviorCellV1.exact(Fraction(-2, 3))

    assert bool_cell.canonical_object(q0.OutputSortId.BOOL) == (1, True)
    assert bit_cell.canonical_object(q0.OutputSortId.BIT) == (1, 1)
    assert bottom.canonical_object(q0.OutputSortId.RATIONAL_VALUE) == (0,)
    assert rational.canonical_object(q0.OutputSortId.RATIONAL_VALUE) == (
        1,
        (-2, 3),
    )
    assert canonical_cbor_encode(bool_cell.canonical_object(q0.OutputSortId.BOOL)).hex() == (
        "8201f5"
    )
    assert canonical_cbor_encode(bit_cell.canonical_object(q0.OutputSortId.BIT)).hex() == (
        "820101"
    )
    assert canonical_cbor_encode(bottom.canonical_object(q0.OutputSortId.BOOL)).hex() == (
        "8100"
    )
    assert canonical_cbor_encode(
        rational.canonical_object(q0.OutputSortId.RATIONAL_VALUE)
    ).hex() == "8201822103"

    with pytest.raises(q0.QuotientContractError) as bool_alias:
        bit_cell.canonical_object(q0.OutputSortId.BOOL)
    assert bool_alias.value.code == "REJECT_Q0_BEHAVIOR_CELL"
    with pytest.raises(q0.QuotientContractError) as bit_alias:
        bool_cell.canonical_object(q0.OutputSortId.BIT)
    assert bit_alias.value.code == "REJECT_Q0_BEHAVIOR_CELL"
    with pytest.raises(q0.QuotientContractError) as non_rational:
        q0.BehaviorCellV1.exact(1).canonical_object(
            q0.OutputSortId.RATIONAL_VALUE
        )
    assert non_rational.value.code == "REJECT_Q0_BEHAVIOR_CELL"
    with pytest.raises(q0.QuotientContractError) as outside_grid:
        q0.BehaviorCellV1.exact(Fraction(1, 9)).canonical_object(
            q0.OutputSortId.RATIONAL_VALUE
        )
    assert outside_grid.value.code == "REJECT_Q0_BEHAVIOR_CELL"


def test_behavior_blob_binds_universe_sort_bottom_positions_and_stable_self_id() -> None:
    root = bytes(range(32))
    cells = (
        q0.BehaviorCellV1.exact(False),
        q0.BehaviorCellV1.bottom(),
        q0.BehaviorCellV1.exact(True),
    )
    first = q0.BehaviorBlobV1(0x7001, root, q0.OutputSortId.BOOL, cells)
    replay = q0.BehaviorBlobV1(0x7001, root, q0.OutputSortId.BOOL, cells)

    assert first.canonical_object()[6] == 3
    assert first.canonical_object()[7] == ((1, False), (0,), (1, True))
    assert first.canonical_bytes == replay.canonical_bytes
    assert first.behavior_id == replay.behavior_id
    assert first.behavior_id == content_hash(
        q0.BEHAVIOR_ID_DOMAIN, first.canonical_object()
    )
    assert len(first.behavior_id) == 32

    other_universe = q0.BehaviorBlobV1(
        0x7001, bytes(reversed(root)), q0.OutputSortId.BOOL, cells
    )
    bit_blob = q0.BehaviorBlobV1(
        0x7001,
        root,
        q0.OutputSortId.BIT,
        (
            q0.BehaviorCellV1.exact(0),
            q0.BehaviorCellV1.bottom(),
            q0.BehaviorCellV1.exact(1),
        ),
    )
    other_signature = q0.BehaviorBlobV1(
        0x7002, root, q0.OutputSortId.BOOL, cells
    )
    assert first.behavior_id != other_universe.behavior_id
    assert first.behavior_id != bit_blob.behavior_id
    assert first.behavior_id != other_signature.behavior_id
    assert first.canonical_bytes != bit_blob.canonical_bytes


def test_probe_input_binds_four_ordered_typed_rows_and_composite_root() -> None:
    probe = q0.Q0ProbeInputV1()
    replay = q0.Q0ProbeInputV1()

    assert probe.rows == q0.Q0_PROBE_SOURCE_ROWS
    assert probe.canonical_object()[:5] == (
        1,
        q0.Q0_PROBE_INPUT_TAG,
        q0.PROBE_INPUT_SCHEMA_ID,
        q0.Q0_PROBE_INPUT_SIGNATURE_ID,
        4,
    )
    assert len(probe.observation_environments()) == 4
    assert tuple(
        environment.input_signature_id
        for environment in probe.observation_environments()
    ) == (1, 1, 2, 2)
    assert probe.canonical_bytes == replay.canonical_bytes
    assert probe.universe_root == replay.universe_root
    assert probe.universe_root == content_hash(
        q0.PROBE_UNIVERSE_ROOT_DOMAIN, probe.canonical_object()
    )
    assert len(probe.canonical_bytes) == 172
    assert probe.canonical_bytes.hex() == (
        "860119360656686567656c2d71302d70726f62652d696e7075742f311970010484"
        "8301193401850119340151686567656c2d6f64642d696e7075742f310585000100"
        "01008301193401850119340151686567656c2d6f64642d696e7075742f31088801"
        "000100010001008302193402870119340252686567656c2d73696e6b2d696e7075"
        "742f31000000008302193402870119340252686567656c2d73696e6b2d696e7075"
        "742f3104010203"
    )
    assert probe.universe_root.hex() == (
        "2c960bcc229175afe6d5e106a34410216669bfe66b14d5c85103762c596f4192"
    )

    with pytest.raises(q0.QuotientContractError) as changed:
        q0.Q0ProbeInputV1(probe.rows[::-1])
    assert changed.value.code == "REJECT_Q0_PROBE_INPUT"

    bool_aliased_first_row = (
        (True, probe.rows[0][1], probe.rows[0][2]),
    ) + probe.rows[1:]
    with pytest.raises(q0.QuotientContractError) as bool_alias:
        q0.Q0ProbeInputV1(bool_aliased_first_row)
    assert bool_alias.value.code == "REJECT_Q0_PROBE_INPUT"


def test_rational_value_grid_is_identical_across_q0_and_old_dsl() -> None:
    assert len(q0.RATIONAL_VALUE_GRID) == 663
    assert q0.RATIONAL_VALUE_GRID == ADAPTER_RATIONAL_VALUE_GRID
    assert q0.RATIONAL_VALUE_GRID == RATIONAL_VALUE_FRACTIONS


def test_frontier_entry_self_id_is_stable_and_binds_representative() -> None:
    first = _derived_entry(("scalar_const", 1))
    replay = _derived_entry(("scalar_const", 1))
    other = _derived_entry(("scalar_const", 3))
    signature = first.signature

    assert first.entry_id == replay.entry_id
    assert first.entry_id == content_hash(
        q0.FRONTIER_ENTRY_ID_DOMAIN, first.canonical_object()
    )
    assert len(first.entry_id) == 32
    assert first.entry_id != other.entry_id

    with pytest.raises(q0.QuotientContractError) as invalid_ast:
        q0.FrontierEntryV1(signature, 0, b"not-cbor", bytes(32))
    assert invalid_ast.value.code == "REJECT_Q0_FRONTIER_AST"

    ast = q0_ast_by_cbor()[Q0_TEST_AST_CBORS[0]]
    with pytest.raises(q0.QuotientContractError) as wrong_hash:
        q0.FrontierEntryV1(signature, 0, ast.cbor_bytes, bytes(32))
    assert wrong_hash.value.code == "REJECT_Q0_FRONTIER_AST_HASH"


@pytest.mark.parametrize(
    "source_ast",
    (
        ("sign", ("absolute", ("bit_to_scalar", ("bit_at", 0)))),
        (
            "top_level_AND",
            ("context_flag", 0),
            ("equal_exact", ("scalar_const", 1), ("scalar_const", 3)),
        ),
    ),
)
def test_frontier_entry_rejects_strict_v16_ast_outside_q0_limits(
    source_ast: object,
) -> None:
    ast = canonicalize_shrink6_source_ast(source_ast)
    assert ast.metrics.depth > 2 or ast.metrics.node_count > 4
    with pytest.raises(q0.QuotientContractError) as raised:
        q0.FrontierEntryV1(
            q0.future_signature_from_ast_v1(ast),
            0,
            ast.cbor_bytes,
            ast.digest,
        )
    assert raised.value.code == "REJECT_Q0_PROJECTION_LIMIT"


@pytest.mark.parametrize(
    "source_ast",
    (("bit_at", 7), ("context_flag", 3), ("task_flag", 1)),
)
def test_frontier_entry_rejects_manifest_external_leaf(source_ast: object) -> None:
    ast = canonicalize_shrink6_source_ast(source_ast)
    with pytest.raises(q0.QuotientContractError) as raised:
        q0.FrontierEntryV1(
            q0.future_signature_from_ast_v1(ast),
            0,
            ast.cbor_bytes,
            ast.digest,
        )
    assert raised.value.code == "REJECT_Q0_PROJECTION_GRAMMAR"


def test_signature_rejects_unrepresented_old_law_composition_depth() -> None:
    accepted = _signature(old_law_composition_depth=0)
    assert accepted.canonical_object()[11] == 0

    with pytest.raises(q0.QuotientContractError) as raised:
        _signature(old_law_composition_depth=1)
    assert raised.value.code == "REJECT_Q0_UNREPRESENTED_LAW_COMPOSITION"


def test_dominance_uses_exact_bitmask_subset_not_only_popcount() -> None:
    empty = _signature(distinct_bit_slot_bitmap=0b0000)
    slot_zero = _signature(distinct_bit_slot_bitmap=0b0001)
    slot_one = _signature(distinct_bit_slot_bitmap=0b0010)
    both = _signature(distinct_bit_slot_bitmap=0b0011)

    assert empty.dominates(slot_zero) is True
    assert slot_zero.dominates(both) is True
    assert slot_one.dominates(both) is True
    assert slot_zero.dominates(slot_one) is False
    assert slot_one.dominates(slot_zero) is False
    assert both.dominates(slot_zero) is False
    assert slot_zero.dominates(slot_zero) is False


def test_normalization_profiles_are_incomparable_even_with_better_resources() -> None:
    general = _signature(
        ast_depth=0,
        ast_node_count=1,
        normalization_profile_id=q0.NormalizationProfileId.GENERAL,
        mdl_length_q32=4 * Q32,
    )
    absolute = _signature(
        ast_depth=2,
        ast_node_count=3,
        normalization_profile_id=q0.NormalizationProfileId.ABSOLUTE_ROOT,
        mdl_length_q32=12 * Q32,
    )

    assert general.dominates(absolute) is False
    assert absolute.dominates(general) is False


def test_mdl_is_part_of_dominance_and_can_block_structural_pruning() -> None:
    shorter = _signature(mdl_length_q32=8 * Q32)
    longer = _signature(mdl_length_q32=9 * Q32)
    assert shorter.dominates(longer) is True
    assert longer.dominates(shorter) is False

    structurally_better_but_longer = _signature(
        ast_depth=0,
        ast_node_count=1,
        mdl_length_q32=20 * Q32,
    )
    structurally_worse_but_shorter = _signature(
        ast_depth=1,
        ast_node_count=2,
        mdl_length_q32=10 * Q32,
    )
    assert structurally_better_but_longer.dominates(
        structurally_worse_but_shorter
    ) is False
    assert structurally_worse_but_shorter.dominates(
        structurally_better_but_longer
    ) is False


def test_single_capacity_sort_uses_lexicographically_smallest_ast_cbor() -> None:
    sign_sources = (
        ("sign", ("aggregate", 0, 3, 0, ())),
        ("sign", ("aggregate", 5, 3, 0, ())),
    )
    sign_asts = tuple(canonicalize_shrink6_source_ast(source) for source in sign_sources)
    assert q0.future_signature_from_ast_v1(sign_asts[0]) == q0.future_signature_from_ast_v1(
        sign_asts[1]
    )
    signature = q0.future_signature_from_ast_v1(sign_asts[0])
    smaller_cbor, larger_cbor = sorted(ast.cbor_bytes for ast in sign_asts)
    larger = _entry(signature, larger_cbor)
    smaller = _entry(signature, smaller_cbor)

    forward = q0.pareto_frontier_v1((larger, smaller))
    reverse = q0.pareto_frontier_v1((smaller, larger))
    assert forward == reverse == (smaller,)


def test_identity_sensitive_sorts_retain_two_distinct_ranked_witnesses() -> None:
    assert q0.normalization_witness_capacity_v1(q0.OutputSortId.BOOL) == 2
    assert q0.normalization_witness_capacity_v1(
        q0.OutputSortId.RATIONAL_VALUE
    ) == 2
    for output_sort in (
        q0.OutputSortId.BIT,
        q0.OutputSortId.SIGN,
        q0.OutputSortId.BOUNDED_INT,
    ):
        assert q0.normalization_witness_capacity_v1(output_sort) == 1

    bool_asts = tuple(
        sorted(
            canonicalize_shrink6_source_ast(source).cbor_bytes
            for source in (
                ("context_flag", 0),
                ("task_flag", 0),
            )
        )
    )
    signature = q0.future_signature_from_ast_v1(
        canonicalize_shrink6_source_ast(("context_flag", 0))
    )
    assert signature == q0.future_signature_from_ast_v1(
        canonicalize_shrink6_source_ast(("task_flag", 0))
    )
    frontier = q0.pareto_frontier_v1(
        tuple(_entry(signature, ast_cbor) for ast_cbor in reversed(bool_asts))
    )
    assert tuple(entry.representative_ast_cbor for entry in frontier) == bool_asts[:2]
    assert tuple(entry.normalization_witness_rank for entry in frontier) == (0, 1)
    assert frontier[0].canonical_object()[4] == 0
    assert frontier[1].canonical_object()[4] == 1

    bit_ast = canonicalize_shrink6_source_ast(("bit_at", 0))
    with pytest.raises(q0.QuotientContractError) as invalid_rank:
        q0.FrontierEntryV1(
            _signature(output_sort_id=q0.OutputSortId.BIT),
            1,
            bit_ast.cbor_bytes,
            bit_ast.digest,
        )
    assert invalid_rank.value.code == "REJECT_Q0_FRONTIER_ENTRY"


def test_cohort_dominance_cannot_reduce_required_witness_multiplicity() -> None:
    better_sources = (("context_flag", 0), ("task_flag", 0))
    worse_sources = (
        ("equal_exact", ("aggregate", 0, 0, 0, ()), ("scalar_const", 3)),
        ("less_equal", ("aggregate", 0, 0, 0, ()), ("scalar_const", 3)),
    )
    better_asts = tuple(canonicalize_shrink6_source_ast(source) for source in better_sources)
    worse_asts = tuple(canonicalize_shrink6_source_ast(source) for source in worse_sources)
    better = q0.future_signature_from_ast_v1(better_asts[0])
    worse = q0.future_signature_from_ast_v1(worse_asts[0])
    assert better == q0.future_signature_from_ast_v1(better_asts[1])
    assert worse == q0.future_signature_from_ast_v1(worse_asts[1])
    context0, context1 = (ast.cbor_bytes for ast in better_asts)
    worse0, worse1 = (ast.cbor_bytes for ast in worse_asts)
    assert better.dominates(worse)

    one_better_two_worse = q0.pareto_frontier_v1(
        (
            _entry(better, context0),
            _entry(worse, worse0),
            _entry(worse, worse1),
        )
    )
    assert len(one_better_two_worse) == 3

    two_better_two_worse = q0.pareto_frontier_v1(
        (
            _entry(better, context0),
            _entry(better, context1),
            _entry(worse, worse0),
            _entry(worse, worse1),
        )
    )
    assert len(two_better_two_worse) == 2
    assert all(entry.signature == better for entry in two_better_two_worse)


def test_pareto_frontier_retains_bit_aggregate_tradeoff_and_drops_dominated_scope() -> None:
    bit_representation = _derived_entry(("bit_to_scalar", ("bit_at", 0)))
    aggregate_representation = _derived_entry(("aggregate", 0, 3, 0, ()))
    dominated_scope_variant = _derived_entry(
        ("aggregate", 0, 3, 0, ((0, True),))
    )

    assert bit_representation.signature.dominates(
        aggregate_representation.signature
    ) is False
    assert aggregate_representation.signature.dominates(
        bit_representation.signature
    ) is False
    assert aggregate_representation.signature.dominates(
        dominated_scope_variant.signature
    ) is True

    frontier = q0.pareto_frontier_v1(
        (
            dominated_scope_variant,
            aggregate_representation,
            bit_representation,
        )
    )
    assert {entry.representative_ast_cbor for entry in frontier} == {
        bit_representation.representative_ast_cbor,
        aggregate_representation.representative_ast_cbor,
    }


def test_quotient_class_record_and_archive_bind_behavior_and_frontier() -> None:
    probe = q0.Q0ProbeInputV1()
    behavior_a = q0.BehaviorBlobV1(
        q0.Q0_PROBE_INPUT_SIGNATURE_ID,
        probe.universe_root,
        q0.OutputSortId.RATIONAL_VALUE,
        tuple(q0.BehaviorCellV1.exact(Fraction(0)) for _ in range(4)),
    )
    behavior_b = q0.BehaviorBlobV1(
        q0.Q0_PROBE_INPUT_SIGNATURE_ID,
        probe.universe_root,
        q0.OutputSortId.RATIONAL_VALUE,
        tuple(q0.BehaviorCellV1.exact(Fraction(1)) for _ in range(4)),
    )
    pairs = sorted(
        (
            (behavior_a, (_derived_entry(("scalar_const", 3)),)),
            (behavior_b, (_derived_entry(("scalar_const", 5)),)),
        ),
        key=lambda pair: (pair[0].behavior_id, pair[0].canonical_bytes),
    )
    records = tuple(
        q0.QuotientClassRecordV1(index, behavior, frontier)
        for index, (behavior, frontier) in enumerate(pairs)
    )

    assert records[0].canonical_object()[1:3] == (
        q0.Q0_QUOTIENT_CLASS_TAG,
        q0.QUOTIENT_CLASS_SCHEMA_ID,
    )
    assert records[0].canonical_object()[5] == records[0].behavior.behavior_id
    assert records[0].canonical_object()[6] == 1
    assert records[0].minimum_mdl_length_q32 == 8 * Q32
    assert records[0].record_id == content_hash(
        q0.QUOTIENT_CLASS_RECORD_ID_DOMAIN, records[0].canonical_object()
    )
    root = q0.quotient_class_archive_root_v1(records)
    assert len(root) == 32
    assert root == q0.quotient_class_archive_root_v1(records)

    with pytest.raises(q0.QuotientContractError) as unordered:
        q0.quotient_class_archive_root_v1(records[::-1])
    assert unordered.value.code == "REJECT_Q0_QUOTIENT_ARCHIVE"
    with pytest.raises(q0.QuotientContractError) as duplicate:
        q0.quotient_class_archive_root_v1((records[0], records[0]))
    assert duplicate.value.code == "REJECT_Q0_QUOTIENT_ARCHIVE"

    bit_entry = _derived_entry(("bit_at", 0))
    with pytest.raises(q0.QuotientContractError) as wrong_sort:
        q0.QuotientClassRecordV1(
            0,
            behavior_a,
            (bit_entry,),
        )
    assert wrong_sort.value.code == "REJECT_Q0_QUOTIENT_CLASS"

    with pytest.raises(q0.QuotientContractError) as wrong_behavior:
        q0.QuotientClassRecordV1(
            0,
            behavior_b,
            (_derived_entry(("scalar_const", 3)),),
        )
    assert wrong_behavior.value.code == "REJECT_Q0_FRONTIER_BEHAVIOR_MISMATCH"

    invalid_bindings = (
        q0.BehaviorBlobV1(
            0x7002,
            probe.universe_root,
            q0.OutputSortId.RATIONAL_VALUE,
            behavior_a.cells,
        ),
        q0.BehaviorBlobV1(
            q0.Q0_PROBE_INPUT_SIGNATURE_ID,
            bytes(32),
            q0.OutputSortId.RATIONAL_VALUE,
            behavior_a.cells,
        ),
        q0.BehaviorBlobV1(
            q0.Q0_PROBE_INPUT_SIGNATURE_ID,
            probe.universe_root,
            q0.OutputSortId.RATIONAL_VALUE,
            behavior_a.cells[:3],
        ),
    )
    for invalid_behavior in invalid_bindings:
        with pytest.raises(q0.QuotientContractError) as invalid_binding:
            q0.QuotientClassRecordV1(
                0,
                invalid_behavior,
                (_derived_entry(("scalar_const", 3)),),
            )
        assert invalid_binding.value.code == "REJECT_Q0_BEHAVIOR_BINDING"


def _pass_receipt(**changes: object) -> q0.Q0SaturationReceiptV1:
    state_root = bytes.fromhex("11" * 32)
    fields: dict[str, object] = {
        "terminal_status_id": (
            q0.Q0TerminalStatusId.DUAL_EXHAUSTIVE_ORACLE_EQUIVALENCE_PASS
        ),
        "syntax_raw_operator_application_count": 100,
        "quotient_raw_operator_application_count": 80,
        "canonical_syntax_program_count": 20,
        "behavior_class_count": 10,
        "frontier_point_count": 15,
        "maximum_frontier_points_per_class": 3,
        "saturation_round_count": 4,
        "syntax_program_archive_root": bytes.fromhex("01" * 32),
        "syntax_oracle_class_archive_root": state_root,
        "quotient_engine_class_archive_root": state_root,
        "syntax_operator_coverage_root": bytes.fromhex("02" * 32),
        "quotient_operator_coverage_root": bytes.fromhex("03" * 32),
        "python_implementation_root": bytes.fromhex("04" * 32),
        "rust_implementation_root": bytes.fromhex("05" * 32),
        "python_endpoint_output_root": bytes.fromhex("06" * 32),
        "rust_endpoint_output_root": bytes.fromhex("07" * 32),
        "host_replay_class_archive_root": state_root,
    }
    fields.update(changes)
    return q0.Q0SaturationReceiptV1(**fields)  # type: ignore[arg-type]


def test_dual_saturation_receipt_is_host_only_and_keeps_q1_not_run() -> None:
    receipt = _pass_receipt()
    body = receipt.canonical_object()

    assert len(body) == 40
    assert body[:3] == (
        1,
        q0.Q0_SATURATION_RECEIPT_TAG,
        q0.SATURATION_RECEIPT_SCHEMA_ID,
    )
    assert body[9] == int(
        q0.Q0TerminalStatusId.DUAL_EXHAUSTIVE_ORACLE_EQUIVALENCE_PASS
    )
    assert body[17:22] == (True, True, True, True, True)
    assert body[32:40] == (14, 0x3FFF, 0, None, 0, False, None, False)
    assert len(receipt.receipt_root) == 32
    assert receipt.receipt_root == content_hash(
        q0.SATURATION_RECEIPT_ROOT_DOMAIN, body
    )

    with pytest.raises(q0.QuotientContractError) as not_run:
        _pass_receipt(terminal_status_id=q0.Q0TerminalStatusId.NOT_RUN)
    assert not_run.value.code == "REJECT_Q0_SATURATION_RECEIPT"
    with pytest.raises(q0.QuotientContractError) as disagreement:
        _pass_receipt(quotient_engine_class_archive_root=bytes.fromhex("12" * 32))
    assert disagreement.value.code == "REJECT_Q0_SATURATION_RECEIPT"
    with pytest.raises(q0.QuotientContractError) as guard:
        _pass_receipt(syntax_raw_operator_application_count=5001)
    assert guard.value.code == "REJECT_Q0_SATURATION_RECEIPT"


def test_bool_bit_role_match_is_explicit_and_rejects_other_sorts() -> None:
    assert q0.bool_bit_role_match_v1(
        q0.OutputSortId.BOOL, q0.BehaviorCellV1.exact(False), 0
    )
    assert q0.bool_bit_role_match_v1(
        q0.OutputSortId.BOOL, q0.BehaviorCellV1.exact(True), 1
    )
    assert q0.bool_bit_role_match_v1(
        q0.OutputSortId.BIT, q0.BehaviorCellV1.exact(0), 0
    )
    assert q0.bool_bit_role_match_v1(
        q0.OutputSortId.BIT, q0.BehaviorCellV1.exact(1), 1
    )
    assert not q0.bool_bit_role_match_v1(
        q0.OutputSortId.BOOL, q0.BehaviorCellV1.bottom(), 0
    )
    assert not q0.bool_bit_role_match_v1(
        q0.OutputSortId.BIT, q0.BehaviorCellV1.bottom(), 1
    )
    assert not q0.bool_bit_role_match_v1(
        q0.OutputSortId.SIGN, q0.BehaviorCellV1.exact(1), 1
    )
    assert not q0.bool_bit_role_match_v1(
        q0.OutputSortId.BOUNDED_INT, q0.BehaviorCellV1.exact(1), 1
    )
    assert not q0.bool_bit_role_match_v1(
        q0.OutputSortId.RATIONAL_VALUE,
        q0.BehaviorCellV1.exact(Fraction(1, 1)),
        1,
    )
    assert not q0.bool_bit_role_match_v1(
        q0.OutputSortId.BOOL, q0.BehaviorCellV1.exact(1), 1
    )

    with pytest.raises(q0.QuotientContractError) as raised:
        q0.bool_bit_role_match_v1(
            q0.OutputSortId.BOOL, q0.BehaviorCellV1.exact(True), True
        )
    assert raised.value.code == "REJECT_Q0_TARGET_BIT"


def test_versions_fourteen_gates_and_authority_guards_are_exact() -> None:
    assert q0.DSL_VERSION == "hegel-old-dsl-v1.6.0"
    assert q0.DSL_FREEZE_VERSION == "hegel-freeze-p2b-p3-v1.6.0"
    assert q0.CLOSURE_SEMANTICS_VERSION == "hegel-quotient-closure-v1.0.1"
    assert q0.Q0_FREEZE_VERSION == "hegel-freeze-p3a-q0-v1.0.1"
    assert q0.Q0_QUALIFICATION_ID == (
        "hegel-phase3a-q0-exact-quotient-qualification-v1"
    )
    assert q0.Q0_READINESS_GATE_TOTAL == 14
    assert len(q0.Q0_READINESS_GATES) == 14
    assert len(set(q0.Q0_READINESS_GATES)) == 14
    assert q0.Q0_READINESS_GATES[0] == "NORMATIVE_DIRECTION_BYTES_BOUND"
    assert q0.Q0_READINESS_GATES[-1] == (
        "DUAL_HOST_AGREEMENT_Q1_OUTPUTS_NULL_NOT_RUN"
    )

    assert q0.Q0_EXECUTION_STATE == "NOT_RUN"
    assert q0.Q0_FORMAL_ROOTS is None
    assert q0.Q0_FORMAL_ROOTS_GENERATED is False
    assert q0.Q0_TARGET_TRUTH_ACCESS_ALLOWED is False
    assert q0.Q0_SPLIT_ACCESS_ALLOWED is False
    assert q0.Q0_ROLE_EVALUATION_ALLOWED is False
    assert q0.Q0_OLD_M3_GATE_COUNT_INHERITED == 0
    assert q0.Q0TerminalStatusId.NOT_RUN == 0
    assert q0.Q0TerminalStatusId.DUAL_EXHAUSTIVE_ORACLE_EQUIVALENCE_PASS == 2
    assert len(q0.Q0_COVERAGE_CODES) == 27
    assert len(set(q0.Q0_COVERAGE_CODES)) == 27
    assert q0.Q0_COVERAGE_CODES == tuple(sorted(q0.Q0_COVERAGE_CODES))
    assert q0.Q0_LEAF_COVERAGE_CODES == tuple(range(15))
    assert q0.Q0_UNARY_COVERAGE_CODES == (0x1000, 0x1001, 0x1002, 0x1003)
    assert q0.Q0_BINARY_COVERAGE_CODES == (
        0x2001,
        0x2002,
        0x2003,
        0x2005,
        0x2006,
    )
    assert q0.Q0_APPROX_COVERAGE_CODES == (0x3001, 0x3002)
    assert q0.Q0_AND2_COVERAGE_CODE == 0x4002
    assert q0.Q0_FROZEN_LEAF_CANONICAL_NODES == tuple(
        ast.value[1]
        for ast in (
            canonicalize_shrink6_source_ast(("scalar_const", 1)),
            canonicalize_shrink6_source_ast(("scalar_const", 3)),
            canonicalize_shrink6_source_ast(("scalar_const", 5)),
            canonicalize_shrink6_source_ast(("bit_at", 0)),
            canonicalize_shrink6_source_ast(("bit_at", 1)),
            canonicalize_shrink6_source_ast(("set_size",)),
            canonicalize_shrink6_source_ast(("aggregate", 0, 3, 0, ())),
            canonicalize_shrink6_source_ast(("aggregate", 1, 3, 0, ())),
            canonicalize_shrink6_source_ast(("aggregate", 5, 3, 0, ())),
            canonicalize_shrink6_source_ast(("aggregate", 0, 0, 0, ())),
            canonicalize_shrink6_source_ast(("aggregate", 0, 3, 1, ())),
            canonicalize_shrink6_source_ast(
                ("aggregate", 0, 3, 0, ((0, True),))
            ),
            canonicalize_shrink6_source_ast(("aggregate", 1, 1, 0, ())),
            canonicalize_shrink6_source_ast(("context_flag", 0)),
            canonicalize_shrink6_source_ast(("task_flag", 0)),
        )
    )
    assert q0.Q0_RESOURCE_GUARD_REGISTRY == tuple(
        (index, guard_id.name.encode("ascii"))
        for index, guard_id in enumerate(q0.Q0ResourceGuardId, start=1)
    )
    assert q0.q0_projection_manifest_root_v1().hex() == (
        "2f39aa248f1305eeaf20a724f6d690cf2b13003f86620d09d2753815831f7ad1"
    )
    assert q0.q0_semantic_binding_root_v1().hex() == (
        "b7ec5e860a007469b8a1b3930f17c130f59a800d2a832dfd438d18a75538ff99"
    )


def _assigned_names(node: ast.Assign | ast.AnnAssign) -> tuple[str, ...]:
    if isinstance(node, ast.AnnAssign):
        return (node.target.id,) if isinstance(node.target, ast.Name) else ()
    return tuple(
        target.id for target in node.targets if isinstance(target, ast.Name)
    )


def _python_tag_literals(path: Path) -> set[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if not any("TAG" in name.upper() for name in _assigned_names(node)):
            continue
        if node.value is None:
            continue
        for child in ast.walk(node.value):
            if (
                isinstance(child, ast.Constant)
                and type(child.value) is int
            ):
                values.add(child.value)
    return values


_RUST_TAG_LITERAL = re.compile(
    r"(?m)^\s*(?:pub\s+)?const\s+[A-Z0-9_]*TAG[A-Z0-9_]*\s*"
    r":\s*[^=]+?=\s*(0x[0-9a-fA-F]+|[0-9]+)\s*;"
)


def _rust_tag_literals(path: Path) -> set[int]:
    return {
        int(match.group(1), 0)
        for match in _RUST_TAG_LITERAL.finditer(path.read_text(encoding="utf-8"))
    }


def test_q0_tags_are_unique_and_do_not_collide_with_existing_registries() -> None:
    q0_tags = {
        q0.Q0_BEHAVIOR_BLOB_TAG,
        q0.Q0_CONSTRUCTION_SIGNATURE_TAG,
        q0.Q0_FRONTIER_ENTRY_TAG,
        q0.Q0_QUOTIENT_CLASS_TAG,
        q0.Q0_SATURATION_RECEIPT_TAG,
        q0.Q0_PROBE_INPUT_TAG,
    }
    assert q0_tags == set(range(0x3601, 0x3607))
    assert len(q0_tags) == 6
    assert q0_tags.isdisjoint(OBJECT_TAGS.values())
    assert q0_tags.isdisjoint(SHADOW_OBJECT_TAGS.values())

    existing_python_tags: set[int] = set()
    for path in sorted((PROJECT_ROOT / "src/hegel_machine").glob("*.py")):
        if path == Q0_CONTRACT_PATH:
            continue
        existing_python_tags.update(_python_tag_literals(path))

    existing_rust_tags: set[int] = set()
    for path in sorted((PROJECT_ROOT / "rust").rglob("*.rs")):
        if Q0_RUST_ROOT in path.parents:
            continue
        existing_rust_tags.update(_rust_tag_literals(path))

    assert q0_tags.isdisjoint(existing_python_tags)
    assert q0_tags.isdisjoint(existing_rust_tags)

    q0_rust_tags: set[int] = set()
    for path in sorted(Q0_RUST_ROOT.rglob("*.rs")):
        q0_rust_tags.update(_rust_tag_literals(path))
    if q0_rust_tags:
        assert q0_rust_tags == q0_tags
