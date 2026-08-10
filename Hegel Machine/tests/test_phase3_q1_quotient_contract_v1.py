from __future__ import annotations

import importlib
import ast
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

q0 = importlib.import_module("hegel_machine.phase3_q0_quotient_contract_v1")
q1 = importlib.import_module("hegel_machine.phase3_q1_quotient_contract_v1")
strict = importlib.import_module("hegel_machine.strict_ast_shrink6_v1")


@pytest.mark.parametrize(
    "source",
    (
        ("scalar_const", 1),
        ("bit_at", 3),
        ("aggregate", 0, 3, 0, ((0, True),)),
        ("absolute", ("scalar_const", 5)),
        ("difference", ("scalar_const", 5), ("scalar_const", 1)),
        (
            "top_level_AND",
            ("context_flag", 0),
            ("task_flag", 0),
        ),
    ),
)
def test_q1_signature_resources_and_mdl_equal_qualified_q0_semantics(source) -> None:
    ast = strict.canonicalize_shrink6_source_ast(source)
    old = q0.future_signature_from_ast_v1(ast)
    new = q1.future_signature_from_ast_v1(ast)
    assert new.resource_tuple() == (
        int(old.output_sort_id),
        old.ast_depth,
        old.ast_node_count,
        old.scalar_parameter_occurrence_count,
        old.aggregate_leaf_count,
        old.distinct_bit_slot_bitmap,
        old.scope_clause_count,
        old.top_level_clause_count,
        old.old_law_composition_depth,
        int(old.normalization_profile_id),
        old.mdl_length_q32,
    )


def test_q1_dominance_and_capacity_equal_qualified_q0_semantics() -> None:
    better_ast = strict.canonicalize_shrink6_source_ast(("scalar_const", 3))
    worse_ast = strict.canonicalize_shrink6_source_ast(
        ("difference", ("scalar_const", 3), ("scalar_const", 3))
    )
    old_better = q0.future_signature_from_ast_v1(better_ast)
    old_worse = q0.future_signature_from_ast_v1(worse_ast)
    new_better = q1.future_signature_from_ast_v1(better_ast)
    new_worse = q1.future_signature_from_ast_v1(worse_ast)
    assert new_better.dominates(new_worse) == old_better.dominates(old_worse)
    for old_sort, new_sort in zip(q0.OutputSortId, q1.OutputSortId, strict=True):
        assert int(old_sort) == int(new_sort)
        assert q1.normalization_witness_capacity_v1(new_sort) == (
            q0.normalization_witness_capacity_v1(old_sort)
        )


def test_q1_contract_has_no_target_split_role_or_q0_contract_dependency() -> None:
    source = Path(q1.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    forbidden_modules = {
        "phase3_q0_quotient_contract_v1",
        "phase3_dsl_v1",
        "phase3_m25_rows_v1",
    }
    assert imported.isdisjoint(forbidden_modules)
    forbidden_exports = ("truth", "split", "role", "target", "match", "seed")
    assert all(
        not any(token in name.lower() for token in forbidden_exports)
        for name in q1.__all__
    )


def test_q1_signature_rejects_bool_aliases_and_nonzero_old_law_depth() -> None:
    with pytest.raises(q1.Q1QuotientContractError):
        q1.FutureAdmissibilitySignatureV1(
            q1.OutputSortId.BOOL,
            True,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            q1.NormalizationProfileId.GENERAL,
            1,
        )
    with pytest.raises(q1.Q1QuotientContractError):
        q1.FutureAdmissibilitySignatureV1(
            q1.OutputSortId.BOOL,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            q1.NormalizationProfileId.GENERAL,
            1,
        )
