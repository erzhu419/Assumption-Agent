"""Target-independent bounded enumerator for ``hegel-old-dsl-v1.6.0``.

The traversal inherits shrink step 5 and stops after the depth-three and
six-node frontiers.  Every generated candidate is checked by the shrink-6
admission boundary before archival.  The inherited conjunction generator
remains AND2-only.
"""

from __future__ import annotations

from typing import Final, NoReturn, Sequence

from . import phase3_m3_bounded_enumerator_v1 as _base
from . import phase3_m3_bounded_enumerator_shrink3_v1 as _shrink3_enumerator
from . import phase3_m3_bounded_enumerator_shrink5_v1 as _parent
from . import strict_ast_shrink2_v1 as _shrink2
from . import strict_ast_shrink6_v1 as _shrink6
from .phase3_m3_shrink6_core_v1 import (
    ACTIVE_FORMAL_BINARY_OPERATOR_IDS,
    ACTIVE_RATIONAL_PARAMETER_IDS,
    ACTIVE_SOURCE_BINARY_OPERATOR_IDS,
    DSL_VERSION,
    FREEZE_VERSION,
    HUMAN_AMENDMENT_ID,
    MAXIMUM_AST_DEPTH,
    MAXIMUM_AST_NODE_COUNT,
    MAX_TOP_LEVEL_CLAUSES,
    MAX_TOTAL_AST_DEPTH,
    MAX_TOTAL_NODE_COUNT,
    PARENT_DSL_VERSION,
    PARENT_FREEZE_VERSION,
    RESERVED_BINARY_OPERATOR_IDS,
    RESERVED_RATIONAL_PARAMETER_IDS,
    SHRINK_STEP_ID,
    SOURCE_ALIAS_BINARY_OPERATOR_IDS,
    TOMBSTONED_AGGREGATE_IDS,
    TOMBSTONED_BINARY_OPERATOR_IDS,
    TOMBSTONED_RATIONAL_PARAMETER_IDS,
)


CANONICAL_PROGRAM_BUDGET: Final = _parent.CANONICAL_PROGRAM_BUDGET
RAW_APPLICATION_CAP: Final = _parent.RAW_APPLICATION_CAP
RECORDS_PER_CHUNK: Final = _parent.RECORDS_PER_CHUNK
SCOPE_EXTENSIONS: Final = _parent.SCOPE_EXTENSIONS
EnumerationBindingsV1 = _parent.EnumerationBindingsV1
BoundedEnumerationError = _parent.BoundedEnumerationError
BoundedEnumerationResultV1 = _parent.BoundedEnumerationResultV1
DUAL_ENUMERATION_QUALIFIED: Final = False
DIAGNOSTIC_EXECUTION_STATE: Final = "NOT_RUN"
DIAGNOSTIC_CLAIM_LEVEL: Final = "NON_FORMAL_DUAL_CHILD_DIAGNOSTIC"
DIAGNOSTIC_REPORT_SCHEMA: Final = (
    "hegel-m3-shrink6-python-closure-enumerator-report/1"
)


def _fail_report(detail: str) -> NoReturn:
    raise BoundedEnumerationError("FAIL_DIAGNOSTIC_REPORT", detail)


class _Shrink6Enumerator(_parent._Shrink5Enumerator):
    """Reuse AND2 traversal while admitting only depth-three programs."""

    def __init__(self, *, raw_cap: int) -> None:
        super().__init__(raw_cap=raw_cap)
        self.buckets = {
            key: value
            for key, value in self.buckets.items()
            if key[1] <= MAX_TOTAL_AST_DEPTH and key[2] <= MAX_TOTAL_NODE_COUNT
        }

    def _admit(
        self,
        expr: _base._strict._Expr,
        key: tuple[int, int, int],
        *,
        known_normal_form: bool = False,
    ) -> None:
        bucket = self._consume(key)
        if _shrink3_enumerator._contains_removed_add(expr):
            _base._fail(
                "FAIL_ENUMERATION_REMOVED_OPERATOR_GENERATED",
                "shrink-6 traversal generated BinaryOperatorId 0/add",
            )
        if not known_normal_form:
            normalized = _shrink2._normalize_child(expr)
            if _base._strict._expr_value(normalized) != _base._strict._expr_value(expr):
                bucket.rewrite_collapses += 1
                return
        try:
            ast = _base._strict._accepted(expr)
            _shrink6.decode_shrink6_canonical_ast(ast.cbor_bytes)
        except _base._strict.StrictAstError as error:
            if error.code == "REJECT_STRUCTURAL_LIMIT":
                bucket.structural_limit_rejections += 1
                return
            _base._fail("FAIL_ENUMERATION_CANONICALIZER", str(error))
        if ast.metrics.output_sort not in _base.OUTPUT_SORT_IDS:
            _base._fail(
                "FAIL_ENUMERATION_SORT",
                f"unregistered output sort {ast.metrics.output_sort}",
            )
        prior = self.seen.get(ast.cbor_bytes)
        if prior is not None:
            if prior.ast.digest != ast.digest:
                _base._fail(
                    "FAIL_AST_HASH_COLLISION",
                    "same bytes carried different AST hashes",
                )
            bucket.syntactic_duplicates += 1
            return
        if (
            ast.metrics.depth != key[1]
            or ast.metrics.node_count != key[2]
            or _base.OUTPUT_SORT_IDS[ast.metrics.output_sort] != key[0]
        ):
            _base._fail(
                "FAIL_ENUMERATION_BUCKET",
                "normal-form construction entered the wrong bucket",
            )
        program = _base._Program(expr, ast)
        self.seen[ast.cbor_bytes] = program
        self.groups.setdefault(
            (ast.metrics.output_sort, ast.metrics.depth, ast.metrics.node_count),
            [],
        ).append(program)

def _witness_status(canonical_budget: int, raw_application_cap: int) -> str:
    if (
        canonical_budget == CANONICAL_PROGRAM_BUDGET
        and raw_application_cap == RAW_APPLICATION_CAP
    ):
        return "DSL_TOO_LARGE"
    return "DIAGNOSTIC_PREFIX_BUDGET_EXCEEDED"


def enumerate_bounded_closure_shrink6_v1(
    bindings: EnumerationBindingsV1,
    *,
    canonical_budget: int = CANONICAL_PROGRAM_BUDGET,
    raw_application_cap: int = RAW_APPLICATION_CAP,
) -> BoundedEnumerationResultV1:
    """Enumerate a target-free child prefix without executing formal M3."""

    if not isinstance(bindings, EnumerationBindingsV1):
        raise TypeError("bindings must be EnumerationBindingsV1")
    if (
        type(canonical_budget) is not int
        or canonical_budget < 1
        or canonical_budget > CANONICAL_PROGRAM_BUDGET
    ):
        raise ValueError(
            f"canonical_budget must be in 1..{CANONICAL_PROGRAM_BUDGET}"
        )
    if (
        type(raw_application_cap) is not int
        or raw_application_cap < 1
        or raw_application_cap > RAW_APPLICATION_CAP
    ):
        raise ValueError(
            f"raw_application_cap must be in 1..{RAW_APPLICATION_CAP}"
        )

    state = _Shrink6Enumerator(raw_cap=raw_application_cap)
    ordered: list[_base._Program] = []
    traversal_complete = False
    stop = False
    sort_names = {value: key for key, value in _base.OUTPUT_SORT_IDS.items()}
    for depth in range(MAX_TOTAL_AST_DEPTH + 1):
        for nodes in range(1, MAX_TOTAL_NODE_COUNT + 1):
            for sort_id in range(1, 6):
                if depth == 0 and nodes == 1:
                    state.leaves(sort_id)
                elif depth >= 1 and nodes >= 2:
                    state.unary(depth, nodes, sort_id)
                    if nodes >= 3:
                        state.binary_and_ternary(depth, nodes, sort_id)
                        if sort_id == _base.OUTPUT_SORT_IDS["Bool"]:
                            state.conjunctions(depth, nodes)
                ordered.extend(
                    sorted(
                        state.groups.get((sort_names[sort_id], depth, nodes), ()),
                        key=lambda program: (
                            program.ast.root_operator_id,
                            program.ast.cbor_bytes,
                        ),
                    )
                )
                if len(ordered) > canonical_budget:
                    stop = True
                    break
            if stop:
                break
        if stop:
            break
    else:
        traversal_complete = True

    prefix: Sequence[_base._Program] = tuple(ordered[:canonical_budget])
    witness = ordered[canonical_budget] if len(ordered) > canonical_budget else None
    if witness is not None:
        status = _witness_status(canonical_budget, raw_application_cap)
    elif traversal_complete:
        status = "COMPLETE"
    else:
        _base._fail(
            "INCONCLUSIVE_BUDGET",
            "enumeration ended without witness or closed frontier",
        )

    records = tuple(
        _base._formal_program_record(index, program, bindings)
        for index, program in enumerate(prefix)
    )
    manifests = _base._chunk_manifests(records)
    bucket_records = _base._bucket_records(state, prefix)
    return BoundedEnumerationResultV1(
        dsl_version=DSL_VERSION,
        closure_status=status,
        raw_operator_application_count=state.raw_count,
        canonical_program_count=len(prefix),
        first_out_of_budget_program_hash=(None if witness is None else witness.ast.digest),
        first_out_of_budget_cbor=(None if witness is None else witness.ast.cbor_bytes),
        canonical_program_records=records,
        program_chunk_manifests=manifests,
        bucket_accounting_records=bucket_records,
        canonical_program_archive_root=_base.rfc6962_root(list(records)),
        program_chunk_manifest_root=_base.rfc6962_root(list(manifests)),
        bucket_accounting_root=_base.rfc6962_root(list(bucket_records)),
        traversal_prefix_complete=witness is not None or traversal_complete,
        authoritative_claim_allowed=False,
    )


def diagnostic_report_shrink6_v1(
    result: BoundedEnumerationResultV1,
    bindings: EnumerationBindingsV1,
    *,
    canonical_budget: int = CANONICAL_PROGRAM_BUDGET,
    raw_application_cap: int = RAW_APPLICATION_CAP,
    loaded_hegel_modules: Sequence[str] = (),
) -> dict[str, object]:
    """Describe a shrink-6 enumeration candidate without formal authority."""

    if not isinstance(result, BoundedEnumerationResultV1):
        raise TypeError("result must be BoundedEnumerationResultV1")
    if not isinstance(bindings, EnumerationBindingsV1):
        raise TypeError("bindings must be EnumerationBindingsV1")
    if result.dsl_version != DSL_VERSION:
        _fail_report("enumeration result carries the wrong child DSL")
    if result.authoritative_claim_allowed:
        _fail_report("diagnostic enumerator attempted an authoritative claim")
    if result.closure_status not in {
        "COMPLETE",
        "DSL_TOO_LARGE",
        "DIAGNOSTIC_PREFIX_BUDGET_EXCEEDED",
    }:
        _fail_report("enumeration result carries an illegal diagnostic status")
    if type(canonical_budget) is not int or not 1 <= canonical_budget <= CANONICAL_PROGRAM_BUDGET:
        raise ValueError("canonical_budget is outside the frozen upper bound")
    if type(raw_application_cap) is not int or not 1 <= raw_application_cap <= RAW_APPLICATION_CAP:
        raise ValueError("raw_application_cap is outside the frozen upper bound")

    has_witness = result.first_out_of_budget_program_hash is not None
    if has_witness != (result.first_out_of_budget_cbor is not None):
        _fail_report("out-of-budget witness hash/CBOR nullability differs")
    full_budget = (
        canonical_budget == CANONICAL_PROGRAM_BUDGET
        and raw_application_cap == RAW_APPLICATION_CAP
    )
    if result.closure_status == "DSL_TOO_LARGE":
        if not full_budget:
            _fail_report("DSL_TOO_LARGE requires both exact frozen budgets")
        if result.canonical_program_count != CANONICAL_PROGRAM_BUDGET:
            _fail_report("DSL_TOO_LARGE requires exactly 50,000 archived programs")
        if not has_witness:
            _fail_report("DSL_TOO_LARGE requires a rank-50,001 witness")
    elif result.closure_status == "DIAGNOSTIC_PREFIX_BUDGET_EXCEEDED":
        if full_budget or not has_witness:
            _fail_report("reduced-budget prefix status is inconsistent")
        if result.canonical_program_count != canonical_budget:
            _fail_report("prefix count differs from its actual budget")
    elif has_witness:
        _fail_report("COMPLETE must not carry an out-of-budget witness")
    if not result.traversal_prefix_complete:
        _fail_report("diagnostic result does not close its traversal boundary")
    expected_bucket_count = 5 * (MAX_TOTAL_AST_DEPTH + 1) * MAX_TOTAL_NODE_COUNT
    if len(result.bucket_accounting_records) != expected_bucket_count:
        _fail_report("diagnostic result carries the wrong six-node bucket lattice")

    loaded = tuple(loaded_hegel_modules)
    if any(type(name) is not str for name in loaded):
        raise TypeError("loaded_hegel_modules must contain only strings")
    if tuple(sorted(set(loaded))) != loaded:
        raise ValueError("loaded_hegel_modules must be sorted and duplicate-free")
    target_free_verified = bool(loaded)
    return {
        "schema_version": DIAGNOSTIC_REPORT_SCHEMA,
        "implementation": "python",
        "claim_level": DIAGNOSTIC_CLAIM_LEVEL,
        "diagnostic_only": True,
        "authoritative_claim_allowed": False,
        "execution_state": DIAGNOSTIC_EXECUTION_STATE,
        "formal_roots_generated": False,
        "formal_roots": None,
        "parent_dsl_version": PARENT_DSL_VERSION,
        "parent_freeze_version": PARENT_FREEZE_VERSION,
        "dsl_version": DSL_VERSION,
        "freeze_version": FREEZE_VERSION,
        "human_amendment_id": HUMAN_AMENDMENT_ID,
        "shrink_step_id": SHRINK_STEP_ID,
        "maximum_ast_depth": MAXIMUM_AST_DEPTH,
        "maximum_ast_node_count": MAXIMUM_AST_NODE_COUNT,
        "maximum_top_level_clauses": MAX_TOP_LEVEL_CLAUSES,
        "formal_bucket_count": expected_bucket_count,
        "and3_generator_attempts_allowed": False,
        "and3_raw_operator_application_count": 0,
        "closure_status": result.closure_status,
        "canonical_program_budget": canonical_budget,
        "raw_operator_application_cap": raw_application_cap,
        "raw_operator_application_count": result.raw_operator_application_count,
        "canonical_program_count": result.canonical_program_count,
        "closure_cardinality_or_null": (
            result.canonical_program_count if result.closure_status == "COMPLETE" else None
        ),
        "frontier_exhausted": result.closure_status == "COMPLETE",
        "all_type_buckets_closed": result.closure_status == "COMPLETE",
        "traversal_prefix_complete": result.traversal_prefix_complete,
        "canonical_program_archive_root": result.canonical_program_archive_root.hex(),
        "program_chunk_manifest_root": result.program_chunk_manifest_root.hex(),
        "bucket_accounting_root": result.bucket_accounting_root.hex(),
        "first_out_of_budget_ordinal_or_null": canonical_budget + 1 if has_witness else None,
        "first_out_of_budget_program_hash_or_null": (
            result.first_out_of_budget_program_hash.hex()
            if result.first_out_of_budget_program_hash is not None
            else None
        ),
        "first_out_of_budget_program_cbor_hex_or_null": (
            result.first_out_of_budget_cbor.hex()
            if result.first_out_of_budget_cbor is not None
            else None
        ),
        "program_record_count": len(result.canonical_program_records),
        "chunk_manifest_count": len(result.program_chunk_manifests),
        "bucket_record_count": len(result.bucket_accounting_records),
        "records_per_chunk": RECORDS_PER_CHUNK,
        "diagnostic_child_dsl_spec_root": bindings.child_dsl_spec_root.hex(),
        "diagnostic_operator_semantics_root": bindings.operator_semantics_root.hex(),
        "diagnostic_identifier_registry_root": bindings.identifier_registry_root.hex(),
        "aliases_excluded_before_count": ["greater_equal", "approx_equal:tolerance=0"],
        "active_aggregate_map_ids": [0, 1, 5],
        "tombstoned_aggregate_map_ids": list(TOMBSTONED_AGGREGATE_IDS),
        "active_rational_parameter_ids": list(ACTIVE_RATIONAL_PARAMETER_IDS),
        "tombstoned_rational_parameter_ids": list(TOMBSTONED_RATIONAL_PARAMETER_IDS),
        "reserved_rational_parameter_ids": list(RESERVED_RATIONAL_PARAMETER_IDS),
        "active_source_binary_operator_ids": list(ACTIVE_SOURCE_BINARY_OPERATOR_IDS),
        "active_formal_canonical_binary_operator_ids": list(ACTIVE_FORMAL_BINARY_OPERATOR_IDS),
        "source_alias_binary_operator_ids": list(SOURCE_ALIAS_BINARY_OPERATOR_IDS),
        "tombstoned_binary_operator_ids": list(TOMBSTONED_BINARY_OPERATOR_IDS),
        "reserved_binary_operator_ids": list(RESERVED_BINARY_OPERATOR_IDS),
        "operator_id_compaction_performed": False,
        "automatic_operator_migration_performed": False,
        "target_roles_evaluated": False,
        "split_material_accessed": False,
        "secrets_accessed": False,
        "loaded_hegel_modules": list(loaded),
        "target_free_isolation_verified": target_free_verified,
        "target_or_split_modules_loaded": False if target_free_verified else None,
    }


__all__ = [
    "BoundedEnumerationError",
    "BoundedEnumerationResultV1",
    "CANONICAL_PROGRAM_BUDGET",
    "DIAGNOSTIC_CLAIM_LEVEL",
    "DIAGNOSTIC_EXECUTION_STATE",
    "DIAGNOSTIC_REPORT_SCHEMA",
    "DUAL_ENUMERATION_QUALIFIED",
    "EnumerationBindingsV1",
    "RAW_APPLICATION_CAP",
    "RECORDS_PER_CHUNK",
    "_Shrink6Enumerator",
    "_witness_status",
    "diagnostic_report_shrink6_v1",
    "enumerate_bounded_closure_shrink6_v1",
]
