"""Target-independent bounded enumerator for ``hegel-old-dsl-v1.2.0``.

The traversal, budgets, record wire, MDL code, buckets, and archive algorithms
are inherited from the qualified shrink-1 engine.  The only generator delta is
the sparse RationalParameter leaf set 1/3/5, and every candidate is checked by
the child-aware shrink-2 normalizer and formal decoder.

This bare engine has no run identity or state-transition authority.  Its
result remains a diagnostic candidate until a new child implementation
qualification, formal-root bridge, execution manifest, and explicit M3 start
have been completed.
"""

from __future__ import annotations

from itertools import product
from typing import Final, NoReturn, Sequence

from . import phase3_m3_bounded_enumerator_v1 as _parent
from . import strict_ast_shrink2_v1 as _shrink2
from .phase3_m3_shrink2_core_v1 import (
    ACTIVE_RATIONAL_PARAMETER_IDS,
    DSL_VERSION,
    FREEZE_VERSION,
    HUMAN_AMENDMENT_ID,
    PARENT_DSL_VERSION,
    PARENT_FREEZE_VERSION,
    SHRINK_STEP_ID,
)


CANONICAL_PROGRAM_BUDGET: Final = _parent.CANONICAL_PROGRAM_BUDGET
RAW_APPLICATION_CAP: Final = _parent.RAW_APPLICATION_CAP
RECORDS_PER_CHUNK: Final = _parent.RECORDS_PER_CHUNK
SCOPE_EXTENSIONS: Final = _parent.SCOPE_EXTENSIONS
EnumerationBindingsV1 = _parent.EnumerationBindingsV1
BoundedEnumerationError = _parent.BoundedEnumerationError
BoundedEnumerationResultV1 = _parent.BoundedEnumerationResultV1
DIAGNOSTIC_CLAIM_LEVEL: Final = "NON_FORMAL_DUAL_CHILD_DIAGNOSTIC"
DIAGNOSTIC_EXECUTION_STATE: Final = "NOT_RUN"
DIAGNOSTIC_REPORT_SCHEMA: Final = (
    "hegel-m3-shrink2-python-closure-enumerator-report/1"
)


def _fail_report(detail: str) -> NoReturn:
    raise BoundedEnumerationError("FAIL_DIAGNOSTIC_REPORT", detail)


class _Shrink2Enumerator(_parent._Enumerator):
    """Reuse the traversal engine while replacing child admission exactly."""

    def _admit(
        self,
        expr: _parent._strict._Expr,
        key: tuple[int, int, int],
        *,
        known_normal_form: bool = False,
    ) -> None:
        bucket = self._consume(key)
        if not known_normal_form:
            normalized = _shrink2._normalize_child(expr)
            if (
                _parent._strict._expr_value(normalized)
                != _parent._strict._expr_value(expr)
            ):
                bucket.rewrite_collapses += 1
                return
        try:
            ast = _parent._strict._accepted(expr)
            _shrink2.decode_shrink2_canonical_ast(ast.cbor_bytes)
        except _parent._strict.StrictAstError as error:
            if error.code == "REJECT_STRUCTURAL_LIMIT":
                bucket.structural_limit_rejections += 1
                return
            _parent._fail("FAIL_ENUMERATION_CANONICALIZER", str(error))
        if ast.metrics.output_sort not in _parent.OUTPUT_SORT_IDS:
            _parent._fail(
                "FAIL_ENUMERATION_SORT",
                f"unregistered output sort {ast.metrics.output_sort}",
            )
        prior = self.seen.get(ast.cbor_bytes)
        if prior is not None:
            if prior.ast.digest != ast.digest:
                _parent._fail(
                    "FAIL_AST_HASH_COLLISION",
                    "same bytes carried different AST hashes",
                )
            bucket.syntactic_duplicates += 1
            return
        if (
            ast.metrics.depth != key[1]
            or ast.metrics.node_count != key[2]
            or _parent.OUTPUT_SORT_IDS[ast.metrics.output_sort] != key[0]
        ):
            _parent._fail(
                "FAIL_ENUMERATION_BUCKET",
                "normal-form construction entered the wrong bucket",
            )
        program = _parent._Program(expr, ast)
        self.seen[ast.cbor_bytes] = program
        self.groups.setdefault(
            (ast.metrics.output_sort, ast.metrics.depth, ast.metrics.node_count), []
        ).append(program)

    def leaves(self, output_sort_id: int) -> None:
        if output_sort_id == 1:
            for context_id in range(4):
                self._admit(
                    _parent._strict._Expr(
                        0, 4, "Bool", parameters=(context_id,)
                    ),
                    (1, 0, 1),
                    known_normal_form=True,
                )
            for task_id in range(2):
                self._admit(
                    _parent._strict._Expr(0, 5, "Bool", parameters=(task_id,)),
                    (1, 0, 1),
                    known_normal_form=True,
                )
        elif output_sort_id == 2:
            for index in range(8):
                self._admit(
                    _parent._strict._Expr(0, 1, "Bit", parameters=(index,)),
                    (2, 0, 1),
                    known_normal_form=True,
                )
        elif output_sort_id == 4:
            self._admit(
                _parent._strict._Expr(0, 2, "BoundedInt"),
                (4, 0, 1),
                known_normal_form=True,
            )
            for scope_id, quantity_id, extension in product(
                range(4), range(2), SCOPE_EXTENSIONS
            ):
                self._admit(
                    _parent._strict._Expr(
                        0,
                        3,
                        "BoundedInt",
                        parameters=(1, scope_id, quantity_id, extension),
                    ),
                    (4, 0, 1),
                    known_normal_form=True,
                )
        elif output_sort_id == 5:
            for index in ACTIVE_RATIONAL_PARAMETER_IDS:
                self._admit(
                    _parent._strict._Expr(
                        0, 0, "RationalValue", parameters=(index,)
                    ),
                    (5, 0, 1),
                    known_normal_form=True,
                )
            for map_id, scope_id, quantity_id, extension in product(
                (0, 5), range(4), range(2), SCOPE_EXTENSIONS
            ):
                self._admit(
                    _parent._strict._Expr(
                        0,
                        3,
                        "RationalValue",
                        parameters=(map_id, scope_id, quantity_id, extension),
                    ),
                    (5, 0, 1),
                    known_normal_form=True,
                )


def enumerate_bounded_closure_shrink2_v1(
    bindings: EnumerationBindingsV1,
    *,
    canonical_budget: int = CANONICAL_PROGRAM_BUDGET,
    raw_application_cap: int = RAW_APPLICATION_CAP,
) -> BoundedEnumerationResultV1:
    """Enumerate the child global prefix without evaluating any target role."""

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
    state = _Shrink2Enumerator(raw_cap=raw_application_cap)
    ordered: list[_parent._Program] = []
    traversal_complete = False
    stop = False
    sort_names = {value: key for key, value in _parent.OUTPUT_SORT_IDS.items()}
    for depth in range(5):
        for nodes in range(1, 8):
            for sort_id in range(1, 6):
                if depth == 0 and nodes == 1:
                    state.leaves(sort_id)
                elif depth >= 1 and nodes >= 2:
                    state.unary(depth, nodes, sort_id)
                    if nodes >= 3:
                        state.binary_and_ternary(depth, nodes, sort_id)
                        if sort_id == 1:
                            state.conjunctions(depth, nodes)
                ordered.extend(
                    sorted(
                        state.groups.get(
                            (sort_names[sort_id], depth, nodes), ()
                        ),
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

    prefix: Sequence[_parent._Program] = tuple(ordered[:canonical_budget])
    witness = ordered[canonical_budget] if len(ordered) > canonical_budget else None
    if witness is not None:
        status = "DSL_TOO_LARGE"
    elif traversal_complete:
        status = "COMPLETE"
    else:
        _parent._fail(
            "INCONCLUSIVE_BUDGET",
            "enumeration ended without witness or closed frontier",
        )
    records = tuple(
        _parent._formal_program_record(index, program, bindings)
        for index, program in enumerate(prefix)
    )
    manifests = _parent._chunk_manifests(records)
    bucket_records = _parent._bucket_records(state, prefix)
    return BoundedEnumerationResultV1(
        dsl_version=DSL_VERSION,
        closure_status=status,
        raw_operator_application_count=state.raw_count,
        canonical_program_count=len(prefix),
        first_out_of_budget_program_hash=(
            None if witness is None else witness.ast.digest
        ),
        first_out_of_budget_cbor=(
            None if witness is None else witness.ast.cbor_bytes
        ),
        canonical_program_records=records,
        program_chunk_manifests=manifests,
        bucket_accounting_records=bucket_records,
        canonical_program_archive_root=_parent.rfc6962_root(list(records)),
        program_chunk_manifest_root=_parent.rfc6962_root(list(manifests)),
        bucket_accounting_root=_parent.rfc6962_root(list(bucket_records)),
        traversal_prefix_complete=witness is not None or traversal_complete,
        authoritative_claim_allowed=False,
    )


def diagnostic_report_shrink2_v1(
    result: BoundedEnumerationResultV1,
    bindings: EnumerationBindingsV1,
    *,
    loaded_hegel_modules: Sequence[str] = (),
) -> dict[str, object]:
    """Describe an enumeration candidate without creating formal M3 state.

    Archive roots in this report bind public diagnostic records only.  They are
    deliberately not child formal roots and have no transition authority.
    """

    if not isinstance(result, BoundedEnumerationResultV1):
        raise TypeError("result must be BoundedEnumerationResultV1")
    if not isinstance(bindings, EnumerationBindingsV1):
        raise TypeError("bindings must be EnumerationBindingsV1")
    if result.dsl_version != DSL_VERSION:
        _fail_report("enumeration result carries the wrong child DSL")
    if result.authoritative_claim_allowed:
        _fail_report("diagnostic enumerator attempted an authoritative claim")
    if result.closure_status not in {"COMPLETE", "DSL_TOO_LARGE"}:
        _fail_report("enumeration result carries an illegal diagnostic status")
    has_witness = result.first_out_of_budget_program_hash is not None
    if has_witness != (result.first_out_of_budget_cbor is not None):
        _fail_report("50,001 witness hash/CBOR nullability differs")
    if result.closure_status == "DSL_TOO_LARGE":
        if result.canonical_program_count != CANONICAL_PROGRAM_BUDGET:
            _fail_report("DSL_TOO_LARGE requires exactly 50,000 archived programs")
        if not has_witness:
            _fail_report("DSL_TOO_LARGE requires a unique 50,001 witness")
    elif has_witness:
        _fail_report("COMPLETE must not carry an out-of-budget witness")
    if not result.traversal_prefix_complete:
        _fail_report("diagnostic result does not close its traversal boundary")

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
        "closure_status": result.closure_status,
        "canonical_program_budget": CANONICAL_PROGRAM_BUDGET,
        "raw_operator_application_cap": RAW_APPLICATION_CAP,
        "raw_operator_application_count": result.raw_operator_application_count,
        "canonical_program_count": result.canonical_program_count,
        "closure_cardinality_or_null": (
            result.canonical_program_count
            if result.closure_status == "COMPLETE"
            else None
        ),
        "frontier_exhausted": result.closure_status == "COMPLETE",
        "all_type_buckets_closed": result.closure_status == "COMPLETE",
        "traversal_prefix_complete": result.traversal_prefix_complete,
        "canonical_program_archive_root": (
            result.canonical_program_archive_root.hex()
        ),
        "program_chunk_manifest_root": result.program_chunk_manifest_root.hex(),
        "bucket_accounting_root": result.bucket_accounting_root.hex(),
        "first_out_of_budget_ordinal_or_null": 50_001 if has_witness else None,
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
        "diagnostic_operator_semantics_root": (
            bindings.operator_semantics_root.hex()
        ),
        "diagnostic_identifier_registry_root": (
            bindings.identifier_registry_root.hex()
        ),
        "target_roles_evaluated": False,
        "split_material_accessed": False,
        "secrets_accessed": False,
        "loaded_hegel_modules": list(loaded),
        "target_free_isolation_verified": target_free_verified,
        "target_or_split_modules_loaded": (
            False if target_free_verified else None
        ),
    }


__all__ = [
    "BoundedEnumerationError",
    "BoundedEnumerationResultV1",
    "CANONICAL_PROGRAM_BUDGET",
    "DIAGNOSTIC_CLAIM_LEVEL",
    "DIAGNOSTIC_EXECUTION_STATE",
    "DIAGNOSTIC_REPORT_SCHEMA",
    "EnumerationBindingsV1",
    "RAW_APPLICATION_CAP",
    "RECORDS_PER_CHUNK",
    "SCOPE_EXTENSIONS",
    "diagnostic_report_shrink2_v1",
    "enumerate_bounded_closure_shrink2_v1",
]
