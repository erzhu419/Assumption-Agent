"""Bounded canonical-closure enumerator for ``hegel-old-dsl-v1.1.0``.

This is the Python reference implementation of the target-independent M3
canonical-program pass.  It deliberately has no target evaluator and no
access to split material.  Formal ``DSL_TOO_LARGE`` output is possible only
with the frozen 50,000/5,000,000 budgets and after a complete traversal bucket
has exposed canonical program 50,001.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations, combinations_with_replacement, product
from typing import Final, Iterator, NoReturn, Sequence

from . import strict_ast_v1 as _strict
from .phase3_m3_record_wire_v1 import build_m3_record_object_v1
from .phase3_m3_shrink1_core_v1 import DSL_VERSION
from .strict_ast_shrink1_v1 import decode_shrink1_canonical_ast
from .strict_cbor_v1 import canonical_cbor_encode, rfc6962_root


CANONICAL_PROGRAM_BUDGET: Final = 50_000
RAW_APPLICATION_CAP: Final = 5_000_000
RECORDS_PER_CHUNK: Final = 4096
Q32_SCALE: Final = 1 << 32
OUTPUT_SORT_IDS: Final = {
    "Bool": 1,
    "Bit": 2,
    "Sign": 3,
    "BoundedInt": 4,
    "RationalValue": 5,
}
CHUNK_BLOB_DOMAIN: Final = b"HEGEL/CHUNK_BLOB/V1"


class BoundedEnumerationError(RuntimeError):
    """Stable fail-closed enumeration failure."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise BoundedEnumerationError(code, detail)


def _root(value: bytes, name: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        _fail("FAIL_ENUMERATION_BINDING", f"{name} must be exactly 32 bytes")
    return value


@dataclass(frozen=True, slots=True)
class EnumerationBindingsV1:
    child_dsl_spec_root: bytes
    operator_semantics_root: bytes
    identifier_registry_root: bytes

    def __post_init__(self) -> None:
        _root(self.child_dsl_spec_root, "child_dsl_spec_root")
        _root(self.operator_semantics_root, "operator_semantics_root")
        _root(self.identifier_registry_root, "identifier_registry_root")


@dataclass(frozen=True, slots=True)
class _Program:
    expr: _strict._Expr
    ast: _strict.CanonicalAst

    @property
    def sort_id(self) -> int:
        try:
            return OUTPUT_SORT_IDS[self.ast.metrics.output_sort]
        except KeyError as error:
            _fail("FAIL_ENUMERATION_SORT", str(error))

    @property
    def structural_key(self) -> tuple[int, int, int]:
        return (self.sort_id, self.ast.metrics.depth, self.ast.metrics.node_count)

    @property
    def global_key(self) -> tuple[int, int, int, int, bytes]:
        return (
            self.ast.metrics.depth,
            self.ast.metrics.node_count,
            self.sort_id,
            self.ast.root_operator_id,
            self.ast.cbor_bytes,
        )

    @property
    def commutative_key(self) -> tuple[bytes, bytes]:
        node = canonical_cbor_encode(_strict._expr_value(self.expr))
        return sha256(node).digest(), node


@dataclass(slots=True)
class _Bucket:
    raw_operator_applications: int = 0
    syntactic_duplicates: int = 0
    type_rejections: int = 0
    structural_limit_rejections: int = 0
    rewrite_collapses: int = 0


@dataclass(frozen=True, slots=True)
class BoundedEnumerationResultV1:
    dsl_version: str
    closure_status: str
    raw_operator_application_count: int
    canonical_program_count: int
    first_out_of_budget_program_hash: bytes | None
    first_out_of_budget_cbor: bytes | None
    canonical_program_records: tuple[tuple[object, ...], ...]
    program_chunk_manifests: tuple[tuple[object, ...], ...]
    bucket_accounting_records: tuple[tuple[object, ...], ...]
    canonical_program_archive_root: bytes
    program_chunk_manifest_root: bytes
    bucket_accounting_root: bytes
    traversal_prefix_complete: bool
    authoritative_claim_allowed: bool


def canonical_scope_extensions_v1() -> tuple[tuple[tuple[int, bool], ...], ...]:
    """Return the 33 exact, sorted, duplicate-free scope extensions."""

    rows: list[tuple[tuple[int, bool], ...]] = [()]
    rows.extend(((context, expected),) for context in range(4) for expected in (False, True))
    rows.extend(
        ((left, left_value), (right, right_value))
        for left, right in combinations(range(4), 2)
        for left_value, right_value in product((False, True), repeat=2)
    )
    result = tuple(rows)
    if len(result) != 33 or len(set(result)) != 33:
        _fail("FAIL_SCOPE_EXTENSION_FREEZE", "scope-extension catalog drift")
    return result


SCOPE_EXTENSIONS: Final = canonical_scope_extensions_v1()


def _elias_delta_length(one_based_index: int) -> int:
    if type(one_based_index) is not int or one_based_index < 1:
        _fail("FAIL_PROGRAM_MDL_LENGTH_UNAVAILABLE", "invalid registry index")
    floor = one_based_index.bit_length() - 1
    return floor + 2 * ((floor + 1).bit_length() - 1) + 1


def _scope_length(extension: Sequence[object]) -> int:
    count = len(extension)
    if count not in {0, 1, 2}:
        _fail("FAIL_PROGRAM_MDL_LENGTH_UNAVAILABLE", "invalid scope clause count")
    return (1 if count == 0 else 2) + 3 * count


def _node_mdl_bits(node: object) -> int:
    if not isinstance(node, tuple) or not node or type(node[0]) is not int:
        _fail("FAIL_PROGRAM_MDL_LENGTH_UNAVAILABLE", "malformed canonical node")
    tag = node[0]
    if tag == 0:
        leaf = node[1]
        if leaf == 0:
            return 2 + 3 + 3
        if leaf == 1:
            return 2 + 3 + _elias_delta_length(int(node[2]) + 1)
        if leaf == 2:
            return 2 + 3
        if leaf == 3:
            return 2 + 3 + 3 + 2 + 1 + _scope_length(node[5])  # type: ignore[arg-type]
        if leaf == 4:
            return 2 + 3 + _elias_delta_length(int(node[2]) + 1)
        if leaf == 5:
            return 2 + 3 + _elias_delta_length(int(node[2]) + 1)
        _fail("FAIL_PROGRAM_MDL_LENGTH_UNAVAILABLE", "old DSL contains unknown leaf")
    if tag == 1 and len(node) == 3:
        return 2 + 2 + _node_mdl_bits(node[2])
    if tag == 2 and len(node) == 4:
        return 2 + 3 + _node_mdl_bits(node[2]) + _node_mdl_bits(node[3])
    if tag == 3 and len(node) == 5:
        return 3 + 1 + _node_mdl_bits(node[2]) + _node_mdl_bits(node[3]) + 2
    if tag == 4 and len(node) == 2 and isinstance(node[1], tuple):
        count = len(node[1])
        shape = {1: 4, 2: 5, 3: 6}.get(count)
        if shape is None:
            _fail("FAIL_PROGRAM_MDL_LENGTH_UNAVAILABLE", "invalid conjunction arity")
        return shape + sum(_node_mdl_bits(child) for child in node[1])
    _fail("FAIL_PROGRAM_MDL_LENGTH_UNAVAILABLE", "unknown canonical node")


def program_mdl_length_q32(ast: _strict.CanonicalAst) -> int:
    """Return exact fixed-prefix old-program length in unsigned Q32 bits."""

    if not isinstance(ast, _strict.CanonicalAst) or ast.value[0] != 1:
        _fail("FAIL_PROGRAM_MDL_LENGTH_UNAVAILABLE", "CanonicalAstV1 required")
    return _node_mdl_bits(ast.value[1]) * Q32_SCALE


def _blob_hash(blob: bytes) -> bytes:
    return sha256(CHUNK_BLOB_DOMAIN + b"\x00" + blob).digest()


def _formal_program_record(
    index: int, program: _Program, bindings: EnumerationBindingsV1
) -> tuple[object, ...]:
    return build_m3_record_object_v1(
        "CanonicalProgramRecordV2",
        {
            "program_index": index,
            "canonical_ast_cbor_bytes": program.ast.cbor_bytes,
            "canonical_ast_hash": program.ast.digest,
            "output_sort_id": program.sort_id,
            "ast_depth": program.ast.metrics.depth,
            "ast_node_count": program.ast.metrics.node_count,
            "distinct_bit_slot_count": len(program.ast.metrics.distinct_bit_slots),
            "program_mdl_length_q32": program_mdl_length_q32(program.ast),
            "child_dsl_spec_root": bindings.child_dsl_spec_root,
            "operator_semantics_root": bindings.operator_semantics_root,
            "identifier_registry_root": bindings.identifier_registry_root,
        },
    )


def _chunk_manifests(records: Sequence[tuple[object, ...]]) -> tuple[tuple[object, ...], ...]:
    manifests: list[tuple[object, ...]] = []
    for chunk_index, first in enumerate(range(0, len(records), RECORDS_PER_CHUNK)):
        chunk = tuple(records[first : first + RECORDS_PER_CHUNK])
        encoded = tuple(canonical_cbor_encode(record) for record in chunk)
        blob = b"".join(len(item).to_bytes(4, "big") + item for item in encoded)
        manifests.append(
            build_m3_record_object_v1(
                "ProgramChunkManifestV2",
                {
                    "chunk_index": chunk_index,
                    "first_program_index": first,
                    "last_program_index": first + len(chunk) - 1,
                    "record_count": len(chunk),
                    "canonical_program_record_subtree_root": rfc6962_root(list(chunk)),
                    "compressed_program_blob_hash": _blob_hash(blob),
                    "uncompressed_program_byte_length": len(blob),
                },
            )
        )
    return tuple(manifests)


class _Enumerator:
    def __init__(self, *, raw_cap: int) -> None:
        self.raw_cap = raw_cap
        self.raw_count = 0
        self.buckets = {
            (sort_id, depth, nodes): _Bucket()
            for sort_id in range(1, 6)
            for depth in range(5)
            for nodes in range(1, 8)
        }
        self.groups: dict[tuple[str, int, int], list[_Program]] = {}
        self.seen: dict[bytes, _Program] = {}

    def _consume(self, key: tuple[int, int, int]) -> _Bucket:
        if self.raw_count >= self.raw_cap:
            _fail("INCONCLUSIVE_BUDGET", "raw operator-application cap reached before a closed traversal bucket")
        self.raw_count += 1
        bucket = self.buckets[key]
        bucket.raw_operator_applications += 1
        return bucket

    def _admit(
        self,
        expr: _strict._Expr,
        key: tuple[int, int, int],
        *,
        known_normal_form: bool = False,
    ) -> None:
        bucket = self._consume(key)
        if not known_normal_form:
            normalized = _strict._normalize(expr)
            if _strict._expr_value(normalized) != _strict._expr_value(expr):
                bucket.rewrite_collapses += 1
                return
        try:
            ast = _strict._accepted(expr)
            decode_shrink1_canonical_ast(ast.cbor_bytes)
        except _strict.StrictAstError as error:
            if error.code == "REJECT_STRUCTURAL_LIMIT":
                bucket.structural_limit_rejections += 1
                return
            _fail("FAIL_ENUMERATION_CANONICALIZER", str(error))
        if ast.metrics.output_sort not in OUTPUT_SORT_IDS:
            _fail("FAIL_ENUMERATION_SORT", f"unregistered output sort {ast.metrics.output_sort}")
        prior = self.seen.get(ast.cbor_bytes)
        if prior is not None:
            if prior.ast.digest != ast.digest:
                _fail("FAIL_AST_HASH_COLLISION", "same bytes carried different AST hashes")
            bucket.syntactic_duplicates += 1
            return
        if ast.metrics.depth != key[1] or ast.metrics.node_count != key[2] or OUTPUT_SORT_IDS[ast.metrics.output_sort] != key[0]:
            _fail("FAIL_ENUMERATION_BUCKET", "normal-form construction entered the wrong bucket")
        program = _Program(expr, ast)
        self.seen[ast.cbor_bytes] = program
        self.groups.setdefault((ast.metrics.output_sort, ast.metrics.depth, ast.metrics.node_count), []).append(program)

    def leaves(self, output_sort_id: int) -> None:
        if output_sort_id == 1:
            for context_id in range(4):
                self._admit(_strict._Expr(0, 4, "Bool", parameters=(context_id,)), (1, 0, 1), known_normal_form=True)
            for task_id in range(2):
                self._admit(_strict._Expr(0, 5, "Bool", parameters=(task_id,)), (1, 0, 1), known_normal_form=True)
        elif output_sort_id == 2:
            for index in range(8):
                self._admit(_strict._Expr(0, 1, "Bit", parameters=(index,)), (2, 0, 1), known_normal_form=True)
        elif output_sort_id == 4:
            self._admit(_strict._Expr(0, 2, "BoundedInt"), (4, 0, 1), known_normal_form=True)
            for scope_id, quantity_id, extension in product(range(4), range(2), SCOPE_EXTENSIONS):
                self._admit(_strict._Expr(0, 3, "BoundedInt", parameters=(1, scope_id, quantity_id, extension)), (4, 0, 1), known_normal_form=True)
        elif output_sort_id == 5:
            for index in range(7):
                self._admit(_strict._Expr(0, 0, "RationalValue", parameters=(index,)), (5, 0, 1), known_normal_form=True)
            for map_id, scope_id, quantity_id, extension in product((0, 5), range(4), range(2), SCOPE_EXTENSIONS):
                self._admit(_strict._Expr(0, 3, "RationalValue", parameters=(map_id, scope_id, quantity_id, extension)), (5, 0, 1), known_normal_form=True)

    def _group(self, sort: str, depth: int, nodes: int) -> tuple[_Program, ...]:
        return tuple(self.groups.get((sort, depth, nodes), ()))

    def _structural_groups(self, sort: str, depth: int, nodes: int, arity: int) -> Iterator[tuple[tuple[int, int], ...]]:
        choices = tuple((d, n) for d in range(depth) for n in range(1, nodes))
        for selected in product(choices, repeat=arity):
            if max(item[0] for item in selected) == depth - 1 and sum(item[1] for item in selected) == nodes - 1:
                yield selected

    def unary(self, depth: int, nodes: int, output_sort_id: int) -> None:
        specs = (
            (0, "Bit", "RationalValue"),
            (1, "BoundedInt", "RationalValue"),
            (2, "RationalValue", "RationalValue"),
            (3, "RationalValue", "Sign"),
        )
        for operator, input_sort, output_sort in specs:
            if OUTPUT_SORT_IDS[output_sort] != output_sort_id:
                continue
            for child in self._group(input_sort, depth - 1, nodes - 1):
                self._admit(
                    _strict._Expr(1, operator, output_sort, (child.expr,)),
                    (OUTPUT_SORT_IDS[output_sort], depth, nodes),
                    known_normal_form=operator in {0, 1, 3},
                )

    def _commutative_pairs(self, sort: str, depth: int, nodes: int) -> Iterator[tuple[_Program, _Program]]:
        group_specs = sorted(set(self._structural_groups(sort, depth, nodes, 2)))
        seen_specs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        for left_spec, right_spec in group_specs:
            unordered = tuple(sorted((left_spec, right_spec)))
            if unordered in seen_specs:
                continue
            seen_specs.add(unordered)
            left = sorted(self._group(sort, *unordered[0]), key=lambda item: item.commutative_key)
            right = sorted(self._group(sort, *unordered[1]), key=lambda item: item.commutative_key)
            if unordered[0] == unordered[1]:
                yield from combinations_with_replacement(left, 2)
            else:
                for first, second in product(left, right):
                    yield tuple(sorted((first, second), key=lambda item: item.commutative_key))  # type: ignore[misc]

    def _ordered_pairs(self, sort: str, depth: int, nodes: int) -> Iterator[tuple[_Program, _Program]]:
        for left_spec, right_spec in self._structural_groups(sort, depth, nodes, 2):
            yield from product(self._group(sort, *left_spec), self._group(sort, *right_spec))

    def binary_and_ternary(self, depth: int, nodes: int, output_sort_id: int) -> None:
        for operator, output in ((0, "RationalValue"), (2, "Bool")):
            if OUTPUT_SORT_IDS[output] != output_sort_id:
                continue
            for left, right in self._commutative_pairs("RationalValue", depth, nodes):
                self._admit(
                    _strict._Expr(2, operator, output, (left.expr, right.expr)),
                    (OUTPUT_SORT_IDS[output], depth, nodes),
                    known_normal_form=operator == 2,
                )
        for operator, output in ((1, "RationalValue"), (3, "Bool")):
            if OUTPUT_SORT_IDS[output] != output_sort_id:
                continue
            for left, right in self._ordered_pairs("RationalValue", depth, nodes):
                self._admit(
                    _strict._Expr(2, operator, output, (left.expr, right.expr)),
                    (OUTPUT_SORT_IDS[output], depth, nodes),
                    known_normal_form=operator == 3,
                )
        if output_sort_id == 1:
            for operator in (5, 6):
                for left, right in self._commutative_pairs("Sign", depth, nodes):
                    self._admit(_strict._Expr(2, operator, "Bool", (left.expr, right.expr)), (1, depth, nodes), known_normal_form=True)
            for left, right in self._commutative_pairs("RationalValue", depth, nodes):
                for tolerance in (1, 2):
                    self._admit(_strict._Expr(3, 0, "Bool", (left.expr, right.expr), (tolerance,)), (1, depth, nodes), known_normal_form=True)

    def conjunctions(self, depth: int, nodes: int) -> None:
        group_keys = tuple(
            sorted(
                (child_depth, child_nodes)
                for (sort, child_depth, child_nodes), values in self.groups.items()
                if sort == "Bool"
                and child_depth < depth
                and child_nodes < nodes
                and any(item.ast.root_operator_id != 0x0400 for item in values)
            )
        )
        for arity in (2, 3):
            for specs in combinations_with_replacement(group_keys, arity):
                if max(item[0] for item in specs) != depth - 1:
                    continue
                if sum(item[1] for item in specs) != nodes - 1:
                    continue
                distinct_specs = tuple(dict.fromkeys(specs))
                choices: list[tuple[tuple[_Program, ...], ...]] = []
                for spec in distinct_specs:
                    count = specs.count(spec)
                    group = tuple(
                        item
                        for item in self._group("Bool", *spec)
                        if item.ast.root_operator_id != 0x0400
                    )
                    choices.append(tuple(combinations(group, count)))
                for selected in product(*choices):
                    children = tuple(
                        sorted(
                            (item for part in selected for item in part),
                            key=lambda item: canonical_cbor_encode(_strict._expr_value(item.expr)),
                        )
                    )
                    self._admit(
                        _strict._Expr(4, 0, "Bool", tuple(child.expr for child in children)),
                        (1, depth, nodes),
                        known_normal_form=True,
                    )


def _bucket_records(
    state: _Enumerator, prefix: Sequence[_Program]
) -> tuple[tuple[object, ...], ...]:
    program_indices: dict[tuple[int, int, int], list[int]] = {key: [] for key in state.buckets}
    for index, program in enumerate(prefix):
        program_indices[program.structural_key].append(index)
    result: list[tuple[object, ...]] = []
    for bucket_index, key in enumerate(sorted(state.buckets)):
        stats = state.buckets[key]
        indices = program_indices[key]
        result.append(build_m3_record_object_v1("BucketAccountingRecordV1", {
            "bucket_index": bucket_index,
            "output_sort_id": key[0],
            "ast_depth": key[1],
            "ast_node_count": key[2],
            "raw_operator_applications": stats.raw_operator_applications,
            "accepted_canonical_programs": len(indices),
            "syntactic_duplicates": stats.syntactic_duplicates,
            "type_rejections": stats.type_rejections,
            "structural_limit_rejections": stats.structural_limit_rejections,
            "rewrite_collapses": stats.rewrite_collapses,
            "first_program_index_or_null": indices[0] if indices else None,
            "last_program_index_or_null": indices[-1] if indices else None,
        }))
    return tuple(result)


def enumerate_bounded_closure_v1(
    bindings: EnumerationBindingsV1,
    *,
    canonical_budget: int = CANONICAL_PROGRAM_BUDGET,
    raw_application_cap: int = RAW_APPLICATION_CAP,
) -> BoundedEnumerationResultV1:
    """Enumerate the exact global prefix; never evaluates a target role."""

    if not isinstance(bindings, EnumerationBindingsV1):
        raise TypeError("bindings must be EnumerationBindingsV1")
    if type(canonical_budget) is not int or canonical_budget < 1:
        raise ValueError("canonical_budget must be positive")
    if type(raw_application_cap) is not int or raw_application_cap < 1:
        raise ValueError("raw_application_cap must be positive")
    state = _Enumerator(raw_cap=raw_application_cap)
    ordered: list[_Program] = []
    traversal_complete = False
    stop = False
    sort_names = {value: key for key, value in OUTPUT_SORT_IDS.items()}
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
    prefix = tuple(ordered[:canonical_budget])
    witness = ordered[canonical_budget] if len(ordered) > canonical_budget else None
    if witness is not None:
        status = "DSL_TOO_LARGE"
    elif traversal_complete:
        status = "COMPLETE"
    else:
        _fail("INCONCLUSIVE_BUDGET", "enumeration ended without witness or closed frontier")
    records = tuple(_formal_program_record(index, program, bindings) for index, program in enumerate(prefix))
    manifests = _chunk_manifests(records)
    bucket_records = _bucket_records(state, prefix)
    return BoundedEnumerationResultV1(
        dsl_version=DSL_VERSION,
        closure_status=status,
        raw_operator_application_count=state.raw_count,
        canonical_program_count=len(prefix),
        first_out_of_budget_program_hash=None if witness is None else witness.ast.digest,
        first_out_of_budget_cbor=None if witness is None else witness.ast.cbor_bytes,
        canonical_program_records=records,
        program_chunk_manifests=manifests,
        bucket_accounting_records=bucket_records,
        canonical_program_archive_root=rfc6962_root(list(records)),
        program_chunk_manifest_root=rfc6962_root(list(manifests)),
        bucket_accounting_root=rfc6962_root(list(bucket_records)),
        traversal_prefix_complete=witness is not None or traversal_complete,
        # A bare enumerator result is only a candidate for a later, bound M3
        # receipt.  Exact budgets do not supply the Commit-A implementation
        # binding, run identity, or state-transition authority required to
        # promote this output.
        authoritative_claim_allowed=False,
    )


__all__ = [
    "BoundedEnumerationError",
    "BoundedEnumerationResultV1",
    "CANONICAL_PROGRAM_BUDGET",
    "EnumerationBindingsV1",
    "RAW_APPLICATION_CAP",
    "SCOPE_EXTENSIONS",
    "canonical_scope_extensions_v1",
    "enumerate_bounded_closure_v1",
    "program_mdl_length_q32",
]
