"""Independent Python Q0 syntax oracle and direct quotient saturation engine.

This module is deliberately target-blind.  It evaluates the frozen four-row
public probe through the production input adapter, enumerates the exact Q0
canonical syntax projection, and independently saturates the same projection
from behavior-class Pareto representatives.  A successful endpoint is still
non-authoritative: only a later Python/Rust host replay may create the formal
``Q0SaturationReceiptV1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, combinations_with_replacement, product
import os
import resource
from time import monotonic
from typing import Final, Iterable, Iterator, NoReturn, Sequence

from . import phase3_q0_input_adapter_v1 as _adapter
from . import phase3_q0_evaluator_v1 as _evaluator
from . import phase3_q0_quotient_contract_v1 as _contract
from .phase3_m3_bounded_enumerator_v1 import program_mdl_length_q32
from .strict_ast_shrink6_v1 import (
    canonicalize_shrink6_source_ast,
    decode_shrink6_canonical_ast,
)
from .strict_ast_v1 import CanonicalAst, StrictAstError
from .strict_cbor_v1 import canonical_cbor_encode, content_hash, rfc6962_root


ENDPOINT_IMPLEMENTATION_ID: Final = b"hegel-q0-python-oracle-v1"
PROGRAM_RECORD_SCHEMA_ID: Final = b"hegel-q0-syntax-program-record/1"
FIXED_POINT_METADATA_SCHEMA_ID: Final = b"hegel-q0-fixed-point-state/1"
SYNTAX_STATE_ROOT_DOMAIN: Final = "HEGEL/Q0/SYNTAX_STATE/V1"
DIRECT_STATE_ROOT_DOMAIN: Final = "HEGEL/Q0/DIRECT_QUOTIENT_STATE/V1"

_SORT_IDS: Final = {
    "Bool": _contract.OutputSortId.BOOL,
    "Bit": _contract.OutputSortId.BIT,
    "Sign": _contract.OutputSortId.SIGN,
    "BoundedInt": _contract.OutputSortId.BOUNDED_INT,
    "RationalValue": _contract.OutputSortId.RATIONAL_VALUE,
}
_UNARY_NAMES: Final = (
    "bit_to_scalar",
    "int_to_scalar",
    "absolute",
    "sign",
)
_BINARY_NAMES: Final = {
    1: "difference",
    2: "equal_exact",
    3: "less_equal",
    5: "same_sign",
    6: "opposite_sign",
}


class Q0OracleError(RuntimeError):
    """Stable fail-closed Python endpoint error."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        guard_id: _contract.Q0ResourceGuardId | None = None,
    ) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.guard_id = guard_id


def _fail(code: str, detail: str) -> NoReturn:
    raise Q0OracleError(code, detail)


def _resource_limit(
    guard_id: _contract.Q0ResourceGuardId,
    detail: str,
) -> NoReturn:
    if not isinstance(guard_id, _contract.Q0ResourceGuardId):
        raise TypeError("guard_id must be Q0ResourceGuardId")
    raise Q0OracleError(
        "INCONCLUSIVE_RESOURCE_LIMIT",
        detail,
        guard_id=guard_id,
    )


@dataclass(frozen=True, slots=True)
class _LeafSeed:
    coverage_code: int
    source_ast: tuple[object, ...]
    canonical_node: tuple[object, ...]


Q0_FROZEN_LEAF_SEEDS: Final = (
    _LeafSeed(0x0000, ("scalar_const", 1), (0, 0, 1)),
    _LeafSeed(0x0001, ("scalar_const", 3), (0, 0, 3)),
    _LeafSeed(0x0002, ("scalar_const", 5), (0, 0, 5)),
    _LeafSeed(0x0003, ("bit_at", 0), (0, 1, 0)),
    _LeafSeed(0x0004, ("bit_at", 1), (0, 1, 1)),
    _LeafSeed(0x0005, ("set_size",), (0, 2)),
    _LeafSeed(
        0x0006,
        ("aggregate", 0, 3, 0, ()),
        (0, 3, 0, 3, 0, ()),
    ),
    _LeafSeed(
        0x0007,
        ("aggregate", 1, 3, 0, ()),
        (0, 3, 1, 3, 0, ()),
    ),
    _LeafSeed(
        0x0008,
        ("aggregate", 5, 3, 0, ()),
        (0, 3, 5, 3, 0, ()),
    ),
    _LeafSeed(
        0x0009,
        ("aggregate", 0, 0, 0, ()),
        (0, 3, 0, 0, 0, ()),
    ),
    _LeafSeed(
        0x000A,
        ("aggregate", 0, 3, 1, ()),
        (0, 3, 0, 3, 1, ()),
    ),
    _LeafSeed(
        0x000B,
        ("aggregate", 0, 3, 0, ((0, True),)),
        (0, 3, 0, 3, 0, ((0, True),)),
    ),
    _LeafSeed(
        0x000C,
        ("aggregate", 1, 1, 0, ()),
        (0, 3, 1, 1, 0, ()),
    ),
    _LeafSeed(0x000D, ("context_flag", 0), (0, 4, 0)),
    _LeafSeed(0x000E, ("task_flag", 0), (0, 5, 0)),
)

if tuple(seed.coverage_code for seed in Q0_FROZEN_LEAF_SEEDS) != (
    _contract.Q0_LEAF_COVERAGE_CODES
):
    raise AssertionError("Q0 leaf coverage registry drift")
if tuple(seed.canonical_node for seed in Q0_FROZEN_LEAF_SEEDS) != (
    _contract.Q0_FROZEN_LEAF_CANONICAL_NODES
):
    raise AssertionError("Q0 frozen leaf manifest drift")
if tuple(range(len(_UNARY_NAMES))) != _contract.Q0_ALLOWED_UNARY_OPERATOR_IDS:
    raise AssertionError("Q0 unary projection drift")
if tuple(_BINARY_NAMES) != _contract.Q0_ALLOWED_BINARY_OPERATOR_IDS:
    raise AssertionError("Q0 binary projection drift")
if (1, 2) != _contract.Q0_ALLOWED_APPROX_TOLERANCE_IDS:
    raise AssertionError("Q0 tolerance projection drift")
if _contract.Q0_TOP_LEVEL_AND_ARITY != 2:
    raise AssertionError("Q0 AND projection drift")


@dataclass(frozen=True, slots=True)
class _Program:
    ast: CanonicalAst

    @property
    def node(self) -> tuple[object, ...]:
        node = self.ast.value[1]
        assert isinstance(node, tuple)
        return node

    @property
    def output_sort_id(self) -> _contract.OutputSortId:
        try:
            return _SORT_IDS[self.ast.metrics.output_sort]
        except KeyError as error:  # pragma: no cover - strict AST closes sorts
            _fail("FAIL_Q0_OUTPUT_SORT", str(error))

    @property
    def global_key(self) -> tuple[object, ...]:
        return (
            self.ast.metrics.depth,
            self.ast.metrics.node_count,
            int(self.output_sort_id),
            self.ast.root_operator_id,
            self.ast.cbor_bytes,
        )

    @property
    def commutative_key(self) -> tuple[bytes, bytes]:
        encoded = canonical_cbor_encode(self.node)
        from hashlib import sha256

        return sha256(encoded).digest(), encoded


@dataclass(frozen=True, slots=True)
class _Candidate:
    coverage_code: int
    source_ast: tuple[object, ...]
    expected_node: tuple[object, ...]


@dataclass(slots=True)
class _Coverage:
    eligible_raw: int = 0
    strict_admitted: int = 0
    rewrite_collapses: int = 0
    canonical_duplicates: int = 0
    new_canonical: int = 0

    def canonical_record(self, code: int) -> tuple[int, int, int, int, int, int]:
        return (
            code,
            self.eligible_raw,
            self.strict_admitted,
            self.rewrite_collapses,
            self.canonical_duplicates,
            self.new_canonical,
        )


@dataclass(frozen=True, slots=True)
class RoundDeltaV1:
    round_index: int
    queued_application_count: int
    new_canonical_program_count: int
    new_behavior_class_count: int
    frontier_mutation_count: int
    bank_mutation_count: int
    complete_state_changed: bool


@dataclass(frozen=True, slots=True)
class Q0OracleEndpointResultV1:
    endpoint_status: str
    syntax_raw_application_count: int
    quotient_raw_application_count: int
    strict_admitted_syntax_application_count: int
    strict_admitted_quotient_application_count: int
    rewrite_collapse_syntax_count: int
    rewrite_collapse_quotient_count: int
    canonical_syntax_program_count: int
    behavior_class_count: int
    frontier_point_count: int
    maximum_frontier_points_per_class: int
    syntax_continuation_bank_point_count: int
    quotient_continuation_bank_point_count: int
    maximum_syntax_bank_points_per_class: int
    maximum_quotient_bank_points_per_class: int
    saturation_round_count: int
    work_queue_empty: bool
    zero_delta_full_round: bool
    final_class_delta: int
    final_frontier_mutation_delta: int
    final_bank_mutation_delta: int
    projection_manifest_root: bytes
    semantic_binding_root: bytes
    syntax_program_archive_root: bytes
    syntax_class_archive_root: bytes
    direct_class_archive_root: bytes
    syntax_operator_coverage_root: bytes
    quotient_operator_coverage_root: bytes
    syntax_state_root: bytes
    direct_state_root: bytes
    endpoint_state_root: bytes
    syntax_saturation_state_preimage: tuple[object, ...]
    direct_saturation_state_preimage: tuple[object, ...]
    syntax_class_records: tuple[_contract.QuotientClassRecordV1, ...]
    direct_class_records: tuple[_contract.QuotientClassRecordV1, ...]
    syntax_coverage_records: tuple[tuple[int, int, int, int, int, int], ...]
    quotient_coverage_records: tuple[tuple[int, int, int, int, int, int], ...]
    round_deltas: tuple[RoundDeltaV1, ...]
    all_guards_respected: bool = True
    target_truth_accessed: bool = False
    split_accessed: bool = False
    role_evaluation_performed: bool = False
    formal_roots_generated: bool = False
    authoritative_claim_allowed: bool = False

    def canonical_state_object(self) -> tuple[object, ...]:
        """Implementation-neutral endpoint state consumed by host replay."""

        return (
            1,
            _contract.ENDPOINT_STATE_SCHEMA_ID,
            _contract.Q0_FREEZE_VERSION.encode("ascii"),
            _contract.DSL_VERSION.encode("ascii"),
            _contract.CLOSURE_SEMANTICS_VERSION.encode("ascii"),
            _contract.Q0_PROJECTION_ID.encode("ascii"),
            _contract.Q0ProbeInputV1().universe_root,
            self.projection_manifest_root,
            self.semantic_binding_root,
            self.endpoint_status.encode("ascii"),
            self.syntax_raw_application_count,
            self.quotient_raw_application_count,
            self.strict_admitted_syntax_application_count,
            self.strict_admitted_quotient_application_count,
            self.rewrite_collapse_syntax_count,
            self.rewrite_collapse_quotient_count,
            self.canonical_syntax_program_count,
            self.behavior_class_count,
            self.frontier_point_count,
            self.maximum_frontier_points_per_class,
            self.syntax_continuation_bank_point_count,
            self.quotient_continuation_bank_point_count,
            self.maximum_syntax_bank_points_per_class,
            self.maximum_quotient_bank_points_per_class,
            self.saturation_round_count,
            self.work_queue_empty,
            self.zero_delta_full_round,
            self.final_class_delta,
            self.final_frontier_mutation_delta,
            self.final_bank_mutation_delta,
            self.syntax_program_archive_root,
            self.syntax_class_archive_root,
            self.direct_class_archive_root,
            self.syntax_operator_coverage_root,
            self.quotient_operator_coverage_root,
            self.syntax_state_root,
            self.direct_state_root,
            self.all_guards_respected,
            self.target_truth_accessed,
            self.split_accessed,
            self.role_evaluation_performed,
            self.formal_roots_generated,
            self.authoritative_claim_allowed,
        )

    @property
    def canonical_state_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_state_object())

    @property
    def syntax_saturation_state_preimage_bytes(self) -> bytes:
        return canonical_cbor_encode(self.syntax_saturation_state_preimage)

    @property
    def direct_saturation_state_preimage_bytes(self) -> bytes:
        return canonical_cbor_encode(self.direct_saturation_state_preimage)


def _bottom(value: object) -> bool:
    return value is _adapter.BOTTOM


def evaluate_canonical_ast_v1(
    ast: CanonicalAst, environment: object
) -> object:
    """Compatibility wrapper over the source-frozen target-blind evaluator."""

    return _evaluator.evaluate_canonical_ast_raw_v1(ast, environment)


def behavior_blob_for_ast_v1(
    ast: CanonicalAst,
    probe: _contract.Q0ProbeInputV1 | None = None,
) -> _contract.BehaviorBlobV1:
    """Return the exact four-cell, sort-bound Q0 behavior identity."""

    frozen_probe = _contract.Q0ProbeInputV1() if probe is None else probe
    environments = frozen_probe.observation_environments()
    cells = tuple(
        _contract.BehaviorCellV1.bottom()
        if _bottom(value := evaluate_canonical_ast_v1(ast, environment))
        else _contract.BehaviorCellV1.exact(value)
        for environment in environments
    )
    behavior = _contract.BehaviorBlobV1(
        input_signature_id=_contract.Q0_PROBE_INPUT_SIGNATURE_ID,
        frozen_universe_root=frozen_probe.universe_root,
        output_sort_id=_SORT_IDS[ast.metrics.output_sort],
        cells=cells,
    )
    _ = behavior.canonical_bytes
    return behavior


def future_signature_for_ast_v1(
    ast: CanonicalAst,
) -> _contract.FutureAdmissibilitySignatureV1:
    return _contract.future_signature_from_ast_v1(ast)


def frontier_entry_for_ast_v1(ast: CanonicalAst) -> _contract.FrontierEntryV1:
    return _contract.FrontierEntryV1(
        signature=future_signature_for_ast_v1(ast),
        normalization_witness_rank=0,
        representative_ast_cbor=ast.cbor_bytes,
        representative_ast_hash=ast.digest,
    )


class QuotientAccumulatorV1:
    """Exact behavior classes with collision checks and Pareto frontiers."""

    def __init__(self, probe: _contract.Q0ProbeInputV1 | None = None) -> None:
        self.probe = _contract.Q0ProbeInputV1() if probe is None else probe
        # Keep the bounded exact-signature cohort bank even when a cohort
        # is currently Pareto-dominated.  A later distinct witness can raise
        # that cohort's normalization multiplicity and make it admissible
        # again; updating only from the previously retained frontier would be
        # non-monotone and could never resurrect it.  Each cohort is truncated
        # immediately to the identity-sensitive arity capacity, so no hidden
        # unguarded third-witness reservoir exists.
        self._classes: dict[
            bytes,
            tuple[
                _contract.BehaviorBlobV1,
                dict[bytes, _contract.FrontierEntryV1],
            ],
        ] = {}
        self._digest_preimages: dict[bytes, bytes] = {}

    @staticmethod
    def _bank_entries(
        entries: Iterable[_contract.FrontierEntryV1],
    ) -> tuple[_contract.FrontierEntryV1, ...]:
        cohorts: dict[tuple[object, ...], list[_contract.FrontierEntryV1]] = {}
        for entry in entries:
            cohorts.setdefault(entry.signature.canonical_object(), []).append(entry)
        bank: list[_contract.FrontierEntryV1] = []
        for cohort in cohorts.values():
            ordered = sorted(cohort, key=lambda item: item.representative_ast_cbor)
            capacity = _contract.normalization_witness_capacity_v1(
                ordered[0].signature.output_sort_id
            )
            bank.extend(ordered[:capacity])
        return tuple(
            sorted(
                bank,
                key=lambda entry: (
                    canonical_cbor_encode(entry.signature.canonical_object()),
                    entry.representative_ast_cbor,
                ),
            )
        )

    def add_ast(self, ast: CanonicalAst) -> tuple[int, int, int]:
        behavior = behavior_blob_for_ast_v1(ast, self.probe)
        behavior_bytes = behavior.canonical_bytes
        behavior_id = behavior.behavior_id
        prior_preimage = self._digest_preimages.get(behavior_id)
        if prior_preimage is not None and prior_preimage != behavior_bytes:
            _fail(
                "FAIL_SHA256_PREIMAGE_COLLISION",
                "one behavior digest has two canonical preimages",
            )
        entry = frontier_entry_for_ast_v1(ast)
        prior = self._classes.get(behavior_bytes)
        if prior is None:
            prospective = dict(self._classes)
            prospective[behavior_bytes] = (
                behavior,
                {entry.representative_ast_cbor: entry},
            )
            self._check_frontier_guards(prospective)
            self._classes[behavior_bytes] = prospective[behavior_bytes]
            self._digest_preimages[behavior_id] = behavior_bytes
            return 1, 1, 1
        entries = dict(prior[1])
        existing = entries.get(entry.representative_ast_cbor)
        if (
            existing is not None
            and existing.representative_ast_hash != entry.representative_ast_hash
        ):
            _fail(
                "FAIL_Q0_AST_IDENTITY_REPLAY_MISMATCH",
                "one cohort AST has two strict hashes",
            )
        prior_frontier = _contract.pareto_frontier_v1(entries.values())
        prior_bank = self._bank_entries(entries.values())
        entries[entry.representative_ast_cbor] = entry
        signature_key = entry.signature.canonical_object()
        cohort = sorted(
            (
                candidate
                for candidate in entries.values()
                if candidate.signature.canonical_object() == signature_key
            ),
            key=lambda candidate: candidate.representative_ast_cbor,
        )
        capacity = _contract.normalization_witness_capacity_v1(
            entry.signature.output_sort_id
        )
        retained_cbor = {
            candidate.representative_ast_cbor for candidate in cohort[:capacity]
        }
        for candidate in cohort[capacity:]:
            if candidate.representative_ast_cbor not in retained_cbor:
                del entries[candidate.representative_ast_cbor]
        new_frontier = _contract.pareto_frontier_v1(entries.values())
        new_bank = self._bank_entries(entries.values())
        changed = int(new_frontier != prior_frontier)
        bank_changed = int(new_bank != prior_bank)
        prospective = dict(self._classes)
        prospective[behavior_bytes] = (prior[0], entries)
        self._check_frontier_guards(prospective)
        self._classes[behavior_bytes] = prospective[behavior_bytes]
        self._digest_preimages[behavior_id] = behavior_bytes
        return 0, changed, bank_changed

    def _check_frontier_guards(
        self,
        classes: dict[
            bytes,
            tuple[
                _contract.BehaviorBlobV1,
                dict[bytes, _contract.FrontierEntryV1],
            ],
        ],
    ) -> None:
        if len(classes) > _contract.Q0_MAX_BEHAVIOR_CLASSES:
            _resource_limit(
                _contract.Q0ResourceGuardId.BEHAVIOR_CLASSES,
                "behavior-class guard reached",
            )
        lengths = tuple(
            len(_contract.pareto_frontier_v1(entries.values()))
            for _, entries in classes.values()
        )
        if sum(lengths) > _contract.Q0_MAX_FRONTIER_POINTS:
            _resource_limit(
                _contract.Q0ResourceGuardId.TOTAL_FRONTIER_POINTS,
                "frontier-point guard reached",
            )
        if lengths and max(lengths) > _contract.Q0_MAX_FRONTIER_POINTS_PER_CLASS:
            _resource_limit(
                _contract.Q0ResourceGuardId.FRONTIER_POINTS_PER_CLASS,
                "per-class frontier guard reached",
            )
        bank_lengths = tuple(
            len(self._bank_entries(entries.values()))
            for _, entries in classes.values()
        )
        if sum(bank_lengths) > _contract.Q0_MAX_CONTINUATION_BANK_POINTS:
            _resource_limit(
                _contract.Q0ResourceGuardId.TOTAL_CONTINUATION_BANK_POINTS,
                "continuation-bank guard reached",
            )
        if (
            bank_lengths
            and max(bank_lengths)
            > _contract.Q0_MAX_CONTINUATION_BANK_POINTS_PER_CLASS
        ):
            _resource_limit(
                _contract.Q0ResourceGuardId.CONTINUATION_BANK_POINTS_PER_CLASS,
                "per-class bank guard reached",
            )

    @property
    def class_count(self) -> int:
        return len(self._classes)

    @property
    def frontier_point_count(self) -> int:
        return sum(
            len(_contract.pareto_frontier_v1(entries.values()))
            for _, entries in self._classes.values()
        )

    @property
    def maximum_frontier_points_per_class(self) -> int:
        return max(
            (
                len(_contract.pareto_frontier_v1(entries.values()))
                for _, entries in self._classes.values()
            ),
            default=0,
        )

    @property
    def continuation_bank_point_count(self) -> int:
        return sum(
            len(self._bank_entries(entries.values()))
            for _, entries in self._classes.values()
        )

    @property
    def maximum_bank_points_per_class(self) -> int:
        return max(
            (
                len(self._bank_entries(entries.values()))
                for _, entries in self._classes.values()
            ),
            default=0,
        )

    def records(self) -> tuple[_contract.QuotientClassRecordV1, ...]:
        material = sorted(
            self._classes.values(),
            key=lambda item: (item[0].behavior_id, item[0].canonical_bytes),
        )
        records = tuple(
            _contract.QuotientClassRecordV1(
                index,
                behavior,
                _contract.pareto_frontier_v1(entries.values()),
            )
            for index, (behavior, entries) in enumerate(material)
        )
        _contract.quotient_class_archive_root_v1(records)
        return records

    def representative_programs(self) -> tuple[_Program, ...]:
        """Return every real representative in the continuation cohort bank.

        Dominated cohorts remain constructively live: a second witness can be
        reachable only through the first latent witness.  Each exact-signature
        cohort therefore contributes its lexicographically least admitted
        representatives up to the sort-specific identity-sensitive arity.
        """

        by_cbor: dict[bytes, _Program] = {}
        for _, entries in self._classes.values():
            for entry in self._bank_entries(entries.values()):
                ast = decode_shrink6_canonical_ast(entry.representative_ast_cbor)
                by_cbor[ast.cbor_bytes] = _Program(ast)
        return tuple(sorted(by_cbor.values(), key=lambda program: program.global_key))

    def continuation_bank_object(self) -> tuple[object, ...]:
        """Canonical bank preimage bound into the fixed-point state root."""

        rows: list[tuple[object, ...]] = []
        for behavior_bytes, (behavior, entries) in self._classes.items():
            cohorts: dict[tuple[object, ...], list[_contract.FrontierEntryV1]] = {}
            for entry in entries.values():
                cohorts.setdefault(entry.signature.canonical_object(), []).append(entry)
            for signature_object, cohort in cohorts.items():
                ordered = sorted(cohort, key=lambda entry: entry.representative_ast_cbor)
                capacity = _contract.normalization_witness_capacity_v1(
                    ordered[0].signature.output_sort_id
                )
                bank = tuple(
                    (
                        rank,
                        entry.representative_ast_cbor,
                        entry.representative_ast_hash,
                    )
                    for rank, entry in enumerate(ordered[:capacity])
                )
                rows.append(
                    (
                        behavior.behavior_id,
                        behavior_bytes,
                        signature_object,
                        bank,
                    )
                )
        return tuple(
            sorted(
                rows,
                key=lambda row: (
                    row[0],
                    row[1],
                    canonical_cbor_encode(row[2]),
                ),
            )
        )


def _canonical_node_to_source(node: tuple[object, ...]) -> tuple[object, ...]:
    tag = node[0]
    if tag == 0:
        leaf = node[1]
        if leaf in (0, 1, 4, 5):
            names = {0: "scalar_const", 1: "bit_at", 4: "context_flag", 5: "task_flag"}
            return (names[leaf], node[2])
        if leaf == 2:
            return ("set_size",)
        if leaf == 3:
            return ("aggregate", node[2], node[3], node[4], node[5])
    if tag == 1:
        return (_UNARY_NAMES[node[1]], _canonical_node_to_source(node[2]))
    if tag == 2:
        return (
            _BINARY_NAMES[node[1]],
            _canonical_node_to_source(node[2]),
            _canonical_node_to_source(node[3]),
        )
    if tag == 3:
        return (
            "approx_equal",
            _canonical_node_to_source(node[2]),
            _canonical_node_to_source(node[3]),
            node[4],
        )
    if tag == 4:
        return ("top_level_AND",) + tuple(
            _canonical_node_to_source(child) for child in node[1]
        )
    _fail("FAIL_Q0_CANONICAL_SOURCE_ADAPTER", "unknown canonical AST node")


def _eligible(children: Sequence[_Program], *, conjunction: bool = False) -> bool:
    metrics = tuple(child.ast.metrics for child in children)
    depth = 1 + max(metric.depth for metric in metrics)
    nodes = 1 + sum(metric.node_count for metric in metrics)
    aggregate = sum(metric.aggregate_leaf_count for metric in metrics)
    scalar = sum(metric.scalar_parameter_occurrences for metric in metrics)
    scopes = sum(metric.scope_clause_count for metric in metrics)
    bits = frozenset().union(*(metric.distinct_bit_slots for metric in metrics))
    return (
        depth <= _contract.Q0_PROJECTION_MAX_AST_DEPTH
        and nodes <= _contract.Q0_PROJECTION_MAX_NODE_COUNT
        and aggregate <= _contract.Q0_PROJECTION_MAX_AGGREGATE_LEAVES
        and scalar <= 3
        and scopes <= 2
        and len(bits) <= 4
        and (not conjunction or len(children) == 2)
    )


def _operator_candidates(programs: Sequence[_Program]) -> Iterator[_Candidate]:
    groups: dict[str, tuple[_Program, ...]] = {}
    for sort in _SORT_IDS:
        groups[sort] = tuple(
            sorted(
                (program for program in programs if program.ast.metrics.output_sort == sort),
                key=lambda program: program.global_key,
            )
        )

    unary_specs = (
        (0, "Bit"),
        (1, "BoundedInt"),
        (2, "RationalValue"),
        (3, "RationalValue"),
    )
    for operator, input_sort in unary_specs:
        for child in groups[input_sort]:
            if not _eligible((child,)):
                continue
            source = (_UNARY_NAMES[operator], _canonical_node_to_source(child.node))
            yield _Candidate(0x1000 + operator, source, (1, operator, child.node))

    rational = groups["RationalValue"]
    commutative_rational = tuple(sorted(rational, key=lambda item: item.commutative_key))
    for left, right in product(rational, repeat=2):
        if not _eligible((left, right)):
            continue
        source = (
            "difference",
            _canonical_node_to_source(left.node),
            _canonical_node_to_source(right.node),
        )
        yield _Candidate(0x2001, source, (2, 1, left.node, right.node))
        source = (
            "less_equal",
            _canonical_node_to_source(left.node),
            _canonical_node_to_source(right.node),
        )
        yield _Candidate(0x2003, source, (2, 3, left.node, right.node))

    for left, right in combinations_with_replacement(commutative_rational, 2):
        if not _eligible((left, right)):
            continue
        source_left = _canonical_node_to_source(left.node)
        source_right = _canonical_node_to_source(right.node)
        yield _Candidate(
            0x2002,
            ("equal_exact", source_left, source_right),
            (2, 2, left.node, right.node),
        )
        for tolerance in (1, 2):
            yield _Candidate(
                0x3000 + tolerance,
                ("approx_equal", source_left, source_right, tolerance),
                (3, 0, left.node, right.node, tolerance),
            )

    signs = tuple(sorted(groups["Sign"], key=lambda item: item.commutative_key))
    for left, right in combinations_with_replacement(signs, 2):
        if not _eligible((left, right)):
            continue
        for operator in (5, 6):
            yield _Candidate(
                0x2000 + operator,
                (
                    _BINARY_NAMES[operator],
                    _canonical_node_to_source(left.node),
                    _canonical_node_to_source(right.node),
                ),
                (2, operator, left.node, right.node),
            )

    bool_atoms = tuple(
        sorted(
            (program for program in groups["Bool"] if program.node[0] != 4),
            key=lambda program: canonical_cbor_encode(program.node),
        )
    )
    for left, right in combinations(bool_atoms, 2):
        if not _eligible((left, right), conjunction=True):
            continue
        yield _Candidate(
            _contract.Q0_AND2_COVERAGE_CODE,
            (
                "top_level_AND",
                _canonical_node_to_source(left.node),
                _canonical_node_to_source(right.node),
            ),
            (4, (left.node, right.node)),
        )


def _memory_bytes() -> int:
    # The endpoint guard is scoped to current resident state.  ``ru_maxrss``
    # includes unrelated earlier pytest/plugin peaks in a reused host process,
    # so Linux replay uses the current /proc value; Docker supplies the hard
    # cgroup ceiling.  The fallback retains portable fail-closed accounting.
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            fields = handle.read().split()
        return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):  # pragma: no cover - Linux replay
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(usage) * 1024


class _Engine:
    def __init__(self, *, direct: bool, start_time: float) -> None:
        self.direct = direct
        self.start_time = start_time
        self.programs: dict[bytes, _Program] = {}
        self.ast_digests: dict[bytes, bytes] = {}
        self.accumulator = QuotientAccumulatorV1()
        self.coverage = {code: _Coverage() for code in _contract.Q0_COVERAGE_CODES}
        self.attempted_sources: set[tuple[object, ...]] = set()
        self.raw_count = 0
        self.strict_admitted_count = 0
        self.rewrite_collapse_count = 0
        self.rounds: list[RoundDeltaV1] = []

    def _guard(self) -> None:
        if self.raw_count > _contract.Q0_MAX_RAW_APPLICATIONS:
            _resource_limit(
                _contract.Q0ResourceGuardId.RAW_OPERATOR_APPLICATIONS,
                "raw application guard reached",
            )
        if len(self.programs) > _contract.Q0_MAX_CANONICAL_SYNTAX:
            _resource_limit(
                _contract.Q0ResourceGuardId.CANONICAL_SYNTAX_PROGRAMS,
                "canonical syntax guard reached",
            )
        if monotonic() - self.start_time > _contract.Q0_MAX_WALL_TIME_SECONDS:
            _resource_limit(
                _contract.Q0ResourceGuardId.WALL_TIME,
                "wall-time guard reached",
            )
        if _memory_bytes() > _contract.Q0_MAX_MEMORY_BYTES:
            _resource_limit(
                _contract.Q0ResourceGuardId.RESIDENT_MEMORY,
                "memory guard reached",
            )

    def _process(self, candidate: _Candidate) -> tuple[int, int, int, int]:
        coverage = self.coverage[candidate.coverage_code]
        coverage.eligible_raw += 1
        self.raw_count += 1
        self._guard()
        try:
            ast = canonicalize_shrink6_source_ast(candidate.source_ast)
            replay = decode_shrink6_canonical_ast(ast.cbor_bytes)
        except StrictAstError as error:
            _fail("FAIL_Q0_STRICT_CANONICALIZER", str(error))
        if replay.digest != ast.digest or replay.cbor_bytes != ast.cbor_bytes:
            _fail(
                "FAIL_Q0_AST_IDENTITY_REPLAY_MISMATCH",
                "strict replay identity differs",
            )
        coverage.strict_admitted += 1
        self.strict_admitted_count += 1
        if ast.value[1] != candidate.expected_node:
            coverage.rewrite_collapses += 1
            self.rewrite_collapse_count += 1
        metrics = ast.metrics
        if (
            metrics.depth > _contract.Q0_PROJECTION_MAX_AST_DEPTH
            or metrics.node_count > _contract.Q0_PROJECTION_MAX_NODE_COUNT
            or metrics.top_level_clause_count > _contract.Q0_PROJECTION_MAX_TOP_LEVEL_CLAUSES
            or metrics.aggregate_leaf_count > _contract.Q0_PROJECTION_MAX_AGGREGATE_LEAVES
        ):
            _fail("FAIL_Q0_PROJECTION_ADMISSION", "strict survivor exceeds Q0 projection")
        prior_digest = self.ast_digests.get(ast.cbor_bytes)
        if prior_digest is not None:
            if prior_digest != ast.digest:
                _fail(
                    "FAIL_Q0_AST_IDENTITY_REPLAY_MISMATCH",
                    "same AST bytes carry two hashes",
                )
            coverage.canonical_duplicates += 1
            return 0, 0, 0, 0
        if len(self.programs) + 1 > _contract.Q0_MAX_CANONICAL_SYNTAX:
            _resource_limit(
                _contract.Q0ResourceGuardId.CANONICAL_SYNTAX_PROGRAMS,
                "canonical syntax guard reached",
            )
        program = _Program(ast)
        # The accumulator performs its class/frontier/bank guard checks on a
        # prospective copy and commits atomically.  Only after that succeeds
        # may this engine expose the new canonical program or coverage delta.
        class_delta, frontier_delta, bank_delta = self.accumulator.add_ast(ast)
        self.ast_digests[ast.cbor_bytes] = ast.digest
        self.programs[ast.cbor_bytes] = program
        coverage.new_canonical += 1
        return 1, class_delta, frontier_delta, bank_delta

    def seed(self) -> None:
        for seed in Q0_FROZEN_LEAF_SEEDS:
            self.attempted_sources.add(seed.source_ast)
            self._process(_Candidate(seed.coverage_code, seed.source_ast, seed.canonical_node))

    def child_programs(self) -> tuple[_Program, ...]:
        if self.direct:
            return self.accumulator.representative_programs()
        return tuple(sorted(self.programs.values(), key=lambda program: program.global_key))

    def saturate(self) -> None:
        for round_index in range(1, _contract.Q0_MAX_SATURATION_ROUNDS + 1):
            before_state = self._core_state_root()
            queue = tuple(
                candidate
                for candidate in _operator_candidates(self.child_programs())
                if candidate.source_ast not in self.attempted_sources
            )
            for candidate in queue:
                self.attempted_sources.add(candidate.source_ast)
            new_programs = 0
            new_classes = 0
            frontier_mutations = 0
            bank_mutations = 0
            for candidate in queue:
                program_delta, class_delta, frontier_delta, bank_delta = self._process(candidate)
                new_programs += program_delta
                new_classes += class_delta
                frontier_mutations += frontier_delta
                bank_mutations += bank_delta
            after_state = self._core_state_root()
            changed = before_state != after_state
            self.rounds.append(
                RoundDeltaV1(
                    round_index,
                    len(queue),
                    new_programs,
                    new_classes,
                    frontier_mutations,
                    bank_mutations,
                    changed,
                )
            )
            if not queue:
                if changed or new_classes or frontier_mutations or bank_mutations:
                    _fail("FAIL_Q0_FIXED_POINT", "empty queue changed complete state")
                return
        _resource_limit(
            _contract.Q0ResourceGuardId.SATURATION_ROUNDS,
            "saturation round guard reached",
        )

    def coverage_records(self) -> tuple[tuple[int, int, int, int, int, int], ...]:
        records = tuple(
            self.coverage[code].canonical_record(code)
            for code in _contract.Q0_COVERAGE_CODES
        )
        if any(len(record) != _contract.Q0_COVERAGE_RECORD_LENGTH for record in records):
            raise AssertionError("coverage wire length drift")
        return records

    def class_records(self) -> tuple[_contract.QuotientClassRecordV1, ...]:
        return self.accumulator.records()

    def _core_state_object(self) -> tuple[object, ...]:
        program_records = _program_records(self.programs.values())
        continuation_bank = self.accumulator.continuation_bank_object()
        class_objects = tuple(record.canonical_object() for record in self.class_records())
        coverage_records = self.coverage_records()
        return program_records, continuation_bank, class_objects, coverage_records

    def _core_state_root(self) -> bytes:
        domain = DIRECT_STATE_ROOT_DOMAIN if self.direct else SYNTAX_STATE_ROOT_DOMAIN
        return content_hash(domain + "/CORE", self._core_state_object())

    def fixed_point_metadata_object(self) -> tuple[object, ...]:
        if not self.rounds:
            _fail("FAIL_Q0_FIXED_POINT", "saturation rounds are absent")
        final = self.rounds[-1]
        zero_delta = (
            final.queued_application_count == 0
            and final.new_canonical_program_count == 0
            and final.new_behavior_class_count == 0
            and final.frontier_mutation_count == 0
            and final.bank_mutation_count == 0
            and not final.complete_state_changed
        )
        all_eligible_tuples_covered = not any(
            candidate.source_ast not in self.attempted_sources
            for candidate in _operator_candidates(self.child_programs())
        )
        return (
            1,
            FIXED_POINT_METADATA_SCHEMA_ID,
            (
                b"hegel-q0-direct-quotient-path/1"
                if self.direct
                else b"hegel-q0-exhaustive-syntax-path/1"
            ),
            len(self.rounds),
            final.queued_application_count == 0,
            zero_delta,
            all_eligible_tuples_covered,
            final.new_canonical_program_count,
            final.new_behavior_class_count,
            final.frontier_mutation_count,
            final.bank_mutation_count,
        )

    def semantic_state_root(self) -> bytes:
        domain = DIRECT_STATE_ROOT_DOMAIN if self.direct else SYNTAX_STATE_ROOT_DOMAIN
        return content_hash(domain, self.saturation_state_object())

    def saturation_state_object(self) -> tuple[object, ...]:
        """Return the complete normative five-tuple state-root preimage."""

        (
            program_records,
            continuation_bank,
            class_objects,
            coverage_records,
        ) = self._core_state_object()
        return (
            program_records,
            continuation_bank,
            class_objects,
            coverage_records,
            self.fixed_point_metadata_object(),
        )


def _program_records(programs: Iterable[_Program]) -> tuple[tuple[object, ...], ...]:
    ordered = tuple(sorted(programs, key=lambda program: program.global_key))
    return tuple(
        (
            1,
            PROGRAM_RECORD_SCHEMA_ID,
            index,
            program.ast.cbor_bytes,
            program.ast.digest,
            int(program.output_sort_id),
            program_mdl_length_q32(program.ast),
        )
        for index, program in enumerate(ordered)
    )


def syntax_program_archive_root_v1(programs: Iterable[_Program]) -> bytes:
    records = _program_records(programs)
    # RFC6962 supplies the archive tree; the schema/domain are present in every
    # record and the endpoint schema binds the root's role.
    return rfc6962_root(list(records))


def operator_coverage_root_v1(
    records: Sequence[tuple[int, int, int, int, int, int]],
) -> bytes:
    material = tuple(records)
    if tuple(record[0] for record in material) != _contract.Q0_COVERAGE_CODES:
        _fail("FAIL_Q0_OPERATOR_COVERAGE", "coverage registry/order differs")
    return rfc6962_root(list(material))


def run_q0_python_oracle_v1() -> Q0OracleEndpointResultV1:
    """Run both independent Python Q0 paths and return a diagnostic endpoint."""

    start = monotonic()
    syntax = _Engine(direct=False, start_time=start)
    syntax.seed()
    syntax.saturate()
    direct = _Engine(direct=True, start_time=start)
    direct.seed()
    direct.saturate()

    syntax_records = syntax.class_records()
    direct_records = direct.class_records()
    syntax_class_root = _contract.quotient_class_archive_root_v1(syntax_records)
    direct_class_root = _contract.quotient_class_archive_root_v1(direct_records)
    syntax_bank = syntax.accumulator.continuation_bank_object()
    direct_bank = direct.accumulator.continuation_bank_object()
    if (
        tuple(record.canonical_bytes for record in syntax_records)
        != tuple(record.canonical_bytes for record in direct_records)
        or syntax_class_root != direct_class_root
        or syntax_bank != direct_bank
    ):
        _fail(
            "FAIL_Q0_SYNTAX_DIRECT_DISAGREEMENT",
            "exhaustive syntax quotient differs from direct quotient saturation",
        )

    syntax_coverage = syntax.coverage_records()
    direct_coverage = direct.coverage_records()
    program_root = syntax_program_archive_root_v1(syntax.programs.values())
    syntax_coverage_root = operator_coverage_root_v1(syntax_coverage)
    direct_coverage_root = operator_coverage_root_v1(direct_coverage)
    final_round = direct.rounds[-1]
    syntax_fixed_point = syntax.fixed_point_metadata_object()
    direct_fixed_point = direct.fixed_point_metadata_object()
    if not all(
        metadata[index] is True
        for metadata in (syntax_fixed_point, direct_fixed_point)
        for index in (4, 5, 6)
    ):
        _fail(
            "FAIL_Q0_FIXED_POINT",
            "queue, zero-delta, or eligible-tuple coverage proof is false",
        )
    if (
        final_round.queued_application_count != 0
        or final_round.new_canonical_program_count != 0
        or final_round.new_behavior_class_count != 0
        or final_round.frontier_mutation_count != 0
        or final_round.bank_mutation_count != 0
        or final_round.complete_state_changed
    ):
        _fail("FAIL_Q0_FIXED_POINT", "final full round is not zero-delta")

    endpoint_kwargs = dict(
        endpoint_status=_contract.Q0_ENDPOINT_PASS_STATUS,
        syntax_raw_application_count=syntax.raw_count,
        quotient_raw_application_count=direct.raw_count,
        strict_admitted_syntax_application_count=syntax.strict_admitted_count,
        strict_admitted_quotient_application_count=direct.strict_admitted_count,
        rewrite_collapse_syntax_count=syntax.rewrite_collapse_count,
        rewrite_collapse_quotient_count=direct.rewrite_collapse_count,
        canonical_syntax_program_count=len(syntax.programs),
        behavior_class_count=len(syntax_records),
        frontier_point_count=syntax.accumulator.frontier_point_count,
        maximum_frontier_points_per_class=syntax.accumulator.maximum_frontier_points_per_class,
        syntax_continuation_bank_point_count=syntax.accumulator.continuation_bank_point_count,
        quotient_continuation_bank_point_count=direct.accumulator.continuation_bank_point_count,
        maximum_syntax_bank_points_per_class=syntax.accumulator.maximum_bank_points_per_class,
        maximum_quotient_bank_points_per_class=direct.accumulator.maximum_bank_points_per_class,
        saturation_round_count=len(direct.rounds),
        work_queue_empty=bool(direct_fixed_point[4]),
        zero_delta_full_round=bool(direct_fixed_point[5]),
        final_class_delta=final_round.new_behavior_class_count,
        final_frontier_mutation_delta=final_round.frontier_mutation_count,
        final_bank_mutation_delta=final_round.bank_mutation_count,
        projection_manifest_root=_contract.q0_projection_manifest_root_v1(),
        semantic_binding_root=_contract.q0_semantic_binding_root_v1(),
        syntax_program_archive_root=program_root,
        syntax_class_archive_root=syntax_class_root,
        direct_class_archive_root=direct_class_root,
        syntax_operator_coverage_root=syntax_coverage_root,
        quotient_operator_coverage_root=direct_coverage_root,
        syntax_state_root=syntax.semantic_state_root(),
        direct_state_root=direct.semantic_state_root(),
        endpoint_state_root=b"\x00" * 32,
        syntax_saturation_state_preimage=syntax.saturation_state_object(),
        direct_saturation_state_preimage=direct.saturation_state_object(),
        syntax_class_records=syntax_records,
        direct_class_records=direct_records,
        syntax_coverage_records=syntax_coverage,
        quotient_coverage_records=direct_coverage,
        round_deltas=tuple(direct.rounds),
    )
    provisional = Q0OracleEndpointResultV1(**endpoint_kwargs)
    endpoint_root = content_hash(
        _contract.ENDPOINT_STATE_ROOT_DOMAIN,
        provisional.canonical_state_object(),
    )
    endpoint_kwargs["endpoint_state_root"] = endpoint_root
    result = Q0OracleEndpointResultV1(**endpoint_kwargs)
    if len(result.canonical_state_bytes) > _contract.Q0_MAX_OUTPUT_BYTES:
        _resource_limit(
            _contract.Q0ResourceGuardId.OUTPUT_BYTES,
            "endpoint output-byte guard reached",
        )
    return result


__all__ = [
    "ENDPOINT_IMPLEMENTATION_ID",
    "Q0_FROZEN_LEAF_SEEDS",
    "Q0OracleEndpointResultV1",
    "Q0OracleError",
    "QuotientAccumulatorV1",
    "RoundDeltaV1",
    "behavior_blob_for_ast_v1",
    "evaluate_canonical_ast_v1",
    "frontier_entry_for_ast_v1",
    "future_signature_for_ast_v1",
    "operator_coverage_root_v1",
    "run_q0_python_oracle_v1",
    "syntax_program_archive_root_v1",
]
