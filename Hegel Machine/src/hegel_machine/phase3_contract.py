"""Fail-closed preregistration contracts for bounded Phase-3 work.

This module binds the frozen old-language surface tables, target, control, and
MDL parameters through ``hegel-freeze-p2b-p3-v1.0.2``.  The strict canonical
AST/CBOR, certificate bridge, and MDL wire are complete as specifications, but
the strict acceptance implementations have now passed the independent
Python/Rust golden vectors and the 64,680-program capacity replay.  Complete
enumerators, program-output replay, trusted key-status execution, and the
formal MDL scorer remain machine-readable implementation blockers.

The module is independent of the Phase-2 selector and does not authorize an
ACTIVE theory mutation.  The verified capacity replay establishes only
``DSL_TOO_LARGE`` for the bounded old DSL; formal Merkle roots remain null and
an independently replaying complete evaluator must still be implemented before
``OUTSIDE_FROZEN_CLOSURE`` can be issued.  Self-declared receipt fields are
never treated as a certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path
import hashlib
import re

from .hashing import stable_hash
from .phase3_certificate_v1 import (
    AST_SHAPE_PREFIXES,
    BINARY_TOKEN_CODES,
    FIXED_POINT_PRECISION_ID,
    FREEZE_VERSION as PHASE3_INHERITED_SURFACE_FREEZE_VERSION,
    FORMAL_MDL_AST_SCORER_IMPLEMENTED,
    FORMAL_OUTSIDE_CERTIFICATE_ISSUANCE_IMPLEMENTED,
    LATEST_KEY_STATUS_RESOLVER_IMPLEMENTED,
    LEAF_CLASS_CODES,
    MDL_DECIMAL_PRECISION,
    MDL_CODE_TABLE_ID,
    PROGRAM_OUTPUT_ARCHIVE_REPLAY_IMPLEMENTED,
    PYTHON_CLOSURE_REPLAY_IMPLEMENTED,
    RATIONAL_PARAMETER_CODES,
    RUST_CLOSURE_REPLAY_IMPLEMENTED,
    SCOPE_CLAUSE_COUNT_CODES,
    TERNARY_TOKEN_CODES,
    TOLERANCE_CODES,
    UNARY_TOKEN_CODES,
)
from .phase3_closure_preflight import (
    CAPACITY_PROOF,
    DSL_TOO_LARGE_STATUS,
    DUAL_STRICT_CAPACITY_REPLAY_REPORT_ID,
    DUAL_STRICT_GATE_REPORT_ID,
    FIRST_OUT_OF_BUDGET_AST_HASH,
    STRICT_CAPACITY_SET_COMMITMENT,
)
from .phase3_dsl_v1 import (
    AGGREGATE_CATALOG,
    BINARY_OPERATORS,
    BINARY_XOR_SANITY,
    BOOLEAN_COMPOSITION,
    BOTTOM_AND_EQUIVALENCE,
    CLOSURE_BUDGET,
    FORBIDDEN_FORMS,
    HIDDEN_TARGET_REGISTRY,
    LEAF_EXPRESSIONS,
    OBSERVED_OMITTED_SINK_CONTROL,
    ODD_REDUCTION_TARGET,
    ODD_REDUCTION_UNIVERSE,
    OLD_DSL_V1,
    PRIMITIVE_SORT_IDS,
    SHRINK_ORDER,
    STRUCTURAL_LIMITS,
    TERNARY_OPERATORS,
    UNARY_OPERATORS,
)


PHASE3_CONTRACT_SCHEMA_VERSION = "hegel-machine-phase3-preregistration/1"
PHASE3_OVERALL_FREEZE_VERSION = "hegel-freeze-p2b-p3-v1.0.2"
CANONICAL_CBOR_PROFILE_ID = "hegel-cbor-det-v1"
CANONICAL_AST_SCHEMA_ID = "hegel-canonical-ast-v1"
Q32_REFERENCE_ALGORITHM_ID = "hegel-mpfr-log2-q32-v1"
NEW_REDUCER_V1_HEADER = 0x4852

SURFACE_PARAMETER_FREEZE_COMPLETE = True
STRICT_ACCEPTANCE_CONTRACT_COMPLETE = True
NORMATIVE_PARAMETER_FREEZE_COMPLETE = True
STRICT_ACCEPTANCE_IMPLEMENTATION_VERIFIED = True
FORMAL_ROOT_GENERATION_ALLOWED = False
TARGET_SYNTHESIS_ALLOWED = False
OUTSIDE_CERTIFICATE_ALLOWED = False
MDL_CERTIFICATE_ALLOWED = False
PHASE2B_FORMAL_EXIT = False
ACTIVE_PROMOTION_ALLOWED = False

if PHASE3_INHERITED_SURFACE_FREEZE_VERSION != "hegel-freeze-p2b-p3-v1.0.1":
    raise AssertionError("v1.0.2 must inherit the frozen v1.0.1 surface")

# These are deliberately hard-disabled implementation facts, rather than
# constructor inputs a caller could set.  The current module defines wire
# contracts and arithmetic only; it does not replay closure archives or MDL
# partitions under a trusted evaluator.
SEALED_CLOSURE_VERIFIER_IMPLEMENTED = (
    FORMAL_OUTSIDE_CERTIFICATE_ISSUANCE_IMPLEMENTED
)
SEALED_MDL_SCORER_IMPLEMENTED = FORMAL_MDL_AST_SCORER_IMPLEMENTED


FROZEN_SORTS = PRIMITIVE_SORT_IDS
FROZEN_LEAVES = tuple(spec.expression_id for spec in LEAF_EXPRESSIONS)
FROZEN_OPERATORS = tuple(
    spec.expression_id
    for spec in (
        UNARY_OPERATORS
        + BINARY_OPERATORS
        + TERNARY_OPERATORS
        + BOOLEAN_COMPOSITION
    )
)
FROZEN_FORBIDDEN_SYMBOLS = FORBIDDEN_FORMS

FROZEN_RATIONAL_GRID_ID = OLD_DSL_V1.rational_grid_id
FROZEN_DSL_SPEC_ID = OLD_DSL_V1.content_id
FROZEN_BOUNDED_UNIVERSE_DIAGNOSTIC_ID = (
    ODD_REDUCTION_TARGET.diagnostic_universe_content_id
)
FROZEN_TARGET_TABLE_DIAGNOSTIC_ID = (
    ODD_REDUCTION_TARGET.diagnostic_target_table_content_id
)
FROZEN_OPERATOR_SEMANTICS_ID = OLD_DSL_V1.operator_semantics_id
FROZEN_EQUIVALENCE_CONTRACT_ID = stable_hash(
    BOTTOM_AND_EQUIVALENCE,
    prefix="equivalence_contract_",
)
FROZEN_MDL_CODE_TABLE_ID = stable_hash(
    {
        "code_table_id": MDL_CODE_TABLE_ID,
        "fixed_point_precision_id": FIXED_POINT_PRECISION_ID,
        "ast_shape_prefixes": AST_SHAPE_PREFIXES,
        "leaf_class_codes": LEAF_CLASS_CODES,
        "unary_token_codes": UNARY_TOKEN_CODES,
        "binary_token_codes": BINARY_TOKEN_CODES,
        "ternary_token_codes": TERNARY_TOKEN_CODES,
        "rational_parameter_codes": RATIONAL_PARAMETER_CODES,
        "tolerance_codes": TOLERANCE_CODES,
        "scope_clause_count_codes": SCOPE_CLAUSE_COUNT_CODES,
        "identifier_code": "Elias-delta over one-based frozen registry index",
        "aggregate_leaf_fields": (
            "AggregateMapId:3",
            "ScopeId:2",
            "QuantityId:1",
            "scope_extension_code",
        ),
        "canonical_ast_schema_id": CANONICAL_AST_SCHEMA_ID,
        "new_reducer_definition": (
            f"NEW_REDUCER_V1 header:16 value=0x{NEW_REDUCER_V1_HEADER:04X}",
            "arity:Elias-delta",
            "input_sort_ids:4_each",
            "output_sort_id:4",
            "reduction_scheme:1",
            "identity_parameter:3",
            "combiner_ast:ordinary_prefix",
            "maximum_set_size:4",
            "scope_code",
            "verifier_hash_reference:256",
        ),
        "binary_data_code": "log2(n+1)+log2(comb(n,k))",
        "minimum_gain": "max(32_bits,ceil_Q32(0.05*old_data_length))",
        "invention_split_rows": (192, 96, 192, 480),
        "decimal_precision": MDL_DECIMAL_PRECISION,
        "rounding": "ceil_to_unsigned_Q32",
        "q32_reference_algorithm_id": Q32_REFERENCE_ALGORITHM_ID,
    },
    prefix="mdl_code_table_",
)
FROZEN_PARITY_TARGET_ID = ODD_REDUCTION_TARGET.content_id
FROZEN_HIDDEN_SINK_CONTROL_ID = OBSERVED_OMITTED_SINK_CONTROL.content_id
FROZEN_HIDDEN_SINK_UNIVERSE_DIAGNOSTIC_ID = (
    OBSERVED_OMITTED_SINK_CONTROL.diagnostic_universe_content_id
)
FROZEN_HIDDEN_SINK_TARGET_TABLE_DIAGNOSTIC_ID = (
    OBSERVED_OMITTED_SINK_CONTROL.diagnostic_target_table_content_id
)
FROZEN_HIDDEN_GENERATOR_SPEC_ID = stable_hash(
    {
        "target": ODD_REDUCTION_TARGET,
        "universe": ODD_REDUCTION_UNIVERSE,
        "fallback_registry": HIDDEN_TARGET_REGISTRY,
        "sink_control": OBSERVED_OMITTED_SINK_CONTROL,
    },
    prefix="hidden_generator_spec_",
)
FROZEN_STRICT_ACCEPTANCE_GATE_ID = DUAL_STRICT_GATE_REPORT_ID

FROZEN_OUTSIDE_CERTIFICATE_IMPLEMENTATION_BLOCKERS = (
    "formal_outside_certificate_issuance_unimplemented",
    "diagnostic_formal_bridge_unimplemented",
    "program_output_archive_replay_unimplemented",
    "python_closure_replay_unimplemented",
    "rust_closure_replay_unimplemented",
    "latest_key_status_resolver_unimplemented",
    "ed25519_three_of_three_envelope_not_issued",
)
FROZEN_FORMAL_MDL_IMPLEMENTATION_BLOCKERS = (
    "formal_mdl_ast_scorer_unimplemented",
    "python_mdl_replay_unimplemented",
    "rust_mdl_replay_unimplemented",
    "mdl_ast_and_new_symbol_wire_replay_unverified",
    "cross_language_q32_log2_replay_unverified",
)


def _require_tuple(value: object, name: str) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be an immutable tuple")


def _require_bool(value: object, name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")


def _require_content_id(value: str | None, name: str, *, optional: bool) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or re.fullmatch(
        r"[a-z][a-z0-9_]*_[0-9a-f]{64}", value
    ) is None:
        raise ValueError(f"{name} must be a content-addressed SHA-256 id")


def _require_fraction(value: object, name: str) -> Fraction:
    if type(value) is not Fraction:
        raise TypeError(f"{name} must be an exact Fraction")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


def _fraction_payload(value: Fraction) -> tuple[int, int]:
    return (value.numerator, value.denominator)


@dataclass(frozen=True, slots=True)
class DslLimits:
    """The exact finite old-language structural and search bounds."""

    maximum_relation_arity: int = 3
    maximum_entity_set_size: int = 8
    maximum_ast_depth: int = 4
    maximum_ast_node_count: int = 7
    maximum_top_level_clauses: int = 3
    maximum_distinct_bit_slots: int = 4
    maximum_aggregate_leaves: int = 1
    maximum_composition_depth: int = 2
    maximum_fitted_parameters: int = 3
    maximum_scope_clauses: int = 2
    maximum_canonical_programs: int = 50_000
    maximum_raw_operator_applications: int = 5_000_000

    def __post_init__(self) -> None:
        for name in self.__slots__:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"DSL limit {name} must be a positive integer")

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="phase3_dsl_limits_")


FROZEN_DSL_LIMITS = DslLimits()

if FROZEN_DSL_LIMITS != DslLimits(
    maximum_relation_arity=3,
    maximum_entity_set_size=8,
    maximum_ast_depth=STRUCTURAL_LIMITS.max_total_ast_depth,
    maximum_ast_node_count=STRUCTURAL_LIMITS.max_total_node_count,
    maximum_top_level_clauses=STRUCTURAL_LIMITS.max_top_level_clauses,
    maximum_distinct_bit_slots=STRUCTURAL_LIMITS.max_distinct_bit_slots,
    maximum_aggregate_leaves=STRUCTURAL_LIMITS.max_aggregate_leaves,
    maximum_composition_depth=STRUCTURAL_LIMITS.max_old_law_composition_depth,
    maximum_fitted_parameters=STRUCTURAL_LIMITS.max_fitted_scalar_parameters,
    maximum_scope_clauses=STRUCTURAL_LIMITS.max_scope_clauses,
    maximum_canonical_programs=CLOSURE_BUDGET.max_canonical_program_count,
    maximum_raw_operator_applications=(
        CLOSURE_BUDGET.max_raw_operator_applications
    ),
):
    raise AssertionError("Phase-3 contract limits drifted from old DSL v1")


class ReadinessBlocker(str, Enum):
    # Specification bindings retained for backwards-compatible wire values.
    # The surface-freeze default resolves these fields; custom missing bindings
    # still fail closed.
    RATIONAL_GRID = "rational_grid_not_frozen"
    DSL_SPEC = "dsl_spec_not_frozen"
    BOUNDED_UNIVERSE = "bounded_universe_diagnostic_id_not_frozen"
    TARGET_TRUTH_TABLE = "target_table_diagnostic_id_not_frozen"
    OPERATOR_SEMANTICS = "operator_semantics_not_frozen"
    EQUIVALENCE_CONTRACT = "equivalence_contract_not_frozen"
    CANONICALIZER = "canonicalizer_implementation_not_bound"
    ENUMERATOR = "enumerator_implementation_not_bound"
    MDL_CODE_TABLE = "mdl_code_table_not_frozen"
    PARITY_TARGET = "parity_target_not_frozen"
    HIDDEN_SINK_CONTROL = "hidden_sink_control_not_frozen"
    HIDDEN_SINK_UNIVERSE = "hidden_sink_universe_diagnostic_id_not_frozen"
    HIDDEN_SINK_TRUTH_TABLE = "hidden_sink_target_table_diagnostic_id_not_frozen"
    HIDDEN_GENERATOR = "hidden_generator_not_frozen"
    CANONICAL_AST_SCHEMA = (
        "strict_acceptance_implementation_not_dual_verified"
    )
    PROGRAM_OUTPUT_ARCHIVE = "program_output_archive_replay_not_implemented"
    PYTHON_CLOSURE_REPLAY = "python_complete_closure_replay_not_implemented"
    RUST_CLOSURE_REPLAY = "rust_complete_closure_replay_not_implemented"
    LATEST_KEY_STATUS = "latest_key_status_resolver_not_implemented"
    FORMAL_MERKLE_ROOTS = (
        "formal_cbor_rfc6962_universe_and_truth_roots_not_generated"
    )
    CAPACITY_CLASSIFICATION = (
        "conditional_capacity_lower_bound_requires_strict_dual_replay"
    )
    SEALED_CLOSURE_VERIFIER = "sealed_closure_verifier_not_implemented"


_BLOCKER_FIELDS = (
    (ReadinessBlocker.RATIONAL_GRID, "rational_grid_id"),
    (ReadinessBlocker.DSL_SPEC, "dsl_spec_id"),
    (
        ReadinessBlocker.BOUNDED_UNIVERSE,
        "bounded_universe_diagnostic_id",
    ),
    (ReadinessBlocker.TARGET_TRUTH_TABLE, "target_table_diagnostic_id"),
    (ReadinessBlocker.OPERATOR_SEMANTICS, "operator_semantics_id"),
    (ReadinessBlocker.EQUIVALENCE_CONTRACT, "equivalence_contract_id"),
    (ReadinessBlocker.CANONICALIZER, "canonicalizer_implementation_id"),
    (ReadinessBlocker.ENUMERATOR, "enumerator_implementation_id"),
    (ReadinessBlocker.MDL_CODE_TABLE, "mdl_code_table_id"),
    (ReadinessBlocker.PARITY_TARGET, "parity_target_id"),
    (ReadinessBlocker.HIDDEN_SINK_CONTROL, "hidden_sink_control_id"),
    (
        ReadinessBlocker.HIDDEN_SINK_UNIVERSE,
        "hidden_sink_universe_diagnostic_id",
    ),
    (
        ReadinessBlocker.HIDDEN_SINK_TRUTH_TABLE,
        "hidden_sink_target_table_diagnostic_id",
    ),
    (ReadinessBlocker.HIDDEN_GENERATOR, "hidden_generator_spec_id"),
)


@dataclass(frozen=True, slots=True)
class Phase3PrerequisiteContract:
    """Content-addressed freeze manifest for the Phase-3 preregistration.

    Surface, diagnostic, and strict-canonicalizer gate IDs are resolved in the
    default instance.  Formal CBOR/Merkle roots and the complete-enumerator
    implementation ID remain unresolved; merely naming a primitive or banning
    ``parity`` cannot fill them.
    """

    schema_version: str = PHASE3_CONTRACT_SCHEMA_VERSION
    freeze_version: str = PHASE3_OVERALL_FREEZE_VERSION
    sorts: tuple[str, ...] = FROZEN_SORTS
    leaves: tuple[str, ...] = FROZEN_LEAVES
    operators: tuple[str, ...] = FROZEN_OPERATORS
    forbidden_symbols: tuple[str, ...] = FROZEN_FORBIDDEN_SYMBOLS
    limits: DslLimits = FROZEN_DSL_LIMITS
    rational_grid_id: str | None = FROZEN_RATIONAL_GRID_ID
    dsl_spec_id: str | None = FROZEN_DSL_SPEC_ID
    bounded_universe_diagnostic_id: str | None = (
        FROZEN_BOUNDED_UNIVERSE_DIAGNOSTIC_ID
    )
    target_table_diagnostic_id: str | None = FROZEN_TARGET_TABLE_DIAGNOSTIC_ID
    operator_semantics_id: str | None = FROZEN_OPERATOR_SEMANTICS_ID
    equivalence_contract_id: str | None = FROZEN_EQUIVALENCE_CONTRACT_ID
    canonicalizer_implementation_id: str | None = FROZEN_STRICT_ACCEPTANCE_GATE_ID
    enumerator_implementation_id: str | None = None
    mdl_code_table_id: str | None = FROZEN_MDL_CODE_TABLE_ID
    parity_target_id: str | None = FROZEN_PARITY_TARGET_ID
    hidden_sink_control_id: str | None = FROZEN_HIDDEN_SINK_CONTROL_ID
    hidden_sink_universe_diagnostic_id: str | None = (
        FROZEN_HIDDEN_SINK_UNIVERSE_DIAGNOSTIC_ID
    )
    hidden_sink_target_table_diagnostic_id: str | None = (
        FROZEN_HIDDEN_SINK_TARGET_TABLE_DIAGNOSTIC_ID
    )
    hidden_generator_spec_id: str | None = FROZEN_HIDDEN_GENERATOR_SPEC_ID
    shadow_only: bool = True
    active_promotion_authorized: bool = False

    def __post_init__(self) -> None:
        for name in ("sorts", "leaves", "operators", "forbidden_symbols"):
            _require_tuple(getattr(self, name), f"Phase-3 {name}")
        if self.schema_version != PHASE3_CONTRACT_SCHEMA_VERSION:
            raise ValueError("unknown Phase-3 preregistration schema version")
        if self.freeze_version != PHASE3_OVERALL_FREEZE_VERSION:
            raise ValueError("overall Phase-2B/Phase-3 freeze version drift")
        frozen_fields = (
            ("sorts", self.sorts, FROZEN_SORTS),
            ("leaves", self.leaves, FROZEN_LEAVES),
            ("operators", self.operators, FROZEN_OPERATORS),
            (
                "forbidden symbols",
                self.forbidden_symbols,
                FROZEN_FORBIDDEN_SYMBOLS,
            ),
            ("DSL limits", self.limits, FROZEN_DSL_LIMITS),
        )
        for name, actual, expected in frozen_fields:
            if actual != expected:
                raise ValueError(f"{name} differ from the frozen Phase-3 decision")
        frozen_bindings = (
            ("rational grid", self.rational_grid_id, FROZEN_RATIONAL_GRID_ID),
            ("DSL spec", self.dsl_spec_id, FROZEN_DSL_SPEC_ID),
            (
                "bounded universe",
                self.bounded_universe_diagnostic_id,
                FROZEN_BOUNDED_UNIVERSE_DIAGNOSTIC_ID,
            ),
            (
                "target truth table diagnostic content",
                self.target_table_diagnostic_id,
                FROZEN_TARGET_TABLE_DIAGNOSTIC_ID,
            ),
            (
                "operator semantics",
                self.operator_semantics_id,
                FROZEN_OPERATOR_SEMANTICS_ID,
            ),
            (
                "equivalence contract",
                self.equivalence_contract_id,
                FROZEN_EQUIVALENCE_CONTRACT_ID,
            ),
            ("MDL code table", self.mdl_code_table_id, FROZEN_MDL_CODE_TABLE_ID),
            ("odd target", self.parity_target_id, FROZEN_PARITY_TARGET_ID),
            (
                "hidden-sink control",
                self.hidden_sink_control_id,
                FROZEN_HIDDEN_SINK_CONTROL_ID,
            ),
            (
                "hidden-sink universe",
                self.hidden_sink_universe_diagnostic_id,
                FROZEN_HIDDEN_SINK_UNIVERSE_DIAGNOSTIC_ID,
            ),
            (
                "hidden-sink truth table diagnostic content",
                self.hidden_sink_target_table_diagnostic_id,
                FROZEN_HIDDEN_SINK_TARGET_TABLE_DIAGNOSTIC_ID,
            ),
            (
                "hidden generator spec",
                self.hidden_generator_spec_id,
                FROZEN_HIDDEN_GENERATOR_SPEC_ID,
            ),
        )
        for name, actual, expected in frozen_bindings:
            if actual != expected:
                raise ValueError(f"{name} differs from the frozen Phase-3 surface")
        if len(set(self.sorts)) != len(self.sorts):
            raise ValueError("Phase-3 sorts must be unique")
        if len(set(self.leaves)) != len(self.leaves):
            raise ValueError("Phase-3 leaves must be unique")
        if len(set(self.operators)) != len(self.operators):
            raise ValueError("Phase-3 operators must be unique")
        if len(set(self.forbidden_symbols)) != len(self.forbidden_symbols):
            raise ValueError("Phase-3 forbidden symbols must be unique")
        for _, field_name in _BLOCKER_FIELDS:
            _require_content_id(
                getattr(self, field_name),
                field_name,
                optional=True,
            )
        _require_bool(self.shadow_only, "shadow_only")
        _require_bool(
            self.active_promotion_authorized,
            "active_promotion_authorized",
        )
        if not self.shadow_only or self.active_promotion_authorized:
            raise ValueError("Phase 2-3 must remain shadow-only and ACTIVE-disabled")

    @property
    def readiness_blockers(self) -> tuple[ReadinessBlocker, ...]:
        blockers = tuple(
            blocker
            for blocker, field_name in _BLOCKER_FIELDS
            if getattr(self, field_name) is None
        )
        if not STRICT_ACCEPTANCE_IMPLEMENTATION_VERIFIED:
            blockers += (ReadinessBlocker.CANONICAL_AST_SCHEMA,)
        if not PROGRAM_OUTPUT_ARCHIVE_REPLAY_IMPLEMENTED:
            blockers += (ReadinessBlocker.PROGRAM_OUTPUT_ARCHIVE,)
        if not PYTHON_CLOSURE_REPLAY_IMPLEMENTED:
            blockers += (ReadinessBlocker.PYTHON_CLOSURE_REPLAY,)
        if not RUST_CLOSURE_REPLAY_IMPLEMENTED:
            blockers += (ReadinessBlocker.RUST_CLOSURE_REPLAY,)
        if not LATEST_KEY_STATUS_RESOLVER_IMPLEMENTED:
            blockers += (ReadinessBlocker.LATEST_KEY_STATUS,)
        blockers += (ReadinessBlocker.FORMAL_MERKLE_ROOTS,)
        if not SEALED_CLOSURE_VERIFIER_IMPLEMENTED:
            blockers += (ReadinessBlocker.SEALED_CLOSURE_VERIFIER,)
        return blockers

    @property
    def ready_for_outside_certificate(self) -> bool:
        return not self.readiness_blockers

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="phase3_preregistration_")


DEFAULT_PHASE3_PREREGISTRATION = Phase3PrerequisiteContract()


class AdequacyVerdict(str, Enum):
    IN_LANGUAGE = "in_language"
    OUTSIDE_FROZEN_CLOSURE = "outside_frozen_closure"
    INCONCLUSIVE_BUDGET = "inconclusive_budget"
    INCONCLUSIVE_SEMANTICS = "inconclusive_semantics"


class TargetRole(str, Enum):
    OUTSIDE_TARGET = "outside_target"
    IN_LANGUAGE_NULL = "in_language_null_control"


class ClosureRunStatus(str, Enum):
    COMPLETE = "COMPLETE"
    DSL_TOO_LARGE = "DSL_TOO_LARGE"
    INCONCLUSIVE_BUDGET = "INCONCLUSIVE_BUDGET"
    INCONCLUSIVE_SEMANTICS = "INCONCLUSIVE_SEMANTICS"


@dataclass(frozen=True, slots=True)
class ClosureEnumerationReceipt:
    """Untrusted replay-claim wire record for one target.

    The fields are structurally validated but not recomputed.  Until a sealed
    closure verifier exists, this object cannot establish either a semantic
    match or completeness of a no-match search.
    """

    contract_id: str
    dsl_spec_id: str
    target_id: str
    target_role: TargetRole
    bounded_universe_diagnostic_id: str
    operator_semantics_id: str
    equivalence_contract_id: str
    enumerator_implementation_id: str
    search_budget: int
    enumerated_canonical_program_count: int
    raw_operator_application_count: int
    closure_cardinality: int | None
    closure_status: ClosureRunStatus
    frontier_exhausted: bool
    all_type_buckets_closed: bool
    raw_expansion_limit_hit: bool
    wall_clock_abort_hit: bool
    first_out_of_budget_program_id: str | None
    semantics_total: bool
    extensional_match_program_ids: tuple[str, ...]
    closure_root: str
    target_table_diagnostic_id: str

    def __post_init__(self) -> None:
        for name in (
            "contract_id",
            "dsl_spec_id",
            "target_id",
            "bounded_universe_diagnostic_id",
            "operator_semantics_id",
            "equivalence_contract_id",
            "enumerator_implementation_id",
            "closure_root",
            "target_table_diagnostic_id",
        ):
            _require_content_id(getattr(self, name), name, optional=False)
        if not isinstance(self.target_role, TargetRole):
            raise TypeError("target_role must be a TargetRole")
        if not isinstance(self.closure_status, ClosureRunStatus):
            raise TypeError("closure_status must be a ClosureRunStatus")
        if self.search_budget != FROZEN_DSL_LIMITS.maximum_canonical_programs:
            raise ValueError("closure receipt changed the frozen search budget")
        if (
            type(self.enumerated_canonical_program_count) is not int
            or self.enumerated_canonical_program_count < 0
            or self.enumerated_canonical_program_count > self.search_budget
        ):
            raise ValueError("enumerated program count is outside the frozen budget")
        if (
            type(self.raw_operator_application_count) is not int
            or self.raw_operator_application_count < 0
            or self.raw_operator_application_count
            > FROZEN_DSL_LIMITS.maximum_raw_operator_applications
        ):
            raise ValueError("raw operator application count is outside the frozen budget")
        if self.closure_cardinality is not None and (
            type(self.closure_cardinality) is not int
            or self.closure_cardinality < 0
        ):
            raise ValueError("closure cardinality must be a nonnegative integer")
        for name in (
            "frontier_exhausted",
            "all_type_buckets_closed",
            "raw_expansion_limit_hit",
            "wall_clock_abort_hit",
            "semantics_total",
        ):
            _require_bool(getattr(self, name), name)
        _require_content_id(
            self.first_out_of_budget_program_id,
            "first out-of-budget program id",
            optional=True,
        )
        _require_tuple(
            self.extensional_match_program_ids,
            "extensional match program ids",
        )
        for match_id in self.extensional_match_program_ids:
            _require_content_id(match_id, "match program id", optional=False)
        if len(set(self.extensional_match_program_ids)) != len(
            self.extensional_match_program_ids
        ):
            raise ValueError("closure receipt repeats an extensional match")
        if self.extensional_match_program_ids != tuple(
            sorted(self.extensional_match_program_ids)
        ):
            raise ValueError("extensional match ids must use canonical order")
        if self.closure_status is ClosureRunStatus.COMPLETE:
            if not self.semantics_total:
                raise ValueError("COMPLETE requires total semantics")
            if not self.frontier_exhausted or not self.all_type_buckets_closed:
                raise ValueError("COMPLETE requires an exhausted, closed frontier")
            if self.raw_expansion_limit_hit or self.wall_clock_abort_hit:
                raise ValueError("COMPLETE cannot carry an execution abort")
            if self.closure_cardinality != self.enumerated_canonical_program_count:
                raise ValueError("complete receipt must bind the full closure cardinality")
            if self.first_out_of_budget_program_id is not None:
                raise ValueError("COMPLETE cannot carry an out-of-budget witness")
        elif self.closure_status is ClosureRunStatus.DSL_TOO_LARGE:
            if not self.semantics_total:
                raise ValueError("DSL_TOO_LARGE requires total admitted-program semantics")
            if self.enumerated_canonical_program_count != self.search_budget:
                raise ValueError("DSL_TOO_LARGE requires 50,000 accepted programs")
            if self.first_out_of_budget_program_id is None:
                raise ValueError("DSL_TOO_LARGE requires the 50,001st program id")
            if self.frontier_exhausted or self.all_type_buckets_closed:
                raise ValueError("DSL_TOO_LARGE cannot claim a closed frontier")
            if self.raw_expansion_limit_hit or self.wall_clock_abort_hit:
                raise ValueError("DSL_TOO_LARGE is distinct from execution abort")
            if self.closure_cardinality is not None:
                raise ValueError("DSL_TOO_LARGE cannot claim closure cardinality")
        elif self.closure_status is ClosureRunStatus.INCONCLUSIVE_BUDGET:
            if not self.semantics_total:
                raise ValueError(
                    "semantic failure must use INCONCLUSIVE_SEMANTICS"
                )
            if not (self.raw_expansion_limit_hit or self.wall_clock_abort_hit):
                raise ValueError("INCONCLUSIVE_BUDGET requires a frozen budget abort")
            if (
                self.raw_expansion_limit_hit
                and self.raw_operator_application_count
                != FROZEN_DSL_LIMITS.maximum_raw_operator_applications
            ):
                raise ValueError(
                    "raw expansion limit requires exactly 5,000,000 applications"
                )
            if self.frontier_exhausted or self.all_type_buckets_closed:
                raise ValueError("INCONCLUSIVE_BUDGET cannot claim a closed frontier")
            if self.first_out_of_budget_program_id is not None:
                raise ValueError("raw-budget failure cannot claim DSL_TOO_LARGE")
            if self.closure_cardinality is not None:
                raise ValueError("incomplete receipt cannot claim closure cardinality")
        else:
            if self.semantics_total:
                raise ValueError("INCONCLUSIVE_SEMANTICS requires partial semantics")
            if self.frontier_exhausted or self.all_type_buckets_closed:
                raise ValueError(
                    "INCONCLUSIVE_SEMANTICS cannot claim a closed frontier"
                )
            if self.raw_expansion_limit_hit or self.wall_clock_abort_hit:
                raise ValueError(
                    "semantic failure cannot also claim a budget abort"
                )
            if self.closure_cardinality is not None:
                raise ValueError("semantic failure cannot claim closure cardinality")
            if self.first_out_of_budget_program_id is not None:
                raise ValueError("semantic failure cannot claim DSL_TOO_LARGE")

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="phase3_closure_receipt_")


@dataclass(frozen=True, slots=True)
class ClosureAssessment:
    contract_id: str
    receipt_id: str
    verdict: AdequacyVerdict
    reason: str
    outside_certificate_id: str | None = None
    shadow_only: bool = True
    active_promotion_authorized: bool = False

    def __post_init__(self) -> None:
        _require_content_id(self.contract_id, "assessment contract id", optional=False)
        _require_content_id(self.receipt_id, "assessment receipt id", optional=False)
        if not isinstance(self.verdict, AdequacyVerdict):
            raise TypeError("assessment verdict must be an AdequacyVerdict")
        if not self.reason:
            raise ValueError("closure assessment needs an auditable reason")
        _require_content_id(
            self.outside_certificate_id,
            "outside certificate id",
            optional=True,
        )
        if self.verdict is AdequacyVerdict.OUTSIDE_FROZEN_CLOSURE:
            raise RuntimeError(
                "formal outside assessments cannot be directly constructed; "
                "sealed certificate issuance is not implemented"
            )
        elif self.outside_certificate_id is not None:
            raise ValueError("only an outside verdict may carry an outside certificate")
        if not self.shadow_only or self.active_promotion_authorized:
            raise ValueError("Phase-3 assessment cannot authorize ACTIVE promotion")

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="phase3_closure_assessment_")


def _assessment(
    contract: Phase3PrerequisiteContract,
    receipt: ClosureEnumerationReceipt,
    verdict: AdequacyVerdict,
    reason: str,
) -> ClosureAssessment:
    if verdict is AdequacyVerdict.OUTSIDE_FROZEN_CLOSURE:
        raise RuntimeError(
            "outside-certificate issuance is unavailable until the sealed "
            "closure verifier replays archives and recomputes their roots"
        )
    return ClosureAssessment(
        contract_id=contract.content_id,
        receipt_id=receipt.content_id,
        verdict=verdict,
        reason=reason,
        outside_certificate_id=None,
    )


def assess_closure(
    contract: Phase3PrerequisiteContract,
    receipt: ClosureEnumerationReceipt,
) -> ClosureAssessment:
    """Derive a fail-closed adequacy verdict from a frozen receipt.

    The current receipt is untrusted input.  Structural inconsistencies are
    classified, but no semantic verdict is issued until a sealed evaluator can
    replay the archive and recompute the program and target roots.
    """

    expected_universe = (
        contract.bounded_universe_diagnostic_id
        if receipt.target_role is TargetRole.OUTSIDE_TARGET
        else contract.hidden_sink_universe_diagnostic_id
    )
    expected_truth_table = (
        contract.target_table_diagnostic_id
        if receipt.target_role is TargetRole.OUTSIDE_TARGET
        else contract.hidden_sink_target_table_diagnostic_id
    )
    bindings = (
        ("contract", receipt.contract_id, contract.content_id),
        ("dsl_spec", receipt.dsl_spec_id, contract.dsl_spec_id),
        (
            "bounded_universe",
            receipt.bounded_universe_diagnostic_id,
            expected_universe,
        ),
        (
            "target_truth_table",
            receipt.target_table_diagnostic_id,
            expected_truth_table,
        ),
        (
            "operator_semantics",
            receipt.operator_semantics_id,
            contract.operator_semantics_id,
        ),
        (
            "equivalence_contract",
            receipt.equivalence_contract_id,
            contract.equivalence_contract_id,
        ),
        (
            "enumerator",
            receipt.enumerator_implementation_id,
            contract.enumerator_implementation_id,
        ),
    )
    for name, actual, expected in bindings:
        if expected is None or actual != expected:
            return _assessment(
                contract,
                receipt,
                AdequacyVerdict.INCONCLUSIVE_SEMANTICS,
                f"{name}_binding_missing_or_mismatched",
            )

    expected_target = (
        contract.parity_target_id
        if receipt.target_role is TargetRole.OUTSIDE_TARGET
        else contract.hidden_sink_control_id
    )
    if expected_target is None or receipt.target_id != expected_target:
        return _assessment(
            contract,
            receipt,
            AdequacyVerdict.INCONCLUSIVE_SEMANTICS,
            "target_binding_missing_or_mismatched",
        )
    if not SEALED_CLOSURE_VERIFIER_IMPLEMENTED:
        return _assessment(
            contract,
            receipt,
            AdequacyVerdict.INCONCLUSIVE_SEMANTICS,
            ReadinessBlocker.SEALED_CLOSURE_VERIFIER.value,
        )
    if contract.readiness_blockers:
        blockers = ",".join(blocker.value for blocker in contract.readiness_blockers)
        return _assessment(
            contract,
            receipt,
            AdequacyVerdict.INCONCLUSIVE_SEMANTICS,
            "readiness_blockers:" + blockers,
        )
    if not receipt.semantics_total:
        return _assessment(
            contract,
            receipt,
            AdequacyVerdict.INCONCLUSIVE_SEMANTICS,
            "target_or_program_semantics_are_partial",
        )
    if receipt.closure_status is ClosureRunStatus.DSL_TOO_LARGE:
        return _assessment(
            contract,
            receipt,
            AdequacyVerdict.INCONCLUSIVE_BUDGET,
            "dsl_too_large_requires_new_dsl_version",
        )
    if receipt.closure_status is ClosureRunStatus.INCONCLUSIVE_BUDGET:
        return _assessment(
            contract,
            receipt,
            AdequacyVerdict.INCONCLUSIVE_BUDGET,
            "frozen_raw_expansion_or_wall_clock_budget_exhausted",
        )
    if receipt.closure_status is ClosureRunStatus.INCONCLUSIVE_SEMANTICS:
        return _assessment(
            contract,
            receipt,
            AdequacyVerdict.INCONCLUSIVE_SEMANTICS,
            "closure_run_reported_partial_semantics",
        )
    if receipt.extensional_match_program_ids:
        return _assessment(
            contract,
            receipt,
            AdequacyVerdict.IN_LANGUAGE,
            "full_bounded_truth_table_has_an_old_language_match",
        )
    if receipt.target_role is TargetRole.IN_LANGUAGE_NULL:
        return _assessment(
            contract,
            receipt,
            AdequacyVerdict.INCONCLUSIVE_SEMANTICS,
            "preregistered_in_language_null_has_no_old_language_match",
        )
    return _assessment(
        contract,
        receipt,
        AdequacyVerdict.OUTSIDE_FROZEN_CLOSURE,
        "complete_frozen_closure_has_no_extensional_match",
    )


@dataclass(frozen=True, slots=True)
class MdlGainReceipt:
    """Exact arithmetic record over caller-supplied code-length claims.

    The logarithmic enumerative-code implementation is intentionally not
    invented here: it is one of the preregistration blockers.  Once a code
    table and sealed scorer exist, recomputed bit lengths may enter as
    ``Fraction`` values, avoiding float-dependent boundary decisions.  The
    present record alone does not attest that recomputation.
    """

    mdl_code_table_id: str
    scoring_partition_id: str
    old_program_length_bits: Fraction
    old_data_length_bits: Fraction
    new_program_length_bits: Fraction
    new_data_length_bits: Fraction

    def __post_init__(self) -> None:
        _require_content_id(
            self.mdl_code_table_id,
            "MDL code table id",
            optional=False,
        )
        _require_content_id(
            self.scoring_partition_id,
            "MDL scoring partition id",
            optional=False,
        )
        for name in (
            "old_program_length_bits",
            "old_data_length_bits",
            "new_program_length_bits",
            "new_data_length_bits",
        ):
            _require_fraction(getattr(self, name), name)

    @property
    def old_total_length_bits(self) -> Fraction:
        return self.old_program_length_bits + self.old_data_length_bits

    @property
    def new_total_length_bits(self) -> Fraction:
        return self.new_program_length_bits + self.new_data_length_bits

    @property
    def compression_gain_bits(self) -> Fraction:
        return self.old_total_length_bits - self.new_total_length_bits

    @property
    def minimum_required_gain_bits(self) -> Fraction:
        return max(Fraction(32, 1), self.old_data_length_bits / 20)

    @property
    def numeric_threshold_passed(self) -> bool:
        return self.compression_gain_bits >= self.minimum_required_gain_bits

    @property
    def content_id(self) -> str:
        return stable_hash(
            {
                "mdl_code_table_id": self.mdl_code_table_id,
                "scoring_partition_id": self.scoring_partition_id,
                "old_program_length_bits": _fraction_payload(
                    self.old_program_length_bits
                ),
                "old_data_length_bits": _fraction_payload(self.old_data_length_bits),
                "new_program_length_bits": _fraction_payload(
                    self.new_program_length_bits
                ),
                "new_data_length_bits": _fraction_payload(self.new_data_length_bits),
            },
            prefix="phase3_mdl_gain_receipt_",
        )


def mdl_gain_gate(
    contract: Phase3PrerequisiteContract,
    receipt: MdlGainReceipt,
) -> bool:
    """Return the formal MDL gate, currently hard-disabled.

    ``MdlGainReceipt.numeric_threshold_passed`` is only an exact arithmetic
    precheck over caller-supplied lengths.  The formal gate remains false until
    a sealed scorer recomputes those lengths from a frozen partition and code
    table.
    """

    return (
        SEALED_MDL_SCORER_IMPLEMENTED
        and contract.mdl_code_table_id is not None
        and receipt.mdl_code_table_id == contract.mdl_code_table_id
        and receipt.numeric_threshold_passed
    )


def xor2_via_absolute_difference(left: int, right: int) -> int:
    """Evaluate XOR2 under intended bit/absolute-difference arithmetic."""

    if type(left) is not int or type(right) is not int:
        raise TypeError("xor2 sanity inputs must be integer bits")
    if left not in (0, 1) or right not in (0, 1):
        raise ValueError("xor2 sanity inputs must belong to {0, 1}")
    return int(abs(Fraction(left, 1) - Fraction(right, 1)))


@dataclass(frozen=True, slots=True)
class Xor2SanityWitness:
    """Mathematical target-design sanity under intended numeric semantics.

    The operator truth conditions are frozen, but this witness has not been
    admitted by the strict canonicalizer or replayed by both complete closure
    implementations.  It therefore cannot serve as a formal ``IN_LANGUAGE``
    receipt.
    """

    expression: str
    operator_ids: tuple[str, ...]
    truth_table: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        _require_tuple(self.operator_ids, "xor2 witness operator ids")
        _require_tuple(self.truth_table, "xor2 witness truth table")
        if not self.expression:
            raise ValueError("xor2 witness needs a descriptive expression")
        if not self.operator_ids or not set(self.operator_ids).issubset(
            FROZEN_OPERATORS
        ):
            raise ValueError("xor2 witness may use only frozen old-language operators")
        if any(symbol in self.operator_ids for symbol in FROZEN_FORBIDDEN_SYMBOLS):
            raise ValueError("xor2 witness illegally names a forbidden primitive")
        expected = tuple(
            (left, right, xor2_via_absolute_difference(left, right))
            for left in (0, 1)
            for right in (0, 1)
        )
        if self.truth_table != expected:
            raise ValueError("xor2 witness truth table is not exact")

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="phase3_xor2_sanity_")


XOR2_ABSOLUTE_DIFFERENCE_WITNESS = Xor2SanityWitness(
    expression=BINARY_XOR_SANITY.type_explicit_candidate_old_dsl_program,
    operator_ids=("bit_to_scalar", "difference", "absolute"),
    truth_table=tuple(
        (left, right, xor2_via_absolute_difference(left, right))
        for left in (0, 1)
        for right in (0, 1)
    ),
)


def phase3_preregistration_report() -> dict[str, object]:
    """Return a deterministic readiness report, never a Phase-3A result."""

    contract = DEFAULT_PHASE3_PREREGISTRATION
    payload: dict[str, object] = {
        "artifact": "phase3_preregistration_readiness_v1",
        "schema_version": contract.schema_version,
        "contract_id": contract.content_id,
        "overall_freeze_version": contract.freeze_version,
        "implementation_id": (
            "phase3_contract_source_sha256_"
            + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        ),
        "status": "STRICT_GATE_VERIFIED_DSL_TOO_LARGE_SHRINK_REQUIRED",
        "inherited_surface_freeze_version": (
            PHASE3_INHERITED_SURFACE_FREEZE_VERSION
        ),
        "formal_phase3a_claim": False,
        "executed_closure_status": DSL_TOO_LARGE_STATUS,
        "formal_roots": None,
        "formal_cbor_archive": False,
        "dsl_too_large_claim_allowed": True,
        "target_synthesis_allowed": TARGET_SYNTHESIS_ALLOWED,
        "outside_certificate_allowed": OUTSIDE_CERTIFICATE_ALLOWED,
        "mdl_certificate_allowed": MDL_CERTIFICATE_ALLOWED,
        "phase2b_formal_exit": PHASE2B_FORMAL_EXIT,
        "active_promotion_allowed": ACTIVE_PROMOTION_ALLOWED,
        "unbounded_outside_language_certificate_issued": False,
        "outside_frozen_closure_certificate_issued": False,
        "unbounded_outside_language_claim_prohibited": True,
        "only_authorized_future_claim": (
            "OUTSIDE_FROZEN_CLOSURE(dsl_version, universe_root, target_root, "
            "exact_extensional)"
        ),
        "ready_for_outside_certificate": contract.ready_for_outside_certificate,
        "closure_receipt_semantics_replayed": False,
        "sealed_closure_verifier_implemented": (
            SEALED_CLOSURE_VERIFIER_IMPLEMENTED
        ),
        "mdl_numeric_threshold_is_formal_gate": False,
        "sealed_mdl_scorer_implemented": SEALED_MDL_SCORER_IMPLEMENTED,
        "surface_parameter_freeze_complete": SURFACE_PARAMETER_FREEZE_COMPLETE,
        "strict_acceptance_contract_complete": (
            STRICT_ACCEPTANCE_CONTRACT_COMPLETE
        ),
        "strict_acceptance_specification_complete": (
            STRICT_ACCEPTANCE_CONTRACT_COMPLETE
        ),
        "normative_parameter_freeze_complete": (
            NORMATIVE_PARAMETER_FREEZE_COMPLETE
        ),
        "strict_acceptance_implementation_verified": (
            STRICT_ACCEPTANCE_IMPLEMENTATION_VERIFIED
        ),
        "formal_root_generation_allowed": FORMAL_ROOT_GENERATION_ALLOWED,
        "specification_resolution_blockers": [],
        "strict_acceptance_specification": {
            "canonical_cbor_profile_id": CANONICAL_CBOR_PROFILE_ID,
            "canonical_ast_schema_id": CANONICAL_AST_SCHEMA_ID,
            "normative_backend": "PROJECT_MINIMAL_ENCODER",
            "implicit_bit_to_rational_coercion": False,
            "scope_alias_is_source_only": True,
            "certificate_groups_resolved_in_spec": [
                f"CERT_{index:02d}" for index in range(1, 10)
            ],
            "implementation_verified": True,
            "gate_status": "VERIFIED",
            "gate_report_id": DUAL_STRICT_GATE_REPORT_ID,
            "python_golden_vectors_passed": 48,
            "rust_golden_vectors_passed": 48,
        },
        "certificate_specification_ready": True,
        "certificate_implementation_ready": False,
        "certificate_issuance_ready": False,
        "outside_certificate_capability_blockers": list(
            FROZEN_OUTSIDE_CERTIFICATE_IMPLEMENTATION_BLOCKERS
        ),
        "formal_mdl_capability_blockers": list(
            FROZEN_FORMAL_MDL_IMPLEMENTATION_BLOCKERS
        ),
        "readiness_blockers": [
            blocker.value for blocker in contract.readiness_blockers
        ],
        "resolved_content_bindings": {
            "rational_grid_id": contract.rational_grid_id,
            "dsl_spec_id": contract.dsl_spec_id,
            "bounded_universe_diagnostic_id": (
                contract.bounded_universe_diagnostic_id
            ),
            "target_table_diagnostic_id": contract.target_table_diagnostic_id,
            "operator_semantics_id": contract.operator_semantics_id,
            "equivalence_contract_id": contract.equivalence_contract_id,
            "mdl_code_table_id": contract.mdl_code_table_id,
            "odd_reduction_target_id": contract.parity_target_id,
            "hidden_sink_control_id": contract.hidden_sink_control_id,
            "hidden_sink_universe_diagnostic_id": (
                contract.hidden_sink_universe_diagnostic_id
            ),
            "hidden_sink_target_table_diagnostic_id": (
                contract.hidden_sink_target_table_diagnostic_id
            ),
            "hidden_generator_spec_id": contract.hidden_generator_spec_id,
            "strict_acceptance_gate_report_id": (
                FROZEN_STRICT_ACCEPTANCE_GATE_ID
            ),
        },
        "unresolved_implementation_bindings": {
            "enumerator_implementation_id": contract.enumerator_implementation_id,
        },
        "frozen_limits": {
            "maximum_relation_arity": contract.limits.maximum_relation_arity,
            "maximum_entity_set_size": contract.limits.maximum_entity_set_size,
            "maximum_ast_depth": contract.limits.maximum_ast_depth,
            "maximum_ast_node_count": contract.limits.maximum_ast_node_count,
            "maximum_top_level_clauses": (
                contract.limits.maximum_top_level_clauses
            ),
            "maximum_distinct_bit_slots": (
                contract.limits.maximum_distinct_bit_slots
            ),
            "maximum_aggregate_leaves": (
                contract.limits.maximum_aggregate_leaves
            ),
            "maximum_composition_depth": (
                contract.limits.maximum_composition_depth
            ),
            "maximum_fitted_parameters": (
                contract.limits.maximum_fitted_parameters
            ),
            "maximum_scope_clauses": contract.limits.maximum_scope_clauses,
            "maximum_canonical_programs": (
                contract.limits.maximum_canonical_programs
            ),
            "maximum_raw_operator_applications": (
                contract.limits.maximum_raw_operator_applications
            ),
        },
        "old_dsl_freeze": {
            "dsl_version": OLD_DSL_V1.dsl_version,
            "dsl_spec_id": OLD_DSL_V1.content_id,
            "primitive_sort_count": len(OLD_DSL_V1.primitive_domains),
            "rational_value_cardinality": 663,
            "scope_count": len(OLD_DSL_V1.scope_catalog),
            "aggregate_count": len(AGGREGATE_CATALOG),
            "transform_count": len(OLD_DSL_V1.transform_catalog),
            "equivalence": "exact_extensional",
            "bottom_is_observable": False,
            "shrink_order": [step.operation for step in SHRINK_ORDER],
        },
        "closure_capacity_preflight": {
            "status": DSL_TOO_LARGE_STATUS,
            "executed_closure_status": DSL_TOO_LARGE_STATUS,
            "capacity_condition_discharged": True,
            "candidate_ast_lower_bound": (
                CAPACITY_PROOF.witness_candidate_ast_count
            ),
            "accepted_unique_canonical_ast_count": (
                CAPACITY_PROOF.witness_candidate_ast_count
            ),
            "canonical_program_budget": CAPACITY_PROOF.canonical_program_budget,
            "strict_rewrite_application_pending": False,
            "strict_canonicalizer_acceptance_verified": True,
            "dual_strict_capacity_replay_complete": True,
            "dual_strict_capacity_replay_report_id": (
                DUAL_STRICT_CAPACITY_REPLAY_REPORT_ID
            ),
            "accepted_set_commitment": STRICT_CAPACITY_SET_COMMITMENT,
            "accepted_set_commitment_is_formal_root": False,
            "first_out_of_budget_ordinal": 50_001,
            "first_out_of_budget_ast_hash": FIRST_OUT_OF_BUDGET_AST_HASH,
            "complete_closure_enumerated": False,
            "formal_archive_roots_generated": False,
            "dsl_too_large_claim_allowed": True,
            "conclusion": (
                "Independent Python and Rust strict replay accepted the same "
                "64,680 unique canonical ASTs and the same 50,001st witness. "
                "The bounded old DSL is DSL_TOO_LARGE, not COMPLETE."
            ),
        },
        "target_freeze": {
            "target_id": ODD_REDUCTION_TARGET.target_id,
            "target_spec_id": ODD_REDUCTION_TARGET.content_id,
            "diagnostic_universe_content_id": (
                ODD_REDUCTION_TARGET.diagnostic_universe_content_id
            ),
            "diagnostic_target_table_content_id": (
                ODD_REDUCTION_TARGET.diagnostic_target_table_content_id
            ),
            "formal_bounded_universe_root": None,
            "formal_target_truth_table_root": None,
            "universe_rows": ODD_REDUCTION_TARGET.universe_rows,
            "set_sizes": list(ODD_REDUCTION_TARGET.set_sizes),
            "discovery_rows": sum(
                split.discovery_train for split in ODD_REDUCTION_TARGET.splits
            ),
            "validation_rows": sum(
                split.validation for split in ODD_REDUCTION_TARGET.splits
            ),
            "sealed_prediction_rows": sum(
                split.sealed_prediction for split in ODD_REDUCTION_TARGET.splits
            ),
            "fallback_registry_size": len(HIDDEN_TARGET_REGISTRY),
            "formal_closure_verdict": None,
        },
        "xor2_sanity": {
            "status": BINARY_XOR_SANITY.status.value,
            "formal_closure_verdict": None,
            "dsl_ast_executed": False,
            "operator_semantics_frozen": True,
            "witness_id": XOR2_ABSOLUTE_DIFFERENCE_WITNESS.content_id,
            "expression": XOR2_ABSOLUTE_DIFFERENCE_WITNESS.expression,
            "source_document_expression": (
                BINARY_XOR_SANITY.candidate_old_dsl_program
            ),
            "source_expression_typechecks_under_frozen_typing": (
                BINARY_XOR_SANITY.source_candidate_typechecks_under_frozen_typing
            ),
            "implicit_bit_to_rational_coercion_frozen": (
                BINARY_XOR_SANITY.implicit_bit_to_rational_coercion_frozen
            ),
            "truth_table": [
                list(row) for row in XOR2_ABSOLUTE_DIFFERENCE_WITNESS.truth_table
            ],
            "conclusion": (
                "Under standard bit and absolute-difference arithmetic, banning "
                "XOR/parity symbol names does not ban XOR2 semantics. This is a "
                "target-design sanity only; any old-language membership verdict "
                "still needs strict canonical admission and dual closure replay."
            ),
        },
        "hidden_sink_role": "in_language_null_control_only",
        "hidden_sink_control": {
            "control_id": OBSERVED_OMITTED_SINK_CONTROL.control_id,
            "control_spec_id": OBSERVED_OMITTED_SINK_CONTROL.content_id,
            "diagnostic_universe_content_id": (
                OBSERVED_OMITTED_SINK_CONTROL.diagnostic_universe_content_id
            ),
            "diagnostic_target_table_content_id": (
                OBSERVED_OMITTED_SINK_CONTROL.diagnostic_target_table_content_id
            ),
            "formal_bounded_universe_root": None,
            "formal_target_truth_table_root": None,
            "universe_rows": OBSERVED_OMITTED_SINK_CONTROL.universe_rows,
            "all_channels_observed": (
                OBSERVED_OMITTED_SINK_CONTROL.all_channels_present_in_public_typed_evidence
            ),
            "latent_sink_allowed": (
                OBSERVED_OMITTED_SINK_CONTROL.latent_sink_allowed
            ),
            "baseline_scope_id": (
                OBSERVED_OMITTED_SINK_CONTROL.baseline_scope_id
            ),
            "source_document_baseline_alias": (
                OBSERVED_OMITTED_SINK_CONTROL.source_document_baseline_label
            ),
            "scope_alias_confirmation_pending": False,
            "scope_alias_status": "RESOLVED_SOURCE_ONLY",
            "scope_alias_semantic_identity": True,
            "fifth_scope_added": False,
            "formal_canonicalizer_accepts_source_alias": False,
            "legacy_migration_adapter_may_rewrite_alias": True,
            "formal_control_run_complete": False,
            "formal_in_language_verdict_allowed": False,
        },
        "mdl_freeze": {
            "code_table_version": MDL_CODE_TABLE_ID,
            "code_table_content_id": FROZEN_MDL_CODE_TABLE_ID,
            "fixed_point_precision": FIXED_POINT_PRECISION_ID,
            "canonical_ast_wire_frozen": True,
            "new_symbol_wire_frozen": True,
            "new_reducer_v1_header_uint": NEW_REDUCER_V1_HEADER,
            "q32_reference_algorithm_id": Q32_REFERENCE_ALGORITHM_ID,
            "q32_reference_algorithm_frozen": True,
            "formal_scorer_status": "HARD_DISABLED",
            "caller_supplied_lengths_are_formal_evidence": False,
        },
        "shadow_only": contract.shadow_only,
        "active_promotion_authorized": contract.active_promotion_authorized,
        "next_gate": "PUBLISH_SHRUNK_OLD_DSL_VERSION_USING_FROZEN_STEP_1",
        "next_frozen_shrink_step": SHRINK_ORDER[0].operation,
        "claim_boundary": (
            "This artifact binds the frozen surface tables for the DSL, target, "
            "null control, closure budget, certificate, and MDL code table, plus "
            "the v1.0.2 strict canonical-AST/CBOR, bridge, certificate, and MDL "
            "specifications. Independent strict implementations and the 64,680 "
            "capacity subset are verified, establishing DSL_TOO_LARGE for this "
            "bounded old DSL. This is not COMPLETE; all formal roots remain "
            "null, and it is not an extensional target verdict, a hidden-sink "
            "verdict, an outside certificate, a formal MDL gate, or a sealed "
            "Phase-3A result."
        ),
    }
    payload["report_id"] = stable_hash(payload, prefix="phase3_prereg_report_")
    return payload
