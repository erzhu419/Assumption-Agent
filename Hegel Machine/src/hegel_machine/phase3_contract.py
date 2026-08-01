"""Fail-closed preregistration contracts for bounded Phase-3 work.

This module freezes only the parts of the old-language experiment that have
already been decided.  It deliberately does *not* pretend that the rational
grid, bounded universe, executable semantics, MDL code table, or hidden task
specifications have been frozen.  Those missing content bindings remain
machine-readable readiness blockers, so the default contract cannot issue an
outside-language certificate.

The module is independent of the Phase-2 selector and does not authorize an
ACTIVE theory mutation.  A future sealed run may fill the content bindings,
but an independently replaying evaluator must also be implemented before
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


PHASE3_CONTRACT_SCHEMA_VERSION = "hegel-machine-phase3-preregistration/1"

# These are deliberately hard-disabled implementation facts, rather than
# constructor inputs a caller could set.  The current module defines wire
# contracts and arithmetic only; it does not replay closure archives or MDL
# partitions under a trusted evaluator.
SEALED_CLOSURE_VERIFIER_IMPLEMENTED = False
SEALED_MDL_SCORER_IMPLEMENTED = False


FROZEN_SORTS = (
    "Entity",
    "Role",
    "Observation",
    "Event",
    "Index",
    "ScaleContext",
    "EntitySet",
    "Bool",
    "Sign",
    "BoundedInt",
    "RationalScalar",
    "IntervalScalar",
    "OrderedCategory",
)

FROZEN_LEAVES = (
    "measurement(entity,quantity_id)",
    "event_value(event,quantity_id)",
    "time_index(event)",
    "space_index(entity)",
    "membership(entity,entity_set)",
    "context_flag(context_id)",
    "task_target(target_id)",
    "uncertainty_interval(measurement)",
)

FROZEN_OPERATORS = (
    "identity",
    "difference",
    "absolute",
    "sign",
    "sum",
    "mean",
    "count",
    "min",
    "max",
    "affine_combination",
    "same_entity",
    "same_role",
    "before",
    "adjacent",
    "subset",
    "aggregate_by",
    "transform_by",
    "approx_equal",
    "less_equal",
    "greater_equal",
    "same_sign",
    "opposite_sign",
    "invariant_equal",
    "within_interval",
    "top_level_conjunction",
)

FROZEN_FORBIDDEN_SYMBOLS = (
    "or",
    "xor",
    "modulo",
    "parity",
    "compound_negation",
    "arbitrary_truth_table_lookup",
    "unbounded_recursion",
    "learned_neural_predicate",
    "case_id_dependent_branch",
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
    """The already-decided finite old-language bounds."""

    maximum_relation_arity: int = 3
    maximum_entity_set_size: int = 8
    maximum_ast_depth: int = 4
    maximum_top_level_clauses: int = 3
    maximum_composition_depth: int = 2
    maximum_fitted_parameters: int = 3
    maximum_scope_clauses: int = 2
    maximum_canonical_programs: int = 50_000

    def __post_init__(self) -> None:
        for name in self.__slots__:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"DSL limit {name} must be a positive integer")

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="phase3_dsl_limits_")


FROZEN_DSL_LIMITS = DslLimits()


class ReadinessBlocker(str, Enum):
    RATIONAL_GRID = "rational_grid_not_frozen"
    BOUNDED_UNIVERSE = "bounded_universe_not_frozen"
    OPERATOR_SEMANTICS = "operator_semantics_not_frozen"
    EQUIVALENCE_CONTRACT = "equivalence_contract_not_frozen"
    CANONICALIZER = "canonicalizer_not_frozen"
    ENUMERATOR = "enumerator_not_frozen"
    MDL_CODE_TABLE = "mdl_code_table_not_frozen"
    PARITY_TARGET = "parity_target_not_frozen"
    HIDDEN_SINK_CONTROL = "hidden_sink_control_not_frozen"
    HIDDEN_GENERATOR = "hidden_generator_not_frozen"
    SEALED_CLOSURE_VERIFIER = "sealed_closure_verifier_not_implemented"


_BLOCKER_FIELDS = (
    (ReadinessBlocker.RATIONAL_GRID, "rational_grid_id"),
    (ReadinessBlocker.BOUNDED_UNIVERSE, "bounded_universe_id"),
    (ReadinessBlocker.OPERATOR_SEMANTICS, "operator_semantics_id"),
    (ReadinessBlocker.EQUIVALENCE_CONTRACT, "equivalence_contract_id"),
    (ReadinessBlocker.CANONICALIZER, "canonicalizer_implementation_id"),
    (ReadinessBlocker.ENUMERATOR, "enumerator_implementation_id"),
    (ReadinessBlocker.MDL_CODE_TABLE, "mdl_code_table_id"),
    (ReadinessBlocker.PARITY_TARGET, "parity_target_id"),
    (ReadinessBlocker.HIDDEN_SINK_CONTROL, "hidden_sink_control_id"),
    (ReadinessBlocker.HIDDEN_GENERATOR, "hidden_generator_spec_id"),
)


@dataclass(frozen=True, slots=True)
class Phase3PrerequisiteContract:
    """Content-addressed freeze manifest for the Phase-3 preregistration.

    Optional ids are intentionally unresolved in the default instance.  Merely
    naming a primitive or banning the word ``parity`` does not fill them.
    """

    schema_version: str = PHASE3_CONTRACT_SCHEMA_VERSION
    sorts: tuple[str, ...] = FROZEN_SORTS
    leaves: tuple[str, ...] = FROZEN_LEAVES
    operators: tuple[str, ...] = FROZEN_OPERATORS
    forbidden_symbols: tuple[str, ...] = FROZEN_FORBIDDEN_SYMBOLS
    limits: DslLimits = FROZEN_DSL_LIMITS
    rational_grid_id: str | None = None
    bounded_universe_id: str | None = None
    operator_semantics_id: str | None = None
    equivalence_contract_id: str | None = None
    canonicalizer_implementation_id: str | None = None
    enumerator_implementation_id: str | None = None
    mdl_code_table_id: str | None = None
    parity_target_id: str | None = None
    hidden_sink_control_id: str | None = None
    hidden_generator_spec_id: str | None = None
    shadow_only: bool = True
    active_promotion_authorized: bool = False

    def __post_init__(self) -> None:
        for name in ("sorts", "leaves", "operators", "forbidden_symbols"):
            _require_tuple(getattr(self, name), f"Phase-3 {name}")
        if self.schema_version != PHASE3_CONTRACT_SCHEMA_VERSION:
            raise ValueError("unknown Phase-3 preregistration schema version")
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


@dataclass(frozen=True, slots=True)
class ClosureEnumerationReceipt:
    """Untrusted replay-claim wire record for one target.

    The fields are structurally validated but not recomputed.  Until a sealed
    closure verifier exists, this object cannot establish either a semantic
    match or completeness of a no-match search.
    """

    contract_id: str
    target_id: str
    target_role: TargetRole
    bounded_universe_id: str
    equivalence_contract_id: str
    enumerator_implementation_id: str
    search_budget: int
    enumerated_canonical_program_count: int
    closure_cardinality: int | None
    complete: bool
    budget_exhausted: bool
    semantics_total: bool
    extensional_match_program_ids: tuple[str, ...]
    closure_root: str
    target_truth_table_root: str

    def __post_init__(self) -> None:
        for name in (
            "contract_id",
            "target_id",
            "bounded_universe_id",
            "equivalence_contract_id",
            "enumerator_implementation_id",
            "closure_root",
            "target_truth_table_root",
        ):
            _require_content_id(getattr(self, name), name, optional=False)
        if not isinstance(self.target_role, TargetRole):
            raise TypeError("target_role must be a TargetRole")
        if self.search_budget != FROZEN_DSL_LIMITS.maximum_canonical_programs:
            raise ValueError("closure receipt changed the frozen search budget")
        if (
            type(self.enumerated_canonical_program_count) is not int
            or self.enumerated_canonical_program_count < 0
            or self.enumerated_canonical_program_count > self.search_budget
        ):
            raise ValueError("enumerated program count is outside the frozen budget")
        if self.closure_cardinality is not None and (
            type(self.closure_cardinality) is not int
            or self.closure_cardinality < 0
        ):
            raise ValueError("closure cardinality must be a nonnegative integer")
        for name in ("complete", "budget_exhausted", "semantics_total"):
            _require_bool(getattr(self, name), name)
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
        if self.complete:
            if self.budget_exhausted:
                raise ValueError("a complete closure cannot be budget exhausted")
            if self.closure_cardinality != self.enumerated_canonical_program_count:
                raise ValueError("complete receipt must bind the full closure cardinality")
        elif self.closure_cardinality is not None:
            raise ValueError("an incomplete receipt cannot claim closure cardinality")
        if self.budget_exhausted and (
            self.enumerated_canonical_program_count != self.search_budget
        ):
            raise ValueError("budget exhaustion requires consuming the full budget")

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
            if self.outside_certificate_id is None:
                raise ValueError("outside verdict needs a content-bound certificate")
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

    bindings = (
        ("contract", receipt.contract_id, contract.content_id),
        ("bounded_universe", receipt.bounded_universe_id, contract.bounded_universe_id),
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
    if receipt.extensional_match_program_ids:
        return _assessment(
            contract,
            receipt,
            AdequacyVerdict.IN_LANGUAGE,
            "full_bounded_truth_table_has_an_old_language_match",
        )
    if not receipt.complete:
        return _assessment(
            contract,
            receipt,
            AdequacyVerdict.INCONCLUSIVE_BUDGET,
            (
                "frozen_search_budget_exhausted"
                if receipt.budget_exhausted
                else "closure_enumeration_incomplete"
            ),
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

    This is not a parsed old-DSL AST and does not use the still-unfrozen
    operator-semantics contract.  It therefore cannot serve as an executable
    closure witness or a formal ``IN_LANGUAGE`` receipt.
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
    expression=(
        "approx_equal(task_target," "absolute(difference(bit_left,bit_right)))"
    ),
    operator_ids=("difference", "absolute", "approx_equal"),
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
        "implementation_id": (
            "phase3_contract_source_sha256_"
            + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        ),
        "status": "decided_limits_frozen_verifiers_and_contract_fields_blocked",
        "formal_phase3a_claim": False,
        "outside_language_certificate_issued": False,
        "ready_for_outside_certificate": contract.ready_for_outside_certificate,
        "closure_receipt_semantics_replayed": False,
        "sealed_closure_verifier_implemented": (
            SEALED_CLOSURE_VERIFIER_IMPLEMENTED
        ),
        "mdl_numeric_threshold_is_formal_gate": False,
        "sealed_mdl_scorer_implemented": SEALED_MDL_SCORER_IMPLEMENTED,
        "readiness_blockers": [
            blocker.value for blocker in contract.readiness_blockers
        ],
        "frozen_limits": {
            "maximum_relation_arity": contract.limits.maximum_relation_arity,
            "maximum_entity_set_size": contract.limits.maximum_entity_set_size,
            "maximum_ast_depth": contract.limits.maximum_ast_depth,
            "maximum_top_level_clauses": (
                contract.limits.maximum_top_level_clauses
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
        },
        "xor2_sanity": {
            "status": "intended_numeric_semantics_sanity_only",
            "formal_closure_verdict": None,
            "dsl_ast_executed": False,
            "operator_semantics_frozen": False,
            "witness_id": XOR2_ABSOLUTE_DIFFERENCE_WITNESS.content_id,
            "expression": XOR2_ABSOLUTE_DIFFERENCE_WITNESS.expression,
            "truth_table": [
                list(row) for row in XOR2_ABSOLUTE_DIFFERENCE_WITNESS.truth_table
            ],
            "conclusion": (
                "Under standard bit and absolute-difference arithmetic, banning "
                "XOR/parity symbol names does not ban XOR2 semantics. This is a "
                "target-design sanity only; the higher-arity target still needs "
                "frozen executable semantics and sealed closure replay."
            ),
        },
        "hidden_sink_role": "in_language_null_control_only",
        "shadow_only": contract.shadow_only,
        "active_promotion_authorized": contract.active_promotion_authorized,
        "claim_boundary": (
            "This artifact freezes decided surface limits, untrusted receipt wire "
            "schemas, and an exact MDL arithmetic precheck. It is not a replayed "
            "old-DSL closure, parity outside proof, hidden-sink result, formal MDL "
            "gate, or sealed Phase-3A result."
        ),
    }
    payload["report_id"] = stable_hash(payload, prefix="phase3_prereg_report_")
    return payload
