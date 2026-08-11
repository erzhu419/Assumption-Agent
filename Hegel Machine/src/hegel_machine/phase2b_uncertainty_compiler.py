"""Bundle-atomic exact Phase-2B uncertainty compilation.

The public Phase-2B wire normalizes every JSON number to an IEEE-754
binary64 value.  This module therefore recovers the *represented* value with
``Fraction.from_float`` and performs all bound arithmetic exactly.  Numeric
endpoints are then rounded outward to the frozen 663-point ``RationalValue``
grid; they are never converted back to float.

Only ``absolute_bound`` is executable.  ``standard_error`` and any endpoint
outside the frozen grid fail the whole bundle without returning a partial set
of compiled observations.  Missing and Boolean observations remain legal and
carry no invented numeric uncertainty.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Final

from .hashing import stable_hash
from .phase2b_freeze_v1 import (
    PHASE2B_EXACT_FREEZE_VERSION,
    FormalUncertaintyKind,
    frozen_phase2b_exact_freeze,
)
from .phase2b_wire import (
    BooleanValue,
    Missingness,
    NumericInterval,
    NumericValue,
    PublicEvidenceBundle,
    TypedObservation,
    UncertaintyModel,
)
from .phase3_dsl_v1 import OLD_DSL_V1, RATIONAL_VALUE_GRID, RationalAtom
from .schema import require_tuple


EXACT_UNCERTAINTY_COMPILER_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-exact-uncertainty-compiler/1"
)
FROZEN_RATIONAL_GRID_ID: Final = (
    "rational_grid_"
    "94131eb37f198c4e42c14266c8c4cacd7eb2a6758997fe5381a2758b6f37277f"
)
FROZEN_PHASE2B_EXACT_FREEZE_ID: Final = (
    "phase2b_exact_freeze_"
    "ffa1fd4fed0b5c2c018803aa9f730b8c85c144efe7e4aa324256681d1c742cbe"
)
_GRID_FRACTIONS: Final = tuple(
    atom.as_fraction() for atom in RATIONAL_VALUE_GRID
)
_GRID_MINIMUM: Final = _GRID_FRACTIONS[0]
_GRID_MAXIMUM: Final = _GRID_FRACTIONS[-1]


class ObservationValueKind(str, Enum):
    NUMERIC_INTERVAL = "numeric_interval"
    BOOLEAN = "boolean"
    MISSING = "missing"


class BundleUncertaintyDisposition(str, Enum):
    COMPLETE = "complete"
    ABSTAIN = "abstain"


@dataclass(frozen=True, slots=True)
class ExactUncertaintyCompilerPolicy:
    schema_version: str = EXACT_UNCERTAINTY_COMPILER_SCHEMA_VERSION
    phase2b_freeze_version: str = PHASE2B_EXACT_FREEZE_VERSION
    phase2b_exact_freeze_id: str = FROZEN_PHASE2B_EXACT_FREEZE_ID
    rational_grid_id: str = FROZEN_RATIONAL_GRID_ID
    rational_grid_cardinality: int = 663
    allowed_uncertainty_kinds: tuple[FormalUncertaintyKind, ...] = (
        FormalUncertaintyKind.ABSOLUTE_BOUND,
    )
    standard_error_status: str = "STANDARD_ERROR_UNSUPPORTED"
    endpoint_rounding: str = "outward_to_frozen_RationalValue_grid"

    def __post_init__(self) -> None:
        exact_freeze = frozen_phase2b_exact_freeze()
        frozen = exact_freeze.uncertainty_policy
        if self.schema_version != EXACT_UNCERTAINTY_COMPILER_SCHEMA_VERSION:
            raise ValueError("exact uncertainty compiler schema drift")
        if self.phase2b_freeze_version != PHASE2B_EXACT_FREEZE_VERSION:
            raise ValueError("Phase-2B exact freeze version drift")
        if (
            exact_freeze.freeze_id != FROZEN_PHASE2B_EXACT_FREEZE_ID
            or self.phase2b_exact_freeze_id != FROZEN_PHASE2B_EXACT_FREEZE_ID
        ):
            raise ValueError("Phase-2B exact freeze identity drift")
        if (
            OLD_DSL_V1.rational_grid_id != FROZEN_RATIONAL_GRID_ID
            or self.rational_grid_id != FROZEN_RATIONAL_GRID_ID
        ):
            raise ValueError("frozen RationalValue grid identity drift")
        if (
            self.rational_grid_cardinality != len(RATIONAL_VALUE_GRID)
            or self.rational_grid_cardinality != 663
        ):
            raise ValueError("frozen RationalValue grid cardinality drift")
        if _GRID_FRACTIONS != tuple(sorted(set(_GRID_FRACTIONS))):
            raise ValueError("frozen RationalValue grid order or uniqueness drift")
        if self.allowed_uncertainty_kinds != frozen.allowed_kinds:
            raise ValueError("allowed formal uncertainty kinds drift")
        if self.standard_error_status != frozen.standard_error_status:
            raise ValueError("standard-error failure status drift")
        if self.endpoint_rounding != frozen.endpoint_rounding:
            raise ValueError("uncertainty endpoint rounding policy drift")

    @property
    def policy_id(self) -> str:
        return stable_hash(self, prefix="phase2b_exact_uncertainty_policy_")


DEFAULT_EXACT_UNCERTAINTY_POLICY: Final = ExactUncertaintyCompilerPolicy()


@dataclass(frozen=True, slots=True)
class ExactRationalInterval:
    lower: RationalAtom
    upper: RationalAtom

    def __post_init__(self) -> None:
        if not isinstance(self.lower, RationalAtom) or not isinstance(
            self.upper,
            RationalAtom,
        ):
            raise TypeError("exact interval endpoints must be RationalAtom values")
        if (
            self.lower not in RATIONAL_VALUE_GRID
            or self.upper not in RATIONAL_VALUE_GRID
        ):
            raise ValueError("exact interval endpoint is outside the frozen grid")
        if self.lower.as_fraction() > self.upper.as_fraction():
            raise ValueError("exact interval lower bound exceeds upper bound")

    @property
    def lower_fraction(self) -> Fraction:
        return self.lower.as_fraction()

    @property
    def upper_fraction(self) -> Fraction:
        return self.upper.as_fraction()


@dataclass(frozen=True, slots=True)
class ExactObservationCompilation:
    observation_id: str
    observation_content_id: str
    compiler_policy_id: str
    phase2b_exact_freeze_id: str
    rational_grid_id: str
    value_kind: ObservationValueKind
    reason: str
    numeric_bounds: tuple[ExactRationalInterval, ...] = ()
    boolean_value: bool | None = None

    def __post_init__(self) -> None:
        require_tuple(self.numeric_bounds, "exact observation numeric bounds")
        if not isinstance(self.value_kind, ObservationValueKind):
            raise TypeError("exact observation value kind has the wrong type")
        if any(
            not isinstance(item, ExactRationalInterval)
            for item in self.numeric_bounds
        ):
            raise TypeError("exact observation contains an invalid interval")
        if not all(
            isinstance(item, str) and bool(item)
            for item in (
                self.observation_id,
                self.observation_content_id,
                self.compiler_policy_id,
                self.phase2b_exact_freeze_id,
                self.rational_grid_id,
                self.reason,
            )
        ):
            raise ValueError("exact observation compilation identity is incomplete")
        if self.phase2b_exact_freeze_id != FROZEN_PHASE2B_EXACT_FREEZE_ID:
            raise ValueError("observation compilation freeze identity drift")
        if self.rational_grid_id != FROZEN_RATIONAL_GRID_ID:
            raise ValueError("observation compilation rational grid identity drift")
        if self.value_kind is ObservationValueKind.NUMERIC_INTERVAL:
            if not self.numeric_bounds or self.boolean_value is not None:
                raise ValueError("numeric compilation needs only interval bounds")
        elif self.value_kind is ObservationValueKind.BOOLEAN:
            if type(self.boolean_value) is not bool or self.numeric_bounds:
                raise ValueError("Boolean compilation needs only an exact Boolean")
        elif self.numeric_bounds or self.boolean_value is not None:
            raise ValueError("missing compilation cannot carry a value")

    @property
    def compilation_id(self) -> str:
        return stable_hash(self, prefix="phase2b_exact_observation_")


@dataclass(frozen=True, slots=True)
class ObservationCompilationFailure:
    observation_id: str
    observation_content_id: str
    error_code: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(item, str) and bool(item)
            for item in (self.observation_id, self.observation_content_id)
        ):
            raise ValueError("observation failure identity is incomplete")
        if not isinstance(self.error_code, str) or not self.error_code:
            raise ValueError("observation failure needs an error code")


@dataclass(frozen=True, slots=True)
class BundleUncertaintyCompilation:
    disposition: BundleUncertaintyDisposition
    reason: str
    bundle_content_id: str
    compiler_policy_id: str
    phase2b_exact_freeze_id: str
    rational_grid_id: str
    observations: tuple[ExactObservationCompilation, ...]
    failures: tuple[ObservationCompilationFailure, ...]

    def __post_init__(self) -> None:
        require_tuple(self.observations, "exact bundle observations")
        require_tuple(self.failures, "exact bundle failures")
        if not isinstance(self.disposition, BundleUncertaintyDisposition):
            raise TypeError("bundle uncertainty disposition has the wrong type")
        if any(
            not isinstance(item, ExactObservationCompilation)
            for item in self.observations
        ):
            raise TypeError("exact bundle contains an invalid observation")
        if any(
            not isinstance(item, ObservationCompilationFailure)
            for item in self.failures
        ):
            raise TypeError("exact bundle contains an invalid failure")
        if not all(
            (
                self.reason,
                self.bundle_content_id,
                self.compiler_policy_id,
                self.phase2b_exact_freeze_id,
                self.rational_grid_id,
            )
        ):
            raise ValueError("exact bundle compilation identity is incomplete")
        if self.phase2b_exact_freeze_id != FROZEN_PHASE2B_EXACT_FREEZE_ID:
            raise ValueError("bundle uncertainty freeze identity drift")
        if self.rational_grid_id != FROZEN_RATIONAL_GRID_ID:
            raise ValueError("bundle uncertainty rational grid identity drift")
        if self.disposition is BundleUncertaintyDisposition.COMPLETE:
            if not self.observations or self.failures:
                raise ValueError("complete uncertainty compilation needs only values")
        elif self.observations or not self.failures:
            raise ValueError("abstaining uncertainty compilation cannot return values")

    @property
    def result_id(self) -> str:
        return stable_hash(self, prefix="phase2b_exact_uncertainty_result_")


class _GridRangeError(ValueError):
    pass


def _round_lower_outward(value: Fraction) -> RationalAtom:
    index = bisect_right(_GRID_FRACTIONS, value) - 1
    if index < 0:
        raise _GridRangeError("RATIONAL_VALUE_GRID_OUT_OF_RANGE")
    return RATIONAL_VALUE_GRID[index]


def _round_upper_outward(value: Fraction) -> RationalAtom:
    index = bisect_left(_GRID_FRACTIONS, value)
    if index >= len(RATIONAL_VALUE_GRID):
        raise _GridRangeError("RATIONAL_VALUE_GRID_OUT_OF_RANGE")
    return RATIONAL_VALUE_GRID[index]


def _exact_binary64(value: float) -> Fraction:
    if type(value) is not float:
        raise TypeError("wire numeric values must already be normalized to float")
    return Fraction.from_float(value)


def _rounded_interval(lower: Fraction, upper: Fraction) -> ExactRationalInterval:
    if lower > upper:
        raise ValueError("exact interval lower bound exceeds upper bound")
    if lower < _GRID_MINIMUM or upper > _GRID_MAXIMUM:
        raise _GridRangeError("RATIONAL_VALUE_GRID_OUT_OF_RANGE")
    return ExactRationalInterval(
        lower=_round_lower_outward(lower),
        upper=_round_upper_outward(upper),
    )


def _compile_observation(
    observation: TypedObservation,
    *,
    policy: ExactUncertaintyCompilerPolicy,
) -> ExactObservationCompilation | ObservationCompilationFailure:
    observation_content_id = stable_hash(
        observation,
        prefix="phase2b_typed_observation_",
    )
    common = {
        "observation_id": observation.observation_id,
        "observation_content_id": observation_content_id,
    }
    compiled_common = {
        **common,
        "compiler_policy_id": policy.policy_id,
        "phase2b_exact_freeze_id": policy.phase2b_exact_freeze_id,
        "rational_grid_id": policy.rational_grid_id,
    }
    if observation.missingness is Missingness.MISSING:
        return ExactObservationCompilation(
            **compiled_common,
            value_kind=ObservationValueKind.MISSING,
            reason="missing_observation_preserved",
        )
    if observation.uncertainty.model is UncertaintyModel.STANDARD_ERROR:
        return ObservationCompilationFailure(
            **common,
            error_code=policy.standard_error_status,
        )
    if isinstance(observation.value, BooleanValue):
        return ExactObservationCompilation(
            **compiled_common,
            value_kind=ObservationValueKind.BOOLEAN,
            reason="exact_boolean_preserved",
            boolean_value=observation.value.value,
        )
    if observation.uncertainty.model is not UncertaintyModel.ABSOLUTE_BOUND:
        return ObservationCompilationFailure(
            **common,
            error_code="UNCERTAINTY_MODEL_UNSUPPORTED",
        )

    radii = tuple(_exact_binary64(value) for value in observation.uncertainty.radius)
    if isinstance(observation.value, NumericValue):
        exact_bounds = tuple(
            (
                _exact_binary64(value) - radius,
                _exact_binary64(value) + radius,
            )
            for value, radius in zip(
                observation.value.values,
                radii,
                strict=True,
            )
        )
    elif isinstance(observation.value, NumericInterval):
        exact_bounds = tuple(
            (
                _exact_binary64(lower) - radius,
                _exact_binary64(upper) + radius,
            )
            for lower, upper, radius in zip(
                observation.value.lower,
                observation.value.upper,
                radii,
                strict=True,
            )
        )
    else:
        return ObservationCompilationFailure(
            **common,
            error_code="OBSERVED_VALUE_KIND_UNSUPPORTED",
        )

    try:
        numeric_bounds = tuple(
            _rounded_interval(lower, upper) for lower, upper in exact_bounds
        )
    except _GridRangeError as exc:
        return ObservationCompilationFailure(
            **common,
            error_code=str(exc),
        )
    return ExactObservationCompilation(
        **compiled_common,
        value_kind=ObservationValueKind.NUMERIC_INTERVAL,
        reason="exact_absolute_bound_rounded_outward",
        numeric_bounds=numeric_bounds,
    )


def compile_bundle_uncertainty(
    bundle: PublicEvidenceBundle,
    *,
    policy: ExactUncertaintyCompilerPolicy = DEFAULT_EXACT_UNCERTAINTY_POLICY,
) -> BundleUncertaintyCompilation:
    """Compile every public observation or atomically reject the bundle.

    All observations are checked so the failure receipt is deterministic and
    independent of caller input order.  If any row fails, every successfully
    compiled row is discarded and ``observations`` is empty.
    """

    if type(bundle) is not PublicEvidenceBundle:
        raise TypeError("exact uncertainty compiler requires PublicEvidenceBundle")
    if type(policy) is not ExactUncertaintyCompilerPolicy:
        raise TypeError("exact uncertainty compiler policy has the wrong type")
    compiled_or_failed = tuple(
        _compile_observation(observation, policy=policy)
        for observation in bundle.observations
    )
    failures = tuple(
        item
        for item in compiled_or_failed
        if isinstance(item, ObservationCompilationFailure)
    )
    common = {
        "bundle_content_id": bundle.content_id,
        "compiler_policy_id": policy.policy_id,
        "phase2b_exact_freeze_id": policy.phase2b_exact_freeze_id,
        "rational_grid_id": policy.rational_grid_id,
    }
    if failures:
        error_codes = tuple(sorted({item.error_code for item in failures}))
        reason = (
            "bundle_uncertainty_preflight:" + error_codes[0]
            if len(error_codes) == 1
            else "bundle_uncertainty_preflight:MULTIPLE_FAILURES"
        )
        return BundleUncertaintyCompilation(
            **common,
            disposition=BundleUncertaintyDisposition.ABSTAIN,
            reason=reason,
            observations=(),
            failures=failures,
        )
    observations = tuple(
        item
        for item in compiled_or_failed
        if isinstance(item, ExactObservationCompilation)
    )
    return BundleUncertaintyCompilation(
        **common,
        disposition=BundleUncertaintyDisposition.COMPLETE,
        reason="complete_exact_uncertainty_bundle",
        observations=observations,
        failures=(),
    )


__all__ = [
    "BundleUncertaintyCompilation",
    "BundleUncertaintyDisposition",
    "DEFAULT_EXACT_UNCERTAINTY_POLICY",
    "EXACT_UNCERTAINTY_COMPILER_SCHEMA_VERSION",
    "FROZEN_PHASE2B_EXACT_FREEZE_ID",
    "FROZEN_RATIONAL_GRID_ID",
    "ExactObservationCompilation",
    "ExactRationalInterval",
    "ExactUncertaintyCompilerPolicy",
    "ObservationCompilationFailure",
    "ObservationValueKind",
    "compile_bundle_uncertainty",
]
