"""Ordered ontology-inadequacy diagnosis.

The diagnoser stops at the first cheaper repair that explains the residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from .hashing import stable_hash


class InadequacyVerdict(str, Enum):
    PARAMETER_DEFECT = "parameter_defect"
    NOISE_OR_INSUFFICIENT_DATA = "noise_or_insufficient_data"
    SCOPE_DEFECT = "scope_defect"
    MIXTURE_DEFECT = "mixture_defect"
    COMPOSITION_DEFECT = "composition_defect"
    IDEALIZATION_CANDIDATE = "idealization_candidate"
    ROBUSTIFICATION_CANDIDATE = "robustification_candidate"
    PROBE_DEFECT = "probe_defect"
    ONTOLOGY_DEFECT = "ontology_defect"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


DIAGNOSTIC_LADDER = (
    InadequacyVerdict.PARAMETER_DEFECT,
    InadequacyVerdict.NOISE_OR_INSUFFICIENT_DATA,
    InadequacyVerdict.SCOPE_DEFECT,
    InadequacyVerdict.MIXTURE_DEFECT,
    InadequacyVerdict.COMPOSITION_DEFECT,
    InadequacyVerdict.IDEALIZATION_CANDIDATE,
    InadequacyVerdict.ROBUSTIFICATION_CANDIDATE,
    InadequacyVerdict.PROBE_DEFECT,
    InadequacyVerdict.ONTOLOGY_DEFECT,
)


@dataclass(frozen=True, slots=True)
class ResidualProfile:
    profile_id: str
    refit_gain: float
    uncertainty_coverage: float
    scope_repair_gain: float
    mixture_gain: float
    low_order_composition_gain: float
    idealization_gain: float
    robustification_tail_gain: float
    added_probe_gain: float
    cross_seed_stability: float
    structural_coherence: float
    uncertainty_excess: float
    compression_gain: float
    preregistered_prediction_gain: float
    case_count: int
    outlier_fraction: float

    def __post_init__(self) -> None:
        bounded = (
            self.refit_gain,
            self.uncertainty_coverage,
            self.scope_repair_gain,
            self.mixture_gain,
            self.low_order_composition_gain,
            self.idealization_gain,
            self.robustification_tail_gain,
            self.added_probe_gain,
            self.cross_seed_stability,
            self.structural_coherence,
            self.uncertainty_excess,
            self.compression_gain,
            self.preregistered_prediction_gain,
            self.outlier_fraction,
        )
        if any(value < 0 or value > 1 for value in bounded):
            raise ValueError("residual profile scores must be in [0, 1]")
        if any(not isfinite(value) for value in bounded):
            raise ValueError("residual profile scores must be finite")
        if self.case_count < 1:
            raise ValueError("residual profile needs at least one case")

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="residual_profile_")


@dataclass(frozen=True, slots=True)
class OntologyInadequacyReport:
    profile_id: str
    verdict: InadequacyVerdict
    checked_steps: tuple[InadequacyVerdict, ...]
    residual_is_persistent: bool
    language_extension_allowed: bool
    rationale: tuple[str, ...]
    profile_content_id: str
    thresholds_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.checked_steps, tuple) or not isinstance(
            self.rationale, tuple
        ):
            raise TypeError("diagnostic report history must be immutable tuples")
        if not self.profile_content_id or not self.thresholds_id:
            raise ValueError("diagnostic report must bind profile and thresholds")
        expected_steps = (
            DIAGNOSTIC_LADDER
            if self.verdict is InadequacyVerdict.INSUFFICIENT_EVIDENCE
            else DIAGNOSTIC_LADDER[
                : DIAGNOSTIC_LADDER.index(self.verdict) + 1
            ]
        )
        if self.checked_steps != expected_steps:
            raise ValueError("diagnostic report does not contain the full ladder prefix")
        if self.residual_is_persistent is not (
            self.verdict is InadequacyVerdict.ONTOLOGY_DEFECT
        ):
            raise ValueError("diagnostic persistence flag is inconsistent")
        if self.language_extension_allowed != (
            self.verdict is InadequacyVerdict.ONTOLOGY_DEFECT
            and self.residual_is_persistent
        ):
            raise ValueError("language flag is inconsistent with diagnostic verdict")
        if not self.checked_steps or not self.rationale:
            raise ValueError("diagnostic report needs checked steps and rationale")

    @property
    def report_id(self) -> str:
        return stable_hash(self, prefix="ontology_report_")


@dataclass(frozen=True, slots=True)
class DiagnosticThresholds:
    repair_gain: float = 0.20
    uncertainty_coverage: float = 0.80
    stability: float = 0.75
    coherence: float = 0.70
    uncertainty_excess: float = 0.60
    compression_gain: float = 0.10
    prediction_gain: float = 0.10
    minimum_cases: int = 3
    maximum_outlier_fraction: float = 0.34

    def __post_init__(self) -> None:
        values = tuple(
            getattr(self, name)
            for name in (
                "repair_gain",
                "uncertainty_coverage",
                "stability",
                "coherence",
                "uncertainty_excess",
                "compression_gain",
                "prediction_gain",
                "maximum_outlier_fraction",
            )
        )
        if any(not isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("diagnostic thresholds must be finite values in [0, 1]")
        if self.minimum_cases < 1:
            raise ValueError("diagnostic minimum cases must be positive")

    @property
    def thresholds_id(self) -> str:
        return stable_hash(self, prefix="diagnostic_policy_")


def diagnose_ontology_inadequacy(
    profile: ResidualProfile,
    thresholds: DiagnosticThresholds = DiagnosticThresholds(),
) -> OntologyInadequacyReport:
    checked: list[InadequacyVerdict] = []

    def repair(verdict: InadequacyVerdict, gain: float, message: str):
        checked.append(verdict)
        if gain >= thresholds.repair_gain:
            return OntologyInadequacyReport(
                profile_id=profile.profile_id,
                verdict=verdict,
                checked_steps=tuple(checked),
                residual_is_persistent=False,
                language_extension_allowed=False,
                rationale=(message,),
                profile_content_id=profile.content_id,
                thresholds_id=thresholds.thresholds_id,
            )
        return None

    result = repair(
        InadequacyVerdict.PARAMETER_DEFECT,
        profile.refit_gain,
        "parameter refit explains the residual",
    )
    if result:
        return result

    checked.append(InadequacyVerdict.NOISE_OR_INSUFFICIENT_DATA)
    if profile.uncertainty_coverage >= thresholds.uncertainty_coverage:
        return OntologyInadequacyReport(
            profile.profile_id,
            InadequacyVerdict.NOISE_OR_INSUFFICIENT_DATA,
            tuple(checked),
            False,
            False,
            ("existing uncertainty already covers the residual",),
            profile.content_id,
            thresholds.thresholds_id,
        )

    ordered_repairs = (
        (
            InadequacyVerdict.SCOPE_DEFECT,
            profile.scope_repair_gain,
            "a scoped repair explains the residual",
        ),
        (
            InadequacyVerdict.MIXTURE_DEFECT,
            profile.mixture_gain,
            "a mixture/regime split explains the residual",
        ),
        (
            InadequacyVerdict.COMPOSITION_DEFECT,
            profile.low_order_composition_gain,
            "existing primitives composed at low order explain the residual",
        ),
        (
            InadequacyVerdict.IDEALIZATION_CANDIDATE,
            profile.idealization_gain,
            "a simpler quotient/restriction preserves task observables",
        ),
        (
            InadequacyVerdict.ROBUSTIFICATION_CANDIDATE,
            profile.robustification_tail_gain,
            "uncertainty expansion repairs tail or safety undercoverage",
        ),
        (
            InadequacyVerdict.PROBE_DEFECT,
            profile.added_probe_gain,
            "an added separating probe resolves the apparent mismatch",
        ),
    )
    for verdict, gain, message in ordered_repairs:
        result = repair(verdict, gain, message)
        if result:
            return result

    persistent_checks = {
        "enough_cases": profile.case_count >= thresholds.minimum_cases,
        "not_single_outliers": profile.outlier_fraction
        <= thresholds.maximum_outlier_fraction,
        "cross_seed_stability": profile.cross_seed_stability >= thresholds.stability,
        "structural_coherence": profile.structural_coherence >= thresholds.coherence,
        "outside_uncertainty": profile.uncertainty_excess
        >= thresholds.uncertainty_excess,
        "compressible": profile.compression_gain >= thresholds.compression_gain,
        "predictive": profile.preregistered_prediction_gain
        >= thresholds.prediction_gain,
    }
    checked.append(InadequacyVerdict.ONTOLOGY_DEFECT)
    if all(persistent_checks.values()):
        return OntologyInadequacyReport(
            profile.profile_id,
            InadequacyVerdict.ONTOLOGY_DEFECT,
            tuple(checked),
            True,
            True,
            tuple(
                f"{name}=pass" for name in sorted(persistent_checks)
            ),
            profile.content_id,
            thresholds.thresholds_id,
        )
    failures = tuple(
        f"{name}=fail" for name, passed in sorted(persistent_checks.items()) if not passed
    )
    return OntologyInadequacyReport(
        profile.profile_id,
        InadequacyVerdict.INSUFFICIENT_EVIDENCE,
        tuple(checked),
        False,
        False,
        failures,
        profile.content_id,
        thresholds.thresholds_id,
    )
