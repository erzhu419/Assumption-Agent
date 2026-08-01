"""Frozen milestone names and claim boundaries for the Hegel Machine.

This module deliberately contains no benchmark implementation.  It is the small,
machine-readable boundary between what Phase-2A demonstrated and the stronger
claims reserved for the sealed and raw-evidence tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable


@dataclass(frozen=True, slots=True)
class MilestoneIdentity:
    """Stable public name paired with a serialization-safe machine ID."""

    machine_id: str
    name: str

    def __post_init__(self) -> None:
        if not self.machine_id or not self.name:
            raise ValueError("milestone machine ID and name are required")
        if not self.machine_id.isascii() or not all(
            character.islower() or character.isdigit() or character == "_"
            for character in self.machine_id
        ):
            raise ValueError(
                "milestone machine ID must use lowercase ASCII letters, digits, "
                "and underscores"
            )


PHASE2A: Final = MilestoneIdentity(
    machine_id="phase2a_controlled_typed_selector_mechanics_qualification",
    name="Phase-2A Controlled Typed-Selector Mechanics Qualification",
)
PHASE2B: Final = MilestoneIdentity(
    machine_id="phase2b_sealed_typed_evidence_structural_identification_qualification",
    name="Phase-2B Sealed Typed-Evidence Structural Identification Qualification",
)
PHASE2R: Final = MilestoneIdentity(
    machine_id="phase2r_raw_evidence_structuralization_qualification",
    name="Phase-2R Raw-Evidence Structuralization Qualification",
)
PHASE3A: Final = MilestoneIdentity(
    machine_id="phase3a_bounded_language_adequacy_and_outside_language_detection",
    name="Phase-3A Bounded Language-Adequacy and Outside-Language Detection",
)
PHASE3B: Final = MilestoneIdentity(
    machine_id="phase3b_bounded_meta_prior_synthesis_and_conservative_integration",
    name="Phase-3B Bounded Meta-Prior Synthesis and Conservative Integration",
)
PHASE3C: Final = MilestoneIdentity(
    machine_id="phase3c_raw_evidence_end_to_end_qualification",
    name="Phase-3C Raw-Evidence End-to-End Qualification",
)

ALL_MILESTONES: Final = (PHASE2A, PHASE2B, PHASE2R, PHASE3A, PHASE3B, PHASE3C)
FORMAL_TRACKS: Final = (PHASE2B, PHASE2R, PHASE3A, PHASE3B, PHASE3C)

# The old report status remains a development-artifact label, not a formal exit.
PHASE2A_LEGACY_REPORT_STATUS: Final = "controlled_api_selector_qualified"

CURRENT_TYPED_SELECTION_CAPABILITY_NAME: Final = (
    "Explicit-Projection Typed Structural Selection"
)
CURRENT_SCALE_CAPABILITY_NAME: Final = (
    "Scale-Indexed Candidate Projection Selection"
)
CURRENT_SCALE_CAPABILITY_ALIAS: Final = "Selection Across Explicitly Declared Scales"

PHASE2B_FORMAL_CLAIM_NAME: Final = (
    "Sealed Typed-Evidence Structural Law Identification and Verification"
)
END_TO_END_RAW_CLAIM_NAME: Final = (
    "End-to-End Raw-Evidence Structural Law Identification"
)

CURRENT_PHASE2A_ALLOWED_CLAIM: Final = (
    "Hegel Machine v0.2 has completed a controlled typed-selector mechanics "
    "qualification on verifier-ready synthetic fixtures."
)
CURRENT_PHASE2A_ALLOWED_CLAIMS: Final = (CURRENT_PHASE2A_ALLOWED_CLAIM,)

PHASE2A_ALLOWED_CAPABILITY_IDS: Final = frozenset(
    {
        "controlled_typed_selector_mechanics",
        "explicit_projection_typed_structural_selection",
        "scale_indexed_candidate_projection_selection",
        "deterministic_development_fixture_replay",
    }
)
PHASE2A_PROHIBITED_CAPABILITY_IDS: Final = frozenset(
    {
        "formal_phase2_exit",
        "raw_evidence_structural_reasoning",
        "autonomous_scale_inference",
        "open_world_law_discovery",
    }
)

DEVELOPMENT_ONLY_FIXTURE_USES: Final = frozenset(
    {
        "adapter_compatibility",
        "development_regression",
        "known_bug_reproduction",
        "numerical_stress",
        "public_demo",
        "unit_tests",
    }
)
PROHIBITED_FORMAL_FIXTURE_USES: Final = frozenset(
    {
        "baseline_margin_claim",
        "formal_validation",
        "phase2_exit_confidence_intervals",
        "phase3_hidden_law_evidence",
        "sealed_holdout",
        "threshold_calibration",
    }
)


@dataclass(frozen=True, slots=True)
class ProhibitedClaim:
    """One overclaim category and phrases that reveal it in publication text."""

    capability_id: str
    phrases: tuple[str, ...]


PHASE2A_PROHIBITED_CLAIMS: Final = (
    ProhibitedClaim(
        "formal_phase2_exit",
        (
            "formal phase 2 exit",
            "formal phase2 exit",
            "phase 2 exit qualified",
            "phase2 exit qualified",
            "正式 phase 2 exit",
            "正式 phase2 exit",
        ),
    ),
    ProhibitedClaim(
        "raw_evidence_structural_reasoning",
        (
            "raw evidence",
            "raw natural language",
            "raw text",
            "raw table",
            "raw trajectory",
            "原始证据推理",
            "从自然语言开始",
        ),
    ),
    ProhibitedClaim(
        "autonomous_scale_inference",
        (
            "autonomous scale",
            "context inferred scale",
            "context conditioned scale inference",
            "scale abstraction learning",
            "scale discovery",
            "自主推断 scale",
            "自主 scale",
            "从上下文推断 scale",
        ),
    ),
    ProhibitedClaim(
        "open_world_law_discovery",
        (
            "open world",
            "law discovery",
            "scientific discovery",
            "discovers arbitrary laws",
            "meta prior invention",
            "开放世界",
            "任意规律发现",
            "科学发现",
            "元先验发明",
        ),
    ),
)


def _normalized_claim(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("claim must be a string")
    normalized = value.casefold()
    for character in "-_/‐‑‒–—―":
        normalized = normalized.replace(character, " ")
    return " ".join(normalized.split())


def prohibited_phase2a_claims(claim: str) -> tuple[str, ...]:
    """Return the stable capability IDs contradicted by a Phase-2A claim."""

    normalized = _normalized_claim(claim)
    return tuple(
        prohibition.capability_id
        for prohibition in PHASE2A_PROHIBITED_CLAIMS
        if any(
            _normalized_claim(phrase) in normalized
            for phrase in prohibition.phrases
        )
    )


def validate_phase2a_claim(claim: str) -> str:
    """Accept only a frozen, non-inflationary public claim for current Phase-2A."""

    normalized = " ".join(claim.split()) if isinstance(claim, str) else claim
    prohibited = prohibited_phase2a_claims(claim)
    if prohibited:
        raise ValueError(
            "Phase-2A claim exceeds its qualification boundary: "
            + ", ".join(prohibited)
        )
    if normalized not in CURRENT_PHASE2A_ALLOWED_CLAIMS:
        raise ValueError("Phase-2A public claim is not in the frozen allowlist")
    return normalized


def validate_phase2a_capabilities(capability_ids: Iterable[str]) -> tuple[str, ...]:
    """Validate machine-readable capabilities attached to a Phase-2A artifact."""

    if isinstance(capability_ids, (str, bytes)):
        raise TypeError("capability IDs must be an iterable of strings")
    normalized: set[str] = set()
    for capability_id in capability_ids:
        if not isinstance(capability_id, str) or not capability_id:
            raise TypeError("capability IDs must be nonempty strings")
        normalized.add(capability_id)
    prohibited = normalized & PHASE2A_PROHIBITED_CAPABILITY_IDS
    if prohibited:
        raise ValueError(
            "Phase-2A artifact asserts prohibited capabilities: "
            + ", ".join(sorted(prohibited))
        )
    unknown = normalized - PHASE2A_ALLOWED_CAPABILITY_IDS
    if unknown:
        raise ValueError(
            "Phase-2A artifact asserts unregistered capabilities: "
            + ", ".join(sorted(unknown))
        )
    return tuple(sorted(normalized))


def validate_phase2a_scope(
    *,
    formal_phase2_exit: bool = False,
    raw_evidence_reasoning: bool = False,
    autonomous_scale_inference: bool = False,
    open_world_discovery: bool = False,
) -> MilestoneIdentity:
    """Reject stronger status flags on the current qualification milestone."""

    flags = {
        "formal_phase2_exit": formal_phase2_exit,
        "raw_evidence_structural_reasoning": raw_evidence_reasoning,
        "autonomous_scale_inference": autonomous_scale_inference,
        "open_world_law_discovery": open_world_discovery,
    }
    if any(type(value) is not bool for value in flags.values()):
        raise TypeError("Phase-2A scope flags must be booleans")
    enabled = tuple(name for name, value in flags.items() if value)
    if enabled:
        raise ValueError(
            "Phase-2A cannot be labeled as " + ", ".join(enabled)
        )
    return PHASE2A


def validate_development_fixture_use(use: str) -> str:
    """Allow the current fixtures only in their frozen development-only roles."""

    if not isinstance(use, str) or not use:
        raise TypeError("fixture use must be a nonempty string")
    if use in PROHIBITED_FORMAL_FIXTURE_USES:
        raise ValueError(f"current Phase-2A fixtures cannot be used for {use}")
    if use not in DEVELOPMENT_ONLY_FIXTURE_USES:
        raise ValueError(f"unregistered Phase-2A fixture use: {use}")
    return use


__all__ = (
    "ALL_MILESTONES",
    "CURRENT_PHASE2A_ALLOWED_CLAIM",
    "CURRENT_PHASE2A_ALLOWED_CLAIMS",
    "CURRENT_SCALE_CAPABILITY_ALIAS",
    "CURRENT_SCALE_CAPABILITY_NAME",
    "CURRENT_TYPED_SELECTION_CAPABILITY_NAME",
    "DEVELOPMENT_ONLY_FIXTURE_USES",
    "END_TO_END_RAW_CLAIM_NAME",
    "FORMAL_TRACKS",
    "MilestoneIdentity",
    "PHASE2A",
    "PHASE2A_ALLOWED_CAPABILITY_IDS",
    "PHASE2A_LEGACY_REPORT_STATUS",
    "PHASE2A_PROHIBITED_CAPABILITY_IDS",
    "PHASE2A_PROHIBITED_CLAIMS",
    "PHASE2B",
    "PHASE2B_FORMAL_CLAIM_NAME",
    "PHASE2R",
    "PHASE3A",
    "PHASE3B",
    "PHASE3C",
    "PROHIBITED_FORMAL_FIXTURE_USES",
    "ProhibitedClaim",
    "prohibited_phase2a_claims",
    "validate_development_fixture_use",
    "validate_phase2a_capabilities",
    "validate_phase2a_claim",
    "validate_phase2a_scope",
)
