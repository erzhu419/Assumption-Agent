from dataclasses import FrozenInstanceError

import pytest

from hegel_machine.milestones import (
    ALL_MILESTONES,
    CURRENT_PHASE2A_ALLOWED_CLAIM,
    CURRENT_SCALE_CAPABILITY_NAME,
    CURRENT_TYPED_SELECTION_CAPABILITY_NAME,
    DEVELOPMENT_ONLY_FIXTURE_USES,
    FORMAL_TRACKS,
    PHASE2A,
    PHASE2A_ALLOWED_CAPABILITY_IDS,
    PHASE2A_LEGACY_REPORT_STATUS,
    PHASE2A_PROHIBITED_CAPABILITY_IDS,
    PHASE2B,
    PHASE2R,
    PHASE3A,
    PHASE3B,
    PHASE3C,
    PROHIBITED_FORMAL_FIXTURE_USES,
    prohibited_phase2a_claims,
    validate_development_fixture_use,
    validate_phase2a_capabilities,
    validate_phase2a_claim,
    validate_phase2a_scope,
)


def test_milestone_names_and_machine_ids_are_frozen_and_unique():
    assert PHASE2A.name == (
        "Phase-2A Controlled Typed-Selector Mechanics Qualification"
    )
    assert PHASE2A.machine_id == (
        "phase2a_controlled_typed_selector_mechanics_qualification"
    )
    assert CURRENT_TYPED_SELECTION_CAPABILITY_NAME == (
        "Explicit-Projection Typed Structural Selection"
    )
    assert CURRENT_SCALE_CAPABILITY_NAME == (
        "Scale-Indexed Candidate Projection Selection"
    )
    assert PHASE2A_LEGACY_REPORT_STATUS == "controlled_api_selector_qualified"
    assert len({milestone.machine_id for milestone in ALL_MILESTONES}) == len(
        ALL_MILESTONES
    )
    assert len({milestone.name for milestone in ALL_MILESTONES}) == len(
        ALL_MILESTONES
    )
    with pytest.raises(FrozenInstanceError):
        PHASE2A.name = "Phase-2 exit"  # type: ignore[misc]


def test_formal_tracks_are_distinct_from_current_phase2a():
    assert PHASE2A not in FORMAL_TRACKS
    assert FORMAL_TRACKS == (PHASE2B, PHASE2R, PHASE3A, PHASE3B, PHASE3C)
    assert tuple(track.name for track in FORMAL_TRACKS) == (
        "Phase-2B Sealed Typed-Evidence Structural Identification Qualification",
        "Phase-2R Raw-Evidence Structuralization Qualification",
        "Phase-3A Bounded Frozen-Closure Adequacy",
        "Phase-3B Bounded Meta-Prior Synthesis and Conservative Integration",
        "Phase-3C Raw-Evidence End-to-End Qualification",
    )


def test_current_public_claim_is_an_exact_allowlist():
    assert validate_phase2a_claim(CURRENT_PHASE2A_ALLOWED_CLAIM) == (
        CURRENT_PHASE2A_ALLOWED_CLAIM
    )
    with pytest.raises(ValueError, match="not in the frozen allowlist"):
        validate_phase2a_claim("Hegel Machine passed a selector benchmark.")


@pytest.mark.parametrize(
    ("claim", "capability_id"),
    (
        ("Phase-2A is a formal Phase-2 exit.", "formal_phase2_exit"),
        (
            "Phase-2A reasons directly from raw natural language.",
            "raw_evidence_structural_reasoning",
        ),
        (
            "Phase-2A performs autonomous scale inference.",
            "autonomous_scale_inference",
        ),
        (
            "Phase-2A performs open-world law discovery.",
            "open_world_law_discovery",
        ),
    ),
)
def test_phase2a_rejects_inflated_public_claims(claim, capability_id):
    assert prohibited_phase2a_claims(claim) == (capability_id,)
    with pytest.raises(ValueError, match=capability_id):
        validate_phase2a_claim(claim)


@pytest.mark.parametrize(
    "flag",
    (
        "formal_phase2_exit",
        "raw_evidence_reasoning",
        "autonomous_scale_inference",
        "open_world_discovery",
    ),
)
def test_phase2a_rejects_inflated_machine_status_flags(flag):
    assert validate_phase2a_scope() is PHASE2A
    with pytest.raises(ValueError, match="Phase-2A cannot be labeled"):
        validate_phase2a_scope(**{flag: True})


def test_phase2a_capability_ids_are_closed_and_non_inflationary():
    assert validate_phase2a_capabilities(PHASE2A_ALLOWED_CAPABILITY_IDS) == tuple(
        sorted(PHASE2A_ALLOWED_CAPABILITY_IDS)
    )
    for capability_id in PHASE2A_PROHIBITED_CAPABILITY_IDS:
        with pytest.raises(ValueError, match=capability_id):
            validate_phase2a_capabilities((capability_id,))
    with pytest.raises(ValueError, match="unregistered"):
        validate_phase2a_capabilities(("future_unfrozen_capability",))


def test_current_fixtures_are_development_only():
    assert isinstance(DEVELOPMENT_ONLY_FIXTURE_USES, frozenset)
    assert isinstance(PROHIBITED_FORMAL_FIXTURE_USES, frozenset)
    for use in DEVELOPMENT_ONLY_FIXTURE_USES:
        assert validate_development_fixture_use(use) == use
    for use in PROHIBITED_FORMAL_FIXTURE_USES:
        with pytest.raises(ValueError, match="cannot be used"):
            validate_development_fixture_use(use)
    with pytest.raises(ValueError, match="unregistered"):
        validate_development_fixture_use("publication_evidence")
