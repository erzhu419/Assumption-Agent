import pytest

from hegel_machine.bootstrap import initial_theory
from hegel_machine.domain import LawBinding, StructuralEpisode
from hegel_machine.pipeline import verify_against_frozen_library, verify_binding
from hegel_machine.schema import EvidenceSplit, LawKind


def symmetry_episode(
    values=(1.0, 2.0),
    transformed=(1.0, 2.0),
    *,
    data_cutoff="2026-07-30T23:59:59+08:00",
):
    return StructuralEpisode.from_mapping(
        episode_id="episode_1",
        observation_ids=("span_1",),
        object_types={"left": "state", "right": "state"},
        role_candidates={"source": "left", "transformed_source": "right"},
        role_observable_witnesses={
            "source": ("forward",),
            "transformed_source": ("transformed", "common_codomains"),
        },
        observables={
            "forward": values,
            "transformed": transformed,
            "common_codomains": True,
        },
        scale_id="phase2_default",
        scope=("controlled_offline_structural_laws",),
        split=EvidenceSplit.TRAIN,
        data_cutoff=data_cutoff,
    )


def symmetry_binding():
    return LawBinding(
        "binding_1",
        "law_symmetry_v1",
        LawKind.SYMMETRY,
        (("source", "left"), ("transformed_source", "right")),
        ("span_1",),
        "phase2_default",
    )


def test_typed_pipeline_emits_verified_law_match():
    theory = initial_theory()
    outcomes = verify_against_frozen_library(
        theory=theory,
        episode=symmetry_episode(),
        bindings=(symmetry_binding(),),
    )
    outcome = outcomes[0]
    assert outcome.evaluation.passed
    assert outcome.match is not None
    assert "accepted_verified_law_match" in outcome.audit_events


def test_executable_violation_blocks_law_match():
    theory = initial_theory()
    law = next(item for item in theory.relation_laws if item.kind is LawKind.SYMMETRY)
    outcome = verify_binding(
        episode=symmetry_episode(transformed=(1.0, 3.0)),
        law=law,
        binding=symmetry_binding(),
        tolerance=0.01,
    )
    assert not outcome.evaluation.passed
    assert outcome.match is None
    assert "rejected_by_executable_violation" in outcome.audit_events


def test_role_incomplete_binding_is_rejected_before_scoring():
    theory = initial_theory()
    law = next(item for item in theory.relation_laws if item.kind is LawKind.SYMMETRY)
    incomplete = LawBinding(
        "bad_binding",
        "law_symmetry_v1",
        LawKind.SYMMETRY,
        (("source", "left"),),
        ("span_1",),
        "phase2_default",
    )
    with pytest.raises(ValueError, match="role schema"):
        verify_binding(
            episode=symmetry_episode(),
            law=law,
            binding=incomplete,
            tolerance=0.01,
        )


def test_source_span_cannot_be_fabricated():
    theory = initial_theory()
    law = next(item for item in theory.relation_laws if item.kind is LawKind.SYMMETRY)
    fabricated = LawBinding(
        "bad_span",
        "law_symmetry_v1",
        LawKind.SYMMETRY,
        symmetry_binding().role_assignments,
        ("not_in_episode",),
        "phase2_default",
    )
    with pytest.raises(ValueError, match="outside"):
        verify_binding(
            episode=symmetry_episode(),
            law=law,
            binding=fabricated,
            tolerance=0.01,
        )


def test_nonexistent_bound_entities_cannot_generate_a_match():
    theory = initial_theory()
    law = next(item for item in theory.relation_laws if item.kind is LawKind.SYMMETRY)
    ghosts = LawBinding(
        "ghost_binding",
        "law_symmetry_v1",
        LawKind.SYMMETRY,
        (("source", "ghost_a"), ("transformed_source", "ghost_b")),
        ("span_1",),
        "phase2_default",
    )
    with pytest.raises(ValueError, match="nonexistent"):
        verify_binding(
            episode=symmetry_episode(),
            law=law,
            binding=ghosts,
            tolerance=0.01,
        )


def test_role_swap_is_rejected_before_residual_scoring():
    theory = initial_theory()
    law = next(item for item in theory.relation_laws if item.kind is LawKind.SYMMETRY)
    swapped = LawBinding(
        "swapped_binding",
        "law_symmetry_v1",
        LawKind.SYMMETRY,
        (("source", "right"), ("transformed_source", "left")),
        ("span_1",),
        "phase2_default",
    )
    with pytest.raises(ValueError, match="role candidates"):
        verify_binding(
            episode=symmetry_episode(),
            law=law,
            binding=swapped,
            tolerance=0.01,
        )


def test_unregistered_scale_is_rejected_before_residual_scoring():
    theory = initial_theory()
    law = next(item for item in theory.relation_laws if item.kind is LawKind.SYMMETRY)
    wrong_scale = LawBinding(
        "wrong_scale",
        "law_symmetry_v1",
        LawKind.SYMMETRY,
        symmetry_binding().role_assignments,
        ("span_1",),
        "case_convenient_scale",
    )
    with pytest.raises(ValueError, match="scales"):
        verify_binding(
            episode=symmetry_episode(),
            law=law,
            binding=wrong_scale,
            tolerance=0.01,
        )


def test_future_episode_cannot_be_verified_against_earlier_theory():
    with pytest.raises(ValueError, match="cutoff"):
        verify_against_frozen_library(
            theory=initial_theory(),
            episode=symmetry_episode(
                data_cutoff="2999-12-31T23:59:59+00:00"
            ),
            bindings=(symmetry_binding(),),
        )
