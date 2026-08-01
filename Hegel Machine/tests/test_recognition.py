from dataclasses import FrozenInstanceError, fields, replace
from hashlib import sha256
from inspect import signature
from pathlib import Path

import pytest

import hegel_machine.recognition as recognition_module
from hegel_machine.bootstrap import initial_theory
from hegel_machine.recognition import (
    RECOGNITION_IMPLEMENTATION_ID,
    RecognitionDisposition,
    RecognitionPolicy,
    StructuralProjection,
    UnboundStructuralEpisode,
    recognize_structural_law,
    replay_recognition_decision,
    verify_preservation,
)
from hegel_machine.schema import EvidenceSplit, LawKind, freeze_pairs


FAIL_OBSERVABLES = {
    LawKind.SYMMETRY: {
        "forward": (1.0, 2.0),
        "transformed": (1.0, 3.0),
        "common_codomains": True,
    },
    LawKind.MONOTONICITY: {
        "x_low": 1.0,
        "x_high": 2.0,
        "y_low": 5.0,
        "y_high": 3.0,
        "direction": 1.0,
    },
    LawKind.CONSERVATION: {
        "storage_delta": 1.0,
        "inflows": (10.0,),
        "outflows": (7.0,),
        "sources": (),
        "sinks": (),
        "boundary_observed": True,
    },
    LawKind.COMPLEMENTARITY: {
        "u_empty": 0.0,
        "u_a": 1.0,
        "u_b": 1.0,
        "u_ab": 1.0,
        "expected_interaction": 1.0,
        "interaction_margin": 0.5,
    },
    LawKind.NEGATIVE_FEEDBACK: {
        "disturbance_delta": 2.0,
        "response_delta": 1.0,
        "deviation_before_response": 2.0,
        "deviation_after_response": 3.0,
        "controlled_quantity_observed": True,
        "disturbance_precedes_response": True,
        "system_induced_response": True,
        "same_controlled_quantity": True,
        "local_stability_window_observed": True,
        "response_margin": 0.5,
        "mitigation_margin": 0.5,
    },
    LawKind.LOCALITY: {
        "conditional_a": (0.7, 0.3),
        "conditional_b": (0.2, 0.8),
        "blanket_observed": True,
        "same_blanket_state": True,
    },
}


PASS_OBSERVABLES = {
    LawKind.SYMMETRY: {
        "forward": (1.0, 2.0),
        "transformed": (1.0, 2.0),
        "common_codomains": True,
    },
    LawKind.MONOTONICITY: {
        "x_low": 1.0,
        "x_high": 2.0,
        "y_low": 3.0,
        "y_high": 5.0,
        "direction": 1.0,
    },
}


def episode(
    *,
    prefix="source",
    passing=(LawKind.SYMMETRY,),
    missing_kind=None,
    near_competitor=False,
    semantic_overlap=0.01,
    semantic_name="unrelated words",
    reverse_projections=False,
):
    theory = initial_theory()
    observation_id = f"obs_{prefix}"
    entity_ids = tuple(f"{prefix}_entity_{index}" for index in range(3))
    projections = []
    for kind in LawKind:
        law = next(item for item in theory.relation_laws if item.kind is kind)
        observables = dict(
            PASS_OBSERVABLES[kind]
            if kind in passing
            else FAIL_OBSERVABLES[kind]
        )
        if near_competitor and kind is LawKind.MONOTONICITY:
            # Completed and failing at residual 0.02: twice the verifier's
            # 0.01 tolerance, so its normalized boundary score is 2.0.
            observables.update(y_low=1.0, y_high=0.98)
        if kind is missing_kind:
            observables.pop(next(iter(observables)))
        roles = {
            role: entity_ids[index]
            for index, role in enumerate(law.roles)
        }
        projections.append(
            StructuralProjection.from_mapping(
                projection_id=f"projection_{prefix}_{kind.value}",
                law_id=law.law_id,
                law_kind=kind,
                role_assignments=roles,
                scale_id="phase2_default",
                evaluator_epoch=theory.evaluator.epoch,
                source_observation_ids=(observation_id,),
                observables=observables,
            )
        )
    if reverse_projections:
        projections.reverse()
    return UnboundStructuralEpisode.from_projections(
        episode_id=f"episode_{prefix}",
        observation_ids=(observation_id,),
        typed_entities={entity_id: "anonymous_state" for entity_id in entity_ids},
        candidate_projections=tuple(projections),
        available_scale_ids=("phase2_default",),
        evaluator_epoch=theory.evaluator.epoch,
        scope=("controlled_offline_structural_laws",),
        split=EvidenceSplit.TRAIN,
        data_cutoff=theory.data_cutoff,
        semantic_metadata={
            "semantic_overlap": semantic_overlap,
            "name_hint": semantic_name,
        },
    )


def two_scale_theory():
    parent = initial_theory()
    alternate = replace(parent.scales[0], scale_id="phase2_alternate")
    laws = tuple(
        replace(law, scale_ids=("phase2_default", "phase2_alternate"))
        for law in parent.relation_laws
    )
    return replace(parent, relation_laws=laws, scales=(parent.scales[0], alternate))


def episode_with_axis_competitors(
    *, binding_competitor="complete", scale_competitor="complete"
):
    base = episode(prefix="axis")
    selected = next(
        projection
        for projection in base.candidate_projections
        if projection.law_kind is LawKind.SYMMETRY
    )
    projections = list(base.candidate_projections)

    def observables(mode):
        if mode == "complete":
            return dict(FAIL_OBSERVABLES[LawKind.SYMMETRY])
        if mode == "abstain":
            return {
                "transformed": (1.0, 3.0),
                "common_codomains": True,
            }
        if mode == "missing":
            return None
        raise ValueError(f"unknown competitor mode: {mode}")

    binding_observables = observables(binding_competitor)
    if binding_observables is not None:
        role_items = selected.role_assignments
        swapped_roles = {
            role_items[0][0]: role_items[1][1],
            role_items[1][0]: role_items[0][1],
        }
        projections.append(
            StructuralProjection.from_mapping(
                projection_id="projection_axis_binding_competitor",
                law_id=selected.law_id,
                law_kind=selected.law_kind,
                role_assignments=swapped_roles,
                scale_id=selected.scale_id,
                evaluator_epoch=selected.evaluator_epoch,
                source_observation_ids=selected.source_observation_ids,
                observables=binding_observables,
            )
        )
    scale_observables = observables(scale_competitor)
    if scale_observables is not None:
        projections.append(
            StructuralProjection.from_mapping(
                projection_id="projection_axis_scale_competitor",
                law_id=selected.law_id,
                law_kind=selected.law_kind,
                role_assignments=dict(selected.role_assignments),
                scale_id="phase2_alternate",
                evaluator_epoch=selected.evaluator_epoch,
                source_observation_ids=selected.source_observation_ids,
                observables=scale_observables,
            )
        )
    return replace(
        base,
        candidate_projections=tuple(projections),
        available_scale_ids=("phase2_default", "phase2_alternate"),
    )


def test_api_has_no_gold_kind_role_or_scale_and_episode_is_immutable():
    episode_fields = {item.name for item in fields(UnboundStructuralEpisode)}
    assert not any(
        forbidden in name
        for name in episode_fields
        for forbidden in ("gold", "expected", "target")
    )
    parameters = set(signature(recognize_structural_law).parameters)
    assert parameters == {"theory", "episode", "policy"}

    candidate = episode()
    assert {item.law_kind for item in candidate.candidate_projections} == set(
        LawKind
    )
    assert len({type(item) for item in candidate.candidate_projections}) == 1
    with pytest.raises(FrozenInstanceError):
        candidate.scope = ("changed",)


def test_recognition_implementation_id_binds_source_bytes():
    expected = "recognition_source_sha256_" + sha256(
        Path(recognition_module.__file__).read_bytes()
    ).hexdigest()
    assert RECOGNITION_IMPLEMENTATION_ID == expected


def test_unique_passing_proposal_is_selected_with_a_margin():
    decision = recognize_structural_law(theory=initial_theory(), episode=episode())
    assert decision.disposition is RecognitionDisposition.UNIQUE_MATCH
    assert not decision.abstained
    assert decision.selected_proposal is not None
    assert decision.selected_proposal.law_kind is LawKind.SYMMETRY
    assert decision.normalized_margin >= 1.0
    assert decision.reason == "unique_pass_with_normalized_margin"
    assert len(decision.evaluated_proposals) == len(LawKind)


def test_projection_order_does_not_change_content_or_decision():
    first_episode = episode()
    reversed_episode = episode(reverse_projections=True)
    assert first_episode.content_id == reversed_episode.content_id
    first = recognize_structural_law(
        theory=initial_theory(), episode=first_episode
    )
    second = recognize_structural_law(
        theory=initial_theory(), episode=reversed_episode
    )
    assert first == second
    assert first.decision_id == second.decision_id


def test_multiple_passes_and_missing_family_evidence_abstain():
    ambiguous = recognize_structural_law(
        theory=initial_theory(),
        episode=episode(passing=(LawKind.SYMMETRY, LawKind.MONOTONICITY)),
    )
    assert ambiguous.abstained
    assert ambiguous.selected_proposal is None
    assert ambiguous.reason == "ambiguous_multiple_passing_proposals"

    incomplete = recognize_structural_law(
        theory=initial_theory(),
        episode=episode(missing_kind=LawKind.SYMMETRY),
    )
    assert incomplete.abstained
    assert incomplete.reason == "incomplete_family_coverage"


def test_unique_pass_with_too_small_normalized_margin_abstains():
    decision = recognize_structural_law(
        theory=initial_theory(),
        episode=episode(near_competitor=True),
        policy=RecognitionPolicy(minimum_normalized_margin=3.0),
    )
    assert decision.abstained
    assert decision.reason == "insufficient_normalized_margin"
    assert decision.normalized_margin == pytest.approx(2.0)


STRICT_AXIS_POLICY = RecognitionPolicy(
    minimum_normalized_margin=1.0,
    require_completed_binding_competitor=True,
    require_completed_scale_competitor=True,
)


def test_policy_flags_are_strict_booleans():
    with pytest.raises(TypeError, match="booleans"):
        RecognitionPolicy(require_completed_binding_competitor=1)
    with pytest.raises(ValueError, match="finite"):
        RecognitionPolicy(minimum_normalized_margin=True)


def test_strict_policy_requires_completed_binding_and_scale_competitors():
    theory = two_scale_theory()
    complete = recognize_structural_law(
        theory=theory,
        episode=episode_with_axis_competitors(),
        policy=STRICT_AXIS_POLICY,
    )
    assert complete.disposition is RecognitionDisposition.UNIQUE_MATCH

    missing_binding = recognize_structural_law(
        theory=theory,
        episode=episode_with_axis_competitors(binding_competitor="missing"),
        policy=STRICT_AXIS_POLICY,
    )
    assert missing_binding.abstained
    assert missing_binding.reason == "missing_completed_binding_competitor"

    missing_scale = recognize_structural_law(
        theory=theory,
        episode=episode_with_axis_competitors(scale_competitor="missing"),
        policy=STRICT_AXIS_POLICY,
    )
    assert missing_scale.abstained
    assert missing_scale.reason == "missing_completed_scale_competitor"


@pytest.mark.parametrize(
    ("axis", "reason"),
    (
        ("binding", "missing_completed_binding_competitor"),
        ("scale", "missing_completed_scale_competitor"),
    ),
)
def test_abstained_axis_competitor_does_not_count_as_completed(axis, reason):
    modes = {
        "binding_competitor": "complete",
        "scale_competitor": "complete",
    }
    modes[f"{axis}_competitor"] = "abstain"
    decision = recognize_structural_law(
        theory=two_scale_theory(),
        episode=episode_with_axis_competitors(**modes),
        policy=STRICT_AXIS_POLICY,
    )
    assert decision.abstained
    assert decision.reason == reason


def test_semantic_and_name_metadata_cannot_change_structural_result():
    low = episode(semantic_overlap=0.01, semantic_name="violet gear")
    high = replace(
        low,
        semantic_metadata=freeze_pairs(
            {
                "semantic_overlap": 0.99,
                "name_hint": "perfect symmetry equivariance",
            }
        ),
    )
    assert low.content_id != high.content_id

    low_decision = recognize_structural_law(theory=initial_theory(), episode=low)
    high_decision = recognize_structural_law(theory=initial_theory(), episode=high)
    assert low_decision.disposition is high_decision.disposition
    assert low_decision.reason == high_decision.reason
    assert (
        low_decision.selected_proposal.proposal_id
        == high_decision.selected_proposal.proposal_id
    )
    assert low_decision.normalized_margin == high_decision.normalized_margin


def test_decision_invariants_block_ambiguous_to_unique_forgery():
    decision = recognize_structural_law(
        theory=initial_theory(),
        episode=episode(passing=(LawKind.SYMMETRY, LawKind.MONOTONICITY)),
    )
    passing = tuple(
        proposal
        for proposal in decision.evaluated_proposals
        if proposal.evaluation.passed
    )
    assert len(passing) == 2
    with pytest.raises(ValueError, match="deterministic derivation"):
        replace(
            decision,
            disposition=RecognitionDisposition.UNIQUE_MATCH,
            reason="unique_pass_with_normalized_margin",
            selected_proposal_id=passing[0].proposal_id,
            normalized_margin=999.0,
        )


def test_decision_binds_theory_version_and_replays_deterministically():
    theory = initial_theory()
    candidate = episode()
    decision = recognize_structural_law(theory=theory, episode=candidate)
    assert (
        replay_recognition_decision(
            theory=theory,
            episode=candidate,
            policy=RecognitionPolicy(),
            decision=decision,
        )
        == decision
    )

    changed_theory = replace(
        theory,
        signature=theory.signature + ("distinct_replay_context",),
    )
    changed_decision = recognize_structural_law(
        theory=changed_theory,
        episode=candidate,
    )
    assert decision.theory_version_id != changed_decision.theory_version_id
    assert decision.decision_id != changed_decision.decision_id

    with pytest.raises(ValueError, match="recognition implementation"):
        replace(
            decision,
            recognition_implementation_id="recognition_source_sha256_changed",
        )

    wrong_binding = replace(
        decision,
        theory_version_id=changed_theory.version_id,
    )
    with pytest.raises(ValueError, match="deterministic replay"):
        replay_recognition_decision(
            theory=theory,
            episode=candidate,
            policy=RecognitionPolicy(),
            decision=wrong_binding,
        )


def test_invalid_registry_scale_and_epoch_fail_closed_before_scoring():
    theory = initial_theory()
    valid = episode()
    with pytest.raises(ValueError, match="epochs"):
        replace(valid, evaluator_epoch="different_epoch")

    first = valid.candidate_projections[0]
    wrong_scale_projection = replace(first, scale_id="unregistered_scale")
    projections = (wrong_scale_projection,) + valid.candidate_projections[1:]
    with pytest.raises(ValueError, match="unavailable"):
        replace(
            valid,
            candidate_projections=projections,
        )


def test_preservation_witness_passes_only_for_epoch_scale_and_role_preservation():
    theory = initial_theory()
    source = recognize_structural_law(theory=theory, episode=episode(prefix="source"))
    target = recognize_structural_law(theory=theory, episode=episode(prefix="target"))
    source_roles = dict(source.selected_proposal.role_assignments)
    target_roles = dict(target.selected_proposal.role_assignments)
    entity_map = tuple(
        (source_roles[role], target_roles[role]) for role in sorted(source_roles)
    )
    scale_map = (("phase2_default", "phase2_default"),)

    witness = verify_preservation(
        source=source,
        target=target,
        entity_map=entity_map,
        scale_map=scale_map,
        evaluator_epoch=theory.evaluator.epoch,
    )
    assert witness.passed
    assert witness.failed_checks == ()
    assert witness.observed_residual_drift == 0.0
    assert witness.witness_id.startswith("preservation_witness_")
    with pytest.raises(ValueError, match="fixed schema"):
        replace(
            witness,
            checks=(("caller_supplied_pass", True),),
            passed=True,
        )

    wrong_epoch = verify_preservation(
        source=source,
        target=target,
        entity_map=entity_map,
        scale_map=scale_map,
        evaluator_epoch="wrong_epoch",
    )
    assert not wrong_epoch.passed
    assert "evaluator_epoch_preserved" in wrong_epoch.failed_checks

    wrong_scale = verify_preservation(
        source=source,
        target=target,
        entity_map=entity_map,
        scale_map=(("phase2_default", "wrong_scale"),),
        evaluator_epoch=theory.evaluator.epoch,
    )
    assert not wrong_scale.passed
    assert "scale_preserved" in wrong_scale.failed_checks

    reversed_targets = tuple(reversed([target for _, target in entity_map]))
    wrong_roles = tuple(
        (source_entity, target_entity)
        for (source_entity, _), target_entity in zip(
            entity_map, reversed_targets, strict=True
        )
    )
    wrong_role_witness = verify_preservation(
        source=source,
        target=target,
        entity_map=wrong_roles,
        scale_map=scale_map,
        evaluator_epoch=theory.evaluator.epoch,
    )
    assert not wrong_role_witness.passed
    assert "role_map_preserved" in wrong_role_witness.failed_checks

    changed_theory = replace(
        theory,
        signature=theory.signature + ("different_preservation_context",),
    )
    changed_target = recognize_structural_law(
        theory=changed_theory,
        episode=episode(prefix="target"),
    )
    wrong_theory = verify_preservation(
        source=source,
        target=changed_target,
        entity_map=entity_map,
        scale_map=scale_map,
        evaluator_epoch=theory.evaluator.epoch,
    )
    assert not wrong_theory.passed
    assert "theory_version_preserved" in wrong_theory.failed_checks


def test_unbound_episode_requires_all_six_frozen_candidate_families():
    valid = episode()
    with pytest.raises(ValueError, match="all six"):
        replace(valid, candidate_projections=valid.candidate_projections[:-1])
