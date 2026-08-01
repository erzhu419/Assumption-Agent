"""Construct the frozen Phase-2 baseline theory."""

from __future__ import annotations

from .hashing import stable_hash
from .laws import VERIFIER_REGISTRY_ID
from .ontology import ACTIVE_FUNCTIONALS, ACTIVE_LAWS, UNIVERSAL_ASSUMPTIONS
from .schema import EvaluatorSpec, ProbeSpec, ScaleContext, TheoryState


def initial_theory(
    *, data_cutoff: str = "2026-07-30T23:59:59+08:00"
) -> TheoryState:
    evaluator = EvaluatorSpec(
        evaluator_id="phase2_structural_evaluator",
        epoch="phase2_epoch_0002",
        version="0.2.0",
        scope=("controlled_offline_structural_laws",),
        anchor_ids=("anchor_low_semantic_positive", "anchor_high_semantic_negative"),
        failure_modes=(
            "missing_observable",
            "role_swap",
            "sign_flip",
            "scale_trap",
            "semantic_shortcut",
        ),
        adversarial_case_ids=(
            "adv_entity_rename",
            "adv_domain_mask",
            "adv_role_swap",
            "adv_sign_flip",
        ),
        frozen_at_cutoff="2026-07-30T00:00:00+08:00",
    )
    scale = ScaleContext(
        scale_id="phase2_default",
        task_id="known_law_recognition",
        axes=("episode", "declared_time_window", "declared_system_boundary"),
        aggregation="maximum_normalized_violation",
        validity_scope=("controlled_offline_fixture",),
    )
    probes = (
        ProbeSpec(
            "probe_exact_residual",
            "2",
            "typed_episode",
            "law_residual",
            "law_specific_violation",
            ("known_law_recognition",),
            evaluator.epoch,
            evaluator.anchor_ids,
            data_cutoff,
        ),
        ProbeSpec(
            "probe_hard_negative",
            "2",
            "typed_episode_pair",
            "contrastive_margin",
            "binary_rejection_margin",
            ("known_law_recognition",),
            evaluator.epoch,
            evaluator.anchor_ids,
            data_cutoff,
        ),
    )
    return TheoryState(
        schema_version="hegel-machine-theory/0.2",
        parent_version_id=None,
        signature=(
            "Observation",
            "TypedRole",
            "ScaleContext",
            "Probe",
            "RelationLaw",
            "ViolationFunctional",
        ),
        ontology_registry_id=stable_hash(
            UNIVERSAL_ASSUMPTIONS,
            prefix="ontology_registry_",
        ),
        verifier_registry_id=VERIFIER_REGISTRY_ID,
        model_classes=("deterministic_structural_episode",),
        representations=("typed_role_binding", "probe_outcome_distribution"),
        relation_laws=ACTIVE_LAWS,
        hypothesis_families=tuple(item.template_id for item in UNIVERSAL_ASSUMPTIONS),
        probes=probes,
        violation_functionals=ACTIVE_FUNCTIONALS,
        scales=(scale,),
        scope=("controlled_offline_structural_laws",),
        evaluator=evaluator,
        conditional_description_length=0.0,
        data_cutoff=data_cutoff,
    )
