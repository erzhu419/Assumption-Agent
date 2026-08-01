import json
from collections import Counter
from dataclasses import fields, replace
from hashlib import sha256
from pathlib import Path

import pytest

import hegel_machine.phase2_exit as phase2_exit_module
from hegel_machine.hashing import canonical_json
from hegel_machine.milestones import (
    CURRENT_SCALE_CAPABILITY_NAME,
    CURRENT_TYPED_SELECTION_CAPABILITY_NAME,
    PHASE2A,
)
from hegel_machine.phase2_exit import (
    EXIT_SCALES,
    PHASE2_EXIT_RECOGNITION_POLICY,
    PROJECTION_ADAPTER_IMPLEMENTATION_ID,
    Phase2ExitThresholds,
    controlled_blinded_cases,
    controlled_blinded_corpus,
    frozen_projection_adapters,
    phase2_exit_theory,
    run_phase2_exit_benchmark,
)
from hegel_machine.recognition import (
    RecognitionDisposition,
    UnboundStructuralEpisode,
    recognize_structural_law,
)
from hegel_machine.schema import LawKind


def _replay_case(case, theory):
    adapters = {
        adapter.law_id: adapter for adapter in frozen_projection_adapters(theory)
    }
    return tuple(
        sorted(
            adapters[projection.law_id]
            .project(
                bundle=case.evidence,
                role_assignments=dict(projection.role_assignments),
                scale_id=projection.scale_id,
                evaluator_epoch=theory.evaluator.epoch,
            )
            .content_id
            for projection in case.episode.candidate_projections
        )
    )


def test_exit_corpus_is_uniform_and_answer_key_is_outside_episode_api():
    theory = phase2_exit_theory()
    episodes, answers = controlled_blinded_corpus(theory)
    assert len(episodes) == len(answers) == 43
    assert {answer.episode_id for answer in answers} == {
        episode.episode_id for episode in episodes
    }
    assert not {
        "expected_disposition",
        "expected_kind",
        "expected_roles",
        "expected_scale_id",
    }.intersection(item.name for item in fields(UnboundStructuralEpisode))
    assert {len(episode.candidate_projections) for episode in episodes} == {24}
    assert all(
        {projection.law_kind for projection in episode.candidate_projections}
        == set(LawKind)
        for episode in episodes
    )
    assert all(
        episode.available_scale_ids == tuple(sorted(EXIT_SCALES))
        for episode in episodes
    )

    for episode in episodes:
        for kind in LawKind:
            family = tuple(
                projection
                for projection in episode.candidate_projections
                if projection.law_kind is kind
            )
            role_maps = {projection.role_assignments for projection in family}
            assert len(family) == 4
            assert len(role_maps) == 2
            assert {projection.scale_id for projection in family} == set(EXIT_SCALES)
            assert {
                (projection.role_assignments, projection.scale_id)
                for projection in family
            } == {
                (role_map, scale_id)
                for role_map in role_maps
                for scale_id in EXIT_SCALES
            }


def test_public_identifiers_do_not_embed_plaintext_answer_or_control_cues():
    cases = controlled_blinded_cases()
    forbidden = {
        kind.value.lower() for kind in LawKind
    } | {
        "target",
        "gold",
        "correct",
        "wrong",
        "distractor",
        "control",
        "positive",
        "negative",
        "ambiguous",
        "missing",
        "rename",
        "binding",
        "scale",
    }
    for case in cases:
        public_ids = (
            case.episode.episode_id,
            *case.episode.observation_ids,
            *(entity_id for entity_id, _ in case.episode.typed_entities),
            *(
                projection.projection_id
                for projection in case.episode.candidate_projections
            ),
        )
        assert all(
            token not in public_id.lower()
            for public_id in public_ids
            for token in forbidden
        )
        semantic_hint = dict(case.episode.semantic_metadata)["semantic_name_hint"]
        assert all(token not in str(semantic_hint).lower() for token in forbidden)


def test_answerable_families_are_crossed_with_both_scales():
    _, answers = controlled_blinded_corpus()
    assert Counter(answer.control for answer in answers) == Counter(
        {
            "low_semantic_positive": 12,
            "entity_rename": 12,
            "high_semantic_hard_negative": 6,
            "sign_or_constraint_flip": 6,
            "missing_evidence": 6,
            "ambiguous": 1,
        }
    )
    answerable = tuple(
        answer
        for answer in answers
        if answer.expected_disposition is RecognitionDisposition.UNIQUE_MATCH
    )
    assert len(answerable) == 24
    assert Counter(
        (answer.expected_kind, answer.expected_scale_id) for answer in answerable
    ) == Counter(
        {
            (kind, scale_id): 2
            for kind in LawKind
            for scale_id in EXIT_SCALES
        }
    )

    preservation_pairs: dict[str, Counter[str]] = {}
    for answer in answerable:
        assert answer.pair_id is not None
        preservation_pairs.setdefault(answer.pair_id, Counter())[answer.control] += 1
    assert len(preservation_pairs) == 12
    assert all(
        controls == Counter({"low_semantic_positive": 1, "entity_rename": 1})
        for controls in preservation_pairs.values()
    )


def test_every_projection_replays_from_the_shared_evidence_bundle():
    theory = phase2_exit_theory()
    cases = controlled_blinded_cases(theory)
    for case in cases:
        adapters = {
            adapter.law_id: adapter
            for adapter in frozen_projection_adapters(theory)
        }
        expected = tuple(
            sorted(
                projection.content_id
                for projection in case.episode.candidate_projections
            )
        )
        assert case.projection_replay_ids == expected
        assert _replay_case(case, theory) == expected
        consumption: dict[str, int] = {}
        for projection in case.episode.candidate_projections:
            resolved = adapters[projection.law_id].resolve_measurements(
                bundle=case.evidence,
                role_assignments=dict(projection.role_assignments),
                scale_id=projection.scale_id,
            )
            for _, measurement in resolved:
                consumption[measurement.content_id] = (
                    consumption.get(measurement.content_id, 0) + 1
                )
        assert set(consumption) == {
            measurement.content_id for measurement in case.evidence.measurements
        }
        assert any(count > 1 for count in consumption.values())

    first = cases[0]
    shortened_evidence = replace(
        first.evidence,
        measurements=first.evidence.measurements[1:],
    )
    altered_case = replace(first, evidence=shortened_evidence)
    assert _replay_case(altered_case, theory) != first.projection_replay_ids

    changed_registry_theory = replace(
        theory,
        verifier_registry_id="verifier_registry_sha256_changed",
    )
    original_adapters = frozen_projection_adapters(theory)
    changed_adapters = frozen_projection_adapters(changed_registry_theory)
    assert {adapter.content_id for adapter in original_adapters}.isdisjoint(
        adapter.content_id for adapter in changed_adapters
    )
    assert _replay_case(first, changed_registry_theory) != first.projection_replay_ids


def test_exit_theory_freezes_two_scales_without_mutating_phase2_parent():
    theory = phase2_exit_theory()
    assert {scale.scale_id for scale in theory.scales} == set(EXIT_SCALES)
    assert all(set(law.scale_ids) == set(EXIT_SCALES) for law in theory.relation_laws)
    assert theory.parent_version_id is not None
    assert theory.evaluator.epoch == "phase2_exit_epoch_0002"
    assert theory.evaluator.version == "0.2.0"
    assert theory.ontology_registry_id.startswith("ontology_registry_")
    assert theory.verifier_registry_id.startswith("verifier_registry_sha256_")
    assert all(probe.evaluator_epoch == theory.evaluator.epoch for probe in theory.probes)


def test_projection_adapter_implementation_id_binds_source_bytes():
    expected = "projection_adapter_source_sha256_" + sha256(
        Path(phase2_exit_module.__file__).read_bytes()
    ).hexdigest()
    assert PROJECTION_ADAPTER_IMPLEMENTATION_ID == expected
    with pytest.raises(ValueError, match="implementation registry drift"):
        replace(
            frozen_projection_adapters(phase2_exit_theory())[0],
            implementation_registry_id="projection_adapter_source_sha256_changed",
        )


def test_exit_thresholds_reject_nonfinite_or_out_of_range_values():
    with pytest.raises(ValueError, match="finite"):
        Phase2ExitThresholds(binding_accuracy=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        Phase2ExitThresholds(binding_accuracy=True)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Phase2ExitThresholds(scale_selection=1.01)


def test_detached_answer_object_is_not_a_recognizer_parameter():
    # This proves only API separation.  Fixture values remain evaluator-
    # conditioned, as disclosed by the benchmark report.
    theory = phase2_exit_theory()
    episodes, answers = controlled_blinded_corpus(theory)
    episode = episodes[0]
    before = recognize_structural_law(
        theory=theory,
        episode=episode,
        policy=PHASE2_EXIT_RECOGNITION_POLICY,
    )
    forged_answer = replace(
        answers[0],
        expected_disposition=RecognitionDisposition.ABSTAIN,
        expected_kind=None,
        expected_roles=(),
        expected_scale_id=None,
    )
    assert forged_answer != answers[0]
    after = recognize_structural_law(
        theory=theory,
        episode=episode,
        policy=PHASE2_EXIT_RECOGNITION_POLICY,
    )
    assert before == after


def test_phase2_exit_report_measures_every_required_control():
    report = run_phase2_exit_benchmark()
    assert report["benchmark"] == "phase2_api_blinded_selector_mechanics_v2"
    assert report["milestone_id"] == PHASE2A.machine_id
    assert report["milestone_name"] == PHASE2A.name
    assert report["capability_name"] == CURRENT_TYPED_SELECTION_CAPABILITY_NAME
    assert report["scale_capability_name"] == CURRENT_SCALE_CAPABILITY_NAME
    assert report["synthetic"] is True
    assert report["development_fixture_only"] is True
    assert report["source_visible_generator"] is True
    assert report["sealed_holdout"] is False
    assert report["formal_phase2_exit_claim"] is False
    assert report["context_conditioned_scale_inference_qualified"] is False
    assert report["status"] == "controlled_api_selector_qualified"
    assert report["recognizer_receives_answer_key"] is False
    assert report["fixture_values_conditioned_on_evaluator_case_spec"] is True
    assert report["independent_raw_evidence_projection_qualified"] is False
    assert report["untrusted_recognizer_isolation_qualified"] is False
    assert report["candidate_labels_are_hypotheses_not_answers"] is True
    assert report["cross_candidate_measurement_reuse_required"] is True
    assert report["raw_extractor_qualified"] is False
    assert report["semantic_metadata_used_for_acceptance"] is False
    assert report["semantic_control_is_real_embedding_baseline"] is False
    assert report["abstention_is_statistically_calibrated"] is False
    assert report["active_graph_mutated"] is False
    assert report["recognition_implementation_id"].startswith(
        "recognition_source_sha256_"
    )
    assert report["projection_adapter_implementation_id"] == (
        PROJECTION_ADAPTER_IMPLEMENTATION_ID
    )
    assert report["case_count"] == 43
    assert report["answerable_case_count"] == 24
    assert report["abstention_case_count"] == 19
    assert report["projection_count_per_case"] == 24
    assert report["expected_preservation_pair_count"] == 12
    assert "frozen evaluator answer table" in report["preservation_mapping_source"]
    assert report["common_outer_schema"] is True
    assert len(report["preservation_witness_ids"]) == 12
    assert len(report["preservation_witnesses"]) == 12
    assert all(witness["passed"] for witness in report["preservation_witnesses"])
    assert all(
        witness["scale_map"]
        == [[scale_id, scale_id] for scale_id in sorted(EXIT_SCALES)]
        for witness in report["preservation_witnesses"]
    )
    assert all(report["exit_checks"].values())
    assert "semantic_control_gap" not in report["exit_checks"]
    assert "synthetic_semantic_decoy_accuracy" not in report["exit_checks"]
    assert "structural_gain_over_synthetic_decoy" not in report["exit_checks"]

    metrics = report["metrics"]
    for name in (
        "family_classification_accuracy",
        "binding_accuracy",
        "scale_selection_accuracy",
        "hard_negative_rejection",
        "role_binding_counterfactual_rejection",
        "scale_counterfactual_rejection",
        "sign_flip_sensitivity",
        "deterministic_abstention_accuracy",
        "shared_measurement_reuse_accuracy",
        "adapter_projection_replay_accuracy",
        "decision_replay_accuracy",
        "identifier_value_invariance_accuracy",
        "cross_episode_preservation",
        "structural_exact_decision_accuracy",
    ):
        assert metrics[name] == 1.0
    assert metrics["synthetic_semantic_decoy_accuracy"] == 0.0
    assert "raw extraction" in report["claim_scope"]
    assert "not a secrecy boundary" in report["identifier_blinding_scope"]
    assert report["recognition_policy"] == {
        "minimum_normalized_margin": 1.0,
        "require_complete_family_coverage": True,
        "require_completed_binding_competitor": True,
        "require_completed_scale_competitor": True,
    }


def test_phase2_exit_report_replays_deterministically():
    first = run_phase2_exit_benchmark()
    second = run_phase2_exit_benchmark()
    assert first == second
    assert first["report_id"] == second["report_id"]


def test_checked_in_phase2_exit_artifact_matches_runtime_report():
    artifact_path = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "phase2_exit_benchmark_v2.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact == run_phase2_exit_benchmark()


def test_shared_measurement_values_are_canonical_json():
    for case in controlled_blinded_cases():
        for measurement in case.evidence.measurements:
            assert canonical_json(measurement.value) == measurement.value_json


def test_shared_evidence_rejects_duplicate_witness_keys():
    case = controlled_blinded_cases()[0]
    first = case.evidence.measurements[0]
    duplicate_key = replace(
        first,
        measurement_id=first.measurement_id + "_duplicate",
    )
    with pytest.raises(ValueError, match="witness key"):
        replace(
            case.evidence,
            measurements=case.evidence.measurements + (duplicate_key,),
        )
