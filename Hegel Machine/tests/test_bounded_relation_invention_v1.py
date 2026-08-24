from dataclasses import replace
import inspect
from pathlib import Path

import pytest

from hegel_machine.bounded_relation_invention_v1 import (
    CLAIM_LEVEL,
    NONCLAIMS,
    STATUS,
    LabeledRelationExample,
    PredictionCommitment,
    Program,
    RelationInventionError,
    build_public_synthetic_fixture,
    derive_ontology_report,
    enumerate_programs,
    preregister_holdout_predictions,
    require_candidate_mechanics_report,
    run_bounded_relation_invention_mechanics_v1,
    synthesize_relation,
    _select_old_program,
)
from hegel_machine.bootstrap import initial_theory
from hegel_machine.diagnostics import InadequacyVerdict
from hegel_machine.hashing import canonical_json, stable_hash
from hegel_machine.schema import (
    EvidenceSplit,
    IdentifiabilityCertificate,
    LawKind,
    PreregisteredPrediction,
    ReductionMap,
    RelationLawCandidate,
    ViolationFunctionalSpec,
)


def _gated_fixture():
    discovery, validation, hard_negatives, vault = build_public_synthetic_fixture()
    profile, report, old_program = derive_ontology_report(discovery, validation)
    synthesis = synthesize_relation(discovery, validation, report)
    return (
        discovery,
        validation,
        hard_negatives,
        vault,
        profile,
        report,
        old_program,
        synthesis,
    )


def test_public_synthetic_relation_invention_is_shadow_only_and_deterministic():
    first = run_bounded_relation_invention_mechanics_v1()
    second = run_bounded_relation_invention_mechanics_v1()

    assert canonical_json(first) == canonical_json(second)
    require_candidate_mechanics_report(first)
    assert first["status"] == STATUS
    assert first["claim_level"] == CLAIM_LEVEL
    assert first["candidate_kind"] is None
    assert first["candidate_program"] == (
        "and(drive>=0,or(drive>=2,prior_on))"
    )
    assert first["candidate_program_node_count"] == 5
    assert first["old_comparison_max_node_count"] == 3
    assert first["old_comparison_accuracy"] == pytest.approx(5 / 6)
    assert first["old_comparison_holdout_accuracy"] == 0.75
    assert first["ontology_verdict"] == "ontology_defect"
    assert first["chronology"] == [
        "synthetic_profile_diagnostic_ladder_replayed",
        "candidate_synthesis_committed",
        "holdout_prediction_preregistered",
        "process_local_holdout_opened",
        "holdout_scored_without_candidate_change",
    ]
    assert first["parent_theory_bytes_preserved"] is True
    assert first["active_theory_mutated"] is False
    assert first["cheaper_repair_evaluators_executed"] is False
    assert first["diagnostic_profile_source"] == (
        "frozen_public_synthetic_fixture_assertions_not_executed_"
        "repair_evaluators"
    )
    assert first["compiler_status"] == (
        "LANGUAGE_COMPILER_CLOSED_CANDIDATE_BRANCH_ONLY"
    )
    assert first["five_distinct_role_actor_ids"] is True
    assert first["role_actor_ids_external"] is False
    assert first["holdout_labels_passed_to_synthesis_api"] is False
    assert first["reduction_map_declared_not_executed"] is True
    assert first["unmeasured_gate_metrics"] == [
        "old_success_preservation",
        "limiting_case_reduction",
        "regression_cost",
    ]
    assert set(first["measured_synthetic_metrics"]) == {
        "synthetic_discovery_accuracy",
        "synthetic_validation_accuracy",
        "process_local_holdout_accuracy",
        "public_hard_negative_accuracy",
        "synthetic_expressivity_gain",
        "synthetic_description_compression_gain",
    }
    assert first["sealed_holdout"] is False
    assert first["external_trust_root"] is False
    assert first["c1_pass"] is False
    assert first["c2_pass"] is False
    assert first["promotion_allowed"] is False
    assert first["nonclaims"] == list(NONCLAIMS)


def test_checked_artifact_is_exact_canonical_replay():
    report = run_bounded_relation_invention_mechanics_v1()
    artifact = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "bounded_relation_invention_mechanics_v1.json"
    )
    assert artifact.read_bytes() == (canonical_json(report) + "\n").encode("utf-8")


def test_frozen_grammar_is_complete_and_candidate_is_a_new_abstraction():
    programs = enumerate_programs()
    assert len(programs) == 1_980
    assert len({program.expression for program in programs}) == len(programs)
    assert tuple(program.node_count for program in programs) == tuple(
        sorted(program.node_count for program in programs)
    )

    *_, synthesis = _gated_fixture()
    assert synthesis.exact_behavior_class_count == 84
    assert synthesis.minimum_exact_node_count == 5
    assert synthesis.old_comparison_program.expression == "drive>=0"
    assert synthesis.old_comparison_program.node_count == 1
    assert synthesis.selected_program.node_count > 3
    with pytest.raises(RelationInventionError, match="not program differs"):
        Program(
            "not",
            children=(Program("not", children=(Program("prior_on"),)),),
        )


def test_synthetic_profile_ladder_precedes_and_binds_synthesis():
    discovery, validation, *_ = build_public_synthetic_fixture()
    assert tuple(inspect.signature(_select_old_program).parameters) == ("discovery",)
    assert _select_old_program(discovery).expression == "drive>=0"
    profile, report, _ = derive_ontology_report(discovery, validation)
    assert report.verdict is InadequacyVerdict.ONTOLOGY_DEFECT
    assert report.language_extension_allowed
    assert profile.refit_gain == 0.0
    assert profile.low_order_composition_gain == 0.0
    assert profile.preregistered_prediction_gain == pytest.approx(1 / 6)

    signature = inspect.signature(synthesize_relation)
    assert tuple(signature.parameters) == (
        "discovery",
        "validation",
        "ontology_report",
    )
    with pytest.raises(RelationInventionError, match="requires an ontology"):
        synthesize_relation(discovery, validation, None)  # type: ignore[arg-type]

    mutated_validation = list(validation)
    mutated_validation[0] = replace(
        mutated_validation[0], next_on=not mutated_validation[0].next_on
    )
    with pytest.raises(
        RelationInventionError,
        match="does not bind|residual mask did not transfer",
    ):
        synthesize_relation(discovery, tuple(mutated_validation), report)


def test_low_order_relation_is_stopped_as_an_old_language_repair():
    discovery, validation, *_ = build_public_synthetic_fixture()

    def relabel(example: LabeledRelationExample) -> LabeledRelationExample:
        return replace(
            example,
            next_on=example.relation_input.drive >= 2,
        )

    simple_discovery = tuple(relabel(item) for item in discovery)
    simple_validation = tuple(relabel(item) for item in validation)
    with pytest.raises(RelationInventionError, match="old low-order language"):
        derive_ontology_report(simple_discovery, simple_validation)


def test_discovery_residual_probe_rejects_false_positives():
    discovery, validation, *_ = build_public_synthetic_fixture()
    old_program = _select_old_program(discovery)
    index = next(
        index
        for index, item in enumerate(discovery)
        if 0 <= item.relation_input.drive < 2
        and item.relation_input.prior_on is False
    )
    mutated = list(discovery)
    mutated[index] = replace(
        mutated[index],
        next_on=old_program.evaluate(mutated[index].relation_input),
    )
    with pytest.raises(RelationInventionError, match="false positives or negatives"):
        derive_ontology_report(tuple(mutated), validation)


def test_holdout_prediction_is_frozen_before_open_and_cannot_be_reopened():
    discovery, validation, _, vault, _, report, _, synthesis = _gated_fixture()
    del discovery, validation, report
    assert vault.opened is False
    prediction = preregister_holdout_predictions(synthesis, vault.inputs)
    assert vault.opened is False
    opened = vault.open(
        prediction,
        expected_synthesis_commitment_id=synthesis.commitment_id,
    )
    assert vault.opened is True
    assert tuple(item.split for item in opened) == (
        EvidenceSplit.HOLDOUT,
    ) * len(opened)
    with pytest.raises(RelationInventionError, match="already opened"):
        vault.open(
            prediction,
            expected_synthesis_commitment_id=synthesis.commitment_id,
        )


def test_holdout_vault_rejects_prediction_for_another_input_generation():
    *_, vault, _, _, _, synthesis = _gated_fixture()
    prediction = preregister_holdout_predictions(synthesis, vault.inputs)
    wrong = replace(
        prediction,
        holdout_input_root="holdout_input_partition_" + "0" * 64,
    )
    with pytest.raises(RelationInventionError, match="does not bind"):
        vault.open(
            wrong,
            expected_synthesis_commitment_id=synthesis.commitment_id,
        )
    assert vault.opened is False


def test_holdout_commitment_rejects_forged_synthesis_and_malformed_vectors():
    *_, vault, _, _, _, synthesis = _gated_fixture()
    prediction = preregister_holdout_predictions(synthesis, vault.inputs)
    forged = replace(prediction, synthesis_commitment_id="forged_synthesis")
    with pytest.raises(RelationInventionError, match="expected synthesis"):
        vault.open(
            forged,
            expected_synthesis_commitment_id=synthesis.commitment_id,
        )
    assert vault.opened is False

    with pytest.raises(RelationInventionError, match="cardinality"):
        PredictionCommitment(
            synthesis.commitment_id,
            prediction.holdout_input_root,
            prediction.input_ids_in_order,
            (),
        )
    with pytest.raises(RelationInventionError, match="Boolean"):
        PredictionCommitment(
            synthesis.commitment_id,
            prediction.holdout_input_root,
            prediction.input_ids_in_order,
            ("true",) * len(prediction.input_ids_in_order),  # type: ignore[arg-type]
        )


def test_candidate_program_contains_no_fixture_ids_or_outcomes():
    discovery, validation, hard_negatives, *_ = _gated_fixture()
    report = run_bounded_relation_invention_mechanics_v1()
    executable = report["candidate_program"]
    assert isinstance(executable, str)
    assert "next_on" not in executable
    for example in discovery + validation + hard_negatives:
        assert example.observation_id not in executable
        assert example.relation_input.input_id not in executable


def test_report_validator_rejects_authority_and_content_id_tampering():
    report = run_bounded_relation_invention_mechanics_v1()
    overclaim = dict(report)
    overclaim["c2_pass"] = True
    with pytest.raises(RelationInventionError, match="overclaims"):
        require_candidate_mechanics_report(overclaim)

    tampered = dict(report)
    tampered["candidate_symbol"] = "post_hoc_symbol"
    with pytest.raises(RelationInventionError, match="content id differs"):
        require_candidate_mechanics_report(tampered)

    recomputed = dict(report)
    recomputed["unmeasured_gate_metrics"] = []
    body = dict(recomputed)
    body.pop("report_content_id")
    recomputed["report_content_id"] = stable_hash(
        body,
        prefix="bounded_relation_invention_report_",
    )
    with pytest.raises(RelationInventionError, match="frozen replay"):
        require_candidate_mechanics_report(recomputed)

    extra_claim = dict(report)
    extra_claim["formal_gate_pass"] = True
    body = dict(extra_claim)
    body.pop("report_content_id")
    extra_claim["report_content_id"] = stable_hash(
        body,
        prefix="bounded_relation_invention_report_",
    )
    with pytest.raises(RelationInventionError, match="frozen replay"):
        require_candidate_mechanics_report(extra_claim)


def test_unknown_candidate_kind_is_supported_but_mismatch_fails_closed():
    prediction = PreregisteredPrediction(
        prediction_id="prediction_test",
        input_condition="frozen input",
        outcome_name="Boolean outcome",
        expected_direction="exact",
        expected_range=(1.0, 1.0),
        failure_criterion="any mismatch",
        registered_at_cutoff="2026-08-24T00:00:00+08:00",
    )
    reduction = ReductionMap(
        reduction_id="reduction_test",
        parent_version_id="parent_test",
        child_candidate_id="candidate_test",
        old_scope=("old",),
        mapping_description="identity outside new scope",
        executable_check_id="check_test",
        maximum_error=0.0,
    )
    functional = ViolationFunctionalSpec(
        "vf_unknown_candidate",
        None,
        ("x", "y"),
        "zero for a match",
        0.0,
    )
    candidate = RelationLawCandidate(
        "candidate_test",
        "new_symbol",
        None,
        1,
        ("entity",),
        "x",
        functional,
        ("new",),
        ("phase2_default",),
        ("positive",),
        ("negative",),
        (prediction,),
        reduction,
        1.0,
        IdentifiabilityCertificate(
            "identifiability_test",
            ("truth vector",),
            ("toggle x",),
            ("renaming",),
            ("outside fixture",),
            True,
        ),
    )
    assert candidate.kind is None
    with pytest.raises(ValueError, match="kinds disagree"):
        replace(
            candidate,
            violation_functional=replace(
                functional, law_kind=LawKind.MONOTONICITY
            ),
        )

    theory = initial_theory()
    with pytest.raises(ValueError, match="concrete registered LawKind"):
        replace(theory.relation_laws[0], kind=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="functional kinds must be concrete"):
        replace(
            theory,
            violation_functionals=(
                replace(
                    theory.violation_functionals[0],
                    law_kind=None,
                ),
                *theory.violation_functionals[1:],
            ),
        )
    with pytest.raises(ValueError, match="registered LawKind or None"):
        replace(functional, law_kind="invented")  # type: ignore[arg-type]


def test_prediction_commitment_requires_exact_input_order():
    *_, vault, _, _, _, synthesis = _gated_fixture()
    prediction = preregister_holdout_predictions(synthesis, vault.inputs)
    wrong = PredictionCommitment(
        prediction.synthesis_commitment_id,
        prediction.holdout_input_root,
        tuple(reversed(prediction.input_ids_in_order)),
        prediction.predicted_outcomes_in_order,
    )
    with pytest.raises(RelationInventionError, match="does not bind"):
        vault.open(
            wrong,
            expected_synthesis_commitment_id=synthesis.commitment_id,
        )
