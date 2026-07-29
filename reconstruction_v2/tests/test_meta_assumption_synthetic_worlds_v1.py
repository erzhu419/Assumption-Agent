from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from assumption_agent.benchmarks import (
    meta_assumption_synthetic_worlds_v1 as subject,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER = PROJECT_ROOT / "scripts" / "qualify_meta_assumption_synthetic_v1.py"


def test_numeric_mechanisms_are_deterministic_and_runtime_qualified() -> None:
    first = subject.qualify()
    second = subject.qualify()

    assert first == second
    assert subject.canonical_bytes(first) == subject.canonical_bytes(second)
    assert first["formal_result"] is False
    assert first["efficacy_evidence"] is False
    assert first["mechanism_families"] == list(subject.FAMILIES)
    assert first["structural_variants"] == ["a", "b"]
    assert first["world_count"] == 10
    assert first["correct_identification_count"] == 10
    assert first["all_known_mechanisms_identified"] is True
    assert first["structural_variant_nonisomorphism_verified"] is True
    assert len(first["structural_variant_commitments"]) == 5
    assert first["numeric_payload_shape"] == {
        "action_fold_utilities": [4, 6],
        "subset_utility_folds": [2, 64],
        "adjacency": [6, 6],
        "node_effect_folds": [4, 6],
        "observation_folds": [4, 8],
        "decision_payoffs": [4, 4],
    }
    assert first["minimum_commitment_two_stage"] is True
    assert first["probe_evidence_bundle_count"] == 50
    assert len(first["probe_verifier_trust_anchors"]) == 5
    assert first["all_probe_receipts_trusted_recomputed"] is True
    assert len(first["probe_matrix_rows"]) == 50
    assert first["safe_recomputed_counts"] == {
        "world_count": 10,
        "correct_identification_count": 10,
        "wrong_claim_count": 40,
        "wrong_claims_with_counterevidence_count": 40,
        "runtime_active_trial_count": 8,
        "runtime_active_differential_count": 8,
        "runtime_noop_trial_count": 2,
        "runtime_noop_semantic_equivalence_count": 2,
        "wrong_operator_trial_count": 32,
        "wrong_operator_harm_count": 32,
        "wrong_operator_harm_world_count": 10,
        "tamper_case_count": 19,
        "tamper_rejected_count": 19,
    }
    assert set(first["metamorphic_trials"]) == {
        "claim_order",
        "probe_rule_order",
        "world_id",
        "expected_label",
    }
    assert all(
        row["trial_count"] == 10
        and row["invariant_count"] == 10
        and row["all_invariant"] is True
        and len(row["content_commitment"]) == 64
        for row in first["metamorphic_trials"].values()
    )
    assert first["wrong_claim_count"] == 40
    assert first["wrong_claims_with_counterevidence_count"] == 40
    assert first["all_wrong_claims_counterevidenced"] is True
    assert first["all_compilation_receipts_valid"] is True
    assert first["runtime_active_trial_count"] == 8
    assert first["runtime_active_differential_count"] == 8
    assert first["runtime_noop_trial_count"] == 2
    assert first["runtime_noop_semantic_equivalence_count"] == 2
    assert first["wrong_operator_trial_count"] == 32
    assert first["wrong_operator_harm_count"] == 32
    assert first["wrong_operator_harm_world_count"] == 10
    assert first["all_wrong_operators_harmful"] is True
    assert len(first["probe_statistic_commitments"]) == 10
    assert first["tamper_case_count"] == 19
    assert first["tamper_rejected_count"] == 19
    assert first["all_tampers_rejected"] is True
    assert all(
        row["rejected"]
        and row["cause_type"] == "PermissionError"
        and row["expected_issue_ids"] == row["observed_issue_ids"]
        for row in first["tamper_rejections"]
    )
    assert first["no_op_disposition"] == (
        "preserve_baseline_program_none"
    )
    for counter in (
        "formal_source_access_count",
        "source_payload_access_count",
        "network_call_count",
        "model_asset_access_count",
        "api_call_count",
        "online_evaluator_call_count",
        "validation_access_count",
        "test_access_count",
    ):
        assert first[counter] == 0
    body = dict(first)
    declared = body.pop("self_sha256")
    assert declared == subject.semantic_hash(body)


def test_selector_receives_only_same_shape_integer_numeric_payload() -> None:
    artifacts = subject.build_qualification_artifacts()
    forbidden_fragments = (*subject.FAMILIES, "expected", "label")

    assert len(artifacts.worlds) == 10
    assert len(
        {world.selector_input.world_id for world in artifacts.worlds}
    ) == 10
    assert len({world.claim_order for world in artifacts.worlds}) > 1
    payload_shapes = set()
    for world in artifacts.worlds:
        safe = world.selector_input.safe_payload()
        encoded = json.dumps(safe, sort_keys=True)
        assert set(safe) == {"world_id", "numeric_payload"}
        assert not any(fragment in encoded for fragment in forbidden_fragments)
        assert set(world.claim_order) == {
            claim.claim_id
            for claim in artifacts.claims_by_family.values()
        }
        payload = world.selector_input.numeric_payload
        assert payload.validate() == ()
        assert all(
            type(value) is int
            for panel in (
                payload.action_fold_utilities,
                payload.subset_utility_folds,
                payload.node_effect_folds,
                payload.observation_folds,
                payload.decision_payoffs,
            )
            for row in panel
            for value in row
        )
        payload_shapes.add(
            (
                tuple(map(len, payload.action_fold_utilities)),
                tuple(map(len, payload.subset_utility_folds)),
                tuple(map(len, payload.adjacency)),
                tuple(map(len, payload.node_effect_folds)),
                tuple(map(len, payload.observation_folds)),
                tuple(map(len, payload.decision_payoffs)),
            )
        )
    assert len(payload_shapes) == 1


def test_two_variants_change_mechanism_structure_not_only_magnitude() -> None:
    worlds = subject._build_worlds()
    commitments = subject._structural_variant_commitments(worlds)
    assert {row["family"] for row in commitments} == set(subject.FAMILIES)
    assert all(
        row["variant_a_structure_hash"] != row["variant_b_structure_hash"]
        for row in commitments
    )

    by_family = {
        family: sorted(
            (
                world
                for world in worlds
                if world.expected_family == family
            ),
            key=lambda world: world.variant,
        )
        for family in subject.FAMILIES
    }
    sparse_descriptors = [
        subject._variant_structure_descriptor(world)
        for world in by_family["sparse"]
    ]
    assert {
        descriptor["active_set_size"]
        for descriptor in sparse_descriptors
    } == {2, 3}
    interaction_descriptors = [
        subject._variant_structure_descriptor(world)
        for world in by_family["set_interaction"]
    ]
    assert (
        interaction_descriptors[0]["degree_sequence"]
        != interaction_descriptors[1]["degree_sequence"]
    )
    contamination_descriptors = [
        subject._variant_structure_descriptor(world)
        for world in by_family["contamination"]
    ]
    assert {
        tuple(descriptor["outlier_count_per_fold"])
        for descriptor in contamination_descriptors
    } == {(1, 1, 1, 1), (2, 2, 2, 2)}
    local_descriptors = [
        subject._variant_structure_descriptor(world)
        for world in by_family["local"]
    ]
    assert (
        local_descriptors[0]["degree_sequence"]
        != local_descriptors[1]["degree_sequence"]
    )
    noop_descriptors = [
        subject._variant_structure_descriptor(world)
        for world in by_family["no_op"]
    ]
    assert (
        noop_descriptors[0]["positive_action_incidence"]
        != noop_descriptors[1]["positive_action_incidence"]
    )
    assert (
        sorted(
            noop_descriptors[0]["positive_action_count_by_context"]
        ),
        sorted(
            noop_descriptors[0]["positive_context_count_by_action"]
        ),
    ) != (
        sorted(
            noop_descriptors[1]["positive_action_count_by_context"]
        ),
        sorted(
            noop_descriptors[1]["positive_context_count_by_action"]
        ),
    )


def test_numeric_probe_matrix_is_exactly_one_support_four_counters() -> None:
    artifacts = subject.build_qualification_artifacts()

    for family, claim in artifacts.claims_by_family.items():
        template = artifacts.ontology.require_template(
            subject.TEMPLATE_ID_BY_FAMILY[family]
        )
        assert claim.observable_predictions == template.support_signatures
        assert claim.counter_predictions == template.counter_signatures

    for result in artifacts.world_qualifications:
        assert result.selected_family == result.world.expected_family
        supported = [
            computation
            for computation in result.probe_computations
            if computation.receipt.disposition
            is subject.ProbeDisposition.SUPPORTED
        ]
        falsified = [
            computation
            for computation in result.probe_computations
            if computation.receipt.disposition
            is subject.ProbeDisposition.FALSIFIED
        ]
        assert len(supported) == 1
        assert len(falsified) == 4
        assert supported[0].receipt.claim_hash == (
            result.selected_claim.claim_hash
        )
        assert supported[0].receipt.support_count == 1
        assert supported[0].receipt.counter_count == 0
        assert supported[0].receipt.observed_support_signature_ids
        for computation in falsified:
            assert computation.receipt.support_count == 0
            assert computation.receipt.counter_count == 1
            assert not computation.receipt.observed_support_signature_ids
            assert computation.receipt.observed_counter_signature_ids
        for computation in result.probe_computations:
            claim = next(
                claim
                for claim in artifacts.claims_by_family.values()
                if claim.claim_hash == computation.receipt.claim_hash
            )
            template_id = claim.template_ids[0]
            plan = artifacts.ontology.require_template(
                template_id
            ).probe_plan
            evidence = computation.evidence_bundle
            assert evidence.validate(
                ontology=artifacts.ontology,
                claim=claim,
            ) == ()
            assert evidence.bundle_id == evidence.expected_bundle_id
            assert (
                computation.receipt.evidence_bundle_hash
                == evidence.evidence_bundle_hash
            )
            assert (
                computation.receipt.probe_trust_anchor_hash
                == artifacts.probe_verifier_registry
                .require_trust_anchor(
                    artifacts.ontology.require_template(template_id)
                )
                .anchor_hash
            )
            assert artifacts.probe_verifier_registry.verify_receipt(
                computation.receipt,
                ontology=artifacts.ontology,
                claim=claim,
                evidence=evidence,
            ) == ()
            committed_values = tuple(
                (key, int(value))
                for key, value in (
                    evidence.observation_statistics[0].statistic_values
                )
            )
            assert computation.statistic_values == committed_values
            assert set(plan.observable_ids).issubset(
                dict(committed_values)
            )
            if template_id == subject.TEMPLATE_ID_BY_FAMILY["no_op"]:
                assert dict(committed_values)[
                    "rule_active_claim_falsified_count"
                ] == (
                    4
                    if result.world.expected_family == "no_op"
                    else 3
                )


def test_minimum_commitment_requires_all_active_probes_falsified() -> None:
    artifacts = subject.build_qualification_artifacts()
    no_op_world = next(
        world
        for world in artifacts.worlds
        if world.expected_family == "no_op"
    )
    claim = artifacts.claims_by_family["no_op"]
    plan = artifacts.probe_plans_by_family["no_op"]
    premature = subject._observe_claim(
        ontology=artifacts.ontology,
        probe_verifier_registry=artifacts.probe_verifier_registry,
        selector_input=no_op_world.selector_input,
        claim=claim,
        plan=plan,
        contextual_statistics={
            "rule_active_claim_falsified_count": 3,
        },
    )
    assert (
        premature.receipt.disposition
        is subject.ProbeDisposition.INCONCLUSIVE
    )
    assert not premature.receipt.observed_support_signature_ids
    assert not premature.receipt.observed_counter_signature_ids


def test_policy_runtime_executes_closed_active_lane_and_real_noop() -> None:
    artifacts = subject.build_qualification_artifacts()
    observed: dict[str, set[str]] = {
        family: set() for family in subject.FAMILIES
    }
    for result in artifacts.world_qualifications:
        operator = subject._operator_from_treatment(result.treatment)
        observed[result.selected_family].add(operator)
        subject.validate_compilation_binding(
            result.compilation_receipt,
            result.binding,
        )
        evidence = result.runtime_evidence
        assert (
            evidence.wrong_operator_harm_count
            == evidence.wrong_operator_trial_count
        )
        if result.selected_family == "no_op":
            assert (
                result.treatment.disposition
                is subject.TreatmentDisposition.PRESERVE_BASELINE
            )
            assert result.treatment.program is None
            assert result.treatment.recipe_ids == ()
            assert result.treatment.recipe_action_bindings == ()
            assert evidence.noop_runtime_equivalent is True
            assert evidence.baseline_plan_hash == evidence.candidate_plan_hash
            assert evidence.baseline_utility == evidence.candidate_utility
            assert evidence.selected_lane == subject.BASELINE_LANE
        else:
            assert (
                result.treatment.disposition
                is subject.TreatmentDisposition.ACTIVE_PROGRAM
            )
            assert result.treatment.program is not None
            assert result.treatment.program.validate() == []
            assert {
                action.operation
                for action in result.treatment.program.action_graph
            } == {"enable_lane", "prioritize_lane", "set_parameter"}
            assert len(result.treatment.recipe_action_bindings) == 1
            assert evidence.active_runtime_differential is True
            assert evidence.candidate_utility > evidence.baseline_utility
            assert evidence.selected_lane == subject.OPERATOR_LANE
            assert evidence.baseline_plan_hash != evidence.candidate_plan_hash
    assert observed == {
        family: {operator}
        for family, operator in subject.OPERATOR_BY_FAMILY.items()
    }


def test_oracle_utility_is_numeric_and_independent_of_expected_label() -> None:
    for world in subject._build_worlds():
        alternate_family = next(
            family
            for family in subject.FAMILIES
            if family != world.expected_family
        )
        relabeled = replace(world, expected_family=alternate_family)
        for recipe in subject.ACTIVE_OPERATORS:
            answer = subject._closed_recipe_decision(
                world.selector_input.numeric_payload,
                recipe,
            )
            assert subject._oracle_utility(world, answer) == (
                subject._oracle_utility(relabeled, answer)
            )
    artifacts = subject.build_qualification_artifacts()
    active = next(
        result
        for result in artifacts.world_qualifications
        if result.selected_family == "sparse"
    )
    selected_probe = next(
        computation.receipt
        for computation in active.probe_computations
        if computation.receipt.claim_hash
        == active.selected_claim.claim_hash
    )
    relabeled_evidence = subject._runtime_evidence(
        world=replace(active.world, expected_family="no_op"),
        claim=active.selected_claim,
        selected_probe=selected_probe,
        treatment=active.treatment,
    )
    assert relabeled_evidence == active.runtime_evidence
    assert relabeled_evidence.wrong_operator_trial_count == 3


def test_safe_probe_matrix_and_pure_recompute_are_auditable() -> None:
    receipt = subject.qualify()
    world_rows = receipt["world_compilations"]
    probe_rows = receipt["probe_matrix_rows"]
    tamper_rows = receipt["tamper_rejections"]
    expected_probe_keys = {
        "world_id",
        "template_id",
        "claim_id",
        "claim_hash",
        "disposition",
        "observed_support_signature_ids",
        "observed_counter_signature_ids",
        "probe_trust_anchor_hash",
        "statistic_commitment_hash",
        "evidence_bundle_hash",
        "probe_receipt_hash",
    }
    assert len(world_rows) == 10
    assert len(probe_rows) == 50
    assert all(
        row["expected_template_id"]
        in subject.TEMPLATE_ID_BY_FAMILY.values()
        for row in world_rows
    )
    assert all(set(row) == expected_probe_keys for row in probe_rows)
    assert not any(
        forbidden in key
        for row in probe_rows
        for key in row
        for forbidden in ("payload", "statistic_values", "observation")
    )
    recomputed = subject.recompute_safe_qualification_counts(
        world_rows=world_rows,
        probe_matrix_rows=probe_rows,
        tamper_rows=tamper_rows,
    )
    assert recomputed == receipt["safe_recomputed_counts"]

    changed_world_rows = [dict(row) for row in world_rows]
    changed_world_rows[0]["selected_template_id"] = (
        "uao.v1.t19_minimum_commitment"
        if changed_world_rows[0]["expected_template_id"]
        != "uao.v1.t19_minimum_commitment"
        else "uao.v1.t02_sparsity"
    )
    changed = subject.recompute_safe_qualification_counts(
        world_rows=changed_world_rows,
        probe_matrix_rows=probe_rows,
        tamper_rows=tamper_rows,
    )
    assert changed["correct_identification_count"] == 9


def test_tamper_matrix_requires_exact_contract_issue_and_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = subject.build_qualification_artifacts()
    outcomes = subject._fixed_tamper_matrix(artifacts)
    assert len(outcomes) == 19
    assert {
        "probe_trust_anchor",
        "probe_evidence_bundle",
        "probe_statistic_commitment",
    }.issubset(outcome.case_id for outcome in outcomes)
    assert all(outcome.rejected for outcome in outcomes)
    assert all(
        outcome.observed_issue_ids == outcome.expected_issue_ids
        and outcome.cause_type == "PermissionError"
        for outcome in outcomes
    )

    def unrelated_failure(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("simulated unrelated verifier failure")

    monkeypatch.setattr(
        subject, "verify_compilation_receipt", unrelated_failure
    )
    unrelated = subject._fixed_tamper_matrix(artifacts)
    assert not any(outcome.rejected for outcome in unrelated)
    assert all(outcome.observed_issue_ids == () for outcome in unrelated)


def test_worker_accepts_no_channels_and_emits_only_canonical_receipt() -> None:
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(WORKER)],
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stderr == b""
    value = json.loads(completed.stdout.decode("ascii"))
    assert completed.stdout == subject.canonical_bytes(value)
    assert value == subject.qualify()

    refused = subprocess.run(
        [sys.executable, "-I", "-B", str(WORKER), "--output", "/tmp/x"],
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert refused.returncode == 2
    assert refused.stdout == b""
    assert refused.stderr == b""


def test_worker_import_closure_has_no_external_capability_module() -> None:
    tree = ast.parse(WORKER.read_text(encoding="ascii"))
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert imported_roots <= {
        "__future__",
        "pathlib",
        "sys",
        "assumption_agent",
    }
