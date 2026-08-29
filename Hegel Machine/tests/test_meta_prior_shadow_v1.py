import copy
import json
from pathlib import Path

import pytest

from hegel_machine.bootstrap import initial_theory
from hegel_machine.diagnostics import InadequacyVerdict
from hegel_machine.meta_prior_shadow_v1 import (
    SHADOW_STATUS,
    run_bounded_meta_prior_shadow,
)
from hegel_machine.schema import EvidenceSplit, PatchCoordinate


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/meta_prior_bounded_dual_branch_shadow_v1.json"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_residual_rows_do_not_supply_aggregate_decisions():
    body = config()
    forbidden = {
        "tail_utility",
        "safety_utility",
        "coverage",
        "complexity_cost",
        "compute_gain",
        "sample_gain",
        "complexity_gain",
        "maximum_probe_discrepancy",
    }
    for branch in ("robustification", "idealization"):
        for group in (
            "discovery_rows",
            "holdout_rows",
            "rejection_control_rows",
        ):
            for row in body[branch][group]:
                assert forbidden.isdisjoint(row)


def test_bounded_dual_branch_shadow_positive_and_rejection_controls():
    report = run_bounded_meta_prior_shadow(config())
    parent = initial_theory(data_cutoff=config()["opened_at_cutoff"])

    assert report.status == SHADOW_STATUS
    assert not report.external_evidence_claimed
    assert not report.formal_state_mutated
    assert not report.q05b_claimed
    assert not report.q1_claimed
    assert not report.c1_claimed
    assert not report.active_claimed

    robust = report.robustification
    assert robust.passed
    assert (
        robust.discovery.diagnosis.verdict
        is InadequacyVerdict.ROBUSTIFICATION_CANDIDATE
    )
    assert robust.discovery.candidate.nominal_utility == pytest.approx(0.885)
    assert robust.discovery.candidate.tail_utility == pytest.approx(0.775)
    assert robust.discovery.parent_tail_utility == pytest.approx(0.35)
    assert robust.discovery.candidate.safety_utility == 1.0
    assert robust.discovery.candidate.coverage == 1.0
    assert robust.discovery.candidate.reduction_map.maximum_error == 0.0
    assert robust.discovery.candidate.reduction_map.old_scope == parent.scope
    assert robust.patch.coordinate is PatchCoordinate.ROBUSTIFICATION
    assert (
        robust.patch.reduction_map_id
        == robust.discovery.candidate.reduction_map.reduction_id
    )
    assert robust.holdout_receipt.split is EvidenceSplit.HOLDOUT
    assert robust.holdout_receipt.independent
    assert robust.holdout_receipt.preregistration_id == robust.patch.prediction_ids[0]
    assert robust.holdout_receipt.observation_ids == robust.holdout_observation_ids
    assert set(robust.discovery.discovery_observation_ids).isdisjoint(
        robust.holdout_observation_ids
    )
    assert not robust.rejected_control.accepted
    assert "tail_gain" in robust.rejected_control.reason
    assert "nominal_loss" in robust.rejected_control.reason

    ideal = report.idealization
    assert ideal.passed
    assert ideal.discovery.diagnosis.verdict is InadequacyVerdict.IDEALIZATION_CANDIDATE
    assert ideal.discovery.contract.maximum_probe_discrepancy == pytest.approx(0.01)
    assert ideal.discovery.contract.compute_gain == pytest.approx(0.7)
    assert ideal.discovery.contract.sample_gain == pytest.approx(0.5)
    assert ideal.discovery.contract.complexity_gain == pytest.approx(0.5)
    assert ideal.discovery.reduction_map.maximum_error == 0.0
    assert ideal.discovery.reduction_map.old_scope == parent.scope
    assert ideal.discovery.contract.failure_boundary == (
        "fine_scale_microstate_sensitive_query",
    )
    assert ideal.discovery.contract.counterexample_ids
    assert ideal.patch.coordinate is PatchCoordinate.IDEALIZATION
    assert ideal.patch.scope == tuple(config()["idealization"]["scope"])
    assert ideal.patch.reduction_map_id == ideal.discovery.reduction_map.reduction_id
    assert ideal.holdout_receipt.split is EvidenceSplit.HOLDOUT
    assert ideal.holdout_receipt.independent
    assert ideal.holdout_receipt.preregistration_id == ideal.patch.prediction_ids[0]
    assert ideal.holdout_receipt.observation_ids == ideal.holdout_observation_ids
    assert set(ideal.discovery.discovery_observation_ids).isdisjoint(
        ideal.holdout_observation_ids
    )
    assert not ideal.rejected_control.accepted
    assert ideal.rejected_control.reason == "failed:probe_preservation"


def test_unseen_stress_outcome_does_not_change_compiled_patch():
    base_config = config()
    base = run_bounded_meta_prior_shadow(base_config)
    failed_config = copy.deepcopy(base_config)
    failed_config["robustification"]["holdout_rows"][0]["candidate_utility"] = 0.36
    failed = run_bounded_meta_prior_shadow(failed_config)

    assert (
        failed.robustification.patch.content_id
        == base.robustification.patch.content_id
    )
    assert (
        failed.robustification.discovery.candidate
        == base.robustification.discovery.candidate
    )
    assert not failed.robustification.holdout_receipt.passed
    assert failed.status == "REJECTED"


def test_reduction_replay_failures_reject_both_branches():
    robust_config = config()
    robust_config["robustification"]["discovery_rows"][-1][
        "candidate_prediction"
    ] = 0.61
    robust_report = run_bounded_meta_prior_shadow(robust_config)
    assert not robust_report.robustification.discovery.decision.accepted
    assert (
        robust_report.robustification.discovery.decision.reason
        == "failed:reduction_replay"
    )

    ideal_config = config()
    ideal_config["idealization"]["discovery_rows"][0]["recovered_output"] = 0.51
    ideal_report = run_bounded_meta_prior_shadow(ideal_config)
    assert not ideal_report.idealization.discovery.decision.accepted
    assert (
        ideal_report.idealization.discovery.decision.reason
        == "failed:reduction_replay"
    )


def test_idealization_requires_a_real_failure_boundary_counterexample():
    body = config()
    boundary = body["idealization"]["discovery_rows"][2]
    boundary["idealized_output"] = boundary["full_output"]

    with pytest.raises(
        ValueError,
        match="boundary-counterexample rows must exceed the probe budget",
    ):
        run_bounded_meta_prior_shadow(body)
