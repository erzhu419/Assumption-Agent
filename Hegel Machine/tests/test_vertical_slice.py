from dataclasses import replace

import pytest

from hegel_machine.bootstrap import initial_theory
from hegel_machine.schema import EvidenceSplit
from hegel_machine.vertical_slice import _receipt, run_controlled_vertical_slice


def test_controlled_vertical_slice_stops_before_unsealed_promotion():
    report = run_controlled_vertical_slice()
    assert report["synthetic"] is True
    assert report["decision"] == "candidate_framework"
    assert report["shadow_graph_edge_count"] == 0
    assert report["shadow_graph_authoritative"] is False
    assert report["sealed_holdout"] is False
    assert report["negative_evidence_retained"] == report["hard_negative_count"]
    assert report["certificate_recorded"] is True
    assert report["metrics"]["old_success_preservation"] == 1.0
    assert report["metrics"]["unseen_prediction_success"] == 1.0
    assert report["metrics"]["regression_cost"] == 0.0
    assert report["promoted_child_version_id"] is None
    assert report["parent_version_id"] != report["candidate_preview_version_id"]
    assert "no downstream efficacy" in report["claim_scope"]


@pytest.mark.parametrize(
    ("metric", "probe_id", "split"),
    (
        (
            "residual_explanation",
            "probe_exact_residual",
            EvidenceSplit.VALIDATION,
        ),
        (
            "hard_negative_rejection",
            "probe_hard_negative",
            EvidenceSplit.HARD_NEGATIVE,
        ),
    ),
)
def test_vertical_receipts_use_the_registered_probe_version(metric, probe_id, split):
    parent = initial_theory()
    receipt = _receipt(
        parent,
        metric=metric,
        value=1.0,
        threshold=0.9,
        higher_is_better=True,
        split=split,
        observation_ids=("obs_dynamic_probe_version",),
    )
    registered = next(probe for probe in parent.probes if probe.probe_id == probe_id)
    assert receipt.probe_id == registered.probe_id
    assert receipt.probe_version == registered.version == "2"


def test_vertical_receipts_fail_closed_when_required_probe_is_unregistered():
    parent = initial_theory()
    parent_without_hard_negative_probe = replace(
        parent,
        probes=tuple(
            probe
            for probe in parent.probes
            if probe.probe_id != "probe_hard_negative"
        ),
    )
    with pytest.raises(ValueError, match="not registered"):
        _receipt(
            parent_without_hard_negative_probe,
            metric="hard_negative_rejection",
            value=1.0,
            threshold=0.9,
            higher_is_better=True,
            split=EvidenceSplit.HARD_NEGATIVE,
            observation_ids=("obs_missing_probe",),
        )
