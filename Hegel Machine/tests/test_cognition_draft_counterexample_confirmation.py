"""Use the real paid failure to check attribution before predictor fallback."""

import pytest

from test_cognition_method_joint_mismatch import joint_case, joint_factory, run_case


def test_fresh_failed_draft_records_actual_binding_not_all_offered_sources(joint_case, joint_factory):
    report, events = run_case(joint_case, joint_factory, additional_rows=72)
    withdrawn = [event for event in events if event["action"] == "withdraw_target_draft"]
    assert len(withdrawn) == 1
    event = withdrawn[0]
    assert event["source_draft_ids"] == ["research_draft_0011"]
    assert joint_case["candidate"]["method_opportunity"]["draft_ids"] == ["research_draft_0006"]
    assert event["evidence"]["noise_assessment"]["reason"] == "residual_loss_shift"
    pair = report["continuation"]["method_fresh_comparisons"][0]
    assert pair["attempted_source_draft_ids"] == event["source_draft_ids"]
    assert pair["attempted_program"] == event["program"]
    assert pair["attempted_fallback"] == event["fallback"]
    assert pair["attempted_channels"] == ["reading"]
    assert pair["attempted_after_loss"] == pytest.approx(110.72188620977069)
    assert pair["before_loss"] == pair["after_loss"] == pytest.approx(3.5745415304130512)
    assert report["program"] == joint_case["report"]["program"]
    assert report["visit_costs"] == {"observation_cost": 72, "measurement_cost": 72,
                                     "compute_cost": 144146, "queries": 3}
