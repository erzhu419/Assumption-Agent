from hegel_machine.vertical_slice import run_controlled_vertical_slice


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
