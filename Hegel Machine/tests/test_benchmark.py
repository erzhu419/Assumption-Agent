from hegel_machine.benchmark import run_phase2_benchmark


def test_phase2_benchmark_has_anti_semantic_controls():
    report = run_phase2_benchmark()
    assert report["synthetic"] is True
    assert report["case_count"] == 12
    assert report["structural_accuracy"] == 1.0
    assert report["positive_recall"] == 1.0
    assert report["hard_negative_rejection"] == 1.0
    assert report["semantic_only_accuracy"] == 0.0
    assert report["entity_rename_invariance"] is True
    assert report["missing_boundary_abstention"] is True
    assert report["claim_scope"].startswith("controlled offline")
