"""The shared-data control runs native SC-KG without cognitive decisions."""

import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("initialization", ["shared_paid32", "native_initial8"])
def test_initialization_preserves_native_sc_kg_decisions_and_global_schedule(monkeypatch, initialization):
    root = Path(__file__).resolve().parents[1]
    project = root.parents[1] / "KG_op" / "SC-OLH-KG"
    if not (project / "algorithms/single_olhkg.py").exists():
        pytest.skip("the optional SC-KG checkout is unavailable")
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    monkeypatch.syspath_prepend(str(root / "scripts"))
    from run_sc_kg_vanilla_comparison import build_vanilla_optimizer

    saved = root / "artifacts/sc_kg_terminal_value_v040_20260910"
    if not (saved / "protocol.json").exists():
        pytest.skip("the actual shared initial observations are unavailable")
    spec = json.loads((saved / "protocol.json").read_text())
    case_id = "forward_1_mechanism_change_noise_0.00"
    case = next(row for row in spec["cases"] if row["case_id"] == case_id)
    reference = json.loads((saved / "runs" / f"{case_id}_terminal_value.json").read_text())
    records = [{"point": point, "output": output} for point, output in
               zip(reference["initial_points"], reference["initial_outputs"])]
    algorithm, problem = build_vanilla_optimizer(spec, case, records, initialization=initialization)
    from algorithms.single_olhkg import SingleOLHKGAlgorithm

    assert len(records) == 32
    assert algorithm.config.n0 == (32 if initialization == "shared_paid32" else 8)
    if initialization == "native_initial8":
        assert type(algorithm) is SingleOLHKGAlgorithm
    assert algorithm.config.N == 64
    assert all(model.basis_map is None for model in algorithm.gpr)
    for name in ("_generate_candidates", "_exact_posterior_update_scores",
                 "_exact_posterior_update_score_one", "_recommendation_pool",
                 "_solve_posterior_recommendation", "_simulate_and_store",
                 "_update_posterior_dominance_incumbent"):
        assert getattr(algorithm, name).__func__ is getattr(SingleOLHKGAlgorithm, name)
    for name in ("adoption_policy", "recovery", "objective_research"):
        assert getattr(algorithm, name, None) is None

    result = algorithm.run()
    assert len(algorithm.history) == result["n_simulations"] == 64
    if initialization == "shared_paid32":
        assert [list(point) for point, _ in algorithm.history[:32]] == reference["initial_points"]
        np.testing.assert_array_equal([output for _, output in algorithm.history[:32]],
                                      reference["initial_outputs"])
    else:
        assert [list(point) for point, _ in algorithm.history[:8]] != reference["initial_points"][:8]
    assert problem.actual_simulator_calls == (32 if initialization == "shared_paid32" else 64)
    assert problem.global_calls == 64
    assert [row["global_stage"] for row in problem.simulation_log] == list(
        range(33 if initialization == "shared_paid32" else 1, 65))
    assert all(row["mechanism"] == case["task"]["problem"] for row in problem.simulation_log
               if row["global_stage"] <= 44)
    assert all(row["mechanism"] == case["next_problem"] for row in problem.simulation_log
               if row["global_stage"] >= 45)
    assert all(model.basis_map is None and model.p == 2 * case["task"]["dimension"] + 1
               for model in algorithm.gpr)
    assert not getattr(algorithm, "objective_scope_revisions", [])
    assert not getattr(algorithm, "scope_revisions", [])
    assert len(algorithm.iteration_log) == 64 - algorithm.config.n0
    assert all(row["exact_kg_mc_samples"] == 1 for row in algorithm.iteration_log)
