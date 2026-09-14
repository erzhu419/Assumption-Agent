"""Check the public continual design without running or selecting an agent."""

from dataclasses import asdict
import importlib.util
from pathlib import Path
from statistics import mean, pvariance
import sys

import pytest

from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.models import Probe


@pytest.mark.parametrize("partition,seed", [("development", 7), ("confirmation", 23), ("forward", 101)])
def test_older_partitions_keep_the_saved_v3_observation_and_evaluation_design(partition, seed):
    package = Path(__file__).resolve().parents[1] / "artifacts" / "cognition_v3_reference"
    name = "cognition_v3_reference"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, package / "__init__.py",
                                                      submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    from cognition_v3_reference.environments import make_suite as saved_suite

    live, saved = make_suite(seed, partition), saved_suite(seed, partition)
    assert len(live) == len(saved)
    for left, right in zip(live, saved):
        assert asdict(left.spec) == asdict(right.spec)
        assert left.evaluation_metadata == right.evaluation_metadata
        for probe in (*left.spec.initial_probes, *left.spec.available_probes):
            assert [asdict(row) for row in left.observe(probe)] == [asdict(row) for row in right.observe(probe)]
        assert [asdict(row) for row in left.evaluation_rows()] == [asdict(row) for row in right.evaluation_rows()]


def test_continual_publishes_long_queries_and_keeps_mechanisms_evaluator_only():
    suite = make_suite(401, "continual")
    assert len(suite) == 12
    assert suite[0].spec.task_id == suite[-1].spec.task_id
    for environment in suite:
        spec = environment.spec
        assert spec.max_probe_length == 32
        assert {probe.cost for probe in spec.available_probes} == {8, 12, 16, 24, 32}
        assert all(probe.cost == 4 for probe in spec.initial_probes)
        assert spec.description == suite[0].spec.description and spec.context == suite[0].spec.context
        assert "mechanism" not in asdict(spec) and "partition" not in asdict(spec)
        assert environment.evaluation_metadata["evaluation_trajectory_length"] == "64"
    assert suite[5].spec.feature_names == ("r0", "r1")
    assert suite[6].spec.feature_names == ("v0", "v1")
    assert suite[10].spec.feature_names == ("w0", "w1")
    for environment in (suite[3], suite[10]):
        assert environment.spec.risk_quantile == 0.9
        assert environment.spec.decision_costs == pytest.approx((0.9, 0.1))


def test_long_trajectories_separate_input_delay_from_short_equivalent_target_delay():
    environment = make_suite(401, "continual")[0]
    probe = environment.spec.available_probes[-1]
    rows = environment.observe(probe)
    slope, delayed = environment._mechanism.slope, environment._mechanism.lag_weight

    def short_equivalent(row):
        prediction = slope * row.features["x"]
        if len(row.history) >= 4:
            prediction += (delayed / slope) * row.history[-4]["__target__"]
        return prediction

    assert [short_equivalent(row) for row in rows[:8]] == pytest.approx([row.target for row in rows[:8]])
    assert any(abs(short_equivalent(row) - row.target) > 0.1 for row in rows[8:])
    assert len(rows) == 32 and rows[0].history == () and len(rows[-1].history) == 31
    assert environment.observe(Probe("same_controls_new_id", probe.sequence)) == rows
    evaluation = environment.evaluation_rows()
    assert any(abs(short_equivalent(row) - row.target) > 0.1 for row in evaluation if len(row.history) >= 32)


def test_long_final_evaluation_is_independent_reproducible_and_resets_each_trajectory():
    environment = make_suite(503, "continual")[3]
    probe = environment.spec.available_probes[-1]
    observed = environment.observe(probe)
    evaluation = environment.evaluation_rows()
    assert len(evaluation) == 8 * 64
    assert [len(row.history) for row in evaluation] == list(range(64)) * 8
    assert environment.evaluation_rows() == evaluation
    assert environment.observe(probe) == observed
    training_controls = {tuple(sorted(point.items())) for query in
                         (*environment.spec.initial_probes, *environment.spec.available_probes)
                         for point in query.sequence}
    assert all(tuple(sorted(row.features.items())) not in training_controls for row in evaluation)
    assert make_suite(607, "continual")[3].evaluation_rows() != evaluation
    assert environment.evaluation_metadata["temporal_evaluation"] == "beyond_maximum_training_horizon"


def test_continual_risk_has_skew_and_control_dependent_noise_under_fixed_costs():
    environment = make_suite(607, "continual")[3]
    rows = environment.evaluation_rows()
    residuals = [(row.features["x"], row.target - (-0.7 * row.features["x"] + 0.35 + 0.25 * row.features["x"] ** 2))
                 for row in rows]
    low = [value for x, value in residuals if abs(x) < 0.7]
    high = [value for x, value in residuals if abs(x) > 1.7]
    assert pvariance(high) > 3 * pvariance(low)
    values = sorted(value for _, value in residuals)
    assert values[int(0.95 * len(values))] > abs(values[int(0.05 * len(values))])
    assert mean(values) > 0.0


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_cognition_v4_research.py"
    spec = importlib.util.spec_from_file_location("cognition_v4_research_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_saved_v3_continual_adapter_and_live_ablations_do_not_change_common_budgets(runner, monkeypatch):
    original_suite = runner.reference_experiments.make_suite
    base = runner.ResearchConfig(seed=401, partition="continual", observation_budget=96, compute_budget=24000)
    seen = []

    def saved_run(config):
        environments = runner.reference_experiments.make_suite(config.seed, partition=config.partition)
        assert config.partition == "forward"
        assert environments[0].evaluation_metadata["partition"] == "continual"
        assert environments[0].spec.max_probe_length == 32
        assert isinstance(environments[0].observe(environments[0].spec.initial_probes[0])[0],
                          runner.ReferenceObservation)
        assert "adaptive_horizon" not in asdict(config)
        return {"config": asdict(config)}

    monkeypatch.setattr(runner.reference_experiments, "run_sequence", saved_run)
    monkeypatch.setattr(runner.live_experiments, "run_sequence",
                        lambda config: seen.append(config) or {"config": asdict(config)})
    monkeypatch.setattr(runner, "evaluate_horizons", lambda run, config: {"additional_evaluation_units": 0})
    result = runner.run_system("reference_v3", base)
    assert result["partition"] == "continual" and result["reference_config_partition"] == "forward"
    assert runner.reference_experiments.make_suite is original_suite
    for system in runner.SYSTEM_CHANGES:
        runner.run_system(system, base)
    for system, config in zip(runner.SYSTEM_CHANGES, seen):
        assert asdict(config) == asdict(base) | runner.SYSTEM_CHANGES[system]
        assert config.proposer == "search"


def test_final_history_bins_score_recorded_programs_and_report_extra_work(runner, monkeypatch):
    from types import SimpleNamespace
    from hegel_machine.cognition.models import Observation

    rows = tuple(Observation({"x": 0.0}, 0.0 if index < 8 else 1.0 if index < 32 else 2.0,
                             tuple({"x": 0.0, "__target__": 0.0} for _ in range(index)))
                 for index in range(64))
    environment = SimpleNamespace(spec=SimpleNamespace(risk_quantile=0.5), evaluation_rows=lambda: rows)
    monkeypatch.setattr(runner, "make_suite", lambda seed, partition: [environment])
    episode = {"status": "completed", "program": runner.live_experiments.Program("const", value=0.0).to_dict(),
               "fallback": 0.0, "evaluation": {"loss_kind": "squared_error"}}
    run = {"system": "cognitive_v4", "episodes": [episode]}
    result = runner.evaluate_horizons(run, runner.ResearchConfig())
    bins = episode["evaluation"]["history_length_bins"]
    assert bins == {"0-7": {"rows": 8, "mean_test_loss": 0.0},
                    "8-31": {"rows": 24, "mean_test_loss": 1.0},
                    "32+": {"rows": 32, "mean_test_loss": 4.0}}
    assert result["additional_evaluation_units"] == 64
    assert result["wall_time_seconds"] >= 0.0
    assert result["by_loss"]["squared_error"]["32+"] == {"tasks": 1, "rows": 32, "mean_test_loss": 4.0}


def test_saved_memory_accepts_injected_counterexamples_with_unchanged_values(runner):
    from cognition_v3_reference.memory import KnowledgeLibrary
    from cognition_v3_reference.models import Candidate

    source, conflict = make_suite(401, "continual")[:2]
    adapted_source, adapted_conflict = runner._ReferenceEnvironment(source), runner._ReferenceEnvironment(conflict)
    initial = source.spec.initial_probes[0]
    observed = adapted_source.observe(initial)
    assert all(isinstance(row, runner.ReferenceObservation) for row in observed)
    assert [asdict(row) for row in observed] == [asdict(row) for row in source.observe(initial)]
    final = adapted_conflict.evaluation_rows()
    assert all(isinstance(row, runner.ReferenceObservation) for row in final)
    assert [asdict(row) for row in final] == [asdict(row) for row in conflict.evaluation_rows()]

    library = KnowledgeLibrary()
    concept_id, created = library.learn(Candidate("source", runner.reference_experiments.Program("feature", name="x"), "refit"),
                                        adapted_source.spec, observations=observed)
    assert created
    counterexample = adapted_conflict.observe(conflict.spec.available_probes[-1])[-1]
    event = library.record_counterexample(concept_id, adapted_conflict.spec, 2.0, 0.0, counterexample)
    assert event["action"] == "restricted"
    assert event["observation"]["features"] == counterexample.features
    assert event["observation"]["history"] == list(counterexample.history)
    assert event["observation"]["target"] == counterexample.target
