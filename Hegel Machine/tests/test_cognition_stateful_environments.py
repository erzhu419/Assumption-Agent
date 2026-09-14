"""Prospective stateful design and version adapters; no agent evaluation."""

from dataclasses import asdict
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.models import Observation, Probe


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_cognition_v5_research.py"
    spec = importlib.util.spec_from_file_location("cognition_v5_research_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("partition,seed", [("development", 7), ("confirmation", 23),
                                            ("forward", 101), ("continual", 401)])
def test_old_four_partitions_keep_saved_v4_values(runner, partition, seed):
    from cognition_v4_reference.environments import make_suite as saved_suite

    live, saved = make_suite(seed, partition), saved_suite(seed, partition)
    assert len(live) == len(saved)
    for current, previous in zip(live, saved):
        assert asdict(current.spec) == asdict(previous.spec)
        assert current.evaluation_metadata == previous.evaluation_metadata
        for probe in (*current.spec.initial_probes, *current.spec.available_probes):
            assert [asdict(row) for row in current.observe(probe)] == [asdict(row) for row in previous.observe(probe)]
        assert [asdict(row) for row in current.evaluation_rows()] == [asdict(row) for row in previous.evaluation_rows()]


def test_state_labels_and_mechanisms_are_absent_from_public_specs_and_history():
    suite = make_suite(809, "stateful")
    assert len(suite) == 12
    assert sum(env.evaluation_metadata["prediction_kind"] == "binary_state" for env in suite) == 8
    for environment in suite:
        spec = environment.spec
        assert spec.max_probe_length == 32
        assert spec.description == suite[0].spec.description and spec.context == suite[0].spec.context
        assert "prediction_kind" not in asdict(spec) and "mechanism" not in asdict(spec)
        assert "partition" not in asdict(spec)
        for row in environment.observe(spec.available_probes[0]):
            assert set(row.features) == set(spec.feature_names)
            assert all(set(previous) == {*spec.feature_names, "__target__"} for previous in row.history)
    assert suite[0].spec == suite[-1].spec


def test_latches_change_direction_and_driver_role_while_retaining_state():
    suite = make_suite(809, "stateful")
    xs = (-2.0, 0.0, 2.0, 0.0, -2.0)
    x_probe = Probe("x_sweep", tuple({"x": x, "z": 0.0} for x in xs))
    z_probe = Probe("z_sweep", tuple({"x": 0.0, "z": z} for z in xs))
    assert [row.target for row in suite[0].observe(x_probe)] == [0, 0, 1, 1, 0]
    assert [row.target for row in suite[1].observe(x_probe)] == [1, 1, 0, 0, 1]
    assert [row.target for row in suite[2].observe(x_probe)] == [0, 0, 0, 0, 0]
    assert [row.target for row in suite[2].observe(z_probe)] == [0, 0, 1, 1, 0]
    renamed = Probe("role_mapped", tuple({"b0": 0.0, "b1": x} for x in xs))
    assert [row.target for row in suite[6].observe(renamed)] == [1, 1, 0, 0, 1]


def test_combined_controls_can_cross_a_threshold_that_neither_crosses_alone():
    environment = make_suite(907, "stateful")[7]
    upper = environment._mechanism.upper_threshold
    x, z = 0.5 * upper, 0.75 * upper / 0.45
    alone_x = environment.observe(Probe("alone_x", ({"x": x, "z": 0.0},)))[0]
    alone_z = environment.observe(Probe("alone_z", ({"x": 0.0, "z": z},)))[0]
    combined = environment.observe(Probe("combined", ({"x": x, "z": z}, {"x": 0.0, "z": 0.0})))
    assert alone_x.target == alone_z.target == 0.0
    assert [row.target for row in combined] == [1.0, 1.0]
    assert environment.observe(Probe("fresh_reset", ({"x": 0.0, "z": 0.0},)))[0].target == 0.0


def test_same_context_changes_both_activation_and_retention_conditions():
    suite = make_suite(1009, "stateful")
    previous, changed = suite[0], suite[-1]
    assert previous.spec == changed.spec
    activation = Probe("activation", ({"x": 1.2, "z": 0.0},))
    retention = Probe("retention", ({"x": 2.0, "z": 0.0}, {"x": -0.4, "z": 0.0}))
    assert previous.observe(activation)[0].target == 1.0 and changed.observe(activation)[0].target == 0.0
    assert previous.observe(retention)[-1].target == 1.0 and changed.observe(retention)[-1].target == 0.0


def test_stateful_evaluation_is_reproducible_long_and_separate_from_queries():
    environment = make_suite(907, "stateful")[4]
    probe = environment.spec.available_probes[-1]
    observed = environment.observe(probe)
    evaluation = environment.evaluation_rows()
    assert len(evaluation) == 8 * 64
    assert [len(row.history) for row in evaluation] == list(range(64)) * 8
    assert environment.evaluation_rows() == evaluation and environment.observe(probe) == observed
    assert make_suite(1009, "stateful")[4].evaluation_rows() != evaluation
    assert environment.spec.risk_quantile == 0.9
    assert environment.spec.decision_costs == pytest.approx((0.9, 0.1))


def test_injected_saved_observations_reach_actual_memory_counterexample_path(runner, monkeypatch):
    from cognition_v4_reference.memory import KnowledgeLibrary
    from cognition_v4_reference.models import Candidate

    original = runner.reference_experiments.make_suite

    def saved_run(config):
        assert config.partition == "continual"
        first, second = runner.reference_experiments.make_suite(config.seed, partition=config.partition)[:2]
        first_rows = first.observe(first.spec.initial_probes[0])
        assert all(isinstance(row, runner.ReferenceObservation) for row in first_rows)
        library = KnowledgeLibrary()
        concept_id, _ = library.learn(Candidate("known", runner.reference_experiments.Program("feature", name="x"), "refit"),
                                      first.spec, observations=first_rows)
        example = second.observe(second.spec.available_probes[0])[-1]
        event = library.record_counterexample(concept_id, second.spec, 1.0, 0.0, example)
        assert event["action"] == "restricted"
        assert isinstance(second.evaluation_rows()[0], runner.ReferenceObservation)
        return {"config": asdict(config)}

    monkeypatch.setattr(runner.reference_experiments, "run_sequence", saved_run)
    monkeypatch.setattr(runner, "evaluate_horizons", lambda run, config: {})
    result = runner.run_system("reference_v4", runner.ResearchConfig(seed=809, partition="stateful"))
    assert result["partition"] == "stateful" and result["reference_config_partition"] == "continual"
    assert runner.reference_experiments.make_suite is original


def test_state_classification_has_correct_threshold_transition_and_history_denominators(runner, monkeypatch):
    targets = (0.0, 1.0, 1.0, 0.0)
    rows = tuple(Observation({"x": 0.0}, target,
                             tuple({"x": 0.0, "__target__": previous} for previous in targets[:index]))
                 for index, target in enumerate(targets))
    environment = SimpleNamespace(spec=SimpleNamespace(risk_quantile=0.5), evaluation_rows=lambda: rows,
                                  evaluation_metadata={"prediction_kind": "binary_state"})
    monkeypatch.setattr(runner, "make_suite", lambda seed, partition: [environment])
    episode = {"status": "completed", "program": runner.live_experiments.Program("const", value=0.5).to_dict(),
               "fallback": 0.0, "evaluation": {"loss_kind": "squared_error"}}
    result = runner.evaluate_horizons({"system": "cognitive_v5", "episodes": [episode]}, runner.ResearchConfig())
    state = result["state_classification"]
    assert state["rows"] == 4 and state["errors"] == 2
    assert state["false_positives"] == 2 and state["false_negatives"] == 0
    assert state["transition_rows"] == 2 and state["transition_errors"] == 1
    assert state["error_rate"] == state["balanced_error_rate"] == state["transition_error_rate"] == 0.5
    assert state["by_history"]["0-7"] == {"rows": 4, "errors": 2, "error_rate": 0.5}
    assert state["by_history"]["32+"]["error_rate"] is None
    assert result["additional_evaluation_units"] == 4


def test_live_policy_ablations_change_one_setting_and_keep_resource_limits(runner, monkeypatch):
    base = runner.ResearchConfig(seed=809, partition="stateful", observation_budget=96, compute_budget=24000)
    seen = []
    monkeypatch.setattr(runner.live_experiments, "run_sequence",
                        lambda config: seen.append(config) or {"config": asdict(config)})
    monkeypatch.setattr(runner, "evaluate_horizons", lambda run, config: {})
    for name in runner.SYSTEM_CHANGES:
        runner.run_system(name, base)
    for name, config in zip(runner.SYSTEM_CHANGES, seen):
        assert asdict(config) == asdict(base) | runner.SYSTEM_CHANGES[name]
        assert config.proposer == "search"
        if name != "without_knowledge":
            assert config.use_memory is True
