"""Choose and use an instrument through the public experimental interface."""

from math import isfinite

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import make_lab
from hegel_machine.cognition.measurement_research import MeasurementResearchAgent
from hegel_machine.cognition.models import Candidate, ResearchConfig, SearchBudget
from hegel_machine.cognition.probing import choose_probe, choose_valuable_probe, construct_probes
from hegel_machine.cognition.selection import representative_probe
from hegel_machine.cognition.workspace import ResearchWorkspace


@pytest.fixture(scope="module")
def instrument_research():
    config = ResearchConfig(observation_budget=128, compute_budget=48000,
                            max_rounds=2, use_memory=False)
    results = {}
    for strategy in ("learned", "controls_only", "all"):
        with make_lab(0, seed=113) as laboratory:
            result = MeasurementResearchAgent(config, measurement_budget=640).run(laboratory, strategy=strategy)
            program = Program.from_dict(result["episode"]["program"])
            costs = dict(laboratory.costs)
            fresh = laboratory.evaluation_rows(result["selected_channels"], seed=833,
                                               length=64, trajectories=3)
            results[strategy] = {
                "result": result, "program": program, "costs": costs,
                "acquisitions": list(laboratory.acquisitions),
                "evaluation_loss": loss(program, fresh, result["episode"]["fallback"], .5),
            }
            assert laboratory.costs == costs
    return config, results


def test_research_selects_the_useful_measurement_and_predicts_long_new_trajectories(instrument_research):
    _, results = instrument_research
    learned, controls, all_instruments = (results[name] for name in ("learned", "controls_only", "all"))

    assert learned["result"]["selected_channels"] == ["sensor_a"]
    assert "sensor_a" in learned["program"].required_features()
    assert learned["evaluation_loss"] < 1e-6
    assert controls["evaluation_loss"] > .1
    assert all_instruments["evaluation_loss"] < 1e-6
    assert all_instruments["result"]["selected_channels"] == ["sensor_a", "sensor_b"]
    assert learned["costs"]["measurement_cost"] < all_instruments["costs"]["measurement_cost"]
    selection = {tuple(item["channels"]): item for item in learned["result"]["selection"]}
    assert selection[("sensor_a",)]["validation_loss"] < selection[()]["validation_loss"]
    assert selection[("sensor_a",)]["cost_per_row"] < selection[("sensor_a", "sensor_b")]["cost_per_row"]


def test_pilots_and_selected_measurements_are_paid_from_the_same_real_budgets(instrument_research):
    config, results = instrument_research
    for arm in results.values():
        result, acquisitions = arm["result"], arm["acquisitions"]
        reported = result["costs"]
        assert 0 < reported["observation_cost"] <= config.observation_budget
        assert 0 <= reported["measurement_cost"] <= result["measurement_budget"]
        assert 0 < reported["compute_cost"] <= config.compute_budget
        assert reported["queries"] == len(acquisitions)
        for name in ("observation_cost", "measurement_cost"):
            assert reported[name] == arm["costs"][name] == sum(query[name] for query in acquisitions)
        pilot_rows = sum(len(trajectory) for trajectory in result["pilot_observations"])
        assert reported["observation_cost"] == pilot_rows + result["episode"]["observation_cost"]
        assert reported["compute_cost"] == (sum(item["compute_cost"] for item in result["selection"])
                                               + result["initial_comparison_compute_cost"]
                                               + result["episode"]["compute_cost"])
        assert all(query["channels"] == result["selected_channels"] for query in acquisitions[2:])
        for action in result["episode"]["actions"]:
            if action["action"] == "observe":
                assert all(set(point) == {"u"} for point in action["details"]["sequence"])
    assert results["controls_only"]["costs"]["measurement_cost"] == 0


def test_sensor_forecasts_replay_acquired_readings_while_actual_probes_only_set_controls(instrument_research):
    _, results = instrument_research
    learned = results["learned"]
    workspace = ResearchWorkspace.from_dict(learned["result"]["episode"]["research_state"])
    program = learned["program"]
    alternate = Program("add", children=(program, Program("mul", children=(
        Program("const", value=.2), Program("feature", name="sensor_a")))))
    candidates = [Candidate("retained", program, "representation"),
                  Candidate("sensor_gain_revision", alternate, "refit")]
    with make_lab(0, seed=197) as laboratory:
        spec = laboratory.spec_for(("sensor_a",))
        probes = construct_probes(spec, candidates, round_number=5)
        representative = representative_probe(spec, seed=29, index=0, remaining=12, candidates=candidates)
        assert representative is not None
        probes.append(representative)
        assert probes and all(set(point) == {"u"} for probe in probes for point in probe.sequence)
        assert all(-1. <= point["u"] <= 1. for probe in probes for point in probe.sequence)
        budget = SearchBudget(6000)
        selected, details = choose_valuable_probe(probes, candidates, workspace.fallback, budget,
                                                 observations=workspace.observations, spec=spec)
        assert selected in probes
        assert details["status"] == "measured_value"
        assert details["sensor_scenarios"] == {
            "measurements": ["sensor_a"], "paths": 3, "source": "acquired_readings"}
        assert len(details["candidate_weights"]) == 2 and details["evaluated_probes"]
        assert isfinite(details["reference_risk"])
        assert 0 < details["compute_cost"] == budget.spent <= 1200
        acquired = laboratory.acquire(selected, ("sensor_a",))
        assert acquired.observation_cost == acquired.measurement_cost == selected.cost
        assert all(set(row.features) == {"u", "sensor_a"} for row in acquired.observations)
        assert isfinite(loss(program, acquired.observations, workspace.fallback, .5))


def test_disagreement_policy_uses_acquired_sensor_paths_and_pays_for_each_forecast(instrument_research):
    _, results = instrument_research
    learned = results["learned"]
    workspace = ResearchWorkspace.from_dict(learned["result"]["episode"]["research_state"])
    candidates = [Candidate("measured_memory", learned["program"], "representation"),
                  Candidate("control_coordinate", Program("feature", name="u"), "representation")]
    with make_lab(0, seed=317) as laboratory:
        probes = construct_probes(laboratory.spec_for(("sensor_a",)), candidates, round_number=6)
        budget = SearchBudget(1000)
        selected, disagreement = choose_probe(probes, candidates, workspace.fallback, budget,
                                              observations=workspace.observations)
        assert selected in probes and isfinite(disagreement) and disagreement > 0
        assert budget.spent == sum(probe.cost for probe in probes) * len(candidates) * 3
        assert budget.spent <= budget.limit
        assert all(set(point) == {"u"} for point in selected.sequence)
        acquired = laboratory.acquire(selected, ("sensor_a",))
        assert acquired.observation_cost == acquired.measurement_cost == selected.cost
        assert isfinite(loss(learned["program"], acquired.observations, workspace.fallback, .5))
