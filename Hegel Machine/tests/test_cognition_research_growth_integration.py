import json
from dataclasses import replace

from hegel_machine.cognition.growth_lab import GrowthLab
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.research_growth import GrowthResearcher


def test_learned_program_survives_persistence_and_transfers_to_renamed_roles():
    config = ResearchConfig(observation_budget=384, compute_budget=200000,
                            max_rounds=6, minimum_improvement=1e-7, seed=8601)
    first = GrowthResearcher(config=config, measurement_budget=1024)
    source = first.run(GrowthLab(seed=8601, stage=0))
    state = json.loads(json.dumps(first.to_dict()))
    restored = GrowthResearcher(config=config, state=state)
    target = GrowthLab(seed=8601, stage=1)

    def unavailable_terminal(*args, **kwargs):
        raise AssertionError("the researcher requested evaluator-only information")

    target.evaluation_rows = unavailable_terminal
    result = restored.run(target)
    assert source["status"] == result["status"] == "confirmed"
    assert len(restored.to_dict()["tasks"]) == 2
    assert any(event["action"] == "fit" and event["details"]["binding_candidates"]
               for event in result["events"])
    original_program = Program.from_dict(source["program"])
    transferred_program = Program.from_dict(result["program"])
    assert original_program.required_features().isdisjoint(transferred_program.required_features())
    terminal = GrowthLab(seed=8601, stage=1).evaluation_rows(
        result["selected_channels"], seed=70861, length=32, trajectories=2)
    mse = sum((transferred_program.predict(row) - row.target) ** 2
              for row in terminal["observations"]) / len(terminal["observations"])
    assert mse < 1e-5
    assert result["costs"]["observation_cost"] == target.costs["observation_cost"]
    assert result["costs"]["measurement_cost"] == target.costs["measurement_cost"]
    assert result["costs"]["compute_cost"] <= config.compute_budget


def test_observability_method_revision_and_irreducible_noise_share_persistent_knowledge():
    config = ResearchConfig(observation_budget=384, compute_budget=200000,
                            max_rounds=6, minimum_improvement=1e-7, seed=8601)
    state, reports = None, {}
    for stage in (0, 3, 4, 5):
        researcher = GrowthResearcher(config=replace(config, seed=8601 + 10007 * stage),
                                      state=state, measurement_budget=1024)
        environment = GrowthLab(seed=8601, stage=stage)
        reports[stage] = researcher.run(environment)
        state = json.loads(json.dumps(researcher.to_dict()))
        assert reports[stage]["costs"]["observation_cost"] == environment.costs["observation_cost"]
        assert reports[stage]["costs"]["measurement_cost"] == environment.costs["measurement_cost"]
    for stage in (3, 4):
        report = reports[stage]
        assert report["status"] == "confirmed"
        assert report["uncertainty"]["kind"] == "deterministic"
        assert len(report["selected_channels"]) == 1
        assert any(event["action"] == "change_representation" for event in report["events"])
        data = GrowthLab(seed=8601, stage=stage).evaluation_rows(report["selected_channels"],
            seed=79001 + stage, length=32, trajectories=2)
        program = Program.from_dict(report["program"])
        assert sum((program.predict(row) - row.target) ** 2 for row in data["observations"]) / 64 < 1e-5
    assert any(method.get("parent_method_id") for method in state["methods"]["methods"])
    assert reports[5]["status"] == "confirmed"
    assert reports[5]["uncertainty"]["kind"] == "stochastic"
    assert .25 <= reports[5]["uncertainty"]["variance"] / (.12 ** 2) <= 4.
    assert any(event["action"] == "expand_uncertainty" for event in reports[5]["events"])
