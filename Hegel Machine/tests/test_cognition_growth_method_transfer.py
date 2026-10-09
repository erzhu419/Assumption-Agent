"""Isolate a learned procedure's value while keeping its program library fixed."""

from copy import deepcopy
from dataclasses import replace
import json

from hegel_machine.cognition.growth_lab import GrowthLab
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.research_growth import GrowthResearcher


class ResearchOnlyLab(GrowthLab):
    def evaluation_rows(self, *args, **kwargs):
        raise AssertionError("terminal outcomes must not enter method learning or selection")


def test_paid_method_revision_saves_research_after_instrument_roles_change():
    config = ResearchConfig(observation_budget=384, compute_budget=200000,
                            max_rounds=6, minimum_improvement=1e-7, seed=8601)
    state, prefix = None, []
    for stage in (0, 3):
        researcher = GrowthResearcher(config=replace(config, seed=8601 + 10007 * stage),
                                     state=state, measurement_budget=1024)
        report = researcher.run(ResearchOnlyLab(seed=8601, stage=stage))
        assert report["status"] == "confirmed"
        prefix.append(report)
        state = json.loads(json.dumps(researcher.to_dict()))
    shared_prefix = deepcopy(state)
    learned = [method for method in state["methods"]["methods"]
               if any(node["recipe"].get("instrument_comparison") for node in method["nodes"])]
    assert learned
    assert [node["action"] for node in learned[-1]["nodes"]] == ["measure", "fit", "confirm"]
    # Both arms get exactly the same purchased prefix and callable knowledge.
    # The intervention removes executable methods, not their source programs.
    reports = {}
    for arm in ("retained", "cleared"):
        initial = deepcopy(shared_prefix)
        if arm == "cleared":
            initial["methods"] = {}
            initial["library"]["research_workflows"] = {"methods": []}
        researcher = GrowthResearcher(config=replace(config, seed=8601 + 10007 * 4),
                                     state=initial, measurement_budget=1024)
        assert {key: value for key, value in researcher.library.to_dict().items()
                if key != "research_workflows"} == {
                    key: value for key, value in shared_prefix["library"].items()
                    if key != "research_workflows"}
        environment = ResearchOnlyLab(seed=8601, stage=4)
        reports[arm] = researcher.run(environment)
        report = reports[arm]
        assert report["status"] == "confirmed"
        assert report["costs"]["observation_cost"] == environment.costs["observation_cost"]
        assert report["costs"]["measurement_cost"] == environment.costs["measurement_cost"]
        assert report["costs"]["compute_cost"] <= config.compute_budget
        assert report["costs"]["observation_cost"] <= config.observation_budget
        assert report["costs"]["measurement_cost"] <= 1024
        assert state == shared_prefix
    # Evaluation uses matched new controls only after both research arms end.
    losses = {}
    for arm, report in reports.items():
        frozen = Program.from_dict(report["program"])
        rows = GrowthLab(seed=8601, stage=4).evaluation_rows(report["selected_channels"],
            seed=902101, length=64, trajectories=2)["observations"]
        losses[arm] = sum((frozen.predict(row) - row.target) ** 2 for row in rows) / len(rows)
    assert max(losses.values()) < 1e-5
    source_order = [channel.name for channel in GrowthLab(seed=8601, stage=3).channels]
    target_order = [channel.name for channel in GrowthLab(seed=8601, stage=4).channels]
    assert source_order.index(prefix[-1]["selected_channels"][0]) != target_order.index(
        reports["retained"]["selected_channels"][0])
    assert any(event["action"] == "change_representation" and
               event["details"].get("source") == "method" and
               event["details"].get("subquestions") == 2
               for event in reports["retained"]["events"])
    assert not any(event["action"] == "replay_method" for event in reports["cleared"]["events"])
    prefix_costs = {resource: sum(item["costs"][resource] for item in prefix)
                    for resource in reports["retained"]["costs"]}
    totals = {arm: {resource: prefix_costs[resource] + report["costs"][resource]
                   for resource in prefix_costs} for arm, report in reports.items()}
    assert totals["retained"]["observation_cost"] < totals["cleared"]["observation_cost"]
    assert totals["retained"]["compute_cost"] < totals["cleared"]["compute_cost"]
    assert totals["retained"]["measurement_cost"] <= totals["cleared"]["measurement_cost"]
