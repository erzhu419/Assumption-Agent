"""Paid counterexamples survive a short pilot and same-view representation repair."""

import json

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import make_lab
from hegel_machine.cognition.measurement_research import (
    MeasurementResearchAgent, _pilot_probe, _trajectories, measured_view,
)
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.workspace import ResearchWorkspace


def _config():
    return ResearchConfig(observation_budget=96, compute_budget=60000,
                          max_rounds=3, seed=501038)


@pytest.fixture(scope="module")
def unresolved_representation():
    with make_lab(1, 501038) as laboratory:
        agent = MeasurementResearchAgent(_config(), measurement_budget=384)
        result = agent.run(laboratory, max_steps=2)
        assert result["research_state"]["stage"] == "reselect"
        assert result["selected_channels"] == ["sensor_a"]
        assert result["episode"]["validation_loss"] > .009
        return json.dumps({"result": result, "library": agent.library.to_dict()})


def test_old_measured_failure_prevents_reuse_despite_two_good_short_pilots(unresolved_representation):
    saved = json.loads(unresolved_representation)
    state = saved["result"]["research_state"]
    workspace = ResearchWorkspace.from_dict(state["workspace"])
    agent = MeasurementResearchAgent(_config(), measurement_budget=384,
        library=KnowledgeLibrary.from_dict(saved["library"]))
    with make_lab(1, 501038) as laboratory:
        laboratory.restore(state["apparatus_acquisitions"])
        spec = laboratory.spec_for(("sensor_a",))
        pilots = [laboratory.acquire(_pilot_probe(laboratory.spec, _config().seed, index, 8),
                                    ("sensor_a", "sensor_b")).observations
                  for index in (101, 201)]
        training, validation = [measured_view(rows, spec.feature_names) for rows in pilots]
        budget = SearchBudget(5000)
        _, reuse = agent._fit_view(spec, training, validation, budget,
            seed_programs=[workspace.incumbent.program],
            counterexamples=_trajectories(workspace.reference))
        trial, = reuse["trials"]
        assert max(trial["training_loss"], trial["validation_loss"]) < _config().minimum_improvement
        assert trial["counterexample_loss"] > .009
        assert not reuse["skipped_search"]
        assert reuse["compute_cost"] == len(training) + len(validation) + len(workspace.reference)
        assert budget.spent > reuse["compute_cost"]
        assert laboratory.costs["observation_cost"] == 88
        assert laboratory.costs["measurement_cost"] == 344


def test_same_instrument_representation_is_revised_and_confirmed_after_process_resume(
        unresolved_representation, tmp_path):
    saved = json.loads(unresolved_representation)
    previous = saved["result"]
    old_program = Program.from_dict(previous["episode"]["program"])
    path = tmp_path / "counterexample.json"
    path.write_text(unresolved_representation)
    saved = json.loads(path.read_text())
    library = KnowledgeLibrary.from_dict(saved["library"])
    with make_lab(1, 501038, backend="process") as laboratory:
        agent = MeasurementResearchAgent(_config(), measurement_budget=384, library=library)
        repaired = agent.run(laboratory, research_state=previous["research_state"], max_steps=1)
        event, = repaired["measurement_events"]
        assert event["action"] == "revise_measurement_representation"
        assert event["adopted"] and event["representation_changed"]
        assert event["counterexample_rows"] == 48 and event["training_rows"] == 24
        assert event["before_loss"] > .009 and event["validation_loss"] < 1e-9
        assert not event["reuse"]["skipped_search"]
        assert repaired["selected_channels"] == previous["selected_channels"]
        assert repaired["costs"]["observation_cost"] == previous["costs"]["observation_cost"]
        assert repaired["costs"]["measurement_cost"] == previous["costs"]["measurement_cost"]
        assert repaired["research_state"]["stage"] == "research"
        assert json.loads(json.dumps(repaired["research_state"]["workspace"]["reference"])) == (
            previous["research_state"]["workspace"]["reference"])
        path.write_text(json.dumps({"state": repaired["research_state"], "library": library.to_dict()}))
        uninterrupted = agent.run(laboratory, research_state=repaired["research_state"])
    saved = json.loads(path.read_text())
    with make_lab(1, 501038, backend="process") as laboratory:
        completed = MeasurementResearchAgent(_config(), measurement_budget=384,
            library=KnowledgeLibrary.from_dict(saved["library"])).run(laboratory, research_state=saved["state"])
        assert completed["status"] == "completed" and not completed["measurement_question"]
        assert completed["selected_channels"] == ["sensor_a"]
        assert completed["costs"] == uninterrupted["costs"]
        assert completed["episode"]["program"] == uninterrupted["episode"]["program"]
        assert completed["costs"]["observation_cost"] > repaired["costs"]["observation_cost"]
        rows = laboratory.evaluation_rows(completed["selected_channels"], seed=1501041,
                                          length=64, trajectories=4)
        assert loss(Program.from_dict(completed["episode"]["program"]), rows,
                    completed["episode"]["fallback"], .5) < 1e-9
        assert loss(old_program, rows, previous["episode"]["fallback"], .5) > .02
        for kind, limit in (("observation", 96), ("measurement", 384)):
            key = f"{kind}_cost"
            assert completed["costs"][key] == sum(query[key] for query in laboratory.acquisitions)
            assert completed["costs"][key] <= limit
        assert completed["costs"]["compute_cost"] <= 60000
