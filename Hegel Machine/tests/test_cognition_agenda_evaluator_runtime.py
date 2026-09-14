"""Pay evaluator work only at real agenda decisions and feedback updates."""

from copy import deepcopy
from dataclasses import replace
import json

from hegel_machine.cognition.agenda import ResearchAgenda
from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Probe, ResearchConfig
from hegel_machine.cognition.workspace import ResearchWorkspace


CONFIG = ResearchConfig(seed=71, observation_budget=32, compute_budget=6000, max_rounds=2)
SOURCE = {"kind": "suite", "seed": 71, "partition": "development", "task_index": 0}


def research_agenda(directory, *, compute_budget=20000):
    environment = make_suite(71)[0]
    observations = [row for probe in environment.spec.initial_probes for row in environment.observe(probe)]
    candidate = Candidate("initial_explanation", Program("const", value=0.), "refit")
    workspace = ResearchWorkspace(environment.spec.task_id, observations=observations,
        reference=list(observations), incumbent=candidate, bank=[candidate], fallback=0.,
        initial_loss=loss(candidate.program, observations, 0., .5))
    agenda = ResearchAgenda(directory=directory, observation_budget=200, compute_budget=compute_budget)
    agenda.add_task("repair", source=SOURCE, config=CONFIG, max_visits=1, research_state=workspace.to_dict())
    return agenda


def test_paid_forecast_is_frozen_before_actual_continuation_and_survives_resume(tmp_path):
    agenda = research_agenda(tmp_path)
    report = agenda.run(max_steps=1)
    visit = report["visits"][0]
    forecast = visit["evaluator_forecast"]
    assert forecast["forecast"] is not None and forecast["expected_gain"] is None
    assert visit["feedback"]["reference_rows"] > 0 and visit["feedback"]["normalized_gain"] > 0
    assert len(report["knowledge"]["research_opportunities"]["evaluator"]["records"]) == 1
    assert report["evaluator_compute_cost"] > 0
    assert report["spent"]["compute_cost"] == visit["costs"]["compute_cost"] + report["evaluator_compute_cost"]
    assert report["questions"]["repair"]["costs"]["compute_cost"] == visit["costs"]["compute_cost"]
    restored = ResearchAgenda(directory=tmp_path, state=json.loads(json.dumps(report)))
    before = deepcopy(restored.to_dict())
    for _ in range(3):
        restored.offers()
        restored.to_dict()
    paused = restored.run(max_steps=0)
    assert paused["spent"] == before["spent"]
    assert paused["evaluator_compute_cost"] == before["evaluator_compute_cost"]
    assert paused["knowledge"]["research_opportunities"]["evaluator"] == before["knowledge"]["research_opportunities"]["evaluator"]


def test_forecast_cannot_spend_the_last_complete_visit_allocation(tmp_path):
    agenda = research_agenda(tmp_path, compute_budget=CONFIG.compute_budget)
    report = agenda.run(max_steps=1)
    assert len(report["visits"]) == 1
    visit = report["visits"][0]
    assert visit["quoted_costs"]["compute_cost"] == CONFIG.compute_budget
    assert report["evaluator_compute_cost"] == 0
    assert visit["feedback"]["reference_rows"] > 0 and visit["feedback"]["normalized_gain"] > 0
    assert report["spent"]["compute_cost"] <= CONFIG.compute_budget
    assert report["knowledge"]["research_opportunities"]["outcomes"][-1]["gain"] > 0


def test_blank_initial_task_receives_no_invented_evaluator_forecast(tmp_path):
    agenda = ResearchAgenda(directory=tmp_path)
    agenda.add_task("new", source=SOURCE, config=CONFIG)
    report = agenda.run(max_steps=1)
    assert report["visits"][0]["evaluator_forecast"] is None
    assert report["evaluator_compute_cost"] == 0
    assert "evaluator" not in report["knowledge"]["research_opportunities"]


def test_forecast_with_no_selected_visit_remains_an_independent_paid_cost(tmp_path):
    environment = make_suite(71)[0]
    solved = CognitiveAgent(CONFIG).run(environment)
    agenda = ResearchAgenda(directory=tmp_path, observation_budget=200, compute_budget=30000)
    agenda.add_task("unchanged", source=SOURCE, config=CONFIG, max_visits=1, research_state=solved.research_state)
    probe = Probe("further_observation", tuple({"x": x, "z": -x} for x in (-1.2, -.4, .4, 1.2)))
    # The supplied workspace initially has no agenda-local validation report.
    # Obtain an actual report, then a return under that now-observed condition.
    for index in range(2):
        agenda.add_evidence("unchanged", replace(probe, probe_id=f"delivery_{index}"), environment.observe(probe))
        first = agenda.run(max_steps=1)
        assert first["visits"][-1]["feedback"]["normalized_gain"] <= 0.
    agenda.add_evidence("unchanged", replace(probe, probe_id="another_delivery"), environment.observe(probe))
    before = deepcopy(agenda.to_dict())
    second = agenda.run(max_steps=1)
    assert len(second["visits"]) == len(before["visits"])
    assert second["evaluator_compute_cost"] > before["evaluator_compute_cost"]
    assert second["spent"]["compute_cost"] == sum(visit["costs"]["compute_cost"] for visit in second["visits"]) + second["evaluator_compute_cost"]
    assert second["spent"]["observation_cost"] == 3 * probe.cost + sum(visit["costs"]["observation_cost"] for visit in second["visits"])
