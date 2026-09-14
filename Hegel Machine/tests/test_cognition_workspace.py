"""Continue paid experiments and useful alternatives across research visits."""

from dataclasses import replace
import json

from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, TaskSpec
from hegel_machine.cognition.session import ResearchSession
from hegel_machine.cognition.workspace import ResearchWorkspace


class ExperimentalTask:
    def __init__(self, task_id="ongoing", *, response=lambda x: 2 * x + .25,
                 bounds=(-1., 1.)):
        low, high = bounds
        self.spec = TaskSpec(
            task_id, "Predict the measured response", ("x",),
            (Probe("initial", tuple({"x": low + (high - low) * fraction}
                                    for fraction in (0., .3, .65, 1.))),), (),
            feature_bounds={"x": bounds}, max_probe_length=8)
        self.response = response
        self.probes = []
        self.observed = 0

    def observe(self, probe):
        self.probes.append(probe)
        self.observed += probe.cost
        history, observations = [], []
        for point in probe.sequence:
            target = self.response(point["x"])
            observations.append(Observation(dict(point), target, tuple(history)))
            history.append({**point, "__target__": target})
        return observations


def config(**kwargs):
    return replace(ResearchConfig(observation_budget=32, compute_budget=12000,
                                  max_rounds=2, seed=61, use_memory=False), **kwargs)


def json_copy(value):
    return json.loads(json.dumps(value))


def test_json_resume_keeps_trajectory_history_and_incumbent_without_rebuying_initial_queries():
    task = ExperimentalTask()
    first = CognitiveAgent(config()).run(task)
    saved = json_copy(first.research_state)
    old_rows = ResearchWorkspace.from_dict(saved).observations
    old_queries = len(task.probes)
    old_cost = task.observed
    resumed = CognitiveAgent(config(observation_budget=8)).run(task, research_state=saved)
    workspace = ResearchWorkspace.from_dict(json_copy(resumed.research_state))

    assert workspace.observations[:len(old_rows)] == old_rows
    # Experiment resets remain resets, and each saved historical target survives.
    assert sum(not row.history for row in old_rows) >= 2
    assert any(row.history and "__target__" in row.history[-1] for row in old_rows)
    assert all(probe.probe_id != "initial" for probe in task.probes[old_queries:])
    assert task.observed - old_cost == resumed.observation_cost <= 8
    assert len(workspace.observations) == len(old_rows) + resumed.observation_cost
    assert resumed.initial_loss == first.initial_loss
    assert loss(resumed.program, old_rows, resumed.fallback, .5) < 1e-12
    assert not any(action.action == "revise" and action.details["before"] != first.program.describe()
                   for action in resumed.actions)


def test_saved_competing_program_changes_the_next_visit_prediction():
    task = ExperimentalTask(response=lambda x: x * x)
    x = Program("feature", name="x")
    square = Program("mul", children=(x, x))
    rows = [Observation({"x": value}, value) for value in (0., 1., 0., 1.)]
    workspace = ResearchWorkspace(
        task.spec.task_id, observations=rows,
        incumbent=Candidate("old_linear", x, "refit"),
        bank=[Candidate("retained_square", square, "composition")],
        fallback=.5, initial_loss=.25)
    saved = json_copy(workspace.to_dict())
    with_alternative = CognitiveAgent(config(strategy="fixed", max_rounds=1)).run(task, research_state=saved)

    saved["bank"] = []
    without_alternative = CognitiveAgent(config(strategy="fixed", max_rounds=1)).run(
        ExperimentalTask(response=lambda x: x * x), research_state=saved)
    evaluation = [Observation({"x": value}, value * value) for value in (-.8, -.3, .2, .7)]
    assert loss(with_alternative.program, evaluation, .5, .5) < 1e-12
    assert loss(without_alternative.program, evaluation, .5, .5) > .05
    assert any(action.action == "revise" and action.details["before"] == x.describe()
               for action in with_alternative.actions)
    assert all(probe.probe_id != "initial" for probe in task.probes)


def test_more_research_on_saved_evidence_improves_the_existing_explanation():
    task = ExperimentalTask(response=lambda x: x * x)
    first = CognitiveAgent(config(strategy="fixed", max_rounds=1)).run(task)
    old_rows = len(first.observations)
    resumed = CognitiveAgent(config(observation_budget=8)).run(
        task, research_state=json_copy(first.research_state))
    evaluation = [Observation({"x": value}, value * value) for value in (-.9, -.4, .1, .6)]

    assert loss(first.program, evaluation, first.fallback, .5) > .03
    assert loss(resumed.program, evaluation, resumed.fallback, .5) < 1e-8
    assert len(resumed.observations) == old_rows + resumed.observation_cost
    assert resumed.observation_cost <= 8
    assert task.observed == first.observation_cost + resumed.observation_cost


def test_completed_session_accepts_a_new_task_without_restarting_finished_work():
    first_task, new_task = ExperimentalTask("first"), ExperimentalTask("arriving")
    session = ResearchSession(config(), [first_task], observation_budget=120,
                              compute_budget=40000, max_visits=2)
    finished = session.run()
    assert finished["status"] == "completed" and finished["completed_visits"] == 1
    old_cost = first_task.observed
    old_episode = json_copy(finished["episodes"][0])
    index = session.add_environment(new_task)
    completed = session.run()

    assert index == 1
    assert completed["visited_tasks"] == 2 and completed["completed_visits"] == 2
    assert json_copy(completed["episodes"][0]) == old_episode
    assert first_task.observed == old_cost
    assert new_task.observed > 0
    assert completed["total_observation_cost"] == first_task.observed + new_task.observed
    assert set(completed["workspaces"]) == {"0", "1"}


def test_external_counterexamples_reopen_a_solved_question_and_change_its_model():
    # Initially only the negative operating range is controllable and the
    # response is constant. External measurements expose an activating regime.
    task = ExperimentalTask(response=lambda x: float(x > 0), bounds=(-1., 0.))
    options = dict(observation_budget=128, compute_budget=40000, max_visits=2)
    session = ResearchSession(config(), [task], **options)
    first = session.run()
    assert first["status"] == "completed" and first["adequate_tasks"] == 1
    previous = ResearchWorkspace.from_dict(first["workspaces"]["0"])
    probe = Probe("external_positive_regime", tuple({"x": value} for value in (.2, .4, .7, 1.)))
    fresh = task.observe(probe)
    assert loss(previous.incumbent.program, fresh, previous.fallback, .5) > .2
    session.add_evidence(0, probe, fresh)
    checkpoint = json_copy(session.to_dict())
    restored = ResearchSession(config(), [task], state=checkpoint, **options)
    result = restored.run()
    final = ResearchWorkspace.from_dict(result["workspaces"]["0"])

    assert result["completed_visits"] == 2
    assert loss(final.incumbent.program, fresh, final.fallback, .5) < 1e-8
    assert loss(final.incumbent.program, previous.observations, final.fallback, .5) < 1e-8
    assert result["total_observation_cost"] == task.observed
    assert result["total_observation_cost"] == sum(row["observation_cost"] for row in result["episodes"]) + probe.cost
    assert sum(probe.probe_id == "initial" for probe in task.probes) == 1
    assert result["episodes"][-1]["scheduler"]["reason"] == "new_evidence"
