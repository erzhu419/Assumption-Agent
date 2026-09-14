"""Unresolved questions survive a visit and require actual preparation evidence."""

import json
from pathlib import Path

from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.experiment_knowledge import ExperimentKnowledge
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.probing import transition_outcome, transition_probe
from hegel_machine.cognition.research_frontier import plan_experiment_frontier
from hegel_machine.cognition.session import ResearchSession
from hegel_machine.cognition.workspace import ResearchWorkspace, observation_from_dict


class RecordedApparatus:
    def __init__(self, environment):
        self.environment, self.spec, self.queries = environment, environment.spec, []

    def observe(self, probe):
        self.queries.append(probe)
        return self.environment.observe(probe)


def _saved_question():
    payload = json.loads((Path(__file__).parent / "fixtures" /
                          "cognition_v018_final_s01_agenda.json").read_text())
    environment = RecordedApparatus(make_suite(payload["seed"], "stateful")[11])
    rows = [observation_from_dict(row) for row in payload["observations"]]
    queried = [Probe(item["probe_id"], tuple(item["sequence"])) for item in payload["used"]]
    program = Program.from_dict(payload["program"])
    assert all(program.predict(row) == row.target for row in rows)
    incumbent = Candidate("retained", program, "state")
    config = ResearchConfig(observation_budget=32, compute_budget=18000, max_rounds=2,
                            use_memory=False, seed=19)
    work = SearchBudget(800)
    frontier = plan_experiment_frontier(environment.spec, [incumbent], rows, queried,
                                        config, work, turn=4)
    workspace = ResearchWorkspace(environment.spec.task_id, observations=rows, reference=rows[12:],
        queried=queried, incumbent=incumbent, bank=[incumbent], fallback=0.,
        initial_loss=sum(row.target ** 2 for row in rows) / len(rows), rounds_completed=4,
        experiment_frontier=frontier)
    session = ResearchSession(config, [environment], observation_budget=len(rows) + 32,
                               compute_budget=20000, max_visits=2)
    # Resume the genuinely observed learning state; no independent responses
    # or invented error are needed to make the remaining experiment eligible.
    session.state["workspaces"]["0"] = workspace.to_dict()
    session.state["total_observation_cost"] = len(rows)
    session.state["total_compute_cost"] = work.spent
    session.state["tasks"][0].update(visits=1, status="unresolved_experiment", unresolved=True,
        validation_loss=0., initial_loss=workspace.initial_loss, normalized_excess_loss=0.)
    return session, environment, workspace


def test_zero_error_question_restores_then_buys_saved_probe_before_fitting(tmp_path):
    session, environment, workspace = _saved_question()
    saved = session.to_dict()
    question = json.loads(json.dumps(workspace.experiment_frontier))
    assert saved["pending"] == [0]
    assert saved["tasks"][0]["revisit_basis"] == "unresolved_distinguishing_experiment"
    assert saved["tasks"][0]["normalized_excess_loss"] == 0.
    path = tmp_path / "research.json"
    path.write_text(json.dumps(saved))
    restored = ResearchSession(session.config, [environment], observation_budget=100,
        compute_budget=20000, max_visits=2, state=json.loads(path.read_text()))
    assert restored.state["workspaces"]["0"]["experiment_frontier"] == question
    result = restored.run(max_steps=1)
    episode = result["episodes"][0]
    assert episode["scheduler"]["reason"] == "unresolved_distinguishing_experiment"
    assert environment.queries[0].probe_id == question["probe"]["probe_id"]
    assert list(environment.queries[0].sequence) == question["probe"]["sequence"]
    assert not {probe.probe_id for probe in environment.queries}.intersection(
        probe.probe_id for probe in environment.spec.initial_probes)
    actions = [action["action"] for action in episode["actions"]]
    assert actions.index("execute_research_question") < actions.index("compare") < actions.index("revise")
    coverage = next(action["details"] for action in episode["actions"]
                    if action["action"] == "measure_experiment_coverage")
    assert coverage["reached_boundaries"] == 3
    assert {target["previous_state"] for target in coverage["targets"]
            if target["refines_interval"]} == {None, False, True}
    added = [observation_from_dict(row) for row in
             result["workspaces"]["0"]["observations"][len(workspace.observations):]]
    replacement = Program.from_dict(episode["program"])
    assert sum(workspace.incumbent.program.predict(row) != row.target for row in added) > 0
    assert all(replacement.predict(row) == row.target for row in added)
    assert episode["observation_cost"] == sum(probe.cost for probe in environment.queries)
    assert episode["observation_cost"] <= session.config.observation_budget
    assert episode["compute_cost"] <= session.config.compute_budget
    # Successful fitting still leaves a concrete smaller question, created by
    # the controller itself and reflected in the scheduler's completed visit.
    final = result["workspaces"]["0"]["experiment_frontier"]
    assert final and final["evidence_rows"] > question["evidence_rows"]
    assert final["uncertainty"] < question["uncertainty"]
    assert result["tasks"][0]["validation_loss"] == 0.
    assert result["tasks"][0]["status"] == "unresolved_experiment"


def test_saved_question_does_not_spend_an_unfunded_followup():
    session, environment, workspace = _saved_question()
    question = json.loads(json.dumps(workspace.experiment_frontier))
    needed = question["observation_cost"] + question["comparison_observation_cost"]
    restored = ResearchSession(session.config, [environment],
        observation_budget=len(workspace.observations) + needed - 1,
        compute_budget=20000, max_visits=2, state=json.loads(json.dumps(session.to_dict())))
    result = restored.run(max_steps=1)
    assert not environment.queries and not result["episodes"]
    assert result["total_observation_cost"] == len(workspace.observations)
    assert result["workspaces"]["0"]["experiment_frontier"] == question


def _observe(program, points):
    rows, history = [], []
    for point in points:
        response = program.predict(Observation(point, 0., tuple(history)))
        rows.append(Observation(point, response, tuple(history)))
        history.append({**point, "__target__": response})
    return rows


def test_failed_preparation_keeps_real_disagreement_but_not_a_successful_procedure():
    x = Program("feature", name="x")

    def cut(value):
        return Program("ge", children=(x, Program("const", value=value)))

    def rule(startup):
        return Program("if", children=(Program("available", lag=2),
            Program("if", children=(Program("ge", children=(
                Program("lag", name="__target__", lag=2), Program("const", value=.5))),
                cut(-.4), cut(.6))), startup))

    delayed, immediate = rule(Program("const", value=0.)), rule(cut(.8))
    rows = _observe(delayed, [{"x": value} for value in (-1., -1., .5, .7, .5, .7, 0., 0., -1., -1.)])
    assert all(delayed.predict(row) == immediate.predict(row) == row.target for row in rows)
    spec = TaskSpec("two_startup_rows", "Predict the controlled response", ("x",), (), (),
                    feature_bounds={"x": (-1., 1.)}, max_probe_length=6)
    candidates = [Candidate("delayed", delayed, "state"), Candidate("immediate", immediate, "state")]
    probe, design = transition_probe(spec, candidates, rows, budget=SearchBudget(500),
        remaining=6, round_number=0, remaining_rounds=1)
    assert probe.cost == 6 and design["boundary_support"][0]["previous_state"] is True
    succeeded = _observe(immediate, probe.sequence)
    failed = _observe(delayed, probe.sequence)
    assert transition_outcome(design, succeeded)["reached_boundaries"] == 1
    outcome = transition_outcome(design, failed)
    assert outcome["reached_boundaries"] == 0
    assert all(target["prepared_state"] is False for target in outcome["targets"])
    assert probe.sequence[0] == probe.sequence[1] == {"x": 1.}
    assert failed[0].target == failed[1].target == 0.

    library = ExperimentKnowledge()
    positive = library.record(spec, candidates, rows, probe, succeeded, fallback=0.,
                              budget=SearchBudget(100), design=design)
    assert positive["procedure"]["source_support"]
    assert all(item["condition_reached"] for item in positive["procedure"]["source_support"])
    # The same public setup now has two startup responses before its mature
    # transition is available. The paid trajectory still splits explanations,
    # but the attempted mature preparation did not reach the desired state.
    negative = library.record(spec, candidates, rows, probe, failed, fallback=0.,
                              budget=SearchBudget(100), design=design)
    assert negative["observed_loss_gap"] > 0.
    assert "procedure" not in negative
    assert "procedure" not in library.records[0]
    assert negative["experiment_id"] == positive["experiment_id"]
