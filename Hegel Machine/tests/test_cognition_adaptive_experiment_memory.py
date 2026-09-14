"""Learn experimental procedures whose questions follow the target evidence."""

import json
from pathlib import Path

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.experiment_knowledge import ExperimentKnowledge
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, SearchBudget, TaskSpec
from hegel_machine.cognition.probing import transition_probe


def _rule(coordinate):
    def threshold(value):
        return Program("ge", children=(coordinate, Program("const", value=value)))
    return Program("if", children=(Program("available", lag=1),
        Program("if", children=(Program("ge", children=(Program("lag", name="__target__", lag=1),
            Program("const", value=.5))), threshold(-.4), threshold(.4))), threshold(.4)))


def _observe(program, points):
    history, rows = [], []
    for point in points:
        value = program.predict(Observation(point, 0., tuple(history)))
        rows.append(Observation(point, value, tuple(history)))
        history.append({**point, "__target__": value})
    return rows


def _learn_boundary_procedure():
    x, z = Program("feature", name="x"), Program("feature", name="z")
    truth = _rule(Program("add", children=(x, Program("mul", children=(z, Program("const", value=.5))))))
    axis = _rule(x)
    candidates = [Candidate("axis", axis, "state"), Candidate("joint", truth, "state")]
    spec = TaskSpec("source", "response", ("x", "z"), (), (),
                    feature_bounds={"x": (-1., 1.), "z": (-1., 1.)}, max_probe_length=24)
    rows = _observe(truth, [{"x": x, "z": 0.} for x in (-1., 0., 1., 0., -1., 1., -1.)])
    probe, design = transition_probe(spec, candidates, rows, budget=SearchBudget(500),
                                     remaining=16, round_number=0)
    book = ExperimentKnowledge()
    learned = book.record(spec, candidates, rows, probe, _observe(truth, probe.sequence),
                          fallback=0., budget=SearchBudget(500), design=design)
    assert learned is not None and learned["procedure"]["kind"] == "conditional_boundary"
    return ExperimentKnowledge(json.loads(json.dumps(book.to_dict())))


def _paid_target():
    saved = json.loads((Path(__file__).parent / "fixtures" / "cognition_f10_boundary_learning.json").read_text())
    rows = [Observation(row["features"], row["target"], tuple(row["history"]))
            for row in saved["observations"]]
    candidate = Candidate("retained", Program.from_dict(saved["program"]), "memory")
    spec = TaskSpec("target", "response", tuple(saved["feature_bounds"]), (), (),
        feature_bounds=saved["feature_bounds"], max_probe_length=24)
    used = [Probe(probe["probe_id"], tuple(probe["sequence"])) for probe in saved["queried"]]
    return spec, candidate, rows, used


def test_real_paid_zero_error_workspace_still_has_unanswered_conditional_boundaries():
    spec, candidate, rows, used = _paid_target()
    assert len(rows) == 86 and loss(candidate.program, rows, 0., .5) == 0.
    probe, details = transition_probe(spec, [candidate], rows, budget=SearchBudget(500),
        remaining=16, round_number=5, used=used)
    assert probe is not None and details["candidate_disagreement"] == 0.
    support = max(details["boundary_support"], key=lambda item: item["relative_width"])
    assert support["previous_state"] is True and support["history_depth"] == 1
    assert support["observed_interval"][0] < -.98 < -.48 < support["observed_interval"][1]
    assert details["unresolved_boundary_width"] > .1
    assert support["free_controls"] == ["z"]
    assert all(probe.sequence != previous.sequence for previous in used)


def test_recalled_procedure_rebinds_to_current_intervals_with_one_surviving_model():
    book = _learn_boundary_procedure()
    spec, candidate, rows, used = _paid_target()
    budget = SearchBudget(500)
    offers = book.propose(spec, [candidate], rows, fallback=0., budget=budget, remaining=16,
                          used=used, evidence_losses={candidate.program: 0.})
    assert offers and 0 < budget.spent <= 500
    recalled = book.last_recall[0]
    assert recalled["binding"] == "current_observed_response_intervals"
    assert recalled["supported_candidates"] == 1
    assert recalled["design"]["unresolved_boundary_width"] > .1
    # Source controls were [-1, 1]. The transferred question uses the new
    # response bracket, rather than replaying the old normalized test values.
    source = book.records[0]["normalized_sequence"]
    literal = tuple({name: low + point[index] * (high-low)
                     for index, (name, (low, high)) in enumerate(spec.feature_bounds.items())}
                    for point in source)
    assert offers[0].sequence != literal


def test_recall_reuses_already_priced_target_boundary_design():
    book = _learn_boundary_procedure()
    spec, candidate, rows, used = _paid_target()
    proposal = transition_probe(spec, [candidate], rows, budget=SearchBudget(500),
        remaining=16, round_number=5, used=used)
    budget = SearchBudget(0)
    offers = book.propose(spec, [candidate], rows, fallback=0., budget=budget, remaining=16,
        used=used, evidence_losses={candidate.program: 0.}, boundary_proposal=proposal)
    assert offers and offers[0].sequence == proposal[0].sequence
    assert budget.spent == 0 and book.last_recall[0]["compute_cost"] == 0


def test_refuted_target_model_does_not_make_old_literal_experiment_relevant():
    x = Program("feature", name="x")
    square = Program("mul", children=(x, x))
    candidates = [Candidate("linear", x, "refit"), Candidate("square", square, "refit")]
    spec = TaskSpec("source", "response", ("x",), (), (), feature_bounds={"x": (0., 2.)})
    old = [Observation({"x": v}, v) for v in (0., 1., 0., 1.)]
    probe = Probe("separate", ({"x": 2.}, {"x": 1.5}))
    acquired = [Observation(point, point["x"] ** 2) for point in probe.sequence]
    book = ExperimentKnowledge()
    assert book.record(spec, candidates, old, probe, acquired, fallback=0., budget=SearchBudget(100))
    # Keep the same public response context, but target evidence has already
    # distinguished the models. Candidate disagreement alone used to recall it.
    rows = [Observation({"x": v}, v) for v in (.25, .75, .25, .75)]
    budget = SearchBudget(100)
    assert book.propose(spec, candidates, rows, fallback=0., budget=budget, remaining=8) == []
    assert budget.spent == len(candidates) * len(rows)
