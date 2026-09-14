"""Small paid experiments distinguish startup and actual prepared histories."""

import json

from hegel_machine.cognition import probing
from hegel_machine.cognition.experiment_knowledge import ExperimentKnowledge
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec


def _spec(names=("x",), extent=1.):
    return TaskSpec("public", "Predict the response", names, (), (),
                    feature_bounds={name: (-extent, extent) for name in names}, max_probe_length=12)


def _rule(coordinate, *, lower=-.4, upper=.4, startup=None, depth=1):
    def threshold(value):
        return Program("ge", children=(coordinate, Program("const", value=value)))

    return Program("if", children=(Program("available", lag=depth),
        Program("if", children=(Program("ge", children=(Program("lag", name="__target__", lag=depth),
                                                        Program("const", value=.5))),
                                threshold(lower), threshold(upper))),
        threshold(upper) if startup is None else startup))


def _observe(program, points):
    history, rows = [], []
    for point in points:
        target = program.predict(Observation(point, 0., tuple(history)))
        rows.append(Observation(point, target, tuple(history)))
        history.append({**point, "__target__": target})
    return rows


def _candidates(*programs):
    return [Candidate(str(index), program, "composition") for index, program in enumerate(programs)]


def test_one_paid_startup_row_separates_programs_without_observed_opposite_labels(monkeypatch):
    x = Program("feature", name="x")
    early = _rule(x)
    late = _rule(x, startup=Program("ge", children=(x, Program("const", value=.8))))
    rows = _observe(early, [{"x": -1.}, {"x": 0.}])
    assert {row.target for row in rows} == {0.}
    calls = []
    evaluate = probing.predict

    def measured(program, row, fallback):
        calls.append(1)
        return evaluate(program, row, fallback)

    monkeypatch.setattr(probing, "predict", measured)
    budget = SearchBudget(80)
    probe, details = probing.transition_probe(_spec(), _candidates(early, late), rows,
        budget=budget, remaining=1, round_number=0)
    assert probe.cost == 1
    assert details["startup_control"] == probe.sequence[0]
    fresh = _observe(early, probe.sequence)
    assert early.predict(fresh[0]) != late.predict(fresh[0])
    assert budget.spent == sum(calls) > 0


def test_prepared_experiment_keeps_the_full_required_history_and_pays_each_row():
    x = Program("feature", name="x")
    startup = Program("const", value=0.)
    early = _rule(x, upper=.3, startup=startup, depth=2)
    late = _rule(x, upper=.7, startup=startup, depth=2)
    rows = _observe(early, [{"x": -1.}] * 3)
    candidates = _candidates(early, late)
    too_short, _ = probing.transition_probe(_spec(), candidates, rows,
        budget=SearchBudget(80), remaining=2, round_number=0)
    assert too_short is None
    probe, details = probing.transition_probe(_spec(), candidates, rows,
        budget=SearchBudget(80), remaining=3, round_number=0)
    assert probe.cost == 3 and details["history_depth"] == 2
    assert details["startup_control"] is None
    left, right = [_observe(program, probe.sequence) for program in (early, late)]
    assert [row.target for row in left[:2]] == [row.target for row in right[:2]]
    assert left[-1].target != right[-1].target
    assert len(left[-1].history) == 2


def test_an_already_paid_startup_prefix_is_not_queried_again():
    x = Program("feature", name="x")
    early, late = _rule(x, upper=.4), _rule(x, upper=.8)
    rows = _observe(early, [{"x": -1.}, {"x": 1.}])
    candidates = _candidates(early, late)
    first, _ = probing.transition_probe(_spec(), candidates, rows, budget=SearchBudget(100),
                                        remaining=1, round_number=0)
    again, _ = probing.transition_probe(_spec(), candidates, [*rows, *_observe(early, first.sequence)],
        budget=SearchBudget(100), remaining=1, round_number=1, used=(first,))
    assert again is None  # The paid outcome has now resolved this disagreement.


def test_cached_paid_scores_keep_refuted_candidates_out_of_startup_selection():
    x = Program("feature", name="x")
    candidate = _rule(x)
    refuted = Program("const", value=.25)
    rows = _observe(candidate, [{"x": value} for value in (-1., 0., 1., 0., -1.)])
    # No extra allowance is needed to reread a score already paid by selection.
    budget = SearchBudget(40)
    probe, details = probing.transition_probe(_spec(), _candidates(candidate, refuted), rows,
        budget=budget, remaining=8, round_number=0, evidence_losses={candidate: 0., refuted: .2})
    assert probe is not None and probe.cost > 1
    assert details.get("design") != "minimal_program_disagreement"
    assert details["relative_width"] > 0.


def test_wide_mature_intervals_keep_a_paired_experiment_ahead_of_cheap_startup():
    x = Program("feature", name="x")
    start = lambda value: Program("ge", children=(x, Program("const", value=value)))
    early, late = _rule(x, startup=start(.1)), _rule(x, startup=start(.2))
    points = [{"x": value} for value in (.05, -.5, .5, 0., -1.)]
    rows = [*_observe(early, points), *_observe(early, [{"x": .25}])]
    candidates = _candidates(early, late)
    full, details = probing.transition_probe(_spec(), candidates, rows, budget=SearchBudget(100),
        remaining=12, round_number=0, evidence_losses={early: 0., late: 0.})
    assert full.cost >= 4
    assert details["mature_relative_width"] > details["startup_relative_width"]
    fresh = _observe(early, full.sequence)
    assert any(a.features == b.features and a.target != b.target for a in fresh for b in fresh)
    four, _ = probing.transition_probe(_spec(), candidates, rows, budget=SearchBudget(100),
        remaining=4, round_number=0, evidence_losses={early: 0., late: 0.})
    assert four.cost == 4
    four_rows = _observe(early, four.sequence)
    assert any(a.features == b.features and a.target != b.target for a in four_rows for b in four_rows)
    # With the paired design unaffordable, the same unresolved startup remains
    # an executable one-row experiment rather than consuming a nonexistent budget.
    short, _ = probing.transition_probe(_spec(), candidates, rows, budget=SearchBudget(100),
        remaining=1, round_number=0, evidence_losses={early: 0., late: 0.})
    assert short.cost == 1
    observation = Observation(short.sequence[0], 0.)
    assert early.predict(observation) != late.predict(observation)


def test_paid_short_experiment_is_recalled_after_role_and_control_range_changes():
    x, z = Program("feature", name="x"), Program("feature", name="z")
    joint = Program("add", children=(x, Program("mul", children=(Program("const", value=.5), z))))
    axis, truth = _rule(x), _rule(joint)
    rows = _observe(truth, [{"x": value, "z": 0.} for value in (-1., 0., 1., 0., -1.)])
    candidates = _candidates(axis, truth)
    spec = _spec(("x", "z"))
    probe, _ = probing.transition_probe(spec, candidates, rows, budget=SearchBudget(500),
                                         remaining=2, round_number=0)
    assert probe is not None and probe.cost <= 2
    acquired = _observe(truth, probe.sequence)
    library = ExperimentKnowledge()
    learned = library.record(spec, candidates, rows, probe, acquired, fallback=0., budget=SearchBudget(100))
    assert learned is not None and learned["observations"] <= 2
    restored = ExperimentKnowledge(json.loads(json.dumps(library.to_dict())))
    u = Program("mul", children=(Program("const", value=.5), Program("feature", name="u")))
    v = Program("mul", children=(Program("const", value=.5), Program("feature", name="v")))
    target_joint = Program("add", children=(u, Program("mul", children=(Program("const", value=.5), v))))
    target_axis, target_truth = _rule(u), _rule(target_joint)
    target_rows = _observe(target_truth, [{"u": value, "v": 0.} for value in (-2., 0., 2., 0., -2.)])
    offers = restored.propose(_spec(("u", "v"), 2.), _candidates(target_axis, target_truth),
        target_rows, fallback=0., budget=SearchBudget(100), remaining=2)
    assert offers and restored.last_recall
    assert all(-2. <= value <= 2. for point in offers[0].sequence for value in point.values())
    fresh = _observe(target_truth, offers[0].sequence)
    assert any(target_axis.predict(row) != row.target for row in fresh)
