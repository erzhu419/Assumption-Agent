"""Paid evidence identifies time structure or changes the next legal experiment."""

from math import sin

from hegel_machine.cognition.evaluation import initial_prediction, loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.observed_condition_duration import propose_observed_condition_duration
from hegel_machine.cognition.observed_condition_experiments import plan_observed_condition_probe


# A retained failed experiment had only three readings above its condition.
# This sequence is measured support, not a controllable instrument input.
READINGS = (-.10026491224375728, -1.0353658186560832, .4544948722968792,
    .8067201229351362, -.9210025259288659, -.8036694518945644,
    -1.019831959865754, -1.017016808801168, -.7335144048918321,
    -.950220304446507, -.1318540063099118, -.2478679352551107,
    -.3074938835771232, -.569309024842584, -.8850955370715023,
    -.5409293650395595, -.16934490697779342, .1027559392182858,
    -.16134272619132511, -.22097878052317216, .24812874028692816,
    -.8647944133860819, -.015386101521532136, -.4014043659364192)


def feature(name):
    return Program("feature", name=name)


def spec():
    return TaskSpec("unlabelled_response", "Measure while applying legal commands.",
        ("drive", "bias", "meter"), (), (),
        feature_bounds={"drive": (-1., 1.), "bias": (-1., 1.)}, max_probe_length=32)


def response_program():
    first = Program("add", children=(Program("mul", children=(Program("const", value=2.),
        feature("drive"))), Program("mul", children=(Program("const", value=.25), feature("meter")))))
    second = Program("add", children=(Program("mul", children=(Program("const", value=-2.),
        feature("drive"))), Program("mul", children=(Program("const", value=.25), feature("meter")))))
    return Program("if", children=(Program("ge", children=(feature("meter"),
        Program("const", value=.16))), Program("window_sum", lag=5, children=(first,)),
        Program("window_sum", lag=5, children=(second,))))


def measured_rows(point, readings=READINGS, *, unexplained=False):
    rows, history = [], []
    program = response_program()
    for reading in readings:
        row = Observation({**point, "meter": reading}, 0., tuple(history))
        target = program.predict(row) + (sin(5. * reading) if unexplained else 0.)
        rows.append(Observation(row.features, target, row.history))
        history.append({**row.features, "__target__": target})
    return rows


def selected_response():
    return {"candidate": Candidate("current_response", response_program(), "research"),
        "research_program": {"selection_steps": [{
            "operation": "discover_conditional_coordinates",
            "gate_coordinate": feature("meter").to_dict(),
            "threshold_interval": [.1027559392182858, .24812874028692816]}]}}


def test_full_measured_boundary_support_recovers_a_rare_current_condition():
    rows = measured_rows({"drive": 1., "bias": 0.})
    budget = SearchBudget(40000)
    claims = propose_observed_condition_duration(spec(), rows, budget, tolerance=1e-7)
    best = claims[0]
    evidence = best["observed_condition_evidence"]
    assert evidence["status"] == "conditional_duration"
    assert evidence["width_candidates"] == [5]
    assert evidence["placement_candidates"] == ["outside_window"]
    assert evidence["threshold_interval"] == [.1027559392182858, .24812874028692816]
    assert best["training_loss"] < 1e-12
    assert loss(best["candidate"].program, rows, initial_prediction(rows, .5), .5) < 1e-12
    assert 0 < budget.spent <= 40000


def test_poor_original_response_fit_remains_unresolved_evidence():
    rows = measured_rows({"drive": 1., "bias": 0.}, unexplained=True)
    budget = SearchBudget(40000)
    claims = propose_observed_condition_duration(spec(), rows, budget, tolerance=1e-7)
    assert claims
    assert all(claim["training_loss"] > 1e-7 for claim in claims)
    assert all(claim["observed_condition_evidence"]["status"] == "unresolved_duration"
               for claim in claims)
    assert all(not claim["observed_condition_evidence"]["response_supported"] for claim in claims)
    assert 0 < budget.spent <= 40000


def test_weak_condition_background_is_changed_and_remeasured():
    weak = {"drive": 0., "bias": 0.}
    weak_rows = measured_rows(weak)
    first_budget = SearchBudget(40000)
    first = propose_observed_condition_duration(spec(), weak_rows, first_budget)
    assert first[0]["training_loss"] < 1e-12
    assert first[0]["observed_condition_evidence"]["status"] == "smooth_duration"
    assert not any(item["observed_condition_evidence"]["status"] == "conditional_duration"
                   for item in first)
    # These calibration observations were paid on legal control backgrounds;
    # only their empirical reading residuals enter experiment forecasts.
    pilot = [Observation({"drive": 0., "bias": bias, "meter": .2 * bias + noise}, 0.)
             for bias in (-1., 0., 1.) for noise in (-.8, -.3, .3, .8)]
    # Calibration buys complete response rows too. Even at zero drive the
    # response retains its measured-meter term; zero targets would refute it.
    pilot = [Observation(row.features, response_program().predict(row), row.history) for row in pilot]
    rows = [*pilot, *weak_rows]
    planning = SearchBudget(15000)
    plan = plan_observed_condition_probe(spec(), rows, selected_response(), planning,
        length=24, previous_experiments=({"condition_field": "meter",
            "fixed_control_background": weak,
            "evidence": [item["observed_condition_evidence"] for item in first]},))
    assert plan is not None
    assert plan["fixed_control_background"] != weak
    assert abs(plan["fixed_control_background"]["drive"]) == 1.
    assert plan["predicted_branch_contrast"] > 0.
    assert plan["estimated_branch_balance"] > 0.
    assert plan["experiment_revision_basis"] == "measured_background_misfit"
    assert plan["compute_cost"] == planning.spent <= 15000
    assert all(set(point) == set(spec().feature_bounds) for point in plan["probe"].sequence)
    acquired = measured_rows(plan["fixed_control_background"])
    second_budget = SearchBudget(40000)
    second = propose_observed_condition_duration(spec(), acquired, second_budget)
    assert second[0]["observed_condition_evidence"]["status"] == "conditional_duration"
    assert second[0]["observed_condition_evidence"]["width_candidates"] == [5]
    assert second[0]["training_loss"] < 1e-12
    assert 0 < second_budget.spent <= 40000


def test_one_control_can_change_a_previously_purchased_background():
    task = TaskSpec("one_command", "Buy a new trajectory.", ("drive", "meter"), (), (),
        feature_bounds={"drive": (-1., 1.)}, max_probe_length=32)
    rows = [Observation({"drive": drive, "meter": .2 * drive + noise}, 1.)
            for drive in (-1., 0., 1.) for noise in (-.8, -.3, .3, .8)]
    selected = {"candidate": Candidate("baseline", feature("drive"), "research")}
    planning = SearchBudget(15000)
    first = plan_observed_condition_probe(task, rows, selected, planning)
    assert first is not None
    second = plan_observed_condition_probe(task, rows, selected, planning,
        previous_experiments=({"fixed_control_background": first["fixed_control_background"]},))
    assert second is not None
    assert second["fixed_control_background"] != first["fixed_control_background"]
    assert all(-1. <= point["drive"] <= 1. for point in second["probe"].sequence)
    assert planning.spent <= 15000
