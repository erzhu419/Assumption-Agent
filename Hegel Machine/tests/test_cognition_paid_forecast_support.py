"""Current paid responses, not large failed extrapolations, price a probe."""

from copy import deepcopy
from dataclasses import replace

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.evaluation import initial_prediction, loss
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.noise_replication import assess_noise_replication
from hegel_machine.cognition.observed_condition_experiments import plan_observed_condition_probe
from hegel_machine.cognition.observed_condition_duration import propose_observed_condition_duration
from hegel_machine.cognition.observed_partial_structure import partial_structure_competitors, partial_structure_nominations
from test_cognition_competing_condition_experiments import apparatus_rows, conditional, entry, feature
from test_cognition_partial_structure_research import agent_before, factory, paid_events, recorded, rows_from


def test_refuted_large_parameters_cannot_redirect_or_inflate_a_supported_question():
    spec, rows = apparatus_rows()
    selected = entry(Program("const", value=0.), "current")
    rival = entry(conditional(feature("drive")), "uncovered_boundary", ((-.1, .1),))
    rejected = entry(Program("add", children=(Program("const", value=1000.),
        conditional(feature("bias")))), "failed_parameters", ((-1., 1.),))
    rejected["partial_support"] = [{"response_supported": True, "local_loss": 0.}]
    saved = deepcopy(rejected)
    before = plan_observed_condition_probe(spec, rows, selected, SearchBudget(15000), finalists=(rival,))
    after = plan_observed_condition_probe(spec, rows, selected, SearchBudget(15000),
        finalists=(rival,), partial_competitors=(rejected,))
    assert before is not None and after is not None
    assert after["fixed_control_background"] == before["fixed_control_background"]
    assert after["expected_disagreement"] == pytest.approx(before["expected_disagreement"])
    assert after["threshold_intervals"] == before["threshold_intervals"]
    assert {item["candidate_id"] for item in after["competing_hypotheses"]} == {
        "current", "uncovered_boundary"}
    assert rejected == saved  # Its structure is still available for refitting.


def test_jointly_failed_parameters_leave_real_residual_exploration_without_forecast_value():
    spec, rows = apparatus_rows()
    rows = [replace(row, target=row.features["bias"] ** 2) for row in rows]
    selected = entry(Program("const", value=2.), "less_wrong")
    rejected = entry(Program("add", children=(Program("const", value=1000.),
        conditional(feature("bias")))), "more_wrong", ((-1., 1.),))
    selected["partial_support"] = rejected["partial_support"] = [{"local_loss": 0.}]
    previous = ({"fixed_control_background": {"drive": 1., "bias": 2.}},)
    before = plan_observed_condition_probe(spec, rows, selected, SearchBudget(15000),
        previous_experiments=previous)
    budget = SearchBudget(15000)
    after = plan_observed_condition_probe(spec, rows, selected, budget,
        partial_competitors=(rejected,), previous_experiments=previous)
    assert before is not None and after is not None
    assert after["fixed_control_background"] == before["fixed_control_background"]
    assert after["fixed_control_background"] != previous[0]["fixed_control_background"]
    assert after["expected_disagreement"] == after["partial_structure_disagreement"] == 0.
    assert after["predicted_branch_contrast"] == after["forecast_length"] == 0
    assert not after["competing_hypotheses"] and not after["threshold_intervals"]
    assert after["experiment_revision_basis"] == "measured_background_misfit"
    assert 0 < after["compute_cost"] == budget.spent <= budget.limit
    assert after["probe"].cost == 24


def test_paid_validation_refutes_a_training_fit_without_inherited_loss_metadata():
    spec, rows = apparatus_rows()
    selected = entry(Program("const", value=0.), "current")
    rival = entry(conditional(feature("drive")), "training_fit", ((-.1, .1),))
    validation = [replace(row, features={**row.features, "drive": 1.}) for row in rows]
    assert plan_observed_condition_probe(spec, rows, selected, SearchBudget(15000),
        finalists=(rival,)) is not None
    assert plan_observed_condition_probe(spec, rows, selected, SearchBudget(15000),
        finalists=(rival,), validation=validation) is None
    # A revised parameterization must be evaluated as itself, even if a caller
    # retained an older numerical loss or condition record in its entry.
    revised = {**selected, "validation_loss": 1000., "condition_uncertainty": {
        "program": rival["candidate"].program.to_dict(), "original_target_loss": 1000.}}
    plan = plan_observed_condition_probe(spec, rows, revised, SearchBudget(15000), finalists=(rival,))
    assert plan is not None and plan["expected_disagreement"] > 0.
    assert {item["candidate_id"] for item in plan["competing_hypotheses"]} == {"current", "training_fit"}


def test_noise_unresolved_rivals_keep_forecasts_when_real_repeat_explains_their_error():
    spec, rows = apparatus_rows()
    original = [replace(row, target=.1 if index % 2 else -.1) for index, row in enumerate(rows)]
    repeated = [replace(row, target=-row.target) for row in original]
    selected = entry(feature("drive"), "instantaneous")
    rival = entry(Program("window_sum", lag=3, children=(feature("drive"),)), "window3")
    for candidate in (selected, rival):
        replication = assess_noise_replication(candidate["candidate"].program,
            original, repeated, SearchBudget(1000))
        assert replication["noise_supported"] and not replication["repeatable_error"]
        candidate["response_noise_evidence"] = {
            "program": candidate["candidate"].program.to_dict(), "supported": False,
            "replication": replication}
    plan = plan_observed_condition_probe(spec, original, selected, SearchBudget(15000),
        finalists=(rival,), validation=repeated)
    assert plan is not None and plan["duration_disagreement"] > 0.
    assert {item["candidate_id"] for item in plan["competing_hypotheses"]} == {"instantaneous", "window3"}

    # A new paid error is not explained by the old noise allowance; the models
    # remain proposals, while the next experiment uses observed residuals.
    changed = [replace(row, target=10. + row.target) for row in repeated]
    later = plan_observed_condition_probe(spec, original, selected, SearchBudget(15000),
        finalists=(rival,), validation=changed)
    assert later is not None and later["expected_disagreement"] == 0.
    assert not later["competing_hypotheses"]
    assert selected["response_noise_evidence"]["replication"]["noise_supported"]


def test_refuted_selected_program_cannot_supply_the_forecast_history_or_branch_contrast():
    spec, rows = apparatus_rows()
    first = entry(feature("drive"), "instantaneous")
    second = entry(Program("add", children=(feature("drive"), Program("if", children=(
        Program("available", lag=1), Program("lag", name="__target__", lag=1),
        Program("const", value=0.))))), "response_feedback")
    bad = entry(conditional(Program("const", value=1000.)), "failed_selected", ((-.1, .1),))
    previous = ({"fixed_control_background": {"drive": 0., "bias": 0.}},)
    before = plan_observed_condition_probe(spec, rows, first, SearchBudget(15000),
        finalists=(second,), previous_experiments=previous)
    after = plan_observed_condition_probe(spec, rows, bad, SearchBudget(15000),
        finalists=(first, second), previous_experiments=previous)
    assert after is not None and before is not None
    assert after["expected_disagreement"] == pytest.approx(before["expected_disagreement"])
    assert after["fixed_control_background"] == before["fixed_control_background"]
    assert after["predicted_branch_contrast"] == 0.
    assert {item["candidate_id"] for item in after["competing_hypotheses"]} == {"instantaneous", "response_feedback"}


def test_actual_paid_partial_structures_keep_exploration_without_using_failed_parameters(recorded, factory):
    varied = recorded["initial_varied_batches"]
    training = [row for index in (0, 1, 2, 4) for row in rows_from(varied[index])]
    training.extend(row for experiment in recorded["paid_experiments"]
                    for row in rows_from(experiment["batch"]))
    validation = rows_from(varied[3])
    events = paid_events(recorded)
    nominated = partial_structure_nominations(events)
    agent = agent_before(recorded)
    with factory(recorded["task"]) as laboratory:
        spec = agent._spec(laboratory, ("meter",))
        claims = partial_structure_competitors(spec, training, events, SearchBudget(20000))
        saved = deepcopy(claims)
        current = [*training, *validation]
        assert len(claims) >= 3
        assert all(loss(claim["candidate"].program, current, initial_prediction(current, .5), .5)
                   > agent.config.minimum_improvement for claim in claims)
        previous = [experiment["plan"] for experiment in recorded["paid_experiments"]]
        budget = SearchBudget(15000)
        plan = plan_observed_condition_probe(spec, training, claims[0], budget,
            validation=validation, partial_competitors=claims[1:], previous_experiments=previous,
            tolerance=agent.config.minimum_improvement)
        assert laboratory.costs["observation_cost"] == laboratory.costs["measurement_cost"] == 0
    assert plan is not None and plan["probe"].cost == 24
    assert plan["expected_disagreement"] == plan["partial_structure_disagreement"] == 0.
    assert plan["experiment_revision_basis"] == "measured_background_misfit"
    assert not plan["competing_hypotheses"]
    assert all(plan["fixed_control_background"] != old["fixed_control_background"] for old in previous)
    assert plan["compute_cost"] == budget.spent > 0
    assert claims == saved and partial_structure_nominations(events) == nominated


def test_failed_global_parameters_retain_a_locally_measured_boundary_on_the_right_instrument():
    from test_cognition_observed_temporal_identifiability import measured_rows, response_program, spec

    local = measured_rows({"drive": 1., "bias": 0.})
    evidence = propose_observed_condition_duration(spec(), local, SearchBudget(40000))[0][
        "observed_condition_evidence"]
    assert evidence["response_supported"] and evidence["condition_identified"]
    _, pilot = apparatus_rows()
    pilot = [replace(row, target=response_program().predict(row)) for row in pilot]
    rows = [replace(row, features={**row.features, "decoy": -row.features["meter"]})
            for row in [*pilot, *local]]
    measured_spec = replace(spec(), feature_names=("drive", "bias", "decoy", "meter"))
    failed = entry(Program("const", value=1000.), "failed_global_parameters")
    failed["observed_condition_evidence"] = evidence
    plan = plan_observed_condition_probe(measured_spec, rows, failed, SearchBudget(15000))
    assert plan is not None and plan["expected_disagreement"] == 0.
    assert plan["condition_field"] == "meter"  # Not the first measured field.
    assert plan["threshold_interval"] == evidence["threshold_interval"]
    assert plan["threshold_intervals"][0]["source_operation"] == "paid_local_condition_support"
    assert plan["forecast_length"] == 0 and not plan["competing_hypotheses"]
