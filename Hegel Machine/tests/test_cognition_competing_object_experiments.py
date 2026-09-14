"""Competing laws choose actual experiments and separate calibration failures."""

from dataclasses import asdict
import json

from hegel_machine.cognition.competing_object_experiments import (
    assess_competing_object_probe, plan_competing_object_probe,
)
from hegel_machine.cognition.continuous_state import _linear_fit, _sum_terms
from hegel_machine.cognition.instrument_models import fit_instrument_model, rebind_instrument_model
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.structure_lab import ObjectSchema
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab


def _entries(*programs):
    return [{"candidate": Candidate(f"candidate_{index}", program, "representation"),
             "hypothesis_id": f"hypothesis_{index}", "training_loss": 0.}
            for index, program in enumerate(programs)]


def _pairs(probe, measured):
    history, pairs = [], []
    for point, row in zip(probe.sequence, measured):
        pairs.append((Observation(point, row.target, tuple(history)), row))
        history.append({**point, "__target__": row.target})
    return pairs


def test_predictions_are_frozen_and_calibration_failure_does_not_refute_a_measured_law():
    spec = TaskSpec("unknown", "unknown", ("command",), (), (),
                    feature_bounds={"command": (-2., 2.)})
    measured = Program("feature", name="sensor")
    entries = _entries(measured, Program("mul", children=(measured, measured)))
    work = SearchBudget(2000)
    # A previously learned identity calibration has changed before this probe.
    plan = plan_competing_object_probe(spec, entries, work,
        binding={"sensor": Program("feature", name="command")})
    frozen = json.dumps({**plan, "probe": asdict(plan["probe"])}, sort_keys=True)
    assert plan["predicted_separation"] > 0 and plan["forecast_status"] == "predicted_distinction"
    actual = [Observation({"sensor": 2 * point["command"]}, 2 * point["command"])
              for point in plan["probe"].sequence]
    result = assess_competing_object_probe(plan, _pairs(plan["probe"], actual), work)
    assert result["status"] == "distinguished" and result["calibration_status"] == "counterexample"
    assert result["hypotheses"][0]["status"] == "supported_on_probe"
    assert result["hypotheses"][0]["forecast_loss"] > 1e-7
    assert result["hypotheses"][1]["status"] == "counterexample"
    assert json.dumps({**plan, "probe": asdict(plan["probe"])}, sort_keys=True) == frozen
    assert 0 < work.spent < work.limit


def test_missing_calibration_buys_measured_fields_without_inventing_forecasts():
    spec = TaskSpec("unknown", "unknown", ("c",), (), (), feature_bounds={"c": (-1., 1.)})
    field = Program("feature", name="m")
    entries = _entries(field, Program("const", value=0.))
    plan = plan_competing_object_probe(spec, entries, SearchBudget(100))
    assert plan["forecast_status"] == "requires_paired_measurements"
    assert plan["probe"] is not None and plan["predicted_fields"] == {}
    assert all(item["predictions"] is None for item in plan["hypotheses"])
    actual = [Observation({"m": point["c"]}, point["c"]) for point in plan["probe"].sequence]
    result = assess_competing_object_probe(plan, _pairs(plan["probe"], actual), SearchBudget(100))
    assert result["status"] == "distinguished"
    assert result["calibration_status"] == "not_forecast"
    assert all(item["forecast_loss"] is None for item in result["hypotheses"])


def test_equal_programs_and_exhausted_search_do_not_request_a_distinction():
    spec = TaskSpec("unknown", "unknown", ("c",), (), (), feature_bounds={"c": (-1., 1.)})
    a, b = Program("feature", name="a"), Program("feature", name="b")
    shared = _entries(Program("add", children=(a, b)), Program("add", children=(b, a)))
    assert plan_competing_object_probe(spec, shared, SearchBudget(100))["forecast_status"] == "algebraically_shared"
    separate = _entries(Program("feature", name="c"), Program("const", value=0.))
    empty = SearchBudget(0)
    plan = plan_competing_object_probe(spec, separate, empty)
    assert plan["probe"] is None and plan["forecast_status"] == "compute_exhausted"


def test_temporal_candidates_forecast_reset_trajectories_and_actual_histories():
    spec = TaskSpec("sequence", "unknown", ("c",), (), (), feature_bounds={"c": (-1., 1.)})
    field = Program("feature", name="c")
    summed = Program("window_sum", lag=3, children=(field,))
    plan = plan_competing_object_probe(spec, _entries(field, summed), SearchBudget(2000))
    history, measured = [], []
    for point in plan["probe"].sequence:
        row = Observation(point, 0., tuple(history))
        value = summed.predict(row)
        measured.append(Observation(point, value, tuple(history)))
        history.append({**point, "__target__": value})
    result = assess_competing_object_probe(plan, _pairs(plan["probe"], measured), SearchBudget(100))
    assert result["status"] == "distinguished"
    assert result["hypotheses"][1]["measured_loss"] == 0.
    assert result["hypotheses"][1]["forecast_loss"] == 0.


def test_purchased_object_response_is_separated_using_learned_instrument_bindings():
    # Only the laboratory's public schema, purchased measurements and response
    # labels enter fitting. No private transition/readout coefficient is read.
    lab = UnifiedObjectLab(objects=3, readout="total", apparatus_seed=33013)
    measured_spec = lab.spec_for(("fine",))
    schema = ObjectSchema.from_dict(json.loads(measured_spec.context["object_schema"]))
    calibration_probe = lab.random_probe(701, length=48)
    calibration_rows = lab.acquire(calibration_probe, ("fine",)).observations
    calibration = fit_instrument_model(lab.spec, measured_spec,
        _pairs(calibration_probe, calibration_rows), SearchBudget(24000))
    binding = rebind_instrument_model(calibration, measured_spec)
    fields = [schema.fields[group][role] for group in schema.objects for role in ("q", "u")]
    terms = [Program("feature", name=name) for name in fields]
    coefficients = _linear_fit([[row.features[name] for row in calibration_rows] for name in fields],
                              [row.target for row in calibration_rows])
    learned = _sum_terms(terms, coefficients[1:], coefficients[0])
    left, right = [Program("feature", name=schema.fields[group]["x"]) for group in schema.objects[:2]]
    alternative = Program("add", children=(learned, Program("sub", children=(left, right))))
    entries = _entries(learned, alternative)
    observations_before, measurements_before = lab.observation_cost, lab.measurement_cost
    plan = plan_competing_object_probe(lab.spec, entries, SearchBudget(4000), binding=binding, serial=31)
    assert lab.observation_cost == observations_before
    probe = plan["probe"]
    assert probe is not None and plan["predicted_separation"] > 1.
    batch = lab.acquire(probe, ("fine",))
    result = assess_competing_object_probe(plan, _pairs(probe, batch.observations), SearchBudget(1000))
    assert lab.observation_cost - observations_before == probe.cost
    assert lab.measurement_cost - measurements_before == 9 * probe.cost
    assert result["status"] == "distinguished" and result["calibration_status"] == "supported_on_probe"
    assert result["hypotheses"][0]["measured_loss"] < 1e-7
    assert result["hypotheses"][1]["measured_loss"] > 1.
