"""A real apparatus resolves a coupled observational prefix by intervention."""

from dataclasses import replace
import json

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.competing_object_experiments import (
    assess_competing_object_probe, plan_competing_object_probe,
)
from hegel_machine.cognition.competing_object_programs import research_competing_object_programs
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.instrument_models import fit_instrument_model, rebind_instrument_model
from hegel_machine.cognition.models import Observation, Probe, SearchBudget
from hegel_machine.cognition.object_induction import infer_object_model
from hegel_machine.cognition.object_interventions import propose_object_probe


def pairs(probe, observations):
    history, result = [], []
    for point, row in zip(probe.sequence, observations):
        result.append((Observation(point, row.target, tuple(history)), row))
        history.append({**point, "__target__": row.target})
    return result


def identify_partial_model(lab):
    spec, records, budget = lab.spec_for(("fine",)), [], SearchBudget(220000)
    model = infer_object_model(spec, records, budget)
    for serial in range(130):
        probe, _ = propose_object_probe(spec, model, records, serial)
        if probe is None:
            break
        batch = lab.acquire(probe, ("fine",))
        records.extend({"controls": point, **transition, "target": observation.target,
            "probe_id": probe.probe_id, "position": index,
            "history": list(probe.sequence[:index])}
            for index, (point, transition, observation) in enumerate(
                zip(probe.sequence, batch.transitions, batch.observations)))
        model = infer_object_model(spec, records, budget)
    assert not model["complete"] and all("f" in group and "x" not in group for group in model["groups"])
    return model


def test_coupled_old_observations_provoke_a_paid_experiment_and_revise_frozen_laws():
    settings = dict(objects=2, apparatus_seed=17011, name_seed=63940, wiring_seed=3792, seed=20097)
    warmup = AnonymousObjectLab(readout="focus", **settings)
    model = identify_partial_model(warmup)
    measured_spec = replace(warmup.spec_for(("fine",)), context={"object_model": model})
    calibration_probe = warmup.random_probe(7117, length=48)
    calibration_rows = warmup.acquire(calibration_probe, ("fine",)).observations
    calibration = fit_instrument_model(warmup.spec, measured_spec,
        pairs(calibration_probe, calibration_rows), SearchBudget(40000))
    binding = rebind_instrument_model(calibration, measured_spec)
    assert binding

    lab = AnonymousObjectLab(readout="position_focus", **settings)
    # The experiment defines an observed operating regime. The researcher
    # receives anonymous commands/values, never this private evaluator schema.
    truth = lab._evaluation_schema()
    positions = [values["x"] for values in truth.controls.values()]
    original = lab.random_probe(39019, length=48)
    prefix = Probe("linked_position_history", tuple({**point,
        **{name: point[positions[0]] for name in positions}} for point in original.sequence))
    acquired = lab.acquire(prefix, ("fine",))
    rows, budget = list(acquired.observations), SearchBudget(180000)
    spec = replace(lab.spec_for(("fine",)), context={"object_model": model})
    researched = research_competing_object_programs(spec, rows, budget, transitions=acquired.transitions)
    state = researched["competition"]
    assert state["status"] == "competing" and state["program_classes"] >= 2
    assert len({json.dumps(item["schema"], sort_keys=True) for item in researched["entries"]}) == 2
    assert all(item["training_loss"] < 1e-7 for item in researched["entries"])
    saved = json.loads(json.dumps(state))
    before = dict(lab.costs)
    plan = plan_competing_object_probe(lab.spec, researched["entries"], budget, binding=binding, serial=0)
    assert plan["forecast_status"] == "predicted_distinction" and plan["predicted_separation"] > .01
    assert lab.costs == before
    probe = plan["probe"]
    assert any(point[positions[0]] != point[positions[1]] for point in probe.sequence)
    new = lab.acquire(probe, ("fine",))
    assessed = assess_competing_object_probe(plan, pairs(probe, new.observations), budget)
    assert assessed["calibration_status"] == "supported_on_probe"
    assert assessed["status"] in {"distinguished", "all_refuted"}
    assert max(item["measured_loss"] for item in assessed["hypotheses"]) > .01
    assert lab.observation_cost - before["observation_cost"] == probe.cost
    assert lab.measurement_cost - before["measurement_cost"] == 6 * probe.cost

    revised = research_competing_object_programs(spec, [*rows, *new.observations], budget,
        previous=saved, new_observations=new.observations,
        transitions=[*acquired.transitions, *new.transitions])
    assert revised["entries"] and revised["competition"]["counterexamples"]
    assert min(loss(item["candidate"].program, new.observations, 0., .5)
               for item in revised["entries"]) < 1e-7
    assert json.loads(json.dumps(state)) == saved
    assert not model["complete"] and model["schema"] is None
    assert 0 < budget.spent < budget.limit
