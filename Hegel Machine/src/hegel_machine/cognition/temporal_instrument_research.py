"""Measured instrument subproblems for the temporal inquiry interface.

This adapts the existing finite-history/state fitter and disagreement planner;
it supplies neither the apparatus family nor its response equation. The caller
opens the child question before purchasing readings and uses a returned program
only after a separate measured trajectory supports a unique explanation.
"""

from .instrument_hypotheses import (
    assess_instrument_hypotheses,
    instrument_model_status,
    plan_instrument_disagreement,
)
from .instrument_models import fit_instrument_model, instrument_binding
from .models import Observation, Probe, SearchBudget, TaskSpec


def _specs(public_spec):
    controls = tuple(public_spec["controls"])
    common = {"initial_probes": (), "available_probes": (),
              "max_probe_length": public_spec["max_probe_length"]}
    command = TaskSpec("temporal_controls", "Observed control histories", controls,
        feature_bounds={name: tuple(public_spec["feature_bounds"][name])
                        for name in controls}, **common)
    measured = TaskSpec("temporal_meter", "Purchased auxiliary readings", ("meter",), **common)
    return command, measured


def _paired_rows(records):
    pairs = []
    for record in records:
        measurements = {item["observation_index"]: item["values"]["meter"]
                        for item in record["measurements"]}
        if not measurements:
            continue
        for index, row in zip(record["row_indices"], record["rows"]):
            # The child receives meter labels, never the parent response target.
            command = Observation(dict(row["features"]), 0.,
                                  tuple(dict(point) for point in row["history"]))
            measured = Observation({"meter": measurements[index]}, 0.)
            pairs.append((command, measured))
    return pairs


def fit_temporal_instrument(records, public_spec, *, compute_budget=100000):
    """Fit competing finite and recurrent programs from complete paid records.

    The existing identification family requires at least six measured rows.
    Short trajectories can deliberately retain both explanations; no single
    fitted explanation is adopted before prospective measured validation.
    """
    command, measured = _specs(public_spec)
    return fit_instrument_model(command, measured, _paired_rows(records),
                                SearchBudget(compute_budget), max_history=4)


def assess_temporal_instrument(model, records, public_spec, *,
                               compute_budget=100000, tolerance=1e-10):
    """Update candidates using newly purchased measurements, without refitting."""
    command, measured = _specs(public_spec)
    result = assess_instrument_hypotheses(model, command, measured,
        _paired_rows(records), SearchBudget(compute_budget), tolerance)
    model["status"] = instrument_model_status(model)
    return result


def select_temporal_instrument_probe(model, public_spec, *, max_length=None,
                                     serial=0, compute_budget=100000,
                                     tolerance=1e-10, methods=None):
    """Choose the next purchase from disagreement between current explanations.

    Returned forecasts and the action are serializable. Selection reads only
    fitted programs and the public control interface, not an apparatus object.
    """
    command, measured = _specs(public_spec)
    result = plan_instrument_disagreement(model, command, measured,
        SearchBudget(compute_budget), max_length=max_length, serial=serial,
        tolerance=tolerance, methods=methods)
    probe = result.pop("probe")
    if probe is not None:
        result["probe_id"] = probe.probe_id
    result["action"] = (None if probe is None else {
        "kind": "trajectory", "sequence": [dict(point) for point in probe.sequence],
        "channels": ["meter"]})
    return result


def observe_temporal_instrument_method(methods, plan, assessment, model, public_spec, *,
                                       task_id, record, tolerance=1e-10):
    """Return actual purchased evidence to the existing experiment-method book."""
    from .instrument_experiment_methods import observe_instrument_method

    command, _ = _specs(public_spec)
    measured_plan = {**plan, "probe": Probe(plan["probe_id"],
        tuple(dict(point) for point in plan["action"]["sequence"]))}
    return observe_instrument_method(methods, measured_plan, assessment, model, command,
        task_id=task_id, observation_rows=len(record["rows"]),
        measurement_cost=record["measurement_cost"], tolerance=tolerance)


def temporal_instrument_program(model):
    """Return a unique prospectively supported controls-only explanation."""
    field = model.get("fields", {}).get("meter", {})
    survivors = [item for item in field.get("hypotheses", [])
                 if item.get("status") not in {"refuted", "counterexample"}]
    if len(survivors) != 1 or survivors[0].get("status") != "supported":
        return None
    return instrument_binding(model).get("meter")
