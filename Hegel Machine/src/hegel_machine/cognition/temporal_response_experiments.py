"""Expose the existing response competition and method book to inquiry records."""

from .competing_object_experiments import (
    assess_competing_object_probe, plan_competing_object_probe,
)
from .language import Program
from .models import Candidate, Observation, Probe, SearchBudget
from .temporal_instrument_research import _specs, temporal_instrument_program


def plan_response_experiment(hypotheses, public_spec, *, methods, instrument=None,
                             serial=0, length=4, compute_budget=100000, task_id=None):
    """Keep the legacy planner's candidate, calibration and method semantics."""
    command, _ = _specs(public_spec)
    entries = [{"hypothesis_id": item["hypothesis_id"],
                "candidate": Candidate(item["hypothesis_id"], Program.from_dict(item["program"]),
                                       "provided_response_hypothesis"),
                "training_loss": item.get("training_loss")} for item in hypotheses]
    needed = {name for entry in entries for name, _ in entry["candidate"].program.observed_variables()
              if name != "__target__"} - set(command.feature_names)
    available = {item["name"] for item in public_spec.get("channels", [])}
    if needed - available:
        raise ValueError(f"response hypotheses require unavailable measured fields: {sorted(needed - available)}")
    meter = temporal_instrument_program(instrument) if instrument is not None else None
    binding = {"meter": meter} if meter is not None and "meter" in needed else {}
    budget = SearchBudget(compute_budget)
    plan = plan_competing_object_probe(command, entries, budget, binding=binding,
        methods=methods, serial=serial, length=length, tolerance=1e-10, task_id=task_id)
    probe = plan.pop("probe")
    plan.update(action=None if probe is None else {"kind": "trajectory",
        "sequence": [dict(point) for point in probe.sequence], "channels": sorted(needed)},
        binding={name: program.to_dict() for name, program in binding.items()},
        control_bounds={name: list(bounds) for name, bounds in command.feature_bounds.items()},
        compute_cost=budget.spent)
    if probe is not None:
        plan["probe_id"] = probe.probe_id
    return plan


def assess_response_experiment(plan, record, public_spec, *, methods, task_id,
                               compute_budget=100000):
    """Use the purchased fields and their real history, never simulated readings."""
    command, _ = _specs(public_spec)
    readings = {item["observation_index"]: item["values"] for item in record["measurements"]}
    pairs, measured_history = [], []
    for index, row in zip(record["row_indices"], record["rows"]):
        controls = Observation(dict(row["features"]), row["target"], tuple(row["history"]))
        fields = {**row["features"], **readings.get(index, {})}
        measured = Observation(fields, row["target"], tuple(measured_history))
        pairs.append((controls, measured))
        measured_history.append({**fields, "__target__": row["target"]})
    measured_plan = {**plan, "probe": Probe(plan["probe_id"],
        tuple(dict(point) for point in plan["action"]["sequence"]))}
    budget = SearchBudget(compute_budget)
    result = assess_competing_object_probe(measured_plan, pairs, budget, tolerance=1e-10)
    result["method_feedback"] = methods.observe(measured_plan, result, task_id=task_id,
        command_spec=command, binding={name: Program.from_dict(program)
                                     for name, program in plan["binding"].items()}, tolerance=1e-10)
    return result
