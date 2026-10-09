"""Compare observed-field explanations and buy experiments that separate them."""

from math import isfinite
from random import Random

from .instrument_condition_probes import unmeasured_instrument_conditions
from .instrument_models import rebind_instrument_model
from .language import MissingHistory
from .models import Observation, Probe, SearchBudget


_REFUTED = {"refuted", "counterexample"}


def instrument_model_status(model):
    counts = [sum(item.get("status") not in _REFUTED
                  for item in field.get("hypotheses", [field]))
              for field in model.get("fields", {}).values()]
    if not any(counts):
        return "counterexample"
    if 0 in counts:
        return "partially_supported_on_measured_trajectory"
    return ("ambiguous_on_measured_trajectory" if max(counts) > 1
            else "supported_on_measured_trajectory")


def _choices(model, measured_spec):
    for name, field in model.get("fields", {}).items():
        hypotheses = field.get("hypotheses")
        if hypotheses is None:
            # Existing stored single explanations enter the same comparison.
            hypotheses = [{**field, "hypothesis_id": f"{name}:hypothesis:0",
                           "status": field.get("status", "training_candidate")}]
        for hypothesis in hypotheses:
            isolated = {**field, "program": hypothesis["program"]}
            isolated.pop("hypotheses", None)
            bindings = rebind_instrument_model({**model, "fields": {name: isolated}}, measured_spec)
            yield name, field, hypothesis, bindings


def _command_row(row, spec):
    names = set(spec.feature_names)
    return Observation({name: value for name, value in row.features.items() if name in names}, 0.,
                       tuple({name: value for name, value in past.items() if name in names}
                             for past in row.history))


def assess_instrument_hypotheses(model, command_spec, measured_spec, paired_rows, budget, tolerance):
    """Assess each hypothesis against purchased fields, never response targets.

    A previously measured counterexample remains a counterexample. Incomplete
    projection or insufficient scoring budget supplies no new evidence. Both
    mean and worst-row error are retained, so a long easy prefix cannot hide a
    measured boundary failure.
    """
    start, pairs = budget.spent, list(paired_rows)
    losses, assessed = {}, {}
    for name, field, hypothesis, bindings in _choices(model, measured_spec):
        if hypothesis.get("status") in _REFUTED:
            continue
        applicable = [(program, _command_row(command, command_spec), measured.features[current])
                      for current, program in bindings.items()
                      for command, measured in pairs if current in measured.features]
        if not applicable or not budget.spend(len(applicable)):
            continue
        try:
            errors = [(program.predict(command) - observed) ** 2
                      for program, command, observed in applicable]
        except (KeyError, MissingHistory):
            continue
        error = sum(errors) / len(errors)
        worst = max(errors)
        hypothesis.update(validation_loss=error, max_squared_error=worst,
                          validation_rows=len(errors),
                          status="supported" if isfinite(worst) and worst <= tolerance else "refuted")
        if "hypotheses" not in field:
            field["hypotheses"] = [hypothesis]
        losses.setdefault(name, {})[hypothesis["hypothesis_id"]] = error
        assessed.setdefault(name, set()).add(hypothesis["hypothesis_id"])
        depth = max(len(command.history) for _, command, _ in applicable)
        field["observed_history_depth"] = max(field.get("observed_history_depth", 0), depth)
        model["observed_history_depth"] = max(model.get("observed_history_depth", 0), depth)
    result = {"field_losses": losses, "supported_fields": [], "ambiguous_fields": [],
              "refuted_fields": [], "unassessed_fields": [], "compute_cost": budget.spent - start}
    for name, field in model.get("fields", {}).items():
        choices = field.get("hypotheses", [field])
        survivors = [item for item in choices if item.get("status") not in _REFUTED]
        if not survivors:
            result["refuted_fields"].append(name)
        elif len(survivors) > 1:
            result["ambiguous_fields"].append(name)
        elif (survivors[0].get("hypothesis_id") in assessed.get(name, ())
              and survivors[0].get("status") == "supported"):
            result["supported_fields"].append(name)
            # The field's summary must describe the surviving executable law.
            field.update({key: value for key, value in survivors[0].items()
                          if key not in {"hypothesis_id", "status"}})
        if any(item.get("hypothesis_id") not in assessed.get(name, ()) for item in survivors):
            result["unassessed_fields"].append(name)
    return result


def instrument_candidates(model, measured_spec):
    """Project the surviving competing explanations onto current measured fields."""
    fields = {}
    for _, _, hypothesis, bindings in _choices(model, measured_spec):
        if hypothesis.get("status") not in _REFUTED:
            for name, program in bindings.items():
                fields.setdefault(name, {})[hypothesis["hypothesis_id"]] = program
    return {name: choices for name, choices in fields.items() if len(choices) > 1}


def forecast_instrument_sequence(fields, sequence, budget, tolerance):
    """Price and execute the current explanations on a proposed control history."""
    result = {"predictions": {}, "field_disagreements": {},
              "predicted_separation": 0., "complete": False}
    cost = len(sequence) * sum(map(len, fields.values()))
    if not fields or not sequence or not budget.spend(cost):
        return result
    rows = [Observation(point, 0., tuple(sequence[:index])) for index, point in enumerate(sequence)]
    complete = True
    for name, choices in fields.items():
        predicted = {}
        for identifier, program in choices.items():
            try:
                values = [program.predict(row) for row in rows]
            except (KeyError, MissingHistory):
                complete = False
                continue
            if all(map(isfinite, values)):
                predicted[identifier] = values
            else:
                complete = False
        if len(predicted) < 2:
            continue
        separation = max((max(values) - min(values)) ** 2 for values in zip(*predicted.values()))
        if separation > tolerance:
            result["predictions"][name] = predicted
            result["field_disagreements"][name] = separation
    result.update(predicted_separation=sum(result["field_disagreements"].values()), complete=complete)
    return result


def instrument_boundary_sequences(fields, command_spec, budget, *, max_length):
    """Resolve current candidate cuts before constructing endpoint or interval probes."""
    if max_length <= 0:
        return []
    binding = {identifier: program for choices in fields.values() for identifier, program in choices.items()}
    questions = unmeasured_instrument_conditions(binding, command_spec, (), budget)
    if not questions:
        return []
    center = {name: (low + high) / 2 for name, (low, high) in command_spec.feature_bounds.items()}
    endpoints = tuple({**center, item["control"]: item["value"]} for item in questions[:max_length])
    cuts = {}
    for question in questions:
        cuts.setdefault(question["control"], set()).add(question["value"])
    middle = tuple({**center, name: (left + right) / 2}
                   for name, values in cuts.items()
                   for left, right in zip(sorted(values), sorted(values)[1:]))
    return [("at", endpoints), *([("between_adjacent", middle[:max_length])] if middle else [])]


def plan_instrument_disagreement(model, command_spec, measured_spec, budget, *,
                                 max_length=None, serial=0, tolerance=1e-7, methods=None):
    """Try a measured procedure, then search current field disagreement if needed."""
    start = budget.spent
    result = {"probe": None, "predictions": {}, "field_names": [], "compute_cost": 0,
              "reason": "no_distinguishable_hypotheses"}
    fields = instrument_candidates(model, measured_spec)
    cap = min(command_spec.max_probe_length,
              command_spec.max_probe_length if max_length is None else max_length)
    bounds = command_spec.feature_bounds
    if not fields or not bounds or cap <= 0 or not budget.remaining:
        if cap <= 0 or not budget.remaining:
            result["reason"] = "insufficient_budget"
        return result
    from .program_experiment_methods import _control_roles
    result["instrument_condition"] = {
        "programs": {name: {identifier: program.to_dict() for identifier, program in choices.items()}
                     for name, choices in fields.items()},
        "bounds": {name: list(interval) for name, interval in bounds.items()}}
    result["control_roles"] = _control_roles(
        [program for choices in fields.values() for program in choices.values()], {}, bounds)
    if methods is not None:
        from .instrument_experiment_methods import propose_instrument_method
        recalled = propose_instrument_method(methods, model, command_spec, measured_spec, budget,
            max_length=cap, serial=serial, tolerance=tolerance)
        if recalled["probe"] is not None:
            return {**recalled, "compute_cost": budget.spent - start}
        result.update({key: recalled[key] for key in
                       ("parent_method_id", "preparation_revision", "method_attempts") if key in recalled})
    depth = max([model.get("observed_history_depth", 0),
                 *(field.get("observed_history_depth", 0) for field in model.get("fields", {}).values())])
    length = min(cap, max(2, depth + 2))
    center = {name: (low + high) / 2 for name, (low, high) in bounds.items()}
    # Exact inferred cuts matter for open/closed membership alternatives.
    # Their extraction cannot consume the budget reserved for physical pulses.
    condition_budget = SearchBudget(budget.remaining // 4)
    boundaries = instrument_boundary_sequences(fields, command_spec, condition_budget, max_length=cap)
    budget.spend(condition_budget.spent)
    sequences = [("learned_boundary", sequence, strategy) for strategy, sequence in boundaries]
    for axes in (tuple(bounds), *((name,) for name in bounds)):
        for side in (1, 0):
            moved = {**center, **{name: bounds[name][side] for name in axes}}
            sequences.append(("pulse", tuple([moved, *[center] * (length - 1)]), None))
            sequences.append(("step", tuple([moved] * length), None))
    rng = Random(524287 + 1009 * serial)
    for _ in range(3):
        sequences.append(("random", tuple({name: rng.uniform(*interval) for name, interval in bounds.items()}
                                           for _ in range(length)), None))
    seen, best = set(), tolerance
    for mode, sequence, strategy in sequences:
        key = tuple(tuple(sorted(point.items())) for point in sequence)
        if key in seen:
            continue
        seen.add(key)
        forecast = forecast_instrument_sequence(fields, sequence, budget, tolerance)
        if not forecast["complete"]:
            if not budget.remaining:
                break
            continue
        separation = forecast["predicted_separation"]
        if separation > best:
            best = separation
            result.update(probe=Probe(f"instrument_disagreement:{serial}:{len(seen)}", sequence),
                          predictions=forecast["predictions"], field_names=sorted(forecast["predictions"]),
                          reason="predicted_field_disagreement", sequence_mode=mode,
                          predicted_separation=separation, field_disagreements=forecast["field_disagreements"])
            if strategy is not None:
                result["target_strategy"] = strategy
            else:
                result.pop("target_strategy", None)
    result["compute_cost"] = budget.spent - start
    return result
