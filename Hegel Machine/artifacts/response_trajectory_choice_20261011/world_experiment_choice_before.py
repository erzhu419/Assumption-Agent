"""Select an authored physical-outcome contrast with the existing probe chooser."""

from copy import deepcopy

from .language import Program
from .models import Candidate, Probe, SearchBudget
from .probing import choose_probe


def forecast_equivalence_classes(experiments):
    """Group members that agree on every currently executable experiment.

    This is the response panel's existing comparison rule. Groups are rebuilt
    for each offered pool; the scientific hypotheses themselves stay separate.
    """
    groups = {}
    if experiments:
        for identifier in experiments[0]['predictions']:
            signature = tuple(e['predictions'][identifier] for e in experiments)
            groups.setdefault(signature, []).append(identifier)
    return list(groups.values())


def select_experiment(design):
    """Compare prospective Boolean forecasts, without fitting or granting support.

    Each experiment declares one physical outcome whose truth all hypotheses
    predict. Forecast columns only adapt that table to the existing chooser;
    they are not world measurements. The author supplies outcome semantics and
    the caller executes and analyzes the selected original actions.
    """
    if not isinstance(design, dict):
        raise ValueError("design must return a dictionary with hypotheses and experiments")
    hypotheses = design.get("hypotheses", [])
    ids = [item.get("id") for item in hypotheses]
    if (len(ids) < 2 or any(not isinstance(value, str) or not value.strip() for value in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("provide at least two hypotheses with unique nonempty string ids")
    if any(not isinstance(item.get("claim"), str) or not item["claim"].strip() for item in hypotheses):
        raise ValueError("each hypothesis needs a nonempty claim")
    experiments = design.get("experiments", [])
    columns = [f"forecast_{index}" for index in range(len(ids))]
    candidates = [Candidate(identifier, Program("feature", name=column), "authored_forecast")
                  for identifier, column in zip(ids, columns)]
    probes, seen = [], set()
    for experiment in experiments:
        identifier = experiment.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
            raise ValueError("each experiment needs a unique nonempty string id")
        seen.add(identifier)
        if not isinstance(experiment.get("outcome"), str) or not experiment["outcome"].strip():
            raise ValueError("each experiment needs one stated physical outcome")
        actions, procedure = experiment.get("actions"), experiment.get("procedure")
        if procedure is not None:
            if (actions is not None or not isinstance(procedure, dict)
                    or not isinstance(procedure.get("path"), str) or not procedure["path"]
                    or type(procedure.get("max_steps", 12)) is not int
                    or procedure.get("max_steps", 12) < 1):
                raise ValueError("provide either actions or a procedure with path and positive max_steps")
        elif not isinstance(actions, list) or not actions:
            raise ValueError("each experiment needs executable actions or a procedure")
        forecasts = experiment.get("predictions", {})
        if (not isinstance(forecasts, dict) or set(forecasts) != set(ids)
                or any(type(value) is not bool for value in forecasts.values())):
            raise ValueError("predictions must contain one Boolean for every hypothesis id, and no others")
        probes.append(Probe(identifier, ({column: float(forecasts[key])
                                         for key, column in zip(ids, columns)},)))
    budget = SearchBudget(len(probes) * len(candidates))
    selected, separation = choose_probe(probes, candidates, 0., budget)
    experiment = next((item for item in experiments
                       if selected is not None and item["id"] == selected.probe_id), None)
    return {"diagnosis": ("no_experiments_proposed" if not probes else
                          "discriminating_experiment_available" if separation > 0 else "observational_ambiguity"),
            "selected_experiment": deepcopy(experiment) if separation > 0 else None,
            "hypotheses": deepcopy(hypotheses), "predicted_separation": separation,
            "experiment_count": len(probes), "comparison_scope": "authored_experiments_only",
            "compute_cost": budget.spent}
