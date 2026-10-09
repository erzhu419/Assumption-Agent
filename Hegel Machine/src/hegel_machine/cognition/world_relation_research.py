"""Bind public world readings to the existing finite numerical researcher.

The caller proposes variables, reading extraction and executable experiments.
This module supplies no world-specific law, target value or action mapping.
"""

from copy import deepcopy
from math import isfinite

from .evaluation import initial_prediction, loss, predict
from .language import Program
from .models import Candidate, Observation, Probe, SearchBudget
from .probing import choose_probe
from .synthesis import _fit_linear, synthesize


def _empirical_prediction(binding, entries, rows, budget):
    """Predict observed successful settings without asserting an exact law.

    Leave-one-out fits reuse the same evidence and declared basis. Their errors
    describe prediction on these measured objects, not fresh confirmation or a
    noise interval around an unknown physical activation range.
    """
    terms = [Program.from_dict(value) for value in binding["basis"]]

    def spend(amount):
        if not budget.spend(amount):
            raise ValueError("numerical comparison exceeds the bounded research allowance")

    def fit(selected):
        spend(len(selected))
        return _fit_linear(terms, selected).simplified()

    def residual(program, entry, row, fallback):
        spend(1)
        prediction = predict(program, row, fallback)
        return {**deepcopy(entry), "prediction": prediction,
                "observed_setting": row.target, "error": prediction - row.target}

    program = fit(rows)
    fallback = initial_prediction(rows, .5)
    fitted = [residual(program, entry, row, fallback) for entry, row in zip(entries, rows)]
    held_out = []
    if len(rows) >= 3:
        for index, (entry, row) in enumerate(zip(entries, rows)):
            training = rows[:index] + rows[index + 1:]
            held_out.append(residual(fit(training), entry, row, initial_prediction(training, .5)))
    probes = []
    for proposal in binding.get("probes", []):
        spend(1)
        prediction = predict(program, Observation(proposal["features"], 0.), fallback)
        probes.append({**deepcopy(proposal), "prediction": prediction})
    return {"program": program.to_dict(),
            "fit_mse": sum(item["error"] ** 2 for item in fitted) / len(fitted),
            "fit_residuals": fitted,
            "leave_one_out_mse": (sum(item["error"] ** 2 for item in held_out) / len(held_out)
                                  if held_out else None),
            "leave_one_out_residuals": held_out,
            "probe_predictions": probes,
            "evidence_use": "in_sample_fit_and_leave_one_out_reuse_not_fresh_confirmation"}


def research_relation(binding, observations, previous=None):
    """Fit, retain and distinguish a relation on explicitly cited public readings.

    ``previous`` is the last result. Its programs remain frozen while new rows
    test them. A validation row's entire evidence must be disjoint from fitting
    evidence. Citation existence is checked; extracting numerical meaning from a
    public message remains the caller's scientific hypothesis. A ``basis`` of
    Program expressions fits one affine coefficient vector with the existing
    fitter; it does not enumerate other functional forms. Explicit hypotheses
    take precedence, and a contradicted fit retains its declared basis. The
    ``operational_setting`` prediction objective additionally describes an
    empirical refit for physical testing; it does not change exact-law status.
    """
    tolerance = float(binding.get("tolerance", 1e-6))
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite nonnegative MSE")
    available = {record.get("index", index) for index, record in enumerate(observations)}
    entries = deepcopy(binding.get("rows", []))
    for row in entries:
        evidence = row.get("evidence", [])
        if not evidence or any(type(index) is not int or index not in available for index in evidence):
            raise ValueError("each numerical row must cite actual observation indices")
        values = [*row["features"].values(), row["target"]]
        if not row["features"] or any(not isfinite(float(value)) for value in values):
            raise ValueError("numerical rows need finite features and targets")
    evidence = sorted({index for row in entries for index in row["evidence"]})
    result = {"status": "inconclusive", "diagnosis": "numerical_evidence_needed",
              "candidates": [], "fit_rows": [], "fit_evidence": [],
              "validation_evidence": [], "evidence": evidence,
              "selected_probe": None, "compute_cost": 0}
    if "basis" in binding:
        result["basis"] = deepcopy(binding["basis"])
    if not entries:
        return result
    names = set(entries[0]["features"])
    if any(set(row["features"]) != names for row in entries):
        raise ValueError("all rows of a relation must bind the same feature names")
    rows = [Observation(row["features"], row["target"]) for row in entries]
    fallback = initial_prediction(rows, .5)
    budget = SearchBudget(50000)

    def score(program, selected):
        if not budget.spend(len(selected)):
            raise ValueError("numerical comparison exceeds the bounded research allowance")
        return loss(program, selected, fallback, .5)

    def fit(*, refit=False):
        supplied = None if refit else binding.get("hypotheses")
        if supplied:
            programs = [Program.from_dict(value.get("program", value)).simplified() for value in supplied]
        elif binding.get("basis") is not None:
            terms = [Program.from_dict(value) for value in binding["basis"]]
            if not budget.spend(len(rows)):
                raise ValueError("numerical comparison exceeds the bounded research allowance")
            programs = [_fit_linear(terms, rows).simplified()]
        else:
            allowance = SearchBudget(min(20000, budget.remaining // 2))
            generated = synthesize(rows, operations=binding.get("operations", ["refit", "composition", "relation"]),
                                   budget=allowance, max_history=0)
            budget.spend(allowance.spent)
            programs = [candidate.program.simplified() for candidate in generated]
        fitted = [{"program": program.to_dict(), "fit_mse": score(program, rows),
                   "validation_mse": None} for program in dict.fromkeys(programs)]
        fitted.sort(key=lambda item: (item["fit_mse"], Program.from_dict(item["program"]).size))
        # Preserve structurally different tied explanations for active tests.
        for item in fitted:
            item["status"] = "plausible" if item["fit_mse"] <= tolerance else "refuted"
        result.update(candidates=fitted[:12], fit_rows=deepcopy(entries), fit_evidence=evidence,
                      validation_evidence=[])

    old_fit = previous.get("fit_rows", []) if previous else []
    old_candidates = [item for item in (previous or {}).get("candidates", [])
                      if item.get("status") != "refuted"]
    binding_changed = bool(previous and (
        any(row not in entries for row in old_fit)
        or previous.get("basis") != binding.get("basis")))
    if old_candidates and not binding_changed:
        fit_evidence = set(previous["fit_evidence"])
        validation_entries = [row for row in entries if set(row["evidence"]).isdisjoint(fit_evidence)]
        validation = [Observation(row["features"], row["target"]) for row in validation_entries]
        result.update(fit_rows=deepcopy(old_fit), fit_evidence=sorted(fit_evidence),
                      validation_evidence=sorted({i for row in validation_entries for i in row["evidence"]}))
        for old in old_candidates:
            item = deepcopy(old)
            measured = score(Program.from_dict(item["program"]), validation) if validation else None
            confirmation_mse = None
            if previous.get("selected_program") == item["program"]:
                after = set(previous["confirmation_after"])
                fresh = [Observation(row["features"], row["target"]) for row in entries
                         if set(row["evidence"]).isdisjoint(after)]
                if fresh:
                    confirmation_mse = score(Program.from_dict(item["program"]), fresh)
            item.update(validation_mse=measured,
                        confirmation_mse=confirmation_mse,
                        status="refuted" if any(value is not None and value > tolerance
                            for value in (measured, confirmation_mse)) else "plausible")
            result["candidates"].append(item)
        if not any(item["status"] == "plausible" for item in result["candidates"]):
            result["rejected_candidates"] = deepcopy(result["candidates"])
            fit(refit=True)
            result.update(status="refuted", diagnosis="refit_needed")
    else:
        fit()
        result["diagnosis"] = "binding_revised_requires_confirmation" if binding_changed else "fresh_confirmation_required"

    survivors = [item for item in result["candidates"] if item["status"] == "plausible"]
    if not survivors:
        result["status"] = "refuted"
        if result["diagnosis"] != "refit_needed":
            result["diagnosis"] = "representation_inadequate"

    supplied_probes = binding.get("probes", [])
    for proposal in supplied_probes:
        if set(proposal["features"]) != names or not proposal.get("actions"):
            raise ValueError("each experiment must bind every feature and executable actions")
    if len(survivors) == 1:
        program = survivors[0]["program"]
        retained_selection = (previous and previous.get("selected_program") == program
                              and not binding_changed and result["diagnosis"] != "refit_needed")
        after = set(previous["confirmation_after"]) if retained_selection else available
        confirmation = [row for row in entries if set(row["evidence"]).isdisjoint(after)]
        result.update(selected_program=deepcopy(program), confirmation_after=sorted(after),
                      selected_evidence=deepcopy(previous["selected_evidence"]) if retained_selection else evidence,
                      confirmation_evidence=sorted({i for row in confirmation for i in row["evidence"]}))
        if confirmation:
            result.update(status="supported", diagnosis="confirmed_selected_candidate")
        elif result["diagnosis"] != "refit_needed":
            result["diagnosis"] = "needs_confirmation"
        if not confirmation:
            proposal = next((item for item in supplied_probes
                             if all(item["features"] != row["features"] for row in entries)), None)
            if proposal is not None:
                result["selected_probe"] = {**deepcopy(proposal), "purpose": "confirmation",
                    "predictions": [Program.from_dict(program).predict(Observation(proposal["features"], 0.))]}
                budget.spend()

    if len(survivors) > 1:
        candidates = [Candidate(str(index), Program.from_dict(item["program"]), "world_relation")
                      for index, item in enumerate(survivors)]
        probes = []
        for index, proposal in enumerate(supplied_probes):
            probes.append(Probe(str(index), (proposal["features"],)))
        selected, separation = choose_probe(probes, candidates, fallback, budget, observations=rows)
        if selected is not None and separation > tolerance:
            proposal = deepcopy(supplied_probes[int(selected.probe_id)])
            probe_row = Observation(proposal["features"], 0.)
            predictions = [candidate.program.predict(probe_row) for candidate in candidates]
            budget.spend(len(candidates))
            proposal.update(predictions=predictions, predicted_separation=separation, purpose="discrimination")
            result["selected_probe"] = proposal
            if result["diagnosis"] != "refit_needed":
                result["diagnosis"] = "discriminating_experiment_available"
        elif result["diagnosis"] != "refit_needed":
            result["diagnosis"] = "observational_ambiguity"
    if binding.get("prediction_objective") == "operational_setting" and binding.get("basis") is not None:
        result["empirical_prediction"] = _empirical_prediction(binding, entries, rows, budget)
    result["compute_cost"] = budget.spent
    return result
