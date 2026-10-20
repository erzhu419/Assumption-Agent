"""Bind public world readings to the existing finite numerical researcher.

The caller proposes variables, reading extraction and executable experiments.
This module supplies no world-specific law, target value or action mapping.
"""

from copy import deepcopy
from math import isfinite
from statistics import mean

from .evaluation import initial_prediction, predict
from .language import Program
from .models import Candidate, Observation, Probe, SearchBudget
from .probing import choose_probe
from .synthesis import _fit_linear, synthesize


def _observation(row):
    return Observation(row["features"], row.get("target", 0.), tuple(row.get("history", ())))


def _finite_history(row):
    return all(isfinite(float(value)) for point in row.get("history", ())
               for value in point.values())


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
        prediction = predict(program, _observation(proposal), fallback)
        probes.append({**deepcopy(proposal), "prediction": prediction})
    return {"program": program.to_dict(),
            "fit_mse": sum(item["error"] ** 2 for item in fitted) / len(fitted),
            "fit_residuals": fitted,
            "leave_one_out_mse": (sum(item["error"] ** 2 for item in held_out) / len(held_out)
                                  if held_out else None),
            "leave_one_out_residuals": held_out,
            "probe_predictions": probes,
            "evidence_use": "in_sample_fit_and_leave_one_out_reuse_not_fresh_confirmation"}


def research_relation(binding, observations, previous=None, *, compute_budget=None):
    """Fit, retain and distinguish a relation on explicitly cited public readings.

    ``previous`` is the last result. Its programs remain frozen while new rows
    test them; explicit ``candidate_additions`` screen new programs on paid rows.
    Rows may distinguish conditioning citations from new outcome citations. The
    host's available records determine when an outcome is prospective, while
    rows without these roles retain whole-row evidence independence. Citation
    existence is checked; extracting numerical meaning from a
    public message remains the caller's scientific hypothesis. A ``basis`` of
    Program expressions fits one affine coefficient vector with the existing
    fitter; it does not enumerate other functional forms. Explicit hypotheses
    take precedence, and a contradicted fit retains its declared basis. The
    ``operational_setting`` prediction objective additionally describes an
    empirical refit for physical testing; it does not change exact-law status.
    Explicit row and probe histories are part of their numerical inputs. A
    supplied SearchBudget shares the caller's account; compute_cost reports
    only the work consumed by this call.
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
        if "conditioning_refs" in row or "outcome_refs" in row:
            if "conditioning_refs" not in row or not row.get("outcome_refs"):
                raise ValueError("conditional rows need conditioning_refs and nonempty outcome_refs")
            refs = [*row["conditioning_refs"], *row["outcome_refs"]]
            if (any(type(index) is not int or index not in available for index in refs)
                    or set(refs) != set(evidence)):
                raise ValueError("conditioning_refs and outcome_refs must cover actual row evidence")
        values = [*row["features"].values(), row["target"]]
        if (not row["features"] or any(not isfinite(float(value)) for value in values)
                or not _finite_history(row)):
            raise ValueError("numerical rows need finite features, histories and targets")
    evidence = sorted({index for row in entries for index in row["evidence"]})
    result = {"status": "inconclusive", "diagnosis": "numerical_evidence_needed",
              "candidates": [], "fit_rows": [], "fit_evidence": [],
              "validation_evidence": [], "evidence": evidence,
              "selected_probe": None, "compute_cost": 0,
              "available_evidence": sorted(available), "bound_rows": deepcopy(entries)}
    if previous and previous.get("binding_history"):
        result["binding_history"] = deepcopy(previous["binding_history"])
    if previous and previous.get("rejected_candidates"):
        result["rejected_candidates"] = deepcopy(previous["rejected_candidates"])
    if "basis" in binding:
        result["basis"] = deepcopy(binding["basis"])
    if not entries:
        return result
    names = set(entries[0]["features"])
    if any(set(row["features"]) != names for row in entries):
        raise ValueError("all rows of a relation must bind the same feature names")
    rows = [_observation(row) for row in entries]
    fallback = initial_prediction(rows, .5)
    budget = compute_budget if compute_budget is not None else SearchBudget(50000)
    spent_before = budget.spent

    def scored_predictions(program, selected):
        if not budget.spend(len(selected)):
            raise ValueError("numerical comparison exceeds the bounded research allowance")
        predictions = [predict(program, row, fallback) for row in selected]
        errors = [(prediction - row.target) ** 2
                  for prediction, row in zip(predictions, selected)]
        return mean(errors), predictions, errors

    def score(program, selected):
        return scored_predictions(program, selected)[0]

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
                                   budget=allowance, max_history=binding.get("max_history", 0))
            budget.spend(allowance.spent)
            programs = [candidate.program.simplified() for candidate in generated]
        fitted = [new_candidate(program) for program in dict.fromkeys(programs)]
        fitted.sort(key=lambda item: (item["fit_mse"], Program.from_dict(item["program"]).size))
        # Preserve structurally different tied explanations for active tests.
        for item in fitted:
            item["status"] = "plausible" if item["fit_mse"] <= tolerance else "refuted"
        result.update(candidates=fitted[:12], fit_rows=deepcopy(entries), fit_evidence=evidence,
                      validation_evidence=[])

    def new_candidate(program):
        mse, predictions, errors = scored_predictions(program, rows)
        worst = sorted(range(len(rows)), key=errors.__getitem__, reverse=True)[:3]
        residuals = {"evaluation_scope": "paid_relation_fit", "evaluated_rows": len(rows),
                     "examples": [{**deepcopy(entries[index]),
                                   "prediction": predictions[index],
                                   "residual": predictions[index] - rows[index].target,
                                   "squared_error": errors[index]} for index in worst]}
        return {"program": program.to_dict(), "fit_mse": mse,
                "paid_fit_residuals": residuals,
                "validation_mse": None, "fit_rows": deepcopy(entries),
                "fit_evidence": evidence, "proposed_after": sorted(available)}

    def fresh_rows(after, *, fit_evidence=None):
        selected = []
        for row in entries:
            if "outcome_refs" in row:
                fresh = set(row["outcome_refs"]).isdisjoint(after)
            else:
                fresh = set(row["evidence"]).isdisjoint(
                    after if fit_evidence is None else fit_evidence)
            if fresh:
                selected.append(row)
        return selected

    def measured(program, selected):
        return score(program, [_observation(row) for row in selected]) if selected else None

    def refs(selected):
        return sorted({index for row in selected for index in row["evidence"]})

    def archive_rejected(items):
        archived = result.get("rejected_candidates", [])
        result["rejected_candidates"] = deepcopy(items) + [deepcopy(old) for old in archived
            if not any(old["program"] == item["program"] for item in items)]

    old_fit = previous.get("fit_rows", []) if previous else []
    old_candidates = (previous or {}).get("candidates", [])
    rows_changed = bool(previous and (
        any(row not in entries for row in previous.get("bound_rows", old_fit))
        or any(row not in entries for item in old_candidates
               for row in item.get("fit_rows", []))))
    old_basis = (previous or {}).get("basis")
    new_basis = binding.get("basis")
    basis_expanded = (new_basis is not None
                      and all(term in new_basis for term in (old_basis or [])))
    binding_changed = bool(previous and (rows_changed
        or (old_basis != new_basis and not basis_expanded)))
    if old_candidates and not binding_changed:
        result.update(fit_rows=deepcopy(old_fit), fit_evidence=deepcopy(previous["fit_evidence"]))
        validation_evidence = set()
        newly_rejected = []
        for old in old_candidates:
            item = deepcopy(old)
            item.setdefault("fit_rows", deepcopy(old_fit))
            item.setdefault("fit_evidence", deepcopy(previous["fit_evidence"]))
            item.setdefault("proposed_after", previous.get("available_evidence", sorted(available)))
            if previous.get("selected_program") == item["program"]:
                item.setdefault("confirmation_after", deepcopy(previous["confirmation_after"]))
                item.setdefault("selected_evidence", deepcopy(previous["selected_evidence"]))
            if item.get("status") == "refuted":
                result["candidates"].append(item)
                continue
            program = Program.from_dict(item["program"])
            validation = fresh_rows(set(item["proposed_after"]),
                                    fit_evidence=set(item["fit_evidence"]))
            confirmation = (fresh_rows(set(item["confirmation_after"]))
                            if "confirmation_after" in item else [])
            # Old paid outcomes can contradict an explanation without becoming
            # prospective confirmation merely because they were bound later.
            screening = [row for row in entries
                         if "outcome_refs" in row and row not in item["fit_rows"]]
            item.update(validation_mse=measured(program, validation),
                        validation_evidence=refs(validation),
                        confirmation_mse=measured(program, confirmation),
                        confirmation_evidence=refs(confirmation),
                        screening_mse=measured(program, screening),
                        screening_evidence=refs(screening))
            validation_evidence.update(item["validation_evidence"])
            item["status"] = "refuted" if any(item[key] is not None and item[key] > tolerance
                for key in ("validation_mse", "confirmation_mse", "screening_mse")) else "plausible"
            if item["status"] == "refuted":
                newly_rejected.append(item)
            result["candidates"].append(item)
        result["validation_evidence"] = sorted(validation_evidence)
        if newly_rejected:
            archive_rejected(newly_rejected)
        if (basis_expanded and any(term not in (old_basis or []) for term in new_basis)
                and not any(item["status"] == "plausible" for item in result["candidates"])):
            # A declared larger family is a new trial after the frozen bank
            # fails. Keep those failures rather than replacing them, and fit
            # only once: the returned basis records this actual trial.
            retained = result["candidates"]
            fit(refit=True)
            result["candidates"] = retained + [item for item in result["candidates"]
                if not any(item["program"] == old["program"] for old in retained)]
            result["diagnosis"] = "expanded_basis_requires_confirmation"
    else:
        if binding_changed:
            result.setdefault("binding_history", []).append({
                "basis": deepcopy(old_basis), "candidates": deepcopy(old_candidates),
                "bound_rows": deepcopy(previous.get("bound_rows", old_fit)),
                "available_evidence": deepcopy(previous.get("available_evidence", []))})
        fit()
        if rows_changed:
            # Re-extracting a row changes its scientific interpretation. Keep
            # executable old proposals, but screen them on the revised rows as
            # new versions; their earlier support remains in binding_history.
            for old in old_candidates:
                program = Program.from_dict(old["program"])
                if program.required_features() <= names and not any(
                        old["program"] == item["program"] for item in result["candidates"]):
                    item = new_candidate(program)
                    item["status"] = "plausible" if item["fit_mse"] <= tolerance else "refuted"
                    result["candidates"].append(item)
        result["diagnosis"] = "binding_revised_requires_confirmation" if binding_changed else "fresh_confirmation_required"

    # New hypotheses must be explicit. Rewriting the initial hypotheses field
    # does not change the frozen bank or silently replace a supported model.
    for supplied in binding.get("candidate_additions", []):
        program = Program.from_dict(supplied.get("program", supplied)).simplified()
        if any(program.to_dict() == item["program"] for item in
               result["candidates"] + result.get("rejected_candidates", [])):
            continue
        item = new_candidate(program)
        item["status"] = "plausible" if item["fit_mse"] <= tolerance else "refuted"
        result["candidates"].append(item)
        if item["status"] == "refuted":
            archive_rejected([item])

    if (old_candidates and not binding_changed and not any(
            item["status"] == "plausible" for item in result["candidates"])
            and (any(item.get("status") != "refuted" for item in old_candidates)
                 or any(row not in old_fit for row in entries))):
        # First test the frozen bank and retain its failures, then refit the
        # declared family. Screening a refit does not confirm it.
        archived = result.get("rejected_candidates", [])
        archive_rejected(archived + [item for item in result["candidates"]
            if item["status"] == "refuted" and not any(
                item["program"] == old["program"] for old in archived)])
        fit(refit=True)
        result.update(status="refuted", diagnosis="refit_needed")

    survivors = [item for item in result["candidates"] if item["status"] == "plausible"]
    if not survivors:
        result["status"] = "refuted"
        if result["diagnosis"] != "refit_needed":
            result["diagnosis"] = "candidate_family_refuted"

    supplied_probes = binding.get("probes", [])
    for proposal in supplied_probes:
        if set(proposal["features"]) != names or not proposal.get("actions"):
            raise ValueError("each experiment must bind every feature and executable actions")
        if (any(not isfinite(float(value)) for value in proposal["features"].values())
                or not _finite_history(proposal)):
            raise ValueError("numerical probes need finite features and histories")
    if len(survivors) == 1:
        selected = survivors[0]
        program = selected["program"]
        retained_selection = ("confirmation_after" in selected
                              and not binding_changed and result["diagnosis"] != "refit_needed")
        after = set(selected["confirmation_after"]) if retained_selection else available
        confirmation = fresh_rows(after)
        selected.update(confirmation_after=sorted(after),
                        selected_evidence=selected["selected_evidence"] if retained_selection else evidence,
                        confirmation_evidence=refs(confirmation))
        result.update(selected_program=deepcopy(program), confirmation_after=sorted(after),
                      selected_evidence=deepcopy(selected["selected_evidence"]),
                      confirmation_evidence=refs(confirmation))
        if confirmation:
            selected["supported_evidence"] = refs(confirmation)
            result.update(status="supported", diagnosis="confirmed_selected_candidate")
        elif result["diagnosis"] != "refit_needed":
            result["diagnosis"] = "needs_confirmation"
        if not confirmation:
            proposal = next((item for item in supplied_probes
                             if all((item["features"], item.get("history", []))
                                    != (row["features"], row.get("history", []))
                                    for row in entries)), None)
            if proposal is not None:
                if not budget.spend():
                    raise ValueError("numerical comparison exceeds the bounded research allowance")
                result["selected_probe"] = {**deepcopy(proposal), "purpose": "confirmation",
                    "predictions": [predict(Program.from_dict(program), _observation(proposal), fallback)]}

    if len(survivors) > 1:
        candidates = [Candidate(str(index), Program.from_dict(item["program"]), "world_relation")
                      for index, item in enumerate(survivors)]
        probes = []
        for index, proposal in enumerate(supplied_probes):
            probes.append(Probe(str(index), (*proposal.get("history", ()), proposal["features"])))
        selected, separation = choose_probe(probes, candidates, fallback, budget, observations=rows)
        if selected is not None and separation > tolerance:
            proposal = deepcopy(supplied_probes[int(selected.probe_id)])
            probe_row = _observation(proposal)
            if not budget.spend(len(candidates)):
                raise ValueError("numerical comparison exceeds the bounded research allowance")
            predictions = [predict(candidate.program, probe_row, fallback) for candidate in candidates]
            proposal.update(predictions=predictions, predicted_separation=separation, purpose="discrimination")
            result["selected_probe"] = proposal
            if result["diagnosis"] != "refit_needed":
                result["diagnosis"] = "discriminating_experiment_available"
        elif result["diagnosis"] != "refit_needed":
            result["diagnosis"] = "observational_ambiguity"
    if binding.get("prediction_objective") == "operational_setting" and binding.get("basis") is not None:
        result["empirical_prediction"] = _empirical_prediction(binding, entries, rows, budget)
    result["compute_cost"] = budget.spent - spent_before
    return result
