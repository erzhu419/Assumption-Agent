"""Finite coefficient and response hypotheses from public calibration records.

This is an adapter to the existing linear fitter and Boolean experiment chooser.
Successful settings are not point labels for an unknown resonance: every actual
positive and negative response constrains a proposed *shared* residual band.
"""

from copy import deepcopy
from itertools import product
from math import ceil, floor

from .language import Program
from .models import Observation
from .synthesis import _fit_linear
from .world_experiment_choice import select_experiment


def _fit_sets(references, responses, terms):
    """Enumerate observed successes, never hypothetical measurement precision."""
    base = [(row["crystal"], row["frequency"]["value"]) for row in references]
    reference_ids = {number for number, _ in base}
    successes = {}
    for item in responses:
        if item["activated"] and item["crystal"] not in reference_ids:
            values = successes.setdefault(item["crystal"], [])
            if item["frequency"] not in values:
                values.append(item["frequency"])
    full_sets = [("initial_references", base)]
    if successes:
        full_sets.extend(("observed_success_combination", base + list(zip(successes, values)))
                         for values in product(*successes.values()))
    for origin, selected in full_sets:
        yield origin, selected
        if len(selected) - 1 >= terms + 1:
            for index, (number, _) in enumerate(selected):
                yield origin + "/leave_out_" + str(number), selected[:index] + selected[index + 1:]


def _setting_proposals(prediction, lower, upper, grid):
    """Reachable support endpoints and their midpoint on the public control grid."""
    step, origin = grid["step"], grid["origin"]
    left = ceil((max(grid["lower"], prediction + lower) - origin) / step)
    right = floor((min(grid["upper"], prediction + upper) - origin) / step)
    if left > right:
        return []
    return [origin + step * index for index in sorted({left, right, (left + right) // 2})]


def _propose_fitted_responses(state, estimators, path):
    """Return unconfirmed, fixed response hypotheses and an executable contrast.

    Coefficient alternatives are a finite set of empirical fits, not a confidence
    region. Band endpoints are also finite hypotheses: positive residual extrema
    and midpoints to the nearest negative residuals. No activation width or sensor
    precision is supplied. Rejection applies to the joint coefficient/band model,
    not to every coefficient vector expressible by its basis.
    """
    table = state.get("research_table", {})
    references = table.get("references", [])
    rows = {row["crystal"]: row for row in references + table.get("targets", [])}
    responses = state.get("response_settings", [])
    number = next((item for item in state.get("target_numbers", [])
                   if item not in state.get("completed", [])), None)
    result = {"status": "response_evidence_needed", "target_number": number,
        "assumptions": ["activation is deterministic for the recorded features and setting",
            "one connected residual activation band is shared across the recorded crystals",
            "finite coefficient fits and finite band hypotheses are not confidence regions"],
        "candidates": [], "hypotheses": [], "selection": None,
        "selected_setting": None, "selected_number": None, "compute_cost": 0,
        "response_count": len(responses),
        "evidence": sorted({index for item in responses for index in item.get("evidence", [])})}
    if number not in rows or not references or not any(item["activated"] for item in responses):
        return result
    features = {key: {name: value["value"] for name, value in row["measurements"].items()}
                for key, row in rows.items()}
    seen = {}
    for estimator in estimators:
        terms = [Program.from_dict(term) for term in estimator["basis"]]
        required = set().union(*(term.required_features() for term in terms))
        if any(item["crystal"] not in features or not required <= features[item["crystal"]].keys()
               for item in responses) or not required <= features[number].keys():
            result["candidates"].append({"family": estimator["id"], "status": "unavailable",
                                          "reason": "missing_public_measurements"})
            continue
        result["evidence"] = sorted(set(result["evidence"])
            | {row["frequency"]["evidence"] for row in references}
            | {row["measurements"][name]["evidence"] for row in rows.values() for name in required})
        for origin, selected in _fit_sets(references, responses, len(terms)):
            training = [Observation(features[key], value) for key, value in selected]
            program = _fit_linear(terms, training).simplified()
            result["compute_cost"] += len(training)
            source = {"family": estimator["id"], "origin": origin,
                      "settings": [{"crystal": key, "frequency": value} for key, value in selected]}
            if program in seen:
                seen[program]["fit_sources"].append(source)
                continue
            predictions = {key: program.predict(Observation(values, 0.)) for key, values in features.items()}
            result["compute_cost"] += len(features)
            residuals = [{**deepcopy(item), "residual": item["frequency"] - predictions[item["crystal"]]}
                         for item in responses]
            positive = [item["residual"] for item in residuals if item["activated"]]
            low, high = min(positive), max(positive)
            conflicts = [item for item in residuals if not item["activated"] and low <= item["residual"] <= high]
            candidate = {"id": "coefficient_" + str(len(seen)), "family": estimator["id"],
                "program": program.to_dict(), "fit_sources": [source],
                "status": "rejected" if conflicts else "compatible",
                "target_prediction": predictions[number], "positive_residual_span": [low, high],
                "fit_mse": sum((predictions[key] - value) ** 2 for key, value in selected) / len(selected),
                "response_count": len(residuals), "conflicting_responses": conflicts}
            seen[program] = candidate
            result["candidates"].append(candidate)
            if conflicts:
                candidate["reason"] = "negative_response_inside_required_positive_band"
                continue
            negatives = [item["residual"] for item in residuals if not item["activated"]]
            before = max((value for value in negatives if value < low), default=None)
            after = min((value for value in negatives if value > high), default=None)
            candidate["lower_boundary_range"] = {"exclusive_min": before, "inclusive_max": low}
            candidate["upper_boundary_range"] = {"inclusive_min": high, "exclusive_max": after}
            lowers = [low] if before is None else [low, (before + low) / 2]
            uppers = [high] if after is None else [high, (high + after) / 2]
            for lower, upper in product(lowers, uppers):
                result["hypotheses"].append({"id": "response_" + str(len(result["hypotheses"])),
                    "claim": "Activation iff setting minus the fixed fitted program is in "
                             + repr([lower, upper]) + " on each recorded crystal",
                    "candidate_id": candidate["id"], "family": estimator["id"],
                    "program": program.to_dict(), "lower": lower, "upper": upper,
                    "target_prediction": predictions[number],
                    "predictions_by_crystal": {str(key): value for key, value in predictions.items()}})
    return _select_response_experiment(result, state, path)


def _select_response_experiment(result, state, path):
    """Use the same Boolean chooser for old fits and jointly revised models."""
    number = result["target_number"]
    responses = state.get("response_settings", [])
    references = state.get("research_table", {}).get("references", [])
    hypotheses = result["hypotheses"]
    if not hypotheses:
        result["status"] = "no_compatible_response_hypotheses"
        return result
    measured_targets = {row["crystal"] for row in state.get("research_table", {}).get("targets", [])}
    target_grids = {target: grid for target in state.get("target_numbers", [])
        if target not in state.get("completed", []) and target in measured_targets
        if (grid := state.get("control_grids", {}).get(str(target))
            or (state.get("control_grid") if target == number else None))}
    if not target_grids:
        result["status"] = "control_grid_needed"
        return result
    candidates = {item["id"]: item for item in result["candidates"] if "id" in item}

    def experiments_for(crystal, control_grid, restore_frequency=None):
        observed = {item["frequency"] for item in responses if item["crystal"] == crystal}
        settings = {setting for item in hypotheses
            for setting in _setting_proposals(item["predictions_by_crystal"][str(crystal)],
                item["lower"], item["upper"], control_grid) if setting not in observed}
        if restore_frequency is not None:
            # A positive reference can reveal a boundary even when the current
            # finite representatives all predict failure beside that setting.
            step, origin = control_grid["step"], control_grid["origin"]
            for item in responses:
                if item["crystal"] != crystal or not item["activated"]:
                    continue
                position = (item["frequency"] - origin) / step
                settings.update(origin + step * index for index in (ceil(position) - 1, floor(position) + 1)
                    if control_grid["lower"] <= origin + step * index <= control_grid["upper"]
                    and origin + step * index not in observed)
            # Retain unresolved empirical boundaries when a new fitted member
            # moves its support elsewhere. Reuse the existing grid constructor
            # on adjacent opposite readouts, rather than on latent predictions.
            actual = sorted((r for r in responses if r["crystal"] == crystal),
                            key=lambda r: r["frequency"])
            for left, right in zip(actual, actual[1:]):
                if left["activated"] != right["activated"]:
                    settings.update(setting for setting in _setting_proposals(
                        0., left["frequency"], right["frequency"], control_grid)
                        if left["frequency"] < setting < right["frequency"]
                        and setting not in observed)
        experiments = []
        for setting in sorted(settings):
            forecasts = {item["id"]: item["lower"] <=
                setting - item["predictions_by_crystal"][str(crystal)] <= item["upper"]
                for item in hypotheses}
            supported = [item for item in hypotheses if forecasts[item["id"]]]
            if not supported and restore_frequency is None:
                continue
            best = min(supported or hypotheses,
                       key=lambda item: candidates[item["candidate_id"]]["fit_mse"])
            experiment = {"id": ("setting_" if crystal == number and restore_frequency is None else
                "crystal_" + str(crystal) + "_setting_") + str(setting),
                "number": crystal, "prediction": setting,
                "family": best["family"], "candidate_id": best["candidate_id"],
                "outcome": "The installed crystal " + str(crystal) + " activates at setting " + str(setting),
                "procedure": {"path": path, "max_steps": 512},
                "execution_binding": "parent resumes selection stored by reference_response in current episode",
                "predictions": forecasts}
            if restore_frequency is not None:
                experiment["restore_frequency"] = restore_frequency
            experiments.append(experiment)
        return experiments

    experiments = [experiment for target, grid in target_grids.items()
                   for experiment in experiments_for(target, grid)]
    target_experiments = list(experiments)
    for reference in references:
        reference_grid = state.get("control_grids", {}).get(str(reference["crystal"]))
        if reference_grid:
            experiments.extend(experiments_for(reference["crystal"], reference_grid,
                                                reference["frequency"]["value"]))
    result["experiments"] = experiments
    if not experiments:
        result["status"] = "no_untried_supported_setting"
        return result
    if len(hypotheses) > 1:
        selection = select_experiment({"hypotheses": hypotheses, "experiments": experiments})
        result["compute_cost"] += selection["compute_cost"]
    else:
        selection = {"diagnosis": "observational_ambiguity", "selected_experiment": None,
                     "hypotheses": deepcopy(hypotheses), "predicted_separation": 0., "compute_cost": 0}
    if not target_experiments and (selection["selected_experiment"] is None
                                  or selection["predicted_separation"] <= 0):
        result["status"] = "no_untried_supported_setting"
        return result
    if selection["selected_experiment"] is None:
        selection.update(diagnosis="confirmation_trial", selected_experiment=deepcopy(target_experiments[0]))
    result.update(status=selection["diagnosis"], selection=selection,
                  selected_setting=selection["selected_experiment"]["prediction"],
                  selected_number=selection["selected_experiment"]["number"])
    return result


def _nested_supports(estimators):
    """Adapt the old sparse search's support deletion to declared response bases.

    Single terms and one-term deletions give at most twice the basis width in
    proposals. They stay inside an existing family; they neither enlarge the
    scientific language nor narrow that family's possible consequences.
    """
    def support(estimator):
        terms = {Program.from_dict(term).simplified() for term in estimator["basis"]}
        return tuple(sorted((term for term in terms if term.op != "const"), key=repr))

    declared = {support(estimator) for estimator in estimators}
    identifiers = {estimator["id"] for estimator in estimators}
    generated = {}
    for estimator in estimators:
        terms = support(estimator)
        if len(terms) < 2:
            continue
        proposals = [(term,) for term in terms]
        proposals.extend(terms[:index] + terms[index + 1:] for index in range(len(terms)))
        for reduced in proposals:
            if reduced in declared:
                continue
            if reduced in generated:
                if estimator["id"] not in generated[reduced]["derived_from"]:
                    generated[reduced]["derived_from"].append(estimator["id"])
                continue
            serial = 1
            identifier = estimator["id"] + "__support_" + str(serial)
            while identifier in identifiers:
                serial += 1
                identifier = estimator["id"] + "__support_" + str(serial)
            identifiers.add(identifier)
            generated[reduced] = {"id": identifier,
                "basis": [term.to_dict() for term in reduced],
                "derived_from": [estimator["id"]], "origin": "declared_basis_support_pruning"}
    return list(generated.values())


def _append_joint_responses(result, state, estimators, path):
    """Refit selected supports against Boolean evidence, retaining old members."""
    table = state["research_table"]
    measured_targets = {row["crystal"] for row in table.get("targets", [])}
    target_grids = {target: grid for target in state.get("target_numbers", [])
        if target not in state.get("completed", []) and target in measured_targets
        if (grid := state.get("control_grids", {}).get(str(target))
            or (state.get("control_grid") if target == result["target_number"] else None))}
    if not target_grids:
        return result
    from .world_response_constraints import fit_response_constraints

    features = {row["crystal"]: {key: value["value"] for key, value in row["measurements"].items()}
                for row in table["references"] + table["targets"]}
    seeds = {}
    for candidate in result["candidates"]:
        if "program" in candidate:
            for source in candidate["fit_sources"]:
                seeds.setdefault(source["family"], Program.from_dict(candidate["program"]))
    models = []
    result["joint_parameter_search"] = []
    for target, grid in target_grids.items():
        extension = fit_response_constraints(features, state["response_settings"], estimators,
                                            target, grid, seeds)
        result["compute_cost"] += extension["compute_cost"]
        result["joint_parameter_search"].extend(
            {**item, "design_target_number": target} for item in extension["search"])
        models.extend((target, model) for model in extension["models"])
    by_id = {item["id"]: item for item in result["candidates"] if "id" in item}
    by_response = {(Program.from_dict(item["program"]), item["lower"], item["upper"]):
                   by_id[item["candidate_id"]] for item in result["hypotheses"]}
    for index, (design_target, model) in enumerate(models):
        program = model["program"]
        source = {"family": model["family"], **model["fit_provenance"],
                  "design_target_number": design_target, "proposed_setting": model["target_setting"]}
        response_key = (program, model["lower"], model["upper"])
        if response_key in by_response:
            by_response[response_key]["fit_sources"].append(source)
            continue
        predictions = {str(n): program.predict(Observation(values, 0.)) for n, values in features.items()}
        coefficient_id = "joint_coefficient_" + str(index)
        positive = [response for response in state["response_settings"] if response["activated"]]
        fit_mse = sum((row["frequency"] - predictions[str(row["crystal"])]) ** 2
                      for row in positive) / len(positive)
        result["candidates"].append({"id": coefficient_id, "family": model["family"],
            "status": "compatible", "program": program.to_dict(), "fit_mse": fit_mse,
            "target_prediction": predictions[str(result["target_number"])],
            "response_count": len(state["response_settings"]),
            "fit_sources": [source],
            "proposed_number": design_target, "proposed_setting": model["target_setting"],
            "lower": model["lower"], "upper": model["upper"]})
        by_response[response_key] = result["candidates"][-1]
        result["hypotheses"].append({"id": "joint_response_" + str(index),
            "claim": "Activation iff setting minus the revised program lies in its shared interval",
            "candidate_id": coefficient_id, "family": model["family"], "program": program.to_dict(),
            "lower": model["lower"], "upper": model["upper"],
            "target_prediction": predictions[str(result["target_number"])],
            "predictions_by_crystal": predictions})
    if not models:
        return result
    result["assumptions"].append("Reachable target support in added models is a test design condition, not observed success")
    return _select_response_experiment(result, state, path)


def _propose_response_members(state, estimators, path):
    """Retain full explanations while testing their simpler coefficient supports."""
    nested = _nested_supports(estimators)
    members = [*estimators, *nested]
    result = _propose_fitted_responses(state, members, path)
    result["nested_supports"] = deepcopy(nested)
    if result["status"] in {"no_compatible_response_hypotheses", "no_untried_supported_setting"}:
        result = _append_joint_responses(result, state, members, path)
    elif result["status"] != "response_evidence_needed":
        compatible = {source["family"] for candidate in result["candidates"]
                      if candidate.get("status") == "compatible"
                      for source in candidate.get("fit_sources", [])}
        rejected_seeds = [item for item in nested if item["id"] not in compatible]
        if rejected_seeds:
            # Successful control settings only seed coefficient proposals. A
            # rejected numeric fit does not reject its simpler response family,
            # even while a wider interpolation still supplies a point trial.
            result = _append_joint_responses(result, state, rejected_seeds, path)
    by_id = {item["id"]: item for item in nested}
    for candidate in result["candidates"]:
        for source in candidate.get("fit_sources", []):
            if source["family"] in by_id:
                support = by_id[source["family"]]
                source.update(derived_from=deepcopy(support["derived_from"]), support_origin=support["origin"])
    for item in result["candidates"] + result["hypotheses"]:
        if item["family"] in by_id:
            support = by_id[item["family"]]
            item.update(derived_from=deepcopy(support["derived_from"]), origin=support["origin"])
    return result


def propose_response_research(state, estimators, path):
    """Compare executable members, then query their declared shared families."""
    members = _propose_response_members(state, estimators, path)
    from .world_response_consequences import assess_response_consequences, select_response_grid_trial
    result = assess_response_consequences(state, estimators, deepcopy(members))
    compatible = {source["family"] for candidate in members["candidates"]
                  if candidate.get("status") == "compatible"
                  for source in candidate.get("fit_sources", [])}
    supports = [item for item in members["nested_supports"] if item["id"] in compatible]
    if not supports:
        return result
    previous = state.get("response_research", {}).get("conditional_simplification", {})
    selection = result.get("selection") or {}
    if (result["status"] in {"family_target_trial", "shared_premise_test"}
            or selection.get("guaranteed_nonlocal_eliminations", 0) > 0):
        if previous:
            result["conditional_simplification"] = deepcopy(previous)
            result["conditional_simplification"].update(adopted=False, compute_cost=0)
        return result
    width = min(len(item["basis"]) for item in supports)
    supports = [item for item in supports if len(item["basis"]) == width]
    assumed_state = {**state, "response_research": {
        "family_research": previous.get("family_research", {})}}
    conditional = deepcopy(members)
    conditional["compute_cost"] = 0
    conditional = assess_response_consequences(assumed_state, supports, conditional)
    conditional = select_response_grid_trial(assumed_state, supports, conditional)
    result["compute_cost"] += conditional["compute_cost"]
    research = conditional.get("family_research", {})
    proposed = conditional.get("selection") or {}
    experiment = proposed.get("selected_experiment")
    role = research.get("selected_role")
    value = next((item for item in research.get("experiment_values", [])
                  if experiment and item["experiment_id"] == experiment["id"]), {})
    target_trial = conditional["status"] == "family_target_trial"
    useful_test = (role in {"nonlocal_family_test", "conditional_family_test"}
                   and (value.get("guaranteed_nonlocal_eliminations", 0) > 0
                        or value.get("possible_nonlocal_eliminations", 0) > 0))
    useful_test |= (role == "attainable_target_search"
                    and proposed.get("guaranteed_search_reduction", 0) > 0)
    adopted = experiment is not None and (target_trial or useful_test)
    assumption = {"preference": "fewest declared basis terms among compatible generated supports",
                  "supports": deepcopy(supports)}
    result["conditional_simplification"] = {
        "assumption": assumption, "status": conditional["status"],
        "family_research": research, "compute_cost": conditional["compute_cost"], "adopted": adopted}
    if not adopted:
        # The next round may receive incidental readouts at this proposed point;
        # they must not be counted as execution of an unselected child test.
        research["selected_test"] = None
        return result
    diagnosis = "conditional_simpler_target_trial" if target_trial else "conditional_simpler_family_experiment"
    proposed.update(diagnosis=diagnosis, conditional_assumption=deepcopy(assumption),
                    consequence_scope="assumed_simpler_supports")
    result.update(status=diagnosis, selection=proposed,
                  selected_number=experiment["number"], selected_setting=experiment["prediction"])
    result["assumptions"].append("This selected experiment assumes a compatible simpler support; "
        "the complete declared families and their unresolved consequences remain retained")
    result["family_research"].update(selected_role=diagnosis,
        selected_test={"crystal": experiment["number"], "frequency": experiment["prediction"]})
    return result
