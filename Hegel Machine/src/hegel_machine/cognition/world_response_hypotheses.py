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
from .world_experiment_choice import forecast_equivalence_classes, select_experiment


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


def _in_response_basis(program, terms):
    """The coefficient programs emitted by the fitter must still use this basis."""
    if program.op == 'const' or program in terms:
        return True
    if program.op == 'add':
        return all(_in_response_basis(child, terms) for child in program.children)
    if program.op == 'mul':
        return any(child.op == 'const' and _in_response_basis(other, terms)
                   for child, other in (program.children, program.children[::-1]))
    return False


def _residual_band(residuals):
    """Shared connected bands must contain every observed positive residual."""
    positive = [item['residual'] for item in residuals if item['activated']]
    low, high = min(positive), max(positive)
    negatives = [item['residual'] for item in residuals if not item['activated']]
    conflicts = [item for item in residuals
                 if not item['activated'] and low <= item['residual'] <= high]
    before = max((value for value in negatives if value < low), default=None)
    after = min((value for value in negatives if value > high), default=None)
    return low, high, conflicts, before, after


def _response_band_witnesses(low, high, before, after):
    """The existing finite endpoint alternatives within the admissible band."""
    lowers = [low] if before is None else [low, (before + low) / 2]
    uppers = [high] if after is None else [high, (high + after) / 2]
    return product(lowers, uppers)


def revise_pending_response_band(state):
    """Reconsider a pending prediction under revised bands, with fixed laws.

    Reuse complete fixed-program band support. A point outside the mandatory
    positive hull can distinguish a narrow band from a wider compatible band;
    their opposite predictions are hypotheses, not acquired positive evidence.
    """
    test = state['test']
    responses = state['response_settings']
    table = state['research_table']
    features = {row['crystal']: {name: value['value'] for name, value in row['measurements'].items()}
                for row in table['references'] + table['targets']}
    originals = [h for h in state['response_research']['hypotheses']
                 if test['response_forecasts'].get(h['id'])]
    result = {'status': 'no_supported_continuation', 'number': test['number'],
        'requested_setting': test['prediction'], 'hypotheses': [], 'candidates': [],
        'positive_supporters': [],
        'compute_cost': 0, 'evidence': sorted({i for row in responses for i in row['evidence']}),
        'scope': 'Conditional prediction under retained coefficients and a revised shared connected band; '
                 'the requested physical setting remains unconfirmed'}
    required = set().union(*(Program.from_dict(h['program']).required_features() for h in originals))
    result['evidence'] = sorted(set(result['evidence']) | {
        i for row in table['references'] + table['targets'] for name in required
        for evidence in [row['measurements'][name]['evidence']]
        for i in (evidence if isinstance(evidence, list) else [evidence])})
    seen = []
    for original in originals:
        program = Program.from_dict(original['program'])
        if program in seen:
            continue
        seen.append(program)
        sources = [h['id'] for h in originals if h['program'] == original['program']]
        predictions = {n: program.predict(Observation(values, 0.)) for n, values in features.items()}
        residuals = [{**row, 'residual': row['frequency'] - predictions[row['crystal']]}
                     for row in responses]
        result['compute_cost'] += len(features) + len(residuals)
        low, high, conflicts, before, after = _residual_band(residuals)
        requested = test['prediction'] - predictions[test['number']]
        candidate = {'program': original['program'], 'family': original['family'],
            'source_hypotheses': sources, 'positive_residual_span': [low, high],
            'requested_residual': requested, 'conflicting_responses': conflicts,
            'lower_boundary_range': {'exclusive_min': before, 'inclusive_max': low},
            'upper_boundary_range': {'inclusive_min': high, 'exclusive_max': after},
            'status': 'rejected' if conflicts else 'compatible'}
        result['candidates'].append(candidate)
        if (conflicts or (before is not None and requested <= before)
                or (after is not None and requested >= after)):
            continue
        bands = [(low, high)]
        if not low <= requested <= high:
            bands.append((min(low, requested), max(high, requested)))
        for lower, upper in bands:
            identifier = 'revised_band_' + str(len(result['hypotheses']))
            positive = lower <= requested <= upper
            result['hypotheses'].append({'id': identifier,
                'family': original['family'], 'program': original['program'],
                'lower': lower, 'upper': upper, 'requested_prediction': positive,
                'target_prediction': predictions[test['number']], 'source_hypotheses': sources,
                'predictions_by_crystal': {str(n): value for n, value in predictions.items()}})
            if positive:
                result['positive_supporters'].append(identifier)
    if result['positive_supporters']:
        result['status'] = 'supported_continuation'
    return result


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

    def add_program(program, source, fit_mse=None):
        if program in seen:
            if source['origin'] == 'retained_fixed_program' and any(
                    prior['family'] == source['family']
                    and prior['origin'] == 'retained_fixed_program'
                    for prior in seen[program]['fit_sources']):
                return
            if source not in seen[program]["fit_sources"]:
                seen[program]["fit_sources"].append(source)
            return
        predictions = {key: program.predict(Observation(values, 0.)) for key, values in features.items()}
        result["compute_cost"] += len(features)
        residuals = [{**deepcopy(item), "residual": item["frequency"] - predictions[item["crystal"]]}
                     for item in responses]
        positive = [item["residual"] for item in residuals if item["activated"]]
        low, high, conflicts, before, after = _residual_band(residuals)
        candidate = {"id": "coefficient_" + str(len(seen)), "family": source["family"],
            "program": program.to_dict(), "fit_sources": [source],
            "status": "rejected" if conflicts else "compatible",
            "target_prediction": predictions[number], "positive_residual_span": [low, high],
            "fit_mse": fit_mse if fit_mse is not None else sum(value * value for value in positive) / len(positive),
            "response_count": len(residuals), "conflicting_responses": conflicts}
        seen[program] = candidate
        result["candidates"].append(candidate)
        if conflicts:
            candidate["reason"] = "negative_response_inside_required_positive_band"
            return
        candidate["lower_boundary_range"] = {"exclusive_min": before, "inclusive_max": low}
        candidate["upper_boundary_range"] = {"inclusive_min": high, "exclusive_max": after}
        for lower, upper in _response_band_witnesses(low, high, before, after):
            result["hypotheses"].append({"id": "response_" + str(len(result["hypotheses"])),
                "claim": "Activation iff setting minus the fixed fitted program is in "
                         + repr([lower, upper]) + " on each recorded crystal",
                "candidate_id": candidate["id"], "family": source["family"],
                "program": program.to_dict(), "lower": lower, "upper": upper,
                "target_prediction": predictions[number],
                "predictions_by_crystal": {str(key): value for key, value in predictions.items()}})

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
            | {index for row in rows.values() for name in required
               for refs in [row["measurements"][name]["evidence"]]
               for index in (refs if isinstance(refs, list) else [refs])})
        for origin, selected in _fit_sets(references, responses, len(terms)):
            training = [Observation(features[key], value) for key, value in selected]
            program = _fit_linear(terms, training).simplified()
            result["compute_cost"] += len(training)
            source = {"family": estimator["id"], "origin": origin,
                      "settings": [{"crystal": key, "frequency": value} for key, value in selected]}
            add_program(program, source, sum((program.predict(Observation(features[key], 0.))
                - value) ** 2 for key, value in selected) / len(selected))

    # A new positive reference is an additional Boolean observation, not a
    # reason to erase a previously proposed coefficient program. Reconstruct
    # its allowed band from all current evidence instead of retaining its old
    # band verdict. The current declaration still determines its family scope.
    available = {item["id"]: {Program.from_dict(term).simplified() for term in item["basis"]}
                 for item in estimators}
    previous = state.get("response_research", {})
    snapshots = [previous, *state.get("response_research_history", [])]
    if any(row['crystal'] not in features for row in responses):
        snapshots = []
    for snapshot in snapshots:
        saved = (snapshot.get("candidates", []) if "candidates" in snapshot
                 else snapshot.get("hypotheses", []))
        for item in saved:
            if "program" not in item:
                continue
            program = Program.from_dict(item["program"])
            if any(not program.required_features() <= values.keys() for values in features.values()):
                continue
            for source in item.get("fit_sources", [{"family": item["family"]}]):
                if (source["family"] in available
                        and _in_response_basis(program, available[source["family"]])):
                    add_program(program, {"family": source["family"],
                        "origin": "retained_fixed_program",
                        "source_question": snapshot.get("question_id"),
                        "source_response_count": snapshot.get("response_count")})
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
    controls = {row["crystal"]: row["frequency"]["value"] for row in references}
    # A solved target supplies an observed control setting, not an exact
    # function label. Revisit its boundary and restore that paid positive.
    for response in responses:
        if (response["activated"] and response["crystal"] in state.get("completed", [])
                and response["crystal"] in measured_targets):
            controls[response["crystal"]] = response["frequency"]
    for control, restore_setting in controls.items():
        reference_grid = state.get("control_grids", {}).get(str(control))
        if reference_grid:
            experiments.extend(experiments_for(control, reference_grid, restore_setting))
    result["experiments"] = experiments
    if not experiments:
        result["status"] = "no_untried_supported_setting"
        return result
    # Parameter enumerations are not independent evidence. Reuse the panel's
    # forecast classes for choice while retaining every member for real feedback.
    classes = forecast_equivalence_classes(experiments)
    representatives = {group[0] for group in classes}
    if len(classes) > 1:
        selection = select_experiment({
            "hypotheses": [h for h in hypotheses if h['id'] in representatives],
            "experiments": [{**e, 'predictions': {key: value for key, value in e['predictions'].items()
                                                  if key in representatives}} for e in experiments]})
        result["compute_cost"] += selection["compute_cost"]
        selected = selection['selected_experiment']
        if selected is not None:
            selection['selected_experiment'] = deepcopy(next(e for e in experiments if e['id'] == selected['id']))
        selection['hypotheses'] = deepcopy(hypotheses)
    else:
        selection = {"diagnosis": "observational_ambiguity", "selected_experiment": None,
                     "hypotheses": deepcopy(hypotheses), "predicted_separation": 0., "compute_cost": 0}
    selection['forecast_classes'] = classes
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


def _prepared_reference_experiments(state, result):
    """Propose a changed control origin; actual preparation must establish it.

    Only the two neighbours of a paid positive are considered. Crossing the
    public boundary can make them reachable, but may also lose that reference's
    old positive setting: the proposed restoration points require actual tests.
    """
    references = {row['crystal'] for row in state['research_table']['references']}
    observed = {(row['crystal'], row['frequency']) for row in state['response_settings']}
    positive = {(row['crystal'], row['frequency']) for row in state['response_settings']
                if row['activated']}
    templates = {}
    for experiment in result.get('experiments', []):
        number = experiment['number']
        if number in references and 'restore_frequency' in experiment:
            templates.setdefault(number, experiment)
    experiments = []
    for number, template in templates.items():
        grid = state.get('control_grids', {}).get(str(number))
        restore = template['restore_frequency']
        if not grid or (number, restore) not in positive:
            continue
        step, boundary = grid['step'], grid['lower']
        phase = (boundary - grid['origin']) / step
        if abs(phase - round(phase)) < 1e-7:
            continue
        prepared_grid = {**grid, 'origin': boundary}
        position = (restore - boundary) / step
        settings = sorted({boundary + step * index for index in (floor(position), ceil(position))
            if grid['lower'] <= boundary + step * index <= grid['upper']},
            key=lambda setting: (abs(setting - restore), setting))
        for setting in settings:
            if (number, setting) in observed:
                continue
            experiments.append({**deepcopy(template),
                'id': 'prepared_crystal_' + str(number) + '_setting_' + str(setting),
                'prediction': setting,
                'outcome': 'Prepare the public boundary, then test reference '
                    + str(number) + ' at setting ' + str(setting),
                'predictions': {item['id']: item['lower'] <= setting
                    - item['predictions_by_crystal'][str(number)] <= item['upper']
                    for item in result['hypotheses']},
                'control_preparation': {'boundary': boundary, 'grid': prepared_grid,
                                        'restore_settings': list(settings)}})
    return experiments


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
    result.setdefault("joint_parameter_search", [])
    for target, grid in target_grids.items():
        extension = fit_response_constraints(features, state["response_settings"], estimators,
                                            target, grid, seeds)
        result["compute_cost"] += extension["compute_cost"]
        result["joint_parameter_search"].extend(
            {**item, "design_target_number": target} for item in extension["search"])
        models.extend((target, model) for model in extension["models"])
    return _retain_response_models(result, state, models, path)


def _append_fixed_program_responses(result, state, supports, path):
    """Put complete fixed-program band support into the existing finite batch.

    Positive residuals require a closed hull; the nearest negative residuals
    are exclusive outer limits. A point strictly between these limits admits
    a band witness, even when the earlier extrema/midpoint bands omit it.
    Only a finite set of grid endpoints, midpoint and near-prediction settings
    is proposed. The caller freezes these witnesses with the joint revision.
    """
    support_ids = {item['id'] for item in supports}
    table = state['research_table']
    targets = {row['crystal']: {key: value['value'] for key, value in row['measurements'].items()}
               for row in table.get('targets', [])}
    observed = {(row['crystal'], row['frequency']) for row in state['response_settings']}
    models = []
    for candidate in result['candidates']:
        families = sorted({source['family'] for source in candidate.get('fit_sources', [])}
                          & support_ids)
        if not families or candidate.get('status') != 'compatible' or 'lower_boundary_range' not in candidate:
            continue
        program = Program.from_dict(candidate['program'])
        low, high = candidate['positive_residual_span']
        before = candidate['lower_boundary_range']['exclusive_min']
        after = candidate['upper_boundary_range']['exclusive_max']
        for target in state.get('target_numbers', []):
            if target in state.get('completed', []) or target not in targets:
                continue
            grid = state.get('control_grids', {}).get(str(target)) or (
                state.get('control_grid') if target == result['target_number'] else None)
            if not grid:
                continue
            prediction = program.predict(Observation(targets[target], 0.))
            result['compute_cost'] += 1
            origin, step = grid['origin'], grid['step']
            left = ceil((grid['lower'] - origin) / step)
            right = floor((grid['upper'] - origin) / step)
            if before is not None:
                left = max(left, floor((prediction + before - origin) / step) + 1)
            if after is not None:
                right = min(right, ceil((prediction + after - origin) / step) - 1)
            if left > right:
                continue
            position = (prediction - origin) / step
            indices = {left, right, (left + right) // 2,
                       min(right, max(left, floor(position))), min(right, max(left, ceil(position)))}
            for index in sorted(indices):
                setting = origin + step * index
                residual = setting - prediction
                if ((target, setting) in observed or (before is not None and residual <= before)
                        or (after is not None and residual >= after)):
                    continue
                lower, upper = min(low, residual), max(high, residual)
                models.append((target, {'family': families[0], 'program': program,
                    'lower': lower, 'upper': upper, 'target_setting': setting,
                    'fit_provenance': {'origin': 'fixed_program_response_support',
                        'candidate_id': candidate['id'],
                        'response_count': len(state['response_settings']),
                        'lower_boundary_range': deepcopy(candidate['lower_boundary_range']),
                        'upper_boundary_range': deepcopy(candidate['upper_boundary_range'])}}))
    return _retain_response_models(result, state, models, path)


def _retain_response_models(result, state, models, path):
    """Merge joint models, including paid models retained from a preceding decision."""
    table = state['research_table']
    features = {row['crystal']: {key: value['value'] for key, value in row['measurements'].items()}
                for row in table['references'] + table['targets']}
    by_id = {item["id"]: item for item in result["candidates"] if "id" in item}
    by_response = {(Program.from_dict(item["program"]), item["lower"], item["upper"]):
                   by_id[item["candidate_id"]] for item in result["hypotheses"]}
    serial = len(result["candidates"])
    for index, (design_target, model) in enumerate(models, start=serial):
        program = model["program"]
        source = {"family": model["family"], **model["fit_provenance"],
                  "design_target_number": design_target, "proposed_setting": model["target_setting"]}
        response_key = (program, model["lower"], model["upper"])
        if response_key in by_response:
            if source not in by_response[response_key]["fit_sources"]:
                by_response[response_key]["fit_sources"].append(source)
            continue
        while "joint_coefficient_" + str(index) in by_id:
            index += 1
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
        by_id[coefficient_id] = result['candidates'][-1]
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


def _joint_revision_scope(state, supports, number):
    """Identify the existing basis, measurements and target control question."""
    from .world_response_consequences import _families

    table = state['research_table']
    required = set().union(*(Program.from_dict(term).required_features()
                            for family in supports for term in family['basis']))
    rows = table['references'] + table.get('targets', [])
    if any(not required <= row['measurements'].keys() for row in rows):
        return None
    return {'bases': sorted((family['basis'] for family in _families(supports)), key=repr),
        'features': {str(row['crystal']): {key: value['value']
                     for key, value in row['measurements'].items() if key in required}
                     for row in rows},
        'target_number': number,
        'grid': state.get('control_grids', {}).get(str(number)) or state.get('control_grid')}


def _retained_joint_supports(state, supports, number):
    """Reuse a paid conditional batch only while its original question is unchanged."""
    previous = state.get('response_research', {}).get('conditional_simplification', {})
    assumed = previous.get('assumption', {}).get('supports', [])
    available = {item['id']: item['basis'] for item in supports}
    if not assumed or any(available.get(item['id']) != item['basis'] for item in assumed):
        return []
    scope = _joint_revision_scope(state, assumed, number)
    if scope is None or scope != previous.get('joint_revision', {}).get('scope'):
        return []
    return assumed


def _compatible_joint_models(state, prior, members):
    """Check saved coefficient/band hypotheses against every paid response."""
    table = state['research_table']
    features = {row['crystal']: {key: value['value'] for key, value in row['measurements'].items()}
                for row in table['references'] + table.get('targets', [])}
    models = []
    for saved in prior.get('models', []):
        program = Program.from_dict(saved['program'])
        predictions = {number: program.predict(Observation(values, 0.))
                       for number, values in features.items()}
        members['compute_cost'] += len(features) + len(state['response_settings'])
        if all((saved['lower'] <= row['frequency'] - predictions[row['crystal']] <= saved['upper'])
               == row['activated'] for row in state['response_settings']):
            models.append((saved['design_target_number'], {**saved, 'program': program}))
    return models


def _propose_response_members(state, estimators, path):
    """Retain full explanations while testing their simpler coefficient supports."""
    nested = _nested_supports(estimators)
    members = [*estimators, *nested]
    result = _propose_fitted_responses(state, members, path)
    result["nested_supports"] = deepcopy(nested)
    cached = _retained_joint_supports(state, nested, result['target_number'])
    cached_ids = {item['id'] for item in cached}
    if result["status"] in {"no_compatible_response_hypotheses", "no_untried_supported_setting"}:
        pending = [item for item in members if item['id'] not in cached_ids]
        if pending:
            result = _append_joint_responses(result, state, pending, path)
    elif result["status"] != "response_evidence_needed":
        compatible = {source["family"] for candidate in result["candidates"]
                      if candidate.get("status") == "compatible"
                      for source in candidate.get("fit_sources", [])}
        rejected_seeds = [item for item in nested
                          if item["id"] not in compatible and item['id'] not in cached_ids]
        if rejected_seeds:
            # Successful control settings only seed coefficient proposals. A
            # rejected numeric fit does not reject its simpler response family,
            # even while a wider interpolation still supplies a point trial.
            result = _append_joint_responses(result, state, rejected_seeds, path)
    if cached:
        prior = state['response_research']['conditional_simplification']['joint_revision']
        saved = _compatible_joint_models(state, prior, result)
        if saved:
            _retain_response_models(result, state, saved, path)
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


def _informative_reference_renewal(state, members, prior):
    """A completed reference experiment may justify one new finite batch."""
    previous = state.get('response_research', {}).get('conditional_simplification', {})
    research = previous.get('family_research', {})
    test = research.get('selected_test')
    if not test or test['crystal'] == members['target_number']:
        return None
    before = research.get('operational_search', {})
    after = members.get('family_research', {}).get('operational_search', {})
    if (before.get('status') != 'complete' or after.get('status') != 'complete'
            or before.get('grid') != after.get('grid')
            or before.get('target_number') != members['target_number']
            or after.get('target_number') != members['target_number']
            or not before.get('support') or not after.get('support')
            or after['support']['grid_count'] >= before['support']['grid_count']):
        return None
    count = research.get('response_count')
    if count is None:
        return None
    measured = next((row for row in state['response_settings'][count:]
                     if (row['crystal'], row['frequency']) ==
                        (test['crystal'], test['frequency'])), None)
    if measured is None:
        return None
    event = {key: deepcopy(measured[key]) for key in
             ('crystal', 'frequency', 'activated', 'evidence') if key in measured}
    if prior.get('renewal', {}).get('event') == event:
        return None
    return {'event': event, 'support_before_after':
            [before['support']['grid_count'], after['support']['grid_count']]}


def _revise_stalled_support(state, supports, members, previous, path):
    """Keep a finite batch until informative reference research warrants renewal."""
    scope = _joint_revision_scope(state, supports, members['target_number'])
    prior = previous.get('joint_revision', {})
    renewal = None
    if prior.get('scope') == scope:
        models = _compatible_joint_models(state, prior, members)
        if models:
            _retain_response_models(members, state, models, path)
        revision = deepcopy(prior)
        trial = revision.get('experiment')
        observed = {(row['crystal'], row['frequency']) for row in state['response_settings']}
        points = {(target, model['target_setting']) for target, model in models
                  if target == members['target_number']} - observed
        choices = [e for e in members.get('experiments', [])
                   if (e['number'], e['prediction']) in points]
        if trial and (trial['number'], trial['prediction']) in points:
            # Recomparison must not replace an experiment still being executed.
            choices = [e for e in choices
                       if (e['number'], e['prediction']) == (trial['number'], trial['prediction'])]
        if choices:
            selection = select_experiment({'hypotheses': members['hypotheses'], 'experiments': choices})
            members['compute_cost'] += selection['compute_cost']
            revision.update(status='retained_joint_target_prediction',
                            experiment=deepcopy(selection['selected_experiment']))
            return revision, selection
        if not models:
            renewal = _informative_reference_renewal(state, members, prior)
        if renewal is None:
            return revision, None
    if renewal is None and previous.get('family_research', {}).get('no_progress_cycles', 0) < 2:
        return None, None
    # A regular joint search may already have paid for these supports. Freeze
    # its models too; appearing in a candidate pool is not an executed test.
    searched = {item['family'] for item in members.get('joint_parameter_search', [])}
    pending = list(supports) if renewal is not None else [
        family for family in supports if family['id'] not in searched]
    revision = {'scope': scope, 'status': 'no_new_target_prediction', 'compute_cost': 0}
    if renewal is not None:
        revision['renewal'] = renewal
    cost = members['compute_cost']
    _append_fixed_program_responses(members, state, supports, path)
    if pending:
        _append_joint_responses(members, state, pending, path)
    revision['compute_cost'] = members['compute_cost'] - cost
    support_ids = {family['id'] for family in supports}
    by_id = {item['id']: item for item in members['candidates'] if 'id' in item}
    revision['models'] = []
    for hypothesis in members['hypotheses']:
        for source in by_id[hypothesis['candidate_id']].get('fit_sources', []):
            if (source['family'] not in support_ids
                    or source.get('origin') not in {
                        'joint_response_inequality_lp', 'fixed_program_response_support'}):
                continue
            residual = source['proposed_setting'] - hypothesis['predictions_by_crystal'][
                str(source['design_target_number'])]
            if not hypothesis['lower'] <= residual <= hypothesis['upper']:
                continue
            saved = {key: deepcopy(hypothesis[key]) for key in ('program', 'lower', 'upper')}
            saved.update(family=source['family'], target_setting=source['proposed_setting'],
                design_target_number=source['design_target_number'], fit_provenance=deepcopy(source))
            if saved not in revision['models']:
                revision['models'].append(saved)
    number = members['target_number']
    observed = {(row['crystal'], row['frequency']) for row in state['response_settings']}
    batch_points = {(item['design_target_number'], item['target_setting'])
                  for item in revision['models'] if item['design_target_number'] == number}
    trials = [experiment for experiment in members.get('experiments', [])
        if (experiment['number'], experiment['prediction']) in batch_points - observed]
    if not trials:
        return revision, None
    selection = select_experiment({'hypotheses': members['hypotheses'], 'experiments': trials})
    members['compute_cost'] += selection['compute_cost']
    revision['compute_cost'] += selection['compute_cost']
    revision.update(status='new_joint_target_prediction',
                    experiment=deepcopy(selection['selected_experiment']))
    return revision, selection


def propose_response_research(state, estimators, path, paid_checks=None):
    """Compare executable members, then query their declared shared families."""
    from .world_response_consequences import reuse_paid_family_exclusions
    from .world_response_transfer import reuse_response_knowledge

    transferred = reuse_response_knowledge(state, estimators, path)
    if transferred is not None and transferred['frozen_transfer']['status'] == 'selected':
        return transferred

    # Research remains a pure comparison; its caller saves these retained
    # proofs alongside the result. Conditional support checks share this local
    # map, rather than mutating the parent's paid state.
    state = {**state, 'family_history_checks': dict(state.get('family_history_checks', {}))}
    retained = reuse_paid_family_exclusions(state, estimators)
    viable = [e for e in estimators if e['id'] not in retained
              and (paid_checks or {}).get(e['id'], {}).get('status') != 'UNSAT_VERIFIED']
    members = _propose_response_members(state, viable, path)
    from .world_response_consequences import assess_response_consequences, select_response_grid_trial
    result = assess_response_consequences(state, estimators, deepcopy(members), paid_checks=paid_checks)
    if transferred is not None:
        result['frozen_transfer'] = transferred['frozen_transfer']
        result['compute_cost'] += transferred['compute_cost']
    result['family_history_checks'] = state['family_history_checks']
    if paid_checks is not None:
        result['paid_family_checks'] = deepcopy(paid_checks)
        result['paid_comparison_cost'] = sum(check['compute_cost'] for check in paid_checks.values())
        result['compute_cost'] += result['paid_comparison_cost']
    compatible = {source["family"] for candidate in members["candidates"]
                  if candidate.get("status") == "compatible"
                  for source in candidate.get("fit_sources", [])}
    # Exhausting a finite batch does not refute its assumed family or erase
    # that batch's scope. Retain it for the same full-family assessment.
    cached = _retained_joint_supports(state, members['nested_supports'], members['target_number'])
    cached_ids = {item['id'] for item in cached}
    supports = [item for item in members["nested_supports"]
                if item["id"] in compatible or item['id'] in cached_ids]
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
    decision = {key: deepcopy(conditional[key]) for key in
                ('status', 'selection', 'selected_number', 'selected_setting')}
    revision, revised_trial = _revise_stalled_support(state, supports, conditional,
        {**previous, 'family_research': conditional['family_research']}, path)
    conditional.update(decision)
    # Scientific stagnation remains recorded. Complete the finite paid revision
    # batch without refitting it; its trials do not certify a family or restart
    # the failed point-fit exploration allowance.
    joint_trial = revised_trial is not None and conditional['status'] == 'family_research_needed'
    if joint_trial:
        experiment = revised_trial['selected_experiment']
        revised_trial.update(diagnosis='conditional_joint_response_trial',
                             guaranteed_nonlocal_eliminations=0)
        conditional.update(status='conditional_joint_response_trial', selection=revised_trial,
            selected_number=experiment['number'], selected_setting=experiment['prediction'])
        conditional['family_research'].update(selected_role='conditional_joint_response_trial',
            selected_test={'crystal': experiment['number'], 'frequency': experiment['prediction']})
    elif conditional['status'] == 'family_research_needed' and revision is not None:
        # Exhausting the paid joint batch can expose a missing experimental
        # operation. Compare the newly prepared controls, without repeating the
        # old zero-gain controls or treating their forecasts as observations.
        prepared = _prepared_reference_experiments(state, conditional)
        if prepared:
            experiments = conditional['experiments']
            old_search = conditional['family_research'].get('operational_search', {})
            conditional['experiments'] = prepared
            conditional = select_response_grid_trial(assumed_state, supports, conditional)
            conditional['experiments'] = experiments + prepared
            search = conditional['family_research']['operational_search']
            search['experiments'] = old_search.get('experiments', []) + search['experiments']
            search['compute_cost'] += old_search.get('compute_cost', 0)
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
    adopted = experiment is not None and (target_trial or useful_test or joint_trial)
    assumption = {"preference": "fewest declared basis terms among compatible generated supports",
                  "supports": deepcopy(supports)}
    result["conditional_simplification"] = {
        "assumption": assumption, "status": conditional["status"],
        "family_research": research, "compute_cost": conditional["compute_cost"], "adopted": adopted}
    if revision is not None:
        result['conditional_simplification']['joint_revision'] = revision
        for key in ('candidates', 'hypotheses', 'experiments', 'joint_parameter_search'):
            if key in conditional:
                result[key] = deepcopy(conditional[key])
    if not adopted:
        operational = research.get("research_obligation", {}).get("operational_search")
        obligation = result["family_research"].get("research_obligation")
        if operational is not None and obligation is not None:
            obligation["conditional_operational_search"] = {
                "assumption": deepcopy(assumption), **deepcopy(operational)}
        # The next round may receive incidental readouts at this proposed point;
        # they must not be counted as execution of an unselected child test.
        research["selected_test"] = None
        return result
    diagnosis = ('conditional_joint_response_trial' if joint_trial else
        "conditional_simpler_target_trial" if target_trial else "conditional_simpler_family_experiment")
    proposed.update(diagnosis=diagnosis, conditional_assumption=deepcopy(assumption),
                    consequence_scope="assumed_simpler_supports")
    result.update(status=diagnosis, selection=proposed,
                  selected_number=experiment["number"], selected_setting=experiment["prediction"])
    result["assumptions"].append("This selected experiment assumes a compatible simpler support; "
        "the complete declared families and their unresolved consequences remain retained")
    result["family_research"].update(selected_role=diagnosis,
        selected_test={"crystal": experiment["number"], "frequency": experiment["prediction"]})
    return result
