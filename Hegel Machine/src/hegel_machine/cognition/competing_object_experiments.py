"""Turn competing object programs into purchased, prospective experiments.

Only public controls and learned calibrations produce forecasts. The caller
purchases the selected probe with instruments; measured fields then distinguish
a failed calibration from a failed response law. The frozen hypotheses are
evaluated before any refitting on that new evidence.
"""

from itertools import combinations
from math import isfinite
from random import Random

from .language import MissingHistory, Program
from .equivalent_object_programs import canonical_object_program
from .models import Observation, Probe, SearchBudget
from .scale_bindings import bind_measurement_arguments, control_history_depth


def review_paid_program_competition(command_spec, hypotheses, paid_records, budget, *,
        score_scope_ref, score_rows, control_levels, tolerance=1e-7, serial=0, length=4):
    """Reconsider a pending experiment using completed frozen-model readouts.

    The caller binds each score to its frozen program and this same paid scope.
    In particular, a successful incumbent's existing confirmation is consumed
    directly. Only still-compatible programs enter the historical planner;
    this function neither fits models nor asks the laboratory for observations.
    """
    from copy import deepcopy
    from .program_disagreement_probes import propose_program_disagreement_probes

    start, checked, survivors = budget.spent, [], []
    for hypothesis in hypotheses:
        score = hypothesis['score']
        loss = score.get('mse')
        complete = (score.get('returncode') == 0 and not score.get('timeout')
            and not score.get('truncated')
            and score.get('evaluation_rows', score.get('rows')) == score_rows
            and isinstance(loss, (int, float)) and isfinite(loss))
        status = ('supported_on_paid_scope' if loss <= tolerance else 'counterexample') \
            if complete else 'incomplete'
        checked.append({'candidate_id': hypothesis['candidate_id'], 'status': status,
            'mse': loss, 'score_ref': deepcopy(hypothesis['score_ref'])})
        if status == 'supported_on_paid_scope':
            survivors.append(hypothesis)
    result = {'status': 'paid_competition_reviewed', 'hypotheses': checked,
        'survivor_ids': [item['candidate_id'] for item in survivors],
        'score_scope_ref': deepcopy(score_scope_ref), 'score_rows': score_rows,
        'plan': None, 'new_fits': 0, 'new_observations': 0, 'new_scores': 0,
        'compute_cost': 0, 'old_plan_reused': False}
    if any(item['status'] == 'incomplete' for item in checked):
        result['status'] = 'paid_comparison_incomplete'
        return result
    if len(survivors) < 2:
        result['status'] = 'competition_resolved_on_paid_scope' if survivors else 'all_refuted'
        return result
    programs = [Program.from_dict(item['program']) for item in survivors]
    proposals = propose_program_disagreement_probes(command_spec, programs, budget,
        serial=serial, length=length, control_levels=control_levels)
    known = [tuple(row['features'] for row in record.get('rows', [])) for record in paid_records]
    eligible = []
    for proposal in proposals:
        if (proposal['predicted_separation'] <= tolerance or any(
                len(values) != proposal['probe'].cost
                or any(value is None or not isfinite(value) for value in values)
                for values in proposal['predictions'])):
            continue
        plan = {'probe': proposal['probe'], 'hypotheses': [
            {'hypothesis_id': item.get('hypothesis_id', item['candidate_id']),
             'candidate_id': item['candidate_id'], 'program': deepcopy(item['program']),
             'predictions': list(predictions), 'score_ref': deepcopy(item['score_ref'])}
            for item, predictions in zip(survivors, proposal['predictions'], strict=True)],
            'predicted_fields': {}, 'forecast_status': 'predicted_distinction',
            'predicted_separation': proposal['predicted_separation'],
            'proposal_source': 'program_disagreement', 'compute_cost': 0,
            'design': {key: deepcopy(value) for key, value in proposal.items()
                       if key not in {'probe', 'predictions'}}}
        plan = shorten_discriminating_probe(plan, budget, tolerance=tolerance)
        # Complete reset prefixes already have measured answers; the planner
        # may not turn one of them into another purchase after shortening.
        if not any(tuple(plan['probe'].sequence) == sequence[:plan['probe'].cost]
                   for sequence in known):
            eligible.append(plan)
    result.update(compute_cost=budget.spent - start,
        proposed_count=len(proposals), eligible_count=len(eligible),
        status='distinguishing_experiment_available' if eligible else 'no_new_distinguishing_probe')
    if eligible:
        result['plan'] = max(eligible, key=lambda plan: (
            plan['predicted_separation'] / plan['probe'].cost, -plan['probe'].cost))
    return result


def _forecast(program, sequence):
    history, values = [], []
    for point in sequence:
        try:
            value = program.predict(Observation(point, 0., tuple(history)))
            value = value if isfinite(value) else None
        except MissingHistory:
            value = None
        except KeyError as error:
            if error.args[0] != "__target__":
                raise
            value = None
        values.append(value)
        history.append({**point, **({"__target__": value} if value is not None else {})})
    return values


def _probe_pool(spec, serial, length):
    bounds = spec.feature_bounds
    center = {name: (low + high) / 2 for name, (low, high) in bounds.items()}
    # Each sequence starts a fresh experiment. Coordinate interventions expose
    # asymmetric membership effects; random contexts expose joint interactions.
    yield tuple(dict(center) for _ in range(length))
    for name, interval in bounds.items():
        for value in interval:
            yield tuple({**center, name: value if step % 2 else center[name]}
                        for step in range(length)) if length > 1 else ({**center, name: value},)
    rng = Random(7919 + 1009 * serial)
    for _ in range(12):
        yield tuple({name: rng.uniform(*interval) for name, interval in bounds.items()}
                    for _ in range(length))


def plan_competing_object_probe(command_spec, entries, budget, *, binding=None,
                                serial=0, length=4, tolerance=1e-7, methods=None,
                                task_id=None, questions=()):
    """Return a probe and frozen alternative predictions, without observing it.

    ``binding`` maps measured fields to already learned command programs. When
    it cannot forecast every hypothesis, request paired measurements without
    claiming a simulated separation. The result's ``probe`` is a normal Probe;
    all other fields are JSON data suitable for the research action record.
    """
    start = budget.spent
    hypotheses = [{"hypothesis_id": entry.get("hypothesis_id", entry["candidate"].candidate_id),
                   "candidate_id": entry["candidate"].candidate_id,
                   "program": entry["candidate"].program.to_dict(),
                   "training_loss": entry.get("training_loss"),
                   "predictions": None}
                  for entry in entries]
    result = {"probe": None, "hypotheses": hypotheses, "predicted_fields": {},
              "forecast_status": "insufficient_competition", "predicted_separation": None,
              "compute_cost": 0, "proposal_source": None,
              "directed_candidates": 0, "method_candidates": 0,
              "generic_candidates": 0, "generation_cost": 0,
              "method_search_cost": 0, "fresh_synthesis_skipped": False}
    if len(hypotheses) < 2 or not command_spec.feature_bounds:
        return result
    programs = [entry["candidate"].program for entry in entries]
    if len({canonical_object_program(program) for program in programs}) < 2:
        result["forecast_status"] = "algebraically_shared"
        return result
    controls = set(command_spec.feature_bounds)
    required = {name for program in programs for name, _ in program.observed_variables()
                if name != "__target__"} - controls
    binding = {name: program for name, program in (binding or {}).items()
               if name in required and {field for field, _ in program.observed_variables()} <= controls}
    length = min(command_spec.max_probe_length, max(1, length,
                 max(program.minimum_history_depth for program in programs) + 1))
    generic_pool = list(_probe_pool(command_spec, serial, length))
    if required - set(binding):
        result.update(probe=Probe(f"competing_object_calibration:{serial}", generic_pool[-12 + serial % 12]),
                      forecast_status="requires_paired_measurements",
                      proposal_source="calibration",
                      missing_fields=sorted(required - set(binding)))
        return result
    executable = [bind_measurement_arguments(program, binding) for program in programs]
    if questions:
        # A local revision already identifies its unresolved predicates. Solve
        # those questions once, then require actual complete response separation.
        from .condition_questions import plan_condition_questions
        generation = SearchBudget(min(6000, budget.remaining // 8))
        pending, skipped, method_cost, recall = list(questions), [], 0, methods
        while pending:
            plan = plan_condition_questions(command_spec, pending, budget, binding=binding,
                methods=recall, serial=serial, length=length, tolerance=tolerance,
                task_id=task_id, generation_budget=generation, response_programs=programs)
            method_cost += plan.get("method_search_cost", 0)
            plan["hypotheses"] = hypotheses
            if plan["probe"] is None or plan["forecast_status"] != "predicted_distinction":
                break
            sequence, predictions = plan["probe"].sequence, []
            for program, hypothesis in zip(executable, hypotheses):
                if not budget.spend(len(sequence)):
                    plan.update(probe=None, forecast_status="compute_exhausted")
                    break
                prediction = _forecast(program, sequence)
                predictions.append(prediction)
                hypothesis["predictions"] = prediction
            if plan["probe"] is None:
                break
            common = [row for row in zip(*predictions) if all(value is not None for value in row)]
            separation = sum((max(row) - min(row)) ** 2 for row in common) / len(common) if common else 0.
            plan["predicted_separation"] = separation
            if separation > tolerance:
                for name, program in binding.items():
                    if not budget.spend(len(sequence)):
                        plan.update(probe=None, forecast_status="compute_exhausted")
                        break
                    plan["predicted_fields"][name] = _forecast(program, sequence)
                break
            question_id = plan["question"]["question_id"]
            if plan["proposal_source"] == "learned_method":
                # A recalled setup can activate the condition while leaving
                # every response zero. Spend the remaining allowance on a new
                # setup for this same question before abandoning it.
                pending.sort(key=lambda question: question["question_id"] != question_id)
                recall = None
                continue
            skipped.append(question_id)
            pending = [question for question in pending if question["question_id"] != question_id]
            recall = methods
            plan.update(probe=None, forecast_status="no_predicted_distinction")
        plan.update(generation_cost=generation.spent, method_search_cost=method_cost,
                    compute_cost=budget.spent - start)
        if skipped:
            plan["questions_without_response_distinction"] = skipped
        return plan
    if any(control_history_depth(program) for program in binding.values()):
        length = min(command_spec.max_probe_length,
                     max(length, *(control_history_depth(program) + 2 for program in executable)))
        generic_pool = list(_probe_pool(command_spec, serial, length))
    from .program_disagreement_probes import propose_program_disagreement_probes
    generation = SearchBudget(min(6000, budget.remaining // 8))
    learned = (methods.propose(command_spec, programs, generation, binding=binding,
        serial=serial, length=length, tolerance=tolerance, task_id=task_id)
        if methods is not None else [])
    learned = [item for item in learned if item["predicted_separation"] > tolerance]
    result.update(method_candidates=len(learned), method_search_cost=generation.spent,
                  fresh_synthesis_skipped=bool(learned))
    method_scope = methods.last_proposal if methods is not None else {}
    if method_scope.get("status") == "blocked_calibration":
        budget.spend(generation.spent)
        result.update(probe=Probe(f"competing_object_recalibration:{serial}", generic_pool[-12 + serial % 12]),
                      forecast_status="requires_paired_measurements", proposal_source="calibration",
                      reason="previous_measurements_refuted_current_calibration",
                      method_scope=method_scope, generation_cost=generation.spent,
                      compute_cost=budget.spent - start)
        return result
    if method_scope.get("blocked_scopes"):
        result["method_scope"] = method_scope
    directed = ([] if learned else propose_program_disagreement_probes(
        command_spec, programs, generation, binding=binding, serial=serial, length=length))
    budget.spend(generation.spent)
    result.update(directed_candidates=len(directed), generation_cost=generation.spent)
    pool = [(item["probe"].sequence, item["predictions"], source,
             {key: value for key, value in item.items() if key not in ("probe", "predictions")})
            for source, proposals in (("learned_method", learned), ("program_disagreement", directed))
            for item in proposals]
    pool.extend((sequence, None, "generic", {}) for sequence in generic_pool)
    best = None
    for sequence, predictions, source, proposal in pool:
        if predictions is None:
            if not budget.spend(len(sequence) * len(executable)):
                break
            predictions = [_forecast(program, sequence) for program in executable]
            result["generic_candidates"] += 1
        common = [values for values in zip(*predictions) if all(value is not None for value in values)]
        separation = sum((max(values) - min(values)) ** 2 for values in common) / len(common) if common else 0.
        if best is None or separation > best[0]:
            best = separation, sequence, predictions, source, proposal
    result["compute_cost"] = budget.spent - start
    if best is None:
        result["forecast_status"] = "compute_exhausted"
        return result
    separation, sequence, predictions, source, proposal = best
    if separation <= tolerance:
        result.update(forecast_status="no_predicted_distinction", predicted_separation=separation)
        return result
    # Forecasting the instrument readings is itself paid computation. If it
    # cannot finish, the existing response forecasts still request real fields.
    fields = {}
    for name, program in binding.items():
        if not budget.spend(len(sequence)):
            break
        fields[name] = _forecast(program, sequence)
    for hypothesis, prediction in zip(hypotheses, predictions):
        hypothesis["predictions"] = prediction
    result.update(probe=Probe(f"competing_object_prediction:{serial}", sequence),
                  predicted_fields=fields, forecast_status="predicted_distinction",
                  proposal_source=source, proposal=proposal,
                  predicted_separation=separation, compute_cost=budget.spent - start)
    return result


def shorten_discriminating_probe(plan, budget, *, tolerance=1e-7):
    """Buy a prefix that keeps every forecast distinction of a relation probe.

    Only the already frozen response predictions choose the length. Each pair
    distinguishable on the full trajectory must be at least as far apart in
    prefix mean squared prediction distance. Preparation history is retained;
    calibration-only experiments and unresolved forecasts keep their length.
    """
    probe = plan.get("probe")
    hypotheses = plan.get("hypotheses", [])
    if (probe is None or plan.get("forecast_status") != "predicted_distinction"
            or len(hypotheses) < 2):
        return plan
    count = probe.cost
    predictions = [item.get("predictions") for item in hypotheses]
    if any(values is None or len(values) != count for values in predictions):
        return plan
    start = budget.spent
    pairs = []
    for left, right in combinations(predictions, 2):
        if not budget.spend(count):
            return {**plan, "compute_cost": plan.get("compute_cost", 0) + budget.spent - start}
        total, observed, prefixes = 0., 0, []
        for a, b in zip(left, right):
            if a is not None and b is not None:
                total += (a - b) ** 2
                observed += 1
            prefixes.append(total / observed if observed else 0.)
        if prefixes[-1] > tolerance:
            pairs.append(prefixes)
    result = {**plan, "compute_cost": plan.get("compute_cost", 0) + budget.spent - start}
    if not pairs:
        return result
    minimum = max(Program.from_dict(item["program"]).minimum_history_depth + 1 for item in hypotheses)
    length = next((size for size in range(max(1, minimum), count)
                   if all(pair[size - 1] >= pair[-1] for pair in pairs)), count)
    if length == count:
        return result
    result.update(probe=Probe(probe.probe_id, probe.sequence[:length]),
        hypotheses=[{**item, "predictions": item["predictions"][:length]} for item in hypotheses],
        predicted_fields={name: values[:length] for name, values in plan.get("predicted_fields", {}).items()},
        prefix_design={"planned_rows": count, "purchased_rows": length,
                       "retained_distinctions": len(pairs)})
    if "question_hypotheses" in plan:
        result["question_hypotheses"] = [{**item, "predictions": item["predictions"][:length]}
                                         for item in plan["question_hypotheses"]]
    # A truncated cycle has not bought every advertised phase of the original.
    result.pop("predicted_question_phases", None)
    common = [row for row in zip(*(item["predictions"] for item in result["hypotheses"]))
              if all(value is not None for value in row)]
    result["predicted_separation"] = (sum((max(row) - min(row)) ** 2 for row in common) / len(common)
                                      if common else 0.)
    return result


def assess_competing_object_probe(plan, paired_rows, budget, *, tolerance=1e-7,
                                  repeated_rows=None):
    """Compare the frozen forecasts and laws against new paired measurements.

    A calibration error invalidates its forecast, not a law that remains right
    on the actually measured fields. Conversely, a law that fails on those
    fields receives its own counterexample even when calibration also failed.
    An independently repeated response can distinguish random observation
    error from a reproducible failure of a frozen law. No parameters or object
    assignments are changed by this function.
    """
    pairs, start = list(paired_rows), budget.spent
    result = {"status": "no_observations", "hypotheses": [], "field_losses": {},
              "calibration_status": "not_forecast", "observation_rows": len(pairs),
              "compute_cost": 0}
    if not pairs:
        return result
    repeated = list(repeated_rows) if repeated_rows is not None else None
    measured_rows = [measured for _, measured in pairs]
    for name, predictions in plan.get("predicted_fields", {}).items():
        values = [(value, measured.features[name])
                  for value, (_, measured) in zip(predictions, pairs)
                  if value is not None and name in measured.features]
        if values and budget.spend(len(values)):
            result["field_losses"][name] = sum((left - right) ** 2 for left, right in values) / len(values)
    if result["field_losses"]:
        result["calibration_status"] = ("counterexample" if any(
            value > tolerance for value in result["field_losses"].values()) else "supported_on_probe")
    for hypothesis in plan["hypotheses"]:
        item = {"hypothesis_id": hypothesis["hypothesis_id"], "candidate_id": hypothesis["candidate_id"],
                "status": "unresolved", "measured_loss": None, "forecast_loss": None, "rows": 0}
        if repeated is not None:
            item.update(deterministic_status="unresolved", deterministic_counterexamples=[],
                        noise_replication=None)
        program = Program.from_dict(hypothesis["program"])
        if not budget.spend(len(pairs)):
            result["hypotheses"].append(item)
            continue
        predictions = hypothesis.get("predictions") or [None] * len(pairs)
        measured_errors, forecast_errors, counterexamples = [], [], []
        for index, (forecast, (_, measured)) in enumerate(zip(predictions, pairs)):
            if forecast is not None:
                forecast_errors.append((forecast - measured.target) ** 2)
            try:
                prediction = program.predict(measured)
            except MissingHistory:
                continue
            error = (prediction - measured.target) ** 2
            measured_errors.append(error)
            if error > tolerance:
                counterexamples.append({"position": index, "prediction": prediction,
                                        "target": measured.target, "squared_error": error})
        if measured_errors:
            value = sum(measured_errors) / len(measured_errors)
            item.update(status="supported_on_probe" if value <= tolerance else "counterexample",
                        measured_loss=value, rows=len(measured_errors), counterexamples=counterexamples)
        if forecast_errors:
            item["forecast_loss"] = sum(forecast_errors) / len(forecast_errors)
        if repeated is not None:
            from .noise_replication import assess_noise_replication

            item["deterministic_status"] = item["status"]
            item["deterministic_counterexamples"] = counterexamples
            replication = assess_noise_replication(program, measured_rows, repeated, budget,
                minimum_improvement=tolerance)
            item["noise_replication"] = replication
            supported = ((replication["noise_supported"]
                          or replication["reason"] == "no_material_repeatable_error")
                and max(replication["original_model_loss"], replication["repeated_model_loss"])
                    <= replication["repeatable_loss_limit"])
            item["status"] = ("counterexample" if replication["repeatable_error"]
                              else "supported_on_probe" if supported else "unresolved")
            if item["status"] != "counterexample":
                item["counterexamples"] = []
        result["hypotheses"].append(item)
    statuses = [item["status"] for item in result["hypotheses"]]
    result.update(status=("distinguished" if "counterexample" in statuses and "supported_on_probe" in statuses
                          else "all_refuted" if statuses and all(value == "counterexample" for value in statuses)
                          else "unresolved"), compute_cost=budget.spent - start)
    return result
