"""Retain learned branch questions when a short transfer fit becomes constant."""

import json
from math import isfinite, sqrt

from .competing_object_experiments import _forecast, plan_competing_object_probe
from .equivalent_object_programs import canonical_object_program
from .language import MissingHistory, Program
from .meta_prior_research import _sources
from .models import Candidate, Observation, SearchBudget
from .program_condition_sets import _active_paths
from .scale_bindings import bind_measurement_arguments


def _conditional(program):
    if program.op == "neighbor_sum":
        return True
    if program.op in {"if", "lt", "ge"}:
        condition = program.children[0] if program.op == "if" else program
        if condition.observed_variables():
            return True
        # An available-history guard chooses a numerical initialization path;
        # it does not nominate an untested input region. Its branches can still
        # contain actual input predicates and must be inspected.
        children = program.children[1:] if program.op == "if" else ()
        return any(_conditional(child) for child in children)
    return any(_conditional(child) for child in program.children)


def _conjunction(path):
    conditions = []
    for expression, threshold, lower, inclusive in sorted(dict.fromkeys(path), key=repr):
        constant = Program("const", value=threshold)
        if lower:
            condition = Program("ge" if inclusive else "lt", children=(expression, constant)
                                if inclusive else (constant, expression))
        else:
            condition = Program("ge" if inclusive else "lt", children=(constant, expression)
                                if inclusive else (expression, constant))
        conditions.append(condition)
    result = conditions[0]
    for condition in conditions[1:]:
        result = Program("and", children=(result, condition))
    return result


def _evidence_rows(paid_rows, paired_rows):
    rows = list(paid_rows)
    for command, measured in paired_rows:
        length = min(len(command.history), len(measured.history))
        history = tuple({**commands, **sensors} for commands, sensors in
                        zip(command.history[-length:], measured.history[-length:])) if length else ()
        combined = Observation({**command.features, **measured.features}, measured.target, history)
        if measured in rows:
            rows[rows.index(measured)] = combined
        elif combined not in rows:
            rows.append(combined)
    return rows


class TransferConditionResearch:
    """One task's callable questions and computations on already paid rows."""

    def __init__(self, retained_calls, *, prevalidated_call_ids=()):
        self.retained_calls = tuple(retained_calls)
        self.calls = [(index, program, program.expanded()) for index, program in enumerate(self.retained_calls)]
        self.calls = [item for item in self.calls if _conditional(item[2])]
        self.prevalidated_call_ids = set(prevalidated_call_ids)
        self.compiled, self.predictions, self.phases = {}, {}, {}
        self.rows = []
        self.counts = dict.fromkeys(("new_prediction_evaluations", "reused_prediction_evaluations",
                                   "new_phase_evaluations", "reused_phase_evaluations"), 0)

    def plan(self, command_spec, trial, budget, **kwargs):
        return plan_transfer_condition_probe(command_spec, trial, self.retained_calls,
                                             budget, _research=self, **kwargs)


    def sync_rows(self, rows):
        # Paired observations may enrich actual fields and histories.
        if len(rows) < len(self.rows) or self.rows != rows[:len(self.rows)]:
            self.predictions.clear()
            self.phases.clear()
        self.rows = rows

    def predict(self, program, position, budget, units=1, history_index=None):
        key = program, position, history_index
        if key in self.predictions:
            self.counts["reused_prediction_evaluations"] += 1
            return self.predictions[key]
        if not budget.spend(units):
            return None, "compute_exhausted"
        self.counts["new_prediction_evaluations"] += 1
        row = self.rows[position]
        if history_index is not None:
            row = Observation(row.history[history_index], 0., row.history[:history_index])
        try:
            value = program.predict(row)
            result = (value, "observed") if isfinite(value) else (None, "nonfinite")
        except (MissingHistory, KeyError):
            result = None, "missing_observation"
        self.predictions[key] = result
        return result

    def condition(self, program, position, budget, history_index=None):
        """Share evaluated comparisons and scalar inputs on the same paid row."""
        if program.op not in {"and", "or", "not", "lt", "ge"}:
            return self.predict(program, position, budget, program.size, history_index)
        key = program, position, history_index
        if key in self.predictions:
            self.counts["reused_prediction_evaluations"] += 1
            return self.predictions[key]
        if not budget.spend(1):
            return None, "compute_exhausted"
        self.counts["new_prediction_evaluations"] += 1
        values = []
        for child in program.children:
            value, status = self.condition(child, position, budget, history_index)
            if status != "observed":
                if status != "compute_exhausted":
                    self.predictions[key] = value, status
                return value, status
            values.append(value)
        # Program evaluates both sides of and/or; preserve missing-history
        # behavior even when the first side already determines a Boolean value.
        if program.op == "not":
            value = not values[0]
        elif program.op == "and":
            value = bool(values[0]) and bool(values[1])
        elif program.op == "or":
            value = bool(values[0]) or bool(values[1])
        elif program.op == "lt":
            value = values[0] < values[1]
        else:
            value = values[0] >= values[1]
        self.predictions[key] = result = float(value), "observed"
        return result

    def phase(self, predicate, depth, position, budget):
        key = predicate, depth, position
        if key in self.phases:
            self.counts["reused_phase_evaluations"] += 1
            return self.phases[key]
        self.counts["new_phase_evaluations"] += 1
        row = self.rows[position]
        if len(row.history) < depth:
            self.phases[key] = None, "insufficient_history"
            return self.phases[key]
        active, status = self.condition(predicate, position, budget)
        if active is None:
            return None, status
        phase = "active" if active and not depth else None
        if depth:
            previous = []
            for history_index in range(len(row.history) - depth, len(row.history)):
                value, status = self.condition(predicate, position, budget, history_index)
                if status == "compute_exhausted":
                    return None, status
                previous.append(value)
            if active:
                phase = ("enter" if any(value == 0. for value in previous) else
                         "maintain" if all(value is not None and value for value in previous) else None)
            elif any(value is not None and value for value in previous):
                phase = "exit"
        self.phases[key] = phase, "observed"
        return self.phases[key]


def _condition_program_key(program):
    return program.expanded().simplified().to_dict()


def _retained_call(retained_calls, call_index):
    """Resolve a call by its original index after conditional filtering."""
    if retained_calls and isinstance(retained_calls[0], tuple):
        indexed = {item[0]: item[1] for item in retained_calls}
        return indexed[call_index]
    return retained_calls[call_index]


def record_condition_coverages(scope, retained_calls, coverages, *, task_id=None):
    """Persist condition coverage established by paired paid rows."""
    def key(source_ids, program, predicate, history_depth):
        return (tuple(source_ids), json.dumps(program, sort_keys=True),
                json.dumps(predicate, sort_keys=True), history_depth)

    existing = {key(item.get("source_ids", ()), item.get("program"),
                    item.get("predicate"), item.get("history_depth"))
                for item in coverages}
    for item in scope.get("calls", ()):
        if item.get("status") != "covered_by_paid_evidence":
            continue
        call = _retained_call(retained_calls, item["call_index"])
        source_ids = tuple(sorted(_sources(call)))
        program = _condition_program_key(call)
        for condition in item.get("conditions", ()):
            if not condition.get("covered"):
                continue
            record = {"source_ids": list(source_ids), "program": program,
                      "predicate": condition["predicate"],
                      "history_depth": item.get("history_depth", 0),
                      "task_id": task_id,
                      "paid_witnesses": list(condition.get("paid_witnesses", ())),
                      "coverage_source": "paid_paired_rows"}
            record_key = key(source_ids, program, record["predicate"], record["history_depth"])
            if record_key not in existing:
                coverages.append(record)
                existing.add(record_key)


def covered_call_ids(retained_calls, coverages):
    """Return call indexes whose every executable branch is already covered."""
    result = set()
    for index, call in enumerate(retained_calls):
        expanded = call.expanded()
        if not _conditional(expanded):
            continue
        paths, complete = _active_paths(expanded)
        if not complete:
            continue
        predicates = list(dict.fromkeys(_conjunction(path) for path in paths if path))
        source_ids = tuple(sorted(_sources(call)))
        program = _condition_program_key(call)
        depth = expanded.history_depth
        if predicates and all(any(tuple(sorted(receipt.get("source_ids", ()))) == source_ids
                and receipt.get("program") == program
                and receipt.get("predicate") == predicate.to_dict()
                and receipt.get("history_depth", 0) == depth for receipt in coverages)
                for predicate in predicates):
            result.add(index)
    return result


def plan_transfer_condition_probe(command_spec, trial, retained_calls, budget, *,
                                  paid_rows=(), paired_rows=(), binding=None, methods=None,
                                  serial=0, length=4, tolerance=1e-7, task_id=None, _research=None,
                                  prior_coverages=()):
    """Ask about untested learned conditions; return no simulated support.

    Conditions are inspected in the original current-field-bound calls, even
    when ``trial`` has lost those calls. Paid activation plus a correct trial
    prediction covers a condition; disagreement of the original call with
    those same measurements is retained for the caller's revision decision.
    ``paired_rows`` supplies actual commands and readings, not calibration
    forecasts. The returned plan uses ordinary frozen full-program hypotheses.
    """
    start = budget.spent
    research = _research if _research is not None else TransferConditionResearch(retained_calls)
    calls = research.calls
    calls_by_index = {index: program for index, program, _ in calls}
    before = dict(research.counts)
    scope = {"has_conditional_calls": bool(calls), "calls": [], "evidence_rows": 0,
             "covered_call_ids": [], "uncovered_call_ids": [], "all_conditions_covered": False,
             "trial_counterexamples": []}
    result = {"probe": None, "hypotheses": [], "predicted_fields": {},
              "forecast_status": "no_condition", "predicted_separation": None,
              "proposal_source": None, "compute_cost": 0, "transfer_scope": scope}
    shared_questions = {}

    def finish(plan):
        scope["sharing"] = {"condition_questions": len(shared_questions), "callable_hypotheses": len(calls),
            **{name: count - before[name] for name, count in research.counts.items()}}
        plan.update(transfer_scope=scope, compute_cost=budget.spent - start)
        return plan

    if not calls:
        return finish(result)
    entries = [{"hypothesis_id": "transfer_trial", "candidate": Candidate(
        "transfer_trial", trial, "short_transfer_fit")}, *[
        {"hypothesis_id": f"transfer_call_{index}", "candidate": Candidate(
            f"transfer_call_{index}", program, "retained_call_condition")}
        for index, program, _ in calls]]
    result["hypotheses"] = [{"hypothesis_id": entry["hypothesis_id"],
        "candidate_id": entry["candidate"].candidate_id,
        "program": entry["candidate"].program.to_dict(), "predictions": None}
        for entry in entries]
    rows = _evidence_rows(paid_rows, paired_rows)
    research.sync_rows(rows)
    scope["evidence_rows"] = len(rows)

    trial_values = []
    for position in range(len(rows)):
        value, status = research.predict(trial, position, budget)
        if status == "compute_exhausted":
            result.update(forecast_status=status, compute_cost=budget.spent - start)
            return finish(result)
        trial_values.append(value)

    for index, program, expanded in calls:
        if index in research.prevalidated_call_ids:
            # Prior coverage saves a new condition experiment, not the check
            # against responses already bought on this task. Short validation
            # can miss a branch that was active in an earlier paid batch.
            counterexamples = [{"hypothesis_id": f"transfer_call_{index}",
                "position": position, "prediction": fitted, "target": row.target,
                "squared_error": (fitted - row.target) ** 2}
                for position, (row, fitted) in enumerate(zip(rows, trial_values))
                if fitted is not None and (fitted - row.target) ** 2 > tolerance]
            scope["trial_counterexamples"].extend(counterexamples)
            scope["calls"].append({"hypothesis_id": f"transfer_call_{index}",
                "call_index": index, "history_depth": expanded.history_depth,
                "status": "trial_refuted_on_paid_conditions" if counterexamples
                    else "covered_by_paid_evidence",
                "coverage_source": "prior_supported_condition", "conditions": []})
            continue
        if index not in research.compiled:
            if not budget.spend(expanded.size):
                result.update(forecast_status="compute_exhausted")
                return finish(result)
            paths, complete = _active_paths(expanded)
            predicates = list(dict.fromkeys(_conjunction(path) for path in paths if path))
            research.compiled[index] = predicates, complete
        predicates, complete = research.compiled[index]
        covered, witnesses = [False] * len(predicates), [[] for _ in predicates]
        history_depth = expanded.history_depth
        phases = ("enter", "maintain", "exit") if history_depth else ("active",)
        phase_witnesses = [{phase: [] for phase in phases} for _ in predicates]
        active_positions, original_errors, active_original_errors = set(), [], []
        active_trial_errors, original_counterexamples = [], []
        for position, (row, fitted) in enumerate(zip(rows, trial_values)):
            value, status = research.predict(program, position, budget)
            if status == "compute_exhausted":
                result.update(forecast_status=status, compute_cost=budget.spent - start)
                return finish(result)
            if value is None:
                continue
            error = (value - row.target) ** 2
            original_errors.append(error)
            # Static zero responses do not excite a multiplicative branch.
            # Temporal steady/exit phases may correctly have zero response.
            if not history_depth and abs(value) <= sqrt(tolerance):
                continue
            if history_depth and len(row.history) < history_depth:
                continue
            active = False
            for condition_index, predicate in enumerate(predicates):
                phase, status = research.phase(predicate, history_depth, position, budget)
                if status == "compute_exhausted":
                    result.update(forecast_status=status)
                    return finish(result)
                if phase is None:
                    continue
                active = True
                if fitted is not None:
                    fitted_error = (fitted - row.target) ** 2
                    if fitted_error <= tolerance:
                        witnesses[condition_index].append(position)
                        phase_witnesses[condition_index][phase].append(position)
                        covered[condition_index] = all(phase_witnesses[condition_index].values())
                    else:
                        scope["trial_counterexamples"].append({"hypothesis_id": f"transfer_call_{index}",
                            "position": position, "prediction": fitted, "target": row.target,
                            "squared_error": fitted_error})
            if active:
                active_positions.add(position)
                active_original_errors.append(error)
                if fitted is not None:
                    active_trial_errors.append((fitted - row.target) ** 2)
                if error > tolerance:
                    original_counterexamples.append({"position": position, "prediction": value,
                        "target": row.target, "squared_error": error})
        call_id = f"transfer_call_{index}"
        covered_call = bool(predicates) and complete and all(covered)
        item = {"hypothesis_id": call_id, "call_index": index,
            "status": "covered_by_paid_evidence" if covered_call else "uncovered_condition",
            "scope_complete": complete, "history_depth": history_depth,
            "active_paid_rows": len(active_positions),
            "active_positions": sorted(active_positions),
            "original_call_loss": sum(original_errors) / len(original_errors) if original_errors else None,
            "active_original_call_loss": sum(active_original_errors) / len(active_original_errors)
                if active_original_errors else None,
            "active_trial_loss": sum(active_trial_errors) / len(active_trial_errors)
                if active_trial_errors else None,
            "original_call_counterexamples": original_counterexamples,
            "conditions": [{"predicate": predicate.to_dict(), "covered": is_covered,
                "paid_witnesses": evidence, "required_phases": list(phases), "phase_witnesses": phase_rows}
                for predicate, is_covered, evidence, phase_rows in zip(predicates, covered, witnesses, phase_witnesses)]}
        scope["calls"].append(item)
        scope["covered_call_ids" if covered_call else "uncovered_call_ids"].append(call_id)
        for condition in item["conditions"]:
            predicate = Program.from_dict(condition["predicate"])
            key = predicate, history_depth
            question = shared_questions.setdefault(key, {"question_id": f"condition_{len(shared_questions)}",
                "predicate": predicate, "history_depth": history_depth,
                "dependent_hypothesis_ids": [], "uncovered_phases": []})
            if call_id not in question["dependent_hypothesis_ids"]:
                question["dependent_hypothesis_ids"].append(call_id)
            for phase, evidence in condition["phase_witnesses"].items():
                if not evidence and phase not in question["uncovered_phases"]:
                    question["uncovered_phases"].append(phase)

    # A later task can keep the source call and predicate while changing only
    # measured field names. Reuse an exact paid coverage receipt in that case;
    # no new probe is justified by a naming change alone.
    for item in scope["calls"]:
        if item.get("coverage_source") == "prior_supported_condition":
            continue
        call = calls_by_index[item["call_index"]]
        source_ids = tuple(sorted(_sources(call)))
        program = _condition_program_key(call)
        for condition in item["conditions"]:
            if condition.get("covered"):
                continue
            matching = next((receipt for receipt in prior_coverages
                if tuple(sorted(receipt.get("source_ids", ()))) == source_ids
                and receipt.get("program") == program
                and receipt.get("predicate") == condition.get("predicate")
                and receipt.get("history_depth", 0) == item.get("history_depth", 0)), None)
            if matching is not None:
                condition["covered"] = True
                condition["coverage_source"] = "prior_paid_condition"
                condition["paid_witnesses"] = list(matching.get("paid_witnesses", ()))
        item["status"] = ("covered_by_paid_evidence" if item.get("scope_complete") and item["conditions"]
                           and all(condition.get("covered") for condition in item["conditions"])
                           else "uncovered_condition")
    scope["covered_call_ids"] = [item["hypothesis_id"] for item in scope["calls"]
                                  if item["status"] == "covered_by_paid_evidence"]
    scope["uncovered_call_ids"] = [item["hypothesis_id"] for item in scope["calls"]
                                    if item["status"] != "covered_by_paid_evidence"]
    scope["questions"] = [{**question, "predicate": question["predicate"].to_dict()}
                          for question in shared_questions.values()]
    if scope["trial_counterexamples"]:
        result.update(forecast_status="trial_refuted_on_paid_conditions", compute_cost=budget.spent - start)
        return finish(result)
    if not scope["uncovered_call_ids"]:
        scope["all_conditions_covered"] = True
        result.update(forecast_status="covered_by_paid_evidence", compute_cost=budget.spent - start)
        return finish(result)
    # An unchanged retained call can agree with the trial everywhere while its
    # activation region remains unmeasured. A constant response consistent with
    # the actual paid prefix is then a real alternative, not a supplied answer.
    if rows and all(isfinite(row.target) for row in rows):
        constant_value = sum(row.target for row in rows) / len(rows)
        errors = [(row.target - constant_value) ** 2 for row in rows]
        baseline = Program("const", value=constant_value)
        if max(errors) <= tolerance and not any(canonical_object_program(
                entry["candidate"].program) == baseline for entry in entries):
            if not budget.spend(len(rows)):
                result.update(forecast_status="compute_exhausted", compute_cost=budget.spent - start)
                return finish(result)
            candidate = Candidate("transfer_observed_constant", baseline, "observed_response_baseline")
            entry = {"hypothesis_id": candidate.candidate_id, "candidate": candidate}
            entries.append(entry)
            result["hypotheses"].append({"hypothesis_id": candidate.candidate_id,
                "candidate_id": candidate.candidate_id, "program": baseline.to_dict(), "predictions": None})
            scope["observational_baseline"] = {"hypothesis_id": candidate.candidate_id,
                "value": constant_value, "paid_rows": len(rows),
                "measured_loss": sum(errors) / len(errors)}
    from .condition_questions import plan_condition_questions
    original_hypotheses = result["hypotheses"]
    pending = [question for question in shared_questions.values() if question["uncovered_phases"]]
    controls = set(command_spec.feature_bounds)
    current_binding = {name: value for name, value in (binding or {}).items()
                       if {field for field, _ in value.observed_variables()} <= controls}
    required = {name for entry in entries for name, _ in entry["candidate"].program.observed_variables()
                if name != "__target__"} - controls
    # Opaque structural predicates and incomplete instruments keep using the
    # complete-program planner. No synthetic condition certifies a response.
    if not pending or required - set(current_binding):
        plan = plan_competing_object_probe(command_spec, entries, budget, binding=binding,
            methods=methods, serial=serial, length=length, tolerance=tolerance, task_id=task_id)
        return finish(plan)

    generation = SearchBudget(min(6000, budget.remaining // 8))
    skipped = []
    method_search_cost = 0
    while pending:
        plan = plan_condition_questions(command_spec, pending, budget, binding=binding,
            methods=methods, serial=serial, length=length, tolerance=tolerance, task_id=task_id,
            generation_budget=generation, response_programs=[entry["candidate"].program for entry in entries])
        method_search_cost += plan.get("method_search_cost", 0)
        plan["hypotheses"] = original_hypotheses
        if plan["probe"] is None or plan["forecast_status"] != "predicted_distinction":
            break
        sequence = plan["probe"].sequence
        predictions = []
        for entry, hypothesis in zip(entries, original_hypotheses):
            if not budget.spend(len(sequence)):
                plan.update(probe=None, forecast_status="compute_exhausted")
                return finish(plan)
            executable = bind_measurement_arguments(entry["candidate"].program.expanded(), current_binding)
            prediction = _forecast(executable, sequence)
            hypothesis["predictions"] = prediction
            predictions.append(prediction)
        common = [values for values in zip(*predictions) if all(value is not None for value in values)]
        separation = sum((max(values) - min(values)) ** 2 for values in common) / len(common) if common else 0.
        plan["predicted_separation"] = separation
        if separation > tolerance:
            for name in sorted(required):
                if not budget.spend(len(sequence)):
                    plan.update(probe=None, forecast_status="compute_exhausted")
                    return finish(plan)
                plan["predicted_fields"][name] = _forecast(current_binding[name], sequence)
            break
        question_id = plan["question"]["question_id"]
        skipped.append(question_id)
        pending = [question for question in pending if question["question_id"] != question_id]
        plan.update(probe=None, forecast_status="no_predicted_distinction")
    plan["generation_cost"] = generation.spent
    plan["method_search_cost"] = method_search_cost
    if skipped:
        plan["questions_without_response_distinction"] = skipped
    return finish(plan)
