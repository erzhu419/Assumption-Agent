"""Solve a shared activation question independently of its response versions.

Question forecasts describe where a branch can be measured. They are kept
separate from response hypotheses: the caller restores and forecasts complete
response programs before purchasing an experiment or learning from its target.
"""

from copy import deepcopy

from .competing_object_experiments import _forecast, _probe_pool
from .language import Program
from .models import Probe, SearchBudget
from .program_disagreement_probes import propose_program_disagreement_probes
from .program_experiment_methods import ProgramExperimentMethods, _control_roles, _current_condition, _same_failed_field
from .scale_bindings import bind_measurement_arguments, control_history_depth


def _controls(program):
    result = []

    def visit(node):
        if node.op in {"feature", "lag"} and node.name not in result:
            result.append(node.name)
        for child in node.children:
            visit(child)

    visit(program)
    return result


def _condition_methods(methods, response_condition=None, question_condition=None, *, response_roles=(), bounds=None):
    """Recall the controls of a saved experiment, excluding unrelated laws."""
    if methods is None:
        return None, {}
    records = deepcopy(methods.records)
    contexts = {}
    for method in records:
        if response_condition is not None:
            for scope in method.get("failure_scopes", []):
                if (scope.get("reason") == "unresolved_under_current_condition" and
                        scope.get("condition") == response_condition):
                    # Compare actual response laws before projecting retrieval.
                    # The existing method lookup can then skip this same scope
                    # while still checking every stored calibration failure.
                    scope["condition"] = question_condition
        if method.get("solver_strategy") == "temporal_prefix":
            # Time recipes already name only the controls of their predicate.
            # Keep their dependencies together when recalling a compact question.
            continue
        boundary = method.get("source_boundary", {})
        source = method.get("source_control_roles", [])
        fractions = method.get("context_fractions", [])
        # The compact predicate only identifies what to solve. Preserve the
        # measured procedure's complete response background when its recorded
        # roles correspond to the current response, including renamed controls.
        if (response_roles and len(source) == len(response_roles) == len(fractions)
                and all(name in boundary.get("context", {}) for name in source)):
            contexts[method["method_id"]] = {
                name: bounds[name][0] + (bounds[name][1] - bounds[name][0]) * fraction
                for name, fraction in zip(response_roles, fractions)}
        expressions = ([item["expression"] for item in boundary["constraints"]]
                       if boundary.get("constraints") else
                       [boundary["expression"]] if "expression" in boundary else [])
        names = list(dict.fromkeys(name for expression in expressions
                                  for name in _controls(Program.from_dict(expression))))
        if not names or any(name not in source for name in names):
            continue
        coordinate = method.get("coordinate_role")
        controlled = source[coordinate] if coordinate is not None else None
        if controlled is not None and controlled not in names:
            continue
        method["context_fractions"] = [method["context_fractions"][source.index(name)] for name in names]
        method["source_control_roles"] = names
        if controlled is not None:
            method["coordinate_role"] = names.index(controlled)
    return ProgramExperimentMethods(records), contexts


def _questions(questions):
    distinct = {}
    for item in questions:
        predicate = item["predicate"].expanded()
        depth = item.get("history_depth", 0)
        key = predicate, depth
        if key not in distinct:
            distinct[key] = {**item, "predicate": predicate,
                "history_depth": depth, "dependent_hypothesis_ids": [], "uncovered_phases": []}
        shared = distinct[key]
        for name in ("dependent_hypothesis_ids", "uncovered_phases"):
            for value in item.get(name, ("active",) if name == "uncovered_phases" and not depth else ()):
                if value not in shared[name]:
                    shared[name].append(value)
    return list(distinct.values())


def _question_program(question):
    selector = Program("if", children=(question["predicate"],
                                       Program("const", value=1.), Program("const", value=0.)))
    depth = question["history_depth"]
    return Program("event_delta", children=(selector,), lag=depth) if depth else selector


def _phases(question, sequence, binding, budget):
    predicate = bind_measurement_arguments(question["predicate"], binding)
    if not budget.spend(len(sequence) * predicate.size):
        return []
    values = _forecast(predicate, sequence)
    depth, phases = question["history_depth"], []
    for position, value in enumerate(values):
        if value is None or position < depth:
            continue
        phase = None
        if not depth:
            phase = "active" if value else None
        else:
            previous = values[position - depth:position]
            if value:
                phase = ("enter" if any(prior == 0. for prior in previous) else
                         "maintain" if all(prior is not None and prior for prior in previous) else None)
            elif any(prior is not None and prior for prior in previous):
                phase = "exit"
        if phase is not None and phase not in phases:
            phases.append(phase)
    return phases


def plan_condition_questions(command_spec, questions, budget, *, binding=None,
                             methods=None, serial=0, length=4, tolerance=1e-7,
                             task_id=None, generation_budget=None, response_programs=None):
    """Propose one shared question under one 6000-unit synthesis allowance.

    A dependent response version adds no predicate solving or compact forecast
    work. Different current field bindings remain different questions. Returned
    ``question_hypotheses`` are conditional indicators, never target forecasts;
    ``hypotheses`` is deliberately empty for the caller to fill with actual laws.
    Pass one ``generation_budget`` across attempts when complete response laws
    reject a proposed question, so retrying never renews the synthesis allowance.
    ``response_programs`` supplies those complete laws for recalling failures
    under their actual response scope rather than the conditional indicator.
    """
    start = budget.spent
    questions = _questions(questions)
    response_programs = list(response_programs) if response_programs is not None else None
    generation = SearchBudget(min(6000, budget.remaining // 8,
        6000 - generation_budget.spent if generation_budget is not None else 6000,
        generation_budget.remaining if generation_budget is not None else 6000))
    response_condition = (_current_condition(response_programs, binding or {},
        command_spec.feature_bounds, task_id) if response_programs is not None else None)
    result = {"probe": None, "hypotheses": [], "question_hypotheses": [],
              "predicted_fields": {}, "forecast_status": "no_condition_question",
              "predicted_separation": None, "proposal_source": None,
              "compute_cost": 0, "generation_cost": 0, "method_search_cost": 0,
              "directed_candidates": 0, "method_candidates": 0,
              "generic_candidates": 0, "fresh_synthesis_skipped": False,
              "question_count": len(questions), "questions_attempted": 0}

    def finish():
        budget.spend(generation.spent)
        if generation_budget is not None:
            generation_budget.spend(generation.spent)
        result.update(generation_cost=generation.spent, compute_cost=budget.spent - start)
        return result

    controls = set(command_spec.feature_bounds)
    bindings = {name: program for name, program in (binding or {}).items()
                if {field for field, _ in program.observed_variables()} <= controls}
    if not questions or not controls:
        return finish()
    response_roles = (_control_roles(response_programs, bindings, command_spec.feature_bounds)
                      if response_programs is not None and methods is not None else ())
    failed_scopes = []
    if response_programs is not None and methods is not None:
        required_responses = {name for program in response_programs for name, _ in program.observed_variables()
                              if name != "__target__"} - controls
        for method in methods.records:
            for scope in method.get("failure_scopes", []):
                if scope["reason"] != "calibration_counterexample":
                    continue
                failed = [field["name"] for field in scope["fields"] if field["name"] in required_responses
                          and _same_failed_field(field, binding or {}, command_spec.feature_bounds)]
                if failed:
                    failed_scopes.append({"method_id": method["method_id"],
                        "reason": "calibration_counterexample", "fields": failed})
    result["forecast_status"] = "no_predicted_distinction"
    for question in questions:
        if not question["uncovered_phases"]:
            continue
        result["questions_attempted"] += 1
        result["question"] = {**question, "predicate": question["predicate"].to_dict()}
        programs = [_question_program(question), Program("const", value=0.)]
        required = {name for name, _ in question["predicate"].observed_variables()
                    if name != "__target__"} - controls
        current_binding = {name: program for name, program in bindings.items() if name in required}
        recall, contexts = _condition_methods(methods, response_condition,
            _current_condition(programs, current_binding, command_spec.feature_bounds, task_id),
            response_roles=response_roles, bounds=command_spec.feature_bounds)
        probe_length = min(command_spec.max_probe_length,
                           max(2, length, question["history_depth"] + 2))
        if any(control_history_depth(program) for program in current_binding.values()):
            probe_length = min(command_spec.max_probe_length,
                max(probe_length, *(control_history_depth(bind_measurement_arguments(program, current_binding)) + 2
                                    for program in programs)))
        if failed_scopes:
            sequence = list(_probe_pool(command_spec, serial, probe_length))[-12 + serial % 12]
            methods.last_proposal = {"status": "blocked_calibration", "blocked_scopes": failed_scopes}
            result.update(probe=Probe(f"condition_recalibration:{serial}", sequence),
                forecast_status="requires_paired_measurements", proposal_source="calibration",
                reason="previous_measurements_refuted_current_calibration",
                failed_fields=sorted({name for scope in failed_scopes for name in scope["fields"]}),
                method_scope=deepcopy(methods.last_proposal))
            return finish()
        if required - set(current_binding):
            sequence = list(_probe_pool(command_spec, serial, probe_length))[-12 + serial % 12]
            result.update(probe=Probe(f"condition_calibration:{serial}", sequence),
                forecast_status="requires_paired_measurements", proposal_source="calibration",
                missing_fields=sorted(required - set(current_binding)))
            return finish()
        before = generation.spent
        learned = recall.propose(command_spec, programs, generation, binding=current_binding,
            serial=serial, length=probe_length, tolerance=tolerance, task_id=task_id,
            contexts=contexts) if recall else []
        result["method_search_cost"] += generation.spent - before
        result["method_candidates"] += len(learned)
        if recall is not None:
            methods.last_proposal = deepcopy(recall.last_proposal)
            scope = recall.last_proposal
            if scope.get("blocked_scopes"):
                result["method_scope"] = deepcopy(scope)
            if scope.get("status") == "blocked_calibration":
                sequence = list(_probe_pool(command_spec, serial, probe_length))[-12 + serial % 12]
                result.update(probe=Probe(f"condition_recalibration:{serial}", sequence),
                    forecast_status="requires_paired_measurements", proposal_source="calibration",
                    reason="previous_measurements_refuted_current_calibration")
                return finish()

        def choose(proposals):
            partial = None
            for item in proposals:
                if item["predicted_separation"] <= tolerance:
                    continue
                phases = _phases(question, item["probe"].sequence, current_binding, generation)
                if not set(phases).intersection(question["uncovered_phases"]):
                    continue
                proposal = item, phases
                if set(question["uncovered_phases"]) <= set(phases):
                    return proposal
                if partial is None:
                    partial = proposal
            return partial

        selected = choose(learned)
        source = "learned_method"
        if selected is not None and set(question["uncovered_phases"]) <= set(selected[1]):
            result["fresh_synthesis_skipped"] = True
        else:
            directed = propose_program_disagreement_probes(command_spec, programs, generation,
                binding=current_binding, serial=serial, length=probe_length)
            result["directed_candidates"] += len(directed)
            fresh = choose(directed)
            if fresh is not None:
                selected, source = fresh, "program_disagreement"
        if selected is None:
            # Generic controls can expose broad or non-inequality predicates.
            # They consume the same shared generation allowance as synthesis.
            executable = [bind_measurement_arguments(program, current_binding) for program in programs]
            for sequence in _probe_pool(command_spec, serial, probe_length):
                if not generation.spend(len(sequence) * len(programs)):
                    break
                predictions = [_forecast(program, sequence) for program in executable]
                result["generic_candidates"] += 1
                common = [row for row in zip(*predictions) if all(value is not None for value in row)]
                separation = sum((max(row) - min(row)) ** 2 for row in common) / len(common) if common else 0.
                fresh = choose([{"probe": Probe(f"condition_generic:{serial}", sequence),
                                 "predictions": predictions, "predicted_separation": separation}])
                if fresh is not None:
                    selected, source = fresh, "generic"
                    break
        if selected is None:
            continue
        item, phases = selected
        result["question_hypotheses"] = [{"question_id": question["question_id"],
            "program": program.to_dict(), "predictions": prediction}
            for program, prediction in zip(programs, item["predictions"])]
        result.update(probe=item["probe"], forecast_status="predicted_distinction",
            proposal_source=source, proposal={key: value for key, value in item.items()
                if key not in {"probe", "predictions"}},
            predicted_separation=item["predicted_separation"], predicted_question_phases=phases)
        return finish()
    if not generation.remaining:
        result["forecast_status"] = "compute_exhausted"
    return finish()
