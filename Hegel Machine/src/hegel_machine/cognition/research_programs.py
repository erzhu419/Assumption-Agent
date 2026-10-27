"""Compose learned computations and retain the data-selected research procedure.

The operator vocabulary is finite. The current residual decides which reusable
computation to add and which temporal question to pursue; task names and private
apparatus parameters never enter this search. Fresh evidence is still needed to
select a proposal. A saved procedure can refit its chosen intermediate terms on
new observations without repeating the entire grammar search.
"""

from itertools import combinations, combinations_with_replacement
from copy import deepcopy
import json
from math import isfinite
from statistics import mean

from .continuous_state import _linear_fit, _sum_terms
from .evaluation import initial_prediction, point_loss
from .language import MissingHistory, Program
from .models import Candidate
from .parallel_prediction import predict_rows
from .research_graph import ResearchGraph
from .scale_bindings import bind_measurement_arguments
from .structure_lab import ObjectSchema


def _sources(program):
    sources = []
    if program.op == "call":
        source = program.definition.source_concept_id or program.definition.definition_id
        sources.append(source)
        sources.extend(_sources(program.definition.body))
    for child in program.children:
        sources.extend(_sources(child))
    return list(dict.fromkeys(sources))


def _temporal(program):
    yield Program("event_delta", lag=1, children=(program,))
    for width in (2, 3, 4):
        yield Program("window_sum", lag=width, children=(program,))
    for decay in (.2, .5, .8, .95):
        yield Program("accumulate", value=decay,
                      children=(program, Program("const", value=0.)))


class _Fitter:
    def __init__(self, rows, budget, quantile):
        self.rows, self.budget, self.quantile = rows, budget, quantile
        self.count = len(rows)
        self.targets = [row.target for row in rows]
        self.fallback = initial_prediction(rows, quantile)
        self.columns, self.fits = {}, {}

    def column(self, term):
        if term in self.columns:
            return self.columns[term]
        if not self.budget.spend(self.count):
            return None
        try:
            values = predict_rows(term, self.rows)
        except (MissingHistory, KeyError):
            values = None
        if values is not None and not all(isfinite(value) for value in values):
            values = None
        self.columns[term] = values
        return values

    def fit(self, terms):
        terms = tuple(dict.fromkeys(terms))
        if not terms:
            return None
        if terms in self.fits:
            return self.fits[terms]
        missing = sum(term not in self.columns for term in terms)
        if not self.budget.can_spend((missing + 2) * self.count):
            return None
        columns = [self.column(term) for term in terms]
        if any(column is None for column in columns):
            self.fits[terms] = None
            return None
        active = [(term, column) for term, column in zip(terms, columns)
                  if max(column) - min(column) > 1e-10]
        if not active:
            self.fits[terms] = None
            return None
        terms, columns = tuple(zip(*active))
        self.budget.spend(self.count)
        coefficients = _linear_fit(columns, self.targets)
        program = _sum_terms(terms, coefficients[1:], coefficients[0])
        self.budget.spend(self.count)
        predictions = predict_rows(program, self.rows)
        if not all(isfinite(value) for value in predictions):
            raise ValueError("candidate produced a non-finite prediction")
        measured = mean(point_loss(value, row.target, self.quantile)
                        for value, row in zip(predictions, self.rows))
        self.budget.completed_fits += 1
        used = tuple(term for term, coefficient in zip(terms, coefficients[1:])
                     if abs(coefficient) >= 1e-8)
        result = {"terms": used, "program": program, "loss": measured,
                  "coefficients": coefficients, "fit_terms": terms}
        self.fits[terms] = result
        return result


def _rank(fit):
    # Once the recorded evidence is explained numerically, retain the smaller
    # explanation instead of choosing a much larger expression for roundoff.
    return (0. if fit["loss"] < 1e-12 else fit["loss"],
            fit["program"].size)


def _plan(fit, steps, count, compute_cost):
    graph = ResearchGraph()
    graph.evidence_count = count
    terms = fit["fit_terms"]
    for index, term in enumerate(terms):
        node = graph.add_question(f"term:{index}",
            "Evaluate a reusable computation in the current representation",
            operations=("reuse",), expression=term,
            required_measurements=sorted(term.required_features()))
        node.answer, node.status = term, "answered"
    previous = None
    for index, step in enumerate(steps):
        key = f"fit:{index}"
        node = graph.add_question(key, step["question"],
            operations=(step["operation"],),
            depends_on=([previous] if previous else []),
            expression=step["program"])
        node.answer, node.status, node.loss = step["program"], "answered", step["loss"]
        previous = key
    dependencies = [f"term:{index}" for index in range(len(terms))]
    expression = _sum_terms([Program("arg", name=key) for key in dependencies],
                            fit["coefficients"][1:], fit["coefficients"][0])
    output = graph.add_question("answer", "Refit and combine the selected computations",
        operations=("refit", "composition"),
        depends_on=[*dependencies, *([previous] if previous else [])], expression=expression)
    output.answer, output.status, output.loss = fit["program"], "answered", fit["loss"]
    output.compute_cost = compute_cost
    return {"kind": "residual_composition", "graph": graph.to_dict(),
            "term_nodes": dependencies, "output_node": "answer",
            "selection_steps": [{**step, "program": step["program"].to_dict()}
                                for step in steps],
            "fit_rows": count, "search_compute": compute_cost,
            "source_ids": _sources(fit["program"]),
            "status": "requires_fresh_validation"}


def _entry(fit, steps, count, compute_cost, index=0):
    plan = _plan(fit, steps, count, compute_cost)
    return {"candidate": Candidate(f"research_composition:{index}", fit["program"],
                "research_program", "Observed residual selects a composition of learned computations",
                tuple(plan["source_ids"])),
            "kind": "research_program", "family_id": "T03", "training_loss": fit["loss"],
            "research_program": plan}


def consolidate_research_programs(entries, *, allocation, candidate_id='research_method'):
    """Group actual procedures by a shared search contract, retaining each seed.

    This learns an optional method, not a shared world mechanism or a utility
    trigger. Conditional replay already re-identifies thresholds, durations,
    placement and coefficients; generic replay re-identifies readout weights.
    Other specialized replayers need their own actual input contract first.
    """
    from .abstractions import AbstractTemplate
    from .models import SearchBudget

    budget = SearchBudget(allocation['compute'])
    groups, requests = {}, []
    for entry in entries:
        payload = entry['research_program']
        program = Program.from_dict(entry['program'])
        if not budget.spend(program.size):
            break
        output = payload['graph']['nodes'][payload['output_node']]['answer']
        if output is None or Program.from_dict(output) != program:
            requests.append({'type': 'AuthorWorkRequest', 'source_ref': entry['source_ref'],
                             'reason': 'saved_procedure_output_does_not_match_candidate'})
            continue
        signatures = entry['role_signature']
        terms = [Program.from_dict(payload['graph']['nodes'][key]['answer'])
                 for key in payload['term_nodes']]
        names = tuple(sorted(set().union(program.required_features(),
                                         *(term.required_features() for term in terms))))
        if any(name not in signatures or any(key not in signatures[name]
                   for key in ('role', 'unit', 'time')) for name in names):
            requests.append({'type': 'AuthorWorkRequest', 'source_ref': entry['source_ref'],
                             'reason': 'procedure_role_unit_time_contract_required'})
            continue
        roles = tuple(f'role_{index}' for index in range(len(names)))
        mapping = dict(zip(names, roles))
        steps = payload.get('selection_steps', [])
        conditional = next((step for step in reversed(steps)
                            if step['operation'] == 'discover_conditional_coordinates'), None)
        specialized = {'discover_decoded_conditional_coordinates', 'discover_composed_conditions',
                       'resolve_relation_intervals'}
        if any(step['operation'] in specialized for step in steps):
            requests.append({'type': 'AuthorWorkRequest', 'source_ref': entry['source_ref'],
                             'reason': 'shared_specialized_replay_contract_not_established'})
            continue
        canonical = lambda value: Program.from_dict(value).remap_features(mapping).to_dict()
        if conditional:
            operation = {'operation': 'discover_conditional_coordinates',
                'gate_coordinate': canonical(conditional['gate_coordinate']),
                'branch_basis': [canonical(term) for term in conditional['branch_basis']]}
            freedoms = ['threshold', 'width', 'placement', 'response_coefficients']
        else:
            operation = {'operation': 'refit_selected_terms', 'terms': [canonical(
                payload['graph']['nodes'][key]['answer']) for key in payload['term_nodes']]}
            freedoms = ['response_coefficients']
        signature = {role: deepcopy(signatures[name]) for name, role in mapping.items()}
        context = entry.get('context_family_ref', entry['context_ref'])
        key = json.dumps([operation, signature, entry['output_unit'], context], sort_keys=True)
        groups.setdefault(key, []).append((entry, roles, names, signature, operation, freedoms))
    candidates = []
    for records in groups.values():
        sources = list(dict.fromkeys(item[0]['source_id'] for item in records))
        if len(sources) < 2:
            requests.append({'type': 'AuthorWorkRequest',
                'reason': 'distinct_physical_research_sources_required', 'source_ids': sources,
                'source_refs': [deepcopy(item[0]['source_ref']) for item in records]})
            continue
        if len(candidates) >= allocation.get('max_groups', 16):
            break
        first, roles, names, signature, operation, freedoms = records[0]
        identifier = f'{candidate_id}:{len(candidates)}'
        template = AbstractTemplate(identifier, first['episode_id'],
            Program('object_sum', children=tuple(Program('feature', name=role) for role in roles)),
            roles, names, ())
        retained = [{key: deepcopy(entry[key]) for key in (
            'episode_id', 'source_id', 'source_ref', 'action_ref', 'evidence_refs', 'cost_refs',
            'failure_refs', 'source_cutoff', 'status', 'source_revision_steps', 'independent_results',
            'cost_scope', 'source_outcome_ref')
            if key in entry} | {'input_mapping': dict(zip(roles, source_names))}
            for entry, _, source_names, *_ in records]
        candidates.append({'id': identifier, 'kind': 'research_program', 'purpose': 'method',
            'producer': 'finite_native_saved_replay_contract_grouping',
            'status': 'EXECUTABLE_CANDIDATE', 'frozen': True, 'active_library': False,
            'construction_only': True, 'global_support': False, 'support_updated': False,
            'independent_confirmation': False, 'transfer_status': 'not_established',
            'template': template.to_dict(), 'role_signature': signature,
            'output_unit': first['output_unit'], 'research_program_contract': operation,
            'free_parameters': freedoms, 'source_ids': sources, 'independent_source_count': len(sources),
            'source_outcomes': retained, 'source_contexts': [deepcopy(item[0]['context_ref']) for item in records],
            'research_program_source_ref': deepcopy(first['source_ref']),
            'research_program_source_refs': [deepcopy(item['source_ref']) for item in retained],
            'source_seed_inputs': [{'source_id': entry['source_id'],
                'source_ref': deepcopy(entry['source_ref']), 'retained_term_refs': [
                    {**deepcopy(entry['source_ref']), 'json_path': [*entry['source_ref']['json_path'],
                     'research_program', 'graph', 'nodes', key, 'answer']}
                    for key in entry['research_program']['term_nodes']]}
                for entry, *_ in records],
            'seed_policy': 'equal_bounded_source_order_distinct_replay_inputs',
            'source_reconstruction_scope': 'individual_saved_graph_identity_only',
            'source_reconstruction': [{'source_id': item['source_id'],
                'source_ref': deepcopy(item['source_ref']),
                'exact': True, 'comparison': 'saved_graph_output_equals_its_actual_candidate_not_shared_method'}
                for item in retained],
            'trigger': {'kind': 'current_research_obligation', 'learned_utility_trigger': False},
            'scope': {'context_ref': deepcopy(first.get('context_family_ref', first['context_ref'])),
                'application': 'unconfirmed', 'relative_utility': 'unconfirmed',
                'measured_coverage': [deepcopy(item['source_ref']) for item in retained]},
            'counterexample_refs': [deepcopy(ref) for item in retained for ref in item.get('failure_refs', [])],
            'consumer': 'hegel_machine.cognition.research_programs:replay_research_program'})
    exhausted = budget.last_denied_request is not None
    return {'status': 'UNKNOWN' if exhausted or requests else 'completed', 'candidates': candidates,
        'author_work': requests, 'excluded': [], 'actual_compute_cost': budget.spent,
        'budget_exhausted': exhausted}


def research_program_replay_inputs(payload, *, feature_mapping=None):
    """Describe the actual finite replayer input, including retained seed terms."""
    def mapped(value):
        program = Program.from_dict(value)
        if feature_mapping:
            program = bind_measurement_arguments(program, feature_mapping)
        return program.to_dict()

    value = {'retained_terms': [mapped(payload['graph']['nodes'][key]['answer'])
                               for key in payload['term_nodes']]}
    saved = next((step for step in reversed(payload.get('selection_steps', []))
                  if step['operation'] == 'discover_conditional_coordinates'), None)
    if saved:
        value.update(operation='discover_conditional_coordinates',
            gate_coordinate=mapped(saved['gate_coordinate']),
            branch_basis=[mapped(term) for term in saved['branch_basis']])
    else:
        value['operation'] = 'refit_selected_terms'
    return value


def propose_research_step_revisions(payload, training=(), *, feature_names=(),
                                    feature_mapping=None, source_ref=None,
                                    max_candidates=8, max_history=None):
    """Change one input that the saved replayer really consumes.

    Selection steps usually describe historical work, rather than instructions
    replayed in order. Editing their prose or old losses is not a method edit.
    These proposals instead alter one retained computation, conditional gate,
    or branch coordinate. The finite vocabulary and guarded measured history
    are shared with ordinary coordinate research; no task label selects a lag.
    """
    from .condition_learning import _condition_coordinates
    from .research_drafts import _rebind_step

    rows = list(training)
    bound = _rebind_step(payload, feature_mapping or {})
    original = research_program_replay_inputs(bound)
    unsupported = {"discover_decoded_conditional_coordinates", "discover_composed_conditions",
                   "resolve_relation_intervals"}
    if any(step["operation"] in unsupported for step in bound.get("selection_steps", [])):
        return {"status": "step_contract_not_established", "candidates": [],
            "reason": "specialized_replay_requires_its_existing_parameter_and_dependency_contract"}
    terms = [Program.from_dict(value) for value in original["retained_terms"]]
    names = (set.intersection(*(set(row.features) for row in rows)) - {"__target__"}) if rows else set(feature_names)
    names &= set(feature_names) if feature_names else names
    current = [Program("feature", name=name) for name in sorted(names)]
    upper = max((len(row.history) for row in rows), default=0)
    upper = min(upper, max_history) if max_history is not None else upper
    coordinates = _condition_coordinates(rows, upper) if rows else current
    coordinates = [term for term in coordinates if term.required_features() <= names | {"__target__"}]
    history = [term for term in coordinates if term.history_depth]
    products = list(dict.fromkeys(Program("mul", children=pair)
        for pair in combinations_with_replacement(current[:8], 2)))
    saved = [Program.from_dict(step["program"]) for step in bound.get("selection_steps", [])
        if step.get("program") and Program.from_dict(step["program"]).required_features() <= names]
    # Round-robin input families keep current-coordinate variants from taking
    # all places before an acquired historical distinction receives a trial.
    pools = [list(history), [*current, *products], list(dict.fromkeys(saved))]
    choices = []
    while any(pools):
        for pool in pools:
            if pool:
                term = pool.pop(0)
                if term not in choices:
                    choices.append(term)
    conditional = next((index for index in reversed(range(len(bound.get("selection_steps", []))))
        if bound["selection_steps"][index]["operation"] == "discover_conditional_coordinates"), None)
    edits = []
    for term in choices:
        if term not in terms:
            edits.append({"kind": "add_retained_input", "term": term.to_dict(),
                "target": ["term_nodes"], "coordinate_source": "paid_public_coordinates_or_saved_computation"})
        if conditional is not None and not term.history_depth:
            step = bound["selection_steps"][conditional]
            if term.to_dict() != step["gate_coordinate"]:
                edits.append({"kind": "replace_gate", "term": term.to_dict(),
                    "before": deepcopy(step["gate_coordinate"]),
                    "target": ["selection_steps", conditional, "gate_coordinate"]})
            # Raw coordinates are already reintroduced by _coordinates. A
            # deletion/addition there would not change this execution step.
            if term not in current and term.to_dict() not in step["branch_basis"]:
                edits.append({"kind": "add_branch_coordinate", "term": term.to_dict(),
                    "target": ["selection_steps", conditional, "branch_basis"]})
    if len(terms) > 1:
        for key, term in zip(bound["term_nodes"], terms):
            edits.append({"kind": "remove_retained_input", "before": term.to_dict(),
                "target": ["term_nodes"], "node": key})
    candidates, seen = [], [original]
    # Different actual steps receive opportunities before more alternatives
    # of the same step, independent of candidate identity or observed return.
    kinds = list(dict.fromkeys(edit["kind"] for edit in edits))
    queues = [[edit for edit in edits if edit["kind"] == kind] for kind in kinds]
    while any(queues) and len(candidates) < max_candidates:
        for queue in queues:
            if not queue or len(candidates) >= max_candidates:
                continue
            edit, revised = queue.pop(0), deepcopy(bound)
            if edit["kind"] == "add_retained_input":
                graph = ResearchGraph.from_dict(revised["graph"])
                key = f"step_input:{len(graph.nodes)}"
                node = graph.add_question(key, "Test an alternative measured computation in this replay step",
                    operations=("reuse",), expression=Program.from_dict(edit["term"]))
                node.answer, node.status = node.expression, "answered"
                revised["graph"] = graph.to_dict()
                revised["term_nodes"].append(key)
            elif edit["kind"] == "remove_retained_input":
                revised["term_nodes"].remove(edit["node"])
            else:
                step = revised["selection_steps"][conditional]
                if edit["kind"] == "replace_gate":
                    step["gate_coordinate"] = deepcopy(edit["term"])
                else:
                    step["branch_basis"].append(deepcopy(edit["term"]))
            inputs = research_program_replay_inputs(revised)
            if inputs in seen:
                continue
            seen.append(inputs)
            revised["status"] = "requires_step_validation"
            revised["method_step_revision"] = {"edit": deepcopy(edit), "source_ref": deepcopy(source_ref)}
            candidates.append({"edit": edit, "research_program": revised, "replay_inputs": inputs,
                "source_refs": [deepcopy(source_ref)] if source_ref else [],
                "status": "unmeasured_step_intervention"})
    return {"status": "proposed" if candidates else "no_distinct_executable_step",
        "candidates": candidates, "baseline_research_program": bound,
        "baseline_replay_inputs": original, "observed_history_bound": upper,
        "scope": "finite_step_interventions_not_world_support"}


def evaluate_research_step_revisions(payload, training, validation, budget, *,
                                     per_method_compute, risk_quantile=.5,
                                     feature_mapping=None, feature_names=(), source_refs=(),
                                     source_ref=None, max_candidates=8, max_history=None,
                                     previous=None, minimum_improvement=1e-10,
                                     paid_baseline=None, on_progress=None):
    """Pair the old method and one-step interventions on the same paid split.

    Every method receives the same fit-plus-assessment ceiling. Only completed
    pairs can support a local step-intervention effect. Saved complete slots
    are reusable under the caller's immutable paid-view references; incomplete
    work and its costs remain visible, never becoming a method counterexample.
    A progress callback can persist each newly charged slot before the next
    method starts; callback failures stop the pass rather than losing the bill.
    """
    from .evaluation import loss
    from .experiment_runtime import PythonExecutionDeadline
    from .models import SearchBudget
    from .research_methods import common_method_process_return

    training, validation = list(training), list(validation)
    proposal = propose_research_step_revisions(payload, training,
        feature_names=feature_names, feature_mapping=feature_mapping, source_ref=source_ref,
        max_candidates=max_candidates, max_history=max_history)
    programs = [proposal.get("baseline_research_program", payload),
        *[item["research_program"] for item in proposal["candidates"]]]
    inputs = [research_program_replay_inputs(item) for item in programs]
    basis = {"replay_inputs": inputs, "source_refs": deepcopy(list(source_refs)),
        "training_rows": len(training), "validation_rows": len(validation),
        "risk_quantile": risk_quantile, "per_method_compute": per_method_compute,
        "minimum_improvement": minimum_improvement}
    reusable = (previous and source_refs and previous.get("basis") == basis)
    paid = deepcopy(previous.get("methods", [])) if reusable else []
    reused = {item["index"]: item for item in paid if item.get("execution_complete")}
    baseline_basis = {"replay_inputs": inputs[0], "source_refs": deepcopy(list(source_refs)),
        "training_rows": len(training), "validation_rows": len(validation), "risk_quantile": risk_quantile}
    baseline_validation_cost = None
    if paid_baseline is not None and 0 not in reused:
        score = paid_baseline.get("validation_loss")
        scored = type(score) in (int, float) and isfinite(score) and score >= 0
        known = (bool(source_refs) and paid_baseline.get("basis") == baseline_basis
            and paid_baseline.get("execution_complete") is True
            and not paid_baseline.get("execution_error")
            and (scored or paid_baseline.get("program")))
        if known:
            baseline_validation_cost = 0 if scored else len(validation)
    required = (len(programs) - len(reused)) * per_method_compute
    if baseline_validation_cost is not None:
        required += baseline_validation_cost - per_method_compute
    result = {"status": "unresolved", "basis": basis, "proposal": proposal, "methods": paid,
        "baseline_basis": baseline_basis,
        "pairs": [], "selected_revision": None, "required_compute": required,
        "compute_cost": 0, "previous_compute_cost": previous.get("paid_compute_cost", 0) if reusable else 0,
        "budget_incomplete": False, "independent_support": False,
        "time_incomplete": False,
        "scope": "same_paid_split_development_step_intervention"}
    if paid_baseline is not None and 0 not in reused and baseline_validation_cost is None:
        return {**result, "status": "paid_baseline_not_reusable", "required_compute": None,
            "reason": "exact_original_seed_program_or_same_split_paid_score_required_no_automatic_refit"}
    if proposal["status"] == "step_contract_not_established":
        return {**result, "status": proposal["status"], "required_compute": 0}
    if not validation or per_method_compute <= len(validation):
        return {**result, "status": "complete_fit_and_validation_contract_required"}
    if not budget.can_spend(required):
        return {**result, "status": "insufficient_budget", "budget_incomplete": True,
            "denied_request": deepcopy(budget.last_denied_request)}
    start, methods = budget.spent, []
    result["retained_incomplete_attempts"] = ([*deepcopy(previous.get("retained_incomplete_attempts", [])),
        *[{key: deepcopy(item.get(key)) for key in ("index", "paid_compute_cost", "execution_error", "denied_request")}
            for item in paid if not item.get("execution_complete")]] if reusable else [])

    def persist_progress():
        if on_progress is None:
            return
        completed_prefix = list(methods)
        known = {item["index"] for item in completed_prefix}
        completed_prefix.extend({**item, "new_compute_cost": 0, "reused": True}
            for index, item in reused.items() if index not in known)
        completed_prefix.sort(key=lambda item: item["index"])
        new_cost = budget.spent - start
        on_progress(deepcopy({**result, "status": "incomplete_step_comparison",
            "methods": completed_prefix, "compute_cost": new_cost,
            "paid_compute_cost": result["previous_compute_cost"] + new_cost,
            "budget_incomplete": any(item.get("denied_request") for item in completed_prefix),
            "unexecuted_method_indices": [index for index in range(len(programs))
                if index not in {item["index"] for item in completed_prefix}]}))

    fallback = initial_prediction(training, risk_quantile) if training else 0.
    for index, program in enumerate(programs):
        if index in reused:
            methods.append({**reused[index], "new_compute_cost": 0, "reused": True})
            continue
        if index == 0 and baseline_validation_cost is not None:
            measured, error = paid_baseline.get("validation_loss"), None
            if baseline_validation_cost:
                budget.spend(baseline_validation_cost)
                try:
                    measured = loss(Program.from_dict(paid_baseline["program"]), validation, fallback, risk_quantile)
                except (ValueError, MissingHistory, KeyError) as exc:
                    error = {"type": type(exc).__name__, "message": str(exc)}
            original_cost = paid_baseline.get("fit_compute_cost")
            cost = (original_cost + baseline_validation_cost
                if type(original_cost) in (int, float) and baseline_validation_cost else
                paid_baseline.get("paid_compute_cost"))
            methods.append({"index": 0, "execution_complete": error is None,
                "new_compute_cost": baseline_validation_cost, "paid_compute_cost": cost,
                "training_loss": paid_baseline.get("training_loss"), "validation_loss": measured,
                "program": deepcopy(paid_baseline.get("program")),
                "research_program": deepcopy(paid_baseline.get("research_program")),
                "cost_ref": deepcopy(paid_baseline.get("cost_ref")),
                "original_fit_compute_cost": original_cost,
                "execution_error": error, "denied_request": None, "reused": True})
            if error and error["type"] == PythonExecutionDeadline.__name__:
                result["time_incomplete"] = True
            if baseline_validation_cost:
                persist_progress()
            if error:
                new_cost = budget.spent - start
                return {**result, "status": "incomplete_paid_baseline_assessment", "methods": methods,
                    "time_incomplete": error["type"] == PythonExecutionDeadline.__name__,
                    "compute_cost": new_cost, "paid_compute_cost": result["previous_compute_cost"] + new_cost}
            continue
        local = SearchBudget(per_method_compute, reserve=len(validation))
        claim, measured, error = None, None, None
        try:
            claim = replay_research_program(program, training, local, risk_quantile=risk_quantile)
            if claim is not None:
                local.release_reserve()
                local.spend(len(validation))
                measured = loss(claim["candidate"].program, validation, fallback, risk_quantile)
        except (ValueError, MissingHistory, KeyError) as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            if isinstance(exc, PythonExecutionDeadline):
                result["time_incomplete"] = True
        complete = claim is not None and measured is not None and error is None and local.last_denied_request is None
        budget.spend(local.spent)
        methods.append({"index": index, "execution_complete": complete,
            "new_compute_cost": local.spent, "paid_compute_cost": local.spent,
            "training_loss": claim["training_loss"] if claim else None,
            "validation_loss": measured, "program": claim["candidate"].program.to_dict() if claim else None,
            "research_program": claim["research_program"] if claim else None,
            "execution_error": error,
            "denied_request": deepcopy(local.last_denied_request), "reused": False})
        persist_progress()
        if result["time_incomplete"]:
            # The shared runtime's deadline is a matrix boundary, not another
            # unproductive method. Preserve the current charge and stop before
            # starting any remaining scientific work.
            known = {item["index"] for item in methods}
            methods.extend({**item, "new_compute_cost": 0, "reused": True}
                for position, item in reused.items() if position not in known)
            methods.sort(key=lambda item: item["index"])
            break
    result["methods"] = methods
    result["unexecuted_method_indices"] = [index for index in range(len(programs))
        if index not in {item["index"] for item in methods}]
    baseline = methods[0]
    for item in methods[1:]:
        candidate = proposal["candidates"][item["index"] - 1]
        complete = baseline["execution_complete"] and item["execution_complete"]
        delta = item["validation_loss"] - baseline["validation_loss"] if complete else None
        process = None
        if complete:
            def side(method):
                return {"execution_complete": True, "completed": True,
                    "quality": {"loss": method["validation_loss"], "threshold": minimum_improvement},
                    "coverage": len(validation), "costs": {"observation_cost": 0,
                        "compute_cost": method["paid_compute_cost"]}}
            process = common_method_process_return({"same_task": True,
                "same_evaluation_contract": True, "source_refs": deepcopy(list(source_refs)),
                "scope": result["scope"], "before": side(baseline), "after": side(item)})
        improved = complete and (delta < -minimum_improvement or process["helpful"] is True)
        result["pairs"].append({"method_index": item["index"], "edit": deepcopy(candidate["edit"]),
            "execution_complete": complete, "same_training_and_validation": True,
            "equal_method_compute_ceiling": per_method_compute, "validation_loss_change": delta,
            "process_return": process,
            "status": "local_step_improvement" if improved else
                "complete_step_unhelpful" if complete else "incomplete_not_negative",
            "scope": result["scope"], "source_refs": deepcopy(list(source_refs))})
    useful = [item for item in result["pairs"] if item["status"] == "local_step_improvement"]
    all_complete = len(methods) == len(programs) and all(item["execution_complete"] for item in methods)
    if useful and all_complete and not result["time_incomplete"]:
        chosen = min(useful, key=lambda pair: (
            0. if methods[pair["method_index"]]["validation_loss"] <= minimum_improvement else
                methods[pair["method_index"]]["validation_loss"],
            methods[pair["method_index"]]["paid_compute_cost"]))
        result["selected_revision"] = deepcopy(proposal["candidates"][chosen["method_index"] - 1])
        selected = methods[chosen["method_index"]]
        result["selected_revision"].update(replayed_research_program=deepcopy(selected["research_program"]),
            paired_program=deepcopy(selected["program"]), validation_loss=selected["validation_loss"])
        result["status"] = "development_step_revision_selected"
    else:
        result["status"] = "complete_no_step_improvement" if all_complete else "incomplete_step_comparison"
    result["compute_cost"] = budget.spent - start
    result["paid_compute_cost"] = result["previous_compute_cost"] + result["compute_cost"]
    result["budget_incomplete"] = any(item.get("denied_request") for item in methods)
    return result


def quote_research_step_revisions(payload, training, validation, *, feature_mapping=None,
                                  feature_names=(), source_ref=None, max_candidates=8,
                                  max_history=None):
    """Bound a complete finite replay/assessment pass without fitting a model.

    Conditional replay can visit only a finite measured gate bracket, integer
    duration and placement cell. Its fitter caches each cell's columns and
    solve. The bound includes all such cells, initial duration fits, both sparse
    deletion paths, retained-input refitting and assessment. This deliberately
    differs from the older bounded-attempt allowance. No clock-time guarantee
    follows from candidate-row accounting.
    """
    from .conditional_coordinates import _coordinates
    from .models import TaskSpec

    training, validation = list(training), list(validation)
    proposal = propose_research_step_revisions(payload, training, feature_names=feature_names,
        feature_mapping=feature_mapping, source_ref=source_ref, max_candidates=max_candidates,
        max_history=max_history)
    result = {"status": "UNKNOWN", "proposal": proposal, "methods": [],
        "per_method_compute_upper": None, "complete_pass_compute_upper": None,
        "coordinate_readout_compute": 0, "new_fits": 0, "new_world_predictions": 0,
        "scope": "finite_candidate_row_upper_not_host_seconds"}
    if proposal["status"] == "step_contract_not_established" or len(training) < 6 or not validation:
        return {**result, "reason": "supported_complete_fit_and_assessment_contract_required"}
    programs = [proposal["baseline_research_program"],
        *[item["research_program"] for item in proposal["candidates"]]]
    count, readouts = len(training), {}
    for index, program in enumerate(programs):
        replay = research_program_replay_inputs(program)
        retained = tuple(dict.fromkeys(Program.from_dict(value) for value in replay["retained_terms"]))
        retained_fit = count * (len(retained) + 2)
        method = {"index": index, "retained_fit_upper": retained_fit,
            "assessment_upper": len(validation), "replay_inputs": replay}
        if replay["operation"] == "refit_selected_terms":
            method["compute_upper"] = retained_fit + len(validation)
            result["methods"].append(method)
            continue
        names = tuple(sorted(set.intersection(*(set(row.features) for row in training)) - {"__target__"}))
        spec = TaskSpec("paid_coordinate_quote", "Count finite replay coordinates", names, (), ())
        available, basis = _coordinates(spec, training,
            [Program.from_dict(term) for term in replay["branch_basis"]])
        gates = list(dict.fromkeys([Program.from_dict(replay["gate_coordinate"]), *available]))[:8]
        gates = [gate for gate in gates if not gate.history_depth and gate.required_features() <= set(names)]
        # _search appends its own intercept coordinate before masking branches.
        branch_terms = 2 * (len(basis) + 1)
        if branch_terms + 3 >= count:
            # The alternate screening path can revisit a failed screen. Its
            # current code does not provide the same finite-cell cost bound.
            return {**result, "reason": "screened_conditional_fit_complete_bound_not_established",
                "unquoted_method_index": index, "branch_terms": branch_terms}
        thresholds = []
        for gate in gates:
            if gate not in readouts:
                try:
                    values = predict_rows(gate, training)
                except (MissingHistory, KeyError, ValueError) as exc:
                    return {**result, "reason": "paid_gate_coordinate_unavailable",
                        "unquoted_method_index": index, "error": str(exc)}
                if not all(isfinite(value) for value in values):
                    return {**result, "reason": "paid_gate_coordinate_nonfinite", "unquoted_method_index": index}
                readouts[gate] = values
                result["coordinate_readout_compute"] += count
            from collections import Counter
            occurrences = Counter(readouts[gate])
            distinct, below, brackets = sorted(occurrences), 0, 0
            for left, right in zip(distinct, distinct[1:]):
                below += occurrences[left]
                brackets += below >= 3 and count - below >= 3
            thresholds.append({"gate_coordinate": gate.to_dict(), "brackets": brackets})
        upper = max(1, max((len(row.history) for row in training), default=0))
        widths, width = [1], 2
        while width < upper:
            widths.append(width)
            width *= 2
        if upper > 1:
            widths.append(upper)
        placements = 2 if upper > 1 else 1
        cells = sum(item["brackets"] for item in thresholds) * upper * placements
        fit_price = count * (branch_terms + 2)
        gate_preparation = 2 * count * len(gates)
        plain_fits = len(widths) * count * (len(basis) + 3)
        # At most two winners: re-fit their active terms once, read their
        # contributions, then delete down to two terms with cached columns.
        deletion = 2 * count * (3 * branch_terms - 2)
        # Keep the existing refinement loop's two-fit reserve available even
        # after its last uncached cell, then let its ordinary no-progress stop.
        completion_reserve = 2 * fit_price
        method.update(gate_brackets=thresholds, observed_history_bound=upper,
            branch_terms=branch_terms, parameter_cells_upper=cells,
            gate_preparation_upper=gate_preparation, plain_fits_upper=plain_fits,
            parameter_fits_upper=cells * fit_price, sparse_deletion_upper=deletion,
            completion_reserve=completion_reserve,
            compute_upper=retained_fit + gate_preparation + plain_fits + cells * fit_price
                + deletion + completion_reserve + len(validation))
        result["methods"].append(method)
    maximum = max(item["compute_upper"] for item in result["methods"])
    return {**result, "status": "quoted", "per_method_compute_upper": maximum,
        "complete_pass_compute_upper": maximum * len(programs),
        "method_count": len(programs), "equal_method_ceiling_required": True}


def coordinate_search_allowance(feature_count, row_count, history_length=0):
    """Quote a bounded attempt using the constructor's two multiplication rounds."""
    first = min(feature_count, 8)
    first_columns = first + first * (first + 1) // 2
    second = min(first_columns, 8)
    second_columns = second + second * (second + 1) // 2
    # Single-column fits, four residual passes and four joint alternatives per
    # pass. Cached columns/duplicate products usually make execution cheaper.
    allowance = row_count * (3 * feature_count + 7 * (first_columns + second_columns) + 72)
    if history_length >= 2:
        # Joint temporal fits need fresh columns for each scale. They consume
        # the same research budget as coordinate composition, not a free pass.
        allowance += row_count * (min(second_columns, 24) + 3) * 16
    return allowance


def propose_coordinate_programs(spec, training, budget, *, max_candidates=8):
    """Build a sparse arithmetic explanation from measured coordinates alone.

    Two rounds of multiplication include self-composition. Residual evidence
    chooses the coordinates retained for the next round; no learned definition
    or task-specific polynomial is needed to open this research direction.
    """
    rows = list(training)
    if len(rows) < 6 or max_candidates < 1:
        return []
    names = set.intersection(*(set(row.features) for row in rows))
    fitter = _Fitter(rows, budget, spec.risk_quantile)
    start = budget.spent
    raw = [fit for name in spec.feature_names if name in names
           if (fit := fitter.fit((Program("feature", name=name),))) is not None]
    if not raw:
        return []
    raw.sort(key=_rank)
    best = raw[0]
    tolerance = max(1e-10, abs(best["loss"]) * 1e-7)
    initial_loss = best["loss"]
    pool = list(dict.fromkeys(term for fit in raw for term in fit["terms"]))[:8]
    path = [{"operation": "fit_measured_coordinate",
        "question": "Explain the response with the currently observed coordinates",
        "program": best["program"], "loss": best["loss"]}]
    found, basis = [], list(pool)

    def product(left, right):
        # Multiplication is commutative for these scalar current coordinates.
        # Canonical factors avoid fitting duplicate columns at the next depth.
        def factors(term):
            return ([part for child in term.children for part in factors(child)]
                    if term.op == "mul" else [term])
        terms = sorted([*factors(left), *factors(right)], key=lambda term: term.name)
        result = terms[0]
        for term in terms[1:]:
            result = Program("mul", children=(result, term))
        return result

    for depth in range(2):
        candidates = list(dict.fromkeys([*pool,
            *(product(left, right) for left, right in combinations_with_replacement(pool, 2))]))
        basis = list(dict.fromkeys([*basis, *candidates]))
        singles = [fit for term in candidates if (fit := fitter.fit((term,))) is not None]
        if not singles:
            break
        singles.sort(key=_rank)
        if _rank(singles[0]) < _rank(best):
            best = singles[0]
        # Current-coordinate expressions are small enough to compare several
        # residual additions jointly; each correlation pass is budgeted.
        for _ in range(4):
            if best["loss"] <= tolerance:
                break
            eligible = [(term, fitter.columns[term]) for term in candidates
                        if term not in best["terms"] and fitter.columns.get(term) is not None]
            if not budget.spend(len(rows) * (len(eligible) + 1)):
                break
            residual = [row.target - value
                        for row, value in zip(rows, predict_rows(best["program"], rows))]
            residual_center = sum(residual) / len(rows)
            ranked = []
            for term, column in eligible:
                center = sum(column) / len(rows)
                norm = sum((value - center) ** 2 for value in column)
                if norm <= 1e-12:
                    continue
                covariance = sum((value - center) * (error - residual_center)
                                 for value, error in zip(column, residual))
                ranked.append((covariance * covariance / norm, term))
            alternatives = [fit for _, term in sorted(ranked, key=lambda item: item[0], reverse=True)[:4]
                            if (fit := fitter.fit((*best["terms"], term))) is not None]
            if not alternatives:
                break
            revised = min(alternatives, key=_rank)
            if revised["loss"] + tolerance >= best["loss"]:
                break
            best = revised
        path = [*path, {"operation": "compose_measured_coordinates",
            "question": "Multiply observed computations and jointly explain the remaining response",
            "program": best["program"], "loss": best["loss"], "composition_depth": depth + 1}]
        if best["loss"] + tolerance < initial_loss:
            found.append((best, path))
        if best["loss"] <= tolerance:
            break
        pool = list(dict.fromkeys([*best["terms"],
            *(term for fit in singles for term in fit["terms"]), *pool]))[:8]
    ranked = sorted(found, key=lambda item: _rank(item[0]))[:max_candidates]
    claims = [_entry(fit, steps, len(rows), budget.spent - start, index)
              for index, (fit, steps) in enumerate(ranked)]
    if best["loss"] > tolerance and max(len(row.history) for row in rows) >= 2:
        from .temporal_coordinates import propose_temporal_coordinates
        claims.extend(propose_temporal_coordinates(spec, rows, budget,
            basis=basis, max_candidates=max_candidates))
    return sorted(claims, key=lambda item: (0. if item["training_loss"] < 1e-12 else item["training_loss"],
                  item["candidate"].program.size))[:max_candidates]


def _role_coordinates(spec):
    """Public roles propose reductions; response evidence must select their use."""
    payload = spec.context.get("object_schema")
    if payload is None:
        return []
    schema = ObjectSchema.from_dict(json.loads(payload) if isinstance(payload, str) else payload)
    if not set(schema.feature_names) <= set(spec.feature_names):
        return []
    available = set.intersection(*(set(schema.fields[group]) for group in schema.objects))
    roles = sorted(available - {"f"}, key=lambda role: (role != "q", role))
    coordinates = []
    for role in roles:
        terms = [Program("feature", name=schema.fields[group][role]) for group in schema.objects]
        aggregate = Program("object_sum", children=tuple(terms))
        basis = [aggregate]
        if "f" in available:
            focused = Program("object_sum", children=tuple(
                Program("mul", children=(Program("feature", name=schema.fields[group]["f"]), term))
                for group, term in zip(schema.objects, terms)))
            basis.append(focused)
        coordinates.append((role, basis))
    return coordinates


def _retains_atom(program, atoms):
    """A joint revision must still use one of the computations it received."""
    return program in atoms or any(_retains_atom(child, atoms) for child in program.children)


def _joint_role_revisions(spec, fitter, raw, tolerance):
    """Retain callable mechanisms while jointly explaining a changed local law.

    A correction can require both a group reduction and a change of focal state.
    Fitting one residual term at a time can miss that joint explanation. The
    public-role blocks below are small competing hypotheses, and every fit uses
    all supplied target observations without changing the retained definitions.
    """
    coordinates = _role_coordinates(spec)
    if not coordinates:
        return []
    anchors = list(dict.fromkeys(term for fit in raw[:4] for term in fit["terms"]))
    parent = fitter.fit(anchors)
    if parent is None:
        return []
    # The raw callable fits are the first paid step; the joint refit is a
    # separate step even when it reaches zero residual immediately. Keeping
    # both steps makes the saved procedure describe the actual distinction
    # between fitting components and confirming their combination.
    first = {"operation": "fit_component",
             "question": "Fit the previously learned components before combining them",
             "program": raw[0]["program"], "loss": raw[0]["loss"]}
    joint = {"operation": "joint_callable_refit",
             "question": "Jointly refit the contributions of previously learned mechanisms",
             "program": parent["program"], "loss": parent["loss"],
             "parent_loss": raw[0]["loss"], "joint_terms": len(parent["terms"])}
    proposals = [(parent, [first, joint])]
    if parent["loss"] <= tolerance:
        return proposals
    temporal = max(len(row.history) for row in fitter.rows) >= 2
    for role, current in coordinates:
        changes = [Program("event_delta", lag=1, children=(term,)) for term in current] if temporal else []
        blocks = [(anchors, current, "current_reduction"),
                  (anchors, [*current, *changes], "reduction_and_event")]
        if temporal:
            for width in (2, 3, 4):
                windowed = [Program("window_sum", lag=width, children=(term,))
                            for term in [*current, *changes]]
                blocks.append((anchors, windowed, f"window_{width}_residual"))
                blocks.append(([Program("window_sum", lag=width, children=(term,)) for term in anchors],
                               windowed, f"window_{width}_joint_mechanism"))
        for retained, residual_terms, operation in blocks:
            fit = fitter.fit([*retained, *residual_terms])
            if fit is None or fit["loss"] + tolerance >= parent["loss"]:
                continue
            # A role-only model is not evidence that a previous mechanism was
            # locally repaired. The ordinary structural proposer handles those.
            if not _retains_atom(fit["program"], anchors):
                continue
            step = {"operation": "joint_role_residual",
                "question": "Explain the residual through measured group and focal roles",
                "program": fit["program"], "loss": fit["loss"],
                "parent_loss": parent["loss"], "role": role, "coordinate_operation": operation,
                "residual_rows": fitter.count, "joint_terms": len(fit["terms"])}
            proposals.append((fit, [first, step]))
            if fit["loss"] <= tolerance:
                return proposals
    return proposals


def _joint_temporal_roles(spec, fitter, raw, tolerance):
    """Separate measured state contributions from a retained temporal mechanism.

    Transforming an entire scalar call alone cannot isolate the change of its
    interaction component. Jointly transforming measured roles lets the evidence
    remove those contributions, while keeping the learned call and its internal
    relations unchanged. Distinct sources compete before repeated role bindings
    of one strongly correlated source can occupy all component slots.
    """
    coordinates = _role_coordinates(spec)
    if not coordinates or max(len(row.history) for row in fitter.rows) < 2:
        return []
    anchors, sources = [], set()
    for fitted in raw:
        for term in fitted["terms"]:
            source = tuple(_sources(term)) or term
            if source not in sources:
                anchors.append(term)
                sources.add(source)
        if len(anchors) >= 4:
            break
    parent = fitter.fit(anchors)
    if parent is None or parent["loss"] <= tolerance:
        return []
    role_terms = [term for _, terms in coordinates for term in terms]
    changes = [Program("event_delta", lag=1, children=(term,))
               for term in [*anchors, *role_terms]]
    alternatives = [(changes, "event"), ([*anchors, *changes], "current_and_event")]
    alternatives.extend(([Program("window_sum", lag=width, children=(term,))
                          for term in [*anchors, *changes]], f"window_{width}_current_and_event")
                        for width in (2, 3, 4))
    first = {"operation": "joint_callable_refit",
        "question": "Compare contributions from distinct previously learned mechanisms",
        "program": parent["program"], "loss": parent["loss"]}
    proposals = []
    for terms, operation in alternatives:
        fitted = fitter.fit(terms)
        if (fitted is None or fitted["loss"] + tolerance >= parent["loss"]
                or not _retains_atom(fitted["program"], anchors)):
            continue
        step = {"operation": "derive_temporal_mechanism",
            "question": "Jointly separate measured state contributions from the change of a learned mechanism",
            "program": fitted["program"], "loss": fitted["loss"],
            "parent_loss": parent["loss"], "roles": [role for role, _ in coordinates],
            "coordinate_operation": operation, "residual_rows": fitter.count,
            "joint_terms": len(fitted["terms"])}
        proposals.append((fitted, [first, step]))
        if fitted["loss"] <= tolerance:
            break
    return proposals


def _object_observation_lifts(spec, fitter, raw, tolerance):
    """Reuse a callable observation at each object, then compare its reduction.

    These are interventions on an executable function's membership arguments,
    not extra observations of the laboratory. Actual response rows must select
    the resulting hypothesis. The bounded search starts with instantaneous
    observations; temporal operators are applied after the object reduction.
    """
    coordinates = _role_coordinates(spec)
    if not coordinates:
        return []
    payload = spec.context["object_schema"]
    schema = ObjectSchema.from_dict(json.loads(payload) if isinstance(payload, str) else payload)
    if not all("f" in schema.fields[group] for group in schema.objects):
        return []
    membership = {schema.fields[group]["f"] for group in schema.objects}
    choices, source_counts = [], {}
    for fitted in raw:
        for call in fitted["terms"]:
            source = tuple(_sources(call))
            if (call.op != "call" or call.history_depth or not membership & call.required_features()
                    or source_counts.get(source, 0) >= 2):
                continue
            source_counts[source] = source_counts.get(source, 0) + 1
            choices.append((call, fitted))
        if len(choices) >= 8:
            break
    proposals = []
    temporal = max(len(row.history) for row in fitter.rows) >= 2
    role_terms = [term for _, terms in coordinates for term in terms]
    for call, parent in choices:
        instances = tuple(bind_measurement_arguments(call, {
            schema.fields[group]["f"]: Program("const", value=float(group == selected))
            for group in schema.objects}) for selected in schema.objects)
        lifted = Program("object_sum", children=instances)
        lifted_fit = fitter.fit([lifted])
        if lifted_fit is None:
            continue
        first = {"operation": "lift_object_observation",
            "question": "Apply the learned observation to each measured object and reduce its answers",
            "program": lifted_fit["program"], "loss": lifted_fit["loss"], "source_ids": _sources(call),
            "objects": list(schema.objects), "membership_fields": sorted(membership),
            "instances": len(instances)}
        alternatives = [(lifted_fit, lifted, "current")]
        if temporal and lifted_fit["loss"] > tolerance:
            changes = [Program("event_delta", lag=1, children=(term,)) for term in [call, *role_terms]]
            alternatives.append((fitter.fit([lifted, *changes]), lifted, "current_and_event"))
            for width in (2, 3, 4):
                lifted_window = Program("window_sum", lag=width, children=(lifted,))
                terms = [lifted_window, *(Program("window_sum", lag=width, children=(term,))
                                         for term in changes)]
                alternatives.append((fitter.fit(terms), lifted_window, f"window_{width}_current_and_event"))
        for fitted, required, operation in alternatives:
            if (fitted is None or required not in fitted["terms"]
                    or fitted["loss"] + tolerance >= raw[0]["loss"]):
                continue
            step = {"operation": "compose_lifted_observation",
                "question": "Fit the reduced observation jointly with measured temporal contributions",
                "program": fitted["program"], "loss": fitted["loss"],
                "parent_loss": parent["loss"], "coordinate_operation": operation,
                "residual_rows": fitter.count, "joint_terms": len(fitted["terms"])}
            proposals.append((fitted, [first, step]))
        if any(fitted["loss"] <= tolerance for fitted, _ in proposals):
            break
    return proposals


def propose_research_programs(spec, training, budget, calls, *, max_candidates=16):
    """Search a bounded residual-driven composition grammar over callable atoms.

    Each column, fit, residual measurement and executable comparison costs one
    pass through the supplied rows. Public object roles can supply a joint local
    mechanism correction; two generic rounds also permit temporal operations on
    an already learned composition.
    Candidate parents remain the actual source concepts, including nested calls.
    """
    rows = list(training)
    if len(rows) < 6 or max_candidates < 1:
        return []
    names = set.intersection(*(set(row.features) for row in rows))
    atoms = list(dict.fromkeys(call for call in calls
                 if call.required_features().issubset(names)))[:max_candidates]
    if not atoms:
        return []
    start = budget.spent
    fitter = _Fitter(rows, budget, spec.risk_quantile)
    raw = [fit for atom in atoms if (fit := fitter.fit((atom,))) is not None]
    if not raw:
        return []
    raw.sort(key=_rank)
    baseline = raw[0]
    tolerance = max(1e-10, abs(baseline["loss"]) * 1e-7)
    if baseline["loss"] <= tolerance:
        return []
    baseline_step = {"operation": "fit_component", "question": "Fit a previously learned component",
                     "program": baseline["program"], "loss": baseline["loss"]}
    paths = {fit["program"]: [{"operation": "fit_component",
             "question": "Fit a previously learned component", "program": fit["program"],
             "loss": fit["loss"]}] for fit in raw}
    found = {}
    beam = raw[:3]
    pool = list(dict.fromkeys(term for fit in raw[:8] for term in fit["terms"]))
    temporal = max(len(row.history) for row in rows) >= 2

    def retain(fit, path):
        if fit is None:
            return
        paths.setdefault(fit["program"], path)
        if fit["loss"] + tolerance < baseline["loss"]:
            old = found.get(fit["program"])
            if old is None or len(path) < len(old[1]):
                found[fit["program"]] = fit, path

    for fit, path in _object_observation_lifts(spec, fitter, raw, tolerance):
        retain(fit, path)
    if not found or min(fit["loss"] for fit, _ in found.values()) > tolerance:
        for fit, path in _joint_role_revisions(spec, fitter, raw, tolerance):
            retain(fit, path)
        if not found or min(fit["loss"] for fit, _ in found.values()) > tolerance:
            for fit, path in _joint_temporal_roles(spec, fitter, raw, tolerance):
                retain(fit, path)
    if found and min(fit["loss"] for fit, _ in found.values()) <= tolerance:
        ranked = sorted(found.values(), key=lambda item: _rank(item[0]))[:max_candidates]
        return [_entry(fit, path, len(rows), budget.spent - start, index)
                for index, (fit, path) in enumerate(ranked)]

    for depth in range(2):
        # Residual fits nominate additional coordinates. The complete joint fit
        # below then decides whether the new coordinate survives recombination.
        candidates = list(pool)
        if temporal:
            seeds = pool if depth == 0 else [fit["program"] for fit in beam[:2]]
            candidates.extend(term for seed in seeds for term in _temporal(seed))
        candidates.extend(Program("mul", children=(left, right))
                          for left, right in combinations(pool[:5], 2))
        candidates = list(dict.fromkeys(candidates))
        singles = []
        for term in candidates:
            fit = fitter.fit((term,))
            if fit is None:
                continue
            singles.append(fit)
            path = [baseline_step, {"operation": "transform_component",
                "question": "Test whether a temporal or interacting computation explains the residual",
                "program": fit["program"], "loss": fit["loss"]}]
            retain(fit, path)
        singles.sort(key=_rank)
        # Include original atoms even when the whole target correlates more
        # strongly with a transformed coordinate than with its small residual.
        options = list(dict.fromkeys([*pool,
            *(term for fit in singles[:12] for term in fit["terms"])]))
        beam = sorted({fit["program"]: fit for fit in [*beam, *singles[:3]]}.values(), key=_rank)[:3]
        for _ in range(2):
            expanded = list(beam)
            for parent in beam:
                if len(parent["terms"]) >= 3 or parent["loss"] <= tolerance:
                    continue
                if not budget.spend(len(rows)):
                    break
                residual = [row.target - value
                            for row, value in zip(rows, predict_rows(parent["program"], rows))]
                # All correlations use cached, paid columns. These are proposals;
                # only an actual joint fit can supply an accepted training answer.
                residual_center = sum(residual) / len(rows)
                eligible = [(term, fitter.columns[term]) for term in options
                            if term not in parent["terms"] and fitter.columns.get(term) is not None]
                # Fund these additional passes before computing correlations.
                if not budget.spend(len(rows) * len(eligible)):
                    break
                ranked = []
                for term, column in eligible:
                    center = sum(column) / len(rows)
                    norm = sum((value - center) ** 2 for value in column)
                    if norm <= 1e-12:
                        continue
                    covariance = sum((value - center) * (error - residual_center)
                                     for value, error in zip(column, residual))
                    ranked.append((covariance * covariance / norm, term))
                for _, term in sorted(ranked, key=lambda item: item[0], reverse=True)[:8]:
                    fit = fitter.fit((*parent["terms"], term))
                    if fit is None or fit["loss"] + tolerance >= parent["loss"]:
                        continue
                    step = {"operation": "explain_residual_and_recombine",
                            "question": "Fit a remaining component and jointly revise its contribution",
                            "program": fit["program"], "loss": fit["loss"],
                            "parent_loss": parent["loss"], "residual_rows": len(rows)}
                    path = [*paths.get(parent["program"], [baseline_step]), step]
                    retain(fit, path)
                    expanded.append(fit)
            beam = sorted({fit["program"]: fit for fit in expanded}.values(), key=_rank)[:3]
        if beam and beam[0]["loss"] <= tolerance:
            break
        pool = list(dict.fromkeys([*pool,
            *(term for fit in beam for term in fit["terms"])]))[:12]
    ranked = sorted(found.values(), key=lambda item: _rank(item[0]))[:max_candidates]
    return [_entry(fit, path, len(rows), budget.spent - start, index)
            for index, (fit, path) in enumerate(ranked)]


def replay_research_program(payload, training, budget, *, risk_quantile=.5,
                            feature_mapping=None):
    """Execute a saved operator plan and refit its answer on newly supplied rows.

    This reuses the research procedure's selected coordinates, not its old
    coefficients or validation verdict. A resumed proposal needs fresh external
    validation just like a newly generated one.
    """
    rows = list(training)
    if len(rows) < 6:
        return None
    if any(step["operation"] == "discover_decoded_conditional_coordinates"
           for step in payload.get("selection_steps", [])):
        from .decoded_conditional_coordinates import replay_decoded_conditional_coordinates
        return replay_decoded_conditional_coordinates(payload, rows, budget,
            risk_quantile=risk_quantile, feature_mapping=feature_mapping)
    if any(step["operation"] == "discover_composed_conditions"
           for step in payload.get("selection_steps", [])):
        from .composed_conditions import replay_composed_conditions
        return replay_composed_conditions(payload, rows, budget,
            risk_quantile=risk_quantile, feature_mapping=feature_mapping)
    if any(step["operation"] == "discover_conditional_coordinates"
           for step in payload.get("selection_steps", [])):
        from .conditional_coordinates import replay_conditional_coordinates
        return replay_conditional_coordinates(payload, rows, budget,
            risk_quantile=risk_quantile, feature_mapping=feature_mapping)
    graph = ResearchGraph.from_dict(payload["graph"])
    terms = [graph.nodes[key].answer for key in payload["term_nodes"]]
    if not terms or any(term is None for term in terms):
        return None
    saved = next((step for step in reversed(payload.get("selection_steps", []))
                  if step["operation"] == "resolve_relation_intervals"), None)
    parameter_terms = [Program.from_dict(term) for term in saved.get("parameter_terms", [])] if saved else []
    if (saved and not parameter_terms and len(terms) == 1 and
            terms[0] == Program.from_dict(saved["program"])):
        parameter_terms = [Program.from_dict(saved["parameter_template"])]
    if feature_mapping:
        terms = [bind_measurement_arguments(term, feature_mapping) for term in terms]
        parameter_terms = [bind_measurement_arguments(term, feature_mapping) for term in parameter_terms]
    start = budget.spent
    fitter = _Fitter(rows, budget, risk_quantile)
    fit = fitter.fit(terms)
    if fit is None:
        return None
    if saved and len(parameter_terms) == len(terms):
        from .relation_uncertainty import _parameter_relations
        unresolved = any(term not in fit["fit_terms"] and any(
            _parameter_relations(template, item["symbol"]) for item in saved["parameters"])
            for term, template in zip(terms, parameter_terms))
        if unresolved:
            # A correction can be zero on this pilot yet matter at its boundary.
            # Retest the saved complete response before removing that dependency.
            retained = Program.from_dict(saved["program"])
            template = Program.from_dict(saved["parameter_template"])
            if feature_mapping:
                retained = bind_measurement_arguments(retained, feature_mapping)
                template = bind_measurement_arguments(template, feature_mapping)
            complete_fit = fitter.fit([retained])
            if complete_fit is not None and _rank(complete_fit)[0] <= _rank(fit)[0]:
                fit, terms, parameter_terms = complete_fit, [retained], [template]
    step = {"operation": "refit_research_program",
            "question": "Retest the saved computation dependencies on new evidence",
            "program": fit["program"], "loss": fit["loss"]}
    steps = [step]
    if saved and len(parameter_terms) == len(terms):
        from .relation_uncertainty import _bind, _parameter_relations
        # Fitting removes constant columns and merges identical executable
        # terms. Follow the terms actually retained by that solve.
        mapped = {}
        for term, template in zip(terms, parameter_terms):
            mapped.setdefault(term, set()).add(template)
        if all(len(mapped[term]) == 1 for term in fit["fit_terms"]):
            actual_templates = [next(iter(mapped[term])) for term in fit["fit_terms"]]
            template = _sum_terms(actual_templates, fit["coefficients"][1:], fit["coefficients"][0])
            parameters = [item for item in saved["parameters"]
                          if _parameter_relations(template, item["symbol"])]
            if parameters:
                active = {item["symbol"] for item in parameters}
                fixed = {item["symbol"]: Program("const", value=item["radius"])
                         for item in saved["parameters"] if item["symbol"] not in active}
                contexts = []
                for context in saved.get("evidence_contexts", []):
                    if not active.intersection(context["parameters"]):
                        continue
                    context = deepcopy(context)
                    context["parameter_template"] = _bind(
                        Program.from_dict(context["parameter_template"]), fixed).to_dict()
                    context["parameters"] = {key: value for key, value in context["parameters"].items() if key in active}
                    contexts.append(context)
                steps.append({**deepcopy(saved), "program": fit["program"], "loss": fit["loss"],
                    "question": "Retest the surviving relation dependencies in the newly fitted complete response",
                    "parameter_template": template.to_dict(), "parameters": deepcopy(parameters),
                    "parameter_terms": [term.to_dict() for term in actual_templates],
                    "evidence_contexts": contexts})
    result = _entry(fit, steps, len(rows), budget.spent - start)
    result["research_program"]["reused_procedure"] = True
    return result
