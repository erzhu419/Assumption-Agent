"""Learn and revise executable research sequences from their later returns."""

from copy import deepcopy
from statistics import mean

from .method_conditions import method_applicability, same_research_condition
from .method_revision import distinguishing_experiment, revision_condition


def common_method_process_return(comparison):
    """Compare completed research under its original task/evaluation contract.

    Resource savings remain separate from prediction gain. Intervals are the
    retained account bounds, not estimates replacing unknown paid work.
    """
    from math import isfinite

    result = {'status': 'UNKNOWN', 'helpful': None, 'scope': comparison.get('scope'),
        'source_refs': deepcopy(comparison.get('source_refs', [])), 'resource_changes': {}}

    def unknown(reason):
        return {**result, 'reason': reason}

    def number(value):
        return type(value) in (int, float) and isfinite(value) and value >= 0

    if (not result['scope'] or comparison.get('same_task') is not True
            or comparison.get('same_evaluation_contract') is not True or not result['source_refs']):
        return unknown('matching_task_evaluation_and_original_references_required')
    before, after = comparison.get('before', {}), comparison.get('after', {})
    if (comparison.get('administrative_censoring') or comparison.get('censored')
            or any(item.get('administrative_censoring') or item.get('censored')
                or item.get('execution_complete') is not True for item in (before, after))):
        return unknown('incomplete_or_administratively_censored_research')
    if any(type(item.get('completed')) is not bool for item in (before, after)):
        return unknown('actual_task_completion_required')
    result['completion'] = {'before': before['completed'], 'after': after['completed']}
    quality = [item.get('quality', {}) for item in (before, after)]
    if (any(not number(item.get('loss')) or not number(item.get('threshold')) for item in quality)
            or quality[0]['threshold'] != quality[1]['threshold']):
        return unknown('same_original_quality_threshold_and_measured_losses_required')
    passed = [item['loss'] <= item['threshold'] for item in quality]
    result['quality'] = {'before_loss': quality[0]['loss'], 'after_loss': quality[1]['loss'],
        'threshold': quality[0]['threshold'], 'before_passed': passed[0], 'after_passed': passed[1],
        'basis': 'original_task_quality_threshold'}
    coverage = [item.get('coverage') for item in (before, after)]
    if any(type(value) is not int or value <= 0 for value in coverage):
        return unknown('measured_evaluation_coverage_required')
    result['coverage'] = {'before': coverage[0], 'after': coverage[1],
        'change': coverage[1] - coverage[0]}
    if (before['completed'] and not after['completed'] or passed[0] and not passed[1]
            or coverage[1] < coverage[0]):
        return {**result, 'status': 'measured', 'helpful': False,
            'reason': 'completed_research_degrades_completion_quality_or_coverage'}
    if not after['completed'] or not passed[1]:
        return unknown('comparable_completed_quality_not_established')

    resources = ['observation_cost', 'compute_cost']
    if any('measurement_cost' in item.get('costs', {})
           or 'measurement_cost' in item.get('cost_intervals', {}) for item in (before, after)):
        resources.append('measurement_cost')

    def interval(item, resource):
        bounds = item.get('cost_intervals', {}).get(resource)
        if bounds is not None:
            return list(bounds) if (isinstance(bounds, (list, tuple)) and len(bounds) == 2
                and all(number(value) for value in bounds) and bounds[0] <= bounds[1]) else None
        value = item.get('costs', {}).get(resource)
        return [value, value] if number(value) else None

    directions = []
    for resource in resources:
        first, second = interval(before, resource), interval(after, resource)
        delta = [second[0] - first[1], second[1] - first[0]] if first and second else None
        direction = ('lower' if delta[1] < 0 else 'higher' if delta[0] > 0
            else 'equal' if delta == [0, 0] else 'not_higher' if delta[1] <= 0
            else 'not_lower' if delta[0] >= 0 else 'UNKNOWN') if delta is not None else 'UNKNOWN'
        result['resource_changes'][resource] = {'before': first, 'after': second,
            'delta_interval': delta, 'direction': direction}
        directions.append(direction)
    if not before['completed'] or not passed[0]:
        return {**result, 'status': 'measured', 'helpful': True,
            'reason': 'completed_research_improves_task_completion_or_original_quality'}
    if 'UNKNOWN' in directions:
        return unknown('resource_order_not_established')
    if all(value in {'lower', 'equal', 'not_higher'} for value in directions) and 'lower' in directions:
        return {**result, 'status': 'measured', 'helpful': True,
            'reason': 'same_completed_quality_and_coverage_with_lower_resources'}
    if all(value in {'higher', 'equal', 'not_lower'} for value in directions):
        return {**result, 'status': 'measured', 'helpful': False,
            'reason': 'same_completed_quality_without_resource_gain'}
    return {**result, 'status': 'measured', 'reason': 'resource_tradeoff_requires_task_preference'}


def observe_common_method_return(feedback, state=None):
    """Adapt one retained Common use, without treating work scores as returns.

    ``feedback_ref`` identifies the original use across later confirmations.
    Only ``paired_future`` can score prediction gain: two frozen predictors on
    the same new paid readings, with the original result references. Comparable
    process returns remain a separate vector; the prediction action account
    covers known method fitting only and other original bills stay referenced.
    """
    from math import isfinite
    from .action_policy import ResearchActionPolicy
    from .models import TaskSpec

    state = deepcopy(state or {})
    state.setdefault('cost_scope', 'known_method_fit_only_other_original_bills_referenced')
    experience = state.setdefault('action_experience', {})
    origins = state.setdefault('origins', [])
    reference, method_ref, profile = (feedback['feedback_ref'], feedback['method_ref'], feedback['profile'])
    entry = next((item for item in origins if item['feedback_ref'] == reference), None)
    if entry is None:
        entry = {'feedback_ref': deepcopy(reference), 'method_ref': deepcopy(method_ref),
            'fit_compute_cost': None, 'cost_refs': [], 'status': 'UNKNOWN'}
        origins.append(entry)
    for cost_ref in feedback.get('cost_refs', []):
        if cost_ref not in entry['cost_refs']:
            entry['cost_refs'].append(deepcopy(cost_ref))
    process = feedback.get('process_comparison')
    if process is not None:
        measured_process = common_method_process_return(process)
        entry.update(process_comparison=deepcopy(process), process_return=measured_process)
        after = process.get('after', {})
        costs = {key: change['after'][0]
            for key, change in measured_process['resource_changes'].items()
            if change['after'] is not None and change['after'][0] == change['after'][1]}
        if (measured_process['status'] == 'measured'
                and measured_process.get('helpful') is not None
                and all(key in costs for key in ('observation_cost', 'compute_cost'))):
            methods = ResearchMethods({'methods': [{'method_id': method_ref['id'],
                'steps': ['replay_research_program']}]})
            entry['process_continuation_outcome'] = methods.observe_continuation(method_ref['id'], profile, {
                'gain': 0., 'normalized_gain': 0., 'reference_rows': after['coverage'],
                'observation_cost': costs['observation_cost'], 'compute_cost': costs['compute_cost'],
                'execution_complete': True, 'process_return': measured_process}, source_task=deepcopy(reference))
        else:
            entry.pop('process_continuation_outcome', None)
    cost = feedback.get('fit_compute_cost')
    if cost is not None:
        if type(cost) not in (int, float) or not isfinite(cost) or cost < 0:
            raise ValueError('Common method fit cost must be known nonnegative work or None')
        if entry['fit_compute_cost'] is not None and entry['fit_compute_cost'] != cost:
            raise ValueError('a retained method use cannot change its original fitting bill')
        entry['fit_compute_cost'] = cost
    cost = entry['fit_compute_cost']
    if cost is None:
        return state

    objective = profile['objective']
    quantile = objective['quantile']
    spec = TaskSpec('common_method_use', 'Public method-use objective',
        tuple(f'role_{index}' for index in range(profile['data']['feature_dimensions'])), (), (),
        risk_quantile=quantile, decision_costs=(quantile, 1. - quantile)
            if objective['loss'] == 'decision' else None)
    policy = ResearchActionPolicy([], spec, experience)
    action = 'replay_research_program'
    policy.set_action_knowledge(action, {'method_ref': method_ref, 'profile': profile})
    context = entry.setdefault('context', policy.action_context(action))
    paid = {'observation_cost': 0, 'measurement_cost': 0, 'compute_cost': cost}
    paired = feedback.get('paired_future') or {}
    measured = (feedback.get('execution_complete') is True and feedback.get('status') == 'completed'
        and not feedback.get('administrative_censoring') and not feedback.get('censored')
        and paired.get('same_rows') is True and paired.get('data_role') == 'new_paid_readings'
        and type(paired.get('reference_rows')) is int and paired['reference_rows'] > 0
        and bool(paired.get('source_refs')) and paired.get('execution_complete') is not False
        and all(type(paired.get(key)) in (int, float) and isfinite(paired[key]) and paired[key] >= 0
                for key in ('before_loss', 'after_loss')))
    if not measured:
        if 'trial_origin' not in entry and not entry.get('unscored'):
            outcome = policy.record_unscored_attempt(action, status='unresolved', context=context, **paid)
            entry.update(status='unscored', unscored=True, unscored_trial=outcome['trials'])
        return state
    if entry.get('paired_future') == paired:
        return state

    if 'trial_origin' not in entry:
        if entry.get('unscored'):
            record = experience[context][action]
            record['unscored_attempts'] -= 1
            record['unresolved_attempts'] -= 1
            for resource, amount in paid.items():
                record[f'unscored_{resource}'] -= amount
            entry['unscored'] = False
            record['last_unscored_trial'] = max((item['unscored_trial'] for item in origins
                if item.get('unscored') and item.get('context') == context), default=-1)
        outcome = policy.update(action, before_loss=paired['before_loss'], after_loss=paired['after_loss'],
            context=context, **paid)
        entry['trial_origin'] = {'context': context, 'action': action,
            'trial': outcome['trials'], 'credited_gain': outcome['last_gain']}
    status = ('supported' if paired['after_loss'] < paired['before_loss'] else
              'refuted' if paired['after_loss'] > paired['before_loss'] else 'unresolved')
    outcome = ResearchActionPolicy.revise_feedback(experience, entry['trial_origin'], status=status,
        before_loss=paired['before_loss'], after_loss=paired['after_loss'],
        evidence_rows=paired['reference_rows'])
    measured_gain = outcome['trial_outcomes'][str(entry['trial_origin']['trial'])]['gain']
    methods = ResearchMethods({'methods': [{'method_id': method_ref['id'], 'steps': [action]}]})
    continuation = methods.observe_continuation(method_ref['id'], profile, {
        'gain': paired['before_loss'] - paired['after_loss'], 'normalized_gain': measured_gain,
        'reference_rows': paired['reference_rows'], 'observation_cost': 0, 'compute_cost': cost,
        'execution_complete': True}, source_task=deepcopy(reference))
    entry.update(status='measured', paired_future=deepcopy(paired), continuation_outcome=continuation)
    return state


def observe_common_method_segment(segment, actual_return, *, actual_return_ref,
                                  profile, source_task, state=None):
    """Feed one measured whole Common segment to the existing sequence learner.

    Artifact reads remain prerequisites, not extra research actions. Individual
    step returns stay unscored: only the original whole-segment return is known.
    The caller supplies the already addressed return; this adapter reads no
    checkpoints and executes no research tool.
    """
    from math import isfinite

    actual_return = actual_return or {}
    book = ResearchMethods(state)
    result = {'status': 'UNKNOWN', 'actual_return_ref': deepcopy(actual_return_ref),
        'root_result_ref': deepcopy(segment.get('root_result_ref')),
        'source_task': deepcopy(source_task), 'method_id': None,
        'cost_refs': deepcopy(segment.get('cost_refs', [])), 'step_refs': [],
        'state': book.to_dict()}
    if (segment.get('missing_dependencies') or actual_return.get('execution_complete') is not True
            or actual_return.get('administrative_censoring') or actual_return.get('censored')
            or not actual_return_ref or not profile
            or any(type(actual_return.get(key)) not in (int, float)
                or not isfinite(actual_return[key]) for key in ('gain', 'observation_cost', 'compute_cost'))
            or any(actual_return[key] < 0 for key in ('observation_cost', 'compute_cost'))):
        return {**result, 'reason': 'complete_addressed_whole_segment_return_required'}
    nodes = segment.get('nodes', [])
    def address(reference):
        return reference.get('file'), tuple(reference.get('json_path', [])), reference.get('version')
    node_index = {address(node['step_ref']): node for node in nodes}
    root = segment.get('root_result_ref', {})
    terminal = next((node for node in nodes if node.get('is_execution') and
        node['step_ref'].get('file') == root.get('file') and
        root.get('json_path', [])[:len(node['step_ref'].get('json_path', []))]
            == node['step_ref'].get('json_path')), None)
    call = terminal.get('call_ref') if terminal else None
    if call is None:
        return {**result, 'reason': 'actual_current_call_execution_required'}
    executions = [node for node in nodes if node.get('is_execution') and node.get('call_ref') == call]
    trace, missing = [], []
    for node in executions:
        dependencies, inputs, pending, visited = [], [], list(node.get('depends_on_refs', [])), []
        while pending:
            reference = pending.pop()
            if reference in visited:
                continue
            visited.append(reference)
            ancestor = node_index.get(address(reference))
            if ancestor is None:
                missing.append(deepcopy(reference))
            elif ancestor.get('is_execution') and ancestor.get('call_ref') == call:
                if reference not in dependencies:
                    dependencies.append(deepcopy(reference))
            else:
                if reference not in inputs:
                    inputs.append(deepcopy(reference))
                pending.extend(ancestor.get('depends_on_refs', []))
        trace.append({'step_ref': deepcopy(node['step_ref']), 'action': node['action'],
            'context': 'common_research_segment', 'call_ref': deepcopy(call),
            'depends_on_refs': dependencies, 'input_refs': inputs,
            'gain': None, 'observation_cost': None, 'compute_cost': None,
            'execution_complete': node.get('execution_status') == 'completed',
            'cost_refs': deepcopy(node.get('cost_refs', [])), 'return_status': 'unscored_step'})
    if missing:
        return {**result, 'reason': 'exact_execution_ancestor_unavailable', 'missing_dependencies': missing}
    # Preserve the actual execution order within one saved call, including
    # steps whose individual contribution is unmeasured. A dependency is not
    # evidence authorizing their deletion from a measured whole sequence.
    trace.sort(key=lambda step: step['step_ref'].get('json_path', [None, 0])[1])
    roots = [entry['step_ref'] for entry in trace]
    origin = next((entry for entry in book.segment_outcomes
        if entry['actual_return_ref'] == actual_return_ref), None)
    returned = {**deepcopy(actual_return), 'actual_return_ref': deepcopy(actual_return_ref),
        'profile': deepcopy(profile), 'cost_refs': deepcopy(result['cost_refs'])}
    if origin is not None:
        if origin['root_result_ref'] != segment.get('root_result_ref'):
            return {**result, 'reason': 'one_return_cannot_join_different_executions'}
        if origin['return'] != returned:
            if any(origin['return'][key] != returned[key] for key in ('observation_cost', 'compute_cost')):
                return {**result, 'reason': 'original_complete_segment_bill_must_remain_unchanged'}
            identifier = origin.get('method_id')
            method = next((item for item in book.methods if item['method_id'] == identifier), None)
            if method is not None:
                method['observed_gain'] = returned['gain']
                method['status'] = 'revising' if returned['gain'] <= 1e-8 else 'proposed'
                method['sequence_outcomes'][0].update(gain=returned['gain'], profile=deepcopy(profile),
                    cost_refs=deepcopy(returned['cost_refs']))
            elif returned['gain'] > 1e-8:
                identifier = book.observe('common_research_segment', trace, condition={}, source_task=source_task,
                    step_refs=roots, segment_return=returned)
            origin['method_id'] = identifier
            origin['return'] = returned
            book.revisions.append({'method_id': identifier, 'change': 'revise_measured_segment_return',
                'actual_return_ref': deepcopy(actual_return_ref), 'gain': returned['gain'],
                'credit_scope': 'whole_segment_only_internal_steps_unscored'})
            return {**result, 'status': 'updated_measured_segment', 'method_id': identifier,
                'step_refs': [entry['step_ref'] for entry in trace], 'state': book.to_dict()}
        return {**result, 'status': 'reused_measured_segment', 'method_id': origin.get('method_id'),
            'step_refs': [entry['step_ref'] for entry in trace], 'state': book.to_dict()}
    identifier = book.observe('common_research_segment', trace, condition={}, source_task=source_task,
        step_refs=roots, segment_return=returned)
    book.segment_outcomes.append({'actual_return_ref': deepcopy(actual_return_ref),
        'root_result_ref': deepcopy(segment.get('root_result_ref')), 'source_task': deepcopy(source_task),
        'method_id': identifier, 'return': returned})
    return {**result, 'status': 'measured_segment', 'method_id': identifier,
        'step_refs': [entry['step_ref'] for entry in trace],
        'unscored_step_refs': [entry['step_ref'] for entry in trace],
        'observation': deepcopy(book.last_observation), 'state': book.to_dict()}


class ResearchMethods:
    def __init__(self, state=None):
        state = state or {}
        self.methods = deepcopy(state.get("methods", []))
        self.revisions = deepcopy(state.get("revisions", []))
        self.segment_outcomes = deepcopy(state.get('segment_outcomes', []))
        self.last_choice = None
        self.last_observation = None

    @staticmethod
    def _method_trace(method, trace):
        steps = [step for step in trace if step.get('method_id') == method['method_id']]
        if method.get('step_refs'):
            call = trace[-1].get('call_ref') if trace else None
            return [step for step in steps if call is not None and step.get('call_ref') == call]
        return steps

    @staticmethod
    def _completed(method, trace):
        return [step['action'] for step in ResearchMethods._method_trace(method, trace)]

    def _exact_steps(self, trace, step_refs):
        """Resolve explicit instances; action names never repair a missing link."""
        call = trace[-1].get('call_ref') if trace else None
        selected, missing, external = [], [], []
        pending = list(step_refs)
        if call is None:
            self.last_observation = {'status': 'UNKNOWN', 'reason': 'actual_call_boundary_required',
                'missing_dependencies': list(step_refs)}
            return None
        while pending:
            reference = pending.pop()
            matches = [index for index, step in enumerate(trace)
                if step.get('step_ref', step.get('action_ref')) == reference]
            if len(matches) != 1:
                if reference not in missing:
                    missing.append(deepcopy(reference))
                continue
            index, = matches
            if index in selected:
                continue
            selected.append(index)
            step = trace[index]
            if step.get('call_ref') != call:
                external.append(deepcopy(reference))
            pending.extend(step.get('depends_on_refs', []))
        self.last_observation = {'status': 'UNKNOWN' if missing else 'reference_closed',
            'call_ref': deepcopy(call), 'missing_dependencies': missing,
            'external_dependency_refs': external,
            'dependency_scope': 'actual_reads_not_causal_contribution'}
        if missing:
            return None
        return [trace[index] for index in sorted(selected) if trace[index].get('call_ref') == call]

    @staticmethod
    def _outcome(step, *, source_task, kind):
        return {"action": step["action"], "profile": deepcopy(step.get("profile")),
                "gain": step["gain"], "observation_cost": step["observation_cost"],
                "compute_cost": step["compute_cost"], "source_task": source_task, "kind": kind}

    def observe_knowledge_transfer(self, production, outcome):
        """Measure later use of a method's output, separately from the producing task.

        The outcome describes actually paid source-local work and paired
        prediction comparisons. It is an observed production route's return,
        rather than the causal effect of removing that method from all research.
        """
        method = next((item for item in self.methods if item["method_id"] == production["method_id"]), None)
        if method is None:
            return None
        record = {**deepcopy(outcome), "method_id": production["method_id"],
            "production_action": production["action"], "production_profile": deepcopy(production.get("profile")),
            "production_task": production["source_task"]}
        method.setdefault("knowledge_outcomes", []).append(record)
        self.revisions.append({"method_id": method["method_id"], "change": "measure_produced_knowledge_return",
            "source_concept_id": outcome["source_concept_id"], "downstream_task": outcome["downstream_task"],
            "retained_gain": outcome["retained_gain"], "exclusive_compute": outcome["exclusive_compute"]})
        return deepcopy(record)

    def _with_knowledge_return(self, method, action, profile, immediate):
        records = [row for row in method.get("knowledge_outcomes", []) if row["production_action"] == action]
        exact = [row for row in records if same_research_condition(row.get("production_profile"), profile, action)]
        matched = exact or [row for row in records if method_applicability(
            row.get("production_profile"), profile, action)["status"] == "supported"]
        conditions = []
        for row in matched:
            for condition in row["downstream_conditions"]:
                if condition not in conditions:
                    conditions.append(deepcopy(condition))
        # These are later consumer tasks. Their observed distribution prices
        # the output of a matching production step; it is not evidence that
        # the current producer has the same response or control coverage.
        knowledge = {"gain": mean(row["retained_gain"] or 0. for row in matched) if matched else 0.,
            "compute_cost": mean(row["exclusive_compute"] for row in matched) if matched else 0.,
            "trials": sum(row["retained_gain"] is not None for row in matched),
            "unmeasured_trials": sum(row["retained_gain"] is None for row in matched),
            "downstream_conditions": conditions,
            "basis": "observed_output_return_under_matching_production_conditions" if matched else "no_matching_output_return"}
        return {**immediate, "knowledge_return": knowledge,
            "investment_gain": immediate["gain"] + knowledge["gain"],
            "investment_compute_cost": immediate["compute_cost"] + knowledge["compute_cost"]}

    def _return(self, method, trace, profile, *, continuation=False):
        """Use returns acquired under the proposed step's evidence conditions.

        A later neutral result in a different condition does not erase an
        earlier useful result. A measured complete continuation replaces the
        source step's promise when pricing another continuation of that kind.
        """
        index = len(self._completed(method, trace))
        if index >= len(method["steps"]):
            return None
        action = method["steps"][index]
        records = [row for row in method.get("step_outcomes", []) if row["action"] == action]
        exact = [row for row in records if same_research_condition(row.get("profile"), profile, action)]
        matched = exact or [row for row in records if
            method_applicability(row.get("profile"), profile, action)["status"] == "supported"]
        transferred = [row for row in matched if row["kind"] == "transfer"]
        matched = transferred or matched
        basis = "conditioned_step_return" if matched else "source_sequence_return"
        sequence_gain = None
        if len(method["steps"]) - index > 1:
            sequences = method.get("sequence_outcomes", [])
            exact_sequences = [row for row in sequences if same_research_condition(row.get("profile"), profile, action)]
            applicable_sequences = exact_sequences or [row for row in sequences if
                method_applicability(row.get("profile"), profile, action)["status"] == "supported"]
            transferred_sequences = [row for row in applicable_sequences if row["kind"] == "transfer"]
            applicable_sequences = transferred_sequences or applicable_sequences
            if applicable_sequences:
                spent_gain = sum(step['gain'] for step in self._method_trace(method, trace)
                    if type(step.get('gain')) in (int, float))
                sequence_gain = max(-1., min(1., mean(row["gain"] for row in applicable_sequences) - spent_gain))
                basis = "remaining_sequence_return"
            elif not sequences:
                sequence_gain = method["observed_gain"]
        if continuation:
            returns = [row for row in method.get("continuation_outcomes", []) if
                       row["action"] == action and same_research_condition(row.get("profile"), profile, action)]
            if returns:
                matched, basis, sequence_gain = returns, "measured_continuation_return", None
        if matched:
            def work(row):
                source = row.get("profile") or {}
                ratio = ((profile or {}).get("observation_count", 0) / source["observation_count"]
                         if source.get("observation_count") and (profile or {}).get("observation_count") else 1.)
                return row["compute_cost"] * ratio
            return self._with_knowledge_return(method, action, profile, {
                    "gain": sequence_gain if sequence_gain is not None else mean(row["gain"] for row in matched),
                    "observation_cost": mean(row["observation_cost"] for row in matched),
                    "compute_cost": mean(work(row) for row in matched), "basis": basis,
                    "trials": len(matched), "transferred": any(row["kind"] == "transfer" for row in matched)})
        recorded = method.get("step_costs", [])
        costs = recorded[index] if index < len(recorded) else {
            "observation_cost": method["mean_observation_cost"], "compute_cost": method["mean_compute_cost"]}
        sources = method.get("step_profiles", [])
        source = sources[index] if index < len(sources) else None
        ratio = profile["observation_count"] / source["observation_count"] if (
            profile and source and source.get("observation_count")) else 1.
        return self._with_knowledge_return(method, action, profile, {
                "gain": method["observed_gain"], "observation_cost": costs["observation_cost"],
                "compute_cost": costs["compute_cost"] * ratio, "basis": basis, "trials": 0,
                "transferred": method["status"] == "supported" and not records})

    def observe_continuation(self, method_id, profile, feedback, *, source_task):
        """Learn what an executed method-backed revisit actually returned.

        The caller compares the saved and final predictors on common new
        evidence. This prices the complete revisit; it does not claim that its
        improvement was caused by this method rather than another action.
        """
        if not feedback.get("reference_rows") or feedback.get("execution_complete") is False:
            return None
        method = next(item for item in self.methods if item["method_id"] == method_id)
        record = {"action": feedback.get("method_action", method["steps"][0]),
                  "profile": deepcopy(profile), "gain": max(-1., min(1., feedback["normalized_gain"])),
                  "raw_gain": feedback["gain"], "reference_rows": feedback["reference_rows"],
                  "observation_cost": feedback["observation_cost"], "compute_cost": feedback["compute_cost"],
                  "source_task": source_task, "kind": "transfer"}
        if 'process_return' in feedback:
            record['process_return'] = deepcopy(feedback['process_return'])
        method.setdefault("continuation_outcomes", []).append(record)
        self.revisions.append({"method_id": method_id, "change": "measure_complete_continuation_return",
                               "gain": record["gain"], "source_task": source_task})
        return deepcopy(record)

    def _applicability(self, method, trace, profile):
        completed = self._completed(method, trace)
        if completed != method["steps"][:len(completed)] or len(completed) >= len(method["steps"]):
            return None
        action = method["steps"][len(completed)]
        if action in {"task_value", "representative"} and revision_condition(profile):
            # A measured joint plan can still buy necessary observations. A
            # standalone coverage promise cannot keep reselling a failed fit.
            if not any(step in {"representation", "continuous_state", "state_transition", "structure"}
                       for step in method["steps"][len(completed) + 1:]):
                return None
        profiles = method.get("step_profiles", [])
        source = profiles[len(completed)] if len(profiles) > len(completed) else None
        applicability = method_applicability(source, profile, action)
        if applicability["status"] == "explore":
            # A successful later transfer teaches a new applicability condition;
            # the method is not forever confined to its first source profile.
            supported = [method_applicability(row.get("profile"), profile, action)
                         for row in method.get("step_outcomes", [])
                         if row["action"] == action and row["gain"] > 0]
            applicability = next((item for item in supported if item["status"] == "supported"), applicability)
        if applicability["status"] == "unsupported":
            return None
        # Target feedback belongs to the action and evidence, not the name of
        # whichever source method happened to recommend it. A different method
        # ID cannot resell the same failed fit. An unfinished sequence can still
        # take a different next step after a neutral preparation.
        for step in reversed(trace):
            if step.get('gain') is None:
                continue  # An unmeasured internal step is not a failed attempt.
            if step["action"] != action or step["gain"] > 1e-8 or not step.get("execution_complete", True):
                continue
            prior = step.get("profile")
            if same_research_condition(prior, profile, action):
                return None
        for failed in method.get("failed_profiles", []):
            if same_research_condition(failed, profile, action):
                return None
        return applicability

    def observe(self, context, trace, *, condition, source_task, step_refs=None, segment_return=None):
        """Learn measured successful steps and their actual prerequisites.

        The shorter sequence is a method hypothesis. Its use in a later task
        supplies a separate return and can reject or specialize that hypothesis.
        """
        if not trace:
            return None
        exact = self._exact_steps(trace, step_refs) if step_refs is not None else None
        if step_refs is not None and exact is None:
            return None
        if segment_return is not None:
            if step_refs is None:
                raise ValueError('a whole segment return requires exact execution references')
            return self._observe_segment_return(context, exact, segment_return, condition, source_task)
        current = trace[-1]
        used = current.get("method_id")
        if used:
            method = next(item for item in self.methods if item["method_id"] == used)
            method["trials"] += 1
            if current.get("execution_complete", True):
                method.setdefault("step_outcomes", []).append(self._outcome(
                    current, source_task=source_task, kind="transfer"))
            measured_steps = self._method_trace(method, trace)
            if step_refs is not None:
                measured_steps = [step for step in measured_steps
                    if step.get('call_ref') == current.get('call_ref')]
            completed = [step["action"] for step in measured_steps] == method["steps"]
            step_index = len(measured_steps) - 1
            if step_index < len(method.get("step_costs", [])):
                for field in ("compute_cost", "observation_cost"):
                    old = method["step_costs"][step_index][field]
                    measured_cost = current[field]
                    profiles = method.get("step_profiles", [])
                    source_profile = profiles[step_index] if step_index < len(profiles) else None
                    target_profile = current.get("profile")
                    if (field == "compute_cost" and source_profile and target_profile
                            and target_profile.get("observation_count")):
                        measured_cost *= source_profile["observation_count"] / target_profile["observation_count"]
                    method["step_costs"][step_index][field] = .5 * old + .5 * measured_cost
            # A neutral preparatory action does not measure the return of an
            # unfinished sequence. Its dependent next step still needs a trial.
            if completed and all(step.get("execution_complete", True) for step in measured_steps):
                method["observed_gain"] = max(-1., min(1., sum(step["gain"] for step in measured_steps)))
                method.setdefault("sequence_outcomes", []).append({
                    "profile": deepcopy(measured_steps[0].get("profile")),
                    "gain": method["observed_gain"], "source_task": source_task, "kind": "transfer"})
                for field in ("compute_cost", "observation_cost"):
                    method[f"mean_{field}"] = .5 * method[f"mean_{field}"] + .5 * sum(
                        step[field] for step in measured_steps)
            if current["gain"] < -1e-8 and current.get("execution_complete", True):
                if current.get("profile"):
                    method.setdefault("failed_profiles", []).append(deepcopy(current["profile"]))
                else:
                    method["excluded_conditions"].append(deepcopy(condition))
                method["status"] = "revising"
                self.revisions.append({"method_id": used, "change": "restrict_failed_condition",
                                       "condition": deepcopy(condition), "gain": current["gain"]})
            elif current["gain"] > 1e-8 and current.get("execution_complete", True):
                method["confirmations"] += 1
                method["status"] = "supported"
        relevant = exact if step_refs is not None else [step for step in trace[-6:] if step["context"] == context]
        if not relevant or relevant[-1]["gain"] <= 1e-8 or not relevant[-1].get("execution_complete", True):
            return None
        original = [step['action'] for step in (trace if step_refs is not None else relevant)
            if step_refs is None or step.get('call_ref') == current.get('call_ref')]
        selected = (set(range(len(relevant))) if step_refs is not None else
            {index for index, step in enumerate(relevant)
                if step['gain'] > 1e-8 and step.get('execution_complete', True)})
        pending = list(selected)
        while pending:
            index = pending.pop()
            for action in (() if step_refs is not None else relevant[index].get("requires_actions", [])):
                dependency = next((before for before in range(index - 1, -1, -1)
                    if relevant[before]["action"] == action
                    and relevant[before].get("execution_complete", True)), None)
                if dependency is not None and dependency not in selected:
                    selected.add(dependency)
                    pending.append(dependency)
        useful_steps = [relevant[index] for index in sorted(selected)]
        # Repeated positive attempts without a dependency are still one step;
        # an explicitly linked preparation is kept in execution order.
        if step_refs is None and not any(step.get("requires_actions") for step in useful_steps):
            unique = {}
            for step in useful_steps:
                unique.setdefault(step["action"], step)
            useful_steps = list(unique.values())
        steps = [step["action"] for step in useful_steps]
        profiles = [deepcopy(step.get("profile")) for step in useful_steps]
        prior = next((item for item in self.methods if item["context"] == context
                      and item["condition"] == condition and item["steps"] == steps
                      and bool(item.get('step_refs')) == (step_refs is not None)
                      and all(revision_condition(old) == revision_condition(new)
                              and ((not old and not new) or method_applicability(old, new, action)["status"] == "supported")
                              for old, new, action in zip(item.get("step_profiles", profiles), profiles, steps))), None)
        if prior:
            if prior["method_id"] != used:
                prior.setdefault("step_outcomes", []).append(self._outcome(
                    current, source_task=source_task, kind="source"))
                prior["observed_gain"] = .5 * prior["observed_gain"] + .5 * relevant[-1]["gain"]
            return prior["method_id"]
        identifier = f"method_{len(self.methods)+1}"
        method = {"method_id": identifier, "context": context, "condition": deepcopy(condition),
                  "steps": steps, "original_steps": original, "source_task": source_task,
                  "observed_gain": relevant[-1]["gain"], "trials": 0, "confirmations": 0,
                  "excluded_conditions": [], "status": "proposed",
                  "mean_compute_cost": sum(step["compute_cost"] for step in useful_steps),
                  "mean_observation_cost": sum(step["observation_cost"] for step in useful_steps)}
        method["step_profiles"] = profiles
        if step_refs is not None:
            method.update(step_refs=[deepcopy(step.get('step_ref', step.get('action_ref')))
                for step in useful_steps], source_call_ref=deepcopy(current['call_ref']),
                dependency_refs=[deepcopy(step.get('depends_on_refs', [])) for step in useful_steps],
                external_dependency_refs=deepcopy(self.last_observation['external_dependency_refs']))
        method["sequence_outcomes"] = [{"profile": deepcopy(profiles[0]),
            "gain": method["observed_gain"], "source_task": source_task, "kind": "source"}]
        method["step_outcomes"] = [self._outcome(step, source_task=source_task, kind="source")
                                   for step in useful_steps]
        method["step_costs"] = [{key: step[key] for key in ("observation_cost", "compute_cost")}
                                for step in useful_steps]
        self.methods.append(method)
        self.revisions.append({"method_id": identifier, "change": 'propose_reference_closed_sequence'
                               if step_refs is not None else "remove_unproductive_steps" if steps != original
                               else "retain_successful_sequence", "before": original, "after": steps,
                               "source_task": source_task})
        return identifier

    def _observe_segment_return(self, context, trace, returned, condition, source_task):
        """Credit the measured sequence; no internal step acquires its gain."""
        self.last_observation.update(return_scope='whole_segment',
            actual_return_ref=deepcopy(returned['actual_return_ref']),
            unscored_step_refs=[deepcopy(step['step_ref']) for step in trace])
        if returned['gain'] <= 1e-8:
            self.last_observation.update(status='measured_nonpositive_segment')
            return None
        steps = [step['action'] for step in trace]
        identifier = f'method_{len(self.methods) + 1}'
        sequence = {'profile': deepcopy(returned['profile']), 'gain': returned['gain'],
            'observation_cost': returned['observation_cost'], 'compute_cost': returned['compute_cost'],
            'actual_return_ref': deepcopy(returned['actual_return_ref']),
            'cost_refs': deepcopy(returned.get('cost_refs', [])), 'source_task': source_task, 'kind': 'source'}
        self.methods.append({'method_id': identifier, 'context': context, 'condition': deepcopy(condition),
            'steps': steps, 'original_steps': steps, 'source_task': source_task, 'status': 'proposed',
            'observed_gain': returned['gain'], 'trials': 0, 'confirmations': 0, 'excluded_conditions': [],
            'step_refs': [deepcopy(step['step_ref']) for step in trace],
            'source_call_ref': deepcopy(trace[-1]['call_ref']),
            'step_profiles': [None for step in trace], 'step_outcomes': [],
            'unscored_steps': [{key: deepcopy(step[key]) for key in
                ('step_ref', 'action', 'gain', 'observation_cost', 'compute_cost', 'cost_refs', 'input_refs')}
                for step in trace], 'sequence_outcomes': [sequence], 'return_scope': 'whole_segment',
            'mean_observation_cost': returned['observation_cost'], 'mean_compute_cost': returned['compute_cost']})
        self.revisions.append({'method_id': identifier, 'change': 'propose_reference_closed_sequence',
            'source_task': source_task, 'actual_return_ref': deepcopy(returned['actual_return_ref']),
            'credit_scope': 'whole_segment_only_internal_steps_unscored'})
        return identifier

    def revise_plan(self, profile, options, trace=()):
        """Offer an affordable structural successor after repeated source failure.

        The caller supplies its executable, already priced options. This is a
        current-evidence revision, not an invented positive return for an
        untried method. Fresh feedback will determine whether it becomes one.
        """
        condition = revision_condition(profile)
        if not condition:
            return None
        eligible = {option.name: option for option in options
                    if method_applicability(None, profile, option.name)["can_try"]}
        structural = ["structure", "structural_discovery"] if profile.get("data", {}).get("object_groups") else []
        structural += ["representation", "state_transition", "continuous_state"]
        successor = next((name for name in structural if name in eligible), None)
        if successor is None:
            return None
        excluded = ["refit"]
        for action in ("task_value", "representative"):
            if action in eligible and not distinguishing_experiment(eligible[action]):
                excluded.append(action)
        return {"action": successor, "excluded_actions": excluded,
                "reason": "repeated_parameter_failure_requires_a_different_representation",
                "applicability": {"status": "supported", "condition": condition},
                "observation_cost": eligible[successor].observation_cost,
                "compute_cost": eligible[successor].compute_cost}

    def choose(self, context, condition, options, trace, *, task_id, observation_remaining, compute_remaining,
               profile=None, ordinary=None, ordinary_evidence=None):
        self.last_choice = None
        eligible = {option.name: option for option in options if option.observation_cost <= observation_remaining
                    and option.compute_cost <= compute_remaining}
        def unfinished(method):
            completed = self._completed(method, trace)
            return bool(completed and completed == method["steps"][:len(completed)]
                        and len(completed) < len(method["steps"]))

        methods = [item for item in self.methods if item["context"] == context
                   and item["condition"] == condition and item["source_task"] != task_id
                   and condition not in item["excluded_conditions"]
                   and (self._return(item, trace, profile) and self._return(item, trace, profile)["investment_gain"] > 0
                        or unfinished(item))]
        def value(method):
            estimate = self._return(method, trace, profile)
            share = max(estimate["observation_cost"] / max(1, observation_remaining),
                        estimate["investment_compute_cost"] / max(1, compute_remaining), 1e-8)
            return estimate["investment_gain"] / share
        for method in sorted(methods, key=lambda item: (
                not unfinished(item),
                next((index for index, step in enumerate(trace) if step.get("method_id") == item["method_id"]), 0),
                -value(item))):
            applicability = self._applicability(method, trace, profile)
            if applicability is None:
                continue
            completed = self._completed(method, trace)
            if completed != method["steps"][:len(completed)] or len(completed) >= len(method["steps"]):
                continue
            next_action = method["steps"][len(completed)]
            if next_action not in eligible:
                continue
            measured_return = self._return(method, trace, profile)
            if not completed and ordinary and ordinary.name != next_action:
                if applicability["status"] == "explore" or not (measured_return["transferred"]
                        or measured_return["knowledge_return"]["trials"] > 0 and measured_return["knowledge_return"]["gain"] > 0):
                    # Source success proposes a method. Its first use can be
                    # measured alongside a matching ordinary action; it does
                    # not yet justify replacing a different available plan.
                    continue
                ordinary_probe = ordinary.payload[1] if ordinary.payload else {}
                proposed_evidence = eligible[next_action].payload[1] if eligible[next_action].payload else {}
                if (ordinary_probe.get("unresolved_boundary_width", 0) > 0
                        and ordinary_probe.get("boundary_support")
                        and not proposed_evidence.get("boundary_support")):
                    continue
                if (ordinary_evidence and ordinary_evidence.get("expected_gain", 0) is not None
                        and ordinary_evidence["expected_gain"] > 0):
                    proposed_share = max(measured_return["observation_cost"] / max(1, observation_remaining),
                                         measured_return["investment_compute_cost"] / max(1, compute_remaining), 1e-8)
                    ordinary_share = ordinary_evidence.get("dominant_resource_share", 1.)
                    if measured_return["investment_gain"] / proposed_share <= ordinary_evidence["expected_gain"] / max(ordinary_share, 1e-8):
                        continue
                if (next_action in {"representative", "task_value"}
                        and (ordinary_probe.get("candidate_disagreement", 0) > 0
                             or ordinary_probe.get("expected_risk_reduction", 0) > 0)
                        and not (proposed_evidence.get("candidate_disagreement", 0) > 0
                                 or proposed_evidence.get("expected_risk_reduction", 0) > 0)):
                    # Current rival predictions support a concrete experiment;
                    # source-wide coverage returns cannot displace that probe.
                    continue
            self.last_choice = {"method_id": method["method_id"], "step": len(completed),
                                "steps": list(method["steps"]), "status": method["status"],
                                "applicability": applicability,
                                "return_estimate": measured_return,
                                "changes_action": ordinary is None or ordinary.name != next_action}
            return eligible[next_action]
        return None

    def opportunity(self, context, condition, *, observation_remaining, compute_remaining, task_id=None, trace=(),
                    profile=None, maximum_gain=None, visit_observation_budget=None, visit_compute_budget=None,
                    visit_quote=None):
        """Estimate a concrete next method using its acquired cost and return."""
        estimates, plans = {}, {}
        def estimate(method):
            identifier = method["method_id"]
            if identifier not in estimates:
                estimates[identifier] = self._return(method, trace, profile, continuation=True)
            return estimates[identifier]
        def plan(method):
            identifier = method["method_id"]
            if identifier not in plans:
                plans[identifier] = visit_quote(method, estimate(method))
            return plans[identifier]
        def costs(method):
            if visit_quote is not None:
                proposed = plan(method)
                return proposed["observation_budget"], proposed["compute_budget"]
            value = estimate(method)
            return value["observation_cost"], value["compute_cost"]
        def investment_compute(method):
            return costs(method)[1] + estimate(method)["knowledge_return"]["compute_cost"]
        choices = [method for method in self.methods if method["context"] == context
                   and method["source_task"] != task_id
                   and method["condition"] == condition and condition not in method["excluded_conditions"]
                   and estimate(method) is not None and estimate(method)["investment_gain"] > 0
                   and costs(method)[0] <= observation_remaining
                   and costs(method)[1] <= compute_remaining]
        choices = [method for method in choices if
                   self._completed(method, trace) == method['steps'][:len(self._completed(method, trace))]
                   and len(self._completed(method, trace)) < len(method['steps'])]
        choices = [method for method in choices if self._applicability(method, trace, profile) is not None
                   and (not profile or self._completed(method, trace)
                        or self._applicability(method, trace, profile)["status"] == "supported")
                   and (visit_observation_budget is None or costs(method)[0] <= visit_observation_budget)
                   and (visit_compute_budget is None or costs(method)[1] <= visit_compute_budget)]
        if not choices:
            return None
        def gain(method):
            return min(estimate(method)["gain"], maximum_gain) if maximum_gain is not None else estimate(method)["gain"]
        def investment_gain(method):
            return gain(method) + estimate(method)["knowledge_return"]["gain"]
        def value(method):
            share = max(costs(method)[0] / max(1, observation_remaining),
                        investment_compute(method) / max(1, compute_remaining), 1e-8)
            return investment_gain(method) / share
        best = max(choices, key=value)
        if investment_gain(best) <= 0:
            return None
        result = {"method_id": best["method_id"], "steps": best["steps"],
                "value": value(best), "expected_gain": gain(best),
                "expected_knowledge_gain": estimate(best)["knowledge_return"]["gain"],
                "investment_gain": investment_gain(best), "investment_compute_cost": investment_compute(best),
                "source_gain": best["observed_gain"], "return_estimate": estimate(best), "applicability": self._applicability(best, trace, profile),
                "observation_cost": costs(best)[0], "compute_cost": costs(best)[1]}
        if visit_quote is not None:
            result["research_plan"] = deepcopy(plan(best))
        return result

    def to_dict(self):
        result = {"methods": self.methods, "revisions": self.revisions}
        if self.segment_outcomes:
            result['segment_outcomes'] = self.segment_outcomes
        return deepcopy(result)
