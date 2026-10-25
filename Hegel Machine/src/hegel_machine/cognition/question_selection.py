"""Choose a declared plan by task-relevant finite-pool prediction differences."""

from copy import deepcopy
from itertools import combinations
from math import isfinite
from random import Random

from .evaluation import predict
from .language import Program
from .models import Observation
from .question_queries import evaluate_on_model, numeric_answer_distance
from .question_semantics import native_evaluation_contract
from .retained_forecasts import retained_forecast_predictor
from .selection import retain_research_comparison_pool
from .world_experiment_choice import forecast_equivalence_classes


def _frozen(value):
    if isinstance(value, dict):
        return tuple((key, _frozen(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item) for item in value)
    return value


def _retain_question_research_comparison(result, evaluated, programs, plans, question, comparison):
    """Choose comparison votes from the already charged task/panel cache."""
    assessed, unknown = [], deepcopy(comparison.get('unknown', []))
    for row in comparison['assessed']:
        identifier = row.candidate.candidate_id
        if identifier not in evaluated:
            continue  # Scientific, context and unit exclusions remain exclusions.
        if programs.get(identifier) != row.candidate.program:
            unknown.append({'id': identifier, 'reason': 'prepared_program_assessment_mismatch'})
            continue
        assessed.append(row)
    assessed_ids = {row.candidate.candidate_id for row in assessed}
    mandatory = set(comparison.get('mandatory_theory_ids', []))
    mandatory.update(question.get('candidate_theory_ids') or [])
    query = question.get('query_semantics', {}).get('query_expression', {})
    if query.get('op') == 'FindDistinguishingExperiment':
        mandatory.update(ref['id'] for ref in query.get('claim_refs', []))
    # Opaque Python columns and models without a qualified same-scope score
    # keep their original votes, even if that exceeds the research subpool cap.
    mandatory.update(identifier for identifier in evaluated if identifier not in assessed_ids)
    reported_unknown = {entry['id'] for entry in unknown}
    unknown.extend({'id': identifier, 'reason': 'no_qualified_same_scope_assessment'}
                   for identifier in evaluated
                   if identifier not in assessed_ids and identifier not in reported_unknown)
    public_predictions = {}
    for row in assessed:
        entry = evaluated[row.candidate.candidate_id]
        public_predictions[row.candidate.program.expanded()] = (
            tuple(_frozen(entry['predictions'][plan['id']]) for plan in plans)
            if entry['known'] and all(plan['id'] in entry['predictions'] for plan in plans) else None)
    retained, details = retain_research_comparison_pool(
        assessed, public_predictions, maximum=comparison['maximum'])
    selected_ids = mandatory | {row.candidate.candidate_id for row in retained}
    selected = {identifier: entry for identifier, entry in evaluated.items() if identifier in selected_ids}
    details.update({key: deepcopy(comparison[key]) for key in (
        'status', 'scope', 'assessment_refs', 'assessment_basis_ref', 'row_count') if key in comparison})
    details.update({'mandatory_theory_ids': [identifier for identifier in evaluated if identifier in mandatory],
        'selected_theory_ids': list(selected), 'selected_theory_count': len(selected),
        'eligible_theory_count': len(evaluated), 'unknown': unknown,
        'assessment_scope': 'qualified_assessed_subpool',
        'all_pool_optimality_established': False, 'support_updated': False})
    result['research_comparison'] = details
    return selected


def _cost(plan):
    return tuple(plan.get(key, 0) for key in ('observation_cost', 'compute_cost', 'model_cost'))


def validate_exploration_policy(policy):
    if policy is not None and (not isinstance(policy, dict)
            or policy.get('kind') != 'seeded_legal' or type(policy.get('seed')) is not int):
        raise ValueError('exploration_policy needs kind seeded_legal and an integer seed')
    return policy


def _physical_plan(plan):
    branches = [_frozen({'when': branch['when'], 'actions': branch['actions']})
                for branch in plan.get('branches', [])]
    # Branches are matched by their conditions, not by their declaration order.
    return _frozen({'actions': plan['actions'], 'branches': sorted(branches, key=repr)})


def _hard_prediction_failure(answer):
    return (answer.get('status') == 'invalid_experiment_binding' or answer.get('reason') in {
        'premise_unknown', 'unknown_due_to_uncertain_mapping',
        'unknown_due_to_opaque_program_semantics', 'undefined_distribution_or_answer_test'}
        or answer.get('detail') == 'context_ref')


def _partial_prediction_scores(result, evaluated, plans, answer_atol):
    """Use one task-known pair universe; missing cells never become negative evidence."""
    members = {}
    for identifier, entry in evaluated.items():
        if not entry.get('task_known'):
            result['groups'].append({'id': f"group_{len(result['groups']):03d}",
                'theory_ids': [identifier], 'known': False,
                'unknown_reason': entry.get('unknown', entry['task']).get('reason')})
            continue
        signature = (_frozen(entry['task']['value']), entry['task'].get('units'), tuple(
            _frozen(entry['predictions'][plan['id']]) if plan['id'] in entry['predictions']
            else ('unknown', entry['prediction_key']) for plan in plans))
        if signature in members:
            members[signature]['theory_ids'].append(identifier)
            continue
        group = {'id': f"group_{len(result['groups']):03d}", 'theory_ids': [identifier],
            'task_answer': deepcopy(entry['task']['value']), 'known': True,
            'panel_predictions': deepcopy(entry['predictions']),
            'prediction_status': deepcopy(entry['prediction_status'])}
        members[signature] = group
        result['groups'].append(group)
    groups = [group for group in result['groups'] if group['known']]
    comparable, unknown, separated = ({plan['id']: [] for plan in plans} for _ in range(3))
    universe, common = [], []
    for left, right in combinations(groups, 2):
        pair = [left['id'], right['id']]
        distance = numeric_answer_distance(left['task_answer'], right['task_answer'])
        relevant = (distance > answer_atol if distance is not None and answer_atol is not None
                    else left['task_answer'] != right['task_answer'])
        if relevant:
            universe.append(pair)
        known_plans, unknown_plans, distinguishing = [], [], []
        for plan in plans:
            name = plan['id']
            if name not in left['panel_predictions'] or name not in right['panel_predictions']:
                unknown_plans.append(name)
                if relevant:
                    unknown[name].append(pair)
                continue
            known_plans.append(name)
            if relevant:
                comparable[name].append(pair)
            if _separates(left['panel_predictions'][name], right['panel_predictions'][name], plan):
                distinguishing.append(name)
                if relevant:
                    separated[name].append(pair)
        if relevant and not unknown_plans:
            common.append(pair)
        result['pairwise_edges'].append({'left': left['id'], 'right': right['id'],
            'different_task_answer': relevant, 'experiments': distinguishing,
            'comparable_experiments': known_plans, 'unknown_experiments': unknown_plans})
    task_unknown = [identifier for identifier, entry in evaluated.items() if not entry.get('task_known')]
    blocked = [plan['id'] for plan in plans if any(
        _hard_prediction_failure(entry.get('prediction_status', {}).get(plan['id'], {}))
        for entry in evaluated.values())]
    task_blocked = any(_hard_prediction_failure(entry['task']) for entry in evaluated.values())
    complete = not task_unknown and not blocked and not task_blocked and all(
        len(group['panel_predictions']) == len(plans) for group in groups)
    result['prediction_coverage'] = {'scope': 'task_answer_known_subpool', 'complete': complete,
        'task_unknown_theory_ids': task_unknown, 'pair_denominator': len(universe),
        'task_relevant_pairs': universe, 'common_comparable_pairs': common,
        'common_comparable_pair_count': len(common), 'contract_blocked_plans': blocked,
        'task_contract_blocked': task_blocked,
        'plans': [{'id': plan['id'], 'comparable_pairs': comparable[plan['id']],
                   'unknown_pairs': unknown[plan['id']]} for plan in plans]}
    for plan in plans:
        name = plan['id']
        result['scores'].append({'experiment_id': name, 'separated_pairs': len(separated[name]),
            'effective_pairs': len(universe), 'comparable_pairs': len(comparable[name]),
            'unknown_pairs': len(unknown[name]), 'common_comparable_pairs': len(common),
            'G': len(separated[name]) / max(1, len(universe)),
            'G_upper': (len(separated[name]) + len(unknown[name])) / max(1, len(universe)),
            'selection_basis': 'confirmed_separation_lower_bound', 'partial': not complete,
            'declared_cost': list(_cost(plan))})
    if not complete:
        result['unknown_reason'] = 'incomplete_candidate_predictions'
    return universe, blocked, task_blocked


def _tolerance(value):
    if type(value) not in (int, float) or not isfinite(value) or value < 0:
        raise ValueError('forecast tolerance must be a declared finite nonnegative number')
    return value


def _separates(left, right, plan):
    if numeric_answer_distance(left[0], right[0]) > 2 * plan['tolerance']:
        return True
    # Different second actions are not observations of the same experiment.
    if len(left) == 3 and left[1] == right[1]:
        branch = plan['branches'][left[1]]
        return numeric_answer_distance(left[2], right[2]) > 2 * branch.get('tolerance', plan['tolerance'])
    return False


def _query_matches_actions(query, actions, prediction_rows=None):
    if not isinstance(query, dict):
        return False
    if query.get('op') == 'Readout':
        if 'finite_readouts' in query:
            return bool(query['finite_readouts']) and all(
                _query_matches_actions(case, actions, prediction_rows) for case in query['finite_readouts'])
        binding = query.get('binding', {})
        if not isinstance(binding, dict) or binding.get('action') not in actions:
            return False
        if prediction_rows is None:
            return True
        try:
            rows = prediction_rows(binding['action'])
        except (ValueError, TypeError):
            return False
        step = query.get('step')
        if type(step) is not int or not 0 <= step < len(rows):
            return False
        row = rows[step]
        return binding.get('features') == row.features and binding.get('history') == list(row.history)
    if query.get('op') == 'Difference':
        return all(_query_matches_actions(query.get(side), actions, prediction_rows)
                   for side in ('left', 'right'))
    if query.get('op') in {'RelationResidual', 'CheckApplication'}:
        return _query_matches_actions(query.get('query'), actions, prediction_rows)
    if query.get('op') == 'AllOnFiniteSet':
        return all(_query_matches_actions(case, actions, prediction_rows)
                   for case in query.get('finite_cases', []))
    return True  # PR1 reports unimplemented query semantics as UNKNOWN.


def diagnose_experiment_ties(selection, experiment_ids=None):
    """Compare existing predictions and edges without choosing between ties.

    Exact forecast partitions and threshold separation edges are different
    diagnostics. Threshold proximity is never made transitively equivalent.
    Full forecast signatures retain a conditional plan's branch identity.
    """
    identifiers = list(selection['tied_experiments'] if experiment_ids is None else experiment_ids)
    groups = [group for group in selection['groups'] if group['known'] and all(
        name in group['panel_predictions'] for name in identifiers)]
    unknown = [group['id'] for group in selection['groups'] if group not in groups]
    result = {'scope': 'completed_candidate_predictions',
        'status': 'partial_unknown' if unknown else 'complete', 'unknown_group_ids': unknown,
        'known_signature_group_count': len(groups),
        'known_theory_count': sum(len(group['theory_ids']) for group in groups),
        'experiments': [], 'comparisons': []}
    if not groups:
        result.update(status='scoped_unknown', reason='no_completed_prediction_groups')
        return result
    scores = {score['experiment_id']: score for score in selection['scores']}
    separated, relevant, partitions = {}, {}, {}
    for identifier in identifiers:
        edges = [edge for edge in selection['pairwise_edges'] if identifier in edge['experiments']]
        separated[identifier] = {(edge['left'], edge['right']) for edge in edges}
        relevant[identifier] = {(edge['left'], edge['right']) for edge in edges
                                if edge['different_task_answer']}
        partitions[identifier] = forecast_equivalence_classes([{'predictions': {
            group['id']: _frozen(group['panel_predictions'][identifier]) for group in groups}}])
        result['experiments'].append({'id': identifier, 'G': scores[identifier]['G'],
            'task_relevant_covered_pairs': len(relevant[identifier]),
            'all_separated_pairs': len(separated[identifier]),
            'exact_prediction_partition': partitions[identifier]})
    for left, right in combinations(identifiers, 2):
        task_difference = sorted(relevant[left] ^ relevant[right])
        edge_difference = sorted(separated[left] ^ separated[right])
        predictions_differ = [group for group in groups if
            _frozen(group['panel_predictions'][left]) != _frozen(group['panel_predictions'][right])]
        witness = predictions_differ[0] if predictions_differ else None
        result['comparisons'].append({'left': left, 'right': right,
            'same_G': scores[left]['G'] == scores[right]['G'],
            'same_task_relevant_coverage': not task_difference,
            'same_pairwise_separation': not edge_difference,
            'same_exact_prediction_partition': partitions[left] == partitions[right],
            'same_candidate_predictions': not predictions_differ,
            'different_task_pair_count': len(task_difference),
            'different_separation_pair_count': len(edge_difference),
            'task_pair_difference_witness': task_difference[0] if task_difference else None,
            'separation_pair_difference_witness': edge_difference[0] if edge_difference else None,
            'different_prediction_group_count': len(predictions_differ),
            'different_prediction_theory_count': sum(len(group['theory_ids']) for group in predictions_differ),
            'prediction_difference_witness': {'group_id': witness['id'],
                'theory_ids': deepcopy(witness['theory_ids']),
                'left': deepcopy(witness['panel_predictions'][left]),
                'right': deepcopy(witness['panel_predictions'][right])} if witness else None})
    return result


def select_question_plan(question, theories, experiments, *, context=None,
                         compute_budget=None, observation_budget=None,
                         public_prediction_rows=None, exploration_policy=None,
                         research_comparison=None):
    """Prepare an action without fitting, acquiring evidence, or changing support.

    ``tolerance`` bounds each forecast, so two forecasts separate beyond twice
    that bound. Exact answer-and-panel signatures have one vote in this call;
    their source theories remain individually identified. No posterior is used.
    A two-step plan supplies branches with ``when: {value, atol}``, actions and
    a query; only the first action is returned for immediate execution.

    Scalar task answers differ when their absolute distance exceeds the
    declared ``answer_resolution_ref.atol``; finite vectors apply the same
    bound to every declared coordinate. Comparisons remain pairwise and
    never merge a chain of nearby signatures. Without atol, equality is exact.
    An explicit ``query_semantics.evaluation`` native_evaluation contract uses
    the retained finite fallback through the existing native predictor. Without
    that contract, missing history remains an unknown model prediction.

    An explicitly checked direct task action has ``direct_task_action: True``
    and ``support: {status: 'supported', evidence: [...]}``. Its support is the
    caller's existing evidence record, never a conclusion of this selector.

    When supplied, the laboratory's ``public_prediction_rows(action)`` checks
    each experimental readout's step, features and history before scoring.
    Its absence is recorded; action membership alone does not verify those rows.
    Distribution answers require an authored distribution/noise test and remain
    UNKNOWN here; unequal empirical weights are not a stable discriminator.
    A prediction-based tie includes ``tie_diagnostics`` comparing coverage,
    pairwise separation, exact partitions and candidate predictions separately.
    ``seeded_legal`` permits a marked operational choice within epistemic ties
    and uses a common task-known pair denominator for incomplete predictions.
    ``research_comparison`` reuses scoped paid assessments and cached panel
    forecasts to retain research alternatives. Explicit and unassessed theories
    remain mandatory; task adoption and scientific support are unchanged.
    """
    validate_exploration_policy(exploration_policy)
    if public_prediction_rows is not None and not callable(public_prediction_rows):
        raise TypeError('public_prediction_rows must be callable or None')
    if compute_budget is not None and (type(compute_budget) is not int or compute_budget < 0):
        raise ValueError('compute_budget must be a nonnegative integer or None')
    if observation_budget is not None and (type(observation_budget) not in (int, float)
            or not isfinite(observation_budget) or observation_budget < 0):
        raise ValueError('observation_budget must be a finite nonnegative number or None')
    evaluation = native_evaluation_contract(question.get('query_semantics'))
    resolution = question.get('answer_resolution_ref')
    answer_atol = (_tolerance(resolution['atol'])
                   if isinstance(resolution, dict) and 'atol' in resolution else None)
    result = {'status': 'scoped_unknown', 'scope': 'finite_candidate_heuristic',
              'selected_experiment': None, 'tied_experiments': [], 'groups': [],
              'pairwise_edges': [], 'scores': [], 'compute_cost': 0,
              'evidence': [], 'support_updated': False, 'theory_status': 'unresolved',
              'context_ref': deepcopy(question.get('context_ref', context)), 'excluded': [],
              'public_row_validation': {'available': public_prediction_rows is not None,
                                        'validated_experiments': []}}
    plans = []
    seen = set()
    for original in experiments:
        plan = deepcopy(original)
        if not plan.get('id') or plan['id'] in seen:
            raise ValueError('experiments need distinct nonempty ids')
        seen.add(plan['id'])
        if any(type(cost) not in (int, float) or not isfinite(cost) or cost < 0 for cost in _cost(plan)):
            raise ValueError('declared plan costs must be finite nonnegative numbers')
        if not plan.get('actions') or not all(isinstance(action, dict) for action in plan['actions']):
            result['excluded'].append({'id': plan['id'], 'reason': 'missing_executable_actions'})
            continue
        if observation_budget is not None and plan.get('observation_cost', 0) > observation_budget:
            result['excluded'].append({'id': plan['id'], 'reason': 'observation_budget'})
            continue
        plans.append(plan)
    direct = [plan for plan in plans if plan.get('direct_task_action') is True
              and plan.get('support', {}).get('status') == 'supported'
              and plan['support'].get('evidence')]

    def choose(options, direct=False):
        cheapest = min(_cost(plan) for plan in options)
        tied = [plan for plan in options if _cost(plan) == cheapest]
        result['tied_experiments'] = [plan['id'] for plan in tied]
        result['status'] = 'tied' if len(tied) > 1 else 'direct_action_available' if direct else 'selected'
        selected = tied[0] if len(tied) == 1 else None
        if exploration_policy is not None:
            physical = {}
            for plan in tied:
                physical.setdefault(_physical_plan(plan), []).append(plan)
            choices = [physical[key] for key in sorted(physical, key=repr)]
            alternatives = Random(exploration_policy['seed']).choice(choices)
            selected = alternatives[0]
            result['operational_decision'] = {'policy': deepcopy(exploration_policy),
                'scope': 'declared_legal_finite_plans', 'epistemic_status': result['status'],
                'selected_experiment_id': selected['id'], 'physical_option_count': len(choices),
                'physical_options': [{'representative_id': items[0]['id'],
                    'alias_ids': [item['id'] for item in items],
                    'same_prediction_contract': len({_frozen({key: value for key, value in item.items()
                        if key not in {'id', 'actions'}}) for item in items}) == 1} for items in choices],
                'support_updated': False, 'all_pool_optimality_established': False}
        if selected is not None:
            result['selected_experiment'] = deepcopy(selected)
            result['query_arguments'] = {'actions': deepcopy(selected['actions'])}
            if selected.get('branches'):
                result['conditional_plan'] = deepcopy(selected)

    if direct:
        choose(direct, direct=True)
        return result
    plans = [plan for plan in plans if not plan.get('direct_task_action')]
    eligible = []
    for plan in plans:
        if 'query' not in plan or 'tolerance' not in plan:
            raise ValueError('research plans need their PR1 query and declared tolerance')
        _tolerance(plan['tolerance'])
        for branch in plan.get('branches', []):
            _tolerance(branch['when']['atol'])
            _tolerance(branch.get('tolerance', plan['tolerance']))
            if not branch.get('actions') or not branch.get('query'):
                raise ValueError('conditional branches need executable actions and a PR1 query')
        if (not _query_matches_actions(plan['query'], plan['actions'])
                or any(not _query_matches_actions(branch['query'], branch['actions'])
                       for branch in plan.get('branches', []))):
            result['excluded'].append({'id': plan['id'], 'reason': 'action_query_mismatch'})
            continue
        if public_prediction_rows is not None:
            if (not _query_matches_actions(plan['query'], plan['actions'], public_prediction_rows)
                    or any(not _query_matches_actions(branch['query'], branch['actions'], public_prediction_rows)
                           for branch in plan.get('branches', []))):
                result['excluded'].append({'id': plan['id'], 'reason': 'public_row_query_mismatch'})
                continue
            result['public_row_validation']['validated_experiments'].append(plan['id'])
        eligible.append(plan)
    plans = eligible
    if not plans:
        result['unknown_reason'] = 'no_affordable_experiment'
        return result
    query = question.get('query_semantics', {}).get('query_expression')
    if not query:
        result['unknown_reason'] = 'missing_executable_question'
        return result
    current_context = question.get('context_ref') if context is None else context
    if isinstance(current_context, dict):
        current_context = current_context.get('context_ref', current_context)
    declared_units = context if isinstance(context, dict) else {}
    current_unit = declared_units.get('units', declared_units.get('output_unit'))
    if current_unit is None and isinstance(current_context, dict):
        current_unit = current_context.get('units', current_context.get('output_unit'))
    evaluated, cache, ids, programs = {}, {}, set(), {}
    for theory in theories:
        identifier = theory['id']
        if identifier in ids:
            raise ValueError('theories need distinct ids; clone sources keep distinct ids')
        ids.add(identifier)
        support = theory.get('support', {})
        support_status = support.get('status') if isinstance(support, dict) else support
        if theory.get('compatible') is False or support_status in {'refuted', 'revoked', 'withdrawn'}:
            result['excluded'].append({'id': identifier, 'reason': 'incompatible_candidate'})
            continue
        if (current_context is not None and theory.get('context_ref') is not None
                and theory['context_ref'] != current_context):
            result['excluded'].append({'id': identifier, 'reason': 'context_mismatch'})
            continue
        if (current_unit is not None and theory.get('output_unit') is not None
                and theory['output_unit'] != current_unit):
            result['excluded'].append({'id': identifier, 'reason': 'unit_mismatch'})
            continue
        table = theory.get('_retained_forecast')
        if theory.get('forecast_ref') is not None and not table:
            result['excluded'].append({'id': identifier, 'reason': 'retained_forecast_unavailable'})
            evaluated[identifier] = {'task': {'status': 'scoped_unknown',
                'reason': 'retained_forecast_unavailable'}, 'known': False, 'predictions': {}}
            continue
        program = None if table is not None else Program.from_dict(theory['program'])
        if program is not None:
            programs[identifier] = program
        computation = (('retained_forecast', _frozen(table['prediction_inputs']), tuple(table['predictions']))
                       if table is not None else program)
        key = (computation, _frozen(theory.get('role_signature')), theory.get('output_unit'),
               _frozen(theory.get('context_ref')))
        if key in cache:
            evaluated[identifier] = cache[key]
            continue
        bound = deepcopy(context) if isinstance(context, dict) else {}
        if isinstance(context, dict) and 'context_ref' not in bound:
            bound['context_ref'] = deepcopy(context)
        if context is not None and not isinstance(context, dict):
            bound['context_ref'] = context
        elif context is None:
            reference = theory.get('context_ref')
            if reference is None:
                reference = question.get('context_ref')
            if reference is not None:
                bound['context_ref'] = deepcopy(reference)
        if theory.get('output_unit') is not None:
            bound.setdefault('output_unit', theory['output_unit'])
        answers = {}

        if table is not None:
            predictor = retained_forecast_predictor(table)
        else:
            def predictor(public):
                row = Observation(public['features'], 0., tuple(public['history']))
                return (predict(program, row, evaluation['fallback']) if evaluation is not None
                        else program.predict(row))

        def forward(expression):
            expression_key = _frozen(expression)
            if expression_key in answers:
                return answers[expression_key]
            remaining = None if compute_budget is None else max(0, compute_budget - result['compute_cost'])
            answer = evaluate_on_model(expression, predictor, context=bound, compute_budget=remaining)
            result['compute_cost'] += answer['compute_cost']
            answers[expression_key] = answer
            return answer

        task = forward(query)
        entry = {'task': task, 'predictions': {}, 'known': task['status'] == 'answered',
                 'prediction_status': {}, 'prediction_key': identifier}
        if entry['known'] and isinstance(task['value'], dict):
            entry.update(known=False, unknown={'reason': 'undefined_distribution_or_answer_test'})
        entry['task_known'] = entry['known']

        def forecast(plan):
            first = forward(plan['query'])
            if first['status'] != 'answered':
                return None, first
            if numeric_answer_distance(first['value'], first['value']) is None:
                return None, {'reason': 'undefined_distribution_or_answer_test'}
            signature = (first['value'],)
            if plan.get('branches'):
                branches = [(index, branch) for index, branch in enumerate(plan['branches'])
                            if (distance := numeric_answer_distance(first['value'], branch['when']['value']))
                            is not None and distance <= branch['when']['atol']]
                if len(branches) != 1:
                    return None, {'reason': 'unresolved_conditional_branch'}
                index, branch = branches[0]
                second = forward(branch['query'])
                if second['status'] != 'answered':
                    return None, second
                if numeric_answer_distance(second['value'], second['value']) is None:
                    return None, {'reason': 'undefined_distribution_or_answer_test'}
                signature = (first['value'], index, second['value'])
            return signature, None

        if entry['task_known']:
            for plan in plans:
                signature, error = forecast(plan)
                if error is not None:
                    entry.update(known=False, unknown=error)
                    entry['prediction_status'][plan['id']] = {key: deepcopy(error[key])
                        for key in ('status', 'reason', 'detail') if key in error}
                    if exploration_policy is None:
                        break
                else:
                    entry['predictions'][plan['id']] = signature
                    entry['prediction_status'][plan['id']] = {'status': 'answered'}
        cache[key] = entry
        evaluated[identifier] = entry
    if research_comparison is not None:
        evaluated = _retain_question_research_comparison(
            result, evaluated, programs, plans, question, research_comparison)
    if exploration_policy is not None:
        pairs, blocked, task_blocked = _partial_prediction_scores(result, evaluated, plans, answer_atol)
        result['next_work'] = {'kind': 'complete_candidate_predictions',
            'task_unknown_theory_ids': result['prediction_coverage']['task_unknown_theory_ids'],
            'missing_plan_predictions': {plan['id']: [identifier for identifier, entry in evaluated.items()
                if plan['id'] not in entry['predictions']] for plan in plans}}
        allowed = [(plan, score) for plan, score in zip(plans, result['scores']) if plan['id'] not in blocked]
        if task_blocked:
            result['unknown_reason'] = 'unresolved_task_contract'
        elif not evaluated:
            result['unknown_reason'] = 'no_compatible_candidates'
        elif not pairs:
            result['unknown_reason'] = 'no_task_answer_disagreement'
        elif allowed and (best := max(score['G'] for _, score in allowed)) > 0:
            choose([plan for plan, score in allowed if score['G'] == best])
        else:
            result['unknown_reason'] = 'no_legal_discriminating_witness' if blocked else 'no_discriminating_witness'
        if result['selected_experiment'] is None:
            result['next_work']['reason'] = result['unknown_reason']
            if result['prediction_coverage']['complete']:
                result['next_work']['kind'] = 'construct_or_validate_candidates'
        elif result['prediction_coverage']['complete']:
            result.pop('next_work')
        if result['status'] == 'tied':
            result['tie_diagnostics'] = diagnose_experiment_ties(result)
        return result
    known = {identifier: entry for identifier, entry in evaluated.items() if entry['known']}
    columns = [{'predictions': {identifier: (_frozen(evaluated[identifier]['task']['value']),
                  evaluated[identifier]['task'].get('units')) for identifier in known}}]
    columns.extend({'predictions': {identifier: _frozen(known[identifier]['predictions'][plan['id']])
                                   for identifier in known}} for plan in plans)
    members = forecast_equivalence_classes(columns)
    for identifiers in members:
        entry = known[identifiers[0]]
        result['groups'].append({'id': f"group_{len(result['groups']):03d}", 'theory_ids': identifiers,
            'task_answer': entry['task']['value'], 'panel_predictions': deepcopy(entry['predictions']), 'known': True})
    for identifier, entry in evaluated.items():
        if not entry['known']:
            result['groups'].append({'id': f"group_{len(result['groups']):03d}", 'theory_ids': [identifier],
                'known': False, 'unknown_reason': entry.get('unknown', entry['task']).get('reason')})
    groups = [group for group in result['groups'] if group['known']]
    pairs = []
    for left, right in combinations(groups, 2):
        separated = [plan['id'] for plan in plans if _separates(
            left['panel_predictions'][plan['id']], right['panel_predictions'][plan['id']], plan)]
        distance = numeric_answer_distance(left['task_answer'], right['task_answer'])
        relevant = (distance > answer_atol
                    if distance is not None and answer_atol is not None
                    else left['task_answer'] != right['task_answer'])
        result['pairwise_edges'].append({'left': left['id'], 'right': right['id'],
            'different_task_answer': relevant, 'experiments': separated})
        if relevant:
            pairs.append(separated)
    for plan in plans:
        count = sum(plan['id'] in separated for separated in pairs)
        result['scores'].append({'experiment_id': plan['id'], 'separated_pairs': count,
            'effective_pairs': len(pairs), 'G': count / max(1, len(pairs)),
            'declared_cost': list(_cost(plan))})
    if len(known) != len(evaluated):
        result['unknown_reason'] = 'incomplete_candidate_predictions'
    elif not evaluated:
        result['unknown_reason'] = 'no_compatible_candidates'
    elif not pairs:
        result['unknown_reason'] = 'no_task_answer_disagreement'
    else:
        best = max(score['G'] for score in result['scores'])
        if best > 0:
            choose([plan for plan, score in zip(plans, result['scores']) if score['G'] == best])
        else:
            result['unknown_reason'] = 'no_discriminating_witness'
    if result['status'] == 'tied':
        result['tie_diagnostics'] = diagnose_experiment_ties(result)
    return result
