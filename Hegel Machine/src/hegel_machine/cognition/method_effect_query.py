"""Read existing paired prior configurations and frozen published methods.

Effects retain their registered scope, failed outcomes and original costs.
Neither adapter runs methods or turns unpaired practice scores into effects.
"""

import ast
from copy import deepcopy
from math import isfinite
from pathlib import Path

from .research_episodes import load_source_values


EFFECT_KIND = 'frozen_prior_configuration_effect'
PUBLISHED_EFFECT_KIND = 'frozen_published_method_transfer_effect'
RECEIPT_SCOPE = 'same_information_frozen_family_keep_remove_historical_replay'


def _ref(reference, *path):
    return {**deepcopy(reference), 'json_path': [*reference.get('json_path', []), *path]}


def _number(value):
    return type(value) in (int, float) and isfinite(value) and value >= 0


def evaluate_method_effect(query, workspace):
    """Read a saved paired effect, with zero executions and acquisition.

    Required query fields: effect_kind='frozen_prior_configuration_effect',
    comparison_ref (a small stored receipt), left_arm and right_arm. The
    supported outcome is suffix_prediction_and_cost; action_benefit requires
    actual action outcomes and is UNKNOWN for this retained receipt.
    frozen_published_method_transfer_effect instead reads the registered window
    execution.final_outcome, pairing baseline_13of13 with published_14of14. Its
    intervention includes publication provenance, not source bytes alone.
    """
    if query.get('effect_kind') == PUBLISHED_EFFECT_KIND:
        return _published_method_effect(query, workspace)
    base = {'evidence': [], 'compute_cost': 0, 'new_method_executions': 0,
            'independent_confirmation': False}

    def unknown(detail, reason='unknown_due_to_missing_evidence', **extra):
        return {**base, 'status': 'scoped_unknown', 'reason': reason, 'detail': detail, **extra}

    if query.get('effect_kind') != EFFECT_KIND:
        return unknown('requested_method_effect_kind_not_established',
                       'unknown_due_to_uncertain_mapping')
    if query.get('outcome', 'suffix_prediction_and_cost') != 'suffix_prediction_and_cost':
        return unknown('requested_action_or_method_outcome_not_executed')
    reference = query.get('comparison_ref')
    if not isinstance(reference, dict):
        return unknown('retained_comparison_reference_required', 'unknown_due_to_uncertain_mapping')
    try:
        loaded, = load_source_values([reference], workspace)
    except (OSError, ValueError, KeyError, TypeError):
        return unknown('retained_comparison_unavailable')
    receipt, reference = loaded['value'], loaded['source_ref']
    if not isinstance(receipt, dict) or receipt.get('scope') != RECEIPT_SCOPE:
        return unknown('no_supported_paired_execution_contract')
    protocol, arms = receipt.get('protocol', {}), receipt.get('arms', {})
    left_id, right_id = query.get('left_arm'), query.get('right_arm')
    if left_id not in arms or right_id not in arms or left_id == right_id:
        return unknown('two_distinct_recorded_arms_required', 'unknown_due_to_uncertain_mapping')
    left, right = arms[left_id], arms[right_id]
    source = receipt.get('source', {})
    incumbent = protocol.get('retained_incumbent_ref')
    counterexample = protocol.get('shared_counterexample_ref')
    if (receipt.get('status') != 'passed' or not protocol.get('target')
            or not protocol.get('frozen_theme') or not protocol.get('retained_sources_in_both_arms')
            or not isinstance(incumbent, dict) or not isinstance(counterexample, dict)
            or any(not source.get(key) or incumbent.get(key) != source[key]
                   or counterexample.get(key) != source[key] for key in ('archive', 'unit', 'member'))
            or not incumbent.get('field') or not counterexample.get('field')
            or not _number(protocol.get('compute_budget'))):
        return unknown('common_start_or_frozen_prior_contract_missing')
    if (type(left.get('new_frozen_object_retained')) is not bool
            or type(right.get('new_frozen_object_retained')) is not bool
            or left['new_frozen_object_retained'] == right['new_frozen_object_retained']):
        return unknown('frozen_prior_intervention_not_identified')
    for key in ('retained_source_program_count', 'old_template_count',
                'old_template_binding_count', 'retained_incumbent_count'):
        if left.get(key) is None or left[key] != right.get(key):
            return unknown('original_shared_capabilities_or_start_differ', field=key)
    if any(arm.get('retained_counterexample', {}).get('source_ref') != counterexample
           for arm in (left, right)):
        return unknown('shared_counterexample_not_retained_in_both_arms')
    row_refs = left.get('evaluation', {}).get('row_refs')
    if (not row_refs or row_refs != right.get('evaluation', {}).get('row_refs')
            or any(any(item.get(key) != source[key] for key in ('archive', 'unit', 'member'))
                   or not item.get('field') for item in row_refs)):
        return unknown('common_evaluation_material_not_established')
    choices = [arm.get('native_parent_choice', {}) for arm in (left, right)]
    for key in ('scope', 'fit_row_indices', 'selection_row_indices', 'suffix_row_indices',
                'selection_rule', 'comparison_and_suffix_budget'):
        if choices[0].get(key) is None or choices[0][key] != choices[1].get(key):
            return unknown('task_or_split_contract_differs', field=key)
    fit, suffix = choices[0]['fit_row_indices'], choices[0]['suffix_row_indices']
    if (not fit or not suffix or set(fit) & set(suffix)
            or any(type(index) is not int or not 0 <= index < len(row_refs) for index in [*fit, *suffix])
            or choices[0]['selection_row_indices'] != fit):
        return unknown('retained_development_and_suffix_split_invalid')
    harness = receipt.get('executed_in_harness', {})
    if (not _number(harness.get('native_propose_compute'))
            or harness['native_propose_compute'] <= 0
            or not _number(harness.get('native_prefix_comparison_compute'))
            or harness['native_prefix_comparison_compute'] <= 0
            or harness.get('native_suffix_evaluation_compute') != 2 * len(suffix)):
        return unknown('paired_method_execution_not_recorded')
    outcomes = []
    for identifier, arm, choice in zip((left_id, right_id), (left, right), choices):
        costs = arm.get('accounted_compute', {})
        total = costs.get('total_replay_compute_excluding_saved_source_learning')
        if (not _number(choice.get('suffix_mse')) or choice.get('suffix_evaluation_rows') != len(suffix)
                or len(choice.get('suffix_predictions', [])) != len(suffix)
                or choice.get('selected_program_id') not in receipt.get('program_catalog', {})
                or costs.get('native_suffix_evaluation') != len(suffix)
                or not _number(total) or any(not _number(value) for value in costs.values())
                or total != sum(value for key, value in costs.items()
                                if key != 'total_replay_compute_excluding_saved_source_learning')):
            return unknown('executed_outcome_or_complete_cost_ledger_missing', arm=identifier)
        outcomes.append({'arm': identifier, 'frozen_prior_available': arm['new_frozen_object_retained'],
            'configuration_ref': _ref(reference, 'arms', identifier),
            'protocol_ref': _ref(reference, 'protocol'),
            'outcome_ref': _ref(reference, 'arms', identifier, 'native_parent_choice'),
            'cost_ref': _ref(reference, 'arms', identifier, 'accounted_compute'),
            'suffix_mse': choice['suffix_mse'], 'suffix_rows': len(suffix), 'compute_cost': total})
    value = {'effect_kind': EFFECT_KIND, 'left': outcomes[0], 'right': outcomes[1],
        'right_minus_left': {'suffix_mse': outcomes[1]['suffix_mse'] - outcomes[0]['suffix_mse'],
                            'compute_cost': outcomes[1]['compute_cost'] - outcomes[0]['compute_cost']},
        'action_benefit': {'status': 'scoped_unknown', 'reason': 'selected_experiment_not_executed'}}
    return {**base, 'status': 'answered', 'value': value, 'source_ref': reference,
        'scope': {'effect_kind': EFFECT_KIND, 'target': protocol['target'],
            'common_start_ref': deepcopy(incumbent), 'protocol_ref': _ref(reference, 'protocol'),
            'frozen_prior_ref': _ref(reference, 'protocol', 'frozen_theme'),
            'evaluation_row_refs': deepcopy(row_refs), 'fit_rows': deepcopy(fit),
            'suffix_rows': deepcopy(suffix), 'source_costs_shared': True,
            'source_costs_ref': _ref(reference, 'historical_costs_shared_in_both_arms'),
            'coverage': choices[0]['scope'], 'literal_method_source_comparison': False,
            'independent_new_confirmation': False, 'general_method_superiority': 'not_established'}}


def _published_runner(root):
    """Verify the retained runner used one original archive and one target runner."""
    runner = root / 'source_at_run/scripts/run_native_method_compilation.py'
    tree = ast.parse(runner.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_targets')
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    transfers = [node for node in calls if node.func.id == 'frozen_transfers']
    memories = [node for node in calls if node.func.id == 'transfer_memory']
    scores = [node for node in calls if node.func.id == 'score_frozen']
    if (len(transfers) != 2 or len(memories) != 2 or len(scores) != 2
            or any(ast.unparse(call.args[3]) != 'sources' for call in transfers)
            or any(ast.unparse(call.args[2]) != 'original' for call in memories)
            or any(not any(keyword.arg == 'agent_type' and ast.unparse(keyword.value) == 'ExperimentalDesignMethodAgent'
                           for keyword in call.keywords) for call in transfers)
            or max(call.lineno for call in transfers) >= min(call.lineno for call in scores)):
        raise ValueError('retained_shared_runner_contract_not_established')
    prefix_runner = root / 'source_at_run/scripts/run_anonymous_method_learning.py'
    tree = ast.parse(prefix_runner.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'trial_preparer')
    if not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == 'confirmation_actions' and len(node.args) == 2
               and ast.unparse(node.args[0]) == '72'
               and ast.unparse(node.args[1]) == "Random(case['prefix_seed'])" for node in ast.walk(function)):
        raise ValueError('retained_common_prefix_contract_not_established')
    return {'runner_ref': {'file': str(runner), 'symbol': 'run_targets'},
            'prefix_runner_ref': {'file': str(prefix_runner), 'symbol': 'trial_preparer'},
            'target_runner': 'ExperimentalDesignMethodAgent', 'prefix_rows': 72}


def _published_method_effect(query, workspace):
    """Read the registered old/new publication package transfer, never execute it.

    comparison_ref points to window_parameter_transfer's execution.final_outcome.
    left_arm/right_arm are baseline_13of13 and published_14of14 in either order.
    Both source and its publication support are the intervention; this is not a
    source-bytes-only effect or a general Code/Hegel organization comparison.
    """
    base = {'evidence': [], 'compute_cost': 0, 'new_method_executions': 0,
            'independent_confirmation': False}

    def unknown(detail, **extra):
        return {**base, 'status': 'scoped_unknown', 'reason': 'unknown_due_to_missing_evidence',
                'detail': detail, **extra}

    if query.get('outcome', 'registered_transfer_and_cost') != 'registered_transfer_and_cost':
        return unknown('requested_source_only_or_general_method_effect_not_established')
    reference = query.get('comparison_ref')
    if not isinstance(reference, dict) or not reference.get('file'):
        return unknown('registered_published_transfer_reference_required')
    labels = ('baseline_13of13', 'published_14of14')
    left, right = query.get('left_arm'), query.get('right_arm')
    if set((left, right)) != set(labels):
        return unknown('two_registered_published_method_arms_required')
    try:
        outcome, registered = load_source_values([reference,
            {'file': reference['file'], 'json_path': ['frozen_evaluation_registration']}], workspace)
        outcome_ref, registration_ref = outcome['source_ref'], registered['source_ref']
        outcome, registration = outcome['value'], registered['value']
        if outcome['status'] != 'completed' or registration.get('source_fits_and_author_history_transfer') is not False:
            raise ValueError('published_transfer_not_completed_or_prediction_knowledge_transferred')
        summaries = [Path(outcome['baseline_summary']), Path(outcome['summary'])]
        roots = [path.parent for path in summaries]
        protocol_refs = [{'file': str(root / 'protocol.json'), 'json_path': []} for root in roots]
        summary_refs = [{'file': str(path), 'json_path': []} for path in summaries]
        loaded = load_source_values([*protocol_refs, *summary_refs], workspace)
        protocols, results = [item['value'] for item in loaded[:2]], [item['value'] for item in loaded[2:]]
        if any(result.get('status') != 'completed' or result.get('method_published') is not True for result in results):
            raise ValueError('both_published_method_executions_required')
        publications = [protocol['frozen_publication'] for protocol in protocols]
        if (any(result.get('frozen_publication') != publication for result, publication in zip(results, publications))
                or len(publications[0].get('confirmed_cases', [])) != 13
                or len(publications[1].get('confirmed_cases', [])) != registration['required_practice_count']
                or publications[1].get('source_study') != registration['source_publication']):
            raise ValueError('actual_published_method_provenance_missing_or_changed')
        for key in ('source', 'limits', 'evaluation', 'method_steps'):
            if protocols[0].get(key) is None or protocols[0][key] != protocols[1].get(key):
                raise ValueError(f'published_pair_contract_differs:{key}')
        for key in ('limits', 'evaluation', 'method_steps'):
            if protocols[1][key] != registration[key]:
                raise ValueError(f'registered_budget_or_evaluation_changed:{key}')
        archives = [(root / 'reference_archive').resolve(strict=True) for root in roots]
        if archives[0] != archives[1]:
            raise ValueError('original_prediction_archive_differs')
        runner = _published_runner(roots[1])
        registered_cases = registration['new_transfer_cases']
        ids = [case['id'] for case in registered_cases]
        if (len(ids) != 8 or len(set(ids)) != 8
                or len({case['parameter_pair'] for case in registered_cases}) != registration['new_transfer_parameter_groups']
                or len(ids) != registration['new_transfer_parameter_groups'] * registration['new_transfer_conditions_per_group']):
            raise ValueError('complete_registered_window_group_required')
        cases = [{case['id']: case for case in protocol['transfer_cases']} for protocol in protocols]
        actual = [{row['case']: row for row in result['new_parameters']} for result in results]
        declared = {row['case']: row for row in outcome['paired_window_cases']}
        if any(set(rows) != set(ids) or len(result['new_parameters']) != len(ids)
               for rows, result in zip(actual, results)) or set(declared) != set(ids):
            raise ValueError('registered_failures_or_results_missing')
        source_files = [root / 'published_method.py' for root in roots]
        sources = [path.read_text() for path in source_files]
        if not all(sources) or sources[0] == sources[1]:
            raise ValueError('two_different_actual_frozen_methods_required')
        catalog_refs, method_refs = [], []
        for case in registered_cases:
            if any(rows.get(case['id']) != case for rows in cases):
                raise ValueError('registered_case_start_or_seeds_changed')
            for index, root in enumerate(roots):
                method = root / 'frozen' / case['id'] / 'shared/consolidated_method.py'
                if method.read_text() != sources[index]:
                    raise ValueError('executed_frozen_method_differs_from_publication')
                method_refs.append({'case': case['id'], 'arm': labels[index], 'source_ref': {'file': str(method)}})
                catalog_refs.append({'file': str(root / 'frozen' / case['id'] / 'source_catalog.json'), 'json_path': []})
        catalogs = load_source_values(catalog_refs, workspace)
        if any(catalogs[index]['value'] != catalogs[index + 1]['value'] for index in range(0, len(catalogs), 2)):
            raise ValueError('initial_prediction_catalog_differs')
        cost_keys = ('observation_cost', 'measurement_cost', 'compute_cost',
                     'python_execution_seconds', 'model_calls', 'reported_tokens')
        arms, pairs = {}, []
        for index, label in enumerate(labels):
            rows = [actual[index][identifier] for identifier in ids]
            for row in rows:
                costs, terminal = row['costs'], row['terminal']
                if (any(not _number(costs.get(key)) for key in cost_keys)
                        or costs['model_calls'] != 0 or not row.get('method_steps')
                        or row.get('status') not in {'completed', 'method_stopped'}):
                    raise ValueError('executed_outcome_or_full_costs_missing')
                if row.get('model') is None:
                    if terminal.get('status') != 'no_submitted_model' or terminal.get('mse') is not None or terminal.get('passed') is not False:
                        raise ValueError('unsubmitted_model_failure_not_preserved')
                elif (terminal.get('returncode') != 0 or not _number(terminal.get('mse'))
                      or type(terminal.get('passed')) is not bool):
                    raise ValueError('submitted_model_terminal_result_missing')
                saved = declared[row['case']]
                prefix = 'old' if index == 0 else 'new'
                if (saved[f'{prefix}_status'] != row['status'] or saved[f'{prefix}_terminal'] != terminal
                        or saved[f'{prefix}_costs'] != costs
                        or saved[f'{prefix}_confirmation_passed'] != row['fresh_confirmation_passed']):
                    raise ValueError('registered_receipt_and_actual_summary_conflict')
            arms[label] = {'arm': label, 'method_source_ref': {'file': str(source_files[index])},
                'publication_ref': _ref(loaded[index]['source_ref'], 'frozen_publication'),
                'protocol_ref': loaded[index]['source_ref'], 'summary_ref': loaded[index + 2]['source_ref'],
                'case_count': len(rows), 'submitted': sum(row.get('model') is not None for row in rows),
                'confirmation_passed': sum(row['fresh_confirmation_passed'] is True for row in rows),
                'terminal_passed': sum(row['terminal']['passed'] is True for row in rows),
                'costs': {key: sum(row['costs'][key] for row in rows) for key in cost_keys},
                'recorded_author_costs': deepcopy(results[index]['author_costs']),
                'recorded_author_costs_ref': _ref(loaded[index + 2]['source_ref'], 'author_costs'),
                'author_cost_scope': 'recorded_author_account; not total shared lineage',
                'prior_author_phases_ref': _ref(loaded[index]['source_ref'], 'prior_author_phases'),
                'prior_transfer_phases_ref': _ref(loaded[index]['source_ref'], 'prior_transfer_phases')
                    if 'prior_transfer_phases' in protocols[index] else None}
        for identifier in ids:
            values = {label: actual[index][identifier] for index, label in enumerate(labels)}
            lrow, rrow = values[left], values[right]
            lmse, rmse = lrow['terminal'].get('mse'), rrow['terminal'].get('mse')
            pairs.append({'case': identifier,
                'registered_case_ref': _ref(registration_ref, 'new_transfer_cases', ids.index(identifier)),
                'left': deepcopy(lrow), 'right': deepcopy(rrow),
                'right_minus_left': {'terminal_mse': rmse - lmse if lmse is not None and rmse is not None else
                    {'status': 'scoped_unknown', 'reason': 'no_submitted_model_in_one_arm'},
                    'costs': {key: rrow['costs'][key] - lrow['costs'][key] for key in cost_keys}}})
        shared, own = [], {label: [] for label in labels}
        phases = [protocol.get('prior_author_phases', []) for protocol in protocols]
        for index, phase in enumerate(phases[0]):
            matches = [j for j, other in enumerate(phases[1]) if phase == other]
            if matches:
                shared.append({labels[0]: _ref(loaded[0]['source_ref'], 'prior_author_phases', index),
                               labels[1]: _ref(loaded[1]['source_ref'], 'prior_author_phases', matches[0])})
            else:
                own[labels[0]].append(_ref(loaded[0]['source_ref'], 'prior_author_phases', index))
        for index, phase in enumerate(phases[1]):
            if phase not in phases[0]:
                own[labels[1]].append(_ref(loaded[1]['source_ref'], 'prior_author_phases', index))
        value = {'effect_kind': PUBLISHED_EFFECT_KIND, 'intervention': 'frozen_published_method_package',
            'left': arms[left], 'right': arms[right], 'cases': pairs,
            'right_minus_left': {key: arms[right][key] - arms[left][key]
                for key in ('submitted', 'confirmation_passed', 'terminal_passed')} | {
                'costs': {key: arms[right]['costs'][key] - arms[left]['costs'][key] for key in cost_keys},
                'terminal_mse': {'status': 'scoped_unknown', 'reason': 'unsubmitted_baseline_models_have_no_MSE'}},
            'source_cost_lineage': {'shared_prior_author_phase_refs': shared,
                'nonidentical_prior_author_phase_refs': own, 'phase_accounts_summed': False,
                'nonidentical_phases_are_independent': 'not_established'}}
        return {**base, 'status': 'answered', 'value': value, 'source_ref': outcome_ref,
            'scope': {'effect_kind': PUBLISHED_EFFECT_KIND, 'intervention': 'frozen_published_method_package',
                'registration_ref': registration_ref, 'registered_case_ids': ids,
                'independent_parameter_groups': registration['new_transfer_parameter_groups'],
                'conditions_per_group': registration['new_transfer_conditions_per_group'],
                'same_original_prediction_archive': {'directory': str(archives[0])},
                'same_initial_catalog_refs': [item['source_ref'] for item in catalogs],
                'executed_method_refs': method_refs, **runner,
                'methods_include_publication_support_provenance': True,
                'literal_source_bytes_only': False, 'general_method_superiority': 'not_established',
                'prediction_knowledge_gain': 'not_established', 'new_independent_confirmation': False}}
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration, SyntaxError) as error:
        return unknown('published_method_transfer_contract_unavailable', cause=str(error))
