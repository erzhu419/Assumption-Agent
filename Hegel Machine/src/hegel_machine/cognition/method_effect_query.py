"""Read existing paired prior configurations and frozen published methods.

Effects retain their registered scope, failed outcomes and original costs.
Neither adapter runs methods or turns unpaired practice scores into effects.
"""

import ast
from copy import deepcopy
import json
from math import isfinite
from pathlib import Path
from types import SimpleNamespace

from .checkpoint_storage import write_text_incrementally
from .research_episodes import load_source_values


EFFECT_KIND = 'frozen_prior_configuration_effect'
PUBLISHED_EFFECT_KIND = 'frozen_published_method_transfer_effect'
SUCCESSOR_EFFECT_KIND = 'registered_successor_research_effect'
RECEIPT_SCOPE = 'same_information_frozen_family_keep_remove_historical_replay'


def event_experiment_selection(event):
    """Read the optional selection recorded by ordinary research hooks."""
    return (event.get('feedback') or {}).get('selection') or {}


def retained_research_progress(agent, *, event_start=0, method_start=0):
    """Project retained meanings and actual new work without reopening files.

    Latest execution counters are not cumulative costs. Event counters below
    describe only the requested suffix; the unchanged ledger supplies totals.
    Programs, paid rows and fitting trees remain at their existing addresses.
    """
    from .research_episodes import _question_versions

    def fields(value, names):
        return {name: deepcopy(value[name]) for name in names if name in value}

    questions, executions = [], []
    retained = getattr(agent, 'theory_consolidation', {}).get('executions', {})
    for (identifier, meaning), (question, reference) in _question_versions(agent).items():
        if question.get('kind') != 'program_revision':
            continue
        target = question.get('theory_target') or {}
        recall = question.get('frozen_recall_execution') or {}
        result = recall.get('result') or {}
        prefix = f'relation_transfer:frozen_recall:{identifier}@{meaning}:'
        groups = deepcopy(result.get('execution_work_groups', []))
        questions.append({'question_id': identifier, 'meaning_version': meaning,
            'source_ref': deepcopy(reference), 'parent_theory_id': target.get('parent_theory_id'),
            'evidence_scope': deepcopy(target.get('paid_evidence_scope')),
            'status': question.get('status'), 'operative_work_groups': groups,
            'object_cursor': deepcopy(recall.get('cursor')),
            'recall_latest': fields(result, ('status', 'search_completed', 'cache_hit',
                'new_compute_cost', 'pending_object_ids', 'unfunded_objects')),
            'revision_export_count': len(question.get('revision_candidate_exports', []))})
        operative_keys = {group['execution_key'] for group in groups if group.get('execution_key')}
        for key, execution in retained.items():
            if not key.startswith(prefix) and key not in operative_keys:
                continue
            basis = execution.get('request_basis') or {}
            cursor = execution.get('cursor') or {}
            output = execution.get('result') or {}
            fitting = cursor.get('fitting') or {}
            grammar = fitting.get('grammar_cursor') or {}
            generated = []
            for item in output.get('generated_theories', []):
                fit = (item.get('binding') or {}).get('fitting') or {}
                generated.append({**fields(item, ('theory_id', 'version', 'status',
                    'reused_existing_theory', 'existing_fitting_ref')),
                    'fitting': fields(fit, ('status', 'fitting_method', 'search_completed',
                        'fitted_rows', 'training_loss', 'fit_reuse_ref', 'fit_reuse_output_ref', 'scope'))})
            executions.append({'execution_key': key, 'question_id': identifier,
                'meaning_version': meaning, 'operative': key in operative_keys,
                'source_ref': {'file': 'shared/theory_library.json',
                               'json_path': ['executions', key]},
                'candidate_id': basis.get('candidate_id'),
                'candidate_version': basis.get('candidate_version'),
                'parent': fields(basis.get('parent') or {}, ('id', 'version', 'source_ref')),
                'components': [fields(item, ('id', 'version'))
                               for item in basis.get('components', [])],
                'cursor': {**fields(cursor, ('layer', 'offset')),
                    'pending_fits': [fields(item, ('indices', 'position'))
                                     for item in cursor.get('pending_fits', [])],
                    'fitting': {'structure_count': len(fitting.get('structures', [])),
                        'grammar_cursor': fields(grammar, ('positions', 'next_operation',
                            'exhausted_operations', 'status', 'budget_exhausted')),
                        'excluded_structure_count': len(fitting.get('excluded_structures', []))}},
                'allowance': execution.get('allowance'),
                'search_completed': execution.get('search_completed'),
                'latest_result': fields(output, ('status', 'cache_hit', 'execution_key',
                    'new_compute_cost', 'budget_exhausted', 'minimum_compute_for_progress',
                    'retained_complete_fit_batches_available', 'retained_fit_recovery_compute',
                    'retained_fit_reuse_count', 'reused_existing_theory_count', 'retained_execution_refs')),
                'generated_refs': generated})

    events, recall_cost, work_cost = [], 0, 0
    for index, event in enumerate(getattr(agent, 'events', [])[event_start:], event_start):
        feedback = event.get('feedback') or {}
        recall = feedback.get('frozen_recall') or {}
        work = [fields(item, ('candidate_id', 'shared_candidate_ids', 'theory_ids',
                             'new_compute_cost', 'status', 'reason')) for item in recall.get('work', [])]
        recall_cost += recall.get('new_compute_cost', 0)
        work_cost += sum(item.get('new_compute_cost', 0) for item in work)
        selection = event_experiment_selection(event)
        evaluations = feedback.get('evaluations') or []
        measured = [item['mse'] for item in evaluations if _number(item.get('mse'))]
        program_action = feedback.get('program_action') or {}
        action_arguments = program_action.get('arguments') or {}
        events.append({'event_index': index, 'action': event.get('action'),
            'operator': feedback.get('operator', (event.get('arguments') or {}).get('operator')),
            **fields(feedback, ('status', 'error', 'new_compute_cost', 'research_track_question_ref',
                               'parent_revision_question_ref')),
            'source_ref': {'file': 'training.json', 'json_path': ['events', index]},
            'frozen_recall': {**fields(recall, ('cache_hit', 'new_compute_cost', 'search_completed',
                'pending_object_ids', 'execution_work_groups')), 'work': work},
            'comparison': {'evaluations': len(evaluations), 'finite_nonnegative_mse_count': len(measured),
                'minimum_recorded_mse': min(measured, default=None),
                **fields(feedback, ('selected_model', 'truncated'))},
            'selected_experiment': deepcopy(selection.get('selected_experiment')),
            'program_action': {'action': program_action.get('action'),
                **fields(action_arguments, ('operator', 'research_question', 'model')),
                'models_count': len(action_arguments.get('models', []))}})
    methods = []
    for index, entry in enumerate(getattr(agent, 'method_results', [])[method_start:], method_start):
        proposal, feedback = entry.get('proposal') or {}, entry.get('feedback') or {}
        methods.append({'method_index': index, 'action': proposal.get('action'),
            'operator': (proposal.get('arguments') or {}).get('operator'),
            **fields(feedback, ('status', 'error', 'selected_model', 'truncated')),
            'source_ref': {'file': 'training.json', 'json_path': ['method_results', index]}})
    from .finite_native_author import _comparison_models
    from .python_research_routing import validated_candidates

    old = (getattr(agent, 'knowledge', {}).get('finite_native_author') or {}).get('predictor_paths', [])
    native = [bank for question in agent.questions
              for bank in question.get('native_condition_research', [])]
    paths = list(dict.fromkeys([*_comparison_models(old, getattr(agent, 'candidate_paths', []), native),
                               *validated_candidates(agent)]))
    return {'status': 'recorded', 'scope': 'retained_agent_metadata_without_fit_prediction_or_observation',
        'questions': questions, 'executions': executions, 'event_start': event_start,
        'method_start': method_start, 'new_events': events, 'new_methods': methods,
        'new_event_compute_sums': {'frozen_recall_total': recall_cost,
            'frozen_recall_work_only': work_cost, 'are_additive_to_each_other': False,
            'scope': 'recorded_suffix_only_not_total_ledger_or_fit_call_counts'},
        'full_pool_count': len(paths),
        'full_pool_scope': 'existing_finite_author_predictors_exports_current_native_banks_and_validated_candidates',
        'current_parent': deepcopy(getattr(agent, 'evaluated_model', None)),
        'costs': deepcopy(getattr(agent, 'costs', {}))}


def _ref(reference, *path):
    return {**deepcopy(reference), 'json_path': [*reference.get('json_path', []), *path]}


def _number(value):
    return type(value) in (int, float) and isfinite(value) and value >= 0


def registered_successor_effect(left_study, right_study=None, *, left_arm='fixed_pr2',
                               right_arm='pr3', source_cost_ref=None, intervention_ref=None,
                               saved_summaries=None):
    """Read completed registered runs, preserving missing results and costs.

    Both ordinary ABC and a keep/remove C pair use the same existing summary,
    Episode extraction and retained forecast columns. No predictor executes.
    """
    from .language import Program
    from .research_episodes import extract_episodes
    from .retained_forecasts import resolve_forecast_references

    roots = [Path(left_study).resolve(), Path(right_study or left_study).resolve()]
    cache = {}

    def read(path):
        path = Path(path).resolve()
        if path not in cache:
            cache[path] = json.loads(path.read_text(encoding='utf-8'))
        return cache[path]

    def reference(path, *keys):
        return {'file': str(Path(path).resolve()), 'json_path': list(keys)}

    def resolve(ref, workspace):
        path = Path(ref['file'])
        value = read(path if path.is_absolute() else workspace / path)
        for key in ref.get('json_path', []):
            value = value[key]
        return value

    def anchor(ref, workspace):
        if isinstance(ref, list):
            return [anchor(item, workspace) for item in ref]
        if not isinstance(ref, dict):
            return ref
        result = {key: anchor(value, workspace) for key, value in ref.items()}
        # Native studies retain the executed events in training.json; live
        # inquiry workspaces may still have the original research_context.
        if result.get('file') == 'research_context.json' and not (workspace / 'research_context.json').is_file():
            result['file'] = 'training.json'
        if result.get('file') and not Path(result['file']).is_absolute():
            result['file'] = str((workspace / result['file']).resolve())
        return result

    result = {'status': 'UNKNOWN', 'effect_kind': SUCCESSOR_EFFECT_KIND,
        'left_arm': left_arm, 'right_arm': right_arm, 'cases': [],
        'new_method_executions': 0, 'new_predictions': 0,
        'scope': 'descriptive_matched_registered_runs_not_general_scientific_superiority',
        'common_scheduler_repair_is_PR3_effect': False, 'intervention_ref': deepcopy(intervention_ref),
        'source_costs': {'ledger_ref': deepcopy(source_cost_ref),
            'project_development_compute_interval': None, 'C_attributed_development_compute_interval': None,
            'C_source_learning_compute_interval': None,
            'unresolved_accounts': None, 'B_source_compute_interval': None,
            'source_net_C_minus_B': None, 'source_fees_summed_into_each_case': False}}
    try:
        plans = [read(root / 'protocol.json') for root in roots]
        summaries = ([read(root / 'summary.json') for root in roots]
                     if saved_summaries is None else saved_summaries)
    except (OSError, ValueError) as error:
        return {**result, 'reason': 'registered_study_unavailable', 'detail': str(error)}
    result['study_refs'] = ([reference(root / 'summary.json') for root in roots]
                           if saved_summaries is None else [])
    if saved_summaries is not None:
        result['summary_input_kind'] = 'provided_in_memory_from_retained_training'
    result['registered_phase'] = plans[0].get('research_organization', {}).get('phase')
    if any(summary.get('status') != 'completed' for summary in summaries):
        return {**result, 'reason': 'all_training_must_finish_before_effect_readout'}
    if any(plans[0].get(key) != plans[1].get(key) for key in
           ('cases', 'limits', 'evaluation', 'prefix_rows', 'author')):
        return {**result, 'reason': 'registered_cases_budgets_or_evaluation_differ'}
    if source_cost_ref:
        try:
            ledger = resolve(source_cost_ref, roots[0])
            formal = ledger.get('formal_compute_ledger', {})
            result['source_costs'].update(
                project_development_compute_interval=deepcopy(formal.get('global_after_interval_excluding_legacy_unknown')),
                C_attributed_development_compute_interval=deepcopy(formal.get('C_after_interval')),
                unresolved_accounts=deepcopy(ledger.get('separate_accounts')))
        except (OSError, ValueError, KeyError, TypeError):
            result['source_costs']['status'] = 'UNKNOWN_source_ledger_unavailable'
    # A frozen C removal reuses the learning that already produced keep C.
    # Its absolute source bill can remain unknown while that common bill
    # cancels in the paired difference. Ordinary B/C do not share this claim.
    intervention = summaries[1].get('target_intervention') or {}
    separate_removal = bool(roots[0] != roots[1]
        and left_arm == right_arm == 'pr3'
        and intervention.get('kind') in {'individual_object', 'new_redundancy_class'}
        and intervention.get('manifest_ref') == intervention_ref
        and intervention.get('keep_study') == str(roots[0])
        and intervention.get('executed_arms') == ['pr3'])
    views = plans[0].get('branch_views') or {}
    same_scene_removal = bool(roots[0] == roots[1]
        and plans[0].get('scope') == 'saved_scene_development_object_interventions'
        and left_arm == 'keep' and right_arm in {'individual_remove', 'class_remove'}
        and views.get(left_arm, {}).get('organization_arm') == 'pr3'
        and views.get(right_arm, {}).get('organization_arm') == 'pr3'
        and views.get(right_arm, {}).get('manifest_ref') == intervention_ref)
    removal_kind = ({'individual_remove': 'individual_object',
                     'class_remove': 'new_redundancy_class'}[right_arm]
                    if same_scene_removal else intervention.get('kind'))
    shared_source = bool(source_cost_ref and intervention_ref
        and (separate_removal or same_scene_removal)
        and all(plan.get('source_cost_ref') == source_cost_ref for plan in plans))
    if shared_source:
        try:
            manifest = resolve(intervention_ref, roots[1])
            shared_source = (manifest.get('status') == 'frozen_ready'
                and manifest.get('kind') == removal_kind
                and manifest.get('source_cost_ref') == source_cost_ref)
        except (OSError, ValueError, KeyError, TypeError):
            shared_source = False
    result['source_costs']['shared_source_cancels_in_removal_difference'] = shared_source
    cost_keys = ('observation_cost', 'measurement_cost', 'compute_cost',
                 'python_execution_seconds', 'model_calls', 'reported_tokens')

    def documents(root, row, case, arm):
        saved = row.get('result_ref') or row.get('training_ref')
        path = Path(saved['file']) if saved else (
            Path(row['archive']).parent / 'training.json' if row.get('archive') else root / case / arm / 'training.json')
        training = read(path)
        workspace = path.resolve().parent
        state_path = workspace / 'research_state.json'
        state = read(state_path)
        return training, state, workspace

    def mechanism(training, state, workspace):
        state_ref = reference(workspace / 'research_state.json')
        proxy = SimpleNamespace(workspace=workspace, questions=state.get('questions', []),
            revisions=state.get('revisions', []), relation_research=state.get('relation_research', {}),
            theory_index=state.get('theory_index', {}), events=training.get('events', []),
            observations=training.get('paid_observations', []), theory_consolidation={},
            config=SimpleNamespace(**training.get('limits', {})),
            _criterion_source_ref=reference(workspace / 'training.json', 'limits', 'confirmation_mse_threshold'))
        # Existing extraction retains exact before/after Programs and original
        # same-paid retention evidence, without using paths as Program identity.
        episodes = extract_episodes(proxy)['episodes']
        changes = [anchor(change, workspace) for episode in episodes
                   for change in episode['actual_changes'] if change['kind'] == 'retained_program_replaced']
        programs, program_refs, model_programs, model_theories = [], [], {}, {}
        for qi, question in enumerate(proxy.questions):
            for ri, bank in enumerate(question.get('native_condition_research', [])):
                for ci, candidate in enumerate(bank.get('candidates', [])):
                    if candidate.get('program'):
                        program = Program.from_dict(candidate['program']).expanded()
                        programs.append(program)
                        program_refs.append(_ref(state_ref, 'questions', qi, 'native_condition_research', ri, 'candidates', ci, 'program'))
                        if candidate.get('model'):
                            model_programs.setdefault(candidate['model'], set()).add(program)
            for exported in question.get('revision_candidate_exports', []):
                identifier = exported.get('theory_ref', {}).get('id')
                theory = proxy.theory_index.get(identifier, {})
                if theory.get('program'):
                    program = Program.from_dict(theory['program']).expanded()
                    programs.append(program)
                    program_refs.append(_ref(state_ref, 'theory_index', identifier, 'program'))
                    model_programs.setdefault(exported['model'], set()).add(program)
                    model_theories[exported['model']] = identifier
        # The completed training result can retain a later author choice than
        # the last checkpoint written by an automatic research hook.
        evaluated = training.get('evaluated_model') or state.get('evaluated_model') or {}
        parent_model = evaluated.get('source_model') or evaluated.get('model')
        parent_versions = model_programs.get(parent_model, set())
        parent = next(iter(parent_versions)) if len(parent_versions) == 1 else None
        # The retained executable can remain identifiable when the current
        # native bank has no typed Program entry for an inherited parent.
        closed_parent = None
        retained_model = evaluated.get('model') or parent_model
        if retained_model:
            from .code_agent import _closed_temporal_export, python_model_context

            try:
                directory, model = python_model_context(workspace, retained_model)
                if _closed_temporal_export(directory, model):
                    closed_parent = {'source_ref': reference(directory / model),
                        'source': (directory / model).read_text(encoding='utf-8')}
            except (OSError, ValueError, TypeError):
                pass
        route_rows, proposals, forecasts, roster, objects = [], [], [], [], {}
        for ei, event in enumerate(proxy.events):
            feedback = event.get('feedback', {})
            recall = feedback.get('frozen_recall') or {}
            for item in recall.get('work', []):
                obj = objects.setdefault(item['candidate_id'], {'attempt_refs': [], 'generated_theory_ids': [], 'adopted_parent_refs': []})
                obj['attempt_refs'].append(reference(workspace / 'training.json', 'events', ei, 'feedback', 'frozen_recall'))
                obj['generated_theory_ids'].extend(item.get('theory_ids', []))
            selection = event_experiment_selection(event)
            if selection.get('selected_experiment'):
                plan = selection['selected_experiment']
                proposals.append(deepcopy(plan.get('actions')))
                route_rows.append(reference(workspace / 'training.json', 'events', ei, 'feedback', 'selection'))
            for group in selection.get('groups', []):
                for identifier in group.get('theory_ids', []):
                    binding = proxy.theory_index.get(identifier, {}).get('consolidation_ref') or {}
                    if binding.get('candidate_id'):
                        obj = objects.setdefault(binding['candidate_id'], {
                            'attempt_refs': [], 'generated_theory_ids': [], 'adopted_parent_refs': []})
                        obj.setdefault('selection_refs', []).append(reference(
                            workspace / 'training.json', 'events', ei, 'feedback', 'selection'))
        for qi, question in enumerate(proxy.questions):
            recall = question.get('frozen_recall_execution') or {}
            roster.extend(item['id'] for item in recall.get('basis', {}).get('candidates', []))
            for item in recall.get('result', {}).get('work', []):
                obj = objects.setdefault(item['candidate_id'], {'attempt_refs': [], 'generated_theory_ids': [], 'adopted_parent_refs': []})
                obj['attempt_refs'].append(_ref(state_ref, 'questions', qi, 'frozen_recall_execution', 'result'))
                obj['generated_theory_ids'].extend(identifier for identifier in item.get('theory_ids', [])
                                                 if identifier not in obj['generated_theory_ids'])
            saved = question.get('python_forecast_execution') or {}
            source = saved.get('source_ref')
            if source:
                try:
                    document = resolve(source, workspace)
                    columns = [{**column, 'forecast_ref': {**source, 'json_path': [*source.get('json_path', []), 'forecasts', i], 'model': column['model']}}
                               for i, column in enumerate(document.get('forecasts', []))]
                    columns = resolve_forecast_references(columns, workspace)
                    scope = {key: question.get(key) for key in
                        ('meaning_version', 'context_ref', 'query_semantics', 'answer_resolution_ref')}
                    scope['typed_target'] = {key: question.get('theory_target', {}).get(key)
                                            for key in ('role_signature', 'output_unit', 'context_family_ref')}
                    forecasts.append({'scope': scope,
                        'source_ref': anchor(source, workspace), 'columns': {column['model']: column for column in columns}})
                except (OSError, ValueError, KeyError, TypeError):
                    forecasts.append({'scope': None, 'source_ref': anchor(source, workspace), 'columns': {}})
        for change in changes:
            identifier = change.get('fit_state_refs', {}).get('after', {}).get('theory_ref', {}).get('id')
            binding = proxy.theory_index.get(identifier, {}).get('consolidation_ref') or {}
            if binding.get('candidate_id'):
                obj = objects.setdefault(binding['candidate_id'], {
                    'attempt_refs': [], 'generated_theory_ids': [], 'adopted_parent_refs': []})
                obj['adopted_parent_refs'].append(deepcopy(change['action_ref']))
        adopted_theory = model_theories.get(parent_model)
        for identifier in set(roster) | set(objects):
            obj = objects.setdefault(identifier, {'attempt_refs': [], 'generated_theory_ids': [], 'adopted_parent_refs': []})
            if adopted_theory and adopted_theory in obj['generated_theory_ids']:
                obj['adopted_parent_refs'].append(_ref(state_ref, 'evaluated_model'))
            obj['usage_status'] = ('adopted' if obj['adopted_parent_refs'] else
                'selection_consumed' if obj.get('selection_refs') else
                'attempted' if obj['attempt_refs'] else 'not_visited_under_recorded_policy')
            obj['utility_after_adoption'] = 'not_tested' if not obj['adopted_parent_refs'] else 'requires_matched_outcome'
        actions, action_refs, requested, unknown_queries, failed_queries, non_acquisitions = [], [], [], [], [], []
        for ei, event in enumerate(proxy.events):
            if (event.get('action') != 'query' or event.get('usage', {}).get('backend') in
                    {'common_paid_prefix', 'restored_paid_prefix'}):
                continue
            requested.append(deepcopy(event.get('arguments', {})))
            feedback = event.get('feedback', {})
            if isinstance(feedback.get('observations'), list):
                records = [(record, reference(workspace / 'training.json', 'events', ei,
                    'feedback', 'observations', i)) for i, record in enumerate(feedback['observations'])]
            elif isinstance(feedback.get('record_indices'), list):
                records = [(proxy.observations[i], reference(workspace / 'training.json',
                    'paid_observations', i)) for i in feedback['record_indices']
                    if 0 <= i < len(proxy.observations)]
                if len(records) != len(feedback['record_indices']):
                    unknown_queries.append(reference(workspace / 'training.json', 'events', ei))
            else:
                records = []
                declared = feedback.get('program_action') or {}
                action = declared.get('action')
                operator = declared.get('arguments', {}).get('operator')
                if action == 'evaluate' or (action == 'run' and operator in {
                        'research_choice', 'refit', 'native_condition_research'}):
                    non_acquisitions.append(reference(workspace / 'training.json', 'events', ei,
                        'feedback', 'program_action'))
                else:
                    unknown_queries.append(reference(workspace / 'training.json', 'events', ei))
            for record, record_ref in records:
                if record.get('action') is None:
                    unknown_queries.append(record_ref)
                else:
                    actions.append(deepcopy(record['action']))
                    action_refs.append(_ref(record_ref, 'action'))
            if feedback.get('error'):
                failed_queries.append(reference(workspace / 'training.json', 'events', ei, 'feedback'))
        parent_forecasts = []
        if parent is not None:
            from .code_agent import _closed_temporal_export
            from .temporal_operator_agent import export_temporal_program

            expected = export_temporal_program(parent, feature_names=tuple(sorted(parent.required_features())))
            for frame in forecasts:
                for model in dict.fromkeys((parent_model, evaluated.get('model'))):
                    column = frame['columns'].get(model) or {}
                    saved = column.get('source_context_ref') or {}
                    directory = Path(saved.get('workspace', ''))
                    if not directory.is_absolute():
                        directory = workspace / directory
                    inner = saved.get('model')
                    try:
                        bound = (inner and _closed_temporal_export(directory, inner)
                                 and (directory / inner).read_text(encoding='utf-8') == expected)
                    except OSError:
                        bound = False
                    if bound and column.get('_retained_forecast'):
                        parent_forecasts.append({'scope': frame['scope'], 'source_ref': frame['source_ref'],
                            'model': model, 'column': column})
                        break
        parent_ref = (reference(workspace / 'training.json', 'evaluated_model')
                      if training.get('evaluated_model') else _ref(state_ref, 'evaluated_model'))
        return {'state_ref': state_ref, 'parent_ref': parent_ref,
            'parent_model': parent_model, 'parent_program': parent, 'closed_parent': closed_parent,
            'candidate_programs': set(programs),
            'candidate_program_refs': program_refs, 'actual_parent_changes': changes,
            'proposed_actions': proposals, 'proposal_refs': route_rows, 'executed_actions': actions,
            'executed_action_refs': action_refs, 'requested_queries': requested,
            'unknown_query_refs': unknown_queries, 'failed_query_refs': failed_queries,
            'non_acquisition_query_refs': non_acquisitions,
            'forecasts': forecasts, 'parent_forecasts': parent_forecasts, 'objects': objects}

    def parent_forecast_difference(left, right):
        for first in left:
            for second in right:
                if first['scope'] != second['scope']:
                    continue
                a, b = [item['column']['_retained_forecast'] for item in (first, second)]
                if a['prediction_inputs'] != b['prediction_inputs']:
                    continue
                witness = next((i for i, (x, y) in enumerate(zip(a['predictions'], b['predictions'])) if x != y), None)
                return {'status': 'recorded', 'changed': witness is not None,
                    'left_model': first['model'], 'right_model': second['model'],
                    'column_refs': [{**deepcopy(item['column']['forecast_ref']), 'file': item['source_ref']['file']}
                                    for item in (first, second)],
                    'difference_witness': None if witness is None else {'public_input_index': witness,
                        'left_prediction': a['predictions'][witness], 'right_prediction': b['predictions'][witness]},
                    'scope': 'each_selected_parent_version_verified_against_its_saved_literal_context_same_query_binding_inputs'}
        return {'status': 'UNKNOWN', 'changed': None,
            'reason': 'saved_columns_bound_to_both_selected_parent_versions_on_common_inputs_required'}

    def forecast_difference(left, right):
        witnesses, compared, unknown = [], 0, 0
        for first in left:
            second = next((item for item in right if item['scope'] is not None and item['scope'] == first['scope']), None)
            if second is None:
                unknown += 1
                continue
            unknown += len(set(first['columns']) ^ set(second['columns']))
            for model in set(first['columns']) & set(second['columns']):
                columns = [item['columns'][model].get('_retained_forecast') for item in (first, second)]
                if not all(columns) or columns[0]['prediction_inputs'] != columns[1]['prediction_inputs']:
                    unknown += 1
                    continue
                compared += 1
                if columns[0]['predictions'] != columns[1]['predictions'] and not witnesses:
                    i = next(i for i, (a, b) in enumerate(zip(columns[0]['predictions'], columns[1]['predictions'])) if a != b)
                    witnesses.append({'model': model, 'public_input_index': i,
                        'left_prediction': columns[0]['predictions'][i], 'right_prediction': columns[1]['predictions'][i],
                        'column_refs': [item['source_ref'] for item in (first, second)]})
        return {'changed': True if witnesses else False if compared and not unknown else None,
            'comparable_columns': compared, 'unavailable_or_incomparable': unknown,
            'difference_witness': witnesses[0] if witnesses else None,
            'scope': 'fixed_same_model_names_only_same_query_binding_scope_and_public_inputs'}

    def terminal_predictions(row, outcome, root, case):
        saved = row.get('prediction_ref')
        if not saved or not outcome.get('training_ref') or not outcome.get('model_submitted'):
            return None
        try:
            document = resolve(saved, root)
            expected = {'case': case['id'], 'kind': case.get('kind'),
                'evaluation_seed': case['evaluation_seed'], 'rows': plans[0]['evaluation']['rows'],
                'trajectory_length': plans[0]['evaluation'].get('trajectory_length')}
            values = document.get('predictions')
            if (document.get('evaluation_contract') != expected
                    or anchor(document.get('training_ref'), root) != outcome['training_ref']
                    or document.get('model') != outcome.get('submitted_model')
                    or not isinstance(values, list) or len(values) != expected['rows']
                    or any(type(value) not in (int, float) or not isfinite(value) for value in values)):
                return None
            return {'predictions': values, 'source_ref': anchor(saved, root), 'contract': expected}
        except (OSError, ValueError, KeyError, TypeError):
            return None

    rows = [{(row['case'], row['arm']): row for row in summary['trials']} for summary in summaries]
    for case in plans[0]['cases']:
        arms, details, terminal_columns = [], [], []
        for root, indexed, arm in zip(roots, rows, (left_arm, right_arm)):
            row = indexed.get((case['id'], arm))
            outcome = {'status': 'missing_registered_result', 'terminal_mse': None,
                'confirmation_passed': None, 'terminal_passed': None, 'costs': {key: None for key in cost_keys}}
            detail = None
            terminal_column = None
            if row is not None:
                terminal = row.get('terminal') or {}
                if not terminal and row.get('evaluation_ref'):
                    try:
                        terminal = resolve(row['evaluation_ref'], root)
                    except (OSError, ValueError, KeyError):
                        pass
                outcome.update(status=row.get('status'), terminal_mse=terminal.get('mse') if _number(terminal.get('mse')) else None,
                    terminal_passed=terminal.get('passed'), confirmation_passed=row.get('fresh_confirmation_passed'),
                    terminal_status=terminal.get('status'), costs={key: row.get('costs', {}).get(key) for key in cost_keys},
                    external_terminal_cost={'rows': terminal.get('evaluation_rows',
                        plans[0].get('evaluation', {}).get('rows') if 'returncode' in terminal
                        else 0 if terminal.get('status') == 'no_submitted_model' else None),
                        'python_execution_seconds': terminal.get('python_execution_seconds'),
                        'scope': 'outside_agent_training_costs'})
                try:
                    training, state, workspace = documents(root, row, case['id'], arm)
                    outcome.update(model_submitted=training.get('model') is not None, submitted_model=training.get('model'),
                        confirmation_passed=(training.get('fresh_confirmation') or {}).get('passed'),
                        training_ref=reference(workspace / 'training.json'))
                    detail = mechanism(training, state, workspace)
                    terminal_column = terminal_predictions(terminal, outcome, root, case)
                except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                    outcome['mechanism_status'] = 'UNKNOWN_saved_state_unavailable'
                    outcome['mechanism_detail'] = str(error)
            arms.append(outcome)
            details.append(detail)
            terminal_columns.append(terminal_column)
        a, b = arms
        right_failure = None
        if a['terminal_passed'] is True:
            if (b['terminal_passed'] is False or b.get('terminal_status') == 'no_submitted_model'
                    or b.get('model_submitted') is False):
                right_failure = True
            elif b['terminal_passed'] is True:
                right_failure = False
        pair = {'case': case['id'], 'registered_case_ref': reference(roots[0] / 'protocol.json', 'cases', len(result['cases'])),
            'left': a, 'right': b,
            'right_minus_left': {'terminal_mse': b['terminal_mse'] - a['terminal_mse']
                if a['terminal_mse'] is not None and b['terminal_mse'] is not None else None,
                'target_costs': {key: b['costs'][key] - a['costs'][key] if _number(a['costs'][key])
                    and _number(b['costs'][key]) else None for key in cost_keys},
                'external_terminal_costs': {key: b.get('external_terminal_cost', {}).get(key)
                    - a.get('external_terminal_cost', {}).get(key)
                    if _number(a.get('external_terminal_cost', {}).get(key))
                    and _number(b.get('external_terminal_cost', {}).get(key)) else None
                    for key in ('rows', 'python_execution_seconds')}},
            'observed_prediction_degradation': b['terminal_mse'] > a['terminal_mse']
                if a['terminal_mse'] is not None and b['terminal_mse'] is not None else None,
            'right_failure_with_left_success': right_failure,
            'object_causal_effect': 'requires_paired_removal' if intervention_ref is None
                else 'requires_validated_removal_and_matched_outcome',
            'selected_parent_terminal_predictions': {'status': 'UNKNOWN', 'changed': None,
                'reason': 'saved_complete_vectors_bound_to_each_submitted_parent_and_same_terminal_contract_required'},
            'mechanism': {'status': 'UNKNOWN_saved_state_unavailable'}}
        if all(terminal_columns):
            first, second = terminal_columns
            i = next((i for i, (x, y) in enumerate(zip(first['predictions'], second['predictions'])) if x != y), None)
            pair['selected_parent_terminal_predictions'] = {'status': 'recorded', 'changed': i is not None,
                'column_refs': [item['source_ref'] for item in terminal_columns], 'rows': len(first['predictions']),
                'difference_witness': None if i is None else {'row_index': i,
                    'left_prediction': first['predictions'][i], 'right_prediction': second['predictions'][i]},
                'scope': 'already_executed_independent_terminal_after_all_training_frozen_no_new_prediction'}
        if all(details):
            first, second = details
            parents_known = first['parent_program'] is not None and second['parent_program'] is not None
            closed = [item['closed_parent'] for item in details]
            executable_comparison = ({'status': 'recorded',
                'changed': closed[0]['source'] != closed[1]['source'],
                'model_refs': [item['source_ref'] for item in closed],
                'scope': 'existing_closed_literal_exports_no_external_bindings'}
                if all(closed) else {'status': 'UNKNOWN', 'changed': None,
                    'reason': 'both_existing_closed_literal_exports_required'})
            pair['mechanism'] = {'status': 'recorded', 'state_refs': [item['state_ref'] for item in details],
                'candidate_programs_changed': first['candidate_programs'] != second['candidate_programs'],
                'candidate_comparison_scope': 'saved_native_and_exported_revision_Programs_not_opaque_Python_or_ID_counts',
                'candidate_program_refs': [item['candidate_program_refs'] for item in details],
                'parent_model_ids_changed': first['parent_model'] != second['parent_model'],
                'parent_program_changed': first['parent_program'] != second['parent_program'] if parents_known else None,
                'selected_parent_executable_comparison': executable_comparison,
                'parent_refs': [item['parent_ref'] for item in details],
                'actual_before_after_revision_refs': [item['actual_parent_changes'] for item in details],
                'selected_experiment_actions_changed': first['proposed_actions'] != second['proposed_actions'],
                'selected_experiment_refs': [item['proposal_refs'] for item in details],
                'requested_query_arguments_changed': first['requested_queries'] != second['requested_queries'],
                'executed_research_actions_changed': first['executed_actions'] != second['executed_actions']
                    if not first['unknown_query_refs'] and not second['unknown_query_refs'] else None,
                'executed_action_refs': [item['executed_action_refs'] for item in details],
                'unknown_query_refs': [item['unknown_query_refs'] for item in details],
                'non_acquisition_query_refs': [item['non_acquisition_query_refs'] for item in details],
                'partial_or_failed_query_refs': [item['failed_query_refs'] for item in details],
                'stored_forecast_comparison': forecast_difference(first['forecasts'], second['forecasts']),
                'selected_parent_forecast_comparison': parent_forecast_difference(
                    first['parent_forecasts'], second['parent_forecasts']),
                'objects': [item['objects'] for item in details]}
        result['cases'].append(pair)
    result.update(status='recorded_comparison', registered_case_count=len(result['cases']),
        paired_terminal_case_count=sum(row['right_minus_left']['terminal_mse'] is not None for row in result['cases']),
        full_source_plus_target_net_cost=('shared_source_cancels_use_recorded_target_and_external_terminal_differences'
            if shared_source else 'UNKNOWN_shared_source_removal_not_established'
            if left_arm == right_arm == 'pr3' or same_scene_removal
            else 'UNKNOWN_missing_comparable_B_source_account'),
        source_learning_amortization={'actual_target_count': len(result['cases']) if
            result['registered_phase'] == 'target' else 0 if result['registered_phase'] == 'source' else None,
            'project_development_cost_divided_by_target_count': False,
            'C_development_cost_divided_by_target_count': False,
            'per_target_C_source_learning_compute_interval': None})
    return result


def evaluate_saved_scene_predictions(study, source_training_ref, *, prefix_records=6,
                                     prefix_rows=97, expected_rows=48,
                                     arms=('fixed_pr2', 'pr3')):
    """Freeze all registered finished parents before scoring the already-paid later rows.

    This is a development diagnostic. It never fits, queries or confirms; its
    predictions use each branch's remaining registered compute/Python allowance.
    Non-literal Python parents remain unmeasured: the existing literal contract
    cannot establish absence of a fit hook that the ordinary evaluator calls.
    Existing scientific results are unchanged.
    """
    from time import monotonic
    from .code_agent import (_closed_temporal_export, _copy_execution_context,
                             evaluate_python_model, python_model_context)

    arms = tuple(arms)
    root = Path(study).resolve()
    path = root / 'post97_predictions.json'
    if path.is_file():
        cached = json.loads(path.read_text(encoding='utf-8'))
        if tuple(entry['arm'] for entry in cached.get('arms', [])) != arms:
            raise ValueError('saved prediction receipt belongs to different registered branches')
        return cached

    def read(file):
        return json.loads(Path(file).read_text(encoding='utf-8'))

    def write(file, value):
        file.parent.mkdir(parents=True, exist_ok=True)
        write_text_incrementally(file, json.dumps(value, ensure_ascii=False))

    def resolve(reference):
        file = Path(reference['file'])
        value = read(file if file.is_absolute() else root / file)
        for key in reference.get('json_path', []):
            value = value[key]
        return value

    receipt = {'status': 'UNKNOWN',
        'scope': 'already_exposed_paid_history_development_prediction_not_independent_confirmation',
        'prediction_conditioning': 'original_recorded_control_and_history_inputs_not_closed_loop_rollout',
        'source_training_ref': deepcopy(source_training_ref),
        'prefix_records': prefix_records, 'prefix_rows': prefix_rows,
        'expected_rows': expected_rows, 'arms': [], 'new_fits': 0,
        'new_observations': 0, 'new_model_calls': 0,
        'common_repair_is_PR3_effect': False,
        'comparison': {'changed': None, 'right_minus_left_mse': None}}
    summary = read(root / 'summary.json')
    trials = {row['arm']: row for row in summary.get('trials', [])}
    if summary.get('status') != 'completed' or set(trials) != set(arms):
        return {**receipt, 'reason': 'both_training_jobs_must_finish_before_freeze'}
    limits = read(root / 'protocol.json')['limits']
    # Finish all checks and copy every retained parent before opening the
    # historical source with the later labels. Copies retain model dependencies.
    training_prefix = None
    for arm in arms:
        trial = trials[arm]
        training_ref = trial.get('result_ref') or trial.get('training_ref')
        if not training_ref:
            return {**receipt, 'reason': 'completed_training_reference_missing'}
        training = resolve(training_ref)
        records = training['paid_observations']
        if len(records) != prefix_records or sum(len(item['rows']) for item in records) != prefix_rows:
            return {**receipt, 'reason': 'training_contains_a_different_paid_prefix'}
        if training_prefix is not None and records != training_prefix:
            return {**receipt, 'reason': 'finished_branches_have_different_paid_material'}
        training_prefix = records
        parent = training.get('evaluated_model') or {}
        if not parent.get('model'):
            return {**receipt, 'reason': 'finished_retained_parent_unavailable'}
        training_file = Path(training_ref['file'])
        workspace = (training_file if training_file.is_absolute() else root / training_file).resolve().parent
        directory, model = python_model_context(workspace, parent['model'])
        frozen = root / 'post97_frozen_models' / arm / '_evaluated_model'
        _copy_execution_context(directory, frozen, [model])
        before = deepcopy(training['costs'])
        entry = {'arm': arm, 'status': 'UNKNOWN', 'training_ref': deepcopy(training_ref),
            'selected_parent_ref': _ref(training_ref, 'evaluated_model'),
            'frozen_model': {'workspace': str(frozen), 'model': model},
            'closed_literal_without_fit': _closed_temporal_export(frozen, model),
            'costs_before': before, 'costs_after': deepcopy(before),
            'charged_costs': {'compute_cost': 0, 'python_execution_seconds': 0.},
            'limits': {'compute_budget': limits['compute_budget'],
                       'execution_seconds': limits['execution_seconds']}}
        receipt['arms'].append(entry)
    record_refs = [_ref(source_training_ref, 'paid_observations', index)
                   for index in range(prefix_records, prefix_records + 4)]
    receipt['paid_record_refs'] = record_refs
    freeze_file = root / 'post97_parent_freeze.json'
    write(freeze_file, {**deepcopy(receipt), 'status': (
        'both_parents_frozen_before_later_labels' if len(arms) == 2
        else 'all_parents_frozen_before_later_labels'),
        'later_labels_opened': False})
    receipt['freeze_ref'] = {'file': str(freeze_file), 'json_path': []}
    source = resolve(source_training_ref)
    original = source['paid_observations']
    if (len(original) != prefix_records + 4
            or original[:prefix_records] != training_prefix):
        receipt['reason'] = 'historical_paid_record_boundary_changed'
        write(path, receipt)
        return receipt
    rows = [row for record in original[prefix_records:] for row in record['rows']]
    if len(rows) != expected_rows:
        receipt['reason'] = 'historical_later_row_count_changed'
        write(path, receipt)
        return receipt
    receipt['rows'] = len(rows)
    for entry in receipt['arms']:
        before, after = entry['costs_before'], entry['costs_after']
        remaining = limits['execution_seconds'] - before['python_execution_seconds']
        available = limits['compute_budget'] - before['compute_cost']
        if not entry['closed_literal_without_fit']:
            entry['reason'] = 'frozen_parent_without_fit_contract_unavailable'
            continue
        if available < len(rows) or remaining <= 0:
            entry['reason'] = 'remaining_registered_prediction_budget_exhausted'
            continue
        after['compute_cost'] += len(rows)
        started = monotonic()
        try:
            frozen = entry['frozen_model']
            score = evaluate_python_model(frozen['workspace'], frozen['model'], rows,
                timeout_seconds=remaining, research_compute_budget=available - len(rows))
            after['compute_cost'] += score.get('research_prediction_count', 0)
            # The sole vectors live here; targets and inputs stay in the paid
            # records. The caller reads this receipt instead of predicting again.
            entry['evaluation'] = score
            entry['status'] = ('recorded' if score.get('returncode') == 0
                               and score.get('predictions') is not None else 'UNKNOWN')
        except (OSError, ValueError, TypeError, KeyError) as error:
            entry['reason'] = f'{type(error).__name__}: {error}'
        finally:
            after['python_execution_seconds'] += monotonic() - started
            entry['charged_costs'] = {key: after[key] - before[key]
                                     for key in entry['charged_costs']}
        # Keep a completed or failed arm before attempting the other one.
        write(path, receipt)
    if all(entry['status'] == 'recorded' for entry in receipt['arms']):
        left = receipt['arms'][0]['evaluation']
        receipt['status'] = 'recorded_comparison'
        comparisons = []
        for entry in receipt['arms'][1:]:
            right = entry['evaluation']
            comparisons.append({'left_arm': arms[0], 'right_arm': entry['arm'],
                'changed': left['predictions'] != right['predictions'],
                'right_minus_left_mse': right['mse'] - left['mse'],
                'mse_threshold': limits['confirmation_mse_threshold'],
                'left_within_threshold': left['mse'] <= limits['confirmation_mse_threshold'],
                'right_within_threshold': right['mse'] <= limits['confirmation_mse_threshold']})
        receipt['comparison'] = {key: value for key, value in comparisons[0].items()
                                 if key not in ('left_arm', 'right_arm')}
        if len(arms) > 2:
            receipt['comparisons'] = comparisons
    write(path, receipt)
    return receipt


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
    if query.get('effect_kind') == SUCCESSOR_EFFECT_KIND:
        def location(value):
            path = Path(value)
            return path if path.is_absolute() else Path(workspace) / path
        if not query.get('left_study'):
            return {'status': 'scoped_unknown', 'reason': 'registered_study_reference_required',
                'compute_cost': 0, 'new_method_executions': 0, 'independent_confirmation': False}
        result = registered_successor_effect(location(query['left_study']),
            location(query['right_study']) if query.get('right_study') else None,
            left_arm=query.get('left_arm', 'fixed_pr2'), right_arm=query.get('right_arm', 'pr3'),
            source_cost_ref=query.get('source_cost_ref'), intervention_ref=query.get('intervention_ref'))
        return {'status': 'answered' if result['status'] == 'recorded_comparison' else 'scoped_unknown',
            'value': result, 'compute_cost': 0, 'new_method_executions': 0,
            'independent_confirmation': False, 'reason': result.get('reason'),
            'source_ref': result.get('study_refs', [None])[0],
            'scope': {'effect_kind': SUCCESSOR_EFFECT_KIND, 'study_refs': result.get('study_refs', []),
                'independent_new_confirmation': False, 'general_method_superiority': 'not_established'}}
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
