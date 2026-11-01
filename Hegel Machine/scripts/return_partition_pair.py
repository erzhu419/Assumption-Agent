#!/usr/bin/env python3
"""Return a frozen native partition pair through the existing method learner."""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from math import isclose
from pathlib import Path
from statistics import mean

from scripts.run_native_decision_pair import ref
from scripts.run_partial_role_parent import load, save
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.common_object_research import (
    CommonObjectDependencyMethods, record_native_parent_research_return)
from hegel_machine.cognition.method_action_research import read_method_reference
from hegel_machine.cognition.method_conditions import method_profile
from hegel_machine.cognition.method_feedback import ensure_python_method_topic
from hegel_machine.cognition.models import ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.object_continuation import observations
from hegel_machine.cognition.question_routing import consume_method_return_agenda
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


def bind_methods(source, output, retained_pair=None, *, artifact_refs=None, evaluation_arm_names=None):
    """Bind the actual callee and unchanged frozen bodies across new sources."""
    pair = read_method_reference(retained_pair, output) if retained_pair else None
    actual = load(source / 'frozen_methods.json')
    methods = {}
    names = evaluation_arm_names or {'baseline': 'P0', 'challenger': 'Pi1'}
    for side, arm in names.items():
        if artifact_refs:
            artifact_ref = artifact_refs[arm]
            artifact = read_method_reference(artifact_ref, output)
            frozen = read_method_reference(artifact['execution']['research_program_ref'], output)
            if frozen != actual[arm]:
                raise ValueError('The declared method artifact differs from the actual frozen invocation')
        elif pair:
            old = read_method_reference(pair[side + '_ref'], output)
            artifact_ref = old['method_artifact_ref']
            artifact = read_method_reference(artifact_ref, output)
            frozen = read_method_reference(artifact['execution']['research_program_ref'], output)
            if frozen != actual[arm]:
                raise ValueError('The retained method body differs from the actual frozen invocation')
        else:
            artifact = {'id': 'native:transfer_partition:' + arm, 'version': 1,
                'kind': 'native_research_segment', 'execution': {
                    'entry': 'replay_research_program',
                    'selection': 'UnifiedResearchAgent.compare_research_decision_return',
                    'research_program_ref': ref(source / 'frozen_methods.json', arm)},
                'source_ref': ref(source / 'runtime/src/hegel_machine/cognition/research_programs.py')}
            artifact_ref = ref(output / (side + '_method.json'))
            save(Path(artifact_ref['file']), artifact)
        methods[side] = {'method_ref': {key: artifact[key] for key in ('id', 'version')},
            'method_artifact_ref': artifact_ref, 'execution': artifact['execution']}
    return methods


def _pre_call_material(source, cfg, compared):
    """Recover the same paid rows and the ordinary complete-pool winner."""
    from scripts.validate_partition_successor import retained_entry

    source = Path(source)
    rows, trace, records = [], [], []
    for location in compared['data_refs']:
        paid = read_method_reference(location, source)
        batch = paid.get('batch', paid)
        for row in observations(batch['observations']):
            if not records or not row.history:
                records.append({'row_indices': [], 'rows': [], 'source_ref': deepcopy(location)})
            records[-1]['row_indices'].append(len(rows))
            records[-1]['rows'].append(asdict(row))
            rows.append(row)
        if paid.get('probe'):
            trace.append({'action': 'observe', 'details': {'sequence': paid['probe']['sequence']}})
    agent = UnifiedResearchAgent(ResearchConfig(**cfg['research_config']))
    pool = compared['scores']
    lowest = min(item['loss'] for item in pool)
    contenders = [(index, dict(retained_entry(item['candidate_ref'], source, index),
        validation_loss=item['loss'])) for index, item in enumerate(pool)
        if item['loss'] <= lowest + agent.config.minimum_improvement]
    best = agent._best([item for _, item in contenders])
    index = next(index for index, item in contenders if item is best)
    before = load(source / 'prepared_state.json')
    return rows, trace, records, pool, best, index, before, agent


def _pre_call_evidence_basis(source, cfg, compared, material):
    rows, _, _, pool, best, index, before, _ = material
    public = read_method_reference(before['public_spec_ref'], source)
    return {'model_ref': deepcopy(compared['scores'][index]['candidate_ref']),
        'data_refs': deepcopy(compared['data_refs']), 'rows': len(rows),
        'full_pool_size': len(pool), 'selected_loss': best['validation_loss'],
        'fallback': before['fallback'], 'risk_quantile': public.get('risk_quantile', .5),
        'selection_minimum_improvement': cfg['research_config'].get('minimum_improvement', .002),
        'squared_error_threshold': cfg['quality_threshold']}


def _retained_research_program(location, source):
    """Recover the two saved binding forms used by the retained pool."""
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.research_drafts import _rebind_step
    from hegel_machine.cognition.structural_research_program import research_program_fit_contract

    payload = read_method_reference(location, source)
    program = Program.from_dict((payload.get('candidate') or payload)['program'])
    procedure = payload.get('research_program')
    if procedure is not None:
        return (deepcopy(procedure), [deepcopy(location)]) if (
            research_program_fit_contract(procedure)['program'] == program) else (None, [])
    origin = payload.get('source_candidate_ref')
    if not origin:
        return None, []
    if payload.get('feature_mapping') is not None:
        mapping = payload['feature_mapping']
    elif payload.get('role_binding_ref'):
        forward = read_method_reference(payload['role_binding_ref'], source)
        if len(set(forward.values())) != len(forward):
            return None, []
        mapping = {value: name for name, value in forward.items()}
    else:
        return None, []
    old = read_method_reference(origin, source)
    old_program = Program.from_dict((old.get('candidate') or old)['program'])
    if old_program.remap_features(mapping) != program:
        return None, []
    procedure, references = _retained_research_program(origin, source)
    if procedure is None:
        return None, []
    rebound = _rebind_step(procedure, mapping)
    if research_program_fit_contract(rebound)['program'] != program:
        return None, []
    return rebound, [deepcopy(location), *references]


def ensure_pre_call_evidence(source, cfg, compared, output, budget, *, predictions=None):
    """Retain the actual pre-call winner's paid column, or fund one complete readout."""
    from hegel_machine.cognition.evaluation import point_loss, predict
    from hegel_machine.cognition.method_conditions import paid_transition_evidence
    from hegel_machine.cognition.transfer_partitions import transfer_partition_view

    source, output = Path(source), Path(output)
    if not compared.get('full_comparison'):
        raise ValueError('Pre-call evidence requires the actual complete-pool selection')
    material = _pre_call_material(source, cfg, compared)
    rows, _, records, _, best, index, _, _ = material
    basis = _pre_call_evidence_basis(source, cfg, compared, material)
    path = output / 'pre_call_evidence.json'
    references = ([compared['pre_call_evidence_ref']] if compared.get('pre_call_evidence_ref') else [])
    if path.exists():
        references.append(ref(path))
    for reference in references:
        prior = read_method_reference(reference, source)
        if prior.get('basis') == basis:
            return {'status': 'pre_call_evidence_reused', 'evidence_ref': deepcopy(reference),
                'new_compute_cost': 0, 'new_predictions': 0, 'budget_incomplete': False}
    if path.exists():
        raise ValueError('Preserve the previous pre-call evidence in its original output directory')

    prediction_ref = compared['scores'][index].get('predictions_ref')
    if predictions is None and prediction_ref is None and compared.get('reused_comparison_ref'):
        previous = read_method_reference(compared['reused_comparison_ref'], source)
        if previous.get('data_refs') == compared['data_refs']:
            prediction_ref = next((item['predictions_ref'] for item in previous.get('scores', [])
                if item.get('predictions_ref') and basis['model_ref'] in item.get('candidate_refs', [])), None)
    if predictions is None and prediction_ref is not None:
        predictions = read_method_reference(prediction_ref, source)
    spent = 0
    if predictions is None:
        if not budget.spend(len(rows)):
            return {'status': 'pre_call_evidence_not_funded', 'evidence_ref': None,
                'new_compute_cost': 0, 'new_predictions': 0, 'required_compute': len(rows),
                'budget_incomplete': True}
        predictions = [predict(best['candidate'].program, row, basis['fallback']) for row in rows]
        spent = len(rows)
    if len(predictions) != len(rows):
        raise ValueError('The retained pre-call prediction column must cover all original paid rows')
    scored_loss = mean(point_loss(value, row.target, basis['risk_quantile'])
        for value, row in zip(predictions, rows))
    if not isclose(scored_loss, best['validation_loss'], rel_tol=1e-10, abs_tol=1e-15):
        raise ValueError('The pre-call column differs from the actual paid winner score')
    payload = read_method_reference(basis['model_ref'], source)
    program_ref = {**deepcopy(basis['model_ref']), 'json_path': [
        *basis['model_ref'].get('json_path', []),
        *(['candidate', 'program'] if 'candidate' in payload else ['program'])]}
    procedure, procedure_refs = _retained_research_program(basis['model_ref'], source)
    binding = (transfer_partition_view(procedure) if procedure else
        {'status': 'binding_unavailable', 'groups': []})
    groups = binding.get('groups', [])
    ready = binding['status'] == 'ready'
    terms = binding.get('term_nodes', [])
    summary = {'status': binding['status'], 'count': len(groups) if ready else None,
        'fit_columns': len(terms) if ready else None,
        'bound_occurrences': sum(len(group['occurrences']) for group in groups) if ready else None,
        'groups': [{'parameter_index': group['parameter_index'],
            'term_indices': [terms.index(item['term_node']) for item in group['occurrences']],
            'occurrence_depths': [len(item['path']) for item in group['occurrences']],
            'source_interval_count': len(group.get('source_intervals', []))} for group in groups] if ready else []}
    program = best['candidate'].program
    program = program.expanded() if program.op == 'call' else program
    body = program.children[0] if program.op == 'state_replay' else program
    body = body.expanded() if body.op == 'call' else body
    guard = body.children[0] if body.op == 'if' else None
    guard_depth = guard.lag if guard is not None and guard.op == 'available' else None
    evidence = {'status': 'pre_call_evidence_ready', 'basis': basis,
        'model_ref': deepcopy(basis['model_ref']), 'program_ref': program_ref,
        'research_program_source_refs': procedure_refs, 'score_loss': scored_loss,
        'data_refs': deepcopy(compared['data_refs']), 'predictions': list(predictions),
        'prediction_source_ref': deepcopy(prediction_ref),
        'transition_evidence': paid_transition_evidence(records, rows, predictions,
            guard_depth=guard_depth, squared_error_threshold=cfg['quality_threshold'], source_refs=[ref(path)]),
        'parameter_groups': summary,
        'relation_binding': {key: deepcopy(value) for key, value in binding.items() if key != 'parameter_terms'},
        'new_compute_cost': spent, 'new_predictions': spent,
        'new_fits': 0, 'new_observations': 0, 'uses_post_method_results': False,
        'budget_incomplete': False}
    save(path, evidence)
    return {'status': evidence['status'], 'evidence_ref': ref(path),
        'new_compute_cost': spent, 'new_predictions': spent, 'budget_incomplete': False}


def paid_pre_call_profile(source, cfg, compared, *, evidence_ref=None):
    """Recover public inputs; consume a saved evidence column without predicting."""
    source = Path(source)
    material = _pre_call_material(source, cfg, compared)
    rows, trace, _, pool, best, _, before, agent = material
    if before.get('public_spec_ref'):
        public = TaskSpec(**read_method_reference(before['public_spec_ref'], source))
        interface_basis = {'kind': 'retained_pre_call_public_spec',
            'public_spec_ref': before['public_spec_ref']}
    else:
        # The first frozen v9 successor retained its task registration but did
        # not yet export public_spec.json. Reconstruct only that public view;
        # all observations and learned roles still come from paid records.
        with AnonymousObjectLab(**cfg['task']) as laboratory:
            public = laboratory.spec_for(('fine',))
        interface_basis = {'kind': 'public_interface_reconstructed_from_registration',
            'registration_ref': ref(source / 'registration.json'), 'new_observations': 0}
    model = read_method_reference(cfg['source_model_ref'], source)
    spec = agent._knowledge_spec(replace(public, context={**public.context, 'object_model': model}))
    evidence_ref = evidence_ref or compared.get('pre_call_evidence_ref')
    evidence = read_method_reference(evidence_ref, source) if evidence_ref else None
    if evidence is not None and evidence['basis'] != _pre_call_evidence_basis(source, cfg, compared, material):
        raise ValueError('Method evidence belongs to another pre-call winner, data or scoring contract')
    state = {'stage': 'native_direction_choice', 'comparison': {
        'stale': False, 'paid_rows': len(rows), 'models_requested': len(pool),
        'models_completed': len(pool)}, 'budget': {'remaining_compute': cfg['equal_method_compute_limit']}}
    if evidence is not None:
        state['parameter_groups'] = deepcopy(evidence['parameter_groups'])
    profile = method_profile(spec, rows, incumbent=best['candidate'], trace=trace,
        residual_loss=best['validation_loss'], minimum_improvement=cfg['quality_threshold'],
        research_state=state, transition_evidence=evidence.get('transition_evidence') if evidence else None)
    if evidence is not None and evidence['transition_evidence']['status'] != 'observed':
        profile['residual']['transition'] = {key: deepcopy(value)
            for key, value in evidence['transition_evidence'].items() if key != 'source_refs'}
    profile['recovery_basis'] = {'kind': 'reconstructed_from_retained_pre_call_records',
        'comparison_ref': ref(source / 'common_comparison.json'),
        'public_interface': interface_basis, 'source_model_ref': cfg['source_model_ref'],
        'control_probe_refs': compared['data_refs'],
        'uses_post_method_predictions': False}
    if evidence_ref:
        profile['recovery_basis']['pre_call_evidence_ref'] = deepcopy(evidence_ref)
    return profile


def return_pair(source, output, prior_review, retained_pair=None, *, allocation_ref=None,
                validation_cohort_refs=None):
    """Read paid outcomes only; this entry never fits, predicts or acquires."""
    source, output, prior_review = Path(source), Path(output), Path(prior_review)
    if (output / 'receipt.json').exists():
        return {**load(output / 'receipt.json'), 'new_compute_cost': 0, 'reused': True}
    cfg, receipt = load(source / 'registration.json'), load(source / 'receipt.json')
    names = cfg.get('evaluation_arm_names', {'baseline': 'P0', 'challenger': 'Pi1'})
    development = (cfg.get('evaluation_role') == 'paid_working_comparison'
        and receipt.get('status') == 'paid_methods_completed'
        and all(item.get('execution_complete') and item.get('full_comparison')
                for item in receipt.get('arms', {}).values())
        and set(receipt.get('arms', {})) == set(names.values()))
    if cfg.get('evaluation_role') == 'paid_working_comparison' and not development:
        return {'status': 'incomplete_method_not_negative', 'new_compute_cost': 0,
            'source_result_ref': ref(source / 'receipt.json'), 'method_pair_created': False,
            'independent_method_gain': None}
    if not development and (receipt.get('confirmation') or {}).get('status') != 'frozen_pair_readout_completed':
        return {'status': 'completed_frozen_pair_required', 'new_compute_cost': 0}
    owner = next(q for q in load(prior_review / 'common/questions.json') if q['id'] == 'task:method_choice')
    allocation = read_method_reference(allocation_ref, output) if allocation_ref else None
    if allocation and allocation['new_observation_limit'] != 0:
        raise ValueError('A feedback allocation only consumes the existing paid records')
    remaining = allocation['maximum_compute'] if allocation else receipt['remaining_compute']
    limit = min(cfg.get('method_return_compute_limit', 100), remaining)
    if limit < 3 + len(owner['method_use_plan']['native_return_refs']):
        raise ValueError('Original balance cannot fund ordinary method return')
    output.mkdir(parents=True, exist_ok=True)
    methods = bind_methods(source, output, retained_pair,
        artifact_refs=cfg.get('native_method_artifact_refs'), evaluation_arm_names=names)
    compared = load(source / 'common_comparison.json')
    profile = paid_pre_call_profile(source, cfg, compared)
    save(output / 'profile.json', profile)
    save(output / 'data_contract.json', {'observations': compared['data_refs']})
    save(output / 'split_contract.json', {'training_rows': cfg['training_rows'],
        'working_rows': cfg['working_rows'], 'working_use': 'available_adaptive_development'})
    save(output / 'quality_contract.json', {'threshold': cfg['quality_threshold'], 'metric': 'MSE'})
    save(output / 'resource_contract.json', {'compute_limit': cfg['equal_method_compute_limit'],
        'comparison_reserve': cfg['per_arm_comparison_reserve'], 'observation_limit': 0,
        'time_limits': {'active_wall_seconds': cfg['active_wall_seconds']},
        'registered_global_contract_ref': ref(source / 'registration.json')})
    save(output / 'evaluation_contract.json', {'working_ref': ref(source / 'common_comparison.json'),
        'role': 'paid_working_comparison' if development else 'paired_future',
        **({} if development else {'frozen_readout_ref': ref(source / 'confirmation/receipt.json')})})
    contracts = {key + '_ref': ref(output / (key + '.json')) for key in (
        'data_contract', 'split_contract', 'quality_contract', 'resource_contract', 'evaluation_contract')}
    contracts = {key.replace('data_contract_ref', 'data_ref').replace('split_contract_ref', 'split_ref')
        .replace('evaluation_contract_ref', 'evaluation_ref'): value for key, value in contracts.items()}
    contracts.update(start_state_ref=ref(source / 'prepared_state.json'), profile_ref=ref(output / 'profile.json'))
    for side, arm in names.items():
        actual = receipt['arms'][arm]
        save(output / (side + '_outcome.json'), {**contracts, **methods[side],
            'status': 'completed', 'execution_complete': actual['execution_complete'],
            'completed': actual['selected_loss'] <= cfg['quality_threshold'],
            'coverage': cfg['training_rows'] + cfg['working_rows'], 'loss': actual['selected_loss'],
            'full_comparison': actual['full_comparison'], 'selected_ref': actual['selected_ref'],
            'actual_outcome_ref': ref(source / arm / 'outcome.json'),
            'costs': {'compute_cost': actual['compute_cost'], 'observation_cost': 0, 'measurement_cost': 0},
            'timing': {'activity_wall_seconds': receipt['execution_timing']['phase_seconds'][arm]}})
    pair = {'status': 'returned', 'comparison_kind': 'matched_native_segments',
        'baseline_ref': ref(output / 'baseline_outcome.json'),
        'challenger_ref': ref(output / 'challenger_outcome.json'),
        'baseline_action_content': methods['baseline']['execution'],
        'challenger_action_content': methods['challenger']['execution'],
        'physical_source_ref': cfg['source_task_ref'], 'origin_ref': ref(source / 'registration.json'),
        'decision_ref': ref(source / 'prepared_state.json'), 'data_role': cfg.get('data_role', 'development'),
        'identification_target': 'fixed_data_method_effect',
        'evaluation_role': 'paid_working_comparison' if development else 'paired_future',
        'evaluation_arm_names': deepcopy(names),
        **({} if development else {'paired_future_ref': ref(source / 'confirmation/receipt.json'),
            'frozen_predictors_ref': ref(source / 'confirmation/frozen_predictors.json')})}
    cohort = validation_cohort_refs or cfg.get('condition_validation_cohort_refs')
    if cohort:
        pair['condition_validation_cohort_refs'] = deepcopy(cohort)
    if cfg.get('data_role') == 'condition_validation' and cfg.get('frozen_conditions_ref'):
        pair['frozen_conditions_ref'] = deepcopy(cfg['frozen_conditions_ref'])
        pair['condition_prediction_ref'] = deepcopy(cfg['condition_prediction_ref'])
    save(output / 'pair.json', pair)
    save(output / 'research_receipt.json', {'status': ('paid_partition_pair_returned' if development
        else 'frozen_partition_pair_returned'),
        'method_pair_ref': ref(output / 'pair.json'), **methods['challenger'],
        'source_task_ref': cfg['source_task_ref'], 'cost_refs': [ref(source / 'receipt.json')]})
    save(output / 'previous_topic.json', {'question': owner, 'questions': [owner], 'methods': []})
    common = CommonObjectDependencyMethods(output / 'common', [], budget=SearchBudget(limit),
        actions=[], source_task_ref=cfg['source_task_ref']).common
    ensure_python_method_topic(common, parent_question_ref={'question_id': 'task'},
        topic_ref=ref(output / 'previous_topic.json'))
    mission = deepcopy(owner['mission'])
    mission['allocation']['compute'] += limit
    feedback = record_native_parent_research_return(common, source_task_ref=cfg['source_task_ref'],
        result_ref=ref(output / 'research_receipt.json'), before_ref=ref(source / 'prepared_state.json'),
        cost_refs=[ref(source / 'receipt.json')], mission=mission)
    reviewed = consume_method_return_agenda(common, feedback, None, compute_limit=limit)
    result = {'status': reviewed['status'], 'return': reviewed, 'method_pair_ref': ref(output / 'pair.json'),
        'compute_cost': common.costs['compute_cost'], 'new_compute_cost': common.costs['compute_cost'],
        'cumulative_compute': receipt['cumulative_compute'] + common.costs['compute_cost'],
        'remaining_compute': receipt['remaining_compute'] - common.costs['compute_cost'],
        'source_cost_refs': [ref(source / 'receipt.json')], 'new_fits': 0, 'new_predictions': 0,
        'new_observations': 0, 'model_calls': 0, 'frozen_result_preserved': True,
        'condition_learning_role': ('new_physical_source_frozen_condition_validation_on_working_return'
            if development and cfg.get('data_role') == 'condition_validation'
            else 'exposed_paid_working_comparison_not_new_validation' if development
            else 'exposed_after_frozen_prediction_not_new_validation')}
    if allocation:
        result.update(allocation_ref=deepcopy(allocation_ref), allocation_id=allocation['allocation_id'],
            cumulative_compute=common.costs['compute_cost'],
            remaining_compute=remaining-common.costs['compute_cost'],
            ledger_scope='this_new_feedback_occurrence_only_prior_paid_accounts_unchanged',
            original_ledger_ref=ref(source / 'receipt.json'),
            prior_method_review_ref=ref(prior_review / 'receipt.json'))
    save(output / 'receipt.json', result)
    return result


def merge_pair_returns(prior_review, output, pair_refs, allocation_ref):
    """Return actual forked pairs to the formed-method topic without redoing them."""
    prior_review, output = Path(prior_review).resolve(), Path(output).resolve()
    basis = {'prior_review': str(prior_review), 'pair_refs': deepcopy(pair_refs),
             'allocation_ref': deepcopy(allocation_ref)}
    if (output / 'receipt.json').exists():
        previous = load(output / 'receipt.json')
        if previous['basis'] != basis:
            raise ValueError('Retain the completed merge and its original inputs')
        return {**previous, 'new_compute_cost': 0, 'reused': True}
    allocation = read_method_reference(allocation_ref, output)
    if allocation['new_observation_limit'] != 0:
        raise ValueError('Pair feedback only consumes already paid records')
    limit = allocation['maximum_compute']
    owner = next(q for q in load(prior_review / 'common/questions.json') if q['id'] == 'task:method_choice')
    if not pair_refs or limit < 1 + len(owner['method_use_plan']['native_return_refs']) + len(pair_refs):
        raise ValueError('The allocated balance cannot fund the combined native-return review')
    retained = deepcopy(owner)
    for field in ('routing_execution', 'method_use_result', 'completed_agenda_basis'):
        retained.pop(field, None)
    # The new occurrence has its own allowance; previous usage stays in the
    # referenced topic, rather than being erased from that historical bill.
    mission = {'purpose': 'method_learning',
        'mission_id': allocation['allocation_id'] + ':' + str(output),
        'allocation': {'compute': limit}, 'resource_usage': {'compute': 0}}
    retained['mission'] = mission
    output.mkdir(parents=True, exist_ok=True)
    save(output / 'previous_topic.json', {'question': retained, 'questions': [retained], 'methods': []})
    common = CommonObjectDependencyMethods(output / 'common', [], budget=SearchBudget(limit), actions=[]).common
    ensure_python_method_topic(common, parent_question_ref={'question_id': 'task'},
        topic_ref=ref(output / 'previous_topic.json'))
    source_cost_refs = [ref(prior_review / 'receipt.json')]
    for index, pair_ref in enumerate(pair_refs):
        pair = read_method_reference(pair_ref, prior_review)
        challenger = read_method_reference(pair['challenger_ref'], prior_review)
        costs = []
        for side in ('baseline', 'challenger'):
            outcome_ref = pair[side + '_ref']
            costs.append({**outcome_ref, 'json_path': [*outcome_ref.get('json_path', []), 'costs']})
        original_return = (prior_review / pair_ref['file']).parent / 'receipt.json'
        if original_return.exists():
            costs.append(ref(original_return))
        for value in costs:
            if value not in source_cost_refs:
                source_cost_refs.append(value)
        returned_ref = ref(output / f'pair_return_{index:03d}.json')
        save(Path(returned_ref['file']), {'status': 'paid_pair_merged',
            'method_pair_ref': deepcopy(pair_ref), 'method_ref': challenger['method_ref'],
            'method_artifact_ref': challenger['method_artifact_ref'],
            'source_task_ref': pair['physical_source_ref'], 'cost_refs': costs})
        feedback = record_native_parent_research_return(common,
            source_task_ref=pair['physical_source_ref'], result_ref=returned_ref,
            before_ref=pair.get('decision_ref'), cost_refs=costs, mission=mission)
    reviewed = consume_method_return_agenda(common, feedback, None, compute_limit=limit)
    spent = common.costs['compute_cost']
    save(output / 'common/questions.json', common.questions)
    result = {'status': reviewed['status'], 'basis': basis, 'return': reviewed,
        'merged_pair_refs': deepcopy(pair_refs), 'source_cost_refs': source_cost_refs,
        'compute_cost': spent, 'new_compute_cost': spent, 'remaining_compute': limit - spent,
        'allocation_ref': deepcopy(allocation_ref), 'allocation_id': allocation['allocation_id'],
        'ledger_scope': 'this_feedback_occurrence_only_prior_paid_accounts_unchanged',
        'new_fits': 0, 'new_predictions': 0, 'new_observations': 0, 'model_calls': 0,
        'profiles_unchanged': True, 'frozen_results_preserved': True}
    save(output / 'receipt.json', result)
    return result


def refresh_pair_profiles(prior_review, output, pair_refs, *, allocation_ref=None):
    """Append corrected pre-use evidence to the same paid method invocations."""
    prior_review, output = Path(prior_review), Path(output)
    basis = {'prior_review': str(prior_review), 'pair_refs': deepcopy(pair_refs),
             'allocation_ref': deepcopy(allocation_ref)}
    if (output / 'receipt.json').exists():
        previous = load(output / 'receipt.json')
        if previous.get('basis', basis) != basis:
            raise ValueError('Retain the completed profile refresh and its original inputs')
        return {**previous, 'new_compute_cost': 0, 'new_predictions': 0, 'reused': True}
    ledger = load(prior_review / 'receipt.json')
    owner = next(q for q in load(prior_review / 'common/questions.json') if q['id'] == 'task:method_choice')
    allocation = read_method_reference(allocation_ref, prior_review) if allocation_ref else None
    if allocation and allocation['new_observation_limit'] != 0:
        raise ValueError('Profile diagnostics use the existing observations only')
    limit = allocation['maximum_compute'] if allocation else min(100, ledger['remaining_compute'])
    review_minimum = 1 + len(owner['method_use_plan']['native_return_refs']) + len(pair_refs)
    if not pair_refs or limit < review_minimum:
        raise ValueError('The current balance must fund the retained-pair review')
    output.mkdir(parents=True, exist_ok=True)
    retained = deepcopy(owner)
    for key in ('routing_execution', 'method_use_result', 'completed_agenda_basis'):
        retained.pop(key, None)
    mission = ({'purpose': 'method_learning',
        'mission_id': allocation['allocation_id'] + ':' + str(output),
        'allocation': {'compute': limit}, 'resource_usage': {'compute': 0}}
        if allocation else deepcopy(owner['mission']))
    if not allocation:
        mission['allocation']['compute'] += limit
    retained['mission'] = mission
    save(output / 'previous_topic.json', {'question': retained, 'questions': [retained], 'methods': []})
    readout_budget = SearchBudget(limit - review_minimum)
    updated, returns, evidence_results = [], [], []
    for index, pair_ref in enumerate(pair_refs):
        pair = read_method_reference(pair_ref, prior_review)
        source = Path(pair['origin_ref']['file']).parent
        cfg = load(source / 'registration.json')
        directory = output / f'pair_{index:03d}'
        compared = load(source / 'common_comparison.json')
        if allocation:
            # A later condition revision keeps the original physical origin,
            # while its paid diagnostic column lives with the preceding view.
            for side in ('baseline', 'challenger'):
                outcome = read_method_reference(pair[side + '_ref'], prior_review)
                profile = read_method_reference(outcome['profile_ref'], prior_review)
                evidence_ref = profile.get('recovery_basis', {}).get('pre_call_evidence_ref')
                if evidence_ref:
                    compared = {**compared, 'pre_call_evidence_ref': evidence_ref}
                    break
            evidence = ensure_pre_call_evidence(source, cfg, compared, directory / 'pre_call', readout_budget)
            evidence_results.append(evidence)
            if evidence.get('budget_incomplete') or not evidence.get('evidence_ref'):
                result = {'status': 'complete_not_funded', 'basis': basis,
                    'original_pair_refs': deepcopy(pair_refs), 'updated_pair_refs': updated,
                    'pre_call_evidence': evidence_results,
                    'compute_cost': readout_budget.spent, 'new_compute_cost': readout_budget.spent,
                    'remaining_compute': limit - readout_budget.spent,
                    'allocation_ref': deepcopy(allocation_ref), 'allocation_id': allocation['allocation_id'],
                    'new_predictions': sum(x['new_predictions'] for x in evidence_results),
                    'new_fits': 0, 'new_observations': 0, 'model_calls': 0, 'new_invocations': 0,
                    'frozen_result_preserved': True, 'budget_incomplete': True}
                save(output / 'receipt.json', result)
                return result
            compared = {**compared, 'pre_call_evidence_ref': evidence['evidence_ref']}
        profile = paid_pre_call_profile(source, cfg, compared)
        profile_ref = ref(directory / 'profile.json')
        save(Path(profile_ref['file']), profile)
        revised = deepcopy(pair)
        for side in ('baseline', 'challenger'):
            outcome = read_method_reference(pair[side + '_ref'], prior_review)
            revised[side + '_ref'] = ref(directory / (side + '_outcome.json'))
            save(Path(revised[side + '_ref']['file']), {**outcome, 'profile_ref': profile_ref})
        revised['profile_recovery_predecessor_ref'] = deepcopy(pair_ref)
        saved_pair = ref(directory / 'pair.json')
        save(Path(saved_pair['file']), revised)
        challenger = read_method_reference(pair['challenger_ref'], prior_review)
        returned = {'status': 'paid_pre_call_profile_recovered', 'method_pair_ref': saved_pair,
            'method_ref': challenger['method_ref'], 'method_artifact_ref': challenger['method_artifact_ref'],
            'source_task_ref': pair['physical_source_ref'], 'original_pair_ref': deepcopy(pair_ref)}
        returned_ref = ref(directory / 'research_receipt.json')
        save(Path(returned_ref['file']), returned)
        returns.append((pair, returned_ref, source))
        updated.append(saved_pair)
    common_limit = limit - readout_budget.spent
    common = CommonObjectDependencyMethods(output / 'common', [], budget=SearchBudget(common_limit), actions=[]).common
    ensure_python_method_topic(common, parent_question_ref={'question_id': 'task'},
        topic_ref=ref(output / 'previous_topic.json'))
    for pair, returned_ref, source in returns:
        feedback = record_native_parent_research_return(common,
            source_task_ref=pair['physical_source_ref'], result_ref=returned_ref,
            before_ref=pair.get('decision_ref'), cost_refs=[ref(source / 'receipt.json')], mission=mission)
    reviewed = consume_method_return_agenda(common, feedback, None, compute_limit=common_limit)
    cost = readout_budget.spent + common.costs['compute_cost']
    result = {'status': reviewed['status'], 'basis': basis, 'return': reviewed, 'updated_pair_refs': updated,
        'original_pair_refs': deepcopy(pair_refs), 'compute_cost': cost, 'new_compute_cost': cost,
        'cumulative_compute': cost if allocation else ledger['cumulative_compute'] + cost,
        'remaining_compute': limit - cost if allocation else ledger['remaining_compute'] - cost,
        'source_cost_refs': [ref(prior_review / 'receipt.json')],
        'pre_call_evidence': evidence_results, 'readout_compute_cost': readout_budget.spent,
        'common_compute_cost': common.costs['compute_cost'],
        'new_fits': 0, 'new_predictions': sum(x['new_predictions'] for x in evidence_results),
        'new_observations': 0, 'model_calls': 0,
        'new_invocations': 0, 'frozen_result_preserved': True}
    if allocation:
        result.update(allocation_ref=deepcopy(allocation_ref), allocation_id=allocation['allocation_id'],
            ledger_scope='this_paid_profile_diagnostic_occurrence_only_original_method_costs_unchanged')
    save(output / 'receipt.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--prior-review', required=True, type=Path)
    parser.add_argument('--retained-pair', type=Path)
    args = parser.parse_args()
    result = return_pair(args.source, args.output, args.prior_review,
        ref(args.retained_pair) if args.retained_pair else None)
    print({key: value for key, value in result.items() if key != 'return'})
