#!/usr/bin/env python3
"""Consume paid stage returns through the ordinary staged-method replayer."""

import argparse
from copy import deepcopy
from pathlib import Path
from math import isclose

from scripts.run_native_decision_pair import ref
from scripts.run_partial_role_parent import load, save
from scripts.validate_partition_successor import freeze_entry, retained_entry, task_settings
from hegel_machine.cognition.checkpoint_storage import write_text_incrementally
from hegel_machine.cognition.experiment_runtime import ExecutionTiming, python_execution_deadline
from hegel_machine.cognition.method_action_research import read_method_reference
from hegel_machine.cognition.models import ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.object_continuation import decode_entry, encode_entry, observations
from hegel_machine.cognition.research_programs import (
    replay_research_program, research_program_replay_inputs)
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


ROOT = Path(__file__).resolve().parents[1]


def _rows(locations, root):
    result = []
    for location in locations:
        value = read_method_reference(location, root)
        result.extend(observations(value.get('batch', value)['observations']))
    return result


def paid_scene(source):
    """Recover exact fitted outputs, not the possibly different full-pool winner."""
    source = Path(source)
    cfg, state = load(source / 'registration.json'), load(source / 'prepared_state.json')
    common, methods = load(source / 'common_comparison.json'), load(source / 'frozen_methods.json')
    if not common['full_comparison']:
        raise ValueError('The retained common pool must be fully scored')
    training = _rows(state['training_refs'], source)
    working = _rows(state['validation_refs'], source)
    if common['rows'] != len(training) + len(working):
        raise ValueError('Paid scores must cover exactly the retained training and working split')
    pool = [dict(retained_entry(x['candidate_ref'], source, i), validation_loss=x['loss'])
            for i, x in enumerate(common['scores'])]
    staged_arms = [arm for arm, body in methods.items() if body.get('staged_research')]
    stage_report = None
    if staged_arms:
        if staged_arms != ['Pi1']:
            raise ValueError('The paid staged contrast must retain its actual fixed and staged arms')
        contract = methods['Pi1']['staged_research']
        if contract['continuation'] != methods['P0']:
            raise ValueError('The fixed paid method must be the staged continuation itself')
        stage_report = load(source / 'Pi1/outcome.json').get('staged_research_result', {})
        if stage_report.get('selected_stage') != 'baseline':
            raise ValueError('The separately saved baseline stage is required for this paid contrast')
        methods = {'P0': contract['baseline'], 'Pi1': contract['continuation']}
        arms = (('baseline', 'Pi1'), ('continuation', 'P0'))
    else:
        arms = (('baseline', 'P0'), ('continuation', 'Pi1'))
    stages, paid_scores, evidence = {}, {}, {}
    for label, arm in arms:
        outcome = load(source / arm / 'outcome.json')
        comparison = load(source / arm / 'comparison.json')
        location = ref(source / arm / 'new_entry.json')
        entry = decode_entry(read_method_reference(location, source))
        matched = next((x for x in comparison['new_scores'] if x['candidate_ref'] == location), None)
        if not (outcome['execution_complete'] and comparison['full_comparison'] and matched):
            raise ValueError('A complete actual stage and its own paid score are required')
        # These two disjoint saved scores cover the declared same 96/24 split.
        # This is SSE accounting, not another evaluation of the predictor.
        feedback_loss = ((len(training) + len(working)) * matched['loss']
                         - len(training) * entry['training_loss']) / len(working)
        if stage_report and label == 'baseline':
            baseline = stage_report.get('baseline', {})
            if (not baseline.get('execution_complete')
                    or baseline.get('program') != entry['candidate'].program.to_dict()
                    or not isclose(baseline.get('all_paid_loss', float('nan')), matched['loss'],
                                   rel_tol=1e-12, abs_tol=0.)):
                raise ValueError('The saved actual prefix must match its own candidate and paid score')
            feedback_loss = baseline['feedback_loss']
        if feedback_loss < 0:
            # Roundoff at an already essentially exact fit is not negative MSE.
            if abs(feedback_loss) > 1e-12:
                raise ValueError('The saved split scores disagree')
            feedback_loss = 0.
        followup = entry.get('transfer_partition_followup') or {}
        if followup.get('feedback_rows') == len(working) and 'feedback_loss' in followup:
            feedback_loss = followup['feedback_loss']
        stages[label] = {'replay_inputs': research_program_replay_inputs(methods['P0' if label == 'baseline' else 'Pi1']),
            'training': training, 'feedback': working, 'risk_quantile': .5,
            'tolerance': cfg['quality_threshold'], 'entry': entry,
            'execution_complete': True, 'feedback_loss': feedback_loss,
            'cost_ref': ref(source / arm / 'outcome.json'),
            'feedback_loss_refs': [ref(source / arm / 'new_entry.json', 'training_loss'),
                ref(source / arm / 'comparison.json', 'new_scores')]}
        paid_scores[entry['candidate'].program] = matched['loss']
        evidence[label] = {'entry_ref': location, 'training_score_ref': ref(source / arm / 'new_entry.json',
                'training_loss'), 'joint_score_ref': ref(source / arm / 'comparison.json', 'new_scores'),
            'feedback_loss': feedback_loss, 'joint_loss': matched['loss'],
            'score_basis': 'actual_stage_training_and_joint_saved_scores',
            'outcome_ref': ref(source / arm / 'outcome.json')}
        if stage_report:
            # These are now paid knowledge as well as method counterfactuals.
            # Keep both outputs, including a losing prefix, in the current pool.
            # The method's own score remains separately visible in its result.
            pool.append(dict(entry, validation_loss=matched['loss']))
    spec = TaskSpec(**read_method_reference(state['public_spec_ref'], source))
    frame = {'spec': spec, 'training': training, 'comparison': [*training, *working],
        'independent_validation': False, 'assessed': pool, 'fallback': state['fallback'],
        'comparison_complete': True, 'uncompared': []}
    return {'cfg': cfg, 'methods': methods, 'training': training, 'working': working,
        'frame': frame, 'paid_stages': stages, 'paid_scores': paid_scores,
        'stage_evidence': evidence, 'state': state, 'common': common}


def parent_bound_paid_scene(source, method, proposal_ref):
    """Recover a real local-parent -> bound-Pi1 execution, without reevaluation."""
    from scripts.review_paid_object_pool import parent_inputs
    from hegel_machine.cognition.transfer_partitions import prepare_parent_parameter_revision

    source = Path(source)
    proposal = read_method_reference(proposal_ref, source)
    basis, staged = proposal['basis'], method['staged_research']
    if (basis.get('operation') != 'revise_then_reidentify'
            or not any(item['research_program'] == method for item in proposal['proposal']['candidates'])
            or staged['baseline'] != proposal['proposal']['baseline_research_program']
            or (source / basis['actual_result_ref']['file']).resolve().parent != source.resolve()):
        raise ValueError('Recover this exact generated composition and its actual sequential source')
    actual = read_method_reference(basis['actual_result_ref'], source)
    action = read_method_reference(basis['method_execution_ref'], source)
    native = action.get('research') or {}
    if (not actual.get('execution_complete') or not actual.get('full_comparison')
            or not action.get('full_comparison') or not native.get('execution_complete')
            or actual.get('method_execution_ref') != basis['method_execution_ref']
            or native.get('parent_ref') != basis['research_parent_ref']
            or native.get('source_method_ref') != basis['source_method_ref']):
        raise ValueError('The actual parent-dependent research must be complete and compared')
    original_method = read_method_reference(basis['source_method_ref'], source)
    if staged['continuation'] != original_method:
        raise ValueError('The composition must retain the actual source Pi1 operation')
    producer_ref = proposal['baseline_execution_ref']
    producer = read_method_reference(producer_ref, source)
    provenance = read_method_reference(proposal['baseline_input_provenance_ref'], source)
    inputs = []
    for side in ('baseline', 'challenger'):
        returned = read_method_reference(provenance[side + '_ref'], source)
        if returned.get('actual_outcome_ref') == producer_ref:
            artifact = read_method_reference(returned['method_artifact_ref'], source)
            inputs.append(read_method_reference(artifact['execution']['research_program_ref'], source))
    if (not producer.get('execution_complete') or not producer.get('full_comparison')
            or staged['baseline'] not in inputs):
        raise ValueError('The first stage must retain its actual completed invocation input')
    parent = retained_entry(basis['research_parent_ref'], source, 0)
    compared_ref = producer.get('comparison_ref') or {
        'file': str((source / producer_ref['file']).with_name('comparison.json'))}
    compared = read_method_reference(compared_ref, source)
    produced = [retained_entry(item['candidate_ref'], source, i) for i, item in
                enumerate(compared.get('new_scores', []))]
    if (not compared.get('full_comparison') or not any(
            entry['candidate'].program == parent['candidate'].program for entry in produced)):
        raise ValueError('The first stage must have actually produced this parent, not just retained another pool model')
    prepared = prepare_parent_parameter_revision(parent, original_method)
    executed = read_method_reference(actual['executed_method_ref'], source)
    if (prepared['status'] != 'ready' or research_program_replay_inputs(prepared['research_program'])
            != research_program_replay_inputs(executed)):
        raise ValueError('The paid continuation must execute Pi1 bound to this actual first-stage parent')
    comparison_ref = actual['comparison_ref']
    comparison = read_method_reference(comparison_ref, source)
    if not comparison.get('full_comparison') or len(comparison.get('new_scores', [])) != 1:
        raise ValueError('Recover one actual sequential output; do not substitute the pool winner or parallel Pi1')
    output_ref = comparison['new_scores'][0]['candidate_ref']
    continuation = retained_entry(output_ref, source, 1)
    frontier, original, rows, _, references, pool, _, scores = parent_inputs(source)
    cfg = load(source / 'registration.json')
    training, working = rows[:cfg['training_rows']], rows[cfg['training_rows']:]
    if not working or comparison['rows'] != len(rows) or producer['development_rows'] != len(rows):
        raise ValueError('Both actual stages must share the complete retained training/feedback view')
    spec = TaskSpec(**read_method_reference(cfg['public_spec_ref'], source))
    producer_root = (source / producer_ref['file']).parent.parent
    state_ref = ref(producer_root / 'prepared_state.json')
    state = read_method_reference(state_ref, source)
    stages, evidence = {}, {}
    for label, entry, body, location, cost_ref in (
            ('baseline', parent, staged['baseline'], basis['research_parent_ref'], producer_ref),
            ('continuation', continuation, prepared['research_program'], output_ref, basis['actual_result_ref'])):
        if entry['research_program']['fit_rows'] != len(training):
            raise ValueError('The paid fitted columns belong to another training view')
        joint = scores[entry['candidate'].program]
        feedback_loss = (len(rows) * joint - len(training) * entry['training_loss']) / len(working)
        if feedback_loss < -1e-12:
            raise ValueError('Saved joint and training scores disagree')
        feedback_loss = max(0., feedback_loss)
        followup = entry.get('transfer_partition_followup') or {}
        if followup.get('feedback_rows') == len(working) and followup.get('feedback_loss') is not None:
            feedback_loss = followup['feedback_loss']
        stages[label] = {'replay_inputs': research_program_replay_inputs(body),
            'training': training, 'feedback': working, 'risk_quantile': spec.risk_quantile,
            'tolerance': cfg['quality_threshold'], 'entry': entry, 'execution_complete': True,
            'feedback_loss': feedback_loss, 'cost_ref': cost_ref,
            'feedback_loss_refs': [location, comparison_ref if label == 'continuation' else compared_ref]}
        evidence[label] = {'entry_ref': location, 'outcome_ref': cost_ref,
            'joint_loss': joint, 'feedback_loss': feedback_loss,
            'score_basis': 'actual_sequential_output_saved_training_and_full_pool_scores'}
    frame = {'spec': spec, 'training': training, 'comparison': rows, 'independent_validation': False,
        'assessed': pool, 'fallback': state['fallback'], 'comparison_complete': True, 'uncompared': []}
    return {'cfg': {**cfg, 'research_config': {'max_candidates': 4,
            'minimum_improvement': cfg['quality_threshold']}},
        'methods': {'P0': staged['baseline'], 'Pi1': staged['continuation']},
        'training': training, 'working': working, 'frame': frame,
        'paid_stages': stages, 'paid_scores': scores, 'stage_evidence': evidence,
        'state': state, 'prepared_state_ref': state_ref,
        'pool_ref': ref(source / 'retained_frontier.json')}


def parent_suffix_scene(source, method):
    """Restore a completed structural prefix and all then-available paid models."""
    from hegel_machine.cognition.transfer_partitions import prepare_parent_parameter_revision

    source = Path(source)
    cfg, state = load(source / 'registration.json'), load(source / 'prepared_state.json')
    methods, common = load(source / 'frozen_methods.json'), load(source / 'common_comparison.json')
    staged = method['staged_research']
    if (staged['operation'] != 'revise_then_reidentify' or methods.get('Pi3') != staged['baseline']
            or cfg.get('feature_mappings') != [{}] or not common['full_comparison']):
        raise ValueError('Reuse the exact completed Pi3 body on its original unremapped paid view')
    training, working = _rows(state['training_refs'], source), _rows(state['validation_refs'], source)
    rows = [*training, *working]
    if not working or common['rows'] != len(rows) or cfg['training_rows'] != len(training):
        raise ValueError('Retain the original complete training and working split')
    scores = deepcopy(common['scores'])
    for arm in ('Pi1', 'Pi3'):
        compared = load(source / arm / 'comparison.json')
        if (not compared['full_comparison']
                or load(source / arm / 'outcome.json')['development_rows'] != len(rows)):
            raise ValueError('Retain both paid arm outputs with their complete same-data scores')
        scores.extend(deepcopy(item) for item in compared['new_scores']
                      if not any(old['candidate_ref'] == item['candidate_ref'] for old in scores))
    pool = [dict(retained_entry(item['candidate_ref'], source, i), validation_loss=item['loss'])
            for i, item in enumerate(scores)]
    producer = load(source / 'Pi3/outcome.json')
    actual = load(source / 'Pi3/comparison.json')['new_scores']
    if (not producer['execution_complete'] or not producer['full_comparison'] or len(actual) != 1
            or producer['development_rows'] != len(rows)):
        raise ValueError('The structural prefix must have delivered one complete actual result')
    parent_ref, joint = actual[0]['candidate_ref'], actual[0]['loss']
    parent = retained_entry(parent_ref, source, len(pool))
    if (not parent.get('transfer_partition_search', {}).get('completed')
            or parent['research_program']['fit_rows'] != len(training)):
        raise ValueError('The retained prefix must contain its completed exact fitted columns')
    feedback = (len(rows) * joint - len(training) * parent['training_loss']) / len(working)
    if feedback < -1e-12:
        raise ValueError('The saved prefix split scores disagree')
    spec = TaskSpec(**read_method_reference(state['public_spec_ref'], source))
    if min(item['validation_loss'] for item in pool) <= cfg['quality_threshold']:
        raise ValueError('This complete paid pool already solves the source; do not repeat its research')
    prepared = prepare_parent_parameter_revision(parent, staged['continuation'])
    if prepared['status'] != 'ready':
        raise ValueError('The actual structural prefix has no complete Pi1 binding: ' + prepared['status'])
    paid = {'replay_inputs': research_program_replay_inputs(staged['baseline']),
        'training': training, 'feedback': working, 'risk_quantile': spec.risk_quantile,
        'tolerance': cfg['quality_threshold'], 'entry': parent, 'execution_complete': True,
        'feedback_loss': max(0., feedback), 'cost_ref': ref(source / 'Pi3/outcome.json'),
        'feedback_loss_refs': [parent_ref, ref(source / 'Pi3/comparison.json')]}
    frame = {'spec': spec, 'training': training, 'comparison': rows, 'independent_validation': False,
        'assessed': pool, 'fallback': state['fallback'], 'comparison_complete': True, 'uncompared': []}
    return {'cfg': cfg, 'state': state, 'training': training, 'working': working, 'frame': frame,
        'paid_stages': {'baseline': paid}, 'paid_scores': {x['candidate'].program: x['validation_loss'] for x in pool},
        'bound_continuation': prepared['research_program'],
        'stage_evidence': {'baseline': {'entry_ref': parent_ref, 'joint_loss': joint,
            'outcome_ref': ref(source / 'Pi3/outcome.json')}},
        'common': {**common, 'scores': scores, 'data_refs': [*state['training_refs'], *state['validation_refs']],
            'source_pool_size': len(scores), 'pre_call_evidence_ref': None},
        'pool_ref': ref(source / 'common_comparison.json')}


def prepare_parent_suffix(sources, method_ref, allocation_ref, output, *, artifact_refs=None):
    """Freeze only the unexecuted suffix at the declared paid post-prefix choices."""
    from hegel_machine.cognition.transfer_partitions import quote_transfer_partition

    output = Path(output)
    method, allocation = read_method_reference(method_ref, output), read_method_reference(allocation_ref, output)
    if allocation['new_observation_limit'] != 0 or not 0 < allocation['maximum_compute'] <= 150000:
        raise ValueError('Use the declared at-most-150000 offline suffix allocation')
    if artifact_refs:
        for arm, expected in (('P0', method['staged_research']['baseline']), ('Pi1', method)):
            artifact = read_method_reference(artifact_refs[arm], output)
            if read_method_reference(artifact['execution']['research_program_ref'], output) != expected:
                raise ValueError('The canonical artifact must retain this exact frozen stage body')
    records = []
    for source in map(Path, sources):
        scene = parent_suffix_scene(source, method)
        quote = quote_transfer_partition(scene['bound_continuation'], scene['training'],
            feedback_rows=scene['working'])
        if quote.get('required_compute') is None:
            raise ValueError('The exact parent-bound suffix is not quotable')
        guard = len(method['staged_research'].get('conditions', [method['staged_research']['condition']]))
        method_upper = quote['required_compute'] + len(scene['working']) + guard
        comparison = 4 * len(scene['frame']['comparison'])
        records.append({'source': str(source), 'source_task_ref': scene['cfg']['source_task_ref'],
            'prepared_state_ref': ref(source / 'prepared_state.json'),
            'pool_ref': scene['pool_ref'], 'pool_size': len(scene['frame']['assessed']),
            'training_rows': len(scene['training']), 'working_rows': len(scene['working']),
            'stage_evidence': scene['stage_evidence'], 'suffix_quote': quote,
            'method_compute_limit': method_upper, 'comparison_reserve': comparison,
            'method_return_compute_limit': 3000, 'maximum_compute': method_upper + comparison,
            'prior_cost_ref': ref(source / 'receipt.json')})
    required = sum(item['maximum_compute'] + item['method_return_compute_limit'] for item in records)
    if not records or len({str(item['source_task_ref']) for item in records}) != len(records):
        raise ValueError('Use distinct declared paid physical sources')
    if required > allocation['maximum_compute']:
        raise ValueError(f'The complete frozen suffixes and returns require {required} compute')
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'frozen_method.json', method)
    artifacts = {}
    for arm, location in (('P0', ref(output / 'frozen_method.json', 'staged_research', 'baseline')),
                          ('Pi1', method_ref)):
        if artifact_refs:
            artifacts[arm] = deepcopy(artifact_refs[arm])
            continue
        artifact = {'id': 'native:staged:' + str(location), 'version': 1, 'kind': 'native_research_segment',
            'execution': {'entry': 'replay_research_program',
                'selection': 'UnifiedResearchAgent.compare_research_decision_return', 'research_program_ref': location},
            'source_ref': location}
        save(output / (arm + '_artifact.json'), artifact)
        artifacts[arm] = ref(output / (arm + '_artifact.json'))
    cfg = {'status': 'registered_paid_parent_suffix', 'source_mode': 'paid_prefix_new_suffix',
        'method_source_ref': method_ref, 'method_ref': ref(output / 'frozen_method.json'),
        'native_method_artifact_refs': artifacts, 'allocation_ref': allocation_ref,
        'allocation_id': allocation['allocation_id'], 'maximum_compute': allocation['maximum_compute'],
        'execution_compute_limit': sum(item['maximum_compute'] for item in records),
        'generation_reserve': 0, 'sources': records, 'required_compute': required,
        'active_wall_seconds': 1200, 'new_observation_limit': 0,
        'evaluation_role': 'exposed_paid_post_prefix_development', 'independent_support': False,
        'stop_rule': 'one frozen continuation per declared paid prefix; no retries or new observations',
        'cost_rule': 'original prefix costs remain referenced; new suffix, comparison and return are charged separately'}
    save(output / 'registration.json', cfg)
    freeze_entry(output, Path(sources[0]))
    return cfg


def _save_suffix_pair(directory, cfg, registered, scene, method, result, compared, elapsed):
    """Publish the existing ordinary-return schema for this paid next-stage choice."""
    before = UnifiedResearchAgent(ResearchConfig(**scene['cfg']['research_config']))._best(scene['frame']['assessed'])
    save(directory / 'P0/selected_entry.json', encode_entry(before))
    base = {'execution_complete': True, 'full_comparison': True,
        'selected_ref': ref(directory / 'P0/selected_entry.json'), 'selected_loss': before['validation_loss'],
        'source_pool_size': len(scene['frame']['assessed']), 'new_candidates': 0,
        'compute_cost': 0, 'method_compute_cost': 0, 'comparison_compute_cost': 0,
        'development_rows': len(scene['frame']['comparison']), 'independent_support': False,
        'shared_paid_prefix_ref': registered['stage_evidence']['baseline']['outcome_ref']}
    after = {**result, 'new_candidates': len(compared['new_claims']),
        'method_compute_cost': result['execution_compute_cost'],
        'source_pool_size': len(scene['frame']['assessed']), 'development_rows': base['development_rows'],
        'same_available_method_cap': registered['method_compute_limit']}
    save(directory / 'P0/outcome.json', base)
    save(directory / 'Pi1/outcome.json', after)
    save(directory / 'P0/comparison.json', {'full_comparison': True, 'new_scores': [],
        'old_comparison_ref': ref(directory / 'common_comparison.json'), 'selected_ref': base['selected_ref']})
    save(directory / 'Pi1/comparison.json', {'full_comparison': compared['full_comparison'],
        'old_comparison_ref': ref(directory / 'common_comparison.json'), 'new_scores': [
            {'candidate_ref': result['new_entry_ref'], 'loss': entry['validation_loss']}
            for entry in compared['new_claims']], 'selected_ref': result['selected_ref']})
    save(directory / 'common_comparison.json', scene['common'])
    save(directory / 'prepared_state.json', scene['state'])
    save(directory / 'frozen_methods.json', {'P0': method['staged_research']['baseline'], 'Pi1': method})
    source_cfg = {key: value for key, value in scene['cfg'].items() if key not in {
        'condition_validation_cohort_refs', 'frozen_conditions_ref', 'condition_prediction_ref',
        'condition_development_ref', 'condition_evaluation_scope', 'confirmation_seed', 'confirmation_rows'}}
    save(directory / 'registration.json', {**source_cfg, 'data_role': 'development',
        'evaluation_role': 'paid_working_comparison', 'operations': {'P0': 'program_replay', 'Pi1': 'program_replay'},
        'evaluation_arm_names': {'baseline': 'P0', 'challenger': 'Pi1'},
        'native_method_artifact_refs': cfg['native_method_artifact_refs'],
        'equal_method_compute_limit': registered['method_compute_limit'],
        'per_arm_comparison_reserve': registered['comparison_reserve'],
        'method_return_compute_limit': registered['method_return_compute_limit'],
        'active_wall_seconds': cfg['active_wall_seconds'], 'working_rows': len(scene['working']),
        'allocation_ref': cfg['allocation_ref'], 'source_cost_refs': [registered['prior_cost_ref']],
        'scope': 'retain completed paid prefix versus append frozen parent-bound continuation',
        'original_ledger': registered['prior_cost_ref']})
    save(directory / 'receipt.json', {**result, 'status': 'paid_methods_completed'
        if result['execution_complete'] else 'incomplete_method_not_negative',
        'arms': {'P0': base, 'Pi1': after}, 'cumulative_compute': result['compute_cost'],
        'remaining_compute': registered['maximum_compute'] + registered['method_return_compute_limit'] - result['compute_cost'],
        'source_cost_refs': [registered['prior_cost_ref']],
        'execution_timing': {'phase_seconds': {'P0': 0., 'Pi1': elapsed}}})


def prepare_paid(sources, method_ref, allocation_ref, output, *, proposal_ref=None):
    """Register two exposed development calls without touching their old ledgers."""
    output = Path(output)
    method = read_method_reference(method_ref, output)
    staged = method.get('staged_research') or {}
    if staged.get('operation') not in {'refit_then_reidentify', 'revise_then_reidentify'}:
        raise ValueError('Freeze an actual generated staged research method')
    if staged['operation'] == 'revise_then_reidentify' and proposal_ref is None:
        raise ValueError('The actual generated parent-dependent proposal is required')
    allocation = read_method_reference(allocation_ref, output)
    if allocation['new_observation_limit'] != 0:
        raise ValueError('Paid development cannot acquire observations')
    maximum = allocation['maximum_compute']
    generation = allocation.get('formation_reserve', 10000)
    per_source = (maximum - generation) // len(sources)
    if maximum > 60000 or per_source <= 0:
        raise ValueError('Use the registered at-most-60000 paid development allocation')
    records = []
    for source in map(Path, sources):
        scene = (parent_bound_paid_scene(source, method, proposal_ref) if
            staged['operation'] == 'revise_then_reidentify' else paid_scene(source))
        if any(staged[side] != scene['methods'][arm]
               for side, arm in (('baseline', 'P0'), ('continuation', 'Pi1'))):
            raise ValueError('The generated method must retain the actual same frozen stages')
        records.append({'source': str(source), 'source_task_ref': scene['cfg']['source_task_ref'],
            'prepared_state_ref': scene.get('prepared_state_ref', ref(source / 'prepared_state.json')),
            'pool_ref': scene.get('pool_ref', ref(source / 'common_comparison.json')),
            'pool_size': len(scene['frame']['assessed']), 'training_rows': len(scene['training']),
            'working_rows': len(scene['working']), 'stage_evidence': scene['stage_evidence'],
            'maximum_compute': per_source, 'prior_cost_ref': ref(source / 'receipt.json')})
    if len({str(x['source_task_ref']) for x in records}) != len(records):
        raise ValueError('Keep the distinct declared physical sources')
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'frozen_method.json', method)
    cfg = {'status': 'registered_paid_development', 'method_source_ref': method_ref,
        'method_ref': ref(output / 'frozen_method.json'), 'allocation_ref': allocation_ref,
        'allocation_id': allocation['allocation_id'], 'maximum_compute': maximum,
        'generation_reserve': generation, 'execution_compute_limit': maximum - generation,
        'sources': records, 'active_wall_seconds': 1200, 'new_observation_limit': 0,
        'evaluation_role': 'exposed_paid_stage_path_development', 'independent_support': False,
        'stop_rule': 'one ordinary staged invocation per source; preserve incomplete outcomes; no retries',
        'cost_rule': 'count new guard/comparison/execution only; old stage cost references are not new charges'}
    if proposal_ref is not None:
        cfg['proposal_ref'] = deepcopy(proposal_ref)
    save(output / 'registration.json', cfg)
    freeze_entry(output, Path(sources[0]))
    relative = Path('scripts/stage_partition_method.py')
    write_text_incrementally(output / 'runtime' / relative, (ROOT / relative).read_text())
    return cfg


def run_paid(output):
    output = Path(output)
    if (output / 'receipt.json').exists():
        return {**load(output / 'receipt.json'), 'new_compute_cost': 0, 'reused': True}
    if ROOT != (output / 'runtime').resolve() or (output / 'started.json').exists():
        raise ValueError('Use the freshly frozen one-time development entry')
    cfg = load(output / 'registration.json')
    method = read_method_reference(cfg['method_ref'], output)
    total, results = SearchBudget(cfg['execution_compute_limit']), []
    timing = ExecutionTiming(tuple(str(i) for i in range(len(cfg['sources']))))
    save(output / 'started.json', {'registration_ref': ref(output / 'registration.json')})
    try:
        for i, registered in enumerate(cfg['sources']):
            suffix = cfg.get('source_mode') == 'paid_prefix_new_suffix'
            scene = (parent_suffix_scene(registered['source'], method) if suffix else
                parent_bound_paid_scene(registered['source'], method, cfg['proposal_ref'])
                if method['staged_research']['operation'] == 'revise_then_reidentify'
                else paid_scene(registered['source']))
            budget = SearchBudget(registered['maximum_compute'])
            directory = output / f'source_{i:03d}'
            try:
                with timing.phase(str(i)), python_execution_deadline(cfg['active_wall_seconds'], activity=True):
                    execution_budget = SearchBudget(registered['method_compute_limit']) if suffix else budget
                    try:
                        entry = replay_research_program(method, scene['training'], execution_budget,
                            feedback_rows=scene['working'], tolerance=scene['cfg']['quality_threshold'],
                            risk_quantile=scene['frame']['spec'].risk_quantile,
                            paid_stages=scene['paid_stages'])
                    finally:
                        if suffix and not budget.spend(execution_budget.spent):
                            raise RuntimeError('The suffix exceeded its registered allocation')
                    execution_cost = budget.spent
                    if entry is not None:
                        save(directory / 'new_entry.json', encode_entry(entry))
                    staged = (entry or {}).get('staged_research_result', {})
                    agent = UnifiedResearchAgent(ResearchConfig(**scene['cfg']['research_config']))
                    actions = []
                    compared = agent.compare_research_decision_return(scene['frame'], 'program_replay',
                        [entry] if entry is not None else [], budget, actions,
                        paid_scores=scene['paid_scores'], search_compute_cost=execution_cost)
                    save(directory / 'selected_entry.json', encode_entry(compared['selected']))
                    result = {'source_task_ref': registered['source_task_ref'],
                        'execution_complete': bool(staged.get('execution_complete') and compared['full_comparison']),
                        'staged_research_result': staged, 'full_comparison': compared['full_comparison'],
                        'old_pool_size': len(scene['frame']['assessed']),
                        'selected_loss': compared['selected']['validation_loss'],
                        'selected_ref': ref(directory / 'selected_entry.json'),
                        'new_entry_ref': ref(directory / 'new_entry.json') if entry else None,
                        'compute_cost': budget.spent, 'execution_compute_cost': execution_cost,
                        'comparison_compute_cost': budget.spent - execution_cost,
                        'budget_denied': bool(budget.denied_requests or execution_budget.denied_requests), 'actions': actions,
                        'paid_stage_refs': registered['stage_evidence'],
                        'new_observations': 0, 'independent_support': False}
                    if not suffix:
                        save(directory / 'receipt.json', result)
                    results.append(result)
                if suffix:
                    _save_suffix_pair(directory, cfg, registered, scene, method, result, compared,
                        timing.result()['phase_seconds'][str(i)])
            finally:
                if not total.spend(budget.spent):
                    raise RuntimeError('Paid development exceeded its registered allocation')
    except BaseException as error:
        save(output / 'failure.json', {'error': repr(error), 'compute_cost': total.spent,
            'completed_sources': results, 'allocation_ref': cfg['allocation_ref'],
            'new_observations': 0, 'execution_timing': timing.result()})
        raise
    result = {'status': ('paid_stage_paths_completed' if all(x['execution_complete'] for x in results)
        else 'paid_stage_path_incomplete_not_negative'), 'sources': results,
        'new_compute_cost': total.spent, 'compute_cost': total.spent,
        'remaining_execution_compute': total.remaining, 'allocation_ref': cfg['allocation_ref'],
        'allocation_id': cfg['allocation_id'], 'generation_cost_scope': 'separate actual receipt, not added twice',
        'new_observations': 0, 'model_calls': 0, 'independent_support': False,
        'old_ledgers_unchanged': True, 'execution_timing': timing.result()}
    save(output / 'receipt.json', result)
    return result


def successor_pool(developed):
    """Retain the common original pool and every exposed source-stage output."""
    sources = [Path(item['source']) for item in developed['sources']]
    original = load(sources[0] / 'common_comparison.json')
    if not original['full_comparison']:
        raise ValueError('The complete original source pool must be retained')
    scores = [{'candidate_ref': deepcopy(item['candidate_ref']), 'loss': None,
        'source_score_ref': ref(sources[0] / 'common_comparison.json', 'scores', i)}
        for i, item in enumerate(original['scores'])]
    old_count = len(scores)
    for source in sources:
        for arm in ('P0', 'Pi1'):
            comparison = load(source / arm / 'comparison.json')
            for i, item in enumerate(comparison['new_scores']):
                # Losing and unsuccessful models remain usable hypotheses.
                # Only a repeated reference is omitted, never a low score.
                if not any(x['candidate_ref'] == item['candidate_ref'] for x in scores):
                    scores.append({'candidate_ref': deepcopy(item['candidate_ref']), 'loss': None,
                        'source_score_ref': ref(source / arm / 'comparison.json', 'new_scores', i)})
    return {'scores': scores, 'original_pool_size': old_count,
        'added_source_candidates': len(scores) - old_count,
        'original_pool_ref': ref(sources[0] / 'common_comparison.json'),
        'exposed_source_refs': [ref(source / 'registration.json') for source in sources],
        'score_data_role': 'target_unscored_source_scores_are_provenance_only',
        'complete_original_pool_preserved': True, 'candidate_selection_by_source_loss': False}


def prepare_successor(paid, source, allocation_ref, prior_review, output):
    """Freeze the supported development body before either new physical source."""
    paid, source, output = map(Path, (paid, source, output))
    completed, developed = load(paid / 'receipt.json'), load(paid / 'registration.json')
    if completed['status'] != 'paid_stage_paths_completed':
        raise ValueError('Both ordinary paid staged paths must complete before independent sources')
    method = load(paid / 'frozen_method.json')
    owner = next(q for q in load(Path(prior_review) / 'common/questions.json')
                 if q['id'] == 'task:method_choice')
    candidate_ref = developed['method_source_ref']
    source_path = (paid / candidate_ref['file']).resolve()
    candidate_path = candidate_ref.get('json_path', [])
    retained_proposal = next((proposal for proposal in owner.get('method_use_plan', {}).get(
            'staged_method_proposal_refs', [])
        if (Path(prior_review) / proposal['file']).resolve() == source_path
        and candidate_path[:len(proposal.get('json_path', [])) + 2]
            == [*proposal.get('json_path', []), 'proposal', 'candidates']), None)
    retained_revision = next((revision for revision in owner.get('method_use_plan', {}).get(
            'staged_method_revision_refs', [])
        if (Path(prior_review) / revision['file']).resolve() == source_path
        and candidate_path == [*revision.get('json_path', []), 'research_program']), None)
    if ((retained_proposal is None and retained_revision is None)
            or read_method_reference(candidate_ref, paid) != method):
        raise ValueError('The prior method topic must retain this actual formed staged proposal or revision')
    allocation = read_method_reference(allocation_ref, output)
    ids = {'staged_validation_held': 106030401, 'staged_validation_changed': 106030402}
    allocation_id = allocation['allocation_id']
    if (allocation_id not in ids or allocation['seed'] != ids[allocation_id]
            or allocation['new_observation_limit'] != 168
            or [allocation[x] for x in ('training_rows', 'working_rows', 'confirmation_rows')] != [96, 24, 48]):
        raise ValueError('Use the two predeclared new sources and their unchanged 96/24/48 split')
    if not any(x['source'] == str(source) for x in developed['sources']):
        raise ValueError('Use a complete paid development source with the same retained pool')
    old = load(source / 'registration.json')
    methods = load(source / 'frozen_methods.json')
    staged = method['staged_research']
    if staged['baseline'] != methods['P0'] or staged['continuation'] != methods['Pi1']:
        raise ValueError('Both complete source stages must remain frozen')
    pool = successor_pool(developed)
    seed = allocation['seed']
    task = {**old['task'], 'seed': seed, 'apparatus_seed': seed}
    if allocation_id.endswith('_changed'):
        task['relation_radius'] = task_settings(seed)['relation_radius']
    # The held branch must use the actual held-source relation, not its label.
    elif old.get('allocation_id') != 'condition_validation_held':
        raise ValueError('The held successor retains its actual held source relation')
    rows, reserve, confirmation, returns = 120, 480, 144, 2000
    common_upper = len(pool['scores']) * rows
    per_arm = (allocation['maximum_compute'] - common_upper - 2 * reserve - confirmation - returns) // 2
    if per_arm <= 0:
        raise ValueError('Both full method calls, full-pool comparison and readout must be funded')
    pool_path = paid.parent / 'staged_successor_source_pool.json'
    save(pool_path, pool)
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'frozen_methods.json', {'P0': methods['Pi1'], 'Pi1': method})
    before = load(source / 'method_return/challenger_outcome.json')
    identity = 'native:staged:' + candidate_ref['file'] + ':' + ':'.join(map(str, candidate_ref.get('json_path', [])))
    artifact = {'id': identity, 'version': 1, 'kind': 'native_research_segment',
        'execution': {'entry': 'replay_research_program',
            'selection': 'UnifiedResearchAgent.compare_research_decision_return',
            'research_program_ref': candidate_ref},
        'source_ref': candidate_ref}
    save(output / 'staged_method_artifact.json', artifact)
    cfg = {**old, 'source': str(source), 'task': task,
        'research_config': {**old['research_config'], 'seed': seed, 'max_candidates': 4},
        'source_task_ref': ref(output / 'registration.json', 'task'),
        'methods': {arm: ref(output / 'frozen_methods.json', arm) for arm in ('P0', 'Pi1')},
        'operations': {'P0': 'program_replay', 'Pi1': 'program_replay'},
        'arm_meanings': {'P0': 'fixed_Pi1', 'Pi1': 'staged_P0_then_Pi1'},
        'native_method_artifact_refs': {'P0': before['method_artifact_ref'],
            'Pi1': ref(output / 'staged_method_artifact.json')},
        'source_pool_ref': ref(pool_path), 'source_pool_size': len(pool['scores']),
        'previous_compute': 0, 'remaining_compute': allocation['maximum_compute'],
        'total_compute_limit': allocation['maximum_compute'],
        'previous_observation_cost': 0, 'previous_measurement_cost': 0,
        'observation_limit': 168, 'measurement_limit': 168 * 3 * task['objects'],
        'training_rows': 96, 'training_probe_rows': [24] * 4, 'working_rows': 24,
        'confirmation_rows': 48, 'confirmation_seed': seed + 1000003,
        'common_comparison_upper': common_upper, 'per_arm_comparison_reserve': reserve,
        'equal_method_compute_limit': per_arm, 'confirmation_compute': confirmation,
        'method_return_compute_limit': returns, 'prefix_readout': True,
        'total_new_compute_upper': common_upper + 2 * (reserve + per_arm) + confirmation + returns,
        'data_role': 'condition_validation', 'prior_method_review': str(prior_review),
        'retained_method_pair_ref': None, 'source_cost_refs': [ref(source / 'receipt.json'), ref(paid / 'receipt.json')],
        'condition_validation_cohort_refs': [ref(output.parent / label / 'registration.json', 'task')
            for label in ('held', 'changed')],
        'allocation_ref': allocation_ref, 'allocation_id': allocation_id,
        'ledger_scope': 'separately_registered_staged_successor_original_accounts_unchanged',
        'original_ledger': ref(source / 'receipt.json'), 'development_ref': ref(paid / 'receipt.json'),
        'scope': 'fixed complete Pi1 against frozen staged method, same new data and entire old pool',
        'stop_rule': 'one predeclared physical source; no post-result body edits, seed replacement or retry'}
    # The old five pre-use guards were refuted. This body evaluates its newly
    # frozen stage guard after P0; do not inherit the earlier guard prediction.
    cfg.pop('frozen_conditions_ref', None)
    cfg.pop('condition_development_ref', None)
    save(output / 'registration.json', cfg)
    freeze_entry(output, source)
    relative = Path('scripts/stage_partition_method.py')
    write_text_incrementally(output / 'runtime' / relative, (ROOT / relative).read_text())
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('revise-paid', 'prepare-paid', 'prepare-parent-suffix', 'run-paid', 'prepare-successor', 'run-successor'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sources', type=Path, nargs='+')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--paid', type=Path)
    parser.add_argument('--prior-review', type=Path)
    parser.add_argument('--method-ref', type=Path, help='Small JSON file containing a method reference')
    parser.add_argument('--proposal-ref', type=Path, help='Small JSON file containing a sequential proposal reference')
    parser.add_argument('--allocation-ref', type=Path, help='Small JSON file containing an allocation reference')
    parser.add_argument('--artifact-refs', type=Path, help='Existing P0/Pi1 native method artifact references')
    args = parser.parse_args()
    if args.mode == 'revise-paid':
        from scripts.return_partition_pair import merge_pair_returns
        retained = load(args.prior_review / 'receipt.json')
        result = merge_pair_returns(args.prior_review, args.output,
            retained['merged_pair_refs'], load(args.allocation_ref))
    elif args.mode == 'prepare-paid':
        result = prepare_paid(args.sources, load(args.method_ref), load(args.allocation_ref), args.output,
            proposal_ref=load(args.proposal_ref) if args.proposal_ref else None)
    elif args.mode == 'prepare-parent-suffix':
        result = prepare_parent_suffix(args.sources, load(args.method_ref), load(args.allocation_ref), args.output,
            artifact_refs=load(args.artifact_refs) if args.artifact_refs else None)
    elif args.mode == 'run-paid':
        result = run_paid(args.output)
    elif args.mode == 'prepare-successor':
        result = prepare_successor(args.paid, args.source, load(args.allocation_ref), args.prior_review, args.output)
    else:
        from scripts.validate_partition_successor import run
        result = run(args.output)
    print({k: v for k, v in result.items() if k in ('status', 'compute_cost', 'new_compute_cost')})
