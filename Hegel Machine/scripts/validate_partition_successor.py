#!/usr/bin/env python3
"""One frozen P-only successor for the learned two-segment partition method."""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from itertools import islice
import json
from pathlib import Path
from statistics import mean

from scripts.run_learned_structure_successor import confirm_pair, freeze_runtime, task_settings
from scripts.run_native_decision_pair import ref
from scripts.run_partial_role_parent import load, save
from hegel_machine.cognition.checkpoint_storage import write_text_incrementally
from hegel_machine.cognition.effective_history import HistoryStore
from hegel_machine.cognition.evaluation import initial_prediction, loss, point_loss, predict
from hegel_machine.cognition.experiment_runtime import ExecutionTiming, python_execution_deadline
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.method_action_research import read_method_reference
from hegel_machine.cognition.models import Candidate, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.object_continuation import decode_entry, encode_entry
from hegel_machine.cognition.research_programs import replay_research_program
from hegel_machine.cognition.structural_research_program import research_program_fit_contract
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


ROOT = Path(__file__).resolve().parents[1]
SEED = 106030101
CONTROL_SEED = 106030102
METHOD_RETURN_RESERVE = 100


def retained_entry(location, source, index):
    payload = read_method_reference(location, source)
    if 'candidate' in payload:
        return decode_entry(payload)
    return {'candidate': Candidate(f'retained_method:{index}', Program.from_dict(payload['program']),
                'research_program'), 'kind': 'research_program', 'family_id': 'T03',
        'research_program': deepcopy(payload['research_program']),
        'training_loss': payload['training_loss'], 'validation_loss': payload['validation_loss']}


def allocation(remaining, pool_size, baseline, *, training_rows=96, working_rows=24,
               method_return_reserve=0):
    comparison_rows = training_rows + working_rows
    common = pool_size * comparison_rows
    comparison = 4 * comparison_rows
    baseline_fit = (len(research_program_fit_contract(baseline)['fit_terms']) + 2) * training_rows
    equal_cap = remaining - common - 2 * comparison - 96 - baseline_fit - method_return_reserve
    if equal_cap < baseline_fit:
        raise ValueError('The unchanged balance cannot fund the complete paired unit')
    return {'common_comparison_upper': common, 'per_arm_comparison_reserve': comparison,
        'baseline_fit_upper': baseline_fit, 'equal_method_compute_limit': equal_cap,
        'confirmation_compute': 96, 'method_return_compute_limit': method_return_reserve,
        'total_new_compute_upper': common + 2 * comparison + 96 + baseline_fit + equal_cap + method_return_reserve}


def freeze_entry(output, source):
    freeze_runtime(output, source)
    for name in ('validate_partition_successor.py', 'return_partition_pair.py',
                 'stage_partition_method.py', 'review_paid_object_pool.py'):
        relative = Path('scripts') / name
        base = source / 'runtime' / relative
        write_text_incrementally(output / 'runtime' / relative, (ROOT / relative).read_text(),
            base=base if base.exists() else None)


def register(source, output):
    source, output = Path(source), Path(output)
    receipt, cfg = load(source / 'receipt.json'), load(source / 'registration.json')
    if receipt['status'] != 'method_ready_for_independent_successor':
        raise ValueError('The paid second segment must complete before any successor is registered')
    ledger = load(source / 'method_return/receipt.json')
    frozen = load(source / 'frozen_method.json')
    original = load(Path(cfg['original_source']) / 'registration.json')
    proposal = read_method_reference(cfg['original_proposal_ref'], source)['proposal']
    baseline, challenger = proposal['baseline_research_program'], frozen['research_program']
    if baseline['transfer_partition']['kind'] != 'Pi0' or not challenger['transfer_partition'].get('second_stage'):
        raise ValueError('The registered pair requires the original fixed partition and frozen two-stage method')
    pool = load(source / 'pool_comparison.json')
    if not pool['full_comparison']:
        raise ValueError('Keep the complete paid source pool before transfer')
    confirmation_source = Path(cfg['exclude_fresh_confirmation_ref']['file']).parent
    confirmed_cfg = load(confirmation_source / 'registration.json')
    confirmed = load(confirmation_source / 'receipt.json')
    previous_observations = confirmed_cfg['previous_observation_cost'] + confirmed['observation_cost']
    previous_measurements = confirmed_cfg['previous_measurement_cost'] + confirmed['measurement_cost']
    if (previous_observations + 168 > confirmed_cfg['observation_limit']
            or previous_measurements + 168 * 12 > confirmed_cfg['measurement_limit']):
        raise ValueError('The unchanged observation allowance cannot fund the complete successor')
    historical = original['historical_knowledge_ref']
    history = HistoryStore(historical['store'], readonly=True)
    try:
        old = [row[0] for row in history.connection.execute(
            'SELECT unit FROM entries WHERE unit LIKE ? LIMIT 3', ('%' + str(SEED) + '%',))]
    finally:
        history.close()
    if old or SEED == original['task']['seed']:
        raise ValueError('This source identity is already present in the declared history')
    limits = allocation(ledger['remaining_compute'], len(pool['scores']), baseline,
        method_return_reserve=METHOD_RETURN_RESERVE)
    task = {**original['task'], 'seed': SEED, 'apparatus_seed': SEED,
            'relation_radius': task_settings(SEED)['relation_radius']}
    output.mkdir(parents=True, exist_ok=False)
    registration = {'source': str(source), 'original_source': cfg['original_source'],
        'source_task_ref': ref(output / 'registration.json', 'task'), 'task': task,
        'methods': {'P0': ref(output / 'frozen_methods.json', 'P0'),
                    'Pi1': ref(output / 'frozen_methods.json', 'Pi1')},
        'operations': {'P0': 'program_replay', 'Pi1': 'program_replay'},
        'source_pool_ref': ref(source / 'pool_comparison.json'), 'source_pool_size': len(pool['scores']),
        'source_model_ref': ref(Path(cfg['original_source']) / 'object_model.json'),
        'source_cost_refs': [ref(source / 'method_return/receipt.json')],
        'prior_method_review': str(source / 'method_return'),
        'source_cost_scope': 'unchanged current successor lineage; older sources remain referenced separately',
        'previous_compute': ledger['cumulative_compute'], 'remaining_compute': ledger['remaining_compute'],
        'total_compute_limit': cfg['total_compute_limit'], **limits,
        'previous_observation_cost': previous_observations,
        'previous_measurement_cost': previous_measurements,
        'observation_limit': confirmed_cfg['observation_limit'],
        'measurement_limit': confirmed_cfg['measurement_limit'],
        'training_rows': 96, 'training_probe_rows': [24, 24, 24, 24],
        'working_rows': 24, 'confirmation_rows': 48,
        'confirmation_seed': SEED + 1000003, 'quality_threshold': cfg['threshold'],
        'active_wall_seconds': 1800, 'research_config': {**original['research_config'],
            'seed': SEED, 'max_candidates': 4, 'minimum_improvement': cfg['threshold']},
        'factor_scope': {'P': 'new coupling and relation parameter instance',
            'S': 'same response composition and window', 'M': 'same learned instrument, names and wiring'},
        'unexposed_check': {'historical_index_matches': old, 'source_seed': original['task']['seed']},
        'scope': 'two frozen method bodies; same new data, complete source pool and available compute cap',
        'stop_rule': 'one source; no retry, body edit, pool removal, budget increase or successor expansion',
        'model_calls': 0}
    save(output / 'frozen_methods.json', {'P0': baseline, 'Pi1': challenger,
        'baseline_proposal_ref': cfg['original_proposal_ref'], 'challenger_ref': ref(source / 'frozen_method.json')})
    save(output / 'registration.json', registration)
    freeze_entry(output, source)
    return registration


def register_control(source, prior_review, output):
    """Hold the source relation in the laboratory; vary a fresh readout instance."""
    source, prior_review, output = map(Path, (source, prior_review, output))
    old, completed = load(source / 'registration.json'), load(source / 'receipt.json')
    if (completed['status'] != 'paired_methods_completed'
            or not all(item['execution_complete'] for item in completed['arms'].values())):
        raise ValueError('Retain a complete prior pair before registering its parameter-held control')
    ledger = load(prior_review / 'receipt.json')
    methods = load(source / 'frozen_methods.json')
    original = load(Path(old['original_source']) / 'registration.json')
    knowledge = load(Path(original['knowledge_root']) / 'registration.json')
    held_task = read_method_reference(knowledge['source_task_ref'], source)
    pool = load(source / 'common_comparison.json')
    if not pool['full_comparison']:
        raise ValueError('The entire current retained pool must be available')
    scores = deepcopy(pool['scores'])
    for arm in old['operations']:
        outcome = load(source / arm / 'comparison.json')
        if not outcome['full_comparison']:
            raise ValueError('The preceding new candidates were not fully compared')
        for score in outcome['new_scores']:
            if score['candidate_ref'] not in [item['candidate_ref'] for item in scores]:
                scores.append(deepcopy(score))
    previous_observations = completed['cumulative_observation_cost']
    previous_measurements = completed['cumulative_measurement_cost']
    if (previous_observations + 96 > old['observation_limit']
            or previous_measurements + 96 * 12 > old['measurement_limit']):
        raise ValueError('The unchanged observation allowance cannot fund this control')
    history = HistoryStore(original['historical_knowledge_ref']['store'], readonly=True)
    try:
        seen = [row[0] for row in history.connection.execute(
            'SELECT unit FROM entries WHERE unit LIKE ? LIMIT 3', ('%' + str(CONTROL_SEED) + '%',))]
    finally:
        history.close()
    if seen or CONTROL_SEED in (old['task']['seed'], original['task']['seed']):
        raise ValueError('The registered control identity is already present')
    limits = allocation(ledger['remaining_compute'], len(scores), methods['P0'],
        training_rows=32, working_rows=16, method_return_reserve=METHOD_RETURN_RESERVE)
    task = {**old['task'], 'seed': CONTROL_SEED, 'apparatus_seed': CONTROL_SEED,
        'relation_radius': held_task['relation_radius']}
    output.mkdir(parents=True, exist_ok=False)
    cfg = {**old, **limits, 'source': str(source), 'source_task_ref': ref(output / 'registration.json', 'task'),
        'task': task, 'methods': {arm: ref(output / 'frozen_methods.json', arm) for arm in old['operations']},
        'source_pool_ref': ref(output / 'source_pool.json'), 'source_pool_size': len(scores),
        'source_cost_refs': [ref(prior_review / 'receipt.json')],
        'previous_compute': ledger['cumulative_compute'], 'remaining_compute': ledger['remaining_compute'],
        'prior_method_review': str(prior_review), 'retained_method_pair_ref': ref(prior_review / 'pair.json'),
        'previous_observation_cost': previous_observations, 'previous_measurement_cost': previous_measurements,
        'training_rows': 32, 'training_probe_rows': [16, 16], 'working_rows': 16, 'confirmation_rows': 48,
        'confirmation_seed': CONTROL_SEED + 1000003,
        'research_config': {**old['research_config'], 'seed': CONTROL_SEED},
        'factor_scope': {'P': 'source relation held; independent coupling/readout instance',
            'S': 'same response composition and window', 'M': 'same learned instrument, names and wiring'},
        'held_parameter_source_ref': knowledge['source_task_ref'],
        'unexposed_check': {'historical_index_matches': seen,
            'source_seeds': [old['task']['seed'], original['task']['seed']]},
        'data_size_scope': '32/16 development plus unchanged48 confirmation; distinct from prior96/24 design',
        'scope': 'same frozen method bodies on independently registered parameter-held control',
        'stop_rule': 'one control; if full old pool explains stop without a method label; no seed replacement'}
    save(output / 'source_pool.json', {'scores': scores, 'full_comparison': True,
        'score_data_role': 'previous_task_only_rescore_on_new_data',
        'source_common_ref': ref(source / 'common_comparison.json'),
        'new_candidate_comparison_refs': [ref(source / arm / 'comparison.json') for arm in old['operations']]})
    save(output / 'frozen_methods.json', methods)
    save(output / 'registration.json', cfg)
    freeze_entry(output, source)
    return cfg


def register_complete_method_successor(request, output):
    """Freeze a confirmed complete method against its actual old Pi3 callee."""
    from scripts.review_paid_object_pool import parent_inputs
    from hegel_machine.cognition.relation_kernel_priors import parameter_relations
    from hegel_machine.cognition.relation_language import relation_objects
    from hegel_machine.cognition.transfer_partitions import quote_transfer_partition, transfer_partition_view

    request, output = Path(request), Path(output)
    supplied = load(request)
    source = Path(supplied['source'])
    old, actual = load(source / 'registration.json'), load(source / 'receipt.json')
    confirmation = read_method_reference(supplied['confirmation_ref'], source)
    if (confirmation.get('status') != 'confirmed'
            or confirmation.get('selected_ref') != actual['selected_ref']
            or confirmation.get('source_task_ref') != old['source_task_ref']):
        raise ValueError('Confirm this same frozen source predictor before a new successor')
    frontier, _, rows, probes, references, entries, _, _ = parent_inputs(source)
    if (not actual['execution_complete'] or not actual['full_comparison']
            or frontier['selected_ref'] != actual['selected_ref']
            or actual['selected_loss'] > old['quality_threshold']):
        raise ValueError('Retain the completed successful source and its entire current pool')
    source_task = read_method_reference(old['source_task_ref'], source)
    seed = supplied['seed']
    seen_refs = [old['source_task_ref'], *supplied['prior_source_task_refs']]
    seen_seeds = [read_method_reference(location, source)['seed'] for location in seen_refs]
    if seed in seen_seeds:
        raise ValueError('The declared successor is already in the retained source cohort')
    task = {**source_task, 'seed': seed, 'apparatus_seed': seed,
        'relation_radius': task_settings(seed)['relation_radius']}
    practice_ref = supplied.get('composition_practice_ref')
    practice = None
    if practice_ref:
        practice = read_method_reference(practice_ref, request.parent)
        practice_root = Path(practice_ref['file']).parent
        practice_cfg = load(practice_root / 'registration.json')
        if (practice.get('status') != 'paid_stage_paths_completed'
                or not practice.get('sources')
                or any(not item.get('execution_complete') or not item.get('full_comparison')
                       for item in practice['sources'])):
            raise ValueError('Complete the generated composition through the paid ordinary entry first')
        method_ref = practice_cfg['method_ref']
        # The exposed path check supports compilation, not independent benefit.
        # Its physical sources also belong to the already seen cohort.
        for item in practice_cfg['sources']:
            location = item['source_task_ref']
            if location not in seen_refs:
                seen_refs.append(location)
                seen_seeds.append(read_method_reference(location, practice_root)['seed'])
        if seed in seen_seeds:
            raise ValueError('The declared successor is already in the retained source cohort')
    else:
        method_ref = actual['executed_method_ref']
        if method_ref != old['research_method_ref']:
            raise ValueError('Freeze the complete method that actually executed on the source')
    methods = {'P0': read_method_reference(supplied['baseline_method_ref'], source),
               'Pi1': read_method_reference(method_ref, source)}
    staged = methods['Pi1'].get('staged_research')
    if practice_ref:
        if (not staged or staged['operation'] != 'revise_then_reidentify'
                or staged['baseline'] != methods['P0']
                or methods['P0'].get('transfer_partition', {}).get('kind') not in {'Pi2', 'Pi3'}
                or staged['continuation'].get('transfer_partition', {}).get('kind') != 'Pi1'
                or not practice_cfg.get('proposal_ref')):
            raise ValueError('Compare the generated composition with its own actual first-stage method')
    elif (methods['P0'].get('transfer_partition', {}).get('kind') != 'Pi3'
          or methods['Pi1'].get('transfer_partition', {}).get('kind') != 'Pi1'
          or methods['Pi1']['transfer_partition'].get('second_stage', {}).get('operation')
              != 'resolve_relation_intervals'):
        raise ValueError('Keep the old local-structure method and the complete bound parameter method')
    baseline_artifact = read_method_reference(supplied['baseline_artifact_ref'], source)
    if read_method_reference(baseline_artifact['execution']['research_program_ref'], source) != methods['P0']:
        raise ValueError('The baseline identity must retain its actual frozen method body')

    training_rows, working_rows, confirmation_rows = 96, 24, 48
    if (old['training_rows'] != training_rows or len(rows) < training_rows + working_rows
            or [probe['rows'] for probe in probes[:5]] != [24, 24, 24, 24, 24]
            or sum(probe['rows'] for probe in probes) != len(rows)):
        raise ValueError('Keep the source complete prefixes and the new 96/24 protocol')
    quotes = {arm: quote_transfer_partition(body, rows[:training_rows],
        feedback_rows=rows[training_rows:training_rows + working_rows]) for arm, body in methods.items()}
    if any(q.get('status') != 'quoted' for q in quotes.values()):
        raise ValueError('Both complete methods need executable resource bounds')
    # The new data are not acquired here. Every historical frame in four
    # complete reset trajectories is one of their 96 measured frames, so the
    # saved syntax gives a geometry ceiling independent of target values.
    q = quotes['Pi1']
    if staged:
        from hegel_machine.cognition.transfer_partitions import _saved_binding
        saved = _saved_binding(staged['baseline'])
        continuation_quote = q['stages']['continuation']
        before = continuation_quote['required_compute']
        for quoted in continuation_quote['parent_alternatives']:
            alternative = staged['baseline']['transfer_partition']['alternatives'][quoted['alternative_index']]
            terms = [Program.from_dict(term) for term in alternative['parameter_terms']]
            relations = list(dict.fromkeys(node for term in terms for parameter in saved['parameters']
                for node in parameter_relations(term, parameter['symbol'])))
            upper = training_rows * sum(3 * len(relation_objects(node)) for node in relations)
            quoted['required_compute'] += upper - quoted['geometry_compute']
            quoted['geometry_compute'] = upper
        continuation_quote['required_compute'] = max(item['required_compute']
            for item in continuation_quote['parent_alternatives'])
        q['required_compute'] += continuation_quote['required_compute'] - before
        q['quote_scope'] = 'all_legal_parent_structures_and_complete_reset_frame_count'
    else:
        view = transfer_partition_view(methods['Pi1'])
        symbol = methods['Pi1']['transfer_partition']['released_groups'][0]
        terms = [Program.from_dict(term) for term in view['parameter_terms']]
        relations = list(dict.fromkeys(node for term in terms for node in parameter_relations(term, symbol)))
        geometry_upper = training_rows * sum(3 * len(relation_objects(node)) for node in relations)
        q['required_compute'] += geometry_upper - q['geometry_compute']
        q.update(geometry_compute=geometry_upper,
            quote_scope='saved_syntax_and_complete_reset_frame_count_not_target_measurements')
    common = len({entry['candidate'].program for entry in entries}) * (training_rows + working_rows)
    comparison, review, terminal = 4 * (training_rows + working_rows), 3000, 2 * confirmation_rows
    required = common + sum(q['required_compute'] for q in quotes.values()) + 2 * comparison + review + terminal
    allocation = read_method_reference(supplied['allocation_ref'], request.parent)
    costs = read_method_reference(supplied['source_costs_ref'], request.parent)
    if (allocation['source_frontier_ref'] != ref(source / 'retained_frontier.json')
            or allocation['new_observation_limit'] != training_rows + working_rows + confirmation_rows
            or required > allocation['maximum_compute']):
        raise ValueError('Fund the complete frozen pair, full old pool, terminal readout and ordinary return')
    if costs['cumulative_protocol_observations'] + 168 > allocation['protocol_observation_limit']:
        raise ValueError('The unchanged protocol observation budget cannot fund this successor')
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'source_pool.json', {'scores': [{'candidate_ref': location} for location, _ in references],
        'source_frontier_ref': ref(source / 'retained_frontier.json'),
        'score_data_role': 'source_models_only_target_scores_not_acquired',
        'candidate_occurrences': len(entries), 'target_scores_available': False})
    save(output / 'frozen_methods.json', methods)
    challenger_ref = ref(output / 'complete_method.json')
    save(Path(challenger_ref['file']), {'id': f'native:complete_parent_method:{output.name}',
        'version': 1, 'kind': 'native_research_segment', 'execution': {
            'entry': 'replay_research_program',
            'selection': 'UnifiedResearchAgent.compare_research_decision_return',
            'research_program_ref': ref(output / 'frozen_methods.json', 'Pi1')},
        'source_ref': method_ref, 'source_confirmation_ref': supplied['confirmation_ref'],
        'source_support_inherited': False})
    cfg = {'source': str(source), 'source_task_ref': ref(output / 'registration.json', 'task'), 'task': task,
        'methods': {arm: ref(output / 'frozen_methods.json', arm) for arm in methods},
        'operations': {arm: 'program_replay' for arm in methods},
        'evaluation_arm_names': {'baseline': 'P0', 'challenger': 'Pi1'},
        'native_method_artifact_refs': {'P0': supplied['baseline_artifact_ref'], 'Pi1': challenger_ref},
        'method_labels': ({'P0': 'original_frozen_first_stage', 'Pi1': 'generated_parent_bound_composition'}
            if staged else {'P0': 'original_frozen_Pi3', 'Pi1': 'learned_structure_with_parameter_and_interval_research'}),
        'source_pool_ref': ref(output / 'source_pool.json'), 'source_pool_size': len(entries),
        'source_model_ref': supplied['source_model_ref'], 'source_confirmation_ref': supplied['confirmation_ref'],
        'source_cost_refs': [supplied['source_costs_ref'], supplied['confirmation_ref'], supplied['allocation_ref'],
            *([practice_ref, practice_cfg['proposal_ref']] if practice_ref else [])],
        'prior_method_review': supplied['prior_method_review'], 'retained_method_pair_ref': None,
        'previous_compute': 0, 'remaining_compute': allocation['maximum_compute'],
        'total_compute_limit': allocation['maximum_compute'], 'total_new_compute_upper': required,
        'method_quotes': quotes, 'common_comparison_upper': common,
        'equal_method_compute_limit': max(q['required_compute'] for q in quotes.values()),
        'per_arm_comparison_reserve': comparison, 'method_return_compute_limit': review,
        'confirmation_compute': terminal, 'training_rows': training_rows,
        'training_probe_rows': [24, 24, 24, 24], 'working_rows': working_rows,
        'confirmation_rows': confirmation_rows, 'confirmation_seed': seed + 1000003,
        'previous_observation_cost': costs['cumulative_protocol_observations'],
        'previous_measurement_cost': 0, 'observation_limit': allocation['protocol_observation_limit'],
        'measurement_account_scope': 'this unit measured separately; all source measurement costs remain referenced',
        'quality_threshold': old['quality_threshold'], 'active_wall_seconds': old['active_wall_seconds'],
        'research_config': {'seed': seed, 'max_candidates': 4, 'minimum_improvement': old['quality_threshold']},
        'allocation_ref': supplied['allocation_ref'], 'allocation_id': allocation['allocation_id'],
        'original_ledger': {'source_costs_ref': supplied['source_costs_ref']},
        'ledger_scope': 'separate_successor_within_unchanged_approved_protocol',
        'factor_scope': {'P': 'new apparatus/readout and relation radius from historical recipe',
            'S': 'source learned structure retained', 'M': 'same measurements, names, wiring and curvature'},
        'unexposed_check': {'declared_prior_source_refs': seen_refs, 'declared_prior_seeds': seen_seeds},
        'scope': ('frozen_generated_composition_vs_its_first_stage' if staged else
            'frozen_complete_method_content_not_autonomous_structure_then_parameter_controller'),
        **({'composition_practice_ref': practice_ref,
            'composition_proposal_ref': practice_cfg['proposal_ref'],
            'practice_scope': 'paid_path_verification_not_independent_support'} if practice_ref else {}),
        'stop_rule': 'One new source; stop if complete old pool solves or a method is incomplete; no seed replacement or body edits',
        'model_calls': 0, 'request_ref': ref(request)}
    save(output / 'registration.json', cfg)
    freeze_entry(output, source)
    return cfg


def register_condition_validation(source, canonical_source, prior_review, output, allocation_ref):
    """Freeze an unexposed condition check using existing paired-run machinery."""
    from hegel_machine.cognition.method_conditions import native_condition_bindings

    source, canonical_source, prior_review, output = map(Path,
        (source, canonical_source, prior_review, output))
    review = load(prior_review / 'receipt.json')
    groups = review.get('return', {}).get('research_result', {}).get(
        'method_action_research', {}).get('paired_condition_research', [])
    eligible = [item for item in groups if item.get('candidates') and not item.get('budget_incomplete')
        and item.get('status') in {'development_condition_without_group_support',
                                  'paired_condition_supported_on_declared_groups'}]
    if not eligible:
        return {'status': 'development_condition_required', 'new_compute_cost': 0,
            'new_observations': 0, 'registered': False, 'method_review_ref': ref(prior_review / 'receipt.json')}
    methods = load(canonical_source / 'frozen_methods.json')
    if load(source / 'frozen_methods.json') != methods:
        # Metadata may differ between files, but actual call bodies may not.
        actual = load(source / 'frozen_methods.json')
        if any(actual[arm] != methods[arm] for arm in ('P0', 'Pi1')):
            raise ValueError('Condition validation requires the same two actual method bodies')
    bindings = native_condition_bindings(eligible, prior_review)
    groups = [binding['research'] for binding in bindings
              if binding['baseline_program'] == methods['P0'] and binding['challenger_program'] == methods['Pi1']]
    if len(groups) != 1:
        raise ValueError('Freeze the one condition family for this exact complete method pair')
    allocation_record = read_method_reference(allocation_ref, output)
    ids = {'condition_validation_held': 106030301, 'condition_validation_changed': 106030302}
    allocation_id = allocation_record['allocation_id']
    if (allocation_id not in ids or allocation_record['seed'] != ids[allocation_id]
            or allocation_record['new_observation_limit'] != 168
            or [allocation_record[name] for name in ('training_rows', 'working_rows', 'confirmation_rows')]
                != [96, 24, 48]):
        raise ValueError('Use the registered two sources and unchanged 96/24/48 design')
    seed = allocation_record['seed']
    canonical_cfg, source_cfg = load(canonical_source / 'registration.json'), load(source / 'registration.json')
    original = load(Path(canonical_cfg['original_source']) / 'registration.json')
    source_knowledge = load(Path(original['knowledge_root']) / 'registration.json')
    held_task = read_method_reference(source_knowledge['source_task_ref'], source)
    task = {**canonical_cfg['task'], 'seed': seed, 'apparatus_seed': seed,
        'relation_radius': (held_task['relation_radius'] if allocation_id.endswith('_held')
                            else task_settings(seed)['relation_radius'])}
    history = HistoryStore(original['historical_knowledge_ref']['store'], readonly=True)
    try:
        seen = [row[0] for row in history.connection.execute(
            'SELECT unit FROM entries WHERE unit LIKE ? LIMIT 1', ('%' + str(seed) + '%',))]
    finally:
        history.close()
    if seen or seed in (canonical_cfg['task']['seed'], original['task']['seed']):
        raise ValueError('The predeclared validation source is already exposed')

    # The 103 pool is expressed in its own learned instrument coordinates.
    # Invert its already admitted ordinary role mappings; do not rename from
    # private wiring or drop a candidate because its public names differ.
    mappings = source_cfg['feature_mappings']
    inverse = []
    for mapping in mappings[:2]:
        if len(set(mapping.values())) != len(mapping) or not all(isinstance(v, str) for v in mapping.values()):
            raise ValueError('This continuation requires the saved bijective learned-role maps')
        inverse.append({value: key for key, value in mapping.items()})
    canonical_names = set(read_method_reference(canonical_cfg['source_model_ref'], source)['raw_sensor_features'])
    pool = load(source / 'common_comparison.json')
    if not pool['full_comparison']:
        raise ValueError('Keep the full completed source pool')
    scores = deepcopy(pool['scores'])
    for arm in ('P0', 'Pi1'):
        compared = load(source / arm / 'comparison.json')
        if not compared['full_comparison']:
            raise ValueError('Both source methods must complete their whole-pool comparison')
        for score in compared['new_scores']:
            if score['candidate_ref'] not in [item['candidate_ref'] for item in scores]:
                scores.append(deepcopy(score))
    rebound = []
    for index, score in enumerate(scores):
        entry = retained_entry(score['candidate_ref'], source, index)
        for mapping_index, mapping in enumerate(inverse):
            program = entry['candidate'].program.remap_features(mapping)
            if not program.required_features() <= canonical_names:
                raise ValueError('A retained source candidate has no complete admitted role binding')
            current = {'candidate': replace(entry['candidate'],
                candidate_id=f'condition_pool:{index}:{mapping_index}', program=program),
                'kind': 'retained', 'family_id': 'T03', 'source_candidate_ref': score['candidate_ref'],
                'role_binding_ref': ref(source / 'registration.json', 'feature_mappings', mapping_index)}
            rebound.append(encode_entry(current))
    if not rebound:
        raise ValueError('Retain the complete old pool before new observations')
    condition_quote = len(groups[0]['candidates'])
    # The actual three-source return required 723 units; reserve the existing
    # learner's next bounded return within this new allocation, not the old bill.
    return_reserve = 2000
    limits = allocation(allocation_record['maximum_compute'] - condition_quote,
        len(rebound), methods['P0'], method_return_reserve=return_reserve)
    limits['total_new_compute_upper'] += condition_quote
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'bound_source_pool.json', rebound)
    save(output / 'source_pool.json', {'scores': [
        {'candidate_ref': ref(output / 'bound_source_pool.json', index), 'loss': None}
        for index in range(len(rebound))], 'full_comparison': True,
        'score_data_role': 'source_pool_preserved_new_target_not_yet_scored',
        'source_comparison_ref': ref(source / 'common_comparison.json'),
        'source_candidates': len(scores), 'binding_rule': 'inverse_of_actual_first_two_ordinary_role_maps'})
    save(output / 'frozen_conditions.json', groups)
    save(output / 'frozen_methods.json', methods)
    cfg = {**canonical_cfg, **limits, 'source': str(source), 'task': task,
        'source_task_ref': ref(output / 'registration.json', 'task'),
        'methods': {arm: ref(output / 'frozen_methods.json', arm) for arm in ('P0', 'Pi1')},
        'source_pool_ref': ref(output / 'source_pool.json'), 'source_pool_size': len(rebound),
        'source_model_ref': canonical_cfg['source_model_ref'],
        'source_cost_refs': [ref(source / 'receipt.json'), ref(prior_review / 'receipt.json')],
        'prior_method_review': str(prior_review), 'retained_method_pair_ref': ref(prior_review / 'pair.json'),
        'previous_compute': 0, 'remaining_compute': allocation_record['maximum_compute'],
        'total_compute_limit': allocation_record['maximum_compute'],
        'previous_observation_cost': 0, 'previous_measurement_cost': 0,
        'observation_limit': 168, 'measurement_limit': 168 * 3 * task['objects'],
        'training_rows': 96, 'training_probe_rows': [24] * 4, 'working_rows': 24, 'confirmation_rows': 48,
        'confirmation_seed': seed + 1000003, 'data_role': 'condition_validation',
        'research_config': {**canonical_cfg['research_config'], 'seed': seed, 'max_candidates': 4},
        'frozen_conditions_ref': ref(output / 'frozen_conditions.json'),
        'condition_validation_cohort_refs': [ref(output.parent / name / 'registration.json', 'task')
            for name in ('condition_validation_held', 'condition_validation_changed')],
        'condition_prediction_compute_upper': condition_quote,
        'condition_development_ref': ref(prior_review / 'receipt.json'),
        'allocation_ref': allocation_ref, 'allocation_id': allocation_id,
        'ledger_scope': 'new_registered_condition_validation_only',
        'original_ledger': ref(source / 'receipt.json'),
        'factor_scope': {'P': allocation_record['factor'], 'S': 'source response structure retained',
            'M': 'canonical learned instrument, names, wiring and curvature retained'},
        'unexposed_check': {'historical_index_matches': seen},
        'scope': 'frozen development-condition prediction before complete matched methods and fresh confirmation',
        'stop_rule': 'one registered source; no candidate editing, retry, replacement or post-score condition selection'}
    save(output / 'registration.json', cfg)
    freeze_entry(output, source)
    return cfg


def record_frozen_condition_prediction(output, cfg, compared, budget):
    """Predict before the method outcomes; this does not admit an unvalidated guard."""
    from scripts.return_partition_pair import paid_pre_call_profile
    from hegel_machine.cognition.method_conditions import _conflict_predictions, method_condition_support

    groups = read_method_reference(cfg['frozen_conditions_ref'], output)
    required = sum(len(group['candidates']) for group in groups)
    if not budget.can_spend(required):
        raise RuntimeError('The frozen condition readout was not funded')
    profile = paid_pre_call_profile(output, cfg, compared)
    predictions = [{'candidate_predictions': _conflict_predictions(group, profile),
        'public_coverage': method_condition_support(profile, group)} for group in groups]
    spent = sum(len(item['candidate_predictions']) for item in predictions)
    budget.spend(spent)
    result = {'status': 'frozen_development_predictions', 'profile': profile, 'predictions': predictions,
        'frozen_conditions_ref': cfg['frozen_conditions_ref'], 'compute_cost': spent,
        'before_method_execution': True, 'ordinary_guard_admitted': False}
    for key in ('frozen_condition_ref', 'condition_research_ref', 'evaluation_arm_names'):
        if key in cfg:
            result[key] = deepcopy(cfg[key])
    save(output / 'condition_prediction.json', result)
    return result


def prepare_paid_condition_validation(common, prior_review, *, cohort_refs):
    """Freeze one explicitly addressed condition family before the new data."""
    from hegel_machine.cognition.method_conditions import native_condition_bindings

    common, prior_review = Path(common), Path(prior_review)
    cfg = load(common / 'registration.json')
    if (cfg.get('evaluation_role') != 'paid_development_preparation'
            or (common / 'started.json').exists() or (common / 'receipt.json').exists()):
        raise ValueError('Freeze the condition before acquiring this common development view')
    frozen = read_method_reference(cfg['frozen_condition_ref'], common)
    research_ref = frozen['condition_research_ref']
    group = read_method_reference(research_ref, common)
    if (not group.get('candidates') or group.get('budget_incomplete') or group.get('status') not in {
            'development_condition_without_group_support', 'paired_condition_supported_on_declared_groups'}):
        raise ValueError('The addressed family must contain completed development conditions')
    canonical = Path(cfg['canonical_source'])
    canonical_cfg, methods = load(canonical / 'registration.json'), load(canonical / 'frozen_methods.json')
    actual = load(common / 'frozen_methods.json')
    names = canonical_cfg['evaluation_arm_names']
    binding, = native_condition_bindings([group], prior_review)
    for side, key in (('baseline', 'baseline_method_artifact_ref'), ('challenger', 'method_artifact_ref')):
        arm = names[side]
        artifact = read_method_reference(group[key], common)
        canonical_artifact = read_method_reference(canonical_cfg['native_method_artifact_refs'][arm], common)
        if (binding[side + '_program'] != methods[arm] or actual[arm] != methods[arm]
                or artifact != canonical_artifact):
            raise ValueError('The frozen condition must address these exact canonical method bodies and artifacts')
    if cfg['source_task_ref'] not in cohort_refs:
        raise ValueError('This new source must belong to the predeclared condition-validation cohort')
    save(common / 'frozen_conditions.json', [group])
    cfg.update(frozen_conditions_ref=ref(common / 'frozen_conditions.json'),
        condition_research_ref=deepcopy(research_ref), condition_development_ref=deepcopy(research_ref),
        condition_validation_cohort_refs=deepcopy(cohort_refs),
        condition_prediction_compute_upper=len(group['candidates']),
        data_role='condition_validation', prior_method_review=str(prior_review),
        evaluation_arm_names=deepcopy(names),
        condition_evaluation_scope='independent_source_method_preference_not_new_predictor_confirmation')
    save(common / 'registration.json', cfg)
    return cfg


def _assess_pre_call_pool(agent, pool, rows, fallback, risk_quantile, budget):
    """Keep the column already computed by the unchanged complete-pool score."""
    assessed, scores = [], {}
    for entry in pool:
        program = entry['candidate'].program
        if program not in scores:
            if not budget.spend(len(rows)):
                raise RuntimeError('The registered complete old-pool comparison was unfunded')
            values = [predict(program, row, fallback) for row in rows]
            scores[program] = (mean(point_loss(value, row.target, risk_quantile)
                for value, row in zip(values, rows)), values)
        assessed.append({**entry, 'validation_loss': scores[program][0],
            'selection_basis': 'all_acquired_evidence'})
    selected = agent._best(assessed)
    return assessed, scores[selected['candidate'].program][1]


def run(output):
    from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab

    output = Path(output)
    if ROOT != (output / 'runtime').resolve() or (output / 'started.json').exists():
        raise ValueError('Execute this newly frozen successor once')
    cfg, methods = load(output / 'registration.json'), load(output / 'frozen_methods.json')
    budget, timing = SearchBudget(cfg['remaining_compute']), ExecutionTiming(('common', 'P0', 'Pi1', 'confirmation'))
    source_pool = read_method_reference(cfg['source_pool_ref'], output)['scores']
    pool = [retained_entry(item['candidate_ref'], output, i) for i, item in enumerate(source_pool)]
    model = read_method_reference(cfg['source_model_ref'], output)
    agent = UnifiedResearchAgent(ResearchConfig(**cfg['research_config']))
    batches, acquired, measured, results = [], 0, 0, {}
    seed = cfg['task'].get('seed', SEED)
    save(output / 'started.json', {'source': cfg['source'], 'seed': seed})
    try:
        with python_execution_deadline(cfg['active_wall_seconds'], activity=True):
            with timing.phase('common'), AnonymousObjectLab(**cfg['task']) as lab:
                public = lab.spec_for(('fine',))
                if (set(model['raw_sensor_features']) != set(public.feature_names)
                        or set(model['raw_control_features']) != set(lab.spec_for(()).feature_names)):
                    raise ValueError('The declared retained measurement interface changed')
                spec = replace(public, context={**public.context, 'object_model': model})
                save(output / 'public_spec.json', asdict(public))
                training, validation = [], []
                train_lengths = cfg.get('training_probe_rows', [24, 24, 24, 24])
                probes = [(f'train{index}', count) for index, count in enumerate(train_lengths)]
                probes.append(('working', cfg.get('working_rows', 24)))
                for index, (name, count) in enumerate(probes):
                    probe = lab.random_probe(seed + 3109 * (index + 1), length=count, name=name)
                    batch = lab.acquire(probe, ('fine',))
                    acquired, measured = lab.observation_cost, lab.measurement_cost
                    save(output / 'observations' / f'{name}.json', {'probe': asdict(probe), 'batch': asdict(batch)})
                    batches.append(ref(output / 'observations' / f'{name}.json'))
                    (validation if name == 'working' else training).extend(batch.observations)
                comparison = [*training, *validation]
                fallback = initial_prediction(training, spec.risk_quantile)
                assessed, selected_predictions = _assess_pre_call_pool(agent, pool, comparison,
                    fallback, spec.risk_quantile, budget)
                common_cost = budget.spent
                save(output / 'common_comparison.json', {'scores': [
                    {**old, 'loss': entry['validation_loss']} for old, entry in zip(source_pool, assessed)],
                    'rows': len(comparison), 'complete_pool_size': len(assessed), 'full_comparison': True,
                    'source_pool_ref': cfg['source_pool_ref'], 'data_refs': batches, 'compute_cost': common_cost})
                save(output / 'prepared_state.json', {'fallback': fallback,
                    'comparison_ref': ref(output / 'common_comparison.json'),
                    'training_refs': batches[:-1], 'validation_refs': batches[-1:],
                    'training_rows': len(training), 'working_rows': len(validation),
                    'public_spec_ref': ref(output / 'public_spec.json')})
                from scripts.return_partition_pair import ensure_pre_call_evidence
                compared = load(output / 'common_comparison.json')
                evidence = ensure_pre_call_evidence(output, cfg, compared, output, budget,
                    predictions=selected_predictions)
                compared['pre_call_evidence_ref'] = evidence['evidence_ref']
                save(output / 'common_comparison.json', compared)
                frame = {'spec': spec, 'training': training, 'comparison': comparison,
                    'independent_validation': False, 'assessed': assessed, 'fallback': fallback,
                    'comparison_complete': True, 'uncompared': []}
                if cfg.get('frozen_conditions_ref'):
                    if cfg.get('evaluation_role') == 'paid_development_preparation':
                        quote = paid_pair_quote(methods, cfg['feature_mappings'], training, validation,
                            len(comparison), return_reserve=cfg.get('method_return_compute_limit', 3000),
                            evaluation_arm_names=cfg['evaluation_arm_names'])
                        cfg.update(pair_quote=quote,
                            equal_method_compute_limit=quote['equal_method_compute_limit'])
                        save(output / 'registration.json', cfg)
                    record_frozen_condition_prediction(output, cfg,
                        load(output / 'common_comparison.json'), budget)
                if agent._best(assessed)['validation_loss'] <= cfg['quality_threshold']:
                    status, confirmation = 'common_pool_already_explained', None
                elif cfg.get('evaluation_role') == 'paid_development_preparation':
                    # This occurrence pays for one common view only. The
                    # existing paid-pair entry decides whether the remaining
                    # balance funds both unchanged methods after seeing it.
                    status, confirmation = 'paid_development_prepared', None
                else:
                    status, confirmation = 'paired_methods_completed', None
                    for arm in cfg['operations']:
                        with timing.phase(arm):
                            local = SearchBudget(cfg['equal_method_compute_limit'])
                            try:
                                entry = replay_research_program(methods[arm], training, local,
                                    feedback_rows=validation if (arm == 'Pi1'
                                        or methods[arm].get('staged_research')
                                        or methods[arm].get('transfer_partition', {}).get('second_stage')) else (),
                                    tolerance=cfg['quality_threshold'], risk_quantile=spec.risk_quantile)
                            finally:
                                if not budget.spend(local.spent):
                                    raise RuntimeError('A method exceeded the registered lineage allowance')
                            followup = (entry or {}).get('transfer_partition_followup', {})
                            search = (entry or {}).get('transfer_partition_search', {})
                            staged = (entry or {}).get('staged_research_result')
                            has_followup = bool(methods[arm].get('transfer_partition', {}).get('second_stage'))
                            prerequisite_unmet = bool(entry is not None and
                                followup.get('status') == 'first_stage_training_unexplained' and
                                entry.get('training_loss', float('inf')) > cfg['quality_threshold'])
                            complete = (bool(staged.get('execution_complete')) if staged is not None
                                else bool(entry is not None and search.get('completed')
                                    and not local.denied_requests
                                    and (not has_followup or followup.get('execution_complete') or prerequisite_unmet)))
                            if entry is not None:
                                entry['method_source_ids'] = [arm]
                                if arm == 'Pi1':
                                    entry['development_feedback_rows'] = len(validation)
                                save(output / arm / 'new_entry.json', encode_entry(entry))
                            compare_budget = SearchBudget(cfg['per_arm_comparison_reserve'])
                            try:
                                result = agent.compare_research_decision_return(frame, 'program_replay',
                                    [entry] if entry is not None else [], compare_budget, [],
                                    search_compute_cost=local.spent)
                            finally:
                                if not budget.spend(compare_budget.spent):
                                    raise RuntimeError('The registered same-pool comparison was unfunded')
                            selected = result['selected']
                            save(output / arm / 'selected_entry.json', encode_entry(selected))
                            outcome = {'execution_complete': complete and result['full_comparison'],
                                'full_comparison': result['full_comparison'],
                                'selected_ref': ref(output / arm / 'selected_entry.json'),
                                'selected_loss': selected['validation_loss'], 'source_pool_size': len(pool),
                                'new_candidates': int(entry is not None), 'compute_cost': local.spent + compare_budget.spent,
                                'method_compute_cost': local.spent, 'comparison_compute_cost': compare_budget.spent,
                                'same_available_method_cap': local.limit,
                                'search': search, 'followup': followup, 'budget_denied': bool(local.denied_requests),
                                'followup_prerequisite_unmet': prerequisite_unmet,
                                'development_rows': len(comparison), 'independent_support': False}
                            if staged is not None:
                                outcome['staged_research_result'] = staged
                                prefix = staged.get('baseline') or {}
                                if prefix.get('program') and cfg.get('prefix_readout'):
                                    save(output / 'prefix_predictor.json', {'staged_prefix': {
                                        'program': prefix['program'], 'fallback': fallback,
                                        'source_ref': ref(output / arm / 'new_entry.json',
                                            'staged_research_result', 'baseline'),
                                        'role': 'actual_frozen_prefix_fixed_readout_not_full_pool_selection'}})
                            save(output / arm / 'outcome.json', outcome)
                            save(output / arm / 'comparison.json', {'full_comparison': result['full_comparison'],
                                'old_comparison_ref': ref(output / 'common_comparison.json'),
                                'new_scores': [{'candidate_ref': ref(output / arm / 'new_entry.json'),
                                    'loss': item['validation_loss']} for item in result['new_claims']],
                                'selected_ref': outcome['selected_ref']})
                            results[arm] = outcome
                    if all(result['execution_complete'] for result in results.values()):
                        with timing.phase('confirmation'):
                            try:
                                confirm_cfg = dict(cfg)
                                if (output / 'prefix_predictor.json').exists():
                                    confirm_cfg['diagnostic_predictors_ref'] = ref(output / 'prefix_predictor.json')
                                confirmation = confirm_pair(output, confirm_cfg)
                            except BaseException:
                                failed = output / 'confirmation/failure.json'
                                if failed.exists():
                                    costs = load(failed)['costs']
                                    budget.spend(costs['compute_cost'])
                                    acquired += costs['observation_cost']
                                    measured += costs['measurement_cost']
                                raise
                            cost = confirmation['shared_actual_costs']
                            if not budget.spend(cost['compute_cost']):
                                raise RuntimeError('The reserved confirmation computation was unfunded')
                            acquired += cost['observation_cost']
                            measured += cost['measurement_cost']
                    else:
                        status = 'incomplete_method_not_negative'
        receipt = {'status': status, 'arms': results, 'confirmation': confirmation,
            'compute_cost': budget.spent, 'cumulative_compute': cfg['previous_compute'] + budget.spent,
            'remaining_compute': budget.remaining, 'total_compute_limit': cfg['total_compute_limit'],
            'observation_cost': acquired, 'measurement_cost': measured,
            'cumulative_observation_cost': cfg['previous_observation_cost'] + acquired,
            'cumulative_measurement_cost': cfg['previous_measurement_cost'] + measured,
            'source_pool_size': len(pool), 'source_cost_refs': cfg['source_cost_refs'],
            'method_content_pair_only': True, 'independent_architecture_support': False,
            'model_calls': 0, 'execution_timing': timing.result()}
        if cfg.get('allocation_ref'):
            receipt.update(allocation_ref=cfg['allocation_ref'], allocation_id=cfg['allocation_id'],
                ledger_scope=cfg['ledger_scope'], original_ledger=cfg['original_ledger'])
        if cfg.get('evaluation_role') == 'paid_development_preparation':
            grouped = {}
            for old, entry in zip(source_pool, assessed):
                item = grouped.setdefault(entry['candidate'].program, {
                    'mse': entry['validation_loss'], 'candidate_refs': []})
                item['candidate_refs'].append(deepcopy(old['candidate_ref']))
            selected = agent._best(assessed)
            receipt.update(full_comparison=True, scores=list(grouped.values()),
                selected_mse=selected['validation_loss'], rows=len(comparison),
                candidate_occurrences=len(assessed), unique_programs=len(grouped),
                data_refs=batches, training_refs=batches[:-1], feedback_refs=batches[-1:],
                canonical_source=cfg['canonical_source'], knowledge_source=cfg['knowledge_source'],
                source_model_ref=cfg['source_model_ref'], method_labels_created=0,
                evaluation_role='paid_development_preparation')
        receipt['pre_call_evidence_ref'] = evidence['evidence_ref']
        if cfg.get('frozen_conditions_ref'):
            receipt['condition_prediction_ref'] = ref(output / 'condition_prediction.json')
        save(output / 'receipt.json', receipt)
        if cfg.get('prior_method_review') and confirmation is not None:
            from scripts.return_partition_pair import return_pair
            receipt['method_return'] = return_pair(output, output / 'method_return',
                prior_review=Path(cfg['prior_method_review']),
                retained_pair=cfg.get('retained_method_pair_ref'))
        return receipt
    except BaseException as error:
        save(output / 'failure.json', {'error': repr(error), 'compute_known': budget.spent,
            'observation_cost': acquired, 'measurement_cost': measured, 'completed_arms': results,
            'source_cost_refs': cfg['source_cost_refs'], 'execution_timing': timing.result()})
        raise


def check_paid_source(source, prior_review, output, batch_index):
    """Read one retained winner on a named paid batch before opening research."""
    from hegel_machine.cognition.object_continuation import observations
    from hegel_machine.cognition.parallel_prediction import predict_rows

    source, prior_review, output = map(Path, (source, prior_review, output))
    if (output / 'receipt.json').exists():
        return {**load(output / 'receipt.json'), 'new_compute_cost': 0, 'reused': True}
    cfg = load(source / 'registration.json')
    state = load(source / 'prepared_state.json')
    retained = load(source / 'challenger/outcome.json')
    if not retained['execution_complete'] or not retained['full_comparison']:
        raise ValueError('The retained winner requires its completed original comparison')
    data_ref = ref(Path(state['source_references']['report']['file']), 'batches', batch_index)
    if data_ref in state['source_references']['training'] or data_ref == state['source_references']['validation']:
        raise ValueError('Reuse the paid score for a batch already compared')
    batch = read_method_reference(data_ref, source)
    rows = observations(batch['observations'])
    entry = retained_entry(retained['selected_ref'], source, 0)
    ledger = load(prior_review / 'receipt.json')
    budget = SearchBudget(ledger['remaining_compute'])
    if not budget.can_spend(len(rows)):
        raise ValueError('Original balance cannot fund the fixed paid readout')
    output.mkdir(parents=True, exist_ok=False)
    freeze_entry(output, source)
    registration = {'source': str(source), 'candidate_ref': retained['selected_ref'],
        'data_ref': data_ref, 'threshold': cfg['quality_threshold'],
        'ledger_ref': ref(prior_review / 'receipt.json'), 'maximum_compute': len(rows),
        'scope': 'exposed_historical_source_screen_not_independent_confirmation',
        'stop_rule': 'one fixed winner and paid batch; no fit, acquisition, new method label or automatic continuation'}
    save(output / 'registration.json', registration)
    timing = ExecutionTiming(('readout',))
    try:
        with timing.phase('readout'):
            budget.spend(len(rows))
            predictions = predict_rows(entry['candidate'].program, rows)
    except BaseException as error:
        save(output / 'failure.json', {'error': repr(error), 'compute_cost': budget.spent,
            'cumulative_compute': ledger['cumulative_compute'] + budget.spent,
            'remaining_compute': budget.remaining, 'source_cost_refs': [registration['ledger_ref']],
            'execution_timing': timing.result()})
        raise
    errors = [(prediction - row.target) ** 2 for prediction, row in zip(predictions, rows)]
    mse = sum(errors) / len(rows)
    result = {**registration, 'status': ('retained_winner_explains_paid_batch'
        if mse <= cfg['quality_threshold'] else 'paid_counterexamples_require_full_pool_review'),
        'rows': len(rows), 'predictions': predictions, 'mse': mse, 'maximum_squared_error': max(errors),
        'compute_cost': budget.spent, 'new_compute_cost': budget.spent,
        'cumulative_compute': ledger['cumulative_compute'] + budget.spent,
        'remaining_compute': budget.remaining, 'new_fits': 0, 'new_observations': 0,
        'model_calls': 0, 'method_labels_created': 0, 'execution_timing': timing.result()}
    save(output / 'receipt.json', result)
    return result


def _paid_pool_inputs(screen):
    """Keep every retained same-scene entry and only its actually observed view."""
    from hegel_machine.cognition.object_continuation import observations

    screen = Path(screen)
    readout = load(screen / 'receipt.json')
    source = Path(readout['source'])
    state = load(source / 'prepared_state.json')
    report_ref = state['source_references']['report']
    report_path = Path(report_ref['file'])
    report = load(report_path)
    cached = {str(report_path): report}

    def read(location):
        filename = location['file']
        if filename not in cached:
            cached[filename] = load(Path(filename))
        item = cached[filename]
        for key in location.get('json_path', []):
            item = item[key]
        return item

    evaluation = load(source / 'evaluation_contract.json')
    working_ref = state['source_references']['validation']
    if (not state['independent_validation'] or evaluation['role'] != 'paid_working_comparison'
            or evaluation['data_ref'] != working_ref or state['spec']['risk_quantile'] != .5):
        raise ValueError('Retained working losses must bind this exact MSE trajectory')
    references, working = [], {}
    for arm in ('baseline', 'challenger'):
        comparison = load(source / arm / 'comparison.json')
        if not comparison['full_comparison']:
            raise ValueError('An incomplete old pool cannot supply completed scores')
        for index, score in enumerate(comparison['scores']):
            location = score['candidate_ref']
            if location not in references:
                references.append(location)
            program = decode_entry(read(location))['candidate'].program
            working.setdefault(program, {'mse': score['loss'],
                'score_ref': ref(source / arm / 'comparison.json', 'scores', index),
                'data_ref': working_ref})
    continuation = report['object_continuation_state']
    later = [ref(report_path, 'object_continuation_state', 'selected')]
    later.extend(ref(report_path, 'object_continuation_state', 'finalists', i)
        for i in range(len(continuation['finalists'])))
    for i, frontier in enumerate(continuation['candidate_frontiers']):
        later.extend(ref(report_path, 'object_continuation_state', 'candidate_frontiers', i, 'entries', j)
            for j in range(len(frontier['entries'])))
    references.extend(location for location in later if location not in references)
    groups = {}
    for location in references:
        program = decode_entry(read(location))['candidate'].program
        groups.setdefault(program, []).append(location)
    available, excluded = [], []
    features = set(state['spec']['feature_names'])
    for index, batch in enumerate(report['batches']):
        location = ref(report_path, 'batches', index)
        rows = observations(batch['observations'])
        if rows and all(set(row.features) == features for row in rows):
            available.append((location, rows))
        else:
            excluded.append({'data_ref': location, 'rows': len(rows),
                'reason': 'different_measured_view_retained_without_zero_fill'})
    if not any(location == working_ref for location, _ in available):
        raise ValueError('The paid working trajectory is absent from this measured view')
    screened_program = decode_entry(read(readout['candidate_ref']))['candidate'].program
    if screened_program not in groups or not any(location == readout['data_ref'] for location, _ in available):
        raise ValueError('The fixed readout must belong to this retained pool and trajectory')
    return dict(source=source, state=state, readout=readout, groups=groups, working=working,
        available=available, excluded=excluded, screened_program=screened_program)


def register_paid_pool(screen, output, method_review):
    """Freeze one missing-score pass, with no new research or observations."""
    screen, output, method_review = map(Path, (screen, output, method_review))
    scene = _paid_pool_inputs(screen)
    readout = scene['readout']
    quote = sum(len(rows) for program in scene['groups'] for location, rows in scene['available']
        if not (program in scene['working'] and location == scene['working'][program]['data_ref'])
        and not (program == scene['screened_program'] and location == readout['data_ref']))
    if quote > readout['remaining_compute']:
        raise ValueError('The original allowance cannot fund this complete paid-pool pass')
    output.mkdir(parents=True, exist_ok=False)
    cfg = {'screen': str(screen), 'source': str(scene['source']),
        'ledger_ref': ref(screen / 'receipt.json'),
        'method_review_ref': ref(method_review / 'common/questions.json'),
        'maximum_compute': quote, 'remaining_compute': readout['remaining_compute'],
        'previous_compute': readout['cumulative_compute'], 'threshold': readout['threshold'],
        'candidate_refs': [r for refs in scene['groups'].values() for r in refs],
        'data_refs': [r for r, _ in scene['available']], 'other_view_refs': scene['excluded'],
        'total_paid_rows': sum(len(rows) for _, rows in scene['available']),
        'scope': 'retrospective_same_scene_complete_pool_no_method_label',
        'stop_rule': 'finish missing paid scores only; no fit, acquisition, research or new condition label'}
    save(output / 'registration.json', cfg)
    freeze_entry(output, scene['source'])
    return cfg


def score_paid_rows(program, rows, budget):
    """Charge one missing readout before predicting; failed work stays charged."""
    from hegel_machine.cognition.parallel_prediction import predict_rows

    if not budget.spend(len(rows)):
        raise RuntimeError('The complete paid comparison exceeded its registered quote')
    values = predict_rows(program, rows)
    return values, sum((value - row.target) ** 2 for value, row in zip(values, rows))


def run_paid_pool(output):

    output = Path(output)
    if (output / 'receipt.json').exists():
        return {**load(output / 'receipt.json'), 'new_compute_cost': 0, 'reused': True}
    if ROOT != (output / 'runtime').resolve() or (output / 'started.json').exists():
        raise ValueError('Run the newly frozen paid-pool pass once')
    cfg = load(output / 'registration.json')
    scene = _paid_pool_inputs(Path(cfg['screen']))
    budget, timing = SearchBudget(cfg['maximum_compute']), ExecutionTiming(('paid_pool',))
    completed = []
    save(output / 'started.json', {'registration_ref': ref(output / 'registration.json')})
    try:
        with timing.phase('paid_pool'):
            for index, (program, references) in enumerate(scene['groups'].items()):
                squared, missing, reused = 0., [], []
                for location, rows in scene['available']:
                    if program in scene['working'] and location == scene['working'][program]['data_ref']:
                        old = scene['working'][program]
                        squared += len(rows) * old['mse']
                        reused.append({**old, 'rows': len(rows)})
                    elif program == scene['screened_program'] and location == scene['readout']['data_ref']:
                        squared += len(rows) * scene['readout']['mse']
                        reused.append({'score_ref': cfg['ledger_ref'], 'data_ref': location,
                            'rows': len(rows), 'mse': scene['readout']['mse']})
                    else:
                        missing.extend(rows)
                values, added_squared = score_paid_rows(program, missing, budget)
                squared += added_squared
                saved = {'candidate_refs': references, 'mse': squared / cfg['total_paid_rows'],
                    'rows': cfg['total_paid_rows'], 'new_prediction_rows': len(missing),
                    'reused_scores': reused, 'new_predictions': values}
                save(output / 'scores' / f'{index:03d}.json', saved)
                completed.append({key: value for key, value in saved.items() if key != 'new_predictions'})
    except BaseException as error:
        save(output / 'failure.json', {'error': repr(error), 'full_comparison': False,
            'completed_scores': completed, 'compute_cost': budget.spent,
            'cumulative_compute': cfg['previous_compute'] + budget.spent,
            'remaining_compute': cfg['remaining_compute'] - budget.spent,
            'source_cost_refs': [cfg['ledger_ref']], 'execution_timing': timing.result()})
        raise
    best = min(completed, key=lambda item: item['mse'])
    result = {'status': ('paid_pool_already_explains' if best['mse'] <= cfg['threshold']
        else 'paid_pool_residual_requires_research_decision'), 'full_comparison': True,
        'candidate_occurrences': len(cfg['candidate_refs']), 'unique_programs': len(completed),
        'rows': cfg['total_paid_rows'], 'other_view_refs': cfg['other_view_refs'],
        'scores': completed, 'selected_ref': best['candidate_refs'][0], 'selected_mse': best['mse'],
        'compute_cost': budget.spent, 'new_compute_cost': budget.spent,
        'cumulative_compute': cfg['previous_compute'] + budget.spent,
        'remaining_compute': cfg['remaining_compute'] - budget.spent,
        'source_cost_refs': [cfg['ledger_ref']], 'method_review_ref': cfg['method_review_ref'],
        'new_fits': 0, 'new_observations': 0, 'model_calls': 0, 'method_labels_created': 0,
        'scope': cfg['scope'], 'execution_timing': timing.result()}
    save(output / 'receipt.json', result)
    return result


def paid_bindings(methods, public, source_model, state, target_model):
    """Use the ordinary learned-role binding rule, without changing either body."""
    from hegel_machine.cognition.abstractions import AbstractTemplate
    from hegel_machine.cognition.memory import KnowledgeLibrary
    from hegel_machine.cognition.research_program_inputs import abstract_research_inputs

    agent = UnifiedResearchAgent(ResearchConfig())
    source_spec = agent._knowledge_spec(replace(public,
        context={**public.context, 'object_model': source_model}))
    recovered = abstract_research_inputs(methods['P0'], source_spec, KnowledgeLibrary())
    if recovered is None:
        raise ValueError('The retained research inputs have no recoverable learned role binding')
    template = AbstractTemplate.from_dict(recovered['input_template'])
    target_spec = agent._knowledge_spec(TaskSpec(**state['spec'], context={'object_model': target_model}))
    admitted = list(islice(template.mappings(target_spec.feature_names,
        object_schema=json.loads(target_spec.context['object_schema'])), 2))
    if not admitted:
        raise ValueError('No ordinary learned-role mapping is available for this paid scene')
    mappings = [{old: mapping[role] for role, old in zip(template.roles, template.source_features)}
                for mapping in admitted]
    return mappings, target_spec, recovered['input_template']


def paid_pair_quote(methods, mappings, training, feedback, comparison_rows, *, return_reserve=1000,
                    evaluation_arm_names=None):
    """Quote the unchanged two bodies and four-draft ceiling before execution."""
    from hegel_machine.cognition.research_drafts import _rebind_step
    from hegel_machine.cognition.transfer_partitions import quote_transfer_partition

    names = evaluation_arm_names or {'baseline': 'P0', 'challenger': 'Pi1'}
    arms = tuple(names[side] for side in ('baseline', 'challenger'))
    quotes = {arm: [quote_transfer_partition(_rebind_step(methods[arm], mapping), training,
        feedback_rows=feedback if (methods[arm].get('staged_research') or
            methods[arm].get('transfer_partition', {}).get('second_stage')) else ())
        for mapping in mappings] for arm in arms}
    if any(item['status'] != 'quoted' for items in quotes.values() for item in items):
        raise ValueError('The unchanged method bodies could not supply an execution quote')
    per_arm = {arm: sum(item['required_compute'] for item in values) for arm, values in quotes.items()}
    comparison = 4 * comparison_rows
    return {'method_quotes': quotes, 'method_compute_upper': per_arm,
        'equal_method_compute_limit': max(per_arm.values()),
        'per_arm_comparison_reserve': comparison, 'method_return_compute_limit': return_reserve,
        'total_new_compute_upper': sum(per_arm.values()) + 2 * comparison + return_reserve,
        'max_candidates': 4, 'confirmation_compute': 0, 'new_observations': 0,
        'evaluation_role': 'paid_adaptive_development', 'independent_support': False}


def register_paid_binding_pool(source, knowledge_source, canonical_source, output):
    """Add current knowledge on the ordinary admitted bindings to a paid pool."""
    from hegel_machine.cognition.object_continuation import observations

    source, knowledge_source, canonical_source, output = map(Path,
        (source, knowledge_source, canonical_source, output))
    ledger, old_cfg = load(source / 'receipt.json'), load(source / 'registration.json')
    if not ledger['full_comparison']:
        raise ValueError('The original paid pool must be complete before adding current knowledge')
    scene = _paid_pool_inputs(Path(old_cfg['screen']))
    state = scene['state']
    cfg = load(canonical_source / 'registration.json')
    methods = load(canonical_source / 'frozen_methods.json')
    mappings, spec, template = paid_bindings(methods,
        TaskSpec(**load(knowledge_source / 'public_spec.json')),
        read_method_reference(cfg['source_model_ref'], source), state,
        read_method_reference(state['runtime_state']['object_model_ref'], source))
    current = load(knowledge_source / 'source_pool.json')['scores']
    current = list(current)
    for arm in ('P0', 'Pi1'):
        compared = load(knowledge_source / arm / 'comparison.json')
        if not compared['full_comparison']:
            raise ValueError('Current retained candidates require their completed comparison')
        for score in compared['new_scores']:
            if score['candidate_ref'] not in [item['candidate_ref'] for item in current]:
                current.append(score)
    bound = []
    for index, item in enumerate(current):
        entry = retained_entry(item['candidate_ref'], knowledge_source, index)
        for mapping_index, mapping in enumerate(mappings):
            program = entry['candidate'].program.remap_features(mapping)
            if not {name for name, _ in program.observed_variables()}.issubset(spec.feature_names):
                raise ValueError('A current candidate has an input outside the admitted learned roles')
            bound.append({'candidate': asdict(Candidate(f'paid_binding:{index}:{mapping_index}',
                program, 'retained')), 'source_candidate_ref': item['candidate_ref'],
                'feature_mapping': mapping, 'binding_index': mapping_index})
            # Program's own serializer retains executable definitions; asdict
            # would turn tuple children into the wrong persisted representation.
            bound[-1]['candidate']['program'] = program.to_dict()
    old_programs = set(scene['groups'])
    new_programs = {Program.from_dict(item['candidate']['program']) for item in bound} - old_programs
    rows = sum(len(values) for _, values in scene['available'])
    quote = rows * len(new_programs)
    if quote > ledger['remaining_compute']:
        raise ValueError('The original balance cannot fund all currently admitted bindings')
    training_refs = state['source_references']['training']
    training = [row for location in training_refs for row in observations(
        read_method_reference(location, source)['observations'])]
    feedback = [row for location, values in scene['available'] if location not in training_refs for row in values]
    pair_quote = paid_pair_quote(methods, mappings, training, feedback, rows)
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'bound_pool.json', bound)
    save(output / 'frozen_methods.json', methods)
    save(output / 'public_spec.json', asdict(spec))
    registration = {'source': str(source), 'screen': old_cfg['screen'], 'knowledge_source': str(knowledge_source),
        'canonical_source': str(canonical_source), 'source_task_ref': load(scene['source'] / 'registration.json')['source_task_ref'],
        'ledger_ref': ref(source / 'receipt.json'), 'method_review_ref': ledger['method_review_ref'],
        'source_pool_ref': ref(source / 'receipt.json'), 'bound_pool_ref': ref(output / 'bound_pool.json'),
        'frozen_methods_ref': ref(output / 'frozen_methods.json'), 'feature_mappings': mappings,
        'binding_rule': 'ordinary_first_two_learned_role_mappings', 'input_template': template,
        'target_model_ref': state['runtime_state']['object_model_ref'],
        'training_refs': training_refs, 'feedback_refs': [r for r, _ in scene['available'] if r not in training_refs],
        'data_refs': [r for r, _ in scene['available']], 'other_view_refs': scene['excluded'],
        'previous_compute': ledger['cumulative_compute'], 'remaining_compute': ledger['remaining_compute'],
        'maximum_compute': quote, 'threshold': old_cfg['threshold'], 'total_paid_rows': rows,
        'current_knowledge_references': len(current), 'bound_references': len(bound),
        'new_unique_programs': len(new_programs), 'pair_quote': pair_quote,
        'method_pair_fundable_after_common_upper': pair_quote['total_new_compute_upper'] <= ledger['remaining_compute'] - quote,
        'scope': 'retrospective_bound_complete_pool_no_method_label',
        'stop_rule': 'finish missing bound scores; retain resolved scene or unfunded research without a new condition label'}
    save(output / 'registration.json', registration)
    freeze_entry(output, scene['source'])
    return registration


def run_paid_binding_pool(output):
    """Reuse completed old scores and evaluate each new bound program once."""
    from hegel_machine.cognition.parallel_prediction import predict_rows

    output = Path(output)
    if (output / 'receipt.json').exists():
        return {**load(output / 'receipt.json'), 'new_compute_cost': 0, 'reused': True}
    if ROOT != (output / 'runtime').resolve() or (output / 'started.json').exists():
        raise ValueError('Run the newly frozen binding comparison once')
    cfg = load(output / 'registration.json')
    scene = _paid_pool_inputs(Path(cfg['screen']))
    previous = read_method_reference(cfg['source_pool_ref'], output)
    groups = {program: list(refs) for program, refs in scene['groups'].items()}
    for index, item in enumerate(load(output / 'bound_pool.json')):
        program = Program.from_dict(item['candidate']['program'])
        groups.setdefault(program, []).append(ref(output / 'bound_pool.json', index))
    old = {decode_entry(read_method_reference(item['candidate_refs'][0], output))['candidate'].program: item
           for item in previous['scores']}
    rows = [row for _, values in scene['available'] for row in values]
    budget, timing = SearchBudget(cfg['maximum_compute']), ExecutionTiming(('bound_pool',))
    completed = []
    save(output / 'started.json', {'registration_ref': ref(output / 'registration.json')})
    try:
        with timing.phase('bound_pool'):
            for index, (program, references) in enumerate(groups.items()):
                if program in old:
                    item = old[program]
                    saved = {'candidate_refs': references, 'mse': item['mse'], 'rows': item['rows'],
                        'new_prediction_rows': 0, 'reused_pool_ref': cfg['source_pool_ref']}
                else:
                    if not budget.spend(len(rows)):
                        raise RuntimeError('The complete admitted binding pool exceeded its registered quote')
                    predictions = predict_rows(program, rows)
                    saved = {'candidate_refs': references,
                        'mse': sum((y - row.target) ** 2 for y, row in zip(predictions, rows)) / len(rows),
                        'rows': len(rows), 'new_prediction_rows': len(rows), 'new_predictions': predictions}
                save(output / 'scores' / f'{index:03d}.json', saved)
                completed.append({key: value for key, value in saved.items() if key != 'new_predictions'})
    except BaseException as error:
        save(output / 'failure.json', {'error': repr(error), 'full_comparison': False,
            'completed_scores': completed, 'compute_cost': budget.spent,
            'cumulative_compute': cfg['previous_compute'] + budget.spent,
            'remaining_compute': cfg['remaining_compute'] - budget.spent,
            'source_cost_refs': [cfg['ledger_ref']], 'execution_timing': timing.result()})
        raise
    best = min(completed, key=lambda item: item['mse'])
    remaining = cfg['remaining_compute'] - budget.spent
    resolved = best['mse'] <= cfg['threshold']
    funded = cfg['pair_quote']['total_new_compute_upper'] <= remaining
    result = {'status': 'bound_pool_already_explains' if resolved else (
        'bound_pool_residual_pair_ready' if funded else 'bound_pool_residual_pair_unfunded'),
        'full_comparison': True, 'candidate_occurrences': sum(len(v) for v in groups.values()),
        'unique_programs': len(groups), 'reused_programs': len(old), 'new_programs': len(groups) - len(old),
        'rows': len(rows), 'other_view_refs': cfg['other_view_refs'], 'scores': completed,
        'selected_ref': best['candidate_refs'][0], 'selected_mse': best['mse'],
        'compute_cost': budget.spent, 'new_compute_cost': budget.spent,
        'cumulative_compute': cfg['previous_compute'] + budget.spent, 'remaining_compute': remaining,
        'source_cost_refs': [cfg['ledger_ref']], 'method_review_ref': cfg['method_review_ref'],
        'source_task_ref': cfg['source_task_ref'], 'pair_quote': cfg['pair_quote'],
        'pair_fundable': funded, 'new_fits': 0, 'new_observations': 0, 'model_calls': 0,
        'method_labels_created': 0, 'independent_support': False,
        'scope': cfg['scope'], 'execution_timing': timing.result()}
    save(output / 'receipt.json', result)
    return result


def register_paid_pair(source, output, allocation_ref=None):
    """Restore the pair under its original balance or an explicit separate allocation."""
    source, output = Path(source), Path(output)
    receipt, bound = load(source / 'receipt.json'), load(source / 'registration.json')
    if not receipt['full_comparison']:
        raise ValueError('Finish the full currently admitted pool before selecting more research')
    preparation = bound.get('evaluation_role') == 'paid_development_preparation'
    threshold = bound['quality_threshold'] if preparation else bound['threshold']
    if receipt['selected_mse'] <= threshold:
        return {'status': 'paid_scene_already_explained_no_method_label', 'new_compute_cost': 0}
    canonical = Path(bound['canonical_source'])
    canonical_cfg = load(canonical / 'registration.json')
    methods = load(source / 'frozen_methods.json')
    canonical_names = canonical_cfg.get('evaluation_arm_names', {'baseline': 'P0', 'challenger': 'Pi1'})
    names = canonical_names if preparation else bound.get('evaluation_arm_names', canonical_names)
    if preparation:
        from scripts.stage_partition_method import _rows
        frozen = load(canonical / 'frozen_methods.json')
        if any(methods[arm] != frozen[arm] for arm in names.values()):
            raise ValueError('The new paid scene must keep the canonical frozen method bodies')
        prepared = load(source / 'prepared_state.json')
        training_refs, feedback_refs = prepared['training_refs'], prepared['validation_refs']
        data_refs = receipt['data_refs']
        training, feedback = _rows(training_refs, source), _rows(feedback_refs, source)
        quote = paid_pair_quote(methods, bound['feature_mappings'], training, feedback,
            receipt['rows'], return_reserve=bound.get('method_return_compute_limit', 3000),
            evaluation_arm_names=names)
        prior_review = bound['prior_method_review']
        retained_pair = bound['retained_method_pair_ref']
        source_model_ref = bound['source_model_ref']
    else:
        quote = bound['pair_quote']
        data_refs, training_refs, feedback_refs = (bound[key] for key in
            ('data_refs', 'training_refs', 'feedback_refs'))
        prior_review = str(Path(bound['method_review_ref']['file']).parent.parent)
        retained_pair = load(Path(bound['knowledge_source']) / 'registration.json')['retained_method_pair_ref']
        source_model_ref = bound['target_model_ref']
    allocation = read_method_reference(allocation_ref, source) if allocation_ref else None
    if allocation is not None:
        if (allocation['source_pool_ref'] != ref(source / 'receipt.json')
                or allocation['new_observation_limit'] != 0):
            raise ValueError('The new allocation must bind this paid pool and no new observations')
        remaining, previous = allocation['maximum_compute'], 0
    else:
        remaining, previous = receipt['remaining_compute'], receipt['cumulative_compute']
    if quote['total_new_compute_upper'] > remaining:
        return {'status': 'complete_paired_unit_unfunded_not_method_negative',
            'remaining_compute': remaining, 'required_compute_upper': quote['total_new_compute_upper'],
            'shortfall_to_upper': quote['total_new_compute_upper'] - remaining,
            'source_ref': ref(source / 'receipt.json'), 'new_compute_cost': 0, 'method_labels_created': 0}
    output.mkdir(parents=True, exist_ok=False)
    cfg = {**quote, 'source': str(source), 'source_task_ref': bound['source_task_ref'],
        'canonical_source': str(canonical), 'knowledge_source': bound['knowledge_source'],
        'source_pool_ref': ref(source / 'receipt.json'), 'source_model_ref': source_model_ref,
        'source_cost_refs': [ref(source / 'receipt.json')],
        'prior_method_review': prior_review, 'retained_method_pair_ref': retained_pair,
        'evaluation_arm_names': names,
        'native_method_artifact_refs': canonical_cfg.get('native_method_artifact_refs'),
        'previous_compute': previous, 'remaining_compute': remaining,
        'allocation_ref': allocation_ref, 'allocation_id': allocation['allocation_id'] if allocation else None,
        'original_ledger': {'receipt_ref': ref(source / 'receipt.json'),
            'cumulative_compute': receipt['cumulative_compute'], 'remaining_compute': receipt['remaining_compute']},
        'ledger_scope': 'separately_registered_method_pair' if allocation else 'original_successor_lineage',
        'feature_mappings': bound['feature_mappings'], 'data_refs': data_refs,
        'training_refs': training_refs, 'feedback_refs': feedback_refs,
        'research_config': {**canonical_cfg['research_config'], 'max_candidates': 4},
        'active_wall_seconds': canonical_cfg['active_wall_seconds'], 'quality_threshold': threshold,
        'evaluation_role': 'paid_working_comparison',
        'source_identity': bound.get('source_identity', 'original_historical_source'),
        'scope': 'same_frozen_bodies_same_current_pool_same_paid_rows_no_new_confirmation',
        'stop_rule': 'one fully funded paired research unit; no repeat, body edit, acquisition or new source identity'}
    if bound.get('frozen_conditions_ref'):
        prediction_ref = receipt['condition_prediction_ref']
        prediction = read_method_reference(prediction_ref, source)
        if (not prediction['before_method_execution']
                or prediction['frozen_conditions_ref'] != bound['frozen_conditions_ref']):
            raise ValueError('Use the previously saved pre-method prediction of this frozen family')
        cfg.update({key: deepcopy(bound[key]) for key in (
            'frozen_condition_ref', 'frozen_conditions_ref', 'condition_research_ref',
            'condition_development_ref', 'condition_validation_cohort_refs', 'data_role',
            'condition_evaluation_scope')})
        cfg['condition_prediction_ref'] = deepcopy(prediction_ref)
    save(output / 'registration.json', cfg)
    save(output / 'frozen_methods.json', methods)
    save(output / 'public_spec.json', load(source / 'public_spec.json'))
    freeze_entry(output, source)
    return cfg


def run_paid_pair(output):
    """Execute the already registered development pair and return its actual outcomes."""
    from scripts.stage_partition_method import _rows

    output = Path(output)
    if (output / 'receipt.json').exists():
        return {**load(output / 'receipt.json'), 'new_compute_cost': 0, 'reused': True}
    if ROOT != (output / 'runtime').resolve() or (output / 'started.json').exists():
        raise ValueError('Execute the newly frozen paid method pair once')
    cfg, methods = load(output / 'registration.json'), load(output / 'frozen_methods.json')
    arm_names = cfg.get('evaluation_arm_names', {'baseline': 'P0', 'challenger': 'Pi1'})
    arms = tuple(arm_names[side] for side in ('baseline', 'challenger'))
    if cfg['total_new_compute_upper'] > cfg['remaining_compute']:
        raise ValueError('A partially funded research unit is not a method preference observation')
    if cfg.get('frozen_conditions_ref'):
        prediction = read_method_reference(cfg['condition_prediction_ref'], output)
        if (not prediction['before_method_execution']
                or prediction['frozen_conditions_ref'] != cfg['frozen_conditions_ref']):
            raise ValueError('The frozen condition prediction must precede both paired method executions')
    spec = TaskSpec(**load(output / 'public_spec.json'))
    training = _rows(cfg['training_refs'], output)
    feedback = _rows(cfg['feedback_refs'], output)
    comparison = _rows(cfg['data_refs'], output)
    prior = read_method_reference(cfg['source_pool_ref'], output)
    scores = [{'candidate_ref': location, 'loss': item['mse']}
        for item in prior['scores'] for location in item['candidate_refs']]
    pool = [{**retained_entry(item['candidate_ref'], output, index), 'validation_loss': item['loss']}
            for index, item in enumerate(scores)]
    paid_scores = {retained_entry(item['candidate_ref'], output, index)['candidate'].program: item['loss']
        for index, item in enumerate(cfg.get('shared_paid_scores', []))}
    fallback = initial_prediction(training, spec.risk_quantile)
    save(output / 'common_comparison.json', {'scores': scores, 'rows': len(comparison),
        'data_refs': cfg['data_refs'], 'full_comparison': True, 'compute_cost': 0,
        'reused_comparison_ref': cfg['source_pool_ref'], **({'pre_call_evidence_ref':
            deepcopy(prior['pre_call_evidence_ref'])} if prior.get('pre_call_evidence_ref') else {})})
    save(output / 'prepared_state.json', {'comparison_ref': ref(output / 'common_comparison.json'),
        'public_spec_ref': ref(output / 'public_spec.json'), 'fallback': fallback,
        'training_refs': cfg['training_refs'], 'validation_refs': cfg['feedback_refs']})
    cfg.update(training_rows=len(training), working_rows=len(feedback))
    save(output / 'registration.json', cfg)
    frame = {'spec': spec, 'training': training, 'comparison': comparison, 'independent_validation': False,
        'assessed': pool, 'fallback': fallback, 'comparison_complete': True, 'uncompared': []}
    agent = UnifiedResearchAgent(ResearchConfig(**cfg['research_config']))
    budget, timing, results = SearchBudget(cfg['remaining_compute']), ExecutionTiming(arms), {}
    save(output / 'started.json', {'registration_ref': ref(output / 'registration.json')})
    try:
        for arm in arms:
            with timing.phase(arm), python_execution_deadline(cfg['active_wall_seconds'], activity=True):
                local, entries, executions = SearchBudget(cfg['equal_method_compute_limit']), [], []
                has_followup = bool(methods[arm].get('transfer_partition', {}).get('second_stage'))
                has_staged = bool(methods[arm].get('staged_research'))
                paid_inputs = cfg.get('paid_fit_inputs', {}).get(arm, {})

                def paid_fit(item):
                    return {'entry': retained_entry(item['entry_ref'], output, 0),
                        'training': _rows(item['training_refs'], output),
                        'risk_quantile': item['risk_quantile'], 'cost_ref': item['cost_ref']}

                initial = paid_fit(paid_inputs['initial']) if paid_inputs.get('initial') else None
                fits = [paid_fit(item) for item in paid_inputs.get('fits', [])]
                try:
                    for mapping in cfg['feature_mappings']:
                        entry = replay_research_program(methods[arm], training, local, feature_mapping=mapping,
                            feedback_rows=feedback if has_followup or has_staged else (), tolerance=cfg['quality_threshold'],
                            risk_quantile=spec.risk_quantile, paid_initial=initial, paid_fits=fits)
                        search, followup = ((entry or {}).get('transfer_partition_search', {}),
                            (entry or {}).get('transfer_partition_followup', {}))
                        staged = (entry or {}).get('staged_research_result')
                        # A conditional fixed-readout stage has no obligation
                        # to run before its training-support prerequisite. A
                        # completed search can deliver an inadequate answer;
                        # that differs from an unfinished computation.
                        prerequisite_unmet = bool(entry is not None and
                            followup.get('status') == 'first_stage_training_unexplained' and
                            entry.get('training_loss', float('inf')) > cfg['quality_threshold'])
                        complete = (bool(staged.get('execution_complete')) if staged is not None else
                            bool(entry is not None and search.get('completed') and (not has_followup or
                                 followup.get('execution_complete') or prerequisite_unmet)))
                        executions.append({'feature_mapping': mapping, 'search': search, 'followup': followup,
                            'followup_declared': has_followup, 'followup_prerequisite_unmet': prerequisite_unmet,
                            'execution_complete': complete and not local.denied_requests,
                            **({'staged_research_result': staged} if staged is not None else {})})
                        if entry is not None:
                            entry['method_source_ids'] = [arm]
                            if has_followup or has_staged:
                                entry['development_feedback_rows'] = len(feedback)
                            entries.append(entry)
                finally:
                    if not budget.spend(local.spent):
                        raise RuntimeError('The paired execution exceeded the original ledger')
                save(output / arm / 'new_entries.json', [encode_entry(entry) for entry in entries])
                compare_budget = SearchBudget(cfg['per_arm_comparison_reserve'])
                try:
                    result = agent.compare_research_decision_return(frame, 'program_replay', entries,
                        compare_budget, [], search_compute_cost=local.spent, paid_scores=paid_scores)
                finally:
                    if not budget.spend(compare_budget.spent):
                        raise RuntimeError('The complete paid-pool comparison exceeded the original ledger')
                selected = result['selected']
                save(output / arm / 'selected_entry.json', encode_entry(selected))
                outcome = {'execution_complete': all(x['execution_complete'] for x in executions) and result['full_comparison'],
                    'full_comparison': result['full_comparison'], 'selected_ref': ref(output / arm / 'selected_entry.json'),
                    'selected_loss': selected['validation_loss'], 'source_pool_size': len(pool),
                    'new_candidates': len(entries), 'compute_cost': local.spent + compare_budget.spent,
                    'method_compute_cost': local.spent, 'comparison_compute_cost': compare_budget.spent,
                    'executions': executions, 'budget_denied': bool(local.denied_requests),
                    'same_available_method_cap': local.limit, 'development_rows': len(comparison), 'independent_support': False}
                if len(executions) == 1 and 'staged_research_result' in executions[0]:
                    outcome['staged_research_result'] = executions[0]['staged_research_result']
                save(output / arm / 'outcome.json', outcome)
                save(output / arm / 'comparison.json', {'full_comparison': result['full_comparison'],
                    'old_comparison_ref': ref(output / 'common_comparison.json'), 'new_scores': [
                        {'candidate_ref': ref(output / arm / 'new_entries.json', index), 'loss': entry['validation_loss']}
                        for index, entry in enumerate(result['new_claims'])], 'selected_ref': outcome['selected_ref']})
                results[arm] = outcome
    except BaseException as error:
        save(output / 'failure.json', {'error': repr(error), 'compute_cost': budget.spent,
            'cumulative_compute': cfg['previous_compute'] + budget.spent, 'remaining_compute': budget.remaining,
            'completed_arms': results, 'source_cost_refs': cfg['source_cost_refs'], 'execution_timing': timing.result()})
        raise
    receipt = {'status': 'paid_methods_completed' if all(x['execution_complete'] for x in results.values())
        else 'incomplete_method_not_negative', 'arms': results, 'compute_cost': budget.spent,
        'cumulative_compute': cfg['previous_compute'] + budget.spent, 'remaining_compute': budget.remaining,
        'source_cost_refs': cfg['source_cost_refs'], 'new_observations': 0, 'new_measurements': 0, 'model_calls': 0,
        'allocation_ref': cfg.get('allocation_ref'), 'allocation_id': cfg.get('allocation_id'),
        'original_ledger': cfg['original_ledger'], 'ledger_scope': cfg['ledger_scope'],
        'independent_support': False, 'execution_timing': timing.result()}
    if cfg.get('condition_prediction_ref'):
        receipt.update(condition_prediction_ref=deepcopy(cfg['condition_prediction_ref']),
            frozen_conditions_ref=deepcopy(cfg['frozen_conditions_ref']),
            data_role=cfg['data_role'], condition_evaluation_scope=cfg['condition_evaluation_scope'])
    save(output / 'receipt.json', receipt)
    if receipt['status'] == 'paid_methods_completed':
        from scripts.return_partition_pair import return_pair
        receipt['method_return'] = return_pair(output, output / 'method_return', cfg['prior_method_review'],
            retained_pair=cfg.get('retained_method_pair_ref'))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'prepare-control', 'run', 'check-paid-source',
        'prepare-paid-pool', 'run-paid-pool', 'prepare-paid-bindings', 'run-paid-bindings',
        'prepare-paid-pair', 'run-paid-pair', 'prepare-condition-validation', 'prepare-complete-successor'))
    parser.add_argument('--request', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--prior-review', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-index', type=int)
    parser.add_argument('--knowledge-source', type=Path)
    parser.add_argument('--canonical-source', type=Path)
    parser.add_argument('--allocation-file', type=Path)
    parser.add_argument('--allocation-key')
    args = parser.parse_args()
    if args.mode == 'prepare-complete-successor':
        result = register_complete_method_successor(args.request, args.output)
    elif args.mode == 'prepare-condition-validation':
        result = register_condition_validation(args.source, args.canonical_source, args.prior_review,
            args.output, ref(args.allocation_file, 'allocations', args.allocation_key))
    elif args.mode == 'prepare-paid-pair':
        allocation_ref = ref(args.allocation_file, 'allocations', args.allocation_key) if args.allocation_file else None
        result = register_paid_pair(args.source, args.output, allocation_ref=allocation_ref)
    elif args.mode == 'run-paid-pair':
        result = run_paid_pair(args.output)
    elif args.mode == 'prepare-paid-bindings':
        result = register_paid_binding_pool(args.source, args.knowledge_source, args.canonical_source, args.output)
    elif args.mode == 'run-paid-bindings':
        result = run_paid_binding_pool(args.output)
    elif args.mode == 'prepare-paid-pool':
        result = register_paid_pool(args.source, args.output, args.prior_review)
    elif args.mode == 'run-paid-pool':
        result = run_paid_pool(args.output)
    elif args.mode == 'check-paid-source':
        result = check_paid_source(args.source, args.prior_review, args.output, args.batch_index)
    elif args.mode == 'prepare-control':
        result = register_control(args.source, args.prior_review, args.output)
    else:
        result = register(args.source, args.output) if args.mode == 'prepare' else run(args.output.resolve())
    print({key: value for key, value in result.items()
           if key not in {'arms', 'task', 'research_config', 'confirmation', 'method_return',
                          'scores', 'candidate_refs', 'data_refs', 'training_refs', 'feedback_refs',
                          'feature_mappings', 'input_template'}})
