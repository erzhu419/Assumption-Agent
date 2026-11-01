#!/usr/bin/env python3
"""Read the complete retained pool on saved object probes without a laboratory."""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

from scripts.run_native_decision_pair import ref
from scripts.run_partial_role_parent import load, save
from scripts.validate_partition_successor import freeze_entry, retained_entry, score_paid_rows
from hegel_machine.cognition.checkpoint_storage import write_text_incrementally
from hegel_machine.cognition.experiment_runtime import ExecutionTiming
from hegel_machine.cognition.method_action_research import read_method_reference
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec


ROOT = Path(__file__).resolve().parents[1]


def restore_object_rows(records, public):
    """Recover before-state sensor histories, resetting at each recorded probe."""
    groups = OrderedDict()
    for record in records:
        groups.setdefault(record['probe_id'], []).append(record)
    rows, probes = [], []
    features = set(public.feature_names)
    for probe_id, group in groups.items():
        group = sorted(group, key=lambda item: item['position'])
        if [item['position'] for item in group] != list(range(len(group))):
            raise ValueError('A paid probe lacks its complete observed prefix')
        history, controls = [], []
        start = len(rows)
        for item in group:
            if item['history'] != controls:
                raise ValueError('The saved control prefix does not match this paid probe')
            if set(item['before']) != features:
                raise ValueError('The paid sensor view differs from the declared public interface')
            rows.append(Observation(dict(item['before']), item['target'], tuple(history)))
            history.append({**item['before'], '__target__': item['target']})
            controls.append(item['controls'])
        probes.append({'probe_id': probe_id, 'start': start, 'rows': len(group)})
    if not rows:
        raise ValueError('No paid sensor observations were retained')
    return rows, probes


def inputs(cfg, output):
    public = TaskSpec(**read_method_reference(cfg['public_spec_ref'], output))
    rows, probes = restore_object_rows(read_method_reference(cfg['paid_records_ref'], output), public)
    groups = OrderedDict()
    for index, location in enumerate(cfg['candidate_refs']):
        program = retained_entry(location, output, index)['candidate'].program
        required = {name for name, _ in program.observed_variables() if name != '__target__'}
        if not required.issubset(public.feature_names):
            raise ValueError('A retained candidate requires an unavailable sensor; no inferred remapping')
        groups.setdefault(program, []).append(location)
    if not groups:
        raise ValueError('The complete retained candidate pool is empty')
    return rows, probes, groups


def register(request, output):
    request, output = Path(request), Path(output)
    cfg = load(request)
    paid_balance = read_method_reference(cfg['ledger_ref'], request.parent)
    if cfg['available_compute'] > paid_balance:
        raise ValueError('The readout cannot increase the retained ledger balance')
    rows, probes, groups = inputs(cfg, request.parent)
    quote = len(rows) * len(groups)
    if quote > cfg['available_compute']:
        raise ValueError('The unchanged allocated balance cannot fund the complete paid pool')
    cfg.update(request_ref=ref(request), maximum_compute=quote,
        total_paid_rows=len(rows), probes=probes,
        unique_programs=len(groups), candidate_occurrences=len(cfg['candidate_refs']),
        evaluation_role='retrospective_paid_readout', independent_support=False,
        stop_rule='Complete this fixed pool only; no fit, acquisition or method-failure label')
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'registration.json', cfg)
    freeze_entry(output, Path(cfg['runtime_source']))
    relative = Path('scripts') / Path(__file__).name
    base = Path(cfg['runtime_source']) / 'runtime' / relative
    write_text_incrementally(output / 'runtime' / relative, (ROOT / relative).read_text(),
        base=base if base.exists() else None)
    return cfg


def run(output):
    output = Path(output)
    if (output / 'receipt.json').exists():
        return {**load(output / 'receipt.json'), 'new_compute_cost': 0, 'reused': True}
    if ROOT != (output / 'runtime').resolve() or (output / 'started.json').exists():
        raise ValueError('Execute this frozen paid-pool readout once')
    cfg = load(output / 'registration.json')
    rows, probes, groups = inputs(cfg, output)
    budget, timing = SearchBudget(cfg['maximum_compute']), ExecutionTiming(('paid_pool',))
    scores = []
    save(output / 'started.json', {'registration_ref': ref(output / 'registration.json')})
    try:
        with timing.phase('paid_pool'):
            for index, (program, references) in enumerate(groups.items()):
                values, squared = score_paid_rows(program, rows, budget)
                score = {'candidate_refs': references, 'mse': squared / len(rows),
                    'rows': len(rows), 'new_prediction_rows': len(rows)}
                save(output / 'scores' / f'{index:03d}.json', {**score, 'predictions': values})
                scores.append({**score, 'predictions_ref': ref(output / 'scores' / f'{index:03d}.json',
                    'predictions')})
    except BaseException as error:
        save(output / 'failure.json', {'error': repr(error), 'full_comparison': False,
            'completed_scores': scores, 'compute_cost': budget.spent,
            'ledger_ref': cfg['ledger_ref'], 'source_cost_refs': cfg['source_cost_refs'],
            'execution_timing': timing.result()})
        raise
    best = min(scores, key=lambda score: score['mse'])
    receipt = {'status': 'paid_pool_already_explains' if best['mse'] <= cfg['quality_threshold']
        else 'paid_pool_residual_requires_research_decision', 'full_comparison': True,
        'candidate_occurrences': len(cfg['candidate_refs']), 'unique_programs': len(groups),
        'rows': len(rows), 'probes': probes, 'scores': scores, 'selected_ref': best['candidate_refs'][0],
        'selected_mse': best['mse'], 'compute_cost': budget.spent, 'new_compute_cost': budget.spent,
        'remaining_compute': cfg['available_compute'] - budget.spent,
        'ledger_ref': cfg['ledger_ref'], 'source_cost_refs': cfg['source_cost_refs'],
        'paid_records_ref': cfg['paid_records_ref'], 'new_fits': 0, 'new_observations': 0,
        'method_labels_created': 0, 'model_calls': 0, 'independent_support': False,
        'execution_timing': timing.result()}
    save(output / 'receipt.json', receipt)
    return receipt


def relation_inputs(pool, method_ref):
    """Keep reset probes intact when separating an exposed fit/feedback prefix."""
    pool = Path(pool)
    cfg, paid = load(pool / 'registration.json'), load(pool / 'receipt.json')
    if not paid['full_comparison']:
        raise ValueError('A relation call needs the completed current-pool comparison')
    rows, probes, groups = inputs(cfg, pool)
    boundaries = [item['start'] for item in probes[1:]]
    if not boundaries:
        raise ValueError('Separate observed reset probes are required for feedback')
    split = min(boundaries, key=lambda count: (abs(count - .8 * len(rows)), count))
    method = read_method_reference(method_ref, pool)
    contract = method.get('transfer_partition', {})
    if (contract.get('kind') != 'Pi1' or method.get('staged_research')
            or not contract.get('second_stage')):
        raise ValueError('Keep the existing complete two-segment Pi1')
    return cfg, paid, rows, probes, groups, split, method


def quote_relation(pool, method_ref):
    """Quote the unchanged complete Pi1, without fitting or evaluating it."""
    from hegel_machine.cognition.transfer_partitions import quote_transfer_partition

    _, _, rows, probes, _, split, method = relation_inputs(pool, method_ref)
    quote = quote_transfer_partition(method, rows[:split], feedback_rows=rows[split:])
    if quote['status'] != 'quoted':
        raise ValueError('The retained complete Pi1 has no executable compute quote')
    return {'method_quote': quote, 'training_rows': split, 'feedback_rows': len(rows) - split,
        'comparison_rows': len(rows), 'probes': probes, 'split_rule': 'nearest_80_percent_complete_probe_prefix',
        'max_candidates': 4, 'method_compute_limit': quote['required_compute'],
        'comparison_compute_reserve': 4 * len(rows),
        'total_compute_upper': quote['required_compute'] + 4 * len(rows),
        'new_observations': 0, 'independent_support': False}


def register_relation(request, output):
    request, output = Path(request), Path(output)
    cfg = load(request)
    pool = Path(cfg['pool'])
    paid = load(pool / 'receipt.json')
    original = load(pool / 'registration.json')
    if paid['selected_mse'] <= original['quality_threshold']:
        return {'status': 'paid_pool_already_explained_no_research', 'new_compute_cost': 0}
    quote = quote_relation(pool, cfg['method_ref'])
    cap = cfg.get('method_compute_limit', quote['method_compute_limit'])
    if cap < quote['method_compute_limit']:
        raise ValueError('Both bounded relation segments must be funded before dispatch')
    balance = read_method_reference(cfg['ledger_ref'], request.parent)
    total = cap + quote['comparison_compute_reserve']
    if cfg['available_compute'] > balance or total > cfg['available_compute']:
        raise ValueError('The complete method and comparison exceed the unchanged account')
    cfg.update(quote, request_ref=ref(request), method_compute_limit=cap,
        total_compute_upper=total, equal_future_method_compute_limit=cap,
        source_pool_ref=ref(pool / 'receipt.json'), quality_threshold=original['quality_threshold'],
        paid_records_ref=original['paid_records_ref'], public_spec_ref=original['public_spec_ref'],
        stop_rule='One unchanged full Pi1 return and complete-pool comparison; no method preference label')
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'registration.json', cfg)
    freeze_entry(output, pool)
    relative = Path('scripts') / Path(__file__).name
    write_text_incrementally(output / 'runtime' / relative, (ROOT / relative).read_text(),
        base=pool / 'runtime' / relative)
    return cfg


def run_relation(output):
    from hegel_machine.cognition.evaluation import initial_prediction
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.object_continuation import encode_entry
    from hegel_machine.cognition.research_programs import replay_research_program
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    from hegel_machine.cognition.experiment_runtime import python_execution_deadline

    output = Path(output)
    if (output / 'receipt.json').exists():
        return {**load(output / 'receipt.json'), 'new_compute_cost': 0, 'reused': True}
    if ROOT != (output / 'runtime').resolve() or (output / 'started.json').exists():
        raise ValueError('Execute this frozen relation call once')
    cfg = load(output / 'registration.json')
    original, prior, rows, probes, _, split, method = relation_inputs(cfg['pool'], cfg['method_ref'])
    if split != cfg['training_rows'] or probes != cfg['probes']:
        raise ValueError('The frozen whole-probe split changed')
    public = TaskSpec(**read_method_reference(cfg['public_spec_ref'], output))
    pool = [{**retained_entry(location, output, index), 'validation_loss': item['mse']}
        for index, item in enumerate(prior['scores']) for location in item['candidate_refs']]
    frame = {'spec': public, 'training': rows[:split], 'comparison': rows,
        'assessed': pool, 'fallback': initial_prediction(rows[:split], public.risk_quantile),
        'comparison_complete': True, 'uncompared': [], 'independent_validation': False}
    agent = UnifiedResearchAgent(ResearchConfig(max_candidates=4, minimum_improvement=cfg['quality_threshold']))
    search = SearchBudget(cfg['method_compute_limit'])
    compare = SearchBudget(cfg['comparison_compute_reserve'])
    timing, actions = ExecutionTiming(('relation', 'comparison')), []
    save(output / 'started.json', {'registration_ref': ref(output / 'registration.json')})
    try:
        with python_execution_deadline(cfg['active_wall_seconds'], activity=True):
            with timing.phase('relation'):
                entry = replay_research_program(method, rows[:split], search,
                    feedback_rows=rows[split:], tolerance=cfg['quality_threshold'],
                    risk_quantile=public.risk_quantile)
            entries = [] if entry is None else [entry]
            save(output / 'new_entries.json', [encode_entry(item) for item in entries])
            with timing.phase('comparison'):
                result = agent.compare_research_decision_return(frame, 'program_replay', entries,
                    compare, actions, search_compute_cost=search.spent)
    except BaseException as error:
        spent = search.spent + compare.spent
        save(output / 'failure.json', {'error': repr(error), 'compute_cost': spent,
            'method_compute_cost': search.spent, 'comparison_compute_cost': compare.spent,
            'remaining_compute': cfg['available_compute'] - spent, 'ledger_ref': cfg['ledger_ref'],
            'source_cost_refs': cfg['source_cost_refs'], 'execution_timing': timing.result()})
        raise
    search_result = (entry or {}).get('transfer_partition_search', {})
    followup = (entry or {}).get('transfer_partition_followup', {})
    complete = bool(search_result.get('completed') and followup.get('execution_complete')
        and result['full_comparison'] and not search.denied_requests)
    selected = result['selected']
    save(output / 'selected_entry.json', encode_entry(selected))
    comparison = {'full_comparison': result['full_comparison'], 'old_comparison_ref': cfg['source_pool_ref'],
        'old_candidate_occurrences': len(pool), 'new_scores': [
            {'candidate_ref': ref(output / 'new_entries.json', index), 'loss': item['validation_loss']}
            for index, item in enumerate(result['new_claims'])],
        'selected_ref': ref(output / 'selected_entry.json'), 'compute_cost': compare.spent}
    save(output / 'comparison.json', comparison)
    status = ('relation_supported' if selected['validation_loss'] <= cfg['quality_threshold']
        else 'completed_relation_residual') if complete else (
        'relation_followup_not_applicable' if search_result.get('completed')
        and followup.get('status') == 'first_stage_training_unexplained'
        and not search.denied_requests and result['full_comparison'] else 'relation_unresolved')
    spent = search.spent + compare.spent
    receipt = {'status': status, 'execution_complete': complete, 'invocation_complete': True,
        'followup_applicability': ('refuted_by_paid_training' if
            followup.get('status') == 'first_stage_training_unexplained' else 'see_native_followup'),
        'bounded_first_search_complete': bool(search_result.get('completed')),
        'search': search_result, 'followup': followup, 'budget_denied': bool(search.denied_requests),
        'full_comparison': result['full_comparison'], 'old_candidate_occurrences': len(pool),
        'new_candidates': len(entries), 'selected_ref': comparison['selected_ref'],
        'selected_loss': selected['validation_loss'], 'prior_selected_loss': prior['selected_mse'],
        'comparison_ref': ref(output / 'comparison.json'), 'method_ref': cfg['method_ref'],
        'compute_cost': spent, 'new_compute_cost': spent, 'method_compute_cost': search.spent,
        'comparison_compute_cost': compare.spent, 'remaining_compute': cfg['available_compute'] - spent,
        'same_available_method_cap': search.limit, 'ledger_ref': cfg['ledger_ref'],
        'source_cost_refs': cfg['source_cost_refs'], 'new_observations': 0, 'model_calls': 0,
        'method_labels_created': 0, 'independent_support': False, 'actions': actions,
        'execution_timing': timing.result()}
    save(output / 'receipt.json', receipt)
    return receipt


def register_structure(request, output):
    request, output = Path(request), Path(output)
    cfg = load(request)
    relation = Path(cfg['relation'])
    returned, parent_cfg = load(relation / 'receipt.json'), load(relation / 'registration.json')
    if (returned['status'] not in {'relation_followup_not_applicable', 'completed_relation_residual'}
            or not returned['bounded_first_search_complete'] or returned['budget_denied']
            or not returned['full_comparison']):
        raise ValueError('Structure comparison needs the actual completed residual or applicability counterexample')
    cap = parent_cfg['method_compute_limit']
    reserve = parent_cfg['comparison_compute_reserve']
    balance = read_method_reference(cfg['ledger_ref'], request.parent)
    if cfg['available_compute'] > balance or cap + reserve > cfg['available_compute']:
        raise ValueError('The same-cap structure and comparison exceed the retained account')
    cfg.update(pool=parent_cfg['pool'], method_ref=parent_cfg['method_ref'],
        relation_return_ref=ref(relation / 'receipt.json'),
        source_pool_ref=parent_cfg['source_pool_ref'], public_spec_ref=parent_cfg['public_spec_ref'],
        paid_records_ref=parent_cfg['paid_records_ref'], quality_threshold=parent_cfg['quality_threshold'],
        training_rows=parent_cfg['training_rows'], feedback_rows=parent_cfg['feedback_rows'],
        comparison_rows=parent_cfg['comparison_rows'], probes=parent_cfg['probes'],
        method_compute_limit=cap, comparison_compute_reserve=reserve, max_candidates=4,
        total_compute_upper=cap + reserve, independent_support=False,
        source_role='old_schema_is_a_candidate_not_the_target_schema',
        stop_rule='One native structure return and same original pool comparison; no automatic method preference')
    public = TaskSpec(**read_method_reference(cfg['public_spec_ref'], request.parent))
    target = read_method_reference(cfg['target_model_ref'], request.parent)
    source = read_method_reference(cfg['source_model_ref'], request.parent)
    for model in (target, source):
        if (set(model['raw_sensor_features']) != set(public.feature_names)
                or set(model['raw_control_features']) != set(public.feature_bounds)):
            raise ValueError('Only already shared measurement interfaces enter this structure comparison')
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'registration.json', cfg)
    freeze_entry(output, relation)
    relative = Path('scripts') / Path(__file__).name
    write_text_incrementally(output / 'runtime' / relative, (ROOT / relative).read_text(),
        base=relation / 'runtime' / relative)
    return cfg


def run_structure(output):
    from dataclasses import replace
    from hegel_machine.cognition.evaluation import initial_prediction
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.object_continuation import encode_entry
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    from hegel_machine.cognition.experiment_runtime import python_execution_deadline

    output = Path(output)
    if (output / 'receipt.json').exists():
        return {**load(output / 'receipt.json'), 'new_compute_cost': 0, 'reused': True}
    if ROOT != (output / 'runtime').resolve() or (output / 'started.json').exists():
        raise ValueError('Execute this frozen structure call once')
    cfg = load(output / 'registration.json')
    _, prior, rows, probes, _, split, _ = relation_inputs(cfg['pool'], cfg['method_ref'])
    if split != cfg['training_rows'] or probes != cfg['probes']:
        raise ValueError('The paired complete-probe split changed')
    target = read_method_reference(cfg['target_model_ref'], output)
    public = TaskSpec(**read_method_reference(cfg['public_spec_ref'], output))
    spec = replace(public, context={**public.context, 'object_model': target})
    records = read_method_reference(cfg['paid_records_ref'], output)
    ordered = [record for probe in probes for record in sorted(
        (item for item in records if item['probe_id'] == probe['probe_id']), key=lambda item: item['position'])]
    transitions = [{'before': record['before'], 'after': record['after']} for record in ordered[:split]]
    pool = [{**retained_entry(location, output, index), 'validation_loss': item['mse']}
        for index, item in enumerate(prior['scores']) for location in item['candidate_refs']]
    frame = {'spec': spec, 'training': rows[:split], 'comparison': rows,
        'assessed': pool, 'fallback': initial_prediction(rows[:split], spec.risk_quantile),
        'comparison_complete': True, 'uncompared': [], 'independent_validation': False}
    agent = UnifiedResearchAgent(ResearchConfig(max_candidates=4, minimum_improvement=cfg['quality_threshold']))
    agent.library.object_models.append(read_method_reference(cfg['source_model_ref'], output))
    search, compare = SearchBudget(cfg['method_compute_limit']), SearchBudget(cfg['comparison_compute_reserve'])
    timing, actions = ExecutionTiming(('structure', 'comparison')), []
    save(output / 'started.json', {'registration_ref': ref(output / 'registration.json')})
    try:
        with python_execution_deadline(cfg['active_wall_seconds'], activity=True):
            with timing.phase('structure'):
                entries = agent._object_explanations(spec, rows[:split], search, actions=actions,
                    parent=agent._best(pool), new_observations=rows[split:], transitions=transitions,
                    tolerance=cfg['quality_threshold'])
            competition = agent._object_competitions.get(tuple(spec.feature_names),
                {'status': 'not_entered', 'reason': 'native_return_before_competing_schema_search'})
            save(output / 'competition.json', competition)
            save(output / 'new_entries.json', [encode_entry(item) for item in entries])
            with timing.phase('comparison'):
                result = agent.compare_research_decision_return(frame, 'structural_research', entries,
                    compare, actions, search_compute_cost=search.spent)
    except BaseException as error:
        save(output / 'competition.json', agent._object_competitions.get(tuple(spec.feature_names)))
        spent = search.spent + compare.spent
        save(output / 'failure.json', {'error': repr(error), 'compute_cost': spent,
            'method_compute_cost': search.spent, 'comparison_compute_cost': compare.spent,
            'remaining_compute': cfg['available_compute'] - spent, 'ledger_ref': cfg['ledger_ref'],
            'source_cost_refs': cfg['source_cost_refs'], 'execution_timing': timing.result()})
        raise
    selected = result['selected']
    save(output / 'selected_entry.json', encode_entry(selected))
    comparison = {'full_comparison': result['full_comparison'], 'old_comparison_ref': cfg['source_pool_ref'],
        'old_candidate_occurrences': len(pool), 'new_scores': [
            {'candidate_ref': ref(output / 'new_entries.json', index), 'loss': item['validation_loss']}
            for index, item in enumerate(result['new_claims'])],
        'selected_ref': ref(output / 'selected_entry.json'), 'compute_cost': compare.spent}
    save(output / 'comparison.json', comparison)
    searches = competition.get('searches', [])
    search_complete = bool(searches) and all(item.get('search_complete') for item in searches)
    spent = search.spent + compare.spent
    relation = read_method_reference(cfg['relation_return_ref'], output)
    receipt = {'status': 'structure_return_compared' if result['full_comparison'] else 'structure_comparison_incomplete',
        'invocation_complete': True, 'execution_complete': bool(search_complete and
            result['full_comparison'] and not search.denied_requests),
        'relation_return_ref': cfg['relation_return_ref'],
        'relation_return_scope': {'bounded_first_search_complete': relation['bounded_first_search_complete'],
            'two_segment_execution_complete': relation['execution_complete'],
            'followup_status': relation['followup']['status'],
            'followup_applicability': ('refuted_by_paid_training' if relation['status'] ==
                'relation_followup_not_applicable' else 'see_native_followup')},
        'competition_ref': ref(output / 'competition.json'), 'per_schema_searches': searches,
        'budget_denied': bool(search.denied_requests), 'full_comparison': result['full_comparison'],
        'old_candidate_occurrences': len(pool), 'new_candidates': len(entries),
        'selected_ref': comparison['selected_ref'], 'selected_loss': selected['validation_loss'],
        'prior_selected_loss': prior['selected_mse'], 'comparison_ref': ref(output / 'comparison.json'),
        'compute_cost': spent, 'new_compute_cost': spent, 'method_compute_cost': search.spent,
        'comparison_compute_cost': compare.spent, 'remaining_compute': cfg['available_compute'] - spent,
        'same_available_method_cap': search.limit, 'ledger_ref': cfg['ledger_ref'],
        'source_cost_refs': cfg['source_cost_refs'], 'new_observations': 0, 'model_calls': 0,
        'method_labels_created': 0, 'independent_support': False, 'actions': actions,
        'execution_timing': timing.result()}
    save(output / 'receipt.json', receipt)
    return receipt


def return_choice(request, output):
    """Return the two actual invocations to the existing Common method topic."""
    from dataclasses import replace
    from scripts.return_partition_pair import merge_pair_returns
    from hegel_machine.cognition.method_conditions import method_profile

    request, output = Path(request), Path(output)
    if (output / 'receipt.json').exists():
        return {**load(output / 'receipt.json'), 'new_compute_cost': 0, 'reused': True}
    cfg = load(request)
    structure = Path(cfg['structure'])
    scfg, structural = load(structure / 'registration.json'), load(structure / 'receipt.json')
    relation = Path(scfg['relation'])
    rcfg, relational = load(relation / 'registration.json'), load(relation / 'receipt.json')
    matched = ('pool', 'paid_records_ref', 'public_spec_ref', 'training_rows', 'feedback_rows',
        'probes', 'method_compute_limit', 'comparison_compute_reserve', 'quality_threshold', 'active_wall_seconds')
    if any(rcfg[key] != scfg[key] for key in matched):
        raise ValueError('The actual native invocations must share their complete recorded start')
    limit = min(3000, cfg.get('maximum_compute', 3000), structural['remaining_compute'])
    if limit <= 0:
        raise ValueError('The existing account cannot fund method-return analysis')
    _, paid, rows, probes, _, split, _ = relation_inputs(rcfg['pool'], rcfg['method_ref'])
    public = TaskSpec(**read_method_reference(rcfg['public_spec_ref'], output))
    target = read_method_reference(scfg['target_model_ref'], output)
    spec = replace(public, context={**public.context, 'object_model': target})
    selected = retained_entry(paid['selected_ref'], output, 0)
    profile = method_profile(spec, rows, incumbent=selected['candidate'], residual_loss=paid['selected_mse'],
        minimum_improvement=rcfg['quality_threshold'], research_state={
            'stage': 'native_direction_choice', 'comparison': {'stale': False, 'paid_rows': len(rows),
                'models_requested': paid['candidate_occurrences'], 'models_completed': paid['candidate_occurrences']},
            'budget': {'remaining_compute': rcfg['method_compute_limit']}})
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'registration.json', {**cfg, 'request_ref': ref(request),
        'ledger_ref': ref(structure / 'receipt.json', 'remaining_compute'), 'maximum_compute': limit})
    save(output / 'profile.json', profile)
    contracts = {
        'start_state': {'pool_ref': rcfg['source_pool_ref'], 'target_model_ref': scfg['target_model_ref'],
            'public_spec_ref': rcfg['public_spec_ref'], 'source_model_candidate_ref': scfg['source_model_ref']},
        'data': {'paid_records_ref': rcfg['paid_records_ref'], 'new_observations': 0},
        'split': {'training_rows': split, 'feedback_rows': len(rows) - split, 'probes': probes,
            'feedback_role': 'available_adaptive_development'},
        'quality_contract': {'threshold': rcfg['quality_threshold'], 'metric': 'MSE', 'coverage': len(rows)},
        'resource_contract': {'compute_limit': rcfg['method_compute_limit'],
            'comparison_reserve': rcfg['comparison_compute_reserve'], 'max_candidates': 4,
            'observation_limit': 0, 'time_limits': {'active_wall_seconds': rcfg['active_wall_seconds']}},
        'evaluation': {'role': 'paid_working_comparison', 'data_ref': rcfg['paid_records_ref'],
            'labels_already_exposed': True}}
    for name, content in contracts.items():
        save(output / (name + '.json'), content)
    identities = {
        'baseline': {'id': 'native:retained_complete_Pi1', 'entry': 'replay_research_program',
            'research_program_ref': rcfg['method_ref']},
        'challenger': {'id': 'native:competing_object_programs',
            'entry': 'UnifiedResearchAgent._object_explanations',
            'source_model_candidate_ref': scfg['source_model_ref']}}
    methods = {}
    for side, directory, actual in (('baseline', relation, relational), ('challenger', structure, structural)):
        identity = identities[side]
        execution = {key: value for key, value in identity.items() if key != 'id'}
        execution.update(selection='UnifiedResearchAgent.compare_research_decision_return', max_candidates=4)
        artifact = {'id': identity['id'], 'version': 1, 'kind': 'native_research_segment',
            'execution': execution, 'source_ref': ref(directory / 'registration.json')}
        save(output / (side + '_method.json'), artifact)
        methods[side] = {'method_ref': {'id': identity['id'], 'version': 1},
            'method_artifact_ref': ref(output / (side + '_method.json')), 'execution': execution}
        outcome = {**methods[side], **{name + '_ref': ref(output / (name + '.json')) for name in contracts},
            'profile_ref': ref(output / 'profile.json'), 'status': actual['status'],
            'execution_complete': actual['execution_complete'],
            'invocation_complete': actual.get('invocation_complete', True),
            'completed': bool(actual['execution_complete'] and actual['selected_loss'] <= rcfg['quality_threshold']),
            'loss': actual['selected_loss'], 'coverage': len(rows), 'full_comparison': actual['full_comparison'],
            'actual_outcome_ref': ref(directory / 'receipt.json'), 'comparison_ref': ref(directory / 'comparison.json'),
            'selected_ref': actual['selected_ref'], 'old_candidates': actual['old_candidate_occurrences'],
            'new_candidates': actual['new_candidates'],
            'costs': {'compute_cost': actual['compute_cost'], 'observation_cost': 0, 'measurement_cost': 0},
            'timing': {'activity_wall_seconds': actual['execution_timing']['wall_seconds']},
            'native_state': {key: actual[key] for key in ('bounded_first_search_complete', 'search',
                'followup', 'per_schema_searches', 'budget_denied') if key in actual},
            'independent_confirmation': False}
        save(output / (side + '_outcome.json'), outcome)
    pair = {'status': 'returned', 'comparison_kind': 'matched_native_segments',
        'baseline_ref': ref(output / 'baseline_outcome.json'), 'challenger_ref': ref(output / 'challenger_outcome.json'),
        'baseline_action_content': methods['baseline']['execution'], 'challenger_action_content': methods['challenger']['execution'],
        'physical_source_ref': cfg['source_task_ref'], 'origin_ref': ref(output / 'registration.json'),
        'decision_ref': ref(output / 'start_state.json'), 'data_role': 'development',
        'evaluation_role': 'paid_working_comparison', 'identification_target': 'fixed_data_method_effect'}
    save(output / 'pair.json', pair)
    allocation = {'allocation_id': 'paid_structure_choice_method_return', 'maximum_compute': limit,
        'new_observation_limit': 0, 'ledger_ref': ref(structure / 'receipt.json', 'remaining_compute')}
    save(output / 'method_review_allocation.json', allocation)
    merged = merge_pair_returns(Path(cfg['prior_review']), output / 'method_return',
        [ref(output / 'pair.json')], ref(output / 'method_review_allocation.json'))
    review = merged['return'].get('research_result', {}).get('method_action_research', {})
    pending = [item for group in review.get('paired_condition_research', [])
        if group.get('method_ref') == methods['challenger']['method_ref']
        for item in group.get('preferences', [])]
    result = {'status': merged['status'], 'method_return_ref': ref(output / 'method_return/receipt.json'),
        'method_pair_ref': ref(output / 'pair.json'), 'current_pair_preferences': pending,
        'continued_topic_ref': ref(output / 'method_return/common/questions.json'),
        'prior_topic_ref': ref(Path(cfg['prior_review']) / 'common/questions.json'),
        'source_task_ref': cfg['source_task_ref'], 'compute_cost': merged['compute_cost'],
        'new_compute_cost': merged['compute_cost'],
        'remaining_compute': structural['remaining_compute'] - merged['compute_cost'],
        'source_cost_refs': [ref(structure / 'receipt.json'), ref(relation / 'receipt.json')],
        'new_fits': 0, 'new_predictions': 0, 'new_observations': 0, 'model_calls': 0,
        'frozen_returns_unchanged': True, 'independent_support': False}
    save(output / 'receipt.json', result)
    return result


def restore_pool_rows(cfg, source):
    """Keep the original paid records and later observation batches by reference."""
    from hegel_machine.cognition.object_continuation import observations

    if 'paid_records_ref' in cfg:
        rows, probes = restore_object_rows(read_method_reference(cfg['paid_records_ref'], source),
            TaskSpec(**read_method_reference(cfg['public_spec_ref'], source)))
    else:
        # New public preparation already persisted canonical measured rows;
        # retain each probe's saved history rather than rebuild raw records.
        rows, probes = [], []
        for location in cfg['data_refs']:
            document = read_method_reference(location, source)
            batch = observations(document.get('batch', document)['observations'])
            probes.append({'probe_id': document['probe']['probe_id'], 'start': len(rows), 'rows': len(batch)})
            rows.extend(batch)
    for location in cfg.get('appended_observation_refs', []):
        document = read_method_reference(location, source)
        batch = observations(document['batch']['observations'])
        probes.append({'probe_id': document['probe']['probe_id'], 'start': len(rows), 'rows': len(batch)})
        rows.extend(batch)
    return rows, probes


def parent_inputs(source):
    """Restore the retained union and its paid losses, not the older paired start."""
    source = Path(source)
    frontier = load(source / 'retained_frontier.json')
    paid = read_method_reference(frontier['original_pool_ref'], source)
    original = load(Path(frontier['original_pool_ref']['file']).parent / 'registration.json')
    if not frontier['full_comparison'] or not paid['full_comparison']:
        raise ValueError('A parent continuation needs the completed retained pool')
    if original.get('evaluation_role') == 'paid_development_preparation':
        original = {**original, 'data_refs': paid['data_refs'], 'comparison_rows': paid['rows'],
            'feedback_rows': original['working_rows'],
            'public_spec_ref': ref(Path(frontier['original_pool_ref']['file']).parent / 'public_spec.json')}
    rows, probes = restore_pool_rows(original, source)
    references = [(location, item['mse']) for item in paid['scores'] for location in item['candidate_refs']]
    for location in frontier['appended_comparison_refs']:
        comparison = read_method_reference(location, source)
        if not comparison['full_comparison']:
            raise ValueError('Retain an incomplete comparison as unfinished, not as paid scores')
        references.extend((item['candidate_ref'], item['loss']) for item in comparison['new_scores']
            if item['candidate_ref'] not in [previous for previous, _ in references])
    if len(references) != frontier['complete_current_pool_occurrences']:
        raise ValueError('The complete retained frontier must remain present')
    entries = [dict(retained_entry(location, source, index), validation_loss=measured)
        for index, (location, measured) in enumerate(references)]
    retained = retained_entry(frontier['selected_ref'], source, 0)
    index = next(i for i, item in enumerate(entries)
        if item['candidate'].program == retained['candidate'].program)
    parent = {**retained, 'validation_loss': entries[index]['validation_loss']}
    entries[index] = parent
    if not parent.get('research_program') or parent['validation_loss'] != frontier['selected_mse']:
        raise ValueError('Restore the actually selected fitted parent and its paid comparison')
    scores = {item['candidate'].program: item['validation_loss'] for item in entries}
    return frontier, original, rows, probes, references, entries, parent, scores


def register_parent(request, output):
    request, output = Path(request), Path(output)
    cfg = load(request)
    source = Path(cfg['source'])
    frontier, original, rows, probes, _, _, parent, _ = parent_inputs(source)
    relation = load(source / 'relation/registration.json')
    ledger_ref = ref(source / 'choice_return/receipt.json', 'remaining_compute')
    available = read_method_reference(ledger_ref, source)
    cap, reserve = relation['method_compute_limit'], relation['comparison_compute_reserve']
    if cap + reserve > available:
        raise ValueError('The unchanged remaining account cannot fund this parent continuation')
    if parent['validation_loss'] <= original['quality_threshold']:
        return {'status': 'retained_parent_already_explains', 'new_compute_cost': 0}
    if probes != relation['probes'] or len(rows) != relation['comparison_rows']:
        raise ValueError('Keep the already frozen probe split and paid data')
    cfg.update(frontier_ref=ref(source / 'retained_frontier.json'), parent_ref=frontier['selected_ref'],
        source_pool_ref=frontier['original_pool_ref'], appended_comparison_refs=frontier['appended_comparison_refs'],
        paid_records_ref=original['paid_records_ref'], public_spec_ref=original['public_spec_ref'],
        ledger_ref=ledger_ref, available_compute=available, method_compute_limit=cap,
        comparison_compute_reserve=reserve, total_compute_upper=cap + reserve,
        prior_method_review=str(source / 'choice_return/method_return'),
        source_task_ref=load(source / 'choice_return/receipt.json')['source_task_ref'],
        training_rows=relation['training_rows'], feedback_rows=relation['feedback_rows'],
        comparison_rows=len(rows), probes=probes, max_candidates=4,
        quality_threshold=original['quality_threshold'], active_wall_seconds=relation['active_wall_seconds'],
        research_action='relation_form_revision', prior_selected_loss=parent['validation_loss'],
        source_cost_refs=[ref(source / 'source_costs.json'), ref(source / 'choice_return/receipt.json')],
        scope='sequential_parent_revision_on_exposed_paid_data_not_the_previous_matched_pair',
        stop_rule='One native parent revision, complete retained-pool comparison and ordinary return; no acquisition',
        independent_support=False, request_ref=ref(request))
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'registration.json', cfg)
    freeze_entry(output, source / 'structure')
    relative = Path('scripts') / Path(__file__).name
    write_text_incrementally(output / 'runtime' / relative, (ROOT / relative).read_text(),
        base=source / 'structure/runtime' / relative)
    return cfg


def register_parameter(request, output):
    """Bind the existing parameter method to the current, fully compared parent."""
    from hegel_machine.cognition.transfer_partitions import (
        prepare_parent_parameter_revision, quote_transfer_partition)

    request, output = Path(request), Path(output)
    cfg = load(request)
    source = Path(cfg['source'])
    frontier, original, rows, probes, _, _, parent, _ = parent_inputs(source)
    previous_ref = cfg.get('previous_registration_ref', ref(source / 'registration.json'))
    result_ref = cfg.get('previous_result_ref', ref(source / 'receipt.json'))
    previous = read_method_reference(previous_ref, source)
    actual = read_method_reference(result_ref, source)
    ledger_ref = cfg.get('ledger_ref', ref(source / 'method_return/receipt.json', 'remaining_compute'))
    available = read_method_reference(ledger_ref, source)
    if parent['validation_loss'] <= original['quality_threshold']:
        return {'status': 'retained_parent_already_explains', 'new_compute_cost': 0}
    if (probes != previous.get('probes', probes)
            or len(rows) != previous.get('comparison_rows', previous['training_rows'] + previous.get('working_rows', 0))
            or actual['selected_ref'] != frontier['selected_ref']):
        raise ValueError('Keep the current selected parent, complete paid pool and probe split')
    method = read_method_reference(cfg['source_method_ref'], source)
    bound = prepare_parent_parameter_revision(parent, method,
        source_method_ref=cfg['source_method_ref'], parent_ref=frontier['selected_ref'])
    if bound.get('research_program') is None:
        raise ValueError('The source method cannot bind to the current retained parent')
    split = previous['training_rows']
    quote = quote_transfer_partition(bound['research_program'], rows[:split], feedback_rows=rows[split:])
    if quote['status'] != 'quoted':
        raise ValueError('The current complete parent has no funded parameter-research quote')
    reserve, review = 4 * len(rows), 3000
    cap = quote['required_compute']
    # The original raw-record scene saved a prediction column. A completed
    # successor pair already has its selected loss; it need not predict again
    # merely to register the same parameter operation on its current parent.
    values_ref = (ref(source / 'diagnostics.json', 'selected_predictions')
        if 'paid_records_ref' in original else None)
    preparation = len(rows) if values_ref else 0
    if preparation + cap + reserve + review > available:
        raise ValueError('The unchanged remaining account cannot fund research and ordinary return')
    values = read_method_reference(values_ref, source) if values_ref else None
    if values_ref and (values is None or len(values) != len(rows)):
        raise ValueError('The actual current parent needs its saved complete predictions')
    output.mkdir(parents=True, exist_ok=False)
    data = {key: original[key] for key in ('paid_records_ref', 'data_refs', 'appended_observation_refs') if key in original}
    if values_ref:
        residuals = [value - row.target for value, row in zip(values, rows)]
        summaries = []
        for probe in probes:
            errors = residuals[probe['start']:probe['start'] + probe['rows']]
            summaries.append({**probe, 'mse': sum(value * value for value in errors) / len(errors),
                'mean_signed_residual': sum(errors) / len(errors),
                'maximum_absolute_residual': max(map(abs, errors))})
        save(output / 'paid_residuals.json', {
            'status': 'saved_predictions_summarized', 'parent_ref': frontier['selected_ref'],
            'predictions_ref': values_ref, **data, 'probes': summaries, 'rows': len(rows),
            'mse': sum(value * value for value in residuals) / len(residuals),
            'compute_cost': preparation, 'new_compute_cost': preparation,
            'remaining_compute': available - preparation, 'ledger_ref': ledger_ref,
            'new_predictions': 0, 'new_fits': 0, 'new_observations': 0, 'model_calls': 0,
            'independent_support': False, 'data_role': 'exposed_adaptive_development'})
    save(output / 'method_instance.json', bound)
    cfg.update(frontier_ref=ref(source / 'retained_frontier.json'), parent_ref=frontier['selected_ref'],
        source_pool_ref=frontier['original_pool_ref'], appended_comparison_refs=frontier['appended_comparison_refs'],
        **data, public_spec_ref=original['public_spec_ref'],
        ledger_ref=ref(output / 'paid_residuals.json', 'remaining_compute') if values_ref else ledger_ref,
        previous_ledger_ref=ledger_ref, available_compute=available - preparation,
        preparation_compute=preparation, method_compute_limit=cap, method_quote=quote,
        comparison_compute_reserve=reserve, total_compute_upper=cap + reserve,
        method_return_compute_limit=review, prior_method_review=cfg.get('prior_method_review', str(source / 'method_return')),
        source_task_ref=actual.get('source_task_ref') or previous['source_task_ref'], training_rows=split,
        feedback_rows=len(rows) - split, comparison_rows=len(rows), probes=probes, max_candidates=4,
        quality_threshold=original['quality_threshold'], active_wall_seconds=previous['active_wall_seconds'],
        research_action='relation_parameter_revision', research_method_ref=ref(output / 'method_instance.json', 'research_program'),
        paid_initializer_cost_ref=result_ref, prior_selected_loss=parent['validation_loss'],
        source_cost_refs=[*cfg.get('source_cost_refs', [ref(source / 'source_costs.json'),
            ref(source / 'method_return/receipt.json')]), *([ref(output / 'paid_residuals.json')] if values_ref else [])],
        scope='sequential_parameter_research_on_current_parent_not_direction_replay_or_old_matched_pair',
        stop_rule='One complete parameter research and retained-pool comparison followed by ordinary return; no acquisition',
        independent_support=False, request_ref=ref(request))
    save(output / 'registration.json', cfg)
    runtime_source = Path(cfg.get('runtime_source', source))
    freeze_entry(output, runtime_source)
    relative = Path('scripts') / Path(__file__).name
    write_text_incrementally(output / 'runtime' / relative, (ROOT / relative).read_text(),
        base=runtime_source / 'runtime' / relative)
    return cfg


def register_frozen_method_development(request, output):
    """Keep the two actual method bodies and complete knowledge for a new source."""
    request, output = Path(request), Path(output)
    supplied = load(request)
    canonical, knowledge = Path(supplied['canonical_source']), Path(supplied['knowledge_source'])
    previous = load(canonical / 'registration.json')
    methods = load(canonical / 'frozen_methods.json')
    old_task = read_method_reference(previous['source_task_ref'], canonical)
    task = supplied['task']
    if task['seed'] == old_task['seed']:
        raise ValueError('Do not repeat the original physical source')
    fixed_interface = ('objects', 'name_seed', 'wiring_seed', 'actuator_curvature',
                       'variant', 'profile', 'readout')
    if any(task.get(key) != old_task.get(key) for key in fixed_interface):
        raise ValueError('This fixed-input method comparison retains the measurement interface')
    frontier, _, _, _, references, entries, _, _ = parent_inputs(knowledge)
    allocation = read_method_reference(supplied['allocation_ref'], request.parent)
    prior = read_method_reference(supplied['prior_ledger_ref'], request.parent)
    lengths, working = supplied['training_probe_rows'], supplied['working_rows']
    rows = sum(lengths) + working
    common = len({entry['candidate'].program for entry in entries}) * rows
    if (rows != allocation['new_observation_limit'] or common > allocation['maximum_compute']
            or allocation['source_frontier_ref'] != ref(knowledge / 'retained_frontier.json')):
        raise ValueError('Register the complete common pool and new observations within their allocation')
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'source_pool.json', {'scores': [{'candidate_ref': location} for location, _ in references],
        'source_frontier_ref': ref(knowledge / 'retained_frontier.json'),
        'candidate_occurrences': len(entries), 'target_scores_available': False})
    save(output / 'frozen_methods.json', methods)
    cfg = {'source': str(canonical), 'canonical_source': str(canonical), 'knowledge_source': str(knowledge),
        'task': task, 'source_task_ref': ref(output / 'registration.json', 'task'),
        'source_pool_ref': ref(output / 'source_pool.json'), 'source_pool_size': len(entries),
        'source_model_ref': previous['source_model_ref'],
        'protected_frontier_ref': ref(knowledge / 'retained_frontier.json'),
        'source_cost_refs': [supplied['prior_ledger_ref'], supplied['prior_source_costs_ref'], supplied['allocation_ref']],
        'evaluation_arm_names': previous['evaluation_arm_names'],
        'native_method_artifact_refs': previous['native_method_artifact_refs'],
        'retained_method_pair_ref': ref(canonical / 'method_return/pair.json'),
        'prior_method_review': supplied['prior_method_review'], 'feature_mappings': [{}],
        'method_return_compute_limit': 3000,
        'previous_compute': 0, 'remaining_compute': allocation['maximum_compute'],
        'total_compute_limit': allocation['maximum_compute'],
        'previous_observation_cost': 0, 'previous_measurement_cost': 0,
        'observation_limit': rows, 'measurement_limit': rows * 3 * task['objects'],
        'training_rows': sum(lengths), 'training_probe_rows': lengths, 'working_rows': working,
        'confirmation_rows': 0, 'common_comparison_upper': common,
        'quality_threshold': previous['quality_threshold'], 'research_config': previous['research_config'],
        'active_wall_seconds': previous['active_wall_seconds'],
        'allocation_ref': supplied['allocation_ref'], 'allocation_id': allocation['allocation_id'],
        'ledger_scope': 'new_development_source_inside_approved_offline_protocol',
        'original_ledger': {'receipt_ref': supplied['prior_ledger_ref'], 'remaining_compute': prior['remaining_compute']},
        'evaluation_role': 'paid_development_preparation', 'data_role': 'development',
        'source_identity': 'new_predeclared_physical_development_source',
        'interface_scope': 'unchanged_measured_channels_and_calibration_not_new_role_mapping',
        'scope': 'new source common preparation; same frozen Pi1/Pi3 and full retained pool',
        'stop_rule': 'Stop after common comparison; only unresolved full pool can enter a fully funded paired unit. No independent confirmation or source replacement.',
        'request_ref': ref(request), 'model_calls': 0, 'independent_support': False}
    if supplied.get('frozen_condition_ref'):
        cfg['frozen_condition_ref'] = supplied['frozen_condition_ref']
    save(output / 'registration.json', cfg)
    if supplied.get('frozen_condition_ref'):
        from scripts.validate_partition_successor import prepare_paid_condition_validation
        cfg = prepare_paid_condition_validation(output, Path(supplied['prior_method_review']),
            cohort_refs=supplied['condition_validation_cohort_refs'])
    freeze_entry(output, canonical)
    relative = Path('scripts') / Path(__file__).name
    write_text_incrementally(output / 'runtime' / relative, (ROOT / relative).read_text(),
        base=canonical / 'runtime' / relative)
    return cfg


def register_parent_method_pair(request, output):
    """Prepare the historical parent decision for the existing paid pair runner."""
    from dataclasses import asdict
    from hegel_machine.cognition.structural_research_program import research_program_fit_contract
    from hegel_machine.cognition.transfer_partitions import (
        prepare_parent_parameter_revision, quote_transfer_partition)

    request, output = Path(request), Path(output)
    supplied = load(request)
    source = Path(supplied['source'])
    frontier, original, rows, probes, references, entries, parent, _ = parent_inputs(source)
    relation = load(source / 'relation/registration.json')
    split = relation['training_rows']
    if (len(rows) != relation['comparison_rows'] or probes != relation['probes']
            or split not in {probe['start'] for probe in probes[1:]}):
        raise ValueError('Restore the historical complete pool and unchanged reset-probe split')
    if parent['validation_loss'] <= original['quality_threshold']:
        return {'status': 'paid_parent_already_explained_no_method_label', 'new_compute_cost': 0}
    baseline_ref = supplied.get('baseline_method_ref', relation['method_ref'])
    challenger_ref = supplied['challenger_method_ref']
    source_method = read_method_reference(baseline_ref, source)
    if not source_method.get('transfer_partition', {}).get('second_stage'):
        raise ValueError('Preserve the complete Pi1 including its conditional interval segment')
    baseline = prepare_parent_parameter_revision(parent, source_method,
        source_method_ref=baseline_ref, parent_ref=frontier['selected_ref'])
    if baseline.get('status') != 'ready':
        raise ValueError('The complete Pi1 must bind to the actual pre-call parent')
    methods = {'Pi1': baseline['research_program'],
        'Pi3': read_method_reference(challenger_ref, source)}
    if (methods['Pi3'].get('transfer_partition', {}).get('kind') != 'Pi3'
            or research_program_fit_contract(methods['Pi3'])['program'] != parent['candidate'].program):
        raise ValueError('The generated Pi3 must retain this exact pre-call parent body')
    quotes = {arm: quote_transfer_partition(body, rows[:split], feedback_rows=rows[split:])
              for arm, body in methods.items()}
    if any(item.get('status') != 'quoted' for item in quotes.values()):
        raise ValueError('Both complete frozen research segments need executable quotes')
    cap, comparison, common = max(item['required_compute'] for item in quotes.values()), 4 * len(rows), 3000
    quote = {'method_quotes': quotes, 'equal_method_compute_limit': cap,
        'per_arm_comparison_reserve': comparison, 'method_return_compute_limit': common,
        'replay_output_comparison_upper': len(methods) * len(rows),
        'total_new_compute_upper': sum(item['required_compute'] for item in quotes.values())
            + len(methods) * len(rows) + common,
        'training_rows': split, 'working_rows': len(rows) - split, 'comparison_rows': len(rows),
        'source_pool_size': len(entries), 'max_candidates': 4, 'common_comparison_upper': 0}
    allocation_ref = supplied.get('allocation_ref')
    if allocation_ref is None:
        return {'status': 'parent_method_pair_quote_only', **quote, 'new_compute_cost': 0,
            'source_frontier_ref': ref(source / 'retained_frontier.json'), 'new_observations': 0}
    allocation = read_method_reference(allocation_ref, request.parent)
    if (allocation['source_frontier_ref'] != ref(source / 'retained_frontier.json')
            or allocation['new_observation_limit'] != 0):
        raise ValueError('The separate allocation must bind the complete historical frontier and no new observations')
    if quote['total_new_compute_upper'] > allocation['maximum_compute']:
        return {'status': 'complete_paired_unit_unfunded_not_method_negative', **quote,
            'remaining_compute': allocation['maximum_compute'], 'new_compute_cost': 0}
    prior_ledger_ref = supplied['prior_ledger_ref']
    prior_ledger = read_method_reference(prior_ledger_ref, request.parent)

    def same_paid_source(location):
        saved = load(Path(location['file']).parent / 'registration.json')
        if (saved['paid_records_ref'] != original['paid_records_ref']
                or saved['training_rows'] != split or saved['comparison_rows'] != len(rows)
                or saved['probes'] != probes or saved.get('appended_observation_refs')):
            raise ValueError('Reuse only fits and scores from the same complete paid view')

    paid_fit_refs = supplied.get('paid_fit_refs', {}).get('Pi3', [])
    for item in paid_fit_refs:
        same_paid_source(item['entry_ref'])
    shared_scores = []
    for location in supplied.get('shared_paid_comparison_refs', []):
        same_paid_source(location)
        compared = read_method_reference(location, source)
        if not compared['full_comparison']:
            raise ValueError('An incomplete comparison is not a shared paid score')
        shared_scores.extend({**item, 'score_ref': {**location,
            'json_path': [*location.get('json_path', []), 'new_scores', index]}}
            for index, item in enumerate(compared['new_scores']))
    raw = read_method_reference(original['paid_records_ref'], source)
    by_probe = OrderedDict()
    for record in raw:
        by_probe.setdefault(record['probe_id'], []).append(record)
    views = []
    for probe in probes:
        group = sorted(by_probe[probe['probe_id']], key=lambda record: record['position'])
        values = rows[probe['start']:probe['start'] + probe['rows']]
        views.append({'probe': {'probe_id': probe['probe_id'], 'sequence': [item['controls'] for item in group]},
            'observations': [asdict(row) for row in values],
            'source_records_ref': original['paid_records_ref']})
    grouped = OrderedDict()
    for (location, measured), entry in zip(references, entries):
        record = grouped.setdefault(entry['candidate'].program, {'mse': measured, 'candidate_refs': []})
        record['candidate_refs'].append(location)
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'paid_observations.json', views)
    data_refs = [ref(output / 'paid_observations.json', index) for index in range(len(views))]
    training_refs = [r for r, probe in zip(data_refs, probes) if probe['start'] < split]
    risk_quantile = read_method_reference(original['public_spec_ref'], source).get('risk_quantile', .5)
    paid_fit_inputs = {'Pi1': {'initial': {'entry_ref': frontier['selected_ref'],
        'training_refs': training_refs, 'risk_quantile': risk_quantile,
        'cost_ref': ref(source / 'relation/receipt.json')}},
        'Pi3': {'fits': [{**item, 'training_refs': training_refs, 'risk_quantile': risk_quantile}
                         for item in paid_fit_refs]}}
    save(output / 'source_pool.json', {'full_comparison': True, 'scores': list(grouped.values()),
        'rows': len(rows), 'source_frontier_ref': ref(source / 'retained_frontier.json'),
        'old_score_refs': [frontier['original_pool_ref'], *frontier['appended_comparison_refs']],
        'candidate_occurrences': len(entries), 'compute_cost': 0, 'new_predictions': 0})
    save(output / 'frozen_methods.json', methods)
    save(output / 'public_spec.json', read_method_reference(original['public_spec_ref'], source))
    artifact_refs = {}
    for arm in methods:
        artifact_refs[arm] = ref(output / (arm + '_method.json'))
        save(Path(artifact_refs[arm]['file']), {'id': f'native:parent_partition:{output.name}:{arm}',
            'version': 1, 'kind': 'native_research_segment', 'execution': {
                'entry': 'replay_research_program',
                'selection': 'UnifiedResearchAgent.compare_research_decision_return',
                'research_program_ref': ref(output / 'frozen_methods.json', arm)},
            'source_ref': baseline_ref if arm == 'Pi1' else challenger_ref,
            'parent_ref': frontier['selected_ref'], 'source_support_inherited': False})
    task_ref = load(source / 'choice_return/receipt.json')['source_task_ref']
    cfg = {**quote, 'source': str(source), 'source_task_ref': task_ref,
        'source_frontier_ref': ref(source / 'retained_frontier.json'), 'parent_ref': frontier['selected_ref'],
        'source_pool_ref': ref(output / 'source_pool.json'),
        'source_model_ref': supplied.get('source_model_ref', ref(Path(original['paid_records_ref']['file']).parent / 'model.json')),
        'source_cost_refs': [ref(source / 'source_costs.json'), prior_ledger_ref, allocation_ref],
        'prior_method_review': supplied['prior_method_review'], 'retained_method_pair_ref': None,
        'native_method_artifact_refs': artifact_refs,
        'paid_fit_inputs': paid_fit_inputs, 'shared_paid_scores': shared_scores,
        'evaluation_arm_names': {'baseline': 'Pi1', 'challenger': 'Pi3'},
        'baseline_source_ref': baseline_ref, 'challenger_source_ref': challenger_ref,
        'previous_compute': 0, 'remaining_compute': allocation['maximum_compute'],
        'allocation_ref': allocation_ref, 'allocation_id': allocation['allocation_id'],
        'original_ledger': {'receipt_ref': prior_ledger_ref, 'remaining_compute': prior_ledger['remaining_compute']},
        'ledger_scope': 'separately_registered_method_pair', 'feature_mappings': [{}],
        'data_refs': data_refs,
        'training_refs': training_refs,
        'feedback_refs': [r for r, probe in zip(data_refs, probes) if probe['start'] >= split],
        'probes': probes, 'quality_threshold': original['quality_threshold'],
        'research_config': {'max_candidates': 4, 'minimum_improvement': original['quality_threshold']},
        'active_wall_seconds': supplied.get('active_wall_seconds', relation.get('active_wall_seconds', 1800)),
        'evaluation_role': 'paid_working_comparison', 'source_identity': 'original_historical_source',
        'scope': 'historical_pre_call_pool_shared_paid_computation_distinct_frozen_methods',
        'execution_identity': 'new_development_pair_with_paid_reuse_not_historical_first_execution',
        'protected_frontier_ref': supplied['protected_frontier_ref'],
        'stop_rule': 'One complete paired research unit; no new observations, confirmation or body revision',
        'request_ref': ref(request), 'independent_support': False, 'new_observations': 0}
    save(output / 'registration.json', cfg)
    freeze_entry(output, source)
    relative = Path('scripts') / Path(__file__).name
    write_text_incrementally(output / 'runtime' / relative, (ROOT / relative).read_text(),
        base=source / 'runtime' / relative)
    return cfg


def run_parent(output):
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.object_continuation import encode_entry
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    from hegel_machine.cognition.experiment_runtime import python_execution_deadline

    output = Path(output)
    if (output / 'receipt.json').exists():
        return {**load(output / 'receipt.json'), 'new_compute_cost': 0, 'reused': True}
    if ROOT != (output / 'runtime').resolve() or (output / 'started.json').exists():
        raise ValueError('Execute the newly frozen parent continuation once')
    cfg = load(output / 'registration.json')
    frontier, _, rows, probes, references, entries, parent, scores = parent_inputs(cfg['source'])
    if probes != cfg['probes'] or frontier['selected_ref'] != cfg['parent_ref']:
        raise ValueError('The frozen parent and paid evidence changed')
    spec = TaskSpec(**read_method_reference(cfg['public_spec_ref'], output))
    agent = UnifiedResearchAgent(ResearchConfig(max_candidates=4, minimum_improvement=cfg['quality_threshold']))
    budget, timing, actions = SearchBudget(cfg['total_compute_upper']), ExecutionTiming(('parent_revision',)), []
    save(output / 'started.json', {'registration_ref': ref(output / 'registration.json')})
    try:
        parameter_inputs = {}
        if cfg['research_action'] == 'relation_parameter_revision':
            parameter_inputs = {'research_method': read_method_reference(cfg['research_method_ref'], output),
                'feedback_rows': rows[cfg['training_rows']:],
                'paid_initial': {'entry': parent, 'training': rows[:cfg['training_rows']],
                    'risk_quantile': spec.risk_quantile, 'cost_ref': cfg['paid_initializer_cost_ref']}}
        with timing.phase('parent_revision'), python_execution_deadline(cfg['active_wall_seconds'], activity=True):
            result = agent._revise_paid_structure(spec, parent, rows, budget, actions,
                finalists=entries, paid_scores=scores, fit_rows=rows[:cfg['training_rows']],
                research_action=cfg['research_action'], **parameter_inputs)
        new = result['new_claims']
        save(output / 'new_entries.json', [encode_entry(item) for item in new])
        checked = {item['candidate'].program: item for item in result['compared_claims']}
        new_scores = [{'candidate_ref': ref(output / 'new_entries.json', index),
            'loss': checked[item['candidate'].program]['validation_loss']}
            for index, item in enumerate(new) if item['candidate'].program in checked]
        selected = result['selected']
        selected_ref = (cfg['parent_ref'] if selected['candidate'].program == parent['candidate'].program else
            next((location for item, (location, _) in zip(entries, references)
                  if item['candidate'].program == selected['candidate'].program), None))
        if selected_ref is None:
            selected_ref = ref(output / 'new_entries.json', next(index for index, item in enumerate(new)
                if item['candidate'].program == selected['candidate'].program))
        comparison = {'full_comparison': result['full_comparison'], 'old_frontier_ref': cfg['frontier_ref'],
            'old_candidate_occurrences': len(entries), 'new_scores': new_scores,
            'selected_ref': selected_ref, 'selected_loss': selected['validation_loss'],
            'rows': len(rows), **({'data_ref': cfg['paid_records_ref']} if 'paid_records_ref' in cfg
                else {'data_refs': cfg['data_refs']})}
        save(output / 'comparison.json', comparison)
        save(output / 'diagnostics.json', {'actions': actions, 'native_action': result['action'],
            'selected_predictions': result.get('selected_predictions'),
            'budget_denied': bool(budget.denied_requests), 'last_denied_request': budget.last_denied_request})
    except BaseException as error:
        save(output / 'failure.json', {'error': repr(error), 'compute_cost': budget.spent,
            'remaining_compute': cfg['available_compute'] - budget.spent, 'actions': actions,
            'source_cost_refs': cfg['source_cost_refs'], 'execution_timing': timing.result()})
        raise
    receipt = {'status': result['status'], 'execution_complete': bool(result.get('execution_complete', False)),
        'invocation_complete': True, 'full_comparison': result['full_comparison'],
        'old_candidate_occurrences': len(entries), 'new_candidates': len(new),
        'selected_ref': selected_ref, 'selected_loss': selected['validation_loss'],
        'prior_selected_loss': cfg['prior_selected_loss'], 'parent_ref': cfg['parent_ref'],
        'comparison_ref': ref(output / 'comparison.json'), 'diagnostics_ref': ref(output / 'diagnostics.json'),
        'compute_cost': budget.spent, 'new_compute_cost': budget.spent,
        'remaining_compute': cfg['available_compute'] - budget.spent,
        'source_cost_refs': cfg['source_cost_refs'], 'source_task_ref': cfg['source_task_ref'],
        'research_action': cfg['research_action'], 'scope': cfg['scope'],
        'new_observations': 0, 'model_calls': 0, 'independent_support': False,
        'execution_timing': timing.result()}
    if cfg['research_action'] == 'relation_parameter_revision':
        receipt.update(method_invoked=bool(result['action'].get('research')),
            executed_method_ref=cfg['research_method_ref'],
            method_execution_ref=ref(output / 'diagnostics.json', 'native_action'),
            research_parent_ref=cfg['parent_ref'])
    save(output / 'receipt.json', receipt)
    return receipt


def interval_inputs(source, confirmation_source=None):
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.object_continuation import observations

    source = Path(source)
    frontier, original, old, probes, references, entries, parent, scores = parent_inputs(source)
    confirmation = Path(confirmation_source) if confirmation_source else source / 'frozen_confirmation'
    if confirmation_source:
        cfg, actual = load(confirmation / 'registration.json'), load(confirmation / 'receipt.json')
        if (Path(cfg['source']).resolve() != source.resolve()
                or cfg['source_selected_ref'] != frontier['selected_ref']
                or actual['selected_ref'] != frontier['selected_ref']
                or actual['status'] not in {'confirmed', 'confirmation_failed'}):
            raise ValueError('Use the completed frozen confirmation of this exact retained parent')
    location = ref(confirmation / 'confirmation/observations.json')
    document = read_method_reference(location, source)
    fresh = observations(document['batch']['observations'])
    result = load(confirmation / 'confirmation/receipt.json')
    frozen = load(confirmation / 'confirmation/frozen_predictors.json')
    paid_suffix = {}
    for arm, assessment in result['arms'].items():
        program = Program.from_dict(frozen[arm]['program'])
        if len(assessment['predictions']) != len(fresh):
            raise ValueError('A paid suffix score must cover its complete saved observation batch')
        if program in scores:
            paid_suffix[program] = {'mse': assessment['mse'],
                'score_ref': ref(confirmation / 'confirmation/receipt.json', 'arms', arm)}
    if parent['candidate'].program not in paid_suffix:
        raise ValueError('The current parent must retain its actual frozen feedback predictions')
    return {'frontier': frontier, 'original': original, 'old': old, 'probes': probes,
        'references': references, 'entries': entries, 'parent': parent, 'scores': scores,
        'suffix': fresh, 'suffix_ref': location, 'suffix_probe': document['probe']['probe_id'],
        'paid_suffix': paid_suffix}


def register_interval(source, output, *, request=None):
    """Quote complete appended-data comparison before the existing interval stage."""
    source, output = Path(source), Path(output)
    supplied = load(request) if request else None
    confirmation_source = supplied['confirmation_source'] if supplied else None
    scene = interval_inputs(source, confirmation_source)
    previous = load(source / 'registration.json')
    parent = scene['parent']
    second = parent['research_program']['transfer_partition'].get('second_stage')
    if not second or second['operation'] != 'resolve_relation_intervals':
        raise ValueError('Continue the current parent\'s already declared interval stage')
    split = previous['training_rows']
    if (parent['training_loss'] > previous['quality_threshold']
            or parent['research_program']['fit_rows'] != split):
        raise ValueError('The original fitted training prefix must remain supported')
    prior_review = (supplied['prior_method_review'] if supplied
        else str(source / 'confirmation_feedback/method_return'))
    ledger = (supplied['ledger_ref'] if supplied
        else ref(Path(prior_review) / 'receipt.json', 'remaining_compute'))
    balance = read_method_reference(ledger, source)
    missing = len(scene['scores']) - len(scene['paid_suffix'])
    comparison = missing * len(scene['suffix'])
    interval = min(15000, second['compute_limit'])
    if comparison + interval + 3000 > balance:
        raise ValueError('The unchanged account cannot fund complete suffix consumption and interval return')
    original = scene['original']
    appended = [*original.get('appended_observation_refs', []), scene['suffix_ref']]
    probes = [*scene['probes'], {'probe_id': scene['suffix_probe'],
        'start': len(scene['old']), 'rows': len(scene['suffix'])}]
    cfg = {'source': str(source), 'frontier_ref': ref(source / 'retained_frontier.json'),
        'parent_ref': scene['frontier']['selected_ref'], 'research_parent_ref': scene['frontier']['selected_ref'],
        'source_task_ref': previous['source_task_ref'],
        **{key: original[key] for key in ('paid_records_ref', 'data_refs') if key in original},
        'public_spec_ref': original['public_spec_ref'], 'appended_observation_refs': appended,
        'training_rows': split, 'feedback_rows': len(scene['old']) + len(scene['suffix']) - split,
        'comparison_rows': len(scene['old']) + len(scene['suffix']), 'probes': probes,
        'old_rows': len(scene['old']), 'suffix_rows': len(scene['suffix']),
        'old_candidate_occurrences': len(scene['entries']), 'old_unique_programs': len(scene['scores']),
        'paid_suffix_programs': len(scene['paid_suffix']), 'new_suffix_programs': missing,
        'comparison_compute_upper': comparison, 'interval_compute_limit': interval,
        'total_compute_upper': comparison + interval, 'method_return_compute_limit': 3000,
        'ledger_ref': ledger, 'available_compute': balance,
        'source_cost_refs': (supplied['source_cost_refs'] if supplied else
            [ref(source / 'final_source_costs.json'), ref(Path(prior_review) / 'receipt.json')]),
        'prior_method_review': prior_review,
        'quality_threshold': previous['quality_threshold'], 'max_new_exports': 3,
        'active_wall_seconds': previous['active_wall_seconds'],
        'research_action': 'paid_interval_continuation',
        'scope': 'exposed_confirmation_consumed_as_adaptive_development',
        'stop_rule': 'One complete expanded-pool comparison and at most one declared interval stage; no fit or acquisition',
        'new_fits': 0, 'new_observations': 0, 'independent_support': False}
    if supplied:
        cfg.update(confirmation_source=str(confirmation_source), request_ref=ref(request))
    parent_ref = cfg['research_parent_ref']
    cfg['executed_method_ref'] = {**parent_ref,
        'json_path': [*parent_ref.get('json_path', []), 'research_program']}
    cfg['research_method_ref'] = cfg['executed_method_ref']
    cfg['method_execution_ref'] = ref(output / 'interval_result.json')
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'registration.json', cfg)
    freeze_entry(output, source)
    for name in (Path(__file__).name, 'continue_successor_relation_intervals.py'):
        relative = Path('scripts') / name
        write_text_incrementally(output / 'runtime' / relative, (ROOT / relative).read_text(),
            base=source / 'runtime' / relative)
    return cfg


def run_interval(output):
    from scripts.continue_successor_relation_intervals import extend_scores
    from hegel_machine.cognition.evaluation import initial_prediction
    from hegel_machine.cognition.object_continuation import encode_entry
    from hegel_machine.cognition.transfer_partitions import continue_transfer_partition
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.experiment_runtime import python_execution_deadline

    output = Path(output)
    if (output / 'receipt.json').exists():
        return {**load(output / 'receipt.json'), 'new_compute_cost': 0, 'reused': True}
    if ROOT != (output / 'runtime').resolve() or (output / 'started.json').exists():
        raise ValueError('Execute the newly frozen expanded-pool interval unit once')
    cfg = load(output / 'registration.json')
    scene = interval_inputs(cfg['source'], cfg.get('confirmation_source'))
    rows = [*scene['old'], *scene['suffix']]
    if len(rows) != cfg['comparison_rows'] or scene['frontier']['selected_ref'] != cfg['parent_ref']:
        raise ValueError('Keep the frozen current parent and complete appended paid evidence')
    old_scores = scene['scores']
    scores = {program: value for program, value in old_scores.items() if program not in scene['paid_suffix']}
    fallback = scene['parent'].get('fallback', initial_prediction(scene['old'], .5))
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=cfg['quality_threshold']))
    budget, timing = SearchBudget(cfg['total_compute_upper']), ExecutionTiming(('compare_suffix', 'interval'))
    save(output / 'started.json', {'registration_ref': ref(output / 'registration.json')})
    try:
        with python_execution_deadline(cfg['active_wall_seconds'], activity=True):
            with timing.phase('compare_suffix'):
                extend_scores(scores, len(scene['old']), scene['suffix'], budget, fallback)
                for program, paid in scene['paid_suffix'].items():
                    scores[program] = (len(scene['old']) * old_scores[program]
                        + len(scene['suffix']) * paid['mse']) / len(rows)
            entries = [dict(item, validation_loss=scores[item['candidate'].program]) for item in scene['entries']]
            before = agent._best(entries)
            before_ref = next(location for item, (location, _) in zip(entries, scene['references'])
                if item['candidate'].program == before['candidate'].program)
            comparison_compute = budget.spent
            save(output / 'expanded_pool_comparison.json', {'full_comparison': True,
                'rows': len(rows), 'old_frontier_ref': cfg['frontier_ref'],
                'selected_ref': before_ref, 'selected_loss': before['validation_loss'],
                'compute_cost': comparison_compute, 'scores': [
                    {'candidate_ref': location, 'loss': item['validation_loss']}
                    for item, (location, _) in zip(entries, scene['references'])],
                'paid_suffix_score_refs': [value['score_ref'] for value in scene['paid_suffix'].values()]})
            stage = {'status': 'complete_pool_already_explains', 'execution_complete': True,
                'candidates': [], 'new_fits': 0, 'new_observations': 0}
            if before['validation_loss'] > cfg['quality_threshold']:
                split = cfg['training_rows']
                feedback = rows[split:]
                groups = [rows[item['start']:item['start'] + item['rows']]
                    for item in cfg['probes'] if item['start'] >= split]
                parent = scene['parent']
                feedback_loss = (len(rows) * scores[parent['candidate'].program]
                    - split * parent['training_loss']) / len(feedback)
                local = SearchBudget(cfg['interval_compute_limit'])
                old_predictions = load(Path(cfg['source']) / 'diagnostics.json')['selected_predictions']
                returned = read_method_reference(scene['paid_suffix'][parent['candidate'].program]['score_ref'], output)
                if len(old_predictions) != len(scene['old']):
                    raise ValueError('Keep the actual current parent\'s complete paid predictions')
                feedback_errors = [(value - row.target) ** 2 for value, row in
                    zip([*old_predictions[split:], *returned['predictions']], feedback)]
                try:
                    with timing.phase('interval'):
                        stage = continue_transfer_partition(parent, rows[:split], feedback, local,
                            tolerance=cfg['quality_threshold'], feedback_loss=feedback_loss,
                            feedback_experiments=groups, feedback_squared_errors=feedback_errors,
                            fit_source_ref=cfg['research_parent_ref'])
                finally:
                    budget.spend(local.spent)
            assessments = []
            for item in stage.get('candidates', []):
                assessments.append({'program': item['candidate'].program.to_dict(),
                    **{key: item.get(key) for key in ('training_loss', 'validation_loss', 'all_paid_loss',
                        'parameter_values', 'radius_interval', 'paid_row_squared_errors', 'evaluation_rows')}})
            save(output / 'interval_assessments.json', {'assessments': assessments,
                'partial_evaluations': stage.get('partial_evaluations', []), 'rows': len(rows)})
            save(output / 'interval_result.json', {key: value for key, value in stage.items()
                if key not in {'selected', 'candidates', 'partial_evaluations'}})
            exports, seen = [], set(old_scores)
            if stage['execution_complete']:
                proposed = sorted(stage.get('candidates', []), key=lambda value:
                    (value['all_paid_loss'], value['candidate'].program.size))
                if stage.get('selected') is not None:
                    proposed.insert(0, stage['selected'])
                for item in proposed:
                    program = item['candidate'].program
                    if program in seen:
                        continue
                    exports.append(item)
                    seen.add(program)
                    if len(exports) == cfg['max_new_exports']:
                        break
            save(output / 'new_entries.json', [encode_entry(item) for item in exports])
            references = [location for location, _ in scene['references']]
            for index, item in enumerate(exports):
                scores[item['candidate'].program] = item['all_paid_loss']
                entries.append(dict(item, validation_loss=item['all_paid_loss']))
                references.append(ref(output / 'new_entries.json', index))
            selected = agent._best(entries)
            if before['validation_loss'] <= selected['validation_loss'] + cfg['quality_threshold']:
                selected = before
            selected_ref = next(location for item, location in zip(entries, references)
                if item['candidate'].program == selected['candidate'].program)
            groups = OrderedDict()
            for entry, location in zip(entries, references):
                groups.setdefault(entry['candidate'].program, []).append(location)
            save(output / 'paid_pool/registration.json', {key: cfg[key] for key in
                ('paid_records_ref', 'data_refs', 'public_spec_ref', 'appended_observation_refs', 'quality_threshold',
                 'training_rows', 'feedback_rows', 'comparison_rows', 'probes') if key in cfg})
            save(output / 'paid_pool/receipt.json', {'full_comparison': True, 'rows': len(rows),
                'scores': [{'candidate_refs': locations, 'mse': scores[program], 'rows': len(rows)}
                    for program, locations in groups.items()],
                'old_frontier_ref': cfg['frontier_ref'], 'selected_ref': selected_ref,
                'selected_mse': selected['validation_loss'], 'candidate_occurrences': len(entries),
                'unique_programs': len(groups), 'score_scope': 'all_original_and_appended_paid_rows'})
            save(output / 'retained_frontier.json', {'full_comparison': True,
                'original_pool_ref': ref(output / 'paid_pool/receipt.json'), 'appended_comparison_refs': [],
                'complete_current_pool_occurrences': len(entries), 'selected_ref': selected_ref,
                'selected_mse': selected['validation_loss'], 'comparison_rows': len(rows),
                'parent_frontier_ref': cfg['frontier_ref'], 'independent_support': False})
    except BaseException as error:
        save(output / 'failure.json', {'error': repr(error), 'compute_cost': budget.spent,
            'remaining_compute': cfg['available_compute'] - budget.spent,
            'source_cost_refs': cfg['source_cost_refs'], 'execution_timing': timing.result()})
        raise
    result = {'status': ('paid_interval_supported' if before['validation_loss'] > cfg['quality_threshold']
        and selected['validation_loss'] <= cfg['quality_threshold'] else stage['status']),
        'execution_complete': bool(stage['execution_complete']),
        'invocation_complete': True, 'full_comparison': True, 'old_candidate_occurrences': len(scene['entries']),
        'new_candidates': len(exports), 'actual_interval_assessments': len(assessments),
        'selected_ref': selected_ref, 'selected_loss': selected['validation_loss'],
        'parent_ref': before_ref, 'research_parent_ref': cfg['research_parent_ref'],
        'prior_selected_loss': before['validation_loss'], 'comparison_rows': len(rows),
        'comparison_ref': ref(output / 'paid_pool/receipt.json'),
        'before_frontier_ref': ref(output / 'expanded_pool_comparison.json'),
        'method_invoked': before['validation_loss'] > cfg['quality_threshold'],
        'executed_method_ref': cfg['executed_method_ref'], 'method_execution_ref': cfg['method_execution_ref'],
        'compute_cost': budget.spent, 'new_compute_cost': budget.spent,
        'comparison_compute': comparison_compute, 'remaining_compute': cfg['available_compute'] - budget.spent,
        'source_task_ref': cfg['source_task_ref'], 'source_cost_refs': cfg['source_cost_refs'],
        'new_fits': 0, 'new_observations': 0, 'model_calls': 0, 'independent_support': False,
        'execution_timing': timing.result()}
    save(output / 'receipt.json', result)
    return result


def register_interval_probe_return(source, output, *, request=None):
    """Continue parameter uncertainty after a measured competing-law rejection."""
    source, output = Path(source), Path(output)
    previous, actual = load(source / 'registration.json'), load(source / 'receipt.json')
    frontier, original, rows, probes, references, pool, parent, scores = parent_inputs(source)
    if actual['status'] != 'response_distinguished' or not actual['full_comparison']:
        raise ValueError('Use a completed distinguishing experiment and its full paid pool')
    split = original['training_rows']
    supplied = load(request) if request else {}
    partition_ref, partition_score_ref = frontier['selected_ref'], None
    donor = parent
    score_sources = [(location, entry) for (location, _), entry in zip(references, pool)
        if location == partition_ref]
    covered = parent['research_program'].get('evaluation_rows', parent['research_program']['fit_rows'])
    if supplied.get('partition_source'):
        prior = Path(supplied['partition_source'])
        old_frontier, _, old_rows, _, old_refs, old_pool, donor, old_scores = parent_inputs(prior)
        covered = donor['research_program'].get('evaluation_rows', donor['research_program']['fit_rows'])
        if covered != len(old_rows) or old_rows != rows[:covered]:
            raise ValueError('Reuse only the actually completed partition on this same paid prefix')
        old_step = donor['research_program']['selection_steps'][-1]
        parent_step = parent['research_program']['selection_steps'][-1]
        if old_step['parameter_template'] != parent_step['parameter_template']:
            raise ValueError('A different complete response cannot inherit this partition')
        if old_scores[parent['candidate'].program] > previous['quality_threshold']:
            raise ValueError('The retained parameter must explain the completed partition evidence')
        partition_ref = old_frontier['selected_ref']
        paid_pool = read_method_reference(old_frontier['original_pool_ref'], prior)
        current_locations = [location for (location, _), entry in zip(old_refs, old_pool)
            if entry['candidate'].program == parent['candidate'].program]
        score_index = next(index for index, score in enumerate(paid_pool['scores'])
            if any(location in score['candidate_refs'] for location in current_locations))
        partition_score_ref = {**old_frontier['original_pool_ref'],
            'json_path': [*old_frontier['original_pool_ref'].get('json_path', []), 'scores', score_index, 'mse']}
        completed_refs = load(prior / 'receipt.json').get('candidate_refs', [])
        score_sources = [(location, entry) for (location, _), entry in zip(old_refs, old_pool)
            if location == partition_ref or location in completed_refs]
    ends = {part['start'] + part['rows'] for part in probes}
    if covered not in ends or not split <= covered < len(rows):
        raise ValueError('The retained completed partition must precede genuinely new complete probes')
    # A pool score alone does not identify the row errors. Reuse only a real
    # vector exported by this completed same-prefix partition, in the same
    # parameter cell and with the fixed complete response unchanged.
    from hegel_machine.cognition.structural_research_program import research_program_fit_contract
    step = donor['research_program']['selection_steps'][-1]
    index = donor['parameter_index']
    symbol = step['parameters'][index]['symbol']
    value = donor['parameter_values'][symbol]
    cell = next((domain for domain in step['parameters'][index]['radius_intervals']
        if domain[0] < value < domain[1]), None)
    donor_fit = research_program_fit_contract(donor['research_program'])
    paid_cell_score_ref = None
    for location, entry in score_sources:
        body = entry.get('research_program') or {}
        stored = next((item for item in reversed(body.get('selection_steps', []))
            if item['operation'] == 'resolve_relation_intervals'), None)
        values = entry.get('parameter_values', {})
        if (cell is None or stored is None or body.get('evaluation_rows', body.get('fit_rows')) != covered
                or len(entry.get('paid_row_squared_errors', [])) != covered
                or entry.get('selection_basis') != 'all_acquired_evidence'
                or entry.get('parameter_index') != index
                or stored['parameter_template'] != step['parameter_template']
                or stored.get('parameter_terms') != step.get('parameter_terms')
                or stored['parameters'][index]['radius_intervals'] != step['parameters'][index]['radius_intervals']
                or set(values) != set(donor['parameter_values'])
                or not cell[0] < values[symbol] < cell[1]
                or any(values[key] != val for key, val in donor['parameter_values'].items() if key != symbol)):
            continue
        fit = research_program_fit_contract(body)
        if fit['coefficients'] == donor_fit['coefficients']:
            paid_cell_score_ref = location
            break
    available = actual['remaining_compute']
    reserve = supplied.get('reserved_compute', 3096)
    if reserve < 96 or available <= reserve:
        raise ValueError('Retain the original return and confirmation reserves')
    cfg = {'source': str(source), 'frontier_ref': ref(source / 'retained_frontier.json'),
        'parent_ref': frontier['selected_ref'], 'research_parent_ref': frontier['selected_ref'],
        'source_task_ref': previous['source_task_ref'],
        'object_model_ref': previous['object_model_ref'], 'prior_method_review': previous['prior_method_review'],
        'binding_ref': actual.get('instrument_binding_ref', ref(source / 'instrument_binding.json')),
        'instrument_models_ref': actual.get('instrument_models_ref', ref(source / 'instrument_models.json')),
        'experiment_methods_ref': actual.get('experiment_methods_ref', ref(source / 'program_experiment_methods.json')),
        'response_evidence_ref': actual.get('response_evidence_ref', ref(source / 'response_assessment.json')),
        'ledger_ref': ref(source / 'receipt.json', 'remaining_compute'),
        'source_cost_refs': [previous['source_costs_ref'], ref(source / 'receipt.json')],
        'available_compute': available, 'compute_limit': available - reserve, 'reserved_compute': reserve,
        'partition_ref': partition_ref, 'partition_parent_score_ref': partition_score_ref,
        'paid_cell_score_ref': paid_cell_score_ref,
        **({'allocation_ref': ref(request)} if request else {}),
        'training_rows': split, 'partition_rows': covered, 'comparison_rows': len(rows),
        'quality_threshold': previous['quality_threshold'], 'active_wall_seconds': previous['active_wall_seconds'],
        'old_candidate_occurrences': len(pool), 'old_unique_programs': len(scores),
        'research_action': 'paid_interval_uncertainty_update',
        'scope': 'same_source_parameter_uncertainty_after_response_discrimination',
        'stop_rule': 'One existing interval pass; preserve the fitted response and old pool; no acquisition or refit'}
    cfg['executed_method_ref'] = {**cfg['parent_ref'],
        'json_path': [*cfg['parent_ref'].get('json_path', []), 'research_program']}
    cfg['method_execution_ref'] = ref(output / 'interval_result.json')
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'registration.json', cfg)
    freeze_entry(output, source)
    for name in (Path(__file__).name, 'continue_successor_relation_intervals.py'):
        relative = Path('scripts') / name
        write_text_incrementally(output / 'runtime' / relative, (ROOT / relative).read_text(),
            base=source / 'runtime' / relative)
    return cfg


def run_interval_probe_return(output):
    from scripts.continue_successor_relation_intervals import (
        compare_pool, save_interval_candidates, retain_interval_evidence)
    from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
    from hegel_machine.cognition.transfer_partitions import finalize_interval_candidates
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.experiment_runtime import python_execution_deadline

    output = Path(output)
    if (output / 'receipt.json').exists():
        return {**load(output / 'receipt.json'), 'new_compute_cost': 0, 'reused': True}
    if ROOT != (output / 'runtime').resolve() or (output / 'started.json').exists():
        raise ValueError('Execute the freshly frozen uncertainty update once')
    cfg = load(output / 'registration.json')
    frontier, original, rows, probes, references, pool, parent, scores = parent_inputs(cfg['source'])
    budget, timing = SearchBudget(cfg['compute_limit']), ExecutionTiming(('interval',))
    save(output / 'started.json', {'registration_ref': ref(output / 'registration.json')})
    try:
        with python_execution_deadline(cfg['active_wall_seconds'], activity=True), timing.phase('interval'):
            donor = retained_entry(cfg.get('partition_ref', cfg['parent_ref']), output, 0)
            research = RelationIntervalResearch(donor['candidate'].program, [], budget,
                structure=donor.get('structure'), research_program=donor['research_program'])
            if not research.reuse_paid_partition(rows[:cfg['partition_rows']], donor):
                raise ValueError('The saved partition does not match this paid prefix and fitted parent')
            paid_scores_reused = False
            if cfg.get('paid_cell_score_ref'):
                paid_scores_reused = research.reuse_paid_cell_scores(rows[:cfg['partition_rows']],
                    retained_entry(cfg['paid_cell_score_ref'], output, 0))
            if donor['candidate'].program != parent['candidate'].program:
                retained = retain_interval_evidence(research, dict(parent, validation_loss=
                    read_method_reference(cfg['partition_parent_score_ref'], output)), [],
                    cfg['partition_rows'], cfg['quality_threshold'])
                if retained is None or retained['candidate'].program != parent['candidate'].program:
                    raise ValueError('The retained answer is outside this supported response family')
                research.adopt(retained)
            before_intervals = research.intervals
            split = cfg['training_rows']
            experiments = [rows[part['start']:part['start'] + part['rows']]
                for part in probes if part['start'] >= split]
            candidates = research.alternatives(rows, budget, cfg['quality_threshold'], experiments=experiments)
            complete = research.status != 'compute_exhausted'
            exports = finalize_interval_candidates(parent, candidates, research, rows[:split], rows[split:],
                feedback_experiments=experiments, fit_source_ref=cfg['parent_ref']) if complete else []
            for entry in exports:
                scores.setdefault(entry['candidate'].program, entry['all_paid_loss'])
            agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=cfg['quality_threshold']))
            selected = compare_pool(agent, [*pool, *exports], scores, parent.get('fallback', 0.))
            if scores[parent['candidate'].program] <= selected['validation_loss'] + cfg['quality_threshold']:
                selected = next((entry for entry in reversed(exports)
                    if entry['candidate'].program == parent['candidate'].program),
                    dict(parent, validation_loss=scores[parent['candidate'].program]))
            if complete:
                retained = retain_interval_evidence(research, selected, exports, len(rows), cfg['quality_threshold'])
                if retained is not None:
                    if not any(item['candidate'].program == retained['candidate'].program for item in exports):
                        exports.append(retained)
                    selected = retained
            saved, _ = save_interval_candidates(output, 0, exports,
                previous=Path(cfg['parent_ref']['file']))
            locations = [location for location, _ in references] + saved
            combined = [*pool, *exports]
            selected_ref = next(location for entry, location in reversed(list(zip(combined, locations)))
                if entry['candidate'].program == selected['candidate'].program)
            groups = OrderedDict()
            for entry, location in zip(combined, locations):
                groups.setdefault(entry['candidate'].program, []).append(location)
            save(output / 'paid_pool/registration.json', original)
            save(output / 'paid_pool/receipt.json', {'full_comparison': True, 'rows': len(rows),
                'scores': [{'candidate_refs': refs, 'mse': scores[program], 'rows': len(rows)}
                    for program, refs in groups.items()],
                'old_frontier_ref': cfg['frontier_ref'], 'selected_ref': selected_ref,
                'selected_mse': scores[selected['candidate'].program],
                'candidate_occurrences': len(combined), 'unique_programs': len(groups)})
            save(output / 'retained_frontier.json', {'full_comparison': True,
                'original_pool_ref': ref(output / 'paid_pool/receipt.json'), 'appended_comparison_refs': [],
                'complete_current_pool_occurrences': len(combined), 'selected_ref': selected_ref,
                'selected_mse': scores[selected['candidate'].program], 'comparison_rows': len(rows),
                'parent_frontier_ref': cfg['frontier_ref'], 'independent_support': False})
            save(output / 'interval_result.json', {'status': research.status, 'execution_complete': complete,
                'fixed_readout_only': True, 'new_fits': 0,
                'before_intervals': before_intervals, 'after_intervals': research.intervals,
                'reuse_statistics': research.reuse_statistics, 'candidate_refs': saved,
                'paid_cell_score_ref': cfg.get('paid_cell_score_ref'),
                'paid_cell_scores_reused': paid_scores_reused,
                'new_geometry_rows': len(rows) - cfg['partition_rows'],
                'parent_program_unchanged': selected['candidate'].program == parent['candidate'].program})
    except BaseException as error:
        save(output / 'failure.json', {'error': repr(error), 'compute_cost': budget.spent,
            'remaining_compute': cfg['available_compute'] - budget.spent,
            'ledger_ref': cfg['ledger_ref'], 'execution_timing': timing.result()})
        raise
    maximum_error_bound = len(rows) * scores[selected['candidate'].program]
    gate = {'admitted': complete and maximum_error_bound <= cfg['quality_threshold']
        and research.status == 'resolved_to_precision',
        'all_paid_squared_error_upper_bound': maximum_error_bound,
        'quality_threshold': cfg['quality_threshold'], 'interval_precision': 1e-6,
        'stop_reason': research.status, 'competition_review_needed': research.status == 'unresolved',
        'basis': 'Complete paid pool; nonnegative squared-error sum bounds every row; original interval stop rule'}
    result = {'status': 'paid_interval_updated' if complete and exports else 'interval_update_incomplete',
        'execution_complete': complete, 'full_comparison': True, 'method_invoked': True,
        'selected_ref': selected_ref, 'selected_loss': scores[selected['candidate'].program],
        'parent_ref': cfg['parent_ref'], 'research_parent_ref': cfg['parent_ref'],
        'parent_program_unchanged': selected['candidate'].program == parent['candidate'].program,
        'candidate_refs': saved, 'comparison_rows': len(rows), 'pool_occurrences': len(combined),
        'paid_cell_score_ref': cfg.get('paid_cell_score_ref'), 'paid_cell_scores_reused': paid_scores_reused,
        'confirmation_gate': gate,
        'old_candidate_occurrences': len(pool), 'new_candidates': len(exports),
        'executed_method_ref': cfg['executed_method_ref'], 'method_execution_ref': cfg['method_execution_ref'],
        'comparison_ref': ref(output / 'paid_pool/receipt.json'), 'source_task_ref': cfg['source_task_ref'],
        'compute_cost': budget.spent, 'new_compute_cost': budget.spent,
        'remaining_compute': cfg['available_compute'] - budget.spent,
        'ledger_ref': cfg['ledger_ref'], 'source_cost_refs': cfg['source_cost_refs'],
        'new_fits': 0, 'new_observations': 0, 'model_calls': 0, 'independent_support': False,
        'execution_timing': timing.result()}
    save(output / 'receipt.json', result)
    return result


def instrument_plan_inputs(source, object_model_ref):
    """Restore paid command/measurement pairs without inventing missing roles."""
    from copy import deepcopy
    from dataclasses import replace
    from scripts.successor_interval_scene import pair_rows
    from hegel_machine.cognition.object_continuation import observations

    source = Path(source)
    frontier, original, rows, _, _, _, parent, _ = parent_inputs(source)
    public = TaskSpec(**read_method_reference(original['public_spec_ref'], source))
    model = read_method_reference(object_model_ref, source)
    measured = replace(public, context={**public.context, 'object_model': deepcopy(model)})
    command = replace(public, feature_names=tuple(public.feature_bounds), context={})
    locations = [*original['data_refs'], *original.get('appended_observation_refs', [])]
    batches = []
    for location in locations:
        document = read_method_reference(location, source)
        batches.append(pair_rows(document['probe']['sequence'],
            observations(document.get('batch', document)['observations'])))
    if [row for batch in batches for _, row in batch] != rows or len(batches) < 2:
        raise ValueError('Retain complete paid trajectories for instrument training and validation')
    locations = [ref(source / 'new_entries.json', index)
        for index in range(len(load(source / 'new_entries.json')))]
    entries = [retained_entry(location, source, index) for index, location in enumerate(locations)]
    return frontier, original, command, measured, batches, entries, locations, parent


def register_instrument_plan(request, output):
    """Use the declared partial roles and paid pairs before planning new evidence."""
    request, output = Path(request), Path(output)
    supplied = load(request)
    source = Path(supplied['source'])
    actual, previous = load(source / 'receipt.json'), load(source / 'registration.json')
    if not actual['execution_complete'] or not actual['full_comparison']:
        raise ValueError('Complete the retained-pool interval research before planning')
    frontier, _, _, _, batches, entries, locations, _ = instrument_plan_inputs(
        source, supplied['object_model_ref'])
    if len(entries) < 2:
        raise ValueError('This entry needs the retained competing interval hypotheses')
    available = read_method_reference(supplied['ledger_ref'], source)
    # Keep the existing ordinary return and two-predictor confirmation reserve.
    reserve = 3000 + 96
    if available <= reserve:
        raise ValueError('The unchanged balance must retain return and confirmation reserves')
    cfg = {**supplied, 'request_ref': ref(request), 'source': str(source),
        'source_task_ref': previous['source_task_ref'],
        'frontier_ref': ref(source / 'retained_frontier.json'),
        'selected_ref': frontier['selected_ref'], 'candidate_refs': locations,
        'pool_occurrences': frontier['complete_current_pool_occurrences'],
        'available_compute': available, 'compute_limit': available - reserve,
        'reserved_compute': reserve, 'quality_threshold': previous['quality_threshold'],
        'active_wall_seconds': previous['active_wall_seconds'],
        'instrument_training_rows': sum(len(part) for part in batches[:-1]),
        'instrument_validation_rows': len(batches[-1]),
        'paired_trajectory_rows': [len(part) for part in batches],
        'new_world_fits': 0, 'new_observations': 0, 'model_calls': 0,
        'scope': 'paid_instrument_research_and_unpurchased_experiment_planning',
        'stop_rule': 'One paid calibration and one existing planner; preserve ambiguity; no acquisition or world refit'}
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'registration.json', cfg)
    freeze_entry(output, source)
    relative = Path('scripts') / Path(__file__).name
    write_text_incrementally(output / 'runtime' / relative, (ROOT / relative).read_text(),
        base=source / 'runtime' / relative)
    return cfg


def run_instrument_plan(output):
    from copy import deepcopy
    from dataclasses import asdict
    from hegel_machine.cognition.competing_object_experiments import (
        plan_competing_object_probe, shorten_discriminating_probe)
    from hegel_machine.cognition.instrument_hypotheses import plan_instrument_disagreement
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    from hegel_machine.cognition.experiment_runtime import python_execution_deadline

    output = Path(output)
    if (output / 'receipt.json').exists():
        return {**load(output / 'receipt.json'), 'new_compute_cost': 0, 'reused': True}
    if ROOT != (output / 'runtime').resolve() or (output / 'started.json').exists():
        raise ValueError('Execute the newly frozen instrument planning unit once')
    cfg = load(output / 'registration.json')
    frontier, _, command, measured, batches, entries, locations, parent = instrument_plan_inputs(
        cfg['source'], cfg['object_model_ref'])
    if frontier['selected_ref'] != cfg['selected_ref'] or locations != cfg['candidate_refs']:
        raise ValueError('Keep the frozen parent, competitors and complete retained pool')
    budget = SearchBudget(cfg['compute_limit'])
    timing = ExecutionTiming(('instruments', 'plan'))
    actions = []
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=cfg['quality_threshold']))
    agent._learn_instruments = True
    agent.library.instrument_models = deepcopy(read_method_reference(cfg['instrument_models_ref'], output))
    agent.library.program_experiment_methods.records = deepcopy(
        read_method_reference(cfg['experiment_methods_ref'], output))
    required = set().union(*(entry['candidate'].program.required_features() for entry in entries))
    save(output / 'started.json', {'registration_ref': ref(output / 'registration.json')})
    try:
        with python_execution_deadline(cfg['active_wall_seconds'], activity=True):
            with timing.phase('instruments'):
                model = agent._fit_instruments(command, measured, batches, budget, actions,
                    required_fields=required)
            calibration_compute = budget.spent
            save(output / 'instrument_models.json', agent.library.instrument_models)
            save(output / 'instrument_actions.json', actions)
            binding = agent._instrument_binding(measured, command)
            missing = required - set(binding) - set(command.feature_names)
            with timing.phase('plan'):
                if missing:
                    kind = 'instrument'
                    local = SearchBudget(min(6000, budget.remaining))
                    try:
                        plan = (plan_instrument_disagreement(model, command, measured, local,
                            max_length=5, serial=0, tolerance=cfg['quality_threshold'],
                            methods=agent.library.program_experiment_methods) if model else
                            {'probe': None, 'reason': 'instrument_mapping_unresolved'})
                    finally:
                        budget.spend(local.spent)
                    decision = ('instrument_distinguishing_experiment' if plan['probe'] is not None
                        else 'instrument_mapping_unresolved')
                else:
                    kind = 'response'
                    research = RelationIntervalResearch(parent['candidate'].program, [], budget,
                        research_program=parent['research_program'])
                    questions = research.questions(entries)
                    plan = plan_competing_object_probe(command, entries, budget, binding=binding,
                        serial=0, length=5, tolerance=cfg['quality_threshold'], questions=questions,
                        methods=agent.library.program_experiment_methods, task_id=measured.task_id)
                    plan = shorten_discriminating_probe(plan, budget, tolerance=cfg['quality_threshold'])
                    decision = ('response_distinguishing_experiment'
                        if plan['forecast_status'] == 'predicted_distinction' and plan['probe'] is not None
                        else 'fresh_confirmation_candidate' if plan['forecast_status'] == 'algebraically_shared'
                        else 'response_disagreement_unresolved')
            save(output / 'plan.json', {**plan,
                'probe': asdict(plan['probe']) if plan.get('probe') is not None else None,
                'planning_kind': kind, 'acquired': False})
            save(output / 'instrument_binding.json', {name: program.to_dict() for name, program in binding.items()})
            save(output / 'program_experiment_methods.json', agent.library.program_experiment_methods.records)
    except BaseException as error:
        save(output / 'instrument_models.json', agent.library.instrument_models)
        save(output / 'instrument_actions.json', actions)
        save(output / 'failure.json', {'error': repr(error), 'compute_cost': budget.spent,
            'remaining_compute': cfg['available_compute'] - budget.spent,
            'ledger_ref': cfg['ledger_ref'], 'execution_timing': timing.result()})
        raise
    result = {'status': decision, 'planning_kind': kind,
        'required_fields': sorted(required), 'bound_fields': sorted(required & set(binding)),
        'missing_fields': sorted(missing), 'instrument_status': model.get('status') if model else None,
        'instrument_training_rows': cfg['instrument_training_rows'],
        'instrument_validation_rows': cfg['instrument_validation_rows'],
        'calibration_compute': calibration_compute, 'planning_compute': budget.spent - calibration_compute,
        'compute_cost': budget.spent, 'new_compute_cost': budget.spent,
        'remaining_compute': cfg['available_compute'] - budget.spent, 'ledger_ref': cfg['ledger_ref'],
        'source_costs_ref': cfg['source_costs_ref'], 'plan_ref': ref(output / 'plan.json'),
        'instrument_models_ref': ref(output / 'instrument_models.json'),
        'experiment_methods_ref': ref(output / 'program_experiment_methods.json'),
        'frontier_ref': cfg['frontier_ref'], 'selected_ref': cfg['selected_ref'],
        'pool_occurrences': cfg['pool_occurrences'], 'candidate_refs': locations,
        'world_predictors_unchanged': True, 'new_world_fits': 0, 'new_observations': 0,
        'model_calls': 0, 'independent_support': False, 'execution_timing': timing.result()}
    save(output / 'receipt.json', result)
    return result


def register_instrument_probe(request, output):
    """Execute a frozen instrument or bound-response probe and consume its rows."""
    request, output = Path(request), Path(output)
    supplied = load(request)
    source = Path(supplied['source'])
    prior, actual, plan = (load(source / name) for name in
        ('registration.json', 'receipt.json', 'plan.json'))
    response = actual['status'] == 'response_distinguishing_experiment'
    if actual['status'] not in ('instrument_distinguishing_experiment', 'response_distinguishing_experiment') or not plan.get('probe'):
        raise ValueError('A saved predicted instrument or response distinction is required')
    world = Path(prior['source'])
    context = load(world / 'registration.json') if response else prior
    library_source = world if response else source
    frontier, original, rows, _, _, _, _, scores = parent_inputs(world)
    if frontier['selected_ref'] != actual['selected_ref']:
        raise ValueError('Keep the original selected response and complete pool')
    sequence = plan['probe']['sequence']
    for location in [*original['data_refs'], *original.get('appended_observation_refs', [])]:
        old = read_method_reference(location, source)['probe']['sequence']
        if sequence == old[:len(sequence)]:
            raise ValueError('The planned control prefix has already been purchased')
    balance = read_method_reference(supplied['ledger_ref'], source)
    costs = read_method_reference(supplied['source_costs_ref'], source)
    if actual.get('instrument_models_ref'):
        models_ref = actual['instrument_models_ref']
        methods_ref = actual.get('experiment_methods_ref', ref(source / 'program_experiment_methods.json'))
    elif response:
        models_ref = prior.get('instrument_models_ref', ref(library_source / 'instrument_models.json'))
        methods_ref = prior.get('experiment_methods_ref', ref(library_source / 'program_experiment_methods.json'))
    else:
        models_ref = ref(library_source / 'instrument_models.json')
        methods_ref = ref(library_source / 'program_experiment_methods.json')
    model = read_method_reference(models_ref, source)[-1]
    instrument_hypotheses = sum(len(field.get('hypotheses', [field])) for field in model['fields'].values())
    hypotheses = (len(plan['hypotheses']) + len(plan['predicted_fields']) + instrument_hypotheses
        if response else instrument_hypotheses)
    update_quote = len(sequence) * (len(scores) + hypotheses)
    reserve = supplied.get('reserved_compute', 3000 + 96)
    if (reserve < 96 or balance <= update_quote + reserve or costs['cumulative_protocol_observations'] + len(sequence)
            > read_method_reference(supplied['observation_limit_ref'], source)):
        raise ValueError(f'Keep the original update, return, confirmation and observation allowances: '
            f'balance={balance}, update={update_quote}, reserve={reserve}')
    cfg = {**supplied, 'source': str(source), 'world_source': str(world),
        'source_task_ref': prior['source_task_ref'], 'object_model_ref': context['object_model_ref'],
        'prior_method_review': prior.get('prior_method_review', context['prior_method_review']),
        'planning_kind': 'response' if response else 'instrument', 'library_source': str(library_source),
        'instrument_models_ref': models_ref, 'experiment_methods_ref': methods_ref,
        **({'candidate_refs': prior['candidate_refs'], 'binding_ref': prior['binding_ref']} if response else {}),
        'plan_ref': ref(source / 'plan.json'), 'selected_ref': actual['selected_ref'],
        'available_compute': balance, 'compute_limit': balance - reserve, 'reserved_compute': reserve,
        'update_compute_upper': update_quote, 'quality_threshold': prior['quality_threshold'],
        'active_wall_seconds': prior['active_wall_seconds'], 'observation_rows': len(sequence),
        'previous_observations': costs['cumulative_protocol_observations'],
        'new_world_fits': 0, 'model_calls': 0,
        'scope': ('one_frozen_response_probe_then_assessment_method_feedback_and_full_pool_update' if response else
            'one_frozen_instrument_probe_then_paid_pool_update_and_response_planning'),
        'stop_rule': 'One selected probe only; no instrument refit, world refit or automatic confirmation'}
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'registration.json', cfg)
    freeze_entry(output, source)
    for name in (Path(__file__).name, 'continue_successor_relation_intervals.py'):
        relative = Path('scripts') / name
        write_text_incrementally(output / 'runtime' / relative, (ROOT / relative).read_text(),
            base=source / 'runtime' / relative)
    return cfg


def run_instrument_probe(output):
    from copy import deepcopy
    from dataclasses import asdict, replace
    from scripts.successor_interval_scene import pair_rows
    from scripts.continue_successor_relation_intervals import extend_scores
    from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
    from hegel_machine.cognition.checkpoint_storage import write_json_checkpoint
    from hegel_machine.cognition.competing_object_experiments import (
        plan_competing_object_probe, shorten_discriminating_probe, assess_competing_object_probe)
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.instrument_hypotheses import assess_instrument_hypotheses
    from hegel_machine.cognition.instrument_experiment_methods import observe_instrument_method
    from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    from hegel_machine.cognition.models import ResearchConfig, Probe
    from hegel_machine.cognition.experiment_runtime import python_execution_deadline

    output = Path(output)
    if (output / 'receipt.json').exists():
        return {**load(output / 'receipt.json'), 'new_compute_cost': 0, 'reused': True}
    if ROOT != (output / 'runtime').resolve() or (output / 'started.json').exists():
        raise ValueError('Execute the frozen instrument probe once')
    cfg = load(output / 'registration.json')
    source, world = Path(cfg['source']), Path(cfg['world_source'])
    response = cfg.get('planning_kind') == 'response'
    library_source = Path(cfg.get('library_source', str(source)))
    frontier, original, paid, probes, references, pool, parent, scores = parent_inputs(world)
    if response:
        public = TaskSpec(**read_method_reference(original['public_spec_ref'], output))
        command = replace(public, feature_names=tuple(public.feature_bounds), context={})
        measured = replace(public, context={**public.context,
            'object_model': read_method_reference(cfg['object_model_ref'], output)})
        candidate_refs = cfg['candidate_refs']
        candidates = [retained_entry(location, output, index) for index, location in enumerate(candidate_refs)]
        binding = {name: Program.from_dict(program) for name, program in
            read_method_reference(cfg['binding_ref'], output).items()}
    else:
        _, _, command, measured, _, candidates, candidate_refs, _ = instrument_plan_inputs(world, cfg['object_model_ref'])
    plan = deepcopy(read_method_reference(cfg['plan_ref'], output))
    probe = Probe(plan['probe']['probe_id'], tuple(plan['probe']['sequence']))
    plan['probe'] = probe
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=cfg['quality_threshold']))
    agent._learn_instruments = True
    agent.library.instrument_models = deepcopy(read_method_reference(
        cfg.get('instrument_models_ref', ref(library_source / 'instrument_models.json')), output))
    agent.library.program_experiment_methods.records = deepcopy(read_method_reference(
        cfg.get('experiment_methods_ref', ref(library_source / 'program_experiment_methods.json')), output))
    model = agent.library.instrument_models[-1]
    budget, timing = SearchBudget(cfg['compute_limit']), ExecutionTiming(('acquire', 'instruments', 'pool', 'plan'))
    observation_cost = measurement_cost = 0
    actions = []
    save(output / 'started.json', {'registration_ref': ref(output / 'registration.json')})
    try:
        with python_execution_deadline(cfg['active_wall_seconds'], activity=True):
            with AnonymousObjectLab(**read_method_reference(cfg['source_task_ref'], output)) as lab:
                with timing.phase('acquire'):
                    batch = lab.acquire(probe, ('fine',))
                    observation_cost, measurement_cost = lab.observation_cost, lab.measurement_cost
                    save(output / 'observations.json', {'probe': asdict(probe), 'batch': asdict(batch)})
            with timing.phase('instruments'):
                if response:
                    assessment = assess_competing_object_probe(plan,
                        pair_rows(probe.sequence, batch.observations), budget, tolerance=cfg['quality_threshold'])
                    if assessment['calibration_status'] == 'counterexample':
                        instrument_failure = assess_instrument_hypotheses(model, command, measured,
                            pair_rows(probe.sequence, batch.observations), budget, tolerance=cfg['quality_threshold'])
                        agent._record_instrument_assessment(model, instrument_failure, measured, observation_cost, actions)
                        save(output / 'instrument_assessment.json', instrument_failure)
                    feedback = agent.library.program_experiment_methods.observe(plan, assessment,
                        task_id=measured.task_id, command_spec=command, binding=binding,
                        tolerance=cfg['quality_threshold'])
                else:
                    assessment = assess_instrument_hypotheses(model, command, measured,
                        pair_rows(probe.sequence, batch.observations), budget, tolerance=cfg['quality_threshold'])
                    feedback = observe_instrument_method(agent.library.program_experiment_methods, plan,
                        assessment, model, command, task_id=measured.task_id,
                        observation_rows=observation_cost, measurement_cost=measurement_cost,
                        tolerance=cfg['quality_threshold'])
                    agent._record_instrument_assessment(model, assessment, measured, observation_cost, actions)
            save(output / ('response_assessment.json' if response else 'instrument_assessment.json'),
                {'assessment': assessment, 'method_feedback': feedback, 'actions': actions})
            write_json_checkpoint(output / 'instrument_models.json', agent.library.instrument_models,
                base=Path(cfg.get('instrument_models_ref',
                    ref(library_source / 'instrument_models.json'))['file']))
            write_json_checkpoint(output / 'program_experiment_methods.json', agent.library.program_experiment_methods.records,
                base=Path(cfg.get('experiment_methods_ref',
                    ref(library_source / 'program_experiment_methods.json'))['file']))
            instrument_compute = budget.spent
            with timing.phase('pool'):
                extend_scores(scores, len(paid), batch.observations, budget, parent.get('fallback', 0.))
            pool_compute = budget.spent - instrument_compute
            for entry in pool:
                entry['validation_loss'] = scores[entry['candidate'].program]
            selected = agent._best(pool)
            if scores[parent['candidate'].program] <= selected['validation_loss'] + cfg['quality_threshold']:
                selected = parent
            selected_ref = (frontier['selected_ref'] if selected is parent else
                next(location for entry, (location, _) in reversed(list(zip(pool, references)))
                    if entry['candidate'].program == selected['candidate'].program))
            grouped = OrderedDict()
            for entry, (location, _) in zip(pool, references):
                grouped.setdefault(entry['candidate'].program, []).append(location)
            save(output / 'paid_pool/registration.json', {**original,
                'appended_observation_refs': [*original.get('appended_observation_refs', []), ref(output / 'observations.json')],
                'comparison_rows': len(paid) + observation_cost,
                'feedback_rows': len(paid) + observation_cost - original['training_rows'],
                'probes': [*probes, {'probe_id': probe.probe_id,
                    'start': len(paid), 'rows': observation_cost}]})
            save(output / 'paid_pool/receipt.json', {'full_comparison': True,
                'rows': len(paid) + observation_cost, 'candidate_occurrences': len(pool), 'unique_programs': len(scores),
                'scores': [{'candidate_refs': locations, 'mse': scores[program]} for program, locations in grouped.items()],
                'selected_ref': selected_ref, 'selected_mse': scores[selected['candidate'].program]})
            save(output / 'retained_frontier.json', {'full_comparison': True,
                'original_pool_ref': ref(output / 'paid_pool/receipt.json'), 'appended_comparison_refs': [],
                'complete_current_pool_occurrences': len(pool), 'selected_ref': selected_ref,
                'selected_mse': scores[selected['candidate'].program], 'comparison_rows': len(paid) + observation_cost,
                'parent_frontier_ref': ref(world / 'retained_frontier.json'), 'independent_support': False})
            binding = agent._instrument_binding(measured, command)
            required = set().union(*(item['candidate'].program.required_features() for item in candidates))
            missing = required - set(binding) - set(command.feature_names)
            next_plan = {'probe': None, 'forecast_status': 'instrument_mapping_unresolved'}
            status = 'instrument_mapping_unresolved'
            with timing.phase('plan'):
                if response:
                    status = ('instrument_mapping_unresolved' if assessment['calibration_status'] == 'counterexample'
                        else 'response_distinguished' if assessment['status'] == 'distinguished'
                        else 'paid_world_counterexample' if assessment['status'] == 'all_refuted'
                        else 'response_disagreement_unresolved')
                    next_plan = {'probe': None, 'forecast_status': status,
                        'assessment_ref': ref(output / 'response_assessment.json')}
                elif not missing:
                    supported = [item for item in candidates if scores[item['candidate'].program] <= cfg['quality_threshold']]
                    if len(supported) < 2:
                        status = 'fresh_confirmation_candidate' if supported else 'paid_world_counterexample'
                    else:
                        research = RelationIntervalResearch(parent['candidate'].program, [], budget,
                            research_program=parent['research_program'])
                        next_plan = plan_competing_object_probe(command, supported, budget, binding=binding,
                            serial=1, length=5, tolerance=cfg['quality_threshold'], questions=research.questions(supported),
                            methods=agent.library.program_experiment_methods, task_id=measured.task_id)
                        next_plan = shorten_discriminating_probe(next_plan, budget, tolerance=cfg['quality_threshold'])
                        status = ('response_distinguishing_experiment'
                            if next_plan['forecast_status'] == 'predicted_distinction' and next_plan['probe'] else
                            'fresh_confirmation_candidate' if next_plan['forecast_status'] == 'algebraically_shared' else
                            'response_disagreement_unresolved')
            save(output / 'plan.json', {**next_plan, 'probe': asdict(next_plan['probe']) if next_plan.get('probe') else None,
                'acquired': False, 'planning_kind': 'response'})
            save(output / 'instrument_binding.json', {name: program.to_dict() for name, program in binding.items()})
    except BaseException as error:
        save(output / 'failure.json', {'error': repr(error), 'compute_cost': budget.spent,
            'remaining_compute': cfg['available_compute'] - budget.spent,
            'observation_cost': observation_cost, 'measurement_cost': measurement_cost,
            'ledger_ref': cfg['ledger_ref'], 'execution_timing': timing.result()})
        raise
    result = {'status': status, 'compute_cost': budget.spent, 'new_compute_cost': budget.spent,
        'instrument_binding_ref': ref(output / 'instrument_binding.json'),
        'instrument_models_ref': ref(output / 'instrument_models.json'),
        'experiment_methods_ref': ref(output / 'program_experiment_methods.json'),
        **({'response_evidence_ref': ref(output / 'response_assessment.json')} if response else {}),
        'instrument_assessment_compute': instrument_compute if not response else 0,
        'response_assessment_compute': instrument_compute if response else 0, 'pool_compute': pool_compute,
        'planning_compute': budget.spent - instrument_compute - pool_compute,
        'remaining_compute': cfg['available_compute'] - budget.spent,
        'observation_cost': observation_cost, 'measurement_cost': measurement_cost,
        'cumulative_protocol_observations': cfg['previous_observations'] + observation_cost,
        'bound_fields': sorted(required & set(binding)), 'missing_fields': sorted(missing),
        'selected_ref': selected_ref, 'selected_loss': scores[selected['candidate'].program],
        'full_comparison': True, 'pool_occurrences': len(pool), 'comparison_rows': len(paid) + observation_cost,
        'comparison_ref': ref(output / 'paid_pool/receipt.json'), 'plan_ref': ref(output / 'plan.json'),
        'new_world_fits': 0, 'new_instrument_fits': 0, 'model_calls': 0, 'independent_support': False,
        'source_costs_ref': cfg['source_costs_ref'], 'ledger_ref': cfg['ledger_ref'], 'execution_timing': timing.result()}
    save(output / 'receipt.json', result)
    return result


def register_response_plan(source, output, *, request=None):
    """Visit the existing full-program planner after the question-only search."""
    source, output = Path(source), Path(output)
    previous, actual = (load(source / name) for name in ('registration.json', 'receipt.json'))
    if actual['status'] == 'paid_interval_updated':
        supplied = load(request)
        if not actual['execution_complete'] or not actual['full_comparison']:
            raise ValueError('Finish the paid interval update before using its competing hypotheses')
        available = read_method_reference(supplied['ledger_ref'], source)
        reserve = supplied.get('reserved_compute', 3096)
        if reserve < 96 or available <= reserve:
            raise ValueError('Preserve the existing return and confirmation reserves')
        from hegel_machine.cognition.equivalent_object_programs import canonical_object_program
        frontier, original, _, _, _, _, _, scores = parent_inputs(source)
        entries = [retained_entry(location, source, index)
            for index, location in enumerate(actual['candidate_refs'])]
        programs = [entry['candidate'].program for entry in entries]
        public = read_method_reference(original['public_spec_ref'], source)
        shared = (len(programs) >= 2 and bool(public['feature_bounds'])
            and len({canonical_object_program(program) for program in programs}) < 2)
        # The native planner can close algebraically shared responses without
        # purchasing a row. Otherwise even one new row must be consumed by the
        # complete pool before confirmation. This is only a necessary lower
        # bound, not a quote for planning, assessment or a complete experiment.
        minimum_rows = 0 if shared else 1
        lower_bound = {'minimum_new_rows': minimum_rows, 'unique_pool_programs': len(scores),
            'pool_consume_compute_lower_bound': minimum_rows * len(scores),
            'reserved_compute': reserve,
            'required_compute_lower_bound': minimum_rows * len(scores) + reserve,
            'algebraically_shared': shared, 'sufficient_quote': False,
            'basis': 'Native canonical equality or at least one new full-pool response; excludes planning and assessment'}
        if available < lower_bound['required_compute_lower_bound']:
            result = {'status': 'complete_response_unit_unfunded', 'source': str(source),
                'frontier_ref': ref(source / 'retained_frontier.json'),
                'selected_ref': frontier['selected_ref'], 'ledger_ref': supplied['ledger_ref'],
                'source_cost_refs': supplied.get('source_cost_refs', []),
                'available_compute': available, 'remaining_compute': available,
                'response_unit_lower_bound': lower_bound,
                'compute_cost': 0, 'new_compute_cost': 0, 'new_observations': 0,
                'new_fits': 0, 'model_calls': 0, 'independent_support': False,
                'scientific_refutation': False,
                'reason': 'Preserve the remaining balance; even the minimum new response and original reserve are unfunded'}
            output.mkdir(parents=True, exist_ok=False)
            save(output / 'admission.json', result)
            return result
        cfg = {**supplied, 'source': str(source), 'planning_kind': 'updated_interval_methods',
            'response_unit_lower_bound': lower_bound,
            'candidate_refs': actual['candidate_refs'],
            'frontier_ref': ref(source / 'retained_frontier.json'), 'selected_ref': actual['selected_ref'],
            **{key: previous[key] for key in ('binding_ref', 'experiment_methods_ref',
                'instrument_models_ref', 'source_task_ref', 'object_model_ref',
                'quality_threshold', 'active_wall_seconds')},
            'prior_plan_ref': previous['response_evidence_ref'],
            'available_compute': available, 'compute_limit': min(6000, available - reserve),
            'reserved_compute': reserve, 'serial': 2, 'length': 5,
            'scope': 'learned_experiment_method_on_updated_paid_intervals',
            'stop_rule': 'One native method-first forecast with current intervals; no acquisition or refit'}
        output.mkdir(parents=True, exist_ok=False)
        save(output / 'registration.json', cfg)
        freeze_entry(output, source)
        relative = Path('scripts') / Path(__file__).name
        write_text_incrementally(output / 'runtime' / relative, (ROOT / relative).read_text(),
            base=source / 'runtime' / relative)
        return cfg
    plan = load(source / 'plan.json')
    if previous.get('scope') == 'one_complementary_full_program_forecast_no_question_or_seed_retry':
        if (actual['status'] != 'response_disagreement_unresolved'
                or plan['forecast_status'] != 'no_predicted_distinction'
                or plan.get('directed_candidates') or plan.get('method_candidates')
                or not plan.get('generic_candidates')):
            raise ValueError('Continue only the unproductive recall with an unvisited directed branch')
        available = actual['remaining_compute']
        if available <= 3096:
            raise ValueError('Preserve the existing return and confirmation reserves')
        cfg = {**previous, 'planning_kind': 'paid_directed_response',
            'available_compute': available, 'compute_limit': available - 3096,
            'ledger_ref': ref(source / 'receipt.json', 'remaining_compute'),
            'source_cost_refs': [*previous['source_cost_refs'], ref(source / 'receipt.json')],
            'prior_plan_ref': ref(source / 'plan.json'),
            'reused_method_search_compute': plan['method_search_cost'],
            'reused_generic_trajectories': plan['generic_candidates'],
            'scope': 'paid_bound_response_review_only_unvisited_directed_synthesis',
            'stop_rule': 'Existing 6000-unit directed solver plus forecast review; no repeated recall, generic forecasts, acquisition or refit'}
        output.mkdir(parents=True, exist_ok=False)
        save(output / 'registration.json', cfg)
        freeze_entry(output, source)
        relative = Path('scripts') / Path(__file__).name
        write_text_incrementally(output / 'runtime' / relative, (ROOT / relative).read_text(),
            base=source / 'runtime' / relative)
        return cfg
    if (actual['status'] != 'response_disagreement_unresolved' or actual['missing_fields']
            or plan['forecast_status'] != 'no_predicted_distinction'
            or not plan.get('questions_attempted') or plan.get('generic_candidates', 0)):
        raise ValueError('Only the completed question search with complete bindings needs this complementary visit')
    available = actual['remaining_compute']
    if available <= 3096:
        raise ValueError('Preserve the existing return and confirmation reserves')
    inherited = load(Path(previous['source']) / 'registration.json')
    cfg = {'source': str(source), 'candidate_refs': inherited['candidate_refs'],
        'frontier_ref': ref(source / 'retained_frontier.json'), 'selected_ref': actual['selected_ref'],
        'binding_ref': ref(source / 'instrument_binding.json'),
        'experiment_methods_ref': ref(source / 'program_experiment_methods.json'),
        'prior_plan_ref': ref(source / 'plan.json'),
        'ledger_ref': ref(source / 'receipt.json', 'remaining_compute'),
        'source_cost_refs': [previous['source_costs_ref'], ref(source / 'receipt.json')],
        'available_compute': available, 'compute_limit': min(6000, available - 3096),
        'reserved_compute': 3096, 'quality_threshold': previous['quality_threshold'],
        'active_wall_seconds': previous['active_wall_seconds'],
        'serial': 1, 'length': 5, 'source_task_ref': previous['source_task_ref'],
        'scope': 'one_complementary_full_program_forecast_no_question_or_seed_retry',
        'stop_rule': 'One original full-program planning pass; no acquisition, refit or automatic confirmation'}
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'registration.json', cfg)
    freeze_entry(output, source)
    relative = Path('scripts') / Path(__file__).name
    write_text_incrementally(output / 'runtime' / relative, (ROOT / relative).read_text(),
        base=source / 'runtime' / relative)
    return cfg


def run_response_plan(output):
    from dataclasses import asdict, replace
    from hegel_machine.cognition.competing_object_experiments import (
        plan_competing_object_probe, shorten_discriminating_probe)
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.experiment_runtime import python_execution_deadline

    output = Path(output)
    if (output / 'receipt.json').exists():
        return {**load(output / 'receipt.json'), 'new_compute_cost': 0, 'reused': True}
    if ROOT != (output / 'runtime').resolve() or (output / 'started.json').exists():
        raise ValueError('Execute the frozen complementary planning pass once')
    cfg = load(output / 'registration.json')
    frontier, original, rows, _, _, _, _, _ = parent_inputs(cfg['source'])
    public = TaskSpec(**read_method_reference(original['public_spec_ref'], output))
    command = replace(public, feature_names=tuple(public.feature_bounds), context={})
    entries = [retained_entry(location, output, index) for index, location in enumerate(cfg['candidate_refs'])]
    binding = {name: Program.from_dict(program) for name, program in
        read_method_reference(cfg['binding_ref'], output).items()}
    agent = UnifiedResearchAgent(ResearchConfig(minimum_improvement=cfg['quality_threshold']))
    agent.library.program_experiment_methods.records = read_method_reference(cfg['experiment_methods_ref'], output)
    budget, timing = SearchBudget(cfg['compute_limit']), ExecutionTiming(('plan',))
    save(output / 'started.json', {'registration_ref': ref(output / 'registration.json')})
    try:
        with python_execution_deadline(cfg['active_wall_seconds'], activity=True), timing.phase('plan'):
            if cfg.get('planning_kind') == 'paid_directed_response':
                from hegel_machine.cognition.competing_object_experiments import review_paid_program_competition
                paid = read_method_reference(frontier['original_pool_ref'], output)
                hypotheses = []
                for entry, location in zip(entries, cfg['candidate_refs'], strict=True):
                    index, score = next((index, item) for index, item in enumerate(paid['scores'])
                        if location in item['candidate_refs'])
                    hypotheses.append({'candidate_id': entry['candidate'].candidate_id,
                        'program': entry['candidate'].program.to_dict(),
                        'score': {'returncode': 0, 'mse': score['mse'], 'rows': len(rows)},
                        'score_ref': {**frontier['original_pool_ref'], 'json_path': ['scores', index]}})
                records = []
                for location in [*original['data_refs'], *original.get('appended_observation_refs', [])]:
                    document = read_method_reference(location, output)
                    records.append({'rows': [{'features': point} for point in document['probe']['sequence']]})
                review = review_paid_program_competition(command, hypotheses, records, budget,
                    score_scope_ref=frontier['original_pool_ref'], score_rows=len(rows),
                    binding=binding, control_levels=None, serial=cfg['serial'], length=cfg['length'],
                    tolerance=cfg['quality_threshold'])
                plan = review['plan'] or {'probe': None, 'hypotheses': hypotheses,
                    'predicted_separation': None, 'forecast_status': review['status']}
                save(output / 'paid_review.json', {key: value for key, value in review.items() if key != 'plan'} |
                    {'plan_ref': ref(output / 'plan.json')})
            else:
                plan = plan_competing_object_probe(command, entries, budget, binding=binding,
                    serial=cfg['serial'], length=cfg['length'], tolerance=cfg['quality_threshold'], questions=(),
                    methods=agent.library.program_experiment_methods, task_id=public.task_id,
                    **({'generation_budget': SearchBudget(cfg['compute_limit'])}
                       if cfg.get('planning_kind') == 'updated_interval_methods' else {}))
                plan = shorten_discriminating_probe(plan, budget, tolerance=cfg['quality_threshold'])
        save(output / 'plan.json', {**plan,
            'probe': asdict(plan['probe']) if plan.get('probe') else None, 'acquired': False})
    except BaseException as error:
        save(output / 'failure.json', {'error': repr(error), 'compute_cost': budget.spent,
            'remaining_compute': cfg['available_compute'] - budget.spent,
            'ledger_ref': cfg['ledger_ref'], 'execution_timing': timing.result()})
        raise
    result = {'status': ('response_distinguishing_experiment'
        if plan['forecast_status'] == 'predicted_distinction' and plan.get('probe') else 'response_disagreement_unresolved'),
        'forecast_status': plan['forecast_status'], 'predicted_separation': plan['predicted_separation'],
        'compute_cost': budget.spent, 'new_compute_cost': budget.spent,
        'remaining_compute': cfg['available_compute'] - budget.spent, 'ledger_ref': cfg['ledger_ref'],
        'source_cost_refs': cfg['source_cost_refs'], 'plan_ref': ref(output / 'plan.json'),
        'prior_plan_ref': cfg['prior_plan_ref'], 'frontier_ref': cfg['frontier_ref'],
        'pool_occurrences': frontier['complete_current_pool_occurrences'], 'selected_ref': cfg['selected_ref'],
        'new_world_fits': 0, 'new_instrument_fits': 0, 'new_observations': 0, 'model_calls': 0,
        'independent_support': False, 'execution_timing': timing.result()}
    save(output / 'receipt.json', result)
    return result


def register_confirmation(source, output, *, request=None):
    """Freeze two delivered programs against the original unpurchased random panel."""
    from dataclasses import asdict
    from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
    from hegel_machine.cognition.evaluation import initial_prediction

    source, output = Path(source), Path(output)
    previous, actual = load(source / 'registration.json'), load(source / 'receipt.json')
    frontier, original, rows, _, _, _, selected, _ = parent_inputs(source)
    if (not actual['execution_complete'] or not actual['full_comparison']
            or actual['selected_ref'] != frontier['selected_ref']
            or actual['selected_loss'] > previous['quality_threshold']):
        raise ValueError('Complete paid research and the selected supported parent are required')
    task_ref = previous['source_task_ref']
    task = read_method_reference(task_ref, source)
    supplied = load(request) if request else None
    gate = actual.get('confirmation_gate')
    if gate is not None:
        competition = (read_method_reference(supplied['competition_review_ref'], source)
            if supplied and supplied.get('competition_review_ref') else {})
        admitted = gate['admitted'] or (competition.get('forecast_status') == 'algebraically_shared'
            and gate['all_paid_squared_error_upper_bound'] <= previous['quality_threshold'])
        if not admitted:
            raise ValueError('The original paid-error and remaining-competition confirmation gate is not met')
    if supplied:
        protocol = read_method_reference(supplied['confirmation_protocol_ref'], source)
        protocol_refs = {'confirmation_protocol_ref': supplied['confirmation_protocol_ref']}
        old_acquisitions = []
        data_refs = [*original['data_refs'], *original.get('appended_observation_refs', [])]
        observed = [(location, batch['probe']['probe_id'], batch['probe']['sequence'])
            for location in data_refs for batch in [read_method_reference(location, source)]]
        data = {'data_refs': data_refs}
        ledger_ref, cost_ref = supplied['ledger_ref'], supplied['source_costs_ref']
        allocation_ref = supplied['observation_limit_ref']
        observation_limit = read_method_reference(allocation_ref, source)
        prior_review = supplied['prior_method_review']
    else:
        cohort = Path(task_ref['file']).parent
        cohort_cfg = load(cohort / 'registration.json')
        terminal_ref = ref(cohort / 'terminal' / str(task['seed']) / 'summary.json')
        terminal = read_method_reference(terminal_ref, source)
        if any(terminal['shared_actual_costs'].values()) or any(
                arm['prediction_measured'] for arm in terminal['arms'].values()):
            raise ValueError('The original fixed random panel has already been measured')
        protocol = cohort_cfg['terminal']
        protocol_refs = {'original_terminal_protocol_ref': ref(cohort / 'registration.json', 'terminal'),
            'original_unpurchased_terminal_ref': terminal_ref}
        old_acquisitions = [ref(cohort / 'runs' / str(task['seed']) / arm / 'failure.json',
            'new_acquisitions') for arm in cohort_cfg['arms']]
        observed = [(location, batch['probe_id'], batch['sequence'])
            for location in old_acquisitions for batch in read_method_reference(location, source)]
        paid = OrderedDict()
        for row in read_method_reference(original['paid_records_ref'], source):
            paid.setdefault(row['probe_id'], []).append(row)
        observed.extend((original['paid_records_ref'], name,
            [row['controls'] for row in sorted(group, key=lambda row: row['position'])])
            for name, group in paid.items())
        data = {'paid_records_ref': original['paid_records_ref']}
        ledger_ref = ref(source / 'method_return/receipt.json', 'remaining_compute')
        cost_ref = ref(source / 'source_costs.json')
        allocation_ref = ref(Path(cohort_cfg['budget_contract_ref']['file']), 'authorized_limits')
        observation_limit = read_method_reference(allocation_ref, source)['observation_cost']
        prior_review = str(source / 'method_return')
    if protocol['length'] != 48 or protocol['trajectories'] != 1:
        raise ValueError('Keep the original single 48-row random confirmation protocol')
    seed = task['seed'] + protocol['random_seed_offset']
    with AnonymousObjectLab(**task) as lab:
        probe = lab.random_probe(seed, length=48, name='shared_frozen_confirmation')
        measurement = 48 * next(channel.cost_per_row for channel in lab.channels if channel.name == 'fine')
    sequence = list(probe.sequence)
    if any(sequence[:min(len(sequence), len(old))] == old[:min(len(sequence), len(old))]
            for _, _, old in observed):
        raise ValueError('The fixed confirmation repeats an already paid control prefix; stop without changing seed')
    available = read_method_reference(ledger_ref, source)
    costs = read_method_reference(cost_ref, source)
    if available < 96 + 3000 or costs['cumulative_protocol_observations'] + 48 > observation_limit:
        raise ValueError('The original account cannot fund confirmation and its reserved ordinary return')
    incumbent_ref = supplied.get('incumbent_ref', previous['parent_ref']) if supplied else previous['parent_ref']
    incumbent = retained_entry(incumbent_ref, source, 0)
    public = TaskSpec(**read_method_reference(original['public_spec_ref'], source))
    fallback = incumbent.get('fallback', initial_prediction(rows, public.risk_quantile))
    output.mkdir(parents=True, exist_ok=False)
    frozen = {name: {'candidate': {'program': entry['candidate'].program.to_dict()},
        'source_ref': location, 'fallback': fallback}
        for name, entry, location in [('incumbent', incumbent, incumbent_ref),
                                     ('selected', selected, actual['selected_ref'])]}
    save(output / 'frozen_entries.json', frozen)
    save(output / 'prepared_state.json', {'fallback': fallback,
        'basis': 'same complete paid rows and incumbent fallback as native research'})
    for arm in frozen:
        save(output / arm / 'outcome.json', {'execution_complete': True, 'full_comparison': True,
            'selected_ref': ref(output / 'frozen_entries.json', arm),
            'source_ref': frozen[arm]['source_ref'], 'role': 'fixed_predictor_readout_not_method_pair'})
    save(output / 'confirmation_plan.json', {'probe': asdict(probe), 'seed': seed,
        'old_acquisition_refs': old_acquisitions, **data,
        'checked_control_sequences': len(observed), 'no_old_prefix_match': True,
        'responses_obtained': False, 'missing_historical_labels_reconstructed': False})
    cfg = {'source': str(source), 'source_task_ref': task_ref, 'task': task,
        'source_selected_ref': actual['selected_ref'], 'incumbent_ref': incumbent_ref,
        'source_frontier_ref': ref(source / 'retained_frontier.json'),
        'frozen_predictors_ref': ref(output / 'frozen_entries.json'),
        'operations': {name: 'fixed_predictor' for name in frozen},
        'confirmation_rows': 48, 'confirmation_seed': seed, 'quality_threshold': previous['quality_threshold'],
        **protocol_refs,
        'frozen_research_method_ref': previous.get('research_method_ref'),
        'method_execution_contract_ref': ref(source / 'registration.json'),
        'new_data_scope': 'same_physical_source_fresh_random_trajectory',
        'comparison_scope': 'fixed_predictors_not_independent_task_transfer_or_original_ABC_correction',
        'ledger_ref': ledger_ref, 'available_compute': available, 'source_cost_refs': [cost_ref],
        'allocation_ref': allocation_ref, 'previous_observations': costs['cumulative_protocol_observations'],
        'confirmation_cost_quote': {'compute_cost': 96, 'observation_cost': 48,
            'measurement_cost': measurement, 'queries': 1, 'new_fits': 0, 'model_calls': 0},
        'method_return_compute_reserve': 3000, 'prior_method_review': prior_review,
        'active_wall_seconds': previous['active_wall_seconds'], 'method_pair_created': False,
        'independent_task_support': False,
        'stop_rule': 'One fixed shared random panel; retain failure; no fit, seed replacement or repeat'}
    save(output / 'registration.json', cfg)
    freeze_entry(output, source)
    for name in (Path(__file__).name, 'run_learned_structure_successor.py'):
        relative = Path('scripts') / name
        write_text_incrementally(output / 'runtime' / relative, (ROOT / relative).read_text(),
            base=source / 'runtime' / relative)
    return cfg


def run_confirmation(output):
    from scripts.run_learned_structure_successor import confirm_pair
    from hegel_machine.cognition.experiment_runtime import python_execution_deadline

    output = Path(output)
    if (output / 'receipt.json').exists():
        return {**load(output / 'receipt.json'), 'new_compute_cost': 0, 'reused': True}
    if ROOT != (output / 'runtime').resolve() or (output / 'started.json').exists():
        raise ValueError('Execute the freshly frozen confirmation once')
    cfg = load(output / 'registration.json')
    save(output / 'started.json', {'registration_ref': ref(output / 'registration.json')})
    timing = ExecutionTiming(('confirmation',))
    try:
        with timing.phase('confirmation'), python_execution_deadline(cfg['active_wall_seconds'], activity=True):
            confirmation = confirm_pair(output, cfg)
    except BaseException as error:
        failed = output / 'confirmation/failure.json'
        cost = load(failed).get('costs') if failed.exists() else None
        save(output / 'failure.json', {'error': repr(error), 'actual_costs': cost,
            'source_cost_refs': cfg['source_cost_refs'], 'ledger_ref': cfg['ledger_ref'],
            'remaining_compute': cfg['available_compute'] - cost['compute_cost'] if cost else None,
            'execution_timing': timing.result()})
        raise
    cost = confirmation['shared_actual_costs']
    result = {'status': 'confirmed' if confirmation['arms']['selected']['passed'] else 'confirmation_failed',
        'confirmation_ref': ref(output / 'confirmation/receipt.json'),
        'frozen_predictors_ref': cfg['frozen_predictors_ref'], 'source_task_ref': cfg['source_task_ref'],
        'selected_ref': cfg['source_selected_ref'], 'incumbent_ref': cfg['incumbent_ref'],
        'compute_cost': cost['compute_cost'], 'new_compute_cost': cost['compute_cost'],
        'observation_cost': cost['observation_cost'], 'measurement_cost': cost['measurement_cost'],
        'remaining_compute': cfg['available_compute'] - cost['compute_cost'],
        'cumulative_protocol_observations': cfg['previous_observations'] + cost['observation_cost'],
        'source_cost_refs': cfg['source_cost_refs'], 'ledger_ref': cfg['ledger_ref'],
        'new_fits': 0, 'model_calls': 0, 'method_pair_created': False,
        'independent_task_support': False, 'comparison_scope': cfg['comparison_scope'],
        'execution_timing': timing.result()}
    save(output / 'receipt.json', result)
    return result


def return_parent(source, output, maximum_compute=3000, *, previous_return=None, allocation_ref=None):
    """Continue the existing method topic with a sequential, non-paired return."""
    from copy import deepcopy
    from hegel_machine.cognition.common_object_research import (
        CommonObjectDependencyMethods, record_native_parent_research_return)
    from hegel_machine.cognition.method_feedback import ensure_python_method_topic
    from hegel_machine.cognition.question_routing import consume_method_return_agenda

    source, output = Path(source), Path(output)
    if (output / 'receipt.json').exists():
        return {**load(output / 'receipt.json'), 'new_compute_cost': 0, 'reused': True}
    cfg, actual = load(source / 'registration.json'), load(source / 'receipt.json')
    prior = Path(previous_return or cfg['prior_method_review'])
    ledger_ref = ref(source / 'receipt.json', 'remaining_compute')
    available = actual['remaining_compute']
    original_ledger = {'receipt_ref': ledger_ref, 'remaining_compute': available}
    if allocation_ref is not None:
        allocation = read_method_reference(allocation_ref, source)
        if allocation['source_return_ref'] != ref(source / 'receipt.json') or previous_return:
            raise ValueError('Use the registered allocation for this first ordinary return only')
        available = allocation['maximum_compute']
        ledger_ref = {**allocation_ref, 'json_path': [*allocation_ref.get('json_path', []), 'maximum_compute']}
    if previous_return:
        previous = load(prior / 'receipt.json')
        if (previous['status'] != 'scoped_unknown'
                or previous['source_return_ref'] != ref(source / 'receipt.json')):
            raise ValueError('Resume the unfinished return of this same completed research')
        ledger_ref = ref(prior / 'receipt.json', 'remaining_compute')
        available = previous['remaining_compute']
    prior_topic_ref = ref(prior / 'common/questions.json')
    if Path(prior_topic_ref['file']).exists():
        previous_questions = load(prior_topic_ref['file'])
    else:
        # Historical all-reused returns saved the input topic and paid result,
        # but omitted their final question checkpoint. Recover that retained
        # topic here; the completed analyses below remain referenced, not rerun.
        prior_topic_ref = ref(prior / 'previous_topic.json', 'questions')
        previous_questions = read_method_reference(prior_topic_ref, prior)
    owner = deepcopy(next(q for q in previous_questions if q['id'] == 'task:method_choice'))
    limit = min(maximum_compute, available)
    if limit < 2 + len(owner['method_use_plan']['native_return_refs']):
        raise ValueError('The remaining account must fund ordinary return review')
    for field in ('routing_execution', 'method_use_result', 'completed_agenda_basis'):
        owner.pop(field, None)
    # Sequential model returns do not change the old method pairs. Address the
    # existing completed condition result instead of refitting it on every visit.
    condition_refs = owner['method_use_plan'].setdefault('paired_condition_research_refs', [])
    for reviewed in dict.fromkeys((Path(cfg['prior_method_review']), prior)):
        if not (reviewed / 'receipt.json').exists():
            continue
        previous_receipt = load(reviewed / 'receipt.json')
        action = previous_receipt.get('return', {}).get('research_result', {}).get('method_action_research', {})
        for index, learned in enumerate(action.get('paired_condition_research', [])):
            if learned.get('budget_incomplete'):
                continue
            location = ref(reviewed / 'receipt.json', 'return', 'research_result',
                'method_action_research', 'paired_condition_research', index)
            if location not in condition_refs:
                condition_refs.append(location)
    mission = {'purpose': 'method_learning', 'mission_id': 'paid_parent_revision:' + str(source),
        'allocation': {'compute': limit}, 'resource_usage': {'compute': 0}}
    owner['mission'] = mission
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'previous_topic.json', {'question': owner, 'questions': [owner], 'methods': []})
    save(output / 'registration.json', {'source': str(source), 'maximum_compute': limit,
        'ledger_ref': ledger_ref, 'prior_method_review': str(prior),
        **({'allocation_ref': allocation_ref, 'original_ledger': original_ledger} if allocation_ref else {})})
    common = CommonObjectDependencyMethods(output / 'common', [], budget=SearchBudget(limit), actions=[]).common
    ensure_python_method_topic(common, parent_question_ref={'question_id': 'task'},
        topic_ref=ref(output / 'previous_topic.json'))
    evidence_ref = previous.get('revision_evidence_ref') if previous_return else None
    before_ref = actual.get('parent_ref', cfg.get('parent_ref'))
    # Earlier completed interval returns omitted the producer's two execution
    # fields. Address the actual zero-fit receipt without rewriting that run;
    # the ordinary method consumer still checks the before/after fitted family.
    if evidence_ref is not None:
        retained_evidence = read_method_reference(evidence_ref, prior)
        if retained_evidence.get('method_execution_ref'):
            actual = {**actual, 'method_execution_ref': retained_evidence['method_execution_ref']}
    elif (cfg.get('research_action') == 'paid_interval_uncertainty_update'
            and actual.get('new_fits') == 0 and actual.get('method_execution_ref')):
        raw_ref = actual['method_execution_ref']
        execution = read_method_reference(raw_ref, source)
        if 'fixed_readout_only' not in execution and 'new_fits' not in execution:
            recovered_ref = ref(output / 'interval_execution_return.json')
            save(Path(recovered_ref['file']), {**execution,
                'fixed_readout_only': True, 'new_fits': 0,
                'raw_execution_ref': raw_ref, 'native_result_ref': ref(source / 'receipt.json')})
            actual = {**actual, 'method_execution_ref': recovered_ref}
    if actual.get('full_comparison') and actual['selected_ref'] != before_ref and evidence_ref is None:
        after = retained_entry(actual['selected_ref'], source, 0)
        if after.get('research_program'):
            evidence_ref = ref(output / 'revision_evidence.json')
            save(Path(evidence_ref['file']), {'source_task_ref': cfg['source_task_ref'],
                'before_entry_ref': before_ref, 'after_entry_ref': actual['selected_ref'],
                'source_operation': cfg['research_action'], 'comparison_ref': actual['comparison_ref'],
                'cost_refs': [ref(source / 'receipt.json')], 'data_role': 'exposed_development',
                'independent_support': False,
                **{key: actual[key] for key in ('executed_method_ref', 'method_execution_ref',
                    'research_parent_ref') if key in actual}})
    feedback = record_native_parent_research_return(common, source_task_ref=cfg['source_task_ref'],
        result_ref=ref(source / 'receipt.json'),
        before_ref=actual.get('before_frontier_ref', cfg.get('frontier_ref', cfg.get('source_frontier_ref'))),
        comparison_ref=actual.get('comparison_ref'), revision_evidence_ref=evidence_ref,
        confirmation_ref=actual.get('confirmation_ref'),
        method_refs=([actual.get('executed_method_ref', cfg.get('executed_method_ref'))]
            if actual.get('method_invoked') else []),
        cost_refs=[ref(source / 'receipt.json')], mission=mission)
    try:
        reviewed = consume_method_return_agenda(common, feedback, None, compute_limit=limit)
    except BaseException as error:
        common._save('questions.json', json.dumps(common.questions, separators=(',', ':')))
        save(output / 'failure.json', {'error': repr(error),
            'compute_cost': common.costs['compute_cost'],
            'remaining_compute': available - common.costs['compute_cost'],
            'ledger_ref': ledger_ref, 'source_return_ref': ref(source / 'receipt.json')})
        raise
    common._save('questions.json', json.dumps(common.questions, separators=(',', ':')))
    spent = common.costs['compute_cost']
    result = {'status': reviewed['status'], 'return': reviewed, 'feedback': feedback,
        'source_task_ref': cfg['source_task_ref'], 'revision_evidence_ref': evidence_ref,
        'continued_topic_ref': ref(output / 'common/questions.json'),
        'prior_topic_ref': prior_topic_ref, 'source_return_ref': ref(source / 'receipt.json'),
        'compute_cost': spent, 'new_compute_cost': spent, 'remaining_compute': available - spent,
        'ledger_ref': ledger_ref,
        'new_fits': 0, 'new_predictions': 0, 'new_observations': 0, 'model_calls': 0,
        'method_pair_created': False, 'independent_support': False,
        **({'allocation_ref': allocation_ref, 'original_ledger': original_ledger} if allocation_ref else {}),
        **{key: actual[key] for key in ('method_execution_ref', 'executed_method_ref', 'method_invoked') if key in actual}}
    save(output / 'receipt.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'prepare-relation', 'run-relation',
        'prepare-structure', 'run-structure', 'return-choice', 'prepare-parent', 'run-parent', 'return-parent',
        'prepare-parameter', 'run-parameter', 'prepare-confirm', 'run-confirm', 'prepare-interval', 'run-interval',
        'prepare-instrument-plan', 'run-instrument-plan',
        'prepare-instrument-probe', 'run-instrument-probe',
        'prepare-response-plan', 'run-response-plan',
        'prepare-interval-probe-return', 'run-interval-probe-return',
        'prepare-parent-method-pair', 'prepare-frozen-method-development'))
    parser.add_argument('--request', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--previous-return', type=Path)
    parser.add_argument('--maximum-compute', type=int, default=3000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'prepare-interval-probe-return':
        result = register_interval_probe_return(args.source, args.output, request=args.request)
    elif args.mode == 'run-interval-probe-return':
        result = run_interval_probe_return(args.output)
    elif args.mode == 'prepare-response-plan':
        result = register_response_plan(args.source, args.output, request=args.request)
    elif args.mode == 'run-response-plan':
        result = run_response_plan(args.output)
    elif args.mode == 'prepare-instrument-probe':
        result = register_instrument_probe(args.request, args.output)
    elif args.mode == 'run-instrument-probe':
        result = run_instrument_probe(args.output)
    elif args.mode == 'prepare-instrument-plan':
        result = register_instrument_plan(args.request, args.output)
    elif args.mode == 'run-instrument-plan':
        result = run_instrument_plan(args.output)
    elif args.mode == 'prepare-frozen-method-development':
        result = register_frozen_method_development(args.request, args.output)
    elif args.mode == 'prepare-parent-method-pair':
        result = register_parent_method_pair(args.request, args.output)
    elif args.mode == 'prepare-interval':
        result = register_interval(args.source, args.output, request=args.request)
    elif args.mode == 'run-interval':
        result = run_interval(args.output)
    elif args.mode == 'prepare-confirm':
        result = register_confirmation(args.source, args.output, request=args.request)
    elif args.mode == 'run-confirm':
        result = run_confirmation(args.output)
    elif args.mode == 'prepare-parameter':
        result = register_parameter(args.request, args.output)
    elif args.mode == 'run-parameter':
        result = run_parent(args.output)
    elif args.mode == 'prepare-parent':
        result = register_parent(args.request, args.output)
    elif args.mode == 'run-parent':
        result = run_parent(args.output)
    elif args.mode == 'return-parent':
        result = return_parent(args.source, args.output, args.maximum_compute,
            previous_return=args.previous_return,
            allocation_ref=load(args.request)['allocation_ref'] if args.request else None)
    elif args.mode == 'return-choice':
        result = return_choice(args.request, args.output)
    elif args.mode == 'prepare-structure':
        result = register_structure(args.request, args.output)
    elif args.mode == 'run-structure':
        result = run_structure(args.output)
    elif args.mode in ('prepare', 'prepare-relation'):
        result = (register if args.mode == 'prepare' else register_relation)(args.request, args.output)
    else:
        result = (run if args.mode == 'run' else run_relation)(args.output)
    print(result['status'] if 'status' in result else 'registered_paid_pool')
