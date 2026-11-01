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


def prepare_paid(sources, method_ref, allocation_ref, output):
    """Register two exposed development calls without touching their old ledgers."""
    output = Path(output)
    method = read_method_reference(method_ref, output)
    staged = method.get('staged_research') or {}
    if staged.get('operation') != 'refit_then_reidentify':
        raise ValueError('Freeze an actual generated staged research method')
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
        scene = paid_scene(source)
        if any(staged[side] != scene['methods'][arm]
               for side, arm in (('baseline', 'P0'), ('continuation', 'Pi1'))):
            raise ValueError('The generated method must retain the actual same frozen stages')
        records.append({'source': str(source), 'source_task_ref': scene['cfg']['source_task_ref'],
            'prepared_state_ref': ref(source / 'prepared_state.json'),
            'pool_ref': ref(source / 'common_comparison.json'),
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
            scene = paid_scene(registered['source'])
            budget = SearchBudget(registered['maximum_compute'])
            directory = output / f'source_{i:03d}'
            try:
                with timing.phase(str(i)), python_execution_deadline(cfg['active_wall_seconds'], activity=True):
                    entry = replay_research_program(method, scene['training'], budget,
                        feedback_rows=scene['working'], tolerance=scene['cfg']['quality_threshold'],
                        risk_quantile=scene['frame']['spec'].risk_quantile,
                        paid_stages=scene['paid_stages'])
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
                        'budget_denied': bool(budget.denied_requests), 'actions': actions,
                        'paid_stage_refs': registered['stage_evidence'],
                        'new_observations': 0, 'independent_support': False}
                    save(directory / 'receipt.json', result)
                    results.append(result)
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
    parser.add_argument('mode', choices=('revise-paid', 'prepare-paid', 'run-paid', 'prepare-successor', 'run-successor'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sources', type=Path, nargs='+')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--paid', type=Path)
    parser.add_argument('--prior-review', type=Path)
    parser.add_argument('--method-ref', type=Path, help='Small JSON file containing a method reference')
    parser.add_argument('--allocation-ref', type=Path, help='Small JSON file containing an allocation reference')
    args = parser.parse_args()
    if args.mode == 'revise-paid':
        from scripts.return_partition_pair import merge_pair_returns
        retained = load(args.prior_review / 'receipt.json')
        result = merge_pair_returns(args.prior_review, args.output,
            retained['merged_pair_refs'], load(args.allocation_ref))
    elif args.mode == 'prepare-paid':
        result = prepare_paid(args.sources, load(args.method_ref), load(args.allocation_ref), args.output)
    elif args.mode == 'run-paid':
        result = run_paid(args.output)
    elif args.mode == 'prepare-successor':
        result = prepare_successor(args.paid, args.source, load(args.allocation_ref), args.prior_review, args.output)
    else:
        from scripts.validate_partition_successor import run
        result = run(args.output)
    print({k: v for k, v in result.items() if k in ('status', 'compute_cost', 'new_compute_cost')})
