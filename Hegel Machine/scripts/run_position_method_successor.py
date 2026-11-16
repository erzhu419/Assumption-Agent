"""Register one fresh position-method ABC source and settle only its new costs.

The preceding ABC program bank, library and raw materials are shared. The
existing successor runner performs ordinary research and the frozen terminal.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

from run_partial_role_successors import ARMS, HistoryStore, load, program_successor_protocol, save
from run_learned_structure_successor import freeze_runtime
from hegel_machine.cognition.checkpoint_storage import read_json_checkpoint, write_text_incrementally
from hegel_machine.cognition.method_action_research import read_method_reference


def protocol(prior):
    original, = prior['tasks']
    if original['seed'] != 106110602 or original['objects'] != 5:
        raise ValueError('Expected the frozen five-object source 602')
    seed = original['seed'] + 1
    new = {**original, 'seed': seed, 'apparatus_seed': seed, 'name_seed': seed + 63857,
           'wiring_seed': seed + 3709}
    limits = {**prior['limits'], 'compute_cost': 200000, 'observation_cost': 256}
    declared = {**prior, 'scope': 'independent_position_method_native_ABC',
        'tasks': [new], 'factor_cases': [{'factor': 'fresh_physics_and_anonymous_apparatus', 'task': new}],
        'nuisance_contract': 'Fresh physical coefficients, sensor names and wiring; five objects and the same relation family.',
        'limits': limits, 'research_config': {**prior['research_config'], 'seed': seed,
            'observation_budget': limits['observation_cost'], 'compute_budget': limits['compute_cost']}}
    return program_successor_protocol(original, declared)


def source_manifest(prior_root, formation, raw, ledger):
    manifest = load(prior_root / 'inputs/source_costs.json')
    # Include the actual 602 history used by every arm as shared source expense.
    # Each receipt is an occurrence; no cumulative ledger is a new expense.
    shared = list(ARMS)
    def add(ref, costs, arms, label, failed=False, path=()):
        address = {'file': str(ref), **({'json_path': list(path)} if path else {})}
        manifest['occurrences'].append({'occurrence_ref': address,
            'scope': 'incremental_execution', 'phase': 'source', 'charged_to': arms,
            'failed': failed, 'label': label, 'costs': {**costs, 'model_calls': 0, 'reported_tokens': 0}})
    recover = prior_root.parent / 'secondary_cost_recovery.json'
    recovered = next(x for x in load(recover)['arms'] if x['arm'] == 'pr3')
    paid = prior_root / 'runs/106110602/pr3/summary.json'
    old = load(paid)['costs']
    add(paid, {**old, 'compute_cost': recovered['recovered_compute_cost']}, shared,
        'shared_paid_602_parent', failed=True)
    history = raw['previous']['position_history_refs']
    squares = [Path(item['completed_relation']['file']).parent for item in history]
    first = Path(load(squares[0] / 'registration.json')['previous'])
    for unit in (first, *squares):
        receipt = load(unit / 'receipt.json')
        measured = receipt.get('laboratory_costs', {})
        add(unit / 'receipt.json', {'compute_cost': receipt['compute_cost'],
            'observation_cost': measured.get('observation_cost', receipt.get('new_observations', 0)),
            'measurement_cost': measured.get('measurement_cost', 0)}, shared, 'paid_source_research')
    formed = load(formation / 'receipt.json')
    costs = [(formed['formation_compute'], ['pr3'], ('formation_compute',), 'freeze_position_method')]
    costs.extend((formed['arms'][arm]['compute'], [arm], ('arms', arm, 'compute'), 'source_reconstruction_acceptance')
                 for arm in ARMS if formed['arms'][arm]['compute'])
    if sum(count for count, *_ in costs) != ledger['new_compute_cost']:
        raise ValueError('Account the complete method formation and source reconstruction cost')
    for count, arms, path, label in costs:
        add(formation / 'receipt.json', {'compute_cost': count, 'observation_cost': 0, 'measurement_cost': 0},
            arms, label, path=path)
    manifest['position_source_accounting'] = {'recovered_compute_ref': {'file': str(recover)},
        'scope': 'All arms receive the raw research. Freeze and source-side reconstruction controls have separate receipts by arm. Earlier upstream uncertainties retained.'}
    return manifest


def register(args):
    prior_root, formation, output = (path.resolve() for path in (args.previous, args.formation, args.output))
    prior = load(prior_root / 'registration.json')
    raw_path = (args.raw_source or formation / 'raw_source.json').resolve()
    method_path = (args.method or formation / 'method.json').resolve()
    ledger_path = (args.ledger or formation / 'closed_source_costs.json').resolve()
    raw, method, acceptance, ledger = (load(path) for path in
        (raw_path, method_path, formation / 'receipt.json', ledger_path))
    if (not acceptance.get('same_body_reconstructed') or not method.get('position_method')
            or method['source_ref'] != raw['source_ref'] or method['model_id'] != raw['model_id']):
        raise ValueError('Require the measured position method and its accepted ordinary reconstruction')
    plan = protocol(prior)
    quote = plan['budget_quote']
    if (quote['total_new_compute_upper'] > ledger['abc_remaining_compute']
            or quote['total_new_observation_upper'] > ledger['abc_remaining_observations']):
        raise ValueError('The full ABC training and terminal exceed the open ABC account')
    seed = plan['tasks'][0]['seed']
    recent = [*prior['recent_source_task_refs'],
              {'file': str(prior_root / 'registration.json'), 'json_path': ['tasks', 0]}]
    store = HistoryStore(prior['historical_knowledge_ref']['store'], readonly=True)
    try:
        old = store.connection.execute('SELECT unit FROM entries WHERE unit LIKE ? LIMIT 1',
                                       ('%' + str(seed) + '%',)).fetchone()
    finally:
        store.close()
    if old or any(read_method_reference(ref, prior_root)['seed'] == seed for ref in recent):
        raise ValueError('The prospective physical source has already appeared in the declared history')
    manifest = source_manifest(prior_root, formation, raw, ledger)
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / 'inputs'
    inputs.mkdir()
    # Frozen data in the same experiment store; retain the actual original bytes.
    for name in ('program_source', 'program_method', 'method_conditions', 'raw_source', 'method'):
        os.link(prior_root / 'inputs' / (name + '.json'), inputs / (name + '.json'))
    save(inputs / 'source_costs.json', manifest)
    freeze_runtime(output, prior_root)
    write_text_incrementally(output / 'runtime/scripts' / Path(__file__).name, Path(__file__).read_text())
    save(output / 'registration.json', {**prior, **plan, 'registered_at': datetime.now(timezone.utc).isoformat(),
        'current_budget_admitted': True, 'recent_source_task_refs': recent, 'unexposed_check': {str(seed): []},
        'source_costs_ref': {'file': 'inputs/source_costs.json'}, 'budget_contract_ref': {'file': str(ledger_path)},
        'successor_protocol_ref': {'file': str(output / 'registration.json'), 'json_path': ['tasks']},
        'previous_protocol_ref': {'file': str(prior_root / 'registration.json')},
        'position_method_source': {'raw_source_ref': {'file': str(raw_path)}, 'method_ref': {'file': str(method_path)},
            'acceptance_ref': {'file': str(formation / 'receipt.json')}},
        'contrast': {'archive': 'ordinary adaptive archive with all preceding methods and same raw position materials',
            'fixed_pr2': 'ordinary Common with preceding methods and same raw position materials',
            'pr3': 'ordinary Common with inherited program bank and the frozen position method'},
        'role_endpoint': 'Record actual target-side rebinding, distinct action and measured narrowing by arm. Lack of activation is a completed negative use result. Compare full predictors only at the unchanged terminal.'})
    return plan


def spent_interval(outcomes, terminal, key, ceiling):
    low = high = terminal[key]
    for row in outcomes:
        value = row['costs'][key]
        if value is None:
            low += row.get('confirmed_compute_lower_bound', 0) if key == 'compute_cost' else 0
            high += row.get('unknown_upper_bounds', {}).get(key, ceiling)
        else:
            low += value
            high += value
    return [low, high]


def position_usage(directory, arm, source):
    method = load(source['method_ref']['file'])
    report = directory / 'response_report.json'
    if not report.exists():
        report = directory / 'common/failed_paid_state.json'
    actions = read_json_checkpoint(report, keys={'actions'}).get('actions') if report.exists() else None
    events = actions if actions is not None else []
    reconstructed = [a for a in events if a['action'] == 'reconstruct_paid_object_method'
                     and a.get('source_ref') == method['source_ref']]
    tested = [a for a in events if a['action'] == 'test_object_dependency_question'
              and a['question'].get('position_experiment')
              and a['question'].get('source_object_model_id') == method['model_id']]
    choices, returns = {}, {}
    for a in events:
        if a['action'] == 'apply_position_owner_method' and a.get('source_ref') == method['source_ref']:
            choices[a['position_first_return_ref']['file']] = a['position_first_return_ref']
    first_stages = set()
    paths = [*directory.glob('common/object_question_*/consumption_*.json'),
             *directory.glob('common/native_position_returns/return_*.json')]
    for path in sorted(paths):
        obj = read_json_checkpoint(path, keys={'position_method_application','position_first_return_ref',
                                                'position_experiment','completed_relation'})
        plan = obj.get('position_experiment') or {}
        measured = obj.get('completed_relation') or {}
        if not obj.get('position_method_application'):
            if plan and measured.get('status') == 'position_domain_narrowed':
                first_stages.add((plan['field'], plan['excited_quantity'], plan['baseline_ref']['probe_id']))
            continue
        if obj['position_method_application']['source_ref'] != method['source_ref']:
            continue
        first = obj['position_first_return_ref']['file']
        choices.setdefault(first, {'file': str(path)})
        if measured:
            returns[first] = {'file': str(path), 'status': measured['status'],
                'field': measured['field'], 'retained_owners': measured.get('retained_owners')}
    # Test events follow acquisition and carry experiment/source identity. Avoid
    # mistaking planned actions or purchases from the first square for method use.
    def count(value):
        return value if actions is not None or value else None
    return {'frozen_method_supplied': arm == 'pr3', 'raw_source_supplied': True,
            'action_trace_ref': str(report) if actions is not None else None,
            'source_reconstruction_count': count(len(reconstructed)),
            'successful_source_reconstructions': count(sum(bool(a.get('method_available')) for a in reconstructed)),
            'first_squares_narrowing': count(len(first_stages)),
            'bound_target_questions': count(len(choices)), 'application_refs': list(choices.values()),
            'acknowledged_purchases': count(len(tested)),
            'purchased_rows': count(sum(a['observation_rows'] for a in tested)),
            'completed_method_squares': count(len(returns)),
            'narrowed_by_method': count(sum(r['status'] == 'position_domain_narrowed' for r in returns.values())),
            'completed_relation_refs': list(returns.values())}


def close(output):
    reg, result = (load(output / (name + '.json')) for name in ('registration', 'summary'))
    before = load(reg['budget_contract_ref']['file'])
    seed = reg['tasks'][0]['seed']
    rows = [load(output / 'runs' / str(seed) / arm / 'summary.json') for arm in ARMS]
    if not all(row['frozen'] for row in rows):
        raise ValueError('All three final outcomes must be frozen before closure')
    terminal, = result['paired_terminal']
    new = {key: spent_interval(rows, terminal['shared_actual_costs'], key, cap)
           for key, cap in reg['limits'].items()}
    comp, obs, measure = (new[key] for key in ('compute_cost', 'observation_cost', 'measurement_cost'))
    if comp[1] > reg['budget_quote']['total_new_compute_upper'] or obs[1] > reg['budget_quote']['total_new_observation_upper']:
        raise ValueError('The measured or reserved upper bound exceeds the registration')
    bounds = before['abc_remaining_compute_interval']
    available = [bounds[0] - comp[1], bounds[1] - comp[0]]
    prior_rows = before.get('cumulative_protocol_observation_interval',
                            [before['cumulative_protocol_observations']] * 2)
    total_rows = [a+b for a,b in zip(prior_rows, obs)]
    cumulative = [a+b for a,b in zip(before['cumulative_protocol_compute_interval'], comp)]
    costs = {**before, 'status': 'closed_position_method_successor', 'previous_source_costs_ref': reg['budget_contract_ref'],
        'receipt_ref': {'file': str(output / 'receipt.json')}, 'new_compute_interval': comp,
        'new_compute_cost': comp[0] if comp[0] == comp[1] else None, 'new_observation_interval': obs,
        'new_observations': obs[0] if obs[0] == obs[1] else None, 'new_measurement_interval': measure,
        'new_measurement_cost': measure[0] if measure[0] == measure[1] else None, 'new_model_calls': 0,
        'cumulative_protocol_compute_interval': cumulative, 'cumulative_protocol_observation_interval': total_rows,
        'cumulative_protocol_observations': total_rows[0] if total_rows[0] == total_rows[1] else None,
        'abc_remaining_compute_interval': available, 'abc_remaining_compute': available[0],
        'abc_actual_compute_spent_to_date':before['abc_allocation_limit'] - available[0] if available[0] == available[1] else None,
        'abc_remaining_observations': before['abc_remaining_observations'] - obs[1],
        'other_accounts_inherited_from': reg['budget_contract_ref']['file']}
    usage = {arm: position_usage(output / 'runs' / str(seed) / arm, arm, reg['position_method_source'])
             for arm in ARMS}
    receipt = {'status': 'closed', 'target_seed': seed, 'new_cost_intervals': new,
        'position_method_use': usage,
        'model_calls': 0, 'registration_ref': str(output / 'registration.json'),
        'terminal_ref': str(output / 'summary.json'), 'outcomes': [
            {'arm': r['arm'], 'status': r['status'], **terminal['arms'][r['arm']]} for r in rows]}
    save(output / 'receipt.json', receipt)
    save(output / 'closed_source_costs.json', costs)
    return receipt


def recover_settled(previous, output):
    """Resolve the 603 comparison-hook exception using its frozen parent accounts."""
    reg = load(previous / 'registration.json')
    task, = reg['tasks']
    if task['seed'] != 106110603:
        raise ValueError('This recovery addresses the original 603 comparison failure')
    runtime = previous / 'runtime/src/hegel_machine/cognition'
    unified = (runtime / 'unified_research.py').read_text()
    common = (runtime / 'common_object_research.py').read_text()
    acquisition = unified.index('claims = self._research_view(spec, train_view, validation_view, view_budget')
    settlement = unified.index('finally:', acquisition)
    settled = unified.index('budget.spend(view_budget.spent)', settlement)
    if settled > unified.index('# A new random trajectory', acquisition) or (
            "self.failure_compute_lower_bound = setup_cost + (state['budget'].spent if state else 0)" not in common):
        raise ValueError('The frozen run must settle the active view before recording the parent budget')
    arm_costs = {}
    for arm in ARMS:
        root = previous / 'runs' / str(task['seed']) / arm
        failure = load(root / 'failure.json')
        trace = failure.get('traceback', '')
        if ("assessed.extend(assess(joint))" not in trace or "in _record_program_comparison" not in trace
                or "AttributeError: 'NoneType' object has no attribute 'get'" not in trace):
            raise ValueError('Another exception can leave unreturned child costs; retain its interval')
        parent_ref = failure['paid_state_ref']
        snapshot = read_json_checkpoint(Path(parent_ref['file']), keys={'confirmed_compute_lower_bound'})
        paid = snapshot['confirmed_compute_lower_bound']
        if paid != failure['confirmed_compute_lower_bound']:
            raise ValueError('The failure and its original parent snapshot must agree')
        arm_costs[arm] = {'compute_cost': paid, 'parent_budget_ref': {**parent_ref,
            'json_path': ['confirmed_compute_lower_bound']}, 'failure_ref': {'file': str(root / 'failure.json')}}
    old = load(previous / 'closed_source_costs.json')
    terminal, = load(previous / 'summary.json')['paired_terminal']
    actual = terminal['shared_actual_costs']['compute_cost'] + sum(r['compute_cost'] for r in arm_costs.values())
    low, high = old['new_compute_interval']
    if not low <= actual <= high:
        raise ValueError('Recovered amount must fall in the original paid interval')
    before = load(reg['budget_contract_ref']['file'])
    available = [n - actual for n in before['abc_remaining_compute_interval']]
    cumulative = [n + actual for n in before['cumulative_protocol_compute_interval']]
    lines = lambda text, position: text.count('\n', 0, position) + 1
    receipt = {'status': 'recovered_exact_settled_compute', 'arms': arm_costs,
        'old_receipt_ref': {'file': str(previous / 'receipt.json')}, 'original_interval': [low, high],
        'settled_603_compute': actual, 'new_scientific_compute': 0, 'new_observations': 0,
        'formula': 'setup + outer parent.spent already includes the full active view in finally. Compilation had returned, joint fits and their comparisons charged the view directly; the reporting hook did not spend. No second addition of the view.',
        'frozen_settlement': {'file': str(runtime / 'unified_research.py'), 'line': lines(unified, settled)},
        'frozen_snapshot': {'file': str(runtime / 'common_object_research.py'),
            'line': lines(common, common.index('self.failure_compute_lower_bound = setup_cost +'))}}
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'receipt.json', receipt)
    save(output / 'closed_source_costs.json', {**old,
        'status': 'closed_position_method_successor_reconciled', 'previous_source_costs_ref': {'file': str(previous / 'closed_source_costs.json')},
        'receipt_ref': {'file': str(output / 'receipt.json')}, 'corrected_603_compute_cost': actual,
        'original_successor_receipt_ref': receipt['old_receipt_ref'], 'old_603_compute_interval': [low, high],
        'new_compute_interval': [0, 0], 'new_compute_cost': 0, 'new_observations': 0,
        'new_observation_interval': [0, 0], 'new_measurement_interval': [0, 0], 'new_measurement_cost': 0,
        'abc_remaining_compute_interval': available, 'abc_remaining_compute': available[0],
        'abc_actual_compute_spent_to_date': before['abc_allocation_limit'] - available[0] if available[0] == available[1] else None,
        'cumulative_protocol_compute_interval': cumulative})
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'close', 'recover-settled'))
    for flag in ('previous', 'formation', 'raw-source', 'method', 'ledger', 'output'):
        parser.add_argument('--' + flag, type=Path, required=flag == 'output')
    args = parser.parse_args()
    if args.action == 'register':
        if not args.previous or not args.formation:
            parser.error('registration requires --previous and --formation')
        print(register(args)['budget_quote'])
    elif args.action == 'recover-settled':
        if not args.previous:
            parser.error('recovery requires --previous original frozen 603 root')
        print(recover_settled(args.previous.resolve(), args.output.resolve())['settled_603_compute'])
    else:
        print(close(args.output.resolve())['new_cost_intervals'])
