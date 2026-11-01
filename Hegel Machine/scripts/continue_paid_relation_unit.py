#!/usr/bin/env python3
"""Run one preregistered, bounded interval-research unit through existing entries."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts import review_paid_object_pool as review
from scripts.run_native_decision_pair import ref
from scripts.run_partial_role_parent import load, save
from scripts.validate_partition_successor import freeze_entry
from hegel_machine.cognition.checkpoint_storage import write_text_incrementally


ROOT = Path(__file__).resolve().parents[1]
RETURN_RESERVE = 6000
CONFIRM_COMPUTE = 96
INTERVAL_RESERVE = 6000


def register(request, output):
    request, output = Path(request), Path(output)
    cfg = load(request)
    source = Path(cfg['source'])
    actual = load(source / 'receipt.json')
    costs = review.read_method_reference(cfg['source_costs_ref'], source)
    protocol = review.read_method_reference(cfg['budget_protocol_ref'], source)
    upper = costs['cumulative_protocol_compute_interval'][1]
    held = (costs['parent_separate_suballocation_remaining']
        + costs['diagnostic_suballocation_remaining']
        + sum(costs['previous_accounts_unchanged'].values())
        + costs.get('complete_relation_unit_remaining', 0))
    available = protocol['authorized_limits']['compute_cost'] - upper - held
    if not 0 < cfg['maximum_compute'] <= available:
        raise ValueError('Keep every registered balance inside the original total authorization')
    if (actual['status'] != 'paid_interval_updated' or not actual['execution_complete']
            or not actual['full_comparison']):
        raise ValueError('Start from a completed paid interval and full retained pool')
    if cfg['maximum_observations'] < 5 * cfg['maximum_experiments'] + 48:
        raise ValueError('Reserve the bounded probes and one original-size confirmation')
    if costs['cumulative_protocol_observations'] + cfg['maximum_observations'] > protocol['authorized_limits']['observation_cost']:
        raise ValueError('Keep the original observation total')
    panel = cfg['confirmation_protocol']
    if panel['length'] != 48 or panel['trajectories'] != 1 or panel['maximum_attempts'] != 1:
        raise ValueError('Keep one preregistered original-size panel, without replacement')
    cfg.update(unallocated_before=available, preserved_balances=held,
        initial_selected_ref=actual['selected_ref'],
        original_science_remaining=actual['remaining_compute'],
        return_compute_reserve=RETURN_RESERVE, confirmation_compute=CONFIRM_COMPUTE,
        interval_compute_reserve=INTERVAL_RESERVE, model_calls=0)
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'registration.json', cfg)
    freeze_entry(output, source)
    for name in (Path(__file__).name, 'review_paid_object_pool.py', 'continue_successor_relation_intervals.py'):
        relative = Path('scripts') / name
        base = source / 'runtime' / relative
        write_text_incrementally(output / 'runtime' / relative, (ROOT / relative).read_text(),
            base=base if base.exists() else None)
    return cfg


def next_action(receipt, steps, maximum_steps, remaining, pool_size):
    """Admit a whole next segment; unchanged scientific gates own confirmation."""
    if receipt['confirmation_gate']['admitted']:
        return 'confirm'
    if steps >= maximum_steps:
        return 'experiment_limit'
    # Native planning cap + at most five rows for the pool and instruments,
    # then the existing interval pass and the two ordinary returns/readout.
    required = 6000 + 5 * (pool_size + 64) + INTERVAL_RESERVE + RETURN_RESERVE + CONFIRM_COMPUTE
    return 'research' if remaining >= required else 'complete_segment_unfunded'


def run(output):
    output = Path(output)
    if (output / 'receipt.json').exists():
        return load(output / 'receipt.json')
    if ROOT != (output / 'runtime').resolve() or (output / 'started.json').exists():
        raise ValueError('Run the frozen complete unit once; preserve interrupted attempts')
    cfg = load(output / 'registration.json')
    source = Path(cfg['source'])
    costs = review.read_method_reference(cfg['source_costs_ref'], source)
    spent = observations = measurements = 0
    phases = []
    current = source
    last_research = None
    method_review = cfg['prior_method_review']
    steps = 0
    began = time.monotonic()
    competition_ref = None
    storage = os.statvfs(output)
    save(output / 'started.json', {'registration_ref': ref(output / 'registration.json'),
        'mount_used_bytes': (storage.f_blocks - storage.f_bfree) * storage.f_frsize})

    def account():
        number = len(phases)
        ledger = output / f'ledger_{number:03d}.json'
        save(ledger, {'remaining_compute': cfg['maximum_compute'] - spent,
            'allocation_ref': ref(output / 'registration.json'), 'phase_refs': phases})
        snapshot = output / f'costs_{number:03d}.json'
        save(snapshot, {**costs, 'previous_source_costs_ref': cfg['source_costs_ref'],
            'cumulative_protocol_compute_interval': [v + spent for v in costs['cumulative_protocol_compute_interval']],
            'cumulative_protocol_observations': costs['cumulative_protocol_observations'] + observations,
            'known_compute_cost': spent, 'new_observations': observations,
            'measurement_cost': measurements, 'research_cost_refs': phases,
            'complete_relation_unit_remaining': cfg['maximum_compute'] - spent,
            'complete_relation_allocation_ref': ref(output / 'registration.json')})
        return {'ledger_ref': ref(ledger, 'remaining_compute'), 'source_costs_ref': ref(snapshot),
            'source_cost_refs': [cfg['source_costs_ref'], *phases],
            'observation_limit_ref': {**cfg['budget_protocol_ref'], 'json_path': ['authorized_limits', 'observation_cost']},
            'prior_method_review': method_review}

    def execute(path, mode):
        nonlocal spent, observations, measurements
        remaining_wall = cfg['active_wall_seconds'] - (time.monotonic() - began)
        if remaining_wall <= 0:
            raise RuntimeError('Complete-unit activity deadline reached before this phase')
        registered = load(path / 'registration.json')
        registered['active_wall_seconds'] = min(registered['active_wall_seconds'], remaining_wall)
        save(path / 'registration.json', registered)
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONPATH': os.pathsep.join(str(path / 'runtime' / part) for part in ('src', '.', 'scripts'))}
        with (path / 'process.log').open('a') as log:
            result = subprocess.run([sys.executable, '-B', '-u',
                str(path / 'runtime/scripts/review_paid_object_pool.py'), mode, '--output', str(path)],
                cwd=path / 'runtime', env=env, stdout=log, stderr=subprocess.STDOUT)
        receipt_path = path / ('failure.json' if result.returncode else 'receipt.json')
        receipt = load(receipt_path)
        charged = receipt.get('actual_costs') if result.returncode else receipt
        if charged is None:
            charged = receipt
        if 'compute_cost' not in charged:
            raise RuntimeError(f'Unresolved failed cost: {receipt_path}')
        spent += charged['compute_cost']
        observations += charged.get('observation_cost', 0)
        measurements += charged.get('measurement_cost', 0)
        phases.append(ref(receipt_path))
        account()
        if result.returncode:
            raise RuntimeError(f'Preserved failed phase: {receipt_path}')
        return receipt

    def ordinary_return(where, name):
        nonlocal spent, method_review
        allocation = output / f'{name}_allocation.json'
        save(allocation, {'source_return_ref': ref(where / 'receipt.json'),
            'maximum_compute': min(3000, cfg['maximum_compute'] - spent),
            'parent_allocation_ref': ref(output / 'registration.json'),
            'ledger_ref': account()['ledger_ref']})
        target = output / name
        # The latest method topic is already threaded through each registered phase.
        try:
            receipt = review.return_parent(where, target, allocation_ref=ref(allocation))
        except Exception:
            if (target / 'failure.json').exists():
                failed = load(target / 'failure.json')
                spent += failed['compute_cost']
                phases.append(ref(target / 'failure.json'))
                account()
            raise
        spent += receipt['compute_cost']
        phases.append(ref(target / 'receipt.json'))
        method_review = str(target)
        account()

    status, confirmation = 'unresolved', None
    try:
        while True:
            if time.monotonic() - began >= cfg['active_wall_seconds']:
                status = 'activity_deadline'
                break
            if os.statvfs(output).f_bavail * os.statvfs(output).f_frsize < 512 * 1024 ** 2:
                status = 'disk_reserve'
                break
            actual = load(current / 'receipt.json')
            pool = load(current / 'paid_pool/receipt.json')
            decision = next_action(actual, steps, cfg['maximum_experiments'],
                cfg['maximum_compute'] - spent, pool['unique_programs'])
            if decision != 'research':
                status = decision
                break
            stage = output / f'step_{steps:03d}'
            stage.mkdir()
            request = stage / 'plan_request.json'
            save(request, {**account(), 'reserved_compute': INTERVAL_RESERVE + RETURN_RESERVE
                + CONFIRM_COMPUTE + 5 * (pool['unique_programs'] + 64)})
            plan = stage / 'plan'
            review.register_response_plan(current, plan, request=request)
            planned = execute(plan, 'run-response-plan')
            last_research = plan
            if planned['forecast_status'] == 'algebraically_shared':
                competition_ref = ref(plan / 'plan.json')
                status = 'confirm'
                break
            if planned['status'] != 'response_distinguishing_experiment':
                status = 'no_executable_distinction'
                break
            request = stage / 'probe_request.json'
            sequence = load(plan / 'plan.json')['probe']['sequence']
            if len(sequence) > 5 or observations + len(sequence) + 48 > cfg['maximum_observations']:
                status = 'observation_reserve'
                break
            save(request, {**account(), 'source': str(plan),
                'reserved_compute': INTERVAL_RESERVE + RETURN_RESERVE + CONFIRM_COMPUTE})
            probe = stage / 'probe'
            review.register_instrument_probe(request, probe)
            observed = execute(probe, 'run-instrument-probe')
            last_research = probe
            if observed['status'] != 'response_distinguished':
                status = 'response_or_instrument_mismatch'
                break
            request = stage / 'interval_request.json'
            save(request, {'partition_source': str(current),
                'reserved_compute': RETURN_RESERVE + CONFIRM_COMPUTE})
            interval = stage / 'interval'
            review.register_interval_probe_return(probe, interval, request=request)
            updated = execute(interval, 'run-interval-probe-return')
            last_research = interval
            current, steps = interval, steps + 1
            save(output / 'progress.json', {'steps': steps, 'current_source': str(current),
                'compute_cost': spent, 'observation_cost': observations,
                'confirmation_gate': updated['confirmation_gate'], 'phase_refs': phases})
            if updated['status'] != 'paid_interval_updated':
                status = 'interval_update_incomplete'
                break
        if last_research is not None:
            ordinary_return(last_research, 'method_return')
        if status == 'confirm':
            request = output / 'confirmation_request.json'
            save(request, {**account(),
                'confirmation_protocol_ref': ref(output / 'registration.json', 'confirmation_protocol'),
                'incumbent_ref': cfg['initial_selected_ref'],
                **({'competition_review_ref': competition_ref} if competition_ref else {})})
            target = output / 'confirmation'
            review.register_confirmation(current, target, request=request)
            confirmation = execute(target, 'run-confirm')
            status = confirmation['status']
            ordinary_return(target, 'confirmation_return')
    except Exception as error:
        status = 'execution_failed'
        save(output / 'failure.json', {'error': repr(error), 'phase_refs': phases,
            'compute_cost': spent, 'observation_cost': observations})
    closed = account()
    storage = os.statvfs(output)
    result = {'status': status, 'steps': steps, 'current_source': str(current),
        'latest_method_review': method_review, 'confirmation': confirmation,
        'compute_cost': spent, 'observation_cost': observations, 'measurement_cost': measurements,
        'remaining_compute': cfg['maximum_compute'] - spent, 'phase_refs': phases,
        'closed_source_costs_ref': closed['source_costs_ref'],
        'wall_seconds': time.monotonic() - began,
        'model_calls': 0, 'independent_task_support': False,
        'mount_used_bytes': (storage.f_blocks - storage.f_bfree) * storage.f_frsize}
    save(output / 'receipt.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'run'))
    parser.add_argument('--request', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = register(args.request, args.output) if args.mode == 'register' else run(args.output)
    print(result.get('status', 'registered'))
