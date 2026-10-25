#!/usr/bin/env python3
"""Compare common selection of input remeasurement, discrimination and mechanism revision."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/hegel_machine/cognition"))
from checkpoint_storage import write_json_checkpoint
OUTPUT = ROOT / 'artifacts/observation_action_study_20260919'
ARMS = ('baseline_v092', 'learned', 'policy_disabled')
COSTS = ('observation_cost', 'measurement_cost', 'compute_cost', 'queries')
METHOD_EVENTS = ('learn_instrument_experiment_method', 'use_instrument_experiment_method',
                 'revise_instrument_experiment_method')


def save(path, payload):
    write_json_checkpoint(path, payload, indent=2, ensure_ascii=True, allow_nan=True)


def prepare(output):
    if (output / 'protocol.json').exists():
        return
    baseline = ROOT / 'artifacts/instrument_method_study_20260919/runtime/src'
    if not (baseline / 'hegel_machine/cognition/unified_session.py').exists():
        raise FileNotFoundError(baseline)
    output.mkdir(parents=True, exist_ok=True)
    laboratories = {}
    for name in ('ambiguous_instrument_lab', 'dynamic_instrument_lab'):
        destination = output / ('shared_' + name + '.py')
        shutil.copyfile(ROOT / 'src/hegel_machine/cognition' / (name + '.py'), destination)
        laboratories[name] = str(destination.relative_to(ROOT))
    save(output / 'protocol.json', {
        'experiment': 'common_observation_and_mechanism_action_selection',
        'fixed_at': datetime.now(timezone.utc).isoformat(),
        'cohorts': {'development': [95201900], 'main': [95201901, 95201902, 95201903, 95201904]},
        'instrument_modes': {'95201900': 'recurrent', '95201901': 'finite2', '95201902': 'finite2',
                             '95201903': 'recurrent', '95201904': 'recurrent'},
        'development_runtime': 'src',
        'arms': {'baseline_v092': {'runtime': str(baseline.relative_to(ROOT))},
                 'learned': {'runtime': str((output / 'runtime/src').relative_to(ROOT))},
                 'policy_disabled': {'runtime': str((output / 'runtime/src').relative_to(ROOT)),
                                     'research_config_overrides': {'learn_research_policy': False}}},
        'shared_laboratories': laboratories,
        'stages': ['ambiguous_short_source', 'expanded_same_apparatus', 'new_dynamic_apparatus',
                   'unchanged_dynamic_apparatus', 'instrument_only_change', 'mechanism_only_change',
                   'both_return_to_dynamic_source'],
        'public_max_probe_lengths': [3, 24, 96, 96, 96, 96, 96],
        'dynamic_apparatus_seed_offset': 500009,
        'research_config': {'observation_budget': 384, 'compute_budget': 360000,
            'minimum_improvement': 1e-7, 'max_rounds': 5, 'max_history': 4, 'active_probes': True},
        'measurement_budget': 1536, 'maximum_concurrent_research_processes': 1,
        'terminal': {'length': 24, 'trajectories': 4, 'seed_offset': 1000003,
            'impulse_length': 24, 'impulse_amplitudes': [-.85, .85], 'mse_threshold': 1e-6},
        'scope': 'One full stream combines unresolved observation alternatives, a new apparatus, an unchanged task, instrument drift, mechanism drift and return. All arms start empty; no supplied source '
            'program, private parameter, required measurement, diagnostic choice or action score. '
            'Finite-history and direct-control response explanations compete normally. '
            'Development and main cohorts are evaluated separately, after all declared learning in each cohort.',
        'persistence': 'Restore complete UnifiedResearchSession JSON between stages.',
        'cost_accounting': 'Source learning and all unsuccessful research included; terminal resources separate.',
        'capability_endpoints': ['paid_common_action_choices', 'fresh_response_based_action_feedback',
            'json_action_experience_transfer', 'instrument_and_mechanism_recovery', 'no_auxiliary_purchase_when_reuse_suffices'],
        'legal_terminal_domain': 'Source uses length 3; all successors use length 24 for terminal trajectories. All obey their public horizons.',
        'formal_evidence': False})


def freeze(output):
    destination = output / 'runtime/src/hegel_machine'
    if destination.exists():
        raise ValueError('runtime already frozen; preserve it and use a new output for a revised release')
    shutil.copytree(ROOT / 'src/hegel_machine', destination,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))


def load(protocol, arm, cohort):
    source = ROOT / (protocol['development_runtime'] if cohort == 'development' and arm != 'baseline_v092'
                     else protocol['arms'][arm]['runtime'])
    sys.path.insert(0, str(source))
    session = importlib.import_module('hegel_machine.cognition.unified_session')
    if not Path(session.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError('execute each runtime in a separate process')
    classes = {}
    for name, class_name, environment in (
            ('ambiguous_instrument_lab', 'AmbiguousInstrumentLab', 'ambiguous_instruments'),
            ('dynamic_instrument_lab', 'DynamicInstrumentLab', 'dynamic_instruments')):
        canonical = 'hegel_machine.cognition.' + name
        spec = importlib.util.spec_from_file_location(canonical, ROOT / protocol['shared_laboratories'][name])
        laboratory = importlib.util.module_from_spec(spec)
        sys.modules[canonical] = laboratory
        spec.loader.exec_module(laboratory)
        classes[environment] = getattr(laboratory, class_name)
    def factory(item):
        arguments = dict(item)
        return classes[arguments.pop('environment')](**arguments)
    return session, factory


def task_sequence(protocol, seed):
    return [
        {'environment': 'ambiguous_instruments', 'task_index': i, 'seed': seed + 10007 * i,
         'apparatus_seed': seed, 'instrument_mode': protocol['instrument_modes'][str(seed)]}
        for i in range(2)] + [
        {'environment': 'dynamic_instruments', 'task_index': i, 'seed': seed + 10007 * (i + 2),
         'apparatus_seed': seed + protocol['dynamic_apparatus_seed_offset'], 'profile': 'confirmation'}
        for i in range(5)]


def run_path(output, cohort, arm, seed):
    return output / cohort / 'runs' / arm / str(seed) / 'checkpoint.json'


def instrument_snapshot(models):
    return [{key: value for key, value in model.items() if key in ('model_id', 'status', 'support_rows')} | {
        'fields': {name: {key: value for key, value in field.items()
                         if key in ('kind', 'fit_loss', 'observed_history_depth', 'validation_loss')} | {
            'hypotheses': [{key: value for key, value in hypothesis.items() if key != 'program'}
                           for hypothesis in field.get('hypotheses', [])]}
                   for name, field in model['fields'].items()}}
        for model in models]


def learn(protocol, output, cohort, arm, seed, stop_after=7):
    module, lab_type = load(protocol, arm, cohort)
    path, laboratories = run_path(output, cohort, arm, seed), []
    def factory(item):
        lab = lab_type(item)
        laboratories.append(lab)
        return lab
    if path.exists():
        session = module.UnifiedResearchSession(state=json.loads(path.read_text()), lab_factory=factory)
    else:
        config = module.ResearchConfig(**{**protocol['research_config'],
            **protocol['arms'][arm].get('research_config_overrides', {})}, seed=seed)
        tasks = task_sequence(protocol, seed)
        session = module.UnifiedResearchSession(config, tasks=tasks,
            measurement_budget=protocol['measurement_budget'], lab_factory=factory)
    while len(session.state['results']) < stop_after:
        incoming = json.loads(json.dumps(session.library.action_experience))
        started = time.perf_counter()
        state = session.run(max_tasks=1)
        record = state['results'][-1]
        record.update(learning_wall_seconds=time.perf_counter() - started,
                      action_experience_before=incoming,
                      action_experience_after=state['knowledge'].get('action_experience', {}),
                      actual_costs=laboratories[-1].costs, acquisitions=laboratories[-1].acquisitions,
                      instrument_models_after_task=instrument_snapshot(state['knowledge'].get('instrument_models', [])),
                      instrument_methods_after_task=[method for method in state['knowledge'].get('program_experiment_methods', [])
                                                     if method.get('solver_strategy') == 'instrument_sequence'])
        for key in ('observation_cost', 'measurement_cost'):
            if record['actual_costs'][key] != record['research']['costs'][key]:
                raise RuntimeError('reported costs differ from actual instrument purchases')
        save(path, state)
        session = module.UnifiedResearchSession(state=json.loads(path.read_text()), lab_factory=factory)
        report = record['research']
        print(json.dumps({'arm': arm, 'seed': seed, 'position': record['task_position'],
            'status': report['status'], 'selected_kind': report['selected_kind'],
            'selected_channels': report['selected_channels'], 'costs': report['costs']}), flush=True)


def ensure_learning(protocol, output, cohort):
    for arm in ARMS:
        for seed in protocol['cohorts'][cohort]:
            state = json.loads(run_path(output, cohort, arm, seed).read_text())
            if len(state['results']) != len(protocol['stages']):
                raise RuntimeError('finish all cohort learning before terminal scoring')


def evaluate(protocol, output, cohort, arm, seed):
    ensure_learning(protocol, output, cohort)
    module, lab_type = load(protocol, arm, cohort)
    from hegel_machine.cognition.models import Probe
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.evaluation import loss
    path = run_path(output, cohort, arm, seed)
    session = module.UnifiedResearchSession(state=json.loads(path.read_text()), lab_factory=lab_type)
    state = session.evaluate(**{key: protocol['terminal'][key] for key in ('length', 'trajectories', 'seed_offset')})
    tails = state.setdefault('long_tail_evaluation', [])
    for position in range(len(tails), len(protocol['stages'])):
        report = state['results'][position]['research']
        lab = lab_type(state['tasks'][position])
        rows = []
        for index, amplitude in enumerate(protocol['terminal']['impulse_amplitudes']):
            controls = ({'u': amplitude}, *({'u': 0.} for _ in range(min(protocol['terminal']['impulse_length'], lab.spec.max_probe_length) - 1)))
            rows.extend(lab.acquire(Probe(f'long_tail_{index}', controls), report['selected_channels']).observations)
        measured = loss(Program.from_dict(report['program']), rows, report['fallback'], .5)
        tails.append({'task_position': position, 'mse': measured,
                      'costs': {**lab.costs, 'compute_cost': len(rows)}})
    save(path, state)


def summarize(protocol, output, cohort):
    from collections import defaultdict
    result = {'scope': protocol['scope'], 'cohort': cohort, 'arms': {}}
    for arm in ARMS:
        rows, costs, terminal_costs, models, methods = [], Counter(), Counter(), [], []
        for seed in protocol['cohorts'][cohort]:
            state = json.loads(run_path(output, cohort, arm, seed).read_text())
            measured = {row['task_position']: row for row in state['evaluation']}
            tails = {row['task_position']: row for row in state['long_tail_evaluation']}
            models.extend({'apparatus_seed': seed, **model} for model in state['knowledge'].get('instrument_models', []))
            methods.extend({'apparatus_seed': seed, **method}
                           for method in state['knowledge'].get('program_experiment_methods', [])
                           if method.get('solver_strategy') == 'instrument_sequence')
            for record in state['results']:
                p, report = record['task_position'], record['research']
                events = report['actions']
                purchased = {item['probe_id']: item for item in record['acquisitions']}
                paid_plans = [{key: value for key, value in event.items()
                               if key in ('probe_id', 'method_id', 'parent_method_id', 'sequence_mode', 'reason')} | {
                    'observation_cost': purchased[event['probe_id']]['observation_cost'],
                    'measurement_cost': purchased[event['probe_id']]['measurement_cost']}
                    for event in events if event['action'] in ('plan_instrument_disagreement', 'plan_observation_research')
                    and event.get('probe_id') in purchased]
                row = {'apparatus_seed': seed, 'position': p, 'stage': protocol['stages'][p],
                    'initial_instrument_mode': protocol['instrument_modes'][str(seed)],
                    'environment': state['tasks'][p]['environment'],
                    'status': report['status'], 'kind': report['selected_kind'], 'channels': report['selected_channels'],
                    'costs': {**report['costs'], 'queries': record['actual_costs']['queries']},
                    'learning_wall_seconds': record['learning_wall_seconds'],
                    'random_mse': measured[p]['mse'], 'long_tail_mse': tails[p]['mse'],
                    'solved': max(measured[p]['mse'], tails[p]['mse']) <= protocol['terminal']['mse_threshold'],
                    'actions': dict(Counter(event['action'] for event in events)),
                    'instrument_events': [event for event in events if 'instrument' in event['action']],
                    'observation_action_events': [{key: value for key, value in event.items()
                                                  if key not in ('program', 'baseline_program')}
                        for event in events
                        if 'observation_research' in event['action']
                        or event.get('research_action') in ('revise_mechanism', 'remeasure_inputs', 'discriminate_inputs')],
                    'observation_action_choices': dict(Counter(event['selected_action'] for event in events
                        if event['action'] == 'choose_observation_research')),
                    'experienced_observation_choices': sum(event.get('trials', 0) > 0 for event in events
                        if event['action'] == 'choose_observation_research'),
                    'confirmed_direction_choices': sum(event.get('reason') == 'confirmed_direction_before_exploration'
                        for event in events if event['action'] == 'choose_observation_research'),
                    'action_experience_before': record.get('action_experience_before', {}),
                    'action_experience_after': record.get('action_experience_after', {}),
                    'instrument_models_after_task': record.get('instrument_models_after_task', []),
                    'instrument_methods_after_task': record.get('instrument_methods_after_task', []),
                    'method_events': dict(Counter(event['action'] for event in events
                                                 if event['action'] in METHOD_EVENTS)),
                    'paid_instrument_plans': paid_plans,
                    'used_sources': report['used_sources']}
                rows.append(row)
                costs.update(row['costs'])
                terminal_costs.update({key: measured[p][key] for key in COSTS if key in measured[p]})
                terminal_costs['queries'] += protocol['terminal']['trajectories']
                terminal_costs.update(tails[p]['costs'])
        sequences = defaultdict(list)
        for row in rows:
            sequences[row['apparatus_seed']].append(row)
        result['arms'][arm] = {'tasks': rows, 'instrument_models': models, 'instrument_experiment_methods': methods,
            'observation_action_choices': dict(sum((Counter(row['observation_action_choices'])
                                                   for row in rows), Counter())),
            'experienced_observation_choices': sum(row['experienced_observation_choices'] for row in rows),
            'confirmed_direction_choices': sum(row['confirmed_direction_choices'] for row in rows),
            'accepted_observation_actions': dict(Counter(event['research_action'] for row in rows
                for event in row['observation_action_events']
                if event['action'] == 'confirm_observation_research' and event['accepted'])),
            'method_events': dict(sum((Counter(row['method_events']) for row in rows), Counter())),
            'paid_stored_method_uses': sum(bool(plan.get('method_id')) for row in rows
                                          for plan in row['paid_instrument_plans']),
            'paid_parent_method_revisions': sum(bool(plan.get('parent_method_id')) for row in rows
                                               for plan in row['paid_instrument_plans']),
            'learning_status': dict(Counter(row['status'] for row in rows)),
            'solved': sum(row['solved'] for row in rows), 'tasks_count': len(rows),
            'instrument_free_solved_tasks': sum(row['solved'] and not row['channels'] for row in rows),
            'complete_sequences': sum(all(row['solved'] for row in sequence) for sequence in sequences.values()),
            'learning_costs': dict(costs), 'terminal_costs': dict(terminal_costs),
            'learning_wall_seconds': sum(row['learning_wall_seconds'] for row in rows)}
    save(output / cohort / 'summary.json', result)
    digest = {arm: {key: value for key, value in data.items()
                   if key not in ('tasks', 'instrument_models', 'instrument_experiment_methods')}
              for arm, data in result['arms'].items()}
    save(output / cohort / 'final_digest.json', digest)
    print(json.dumps(digest, indent=2), flush=True)


def queue(protocol, output, cohort):
    jobs = [('learn', arm, seed) for i, seed in enumerate(protocol['cohorts'][cohort])
            for arm in (('learned', 'policy_disabled', 'baseline_v092') if cohort == 'development'
                        else ARMS[i % len(ARMS):] + ARMS[:i % len(ARMS)])]
    jobs.extend(('evaluate', arm, seed) for seed in protocol['cohorts'][cohort] for arm in ARMS)
    execution = {'status': 'running', 'completed': []}
    for mode, arm, seed in jobs:
        log = output / cohort / 'logs' / f'{mode}_{arm}_{seed}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        save(output / cohort / 'execution.json', {**execution, 'active': [mode, arm, seed]})
        with log.open('a') as stream:
            code = subprocess.call([sys.executable, '-B', str(Path(__file__).resolve()), mode,
                '--output', str(output), '--cohort', cohort, '--arm', arm, '--seed', str(seed)],
                stdout=stream, stderr=subprocess.STDOUT)
        if code:
            save(output / cohort / 'execution.json', {**execution, 'status': 'failed', 'exit_code': code})
            raise SystemExit(code)
        execution['completed'].append([mode, arm, seed])
        print(json.dumps({'finished': [mode, arm, seed]}), flush=True)
    summarize(protocol, output, cohort)
    save(output / cohort / 'execution.json', {**execution, 'status': 'completed', 'exit_code': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'freeze', 'learn', 'evaluate', 'queue', 'summarize'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--cohort', choices=('development', 'main'), default='development')
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--stop-after', type=int, choices=(1, 2, 3, 4, 5, 6, 7), default=7)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.output)
        return
    protocol = json.loads((args.output / 'protocol.json').read_text())
    if args.mode == 'freeze':
        freeze(args.output)
    elif args.mode == 'queue':
        queue(protocol, args.output, args.cohort)
    elif args.mode == 'summarize':
        summarize(protocol, args.output, args.cohort)
    else:
        if args.mode == 'learn':
            learn(protocol, args.output, args.cohort, args.arm, args.seed, args.stop_after)
        else:
            evaluate(protocol, args.output, args.cohort, args.arm, args.seed)


if __name__ == '__main__':
    main()
