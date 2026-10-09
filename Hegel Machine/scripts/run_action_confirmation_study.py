#!/usr/bin/env python3
"""Compare final research returns and exploration order, with failure/recovery tasks."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import time

import run_unified_action_policy_study as shared

ROOT = shared.ROOT
OUTPUT = ROOT / 'artifacts/action_confirmation_returns_20260919'
ARMS = ('baseline_v0871', 'learned', 'policy_disabled')


def prepare(output):
    if (output / 'protocol.json').exists():
        return
    previous = json.loads((ROOT / 'artifacts/unified_action_costs_v2_20260919/protocol.json').read_text())
    seeds = list(range(94601901, 94601907))
    protocol = {key: previous[key] for key in ('shared_laboratory_source', 'stages',
        'research_config', 'measurement_budget', 'terminal')}
    protocol.update(experiment='final_action_returns_and_exploration_order',
        fixed_at=datetime.now(timezone.utc).isoformat(), apparatus_seeds=seeds,
        arms={'baseline_v0871': {'runtime': 'artifacts/unified_action_costs_v2_20260919/runtime/src'},
            'learned': {'runtime': str((output / 'runtime/src').relative_to(ROOT))},
            'policy_disabled': {'runtime': str((output / 'runtime/src').relative_to(ROOT)),
                'research_config_overrides': {'learn_research_policy': False}}},
        learning_order=[{'seed': seed, 'arms': list(ARMS[index % 3:] + ARMS[:index % 3])}
                        for index, seed in enumerate(seeds)],
        maximum_concurrent_research_processes=1, formal_evidence=False, status='prepared',
        persistence='Complete KnowledgeLibrary JSON restored after every task, with all source costs included.',
        scope='Six new apparatus seeds, four complete empty-library Unified tasks per arm; '
              'same source, event, changed-input position and composed readout sequence as the previous study. '
              'No terminal result or private cause is supplied to research. '
              'Three separate controlled supplied-family recovery sequences are reported separately.',
        recovery={'source_seeds': [31, 41, 59], 'learner_seeds': {'31': 7, '41': 17, '59': 35},
            'source_radius': .6, 'source_rows': 64,
            'source_family': 'Explicit supplied nested relation/event/window expression, checked on paid source observations.',
            'source_cost_accounting': 'Source observations and scored prediction rows counted once in each arm; identical source JSON shared.',
            'shared_lab': str((output / 'shared_nested_lab.py').relative_to(ROOT)),
            'stages': [
                {'name': 'bounded_changed_relation_trial', 'radius': 1.2, 'observation_budget': 240,
                 'compute_budget': 240000, 'max_rounds': 4},
                {'name': 'funded_successor_recovery', 'radius': 1.2, 'observation_budget': 512,
                 'compute_budget': 480000, 'max_rounds': 6},
                {'name': 'original_relation_return', 'radius': .6, 'observation_budget': 240,
                 'compute_budget': 240000, 'max_rounds': 4}],
            'scope': 'Full run with a supplied paid source concept; this component test is not empty-library source discovery. '
                     'Larger-budget recovery is an experiment, not an assumed success. All failed or unresolved results remain.'})
    shared.save(output / 'protocol.json', protocol)
    shutil.copyfile(ROOT / 'scripts/action_confirmation_lab.py', output / 'shared_nested_lab.py')


def load(protocol, arm):
    module, source = shared.load(protocol, arm)
    path = ROOT / protocol['recovery']['shared_lab']
    spec = importlib.util.spec_from_file_location('_action_confirmation_lab', path)
    laboratory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(laboratory)
    return module, source, laboratory


def prepare_sources(protocol, output):
    module, _, apparatus = load(protocol, 'baseline_v0871')
    from hegel_machine.cognition.models import Candidate
    from hegel_machine.cognition.memory import KnowledgeLibrary
    for seed in protocol['recovery']['source_seeds']:
        path = output / 'recovery_sources' / f'{seed}.json'
        if path.exists():
            continue
        laboratory = apparatus.NestedResponseLab(protocol['recovery']['source_radius'])
        rows = laboratory.acquire(laboratory.random_probe(seed,
            length=protocol['recovery']['source_rows'], name='paid_source')).observations
        program = apparatus.supplied_source_family(protocol['recovery']['source_radius'])
        error = sum((program.predict(row) - row.target) ** 2 for row in rows) / len(rows)
        if error > 1e-12:
            raise ValueError('supplied source family does not explain the paid source evidence')
        library = KnowledgeLibrary()
        library.learn(Candidate('paid_source_response', program, 'observed_response'), laboratory.spec, rows)
        shared.save(path, {'knowledge': library.to_dict(), 'source_loss': error,
            'costs': {**laboratory.costs, 'compute_cost': len(rows)},
            'acquisitions': laboratory.acquisitions, 'source_program': program.to_dict()})


def recovery_path(output, arm, seed):
    return output / 'recovery' / arm / f'{seed}.json'


def learn_recovery(protocol, output, arm, seed):
    module, runtime, apparatus = load(protocol, arm)
    from hegel_machine.cognition.memory import KnowledgeLibrary
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent
    source = json.loads((output / 'recovery_sources' / f'{seed}.json').read_text())
    path = recovery_path(output, arm, seed)
    state = json.loads(path.read_text()) if path.exists() else {
        'source_seed': seed, 'source_costs': source['costs'], 'results': [], 'evaluation': [],
        'knowledge': source['knowledge'], 'runtime_source': str(runtime.relative_to(ROOT))}
    library = KnowledgeLibrary.from_dict(state['knowledge'])
    for position in range(len(state['results']), len(protocol['recovery']['stages'])):
        stage = protocol['recovery']['stages'][position]
        config = ResearchConfig(observation_budget=stage['observation_budget'],
            compute_budget=stage['compute_budget'], max_rounds=stage['max_rounds'],
            minimum_improvement=1e-7, active_probes=True, max_candidates=8,
            seed=protocol['recovery']['learner_seeds'][str(seed)] + 10007 * position,
            **protocol['arms'][arm].get('research_config_overrides', {}))
        laboratory = apparatus.NestedResponseLab(stage['radius'])
        incoming = json.loads(json.dumps(library.action_experience))
        started = time.perf_counter()
        report = UnifiedResearchAgent(config, library=library).run(laboratory)
        record = {'position': position, 'name': stage['name'], 'research': report,
            'learning_wall_seconds': time.perf_counter() - started,
            'action_experience_before': incoming, 'action_experience_after': library.action_experience,
            'actual_costs': laboratory.costs, 'acquisitions': laboratory.acquisitions}
        if report['costs']['observation_cost'] != laboratory.costs['observation_cost']:
            raise RuntimeError('recovery observation cost differs from actual acquisition')
        state['results'].append(record)
        state['knowledge'] = library.to_dict()
        state['status'] = 'completed' if position + 1 == len(protocol['recovery']['stages']) else 'paused'
        shared.save(path, state)
        state = json.loads(path.read_text())
        library = KnowledgeLibrary.from_dict(state['knowledge'])
        print(json.dumps({'arm': arm, 'source_seed': seed, 'stage': stage['name'],
            'status': report['status'], 'confirmation_loss': report['confirmation_loss'],
            'costs': report['costs']}), flush=True)


def ensure_all_learning(protocol, output):
    for arm in protocol['arms']:
        for seed in protocol['apparatus_seeds']:
            path = shared.run_path(output, arm, seed)
            state = json.loads(path.read_text()) if path.exists() else {}
            if len(state.get('results', [])) != len(protocol['stages']):
                raise ValueError(f'main learning incomplete: {arm}/{seed}')
        for seed in protocol['recovery']['source_seeds']:
            path = recovery_path(output, arm, seed)
            state = json.loads(path.read_text()) if path.exists() else {}
            if len(state.get('results', [])) != len(protocol['recovery']['stages']):
                raise ValueError(f'recovery learning incomplete: {arm}/{seed}')


def evaluate_recovery(protocol, output, arm, seed):
    _, _, apparatus = load(protocol, arm)
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.evaluation import loss
    path = recovery_path(output, arm, seed)
    state = json.loads(path.read_text())
    for position in range(len(state['evaluation']), len(state['results'])):
        stage = protocol['recovery']['stages'][position]
        laboratory = apparatus.NestedResponseLab(stage['radius'])
        rows = laboratory.evaluation_rows(seed=seed + 10007 * position + protocol['terminal']['seed_offset'],
            length=protocol['terminal']['length'], trajectories=protocol['terminal']['trajectories'])
        result = state['results'][position]['research']
        score = loss(Program.from_dict(result['program']), rows, result['fallback'], .5)
        state['evaluation'].append({'position': position, 'mse': score,
            'solved': score <= protocol['terminal']['mse_threshold'],
            'observation_cost': len(rows), 'measurement_cost': 0, 'compute_cost': len(rows)})
    shared.save(path, state)
    print(json.dumps({'arm': arm, 'source_seed': seed, 'evaluation': state['evaluation']}), flush=True)


def summarize(protocol, output):
    shared.summarize(protocol, output)
    summary = json.loads((output / 'summary.json').read_text())
    summary['scope'] = protocol['scope']
    summary['recovery'] = {'scope': protocol['recovery']['scope'], 'arms': {}}
    for arm in protocol['arms']:
        records, costs, terminal_costs, wall = [], Counter(), Counter(), 0.
        for seed in protocol['recovery']['source_seeds']:
            path = recovery_path(output, arm, seed)
            if not path.exists():
                continue
            state = json.loads(path.read_text())
            costs.update(state['source_costs'])
            evaluated = {row['position']: row for row in state['evaluation']}
            for record in state['results']:
                report = record['research']
                costs.update(report['costs'])
                score = evaluated.get(record['position'])
                if score:
                    terminal_costs.update({key: score[key] for key in shared.COSTS})
                wall += record['learning_wall_seconds']
                records.append({'source_seed': seed, 'position': record['position'], 'name': record['name'],
                    'status': report['status'], 'costs': report['costs'],
                    'confirmation_loss': report['confirmation_loss'], 'terminal': score,
                    'confirmation_attempts': [row for row in report['actions']
                        if row['action'] == 'validate_frozen_program'],
                    'action_outcomes': [{key: value for key, value in row.items()
                        if key not in ('program', 'baseline_program')} for row in report['actions']
                        if row['action'].startswith('research_action') or row['action'] == 'choose_research_action'],
                    'incoming_experience': record['action_experience_before'],
                    'outgoing_experience': record['action_experience_after']})
        summary['recovery']['arms'][arm] = {'tasks': records, 'learning_costs_including_source': dict(costs),
            'terminal_costs': dict(terminal_costs), 'learning_wall_seconds': wall,
            'completed_tasks': len(records), 'evaluated_tasks': sum(row['terminal'] is not None for row in records),
            'solved': sum(bool(row['terminal'] and row['terminal']['solved']) for row in records),
            'failed_learning_confirmations': sum(not attempt['accepted'] for row in records
                for attempt in row['confirmation_attempts'])}
    shared.save(output / 'summary.json', summary)
    print(json.dumps({'recovery': {arm: {key: value for key, value in record.items() if key != 'tasks'}
        for arm, record in summary['recovery']['arms'].items()}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'freeze', 'prepare_sources', 'learn', 'learn_recovery',
                                         'evaluate', 'evaluate_recovery', 'summarize'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.output)
        return
    protocol = json.loads((args.output / 'protocol.json').read_text())
    if args.mode == 'freeze':
        shared.freeze(args.output)
    elif args.mode == 'prepare_sources':
        prepare_sources(protocol, args.output)
    elif args.mode == 'summarize':
        summarize(protocol, args.output)
    else:
        recovery = args.mode.endswith('_recovery')
        seeds = protocol['recovery']['source_seeds'] if recovery else protocol['apparatus_seeds']
        if args.arm not in protocol['arms'] or args.seed not in seeds:
            parser.error('declared arm and seed required')
        if args.mode.startswith('evaluate'):
            ensure_all_learning(protocol, args.output)
        operation = {'learn': shared.learn, 'learn_recovery': learn_recovery,
                     'evaluate': shared.evaluate, 'evaluate_recovery': evaluate_recovery}[args.mode]
        operation(protocol, args.output, args.arm, args.seed)


if __name__ == '__main__':
    main()
