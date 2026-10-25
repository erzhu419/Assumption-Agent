#!/usr/bin/env python3
"""Full Unified sequences with persisted empirical research-action selection."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/hegel_machine/cognition"))
from checkpoint_storage import write_json_checkpoint
OUTPUT = ROOT / 'artifacts/unified_action_policy_20260919'
ARMS = ('fixed_v086', 'learned', 'policy_disabled')
COSTS = ('observation_cost', 'measurement_cost', 'compute_cost')


def save(path, payload):
    write_json_checkpoint(path, payload, indent=2, ensure_ascii=True, allow_nan=True)


def prepare(output):
    protocol = {
        'experiment': 'unified_program_structure_action_policy',
        'fixed_at': datetime.now(timezone.utc).isoformat(),
        'apparatus_seeds': [93701901, 93701902, 93701903],
        'arms': {
            'fixed_v086': {'runtime': 'artifacts/compact_relation_revision_20260919/runtime/src'},
            'learned': {'runtime': str((output / 'runtime/src').relative_to(ROOT))},
            'policy_disabled': {'runtime': str((output / 'runtime/src').relative_to(ROOT)),
                                'research_config_overrides': {'learn_research_policy': False}}},
        'shared_laboratory_source': 'artifacts/compact_relation_revision_20260919/runtime/src/hegel_machine/cognition',
        'stages': [
            {'readout': 'focus', 'actuator_curvature': .12, 'rename_prefix': ''},
            {'readout': 'event', 'actuator_curvature': .12, 'rename_prefix': 'event_'},
            {'readout': 'position_focus', 'actuator_curvature': .31, 'rename_prefix': 'position_'},
            {'readout': 'window_total_event', 'actuator_curvature': .31, 'rename_prefix': 'combined_'}],
        'research_config': {'observation_budget': 384, 'compute_budget': 360000,
            'minimum_improvement': 1e-7, 'max_history': 4, 'max_rounds': 6, 'active_probes': True},
        'measurement_budget': 4608,
        'terminal': {'length': 64, 'trajectories': 4, 'seed_offset': 1000003,
            'mse_threshold': 1e-6, 'after_all_declared_learning': True},
        'learning_order': [{'seed': seed, 'arms': list(ARMS if index % 2 == 0 else reversed(ARMS))}
            for index, seed in enumerate([93701901, 93701902, 93701903])],
        'maximum_concurrent_research_processes': 1,
        'scope': 'Three new apparatus seeds, four complete dependent Unified tasks per arm. '
            'No supplied source law, intervention, selected concept or private cause is passed to the learner. '
            'The policy change concerns executable program and structural research; this is not a claim '
            'that every measurement and joint-revision action has become policy-selected.',
        'persistence': 'Every task boundary saves and reconstructs the complete Unified session from JSON.',
        'cost_accounting': 'Source is the first real Unified task and counted once per arm; '
            'all later tasks share its paid learned knowledge. Evaluation costs are separate.',
        'formal_evidence': False}
    path = output / 'protocol.json'
    if path.exists():
        return json.loads(path.read_text())
    save(path, protocol)
    return protocol


def freeze(output):
    destination = output / 'runtime/src/hegel_machine'
    if destination.exists():
        raise ValueError('runtime already frozen; use the preserved release')
    shutil.copytree(ROOT / 'src/hegel_machine', destination,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))


def load(protocol, arm):
    source = (ROOT / protocol['arms'][arm]['runtime']).resolve()
    sys.path.insert(0, str(source))
    session = importlib.import_module('hegel_machine.cognition.unified_session')
    if not Path(session.__file__).resolve().is_relative_to(source):
        raise RuntimeError('mixed runtimes: execute each arm in a separate process')
    laboratory_source = ROOT / protocol['shared_laboratory_source']
    for name in ('structure_lab', 'unified_object_lab'):
        canonical = 'hegel_machine.cognition.' + name
        spec = importlib.util.spec_from_file_location(canonical, laboratory_source / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[canonical] = module
        spec.loader.exec_module(module)
    return session, source


def tasks(protocol, seed):
    return [{'environment': 'objects', 'objects': 2, 'apparatus_seed': seed,
        'seed': seed + 10007 * index, **stage} for index, stage in enumerate(protocol['stages'])]


def run_path(output, arm, seed):
    return output / 'runs' / arm / str(seed) / 'checkpoint.json'


def learn(protocol, output, arm, seed):
    module, source = load(protocol, arm)
    path = run_path(output, arm, seed)
    previous = json.loads(path.read_text()) if path.exists() else None
    laboratories = []

    def factory(item):
        lab = module.laboratory_for(item)
        laboratories.append(lab)
        return lab

    session = module.UnifiedResearchSession(state=previous, lab_factory=factory) if previous else \
        module.UnifiedResearchSession(module.ResearchConfig(**{**protocol['research_config'],
            **protocol['arms'][arm].get('research_config_overrides', {})}, seed=seed),
            tasks=tasks(protocol, seed), measurement_budget=protocol['measurement_budget'], lab_factory=factory)
    while len(session.state['results']) < len(session.state['tasks']):
        incoming = json.loads(json.dumps(session.library.action_experience))
        started = time.perf_counter()
        report = session.run(max_tasks=1)
        record = report['results'][-1]
        record['learning_wall_seconds'] = time.perf_counter() - started
        record['action_experience_before'] = incoming
        record['action_experience_after'] = report['knowledge'].get('action_experience', {})
        record['actual_costs'] = laboratories[-1].costs
        record['acquisitions'] = laboratories[-1].acquisitions
        record['runtime_source'] = str(source.relative_to(ROOT))
        for key in ('observation_cost', 'measurement_cost'):
            if record['actual_costs'][key] != record['research']['costs'][key]:
                raise RuntimeError('learning cost disagrees with actual laboratory acquisitions')
        save(path, report)
        session = module.UnifiedResearchSession(state=json.loads(path.read_text()), lab_factory=factory)
        print(json.dumps({'arm': arm, 'seed': seed, 'position': record['task_position'],
            'status': record['research']['status'], 'costs': record['research']['costs'],
            'wall_seconds': record['learning_wall_seconds'],
            'action_choices': [row for row in record['research']['actions']
                if row['action'] == 'choose_research_action']}), flush=True)


def evaluate(protocol, output, arm, seed):
    for named in protocol['arms']:
        for apparatus in protocol['apparatus_seeds']:
            path = run_path(output, named, apparatus)
            state = json.loads(path.read_text()) if path.exists() else {}
            if len(state.get('results', [])) != len(protocol['stages']):
                raise RuntimeError('complete every declared learning task before terminal evaluation')
    module, _ = load(protocol, arm)
    path = run_path(output, arm, seed)
    state = json.loads(path.read_text())
    if len(state.get('evaluation', [])) == len(state['tasks']):
        return
    session = module.UnifiedResearchSession(state=state)
    report = session.evaluate(**{key: protocol['terminal'][key]
        for key in ('length', 'trajectories', 'seed_offset')})
    save(path, report)
    print(json.dumps({'arm': arm, 'seed': seed, 'evaluation': report['evaluation']}), flush=True)


def summarize(protocol, output):
    result = {'experiment': protocol['experiment'], 'formal_evidence': False,
              'expected_tasks_per_arm': len(protocol['apparatus_seeds']) * len(protocol['stages']), 'arms': {}}
    for arm in protocol['arms']:
        rows, final_states = [], []
        for seed in protocol['apparatus_seeds']:
            path = run_path(output, arm, seed)
            if not path.exists():
                continue
            state = json.loads(path.read_text())
            final_states.append(state)
            evaluated = {row['task_position']: row for row in state['evaluation']}
            for record in state['results']:
                research = record['research']
                position = record['task_position']
                choices = [row for row in research['actions'] if row['action'] == 'choose_research_action']
                feedback = [row for row in research['actions'] if row['action'] == 'research_action_feedback']
                terminal = evaluated.get(position)
                rows.append({'seed': seed, 'position': position, 'stage': protocol['stages'][position],
                    'status': research['status'], 'costs': research['costs'],
                    'queries': record['actual_costs']['queries'],
                    'learning_wall_seconds': record['learning_wall_seconds'],
                    'selected_kind': research['selected_kind'], 'selected_channels': research['selected_channels'],
                    'action_choices': choices, 'action_feedback': feedback,
                    'action_outcomes': [{key: value for key, value in event.items()
                        if key not in ('program', 'baseline_program')}
                        for event in research['actions'] if event['action'] == 'research_action_confirmation'],
                    'action_counts': dict(Counter(row['action'] for row in research['actions'])),
                    'incoming_experience': record['action_experience_before'],
                    'outgoing_experience': record['action_experience_after'],
                    'terminal': terminal,
                    'solved': None if terminal is None else terminal['loss'] <= protocol['terminal']['mse_threshold']})
        result['arms'][arm] = {'completed_tasks': len(rows),
            'evaluated_tasks': sum(row['terminal'] is not None for row in rows),
            'solved': sum(row['solved'] is True for row in rows),
            'learning_costs': {key: sum(row['costs'][key] for row in rows) for key in COSTS},
            'queries': sum(row['queries'] for row in rows),
            'learning_wall_seconds': sum(row['learning_wall_seconds'] for row in rows),
            'terminal_costs': {key: sum(row['terminal'][key] for row in rows if row['terminal']) for key in COSTS},
            'tasks_with_policy_choices': sum(bool(row['action_choices']) for row in rows), 'tasks': rows}
    result['comparisons'] = {}
    revised = result['arms']['learned']
    for reference in (name for name in protocol['arms'] if name != 'learned'):
        base = result['arms'][reference]
        result['comparisons']['learned_vs_' + reference] = {
            'completed_tasks': [base['completed_tasks'], revised['completed_tasks']],
            'solved': [base['solved'], revised['solved']],
            'learning_costs': {key: [base['learning_costs'][key], revised['learning_costs'][key]]
                               for key in COSTS},
            'learning_wall_seconds': [base['learning_wall_seconds'], revised['learning_wall_seconds']]}
    if protocol.get('previous_learned_summary'):
        previous = json.loads((ROOT / protocol['previous_learned_summary']).read_text())['arms']['learned']
        result['comparisons']['learned_vs_first_policy'] = {
            'solved': [previous['solved'], revised['solved']],
            'learning_costs': {key: [previous['learning_costs'][key], revised['learning_costs'][key]]
                               for key in COSTS},
            'queries': [previous['queries'], revised['queries']],
            'learning_wall_seconds': [previous['learning_wall_seconds'], revised['learning_wall_seconds']]}
    result['policy_evidence'] = {
        'choices_with_prior_trials': sum(choice.get('trials', 0) > 0 for row in revised['tasks']
                                        for choice in row['action_choices']),
        'successor_tasks_using_prior_trials': sum(row['position'] > 0 and any(
            choice.get('trials', 0) > 0 for choice in row['action_choices']) for row in revised['tasks']),
        'choices_by_reason': dict(Counter(choice.get('reason') for row in revised['tasks']
                                         for choice in row['action_choices']))}
    save(output / 'summary.json', result)
    print(json.dumps({arm: {key: value for key, value in data.items() if key != 'tasks'}
                      for arm, data in result['arms'].items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'freeze', 'learn', 'evaluate', 'summarize'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--arm')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.output)
        return
    protocol = json.loads((args.output / 'protocol.json').read_text())
    if args.mode == 'freeze':
        freeze(args.output)
    elif args.mode == 'summarize':
        summarize(protocol, args.output)
    else:
        if args.arm not in protocol['arms'] or args.seed not in protocol['apparatus_seeds']:
            parser.error('a declared arm and seed are required')
        (learn if args.mode == 'learn' else evaluate)(protocol, args.output, args.arm, args.seed)


if __name__ == '__main__':
    main()
