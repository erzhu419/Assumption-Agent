#!/usr/bin/env python3
"""Full-session object input/relation research with paid actions and JSON transfer."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib
import importlib.util
import json
from pathlib import Path
from random import Random
import subprocess
import sys

import run_observation_action_study as common

ROOT = common.ROOT
OUTPUT = ROOT / 'artifacts/object_action_study_20260919'
ARMS = ('baseline_v094', 'learned', 'policy_disabled')
STAGES = ('source', 'input_only', 'relation_only', 'joint_change', 'return_source')
common.ARMS = ARMS


def prepare(output):
    if (output / 'protocol.json').exists():
        return
    baseline = ROOT / 'artifacts/observation_applicability_study_20260919/runtime/src'
    if not (baseline / 'hegel_machine/cognition/unified_session.py').exists():
        raise FileNotFoundError(baseline)
    designs = {95301900: (3, 'event'), 95301901: (3, 'focus'),
               95301902: (3, 'event'), 95301903: (4, 'focus'), 95301904: (4, 'event')}
    tasks, parameters = {}, {}
    for seed, (objects, readout) in designs.items():
        rng = Random(seed + 17011)
        curves = [rng.uniform(*bounds) for bounds in ((.08, .16), (.28, .36), (.44, .52))]
        radii = [rng.uniform(*bounds) for bounds in ((.70, .85), (1.05, 1.20), (.45, .60))]
        parameters[str(seed)] = {'objects': objects, 'readout': readout,
                                 'actuator_curvatures': curves, 'relation_radii': radii}
        tasks[str(seed)] = [{'environment': 'objects', 'objects': objects,
            'apparatus_seed': seed, 'seed': seed + 10007 * position,
            'readout': readout, 'variant': 'exchange', 'profile': 'confirmation',
            'relation_direction': 'symmetric', 'relation_condition': 'always',
            'actuator_curvature': curves[c], 'relation_radius': radii[r]}
            for position, (c, r) in enumerate(((0, 0), (1, 0), (1, 1), (2, 2), (0, 0)))]
    common.save(output / 'protocol.json', {
        'experiment': 'common_object_input_and_relation_research',
        'fixed_at': datetime.now(timezone.utc).isoformat(),
        'cohorts': {'development': [95301900], 'main': [95301901, 95301902, 95301903, 95301904]},
        'stages': list(STAGES), 'apparatus_parameters': parameters, 'tasks_by_seed': tasks,
        'development_runtime': 'src',
        'arms': {'baseline_v094': {'runtime': str(baseline.relative_to(ROOT))},
                 'learned': {'runtime': str((output / 'runtime/src').relative_to(ROOT))},
                 'policy_disabled': {'runtime': str((output / 'runtime/src').relative_to(ROOT)),
                                     'research_config_overrides': {'learn_research_policy': False}}},
        'shared_laboratory_source': str((baseline / 'hegel_machine/cognition').relative_to(ROOT)),
        'research_config': {'observation_budget': 512, 'compute_budget': 480000,
            'minimum_improvement': 1e-7, 'max_rounds': 6, 'max_history': 4, 'active_probes': True},
        'measurement_budget': 6144, 'maximum_concurrent_research_processes': 1,
        'terminal': {'length': 64, 'trajectories': 4, 'seed_offset': 1000003,
            'mse_threshold': 1e-6, 'focus_offsets': [-1e-6, 0., 1e-6],
            'public_pair_distances': [.35, .65, .95, 1.25],
            'phases': ['off', 'on', 'on', 'on', 'on', 'off', 'off', 'off', 'off']},
        'scope': 'One empty-library full session per independently parameterized apparatus. '
            'The five-stage stream changes actuator mapping, then relation radius, then both, '
            'then returns. Main crosses 3/4 objects with focus/event readouts. Ground-truth '
            'settings are laboratory construction only; the learner receives the existing '
            'public command interface, purchasable views, object roles and actual observations. '
            'No supplied source law, diagnostic action or action score.',
        'seed_generation': 'Fixed roots 95301900..04; Random(root+17011) samples three disjoint '
            'curvature and radius ranges before any outcome; coupling comes from the existing confirmation apparatus.',
        'persistence': 'Restore complete UnifiedResearchSession JSON between every stage.',
        'evaluation_contract': 'All cohort learning finishes before any terminal scoring. Random trajectories '
            'and public focus-cut/pair-distance probes never update knowledge or action policy. '
            'The stress probes do not use true relation radii.',
        'cost_accounting': 'All source learning, unsuccessful research and actual purchases are counted; '
            'terminal costs separate. Runtime arms use the same search accounting and finite budgets.',
        'formal_evidence': False})


def load(protocol, arm, cohort):
    source = ROOT / (protocol['development_runtime'] if cohort == 'development' and arm != 'baseline_v094'
                     else protocol['arms'][arm]['runtime'])
    sys.path.insert(0, str(source))
    # Use exactly the same existing physical apparatus in every arm.
    for name in ('structure_lab', 'unified_object_lab'):
        canonical = 'hegel_machine.cognition.' + name
        spec = importlib.util.spec_from_file_location(canonical,
            ROOT / protocol['shared_laboratory_source'] / (name + '.py'))
        laboratory = importlib.util.module_from_spec(spec)
        sys.modules[canonical] = laboratory
        spec.loader.exec_module(laboratory)
    session = importlib.import_module('hegel_machine.cognition.unified_session')
    if not Path(session.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError('execute each runtime in a separate process')
    lab_type = sys.modules['hegel_machine.cognition.unified_object_lab'].UnifiedObjectLab
    def factory(item):
        arguments = dict(item)
        arguments.pop('environment')
        return lab_type(**arguments)
    return session, factory


common.load = load
common.task_sequence = lambda protocol, seed: protocol['tasks_by_seed'][str(seed)]


def evaluate(protocol, output, cohort, arm, seed):
    common.ensure_learning(protocol, output, cohort)
    module, factory = load(protocol, arm, cohort)
    from hegel_machine.cognition.models import Probe
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.evaluation import loss
    path = common.run_path(output, cohort, arm, seed)
    session = module.UnifiedResearchSession(state=json.loads(path.read_text()), lab_factory=factory)
    state = session.evaluate(**{key: protocol['terminal'][key] for key in ('length', 'trajectories', 'seed_offset')})
    tests = state.setdefault('object_boundary_evaluation', [])
    for position in range(len(tests), len(STAGES)):
        report, task = state['results'][position]['research'], state['tasks'][position]
        lab, rows = factory(task), []
        count = task['objects']
        for cut in range(count - 1):
            for offset in protocol['terminal']['focus_offsets']:
                for distance in protocol['terminal']['public_pair_distances']:
                    points = []
                    for phase in protocol['terminal']['phases']:
                        positions = [-1.95 + 3.9 * i / (count - 1) for i in range(count)]
                        if phase == 'on':
                            positions[1] = positions[0] + distance
                        point = {f'{role}_o{i}': value for role, values in (
                            ('q', [.65, -.35, .45, -.55][:count]),
                            ('x', positions), ('u', [.02, -.03, .04, -.01][:count]))
                            for i, value in enumerate(values)}
                        point['focus'] = cut + .5 + offset
                        points.append(point)
                    probe = Probe(f'public_object_boundary:{cut}:{offset}:{distance}', tuple(points))
                    rows.extend(lab.acquire(probe, report['selected_channels']).observations)
        tests.append({'task_position': position,
            'mse': loss(Program.from_dict(report['program']), rows, report['fallback'], .5),
            'costs': {**lab.costs, 'compute_cost': len(rows)}})
    common.save(path, state)


def summarize(protocol, output, cohort):
    result = {'scope': protocol['scope'], 'cohort': cohort, 'arms': {}}
    for arm in ARMS:
        rows, learning, terminal = [], Counter(), Counter()
        for seed in protocol['cohorts'][cohort]:
            state = json.loads(common.run_path(output, cohort, arm, seed).read_text())
            random = {row['task_position']: row for row in state['evaluation']}
            boundary = {row['task_position']: row for row in state['object_boundary_evaluation']}
            for record in state['results']:
                p, report = record['task_position'], record['research']
                events = report['actions']
                no_candidate, choice = [], {}
                for event in events:
                    if event['action'] == 'choose_object_research':
                        choice = event
                    elif event['action'] == 'propose_object_research' and event['status'] == 'no_candidate':
                        context, direction = choice.get('context'), event['direction']
                        before = record['action_experience_before'].get(context, {}).get(direction)
                        after = record['action_experience_after'].get(context, {}).get(direction)
                        no_candidate.append({'direction': direction, 'context': context,
                            'field_status': event.get('field_status'),
                            'costs': {key: event[key] for key in common.COSTS if key in event},
                            'existing_experience': before is not None,
                            'experience_unchanged': before == after})
                row = {'apparatus_seed': seed, 'position': p, 'stage': STAGES[p],
                    'objects': state['tasks'][p]['objects'], 'readout': state['tasks'][p]['readout'],
                    'status': report['status'], 'kind': report['selected_kind'],
                    'channels': report['selected_channels'],
                    'costs': report['costs'] | {'queries': record['actual_costs']['queries']},
                    'learning_wall_seconds': record['learning_wall_seconds'],
                    'random_mse': random[p]['mse'], 'boundary_mse': boundary[p]['mse'],
                    'solved': max(random[p]['mse'], boundary[p]['mse']) <= protocol['terminal']['mse_threshold'],
                    'actions': dict(Counter(e['action'] for e in events)),
                    'object_action_choices': dict(Counter(e['selected_action'] for e in events
                        if e['action'] == 'choose_object_research' and e.get('selected_action'))),
                    'experienced_object_choices': sum(e.get('trials', 0) > 0 for e in events
                        if e['action'] == 'choose_object_research'),
                    'supported_object_actions': dict(Counter(e['direction'] for e in events
                        if e['action'] == 'object_research_feedback' and e['status'] == 'supported')),
                    'no_candidate_visits': no_candidate,
                    'research_events': [{k: v for k, v in e.items() if k not in ('program', 'baseline_program')}
                        for e in events if any(word in e['action'] for word in ('object', 'relation', 'instrument', 'observation_research'))],
                    'action_experience_before': record['action_experience_before'],
                    'action_experience_after': record['action_experience_after'],
                    'acquisitions': record['acquisitions'], 'used_sources': report['used_sources']}
                rows.append(row)
                learning.update(row['costs'])
                terminal.update({key: random[p][key] for key in common.COSTS if key in random[p]})
                terminal['queries'] += protocol['terminal']['trajectories']
                terminal.update(boundary[p]['costs'])
        result['arms'][arm] = {'tasks': rows, 'solved': sum(r['solved'] for r in rows),
            'tasks_count': len(rows), 'learning_status': dict(Counter(r['status'] for r in rows)),
            'complete_sequences': sum(all(r['solved'] for r in rows if r['apparatus_seed'] == seed)
                for seed in protocol['cohorts'][cohort]),
            'instrument_free_solved_tasks': sum(r['solved'] and not r['channels'] for r in rows),
            'object_action_choices': dict(sum((Counter(r['object_action_choices']) for r in rows), Counter())),
            'experienced_object_choices': sum(r['experienced_object_choices'] for r in rows),
            'supported_object_actions': dict(sum((Counter(r['supported_object_actions']) for r in rows), Counter())),
            'no_candidate_visit_counts': dict(Counter(v['direction'] for r in rows for v in r['no_candidate_visits'])),
            'no_candidate_visit_costs': dict(sum((Counter(v['costs']) for r in rows
                for v in r['no_candidate_visits']), Counter())),
            'no_candidate_existing_experience_unchanged': sum(v['existing_experience'] and v['experience_unchanged']
                for r in rows for v in r['no_candidate_visits']),
            'learning_costs': dict(learning), 'terminal_costs': dict(terminal),
            'learning_wall_seconds': sum(r['learning_wall_seconds'] for r in rows)}
    common.save(output / cohort / 'summary.json', result)
    digest = {arm: {key: value for key, value in data.items() if key != 'tasks'}
              for arm, data in result['arms'].items()}
    common.save(output / cohort / 'final_digest.json', digest)
    print(json.dumps(digest, indent=2), flush=True)


def queue(protocol, output, cohort):
    jobs = [('learn', arm, seed) for i, seed in enumerate(protocol['cohorts'][cohort])
            for arm in (('learned', 'policy_disabled', 'baseline_v094') if cohort == 'development'
                        else ARMS[i % len(ARMS):] + ARMS[:i % len(ARMS)])]
    jobs.extend(('evaluate', arm, seed) for seed in protocol['cohorts'][cohort] for arm in ARMS)
    execution = {'status': 'running', 'completed': []}
    for mode, arm, seed in jobs:
        log = output / cohort / 'logs' / f'{mode}_{arm}_{seed}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        common.save(output / cohort / 'execution.json', {**execution, 'active': [mode, arm, seed]})
        with log.open('a') as stream:
            code = subprocess.call([sys.executable, '-B', str(Path(__file__).resolve()), mode,
                '--output', str(output), '--cohort', cohort, '--arm', arm, '--seed', str(seed)],
                stdout=stream, stderr=subprocess.STDOUT)
        if code:
            common.save(output / cohort / 'execution.json', {**execution, 'status': 'failed', 'exit_code': code})
            raise SystemExit(code)
        execution['completed'].append([mode, arm, seed])
        print(json.dumps({'finished': [mode, arm, seed]}), flush=True)
    summarize(protocol, output, cohort)
    common.save(output / cohort / 'execution.json', {**execution, 'status': 'completed', 'exit_code': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'freeze', 'learn', 'evaluate', 'queue', 'summarize'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--cohort', choices=('development', 'main'), default='development')
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.output)
        return
    protocol = json.loads((args.output / 'protocol.json').read_text())
    if args.mode == 'freeze':
        common.freeze(args.output)
    elif args.mode == 'queue':
        queue(protocol, args.output, args.cohort)
    elif args.mode == 'summarize':
        summarize(protocol, args.output, args.cohort)
    elif args.mode == 'learn':
        common.learn(protocol, args.output, args.cohort, args.arm, args.seed, stop_after=len(STAGES))
    else:
        evaluate(protocol, args.output, args.cohort, args.arm, args.seed)


if __name__ == '__main__':
    main()
