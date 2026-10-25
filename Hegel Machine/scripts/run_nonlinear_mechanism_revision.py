#!/usr/bin/env python3
"""Learn nested scalar responses, revise their inner mechanisms, and test restoration."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from random import Random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_knowledge_lineage_study import ROOT, COSTS, read, save, load_runtime, checkpoint_path


def laboratory(task):
    from hegel_machine.cognition.measurement_lab import MeasurementBatch
    from hegel_machine.cognition.models import Observation, Probe, TaskSpec

    class NestedScalarLab:
        channels = ()
        task_interface = """Investigate an unknown apparatus through its public controls.
query {actions:[{kind:'trajectory',sequence:[{control_name:number,...},...],channels:[]},...]}
buys one response per supplied control step. Each trajectory starts independently
at the apparatus reset state. Supply every public control within its bounds and
respect max_probe_length. There are no auxiliary measurement channels.
Write Python predict(row) using row['features'] and row['history']; history
contains only preceding controls since reset, never past response targets.
Fresh confirmation and terminal prediction use independent control trajectories
within the same public bounds and horizon. The apparatus is fixed per episode.
"""

        def __init__(self):
            rng = Random(task['apparatus_seed'])
            self._a, self._b = rng.uniform(.65, 1.35), rng.uniform(.65, 1.35)
            self._scale, self._offset = rng.uniform(.7, 1.4), rng.uniform(-.2, .2)
            self._noise = Random(task.get('noise_seed', task['seed']) + 49979687)
            self.spec = TaskSpec(f"response_{task['seed']}", 'Predict the measured response under controllable inputs',
                ('x', 'z'), (), (), feature_bounds={'x': (-1.5, 1.5), 'z': (-1.5, 1.5)}, max_probe_length=64)
            self.acquisitions = []
            self.observation_cost = 0

        def spec_for(self, channels=()):
            assert not channels
            return self.spec

        def public_spec(self):
            return {"controls": list(self.spec.feature_names),
                "feature_bounds": {name: list(bounds)
                    for name, bounds in self.spec.feature_bounds.items()},
                "action_kinds": ["trajectory"], "channels": [],
                "max_probe_length": self.spec.max_probe_length,
                "observation_cost_per_row": 1,
                "reset": "Every trajectory starts from the apparatus reset state.",
                "prediction_contract": "Current and preceding controls only; no past targets."}

        def action_cost(self, action):
            if action.get("kind") != "trajectory":
                raise ValueError("the available action kind is trajectory")
            sequence = action.get("sequence")
            if not isinstance(sequence, list) or not 1 <= len(sequence) <= self.spec.max_probe_length:
                raise ValueError("trajectory length must respect the public maximum probe length")
            if action.get("channels", []) != []:
                raise ValueError("the apparatus has no auxiliary channels")
            for point in sequence:
                if not isinstance(point, dict) or set(point) != set(self.spec.feature_names):
                    raise ValueError("each step must supply exactly the public controls")
                for name, (low, high) in self.spec.feature_bounds.items():
                    if not isinstance(point[name], (int, float)) or not low <= point[name] <= high:
                        raise ValueError(f"control {name} must be a number within its public bounds")
            return len(sequence)

        def prediction_rows(self, action):
            """Construct control-only forecasts without consulting the apparatus."""
            self.action_cost(action)
            history, rows = [], []
            for point in action["sequence"]:
                rows.append(Observation(dict(point), 0., tuple(deepcopy(history))))
                history.append(dict(point))
            return rows

        def execute(self, action):
            count = self.action_cost(action)
            first = self.observation_cost
            probe = Probe(f"inquiry_{len(self.acquisitions)}", tuple(deepcopy(action["sequence"])))
            batch = self.acquire(probe)
            rows = [Observation(dict(row.features), row.target,
                tuple({name: point[name] for name in self.spec.feature_names}
                      for point in row.history)) for row in batch.observations]
            return {"observation_index": first, "row_indices": list(range(first, first + count)),
                "action": deepcopy(action), "rows": [asdict(row) for row in rows],
                "measurements": [], "observation_cost": batch.observation_cost,
                "measurement_cost": batch.measurement_cost}

        def observation_rows(self, record):
            return [Observation(dict(row["features"]), row["target"], tuple(deepcopy(row["history"])))
                    for row in record["rows"]]

        @staticmethod
        def confirmation_precondition(score, *, mse_threshold):
            from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
            return AnonymousHistoryLab.confirmation_precondition(score, mse_threshold=mse_threshold)

        def confirmation_actions(self, n, rng):
            actions = []
            while n > 0:
                count = min(n, self.spec.max_probe_length)
                actions.append({"kind": "trajectory", "channels": [],
                    "sequence": [{name: rng.uniform(*bounds)
                        for name, bounds in self.spec.feature_bounds.items()} for _ in range(count)]})
                n -= count
            return actions

        def evaluation_rows(self, *, length=256, seed=1000003):
            separate = laboratory({**task, "noise_seed": seed})
            return [row for action in separate.confirmation_actions(length, Random(seed))
                    for row in separate.observation_rows(separate.execute(action))]

        def restore_paid_account(self, observations, costs):
            """Restore paid records and the noise stream, without repeating physics."""
            if self.observation_cost or self.acquisitions:
                raise ValueError("restore the paid account only into a fresh apparatus")
            self.observation_cost = costs["observation_cost"]
            self.acquisitions = [{"probe_id": f"inquiry_{index}",
                "sequence": deepcopy(record["action"]["sequence"]), "channels": []}
                for index, record in enumerate(observations)]
            if task.get("noise_sigma", 0.):
                for _ in range(self.observation_cost):
                    self._noise.gauss(0., task["noise_sigma"])

        @property
        def costs(self):
            return {'observation_cost': self.observation_cost, 'measurement_cost': 0,
                    'queries': len(self.acquisitions)}

        def acquire(self, probe, channels=()):
            assert not channels
            stage = task['response']
            weight = task.get('inner_weight', {'remove': 0., 'replace': .55}.get(stage, 1.))
            rows, history, responses = [], [], []
            for point in probe.sequence:
                inner = self._a * point['x'] ** 2 + weight * self._b * point['z'] ** 2
                target = inner if stage == 'base' else self._offset + self._scale * inner ** 2
                responses.append(target)
                target = sum(responses[-task.get('window_width', 1):])
                if task.get('noise_sigma', 0.):
                    target += self._noise.gauss(0., task['noise_sigma'])
                rows.append(Observation(dict(point), target, tuple(history)))
                history.append({**point, '__target__': target})
            self.observation_cost += len(rows)
            self.acquisitions.append({'probe_id': probe.probe_id, 'sequence': list(probe.sequence),
                                      'channels': []})
            return MeasurementBatch(rows, len(rows), 0, ())

        def random_probe(self, seed, *, length=24, name='sample'):
            rng = Random(seed)
            return Probe(name, tuple({key: rng.uniform(*bounds) for key, bounds in self.spec.feature_bounds.items()}
                                    for _ in range(length)))

        def restore(self, acquisitions):
            self.__init__()
            for item in acquisitions:
                self.acquire(Probe(item['probe_id'], tuple(item['sequence'])))

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    return NestedScalarLab()


def source(output, protocol, root):
    path = checkpoint_path(output / 'sources' / f'{root}.json')
    if path.exists():
        return
    module = load_runtime(protocol, 'baseline')
    tasks = protocol['roots'][root]['source_tasks']
    started = time.process_time()
    session = module.UnifiedResearchSession(module.ResearchConfig(**protocol['config']),
        tasks=tasks, measurement_budget=0, lab_factory=laboratory)
    state = session.run()
    state['cpu_seconds'] = time.process_time() - started
    save(path, state)
    print({'root': root, 'source_statuses': [row['research']['status'] for row in state['results']],
           'source_costs': state['costs'], 'source_cpu_seconds': state['cpu_seconds']}, flush=True)


def run_dir(output, arm, root):
    return output / 'runs' / arm / root


def learn_independent(output, protocol, arm, root):
    """Keep within-task learning, resetting the complete library between tasks."""
    module = load_runtime(protocol, arm)
    directory = run_dir(output, arm, root)
    path = checkpoint_path(directory / 'checkpoint.json')
    if path.exists() and read(path)['study']['learning_complete']:
        return
    tasks = protocol['roots'][root]['tasks']
    state = {'tasks': tasks, 'results': [], 'revisits': [], 'pending_revisits': [],
             'costs': dict.fromkeys(COSTS, 0), 'knowledge': {},
             'study': {'steps': 0, 'cpu_seconds': 0., 'learning_complete': False}}
    for position, task in enumerate(tasks):
        task_path = checkpoint_path(directory / f'task{position:02d}.json')
        if task_path.exists():
            learned = read(task_path)
        else:
            session = module.UnifiedResearchSession(module.ResearchConfig(**protocol['config']),
                tasks=[task], measurement_budget=0, lab_factory=laboratory)
            started = time.process_time()
            try:
                learned = session.run()
            except Exception as error:
                save(directory / 'failure.json', {'error': repr(error), 'task_position': position,
                     'interrupted_cpu_seconds': time.process_time() - started})
                raise
            learned['cpu_seconds'] = time.process_time() - started
            save(task_path, learned)
        state['results'].append({**learned['results'][0], 'task_position': position})
        for key in COSTS:
            state['costs'][key] += learned['costs'][key]
        state['study']['cpu_seconds'] += learned['cpu_seconds']
        state['study']['steps'] = position + 1
        state['study']['learning_complete'] = position + 1 == len(tasks)
        save(path, state)
        print({'arm': arm, 'root': root, 'steps': position + 1,
               'complete': state['study']['learning_complete'], 'costs': state['costs']}, flush=True)


def learn(output, protocol, arm, root):
    if protocol['arms'][arm]['knowledge'] == 'independent':
        return learn_independent(output, protocol, arm, root)
    module = load_runtime(protocol, arm)
    from hegel_machine.cognition.memory import KnowledgeLibrary
    path = checkpoint_path(run_dir(output, arm, root) / 'checkpoint.json')
    if path.exists():
        state = read(path)
        if state['study']['learning_complete']:
            return
        session = module.UnifiedResearchSession(state=state, lab_factory=laboratory)
    else:
        knowledge = (read(checkpoint_path(output / 'sources' / f'{root}.json'))['knowledge']
                     if protocol['arms'][arm]['knowledge'] == 'source' else {})
        session = module.UnifiedResearchSession(module.ResearchConfig(**protocol['config']),
            tasks=protocol['roots'][root]['tasks'], library=KnowledgeLibrary.from_dict(deepcopy(knowledge)),
            measurement_budget=0, lab_factory=laboratory)
        session.state['study'] = {'steps': 0, 'cpu_seconds': 0., 'learning_complete': False}
    while len(session.state['results']) < len(session.state['tasks']) or session.state['pending_revisits']:
        started = time.process_time()
        try:
            state = session.run(max_tasks=1)
        except Exception as error:
            save(path.parent / 'failure.json', {'error': repr(error), 'steps': session.state['study']['steps'],
                 'interrupted_cpu_seconds': time.process_time() - started})
            raise
        state['study']['steps'] += 1
        state['study']['cpu_seconds'] += time.process_time() - started
        state['study']['learning_complete'] = (len(state['results']) == len(state['tasks'])
                                               and not state['pending_revisits'])
        save(path, state)
        save(checkpoint_path(path.parent / f"visit{state['study']['steps']:02d}.json"), state, base=path)
        print({'arm': arm, 'root': root, 'steps': state['study']['steps'],
               'complete': state['study']['learning_complete'], 'costs': state['costs']}, flush=True)
        session = module.UnifiedResearchSession(state=state, lab_factory=laboratory)


def evaluate(output, protocol, arm, root):
    for name in protocol['arms']:
        for key in protocol['roots']:
            if not read(checkpoint_path(run_dir(output, name, key) / 'checkpoint.json'))['study']['learning_complete']:
                raise ValueError('all arms must finish learning before terminal scoring')
    load_runtime(protocol, arm)
    from hegel_machine.cognition.evaluation import loss
    from hegel_machine.cognition.language import Program
    path = run_dir(output, arm, root) / 'evaluation.json'
    if path.exists():
        return
    state = read(checkpoint_path(path.parent / 'checkpoint.json'))
    scored, settings = [], protocol['terminal']
    for position, task in enumerate(state['tasks']):
        report = state['results'][position]['research']
        program = Program.from_dict(report['program'])
        cpu = time.process_time()
        with laboratory({**task, 'noise_seed': task['seed'] + settings['seed_offset']}) as lab:
            errors, mean_errors = [], []
            for index in range(settings['trajectories']):
                probe = lab.random_probe(task['seed'] + settings['seed_offset'] + 10007 * index,
                                        length=settings['length'], name=f'terminal_{index}')
                observations = lab.acquire(probe).observations
                errors.append(loss(program, observations, report['fallback'], .5))
                if task.get('noise_sigma', 0.):
                    # The private mean is used only after all research ends.
                    # Keep noisy target history, including autoregressive inputs.
                    with laboratory({**task, 'noise_sigma': 0.}) as oracle:
                        expected = oracle.acquire(probe).observations
                    mean_rows = [replace(row, target=truth.target) for row, truth in zip(observations, expected)]
                    mean_errors.append(loss(program, mean_rows, report['fallback'], .5))
            scored.append({'position': position, 'stage': protocol['stages'][position],
                'mse': sum(errors) / len(errors), 'trajectories': errors, 'status': report['status'],
                'costs': {**lab.costs, 'compute_cost': settings['length'] * settings['trajectories'] * (2 if mean_errors else 1)},
                'mean_response_mse': sum(mean_errors) / len(mean_errors) if mean_errors else None,
                'mean_response_trajectories': mean_errors,
                'cpu_seconds': time.process_time() - cpu})
    save(path, scored)


def summarize(output, protocol):
    rows, totals = [], {}
    for arm, settings in protocol['arms'].items():
        total = {'tasks': 0, 'completed': 0, 'passed': 0, 'completed_and_passed': 0, 'false_completed': 0,
                 'source_costs': dict.fromkeys(COSTS, 0), 'learning_costs': dict.fromkeys(COSTS, 0),
                 'terminal_costs': dict.fromkeys(COSTS, 0), 'source_cpu_seconds': 0.,
                 'learning_cpu_seconds': 0., 'terminal_cpu_seconds': 0., 'revisits': 0}
        for root in protocol['roots']:
            directory = run_dir(output, arm, root)
            state = read(checkpoint_path(directory / 'checkpoint.json'))
            if settings['knowledge'] == 'source':
                original = read(checkpoint_path(output / 'sources' / f'{root}.json'))
                for key in COSTS:
                    total['source_costs'][key] += original['costs'][key]
                total['source_cpu_seconds'] += original['cpu_seconds']
            for key in COSTS:
                total['learning_costs'][key] += state['costs'][key]
            total['learning_cpu_seconds'] += state['study']['cpu_seconds']
            total['revisits'] += len(state['revisits'])
            for score in read(directory / 'evaluation.json'):
                report = state['results'][score['position']]['research']
                total['tasks'] += 1
                total['completed'] += score['status'] == 'completed'
                if protocol['pass_mse'] is not None:
                    passed = score['mse'] <= protocol['pass_mse']
                    total['passed'] += passed
                    total['completed_and_passed'] += passed and score['status'] == 'completed'
                    total['false_completed'] += not passed and score['status'] == 'completed'
                for key in COSTS:
                    total['terminal_costs'][key] += score['costs'][key]
                total['terminal_cpu_seconds'] += score['cpu_seconds']
                rows.append({'arm': arm, 'root': root, **score, 'kind': report['selected_kind'],
                    'learning_costs': report['costs'], 'used_sources': report['used_sources'],
                    'actions': dict(Counter(action['action'] for action in report['actions'])),
                    'research_operations': [step['operation'] for step in
                        (report.get('research_program') or {}).get('selection_steps', [])],
                    'reused_procedure': bool((report.get('research_program') or {}).get('reused_procedure')),
                    'mechanism_experiment_rows': sum(action['observation_cost'] for action in report['actions']
                        if action['action'] == 'acquire' and action.get('purpose') == 'distinguish_mechanisms')})
        total['all_costs'] = {key: sum(total[name][key] for name in
            ('source_costs', 'learning_costs', 'terminal_costs')) for key in COSTS}
        if protocol['pass_mse'] is None:
            for key in ('passed', 'completed_and_passed', 'false_completed'):
                total[key] = None
        selected = [row for row in rows if row['arm'] == arm]
        total['mean_mse'] = sum(row['mse'] for row in selected) / len(selected)
        mean_rows = [row for row in selected if row.get('mean_response_mse') is not None]
        total['mean_response_mse'] = (sum(row['mean_response_mse'] for row in mean_rows) / len(mean_rows)
                                      if mean_rows else None)
        totals[arm] = total
    by_stage = {}
    for arm in protocol['arms']:
        by_stage[arm] = {}
        for position, stage in enumerate(protocol['stages']):
            selected = [row for row in rows if row['arm'] == arm and row['position'] == position]
            by_stage[arm][stage] = {'tasks': len(selected),
                'passed': (sum(row['mse'] <= protocol['pass_mse'] for row in selected)
                           if protocol['pass_mse'] is not None else None),
                'completed': sum(row['status'] == 'completed' for row in selected),
                'selected_kinds': dict(Counter(row['kind'] for row in selected)),
                'reused_procedures': sum(row['reused_procedure'] for row in selected),
                'tasks_using_sources': sum(bool(row['used_sources']) for row in selected),
                'research_operations': dict(Counter(operation for row in selected
                    for operation in row['research_operations']))}
    comparisons = {reference: {key: totals['mechanism_revision']['all_costs'][key] /
        totals[reference]['all_costs'][key] - 1 for key in ('observation_cost', 'compute_cost')}
        for reference in ('baseline', 'empty_start')} if 'mechanism_revision' in totals else {}
    summary = {'protocol': str(output / 'protocol.json'), 'totals': totals, 'by_stage': by_stage,
               'new_seeded_cost_change_vs': comparisons, 'rows': rows}
    save(output / 'summary.json', summary)
    print(totals, flush=True)


def dispatch(output, protocol, phase, workers):
    jobs = [(root, None) for root in protocol['roots']] if phase == 'source' else [
        (root, arm) for arm in protocol['arms'] for root in protocol['roots']]
    def run(job):
        root, arm = job
        command = [sys.executable, str(Path(__file__).resolve()), '--output', str(output),
                   '--phase', phase, '--root', root]
        if arm:
            command.extend(('--arm', arm))
        subprocess.run(command, cwd=ROOT, check=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(run, jobs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('source', 'learn', 'evaluate', 'summarize'), required=True)
    parser.add_argument('--root')
    parser.add_argument('--arm')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    protocol = read(output / 'protocol.json')
    if args.phase == 'summarize':
        summarize(output, protocol)
    elif args.root:
        if args.phase == 'source':
            source(output, protocol, args.root)
        else:
            {'learn': learn, 'evaluate': evaluate}[args.phase](output, protocol, args.arm, args.root)
    else:
        dispatch(output, protocol, args.phase, args.workers)


if __name__ == '__main__':
    main()
