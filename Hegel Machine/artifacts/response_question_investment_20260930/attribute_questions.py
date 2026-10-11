"""After learning ends, score the actual retained programs before and after investments."""
import argparse
from collections import Counter
from dataclasses import replace
import gzip
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from run_nonlinear_mechanism_revision import laboratory
from run_knowledge_lineage_study import load_runtime

parser = argparse.ArgumentParser()
parser.add_argument('--group', required=True)
args = parser.parse_args()
output = BASE / args.group
protocol = json.loads((output / 'protocol.json').read_text())
summary = json.loads((output / 'summary.json').read_text())
load_runtime(protocol, 'new_cumulative')
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program

def read(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)

def score(task, before, after):
    """Use common held-out noisy history and private mean targets only for scoring."""
    settings = protocol['terminal']
    a, b = Program.from_dict(before['program']), Program.from_dict(after['program'])
    errors = [[], []]
    started = time.process_time()
    with laboratory({**task, 'noise_seed': task['seed'] + settings['seed_offset']}) as lab:
        with laboratory({**task, 'noise_sigma': 0.}) as oracle:
            for index in range(settings['trajectories']):
                probe = lab.random_probe(task['seed'] + settings['seed_offset'] + 10007 * index,
                    length=settings['length'], name=f'terminal_{index}')
                rows = lab.acquire(probe).observations
                expected = oracle.acquire(probe).observations
                means = [replace(row, target=truth.target) for row, truth in zip(rows, expected)]
                errors[0].append(loss(a, means, before['fallback'], .5))
                errors[1].append(loss(b, means, after['fallback'], .5))
    return {'before_mean_response_mse': sum(errors[0]) / len(errors[0]),
            'after_mean_response_mse': sum(errors[1]) / len(errors[1]),
            'diagnostic_cpu_seconds': time.process_time() - started,
            'diagnostic_observation_cost': settings['length'] * settings['trajectories'],
            'diagnostic_compute_cost': 2 * settings['length'] * settings['trajectories']}

arms = {}
for arm, settings in protocol['arms'].items():
    questions, investments, statuses = [], [], Counter()
    remaining = dict.fromkeys(('observation_cost', 'measurement_cost', 'compute_cost'), 0)
    remaining_by_root = {}
    initial_questions = 0
    for root in protocol['roots']:
        directory = output / 'runs' / arm / root
        state = read(directory / 'checkpoint.json.gz')
        if not state['study']['learning_complete'] or state['pending_revisits']:
            raise ValueError('all investments must finish before question attribution')
        limits = {'observation_cost': protocol['limits']['chain_observation'],
                  'measurement_cost': 0, 'compute_cost': protocol['limits']['chain_compute']}
        for key, value in limits.items():
            remaining[key] += value - state['costs'][key]
        remaining_by_root[root] = {key: value - state['costs'][key] for key, value in limits.items()}
        for position, record in enumerate(state['results']):
            for question in record['research'].get('response_questions', []):
                statuses[question['status']] += 1
                questions.append({'root': root, 'position': position,
                    **{key: value for key, value in question.items() if key not in ('parent', 'candidate')},
                    'parent_kind': question['parent'].get('kind'),
                    'candidate_kind': question['candidate'].get('kind')})
        if settings['knowledge'] == 'independent':
            immediate = []
            for position, task_file in enumerate(sorted(directory.glob('task*.json.gz'))):
                learned = read(task_file)
                revisits = learned['revisits']
                for index, visit in enumerate(revisits):
                    later = next((other['previous'] for other in revisits[index + 1:]
                                  if other['task_position'] == visit['task_position']), None)
                    after = later if later is not None else learned['results'][0]['research']
                    immediate.append((position, visit, after, None))
                first = revisits[0]['previous'] if revisits else learned['results'][0]['research']
                initial_questions += len(first.get('response_questions', []))
        else:
            immediate = []
            previous_revisits, previous_cpu = 0, 0.
            previous_tasks = 0
            for visit_file in sorted(directory.glob('visit*.json.gz')):
                learned = read(visit_file)
                if len(learned['results']) > previous_tasks:
                    initial_questions += len(learned['results'][-1]['research'].get('response_questions', []))
                if len(learned['revisits']) > previous_revisits:
                    visit = learned['revisits'][-1]
                    position = visit['task_position']
                    immediate.append((position, visit, learned['results'][position]['research'],
                                      learned['study']['cpu_seconds'] - previous_cpu))
                previous_revisits = len(learned['revisits'])
                previous_tasks = len(learned['results'])
                previous_cpu = learned['study']['cpu_seconds']
        for position, visit, after, cpu in immediate:
            question = visit['candidate'].get('response_question')
            if question is None:
                continue
            before = visit['previous']
            evidence = next((q for q in after.get('response_questions', [])
                             if q['question_id'] == question['question_id']), None)
            outcome = score(state['tasks'][position], before, after)
            a, b = outcome['before_mean_response_mse'], outcome['after_mean_response_mse']
            investments.append({'root': root, 'position': position,
                'stage': protocol['stages'][position], 'question_id': question['question_id'],
                'question_status': evidence['status'] if evidence else None,
                'resolution_events': [item for item in after['actions'][len(before['actions']):]
                                      if item['action'] == 'resolve_response_question'],
                'program_changed': before['program'] != after['program'],
                'before_status': before['status'], 'after_status': after['status'],
                'research_costs': visit['visit_costs'], 'research_cpu_seconds': cpu,
                'result': 'win' if b < a else 'loss' if b > a else 'tie', **outcome})
    resolutions = [event for row in investments for event in row['resolution_events']]
    arms[arm] = {'initial_question_count': initial_questions, 'final_question_statuses': dict(statuses),
        'investments': len(investments), 'investment_outcomes': dict(Counter(row['result'] for row in investments)),
        'fresh_selection_counts': dict(Counter(event['outcome'] for event in resolutions)),
        'third_incumbent_comparisons': sum(event.get('incumbent_loss') is not None for event in resolutions),
        'third_incumbent_retained': sum(event.get('incumbent_loss') is not None and
                                        event['outcome'] == 'incumbent' for event in resolutions),
        'program_changes': sum(row['program_changed'] for row in investments),
        'investment_research_costs': {key: sum(row['research_costs'][key] for row in investments)
            for key in ('observation_cost', 'measurement_cost', 'compute_cost')},
        'investment_cpu_seconds': (None if settings['knowledge'] == 'independent' else
                                   sum(row['research_cpu_seconds'] or 0. for row in investments)),
        'unused_chain_resources': remaining,
        'unused_resources_by_root': remaining_by_root,
        'diagnostic_costs': {key: sum(row[key] for row in investments)
            for key in ('diagnostic_cpu_seconds', 'diagnostic_observation_cost', 'diagnostic_compute_cost')},
        'rows': investments, 'questions': questions}
report = {'scope': 'Only after all frozen terminal scoring. Investment outcomes compare programs immediately before/after each real revisit on common terminal trajectories, retaining noisy history. Diagnostics are retrospective, excluded from research and primary terminal cost, and explicitly charged here. Independent sessions do not have per-visit CPU timers; their full research CPU remains in summary.json.',
          'arms': arms}
(output / 'question_attribution.json').write_text(json.dumps(report, indent=2) + '\n')
print({arm: {key: value[key] for key in ('initial_question_count', 'final_question_statuses',
    'investments', 'investment_outcomes', 'program_changes', 'investment_research_costs',
    'unused_chain_resources')} for arm, value in arms.items()})
