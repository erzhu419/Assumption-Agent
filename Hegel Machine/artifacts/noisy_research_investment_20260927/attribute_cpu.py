"""Attribute existing research-call CPU to task visits; no performance reruns."""
from collections import defaultdict
import gzip
import json
from pathlib import Path

base = Path(__file__).resolve().parent
output = base / 'fresh_replication'
if not (output / 'summary.json').exists():
    raise SystemExit('Run after the fixed study completes.')
protocol = json.loads((output / 'protocol.json').read_text())
summary = json.loads((output / 'summary.json').read_text())

def read(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)

result = {'scope': 'CPU measured around session.run; excludes checkpoint serialization and reload. New-task visits also include offering their new knowledge to prior tasks.', 'arms': {}}
for arm, settings in protocol['arms'].items():
    visits = []
    for root in protocol['roots']:
        directory = output / 'runs' / arm / root
        if settings['knowledge'] == 'independent':
            for position, path in enumerate(sorted(directory.glob('task*.json.gz'))):
                state = read(path)
                visits.append({'root': root, 'position': position, 'stage': protocol['stages'][position],
                    'visit': 'independent_task', 'cpu_seconds': state['cpu_seconds'],
                    'compute_cost': state['costs']['compute_cost'],
                    'concepts_after_visit': len(state['knowledge']['concepts'])})
        else:
            previous_cpu, previous_compute, previous_tasks = 0., 0, 0
            for path in sorted(directory.glob('visit*.json.gz')):
                state = read(path)
                new_task = len(state['results']) > previous_tasks
                position = len(state['results']) - 1 if new_task else state['revisits'][-1]['task_position']
                visits.append({'root': root, 'position': position, 'stage': protocol['stages'][position],
                    'visit': 'new_task_and_knowledge_offer' if new_task else 'revisit',
                    'cpu_seconds': state['study']['cpu_seconds'] - previous_cpu,
                    'compute_cost': state['costs']['compute_cost'] - previous_compute,
                    'concepts_after_visit': len(state['knowledge']['concepts'])})
                previous_cpu, previous_compute = state['study']['cpu_seconds'], state['costs']['compute_cost']
                previous_tasks = len(state['results'])
    stages = {}
    for stage in protocol['stages']:
        selected = [visit for visit in visits if visit['stage'] == stage]
        stages[stage] = {'visits': len(selected), 'cpu_seconds': sum(v['cpu_seconds'] for v in selected),
            'compute_cost': sum(v['compute_cost'] for v in selected)}
    measured = sum(visit['cpu_seconds'] for visit in visits)
    assert abs(measured - summary['totals'][arm]['learning_cpu_seconds']) < 1e-7
    result['arms'][arm] = {'by_stage': stages, 'total_research_cpu_seconds': measured,
        'slowest_visits': sorted(visits, key=lambda visit: visit['cpu_seconds'], reverse=True)[:10],
        'visits': visits}
(output / 'cpu_attribution.json').write_text(json.dumps(result, indent=2) + '\n')
print({arm: value['by_stage'] for arm, value in result['arms'].items()})
