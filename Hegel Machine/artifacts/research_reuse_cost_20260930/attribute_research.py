"""Count already recorded research actions and cross-task knowledge offers."""
from collections import Counter
import gzip
import json
from pathlib import Path

base = Path(__file__).resolve().parent
wanted = ('execute_research_programs', 'test_response_noise_before_expansion',
          'defer_research_expansion_after_repeat')
for group in ('development', 'fresh_replication'):
    output = base / group
    if not (output / 'summary.json').exists():
        continue
    protocol = json.loads((output / 'protocol.json').read_text())
    arms = {}
    for arm in protocol['arms']:
        actions, events = Counter(), Counter()
        supported, screening, offers, using_sources, later_using_sources, tasks, later_tasks = 0, 0, 0, 0, 0, 0, 0
        for root in protocol['roots']:
            path = output / 'runs' / arm / root / 'checkpoint.json.gz'
            with gzip.open(path, 'rt') as stream:
                state = json.load(stream)
            screening += state.get('screening_compute_cost', 0)
            events.update(item['action'] for item in state.get('scheduler_events', []))
            offers += sum(item['compute_cost'] for item in state.get('scheduler_events', [])
                          if item['action'] == 'try_successor_knowledge')
            for position, row in enumerate(state['results']):
                report = row['research']
                tasks += 1
                using_sources += bool(report['used_sources'])
                later_tasks += position > 0
                later_using_sources += position > 0 and bool(report['used_sources'])
                actions.update(item['action'] for item in report['actions'])
                supported += sum(item.get('supported', False) for item in report['actions']
                                 if item['action'] == 'test_response_noise_before_expansion')
        arms[arm] = {'task_reports': tasks, 'reported_action_counts': {name: actions[name] for name in wanted},
                     'early_noise_supported': supported, 'try_successor_knowledge': events['try_successor_knowledge'],
                     'successor_offer_compute_cost': offers, 'screening_compute_cost': screening,
                     'tasks_using_sources': using_sources, 'successor_tasks_using_sources': later_using_sources,
                     'successor_tasks': later_tasks}
    result = {'scope': 'Research actions count final retained task reports. Scheduler offers and their compute charges cover the full cumulative chain, including discarded offers and revisits. These action counts do not replace the full study cost totals.', 'arms': arms}
    (output / 'research_attribution.json').write_text(json.dumps(result, indent=2) + '\n')
    print(group, arms)
