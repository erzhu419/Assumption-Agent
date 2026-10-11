"""Count already recorded research actions and cross-task knowledge offers."""
from collections import Counter
import gzip
import json
from pathlib import Path

base = Path(__file__).resolve().parent
wanted = ('execute_research_programs', 'test_response_noise_before_expansion',
          'defer_research_expansion_after_repeat', 'plan_noisy_response_experiment',
          'test_noisy_response_experiment', 'select_noisy_response_refit',
          'confirm_noisy_response_refit')
for group in ('development', 'fresh_replication'):
    output = base / group
    if not (output / 'summary.json').exists():
        continue
    protocol = json.loads((output / 'protocol.json').read_text())
    arms = {}
    for arm in protocol['arms']:
        actions, events = Counter(), Counter()
        supported, screening, offers, using_sources, later_using_sources, tasks, later_tasks = 0, 0, 0, 0, 0, 0, 0
        distinguishing_rows, distinguishing_supported, distinguishing_refuted = 0, 0, 0
        response_refits_confirmed, directed_refits_confirmed = 0, 0
        ordinary_pairs, parent_retained = Counter(), 0
        retained_after_directed_support, accepted_after_directed_uncertainty = 0, 0
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
                distinguishing_rows += sum(item['observation_cost'] for item in report['actions']
                    if item['action'] == 'acquire' and item.get('purpose') == 'distinguish_noisy_response_refits')
                distinguishing_supported += sum(item.get('supported', False) for item in report['actions']
                    if item['action'] == 'test_noisy_response_experiment')
                distinguishing_refuted += sum(item.get('refuted', False) for item in report['actions']
                    if item['action'] == 'test_noisy_response_experiment')
                all_confirmations = [item for item in report['actions']
                    if item['action'] == 'confirm_noisy_response_refit']
                confirmations = [item for item in all_confirmations if item['accepted']]
                response_refits_confirmed += len(confirmations)
                directed_refits_confirmed += sum(bool(item.get('distinguishing_evidence')) for item in confirmations)
                parent_retained += sum(not item['accepted'] for item in all_confirmations)
                for item in all_confirmations:
                    comparison = item.get('comparison')
                    if comparison:
                        ordinary_pairs['supported' if comparison['supported'] else
                            'refuted' if comparison['refuted'] else 'unresolved'] += 1
                    directed = item.get('distinguishing_evidence')
                    if directed:
                        retained_after_directed_support += directed['supported'] and not item['accepted']
                        accepted_after_directed_uncertainty += (not directed['supported'] and
                            not directed['refuted'] and item['accepted'])
        arms[arm] = {'task_reports': tasks, 'reported_action_counts': {name: actions[name] for name in wanted},
                     'early_noise_supported': supported, 'try_successor_knowledge': events['try_successor_knowledge'],
                     'successor_offer_compute_cost': offers, 'screening_compute_cost': screening,
                     'tasks_using_sources': using_sources, 'successor_tasks_using_sources': later_using_sources,
                     'successor_tasks': later_tasks,
                     'directed_discrimination_rows': distinguishing_rows,
                     'directed_discrimination_supported': distinguishing_supported,
                     'directed_discrimination_refuted': distinguishing_refuted,
                     'response_refits_confirmed': response_refits_confirmed,
                     'directed_refits_confirmed': directed_refits_confirmed,
                     'ordinary_paired_confirmation': dict(ordinary_pairs),
                     'parent_response_retained': parent_retained,
                     'parent_retained_after_directed_support': retained_after_directed_support,
                     'response_accepted_after_directed_uncertainty': accepted_after_directed_uncertainty}
    result = {'scope': 'Research actions count final retained task reports. Scheduler offers and their compute charges cover the full cumulative chain, including discarded offers and revisits. These action counts do not replace the full study cost totals.', 'arms': arms}
    (output / 'research_attribution.json').write_text(json.dumps(result, indent=2) + '\n')
    print(group, arms)
