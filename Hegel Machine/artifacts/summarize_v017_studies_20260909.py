"""Compare fixed cognitive versions and the producer-feedback intervention."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('previous_summary', ROOT / 'summarize_v016_studies_20260909.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def mechanisms(session):
    methods = [record for episode in session['episodes'] for action in episode['actions']
               if action['action'] == 'knowledge_return' for record in action['details'].get('method_returns', [])]
    quoted = [{'task_index': episode['task_index'], 'task_id': episode['task_id'], 'visit': episode['visit'],
               'round': action['round'], 'choice': action['details']['learned_method']}
              for episode in session['episodes'] for action in episode['actions']
              if action['details'].get('learned_method', {}).get('return_estimate', {}).get('knowledge_return', {}).get('trials')]
    return {'linked_concepts': len({action['details']['concept_id'] for episode in session['episodes']
                                   for action in episode['actions'] if action['action'] == 'link_knowledge_method'}),
            'method_return_records': len(methods),
            'positive_exclusive_returns': sum(record['retained_gain'] is not None and record['retained_gain'] > 0 for record in methods),
            'negative_exclusive_returns': sum(record['retained_gain'] is not None and record['retained_gain'] < 0 for record in methods),
            'method_choices_with_measured_output_returns': quoted,
            'method_opportunities_with_output_returns': [episode['scheduler'] for episode in session['episodes']
                if episode['scheduler'].get('method_opportunity', {}).get('return_estimate', {}).get('knowledge_return', {}).get('trials')]}


def comparison(before, after):
    result = previous.compare(before, after)
    result['mechanisms_after'] = mechanisms(after[0])
    return result


if __name__ == '__main__':
    result = {'experiment': 'v017_fixed_run_comparisons', 'seeds': {}}
    for seed, baseline in [(1237, 'v016_mainline'), (2027, 'v017_baseline')]:
        result['seeds'][str(seed)] = {arm: comparison(
            previous.run(seed, baseline, '20260909', arm), previous.run(seed, 'v017_mainline', '20260909', arm))
            for arm in ('methods', 'ordinary')}
    result['producer_feedback_ablation_seed1237'] = comparison(
        previous.run(1237, 'v017_without_method_knowledge_return', '20260909', 'methods'),
        previous.run(1237, 'v017_mainline', '20260909', 'methods'))
    result['previous_positive_seed1613_methods'] = comparison(
        previous.run(1613, 'v016_mainline', '20260909', 'methods'),
        previous.run(1613, 'v017_mainline', '20260909', 'methods'))
    path = ROOT / 'hegel_machine_v017_knowledge_production_summary_20260909.json'
    with path.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(path)
