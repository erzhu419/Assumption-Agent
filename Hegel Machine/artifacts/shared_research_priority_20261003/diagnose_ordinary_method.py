"""Execute the two queued development proposals once from identical paid state."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

from test_cognition_method_revisit_opportunity import (
    CONFIG, TASKS, MethodLab, UnifiedResearchSession, persisted, real_method_stream,
)
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.numeric_method_revisits import reprice_numeric_method_offer
from hegel_machine.cognition.research_resources import research_resource_share


stream = real_method_stream.__wrapped__()
purchases = []
factory = lambda task: MethodLab(purchases, **task)
tasks = deepcopy(TASKS)
paid_rows = sum(item['research']['costs']['observation_cost'] for item in stream['queued']['results'])
tasks[0]['research_limits'] = {'observation_budget': paid_rows + 52 - CONFIG.observation_budget}
tasks.append(dict(seed=4811, width=3, placement='inside_window', smooth=True))
session = UnifiedResearchSession(CONFIG, tasks=tasks, measurement_budget=320, lab_factory=factory)
old = deepcopy(stream['paused']['results'][0]['research'])
session.state['results'].append(dict(task_position=0, available_concepts=[], research=old))
deferred = persisted(session.run(max_tasks=1))
seen = []
class CaptureSession(UnifiedResearchSession):
    def _queue_knowledge_offer(self, position, source_position, offer, screening_cost, attempts, event):
        super()._queue_knowledge_offer(position, source_position, offer, screening_cost, attempts, event)
        if offer is not None and position == 0:
            seen.append(deepcopy(offer))
resumed = CaptureSession(state=deferred, lab_factory=factory)
queued = persisted(resumed.run(max_tasks=1))
ordinary = next(item for item in seen if item['candidate']['kind'] == 'successor_knowledge')
method = deepcopy(queued['pending_revisits'][0])
assert method['candidate']['kind'] == 'successor_partial_method'
pool = resumed._remaining()
out = {'scope': 'Fixed existing development regression scenario; two actual paid visits, no terminal rows.',
       'remaining_before': pool, 'initial_loss': old['confirmation_loss'], 'choices': {}}
for label, offer in [('ordinary', ordinary), ('method', method)]:
    start = UnifiedResearchSession(state=queued, lab_factory=factory)
    start.state['pending_revisits'] = []
    screening_before = start.state['screening_compute_cost']
    assert start._perform_revisit(deepcopy(offer))
    visit = start.state['revisits'][-1]
    report = start.state['results'][0]['research']
    events = report['actions'][len(old['actions']):]
    allocation = {key: visit['resource_limits'][key] - old['costs'][key] for key in pool}
    opportunity = offer['candidate'].get('method_opportunity', {})
    value = offer.get('expected_marginal_value', offer['relative_gain'])
    out['choices'][label] = {
        'paid_gain': offer['old_evidence_gain'], 'original_relative_gain': offer['relative_gain'],
        'expected_value': value, 'minimum_resources': offer['minimum_resources'],
        'recommended_resources': opportunity.get('recommended_resources'),
        'allocation': allocation, 'cap_share': research_resource_share(allocation, pool),
        'priority_using_executor_cap': value / research_resource_share(allocation, pool),
        'actual_visit_costs': report['visit_costs'],
        'extra_screening_after_visit': start.state['screening_compute_cost'] - screening_before,
        'remaining_after': start._remaining(), 'status': report['status'],
        'selection_loss': report.get('selection_loss'),
        'confirmation_loss': report.get('confirmation_loss'),
        'fresh_comparisons': report.get('continuation', {}).get('method_fresh_comparisons'),
        'comparison_events': [e for e in events if e['action'] in (
            'compare_continuation_candidates', 'validate_continued_program',
            'retain_method_confirmation_counterexample')],
        'original_paid_batches_preserved': report['batches'][:len(old['batches'])] == old['batches'],
    }
keys = {'action', 'probe_id', 'observation_rows', 'before_loss', 'after_loss',
        'attempted_after_loss', 'risk_quantile', 'loss', 'accepted',
        'current_fit_confirmed', 'attempt', 'selected_kind', 'status'}
for result in out['choices'].values():
    for field in ('fresh_comparisons', 'comparison_events'):
        if result[field] is not None:
            result[field] = [{k: v for k, v in item.items() if k in keys}
                            for item in result[field]]
path = Path('artifacts/shared_research_priority_20261003/ordinary_method_paid_diagnosis.json')
path.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps({key: value for key, value in out.items() if key != 'choices'}, indent=2), flush=True)
for label, result in out['choices'].items():
    print(label, json.dumps({key: value for key, value in result.items()
        if key not in ('comparison_events', 'fresh_comparisons')}, indent=2), flush=True)
