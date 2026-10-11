"""One paid development retry of an existing source after real method returns."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_observed_condition_decoding_study import laboratory
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.numeric_method_revisits import offer_numeric_method_experiment
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_session import UnifiedResearchSession

source = Path('artifacts/shared_research_priority_20261003/successor_regression/runs/new_cumulative/root11/checkpoint.json.gz')
state = json.load(gzip.open(source))
# Evaluation fields are neither inspected nor forwarded into the development session.
state['evaluation'] = []
session = UnifiedResearchSession(state=state, lab_factory=laboratory)
position = 9
previous = deepcopy(session.state['results'][position]['research'])
draft = next(d for d in session.library.research_drafts if d['draft_id'] == 'research_draft_0002')
remaining = session._remaining()
actions = []
with laboratory(session._laboratory_arguments(position)) as lab:
    length = min(24, lab.spec.max_probe_length)
    work = min(session.config.compute_budget // 8,
        max(0, remaining['compute_cost'] - 6 * previous['costs']['observation_cost'] - 32 * length))
    budget = SearchBudget(work)
    agent = UnifiedResearchAgent(session.config, library=session.library,
        measurement_budget=session._task_limits(position)['measurement_budget'])
    offer = offer_numeric_method_experiment(agent, lab, previous, {draft['draft_id']}, budget,
        actions=actions, remaining_resources=remaining)
session.state['screening_compute_cost'] += budget.spent
out = {'scope': 'One development screen and, if fundable, one fresh continuation from completed0.167 keep paid state; no terminal evaluation.',
    'source': str(source), 'target_position': position, 'source_draft': draft['draft_id'],
    'source_task_id': draft['source_task_id'], 'source_position': next(
        r['task_position'] for r in state['results'] if r['research']['task_id'] == draft['source_task_id']),
    'new_delete_arm_draft_was_added': False,
    'initial_confirmation_loss': previous['confirmation_loss'], 'remaining_before': remaining,
    'screening_compute_cost': budget.spent, 'screening_budget': work,
    'screening_actions': [{k:v for k,v in e.items() if k not in ('program','research_program')}
        for e in actions], 'offer': None}
if offer is not None:
    opportunity = offer['candidate']['method_opportunity']
    out['offer'] = {'old_evidence_gain': offer['old_evidence_gain'],
        'relative_gain': offer['relative_gain'], 'minimum_resources': offer['minimum_resources'],
        'expected_marginal_value': offer['expected_marginal_value'],
        'expected_resource_costs': offer['expected_resource_costs'],
        'resource_share': offer['resource_share'],
        'draft_ids': opportunity['draft_ids'], 'paid_counterexample_loss': opportunity['paid_counterexample_loss'],
        'candidate_paid_loss': opportunity['candidate_paid_loss'],
        'observed_visit_trials': opportunity['observed_visit_trials'],
        'observed_expected_gain': opportunity['observed_expected_gain'],
        'predicted_separation': opportunity['predicted_separation'],
        'confirmation_rows': opportunity['confirmation_rows'],
        'method_applicability': opportunity['method_applicability']}
    print(json.dumps({'screen': out['offer'], 'compute_cost': budget.spent}), flush=True)
    offer.update(task_position=position, source_task_position=out['source_position'],
        screening_compute_cost=budget.spent)
    before_screening = session.state['screening_compute_cost']
    executed = session._perform_revisit(offer)
    out['executed'] = executed
    if executed:
        current = session.state['results'][position]['research']
        visit = session.state['revisits'][-1]
        keep = {'action','probe_id','before_loss','after_loss','attempted_after_loss',
            'loss','accepted','current_fit_confirmed','observation_rows','attempt'}
        out.update(visit_costs=current['visit_costs'], resource_limits=visit['resource_limits'],
            whole_attributed_costs=visit['method_visit_outcome']['whole_visit_costs'],
            final_status=current['status'], final_confirmation_loss=current['confirmation_loss'],
            final_selection_loss=current['selection_loss'], final_draft_id=current.get('research_draft_id'),
            final_condition_status=(current.get('condition_uncertainty') or {}).get('status'),
            initial_paid_batches_preserved=current['batches'][:len(previous['batches'])] == previous['batches'],
            fresh_pairs=[{k:v for k,v in e.items() if k in keep}
                for e in current['continuation'].get('method_fresh_comparisons',[])],
            confirmations=[{k:v for k,v in e.items() if k in keep}
                for e in current['actions'][len(previous['actions']):]
                if e['action']=='validate_continued_program'],
            remaining_after=session._remaining(),
            additional_screening_after_visit=session.state['screening_compute_cost'] - before_screening)
path=Path('artifacts/shared_research_priority_20261003/reopen_existing_method_paid_diagnosis.json')
path.write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('screening_actions','offer')}, indent=2),flush=True)
