"""Re-screen frozen conditions on paid live evidence; no world or model calls."""
from collections import Counter
import json
from pathlib import Path
import sys
from time import monotonic
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from hegel_machine.cognition.world_reference_response import response_premise_proposal_inputs
from hegel_machine.cognition.world_response_conditions import synthesize_response_conditions

BASE = Path(__file__).parent
branch = sys.argv[1] if len(sys.argv) > 1 else 'retry'
workspace = BASE / branch / 'agent'
prefix = '' if branch == 'retry' else branch + '_'
start_observation = json.loads((BASE / branch / 'result.json').read_text())['costs_before']['observation_cost']
author = 'shared/response_representation_proposal_premise.py'
inquiries = json.loads((workspace / 'llm_inquiries.json').read_text())
observations = json.loads((workspace / 'laboratory_observations.json').read_text())
agent = SimpleNamespace(workspace=workspace, llm_inquiries=inquiries,
    observations=observations,
    lab=SimpleNamespace(public_observation=lambda: observations[-1]['observation']))
previous = inquiries['proposal:' + author]['condition_research']
started = monotonic()
packet = response_premise_proposal_inputs(agent, author)
updated = synthesize_response_conditions(packet, previous=previous)
elapsed = monotonic() - started
before = {row['id']: row for row in previous['candidates']}
changed = [{'id': row['id'], 'before': before[row['id']]['status'], 'after': row['status']}
    for row in updated['candidates'] if row['id'] in before
    and row['status'] != before[row['id']]['status']]
report = {'source': branch + '/agent', 'scope': 'post_run_paid_evidence_reanalysis',
    'not_seen_by_live_model': True, 'model_calls': 0, 'new_observations': 0,
    'available_observations': len(observations),
    'response_count_before': len(previous['inputs']['responses']),
    'response_count_after': len(updated['inputs']['responses']),
    'candidate_status_before': dict(Counter(row['status'] for row in previous['candidates'])),
    'candidate_status_after': dict(Counter(row['status'] for row in updated['candidates'])),
    'changed_candidates': changed, 'status': updated['status'],
    'regenerated': updated['regenerated'], 'compute_cost': updated['compute_cost'],
    'elapsed_seconds': elapsed, 'experiment_action': updated['experiment_action'],
    'target_consequences': updated.get('target_consequences'),
    'new_paid_responses': [row for row in packet['responses']
        if any(index >= start_observation for index in row['evidence'])]}
(BASE / (prefix + 'conditions_after.json')).write_text(json.dumps(updated, indent=2) + '\n')
(BASE / (prefix + 'paid_reanalysis.json')).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
