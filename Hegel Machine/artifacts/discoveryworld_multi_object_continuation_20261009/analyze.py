"""Compare both continuations against the same frozen initial explanations."""
import importlib.util
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
plan = json.loads((OUT / 'protocol.json').read_text())
spec = importlib.util.spec_from_file_location('frozen_response_recipe',
    OUT / 'host_sources/world_reference_recalibration.py')
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)
rows = []
key = lambda r: (r['crystal'], r['frequency'], r['activated'])
for job in plan['jobs']:
    case = next(c for c in plan['cases'] if c['id'] == job['case'])
    prefix = json.loads((ROOT / case['prefix'] / 'agent/shared/reference_recalibration.json').read_text())
    path = OUT / job['case'] / job['arm']
    result = json.loads((path / 'result.json').read_text())
    observations = json.loads((path / 'agent/laboratory_observations.json').read_text())
    current = json.loads((path / 'agent/current_observation.json').read_text())
    _, crystals, reactors, _, _ = recipe._scan(observations, current)
    responses = recipe._response_settings(observations, crystals, reactors)
    previous = {key(r) for r in prefix['response_settings']}
    new = [r for r in responses if key(r) not in previous]
    hypotheses = prefix.get('response_research', {}).get('hypotheses', [])
    refuted = {h['id']: [r for r in new if
        (h['lower'] <= r['frequency'] - h['predictions_by_crystal'][str(r['crystal'])] <= h['upper'])
        != r['activated']] for h in hypotheses}
    refuted = {identifier: evidence for identifier, evidence in refuted.items() if evidence}
    groups = {h['candidate_id']: {x['id'] for x in hypotheses
        if x['candidate_id'] == h['candidate_id']} for h in hypotheses}
    rows.append({**job, 'status': result['execution']['status'], 'score': result['score'],
        'new_costs': result['new_costs'], 'cumulative_new_costs': result['cumulative_new_costs'], 'initial_hypotheses': len(hypotheses),
        'initial_coefficient_candidates': len(groups),
        'refuted_initial_hypotheses': refuted,
        'exhausted_initial_coefficients': [cid for cid, ids in groups.items() if ids <= refuted.keys()],
        'new_public_responses': new})
output = {'scope': 'Diagnostic of discrimination of the same frozen finite explanations, using all '
    'new actual public responses including intermediate tuning. Exhausting an original coefficient '
    'candidate means all its original windows were refuted, not that the whole basis was disproved.',
    'runs': rows}
(OUT / 'fixed_explanation_outcomes.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
for row in rows:
    print(json.dumps({**{k:row[k] for k in ('case', 'arm', 'status', 'initial_hypotheses')},
        'continuation_actions': row['new_costs']['environment_steps'],
        'cumulative_new_actions': row['cumulative_new_costs']['environment_steps'],
        'refuted': len(row['refuted_initial_hypotheses']),
        'exhausted_coefficients': len(row['exhausted_initial_coefficients']),
        'success': row['score']['completedSuccessfully']}))
