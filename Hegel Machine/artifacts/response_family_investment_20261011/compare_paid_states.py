"""Compare the staged scope repair on two saved public response histories."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hegel_machine.cognition import world_response_hypotheses as hypotheses
from hegel_machine.cognition.world_response_constraints import query_response_family


def module(variant):
    name = 'hegel_machine.cognition.world_response_consequences'
    spec = importlib.util.spec_from_file_location(name, OUT / ('runtime_' + variant) / 'world_response_consequences.py')
    selected = importlib.util.module_from_spec(spec)
    sys.modules[name] = selected
    spec.loader.exec_module(selected)
    return selected


CASES = [
    ('excluded_cross', 'response_assignment_resume_20261011', 3, 17, 18),
    ('retained_cubic', 'response_research_reopen_20261010', 3, 22, 23),
]


def compare(case):
    name, cohort, repeat, old_number, current_number = case
    source = ROOT / 'artifacts' / cohort / 'live' / f'repeat_{repeat:02d}' / 'after/agent/shared/reference_recalibration.json'
    saved = json.loads(source.read_text())
    history = {row['question_id']: row for row in saved['response_research_history']}
    previous = history[f'reference_response_5_{old_number}']
    current = history[f'reference_response_5_{current_number}']
    count = current['family_research']['response_count']
    cutoff = max(current['evidence'])
    responses = deepcopy(saved['response_settings'][:count])
    for row in responses:
        row['evidence'] = [index for index in row['evidence'] if index <= cutoff]
    # Only public measurements, control grids and response records available at
    # this saved comparison enter the reconstructed input. No final candidates,
    # future exclusion proofs or successful target labels are restored.
    state = {key: deepcopy(saved[key]) for key in (
        'research_table', 'control_grids', 'control_grid', 'target_numbers', 'reference_numbers')}
    state.update(response_settings=responses,
        completed=deepcopy(previous['family_research']['completed']),
        response_research=deepcopy(previous), family_history_checks={})
    for row in state['research_table']['references'] + state['research_table']['targets']:
        expected = current['family_research']['features'][str(row['crystal'])]
        assert {k: v['value'] for k, v in row['measurements'].items()} == expected
        for reading in list(row['measurements'].values()) + ([row['frequency']] if 'frequency' in row else []):
            evidence = reading['evidence']
            assert all(index <= cutoff for index in (evidence if isinstance(evidence, list) else [evidence]))
    for family in previous['family_research']['families']:
        if family['status'] != 'UNSAT_VERIFIED':
            continue
        state['family_history_checks'][family['id']] = {
            **deepcopy(family), 'queries': [],
            'features': deepcopy(previous['family_research']['features']),
            'responses': [[r['crystal'], r['frequency'], r['activated']]
                          for r in responses[:previous['family_research']['response_count']]],
            'evidence': deepcopy(previous['evidence']),
        }
    estimators = [{key: deepcopy(family[key]) for key in ('id', 'basis')}
                  for family in current['family_research']['families']]
    features = {int(k): v for k, v in current['family_research']['features'].items()}
    old_bases = previous['family_research']['bases']
    paid = {family['id']: query_response_family(features, responses, family,
            branch_limit=None, compute_limit=50000)
            for family in estimators if family['basis'] not in old_bases}
    results = {}
    for variant in ('before', 'candidate'):
        module(variant)
        result = hypotheses.propose_response_research(deepcopy(state), estimators,
            'shared/consolidated_method.py', paid_checks=deepcopy(paid))
        research = result['family_research']
        selected = (result.get('selection') or {}).get('selected_experiment')
        results[variant] = {
            'status': result['status'], 'no_progress_cycles': research['no_progress_cycles'],
            'nonlocal_progress': research['nonlocal_progress'], 'selected_role': research['selected_role'],
            'selected': {key: selected.get(key) for key in ('number', 'prediction', 'family')} if selected else None,
            'operational_fallback': (result.get('operational_fallback') or {}).get('selected_experiment', {}).get('prediction'),
            'panel': research['panel'], 'compute_cost': result['compute_cost'],
            'families': {family['id']: family['status'] for family in research['families']},
        }
    output = {'case': name, 'source': str(source.relative_to(ROOT)),
        'response_count': count, 'maximum_evidence_index': cutoff,
        'previous_stagnation': previous['family_research']['no_progress_cycles'],
        'new_family_checks': {key: value['status'] for key, value in paid.items()},
        'results': results, 'new_model_calls': 0,
        'scope': 'Reconstructed saved public inputs with newly recomputed full-family checks; same-input selection comparison, not a full physical replay or fresh-model performance result.'}
    (OUT / f'{name}_comparison.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output), flush=True)
    return output


if __name__ == '__main__':
    for case in CASES:
        compare(case)
