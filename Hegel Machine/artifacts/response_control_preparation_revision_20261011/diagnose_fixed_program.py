"""Recompute the saved fixed-program check from public paid artifacts only."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.world_reference_recalibration import _control_grids
from hegel_machine.cognition.world_response_hypotheses import _nested_supports, _propose_fitted_responses

for case in ('cubic_original_1', 'cubic_shifted_1'):
    folder = ROOT / 'artifacts/response_composition_export_20261011' / case / 'auto_composition'
    paid = json.loads((folder / 'research_trace.json').read_text())
    observations = json.loads((folder / 'evidence_observations.json').read_text())
    declarations = list(next(ast.literal_eval(node.value)
        for node in ast.parse((folder / 'final_controller.py').read_text()).body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ESTIMATORS'
                                               for t in node.targets)))
    for mode in ('current', 'earliest_paid_reference') if case.endswith('shifted_1') else ('current',):
        table = deepcopy(paid['research_table'])
        if mode == 'earliest_paid_reference':
            for row in table['references']:
                first = min((r for r in paid['response_settings']
                    if r['crystal'] == row['crystal'] and r['activated']), key=lambda r: min(r['evidence']))
                row['frequency'] = {'value': first['frequency'], 'evidence': min(first['evidence'])}
        state = {'research_table': table, 'response_settings': paid['response_settings'],
            'target_numbers': [4, 5], 'completed': [4],
            'control_grids': _control_grids(observations, paid['response_settings'], {1, 2, 3})}
        result = _propose_fitted_responses(state, declarations + _nested_supports(declarations),
                                           'shared/consolidated_method.py')
        candidate = next(c for c in result['candidates'] if any(
            s['origin'] == 'initial_references' and s['family'] == 'grammar_signal_3__support_1'
            for s in c.get('fit_sources', [])))
        program = Program.from_dict(candidate['program'])
        features = {r['crystal']: {k: v['value'] for k, v in r['measurements'].items()}
                    for r in table['references'] + table['targets']}
        residuals = [(r, r['frequency'] - program.predict(Observation(features[r['crystal']], 0.)))
                     for r in paid['response_settings']]
        setting = round(candidate['target_prediction'])
        residual = setting - candidate['target_prediction']
        positive = [value for row, value in residuals if row['activated']]
        low, high = min(*positive, residual), max(*positive, residual)
        bands = [[h['lower'], h['upper']] for h in result['hypotheses'] if h['candidate_id'] == candidate['id']]
        print(json.dumps({'case': case, 'mode': mode, 'candidate': candidate,
            'setting': setting, 'finite_bands': bands,
            'finite_predictions': [a <= residual <= b for a, b in bands],
            'minimal_band': [low, high], 'checked_response_count': len(residuals),
            'negative_conflicts': [row for row, value in residuals if not row['activated'] and low <= value <= high]}))
