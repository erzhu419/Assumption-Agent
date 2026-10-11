"""Reconstruct paid-menu tune segments before comparing them with later records."""
from copy import deepcopy
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from hegel_machine.cognition.world_reference_recalibration import _dialog, _tune
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation

SOURCE = ROOT / 'artifacts/response_forecast_classes_20261011'
CASES = ('cubic_shifted_2__gain_higher__adapted_law',
         'cubic_original_3__gain_lower__adapted_law')


def planned_segment(record, target):
    # Only this paid menu and the requested target enter the prediction.
    view = deepcopy(record['observation'])
    bounds = re.search(r'allowable range is ([\d.]+) to ([\d,]+)',
                       view['dialog_box']['dialogIn'], re.I)
    low, high = float(bounds[1]), float(bounds[2].replace(',', ''))
    rows = []
    while command := _tune(view, target):
        action = command['actions'][0]
        label = view['dialog_box']['dialogOptions'][str(action['chosen_dialog_option_int'])]
        match = re.search(r'(increase|decrease) frequency by (\d+) hz', label, re.I)
        frequency = _dialog(view)[1] + int(match[2]) * (1 if match[1].lower() == 'increase' else -1)
        if not low <= frequency <= high:
            return {'status': 'unmodeled_boundary', 'rows': rows}
        rows.append({'action': action, 'frequency': frequency})
        view['dialog_box']['dialogIn'] = re.sub(r'(frequence is:\s*)[\d.]+',
            lambda m: m[1] + str(frequency), view['dialog_box']['dialogIn'])
    return {'status': 'reconstructed_interior', 'rows': rows,
            'source_observation': record['index'], 'start_frequency': _dialog(record['observation'])[1],
            'requested_frequency': target, 'step_count': len(rows)}


rows = []
for phase in ('before', 'after'):
    for case in CASES:
        directory = SOURCE / phase / case / 'frozen_numeric'
        read = lambda name: json.loads((directory / (name + '.json')).read_text())
        trace, saved, actions = read('research_trace'), read('evidence_observations'), read('action_trace')
        question = next(q for q in trace['questions'] if q.get('answer', {}).get('frozen_transfer'))
        cut = question['opened_before_observation']
        experiment = trace['comparisons'][0]['selected_experiment']
        start = next(r for r in saved if r['index'] >= cut
            and r['observation'].get('dialog_box', {}).get('is_in_dialog')
            and _dialog(r['observation'])[0] == experiment['number'])
        planned = planned_segment(start, experiment['prediction'])
        actual = {r['index']: r for r in saved}
        comparisons = []
        for offset, forecast in enumerate(planned['rows'], 1):
            index = start['index'] + offset
            comparisons.append({'index': index,
                'action_matches': actions[index]['action'] == forecast['action'],
                'frequency_matches': _dialog(actual[index]['observation'])[1] == forecast['frequency']
                    if index in actual else None})
        prefixes = [r for r in saved if r['index'] < cut]
        available_menus = {str(_dialog(r['observation'])[0]): r['index'] for r in prefixes
            if r['observation'].get('dialog_box', {}).get('dialogOptions')}
        stale_text = {str(_dialog(r['observation'])[0]): _dialog(r['observation'])[1]
            for r in prefixes if _dialog(r['observation'])[0] is not None}
        entry = {'phase': phase, 'case': case, 'first_choice_cut': cut,
            'first_choice_saved_menus': available_menus, 'first_choice_saved_frequency_text': stale_text,
            'selected': {'number': experiment['number'], 'setting': experiment['prediction']},
            'selected_menu_missing_at_first_choice': str(experiment['number']) not in available_menus,
            'preparation_actions': [a['action'] for a in actions[cut:start['index'] + 1]],
            'planned': planned, 'comparison_with_later_records': comparisons}
        if phase == 'after' and case == CASES[0]:
            # Recovery is planned only after the actual positive and final refutation were paid.
            positive = next(r for r in trace['state']['response_settings']
                if r['crystal'] == 5 and r['activated'] and r['frequency'] == 2169.)
            end = actual[start['index'] + len(planned['rows'])]
            recovery = planned_segment(end, positive['frequency'])
            recovery['positive_evidence'] = [i for i in positive['evidence'] if i <= end['index']]
            recovery['action_match_indices'] = [a['index'] for a, p in zip(actions[50:52], recovery['rows'])
                if a['action'] == p['action']]
            recovery['saved_final_frequency_match'] = _dialog(actual[51]['observation'])[1] == recovery['rows'][-1]['frequency']
            recovery['missing_intermediate_observation'] = 50
            entry['recovery_after_observation_39'] = recovery
            models = trace['state']['frozen_response_knowledge']['models']
            feature_row = next(r for r in trace['state']['research_table']['targets'] if r['crystal'] == 5)
            features = {k: v['value'] for k, v in feature_row['measurements'].items()}
            values = [Program.from_dict(m['program']).predict(Observation(features, 0.)) for m in models]
            entry['prospective_positive_counts'] = [{'frequency': p['frequency'],
                'positive_models': sum(m['lower'] <= p['frequency'] - value <= m['upper']
                    for m, value in zip(models, values)), 'total_models': len(models)} for p in planned['rows']]
        rows.append(entry)

output = {'scope': 'Archived public menus only; predictions use one paid menu before later action/readout comparison. No task execution or truth lookup.',
    'first_choice_limitation': 'Target menu observations 6 and 8 occurred but were omitted from evidence-only export. Runtime full observations contain them. Stale C5 text lacks menu increments.',
    'assumption': 'Published signed menu increments remain available across interior steps. No boundary saturation, navigation or future activation is inferred.',
    'cases': rows}
destination = Path(__file__).with_name('saved_path_diagnosis.json')
destination.write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps({'cases': len(rows), 'matched_actions': sum(sum(c['action_matches'] for c in r['comparison_with_later_records']) for r in rows),
    'matched_frequency_readouts': sum(sum(c['frequency_matches'] is True for c in r['comparison_with_later_records']) for r in rows),
    'planned_steps': sum(r['planned']['step_count'] for r in rows), 'output': str(destination)}))
