"""Attribute the completed frozen-knowledge comparison using saved paid rows only."""
from collections import Counter
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation


def read(path):
    return json.loads(path.read_text())


def source(model, archive):
    return model.get('source_id', archive['source_id'])


def revised(model):
    return model.get('band_revision', {}).get('origin') == 'final_paid_response_band_revision'


def audit_case(case, phase):
    directory = OUT / phase / case / 'frozen_numeric'
    trace, paid, brief = (read(directory / name) for name in (
        'research_trace.json', 'final_paid_inputs.json', 'brief.json'))
    archive = trace['state']['frozen_response_knowledge']
    features = {row['crystal']: {name: item['value'] for name, item in row['measurements'].items()}
        for rows in paid['research_table'].values() if isinstance(rows, list) for row in rows}
    rows = sorted(paid['response_settings'], key=lambda row: min(row['evidence']))
    models = []
    for item in archive['models']:
        program = Program.from_dict(item['program'])
        predictions = {n: program.predict(Observation(values, 0.)) for n, values in features.items()}
        contradictions = []
        for row in rows:
            residual = row['frequency'] - predictions[row['crystal']]
            predicted = item['lower'] <= residual <= item['upper']
            if predicted != row['activated']:
                contradictions.append({'observation': min(row['evidence']), 'number': row['crystal'],
                    'setting': row['frequency'], 'actual': row['activated'], 'predicted': predicted,
                    'residual': residual, 'band': [item['lower'], item['upper']]})
        models.append({'saved': item, 'program': program, 'predictions': predictions,
                       'first_counterexample': contradictions[0] if contradictions else None})
    questions = {question['id']: question for question in trace['questions']}
    confirmations = {item['question_id']: item for item in brief['point_confirmations']}
    trials = {trial['question']: trial for trial in
        [*trace['state'].get('tests', []), trace['state'].get('test', {})] if 'question' in trial}
    decisions = []
    for comparison in trace['comparisons']:
        transfer = comparison.get('frozen_transfer')
        if not transfer:
            continue
        question_id = comparison['question_id']
        cutoff = questions[question_id]['opened_before_observation']
        prefix = [row for row in rows if min(row['evidence']) < cutoff]
        retained = [model for model in models if not model['first_counterexample']
                    or model['first_counterexample']['observation'] >= cutoff]
        counts = Counter(source(model['saved'], archive) for model in retained)
        # If the paid-prefix binding disagrees, this audit must not attribute a
        # selection to reconstructed member ids.
        assert len(prefix) == comparison['response_count'], (case, phase, question_id, 'prefix')
        assert len(retained) == transfer['retained_count'], (case, phase, question_id, 'retained')
        for name, saved in transfer.get('source_counts', {}).items():
            assert counts[name] == saved['retained_count'], (case, phase, question_id, name)
        decision = {'question': question_id, 'opened_before_observation': cutoff,
            'stage': transfer.get('research_stage'), 'transfer_status': transfer['status'],
            'source_counts': transfer.get('source_counts', {}),
            'new_models_retained': sum(revised(model['saved']) for model in retained)}
        selected = comparison.get('selected_experiment')
        if selected and transfer['status'] == 'selected':
            members, seen = {}, set()
            for index, model in enumerate(retained):
                item = model['saved']
                key = (model['program'], item['lower'], item['upper'])
                if key not in seen:
                    seen.add(key)
                    members[f'joint_response_{index}'] = model
            predictions = {key: model['saved']['lower'] <= selected['prediction']
                - model['predictions'][selected['number']] <= model['saved']['upper']
                for key, model in members.items()}
            assert predictions == selected['predictions'], (case, phase, question_id, 'predictions')
            chosen = members[selected['candidate_id'].replace('joint_coefficient_', 'joint_response_')]
            supporters = [model for key, model in members.items() if predictions[key]]
            physical = questions.get(question_id + '_physical', {})
            answer = physical.get('answer', {})
            feedback = answer.get('response_feedback', answer)
            trial = trials.get(question_id + '_physical', {})
            revision = trial.get('band_revision', {})
            confirmation = confirmations.get(question_id + '_physical', {})
            new_positive = [model for model in supporters if revised(model['saved'])]
            actual = [{'observation': index, 'number': row['crystal'], 'setting': row['frequency'],
                       'activated': row['activated']}
                for row in rows if row['crystal'] == selected['number']
                and row['frequency'] == selected['prediction']
                for index in row['evidence'] if index >= cutoff]
            decision['selection'] = {'number': selected['number'], 'setting': selected['prediction'],
                'candidate_source': source(chosen['saved'], archive),
                'candidate_is_new_revised_model': revised(chosen['saved']),
                'positive_support_by_source': dict(Counter(source(model['saved'], archive) for model in supporters)),
                'new_positive_supporters': len(new_positive),
                'physical_status': physical.get('status'),
                'requested_point_confirmed': confirmation.get('physically_confirmed', False),
                'actual_requested_responses': actual,
                'new_members_remaining_after_physical_feedback': sum(revised(members[key]['saved'])
                    for key in feedback.get('remaining_hypotheses', []) if key in members),
                'pending_band_revision': {'proposal_status': revision.get('status'),
                    'question': revision.get('question_id'),
                    'feedback': feedback.get('band_revision')} if revision else None,
                'retirement': trial.get('retirement')}
        decisions.append(decision)
    new = [model for model in models if revised(model['saved'])]
    grouped = {}
    for model in new:
        contradiction = model['first_counterexample']
        if contradiction:
            key = (contradiction['observation'], contradiction['number'], contradiction['setting'], contradiction['actual'])
            group = grouped.setdefault(key, {'models': 0, 'example': contradiction})
            group['models'] += 1
    return {'new_model_count': len(new),
        'new_models_compatible_with_final_paid_rows': sum(model['first_counterexample'] is None for model in new),
        'new_model_first_refutations': list(grouped.values()), 'decisions': decisions}, brief


def main():
    summary = read(OUT / 'summary.json')
    assert summary['status'] == 'finished' and summary['completed'] == 24, 'Wait for all fixed runs'
    pairs, phase_audits, briefs = [], {}, {}
    for pair in summary['paired']:
        case = pair['case']
        phase_audits[case], briefs[case] = {}, {}
        for phase in ('before', 'after'):
            phase_audits[case][phase], briefs[case][phase] = audit_case(case, phase)
        after = phase_audits[case]['after']
        selections = [d['selection'] for d in after['decisions'] if 'selection' in d]
        changed = {number: [pair['quality']['before']['requested_point_confirmed'][number],
                           pair['quality']['after']['requested_point_confirmed'][number]]
            for number in ('4', '5') if pair['quality']['before']['requested_point_confirmed'][number]
            != pair['quality']['after']['requested_point_confirmed'][number]}
        export_delta = briefs[case]['after']['knowledge_export_compute'] - briefs[case]['before']['knowledge_export_compute']
        pairs.append({**pair, 'requested_confirmation_changes': changed,
            'diagnostic_export_compute_difference': export_delta,
            'charged_plus_diagnostic_compute_difference': pair['compute_difference'] + export_delta,
            'new_model_usage': {'archive_models': after['new_model_count'],
                'decisions_with_new_positive_support': sum(item['new_positive_supporters'] > 0 for item in selections),
                'selected_candidate_from_new_source': sum(item['candidate_is_new_revised_model'] for item in selections),
                'confirmed_selected_candidates_from_new_source': sum(item['candidate_is_new_revised_model']
                    and item['requested_point_confirmed'] for item in selections),
                'confirmed_requests_with_new_positive_support': sum(item['new_positive_supporters'] > 0
                    and item['requested_point_confirmed'] for item in selections),
                'all_new_models_rejected_in_recorded_recall': any(
                    decision['new_models_retained'] == 0 for decision in after['decisions']),
                'final_compatible_models': after['new_models_compatible_with_final_paid_rows']}})
    groups = {}
    for condition in ('source_gain', 'shifted_gain'):
        subset = [pair for pair in pairs if pair['condition'] == condition]
        groups[condition] = {'pairs': len(subset), 'quality': summary['conditions'][condition],
            'actions_difference': sum(pair['actions_difference'] for pair in subset),
            'charged_compute_difference': sum(pair['compute_difference'] for pair in subset),
            'diagnostic_export_compute_difference': sum(pair['diagnostic_export_compute_difference'] for pair in subset),
            'charged_plus_diagnostic_compute_difference': sum(pair['charged_plus_diagnostic_compute_difference'] for pair in subset),
            'cases_new_models_all_refuted_by_final_paid_rows': sum(
                pair['new_model_usage']['final_compatible_models'] == 0 for pair in subset),
            'cases_with_new_source_candidate_selected': sum(
                pair['new_model_usage']['selected_candidate_from_new_source'] > 0 for pair in subset),
            'cases_with_confirmed_new_source_candidate': sum(
                pair['new_model_usage']['confirmed_selected_candidates_from_new_source'] > 0 for pair in subset),
            'cases_all_new_models_rejected_in_recorded_recall': sum(
                pair['new_model_usage']['all_new_models_rejected_in_recorded_recall'] for pair in subset)}
    source_delta = summary['source_reexport_compute']['after'] - summary['source_reexport_compute']['before']
    quality_losses = [pair['case'] for pair in pairs if
        (pair['quality']['before']['success'] and not pair['quality']['after']['success'])
        or any(old and not new for old, new in pair['requested_confirmation_changes'].values())
        or pair['quality']['after']['unsupported_positive_verdicts'] > pair['quality']['before']['unsupported_positive_verdicts']]
    result = {'scope': summary['source_scope'], 'completed': 24, 'paired_completed': 12,
        'evidence': 'All attributions reconstruct archive-member identities from actual saved paid prefixes. No fitting, environment execution, or successor labels used to construct source knowledge.',
        'quality': summary['overall'], 'case_level_quality_losses': quality_losses,
        'by_condition': groups,
        'source_reexport_compute': summary['source_reexport_compute'],
        'total_charged_plus_diagnostic_compute_difference': sum(pair['charged_plus_diagnostic_compute_difference'] for pair in pairs),
        'with_once_only_source_reexport_difference': source_delta + sum(pair['charged_plus_diagnostic_compute_difference'] for pair in pairs),
        'previously_paid_source_execution': summary['source_execution'],
        'pairwise': pairs, 'after_knowledge_traces': {case: phases['after'] for case, phases in phase_audits.items()},
        'representative_paired_traces': {case: phases for case, phases in phase_audits.items()
            if case.startswith('preservation__square_original__gain_higher__adapted_law__objects_2')},
        'interpretation': {
            'supported': 'Final source interval revision produced frozen knowledge that was actually selected and physically tested on new objects, with a paired improvement in one requested-point confirmation and aggregate compute reduction.',
            'positive_trace': 'adapted_law/objects_2/source_gain: C4=884 is confirmed at observation 38; all inherited members then fail. Ten new-source members support continuation; its selected C5=3237 is confirmed at observation 55 after a further online band revision. Before has no frozen survivor at that continuation and never confirms its requested C5 point.',
            'negative_trace': 'adapted_law/objects_2/shifted_gain: the new source selects C5=3237, but intervening paid responses refute both frozen and locally revised windows. The original question remains inconclusive and a later recall rejects the new archive; it is never counted as a confirmed request.',
            'remaining_core': 'Under the fixed 1% gain change, correct withdrawal is demonstrated but successor research increases by five actions and 11787 compute units including diagnostic exports. The next evidence gap is using those failed transfer observations to start productive revised research while preserving same-gain transfer, rather than further proving that source knowledge reaches the controller.'},
        'limitations': ['Twelve deterministic successor conditions of three related source branches; not independent mechanisms or LLM trials.',
            'A confirmed requested physical point does not confirm every frozen interval; online band revision may remain necessary.',
            'Source-gain and shifted-gain effects must remain separate; diagnostic exports and one-time source re-export are reported separately from charged execution.']}
    (OUT / 'outcome_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('completed', 'paired_completed', 'by_condition',
        'total_charged_plus_diagnostic_compute_difference', 'with_once_only_source_reexport_difference')}, indent=2))


if __name__ == '__main__':
    main()
