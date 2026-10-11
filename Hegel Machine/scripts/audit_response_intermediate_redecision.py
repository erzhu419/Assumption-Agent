#!/usr/bin/env python3
"""Attribute the completed cohort from saved traces; no execution or fitting."""
import argparse
from functools import lru_cache
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/response_intermediate_redecision_20261011'


def audit(output):
    def read(path):
        return json.loads(path.read_text())

    @lru_cache(None)
    def case(phase, name):
        directory = output / phase / name / 'frozen_numeric'
        trace = read(directory / 'research_trace.json')
        return trace, read(directory / 'brief.json'), {q['id']: q for q in trace['questions']}

    summary = read(output / 'summary.json')
    diagnostics = read(output / 'diagnostic_brief.json')
    pairs = {p['case']: p for p in summary['paired']}
    retirements, regressions = [], []
    for diagnostic in diagnostics:
        if diagnostic['phase'] != 'after' or not diagnostic['retirement_count']:
            continue
        name = diagnostic['case']
        trace, brief, questions = case('after', name)
        old_trace, old_brief, old_questions = case('before', name)
        for retired in diagnostic['retirements']:
            qid = retired['question_id']
            question, details = questions[qid], retired['retirement']
            last = max(details['evidence'])
            observed = question['answer']['observed_responses']
            trigger = next(row for row in observed if last in row['evidence'])
            next_qid = retired['next_comparison_question']
            next_question = questions.get(next_qid, {})
            next_point = next((point for point in brief['point_confirmations']
                               if point['question_id'] == str(next_qid) + '_physical'), None)
            old_point = next((point for point in old_brief['point_confirmations']
                             if point['question_id'] == qid), None)
            retirements.append({'case': name, 'question_id': qid,
                'number': retired['number'], 'requested_setting': retired['requested_setting'],
                'trigger': trigger, 'question_status': question['status'],
                'supporters_remaining': sorted(set(details['requested_supporters']) & set(details['remaining_hypotheses'])),
                'retained_member_count_at_retirement': len(details['remaining_hypotheses']),
                'last_evidence': last, 'next_research_opened_before_observation': next_question.get('opened_before_observation'),
                'paid_evidence_precedes_reselection': last < next_question.get('opened_before_observation', -1),
                'next_status': retired['next_status'],
                'next_research_stage': (retired['next_frozen_transfer'] or {}).get('research_stage'),
                'next_selected_point': {key: (retired['next_selected_experiment'] or {}).get(key) for key in ('number', 'prediction')},
                'next_selected_point_physically_confirmed': None if next_point is None else next_point['physically_confirmed'],
                'later_physically_confirmed_points': [
                    {'number': point['number'], 'prediction': point['prediction'], 'evidence': point['actual_positive_evidence']}
                    for point in brief['point_confirmations'] if point['physically_confirmed']
                    and questions.get(point['question_id'], {}).get('opened_before_observation', -1) > last],
                'same_trial_physically_confirmed_in_baseline': None if old_point is None else old_point['physically_confirmed'],
                'actions_difference': pairs[name]['actions_difference'],
                'research_compute_difference': pairs[name]['compute_difference']})
        if pairs[name]['actions_difference'] > 0:
            compute = lambda saved: sum(item['compute_cost'] for item in saved['comparisons'])
            regressions.append({'case': name,
                'actions_difference': pairs[name]['actions_difference'],
                'research_compute_difference': pairs[name]['compute_difference'],
                'comparison_compute_before': compute(old_trace), 'comparison_compute_after': compute(trace),
                'comparison_cost_explains_full_compute_difference': compute(trace) - compute(old_trace) == pairs[name]['compute_difference'],
                'comparison_sequence_before': [{'question_id': c['question_id'], 'compute_cost': c['compute_cost'],
                    'status': c['status']} for c in old_trace['comparisons']],
                'comparison_sequence_after': [{'question_id': c['question_id'], 'compute_cost': c['compute_cost'],
                    'status': c['status']} for c in trace['comparisons']],
                'explanation': 'At 471 the paid negative response removes all finite members supporting 472. The baseline nevertheless activates at its next setting, 472. Early withdrawal visits C5 first and returns to a fitted C4 setting near 472, adding two physical actions. The extra and changed comparison calls explain the full compute increase. Finite-member refutation is not endpoint impossibility.'})
    preserved = read(output / 'baseline_preservation.json')
    result = {'source': 'Saved completed paired run; read-only attribution, no new simulation or model calls',
        'completed_executions': summary['completed'],
        'casewise_confirmation_preserved': all(p['confirmations_before'] == p['confirmations_after'] for p in pairs.values()),
        'casewise_success_preserved': all(p['success_before'] == p['success_after'] for p in pairs.values()),
        'identical_full_action_traces': sum(p['full_actions_preserved'] for p in pairs.values()),
        'changed_action_cases': sum(not p['full_actions_preserved'] for p in pairs.values()),
        'retirement_count': len(retirements),
        'retirements_inconclusive': sum(r['question_status'] == 'inconclusive' for r in retirements),
        'old_baseline_exact_cases': sum(all(row[key] for key in ('full_action_trace_preserved', 'endpoints_preserved', 'research_compute_preserved')) for row in preserved),
        'retirements': retirements, 'regressions': regressions,
        'positive_counterexample': {
            'case': 'cubic_shifted_2__gain_higher__adapted_law',
            'trajectory': 'Both arms observe actual C5 activation at 2169 (observation 37), eliminating the sole supporter of requested 2167. The candidate exits at observation 38 and selects C4=309 in continuation_recall at observation 39. It never visits 2168 or 2167 and preserves active C5; C4 is confirmed at observation 45.',
            'effect': 'Six fewer physical actions and 20 fewer research units; C5 requested-point confirmation remains false in both arms.'},
        'next_core': {
            'finding': 'Online feedback now reaches parent research and preserves all casewise confirmations. No additional implementation fault was found in the completed traces.',
            'remaining_question': 'Finite knowledge can lose support even when the requested endpoint would succeed: both square_original gain_higher cases demonstrate this. A decision about keeping a nearby informative endpoint versus switching research requires target value beyond finite-member survival.',
            'needed_evidence': 'Before changing this policy, inspect historical expected-task-value and family-consequence mechanisms. Evaluate both continuation and interruption under matched paid evidence on new or frozen development conditions, including finite-archive misspecification and already active targets. Do not promote a one-step-distance rule or tune to 472 from these two outcomes.',
            'claim_boundary': 'This cohort establishes a small cost reduction and evidence-driven replanning in an offline controller, not extra hypothesis confirmations or universal recursive-agent superiority.'}}
    (output / 'outcome_audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = audit(args.output.resolve())
    print(json.dumps({key: result[key] for key in ('completed_executions', 'casewise_confirmation_preserved',
        'identical_full_action_traces', 'changed_action_cases', 'retirement_count', 'retirements_inconclusive')}, indent=2))
