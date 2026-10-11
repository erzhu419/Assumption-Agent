#!/usr/bin/env python3
"""Summarize completed band revisions and their saved physical counterexamples."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition import world_reference_recalibration as recipe

OUTPUT = ROOT / 'artifacts/response_pending_band_revision_20261011'


def audit(output):
    def read(path):
        return json.loads(path.read_text())

    summary = read(output / 'summary.json')
    assert summary['completed'] == summary['planned'] == 78, 'Complete frozen cohort before attribution'
    diagnostics = read(output / 'band_revision_diagnostic.json')
    revisions = []
    for diagnostic in diagnostics:
        if not diagnostic['revision_count']:
            continue
        directory = output / diagnostic['cohort'] / diagnostic['phase'] / diagnostic['case'] / 'frozen_numeric'
        trace = read(directory / 'research_trace.json')
        state = trace['state']
        trials = [*state.get('tests', []), *([state['test']] if state.get('test') else [])]
        records = read(directory / 'evidence_observations.json')
        current = records[-1]['observation']
        _, crystals, reactors, _, _ = recipe._scan(records, current)
        # Evidence export is sparse: the final positive is retained even when
        # its earlier TALK is absent. That TALK established the saved trial's
        # number/UUID binding. Restore this paid binding before parsing the
        # final readout; otherwise _response_settings silently skips it.
        crystals.update({str(trial['number']): trial['crystal'] for trial in trials})
        reactors.update({str(trial['number']): trial['reactor'] for trial in trials})
        # Compact state ends before some ordinary final trials. The saved final
        # public readout supplies their positive evidence, as in actual export.
        paid = state['response_settings'] + recipe._response_settings(records, crystals, reactors)
        features = {row['crystal']: {key: value['value'] for key, value in row['measurements'].items()}
                    for row in state['research_table']['references'] + state['research_table']['targets']}
        exported = read(directory / 'learned_response_knowledge.json')
        exported_keys = {(Program.from_dict(m['program']), m['lower'], m['upper']) for m in exported['models']}
        for row in diagnostic['revisions']:
            trial = next(t for t in trials if t['question'] == row['physical_question_id'])
            revision = trial['band_revision']
            fates = []
            for hypothesis in revision['hypotheses']:
                if hypothesis['id'] not in revision['positive_supporters']:
                    continue
                program = Program.from_dict(hypothesis['program'])
                predictions = {n: program.predict(Observation(values, 0.)) for n, values in features.items()}
                lower, upper = hypothesis['lower'], hypothesis['upper']
                contradictions = []
                for response in paid:
                    residual = response['frequency'] - predictions[response['crystal']]
                    if (lower <= residual <= upper) != response['activated']:
                        contradictions.append({'crystal': response['crystal'], 'frequency': response['frequency'],
                            'activated': response['activated'], 'observation': min(response['evidence']),
                            'residual': residual, 'distance_outside_band': max(lower - residual, residual - upper, 0.)})
                fates.append({'id': hypothesis['id'], 'band': [lower, upper],
                    'first_saved_counterexample': min(contradictions, key=lambda r: r['observation']) if contradictions else None,
                    'exact_model_exported': (program, lower, upper) in exported_keys})
            revisions.append({'cohort': diagnostic['cohort'], 'case': diagnostic['case'],
                'physical_question_id': row['physical_question_id'], 'revision_question_id': row['revision_question_id'],
                'number': row['number'], 'requested_setting': row['requested_setting'],
                'last_revision_evidence': max(row['evidence']), 'revision_status': row['revision_status'],
                'research_question_status': row['research_question_status'], 'compute_cost': row['compute_cost'],
                'hypothesis_count': row['hypothesis_count'], 'positive_supporter_count': row['positive_supporter_count'],
                'physically_confirmed': row['physically_confirmed'], 'positive_evidence': row['positive_evidence'],
                'revised_feedback_at_trial': row['revised_feedback'],
                'positive_supporter_final_fates': fates,
                'exported_revision_origin_models': sum(m.get('source_question') == row['revision_question_id'] for m in exported['models']),
                'exported_all_models': len(exported['models'])})
    cohorts = {}
    for cohort, aggregate in summary['cohorts'].items():
        pairs = [p for p in summary['paired'] if p['cohort'] == cohort]
        selected = [r for r in revisions if r['cohort'] == cohort]
        before, after = aggregate['overall']['before'], aggregate['overall']['after']
        charged = after['costs']['research_compute_units'] - before['costs']['research_compute_units']
        exported_compute = after['diagnostic_export_compute'] - before['diagnostic_export_compute']
        cohorts[cohort] = {'paired_cases': len(pairs), 'overall': aggregate['overall'],
            'cost_difference': {
                'physical_actions': after['costs']['environment_steps'] - before['costs']['environment_steps'],
                'charged_research_compute': charged, 'diagnostic_export_compute': exported_compute,
                'recorded_compute_including_export': charged + exported_compute},
            'casewise_success_preserved': all(p['success_before'] == p['success_after'] for p in pairs),
            'casewise_object_confirmation_preserved': all(p['confirmations_before'] == p['confirmations_after'] for p in pairs),
            'identical_full_action_traces': sum(p['full_actions_preserved'] for p in pairs),
            'changed_cases': [{key: p[key] for key in ('case', 'actions_difference', 'compute_difference',
                'first_different_action', 'confirmations_before', 'confirmations_after')} for p in pairs
                if p['actions_difference'] or p['compute_difference'] or not p['full_actions_preserved']],
            'revision_count': len(selected), 'revision_compute': sum(r['compute_cost'] for r in selected),
            'statuses': dict(Counter(r['revision_status'] for r in selected)),
            'revised_trials_confirmed': sum(r['physically_confirmed'] for r in selected),
            'exported_revision_origin_models': sum(r['exported_revision_origin_models'] for r in selected),
            'exported_positive_revision_bands': sum(f['exact_model_exported'] for r in selected for f in r['positive_supporter_final_fates'])}
    result = {'source': 'Completed 78 new runs plus 26 archived controls; read-only trace attribution',
        'cohorts': cohorts, 'revisions': revisions,
        'fresh_success_trace': {
            'case': 'square_original__gain_lower__adapted_law', 'cohort': 'new_objects',
            'evidence': 'At observation 32, C4=570 is inactive and the original band loses support for requested 569. A child study freezes the coefficients and compares a narrow band [878.5644995844,878.6404995407] with an admissible wider band [878.1062498223,878.6404995407]. They predict opposite outcomes at 569. Observation 33 activates 569, rejecting the narrow band. The baseline instead tries C5=2137 and fails before returning to C4. The revision reduces actions 57 to 52 and compute 815 to 469; both objects remain confirmed.',
            'later_boundary': 'Final C5=2135 activation at observation 51 gives residual 876.6062507234, outside the retained wider band. Its local success therefore does not yield a surviving shared-band archive.'},
        'conclusion': 'Band revision rescues three pending physical experiments after their old numerical bands fail, including one at new target values. Every case preserves task success and object-level confirmation. Physical actions decrease by four in preservation and five on new objects. Charged research computation decreases by 717 and 31, but adding diagnostic exports gives changes of -201 and +22 respectively: this is a small computational tradeoff, not a demonstrated full-cost net gain on the new objects.',
        'remaining_question': 'All three locally successful revisions are refuted by final C5 evidence and export no surviving revision model. Turning local interval revisions into reusable knowledge requires evidenced applicability conditions; a refuted shared-band model cannot simply be reactivated.',
        'limitations': [
            'The development fixture and coefficient families remain designed and familiar. New objects are not new mechanism families or independent random seeds.',
            'Final positive responses on another object can refute locally successful bands. Exported revision-origin count and exact-band membership are reported separately, so provenance deduplication cannot be mistaken for a failed export.',
            'In the preserved square_original original-law-return case, one final C5 counterexample exceeds a revised upper boundary by only about 7.65e-7. The other revised positive bands fail by much larger amounts; no tolerance was tuned or run repeated.',
            'Only saved paid records are used to locate explanatory counterexamples; the actual exporter checked the complete final observation history. This audit creates no new scientific evidence.']}
    (output / 'outcome_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({cohort: {key: value[key] for key in ('paired_cases', 'casewise_success_preserved',
        'casewise_object_confirmation_preserved', 'revision_count', 'revised_trials_confirmed',
        'exported_revision_origin_models', 'exported_positive_revision_bands')} for cohort, value in cohorts.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    audit(args.output.resolve())
