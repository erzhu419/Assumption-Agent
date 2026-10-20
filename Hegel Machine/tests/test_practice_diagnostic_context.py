"""Paid practice diagnostics retain errors without resending complete row histories."""

from copy import deepcopy
import json

from hegel_machine.cognition.code_provider_context import prepare_code_context
from hegel_machine.cognition.code_provider_context import _compact_practice_feedback


def test_repeated_evaluation_feedback_keeps_scores_and_retrievable_diagnostic_details(tmp_path):
    rows = [{'row_index': index, 'features': {'u': index}, 'target': 2.,
             'history': [{'u': previous / 100} for previous in range(300)],
             'prediction': 3., 'squared_error': 1.} for index in range(3)]
    residuals = {'evaluation_scope': 'paid_response_evaluation', 'model': 'candidate.py',
                 'evaluated_rows': 2901, 'examples': rows}
    terms = [{'row_index': index, 'coefficient': .5,
              'measured_value': 2., 'predicted_value': 3.} for index in range(5)]
    revisions = [{'revision_index': index, 'version': 2, 'check_count': 8,
                  'max_abs_residual': 1., 'examples': [
                      {'id': f'check_{index}_{row}', 'scope': {'records': [4, 7]},
                       'measured_value': 2., 'predicted_value': 3., 'residual': 1.,
                       'evidence_status': 'supported', 'terms': deepcopy(terms)}
                      for row in range(3)]} for index in range(11)]
    diagnostic = {'scope': 'paid_model_consistency', 'evaluated_rows': 2901,
                  'revision_count': 11, 'check_count': 88, 'term_count': 440,
                  'revisions': revisions}
    feedback = {'mse': .25, 'model': 'candidate.py', 'paid_residuals': residuals,
                'paid_relation_diagnostic': diagnostic,
                'research_repairs': [{'model': 'old.py', 'current_evaluation': {
                    'mse': .5, 'evaluation_scope': 'paid_response_evaluation',
                    'paid_residuals': deepcopy(residuals)}}]}
    request = {'events': [{'action': 'run', 'arguments': {}, 'feedback': {'edit_practice': {
        'directory': 'trials/trial_012', 'steps': [
            {'proposal': {'action': 'evaluate', 'arguments': {'model': 'candidate.py'}},
             'feedback': deepcopy(feedback)} for _ in range(12)]}}}],
        'files': {}, 'session_knowledge': {}, 'observations': []}
    original = deepcopy(request)
    context = prepare_code_context(tmp_path, request, schema={}, instructions='fixture')
    stored = json.loads((tmp_path / 'research_context.json').read_text())
    steps = context['events'][0]['feedback']['edit_practice']['steps']
    assert len(steps) == 12 and stored == request == original

    def read(reference):
        assert reference['full_content_file'] == 'research_context.json'
        value = stored
        for key in reference['json_path']:
            value = value[key]
        return value

    for step_index, step in enumerate(steps):
        actual = step['feedback']
        assert actual['mse'] == .25 and actual['model'] == 'candidate.py'
        residual = actual['paid_residuals']
        assert residual['evaluated_rows'] == 2901 and residual['example_count'] == 3
        if step_index:
            assert read(residual['examples']) == rows
        else:
            for index, example in enumerate(residual['examples']):
                assert 'history' not in example and example['history_length'] == 300
                assert example['prediction'] == 3. and example['squared_error'] == 1.
                assert read(example) == rows[index]
        relation = actual['paid_relation_diagnostic']
        assert relation['check_count'] == 88 and relation['term_count'] == 440
        assert len(relation['revisions']) == 11
        for index, revision in enumerate(relation['revisions']):
            assert revision['revision_index'] == index and revision['max_abs_residual'] == 1.
            assert revision['check_count'] == 8 and revision['version'] == 2
            if step_index:
                assert read(revision['examples']) == revisions[index]['examples']
            else:
                for row, example in enumerate(revision['examples']):
                    assert example['scope'] == {'records': [4, 7]} and example['residual'] == 1.
                    assert read(example) == revisions[index]['examples'][row]
                    assert read(example['terms']) == terms
        repair = actual['research_repairs'][0]['current_evaluation']
        assert repair['mse'] == .5
        assert read(repair['paid_residuals']['examples']) == rows
    assert len(json.dumps(context)) < len(json.dumps(request)) * .65


def test_new_evidence_or_changed_residual_keeps_its_own_inline_examples(tmp_path):
    first = {'evaluated_rows': 7, 'model': 'old.py', 'evaluation_scope': 'paid_response_evaluation',
             'examples': [{'row_index': 6, 'history': [{'u': 1}], 'prediction': 1.,
                           'target': 2., 'squared_error': 1.}]}
    changed = deepcopy(first)
    changed.update(evaluated_rows=8, model='new.py')
    changed['examples'][0].update(row_index=7, prediction=2., squared_error=0.)
    request = {'events': [{'action': 'run', 'arguments': {'operator': 'practice'},
        'feedback': {'steps': [{'feedback': {'paid_residuals': item}} for item in
            (first, deepcopy(first), changed)]}}], 'files': {}, 'session_knowledge': {}}
    context = prepare_code_context(tmp_path, request, schema={}, instructions='fixture')
    results = context['events'][0]['feedback']['steps']
    assert isinstance(results[1]['feedback']['paid_residuals']['examples'], dict)
    last = results[2]['feedback']['paid_residuals']
    assert last['evaluated_rows'] == 8 and last['model'] == 'new.py'
    assert last['examples'][0]['row_index'] == 7 and last['examples'][0]['squared_error'] == 0.


def test_restored_relation_feedback_keeps_candidate_and_historical_status_separate():
    steps = []
    for step, model, residual in [(110, 'histgb.py', 1.0754), (114, 'interaction.py', 1.158)]:
        steps.append({'step': step, 'feedback': {
            'evaluation_scope': 'paid_fit',
            'paid_residuals': {'model': model, 'evaluated_rows': 3029, 'examples': []},
            'evaluated_model': {'model': 'retained/model.py', 'mse': 0.},
            'paid_relation_diagnostic': {'evaluated_rows': 3029, 'revisions': [
                {'revision_index': 12, 'examples': [{'evidence_status': 'satisfied',
                 'measured_value': 2., 'predicted_value': 2. + residual, 'residual': residual}]}]}}})
    original = deepcopy(steps)
    shown = _compact_practice_feedback(steps, ['steps'], 'saved_practices.json')
    for index, step in enumerate(shown):
        relation = step['feedback']['paid_relation_diagnostic']
        assert relation['model'] == original[index]['feedback']['paid_residuals']['model']
        assert relation['evaluation_scope'] == 'paid_fit'
        assert relation['evaluation_feedback'] == {
            'full_content_file': 'saved_practices.json', 'json_path': ['steps', index, 'feedback']}
        example = relation['revisions'][0]['examples'][0]
        assert example['historical_check'] == {'status': 'satisfied'}
        assert example['residual'] == original[index]['feedback'][
            'paid_relation_diagnostic']['revisions'][0]['examples'][0]['residual']
        assert 'not the current predictor accuracy' in relation['field_meanings']['evidence_status']
    assert steps == original
