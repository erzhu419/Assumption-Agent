"""Return already-paid same-case evidence when a revised method fails."""

from copy import deepcopy
import json
from pathlib import Path

from hegel_machine.cognition.code_provider_context import prepare_code_context
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from hegel_machine.cognition.temporal_paired_practice import practice_execution_summary
from test_python_method_learning import archived_predictor, make_learner, practice


FAILED_METHOD = r'''import json
from pathlib import Path
def select_actions(spec, records, memory):
    if json.load(open('research_state.json'))['method_results']:
        return {'stop': 'inspect failed confirmation'}
    table = {row['features']['u']: row['target'] for record in records for row in record['rows']}
    Path('bad.py').write_text('TABLE = ' + repr(table) +
        '\ndef predict(row):\n    return TABLE.get(row["features"]["u"], 0.0)\n')
    return {'model': 'bad.py'}
'''


def test_actual_failure_exposes_same_case_success_despite_other_case_failure(
        tmp_path, archived_predictor, monkeypatch):
    author, prepared = make_learner(tmp_path, archived_predictor)
    supported = practice(author, 'case_a')
    other_case = practice(author, 'case_b')
    assert supported['fresh_confirmation']['passed']
    assert other_case['status'] != 'completed'
    author._execute('write', {'path': METHOD_PATH, 'code': FAILED_METHOD}, author.interface)
    failed = practice(author, 'case_a')
    assert failed['fresh_confirmation']['passed'] is False
    pair = failed['failed_case_paid_comparison']
    assert pair['failed_practice']['directory'] == 'trials/trial_002'
    assert pair['previous_supported_practice']['directory'] == 'trials/trial_000'
    assert pair['previous_supported_practice']['execution_summary']['latest_confirmation']['passed']
    assert not pair['failed_practice']['execution_summary']['latest_confirmation']['passed']
    assert len(prepared) == len(author.practice_trials) == 3
    costs = deepcopy(author.costs)
    assert costs['model_calls'] == costs['reported_tokens'] == 0
    assert costs['observation_cost'] == sum(x['costs']['observation_cost'] for x in
        (supported, other_case, failed))

    # Older ledgers lack the richer execution summary. Read only the selected
    # paid trial once, leave its record unchanged, and retain the view on resume.
    author.practice_trials[0].pop('execution_summary')
    ledger = deepcopy(author.practice_trials)
    paid_result = author.workspace / 'trials/trial_000/result.json'
    read_text, reads = Path.read_text, []

    def counted_read(path, *args, **kwargs):
        if path == paid_result:
            reads.append(path)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', counted_read)
    for _ in range(2):
        author._prepare_program_state()
        decision = json.loads(author.files['method_decision.json'])
        assert decision['failed_case_paid_comparison'] == pair
    assert reads == [paid_result]
    context = prepare_code_context(author.workspace, {'files': dict(author.files),
        'events': [], 'observations': [], 'task_interface': author.common_interface,
        'session_knowledge': author.knowledge}, schema={}, instructions='test')
    assert context['method_decision']['failed_case_paid_comparison'] == pair
    assert author.practice_trials == ledger and author.costs == costs
    assert author.files[METHOD_PATH] == FAILED_METHOD and author.published_source is None


def test_same_source_later_same_case_failure_removes_its_local_support(tmp_path, archived_predictor):
    author, _ = make_learner(tmp_path, archived_predictor)
    practice(author, 'case_a')
    old = deepcopy(author.practice_trials[0])
    # Bookkeeping-only ordering: this is not a numerical or scientific result.
    refuted = {**deepcopy(old), 'directory': 'trials/trial_001',
        'status': 'confirmation_failed', 'fresh_confirmation': {'passed': False}}
    new_failed = {**deepcopy(refuted), 'directory': 'trials/trial_002',
        'method_source': FAILED_METHOD}
    author.practice_trials.extend((refuted, new_failed))
    costs = deepcopy(author.costs)
    assert author._failed_case_paid_comparison() is None
    assert author.costs == costs and len(author.practice_trials) == 3
    # A genuine later recovery would restore support for that source and case.
    recovered = {**deepcopy(old), 'directory': 'trials/trial_003'}
    author.practice_trials.insert(-1, recovered)
    pair = author._failed_case_paid_comparison()
    assert pair['previous_supported_practice']['directory'] == 'trials/trial_003'
    author.practice_trials.append({**deepcopy(new_failed), 'status': 'completed',
        'fresh_confirmation': {'passed': True}})
    assert author._failed_case_paid_comparison() is None


def test_execution_summary_preserves_measured_research_sequence_without_large_arrays():
    result = {'status': 'completed', 'method_results': [
        {'step': 0, 'proposal': {'action': 'run', 'arguments': {
            'operator': 'compare_experiments'}}, 'feedback': {'comparison': {
            'index': 0, 'status': 'compared', 'evidence_cutoff': {'rows': 72},
            'prior_fit': [{'model': 'candidate.py', 'mse': 1e-31,
                'evaluation_scope': 'paid_fit', 'fitting_view': {'rows': 72}}],
            'selected_experiment': {'id': 'long', 'predicted_separation': .06,
                'observation_cost': 48, 'actions': ['large actions'] * 10000}}}},
        {'step': 1, 'proposal': {'action': 'query', 'arguments': {'actions': [
            {'kind': 'trajectory', 'sequence': [{'u': 0.}] * 24},
            {'kind': 'trajectory', 'sequence': [{'u': 1.}] * 24}]}},
            'feedback': {'observation_cost': 48, 'revision': {'status': 'observed',
                'action_row_counts': [24, 24], 'forecasts': [{'model': 'candidate.py',
                    'prospective_mse': 1.3, 'scored_rows': 48, 'complete_experiment': True,
                    'predictions': [0.] * 10000, 'prediction_inputs': ['large'] * 10000}]}}},
        {'step': 2, 'proposal': {'model': 'refitted.py'}, 'feedback': {'mse': 5e-32,
            'evaluation_scope': 'paid_fit', 'fitting_view': {'rows': 120},
            'fresh_confirmation': {'passed': True, 'mse': 3e-32, 'rows': 96}}}]}
    summary = practice_execution_summary(result, 'trials/trial_115')
    compare, acquire, submit = summary['research_steps']
    assert compare['operator'] == 'compare_experiments'
    assert compare['comparison']['prior_fit'][0]['fitting_view']['rows'] == 72
    assert acquire['trajectory_lengths'] == [24, 24]
    assert acquire['observation_cost'] == 48
    assert acquire['revision']['forecasts'][0]['prospective_mse'] == 1.3
    assert submit['paid_score']['fitting_view']['rows'] == 120
    assert summary['latest_submission_paid_fit']['mse'] == 5e-32
    assert summary['latest_confirmation']['mse'] == 3e-32
    assert 'latest_evaluation' not in summary
    assert len(json.dumps(summary)) < 3000
