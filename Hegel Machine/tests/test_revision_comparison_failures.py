"""Paid ordinary comparisons refute exact exported instances without new execution."""
from copy import deepcopy

from hegel_machine.cognition import common_archive_inquiry
from hegel_machine.cognition.finite_native_author import export_revision_candidates
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.question_routing import register_theories, resolved_theories
from hegel_machine.cognition.research_episodes import load_source_values
from hegel_machine.cognition.theory_consolidation import _checked_transfer, install_theory_library
from scripts.run_anonymous_revision_inquiry import RegisteredOrganizationAgent
from test_finite_native_author import ChoiceLab
from test_inquiry_revision import action
from test_native_condition_inquiry import PaidConditionalLab
from test_registered_revision_target import revision_scene


def comparison_scene(tmp_path):
    agent, lab, declaration = revision_scene(tmp_path)
    owner = next(q for q in agent.questions if q['id'] == declaration['id'])
    target = owner['theory_target']
    identifier = f'frozen_recall:{owner["id"]}@1:fixture:binding:0:fit:0'
    parent = next((name, value) for name, value in agent.theory_index.items()
                  if value.get('source_ref', {}).get('kind') == 'native_candidate')
    criterion = {'kind': 'paid_mse', 'evaluated_rows': len(agent.rows),
        'mse_threshold': agent.config.confirmation_mse_threshold,
        'criterion_ref': {'file': 'research_state.json',
                          'json_path': ['limits', 'confirmation_mse_threshold']}}
    checks = [{'source_ref': {'file': 'laboratory_observations.json',
                'json_path': [ri, 'rows', row]}, 'context_ref': deepcopy(target['context_ref']),
               'criterion': deepcopy(criterion)}
              for ri, record in enumerate(agent.observations)
              for row in range(len(record['rows']))]
    binding = {'library_path': 'shared/theory_library.json', 'candidate_id': 'fixture',
        'candidate_version': 1, 'components': {}, 'context_ref': deepcopy(target['context_ref']),
        'fitting': {'evidence_refs': [deepcopy(item['source_ref']) for item in checks]},
        'parent': {'component_id': parent[0], 'version': parent[1]['version'],
                   'source_ref': deepcopy(parent[1]['source_ref'])}}
    # This real zero-allowance transfer has no executed point checks. Its bad
    # fixed Program becomes known to fail only through the ordinary comparison.
    transfer = _checked_transfer(agent, {'theory_id': identifier,
        'role_signature': target['role_signature'], 'output_unit': target['output_unit'],
        'context_ref': target['context_ref'], 'evidence_checks': checks},
        {'id': 'fixture'}, Program('const', value=0.), binding, compute_allowance=0)
    assert transfer['status'] == 'UNKNOWN'
    owner['frozen_recall_execution'] = {'basis': {'target': deepcopy(target)},
        'result': {'work': [{'theory_ids': [identifier]}]}}
    exported = export_revision_candidates(agent, owner)
    assert exported['new_exports'] == 1 and not exported['errors']
    models = [*agent.knowledge['finite_native_author']['predictor_paths']]
    models.extend(candidate['model'] for round_ in owner['native_condition_research']
                  for candidate in round_['candidates'])
    models.append(exported['exports'][0]['model'])
    agent._write_evidence()
    agent._prepare_program_state()
    return agent, lab, owner, identifier, list(dict.fromkeys(models))


def compare(agent, lab, owner, models):
    return agent._execute('run', {'operator': 'research_choice', 'models': models,
        'research_question': {'id': owner['id']}}, lab)


def test_current_consumer_records_the_scored_counterexample_for_its_historical_export_owner(tmp_path):
    agent, lab, owner, identifier, models = comparison_scene(tmp_path)
    consumer = {key: deepcopy(owner[key]) for key in
        ('question', 'parent_id', 'depends_on', 'kind', 'context_ref', 'answer_type',
         'answer_resolution_ref', 'query_semantics', 'use_sites', 'theory_target') if key in owner}
    consumer['id'] = 'current_parent_consumer'
    agent._open_question({'id': owner['id'], 'subject_refs': [{'kind': 'theory', 'id': 'changed_parent'}]})
    assert owner['meaning_history'][0]['revision_candidate_exports']
    result = agent._execute('run', {'operator': 'research_choice', 'models': models,
                                  'research_question': consumer}, lab)
    failures = result['revision_comparison_failures']
    assert failures['new_failure_record_count'] == failures['new_refutation_count'] == 1
    assert not failures['unknown']
    assert failures['owner_results'][0]['question_ref'] == {
        'question_id': owner['id'], 'meaning_version': 1}
    assert not agent.theory_index[identifier]['compatible']
    assert owner['meaning_version'] == 2


def test_older_fit_is_compared_on_new_rows_without_borrowing_its_old_mse_criterion(tmp_path):
    agent, lab, owner, identifier, models = comparison_scene(tmp_path)
    original = deepcopy(agent.theory_consolidation['transfers'])
    fit_rows = len(agent.rows)
    agent._execute('query', {'actions': [action(-1., 1.)]}, lab)
    result = compare(agent, lab, owner, models)
    assert result['paid_rows'] > fit_rows
    assert [item['path'] for item in result['evaluations']] == models
    actual = next(item for item in result['evaluations'] if item['path'] == models[-1])
    assert actual['paid_residuals']['evaluated_rows'] == len(agent.rows)
    failures = result['revision_comparison_failures']
    assert failures['new_failure_record_count'] == 0
    assert [item['reason'] for item in failures['unknown']] == [
        'original_complete_paid_mse_criterion_and_rows_required']
    assert agent.theory_consolidation['transfers'] == original
    assert agent.theory_index[identifier]['support']['status'] != 'refuted'


def test_registered_ordinary_score_refutes_unknown_revision_and_its_exact_scoped_alias(tmp_path, monkeypatch):
    from hegel_machine.cognition import online_inquiry_llm

    agent, lab, owner, identifier, models = comparison_scene(tmp_path)
    original = deepcopy(agent.theory_consolidation['transfers'][0])
    before = deepcopy(agent.costs)
    calls = []
    evaluate = online_inquiry_llm.evaluate_python_model

    def counted(*args, **kwargs):
        calls.append(args[1])
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(online_inquiry_llm, 'evaluate_python_model', counted)
    result = compare(agent, lab, owner, models)
    assert [item['path'] for item in result['evaluations']] == models
    assert [item['reason'] for item in result['revision_comparison_failures']['unknown']] == []
    assert result['revision_comparison_failures']['new_refutation_count'] == 1, result['revision_comparison_failures']
    assert agent.theory_consolidation['transfers'][0] == original
    retained = agent.theory_consolidation['transfers'][-1]
    assert retained['status'] == 'refuted' and retained['paid_mse_check']['status'] == 'failed'
    assert 'checks' not in retained and 'check_basis' not in retained
    checks, basis = load_source_values([retained['checks_ref'], retained['check_basis_ref']], agent.workspace)
    assert checks['value'] == original['checks'] and basis['value'] == original['check_basis']
    assert all(check['status'] == 'UNKNOWN' for check in checks['value'])
    assert retained['paid_mse_check']['mse'] > agent.config.confirmation_mse_threshold
    assert retained['retained_counterexample_refs']
    assert 'observation' not in retained['ordinary_comparison_failure']
    # The exported constant is the third predictor under a new filename.
    # Its literal reuse must avoid one of the four actual model executions.
    assert len(models) == 4 and calls == models[:3]
    assert result['new_scored_count'] == 3 and result['reused_score_count'] == 1
    assert agent.costs['compute_cost'] - before['compute_cost'] == 3 * len(agent.rows)
    assert agent.costs['observation_cost'] == before['observation_cost'] and agent.costs['model_calls'] == 0
    actual = agent.theory_index[identifier]
    assert actual['version'] == 1 and not actual['compatible'] and actual['support']['status'] == 'refuted'
    renamed = {key: deepcopy(actual[key]) for key in
               ('program', 'role_signature', 'output_unit', 'context_ref')}
    register_theories(agent, [{'id': 'ordinary_alias', **renamed},
        {'id': 'other_context', **renamed, 'context_ref': {'scene': 'different'}}])
    resolved = {item['id']: item for item in resolved_theories(agent)}
    assert resolved['ordinary_alias']['scope_refutation']['status'] == 'refuted'
    assert not resolved['ordinary_alias']['compatible']
    assert not resolved['other_context'].get('scope_refutation')
    # Re-consuming the same saved execution adds neither predictions nor facts.
    costs, records = deepcopy(agent.costs), deepcopy(agent.theory_consolidation['transfers'])
    repeated = common_archive_inquiry.record_revision_comparison_failures(
        agent, owner, result, source_ref={'file': 'research_state.json',
            'json_path': ['operator_results', len(agent.operator_results) - 1]})
    assert repeated['new_refutation_count'] == 0 and agent.costs == costs
    assert agent.theory_consolidation['transfers'] == records
    # A later ordinary transfer still sees both its original point-check
    # record and the new thin aggregate event, without assuming both contain
    # inline check arrays.
    retry = _checked_transfer(agent, {'theory_id': identifier,
        'role_signature': actual['role_signature'], 'output_unit': actual['output_unit'],
        'context_ref': actual['context_ref'], 'evidence_checks': original['check_basis']},
        {'id': 'fixture'}, Program.from_dict(actual['program']), original['binding'], compute_allowance=0)
    assert retry['status'] == 'refuted' and retry['new_compute_cost'] == 0


def saved_unconsumed_comparison(tmp_path, monkeypatch):
    agent, lab, owner, identifier, models = comparison_scene(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(common_archive_inquiry, 'record_revision_comparison_failures',
                      lambda *args, **kwargs: {'new_refutation_count': 0})
        result = compare(agent, lab, owner, models)
    reference = {'file': 'research_state.json',
                 'json_path': ['operator_results', len(agent.operator_results) - 1]}
    return agent, lab, owner, identifier, result, reference


def test_saved_failure_does_not_refute_a_new_version_of_the_exported_theory(tmp_path, monkeypatch):
    agent, lab, owner, identifier, result, reference = saved_unconsumed_comparison(tmp_path, monkeypatch)
    original = deepcopy(agent.theory_index[identifier])
    register_theories(agent, [{**original, 'program': Program('const', value=1.).to_dict()}])
    costs, transfers = deepcopy(agent.costs), deepcopy(agent.theory_consolidation['transfers'])
    outcome = common_archive_inquiry.record_revision_comparison_failures(
        agent, owner, result, source_ref=reference)
    assert outcome['new_refutation_count'] == 0 and outcome['unknown']
    assert agent.theory_index[identifier]['compatible']
    assert agent.costs == costs and agent.theory_consolidation['transfers'] == transfers


def test_saved_failure_does_not_borrow_a_different_paid_scope(tmp_path, monkeypatch):
    agent, lab, owner, identifier, result, reference = saved_unconsumed_comparison(tmp_path, monkeypatch)
    agent._execute('query', {'actions': [action(2.)]}, lab)
    costs, transfers = deepcopy(agent.costs), deepcopy(agent.theory_consolidation['transfers'])
    outcome = common_archive_inquiry.record_revision_comparison_failures(
        agent, owner, result, source_ref=reference)
    assert outcome['new_refutation_count'] == 0 and outcome['unknown']
    assert agent.theory_index[identifier]['compatible']
    assert agent.costs == costs and agent.theory_consolidation['transfers'] == transfers


def test_restored_raw_index_uses_the_original_physical_transfer_and_assessment(tmp_path, monkeypatch):
    agent, lab, owner, identifier, result, reference = saved_unconsumed_comparison(tmp_path, monkeypatch)
    agent._write_evidence()
    agent._prepare_program_state()
    previous = agent.result('researching')
    restored = RegisteredOrganizationAgent(None, tmp_path / 'restored', arm='fixed_pr2',
        research_organization=deepcopy(agent._organization_registration), source_task=False,
        knowledge={'files': {}}, config=agent.config)

    class RestorableLab(ChoiceLab):
        costs = PaidConditionalLab.costs
        restore_paid_account = PaidConditionalLab.restore_paid_account

    restored.restore_research(previous, agent.workspace, RestorableLab(), confirmation_seed=0)
    restored.theory_consolidation = install_theory_library(restored,
        {'file': str(agent.workspace / 'shared/theory_library.json'), 'json_path': []})
    current = next(q for q in restored.questions if q['id'] == owner['id'])
    raw = restored.theory_index[identifier]['consolidation_ref']
    assert raw['context_ref'] == current['theory_target']['context_ref']
    assert restored.theory_consolidation['transfers'][0]['check_basis'][0]['source_ref']['file'] == str(
        agent.workspace / 'laboratory_observations.json')
    reference['file'] = str(agent.workspace / 'research_state.json')
    costs = deepcopy(restored.costs)

    def no_prediction(*args, **kwargs):
        raise AssertionError('Saved negative evidence must not execute a Program')

    monkeypatch.setattr(Program, 'predict', no_prediction)
    outcome = common_archive_inquiry.record_revision_comparison_failures(
        restored, current, result, source_ref=reference)
    assert outcome['new_refutation_count'] == 1, outcome['unknown']
    assert not restored.theory_index[identifier]['compatible'] and restored.costs == costs
    assert restored.theory_index[identifier]['consolidation_ref']['fitting']['evidence_refs'][0]['file'] == str(
        agent.workspace / 'laboratory_observations.json')
