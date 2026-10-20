"""Provider-declared work follows ordinary author edits into the paid practice.

All authored models and declarations below are synthetic fixtures, not live inputs.
"""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from test_cognition_code_provider_json import returned_action
from test_paid_scene_method_repair import make_repair, REFIT_METHOD


def decoded_action(author, monkeypatch, action, arguments, source, question):
    # Exercise the real top-level response mapping without any external call.
    with monkeypatch.context() as patched:
        returned_action(patched, {'action': action,
            'arguments_json': json.dumps(arguments), 'source_code': source,
            'research_question': question})
        return CodexCodeProvider(author.workspace / 'provider', executable='fixture').next_action(
            {'research_root': author.research_root})


@pytest.mark.parametrize('action', ['write', 'run'])
def test_declared_child_replaces_returned_reassessment_in_actual_edit_practice(
        tmp_path, monkeypatch, action):
    author, checkpoint, previous = make_repair(tmp_path)
    original = checkpoint.read_bytes()
    original_source = (checkpoint.parent / METHOD_PATH).read_text()
    root = deepcopy(author.research_root['assignment'])
    author._execute('run', {'operator': 'procedure', 'path': METHOD_PATH,
        'question': {'id': 'old_review', 'question': 'Recheck the old conclusion.'}}, author.interface)
    author._execute('finish', {'inconclusive': True, 'lesson': 'The old program still stops.'},
                    author.interface)
    assert author.research_work['question']['id'] == 'old_review'
    declared = {'id': 'revised_fit', 'question': 'Refit the fixture and test its held-out predictions.',
                'parent_id': 'task', 'depends_on': []}
    arguments = {'path': METHOD_PATH} if action == 'write' else {}
    source = (REFIT_METHOD if action == 'write' else
              f'from pathlib import Path\nPath({METHOD_PATH!r}).write_text({REFIT_METHOD!r})')
    decoded = decoded_action(author, monkeypatch, action, arguments, source, declared)
    assert decoded['arguments']['research_question'] == declared
    result = author._execute(decoded['action'], decoded['arguments'], author.interface)
    practice = result['edit_practice']
    assert practice['status'] == 'completed'
    assert practice['fresh_confirmation']['passed'] is True
    trial = author.workspace / practice['directory']
    saved = json.loads((trial / 'result.json').read_text())
    work = saved['research_work']
    assert work == author.research_work == json.loads((trial / 'research_work.json').read_text())
    assert work['question']['id'] == 'revised_fit'
    assert work['question']['parent_id'] == root['id']
    assert work['source_workspace'] == str(author.workspace)
    assert work['question']['assignment'] == {**declared, 'parent_id': root['id']}
    inner = next(q for q in saved['online_questions'] if q['id'] == 'revised_fit')
    assert inner['assignment'] == work['question']['assignment']
    assert saved['paid_observations'][:len(previous['paid_observations'])] == previous['paid_observations']
    assert practice['continuation_cost']['observation_cost'] == author.costs['observation_cost'] == 12
    assert author.costs['model_calls'] == 0
    assert checkpoint.read_bytes() == original
    assert (checkpoint.parent / METHOD_PATH).read_text() == original_source
    assert author.research_root['assignment'] == root
    assert author._active_work_id == 'revised_fit'
    assert author._current_source_coverage()['confirmed_cases'] == ['same_scene']

    # Keeping the same method text does not lend this confirmation to another work.
    author.config = replace(author.config, observation_budget=author.costs['observation_cost'])
    author._execute('run', {'code': 'print("another local inquiry")',
        'research_question': {'id': 'another_work', 'question': 'Consider another evidence scope.'}},
        author.interface)
    assert author._current_source_coverage()['confirmed_cases'] == []
    assert 'error' in author._execute('finish', {'path': METHOD_PATH}, author.interface)


def test_same_child_keeps_assignment_and_routine_declarations_do_not_replace_root(tmp_path, monkeypatch):
    author, checkpoint, _ = make_repair(tmp_path)
    before = checkpoint.read_bytes()
    root = deepcopy(author.research_root)
    for question in (None, {'id': root['id'], 'question': 'Inspect a small part of the parent.',
                            'parent_id': None, 'depends_on': []}):
        proposal = decoded_action(author, monkeypatch, 'run', {}, 'print("inspection")', question)
        author._execute(proposal['action'], proposal['arguments'], author.interface)
        assert author.research_work is None and author._active_work_id is None
        assert author.questions == [root]

    child = {'id': 'local', 'question': 'Assess the existing empirical discrepancy.', 'depends_on': []}
    author._execute('run', {'code': 'print("evidence read")', 'research_question': child}, author.interface)
    assignment = deepcopy(author.research_work['question']['assignment'])
    revised = {**child, 'question': 'Revise the local hypothesis after the evidence read.'}
    author._execute('run', {'code': 'print("revised interpretation")',
                           'research_question': revised}, author.interface)
    assert author.research_work['question']['assignment'] == assignment
    assert author.research_work['question']['question'] == revised['question']
    assert author.research_root == root
    assert not author.practice_trials and author.costs['observation_cost'] == 0
    assert checkpoint.read_bytes() == before


def test_explicit_procedure_uses_provider_declaration_before_legacy_question(tmp_path, monkeypatch):
    author, _, _ = make_repair(tmp_path)
    declared = {'id': 'declared', 'question': 'Inspect this fixture program.',
                'parent_id': 'task', 'depends_on': []}
    proposal = decoded_action(author, monkeypatch, 'run', {
        'operator': 'procedure', 'path': METHOD_PATH,
        'question': {'id': 'legacy', 'question': 'An older question.'}}, None, declared)
    result = author._execute(proposal['action'], proposal['arguments'], author.interface)
    assert result['status'] == 'method_stopped'
    assert author.research_work['question']['id'] == 'declared'
    assert not any(q['id'] == 'legacy' for q in author.questions)


def test_revising_or_reselecting_work_keeps_completed_program_result_without_lending_support(tmp_path):
    author, checkpoint, _ = make_repair(tmp_path / 'first')
    original = checkpoint.read_bytes()
    work = {'id': 'fitting', 'question': 'Fit and compare the fixture candidate.'}
    author._execute('write', {'path': METHOD_PATH, 'code': REFIT_METHOD,
        'research_question': work}, author.interface)
    completed = deepcopy(author.research_work_result)
    assert completed['confirmation']['passed'] is True
    assert [op['operator'] for op in completed['operations'][:2]] == ['refit', None]

    # No further computation is permitted. This is a revision of the task and
    # source, not permission to reuse the old draft's successful confirmation.
    author.config = replace(author.config, observation_budget=author.costs['observation_cost'])
    revised = {**work, 'question': 'Reconsider the fit using its completed comparison.'}
    source = REFIT_METHOD + '\n# An author revision awaiting its own execution.\n'
    author._execute('write', {'path': METHOD_PATH, 'code': source,
        'research_question': revised}, author.interface)
    request = author._request(author.interface)
    result = request['research_work_result']
    assert result['directory'] == completed['directory']
    assert result['operations'] == completed['operations']
    assert result['question'] == completed['question']
    assert result['source_matches_current'] is False
    assert result['question_matches_current'] is False
    assert request['research_root']['phase'] == 'choose_or_revise_program'
    assert author._current_source_coverage()['confirmed_cases'] == []
    assert len(author.practice_trials) == 1

    # Returning to the same child retrieves its real completed work instead of
    # clearing it or treating another child's stages as its own.
    author._execute('run', {'code': 'pass', 'research_question': {
        'id': 'other', 'question': 'Inspect another question.'}}, author.interface)
    assert author.research_work_result is None
    author._execute('run', {'code': 'pass', 'research_question': revised}, author.interface)
    assert author.research_work_result == completed
    previous = author.result('model_call_budget_exhausted')
    resumed, _, _ = make_repair(tmp_path / 'resumed')
    resumed.restore_practice(previous)
    resumed_request = resumed._request(resumed.interface)
    assert resumed_request['research_work_result']['operations'] == completed['operations']
    assert resumed_request['research_work_result']['source_matches_current'] is False
    assert resumed_request['research_work_result']['question_matches_current'] is False
    assert resumed._current_source_coverage()['confirmed_cases'] == []
    assert (resumed.workspace / completed['directory']).is_symlink()
    assert resumed.costs['model_calls'] == resumed.costs['observation_cost'] == 0
    assert checkpoint.read_bytes() == original
