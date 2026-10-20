"""Changing author work retains selection evidence without borrowing confirmation."""

from copy import deepcopy
from dataclasses import replace
import json

from hegel_machine.cognition.code_provider_context import prepare_code_context
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from test_paid_scene_method_repair import make_repair, REFIT_METHOD


def test_changed_work_keeps_historical_program_outcomes_visible(tmp_path):
    author, _, _ = make_repair(tmp_path)
    author._practice('same_scene')
    selected = 'shared/chosen_program.py'
    (author.workspace / selected).write_text(REFIT_METHOD)
    first_work = {'id': 'research', 'question': 'Refit the existing candidate.', 'depends_on': []}
    result = author._execute('run', {'operator': 'procedure', 'path': selected,
        'question': first_work}, author.interface)
    assert result['status'] == 'completed'
    assert author._current_source_coverage()['confirmed_cases'] == ['same_scene']

    # A later assignment has no execution yet. Do not spend more observations
    # just to test the separation between its eligibility and paid history.
    author.config = replace(author.config, observation_budget=author.costs['observation_cost'])
    revised = deepcopy(first_work)
    revised['question'] = 'A different research assignment.'
    denied = author._execute('run', {'operator': 'procedure', 'path': selected,
        'question': revised}, author.interface)
    assert 'error' in denied
    request = author._request(author.interface)
    decision = json.loads(author.files['method_decision.json'])
    current = next(entry for entry in decision['candidates'] if entry['current'])
    stopped = next(entry for entry in decision['candidates'] if not entry['current'])
    assert current['supported_cases'] == ['same_scene']
    assert current['latest_case_evidence'][0]['directory'] == 'trials/trial_001'
    assert stopped['failed_cases'] == ['same_scene']
    assert stopped['latest_case_evidence'][0]['status'] == 'method_stopped'
    assert 'Historical executions' in current['evidence_scope']
    assert decision['current_source_coverage']['confirmed_cases'] == []
    assert decision['current_source_latest_failure'] is None
    assert 'error' in author._execute('finish', {'path': METHOD_PATH}, author.interface)
    assert author.model is None

    # The actual code-provider compaction must preserve these selection outcomes.
    context = prepare_code_context(tmp_path / 'provider', request, schema={}, instructions='')
    shown = context['method_decision']['candidates']
    assert shown == decision['candidates']
    assert context['method_decision']['current_source_coverage']['confirmed_cases'] == []
