"""Current citations keep executed evidence visible beyond the recent-action window."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.code_provider_context import prepare_code_context
from test_common_archive_online_research import prepared
from test_common_archive_resume import SEED, execute
from test_executed_evidence_notebook import AccountedPublicLab, evidence_scene


def add_analyses(agent, count=12):
    start = len(agent.events)
    for offset in range(count):
        index = start + offset
        agent.events.append({'action': 'run', 'arguments': {'path': f'analysis_{index}.py'},
            'feedback': {'returncode': 0, 'stdout': f'executed result {index}', 'stderr': '',
                         'executed_path': f'analysis_{index}.py'}})
    return list(range(start, start + count))


def analysis_facts(notebook):
    return {key: value for key, value in notebook['facts'].items() if key.startswith('analysis:')}


@pytest.mark.parametrize('arm', ['code_react', 'hegel'])
def test_cited_result_survives_fact_and_action_windows_without_reviving_all_history(tmp_path, arm):
    agent, lab = evidence_scene(tmp_path, arm)
    indices = add_analyses(agent)
    cited, uncited = indices[:2]
    note = f'analysis:{cited} suggested a candidate; the meaning remains uncertain.'
    (agent.workspace / 'research_notes.md').write_text(note)
    before = deepcopy(agent.events)

    request = agent._request(lab)
    context = prepare_code_context(agent.workspace, request, schema={}, instructions='test')
    facts = analysis_facts(context['evidence_notebook'])
    assert set(facts) == {f'analysis:{i}' for i in [cited, *indices[-2:]]}
    assert facts[f'analysis:{cited}']['stdout'] == f'executed result {cited}'
    assert facts[f'analysis:{cited}']['source']['json_path'] == ['events', cited, 'feedback']
    assert facts[f'analysis:{cited}']['executed_path'] == f'analysis_{cited}.py'
    assert f'analysis:{uncited}' not in facts
    assert context['event_history']['preview_start'] > cited
    assert context['research_notes'] == note
    assert 'answer' not in facts[f'analysis:{cited}']
    assert agent.events == before


@pytest.mark.parametrize('separator', ['-', '–'])
def test_ranges_select_only_real_analyses_and_preserve_errors_and_interruption(tmp_path, separator):
    agent, lab = evidence_scene(tmp_path, 'hegel')
    indices = add_analyses(agent)
    failed, interrupted, nonanalysis = indices[:3]
    agent.events[failed]['feedback'].update(returncode=1, stdout='', stderr='ValueError: unfinished')
    agent.events[interrupted]['feedback'] = {
        'status': 'runtime_interrupted', 'stdout': '', 'stderr': '',
        'executed_path': 'slow.py', 'execution_seconds': 300.}
    agent.events[nonanalysis]['arguments']['operator'] = 'procedure'
    note = (f'analysis:{failed}{separator}{nonanalysis} needs review; '
            f'analysis:{len(agent.events) + 5}{separator}999999999 is not an executed record.')
    (agent.workspace / 'research_notes.md').write_text(note)

    facts = analysis_facts(agent._request(lab)['evidence_notebook'])

    assert set(facts) == {f'analysis:{i}' for i in [failed, interrupted, *indices[-2:]]}
    assert facts[f'analysis:{failed}']['returncode'] == 1
    assert facts[f'analysis:{failed}']['stderr'] == 'ValueError: unfinished'
    assert facts[f'analysis:{interrupted}']['status'] == 'runtime_interrupted'
    assert 'returncode' not in facts[f'analysis:{interrupted}']
    assert 'mse' not in facts[f'analysis:{interrupted}']


def test_current_citations_restore_with_facts_without_repeating_paid_work(tmp_path):
    old, lab = evidence_scene(tmp_path / 'source', 'hegel')
    indices = add_analyses(old)
    cited = indices[0]
    note = f'Carry analysis:{cited} into the next decision; no confirmation was performed.'
    execute(old, lab, 'write', {'path': 'research_notes.md', 'code': note})
    notebook = deepcopy(old._request(lab)['evidence_notebook'])
    previous = old.result('model_call_budget_exhausted')
    resumed, target = prepared(tmp_path / 'restored', 'hegel'), AccountedPublicLab()

    restored = resumed.restore_research(previous, old.workspace, target, confirmation_seed=SEED)

    assert restored['new_observations'] == 0
    assert target.actions == []
    assert resumed._request(target)['evidence_notebook'] == notebook
    assert resumed.costs == old.costs
    assert resumed.events == old.events
    assert (resumed.workspace / 'research_notes.md').read_text() == note

    # An old citation does not remain selected after the current author drops it.
    execute(resumed, target, 'write', {'path': 'research_notes.md', 'code': 'Review the latest results.'})
    facts = analysis_facts(resumed._request(target)['evidence_notebook'])
    assert set(facts) == {f'analysis:{i}' for i in indices[-2:]}
