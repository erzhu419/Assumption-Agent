from copy import deepcopy
from pathlib import Path
import json
import pytest
from hegel_machine.cognition import research_episodes
from hegel_machine.cognition.research_episodes import project_method_use_feedback
from test_common_method_returns import feedback
from test_research_program_consolidation import compile_sources
from test_theory_consolidation import agent_scene
from hegel_machine.cognition.theory_consolidation import install_theory_library


def native_source(path, method):
    path.mkdir(parents=True)
    record = feedback()
    profile = record.pop('profile')
    record.pop('feedback_ref')
    record['method_ref'] = deepcopy(method)
    record['invoked_method_ref'] = {**method, 'version': method['version'] + 3}
    profile_path = path / 'profile.json'
    profile_path.write_text(json.dumps(profile))
    record['profile_ref'] = {'file': str(profile_path), 'json_path': []}
    record['evidence_basis'] = {'mapping': {'role_0': 'u'},
        'context_ref': {'source_id': 'one_physical_task'}}
    source_path = path / 'use.json'
    source_path.write_text(json.dumps(record))
    return {'file': str(source_path), 'json_path': []}, record, profile


@pytest.mark.parametrize('incomplete', [False, True])
def test_native_return_preserves_body_invocation_cost_and_unscored_state(tmp_path, incomplete):
    reference, original, profile = native_source(tmp_path / 'source', {'id':'method', 'version':2})
    if incomplete:
        original.update(status='execution_error', execution_complete=False,
            execution_error={'type':'KeyError','message':'checks'})
        Path(reference['file']).write_text(json.dumps(original))
    actual = project_method_use_feedback(reference, tmp_path / 'target')
    assert actual['profile'] == profile
    assert actual['origin_ref'] == actual['source_ref'] == reference
    assert {k:v for k,v in actual.items() if k not in {'profile','origin_ref','source_ref'}} == original
    assert json.loads(Path(reference['file']).read_text()) == original
    assert 'paired_future' not in actual
    assert actual['method_ref']['version'] == 2
    assert actual['invoked_method_ref']['version'] == 5


def test_common_import_freezes_native_return_once_without_scientific_work(tmp_path, monkeypatch):
    learner, compiled = compile_sources(tmp_path / 'library')
    agent, lab = agent_scene(tmp_path / 'target')
    agent._save('shared/theory_library.json', (learner.workspace / compiled['library_path']).read_text())
    agent.theory_consolidation = install_theory_library(agent)
    method, = agent.theory_consolidation['candidates'].values()
    reference, original, _ = native_source(tmp_path / 'feedback',
        {'id':method['id'], 'version':method['version']})
    def forbidden(*args, **kwargs):
        raise AssertionError('Feedback ingestion must not fit, predict, or query the world')
    monkeypatch.setattr(lab, 'execute', forbidden)
    import hegel_machine.cognition.research_programs as programs
    monkeypatch.setattr(programs, 'replay_research_program', forbidden)
    args = {'operator':'theory_consolidation','object_type':'method_use_condition',
        'candidate_ids':[method['id']], 'method_feedback_source_refs':[reference],
        'mission':{'purpose':'method_learning','theory_learning_allocation':{'compute':10}},
        'allocation':{'compute':10}}
    before = deepcopy(agent.costs)
    result = agent._execute('run', args, lab)
    assert result['new_compute_cost'] == 2
    assert agent.costs['compute_cost'] == before['compute_cost'] + 2
    current = agent.theory_consolidation['candidates'][method['id']]
    assert current['version'] == method['version'] + 1
    group = agent.theory_consolidation['method_feedback'][method['id']]
    assert len(group['outcome_refs']) == len(group['returns']['origins']) == 1
    assert group['returns']['origins'][0]['feedback_ref'] == reference
    assert group['returns']['origins'][0]['status'] == 'unscored'
    paid = deepcopy(agent.costs)
    repeated = agent._execute('run', args, lab)
    assert repeated['new_compute_cost'] == 0
    assert agent.costs == paid
    assert len(group['outcome_refs']) == len(group['returns']['origins']) == 1
    assert json.loads(Path(reference['file']).read_text()) == original
