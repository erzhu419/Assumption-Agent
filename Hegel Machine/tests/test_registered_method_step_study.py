"""The development driver uses Common and exports every completed paid slot."""

from copy import deepcopy
from pathlib import Path

import pytest

from hegel_machine.cognition.research_episodes import load_source_values
from hegel_machine.cognition.method_feedback import _read
from scripts import run_anonymous_revision_inquiry as study
from test_common_method_step_revision import step_question
from test_prepare_pr3_source_successor import save


def registration(tmp_path, monkeypatch):
    agent, _, method, _, _ = step_question(tmp_path / 'paid')
    reference, = agent.theory_consolidation['method_feedback'][method['id']]['outcome_refs']
    feedback = _read(agent, reference)
    feedback['invocation_binding']['library_path'] = str(agent.workspace / 'shared/theory_library.json')
    reference = save(tmp_path / 'original_paid_feedback.json', feedback)
    # Historical projection is a pure input adapter; this control already has
    # its canonical Common feedback and exercises the real downstream route.
    import hegel_machine.cognition.research_episodes as episodes
    monkeypatch.setattr(episodes, 'project_method_use_feedback',
        lambda ref, workspace: deepcopy(feedback))
    return {'stage': 'method_step_development', 'execution_authorized': True,
        'approval_ref': {'file': '/approved/control.json'}, 'external_python_seconds': 0,
        'max_candidates': 8, 'compute_budget': 5000, 'native_host_stop_seconds': 30,
        'feedback_ref': reference,
        'baseline_library_ref': {'file': str(agent.workspace / 'shared/theory_library.json')},
        'baseline_method_ref': {'id': method['id'], 'version': method['version']},
        'prior_costs_ref': {'file': str(agent.workspace / 'research_state.json'), 'json_path': ['costs']}}


def test_authorized_driver_uses_Common_without_provider_world_or_duplicate_baseline(tmp_path, monkeypatch):
    registered = registration(tmp_path, monkeypatch)
    import hegel_machine.cognition.research_programs as programs
    original = programs.replay_research_program
    def changed_only(payload, *args, **kwargs):
        assert payload.get('steps') != []
        return original(payload, *args, **kwargs)
    monkeypatch.setattr(programs, 'replay_research_program', changed_only)
    monkeypatch.setattr(study, 'AnonymousHistoryLab', lambda *a, **k: pytest.fail('constructed a new world'))
    monkeypatch.setattr(study, 'evaluate_python_model', lambda *a, **k: pytest.fail('rescored strong pool'))
    result = study.run_method_step_study(registered, tmp_path / 'development_study')
    assert result['status'] in {'method_steps_revised', 'method_steps_researched'}
    assert result['costs']['observation_cost'] == result['costs']['model_calls'] == 0
    assert result['costs']['compute_cost'] <= registered['compute_budget']
    assert result['external_python_execution_seconds'] == 0
    assert not result['automatic_successor_execution']
    saved = load_source_values([result['study_ref']], tmp_path)[0]['value']
    assert all(seed['methods'][0]['reused'] and seed['methods'][0]['new_compute_cost'] == 0
        for seed in saved['seed_studies'])
    complete = sum(bool(method.get('execution_complete') and method.get('program'))
        for seed in saved['seed_studies'] for method in seed['methods'])
    assert len(result['exported_candidates']) == complete
    assert not result['export_errors']
    assert all(Path(item['model']).is_file() for item in result['exported_candidates'])
    assert result['prior_costs_ref'] == registered['prior_costs_ref']


def test_unapproved_driver_does_not_construct_or_repeat_work(tmp_path, monkeypatch):
    registered = registration(tmp_path, monkeypatch)
    registered['execution_authorized'] = False
    monkeypatch.setattr(study, 'CommonArchiveInquiryAgent', lambda *a, **k: pytest.fail('started work'))
    with pytest.raises(ValueError, match='authorization'):
        study.run_method_step_study(registered, tmp_path / 'forbidden')
    assert not (tmp_path / 'forbidden').exists()
