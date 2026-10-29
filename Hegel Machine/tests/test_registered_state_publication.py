"""Ordinary consumers publish once after all public before-turn work."""

import json

from hegel_machine.cognition import common_archive_inquiry, finite_native_author
from scripts import run_anonymous_revision_inquiry as study
from test_finite_choice_reuse import choice_scene


def test_before_hook_leaves_publication_to_the_actual_consumer(tmp_path, monkeypatch):
    agent, lab = choice_scene(tmp_path)
    monkeypatch.setattr(common_archive_inquiry.CommonArchiveInquiryAgent,
                        '_before_model_turn', lambda *_: False)
    monkeypatch.setattr(agent, '_bind_revision_target', lambda *_: None)
    monkeypatch.setattr(agent, '_consolidate_registered_source', lambda *_: False)
    monkeypatch.setattr(finite_native_author, 'ordinary_completion_reserve',
        lambda *_: {'compute_cost': 7, 'paid_rows': len(agent.rows)})
    monkeypatch.setattr(finite_native_author, 'export_revision_candidates',
        lambda *_: {'changed': True, 'new_exports': 1})
    monkeypatch.setattr(agent, '_automatic_work', lambda *_: False)
    original = agent._prepare_program_state
    publications = []

    def publish():
        publications.append(len(agent.operator_results))
        return original()

    monkeypatch.setattr(agent, '_prepare_program_state', publish)
    assert agent._before_model_turn(lab)
    assert publications == []
    request = agent._request(lab)
    assert len(publications) == 1
    state = json.loads((agent.workspace / 'research_state.json').read_text())
    current = next(item for item in state['questions'] if item['id'] == 'task')
    assert current['ordinary_completion_reserve']['compute_cost'] == 7
    assert state['limits']['compute_budget'] == request['limits']['compute_budget'] == agent.config.compute_budget
