"""Registered hooks preserve and export the selected original research meaning."""
from copy import deepcopy

from hegel_machine.cognition import finite_native_author
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from test_registered_revision_target import revision_scene


def test_direct_parent_obligation_is_not_rebound_and_exports_selected_history(tmp_path, monkeypatch):
    agent, lab, child = revision_scene(tmp_path)
    question = {
        'id': 'parent_revision:task:' + child['id'],
        'question': 'Continue revision of this parent for the original task use.',
        'kind': 'program_revision', 'parent_id': 'task', 'depends_on': [],
        'origin': {'kind': 'unfinished_parent_revision'},
        'subject_refs': [{'kind': 'theory', 'id': 'original_before', 'version': 1}],
        'theory_target': {'parent_theory_id': 'original_before'},
    }
    owner = agent._open_question(question)
    agent._open_question({**question,
        'subject_refs': [{'kind': 'theory', 'id': 'new_before', 'version': 1}],
        'theory_target': {'parent_theory_id': 'new_before'}})
    historical = owner['meaning_history'][0]
    original = deepcopy((agent.evaluated_model, agent.costs,
                         owner['theory_target'], historical['theory_target']))
    agent._active_question_id = owner['id']
    monkeypatch.setattr(CommonArchiveInquiryAgent, '_before_model_turn', lambda *_: False)
    monkeypatch.setattr(agent, '_consolidate_registered_source', lambda *_: False)
    monkeypatch.setattr(agent, '_bind_revision_target',
                        lambda *_: (_ for _ in ()).throw(AssertionError('Original parent must stay fixed')))
    monkeypatch.setattr(finite_native_author, 'ordinary_completion_reserve', lambda *_: None)

    def route(arguments, laboratory, kind):
        assert arguments['research_question'] == {'id': owner['id']}
        agent._organization_state['routing'] = {'event_index': len(agent.events)}
        agent.events.append({'feedback': {'research_track_question_ref': {
            'question_id': owner['id'], 'meaning_version': 1}}})
        return True

    exported = []

    def export(current, selected):
        assert current is agent
        exported.append(selected)
        return {'changed': False, 'new_exports': 0}

    monkeypatch.setattr(agent, '_automatic_work', route)
    monkeypatch.setattr(finite_native_author, 'export_revision_candidates', export)
    assert agent._before_model_turn(lab)
    assert len(exported) == 1 and exported[0] is historical
    assert (agent.evaluated_model, agent.costs,
            owner['theory_target'], historical['theory_target']) == original
