"""Relation references retain the exact executed round without copying banks."""
from copy import deepcopy
from types import SimpleNamespace

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.question_routing import register_theories, resolved_theories


X = Program('feature', name='u')
SQUARE = Program('mul', children=(X, X))


def bank(program=X, *, round_index=0, status='plausible'):
    return {'round_index': round_index, 'candidates': [
        {'program': program.to_dict(), 'status': status}], 'bound_rows': ['retained_once']}


def feedback(entry, *, meaning=1):
    return {'operator': 'research_update', 'research_question_id': 'source',
            'question_meaning_version': meaning, 'relation_research': deepcopy(entry)}


def scene(entry=None, events=None):
    return SimpleNamespace(theory_index={}, questions=[{'id': 'source', 'meaning_version': 1}],
        relation_research={'source': entry or bank()}, events=events or [])


def declaration(identifier='retained', *, round_index=0, meaning=None):
    source_ref = {'kind': 'relation_candidate', 'question_id': 'source',
                  'round_index': round_index, 'candidate_index': 0}
    if meaning is not None:
        source_ref['meaning_version'] = meaning
    return {'id': identifier, 'source_ref': source_ref}


def test_repeated_coordinates_keep_the_original_event_and_meaning_across_new_round_zero():
    old = bank()
    agent = scene(old, [{'feedback': feedback(old)}])
    requested = declaration()
    register_theories(agent, [requested])
    retained = deepcopy(agent.theory_index['retained'])
    assert retained['source_ref']['event_index'] == 0
    agent.questions[0]['meaning_version'] = 2
    agent.relation_research['source'] = bank(SQUARE)
    agent.events.append({'feedback': feedback(agent.relation_research['source'], meaning=2)})
    register_theories(agent, [requested])
    assert agent.theory_index['retained'] == retained
    assert resolved_theories(agent)[0]['program'] == X.to_dict()
    register_theories(agent, [declaration('new_meaning', meaning=2)])
    new = next(item for item in resolved_theories(agent) if item['id'] == 'new_meaning')
    assert new['program'] == SQUARE.to_dict() and new['source_ref']['event_index'] == 1
    assert 'program' not in retained and 'bound_rows' not in retained
    assert 'meaning_version' not in requested['source_ref']


def test_exact_offline_current_bank_remains_available_without_loading_events():
    agent = scene(bank(status='refuted'))
    register_theories(agent, [declaration()])
    assert 'event_index' not in agent.theory_index['retained']['source_ref']
    result = resolved_theories(agent)[0]
    assert result['program'] == X.to_dict() and result['source_status'] == 'refuted'
    assert result['compatible'] is False
    agent.relation_research['source'] = bank(SQUARE, round_index=1)
    unavailable = resolved_theories(agent)[0]
    assert unavailable['source_status'] == 'unavailable_retained_version'
    assert unavailable['compatible'] is False and 'program' not in unavailable


def test_later_registration_can_pin_the_previously_offline_bank_once_event_is_available():
    original = bank()
    agent = scene(original)
    requested = declaration()
    register_theories(agent, [requested])
    agent.events.append({'feedback': feedback(original)})
    agent.relation_research['source'] = bank(SQUARE, round_index=1)
    register_theories(agent, [requested])
    assert agent.theory_index['retained']['source_ref']['event_index'] == 0
    assert resolved_theories(agent)[0]['program'] == X.to_dict()


def test_pinned_event_never_reads_another_questions_or_meanings_round_zero():
    agent = scene(bank(SQUARE, round_index=1), [{'feedback': feedback(bank(), meaning=2)}])
    wrong = declaration(meaning=1)
    wrong['source_ref']['event_index'] = 0
    register_theories(agent, [wrong])
    assert resolved_theories(agent)[0]['source_status'] == 'unavailable_retained_version'
    agent.events[0]['feedback'] = feedback(bank(), meaning=1)
    agent.events[0]['feedback']['research_question_id'] = 'other_question'
    assert resolved_theories(agent)[0]['source_status'] == 'unavailable_retained_version'


def test_unmarked_historical_events_only_bind_the_original_meaning():
    original = feedback(bank())
    original.pop('question_meaning_version')
    agent = scene(events=[{'feedback': original}])
    register_theories(agent, [declaration()])
    assert agent.theory_index['retained']['source_ref']['event_index'] == 0
    agent.questions[0]['meaning_version'] = 2
    register_theories(agent, [declaration('new', meaning=2)])
    assert 'event_index' not in agent.theory_index['new']['source_ref']


def test_actual_direct_and_after_observation_relation_events_survive_bank_replacement(tmp_path):
    from test_structural_inquiry import prepared, update, ROOT, RELATION
    from test_common_archive_llm_inquiry import RestorableConfirmationLab
    from test_scientific_generalization import execute
    from test_inquiry_revision import action

    agent, lab = prepared(tmp_path), RestorableConfirmationLab()
    execute(agent, lab, 'query', {'question': ROOT, 'actions': [action(0., 1.)]})
    first = update(agent, lab)
    source_bank = agent.relation_research[RELATION['id']]
    candidate_index = next(index for index, candidate in enumerate(source_bank['candidates'])
                           if candidate['program'] == SQUARE.to_dict())
    requested = {'id': 'old_square', 'source_ref': {'kind': 'relation_candidate',
        'question_id': RELATION['id'], 'round_index': 0, 'candidate_index': candidate_index}}
    register_theories(agent, [requested])
    old_index = deepcopy(agent.theory_index['old_square'])
    assert old_index['source_ref']['event_index'] == len(agent.events) - 1
    selected = first['next_action']
    execute(agent, lab, selected['action'], selected['arguments'])
    assert agent.relation_research[RELATION['id']]['round_index'] == 1
    register_theories(agent, [requested])
    assert agent.theory_index['old_square'] == old_index
    assert resolved_theories(agent)[0]['program'] == SQUARE.to_dict()
    current = deepcopy(requested)
    current['id'] = 'screened_square'
    current['source_ref']['round_index'] = 1
    register_theories(agent, [current])
    source_ref = agent.theory_index['screened_square']['source_ref']
    assert source_ref['event_index'] == len(agent.events) - 1
    assert source_ref['after_observation_index'] == 0
    screened = next(item for item in resolved_theories(agent) if item['id'] == 'screened_square')
    assert screened['source_status'] == 'refuted' and screened['compatible'] is False
    assert 'program' not in agent.theory_index['screened_square']
    assert agent.costs['model_calls'] == 0 and not agent.confirmations
