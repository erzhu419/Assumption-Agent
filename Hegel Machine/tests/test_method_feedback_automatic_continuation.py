"""A late paid pair reopens method work through the existing Common author."""

from copy import deepcopy

from hegel_machine.cognition.finite_native_author import select_actions
from hegel_machine.cognition.method_feedback import _condition_outcomes, _read
from test_common_method_feedback import scene, use_and_compare
from test_common_method_future_returns import acquire_new_rows


def method_question(agent, method, parent):
    question, = [item for item in agent.questions if item.get('kind') == 'method_use'
        and item.get('parent_id') == parent['id']
        and item.get('method_use_plan', {}).get('candidate_id') == method['id']]
    return question


def analyze(agent, lab, question):
    return agent._execute('run', {'operator': 'research_route',
        'research_question': {'id': question['id']}, 'allocation': {'compute': 1000}}, lab)


def test_same_paid_rows_late_pair_reopens_existing_method_question_and_native_author(
        tmp_path, monkeypatch):
    agent, lab, method, parent = scene(tmp_path, useful=True)
    mission = {'mission_id': 'actual_method_followup', 'purpose': 'method_learning',
        'allocation': {'compute': 2000}}
    parent['mission'] = mission
    parent['context_ref'] = deepcopy(parent['theory_target']['context_ref'])
    _, initial_comparison = use_and_compare(agent, lab, parent)
    question = method_question(agent, method, parent)
    assert question['use_sites'] and question['mission']['allocation'] == {'compute': 2000}
    first = analyze(agent, lab, question)
    assert first['condition_learning']['frozen_candidates']
    assert agent.theory_consolidation['candidates'][method['id']]['version'] == method['version'] + 1
    feedback_ref, = agent.theory_consolidation['method_feedback'][method['id']]['outcome_refs']
    original = _read(agent, feedback_ref)
    assert original['working_comparison']['rows'] == 16 and 'paired_future' not in original

    measured = acquire_new_rows(agent, lab, parent)
    assert len(agent.rows) == 18
    before_pair = analyze(agent, measured, question)
    assert before_pair['status'] in {'method_use_decided', 'method_use_unresolved'}
    assert before_pair['condition_learning']['new_compute_cost'] == 0
    assert question['completed_agenda_basis'] == question['agenda_basis']
    basis = deepcopy(question['agenda_basis'])
    frozen = deepcopy(agent.theory_consolidation['candidates'][method['id']])
    assert frozen['version'] == method['version'] + 1
    assert 'paired_future' not in _condition_outcomes(agent,
        frozen['method_use_conditions']['outcomes_ref'])[0]

    models = ['old.py', *[item['model'] for item in parent['revision_candidate_exports']]]
    compared = agent._execute('run', {'operator': 'research_choice', 'models': models,
        'research_question': {'id': parent['id']}}, measured)
    assert compared['models_requested'] == compared['models_completed'] and not compared.get('truncated')
    updated = _read(agent, feedback_ref)
    assert updated['paired_future']['reference_rows'] == 2
    assert updated['paired_future']['response_row_range'] == [16, 17]
    assert len(agent.rows) == 18 and updated['fit_compute_cost'] == original['fit_compute_cost']
    assert method_question(agent, method, parent) is question
    assert question['agenda_basis'] != basis and question['status'] == 'open'
    assert question.get('completed_agenda_basis') != question['agenda_basis']
    entry = next(item for item in agent.research_agenda['theory_agenda']
                 if item['question_id'] == question['id'])
    assert entry['status'] == 'ready' and entry['work_kind'] == 'method_conditions'
    assert agent.theory_consolidation['candidates'][method['id']]['version'] == frozen['version']

    # Let the existing author consume the refreshed question, with its actual
    # complete comparison, rather than calling condition consolidation directly.
    memory = {'finite_native_author': {'question': deepcopy(parent),
        'predictor_paths': ['old.py'], 'predictor_catalog_ref': {'file': 'retained_catalog.json'},
        'max_compute': 3000, 'max_candidates': 4, 'boundary_resolution': .1},
        'program_results': [{'proposal': {'action': 'run', 'arguments': {
            'operator': 'research_choice', 'models': models}}, 'feedback': compared,
            'evidence_after': {'rows': len(agent.rows)}}]}
    monkeypatch.chdir(agent.workspace)
    agent._prepare_program_state()
    selected = select_actions(measured.public_spec(), agent.observations, memory)
    assert selected['action'] == 'run' and selected['arguments']['operator'] == 'research_route'
    assert selected['arguments']['research_question']['id'] == question['id']
    assert selected['arguments']['allocation']['compute'] <= mission['allocation']['compute']
    paid = deepcopy(agent.costs)
    continued = agent._execute(selected['action'], selected['arguments'], measured)
    current = agent.theory_consolidation['candidates'][method['id']]
    assert current['version'] == frozen['version'] + 1
    assert continued['condition_learning']['new_compute_cost'] == 1
    assert agent.costs['compute_cost'] - paid['compute_cost'] == 1
    assert agent.costs['observation_cost'] == paid['observation_cost']
    assert question['mission']['allocation'] == {'compute': 2000}
    assert _read(agent, current['previous_version_ref']) == frozen
    assert _condition_outcomes(agent, current['method_use_conditions']['outcomes_ref'])[0][
        'paired_future'] == updated['paired_future']
    assert 'paired_future' not in _condition_outcomes(agent,
        frozen['method_use_conditions']['outcomes_ref'])[0]
    returns = agent.theory_consolidation['method_feedback'][method['id']]['returns']
    origin, = returns['origins']
    trial = returns['action_experience'][origin['context']]['replay_research_program']
    assert trial['trials'] == 1 and trial['mean_compute_cost'] == original['fit_compute_cost']
    assert not agent.confirmations and agent.costs['model_calls'] == 0


def test_first_complete_method_feedback_is_visible_without_creating_a_mission_budget(tmp_path):
    agent, lab, method, parent = scene(tmp_path, useful=True)
    use_and_compare(agent, lab, parent)
    question = method_question(agent, method, parent)
    assert question['use_sites'] and not question.get('mission')
    entry = next(item for item in agent.research_agenda['theory_agenda']
                 if item['question_id'] == question['id'])
    assert entry['status'] == 'deferred' and entry['reason'] == 'no_theory_mission'
    assert agent.theory_consolidation['candidates'][method['id']]['version'] == method['version']
    assert agent.config.compute_budget == 50000 and not agent.research_mission
    assert lab.paid == 0 and agent.costs['model_calls'] == 0 and not agent.confirmations
