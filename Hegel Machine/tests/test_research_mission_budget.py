"""Actual question owners share global costs while retaining finite theory slices."""

from copy import deepcopy

from test_question_routing import CONTEXT, future_question, route, scene


def mission(compute=2, identifier='offline_theory'):
    return {'mission_id': identifier, 'purpose': 'theory_learning',
            'theory_learning_allocation': {'compute': compute}}


def measured_question(agent, identifier='measured_theory', *, difference=False, owner=None):
    record = agent.observations[0]

    def readout(step):
        row = record['rows'][step]
        return {'op': 'Readout', 'record_index': 0, 'step': step,
            'context_ref': deepcopy(CONTEXT), 'units': 'V',
            'binding': {'action': deepcopy(record['action']),
                'features': deepcopy(row['features']), 'history': list(row.get('history', []))}}

    query = ({'op': 'Difference', 'left': readout(1), 'right': readout(0)}
             if difference else readout(0))
    question = future_question(agent, identifier)
    question.update(queue='theory', use_sites=[], query_semantics={
        'mode': 'model_query', 'query_expression': query})
    if owner is not None:
        question['mission'] = deepcopy(owner)
    return question


def retained(agent, identifier='measured_theory'):
    return next(q for q in agent.questions if q['id'] == identifier)


def test_question_own_theory_mission_executes_without_replacing_global_task_mission(tmp_path):
    agent, lab = scene(tmp_path)
    question = measured_question(agent, owner=mission())
    task = {'mission_id': 'main_task', 'purpose': 'task_prediction'}
    before = deepcopy(agent.costs)
    result = route(agent, lab, question=question, mission=task)
    assert result['answer_execution']['status'] == 'answer_ready_no_parent_use'
    assert result['answer_execution']['answer']['value'] == 0.
    assert retained(agent)['mission']['resource_usage']['compute'] == 1
    assert agent.research_mission == task
    assert agent.costs['compute_cost'] - before['compute_cost'] == 1
    assert agent.costs['observation_cost'] == before['observation_cost']
    assert agent.costs['model_calls'] == 0


def test_one_unit_slice_stops_two_readout_query_and_actual_debit_is_retained(tmp_path):
    agent, lab = scene(tmp_path)
    agent._open_question(measured_question(agent, difference=True))
    before = agent.costs['compute_cost']
    result = route(agent, lab, mission=mission(1))
    assert result['answer_execution']['status'] == 'scoped_unknown'
    assert result['answer_execution']['reason'] == 'unknown_due_to_incomplete_search'
    assert agent.costs['compute_cost'] - before == 1
    assert agent.research_mission['resource_usage']['compute'] == 1
    again = route(agent, lab, mission=mission(1))
    assert again['status'] == 'research_deferred'
    assert again['agenda']['selected_question_id'] is None
    assert agent.costs['compute_cost'] - before == 1


def test_repeated_global_declaration_does_not_refund_prior_questions_or_cached_work(tmp_path):
    agent, lab = scene(tmp_path)
    before = agent.costs['compute_cost']
    for identifier in ('first', 'second'):
        result = route(agent, lab, question=measured_question(agent, identifier), mission=mission(2))
        assert result['answer_execution']['status'] == 'answer_ready_no_parent_use'
    assert agent.research_mission['resource_usage']['compute'] == 2
    assert agent.costs['compute_cost'] - before == 2
    cached = route(agent, lab, question={'id': 'first'}, mission=mission(2))
    assert cached['new_compute_cost'] == 0
    assert cached['answer_execution']['answer']['value'] == 0.
    assert agent.research_mission['resource_usage']['compute'] == 2
    third = route(agent, lab, question=measured_question(agent, 'third'), mission=mission(2))
    assert third['answer_execution']['status'] == 'scoped_unknown'
    assert agent.costs['compute_cost'] - before == 2


def test_full_question_redeclaration_keeps_its_local_mission_debit_and_old_answer_cache(tmp_path):
    agent, lab = scene(tmp_path)
    question = measured_question(agent, owner=mission())
    route(agent, lab, question=question, mission={'purpose': 'task_prediction'})
    before = deepcopy(agent.costs)
    repeated = route(agent, lab, question=question)
    assert repeated['new_compute_cost'] == 0 and agent.costs == before
    assert retained(agent)['mission']['resource_usage']['compute'] == 1


def test_two_questions_of_one_named_mission_share_usage_under_a_distinct_global_task(tmp_path):
    agent, lab = scene(tmp_path)
    before = agent.costs['compute_cost']
    first = route(agent, lab, question=measured_question(agent, 'first', owner=mission(1)),
                  mission={'purpose': 'task_prediction'})
    assert first['answer_execution']['answer']['value'] == 0.
    second = route(agent, lab, question=measured_question(agent, 'second', owner=mission(1)))
    assert second['answer_execution']['status'] == 'scoped_unknown'
    assert agent.costs['compute_cost'] - before == 1
    assert retained(agent, 'first')['mission']['resource_usage'] == {'compute': 1}
    assert retained(agent, 'second')['mission']['resource_usage'] == {'compute': 1}
    assert agent.research_mission == {'purpose': 'task_prediction'}


def test_explicitly_increased_same_mission_slice_reopens_unknown_without_refunding_work(tmp_path):
    agent, lab = scene(tmp_path)
    question = measured_question(agent, difference=True)
    first = route(agent, lab, question=question, mission=mission(1))
    assert first['answer_execution']['status'] == 'scoped_unknown'
    second = route(agent, lab, question={'id': question['id']}, mission=mission(3))
    assert second['answer_execution']['answer']['value'] == 1.
    assert agent.research_mission['resource_usage']['compute'] == 3
    third = route(agent, lab, question={'id': question['id']}, mission=mission(3))
    assert third['new_compute_cost'] == 0


def test_task_owner_keeps_original_global_budget_and_no_theory_debit(tmp_path):
    agent, lab = scene(tmp_path)
    question = measured_question(agent, difference=True)
    question.update(queue='task', use_sites=[{'kind': 'choose_parent_experiment', 'parent_id': 'task'}])
    before = agent.costs['compute_cost']
    result = route(agent, lab, question=question,
        mission={'purpose': 'task_prediction', 'allocation': {'compute': 1}})
    assert result['answer_execution']['answer']['value'] == 1.
    assert agent.costs['compute_cost'] - before == 2
    assert 'resource_usage' not in agent.research_mission


def test_find_distinguishing_matching_and_selection_debit_one_shared_slice(tmp_path):
    from test_find_distinguishing_experiment import search_question
    from test_question_routing import SQUARE, X, probe, theory
    from hegel_machine.cognition.question_routing import register_theories

    agent, lab = scene(tmp_path)
    register_theories(agent, [theory('linear', X), theory('quadratic', SQUARE)])
    question = search_question(agent)
    question['mission'] = mission(3, 'bounded_distinguishing')
    question['resolution_experiments'] = [probe(agent)]
    before = deepcopy(agent.costs)
    result = agent._execute('run', {'operator': 'research_step', 'research_question': question}, lab)
    assert result['query_answer']['status'] == 'scoped_unknown'
    actual = agent.costs['compute_cost'] - before['compute_cost']
    assert 0 < actual <= 3
    assert retained(agent, question['id'])['mission']['resource_usage']['compute'] == actual
    assert result['selection']['selected_experiment'] is None
    assert agent.costs['observation_cost'] == before['observation_cost']


def test_default_route_returns_missing_authored_object_without_invoking_provider(tmp_path):
    agent, lab = scene(tmp_path)
    agent._open_question({'id': 'construct', 'question': 'Construct a response representation',
        'kind': 'discover_object', 'answer_type': 'program',
        'answer_resolution_ref': {'kind': 'executable_program'},
        'query_semantics': {'mode': 'solution_relation'},
        'use_sites': [{'kind': 'supply_component', 'model': 'parent.py', 'slot': 'state'}]})
    before = deepcopy(agent.costs)
    result = route(agent, lab)
    assert result['status'] == 'author_work_request'
    assert result['question_id'] == 'construct'
    assert retained(agent, 'construct')['pending_author_work']['question_id'] == 'construct'
    assert result['agenda']['deferred'] and agent.costs == before
    assert agent.costs['model_calls'] == 0


def test_parent_refit_exhausted_slice_returns_unknown_and_restores_original_budget(tmp_path):
    from test_evidence_procedure_execution import prepared
    from test_executable_evidence_binding import return_region
    from test_question_continuation import region_use

    agent, lab = prepared(tmp_path)
    return_region(agent, lab, .9, 1.1)
    question = region_use(agent)
    question['mission'] = mission(1, 'parent_refit')
    before, config = deepcopy(agent.costs), agent.config
    result = agent._execute('run', {'operator': 'research_step',
        'research_question': {'id': question['id']}}, lab)
    assert result['status'] == 'scoped_unknown'
    assert result['reason'] == 'unknown_due_to_incomplete_search'
    assert not result.get('parent_revision_completed', False)
    assert 'applied_question_use' not in retained(agent, question['id'])
    assert agent.config is config
    assert all(agent.costs[key] == before[key] for key in
               ('compute_cost', 'observation_cost', 'measurement_cost', 'model_calls'))
