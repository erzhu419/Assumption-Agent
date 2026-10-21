"""Explicit plan readouts bind real Common purchases, without scientific runs."""

from copy import deepcopy

from hegel_machine.cognition.question_queries import evaluate_on_evidence
from test_inquiry_revision import action
from test_question_routing import CONTEXT, future_question, parent, probe, readout, route, scene, theory, X, SQUARE


def test_pending_readout_uses_its_actual_purchase_after_an_intervening_probe(tmp_path):
    agent, lab = scene(tmp_path)
    agent._execute('query', {'actions': [action(0.), action(1.)]}, lab)
    assert len(agent.observations) == 3
    reference = {'experiment_id': 'declared_reset', 'action_index': 0}
    question = future_question(agent)
    question['query_semantics']['query_expression'] = readout(3., reference)
    panel = [probe(agent, 3., 'declared_reset')]
    panel[0]['query']['record_index'] = deepcopy(reference)
    agent._open_question(question)
    # The original next index has been occupied by another real operation.
    agent._execute('query', {'actions': [action(-1.)]}, lab)
    selected = route(agent, lab, question=question,
                     theories=[theory('linear', X), theory('quadratic', SQUARE)], experiments=panel)
    assert selected['status'] == 'selected'
    assert selected['answer_execution']['status'] == 'scoped_unknown'
    assert selected['answer_execution']['reason'] == 'unknown_due_to_missing_evidence'
    assert lab.paid == agent.costs['observation_cost'] == 5
    purchased = agent._execute('query', {'pending_question': 'task'}, lab)
    assert purchased['record_indices'] == [4]
    executed, = parent(agent)['executed_pending_queries']
    assert executed['source']['experiment_id'] == 'declared_reset'
    assert 'pending_query' not in parent(agent)
    before = deepcopy(agent.costs)
    answered = route(agent, lab, question={'id': question['id']})
    current = next(q for q in agent.questions if q['id'] == question['id'])
    assert current['semantic_answer']['value'] == 3.
    assert current['semantic_answer']['evidence'] == agent.observations[4]['row_indices']
    assert answered['selection'] is None and 'pending_query' not in parent(agent)
    assert agent.costs['observation_cost'] == before['observation_cost'] == lab.paid == 6
    # An absolute index continues to mean that exact retained record.
    absolute = evaluate_on_evidence(readout(3., 3), agent.observations, context=CONTEXT)
    assert absolute['status'] == 'invalid_experiment_binding'
    assert absolute['reason'] == 'action_mismatch'
    other = deepcopy(question)
    other['id'] = 'another_question'
    unbound = agent._execute('run', {'operator': 'research_step', 'research_question': other}, lab)
    assert unbound['reason'] == 'unknown_due_to_missing_evidence'
    changed = agent._open_question({'id': question['id'], 'answer_resolution_ref': {'atol': .1}})
    assert changed['meaning_version'] == 2
    unbound = agent._execute('run', {'operator': 'research_step',
        'research_question': {'id': question['id']}}, lab)
    assert unbound['reason'] == 'unknown_due_to_missing_evidence'
    assert 'semantic_answer' not in changed
    assert agent.model is None and not agent.confirmations and agent.costs['model_calls'] == 0
