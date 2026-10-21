"""The actual Registered hook shares old Python predictions with PR2."""
from copy import deepcopy

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from scripts.run_anonymous_revision_inquiry import RegisteredOrganizationAgent
from test_finite_native_author import ChoiceLab, declaration
from test_inquiry_revision import action
from test_question_routing import future_question, probe


def test_registered_python_roster_enters_PR2_and_same_scope_scores_are_reused(tmp_path, monkeypatch):
    library = tmp_path / 'empty.json'
    library.write_text('{"candidates": {}}')
    models = ['archive/first/_evaluated_model/model.py',
              'archive/second/_evaluated_model/model.py']
    agent = RegisteredOrganizationAgent(None, tmp_path / 'work', arm='fixed_pr2',
        research_organization={'phase': 'target', 'initial_library_path': str(library),
                               'exploration_policy': {'kind': 'seeded_legal', 'seed': 11}},
        source_task=False, knowledge={'files': {
            models[0]: 'def predict(row): return row["features"]["u"]\n',
            models[1]: 'def predict(row): return -row["features"]["u"]\n'}},
        config=InquiryLLMConfig(observation_budget=20, compute_budget=3000,
                               confirmation_rows=2, execution_seconds=30., max_model_calls=0))
    lab = ChoiceLab()
    agent._execute('query', {'actions': [action(1., 2.)]}, lab)
    question = future_question(agent)
    question['resolution_experiments'] = [probe(agent, 2., 'positive'), probe(agent, -2., 'negative')]
    owner = agent._open_question(question)
    contract = declaration()
    contract.update(question=deepcopy(owner), predictor_paths=models)
    agent.knowledge['finite_native_author'] = contract
    agent._active_question_id = owner['id']
    before = deepcopy(agent.costs)
    def no_acquisition(*args, **kwargs):
        raise AssertionError('PR2 preparation must not acquire observations')

    monkeypatch.setattr(lab, 'execute', no_acquisition)
    agent._before_model_turn(lab)
    routing = owner['routing_execution']['result']
    assert routing['python_forecasts']['model_count'] == 2
    assert routing['python_forecasts']['ready_count'] == 2
    assert routing['selection']['prediction_coverage']['complete']
    assert len(routing['selection']['groups']) == 2
    parent = next(q for q in agent.questions if q['id'] == 'task')
    assert parent['pending_query_source']['question_id'] == owner['id']
    comparison = {'operator': 'research_choice', 'models': models,
                  'research_question': {'id': owner['id']}}
    first = agent._execute('run', comparison, lab)
    assert first['new_scored_count'] == 2
    measured = deepcopy(agent.costs)
    retained = deepcopy(agent.evaluated_model)
    second = agent._execute('run', comparison, lab)
    assert second['reused_score_count'] == 2 and second['new_scored_count'] == 0
    assert [x['mse'] for x in second['evaluations']] == [x['mse'] for x in first['evaluations']]
    assert agent.costs['compute_cost'] == measured['compute_cost']
    assert agent.costs['python_execution_seconds'] >= measured['python_execution_seconds']
    assert all(item['paid_score_reuse']['additional_scoring_compute'] == 0
               and item['paid_score_reuse']['additional_scoring_python_seconds'] == 0.
               for item in second['evaluations'])
    assert agent.evaluated_model == retained
    assert agent.costs['observation_cost'] == before['observation_cost'] == 2
    assert agent.costs['model_calls'] == 0 and not agent.confirmations
