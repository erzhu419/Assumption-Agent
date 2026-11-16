"""Paid instrument answers constrain the live parent and return to the method topic."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.common_object_research import CommonObjectDependencyMethods, CommonObjectResearchAgent
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget
from hegel_machine.cognition.object_alternatives import enumerate_object_schemas
from test_paid_position_planner import scene
from test_position_method_formation import measured_stages


def paid_scene(tmp_path):
    _, model, _, _, _ = scene()
    qs = model['dynamic_fields']
    model.update(model_id='test_apparatus', unresolved=['x', 'f'],
        groups=[{'object_id': q, 'q': q, 'u': 'drive-' + q,
                 'controls': {'q': q, 'u': 'drive-' + q}} for q in qs])
    model['focus_candidates']['fc'] = {'quantities': qs}
    model['schema_alternatives'] = enumerate_object_schemas(model, SearchBudget(1000), max_schemas=16)
    refs = []
    for i, stage in enumerate(measured_stages()):
        pair = {}
        for name, value in stage.items():
            path = tmp_path / f'{i}-{name}.json'
            path.write_text(json.dumps(value))
            pair[name] = {'file': str(path)}
        refs.append(pair)
    return model, refs


def test_paid_position_returns_constrain_parent_and_close_same_source_research(tmp_path, monkeypatch):
    import hegel_machine.cognition.object_induction as induction
    import hegel_machine.cognition.unified_research as unified

    model, references = paid_scene(tmp_path)
    original = deepcopy(model)
    source = {'file': str(tmp_path / 'source.json')}
    actions, budget = [], SearchBudget(0)
    consumer = CommonObjectDependencyMethods(tmp_path / 'common', [], budget=budget,
        actions=actions, source_task_ref=source)
    def no_research(*args, **kwargs):
        raise AssertionError('Paid readout, fitting and scoring must be reused')
    monkeypatch.setattr(induction, 'assess_position_experiment', no_research)
    monkeypatch.setattr(consumer.common, '_execute', no_research)
    for pair in references:
        model = consumer.consume_paid_position_return(model,
            plan_ref=pair['position_experiment'], assessment_ref=pair['completed_relation'])
    assert model['position_candidates']['xb']['quantities'] == ['qb']
    assert next(group for group in model['groups'] if group['q'] == 'qb')['x'] == 'xb'
    assert all('qb' not in value['quantities'] for key, value in model['position_candidates'].items() if key != 'xb')
    assert 'schema_alternatives' not in model and original['schema_alternatives']['schemas']
    assert len([q for q in consumer.common.questions if q.get('paid_position_refs')]) == 2
    repeated = consumer.consume_paid_position_return(model,
        plan_ref=references[-1]['position_experiment'], assessment_ref=references[-1]['completed_relation'])
    assert repeated is model and len(actions) == 2 and budget.spent == 0
    assert consumer.consume_paid_position_return(model,
        plan_ref=references[0]['position_experiment'],
        assessment_ref=references[0]['completed_relation']) is model
    assert consumer.result['position_assessment']['retained_owners'] == ['qb']
    enumeration = enumerate_object_schemas(model, SearchBudget(1000), max_schemas=4)
    assert enumeration['schemas']
    assert all(next(item for item in schema['fields'].values() if item['q'] == 'qb')['x'] == 'xb'
               for schema in enumeration['schemas'])

    # Use the same retained and paid candidate scores, preserving the better
    # incumbent even though the new local role relation has been established.
    agent = CommonObjectResearchAgent(workspace=tmp_path / 'response', organization='pr3')
    parent = {'candidate': Candidate('retained', Program('const', value=1.), 'test', ''),
              'validation_loss': .1, 'training_loss': .1, 'kind': 'retained'}
    alternative = {'candidate': Candidate('old-fit', Program('const', value=2.), 'test', ''),
                   'training_loss': .2, 'kind': 'working_role_revision'}
    frame = dict(assessed=[parent], comparison_complete=True, uncompared=[], fallback=0.,
        spec=SimpleNamespace(risk_quantile=.5), independent_validation=False,
        comparison=[Observation({'x': 0.}, 1.)])
    monkeypatch.setattr(unified, 'loss', no_research)
    compared = agent.compare_research_decision_return(frame, 'structural_research', [alternative],
        budget, actions, paid_scores={alternative['candidate'].program: .2}, reused_paid_return=True)
    assert compared['selected'] is parent and compared['execution_complete'] and budget.spent == 0
    receipt = consumer.record_parent_response({'status': compared['status'],
        'execution_complete': compared['execution_complete'], 'program': parent['candidate'].program.to_dict(),
        'object_research': model, 'costs': {'compute_cost': 0}, 'actions': actions})
    feedback = json.loads(open(receipt['method_feedback']['feedback_ref']['file']).read())
    assert feedback['source_task_ref'] == source and feedback['parent_returned']
    assert feedback['execution_complete'] and feedback['independent_method_gain'] is None
    question = next(q for q in consumer.common.questions if q['id'] == 'task:method_choice')
    assert len(question['method_use_plan']['native_return_refs']) == 1


def test_paid_return_does_not_apply_another_parent_or_another_question(tmp_path):
    model, refs = paid_scene(tmp_path)
    consumer = CommonObjectDependencyMethods(tmp_path / 'common', [], budget=SearchBudget(0), actions=[])
    pair = refs[0]
    with pytest.raises(ValueError, match='parent and experiment'):
        consumer.consume_paid_position_return({**model, 'model_id': 'another_parent'},
            plan_ref=pair['position_experiment'], assessment_ref=pair['completed_relation'])
    plan = json.loads(open(pair['position_experiment']['file']).read())
    plan['excited_quantity'] = 'qa'
    open(pair['position_experiment']['file'], 'w').write(json.dumps(plan))
    with pytest.raises(ValueError, match='parent and experiment'):
        consumer.consume_paid_position_return(model,
            plan_ref=pair['position_experiment'], assessment_ref=pair['completed_relation'])
    assert not consumer.actions


@pytest.mark.parametrize('status', ['position_domain_conflict', 'unverified_position_intervention'])
def test_paid_conflict_preserves_evidence_and_keeps_membership_unresolved(tmp_path, status):
    from hegel_machine.cognition.object_induction import apply_position_experiment, assess_position_experiment

    model, refs = paid_scene(tmp_path)
    pair = refs[0] if status == 'unverified_position_intervention' else refs[1]
    if status == 'position_domain_conflict':
        model = apply_position_experiment(model, json.loads(open(refs[0]['completed_relation']['file']).read()))
    plan = json.loads(open(pair['position_experiment']['file']).read())
    record = scene()[-1]
    corner_refs = plan.get('paid_corner_refs', [plan['baseline_ref'], None, None, None])
    rows = [record(ref['probe_id'] if ref else f'corner-{i}', point)
            for i, (ref, point) in enumerate(zip(corner_refs, plan['corners']))]
    if status == 'unverified_position_intervention':
        # The nominal qb intervention also moves qa: the real assessor exits
        # before constructing response comparisons or surviving owner labels.
        rows[1]['before']['qa'] += .25
    else:
        rows[-1]['after']['qb'] += 1.
        rows[-1]['after']['qc'] += 1.
    measured = assess_position_experiment(plan, model, rows, SearchBudget(1000))
    assert measured['status'] == status
    assert ('comparison' in measured) == (status == 'position_domain_conflict')
    original = deepcopy(model)
    path = pair['completed_relation']['file']
    open(path, 'w').write(json.dumps(measured))
    consumer = CommonObjectDependencyMethods(tmp_path / 'common', [], budget=SearchBudget(0), actions=[])
    returned = consumer.consume_paid_position_return(model,
        plan_ref=pair['position_experiment'], assessment_ref=pair['completed_relation'])
    assert returned == original
    child = next(q for q in consumer.common.questions if q.get('paid_position_refs'))
    assert child['status'] == 'unresolved' and child['answer']['status'] == status
    assert child['evidence'] == [pair['completed_relation']]
