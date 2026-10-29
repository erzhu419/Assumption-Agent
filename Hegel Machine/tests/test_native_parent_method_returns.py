"""Native parent returns retain evidence without creating a method victory."""

import json

import pytest

from hegel_machine.cognition.common_object_research import (
    CommonObjectDependencyMethods, record_native_parent_research_return,
)
from hegel_machine.cognition.models import SearchBudget


def make_consumer(tmp_path, source=None):
    return CommonObjectDependencyMethods(tmp_path / 'common', [],
        budget=SearchBudget(100), actions=[], source_task_ref=source)


def saved(path, value):
    path.write_text(json.dumps(value))
    return {'file': str(path)}


def test_paid_native_receipt_enters_existing_topic_without_scalar_or_pair_claim(tmp_path):
    source = saved(tmp_path / 'targets.json', {'seed': 106029101})
    receipt = saved(tmp_path / 'receipt.json', {'status': 'confirmed', 'full_comparison': True,
        'paid_rows': 236, 'paid_mse': 1.27e-21, 'compute_cost': 17822,
        'confirmation': {'passed': True, 'rows': 48}})
    options = saved(tmp_path / 'options.json', {'options': []})
    consumer = make_consumer(tmp_path)
    common = consumer.common
    before = dict(common.costs)
    result = record_native_parent_research_return(common, source_task_ref=source,
        result_ref=receipt, comparison_ref={**receipt, 'json_path': ['full_comparison']},
        confirmation_ref={**receipt, 'json_path': ['confirmation']},
        cost_refs=[{**receipt, 'json_path': ['compute_cost']}], decision_options_ref=options)
    record = json.loads(open(result['feedback_ref']['file']).read())
    assert record['source_task_ref'] == source
    assert record['actual_result_ref'] == receipt
    assert record['decision_options_ref'] == options
    assert record['parent_returned'] is True
    assert 'execution_complete' not in record
    assert record['paired_method_effect'] is record['independent_method_gain'] is None
    owner = next(q for q in common.questions if q['id'] == 'task:method_choice')
    assert owner['kind'] == 'method_use' and owner['answer_type'] == 'research_action'
    assert owner['method_use_plan'] == {'native_return_refs': [result['feedback_ref']]}
    assert not owner['candidate_method_refs']
    assert owner['mission']['allocation']['compute'] == 0
    assert common.costs == before


def test_duplicate_return_is_reused_new_return_reopens_same_source_topic(tmp_path):
    source = saved(tmp_path / 'target.json', {'seed': 106029101})
    old = saved(tmp_path / 'old.json', {'status': 'unresolved'})
    new = saved(tmp_path / 'new.json', {'status': 'confirmed'})
    common = make_consumer(tmp_path).common
    first = record_native_parent_research_return(common, source_task_ref=source, result_ref=old,
        mission={'purpose': 'method_learning', 'allocation': {'compute': 8}})
    owner = next(q for q in common.questions if q['id'] == 'task:method_choice')
    owner.update(status='answered', completed_agenda_basis=owner['agenda_basis'])
    repeated = record_native_parent_research_return(common, source_task_ref=source, result_ref=old)
    assert repeated['status'] == 'reused_native_return'
    assert repeated['feedback_ref'] == first['feedback_ref']
    assert owner['status'] == 'answered'
    assert len(list((common.workspace / 'native_method_returns').iterdir())) == 1
    second = record_native_parent_research_return(common, source_task_ref=source, result_ref=new)
    assert second['method_question_refs'] == first['method_question_refs']
    assert owner['status'] == 'open' and 'completed_agenda_basis' not in owner
    assert owner['mission']['allocation']['compute'] == 8
    records = [json.loads(open(ref['file']).read()) for ref in owner['method_use_plan']['native_return_refs']]
    assert len(records) == 2 and all(record['source_task_ref'] == source for record in records)
    assert not common.theory_consolidation.get('candidates')


def test_ordinary_parent_return_preserves_versions_and_actual_published_options(tmp_path):
    source = saved(tmp_path / 'target.json', {'seed': 106029101})
    consumer = make_consumer(tmp_path, source)
    actual = {'status': 'unresolved', 'program': {'op': 'const', 'value': 1},
        'costs': {'compute_cost': 20}, 'actions': [{'action': 'publish_object_method_options',
            'options': [{'id': 'ordinary'}], 'ordinary_option_id': 'ordinary'}]}
    first = consumer.record_parent_response(actual)
    original_text = open(first['response_ref']['file']).read()
    repeated = consumer.record_parent_response(actual)
    assert repeated['method_feedback']['status'] == 'reused_native_return'
    actual['program']['value'] = 2
    second = consumer.record_parent_response(actual)
    assert second['response_ref'] != first['response_ref']
    assert open(first['response_ref']['file']).read() == original_text
    record = json.loads(open(second['method_feedback']['feedback_ref']['file']).read())
    assert record['status'] == 'unresolved' and 'execution_complete' not in record
    assert record['decision_options_ref']['json_path'] == ['decision_options']
    assert record['source_task_ref'] == source


def test_missing_target_identity_is_unresolved_and_not_replaced_by_workspace(tmp_path):
    consumer = make_consumer(tmp_path)
    result = consumer.record_parent_response({'status': 'unresolved', 'program': None})
    record = json.loads(open(result['method_feedback']['feedback_ref']['file']).read())
    assert record['source_task_ref'] is None and record['source_identity_status'] == 'unresolved'


def test_existing_topic_transport_retains_native_returns_across_episodes(tmp_path):
    from hegel_machine.cognition.method_feedback import (
        ensure_python_method_topic, export_python_method_topic,
    )

    source = saved(tmp_path / 'target.json', {'target': 1})
    receipt = saved(tmp_path / 'receipt.json', {'status': 'confirmed'})
    original = make_consumer(tmp_path / 'first').common
    recorded = record_native_parent_research_return(original,
        source_task_ref=source, result_ref=receipt)
    owner = next(q for q in original.questions if q['id'] == 'task:method_choice')
    exported = export_python_method_topic(original, owner)
    restored = make_consumer(tmp_path / 'second').common
    ensure_python_method_topic(restored, parent_question_ref={'question_id': 'task'},
        topic_ref=exported)
    repeated = record_native_parent_research_return(restored,
        source_task_ref=source, result_ref=receipt)
    assert repeated['status'] == 'reused_native_return'
    assert repeated['feedback_ref'] == recorded['feedback_ref']
    assert not (restored.workspace / 'native_method_returns').exists()
    assert not restored.theory_consolidation.get('candidates')


def test_one_return_cannot_be_relabelled_as_another_target(tmp_path):
    receipt = saved(tmp_path / 'receipt.json', {'status': 'confirmed'})
    common = make_consumer(tmp_path).common
    record_native_parent_research_return(common, source_task_ref={'target': 1}, result_ref=receipt)
    with pytest.raises(ValueError, match='original target identity'):
        record_native_parent_research_return(common, source_task_ref={'target': 2}, result_ref=receipt)
