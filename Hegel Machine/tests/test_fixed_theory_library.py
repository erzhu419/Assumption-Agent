"""A registered recall library controls real PR2 behavior, without acquisition."""

from copy import deepcopy
import json

from hegel_machine.cognition.question_continuation import pending_query_arguments
from hegel_machine.cognition.question_routing import resolved_theories
from hegel_machine.cognition.theory_consolidation import LIBRARY_PATH
from test_question_routing import CONTEXT, SQUARE, future_question, probe, theory
from test_theory_consolidation import agent_scene, affine, consolidate, transfer, assert_no_new_science


def scene(tmp_path):
    source, source_lab = agent_scene(tmp_path / 'source')
    compiled = consolidate(source, source_lab)
    identifier = compiled['frozen_candidates'][0]['candidate_id']
    target, lab = agent_scene(tmp_path / 'target')
    target.theory_consolidation = deepcopy(source.theory_consolidation)
    working = target.theory_consolidation['candidates'][identifier]
    working['version'] = 2
    working['instances'][0]['parameters'][0] = 4.
    target._save(LIBRARY_PATH, json.dumps(target.theory_consolidation))
    return source, source_lab, target, lab, identifier


def route(agent, lab, path, *, full=False):
    arguments = {'operator': 'research_route', 'library_path': path,
        'research_question': future_question(agent) if full else {'id': 'future_response'},
        'mission': {'purpose': 'task', 'mission_id': 'fixed_library_task'}}
    if full:
        arguments.update(theories=[theory('square', SQUARE)], experiments=[probe(agent)])
    return agent._execute('run', arguments, lab)


def test_fixed_source_drives_recall_cache_and_pending_despite_working_library_changes(tmp_path):
    source, source_lab, target, lab, identifier = scene(tmp_path)
    fixed = str(source.workspace / LIBRARY_PATH)
    original = (source.workspace / LIBRARY_PATH).read_bytes()
    result = route(target, lab, fixed, full=True)
    assert result['status'] == 'selected'
    assert sorted(group['task_answer'] for group in result['selection']['groups']) == [7., 9.]
    recalled, = result['frozen_recall']['generated_theory_ids']
    assert target.theory_index[recalled]['program'] == affine().to_dict()
    question = next(q for q in target.questions if q['id'] == 'future_response')
    assert question['routing_execution']['basis']['frozen_library_path'] == fixed
    assert question['routing_execution']['basis']['frozen_library'] == [{'id': identifier, 'version': 1}]
    before = pending_query_arguments(target, 'task')
    target.theory_consolidation['candidates'][identifier]['version'] = 3
    target._save(LIBRARY_PATH, json.dumps(target.theory_consolidation))
    assert pending_query_arguments(target, 'task') == before
    again = route(target, lab, fixed)
    assert again['cache_hit'] and again['new_compute_cost'] == 0
    assert (source.workspace / LIBRARY_PATH).read_bytes() == original
    assert_no_new_science(source, source_lab)
    assert_no_new_science(target, lab)


def test_empty_registered_library_does_not_recall_the_working_library(tmp_path):
    source, source_lab, target, lab, identifier = scene(tmp_path)
    empty = tmp_path / 'initial_empty_library.json'
    empty.write_text('{"candidates": {}}')
    result = route(target, lab, str(empty), full=True)
    assert result['frozen_recall']['status'] == 'no_frozen_candidates'
    assert result['frozen_recall']['generated_theory_ids'] == []
    assert identifier in target.theory_consolidation['candidates']
    assert_no_new_science(target, lab)


def test_switching_registered_library_changes_recalled_predictions_and_cache_basis(tmp_path):
    source, source_lab, target, lab, identifier = scene(tmp_path)
    first = route(target, lab, str(source.workspace / LIBRARY_PATH), full=True)
    second = route(target, lab, LIBRARY_PATH)
    assert not second['cache_hit']
    assert sorted(group['task_answer'] for group in first['selection']['groups']) == [7., 9.]
    assert sorted(group['task_answer'] for group in second['selection']['groups']) == [9., 13.]
    recalled, = second['frozen_recall']['generated_theory_ids']
    assert target.theory_index[recalled]['program'] == affine(4.).to_dict()
    assert target.theory_index[recalled]['version'] == 2
    assert_no_new_science(target, lab)


def test_external_selected_library_keeps_its_paid_counterexample_without_target_index(tmp_path):
    source, source_lab = agent_scene(tmp_path / 'source')
    compiled = consolidate(source, source_lab)
    identifier = compiled['frozen_candidates'][0]['candidate_id']
    failed = transfer(source, source_lab, identifier, evidence_checks=[{
        'source_ref': {'file': 'retained_bad_row.json', 'json_path': [0]},
        'context_ref': deepcopy(CONTEXT), 'tolerance': 0.,
        'observation': {'features': {'u': 1.}, 'target': 30., 'history': []}}])
    assert failed['status'] == 'refuted'
    target, lab = agent_scene(tmp_path / 'target')
    result = route(target, lab, str(source.workspace / LIBRARY_PATH), full=True)
    assert result['frozen_recall']['generated_theory_ids'] == []
    assert result['frozen_recall']['work'][0]['status'] == 'refuted'
    target._execute('run', {'operator': 'research_route', 'theories': [theory('failed_alias', affine())]}, lab)
    alias = next(item for item in resolved_theories(target) if item['id'] == 'failed_alias')
    assert alias['compatible'] is False
    assert alias['scope_refutation']['counterexample_refs']
    assert_no_new_science(target, lab)
