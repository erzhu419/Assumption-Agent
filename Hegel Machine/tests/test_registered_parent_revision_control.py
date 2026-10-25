"""A declared parent use reaches the registered runner without changing the Readout."""
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hegel_machine.cognition import registered_source_control as control
from hegel_machine.cognition import registered_source_scene as scenes
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.question_routing import register_theories


@dataclass
class Limits:
    compute_budget: int = 1000
    execution_seconds: float = 300.


def metadata_scene(tmp_path):
    from test_theory_consolidation import agent_scene
    from test_question_routing import theory, CONTEXT

    agent, lab = agent_scene(tmp_path)
    parent = theory('actual_parent195', Program('feature', name='u'))
    old = theory('old_parent65', Program('const', value=0.))
    register_theories(agent, [parent, old])
    rows = [{'features': {'u': float(i)}, 'target': float(i)+1., 'history': []} for i in range(4)]
    agent.rows = [Observation(x['features'], x['target']) for x in rows]
    agent.observations = [{'rows': rows}]
    agent.evaluated_model = {'source_model': 'retained.py', 'current_mse': .1, 'evaluated_rows': 4}
    source = agent._open_question({'id': 'native_public_response', 'parent_id': 'task',
        'question': 'The measured response.', 'kind': 'value', 'answer_type': 'number',
        'context_ref': deepcopy(CONTEXT), 'theory_target': {'parent_theory_id': old['id'],
            **{key: deepcopy(parent[key]) for key in ('role_signature', 'output_unit', 'context_ref')}},
        'query_semantics': {'mode': 'model_query', 'query_expression': {
            'op': 'Readout', 'record_index': 0, 'step': 0}},
        'use_sites': [{'kind': 'choose_parent_experiment', 'parent_id': 'task'}]})
    source['semantic_answer'] = {'status': 'answered', 'value': .25, 'evidence': [0]}
    source['frozen_recall_execution'] = {'basis': {'parent': {'id': old['id']}},
        'cursor': {'next_candidate_id': 'old'}, 'result': {'search_completed': False}}
    agent.theory_consolidation = {'candidates': {'old': {'id': 'old'}, 'new': {'id': 'new'}},
        'executions': {'relation_transfer:frozen_recall:native_public_response@1:old': {
            'cursor': {'layer': 3, 'offset': 8, 'pending_fits': [{'original': True}]}}}}
    document = tmp_path/'declared.json'
    ref = lambda *parts: {'file': str(document), 'json_path': list(parts)}
    checks = [{'source_ref': ref('rows', i), 'context_ref': deepcopy(CONTEXT),
        'criterion': {'kind': 'paid_mse', 'evaluated_rows': 4,
            'mse_threshold': agent.config.confirmation_mse_threshold,
            'criterion_ref': ref('threshold')}} for i in range(4)]
    fitting = {'operations': ['refit'], 'risk_quantile': .5, 'max_history': 2, 'max_structures': 1}
    roster = [{'id': old['id'], 'version': 1, 'source_ref': None}]
    grammar = {'structures': [{'program': Program('const', value=0.).to_dict(),
        'parameter_paths': [[]]}], 'grammar_cursor': {'status': 'completed'},
        'excluded_structures': [{'parent65_only': True}]}
    document.write_text(json.dumps({'rows': rows, 'checks': checks, 'fitting': fitting,
        'roster': roster, 'grammar': grammar, 'threshold': agent.config.confirmation_mse_threshold,
        'request': {'components': roster, 'evidence_checks': checks, 'fitting': fitting}}))
    registration = {'parent_research': {'source_question_id': source['id'],
        'question_id': 'parent_revision:task:native_public_response', 'meaning_version': 1,
        'parent_theory_ref': {'id': parent['id'], 'version': 1},
        'evidence_checks_ref': ref('checks'), 'fitting_ref': ref('fitting'),
        'component_roster_ref': ref('roster'), 'grammar_preparation_ref': ref('grammar'),
        'grammar_context_ref': ref('request')}}
    return {'agent': agent, 'lab': lab, 'models': ['retained.py', 'failed.py'],
        'retained_fit_constructors': {**agent.theory_consolidation['candidates'],
            'historical_removed': {'id': 'historical_removed', 'version': 1}},
        'historical_fit_library_ref': {'file': 'retained/shared/theory_library.json',
                                       'json_path': []}}, registration


def test_production_parent_setup_preserves_Readout_65_cursor_and_normalizes_full_paid_contract(
        tmp_path, monkeypatch):
    scene, registration = metadata_scene(tmp_path)
    agent = scene['agent']
    source, costs = deepcopy(agent.questions[-1]), deepcopy(agent.costs)
    executions = deepcopy(agent.theory_consolidation['executions'])
    monkeypatch.setattr(Program, 'predict', lambda *args: pytest.fail('metadata setup must not predict'))
    result = scenes.prepare_registered_parent_revision(scene, registration)
    obligation = next(q for q in agent.questions if q['id'] == result['research_question_ref']['question_id'])
    assert obligation['theory_target']['parent_theory_id'] == 'actual_parent195'
    assert obligation['subject_refs'][0] == {'kind': 'theory', 'id': 'actual_parent195', 'version': 1}
    assert obligation['paid_score_question_ref'] == {'question_id': source['id'], 'meaning_version': 1}
    assert next(q for q in agent.questions if q['id'] == source['id']) == source
    assert agent.costs == costs and agent.theory_consolidation['executions'] == executions
    assert not result['current_parent_scope_has_prior_executions']
    grammar = scene['registered_revision_inputs']['_grammar_preparation']
    assert set(grammar['state']) == {'structures', 'grammar_cursor'}
    assert 'excluded_structures' not in grammar['state']
    assert len(obligation['theory_target']['evidence_checks']) == 4
    inputs = scene['registered_revision_inputs']['_retained_execution_inputs']
    assert all(x['_retained_parent']['id'] == 'actual_parent195' for x in inputs.values())
    assert all([x['id'] for x in v['_retained_components']] == ['old_parent65'] for v in inputs.values())
    assert 'historical_removed' in scene['registered_revision_inputs']['_retained_fit_constructors']
    assert 'historical_removed' not in agent.theory_consolidation['candidates']


def test_parent_setup_rejects_an_unresolved_retained_typed_contract(tmp_path, monkeypatch):
    scene, registration = metadata_scene(tmp_path)
    monkeypatch.setattr('hegel_machine.cognition.theory_consolidation._retained_parent_contract',
        lambda agent, parent: {**parent, '_contract_source': {
            'reason': 'retained_parent_transfer_reference_required'}})
    before = deepcopy(scene['agent'].questions)
    with pytest.raises(ValueError, match='registered parent typed contract is unresolved'):
        scenes.prepare_registered_parent_revision(scene, registration)
    assert scene['agent'].questions == before
    assert not (tmp_path/'parent_revision_acceptance.json').exists()


def test_parent_setup_rejects_three_witness_rows_as_substitute_for_full_fitting_evidence(tmp_path):
    scene, registration = metadata_scene(tmp_path)
    reference = registration['parent_research']['evidence_checks_ref']
    path = __import__('pathlib').Path(reference['file'])
    data = json.loads(path.read_text())
    data['checks'] = data['checks'][:3]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='complete original paid-row contract'):
        scenes.prepare_registered_parent_revision(scene, registration)


def runner_scene(tmp_path, monkeypatch, *, wrong_owner=False, missing_unit=False):
    reference = {'question_id': 'parent_revision:task:readout', 'meaning_version': 1}
    question = {'id': reference['question_id'], 'meaning_version': 1,
                'theory_target': {'parent_theory_id': 'parent195'}}
    old = {'id': 'readout', 'semantic_answer': {'status': 'answered', 'value': .25},
           'frozen_recall_execution': {'cursor': {'next_candidate_id': 'old65'}}}
    original = deepcopy(old)
    agent = SimpleNamespace(config=Limits(), costs={'compute_cost': 100,
        'python_execution_seconds': 0., 'observation_cost': 97, 'model_calls': 0},
        questions=[old, question], rows=[None]*97, candidate_paths=[], theory_index={},
        workspace=tmp_path)
    agent._evaluated_summary = lambda: {'model': 'actual_winner.py', 'mse': .01}
    units = [{'round': r, 'candidate_id': owner, 'candidate_ids': [owner],
              'search_position': {'layer': r-1, 'offset': 1}, 'compute': 20}
             for r in range(1, 5) for owner in ('old', 'new')]
    contract = {'rounds': 4, 'parent_theory_ref': {'id': 'parent195', 'version': 1},
                'complete_unit_compute_limits': {'keep': units}}
    descriptor = {'state': {'structures': []}, 'source_ref': {'file': 'old_grammar.json'}}
    inputs = {'parent_theory_id': 'parent195', '_grammar_preparation': descriptor,
              '_retained_components': [{'id': 'full_original_component'}]}
    scene = {'agent': agent, 'lab': object(), 'models': [f'model_{i}.py' for i in range(2847)],
        'research_question_ref': reference, 'registered_arm': 'keep',
        'registered_revision_inputs': inputs}
    calls = []

    def execute(action, args, lab):
        assert action == 'run' and args['research_question'] == {'id': question['id']}
        assert question['theory_target']['parent_theory_id'] == 'parent195'
        if args['operator'] == 'research_choice':
            assert len(args['models']) == 2847
            calls.append('compare')
            agent.costs['compute_cost'] += 1
            return {'status': 'completed', 'reused_score_count': 2847, 'new_scored_count': 0,
                    'evaluations': [{'path': 'must_stay_in_formal_result_only.py'}]}
        pytest.fail('fixed-parent units must use retained recall rather than full research routing')

    def recall(actual_agent, actual_question, args):
        assert actual_agent is agent and actual_question is question
        index = calls.count('recall')
        calls.append('recall')
        assert args['_grammar_preparation'] is descriptor
        assert args['_retained_components'] == inputs['_retained_components']
        assert args['allocation'] == {'compute': 20, 'comparison_rows': 97}
        agent.costs['compute_cost'] += 12
        unit = units[index]
        visit = {'candidate_id': 'unexpected' if wrong_owner else unit['candidate_id'],
            'search_position': unit['search_position'], 'status': 'complete_visit_executed'}
        return {'status': 'completed', 'complete_research_visits': [] if missing_unit else [visit]}

    agent._execute = execute
    monkeypatch.setattr(control.consolidation, 'recall_frozen_theories', recall)
    monkeypatch.setattr(control, '_checkpoint', lambda *args: None)
    monkeypatch.setattr(control, 'quote_joint_paid_scores', lambda scene: {
        'unpaid_scoring_compute': 0})
    monkeypatch.setattr('hegel_machine.cognition.finite_native_author.export_revision_candidates',
        lambda *args: {'new_exports': 0})
    return scene, {'parent_research': contract}, calls, old, original


def test_runner_executes_four_uniform_rounds_on_fixed_parent_and_preserves_full_pool(tmp_path, monkeypatch):
    scene, registration, calls, old, original = runner_scene(tmp_path, monkeypatch)
    result = control._arm_run(scene, registration)
    assert result['status'] == 'registered_units_completed'
    assert result['completed_units_by_group'] == {'old': 4, 'new': 4}
    assert calls == ['compare', *['recall', 'compare']*8]
    assert old == original
    assert result['continuation_costs']['compute_cost'] == 8*12+9
    actions = json.loads((tmp_path/'registered_actions.json').read_text())
    assert sum(x['action'] == 'frozen_recall_on_registered_parent_duty' for x in actions) == 8
    comparisons = [x for x in actions if x['action'] == 'research_choice']
    assert all('feedback' not in x for x in comparisons)
    for comparison in comparisons:
        assert len(json.loads(__import__('pathlib').Path(
            comparison['complete_model_roster_ref']['file']).read_text())) == 2847
        assert comparison['complete_paid_assessment_index_ref']['file'].endswith('assessments.json')


@pytest.mark.parametrize('initial', [True, False])
def test_partial_comparison_stops_without_counting_or_repeating_a_research_unit(
        tmp_path, monkeypatch, initial):
    scene, registration, calls, old, original = runner_scene(tmp_path, monkeypatch)
    execute = scene['agent']._execute

    def incomplete(action, args, lab):
        result = execute(action, args, lab)
        if initial or calls.count('compare') > 1:
            result.update(status='comparison_incomplete_python_budget', truncated=True,
                          models_completed=20, reason='python_execution_budget_exhausted')
        return result

    scene['agent']._execute = incomplete
    result = control._arm_run(scene, registration)
    assert result['status'] == 'comparison_incomplete_python_budget'
    assert result['completed_units_by_group'] == {}
    assert calls.count('recall') == (0 if initial else 1)
    assert old == original


@pytest.mark.parametrize('spent', [300., 314.9189407469239])
def test_exhausted_original_python_budget_does_not_run_either_final_readout(
        tmp_path, monkeypatch, spent):
    from test_registered_source_control import _fake_finished_control

    registration = _fake_finished_control(tmp_path, monkeypatch)
    accepted, acceptance = control.prepare_registered_control(registration, tmp_path)
    registration.update(parent_research={'question_id': 'actual_parent195'},
                        policy={'frozen_policy_files': []})
    failure = tmp_path / 'retained_actions.json'
    failure.write_text(json.dumps([{'action': 'execution_failure',
        'reason': 'original Python exhaustion', 'retained_fee': spent}]))
    original_failure = failure.read_bytes()

    def finished(scene, registration):
        return {'workspace': str(scene['workspace']),
            'status': 'comparison_incomplete_python_budget',
            'retained_parent': {'model': '_evaluated_model/model.py'},
            'actions_ref': str(failure), 'completed_units_by_group': {'old': 1},
            'continuation_costs': {'compute_cost': 123, 'python_execution_seconds': spent}}

    monkeypatch.setattr(control, '_arm_run', finished)
    monkeypatch.setattr('hegel_machine.cognition.question_queries.evaluate_on_model',
        lambda *args, **kwargs: pytest.fail('no Python allowance remains for a final readout'))
    summary = control.run_registered_control(registration, tmp_path, workers=8,
        accepted_scene=(accepted, acceptance))
    predictions = json.loads((tmp_path / 'frozen_predictions.json').read_text())
    assert summary['status'] == 'failed_or_partial'
    for arm in summary['arms']:
        assert arm['status'] == 'Python_budget_exhausted'
        assert arm['completed_units_by_group'] == {'old': 1}
        assert arm['continuation_costs'] == {
            'compute_cost': 123, 'python_execution_seconds': spent}
        assert arm['actions_ref'] == str(failure)
        result = predictions[arm['arm']]
        assert result['compute_cost'] == 0 and len(result['answers']) == 2
        assert all(answer == {'status': 'UNKNOWN',
            'reason': 'python_execution_budget_exhausted', 'compute_cost': 0}
            for answer in result['answers'])
    assert failure.read_bytes() == original_failure


@pytest.mark.parametrize('failure', ['wrong_owner', 'missing_unit', 'unfunded'])
def test_runner_records_deviation_or_unfunded_unit_without_appending_research(tmp_path, monkeypatch, failure):
    scene, registration, calls, old, original = runner_scene(tmp_path, monkeypatch,
        wrong_owner=failure == 'wrong_owner', missing_unit=failure == 'missing_unit')
    if failure == 'unfunded':
        scene['agent'].config = Limits(compute_budget=110)
    result = control._arm_run(scene, registration)
    expected = {'wrong_owner': 'research_failed', 'missing_unit': 'registered_unit_not_completed',
                'unfunded': 'research_budget_incomplete'}
    assert result['status'] == expected[failure]
    assert result['completed_units_by_group'] == {}
    assert calls.count('recall') == (0 if failure == 'unfunded' else 1)
    assert old == original


def test_repeated_position_is_charged_but_does_not_count_as_completed(tmp_path, monkeypatch):
    scene, registration, calls, _, _ = runner_scene(tmp_path, monkeypatch)
    units = registration['parent_research']['complete_unit_compute_limits']['keep']
    units[:] = [units[0], {**units[0], 'round': 2,
                          'search_position': {'layer': 1, 'offset': 1}}]
    recall = control.consolidation.recall_frozen_theories

    def repeated_position(*args):
        result = recall(*args)
        if calls.count('recall') == 2:
            result['complete_research_visits'][0]['search_position'] = {'layer': 0, 'offset': 1}
        return result

    monkeypatch.setattr(control.consolidation, 'recall_frozen_theories', repeated_position)
    result = control._arm_run(scene, registration)
    assert result['status'] == 'research_failed'
    assert result['completed_units_by_group'] == {'old': 1}
    assert calls == ['compare', 'recall', 'compare', 'recall', 'compare']
    assert result['continuation_costs']['compute_cost'] == 27


def test_registered_parent_CLI_uses_completed_scene_admission(tmp_path, monkeypatch):
    from test_registered_source_control import _fake_finished_control
    registration = _fake_finished_control(tmp_path, monkeypatch)
    accepted, receipt = control.prepare_registered_control(registration, tmp_path)
    registration.update(parent_research={'question_id': 'new_parent_use'},
                        policy={'frozen_policy_files': []})
    monkeypatch.setattr(control, 'prepare_registered_control',
        lambda *args: pytest.fail('a completed parent control must not restore an original source'))
    calls = []

    def completed(*args):
        calls.append('completed')
        return accepted, receipt

    monkeypatch.setattr(scenes, 'prepare_completed_source_control', completed)
    result = control.run_registered_control(registration, tmp_path, workers=8)
    assert result['status'] == 'completed' and calls == ['completed']
    assert all(scene['registered_arm'] == arm for arm, scene in accepted.items())


def test_unfunded_remaining_parent_plan_cannot_restore_or_start_science(tmp_path, monkeypatch):
    registration = {'parent_research': {}, 'budget': {'additional_Python_seconds_per_arm': 0.}}
    monkeypatch.setattr(scenes, 'prepare_completed_source_control',
        lambda *args: pytest.fail('unfunded continuation must not start restoring'))
    with pytest.raises(ValueError, match='no authorized Python execution budget'):
        control.run_registered_control(registration, tmp_path/'unfunded')
    assert not (tmp_path/'unfunded').exists()


def test_remaining_control_restores_each_own_ended_scene_without_resetting_consumed_units(
        tmp_path, monkeypatch):
    from hegel_machine.cognition.registered_source_budget import retained_four_round_progress

    unit = {'candidate_id': 'old', 'search_position': {'layer': 0, 'offset': 1}, 'compute': 10}
    original = tmp_path/'original.json'
    original.write_text(json.dumps({'parent_research': {
        'complete_unit_compute_limits': {'remove': [unit], 'keep': [unit]}}}))
    sources, counts, plans = {}, {}, {}
    for arm in ('remove', 'keep'):
        source = tmp_path/'ended'/arm
        source.mkdir(parents=True)
        actions = [{'action': 'frozen_recall_on_registered_parent_duty', 'unit_index': 0,
            'registered_unit': unit, 'feedback': {'complete_research_visits': [{
                'candidate_id': 'old', 'search_position': unit['search_position'],
                'status': 'complete_visit_executed'}]}}]
        (source/'registered_actions.json').write_text(json.dumps(actions))
        progress = retained_four_round_progress([unit], actions)
        sources[arm] = str(source)
        counts[arm], plans[arm] = progress['completed_unit_counts_by_group'], progress['remaining_units']
    registration = {'arms': {'remove': {}, 'keep': {}},
        'continuation': {'original_registration_ref': {'file': str(original), 'json_path': []}},
        'material': {'same_closed_scene_by_arm': sources},
        'parent_research': {'complete_unit_compute_limits': plans,
                           'completed_unit_counts_by_group': counts}}
    loaded = []
    monkeypatch.setattr(scenes, 'load_completed_source_material',
        lambda local: loaded.append(local['material']['same_closed_scene']) or {})
    monkeypatch.setattr(scenes, 'restore_completed_source_scene',
        lambda local, arm, output, material: {'validation': {'source': sources[arm]}})
    monkeypatch.setattr(scenes, 'prepare_registered_parent_revision',
        lambda scene, local: {'current_parent_scope_has_prior_executions': True})
    monkeypatch.setattr(scenes, 'quote_joint_paid_scores',
        lambda scene: {'models': 2847, 'reused_score_count': 2847, 'unpaid_score_count': 0})
    _, receipt = scenes.prepare_completed_source_control(registration, tmp_path/'resumed')
    assert loaded == [sources['remove'], sources['keep']]
    assert all(row['consumed_units'] == 1 and row['remaining_units'] == 0
               and row['comparison_of_committed_work_required'] for row in receipt['arms'].values())


def test_freeze_validation_uses_explicit_repo_paths_for_modules_and_CLI_scripts(tmp_path):
    root = Path(control.__file__).resolve().parents[3]
    names = ['src/hegel_machine/cognition/registered_source_control.py',
             'scripts/prepare_pr3_source_successor.py']
    files = []
    for name in names:
        frozen = tmp_path/'policy_at_registration'/name
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes((root/name).read_bytes())
        files.append({'file': str(frozen), 'relative_path': name})
    registration = {'policy': {'frozen_policy_files': files}}
    control._validate_registered_policy(registration)
    Path(files[-1]['file']).write_text('changed frozen entry point')
    with pytest.raises(ValueError, match='scripts/prepare_pr3_source_successor.py'):
        control._validate_registered_policy(registration)


def test_parent_runner_reuses_only_validated_frozen_source_files(tmp_path, monkeypatch):
    from test_registered_source_control import _fake_finished_control

    registration = _fake_finished_control(tmp_path, monkeypatch)
    scenes, acceptance = control.prepare_registered_control(registration, tmp_path)
    root = Path(control.__file__).resolve().parents[3]
    names = ['src/hegel_machine/cognition/registered_source_control.py',
             'scripts/prepare_pr3_source_successor.py']
    files = []
    for name in names:
        frozen = tmp_path/'policy_at_registration'/name
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes((root/name).read_bytes())
        files.append({'file': str(frozen), 'relative_path': name})
    registration.update(parent_research={'question_id': 'registered_parent195'},
                        policy={'frozen_policy_files': files})
    copied = []
    monkeypatch.setattr(control, 'clone_or_copy',
        lambda source, destination: copied.append((Path(source), Path(destination))))
    result = control.run_registered_control(registration, tmp_path, workers=8,
        accepted_scene=(scenes, acceptance))
    assert result['status'] == 'completed'
    assert copied == [(Path(item['file']), tmp_path/'source_at_run'/item['relative_path'])
                      for item in files]


@pytest.mark.parametrize('registered_estimate', [0, 17])
def test_runner_carries_actual_restoration_acceptance_once_without_rewriting_registration(
        tmp_path, monkeypatch, registered_estimate):
    from hegel_machine.cognition import research_episodes
    from test_registered_source_control import _fake_finished_control

    registration = _fake_finished_control(tmp_path, monkeypatch)
    scenes, acceptance = control.prepare_registered_control(registration, tmp_path)
    acceptance['metadata_acceptance_compute'] = 17
    prior = {'known_source_and_development_compute_once': 50,
        'known_scientific_compute_including_sources_once': 100,
        'known_compute_including_engineering_acceptance': 200,
        'known_billed_Python_seconds': 0., 'other_D3_costs': 'unknown'}
    registration['costs'] = {'old_sources_and_all_failures_ref': 'prior.json',
        'new_mainline_induction_once': {'compute_cost': 0, 'python_execution_seconds': 0.},
        'restoration_acceptance_compute_once': registered_estimate}
    original = deepcopy(registration)
    load = research_episodes.load_source_values
    monkeypatch.setattr(research_episodes, 'load_source_values', lambda refs, workspace:
        [{'value': prior}] if refs[0].get('file') == 'prior.json' else load(refs, workspace))
    result = control.run_registered_control(registration, tmp_path, workers=8,
        accepted_scene=(scenes, acceptance))
    assert result['restoration_acceptance_compute_once'] == 17
    assert result['known_scientific_compute_including_sources_once'] == 104
    assert result['known_compute_including_engineering_acceptance'] == 221
    assert registration == original
