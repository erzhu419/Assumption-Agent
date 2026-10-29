"""Production factories stay inert until admission and reuse selected restore."""

import json
from types import SimpleNamespace

import pytest

from scripts.prepare_pr3_source_successor import (
    make_retained_method_pairing_factories, retained_method_use_continuation,
    quote_retained_method_use, retained_method_use_question, _reserve_registered_method_closure)
from hegel_machine.cognition import registered_source_scene as scenes


def test_factory_construction_does_not_restore_acquire_or_bind_workers(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('building the factory must not execute a retained scene')
    monkeypatch.setattr(scenes, 'restore_retained_common_scene', forbidden)
    callbacks = make_retained_method_pairing_factories(
        {'source_scene': {}, 'execution_authorized': False}, tmp_path)
    assert set(callbacks) == {'restore_cell', 'prepare_evaluation', 'evaluate_frozen', 'ordinary_use'}
    with pytest.raises(ValueError, match='admitted frozen registration'):
        callbacks['restore_cell']('00', {}, tmp_path / '00')
    with pytest.raises(ValueError, match='admitted frozen registration'):
        callbacks['prepare_evaluation']()
    assert not (tmp_path / '00').exists()


def test_actual_factory_resolves_archive_and_enters_restore_with_original_paid_account(tmp_path, monkeypatch):
    from hegel_machine.cognition import experiment_runtime

    costs = {'compute_cost': 8, 'observation_cost': 1, 'measurement_cost': 0,
        'python_execution_seconds': 2., 'model_calls': 0, 'reported_tokens': 0}
    def saved(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return {'file': str(path), 'json_path': []}
    training = {'file': str(tmp_path / 'old/training.json'), 'json_path': []}
    data = {**training, 'json_path': ['paid_observations']}
    source = {'case_id': 'case_002', 'training_ref': training, 'paid_observations_ref': data,
        'costs_ref': saved('costs.json', costs),
        'plan_ref': saved('plan.json', {'limits': {}, 'cases': [{'id': 'case_002'}]}),
        'initial_archive_ref': saved('archive.json', {'files': {}, 'paid_fact': 'retained'})}
    contract = {'start_state_ref': training, 'data_ref': data, 'compute_cap': 20,
        'observation_cap': 2, 'time_limits': {'billed_python_seconds': 30.}}
    class RestoreEntered(RuntimeError):
        pass
    def restore(source_path, workspace, **kwargs):
        assert kwargs['knowledge']['paid_fact'] == 'retained'
        assert kwargs['starting_costs'] == costs
        assert kwargs['limits'].compute_budget == 28
        assert kwargs['limits'].observation_budget == 3
        assert kwargs['limits'].execution_seconds == 32.
        assert kwargs['limits'].max_model_calls == 0
        raise RestoreEntered('production factory reached existing restore; no science')
    monkeypatch.setattr(scenes, 'restore_retained_common_scene', restore)
    monkeypatch.setattr(experiment_runtime, '_initialize_worker', lambda _: None)
    monkeypatch.setattr(experiment_runtime, 'physical_cpu_groups', lambda *_: [[0]])
    callbacks = make_retained_method_pairing_factories({'source_scene': source,
        'execution_authorized': True, 'approval_ref': {'file': 'approved.json'},
        'maximum_physical_cores': 1}, tmp_path / 'output')
    with pytest.raises(RestoreEntered, match='production factory reached existing restore'):
        callbacks['restore_cell']('00', contract, tmp_path / 'output/00')


def test_opaque_old_winner_stays_unknown_without_refit_or_weaker_substitute(tmp_path, monkeypatch):
    from hegel_machine.cognition import code_agent
    monkeypatch.setattr(code_agent, '_closed_temporal_export', lambda *args: False)
    callbacks = make_retained_method_pairing_factories({'source_scene': {},
        'execution_authorized': True, 'approval_ref': {'file': 'approved.json'}}, tmp_path)
    account = SimpleNamespace(costs={'compute_cost': 7, 'python_execution_seconds': 3.})
    result = callbacks['evaluate_frozen']('00', {'workspace': str(tmp_path), 'model': 'old.py'}, {}, account)
    assert result['prediction_status'] == 'UNKNOWN' and result['loss'] is None
    assert account.costs == {'compute_cost': 7, 'python_execution_seconds': 3.}


def test_retained_projection_selects_checkpoint_fields_and_borrows_one_index(tmp_path, monkeypatch):
    costs = {'observation_cost': 1, 'measurement_cost': 0, 'compute_cost': 11,
        'python_execution_seconds': 2., 'model_calls': 0, 'reported_tokens': 0}
    saved = {'costs': costs, 'inherited_files': ['model.py'], 'events': [],
        'online_questions': [{'id': 'q', 'native_condition_research': []}],
        'revision_proposals': [], 'theory_index': {'retained': {'version': 3}},
        'outgoing_knowledge': {'manifest': {'hypotheses': []},
            'files': {'must_not_decode_portable_archive': 'x' * 100000}}}
    (tmp_path / 'training.json').write_text(json.dumps(saved))
    (tmp_path / 'research_state.json').write_text(json.dumps({'costs': costs,
        'operator_results': [], 'exported_candidate_paths': [], 'online_research': {}}))
    original = scenes.read_json_checkpoint
    requested = []
    def selected(path, *, keys=None):
        assert keys is not None, 'do not decode the complete portable training checkpoint'
        requested.append((str(path), set(keys)))
        return original(path, keys=keys)
    monkeypatch.setattr(scenes, 'read_json_checkpoint', selected)
    previous, state = scenes.load_retained_common_checkpoint(tmp_path)
    assert previous['theory_index'] is state['theory_index']
    assert previous['online_questions'] is state['questions']
    assert previous['costs'] == costs
    assert previous['outgoing_knowledge']['manifest'] == {'hypotheses': []}
    assert 'must_not_decode_portable_archive' not in previous['outgoing_knowledge']['files']
    assert all('outgoing_knowledge' not in keys for _, keys in requested)


def test_ordinary_use_continues_paid_cell_instead_of_rebuying_its_query(tmp_path):
    ended = tmp_path / '10' / 'case_002' / 'pr3' / 'training.json'
    state_ref = {'file': str(ended), 'json_path': []}
    costs = {'observation_cost': 169, 'compute_cost': 1234567}
    binding = {'candidate_id': 'P1', 'candidate_version': 1,
        'origin_ref': {'file': str(ended.parent / 'research_state.json'), 'json_path': ['method_results', 7]},
        'paid_cutoff': {'records': 10, 'rows': 145}}
    outcome = tmp_path / '10' / 'outcome.json'
    outcome.parent.mkdir(parents=True)
    outcome.write_text(json.dumps({'execution_complete': True,
        'continuation_state_ref': state_ref,
        'method_artifact_ref': {'file': str(outcome.parent / 'shared/theory_library.json'),
            'json_path': ['candidates', 'P1']},
        'method_invocation_binding': binding,
        'continuation_scope': {'paid_rows': 169, 'pool_models': 4890},
        'costs_after': costs}))
    source = {'training_ref': {'file': 'old145/training.json', 'json_path': []},
        'costs_ref': {'file': 'old145/summary.json'}, 'unknown_cost_ref': {'file': 'old_unknown.json'}}
    use = {'continue_cell': '10', 'compute_cap': 999,
        'scope': {'paid_rows': 145, 'pool_models': 4762, 'max_exports': 246},
        'continuation_scope_limits': {'max_paid_rows': 169, 'max_pool_models': 5008}}
    rebound = retained_method_use_continuation(source, use,
        {'10': {'file': str(outcome), 'json_path': []}})
    assert rebound['source']['training_ref'] == state_ref
    assert rebound['source']['paid_observations_ref'] == {**state_ref, 'json_path': ['paid_observations']}
    assert rebound['source']['costs_ref'] == {'file': str(outcome), 'json_path': ['costs_after']}
    assert rebound['source']['unknown_cost_ref'] == source['unknown_cost_ref']
    assert rebound['contract']['scope'] == {'paid_rows': 169, 'pool_models': 4890, 'max_exports': 246}
    assert rebound['contract']['compute_cap'] == 999
    assert rebound['tested_candidate_ref']['json_path'] == ['candidates', 'P1']
    assert rebound['method_invocation_binding'] == binding
    assert use['scope']['paid_rows'] == 145 and source['training_ref']['file'].startswith('old145')
    with pytest.raises(ValueError, match='exceeds the registered'):
        retained_method_use_continuation(source,
            {**use, 'continuation_scope_limits': {'max_paid_rows': 145, 'max_pool_models': 5008}},
            {'10': {'file': str(outcome), 'json_path': []}})
    with pytest.raises(ValueError, match='explicitly registered'):
        retained_method_use_continuation(source, {**use, 'continue_cell': None}, {})


def test_method_problem_dispatch_uses_actual_invocation_and_restored_parent():
    from hegel_machine.cognition.question_agenda import has_method_use_plan
    candidate = {'id': 'v8_case002_P1', 'version': 1}
    binding = {'candidate_id': candidate['id'], 'candidate_version': candidate['version'],
        'origin_ref': {'file': 'cell10/research_state.json', 'json_path': ['method_results', 7]},
        'paid_cutoff': {'records': 10, 'rows': 145}}
    use = {'research_question': {'id': 'method_problem', 'question': 'Which research action is useful?',
        'parent_id': 'old_parent', 'method_use_plan': {'candidate_id': candidate['id'],
            'binding_from_outcome': {'cell': '10', 'field': 'method_invocation_binding'}}}}
    assert not has_method_use_plan(use['research_question']), 'a preparation reference is not a paid binding'
    actual = retained_method_use_question(use, candidate, binding, {'id': 'real_parent', 'meaning_version': 3})
    assert has_method_use_plan(actual)
    assert actual['method_use_plan'] == {'candidate_id': candidate['id'], 'binding': binding}
    assert actual['parent_question_ref'] == {'question_id': 'real_parent', 'meaning_version': 3}
    assert actual['parent_id'] == 'real_parent'
    assert 'binding' not in use['research_question']['method_use_plan']
    with pytest.raises(ValueError, match='actual executed candidate binding'):
        retained_method_use_question(use, candidate, {**binding, 'candidate_version': 2}, {'id': 'real_parent'})
    with pytest.raises(ValueError, match='actual executed candidate binding'):
        retained_method_use_question(use, candidate, {}, {'id': 'real_parent'})


def test_next_state_quote_uses_live_paid_selection_and_reserves_its_own_closure(tmp_path, monkeypatch):
    from hegel_machine.cognition import finite_native_author as finite, method_feedback
    from hegel_machine.cognition.research_programs import export_research_selection_program
    proposal = {'action': 'run', 'arguments': {'operator': 'native_condition_research',
        'question': {'id': 'q'}, 'max_compute': 60}}
    state = {'questions': [{'id': 'q'}], 'costs': {'compute_cost': 23}}
    plan = {'ordinary_method_path': 'ordinary.py'}
    (tmp_path / 'ordinary.py').write_text("""
def select_actions(public, observations, memory):
    assert _state(memory) is memory['research_state']
    return memory['actual_next_proposal']
def select_research_segment(public, observations, memory, *, plan, ordinary_select_actions, condition=None):
    return ordinary_select_actions(public, observations, memory)
""")
    (tmp_path / 'P1.py').write_text(export_research_selection_program(plan))
    # A file-backed state is present, but the projection must use the already
    # restored live view rather than decode a second large archive.
    (tmp_path / 'research_state.json').write_text('not to be decoded')
    agent = SimpleNamespace(workspace=tmp_path, _program_state=lambda: state,
        program_contexts={}, method_results=[], rows=[object()] * 169, observations=[],
        knowledge={'finite_native_author': {'predictor_paths': ['old.py', 'paid.py']},
            'actual_next_proposal': proposal}, costs={'compute_cost': 23})
    option = {'family': 'N', 'proposal': proposal,
        'semantic_action': finite.research_action_semantics(proposal),
        'proposal_ref': {'file': 'method_decision.json', 'json_path': ['fresh']},
        'complete_quote': {'status': 'available', 'compute_cost': 100, 'observation_cost': 0,
            'paid_rows': 169, 'retained_pool_models': 2, 'max_exported_models': 1,
            'research_unit': {'completion_reserve': {'compute_cost': 40}}}}
    monkeypatch.setattr(finite, 'publish_research_options', lambda *_: [{'fresh': True}])
    def current_options(public, observations, memory, **kwargs):
        assert kwargs['state'] is state
        assert kwargs['decision']['research_options'] == [{'fresh': True}]
        return {'options': [option], 'excluded': []}
    monkeypatch.setattr(finite, 'research_segment_options', current_options)
    monkeypatch.setattr(method_feedback, 'begin_method_feedback', lambda *_args, **_kwargs: {'profile': {}})
    candidate = {'selection_plan': plan, 'execution': {'path': 'P1.py'}}
    quoted = quote_retained_method_use(agent, {}, candidate, compute_cap=100)
    assert quoted['status'] == 'available'
    assert quoted['closure_reserve_compute'] == 40 and quoted['required_compute'] == 100
    assert quoted['scope']['paid_rows'] == 169 and quoted['ordinary_proposal'] == proposal
    assert agent.costs == {'compute_cost': 23}
    assert quote_retained_method_use(agent, {}, candidate, compute_cap=99)['status'] == 'complete_unit_unfunded'
    option['complete_quote']['observation_cost'] = 24
    blocked = quote_retained_method_use(agent, {}, candidate, compute_cap=100)
    assert blocked['status'] == 'UNKNOWN'
    assert blocked['reason'] == 'next_work_requires_unapproved_new_observations'


def test_incomplete_cell_returns_unresolved_without_restoration_or_research(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail('a failed paid cell must not be cold-restored for another attempt')
    monkeypatch.setattr(scenes, 'restore_retained_common_scene', forbidden)
    failed = tmp_path / 'outcome.json'
    failed.write_text(json.dumps({'execution_complete': False, 'status': 'time_exhausted'}))
    callbacks = make_retained_method_pairing_factories({
        'source_scene': {'unknown_cost_ref': {'file': 'old_unknown.json'}},
        'execution_authorized': True, 'approval_ref': {'file': 'approved.json'},
        'ordinary_use': {'continue_cell': '10'}}, tmp_path)
    result = callbacks['ordinary_use']([], {'10': {'file': str(failed), 'json_path': []}}, tmp_path)
    assert result['status'] == 'ordinary_use_unresolved_incomplete_cell'
    assert result['new_compute_cost'] == result['new_observations'] == 0
    assert (tmp_path / 'ordinary_use_receipt.json').is_file()


def test_ordinary_quote_keeps_registered_common_closure_without_double_counting():
    fresh = {'status': 'available', 'required_compute': 100, 'closure_reserve_compute': 40}
    adjusted = _reserve_registered_method_closure(fresh,
        {'closure_compute_reserved': 70}, compute_cap=130)
    assert adjusted['status'] == 'available'
    assert adjusted['required_compute'] == 130 and adjusted['closure_reserve_compute'] == 70
    assert adjusted['fresh_closure_reserve_compute'] == 40
    assert fresh == {'status': 'available', 'required_compute': 100, 'closure_reserve_compute': 40}
    unchanged = _reserve_registered_method_closure(fresh,
        {'closure_compute_reserved': 30}, compute_cap=100)
    assert unchanged['required_compute'] == 100 and unchanged['closure_reserve_compute'] == 40
    assert _reserve_registered_method_closure(fresh, {}, compute_cap=100) is fresh


def test_ordinary_quote_rejects_work_when_registered_closure_exceeds_actual_balance():
    adjusted = _reserve_registered_method_closure(
        {'status': 'available', 'required_compute': 100, 'closure_reserve_compute': 40},
        {'closure_compute_reserved': 70}, compute_cap=129)
    assert adjusted['status'] == 'complete_unit_unfunded'
    assert adjusted['required_compute'] == 130 and adjusted['closure_reserve_compute'] == 70


def test_unknown_ordinary_quote_does_not_turn_missing_costs_into_zero():
    unknown = {'status': 'UNKNOWN', 'reason': 'fresh_selected_work_has_no_complete_paid_only_quote'}
    assert _reserve_registered_method_closure(unknown,
        {'closure_compute_reserved': 70}, compute_cap=100) is unknown
    incomplete = _reserve_registered_method_closure(
        {'status': 'available', 'required_compute': 100},
        {'closure_compute_reserved': 70}, compute_cap=200)
    assert incomplete['status'] == 'UNKNOWN'
    assert incomplete['reason'] == 'complete_closure_quote_required'
    assert 'closure_reserve_compute' not in incomplete


@pytest.mark.parametrize('rows,reserved,reason', [(2, 3, 'actions differ'), (3, 2, 'acquisition reserve')])
def test_qeval_row_contract_checked_before_any_purchase(tmp_path, monkeypatch, rows, reserved, reason):
    from scripts import run_anonymous_revision_inquiry as study
    calls = []
    class PublicOnlyLab:
        def __init__(self, **kwargs):
            pass
        action_cost = staticmethod(lambda action: len(action['sequence']))
        def execute(self, action):
            calls.append(action)
            pytest.fail('reject inconsistent shared Qeval before acquiring its first row')
    monkeypatch.setattr(study, 'AnonymousHistoryLab', PublicOnlyLab)
    plan, evaluation = tmp_path / 'plan.json', tmp_path / 'evaluation.json'
    plan.write_text(json.dumps({'cases': [{'id': 'case_002', 'kind': 'fixture', 'seed': 1}]}))
    evaluation.write_text(json.dumps({'rows': rows,
        'actions': [{'kind': 'trajectory', 'sequence': [{'u': 0.}] * 3}]}))
    (tmp_path / 'frozen_outputs.json').write_text('{}')
    callbacks = make_retained_method_pairing_factories({
        'source_scene': {'plan_ref': {'file': str(plan)}, 'case_id': 'case_002'},
        'execution_authorized': True, 'approval_ref': {'file': 'approved.json'},
        'components': {'Qeval_predictions': {'observations': reserved}},
        'cells': {'00': {'evaluation_ref': {'file': str(evaluation)}}}}, tmp_path)
    with pytest.raises(ValueError, match=reason):
        callbacks['prepare_evaluation']()
    assert calls == []
