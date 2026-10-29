"""A completed comparison remains the authoritative starting scene."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hegel_machine.cognition import registered_source_scene as scene
from hegel_machine.cognition import question_routing, theory_consolidation
from scripts import prepare_pr3_source_successor as prepare
from scripts import run_anonymous_revision_inquiry as study


OLD = 'consolidated:18:relation:revision:0'
NEW = 'consolidated:19:relation:revision:0'
PREFIX = 'relation_transfer:frozen_recall:native_public_response@1:'


def completed_fixture(tmp_path):
    source = tmp_path / 'completed' / 'keep'
    source.mkdir(parents=True)
    reference = lambda name: {'file': str(source / name), 'json_path': []}
    costs = {'observation_cost': 2, 'measurement_cost': 0, 'compute_cost': 100,
             'python_execution_seconds': 3., 'model_calls': 0, 'reported_tokens': 0}
    starting = {**costs, 'compute_cost': 80, 'python_execution_seconds': 2.}
    final_delta = {**costs, 'observation_cost': 0, 'compute_cost': 22,
                   'python_execution_seconds': 1.25}
    paid = {**costs, 'compute_cost': 102, 'python_execution_seconds': 3.25}
    parent = {'id': 'parent', 'version': 2, 'program': {'op': 'var', 'name': 'u'},
              'role_signature': {'u': {'role': 'input', 'unit': '1', 'time': 'current'}},
              'output_unit': '1', 'context_ref': {'id': 'same-paid-scene'}}
    old_component = {'id': 'component', 'version': 1,
                     'program': {'op': 'const', 'value': 1.}}
    pool = {'parent': parent, 'opaque_python': {'id': 'opaque_python', 'version': 1,
            'model': 'strong.py'}, 'component': {**deepcopy(old_component), 'version': 2,
            'history': [deepcopy(old_component)]},
            'failed_candidate': {'id': 'failed_candidate', 'version': 1,
                                 'status': 'refuted', 'counterexamples': [{'record': 1}]}}
    old_candidates = {**{f'old{i}': {'id': f'old{i}', 'version': 1} for i in range(18)},
                      OLD: {'id': OLD, 'version': 1}}
    candidates = {**old_candidates, NEW: {'id': NEW, 'version': 1,
                                         'parent_reference_relations': [{'id': 'learned'}]}}
    execution = {'request_basis': {'candidate_id': OLD, 'candidate_version': 1,
        'parent_theory_id': 'parent', 'component_theory_ids': ['component'],
        'components': [old_component], 'fitting': {'operations': ['refit']},
        'evidence_checks': [], 'allocation': {'compute': 150, 'comparison_rows': 2}},
        'cursor': {'layer': 3, 'offset': 8,
                   'pending_fits': [{'indices': [index], 'position': {'layer': 3, 'offset': index + 5}}
                                    for index in range(4)]},
        'result': {'status': 'UNKNOWN', 'new_compute_cost': 17,
                   'failed_fits': [{'reason': 'original-failure', 'compute_cost': 2}]}}
    library = {'candidates': deepcopy(old_candidates),
        'executions': {PREFIX + OLD: execution},
        'transfers': [{'theory_id': 'failed_candidate', 'status': 'refuted', 'compute_cost': 2}],
        'material_cursors': {'source': 5}, 'conflicts': [{'source': 'retained-counterexample'}]}
    previous = {'costs': costs, 'paid_observations': [{'rows': [1, 2]}],
        'evaluated_model': {'source_model': 'winner.py', 'mse': .1},
        'theory_index': deepcopy(pool), 'events': [{'feedback': {'operator': 'research_choice',
            'evaluations': [{'model': 'winner.py'}, {'model': 'strong.py'}]}}]}
    question = {'id': 'native_public_response', 'meaning_version': 1,
        'frozen_library_path': scene.LIBRARY_PATH,
        'semantic_answer': {'status': 'answered', 'value': .25, 'evidence': [0]},
        'use_sites': [{'kind': 'choose_parent_experiment', 'parent_id': 'parent'}],
        'theory_target': {'parent_theory_id': 'parent'},
        'resolution_experiments': [{'query': {'kind': 'point', 'input': .2}},
                                   {'query': {'kind': 'point', 'input': .8}}]}
    previous['online_questions'] = [question]
    case = {'id': 'case_002', 'kind': 'hysteresis', 'seed': 17, 'confirmation_seed': 29}
    plan = {'cases': [case], 'limits': {}, 'research_organization': {'phase': 'source'},
            'prepared_public_questions': {'case_002': deepcopy(question)}}
    registration = {'material': {'same_closed_scene': str(source), 'paid_rows': 2,
        'training_ref': reference('training.json'),
        'research_state_ref': reference('research_state.json'),
        'new_object_ref': reference('new_object.json'),
        'closed_scene_theory_count': len(pool), 'closed_scene_scored_models': 2},
        'arms': {arm: {'library_ref': reference(arm + '.json'),
            'candidate_versions': [{'id': identifier, 'version': 1} for identifier in declared]}
                 for arm, declared in [('remove', old_candidates), ('keep', candidates)]},
        'budget': {'additional_compute_per_arm': 500,
                   'additional_Python_seconds_per_arm': 30., 'per_consumption_compute': 150},
        'restoration_admission': {'old19_original_pending_positions': ['3:5', '3:6', '3:7', '3:8']},
        'costs': {'new_mainline_induction_once': {'compute_cost': 7}}}
    for name, value in [('training.json', previous), ('final_costs.json', final_delta),
                        ('arm_summary.json', {'starting_costs': starting}),
                        ('remove.json', {'candidates': old_candidates}),
                        ('keep.json', {'candidates': candidates}), ('new_object.json', candidates[NEW])]:
        (source / name).write_text(json.dumps(value))
    material = {'source': source, 'previous': previous, 'plan': plan, 'case': case,
        'knowledge': {'finite_native_author': {'question': deepcopy(question)}, 'files': {}},
        'models': ['strong.py'], 'frozen_candidates': candidates,
        'new_candidate': deepcopy(candidates[NEW]), 'retained_training_costs': deepcopy(costs),
        'starting_costs': paid, 'source_registration': {'arms': {
            'remove': {'candidate_versions': [{'id': key, 'version': 1}
                         for key in old_candidates if key != OLD]},
            'keep': {'candidate_versions': [{'id': key, 'version': 1}
                       for key in old_candidates]}}}}
    origin = tmp_path / 'original' / 'case_002' / 'pr3'
    origin.mkdir(parents=True)
    material['source_registration']['material'] = {
        'source_training_ref': {'file': str(origin / 'training.json'), 'json_path': []},
        'common_executable_model_roster_ref': reference('models.json')}
    constructor = {name: reference(name + '.json') for name in
                   ('pattern_ref', 'holes_ref', 'components_ref')}
    parent_ref = reference('native_parent.json')
    constructor['program_ref'] = parent_ref
    parent['program_source_ref'] = deepcopy(parent_ref)
    pool['parent']['program_source_ref'] = deepcopy(parent_ref)
    material['source_registration']['material'].update(
        parent_constructor_source=constructor, parent_Program_ref=parent_ref)
    material['parent_contract'] = {'source_ref': deepcopy(constructor),
        '_pattern': deepcopy(parent['program']), '_holes': [], '_bound_holes': {}}
    library['executions'][PREFIX + OLD]['request_basis']['parent'] = {
        'id': 'parent', 'version': 2, 'source_ref': None, 'source_status': None,
        'contract_source': deepcopy(constructor)}
    for name, value in [('pattern_ref.json', parent['program']), ('holes_ref.json', []),
                        ('components_ref.json', {})]:
        (source / name).write_text(json.dumps(value))
    (source / 'models.json').write_text(json.dumps(material['models']))
    (source.parent / 'registration.json').write_text(json.dumps(material['source_registration']))
    (source.parent / 'summary.json').write_text(json.dumps({'arms': [
        {'workspace': str(source), 'starting_costs': starting}]}))
    (origin.parent.parent / 'protocol.json').write_text(json.dumps(plan))
    (origin.parent.parent / 'initial_archive.json').write_text(json.dumps(material['knowledge']))
    state = {**deepcopy(previous), 'questions': deepcopy(previous['online_questions']),
        'revisions': [], 'theory_index': deepcopy(pool),
        'operator_results': [{'operator': 'research_choice', 'evaluations': [
            {'path': 'strong.py', 'paid_residuals': {'examples': [{'row': 1}]}}]}]}
    (source / 'research_state.json').write_text(json.dumps(state))
    material['previous'] = deepcopy(previous)
    material['previous']['costs'] = deepcopy(paid)
    return registration, material, pool, library


def stub_restoration(monkeypatch, material, pool, library, *, restore_source=None,
                     retain_declared_contract=False):
    """Keep the real availability/ledger decisions, replacing only scene I/O."""
    agents = []

    class Agent:
        def __init__(self, provider, workspace, **kwargs):
            assert provider is None and kwargs['arm'] == 'pr3'
            self.workspace = Path(workspace)
            self.workspace.mkdir(parents=True)
            self.config, self.knowledge = kwargs['config'], kwargs['knowledge']
            self._organization_state = {}
            self.files = {}
            agents.append(self)

        def restore_research(self, previous, source, laboratory, **kwargs):
            assert source == (restore_source or material['source'])
            self.retained_state = kwargs.get('retained_state')
            self.costs = deepcopy(previous['costs'])
            self.observations = deepcopy(previous['paid_observations'])
            self.rows = [1, 2]
            self.questions = deepcopy(previous['online_questions'])
            self.theory_index = deepcopy(pool)
            self.theory_consolidation = deepcopy(library)
            self.evaluated_model = {'source_model': 'winner.py', 'current_mse': .1}
            self.candidate_paths = ['winner.py']
            for filename in [*material['models'], *self.candidate_paths]:
                (self.workspace / filename).write_text('# retained model; do not execute\n')
            return {'restored_paid_rows': 2, 'new_observations': 0}

        def _prepare_program_state(self):
            pass

        def _execute(self, *args, **kwargs):
            pytest.fail('restoration must not execute a fit, score or experiment')

        def _evaluated_summary(self):
            return {'source_model': 'winner.py', 'mse': .1}

    old_component = deepcopy(pool['component']['history'][0])
    parent = deepcopy(pool['parent'])
    if retain_declared_contract:
        parent['_contract_source'] = deepcopy(material['parent_contract'])
    inputs = {'_retained_parent': parent, '_retained_components': [pool['component']],
        'component_theory_ids': ['component'], '_retained_execution_inputs': {OLD: {
            '_retained_parent': parent, '_retained_components': [old_component],
            'component_theory_ids': ['component']}}}
    monkeypatch.setattr(study, 'RegisteredOrganizationAgent', Agent)
    monkeypatch.setattr(study, 'AnonymousHistoryLab', lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(scene, 'archived_paid_trace_references', lambda *args, **kwargs: [])
    monkeypatch.setattr(scene, 'save_theory_library', lambda agent: None)
    monkeypatch.setattr(scene, 'save_theory_index', lambda agent: None)
    monkeypatch.setattr(scene, 'resolved_theories', lambda agent: list(agent.theory_index.values()))
    monkeypatch.setattr(question_routing, '_retained_revision_inputs',
                        lambda *args: deepcopy(inputs))
    monkeypatch.setattr(prepare, '_retain_scene_score_index', lambda *args, paid_rows:
                        {'retained_scopes': 1, 'retained_assessments': 2, 'available_paid_rows': paid_rows})
    monkeypatch.setattr(theory_consolidation, '_same_relation_work', lambda *args: True)
    monkeypatch.setattr(theory_consolidation, '_retained_parent_contract',
                        lambda agent, parent: deepcopy(parent))
    return agents


@pytest.mark.parametrize('arm', ['remove', 'keep'])
def test_restore_completed_scene_keeps_all_local_work_and_readout_fees(
        tmp_path, monkeypatch, arm):
    registration, material, pool, library = completed_fixture(tmp_path)
    stub_restoration(monkeypatch, material, pool, library)
    unchanged_material = deepcopy(material)
    result = scene.restore_completed_source_scene(
        registration, arm, tmp_path / arm, material=material)
    agent = result['agent']

    # Saved readouts were already paid; induction7 is a separate once-only bill.
    assert agent.costs == material['starting_costs']
    assert agent.costs['compute_cost'] == 102
    assert agent.costs['python_execution_seconds'] == 3.25
    assert agent.config.compute_budget == 602
    assert agent.config.execution_seconds == 33.25
    assert agent.config.max_model_calls == 0
    assert agent.config.token_budget == 0
    assert agent.config.observation_budget == 2
    assert agent.theory_index == pool
    for key in ('executions', 'transfers', 'material_cursors', 'conflicts'):
        assert agent.theory_consolidation[key] == library[key]
    assert PREFIX + NEW not in agent.theory_consolidation['executions']
    assert agent.theory_consolidation['executions'][PREFIX + OLD]['cursor'] == {
        'layer': 3, 'offset': 8,
        'pending_fits': [{'indices': [index], 'position': {'layer': 3, 'offset': index + 5}}
                         for index in range(4)]}
    expected_ids = [item['id'] for item in registration['arms'][arm]['candidate_versions']]
    assert list(agent.theory_consolidation['candidates']) == expected_ids
    assert (NEW in agent.theory_consolidation['candidates']) == (arm == 'keep')
    assert agent.questions[0]['semantic_answer'] == material['previous'][
        'online_questions'][0]['semantic_answer']
    assert agent.questions[0]['use_sites'] == [{'kind': 'choose_parent_experiment',
                                              'parent_id': 'parent'}]
    assert result['retained_components'][0]['version'] == 1
    assert agent.theory_index['component']['version'] == 2
    assert result['source_component_theory_ids'] == ['component']
    assert result['models'] == ['strong.py', 'winner.py']
    assert result['transfer_arguments']['candidate_id'] == NEW
    assert result['validation']['new_fits'] == 0
    assert result['validation']['new_predictions'] == 0
    assert result['validation']['new_observations'] == 0
    assert material == unchanged_material


def test_completed_restoration_reads_state_once_and_transfers_owned_graph_with_paid_tail(
        tmp_path, monkeypatch):
    registration, material, pool, library = completed_fixture(tmp_path)
    stub_restoration(monkeypatch, material, pool, library)
    state_file = material['source'] / 'research_state.json'
    original_bytes = state_file.read_bytes()
    read = scene.read_json_checkpoint
    decoded = []

    def read_once(path):
        assert Path(path) == state_file
        assert not decoded, 'restoration decoded the same full state twice'
        decoded.append(read(path))
        return decoded[0]

    agent_class = study.RegisteredOrganizationAgent
    restore = agent_class.restore_research

    def adopt(self, previous, source, laboratory, **kwargs):
        state = kwargs['retained_state']
        assert state is decoded[0]
        assert previous['online_questions'] is state['questions']
        assert previous['theory_index'] is state['theory_index']
        result = restore(self, previous, source, laboratory, **kwargs)
        self.questions = state['questions']
        self.theory_index = state['theory_index']
        self.operator_results = state['operator_results']
        return result

    monkeypatch.setattr(scene, 'read_json_checkpoint', read_once)
    monkeypatch.setattr(agent_class, 'restore_research', adopt)
    result = scene.restore_completed_source_scene(registration, 'keep', tmp_path / 'keep',
                                                   material=material)
    agent = result['agent']
    assert len(decoded) == 1
    assert agent.questions is decoded[0]['questions']
    assert agent.theory_index is decoded[0]['theory_index']
    assert agent.operator_results is decoded[0]['operator_results']
    assert agent.costs == decoded[0]['costs'] == material['starting_costs']
    assert result['validation']['post_checkpoint_fees_carried']['compute_cost'] == 2
    assert result['validation']['post_checkpoint_fees_carried']['python_execution_seconds'] == .25
    assert 'authoritative_scientific_checkpoint' not in result['validation']
    assert material.get('retained_state') is None
    assert state_file.read_bytes() == original_bytes


def test_already_admitted_interrupted_state_is_not_read_again(tmp_path, monkeypatch):
    registration, material, pool, library = completed_fixture(tmp_path)
    stub_restoration(monkeypatch, material, pool, library)
    state = scene.read_json_checkpoint(material['source'] / 'research_state.json')
    state['costs'] = deepcopy(material['starting_costs'])
    material.update(retained_state=state, interrupted_progress={'verified_prefix': 7},
        retained_comparison_roster_ref={'file': 'paid-roster.json', 'json_path': []},
        retained_score_binding={'question_id': 'native_public_response'})
    monkeypatch.setattr(scene, 'read_json_checkpoint',
        lambda *args: pytest.fail('the caller already transferred the admitted state'))
    result = scene.restore_completed_source_scene(registration, 'keep', tmp_path / 'keep',
                                                   material=material)
    assert result['agent'].retained_state is state
    assert result['validation']['authoritative_scientific_checkpoint']['paid_prefix'] == {
        'verified_prefix': 7}


@pytest.mark.parametrize(('field', 'value', 'message'), [
    ('compute_cost', 103, 'account precedes its committed research state'),
    ('observation_cost', 1, 'differs from its paid acquisitions'),
])
def test_completed_state_transfer_never_refunds_paid_work_or_fills_missing_acquisitions(
        tmp_path, monkeypatch, field, value, message):
    registration, material, pool, library = completed_fixture(tmp_path)
    stub_restoration(monkeypatch, material, pool, library)
    path = material['source'] / 'research_state.json'
    state = json.loads(path.read_text())
    state['costs'][field] = value
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match=message):
        scene.restore_completed_source_scene(registration, 'keep', tmp_path / 'keep',
                                               material=material)


def test_load_completed_material_carries_post_checkpoint_fees_without_double_billing(
        tmp_path, monkeypatch):
    registration, material, _, _ = completed_fixture(tmp_path)
    read = scene._read
    references = []

    def counted(reference, workspace):
        references.append(deepcopy(reference))
        assert Path(reference['file']).name not in {'research_state.json', 'theory_library.json'}
        return read(reference, workspace)

    monkeypatch.setattr(scene, '_read', counted)
    loaded = scene.load_completed_source_material(registration)
    assert loaded['retained_training_costs']['compute_cost'] == 100
    assert loaded['retained_training_costs']['python_execution_seconds'] == 3.
    assert loaded['starting_costs'] == material['starting_costs']
    assert loaded['previous']['costs'] == material['starting_costs']
    assert loaded['models'] == ['strong.py']
    assert loaded['new_candidate']['id'] == NEW
    assert loaded['parent_contract'] == material['parent_contract']
    assert [Path(item['file']).name for item in references].count('training.json') == 1
    assert json.loads((material['source'] / 'training.json').read_text())['costs'][
        'compute_cost'] == 100


def test_load_rejects_account_that_drops_already_paid_checkpoint_work(tmp_path):
    registration, material, _, _ = completed_fixture(tmp_path)
    path = material['source'] / 'final_costs.json'
    delta = json.loads(path.read_text())
    delta['compute_cost'] = 19
    path.write_text(json.dumps(delta))
    with pytest.raises(ValueError, match='account precedes'):
        scene.load_completed_source_material(registration)


def test_restore_rejects_new_object_with_inherited_old_cursor(tmp_path, monkeypatch):
    registration, material, pool, library = completed_fixture(tmp_path)
    library['executions'][PREFIX + NEW] = deepcopy(library['executions'][PREFIX + OLD])
    stub_restoration(monkeypatch, material, pool, library)
    with pytest.raises(ValueError, match='new20.*pre-existing execution'):
        scene.restore_completed_source_scene(registration, 'keep', tmp_path / 'keep', material=material)


def test_native_parent_recovers_original_declared_constructor_before_scope_matching(
        tmp_path, monkeypatch):
    registration, material, pool, library = completed_fixture(tmp_path)
    assert '_contract_source' not in pool['parent']
    stub_restoration(monkeypatch, material, pool, library)
    matched = []

    def exact_parent_contract(agent, original, recovered):
        matched.append(deepcopy(recovered['parent']))
        return recovered['parent'] == original['parent']

    monkeypatch.setattr(theory_consolidation, '_same_relation_work', exact_parent_contract)
    restored = scene.restore_completed_source_scene(
        registration, 'keep', tmp_path / 'keep', material=material)
    assert matched == [library['executions'][PREFIX + OLD]['request_basis']['parent']]
    assert restored['parent']['_contract_source'] == material['parent_contract']
    assert restored['parent']['program_source_ref'] == material['source_registration'][
        'material']['parent_Program_ref']
    assert restored['agent'].theory_index == pool


def relocated_references(tmp_path):
    left_workspace, right_workspace = tmp_path / 'remove', tmp_path / 'keep'
    external = str(tmp_path / 'original' / 'source.json')
    left = {'program_source_ref': {'file': str(left_workspace / 'research_state.json'),
            'json_path': ['questions', 1, 'program']},
        'execution': {'workspace': str(left_workspace), 'version': 2,
                      'fitted_values': [.125, -.5]},
        'transfers': [{'prior_transfer_ref': {'file': str(left_workspace / 'shared/theory_library.json'),
                                            'json_path': ['transfers', 280]}}],
        'source_ref': {'file': external, 'json_path': ['paid_rows', 1]}}
    right = deepcopy(left)
    right['program_source_ref']['file'] = str(right_workspace / 'research_state.json')
    right['execution']['workspace'] = str(right_workspace)
    right['transfers'][0]['prior_transfer_ref']['file'] = str(right_workspace / 'shared/theory_library.json')
    return left, right, left_workspace, right_workspace


def test_same_restored_value_allows_corresponding_local_reference_relocation(tmp_path):
    left, right, left_workspace, right_workspace = relocated_references(tmp_path)
    original = deepcopy((left, right))
    assert scene._same_restored_value(left, right, left_workspace, right_workspace)
    assert (left, right) == original


@pytest.mark.parametrize('difference', ['json_path', 'version', 'fitted_value'])
def test_same_restored_value_keeps_path_versions_and_scientific_values_exact(tmp_path, difference):
    left, right, left_workspace, right_workspace = relocated_references(tmp_path)
    if difference == 'json_path':
        right['transfers'][0]['prior_transfer_ref']['json_path'][-1] = 281
    elif difference == 'version':
        right['execution']['version'] = 3
    else:
        right['execution']['fitted_values'][0] = .126
    assert not scene._same_restored_value(left, right, left_workspace, right_workspace)


def test_same_restored_value_does_not_collapse_external_source_references(tmp_path):
    left, right, left_workspace, right_workspace = relocated_references(tmp_path)
    right['source_ref']['file'] = str(tmp_path / 'another_source' / 'source.json')
    assert not scene._same_restored_value(left, right, left_workspace, right_workspace)


def first_pass_fixture(tmp_path, arm, *, route_compute=0):
    registration, material, pool, library = completed_fixture(tmp_path)
    prior = tmp_path / 'first_pass' / arm
    prior.mkdir(parents=True)
    original = deepcopy(library['executions'][PREFIX + OLD])
    new_cursor = {'layer': 0, 'offset': 0,
                  'fitting': {'structures': [], 'retained_metadata': 'new20-own-search'}}
    if arm == 'keep':
        library['candidates'][NEW] = deepcopy(material['new_candidate'])
        current = library['executions'][PREFIX + OLD]
        current['request_basis']['parent'].pop('contract_source')
        current['cursor'] = {'layer': 0, 'offset': 0, 'pending_fits': []}
        current['result'] = {'status': 'UNKNOWN', 'new_compute_cost': 19}
        library['executions'][PREFIX + NEW] = {
            'request_basis': {'candidate_id': NEW, 'candidate_version': 1},
            'cursor': new_cursor,
            'result': {'status': 'UNKNOWN', 'new_compute_cost': 18}}
    else:
        library['executions'][PREFIX + OLD]['result']['retained_note'] = 'do-not-overwrite'
    delta = {**material['starting_costs'], 'observation_cost': 0,
             'compute_cost': 50, 'python_execution_seconds': 5.}
    previous = deepcopy(material['previous'])
    previous['costs'].update(compute_cost=150, python_execution_seconds=8.)
    actions = [{'action': 'frozen_recall', 'compute_delta': 46},
               {'action': 'research_choice', 'compute_delta': 2, 'models_requested': 2},
               {'action': 'research_route', 'compute_delta': route_compute},
               {'action': 'research_choice', 'compute_delta': 0, 'models_requested': 2}]
    for name, value in [('training.json', previous), ('final_costs.json', delta),
                        ('arm_summary.json', {'starting_costs': material['starting_costs']}),
                        ('registered_actions.json', actions)]:
        (prior / name).write_text(json.dumps(value))
    return registration, material, pool, library, prior, original


@pytest.mark.parametrize('arm', ['remove', 'keep'])
def test_resume_unused_allocation_preserves_actual_costs_caps_and_separate_searches(
        tmp_path, monkeypatch, arm):
    registration, material, pool, library, prior, original = first_pass_fixture(tmp_path, arm)
    before_library = deepcopy(library)
    stub_restoration(monkeypatch, material, pool, library, restore_source=prior,
                     retain_declared_contract=True)
    monkeypatch.setattr(theory_consolidation, '_same_relation_work',
        lambda agent, left, right: theory_consolidation._relation_work_basis(left)
        == theory_consolidation._relation_work_basis(right))
    result = scene.resume_completed_source_scene(
        registration, arm, prior, tmp_path / 'second' / arm,
        material=material, original_execution=original)
    agent = result['agent']
    assert agent.costs['compute_cost'] == 152
    assert agent.costs['python_execution_seconds'] == 8.25
    assert agent.costs['observation_cost'] == 2
    assert agent.config.compute_budget == 602
    assert agent.config.execution_seconds == 33.25
    assert result['original_starting_costs'] == material['starting_costs']
    assert result['prior_continuation_costs']['compute_cost'] == 50
    assert result['resume_phase'] == 'last_research_allocation'
    assert result['first_allocation_remaining'] == 150
    assert agent.theory_index == pool
    assert agent.theory_consolidation['transfers'] == library['transfers']
    assert agent.theory_consolidation['material_cursors'] == library['material_cursors']
    assert result['models'] == ['strong.py', 'winner.py']
    assert result['parent']['_contract_source'] == material['parent_contract']
    executions = agent.theory_consolidation['executions']
    preserved = result['validation']['preserved_misbound_execution_key']
    if arm == 'keep':
        assert executions[PREFIX + OLD] == original
        assert preserved and executions[preserved] == before_library['executions'][PREFIX + OLD]
        assert executions[PREFIX + NEW] == before_library['executions'][PREFIX + NEW]
        assert executions[PREFIX + NEW]['cursor'] != original['cursor']
    else:
        assert preserved is None
        assert executions[PREFIX + OLD] == before_library['executions'][PREFIX + OLD]
        assert PREFIX + NEW not in executions
    assert result['validation']['new_fits'] == 0
    assert result['validation']['new_scores'] == 0
    assert result['validation']['new_predictions'] == 0


def test_resume_rejects_second_allocation_already_spent(tmp_path, monkeypatch):
    registration, material, _, _, prior, original = first_pass_fixture(
        tmp_path, 'keep', route_compute=1)
    monkeypatch.setattr(study, 'RegisteredOrganizationAgent',
                        lambda *args, **kwargs: pytest.fail('spent allowance cannot restore'))
    with pytest.raises(ValueError, match='second registered allowance was already consumed'):
        scene.resume_completed_source_scene(registration, 'keep', prior, tmp_path / 'second',
            material=material, original_execution=original)
