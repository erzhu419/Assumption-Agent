"""D3 preparation and static context loading; no scientific execution."""

from copy import deepcopy
import json
from pathlib import Path
from random import Random
from types import SimpleNamespace

import pytest

from scripts import prepare_pr3_source_successor as preparation
from scripts import run_anonymous_revision_inquiry as study


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {'file': str(path), 'json_path': []}


def test_saved_scene_allows_second_native_step_and_source_keeps_one(tmp_path, monkeypatch):
    from hegel_machine.cognition import temporal_paired_method

    author = tmp_path / 'author.py'
    author.write_text("def select_actions(spec, observations, memory):\n"
                      "    return {'action':'run', 'arguments':{'operator':'native_condition_research'}}\n")
    plan = {'author': {'path': author.name, 'max_steps': 2}}
    lab = SimpleNamespace(public_spec=lambda: {})

    def agent():
        return SimpleNamespace(workspace=tmp_path, observations=[], method_results=[],
            costs={'python_execution_seconds': 0.},
            knowledge={'finite_native_author': {'question': {'id': 'existing_question'}}},
            _organization_state={'finite_author': {'completed_steps': 0}},
            _procedure_budget_status=lambda: None, _before_model_turn=lambda lab: None,
            _after_model_turn=lambda lab: None, _prepare_program_state=lambda: None,
            _program_memory=lambda path, arguments: {})

    def execute(current, laboratory, path, **kwargs):
        current.method_results.append({'feedback': {'operator': 'native_condition_research'}})
        kwargs['after_step'](laboratory)
        return {'status': 'method_step_limit'}

    monkeypatch.setattr(temporal_paired_method, 'execute_procedure', execute)
    legacy = preparation._source_research_steps(agent(), lab, plan, lambda data: None)
    saved_scene = preparation._source_research_steps(agent(), lab, plan, lambda data: None,
        native_attempt_limit=8)
    assert legacy['status'] == 'registered_native_attempt_limit'
    assert legacy['native_attempts'] == legacy['new_method_results'] == 1
    assert saved_scene['status'] == 'method_step_limit'
    assert saved_scene['native_attempts'] == saved_scene['new_method_results'] == 2


@pytest.mark.parametrize('has_saved_purchase', [True, False])
def test_saved_scene_restores_only_actual_prefix_readout_bindings(tmp_path, has_saved_purchase):
    from hegel_machine.cognition.question_continuation import executed_plan_records
    from hegel_machine.cognition.question_queries import evaluate_on_evidence

    context = {'source_id': 'saved/target/delay', 'domain': 'anonymous_public_u',
               'reset': 'per_trajectory'}
    action = {'kind': 'trajectory', 'sequence': [{'u': .5}]}
    query = {'op': 'Readout', 'record_index': {
        'experiment_id': 'public_positive_half_reset', 'action_index': 0},
        'step': 0, 'target': 'response', 'units': '1', 'context_ref': context,
        'binding': {'action': action, 'features': {'u': .5}, 'history': []}}
    public = {'id': 'native_public_response', 'parent_id': 'task',
        'query_semantics': {'mode': 'model_query', 'query_expression': query}}
    records, offset = [], 0
    for count in (24, 24, 24, 1, 12, 12):
        records.append({'action': action, 'row_indices': list(range(offset, offset + count)),
            'rows': [{'features': {'u': .5}, 'target': .21383203807282364, 'history': []}
                     for _ in range(count)]})
        offset += count
    source = {'kind': 'question_selection', 'question_id': public['id'], 'meaning_version': 1,
              'experiment_id': 'public_positive_half_reset', 'routing_version': 2}
    actual = {'source': source, 'record_indices': [3], 'query': {'actions': [action]}}
    future = {'source': {**source, 'experiment_id': 'future_panel'}, 'record_indices': [6]}
    mixed = {'source': {**source, 'experiment_id': 'partly_future'}, 'record_indices': [3, 6]}
    old_meaning = {'source': {**source, 'meaning_version': 2}, 'record_indices': [3]}
    parent = {'id': 'task', 'executed_pending_queries': [future, mixed, old_meaning]}
    if has_saved_purchase:
        parent['executed_pending_queries'].insert(0, actual)
    candidates = [{'model': f'native/{i}.py', 'program': {}} for i in range(14)]
    banks = [{'fit_evidence': [0], 'candidates': chunk} for chunk in
             (candidates[:0], candidates[:6], candidates[6:])]
    state = {'questions': [parent, {**public, 'native_condition_research': banks}],
        'operator_results': [{} for _ in range(8)] + [
            {'operator': 'research_choice', 'paid_rows': 97}]}
    events = [{'feedback': {}} for _ in range(20)]
    events[18]['feedback']['answer_execution'] = {'answer': {
        'status': 'answered', 'value': .21383203807282364, 'evidence': [72]}}
    events[19]['feedback']['evaluated_model'] = {'evaluated_rows': 97, 'current_paid_rows': 97}
    methods = [{'feedback': {}, 'proposal': {}} for _ in range(10)]
    methods[0]['proposal']['arguments'] = {'models': [f'old/{i}.py' for i in range(1932)]}
    methods[9]['proposal']['research_question'] = public
    training = {'events': events, 'method_results': methods, 'paid_observations': records,
                'theory_index': {}}
    seed = {'online_questions': [{'id': 'task'}], 'theory_index': {}}
    previous, projected, _, models = preparation._saved_scene_projection(
        tmp_path, training, state, seed)
    assert models == [f'old/{i}.py' for i in range(1932)] + [f'native/{i}.py' for i in range(14)]
    bindings = executed_plan_records(SimpleNamespace(questions=previous['online_questions']),
                                    previous['online_questions'][1])
    assert bindings == ({'public_positive_half_reset': [3]} if has_saved_purchase else {})
    assert projected['questions'] == previous['online_questions']
    assert parent['executed_pending_queries'][-3:] == [future, mixed, old_meaning]
    answer = evaluate_on_evidence(query, previous['paid_observations'], context=context,
                                 record_bindings=bindings)
    if has_saved_purchase:
        assert answer['status'] == 'answered' and answer['evidence'] == [72]
        assert answer['value'] == previous['online_questions'][1]['semantic_answer']['value']
    else:
        assert answer['status'] == 'scoped_unknown'
        assert answer['reason'] == 'unknown_due_to_missing_evidence'


def source_spec(tmp_path):
    base = tmp_path / 'baseline'
    model = base / '_evaluated_model/base.py'
    model.parent.mkdir(parents=True)
    model.write_text("raise RuntimeError('fixture must never execute')\n")
    save(model.parent / 'params.json', {'coefficient': 3})
    archive = save(tmp_path / 'archive.json', {'workspace': str(base),
        'research_roles': {'predictors': ['_evaluated_model/base.py']}})
    native = base / '_experiment_models/native/_evaluated_model/native.py'
    native.parent.mkdir(parents=True)
    native.write_text("raise RuntimeError('native fixture must never execute')\n")
    native_ref = save(tmp_path / 'method_decision.json', {
        'research_options': [{'current_executions': [{}, {'models': [
            '_evaluated_model/base.py', '_experiment_models/native/_evaluated_model/native.py']}]}]})
    native_ref['json_path'] = ['research_options', 0, 'current_executions', 1, 'models']
    ctx = {'source_id': 'old/source/case_000', 'domain': 'anonymous_public_u', 'reset': 'per_trajectory'}
    panels = []
    for level, identifier in ((-.5, 'negative'), (.5, 'positive')):
        action = {'kind': 'trajectory', 'sequence': [{'u': level}]}
        query = {'op': 'Readout', 'record_index': 3, 'step': 0,
            'context_ref': deepcopy(ctx), 'target': 'response', 'units': '1',
            'binding': {'action': action, 'features': {'u': level}, 'history': []}}
        panels.append({'id': identifier, 'actions': [action], 'query': query,
                       'tolerance': 1e-6, 'observation_cost': 1})
    question = {'id': 'native_public_response', 'question': 'Declared public scalar response',
        'context_ref': ctx, 'answer_resolution_ref': {'atol': 1e-6},
        'theory_target': {'context_ref': deepcopy(ctx), 'output_unit': '1', 'role_signature': {
            'u': {'role': 'public_control', 'unit': '1', 'time': 'current'}}},
        'query_semantics': {'mode': 'model_query', 'query_expression': deepcopy(panels[1]['query'])},
        'resolution_experiments': panels,
        'use_sites': [{'kind': 'choose_parent_experiment', 'parent_id': 'task'}]}
    question_ref = save(tmp_path / 'question.json', question)
    fitting_ref = save(tmp_path / 'fitting.json', {'operations': ['refit', 'representation',
        'relation', 'composition', 'robustification', 'idealization'], 'risk_quantile': .5,
        'max_history': 4, 'max_structures': 8})
    batches = []
    for identifier, coefficient in (('old82', 8), ('old109', 9)):
        workspace = tmp_path / identifier
        context = workspace / '_experiment_models/revision/_evaluated_model'
        context.mkdir(parents=True)
        (context / 'candidate_0.py').write_text("raise RuntimeError('never execute')\n")
        save(context / 'params.json', {'coefficient': coefficient})
        save(context / 'model_observations.json', [{'features': {'u': 1}, 'target': coefficient}])
        save(context / 'model_bindings.json', {'candidate_0.py': {
            'old': '_submitted_models/old/_evaluated_model/child.py'}})
        child = context / '_submitted_models/old/_evaluated_model/child.py'
        child.parent.mkdir(parents=True)
        child.write_text("raise RuntimeError('bound child must not execute')\n")
        save(child.parent / 'params.json', {'frozen_child': coefficient + 1})
        batches.append({'id': identifier, 'workspace': str(workspace),
            'pattern': '_experiment_models/revision/_evaluated_model/candidate_*.py',
            'expected_count': 1, 'evidence_ref': save(workspace / 'readout.json', {'status': 'retained_failed'})})
    caps = {'observation_budget': 576, 'compute_budget': 250000, 'confirmation_rows': 96,
        'confirmation_mse_threshold': 1e-10, 'execution_seconds': 300.,
        'max_model_calls': 0, 'token_budget': 0}
    return {'namespace': 'd3_fixture', 'materials': {'baseline_archive': archive['file'],
        'native_models_ref': native_ref, 'initial_library_ref': save(tmp_path / 'library.json', {'candidates': {}}),
        'public_question_ref': question_ref, 'revision_fitting_ref': fitting_ref,
        'extra_predictor_batches': batches, 'regression_refs': []},
        'author': {'kind': 'finite_native', 'path': 'shared/finite_native_author.py',
                   'max_steps': 48, 'confirm': True},
        'native_search': {'max_compute': 50000, 'max_candidates': 64, 'boundary_resolution': .1},
        'source': {'kinds': ['memoryless', 'delay', 'hysteresis'], 'cases': 1,
                   'seed': 202610260101, 'limits': caps},
        'target': {'kinds': ['memoryless', 'delay', 'hysteresis'], 'cases': 1,
                   'seed': 202610270101, 'limits': {**caps, 'compute_budget': 500000}},
        'organization': {'exploration_policy': {'kind': 'seeded_legal', 'seed': 20261021},
            'routing_allocation': {'compute': 150000}, 'consolidation': {
                'object_type': 'program_revision_relation', 'source_theory_ids': [],
                'source_question_refs': [{'question_id': question['id'], 'meaning_version': 1}],
                'allocation': {'compute': 25000, 'max_groups': 8},
                'mission': {'purpose': 'theory_learning', 'mission_id': 'd3_fixture_source',
                    'theory_learning_allocation': {'compute': 25000}}}},
        'future_targets': {'frozen_library_path': str(tmp_path / 'future/keep.json')},
        'design': {'scope': 'scripted preparation fixture, not scientific evidence'}}


def test_prepare_reads_only_metadata_and_preserves_registered_contracts(tmp_path, monkeypatch):
    spec = source_spec(tmp_path)
    spec_path = tmp_path / 'spec.json'
    save(spec_path, spec)
    original_read = Path.read_text

    def read(path, *args, **kwargs):
        if path.suffix == '.py' and path.is_relative_to(tmp_path):
            pytest.fail('prepare read a model implementation')
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr(study, 'run_study', lambda *a, **k: pytest.fail('prepare ran a study'))
    monkeypatch.setattr(study, 'AnonymousHistoryLab', lambda *a, **k: pytest.fail('prepare made a laboratory'))
    monkeypatch.setattr(preparation, '_source_knowledge', lambda *a: pytest.fail('prepare expanded files'))
    output = tmp_path / 'prepared'
    registration = preparation.prepare(spec_path, output)
    assert registration['status'] == 'prepared_not_executed'
    assert registration['baseline_model_count'] == 4
    assert registration['baseline_materialization'] == 'pending_full_context_export'
    source = preparation._read(output / 'source_plan.json')
    target = preparation._read(output / 'target_plan.json')
    assert source['arms'] == target['arms'] == list(study.ORGANIZATION_ARMS)
    cohort = source['source_cohort_allocation']
    assert cohort['source_compute_limit'] == 750000
    assert cohort['aggregation_compute'] == 25000
    assert list(cohort['source_stage_compute'].values()) == [241666, 241667, 241667]
    assert cohort['local_consolidation_compute'] == 0
    assert source['limits']['compute_budget'] == 250000
    assert target['limits']['compute_budget'] == 500000
    assert source['research_organization']['consolidation']['object_type'] == 'program_revision_relation'
    for phase, plan in [('source', source), ('target', target)]:
        assert plan['limits']['max_model_calls'] == plan['limits']['token_budget'] == 0
        assert plan['limits']['confirmation_rows'] == 96
        assert plan['limits']['confirmation_mse_threshold'] == 1e-10
        for case in plan['cases']:
            question = plan['prepared_public_questions'][case['id']]
            expected = f'd3_fixture/{phase}/{case["id"]}'
            assert question['context_ref']['source_id'] == expected
            assert question['theory_target']['revision_fitting']['max_structures'] == 8
            query = question['query_semantics']['query_expression']
            assert query['record_index'] == {'experiment_id': 'positive', 'action_index': 0}
            for panel in question['resolution_experiments']:
                assert panel['query']['record_index'] == {'experiment_id': panel['id'], 'action_index': 0}
                assert panel['query']['context_ref']['source_id'] == expected
    assert not Path(spec['future_targets']['frozen_library_path']).exists()
    assert registration['new_observations'] == registration['new_predictions'] == registration['new_fits'] == 0


def test_static_loader_preserves_distinct_contexts_fit_views_and_bound_children(tmp_path):
    spec = source_spec(tmp_path)
    prior = save(tmp_path / 'prior_diagnosis.json', {'status': 'retained_failed', 'compute': 19})
    spec['materials']['regression_refs'] = [prior['file']]
    output = tmp_path / 'prepared'
    registration = preparation.prepare(spec, output)
    inventory = preparation._read(output / 'baseline_inventory.json')
    knowledge = preparation._source_knowledge(registration, inventory)
    paths = knowledge['finite_native_author']['predictor_paths']
    assert len(paths) == 4
    assert paths[:2] == ['_evaluated_model/base.py', '_experiment_models/native/_evaluated_model/native.py']
    for batch, coefficient in [('old82', 8), ('old109', 9)]:
        root = f'retained_batches/{batch}/_experiment_models/revision/_evaluated_model'
        assert f'{root}/candidate_0.py' in paths
        assert json.loads(knowledge['files'][f'{root}/params.json']) == {'coefficient': coefficient}
        assert json.loads(knowledge['files'][f'{root}/model_observations.json'])[0]['target'] == coefficient
        assert json.loads(knowledge['files'][f'{root}/_submitted_models/old/_evaluated_model/params.json']) == {
            'frozen_child': coefficient + 1}
        assert f'{root}/model_bindings.json' in knowledge['files']
    assert knowledge['files']['shared/finite_native_author.py'] == (output / spec['author']['path']).read_text()
    from hegel_machine.cognition.research_episodes import load_source_values
    index = json.loads(knowledge['files']['shared/regression_evidence_refs.json'])
    assert index['sources'] == knowledge['regression_evidence_refs'] == [prior]
    assert load_source_values(index['sources'], output)[0]['value'] == {
        'status': 'retained_failed', 'compute': 19}
    assert 'prior_diagnosis.json' not in knowledge['files']


def test_prepare_rejects_incomplete_declared_batch_before_writing_plans(tmp_path):
    spec = source_spec(tmp_path)
    spec['materials']['extra_predictor_batches'][0]['expected_count'] = 2
    with pytest.raises(ValueError, match='expected 2 models, found 1'):
        preparation.prepare(spec, tmp_path / 'prepared')
    assert not (tmp_path / 'prepared/registration.json').exists()


@pytest.mark.parametrize('workers', [None, 3])
def test_authorized_source_function_uses_existing_loop_without_provider(tmp_path, monkeypatch, workers):
    spec = source_spec(tmp_path)
    output = tmp_path / 'prepared'
    preparation.prepare(spec, output)
    called = []

    def existing_loop(plan, path, **kwargs):
        called.append((plan, path, kwargs))
        return {'status': 'fixture_not_executed'}

    monkeypatch.setattr(study, 'run_study', existing_loop)
    options = {} if workers is None else {'workers': workers}
    assert preparation.run_prepared_source(output, **options) == {'status': 'fixture_not_executed'}
    plan, path, kwargs = called[0]
    assert path == output / 'source'
    assert 'provider_factory' not in kwargs
    assert kwargs['workers'] == (8 if workers is None else workers)
    assert plan['provider'] == {'backend': 'finite_native_procedure', 'model_calls': 0}
    assert len(kwargs['knowledge']['research_roles']['predictors']) == 4
    monkeypatch.setattr(study, 'RegisteredOrganizationAgent', lambda provider, workspace, **kw: (provider, kw))
    for case in plan['cases']:
        provider, actual = kwargs['agent_factory'](None, output / 'source' / case['id'] / 'pr3',
            knowledge=deepcopy(kwargs['knowledge']))
        assert provider is None
        assert actual['knowledge']['finite_native_author']['question']['context_ref']['source_id'] == (
            f'd3_fixture/source/{case["id"]}')


def test_source_bridge_binds_each_case_and_uses_original_paid_prefix_contract(tmp_path, monkeypatch):
    spec = source_spec(tmp_path)
    output = tmp_path / 'prepared'
    preparation.prepare(spec, output)
    expected_plan = preparation._read(output / 'source_plan.json')
    calls = []

    class StubAgent:
        def __init__(self, provider, workspace, **kwargs):
            assert provider is None
            self.knowledge = kwargs['knowledge']
            self.events = []
            self.calls = []

        def _execute(self, action, arguments, laboratory):
            self.calls.append((action, deepcopy(arguments), laboratory))
            return {'status': 'stub_prefix_only'}

        def _open_question(self, question):
            self.calls.append(('open_question', deepcopy(question)))
            return question

    class StubLab:
        def confirmation_actions(self, rows, rng):
            self.request = (rows, rng.randrange(1000000))
            return [{'kind': 'trajectory', 'sequence': [{'u': 0.}]}]

    def existing_loop(plan, destination, **kwargs):
        assert plan == expected_plan
        assert destination == output / 'source'
        assert 'provider_factory' not in kwargs
        assert kwargs['workers'] == 8
        common = kwargs['knowledge']
        assert 'question' not in common['finite_native_author']
        assert len(common['finite_native_author']['predictor_paths']) == 4
        for case in plan['cases']:
            for arm in plan['arms']:
                agent = kwargs['agent_factory'](None, destination / case['id'] / arm,
                    knowledge=deepcopy(common), arm=arm,
                    research_organization=plan['research_organization'])
                question = plan['prepared_public_questions'][case['id']]
                assert agent.knowledge['finite_native_author']['question'] == question
                assert question['context_ref']['source_id'] == f'd3_fixture/source/{case["id"]}'
                laboratory = StubLab()
                assert kwargs['prepare_prefix'](agent, laboratory, case) == {}
                assert laboratory.request == (plan['prefix_rows'],
                    Random(case['prefix_seed']).randrange(1000000))
                assert [call[0] for call in agent.calls] == ['query', 'open_question']
                assert agent.calls[0][1]['actions'] == agent.events[0]['arguments']['actions']
                assert agent.calls[0][2] is laboratory
                assert agent.calls[1][1] == question
                assert agent._active_question_id == question['id']
                assert agent.events[0]['usage']['backend'] == 'common_paid_prefix'
                calls.append((case['id'], arm))
        assert 'question' not in common['finite_native_author']
        return {'status': 'stub_bridge_complete'}

    monkeypatch.setattr(study, 'RegisteredOrganizationAgent', StubAgent)
    monkeypatch.setattr(study, 'AnonymousHistoryLab', lambda *a, **k: pytest.fail('actual lab constructed'))
    monkeypatch.setattr(study, 'run_study', existing_loop)
    assert preparation.run_prepared_source(output) == {'status': 'stub_bridge_complete'}
    assert len(calls) == len(set(calls)) == 9


def saved_readout_scene(tmp_path):
    """Real query execution over a97-row paid prefix, with no predictor involved."""
    context = {'source_id': 'saved/target/delay', 'domain': 'anonymous_public_u',
               'reset': 'per_trajectory'}
    action = {'kind': 'trajectory', 'sequence': [{'u': .5}]}
    declaration = {'id': 'native_public_response', 'meaning_version': 1,
        'parent_id': 'task', 'kind': 'value', 'answer_type': 'number',
        'context_ref': context, 'answer_resolution_ref': {'atol': 0.},
        'query_semantics': {'mode': 'model_query', 'query_expression': {
            'op': 'Readout', 'record_index': {'experiment_id': 'public_positive_half_reset',
                                          'action_index': 0},
            'step': 0, 'target': 'response', 'units': '1', 'context_ref': context,
            'binding': {'action': action, 'features': {'u': .5}, 'history': []}}},
        'theory_target': {'revision_fitting': {'operations': ['refit']},
            'parent_theory_id': 'saved_parent', 'evidence_checks': [{'row': 72}]},
        'semantic_answer': {'status': 'answered', 'value': .21383203807282364,
                            'evidence': [72]},
        'use_sites': [{'kind': 'choose_parent_experiment', 'parent_id': 'task'}]}
    parent = {'id': 'task', 'executed_pending_queries': [{
        'source': {'kind': 'question_selection', 'question_id': declaration['id'],
                   'meaning_version': 1, 'experiment_id': 'public_positive_half_reset'},
        'record_indices': [3]}]}
    records, offset = [], 0
    for count in (24, 24, 24, 1, 12, 12):
        records.append({'action': action, 'row_indices': list(range(offset, offset + count)),
            'rows': [{'features': {'u': .5}, 'target': .21383203807282364, 'history': []}
                     for _ in range(count)]})
        offset += count
    agent = SimpleNamespace(workspace=tmp_path, questions=[parent, deepcopy(declaration)],
        config=SimpleNamespace(confirmation_mse_threshold=1e-10, compute_budget=1500000),
        costs={'compute_cost': 100, 'observation_cost': 145, 'measurement_cost': 0,
               'python_execution_seconds': 10., 'model_calls': 0, 'reported_tokens': 0},
        research_mission={}, research_input_bindings={}, research_candidates={},
        observations=records, rows=[row for record in records for row in record['rows']],
        evaluated_model={'current_mse': .00216, 'evaluated_rows': 97,
                         'source_model': 'saved.py'},
        theory_index={'saved_parent': {'id': 'saved_parent', 'version': 2}},
        _bind_revision_target=lambda question: None, _prepare_program_state=lambda: None,
        _python=lambda *a, **k: pytest.fail('Acceptance must not execute Python'),
        _execute=lambda *a, **k: pytest.fail('Acceptance must not fit, predict or purchase'))

    def open_question(question):
        question = deepcopy(question)
        question.setdefault('meaning_version', 1)
        agent.questions.append(question)
        return question

    agent._open_question = open_question
    agent.result = lambda status: {'status': status, 'online_questions': agent.questions,
                                   'costs': deepcopy(agent.costs)}
    manifest = {'old_readout_ref': save(tmp_path / 'declared.json', declaration),
                'parent_theory_id': 'saved_parent'}
    return agent, SimpleNamespace(), manifest


def test_saved_scene_acceptance_preserves_paid_answer_and_separate_duty_without_recall(tmp_path):
    agent, lab, manifest = saved_readout_scene(tmp_path)
    receipt = preparation._accept_saved_scene(agent, lab, manifest)
    assert receipt['status'] == 'accepted'
    assert receipt['semantic_answer']['value'] == .21383203807282364
    assert receipt['named_record_bindings'] == {'public_positive_half_reset': [3]}
    assert receipt['parent_obligation']['native_research_ref'] == {
        'question_id': 'native_public_response', 'meaning_version': 1}
    assert receipt['parent_obligation']['id'] != 'native_public_response'
    assert receipt['new_costs']['compute_cost'] == 1
    assert sum(receipt[key] for key in ('new_fits', 'new_predictions', 'new_observations', 'new_recalls')) == 0
    checkpoint = json.loads((tmp_path / 'scene_acceptance_checkpoint.json').read_text())
    assert checkpoint['costs']['compute_cost'] == 101
    assert len(checkpoint['online_questions']) == 3


@pytest.mark.parametrize('failure', ['missing_purchase', 'wrong_answer', 'missing_obligation'])
def test_saved_scene_acceptance_keeps_failed_scene_and_blocks_science(tmp_path, failure):
    agent, lab, manifest = saved_readout_scene(tmp_path)
    if failure == 'missing_purchase':
        agent.questions[0]['executed_pending_queries'] = []
    elif failure == 'wrong_answer':
        agent.questions[1]['semantic_answer']['value'] = 0.
    else:
        agent.questions[1]['theory_target']['evidence_checks'] = []
    with pytest.raises(ValueError):
        preparation._accept_saved_scene(agent, lab, manifest)
    receipt = json.loads((tmp_path / 'scene_acceptance.json').read_text())
    assert receipt['status'] == 'failed'
    assert (tmp_path / 'scene_acceptance_checkpoint.json').is_file()
    assert not (tmp_path / 'source_at_run').exists()
    assert agent.costs['model_calls'] == 0
    assert receipt['new_costs']['compute_cost'] == (1 if failure == 'missing_obligation' else 0)


def test_saved_scene_run_stops_before_freezing_when_acceptance_fails(tmp_path, monkeypatch):
    def reject(*args):
        raise ValueError('second restored arm has no real record binding')
    monkeypatch.setattr(preparation, 'accept_prepared_saved_scene', reject)
    with pytest.raises(ValueError, match='real record binding'):
        preparation.run_prepared_saved_scene(tmp_path, tmp_path / 'run')
    assert not (tmp_path / 'run/source_at_run').exists()


def test_saved_scene_freezes_expanded_libraries_and_routes_to_unchanged_versions(tmp_path):
    old = {'old': {'id': 'old', 'version': 4}}
    new = {'new': {'id': 'new', 'version': 7}}
    library = {'candidates': {**old, **new}, 'executions': {}}
    base = tmp_path / 'base/shared/theory_library.json'
    save(base, library)
    arms = {}
    for arm in ('fixed_pr2', 'pr3'):
        workspace = tmp_path / arm
        local = workspace / 'shared/theory_library.json'
        selected = {'source_ref': {'file': str(base), 'json_path': []},
                    'removed_candidates': ['new'] if arm == 'fixed_pr2' else []}
        save(local, selected)
        agent = SimpleNamespace(workspace=workspace, questions=[], files={},
            _organization_state={'automatic_library_path': str(local)},
            _save=lambda name, text, workspace=workspace: save(workspace / name, json.loads(text)),
            _prepare_program_state=lambda: None)
        arms[arm] = {'agent': agent}
    frozen = preparation._freeze_saved_scene_libraries({'output': tmp_path / 'run',
        'arms': arms, 'manifest': {'library_views': {'old_ids': ['old'], 'new_ids': ['new']}}})
    for arm, candidates in (('fixed_pr2', old), ('pr3', {**old, **new})):
        path = frozen[arm]['library_ref']['file']
        actual = json.loads(Path(path).read_text())
        assert actual['candidates'] == candidates
        assert 'source_ref' not in actual
        agent = arms[arm]['agent']
        assert agent._organization_state['automatic_library_path'] == path
        assert agent.theory_consolidation['candidates'] == candidates
        assert agent._theory_library_source_ref['file'] == path
