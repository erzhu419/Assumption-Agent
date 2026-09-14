"""Executable common-source information and separately paid Code terminal rows."""

from copy import deepcopy
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition import unified_session

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_instrument_relation_diagnosis as controlled
from instrument_relation_code_source import export_source


@pytest.fixture(scope='module')
def common_source():
    # This is the study's already declared source setup on an existing dev
    # apparatus. No new research cohort or model-provider call is run here.
    namespaces = {'models': ('Observation', 'Probe', 'SearchBudget', 'ResearchConfig', 'Candidate'),
        'language': ('Program',), 'memory': ('KnowledgeLibrary',),
        'unified_research': ('UnifiedResearchAgent',), 'research_graph': ('ResearchGraph',),
        'competing_object_experiments': ('_forecast', 'assess_competing_object_probe'),
        'scale_bindings': ('bind_measurement_arguments',), 'instrument_models': ('rebind_instrument_model',)}
    for name, attributes in namespaces.items():
        namespace = importlib.import_module('hegel_machine.cognition.' + name)
        for attribute in attributes:
            setattr(controlled, attribute, getattr(namespace, attribute))
    protocol = json.loads((ROOT / 'artifacts/instrument_relation_diagnosis_20260914/protocol.json').read_text())
    source = controlled.make_source(unified_session, protocol, 9013)
    assert source['costs']['observation_cost'] == 89
    return protocol, json.loads(json.dumps(source))


def test_exported_source_python_matches_the_given_mechanism_and_paid_calibration(common_source):
    protocol, source = common_source
    library = controlled.KnowledgeLibrary.from_dict(json.loads(json.dumps(source['knowledge'])))
    with controlled.laboratory(unified_session, protocol, 9013, 'source_less') as lab:
        spec = lab.spec_for(('fine',))
        agent = controlled.agent_for(protocol, 9013, library)
        binding = agent._instrument_binding(spec, lab.spec)
        code = export_source(source['measured_program'],
                             {name: program.to_dict() for name, program in binding.items()},
                             json.loads(spec.context['object_schema']))
        exported = {}
        exec(code, exported)
        native = Program.from_dict(source['measured_program'])
        rebound = controlled.bind_measurement_arguments(native, binding)
        batches = controlled.restore_pairs(source)
        assert [len(batch) for batch in batches] == [48, 9, 32]
        for batch in batches:
            for commands, measured in batch:
                command_row = {'features': commands.features, 'history': commands.history}
                measured_row = {'features': measured.features, 'history': measured.history}
                for radius in (.8, .7):
                    native_variant = controlled.radius_variant(native, radius)
                    command_variant = controlled.bind_measurement_arguments(native_variant, binding)
                    assert exported['predict'](command_row, radius) == pytest.approx(command_variant.predict(commands), abs=1e-10)
                    assert exported['predict_measured'](measured_row, radius) == pytest.approx(native_variant.predict(measured), abs=1e-10)
                assert exported['calibrate'](command_row) == pytest.approx(
                    {name: mapping.predict(commands) for name, mapping in binding.items()}, abs=1e-12)
        assert [record['batch']['transitions'] for record in source['batches']]


@pytest.fixture
def code_study(common_source):
    import run_instrument_relation_code_comparison as study
    from hegel_machine.cognition.code_agent import CodeAgentConfig, evaluate_python_model
    from hegel_machine.cognition.unified_code_control import UnifiedCodeSession
    study.CodeAgentConfig = CodeAgentConfig
    study.evaluate_python_model = evaluate_python_model
    study.UnifiedCodeSession = UnifiedCodeSession
    protocol, source = common_source
    protocol = deepcopy(protocol)
    protocol['code_config'] = asdict(CodeAgentConfig(observation_budget=512, measurement_budget=6144,
        compute_budget=480000, max_steps=12, max_model_calls=12, token_budget=240000))
    protocol['provider'] = {}
    return study, protocol, source


class ScriptedProvider:
    def __init__(self, actions, inspect=None):
        self.actions, self.inspect, self.calls = actions, inspect, 0

    def next_action(self, request):
        if self.inspect:
            self.inspect(request, self.calls)
        action, arguments = self.actions[self.calls]
        self.calls += 1
        return {'action': action, 'arguments': arguments, 'usage': {'total_tokens': 3}}


def test_common_source_files_first_counterexample_budget_and_sibling_workspaces(code_study, tmp_path):
    study, protocol, source = code_study
    frozen_source = deepcopy(source)
    protocol['code_config']['observation_budget'] = 9
    first = study.prepare_case(unified_session, protocol, 9013, 'input_only', source)
    second = study.prepare_case(unified_session, protocol, 9013, 'joint', source)
    source_data = json.loads(first['data_files']['source_evidence.json'])['batches']
    assert len(source_data) == len(source['batches']) == 3
    for copied, original in zip(source_data, source['batches']):
        assert copied['probe'] == original['probe']
        assert copied['observations'] == original['batch']['observations']
        assert copied['transitions'] == original['batch']['transitions']
    assert sum(len(batch['observations']) for batch in source_data) == 89
    assert len(first['agent_state']['observations']) == 9
    assert first['agent_state']['costs']['observation_cost'] == 9
    assert first['agent_state']['costs']['model_calls'] == 0
    # The correct-input case also receives its already paid source endpoint
    # preparation and revised calibration, with those costs in its branch.
    from hegel_machine.cognition.instrument_condition_probes import instrument_condition_probe
    records = []
    with controlled.laboratory(unified_session, protocol, 9013, 'source_less') as lab:
        agent = controlled.agent_for(protocol, 9013,
            controlled.KnowledgeLibrary.from_dict(deepcopy(source['knowledge'])))
        boundary, _ = instrument_condition_probe(agent._instrument_binding(lab.spec_for(('fine',)), lab.spec),
            lab.spec, controlled.restore_pairs(source), controlled.SearchBudget(10000),
            reference=controlled.edge(protocol, 'source_background', focus=1.).sequence[0], serial='source')
        for probe in (boundary, lab.random_probe(1885, length=32)):
            batch = lab.acquire(probe, ('fine',))
            records.append({'probe': asdict(probe), 'batch': batch.to_dict()})
    correct = study.prepare_case(unified_session, protocol, 9013, 'relation_only', source,
                                 correct_input_records=records)
    extra_rows = sum(record['batch']['observation_cost'] for record in records)
    assert correct['agent_state']['costs']['observation_cost'] == 9 + extra_rows
    assert correct['agent_state']['costs']['measurement_cost'] == 12 * extra_rows
    assert len(correct['agent_state']['observations']) == 9
    actual_source = json.loads(correct['data_files']['source_evidence.json'])['batches']
    assert sum(len(record['observations']) for record in actual_source) == 89 + extra_rows
    for copied, original in zip(actual_source[3:], records):
        assert copied['observations'] == json.loads(json.dumps(original['batch']['observations']))
        assert copied['transitions'] == json.loads(json.dumps(original['batch']['transitions']))
    directories = [study.case_directory(tmp_path, protocol, 9013, case) for case in ('input_only', 'joint')]
    seen = []

    def provider_factory(path):
        assert all(label not in str(path) for label in protocol['cases'])
        assert json.loads((path / 'source_evidence.json').read_text())['batches'] == source_data
        def inspect(request, index):
            if index == 0:
                seen.append(deepcopy(request))
                encoded = json.dumps(request)
                assert all(word not in encoded for word in (*protocol['cases'], 'current_relation_condition',
                                                            'boundary_focus', 'solved_threshold'))
                assert request['costs']['observation_cost'] == 9
                assert len(request['observations']) == 9
                assert request['events'] == []
            if index == 1:
                assert request['events'][0]['feedback'] == {'error': 'observation budget exhausted'}
        return ScriptedProvider([
            ('query', {'sequence': [first['first_probe']['sequence'][0]]}),
            ('write', {'path': 'shared/sibling_note.txt', 'code': 'only this branch learned this'}),
            ('finish', {'model': 'shared/source_model.py', 'channels': []})], inspect)

    first_session = study.new_session(unified_session, protocol, directories[0], first,
                                      provider_factory=provider_factory)
    first_result = first_session.run()
    assert first_result['costs']['observation_cost'] == 9
    assert first_result['costs']['model_calls'] == 3
    assert 'shared/sibling_note.txt' in first_result['knowledge']['files']
    second_seen = []
    def sibling(path):
        assert not (path / 'shared/sibling_note.txt').exists()
        def inspect(request, index):
            second_seen.append(request)
            assert 'shared/sibling_note.txt' not in request['files']
            assert request['costs']['model_calls'] == 0
        return ScriptedProvider([('finish', {'model': 'shared/source_model.py', 'channels': []})], inspect)
    second_session = study.new_session(unified_session, protocol, directories[1], second, provider_factory=sibling)
    second_session.run()
    assert seen and second_seen
    assert source == frozen_source


def test_measured_terminal_and_last_successful_snapshot_keep_costs_and_feedback_separate(code_study, tmp_path):
    study, protocol, source = code_study
    prepared = study.prepare_case(unified_session, protocol, 9013, 'joint', source)
    good = "def fit(rows):\n    assert len(rows) == 2\n    assert all(all(name.startswith('measured_') for name in row['features']) for row in rows)\ndef predict(row):\n    assert 'target' not in row\n    assert all(name.startswith('measured_') for name in row['features'])\n    return 0.0\n"
    provider = ScriptedProvider([
        ('query', {'sequence': prepared['first_probe']['sequence'][:2], 'channels': ['fine']}),
        ('write', {'path': 'model.py', 'code': good}),
        ('evaluate', {'model': 'model.py', 'channels': ['fine']}),
        ('write', {'path': 'model.py', 'code': "def predict(row):\n    raise RuntimeError('unfinished change')\n"})])
    protocol['code_config']['max_model_calls'] = 4
    directory = study.case_directory(tmp_path, protocol, 9013, 'joint')
    session = study.new_session(unified_session, protocol, directory, prepared,
                                provider_factory=lambda _: provider)
    state = session.run()
    model = state['results'][0]
    assert model['model_origin'] == 'last_successful_evaluation'
    assert model['status'] == 'model_call_budget_exhausted'
    assert model['selected_channels'] == ['fine']
    assert 'unfinished change' in (Path(model['research_workspace']) / 'model.py').read_text()
    assert 'unfinished change' not in (Path(model['workspace']) / 'model.py').read_text()
    assert state['costs']['observation_cost'] == 11
    assert state['costs']['measurement_cost'] == 24
    frozen = deepcopy(state)
    restored = study.new_session(unified_session, protocol, directory, prepared,
        state=json.loads(json.dumps(state)), provider_factory=lambda _: pytest.fail('stopped research was restarted'))
    assert json.loads(json.dumps(restored.run())) == json.loads(json.dumps(frozen))
    report = study.evaluate_case(unified_session, protocol, {'session': state})
    assert report['status'] == 'completed'
    assert report['model_origin'] == 'last_successful_evaluation'
    assert report['rows'] == report['observation_cost'] == report['compute_cost'] == 337
    assert report['measurement_cost'] == 337 * 12
    assert report['random_mse'] > 0 and report['boundary_mse'] > 0 and not report['solved']
    assert state == frozen
    assert provider.calls == 4
    missing = deepcopy(state)
    missing['results'][0]['model'] = None
    failure = study.evaluate_case(unified_session, protocol, {'session': missing})
    assert failure['status'] == 'missing_model' and not failure['solved']
    assert failure['expected_rows'] == 337 and failure['rows'] == 0
    assert failure['random_mse'] is None and failure['boundary_mse'] is None


def test_real_provider_adapter_passes_the_declared_model_effort_without_a_live_request(tmp_path, monkeypatch):
    from hegel_machine.cognition import code_agent
    configured = json.loads((ROOT / 'config/cognition_instrument_relation_code_comparison.json').read_text())
    seen = []
    def local_completion(command, payload, timeout, cwd):
        seen.append(command)
        destination = Path(command[command.index('--output-last-message') + 1])
        destination.write_text(json.dumps({'action': 'finish', 'arguments_json': '{"model":"model.py"}'}))
        return 0, json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 17, 'output_tokens': 5}}), '', False
    monkeypatch.setattr(code_agent, '_run', local_completion)
    provider = code_agent.CodexCodeProvider(tmp_path, **{**configured['provider'], 'executable': '/test/codex'})
    reply = provider.next_action({'capabilities': {}, 'observations': []})
    command = seen[0]
    assert command[command.index('--model') + 1] == configured['provider']['model']
    assert 'model_reasoning_effort=' + json.dumps(configured['provider']['reasoning_effort']) in command
    assert reply['usage']['requested_reasoning_effort'] == configured['provider']['reasoning_effort']
    assert reply['usage']['total_tokens'] == 22


def test_run_fit_and_terminal_share_selected_paid_view(code_study, tmp_path):
    study, protocol, source = code_study
    prepared = study.prepare_case(unified_session, protocol, 9013, 'input_only', source)
    inspect_view = """import json
rows = json.load(open('model_observations.json'))
assert len(rows) == 9
assert len(json.load(open('observations.json'))) == 18
assert all(all(k.startswith('measured_') for k in row['features']) for row in rows)
assert all(all(k.startswith('measured_') or k == '__target__' for k in point)
           for row in rows for point in row['history'])
json.dump(rows, open('training_reference.json', 'w'))
print(len(rows))
"""
    fitted = """import json
from shared.source_model import predict_measured as predict

def fit(rows):
    # This is deliberately a fit contract check: prediction-time rows may
    # differ, but fitting must keep exactly the nine acquired fine rows.
    assert rows == json.load(open('training_reference.json'))
"""
    provider = ScriptedProvider([
        ('query', {'sequence': prepared['first_probe']['sequence'], 'channels': ['fine']}),
        ('run', {'code': inspect_view, 'channels': ['fine']}),
        ('evaluate', {'model': 'shared/source_measured_model.py', 'channels': ['fine']}),
        ('write', {'path': 'model.py', 'code': fitted}),
        ('finish', {'model': 'model.py', 'channels': ['fine']})])
    directory = study.case_directory(tmp_path, protocol, 9013, 'input_only')
    session = study.new_session(unified_session, protocol, directory, prepared,
                                provider_factory=lambda _: provider)
    state = session.run()
    result = state['results'][0]
    assert result['status'] == 'completed'
    run_feedback = result['events'][1]['feedback']
    assert run_feedback['returncode'] == 0 and run_feedback['stdout'].strip() == '9'
    assert run_feedback['model_observation_count'] == 9
    assert result['events'][2]['feedback']['mse'] < 1e-20
    assert result['events'][-1]['feedback']['mse'] < 1e-20
    assert result['costs']['compute_cost'] - prepared['setup_costs']['compute_cost'] == 18
    frozen = deepcopy(state)
    terminal = study.evaluate_case(unified_session, protocol, {'session': state})
    assert terminal['status'] == 'completed' and terminal['solved']
    assert terminal['rows'] == 337
    assert terminal['measurement_cost'] == 337 * 12
    assert state == frozen and provider.calls == 5
