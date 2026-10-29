"""Common methods retain actual instances and call boundaries, not fit names."""

from copy import deepcopy
import json
from types import SimpleNamespace

from hegel_machine.cognition.action_policy import ActionOption
from hegel_machine.cognition.research_episodes import extract_method_segment
from hegel_machine.cognition.research_methods import ResearchMethods
from hegel_machine.cognition.research_methods import observe_common_method_segment


def ref(index, *, field='events'):
    return {'file': 'trace.json', 'json_path': [field, index]}


def step(index, action, gain=0., *, dependencies=(), call='first', method_id=None):
    return {'step_ref': ref(index), 'action': action, 'context': 'method_use',
        'call_ref': {'file': 'trace.json', 'json_path': ['calls', call]},
        'depends_on_refs': list(dependencies), 'gain': gain, 'observation_cost': 0,
        'compute_cost': 10, 'execution_complete': True, 'method_id': method_id}


def test_T01_exact_fit_instance_is_not_the_nearest_same_action():
    trace = [step(0, 'prepare'), step(1, 'fit', dependencies=[ref(0)]),
        step(2, 'fit'), step(3, 'compare', .5, dependencies=[ref(1)])]
    book = ResearchMethods()
    identifier = book.observe('method_use', trace, condition={}, source_task='source', step_refs=[ref(3)])
    method = next(item for item in book.methods if item['method_id'] == identifier)
    assert method['steps'] == ['prepare', 'fit', 'compare']
    assert method['step_refs'] == [ref(0), ref(1), ref(3)]
    assert method['dependency_refs'] == [[], [ref(0)], [ref(1)]]
    assert ref(2) not in method['step_refs']


def test_T02_reference_closure_crosses_six_steps_and_missing_stays_unresolved():
    trace = [step(0, 'prepare'), *[step(i, 'unrelated') for i in range(1, 8)],
        step(8, 'fit', .6, dependencies=[ref(0)])]
    book = ResearchMethods()
    identifier = book.observe('method_use', trace, condition={}, source_task='source', step_refs=[ref(8)])
    assert book.methods[0]['method_id'] == identifier
    assert book.methods[0]['step_refs'] == [ref(0), ref(8)]
    incomplete = deepcopy(trace)
    incomplete[-1]['depends_on_refs'] = [ref(99)]
    before = deepcopy(book.to_dict())
    assert book.observe('method_use', incomplete, condition={}, source_task='missing', step_refs=[ref(8)]) is None
    assert book.last_observation['status'] == 'UNKNOWN'
    assert book.last_observation['missing_dependencies'] == [ref(99)]
    assert book.to_dict() == before


def test_T03_neutral_dependency_is_kept_without_claiming_causal_step_credit():
    trace = [step(0, 'prepare'), step(1, 'fit', .7, dependencies=[ref(0)])]
    book = ResearchMethods()
    book.observe('method_use', trace, condition={}, source_task='source', step_refs=[ref(1)])
    assert book.methods[0]['steps'] == ['prepare', 'fit']
    assert book.methods[0]['step_outcomes'][0]['gain'] == 0.
    assert book.revisions[-1]['change'] == 'propose_reference_closed_sequence'
    assert book.last_observation['dependency_scope'] == 'actual_reads_not_causal_contribution'


def test_T04_same_method_different_calls_cannot_complete_one_sequence():
    book = ResearchMethods()
    book.observe('method_use', [step(0, 'prepare'), step(1, 'fit', .7, dependencies=[ref(0)])],
        condition={}, source_task='source', step_refs=[ref(1)])
    method = book.methods[0]
    identifier = method['method_id']
    original_gain = method['observed_gain']
    trace = [step(2, 'prepare', call='call_a', method_id=identifier),
        step(3, 'fit', .2, call='call_b', method_id=identifier)]
    book.observe('method_use', trace, condition={}, source_task='target', step_refs=[ref(3)])
    assert method['observed_gain'] == original_gain
    assert not method.get('sequence_outcomes', [])[1:]
    assert book._completed(method, trace) == ['fit']
    assert book.choose('method_use', {}, [ActionOption('prepare', 0, 10), ActionOption('fit', 0, 10)],
        trace, task_id='target', observation_remaining=10, compute_remaining=100) is None


def test_exact_mode_keeps_external_call_dependency_as_input_not_a_completed_step():
    trace = [step(0, 'prepare', call='old'),
        step(1, 'fit', .5, dependencies=[ref(0)], call='new')]
    book = ResearchMethods()
    book.observe('method_use', trace, condition={}, source_task='target', step_refs=[ref(1)])
    assert book.methods[0]['steps'] == ['fit']
    assert book.methods[0]['external_dependency_refs'] == [ref(0)]
    assert book.last_observation['external_dependency_refs'] == [ref(0)]


def test_thin_segment_follows_only_exact_product_ancestors_and_keeps_original_cost_refs(tmp_path, monkeypatch):
    program_a = {'file': 'trace.json', 'json_path': ['programs', 'A']}
    program_b = {'file': 'trace.json', 'json_path': ['programs', 'B']}
    trace = {'events': [
        {'action': 'prepare', 'status': 'completed', 'compute_cost': 3},
        {'action': 'fit', 'status': 'completed', 'read_refs': [ref(0)],
            'produced_refs': [program_a], 'compute_cost': 7, 'program': {'large': 'x' * 100_000}},
        {'action': 'fit', 'status': 'completed', 'produced_refs': [program_b]},
        {'action': 'compare', 'status': 'completed', 'read_refs': [program_a],
            'question_ref': {'question_id': 'q', 'meaning_version': 2}, 'compute_cost': 11}],
        'programs': {'A': {'source_ref': ref(1), 'program': {'op': 'const', 'value': 1}},
                     'B': {'source_ref': ref(2), 'program': {'op': 'const', 'value': 2}}},
        'unrelated_payload': 'private-unneeded-' * 100_000}
    (tmp_path / 'trace.json').write_text(json.dumps(trace))
    real_loads = json.loads
    def thin_loads(value, *args, **kwargs):
        assert len(value) < 2_000, 'projection decoded a scientific or trajectory payload'
        return real_loads(value, *args, **kwargs)
    monkeypatch.setattr('hegel_machine.cognition.research_episodes.json.loads', thin_loads)
    result = extract_method_segment(SimpleNamespace(workspace=tmp_path),
        root_result_ref=ref(3), decision_start_ref=ref(0))
    source = str(tmp_path / 'trace.json')
    absolute = lambda reference: {**reference, 'file': source}
    by_ref = lambda reference: next(node for node in result['nodes'] if node['step_ref'] == absolute(reference))
    assert result['status'] == 'reference_grounded'
    assert result['question_ref'] == {'question_id': 'q', 'meaning_version': 2}
    assert by_ref(ref(1))['depends_on_refs'] == [absolute(ref(0))]
    assert absolute(ref(1)) in by_ref(ref(3))['depends_on_refs']
    assert all(node['step_ref'] != absolute(ref(2)) for node in result['nodes'])
    assert by_ref(ref(1))['cost_refs'] == [absolute({**ref(1), 'json_path': ['events', 1, 'compute_cost']})]
    assert 'private-unneeded' not in json.dumps(result) and '"op"' not in json.dumps(result)
    assert result['execution_status'] == 'completed'


def test_thin_segment_missing_reference_is_explicit_not_an_invented_fit(tmp_path):
    (tmp_path / 'trace.json').write_text(json.dumps({'events': [
        {'action': 'compare', 'status': 'completed', 'read_refs': [ref(99)]}]}))
    result = extract_method_segment(SimpleNamespace(workspace=tmp_path),
        root_result_ref=ref(0), decision_start_ref=ref(0))
    assert result['status'] == 'UNKNOWN'
    assert result['missing_dependencies'] == [{'ref': {**ref(99), 'file': str(tmp_path / 'trace.json')},
        'reason': 'referenced_instance_unavailable'}]
    assert all(node['action'] != 'fit' for node in result['nodes'])


def test_native_export_links_the_exact_fitted_candidate_and_its_paid_records(tmp_path):
    model = '_experiment_models/native_conditions_0000_0009/_evaluated_model/candidate_0118.py'
    (tmp_path / model).parent.mkdir(parents=True)
    (tmp_path / model).write_text('def predict(row): return 0.\n')
    saved = {'file': 'trace.json', 'json_path': ['questions', 0, 'native_condition_research', 0]}
    candidate = {**saved, 'json_path': [*saved['json_path'], 'candidates', 0]}
    document = {'method_results': [
        {'path': 'method.py', 'feedback': {'operator': 'native_condition_research',
            'native_condition_research': {'full_content_file': 'trace.json',
                'json_path': saved['json_path'], 'status': 'candidates_require_evaluation',
                'candidates': [{'model': model, 'program': {'op': 'const', 'value': 0.}}]}}},
        {'feedback': {'operator': 'research_choice', 'models_requested': 1, 'models_completed': 1,
            'evaluations': [{'path': model}]}}],
        'questions': [{'id': 'q', 'meaning_version': 2, 'native_condition_research': [{
            'fit_record_indices': [0], 'candidates': [{'model': model,
                'program': {'op': 'const', 'value': 0.}}]}]}]}
    (tmp_path / 'trace.json').write_text(json.dumps(document))
    (tmp_path / 'laboratory_observations.json').write_text(json.dumps([{'rows': []}]))
    segment = extract_method_segment(SimpleNamespace(workspace=tmp_path),
        root_result_ref=ref(1, field='method_results'), decision_start_ref=ref(0, field='method_results'))
    absolute = {**candidate, 'file': str(tmp_path / 'trace.json')}
    fitted = next(node for node in segment['nodes'] if node['step_ref'] == absolute)
    comparison = next(node for node in segment['nodes']
        if node['step_ref']['json_path'] == ['method_results', 1])
    assert absolute in comparison['depends_on_refs']
    assert {'file': str(tmp_path / model), 'json_path': []} in fitted['produced_refs']
    assert fitted['question_ref'] == {'question_id': 'q', 'meaning_version': 2}


def common_segment():
    call = {'file': 'trace.json', 'json_path': ['calls', 'one']}
    artifact = {'file': 'paid.json', 'json_path': [7]}
    cost = {'file': 'return.json', 'json_path': ['original_costs']}
    nodes = [{'step_ref': ref(i), 'action': action, 'is_execution': True,
        'call_ref': call, 'execution_status': 'completed', 'cost_refs': [cost],
        'depends_on_refs': dependencies} for i, action, dependencies in (
            (0, 'prepare', []), (1, 'fit', [artifact]), (2, 'research_choice', [ref(1), ref(99)]))]
    nodes.extend([{'step_ref': artifact, 'action': 'read_retained_artifact',
        'call_ref': None, 'is_execution': False, 'depends_on_refs': [ref(0)]},
        {'step_ref': ref(99), 'action': 'old_fit', 'is_execution': True,
         'call_ref': {'file': 'trace.json', 'json_path': ['calls', 'previous']}, 'depends_on_refs': []}])
    return {'root_result_ref': {**ref(2), 'json_path': ['events', 2, 'feedback']},
        'nodes': nodes, 'cost_refs': [cost], 'missing_dependencies': []}


def test_common_adapter_actually_consumes_exact_sequence_without_fabricating_step_credit(monkeypatch):
    original, calls = ResearchMethods.observe, []
    def observed(self, context, trace, **kwargs):
        calls.append((deepcopy(trace), deepcopy(kwargs)))
        return original(self, context, trace, **kwargs)
    monkeypatch.setattr(ResearchMethods, 'observe', observed)
    segment, returned = common_segment(), {'execution_complete': True,
        'gain': .4, 'observation_cost': 0, 'compute_cost': 100}
    source = {'file': 'return.json', 'json_path': ['measured_pair']}
    result = observe_common_method_segment(segment, returned, actual_return_ref=source,
        profile={'public_phase': 'research'}, source_task='measured_source')
    assert result['status'] == 'measured_segment' and len(calls) == 1
    trace, kwargs = calls[0]
    assert kwargs['step_refs'] == [ref(0), ref(1), ref(2)] and kwargs['segment_return']['actual_return_ref'] == source
    assert [step['action'] for step in trace] == ['prepare', 'fit', 'research_choice']
    assert trace[1]['depends_on_refs'] == [ref(0)]
    assert ref(99) in trace[2]['input_refs']
    assert all(step['gain'] is None and step['compute_cost'] is None for step in trace)
    method, = result['state']['methods']
    assert method['step_outcomes'] == []
    assert method['observed_gain'] == method['sequence_outcomes'][0]['gain'] == .4
    assert method['mean_compute_cost'] == 100
    assert result['state']['segment_outcomes'][0]['return']['cost_refs'] == segment['cost_refs']
    repeated = observe_common_method_segment(segment, returned, actual_return_ref=source,
        profile={'public_phase': 'research'}, source_task='measured_source', state=result['state'])
    assert repeated['status'] == 'reused_measured_segment' and len(calls) == 1
    assert repeated['state'] == result['state']
    counterexample = observe_common_method_segment(segment, {**returned, 'gain': -.2},
        actual_return_ref=source, profile={'public_phase': 'research'}, source_task='measured_source',
        state=repeated['state'])
    assert counterexample['status'] == 'updated_measured_segment' and len(calls) == 1
    assert len(counterexample['state']['segment_outcomes']) == 1
    assert counterexample['state']['methods'][0]['status'] == 'revising'
    assert counterexample['state']['methods'][0]['step_outcomes'] == []


def test_common_adapter_missing_or_censored_whole_return_does_not_train(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Missing whole return must not train a sequence')
    monkeypatch.setattr(ResearchMethods, 'observe', forbidden)
    for returned in ({}, {'execution_complete': True, 'gain': 1., 'observation_cost': 0,
                          'compute_cost': 10, 'administrative_censoring': True}):
        result = observe_common_method_segment(common_segment(), returned,
            actual_return_ref={'file': 'return.json', 'json_path': []},
            profile={'public_phase': 'research'}, source_task='source')
        assert result['status'] == 'UNKNOWN' and result['state']['methods'] == []
        assert result['cost_refs'] == common_segment()['cost_refs']
