"""The added organization shares the old finite author and its paid budget."""

from copy import deepcopy
import json

from hegel_machine.cognition.common_object_research import (
    CommonObjectDependencyMethods, CommonObjectResearchAgent, reconstruct_partial_role_source,
)
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.object_dependency_methods import ObjectDependencyMethods
from test_v8_partial_role_experiment import scene


def source_material():
    from hegel_machine.cognition.object_dependency_methods import propose_partial_role_experiment

    spec, model, records, _ = scene()
    for row in records:
        selector = row['controls']['selector']
        row['before'].update(fa=float(selector == 0.), fb=float(selector == 1.))
    previous = propose_partial_role_experiment(spec, model, records,
        selected_values={'fa': 0., 'fb': 1.}, compute_limit=10000, remaining_rows=8)
    for i, action in enumerate(previous['actions']):
        point = action['sequence'][-1]
        records.append({'controls': point, 'before': {'qa': 1., 'qb': 1.},
            'after': {'qa': 2., 'qb': 1.}, 'target': 2. + i,
            'history': action['sequence'][:-1], 'position': len(action['sequence']) - 1,
            'probe_id': f'source:{i}'})
    return {'spec': {'feature_bounds': spec.feature_bounds,
                     'max_probe_length': spec.max_probe_length},
        'model': model, 'records': records, 'previous': previous,
        'source_ref': {'file': 'actual_source.json'}, 'model_id': 'source_method'}


def test_archive_keeps_original_methods_and_entire_old_sources(tmp_path):
    _, _, _, sources = scene()
    sources.append({'model_id': 'failed', 'status': 'counterexample',
                    'dependency_methods': [{'status': 'uninformative_pilot'}]})
    before = deepcopy(sources)
    agent = CommonObjectResearchAgent(workspace=tmp_path, organization='archive')
    budget = SearchBudget(1000)
    methods = agent._dependency_methods(sources, budget=budget, actions=[])
    assert type(methods) is ObjectDependencyMethods
    assert methods.sources == before and sources == before
    assert methods.budget is budget
    assert not agent.common_object_consumers


def test_common_consumes_whole_pair_and_charges_parent_once(tmp_path):
    spec, model, records, sources = scene()
    budget = SearchBudget(10000, spent=17)
    methods = CommonObjectDependencyMethods(tmp_path, sources, budget=budget, actions=[])
    selected = {'fa': 0., 'fb': 1.}
    start = budget.spent
    for serial in range(2):
        probe, question = methods.propose(spec, model, records, serial, selected, remaining_rows=8)
        assert probe is not None
        assert question['interpret_dependencies'] is (serial == 1)
        point = probe.sequence[-1]
        records.append({'controls': point, 'before': {'qa': 1., 'qb': 1.},
            'after': {'qa': 2., 'qb': 1.}, 'target': 2. + serial,
            'history': list(probe.sequence[:-1]), 'position': len(probe.sequence) - 1,
            'probe_id': probe.probe_id})
        methods.observe(records)
        answers = methods.common.questions[0].get('child_answers', [])
        assert bool(answers) is (serial == 1)
    assert answers[0]['answer']['status'] == 'discriminating'
    assert methods.common._active_question_id == 'task'
    assert methods.snapshot()[0]['status'] == 'discriminating'
    assert budget.spent - start == methods.common.costs['compute_cost']
    assert methods.common.costs['observation_cost'] == 0
    paid = budget.spent
    methods.observe(records)
    assert budget.spent == paid
    assert len(methods.common.questions[0]['child_answers']) == 1


def test_frozen_source_changes_common_choice_without_removing_defaults(tmp_path):
    spec, model, records, old = scene()
    learned = {'model_id': 'measured_new_source', 'source_ref': {'file': 'source.json'},
        'dependency_methods': [{'status': 'discriminating', 'excitation_role': 'quantity',
            'matched_prefix_rows': 0, 'scope': {'silent_resets_with_history_signal': False}}]}
    choices = {}
    for arm in ('fixed_pr2', 'pr3'):
        agent = CommonObjectResearchAgent(workspace=tmp_path / arm, organization=arm,
            learned_methods=[learned] if arm == 'pr3' else [])
        methods = agent._dependency_methods(old, budget=SearchBudget(10000), actions=[])
        _, question = methods.propose(spec, model, records, 0, {'fa': 0., 'fb': 1.}, remaining_rows=8)
        choices[arm] = question['research_program']['excitation_role']
        assert methods.sources[:len(old)] == old
        assert not methods.common.questions[0].get('child_answers')
    assert choices == {'fixed_pr2': 'drive', 'pr3': 'quantity'}
    assert old[0]['dependency_methods'][0]['excitation_role'] == 'drive'


def test_archive_reconstructs_raw_source_only_at_actual_membership_choice(tmp_path):
    spec, model, records, old = scene()
    source = source_material()
    path = tmp_path / 'raw_source.json'
    path.write_text(json.dumps(source))
    agent = CommonObjectResearchAgent(workspace=tmp_path, organization='archive',
        raw_sources=[{'file': str(path)}])
    budget = SearchBudget(10000)
    methods = agent._dependency_methods(old, budget=budget, actions=[])
    assert budget.spent == 0 and not agent.source_reconstruction_receipts
    methods.propose(spec, model, records, 0, {'fa': 0., 'fb': 1.}, remaining_rows=8)
    receipt = agent.source_reconstruction_receipts[0]
    assert receipt['method']['dependency_methods'][0]['status'] == 'discriminating'
    assert receipt['compute_cost'] > 0
    assert budget.spent >= receipt['compute_cost']
    assert methods.sources[:len(old)] == old
    assert len(methods.sources) == len(old) + 1
    methods.propose(spec, model, records, 1, {'fa': 0., 'fb': 1.}, remaining_rows=8)
    assert len(agent.source_reconstruction_receipts) == 1


def test_frozen_method_skips_source_reconstruction_but_uses_real_common_choice(tmp_path):
    source = source_material()
    frozen = reconstruct_partial_role_source(**source, budget=SearchBudget(10000))['method']
    agent = CommonObjectResearchAgent(workspace=tmp_path, organization='pr3',
        learned_methods=[frozen], raw_sources=[source])
    budget = SearchBudget(10000)
    spec, model, records, old = scene()
    methods = agent._dependency_methods(old, budget=budget, actions=[])
    probe, _ = methods.propose(spec, model, records, 0, {'fa': 0., 'fb': 1.}, remaining_rows=8)
    assert probe is not None
    assert not agent.source_reconstruction_receipts
    assert methods.sources[-1]['source_ref'] == source['source_ref']
    assert budget.spent == methods.common.costs['compute_cost'] > 0


def test_complete_parent_research_receives_common_answers(tmp_path):
    from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab

    config = ResearchConfig(seed=73, observation_budget=96, compute_budget=80000,
        minimum_improvement=1e-7, max_rounds=2, active_probes=True)
    agent = CommonObjectResearchAgent(config, workspace=tmp_path, organization='fixed_pr2',
        measurement_budget=1152, source_task_ref={'target': 'ordinary_parent_control'})
    with AnonymousObjectLab(objects=3, readout='focus', apparatus_seed=9013,
                            name_seed=5177, wiring_seed=313) as lab:
        result = agent.run(lab)
    assert agent.common_object_consumers
    consumer = agent.common_object_consumers[0]
    root = consumer.common.questions[0]
    assert root['status'] == 'object_research_returned'
    saved = json.loads((consumer.common.workspace / root['object_research_result_ref']['file']).read_text())
    for key in ('model_id', 'groups', 'focus_control', 'dependency_methods'):
        assert saved[key] == json.loads(json.dumps(result['object_research'][key]))
    assert root.get('child_answers')
    assert any(answer['answer']['status'] == 'discriminating' for answer in root['child_answers'])
    assert result['costs']['compute_cost'] >= consumer.common.costs['compute_cost'] > 0
    assert 'program' in result
    response = json.loads((consumer.common.workspace / root['response_result_ref']['file']).read_text())
    assert response['program'] == result['program']
    assert response['confirmation_loss'] == result['confirmation_loss']
    feedback = result['common_object_research'][0]['method_feedback']
    returned = json.loads(open(feedback['feedback_ref']['file']).read())
    assert returned['source_task_ref'] == {'target': 'ordinary_parent_control'}
    assert returned['parent_returned'] is True
    assert returned['independent_method_gain'] is None
