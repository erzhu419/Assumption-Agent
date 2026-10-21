"""The native active caller uses the same finite PR2 selector, without science runs."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.probing import choose_question_probe
from hegel_machine.cognition.question_selection import select_question_plan


POLICY = {'kind': 'seeded_legal', 'seed': 0}
ROLES = {'x': 'public_control', 'z': 'public_control', 'response': 'scalar_response'}


def action(probe):
    return {'kind': 'trajectory', 'sequence': list(deepcopy(probe.sequence)), 'channels': []}


def readout(probe, context):
    return {'op': 'Readout', 'record_index': 0, 'step': 0, 'units': 'V', 'context_ref': context,
            'binding': {'action': action(probe), 'features': dict(probe.sequence[0]), 'history': []}}


def contract(spec, probes, *, task_probe=None):
    context = {'task_id': spec.task_id, 'output_unit': 'V'}
    return {'question': {'id': 'registered_task_query', 'context_ref': context,
                'query_semantics': {'mode': 'model_query',
                    'query_expression': readout(task_probe or probes[0], context)},
                'answer_resolution_ref': {'atol': 0.}},
            'role_signature': deepcopy(ROLES), 'output_unit': 'V',
            'exploration_policy': deepcopy(POLICY),
            'experiments': [{'id': probe.probe_id, 'actions': [action(probe)],
                'query': readout(probe, context), 'tolerance': 0., 'observation_cost': probe.cost}
                for probe in probes]}


def fixture():
    x = Program('feature', name='x')
    candidates = [Candidate('linear', x, 'refit'),
                  Candidate('square', Program('mul', children=(x, x)), 'composition')]
    probes = [Probe('half', ({'x': .5, 'z': .5},)), Probe('three_quarters', ({'x': .75, 'z': .75},))]
    spec = TaskSpec('native_public', 'Predict scalar', ('x', 'z'), (), tuple(probes), max_probe_length=1)
    return spec, probes, candidates, contract(spec, probes)


def test_native_probe_identity_and_seeded_tie_match_the_common_selector_with_actual_compute():
    spec, probes, candidates, registered = fixture()
    alias = deepcopy(registered['experiments'][0])
    alias['id'] = 'same_physical_action_alias'
    registered['experiments'].append(alias)
    original = deepcopy((registered, candidates, probes))
    work = SearchBudget(20)
    selected, result = choose_question_probe(probes, candidates, 0., work, spec=spec,
        selection_contract=registered, observation_budget=2)

    def public_rows(value):
        return [Observation(dict(point), 0., tuple(value['sequence'][:index]))
                for index, point in enumerate(value['sequence'])]

    theories = [{'id': f'native:{index}:{candidate.candidate_id}', 'program': candidate.program.to_dict(),
        'role_signature': ROLES, 'output_unit': 'V', 'context_ref': registered['question']['context_ref'],
        'support': {'status': 'candidate', 'evidence': []}} for index, candidate in enumerate(candidates)]
    common = select_question_plan(registered['question'], theories, registered['experiments'],
        context={'context_ref': registered['question']['context_ref'], 'output_unit': 'V'},
        compute_budget=20, observation_budget=2, public_prediction_rows=public_rows, exploration_policy=POLICY)
    assert selected is probes[1]
    assert result['status'] == common['status'] == 'tied'
    assert result['scores'] == common['scores'] and result['tied_experiments'] == common['tied_experiments']
    assert result['operational_decision'] == common['operational_decision']
    assert result['operational_decision']['physical_option_count'] == 2
    assert work.spent == result['compute_cost'] == common['compute_cost'] == 4
    assert result['selected_experiment']['actions'] == [action(selected)]
    assert not result['support_updated'] and not result['evidence']
    assert (registered, candidates, probes) == original


@pytest.mark.parametrize('missing', ['role_signature', 'output_unit'])
def test_missing_scientific_contract_is_author_work_without_predictions_or_acquisition(missing):
    spec, probes, candidates, registered = fixture()
    registered.pop(missing)
    work = SearchBudget(20)
    selected, result = choose_question_probe(probes, candidates, 0., work, spec=spec,
        selection_contract=registered, observation_budget=2)
    assert selected is None and result['status'] == 'author_work_request' and work.spent == 0


def test_unknown_public_sensor_rows_are_not_filled_with_model_or_generator_labels():
    spec, probes, candidates, registered = fixture()
    spec = TaskSpec(spec.task_id, spec.description, ('x', 'z', 'sensor'), (), tuple(probes),
                    feature_bounds={'x': (0., 1.), 'z': (0., 1.)})
    selected, result = choose_question_probe(probes, candidates, 0., SearchBudget(20), spec=spec,
        selection_contract=registered, observation_budget=2)
    assert selected is None and result['status'] == 'author_work_request'
    assert result['reason'] == 'native_pr2_public_measurement_constructor_required'


def test_retained_readout_without_a_future_public_action_is_unresolved_before_matching():
    spec, probes, candidates, registered = fixture()
    registered['question']['query_semantics']['query_expression'].pop('binding')
    work = SearchBudget(20)
    selected, result = choose_question_probe(probes, candidates, 0., work, spec=spec,
        selection_contract=registered, observation_budget=2)
    assert selected is None and result['status'] == 'scoped_unknown'
    assert result['reason'] == 'native_pr2_task_public_row_mismatch'
    assert result['compute_cost'] == work.spent == 0


@pytest.mark.parametrize('changed', ['task_binding', 'action', 'quote', 'unit'])
def test_public_actions_quotes_task_bindings_and_units_keep_their_hard_contract(changed):
    spec, probes, candidates, registered = fixture()
    if changed == 'task_binding':
        registered['question']['query_semantics']['query_expression']['binding']['features']['x'] = -9.
    elif changed == 'unit':
        registered['question']['query_semantics']['query_expression']['units'] = 'wrong_unit'
    else:
        for plan in registered['experiments']:
            if changed == 'action':
                plan['actions'][0]['sequence'][0]['x'] = -9.
            else:
                plan['observation_cost'] = 0
    work = SearchBudget(20)
    selected, result = choose_question_probe(probes, candidates, 0., work, spec=spec,
        selection_contract=registered, observation_budget=2)
    assert selected is None and result['status'] == 'scoped_unknown'
    assert result['compute_cost'] == work.spent == 0


def test_selection_budget_stays_unknown_instead_of_borrowing_or_running_another_probe():
    spec, probes, candidates, registered = fixture()
    work = SearchBudget(1)
    selected, result = choose_question_probe(probes, candidates, 0., work, spec=spec,
        selection_contract=registered, observation_budget=2)
    assert selected is None and work.spent == result['compute_cost'] == 1
    assert result['prediction_coverage']['complete'] is False


def run_controller(registered=True, *, task_initial=False, missing_context=False, offset=0.):
    x, z = Program('feature', name='x'), Program('feature', name='z')
    candidates = [Candidate('old_x', x, 'refit'), Candidate('old_z', z, 'refit'),
                  Candidate('research_square', Program('mul', children=(x, x)), 'composition')]

    class Proposer:
        def propose(self, *args):
            return SimpleNamespace(candidates=candidates, probes=[], usage={'calls': 0, 'total_tokens': 0})

    class Lab:
        def __init__(self):
            point = lambda value: {'x': value, 'z': value}
            self.initial = Probe('initial', tuple(point(value) for value in (0., 1., 0.)))
            self.panel = (Probe('representative', tuple(point(.01) for _ in range(3))),
                          Probe('half', (point(.5),)), Probe('three_quarters', (point(.75),)))
            self.spec = TaskSpec('controller_fixture', 'Predict scalar', ('x', 'z'),
                (self.initial,), self.panel, max_probe_length=3)
            self.queries = []

        def observe(self, probe):
            self.queries.append(probe)
            return [Observation(dict(point), point['x']+offset) for point in probe.sequence]

    lab = Lab()
    config = ResearchConfig(observation_budget=20, compute_budget=20000, max_rounds=2,
        max_candidates=2, proposer='codex', use_memory=False, use_abstractions=False,
        use_warm_start=False, state_search=False, risk_learning=False, transition_probes=False,
        synthesize_probes=False, joint_planning=False, transfer_scheduling=False)
    settings = contract(lab.spec, lab.panel[1:], task_probe=lab.initial if task_initial else None)
    if missing_context:
        settings['question'].pop('context_ref')
    result = CognitiveAgent(config, proposer=Proposer()).run(lab,
        research_selection=settings if registered else None)
    return lab, result


def test_actual_controller_keeps_old_value_path_and_both_registered_groups_use_seeded_pr2():
    old_lab, old = run_controller(False)
    b_lab, b = run_controller()
    c_lab, c = run_controller()
    assert [probe.probe_id for probe in old_lab.queries] == ['initial', 'representative', 'half']
    assert [probe.probe_id for probe in b_lab.queries] == [probe.probe_id for probe in c_lab.queries] == \
        ['initial', 'representative', 'three_quarters']
    for lab, result in ((b_lab, b), (c_lab, c)):
        planned = next(action for action in result.actions if action.action == 'plan_experiment')
        choice = planned.details['question_selection']
        assert choice['status'] == 'tied' and choice['tied_experiments'] == ['half', 'three_quarters']
        assert choice['selected_experiment']['actions'] == [action(lab.queries[-1])]
        assert choice['compute_cost'] == 4
        assert choice['native_execution']['status'] == 'executed'
        assert choice['native_execution']['probe_id'] == lab.queries[-1].probe_id
        assert choice['native_execution']['observation_cost'] == lab.queries[-1].cost
        checked = next(action for action in result.actions if action.action == 'check_active_research_predictions')
        assert checked.compute_cost == 2
        assert any(row['candidate_id'] == 'research_square' and row['loss'] == .03515625
                   for row in checked.details['checks'])
        assert result.program == old.program and result.fallback == old.fallback
        assert result.observation_cost == 7 and result.compute_cost <= 20000
        assert result.llm_usage['calls'] == 0 and choice['support_updated'] is False


@pytest.mark.parametrize('settings', [{'task_initial': True}, {'missing_context': True}])
def test_real_unknown_or_zero_g_does_not_fall_back_to_unregistered_representative(settings):
    lab, result = run_controller(**settings)
    assert [probe.probe_id for probe in lab.queries] == ['initial', 'representative']
    plan = next(action for action in result.actions if action.action == 'plan_experiment')
    assert plan.details['question_selection']['status'] in {'scoped_unknown', 'author_work_request'}
    assert plan.details['question_selection'].get('selected_experiment') is None
    assert plan.details['question_selection']['native_execution']['status'] == 'not_prepared'
    assert result.observation_cost == 6 and result.llm_usage['calls'] == 0


def test_task_policy_refit_keeps_a_prepared_pr2_experiment_distinct_from_executed_work(monkeypatch):
    from hegel_machine.cognition.action_policy import ResearchActionPolicy

    original = ResearchActionPolicy.choose

    def prefer_existing_refit(policy, options, **kwargs):
        refit = next((option for option in options if option.name == 'refit'), None)
        if refit is not None:
            policy.last_decision = {'selected_action': 'refit', 'reason': 'fixture_existing_task_work'}
            return refit
        return original(policy, options, **kwargs)

    monkeypatch.setattr(ResearchActionPolicy, 'choose', prefer_existing_refit)
    _, result = run_controller(offset=.05)
    planned = next(action for action in result.actions if action.action == 'plan_experiment')
    choice = planned.details['question_selection']
    assert choice['selected_experiment'] and choice['query_arguments']
    assert choice['native_execution'] == {'status': 'prepared_not_executed',
                                          'actual_action': 'refit', 'observation_cost': 0}
    assert result.compute_cost <= 20000 and result.llm_usage['calls'] == 0
