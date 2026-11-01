"""Already paid confirmation narrows the experiment's actual competitors."""

from hegel_machine.cognition.competing_object_experiments import (
    assess_competing_object_probe, review_paid_program_competition,
)
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.native_condition_research import _spec
from hegel_machine.cognition.program_experiment_methods import ProgramExperimentMethods


def fixture(*, levels=None):
    public = {'controls': ['u'], 'feature_bounds': {'u': [-1., 1.]},
        'action_kinds': ['trajectory'], 'max_probe_length': 24,
        'control_levels': {'u': levels or [index / 10 for index in range(-10, 11)]}}

    def threshold(value):
        return Program('ge', children=(Program('feature', name='u'), Program('const', value=value)))

    hypotheses = []
    for index, (boundary, loss) in enumerate(((.35, 0.), (.225, 1. / 3), (.475, 0.))):
        program = Program('state_replay', children=(Program('if', children=(
            Program('available', lag=1), threshold(.55), threshold(boundary))),))
        hypotheses.append({'candidate_id': f'candidate{index}', 'program': program.to_dict(),
            'score': {'mse': loss, 'returncode': 0, 'evaluation_rows': 3},
            'score_ref': {'file': 'paid_confirmation.json', 'json_path': ['scores', index]}})
    records = [{'rows': [{'features': {'u': value}, 'history': [], 'target': target}]}
        for value, target in ((-.7, 0.), (.3, 0.), (.7, 1.))]
    arguments = {'score_scope_ref': {'file': 'paid_confirmation.json', 'json_path': ['rows']},
        'score_rows': 3, 'control_levels': public['control_levels'], 'tolerance': 1e-10}
    return _spec(public), hypotheses, records, arguments


def test_refuted_old_competitor_is_excluded_and_new_experiment_uses_real_remaining_disagreement():
    spec, hypotheses, records, arguments = fixture()
    result = review_paid_program_competition(spec, hypotheses, records, SearchBudget(6000), **arguments)
    assert result['survivor_ids'] == ['candidate0', 'candidate2']
    assert result['hypotheses'][1]['status'] == 'counterexample'
    assert not result['old_plan_reused']
    assert result['status'] == 'distinguishing_experiment_available'
    plan = result['plan']
    assert [entry['candidate_id'] for entry in plan['hypotheses']] == ['candidate0', 'candidate2']
    assert plan['probe'].sequence == ({'u': .4},)
    assert plan['hypotheses'][0]['predictions'] != plan['hypotheses'][1]['predictions']
    assert plan['prefix_design']['purchased_rows'] == 1
    assert result['new_scores'] == result['new_fits'] == result['new_observations'] == 0
    assert hypotheses[1]['score']['mse'] == 1. / 3  # source score retained


def test_only_confirmed_program_remaining_needs_no_planning_or_purchase():
    spec, hypotheses, records, arguments = fixture()
    records.append({'rows': [{'features': {'u': .4}, 'history': [], 'target': 1.}]})
    arguments['score_rows'] = 4
    for index, hypothesis in enumerate(hypotheses):
        hypothesis['score'].update(evaluation_rows=4, mse=0. if index == 0 else .25)
    result = review_paid_program_competition(spec, hypotheses, records, SearchBudget(6000), **arguments)
    assert result['status'] == 'competition_resolved_on_paid_scope'
    assert result['survivor_ids'] == ['candidate0']
    assert result['plan'] is None and result['compute_cost'] == 0


def test_surviving_programs_with_same_legal_predictions_do_not_create_purchase():
    spec, hypotheses, records, arguments = fixture(levels=[-1., 0., 1.])
    records = [{'rows': [{'features': {'u': value}, 'history': [], 'target': target}]}
        for value, target in ((-1., 0.), (0., 0.), (1., 1.))]
    hypotheses = [hypotheses[0], hypotheses[2]]
    result = review_paid_program_competition(spec, hypotheses, records, SearchBudget(6000), **arguments)
    assert result['status'] == 'no_new_distinguishing_probe'
    assert result['survivor_ids'] == ['candidate0', 'candidate2']
    assert result['plan'] is None


def test_paid_review_binds_sensors_and_does_not_repurchase_a_completed_control_prefix():
    spec = TaskSpec('anonymous_successor', 'Bound paid competitors', ('u',), (), (),
        feature_bounds={'u': (-1., 1.)}, max_probe_length=8)
    sensor = Program('feature', name='measured')
    binding = {'measured': Program('add', children=(Program('const', value=1.),
        Program('mul', children=(Program('const', value=2.), Program('feature', name='u')))))}
    hypotheses = [{'candidate_id': str(index),
        'program': Program('ge', children=(sensor, Program('const', value=cut))).to_dict(),
        'score': {'mse': 0., 'returncode': 0, 'rows': 2},
        'score_ref': {'file': 'paid_pool.json', 'json_path': ['scores', index]}}
        for index, cut in enumerate((1.7, 1.95))]
    records = [{'rows': [{'features': {'u': value}}]} for value in (-1., 1.)]
    arguments = {'score_scope_ref': {'file': 'paid_pool.json'}, 'score_rows': 2,
        'control_levels': None, 'binding': binding, 'length': 5, 'serial': 1}
    budget = SearchBudget(6000)
    result = review_paid_program_competition(spec, hypotheses, records, budget, **arguments)
    assert result['status'] == 'distinguishing_experiment_available'
    plan = result['plan']
    assert plan['hypotheses'][0]['predictions'] != plan['hypotheses'][1]['predictions']
    assert all(set(point) == {'u'} and -1. <= point['u'] <= 1. for point in plan['probe'].sequence)
    predicted_sensor = [binding['measured'].predict(Observation(point, 0.))
        for point in plan['probe'].sequence]
    assert plan['predicted_fields'] == {'measured': predicted_sensor}
    assert result['compute_cost'] == budget.spent > 0
    assert result['new_scores'] == result['new_fits'] == result['new_observations'] == 0

    measured_history, command_history, pairs = [], [], []
    truth = Program.from_dict(hypotheses[0]['program'])
    for point in plan['probe'].sequence:
        fields = {'measured': binding['measured'].predict(Observation(point, 0., tuple(command_history)))}
        target = truth.predict(Observation(fields, 0., tuple(measured_history)))
        pairs.append((Observation(point, target, tuple(command_history)),
            Observation(fields, target, tuple(measured_history))))
        command_history.append({**point, '__target__': target})
        measured_history.append({**fields, '__target__': target})
    assessment = assess_competing_object_probe(plan, pairs, SearchBudget(1000))
    methods = ProgramExperimentMethods()
    learned = methods.observe(plan, assessment, task_id=spec.task_id, command_spec=spec, binding=binding)
    assert assessment['status'] == 'distinguished' and assessment['calibration_status'] == 'supported_on_probe'
    assert learned['action'] == 'learn_program_experiment_method'
    assert len(methods.records) == 1 and methods.records[0]['support'][0]['probe_id'] == plan['probe'].probe_id
    assert methods.records[0]['source_boundary'] == plan['design']['boundary']

    # This exact reset prefix already has a measured answer, even if its
    # retained trajectory continued beyond the short distinguishing probe.
    paid = list(plan['probe'].sequence) + [{'u': -.8}]
    records.append({'rows': [{'features': point} for point in paid]})
    again = review_paid_program_competition(spec, hypotheses, records, SearchBudget(6000), **arguments)
    if again['plan'] is not None:
        following = again['plan']['probe']
        assert list(following.sequence) != paid[:following.cost]
    else:
        assert again['status'] == 'no_new_distinguishing_probe'
