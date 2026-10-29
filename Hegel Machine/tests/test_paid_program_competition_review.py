"""Already paid confirmation narrows the experiment's actual competitors."""

from hegel_machine.cognition.competing_object_experiments import review_paid_program_competition
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.native_condition_research import _spec


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
