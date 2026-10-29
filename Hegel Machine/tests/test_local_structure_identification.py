"""A relative local winner must not exclude unrefuted fitted structures."""

from random import Random

from hegel_machine.cognition.conditional_coordinates import (
    propose_experimental_conditional_coordinates,
)
from hegel_machine.cognition.conditional_experiments import assess_conditional_experiment
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec


def spec():
    return TaskSpec('engineering_structure_scope', 'Predict a purchased response',
        ('u',), (), (), feature_bounds={'u': (-1., 1.)}, max_probe_length=16)


def rows(sequence, targets):
    history, result = [], []
    for value, target in zip(sequence, targets):
        point = {'u': value}
        result.append(Observation(point, target, tuple(history)))
        history.append(point)
    return result


def test_misfitting_local_winner_retains_the_structures_actually_tested():
    sequence = [-.4, -.8, -.4, 0., .4, .8, .4, -.8]
    reference = rows([-.4] * 8, [0.] * 8)
    measured = rows(sequence, [0., 1., 0., 1., 1., 0., 1., 0.])
    work = SearchBudget(50000)
    result = assess_conditional_experiment(spec(),
        {'control': 'u', 'background': {'u': -.4}, 'mode': 'paid_matched'},
        reference, measured, work)

    assert result['gain'] > 1e-10
    assert result['conditional_loss'] > 1e-10
    assert not result['response_supported'] and not result['condition_identified']
    assert result['width_candidates'] == list(range(1, 8))
    assert set(result['placement_candidates']) == {'inside_window', 'outside_window'}
    assert {(entry['width'], entry['placement']) for entry in result['competing_structures']} == {
        (width, placement) for width in range(1, 8)
        for placement in ('inside_window', 'outside_window')}
    assert 0 < work.spent <= work.limit


def joint_rows():
    # This synthetic response makes the global test distinct from the supplied
    # misfitting local nomination. No production observations are executed.
    rng = Random(27)
    result = []
    for _ in range(3):
        sequence = [rng.choice([-1., -.5, 0., .5, 1.]) for _ in range(16)]
        targets = [sum(value if current >= 0. else -.5 * value ** 2
                       for value in sequence[max(0, index - 1):index + 1])
                   for index, current in enumerate(sequence)]
        result.extend(rows(sequence, targets))
    return result


def run_joint(evidence):
    feature = Program('feature', name='u')
    work = SearchBudget(50000)
    claims = propose_experimental_conditional_coordinates(spec(), joint_rows(), work,
        basis=[feature, Program('mul', children=(feature, feature))],
        evidence=evidence if isinstance(evidence, list) else [evidence], max_candidates=8)
    assert 0 < work.spent <= work.limit
    best = min(claims, key=lambda claim: claim['training_loss'])
    step = best['research_program']['selection_steps'][-1]
    return best, step, work.spent


def test_joint_generator_actually_refits_a_non_argmin_structure_after_local_misfit():
    best, step, _ = run_joint({'control': 'u', 'gain': .04,
        'conditional_loss': .06, 'condition_identified': False,
        'width': 3, 'placement': 'inside_window', 'threshold_interval': [.4, .5],
        'width_candidates': [3], 'placement_candidates': ['inside_window'],
        'competing_structures': [
            {'width': 3, 'placement': 'inside_window', 'conditional_loss': .06},
            {'width': 2, 'placement': 'outside_window', 'conditional_loss': .08}]})

    assert best['training_loss'] < 1e-10
    assert (step['width'], step['placement']) == (2, 'outside_window')
    assert {(entry['width'], entry['placement']) for entry in step['parameter_trials']} == {
        (3, 'inside_window'), (2, 'outside_window')}
    assert step['threshold_interval'][1] <= 0.


def test_explained_local_evidence_keeps_its_identified_structure_restriction():
    best, step, _ = run_joint({'control': 'u', 'gain': .1,
        'conditional_loss': 0., 'condition_identified': True,
        'width': 2, 'placement': 'outside_window', 'threshold_interval': [-.5, 0.],
        'width_candidates': [2], 'placement_candidates': ['outside_window'],
        'competing_structures': [{'width': 3, 'placement': 'inside_window'}]})

    assert best['training_loss'] < 1e-10
    assert {(entry['width'], entry['placement']) for entry in step['parameter_trials']} == {
        (2, 'outside_window')}


def test_a_complete_supported_response_does_not_pay_for_misfitting_nominations():
    supported = {'control': 'u', 'gain': .1, 'conditional_loss': 0.,
        'condition_identified': True, 'width': 2, 'placement': 'outside_window',
        'threshold_interval': [-.5, 0.], 'width_candidates': [2],
        'placement_candidates': ['outside_window']}
    weak = {'control': 'u', 'gain': .04, 'conditional_loss': .06,
        'condition_identified': False, 'width': 3, 'placement': 'inside_window',
        'threshold_interval': [.4, .5],
        'competing_structures': [{'width': width, 'placement': placement}
            for width in range(3, 8)
            for placement in ('inside_window', 'outside_window')]}
    baseline, _, baseline_spent = run_joint(supported)
    result, _, result_spent = run_joint([supported, weak])
    assert result['candidate'].program == baseline['candidate'].program
    assert result_spent == baseline_spent
