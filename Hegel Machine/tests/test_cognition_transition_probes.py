"""Transition experiments resolve rules that agree on existing observations."""

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.probing import transition_probe
from hegel_machine.cognition.evaluation import loss


def _rule(lower, upper):
    x = Program('feature', name='control')
    def threshold(value):
        return Program('ge', children=(x, Program('const', value=value)))
    return Program('if', children=(Program('available', lag=1),
        Program('if', children=(Program('ge', children=(Program('lag', name='__target__', lag=1),
                                                   Program('const', value=.5))),
                                threshold(lower), threshold(upper))), threshold(upper)))


def _observe(program, values):
    history, rows = [], []
    for value in values:
        point = {'control': value}
        target = program.predict(Observation(point, 0., tuple(history)))
        rows.append(Observation(point, target, tuple(history)))
        history.append({**point, '__target__': target})
    return rows


def test_identical_training_predictions_split_after_preparing_both_states():
    truth, alternative = _rule(-.3, .65), _rule(-.5, .5)
    rows = _observe(truth, [-1., 0., 1., 0., -1., 1., -1.])
    assert loss(alternative, rows, 0., .5) == 0.
    spec = TaskSpec('opaque', 'Predict the response', ('control',), (), (),
                    feature_bounds={'control': (-1., 1.)}, max_probe_length=12)
    budget = SearchBudget(100)
    probe, details = transition_probe(spec, [Candidate('rule', alternative, 'relation')], rows,
                        budget=budget, remaining=12, round_number=0)
    fresh = _observe(truth, [point['control'] for point in probe.sequence])
    assert loss(alternative, fresh, 0., .5) > 0.
    assert details['policy'] == 'conditional_transition_interval'
    assert 0 < budget.spent <= 100 and probe.cost <= 12
    interior = [r for r in fresh if r.features['control'] not in (-1., 1.)]
    assert any(a.features == b.features and a.target != b.target for a in interior for b in interior)


def test_continuous_outputs_do_not_trigger_binary_transition_experiments():
    spec = TaskSpec('opaque', 'Predict the response', ('control',), (), (),
                    feature_bounds={'control': (-1., 1.)}, max_probe_length=12)
    rows = [Observation({'control': x}, x) for x in (-1., 0., 1.)]
    probe, _ = transition_probe(spec, [Candidate('rule', _rule(-.5, .5), 'relation')], rows,
                               budget=SearchBudget(100), remaining=12, round_number=0)
    assert probe is None
