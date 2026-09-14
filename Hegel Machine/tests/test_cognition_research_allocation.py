"""Research queries must support comparisons, dynamics and final calibration."""

from random import Random

from hegel_machine.cognition.controller import CognitiveAgent
from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.selection import representative_probe, warm_start_candidates


def test_each_representative_query_has_a_funded_revision_comparison():
    env = make_suite(101, 'forward')[8]
    result = CognitiveAgent(ResearchConfig(compute_budget=12000, use_memory=False)).run(env)
    queries = [a for a in result.actions if a.action == 'observe'
               and a.details['probe_id'].startswith('representative_')]
    assert len(queries) >= 2
    for query in queries:
        comparison = [a for a in result.actions if a.round == query.round and a.action == 'compare']
        assert comparison and comparison[0].details['candidates_scored'] >= 2
    assert result.compute_cost <= 12000 and result.observation_cost <= 96


def test_longer_representative_query_separates_input_delay_and_output_recursion():
    x = Program('feature', name='x')
    startup = Program('mul', children=(Program('const', value=.55), x))
    def candidate(name, term, coefficient):
        mature = Program('add', children=(startup, Program('mul', children=(Program('const', value=coefficient), term))))
        return Candidate(name, Program('if', children=(Program('available', lag=4), mature, startup)), 'representation')
    truth = candidate('input', Program('lag', name='x', lag=4), 1.2)
    alternative = candidate('output', Program('lag', name='__target__', lag=4), 1.2/.55)
    spec = TaskSpec('dynamics', 'Predict observations', ('x',), (), (),
                    feature_bounds={'x': (-2., 2.)}, max_probe_length=32)
    for adaptive in (False, True):
        probe = representative_probe(spec, seed=19, index=0, remaining=32,
                                     candidates=[truth, alternative], adaptive_horizon=adaptive)
        history, rows = [], []
        for features in probe.sequence:
            target = truth.program.predict(Observation(features, 0., tuple(history)))
            rows.append(Observation(features, target, tuple(history)))
            history.append({**features, '__target__': target})
        if adaptive:
            assert probe.cost > 8 and loss(alternative.program, rows, 0., .5) > .1
        else:
            assert probe.cost == 8 and loss(alternative.program, rows, 0., .5) < 1e-12


def test_startup_joint_coordinates_are_refitted_independently_of_mature_coefficients():
    x, z = Program('feature', name='x'), Program('feature', name='z')
    square, product = Program('mul', children=(x, x)), Program('mul', children=(x, z))
    mature = Program('add', children=(Program('add', children=(x, square)),
                       Program('add', children=(product, Program('lag', name='x', lag=2)))))
    rng, rows = Random(21), []
    for _ in range(8):
        history = []
        for index in range(5):
            features = {'x': rng.uniform(-2, 2), 'z': rng.uniform(-2, 2)}
            target = (1.2*features['x'] + .6*features['x']**2 + .9*features['x']*features['z']
                      if index < 2 else mature.predict(Observation(features, 0., tuple(history))))
            rows.append(Observation(features, target, tuple(history)))
            history.append({**features, '__target__': target})
    candidates = warm_start_candidates([Candidate('joint', mature, 'composition')], rows,
                  budget=SearchBudget(700), q=.5, complexity_weight=0.)
    assert min(loss(c.program, rows, 0., .5) for c in candidates) < 1e-12


def test_risk_calibration_trajectory_is_acquired_after_program_selection():
    result = CognitiveAgent(ResearchConfig(compute_budget=12000, max_rounds=3)).run(make_suite()[3])
    allocation = next(i for i,a in enumerate(result.actions) if a.action == 'plan_calibration')
    calibration = [i for i,a in enumerate(result.actions) if a.action == 'observe' and i > allocation]
    initial = next(a for a in result.actions if a.action == 'compare_initial_calibration')
    additional = [a for a in result.actions if a.action == 'extend_decision_comparison']
    assert len(calibration) == 2 + 2 * len(additional)
    assert [result.actions[i].observation_cost for i in calibration] == [8] * len(calibration)
    assert initial.details['trajectories'] == 2 and initial.compute_cost == 32
    assert all(a.compute_cost == 32 for a in additional)
    assert not any(a.action in ('compare','revise','rollback') for a in result.actions[calibration[0]+1:])
    assert initial.details['previous_program'] == initial.details['candidate_program'] == result.program.to_dict()
    assert initial.details['previous_model'] == {'kind': 'nominal'}
    final_comparison = additional[-1].details if additional else initial.details
    expected = initial.details['candidate_model'] if final_comparison['adopted'] else {'kind': 'nominal'}
    assert result.uncertainty_model == expected
    feedback = [a for a in result.actions if a.action == 'action_feedback' and a.details['action'] == 'calibration']
    if result.research_state['decision_frontier'].get('evidence_states'):
        assert not feedback  # conflicting decisions remain an unfinished method
    else:
        assert feedback and feedback[-1].details['reference_rows'] == 8 * len(calibration)
    assert result.compute_cost <= 12000 and result.observation_cost <= 96
