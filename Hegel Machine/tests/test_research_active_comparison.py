"""Finite synthetic inputs check research design, without a scientific run."""

from copy import deepcopy
from types import SimpleNamespace

from hegel_machine.cognition.candidate_comparison import compare_candidate_adoption
from hegel_machine.cognition.controller import (
    CognitiveAgent, _check_active_research_predictions, _refresh_active_research_pool,
)
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.probing import choose_valuable_probe
from hegel_machine.cognition.selection import rank_candidates


def fixture():
    x = Program('feature', name='x')
    old = [Candidate('old_0', x, 'refit')]
    for index in range(1, 8):
        old.append(Candidate(f'old_{index}',
            Program('add', children=(old[-1].program, Program('const', value=0.))), 'composition'))
    square = Candidate('research_square', Program('mul', children=(x, x)), 'composition')
    observed = [Observation({'x': value, 'z': 0.}, value) for value in (0., 1.)]
    assessed = rank_candidates(old, observed, observed, fresh=observed, fallback=0., q=.5,
        budget=SearchBudget(48), complexity_weight=.001)
    public = {candidate.program.expanded(): (0.25 if candidate is not square else 0.0625,)
              for candidate in [*old, square]}
    return old, square, observed, assessed, public


def review(assessed, candidates, public, observed, fresh, budget):
    return _refresh_active_research_pool(assessed, candidates, public, observed, fresh, fresh,
        budget=budget, maximum=4, fallback=0., q=.5, complexity_weight=.001,
        minimum_improvement=ResearchConfig().minimum_improvement)


def test_reviewed_extra_reaches_actual_task_value_computation_beyond_old_first_eight():
    old, square, observed, assessed, public = fixture()
    old_task = deepcopy(assessed)
    before_work = SearchBudget(6)
    active, _, details = review(assessed, [*old, square], public, observed, observed, before_work)
    assert before_work.spent == details['compute_cost'] == 6
    assert [candidate.candidate_id for candidate in active][:2] == ['old_0', 'research_square']
    irrelevant = Probe('same_answers', ({'x': 0., 'z': 1.}, {'x': 1., 'z': -1.}))
    informative = Probe('different_answers', ({'x': .25, 'z': 0.}, {'x': .75, 'z': 0.}))
    spec = TaskSpec('test_public', 'Predict public scalar', ('x', 'z'), (), (),
                    feature_bounds={'x': (0., 1.), 'z': (-1., 1.)}, max_probe_length=2)
    legacy, legacy_details = choose_valuable_probe([irrelevant, informative], [*old, square], 0.,
        SearchBudget(4000), observations=observed, spec=spec)
    budget = SearchBudget(4000)
    selected, computed = choose_valuable_probe([irrelevant, informative], [*old, square], 0., budget,
        observations=observed, spec=spec, reviewed_research_pool=active)
    assert legacy == irrelevant and legacy_details['expected_risk_reduction'] == 0
    assert selected == informative and computed['expected_risk_reduction'] > 0
    assert 'research_square' in {row['candidate_id'] for row in computed['candidate_weights']}
    by_id = {row['probe_id']: row for row in computed['evaluated_probes']}
    assert by_id['different_answers']['expected_risk_reduction'] > by_id['same_answers']['expected_risk_reduction']
    assert budget.spent == computed['compute_cost'] > 0
    assert assessed == old_task
    arguments = {'fallback': 0., 'spec': spec, 'config': ResearchConfig()}
    assert compare_candidate_adoption(assessed, old[0], observed, [], observed, **arguments) == \
        compare_candidate_adoption(old_task, old[0], observed, [], observed, **arguments)


def test_unfunded_full_review_keeps_unknown_extra_out_of_selection_without_free_predictions(monkeypatch):
    old, square, observed, assessed, public = fixture()
    original = Program.predict
    calls = []

    def track(program, row, context=None):
        calls.append(program)
        return original(program, row, context=context)

    monkeypatch.setattr(Program, 'predict', track)
    budget = SearchBudget(5)
    active, _, details = review(assessed, [*old, square], public, observed, observed, budget)
    assert details['status'] == 'partial_unknown'
    assert details['unknown_candidate_ids'] == ['research_square']
    assert square not in active and not calls
    assert budget.spent == details['compute_cost'] == 0 and budget.denied_requests == 1


def test_actual_paid_intervention_error_removes_research_eligibility_but_retains_task_and_source():
    old, square, observed, assessed, public = fixture()
    fresh = [Observation({'x': .5}, .5)]
    before = deepcopy(assessed)
    work = SearchBudget(4)
    active, _, details = review(assessed, [*old, square], public, observed, fresh, work)
    assert square not in active and assessed == before and work.spent == 4
    failure, = details['scoped_failures']
    assert failure['candidate_id'] == square.candidate_id
    assert failure['losses']['fresh_loss'] == .0625
    assert 'fresh_loss' in failure['violated_losses']
    assert not failure['scientific_family_refuted']
    assert details['criterion'] == 'finite_research_loss_heuristic_best_plus_minimum_improvement'
    checks = _check_active_research_predictions([old[0], square], fresh, observation_offset=9,
        budget=SearchBudget(2), fallback=0., q=.5, minimum_improvement=.002)
    assert checks['evidence_ref'] == {'field': 'observations', 'row_indices': [9]}
    by_id = {row['candidate_id']: row for row in checks['checks']}
    assert by_id['old_0']['loss'] == 0
    assert by_id['research_square']['loss'] == .0625
    assert by_id['research_square']['status'] == 'outside_research_loss_tolerance'
    assert checks['compute_cost'] == 2


def test_acquired_prediction_check_exhaustion_retains_unknown_and_actual_charges():
    old, square, _, _, _ = fixture()
    budget = SearchBudget(1)
    result = _check_active_research_predictions([old[0], square], [Observation({'x': .5}, .5)],
        observation_offset=7, budget=budget, fallback=0., q=.5, minimum_improvement=.002)
    assert result['status'] == 'partial_unknown' and result['compute_cost'] == budget.spent == 1
    assert result['checks'][1]['status'] == 'UNKNOWN' and result['checks'][1]['loss'] is None


def test_value_budget_shrink_reports_reviewed_candidates_that_did_not_participate():
    old, square, observed, _, _ = fixture()
    spec = TaskSpec('small', 'Predict scalar', ('x',), (), (), feature_bounds={'x': (0., 1.)})
    _, result = choose_valuable_probe([Probe('p', ({'x': .5},))], old, 0., SearchBudget(4),
        observations=observed, spec=spec, max_compute=4, reviewed_research_pool=[old[0], square])
    assert result['status'] == 'coverage_only'
    assert result['unvalued_candidate_ids'] == ['research_square']
    assert [row['candidate_id'] for row in result['candidate_weights']] == ['old_0']


def test_full_review_order_differing_from_sample_fit_keeps_finite_normalized_weights():
    sampled = {round(index*31/15) for index in range(16)}
    observed = [Observation({'x': 0. if index in sampled else 1.}, 0.) for index in range(32)]
    first = Candidate('full_evidence_best', Program('const', value=.01), 'refit')
    second = Candidate('sample_best', Program('mul', children=(Program('const', value=.04),
                        Program('feature', name='x'))), 'composition')
    assessed = rank_candidates([first, second], observed, [], fallback=0., q=.5,
        budget=SearchBudget(64), complexity_weight=0.)
    assert assessed[0].candidate == first and assessed[1].evidence_loss < .002
    spec = TaskSpec('sample_order', 'Predict scalar', ('x',), (), (),
                    feature_bounds={'x': (0., 1.)}, max_probe_length=2)
    _, result = choose_valuable_probe([Probe('p', ({'x': .5},))], [first, second], 0.,
        SearchBudget(4000), observations=observed, spec=spec, reviewed_research_pool=[first, second])
    weights = {row['candidate_id']: row['weight'] for row in result['candidate_weights']}
    assert sum(weights.values()) == 1. and weights['sample_best'] > weights['full_evidence_best']


def test_controller_uses_reviewed_unadmitted_candidate_in_real_selection_and_checks_new_rows():
    x, z = Program('feature', name='x'), Program('feature', name='z')
    candidates = [Candidate('old_x', x, 'refit'), Candidate('old_z', z, 'refit'),
                  Candidate('research_square', Program('mul', children=(x, x)), 'composition')]

    class Proposer:
        def propose(self, *args):
            return SimpleNamespace(candidates=candidates, probes=[], usage={'calls': 0, 'total_tokens': 0})

    class Lab:
        def __init__(self):
            point = lambda value: {'x': value, 'z': value}
            self.spec = TaskSpec('controller_fixture', 'Predict scalar', ('x', 'z'),
                (Probe('initial', tuple(point(value) for value in (0., 1., 0.))),),
                (Probe('representative', tuple(point(.01) for _ in range(3))),
                 Probe('active', (point(.5),))), max_probe_length=3)
            self.cost = 0

        def observe(self, probe):
            self.cost += probe.cost
            return [Observation(dict(point), point['x']) for point in probe.sequence]

    lab = Lab()
    config = ResearchConfig(observation_budget=20, compute_budget=20000, max_rounds=2,
        max_candidates=2, proposer='codex', use_memory=False, use_abstractions=False,
        use_warm_start=False, state_search=False, risk_learning=False, transition_probes=False,
        synthesize_probes=False, joint_planning=False, transfer_scheduling=False)
    result = CognitiveAgent(config, proposer=Proposer()).run(lab)
    reviews = [action for action in result.actions if action.action == 'review_active_research_pool']
    assert reviews and any('research_square' in [row['candidate_id'] for row in
        action.details['retention']['retained']] for action in reviews)
    design = next(action for action in result.actions if action.action == 'research_comparison_pool')
    assert 'research_square' not in design.details['task_candidate_ids']
    plans = [action.details for action in result.actions if action.action == 'plan_experiment']
    assert any('research_square' in {row['candidate_id'] for row in plan.get('candidate_weights', [])}
               and plan.get('evaluated_probes') for plan in plans)
    checks = [action for action in result.actions if action.action == 'check_active_research_predictions']
    assert checks and checks[0].compute_cost > 0
    assert any(row['candidate_id'] == 'research_square' and row['loss'] == .0625
               for row in checks[0].details['checks'])
    assert result.observation_cost == lab.cost <= config.observation_budget
    assert result.compute_cost <= config.compute_budget
    assert result.llm_usage['calls'] == 0
    assert all(action.details.get('after_program') != candidates[2].program.to_dict()
               for action in result.actions if action.action == 'revise')
