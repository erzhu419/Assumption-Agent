"""Actual research costs guide routes without erasing history-dependent answers."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.action_policy import ResearchActionPolicy
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import MeasurementBatch
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation
from test_cognition_nonlinear_discovery_research import (
    CONFIG, SignedCompositionLab, confirmation, fresh_error, research, restore,
)
from test_cognition_unified_direction_policy import _assert_confirmation_acquisition_is_charged
from test_cognition_unified_action_confirmation import _assert_fresh_comparison


class RouteResponseLab(SignedCompositionLab):
    def __init__(self, episode, mode, gain, names=('drive', 'load')):
        super().__init__(episode, gain=gain, names=names)
        self._mode = mode

    def acquire(self, probe, channels=()):
        assert not channels
        rows, history, inner_values = [], [], []
        for point in probe.sequence:
            inner = .8 * point[self._names[0]] ** 2 + self._gain * point[self._names[1]]
            inner_values.append(inner)
            window = sum(inner_values[-3:])
            target = {'inner': inner, 'square': .2 + 1.1 * inner ** 2,
                      'novel_square': .2 + 1.1 * inner ** 2 + .7 * point[self._names[0]] ** 3,
                      'window': window, 'window_square': .2 + 1.1 * window ** 2}[self._mode]
            rows.append(Observation(dict(point), target, tuple(history)))
            history.append({**point, '__target__': target})
        self.acquisitions.append({'probe': probe, 'observations': rows})
        return MeasurementBatch(rows, len(rows), 0, ())


def learned_chain(stages, names):
    library, reports, labs, snapshots = KnowledgeLibrary(), [], [], []
    assert not library.concepts
    for index, (mode, gain) in enumerate(stages):
        lab = RouteResponseLab(f'route_episode_{index}', mode, gain, names)
        reports.append(research(lab, library))
        labs.append(lab)
        snapshots.append(deepcopy(library.to_dict()))
        library = restore(library)
    return library, reports, labs, snapshots


@pytest.fixture(scope='module')
def static_routes():
    library, reports, labs, snapshots = learned_chain(
        [('inner', -.65), ('square', -.65), ('square', .35)], ('drive', 'load'))
    errors = [fresh_error(report, lab, 70101 + index * 1009)
              for index, (report, lab) in enumerate(zip(reports, labs))]
    return locals()


@pytest.fixture(scope='module')
def replay_routes(static_routes):
    # A newly appearing cubic term cannot be obtained by changing the old
    # square's coefficients. This task actually needs new response knowledge.
    library = KnowledgeLibrary.from_dict(deepcopy(static_routes['snapshots'][1]))
    prior_lab = RouteResponseLab('new_coordinate_episode', 'novel_square', .35)
    prior_report = research(prior_lab, library)
    snapshot = deepcopy(library.to_dict())
    retained = restore(library)
    forgotten = restore(library)
    forgotten.action_experience.clear()
    next_labs = [RouteResponseLab('next_route_episode', 'novel_square', -.25) for _ in range(2)]
    next_reports = [research(lab, knowledge)
                    for lab, knowledge in zip(next_labs, (retained, forgotten))]
    # Both policies finish before either receives any evaluation response.
    prior_error = fresh_error(prior_report, prior_lab, 73291)
    next_errors = [fresh_error(report, lab, 79121)
                   for report, lab in zip(next_reports, next_labs)]
    return locals()


@pytest.fixture(scope='module')
def history_routes():
    library, reports, labs, snapshots = learned_chain(
        [('inner', -.65), ('window', -.65), ('window_square', -.65),
         ('window_square', .35)], ('rate', 'stress'))
    errors = [fresh_error(report, lab, 81211 + index * 1009)
              for index, (report, lab) in enumerate(zip(reports, labs))]
    return locals()


def test_static_change_confirms_the_direction_that_actually_explains_the_change(static_routes):
    reports = static_routes['reports']
    assert all(report['status'] == 'completed' for report in reports)
    assert all(error < CONFIG.minimum_improvement for error in static_routes['errors'])
    changed = reports[-1]
    if changed['selected_kind'] == 'research_program':
        steps = changed['research_program']['selection_steps']
        assert any(step['operation'] in {'compose_measured_coordinates', 'refit_research_program'}
                   for step in steps)
    else:
        assert changed['selected_kind'] == 'mechanism_parameter_refit'
        assert changed['mechanism_revision']['parameter_revisions']
    outcomes = [action for action in changed['actions']
                if action['action'] == 'research_action_confirmation'
                and action['research_action'] in {'coordinate_reconstruction', 'program_replay',
                                                  'program_revision', 'mechanism_revision'}]
    assert outcomes
    for outcome in outcomes:
        assert outcome['status'] == 'supported'
        before, after = _assert_fresh_comparison(changed, outcome)
        assert before > after and after < CONFIG.minimum_improvement
        _assert_confirmation_acquisition_is_charged(changed, outcome)
    validated, _ = confirmation(changed)
    assert validated['loss'] < CONFIG.minimum_improvement
    assert changed['concepts_added']


def test_history_dependent_change_remains_researchable_after_static_route_is_available(history_routes, record_property):
    reports = history_routes['reports']
    assert all(report['status'] == 'completed' for report in reports)
    assert all(error < CONFIG.minimum_improvement for error in history_routes['errors'])
    changed = reports[-1]
    record_property('history_route_kind', changed['selected_kind'])
    record_property('history_route_costs', json.dumps(changed['costs']))
    assert Program.from_dict(changed['program']).history_depth >= 2
    validated, acquired = confirmation(changed)
    assert validated['loss'] < CONFIG.minimum_improvement
    assert any(action['action'] in {'propose_mechanism_revisions', 'execute_research_programs'}
               for action in changed['actions'][:changed['actions'].index(acquired)])
    original_sources = {identifier for report in reports[:3] for identifier in report['concepts_added']}
    assert original_sources & set(changed['used_sources'])
    if changed['selected_kind'] == 'mechanism_parameter_refit':
        tested = next(action for action in changed['actions']
                      if action['action'] == 'test_mechanism_predictions')
        batch = next(item for item in history_routes['labs'][-1].acquisitions
                     if item['probe'].probe_id == tested['probe_id'])
        outcomes = {}
        for alternative, hypothesis, outcome in zip(tested['alternatives'],
                tested['plan']['hypotheses'], tested['evidence']['hypotheses']):
            assert outcome['measured_loss'] == pytest.approx(loss(
                Program.from_dict(hypothesis['program']), batch['observations'], 0., .5))
            outcomes.setdefault(alternative['kind'], []).append(outcome)
        assert outcomes['retained_mechanism'][0]['status'] == 'counterexample'
        assert any(item['status'] == 'supported_on_probe'
                   for item in outcomes['mechanism_parameter_refit'])
        assert tested['evidence']['compute_cost'] == len(batch['observations']) * len(tested['plan']['hypotheses'])
        assert changed['actions'].index(tested) < changed['actions'].index(acquired)
        assert acquired['probe_id'] != tested['probe_id']
    for definition in history_routes['snapshots'][2]['definitions']:
        assert definition in history_routes['snapshots'][3]['definitions']


def test_new_replay_knowledge_reopens_its_trial_without_erasing_paid_experience(replay_routes, record_property):
    retained, forgotten = replay_routes['next_reports']
    record_property('retained_costs', json.dumps(retained['costs']))
    record_property('forgotten_costs', json.dumps(forgotten['costs']))
    assert all(report['status'] == 'completed' for report in (retained, forgotten))
    assert all(error < CONFIG.minimum_improvement for error in replay_routes['next_errors'])
    assert replay_routes['prior_report']['status'] == 'completed'
    assert replay_routes['prior_error'] < CONFIG.minimum_improvement
    stored = replay_routes['snapshot']['action_experience']
    prior_report = replay_routes['prior_report']
    old_choice = next(action for action in prior_report['actions']
        if action['action'] == 'choose_research_action'
        and action['selected_action'] == 'program_replay')
    old_replay = stored[old_choice['context']]['program_replay']
    assert old_replay['trials'] > 0 and old_replay['expected_gain'] < 1.
    original_confirmation = next(action for action in prior_report['actions']
        if action['action'] == 'research_action_confirmation'
        and action['research_action'] == 'coordinate_reconstruction' and action['status'] == 'supported')
    old_coordinate = stored[original_confirmation['context']]['coordinate_reconstruction']

    chosen = next(action for action in retained['actions']
        if action['action'] == 'choose_research_action' and action['selected_action'] == 'program_replay')
    assert chosen['trials'] == 0 and chosen['context'] != old_choice['context']
    after = replay_routes['retained'].action_experience
    new_replay = after[chosen['context']]['program_replay']
    assert new_replay['knowledge_applicability'] != old_replay['knowledge_applicability']
    assert after[old_choice['context']]['program_replay'] == old_replay
    assert after[original_confirmation['context']]['coordinate_reconstruction'] == old_coordinate
    assert retained['research_program']['reused_procedure']
    assert set(prior_report['concepts_added']) & set(retained['method_source_ids'])
    outcome = next(action for action in retained['actions']
        if action['action'] == 'research_action_confirmation' and action['context'] == chosen['context'])
    assert outcome['research_action'] == 'program_replay' and outcome['status'] == 'supported'
    before, after_loss = _assert_fresh_comparison(retained, outcome)
    assert before > after_loss and after_loss < CONFIG.minimum_improvement
    _assert_confirmation_acquisition_is_charged(retained, outcome)
    for cost in ('observation_cost', 'measurement_cost', 'compute_cost'):
        assert retained['costs'][cost] <= forgotten['costs'][cost]
    if chosen.get('fallback_action') == 'coordinate_reconstruction':
        assert chosen['fallback_costs'] == pytest.approx(ResearchActionPolicy._estimated_costs(old_coordinate))
    record_property('retained_choice', json.dumps(chosen))
    record_property('previous_replay_choice', json.dumps(old_choice))
    record_property('forgotten_first_choice', json.dumps(next(action for action in forgotten['actions']
        if action['action'] == 'choose_research_action')))
