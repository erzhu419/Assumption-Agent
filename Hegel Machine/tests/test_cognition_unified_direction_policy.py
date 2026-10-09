"""Callable reuse and dependent revision learn from actual measured trajectories."""

from copy import deepcopy
from dataclasses import replace
import json
from random import Random

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Probe, SearchBudget
from hegel_machine.cognition.research_graph import ResearchGraph
from hegel_machine.cognition.unified_research import UnifiedResearchAgent, _fit, _terms
from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab
from test_cognition_unified_research import CONFIG, PublicLaboratory
from test_cognition_unified_action_confirmation import _assert_fresh_comparison


def _events(report, name):
    return [event for event in report['actions'] if event['action'] == name]


def _assert_confirmation_acquisition_is_charged(report, outcome):
    acquired = next(event for event in _events(report, 'acquire')
                    if event['probe_id'] == outcome['probe_id'])
    charged = [event for event in report['actions'][:report['actions'].index(outcome)]
               if event['action'] == 'research_action_followup'
               and event['stage'] == 'frozen_confirmation'
               and event['context'] == outcome['context']
               and event['research_action'] == outcome['research_action']
               and event['trial'] == outcome['trial']]
    assert charged
    for resource in ('observation_cost', 'measurement_cost'):
        assert charged[-1][resource] == acquired[resource]
    assert charged[-1]['compute_cost'] >= acquired['observation_cost']
    # The additional baseline evaluation is charged separately from predicting
    # the selected program on this same newly acquired trajectory.
    assert outcome['compute_cost'] == acquired['observation_cost']


def _assert_actual_direction_costs(report, outcome, before, after):
    context, direction = outcome['context'], outcome['research_action']
    previous = before.get(context, {}).get(direction, {})
    current = after[context][direction]
    feedback = [event for event in _events(report, 'research_action_feedback')
                if event['context'] == context and event['research_action'] == direction]
    assert current['trials'] == previous.get('trials', 0) + len(feedback)
    paid = [event for event in report['actions']
            if event['action'] in {'research_action_feedback', 'research_action_followup',
                                   'research_action_confirmation'}
            and event.get('context') == context and event.get('research_action') == direction]
    for resource in ('observation_cost', 'measurement_cost', 'compute_cost'):
        old_cost = previous.get(f'mean_{resource}', 0.) * previous.get('trials', 0)
        observed_cost = sum(event.get(resource, 0) for event in paid)
        assert current[f'mean_{resource}'] * current['trials'] == pytest.approx(old_cost + observed_cost)


def _run(task, seed, library, config=CONFIG):
    laboratory = UnknownTaskLab(task, seed)
    report = UnifiedResearchAgent(config, library=library).run(PublicLaboratory(laboratory))
    # Evaluation is independent and occurs only after the research has ended.
    rows = laboratory.evaluation_rows(report['selected_channels'], seed=733,
                                     length=40, trajectories=2)
    measured = loss(Program.from_dict(report['program']), rows, report['fallback'], .5)
    assert report['costs']['observation_cost'] == laboratory.observation_cost
    assert report['costs']['measurement_cost'] == laboratory.measurement_cost
    assert report['costs']['observation_cost'] <= config.observation_budget
    assert report['costs']['compute_cost'] <= config.compute_budget
    return report, measured


@pytest.fixture(scope='module')
def measured_direction_chain():
    library = KnowledgeLibrary()
    source, source_error = _run(2, 113, library)
    source_saved = json.loads(json.dumps(library.to_dict()))
    restored = KnowledgeLibrary.from_dict(source_saved)
    # This budget cannot fund both the independent shortcut and recovery, so
    # the ordinary research loop must decide how to reuse the learned call.
    config = replace(CONFIG, observation_budget=160)
    successor, successor_error = _run(3, 211, restored, config)
    successor_saved = json.loads(json.dumps(restored.to_dict()))
    return source, source_error, successor, successor_error, source_saved, successor_saved, config


def test_joint_task_executes_dependent_revision_and_confirms_its_own_trial(measured_direction_chain, record_property):
    source, measured, _, _, _, _, _ = measured_direction_chain
    assert source['status'] == 'completed'
    assert measured < 1e-12
    record_property('source_costs', json.dumps(source['costs']))
    executions = _events(source, 'execute_dependent_research')
    assert any(event['answered'] for event in executions)
    feedback = [event for event in _events(source, 'research_action_feedback')
                if event['research_action'] == 'dependent_residual_revision']
    assert feedback
    confirmations = [event for event in _events(source, 'research_action_confirmation')
                     if event['research_action'] == 'dependent_residual_revision']
    assert confirmations
    for event in confirmations:
        assert event['status'] == 'supported'
        before, after = _assert_fresh_comparison(source, event)
        _assert_confirmation_acquisition_is_charged(source, event)
        assert after < 1e-12 and before > after
        origin = next(item for item in feedback if item['context'] == event['context']
                      and item['feedback']['trials'] == event['trial'])
        assert origin['feedback']['trials'] == event['feedback']['trials']


def test_selected_direction_has_real_feedback_after_json_restore(measured_direction_chain, record_property):
    source, _, successor, measured, source_saved, saved, _ = measured_direction_chain
    assert successor['status'] == 'completed'
    assert measured < 1e-12
    record_property('successor_costs', json.dumps(successor['costs']))
    assert set(successor['used_sources']) & set(source['concepts_added'])
    assert not _events(successor, 'test_learned_acquisition_method')
    confirmations = [event for event in _events(successor, 'research_action_confirmation')
                     if event['program'] == successor['program']]
    assert confirmations
    for event in confirmations:
        direction = event['research_action']
        feedback = next(item for item in _events(successor, 'research_action_feedback')
                        if item['research_action'] == direction and item['context'] == event['context']
                        and item['feedback']['trials'] == event['trial'])
        chosen = next(item for item in _events(successor, 'choose_research_action')
                      if item['selected_action'] == direction and item['context'] == event['context'])
        assert successor['actions'].index(chosen) < successor['actions'].index(feedback) < successor['actions'].index(event)
        before, after = _assert_fresh_comparison(successor, event)
        _assert_confirmation_acquisition_is_charged(successor, event)
        assert event['status'] == 'supported' and before > after
        record = saved['action_experience'][event['context']][direction]
        assert record['trial_outcomes'][str(event['trial'])]['status'] == 'supported'
        _assert_actual_direction_costs(successor, event, source_saved['action_experience'],
                                       saved['action_experience'])


def test_json_confirmed_direction_retains_its_evidence_and_actual_costs(measured_direction_chain, record_property):
    _, _, successor, _, _, saved, config = measured_direction_chain
    retained = KnowledgeLibrary.from_dict(deepcopy(saved))
    forgotten = KnowledgeLibrary.from_dict(deepcopy(saved))
    forgotten.action_experience.clear()
    improved, improved_error = _run(3, 307, retained, config)
    control, control_error = _run(3, 307, forgotten, config)
    assert improved_error < 1e-12 and control_error < 1e-12
    prior_outcome = next(event for event in _events(successor, 'research_action_confirmation')
                         if event['program'] == successor['program'])
    context, direction = prior_outcome['context'], prior_outcome['research_action']
    prior = saved['action_experience'][context][direction]
    known = next(event for event in _events(improved, 'choose_research_action')
                 if event['context'] == context and (event['selected_action'] == direction
                                                     or event.get('fallback_action') == direction))
    if known['selected_action'] == direction:
        assert known['trials'] == prior['trials']
        for resource in ('observation_cost', 'measurement_cost', 'compute_cost'):
            assert known['estimated_costs'][resource] == pytest.approx(prior[f'mean_{resource}'])
    else:
        assert known['reason'] == 'bounded_trial_before_confirmed_direction'
        for resource in ('observation_cost', 'measurement_cost', 'compute_cost'):
            assert known['fallback_costs'][resource] >= prior[f'mean_{resource}']
    control_choices = [event for event in _events(control, 'choose_research_action')
                       if event['context'] == context]
    assert control_choices[0]['reason'] == 'untried_action'
    assert control_choices[0]['trials'] == 0
    retained_record = retained.action_experience[context][direction]
    assert retained_record['trial_outcomes'][str(prior_outcome['trial'])] == (
        prior['trial_outcomes'][str(prior_outcome['trial'])])
    for report, library, initial in ((improved, retained, saved['action_experience']),
                                      (control, forgotten, {})):
        confirmed = next(event for event in _events(report, 'research_action_confirmation')
                         if event['program'] == report['program'])
        before, after = _assert_fresh_comparison(report, confirmed)
        assert confirmed['status'] == 'supported' and after < 1e-12 and before > after
        _assert_confirmation_acquisition_is_charged(report, confirmed)
        _assert_actual_direction_costs(report, confirmed, initial, library.action_experience)
    record_property('retained_costs', json.dumps(improved['costs']))
    record_property('forgotten_costs', json.dumps(control['costs']))


@pytest.mark.parametrize('comparison_rows', ['parent_fitted', 'none'])
def test_parent_fitted_to_the_comparison_rows_cannot_create_an_independent_gain(comparison_rows):
    laboratory = UnknownTaskLab(2, 113)
    rng = Random(37)
    batches = [laboratory.acquire(Probe(f'residual_evidence_{index}', tuple(
        {'u': rng.uniform(-1., 1.)} for _ in range(24))), ('sensor_a',))
        for index in range(2)]
    training, validation = [batch.observations for batch in batches]
    budget = SearchBudget(CONFIG.compute_budget)
    parent = _fit(_terms(('u',)), [*training, *validation], budget,
                  'paid_parent', q=.5)
    parent['selection_basis'] = 'all_acquired_evidence'
    library = KnowledgeLibrary()
    agent = UnifiedResearchAgent(CONFIG, library=library)
    actions = []
    fit_rows, check_rows = ((training, validation) if comparison_rows == 'parent_fitted'
                           else ([*training, *validation], []))
    claims = agent._research_view(laboratory.spec_for(('sensor_a',)), fit_rows,
        check_rows, budget, actions, [], ResearchGraph(), parent=parent,
        observation_remaining=CONFIG.observation_budget - 48, measurement_remaining=720)
    revised = [claim for claim in claims if claim['kind'] == 'dependent_residual_revision']
    assert revised and min(claim['validation_loss'] for claim in revised) < 1e-12
    assert not [event for event in actions if event['action'] == 'research_action_feedback'
                and event['research_action'] == 'dependent_residual_revision']
    assert any(event['action'] == 'research_action_pending' for event in actions)
    for claim in revised:
        origin = claim['research_action_origin']
        assert origin['action'] == 'dependent_residual_revision'
        assert origin['trial'] is None
    # Independently fitted coordinate programs may get validation feedback;
    # the parent-fitted dependent revision must remain pending.
    assert not any(record.get('dependent_residual_revision', {}).get('trials', 0)
                   for record in library.action_experience.values())


def test_new_directions_do_not_write_policy_when_memory_is_disabled(measured_direction_chain):
    _, _, _, _, source_saved, _, _ = measured_direction_chain
    library = KnowledgeLibrary.from_dict(deepcopy(source_saved))
    before = deepcopy(library.action_experience)
    report, measured = _run(3, 211, library, replace(CONFIG, use_memory=False))
    assert report['status'] == 'completed' and measured < 1e-12
    assert _events(report, 'execute_dependent_research')
    assert not any(event['action'].startswith('research_action_') for event in report['actions'])
    assert library.action_experience == before


def test_an_affordable_first_callable_fit_is_not_blocked_by_the_half_budget_cap(measured_direction_chain):
    _, _, _, _, source_saved, _, _ = measured_direction_chain
    library = KnowledgeLibrary.from_dict(deepcopy(source_saved))
    laboratory = UnknownTaskLab(3, 211)
    rng = Random(53)
    batches = [laboratory.acquire(Probe(f'quota_evidence_{index}', tuple(
        {'x': rng.uniform(-1., 1.)} for _ in range(length))), ('channel_p',))
        for index, length in enumerate([*([64] * 8), 24])]
    training = [row for batch in batches[:-1] for row in batch.observations]
    validation = batches[-1].observations
    # The affine baseline costs 2,072. Of the remaining 5,000, the first
    # recalled-call fit needs 3,072, so a blind half-budget cap cannot start it.
    config = replace(CONFIG, observation_budget=600, compute_budget=7072, max_question_depth=0)
    agent = UnifiedResearchAgent(config, library=library)
    actions, budget = [], SearchBudget(config.compute_budget)
    claims = agent._research_view(laboratory.spec_for(('channel_p',)), training, validation,
        budget, actions, [], ResearchGraph(), observation_remaining=64,
        measurement_remaining=232)
    feedback = [event for event in actions if event['action'] == 'research_action_feedback'
                and event['research_action'] == 'callable_transfer']
    assert feedback and feedback[0]['compute_cost'] == 3096
    assert feedback[0]['after_loss'] < 1e-12
    assert agent._best(claims)['kind'] == 'callable_transfer'
    assert budget.spent == 5168 <= config.compute_budget
