"""Fresh confirmation corrects the originating action's measured return."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.action_policy import ActionOption, ResearchActionPolicy
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_research import UnifiedResearchAgent, _fit, _terms
from hegel_machine.cognition.unified_session import UnifiedResearchSession
from test_cognition_program_action_followup import NestedResponseLab
from test_cognition_nested_relation_revisions import composition
from test_cognition_unified_action_followup import CONFIG, TASKS


class RecordedOrigins(UnifiedResearchAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._origin_references = []

    @property
    def origins(self):
        # A pending direction receives its trial only when a fresh comparison
        # becomes available. Keep the actual origin through that transition.
        return {(origin['context'], origin['action'], origin['trial']): deepcopy(origin)
                for origin in self._origin_references if origin['trial'] is not None}

    def _research_view(self, *args, **kwargs):
        claims = super()._research_view(*args, **kwargs)
        for claim in claims:
            origin = claim.get('research_action_origin')
            if origin and not any(origin is previous for previous in self._origin_references):
                self._origin_references.append(origin)
        return claims


@pytest.fixture(scope='module')
def failed_research():
    source = NestedResponseLab(.6)
    paid_source = source.sample(31, 64).observations
    library = KnowledgeLibrary()
    library.learn(Candidate('paid_source', composition(.6).expanded(), 'observed_response'),
                  source.spec, paid_source)
    config = ResearchConfig(observation_budget=240, compute_budget=240000,
        minimum_improvement=1e-7, max_rounds=4, active_probes=True, max_candidates=8)
    laboratory = NestedResponseLab(1.2)
    agent = RecordedOrigins(config, library=library)
    report = agent.run(laboratory)
    return report, library, laboratory, agent.origins


@pytest.fixture(scope='module')
def successful_research_chain():
    session = UnifiedResearchSession(CONFIG, tasks=TASKS, measurement_budget=4608)
    states = []
    for _ in TASKS:
        report = session.run(max_tasks=1)
        states.append(json.loads(json.dumps(report)))
        session = UnifiedResearchSession(state=states[-1])
    return states


@pytest.fixture(scope='module')
def deferred_research():
    from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab
    from test_cognition_unified_research import CONFIG as NUMERIC_CONFIG, PublicLaboratory

    class PaidParentResearch(RecordedOrigins):
        def _research_view(self, spec, training, validation, budget, actions, transfer, graph, **kwargs):
            # Exercise the supported fitted-parent input with a real paid fit.
            # The ordinary residual search and fresh confirmation are unchanged.
            parent = _fit(_terms(spec.feature_names), [*training, *validation],
                          budget, 'paid_parent', q=spec.risk_quantile)
            assert parent is not None
            parent['selection_basis'] = 'all_acquired_evidence'
            kwargs['parent'] = parent
            return super()._research_view(spec, training, validation, budget,
                                         actions, transfer, graph, **kwargs)

    laboratory = UnknownTaskLab(2, 113)
    library = KnowledgeLibrary()
    agent = PaidParentResearch(NUMERIC_CONFIG, library=library)
    report = agent.run(PublicLaboratory(laboratory))
    return report, library, laboratory, agent.origins


def _confirmations(report):
    return [event for event in report['actions'] if event['action'] == 'research_action_confirmation']


def _batch_rows(report, probe_id):
    batch = next(batch for batch in report['batches'] if batch['probe']['probe_id'] == probe_id)
    return [Observation(item['features'], item['target'], tuple(item['history']))
            for item in batch['observations']]


def _assert_fresh_comparison(report, event):
    rows = _batch_rows(report, event['probe_id'])
    assert len(rows) == event['evidence_rows']
    before = loss(Program.from_dict(event['baseline_program']), rows,
                  event['baseline_fallback'], event['risk_quantile'])
    after = loss(Program.from_dict(event['program']), rows,
                 event['program_fallback'], event['risk_quantile'])
    assert event['before_loss'] == pytest.approx(before)
    assert event['after_loss'] == pytest.approx(after)
    assert event['compute_cost'] >= len(rows)
    index = report['actions'].index(event)
    acquired = next(item for item in report['actions'][:index]
                    if item['action'] == 'acquire' and item['probe_id'] == event['probe_id'])
    assert acquired['purpose'] == 'confirm_frozen_program'
    return before, after


def _assert_object_fresh_comparison(report, event, task):
    from hegel_machine.cognition.unified_session import laboratory_for
    batch = next(item for item in report['batches'] if item['probe']['probe_id'] == event['probe_id'])
    raw = _batch_rows(report, event['probe_id'])
    with laboratory_for(task) as laboratory:
        def projected(channels):
            if channels:
                return laboratory.project_rows(raw, batch['channels'], channels)
            history, commands = [], []
            for point, row in zip(batch['probe']['sequence'], raw):
                commands.append(Observation(point, row.target, tuple(history)))
                history.append({**point, '__target__': row.target})
            return commands
        before = loss(Program.from_dict(event['baseline_program']), projected(event['baseline_channels']),
                      event['baseline_fallback'], event['risk_quantile'])
        after = loss(Program.from_dict(event['program']), projected(event['channels']),
                     event['fallback'], event['risk_quantile'])
    assert event['before_loss'] == pytest.approx(before)
    assert event['after_loss'] == pytest.approx(after)
    assert event['observation_rows'] == len(raw)
    acquired = next(item for item in report['actions'][:report['actions'].index(event)]
                    if item['action'] == 'acquire' and item['probe_id'] == event['probe_id'])
    assert acquired['purpose'] == 'confirm_frozen_program'
    return before, after


def test_paid_counterexample_revises_the_real_trial_without_inventing_fresh_losses(failed_research, record_property):
    report, library, laboratory, origins = failed_research
    initial = [event for event in report['actions'] if event['action'] == 'research_action_feedback']
    assert any(event['feedback']['last_gain'] > 0 for event in initial)
    outcomes = _confirmations(report)
    paired = [event for event in outcomes if event['before_loss'] is not None]
    assert outcomes
    record_property('research_costs', json.dumps(report['costs']))
    record_property('outcomes', json.dumps([{key: event[key] for key in ('status', 'evidence_basis',
        'evidence_rows', 'before_loss', 'after_loss', 'compute_cost', 'counterexample_loss')} for event in outcomes]))
    record_property('confirmation_loss', report['confirmation_loss'])
    record_property('action_tail', json.dumps([{key: event[key] for key in
        ('action', 'purpose', 'probe_id', 'status', 'selected_kind', 'candidate_id', 'reason') if key in event}
        for event in report['actions'][-30:]]))
    for event in paired:
        before, after = _assert_fresh_comparison(report, event)
        key = event['context'], event['research_action'], event['trial']
        origin = origins[key]
        assert event['baseline_program'] == origin['baseline_program']
        outcome = event['feedback']['trial_outcomes'][str(event['trial'])]
        assert outcome['gain'] == pytest.approx(max(-1., min(1., (before - after) / max(before, 1e-8))))
        assert outcome['status'] == event['status']
    for context, bucket in library.action_experience.items():
        for name, record in bucket.items():
            trials = [event for event in initial if event['context'] == context and event['research_action'] == name]
            assert record['trials'] == len(trials)
    if report['status'] == 'completed':
        assert report['confirmation_loss'] <= 1e-7
    refutations = [event for event in outcomes if event['evidence_basis'] == 'paid_counterexample']
    assert refutations
    for event in refutations:
        rows = _batch_rows(report, event['probe_id'])
        measured = loss(Program.from_dict(event['program']), rows, 0., event['risk_quantile'])
        assert event['counterexample_loss'] == pytest.approx(measured)
        assert measured > 1e-7
        assert event['before_loss'] is None and event['after_loss'] is None
        assert event['compute_cost'] == 0
        assert event['status'] == 'refuted'
        assert event['evidence_rows'] == len(rows)
    assert any(event['status'] in {'refuted', 'unresolved'} for event in outcomes)
    assert report['costs']['observation_cost'] == sum(len(batch['observations']) for batch in laboratory.acquisitions)
    assert report['costs']['observation_cost'] <= 240
    assert report['costs']['compute_cost'] <= 240000


def test_fitted_parent_defers_credit_and_preserves_all_costs_until_fresh_confirmation(deferred_research):
    report, library, laboratory, origins = deferred_research
    feedback = [event for event in report['actions'] if event['action'] == 'research_action_feedback']
    deferred = [event for event in feedback if event.get('cost_basis') == 'deferred_trial_total']
    assert deferred and report['status'] == 'completed'
    for event in deferred:
        assert event['research_action'] == 'dependent_residual_revision'
        pending = next(item for item in report['actions']
                       if item['action'] == 'research_action_pending'
                       and item['pending_id'] == event['pending_id'])
        paid = [pending, *[item for item in report['actions'][:report['actions'].index(event)]
            if item['action'] == 'research_action_pending_cost'
            and item['pending_id'] == event['pending_id']]]
        assert pending['trial'] is None
        assert pending['evidence_basis'] == 'reference_fitted_comparison'
        assert len(paid) > 1
        for resource in ('observation_cost', 'measurement_cost', 'compute_cost'):
            assert event[resource] == sum(item[resource] for item in paid)
        outcome = next(item for item in _confirmations(report) if item['context'] == event['context']
                       and item['research_action'] == event['research_action']
                       and item['trial'] == event['feedback']['trials'])
        before, after = _assert_fresh_comparison(report, outcome)
        assert before > after and after < 1e-12
        assert outcome['baseline_program'] == origins[
            outcome['context'], outcome['research_action'], outcome['trial']]['baseline_program']
        assert report['actions'].index(pending) < report['actions'].index(event)
        assert report['actions'].index(event) < report['actions'].index(outcome)
        record = library.action_experience[event['context']][event['research_action']]
        assert record['trials'] == 1
        assert record['mean_compute_cost'] == event['compute_cost'] + outcome['compute_cost']
        assert record['trial_outcomes']['1']['status'] == 'supported'
    assert all(event.get('cost_basis') == 'deferred_trial_total' for event in feedback)
    assert report['costs']['observation_cost'] == laboratory.observation_cost
    assert report['costs']['measurement_cost'] == laboratory.measurement_cost


def test_successful_directions_keep_their_fresh_support_and_correct_trial_owner(successful_research_chain):
    final = successful_research_chain[-1]
    records = final['knowledge']['action_experience']
    supported = []
    for result in final['results']:
        report = result['research']
        assert report['status'] == 'completed'
        for event in _confirmations(report):
            if event['before_loss'] is not None:
                before, after = _assert_fresh_comparison(report, event)
                if event['status'] == 'supported':
                    assert after <= CONFIG.minimum_improvement
                    assert before > after
                    supported.append(event)
            source = next(item for item in report['actions']
                if item['action'] == 'research_action_feedback' and item['context'] == event['context']
                and item['research_action'] == event['research_action']
                and item['feedback']['trials'] == event['trial'])
            assert report['actions'].index(source) < report['actions'].index(event)
            assert event['feedback']['trials'] >= event['trial']
        for event in report['actions']:
            if event['action'] != 'object_research_feedback' or event['before_loss'] is None:
                continue
            before, after = _assert_object_fresh_comparison(report, event, final['tasks'][result['task_position']])
            assert event['feedback']['trials'] >= event['trial']
            if event['status'] == 'supported':
                assert event['confirmed'] and after <= CONFIG.minimum_improvement
                if before > after:
                    supported.append({**event, 'research_action': event['direction']})
    assert supported
    for context, bucket in records.items():
        for action, record in bucket.items():
            originals = [event for result in final['results'] for event in result['research']['actions']
                if event['action'] == 'research_action_feedback' and event['context'] == context
                and event['research_action'] == action]
            inner_trials = {event['feedback']['trials'] for event in originals}
            outer_trials = {event['trial'] for result in final['results'] for event in result['research']['actions']
                if event['action'] == 'object_research_feedback' and event['context'] == context
                and event['direction'] == action and event['trial'] is not None}
            # Inner initial feedback creates its trial once. Object recovery
            # creates its own trial on fresh response evidence; later feedback
            # may revise that trial but cannot claim an inner trial as its own.
            assert len(originals) == len(inner_trials)
            assert not (inner_trials & outer_trials)
            assert inner_trials | outer_trials == set(range(1, record['trials'] + 1))
    # Earlier serialized sessions stay independent of later support revisions.
    restored = UnifiedResearchSession(state=json.loads(json.dumps(final)))
    assert restored.library.action_experience == records
    for event in supported:
        latest = records[event['context']][event['research_action']]['trial_outcomes'][str(event['trial'])]
        assert latest['status'] == 'supported'
        assert latest['gain'] > 0


def test_unresolved_real_outcome_survives_json_and_does_not_suppress_untried_work(failed_research):
    report, library, laboratory, _ = failed_research
    outcomes = [event for event in _confirmations(report) if event['status'] in {'refuted', 'unresolved'}]
    assert outcomes
    last = outcomes[-1]
    # Replay this actual decision state. Later research may legitimately
    # resolve the same trial; its final state is no longer this counterexample.
    snapshot = library.to_dict()
    snapshot['action_experience'] = {last['context']: {last['research_action']: last['feedback']}}
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(snapshot)))
    record = restored.action_experience[last['context']][last['research_action']]
    assert record['trials'] == last['trial']
    assert record['trial_outcomes'][str(last['trial'])]['status'] != 'supported'
    rows = laboratory.acquisitions[0]['observations']
    policy = ResearchActionPolicy(rows, laboratory.spec, restored.action_experience)
    condition = last['context'][len(policy.base_context) + 1:]
    policy.set_condition(condition)
    picked = policy.choose([ActionOption(last['research_action'], 0, 1),
                            ActionOption('new_research_direction', 0, 1)],
                           observation_remaining=240, compute_remaining=240000)
    assert picked.name == 'new_research_direction'
    assert policy.last_decision['reason'] == 'untried_action'


def test_no_memory_confirmation_does_not_revise_retained_action_experience(successful_research_chain):
    library = KnowledgeLibrary.from_dict(deepcopy(successful_research_chain[-1]['knowledge']))
    before = deepcopy(library.action_experience)
    with UnifiedObjectLab(objects=2, seed=TASKS[0]['seed'], apparatus_seed=TASKS[0]['apparatus_seed'],
            readout='focus', actuator_curvature=.12) as laboratory:
        report = UnifiedResearchAgent(replace(CONFIG, use_memory=False),
            library=library, measurement_budget=4608).run(laboratory)
    assert any(event['action'] == 'validate_frozen_program' for event in report['actions'])
    assert not _confirmations(report)
    assert library.action_experience == before


def test_supported_direction_does_not_block_a_new_structural_answer(successful_research_chain):
    from hegel_machine.cognition.models import SearchBudget
    from hegel_machine.cognition.research_graph import ResearchGraph
    from hegel_machine.cognition.structure_lab import StructureLab
    library = KnowledgeLibrary.from_dict(deepcopy(successful_research_chain[-1]['knowledge']))
    remembered = deepcopy(library.action_experience)
    # The remembered programs were learned on two objects. All four current
    # objects take part in this newly purchased local interaction response.
    laboratory = StructureLab(objects=4, readout='focus', seed=91)
    training = laboratory.acquire(laboratory.random_probe(11, length=24))
    validation = laboratory.acquire(laboratory.random_probe(41, length=24))
    agent = UnifiedResearchAgent(CONFIG, library=library)
    actions, budget = [], SearchBudget(CONFIG.compute_budget)
    # This learned direction belongs to research with an existing response.
    # Supply an actual current fit, rather than reusing its context by fiat.
    parent = _fit(_terms(laboratory.spec.feature_names), training.observations,
                  budget, 'paid_current_parent', q=laboratory.spec.risk_quantile)
    claims = agent._research_view(laboratory.spec, training.observations,
        validation.observations, budget, actions, [], ResearchGraph(),
        parent=parent, transitions=training.transitions, observation_remaining=CONFIG.observation_budget - 48,
        measurement_remaining=4608 - laboratory.measurement_cost)
    choices = [event for event in actions if event['action'] == 'choose_research_action']
    feedback = [event for event in actions if event['action'] == 'research_action_feedback']
    first = choices[0]
    bounded = first['reason'] == 'bounded_trial_before_confirmed_direction'
    prior_context = first['fallback_context'] if bounded else first['context']
    prior_action = first['fallback_action'] if bounded else first['selected_action']
    prior_experience = remembered[prior_context]
    chosen_record = prior_experience[prior_action]
    remembered_outcomes = chosen_record['trial_outcomes']
    supported = remembered_outcomes[str(chosen_record['trials'])]
    assert supported['status'] == 'supported' and supported['gain'] > 0
    assert chosen_record['expected_gain'] > 0
    if bounded:
        assert first['trials'] == 0 and first['expected_gain'] is None
        assert not remembered.get(first['context'], {}).get(first['selected_action'], {}).get('trials')
        measured_costs = ResearchActionPolicy._estimated_costs(chosen_record)
        for resource in ('observation', 'measurement', 'compute'):
            fallback = first['fallback_costs'][f'{resource}_cost']
            trial = (max(first['quoted_compute_cost'], first['bounded_compute_limit'])
                     if resource == 'compute' else first[f'quoted_{resource}_cost'])
            assert fallback >= measured_costs[f'{resource}_cost']
            assert trial <= fallback
            assert trial + fallback <= first[f'{resource}_remaining']
    else:
        assert first['expected_gain'] == pytest.approx(chosen_record['expected_gain'])
        assert first['trials'] == chosen_record['trials']
        has_untried = any(not prior_experience.get(name, {}).get('trials') for name in first['eligible_actions'])
        assert first['reason'] == ('confirmed_direction_before_exploration' if has_untried
                                   else 'measured_improvement_per_resource')
    prior = next(event for event in feedback if event['research_action'] == first['selected_action']
                 and event['context'] == first['context'])
    if bounded:
        assert prior['compute_cost'] <= first['bounded_compute_limit']
    revised = next(event for event in feedback if event['research_action'] == 'structural_research')
    assert prior['after_loss'] > CONFIG.minimum_improvement
    assert revised['after_loss'] < prior['after_loss']
    assert feedback.index(revised) > feedback.index(prior)
    assert agent._best(claims)['validation_loss'] < CONFIG.minimum_improvement
    assert budget.spent <= CONFIG.compute_budget
