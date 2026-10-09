"""A research direction pays for the real experiments its selected result causes."""

from collections import defaultdict
from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_session import UnifiedResearchSession


SEED = 93701901
CONFIG = ResearchConfig(observation_budget=384, compute_budget=360000,
    minimum_improvement=1e-7, max_history=4, max_rounds=6, active_probes=True, seed=SEED)
STAGES = [('focus', .12, ''), ('event', .12, 'event_'),
          ('position_focus', .31, 'position_')]
TASKS = [{'environment': 'objects', 'objects': 2, 'apparatus_seed': SEED,
          'seed': SEED + 10007 * index, 'readout': readout,
          'actuator_curvature': curvature, 'rename_prefix': prefix}
         for index, (readout, curvature, prefix) in enumerate(STAGES)]


@pytest.fixture(scope='module')
def full_research_history():
    session = UnifiedResearchSession(CONFIG, tasks=TASKS, measurement_budget=4608)
    snapshots = []
    for _ in TASKS:
        report = session.run(max_tasks=1)
        saved = json.loads(json.dumps(report))
        snapshots.append(saved)
        session = UnifiedResearchSession(state=saved)
    return snapshots


def _events(history, action):
    return [event for result in history['results'] for event in result['research']['actions']
            if event['action'] == action]


def test_full_run_charges_paid_structural_and_relation_work_to_the_existing_trial(full_research_history):
    source = full_research_history[0]
    research = source['results'][0]['research']
    followups = _events(source, 'research_action_followup')
    assert research['status'] == 'completed'
    assert {event['stage'] for event in followups} == {
        'structural_experiments_and_revision', 'inherited_relation_research', 'frozen_confirmation'}
    acquisitions = [event for event in research['actions'] if event['action'] == 'acquire']
    structural = [event for event in acquisitions if event['purpose'] == 'resolve_structure_boundary'
                  or (event['purpose'].startswith('test_')
                      and event['purpose'].endswith(('_baseline', '_intervention')))]
    structural_charge = next(event for event in followups
                             if event['stage'] == 'structural_experiments_and_revision')
    relation_charge = next(event for event in followups
                           if event['stage'] == 'inherited_relation_research')
    # The relation experiment first distinguishes its instrument hypotheses;
    # that paid input calibration belongs to the same originating trial.
    relation_work = research['actions'][research['actions'].index(structural_charge) + 1:
                                        research['actions'].index(relation_charge)]
    relation = [event for event in relation_work if event['action'] == 'acquire']
    assert {event['purpose'] for event in relation} == {
        'distinguish_instrument_hypotheses', 'distinguish_inherited_relation'}
    for stage, paid in [('structural_experiments_and_revision', structural),
                        ('inherited_relation_research', relation)]:
        charged = [event for event in followups if event['stage'] == stage]
        assert paid and charged
        for resource in ('observation_cost', 'measurement_cost'):
            assert sum(event[resource] for event in charged) == sum(event[resource] for event in paid)
        assert sum(event['compute_cost'] for event in charged) > 0
    refinements = [event for event in research['actions'] if event['action'] == 'revise_structure_boundary']
    assert structural_charge['compute_cost'] >= sum(event['compute_cost'] for event in refinements)
    original = next(event for event in _events(source, 'research_action_feedback')
        if event['context'] == structural_charge['context']
        and event['research_action'] == structural_charge['research_action']
        and event['feedback']['trials'] == structural_charge['trial'])
    assert structural_charge['feedback']['trials'] == original['feedback']['trials']
    for key in ('expected_gain', 'last_gain', 'last_before_loss', 'last_after_loss'):
        assert structural_charge['feedback'][key] == original['feedback'][key]
    trials = original['feedback']['trials']
    assert structural_charge['feedback']['mean_compute_cost'] * trials == pytest.approx(
        original['feedback']['mean_compute_cost'] * trials + structural_charge['compute_cost'])


def test_session_json_retains_total_costs_without_counting_followup_as_new_trials(full_research_history):
    final = full_research_history[-1]
    totals = defaultdict(lambda: defaultdict(float))
    for event in _events(final, 'research_action_feedback'):
        key = event['context'], event['research_action']
        totals[key]['trials'] += 1
        for resource in ('observation', 'measurement', 'compute'):
            totals[key][resource] += event.get(resource + '_cost', 0)
    followups = _events(final, 'research_action_followup')
    assert followups
    for event in followups:
        key = event['context'], event['research_action']
        assert 1 <= event['trial'] <= totals[key]['trials']
        for resource in ('observation', 'measurement', 'compute'):
            totals[key][resource] += event[resource + '_cost']
    for event in _events(final, 'research_action_confirmation'):
        key = event['context'], event['research_action']
        totals[key]['compute'] += event['compute_cost']
    restored = UnifiedResearchSession(state=json.loads(json.dumps(final)))
    for (context, action), paid in totals.items():
        record = restored.library.action_experience[context][action]
        assert record['trials'] == paid['trials']
        for resource in ('observation', 'measurement', 'compute'):
            assert record['mean_' + resource + '_cost'] == pytest.approx(paid[resource] / paid['trials'])
    assert len(final['results']) == 3
    assert all(result['research']['status'] == 'completed' for result in final['results'])


# Confirmed directions now precede unexplored work. The old fixed-order
# cost-ablation trajectory is covered by the confirmation/fresh-counterexample
# regression; cost aggregation and JSON persistence stay exact above.


def test_memory_disabled_still_executes_followup_without_changing_action_experience(full_research_history):
    from dataclasses import replace
    library = KnowledgeLibrary.from_dict(deepcopy(full_research_history[-1]['knowledge']))
    before = deepcopy(library.action_experience)
    with UnifiedObjectLab(objects=2, seed=SEED, apparatus_seed=SEED,
            readout='focus', actuator_curvature=.12) as laboratory:
        report = UnifiedResearchAgent(replace(CONFIG, use_memory=False),
            library=library, measurement_budget=4608).run(laboratory)
    assert any(event['action'] == 'route_structural_boundary_research' for event in report['actions'])
    # Range refinement now distinguishes complete response programs, including
    # their history and instrument inputs, instead of a bare radius response.
    acquired = {event['probe_id'] for event in report['actions']
                if event['action'] == 'acquire' and event['purpose'] == 'distinguish_inherited_relation'}
    tested = {event['probe_id'] for event in report['actions']
              if event['action'] == 'test_inherited_relation_predictions'}
    assert acquired and tested == acquired
    assert report['status'] == 'completed'
    assert not any(event['action'] in {'research_action_feedback', 'research_action_followup'}
                   for event in report['actions'])
    assert library.action_experience == before
