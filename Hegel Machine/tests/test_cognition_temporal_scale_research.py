"""A learned response changes temporal scale through paid research and JSON reuse."""

from copy import deepcopy
from dataclasses import replace
import gzip
import json

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, ResearchConfig
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_cognition_noise_research_reopening import _paid_costs, laboratory


CONFIG = ResearchConfig(observation_budget=240, compute_budget=240000,
    minimum_improvement=1e-7, max_rounds=4, active_probes=True, max_candidates=8)
WIDTHS = (1, 3, 5, 7, 2, 3)


@pytest.fixture(scope='module')
def learned_temporal_changes(tmp_path_factory):
    library, reports, snapshots, tasks = KnowledgeLibrary(), [], [], []
    assert not library.concepts and not library.definitions
    for position, width in enumerate(WIDTHS):
        # Width belongs to the physical laboratory only. The public task uses
        # the same input names and generic prediction instruction at every step.
        task = {'apparatus_seed': 9922701, 'seed': 9923701 + position * 1009,
                'response': 'base', 'inner_weight': -.45, 'window_width': width}
        lab = laboratory(task)
        config = replace(CONFIG, seed=task['seed'])
        report = UnifiedResearchAgent(config, library=library, measurement_budget=0).run(lab)
        _paid_costs(report, lab, config, 0)
        reports.append(report)
        tasks.append(task)
        snapshots.append(deepcopy(library.to_dict()))
        library = KnowledgeLibrary.from_dict(json.loads(json.dumps(snapshots[-1])))

    # No evaluation response can influence a later task's research choices.
    errors = []
    for position, (task, report) in enumerate(zip(tasks, reports)):
        lab, rows = laboratory(task), []
        for repeat, length in enumerate((1, 3, 9, 64, 64)):
            probe = lab.random_probe(9928701 + position * 1009 + repeat,
                                    length=length, name=f'held_out_{repeat}')
            rows.extend(lab.acquire(probe).observations)
        errors.append(loss(Program.from_dict(report['program']), rows, report['fallback'], .5))
    saved = tmp_path_factory.mktemp('temporal_scale_research') / 'learned_changes.json.gz'
    saved.write_bytes(gzip.compress(json.dumps({'tasks': tasks, 'reports': reports,
        'knowledge': snapshots[-1], 'terminal_errors': errors}).encode()))
    return reports, snapshots, errors, saved


def _independent_temporal_evidence(report):
    events = report['actions']
    tested = next(event for event in events if event['action'] == 'test_mechanism_predictions'
                  and any(item['kind'] == 'mechanism_temporal_refit' for item in event['alternatives']))
    acquired = next(event for event in events if event['action'] == 'acquire'
                    and event['probe_id'] == tested['probe_id'])
    planned = next(event for event in events if event['action'] == 'plan_mechanism_experiment'
                   and event['probe_id'] == tested['probe_id'])
    assert acquired['purpose'] == 'distinguish_mechanisms'
    assert planned['predicted_separation'] > CONFIG.minimum_improvement
    assert events.index(planned) < events.index(acquired) < events.index(tested)
    batch = next(item for item in report['batches'] if item['probe']['probe_id'] == tested['probe_id'])
    rows = [Observation(item['features'], item['target'], tuple(item['history']))
            for item in batch['observations']]
    supported = []
    for alternative, hypothesis, outcome in zip(tested['alternatives'],
            tested['plan']['hypotheses'], tested['evidence']['hypotheses']):
        assert alternative['candidate_id'] == hypothesis['candidate_id'] == outcome['candidate_id']
        assert outcome['measured_loss'] == pytest.approx(loss(
            Program.from_dict(hypothesis['program']), rows, 0., .5))
        if alternative['kind'] == 'mechanism_temporal_refit':
            supported.append(outcome['status'] == 'supported_on_probe')
            assert alternative['revision']['temporal_revisions']
    assert any(supported)
    assert tested['evidence']['compute_cost'] == len(rows) * len(tested['plan']['hypotheses'])
    validated, fresh = _confirmation(report)
    assert validated['loss'] < CONFIG.minimum_improvement
    assert events.index(tested) < events.index(fresh)
    assert fresh['probe_id'] != tested['probe_id']
    fresh_batch = next(item for item in report['batches'] if item['probe']['probe_id'] == fresh['probe_id'])
    assert fresh_batch['probe']['sequence'] != batch['probe']['sequence']
    assert fresh['observation_cost'] == validated['observation_rows'] > 0
    charged = next(event for event in events if event['action'] in
        {'research_action_followup', 'research_action_pending_cost'}
        and event.get('stage') == 'mechanism_experiment')
    assert charged['observation_cost'] == acquired['observation_cost']
    assert charged['compute_cost'] >= planned['compute_cost'] + tested['evidence']['compute_cost']
    return tested


def _confirmation(report):
    events = report['actions']
    validated = next(event for event in events if event['action'] == 'validate_frozen_program'
                     and event['accepted'])
    fresh = next(event for event in reversed(events[:events.index(validated)])
                 if event['action'] == 'acquire' and event['purpose'] in
                 {'confirm_frozen_program', 'confirm_short_frozen_program'})
    assert fresh['observation_cost'] == validated['observation_rows'] > 0
    assert any(event['action'] == 'acquire' and event['probe_id'] != fresh['probe_id']
               for event in events[:events.index(fresh)])
    return validated, fresh


def _selected_temporal_evidence(report, learned_sources):
    if report['selected_kind'] == 'mechanism_temporal_refit':
        assert learned_sources & set(report['method_source_ids'])
        return _independent_temporal_evidence(report)
    # Direct joint discovery may now explain the change more cheaply than a
    # revision of the remembered mechanism; it still needs fresh paid evidence.
    assert report['selected_kind'] == 'research_program'
    plan = report['research_program']
    assert not plan['source_ids']
    assert any(step['operation'] == 'discover_temporal_coordinates'
               for step in plan['selection_steps'])
    assert plan['status'] == 'supported_on_fresh_trajectory'
    validated, fresh = _confirmation(report)
    proposed = next(event for event in report['actions']
                    if event['action'] == 'propose_coordinate_reconstruction'
                    and any(item['program'] == report['program'] for item in event['alternatives']))
    assert proposed['compute_cost'] > 0
    assert report['actions'].index(proposed) < report['actions'].index(fresh)
    batches = {batch['probe']['probe_id']: batch for batch in report['batches']}
    evidence = batches[fresh['probe_id']]
    assert all(evidence['probe']['sequence'] != batches[event['probe_id']]['probe']['sequence']
               for event in report['actions'][:report['actions'].index(fresh)]
               if event['action'] == 'acquire')
    rows = [Observation(item['features'], item['target'], tuple(item['history']))
            for item in evidence['observations']]
    assert validated['loss'] == pytest.approx(loss(
        Program.from_dict(report['program']), rows, report['fallback'], .5))
    assert validated['loss'] < CONFIG.minimum_improvement


def test_acquired_window_can_grow_twice_after_json_handoff(learned_temporal_changes, record_property):
    reports, _, errors, saved = learned_temporal_changes
    record_property('research_reports', str(saved))
    assert all(report['status'] == 'completed' for report in reports[:4])
    assert all(error < CONFIG.minimum_improvement for error in errors[:4])
    source = reports[1]
    assert Program.from_dict(source['program']).history_depth >= 2
    for position, width in ((2, 5), (3, 7)):
        changed = reports[position]
        assert Program.from_dict(changed['program']).history_depth >= width - 1
        assert changed['concepts_added']
        learned_sources = {source for report in reports[:position] for source in report['concepts_added']}
        _selected_temporal_evidence(changed, learned_sources)


def test_acquired_long_window_can_shrink_with_new_distinction(learned_temporal_changes):
    reports, _, errors, _ = learned_temporal_changes
    changed = reports[4]
    assert changed['status'] == 'completed' and errors[4] < CONFIG.minimum_improvement
    assert Program.from_dict(changed['program']).history_depth < Program.from_dict(reports[3]['program']).history_depth
    assert changed['concepts_added']
    learned_sources = {source for report in reports[:4] for source in report['concepts_added']}
    _selected_temporal_evidence(changed, learned_sources)


def test_returning_window_keeps_original_and_revised_knowledge(learned_temporal_changes):
    reports, snapshots, errors, _ = learned_temporal_changes
    returned = reports[-1]
    assert returned['status'] == 'completed' and errors[-1] < CONFIG.minimum_improvement
    original_sources = {source for report in reports[:2] for source in report['concepts_added']}
    assert original_sources & set(returned['used_sources'])
    final_ids = {item['concept_id'] for item in snapshots[-1]['concepts']}
    assert {source for report in reports for source in report['concepts_added']} <= final_ids
    for earlier, later in zip(snapshots, snapshots[1:]):
        assert all(definition in later['definitions'] for definition in earlier['definitions'])
    _confirmation(returned)
