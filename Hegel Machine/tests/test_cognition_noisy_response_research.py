"""Actual learned noisy responses improve before a separately purchased confirmation."""

from dataclasses import replace
import gzip
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_cognition_noise_research_reopening import _paid_costs, laboratory


def _research(root, *, max_rounds=None):
    path = Path(__file__).parent / f'fixtures/noisy_window_{root}_incoming.json.gz'
    payload = json.loads(gzip.decompress(path.read_bytes()))
    config = ResearchConfig(**{**payload['config'], 'seed': payload['task']['seed']})
    if max_rounds is not None:
        config = replace(config, max_rounds=max_rounds)
    library, lab = KnowledgeLibrary.from_dict(payload['knowledge']), laboratory(payload['task'])
    report = UnifiedResearchAgent(config, library=library, measurement_budget=0).run(lab)
    _paid_costs(report, lab, config, 0)
    return report, library, payload


@pytest.fixture(scope='module')
def noisy_research():
    improved = _research('root01')
    recovered = _research('root04')
    return improved, recovered


def test_learned_window_refit_improves_and_is_confirmed_on_another_trajectory(noisy_research, record_property):
    report, library, payload = noisy_research[0]
    events = report['actions']
    proposed = next(item for item in events if item['action'] == 'propose_noisy_response_refits')
    selected = next(item for item in events if item['action'] == 'select_noisy_response_refit')
    confirmed = next(item for item in events if item['action'] == 'confirm_noisy_response_refit')
    assert proposed['training_rows'] == 48 and proposed['compute_cost'] > 0
    assert any(item['action'] == 'compare_noisy_response_refit' and item['supported']
               and item['gain'] > item['margin'] for item in events)
    assert selected['requires_fresh_confirmation']
    assert confirmed['probe_id'] != selected['comparison_probe_id']
    assert confirmed['accepted'] and confirmed['after_loss'] < confirmed['before_loss']
    assert report['status'] == 'completed' and report['concepts_added']
    assert report['program'] == selected['program'] != selected['parent_program']
    assert Program.from_dict(report['program']).history_depth == 2
    batches = {batch['probe']['probe_id']: batch for batch in report['batches']}
    assert batches[confirmed['probe_id']]['probe']['sequence'] != batches[selected['comparison_probe_id']]['probe']['sequence']
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    assert any(concept.concept_id in report['concepts_added']
               and concept.program.to_dict() == report['program'] for concept in restored.concepts)

    # Truth is confined to terminal scoring, after both research runs have ended.
    task = payload['task']
    noisy, mean_lab = laboratory(task), laboratory({**task, 'noise_sigma': 0.})
    terminal_rows = []
    for index in range(2):
        probe = noisy.random_probe(task['seed'] + 710003 + index * 10007,
                                   length=64, name=f'terminal_{index}')
        observed, means = noisy.acquire(probe).observations, mean_lab.acquire(probe).observations
        terminal_rows.extend(replace(row, target=truth.target) for row, truth in zip(observed, means))
    before = loss(Program.from_dict(payload['baseline']['program']), terminal_rows,
                  payload['baseline']['fallback'], .5)
    after = loss(Program.from_dict(report['program']), terminal_rows, report['fallback'], .5)
    assert after < before
    record_property('mean_response_mse_before', before)
    record_property('mean_response_mse_after', after)
    record_property('research_costs', json.dumps(report['costs']))


def test_noisy_directed_counterexample_is_remeasured_at_its_own_full_history(noisy_research):
    report, _, payload = noisy_research[1]
    assert payload['baseline']['status'] == 'unresolved' and report['status'] == 'completed'
    reassessed = [item for item in report['actions'] if item['action'] == 'reassess_noisy_counterexample']
    assert len(reassessed) == 1
    event = reassessed[0]
    assert set(event['reasons']) == {'program_experiment', 'inherited_relation'}
    assert event['adequate'] and event['replication']['noise_supported']
    original = next(batch for batch in report['batches'] if batch['probe']['probe_id'] == event['probe_id'])
    repeated = [batch for batch in report['batches']
                if batch['probe']['probe_id'] != event['probe_id']
                and batch['probe']['sequence'] == original['probe']['sequence']]
    assert len(repeated) == 1
    assert len(original['observations']) == len(repeated[0]['observations']) == event['replication']['matched_rows'] == 7
    assert repeated[0]['channels'] == original['channels']
    for first, second in zip(original['observations'], repeated[0]['observations']):
        assert first['features'] == second['features']
        assert [{k:v for k,v in row.items() if k != '__target__'} for row in first['history']] == [
            {k:v for k,v in row.items() if k != '__target__'} for row in second['history']]


def test_no_confirmation_round_remaining_preserves_supported_original():
    # Reducing observation_budget also changes the initial sample lengths, so
    # the round limit isolates the unavailable additional confirmation instead.
    report, library, payload = _research('root01', max_rounds=1)
    old = payload['baseline']['program']
    assert report['status'] == 'completed' and report['program'] == old
    assert not any(item['action'] == 'select_noisy_response_refit' for item in report['actions'])
    assert report['noise_assessment']['replication']['noise_supported']
    assert all(concept.program.to_dict() == old for concept in library.concepts
               if concept.concept_id in report['concepts_added'])


def test_noisy_static_failure_is_researched_before_transferring_to_a_window(record_property, tmp_path):
    path = Path(__file__).parent / 'fixtures/noisy_root08_research_protocol.json'
    payload = json.loads(path.read_text())
    library, reports = KnowledgeLibrary(), []
    for task in payload['tasks']:
        config = ResearchConfig(**{**payload['config'], 'seed': task['seed']})
        lab = laboratory(task)
        report = UnifiedResearchAgent(config, library=library, measurement_budget=0).run(lab)
        _paid_costs(report, lab, config, 0)
        reports.append(report)
        library = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    saved = tmp_path / 'root08_research.json.gz'
    saved.write_bytes(gzip.compress(json.dumps(reports).encode()))
    record_property('research_reports', str(saved))
    static, window = reports
    assert static['status'] == window['status'] == 'completed'
    assert set(static['concepts_added']) & set(window['used_sources'])
    validations = [event for event in static['actions'] if event['action'] == 'validate_frozen_program']
    assert len(validations) >= 2 and not validations[0]['accepted'] and validations[-1]['accepted']
    assert any(event['action'] == 'compare_noise_replication' and event['repeatable_error']
               and not event['noise_supported'] for event in static['actions'])
    fitted = next(event for event in static['actions'] if event['action'] == 'propose_noise_measurement_refit')
    selected = next(event for event in static['actions'] if event['action'] == 'select_noisy_response_refit')
    decision = next(event for event in static['actions'] if event['action'] == 'confirm_noisy_response_refit')
    assert fitted['training_rows'] == 96 and fitted['after_loss'] < fitted['before_loss']
    assert selected['program'] != selected['parent_program']
    assert decision['probe_id'] != selected['comparison_probe_id']
    assert not decision['accepted'] and decision['before_loss'] < decision['after_loss']
    assert static['program'] == selected['parent_program']
    assert static['noise_assessment']['replication']['noise_supported']

    pending = next(event for event in static['actions']
                   if event['action'] == 'research_action_pending' and event['research_action'] == 'response_refit')
    followup = next(event for event in static['actions']
        if event['action'] == 'research_action_pending_cost' and event['pending_id'] == pending['pending_id'])
    # The rejected refit's trial owns both the new confirmation and the repeat
    # later used to assess its better old rival, including their computations.
    assert followup['observation_cost'] == 2 * decision['observation_rows']
    repeats = [event for event in static['actions'][static['actions'].index(pending):]
               if event['action'] == 'compare_noise_replication']
    assert repeats and followup['compute_cost'] >= sum(event['compute_cost'] for event in repeats)
    context = pending['context']
    feedback = [event for report in reports for event in report['actions']
        if event['action'] in {'research_action_feedback', 'research_action_confirmation', 'research_action_followup'}
        and event.get('research_action') == 'response_refit' and event.get('context') == context]
    record = library.action_experience[context]['response_refit']
    for resource in ('observation_cost', 'measurement_cost', 'compute_cost'):
        assert record[f'mean_{resource}'] * record['trials'] == pytest.approx(
            sum(event.get(resource, 0) for event in feedback))
    window_decision = next(event for event in window['actions'] if event['action'] == 'confirm_noisy_response_refit')
    assert window_decision['accepted'] and window_decision['after_loss'] < window_decision['before_loss']

    # Both learning stages finish before private mean-response scoring starts.
    means, initial_errors = [], []
    for task, report in zip(payload['tasks'], reports):
        settings = payload['terminal']
        noisy = laboratory({**task, 'noise_seed': task['seed'] + settings['seed_offset']})
        mean_lab = laboratory({**task, 'noise_sigma': 0.})
        rows = []
        for index in range(settings['trajectories']):
            probe = noisy.random_probe(task['seed'] + settings['seed_offset'] + index * 10007,
                                       length=settings['length'], name=f'terminal_{index}')
            observed, truth = noisy.acquire(probe).observations, mean_lab.acquire(probe).observations
            rows.extend(replace(row, target=target.target) for row, target in zip(observed, truth))
        means.append(loss(Program.from_dict(report['program']), rows, report['fallback'], .5))
        first = next(event['program'] for event in report['actions'] if event['action'] == 'validate_frozen_program')
        initial_errors.append(loss(Program.from_dict(first), rows, report['fallback'], .5))
    assert means[0] == pytest.approx(initial_errors[0])
    assert means[1] < initial_errors[1]
    record_property('initial_mean_errors', json.dumps(initial_errors))
    record_property('final_mean_errors', json.dumps(means))
    record_property('research_costs', json.dumps([report['costs'] for report in reports]))
