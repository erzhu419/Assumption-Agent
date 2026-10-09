"""Measured noise leaves the original response improvement experiment affordable."""

from dataclasses import replace
import gzip
import json

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_cognition_noise_research_reopening import _paid_costs, laboratory


def test_paid_noise_repeat_preserves_a_freshly_confirmed_response_refit(record_property):
    # First visit of the original 0.131 root01 protocol. The agent receives
    # public controls and purchased noisy observations, never this task record.
    task = dict(apparatus_seed=9914301, seed=10314304, response='square',
                inner_weight=.35, window_width=3, noise_sigma=.01)
    config = ResearchConfig(observation_budget=240, compute_budget=240000,
        max_rounds=4, max_candidates=8, minimum_improvement=1e-7,
        active_probes=True, proposer='search', seed=task['seed'])
    lab = laboratory(task)
    report = UnifiedResearchAgent(config, library=KnowledgeLibrary(),
                                  measurement_budget=0).run(lab)
    _paid_costs(report, lab, config, 0)
    events = report['actions']
    tested = next(event for event in events
        if event['action'] == 'test_response_noise_before_expansion' and event['supported'])
    assert tested['noise_supported'] and not tested['repeatable_error']
    assert tested['repeat_noise_variance'] > config.minimum_improvement

    # The same finite observation allowance must still purchase a fitted rival
    # and an independent confirmation, after the noise explanation is tested.
    proposed = next((event for event in events
                     if event['action'] == 'propose_noisy_response_refits'), None)
    assert proposed is not None, 'Conditional research consumed the paid response-refit allowance'
    selected = next(event for event in events if event['action'] == 'select_noisy_response_refit')
    confirmed = next(event for event in events if event['action'] == 'confirm_noisy_response_refit')
    assert events.index(tested) < events.index(proposed) < events.index(selected) < events.index(confirmed)
    assert selected['requires_fresh_confirmation'] and confirmed['accepted']
    assert confirmed['after_loss'] < confirmed['before_loss']
    assert report['status'] == 'completed' and report['concepts_added']
    assert report['program'] == selected['program'] != selected['parent_program']
    assert not any(event['action'] == 'research_conditional_matched_experiments' for event in events)

    batches = {batch['probe']['probe_id']: batch for batch in report['batches']}
    comparison, confirmation = (batches[probe_id] for probe_id in
        (selected['comparison_probe_id'], confirmed['probe_id']))
    assert comparison['probe']['probe_id'] != confirmation['probe']['probe_id']
    assert comparison['probe']['sequence'] != confirmation['probe']['sequence']
    assert len(confirmation['observations']) == confirmed['observation_rows']

    # Mean-response targets are acquired only after the entire learning run.
    noisy = laboratory({**task, 'noise_seed': task['seed'] + 710003})
    means = laboratory({**task, 'noise_sigma': 0.})
    terminal = []
    for index in range(4):
        probe = noisy.random_probe(task['seed'] + 710003 + index * 10007,
                                   length=64, name=f'terminal_{index}')
        observed, truth = noisy.acquire(probe).observations, means.acquire(probe).observations
        terminal.extend(replace(row, target=mean.target) for row, mean in zip(observed, truth))
    before = loss(Program.from_dict(selected['parent_program']), terminal, report['fallback'], .5)
    after = loss(Program.from_dict(report['program']), terminal, report['fallback'], .5)
    assert after < before
    record_property('mean_response_mse_before', before)
    record_property('mean_response_mse_after', after)
    record_property('research_costs', json.dumps(report['costs']))


def test_noisy_mechanism_distinction_repeats_before_rejecting_the_learned_window(record_property, tmp_path):
    task = dict(apparatus_seed=9914301, seed=10314304, response='square',
                inner_weight=.35, window_width=3, noise_sigma=.01)
    config = ResearchConfig(observation_budget=240, compute_budget=240000,
        max_rounds=4, max_candidates=8, minimum_improvement=1e-7,
        active_probes=True, proposer='search', seed=task['seed'])
    library = KnowledgeLibrary()
    source_lab = laboratory(task)
    source = UnifiedResearchAgent(config, library=library, measurement_budget=0).run(source_lab)
    _paid_costs(source, source_lab, config, 0)
    assert source['status'] == 'completed' and source['concepts_added']
    # Transfer only what this actual first visit learned, through the ordinary
    # JSON handoff; no frozen result or true response is supplied to the agent.
    library = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    task = {**task, 'seed': 10315313, 'window_width': 5}
    config = replace(config, seed=task['seed'])
    lab = laboratory(task)
    report = UnifiedResearchAgent(config, library=library, measurement_budget=0).run(lab)
    _paid_costs(report, lab, config, 0)
    saved = tmp_path / 'root01_window_change.json.gz'
    saved.write_bytes(gzip.compress(json.dumps({'source': source, 'changed': report}).encode()))
    record_property('research_reports', str(saved))
    assert set(source['concepts_added']) & set(report['used_sources'])
    events = report['actions']
    noise = next(event for event in events
        if event['action'] == 'test_response_noise_before_expansion' and event['supported'])
    tested = next(event for event in events if event['action'] == 'test_mechanism_predictions')
    assert events.index(noise) < events.index(tested)
    assert noise['noise_supported'] and not noise['repeatable_error']
    outcomes = tested['evidence']['hypotheses']
    best = min(outcomes, key=lambda item: item['measured_loss'])
    original = next(item for item in outcomes if item['candidate_id'] == 'mechanism_original')
    assert best['candidate_id'] != original['candidate_id']
    assert best['measured_loss'] > config.minimum_improvement
    # This short repeat explains the variation but cannot yet support the
    # complete model. Its decision must remain pending for fresh confirmation.
    assert best['status'] == 'unresolved', 'A noisy mechanism probe rejected the learned window'
    assert best['noise_replication']['noise_supported']
    assert not best['noise_replication']['repeatable_error']
    assert original['status'] == 'counterexample'
    assert original['noise_replication']['repeatable_error']
    assert not original['noise_replication']['noise_supported']
    assert original['measured_loss'] > best['measured_loss']
    hypotheses = {item['candidate_id']: Program.from_dict(item['program'])
                  for item in tested['plan']['hypotheses']}
    learned = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    assert learned.transfer_refuted(lab.spec, hypotheses[original['candidate_id']])
    assert not learned.transfer_refuted(lab.spec, hypotheses[best['candidate_id']])
    assert not learned.transfer_refuted(source_lab.spec, Program.from_dict(source['program']))
    assert any(event['action'] == 'withdraw_target_transfer' and event['probe_id'] == tested['probe_id']
               for event in events)

    batches = {batch['probe']['probe_id']: batch for batch in report['batches']}
    measured = batches[tested['probe_id']]
    repeats = [batch for batch in report['batches']
        if batch['probe']['probe_id'] != tested['probe_id']
        and batch['probe']['sequence'] == measured['probe']['sequence']]
    assert len(repeats) == 1
    repeated = repeats[0]
    assert tested['repeat_probe_id'] == repeated['probe']['probe_id']
    assert repeated['channels'] == measured['channels']
    assert len(repeated['observations']) == len(measured['observations']) == best['rows']
    assert any(left['target'] != right['target'] for left, right in
               zip(measured['observations'], repeated['observations']))
    repeat_purchase = next(event for event in events if event['action'] == 'acquire'
                          and event['probe_id'] == repeated['probe']['probe_id'])
    assert repeat_purchase['purpose'] == 'repeat_mechanism_predictions'
    assert repeat_purchase['observation_cost'] == best['rows']
    assert events.index(repeat_purchase) < events.index(tested)
    assert not any(event['action'] == 'research_conditional_matched_experiments' for event in events)

    validated = next(event for event in events
        if event['action'] == 'validate_frozen_program' and event['accepted'])
    confirmation = next(event for event in reversed(events[:events.index(validated)])
        if event['action'] == 'acquire' and event['purpose'] == 'confirm_frozen_program')
    assert events.index(tested) < events.index(confirmation) < events.index(validated)
    assert batches[confirmation['probe_id']]['probe']['sequence'] != measured['probe']['sequence']
    assert report['status'] == 'completed' and report['concepts_added']
    assert Program.from_dict(report['program']).history_depth >= 4

    # Evaluate the changed apparatus only after both learning stages finish.
    noisy = laboratory({**task, 'noise_seed': task['seed'] + 710003})
    means = laboratory({**task, 'noise_sigma': 0.})
    terminal = []
    for index in range(4):
        probe = noisy.random_probe(task['seed'] + 710003 + index * 10007,
                                   length=64, name=f'terminal_{index}')
        observed, truth = noisy.acquire(probe).observations, means.acquire(probe).observations
        terminal.extend(replace(row, target=mean.target) for row, mean in zip(observed, truth))
    before = loss(Program.from_dict(source['program']), terminal, source['fallback'], .5)
    after = loss(Program.from_dict(report['program']), terminal, report['fallback'], .5)
    assert after < before
    record_property('mean_response_mse_before', before)
    record_property('mean_response_mse_after', after)
    record_property('research_costs', json.dumps(report['costs']))
