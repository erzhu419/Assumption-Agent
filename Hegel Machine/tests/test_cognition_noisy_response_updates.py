"""Noise support leaves room for budgeted, independently confirmed improvement."""

from copy import deepcopy
from dataclasses import replace
import gzip
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.action_policy import ActionOption, ResearchActionPolicy
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, ResearchConfig, TaskSpec
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_cognition_noise_research_reopening import _paid_costs, laboratory


@pytest.fixture(scope='module', params=['root02', 'root12'])
def updated_noisy_chain(request, tmp_path_factory):
    path = Path(__file__).parent / 'fixtures/noisy_response_update_protocol.json.gz'
    payload = json.loads(gzip.decompress(path.read_bytes()))
    root = request.param
    library, reports, snapshots = KnowledgeLibrary(), [], []
    for task in payload['roots'][root]['tasks']:
        config = ResearchConfig(**{**payload['config'], 'seed': task['seed']})
        lab = laboratory(task)
        report = UnifiedResearchAgent(config, library=library, measurement_budget=0).run(lab)
        _paid_costs(report, lab, config, 0)
        reports.append(report)
        snapshots.append(json.loads(json.dumps(library.to_dict())))
        library = KnowledgeLibrary.from_dict(snapshots[-1])
    saved = tmp_path_factory.mktemp('noisy_response_updates') / f'{root}.json.gz'
    saved.write_bytes(gzip.compress(json.dumps({'reports': reports, 'knowledge': snapshots[-1]}).encode()))
    return root, payload, reports, snapshots, saved


def test_supported_noisy_source_considers_a_budgeted_update_before_transfer(updated_noisy_chain, record_property):
    root, payload, reports, snapshots, saved = updated_noisy_chain
    source, successor = reports
    record_property('research_reports', str(saved))
    assert source['status'] == successor['status'] == 'completed'
    assert set(source['concepts_added']) & set(successor['used_sources'])
    events = source['actions']
    chosen = next(event for event in events if event['action'] == 'choose_noisy_response_update')
    proposed = next(event for event in events if event['action'] == 'propose_noisy_response_update')
    selected = next(event for event in events if event['action'] == 'select_noisy_response_refit')
    decision = next(event for event in events if event['action'] == 'confirm_noisy_response_refit')
    assert chosen['selected_action'] == 'response_update'
    assert proposed['training_rows'] == 72 and proposed['heldout_rows'] == 24
    assert 0 < proposed['gain'] < proposed['margin']
    assert events.index(proposed) < events.index(selected) < events.index(decision)
    assert selected['program'] != selected['parent_program']
    assert decision['probe_id'] != selected['comparison_probe_id']
    assert selected['requires_fresh_confirmation']
    if decision['accepted']:
        assert decision['after_loss'] < decision['before_loss']
        assert source['program'] == selected['program']
    else:
        assert decision['before_loss'] <= decision['after_loss'] + payload['config']['minimum_improvement']
        assert source['program'] == selected['parent_program']
        assert source['noise_assessment']['replication']['noise_supported']
    acquired = [event for event in events if event['action'] == 'acquire']
    confirmation = next(event for event in acquired if event['probe_id'] == decision['probe_id'])
    assert confirmation['observation_cost'] == decision['observation_rows']
    record = snapshots[0]['action_experience'][chosen['context']]['response_update']
    assert record['trials'] == 1 and record['knowledge_applicability']
    assert record['mean_observation_cost'] >= decision['observation_rows']
    outcome = next(event for event in events if event['action'] == 'research_action_confirmation'
                   and event.get('research_action') == 'response_update')
    assert outcome['context'] == chosen['context']
    assert outcome['status'] == ('supported' if decision['accepted'] else 'refuted')
    pending = next(event for event in events if event['action'] == 'research_action_pending'
                   and event.get('research_action') == 'response_update')
    costs = next(event for event in events if event['action'] == 'research_action_pending_cost'
                 and event['pending_id'] == pending['pending_id'])
    assert costs['observation_cost'] == 2 * decision['observation_rows']
    feedback = next(event for event in events if event['action'] == 'research_action_feedback'
                    and event.get('pending_id') == pending['pending_id'])
    for resource in ('observation_cost', 'measurement_cost', 'compute_cost'):
        assert feedback[resource] == pending[resource] + costs[resource]
        assert record[f'mean_{resource}'] * record['trials'] == feedback[resource] + outcome.get(resource, 0)
    assert snapshots[1]['action_experience'][chosen['context']]['response_update'] == record

    # Baseline programs are used only here, after both actual learning stages.
    settings = payload['terminal']
    comparisons = []
    for task, report, baseline in zip(payload['roots'][root]['tasks'], reports,
            payload['roots'][root]['baseline_evaluation_only']):
        noisy = laboratory({**task, 'noise_seed': task['seed'] + settings['seed_offset']})
        mean_lab, rows = laboratory({**task, 'noise_sigma': 0.}), []
        for index in range(settings['trajectories']):
            probe = noisy.random_probe(task['seed'] + settings['seed_offset'] + index * 10007,
                                       length=settings['length'], name=f'terminal_{index}')
            observations = noisy.acquire(probe).observations
            means = mean_lab.acquire(probe).observations
            rows.extend(replace(row, target=mean.target) for row, mean in zip(observations, means))
        comparisons.append({'before': loss(Program.from_dict(baseline['program']), rows, baseline['fallback'], .5),
                            'after': loss(Program.from_dict(report['program']), rows, report['fallback'], .5)})
    record_property('mean_response_comparison', json.dumps(comparisons))
    record_property('research_costs', json.dumps([report['costs'] for report in reports]))


def test_negative_response_update_stops_after_json_without_suppressing_new_basis():
    rows = [Observation({'drive': -.4, 'load': .2}, .1),
            Observation({'drive': .7, 'load': -.1}, .8)]
    spec = TaskSpec('response', 'Predict the measured response', ('drive', 'load'), (), ())
    static = {'history_depth': 0, 'kind': 'research_program', 'operators': ['add', 'mul', 'feature', 'const']}
    window = {**static, 'history_depth': 2, 'operators': [*static['operators'], 'window_sum']}
    experience = {}
    policy = ResearchActionPolicy(rows, spec, experience)
    policy.set_condition('noisy_response')
    old_context = policy.set_action_knowledge('response_update', static)
    feedback = policy.update('response_update', before_loss=.00015, after_loss=.00017,
        observation_cost=96, measurement_cost=0, compute_cost=7100)
    origin = {'context': old_context, 'action': 'response_update', 'trial': feedback['trials'],
              'credited_gain': feedback['last_gain']}
    policy.revise_feedback(experience, origin, status='refuted', compute_cost=48)
    old_record = deepcopy(policy.action_record('response_update'))
    restored = ResearchActionPolicy(rows, spec, json.loads(json.dumps(experience)))
    restored.set_condition('noisy_response')
    assert restored.set_action_knowledge('response_update', static) == old_context
    option = ActionOption('response_update', 96, 7200)
    assert restored.choose([option], observation_remaining=120, compute_remaining=200000,
                            measurement_remaining=0) is None
    assert restored.last_decision['reason'] == 'no_measured_positive_return'
    assert restored.action_record('response_update') == old_record
    new_context = restored.set_action_knowledge('response_update', window)
    assert new_context != old_context
    assert restored.choose([option], observation_remaining=120, compute_remaining=200000,
                           measurement_remaining=0) == option
    assert restored.last_decision['reason'] == 'untried_action'
    assert restored._store[old_context]['response_update'] == old_record
    assert restored.choose([option], observation_remaining=95, compute_remaining=200000,
                           measurement_remaining=0) is None
    assert restored.last_decision['reason'] == 'no_affordable_action'
