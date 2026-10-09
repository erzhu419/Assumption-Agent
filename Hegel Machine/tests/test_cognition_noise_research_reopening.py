"""Paid repetitions challenge a noise claim before research can complete."""

import gzip
import json
from math import sin
from pathlib import Path
from random import Random
import sys

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import MeasurementLab
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_research import UnifiedResearchAgent

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_nonlinear_mechanism_revision import laboratory
sys.path.remove(str(ROOT / 'scripts'))


def _root02(rounds):
    # This is the actual incoming learned library from the false-completion
    # case, including action experience. No true program is inserted or reset.
    payload = json.loads(gzip.decompress((ROOT / 'tests/fixtures/'
        'noise_repeat_root02_knowledge.json.gz').read_bytes()))
    config = ResearchConfig(**{**payload['config'], 'seed': payload['task']['seed'],
                               'max_rounds': rounds})
    library = KnowledgeLibrary.from_dict(payload['knowledge'])
    lab = laboratory(payload['task'])
    report = UnifiedResearchAgent(config, library=library, measurement_budget=0).run(lab)
    _paid_costs(report, lab, config, 0)
    return report, lab, payload


def _paid_costs(report, lab, config, measurement_budget):
    acquired = [action for action in report['actions'] if action['action'] == 'acquire']
    assert report['costs']['queries'] == len(acquired) == len(lab.acquisitions)
    for name, limit in (('observation_cost', config.observation_budget),
                        ('measurement_cost', measurement_budget)):
        assert report['costs'][name] == lab.costs[name] == sum(
            action[name] for action in acquired) <= limit
    assert report['costs']['compute_cost'] <= config.compute_budget


def _repeat_of_current_confirmation(report, compared):
    before = report['actions'][:report['actions'].index(compared)]
    repeat = next(action for action in reversed(before) if action['action'] == 'acquire'
        and action['purpose'] == 'distinguish_noise_from_repeatable_error')
    original = next(action for action in reversed(before) if action['action'] == 'acquire'
        and action['purpose'] in {'confirm_frozen_program', 'confirm_short_frozen_program'})
    batches = {batch['probe']['probe_id']: batch for batch in report['batches']}
    left, right = batches[original['probe_id']], batches[repeat['probe_id']]
    assert original['probe_id'] != repeat['probe_id']
    assert left['probe']['sequence'] == right['probe']['sequence']
    assert left['channels'] == right['channels']
    assert repeat['observation_cost'] == original['observation_cost'] == compared['matched_rows']
    assert repeat['measurement_cost'] == original['measurement_cost']
    assert compared['matched_rows'] == compared['original_rows'] == compared['repeated_rows']
    # Every prefix participates, not only the trajectory's last point. Targets
    # are outcomes here; the candidate does not read historical responses.
    assert not compared['target_history_required']
    for index, (first, second) in enumerate(zip(left['observations'], right['observations'])):
        assert first['features'] == second['features']
        assert len(first['history']) == len(second['history']) == index
        assert [{k: v for k, v in past.items() if k != '__target__'}
                for past in first['history']] == [
                    {k: v for k, v in past.items() if k != '__target__'}
                    for past in second['history']]
    passes = 4 if compared.get('reproducible_mse') is not None else 3
    assert compared['compute_cost'] == passes * compared['matched_rows']
    return original, repeat


def test_reproducible_sub_ten_tolerance_error_reopens_research_and_gets_fresh_confirmation(record_property):
    report, _, payload = _root02(4)
    compared = next(action for action in report['actions']
                    if action['action'] == 'compare_noise_replication')
    tolerance = payload['config']['minimum_improvement']
    assert tolerance < compared['original_model_loss'] < 10 * tolerance
    assert compared['repeat_loss'] == compared['repeat_mse'] == 0.
    assert compared['repeatable_error'] and not compared['noise_supported']
    original, repeat = _repeat_of_current_confirmation(report, compared)
    validations = [action for action in report['actions']
                   if action['action'] == 'validate_frozen_program']
    failed, supported = validations
    assert not failed['accepted']
    assert failed['noise_assessment']['reason'] == 'repeatable_unexplained_response'
    assert supported['accepted'] and supported['loss'] < tolerance
    assert supported['program'] != failed['program']
    after_failure = report['actions'][report['actions'].index(failed) + 1:
                                     report['actions'].index(supported)]
    assert any(action['action'] == 'propose_mechanism_revisions' for action in after_failure)
    confirmation = next(action for action in after_failure if action['action'] == 'acquire'
                        and action['purpose'] == 'confirm_frozen_program')
    assert confirmation['probe_id'] not in {original['probe_id'], repeat['probe_id']}
    batches = {batch['probe']['probe_id']: batch for batch in report['batches']}
    assert batches[confirmation['probe_id']]['probe']['sequence'] != batches[original['probe_id']]['probe']['sequence']
    assert report['status'] == 'completed' and report['concepts_added']

    # The terminal apparatus is separate and queried only after the decision.
    terminal = laboratory(payload['task'])
    rows = [row for index, length in enumerate((1, 3, 64, 64))
            for row in terminal.acquire(terminal.random_probe(13012021 + index,
                length=length, name=f'terminal_{index}')).observations]
    error = loss(Program.from_dict(report['program']), rows, report['fallback'], .5)
    assert error < tolerance
    assert terminal.costs['observation_cost'] == 132
    record_property('terminal_mse', error)
    record_property('research_costs', json.dumps(report['costs']))


def test_repeat_counterexample_remains_unresolved_when_no_research_round_remains():
    report, _, _ = _root02(1)
    compared = next(action for action in report['actions']
                    if action['action'] == 'compare_noise_replication')
    _repeat_of_current_confirmation(report, compared)
    assert compared['repeatable_error'] and not compared['noise_supported']
    assert report['status'] == 'unresolved'
    assert report['noise_assessment']['reason'] == 'repeatable_unexplained_response'
    assert not report['concepts_added']
    assert not any(action['accepted'] for action in report['actions']
                   if action['action'] == 'validate_frozen_program')


class NoisyInstrumentLab(MeasurementLab):
    """A reproducible instrument with independent response noise per experiment."""

    def _experiment(self, probe):
        key = probe.probe_id, tuple(point['u'] for point in probe.sequence)
        if key not in self._experiments:
            random = Random(f'{self.seed}:{key!r}')
            self._experiments[key] = [({'u': u, 'sensor_a': sin(3 * u),
                'sensor_b': u * u}, .15 + .55 * u + .8 * sin(3 * u)
                    + random.uniform(-.08, .08)) for u in key[1]]
        return self._experiments[key]


class VaryingInstrumentLab(NoisyInstrumentLab):
    def _experiment(self, probe):
        rows = super()._experiment(probe)
        random = Random(f'{self.seed}:{probe.probe_id}:instrument_noise')
        return [({**readings, 'sensor_a': readings['sensor_a'] + random.uniform(-.001, .001)}, target)
                for readings, target in rows]


def test_independent_response_noise_can_complete_with_paid_instrument_repetition(record_property):
    config = ResearchConfig(observation_budget=240, compute_budget=160000,
        max_rounds=4, max_candidates=8, minimum_improvement=1e-7, seed=70120)
    lab, library = NoisyInstrumentLab(seed=9013), KnowledgeLibrary()
    assert not library.concepts and not library.meta_priors
    report = UnifiedResearchAgent(config, measurement_budget=768, library=library).run(lab)
    _paid_costs(report, lab, config, 768)
    record_property('noisy_status', report['status'])
    record_property('noisy_costs', json.dumps(report['costs']))
    assert report['status'] == 'completed' and report['concepts_added']
    assert report['selected_channels'] == ['sensor_a']
    assert not any(action['action'] == 'select_noisy_response_refit' for action in report['actions'])
    assert report['noise_assessment']['reason'] == 'noise_compatible'
    compared = next(action for action in reversed(report['actions'])
                    if action['action'] == 'compare_noise_replication')
    assert compared['reason'] == 'response_varies_on_repeat'
    assert compared['noise_supported'] and not compared['repeatable_error']
    assert compared['repeat_loss'] > config.minimum_improvement
    _, repeated = _repeat_of_current_confirmation(report, compared)
    assert repeated['measurement_cost'] > 0
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    assert set(report['concepts_added']) <= {concept.concept_id for concept in restored.concepts}


def test_changing_instrument_inputs_leave_noisy_replication_inconclusive():
    config = ResearchConfig(observation_budget=240, compute_budget=160000,
        max_rounds=2, max_candidates=8, minimum_improvement=1e-7, seed=70120)
    lab = VaryingInstrumentLab(seed=9013)
    report = UnifiedResearchAgent(config, measurement_budget=768, library=KnowledgeLibrary()).run(lab)
    _paid_costs(report, lab, config, 768)
    comparisons = [action for action in report['actions']
                   if action['action'] == 'compare_noise_replication']
    assert comparisons
    assert any(item['reason'] in {'insufficient_matched_rows', 'unmatched_trajectory_inputs'}
               and item['repeat_mse'] is None and item['reproducible_mse'] is None
               and not item['noise_supported'] for item in comparisons)
    assert report['status'] == 'unresolved' and not report['concepts_added']
