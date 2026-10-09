"""Paid, uncontrolled condition readings compete on the original response."""

from dataclasses import replace
import gzip
import json
from random import Random

import pytest

from hegel_machine.cognition.evaluation import initial_prediction, loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import MeasurementBatch, MeasurementChannel
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.research_graph import ResearchGraph, bind_answers
from hegel_machine.cognition.unified_research import UnifiedResearchAgent, _terms


class DecodedObservedLab:
    """Independent physics; the condition is a paid reading, never a command."""

    def __init__(self, seed=9438101, *, width=3, placement='inside_window', smooth=False,
                 names=('alpha', 'beta', 'gamma'), meter='meter', threshold=.173,
                 control_scale=1., meter_scale=1., response_scale=1.):
        self.spec = TaskSpec('unlabelled_observed_apparatus',
            'Apply controllable input sequences and predict the measured response.',
            names, (), (), feature_bounds={name: (-1.5 * control_scale, 1.5 * control_scale)
                for name in names}, max_probe_length=64)
        self.channels = (MeasurementChannel(meter, 1, 'Read the auxiliary process instrument.'),)
        self._seed, self._random, self._experiments = seed, Random(seed), {}
        self._meter, self._width, self._placement, self._smooth = meter, width, placement, smooth
        self._threshold = threshold
        self._control_scale, self._meter_scale, self._response_scale = control_scale, meter_scale, response_scale
        self.observation_cost, self.measurement_cost, self.acquisitions = 0, 0, []

    def spec_for(self, channels=()):
        assert set(channels) <= {self._meter}
        return replace(self.spec, feature_names=(*self.spec.feature_names, *channels))

    def acquire(self, probe, channels=()):
        assert set(channels) <= {self._meter}
        assert all(set(point) == set(self.spec.feature_bounds) for point in probe.sequence)
        assert len(probe.sequence) <= self.spec.max_probe_length
        key = probe.probe_id, tuple(tuple(point[name] for name in self.spec.feature_names)
                                   for point in probe.sequence)
        if key not in self._experiments:
            raw, branches, readings = [], [], []
            for point in probe.sequence:
                x, z, bias = (point[name] / self._control_scale for name in self.spec.feature_names)
                reading = .35 * bias + self._random.uniform(-1., 1.)
                readings.append(reading)
                branches.append((.13 + .8 * x + .6 * z ** 2 + .25 * reading,
                                 -.11 - .9 * x - .45 * z ** 2 + .25 * reading))
                target = 0.
                for index in range(max(0, len(branches) - self._width), len(branches)):
                    gate = readings[index if self._placement == 'inside_window' else -1]
                    target += branches[index][0 if self._smooth or gate >= self._threshold else 1]
                raw.append(({**point, self._meter: self._meter_scale * reading},
                            self._response_scale * target))
            self._experiments[key] = raw
        observations, history = [], []
        for readings, target in self._experiments[key]:
            features = {name: readings[name] for name in self.spec_for(channels).feature_names}
            observations.append(Observation(features, target, tuple(history)))
            history.append({**features, '__target__': target})
        measurement_cost = len(observations) * len(channels)
        self.observation_cost += len(observations)
        self.measurement_cost += measurement_cost
        self.acquisitions.append({'probe': probe, 'channels': tuple(channels), 'observations': observations})
        return MeasurementBatch(observations, len(observations), measurement_cost, tuple(channels))

    def restore(self, acquisitions):
        self._random, self._experiments = Random(self._seed), {}
        self.observation_cost, self.measurement_cost, self.acquisitions = 0, 0, []
        for item in acquisitions:
            probe = item.get('probe', item)
            self.acquire(Probe(probe['probe_id'], tuple(probe['sequence'])), item.get('channels', ()))


CONFIG = ResearchConfig(observation_budget=320, compute_budget=320000,
    minimum_improvement=1e-7, max_rounds=4, active_probes=True, max_candidates=8)
FRESH_SEEDS = (9738101, 9739110)


def random_probe(spec, seed, length=32):
    random = Random(seed)
    return Probe(f'decoded_paid_{seed}', tuple({name: random.uniform(*bounds)
        for name, bounds in spec.feature_bounds.items()} for _ in range(length)))


def original_fresh_rows(laboratory):
    return [row for seed in FRESH_SEEDS for row in
        laboratory.acquire(random_probe(laboratory.spec, seed, 64), (laboratory._meter,)).observations]


def record_development(tmp_path, report, independent, name):
    fresh = original_fresh_rows(independent)
    measured_loss = loss(Program.from_dict(report['program']), fresh, report['fallback'], .5)
    payload = {'precision_goal': 1e-6, 'independent_mse': measured_loss,
               'precision_goal_pass': measured_loss <= 1e-6,
               'actual_width': independent._width, 'actual_placement': independent._placement,
               'actual_meter_threshold': independent._threshold * independent._meter_scale,
               'report': report}
    path = tmp_path / f'{name}.json.gz'
    with gzip.open(path, 'wt') as stream:
        json.dump(payload, stream)
    return fresh, measured_loss, path


def assert_actual_accounting(report, laboratory):
    assert report['costs']['observation_cost'] == laboratory.observation_cost <= CONFIG.observation_budget
    assert report['costs']['measurement_cost'] == laboratory.measurement_cost <= 320
    assert report['costs']['queries'] == len(laboratory.acquisitions)
    assert 0 < report['costs']['compute_cost'] <= CONFIG.compute_budget
    assert all(set(point) == set(laboratory.spec.feature_bounds)
        for item in laboratory.acquisitions for point in item['probe'].sequence)
    assert laboratory._meter not in laboratory.spec_for((laboratory._meter,)).feature_bounds


def assert_current_graph(report, fresh):
    assert report['research_program'] is not None
    graph = ResearchGraph.from_dict(json.loads(json.dumps(report['research_program']['graph'])))
    output = graph.nodes[report['research_program']['output_node']]
    compiled = bind_answers(output.expression, {key: graph.nodes[key].answer for key in output.depends_on})
    actual = Program.from_dict(report['program'])
    assert [compiled.predict(row) for row in fresh] == pytest.approx([actual.predict(row) for row in fresh])


@pytest.mark.parametrize('width,placement', [(3, 'inside_window'), (5, 'inside_window'), (3, 'outside_window')])
def test_actual_observed_condition_keeps_original_targets_and_legal_controls(width, placement, tmp_path):
    laboratory = DecodedObservedLab(width=width, placement=placement)
    report = UnifiedResearchAgent(replace(CONFIG, seed=9438101),
        library=KnowledgeLibrary(), measurement_budget=320).run(laboratory)
    fresh, measured_loss, saved = record_development(tmp_path, report,
        DecodedObservedLab(seed=9538101, width=width, placement=placement), f'{placement}_{width}')
    assert measured_loss <= 1e-6, str(saved)
    assert report['status'] == 'unresolved', str(saved)
    state = report['condition_uncertainty']
    assert state['status'] == 'unresolved'
    assert state['expected_prediction_ambiguity'] > CONFIG.minimum_improvement
    assert not report['concepts_added']
    decisions = [action for action in report['actions']
        if action['action'] == 'choose_condition_research_investment']
    assert decisions
    assert (any(action['action'] == 'plan_competing_observed_conditions' for action in report['actions'])
        or any(not action['buy'] and action['reason'] == 'retain_question_for_more_valuable_future_evidence'
               for action in decisions))
    assert report['selected_channels'] == ['meter'], str(saved)
    assert_actual_accounting(report, laboratory)
    assert_current_graph(report, fresh)
    if placement == 'inside_window':
        assert any(step.get('operation') == 'discover_decoded_conditional_coordinates'
            or (step.get('operation') == 'discover_conditional_coordinates'
                and step.get('experiment_evidence'))
            for step in report['research_program']['selection_steps']), str(saved)


def test_decoding_wrong_window_cannot_win_the_original_response_comparison(tmp_path):
    from hegel_machine.cognition.decoded_conditional_coordinates import propose_decoded_conditional_coordinates

    laboratory = DecodedObservedLab(width=3)
    training = [row for seed in (9438101, 9439110) for row in
        laboratory.acquire(random_probe(laboratory.spec, seed), ('meter',)).observations]
    validation = laboratory.acquire(random_probe(laboratory.spec, 9440119, 48), ('meter',)).observations
    spec = laboratory.spec_for(('meter',))
    fallback = initial_prediction(training, .5)
    budget = SearchBudget(320000)
    basis = _terms(spec.feature_names, quadratic=True)
    correct = propose_decoded_conditional_coordinates(spec, training, budget,
        basis=basis, gate_names={'meter'}, width_candidates=(3,), max_candidates=4)
    wrong = propose_decoded_conditional_coordinates(spec, training, budget,
        basis=basis, gate_names={'meter'}, width_candidates=(5,), max_candidates=4)
    assert correct
    correct_loss = min(loss(entry['candidate'].program, validation, fallback, .5) for entry in correct)
    wrong_loss = min((loss(entry['candidate'].program, validation, fallback, .5) for entry in wrong), default=float('inf'))
    with gzip.open(tmp_path / 'wrong_window_original_comparison.json.gz', 'wt') as stream:
        json.dump({'observation_cost': laboratory.observation_cost,
                   'measurement_cost': laboratory.measurement_cost, 'compute_cost': budget.spent,
                   'original_validation_loss': correct_loss,
                   'wrong_window_original_validation_loss': wrong_loss if wrong else None}, stream)
    assert correct_loss + 1e-6 < wrong_loss
    for entry in correct + wrong:
        assert entry['training_loss'] == pytest.approx(loss(entry['candidate'].program, training, fallback, .5))
    assert 0 < budget.spent <= budget.limit


def test_smooth_observed_response_retains_a_common_duration(tmp_path):
    laboratory = DecodedObservedLab(width=5, smooth=True)
    report = UnifiedResearchAgent(replace(CONFIG, seed=9438101),
        library=KnowledgeLibrary(), measurement_budget=320).run(laboratory)
    fresh, measured_loss, saved = record_development(tmp_path, report,
        DecodedObservedLab(seed=9538101, width=5, smooth=True), 'smooth_window5')
    assert report['status'] == 'completed' and measured_loss <= 1e-6, str(saved)
    assert_actual_accounting(report, laboratory)
    assert_current_graph(report, fresh)
    assert not any(step.get('operation') == 'discover_decoded_conditional_coordinates'
        for step in report['research_program']['selection_steps']), str(saved)


def test_json_restored_observed_method_changes_units_names_and_current_graph(tmp_path):
    library = KnowledgeLibrary()
    source = DecodedObservedLab(width=3)
    first = UnifiedResearchAgent(replace(CONFIG, seed=9438101), library=library,
        measurement_budget=320).run(source)
    _, _, first_saved = record_development(tmp_path, first,
        DecodedObservedLab(seed=9538101, width=3), 'restore_source')
    assert first['status'] == 'unresolved', str(first_saved)
    assert library.research_drafts and not first['concepts_added']
    restored = KnowledgeLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    settings = dict(width=5, placement='inside_window', names=('u', 'v', 'w'), meter='gauge',
                    threshold=-.271, control_scale=2.3, meter_scale=1.7, response_scale=1.25)
    changed = DecodedObservedLab(seed=9439110, **settings)
    report = UnifiedResearchAgent(replace(CONFIG, seed=9439110), library=restored,
        measurement_budget=320).run(changed)
    fresh, measured_loss, saved = record_development(tmp_path, report,
        DecodedObservedLab(seed=9539110, **settings), 'restored_renamed_scaled')
    assert measured_loss <= 1e-6, str(saved)
    assert report['status'] == 'unresolved', str(saved)
    assert report['condition_uncertainty']['expected_prediction_ambiguity'] > CONFIG.minimum_improvement
    assert report['selected_channels'] == ['gauge'], str(saved)
    assert_actual_accounting(report, changed)
    assert_current_graph(report, fresh)
    assert Program.from_dict(report['program']).required_features() <= set(changed.spec_for(('gauge',)).feature_names)


def test_unfinished_paid_response_survives_a_worse_raw_revision_and_json_restore():
    from test_cognition_observation_research import ActualPurchases
    from hegel_machine.cognition.observation_research import research_observation_actions
    from hegel_machine.cognition.research_programs import _Fitter, _entry

    # Fit the source using actual readings. The changed gain makes this an
    # imperfect response, while the uncontrolled reading remains useful.
    source = DecodedObservedLab(width=1, smooth=True)
    source_rows = source.acquire(random_probe(source.spec, 9768101, 48), ('meter',)).observations
    source_budget = SearchBudget(10000)
    fit = _Fitter(source_rows, source_budget, .5).fit(
        _terms(source.spec_for(('meter',)).feature_names, quadratic=True))
    assert fit is not None and fit['loss'] <= CONFIG.minimum_improvement
    entry = _entry(fit, [], len(source_rows), source_budget.spent)
    source_valid = source.acquire(random_probe(source.spec, 9769110, 24), ('meter',)).observations
    source_valid_loss = loss(fit['program'], source_valid, 0., .5)
    assert source_valid_loss <= CONFIG.minimum_improvement
    saved = {'program': fit['program'].to_dict(), 'channels': ['meter'], 'fallback': 0.,
             'kind': entry['kind'], 'family_id': entry['family_id'],
             'source_concept_ids': [], 'research_program': entry['research_program'],
             'training_loss': fit['loss'], 'validation_loss': source_valid_loss}

    class ChangedLab(DecodedObservedLab):
        @property
        def costs(self):
            return {'observation_cost': self.observation_cost,
                    'measurement_cost': self.measurement_cost}

    laboratory = ChangedLab(seed=9778101, width=1, smooth=True, response_scale=1.02)
    budget, actions = SearchBudget(120000), []
    agent = UnifiedResearchAgent(replace(CONFIG, use_memory=False), measurement_budget=96)
    purchases = ActualPurchases(laboratory, budget, actions,
        observation_budget=72, measurement_budget=96)
    for _ in range(2):
        assert purchases.acquire(('meter',), 'current_paid_evidence', rows_count=24) is not None

    def investigate(alternatives):
        return research_observation_actions(agent, laboratory, acquire=purchases.acquire,
            paired_for=purchases.paired_for, command_batches=purchases.command_batches,
            budget=budget, actions=actions, transfer=[], graph=ResearchGraph(),
            remaining=purchases.remaining, length=24, alternatives=alternatives)

    result = investigate(json.loads(json.dumps([saved])))
    assert result['selected']['candidate'].program == fit['program']
    assert not result['confirmed']
    assert max(result['selected']['training_loss'], result['selected']['validation_loss']) > CONFIG.minimum_improvement
    retained = [item for item in actions if item['action'] == 'retain_observed_response']
    assert retained and all(item['current_evidence_scored'] for item in retained)
    assert all(item['before_loss'] < item['after_loss'] and item['compute_cost'] > 0 for item in retained)
    assert purchases.batches[-1]['purpose'] == 'confirm_observation_research'
    assert purchases.batches[-1]['channels'] == ('meter',)

    payload = json.loads(json.dumps(result['alternatives']))
    purchases.observation_budget = 96
    restored = investigate(payload)
    assert not restored['confirmed']
    assert restored['selected']['candidate'].program == fit['program']
    assert any(Program.from_dict(item['program']) == fit['program'] for item in restored['alternatives'])
    assert laboratory.observation_cost == laboratory.measurement_cost == 96
    assert 0 < budget.spent <= budget.limit
