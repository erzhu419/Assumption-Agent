"""Paid interval continuation retains the pool, current evidence and original allowance."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from scripts import continue_successor_relation_intervals as study
from scripts import successor_interval_scene as scene_io
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, Probe, SearchBudget, TaskSpec
from hegel_machine.cognition.method_action_research import read_method_reference
from hegel_machine.cognition.object_continuation import decode_entry
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


def entry(program, name='old', **extra):
    return {'candidate': Candidate(name, program, 'retained'), 'kind': 'retained',
            'training_loss': 0., 'validation_loss': 0., 'fallback': 0., **extra}


def selector():
    agent = SimpleNamespace(config=SimpleNamespace(minimum_improvement=1e-7))
    agent._best = lambda claims: UnifiedResearchAgent._best(agent, claims)
    return agent


def law(radius):
    relation = Program('neighbor_sum', children=(Program('const', value=radius),
        *(Program('feature', name=name) for name in ('f0', 'x0', 'q0', 'f1', 'x1', 'q1'))))
    return Program('window_sum', lag=3, children=(relation,))


def rows(program, distances):
    history, result = [], []
    for distance in distances:
        fields = dict(f0=1., x0=0., q0=0., f1=0., x1=distance, q1=1.)
        row = Observation(fields, 0., tuple(history))
        result.append(replace(row, target=program.predict(row)))
        history.append({**fields, '__target__': result[-1].target})
    return result


def test_retained_pool_scores_only_missing_new_rows_once(monkeypatch):
    one, two, three = [Program('const', value=value) for value in (1., 2., 3.)]
    exposed = [Observation({}, 0.) for _ in range(48)]
    scene = {'paid_before': [None] * 197, 'exposed': exposed,
        'pool': [entry(one), entry(two), entry(three), entry(three, 'duplicate')],
        'old_scores': {one: 2., two: 3., three: 4.},
        'exposed_scores': {one: 1., two: 4.}, 'parent': {'fallback': 7.},
        'spec': SimpleNamespace(risk_quantile=.5)}
    observed = []

    def readout(program, actual_rows, fallback, quantile):
        assert actual_rows is exposed and program == three
        assert (fallback, quantile) == (7., .5)
        observed.append(program)
        return 9.

    monkeypatch.setattr(scene_io, 'loss', readout)
    budget = SearchBudget(48)
    scores, cost = scene_io.score_retained(scene, budget)
    assert observed == [three] and cost == budget.spent == 48
    assert scores == {one: (197 * 2 + 48) / 245,
        two: (197 * 3 + 48 * 4) / 245, three: (197 * 4 + 48 * 9) / 245}
    observed.clear()
    with pytest.raises(RuntimeError, match='not funded'):
        scene_io.score_retained(scene, SearchBudget(47))
    assert not observed


def test_new_probe_updates_all_unique_pool_programs_without_revisiting_old_rows(monkeypatch):
    one, two = Program('const', value=1.), Program('const', value=2.)
    acquired = [Observation({}, 0.) for _ in range(4)]
    scores, calls = {one: 2., two: 3.}, []

    def readout(program, actual, fallback, quantile):
        assert actual is acquired
        calls.append(program)
        return program.value ** 2

    monkeypatch.setattr(study, 'loss', readout)
    before = dict(scores)
    with pytest.raises(RuntimeError, match='complete retained pool'):
        study.extend_scores(scores, 245, acquired, SearchBudget(7), 0.)
    assert scores == before and calls == []
    budget = SearchBudget(8)
    study.extend_scores(scores, 245, acquired, budget, 0.)
    assert calls == [one, two] and budget.spent == 8
    assert scores == {one: (245 * 2 + 4) / 249, two: (245 * 3 + 16) / 249}


def test_old_supported_parameter_inherits_new_interval_without_refitting(monkeypatch):
    research = RelationIntervalResearch(law(.2), [], SearchBudget(10000),
        structure={'fits': [{'radius': .2}], 'radius_interval': [0., 1.]})
    original_rows = rows(law(.65), (.1, .9, .1, .9))
    research.alternatives(original_rows, SearchBudget(10000), tolerance=1e-7)
    symbol = research.parameters[0]['symbol']
    old = research._entry({symbol: .625}, [.1, .9], 0, 0., len(original_rows), 0)
    old_snapshot = deepcopy(old)
    exposed = rows(law(.65), (.6, .8, .6, .8))
    paid = [*original_rows, *exposed]
    candidates = research.alternatives(paid, SearchBudget(10000), tolerance=1e-7,
                                       experiments=[tuple(exposed)])
    assert candidates and all(item['candidate'].program != old['candidate'].program for item in candidates)
    assert research.intervals[0]['radius_interval'] == [.6, .8]
    monkeypatch.setattr(Program, 'predict', lambda *args: pytest.fail('Do not repeat paid prediction'))
    retained = study.retain_interval_evidence(research, old, candidates, len(paid), 1e-7)
    assert retained is not None and retained['candidate'].program == old['candidate'].program
    assert retained['parameter_values'] == old['parameter_values']
    step = retained['research_program']['selection_steps'][-1]
    assert step['parameters'][0]['radius_intervals'] == [[.6, .8]]
    assert len(step['evidence_contexts'][0]['trajectories'][0]) == len(exposed)
    assert old == old_snapshot
    pool = [old, *candidates, retained]
    selected = study.compare_pool(selector(), pool,
        {item['candidate'].program: 0. for item in pool}, 0.)
    assert selected['research_program'] == retained['research_program']
    assert len(pool) == len(candidates) + 2 and pool[0] == old_snapshot
    research.adopt(retained)
    assert research.intervals[0]['radius_interval'] == [.6, .8]


def test_complete_pool_can_retain_better_noninterval_explanation():
    original, new = entry(Program('const', value=1.)), entry(law(.7), 'interval')
    selected = study.compare_pool(selector(), [original, new],
        {original['candidate'].program: .01, new['candidate'].program: .1}, 3.)
    assert selected['candidate'] == original['candidate']
    assert selected['validation_loss'] == selected['training_loss'] == .01
    assert original['validation_loss'] == 0. and selected['fallback'] == 3.


def test_paid_pairs_keep_measured_history_and_reset_command_history():
    measured = rows(law(.6), (.1, .9, .1))
    first = scene_io.pair_rows([{'command': i} for i in range(3)], measured)
    second = scene_io.pair_rows([{'command': 99}], measured[:1])
    assert first[2][1] is measured[2] and len(first[2][1].history) == 2
    assert first[2][0].history == ({'command': 0, '__target__': measured[0].target},
                                  {'command': 1, '__target__': measured[1].target})
    assert second[0][0].history == ()


def test_registration_carries_remaining_original_allowance_and_threshold(tmp_path, monkeypatch):
    root, source, output = tmp_path / 'original', tmp_path / 'source', tmp_path / 'new'
    study.save(root / 'registration.json', {'observation_limit': 512, 'measurement_limit': 6144,
        'research_config': {'max_rounds': 8}})
    old = {'source': str(root), 'source_task_ref': {'file': 'source-task.json'},
        'total_compute_limit': 270008, 'previous_costs': {'observation_cost': 197, 'measurement_cost': 2364},
        'threshold': 1e-7, 'confirmation_rows': 48, 'confirmation_seed': 123,
        'active_wall_seconds': 1800, 'source_cost_refs': [{'file': 'old-paid.json'}]}
    receipt = {'full_comparison': True, 'confirmation': {'selected': {'passed': False}},
        'cumulative_compute': 187000, 'observation_cost': 48, 'measurement_cost': 576}
    study.save(source / 'registration.json', old)
    study.save(source / 'receipt.json', receipt)
    monkeypatch.setattr(study, 'freeze_runtime', lambda *args: None)
    monkeypatch.setattr(study, 'write_text_incrementally', lambda *args, **kwargs: None)
    config = study.register(source, output)
    assert config['remaining_compute'] + config['previous_compute'] == 270008
    assert config['threshold'] == 1e-7 and config['previous_observations'] == 245
    assert config['previous_measurement_cost'] == 2940
    assert config['source_cost_refs'][-1] == {'file': 'old-paid.json'}
    assert study.load(source / 'registration.json') == old
    assert study.load(source / 'receipt.json') == receipt


def test_candidate_dicts_keep_complete_values_and_continuous_shared_bases(tmp_path, monkeypatch):
    evidence = [{'trajectories': [[{'history': [{'field': index} for index in range(6000)],
                                  'features': {}, 'target': 0.}]]}]
    candidates = [entry(Program('const', value=float(index)), f'candidate:{index}',
        research_program={'selection_steps': [{'operation': 'resolve_relation_intervals',
            'radius': index / 10, 'evidence_contexts': deepcopy(evidence)}]}) for index in range(3)]
    write, calls = study.write_json_checkpoint, []

    def recorded(path, value, *, base=None):
        assert isinstance(value, dict)
        calls.append((path, base))
        return write(path, value, base=base)

    monkeypatch.setattr(study, 'write_json_checkpoint', recorded)
    first, base = study.save_interval_candidates(tmp_path, 1, candidates[:2])
    following, last = study.save_interval_candidates(tmp_path, 2, candidates[2:], previous=base)
    assert [item[1] for item in calls] == [None, calls[0][0], calls[1][0]]
    assert last == calls[-1][0]
    refs = [*first, *following]
    assert all('json_path' not in reference for reference in refs)
    assert [decode_entry(read_method_reference(reference, tmp_path)) for reference in refs] == candidates
    old_text = calls[0][0].read_bytes()
    changed = deepcopy(read_method_reference(first[1], tmp_path))
    changed['research_program']['selection_steps'][0]['evidence_contexts'][0]['trajectories'][0][0]['target'] = 9.
    write(calls[1][0], changed, base=calls[0][0])
    assert calls[0][0].read_bytes() == old_text
    assert decode_entry(read_method_reference(first[0], tmp_path)) == candidates[0]


def run_scene(tmp_path, monkeypatch, status):
    output = tmp_path / 'run'
    config = {'source': str(tmp_path / 'paid'), 'remaining_compute': 100000,
        'previous_compute': 170008, 'confirmation_rows': 48, 'threshold': 1e-7,
        'active_wall_seconds': 30, 'maximum_experiments': 2,
        'previous_observations': 245, 'observation_limit': 512,
        'previous_measurement_cost': 2940, 'measurement_limit': 6144,
        'confirmation_seed': 999, 'source_cost_refs': [{'file': 'paid-source.json'}]}
    study.save(output / 'registration.json', config)
    parent = entry(law(.8), structure={'fits': [{'radius': .8}], 'radius_interval': [0., 1.]})
    before = [Observation(rows(law(.8), (1.5,))[0].features, 0.) for _ in range(197)]
    exposed = rows(law(.8), (1.5,) * 48)
    paid = [*before, *exposed]
    spec = TaskSpec('public', 'Controls', tuple(sorted(parent['candidate'].program.required_features())),
        (), (), feature_bounds={name: (-2., 2.) for name in parent['candidate'].program.required_features()})
    agent = selector()
    agent.config.seed = 104
    agent.library = SimpleNamespace(meta_priors=[], instrument_models=[],
        program_experiment_methods=SimpleNamespace(to_dict=lambda: {}))
    agent._fit_instruments = lambda *args, **kwargs: None
    agent._instrument_binding = lambda *args: {}
    scene = {'agent': agent, 'spec': spec, 'paid': paid, 'pool': [parent],
        'refs': [{'file': 'old-selected.json'}], 'parent': parent, 'exposed': exposed,
        'paired_batches': [], 'original_registration': {'task': {'seed': 104}}}
    monkeypatch.setattr(study, 'ROOT', output / 'runtime')
    monkeypatch.setattr(study, 'restore_interval_scene', lambda *args: scene)
    monkeypatch.setattr(study, 'score_retained', lambda *args: ({parent['candidate'].program: 0.}, 0))
    observed_alternatives = []

    class RecordedResearch(RelationIntervalResearch):
        def alternatives(self, actual, budget, tolerance=1e-7, *, experiments=()):
            assert len(actual) == 245 and experiments == [tuple(exposed)]
            observed_alternatives.append(actual)
            return super().alternatives(actual, budget, tolerance, experiments=experiments)

    monkeypatch.setattr(study, 'RelationIntervalResearch', RecordedResearch)
    purchases = []

    class PublicLab:
        def __init__(self, **kwargs):
            self.observation_cost = self.measurement_cost = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def spec_for(self, channels):
            assert channels == ()
            return spec

        def acquire(self, probe, channels):
            assert probe.probe_id == 'distinction'
            purchases.append(probe)
            self.observation_cost += probe.cost
            self.measurement_cost += 12 * probe.cost
            from hegel_machine.cognition.unified_object_lab import ObjectMeasurementBatch
            return ObjectMeasurementBatch(rows(law(.8), (1.5,) * probe.cost),
                                          probe.cost, 12 * probe.cost, channels, [])

        def random_probe(self, *args, **kwargs):
            pytest.fail('Unresolved distinction must not trigger random confirmation')

    monkeypatch.setattr(study, 'AnonymousObjectLab', PublicLab)

    def plan(command_spec, candidates, budget, **kwargs):
        assert command_spec is spec and kwargs['tolerance'] == 1e-7 and kwargs['questions']
        probe = Probe('distinction', (dict(exposed[0].features),)) if status == 'predicted_distinction' else None
        return {'probe': probe, 'forecast_status': status,
            'predicted_separation': 1. if probe else 0.,
            'hypotheses': [{'program': candidate['candidate'].program.to_dict()} for candidate in candidates],
            'predicted_fields': {}}

    monkeypatch.setattr(study, 'plan_competing_object_probe', plan)
    monkeypatch.setattr(study, 'shorten_discriminating_probe', lambda plan, *args, **kwargs: plan)
    return output, purchases, observed_alternatives


def test_no_full_response_distinction_keeps_paid_result_without_new_purchase(tmp_path, monkeypatch):
    output, purchases, researched = run_scene(tmp_path, monkeypatch, 'no_predicted_distinction')
    receipt = study.run(output)
    assert researched and not purchases
    assert receipt['stop_reason'] == 'no_predicted_distinction'
    assert not receipt['confirmation_admitted'] and receipt['confirmation'] is None
    assert receipt['observation_cost'] == receipt['measurement_cost'] == 0
    assert receipt['cumulative_compute'] == 170008 + receipt['compute_cost'] <= 270008
    selected = study.load(output / 'selected_entry.json')
    evidence = selected['research_program']['selection_steps'][-1]['evidence_contexts']
    assert len(evidence[0]['trajectories'][0]) == 48


def test_purchased_trajectory_is_saved_before_assessment_failure(tmp_path, monkeypatch):
    output, purchases, _ = run_scene(tmp_path, monkeypatch, 'predicted_distinction')

    def interrupted(plan, pairs, budget, **kwargs):
        saved = study.load(output / 'observations/00.json')
        assert len(saved['batch']['observations']) == len(pairs) == 1
        assert saved['probe']['probe_id'] == 'distinction'
        raise RuntimeError('interrupted after paid acquisition')

    monkeypatch.setattr(study, 'assess_competing_object_probe', interrupted)
    with pytest.raises(RuntimeError, match='after paid acquisition'):
        study.run(output)
    assert len(purchases) == 1
    failure = study.load(output / 'failure.json')
    assert failure['observation_cost'] == 1 and failure['measurement_cost'] == 12
    assert failure['compute_known'] <= failure['compute_upper_bound'] == 100000
    assert failure['paid_candidate_refs'][0] == {'file': 'old-selected.json'}
