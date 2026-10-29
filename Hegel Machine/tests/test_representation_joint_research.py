"""A recovered historical distinction must survive selection and change the relation."""

from copy import deepcopy
from dataclasses import replace
from random import Random

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.representation_diagnostics import representation_ambiguity
from hegel_machine.cognition.representation_inquiry import load_native_representation_result
from hegel_machine.cognition.temporal_operator_agent import (
    export_temporal_program, _PROGRAM_REPRESENTATION_RUNNER,
)
from test_representation_counterexample_coordinates import setup


def _key(row):
    return tuple(sorted(row.features.items())), tuple(
        tuple(sorted(point.items())) for point in row.history)


def _interaction_scene(tmp_path, monkeypatch):
    # Reuse the real Common route and the existing omitted-distinction fixture.
    # The two unrelated old generators are disabled by setup, so a joint answer
    # must consume the witness coordinates rather than arrive through them.
    agent, parent, arguments, lab = setup(tmp_path, monkeypatch)
    agent.config = replace(agent.config, compute_budget=200000)
    for row in agent.rows:
        row['target'] = row['features']['u'] * (
            row['history'][-1]['u'] if row['history'] else 0.)
    namespace = {}
    exec(export_temporal_program(parent) + _PROGRAM_REPRESENTATION_RUNNER, namespace)
    diagnostic = representation_ambiguity(
        [namespace['represent'](row) for row in agent.rows], agent.rows)
    assert diagnostic['conflicting_groups'] > 0
    arguments['research_question']['representation_diagnostic'] = diagnostic
    arguments['native_representation_plan']['compute_limit'] = 90000
    return agent, parent, arguments, lab


def test_incumbent_coordinate_repair_reaches_validation_before_candidate_cap(monkeypatch):
    from hegel_machine.cognition import representation_inquiry

    # Isolate the diagnosed train-only prefilter from the new relation search.
    monkeypatch.setattr(representation_inquiry, '_witness_relation_claims',
        lambda *args, **kwargs: [], raising=False)
    rng = Random(240)
    training, validation = [], []
    for destination in (training, validation):
        for index in range(30):
            u, previous, good = (rng.uniform(-1., 1.) for _ in range(3))
            if index < 2:
                u, good = .25, .3  # Same incumbent/current view, different history.
            target = 2. * good + .7 * previous
            if destination is training:
                target += .03 * rng.uniform(-1., 1.)
            # This source is exact on training but reverses on the next trajectory.
            bad = target if destination is training else -target
            destination.append(Observation({'u': u, 'good': good, 'bad': bad},
                target, ({'u': previous},)))
    parents = {'train_only': Program('feature', name='bad'),
        'incumbent': Program('feature', name='good')}
    budget, actions = SearchBudget(10000), []
    claims = representation_inquiry._witness_coordinate_claims(
        TaskSpec('selection_fixture', '', ('u',), (), ()), training, parents,
        [[0, 1]], budget, actions, max_candidates=1, validation=validation, fallback=0.)
    assert budget.completed_fits >= 2
    assert len(claims) == 1
    assert claims[0]['validation_loss'] < .001
    assert claims[0]['training_loss'] > 1e-8  # The zero-training-loss rival lost.
    steps = claims[0]['research_program']['selection_steps']
    assert any(step.get('parent_theory_id') == 'incumbent' for step in steps)
    assert 0 < budget.spent <= budget.limit


def test_common_witness_coordinate_drives_joint_predictor_without_validation_fitting(
        tmp_path, monkeypatch):
    from hegel_machine.cognition.research_programs import _Fitter
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent

    agent, parent, arguments, lab = _interaction_scene(tmp_path, monkeypatch)
    original = (agent.workspace / 'old.py').read_bytes()
    train = [Observation(row['features'], row['target'], tuple(row['history']))
        for record in agent.observations[:4] for row in record['rows']]
    fitting_targets = {_key(row): row.target for row in train}
    fits = []
    assessed = []
    original_init = _Fitter.__init__
    original_assess = UnifiedResearchAgent._assess

    def fit_init(self, rows, *args, **kwargs):
        rows = list(rows)
        assert rows and all(_key(row) in fitting_targets for row in rows)
        assert all(row.target == fitting_targets[_key(row)] for row in rows)
        fits.append(len(rows))
        original_init(self, rows, *args, **kwargs)

    monkeypatch.setattr(_Fitter, '__init__', fit_init)

    def assess(self, claims, validation, *args, **kwargs):
        claims = list(claims)
        assessed.extend(entry['candidate'].program for entry in claims)
        return original_assess(self, claims, validation, *args, **kwargs)

    monkeypatch.setattr(UnifiedResearchAgent, '_assess', assess)
    result = agent._execute('run', arguments, lab)
    native = load_native_representation_result(result['native_representation_ref'], agent.workspace)
    assert fits and all(count == 48 for count in fits)
    assert len(assessed) == len(set(assessed)), 'A paid validation was repeated after export selection'
    assert native['selected']['validation_mse'] < 1e-10
    assert not native['selected']['retained']
    assert any(action['action'] == 'compose_witness_relations' for action in native['actions'])
    namespace = {}
    exec((agent.workspace / result['selected_model']).read_text(), namespace)
    # These predictions were neither fitting rows nor the declared validation.
    rng, history = Random(537), []
    for _ in range(25):
        current = rng.uniform(-1., 1.)
        row = {'features': {'u': current}, 'history': list(history)}
        expected = current * (history[-1]['u'] if history else 0.)
        assert namespace['predict'](row) == pytest.approx(expected, abs=1e-7)
        history.append({'u': current})
    assert (agent.workspace / 'old.py').read_bytes() == original
    assert native['all_parent_candidates_retained']
    assert agent.costs['observation_cost'] == agent.costs['model_calls'] == 0
    assert 0 < native['compute_cost'] <= arguments['native_representation_plan']['compute_limit']


def test_joint_result_reuses_its_completed_paid_work(tmp_path, monkeypatch):
    agent, _, arguments, lab = _interaction_scene(tmp_path, monkeypatch)
    first = agent._execute('run', arguments, lab)
    before = deepcopy(agent.costs)
    second = agent._execute('run', arguments, lab)
    assert second['cache_hit'] and second['new_compute_cost'] == 0
    assert second['exported_models'] == first['exported_models']
    assert agent.costs == before


def test_overlapping_old_direction_reuses_witness_validation(tmp_path, monkeypatch):
    from hegel_machine.cognition.models import Candidate
    from hegel_machine.cognition.unified_research import UnifiedResearchAgent

    first_agent, _, first_arguments, lab = _interaction_scene(tmp_path / 'first', monkeypatch)
    first = first_agent._execute('run', first_arguments, lab)
    evidence = load_native_representation_result(
        first['native_representation_ref'], first_agent.workspace)
    shared = Program.from_dict(evidence['selected']['program'])
    agent, _, arguments, lab = _interaction_scene(tmp_path / 'overlap', monkeypatch)
    # Existing directions can independently reach exactly the same Program.
    # Keep its original provenance but reuse the later paid witness assessment.
    monkeypatch.setattr(UnifiedResearchAgent, '_coordinate_claims',
        lambda *args: [{'candidate': Candidate('earlier_direction', shared, 'fixture'),
            'training_loss': evidence['selected']['training_mse']}])
    assessments = []
    original_assess = UnifiedResearchAgent._assess

    def assess(self, claims, validation, *args, **kwargs):
        claims = list(claims)
        assessments.extend(entry['candidate'].program for entry in claims)
        return original_assess(self, claims, validation, *args, **kwargs)

    monkeypatch.setattr(UnifiedResearchAgent, '_assess', assess)
    result = agent._execute('run', arguments, lab)
    assert assessments.count(shared) == 1
    native = load_native_representation_result(result['native_representation_ref'], agent.workspace)
    assert native['selected']['validation_mse'] < 1e-10


def test_unfunded_joint_unit_preserves_parent_and_does_not_charge_unexecuted_search(
        tmp_path, monkeypatch):
    agent, _, arguments, lab = _interaction_scene(tmp_path, monkeypatch)
    arguments['native_representation_plan']['compute_limit'] = 20
    original = (agent.workspace / 'old.py').read_bytes()
    result = agent._execute('run', arguments, lab)
    native = load_native_representation_result(result['native_representation_ref'], agent.workspace)
    assert native['selected'] is None and native['candidates'] == []
    assert native['compute_cost'] == 0
    assert native['required_comparison_compute'] > 20
    assert (agent.workspace / 'old.py').read_bytes() == original
    assert agent.costs['observation_cost'] == agent.costs['model_calls'] == 0


def test_joint_exception_preserves_completed_fitting_cost_and_evidence(tmp_path, monkeypatch):
    from hegel_machine.cognition import representation_inquiry

    agent, parent, arguments, _ = _interaction_scene(tmp_path, monkeypatch)
    question = deepcopy(arguments['research_question'])

    def interrupted(spec, training, parents, coordinates, budget, actions, **kwargs):
        assert coordinates and budget.spend(137)
        raise RuntimeError('joint execution interrupted after paid work')

    monkeypatch.setattr(representation_inquiry, '_witness_relation_claims', interrupted)
    with pytest.raises(RuntimeError, match='joint execution interrupted'):
        representation_inquiry.execute_native_representation_inquiry(agent, question,
            spec=TaskSpec('interrupted_fixture', '', ('u',), (), ()),
            parent_programs={'old': parent}, validation_records=[4],
            compute_limit=90000, max_candidates=4)
    receipt = question['native_representation_research'][-1]
    native = load_native_representation_result(receipt['source_ref'], agent.workspace)
    assert native['status'] == 'native_representation_execution_failed'
    assert native['compute_cost'] == agent.costs['compute_cost'] > 137
    witness = next(action for action in native['actions']
        if action['action'] == 'research_missing_public_distinctions')
    assert any('training_mse' in attempt for attempt in witness['attempts'])
    assert native['selected'] is None
    assert agent.costs['observation_cost'] == agent.costs['model_calls'] == 0
