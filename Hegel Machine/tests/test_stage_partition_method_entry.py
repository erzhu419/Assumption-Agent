"""Paid stage verification must consume actual fitted outputs on their own split."""

from copy import deepcopy
from dataclasses import asdict

import pytest

from scripts import stage_partition_method as runner
from scripts.run_partial_role_parent import load, save
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, TaskSpec
from hegel_machine.cognition.object_continuation import encode_entry


def fixture(tmp_path, monkeypatch):
    source, output = tmp_path / 'paid', tmp_path / 'new'
    x = Program('feature', name='x')
    twice = Program('mul', children=(Program('const', value=2.), x))
    bodies = {arm: {'graph': {'nodes': {'term': {'answer': program.to_dict()}}},
        'term_nodes': ['term'], 'selection_steps': []} for arm, program in [('P0', x), ('Pi1', twice)]}
    staged = {**deepcopy(bodies['P0']), 'staged_research': {
        'operation': 'refit_then_reidentify', 'baseline': bodies['P0'], 'continuation': bodies['Pi1'],
        'condition': Program('const', value=1.).to_dict(), 'coordinate_paths': []}}
    save(tmp_path / 'method.json', staged)
    training = [Observation({'x': 1.}, 2.), Observation({'x': 2.}, 4.)]
    working = [Observation({'x': 3.}, 6.)]
    for label, rows in [('training', training), ('working', working)]:
        save(source / (label + '.json'), {'batch': {'observations': [asdict(row) for row in rows]}})
    public = TaskSpec('paid_source', 'retained split', ('x',), (), ())
    save(source / 'public.json', asdict(public))
    save(source / 'prepared_state.json', {'training_refs': [runner.ref(source / 'training.json')],
        'validation_refs': [runner.ref(source / 'working.json')],
        'public_spec_ref': runner.ref(source / 'public.json'), 'fallback': 3.})
    save(source / 'registration.json', {'source_task_ref': {'source_id': 'paid_source'},
        'quality_threshold': 1e-7, 'research_config': {'max_candidates': 4}})
    save(source / 'frozen_methods.json', bodies)
    save(source / 'receipt.json', {'compute_cost': 100, 'remaining_compute': 999})
    old = {'candidate': Candidate('complete_old_pool', Program('const', value=0.), 'retained'),
        'kind': 'retained', 'family_id': 'T03', 'training_loss': 10.}
    save(source / 'old.json', encode_entry(old))
    save(source / 'common_comparison.json', {'full_comparison': True, 'rows': 3,
        'scores': [{'candidate_ref': runner.ref(source / 'old.json'), 'loss': 56. / 3.}]})
    for arm, program, loss in [('P0', x, 14. / 3.), ('Pi1', twice, 0.)]:
        entry = {'candidate': Candidate(arm, program, 'research_program'),
            'kind': 'research_program', 'family_id': 'T03',
            'training_loss': 2.5 if arm == 'P0' else 0., 'research_program': bodies[arm]}
        save(source / arm / 'new_entry.json', encode_entry(entry))
        save(source / arm / 'outcome.json', {'execution_complete': True, 'selected_loss': 0.})
        save(source / arm / 'comparison.json', {'full_comparison': True,
            'new_scores': [{'candidate_ref': runner.ref(source / arm / 'new_entry.json'), 'loss': loss}]})
    save(tmp_path / 'allocation.json', {'allocation_id': 'paid_stage_development',
        'maximum_compute': 60000, 'formation_reserve': 10000, 'new_observation_limit': 0})
    monkeypatch.setattr(runner, 'freeze_entry', lambda *args: None)
    monkeypatch.setattr(runner, 'write_text_incrementally', lambda *args: None)
    return source, output, runner.ref(tmp_path / 'method.json'), runner.ref(tmp_path / 'allocation.json')


def test_recovered_stage_score_belongs_to_actual_fit_not_retained_pool_winner(tmp_path, monkeypatch):
    source, *_ = fixture(tmp_path, monkeypatch)
    scene = runner.paid_scene(source)
    assert scene['paid_stages']['baseline']['feedback_loss'] == pytest.approx(9.)
    assert scene['paid_stages']['baseline']['entry']['candidate'].candidate_id == 'P0'
    assert scene['paid_stages']['continuation']['feedback_loss'] == 0.
    assert len(scene['frame']['assessed']) == 1
    assert len(scene['training']) == 2 and len(scene['working']) == 1
    assert len(scene['paid_stages']['baseline']['feedback_loss_refs']) == 2


def test_registration_preserves_same_bodies_full_pool_and_prior_costs(tmp_path, monkeypatch):
    source, output, method, allocation = fixture(tmp_path, monkeypatch)
    old = load(source / 'receipt.json')
    cfg = runner.prepare_paid([source], method, allocation, output)
    assert cfg['execution_compute_limit'] == 50000
    assert cfg['sources'][0]['pool_size'] == 1
    assert load(output / 'frozen_method.json') == load(tmp_path / 'method.json')
    assert load(source / 'receipt.json') == old
    assert not (output / 'observations').exists()


def test_mismatched_stage_body_cannot_reuse_another_method_result(tmp_path, monkeypatch):
    source, output, method, allocation = fixture(tmp_path, monkeypatch)
    body = load(tmp_path / 'method.json')
    body['staged_research']['baseline']['term_nodes'] = []
    save(tmp_path / 'method.json', body)
    with pytest.raises(ValueError, match='same frozen stages'):
        runner.prepare_paid([source], method, allocation, output)
    assert not output.exists()


@pytest.mark.parametrize('complete', [True, False])
def test_ordinary_replay_reuses_scoring_and_incomplete_stage_is_not_success(tmp_path, monkeypatch, complete):
    source, output, method, allocation = fixture(tmp_path, monkeypatch)
    runner.prepare_paid([source], method, allocation, output)
    monkeypatch.setattr(runner, 'ROOT', (output / 'runtime').resolve())
    calls = []

    def replay(payload, training, budget, *, feedback_rows, paid_stages, **kwargs):
        calls.append(payload)
        assert training == paid_stages['baseline']['training']
        assert feedback_rows == paid_stages['continuation']['feedback']
        assert paid_stages['baseline']['feedback_loss'] == pytest.approx(9.)
        budget.spend(1)
        entry = deepcopy(paid_stages['continuation']['entry'])
        entry['staged_research_result'] = {'execution_complete': complete,
            'budget_incomplete': not complete, 'compute_cost': 1, 'selected_stage': 'continuation'}
        return entry

    monkeypatch.setattr(runner, 'replay_research_program', replay)
    result = runner.run_paid(output)
    assert len(calls) == 1 and result['compute_cost'] == 1
    assert result['sources'][0]['comparison_compute_cost'] == 0
    assert result['sources'][0]['full_comparison']
    assert result['status'] == ('paid_stage_paths_completed' if complete
        else 'paid_stage_path_incomplete_not_negative')
    assert result['independent_support'] is False
    assert runner.run_paid(output)['new_compute_cost'] == 0 and len(calls) == 1


def test_failed_new_development_keeps_cost_and_old_records(tmp_path, monkeypatch):
    source, output, method, allocation = fixture(tmp_path, monkeypatch)
    runner.prepare_paid([source], method, allocation, output)
    monkeypatch.setattr(runner, 'ROOT', (output / 'runtime').resolve())

    def fail(payload, training, budget, **kwargs):
        budget.spend(7)
        raise RuntimeError('paid-stage execution failed')

    monkeypatch.setattr(runner, 'replay_research_program', fail)
    with pytest.raises(RuntimeError, match='execution failed'):
        runner.run_paid(output)
    assert load(output / 'failure.json')['compute_cost'] == 7
    assert load(source / 'receipt.json')['remaining_compute'] == 999
    assert not (output / 'receipt.json').exists()


def test_successor_waits_for_complete_paid_paths_and_keeps_full_pool_and_identity(tmp_path, monkeypatch):
    source, paid, method, allocation = fixture(tmp_path, monkeypatch)
    runner.prepare_paid([source], method, allocation, paid)
    target = tmp_path / 'held'
    prior = tmp_path / 'topic'
    save(paid / 'receipt.json', {'status': 'paid_stage_path_incomplete_not_negative'})
    with pytest.raises(ValueError, match='Both ordinary paid'):
        runner.prepare_successor(paid, source, allocation, prior, target)
    assert not target.exists()
    save(paid / 'receipt.json', {'status': 'paid_stage_paths_completed'})
    original = load(source / 'registration.json')
    original.update(allocation_id='condition_validation_held', task={
        'seed': 106030301, 'objects': 4, 'relation_radius': .8},
        frozen_conditions_ref=runner.ref(source / 'old_conditions.json'))
    save(source / 'registration.json', original)
    body = load(source / 'frozen_methods.json')['Pi1']
    save(tmp_path / 'old_pi1.json', {'id': 'actual_old_pi1', 'version': 1,
        'execution': {'research_program_ref': runner.ref(source / 'frozen_methods.json', 'Pi1')}})
    save(source / 'method_return/challenger_outcome.json', {
        'method_artifact_ref': runner.ref(tmp_path / 'old_pi1.json')})
    save(tmp_path / 'validation_allocation.json', {'allocation_id': 'staged_validation_held',
        'maximum_compute': 250000, 'new_observation_limit': 168, 'seed': 106030401,
        'training_rows': 96, 'working_rows': 24, 'confirmation_rows': 48})
    save(prior / 'common/questions.json', [{'id': 'task:method_choice', 'method_use_plan': {}}])
    with pytest.raises(ValueError, match='retain this actual formed staged proposal'):
        runner.prepare_successor(paid, source, runner.ref(tmp_path / 'validation_allocation.json'), prior, target)
    assert not target.exists()
    proposal = tmp_path / 'formed_method.json'
    save(proposal, {'proposal': {'candidates': [{'research_program': load(tmp_path / 'method.json')}]}})
    developed = load(paid / 'registration.json')
    developed['method_source_ref'] = runner.ref(proposal, 'proposal', 'candidates', 0, 'research_program')
    save(paid / 'registration.json', developed)
    save(prior / 'common/questions.json', [{'id': 'task:method_choice', 'method_use_plan': {
        'staged_method_proposal_refs': [runner.ref(proposal)]}}])
    cfg = runner.prepare_successor(paid, source, runner.ref(tmp_path / 'validation_allocation.json'), prior, target)
    assert cfg['task']['relation_radius'] == .8 and cfg['task']['seed'] == 106030401
    assert cfg['source_pool_ref'] == runner.ref(paid.parent / 'staged_successor_source_pool.json')
    assert cfg['source_pool_size'] == 3
    assert cfg['common_comparison_upper'] == 3 * 120
    assert cfg['confirmation_compute'] == 144 and cfg['prefix_readout']
    assert cfg['total_new_compute_upper'] <= 250000
    assert 'frozen_conditions_ref' not in cfg
    assert load(target / 'frozen_methods.json')['P0'] == body
    assert cfg['native_method_artifact_refs']['P0'] == runner.ref(tmp_path / 'old_pi1.json')
    from scripts.return_partition_pair import bind_methods
    bound = bind_methods(target, target, artifact_refs=cfg['native_method_artifact_refs'])
    assert bound['baseline']['method_ref']['id'] == 'actual_old_pi1'
    assert bound['challenger']['method_ref']['id'].startswith('native:staged:')
    assert load(source / 'receipt.json')['remaining_compute'] == 999


def test_successor_pool_adds_both_fixed_sources_without_filtering_bad_models(tmp_path, monkeypatch):
    first, *_ = fixture(tmp_path / 'first', monkeypatch)
    second, *_ = fixture(tmp_path / 'second', monkeypatch)
    failed = load(second / 'P0/comparison.json')
    failed['new_scores'][0].update(loss=99., status='failed_quality_threshold')
    save(second / 'P0/comparison.json', failed)
    developed = {'sources': [{'source': str(first)}, {'source': str(second)}]}
    pool = runner.successor_pool(developed)
    assert pool['original_pool_size'] == 1 and pool['added_source_candidates'] == 4
    assert len(pool['scores']) == 5
    refs = [item['candidate_ref'] for item in pool['scores']]
    assert refs[0] == runner.ref(first / 'old.json')
    assert runner.ref(second / 'P0/new_entry.json') in refs
    assert all(item['loss'] is None for item in pool['scores'])
    # Registration is independent of which target will consume the fixed pool.
    assert runner.successor_pool(developed) == pool
    assert pool['scores'][-2]['source_score_ref'] == runner.ref(
        second / 'P0/comparison.json', 'new_scores', 0)


def test_prefix_predictor_frozen_before_shared_readout_without_changing_pair_pass(tmp_path, monkeypatch):
    from dataclasses import dataclass
    from scripts import run_learned_structure_successor as confirm
    from hegel_machine.cognition.models import Probe
    constant = Program('const', value=2.)
    for arm in ('P0', 'Pi1'):
        save(tmp_path / arm / 'selected.json', encode_entry({
            'candidate': Candidate(arm, constant, 'retained')}))
        save(tmp_path / arm / 'outcome.json', {'execution_complete': True,
            'full_comparison': True, 'selected_ref': runner.ref(tmp_path / arm / 'selected.json')})
    save(tmp_path / 'prepared_state.json', {'fallback': 0.})
    save(tmp_path / 'prefix.json', {'staged_prefix': {'program': Program('const', value=0.).to_dict(),
        'fallback': 0., 'role': 'frozen_actual_prefix'}})

    @dataclass
    class Batch:
        observations: list

    class Lab:
        def __init__(self, **kwargs):
            self.costs = {'observation_cost': 0, 'measurement_cost': 0, 'queries': 0}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def random_probe(self, seed, length, name):
            return Probe(name, ({'x': 1.},) * length)

        def acquire(self, probe, channels):
            frozen = load(tmp_path / 'confirmation/frozen_predictors.json')
            assert set(frozen) == {'P0', 'Pi1', 'staged_prefix'}
            self.costs.update(observation_cost=2, queries=1)
            return Batch([Observation({'x': 1.}, 2.), Observation({'x': 2.}, 2.)])

    monkeypatch.setattr(confirm, 'AnonymousObjectLab', Lab)
    result = confirm.confirm_pair(tmp_path, {'operations': {'P0': 'program_replay', 'Pi1': 'program_replay'},
        'diagnostic_predictors_ref': runner.ref(tmp_path / 'prefix.json'), 'task': {},
        'confirmation_rows': 2, 'confirmation_seed': 1, 'quality_threshold': 1e-7})
    assert result['all_passed'] and not result['arms']['staged_prefix']['passed']
    assert result['diagnostic_arms'] == ['staged_prefix']
    assert result['shared_actual_costs']['compute_cost'] == 6
    assert result['shared_actual_costs']['observation_cost'] == 2


def staged_contrast(tmp_path, monkeypatch):
    source, output, method, allocation = fixture(tmp_path, monkeypatch)
    methods = load(source / 'frozen_methods.json')
    (source / 'P0').rename(source / 'prefix')
    (source / 'Pi1').rename(source / 'P0')
    (source / 'prefix').rename(source / 'Pi1')
    for arm in ('P0', 'Pi1'):
        compared = load(source / arm / 'comparison.json')
        compared['new_scores'][0]['candidate_ref'] = runner.ref(source / arm / 'new_entry.json')
        save(source / arm / 'comparison.json', compared)
    prefix = load(source / 'Pi1/new_entry.json')
    outcome = load(source / 'Pi1/outcome.json')
    # The selected old-pool score is deliberately misleading; use prefix only.
    outcome.update(selected_loss=0., staged_research_result={
        'execution_complete': True, 'selected_stage': 'baseline',
        'baseline': {'execution_complete': True, 'program': prefix['candidate']['program'],
            'training_loss': 2.5, 'feedback_loss': 9., 'all_paid_loss': 14. / 3.}})
    save(source / 'Pi1/outcome.json', outcome)
    save(source / 'frozen_methods.json', {'P0': methods['Pi1'], 'Pi1': load(tmp_path / 'method.json')})
    return source, output, method, allocation


def test_paid_staged_counterexample_keeps_actual_prefix_fixed_continuation_and_entire_pool(tmp_path, monkeypatch):
    source, *_ = staged_contrast(tmp_path, monkeypatch)
    scene = runner.paid_scene(source)
    assert scene['paid_stages']['baseline']['entry']['candidate'].candidate_id == 'P0'
    assert scene['paid_stages']['baseline']['feedback_loss'] == 9.
    assert scene['paid_stages']['continuation']['entry']['candidate'].candidate_id == 'Pi1'
    assert scene['stage_evidence']['baseline']['joint_loss'] == pytest.approx(14. / 3.)
    assert len(scene['frame']['assessed']) == 3
    assert len(scene['paid_scores']) == 2
    assert scene['methods']['Pi1'] == load(source / 'frozen_methods.json')['P0']


def test_paid_staged_counterexample_checks_own_candidate_instead_of_old_winner(tmp_path, monkeypatch):
    source, *_ = staged_contrast(tmp_path, monkeypatch)
    outcome = load(source / 'Pi1/outcome.json')
    outcome['staged_research_result']['baseline']['program'] = Program('const', value=4.).to_dict()
    save(source / 'Pi1/outcome.json', outcome)
    with pytest.raises(ValueError, match='actual prefix'):
        runner.paid_scene(source)


def test_paid_revised_condition_reuses_both_actual_stages_and_pool_scores(tmp_path, monkeypatch):
    source, output, method, allocation = staged_contrast(tmp_path, monkeypatch)
    runner.prepare_paid([source], method, allocation, output)
    monkeypatch.setattr(runner, 'ROOT', (output / 'runtime').resolve())
    def replay(payload, training, budget, *, paid_stages, **kwargs):
        assert paid_stages['baseline']['entry']['candidate'].candidate_id == 'P0'
        assert paid_stages['continuation']['entry']['candidate'].candidate_id == 'Pi1'
        budget.spend(1)
        entry = deepcopy(paid_stages['continuation']['entry'])
        entry['staged_research_result'] = {'execution_complete': True,
            'compute_cost': 1, 'selected_stage': 'continuation'}
        return entry
    monkeypatch.setattr(runner, 'replay_research_program', replay)
    result = runner.run_paid(output)
    assert result['compute_cost'] == 1
    assert result['sources'][0]['old_pool_size'] == 3
    assert result['sources'][0]['comparison_compute_cost'] == 0
    assert result['sources'][0]['selected_loss'] == 0.
    assert result['independent_support'] is False


def test_successor_accepts_actual_retained_revision_but_still_requires_fresh_protocol(tmp_path, monkeypatch):
    source, paid, method, allocation = fixture(tmp_path, monkeypatch)
    runner.prepare_paid([source], method, allocation, paid)
    save(paid / 'receipt.json', {'status': 'paid_stage_paths_completed'})
    revision = tmp_path / 'revision.json'
    save(revision, {'research_program': load(tmp_path / 'method.json')})
    cfg = load(paid / 'registration.json')
    cfg['method_source_ref'] = runner.ref(revision, 'research_program')
    save(paid / 'registration.json', cfg)
    prior = tmp_path / 'topic'
    save(prior / 'common/questions.json', [{'id': 'task:method_choice', 'method_use_plan': {
        'staged_method_revision_refs': [runner.ref(revision)]}}])
    save(tmp_path / 'new_protocol.json', {'allocation_id': 'not_yet_registered_new_source', 'seed': 123})
    with pytest.raises(ValueError, match='predeclared new sources'):
        runner.prepare_successor(paid, source, runner.ref(tmp_path / 'new_protocol.json'), prior, tmp_path / 'unregistered_successor')
    assert not (tmp_path / 'unregistered_successor').exists()


def test_actual_paid_staged_replay_never_calls_fitter_or_evaluator(tmp_path, monkeypatch):
    import hegel_machine.cognition.evaluation as evaluation
    import hegel_machine.cognition.transfer_partitions as partitions
    from hegel_machine.cognition.models import SearchBudget
    source, *_ = staged_contrast(tmp_path, monkeypatch)
    methods = load(source / 'frozen_methods.json')
    contract = methods['Pi1']['staged_research']
    for label, arm, kind in [('baseline', 'Pi1', 'Pi0'), ('continuation', 'P0', 'Pi1')]:
        contract[label]['transfer_partition'] = {'kind': kind}
        entry = load(source / arm / 'new_entry.json')
        entry['transfer_partition_search'] = {'completed': True}
        save(source / arm / 'new_entry.json', entry)
    methods['P0'] = deepcopy(contract['continuation'])
    save(source / 'frozen_methods.json', methods)
    training = load(source / 'training.json')
    training['batch']['observations'] *= 3
    save(source / 'training.json', training)
    common = load(source / 'common_comparison.json')
    common.update(rows=7)
    save(source / 'common_comparison.json', common)
    prefix = load(source / 'Pi1/outcome.json')
    prefix['staged_research_result']['baseline']['all_paid_loss'] = 24. / 7.
    save(source / 'Pi1/outcome.json', prefix)
    comparison = load(source / 'Pi1/comparison.json')
    comparison['new_scores'][0]['loss'] = 24. / 7.
    save(source / 'Pi1/comparison.json', comparison)
    scene = runner.paid_scene(source)
    monkeypatch.setattr(partitions, 'replay_transfer_partition',
        lambda *a, **kw: pytest.fail('repeated paid fit'))
    monkeypatch.setattr(evaluation, 'loss', lambda *a, **kw: pytest.fail('repeated paid prediction'))
    budget = SearchBudget(1)
    entry = runner.replay_research_program(methods['Pi1'], scene['training'], budget,
        feedback_rows=scene['working'], paid_stages=scene['paid_stages'])
    result = entry['staged_research_result']
    assert result['execution_complete'] and result['selected_stage'] == 'continuation'
    assert result['baseline']['reused'] and result['continuation']['reused']
    assert budget.spent == 1


def test_paid_stage_accepts_roundoff_between_stored_sse_and_mean_without_rescoring(tmp_path, monkeypatch):
    from math import nextafter
    source, *_ = staged_contrast(tmp_path, monkeypatch)
    compared = load(source / 'Pi1/comparison.json')
    original = compared['new_scores'][0]['loss']
    compared['new_scores'][0]['loss'] = nextafter(original, 0.)
    save(source / 'Pi1/comparison.json', compared)
    scene = runner.paid_scene(source)
    assert scene['stage_evidence']['baseline']['joint_loss'] == nextafter(original, 0.)
    assert scene['paid_stages']['baseline']['feedback_loss'] == 9.
