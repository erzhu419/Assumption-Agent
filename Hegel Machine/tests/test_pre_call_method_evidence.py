"""Pre-call diagnostics consume the selected paid model, never later arm answers."""

from copy import deepcopy
from dataclasses import asdict

import pytest

from scripts import return_partition_pair as returns
from scripts import validate_partition_successor as runner
from scripts.run_partial_role_parent import load, save
from hegel_machine.cognition import evaluation
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.object_continuation import encode_entry
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


def scene(tmp_path, *, rows=None, entries=None):
    source = tmp_path / 'source'
    if rows is None:
        rows = [Observation({'x': 1.}, 7.),
            Observation({'x': 2.}, 1., ({'x': 1., '__target__': 7.},)),
            Observation({'x': 5.}, 7.),
            Observation({'x': 9.}, 5., ({'x': 5., '__target__': 7.},))]
    if entries is None:
        lag = Program('lag', name='x', lag=1)
        entries = [{'candidate': Candidate(str(index), program, 'retained')} for index, program in enumerate((
            Program('add', children=(lag, Program('const', value=0.))), lag, Program('const', value=0.)))]
    features = tuple(rows[0].features)
    public = TaskSpec('fixture', 'paid fixture', features, (), (),
        feature_bounds={name: (-10., 10.) for name in features})
    save(source / 'public_spec.json', asdict(public))
    save(source / 'model.json', {'raw_sensor_features': list(features), 'raw_control_features': list(features)})
    scores = []
    for index, entry in enumerate(entries):
        save(source / f'model_{index}.json', encode_entry(entry))
        scores.append({'candidate_ref': runner.ref(source / f'model_{index}.json'),
            'loss': evaluation.loss(entry['candidate'].program, rows, 7., .5)})
    save(source / 'observations.json', {'batch': {'observations': [asdict(row) for row in rows]}})
    compared = {'scores': scores, 'data_refs': [runner.ref(source / 'observations.json')],
        'full_comparison': True, 'rows': len(rows)}
    save(source / 'common_comparison.json', compared)
    save(source / 'prepared_state.json', {'fallback': 7.,
        'public_spec_ref': runner.ref(source / 'public_spec.json')})
    cfg = {'research_config': {}, 'source_model_ref': runner.ref(source / 'model.json'),
        'quality_threshold': 1e-7, 'equal_method_compute_limit': 1000,
        'task': {'private_direction': 'must_never_become_a_feature'}}
    return source, cfg, compared, rows, entries


def prohibit_prediction(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('An already paid column was predicted again')
    monkeypatch.setattr(evaluation, 'predict', unexpected)


def test_original_fallback_resets_and_complexity_tie_select_the_actual_parent(tmp_path, monkeypatch):
    source, cfg, compared, rows, _ = scene(tmp_path)
    # A later arm's perfect answer is irrelevant to this earlier selection.
    save(source / 'Pi3/selected_entry.json', {'program': {'op': 'const', 'value': 999.}})
    budget = SearchBudget(len(rows))
    result = returns.ensure_pre_call_evidence(source, cfg, compared, tmp_path / 'evidence', budget)
    evidence = load(result['evidence_ref']['file'])
    assert evidence['model_ref'] == compared['scores'][1]['candidate_ref']
    assert evidence['predictions'] == [7., 1., 7., 5.]
    assert evidence['basis']['fallback'] == 7.
    assert evidence['transition_evidence']['correct_trajectories'] == 2
    assert evidence['transition_evidence']['error_rows'] == 0
    assert evidence['transition_evidence']['guard_depth'] is None
    assert evidence['transition_evidence']['startup_first_divergences'] is None
    assert evidence['parameter_groups']['status'] == 'binding_unavailable'
    assert evidence['parameter_groups']['count'] is None
    assert evidence['parameter_groups']['fit_columns'] is None
    assert budget.spent == result['new_compute_cost'] == result['new_predictions'] == 4
    prohibit_prediction(monkeypatch)
    cached = returns.ensure_pre_call_evidence(source, cfg, compared, tmp_path / 'evidence', SearchBudget(0))
    assert cached['evidence_ref'] == result['evidence_ref']
    assert cached['new_compute_cost'] == cached['new_predictions'] == 0
    assert not cached['budget_incomplete']


def test_paid_loss_prefilter_preserves_full_pool_and_original_winner_index(tmp_path, monkeypatch):
    source, cfg, compared, _, _ = scene(tmp_path)
    decoded = []
    original = runner.retained_entry
    def retained(location, base, index):
        assert index != 2, 'A candidate excluded by the paid loss was decoded'
        decoded.append(index)
        return original(location, base, index)
    monkeypatch.setattr(runner, 'retained_entry', retained)
    material = returns._pre_call_material(source, cfg, compared)
    assert decoded == [0, 1]
    assert material[5] == 1
    assert len(material[3]) == len(compared['scores']) == 3
    assert returns._pre_call_evidence_basis(source, cfg, compared, material)['full_pool_size'] == 3
    profile = returns.paid_pre_call_profile(source, cfg, compared)
    assert profile['research']['comparison']['models_requested'] == 3
    assert profile['research']['comparison']['models_completed'] == 3


def test_insufficient_budget_does_not_begin_partial_readout(tmp_path, monkeypatch):
    source, cfg, compared, rows, _ = scene(tmp_path)
    prohibit_prediction(monkeypatch)
    budget = SearchBudget(len(rows) - 1)
    result = returns.ensure_pre_call_evidence(source, cfg, compared, tmp_path / 'unstarted', budget)
    assert result['budget_incomplete'] and result['evidence_ref'] is None
    assert result['required_compute'] == len(rows)
    assert result['new_compute_cost'] == result['new_predictions'] == budget.spent == 0
    assert not (tmp_path / 'unstarted').exists()


def test_profile_only_consumes_saved_evidence_and_rejects_another_scoring_basis(tmp_path, monkeypatch):
    source, cfg, compared, _, _ = scene(tmp_path)
    result = returns.ensure_pre_call_evidence(source, cfg, compared, tmp_path / 'evidence', SearchBudget(4))
    prohibit_prediction(monkeypatch)
    original = returns.paid_pre_call_profile(source, cfg, compared)
    assert 'transition' not in original['residual']
    profile = returns.paid_pre_call_profile(source, cfg, compared, evidence_ref=result['evidence_ref'])
    assert profile['residual']['transition']['correct_trajectories'] == 2
    assert profile['research']['parameter_groups']['count'] is None
    compared['pre_call_evidence_ref'] = result['evidence_ref']
    assert returns.paid_pre_call_profile(source, cfg, compared) == profile
    state = load(source / 'prepared_state.json')
    save(source / 'prepared_state.json', {**state, 'fallback': 0.})
    with pytest.raises(ValueError, match='scoring contract'):
        returns.paid_pre_call_profile(source, cfg, compared)


def test_saved_score_column_is_reused_and_incomplete_reset_stays_unknown(tmp_path, monkeypatch):
    row = Observation({'x': 2.}, 1., ({'x': 1., '__target__': 7.},))
    source, cfg, compared, _, _ = scene(tmp_path, rows=[row])
    winner = UnifiedResearchAgent(ResearchConfig())._best([
        {**runner.retained_entry(item['candidate_ref'], source, index), 'validation_loss': item['loss']}
        for index, item in enumerate(compared['scores'])])
    index = next(i for i, item in enumerate(compared['scores'])
        if runner.retained_entry(item['candidate_ref'], source, i)['candidate'] == winner['candidate'])
    save(source / 'paid_predictions.json', [1.])
    compared['scores'][index]['predictions_ref'] = runner.ref(source / 'paid_predictions.json')
    prohibit_prediction(monkeypatch)
    result = returns.ensure_pre_call_evidence(source, cfg, compared, tmp_path / 'evidence', SearchBudget(0))
    assert result['new_compute_cost'] == result['new_predictions'] == 0
    evidence = load(result['evidence_ref']['file'])
    assert evidence['transition_evidence']['status'] == 'unknown'
    assert 'error_rows' not in evidence['transition_evidence']
    profile = returns.paid_pre_call_profile(source, cfg, compared, evidence_ref=result['evidence_ref'])
    assert profile['residual']['transition']['status'] == 'unknown'
    assert 'error_rows' not in profile['residual']['transition']


def test_future_complete_pool_keeps_winner_column_without_extra_prediction_or_cost(tmp_path, monkeypatch):
    source, cfg, compared, rows, entries = scene(tmp_path)
    entries.append(deepcopy(entries[1]))
    calls = []
    original = runner.predict
    def counted(program, row, fallback):
        calls.append(program)
        return original(program, row, fallback)
    monkeypatch.setattr(runner, 'predict', counted)
    budget = SearchBudget(3 * len(rows))
    assessed, values = runner._assess_pre_call_pool(UnifiedResearchAgent(ResearchConfig()),
        entries, rows, 7., .5, budget)
    assert len(calls) == budget.spent == 3 * len(rows)
    assert [item['validation_loss'] for item in assessed[:3]] == [item['loss'] for item in compared['scores']]
    assert values == [7., 1., 7., 5.]
    prohibit_prediction(monkeypatch)
    captured = returns.ensure_pre_call_evidence(source, cfg, compared, tmp_path / 'captured', budget,
        predictions=values)
    assert captured['new_compute_cost'] == captured['new_predictions'] == 0
    assert budget.spent == len(calls) == 3 * len(rows)


def test_parameter_summary_comes_from_actual_research_program_not_private_task(tmp_path):
    from test_transfer_partitions import plan, observations
    from hegel_machine.cognition.structural_research_program import research_program_fit_contract

    procedure = plan(.4)
    entry = {'candidate': Candidate('bound', research_program_fit_contract(procedure)['program'], 'retained'),
        'research_program': procedure}
    source, cfg, compared, rows, _ = scene(tmp_path, rows=observations(.4, weight=1., count=2), entries=[entry])
    values = [evaluation.predict(entry['candidate'].program, row, 7.) for row in rows]
    result = returns.ensure_pre_call_evidence(source, cfg, compared, tmp_path / 'evidence', SearchBudget(0),
        predictions=values)
    summary = load(result['evidence_ref']['file'])['parameter_groups']
    assert summary['status'] == 'ready' and summary['count'] == 1
    assert summary['fit_columns'] == len(procedure['term_nodes'])
    assert summary['groups'][0]['term_indices'] == [1]
    cfg['task']['private_direction'] = 'different_private_mechanism'
    profile = returns.paid_pre_call_profile(source, cfg, compared, evidence_ref=result['evidence_ref'])
    assert profile['research']['parameter_groups'] == summary
    assert 'private_direction' not in str(profile)


@pytest.mark.parametrize('replayed', [False, True])
def test_actual_available_guard_is_retained_even_inside_state_replay(tmp_path, replayed):
    model = Program('if', children=(Program('available', lag=1),
        Program('lag', name='x', lag=1), Program('const', value=7.)))
    if replayed:
        model = Program('state_replay', children=(model,))
    source, cfg, compared, _, _ = scene(tmp_path,
        entries=[{'candidate': Candidate('guarded', model, 'retained')}])
    result = returns.ensure_pre_call_evidence(source, cfg, compared,
        tmp_path / 'evidence', SearchBudget(4))
    transition = load(result['evidence_ref']['file'])['transition_evidence']
    assert transition['guard_depth'] == 1
    assert transition['startup_first_divergences'] == 0
    assert transition['mature_first_divergences'] == 0


def test_two_saved_binding_hops_restore_only_the_actual_retained_contract(tmp_path):
    from test_transfer_partitions import plan, observations
    from hegel_machine.cognition.research_drafts import _rebind_step
    from hegel_machine.cognition.structural_research_program import research_program_fit_contract

    procedure = plan(.4)
    program = research_program_fit_contract(procedure)['program']
    forward = {'drive': 'middle_drive', 'x1': 'middle_x1'}
    reverse_role = {'final_drive': 'middle_drive', 'final_x1': 'middle_x1'}
    inverse = {value: key for key, value in reverse_role.items()}
    final_body = _rebind_step(_rebind_step(procedure, forward), inverse)
    final_program = research_program_fit_contract(final_body)['program']
    paid_rows = [Observation({inverse.get(forward.get(key, key), forward.get(key, key)): value
        for key, value in row.features.items()}, row.target)
        for row in observations(.4, weight=1., count=2)]
    source, cfg, compared, rows, _ = scene(tmp_path, rows=paid_rows,
        entries=[{'candidate': Candidate('bound', final_program, 'retained')}])
    save(source / 'origin.json', encode_entry({'candidate': Candidate('origin', program, 'retained'),
        'research_program': procedure}))
    save(source / 'middle.json', encode_entry({
        'candidate': Candidate('middle', program.remap_features(forward), 'retained'),
        'source_candidate_ref': runner.ref(source / 'origin.json'), 'feature_mapping': forward}))
    save(source / 'role_binding.json', reverse_role)
    selected = load(source / 'model_0.json')
    selected.update(source_candidate_ref=runner.ref(source / 'middle.json'),
        role_binding_ref=runner.ref(source / 'role_binding.json'))
    save(source / 'model_0.json', selected)
    result = returns.ensure_pre_call_evidence(source, cfg, compared,
        tmp_path / 'evidence', SearchBudget(len(rows)))
    evidence = load(result['evidence_ref']['file'])
    assert evidence['parameter_groups']['status'] == 'ready'
    assert evidence['parameter_groups']['count'] == 1
    assert len(evidence['research_program_source_refs']) == 3

    selected['role_binding_ref'] = runner.ref(source / 'wrong_roles.json')
    save(source / 'wrong_roles.json', {'unrelated': 'middle_drive'})
    save(source / 'model_0.json', selected)
    cfg['source_model_ref'] = runner.ref(source / 'origin.json')
    missing = returns.ensure_pre_call_evidence(source, cfg, compared,
        tmp_path / 'unbound', SearchBudget(len(rows)))
    summary = load(missing['evidence_ref']['file'])['parameter_groups']
    assert summary['status'] == 'binding_unavailable'
    assert summary['count'] is None and summary['fit_columns'] is None


def test_saved_prediction_column_must_match_the_original_winner_score(tmp_path):
    source, cfg, compared, _, _ = scene(tmp_path)
    with pytest.raises(ValueError, match='actual paid winner score'):
        returns.ensure_pre_call_evidence(source, cfg, compared,
            tmp_path / 'evidence', SearchBudget(0), predictions=[0., 1., 0., 5.])
    assert not (tmp_path / 'evidence/pre_call_evidence.json').exists()
