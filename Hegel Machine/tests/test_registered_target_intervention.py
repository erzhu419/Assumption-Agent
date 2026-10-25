"""Single-C interventions reuse the original registered finite study loop."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import method_effect_query
from hegel_machine.cognition.theory_consolidation import _read_library
from scripts import run_anonymous_revision_inquiry as study
from test_registered_native_author import (
    DRIVER, SOURCE, LocalLab, knowledge, paid_prefix, plan, registration, runtime_root,
)
from test_theory_consolidation import agent_scene, candidate, consolidate


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {'file': str(path), 'json_path': []}


@pytest.fixture
def effects(monkeypatch):
    calls = []

    def read(*args, **kwargs):
        calls.append((args, kwargs))
        return {'status': 'controlled_readout', 'new_executions': 0}

    monkeypatch.setattr(method_effect_query, 'registered_successor_effect', read, raising=False)
    return calls


def frozen_pair(tmp_path, *, failed=False, runtime=()):
    source, lab = agent_scene(tmp_path / 'source_fixture')
    consolidate(source, lab)
    identifier = candidate(source)['id']
    base = _read_library({'file': str(source.workspace / 'shared/theory_library.json'),
                         'json_path': []}, source.workspace)['candidates'][identifier]
    old = {**deepcopy(base), 'id': 'old'}
    new = {**deepcopy(base), 'id': 'new'}
    alias = {**deepcopy(base), 'id': 'new_alias'}
    fixed = write(tmp_path / 'old_library.json', {'candidates': {'old': old}})
    keep_library = write(tmp_path / 'keep_library.json', {
        'candidates': {'old': old, 'new': new, 'new_alias': alias}})
    organization = registration() | {'phase': 'target', 'initial_library_path': fixed,
                                    'pr3_library_path': keep_library}
    configured = plan(registered=organization)
    configured['runtime_snapshot'] = sorted(set(configured['runtime_snapshot']) | set(runtime))
    configured['source_cost_ref'] = write(tmp_path / 'closed_costs.json', {'source_compute': 17})
    driver = "def select_actions(spec, observations, memory): return {'stop': 'Retained failure.'}\n" if failed else SOURCE
    keep = tmp_path / 'keep'
    study.run_study(configured, keep, knowledge=knowledge(driver),
        prepare_prefix=paid_prefix, laboratory_factory=lambda _: LocalLab())
    return configured, keep


@pytest.mark.parametrize('changed', [False, True])
def test_registered_preparation_callback_is_frozen_and_paired(tmp_path, runtime_root, effects, changed):
    callback = 'scripts/prepare_pr3_source_successor.py'
    live = runtime_root / callback
    live.write_text('# current registered prefix and C-index callback\n')
    configured, keep = frozen_pair(tmp_path, runtime=(callback,))
    assert (keep / 'source_at_run' / callback).read_bytes() == live.read_bytes()
    selected = removed_plan(tmp_path, configured)
    if changed:
        live.write_text('# changed prefix or C-index callback\n')
        with pytest.raises(ValueError, match='same current frozen runtime'):
            study.run_target_intervention(selected, tmp_path / 'removed', keep_study=keep,
                laboratory_factory=lambda _: pytest.fail('changed callback reached laboratory'))
        assert not (tmp_path / 'removed').exists()
    else:
        result = study.run_target_intervention(selected, tmp_path / 'removed', keep_study=keep,
            prepare_prefix=paid_prefix, laboratory_factory=lambda _: LocalLab())
        assert [row['arm'] for row in result['trials']] == ['pr3']
        assert result['trials'][0]['terminal']['passed']


def removed_plan(tmp_path, configured, *, kind='individual_object', remove=('new',)):
    result = deepcopy(configured)
    result['research_organization']['pr3_library_path'] = write(tmp_path / 'removed_library.json', {
        'source_ref': configured['research_organization']['pr3_library_path'],
        'removed_candidates': list(remove)})
    manifest = {'status': 'frozen_ready', 'kind': kind, 'removed_candidate_ids': list(remove),
                'preserved_old_candidate_ids': ['old']}
    result['target_intervention'] = {'kind': kind,
        'manifest_ref': write(tmp_path / 'removal_manifest.json', manifest)}
    return result


@pytest.mark.parametrize('kind,removed', [
    ('individual_object', ('new',)), ('new_redundancy_class', ('new', 'new_alias'))])
def test_only_fresh_C_runs_with_removed_library_same_inputs_and_original_finish(
        tmp_path, runtime_root, effects, kind, removed):
    configured, keep = frozen_pair(tmp_path)
    selected = removed_plan(tmp_path, configured, kind=kind, remove=removed)
    original_common = (keep / 'initial_archive.json').read_bytes()
    original_new = (keep / 'case_000/pr3/new.py').read_bytes()
    calls = []

    def factory(*args, **kwargs):
        assert kwargs['arm'] == 'pr3'
        # Only the common initial input is supplied. No paid keep prefix,
        # chosen parent, revision or terminal outcome is installed in the agent.
        assert kwargs['knowledge']['files'] == knowledge()['files']
        agent = study.RegisteredOrganizationAgent(*args, **kwargs)
        assert agent.rows == [] and agent.revisions == [] and agent.model is None
        assert set(agent.theory_consolidation['candidates']) == {'old', 'new', 'new_alias'} - set(removed)
        calls.append(agent)
        return agent

    destination = tmp_path / 'removed'
    result = study.run_target_intervention(selected, destination, keep_study=keep,
        prepare_prefix=paid_prefix, laboratory_factory=lambda _: LocalLab(), agent_factory=factory)
    assert len(calls) == 1 and [row['arm'] for row in result['trials']] == ['pr3']
    row, = result['trials']
    assert row['fresh_confirmation_passed'] and row['terminal']['passed']
    assert row['costs']['observation_cost'] == 4  # Original2prefix/probe +2confirmation.
    assert row['costs']['model_calls'] == row['costs']['reported_tokens'] == 0
    assert result['target_intervention']['keep_costs_recharged'] is False
    assert len(result['target_intervention']['keep_outcome_refs']) == 3
    assert result['research_effects_ref']['file'] == str(destination / 'research_effects.json')
    assert effects[-1][0] == (str(keep), destination)
    assert effects[-1][1]['left_arm'] == effects[-1][1]['right_arm'] == 'pr3'
    assert effects[-1][1]['source_cost_ref'] == configured['source_cost_ref']
    assert not (destination / 'case_000/archive').exists()
    assert not (destination / 'case_000/fixed_pr2').exists()
    assert (keep / 'initial_archive.json').read_bytes() == original_common
    assert (destination / 'initial_archive.json').read_bytes() == original_common
    (destination / 'case_000/pr3/new.py').write_text('# independent subsequent fit\n')
    assert (keep / 'case_000/pr3/new.py').read_bytes() == original_new


def test_failed_keep_outcomes_remain_valid_pairs_and_are_not_rerun(tmp_path, runtime_root, effects):
    configured, keep = frozen_pair(tmp_path, failed=True)
    selected = removed_plan(tmp_path, configured)
    result = study.run_target_intervention(selected, tmp_path / 'removed', keep_study=keep,
        prepare_prefix=paid_prefix, laboratory_factory=lambda _: LocalLab())
    row, = result['trials']
    assert row['model'] is None and row['terminal']['mse'] is None
    assert not row['fresh_confirmation_passed']
    assert row['terminal']['evaluation_rows'] == 0 and row['terminal']['python_execution_seconds'] == 0.
    assert 'prediction_ref' not in row['terminal']
    assert len(result['target_intervention']['keep_outcome_refs']) == 3
    assert result['target_intervention']['executed_arms'] == ['pr3']


@pytest.mark.parametrize('changed', ['budget', 'cases', 'policy', 'runtime', 'old_candidate', 'order',
                                     'paid_score_refs'])
def test_unmatched_or_weakened_intervention_stops_before_any_new_lab(
        tmp_path, runtime_root, effects, changed):
    configured, keep = frozen_pair(tmp_path)
    selected = removed_plan(tmp_path, configured)
    if changed == 'budget':
        selected['limits']['compute_budget'] += 1
    elif changed == 'cases':
        selected['cases'][0]['prefix_seed'] += 1
    elif changed == 'policy':
        selected['research_organization']['exploration_policy']['seed'] += 1
    elif changed == 'runtime':
        (runtime_root / 'src/hegel_machine/current_module.py').write_text('# changed common algorithm\n')
    elif changed == 'paid_score_refs':
        selected['paid_score_index_refs_by_case'] = {
            'case_000': {'file': str(tmp_path / 'different_score_index.json'), 'json_path': []}}
    elif changed == 'order':
        library = _read_library(configured['research_organization']['pr3_library_path'], tmp_path)
        remaining = {key: library['candidates'][key] for key in ('new_alias', 'old')}
        selected['research_organization']['pr3_library_path'] = write(
            tmp_path / 'reordered_library.json', {'candidates': remaining})
    else:
        selected = removed_plan(tmp_path, configured, remove=('old', 'new'))
    with pytest.raises(ValueError):
        study.run_target_intervention(selected, tmp_path / 'rejected', keep_study=keep,
            laboratory_factory=lambda _: pytest.fail('unmatched plan created a laboratory'))
    assert not (tmp_path / 'rejected').exists()


def test_original_study_does_not_accept_an_unregistered_single_C_shortcut(tmp_path, runtime_root, effects):
    organization = registration() | {'phase': 'target', 'initial_library_path': '', 'pr3_library_path': ''}
    configured = plan(registered=organization)
    configured['arms'] = ['pr3']
    with pytest.raises(ValueError, match='register archive, fixed_pr2 and pr3 together'):
        study.run_study(configured, tmp_path / 'invalid', knowledge=knowledge(),
            laboratory_factory=lambda _: pytest.fail('single-arm shortcut created a laboratory'))


def test_readout_failure_keeps_completed_training_and_original_costs(tmp_path, runtime_root, effects, monkeypatch):
    configured, keep = frozen_pair(tmp_path)
    selected = removed_plan(tmp_path, configured)
    monkeypatch.setattr(method_effect_query, 'registered_successor_effect',
        lambda *a, **k: (_ for _ in ()).throw(ValueError('retained readout unavailable')))
    destination = tmp_path / 'removed'
    result = study.run_target_intervention(selected, destination, keep_study=keep,
        prepare_prefix=paid_prefix, laboratory_factory=lambda _: LocalLab())
    assert result['status'] == 'completed' and result['trials'][0]['costs']['observation_cost'] == 4
    effect = json.loads((destination / 'research_effects.json').read_text())
    assert effect['status'] == 'UNKNOWN' and effect['training_and_evaluation_retained']
    assert (destination / 'case_000/pr3/training.json').is_file()


def test_completed_real_loop_automatically_writes_the_read_only_paired_effect(tmp_path, runtime_root, monkeypatch):
    evaluate = study.evaluate_python_model
    calls = []

    def counted(*args, **kwargs):
        calls.append(args)
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(study, 'evaluate_python_model', counted)
    configured, keep = frozen_pair(tmp_path)
    selected = removed_plan(tmp_path, configured)
    destination = tmp_path / 'removed'
    result = study.run_target_intervention(selected, destination, keep_study=keep,
        prepare_prefix=paid_prefix, laboratory_factory=lambda _: LocalLab())
    effect = json.loads(Path(result['research_effects_ref']['file']).read_text())
    assert effect['status'] == 'recorded_comparison'
    assert effect['left_arm'] == effect['right_arm'] == 'pr3'
    assert effect['paired_terminal_case_count'] == effect['registered_case_count'] == 1
    assert effect['cases'][0]['right_minus_left']['terminal_mse'] == 0.
    terminal = effect['cases'][0]['selected_parent_terminal_predictions']
    assert terminal['status'] == 'recorded' and terminal['changed'] is False and terminal['rows'] == 3
    assert effect['new_method_executions'] == effect['new_predictions'] == 0
    assert effect['source_costs']['ledger_ref'] == configured['source_cost_ref']
    assert json.loads((keep / 'research_effects.json').read_text())['right_arm'] == 'pr3'
    assert len(calls) == 4  # Three original terminal calls plus one fresh removal C.
    row, = result['trials']
    column = json.loads(Path(row['terminal']['prediction_ref']['file']).read_text())
    training_path = destination / 'case_000/pr3/training.json'
    assert column['training_ref'] == {'file': str(training_path), 'json_path': []}
    assert column['model_ref'] == {'file': str(training_path), 'json_path': ['model']}
    assert column['model'] == json.loads(training_path.read_text())['model']
    assert len(column['predictions']) == row['terminal']['evaluation_rows'] == 3
    assert row['terminal']['python_execution_seconds'] > 0.
    assert row['costs'] == json.loads(training_path.read_text())['costs']
    assert not {'targets', 'observations', 'histories'}.intersection(column)
    assert 'predictions' not in row['terminal']


def test_completed_recovery_column_references_the_actual_old_training_workspace(tmp_path, runtime_root, effects):
    configured, keep = frozen_pair(tmp_path)
    recovered = tmp_path / 'recovered'
    result = study.run_study(configured, recovered, previous_study=keep,
        prepare_prefix=lambda *a: pytest.fail('completed training must not buy a new prefix'),
        agent_factory=lambda *a, **k: pytest.fail('completed training must not construct an agent'),
        laboratory_factory=lambda _: LocalLab())
    assert len(result['trials']) == 3
    for row in result['trials']:
        column = json.loads(Path(row['terminal']['prediction_ref']['file']).read_text())
        original = keep / row['case'] / row['arm'] / 'training.json'
        assert column['training_ref'] == {'file': str(original), 'json_path': []}
        assert column['model'] == json.loads(original.read_text())['model']
        assert not (recovered / row['case'] / row['arm'] / 'training.json').exists()
