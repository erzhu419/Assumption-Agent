"""Usage-condition interventions restore the existing method, without research."""

from copy import deepcopy
from pathlib import Path

import pytest

from hegel_machine.cognition.theory_consolidation import _read_library
from scripts import prepare_pr3_source_successor as preparation
from scripts import run_anonymous_revision_inquiry as study
from test_prepare_frozen_successor import frozen_spec, prohibit_science
from test_prepare_pr3_source_successor import save
from test_research_program_consolidation import compile_sources


def condition_revision(tmp_path, *, thin=False):
    spec, _ = frozen_spec(tmp_path)
    learner, result = compile_sources(tmp_path / 'saved_methods')
    identifier = result['frozen_candidates'][0]['candidate_id']
    old = deepcopy(learner.theory_consolidation['candidates'][identifier])
    old['version'] = 2
    old_ref = save(tmp_path / 'method_v2.json', old)
    source = save(tmp_path / 'paid_use.json', {'status': 'completed'})
    outcomes = [{'method_ref': {'id': identifier, 'version': 2},
        'source_feedback_ref': source, 'status': 'completed', 'execution_complete': True,
        'profile': {}, 'profile_ref': save(tmp_path / 'public_profile.json', {}),
        'evidence_basis': {'paid_evidence_ref': source},
        'working_comparison': {'complete': True, 'same_rows': True,
            'before_loss': .1, 'after_loss': .02, 'data_role': 'exposed_development'}}]
    outcomes[0]['feedback_ref'] = save(tmp_path / 'use_0.json', outcomes[0])
    partial = save(tmp_path / 'first_conditions.json', outcomes)
    intermediate = deepcopy(old)
    intermediate.update(version=3, previous_version_ref=old_ref,
        method_use_conditions={'usage_version': {'id': identifier, 'version': 2},
            'outcomes_ref': partial}, trigger={'kind': 'observed_method_use_conditions'})
    intermediate_ref = save(tmp_path / 'method_v3.json', intermediate)
    outcomes.append({**deepcopy(outcomes[0]), 'source_feedback_ref': {**source, 'json_path': []},
        'working_comparison': {**outcomes[0]['working_comparison'], 'after_loss': .2}})
    outcomes[1]['feedback_ref'] = save(tmp_path / 'use_1.json', outcomes[1])
    if thin:
        from hegel_machine.cognition.method_feedback import _condition_state
        outcomes = [save(tmp_path / f'condition_use_{index}.json', _condition_state(outcome))
            for index, outcome in enumerate(outcomes)]
    current = deepcopy(intermediate)
    current.update(version=4, previous_version_ref=intermediate_ref,
        method_use_conditions={'usage_version': {'id': identifier, 'version': 2},
            'outcomes_ref': save(tmp_path / 'all_conditions.json', outcomes)})
    baseline = preparation._value(spec['materials']['initial_library_ref'])
    baseline['candidates'][identifier] = old
    baseline['executions'] = {'paid_method_fit': {'binding': {'candidate_id': identifier},
        'result': {'paid_fit_ref': source}}}
    baseline['transfers'] = [{'binding': {'candidate_id': identifier}, 'status': 'candidate_registered'}]
    save(Path(spec['materials']['initial_library_ref']['file']), baseline)
    updated = deepcopy(baseline)
    updated['candidates'][identifier] = current
    save(Path(spec['frozen_source']['library_ref']['file']), updated)
    spec['frozen_source']['new_candidate_ids'] = [identifier]
    spec['target']['research_program_fitting'] = {'validation_records': [2]}
    baseline_view = save(tmp_path / 'baseline_view.json',
        {'source_ref': spec['materials']['initial_library_ref']})
    updated_view = save(tmp_path / 'updated_view.json',
        {'source_ref': spec['frozen_source']['library_ref']})
    return spec, identifier, _read_library(baseline_view, tmp_path), _read_library(updated_view, tmp_path)


@pytest.mark.parametrize('thin', [False, True])
def test_preparation_restores_same_id_baseline_method_and_retains_paid_facts(tmp_path, monkeypatch, thin):
    spec, identifier, baseline, updated = condition_revision(tmp_path, thin=thin)
    prohibit_science(monkeypatch)
    result = preparation.prepare_frozen_successor(spec, tmp_path / 'prepared')
    assert result['qualified_local_proposal_count'] == 1
    qualification, = result['proposal_qualification']
    assert qualification['revision_kind'] == 'method_use_condition'
    assert qualification['baseline_version'] == 2 and qualification['version'] == 4
    assert qualification['completed_local_feedback_count'] == 2
    keep = _read_library(result['automatic_C_library_ref'], tmp_path)
    removed = _read_library(result['individual_object']['frozen_remove_library_ref'], tmp_path)
    assert keep == updated
    assert removed == baseline
    assert list(removed['candidates']) == list(keep['candidates'])
    manifest = preparation._value(result['individual_object']['manifest_ref'])
    assert manifest['removed_candidate_ids'] == []
    restored, = manifest['restored_candidate_versions']
    assert restored['id'] == identifier
    assert restored['removed_version'] == 4 and restored['restored_version'] == 2
    assert preparation._value(restored['restored_candidate_ref']) == baseline['candidates'][identifier]
    assert manifest['source_cost_ref'] == spec['frozen_source']['accounting_ref']
    assert not manifest['removed_execution_keys'] and not manifest['removed_transfer_indices']
    assert result['new_class_intervention']['status'] == 'UNKNOWN'
    assert not result['successor_execution_authorized']
    assert result['new_fits'] == result['new_predictions'] == result['new_observations'] == 0


def test_old_condition_arm_also_receives_the_complete_current_paid_cache(tmp_path, monkeypatch):
    spec, identifier, baseline, updated = condition_revision(tmp_path)
    paid_ref = updated['executions']['paid_method_fit']['result']['paid_fit_ref']
    updated['executions']['subsequent_shared_fit'] = {'binding': {'candidate_id': identifier,
        'candidate_version': 4}, 'result': {'paid_fit_ref': paid_ref}}
    updated['transfers'].append({'binding': {'candidate_id': identifier,
        'candidate_version': 4}, 'status': 'candidate_registered'})
    save(Path(spec['frozen_source']['library_ref']['file']), updated)
    prohibit_science(monkeypatch)
    prepared = preparation.prepare_frozen_successor(spec, tmp_path / 'prepared')
    keep = _read_library(prepared['automatic_C_library_ref'], tmp_path)
    restored = _read_library(prepared['individual_object']['frozen_remove_library_ref'], tmp_path)
    assert restored['candidates'] == baseline['candidates']
    assert restored['executions'] == keep['executions'] == updated['executions']
    assert restored['transfers'] == keep['transfers'] == updated['transfers']
    assert len(restored['candidates']) == len(keep['candidates'])


@pytest.mark.parametrize('changed', ['seed', 'lineage', 'feedback', 'unchanged', 'other_old'])
def test_only_addressed_condition_revisions_preserving_the_old_contract_are_prepared(
        tmp_path, monkeypatch, changed):
    spec, identifier, baseline, updated = condition_revision(tmp_path)
    current = updated['candidates'][identifier]
    if changed == 'seed':
        current['source_seed_inputs'] = current['source_seed_inputs'][:1]
    elif changed == 'lineage':
        current['previous_version_ref'] = save(tmp_path / 'wrong_old.json',
            {**baseline['candidates'][identifier], 'version': 1})
    elif changed == 'feedback':
        current['method_use_conditions']['outcomes_ref'] = save(tmp_path / 'no_feedback.json', [])
    elif changed == 'unchanged':
        updated['candidates'][identifier] = deepcopy(baseline['candidates'][identifier])
    else:
        updated['candidates']['old']['purpose'] = 'different_old_behavior'
    save(Path(spec['frozen_source']['library_ref']['file']), updated)
    prohibit_science(monkeypatch)
    with pytest.raises(ValueError):
        preparation.prepare_frozen_successor(spec, tmp_path / 'rejected')
    assert not (tmp_path / 'rejected/registration.json').exists()


def frozen_keep_fixture(tmp_path, monkeypatch):
    spec, identifier, baseline, updated = condition_revision(tmp_path)
    prohibit_science(monkeypatch)
    prepared = preparation.prepare_frozen_successor(spec, tmp_path / 'prepared')
    plan = preparation._value(prepared['whole_strategy']['plan_ref'])
    selected = preparation._value(prepared['individual_object']['plan_ref'])
    # Only runtime identity and intervention admission execute in this control.
    root = tmp_path / 'minimal_runtime'
    runtime = ['scripts/run_anonymous_revision_inquiry.py', 'src/hegel_machine/module.py']
    keep = tmp_path / 'completed_keep'
    for name in runtime:
        live, frozen = root / name, keep / 'source_at_run' / name
        live.parent.mkdir(parents=True, exist_ok=True)
        frozen.parent.mkdir(parents=True, exist_ok=True)
        live.write_text('# same frozen runtime\n')
        frozen.write_bytes(live.read_bytes())
    plan['runtime_snapshot'] = selected['runtime_snapshot'] = runtime
    save(keep / 'protocol.json', plan)
    save(keep / 'summary.json', {'status': 'completed', 'trials': [
        {'case': case['id'], 'arm': arm, 'status': 'retained_failure',
         'costs': {'compute_cost': 17}} for case in plan['cases'] for arm in study.ORGANIZATION_ARMS]})
    save(keep / 'pr3_theory_library.json', {'source_ref': prepared['automatic_C_library_ref']})
    save(keep / 'initial_archive.json', {'source_failures': ['retained'], 'paid_fits_ref':
        spec['frozen_source']['accounting_ref']})
    monkeypatch.setattr(study, 'ROOT', root)
    monkeypatch.setattr(study, 'AnonymousHistoryLab', lambda *a, **k: pytest.fail('created laboratory'))
    monkeypatch.setattr(study, 'evaluate_python_model', lambda *a, **k: pytest.fail('scored model'))
    return identifier, baseline, updated, selected, keep


def test_prepared_version_intervention_reaches_the_original_runner_without_scientific_work(
        tmp_path, monkeypatch):
    identifier, baseline, updated, selected, keep = frozen_keep_fixture(tmp_path, monkeypatch)
    calls = []

    def admit(plan, output, **kwargs):
        calls.append(kwargs)
        reference = plan['research_organization']['pr3_library_path']
        assert _read_library({'file': reference, 'json_path': []}, tmp_path) == baseline
        assert _read_library({'file': str(keep / 'pr3_theory_library.json'), 'json_path': []}, tmp_path) == updated
        assert kwargs['target_intervention']['executed_arms'] == ['pr3']
        assert kwargs['target_intervention']['keep_costs_recharged'] is False
        assert kwargs['knowledge']['source_failures'] == ['retained']
        assert plan['limits'] == preparation._read(keep / 'protocol.json')['limits']
        return {'status': 'identity_admitted_science_blocked', 'new_compute': 0}

    monkeypatch.setattr(study, '_run_study', admit)
    result = study.run_target_intervention(selected, tmp_path / 'not_executed', keep_study=keep,
        laboratory_factory=lambda _: pytest.fail('entered scientific execution'))
    assert result == {'status': 'identity_admitted_science_blocked', 'new_compute': 0}
    assert len(calls) == 1 and not (tmp_path / 'not_executed').exists()


@pytest.mark.parametrize('changed', ['missing_restore', 'wrong_version', 'changed_contract', 'lost_old'])
def test_version_intervention_rejects_unregistered_or_incomplete_restoration(
        tmp_path, monkeypatch, changed):
    identifier, baseline, updated, selected, keep = frozen_keep_fixture(tmp_path, monkeypatch)
    reference = selected['target_intervention']['manifest_ref']
    manifest = preparation._value(reference)
    if changed == 'missing_restore':
        manifest['restored_candidate_versions'] = []
    elif changed == 'wrong_version':
        manifest['restored_candidate_versions'][0]['restored_version'] = 3
    else:
        library = _read_library({'file': selected['research_organization']['pr3_library_path'],
            'json_path': []}, tmp_path)
        if changed == 'lost_old':
            library['candidates'].pop('old')
        else:
            library['candidates'][identifier]['procedure_contract'] = {'operation': 'other'}
        save(Path(selected['research_organization']['pr3_library_path']), library)
    save(Path(reference['file']), manifest)
    monkeypatch.setattr(study, '_run_study', lambda *a, **k: pytest.fail('invalid restoration ran'))
    with pytest.raises(ValueError):
        study.run_target_intervention(selected, tmp_path / 'rejected', keep_study=keep)
    assert not (tmp_path / 'rejected').exists()
