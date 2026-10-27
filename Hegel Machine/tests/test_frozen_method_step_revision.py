"""Actual step-proof identity and a fresh-scene consumer, without a science run."""

from copy import deepcopy
from pathlib import Path

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.research_drafts import _rebind_step
from hegel_machine.cognition.research_programs import _plan, research_program_replay_inputs
from hegel_machine.cognition.theory_consolidation import _read_library
from scripts import prepare_pr3_source_successor as prepare
from scripts import run_anonymous_revision_inquiry as study
from test_frozen_method_condition_revision import condition_revision, frozen_keep_fixture
from test_prepare_pr3_source_successor import save


def step_candidate(tmp_path, old):
    identifier = old['id']
    identity = {'id': identifier, 'version': old['version']}
    revised = deepcopy(old)
    revised.update(version=old['version'] + 1,
        previous_version_ref=save(tmp_path / 'old_method.json', old))
    revised.pop('method_use_conditions', None)
    revised['trigger']['learned_utility_trigger'] = False
    revised['research_program_source_refs'], revised['source_seed_inputs'] = [], []
    study_path = tmp_path / 'step_study.json'
    study_ref = {'file': str(study_path), 'json_path': []}
    seed_studies = []
    for index, original in enumerate(old['research_program_source_refs']):
        feature = next(iter(old['source_outcomes'][index]['input_mapping'].values()))
        term = Program('mul', children=(Program('feature', name=feature),) * 2)
        recipe = _plan({'fit_terms': (term,), 'coefficients': [0., 1.], 'program': term,
            'loss': .01}, [{'operation': 'fit_component', 'question': 'Fit the measured squared coordinate',
                'program': term, 'loss': .01}], 48, 100)
        seed = {'basis': {'source_refs': [original], 'training_rows': 48, 'validation_rows': 24,
            'per_method_compute': 1000, 'minimum_improvement': 1e-10},
            'methods': [{'index': 0, 'execution_complete': True, 'program': {'old': index}},
                {'index': 1, 'execution_complete': True, 'program': term.to_dict(),
                 'research_program': recipe}],
            'pairs': [{'method_index': 1, 'edit': {'operation': 'add_square'},
                'execution_complete': True, 'same_training_and_validation': True,
                'equal_method_compute_ceiling': 1000, 'validation_loss_change': -.01,
                'status': 'local_step_improvement'}]}
        seed_studies.append(seed)
        payload = {'research_program': recipe, 'program': term.to_dict(),
            'parent_source_ref': original, 'data_role': 'exposed_development',
            'paired_replay_ref': {**study_ref, 'json_path': ['seed_studies', index]}}
        reference = save(tmp_path / f'step_payload_{index}.json', payload)
        revised['research_program_source_refs'].append(reference)
        revised['source_outcomes'][index]['source_ref'] = reference
        revised['source_seed_inputs'].append({'source_id': old['source_ids'][index],
            'source_ref': reference, 'retained_term_refs': []})
    save(study_path, {'status': 'development_step_revision_selected',
        'baseline_method_ref': identity, 'data_role': 'exposed_development',
        'seed_studies': seed_studies})
    revised['research_program_source_ref'] = revised['research_program_source_refs'][0]
    first = prepare._value(revised['research_program_source_ref'])
    mapping = revised['source_outcomes'][0]['input_mapping']
    revised['research_program_contract'] = research_program_replay_inputs(_rebind_step(
        first['research_program'], {name: role for role, name in mapping.items()}))
    proof = save(tmp_path / 'actual_step_revision.json', {'status': 'completed',
        'baseline_method_ref': identity, 'data_role': 'exposed_development',
        'scope': 'one_paid_target_split_not_new_independent_physical_sources',
        'selected_method_payload_refs': revised['research_program_source_refs'],
        'paired_replay_refs': [study_ref]})
    revised['method_step_revision'] = {'baseline_method_ref': identity,
        'study_ref': study_ref, 'actual_step_revision_ref': proof}
    return revised


def test_actual_step_proof_preserves_source_identity_and_paired_replay_contract(tmp_path):
    _, identifier, _, updated = condition_revision(tmp_path)
    old = updated['candidates'][identifier]
    new = step_candidate(tmp_path / 'step', old)
    result = prepare._method_body_revision(identifier, old, new)
    assert result['paired_seed_count'] == 2
    assert new['source_ids'] == old['source_ids']
    assert new['independent_source_count'] == old['independent_source_count']
    assert not result['independent_transfer_support']


@pytest.mark.parametrize('change', ['parent', 'payload', 'incomplete', 'split', 'cap',
    'regression', 'different_edit', 'source_costs', 'contract', 'old_conditions'])
def test_unproved_step_changes_cannot_enter_the_frozen_successor(tmp_path, change):
    _, identifier, _, updated = condition_revision(tmp_path)
    old = updated['candidates'][identifier]
    new = step_candidate(tmp_path / 'step', old)
    ref = new['method_step_revision']['study_ref']
    evidence = prepare._value(ref)
    if change == 'parent':
        new['previous_version_ref'] = save(tmp_path / 'wrong.json', {**old, 'version': 1})
    elif change == 'payload':
        new['research_program_source_refs'] = new['research_program_source_refs'][:1]
    elif change == 'source_costs':
        new['source_outcomes'][0]['cost_refs'] = []
    elif change == 'contract':
        new['research_program_contract'] = {'operation': 'fabricated'}
    elif change == 'old_conditions':
        new['method_use_conditions'] = old['method_use_conditions']
    else:
        seed = evidence['seed_studies'][1]
        if change == 'incomplete': seed['methods'][0]['execution_complete'] = False
        elif change == 'split': seed['pairs'][0]['same_training_and_validation'] = False
        elif change == 'cap': seed['pairs'][0]['equal_method_compute_ceiling'] = 10
        elif change == 'regression': seed['pairs'][0]['validation_loss_change'] = .1
        else: seed['pairs'][0]['edit'] = {'operation': 'different'}
        save(Path(ref['file']), evidence)
    with pytest.raises(ValueError):
        prepare._method_body_revision(identifier, old, new)


def test_optional_D_reaches_existing_runner_from_common_initial_scene_only(tmp_path, monkeypatch):
    identifier, _, updated, _, keep = frozen_keep_fixture(tmp_path, monkeypatch)
    # The completed ABC C is the baseline of D, not its old-condition B.
    candidate = step_candidate(tmp_path / 'actual_step', updated['candidates'][identifier])
    prepared = prepare.prepare_method_body_intervention(tmp_path / 'prepared',
        save(tmp_path / 'revised_method.json', candidate))
    plan = prepare._value(prepared['plan_ref'])
    plan['runtime_snapshot'] = prepare._read(keep / 'protocol.json')['runtime_snapshot']
    seen = []
    def admit(actual, output, **kwargs):
        library = _read_library({'file': actual['research_organization']['pr3_library_path']}, tmp_path)
        assert list(library['candidates']) == list(updated['candidates'])
        assert library['candidates'][identifier] == candidate
        assert library['executions'] == updated['executions']
        assert kwargs['knowledge']['source_failures'] == ['retained']
        assert kwargs['target_intervention']['executed_arms'] == ['pr3']
        assert kwargs['target_intervention']['keep_costs_recharged'] is False
        assert actual['cases'] == prepare._read(keep / 'protocol.json')['cases']
        seen.append(True)
        return {'status': 'identity_admitted_science_blocked'}
    monkeypatch.setattr(study, '_run_study', admit)
    assert study.run_target_intervention(plan, tmp_path / 'not_run', keep_study=keep)[
        'status'] == 'identity_admitted_science_blocked'
    assert seen and not (tmp_path / 'not_run').exists()


def test_only_matching_question_cache_is_activated_and_old_refs_survive(tmp_path):
    question = {'id': 'old-task', 'meaning_version': 1}
    scope = save(tmp_path / 'scope.json', {'question': question, 'rows': ['paid']})
    match = save(tmp_path / 'score.json', {'scope_ref': scope, 'model': 'paid.py'})
    index = save(tmp_path / 'index.json', {'scopes': [scope], 'assessments': [match], 'contexts': []})
    selected = prepare._target_paid_score_index(index, question)
    assert selected['assessments'] == [match]
    assert selected['historical_score_index_ref'] == index
    assert prepare._target_paid_score_index(index, {'id': 'old-task'})['assessments'] == [match]
    # A future question has no possible score/prefix reuse; its unrelated huge
    # assessment payloads need never be read, but remain available by source ref.
    Path(match['file']).unlink()
    new = prepare._target_paid_score_index(index, {'id': 'fresh-task', 'meaning_version': 1})
    assert new['scopes'] == new['assessments'] == []
    assert new['historical_score_index_ref'] == index
