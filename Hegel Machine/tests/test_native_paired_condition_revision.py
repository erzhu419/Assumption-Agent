"""Expose completed validation returns only in a new, unvalidated condition study."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

from hegel_machine.cognition import method_conditions
from hegel_machine.cognition.method_feedback import (
    load_paired_method_returns, revise_paired_method_conditions,
)
from test_frozen_paired_condition_validation import BASELINE, METHOD, frozen_study


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {'file': str(path)}


def read(reference):
    value = json.loads(Path(reference['file']).read_text())
    for key in reference.get('json_path', []):
        value = value[key]
    return value


def fixture(tmp_path, *, failed=True):
    development, validation, frozen = frozen_study()
    if failed:
        # The actual frozen condition says baseline here, but the challenger
        # completes the same task more cheaply. This is a condition counterexample.
        validation[-1]['challenger']['costs']['compute_cost'] = 50
    artifacts = {}
    for side, identity in [('baseline', BASELINE), ('challenger', METHOD)]:
        artifacts[side] = save(tmp_path / 'paid' / (side + '_method.json'),
            {**identity, 'execution': {'entry': side + '_actual_program'}})
    frozen.update(baseline_method_artifact_ref=artifacts['baseline'],
        method_artifact_ref=artifacts['challenger'])
    frozen_ref = save(tmp_path / 'paid/frozen_conditions.json', [frozen])
    references = []
    for index, pair in enumerate([*development, *validation]):
        raw = {key: deepcopy(value) for key, value in pair.items()
            if key not in ('baseline', 'challenger', 'feedback_ref',
                           'frozen_conditions', 'condition_prediction')}
        for side, identity in [('baseline', BASELINE), ('challenger', METHOD)]:
            outcome = deepcopy(pair[side])
            outcome.update(outcome.pop('basis'))
            outcome['profile_ref'] = save(tmp_path / 'paid' / f'profile_{index}_{side}.json',
                outcome.pop('profile'))
            outcome.update(method_ref=deepcopy(identity), method_artifact_ref=artifacts[side])
            raw[side + '_ref'] = save(tmp_path / 'paid' / f'outcome_{index}_{side}.json', outcome)
        if pair['data_role'] == 'condition_validation':
            prediction = deepcopy(pair['condition_prediction'])
            prediction['frozen_conditions_ref'] = frozen_ref
            raw['frozen_conditions_ref'] = frozen_ref
            raw['condition_prediction_ref'] = save(tmp_path / 'paid' / f'prediction_{index}.json', prediction)
        references.append(save(tmp_path / 'paid' / f'pair_{index}.json', raw))
    agent = SimpleNamespace(workspace=tmp_path / 'research', costs={'compute_cost': 0})
    agent.workspace.mkdir()
    return agent, references, frozen


def test_failed_validation_becomes_five_exposed_development_uses_without_rewriting_paid_data(tmp_path):
    agent, references, frozen = fixture(tmp_path)
    original = {path: path.read_bytes() for path in (tmp_path / 'paid').rglob('*.json')}
    result = revise_paired_method_conditions(agent, METHOD, references, 10000)
    assert result['status'] == 'conflicting_public_view'
    assert result['original_pair_refs'] == references
    assert len(result['development_pair_refs']) == result['research']['paired_invocations'] == 5
    assert [item['previous_data_role'] for item in result['source_roles']] == [
        'development', 'development', 'development', 'condition_validation', 'condition_validation']
    assert result['counterexample_refs'] == references[-1:]
    assert result['research']['condition'] is None
    assert not result['research']['general_guard_supported']
    assert not result['research']['independent_support']
    for key in ('method_ref', 'baseline_method_ref', 'method_artifact_ref', 'baseline_method_artifact_ref'):
        assert result['research'][key] == frozen[key]
    assert result['new_fits'] == result['new_predictions'] == result['new_observations'] == 0
    assert agent.costs['compute_cost'] == result['new_compute_cost'] > 0
    for old_ref, new_ref in zip(references, result['development_pair_refs']):
        before, after = read(old_ref), read(new_ref)
        assert after['data_role'] == 'development'
        assert not any(key in after for key in (
            'condition_validation_cohort_refs', 'frozen_conditions_ref', 'condition_prediction_ref'))
        for key in ('origin_ref', 'physical_source_ref', 'baseline_ref', 'challenger_ref',
                    'baseline_action_content', 'challenger_action_content'):
            assert before[key] == after[key]
        if before['data_role'] == 'condition_validation':
            assert after['original_pair_ref'] == old_ref
    assert all(path.read_bytes() == value for path, value in original.items())
    exposed = load_paired_method_returns(agent, result['development_pair_refs'])
    old = load_paired_method_returns(agent, references)
    assert all(new[side]['costs'] == prior[side]['costs'] for new, prior in zip(exposed, old)
        for side in ('baseline', 'challenger'))
    assert read(result['research_ref']) == result['research']


def test_completed_same_basis_reuses_condition_result_without_new_learning(tmp_path, monkeypatch):
    agent, references, _ = fixture(tmp_path)
    result = revise_paired_method_conditions(agent, METHOD, references, 10000)
    costs = dict(agent.costs)
    files = set(agent.workspace.rglob('*.json'))
    def unexpected(*args, **kwargs):
        raise AssertionError('The completed paid condition study was repeated')
    monkeypatch.setattr(method_conditions, 'research_paired_method_conditions', unexpected)
    cached = revise_paired_method_conditions(agent, METHOD, [*references, references[-1]], 10000,
        previous_revision_ref=result['revision_ref'])
    assert cached['cache_hit'] and cached['new_compute_cost'] == 0
    assert cached['revision_ref'] == result['revision_ref']
    assert cached['development_pair_refs'] == result['development_pair_refs']
    assert agent.costs == costs and set(agent.workspace.rglob('*.json')) == files


def test_underfunded_condition_search_reuses_completed_projection_when_resumed(tmp_path):
    agent, references, _ = fixture(tmp_path)
    partial = revise_paired_method_conditions(agent, METHOD, references, 6)
    assert partial['budget_incomplete'] and partial['new_compute_cost'] == 6
    assert len(partial['development_pair_refs']) == 5
    denied = revise_paired_method_conditions(agent, METHOD, references, 6,
        previous_revision_ref=partial['revision_ref'])
    assert denied['cache_hit'] and denied['new_compute_cost'] == 0
    completed = revise_paired_method_conditions(agent, METHOD, references, 10000,
        previous_revision_ref=partial['revision_ref'])
    assert not completed['budget_incomplete'] and completed['projection_reused']
    assert completed['version'] == partial['version']
    assert completed['development_pair_refs'] == partial['development_pair_refs']
    assert completed['total_new_compute_cost'] == 6 + completed['new_compute_cost']
    assert agent.costs['compute_cost'] == completed['total_new_compute_cost']
    assert len(list(agent.workspace.glob('method_feedback/paired_condition_revision_*/pair_*.json'))) == 2


def test_unfinished_registered_cohort_does_not_change_evidence_role(tmp_path):
    agent, references, _ = fixture(tmp_path)
    result = revise_paired_method_conditions(agent, METHOD, references[:-1], 10000)
    assert result['status'] == 'awaiting_registered_condition_validation'
    assert result['revision_ref'] is None and result['new_compute_cost'] == 0
    assert not list(agent.workspace.rglob('*.json'))
    assert read(references[-2])['data_role'] == 'condition_validation'


def test_validation_without_a_condition_counterexample_does_not_start_revision(tmp_path):
    agent, references, _ = fixture(tmp_path, failed=False)
    result = revise_paired_method_conditions(agent, METHOD, references, 10000)
    assert result['status'] == 'frozen_condition_has_unrefuted_candidate'
    assert result['revision_ref'] is None and result['new_compute_cost'] == 0
    assert not list(agent.workspace.rglob('*.json'))


def test_later_same_origin_outcome_replaces_its_old_use_and_invalidates_completed_cache(tmp_path):
    agent, references, _ = fixture(tmp_path)
    first = revise_paired_method_conditions(agent, METHOD, references, 10000)
    replacement = read(references[-1])
    outcome = read(replacement['challenger_ref'])
    outcome['costs']['compute_cost'] = 25
    replacement['challenger_ref'] = save(tmp_path / 'paid/later_outcome.json', outcome)
    later = save(tmp_path / 'paid/later_pair.json', replacement)
    result = revise_paired_method_conditions(agent, METHOD, [*references, later], 10000,
        previous_revision_ref=first['revision_ref'])
    assert not result['cache_hit'] and result['version'] == first['version'] + 1
    assert result['research']['paired_invocations'] == 5
    assert result['original_pair_refs'] == [*references[:-1], later]
    assert result['counterexample_refs'] == [later]
