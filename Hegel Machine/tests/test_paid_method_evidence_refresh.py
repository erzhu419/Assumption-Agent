"""Paid pre-call diagnostics refresh development evidence, not old validation."""

from copy import deepcopy
from pathlib import Path

from hegel_machine.cognition import method_conditions
from hegel_machine.cognition.models import SearchBudget
from scripts import return_partition_pair as entry
from scripts.run_partial_role_parent import save
from test_native_condition_revision_routing import (
    completed_revision, freeze_returns, measured, read, route,
)


def seven_source_scene(tmp_path):
    prior = tmp_path / 'prior'
    agent, _, _, _, _, _, first = completed_revision(prior)
    validation = [measured(agent, prior, name, count, False)
        for name, count in (('next0', 15), ('next1', 28))]
    cohort = [read(reference)['physical_source_ref'] for reference in validation]
    freeze_returns(prior / 'second_frozen', read(first['research_ref']), validation, cohort=cohort)
    revised = route(agent)['method_action_research']['native_condition_revisions'][0]
    assert revised['version'] == 2 and not revised['research']['candidates']
    references = revised['development_pair_refs']
    save(prior / 'common/questions.json', agent.questions)
    # The actual merged-return receipt has its own balance and no cumulative
    # compute field. A separate allocation must not assume the old field.
    save(prior / 'receipt.json', {'remaining_compute': 2010, 'compute_cost': 3990})
    allocation_ref = {'file': str(tmp_path / 'allocation.json')}
    save(Path(allocation_ref['file']), {'allocation_id': 'paid_profile_refresh_control',
        'maximum_compute': 100000, 'new_observation_limit': 0})
    for reference in references:
        pair = read(reference)
        source = Path(pair['origin_ref']['file']).parent
        save(source / 'registration.json', {})
        save(source / 'common_comparison.json', {'data_refs': [], 'scores': []})
        previous_receipt = read({'file': str(source / 'receipt.json')})
        save(source / 'receipt.json', {**previous_receipt, 'compute_cost': 100})
    originals = {path: path.read_bytes() for path in prior.rglob('*.json')}
    return prior, references, allocation_ref, originals


def install_paid_diagnostic(monkeypatch, *, unfunded=False):
    calls = []

    def ensure(source, cfg, compared, output, budget):
        calls.append(str(source))
        if unfunded:
            assert not budget.spend(budget.remaining + 1)
            return {'status': 'complete_not_funded', 'evidence_ref': None,
                'new_compute_cost': 0, 'new_predictions': 0}
        assert budget.spend(3)
        reference = {'file': str(Path(output) / 'pre_call_evidence.json')}
        # A controlled public diagnostic. The production learner still has
        # to consume it; no condition formula or preferred method is supplied.
        save(Path(reference['file']), {'new_distinction': Path(source).name in {'d0', 'd1'}})
        return {'status': 'completed', 'evidence_ref': reference,
            'new_compute_cost': 3, 'new_predictions': 2}

    def profile(source, cfg, compared):
        result = read({'file': str(Path(source) / 'profile.json')})
        evidence = read(compared['pre_call_evidence_ref'])
        result.setdefault('research', {}).setdefault('evidence', {})['new_distinction'] = evidence['new_distinction']
        return result

    monkeypatch.setattr(entry, 'ensure_pre_call_evidence', ensure)
    monkeypatch.setattr(entry, 'paid_pre_call_profile', profile)
    return calls


def test_paid_refresh_reaches_common_as_seven_updated_sources_and_keeps_originals(tmp_path, monkeypatch):
    prior, references, allocation, originals = seven_source_scene(tmp_path)
    calls = install_paid_diagnostic(monkeypatch)
    learn = method_conditions.research_method_conflict
    learning_rows = []

    def observed(*args, **kwargs):
        measured_returns, _ = kwargs['paired_returns']
        learning_rows.append(deepcopy(measured_returns))
        return learn(*args, **kwargs)

    monkeypatch.setattr(method_conditions, 'research_method_conflict', observed)
    result = entry.refresh_pair_profiles(prior, tmp_path / 'updated', references, allocation_ref=allocation)
    group, = result['return']['research_result']['method_action_research']['paired_condition_research']
    assert len(calls) == 7 and len(learning_rows) == 1
    assert len(learning_rows[0]) == group['paired_invocations'] == 7
    assert group['candidates'] and not group['general_guard_supported']
    assert not group.get('reused')
    assert not group.get('group_validation_refs') and not group.get('condition_validation_cohort_refs')
    assert result['new_invocations'] == result['new_fits'] == result['new_observations'] == 0
    assert result['new_predictions'] == 14
    assert result['compute_cost'] >= 21 and result['remaining_compute'] == 100000 - result['compute_cost']
    for old_ref, new_ref in zip(references, result['updated_pair_refs']):
        old, updated = read(old_ref), read(new_ref)
        assert updated['origin_ref'] == old['origin_ref']
        assert updated['physical_source_ref'] == old['physical_source_ref']
        assert updated['data_role'] == 'development'
        assert updated['profile_recovery_predecessor_ref'] == old_ref
        for side in ('baseline', 'challenger'):
            before, after = read(old[side + '_ref']), read(updated[side + '_ref'])
            assert {key: value for key, value in before.items() if key != 'profile_ref'} == {
                key: value for key, value in after.items() if key != 'profile_ref'}
    assert all(path.read_bytes() == content for path, content in originals.items())


def test_completed_refresh_reuses_receipt_without_readout_or_condition_learning(tmp_path, monkeypatch):
    prior, references, allocation, _ = seven_source_scene(tmp_path)
    install_paid_diagnostic(monkeypatch)
    output = tmp_path / 'updated'
    first = entry.refresh_pair_profiles(prior, output, references, allocation_ref=allocation)

    def prohibited(*args, **kwargs):
        raise AssertionError('A completed refresh must reuse its paid result')

    monkeypatch.setattr(entry, 'ensure_pre_call_evidence', prohibited)
    monkeypatch.setattr(method_conditions, 'research_method_conflict', prohibited)
    again = entry.refresh_pair_profiles(prior, output, references, allocation_ref=allocation)
    assert again['reused'] and again['new_compute_cost'] == 0
    assert again['updated_pair_refs'] == first['updated_pair_refs']
    assert again['remaining_compute'] == first['remaining_compute']


def test_unfunded_paid_diagnostic_does_not_become_completed_condition_evidence(tmp_path, monkeypatch):
    prior, references, allocation, originals = seven_source_scene(tmp_path)
    calls = install_paid_diagnostic(monkeypatch, unfunded=True)

    def prohibited(*args, **kwargs):
        raise AssertionError('An unfinished diagnostic cannot supply new condition evidence')

    monkeypatch.setattr(method_conditions, 'research_method_conflict', prohibited)
    result = entry.refresh_pair_profiles(prior, tmp_path / 'unfunded', references, allocation_ref=allocation)
    assert result['status'] == 'complete_not_funded'
    assert len(calls) == 1
    assert not result.get('updated_pair_refs')
    assert result['new_predictions'] == result['new_observations'] == result['new_fits'] == 0
    assert all(path.read_bytes() == content for path, content in originals.items())


def test_new_revision_directory_reuses_original_paid_columns_without_predictions(tmp_path, monkeypatch):
    from test_pre_call_method_evidence import scene, prohibit_prediction

    prior, references, allocation, _ = seven_source_scene(tmp_path)
    evidence_refs, paid_files = [], {}
    for index, reference in enumerate(references):
        pair = read(reference)
        origin = Path(pair['origin_ref']['file']).parent
        material, cfg, compared, rows, _ = scene(tmp_path / 'physical' / str(index))
        save(origin / 'registration.json', cfg)
        save(origin / 'common_comparison.json', compared)
        save(origin / 'prepared_state.json', read({'file': str(material / 'prepared_state.json')}))
        paid = entry.ensure_pre_call_evidence(origin, cfg, compared,
            tmp_path / 'paid' / str(index), SearchBudget(len(rows)))
        evidence_refs.append(paid['evidence_ref'])
        location = Path(paid['evidence_ref']['file'])
        paid_files[location] = location.read_bytes()
        for side in ('baseline', 'challenger'):
            outcome = read(pair[side + '_ref'])
            profile = read(outcome['profile_ref'])
            profile['recovery_basis'] = {'pre_call_evidence_ref': paid['evidence_ref']}
            save(Path(outcome['profile_ref']['file']), profile)
    # A new output directory must use the predecessor view's paid reference;
    # the physical origin's common comparison deliberately has no such link.
    prohibit_prediction(monkeypatch)
    result = entry.refresh_pair_profiles(prior, tmp_path / 'next_revision',
        references, allocation_ref=allocation)
    assert result['readout_compute_cost'] == result['new_predictions'] == 0
    assert [item['evidence_ref'] for item in result['pre_call_evidence']] == evidence_refs
    assert all(item['status'] == 'pre_call_evidence_reused' for item in result['pre_call_evidence'])
    assert all(location.read_bytes() == value for location, value in paid_files.items())
    assert not list((tmp_path / 'next_revision').glob('pair_*/pre_call/pre_call_evidence.json'))
    group, = result['return']['research_result']['method_action_research']['paired_condition_research']
    assert group['paired_invocations'] == 7
