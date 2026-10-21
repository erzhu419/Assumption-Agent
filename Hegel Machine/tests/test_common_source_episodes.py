"""Actual Common source references keep native identities and executed failures."""
from copy import deepcopy
import json

from hegel_machine.cognition.anonymous_method_learning import compact_workspace_result
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.question_routing import register_theories, resolved_theories
from hegel_machine.cognition.research_episodes import (
    extract_episodes, load_common_sources, load_source_values,
)
from hegel_machine.cognition.theory_induction import consolidate_theories
from hegel_machine.cognition.theory_consolidation import transfer_theory, _save as save_library
from scripts.run_online_inquiry_llm import write_json
from test_native_condition_inquiry import PaidConditionalLab, buy_fixture, execute, prepared
from test_native_theory_registration import TARGET


def source(path, *, physical_id='fixture-apparatus-A', offset=1., conflicting_id=None):
    agent, lab = prepared(path), PaidConditionalLab()
    agent.provider = None
    buy_fixture(agent, lab)
    context = {'domain': 'local-fixture', **({'source_id': physical_id} if physical_id else {})}
    target = {**deepcopy(TARGET), 'context_ref': {**context,
        **({'source_id': conflicting_id} if conflicting_id else {})},
        'context_family_ref': 'preregistered-fixture-theme'}
    program = Program('add', children=(Program('feature', name='drive'), Program('const', value=offset)))
    feedback = execute(agent, lab, 'run', {'operator': 'native_condition_research',
        'research_question': {'id': 'same-q', 'question': 'Compare fixture expressions',
            'parent_id': 'task', 'depends_on': [], 'context_ref': context, 'theory_target': target},
        # One paid-archive scan plus one existing candidate-row score.
        'programs': [program.to_dict()], 'max_compute': 160,
        'matched_pairs': [{'reference_record_index': 0, 'record_index': 1}]})
    assert 'error' not in feedback
    assert feedback['native_condition_research']['status'] == 'search_incomplete'
    identifier = feedback['native_theory_registration']['registered'][0]['theory_id']
    # Retain an actual, nonzero saved fit outcome as the author's negative
    # reference; this does not establish a scientific family refutation.
    agent.theory_index[identifier]['counterexample_refs'] = [{'file': 'research_state.json',
        'json_path': ['questions', 1, 'native_condition_research', 0, 'candidates', 0, 'fit_mse']}]
    agent._prepare_program_state()
    result = compact_workspace_result(agent, agent.result('researching'))
    write_json(path / 'training.json', result)
    return agent, identifier, {'file': str(path / 'training.json'), 'json_path': []}


def scientific_entries(intake):
    pool = {theory['id']: theory for theory in intake['theories']}
    return [{**episode, **{key: pool[identifier][key] for key in (
                'program', 'role_signature', 'output_unit', 'context_ref', 'context_family_ref')},
             'source_ref': pool[identifier]['program_source_ref']}
            for episode in intake['episodes'] for identifier in episode['theory_ids']]


def test_distinct_common_sources_with_same_native_id_reconstruct_without_overwriting(tmp_path):
    first, identifier, first_ref = source(tmp_path / 'first')
    second, same_identifier, second_ref = source(tmp_path / 'second', physical_id='fixture-apparatus-B', offset=2.)
    assert identifier == same_identifier
    old_costs = deepcopy([first.costs, second.costs])
    intake = load_common_sources([first_ref, second_ref], tmp_path)
    assert intake['status'] == 'completed'
    assert len(intake['theories']) == len(set(intake['theory_ids'])) == 2
    assert {theory['source_snapshot']['source_ref']['question_id'] for theory in intake['theories']} == {'same-q'}
    entries = scientific_entries(intake)
    assert {entry['source_id'] for entry in entries} == {'fixture-apparatus-A', 'fixture-apparatus-B'}
    assert len({entry['episode_id'] for entry in entries}) == 2
    for theory, entry in zip(intake['theories'], entries):
        assert load_source_values([theory['program_source_ref']], tmp_path)[0]['value'] == theory['program']
        assert entry['failure_refs']
        assert entry['cost_scope'] == 'whole_Common_source_account_not_additive_per_question'
        account, = load_source_values(entry['cost_refs'], tmp_path)
        assert account['value']['compute_cost'] == first.costs['compute_cost']
        assert load_source_values([entry['source_outcome_ref']], tmp_path)[0]['value'] == 'researching'
        for reference in theory['counterexample_refs']:
            assert load_source_values([reference], tmp_path)[0]['value'] > 0
        assert entry['change_evidence'] and all(ref['file'].startswith(str(tmp_path)) for ref in entry['change_evidence'])
    result = consolidate_theories(entries, allocation={'compute': 2000, 'max_groups': 4})
    assert result['candidates']
    candidate = result['candidates'][0]
    assert candidate['independent_source_count'] == 2
    assert all(check['exact'] for check in candidate['source_reconstruction'])
    assert candidate['global_support'] is False
    assert [first.costs, second.costs] == old_costs
    assert all('rows' not in episode and 'program' not in episode for episode in intake['episodes'])


def test_same_physical_source_across_arms_or_recovery_does_not_gain_independence(tmp_path):
    _, _, first_ref = source(tmp_path / 'arm-C')
    _, _, copied_ref = source(tmp_path / 'same-case-other-arm', offset=2.)
    copied_ref['file'] = str(tmp_path / 'same-case-other-arm' / 'research_state.json')
    intake = load_common_sources([first_ref, copied_ref, first_ref], tmp_path)
    assert len(intake['theories']) == 2
    entries = scientific_entries(intake)
    assert len({entry['source_id'] for entry in entries}) == 1
    result = consolidate_theories(entries, allocation={'compute': 2000, 'max_groups': 4})
    assert result['candidates'][0]['independent_source_count'] == 1
    assert result['candidates'][0]['active_library'] is False


def test_explicit_empty_source_scope_keeps_executed_failures_and_no_theories(tmp_path):
    agent, identifier, reference = source(tmp_path / 'source')
    reference['source_theory_ids'] = []
    intake = load_common_sources([reference], tmp_path)
    assert intake['status'] == 'completed'
    assert not intake['theories'] and not intake['theory_ids']
    retained = next(episode for episode in intake['episodes'] if episode['episode_id'].endswith('same-q@meaning1'))
    assert retained['failure_refs'] and retained['program_refs'] and not retained['theory_ids']
    original = extract_episodes(agent, excluded_theory_ids=[identifier])
    episode = next(item for item in original['episodes'] if item['episode_id'] == 'same-q@meaning1')
    assert not episode['theory_ids'] and episode['failure_refs']


def test_missing_or_conflicting_original_physical_identity_remains_unknown(tmp_path):
    _, _, missing_ref = source(tmp_path / 'missing', physical_id=None)
    _, _, conflict_ref = source(tmp_path / 'conflict', conflicting_id='different-declared-apparatus')
    intake = load_common_sources([missing_ref, conflict_ref], tmp_path)
    assert intake['status'] == 'UNKNOWN' and not intake['theories']
    assert {request['reason'] for request in intake['author_work']} == {
        'source_identity_required', 'conflicting_source_identity'}
    assert any(episode['failure_refs'] for episode in intake['episodes'])
    saved = json.loads((tmp_path / 'conflict' / 'research_state.json').read_text())
    assert saved['questions'][1]['context_ref']['source_id'] == 'fixture-apparatus-A'
    assert saved['theory_index'][next(iter(saved['theory_index']))]['context_ref']['source_id'] == 'different-declared-apparatus'


def test_executed_source_counterexample_stays_rejected_after_import_and_resolution(tmp_path):
    first, identifier, first_ref = source(tmp_path / 'first')
    _, _, second_ref = source(tmp_path / 'second', physical_id='fixture-apparatus-B', offset=2.)
    initial = load_common_sources([first_ref, second_ref], tmp_path)
    candidate = consolidate_theories(scientific_entries(initial), allocation={
        'compute': 2000, 'max_groups': 4})['candidates'][0]
    candidate['version'] = 1
    first.theory_consolidation = {'candidates': {candidate['id']: candidate}, 'transfers': []}
    save_library(first)
    declaration = first.theory_index[identifier]
    paid = first.observations[0]['rows'][0]
    failed = transfer_theory(first, {'candidate_id': candidate['id'], 'theory_id': 'actual_failed_binding',
        'mapping': {'role_0': 'drive'}, 'parameters': [1.],
        'role_signature': declaration['role_signature'], 'output_unit': declaration['output_unit'],
        'context_ref': declaration['context_ref'], 'evidence_checks': [{
            'source_ref': {'file': str(first.workspace / 'laboratory_observations.json'),
                           'json_path': [0, 'rows', 0]},
            'context_ref': declaration['context_ref'], 'observation': paid, 'tolerance': 0.}]})
    assert failed['status'] == 'refuted'
    first._prepare_program_state()
    write_json(first.workspace / 'training.json', compact_workspace_result(first, first.result('researching')))
    intake = load_common_sources([{**first_ref, 'source_theory_ids': [identifier]}], tmp_path)
    snapshot, = intake['theories']
    executed = [ref for ref in snapshot['counterexample_refs'] if ref.get('json_path', [])[:1] == ['transfers']]
    assert executed and load_source_values(executed, tmp_path)[0]['value']['status'] == 'failed'
    recipient = prepared(tmp_path / 'recipient')
    register_theories(recipient, intake['theories'])
    imported, = resolved_theories(recipient)
    # The original index protocol's compatible=False survives recomputation;
    # no derived scope field or new label is needed to reject this binding.
    assert imported['compatible'] is False
    episode = next(item for item in intake['episodes'] if snapshot['id'] in item['theory_ids'])
    assert all(ref in episode['failure_refs'] for ref in executed)
    assert load_source_values(episode['cost_refs'], tmp_path)[0]['value'] == first.costs
