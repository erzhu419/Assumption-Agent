"""Recorded retained exports keep their actual source and explicit intake scope."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.research_episodes import load_common_sources, load_source_values
from hegel_machine.cognition.theory_consolidation import _revision_entries
from scripts.run_anonymous_revision_inquiry import RegisteredOrganizationAgent
from test_retained_revision_exports import exported_state


def saved_exported_source(path):
    agent = exported_state(path)
    # These hand-built exports advertise the same constructor and transfer
    # addresses that the ordinary transfer writer persists before exporting.
    # Save that declared fixture provenance as well as its state and training.
    pattern = Program('arg', name='value')
    unit = agent.questions[0]['theory_target']['output_unit']
    constructor = {'id': 'old_frozen_object', 'version': 1,
        'kind': 'program_context_relation', 'output_unit': unit,
        'definition': Definition('old_frozen_object', ('value',), pattern).to_dict(),
        'holes': [{'name': 'value', 'output_unit': unit, 'time_contract': []}],
        'fixed_role_signature': {}, 'learned_constraints': []}
    transfers = []
    for index, exported in enumerate(agent.questions[0]['revision_candidate_exports']):
        identifier = exported['theory_ref']['id']
        binding = agent.theory_index[identifier]['consolidation_ref']
        binding.update(library_path='shared/theory_library.json', candidate_version=1,
            components={'value': {'component_id': f'fixture_value_{index}', 'version': 1}})
        transfers.append({'theory_ref': deepcopy(exported['theory_ref']),
                          'binding': deepcopy(binding)})
    shared = agent.workspace / 'shared'
    shared.mkdir(exist_ok=True)
    (shared / 'theory_library.json').write_text(json.dumps({
        'candidates': {constructor['id']: constructor}, 'transfers': transfers}))
    state = {'questions': agent.questions, 'theory_index': agent.theory_index,
             'limits': {'confirmation_mse_threshold': 1e-10}}
    (agent.workspace / 'research_state.json').write_text(json.dumps(state))
    (agent.workspace / 'laboratory_observations.json').write_text(json.dumps(agent.observations))
    training = {'online_questions': agent.questions, 'workspace': str(agent.workspace),
        'paid_observations': agent.observations, 'events': agent.events,
        'limits': state['limits'], 'costs': agent.costs, 'status': 'method_stopped'}
    file = agent.workspace / 'training.json'
    file.write_text(json.dumps(training))
    return agent, {'file': str(file), 'json_path': []}


def test_registered_question_scope_admits_actual_retained_exports_only(tmp_path, monkeypatch):
    agent = exported_state(tmp_path)
    agent._organization_state = {'source_theory_scopes': {}}
    agent._organization_registration = {'consolidation': {
        'source_question_refs': [{'question_id': 'measured', 'meaning_version': 1}]}}
    original = deepcopy((agent.questions, agent.events, agent.theory_index, agent.costs))
    monkeypatch.setattr(Program, 'predict', lambda *a, **k: pytest.fail('metadata only'))
    assert RegisteredOrganizationAgent._registered_source_ids(agent) == ['revision_one', 'revision_two']
    agent._organization_registration['consolidation']['source_question_refs'][0]['meaning_version'] = 2
    assert RegisteredOrganizationAgent._registered_source_ids(agent) == []
    assert (agent.questions, agent.events, agent.theory_index, agent.costs) == original


def test_explicit_after_scope_retains_exact_parent_and_old_library_provenance(tmp_path, monkeypatch):
    agent, reference = saved_exported_source(tmp_path)
    original = deepcopy(agent.costs)
    monkeypatch.setattr(Program, 'predict', lambda *a, **k: pytest.fail('metadata only'))
    intake = load_common_sources([{**reference, 'source_theory_ids': ['revision_two']}], tmp_path)
    assert intake['status'] == 'completed'
    theory, = intake['theories']
    assert theory['id'].endswith('::revision_two')
    assert theory['consolidation_ref'] == agent.theory_index['revision_two']['consolidation_ref']
    assert theory['support']['status'] == 'candidate'
    assert load_source_values([theory['program_source_ref']], tmp_path)[0]['value'] == theory['program']
    episode, = [e for e in intake['episodes'] if theory['id'] in e['theory_ids']]
    pool = {theory['id']: theory}
    compiler = SimpleNamespace(workspace=tmp_path, questions=[], revisions=[], observations=[],
        events=[], theory_index=pool, config=SimpleNamespace(confirmation_mse_threshold=1e-10))
    # This older hand-built trace contains individual retentions but omits
    # the declared roster and terminal incumbent. It is not a complete step.
    assert all(step['status'] == 'UNKNOWN' for step in episode['source_revision_steps'])
    assert _revision_entries(compiler, [{**episode, 'theory_id': theory['id']}], pool, retained=[]) == []
    # Earlier Episode records still support their explicit pair and each
    # endpoint's own constructor; they are not relabeled as whole actions.
    earlier_episode = {key: value for key, value in episode.items() if key != 'source_revision_steps'}
    entries = _revision_entries(compiler, [{**earlier_episode, 'theory_id': theory['id']}], pool, retained=[])
    entry, = entries
    assert entry['before_program'] == agent.theory_index['revision_one']['program']
    assert entry['after_program'] == agent.theory_index['revision_two']['program']
    assert entry['source_id'] == 'toy_physical_source'
    assert entry['source_comparison']['before_evaluation_ref']['file'] == reference['file']
    assert entry['source_comparison']['after_evaluation_ref']['file'] == reference['file']
    assert not entry['source_comparison']['scope']['independent_confirmation']
    for side, index in [('before', 0), ('after', 1)]:
        contract = entry[f'_{side}_contract_source']
        provenance = contract['source_ref']
        assert provenance['program_ref'] == entry[f'{side}_ref']
        assert provenance['pattern_ref'] == {
            'file': str(agent.workspace / 'shared/theory_library.json'),
            'json_path': ['candidates', 'old_frozen_object', 'definition', 'body']}
        assert provenance['transfer_ref']['json_path'] == ['transfers', index]
        assert contract['_holes'] == [{'name': 'value', 'output_unit': 'V', 'time_contract': []}]
        assert contract['_bound_holes']['value']['component_id'] == f'fixture_value_{index}'
    empty = load_common_sources([{**reference, 'source_theory_ids': []}], tmp_path)
    assert not empty['theories'] and not empty['theory_ids']
    assert any(c.get('before_ref') for e in empty['episodes'] for c in e['actual_changes'])
    assert agent.costs == original


def test_same_physical_source_recovery_copies_do_not_add_independent_identity(tmp_path, monkeypatch):
    first, first_ref = saved_exported_source(tmp_path / 'first')
    second, second_ref = saved_exported_source(tmp_path / 'recovered')
    monkeypatch.setattr(Program, 'predict', lambda *a, **k: pytest.fail('metadata only'))
    intake = load_common_sources([{**ref, 'source_theory_ids': ['revision_two']}
                                  for ref in (first_ref, second_ref)], tmp_path)
    assert len(intake['theories']) == len(set(intake['theory_ids'])) == 2
    episodes = [e for e in intake['episodes'] if e['theory_ids']]
    assert {e['source_id'] for e in episodes} == {'toy_physical_source'}
    assert all(e['failure_refs'] for e in episodes)
    assert all(t['consolidation_ref']['candidate_id'] == 'old_frozen_object'
               for t in intake['theories'])
    assert first.costs == second.costs == {'compute_cost': 6}


@pytest.mark.parametrize('existing_family,declared_family,expected', [
    (None, 'declared_public_coordinates', 'declared_public_coordinates'),
    ('different_explicit_topic', 'declared_public_coordinates', 'different_explicit_topic'),
    (None, None, None),
])
def test_retained_export_intake_uses_only_its_original_declared_family(
        tmp_path, monkeypatch, existing_family, declared_family, expected):
    agent, reference = saved_exported_source(tmp_path)
    if declared_family is not None:
        agent.questions[0]['theory_target']['context_family_ref'] = declared_family
    if existing_family is not None:
        agent.theory_index['revision_two']['context_family_ref'] = existing_family
    state = {'questions': agent.questions, 'theory_index': agent.theory_index,
             'limits': {'confirmation_mse_threshold': 1e-10}}
    source_file = agent.workspace / 'research_state.json'
    source_file.write_text(json.dumps(state))
    (agent.workspace / 'laboratory_observations.json').write_text(json.dumps(agent.observations))
    original = deepcopy((agent.questions, agent.theory_index, agent.costs))
    monkeypatch.setattr(Program, 'predict', lambda *a, **k: pytest.fail('metadata only'))
    intake = load_common_sources([{**reference, 'source_theory_ids': ['revision_two']}], tmp_path)
    theory, = intake['theories']
    assert theory.get('context_family_ref') == expected
    assert ('context_family_ref' in theory) is (expected is not None)
    assert theory['context_ref']['source_id'] == 'toy_physical_source'
    episode, = [e for e in intake['episodes'] if theory['id'] in e['theory_ids']]
    assert episode['source_id'] == 'toy_physical_source'
    assert theory['consolidation_ref'] == agent.theory_index['revision_two']['consolidation_ref']
    assert (agent.questions, agent.theory_index, agent.costs) == original
    assert json.loads(source_file.read_text()) == state


def test_real_recall_retains_the_explicit_target_family_without_inheriting_support(tmp_path):
    from test_frozen_theory_recall import frozen_scene
    from test_question_routing import future_question, probe, route
    from test_theory_consolidation import MISSION, assert_no_new_science

    agent, lab = frozen_scene(tmp_path)
    question = future_question(agent)
    family = 'declared_drive_theme'
    question['theory_target']['context_family_ref'] = family
    context = deepcopy(question['theory_target']['context_ref'])
    result = route(agent, lab, question=question, experiments=[probe(agent)], mission=deepcopy(MISSION))
    identifier, = result['frozen_recall']['generated_theory_ids']
    theory = agent.theory_index[identifier]
    assert theory['context_family_ref'] == family
    assert theory['context_ref'] == context
    assert theory['support']['status'] == 'candidate' and not theory['support']['evidence']
    assert not result['frozen_recall']['support_updated']
    spent = deepcopy(agent.costs)
    repeated = route(agent, lab, question={'id': question['id']})
    assert repeated['cache_hit'] and repeated['new_compute_cost'] == 0
    assert agent.costs == spent and agent.theory_index[identifier]['version'] == theory['version']
    assert_no_new_science(agent, lab)
