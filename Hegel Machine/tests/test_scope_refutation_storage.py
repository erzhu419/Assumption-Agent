"""Repeated complete refutations use the library's existing field references."""

from copy import deepcopy
import json
from types import SimpleNamespace

from hegel_machine.cognition.checkpoint_storage import _json_values_equal
from hegel_machine.cognition.research_episodes import load_source_values
from hegel_machine.cognition import theory_consolidation as library


def scene():
    scope = {'status': 'refuted', 'counterexample_refs': [
        {'file': '/retained/source/research_state.json', 'json_path': ['operator_results', 2]},
        {'archive': '/retained/history', 'unit': 'paid', 'member': 'records.json',
         'field': 'observations[3]'}],
        'program': {'op': 'add', 'children': [{'op': 'feature', 'name': 'u'},
                                             {'op': 'const', 'value': 0.2}]}}
    first = {'theory_id': 'first', 'scope_refutation': scope, 'status': 'refuted', 'cost': 11}
    second = {'theory_id': 'alias', 'scope_refutation': deepcopy(scope), 'status': 'refuted', 'cost': 13}
    unique = deepcopy(second)
    unique['theory_id'] = 'different_evidence'
    unique['scope_refutation']['counterexample_refs'].append(
        {'file': '/retained/source/research_state.json', 'json_path': ['operator_results', 4]})
    return {'candidates': {'old': {'id': 'old', 'version': 1}},
            'transfers': [first, second, unique]}


def save(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))


def test_only_complete_equal_refutations_share_first_value_and_preserve_order(tmp_path):
    original = scene()
    before = deepcopy(original)
    compact = library._compact_library_fields(original)
    assert compact['transfers'][0]['scope_refutation'] == original['transfers'][0]['scope_refutation']
    assert compact['transfers'][1]['scope_refutation_ref'] == {
        'json_path': ['transfers', 0, 'scope_refutation']}
    assert 'scope_refutation' not in compact['transfers'][1]
    assert compact['transfers'][2]['scope_refutation'] == original['transfers'][2]['scope_refutation']
    assert 'scope_refutation_ref' not in compact['transfers'][2]
    assert original == before
    path = tmp_path / 'library.json'
    save(path, compact)
    restored, = load_source_values([{'file': str(path), 'json_path': []}], tmp_path)
    assert _json_values_equal(restored['value'], original)
    assert library._compact_library_fields(compact) == compact
    restored['value']['transfers'][1]['scope_refutation']['counterexample_refs'].append({'new': True})
    assert restored['value']['transfers'][0]['scope_refutation'] == original['transfers'][0]['scope_refutation']


def test_json_path_and_field_addresses_still_resolve_duplicate_counterexamples(tmp_path):
    original = scene()
    path = tmp_path / 'library.json'
    save(path, library._compact_library_fields(original))
    direct, legacy = load_source_values([
        {'file': str(path), 'json_path': ['transfers', 1, 'scope_refutation', 'counterexample_refs']},
        {'file': str(path), 'field': 'transfers[1].scope_refutation.counterexample_refs'}], tmp_path)
    expected = original['transfers'][1]['scope_refutation']['counterexample_refs']
    assert direct['value'] == legacy['value'] == expected
    # A reference to a whole transfer expands its new field too.
    whole, = load_source_values([{'file': str(path), 'field': 'transfers[1]'}], tmp_path)
    assert _json_values_equal(whole['value'], original['transfers'][1])


def test_installed_library_and_inherited_entry_keep_original_counterexample_addresses(tmp_path):
    original = scene()
    frozen = tmp_path / 'source/shared/theory_library.json'
    save(frozen, library._compact_library_fields(original))
    agent = SimpleNamespace(workspace=tmp_path / 'target')
    agent.theory_consolidation = library.install_theory_library(agent,
        {'file': str(frozen), 'json_path': []})
    assert _json_values_equal(agent.theory_consolidation, original)
    reference = library.library_entry_reference(agent, 'transfers', 1, 'scope_refutation')
    addressed, = load_source_values([reference], agent.workspace)
    assert addressed['value'] == original['transfers'][1]['scope_refutation']
    agent.theory_consolidation['transfers'][1]['scope_refutation']['counterexample_refs'].append({'local': True})
    assert agent.theory_consolidation['transfers'][0] == original['transfers'][0]
    assert agent._theory_library_base['transfers'] == original['transfers']


def test_equal_values_with_different_nested_field_order_are_not_collapsed(tmp_path):
    original = scene()
    scope = original['transfers'][1]['scope_refutation']
    scope['program'] = dict(reversed(list(scope['program'].items())))
    compact = library._compact_library_fields(original)
    assert 'scope_refutation' in compact['transfers'][1]
    assert 'scope_refutation_ref' not in compact['transfers'][1]
