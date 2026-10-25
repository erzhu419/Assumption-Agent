"""Frozen index anchoring itself owns the independent nested snapshot."""

from copy import deepcopy

from hegel_machine.cognition import theory_consolidation as consolidation
from hegel_machine.cognition.language import Program
from test_question_routing import theory


def test_snapshot_anchors_each_history_record_once_without_an_initial_tree_copy(
        tmp_path, monkeypatch):
    reference = {'file': str(tmp_path / 'research_state.json'), 'json_path': ['theory_index']}
    shared_program = Program('add', children=(Program('feature', name='u'),
                                            Program('const', value=2.)))
    old = theory('retained', shared_program, version=1,
        source_snapshot={'version': 1, 'index_ref': {'file': 'original_index.json', 'json_path': [0]}},
        counterexamples=[{'file': 'old_paid.json', 'json_path': ['rows', 0]}])
    # The actual index can retain the same literal Program in several versions.
    prior = {**deepcopy(old), 'version': 2, 'program': old['program'], 'history': [old]}
    current = {**deepcopy(old), 'version': 3, 'program': old['program'], 'history': [prior]}
    indexed = {'retained': current}
    before = deepcopy(indexed)
    expected = consolidation._anchor_file_references(indexed, tmp_path)
    entries = (current, prior, old)
    copy, anchor = consolidation.deepcopy, consolidation._anchor_file_references
    visited = []

    def no_redundant_entry_copy(value, *args, **kwargs):
        assert all(value is not entry for entry in entries), 'copied a full entry before anchoring'
        return copy(value, *args, **kwargs)

    def once_per_record(value, origin):
        assert 'history' not in value, 'anchored already snapshotted histories again'
        visited.append(value['version'])
        return anchor(value, origin)

    monkeypatch.setattr(consolidation, 'deepcopy', no_redundant_entry_copy)
    monkeypatch.setattr(consolidation, '_anchor_file_references', once_per_record)
    result = consolidation._snapshot_index(indexed, {}, reference)

    assert visited == [3, 2, 1]
    assert result == expected and indexed == before
    result['retained']['program']['children'][1]['value'] = 9.
    result['retained']['history'][0]['counterexamples'][0]['json_path'].append('changed')
    assert result['retained']['history'][0]['program']['children'][1]['value'] == 2.
    oldest = result['retained']['history'][0]['history'][0]
    assert oldest['program']['children'][1]['value'] == 2.
    assert oldest['counterexamples'][0]['json_path'] == ['rows', 0]
    assert indexed == before


def test_snapshot_resolves_source_versions_without_mutating_original_refs_or_banks(tmp_path):
    bank = [{'candidates': [{'program': Program('const', value=value).to_dict(),
                            'status': 'refuted' if value == 1. else 'candidate'}]}
            for value in (1., 2.)]
    root = {'questions': [{'id': 'source', 'meaning_version': 1,
                          'native_condition_research': bank}]}
    source_ref = lambda round_index: {'kind': 'native_candidate', 'question_id': 'source',
        'meaning_version': 1, 'round_index': round_index, 'candidate_index': 0,
        'original_notes': ['retained']}
    old = theory('native', Program('const', value=-99.), version=1, source_ref=source_ref(0),
                 support={'status': 'candidate', 'evidence': [{'file': 'paid.json', 'json_path': [0]}]})
    current = theory('native', Program('const', value=-99.), version=2,
        source_ref=source_ref(1), history=[old])
    indexed = {'native': current}
    before, root_before = deepcopy(indexed), deepcopy(root)
    reference = {'file': str(tmp_path / 'research_state.json'), 'json_path': ['theory_index']}

    result = consolidation._snapshot_index(indexed, root, reference)['native']

    assert indexed == before and root == root_before
    assert result['program']['value'] == 2. and result['version'] == 2
    assert result['history'][0]['program']['value'] == 1.
    assert result['history'][0]['compatible'] is False
    assert result['source_snapshot'] == {
        'source_ref': source_ref(1), 'index_ref': reference, 'version': 2}
    assert result['history'][0]['source_snapshot']['version'] == 1
    assert result['history'][0]['support']['evidence'][0]['file'] == str(tmp_path / 'paid.json')
    assert 'source_ref' not in result and 'source_ref' not in result['history'][0]
    result['program']['value'] = 999.
    result['source_snapshot']['source_ref']['original_notes'].append('local mutation')
    result['history'][0]['source_snapshot']['index_ref']['json_path'].append('changed')
    assert indexed == before and root == root_before
    assert result['source_snapshot']['index_ref']['json_path'] == ['theory_index']
