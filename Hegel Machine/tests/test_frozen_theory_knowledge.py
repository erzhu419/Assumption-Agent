"""Frozen references install into Common without copying source research state."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.question_routing import register_theories, resolved_theories
from hegel_machine.cognition.research_episodes import load_source_values
from hegel_machine.cognition.theory_consolidation import LIBRARY_PATH, INDEX_PATH, _save as save_library
from test_theory_consolidation import agent_scene, affine, consolidate, transfer, assert_no_new_science
from test_question_routing import CONTEXT, SQUARE, future_question, probe, route, theory


def refs(source):
    source._prepare_program_state()
    return {'files': {'shared/old_ability.py': '# retained old ability\n'},
        'theory_consolidation': {'file': str(source.workspace / LIBRARY_PATH), 'json_path': []},
        'theory_index': {'file': str(source.workspace / 'research_state.json'), 'json_path': ['theory_index']}}


def recipient(path, knowledge):
    agent, lab = agent_scene(path)
    # Reconstruct through the actual Common constructor, not restore_research.
    return CommonArchiveInquiryAgent(None, path, arm='hegel', source_task=False,
        config=agent.config, knowledge=knowledge), lab


def test_frozen_library_is_automatically_recalled_and_only_local_changes_are_saved(tmp_path):
    source, lab = agent_scene(tmp_path / 'source')
    consolidated = consolidate(source, lab)
    candidate_id = consolidated['frozen_candidates'][0]['candidate_id']
    knowledge = refs(source)
    original = {key: Path(ref['file']).read_bytes() for key, ref in knowledge.items() if key != 'files'}
    target, target_lab = recipient(tmp_path / 'target', knowledge)
    selected = route(target, target_lab, question=future_question(target),
        theories=[theory('square', SQUARE)], experiments=[probe(target)])
    assert selected['status'] == 'selected'
    assert selected['frozen_recall']['generated_theory_ids']
    assert any(candidate_id in identifier for identifier in selected['frozen_recall']['generated_theory_ids'])
    assert sorted(group['task_answer'] for group in selected['selection']['groups']) == [7., 9.]
    local = json.loads((target.workspace / LIBRARY_PATH).read_text())
    assert local['source_ref'] == knowledge['theory_consolidation']
    assert not set(local).intersection({'candidates', 'episodes', 'materials'})
    assert local['transfers'] and selected['selection']['support_updated'] is False
    saved = target.result('researching')
    assert saved['theory_index'] == target.theory_index  # Existing execution interface is expanded.
    assert LIBRARY_PATH not in saved['outgoing_knowledge']['files']
    assert INDEX_PATH not in saved['outgoing_knowledge']['files']
    index = json.loads((target.workspace / INDEX_PATH).read_text())
    assert not {'source_a', 'source_b'}.intersection(index['entries'])
    again, again_lab = recipient(tmp_path / 'another_target', saved['outgoing_knowledge'])
    assert candidate_id in again.theory_consolidation['candidates']
    assert again.theory_index['square']['program'] == SQUARE.to_dict()
    assert (again.workspace / 'shared/old_ability.py').read_text() == '# retained old ability\n'
    missing = future_question(again, 'missing_contract')
    missing.pop('theory_target')
    result = route(again, again_lab, question=missing, theories=[], experiments=[probe(again)])
    assert result['frozen_recall']['author_work'][0]['reason'] == 'typed_target_binding_required'
    for key, value in original.items():
        assert Path(knowledge[key]['file']).read_bytes() == value
    assert_no_new_science(target, target_lab)
    assert_no_new_science(again, again_lab)


def test_native_source_and_its_versions_do_not_bind_to_target_task_or_evidence(tmp_path):
    source, lab = agent_scene(tmp_path / 'source')
    task = next(q for q in source.questions if q['id'] == 'task')
    task['native_condition_research'] = [{'candidates': [{'program': affine(slope).to_dict(),
        'status': 'candidate'}]} for slope in (2., 4.)]
    source._active_question_id = None  # A saved source need not have a pending native continuation.
    (source.workspace / 'evidence.json').write_text('{"value": 17}')
    for round_index in (0, 1):
        declared = theory('native', context_ref={'episode': 'source'},
            source_ref={'kind': 'native_candidate', 'question_id': 'task',
                        'round_index': round_index, 'candidate_index': 0},
            program_source_ref={'file': 'evidence.json', 'json_path': ['value']},
            support={'status': 'candidate', 'evidence': [{'file': 'evidence.json', 'json_path': ['value']}]})
        declared.pop('program')
        register_theories(source, [declared])
    knowledge = refs(source)
    knowledge.pop('theory_consolidation')  # This test uses the original native index, not a learned library.
    target, target_lab = recipient(tmp_path / 'target', knowledge)
    target_task = next(q for q in target.questions if q['id'] == 'task')
    target_task['native_condition_research'] = [{'candidates': [{'program': SQUARE.to_dict()}]}]
    (target.workspace / 'evidence.json').write_text('{"value": -99}')
    resolved = {item['id']: item for item in resolved_theories(target)}
    assert resolved['native']['program'] == affine(4.).to_dict()
    assert target.theory_index['native']['history'][0]['program'] == affine(2.).to_dict()
    assert 'source_ref' not in resolved['native']
    assert resolved['native']['source_snapshot']['source_ref']['question_id'] == 'task'
    assert resolved['native']['version'] == 2
    reference = resolved['native']['support']['evidence'][0]
    assert reference['file'] == str(source.workspace / 'evidence.json')
    assert load_source_values([reference], target.workspace)[0]['value'] == 17
    assert resolved['native']['program_source_ref'] == reference
    assert_no_new_science(target, target_lab)


def test_checked_source_refutation_keeps_exact_source_addresses_and_local_delta(tmp_path):
    source, lab = agent_scene(tmp_path / 'source')
    consolidated = consolidate(source, lab)
    identifier = consolidated['frozen_candidates'][0]['candidate_id']
    (source.workspace / 'paid_row.json').write_text('{"target": 99}')
    failed = transfer(source, lab, identifier, evidence_checks=[{
        'source_ref': {'file': 'paid_row.json', 'json_path': []},
        'observation': {'features': {'u': 2.}, 'target': 99., 'history': []},
        'context_ref': deepcopy(CONTEXT), 'tolerance': 0.}])
    assert failed['status'] == 'refuted'
    knowledge = refs(source)
    before = Path(knowledge['theory_consolidation']['file']).read_bytes()
    target, target_lab = recipient(tmp_path / 'target', knowledge)
    register_theories(target, [theory('ordinary_alias', affine())])
    alias = next(item for item in resolved_theories(target) if item['id'] == 'ordinary_alias')
    assert alias['compatible'] is False
    source_checks = [ref for ref in alias['scope_refutation']['counterexample_refs']
                     if ref.get('json_path', [])[:1] == ['transfers']]
    assert source_checks and source_checks[0]['file'] == knowledge['theory_consolidation']['file']
    assert load_source_values(source_checks, target.workspace)[0]['value']['status'] == 'failed'
    inherited_check = target.theory_consolidation['transfers'][0]['checks'][0]['source_ref']
    assert inherited_check['file'] == str(source.workspace / 'paid_row.json')
    assert load_source_values([inherited_check], target.workspace)[0]['value']['target'] == 99
    again = transfer(target, target_lab, identifier, theory_id='same_failed_binding')
    assert again['status'] == 'refuted'
    local = json.loads((target.workspace / LIBRARY_PATH).read_text())
    assert len(local['transfers']) == 1 and local['transfers'][0]['theory_id'] == 'same_failed_binding'
    assert not set(local).intersection({'candidates', 'episodes', 'materials'})
    assert Path(knowledge['theory_consolidation']['file']).read_bytes() == before
    assert_no_new_science(target, target_lab)


def test_missing_frozen_reference_fails_visibly_before_any_research(tmp_path):
    with pytest.raises(FileNotFoundError):
        recipient(tmp_path / 'target', {'theory_consolidation': {
            'file': str(tmp_path / 'missing_library.json'), 'json_path': []}})


def test_removed_frozen_objects_stay_removed_after_export_and_reinstallation(tmp_path):
    source, lab = agent_scene(tmp_path / 'source')
    compiled = consolidate(source, lab)
    identifier = compiled['frozen_candidates'][0]['candidate_id']
    knowledge = refs(source)
    original = Path(knowledge['theory_consolidation']['file']).read_bytes()
    target, target_lab = recipient(tmp_path / 'removed', knowledge)
    target.theory_consolidation['candidates'] = {}
    save_library(target)
    local = json.loads((target.workspace / LIBRARY_PATH).read_text())
    assert local['removed_candidates'] == [identifier]
    assert not local.get('candidates')
    outgoing = target.result('researching')['outgoing_knowledge']
    again, again_lab = recipient(tmp_path / 'reinstalled', outgoing)
    assert again.theory_consolidation['candidates'] == {}
    assert {'source_a', 'source_b'}.issubset(again.theory_index)
    assert Path(knowledge['theory_consolidation']['file']).read_bytes() == original
    assert_no_new_science(target, target_lab)
    assert_no_new_science(again, again_lab)


def test_relocated_library_uses_explicit_original_workspace_for_relative_material_refs(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'evidence.json').write_text('{"value": 17}')
    snapshot = tmp_path / 'frozen_copy.json'
    snapshot.write_text(json.dumps({'materials': [{'source_ref': {
        'file': 'evidence.json', 'json_path': ['value']}}]}))
    reference = {'file': str(snapshot), 'json_path': []}
    with pytest.raises(ValueError, match='requires its original workspace'):
        recipient(tmp_path / 'unknown_origin', {'theory_consolidation': reference})
    target, lab = recipient(tmp_path / 'target', {'theory_consolidation': {
        **reference, 'workspace': str(source)}})
    (target.workspace / 'evidence.json').write_text('{"value": -99}')
    evidence = target.theory_consolidation['materials'][0]['source_ref']
    assert evidence['file'] == str(source / 'evidence.json')
    assert load_source_values([evidence], target.workspace)[0]['value'] == 17
    assert_no_new_science(target, lab)
