"""The D2 reader audits saved views and uses the actual target-role helpers."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import SearchBudget
from scripts.audit_parameter_targets import (
    d1_strata, main, retained_views, revision_alignments, view_conflicts,
)


def test_original_attempts_are_indexed_once_without_dropping_duplicate_views():
    attempts = [
        {'output_hole': 'after', 'input_holes': ['old'], 'evidence_inputs': ['residual']},
        {'output_hole': 'after', 'input_holes': ['old'], 'evidence_inputs': ['residual']},
        {'output_hole': 'other', 'input_holes': ['old'], 'evidence_inputs': ['residual']},
    ]
    original = deepcopy(attempts)
    budget = SearchBudget(10)
    views = retained_views({'attempts': attempts}, budget)
    assert [item['attempt_indices'] for item in views] == [[0, 1], [2]]
    assert views[0]['input_names'] == ['old', 'residual']
    assert budget.spent == 3 and attempts == original


def test_held_conflict_is_diagnostic_and_cannot_prune_the_training_view():
    scope = {'records': [{'features': {'x': 1.}, 'targets': {'after': value}}
                          for value in (3., 3., 100.)], 'training_indices': [0, 1], 'held_indices': [2]}
    views = [{'output_hole': 'after', 'input_names': ['x'], 'attempt_indices': [0, 1]}]
    budget = SearchBudget(30)
    item, = view_conflicts(scope, views, 1e-10, budget)
    assert item['training']['status'] == 'UNKNOWN'
    assert item['held_inclusive_diagnostic']['status'] == 'conflict'
    assert not item['prune_from_training_conflict'] and not item['held_used_for_admission']
    assert budget.spent == 5
    scope['records'][1]['targets']['after'] = 4.
    item, = view_conflicts(scope, views, 1e-10, budget)
    assert item['prune_from_training_conflict']
    assert item['training']['witnesses'][0]['source_record_indices'] == [0, 1]


def candidate(identifier):
    old, new = Program('arg', name='old'), Program('arg', name='new')
    u = Program('feature', name='u')
    return {'id': identifier, 'parent_holes': ['old'], 'holes': [
        {'name': name, 'output_unit': '1'} for name in ('old', 'new')],
        'before_pattern': Program('mul', children=(old, u)).to_dict(),
        'definition': Definition(identifier, ('old', 'new'),
            Program('mul', children=(u, new))).to_dict(),
        'fixed_role_signature': {'u': {'role': 'public_control', 'unit': '1', 'time': 'current'}},
        'output_unit': '1'}


def test_actual_replacement_alignment_uses_role_not_matching_slot_position(monkeypatch):
    monkeypatch.setattr(Program, 'predict', lambda *a, **k: pytest.fail('No prediction'))
    before = Program('mul', children=(Program('const', value=1.), Program('feature', name='u')))
    after = Program('mul', children=(Program('feature', name='u'), Program('const', value=2.)))
    c = candidate('same')
    record = {'source_id': 'toy', 'before_ref': {'file': 'old.json'}, 'after_ref': {'file': 'new.json'},
        'bindings': {'old': before.children[0], 'new': after.children[1]},
        '_before_program': before, '_after_program': after,
        '_roles': c['fixed_role_signature'], '_output_unit': '1'}
    result = revision_alignments({6: c, 7: c}, {6: [record], 7: [record]}, SearchBudget(1000))
    item, = result
    assert item['alignment']['status'] == 'known'
    mapping, = item['alignment']['mappings']
    assert mapping['before_parameter_paths'] == [[0]]
    assert mapping['after_parameter_paths'] == [[1]]
    assert mapping['parameter_scale'] == 1.


def test_comparable_output_does_not_make_an_unmatched_parent_input_comparable():
    cases = [{'case_id': '6:0', 'observed_outcome': 'training_failed',
              'parameter_identifiability': 'full_rank_linear_design',
              'ridge_training_rejection_removed': False}]
    searches = {6: {'attempts': [{'output_hole': 'offset', 'input_holes': ['wrong_role']}]}}
    item, = d1_strata(cases, searches, {6: {'offset': {'status': 'comparable'}}},
                      {6: {'actual_offset'}}, SearchBudget(10))
    assert item['target_status'] == 'comparable'
    assert item['input_role_status'] == item['full_view_role_status'] == 'UNKNOWN'
    assert item['unmatched_parent_input_holes'] == ['wrong_role']
    assert item['D1_observed_outcome'] == 'training_failed'


def test_small_full_reader_preserves_saved_results_and_never_generates_or_fits(tmp_path, monkeypatch):
    from hegel_machine.cognition import revision_parameter_relations, synthesis

    def forbidden(*args, **kwargs):
        pytest.fail('The historical audit cannot predict, fit or enumerate')

    monkeypatch.setattr(Program, 'predict', forbidden)
    monkeypatch.setattr(revision_parameter_relations, 'fit_parameter_slots', forbidden)
    monkeypatch.setattr(synthesis, 'iter_synthesis_structures', forbidden)
    programs, records, revisions, numeric = [], [], [], []
    for index in range(2):
        before = Program('mul', children=(Program('const', value=float(index + 1)),
                                          Program('feature', name='u')))
        after = Program('mul', children=(Program('feature', name='u'),
                                         Program('const', value=float(2 * (index + 1)))))
        programs.append({'before': before.to_dict(), 'after': after.to_dict()})
        refs = {side + '_ref': {'file': str(tmp_path / 'programs.json'), 'json_path': [index, side]}
                for side in ('before', 'after')}
        records.append({**refs, 'source_id': f'toy_{index}', 'episode_id': f'toy_{index}'})
        revisions.append({**refs, 'fit_state_refs': {}})
        numeric.append({**refs, 'source_id': f'toy_{index}',
            'features': {'old': float(index + 1), 'evidence:residual': 1.},
            'targets': {'new': float(2 * (index + 1))}})
    searches, metadata, candidates = [{} for _ in range(6)], {}, {}
    for index in (6, 7, 8):
        name = f'toy_{index}'
        candidates[name] = candidate(name)
        searches.append({'candidate_id': name, 'source_records': records,
            'input_scope': 'sparse_parent_evidence_pairs', 'contract': {'tolerance': 1e-10},
            'attempts': [{'output_hole': 'new', 'input_holes': ['old'],
                          'evidence_inputs': ['evidence:residual']}]})
        metadata[str(index)] = {'candidate_id': name, 'records': numeric,
                               'training_indices': [0], 'held_indices': [1]}
    case = lambda index: {'case_id': f'{index}:0', 'observed_outcome': 'training_failed',
        'parameter_identifiability': 'full_rank_linear_design', 'ridge_training_rejection_removed': False}
    files = {'programs.json': programs, 'library.json': {'candidates': candidates},
        'source.json': {'library_ref': {'file': str(tmp_path / 'library.json'), 'json_path': []},
                        'feedback': {'parameter_relation_searches': searches}},
        'metadata.json': metadata, 'fit_state.json': {'revisions': revisions},
        'pilot.json': {'cases': [case(6)]}, 'all.json': {'cases': [
            {**case(6), 'reused_from': {'file': str(tmp_path / 'pilot.json'), 'json_path': ['cases', 0]}},
            case(7), case(8)]}, 'manifest.json': {'scope': 'Toy metadata reader only'}}
    for name, data in files.items():
        (tmp_path / name).write_text(json.dumps(data))
    before_files = {name: (tmp_path / name).read_text() for name in files}
    output = main(['--summary', str(tmp_path / 'source.json'), '--metadata-rows', str(tmp_path / 'metadata.json'),
        '--source-fit-state', str(tmp_path / 'fit_state.json'), '--d1-all', str(tmp_path / 'all.json'),
        '--d1-pilot', str(tmp_path / 'pilot.json'), '--runtime-manifest', str(tmp_path / 'manifest.json'),
        '--output-dir', str(tmp_path / 'audit'), '--compute-budget', '10000'])
    assert output['summary']['scope_count'] == 3 and output['summary']['D1_case_count'] == 3
    assert output['summary']['actual_retention_revisions'] == 2
    assert output['summary']['D1_target_status'] == {'comparable': 3}
    assert output['new_compute_cost'] == sum(output['charged_work'].values()) > 0
    assert not output['new_predictions'] and not output['new_fits'] and not output['factory_enumeration']
    assert all(report['target_diagnosis']['next_actions'] for report in output['scopes'])
    assert all(report['full_input_sufficiency'] == 'UNKNOWN' for report in output['scopes'])
    pack = json.loads((tmp_path / 'audit/source_pack.json').read_text())
    assert len(pack['program_endpoints']) == 4
    assert all((tmp_path / name).read_text() == text for name, text in before_files.items())
