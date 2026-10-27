"""Equal paid contents share method evidence; equal counts alone never do."""

from copy import deepcopy
from types import SimpleNamespace

from hegel_machine.cognition import method_feedback as feedback
from hegel_machine.cognition.checkpoint_storage import read_json_checkpoint, write_json_checkpoint
from hegel_machine.cognition.models import Observation
from test_method_use_conditions import profile


METHOD = {'id': 'method', 'version': 2}
BINDING = {'candidate_id': 'method', 'candidate_version': 2,
    'library_path': 'shared/theory_library.json', 'mapping': {'role_0': 'u'},
    'paid_cutoff': {'rows': 4, 'records': 2},
    'fit_split': {'training_records': [0], 'validation_records': [1]}}


def rows():
    return [Observation({'u': value}, value * 2., ()) for value in (-1., -.5, .5, 1.)]


def agent(path, paid=None):
    path.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(workspace=path, rows=rows() if paid is None else paid,
        theory_consolidation={'candidates': {}, 'method_feedback': {}}, _method_use_profile=profile())


def write(path, value):
    write_json_checkpoint(path, value, ensure_ascii=False, indent=None)
    return {'file': str(path), 'json_path': []}


def outcome(source, basis):
    return {'method_ref': deepcopy(METHOD), 'profile': profile(),
        'feedback_ref': {'file': str(source.workspace / 'method_return.json'), 'json_path': []},
        'evidence_basis': basis, 'status': 'completed',
        'working_comparison': {'complete': True, 'same_rows': True, 'before_loss': .1,
            'after_loss': .2, 'data_role': 'exposed_development'}}


def candidate(path, outcomes):
    return {'id': 'method', 'version': 3, 'kind': 'research_program',
        'method_use_conditions': {'usage_version': deepcopy(METHOD),
            'outcomes_ref': write(path / 'conditions.json', outcomes)}}


def test_same_count_but_changed_actual_paid_contents_reopens_method_use(tmp_path):
    current = agent(tmp_path / 'target')
    old_basis = feedback.method_evidence_basis(BINDING, current)
    method = candidate(current.workspace, [outcome(current, old_basis)])
    current.theory_consolidation['candidates']['method'] = method
    assert feedback.procedure_use_decision(current, method, BINDING)['admission'] == 'defer'
    original = deepcopy(current.rows)
    current.rows[-1] = Observation(original[-1].features, original[-1].target + .25, original[-1].history)
    decision = feedback.procedure_use_decision(current, method, BINDING)
    assert decision['admission'] == 'bounded_exploration'
    assert decision['scope']['evidence_basis']['paid_cutoff'] == old_basis['paid_cutoff']
    assert decision['scope']['evidence_basis']['paid_evidence_ref'] != old_basis['paid_evidence_ref']
    assert read_json_checkpoint(old_basis['paid_evidence_ref']['file'])[-1]['target'] == original[-1].target


def test_restore_to_another_workspace_reuses_identical_paid_view_without_new_snapshot(tmp_path):
    source = agent(tmp_path / 'source')
    basis = feedback.method_evidence_basis(BINDING, source)
    saved_rows = read_json_checkpoint(basis['paid_evidence_ref']['file'])
    restored = agent(tmp_path / 'restored', saved_rows)
    method = candidate(restored.workspace, [outcome(source, basis)])
    restored.theory_consolidation['candidates']['method'] = method
    actual = feedback.method_evidence_basis(BINDING, restored)
    assert actual == basis
    assert feedback.procedure_use_decision(restored, method, BINDING)['admission'] == 'defer'
    assert not list((restored.workspace / 'method_feedback').glob('paid_*.json'))
    assert read_json_checkpoint(actual['paid_evidence_ref']['file']) == saved_rows


def test_stale_workspace_observations_are_not_adopted_as_current_paid_data(tmp_path, monkeypatch):
    current = agent(tmp_path / 'target')
    stale = [{'features': row.features, 'target': 0., 'history': []} for row in current.rows]
    write(current.workspace / 'observations.json', stale)
    monkeypatch.setattr(feedback, 'clone_or_copy', lambda *a: (_ for _ in ()).throw(
        AssertionError('Stale observations cannot become the paid evidence view')))
    basis = feedback.method_evidence_basis(BINDING, current)
    retained = read_json_checkpoint(basis['paid_evidence_ref']['file'])
    assert [row['target'] for row in retained] == [row.target for row in current.rows]
    assert read_json_checkpoint(current.workspace / 'observations.json') == stale


def test_verified_workspace_view_is_shared_once_and_retains_original_contents(tmp_path, monkeypatch):
    current = agent(tmp_path / 'target')
    paid = [{'features': row.features, 'target': row.target, 'history': []} for row in current.rows]
    source = current.workspace / 'observations.json'
    write(source, paid)
    clone, calls = feedback.clone_or_copy, []

    def counted(before, after):
        calls.append((before, after))
        return clone(before, after)

    monkeypatch.setattr(feedback, 'clone_or_copy', counted)
    first = feedback.method_evidence_basis(BINDING, current)
    current.rows = deepcopy(paid)
    assert feedback.method_evidence_basis(BINDING, current) == first
    assert len(calls) == 1
    assert len(list((current.workspace / 'method_feedback').glob('paid_*.json'))) == 1
    write(source, [{'features': {'u': 0.}, 'target': 99., 'history': []}])
    assert read_json_checkpoint(first['paid_evidence_ref']['file']) == paid


def test_equal_imported_evidence_references_align_without_mutating_frozen_outcomes(tmp_path):
    left, right = agent(tmp_path / 'left'), agent(tmp_path / 'right')
    left_basis = feedback.method_evidence_basis(BINDING, left)
    right_basis = feedback.method_evidence_basis(BINDING, right)
    assert left_basis != right_basis
    records = [outcome(left, left_basis), outcome(right, right_basis)]
    current = agent(tmp_path / 'consumer')
    method = candidate(current.workspace, records)
    original = read_json_checkpoint(method['method_use_conditions']['outcomes_ref']['file'])
    assert feedback.procedure_use_decision(current, method, BINDING)['admission'] == 'defer'
    assert read_json_checkpoint(method['method_use_conditions']['outcomes_ref']['file']) == original
    assert not list((current.workspace / 'method_feedback').glob('paid_*.json'))
