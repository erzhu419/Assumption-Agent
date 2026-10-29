"""Paid restoration preserves versions, trajectory scope and local metadata."""

from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation
from hegel_machine.cognition.object_continuation import encode_entry
from scripts.partial_role_paid_scene import load_paid_scene


@pytest.fixture
def saved_scene(tmp_path, monkeypatch):
    def unexpected_prediction(*args, **kwargs):
        raise AssertionError('Restoring paid scores must not predict again')

    monkeypatch.setattr(Program, 'predict', unexpected_prediction)
    root, seed, prior = (tmp_path / name for name in ('root', 'seed', 'repair'))
    report_path = root / 'runs/106029101/pr3/response_report.json'
    x = Program('feature', name='x')
    winner = Program('add', children=(x, Program('const', value=1.)))

    def entry(name, program, structure=None):
        value = {'candidate': Candidate(name, program, 'retained'), 'kind': 'structural:test',
                 'family_id': 'T08', 'training_loss': 99., 'validation_loss': 99.}
        if structure:
            value['structure'] = structure
        return encode_entry(value)

    structure = {'radius_interval': [.2, .4], 'fits': [{'response_terms': [x.to_dict()]}]}
    old = entry('old', x)
    wrong = entry('reused_id', Program('const', value=4.), {'radius_interval': [8., 9.]})
    new = entry('reused_id', winner, structure)
    first = [asdict(Observation({'x': 0.}, 0., ())),
             asdict(Observation({'x': 1.}, 1., ({'x': 0., '__target__': 0.},)))]
    report = {'program': x.to_dict(), 'fallback': .3,
        'object_continuation_state': {'finalists': [old], 'candidate_frontiers': [{'entries': [wrong]}],
                                      'selected': old},
        'actions': [{'noise_assessment': {'counterexample_baseline_program': x.to_dict(),
            'structural_counterexample': {'program': Program('const', value=2.).to_dict()}}}],
        'batches': [
            {'channels': ['fine'], 'observations': first, 'transitions': [{'before': {'x': 0.}, 'after': {'x': 1.}}]},
            {'channels': [], 'observations': [asdict(Observation({'control': 2.}, 2., ()))]},
            {'channels': ['fine'], 'observations': deepcopy(first), 'transitions': []}]}
    selection = {'retained_pool_ref': {'report': str(report_path)},
        'seed_claims_ref': str(seed / 'new_claims.json'), 'source_support_action': 0,
        'action': {'question_id': 'paid_residual_0', 'uncompared_candidate_ids': []},
        'paid_rows': 4, 'scores': [
            {'candidate_id': 'old', 'loss': 1.}, {'candidate_id': 'reused_id', 'loss': 9.},
            {'candidate_id': 'reused_id', 'loss': .2}, {'candidate_id': 'old', 'loss': 1.},
            {'candidate_id': 'answer:paid_residual_0:combined', 'loss': 4.}]}

    def write(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    write(report_path, report)
    write(root / 'registration.json', {'tasks': [{'seed': 106029101}]})
    write(seed / 'new_claims.json', [new])
    write(prior / 'working_selection.json', selection)
    write(prior / 'frozen_predictor.json', {'program': winner.to_dict(), 'fallback': .3,
        'selected_channels': ['fine'], 'source_support_ref': {'file': str(report_path), 'action': 0}})
    write(prior / 'receipt.json', {'confirmation': {'rows': 2, 'incumbent_mse': .5, 'selected_mse': 0.}})
    write(prior / 'confirmation_observations.json', {'batch': {'channels': ['fine'],
        'observations': deepcopy(first), 'transitions': []}})
    write(prior / 'paid_residual_readout.json', {'cumulative_development_compute': 35511,
        'groups': [{'batch': 2, 'counterexamples': [{'row': 1}]}]})
    return root, seed, prior, x, winner, structure


def test_versions_with_one_id_keep_their_own_paid_scores(saved_scene):
    root, seed, prior, x, winner, _ = saved_scene
    scene = load_paid_scene(root, seed, prior)
    assert len(scene['pool']) == 5
    assert len(scene['prior_scores']) == 4
    assert scene['prior_scores'][winner] == .2
    assert scene['prior_scores'][Program('const', value=4.)] == 9.
    assert scene['exposed_scores'] == {x: .5, winner: 0.}
    child = Program('add', children=(x, Program('const', value=2.)))
    assert scene['pool'][-1]['candidate'].program == child
    assert scene['previous_compute'] == 35511


def test_equal_measurements_and_full_histories_are_not_deduplicated(saved_scene):
    root, seed, prior, *_ = saved_scene
    scene = load_paid_scene(root, seed, prior)
    assert len(scene['prior_rows']) == 4 and len(scene['exposed_rows']) == 2
    assert len(scene['paid_rows']) == 6
    assert scene['paid_rows'][0] == scene['paid_rows'][2] == scene['paid_rows'][4]
    assert [len(row.history) for row in scene['paid_rows']] == [0, 1, 0, 1, 0, 1]
    controls = scene['source_refs']['other_instrument_trajectories']
    assert len(controls) == 1 and controls[0]['channels'] == []
    assert all('control' not in row.features for row in scene['paid_rows'])
    assert scene['source_refs']['paid_trajectories'][-1]['role'] == 'already_exposed_evidence'
    assert scene['counterexample_indices'] == [3]
    assert scene['paid_rows'][scene['counterexample_indices'][0]].history == ({'x': 0., '__target__': 0.},)


def test_parent_metadata_comes_from_actual_program_not_reused_id(saved_scene):
    root, seed, prior, _, winner, structure = saved_scene
    scene = load_paid_scene(root, seed, prior)
    assert scene['parent']['candidate'].program == winner
    assert scene['parent']['structure'] == structure
    assert scene['source_refs']['parent'] == {'file': str(seed / 'new_claims.json'), 'json_path': [0]}
    assert scene['source_refs']['parent'] != scene['source_refs']['consumed_answer']


def test_score_order_mismatch_stops_instead_of_assigning_another_candidate_loss(saved_scene):
    root, seed, prior, *_ = saved_scene
    path = prior / 'working_selection.json'
    selection = json.loads(path.read_text())
    selection['scores'][0], selection['scores'][1] = selection['scores'][1], selection['scores'][0]
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match='score order'):
        load_paid_scene(root, seed, prior)
