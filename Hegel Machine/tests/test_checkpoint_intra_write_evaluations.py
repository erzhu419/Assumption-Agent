"""Ordinary comparison histories share exact scores without losing their owners."""
from copy import deepcopy
from contextlib import ExitStack
import errno
import json
import os
import shutil
import struct

import pytest

from hegel_machine.cognition import checkpoint_storage as storage
from hegel_machine.cognition.research_episodes import load_source_values


def mock_reflinks(monkeypatch, mode='reflink'):
    same_file_ranges = []

    def clone(source, destination):
        shutil.copyfile(source, destination)
        return True

    def clone_range(destination, operation, arguments):
        assert operation == 0x4020940D
        descriptor, first, length, target = struct.unpack('qQQQ', arguments)
        assert first % 4096 == target % 4096 == length % 4096 == 0
        if descriptor == destination:
            # The first complete value precedes the second; no overlap is legal.
            assert first + length <= target
            same_file_ranges.append((first, length, target))
            if mode == 'same_descriptor_fallback':
                raise OSError(errno.EOPNOTSUPP, 'same-file clone not supported')
        os.pwrite(destination, os.pread(descriptor, length, first), target)

    monkeypatch.setattr(storage, '_clone_file', clone)
    monkeypatch.setattr(storage.fcntl, 'ioctl', clone_range)
    return same_file_ranges


def field_span(path, parts):
    with ExitStack() as opened:
        source = storage._JsonSource(path, opened)
        node = None
        for part in parts:
            node = source.members(node)[part]
        return source.byte_position(node[1]), source.byte_position(node[2])


def shared_between(ranges, first, second):
    return sum(length for source, length, target in ranges
        if first[0] <= source and source + length <= first[1]
        and second[0] <= target and target + length <= second[1])


@pytest.mark.parametrize('mode', ['reflink', 'same_descriptor_fallback', 'different_role_order'])
def test_new_comparison_shared_with_method_feedback_without_losing_unique_metadata(
        tmp_path, monkeypatch, mode):
    same_file_ranges = mock_reflinks(monkeypatch, mode)
    source, target = tmp_path / 'old.json', tmp_path / 'research_state.json'
    scores = [{'path': f'model_{index}.py', 'mse': index / 100,
        'roles': {'parent': 1, 'child': 2},
        'paid_residuals': {'examples': [{'target': 1., 'prediction': 2.,
            'history': [0.1] * 400}]}}
        for index in range(1000)]
    previous = {'operator_results': [{'index': 0, 'evaluations': scores,
        'costs': {'compute': 1}}],
        'method_results': [{'step': 0, 'feedback': {'evaluations': deepcopy(scores),
            'costs': {'compute': 1}}}]}
    source.write_text(json.dumps(previous))
    original_bytes = source.read_bytes()
    new_scores = deepcopy(scores)
    for score in new_scores:
        score['mse'] += 1.
        score['paid_residuals']['examples'][0]['prediction'] = 3.
    current = deepcopy(previous)
    current['operator_results'].append({'index': 1, 'evaluations': new_scores,
        'costs': {'compute': 97}, 'failure': 'counterexample A',
        'adoption': 'incumbent_retained'})
    current['method_results'].append({'step': 1, 'feedback': {
        'evaluations': deepcopy(new_scores), 'costs': {'compute': 194},
        'failure': 'different failure', 'adoption': 'candidate_retained'}})
    if mode == 'different_role_order':
        current['method_results'][1]['feedback']['evaluations'][0]['roles'] = {
            'child': 2, 'parent': 1}
    storage.write_json_checkpoint(target, current, base=source, indent=None)
    restored = json.loads(target.read_text())
    assert storage._json_values_equal(restored, current)
    assert source.read_bytes() == original_bytes
    assert restored['operator_results'][1]['costs']['compute'] == 97
    assert restored['method_results'][1]['feedback']['costs']['compute'] == 194
    assert restored['operator_results'][1]['failure'] == 'counterexample A'
    assert restored['method_results'][1]['feedback']['failure'] == 'different failure'
    assert restored['operator_results'][1]['adoption'] == 'incumbent_retained'
    assert restored['method_results'][1]['feedback']['adoption'] == 'candidate_retained'
    if mode == 'different_role_order':
        assert list(restored['method_results'][1]['feedback']['evaluations'][0]['roles']) == [
            'child', 'parent']
        assert not shared_between(same_file_ranges,
            field_span(target, ['operator_results', 1, 'evaluations']),
            field_span(target, ['method_results', 1, 'feedback', 'evaluations']))
    else:
        # More than one fallback chunk catches seek aliasing on the same fd.
        assert sum(length for _, length, _ in same_file_ranges) > 1024 * 1024


@pytest.mark.parametrize(('prior', 'difference'), [
    ('no_large_prior', None), ('equal_ancestors', None),
    ('no_large_prior', 'role_order'), ('equal_ancestors', 'failure_decision')])
def test_comparison_owners_visit_scores_without_large_or_changed_prior_fields(
        tmp_path, monkeypatch, prior, difference):
    ranges = mock_reflinks(monkeypatch)
    scores = [{'path': f'model_{index}.py', 'mse': index / 100,
        'roles': {'parent': 1, 'child': 2},
        'retention_decision': {'status': 'incumbent_retained', 'failed': False},
        'evaluated_model': {'path': 'old.py', 'retained_parameters': [0.1] * 200}}
        for index in range(120)]
    value = {'stage': 'saved', 'retained': 'old paid record ' * 10000,
        'method_results': [{'step': index, 'feedback': {'operator': 'research_choice',
            'evaluations': deepcopy(scores), 'costs': {'compute': index + 7},
            'failure': 'method witness', 'adoption': 'incumbent_retained'}} for index in range(2)],
        'operator_results': [{'index': index, 'operator': 'research_choice',
            'evaluations': deepcopy(scores), 'costs': {'compute': index + 11},
            'failure': 'operator witness', 'adoption': 'candidate_retained'} for index in range(2)],
        'events': [{'action': 'query', 'feedback': {'operator': 'research_choice',
            'evaluations': deepcopy(scores), 'event_only': 'keep this occurrence'}}]}
    source, target = tmp_path / 'old.json', tmp_path / 'research_state.json'
    previous = deepcopy(value)
    if prior == 'no_large_prior':
        previous.update(method_results=[{'step': 0, 'feedback': {'status': 'unresolved'}}],
                        operator_results=[{'index': 0, 'status': 'unresolved'}], events=[])
    source.write_text(json.dumps(previous))
    original_bytes = source.read_bytes()
    value['stage'] = 'current'
    if difference == 'role_order':
        value['operator_results'][1]['evaluations'][0]['roles'] = {'child': 2, 'parent': 1}
    elif difference == 'failure_decision':
        value['operator_results'][1]['evaluations'][0]['retention_decision'].update(
            status='candidate_failed', failed=True)
    storage.write_json_checkpoint(target, value, base=source, indent=None)
    assert storage._json_values_equal(json.loads(target.read_text()), value)
    assert source.read_bytes() == original_bytes
    # The method bank comes first, including two wholly unchanged inherited
    # records. Neither a whole-field clone nor an equal list run may skip it.
    method = field_span(target, ['method_results', 0, 'feedback', 'evaluations'])
    operator = field_span(target, ['operator_results', 0, 'evaluations'])
    assert shared_between(ranges, method, operator) > storage._JSON_SHARED_MINIMUM
    if difference:
        changed = field_span(target, ['operator_results', 1, 'evaluations'])
        assert not shared_between(ranges, method, changed)
    reference = {'file': str(target),
        'json_path': ['method_results', 0, 'feedback', 'evaluations', 0, 'mse']}
    assert load_source_values([reference], tmp_path)[0]['value'] == scores[0]['mse']
    before_bytes, before_stat = target.read_bytes(), target.stat()

    def unexpected(*args, **kwargs):
        raise AssertionError('an equal comparison checkpoint must remain a no-op')

    monkeypatch.setattr(storage, '_clone_file', unexpected)
    monkeypatch.setattr(storage.fcntl, 'ioctl', unexpected)
    storage.write_json_checkpoint(target, value, indent=None)
    after_stat = target.stat()
    assert target.read_bytes() == before_bytes
    assert (after_stat.st_ino, after_stat.st_mtime_ns) == (before_stat.st_ino, before_stat.st_mtime_ns)
    assert not target.with_name(target.name + '.writing').exists()
