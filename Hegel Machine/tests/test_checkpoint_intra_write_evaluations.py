"""Staged only: do not run until the current frozen comparison has closed."""
from copy import deepcopy
import errno
import json
import os
import shutil
import struct

import pytest

from hegel_machine.cognition import checkpoint_storage as storage


@pytest.mark.parametrize('mode', ['reflink', 'same_descriptor_fallback', 'different_role_order'])
def test_new_comparison_shared_with_method_feedback_without_losing_unique_metadata(
        tmp_path, monkeypatch, mode):
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
        assert not same_file_ranges
    else:
        # More than one fallback chunk catches seek aliasing on the same fd.
        assert sum(length for _, length, _ in same_file_ranges) > 1024 * 1024
