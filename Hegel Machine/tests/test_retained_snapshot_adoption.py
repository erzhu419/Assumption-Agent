"""Finite retained-file reclamation preserves every distinct result and its costs."""

import json
import os
from pathlib import Path
import struct

from scripts import adopt_effective_history as adoption


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def kernel_equal_ranges(monkeypatch):
    calls = []

    def ioctl(original_fd, operation, request, mutate):
        assert operation == 0xC0189436 and mutate
        values = list(struct.unpack('=QQHHIqQQiI', request))
        source_offset, length, destination_fd, destination_offset = values[0], values[1], values[5], values[6]
        assert source_offset % 4096 == destination_offset % 4096 == length % 4096 == 0
        first = os.pread(original_fd, length, source_offset)
        second = os.pread(destination_fd, length, destination_offset)
        assert first == second, 'Only confirmed equal ranges may reach physical deduplication.'
        calls.append((source_offset, length))
        values[7], values[8] = length, 0
        request[:] = struct.pack('=QQHHIqQQiI', *values)
        return 0

    monkeypatch.setattr(adoption.fcntl, 'ioctl', ioctl)
    return calls


def test_different_counterexample_and_cost_blocks_survive_shared_equal_ranges(tmp_path, monkeypatch):
    calls = kernel_equal_ranges(monkeypatch)
    root = tmp_path / 'completed'
    first = write(root / 'observations.json', {'paid': 'same evidence ' * 3000,
        'counterexample': {'target': 7}, 'costs': {'observation_cost': 97, 'model_calls': 0}})
    second = write(root / 'model_observations.json', {'paid': 'same evidence ' * 3000,
        'counterexample': {'target': 8}, 'costs': {'observation_cost': 98, 'model_calls': 0}})
    originals = [path.read_bytes() for path in (first, second)]
    result = adoption.adopt_retained_snapshots([root], tmp_path / 'receipt.jsonl', apply=True)
    assert calls and result['confirmed_deduped_bytes'] > 0
    assert result['results'][0]['different_content_retained']
    assert [path.read_bytes() for path in (first, second)] == originals
    assert json.loads(second.read_text())['counterexample'] == {'target': 8}
    assert json.loads(second.read_text())['costs']['observation_cost'] == 98
    with second.open('r+b') as stream:
        stream.write(b'changed')
    assert first.read_bytes() == originals[0]
    assert not first.is_symlink() and not second.is_symlink()


def test_exact_completed_checkpoint_exclusions_keep_other_pairs_visible(tmp_path, monkeypatch):
    root = tmp_path / 'partially_completed'
    final = write(root / 'training.json', {'status': 'completed', 'costs': {'compute_cost': 97}})
    active = write(root / 'active.json', {'status': 'researching', 'costs': {'compute_cost': 96}})
    write(root / 'observations.json', {'rows': [1, 2, 3]})
    write(root / 'model_observations.json', {'rows': [1, 2, 3]})
    originals = [path.read_bytes() for path in (final, active)]

    def already_completed(*args, **kwargs):
        raise AssertionError('Completed exact files must not be canonicalized again.')

    monkeypatch.setattr(adoption, '_canonicalize_retained_host_pair', already_completed)
    result = adoption.adopt_retained_snapshots([root], tmp_path / 'resume.jsonl',
        exclude=[final, active], retained_store=tmp_path / 'unused.sqlite3')
    assert result['pairs'] == 1
    assert [row['kind'] for row in result['results']] == ['fitting_views']
    assert [path.read_bytes() for path in (final, active)] == originals


def test_only_finished_exact_checkpoint_aliases_preserve_failed_evidence_and_old_costs(tmp_path):
    root = tmp_path / 'completed'
    final = write(root / 'training.json', {'status': 'provider_failed',
        'counterexamples': [{'target': 4, 'prediction': 3}],
        'costs': {'model_calls': 29, 'observation_cost': 97}, 'failure': 'original provider error'})
    active = root / 'active.json'
    active.write_bytes(final.read_bytes())
    expected = json.loads(active.read_text())
    preview = adoption.adopt_retained_snapshots([root], tmp_path / 'preview.jsonl')
    assert preview['results'][0]['action'] == 'ready_to_alias_identical'
    assert not active.is_symlink()
    result = adoption.adopt_retained_snapshots([root], tmp_path / 'apply.jsonl', apply=True)
    assert result['aliases'] == 1 and active.is_symlink()
    assert active.resolve() == final.resolve()
    assert json.loads(active.read_text()) == expected
    assert result['results'][0]['original_bytes_restored']
    assert result['results'][0]['original_status'] == 'provider_failed'


def test_unequal_unfinished_or_unreadable_snapshots_are_not_aliased(tmp_path):
    root = tmp_path / 'retained'
    cases = [('different', {'status': 'completed', 'costs': {'compute_cost': 12}},
              {'status': 'completed', 'costs': {'compute_cost': 13}}),
             ('unfinished', {'status': 'researching', 'failure': 'still pending'}, None)]
    for name, final, active in cases:
        path = write(root / name / 'training.json', final)
        write(path.parent / 'active.json', final if active is None else active)
    partial = root / 'incomplete'
    partial.mkdir()
    (partial / 'training.json').write_bytes(b'{"status":"completed","events":[')
    (partial / 'active.json').write_bytes((partial / 'training.json').read_bytes())
    before = {path: path.read_bytes() for path in root.rglob('*.json')}
    result = adoption.adopt_retained_snapshots([root], tmp_path / 'receipt.jsonl', apply=True)
    assert result['aliases'] == 0
    assert all(not path.is_symlink() and path.read_bytes() == content for path, content in before.items())
    assert any(row['action'] == 'retained_unfinished_checkpoint' for row in result['results'])
    assert any(row.get('original_status') == 'unreadable_incomplete_json' for row in result['results'])


def test_excluded_current_tree_is_never_inspected(tmp_path, monkeypatch):
    root = tmp_path / 'retained'
    old = root / 'completed'
    current = root / 'current'
    for path in (old, current):
        write(path / 'observations.json', [{'target': 7}])
        write(path / 'model_observations.json', [{'target': 7}])
    compared = adoption._snapshot_ranges

    def only_old(source, destination):
        assert not source.is_relative_to(current) and not destination.is_relative_to(current)
        return compared(source, destination)

    monkeypatch.setattr(adoption, '_snapshot_ranges', only_old)
    result = adoption.adopt_retained_snapshots([root], tmp_path / 'receipt.jsonl', exclude=[current])
    assert result['pairs'] == 1
    assert result['excluded_roots'] == [str(current.resolve())]


def test_kernel_rejection_retains_original_and_records_failure(tmp_path, monkeypatch):
    root = tmp_path / 'completed'
    content = {'paid': 'same' * 3000, 'failed': ['retained counterexample'], 'costs': 97}
    first, second = (write(root / name, content) for name in
                     ('observations.json', 'model_observations.json'))
    before = [path.read_bytes() for path in (first, second)]

    def rejected(original_fd, operation, request, mutate):
        values = list(struct.unpack('=QQHHIqQQiI', request))
        values[7], values[8] = 0, -95
        request[:] = struct.pack('=QQHHIqQQiI', *values)
        return 0

    monkeypatch.setattr(adoption.fcntl, 'ioctl', rejected)
    result = adoption.adopt_retained_snapshots([root], tmp_path / 'receipt.jsonl', apply=True)
    assert result['status'] == 'completed_with_retained_dedupe_failures'
    assert result['confirmed_deduped_bytes'] == 0
    assert [path.read_bytes() for path in (first, second)] == before
    assert result['results'][0]['dedupe'][0]['status'] == -95
