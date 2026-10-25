"""Shifted host snapshots retain exact old versions before shared canonical writes."""

import gzip
import json
import os
from pathlib import Path

import pytest

from hegel_machine.cognition.effective_history import HistoryStore, compact_unit, restore_unit
from scripts import adopt_effective_history as adoption
from scripts import run_online_inquiry_llm as runner


def write_host_pair(root):
    root.mkdir(parents=True)
    active = {'status': 'researching', 'evidence': ['paid input'] * 2000,
              'counterexamples': [{'target': 3, 'prediction': 4}],
              'costs': {'observation_cost': 97, 'compute_cost': 1200}}
    final = {'status': 'method_step_limit', 'evidence': active['evidence'],
             'counterexamples': active['counterexamples'] + [{'target': 9, 'prediction': 7}],
             'costs': {'observation_cost': 97, 'compute_cost': 1400}, 'failure': 'unresolved'}
    for name, value in [('active.json', active), ('training.json', final)]:
        (root / name).write_text(json.dumps(value))
    return active, final


def test_file_kind_retains_even_valid_gzip_as_exact_raw_bytes_and_metadata(tmp_path):
    path = tmp_path / 'raw_checkpoint'
    original = gzip.compress(b'raw file bytes, not an instruction to decompress')
    path.write_bytes(original)
    path.chmod(0o640)
    os.utime(path, ns=(123456789012345678, 234567890123456789))
    store = HistoryStore(tmp_path / 'store.sqlite3')
    try:
        saved = compact_unit(store, {'unit': 'raw/version', 'source': str(path),
            'kind': 'file', 'member': 'raw_checkpoint'}, tmp_path)
        assert saved['status'] == 'verified' and saved['verified_payload_bytes'] == len(original)
        metadata = json.loads(store.connection.execute('SELECT metadata FROM entries').fetchone()[0])
        assert metadata['encoding'] == 'raw' and 'source_gzip_header' not in metadata
        assert metadata['mtime_ns'] == 234567890123456789
        assert restore_unit(store, 'raw/version', tmp_path / 'restored') == 1
    finally:
        store.close()
    restored = tmp_path / 'restored/raw_checkpoint'
    assert restored.read_bytes() == original
    assert restored.stat().st_mtime_ns == 234567890123456789
    assert restored.stat().st_mode & 0o777 == 0o640


def test_shifted_host_versions_recover_exact_bytes_and_keep_distinct_values(tmp_path):
    root = tmp_path / 'retained'
    active, final = write_host_pair(root)
    paths = [root / 'training.json', root / 'active.json']
    originals = [path.read_bytes() for path in paths]
    store_path = tmp_path / 'store.sqlite3'
    result = adoption.adopt_retained_snapshots([root], tmp_path / 'receipt.jsonl',
        retained_store=store_path, apply=True)
    record = result['results'][0]
    assert result['canonicalized_host_pairs'] == 1
    assert record['action'] == 'canonicalized_preserving_original_versions'
    assert json.loads(paths[0].read_text()) == final
    assert json.loads(paths[1].read_text()) == active
    assert not any(path.is_symlink() for path in paths)
    assert all(list(json.loads(path.read_text()))[-1] == 'status' for path in paths)
    assert record['canonical_comparison']['equal_ranges'][0][1] >= 16384
    store = HistoryStore(store_path, readonly=True)
    try:
        for version, original in zip(record['original_versions'], originals):
            destination = tmp_path / ('restore-' + Path(version['source']).name)
            assert restore_unit(store, version['unit'], destination) == 1
            assert (destination / version['member']).read_bytes() == original
    finally:
        store.close()
    source_before = paths[0].read_bytes()
    with paths[1].open('r+b') as stream:
        stream.seek(100)
        stream.write(b'changed')
    assert paths[0].read_bytes() == source_before


def test_canonical_value_mismatch_leaves_both_entities_and_recoverable_versions(tmp_path, monkeypatch):
    root = tmp_path / 'retained'
    write_host_pair(root)
    paths = [root / 'training.json', root / 'active.json']
    originals = [path.read_bytes() for path in paths]
    real_write = runner.write_json

    def bad_value(path, value, **kwargs):
        if Path(path).name == 'active.json':
            value = {**value, 'costs': {'compute_cost': 0}}
        real_write(path, value, **kwargs)

    monkeypatch.setattr(runner, 'write_json', bad_value)
    manifest, store_path = tmp_path / 'receipt.jsonl', tmp_path / 'store.sqlite3'
    with pytest.raises(ValueError, match='changes scientific values'):
        adoption.adopt_retained_snapshots([root], manifest, retained_store=store_path, apply=True)
    assert [path.read_bytes() for path in paths] == originals
    published = adoption.json_lines(manifest)[0]
    assert published['stage'] == 'retained_host_originals_verified'
    assert all(version['status'] == 'verified' for version in published['original_versions'])


def test_retained_store_preview_does_not_create_database_or_replace_hosts(tmp_path):
    root = tmp_path / 'retained'
    write_host_pair(root)
    paths = [root / 'training.json', root / 'active.json']
    originals = [path.read_bytes() for path in paths]
    store_path = tmp_path / 'store.sqlite3'
    result = adoption.adopt_retained_snapshots([root], tmp_path / 'receipt.jsonl', retained_store=store_path)
    assert result['results'][0]['action'] == 'ready_to_canonicalize_retained_versions'
    assert [path.read_bytes() for path in paths] == originals
    assert not store_path.exists()


def test_second_original_store_verification_failure_keeps_both_original_entities(tmp_path, monkeypatch):
    from hegel_machine.cognition import effective_history

    root = tmp_path / 'retained'
    write_host_pair(root)
    paths = [root / 'training.json', root / 'active.json']
    originals = [path.read_bytes() for path in paths]
    real_compact, calls = effective_history.compact_unit, []

    def failed_second(store, item, work_root, **kwargs):
        calls.append(item['source'])
        if len(calls) == 2:
            raise ValueError('second original restore replay failed')
        return real_compact(store, item, work_root, **kwargs)

    monkeypatch.setattr(effective_history, 'compact_unit', failed_second)
    manifest, store_path = tmp_path / 'receipt.jsonl', tmp_path / 'store.sqlite3'
    with pytest.raises(ValueError, match='second original restore replay failed'):
        adoption.adopt_retained_snapshots([root], manifest, retained_store=store_path, apply=True)
    assert calls == [str(path) for path in paths]
    assert [path.read_bytes() for path in paths] == originals
    assert all(not path.is_symlink() for path in paths)
    assert not manifest.exists(), 'Neither original may be published as replaced before both replays pass.'
    store = HistoryStore(store_path, readonly=True)
    try:
        assert store.connection.execute('SELECT COUNT(*) FROM units WHERE status="verified"').fetchone()[0] == 1
    finally:
        store.close()


def test_host_rewrite_does_not_touch_same_named_model_inputs(tmp_path):
    root = tmp_path / 'retained'
    for relative in ('_evaluated_model', '_submitted_models/saved', 'data'):
        write_host_pair(root / relative)
    originals = {path: path.read_bytes() for path in root.rglob('*.json')}
    store_path = tmp_path / 'store.sqlite3'
    result = adoption.adopt_retained_snapshots([root], tmp_path / 'receipt.jsonl',
        retained_store=store_path, apply=True)
    assert result['pairs'] == 0 and not store_path.exists()
    assert {path: path.read_bytes() for path in root.rglob('*.json')} == originals


def test_closed_interrupted_history_keeps_pending_state_and_complete_recovery(tmp_path):
    root = tmp_path / 'interrupted'
    active, final = write_host_pair(root)
    final['status'] = 'researching'
    (root / 'training.json').write_text(json.dumps(final))
    original = (root / 'training.json').read_bytes()
    store_path = tmp_path / 'store.sqlite3'
    result = adoption.adopt_retained_snapshots([root], tmp_path / 'receipt.jsonl',
        retained_store=store_path, apply=True)
    assert result['canonicalized_host_pairs'] == 1
    assert json.loads((root / 'training.json').read_text()) == final
    assert json.loads((root / 'active.json').read_text()) == active
    version = result['results'][0]['original_versions'][0]
    store = HistoryStore(store_path, readonly=True)
    try:
        restore_unit(store, version['unit'], tmp_path / 'restore')
    finally:
        store.close()
    assert (tmp_path / 'restore' / version['member']).read_bytes() == original
