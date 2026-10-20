"""Historical records retain every byte while repeated content is stored once."""

import copy
import gzip
import io
import json
import os
from pathlib import Path
import random
import stat
import subprocess
import sys
import tarfile
import zlib

import pytest

from hegel_machine.cognition.effective_history import (
    HistoryStore,
    compact_unit,
    restore_unit,
)


def gzip_bytes(payload, *, filename='checkpoint.json', mtime=1723456789):
    output = io.BytesIO()
    with gzip.GzipFile(filename=filename, fileobj=output, mode='wb',
                       compresslevel=9, mtime=mtime) as stream:
        stream.write(payload)
    return output.getvalue()


def snapshot_bytes():
    generator = random.Random(419)
    value = {
        'status': 'provider_failed',
        'costs': {'observation_cost': 123, 'compute_cost': 4.125,
                  'reported_tokens': 9876},
        'pending_tasks': [3, 4, 5],
        'native_condition_research': [{'round_index': 0, 'evidence': [],
                                       'completion_reserve': None}],
        'results': [{'round': index, 'evidence': generator.randbytes(1024).hex()}
                    for index in range(512)],
        'outgoing_knowledge': {'files': {'model.py': 'def predict(x):\n    return -0.125\n'}},
    }
    return json.dumps(value, ensure_ascii=False, indent=2).encode()


def gzip_item(tmp_path, unit, payload, *, mtime=1723456789):
    source = tmp_path / (unit.replace('/', '_') + '.json.gz')
    source.write_bytes(gzip_bytes(payload, mtime=mtime))
    return {'unit': unit, 'kind': 'gzip', 'source': str(source),
            'member': unit + '.json'}


@pytest.fixture
def history_store(tmp_path):
    store = HistoryStore(tmp_path / 'history.sqlite3')
    try:
        yield store
    finally:
        store.close()


def test_identical_large_snapshots_across_units_reuse_committed_blocks(tmp_path):
    payload = snapshot_bytes()
    database = tmp_path / 'history.sqlite3'
    first = HistoryStore(database)
    try:
        receipt = compact_unit(first, gzip_item(tmp_path, 'C/first', payload), tmp_path)
        assert receipt['chunks_added'] > 1
        block_count = first.connection.execute('SELECT COUNT(*) FROM blocks').fetchone()[0]
        sequence_count = first.connection.execute('SELECT COUNT(*) FROM sequences').fetchone()[0]
    finally:
        first.close()

    second = HistoryStore(database)
    try:
        item = gzip_item(tmp_path, 'D/second', payload, mtime=1723456790)
        receipt = compact_unit(second, item, tmp_path)
        assert receipt['new_chunk_bytes'] == receipt['chunks_added'] == 0
        assert receipt['reused_chunk_bytes'] == len(payload)
        assert receipt['snapshot_aliases'] == 1
        assert second.connection.execute('SELECT COUNT(*) FROM blocks').fetchone()[0] == block_count
        assert second.connection.execute('SELECT COUNT(*) FROM sequences').fetchone()[0] == sequence_count
        assert restore_unit(second, item['unit'], tmp_path / 'restored') == 1
        assert (tmp_path / 'restored' / item['member']).read_bytes() == payload
    finally:
        second.close()


def test_middle_insert_delete_append_preserve_raw_json_failure_costs_and_old_versions(
        tmp_path, history_store):
    original = snapshot_bytes()
    base = json.loads(original)
    versions = [('base', original)]
    for operation in ['insert', 'delete', 'append']:
        value = copy.deepcopy(base)
        if operation == 'insert':
            value['results'].insert(256, {'round': 'inserted', 'evidence': '独有失败证据'})
        elif operation == 'delete':
            del value['results'][256]
        else:
            value['results'].append({'round': 512, 'evidence': 'new paid observation'})
        value['costs']['compute_cost'] = 91.03125
        value['status'] = 'token_budget_exhausted'
        value['pending_tasks'].pop(0)
        value['native_condition_research'].append({'round_index': 1, 'evidence': [False, 0, None]})
        versions.append((operation, json.dumps(value, ensure_ascii=False, indent=2).encode()))

    for name, payload in versions:
        item = gzip_item(tmp_path, 'C/' + name, payload)
        receipt = compact_unit(history_store, item, tmp_path)
        assert receipt['status'] == 'verified'
        if name != 'base':
            assert receipt['reused_chunk_bytes'] > len(payload) // 2
        restore_unit(history_store, item['unit'], tmp_path / 'restored')
        assert (tmp_path / 'restored' / item['member']).read_bytes() == payload

    # Later edits must leave the exact earlier failed state independently restorable.
    restore_unit(history_store, 'C/base', tmp_path / 'earlier')
    assert (tmp_path / 'earlier/C/base.json').read_bytes() == original


def test_tar_gzip_member_restores_original_header_and_decoded_bytes(tmp_path, history_store):
    payload = b'{\n "status": "researching", "costs": {"compute_cost": 123.2500}\n}\n'
    compressed = gzip_bytes(payload, filename='visit03.json', mtime=123456789)
    source = tmp_path / 'history.tar.gz'
    with tarfile.open(source, 'w:gz', format=tarfile.PAX_FORMAT) as archive:
        member = tarfile.TarInfo('study/visit03.json.gz')
        member.size, member.mode, member.mtime = len(compressed), 0o640, 1234.5
        archive.addfile(member, io.BytesIO(compressed))
    item = {'unit': 'D/gzip', 'kind': 'tar', 'source': str(source)}
    compact_unit(history_store, item, tmp_path)
    assert restore_unit(history_store, item['unit'], tmp_path / 'restored') == 1
    restored = (tmp_path / 'restored/study/visit03.json.gz').read_bytes()
    header_end = compressed.index(b'\0', 10) + 1  # Actual filename-bearing gzip header.
    assert restored[:header_end] == compressed[:header_end]
    assert gzip.decompress(restored) == payload
    assert stat.S_IMODE((tmp_path / 'restored/study/visit03.json.gz').stat().st_mode) == 0o640


def test_truncated_gzip_member_keeps_original_incomplete_bytes(tmp_path, history_store):
    incomplete = gzip_bytes(snapshot_bytes())[:-7]
    source = tmp_path / 'interrupted.tar.gz'
    with tarfile.open(source, 'w:gz') as archive:
        member = tarfile.TarInfo('study/checkpoint.json.gz.tmp')
        member.size = len(incomplete)
        archive.addfile(member, io.BytesIO(incomplete))
    item = {'unit': 'D/interrupted', 'kind': 'tar', 'source': str(source)}
    receipt = compact_unit(history_store, item, tmp_path)
    assert receipt['status'] == 'verified'
    metadata = json.loads(history_store.connection.execute(
        'SELECT metadata FROM entries WHERE unit=?', (item['unit'],)).fetchone()[0])
    assert metadata['invalid_gzip_retained_raw'] is True
    restore_unit(history_store, item['unit'], tmp_path / 'restored')
    assert (tmp_path / 'restored/study/checkpoint.json.gz.tmp').read_bytes() == incomplete


def test_tar_code_modes_symlinks_and_hardlinks_restore(tmp_path, history_store):
    source = tmp_path / 'code.tar.gz'
    code = b'#!/usr/bin/env python3\nprint("retained source")\n'
    with tarfile.open(source, 'w:gz', format=tarfile.PAX_FORMAT) as archive:
        directory = tarfile.TarInfo('study')
        directory.type, directory.mode, directory.mtime = tarfile.DIRTYPE, 0o750, 1234.5
        archive.addfile(directory)
        member = tarfile.TarInfo('study/model.py')
        member.size, member.mode, member.mtime = len(code), 0o755, 1234.5
        archive.addfile(member, io.BytesIO(code))
        symbolic = tarfile.TarInfo('study/current.py')
        symbolic.type, symbolic.linkname = tarfile.SYMTYPE, 'model.py'
        symbolic.mtime = 1234.5
        archive.addfile(symbolic)
        hard = tarfile.TarInfo('study/model-copy.py')
        hard.type, hard.linkname, hard.mode, hard.mtime = tarfile.LNKTYPE, 'study/model.py', 0o755, 1234.5
        archive.addfile(hard)
    item = {'unit': 'D/code', 'kind': 'tar', 'source': str(source)}
    compact_unit(history_store, item, tmp_path)
    assert restore_unit(history_store, item['unit'], tmp_path / 'restored') == 4
    restored = tmp_path / 'restored/study'
    assert (restored / 'model.py').read_bytes() == code
    assert stat.S_IMODE((restored / 'model.py').stat().st_mode) == 0o755
    assert stat.S_IMODE(restored.stat().st_mode) == 0o750
    assert (restored / 'model.py').stat().st_mtime == 1234.5
    assert (restored / 'current.py').is_symlink()
    assert os.readlink(restored / 'current.py') == 'model.py'
    assert (restored / 'model-copy.py').stat().st_ino == (restored / 'model.py').stat().st_ino

    # Historical restoration preserves evidence already present at its original path.
    existing = b'# later retained local evidence\n'
    (restored / 'model.py').write_bytes(existing)
    restored.chmod(0o700)
    restore_unit(history_store, item['unit'], tmp_path / 'restored')
    restore_unit(history_store, item['unit'], tmp_path / 'restored', 'study/model.py')
    assert (restored / 'model.py').read_bytes() == existing
    assert stat.S_IMODE(restored.stat().st_mode) == 0o700


def test_committed_store_replay_mismatch_never_becomes_verified(tmp_path, history_store):
    item = gzip_item(tmp_path, 'C/mismatch', snapshot_bytes())
    original = Path(item['source']).read_bytes()

    def corrupt_committed_payload(event):
        if event['stage'] != 'verifying':
            return
        identifier, compressed = history_store.connection.execute(
            'SELECT id,payload FROM blocks ORDER BY id LIMIT 1').fetchone()
        data = zlib.decompress(compressed)
        changed = bytes([data[0] ^ 1]) + data[1:]
        history_store.connection.execute('UPDATE blocks SET payload=? WHERE id=?',
                                         (zlib.compress(changed), identifier))
        history_store.connection.commit()

    with pytest.raises(ValueError, match='Restored snapshot differs'):
        compact_unit(history_store, item, tmp_path, progress=corrupt_committed_payload)
    assert history_store.receipt(item['unit'])[0] == 'packed'
    with pytest.raises(ValueError, match='fully verified'):
        restore_unit(history_store, item['unit'], tmp_path / 'restored')
    assert Path(item['source']).read_bytes() == original


def test_interrupted_packed_unit_resumes_verification_without_repacking(tmp_path, monkeypatch):
    payload = snapshot_bytes()
    item = gzip_item(tmp_path, 'C/interrupted-verification', payload)
    original = Path(item['source']).read_bytes()
    database = tmp_path / 'history.sqlite3'
    first = HistoryStore(database)

    def interrupt_before_verification(event):
        if event['stage'] == 'verifying':
            raise KeyboardInterrupt('Worker stopped after the packing commit')

    try:
        with pytest.raises(KeyboardInterrupt):
            compact_unit(first, item, tmp_path, progress=interrupt_before_verification)
        assert first.receipt(item['unit'])[0] == 'packed'
    finally:
        first.close()
    assert Path(item['source']).read_bytes() == original

    resumed = HistoryStore(database)
    try:
        def reject_repacking(*args, **kwargs):
            raise AssertionError('Committed packed data must be verified without repacking')

        monkeypatch.setattr(resumed, 'save_stream', reject_repacking)
        receipt = compact_unit(resumed, item, tmp_path)
        assert receipt['status'] == resumed.receipt(item['unit'])[0] == 'verified'
        assert receipt['verified_payload_bytes'] == len(payload)
        assert Path(item['source']).read_bytes() == original
        assert restore_unit(resumed, item['unit'], tmp_path / 'restored') == 1
        assert (tmp_path / 'restored' / item['member']).read_bytes() == payload
    finally:
        resumed.close()


def test_real_7z_consumer_memory_error_returns_without_pipe_deadlock(tmp_path):
    raw_tar = tmp_path / 'archive.tar'
    payload = random.Random(12).randbytes(5 * 1024 * 1024)
    with tarfile.open(raw_tar, 'w') as archive:
        member = tarfile.TarInfo('study/large_snapshot.dat')
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    packed = tmp_path / 'history.tar.7z'
    subprocess.run(['7z', 'a', '-t7z', '-mx=0', str(packed), raw_tar.name],
                   cwd=tmp_path, check=True, capture_output=True, timeout=15)
    consumer = '''
import sys
from hegel_machine.cognition.effective_history import source_tar
with source_tar(sys.argv[1]) as archive:
    member = next(iter(archive))
    assert archive.extractfile(member).read(1024)
    print('consumer reached payload', flush=True)
    raise OSError(12, 'injected snapshot consumer allocation failure')
'''
    environment = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
    result = subprocess.run([sys.executable, '-c', consumer, str(packed)],
                            capture_output=True, text=True, env=environment, timeout=15)
    assert result.returncode != 0
    assert 'consumer reached payload' in result.stdout
    assert 'OSError: [Errno 12] injected snapshot consumer allocation failure' in result.stderr
