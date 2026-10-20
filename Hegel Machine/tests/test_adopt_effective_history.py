"""Verified cold units publish their restore references before old data is removed."""
from copy import deepcopy
import json
import gzip
import os
from pathlib import Path
import shlex

import pytest

from hegel_machine.cognition.effective_history import HistoryStore, compact_unit, restore_unit
from scripts import adopt_effective_history as adoption


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def cold(tmp_path):
    artifacts, d_base, c_base = tmp_path / 'artifacts', tmp_path / 'D', tmp_path / 'C'
    sources = {'D/old': d_base / 'archives/old.tar.7z',
        'C/report.json': c_base / 'artifacts/report.json.gz',
        'C-subtree/study/author/trials': c_base / 'subtrees/study/author/trials.tar.gz'}
    for unit, path in sources.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('original container ' + unit).encode())
    d_plan, c_index = d_base / 'archive_plan.json', c_base / 'original_paths.json'
    pointer, state = artifacts / 'old/ARCHIVE_POINTER.json', d_base / 'states/old.json'
    write(pointer, {'archive': str(sources['D/old']), 'manifest': 'keep-original-manifest'})
    write(state, {'status': 'completed', 'unit': 'old', 'archive': str(sources['D/old']),
                  'removed_files': 5, 'removed_bytes': 123, 'verified_bytes': 456})
    write(d_plan, {'root': str(artifacts), 'destination': str(d_base),
                   'units': [{'name': 'old', 'directory': True, 'manifest': 'keep-original-manifest'}]})
    write(c_index, {'scope': 'original scientific records', 'destination': str(c_base / 'artifacts'),
        'files': [{'source': str(artifacts / 'report.json'), 'bytes_at_scan': 900,
                   'preserved_summaries': ['original-summary.json']}],
        'subtree_archives': [{'source': str(artifacts / 'study/author/trials'),
            'archive': str(sources['C-subtree/study/author/trials']), 'files': 20,
            'original_bytes': 1000, 'status': 'original_removed'}]})
    store_path, prepared = tmp_path / 'shared.sqlite3', tmp_path / 'prepared.jsonl'
    store = HistoryStore(store_path)
    receipts = []
    for unit, source in sources.items():
        stat = source.stat()
        receipt = {'unit': unit, 'source': str(source), 'source_bytes': stat.st_size,
            'source_mtime_ns': stat.st_mtime_ns, 'store': str(store_path), 'status': 'verified',
            'new_chunk_stored_bytes': 7, 'verified_payload_bytes': 1000}
        receipts.append(receipt)
        store.connection.execute('INSERT INTO units VALUES(?,?,?)',
                                  (unit, 'verified', json.dumps(receipt)))
    store.connection.commit()
    store.close()
    prepared.write_text(''.join(json.dumps(row) + '\n' for row in receipts))
    return {'sources': sources, 'd_plan': d_plan, 'c_index': c_index, 'pointer': pointer,
            'state': state, 'store': store_path, 'prepared': prepared,
            'manifest': tmp_path / 'adoption.jsonl', 'artifacts': artifacts, 'receipts': receipts}


def run(cold, **kwargs):
    return adoption.adopt(cold['prepared'], cold['manifest'], d_plan=cold['d_plan'],
                          c_index=cold['c_index'], **kwargs)


def test_d_publishes_pointer_state_and_plan_before_original_unlink(cold, monkeypatch):
    original = cold['sources']['D/old']
    unlink = Path.unlink

    def after_references(path, *args, **kwargs):
        if path == original:
            for target in [cold['pointer'], cold['state']]:
                value = json.loads(target.read_text())
                assert value['archive'] == str(cold['store']) and value['archive_unit'] == 'D/old'
            assert json.loads(cold['d_plan'].read_text())['units'][0]['archive'] == str(cold['store'])
            assert adoption.json_lines(cold['manifest'])[-1]['stage'] == 'references_updated'
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', after_references)
    result = run(cold, apply=True, only=['D/old'])
    assert not original.exists() and cold['sources']['C/report.json'].exists()
    state = json.loads(cold['state'].read_text())
    assert state['removed_files'] == 5 and state['verified_bytes'] == 456
    assert state['archive_bytes'] == 7 and state['archive_shared_store'] is True
    assert shlex.split(state['restore'])[-1] == str(cold['artifacts'])
    assert len(result['shared_stores']) == 1
    assert adoption.json_lines(cold['manifest'])[-1]['stage'] == 'adopted'


def test_c_gzip_and_subtree_update_existing_index_without_repeating_store_size(cold):
    before = json.loads(cold['c_index'].read_text())
    result = run(cold, apply=True, only=['C/report.json', 'C-subtree/study/author/trials'])
    index = json.loads(cold['c_index'].read_text())
    for field in ['source', 'bytes_at_scan', 'preserved_summaries']:
        assert index['files'][0][field] == before['files'][0][field]
    for field in ['source', 'files', 'original_bytes', 'status']:
        assert index['subtree_archives'][0][field] == before['subtree_archives'][0][field]
    assert index['files'][0]['member'] == 'report.json'
    assert index['files'][0]['archive_unit'] == 'C/report.json'
    assert index['subtree_archives'][0]['archive_unit'] == 'C-subtree/study/author/trials'
    assert all(row['archive'] == str(cold['store']) for row in [index['files'][0], index['subtree_archives'][0]])
    assert all(row['archive_bytes'] == 7 and row['archive_shared_store'] is True
               for row in [index['files'][0], index['subtree_archives'][0]])
    assert not cold['sources']['C/report.json'].exists()
    assert not cold['sources']['C-subtree/study/author/trials'].exists()
    assert cold['sources']['D/old'].exists()
    assert len(result['shared_stores']) == 1
    assert result['shared_stores'][0]['file_bytes'] >= cold['store'].stat().st_size
    assert 'DESTINATION' not in (cold['c_index'].parent / 'README.md').read_text()


@pytest.mark.parametrize('failure', ['store_unverified', 'source_size_changed', 'source_mtime_changed'])
def test_unverified_or_changed_unit_keeps_original_and_old_references(cold, failure):
    original = cold['sources']['D/old']
    before = cold['pointer'].read_bytes()
    if failure == 'store_unverified':
        store = HistoryStore(cold['store'])
        store.connection.execute("UPDATE units SET status='packed' WHERE name='D/old'")
        store.connection.commit()
        store.close()
    elif failure == 'source_size_changed':
        original.write_bytes(original.read_bytes() + b'changed')
    else:
        stat = original.stat()
        os.utime(original, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000))
    with pytest.raises(ValueError):
        run(cold, apply=True, only=['D/old'])
    assert original.exists() and cold['pointer'].read_bytes() == before
    assert not cold['manifest'].exists()


def test_external_unverified_receipt_is_not_adopted(cold):
    rows = deepcopy(cold['receipts'])
    rows[0]['status'] = 'packed'
    cold['prepared'].write_text(json.dumps(rows[0]) + '\n')
    assert run(cold, apply=True)['units'] == 0
    assert cold['sources']['D/old'].exists() and not cold['manifest'].exists()


def test_resume_after_unlink_and_before_final_event_uses_published_references(cold, monkeypatch):
    record = adoption.event

    def interrupted(path, value):
        if value['stage'] == 'adopted':
            raise OSError('interrupted after original removal')
        record(path, value)

    monkeypatch.setattr(adoption, 'event', interrupted)
    with pytest.raises(OSError):
        run(cold, apply=True, only=['D/old'])
    assert not cold['sources']['D/old'].exists()
    assert adoption.json_lines(cold['manifest'])[-1]['stage'] == 'references_updated'
    monkeypatch.setattr(adoption, 'event', record)
    resumed = run(cold, apply=True, only=['D/old'])
    assert resumed['units'] == 1 and adoption.json_lines(cold['manifest'])[-1]['stage'] == 'adopted'
    lines = cold['manifest'].read_bytes()
    assert run(cold, apply=True, only=['D/old'])['units'] == 1
    assert cold['manifest'].read_bytes() == lines


def test_recorded_restore_command_preserves_existing_scientific_file(cold, tmp_path):
    source = cold['sources']['C/report.json']
    with gzip.open(source, 'wb') as stream:
        stream.write(b'{"historical_paid_evidence":[1,2,3]}')
    actual_store = tmp_path / 'actual.sqlite3'
    work = tmp_path / 'work'
    work.mkdir()
    store = HistoryStore(actual_store)
    try:
        receipt = compact_unit(store, {'unit': 'C/report.json', 'source': str(source),
            'kind': 'gzip', 'member': 'report.json'}, work)
    finally:
        store.close()
    cold['prepared'].write_text(json.dumps(receipt) + '\n')
    original = cold['artifacts'] / 'report.json'
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(b'{"current_unique_evidence":true}')
    original.chmod(0o640)
    os.utime(original, ns=(123456789000, 123456789000))
    before = original.stat()
    run(cold, apply=True, only=['C/report.json'])
    row = json.loads(cold['c_index'].read_text())['files'][0]
    command = shlex.split(row['restore'])
    store = HistoryStore(command[command.index('--store') + 1], readonly=True)
    try:
        assert restore_unit(store, command[command.index('--unit') + 1],
                            command[command.index('--destination') + 1]) == 0
    finally:
        store.close()
    assert original.read_bytes() == b'{"current_unique_evidence":true}'
    assert (original.stat().st_mode, original.stat().st_mtime_ns) == (before.st_mode, before.st_mtime_ns)
    assert not source.exists()
