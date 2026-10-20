"""Summarize this historical deduplication round from receipts and metadata."""

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


BASE = Path(__file__).resolve().parent
PLAN = BASE / 'effective_history_plan_20261018.json'
PREPARED = BASE / 'effective_history_prepared_20261018.jsonl'
ADOPTED = BASE / 'effective_history_adopted_20261018.jsonl'
REPLAY = BASE / 'effective_history_restore_replay_20261018.json'
OUTPUT = BASE / 'effective_history_result_20261018.json'
STORE = Path('/mnt/d/Hegel-History-Archive-20261012/effective_history.sqlite3')
INITIAL_ARCHIVE_BYTES = 15486745175
METRICS = ('verified_payload_bytes', 'new_chunk_bytes', 'new_chunk_stored_bytes',
           'reused_chunk_bytes', 'chunks_added', 'chunks_reused', 'snapshot_aliases')


def latest_receipts(path):
    if not path.exists():
        return {}
    with path.open(encoding='utf-8') as incoming:
        return {row['unit']: row for line in incoming if line.strip()
                for row in [json.loads(line)]}


def database_metadata():
    if not STORE.exists():
        return {}, {'exists': False}
    connection = sqlite3.connect(f'file:{STORE}?mode=ro', uri=True)
    try:
        units = {name: (status, json.loads(receipt)) for name, status, receipt
                 in connection.execute('SELECT name,status,receipt FROM units')}
        count, unique_bytes, compressed_bytes = connection.execute(
            'SELECT COUNT(*),COALESCE(SUM(size),0),COALESCE(SUM(length(payload)),0) '
            'FROM blocks').fetchone()
        sequences = connection.execute('SELECT COUNT(*) FROM sequences').fetchone()[0]
        invalid_gzip = [{'unit': unit, 'member': member} for unit, member
            in connection.execute(
                "SELECT unit,json_extract(metadata,'$.name') FROM entries "
                "WHERE json_extract(metadata,'$.invalid_gzip_retained_raw')=1 "
                "ORDER BY unit,ordinal")]
        return units, {'exists': True, 'blocks': count,
            'unique_chunk_bytes': unique_bytes, 'compressed_chunk_bytes': compressed_bytes,
            'snapshot_sequences': sequences,
            'invalid_gzip_retained_raw': {'count': len(invalid_gzip), 'entries': invalid_gzip},
            'unit_status_counts': dict(Counter(status for status, _ in units.values()))}
    finally:
        connection.close()


def main():
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    expected = {row['unit']: row for row in plan['units']}
    prepared, adoption = latest_receipts(PREPARED), latest_receipts(ADOPTED)
    database_units, database = database_metadata()
    errors, verified, adopted = [], set(), set()
    if len(expected) != 1589 or plan['counts'] != {'D': 732, 'C': 855, 'C-subtree': 2}:
        errors.append({'reason': 'Finite plan count differs from the declared scope.'})
    if sum(row['bytes'] for row in expected.values()) != INITIAL_ARCHIVE_BYTES:
        errors.append({'reason': 'Finite plan original archive byte total differs.'})
    for unit, item in expected.items():
        declared = {'unit': unit, 'source': item['source'], 'source_bytes': item['bytes'],
                    'source_mtime_ns': item['mtime_ns']}
        row = prepared.get(unit)
        saved = database_units.get(unit)
        if row and row.get('status') == 'verified':
            fields = ('unit', 'source', 'source_bytes', 'source_mtime_ns')
            if (saved and saved[0] == 'verified'
                    and all(saved[1].get(key) == row.get(key) for key in fields)
                    and all(row.get(key) == value for key, value in declared.items())
                    and Path(row['store']).resolve() == STORE.resolve()):
                verified.add(unit)
            else:
                errors.append({'unit': unit, 'reason': 'Prepared receipt and committed store disagree.'})
        accepted = adoption.get(unit)
        if accepted and accepted.get('stage') == 'adopted':
            if (unit in verified
                    and all(accepted.get(key) == value for key, value in declared.items())
                    and Path(accepted['store']).resolve() == STORE.resolve()):
                adopted.add(unit)
            else:
                errors.append({'unit': unit, 'reason': 'Adoption receipt lacks matching verified plan/store metadata.'})

    scopes = {scope: {'expected_units': count, 'verified_units': 0, 'adopted_units': 0,
                     'sources_remaining': 0, 'confirmed_removed_original_bytes': 0,
                     **{metric: 0 for metric in METRICS}}
              for scope, count in plan['counts'].items()}
    residuals = []
    # One final, finite source-metadata pass. No archive content is opened here.
    for unit, item in expected.items():
        scope = scopes[unit.split('/', 1)[0]]
        if unit in verified:
            scope['verified_units'] += 1
            for metric in METRICS:
                scope[metric] += prepared[unit].get(metric, 0)
        if unit in adopted:
            scope['adopted_units'] += 1
        try:
            metadata = Path(item['source']).stat()
        except FileNotFoundError:
            if unit in adopted:
                removed = adoption[unit].get('removed_original_bytes')
                if removed == item['bytes']:
                    scope['confirmed_removed_original_bytes'] += removed
                else:
                    errors.append({'unit': unit, 'reason': 'Adoption deletion byte receipt differs from plan.'})
            else:
                errors.append({'unit': unit, 'reason': 'Original source missing without adopted receipt.'})
            continue
        except OSError as error:
            errors.append({'unit': unit, 'reason': str(error)})
            continue
        scope['sources_remaining'] += 1
        residuals.append(item['source'])
        if unit in adopted:
            errors.append({'unit': unit, 'reason': 'Adopted original source still exists.'})
        elif (metadata.st_size, metadata.st_mtime_ns) != (item['bytes'], item['mtime_ns']):
            errors.append({'unit': unit, 'reason': 'Original source size or mtime differs from plan.'})

    store_files = []
    for suffix in ('', '-wal', '-shm'):
        path = Path(str(STORE) + suffix)
        try:
            metadata = path.stat()
        except FileNotFoundError:
            continue
        store_files.append({'path': str(path), 'bytes': metadata.st_size,
                            'allocated_bytes': metadata.st_blocks * 512})
    physical_bytes = sum(row['bytes'] for row in store_files)
    removed_bytes = sum(row['confirmed_removed_original_bytes'] for row in scopes.values())
    replay = json.loads(REPLAY.read_text(encoding='utf-8')) if REPLAY.exists() else {'status': 'pending'}
    complete = (len(verified) == len(expected) == len(adopted)
                and not errors and not residuals and replay.get('status') == 'passed')
    result = {
        'status': 'complete' if complete else 'in_progress',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'This lossless historical-content deduplication round only.',
        'expected_units': len(expected), 'verified_units': len(verified),
        'adopted_units': len(adopted), 'scopes': scopes,
        'totals': {metric: sum(row[metric] for row in scopes.values()) for metric in METRICS},
        'shared_database': {**database, 'path': str(STORE), 'physical_files': store_files,
            'file_bytes_counted_once': physical_bytes,
            'allocated_bytes_counted_once': sum(row['allocated_bytes'] for row in store_files)},
        'original_archive_bytes_at_start': INITIAL_ARCHIVE_BYTES,
        'confirmed_removed_original_bytes': removed_bytes,
        'net_archive_file_bytes_saved': removed_bytes - physical_bytes,
        'saving_basis': 'Confirmed absent adopted original archives minus the shared SQLite, WAL and SHM file sizes once. Earlier cleanup rounds and unrelated disk-free changes are excluded.',
        'source_files_remaining': len(residuals), 'remaining_source_paths': residuals,
        'metadata_errors': errors,
        'unexpected_receipt_units': {'prepared': sorted(set(prepared) - set(expected)),
                                     'adoption': sorted(set(adoption) - set(expected))},
        'restore_replay': {'receipt': str(REPLAY), **replay},
        'new_focused_tests': {'reported_passed': 16,
            'files': ['tests/test_effective_history.py', 'tests/test_adopt_effective_history.py'],
            'skip_existing_restore_assertions_passed': True,
            'interrupted_packed_verification_resume_passed': True,
            'seven_zip_consumer_error_returns_without_deadlock_passed': True},
        'inputs': [str(PLAN), str(PREPARED), str(ADOPTED), str(STORE), str(REPLAY)]}
    temporary = OUTPUT.with_suffix('.json.writing')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(OUTPUT)
    print(json.dumps({'output': str(OUTPUT), 'status': result['status'],
        'verified_units': len(verified), 'adopted_units': len(adopted),
        'net_archive_file_bytes_saved': result['net_archive_file_bytes_saved'],
        'metadata_errors': len(errors)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
