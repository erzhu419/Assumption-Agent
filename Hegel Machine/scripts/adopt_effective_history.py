"""Publish verified historical-unit references, then remove their old containers."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hegel_machine.cognition.effective_history import HistoryStore

D_PLAN = Path('/mnt/d/Hegel-History-Archive-20261012/archive_plan.json')
C_INDEX = Path('/mnt/c/Users/erzhu/Hegel-History-Archive-20261012/original_paths.json')


def atomic_text(path, text):
    temporary = path.with_name(path.name + '.adopt-writing')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        shutil.copystat(path, temporary)
    temporary.replace(path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def event(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'time_utc': datetime.now(timezone.utc).isoformat(), **value},
                                ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def json_lines(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def update_readme(base, artifacts, store):
    path = base / 'README.md'
    text = path.read_text() if path.exists() else ''
    marker = '## Effective historical storage'
    if marker in text:
        return
    command = shlex.join(['python3', str(ROOT / 'scripts/compact_effective_history.py'),
        'restore', '--store', str(store),
        '--unit', 'UNIT', '--destination', str(artifacts)])
    text += '\n' + marker + '\n\n'
    text += ('Current archive and unit references are recorded in each source ARCHIVE_POINTER/state '
        'or original_paths.json entry. Migrated units share effective_history.sqlite3: unchanged '
        'content is stored once, while exact versions retain their block references. The complete '
        'shared store is required for restoration. Earlier archive paths in operation logs remain '
        'historical provenance. Unmigrated gzip/tar units retain their earlier restore commands.\n\n'
        'Restore the selected unit to its original paths using its recorded command; UNIT is '
        '`D/EXPERIMENT`, `C/ORIGINAL_RELATIVE_PATH` or `C-subtree/ORIGINAL_RELATIVE_DIRECTORY`. '
        'Prefer `--member ORIGINAL_RELATIVE_PATH` for an individual checkpoint. Choose a destination '
        'with enough space when restoring a whole unit. Existing local evidence is preserved.\n\n'
        '```bash\n' + command + '\n```\n')
    atomic_text(path, text)


def adopt(prepared_receipt, manifest, *, apply=False, only=None, d_plan=D_PLAN, c_index=C_INDEX):
    d_plan, c_index, manifest = Path(d_plan), Path(c_index), Path(manifest)
    plan = json.loads(d_plan.read_text())
    index = json.loads(c_index.read_text())
    artifacts = Path(plan['root'])
    d_units = {row['name']: row for row in plan['units']}
    c_files = {str(Path(row['source']).relative_to(artifacts)): row for row in index['files']}
    c_subtrees = {str(Path(row['source']).relative_to(artifacts)): row
                  for row in index.get('subtree_archives', [])}
    previous = {row['unit']: row for row in json_lines(manifest)} if manifest.exists() else {}
    stores, results = {}, []
    try:
        for receipt in json_lines(Path(prepared_receipt)):
            unit = receipt['unit']
            if (receipt['status'] != 'verified' or (only and unit not in only)):
                continue
            source, store_path = Path(receipt['source']), Path(receipt['store']).resolve()
            store = stores.setdefault(str(store_path), None)
            if store is None:
                store = stores[str(store_path)] = HistoryStore(store_path, readonly=True)
            saved = store.receipt(unit)
            if (not saved or saved[0] != 'verified'
                    or saved[1]['source'] != str(source)
                    or saved[1]['source_bytes'] != receipt['source_bytes']
                    or saved[1]['source_mtime_ns'] != receipt['source_mtime_ns']):
                raise ValueError('Store has no matching verified unit: ' + unit)
            past = previous.get(unit)
            if past and past['stage'] == 'adopted':
                results.append(past)
                continue
            if source.exists():
                stat = source.stat()
                if (stat.st_size, stat.st_mtime_ns) != (receipt['source_bytes'], receipt['source_mtime_ns']):
                    raise ValueError('Original container changed since verification: ' + str(source))
            elif not past or past['stage'] != 'references_updated':
                raise ValueError('Original container is missing before references were recorded: ' + str(source))
            restore = shlex.join(['python3', str(ROOT / 'scripts/compact_effective_history.py'),
                'restore', '--store', str(store_path), '--unit', unit, '--destination', str(artifacts)])
            reference = {'archive': str(store_path), 'archive_unit': unit,
                'archive_format': 'effective_history_sqlite', 'restore': restore,
                'archive_bytes': receipt['new_chunk_stored_bytes'], 'archive_shared_store': True}
            updates = []
            if unit.startswith('D/'):
                name = unit[2:]
                row = d_units[name]
                pointer = artifacts / name / 'ARCHIVE_POINTER.json' if row['directory'] else artifacts / (name + '.ARCHIVE_POINTER.json')
                state_path = d_plan.parent / 'states' / (name + '.json')
                pointer_data, state = json.loads(pointer.read_text()), json.loads(state_path.read_text())
                if state['status'] != 'completed':
                    raise ValueError('Historical unit is not completed: ' + unit)
                for value in [pointer_data.get('archive'), state.get('archive'), row.get('archive')]:
                    if value is not None and value not in {str(source), str(store_path)}:
                        raise ValueError('Archive reference changed unexpectedly: ' + unit)
                pointer_data.update(reference)
                state.update(reference)
                row.update(reference)
                updates = [(pointer, pointer_data), (state_path, state), (d_plan, plan)]
                readme_base = d_plan.parent
            elif unit.startswith('C-subtree/'):
                relative = unit[len('C-subtree/'):]
                row = c_subtrees[relative]
                if row['archive'] not in {str(source), str(store_path)}:
                    raise ValueError('Subtree reference changed unexpectedly: ' + unit)
                row.update(reference)
                updates = [(c_index, index)]
                readme_base = c_index.parent
            elif unit.startswith('C/'):
                relative = unit[2:]
                row = c_files[relative]
                expected = Path(index['destination']) / (relative + '.gz')
                if source != expected or row.get('archive', str(source)) not in {str(source), str(store_path)}:
                    raise ValueError('Individual gzip reference changed unexpectedly: ' + unit)
                row.update(reference, member=relative)
                updates = [(c_index, index)]
                readme_base = c_index.parent
            else:
                raise ValueError('Unit is outside the declared C/D scope: ' + unit)
            data = {'unit': unit, 'source': str(source), 'store': str(store_path),
                'source_bytes': receipt['source_bytes'], 'source_mtime_ns': receipt['source_mtime_ns'],
                'verified_payload_bytes': receipt['verified_payload_bytes'],
                'unit_new_stored_bytes': receipt['new_chunk_stored_bytes'],
                'restore': restore, 'metadata': [str(path) for path, _ in updates]}
            if not apply:
                results.append({'stage': 'ready_to_adopt', **data})
                continue
            update_readme(readme_base, artifacts, store_path)
            for path, value in updates:
                atomic_json(path, value)
            event(manifest, {'stage': 'references_updated', **data})
            if source.exists():
                source.unlink()
            adopted = {'stage': 'adopted', **data,
                       'removed_original_bytes': receipt['source_bytes']}
            event(manifest, adopted)
            previous[unit] = adopted
            results.append(adopted)
            print(json.dumps(adopted, ensure_ascii=False), flush=True)
    finally:
        for store in stores.values():
            if store is not None:
                store.close()
    shared = []
    for name in stores:
        files = [p for suffix in ['', '-wal', '-shm'] if (p := Path(name + suffix)).is_file()]
        shared.append({'store': name, 'physical_bytes': sum(p.stat().st_blocks * 512 for p in files),
                       'file_bytes': sum(p.stat().st_size for p in files)})
    return {'apply': apply, 'units': len(results),
            'removed_original_bytes': sum(r.get('removed_original_bytes', 0) for r in results),
            'shared_stores': shared, 'results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-receipt', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--only', action='append')
    args = parser.parse_args()
    result = adopt(args.prepared_receipt, args.manifest, apply=args.apply, only=args.only)
    print(json.dumps({key: value for key, value in result.items() if key != 'results'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
