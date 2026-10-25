"""Publish verified historical-unit references, then remove their old containers."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
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



def _snapshot_ranges(source, destination):
    """Compare only the finite known pair, without a new hash index."""
    sizes = [path.stat().st_size for path in (source, destination)]
    ranges, different = [], 0
    aligned = min(sizes) // 4096 * 4096
    with source.open('rb') as first, destination.open('rb') as second:
        position = 0
        while position < aligned:
            length = min(1024 * 1024, aligned - position)
            left, right = first.read(length), second.read(length)
            pieces = [(0, length)] if left == right else [
                (offset, 4096) for offset in range(0, length, 4096)
                if left[offset:offset + 4096] == right[offset:offset + 4096]]
            equal = sum(size for _, size in pieces)
            different += length - equal
            for offset, size in pieces:
                start = position + offset
                if ranges and ranges[-1][0] + ranges[-1][1] == start:
                    ranges[-1][1] += size
                else:
                    ranges.append([start, size])
            position += length
        tail_size = min(sizes) - aligned
        first_tail, second_tail = first.read(tail_size), second.read(tail_size)
    fully_equal = sizes[0] == sizes[1] and different == 0 and first_tail == second_tail
    return {'source_bytes': sizes[0], 'destination_bytes': sizes[1],
            'compared_aligned_bytes': aligned, 'different_aligned_bytes': different,
            'fully_equal': fully_equal, 'equal_ranges': ranges,
            'unaligned_tail_bytes': [size - aligned for size in sizes]}


def _dedupe_range(source, destination, offset, length):
    """FIDEDUPERANGE compares bytes in the kernel before sharing writable extents."""
    attempts, position = [], offset
    with source.open('rb') as original, destination.open('r+b') as target:
        while position < offset + length:
            requested = min(16 * 1024 * 1024, offset + length - position)
            # linux/fs.h: file_dedupe_range header plus one destination info.
            request = bytearray(struct.pack('=QQHHIqQQiI', position, requested,
                1, 0, 0, target.fileno(), position, 0, 0, 0))
            attempt = {'operation': 'FIDEDUPERANGE', 'source_offset': position,
                       'destination_offset': position, 'requested_bytes': requested}
            try:
                fcntl.ioctl(original.fileno(), 0xC0189436, request, True)
                values = struct.unpack('=QQHHIqQQiI', request)
                shared, status = values[7], values[8]
                attempt.update(status=status, bytes_deduped=shared)
            except OSError as error:
                attempt.update(status='error', errno=error.errno, error=str(error), bytes_deduped=0)
                attempts.append(attempt)
                break
            attempts.append(attempt)
            if status != 0 or shared == 0:
                break
            position += shared
    return attempts


def _finished_snapshot(path):
    try:
        with path.open(encoding='utf-8') as stream:
            value = json.load(stream)
    except json.JSONDecodeError:
        return 'unreadable_incomplete_json', False
    status = value.get('status') if isinstance(value, dict) else None
    terminal = {'completed', 'source_repair_inconclusive', 'inconclusive',
        'provider_failed', 'token_usage_unavailable', 'model_call_budget_exhausted',
        'confirmation_budget_exhausted', 'python_execution_budget_exhausted',
        'token_budget_exhausted', 'budget_exhausted', 'stopped_failure_preserved',
        'method_step_limit'}
    return status, status in terminal



def _canonicalize_retained_host_pair(source, destination, store_path, manifest, *, apply):
    """Retain exact old versions before aligning equivalent host JSON fields."""
    from hegel_machine.cognition.effective_history import compact_unit
    from scripts.run_online_inquiry_llm import write_json

    # These explicitly selected historical roots may contain interrupted runs.
    # Preserve their pending status and exact recovery bytes rather than treating
    # an unfinished scientific answer as a reason to retain duplicate payloads.
    status, _ = _finished_snapshot(source)
    try:
        with source.open(encoding='utf-8') as stream:
            source_value = json.load(stream)
        with destination.open(encoding='utf-8') as stream:
            destination_value = json.load(stream)
    except json.JSONDecodeError:
        return {'action': 'retained_incomplete_checkpoint', 'original_status': status}
    if not isinstance(source_value, dict) or not isinstance(destination_value, dict):
        return {'action': 'retained_non_object_checkpoint', 'original_status': status}
    versions = []
    for path in (source, destination):
        stat = path.stat()
        versions.append({'unit': 'retained-files/' + str(path).lstrip('/') + '/' + str(stat.st_mtime_ns),
            'source': str(path), 'kind': 'file', 'member': str(path).lstrip('/'),
            'original_metadata': {'mode': stat.st_mode & 0o7777,
                'mtime_ns': stat.st_mtime_ns, 'bytes': stat.st_size}})
    result = {'action': 'ready_to_canonicalize_retained_versions',
              'original_status': status, 'original_versions': versions, 'store': str(store_path)}
    if not apply:
        return result
    # Staging on the same filesystem makes independent copies share extents.
    with tempfile.TemporaryDirectory(prefix='.retained-host-', dir=source.parent) as temporary:
        store = HistoryStore(store_path)
        try:
            for version in versions:
                saved = compact_unit(store, version, temporary)
                version.update(status=saved['status'], verified_payload_bytes=saved['verified_payload_bytes'],
                               restore=saved['restore'], store=str(store.path))
        finally:
            store.close()
        # A durable recovery reference precedes either original being replaced.
        event(manifest, {'stage': 'retained_host_originals_verified', **result})
        before = os.statvfs(source.parent)
        temporary = Path(temporary)
        first, second = temporary / source.name, temporary / destination.name
        write_json(first, source_value, base=source)
        write_json(second, destination_value, base=first)
        for path, expected, original in ((first, source_value, versions[0]),
                                         (second, destination_value, versions[1])):
            with path.open(encoding='utf-8') as stream:
                if json.load(stream) != expected:
                    raise ValueError('Canonical host checkpoint changes scientific values: ' + str(path))
            path.chmod(original['original_metadata']['mode'])
        result['canonical_comparison'] = _snapshot_ranges(first, second)
        first.replace(source)
        second.replace(destination)
        after = os.statvfs(source.parent)
        result['canonical_rewrite_space'] = {
            'available_bytes_before': before.f_bavail * before.f_frsize,
            'available_bytes_after': after.f_bavail * after.f_frsize,
            'observed_net_available_increase_bytes':
                after.f_bavail * after.f_frsize - before.f_bavail * before.f_frsize}
    result.update(action='canonicalized_preserving_original_versions',
                  canonical_json_values_verified_before_replace=True,
                  exact_original_bytes_verified_from_committed_store=True,
                  writable_files_independent=True)
    return result


def adopt_retained_snapshots(roots, manifest, *, apply=False, exclude=None, retained_store=None):
    """Share equal ranges in named old trees; retain every different scientific byte."""
    from scripts.run_online_inquiry_llm import final_checkpoint_alias

    roots = [Path(root).resolve() for root in roots]
    excluded = [Path(root).resolve() for root in exclude or []]
    manifest = Path(manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    filesystems, results = {}, []
    pairs = []
    for root in roots:
        if not root.is_dir():
            raise ValueError('Retained snapshots root is not a directory: ' + str(root))
        if any(root == path or root.is_relative_to(path) for path in excluded):
            continue
        device = root.stat().st_dev
        if device not in filesystems:
            space = os.statvfs(root)
            filesystems[device] = {'sample_root': str(root),
                'available_bytes_before': space.f_bavail * space.f_frsize}
        for directory, children, filenames in os.walk(root, followlinks=False):
            directory = Path(directory)
            children[:] = sorted(name for name in children
                if not (directory / name).is_symlink() and not any(
                    directory / name == path or (directory / name).is_relative_to(path)
                    for path in excluded))
            def regular(name):
                path = directory / name
                return (name in filenames and not path.is_symlink() and path.is_file()
                    and not any(path == item or path.is_relative_to(item) for item in excluded))
            if regular('observations.json') and regular('model_observations.json'):
                pairs.append(('fitting_views', directory / 'observations.json',
                              directory / 'model_observations.json'))
            host_checkpoint = not any(part in {'_evaluated_model', '_submitted_models',
                '_observable_models', '_experiment_models', '_target_work'} for part in directory.parts)
            if regular('active.json') and host_checkpoint and directory.name != 'data':
                final = next((name for name in ('training.json', 'result.json') if regular(name)), None)
                if final:
                    pairs.append(('final_checkpoint', directory / final, directory / 'active.json'))
    seen = set()
    for kind, source, destination in pairs:
        key = (str(source), str(destination))
        if key in seen:
            continue
        seen.add(key)
        comparison = _snapshot_ranges(source, destination)
        record = {'kind': kind, 'source': str(source), 'destination': str(destination),
                  'comparison': comparison, 'all_paths_retained': True, 'dedupe': []}
        if kind == 'final_checkpoint' and not comparison['fully_equal'] and retained_store:
            record.update(_canonicalize_retained_host_pair(source, destination, retained_store,
                                                           manifest, apply=apply))
        elif kind == 'final_checkpoint' and comparison['fully_equal']:
            status, finished = _finished_snapshot(source)
            record['original_status'] = status
            if finished:
                record['action'] = 'aliased_identical' if apply else 'ready_to_alias_identical'
                if apply:
                    final_checkpoint_alias(source, destination)
                    if not _snapshot_ranges(source, destination)['fully_equal']:
                        raise ValueError('Aliased checkpoint does not restore its original: ' + str(destination))
                    record['original_bytes_restored'] = True
            else:
                record['action'] = 'retained_unfinished_checkpoint'
        else:
            record['action'] = ('shared_equal_ranges' if apply else 'ready_to_share_equal_ranges') \
                if comparison['equal_ranges'] else 'retained_unique_content'
        if record['action'] not in {'aliased_identical', 'canonicalized_preserving_original_versions'} and apply:
            if source.stat().st_dev == destination.stat().st_dev:
                for offset, length in comparison['equal_ranges']:
                    record['dedupe'].extend(_dedupe_range(source, destination, offset, length))
            else:
                record['action'] = 'retained_different_filesystems'
        record['confirmed_deduped_bytes'] = sum(item['bytes_deduped'] for item in record['dedupe'])
        record['different_content_retained'] = not comparison['fully_equal']
        results.append(record)
        event(manifest, {'stage': 'retained_snapshot_pair', **record})
    for item in filesystems.values():
        space = os.statvfs(item['sample_root'])
        item['available_bytes_after'] = space.f_bavail * space.f_frsize
        item['observed_net_available_increase_bytes'] = (
            item['available_bytes_after'] - item['available_bytes_before'])
    failures = [item for result in results for item in result['dedupe'] if item['status'] != 0]
    result = {'apply': apply, 'status': 'completed_with_retained_dedupe_failures' if failures else 'completed',
        'roots': [str(root) for root in roots], 'excluded_roots': [str(root) for root in excluded],
        'retained_store': str(retained_store) if retained_store else None,
        'canonicalized_host_pairs': sum(row['action'] == 'canonicalized_preserving_original_versions'
                                       for row in results),
        'pairs': len(results), 'aliases': sum(row['action'] == 'aliased_identical' for row in results),
        'confirmed_deduped_bytes': sum(row['confirmed_deduped_bytes'] for row in results),
        'filesystems': list(filesystems.values()),
        'physical_saving_basis': 'Observed statvfs net change, including concurrent filesystem activity; '
            'deduped bytes may already have shared extents and are not claimed as newly freed bytes.',
        'results': results}
    event(manifest, {'stage': 'retained_snapshots_summary', **{key: value for key, value in result.items()
                                                           if key != 'results'}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-receipt')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--only', action='append')
    parser.add_argument('--retained-snapshots-root', action='append')
    parser.add_argument('--exclude-retained-snapshots-root', action='append')
    parser.add_argument('--retained-store')
    args = parser.parse_args()
    if args.retained_snapshots_root:
        result = adopt_retained_snapshots(args.retained_snapshots_root, args.manifest,
            apply=args.apply, exclude=args.exclude_retained_snapshots_root, retained_store=args.retained_store)
    else:
        if not args.prepared_receipt:
            parser.error('--prepared-receipt is required for cold archive adoption')
        result = adopt(args.prepared_receipt, args.manifest, apply=args.apply, only=args.only)
    print(json.dumps({key: value for key, value in result.items() if key != 'results'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
