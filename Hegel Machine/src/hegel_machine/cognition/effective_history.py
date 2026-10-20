"""Lossless historical snapshots: shared content blocks and per-file references.

Digests index block reuse, avoiding repeated reads of already stored payloads.
Every original payload is compared with committed-store reconstruction before
the separate adopter can delete its container.
"""

from collections import OrderedDict
from contextlib import contextmanager
import base64
import ctypes
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import sqlite3
import struct
import subprocess
import tarfile
import tempfile
import zlib


_chunk_library = None


def content_chunks(stream):
    global _chunk_library
    if _chunk_library is None:
        source = Path(__file__).resolve().parents[3] / 'scripts/effective_history_chunks.c'
        library = Path(tempfile.gettempdir()) / 'hegel-effective-history-chunks.so'
        if not library.exists() or library.stat().st_mtime_ns < source.stat().st_mtime_ns:
            output = library.with_name(library.name + f'.{os.getpid()}')
            subprocess.run(['gcc', '-O3', '-shared', '-fPIC', str(source), '-o', str(output)], check=True)
            output.replace(library)
        _chunk_library = ctypes.CDLL(str(library))
        _chunk_library.history_chunks_init()
        _chunk_library.history_chunk_end.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        _chunk_library.history_chunk_end.restype = ctypes.c_size_t
    pending = b''
    while True:
        incoming = stream.read(4 * 1024 * 1024)
        pending += incoming
        offset = 0
        pointer = ctypes.cast(ctypes.c_char_p(pending), ctypes.c_void_p).value
        while len(pending) - offset >= 1024:
            end = _chunk_library.history_chunk_end(pointer + offset, len(pending) - offset)
            if not end:
                break
            yield pending[offset:offset + end]
            offset += end
        pending = pending[offset:]
        if not incoming:
            if pending:
                yield pending
            return


class HistoryStore:
    def __init__(self, path, *, readonly=False):
        self.path = Path(path).resolve()
        self.connection = sqlite3.connect(
            f'file:{self.path}?mode=ro' if readonly else str(self.path), uri=readonly)
        self.connection.execute('PRAGMA cache_size=-262144')
        self.cache = OrderedDict()
        self.read_cache = OrderedDict()
        self.read_cache_bytes = 0
        self.stats = {}
        self.progress = None
        if not readonly:
            self.connection.executescript('''
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=FULL;
                CREATE TABLE IF NOT EXISTS blocks (
                    id INTEGER PRIMARY KEY, digest BLOB UNIQUE NOT NULL,
                    size INTEGER NOT NULL, payload BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS sequences (
                    id INTEGER PRIMARY KEY, digest BLOB UNIQUE NOT NULL,
                    payload BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS units (
                    name TEXT PRIMARY KEY, status TEXT NOT NULL, receipt TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS entries (
                    unit TEXT NOT NULL, ordinal INTEGER NOT NULL,
                    metadata TEXT NOT NULL, sequence INTEGER,
                    PRIMARY KEY (unit, ordinal));
            ''')

    def close(self):
        self.connection.close()

    def _remember(self, key, identifier):
        self.cache[key] = identifier
        self.cache.move_to_end(key)
        while len(self.cache) > 100000:
            self.cache.popitem(last=False)

    def block(self, data):
        digest = hashlib.sha256(data).digest()
        cached = self.cache.get(digest)
        if cached is not None:
            identifier = cached
            self.cache.move_to_end(digest)
        else:
            row = self.connection.execute(
                'SELECT id FROM blocks WHERE digest=?', (digest,)).fetchone()
            if row:
                identifier = row[0]
            else:
                payload = zlib.compress(data, 3)
                identifier = self.connection.execute(
                    'INSERT INTO blocks(digest,size,payload) VALUES(?,?,?)',
                    (digest, len(data), payload)).lastrowid
                self.stats['new_chunk_bytes'] += len(data)
                self.stats['new_chunk_stored_bytes'] += len(payload)
                self.stats['chunks_added'] += 1
                self._remember(digest, identifier)
                return identifier
            self._remember(digest, identifier)
        self.stats['reused_chunk_bytes'] += len(data)
        self.stats['chunks_reused'] += 1
        return identifier

    def save_stream(self, stream):
        identifiers = []
        for data in content_chunks(stream):
            identifiers.append(self.block(data))
            processed = self.stats['new_chunk_bytes'] + self.stats['reused_chunk_bytes']
            if self.progress and processed >= self.next_pack_report:
                self.progress({'unit': self.unit, 'stage': 'packing',
                               'processed_payload_bytes': processed, **self.stats})
                self.next_pack_report = processed + 256 * 1024 * 1024
        raw = struct.pack(f'<{len(identifiers)}Q', *identifiers)
        digest = hashlib.sha256(raw).digest()
        row = self.connection.execute(
            'SELECT id,payload FROM sequences WHERE digest=?', (digest,)).fetchone()
        if row:
            if zlib.decompress(row[1]) != raw:
                raise ValueError('Snapshot reference index differs')
            self.stats['snapshot_aliases'] += 1
            return row[0]
        return self.connection.execute('INSERT INTO sequences(digest,payload) VALUES(?,?)',
                                       (digest, zlib.compress(raw, 3))).lastrowid

    def read_sequence(self, identifier):
        row = self.connection.execute('SELECT payload FROM sequences WHERE id=?',
                                      (identifier,)).fetchone()
        if row is None:
            raise ValueError(f'Missing snapshot sequence {identifier}')
        raw = zlib.decompress(row[0])
        for (block,) in struct.iter_unpack('<Q', raw):
            data = self.read_cache.get(block)
            if data is not None:
                self.read_cache.move_to_end(block)
            else:
                row = self.connection.execute('SELECT payload FROM blocks WHERE id=?', (block,)).fetchone()
                if row is None:
                    raise ValueError(f'Missing snapshot block {block}')
                data = zlib.decompress(row[0])
                self.read_cache[block] = data
                self.read_cache_bytes += len(data)
                while self.read_cache_bytes > 64 * 1024 * 1024:
                    _, old = self.read_cache.popitem(last=False)
                    self.read_cache_bytes -= len(old)
            yield data

    def receipt(self, unit):
        row = self.connection.execute('SELECT status,receipt FROM units WHERE name=?', (unit,)).fetchone()
        return (row[0], json.loads(row[1])) if row else None


@contextmanager
def source_tar(path):
    path = Path(path)
    if path.suffix == '.7z':
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(['7z', 'e', '-so', str(path), 'archive.tar'],
                                       stdout=subprocess.PIPE, stderr=errors)
            try:
                with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
                    yield archive
                # Consume tar padding before waiting on the producer.
                while process.stdout.read(1024 * 1024):
                    pass
                process.stdout.close()
                if process.wait():
                    errors.seek(0)
                    raise ValueError('Archive decoder failed: ' + errors.read().decode(errors='replace')[-1000:])
            finally:
                # A consumer error can leave 7z blocked on a full stdout pipe.
                # Close that pipe before waiting for the decoder to stop.
                if not process.stdout.closed:
                    process.stdout.close()
                if process.poll() is None:
                    process.terminate()
                    process.wait()
    else:
        with tarfile.open(path, mode='r|gz' if path.suffix == '.gz' else 'r|') as archive:
            yield archive


def tar_metadata(member):
    result = member.get_info()
    result['type'] = member.type.decode('ascii')
    result['pax_headers'] = member.pax_headers
    return result


def gzip_header(stream):
    stream.seek(0)
    prefix = stream.read(10)
    if len(prefix) != 10 or prefix[:3] != b'\x1f\x8b\x08':
        raise gzip.BadGzipFile('Not a complete gzip header')
    flags = prefix[3]
    if flags & 4:
        length = stream.read(2)
        prefix += length + stream.read(struct.unpack('<H', length)[0])
    for flag in (8, 16):
        if flags & flag:
            while True:
                value = stream.read(1)
                if not value:
                    raise gzip.BadGzipFile('Truncated gzip header')
                prefix += value
                if value == b'\0':
                    break
    if flags & 2:
        prefix += stream.read(2)
    return base64.b64encode(prefix).decode('ascii')


def save_file(store, stream, metadata, work_root, *, gzip_input=False, restore_gzip=False):
    metadata = dict(metadata)
    metadata['encoding'] = 'raw'
    if not gzip_input:
        return metadata, store.save_stream(stream)
    with tempfile.TemporaryFile(dir=work_root) as original:
        while data := stream.read(1024 * 1024):
            original.write(data)
        original.seek(0)
        store.connection.execute('SAVEPOINT gzip_member')
        before = store.stats.copy()
        try:
            header = gzip_header(original)
            original.seek(0)
            with gzip.GzipFile(fileobj=original) as decoded:
                sequence = store.save_stream(decoded)
            metadata.update(encoding='gzip' if restore_gzip else 'raw', source_gzip_header=header)
            store.connection.execute('RELEASE gzip_member')
            return metadata, sequence
        except (gzip.BadGzipFile, EOFError, zlib.error, struct.error):
            # Interrupted .gz.tmp records retain their original incomplete bytes.
            store.connection.execute('ROLLBACK TO gzip_member')
            store.connection.execute('RELEASE gzip_member')
            store.cache.clear()
            store.stats = before
            original.seek(0)
            metadata['invalid_gzip_retained_raw'] = True
            return metadata, store.save_stream(original)


def verify_stream(store, sequence, original):
    size = 0
    for data in store.read_sequence(sequence):
        parts = bytearray()
        while len(parts) < len(data):
            value = original.read(len(data) - len(parts))
            if not value:
                break
            parts.extend(value)
        if parts != data:
            raise ValueError(f'Restored snapshot differs at payload offset {size}')
        size += len(data)
        if store.progress:
            store.verified_total += len(data)
            if store.verified_total >= store.next_verify_report:
                store.progress({'unit': store.unit, 'stage': 'verifying',
                                'verified_payload_bytes': store.verified_total})
                store.next_verify_report = store.verified_total + 256 * 1024 * 1024
    if original.read(1):
        raise ValueError('Restored snapshot omits trailing original data')
    return size


def _verify_file(store, sequence, stream, metadata, work_root):
    if not metadata.get('source_gzip_header'):
        return verify_stream(store, sequence, stream)
    with tempfile.TemporaryFile(dir=work_root) as original:
        while data := stream.read(1024 * 1024):
            original.write(data)
        if gzip_header(original) != metadata['source_gzip_header']:
            raise ValueError('Original gzip metadata changed')
        original.seek(0)
        with gzip.GzipFile(fileobj=original) as decoded:
            return verify_stream(store, sequence, decoded)


def compact_unit(store, item, work_root, *, progress=None):
    source = Path(item['source'])
    unit = item['unit']
    previous = store.receipt(unit)
    if previous and previous[0] == 'verified':
        return previous[1]
    if previous and previous[0] == 'packed':
        return verify_unit(store, previous[1], work_root, progress=progress)
    initial = source.stat()
    # Packing needs digest IDs, not decoded blocks left by the previous replay.
    store.read_cache.clear()
    store.read_cache_bytes = 0
    store.stats = {key: 0 for key in ('new_chunk_bytes', 'new_chunk_stored_bytes',
        'reused_chunk_bytes', 'chunks_added', 'chunks_reused', 'snapshot_aliases')}
    store.progress, store.unit = progress, unit
    store.next_pack_report = store.next_verify_report = 256 * 1024 * 1024
    store.verified_total = 0
    store.connection.execute('BEGIN')
    store.connection.execute('DELETE FROM entries WHERE unit=?', (unit,))
    count = 0
    try:
        if item['kind'] == 'tar':
            with source_tar(source) as archive:
                for member in archive:
                    metadata, sequence = tar_metadata(member), None
                    if member.isfile():
                        with archive.extractfile(member) as payload:
                            metadata, sequence = save_file(store, payload, metadata, work_root,
                                gzip_input=member.name.endswith(('.gz', '.gz.tmp')),
                                restore_gzip=True)
                    store.connection.execute('INSERT INTO entries VALUES(?,?,?,?)',
                        (unit, count, json.dumps(metadata), sequence))
                    count += 1
                    if progress and count % 500 == 0:
                        progress({'unit': unit, 'stage': 'packing', 'members': count,
                                  **store.stats})
        else:
            metadata = {'name': item['member'], 'mode': initial.st_mode & 0o7777,
                        'mtime': initial.st_mtime, 'type': '0', 'size': initial.st_size,
                        'uid': initial.st_uid, 'gid': initial.st_gid, 'linkname': '',
                        'uname': '', 'gname': '', 'pax_headers': {}}
            with source.open('rb') as payload:
                metadata, sequence = save_file(store, payload, metadata, work_root, gzip_input=True)
            store.connection.execute('INSERT INTO entries VALUES(?,?,?,?)',
                                      (unit, 0, json.dumps(metadata), sequence))
            count = 1
        receipt = {**item, 'store': str(store.path), 'source_bytes': initial.st_size,
            'source_mtime_ns': initial.st_mtime_ns, 'member_count': count,
            'status': 'packed', **store.stats}
        store.connection.execute('INSERT OR REPLACE INTO units VALUES(?,?,?)',
                                  (unit, 'packed', json.dumps(receipt)))
        store.connection.commit()
    except BaseException:
        store.connection.rollback()
        store.cache.clear()
        raise
    return verify_unit(store, receipt, work_root, progress=progress)


def verify_unit(store, receipt, work_root, *, progress=None):
    source, unit = Path(receipt['source']), receipt['unit']
    count = receipt['member_count']
    initial = source.stat()
    if (initial.st_size, initial.st_mtime_ns) != (receipt['source_bytes'], receipt['source_mtime_ns']):
        raise ValueError('Source changed since committed snapshot packing')
    store.progress, store.unit = progress, unit
    store.verified_total = 0
    store.next_verify_report = 256 * 1024 * 1024
    # Read back the committed store against every original payload and tar record.
    store.read_cache.clear()
    store.read_cache_bytes = 0
    if progress:
        progress({'unit': unit, 'stage': 'verifying', 'members': count})
    verified = 0
    rows = iter(store.connection.execute(
        'SELECT metadata,sequence FROM entries WHERE unit=? ORDER BY ordinal', (unit,)))
    if receipt['kind'] == 'tar':
        with source_tar(source) as archive:
            for member in archive:
                metadata, sequence = next(rows)
                metadata = json.loads(metadata)
                if any(metadata.get(key) != value for key, value in tar_metadata(member).items()):
                    raise ValueError('Restored tar metadata differs: ' + member.name)
                if member.isfile():
                    with archive.extractfile(member) as payload:
                        verified += _verify_file(store, sequence, payload, metadata, work_root)
    else:
        metadata, sequence = next(rows)
        with source.open('rb') as payload:
            verified += _verify_file(store, sequence, payload, json.loads(metadata), work_root)
    if next(rows, None) is not None:
        raise ValueError('Store has extra archive members')
    final = source.stat()
    if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
        raise ValueError('Source changed during historical conversion')
    receipt.update(status='verified', verified_payload_bytes=verified,
        restore=(f'python3 {shlex.quote(str(Path(__file__).resolve().parents[3] / "scripts/compact_effective_history.py"))}'
                 f' restore --store {shlex.quote(str(store.path))} --unit {shlex.quote(unit)} --destination DESTINATION'))
    store.connection.execute('UPDATE units SET status=?,receipt=? WHERE name=?',
                              ('verified', json.dumps(receipt), unit))
    store.connection.commit()
    store.connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    return receipt


def restore_unit(store, unit, destination, member=None):
    saved = store.receipt(unit)
    if not saved or saved[0] != 'verified':
        raise ValueError('Restore requires a fully verified historical unit')
    destination = Path(destination)
    directories = []
    restored = 0
    matched = 0
    for metadata, sequence in store.connection.execute(
            'SELECT metadata,sequence FROM entries WHERE unit=? ORDER BY ordinal', (unit,)):
        metadata = json.loads(metadata)
        if member is not None and metadata['name'] != member:
            continue
        matched += 1
        path = destination / metadata['name']
        # Match the old archive's --skip-old-files recovery contract.
        if path.exists() or path.is_symlink():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if sequence is not None:
            with path.open('wb') as output:
                chunks = store.read_sequence(sequence)
                if metadata['encoding'] == 'gzip':
                    output.write(base64.b64decode(metadata['source_gzip_header']))
                    compressor = zlib.compressobj(6, zlib.DEFLATED, -15)
                    crc = length = 0
                    for data in chunks:
                        crc = zlib.crc32(data, crc)
                        length += len(data)
                        output.write(compressor.compress(data))
                    output.write(compressor.flush())
                    output.write(struct.pack('<II', crc, length & 0xffffffff))
                else:
                    for data in chunks:
                        output.write(data)
        elif metadata['type'] == '5':
            path.mkdir(exist_ok=True)
            directories.append((path, metadata))
            restored += 1
            continue
        elif metadata['type'] == '2':
            path.symlink_to(metadata['linkname'])
        elif metadata['type'] == '1':
            os.link(destination / metadata['linkname'], path)
        else:
            raise ValueError('Unsupported historical tar member: ' + metadata['type'])
        if not path.is_symlink():
            os.chmod(path, metadata['mode'])
            os.utime(path, (metadata['mtime'], metadata['mtime']))
        restored += 1
    for path, metadata in reversed(directories):
        os.chmod(path, metadata['mode'])
        os.utime(path, (metadata['mtime'], metadata['mtime']))
    if member is not None and matched == 0:
        raise ValueError('Historical member was not found: ' + member)
    return restored
