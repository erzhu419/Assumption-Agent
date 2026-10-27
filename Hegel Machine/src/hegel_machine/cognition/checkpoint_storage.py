"""Ordinary files with independent writes and shared unchanged disk blocks."""

import errno
import fcntl
import json
from math import isfinite, isnan
import mmap
import os
from pathlib import Path
import re
import shutil
import struct
from collections import OrderedDict
from contextlib import ExitStack


_checkpoint_sources = {}
_JSON_SHARED_MINIMUM = 64 * 1024
_JSON_ENCODED_COMPARE_MAXIMUM = 1024 * 1024
_JSON_DECODE_CHUNK = 8 * 1024 * 1024
_JSON_ENCODE_CHUNK = 64 * 1024
_JSON_MEMBER_CACHE_ENTRIES = 16 * 1024
_JSON_LAYOUT_TOKEN = re.compile(rb'"[^"\\]*(?:\\.[^"\\]*)*"|[{}\[\]]|"')
_JSON_CONTAINER_BOUNDARY = re.compile(
    rb'(?:"[^"\\]*(?:\\.[^"\\]*)*"|[^"{}\[\]]+)*([{}\[\]]|")')
_JSON_SCALAR_TOKEN = re.compile(rb'[^,\]}\s]+')


def _published_text_version(path):
    state = path.stat()
    return (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns)


def published_checkpoint_text(agent, path):
    """Read a just-published theory file once, sharing its existing text value.

    The index/library publishers own these two entries. This cache holds the
    same strings installed in agent.files, rather than another text copy.
    """
    path = Path(path).absolute()
    cache = getattr(agent, '_published_checkpoint_text', None)
    if cache is None:
        cache = agent._published_checkpoint_text = {}
    version = _published_text_version(path)
    saved = cache.get(path)
    if saved is not None and saved[0] == version:
        return saved[1]
    text = path.read_text(encoding='utf-8')
    cache[path] = (version, text)
    return text


def checkpoint_file_text(agent, path):
    """Reuse unchanged published theory text; read all ordinary files normally."""
    path = Path(path).absolute()
    cache = getattr(agent, '_published_checkpoint_text', {})
    saved = cache.get(path)
    if saved is not None:
        if _published_text_version(path) == saved[0]:
            return saved[1]
        del cache[path]
    return path.read_text(encoding='utf-8')


def _json_value_end(text, start):
    """Locate a retained JSON value without constructing another value tree."""
    if text[start] == 34:
        token = _JSON_LAYOUT_TOKEN.match(text, start)
        if token is None or token.end() == start + 1:
            raise ValueError(f'Unterminated JSON string at byte {start}')
        return token.end()
    if text[start] not in (123, 91):
        return _JSON_SCALAR_TOKEN.match(text, start).end()
    depth = 0
    # Skip complete strings and scalar runs in C. In particular a long numeric
    # history has one closing boundary, rather than one Python match per key or
    # quoted value. A bare quote still identifies an unterminated saved string.
    for token in _JSON_CONTAINER_BOUNDARY.finditer(text, start):
        mark = text[token.end() - 1]
        if mark == 34:
            raise ValueError(f'Unterminated JSON string at byte {token.end() - 1}')
        depth += 1 if mark in (123, 91) else -1
        if not depth:
            return token.end()
    raise ValueError(f'Unterminated JSON container at byte {start}')


def _json_size_bound(value, limit, active=None):
    """Bound one C-encoder batch without constructing its JSON text."""
    if isinstance(value, str):
        # A non-BMP code point can become two six-byte surrogate escapes.
        return 2 + (6 if value.isascii() else 12) * len(value)
    if not isinstance(value, (dict, list, tuple)):
        return len(str(value)) + 8
    if active is None:
        active = set()
    identity = id(value)
    if identity in active:
        raise ValueError('Circular reference detected')
    if len(value) * 2 > limit:
        return limit + 1
    active.add(identity)
    size = 2
    try:
        children = value.items() if isinstance(value, dict) else enumerate(value)
        for key, child in children:
            if isinstance(value, dict):
                size += _json_size_bound(str(key), limit - size) + 2
            size += _json_size_bound(child, limit - size, active) + 2
            if size > limit:
                break
        return size
    finally:
        active.remove(identity)


def _bounded_json_chunks(value, encoder):
    """Use the standard C encoder on bounded compact subtrees, not all state.

    Default ``iterencode`` visits every JSON token in Python. Its one-shot C
    variant retains the entire output, so batch actual child values instead.
    Pretty standalone metadata keeps the existing JSONEncoder layout.
    """
    if encoder.indent is not None:
        yield from encoder.iterencode(value)
        return
    active = set()

    def encode(item, *, large=False):
        if isinstance(item, str) and _json_size_bound(item, _JSON_ENCODE_CHUNK) > _JSON_ENCODE_CHUNK:
            yield '"'
            width = _JSON_ENCODE_CHUNK // (6 if item.isascii() else 12)
            for first in range(0, len(item), width):
                yield encoder.encode(item[first:first + width])[1:-1]
            yield '"'
            return
        container = isinstance(item, (dict, list, tuple))
        if not container or (not large and _json_size_bound(item, _JSON_ENCODE_CHUNK)
                             <= _JSON_ENCODE_CHUNK):
            yield encoder.encode(item)
            return
        identity = id(item)
        if identity in active:
            raise ValueError('Circular reference detected')
        active.add(identity)
        mapping = isinstance(item, dict)
        yield '{' if mapping else '['
        pending, size, emitted = [], 2, False

        def batch():
            grouped = dict(pending) if mapping else [child for _, child in pending]
            return encoder.encode(grouped)[1:-1]

        try:
            children = item.items() if mapping else enumerate(item)
            for key, child in children:
                cost = _json_size_bound(child, _JSON_ENCODE_CHUNK)
                if mapping:
                    cost += _json_size_bound(str(key), _JSON_ENCODE_CHUNK) + 2
                if pending and size + cost + 2 > _JSON_ENCODE_CHUNK:
                    if emitted:
                        yield encoder.item_separator
                    yield batch()
                    pending, size, emitted = [], 2, True
                if cost + 2 > _JSON_ENCODE_CHUNK:
                    if emitted:
                        yield encoder.item_separator
                    if mapping:
                        # Preserve the standard key conversion for JSON keys.
                        yield encoder.encode({key: None})[1:-5]
                    yield from encode(child, large=True)
                    emitted = True
                else:
                    pending.append((key, child))
                    size += cost + 2
            if pending:
                if emitted:
                    yield encoder.item_separator
                yield batch()
            yield '}' if mapping else ']'
        finally:
            active.remove(identity)

    yield from encode(value)


def retain_checkpoint_sources(path, *sources):
    """Declare the actual retained canonical files available to one writer."""
    _checkpoint_sources[str(Path(path).absolute())] = tuple(Path(source).resolve() for source in sources)


def _json_values_equal(saved, current, allow_nan=True):
    if not allow_nan and isinstance(current, float) and not isfinite(current):
        raise ValueError('Out of range float values are not JSON compliant')
    if saved is current:
        return True
    if isinstance(saved, dict) and isinstance(current, dict):
        return list(saved) == list(current) and all(_json_values_equal(saved[key], current[key], allow_nan) for key in saved)
    if isinstance(saved, (list, tuple)) and isinstance(current, (list, tuple)):
        return len(saved) == len(current) and all(_json_values_equal(a, b, allow_nan) for a, b in zip(saved, current))
    return saved == current or (
        isinstance(saved, float) and isinstance(current, float) and isnan(saved) and isnan(current))


def _clone_file(source, destination):
    with open(source, 'rb') as original, open(destination, 'wb') as copied:
        try:
            fcntl.ioctl(copied.fileno(), 0x40049409, original.fileno())  # FICLONE
        except OSError as error:
            if error.errno not in {errno.EOPNOTSUPP, errno.ENOTTY, errno.EXDEV}:
                raise
            return False
    return True


def clone_or_copy(source, destination):
    if not _clone_file(source, destination):
        return shutil.copy2(source, destination)
    shutil.copystat(source, destination)
    return destination


def write_chunks_incrementally(path, chunks, *, base=None):
    """Atomically write UTF-8 text, retaining equal blocks of a prior reflink.

    Explicit ``base`` selects the canonical version to share. Equal writes to
    the existing file do nothing, including on filesystems without reflinks.
    No special reader is needed, and a failed write preserves the prior file.
    """
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    try:
        original = Path(base) if base else path
        with ExitStack() as opened:
            source = opened.enter_context(original.open('rb')) if original.is_file() else None
            output = None
            shared = False
            pending = bytearray()
            position = 0

            def start_output():
                nonlocal output, shared
                shared = source is not None and _clone_file(original, temporary)
                output = opened.enter_context(temporary.open('r+b' if shared else 'w+b'))
                if source is not None and not shared and position:
                    # On an ordinary filesystem copy only the already-compared
                    # prefix, then stream the remaining content once.
                    with original.open('rb') as prior:
                        remaining = position
                        while remaining:
                            data = prior.read(min(remaining, 1024 * 1024))
                            output.write(data)
                            remaining -= len(data)

            def block(data):
                nonlocal position
                equal = source is not None and source.read(len(data)) == data
                if not equal and output is None:
                    start_output()
                if output is not None and (not shared or not equal):
                    output.seek(position)
                    output.write(data)
                position += len(data)

            for chunk in chunks:
                # The C JSON encoder can yield one multi-gigabyte string.
                # Bound its UTF-8 and bytearray copies independently of that
                # string's size while retaining the same disk block boundaries.
                for start in range(0, len(chunk), 1024 * 1024):
                    pending.extend(chunk[start:start + 1024 * 1024].encode('utf-8'))
                    end = len(pending) // 4096 * 4096
                    for offset in range(0, end, 4096):
                        block(pending[offset:offset + 4096])
                    del pending[:end]
            if pending:
                block(pending)
            if output is None:
                if source is not None and original == path and os.fstat(source.fileno()).st_size == position:
                    return
                start_output()
            # XFS truncation at an unchanged non-block-aligned EOF can unshare
            # its last block. An exact clone needs no truncation at all.
            output.flush()
            if os.fstat(output.fileno()).st_size != position:
                output.truncate(position)
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_text_incrementally(path, text, *, base=None):
    write_chunks_incrementally(path, (text,), base=base)


class _JsonSource:
    """Positions in an existing ordinary JSON file; no new on-disk format."""

    def __init__(self, path, opened):
        self.path = path
        self.opened = opened
        self.file = opened.enter_context(path.open('rb'))
        self.text = None
        self.fields = None
        self.comparison_encoders = ()
        self._member_cache = OrderedDict()
        self._member_cache_entries = 0
        self._comparison_cache = {}

    def load_text(self):
        if self.text is None:
            # Retained checkpoints can be hundreds of MiB. Keep their UTF-8
            # bytes file-backed instead of allocating a whole Unicode copy.
            self.text = self.opened.enter_context(mmap.mmap(
                self.file.fileno(), 0, access=mmap.ACCESS_READ))

    def members(self, node=None, *, keys=None):
        self.load_text()
        requested = None if keys is None else set(keys)
        if requested == set():
            return {}
        if node is None and self.fields is not None:
            return (self.fields if requested is None else
                    {key: child for key, child in self.fields.items() if key in requested})
        if node is not None and node[1] in self._member_cache:
            self._member_cache.move_to_end(node[1])
            fields = self._member_cache[node[1]]
            return (fields if requested is None else
                    {key: child for key, child in fields.items() if key in requested})
        start = 0 if node is None else node[1]
        text = self.text
        while text[start] in b' \t\r\n':
            start += 1
        kind = text[start]
        if kind not in (123, 91):
            return {}
        position, index, result = start + 1, 0, {}
        while True:
            while text[position] in b' \t\r\n':
                position += 1
            if text[position] in (125, 93):
                break
            if kind == 123:
                end = _json_value_end(text, position)
                key, position = json.loads(text[position:end]), end
                while text[position] in b' \t\r\n':
                    position += 1
                position += 1  # colon
                while text[position] in b' \t\r\n':
                    position += 1
            else:
                key = index
            first = position
            if (requested is None or key in requested) and text[position] not in (123, 91):
                end = _json_value_end(text, position)
                value, position = json.loads(text[position:end]), end
            else:
                # Containers retain only their source span. Compare members
                # when needed instead of decoding a second complete old tree.
                position = _json_value_end(text, position)
                value = None
            if requested is None or key in requested:
                result[key] = (value, first, position)
            if requested is not None and len(result) == len(requested):
                # Questions are at the beginning of real retained checkpoints;
                # finding them must not scan a gigabyte of later score history.
                break
            index += 1
            while text[position] in b' \t\r\n':
                position += 1
            if text[position] == 44:
                position += 1
        if node is None and requested is None:
            self.fields = result
        elif (node is not None and self.comparison_encoders and requested is None
              and node[2] - node[1] >= _JSON_SHARED_MINIMUM
              and len(result) <= _JSON_MEMBER_CACHE_ENTRIES):
            # A changed parent is inspected for equality, then written using
            # the very same spans. Keep a bounded number of those positions
            # for this save only, rather than repeatedly scanning its JSON.
            while (self._member_cache and self._member_cache_entries + len(result)
                   > _JSON_MEMBER_CACHE_ENTRIES):
                _, expired = self._member_cache.popitem(last=False)
                self._member_cache_entries -= len(expired)
            self._member_cache[node[1]] = result
            self._member_cache_entries += len(result)
        return result

    def value_equal(self, node, current, allow_nan=True):
        if node[2] - node[1] < _JSON_SHARED_MINIMUM:
            return self._value_equal(node, current, allow_nan)
        # The caller borrows an unchanged object tree for this synchronous
        # write. Rechecking one large subtree against one retained span cannot
        # change its answer during the write; no cache survives the save.
        key = (node[1], node[2], id(current), allow_nan)
        if key not in self._comparison_cache:
            self._comparison_cache[key] = self._value_equal(node, current, allow_nan)
        return self._comparison_cache[key]

    def _value_equal(self, node, current, allow_nan=True):
        kind = self.text[node[1]]
        if kind not in (123, 91):
            return _json_values_equal(node[0], current, allow_nan)
        mapping = kind == 123
        if not isinstance(current, dict if mapping else (list, tuple)):
            return False
        # The saved record/Program commonly has one of our two ordinary JSON
        # layouts. Compare a bounded encoding in C before walking its nodes.
        # Whole indexes remain spans rather than another giant string/tree.
        length = node[2] - node[1]
        if mapping and length <= _JSON_ENCODED_COMPARE_MAXIMUM:
            for encoder in self.comparison_encoders:
                if current:
                    first = next(iter(current))
                    if not isinstance(current[first], (dict, list, tuple)):
                        prefix = encoder.encode({first: current[first]})[:-1].encode('utf-8')
                        if self.text[node[1]:node[1] + len(prefix)] != prefix:
                            continue
                encoded = encoder.encode(current).encode('utf-8')
                if len(encoded) == length and self.text[node[1]:node[2]] == encoded:
                    return True
        members = self.members(node)
        if mapping:
            if list(members) != list(current):
                return False
            values = current.items()
        else:
            if len(members) != len(current):
                return False
            values = enumerate(current)
        return all(self.value_equal(members[key], child, allow_nan)
                   for key, child in values)

    def byte_position(self, position):
        return position


def read_json_checkpoint(path, *, keys=None):
    """Restore JSON, decoding only selected top-level fields when requested."""
    with ExitStack() as opened:
        source = _JsonSource(Path(path), opened)
        source.load_text()
        start = 0
        while source.text[start] in b' \t\r\n':
            start += 1

        def decode(node):
            _, first, end = node
            kind = source.text[first]
            if end - first <= _JSON_DECODE_CHUNK or kind not in (123, 91):
                return json.loads(source.text[first:end])
            members = source.members(node)
            if kind == 123:
                return {key: decode(child) for key, child in members.items()}
            return [decode(child) for child in members.values()]

        if keys is not None:
            return {key: decode(child) for key, child in
                    source.members(keys=set(keys)).items()}
        # Ordinary checkpoints contain one complete value and trailing JSON
        # whitespace. The file boundary already supplies its end; scanning the
        # whole tree here would immediately be repeated by members/loads.
        end = len(source.text)
        while source.text[end - 1] in b' \t\r\n':
            end -= 1
        return decode((None, start, end))


def _write_json_ranges(path, value, sources, *, encoder):
    """Share equal JSON values even when preceding metadata moved their bytes."""
    temporary = path.with_name(path.name + '.writing')
    with ExitStack() as opened:
        retained = [_JsonSource(source, opened) for source in sources]
        comparison_encoders = tuple(json.JSONEncoder(ensure_ascii=encoder.ensure_ascii,
            allow_nan=encoder.allow_nan, separators=separators)
            for separators in ((',', ':'), (', ', ': ')))
        for source in retained:
            source.comparison_encoders = comparison_encoders
        all_sources = retained
        original = next((source for source in retained if source.path == path), None)
        output, position = None, 0
        output_shared = False
        emitted_evaluations = {}

        def start_output():
            nonlocal output, output_shared
            output_shared = bool(retained and _clone_file(retained[0].path, temporary))
            output = opened.enter_context(temporary.open('r+b' if output_shared else 'w+b', buffering=0))
            if not output_shared and original is not None and position:
                original.file.seek(0)
                remaining = position
                while remaining:
                    data = original.file.read(min(remaining, 1024 * 1024))
                    output.write(data)
                    remaining -= len(data)

        def emit(data):
            nonlocal position
            if output is None and original is not None:
                equal = os.pread(original.file.fileno(), len(data), position) == data
                if not equal:
                    start_output()
            elif output is None:
                start_output()
            if output is not None:
                output.seek(position)
                output.write(data)
            position += len(data)

        def copy_range(descriptor, first, end, *, unchanged_source=False):
            nonlocal position
            # JSON whitespace can align a moved value with its original blocks.
            emit(b' ' * ((first - position) % 4096))
            prefix = min((-first) % 4096, end - first)
            emit(os.pread(descriptor, prefix, first))
            first += prefix
            shared = (end - first) // 4096 * 4096
            if shared:
                unchanged = (unchanged_source and first == position
                             and (output is None or output_shared))
                if not unchanged:
                    if output is None:
                        start_output()
                    try:
                        fcntl.ioctl(output.fileno(), 0x4020940D,
                            struct.pack('qQQQ', descriptor, first, shared, position))
                    except OSError as error:
                        if error.errno not in {errno.EOPNOTSUPP, errno.ENOTTY, errno.EXDEV, errno.EINVAL}:
                            source = next((saved.path for saved in all_sources
                                           if saved.file.fileno() == descriptor), temporary)
                            raise OSError(error.errno,
                                f'{error.strerror}; checkpoint range reflink '
                                f'{source} offset={first} length={shared} '
                                f'size={os.fstat(descriptor).st_size} -> '
                                f'{temporary} offset={position} '
                                f'size={os.fstat(output.fileno()).st_size}') from error
                        offset = 0
                        while offset < shared:
                            data = os.pread(descriptor, min(shared - offset, 1024 * 1024),
                                            first + offset)
                            os.pwrite(output.fileno(), data, position + offset)
                            offset += len(data)
                position += shared
                first += shared
            emit(os.pread(descriptor, end - first, first))

        def copy_value(source, node):
            copy_range(source.file.fileno(), source.byte_position(node[1]),
                       source.byte_position(node[2]), unchanged_source=source is original)

        def literal(item):
            for chunk in _bounded_json_chunks(item, encoder):
                emit(chunk.encode('utf-8'))

        def comparison_owner(item, field=None):
            if isinstance(item, list):
                return field in {'operator_results', 'method_results', 'events',
                                 'after_observation_results'}
            if not isinstance(item, dict):
                return False
            feedback = item.get('feedback')
            return (isinstance(item.get('evaluations'), list)
                    or isinstance(feedback, dict)
                    and isinstance(feedback.get('evaluations'), list))

        def write_value(item, candidates, *, field=None):
            # One real comparison is copied into both operator and method
            # feedback. Retain its already written range, not another value
            # tree, and share only after complete ordered equality succeeds.
            comparison = field == 'evaluations' and isinstance(item, list)
            if comparison:
                for saved, first, end in emitted_evaluations.get(len(item), ()):
                    if _json_values_equal(saved, item, encoder.allow_nan):
                        copy_range(output.fileno(), first, end)
                        return
            first = position
            if comparison_owner(item, field):
                # New or completely equal history shells still have to visit
                # their evaluations; otherwise cross-owner sharing is skipped.
                write_container(item, candidates)
            else:
                for source, node in candidates:
                    if (node[2] - node[1] >= _JSON_SHARED_MINIMUM
                            and source.value_equal(node, item, encoder.allow_nan)):
                        copy_value(source, node)
                        break
                else:
                    large = any(node[2] - node[1] >= _JSON_SHARED_MINIMUM for _, node in candidates)
                    if isinstance(item, (dict, list)) and large:
                        write_container(item, candidates)
                    else:
                        literal(item)
            if comparison and output is not None and position - first >= _JSON_SHARED_MINIMUM:
                emitted_evaluations.setdefault(len(item), []).append((item, first, position))

        def write_container(item, candidates):
            mapping = isinstance(item, dict)
            sources_by_key = [(source, source.members(node)) for source, node in candidates
                              if source.text[node[1]] == (123 if mapping else 91)]
            equal_run_ends = [0] * len(sources_by_key)
            emit(b'{' if mapping else b'[')
            members = list(item.items()) if mapping else list(enumerate(item))
            index = 0
            while index < len(members):
                key, child = members[index]
                if index:
                    emit(encoder.item_separator.encode())
                if mapping:
                    literal(str(key))
                    emit(encoder.key_separator.encode())
                elif not comparison_owner(child):
                    # A full-pool comparison consists of many small scores.
                    # Reuse a contiguous equal run as one range instead of
                    # rewriting it because each score is below the threshold.
                    copied_until = None
                    for source_index, (source, fields) in enumerate(sources_by_key):
                        end = equal_run_ends[source_index]
                        if end <= index:
                            end = index
                            while (end < len(item) and end in fields and
                                   not comparison_owner(item[end]) and
                                   source.value_equal(fields[end], item[end], encoder.allow_nan)):
                                end += 1
                            equal_run_ends[source_index] = end
                        if (end > index and fields[end - 1][2] - fields[index][1]
                                >= _JSON_SHARED_MINIMUM):
                            copy_value(source, (None, fields[index][1], fields[end - 1][2]))
                            copied_until = end
                            break
                    if copied_until is not None:
                        index = copied_until
                        continue
                children = []
                for source, fields in sources_by_key:
                    node = fields.get(key)
                    if node is None and not mapping and fields and key >= len(fields):
                        # An appended execution has no old ordinal. Its actual
                        # preceding execution still supplies unchanged fields
                        # and paid-score prefixes; equality remains mandatory.
                        node = fields[len(fields) - 1]
                    if node is not None:
                        children.append((source, node))
                write_value(child, children, field=key if mapping else None)
                index += 1
            emit(b'}' if mapping else b']')

        try:
            # Index each retained top-level field once. Equal large fields are
            # compared as values and copied without encoding their payload again.
            for source in retained:
                source.load_text()
            fields = [(source, source.members(keys=value)) for source in retained]
            emit(b'{')
            for index, (key, item) in enumerate(value.items()):
                if index:
                    emit(encoder.item_separator.encode())
                literal(str(key))
                emit(encoder.key_separator.encode())
                write_value(item, [(source, entries[key]) for source, entries in fields if key in entries], field=key)
            emit(b'}\n')
            if output is None:
                if os.fstat(original.file.fileno()).st_size == position:
                    return
                start_output()
            if os.fstat(output.fileno()).st_size != position:
                output.truncate(position)
            # All value comparisons and source-range reads are finished. Drop
            # writer-owned JSON trees before the durable filesystem commit.
            fields.clear()
            emitted_evaluations.clear()
            for source in all_sources:
                source.text.close()
                source.text = source.fields = None
                source._member_cache.clear()
                source._comparison_cache.clear()
                source.file.close()
            retained.clear()
            all_sources.clear()
            original = None
            os.fsync(output.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def write_json_checkpoint(path, value, *, base=None, indent=2, ensure_ascii=True,
                          allow_nan=None, separators=None):
    """Write complete ordinary JSON, sharing retained equal fields and records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bulk = path.name in {'active.json', 'training.json', 'result.json'}
    if bulk and isinstance(value, dict) and 'status' in value:
        value = {**{key: item for key, item in value.items() if key != 'status'}, 'status': value['status']}
    candidates = [path] if path.is_file() else []
    if base is not None:
        candidates.extend(base if isinstance(base, (list, tuple)) else [base])
    if not path.is_file() and bulk:
        names = {'training.json': ('active.json', 'research_state.json'),
                 'active.json': ('research_state.json',),
                 'result.json': ('training.json', 'active.json', 'research_state.json')}.get(path.name, ())
        candidates.extend(path.with_name(name) for name in names)
        if isinstance(value, dict) and value.get('resume_origin'):
            candidates.append(Path(value['resume_origin']))
    candidates.extend(_checkpoint_sources.get(str(path.absolute()), ()))
    sources = []
    for candidate in candidates:
        candidate = Path(candidate).resolve()
        if candidate.is_file() and candidate not in sources:
            sources.append(candidate)
    encoder = json.JSONEncoder(indent=None if bulk else indent,
        separators=(',', ':') if bulk else separators, ensure_ascii=ensure_ascii,
        allow_nan=not bulk if allow_nan is None else allow_nan)
    if isinstance(value, dict) and any(source.stat().st_size >= _JSON_SHARED_MINIMUM for source in sources):
        # A finished active.json can alias training.json. Replace that logical
        # checkpoint atomically rather than writing its retained final target.
        result = _write_json_ranges(path.absolute(), value, sources, encoder=encoder)
        # The newly committed complete local file now owns all retained fields.
        # Future writes compare it, without reparsing old large source archives.
        _checkpoint_sources.pop(str(path.absolute()), None)
        return result
    def chunks():
        yield from _bounded_json_chunks(value, encoder)
        yield '\n'
    write_chunks_incrementally(path, chunks(), base=sources[0] if sources else None)
    _checkpoint_sources.pop(str(path.absolute()), None)
