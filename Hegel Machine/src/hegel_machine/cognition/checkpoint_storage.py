"""Ordinary files with independent writes and shared unchanged disk blocks."""

import errno
import fcntl
import os
from pathlib import Path
import shutil


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

    On filesystems without reflinks this remains an ordinary atomic write.
    No special reader is needed, and a failed write preserves the prior file.
    """
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    try:
        original = path if path.is_file() else Path(base) if base else None
        shared = original is not None and original.is_file() and _clone_file(original, temporary)
        with temporary.open('r+b' if shared else 'w+b') as output:
            pending = bytearray()
            position = 0

            def block(data):
                nonlocal position
                if not shared or output.read(len(data)) != data:
                    output.seek(position)
                    output.write(data)
                position += len(data)

            for chunk in chunks:
                pending.extend(chunk.encode('utf-8'))
                end = len(pending) // 4096 * 4096
                for offset in range(0, end, 4096):
                    block(pending[offset:offset + 4096])
                del pending[:end]
            if pending:
                block(pending)
            output.truncate(position)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_text_incrementally(path, text, *, base=None):
    write_chunks_incrementally(path, (text,), base=base)
