"""Keep raw JSON readable and paid originals unchanged during incremental writes."""

import json
from pathlib import Path
import shutil
import tracemalloc

import pytest

from hegel_machine.cognition import checkpoint_storage as storage


def test_block_updates_preserve_exact_text_when_growing_and_shrinking(tmp_path, monkeypatch):
    # Exercise the shared-file update branch also on ext4 test workspaces.
    def cloned(source, destination):
        shutil.copyfile(source, destination)
        return True

    monkeypatch.setattr(storage, '_clone_file', cloned)
    path = tmp_path / 'research_state.json'
    old = json.dumps({'notes': '实测' * 5000, 'events': [1, 2, 3]}, ensure_ascii=False)
    path.write_text(old)
    for text in (old, old.replace('[1, 2, 3]', '[1, 2, 3, 4]'), '{"events": []}'):
        storage.write_chunks_incrementally(path, (text[:29], text[29:4101], text[4101:]))
        assert path.read_text() == text
        assert json.loads(path.read_text()) == json.loads(text)
    assert not path.with_name(path.name + '.writing').exists()


def test_result_uses_prior_file_without_publishing_it_before_commit(tmp_path, monkeypatch):
    original, new = tmp_path / 'prior.json', tmp_path / 'result.json'
    original.write_text('{"costs": {"model_calls": 272}, "events": [1]}')
    prior = original.read_bytes()

    def failed(fd):
        assert not new.exists(), 'An unfinished practice must not look completed.'
        raise OSError('writeback failed')

    with monkeypatch.context() as patch:
        patch.setattr(storage.os, 'fsync', failed)
        with pytest.raises(OSError, match='writeback failed'):
            storage.write_text_incrementally(new, '{"events": [1,2]}', base=original)
    assert not new.exists()
    assert original.read_bytes() == prior
    storage.write_text_incrementally(new, '{"events": [1,2]}', base=original)
    assert json.loads(new.read_text()) == {'events': [1, 2]}
    assert original.read_bytes() == prior
    assert not new.is_symlink()


def test_failed_serialization_keeps_previous_checkpoint(tmp_path):
    path = tmp_path / 'active.json'
    path.write_text('{"model_calls": 272}')

    def interrupted():
        yield '{"events": ['
        raise ValueError('missing current event')

    with pytest.raises(ValueError, match='missing current event'):
        storage.write_chunks_incrementally(path, interrupted())
    assert json.loads(path.read_text()) == {'model_calls': 272}
    assert not Path(str(path) + '.writing').exists()


def test_operator_library_save_uses_incremental_storage_and_keeps_failed_write(tmp_path, monkeypatch):
    from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent

    agent = object.__new__(TemporalOperatorAgent)
    agent.workspace, agent.files = tmp_path, {}
    name = 'shared/theory_library.json'
    original = json.dumps({'transfers': [{'status': 'refuted', 'evidence': 'x' * 9000}]})
    agent._save(name, original)
    assert agent.files[name] == original
    attempts = []

    def failed_clone(source, destination):
        attempts.append(Path(source))
        raise OSError('storage full')

    monkeypatch.setattr(storage, '_clone_file', failed_clone)
    with pytest.raises(OSError, match='storage full'):
        agent._save(name, original + ' ')
    assert attempts == [tmp_path / name]
    assert (tmp_path / name).read_text() == agent.files[name] == original


def test_identical_checkpoint_is_not_replaced_or_cloned(tmp_path, monkeypatch):
    path = tmp_path / 'active.json'
    text = json.dumps({'old_cost': 272, 'evidence': '旧反例' * 6000}, ensure_ascii=False)
    path.write_text(text)
    before = path.stat()

    def unexpected(*args):
        pytest.fail('An identical checkpoint must not allocate another snapshot.')

    monkeypatch.setattr(storage, '_clone_file', unexpected)
    storage.write_chunks_incrementally(path, (text[:11], text[11:]))
    assert path.stat().st_ino == before.st_ino
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert path.read_text() == text
    assert not Path(str(path) + '.writing').exists()


def test_latest_explicit_base_is_shared_even_when_destination_already_exists(tmp_path, monkeypatch):
    canonical = tmp_path / 'observations.json'
    mirror = tmp_path / 'model_observations.json'
    canonical.write_text(json.dumps({'rows': list(range(2000))}))
    mirror.write_text('{"rows": []}')
    selected = []

    def cloned(source, destination):
        selected.append(Path(source))
        shutil.copyfile(source, destination)
        return True

    monkeypatch.setattr(storage, '_clone_file', cloned)
    storage.write_text_incrementally(mirror, canonical.read_text(), base=canonical)
    assert selected == [canonical]
    assert mirror.read_bytes() == canonical.read_bytes()
    mirror.write_text('{"rows": ["refit only"]}')
    assert json.loads(canonical.read_text())['rows'] == list(range(2000))


def test_ordinary_filesystem_streams_changed_suffix_without_losing_compared_prefix(tmp_path, monkeypatch):
    path = tmp_path / 'active.json'
    prefix = 'x' * (3 * 4096)
    path.write_text(prefix + 'old')
    monkeypatch.setattr(storage, '_clone_file', lambda *args: False)
    storage.write_chunks_incrementally(path, (prefix[:9], prefix[9:], 'new effective event'))
    assert path.read_text() == prefix + 'new effective event'


@pytest.mark.parametrize(('old', 'new', 'expected_truncations'), [
    (None, 'x' * 4096 + 'buffered tail', []),
    ('x' * 4096 + 'z', 'x' * 4096 + 'z\n', []),
    ('x' * 4096 + 'tail', 'x' * 4096 + 't', [4097]),
], ids=['new-buffered-tail', 'append-one-byte', 'shorten-existing-file'])
def test_flushes_buffered_tail_before_deciding_to_truncate(
        tmp_path, monkeypatch, old, new, expected_truncations):
    path = tmp_path / 'training.json'
    if old is not None:
        path.write_text(old)
    original_open = Path.open
    truncations = []

    class CheckedOutput:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __enter__(self):
            self.wrapped.__enter__()
            return self

        def __exit__(self, *args):
            return self.wrapped.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def truncate(self, size):
            # BufferedRandom.truncate flushes first. Reject a call whose only
            # reason was the stale size observed before that flush.
            self.wrapped.flush()
            assert storage.os.fstat(self.wrapped.fileno()).st_size != size
            truncations.append(size)
            return self.wrapped.truncate(size)

    def checked_open(target, mode='r', *args, **kwargs):
        opened = original_open(target, mode, *args, **kwargs)
        if target.name.endswith('.writing') and mode in {'r+b', 'w+b'}:
            return CheckedOutput(opened)
        return opened

    def cloned(source, destination):
        shutil.copyfile(source, destination)
        return True

    monkeypatch.setattr(Path, 'open', checked_open)
    monkeypatch.setattr(storage, '_clone_file', cloned)
    storage.write_chunks_incrementally(path, (new,))
    assert path.read_text() == new
    assert truncations == expected_truncations
    assert not Path(str(path) + '.writing').exists()


def test_giant_encoder_chunk_has_bounded_additional_memory(tmp_path):
    path = tmp_path / 'training.json'
    # Allocate the encoder's existing result before measuring writer overhead.
    text = 'x' * (8 * 1024 * 1024)
    tracemalloc.start()
    try:
        storage.write_chunks_incrementally(path, (text,))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 10 * 1024 * 1024
    assert path.read_bytes() == text.encode('utf-8')


def test_giant_unicode_chunk_preserves_bytes_noop_and_shortening(tmp_path, monkeypatch):
    path = tmp_path / 'training.json'
    # The emoji crosses a 4096-byte boundary; the final characters cross the
    # new 1Mi-character conversion boundary with multibyte UTF-8 on both sides.
    text = 'x' * 4095 + '🧪' + 'é' * (1024 * 1024 - 4096) + '中尾'

    def cloned(source, destination):
        shutil.copyfile(source, destination)
        return True

    monkeypatch.setattr(storage, '_clone_file', cloned)
    storage.write_chunks_incrementally(path, (text,))
    assert path.read_bytes() == text.encode('utf-8')
    before = path.stat()
    storage.write_chunks_incrementally(path, (text,))
    assert (path.stat().st_ino, path.stat().st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    shortened = text[:-3]
    storage.write_chunks_incrementally(path, (shortened,))
    assert path.read_bytes() == shortened.encode('utf-8')
