"""Keep raw JSON readable and paid originals unchanged during incremental writes."""

import json
from pathlib import Path
import shutil

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
