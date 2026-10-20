"""Finished checkpoints share the final result without changing paid records."""

import json

import pytest

from scripts.run_online_inquiry_llm import final_checkpoint_alias, write_json


def test_final_checkpoint_reuses_final_result_and_keeps_future_writes_separate(tmp_path):
    final = tmp_path / 'training.json'
    checkpoint = tmp_path / 'active.json'
    result = {'status': 'completed', 'paid_records': [{'observation': 3.}]}
    write_json(checkpoint, {**result, 'status': 'researching'})
    write_json(final, result)
    final_bytes = final.read_bytes()

    final_checkpoint_alias(final, checkpoint)

    assert checkpoint.is_symlink()
    assert checkpoint.readlink().as_posix() == 'training.json'
    assert json.loads(checkpoint.read_text()) == result
    assert final.read_bytes() == final_bytes

    # The existing atomic checkpoint writer detaches this alias on continuation.
    write_json(checkpoint, {'status': 'researching', 'paid_records': []})
    assert not checkpoint.is_symlink()
    assert final.read_bytes() == final_bytes
    assert json.loads(checkpoint.read_text())['status'] == 'researching'


def test_missing_final_result_keeps_existing_checkpoint(tmp_path):
    checkpoint = tmp_path / 'active.json'
    write_json(checkpoint, {'status': 'researching', 'paid_records': [4.]})
    previous = checkpoint.read_bytes()

    with pytest.raises(FileNotFoundError):
        final_checkpoint_alias(tmp_path / 'training.json', checkpoint)

    assert not checkpoint.is_symlink()
    assert checkpoint.read_bytes() == previous
