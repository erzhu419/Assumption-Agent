"""A writeback failure must keep the previous paid checkpoint readable."""

import json

import pytest

from scripts.run_online_inquiry_llm import write_json


def test_streamed_checkpoint_keeps_all_fields(tmp_path):
    value = {'status': 'researching', 'costs': {'model_calls': 621},
             'files': {'model.py': 'def predict(row):\n    return 0.\n'},
             'rows': [{'features': {'x': .5}, 'target': -2.5, 'history': []}]}
    target = tmp_path / 'active.json'
    write_json(target, value)
    assert json.loads(target.read_text()) == value
    assert not target.with_name('active.json.writing').exists()


def test_failed_writeback_preserves_previous_paid_checkpoint(tmp_path, monkeypatch):
    target = tmp_path / 'active.json'
    prior = {'costs': {'model_calls': 620}, 'paid_observations': [{'row_indices': [0]}]}
    write_json(target, prior)

    def failed_writeback(fd):
        raise OSError(5, 'Input/output error')

    monkeypatch.setattr('scripts.run_online_inquiry_llm.os.fsync', failed_writeback)
    with pytest.raises(OSError, match='Input/output error'):
        write_json(target, {'costs': {'model_calls': 621}})
    assert json.loads(target.read_text()) == prior
    assert not target.with_name('active.json.writing').exists()
