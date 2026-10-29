"""An activity cutoff must preserve the last committed checkpoint and layout."""

import json

import pytest

from hegel_machine.cognition import checkpoint_storage as storage
from hegel_machine.cognition.experiment_runtime import ActivityExecutionDeadline


def test_activity_cutoff_during_write_preserves_paid_state_and_layout(tmp_path, monkeypatch):
    path = tmp_path / 'training.json'
    paid = {'scores': [{'model': i, 'mse': .125} for i in range(4000)],
            'cursor': 195, 'compute_cost': 403962, 'note': 'retained', 'status': 'researching'}
    storage.write_json_checkpoint(path, paid)
    storage.write_json_checkpoint(path, paid)
    before = path.read_bytes()
    version, fields = storage._checkpoint_layouts[path.absolute()]
    changed = {**paid, 'note': 'new effective information ' * 5000}
    encode = storage._bounded_json_chunks

    def interrupted(item, encoder):
        for index, chunk in enumerate(encode(item, encoder)):
            if item is changed['note'] and index == 2:
                # Same exception raised by the outer activity timer, after
                # writing part of a new payload but before atomic commit.
                raise ActivityExecutionDeadline('Research activity time budget exhausted')
            yield chunk

    with monkeypatch.context() as patch:
        patch.setattr(storage, '_bounded_json_chunks', interrupted)
        with pytest.raises(ActivityExecutionDeadline):
            storage.write_json_checkpoint(path, changed)
    assert path.read_bytes() == before
    assert not path.with_name('training.json.writing').exists()
    assert storage._checkpoint_layouts[path.absolute()] == (version, fields)
    assert storage.read_json_checkpoint(path) == paid

    storage.write_json_checkpoint(path, changed)
    assert json.loads(path.read_bytes()) == changed
    assert storage.read_json_checkpoint(path) == changed
