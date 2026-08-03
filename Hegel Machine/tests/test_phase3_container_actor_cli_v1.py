from __future__ import annotations

import json

import pytest

from hegel_machine.phase3_container_actor_cli_v1 import _publish_exclusive


def test_publication_is_exact_and_never_overwrites(tmp_path) -> None:
    destination = tmp_path / "nested" / "qualification.json"
    payload = b'{"eligible":true}\n'
    _publish_exclusive(destination, payload)
    assert destination.read_bytes() == payload
    assert destination.stat().st_mode & 0o777 == 0o644
    assert json.loads(destination.read_text(encoding="ascii")) == {"eligible": True}

    with pytest.raises(FileExistsError):
        _publish_exclusive(destination, b'{"eligible":false}\n')
    assert destination.read_bytes() == payload
