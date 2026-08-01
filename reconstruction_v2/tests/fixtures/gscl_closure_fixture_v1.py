"""Tiny import graph for source-free runtime-closure qualification."""

from __future__ import annotations

import hashlib
import json


def stable_fixture_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True).encode("ascii")
    ).hexdigest()
