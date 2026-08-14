"""Minimal package boundary for the isolated Phase-2B recognizer image.

The repository package initializer intentionally exposes a broad development
API.  Importing that initializer inside the recognizer would pull unrelated
Phase-3 and evaluator modules across the container boundary.  The image copies
this inert initializer in its place and imports only the exact frozen module
closure required by the recognizer CLI.
"""

from __future__ import annotations


__all__: tuple[str, ...] = ()
