"""Inert package boundary for the isolated Phase-2B scorer image.

The repository initializer exposes a broad development API.  The scorer image
imports only the frozen module closure copied by its Dockerfile, so its package
initializer intentionally performs no eager imports.
"""

from __future__ import annotations


__all__: tuple[str, ...] = ()
