"""CLI-only entry point for the pure-offline fixed qualification aggregate."""

from __future__ import annotations

from .fixed_public_qualification import main_aggregate


if __name__ == "__main__":
    raise SystemExit(main_aggregate())
