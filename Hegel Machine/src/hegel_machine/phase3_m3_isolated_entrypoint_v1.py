"""Direct-script entrypoint that does not execute ``hegel_machine.__init__``.

The normal package initializer intentionally re-exports benchmark contracts.
The target-independent M3 enumerator container must not even be able to read
those modules, so its committed snapshot uses this direct entrypoint and a
minimal in-memory package shell.
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType


if __package__ in {None, ""}:
    package = ModuleType("hegel_machine")
    package.__path__ = [str(Path(__file__).resolve().parent)]  # type: ignore[attr-defined]
    package.__package__ = "hegel_machine"
    sys.modules["hegel_machine"] = package
    __package__ = "hegel_machine"

from .phase3_m3_bounded_enumerator_cli_v1 import main


if __name__ == "__main__":
    raise SystemExit(main())
