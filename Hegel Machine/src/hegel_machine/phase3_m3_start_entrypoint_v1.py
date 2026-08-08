"""Direct entrypoint for the committed formal M3 start command.

The minimal package shell prevents ``hegel_machine.__init__`` from importing
modules outside the runtime-source closure before source identity is checked.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import sys
from types import ModuleType


EXPECTED_INTERPRETER_SHA256 = (
    "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
)
EXPECTED_INTERPRETER_VERSION = (
    "3.10.12 (main, Jun 22 2026, 18:55:27) [GCC 11.4.0]"
)
if __package__ not in {None, ""}:
    raise RuntimeError("formal M3 entrypoint must be invoked by its committed file path")
if (
    sys.flags.isolated != 1
    or sys.flags.no_site != 1
    or not sys.dont_write_bytecode
):
    raise RuntimeError("formal M3 entrypoint requires python -I -S -B")
interpreter = Path(sys.executable).resolve(strict=True)
interpreter_metadata = interpreter.lstat()
if (
    interpreter != Path("/usr/bin/python3.10")
    or not stat.S_ISREG(interpreter_metadata.st_mode)
    or interpreter_metadata.st_uid != 0
    or stat.S_IMODE(interpreter_metadata.st_mode) != 0o755
    or sys.version != EXPECTED_INTERPRETER_VERSION
    or hashlib.sha256(interpreter.read_bytes()).hexdigest()
    != EXPECTED_INTERPRETER_SHA256
):
    raise RuntimeError("formal M3 host interpreter identity differs")
if sys.pycache_prefix is None:
    raise RuntimeError("formal M3 entrypoint requires a fresh -X pycache_prefix")
cache_root = Path(sys.pycache_prefix)
cache_metadata = cache_root.lstat()
if (
    not cache_root.is_absolute()
    or stat.S_ISLNK(cache_metadata.st_mode)
    or not stat.S_ISDIR(cache_metadata.st_mode)
    or cache_root.resolve(strict=True) != cache_root
    or cache_metadata.st_uid != os.geteuid()
    or stat.S_IMODE(cache_metadata.st_mode) != 0o700
    or any(cache_root.iterdir())
):
    raise RuntimeError("formal M3 pycache prefix must be an empty owned mode-0700 directory")
package = ModuleType("hegel_machine")
package.__path__ = [str(Path(__file__).resolve().parent)]  # type: ignore[attr-defined]
package.__package__ = "hegel_machine"
sys.modules["hegel_machine"] = package
__package__ = "hegel_machine"

from .phase3_m3_start_cli_v1 import _DIRECT_ENTRYPOINT_SEAL, main


if __name__ == "__main__":
    raise SystemExit(main(_launch_capability=_DIRECT_ENTRYPOINT_SEAL))
