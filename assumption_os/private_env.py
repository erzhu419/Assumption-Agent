"""Safe local private environment loading for local HLE/source runs.

The loader intentionally supports only simple dotenv-style ``KEY=VALUE`` lines.
It does not evaluate shell syntax, expand variables, or log values.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import MutableMapping


DEFAULT_PRIVATE_ENV_PATH = Path("~/.config/assumption-agent/private.env").expanduser()
DEFAULT_PRIVATE_ENV_REQUIRED_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "SEMANTIC_SCHOLAR_API_KEY",
    "OPENALEX_API_KEY",
    "OPENALEX_MAILTO",
)

_KEY_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def load_private_env(
    *,
    environ: MutableMapping[str, str] | None = None,
    path: str | os.PathLike[str] | None = None,
    override: bool = False,
    required_keys: tuple[str, ...] = DEFAULT_PRIVATE_ENV_REQUIRED_KEYS,
) -> dict[str, object]:
    """Load private env vars from a permission-checked local file.

    Returned metadata includes variable names and presence booleans only.  Values
    are never returned, printed, or otherwise persisted by this helper.
    """
    env = environ if environ is not None else os.environ
    requested_path = Path(
        path
        or env.get("ASSUMPTION_AGENT_PRIVATE_ENV")
        or DEFAULT_PRIVATE_ENV_PATH
    ).expanduser()
    metadata: dict[str, object] = {
        "path": str(requested_path),
        "exists": requested_path.exists(),
        "loaded": False,
        "loaded_keys": [],
        "skipped_keys": [],
        "required_key_present": {
            key: bool(str(env.get(key) or "").strip())
            for key in required_keys
        },
        "raw_content_persisted": False,
    }
    if not requested_path.exists():
        metadata["skipped_reason"] = "missing_private_env_file"
        return metadata

    try:
        mode = requested_path.stat().st_mode & 0o777
    except OSError as exc:
        metadata["skipped_reason"] = f"stat_failed:{type(exc).__name__}"
        return metadata
    metadata["mode"] = oct(mode)
    if mode & 0o077 and not _env_truthy(env, "ASSUMPTION_AGENT_ALLOW_INSECURE_PRIVATE_ENV"):
        metadata["skipped_reason"] = "private_env_file_not_private"
        return metadata

    loaded_keys: list[str] = []
    skipped_keys: list[str] = []
    try:
        lines = requested_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        metadata["skipped_reason"] = f"read_failed:{type(exc).__name__}"
        return metadata
    for line in lines:
        parsed = _parse_private_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if not override and str(env.get(key) or "").strip():
            skipped_keys.append(key)
            continue
        env[key] = value
        loaded_keys.append(key)

    metadata["loaded"] = bool(loaded_keys)
    metadata["loaded_keys"] = sorted(loaded_keys)
    metadata["skipped_keys"] = sorted(skipped_keys)
    metadata["required_key_present"] = {
        key: bool(str(env.get(key) or "").strip())
        for key in required_keys
    }
    if not loaded_keys:
        metadata["skipped_reason"] = "no_new_keys_loaded"
    return metadata


def _parse_private_env_line(line: str) -> tuple[str, str] | None:
    stripped = str(line or "").strip()
    if not stripped or stripped.startswith("#"):
        return None
    match = _KEY_RE.match(stripped)
    if not match:
        return None
    key, raw_value = match.groups()
    value = _strip_inline_comment(raw_value.strip())
    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in {"'", '"'}
    ):
        value = value[1:-1]
    return key, value


def _strip_inline_comment(value: str) -> str:
    quote: str | None = None
    for index, char in enumerate(value):
        if char in {"'", '"'}:
            quote = None if quote == char else char if quote is None else quote
            continue
        if char == "#" and quote is None and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value


def _env_truthy(env: MutableMapping[str, str], key: str) -> bool:
    return str(env.get(key, "")).strip().lower() in {"1", "true", "yes", "on"}
