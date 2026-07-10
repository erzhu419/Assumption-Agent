from __future__ import annotations

import os
from pathlib import Path


LOCKED_MODEL = "gpt-5.3-codex-spark"


def alternate_model_allowed() -> bool:
    return os.environ.get("ASSUMPTION_V2_ALLOW_ALTERNATE_MODEL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def configured_model(*, enforce_policy: bool = True) -> str:
    model = os.environ.get("ASSUMPTION_V2_MODEL", LOCKED_MODEL).strip() or LOCKED_MODEL
    if enforce_policy and model != LOCKED_MODEL and not alternate_model_allowed():
        raise RuntimeError(
            f"reconstruction v2 is locked to {LOCKED_MODEL} until "
            "ASSUMPTION_V2_ALLOW_ALTERNATE_MODEL=1"
        )
    return model


def configured_skilllearn_provider_mode() -> str:
    mode = os.environ.get(
        "ASSUMPTION_V2_SKILLLEARN_PROVIDER_MODE",
        "codex_subscription",
    ).strip().lower()
    if mode not in {"codex_subscription", "openai_compatible"}:
        raise ValueError("unsupported SkillLearn trial provider mode")
    return mode


def resolve_codex_auth_path() -> Path | None:
    candidates: list[Path] = []
    configured = os.environ.get("ASSUMPTION_V2_CODEX_AUTH_PATH", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    codex_home = os.environ.get("CODEX_HOME", "").strip()
    if codex_home:
        candidates.append(Path(codex_home).expanduser() / "auth.json")
    candidates.append(Path.home() / ".codex" / "auth.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def load_dotenv(path: str | Path, *, override: bool = False) -> tuple[str, ...]:
    """Load a simple dotenv file without returning or logging secret values."""

    source = Path(path).expanduser()
    loaded: list[str] = []
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return tuple(sorted(loaded))


def map_legacy_model_env(*, override: bool = False) -> dict[str, bool | str]:
    aliases = {
        "ASSUMPTION_V2_API_BASE": ("RUOLI_BASE_URL", "GPT5_BASE_URL"),
        "ASSUMPTION_V2_API_KEY": ("RUOLI_GPT_KEY", "GPT5_API_KEY"),
    }
    for target, sources in aliases.items():
        if os.environ.get(target) and not override:
            continue
        for source in sources:
            value = os.environ.get(source, "").strip()
            if value:
                os.environ[target] = value
                break
    os.environ.setdefault("ASSUMPTION_V2_MODEL", LOCKED_MODEL)
    model = configured_model(enforce_policy=False)
    allow_alternate = alternate_model_allowed()
    return {
        "base_url_present": bool(os.environ.get("ASSUMPTION_V2_API_BASE", "").strip()),
        "api_key_present": bool(os.environ.get("ASSUMPTION_V2_API_KEY", "").strip()),
        "model": model,
        "model_policy_passed": model == LOCKED_MODEL or allow_alternate,
        "alternate_model_allowed": allow_alternate,
        "secret_value_persisted": False,
    }
