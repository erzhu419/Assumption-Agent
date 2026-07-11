from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit


LOCKED_MODEL = "gpt-5.4-mini"
APPROVED_PAPER_MODELS = frozenset({LOCKED_MODEL})


def alternate_model_allowed() -> bool:
    return os.environ.get("ASSUMPTION_V2_ALLOW_ALTERNATE_MODEL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def paper_model_allowed(model: str) -> bool:
    return model.strip() in APPROVED_PAPER_MODELS


def configured_model(*, enforce_policy: bool = True) -> str:
    model = os.environ.get("ASSUMPTION_V2_MODEL", LOCKED_MODEL).strip() or LOCKED_MODEL
    if enforce_policy and not paper_model_allowed(model) and not alternate_model_allowed():
        raise RuntimeError(
            "reconstruction v2 only permits a protocol-approved paper model until "
            "ASSUMPTION_V2_ALLOW_ALTERNATE_MODEL=1"
        )
    return model


def configured_skilllearn_provider_mode() -> str:
    mode = os.environ.get(
        "ASSUMPTION_V2_SKILLLEARN_PROVIDER_MODE",
        "openai_compatible",
    ).strip().lower()
    if mode != "openai_compatible":
        raise ValueError("unsupported SkillLearn trial provider mode")
    return mode


def configured_api_origin() -> str:
    base_url = os.environ.get("ASSUMPTION_V2_API_BASE", "").strip()
    if not base_url:
        return ""
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("ASSUMPTION_V2_API_BASE must be an HTTP(S) URL")
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}"


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
        "model_policy_passed": paper_model_allowed(model) or allow_alternate,
        "alternate_model_allowed": allow_alternate,
        "secret_value_persisted": False,
    }
