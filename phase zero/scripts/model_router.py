"""Model router — tier-aware client factory.

Cheap tier (high-volume bulk labor: labeling, judging, answers):
  - gemini-3.5-flash-low (Google family)
  - claude-haiku-4-5-20251001 (Anthropic family)
  - gpt-5.4-mini (OpenAI family)

Expensive tier (reasoning, design, distillation):
  - claude-opus-4-6 (Anthropic)
  - gpt-5.5 (OpenAI)
"""

import os
from pathlib import Path
from openai import OpenAI

try:
    from dotenv import load_dotenv
    project_root = Path(__file__).resolve().parent.parent.parent
    for candidate in (project_root / ".env", project_root / "phase zero" / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)
    load_dotenv(override=False)
except ImportError:
    pass


# ---- endpoint config ----
# Prefer the live RUOLI_* env vars (current convention in ~/.api_keys);
# fall back to the legacy CLAUDE_PROXY_API_KEY / GPT5_API_KEY names
# (still set in older phase .env files) so older callers don't break.
_RUOLI_BASE_DEFAULT = os.environ.get("RUOLI_BASE_URL", "https://ruoli.dev") + "/v1"
_CLAUDE_BASE = os.environ.get("CLAUDE_PROXY_BASE_URL", _RUOLI_BASE_DEFAULT)
_CLAUDE_KEY = os.environ.get("RUOLI_CLAUDE_KEY") or os.environ.get("CLAUDE_PROXY_API_KEY", "")
_GPT_BASE = os.environ.get("GPT5_BASE_URL", _RUOLI_BASE_DEFAULT)
_GPT_KEY = os.environ.get("RUOLI_GPT_KEY") or os.environ.get("GPT5_API_KEY", "")
_GEMINI_BASE = os.environ.get("GEMINI_PROXY_BASE_URL", _RUOLI_BASE_DEFAULT)
_GEMINI_KEY = os.environ.get("RUOLI_GEMINI_KEY") or os.environ.get("GEMINI_PROXY_API_KEY", _GPT_KEY)
# deepseek lives on a different ruoli.dev channel/group
_DEEPSEEK_BASE = os.environ.get("DEEPSEEK_BASE_URL", "https://ruoli.dev/v1")
_DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
# Google official Gemini API (OpenAI-compatible endpoint) — use as
# headroom-control fallback when proxy-routed models saturate
_GOOGLE_BASE = os.environ.get("GOOGLE_GEMINI_BASE_URL",
                                  "https://generativelanguage.googleapis.com/v1beta/openai")
_GOOGLE_KEY = os.environ.get("GOOGLE_GEMINI_KEY", "")
try:
    _REQUEST_TIMEOUT = float(os.environ.get("MODEL_ROUTER_TIMEOUT", "60"))
except ValueError:
    _REQUEST_TIMEOUT = 60.0


class UnifiedClient:
    """Uniform .generate() interface across all vendor endpoints."""
    def __init__(self, model, base_url, api_key, family):
        self.model = model
        self.family = family
        self.provider = f"{family}/{base_url}"
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=_REQUEST_TIMEOUT)

    def generate(self, prompt, max_tokens=2000, temperature=0.3):
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = resp.choices[0].message.content or ""
        return {"text": text, "model": resp.model,
                "usage": resp.usage.model_dump() if resp.usage else {}}


# ---- tier presets ----

CHEAP_MODELS = {
    "gemini": (os.environ.get("GEMINI_PROXY_MODEL", "gemini-3.5-flash-low"), _GEMINI_BASE, _GEMINI_KEY),
    "gemini_flash_low": (os.environ.get("GEMINI_FLASH_LOW_MODEL", "gemini-3.5-flash-low"), _GEMINI_BASE, _GEMINI_KEY),
    "gemini_pro": (os.environ.get("GEMINI_PRO_MODEL", "gemini-3.1-pro"), _GEMINI_BASE, _GEMINI_KEY),
    "claude_haiku": (os.environ.get("CLAUDE_HAIKU_MODEL", "claude-haiku-4-5-20251001"), _CLAUDE_BASE, _CLAUDE_KEY),
    "gpt_mini": (os.environ.get("GPT_MINI_MODEL", "gpt-5.4-mini"), _GPT_BASE, _GPT_KEY),
    # weaker tier — for headroom-sensitive experiments where stronger
    # cheap-tier models saturate the task
    "deepseek_flash": ("deepseek-v4-flash", _DEEPSEEK_BASE, _DEEPSEEK_KEY),
    "gemini25_flash": ("gemini-2.5-flash", _GOOGLE_BASE, _GOOGLE_KEY),
}

EXPENSIVE_MODELS = {
    "claude_opus": (os.environ.get("CLAUDE_OPUS_MODEL", "claude-opus-4-6"), _CLAUDE_BASE, _CLAUDE_KEY),
    "gpt5": (os.environ.get("GPT5_EXPENSIVE_MODEL", "gpt-5.5"), _GPT_BASE, _GPT_KEY),
    "gpt55": (os.environ.get("GPT55_MODEL", "gpt-5.5"), _GPT_BASE, _GPT_KEY),
}


def cheap(name="gemini"):
    """Return a cheap-tier client by name."""
    if name not in CHEAP_MODELS:
        raise ValueError(f"unknown cheap model {name}; options: {list(CHEAP_MODELS)}")
    model, base, key = CHEAP_MODELS[name]
    if not key:
        raise RuntimeError(f"No API key for {name}; check .env")
    return UnifiedClient(model, base, key, name)


def expensive(name="claude_opus"):
    if name not in EXPENSIVE_MODELS:
        raise ValueError(f"unknown expensive model {name}; options: {list(EXPENSIVE_MODELS)}")
    model, base, key = EXPENSIVE_MODELS[name]
    if not key:
        raise RuntimeError(f"No API key for {name}")
    return UnifiedClient(model, base, key, name)


def cheap_panel():
    """Return 3 cheap clients from 3 families — for inter-rater κ."""
    return [cheap(n) for n in ("gemini", "claude_haiku", "gpt_mini")]


if __name__ == "__main__":
    # Smoke-test everything
    for name in list(CHEAP_MODELS) + list(EXPENSIVE_MODELS):
        try:
            c = cheap(name) if name in CHEAP_MODELS else expensive(name)
            r = c.generate("Reply exactly: PING", max_tokens=10, temperature=0.0)
            print(f"  {name:15s} {c.model:30s} ✓ {r['text'][:30]}")
        except Exception as e:
            print(f"  {name:15s} ✗ {str(e)[:80]}")
