"""Model router — tier-aware client factory.

Cheap tier (high-volume bulk labor: labeling, judging, answers):
  - gemini-3-flash (Google family)
  - claude-haiku-4-5-20251001 (Anthropic family)
  - gpt-5.4-mini (OpenAI family)

Expensive tier (reasoning, design, distillation):
  - claude-opus-4-6 (Anthropic)
  - gpt-5.4 (OpenAI)
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
_CLAUDE_BASE = os.environ.get("CLAUDE_PROXY_BASE_URL", "https://ruoli.dev/v1")
_CLAUDE_KEY = os.environ.get("CLAUDE_PROXY_API_KEY", "")
_GPT_BASE = os.environ.get("GPT5_BASE_URL", "https://ruoli.dev/v1")
_GPT_KEY = os.environ.get("GPT5_API_KEY", "")
_GEMINI_BASE = os.environ.get("GEMINI_PROXY_BASE_URL", "https://ruoli.dev/v1")
_GEMINI_KEY = os.environ.get("GEMINI_PROXY_API_KEY", _GPT_KEY)


class UnifiedClient:
    """Uniform .generate() interface across all vendor endpoints."""
    def __init__(self, model, base_url, api_key, family):
        self.model = model
        self.family = family
        self.provider = f"{family}/{base_url}"
        self._client = OpenAI(base_url=base_url, api_key=api_key)

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
    "gemini": ("gemini-3-flash", _GEMINI_BASE, _GEMINI_KEY),
    "claude_haiku": ("claude-haiku-4-5-20251001", _CLAUDE_BASE, _CLAUDE_KEY),
    "gpt_mini": ("gpt-5.4-mini", _GPT_BASE, _GPT_KEY),
}

EXPENSIVE_MODELS = {
    "claude_opus": ("claude-opus-4-6", _CLAUDE_BASE, _CLAUDE_KEY),
    "gpt5": ("gpt-5.4", _GPT_BASE, _GPT_KEY),
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
