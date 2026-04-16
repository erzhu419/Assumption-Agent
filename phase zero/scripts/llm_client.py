"""
Unified LLM client — supports both Gemini and Claude.
Configure via .env file or environment variables.

Priority: GEMINI_API_KEY → ANTHROPIC_API_KEY (uses whichever is set)

.env example:
    LLM_PROVIDER=gemini          # or "claude"
    GEMINI_API_KEY=AIza...
    GEMINI_MODEL=gemini-2.5-flash
    # or
    ANTHROPIC_API_KEY=sk-ant-...
    ANTHROPIC_MODEL=claude-sonnet-4-20250514
"""

import os
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------

def _load_env():
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

_load_env()

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def detect_provider() -> str:
    explicit = os.environ.get("LLM_PROVIDER", "").lower()
    if explicit in ("gemini", "google"):
        return "gemini"
    if explicit in ("claude", "anthropic"):
        return "claude"
    # Auto-detect by which key is set
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    raise ValueError(
        "No LLM provider configured. Set GEMINI_API_KEY or ANTHROPIC_API_KEY "
        "in environment or in phase zero/.env"
    )

# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

class GeminiClient:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")

        try:
            from google import genai
            self.client = genai.Client(api_key=api_key)
            self._sdk = "google-genai"
        except ImportError:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self._sdk = "google-generativeai"
                self._genai = genai
            except ImportError:
                raise ImportError(
                    "Install Gemini SDK: pip install google-genai\n"
                    "  or: pip install google-generativeai"
                )

        self.model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.provider = "gemini"

    def generate(self, prompt: str, max_tokens: int = 4096,
                 temperature: float = 0.3) -> dict:
        """Returns {"text": str, "input_tokens": int, "output_tokens": int}"""

        if self._sdk == "google-genai":
            from google.genai import types
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=0  # Disable thinking to avoid token theft
                    ),
                ),
            )
            text = response.text
            usage = getattr(response, "usage_metadata", None)
            input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
            output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
        else:
            # google-generativeai (older SDK)
            model = self._genai.GenerativeModel(
                self.model,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                }
            )
            response = model.generate_content(prompt)
            text = response.text
            input_tokens = 0
            output_tokens = 0

        return {
            "text": text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


# ---------------------------------------------------------------------------
# Claude client
# ---------------------------------------------------------------------------

class ClaudeClient:
    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Install: pip install anthropic")

        self.client = Anthropic(api_key=api_key)
        self.model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        self.provider = "claude"

    def generate(self, prompt: str, max_tokens: int = 4096,
                 temperature: float = 0.3) -> dict:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        return {
            "text": text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_client():
    """Create LLM client based on configuration."""
    provider = detect_provider()
    if provider == "gemini":
        client = GeminiClient()
    else:
        client = ClaudeClient()
    print(f"LLM provider: {client.provider}, model: {client.model}")
    return client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_json_from_llm(raw: str):
    """Extract JSON from LLM response, handling markdown fences."""
    raw = raw.strip()
    # Strip ```json ... ``` or ``` ... ```
    if raw.startswith("```"):
        first_newline = raw.index("\n") if "\n" in raw else len(raw)
        raw = raw[first_newline + 1:]
    if raw.endswith("```"):
        raw = raw[:raw.rfind("```")]
    raw = raw.strip()
    return json.loads(raw)
