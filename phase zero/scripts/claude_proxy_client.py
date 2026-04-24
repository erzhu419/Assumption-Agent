"""Claude (Anthropic) via ruoli.dev NewAPI proxy.

Used as a cross-model judge for paper Experiment 1 (cross-judge
validation of KEEP decisions). Distinct from the ANTHROPIC_API_KEY
path in llm_client.py which uses the official SDK.
"""

import os
from pathlib import Path
from openai import OpenAI

try:
    from dotenv import load_dotenv
    # Look for .env at project root (phase zero/scripts/../../) in addition to CWD
    project_root = Path(__file__).resolve().parent.parent.parent
    for candidate in (project_root / ".env", project_root / "phase zero" / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)
    load_dotenv(override=False)  # also try CWD
except ImportError:
    pass

BASE_URL = os.environ.get("CLAUDE_PROXY_BASE_URL", "https://ruoli.dev/v1")
API_KEY = os.environ.get("CLAUDE_PROXY_API_KEY", "")
DEFAULT_MODEL = os.environ.get("CLAUDE_PROXY_MODEL", "claude-opus-4-6")
if not API_KEY:
    raise RuntimeError("CLAUDE_PROXY_API_KEY environment variable must be set")


class ClaudeProxyClient:
    def __init__(self, base_url: str = BASE_URL, api_key: str = API_KEY,
                 model: str = DEFAULT_MODEL):
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.provider = f"newapi-claude/{base_url}"

    def generate(self, prompt: str, max_tokens: int = 2000,
                 temperature: float = 0.3) -> dict:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = resp.choices[0].message.content or ""
        return {"text": text,
                "model": resp.model,
                "usage": resp.usage.model_dump() if resp.usage else {}}


if __name__ == "__main__":
    client = ClaudeProxyClient()
    print(f"model: {client.model}")
    r = client.generate("Reply with exactly 'claude-ok' and nothing else.",
                        max_tokens=20)
    print(f"output: {r['text']!r}")
