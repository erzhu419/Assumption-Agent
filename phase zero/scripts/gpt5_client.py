"""
Minimal GPT-5.4 client via ruoli.dev NewAPI proxy.

For mining high-quality aphorism triggers (Phase 2 v2). Keeps Gemini Flash
as the cheap workhorse for everything else.
"""

import os
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_URL = os.environ.get("GPT5_BASE_URL", "https://ruoli.dev/v1")
API_KEY = os.environ.get("GPT5_API_KEY", "")
DEFAULT_MODEL = os.environ.get("GPT5_MODEL", "gpt-5.4")
if not API_KEY:
    raise RuntimeError("GPT5_API_KEY environment variable must be set")


class GPT5Client:
    """Thin wrapper with .generate(prompt) -> dict like our Gemini client."""

    def __init__(self, base_url: str = BASE_URL, api_key: str = API_KEY,
                 model: str = DEFAULT_MODEL):
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.provider = f"newapi/{base_url}"

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
    # Quick smoke test
    client = GPT5Client()
    print(f"model: {client.model}  base_url: {BASE_URL}")
    r = client.generate("Reply with exactly 'hello world' and nothing else.", max_tokens=30)
    print(f"output: {r['text']!r}")
    print(f"usage: {r.get('usage', {})}")
