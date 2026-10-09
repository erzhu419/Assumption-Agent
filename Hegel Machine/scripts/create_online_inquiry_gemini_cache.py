#!/usr/bin/env python3
"""Cache the common inquiry instructions and tools in the official Gemini API.

Set GEMINI_API_KEY, then create a cache for the explicitly chosen model:
 python scripts/create_online_inquiry_gemini_cache.py --model gemini-3.5-flash-lite \
   --output artifacts/inquiry_gemini_cache.json
Pass that file to run_online_inquiry_llm.py with --gemini-cache-file.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.chat_completions_code_provider import CHAT_INSTRUCTIONS
from hegel_machine.cognition.code_agent import _ACTION_SCHEMA
from hegel_machine.cognition.online_inquiry_llm import INTERFACE


def cache_request(model, ttl_seconds):
    return {
        "model": "models/" + model.removeprefix("models/"),
        "displayName": "hegel-inquiry-shared-interface",
        "ttl": f"{ttl_seconds}s",
        "systemInstruction": {"parts": [{"text": CHAT_INSTRUCTIONS}]},
        "contents": [{"role": "user", "parts": [
            {"text": json.dumps({"task_interface": INTERFACE})}]}],
        "tools": [{"functionDeclarations": [{
            "name": "research_action",
            "description": "Submit one laboratory or Python action for the host to execute.",
            "parametersJsonSchema": _ACTION_SCHEMA}]}],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--env-key", default="GEMINI_API_KEY")
    parser.add_argument("--ttl-seconds", type=int, default=7200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    body = cache_request(args.model, args.ttl_seconds)
    request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/cachedContents",
        data=json.dumps(body).encode(), headers={
            "x-goog-api-key": os.environ[args.env_key], "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.loads(response.read())
    metadata = {name: result[name] for name in ("name", "model", "expireTime", "usageMetadata")
                if name in result}
    metadata["task_interface"] = INTERFACE
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({name: value for name, value in metadata.items() if name != "task_interface"}))


if __name__ == "__main__":
    main()
