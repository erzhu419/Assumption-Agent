#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assumption_agent.benchmarks.skilllearn_lifecycle import (
    PREBUILT_IMAGE_POLICY_VERSION,
    SHARED_CODEX_CLI_PACKAGE,
    SHARED_CODEX_CLI_VERSION,
    SkillLearnPrebuiltImageCache,
    SkillLearnSubprocessBackend,
    codex_action_supervisor_hash,
    shared_codex_agent_runtime_key,
)
from assumption_agent.events import JsonlEventSink
from assumption_agent.models import stable_hash


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Explicitly prepare the content-addressed Codex runtime before "
            "cache-only paper prewarm."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--item-id", required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-network-download", action="store_true")
    args = parser.parse_args()
    if not args.allow_network_download:
        raise PermissionError(
            "runtime preparation requires explicit --allow-network-download"
        )

    root = args.root.expanduser().resolve()
    sink = JsonlEventSink(args.events)
    cache = SkillLearnPrebuiltImageCache(
        root,
        cache_only=False,
        event_sink=sink,
    )
    backend = SkillLearnSubprocessBackend(root, event_sink=sink)
    runner = backend._load_runner()
    image = cache.ensure(
        family=args.family,
        item_id=args.item_id,
        agent_id="codex",
        runner=runner,
        trace_id="explicit-codex-runtime-preparation",
    )
    expected_runtime_key = shared_codex_agent_runtime_key()
    if image.agent_runtime_key != expected_runtime_key:
        raise RuntimeError("prepared Codex runtime key mismatch")
    payload = {
        "preparation_version": "explicit_codex_runtime_preparation_v1",
        "policy": PREBUILT_IMAGE_POLICY_VERSION,
        "codex_cli_package": SHARED_CODEX_CLI_PACKAGE,
        "codex_cli_version": SHARED_CODEX_CLI_VERSION,
        "codex_action_supervisor_sha256": codex_action_supervisor_hash(),
        "agent_runtime_key": image.agent_runtime_key,
        "agent_runtime_version": image.agent_runtime_version,
        "prebuilt_image_key": image.cache_key,
        "prebuilt_image_id": image.image_id,
        "prebuilt_image_reused": image.reused,
        "network_download_explicitly_authorized": True,
        "model_inference_performed": False,
        "scoring_performed": False,
        "claim_eligible": False,
        "test_content_accessed": False,
        "secret_value_persisted": False,
        "raw_content_persisted": False,
    }
    payload["receipt_hash"] = stable_hash(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
