from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..events import JsonlEventSink
from ..models import SplitName, stable_hash
from ..secure_env import (
    configured_model,
    configured_skilllearn_provider_mode,
    load_dotenv,
    map_legacy_model_env,
)
from .skilllearn_lifecycle import (
    SkillLearnPrebuiltImageCache,
    SkillLearnSubprocessBackend,
    SkillLearnTrialRequest,
    TrialVariant,
)
from .skilllearnbench import SkillLearnBenchAdapter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one diagnostic raw SkillLearnBench trial without retry or learning."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--item-id", required=True)
    parser.add_argument("--allowed-ipv4", action="append", required=True)
    parser.add_argument("--trials-dir", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()

    load_dotenv(args.env_file)
    map_legacy_model_env()
    os.environ["ASSUMPTION_V2_API_ALLOWED_IPV4S"] = ",".join(args.allowed_ipv4)
    os.environ["ASSUMPTION_V2_SKILLLEARN_CACHE_ONLY"] = "1"
    root = args.root.expanduser().resolve()
    adapter = SkillLearnBenchAdapter(root)
    item = next(
        (
            row
            for row in adapter.discover()
            if row.id == args.item_id and row.family == args.family
        ),
        None,
    )
    if item is None:
        raise ValueError("canary item is absent from the local benchmark inventory")
    sink = JsonlEventSink(args.events)
    cache = SkillLearnPrebuiltImageCache(root, cache_only=True, event_sink=sink)
    model = configured_model()
    provider_mode = configured_skilllearn_provider_mode()
    backend = SkillLearnSubprocessBackend(
        root,
        model=model,
        max_steps=args.max_steps,
        provider_mode=provider_mode,
        trials_dir=args.trials_dir,
        record_upstream=True,
        prebuilt_cache=cache,
        event_sink=sink,
    )
    request = SkillLearnTrialRequest(
        item_id=args.item_id,
        family=args.family,
        split=SplitName.TRAIN,
        variant=TrialVariant.POLICY_OFF,
        evaluator_epoch="diagnostic-canary-only",
        pair_id="diagnostic-canary-only",
        repeat=0,
        agent_id="codex",
        model=model,
        max_steps=args.max_steps,
        manifest_hash=stable_hash(
            {
                "diagnostic": True,
                "family": args.family,
                "item_id": args.item_id,
            }
        ),
    )
    observation = backend.run(
        request,
        skill_source_dir=None,
        trace_id=f"diagnostic-canary:{request.request_hash[:20]}",
    )
    payload = {
        "report_version": "skilllearn_single_trial_canary_v1",
        "claim_eligible": False,
        "learning_enabled": False,
        "retry_enabled": False,
        "observation": observation.to_dict(),
        "raw_content_persisted": False,
    }
    payload["report_hash"] = stable_hash(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not observation.valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
