from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..events import JsonlEventSink
from ..models import SplitName, stable_hash
from ..secure_env import (
    configured_api_origin,
    configured_model,
    configured_skilllearn_provider_mode,
    load_dotenv,
    map_legacy_model_env,
)
from ..splits import SplitManifest
from .docker_egress import configured_trial_network_byte_limit
from .paper_protocol import PaperProtocol
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
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--item-id", required=True)
    parser.add_argument("--trials-dir", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--diagnostic-max-steps", type=int)
    args = parser.parse_args()

    load_dotenv(args.env_file)
    map_legacy_model_env()
    protocol = PaperProtocol.read(args.protocol)
    manifest = SplitManifest.read(args.manifest)
    protocol_root = protocol.path.parent.parent
    if args.manifest.expanduser().resolve() != (
        protocol_root / str(protocol.payload["primary_manifest"])
    ).resolve():
        raise ValueError("diagnostic canary requires the frozen primary manifest")
    if configured_model() != protocol.payload["model"]:
        raise ValueError("diagnostic canary model does not match the protocol")
    if configured_skilllearn_provider_mode() != protocol.payload["trial_provider_mode"]:
        raise ValueError("diagnostic canary provider mode does not match the protocol")
    if configured_api_origin() != protocol.payload["provider_endpoint_origin"]:
        raise ValueError("diagnostic canary provider endpoint does not match the protocol")
    if configured_trial_network_byte_limit() != protocol.payload["execution"][
        "trial_network_byte_limit"
    ]:
        raise ValueError("diagnostic canary network cap does not match the protocol")
    os.environ["ASSUMPTION_V2_API_ALLOWED_IPV4S"] = ",".join(
        str(value) for value in protocol.payload["provider_endpoint_ipv4s"]
    )
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
    if args.item_id not in manifest.train_ids or manifest.family_by_id.get(
        args.item_id
    ) != args.family:
        raise PermissionError("diagnostic canary item is not in the frozen train split")
    sink = JsonlEventSink(args.events)
    cache = SkillLearnPrebuiltImageCache(root, cache_only=True, event_sink=sink)
    model = str(protocol.payload["model"])
    provider_mode = str(protocol.payload["trial_provider_mode"])
    protocol_max_steps = int(protocol.payload["max_steps"])
    max_steps = (
        protocol_max_steps
        if args.diagnostic_max_steps is None
        else args.diagnostic_max_steps
    )
    if not 1 <= max_steps <= protocol_max_steps:
        raise ValueError(
            "diagnostic max steps must be within the frozen protocol budget"
        )
    backend = SkillLearnSubprocessBackend(
        root,
        model=model,
        max_steps=max_steps,
        provider_mode=provider_mode,
        trials_dir=args.trials_dir,
        record_upstream=True,
        prebuilt_cache=cache,
        codex_agent_execution_policy=protocol.codex_agent_execution_policy,
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
        max_steps=max_steps,
        manifest_hash=manifest.manifest_hash,
        codex_agent_execution_policy_hash=(
            protocol.codex_agent_execution_policy.policy_hash
        ),
    )
    observation = backend.run(
        request,
        skill_source_dir=None,
        trace_id=f"diagnostic-canary:{request.request_hash[:20]}",
    )
    payload = {
        "report_version": "skilllearn_single_trial_canary_v1",
        "paper_protocol_id": protocol.id,
        "paper_protocol_hash": protocol.protocol_hash,
        "manifest_hash": manifest.manifest_hash,
        "codex_agent_execution_policy": (
            protocol.codex_agent_execution_policy.to_dict()
        ),
        "codex_agent_execution_policy_hash": (
            protocol.codex_agent_execution_policy.policy_hash
        ),
        "protocol_max_steps": protocol_max_steps,
        "diagnostic_max_steps": max_steps,
        "claim_eligible": False,
        "learning_enabled": False,
        "retry_enabled": False,
        "upstream_trial_artifacts_persisted": True,
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
