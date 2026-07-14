#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
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
from assumption_agent.benchmarks.prewarm import (
    FrozenTaskInputPrebuiltImageCache,
)
from assumption_agent.benchmarks.task_input_closure import (
    family_requires_task_input_closure,
)
from assumption_agent.benchmarks.task_input_freeze import (
    load_frozen_task_input_closure,
)
from assumption_agent.events import JsonlEventSink
from assumption_agent.models import stable_hash
from assumption_agent.splits import SplitManifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Explicitly prepare the content-addressed Codex runtime before "
            "cache-only paper prewarm."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--family")
    parser.add_argument("--item-id")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--scope",
        choices=("affected", "all"),
        default="all",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=0,
        help="0 uses one worker per selected item",
    )
    parser.add_argument("--task-input-cache-root", type=Path)
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
    if (args.family is None) != (args.item_id is None):
        raise ValueError("--family and --item-id must be supplied together")
    if args.family is not None and args.manifest is not None:
        raise ValueError("use either one item or --manifest selection")
    if args.family is None and args.manifest is None:
        raise ValueError("image preparation requires one item or --manifest")
    frozen_task_inputs = None
    if args.protocol is not None:
        protocol_payload = json.loads(args.protocol.read_text(encoding="utf-8"))
        if not isinstance(protocol_payload, dict):
            raise ValueError("paper protocol must contain one JSON object")
        frozen_task_inputs = load_frozen_task_input_closure(
            protocol_payload,
            project_root=args.project_root,
        )
    cache = (
        FrozenTaskInputPrebuiltImageCache(
            root,
            cache_only=False,
            event_sink=sink,
            frozen_task_inputs=frozen_task_inputs,
            task_input_cache_root=args.task_input_cache_root,
        )
        if frozen_task_inputs is not None
        else SkillLearnPrebuiltImageCache(
            root,
            cache_only=False,
            event_sink=sink,
        )
    )
    backend = SkillLearnSubprocessBackend(root, event_sink=sink)
    runner = backend._load_runner()
    if args.manifest is not None:
        manifest = SplitManifest.read(args.manifest)
        item_rows = [
            (item_id, manifest.family_by_id[item_id])
            for item_id in (
                *manifest.train_ids,
                *manifest.validation_ids,
                *manifest.test_ids,
            )
        ]
    else:
        item_rows = [(str(args.item_id), str(args.family))]
    if args.scope == "affected":
        item_rows = [
            row for row in item_rows if family_requires_task_input_closure(row[1])
        ]
    if not item_rows:
        raise ValueError("image preparation selection is empty")
    if args.parallel_workers < 0:
        raise ValueError("--parallel-workers must be nonnegative")
    parallel_workers = args.parallel_workers or len(item_rows)

    def prepare_one(row: tuple[str, str]):
        item_id, family = row
        image = cache.ensure(
            family=family,
            item_id=item_id,
            agent_id="codex",
            runner=runner,
            trace_id=(
                "explicit-codex-runtime-preparation:"
                f"{stable_hash({'item_id': item_id})[:20]}"
            ),
        )
        if (
            image.agent_runtime_key != shared_codex_agent_runtime_key()
            or image.agent_runtime_version != SHARED_CODEX_CLI_VERSION
        ):
            raise RuntimeError("prepared Codex runtime key mismatch")
        return {
            "item_id_hash": stable_hash({"item_id": item_id}),
            "family_hash": stable_hash({"family": family}),
            "prebuilt_image_key": image.cache_key,
            "prebuilt_image_id": image.image_id,
            "prebuilt_image_reused": image.reused,
            "task_input_closure_required": image.task_input_closure_required,
            "task_input_closure_hash": image.task_input_closure_hash,
            "task_input_integrity_receipt_hash": (
                image.task_input_integrity_receipt_hash
            ),
        }

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=parallel_workers
    ) as executor:
        image_rows = list(executor.map(prepare_one, item_rows))
    expected_runtime_key = shared_codex_agent_runtime_key()
    payload = {
        "preparation_version": "explicit_codex_runtime_and_image_preparation_v2",
        "policy": PREBUILT_IMAGE_POLICY_VERSION,
        "codex_cli_package": SHARED_CODEX_CLI_PACKAGE,
        "codex_cli_version": SHARED_CODEX_CLI_VERSION,
        "codex_action_supervisor_sha256": codex_action_supervisor_hash(),
        "agent_runtime_key": expected_runtime_key,
        "agent_runtime_version": SHARED_CODEX_CLI_VERSION,
        "selection_scope": args.scope,
        "selected_item_count": len(image_rows),
        "completed_item_count": len(image_rows),
        "parallel_workers": parallel_workers,
        "task_input_closure_policy": (
            frozen_task_inputs.policy if frozen_task_inputs is not None else None
        ),
        "task_input_freeze_hash": (
            frozen_task_inputs.freeze_hash
            if frozen_task_inputs is not None
            else None
        ),
        "items": image_rows,
        "network_download_explicitly_authorized": True,
        "model_inference_performed": False,
        "scoring_performed": False,
        "claim_eligible": False,
        "test_content_accessed": False,
        "secret_value_persisted": False,
        "raw_content_persisted": False,
    }
    if len(image_rows) == 1:
        payload.update(
            {
                "prebuilt_image_key": image_rows[0]["prebuilt_image_key"],
                "prebuilt_image_id": image_rows[0]["prebuilt_image_id"],
                "prebuilt_image_reused": image_rows[0][
                    "prebuilt_image_reused"
                ],
            }
        )
    payload["receipt_hash"] = stable_hash(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
