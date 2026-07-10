from __future__ import annotations

import argparse
import concurrent.futures
import json
import queue
import time
from pathlib import Path
from typing import Any, Mapping

from ..events import Event, JsonlEventSink
from ..models import stable_hash
from ..secure_env import (
    configured_skilllearn_provider_mode,
    load_dotenv,
    map_legacy_model_env,
)
from ..splits import SplitManifest
from .preflight import build_preflight
from .skilllearn_lifecycle import (
    SkillLearnPrebuiltImageCache,
    SkillLearnSubprocessBackend,
)


DEVELOPMENT_PREWARM_VERSION = "train_validation_images_v1"


def prewarm_development_images(
    *,
    benchmark_root: str | Path,
    manifest: SplitManifest,
    events_path: str | Path,
    parallel_workers: int = 4,
    attempts: int = 3,
    trial_provider_mode: str = "codex_subscription",
) -> dict[str, Any]:
    if parallel_workers <= 0:
        raise ValueError("parallel_workers must be positive")
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    selected_ids = (*manifest.train_ids, *manifest.validation_ids)
    all_manifest_ids = (*selected_ids, *manifest.test_ids)
    preflight = build_preflight(
        benchmark_root,
        trial_provider_mode=trial_provider_mode,
        item_ids=all_manifest_ids,
    )
    if preflight["blockers"]:
        raise RuntimeError(f"development prewarm preflight failed: {preflight['blockers']}")

    sink = JsonlEventSink(events_path)
    cache = SkillLearnPrebuiltImageCache(benchmark_root, event_sink=sink)
    backends: queue.Queue[SkillLearnSubprocessBackend] = queue.Queue()
    for _ in range(parallel_workers):
        backends.put(
            SkillLearnSubprocessBackend(
                benchmark_root,
                provider_mode=trial_provider_mode,
                record_upstream=False,
                prebuilt_cache=cache,
                event_sink=sink,
            )
        )

    def warm_one(item_id: str) -> dict[str, Any]:
        backend = backends.get()
        try:
            last_error: Exception | None = None
            for attempt in range(1, attempts + 1):
                trace_id = (
                    f"prewarm-{manifest.manifest_hash[:12]}:"
                    f"{stable_hash({'item_id': item_id})[:20]}:attempt-{attempt}"
                )
                sink.emit(
                    Event(
                        event="skilllearn_development_prewarm_attempted",
                        stage="benchmark.skilllearn.prewarm",
                        trace_id=trace_id,
                        payload={
                            "item_id_hash": stable_hash({"item_id": item_id}),
                            "family_hash": stable_hash(
                                {"family": manifest.family_by_id[item_id]}
                            ),
                            "attempt": attempt,
                            "manifest_hash": manifest.manifest_hash,
                        },
                    )
                )
                try:
                    image = backend.prewarm_environment(
                        family=manifest.family_by_id[item_id],
                        item_id=item_id,
                        trace_id=trace_id,
                    )
                    return {
                        "item_id_hash": stable_hash({"item_id": item_id}),
                        "family_hash": stable_hash(
                            {"family": manifest.family_by_id[item_id]}
                        ),
                        "attempt_count": attempt,
                        "passed": True,
                        "prebuilt_image_key": image.cache_key,
                        "prebuilt_image_id": image.image_id,
                        "agent_runtime_key": image.agent_runtime_key,
                        "agent_runtime_version": image.agent_runtime_version,
                        "error_type": None,
                        "error_message_hash": None,
                    }
                except Exception as exc:  # Infrastructure evidence, never task evidence.
                    last_error = exc
                    sink.emit(
                        Event(
                            event="skilllearn_development_prewarm_failed",
                            stage="benchmark.skilllearn.prewarm",
                            trace_id=trace_id,
                            payload={
                                "item_id_hash": stable_hash({"item_id": item_id}),
                                "attempt": attempt,
                                "error_type": type(exc).__name__,
                                "error_message_hash": stable_hash({"message": str(exc)}),
                                "secret_value_persisted": False,
                            },
                        )
                    )
                    if attempt < attempts:
                        time.sleep(float(attempt))
            assert last_error is not None
            return {
                "item_id_hash": stable_hash({"item_id": item_id}),
                "family_hash": stable_hash(
                    {"family": manifest.family_by_id[item_id]}
                ),
                "attempt_count": attempts,
                "passed": False,
                "prebuilt_image_key": "",
                "prebuilt_image_id": "",
                "agent_runtime_key": "",
                "agent_runtime_version": "",
                "error_type": type(last_error).__name__,
                "error_message_hash": stable_hash({"message": str(last_error)}),
            }
        finally:
            backends.put(backend)

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        rows = list(executor.map(warm_one, selected_ids))

    passed = all(bool(row["passed"]) for row in rows)
    payload: dict[str, Any] = {
        "prewarm_version": DEVELOPMENT_PREWARM_VERSION,
        "manifest_hash": manifest.manifest_hash,
        "split_names": ["train", "validation"],
        "selected_item_set_hash": _selected_item_set_hash(manifest),
        "selected_item_count": len(selected_ids),
        "completed_item_count": len(rows),
        "passed_item_count": sum(bool(row["passed"]) for row in rows),
        "failed_item_count": sum(not bool(row["passed"]) for row in rows),
        "unique_image_count": len(
            {str(row["prebuilt_image_key"]) for row in rows if row["passed"]}
        ),
        "parallel_workers": parallel_workers,
        "maximum_attempts": attempts,
        "passed": passed,
        "items": rows,
        "test_content_accessed": False,
        "secret_value_persisted": False,
        "raw_content_persisted": False,
    }
    payload["receipt_hash"] = stable_hash(payload)
    return payload


def validate_development_prewarm_receipt(
    receipt: Mapping[str, Any],
    *,
    manifest: SplitManifest,
) -> str:
    declared_hash = str(receipt.get("receipt_hash") or "")
    calculated_hash = stable_hash(
        {key: value for key, value in receipt.items() if key != "receipt_hash"}
    )
    if not declared_hash or declared_hash != calculated_hash:
        raise ValueError("development prewarm receipt hash mismatch")
    expected = {
        "prewarm_version": DEVELOPMENT_PREWARM_VERSION,
        "manifest_hash": manifest.manifest_hash,
        "split_names": ["train", "validation"],
        "selected_item_set_hash": _selected_item_set_hash(manifest),
        "selected_item_count": len(manifest.train_ids) + len(manifest.validation_ids),
        "completed_item_count": len(manifest.train_ids) + len(manifest.validation_ids),
        "failed_item_count": 0,
        "passed": True,
        "test_content_accessed": False,
        "secret_value_persisted": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"development prewarm receipt mismatch: {key}")
    rows = receipt.get("items")
    if not isinstance(rows, list) or len(rows) != expected["selected_item_count"]:
        raise ValueError("development prewarm item rows are incomplete")
    if any(
        not isinstance(row, Mapping)
        or not row.get("passed")
        or not row.get("prebuilt_image_key")
        or not row.get("prebuilt_image_id")
        or not row.get("agent_runtime_key")
        for row in rows
    ):
        raise ValueError("development prewarm item provenance is incomplete")
    return declared_hash


def _selected_item_set_hash(manifest: SplitManifest) -> str:
    return stable_hash(
        {
            "train_item_hashes": sorted(
                stable_hash({"item_id": item_id}) for item_id in manifest.train_ids
            ),
            "validation_item_hashes": sorted(
                stable_hash({"item_id": item_id}) for item_id in manifest.validation_ids
            ),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prebuild every train/validation image before model execution."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parallel-workers", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument(
        "--trial-provider-mode",
        choices=("codex_subscription", "openai_compatible"),
    )
    parser.add_argument("--require-passed", action="store_true")
    args = parser.parse_args()
    load_dotenv(args.env_file)
    map_legacy_model_env()
    manifest = SplitManifest.read(args.manifest)
    receipt = prewarm_development_images(
        benchmark_root=args.root,
        manifest=manifest,
        events_path=args.events,
        parallel_workers=args.parallel_workers,
        attempts=args.attempts,
        trial_provider_mode=(
            args.trial_provider_mode or configured_skilllearn_provider_mode()
        ),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if args.require_passed and not receipt["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
