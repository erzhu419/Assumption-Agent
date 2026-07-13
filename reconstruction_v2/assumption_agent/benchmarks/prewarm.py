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
    PREBUILT_IMAGE_POLICY_VERSION,
    SHARED_CODEX_CLI_VERSION,
    SkillLearnPrebuiltImageCache,
    SkillLearnSubprocessBackend,
    codex_action_supervisor_hash,
    shared_codex_agent_runtime_key,
)
from .codex_action_budget import CODEX_ACTION_BUDGET_POLICY_VERSION
from .docker_egress import DEPENDENCY_CACHE_POLICY_VERSION
from .offline_verifier import (
    OFFLINE_VERIFIER_POLICY_VERSION,
    offline_verifier_profile_for_family,
    offline_verifier_runtime_key,
)


LEGACY_DEVELOPMENT_PREWARM_VERSION = (
    "all_manifest_images_and_offline_verifiers_v3"
)
DEVELOPMENT_PREWARM_VERSION = "all_manifest_images_and_offline_verifiers_v4"


def development_prewarm_version_for_protocol(
    protocol_version: object,
) -> str | None:
    return {
        "3.1.0": LEGACY_DEVELOPMENT_PREWARM_VERSION,
        "3.2.0": LEGACY_DEVELOPMENT_PREWARM_VERSION,
        "3.3.0": LEGACY_DEVELOPMENT_PREWARM_VERSION,
        "3.4.0": DEVELOPMENT_PREWARM_VERSION,
        "3.5.0": DEVELOPMENT_PREWARM_VERSION,
        "3.6.0": DEVELOPMENT_PREWARM_VERSION,
        "3.7.0": DEVELOPMENT_PREWARM_VERSION,
        "3.8.0": DEVELOPMENT_PREWARM_VERSION,
        "3.9.0": DEVELOPMENT_PREWARM_VERSION,
        "3.10.0": DEVELOPMENT_PREWARM_VERSION,
        "3.11.0": DEVELOPMENT_PREWARM_VERSION,
    }.get(str(protocol_version or ""))


def prewarm_development_images(
    *,
    benchmark_root: str | Path,
    manifest: SplitManifest,
    events_path: str | Path,
    parallel_workers: int = 4,
    attempts: int = 3,
    trial_provider_mode: str = "openai_compatible",
) -> dict[str, Any]:
    if parallel_workers <= 0:
        raise ValueError("parallel_workers must be positive")
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    selected_ids = (
        *manifest.train_ids,
        *manifest.validation_ids,
        *manifest.test_ids,
    )
    preflight = build_preflight(
        benchmark_root,
        trial_provider_mode=trial_provider_mode,
        item_ids=selected_ids,
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
                    image, verifier_runtime = backend.prewarm_trial_environment(
                        family=manifest.family_by_id[item_id],
                        item_id=item_id,
                        trace_id=trace_id,
                    )
                    verifier_runtime_mode = (
                        "local_profile"
                        if verifier_runtime is not None
                        else "native_image"
                    )
                    row = {
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
                        "verifier_runtime_mode": verifier_runtime_mode,
                        "offline_verifier_profile_id": (
                            verifier_runtime.profile.profile_id
                            if verifier_runtime is not None
                            else None
                        ),
                        "offline_verifier_profile_hash": (
                            verifier_runtime.profile.profile_hash
                            if verifier_runtime is not None
                            else None
                        ),
                        "offline_verifier_runtime_key": (
                            verifier_runtime.runtime_key
                            if verifier_runtime is not None
                            else None
                        ),
                        "verifier_runtime_network": "none",
                        "error_type": None,
                        "error_message_hash": None,
                    }
                    sink.emit(
                        Event(
                            event="skilllearn_development_prewarm_completed",
                            stage="benchmark.skilllearn.prewarm",
                            trace_id=trace_id,
                            payload={
                                "item_id_hash": row["item_id_hash"],
                                "family_hash": row["family_hash"],
                                "attempt": attempt,
                                "prebuilt_image_key": image.cache_key,
                                "prebuilt_image_id": image.image_id,
                                "agent_runtime_key": image.agent_runtime_key,
                                "verifier_runtime_mode": verifier_runtime_mode,
                                "offline_verifier_profile_hash": row[
                                    "offline_verifier_profile_hash"
                                ],
                                "offline_verifier_runtime_key": row[
                                    "offline_verifier_runtime_key"
                                ],
                                "verifier_runtime_network": "none",
                                "secret_value_persisted": False,
                                "raw_content_persisted": False,
                            },
                        )
                    )
                    return row
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
                "verifier_runtime_mode": "",
                "offline_verifier_profile_id": None,
                "offline_verifier_profile_hash": None,
                "offline_verifier_runtime_key": None,
                "verifier_runtime_network": "none",
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
        "split_names": ["train", "validation", "test"],
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
        "dependency_cache_policy": DEPENDENCY_CACHE_POLICY_VERSION,
        "dependency_cache_only_enforced": True,
        "agent_runtime_policy": PREBUILT_IMAGE_POLICY_VERSION,
        "agent_runtime_key": shared_codex_agent_runtime_key(),
        "agent_runtime_version": SHARED_CODEX_CLI_VERSION,
        "codex_action_supervisor_policy": (
            CODEX_ACTION_BUDGET_POLICY_VERSION
        ),
        "codex_action_supervisor_sha256": codex_action_supervisor_hash(),
        "offline_verifier_policy": OFFLINE_VERIFIER_POLICY_VERSION,
        "offline_verifier_runtime_network": "none",
        "offline_verifier_runtime_network_fallback_allowed": False,
        "local_profile_item_count": sum(
            row["verifier_runtime_mode"] == "local_profile" for row in rows
        ),
        "native_image_verifier_item_count": sum(
            row["verifier_runtime_mode"] == "native_image" for row in rows
        ),
        "unique_offline_verifier_profile_count": len(
            {
                str(row["offline_verifier_profile_hash"])
                for row in rows
                if row["passed"] and row["offline_verifier_profile_hash"]
            }
        ),
        "unique_offline_verifier_runtime_count": len(
            {
                str(row["offline_verifier_runtime_key"])
                for row in rows
                if row["passed"] and row["offline_verifier_runtime_key"]
            }
        ),
        "offline_verifier_profile_set_hash": stable_hash(
            sorted(
                {
                    str(row["offline_verifier_profile_hash"])
                    for row in rows
                    if row["passed"] and row["offline_verifier_profile_hash"]
                }
            )
        ),
        "offline_verifier_runtime_set_hash": stable_hash(
            sorted(
                {
                    str(row["offline_verifier_runtime_key"])
                    for row in rows
                    if row["passed"] and row["offline_verifier_runtime_key"]
                }
            )
        ),
        "online_build_attempted": False,
        "passed": passed,
        "items": rows,
        "test_infrastructure_inspected": bool(manifest.test_ids),
        "sealed_test_scoring_performed": False,
        "sealed_test_bytes_exposed_to_model": False,
        "secret_value_persisted": False,
        "raw_content_persisted": False,
    }
    payload["receipt_hash"] = stable_hash(payload)
    return payload


def validate_development_prewarm_receipt(
    receipt: Mapping[str, Any],
    *,
    manifest: SplitManifest,
    expected_version: str = DEVELOPMENT_PREWARM_VERSION,
) -> str:
    declared_hash = str(receipt.get("receipt_hash") or "")
    calculated_hash = stable_hash(
        {key: value for key, value in receipt.items() if key != "receipt_hash"}
    )
    if not declared_hash or declared_hash != calculated_hash:
        raise ValueError("development prewarm receipt hash mismatch")
    expected = {
        "prewarm_version": expected_version,
        "manifest_hash": manifest.manifest_hash,
        "split_names": ["train", "validation", "test"],
        "selected_item_set_hash": _selected_item_set_hash(manifest),
        "selected_item_count": (
            len(manifest.train_ids)
            + len(manifest.validation_ids)
            + len(manifest.test_ids)
        ),
        "completed_item_count": (
            len(manifest.train_ids)
            + len(manifest.validation_ids)
            + len(manifest.test_ids)
        ),
        "failed_item_count": 0,
        "dependency_cache_policy": DEPENDENCY_CACHE_POLICY_VERSION,
        "dependency_cache_only_enforced": True,
        "offline_verifier_policy": OFFLINE_VERIFIER_POLICY_VERSION,
        "offline_verifier_runtime_network": "none",
        "offline_verifier_runtime_network_fallback_allowed": False,
        "online_build_attempted": False,
        "passed": True,
        "secret_value_persisted": False,
        "raw_content_persisted": False,
    }
    if expected_version == DEVELOPMENT_PREWARM_VERSION:
        expected.update(
            {
                "test_infrastructure_inspected": bool(manifest.test_ids),
                "sealed_test_scoring_performed": False,
                "sealed_test_bytes_exposed_to_model": False,
                "agent_runtime_policy": PREBUILT_IMAGE_POLICY_VERSION,
                "agent_runtime_key": shared_codex_agent_runtime_key(),
                "agent_runtime_version": SHARED_CODEX_CLI_VERSION,
                "codex_action_supervisor_policy": (
                    CODEX_ACTION_BUDGET_POLICY_VERSION
                ),
                "codex_action_supervisor_sha256": (
                    codex_action_supervisor_hash()
                ),
            }
        )
    elif expected_version == LEGACY_DEVELOPMENT_PREWARM_VERSION:
        expected["test_content_accessed"] = False
    else:
        raise ValueError("development prewarm version is unsupported")
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"development prewarm receipt mismatch: {key}")
    rows = receipt.get("items")
    if not isinstance(rows, list) or len(rows) != expected["selected_item_count"]:
        raise ValueError("development prewarm item rows are incomplete")
    expected_item_hashes = {
        stable_hash({"item_id": item_id})
        for item_id in (
            *manifest.train_ids,
            *manifest.validation_ids,
            *manifest.test_ids,
        )
    }
    expected_profile_by_item_hash = {
        stable_hash({"item_id": item_id}): offline_verifier_profile_for_family(
            manifest.family_by_id[item_id]
        )
        for item_id in (
            *manifest.train_ids,
            *manifest.validation_ids,
            *manifest.test_ids,
        )
    }
    observed_item_hashes: set[str] = set()
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or not row.get("passed")
            or not row.get("prebuilt_image_key")
            or not row.get("prebuilt_image_id")
            or not row.get("agent_runtime_key")
            or row.get("verifier_runtime_network") != "none"
        ):
            raise ValueError("development prewarm item provenance is incomplete")
        item_hash = str(row.get("item_id_hash") or "")
        if not item_hash or item_hash in observed_item_hashes:
            raise ValueError("development prewarm item hashes are incomplete")
        observed_item_hashes.add(item_hash)
        if expected_version == DEVELOPMENT_PREWARM_VERSION and (
            row.get("agent_runtime_key") != shared_codex_agent_runtime_key()
            or row.get("agent_runtime_version") != SHARED_CODEX_CLI_VERSION
        ):
            raise ValueError(
                "development prewarm agent runtime does not match the active runtime"
            )
        mode = row.get("verifier_runtime_mode")
        expected_profile = expected_profile_by_item_hash.get(item_hash)
        if mode == "local_profile":
            if (
                not row.get("offline_verifier_profile_id")
                or not row.get("offline_verifier_profile_hash")
                or not row.get("offline_verifier_runtime_key")
            ):
                raise ValueError(
                    "development prewarm offline verifier provenance is incomplete"
                )
            if expected_profile is None or (
                row.get("offline_verifier_profile_id") != expected_profile.profile_id
                or row.get("offline_verifier_profile_hash")
                != expected_profile.profile_hash
                or row.get("offline_verifier_runtime_key")
                != offline_verifier_runtime_key(profile=expected_profile)
            ):
                raise ValueError(
                    "development prewarm offline verifier profile does not match family"
                )
        elif mode == "native_image":
            if any(
                row.get(key) is not None
                for key in (
                    "offline_verifier_profile_id",
                    "offline_verifier_profile_hash",
                    "offline_verifier_runtime_key",
                )
            ):
                raise ValueError(
                    "development prewarm native verifier provenance is malformed"
                )
            if expected_profile is not None:
                raise ValueError(
                    "development prewarm declared profile was not prewarmed"
                )
        else:
            raise ValueError("development prewarm verifier runtime mode is invalid")
    if observed_item_hashes != expected_item_hashes:
        raise ValueError("development prewarm item hashes do not match the manifest")
    local_profile_count = sum(
        row.get("verifier_runtime_mode") == "local_profile" for row in rows
    )
    native_image_count = len(rows) - local_profile_count
    if receipt.get("local_profile_item_count") != local_profile_count:
        raise ValueError("development prewarm local profile count mismatch")
    if receipt.get("native_image_verifier_item_count") != native_image_count:
        raise ValueError("development prewarm native verifier count mismatch")
    profile_hashes = {
        str(row["offline_verifier_profile_hash"])
        for row in rows
        if row.get("offline_verifier_profile_hash")
    }
    runtime_keys = {
        str(row["offline_verifier_runtime_key"])
        for row in rows
        if row.get("offline_verifier_runtime_key")
    }
    if receipt.get("unique_offline_verifier_profile_count") != len(profile_hashes):
        raise ValueError("development prewarm offline verifier profile count mismatch")
    if receipt.get("unique_offline_verifier_runtime_count") != len(runtime_keys):
        raise ValueError("development prewarm offline verifier runtime count mismatch")
    if receipt.get("offline_verifier_profile_set_hash") != stable_hash(
        sorted(profile_hashes)
    ):
        raise ValueError("development prewarm offline verifier profile set mismatch")
    if receipt.get("offline_verifier_runtime_set_hash") != stable_hash(
        sorted(runtime_keys)
    ):
        raise ValueError("development prewarm offline verifier runtime set mismatch")
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
            "test_item_hashes": sorted(
                stable_hash({"item_id": item_id}) for item_id in manifest.test_ids
            ),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prebuild every manifest image before model execution."
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
        choices=("openai_compatible",),
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
