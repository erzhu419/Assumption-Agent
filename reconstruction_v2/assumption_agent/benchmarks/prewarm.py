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
from .task_input_closure import (
    TASK_INPUT_BUILD_CONTEXT_POLICY_VERSION,
    TASK_INPUT_CLOSURE_POLICY_VERSION,
    family_requires_task_input_closure,
)
from .task_input_freeze import (
    FrozenTaskInputClosure,
    expected_prewarm_closure_rows,
    load_frozen_task_input_closure,
    verify_current_task_input_closure,
)


LEGACY_DEVELOPMENT_PREWARM_VERSION = (
    "all_manifest_images_and_offline_verifiers_v3"
)
DEVELOPMENT_PREWARM_VERSION = "all_manifest_images_and_offline_verifiers_v4"
TASK_INPUT_CLOSURE_DEVELOPMENT_PREWARM_VERSION = (
    "all_manifest_images_offline_verifiers_and_public_inputs_v5"
)


class FrozenTaskInputPrebuiltImageCache(SkillLearnPrebuiltImageCache):
    """Fail closed against both the protocol ledger and a passed v5 prewarm."""

    def __init__(
        self,
        benchmark_root: str | Path,
        *,
        frozen_task_inputs: FrozenTaskInputClosure,
        expected_prewarm_rows: Mapping[str, Mapping[str, Any]] | None = None,
        cache_only: bool = True,
        event_sink=None,
        task_input_cache_root: str | Path | None = None,
    ) -> None:
        verified_root = verify_current_task_input_closure(
            frozen_task_inputs,
            cache_root=task_input_cache_root,
        )
        super().__init__(
            benchmark_root,
            cache_only=cache_only,
            event_sink=event_sink,
            task_input_closure_policy=TASK_INPUT_CLOSURE_POLICY_VERSION,
            task_input_cache_root=verified_root,
        )
        self.frozen_task_inputs = frozen_task_inputs
        self.expected_prewarm_rows = dict(expected_prewarm_rows or {})

    def ensure(self, **kwargs):
        family = str(kwargs.get("family") or "")
        item_id = str(kwargs.get("item_id") or "")
        image = super().ensure(**kwargs)
        if not family_requires_task_input_closure(family):
            return image
        item_hash = stable_hash({"item_id": item_id})
        frozen_row = self.frozen_task_inputs.ledger_by_item_hash.get(item_hash)
        if (
            frozen_row is None
            or frozen_row.get("family_hash") != stable_hash({"family": family})
            or image.task_input_closure_policy != TASK_INPUT_CLOSURE_POLICY_VERSION
            or image.task_input_closure_hash != frozen_row.get("closure_hash")
        ):
            raise PermissionError("runtime task input image is not frozen by the protocol ledger")
        expected = self.expected_prewarm_rows.get(item_hash)
        if self.expected_prewarm_rows and expected is None:
            raise PermissionError("runtime task input image has no validated v5 prewarm row")
        if expected is not None and (
            expected.get("family_hash") != stable_hash({"family": family})
            or expected.get("task_input_closure_hash") != image.task_input_closure_hash
            or expected.get("prebuilt_image_key") != image.cache_key
            or expected.get("prebuilt_image_id") != image.image_id
            or expected.get("task_input_integrity_receipt_hash")
            != image.task_input_integrity_receipt_hash
            or image.task_input_integrity_container_network != "none"
        ):
            raise PermissionError("runtime task input image differs from its validated v5 prewarm row")
        return image


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
        "3.12.0": DEVELOPMENT_PREWARM_VERSION,
        "3.13.0": DEVELOPMENT_PREWARM_VERSION,
        "3.14.0": DEVELOPMENT_PREWARM_VERSION,
        "3.15.0": DEVELOPMENT_PREWARM_VERSION,
        "3.16.0": DEVELOPMENT_PREWARM_VERSION,
        "3.17.0": DEVELOPMENT_PREWARM_VERSION,
        "3.18.0": DEVELOPMENT_PREWARM_VERSION,
        "3.19.0": TASK_INPUT_CLOSURE_DEVELOPMENT_PREWARM_VERSION,
        "3.20.0": TASK_INPUT_CLOSURE_DEVELOPMENT_PREWARM_VERSION,
    }.get(str(protocol_version or ""))


def prewarm_development_images(
    *,
    benchmark_root: str | Path,
    manifest: SplitManifest,
    events_path: str | Path,
    parallel_workers: int = 4,
    attempts: int = 3,
    trial_provider_mode: str = "openai_compatible",
    prewarm_version: str = DEVELOPMENT_PREWARM_VERSION,
    task_input_cache_root: str | Path | None = None,
    frozen_task_inputs: FrozenTaskInputClosure | None = None,
) -> dict[str, Any]:
    if parallel_workers <= 0:
        raise ValueError("parallel_workers must be positive")
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    if prewarm_version not in {
        DEVELOPMENT_PREWARM_VERSION,
        TASK_INPUT_CLOSURE_DEVELOPMENT_PREWARM_VERSION,
    }:
        raise ValueError("development prewarm version is unsupported")
    task_input_closure_enabled = (
        prewarm_version == TASK_INPUT_CLOSURE_DEVELOPMENT_PREWARM_VERSION
    )
    if task_input_closure_enabled != (frozen_task_inputs is not None):
        raise ValueError(
            "v5 development prewarm and frozen task input source must be paired"
        )
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
    cache = (
        FrozenTaskInputPrebuiltImageCache(
            benchmark_root,
            event_sink=sink,
            frozen_task_inputs=frozen_task_inputs,
            task_input_cache_root=task_input_cache_root,
        )
        if frozen_task_inputs is not None
        else SkillLearnPrebuiltImageCache(benchmark_root, event_sink=sink)
    )
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
                    if task_input_closure_enabled:
                        row.update(
                            {
                                "task_input_closure_required": (
                                    image.task_input_closure_required
                                ),
                                "task_input_closure_policy": (
                                    image.task_input_closure_policy
                                ),
                                "task_input_closure_hash": (
                                    image.task_input_closure_hash
                                ),
                                "task_input_integrity_receipt_hash": (
                                    image.task_input_integrity_receipt_hash
                                ),
                                "task_input_integrity_container_network": (
                                    image.task_input_integrity_container_network
                                ),
                            }
                        )
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
                                "task_input_closure_required": (
                                    image.task_input_closure_required
                                    if task_input_closure_enabled
                                    else None
                                ),
                                "task_input_closure_hash": (
                                    image.task_input_closure_hash
                                    if task_input_closure_enabled
                                    else None
                                ),
                                "task_input_integrity_receipt_hash": (
                                    image.task_input_integrity_receipt_hash
                                    if task_input_closure_enabled
                                    else None
                                ),
                                "task_input_integrity_container_network": (
                                    image.task_input_integrity_container_network
                                    if task_input_closure_enabled
                                    else None
                                ),
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
            row = {
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
            if task_input_closure_enabled:
                row.update(
                    {
                        "task_input_closure_required": family_requires_task_input_closure(
                            manifest.family_by_id[item_id]
                        ),
                        "task_input_closure_policy": None,
                        "task_input_closure_hash": None,
                        "task_input_integrity_receipt_hash": None,
                        "task_input_integrity_container_network": None,
                    }
                )
            return row
        finally:
            backends.put(backend)

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        rows = list(executor.map(warm_one, selected_ids))

    passed = all(bool(row["passed"]) for row in rows)
    payload: dict[str, Any] = {
        "prewarm_version": prewarm_version,
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
    if task_input_closure_enabled:
        assert frozen_task_inputs is not None
        required_rows = [
            row for row in rows if row["task_input_closure_required"]
        ]
        verified_rows = [
            row
            for row in required_rows
            if row["passed"]
            and row["task_input_integrity_receipt_hash"]
            and row["task_input_integrity_container_network"] == "none"
        ]
        payload.update(
            {
                "task_input_closure_policy": (
                    TASK_INPUT_CLOSURE_POLICY_VERSION
                ),
                "task_input_build_context_policy": (
                    TASK_INPUT_BUILD_CONTEXT_POLICY_VERSION
                ),
                "task_input_integrity_container_network": "none",
                "task_input_runtime_network_fallback_allowed": False,
                "task_input_closure_required_item_count": len(required_rows),
                "task_input_closure_verified_item_count": len(verified_rows),
                "task_input_closure_set_hash": stable_hash(
                    sorted(
                        str(row["task_input_closure_hash"])
                        for row in required_rows
                        if row["task_input_closure_hash"]
                    )
                ),
                "task_input_integrity_receipt_set_hash": stable_hash(
                    sorted(
                        str(row["task_input_integrity_receipt_hash"])
                        for row in verified_rows
                    )
                ),
                "task_input_preparation_receipt_file_sha256": (
                    frozen_task_inputs.source[
                        "preparation_receipt_file_sha256"
                    ]
                ),
                "task_input_preparation_receipt_hash": (
                    frozen_task_inputs.source["preparation_receipt_hash"]
                ),
                "task_input_closure_ledger_item_count": (
                    frozen_task_inputs.source["closure_ledger_item_count"]
                ),
                "task_input_closure_ledger_hash": (
                    frozen_task_inputs.source["closure_ledger_hash"]
                ),
                "task_input_content_object_count": (
                    frozen_task_inputs.source["content_object_count"]
                ),
                "task_input_object_set_hash": (
                    frozen_task_inputs.source["object_set_hash"]
                ),
                "task_input_freeze_hash": frozen_task_inputs.freeze_hash,
            }
        )
    payload["receipt_hash"] = stable_hash(payload)
    return payload


def validate_development_prewarm_receipt(
    receipt: Mapping[str, Any],
    *,
    manifest: SplitManifest,
    expected_version: str = DEVELOPMENT_PREWARM_VERSION,
    frozen_task_inputs: FrozenTaskInputClosure | None = None,
) -> str:
    if (
        expected_version == TASK_INPUT_CLOSURE_DEVELOPMENT_PREWARM_VERSION
        and frozen_task_inputs is None
    ):
        raise ValueError(
            "v5 development prewarm validation requires the frozen task input ledger"
        )
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
    modern_versions = {
        DEVELOPMENT_PREWARM_VERSION,
        TASK_INPUT_CLOSURE_DEVELOPMENT_PREWARM_VERSION,
    }
    if expected_version in modern_versions:
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
        if expected_version == TASK_INPUT_CLOSURE_DEVELOPMENT_PREWARM_VERSION:
            expected.update(
                {
                    "task_input_closure_policy": (
                        TASK_INPUT_CLOSURE_POLICY_VERSION
                    ),
                    "task_input_build_context_policy": (
                        TASK_INPUT_BUILD_CONTEXT_POLICY_VERSION
                    ),
                    "task_input_integrity_container_network": "none",
                    "task_input_runtime_network_fallback_allowed": False,
                }
            )
            if frozen_task_inputs is not None:
                expected.update(
                    {
                        "task_input_preparation_receipt_file_sha256": (
                            frozen_task_inputs.source[
                                "preparation_receipt_file_sha256"
                            ]
                        ),
                        "task_input_preparation_receipt_hash": (
                            frozen_task_inputs.source[
                                "preparation_receipt_hash"
                            ]
                        ),
                        "task_input_closure_ledger_item_count": (
                            frozen_task_inputs.source[
                                "closure_ledger_item_count"
                            ]
                        ),
                        "task_input_closure_ledger_hash": (
                            frozen_task_inputs.source["closure_ledger_hash"]
                        ),
                        "task_input_content_object_count": (
                            frozen_task_inputs.source["content_object_count"]
                        ),
                        "task_input_object_set_hash": (
                            frozen_task_inputs.source["object_set_hash"]
                        ),
                        "task_input_freeze_hash": frozen_task_inputs.freeze_hash,
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
    expected_closure_by_item_hash = {
        stable_hash({"item_id": item_id}): family_requires_task_input_closure(
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
        if item_hash not in expected_closure_by_item_hash:
            raise ValueError(
                "development prewarm item hashes do not match the manifest"
            )
        if expected_version in modern_versions and (
            row.get("agent_runtime_key") != shared_codex_agent_runtime_key()
            or row.get("agent_runtime_version") != SHARED_CODEX_CLI_VERSION
        ):
            raise ValueError(
                "development prewarm agent runtime does not match the active runtime"
            )
        if expected_version == TASK_INPUT_CLOSURE_DEVELOPMENT_PREWARM_VERSION:
            closure_required = expected_closure_by_item_hash[item_hash]
            if row.get("task_input_closure_required") is not closure_required:
                raise ValueError(
                    "development prewarm task input closure requirement mismatch"
                )
            closure_fields = (
                row.get("task_input_closure_policy"),
                row.get("task_input_closure_hash"),
                row.get("task_input_integrity_receipt_hash"),
                row.get("task_input_integrity_container_network"),
            )
            if closure_required:
                if (
                    closure_fields[0] != TASK_INPUT_CLOSURE_POLICY_VERSION
                    or not _is_sha256_text(closure_fields[1])
                    or not _is_sha256_text(closure_fields[2])
                    or closure_fields[3] != "none"
                ):
                    raise ValueError(
                        "development prewarm task input integrity provenance is incomplete"
                    )
                if frozen_task_inputs is not None:
                    frozen_row = frozen_task_inputs.ledger_by_item_hash.get(
                        item_hash
                    )
                    if (
                        frozen_row is None
                        or frozen_row.get("family_hash")
                        != row.get("family_hash")
                        or frozen_row.get("closure_hash")
                        != row.get("task_input_closure_hash")
                    ):
                        raise ValueError(
                            "development prewarm task input row differs from frozen ledger"
                        )
            elif any(value is not None for value in closure_fields):
                raise ValueError(
                    "development prewarm task input provenance is unexpected"
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
    if expected_version == TASK_INPUT_CLOSURE_DEVELOPMENT_PREWARM_VERSION:
        closure_rows = [
            row for row in rows if row.get("task_input_closure_required")
        ]
        if receipt.get("task_input_closure_required_item_count") != len(
            closure_rows
        ):
            raise ValueError(
                "development prewarm task input closure count mismatch"
            )
        if receipt.get("task_input_closure_verified_item_count") != len(
            closure_rows
        ):
            raise ValueError(
                "development prewarm task input verified count mismatch"
            )
        closure_hashes = sorted(
            str(row["task_input_closure_hash"]) for row in closure_rows
        )
        integrity_hashes = sorted(
            str(row["task_input_integrity_receipt_hash"])
            for row in closure_rows
        )
        if receipt.get("task_input_closure_set_hash") != stable_hash(
            closure_hashes
        ):
            raise ValueError(
                "development prewarm task input closure set mismatch"
            )
        if receipt.get("task_input_integrity_receipt_set_hash") != stable_hash(
            integrity_hashes
        ):
            raise ValueError(
                "development prewarm task input integrity receipt set mismatch"
            )
        if frozen_task_inputs is not None and set(
            expected_prewarm_closure_rows(receipt)
        ) != set(frozen_task_inputs.ledger_by_item_hash):
            raise ValueError(
                "development prewarm task input rows do not cover the frozen ledger"
            )
    return declared_hash


def _is_sha256_text(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


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
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parallel-workers", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument(
        "--prewarm-version",
        choices=(
            DEVELOPMENT_PREWARM_VERSION,
            TASK_INPUT_CLOSURE_DEVELOPMENT_PREWARM_VERSION,
        ),
        default=DEVELOPMENT_PREWARM_VERSION,
    )
    parser.add_argument("--task-input-cache-root", type=Path)
    parser.add_argument(
        "--trial-provider-mode",
        choices=("openai_compatible",),
    )
    parser.add_argument("--require-passed", action="store_true")
    args = parser.parse_args()
    load_dotenv(args.env_file)
    map_legacy_model_env()
    manifest = SplitManifest.read(args.manifest)
    frozen_task_inputs: FrozenTaskInputClosure | None = None
    if args.protocol is not None:
        protocol_payload = json.loads(args.protocol.read_text(encoding="utf-8"))
        if not isinstance(protocol_payload, Mapping):
            raise ValueError("paper protocol must contain one JSON object")
        declared_version = dict(protocol_payload.get("execution") or {}).get(
            "development_prewarm"
        )
        if declared_version != args.prewarm_version:
            raise ValueError(
                "prewarm CLI version differs from the paper protocol"
            )
        frozen_task_inputs = load_frozen_task_input_closure(
            protocol_payload,
            project_root=args.project_root,
        )
    elif args.prewarm_version == TASK_INPUT_CLOSURE_DEVELOPMENT_PREWARM_VERSION:
        raise ValueError("v5 development prewarm requires --protocol")
    receipt = prewarm_development_images(
        benchmark_root=args.root,
        manifest=manifest,
        events_path=args.events,
        parallel_workers=args.parallel_workers,
        attempts=args.attempts,
        trial_provider_mode=(
            args.trial_provider_mode or configured_skilllearn_provider_mode()
        ),
        prewarm_version=args.prewarm_version,
        task_input_cache_root=args.task_input_cache_root,
        frozen_task_inputs=frozen_task_inputs,
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
