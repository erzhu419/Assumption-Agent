from __future__ import annotations

"""Build and freeze local image/verifier caches for four sealed tasks."""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks.offline_verifier import (
    OFFLINE_VERIFIER_POLICY_VERSION,
    SkillLearnOfflineVerifierRuntimeCache,
    offline_verifier_profile_for_family,
    offline_verifier_runtime_key,
    prepare_offline_verifier_runtime,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    SkillLearnPrebuiltImageCache,
    SkillLearnSubprocessBackend,
)
from assumption_agent.events import JsonlEventSink
from replication_runtime.financial_semantic_v2.pack import (
    payload_hash,
    read_json,
    sha256_file,
    write_json,
)

from .hygienic_materialize import FAMILY, _versioned_tree_receipt
from .hygienic_prewarm import (
    OFFLINE_VERIFIER_PROFILE_ID,
    OFFLINE_VERIFIER_REQUIREMENTS,
)
from .sealed_materialize import (
    MATERIALIZATION_REPORT_NAME,
    MATERIALIZATION_VERSION,
    sealed_benchmark_tree_receipt_v1,
)
from .sealed_prepare import verify_sealed_payload_v1


PREWARM_VERSION = "financial_sec13f_replication_c_sealed_prewarm_v1"
PREWARM_FILENAME = "sealed.prewarm.json"


class SealedPrewarmError(RuntimeError):
    """The sealed cache preparation failed closed."""


def _verify_materialization(
    benchmark: Path,
    *,
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    path = benchmark / MATERIALIZATION_REPORT_NAME
    if path.is_symlink() or not path.is_file():
        raise SealedPrewarmError("sealed materialization is unavailable")
    report = read_json(path)
    body = dict(report)
    declared = body.pop("materialization_hash", None)
    items = report.get("items")
    item_ids = [str(item["item_id"]) for item in payload["sealed_items"]]
    if (
        declared != payload_hash(body)
        or report.get("materialization_version") != MATERIALIZATION_VERSION
        or report.get("sealed_payload_hash") != payload["sealed_payload_hash"]
        or report.get("private_pack_hash") != payload["private_pack_hash"]
        or report.get("item_count") != 4
        or report.get("sealed_task_count_materialized") != 4
        or report.get("measurement_task_count_materialized") != 0
        or not isinstance(items, list)
        or len(items) != 4
        or report.get("item_set_hash") != payload_hash(items or ())
        or report.get("model_calls") != 0
        or report.get("online_judge_calls") != 0
        or report.get("sealed_content_persisted_in_report") is not False
        or report.get("sealed_gold_persisted_in_report") is not False
    ):
        raise SealedPrewarmError("sealed materialization drifted")
    task_root = benchmark / "tasks" / FAMILY
    if task_root.is_symlink() or not task_root.is_dir():
        raise SealedPrewarmError("sealed task root is unavailable")
    if {path.name for path in task_root.iterdir()} != set(item_ids):
        raise SealedPrewarmError("sealed task set drifted")
    receipts = {
        str(row.get("item_id_hash")): row
        for row in items
        if isinstance(row, Mapping)
    }
    for item in payload["sealed_items"]:
        item_id = str(item["item_id"])
        task = task_root / item_id
        receipt = receipts.get(payload_hash({"item_id": item_id}))
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("instruction_sha256") != item["instruction_sha256"]
            or receipt.get("instruction_sha256")
            != sha256_file(task / "instruction.md")
            or receipt.get("environment_tree_hash")
            != _versioned_tree_receipt(task / "environment")["tree_hash"]
            or receipt.get("tests_tree_hash")
            != _versioned_tree_receipt(task / "tests")["tree_hash"]
        ):
            raise SealedPrewarmError("sealed item receipt drifted")
    actual_tree = sealed_benchmark_tree_receipt_v1(benchmark)
    if report.get("benchmark_tree_hash") != actual_tree["tree_hash"]:
        raise SealedPrewarmError("sealed benchmark tree drifted")
    return report, item_ids


def prepare_sealed_runtime_v1(
    *,
    benchmark_root: str | Path,
    measurement_view: Mapping[str, Any],
    sealed_payload: Mapping[str, Any],
    output_root: str | Path,
    agent_id: str = "codex",
) -> dict[str, Any]:
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    benchmark = Path(benchmark_root).expanduser().resolve(strict=True)
    destination = Path(output_root).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if destination == benchmark or benchmark in destination.parents:
        raise SealedPrewarmError("prewarm output overlaps sealed benchmark")
    payload = verify_sealed_payload_v1(
        sealed_payload,
        measurement_view=measurement_view,
    )
    materialization, item_ids = _verify_materialization(
        benchmark, payload=payload
    )
    pre_tree = sealed_benchmark_tree_receipt_v1(benchmark)
    profile = offline_verifier_profile_for_family(FAMILY)
    if (
        profile is None
        or profile.profile_id != OFFLINE_VERIFIER_PROFILE_ID
        or profile.requirements != OFFLINE_VERIFIER_REQUIREMENTS
        or profile.python_version != "3.12"
        or profile.python_abi != "cp312"
    ):
        raise SealedPrewarmError("offline verifier profile drifted")
    destination.mkdir(parents=True, mode=0o700)
    event_sink = JsonlEventSink(destination / "prewarm.events.jsonl")
    build_cache = SkillLearnPrebuiltImageCache(
        benchmark, cache_only=False, event_sink=event_sink
    )
    loader = SkillLearnSubprocessBackend(
        benchmark,
        agent_id=agent_id,
        provider_mode="openai_compatible",
        record_upstream=False,
        prebuilt_cache=build_cache,
        event_sink=event_sink,
    )
    runner = loader._load_runner()
    preparation_rows: list[dict[str, Any]] = []
    images = []
    for item_id in item_ids:
        image = build_cache.ensure(
            family=FAMILY,
            item_id=item_id,
            agent_id=agent_id,
            runner=runner,
            trace_id="sec13f-sealed-prewarm:" + payload_hash({"item_id": item_id})[:20],
        )
        images.append(image)
        preparation_rows.append(
            {
                "item_id_hash": payload_hash({"item_id": item_id}),
                "cache_key": image.cache_key,
                "environment_hash": image.environment_hash,
                "source_environment_hash": image.source_environment_hash,
                "image_id": image.image_id,
                "agent_runtime_key": image.agent_runtime_key,
                "agent_runtime_version": image.agent_runtime_version,
            }
        )
    if (
        len(images) != 4
        or len({image.image_id for image in images}) != 1
        or len({image.cache_key for image in images}) != 1
        or len({image.tag for image in images}) != 1
    ):
        raise SealedPrewarmError("sealed tasks do not share one local image")
    prepared = images[0]
    verifier_path = destination / "offline-verifier.preparation.json"
    verifier = dict(
        prepare_offline_verifier_runtime(
            profile=profile,
            base_image_tag=prepared.tag,
            report_path=verifier_path,
            delegate=runner.subprocess,
            event_sink=event_sink,
            trace_id="sec13f-sealed-offline-verifier-preparation",
        )
    )
    expected_runtime_key = offline_verifier_runtime_key(profile=profile)
    if (
        not verifier_path.is_file()
        or verifier.get("runtime_key") != expected_runtime_key
        or verifier.get("docker_install_network") != "none"
        or verifier.get("probe_passed") is not True
    ):
        raise SealedPrewarmError("offline verifier preparation drifted")
    formal_cache = SkillLearnPrebuiltImageCache(
        benchmark, cache_only=True, event_sink=event_sink
    )
    offline_cache = SkillLearnOfflineVerifierRuntimeCache(event_sink=event_sink)
    formal_rows: list[dict[str, Any]] = []
    for item_id in item_ids:
        image = formal_cache.ensure(
            family=FAMILY,
            item_id=item_id,
            agent_id=agent_id,
            runner=runner,
            trace_id="sec13f-sealed-formal:" + payload_hash({"item_id": item_id})[:20],
        )
        runtime = offline_cache.ensure(
            profile=profile,
            base_image_tag=image.tag,
            base_image_id=image.image_id,
            delegate=runner.subprocess,
            trace_id="sec13f-sealed-verifier:" + payload_hash({"item_id": item_id})[:20],
        )
        if (
            image.image_id != prepared.image_id
            or not image.reused
            or runtime.runtime_key != expected_runtime_key
            or not runtime.reused
        ):
            raise SealedPrewarmError("formal sealed cache lookup is not cache-only")
        formal_rows.append(
            {
                "item_id_hash": payload_hash({"item_id": item_id}),
                "cache_key": image.cache_key,
                "environment_hash": image.environment_hash,
                "source_environment_hash": image.source_environment_hash,
                "image_id": image.image_id,
                "agent_runtime_key": image.agent_runtime_key,
                "agent_runtime_version": image.agent_runtime_version,
                "offline_verifier_profile_id": profile.profile_id,
                "offline_verifier_profile_hash": profile.profile_hash,
                "offline_verifier_runtime_key": runtime.runtime_key,
                "prebuilt_cache_reused": True,
                "offline_verifier_runtime_reused": True,
                "verifier_runtime_network": "none",
            }
        )
    post_tree = sealed_benchmark_tree_receipt_v1(benchmark)
    if post_tree != pre_tree:
        raise SealedPrewarmError("prewarm modified sealed benchmark")
    body = {
        "prewarm_version": PREWARM_VERSION,
        "sealed_payload_hash": payload["sealed_payload_hash"],
        "materialization_hash": materialization["materialization_hash"],
        "benchmark_tree_hash": pre_tree["tree_hash"],
        "pre_prewarm_tree_hash": pre_tree["tree_hash"],
        "post_prewarm_tree_hash": post_tree["tree_hash"],
        "benchmark_tree_unchanged": True,
        "item_count": 4,
        "preparation_rows": preparation_rows,
        "preparation_row_set_hash": payload_hash(preparation_rows),
        "formal_cache_rows": formal_rows,
        "formal_cache_row_set_hash": payload_hash(formal_rows),
        "offline_verifier_profile_id": profile.profile_id,
        "offline_verifier_profile_hash": profile.profile_hash,
        "offline_verifier_runtime_key": expected_runtime_key,
        "offline_verifier_preparation_file_sha256": sha256_file(verifier_path),
        "formal_execution_cache_only": True,
        "formal_image_cache_only": True,
        "formal_offline_verifier_cache_only": True,
        "formal_verifier_network": "none",
        "python_dont_write_bytecode": True,
        "python_dont_write_bytecode_env": "1",
        "model_calls": 0,
        "online_judge_calls": 0,
        "sealed_content_persisted_in_receipt": False,
        "secret_value_persisted": False,
    }
    report = {**body, "prewarm_hash": payload_hash(body)}
    write_json(destination / PREWARM_FILENAME, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--measurement-view", type=Path, required=True)
    parser.add_argument("--sealed-payload", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = prepare_sealed_runtime_v1(
        benchmark_root=args.benchmark_root,
        measurement_view=read_json(args.measurement_view),
        sealed_payload=read_json(args.sealed_payload),
        output_root=args.output_root,
    )
    print(json.dumps({"prewarm_hash": report["prewarm_hash"], "item_count": 4}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
