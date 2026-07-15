from __future__ import annotations

"""Prewarm the hygienic SEC-13F measurement benchmark without tree drift."""

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
    verify_measurement_view,
    write_json,
)

from .hygienic_materialize import (
    CURRENT_ALIAS,
    FAMILY,
    FIXED_FAILURE_CODE,
    GOLD_CONTAINER_PATH,
    GOLD_ISOLATION_POLICY_VERSION,
    MATERIALIZATION_REPORT_NAME,
    MATERIALIZATION_VERSION,
    PREVIOUS_ALIAS,
    TREE_RECEIPT_VERSION,
    VERIFIER_EVIDENCE_POLICY_VERSION,
    _versioned_tree_receipt,
    measurement_benchmark_tree_receipt_v2,
)


PREWARM_VERSION = "financial_sec13f_contract_hygienic_measurement_prewarm_v2"
OFFLINE_VERIFIER_PROFILE_ID = "common-pytest-ctrf-py312-v1"
OFFLINE_VERIFIER_REQUIREMENTS = (
    "pytest==8.4.1",
    "pytest-json-ctrf==0.3.5",
)


class HygienicPrewarmError(RuntimeError):
    """The cache-only hygienic runtime could not be frozen."""


def _disable_bytecode_writes() -> None:
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def _verify_self_hash(payload: Mapping[str, Any], field: str) -> str:
    body = dict(payload)
    declared = body.pop(field, None)
    if not isinstance(declared, str) or declared != payload_hash(body):
        raise HygienicPrewarmError(f"{field} self hash mismatch")
    return declared


def _load_materialization(benchmark_root: Path) -> dict[str, Any]:
    path = benchmark_root / MATERIALIZATION_REPORT_NAME
    if path.is_symlink() or not path.is_file():
        raise HygienicPrewarmError(
            "measurement materialization report is not a regular file"
        )
    value = read_json(path)
    if (
        value.get("materialization_version") != MATERIALIZATION_VERSION
        or value.get("tree_receipt_version") != TREE_RECEIPT_VERSION
    ):
        raise HygienicPrewarmError("measurement materialization drifted")
    _verify_self_hash(value, "materialization_hash")
    expected_fields = {
        "materialization_version",
        "tree_receipt_version",
        "project_authored_extension",
        "official_skilllearnbench_score",
        "private_pack_hash",
        "private_pack_accessed",
        "measurement_view_hash",
        "measurement_gold_hash",
        "previous_archive_sha256",
        "current_archive_sha256",
        "period_source_receipts",
        "period_aliases",
        "verifier_isolation",
        "verifier_evidence",
        "item_count",
        "items",
        "item_set_hash",
        "benchmark_tree_hash",
        "sealed_task_count_materialized",
        "sealed_content_accessed_by_materializer",
        "sealed_content_accessed_by_measurement_root",
        "sealed_content_persisted",
        "sealed_gold_accessed",
        "model_calls",
        "online_judge_calls",
        "secret_value_persisted",
        "materialization_hash",
    }
    items = value.get("items")
    expected_isolation = {
        "policy_version": GOLD_ISOLATION_POLICY_VERSION,
        "candidate_namespace": "/root",
        "verifier_namespace": "/tests",
        "gold_container_path": GOLD_CONTAINER_PATH,
        "gold_written_outside_environment_tree": True,
        "gold_copied_into_environment_image": False,
        "candidate_visible_gold_path_count": 0,
        "post_agent_verifier_mount_required": True,
    }
    expected_evidence = {
        "policy_version": VERIFIER_EVIDENCE_POLICY_VERSION,
        "failure_message_kind": "fixed_code_only",
        "fixed_failure_code": FIXED_FAILURE_CODE,
        "pytest_traceback_mode": "none",
        "raw_expected_in_failure_evidence": False,
        "raw_observed_in_failure_evidence": False,
    }
    if (
        set(value) != expected_fields
        or value.get("project_authored_extension") is not True
        or value.get("official_skilllearnbench_score") is not False
        or value.get("private_pack_accessed") is not False
        or value.get("verifier_isolation") != expected_isolation
        or value.get("verifier_evidence") != expected_evidence
        or value.get("item_count") != 8
        or not isinstance(items, list)
        or len(items) != 8
        or value.get("item_set_hash") != payload_hash(items)
        or value.get("sealed_task_count_materialized") != 0
        or value.get("sealed_content_accessed_by_materializer") is not False
        or value.get("sealed_content_accessed_by_measurement_root") is not False
        or value.get("sealed_content_persisted") is not False
        or value.get("sealed_gold_accessed") is not False
        or value.get("model_calls") != 0
        or value.get("online_judge_calls") != 0
        or value.get("secret_value_persisted") is not False
    ):
        raise HygienicPrewarmError(
            "measurement materialization policy drifted"
        )
    actual_tree = measurement_benchmark_tree_receipt_v2(benchmark_root)
    if value.get("benchmark_tree_hash") != actual_tree["tree_hash"]:
        raise HygienicPrewarmError("materialized benchmark tree drifted")
    return value


def _require_exact_entries(
    root: Path,
    *,
    expected: set[str],
    label: str,
) -> None:
    if root.is_symlink() or not root.is_dir():
        raise HygienicPrewarmError(f"{label} is not a regular directory")
    entries = tuple(root.iterdir())
    if any(path.is_symlink() for path in entries) or {
        path.name for path in entries
    } != expected:
        raise HygienicPrewarmError(f"{label} entries drifted")


def _validate_measurement_task_tree(
    benchmark: Path,
    *,
    item_ids: Sequence[str],
) -> None:
    _require_exact_entries(
        benchmark,
        expected={"core", "agents", "tasks", MATERIALIZATION_REPORT_NAME},
        label="materialized benchmark root",
    )
    tasks = benchmark / "tasks"
    _require_exact_entries(tasks, expected={FAMILY}, label="task family root")
    family = tasks / FAMILY
    _require_exact_entries(
        family,
        expected=set(item_ids),
        label="measurement task root",
    )
    for item_id in item_ids:
        task = family / item_id
        _require_exact_entries(
            task,
            expected={"instruction.md", "task.toml", "environment", "tests"},
            label="measurement item",
        )
        _require_exact_entries(
            task / "environment",
            expected={
                "Dockerfile",
                "period-previous.zip",
                "period-current.zip",
            },
            label="measurement environment",
        )
        _require_exact_entries(
            task / "tests",
            expected={"test.sh", "test_outputs.py", "expected_output.json"},
            label="measurement verifier",
        )


def _validate_item_receipts(
    *,
    benchmark: Path,
    measurement_view: Mapping[str, Any],
    materialization: Mapping[str, Any],
) -> list[str]:
    item_ids = [
        str(item["item_id"])
        for item in measurement_view["measurement_items"]
    ]
    if len(item_ids) != 8 or len(set(item_ids)) != 8:
        raise HygienicPrewarmError("measurement item set drifted")
    _validate_measurement_task_tree(benchmark, item_ids=item_ids)
    receipt_rows = materialization["items"]
    receipt_fields = {
        "item_id",
        "item_id_hash",
        "fold",
        "template",
        "instruction_sha256",
        "task_toml_sha256",
        "tree_receipt_version",
        "environment_tree_hash",
        "tests_tree_hash",
        "expected_output_sha256",
        "answers_hash",
        "gold_container_path",
        "gold_in_environment_tree",
        "failure_evidence_policy",
        "raw_content_persisted_in_report",
    }
    if (
        not isinstance(receipt_rows, list)
        or not all(isinstance(row, Mapping) for row in receipt_rows)
        or [str(row.get("item_id")) for row in receipt_rows] != item_ids
        or any(set(row) != receipt_fields for row in receipt_rows)
    ):
        raise HygienicPrewarmError("measurement item receipts drifted")
    receipts_by_id = {str(row["item_id"]): row for row in receipt_rows}
    for item in measurement_view["measurement_items"]:
        item_id = str(item["item_id"])
        receipt = receipts_by_id[item_id]
        task = benchmark / "tasks" / FAMILY / item_id
        environment_receipt = _versioned_tree_receipt(task / "environment")
        tests_receipt = _versioned_tree_receipt(task / "tests")
        if (
            receipt.get("item_id_hash")
            != payload_hash({"item_id": item_id})
            or receipt.get("fold") != item.get("fold")
            or receipt.get("template") != item.get("template")
            or receipt.get("instruction_sha256")
            != item.get("instruction_sha256")
            or receipt.get("instruction_sha256")
            != sha256_file(task / "instruction.md")
            or receipt.get("task_toml_sha256")
            != sha256_file(task / "task.toml")
            or receipt.get("tree_receipt_version") != TREE_RECEIPT_VERSION
            or receipt.get("environment_tree_hash")
            != environment_receipt["tree_hash"]
            or receipt.get("tests_tree_hash") != tests_receipt["tree_hash"]
            or receipt.get("expected_output_sha256")
            != sha256_file(task / "tests" / "expected_output.json")
            or receipt.get("gold_container_path") != GOLD_CONTAINER_PATH
            or receipt.get("gold_in_environment_tree") is not False
            or receipt.get("failure_evidence_policy")
            != VERIFIER_EVIDENCE_POLICY_VERSION
            or receipt.get("raw_content_persisted_in_report") is not False
        ):
            raise HygienicPrewarmError(
                "materialized measurement item drifted"
            )
    return item_ids


def _validate_period_sources(
    *,
    measurement_view: Mapping[str, Any],
    materialization: Mapping[str, Any],
) -> None:
    expected_aliases = {
        "previous": PREVIOUS_ALIAS,
        "current": CURRENT_ALIAS,
        "aliases_are_calendar_labels": False,
    }
    source_receipts = materialization.get("period_source_receipts")
    if (
        materialization.get("period_aliases") != expected_aliases
        or not isinstance(source_receipts, Mapping)
        or set(source_receipts) != {"previous", "current"}
    ):
        raise HygienicPrewarmError("period archive aliases drifted")
    for role, alias, archive_field in (
        ("previous", PREVIOUS_ALIAS, "previous_archive_sha256"),
        ("current", CURRENT_ALIAS, "current_archive_sha256"),
    ):
        source = source_receipts.get(role)
        expected_source = measurement_view["sources"][role]
        if (
            not isinstance(source, Mapping)
            or set(source)
            != {
                "container_alias",
                "archive_sha256",
                "coverpage_sha256",
                "infotable_sha256",
                "source_fingerprint",
                "source_path_persisted",
            }
            or source.get("container_alias") != alias
            or source.get("archive_sha256")
            != materialization.get(archive_field)
            or source.get("coverpage_sha256")
            != expected_source.get("coverpage_sha256")
            or source.get("infotable_sha256")
            != expected_source.get("infotable_sha256")
            or source.get("source_fingerprint")
            != expected_source.get("source_fingerprint")
            or source.get("source_path_persisted") is not False
        ):
            raise HygienicPrewarmError("period source receipt drifted")


def prepare_measurement_runtime_v2(
    *,
    benchmark_root: str | Path,
    measurement_view: Mapping[str, Any],
    output_root: str | Path,
    agent_id: str = "codex",
) -> dict[str, Any]:
    """Build once, then prove all formal runtime lookups are cache-only."""

    _disable_bytecode_writes()
    benchmark = Path(benchmark_root).expanduser().resolve(strict=True)
    destination = Path(output_root).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if destination == benchmark or benchmark in destination.parents:
        raise HygienicPrewarmError(
            "prewarm output must not modify the frozen benchmark"
        )
    view = verify_measurement_view(measurement_view)
    materialization = _load_materialization(benchmark)
    if (
        materialization.get("measurement_view_hash")
        != view.get("measurement_view_hash")
        or materialization.get("private_pack_hash")
        != view.get("private_pack_hash")
    ):
        raise HygienicPrewarmError(
            "benchmark binds another measurement view"
        )
    item_ids = _validate_item_receipts(
        benchmark=benchmark,
        measurement_view=view,
        materialization=materialization,
    )
    _validate_period_sources(
        measurement_view=view,
        materialization=materialization,
    )
    pre_prewarm_tree = measurement_benchmark_tree_receipt_v2(benchmark)
    if pre_prewarm_tree["tree_hash"] != materialization["benchmark_tree_hash"]:
        raise HygienicPrewarmError("prewarm input tree drifted")
    profile = offline_verifier_profile_for_family(FAMILY)
    if (
        profile is None
        or profile.profile_id != OFFLINE_VERIFIER_PROFILE_ID
        or profile.requirements != OFFLINE_VERIFIER_REQUIREMENTS
        or profile.python_version != "3.12"
        or profile.python_abi != "cp312"
    ):
        raise HygienicPrewarmError(
            "financial offline verifier dependency profile drifted"
        )

    destination.mkdir(parents=True)
    try:
        event_sink = JsonlEventSink(destination / "prewarm.events.jsonl")
        build_cache = SkillLearnPrebuiltImageCache(
            benchmark,
            cache_only=False,
            event_sink=event_sink,
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
        built_rows: list[dict[str, Any]] = []
        preparation_images: list[Any] = []
        for item_id in item_ids:
            image = build_cache.ensure(
                family=FAMILY,
                item_id=item_id,
                agent_id=agent_id,
                runner=runner,
                trace_id=(
                    "sec13f-v2-prewarm:"
                    + payload_hash({"item_id": item_id})[:20]
                ),
            )
            preparation_images.append(image)
            built_rows.append(
                {
                    "item_id": item_id,
                    "item_id_hash": payload_hash({"item_id": item_id}),
                    "cache_key": image.cache_key,
                    "environment_hash": image.environment_hash,
                    "source_environment_hash": image.source_environment_hash,
                    "image_id": image.image_id,
                    "agent_runtime_key": image.agent_runtime_key,
                    "agent_runtime_version": image.agent_runtime_version,
                    "prepared_before_formal_cache_check": True,
                }
            )
        if (
            len({image.image_id for image in preparation_images}) != 1
            or len({image.cache_key for image in preparation_images}) != 1
            or len({image.tag for image in preparation_images}) != 1
        ):
            raise HygienicPrewarmError(
                "preparation tasks do not share one local image"
            )
        prepared_image = preparation_images[0]
        preparation_receipt_path = (
            destination / "offline-verifier.preparation.json"
        )
        preparation_receipt = dict(
            prepare_offline_verifier_runtime(
                profile=profile,
                base_image_tag=prepared_image.tag,
                report_path=preparation_receipt_path,
                delegate=runner.subprocess,
                event_sink=event_sink,
                trace_id="sec13f-v2-offline-verifier-preparation",
            )
        )
        _verify_self_hash(preparation_receipt, "receipt_hash")
        expected_runtime_key = offline_verifier_runtime_key(profile=profile)
        if (
            not preparation_receipt_path.is_file()
            or read_json(preparation_receipt_path) != preparation_receipt
            or preparation_receipt.get("report_version")
            != "offline_verifier_preparation_receipt_v2"
            or preparation_receipt.get("policy")
            != OFFLINE_VERIFIER_POLICY_VERSION
            or preparation_receipt.get("profile_id") != profile.profile_id
            or preparation_receipt.get("profile_hash")
            != profile.profile_hash
            or preparation_receipt.get("runtime_key")
            != expected_runtime_key
            or preparation_receipt.get("base_image_tag")
            != prepared_image.tag
            or preparation_receipt.get("base_image_id")
            != prepared_image.image_id
            or preparation_receipt.get("docker_install_network") != "none"
            or preparation_receipt.get("probe_passed") is not True
            or preparation_receipt.get("raw_content_persisted") is not False
        ):
            raise HygienicPrewarmError(
                "offline verifier preparation receipt drifted"
            )

        formal_cache = SkillLearnPrebuiltImageCache(
            benchmark,
            cache_only=True,
            event_sink=event_sink,
        )
        offline_cache = SkillLearnOfflineVerifierRuntimeCache(
            event_sink=event_sink
        )
        formal_rows: list[dict[str, Any]] = []
        runtime_keys: set[str] = set()
        image_ids: set[str] = set()
        cache_keys: set[str] = set()
        for item_id in item_ids:
            image = formal_cache.ensure(
                family=FAMILY,
                item_id=item_id,
                agent_id=agent_id,
                runner=runner,
                trace_id=(
                    "sec13f-v2-formal-cache:"
                    + payload_hash({"item_id": item_id})[:20]
                ),
            )
            runtime = offline_cache.ensure(
                profile=profile,
                base_image_tag=image.tag,
                base_image_id=image.image_id,
                delegate=runner.subprocess,
                trace_id=(
                    "sec13f-v2-offline-verifier:"
                    + payload_hash({"item_id": item_id})[:20]
                ),
            )
            runtime_keys.add(runtime.runtime_key)
            image_ids.add(image.image_id)
            cache_keys.add(image.cache_key)
            formal_rows.append(
                {
                    "item_id": item_id,
                    "item_id_hash": payload_hash({"item_id": item_id}),
                    "cache_key": image.cache_key,
                    "environment_hash": image.environment_hash,
                    "source_environment_hash": image.source_environment_hash,
                    "image_id": image.image_id,
                    "agent_runtime_key": image.agent_runtime_key,
                    "agent_runtime_version": image.agent_runtime_version,
                    "prebuilt_cache_reused": image.reused,
                    "offline_verifier_profile_id": profile.profile_id,
                    "offline_verifier_profile_hash": profile.profile_hash,
                    "offline_verifier_runtime_key": runtime.runtime_key,
                    "offline_verifier_runtime_reused": runtime.reused,
                    "verifier_runtime_network": "none",
                }
            )
        if (
            len(image_ids) != 1
            or len(cache_keys) != 1
            or len(runtime_keys) != 1
            or not all(row["prebuilt_cache_reused"] for row in formal_rows)
            or not all(
                row["offline_verifier_runtime_reused"]
                for row in formal_rows
            )
            or runtime_keys != {expected_runtime_key}
        ):
            raise HygienicPrewarmError(
                "formal measurement tasks do not share one frozen local runtime"
            )
        post_prewarm_tree = measurement_benchmark_tree_receipt_v2(benchmark)
        if (
            post_prewarm_tree["tree_hash"]
            != materialization["benchmark_tree_hash"]
            or post_prewarm_tree != pre_prewarm_tree
        ):
            raise HygienicPrewarmError(
                "prewarm modified the frozen measurement benchmark"
            )
        body = {
            "prewarm_version": PREWARM_VERSION,
            "tree_receipt_version": TREE_RECEIPT_VERSION,
            "measurement_view_hash": view["measurement_view_hash"],
            "materialization_hash": materialization["materialization_hash"],
            "benchmark_tree_hash": materialization["benchmark_tree_hash"],
            "pre_prewarm_tree_hash": pre_prewarm_tree["tree_hash"],
            "post_prewarm_tree_hash": post_prewarm_tree["tree_hash"],
            "benchmark_tree_unchanged": True,
            "python_dont_write_bytecode": sys.dont_write_bytecode,
            "python_dont_write_bytecode_env": os.environ.get(
                "PYTHONDONTWRITEBYTECODE"
            ),
            "item_count": len(item_ids),
            "preparation_rows": built_rows,
            "preparation_row_set_hash": payload_hash(built_rows),
            "formal_cache_rows": formal_rows,
            "formal_cache_row_set_hash": payload_hash(formal_rows),
            "unique_image_id_hash": payload_hash(
                {"image_id": next(iter(image_ids))}
            ),
            "unique_cache_key_hash": payload_hash(
                {"cache_key": next(iter(cache_keys))}
            ),
            "offline_verifier_profile_id": profile.profile_id,
            "offline_verifier_profile_hash": profile.profile_hash,
            "offline_verifier_requirements": list(profile.requirements),
            "offline_verifier_requirements_hash": payload_hash(
                list(profile.requirements)
            ),
            "offline_verifier_runtime_key": next(iter(runtime_keys)),
            "offline_verifier_preparation": {
                "relative_path": preparation_receipt_path.name,
                "file_sha256": sha256_file(preparation_receipt_path),
                "receipt_hash": preparation_receipt["receipt_hash"],
                "network_allowed_only_during_preparation": True,
                "docker_install_network": "none",
                "probe_passed": True,
            },
            "formal_execution_cache_only": True,
            "formal_image_cache_only": True,
            "formal_offline_verifier_cache_only": True,
            "preparation_network_allowed": True,
            "formal_verifier_network": "none",
            "model_calls": 0,
            "online_judge_calls": 0,
            "sealed_task_count": 0,
            "sealed_content_accessed": False,
            "secret_value_persisted": False,
        }
        report = {**body, "prewarm_hash": payload_hash(body)}
        write_json(destination / "measurement.prewarm.json", report)
        return report
    except Exception:
        # Preserve preparation evidence for diagnosis.  It is outside the
        # frozen benchmark and is not a formal outcome root.
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--measurement-view", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = prepare_measurement_runtime_v2(
        benchmark_root=args.benchmark_root,
        measurement_view=read_json(args.measurement_view),
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "prewarm_hash": report["prewarm_hash"],
                "benchmark_tree_unchanged": True,
                "item_count": report["item_count"],
                "formal_execution_cache_only": True,
                "model_calls": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
