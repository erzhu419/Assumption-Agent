"""Plan and finalize the frozen all-remote P16 three-arm measurement."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from reconstruction_v2.assumption_agent.benchmarks import (
    bright_p14_direct_c_confirm_v1 as p14,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    bright_p16_extension_acquisition_v1 as acquisition,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    bright_reasoning_retrieval_core_v1 as metric,
)


PLAN_SCHEMA = "bright_p16_all_remote_execution_plan_v1"
SELECTION_SCHEMA = "bright_p16_all_remote_complete_case_selection_v1"
ACTION_SCHEMA = "bright_p16_all_remote_three_arm_actions_v1"
REMOTE_RESULT_SCHEMA = "bright_p16_all_remote_action_result_v1"
RESULT_SCHEMA = "bright_p16_all_remote_c_confirm_result_v1"
FREEZE_SCHEMA = "bright_p16_all_remote_implementation_freeze_v1"
FINGERPRINT_SCHEMA = "bright_p16_remote_runtime_fingerprint_v1"

FAMILIES = acquisition.FAMILIES
ATTEMPT_COUNT = acquisition.ATTEMPT_COUNT
TARGET_PER_FAMILY = 10
SELECTED_COUNT = TARGET_PER_FAMILY * len(FAMILIES)
REMOTE_HOST_ALIAS = "jtl311linux"
REMOTE_HOSTNAME = "zhengliang-C246-WU4"
REMOTE_ROOT = Path("/home/erzhu419/p16_all_remote_20260722")
REMOTE_BASE = REMOTE_ROOT / "runtime/reconstruction_v2"
REMOTE_WORK_ROOT = REMOTE_ROOT / "work"

RUN_ROOT_RELATIVE = Path("artifacts/bright_p16_all_remote_c_confirm_v1")
PLAN_RELATIVE = RUN_ROOT_RELATIVE / "remote_execution.plan.json"
REMOTE_ARCHIVE_RELATIVE = RUN_ROOT_RELATIVE / "remote_archive"
RESULT_RELATIVE = Path(
    "manifests/bright_p16_all_remote_c_confirm_result_v1.json"
)
FREEZE_RELATIVE = Path(
    "manifests/bright_p16_all_remote_implementation_freeze_v1.json"
)
FINGERPRINT_RELATIVE = Path(
    "manifests/bright_p16_remote_runtime_fingerprint_v1.json"
)
IMPLEMENTATION_RELATIVE = Path(
    "assumption_agent/benchmarks/bright_p16_all_remote_c_confirm_v1.py"
)
RUNNER_RELATIVE = Path(
    "replication_runtime/bright_p16_all_remote_v1/runner.py"
)
TEST_RELATIVE = Path("tests/test_bright_p16_all_remote_c_confirm_v1.py")
MINILM_ENCODER_RELATIVE = Path(
    "replication_runtime/bright_minilm_v1/encoder.py"
)


class P16AllRemoteError(RuntimeError):
    """The all-remote P16 plan, action archive, or score failed closed."""


class OneShotRefusal(P16AllRemoteError):
    """The P16 plan or final result was already consumed."""


def _file_sha256(path: Path) -> str:
    return acquisition.p14_acquisition.utilities.file_sha256(path)


def _stable_hash(value: Any) -> str:
    return acquisition.p14_acquisition.utilities.stable_hash(value)


def _self_hashed(value: Mapping[str, Any], field: str = "pack_sha256") -> dict[str, Any]:
    return acquisition.p14_acquisition.utilities.self_hashed(value, field=field)


def _read_json(path: Path, name: str, *, canonical: bool = False) -> Mapping[str, Any]:
    try:
        value = acquisition._read_json(path, name)
    except Exception as exc:
        raise P16AllRemoteError(f"{name} is unavailable or invalid") from exc
    if canonical and acquisition.p14_acquisition.utilities.canonical_json_bytes(value) != path.read_bytes():
        raise P16AllRemoteError(f"{name} is not canonical")
    return value


def _verify_hash(value: Mapping[str, Any], field: str, name: str) -> str:
    body = dict(value)
    declared = body.pop(field, None)
    if not isinstance(declared, str) or _stable_hash(body) != declared:
        raise P16AllRemoteError(f"{name} self hash drifted")
    return declared


def _git_head(project_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_is_ancestor(commit: str, project_root: Path) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=project_root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def load_acquisition(base: Path) -> tuple[Mapping[str, Any], tuple[acquisition.RuntimeItem, ...]]:
    path = base / acquisition.RESULT_RELATIVE
    result = _read_json(path, "P16 acquisition result")
    declared = result.get("self_sha256")
    if (
        result.get("schema") != acquisition.SCHEMA
        or result.get("status") != "passed_view_only_ready_for_P16_all_remote_action"
        or not isinstance(declared, str)
    ):
        raise P16AllRemoteError("P16 acquisition result drifted")
    acquisition._verify_self(result, declared, "P16 acquisition result")
    if result.get("study_design_self_sha256") != acquisition.DESIGN_SELF_SHA256:
        raise P16AllRemoteError("P16 acquisition design drifted")
    return result, acquisition.load_views(base, result)


def load_freeze(base: Path, project_root: Path) -> Mapping[str, Any]:
    value = _read_json(base / FREEZE_RELATIVE, "P16 implementation freeze")
    if value.get("schema") != FREEZE_SCHEMA:
        raise P16AllRemoteError("P16 implementation freeze schema drifted")
    _verify_hash(value, "self_sha256", "P16 implementation freeze")
    rows = value.get("implementation_bindings")
    observed = {
        row.get("relative_path"): row.get("sha256")
        for row in rows if isinstance(row, Mapping)
    } if isinstance(rows, list) else {}
    required = {
        IMPLEMENTATION_RELATIVE.as_posix(),
        MINILM_ENCODER_RELATIVE.as_posix(),
        RUNNER_RELATIVE.as_posix(),
        TEST_RELATIVE.as_posix(),
    }
    if set(observed) != required:
        raise P16AllRemoteError("P16 implementation set drifted")
    for relative, expected in observed.items():
        if _file_sha256(base / relative) != expected:
            raise P16AllRemoteError("P16 implementation file drifted")
    commit = value.get("formal_implementation_commit")
    if not isinstance(commit, str) or not _git_is_ancestor(commit, project_root):
        raise P16AllRemoteError("P16 implementation commit drifted")
    if value.get("study_design_self_sha256") != acquisition.DESIGN_SELF_SHA256:
        raise P16AllRemoteError("P16 freeze design binding drifted")
    return value


def load_fingerprint(base: Path) -> Mapping[str, Any]:
    value = _read_json(base / FINGERPRINT_RELATIVE, "P16 remote fingerprint")
    if value.get("schema") != FINGERPRINT_SCHEMA:
        raise P16AllRemoteError("P16 remote fingerprint schema drifted")
    _verify_hash(value, "self_sha256", "P16 remote fingerprint")
    if (
        value.get("remote_host_alias") != REMOTE_HOST_ALIAS
        or value.get("remote_hostname") != REMOTE_HOSTNAME
        or value.get("item_identity_or_label_access_count") != 0
    ):
        raise P16AllRemoteError("P16 remote fingerprint policy drifted")
    return value


def make_plan(project_root: Path) -> Mapping[str, Any]:
    project_root = project_root.resolve(strict=True)
    base = project_root / "reconstruction_v2"
    path = base / PLAN_RELATIVE
    if path.exists() or path.is_symlink():
        raise OneShotRefusal("P16 remote plan already exists")
    if (base / RESULT_RELATIVE).exists():
        raise OneShotRefusal("P16 final result already exists")
    result, _items = load_acquisition(base)
    freeze = load_freeze(base, project_root)
    fingerprint = load_fingerprint(base)
    view = result["pack_bindings"]["C_confirm_view"]
    plan = _self_hashed(
        {
            "acquisition_result": {
                "file_sha256": _file_sha256(base / acquisition.RESULT_RELATIVE),
                "relative_path": acquisition.RESULT_RELATIVE.as_posix(),
                "self_sha256": result["self_sha256"],
            },
            "attempt_count": ATTEMPT_COUNT,
            "candidate_freeze_self_sha256": p14.CANDIDATE_FREEZE_SELF_SHA256,
            "execution_policy": {
                "cross_encoder_visible_GPU": "1",
                "external_network_call_count_allowed": 0,
                "HippoRAG_concurrency": 8,
                "HippoRAG_visible_GPU": "",
                "prior_P14_or_P15_output_reuse_count": 0,
                "MiniLM_and_Qwen_visible_GPU": "0",
                "online_evaluator_call_count": 0,
                "retry_replay_or_resample_count": 0,
            },
            "implementation_freeze_self_sha256": freeze["self_sha256"],
            "recorded_date": "2026-07-22",
            "remote_base": str(REMOTE_BASE),
            "remote_hostname": REMOTE_HOSTNAME,
            "remote_root": str(REMOTE_ROOT),
            "remote_runtime_fingerprint_self_sha256": fingerprint["self_sha256"],
            "remote_work_root": str(REMOTE_WORK_ROOT),
            "schema": PLAN_SCHEMA,
            "source_document_bindings": {
                family: acquisition.source.SOURCE_FILES[
                    f"documents/{acquisition.source.SLUGS[family]}-00000-of-00001.parquet"
                ]
                for family in FAMILIES
            },
            "study_design_self_sha256": acquisition.DESIGN_SELF_SHA256,
            "target_terminal_count_per_family": TARGET_PER_FAMILY,
            "view_binding": dict(view),
        }
    )
    path.parent.mkdir(mode=0o700, parents=True)
    acquisition.p14_acquisition.utilities._write_json(path, plan)
    return plan


def _load_selected_gold_ids(
    base: Path, selected: Sequence[acquisition.RuntimeItem]
) -> Mapping[str, tuple[str, ...]]:
    """This is the only function that reads gold_ids, after action verification."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise P16AllRemoteError("pyarrow is unavailable") from exc
    by_family = {
        family: {
            item.source_query_id: item.item_key
            for item in selected if item.family == family
        }
        for family in FAMILIES
    }
    output: dict[str, tuple[str, ...]] = {}
    for family in FAMILIES:
        slug = acquisition.source.SLUGS[family]
        relative = f"examples/{slug}-00000-of-00001.parquet"
        path = base / acquisition.source.SOURCE_ROOT_RELATIVE / relative
        binding = acquisition.source.SOURCE_FILES[relative]
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != binding["size_bytes"]
            or _file_sha256(path) != binding["sha256"]
        ):
            raise P16AllRemoteError("P16 label source drifted")
        wanted = sorted(by_family[family])
        table = pq.read_table(
            path,
            columns=["id", "gold_ids"],
            filters=[("id", "in", wanted)],
            use_threads=False,
        )
        if table.column_names != ["id", "gold_ids"]:
            raise P16AllRemoteError("P16 label source schema drifted")
        for row in table.to_pylist():
            identifier = row.get("id")
            gold = row.get("gold_ids")
            if identifier not in by_family[family] or not isinstance(gold, list):
                raise P16AllRemoteError("P16 label row drifted")
            values = tuple(gold)
            if (
                not values
                or len(values) != len(set(values))
                or any(not isinstance(value, str) or not value for value in values)
            ):
                raise P16AllRemoteError("P16 gold IDs drifted")
            output[by_family[family][identifier]] = values
    if set(output) != {item.item_key for item in selected}:
        raise P16AllRemoteError("P16 selected label set drifted")
    return output


def _verify_remote_archive(
    base: Path,
    items: Sequence[acquisition.RuntimeItem],
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    tuple[acquisition.RuntimeItem, ...],
]:
    archive = base / REMOTE_ARCHIVE_RELATIVE
    result = _read_json(archive / "remote_action.result.json", "remote action result", canonical=True)
    if result.get("schema") != REMOTE_RESULT_SCHEMA:
        raise P16AllRemoteError("remote action result schema drifted")
    _verify_hash(result, "pack_sha256", "remote action result")
    plan = _read_json(base / PLAN_RELATIVE, "P16 remote plan", canonical=True)
    _verify_hash(plan, "pack_sha256", "P16 remote plan")
    if (
        result.get("plan_pack_sha256") != plan["pack_sha256"]
        or result.get("remote_hostname") != REMOTE_HOSTNAME
        or result.get("external_network_call_count") != 0
        or result.get("prior_P14_or_P15_output_reuse_count") != 0
    ):
        raise P16AllRemoteError("remote action policy drifted")
    selection_binding = result.get("selection_binding")
    if not isinstance(selection_binding, Mapping):
        raise P16AllRemoteError("remote selection binding is absent")
    selection_path = archive / "complete_case.selection.json"
    if _file_sha256(selection_path) != selection_binding.get("file_sha256"):
        raise P16AllRemoteError("remote selection file drifted")
    selection = _read_json(selection_path, "remote selection", canonical=True)
    if selection.get("schema") != SELECTION_SCHEMA:
        raise P16AllRemoteError("remote selection schema drifted")
    _verify_hash(selection, "pack_sha256", "remote selection")
    if selection.get("pack_sha256") != selection_binding.get("pack_sha256"):
        raise P16AllRemoteError("remote selection pack drifted")
    selection_rows = selection.get("items")
    if not isinstance(selection_rows, list) or len(selection_rows) != ATTEMPT_COUNT:
        raise P16AllRemoteError("remote selection rows drifted")
    terminal_ordinals = tuple(
        row["ordinal"] for row in selection_rows
        if isinstance(row, Mapping) and row.get("terminal") is True
    )
    converted = tuple(
        p14.RuntimeItem(
            ordinal=item.ordinal,
            family=item.family,
            attempt_ordinal=item.attempt_ordinal,
            family_hmac_position=item.family_hmac_position,
            item_key=item.item_key,
            query=item.query,
            source_query_id=item.source_query_id,
            excluded_ids=item.excluded_ids,
        )
        for item in items
    )
    capacity, p14_selected, terminal_counts = p14.select_complete_cases(
        converted, terminal_ordinals
    )
    if (
        capacity != selection.get("capacity_passed")
        or dict(terminal_counts) != selection.get("terminal_counts_by_family")
    ):
        raise P16AllRemoteError("remote complete-case decision drifted")
    if not capacity:
        return result, selection, ()
    selected_ordinals = tuple(item.ordinal for item in p14_selected)
    declared_selected = tuple(
        row["ordinal"] for row in selection_rows
        if isinstance(row, Mapping) and row.get("selected") is True
    )
    if declared_selected != selected_ordinals:
        raise P16AllRemoteError("remote selected order drifted")
    return result, selection, tuple(items[index] for index in selected_ordinals)


def finalize(project_root: Path) -> Mapping[str, Any]:
    project_root = project_root.resolve(strict=True)
    base = project_root / "reconstruction_v2"
    result_path = base / RESULT_RELATIVE
    if result_path.exists() or result_path.is_symlink():
        raise OneShotRefusal("P16 final result already exists")
    acquisition_result, items = load_acquisition(base)
    freeze = load_freeze(base, project_root)
    remote_result, selection, selected = _verify_remote_archive(base, items)
    if not selected:
        result = _self_hashed(
            {
                "capacity_passed": False,
                "claim_boundary": {
                    "external_network_call_count": 0,
                    "gold_ID_column_read_count": 0,
                    "performance_score_count": 0,
                },
                "formal_binding": {
                    "acquisition_result_self_sha256": acquisition_result["self_sha256"],
                    "implementation_freeze_self_sha256": freeze["self_sha256"],
                    "remote_action_result_pack_sha256": remote_result["pack_sha256"],
                    "selection_pack_sha256": selection["pack_sha256"],
                },
                "primary_evaluated": False,
                "recorded_date": "2026-07-22",
                "schema": RESULT_SCHEMA,
                "status": "P16_capacity_failed_labels_remain_unopened",
            },
            field="self_sha256",
        )
        acquisition.p14_acquisition.utilities._write_json(result_path, result, mode=0o644)
        return result
    action_binding = remote_result.get("action_binding")
    if not isinstance(action_binding, Mapping):
        raise P16AllRemoteError("remote action binding is absent")
    action_path = base / REMOTE_ARCHIVE_RELATIVE / "three_arm.actions.json"
    if _file_sha256(action_path) != action_binding.get("file_sha256"):
        raise P16AllRemoteError("remote action file drifted")
    actions = _read_json(action_path, "remote actions", canonical=True)
    if actions.get("schema") != ACTION_SCHEMA:
        raise P16AllRemoteError("remote action schema drifted")
    _verify_hash(actions, "pack_sha256", "remote actions")
    if (
        actions.get("pack_sha256") != action_binding.get("pack_sha256")
        or actions.get("item_count") != SELECTED_COUNT
        or actions.get("complete_case_selection_pack_sha256")
        != selection["pack_sha256"]
    ):
        raise P16AllRemoteError("remote action pack drifted")
    rows = actions.get("items")
    if not isinstance(rows, list) or len(rows) != SELECTED_COUNT:
        raise P16AllRemoteError("remote action rows drifted")
    for item, row in zip(selected, rows):
        if (
            not isinstance(row, Mapping)
            or row.get("ordinal") != item.ordinal
            or row.get("item_key") != item.item_key
            or row.get("source_query_id") != item.source_query_id
            or row.get("family") != item.family
        ):
            raise P16AllRemoteError("remote action identity drifted")
        for arm in ("Agent_document_ids", "RAW_document_ids", "HippoRAG_document_ids"):
            ranking = row.get(arm)
            if (
                not isinstance(ranking, list)
                or len(ranking) != 10
                or len(ranking) != len(set(ranking))
                or any(not isinstance(value, str) or not value for value in ranking)
            ):
                raise P16AllRemoteError("remote action ranking drifted")
    # Gold access is deliberately below every action and identity check above.
    labels = _load_selected_gold_ids(base, selected)
    arm_scores = {"Agent": [], "RAW": [], "HippoRAG": []}
    for item, row in zip(selected, rows):
        gold = labels[item.item_key]
        arm_scores["Agent"].append(metric.integer_ndcg_at_10(row["Agent_document_ids"], gold))
        arm_scores["RAW"].append(metric.integer_ndcg_at_10(row["RAW_document_ids"], gold))
        arm_scores["HippoRAG"].append(metric.integer_ndcg_at_10(row["HippoRAG_document_ids"], gold))
    primary_passed, comparisons = p14.primary_decision(items=selected, arm_scores=arm_scores)
    aggregates = {
        arm: {
            "mean_ndcg_at_10": sum(values) / (SELECTED_COUNT * 1_000_000_000),
            "sum_integer_ndcg": sum(values),
        }
        for arm, values in arm_scores.items()
    }
    family_aggregates = {
        family: {
            arm: sum(
                arm_scores[arm][index]
                for index, item in enumerate(selected)
                if item.family == family
            )
            for arm in arm_scores
        }
        for family in FAMILIES
    }
    result = _self_hashed(
        {
            "aggregates": aggregates,
            "attempt_count": ATTEMPT_COUNT,
            "capacity_passed": True,
            "claim_boundary": {
                "external_network_call_count": 0,
                "gold_ID_column_read_after_remote_action_seal": True,
                "online_evaluator_call_count": 0,
                "population_inference": False,
                "selected_label_score_count": SELECTED_COUNT,
            },
            "comparisons": comparisons,
            "family_aggregates": family_aggregates,
            "formal_binding": {
                "acquisition_result_self_sha256": acquisition_result["self_sha256"],
                "action_file_sha256": action_binding["file_sha256"],
                "action_pack_sha256": actions["pack_sha256"],
                "formal_execution_commit": _git_head(project_root),
                "implementation_freeze_self_sha256": freeze["self_sha256"],
                "remote_action_result_pack_sha256": remote_result["pack_sha256"],
                "selection_pack_sha256": selection["pack_sha256"],
            },
            "item_count": SELECTED_COUNT,
            "primary_evaluated": True,
            "primary_passed": primary_passed,
            "primary_rule": "Agent_minus_RAW_and_Agent_minus_HippoRAG_strictly_positive_in_aggregate_and_each_family",
            "recorded_date": "2026-07-22",
            "schema": RESULT_SCHEMA,
            "status": (
                "P16_primary_passed" if primary_passed
                else "P16_primary_failed_same_candidate_stops"
            ),
        },
        field="self_sha256",
    )
    acquisition.p14_acquisition.utilities._write_json(result_path, result, mode=0o644)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--make-plan", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    arguments = parser.parse_args(argv)
    result = make_plan(arguments.project_root) if arguments.make_plan else finalize(arguments.project_root)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
